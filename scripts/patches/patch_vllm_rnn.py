"""Teach the vendored vLLM to SERVE markov_head_type="rnn".

Job 382 measured, in one run with per-lane baselines, that ~77% of our AL
deficit against MTP is the within-block per-step conditional (ours 0.8101,
MTP 0.8408) and only ~10% is position-1 accuracy (0.8467 vs 0.8582). MTP
conditions draft position j on the sampled tokens 1..j-1 through real forward
passes; DSpark does ONE parallel forward, so its only channel from an earlier
sampled token to a later position is the Markov logit bias -- which at
head_type="vanilla" is first-order, seeing the immediately previous token and
nothing before it.

speculators' MarkovHead already implements head_type="rnn", which carries a
recurrent state across block positions so every position's bias sees the whole
sampled prefix, for 2.3M parameters (live 761.6M -> 763.9M). vLLM implements
vanilla only, so such a checkpoint could not be served. This patch closes that.

Three edits, all mirroring src/speculators/models/dspark/model_definitions.py
MarkovHead.block_bias so train and serve compute the same function:

  1. DSparkMarkovHead gains joint_proj and rnn_step(). The recursion is
     state' = g*state + (1-g)*tanh(cand), bias_emb = tanh(out), with
     [g; cand; out] = joint_proj([state; prev_emb; hidden]) -- identical to
     training, including taking the bias from tanh(out) rather than from
     prev_emb. Any head_type other than vanilla/rnn now raises here instead of
     silently loading as vanilla, which would be a train/serve mismatch of
     exactly the kind that halves acceptance.
  2. Both sequential sampling loops thread the state. The loops already
     iterate positions and already call markov_embed(prev) per position, so
     this adds no forward pass and no sampling change. The confidence head
     keeps receiving the RAW prev-token embedding, because that is what
     core.py:373 feeds it in training for rnn as well as vanilla.
  3. The config validator accepts "rnn".

The state lives in a preallocated buffer zeroed per step rather than a fresh
allocation, so FULL CUDA-graph capture of the draft step still sees static
shapes and a deterministic initial value.
"""
import sys
from pathlib import Path

P = Path("/nvmedata/chenw/envs/speculator-vllm031/lib/python3.12/site-packages/vllm")
edits = 0


def sub(path: Path, old: str, new: str, *, count: int = 1) -> None:
    global edits
    s = path.read_text()
    found = s.count(old)
    if found != count:
        raise SystemExit(f"ABORT {path.name}: expected {count} occurrence(s), found {found}\n---\n{old[:200]}")
    path.write_text(s.replace(old, new))
    edits += 1


# ---------------------------------------------------------------- 1. the head
f = P / "model_executor/models/qwen3_dspark.py"

sub(f, """        prefix: str,
        quant_config: QuantizationConfig | None = None,
        retain_weight_for_gather: bool = False,
    ) -> None:
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)""",
"""        prefix: str,
        quant_config: QuantizationConfig | None = None,
        retain_weight_for_gather: bool = False,
        head_type: str = "vanilla",
        hidden_size: int = 0,
    ) -> None:
        super().__init__()
        # Fail loudly on a head type this file does not implement. Loading it as
        # vanilla would run a different function than the one trained.
        if head_type not in ("vanilla", "rnn"):
            raise ValueError(
                f"DSpark serving implements markov_head_type 'vanilla' and 'rnn'; "
                f"got {head_type!r}. A checkpoint trained with another head type "
                f"would be served with a different bias function than it learned."
            )
        self.head_type = head_type
        self.markov_rank = markov_rank
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)""")

sub(f, """    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        \"\"\"r-dim Markov embedding of ``token_ids`` ([B] -> [B, r]).\"\"\"
        return self.markov_w1(token_ids)""",
"""    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        \"\"\"r-dim Markov embedding of ``token_ids`` ([B] -> [B, r]).\"\"\"
        return self.markov_w1(token_ids)

    def rnn_step(
        self,
        state: torch.Tensor,  # [B, r]
        prev_emb: torch.Tensor,  # [B, r]
        hidden: torch.Tensor,  # [B, hidden]
        ) -> tuple[torch.Tensor, torch.Tensor]:
        \"\"\"Advance the recurrent state one draft position.

        Mirrors speculators MarkovHead.block_bias head_type=="rnn": the bias is
        taken from ``tanh(out)``, NOT from ``prev_emb``, so the caller must feed
        the returned embedding to ``bias``/``apply_bias_gathered``.
        \"\"\"
        dtype = self.markov_w2.weight.dtype
        z = torch.cat(
            [state, prev_emb.to(dtype), hidden.to(dtype)], dim=-1
        )
        gate_raw, cand_raw, out_raw = self.joint_proj(z).chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        state = gate * state + (1.0 - gate) * torch.tanh(cand_raw)
        return state, torch.tanh(out_raw)""")

# joint_proj is declared after markov_w2 so its dtype is available above.
sub(f, """        self.markov_w2._retain_weight_for_gather = retain_weight_for_gather
        self.markov_w2.is_w4a16_nvfp4 = False""",
"""        self.markov_w2._retain_weight_for_gather = retain_weight_for_gather
        self.markov_w2.is_w4a16_nvfp4 = False
        if head_type == "rnn":
            if hidden_size <= 0:
                raise ValueError(
                    "markov_head_type='rnn' needs the draft hidden_size to size "
                    f"joint_proj; got {hidden_size}."
                )
            # Joint [gate; candidate; output] over [state; prev_emb; hidden],
            # replicated like markov_w1/w2 because it runs per draft position.
            self.joint_proj = nn.Linear(
                2 * markov_rank + hidden_size, 3 * markov_rank
            )""")

sub(f, """        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
            quant_config=self.quant_config,
            retain_weight_for_gather=(
                getattr(config, "dspark_draft_topk", None) is not None
            ),
        )""",
"""        self.markov_head = DSparkMarkovHead(
            config.vocab_size,
            draft_vocab_size,
            config.markov_rank,
            prefix=maybe_prefix(prefix, "markov_head"),
            quant_config=self.quant_config,
            retain_weight_for_gather=(
                getattr(config, "dspark_draft_topk", None) is not None
            ),
            head_type=getattr(config, "markov_head_type", "vanilla"),
            hidden_size=config.hidden_size,
        )""")

sub(f, """    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)""",
"""    @property
    def markov_head_type(self) -> str:
        return self.model.markov_head.head_type

    def markov_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.model.markov_head.embed(token_ids)

    def markov_rnn_step(
        self,
        state: torch.Tensor,
        prev_emb: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.markov_head.rnn_step(state, prev_emb, hidden)""")


# ------------------------------------------------------- 2. the sampling loops
g = P / "v1/worker/gpu/spec_decode/dspark/speculator.py"

sub(g, """        self.use_confidence_head: bool = False""",
"""        self.use_confidence_head: bool = False

        # markov_head_type="rnn" carries a recurrent state across the block's
        # positions, so every position's bias sees the whole sampled prefix
        # rather than just the previous token. Preallocated and zeroed per step
        # so FULL CUDA-graph capture sees a static shape and a deterministic
        # initial state.
        self._markov_head_type: str = getattr(
            self.draft_model_config.hf_config, "markov_head_type", "vanilla"
        )
        self._markov_state: torch.Tensor | None = None
        if self._markov_head_type == "rnn":
            markov_rank = getattr(
                self.draft_model_config.hf_config, "markov_rank", 0
            )
            if not markov_rank:
                raise ValueError(
                    "markov_head_type='rnn' requires a positive markov_rank."
                )
            self._markov_state = torch.zeros(
                self.max_num_reqs, markov_rank, dtype=self.dtype, device=device
            )""")

loop_pre = """        idx_map = self.sample_idx_mapping[:num_sample].view(num_reqs, n_spec)
        sample_pos = self.sample_pos[:num_sample].view(num_reqs, n_spec)
        confidence_markov_embeds = []
"""
rnn_pre = """        # rnn head: zero the carried state and expose per-position hidden.
        state = None
        rnn_hidden = None
        if self._markov_state is not None:
            state = self._markov_state[:num_reqs]
            state.zero_()
            rnn_hidden = sample_hidden.view(num_reqs, n_spec, -1)
"""
# dense path
sub(g, loop_pre + """
        # Anchor (bonus) token per request = the input id at query offset 0,
        # read via the precomputed persistent index (fixed buffer for capture).
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            if self.use_confidence_head:
                confidence_markov_embeds.append(markov_embed)
            bias = self.model.markov_bias(markov_embed)""",
loop_pre + rnn_pre + """
        # Anchor (bonus) token per request = the input id at query offset 0,
        # read via the precomputed persistent index (fixed buffer for capture).
        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            # Sequential stage: Markov bias from the previously sampled token.
            markov_embed = self.model.markov_embed(prev)
            if self.use_confidence_head:
                # The confidence head is trained on the RAW prev-token
                # embedding for every head type (core.py:373), so it keeps
                # getting markov_embed even when the bias comes from the state.
                confidence_markov_embeds.append(markov_embed)
            bias_emb = markov_embed
            if state is not None:
                state, bias_emb = self.model.markov_rnn_step(
                    state, markov_embed, rnn_hidden[:, i]
                )
            bias = self.model.markov_bias(bias_emb)""")

# top-k path
sub(g, loop_pre + """        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            markov_embed = self.model.markov_embed(prev)
            if self.use_confidence_head:
                confidence_markov_embeds.append(markov_embed)
            logits_i = self.model.apply_markov_bias_gathered(
                markov_embed,""",
loop_pre + rnn_pre + """        prev = self.input_buffers.input_ids[self._anchor_idx[:num_reqs]]

        for i in range(n_spec):
            markov_embed = self.model.markov_embed(prev)
            if self.use_confidence_head:
                confidence_markov_embeds.append(markov_embed)
            bias_emb = markov_embed
            if state is not None:
                state, bias_emb = self.model.markov_rnn_step(
                    state, markov_embed, rnn_hidden[:, i]
                )
            logits_i = self.model.apply_markov_bias_gathered(
                bias_emb,""")


# ------------------------------------------------------------ 3. the validator
h = P / "config/speculative.py"
sub(h, """    markov_head_type = getattr(draft_hf_config, "markov_head_type", None)
    if markov_head_type != "vanilla":
        raise ValueError(
            "Qwen3-Omni DSpark currently requires markov_head_type='vanilla'; "
            f"got {markov_head_type!r}."
        )""",
"""    markov_head_type = getattr(draft_hf_config, "markov_head_type", None)
    if markov_head_type not in ("vanilla", "rnn"):
        raise ValueError(
            "Qwen3-Omni DSpark supports markov_head_type 'vanilla' and 'rnn'; "
            f"got {markov_head_type!r}."
        )""")

print(f"applied {edits} edits")
