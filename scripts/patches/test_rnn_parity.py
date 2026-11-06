"""Train/serve parity for markov_head_type="rnn".

Binds the PATCHED vLLM DSparkMarkovHead.rnn_step to a stub holding the very
same joint_proj module as a speculators MarkovHead, then checks that looping
rnn_step and applying markov_w2 reproduces MarkovHead.block_bias exactly.
This is the check that a vanilla-vs-rnn mismatch would fail.
"""
import importlib.util, types
import torch
torch.manual_seed(0)

# Load model_definitions.py directly: importing the speculators package pulls
# in `datasets`, which the serving env does not have. The file itself only
# needs torch, and loading it by path guarantees we test the REAL training
# code rather than a copy of it.
_spec = importlib.util.spec_from_file_location(
    "dspark_model_definitions",
    "/nvmedata/chenw/speculators/src/speculators/models/dspark/model_definitions.py",
)
_md = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_md)
MarkovHead = _md.MarkovHead
from vllm.model_executor.models.qwen3_dspark import DSparkMarkovHead

V, DV, R, H, N, B = 50, 50, 8, 16, 3, 5
train = MarkovHead(verifier_vocab_size=V, draft_vocab_size=DV,
                   markov_rank=R, hidden_size=H, head_type="rnn").eval()

prev_ids = torch.randint(0, V, (N, B))
hidden = torch.randn(N, B, H)

with torch.no_grad():
    want = train.block_bias(prev_token_ids=prev_ids, hidden_states=hidden)

# Stub carrying the identical joint_proj; rnn_step only needs these two attrs.
stub = types.SimpleNamespace(
    joint_proj=train.joint_proj,
    markov_w2=types.SimpleNamespace(weight=train.markov_w2.weight),
)
with torch.no_grad():
    prev_emb_all = train.prev_embeddings(prev_ids)
    state = torch.zeros(N, R)
    embs = []
    for k in range(B):
        state, e = DSparkMarkovHead.rnn_step(stub, state, prev_emb_all[:, k], hidden[:, k])
        embs.append(e)
    got = torch.stack(embs, dim=1) @ train.markov_w2.weight.T

err = (want - got).abs().max().item()
print(f"  block_bias  {tuple(want.shape)}   max |train - serve| = {err:.3e}")
assert err < 1e-5, f"RNN PARITY FAILED: {err}"

# A vanilla serve path against an rnn-trained head must NOT match -- proves the
# test has teeth and that serving the wrong head type is a real mismatch.
with torch.no_grad():
    vanilla_got = prev_emb_all @ train.markov_w2.weight.T
vanilla_err = (want - vanilla_got).abs().max().item()
print(f"  vanilla-instead-of-rnn      max |train - serve| = {vanilla_err:.3e}  (must be large)")
assert vanilla_err > 1e-2, "test has no teeth"

# The state must actually carry history: perturbing position 0's token has to
# move position B-1's bias, which is the whole point of the head.
with torch.no_grad():
    alt = prev_ids.clone(); alt[:, 0] = (alt[:, 0] + 1) % V
    want_alt = train.block_bias(prev_token_ids=alt, hidden_states=hidden)
shift = (want[:, -1] - want_alt[:, -1]).abs().max().item()
print(f"  position 0 token -> position {B-1} bias delta = {shift:.3e}  (must be nonzero)")
assert shift > 1e-4, "state carries no history -- the head is effectively first-order"
print("RNN PARITY OK")

# ---------------------------------------------------------------------------
# SLOT ALIGNMENT. The checks above feed both sides the SAME slot range, so they
# pass even if serving starts the recurrence on the wrong slot -- which is
# exactly the bug that made the 6-layer rnn draft serve 4.33 accepted tokens
# against the vanilla draft's 4.44 while teacher-forced validation said it was
# better. With sample_from_anchor=False, training runs the state over all
# block_size slots from zero, but vLLM samples only slots 1..K. Serving must
# therefore PRIME the state on slot 0 before its first sampled position.
BS, K = 16, 5
tok = torch.randint(0, V, (N, BS))
# Training's shifted conditioning: slot k's previous token is tok[:, k-1],
# and slot 0 reuses the anchor.
prev_train = torch.cat([tok[:, :1], tok[:, :-1]], dim=1)
hid_train = torch.randn(N, BS, H)
with torch.no_grad():
    bias_train = train.block_bias(prev_token_ids=prev_train, hidden_states=hid_train)

def serve(prime: bool):
    """Reproduce vLLM's loop over sampled slots 1..K."""
    with torch.no_grad():
        state = torch.zeros(N, R)
        if prime:
            # _prime_markov_state: anchor token + anchor slot's hidden, bias thrown away.
            state, _ = DSparkMarkovHead.rnn_step(
                stub, state, train.prev_embeddings(tok[:, 0]), hid_train[:, 0]
            )
        out = []
        for i in range(K):
            # Served position i is training slot i+1: previous token tok[:, i],
            # backbone hidden at slot i+1.
            state, e = DSparkMarkovHead.rnn_step(
                stub, state, train.prev_embeddings(tok[:, i]), hid_train[:, i + 1]
            )
            out.append(e)
        return torch.stack(out, dim=1) @ train.markov_w2.weight.T

want_served = bias_train[:, 1 : K + 1]
primed_err = (want_served - serve(prime=True)).abs().max().item()
unprimed_err = (want_served - serve(prime=False)).abs().max().item()
print(f"  served slots 1..{K}, primed     max |train - serve| = {primed_err:.3e}")
print(f"  served slots 1..{K}, UNPRIMED   max |train - serve| = {unprimed_err:.3e}  (must be large)")
assert primed_err < 1e-5, f"SLOT ALIGNMENT FAILED even with priming: {primed_err}"
assert unprimed_err > 1e-2, (
    "the unprimed path matches, so this test cannot detect the off-by-one-slot bug"
)
print("SLOT ALIGNMENT OK")

# ===========================================================================
# GATED HEAD. Position-local: the bias is the previous-token embedding scaled
# by a gate computed from that position's hidden state. Nothing is carried
# between positions, so unlike the rnn head it needs no anchor-slot priming --
# and the slot-alignment check below must pass WITHOUT any.
gated = MarkovHead(verifier_vocab_size=V, draft_vocab_size=DV,
                   markov_rank=R, hidden_size=H, head_type="gated").eval()
gstub = types.SimpleNamespace(
    gate_proj=gated.gate_proj,
    markov_w2=types.SimpleNamespace(weight=gated.markov_w2.weight),
)
with torch.no_grad():
    want_g = gated.block_bias(prev_token_ids=prev_train, hidden_states=hid_train)
    out = []
    for i in range(K):
        e = DSparkMarkovHead.gated_step(
            gstub, gated.prev_embeddings(tok[:, i]), hid_train[:, i + 1]
        )
        out.append(e)
    got_g = torch.stack(out, dim=1) @ gated.markov_w2.weight.T
gerr = (want_g[:, 1 : K + 1] - got_g).abs().max().item()
print(f"  gated, served slots 1..{K}            max |train - serve| = {gerr:.3e}")
assert gerr < 1e-5, f"GATED PARITY FAILED: {gerr}"

# Reversing the concat order indexes the other half of gate_proj's weight, so
# it must NOT match -- this is the easiest way to get the gated head wrong.
with torch.no_grad():
    bad = []
    for i in range(K):
        pe = gated.prev_embeddings(tok[:, i]).to(gated.markov_w2.weight.dtype)
        gate = torch.sigmoid(gated.gate_proj(torch.cat([pe, hid_train[:, i + 1]], dim=-1)))
        bad.append(gate * pe)
    bad_g = torch.stack(bad, dim=1) @ gated.markov_w2.weight.T
bad_err = (want_g[:, 1 : K + 1] - bad_g).abs().max().item()
print(f"  gated, concat order REVERSED          max |train - serve| = {bad_err:.3e}  (must be large)")
assert bad_err > 1e-3, "concat order does not matter, so this test has no teeth"

# The gate must actually depend on the hidden state, else it is just vanilla.
with torch.no_grad():
    plain = gated.prev_embeddings(tok[:, :K]) @ gated.markov_w2.weight.T
dep = (want_g[:, 1 : K + 1] - plain).abs().max().item()
print(f"  gated vs plain (same tokens)          delta = {dep:.3e}  (must be nonzero)")
assert dep > 1e-3, "gated collapses to vanilla"
print("GATED PARITY OK")
