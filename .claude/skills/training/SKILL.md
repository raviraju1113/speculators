# Context: DSpark-Gemma4 training setup (from colleague's Eagle3 log)

## Colleague's proven Eagle3-Gemma4-31B config (reference baseline)
- Environment: Docker, NGC image nvcr.io/nvidia/pytorch:25.12-py3 (bundles CUDA/NCCL); torchrun single-node c10d
- Required docker flags: --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 (default 64MB SHMEM caused failures)
- GPU split: 4 for vLLM server (TP=4, --no-enable-prefix-caching) + 4 for training; 8 CPUs / 100GB RAM
- Settings: --total-seq-len 8192 (data only generated up to 8k), --lr 1e-4 (1e-5 was significantly worse; linear decay), --epochs 3 (300k samples) or 5 (50k)
- Findings: 25k×10ep overfits vs 50k×5ep → scale data, not epochs; acceptance plateaued ~50-60% due to small data + 1-layer capacity
- His Eagle3 baseline to beat: accept_len 2.3-2.8, speedup 1.47-1.8x (AIME/GPQA/LCB)

## What transfers to DSpark vs not
- Reuse: vLLM launch pattern, 4+4 GPU split, torchrun invocation, seq-len 8192, lr 1e-4 anchor, 3-5 epochs, wandb logging, his eval harness
- Drop (Eagle3-specific): --ttt-steps, --use-off-policy-tokens, --draft-arch llama
- Add (DSpark): --speculator-type dspark, --block-size, --max-anchors, --num-layers 5, --target-layer-ids (must ALSO be passed to launch_vllm.py — Eagle3 didn't need it), Markov/confidence flags
- LR: his 1e-4 finding is for 1-layer Eagle3; DFlash tutorial default for 5-layer drafter is 3e-4 → sweep {1e-4, 3e-4}

## Online vs offline training: BOTH, by scale
- Hidden states cost 64.5 KB/token, which is ~74 MB per PREPARED sample (prepared
  rows average ~1356 tokens). MEASURED: 256 samples = 21 GB | 30k ≈ 2.2 TB |
  100k ≈ 7.4 TB | full 349k ≈ 26 TB. /import/ml-sc-scratch1 has ~4 TB free (90%
  used), so offline is practical only for the gate and small ablations -- NOT 30k.
- Offline WINS for the overfit + tuning phase: online with --on-generate delete
  regenerates every hidden state every epoch (5 epochs = 5x the dominant cost),
  while offline pays once and makes each hyperparameter re-run nearly free.
- Online is required for the final full-scale run (storage), and is the only
  option when the dataset changes often.
- NOT a reason to prefer online: TV loss / confidence labels do NOT require a live
  server. Target distributions are computed inside the draft forward by applying
  the frozen verifier LM head to verifier_last_hidden_states
  (dflash/core.py:327-333). Offline caches those same tensors -- identical math.
- Colleague used offline (--on-missing skip) for 50k ultrachat, online for kimi 300k
  -- i.e. the same scale-driven split.
- Pattern:
  GPUs 0-3: launch_vllm.py <gemma4> --tensor-parallel-size 4 --no-enable-prefix-caching --target-layer-ids <...>
  GPUs 4-7: torchrun --standalone --nproc_per_node 4 scripts/train.py --vllm-endpoint http://localhost:8000/v1 --on-missing generate --on-generate delete ...

## Overfit run (first milestone)
- ~1-2k samples (or 32), constant lr 1e-4, high epochs, no early stopping
- Pass criteria (using metrics upstream actually logs -- there is no AUC metric):
  loss → ~0, position_1_acc > 95%, confidence_pred_mean moves off 0.5,
  accept_len climbing, confidence_abs_error falling. Then: export → vllm serve
  loads → nonzero acceptance on training prompts.
- Use online mode even here, to exercise the real pipeline

## Cluster gotchas (sc3-c98, learned 2026-07-31)
- Usable nodes: sc3-c98 AND sc3-c81, both 595.71.05 / CUDA 13.2, 4x A100-80 each.
  To let SLURM pick EITHER, use `--exclude sc-c96,sc3-c97,sc-c82` (with
  `--gputype a100m80`, which already rules out the H100/H200 boxes).
  Do NOT use `--nodelist sc3-c98,sc3-c81`: --nodelist means "include ALL of
  these", so it requests 2 nodes and multiplies the GPU ask per node.
  sc-c96, sc3-c97, sc-c82, sc-c120 are all 565.57.01 / CUDA 12.7 -- too old for
  torch 2.11+cu130. The 8x H200 nodes (sc3-c126..129) are reserved/maint.
- NCCL segfaults on bare metal inside ncclNetPluginInit (plugin/net.cc:216) during
  ncclCommInitRank -- hits vLLM TP>=2 and any torchrun run. NCCL_NET_PLUGIN=none
  did not help. Colleague's TP=4 works INSIDE the NGC container, so the container
  is the likely fix; untested by us.
- No-NCCL workaround that does work: vLLM TP=1 + train launched as plain `python`
  (NOT torchrun -- torchrun sets LOCAL_RANK, and maybe_setup_distributed then calls
  init_process_group("nccl") even for world_size=1, distributed.py:143-154).
- vLLM flags: --max-model-len must be STRICTLY GREATER than prepare_data
  --seq-length (vLLM counts prompt + output; equal drops every sample sitting at
  the truncation ceiling). Sizing it from sampled conversation lengths is wrong.
  At TP=1, 8192 of context needs ~11.9 GiB of KV and util 0.92 leaves only
  ~10.9 GiB, so 0.95 is required. (The full-run OOM was NOT this -- it was
  training-side, see max_anchors below.) Also --no-enable-prefix-caching.
- /dev/shm on bare metal is 504 GB, so the docker --shm-size workaround is a
  container-only problem, not a cluster one.
- sngpu forwards the job command to `sbatch --wrap`, which accepts only
  `bash <one-script-path>` -- no arguments, no `env VAR=... bash ...`. Put presets
  in wrapper scripts (see examples/train/run_*.sh).
- `sngpu --image` works only with --interactive; it fails in batch (needs sudo/TTY).
  So a long training job cannot just "use the container" -- for batch you must port
  Ravi's pattern: bare sngpu node + the job script docker-runs the image itself with
  --shm-size 16G --ipc=host (scripts/submit_train.sh on origin/ravir/eagle3_branch).
- nvcr.io/nvidia/pytorch:25.12-py3 is public (docker manifest inspect works, no NGC
  auth). docker exists on the login node but compute nodes pull their own copy.
- Containers are ephemeral: pip installs inside vanish. Only installs into the conda
  env on /import persist -- so try reusing that env inside the container first
  (examples/train/container_setup.sh phase 1), before building anything.
- Container is an OPTIMIZATION, not a requirement: bare-metal vLLM TP=1 + training as
  plain `python` already trains. The container buys TP=2/4 hidden-state throughput.
- Queue reality: one user runs ~36 pending 1-GPU jobs at higher priority with
  TIME_LIMIT=10-16:00:00 declared (actual <1 h). SLURM's StartTime estimate is then
  fiction, and multi-GPU requests starve because freed GPUs get absorbed one at a
  time. Ask for shorter declared --time (enables backfill) and for sc3-c98 to be
  left clear.

## Verified results (2026-08-01)

### Overfit gate -- PASSED in substance (job 58640223, 50 epochs, 256 samples)
- train: position_1_acc 0.79-0.88, position_7_acc 0.685, accept_len 2.96-4.2, accept_rate 0.655
- val:   position_1_acc 0.277, accept_len 1.535, accept_rate 0.215
- The train/val gap IS the result: the draft memorizes its 256 samples, so layer
  ids / mask token / vocab mapping / Markov head / confidence head are all correct.
- position_7_acc (0.685) barely below position_1 (0.788): accuracy is FLAT across
  the block, which is exactly what the Markov head exists to produce.
- 50 epochs did NOT reach the >0.95 bar; the extension to epoch 200 (1 GPU, cached
  states, no vLLM) DID, conclusively: **train position_0..7_acc all 1.0000,
  accept_len 8.43, accept_rate 0.985, ce_loss 0.0001**, val unchanged (~0.2).
  So it was step-starvation, not a defect. Think in OPTIMIZER STEPS, not epochs:
  the gate needed ~5,000 steps total (25/epoch x 200); ONE epoch of the full 349k
  corpus is ~57,700 steps. The gate therefore implies nothing about needing more
  epochs at scale -- decide that from the val curve during the real run.
- Confidence head: unbiased in-distribution (pred 0.643 vs actual 0.655) but badly
  overconfident on val (0.771 vs 0.215). Recheck at scale; if it persists, STS
  calibration moves from optional to necessary.

### Measured throughput / sizing
- Hidden-state generation at TP=1: ~6k tok/s.
- Hidden states: ~74 MB per PREPARED sample (not 26 MB -- prepared rows average
  ~1356 tokens, far more than raw conversations). 256 -> 21 GB, 349k -> ~26 TB.
- Per-epoch on 256 samples: 3.2 min with --on-generate delete, 2.25 min with cache
  => generation ~1 min, training ~2.25 min. TRAINING, not generation, was the
  larger term even at max_anchors=256; the full run uses 3072, which is heavier
  still. Do not assume generation dominates.
- Full corpus: 349,138 samples, ~473M tokens/epoch.

## DeepSpec's own gemma-4 recipe (config/dspark/dspark_gemma4_12b.py)
Authoritative, better than the paper table:
- block_size 7, num_draft_layers 5, num_anchors **512** (we use 3072), mask_token_id 4
- markov_rank 256 vanilla, confidence_head_alpha 1.0, confidence_head_with_markov True
- **loss_decay_gamma 4.0** -- confirms 4.0, NOT block_size, despite the paper writing
  w_k = exp(-(k-1)/gamma). Our default was already right; that ablation is closed.
- ce/l1 alpha 0.1/0.9, max_grad_norm 1.0 (ours hardcoded in trainer.py:470)
- lr **6.0e-4**, warmup_ratio 0.04, weight_decay 0.0, 10 epochs, max_length 4096
- global_batch_size **512** via gradient accumulation from local 1.
  DO NOT copy their lr: this tree has NO gradient accumulation (upstream PR #859
  still open), so our effective batch is one packed sequence per rank per step.
  Keep the {1e-4, 3e-4} sweep.

## Cluster gotchas learned the hard way (all cost real time)
- **Never edit a script while a job is executing it.** bash reads scripts by byte
  offset; an edit shifts the offsets and the job exits silently with status 0
  (observed: job 58650593 stopped right after "vLLM ready", no error). All
  wrappers now `cp` the driver to $SLURM_TMPDIR and exec the copy.
- **GPUs are NOT isolated between jobs here.** Two concurrent jobs both see all 4
  cards and both get CUDA_VISIBLE_DEVICES starting at 0. Neither hardcoding 0,1
  nor trusting CUDA_VISIBLE_DEVICES works. Free-memory detection also races (a job
  reserves its training GPU but does not touch it for ~7 min while vLLM loads, so
  the other job sees it idle). Use explicit GPU_IDS=... per job.
- **--max-model-len must be STRICTLY > prepare_data --seq-length.** Equal still
  drops the longest samples: vLLM counts prompt + output, so an 8192-token sample
  plus 1 output token is 8193 > 8192 -> HTTP 400 -> silently skipped every epoch.
  0.78% of the corpus (2,728 of 349,138) sits exactly at the ceiling.
- **Failed hidden-state fetches are silent** (data.py returns None + a warning).
  Always run: grep -c "Failed to load/cache hidden states" <log>
- **max_anchors=3072 OOMs the TRAINING process** (not vLLM -- the traceback is in
  torch._inductor / _aot_autograd, i.e. the compiled draft backward). The loss
  materializes logits of [1, anchors*block_size, draft_vocab] = [1, 24576, 32000];
  with targets plus softmax/autograd copies that reaches tens of GB. The overfit
  gate survived only because it used max_anchors=256.
  FIX: max_anchors 512 (DeepSpec's own num_anchors) -> 4096 positions, ~2x the
  gate's footprint. More training GPUs do NOT help: FSDP shards parameters and
  optimizer state, not activations.
- Model size pushes us TOWARD DeepSpec's smaller values, not away: gemma-4-31B has
  hidden 5376 over 60 layers, so KV/token (~1.45 MiB) and per-anchor activation are
  both larger than their 12B. Their max_length 4096 / num_anchors 512 are more
  necessary for us, not less.
- **Use sub-epoch checkpointing at scale.** An epoch on the full corpus is ~57,700
  steps and >20 h; --checkpoint-freq 0.25 caps what a timeout throws away.
- Use `--exclude sc-c96,sc3-c97,sc-c82` to let SLURM pick either good node;
  `--nodelist a,b` means "include BOTH" and requests 2 nodes.

## Disk: checkpoints are 13 GB each -- pair high epoch counts with --save-best
- MEASURED: one checkpoint of the 5-layer / 2.4B-param DSpark draft (weights +
  optimizer state) is **13 GB**.
- The 200-epoch overfit extension therefore wrote **2.5 TB** (200 x 13 GB) and took
  /import/ml-sc-scratch1 from 4 TB free to 1.5 TB (97% full) in two days. Recovered
  by keeping only the best + last epoch.
- There is NO automatic pruning unless you pass `--save-best`, which calls
  Checkpointer.cleanup_keep_only_best() on each new best val loss (trainer.py:636)
  and deletes the other epoch dirs, keeping the footprint flat.
- Rule: EPOCHS x (1/CHECKPOINT_FREQ) x 13 GB is what you will consume without it.
  freq 0.5 over 10 epochs = 20 saves = 260 GB; freq 0.25 = 520 GB.
- Deleting epoch dirs leaves dangling `epochN_end` symlinks; clean with
  `find <ckpt-dir> -xtype l -delete`.
- rm -rf over many 13 GB dirs on NFS is silent and slow -- it looks hung when it is
  not, and `df` reclaims lazily.
