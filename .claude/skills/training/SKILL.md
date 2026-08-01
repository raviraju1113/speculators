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
- Hidden states cost 64.5 KB/token (5 aux layers + last, bf16, hidden_size 5376).
  Measured: 256 samples = 6.6 GB | 30k ≈ 774 GB | 100k = 2.6 TB | full 349k ≈ 9 TB.
  /import/ml-sc-scratch1 has ~4 TB free (90% used), so offline is fine up to ~30k
  and impossible at full scale.
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
- vLLM flags: --max-model-len MUST be >= prepare_data --seq-length (8192); sizing
  it from sampled conversation lengths is wrong. At TP=1 that needs
  --gpu-memory-utilization ~0.95 (weights 62.6 GB leave only ~10.9 GiB KV at 0.92,
  and 8192 context needs ~11.9 GiB). Also --no-enable-prefix-caching.
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