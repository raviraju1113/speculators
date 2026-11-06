# Reproducing the Qwen3.5 L1 speculative-decoding numbers

Everything here produces the tables in
`docs/experiments/qwen35_l1234_v25_results.md` and on the published model card.

## The two environments are different, and this matters

| | path | vLLM | torch | transformers | flashinfer |
|:--|:--|--:|--:|--:|--:|
| **eval / serving** | `/nvmedata/chenw/envs/speculator-vllm031` | **0.31.0** | 2.13.0+cu129 | 5.17.0 | 0.7.0.post1 |
| **training** (hidden-state server only) | `/root/miniconda3/envs/speculator` | 0.24.0 | 2.11.0+cu129 | | |

**Every served number was produced with vLLM 0.31.0 from the first env.** The
conda env is for training only. Serving a DFlash draft from it puts you on
vLLM 0.24.0 and nothing below will match.

```bash
export PATH="/nvmedata/chenw/envs/speculator-vllm031/bin:/usr/local/cuda-12.9/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.9
export FLASHINFER_DISABLE_VERSION_CHECK=1
export FLASHINFER_WORKSPACE_BASE=/nvmedata/chenw/jitcache
export VLLM_CACHE_ROOT=/nvmedata/chenw/jitcache/vllm
export TRITON_CACHE_DIR=/nvmedata/chenw/jitcache/triton
export TORCHINDUCTOR_CACHE_DIR=/nvmedata/chenw/jitcache/inductor
export LD_LIBRARY_PATH="/nvmedata/chenw/envs/speculator-vllm031/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH"
```

`CUDA_HOME` and the cuda-12.9 PATH entry are load-bearing: `/usr/bin/nvcc` on
this host is CUDA 10.1, and FlashInfer's JIT fails on it with
`nvcc fatal : Unknown option '-generate-dependencies-with-compile'`. A failed
JIT also poisons the cache, so after fixing PATH also run
`rm -rf /nvmedata/chenw/jitcache/.cache/flashinfer/*/80/cached_ops/sampling`.

## One arm, end to end

```bash
M=/nvmedata/hf_checkpoints/qwen35-l1234-v25-step6000
DRAFT=/nvmedata/chenw/speculators/output/e4_4l_dflash_qwen35/checkpoints/checkpoint_best
DATA=/nvmedata/chenw/datasets/l1-v2-gpt56-test/l1_v2_gpt56_test.jsonl

CUDA_VISIBLE_DEVICES=0 setsid nohup vllm serve "$M" \
  --host 127.0.0.1 --port 8910 --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 --max-model-len 32768 --max-num-seqs 64 \
  --no-enable-prefix-caching --trust-remote-code \
  --speculative-config "{\"model\":\"$DRAFT\",\"num_speculative_tokens\":7,\"method\":\"dflash\"}" \
  > server.log 2>&1 &
srv=$!

python scripts/evaluate/qwen35_l1/qwen35_mtp_sweep.py \
  --base-url http://127.0.0.1:8910 --input "$DATA" --model "$M" \
  --label my_run --out my_run.json \
  --limit 1200 --concurrency 32 --force-temperature 0 --no-thinking

kill -9 -$srv          # the process GROUP, see below
```

### The four flags you cannot change and still match

- `--no-thinking`. Without it the default chat template makes this target begin
  answering mid-string. Every number measured without it describes a broken
  serving mode.
- `--force-temperature 0`. Greedy.
- `--limit 1200 --concurrency 32`. The speedup is a throughput ratio under
  load; it is not the same number at concurrency 1.
- `--no-enable-prefix-caching` on the server. These prompts share long
  prefixes, and caching them changes the baseline more than the draft.

## Why your number may differ even when the setup matches

1. **The baseline must come from the same GPU.** Four identical A100 80 GB PCIe
   cards in this host differ by a reproducible ~7% (measured unaccelerated:
   1.2282 / 1.3241 / 1.2262 / 1.3133 req/s). Two solo runs on one card agree to
   0.1%, so it is not noise. The published 1.2411 is
   `4L DFlash on GPU 0 / no-draft on GPU 0`. Divide by another card's baseline
   and you get anything from 1.16 to 1.25.
2. **Run the baseline yourself, in the same session.** Do not reuse one from an
   earlier run.
3. **Check finish reasons.** A single request that runs to the token cap can
   inflate a lane's baseline and flatter every arm measured against it. One
   such row appeared on GPU 2 in two separate jobs
   (`tok=8192, lat=149.3s`, against 1906 tokens for the same prompt elsewhere).
4. **Compare accepted length, not accept_rate.** Eval `accept_rate` is
   `(accepted_length - 1) / k` and changes with k for the same weights.
5. **Byte-identity is not a correctness check here.** At temperature 0, two
   *unaccelerated* runs of the same 1,200 prompts agreed byte-for-byte on only
   35% of completions on this host; draft arms agree with their baseline on
   31–33%, i.e. the same floor. Use validity and finish reason instead.

## Teardown

Killing the server by matching its port kills only the parent; the
`VLLM::EngineCore` child survives holding ~71 of 79 GiB and the next arm dies
with `Free memory on device cuda:0 (8.34/79.25 GiB) ... less than desired`.
This cost six of seven arms in one job and blocked another for 46 minutes.
Launch under `setsid`, `kill -9 -$srv` the group, then poll
`nvidia-smi --query-compute-apps=pid` until the card is actually empty.

## The jobs

`jobs/` holds the scripts that produced each result, with their reasoning in
the header comments.

| job | what |
|:--|:--|
| `415_wideprep.sh` | re-prep at `--seq-length 16384`; recovers 99.8% of response tokens vs 48.3% at 8192 |
| `417_wide_train.sh` | 6L DSpark, wide, 27,920 steps — **collapsed to 0.9532, below baseline** |
| `420_parlane.sh` | first 4-GPU parallel eval with per-card baselines |
| `421_dflash_decide.sh` | the verdict lane: MTP vs both DFlash depths on one card |
| `422_fix_collapse.sh` | 6L DSpark, wide, 12,000 steps (43% of corpus) |
| `423_dflash_wide.sh` | 4L DFlash, wide, 12,000 steps (43%) |
| `424_data_verdict.sh` | 24.1% vs 43% on one card |
| `425/426_ksweep*.sh` | single-card k sweeps (the 421 sweep was cross-card and inconclusive) |
| `427_onpolicy.sh` | the on-policy block-7 DSpark checkpoint |
