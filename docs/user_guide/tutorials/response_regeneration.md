# Response Regeneration

This tutorial walks you through regenerating assistant responses in an existing dataset using a target model served by vLLM. The resulting dataset pairs the original user prompts with freshly generated responses (on-policy data), and is the recommended starting point for speculator training: the drafter learns to predict what the target model actually generates, not what the dataset's original authors wrote. For multi-turn conversations, each assistant turn is regenerated sequentially against the model's own prior responses, keeping the entire history on-policy. Training directly on the dataset's original responses (off-policy) is a cheaper fallback, since it skips a full target-model pass over the data, but costs acceptance length at inference time.

## Overview

**Time required:** ~10 mins on 2x H100 GPUs (for 1K samples)

**Prerequisites:**

- Python 3.10+
- CUDA-capable GPU(s)
- `vllm` installed (`uv pip install "vllm>=0.19.1"`)

## Step 1: Run the Pipeline

The simplest way to regenerate responses is using the `run_all.sh` script, which handles starting a vLLM server, running the regeneration, and stopping the server.

```bash
./scripts/response_regeneration/run_all.sh \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --dataset magpie \
  --limit 1000
```

This will:

1. Start a vLLM server with the specified model
2. Extract conversation turns from the dataset and regenerate assistant responses turn-by-turn
3. Save pre-tokenized results to a JSONL file (e.g., `magpie_Llama-3.3-70B-Instruct.jsonl`)
4. Stop the server

### Multi-GPU Configurations

For larger models, use data parallelism and/or tensor parallelism:

```bash
# Llama 3.3 70B on 8 GPUs (4 data-parallel replicas with TP=2)
./scripts/response_regeneration/run_all.sh \
  --model "meta-llama/Llama-3.3-70B-Instruct" \
  --dp-size 4 --tp-size 2 \
  --dataset magpie

# Select specific GPUs
./scripts/response_regeneration/run_all.sh \
  --model "Qwen/Qwen2.5-72B-Instruct" \
  --gpus 0,1,2,4 --tp-size 4 \
  --dataset magpie
```

### Tool-Call Regeneration

For tool-calling datasets (e.g. `hermes-fc`), pass the model's `--tool-call-parser` (and `--reasoning-parser`, for thinking models) so the server returns structured `tool_calls` and separated reasoning instead of plain text. Both are model-specific, so look them up in the model's [vLLM recipe](https://recipes.vllm.ai/):

```bash
# Qwen3
./scripts/response_regeneration/run_all.sh \
  --model "Qwen/Qwen3-8B" \
  --tool-call-parser hermes --reasoning-parser qwen3 \
  --dataset hermes-fc

# Gemma 4
./scripts/response_regeneration/run_all.sh \
  --model "google/gemma-4-E2B-it" \
  --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --dataset hermes-fc

# gpt-oss (reasoning is parsed automatically)
./scripts/response_regeneration/run_all.sh \
  --model "openai/gpt-oss-20b" \
  --tool-call-parser openai \
  --dataset hermes-fc
```

This is **semi-on-policy** tool-call regeneration: the target regenerates the tool-call tokens on-policy, but tools are not executed. The *i*-th cached tool result from the source data is spliced positionally after the target's *i*-th regenerated call.

**Limitation:** parallel tool calls are under development; the turn is currently truncated to the first call.

## Step 2: Verify the Output

The output is a JSONL file with one pre-tokenized row per target generation. `loss_mask` is `0` over the prompt the target conditioned on and `1` over the tokens it generated, so training needs no further masking:

```json
{
  "id": "conv-abc_gen0",
  "primary_id": "conv-abc",
  "input_ids": [151644, 872, ...],
  "loss_mask": [0, 0, ..., 1, 1],
  "conversations": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ],
  "metadata": {
    "idx": 0,
    "finish_reason": "stop",
    "is_tool_call": false,
    "usage": {...},
    "endpoint": "http://127.0.0.1:8000/v1/chat/completions"
  }
}
```

Each assistant turn produces at least one row — and more when the target calls a tool, since every call is its own generation — so expect more lines than input conversations. `conversations` is a review-only twin of `input_ids`; training drops it.

For multi-turn datasets, later turns include the regenerated history as context. For example, the second turn of the same conversation would be:

```json
{
  "id": "conv-abc_gen1",
  "primary_id": "conv-abc",
  "input_ids": [151644, 872, ...],
  "loss_mask": [0, 0, ..., 1, 1],
  "conversations": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What about Germany?"},
    {"role": "assistant", "content": "The capital of Germany is Berlin."}
  ],
  "metadata": {
    "idx": 0,
    "finish_reason": "stop",
    "is_tool_call": false,
    "usage": {...},
    "endpoint": "http://127.0.0.1:8000/v1/chat/completions"
  }
}
```

Check that the output looks correct:

```bash
# Count completed rows
wc -l magpie_Llama-3.3-70B-Instruct.jsonl

# Inspect first row
head -1 magpie_Llama-3.3-70B-Instruct.jsonl | python -m json.tool
```

## Step 3: Use the Data for Training

The output JSONL can be passed directly to `prepare_data.py` for speculator training:

```bash
python scripts/prepare_data.py \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --data ./magpie_Llama-3.3-70B-Instruct.jsonl \
  --output ./output \
  --seq-length 8192
```

## Advanced: Manual Control

If you prefer to manage the vLLM server yourself (e.g., to reuse a server across multiple runs), you can run the regeneration script directly:

```bash
# 1. Start vLLM server
vllm serve "meta-llama/Llama-3.3-70B-Instruct" \
  --data-parallel-size 4 --tensor-parallel-size 2 \
  --port 8000

# 2. Run regeneration (model auto-detected from server)
python scripts/response_regeneration/script.py \
  --dataset magpie \
  --limit 1000

# 3. Stop server when done (Ctrl+C)
```

### Multiple Servers (round-robin)

`--endpoint` accepts more than one server, and requests are round-robined across
all of them — useful when you run several independent vLLM servers (e.g. one per
GPU) instead of one data-parallel server, for higher regeneration throughput:

```bash
# Start several single-GPU servers
CUDA_VISIBLE_DEVICES=0 vllm serve "meta-llama/Llama-3.3-70B-Instruct" --port 8000 &
CUDA_VISIBLE_DEVICES=1 vllm serve "meta-llama/Llama-3.3-70B-Instruct" --port 8001 &

# Fan out across them (unreachable endpoints are probed and dropped at startup)
python scripts/response_regeneration/script.py \
  --dataset magpie --limit 1000 \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
             http://127.0.0.1:8001/v1/chat/completions
```

Pass `--skip-endpoint-validation` to skip the startup health check.

### Regenerating a Local Conversations Dataset

Besides the built-in single-prompt datasets, you can regenerate an existing
**multi-turn conversations JSONL** with your target model — useful for aligning a
training set (e.g. a ShareGPT-style corpus) to the target's own outputs. Each row
has a `conversations` list of `{from, value}` (or `{role, content}`) turns; every
assistant turn is regenerated in context, keeping the system/user turns:

```bash
python scripts/response_regeneration/script.py \
  --input-jsonl /path/to/dataset/train.jsonl \
  --outfile /path/to/train_regen.jsonl \
  --endpoint http://127.0.0.1:8001/v1/chat/completions \
             http://127.0.0.1:8002/v1/chat/completions \
  --concurrency 128 --max-tokens 2048 --resume
```

- `--input-jsonl` reads a plain JSONL, so it does **not** require `datasets`/`pyarrow`.
- Multimodal / tool-call rows are skipped by default (`--skip-sources`); use
  `--sources` to regenerate only specific sources.
- Failed rows (e.g. a conversation longer than the server's `max_model_len`) are
  written to `<outfile>.errors.jsonl` — the main output stays clean, and
  `--resume` retries them.

**One-command version.** To launch the servers *and* run the client in a single
step (N independent single-GPU servers + the round-robin client, resumable), use
`scripts/response_regeneration/run_regen_multigpu.sh`:

```bash
MODEL=/path/to/target INPUT_JSONL=/path/to/train.jsonl OUTFILE=/path/to/train_regen.jsonl \
  NUM_GPUS=8 \
  bash scripts/response_regeneration/run_regen_multigpu.sh
# add CUDA_COMPAT=/path/to/cuda-compat-13.0 on an old (<570) driver; DRY_RUN=1 to preview.
```

It spins up one vLLM server per GPU, waits for all to be healthy, regenerates the
whole file with `--resume`, and stops the servers on exit.

### Resuming Interrupted Processing

If processing is interrupted, use the `--resume` flag to skip already-processed
rows (matched by uuid, id, or `metadata.idx`; error rows are retried):

```bash
python scripts/response_regeneration/script.py \
  --dataset magpie \
  --outfile magpie_Llama-3.3-70B-Instruct.jsonl \
  --resume
```

### Keeping the Server Running

Use `--keep-server` with `run_all.sh` to leave the vLLM server running after processing, useful when running multiple regeneration jobs:

```bash
# First run - start server and keep it
./scripts/response_regeneration/run_all.sh \
  --model "Qwen/Qwen2.5-72B-Instruct" \
  --dataset magpie --keep-server

# Second run - use the already-running server directly
python scripts/response_regeneration/script.py --dataset ultrachat
```

## Next Steps

After regenerating your dataset:

1. **Train a speculator** - See [Train a Speculator](train.md).
2. **Evaluate performance** - See [Evaluating Performance](evaluating_performance.md)
3. **Deploy to production** - See [Serve in vLLM](serve_vllm.md)

For the full list of arguments for both scripts, see the [response_regeneration CLI reference](/cli/response_regeneration.md).
