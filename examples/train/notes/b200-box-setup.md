---
name: b200-box-setup
description: "Standalone 8x B200 machine profile and how DSpark training runs on it (podman, no SLURM)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 976456e9-265b-4d6a-afd3-854bb7665b28
  modified: 2026-08-06T18:42:10.143Z
---

The `/sms-scratch/mengmengj` machine (hostname trn-b200x8-098, surveyed 2026-08-05) is a STANDALONE 8x NVIDIA B200 (183 GB each) box, driver 595.71.05 (CUDA 13) — not the SambaNova SLURM cluster. Key facts:

- **No sngpu/SLURM** (`sinfo` fails); `docker` is podman emulation, rootless, GPU access via CDI (`/var/run/cdi/nvidia.yaml`) — `--device nvidia.com/gpu=all`, no sudo needed.
- **Image**: `localhost/mengmengj-vllm-official` — built on `vllm/vllm-openai:v0.26.0-cu129-ubuntu2404` (colleague johnl's pattern, see `/sms-scratch/johnl/`), with the one-line `algos.py` patch baked in (`dspark_bonus_anchor = not sample_from_anchor` — serves both kinds of checkpoint correctly).
- **Launch path**: `examples/train/run_local_podman.sh` (SETUP=1 builds venv, SMOKE=1 sanity check, default launches training); preset `run_accum23x4_lr6e4_local.sh` = 4 vLLM DP engines + 4 train ranks, ACCUM 23, lr 6e-4.
- **Thin venv** at `/sms-scratch/mengmengj/envs/dspark`, created `--system-site-packages` INSIDE the container: torch/vllm/transformers come from the image; only speculators+hs_connectors (editable), wandb, datasets, typer live in the venv. (Old full venv at `/home/mengmengj/envs/dspark` is NGC-era, unused.)
- **Paths**: model `/sms-scratch/checkpoints/gemma-4-31B-it`, data `/sms-scratch/mengmengj/data/gemma4_dspark`, outputs `/home/mengmengj/dspark_output` (local disk, ~700G), hidden states `/dev/shm` (1.5T RAM-backed, host RAM 3 TiB), compile caches persisted via `-v /home/mengmengj/container_cache:/root/.cache`.
- **Disk warning**: `/sms-scratch` is NFS at 97% (~370G free) — do NOT write checkpoints there. Checkpoints are 13 GB each; prune per NEXT_STEPS.md step 4.
- Egress works (nvcr.io, wandb). W&B: `~/.netrc` present since 2026-08-06 (login done host-side; venv works on host too since host python is also 3.12) → launcher auto-selects online mode.

Five launch crashes, all fixed in the scripts (details in script comments):
1. rootless podman default **2048-pid limit** kills DP=4 gloo init → `--pids-limit=-1` (and `nofile` must NOT exceed host hard limit 524288 or crun EPERMs).
2. slim image has no bare `python` → driver port-probe uses `python3`.
3. vLLM `VLLM_ENGINE_READY_TIMEOUT_S` defaults 600s; cold B200 engine init took 621s → set 3600.
4. **SELinux Enforcing denies container readdir on NFS /sms-scratch** (files stat fine, dirs won't list → phantom ModuleNotFoundErrors) → `--security-opt label=disable`.
5. bare `torchrun` in a thin venv is the SYSTEM python's console script → workers escape the venv → launch via `python -m torch.distributed.run`.

Related: [[dspark-run-recipe]]

## Fleet update (2026-08): SLURM live, 5 nodes
- lax01-b200: trn-b200x8-[097-101], submit from BASTION only (sn-bastion / la-dc);
  compute nodes cannot reach the controller. --gres=gpu:b200:N.
- SLURM only sees SLURM jobs: jupyter/ray squatters invisible to sinfo — check
  `squeue -w <node>` AND `nvidia-smi` before claiming. Wrappers in examples/train
  self-heal (runtime dir, image load, squatter wait).
- `loginctl enable-linger` once per node or logout kills rootless containers
  (killed two runs). Done on 097/098/101.
- Image distributed via /sms-scratch/mengmengj/docker/mengmengj-vllm-official.tar
  (podman load -i). ~/.netrc is node-local; stage via install -m 600 to /sms-scratch.
- Podman on NFS needs --security-opt label=disable (virt_use_nfs is off);
  --pids-limit=-1; nofile capped at the job's hard limit (SLURM jobs < ssh shells).
- 101 had 2 GPUs off the bus (NVML+Triton break vLLM there) — pending reboot; check.
- Host-side eval venv (no podman): /sms-scratch/mengmengj/envs/dspark-host (pip vllm
  0.26.0 + algos.py patch applied; works on any 595-driver node).
