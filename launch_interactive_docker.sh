#!/bin/bash
# Run from WITHIN an allocated interactive shell, i.e. after:
#   sngpu --cpu 92 --mem 900000 --gpu 4 --gputype a100m80 --time 5:59:00 --interactive
#
# Usage: bash launch_interactive_docker.sh [gpu-devices] [net]
#   [gpu-devices]  value for --gpus device=...  (default: all)
#   [net]          docker --net value           (default: host)

NV_VISIBLE_DEVICES=${1:-"all"}
DOCKER_BRIDGE=${2:-"host"}

IMAGE=sc-artifacts2.sambanovasystems.com/sw-docker-scratch/speculators:ngc-14.12.t3

# The wrapper names the container <compute-node>-DOCKER, which isn't resolvable
# inside the container. Map it to loopback so torchrun/c10d rendezvous doesn't
# hang on gai error -2. --add-host makes Docker write /etc/hosts as root at
# creation, so no in-container root is needed.
CONTAINER_HOST="$(hostname)-DOCKER"

# Overlay the live host checkout onto the image's baked-in copy so /workspace/speculators
# IS this checkout. Without this, `-w /workspace/speculators` drops you into the stale
# code baked into the image, while `pip install -e .` still imports src/ from that path
# -- a mismatch that runs an old scripts/train.py against patched src/speculators.
#
# But cuda-docker-run-wrapper rejects any -v whose source isn't under $SLURM_TMPDIR
# (job-local scratch) and owned by us -- NFS paths like /import are refused ("Volume
# must be within $SLURM_TMPDIR"). The checkout is 2.3TB (1.9TB dataset + 315G .git +
# 45G checkpoints) but the actual *code* is only a few MB, so stage just the code onto
# scratch and mount THAT. Data/checkpoints stay on /import, reached by absolute path or
# via the symlinks created below for scripts that use relative paths.
REPO_SRC="$(cd "$(dirname "$0")" && pwd)"
: "${SLURM_TMPDIR:=/scratch/jobs/${SLURM_JOB_ID:?run this inside an sngpu --interactive allocation}}"
STAGE="$SLURM_TMPDIR/speculators"

# Rsync only the code/config tree -- fast, and never descends into the huge data dirs.
# Re-run this script to re-sync after editing code on the host (it's a copy, not live).
CODE_ITEMS=(src scripts tests examples docs
            pyproject.toml setup.py Makefile MANIFEST.in mkdocs.yml
            README.md LICENSE CONTRIBUTING.md CODE_OF_CONDUCT.md)
mkdir -p "$STAGE"
for item in "${CODE_ITEMS[@]}"; do
  [ -e "$REPO_SRC/$item" ] && rsync -a --delete "$REPO_SRC/$item" "$STAGE/"
done

# Symlink every top-level entry we did NOT stage back to the /import checkout, so
# scripts referencing data/checkpoints by relative path resolve (needs /import visible
# inside the container; a broken link is harmless if that path is never read).
shopt -s nullglob dotglob
for entry in "$REPO_SRC"/*; do
  name="$(basename "$entry")"
  [ "$name" = "." ] || [ "$name" = ".." ] && continue
  [ -e "$STAGE/$name" ] && continue
  ln -sfn "$entry" "$STAGE/$name"
done
shopt -u dotglob
echo "Staged code -> $STAGE (mounted at /workspace/speculators)"

sudo -g docker /usr/bin/cuda-docker-run-wrapper -it --rm \
  --add-host "$CONTAINER_HOST":127.0.0.1 \
  --net=$DOCKER_BRIDGE \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --shm-size 64G \
  -e NVIDIA_VISIBLE_DEVICES=$NV_VISIBLE_DEVICES \
  -e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  -e WANDB_API_KEY \
  -e WANDB_PROJECT \
  -e WANDB_ENTITY \
  -e WANDB_MODE \
  -v "$STAGE":/workspace/speculators \
  -w /workspace/speculators \
  "$IMAGE" bash
