# DSpark training/eval image for the standalone B200 boxes, following johnl's
# pattern (/sms-scratch/johnl/vllm-official-Dockerfile): the OFFICIAL
# vllm/vllm-openai base supplies the coherent native stack (torch, CUDA,
# flashinfer, flash-attn) matched to vLLM 0.26.0; python packages we own
# (speculators, hs_connectors, wandb) live in a --system-site-packages venv on
# /sms-scratch, NOT in the image.
#
# Build:  cd /sms-scratch/mengmengj/docker && bash build.sh
ARG VLLM_IMAGE_TAG=v0.26.0-cu129-ubuntu2404
FROM docker.io/vllm/vllm-openai:${VLLM_IMAGE_TAG}

# The official image's ENTRYPOINT launches `vllm serve`; clear it so the
# container runs whatever command we pass.
ENTRYPOINT []

ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /workspace

# git + toolchain for editable installs; python3.12-venv for the venv on scratch.
RUN apt-get update && apt-get install -y \
    git build-essential curl python3.12-venv \
 && rm -rf /var/lib/apt/lists/*

# sample_from_anchor mismatch fix (mengmengj, 2026-08-05): upstream vLLM
# hardcodes dspark_bonus_anchor=True, i.e. assumes checkpoints trained with
# sample_from_anchor=False. We train with True (the speculators default), so
# derive it from the checkpoint config instead. Same one-liner as the cluster's
# patched env (see speculators commit dfa1606 /
# gemma4-31b-dspark-anchorpatch.yaml). Original kept as algos.py.orig.
# The build FAILS if the pattern is not found exactly once (guards version bumps).
RUN python3 - <<'EOF'
import vllm.transformers_utils.configs.speculators.algos as m
import shutil
p = m.__file__
old = '    pre_trained_config["dspark_bonus_anchor"] = True'
new = ('    pre_trained_config["dspark_bonus_anchor"] = not config_dict.get(\n'
       '        "sample_from_anchor", False)')
s = open(p).read()
assert s.count(old) == 1, f"dspark_bonus_anchor pattern not found exactly once in {p}"
shutil.copy(p, p + ".orig")
open(p, "w").write(s.replace(old, new))
print("patched", p)
EOF
