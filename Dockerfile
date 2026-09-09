FROM nvcr.io/nvidia/pytorch:24.12-py3

USER root

WORKDIR /workspace/speculators

COPY . .

RUN python -m pip install --upgrade pip setuptools wheel

# Install torch/vision/audio from the CUDA 12.6 wheel index so the resulting
# build runs on GPU nodes with a CUDA 12.6/12.7 driver (default PyPI torch 2.9
# ships a cu128 build that requires a CUDA 12.8+ driver). Doing this before
# `pip install -e .` means pip sees the requirement already satisfied and won't
# replace it with the cu128 wheel.
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu126 \
    "torch>=2.9.0,<=2.12.1" torchvision torchaudio

RUN python -m pip install -e .

# vLLM serves the OpenAI-compatible endpoint that the response-regeneration
# pipeline (scripts/response_regeneration) talks to; it is intentionally not a
# `speculators` runtime dependency (the package only needs the `openai` client),
# so it must be installed explicitly here. Installed AFTER the cu126 torch above
# so pip sees the torch requirement already satisfied and keeps the cu126 build
# instead of pulling vLLM's default cu128 wheel, which needs a 12.8+ driver.
RUN python -m pip install "vllm==0.23.0"

# vLLM's resolver pulls torch 2.11.0 from PyPI, which is now a cu130 build (needs
# a CUDA 13 driver) and overwrites the cu126 torch installed above, also leaving
# torchvision/torchaudio on a mismatched CUDA. Pin the whole trio back to the
# cu126 build of the same versions (--no-deps so numpy/pillow etc. aren't
# re-fetched from the torch index). 2.11.0 satisfies vLLM 0.23.0's torch pin and
# runs on the 12.6/12.7 driver; verified vllm imports + CUDA inits on cu126.
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu126 \
    --force-reinstall --no-deps \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

# Bring SciPy up to a version compatible with numpy 2.4.x to silence the
# NumPy/SciPy version-mismatch warning from the NGC-bundled SciPy.
RUN python -m pip install --upgrade scipy

CMD ["/bin/bash"]
