"""Compatibility entry point: ``torchrun ... scripts/train.py <flags>``.

Upstream removed this script in favour of ``python -m speculators.train``
(vllm-project/speculators#1228); the fork's launch scripts still call it, so it
forwards to the same entry point.
"""

from speculators.train.cli import main
from speculators.train.config import TrainConfig

if __name__ == "__main__":
    main(TrainConfig.resolve())
