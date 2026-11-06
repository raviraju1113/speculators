"""Shared primitives for train/eval contamination detection.

Pure functions + small classes; no I/O beyond what the caller passes in. The
heavy scan lives in ``detect_contamination.py``, the decision step in
``impact_analysis.py``.

Three things happen here:

1. **Template stripping** (``eval_core``). The eval harness wraps every raw
   problem in scaffolding -- ``prepare_data.py``'s ``AIME_TMPL`` /
   ``GPQA_TMPL`` / ``LCB_TMPL`` and ``convert_eval_datasets_to_jsonl.py``'s
   ``REASONING_SUFFIX``. Training rows carry the raw problem (or a *different*
   wrapper, e.g. Nemotron math's "Solve the following math problem..."). Compare
   the wrapped forms and every method below returns ~0 hits by construction, so
   the scaffolding must come off first. This is the single easiest way to get a
   falsely clean contamination report.

2. **Tokenization + n-grams** (``tokens``, ``ngram_hashes``). Word-level, case
   and punctuation folded, so LaTeX/whitespace reformatting between a training
   copy and the eval copy does not hide a match.

3. **MinHash** (``MinHasher``). Near-duplicate detection for restatements that
   share no long verbatim run.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np

try:  # xxhash is ~5x faster and we hash O(1e9) n-grams.
    import xxhash

    def _h64(text: str) -> int:
        return xxhash.xxh64_intdigest(text)

    def _h32(text: str) -> int:
        return xxhash.xxh32_intdigest(text) & 0x7FFFFFFF

except ImportError:  # pragma: no cover - fallback keeps the tool runnable
    import hashlib

    def _h64(text: str) -> int:
        return int.from_bytes(
            hashlib.blake2b(text.encode(), digest_size=8).digest(), "little"
        )

    def _h32(text: str) -> int:
        return _h64(text) & 0x7FFFFFFF


# --------------------------------------------------------------------------
# Normalization / template stripping
# --------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Scaffolding added by the eval harness, stripped before any comparison. Order
# matters: longest/most specific first.
_EVAL_PREFIXES = (
    # mtp_server_eval/prepare_data.py
    "You are an expert Python programmer. Solve the following competitive "
    "programming problem. Provide a complete, correct solution.",
    "Return your final response within \\boxed{} and only include the letter "
    "choice (A, B, C, or D) as your final response.",
    # eval_datasets/convert_eval_datasets_to_jsonl.py
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program that "
    "matches the specification and passes all tests. You will NOT return "
    "anything except for the program",
    "Write a solution to the following problem and make sure that it passes "
    "the tests:",
    "Problem Statement:",
    "### Problem",
    "Problem:",
)

_EVAL_SUFFIXES = (
    "Please reason step by step, and put your final answer within \\boxed{}.",
    "Mark your solution with \\boxed Answer:",
    "### Answer: (use the provided format with backticks)",
    "Please fix the issue described above.",
    "### Solution",
    "Answer:",
)

# Cut points: everything from here on is harness-generated, not problem text.
# GPQA's option block is shuffled with a local seed (prepare_data.prep_gpqa),
# so it would never match a training copy and only dilutes the signal.
_EVAL_CUTS = (
    "\nOptions:",
    "\n\n### Starter code",
    "\n### Starter code",
    "\n\n### Format:",
    "\n### Format:",
    "\n\n### Solution",
    "\n### Solution",
)

# Instruction wrappers seen on the *training* side (Nemotron split lead-ins).
# Removing them raises MinHash recall; n-gram containment is unaffected because
# these strings never survive on the eval side anyway.
# Trailing "A. x / B. y / C. z / D. w" block on the ungated GPQA mirror. The
# letter->answer assignment is shuffled per run (prepare_data.prep_gpqa seeds a
# local RNG), so it can never match a training copy; leaving it in only shrinks
# the containment denominator. 195/198 gpqa_diamond items carry one.
_MCQ_TAIL_RE = re.compile(
    r"\n\s*A[.)]\s.*?\n\s*B[.)]\s.*?\n\s*C[.)]\s.*?\n\s*D[.)]\s.*\Z",
    re.DOTALL,
)

_TRAIN_PREFIXES = (
    "Solve the following math problem. Make sure to put the answer (and only "
    "answer) inside \\boxed{}.",
    "detailed thinking on",
    "detailed thinking off",
)


def _strip_affixes(text: str, prefixes: tuple[str, ...], suffixes: tuple[str, ...]) -> str:
    """Repeatedly peel known prefixes/suffixes until nothing more matches."""
    changed = True
    while changed:
        changed = False
        stripped = text.strip()
        for prefix in prefixes:
            if stripped.startswith(prefix):
                text, changed = stripped[len(prefix) :], True
                break
        if changed:
            continue
        stripped = text.strip()
        for suffix in suffixes:
            if stripped.endswith(suffix):
                text, changed = stripped[: -len(suffix)], True
                break
    return text.strip()


def eval_core(prompt: str) -> str:
    """Reduce a harness-formatted eval prompt to its raw problem statement."""
    text = unicodedata.normalize("NFKC", prompt)
    for cut in _EVAL_CUTS:
        idx = text.find(cut)
        if idx > 0:
            text = text[:idx]
    text = _MCQ_TAIL_RE.sub("", text)
    return _strip_affixes(text, _EVAL_PREFIXES, _EVAL_SUFFIXES)


def train_core(text: str) -> str:
    """Reduce a training turn to comparable problem text."""
    text = unicodedata.normalize("NFKC", text)
    # Training rows may themselves carry eval-style scaffolding (a training row
    # *is* allowed to look like a benchmark prompt -- that is the thing we are
    # hunting), so peel both affix sets.
    return _strip_affixes(text, _TRAIN_PREFIXES + _EVAL_PREFIXES, _EVAL_SUFFIXES)


def tokens(text: str) -> list[str]:
    """Case-folded alphanumeric word tokens; punctuation and LaTeX marks drop out."""
    return _TOKEN_RE.findall(text.lower())


def exact_key(text: str) -> int:
    """Hash of the fully normalized token stream -- the zero-false-positive test."""
    return _h64(" ".join(tokens(text)))


def ngram_hashes(toks: list[str], n: int) -> list[int]:
    """64-bit hash per n-gram position. Empty when the text is shorter than n."""
    if len(toks) < n:
        return []
    return [_h64(" ".join(toks[i : i + n])) for i in range(len(toks) - n + 1)]


def longest_run(positions: set[int]) -> int:
    """Longest streak of consecutive integers in ``positions`` (0 if empty)."""
    if not positions:
        return 0
    best = run = 1
    ordered = sorted(positions)
    for prev, cur in zip(ordered, ordered[1:]):
        run = run + 1 if cur == prev + 1 else 1
        best = max(best, run)
    return best


# --------------------------------------------------------------------------
# MinHash
# --------------------------------------------------------------------------

# 2**31-1 (Mersenne prime). With shingle hashes and coefficients both < 2**31,
# a*h+b stays under 2**62 and int64 arithmetic never overflows.
_MERSENNE = (1 << 31) - 1


@dataclass
class MinHasher:
    """Banded MinHash over word shingles, for near-duplicate retrieval.

    ``num_perm``/``bands`` default to 64/16 (4 rows per band), which makes pairs
    at Jaccard >= ~0.6 near-certain to become candidates while keeping the
    per-document cost to one small numpy op.
    """

    num_perm: int = 64
    bands: int = 16
    shingle: int = 5
    seed: int = 0xC0FFEE
    _a: np.ndarray = field(init=False, repr=False)
    _b: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.num_perm % self.bands:
            raise ValueError("num_perm must be divisible by bands")
        rng = np.random.default_rng(self.seed)
        self._a = rng.integers(1, _MERSENNE, size=self.num_perm, dtype=np.int64)
        self._b = rng.integers(0, _MERSENNE, size=self.num_perm, dtype=np.int64)

    @property
    def rows(self) -> int:
        return self.num_perm // self.bands

    def shingles(self, toks: list[str]) -> np.ndarray:
        """Distinct shingle hashes as int64. Short texts fall back to unigrams."""
        n = self.shingle
        if len(toks) < n:
            raw = [_h32(t) for t in toks]
        else:
            raw = [_h32(" ".join(toks[i : i + n])) for i in range(len(toks) - n + 1)]
        if not raw:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.asarray(raw, dtype=np.int64))

    def signature(self, shingles: np.ndarray) -> np.ndarray | None:
        """MinHash signature (``num_perm`` int64s), or None for empty input."""
        if shingles.size == 0:
            return None
        permuted = (self._a[None, :] * shingles[:, None] + self._b[None, :]) % _MERSENNE
        return permuted.min(axis=0)

    def band_keys(self, signature: np.ndarray) -> list[int]:
        """One LSH bucket key per band."""
        rows = self.rows
        return [
            _h64(f"{band}:" + ",".join(map(str, signature[band * rows : (band + 1) * rows])))
            for band in range(self.bands)
        ]

    @staticmethod
    def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """Fraction of agreeing hash slots == unbiased Jaccard estimate."""
        return float((sig_a == sig_b).mean())
