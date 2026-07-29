"""Analysis for the draft model's per-token mistakes.

Pure Python (pandas/numpy) so it runs anywhere -- no GPU or server needed. The
notebook ``mistake_analysis.ipynb`` calls these; you can also use them directly.

Three lenses (the ones we care about for choosing a training corpus):

  1. token_category_breakdown  -- what KIND of token the draft misses
  2. rare_in_training_crossref -- missed tokens that were rare/absent in training
                                  (the direct signal for "add this corpus")
  3. position_depth_profile    -- where in the sequence / which TTT depth fails

Input: the ``mistakes.jsonl`` produced by ``score_mistakes.py`` with rows:
  {benchmark, id, ttt_step, aligned_pos, seq_len, rel_pos,
   pred_id, target_id, correct, cond_correct}
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Loading + token decoding
# --------------------------------------------------------------------------- #


def load_mistakes(path: str | Path) -> pd.DataFrame:
    """Load the scorer's JSONL into a DataFrame."""
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No records in {path}")
    return df


def attach_token_strings(df: pd.DataFrame, tokenizer) -> pd.DataFrame:
    """Add ``target_str`` / ``pred_str`` columns by decoding the token ids.

    Decodes each *single* token id (not a sequence) so we see the raw piece,
    including leading-space / newline markers.
    """
    uniq = pd.unique(pd.concat([df["target_id"], df["pred_id"]], ignore_index=True))
    decode = {int(t): tokenizer.decode([int(t)]) for t in uniq}
    out = df.copy()
    out["target_str"] = out["target_id"].map(decode)
    out["pred_str"] = out["pred_id"].map(decode)
    return out


# --------------------------------------------------------------------------- #
# Lens 1: token categories
# --------------------------------------------------------------------------- #

_IDENT_RE = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*$")
_NUM_RE = re.compile(r"^\s*[-+]?\d[\d.,]*$")
_CODE_PUNCT = set("(){}[]<>;:=+-*/%&|^~!.,@#\\`\"'")
_LATEX_RE = re.compile(r"\\[A-Za-z]+|[\\{}^_$]|\\boxed|\\frac|\\sqrt")
_WORD_RE = re.compile(r"^\s*[A-Za-z][A-Za-z'\-]*$")


def categorize_token(s: str) -> str:
    """Heuristically bucket a decoded token string.

    Categories: indent_ws, newline, number, latex_math, code_punct, identifier,
    natural_word, subword, other. Tuned for contrasting code/math (livecodebench,
    gpqa) against prose.
    """
    if s == "" or s.isspace():
        if "\n" in s:
            return "newline"
        return "indent_ws"
    stripped = s.strip()
    if _NUM_RE.match(s):
        return "number"
    if _LATEX_RE.search(s):
        return "latex_math"
    if stripped and all(c in _CODE_PUNCT for c in stripped):
        return "code_punct"
    if _WORD_RE.match(s):
        # a whole word (usually starts with a leading space in BPE)
        return "natural_word"
    if _IDENT_RE.match(s):
        return "identifier"
    if re.match(r"^\s*[A-Za-z0-9_]+$", s):
        return "subword"
    return "other"


def add_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``category`` column based on the *target* token (what should appear)."""
    if "target_str" not in df.columns:
        raise ValueError("Call attach_token_strings() first")
    out = df.copy()
    out["category"] = out["target_str"].map(categorize_token)
    return out


def token_category_breakdown(df: pd.DataFrame, ttt_step: int = 0) -> pd.DataFrame:
    """Per (benchmark, category): sample count, accuracy, and share of mistakes.

    Uses step-0 acceptance by default (the base accept rate that dominates
    end-to-end acceptance length). ``mistake_share`` = fraction of *all* this
    benchmark's step-0 mistakes that fall in this category -- i.e. where the
    errors actually concentrate.
    """
    d = df[df["ttt_step"] == ttt_step].copy()
    d["wrong"] = ~d["correct"]
    g = d.groupby(["benchmark", "category"])
    out = g.agg(n=("correct", "size"), accuracy=("correct", "mean"),
                mistakes=("wrong", "sum")).reset_index()
    tot = out.groupby("benchmark")["mistakes"].transform("sum")
    out["mistake_share"] = out["mistakes"] / tot.replace(0, np.nan)
    return out.sort_values(["benchmark", "mistakes"], ascending=[True, False])


# --------------------------------------------------------------------------- #
# Lens 2: rare-in-training cross-reference
# --------------------------------------------------------------------------- #


def load_token_freq(path: str | Path) -> dict[int, int]:
    """Load ``token_freq.pt`` ({token_id: count}) as a plain dict."""
    import torch  # local import so the lib stays torch-optional otherwise

    tf = torch.load(path, weights_only=True)
    return {int(k): int(v) for k, v in tf.items()}


def add_training_freq(df: pd.DataFrame, token_freq: dict[int, int]) -> pd.DataFrame:
    """Add the *target* token's training-set frequency and a coarse bucket."""
    out = df.copy()
    out["train_freq"] = out["target_id"].map(lambda t: token_freq.get(int(t), 0))

    def bucket(f: int) -> str:
        if f == 0:
            return "absent"
        if f < 100:
            return "rare (<100)"
        if f < 10_000:
            return "medium"
        return "common (>=10k)"

    out["freq_bucket"] = out["train_freq"].map(bucket)
    return out


def accuracy_by_freq_bucket(df: pd.DataFrame, ttt_step: int = 0) -> pd.DataFrame:
    """Step-0 accuracy per training-frequency bucket, per benchmark.

    The core corpus signal: if accuracy drops sharply for ``absent``/``rare``
    tokens, the draft is failing on content under-represented in training.
    """
    d = df[df["ttt_step"] == ttt_step]
    order = ["absent", "rare (<100)", "medium", "common (>=10k)"]
    out = (
        d.groupby(["benchmark", "freq_bucket"])
        .agg(n=("correct", "size"), accuracy=("correct", "mean"))
        .reset_index()
    )
    out["freq_bucket"] = pd.Categorical(out["freq_bucket"], order, ordered=True)
    return out.sort_values(["benchmark", "freq_bucket"])


def top_missed_rare_tokens(
    df: pd.DataFrame, ttt_step: int = 0, max_train_freq: int = 100, top_n: int = 40
) -> pd.DataFrame:
    """Tokens that are frequently missed AND rare/absent in training.

    This is the shortlist that most directly answers "what corpus would help":
    high mistake count here means the draft keeps failing on tokens it barely
    saw during training. Ranked by number of mistakes.
    """
    d = df[(df["ttt_step"] == ttt_step) & (~df["correct"])].copy()
    d = d[d["train_freq"] <= max_train_freq]
    g = (
        d.groupby(["target_id", "target_str"])
        .agg(mistakes=("correct", "size"),
             train_freq=("train_freq", "first"),
             benchmarks=("benchmark", lambda s: ",".join(sorted(set(s)))))
        .reset_index()
        .sort_values("mistakes", ascending=False)
        .head(top_n)
    )
    return g


# --------------------------------------------------------------------------- #
# Lens 3: position & TTT-depth profile
# --------------------------------------------------------------------------- #


def depth_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Accuracy at each TTT step, per benchmark (both full and conditional)."""
    return (
        df.groupby(["benchmark", "ttt_step"])
        .agg(n=("correct", "size"),
             full_acc=("correct", "mean"),
             cond_acc=("cond_correct", "mean"))
        .reset_index()
        .sort_values(["benchmark", "ttt_step"])
    )


def position_profile(df: pd.DataFrame, ttt_step: int = 0, bins: int = 10) -> pd.DataFrame:
    """Step-0 accuracy across normalized-position deciles, per benchmark.

    Shows whether the draft degrades deep into long generations (a common failure
    mode for code / long reasoning traces).
    """
    d = df[df["ttt_step"] == ttt_step].copy()
    d["pos_bin"] = pd.cut(d["rel_pos"], bins=np.linspace(0, 1, bins + 1),
                          labels=[f"{i}/{bins}" for i in range(bins)],
                          include_lowest=True)
    return (
        d.groupby(["benchmark", "pos_bin"], observed=True)
        .agg(n=("correct", "size"), accuracy=("correct", "mean"))
        .reset_index()
    )


def summary(df: pd.DataFrame) -> pd.DataFrame:
    """Headline step-0 (base) and mean-over-steps acceptance per benchmark."""
    base = (
        df[df["ttt_step"] == 0]
        .groupby("benchmark")["correct"].mean()
        .rename("step0_accept")
    )
    chain = df.groupby("benchmark")["cond_correct"].mean().rename("mean_cond_accept")
    n = df[df["ttt_step"] == 0].groupby("benchmark")["correct"].size().rename("n_tokens")
    return pd.concat([n, base, chain], axis=1).reset_index()
