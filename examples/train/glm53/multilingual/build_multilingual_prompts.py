#!/usr/bin/env python
"""Build the multilingual prompt set for GLM-5.3 on-policy regeneration (prompts only).

Sources (responses are NOT used; every row gets a fresh GLM-5.3 reply at regen time):
  * nvidia/Nemotron-Post-Training-Dataset-v2 multilingual_{es,fr,de,ja,it} (CC BY 4.0, gated:
    needs an HF token that accepted the terms) -- user turns only; its replies are other models'
    and its reasoning traces are English-only. The raw splits are ~60% math / 15% code / 20%
    chat, so they are sampled stratified (NEMOTRON_MIX) toward chat.
  * allenai/WildChat-4.8M (ODC-BY) -- first user turn of real chats, toxic/redacted dropped,
    language from WildChat's own `language` field, for every language (real-chat complement
    to the translated Nemotron prompts).
  * Top-ups when WildChat is short: airesearch/WangchanThaiInstruct (CC BY-SA 4.0, human-written
    Thai), nlpai-lab/kullm-v2 (Apache-2.0, Korean), then CohereLabs/aya_dataset (Apache-2.0).
  * Helsinki-NLP/opus-100 train splits -- translation instructions in both directions over
    many pairs (incl. low-resource targets), several phrasings; plus German->English.

Every prompt that appears in (or contains / is contained in) any eval benchmark prompt under
mtp_server_eval/data is dropped, and prompts are de-duplicated on normalized text.

Output = Mengmeng's regen input format, one JSON per line:
  {"id": "mling:<lang>:<source>:<n>", "source": "<source>", "conversations": [{"from": "human", "value": ...}]}
(a non-empty, non-reasoning-toggle system prompt is kept as a leading {"from": "system"} turn).

    python examples/train/glm53/multilingual/build_multilingual_prompts.py \
        --raw  /import/ml-sc-scratch1/ravir/spec_decoding_work/multilingual_data/raw \
        --out  /import/ml-sc-scratch1/ravir/spec_decoding_work/multilingual_data/prompts_v2
"""

import argparse
import ast
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

EVAL_DATA = Path("/import/ml-sc-scratch1/mengmengj/speculators/scripts/evaluate/mtp_server_eval/data")

# prompts per language / bucket
NEMOTRON = {"es": 10000, "fr": 10000, "de": 10000, "ja": 10000, "it": 5000}
NEMOTRON_SHARDS_PER_LANG = 8                      # evenly spaced over each language's shards
NEMOTRON_MIX = {"chat": 0.40, "math": 0.40, "code": 0.15, "mc": 0.05}
# WildChat language name and prompts wanted (es/fr/de/ja/it are on top of Nemotron).
WILDCHAT = {"es": ("Spanish", 5000), "fr": ("French", 5000), "de": ("German", 5000),
            "ja": ("Japanese", 5000), "it": ("Italian", 2500),
            "zh": ("Chinese", 25000), "ru": ("Russian", 7500), "th": ("Thai", 7500),
            "ar": ("Arabic", 7500), "ko": ("Korean", 7500), "pt": ("Portuguese", 7500),
            "vi": ("Vietnamese", 7500)}
WILDCHAT_REPO = "allenai/WildChat-4.8M"
AYA_LANG = {"th": "Thai", "ar": "Standard Arabic", "ko": "Korean", "pt": "Portuguese",
            "vi": "Vietnamese", "ru": "Russian", "zh": "Simplified Chinese"}
MATH_RE = re.compile(r"\\boxed|\\\(|\\\[|\$[^$]+\$")
MC_RE = re.compile(r"(^|\n)\s*[A-J][:.)]\s", re.M)
CODE_RE = re.compile(r"```|def |#include|public static|print\(|input\(|stdin|stdout|\bpython\b|\bC\+\+|\bJava\b", re.I)


def task_type(text: str) -> str:
    if CODE_RE.search(text):
        return "code"
    if MC_RE.search(text):
        return "mc"
    if MATH_RE.search(text):
        return "math"
    return "chat"
TRANSLATION_TOTAL = 15000
DE_EN = 2000                                      # matches the `translation` benchmark (German->English)
# opus-100 pairs (xx-en or en-xx dirs); long-tail targets mirror speed-multilingual.
OPUS_LANGS = {
    "de": "German", "fr": "French", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
    "nl": "Dutch", "ru": "Russian", "uk": "Ukrainian", "pl": "Polish", "cs": "Czech",
    "ro": "Romanian", "hu": "Hungarian", "el": "Greek", "sv": "Swedish", "tr": "Turkish",
    "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "he": "Hebrew",
    "fa": "Persian", "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "gu": "Gujarati",
    "th": "Thai", "vi": "Vietnamese", "id": "Indonesian", "ms": "Malay", "sw": "Swahili",
    "eu": "Basque", "nn": "Norwegian Nynorsk", "gd": "Scottish Gaelic", "mg": "Malagasy",
    "cy": "Welsh", "ga": "Irish", "is": "Icelandic", "af": "Afrikaans", "ka": "Georgian",
}
TEMPLATES = [
    "Translate the following text from {src} to {tgt}:\n\n{text}",
    "Translate this {src} text into {tgt}.\n\n{text}",
    "Please translate into {tgt}:\n{text}",
    "{text}\n\nTranslate the above from {src} to {tgt}.",
    "Can you translate this into {tgt} for me?\n\n\"{text}\"",
]
REASONING_TOGGLES = re.compile(r"^\s*(/no_think|/think|detailed thinking (on|off)|reasoning:\s*(on|off|low|medium|high))\s*$", re.I)
MIN_CHARS, MAX_CHARS = 20, 6000


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return re.sub(r"\s+", " ", s).strip()


def key(s: str) -> str:
    return hashlib.md5(norm(s).encode()).hexdigest()


def load_eval_texts() -> list[str]:
    texts = []
    for f in sorted(EVAL_DATA.glob("*.jsonl")):
        for line in f.open(encoding="utf-8"):
            if line.strip():
                p = json.loads(line).get("prompt")
                if isinstance(p, str) and len(p) >= 40:
                    texts.append(norm(p))
    return texts


class Dedup:
    def __init__(self, eval_texts):
        self.seen = set()
        self.eval_keys = {hashlib.md5(t.encode()).hexdigest() for t in eval_texts}
        self.eval_texts = eval_texts
        self.dropped = Counter()

    def ok(self, text: str, check_eval_substring: bool = False) -> bool:
        k = key(text)
        if k in self.seen:
            self.dropped["dup"] += 1
            return False
        if k in self.eval_keys:
            self.dropped["eval_exact"] += 1
            return False
        if check_eval_substring:
            n = norm(text)
            if len(n) >= 40 and any(n in t for t in self.eval_texts):
                self.dropped["eval_substring"] += 1
                return False
        self.seen.add(k)
        return True


def parse_list(v):
    if isinstance(v, list):
        return v
    try:
        return json.loads(v)
    except Exception:  # noqa: BLE001
        return ast.literal_eval(v)


def fetch(repo: str, fname: str, raw: Path) -> Path:
    return Path(hf_hub_download(repo, fname, repo_type="dataset", local_dir=raw / repo.replace("/", "__")))


def row(lang, source, n, text, system=None):
    conv = ([{"from": "system", "value": system}] if system else []) + [{"from": "human", "value": text}]
    return {"id": f"mling:{lang}:{source}:{n}", "source": source, "conversations": conv}


def nemotron(raw, dd, rng, api):
    files = sorted(s.rfilename for s in api.dataset_info("nvidia/Nemotron-Post-Training-Dataset-v2").siblings)
    out = {}
    for lang, want in NEMOTRON.items():
        shards = [f for f in files if f.startswith(f"data/multilingual_{lang}-") and f.endswith(".parquet")]
        idx = sorted({round(i * (len(shards) - 1) / (NEMOTRON_SHARDS_PER_LANG - 1)) for i in range(NEMOTRON_SHARDS_PER_LANG)})
        cands = []
        for i in idx:
            t = pq.read_table(fetch("nvidia/Nemotron-Post-Training-Dataset-v2", shards[i], raw), columns=["messages", "reasoning"])
            for msgs in t.column("messages").to_pylist():
                m = parse_list(msgs)
                users = [x for x in m if x.get("role") == "user"]
                if len(users) != 1:
                    continue                                   # single-turn only
                sys_ = next((x.get("content", "") for x in m if x.get("role") == "system"), "") or ""
                sys_ = "" if REASONING_TOGGLES.match(sys_) else sys_.strip()
                text = (users[0].get("content") or "").strip()
                if MIN_CHARS <= len(text) <= MAX_CHARS:
                    cands.append((text, sys_ or None))
        rng.shuffle(cands)
        by_type = {}
        for text, sys_ in cands:
            by_type.setdefault(task_type(text), []).append((text, sys_))
        rows, taken = [], Counter()

        def take(pool, n):
            for text, sys_ in pool:
                if n <= 0:
                    break
                if dd.ok(text, check_eval_substring=False):
                    rows.append(row(lang, "nemotron_v2", len(rows), text, sys_))
                    n -= 1
            return n

        leftovers = []
        for typ, frac in NEMOTRON_MIX.items():
            pool = by_type.get(typ, [])
            quota = round(want * frac)
            short = take(pool[:quota * 2], quota)
            taken[typ] = quota - short
            leftovers += pool[quota * 2:]
        take(leftovers, want - len(rows))           # fill any short category from the rest
        out[lang] = rows
        mix = Counter(task_type(r["conversations"][-1]["value"]) for r in rows)
        print(f"[nemotron] {lang}: {len(rows)}/{want} from {len(cands)} candidates ({len(idx)} shards), mix {dict(mix)}")
    return out


def wildchat(raw, dd, rng, api):
    files = sorted(s.rfilename for s in api.dataset_info(WILDCHAT_REPO).siblings if s.rfilename.endswith(".parquet"))
    by_lang = {lang: [] for lang in WILDCHAT}
    name2lang = {name: lang for lang, (name, _) in WILDCHAT.items()}
    for f in files:
        t = pq.read_table(fetch(WILDCHAT_REPO, f, raw), columns=["conversation", "language", "toxic", "redacted"])
        for conv, language, toxic, redacted in zip(*(t.column(c).to_pylist() for c in ("conversation", "language", "toxic", "redacted"))):
            lang = name2lang.get(language)
            if lang is None or toxic or redacted:
                continue
            c = parse_list(conv) if not isinstance(conv, list) else conv
            first = next((x for x in c if x.get("role") == "user"), None)
            text = ((first or {}).get("content") or "").strip()
            if MIN_CHARS <= len(text) <= MAX_CHARS:
                by_lang[lang].append(text)
    out = {}
    for lang, (_, want) in WILDCHAT.items():
        cands = by_lang[lang]
        rng.shuffle(cands)
        rows = []
        for text in cands:
            if len(rows) >= want:
                break
            if dd.ok(text, check_eval_substring=True):
                rows.append(row(lang, "wildchat4.8m", len(rows), text))
        out[lang] = rows
        print(f"[wildchat] {lang}: {len(rows)}/{want} from {len(cands)} candidates")
    return out


def instruct_topup(raw, dd, rng, out, lang, repo, fname, source):
    """Top up `lang` from an instruction dataset with instruction(+input) columns."""
    want = WILDCHAT[lang][1]
    have = out.get(lang, [])
    need = want - len(have)
    if need <= 0:
        return
    p = fetch(repo, fname, raw)
    if p.suffix == ".parquet":
        recs = pq.read_table(p).to_pylist()
    else:
        recs = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    cands = []
    for r in recs:
        ins = (r.get("Instruction") or r.get("instruction") or "").strip()
        inp = (r.get("Input") or r.get("input") or "").strip()
        text = f"{ins}\n\n{inp}" if inp else ins
        if MIN_CHARS <= len(text) <= MAX_CHARS:
            cands.append(text)
    rng.shuffle(cands)
    added = 0
    for text in cands:
        if added >= need:
            break
        if dd.ok(text, check_eval_substring=True):
            have.append(row(lang, source, added, text))
            added += 1
    out[lang] = have
    print(f"[{source}] {lang}: +{added} (needed {need}, {len(cands)} candidates) -> {len(have)}/{want}")


def aya_topup(raw, dd, rng, out):
    t = pq.read_table(fetch("CohereLabs/aya_dataset", "data/train-00000-of-00001.parquet", raw), columns=["inputs", "language"])
    by = {}
    for text, language in zip(t.column("inputs").to_pylist(), t.column("language").to_pylist()):
        by.setdefault(language, []).append((text or "").strip())
    for lang, (_, want) in WILDCHAT.items():
        have = out.get(lang, [])
        need = want - len(have)
        aname = AYA_LANG.get(lang)
        if need <= 0 or aname is None:
            continue
        cands = [x for x in by.get(aname, []) if MIN_CHARS <= len(x) <= MAX_CHARS]
        rng.shuffle(cands)
        added = 0
        for text in cands:
            if added >= need:
                break
            if dd.ok(text, check_eval_substring=True):
                have.append(row(lang, "aya", added, text))
                added += 1
        out[lang] = have
        print(f"[aya] {lang}: +{added} (needed {need}, {len(cands)} candidates) -> {len(have)}/{want}")


def translation(raw, dd, rng, api):
    files = set(s.rfilename for s in api.dataset_info("Helsinki-NLP/opus-100").siblings)
    pools = {}
    for code in OPUS_LANGS:
        pair = f"{code}-en" if f"{code}-en/train-00000-of-00001.parquet" in files else f"en-{code}"
        f = f"{pair}/train-00000-of-00001.parquet"
        if f not in files:
            print(f"[opus] missing pair for {code}")
            continue
        t = pq.read_table(fetch("Helsinki-NLP/opus-100", f, raw), columns=["translation"])
        pairs = []
        for tr in t.column("translation").to_pylist():
            d = tr if isinstance(tr, dict) else ast.literal_eval(tr)
            en, xx = (d.get("en") or "").strip(), (d.get(code) or "").strip()
            if 80 <= len(en) <= 1200 and len(xx) >= 20:
                pairs.append((en, xx))
        rng.shuffle(pairs)
        pools[code] = pairs[:20000]
    rows = []

    def add(src_code, tgt_code, text):
        src = "English" if src_code == "en" else OPUS_LANGS[src_code]
        tgt = "English" if tgt_code == "en" else OPUS_LANGS[tgt_code]
        if not dd.ok(text, check_eval_substring=True):
            return False
        prompt = rng.choice(TEMPLATES).format(src=src, tgt=tgt, text=text)
        rows.append(row(f"{src_code}2{tgt_code}", "opus100_translate", len(rows), prompt))
        return True

    de = pools.get("de", [])
    n = 0
    for en, xx in de:
        if n >= DE_EN:
            break
        n += add("de", "en", xx)
    langs = [c for c in pools if pools[c]]
    per_dir = (TRANSLATION_TOTAL - n) // (2 * len(langs))
    for code in langs:
        pool = pools[code][DE_EN if code == "de" else 0:]
        got_out = got_in = 0
        for en, xx in pool:
            if got_out < per_dir:
                got_out += add("en", code, en)
            elif got_in < per_dir:
                got_in += add(code, "en", xx)
            else:
                break
    print(f"[opus] translation prompts: {len(rows)} ({n} de->en, {per_dir}/direction over {len(langs)} languages)")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    api = HfApi()
    dd = Dedup(load_eval_texts())
    print(f"[eval] {len(dd.eval_texts)} eval prompts loaded for decontamination")

    nem = nemotron(args.raw, dd, rng, api)
    wc = wildchat(args.raw, dd, rng, api)
    instruct_topup(args.raw, dd, rng, wc, "th", "airesearch/WangchanThaiInstruct",
                   "data/train-00000-of-00001.parquet", "wangchan_thai_instruct")
    instruct_topup(args.raw, dd, rng, wc, "ko", "nlpai-lab/kullm-v2", "kullm-v2.jsonl", "kullm_v2")
    aya_topup(args.raw, dd, rng, wc)
    buckets = {lang: nem.get(lang, []) + wc.get(lang, []) for lang in list(NEMOTRON) + [l for l in WILDCHAT if l not in NEMOTRON]}
    buckets["translation"] = translation(args.raw, dd, rng, api)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    allrows = []
    for name, rows in buckets.items():
        p = args.out / f"prompts_multilingual_{name}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        manifest[name] = {"rows": len(rows), "sources": dict(Counter(r["source"] for r in rows))}
        allrows += rows
    rng.shuffle(allrows)
    with (args.out / "prompts_multilingual_all.jsonl").open("w", encoding="utf-8") as f:
        for r in allrows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest["_total"] = len(allrows)
    manifest["_dropped"] = dict(dd.dropped)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
