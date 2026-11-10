#!/usr/bin/env python3
"""Label every nemo782k prompt by heuristic category and join with the regenerated rows.
Writes analysis/prompt_categories.tsv (id, cat, lang, n_human, escaped, prompt_chars) and
prints prompt-side and response-side distributions."""
import json, re, sys, statistics as st, collections, time
D="/import/ml-sc-scratch1/mengmengj/speculators/data/glm53"
OUT=f"{D}/analysis/prompt_categories.tsv"
GEN=["nemo782k_regen_glm53.jsonl","nemo782k_regen_glm53_shardA_rem1.jsonl","nemo782k_regen_glm53_shardA_rem2.jsonl",
     "nemo782k_regen_glm53_shardB1.jsonl","nemo782k_regen_glm53_shardB2.jsonl","nemo782k_regen_glm53_shardB3.jsonl"]
R_MATH=re.compile(r"Solve the following math problem|put the answer \(and only answer\) inside \\boxed|\\boxed\{\}")
R_OPT=re.compile(r"^\s*\(?[A-J][:)\.]\s", re.M)
R_CODEFENCE=re.compile(r"```")
R_CODE=re.compile(r"\b(python|javascript|typescript|java|c\+\+|c#|rust|golang|\bgo\b|sql|bash|shell|html|css|php|ruby|kotlin|swift|scala|haskell|matlab|numpy|pandas|pytorch|tensorflow|react|django|flask|regex|api|json|xml|docker|kubernetes|git)\b|\b(write|implement|create|fix|debug|refactor|optimi[sz]e)\b.{0,40}\b(function|program|script|code|class|method|algorithm|query|snippet)\b|\bdef \w+\(|\bclass \w+|#include|import \w+|SELECT .* FROM|console\.log|print\(", re.I|re.S)
R_TOOL=re.compile(r"<tools>|\"tools\"\s*:|function[_ ]call|tool[_ ]call", re.I)
R_IF=re.compile(r"In your (entire )?response|Your (entire )?(response|answer) (should|must)|Answer with (at least|at most|one of)|Do not (use|include) (the )?(keywords?|words?)|Wrap your entire response|Finish your response with|at least \d+ (words|sentences|paragraphs|bullet)|There should be \d+ paragraphs|highlight(ed)? (at least )?\d+ sections|Include keywords|include the keywords|P\.S\.|postscript|all lowercase|ALL CAPITAL|no commas", re.I)
R_ESC=re.compile(r"\\u[0-9a-fA-F]{4}")
R_CJK=re.compile(r"[一-鿿぀-ヿ가-힯]")
R_CYR=re.compile(r"[Ѐ-ӿ]")
R_LAT=re.compile(r"[A-Za-z]")

def lang(t):
    c=len(R_CJK.findall(t)); y=len(R_CYR.findall(t)); l=len(R_LAT.findall(t))
    if c>=20 and c>0.3*(l+1): return "cjk"
    if y>=20 and y>0.3*(l+1): return "cyrillic"
    return "en/latin"

def cat(hum_first, hum_all, n_human):
    if R_MATH.search(hum_first): return "math"
    if len(R_OPT.findall(hum_first))>=4: return "stem_mcq"
    if R_TOOL.search(hum_all): return "tool_calling"
    if R_CODEFENCE.search(hum_all) or R_CODE.search(hum_first): return "code"
    if R_IF.search(hum_first): return "chat_ifeval"
    return "chat_other"

t0=time.time()
info={}
with open(f"{D}/prompts_nemo782k.jsonl",encoding="utf-8") as fh, open(OUT,"w") as out:
    out.write("id\tcat\tlang\tn_human\tescaped\tprompt_chars\n")
    for line in fh:
        r=json.loads(line)
        hum=[c["value"] for c in r["conversations"] if c["from"]=="human"]
        hum_all="\n".join(hum)
        c=cat(hum[0],hum_all,len(hum)); lg=lang(hum_all); esc=bool(R_ESC.search(hum_all))
        info[r["id"]]=(c,lg,len(hum),esc,len(hum_all))
        out.write(f"{r['id']}\t{c}\t{lg}\t{len(hum)}\t{int(esc)}\t{len(hum_all)}\n")
print(f"labelled {len(info)} prompts in {time.time()-t0:.0f}s", flush=True)

def pct(n,d): return f"{100*n/d:5.1f}%" if d else "  -  "
N=len(info)
print("\n== PROMPT-SIDE DISTRIBUTION (all 782k prompts) ==")
cc=collections.Counter(v[0] for v in info.values())
for k,n in cc.most_common(): print(f"  {k:13s} {n:7d} {pct(n,N)}")
print("  language:", {k:f"{n} ({100*n/N:.1f}%)" for k,n in collections.Counter(v[1] for v in info.values()).most_common()})
print("  multi-turn (>1 human):", sum(v[2]>1 for v in info.values()), " unicode-escaped prompts:", sum(v[3] for v in info.values()))
print("  prompt chars median:", st.median(v[4] for v in info.values()), " 8000+:", sum(v[4]>=8000 for v in info.values()))
print("  category x language:")
cl=collections.Counter((v[0],v[1]) for v in info.values())
for k in cc:
    print(f"    {k:13s}", {l:cl[(k,l)] for l in ("en/latin","cjk","cyrillic") if cl[(k,l)]})

# response side
seen=set(); rows=collections.defaultdict(list)
for f in GEN:
    try: fh=open(f"{D}/{f}",encoding="utf-8",errors="replace")
    except FileNotFoundError: continue
    for line in fh:
        try: r=json.loads(line)
        except Exception: continue
        rid=r.get("id")
        if rid in seen or rid not in info: continue
        seen.add(rid); m=r.get("metadata",{})
        rep=[c["value"] for c in r["conversations"] if c["from"]=="gpt"]; rep=rep[-1] if rep else ""
        ct=(m.get("usage") or {}).get("completion_tokens") or 0
        th=rep.find("</think>"); ts=(th/len(rep)) if th>=0 and rep else (1.0 if rep else 0.0)
        rows[info[rid][0]].append((bool(m.get("truncated")), ct, ts, (m.get("finish_reasons") or [""])[-1]=="stop"))
    print(f"  read {f}: cumulative {len(seen)} rows ({time.time()-t0:.0f}s)", flush=True)
print(f"\n== RESPONSE-SIDE (GLM-5.3 regen rows so far: {len(seen)}) ==")
print(f"  {'category':13s} {'prompts':>8s} {'generated':>9s} {'done%':>6s} {'trunc%':>7s} {'stop%':>6s} {'ct med':>7s} {'ct mean':>8s} {'ct p90':>7s} {'think%':>7s} {'kept(non-trunc)':>15s}")
tot_t=tot=0
for k,n in cc.most_common():
    rs=rows[k]
    if not rs: print(f"  {k:13s} {n:8d} {0:9d}"); continue
    cts=sorted(x[1] for x in rs); tr=sum(x[0] for x in rs); tot+=len(rs); tot_t+=tr
    print(f"  {k:13s} {n:8d} {len(rs):9d} {pct(len(rs),n):>6s} {pct(tr,len(rs)):>7s} {pct(sum(x[3] for x in rs),len(rs)):>6s} {cts[len(cts)//2]:7d} {st.mean(cts):8.0f} {cts[int(len(cts)*0.9)]:7d} {100*st.mean(x[2] for x in rs):6.1f}% {len(rs)-tr:15d}")
print(f"  overall truncated {tot_t}/{tot} = {pct(tot_t,tot)}  -> kept after dropping truncated: {tot-tot_t}")
print("\n== truncation by language / multi-turn ==")
by=collections.defaultdict(list)
for f in []: pass
