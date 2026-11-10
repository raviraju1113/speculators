import csv, json, collections, statistics as st
D="/import/ml-sc-scratch1/mengmengj/speculators/data/glm53"
info={r["id"]:r for r in csv.DictReader(open(f"{D}/analysis/prompt_categories.tsv"),delimiter="\t")}
GEN=["nemo782k_regen_glm53.jsonl","nemo782k_regen_glm53_shardA_rem1.jsonl","nemo782k_regen_glm53_shardA_rem2.jsonl",
     "nemo782k_regen_glm53_shardB1.jsonl","nemo782k_regen_glm53_shardB2.jsonl","nemo782k_regen_glm53_shardB3.jsonl"]
def pb(n): return "<500" if n<500 else "<2000" if n<2000 else "<8000" if n<8000 else "8000+"
seen=set(); agg=collections.defaultdict(lambda:[0,0,0,0])  # key -> [rows, truncated, tokens_all, tokens_kept]
def add(key,tr,ct):
    a=agg[key]; a[0]+=1; a[1]+=tr; a[2]+=ct; a[3]+=0 if tr else ct
for f in GEN:
    for line in open(f"{D}/{f}",encoding="utf-8",errors="replace"):
        try: r=json.loads(line)
        except Exception: continue
        rid=r.get("id")
        if rid in seen or rid not in info: continue
        seen.add(rid); i=info[rid]; m=r.get("metadata",{})
        tr=int(bool(m.get("truncated"))); ct=(m.get("usage") or {}).get("completion_tokens") or 0
        add(("cat",i["cat"]),tr,ct); add(("multi","multi" if int(i["n_human"])>1 else "single"),tr,ct)
        add(("esc","escaped" if i["escaped"]=="1" else "clean"),tr,ct); add(("len",pb(int(i["prompt_chars"]))),tr,ct)
        add(("lang",i["lang"]),tr,ct); add(("all","all"),tr,ct)
T=agg[("all","all")]
for group in ("cat","multi","esc","len","lang"):
    print(f"\n== {group} ==   rows  trunc%  tokens(all) share  tokens(kept) share")
    for (g,k),a in sorted(agg.items(), key=lambda x:-x[1][0]):
        if g!=group: continue
        print(f"  {k:10s} {a[0]:7d} {100*a[1]/a[0]:6.1f}% {a[2]/1e6:8.0f}M {100*a[2]/T[2]:5.1f}% {a[3]/1e6:8.0f}M {100*a[3]/T[3]:5.1f}%")
print(f"\ntotal rows {T[0]}, tokens all {T[2]/1e9:.2f}B, tokens kept(non-trunc) {T[3]/1e9:.2f}B")
