# RDU MTP k=5 eval

Same 25 benchmarks as [`../experiments/gemma4-31b-full.yaml`](../experiments/gemma4-31b-full.yaml):
50 samples (seed 42), greedy, `max_tokens=4096`. Compare **accept_length** and
**accept_rate** to GPU `assistant_k5`. Decode tok/s is logged but not comparable.

```bash
cd scripts/evaluate/rdu_mtp_eval
./submit_snrdu.sh
```

Results: `../experiments/results/gemma4-31b-rdu-k5/`
Log: `logs/snrdu.log`. Details append per sample; resubmit resumes.
Timeout default is 7 days (~2 tok/s × ~550k GPU completion tokens).
