# AgentX raw artifacts — PROVENANCE WARNING

These are the raw `aiperf` exports from the **2026-10-21 parallel** AgentX run
(section 29 of `docs/experiments/gemma4_26b_moe_results.md`).

**Only ONE arm's data survives here, and it is ours_b16.** Both arms of that run
wrote to this same path: `agentx_tp1_c1.sh` passed `OUTPUT_DIR`, but
`scripts/evaluate/mtp_server_eval/run_agentx.sh:61` reads `RESULT_DIR`, so both
fell back to the shared default `./results/agentx/users1`. The arm finishing
second (ours, 11:00:10) overwrote the first (the Google MTP assistant,
10:59:51). `out_tok_s` here is 50.848, matching ours_b16's parsed summary, which
is how the surviving arm was identified.

Do not read these files as a two-arm comparison. The per-arm *parsed* summaries
in each `agentx.log` are the record of truth for section 29, and they are
intact: `AR = (AL-1)/7` holds exactly for both arms (4.007 -> 0.4296 and
4.426 -> 0.4894), and the two arms disagree with each other, which could not
happen if one had parsed the other's file.

Fixed 2026-10-21: the runner now passes `RESULT_DIR="$R"` per arm, so later runs
land in `scripts/evaluate/experiments/results/agentx-tp1-c1/<arm>/`. Section
29.1's serial rerun supersedes this run and has per-arm raw exports.
