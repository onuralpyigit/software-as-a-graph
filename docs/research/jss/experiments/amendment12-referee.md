# Amendment 12: round-7 referee analyses (exploratory)

Answers the round-7 referee report
([../reviews/review_2026-09-26_round7.md](../reviews/review_2026-09-26_round7.md)). It was registered in
[PREREGISTRATION.md](../PREREGISTRATION.md) (Amendment 12) before its arms ran. Its deviations are logged
directly below the registration.

## What it asks

| Arm | Question (referee comment) | Artifact |
|---|---|---|
| R1 | Is the derivation needed, or only the choice of metric? Raw-multigraph rankers: Degree-raw, Pubs-raw, PR-raw, RevPR-raw and Reach-R1, against `I*`, `I_dyn` and `I_comp` (M2) | `data/benchmarks/referee_round7_raw_baselines.json` |
| R2 | Does any ranker carry `I_dyn` signal beyond `I*`? Partial Spearman ρ given `I*` and given the first-order expansion (M3) | `referee_round7_partial.json` |
| R3 | Are the learned engines any better on the other oracles? Saved predictions re-scored (M3v) | `referee_round7_learned_oracles.json` |
| R4 | How much must be flagged to catch the critical set? Tie-aware recall@k (M10) | `referee_round7_recall.json` |
| R5 | Table 9 on one harness, and how the system models differ structurally (M9) | `referee_round7_zeroshot.json` |
| R6 | What does counting cost, by size, against one `I*` pass? (M6) | `referee_round7_latency.json` |
| — | Fisher-z and size-weighted means; seed spread of the learned engines (minor 6, M11; descriptive) | `referee_round7_averaging.json` |

## Reproduce

```bash
make -f reproduce/Makefile rq-referee-round7            # R1–R6 and averaging
PYTHONPATH=. python reproduce/render_referee_tables.py  # Supplementary S-referee tables
PYTHONPATH=. python reproduce/render_recall_figure.py   # manuscript Figure 6
```

The script runs on CPU and trains nothing. R1–R5 take a few minutes. R6 takes about an hour,
because `I*` labelling time grows roughly quadratically with size.

## Deviations (logged in PREREGISTRATION.md)

1. **`I_dyn` is the published n = 30 lexical sample.** Amendment 11's full-population labelling was
   stopped before it wrote any labels. The lexical-sample limitation therefore stands.
2. **Gate G3 fails by construction.** The saved learned predictions are seed ensembles, so R3 reports
   seed-ensemble ρ. The per-seed logs reproduce every published mean.
3. **R6 does not time `I*` at 10,000 components.** It times `I*` once at 5,000.
4. **Population defect.** `subscriber_count_raw` and `pubs_raw` emit only publishers, and the metric code
   scores only emitted nodes. Fixed by zero-filling, which also corrects Amendment 10's Pubs-raw arm
   (see [amendment10-derivation.md](amendment10-derivation.md)).

## Headline outcomes

The manuscript tables carry the numbers; see Tables 6 and 7, Figure 6 and Supplementary S-referee.

- **R1.** `InDeg` equals the raw two-hop subscriber count on every graph. Untyped raw metrics fail on
  `I*` (degree 0.199, reverse PageRank 0.089, PageRank constant). Topics published comes within 0.033
  of `InDeg`. On `I_comp`, raw degree (0.719) beats `InDeg` (0.650).
- **R2.** `InDeg` keeps a partial ρ of 0.259 [0.143, 0.367] with `I_dyn` beyond `I*`. `Reach` keeps
  nothing.
- **R3.** No learned engine beats `InDeg` on `I_dyn`. Without a centrality prior, they collapse on
  `I_comp`.
- **R4.** To catch 80% of the true top-20% set, `InDeg` must flag its top 45%.
- **R5.** `InDeg` and `Reach` are unchanged on the learned harness's labels. The system models have
  more inert Applications and more concentrated fan-in than the synthetic folds.
