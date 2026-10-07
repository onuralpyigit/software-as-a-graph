# Amendment 14: round-8 referee arms and re-analysis

**Paper:** §4.2 (selection-rule deviation), §4.3 (full-population $I_\text{dyn}$), §4.4 (reference
criterion), §5.3 (`sec:6.3`, status tiers), §6.1 (`sec:rq1`: Table 6 `tab:independent_oracles`,
hybrid attribution, TOST), §6.2 (`sec:rq2`: Table 7 `tab:controls`, F1, F2, F4; nested selection),
§6.4 (Table 9 `tab:cost-ll`), §7.4 (seed sensitivity).
**Supplement:** §S40 (`supp:round8`, Tables S59–S65), §S42 (`supp:controls`, Table S69 `tab:a14`),
§S27 (rows "Deviation" and A14).
**Status:** registered secondary. Committed at `714e70fc`, together with the deviation record above
it, before any Amendment 14 arm ran. The TOST margin is not blind (see the amendment's "Status when
written").
**Registration:** [`../../PREREGISTRATION.md`](../../PREREGISTRATION.md), Amendment 14 and the deviation
record.
**Reviews:** [R1](../../reviews/review_2026-09-26_round8.md), [R2](../../reviews/review_2026-09-26_round8_r2.md);
response: [response_round8.md](../../reviews/response_round8.md).

## Reproduce

```bash
make -f reproduce/Makefile rq-amendment14      # arms D, S, W, Y + comparators, one CPU invocation (~2.5 h on 20 cores)
make -f reproduce/Makefile rq-referee-round8   # F3, F7, full-I_dyn Table 6, hierarchical CIs (minutes)
PYTHONPATH=. python reproduce/referee_round8.py degree_leak nested
make -f reproduce/Makefile rq-cost-reconcile   # alone, on an idle machine
```

Arm N (nested selection) runs one process per outer fold:
`reproduce/nested_loso_search.py --grid stage1 --inner-mode holdout --inner-k 2 --outer <fold>`
for `hgl_qos` and `gl_proj_qos16_cap`; shard reports go to `results/nested_a14_shards/`.

Run the LOSO sweep from the main checkout, or export `LOSO_CACHE_DIR`: the training-free `topo_qos`
row resolves its cache through a relative path (`reproduce/main_table._find_cache_dir`), and from a
worktree without `output/loso_cache` it silently scores a different substrate. That is what happened
in the recorded run; gate G0 records it, and no Amendment 14 contrast reads that row.

## Artifacts

| Artifact | Contents |
|---|---|
| `results/loso_amendment14_cpu.json` | 17 arms × 12 folds × 5 seeds |
| `results/loso_significance_amendment14_cpu.json` | w_in-held 2×2 factorial (F4) |
| `results/realworld_zeroshot_*_amendment14.json` | zero-shot, with per-seed predictions |
| `data/benchmarks/referee_round8_amendment14.json` | G0, F1–F5, every arm on three oracles |
| `data/benchmarks/referee_round8_{hybrid,tost,table7,hierarchical,degree_leak}.json` | re-analysis |
| `data/benchmarks/referee_round8_nested.json` | F6 (arm N) |
| `data/benchmarks/referee_round8_cost.json` | like-for-like timing |

## Outcome (decision rules of Amendment 14)

- **F1a.** The dependency-graph GAT needs its degree features: −0.136 without `in_degree`/`w_in`,
  −0.161 without the strict set (Holm 0.005).
- **F2b.** No significant aggregator effect after Holm (+0.108, +0.124; Holm 0.157). GIN keeps
  0.721 / 0.711 without degree features.
- **F3b.** No learner is equivalent to `InDeg` within ±0.05; "never exceeds" is withdrawn.
- **F4b.** With `w_in` held, the "QoS" main effect is +0.030 (Holm 0.33), down from +0.073.
- **F5b.** The GNN surrogate for `I_dyn` fails (0.598, below Analytic-I* 0.706 and GBM 0.799).
- **F6 (arm N).** The plan's nested selection on the stage-1 grid moves `HGT-QoS` from 0.622 to
  0.677 and `GAT-P-QoS` from 0.748 to 0.693, neither significantly (Holm 0.259). Nested `HGT-QoS` vs
  `Topo-QoS` is +0.123 (9/12, Holm 0.192). Per-fold values are in Table S64 (`tab:r8-nested`).
- **F7.** Hybrids do not beat their own base learners. Only Hybrid-GAT beats unweighted betweenness
  and constant weights after Holm (0.049).
- **Full-population $I_\text{dyn}$** (rule R of Amendment 11) replaces the n = 30 sample in Table 6.

## Not done

- Equal-budget tuning beyond the stage-1 grid, and for arms other than `hgl_qos` and
  `gl_proj_qos16_cap`.
- A second modeler.
- Validation against observed outages.
- Energy measurement.
