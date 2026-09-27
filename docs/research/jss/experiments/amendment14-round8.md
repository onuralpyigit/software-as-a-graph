# Amendment 14: round-8 referee arms and re-analysis

**Paper:** §4.2 (selection-rule deviation), §4.3 (full-population `I_dyn`), §4.4 (reference
criterion), §6.3 (status tiers), §7.1 (Table `tab:independent_oracles`, hybrid attribution, TOST),
§7.2 (Table `tab:a14`), §7.4 (cost), §8, and Supplementary Section `supp:round8`.
**Registered:** [`PREREGISTRATION.md`](../PREREGISTRATION.md), Amendment 14 and the deviation record
above it, committed at `714e70fc` before any Amendment 14 arm ran. The TOST margin is not blind (see
the amendment's "Status when written").
**Reviews:** [R1](../reviews/review_2026-09-26_round8.md), [R2](../reviews/review_2026-09-26_round8_r2.md);
response: [response_round8.md](../reviews/response_round8.md).

## Reproduce

```bash
make -f reproduce/Makefile rq-amendment14      # arms D, S, W, Y + comparators, one CPU invocation (~2.5 h on 20 cores)
make -f reproduce/Makefile rq-referee-round8   # F3, F7, full-I_dyn Table 7, hierarchical CIs (minutes)
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

## Outcomes (decision rules of Amendment 14)

- **F1a.** The dependency-graph GAT needs its degree features: −0.136 without `in_degree`/`w_in`,
  −0.161 without the strict set (Holm 0.005).
- **F2b.** No significant aggregator effect after Holm (+0.108, +0.124; Holm 0.157). GIN keeps
  0.721 / 0.711 without degree features.
- **F3b.** No learner is equivalent to `InDeg` within ±0.05; "never exceeds" is withdrawn.
- **F4b.** With `w_in` held, the "QoS" main effect is +0.030 (Holm 0.33), down from +0.073.
- **F5b.** The GNN surrogate for `I_dyn` fails (0.598, below Analytic-I* 0.706 and GBM 0.799).
- **F6.** See `referee_round8_nested.json` and §7.2.
- **F7.** Hybrids do not beat their own base learners; only Hybrid-GAT beats unweighted
  betweenness and constant weights after Holm (0.049).

## Not done

Equal-budget tuning beyond the stage-1 grid, and for arms other than `hgl_qos` and
`gl_proj_qos16_cap`; a second modeller; validation against observed outages; energy measurement.
