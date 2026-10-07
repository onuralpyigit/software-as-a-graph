# RQ4 — Cost

**Paper:** §6.4 (`sec:rq4`), Table 10 (`tab:cost-ll`); energy in §7.4 (`sec:threats`, "Energy
estimation and Green AI").
**Supplement:** §S41 (`supp:round8`, Table S66 `tab:r8-cost`: per-fold like-for-like cost), §S33
(`supp:gate-ratio`: the analysis gate against direct simulation, and training cost), §S34
(`supp:scale`: legacy latency and scaling micro-benchmarks).
**Status:** descriptive measurements. This is the plan's RQ5. The like-for-like table was added by
Amendment 14 ([page](amendments/a14-round8.md)).
**Registration:** [`../PREREGISTRATION.md`](../PREREGISTRATION.md), the plan (RQ5), Amendments 4 and 14.

## Question

What does each ranking path cost, measured like for like? The paths are:
- running the reachability oracle $I^*$ directly;
- counting dependents;
- extracting the learned rankers' features;
- labeling with the queue-flow oracle $I_\text{dyn}$.

## Protocol

Every stage is timed on the same graph in one session: CPU, one thread, median of three runs. Each
corpus fold is timed, plus generated graphs of 249–4,995 components.
- **Count:** the Application–Library projection plus `InDeg`.
- **One $I^*$ pass:** seed 42, Applications only.
- **Sweep:** the published five-seed labeling run over Applications, Brokers and Libraries.
- **Features:** the analysis that produces the learned rankers' node features.
- **Gate:** the system-layer analysis with 18 anti-pattern detectors, for reference.

## Reproduce

```bash
make -f reproduce/Makefile rq-cost-reconcile    # Table 10 and Table S66; run alone, on an idle machine
make -f reproduce/Makefile inference-latency    # per-stage latency (§S34; paper artifact inference_latency_v3.json)
PYTHONPATH=. python reproduce/oracle_timing.py \
    --repeats 3 --output results/oracle_timing_jss12.json   # oracle side of §S33 + gate_oracle_ratio.json
```

## Artifacts

- `data/benchmarks/referee_round8_cost.json`: Table 10 and Table S66, checked by
  `reconcile_manuscript.py::check_cost_ll`.
- `data/benchmarks/oracle_timing_jss12.json`, `gate_oracle_ratio.json` and
  `results/detection_validation_timed_jss12.json` (`summary.gate_seconds`): §S33.
- `data/benchmarks/oracle_robust_ltr.json` (`cost` block) and `idyn_full_labels_jss12.json`: the
  $I_\text{dyn}$ labeling time ([A11](amendments/a11-oracle-robust.md)).
- `results/dependency_count_cost.json`: count and `Reach` cost per scenario
  ([A7](amendments/a07-training-free.md)).
- `results/energy_estimate.json` (`reproduce/energy_estimate.py`): the energy figures.

## Headline result

- **Counting is far cheaper than one oracle pass.** One $I^*$ pass over a corpus architecture's
  Applications takes 0.01–0.72 s. The counting path that restates its first wave is 17–176× cheaper
  (median 45×).
- **Learned rankers pay for their features.** Feature extraction costs 4.5–72.5× one $I^*$ pass
  (median 16.9×). The ordering holds on generated graphs up to 5,000 components. The articulation and
  CDI phase accounts for 88–91% of it, so this cost belongs to the chosen feature set, not to learned
  ranking as such.
- **Inference is negligible.** Tens of milliseconds at 2,000 components. The count scales to 10,000
  components (1.1 s); `Reach` less well (23 s).
- **The expensive oracle is $I_\text{dyn}$.** Labeling the corpus took 12.7 CPU-hours, against at most
  about a millisecond per architecture for the rate-weighted reference (Eq. 7). No learned
  approximation of $I_\text{dyn}$ breaks even against Eq. 7 on this corpus: it must pay for its labels
  and training, and it is not more accurate.

## Notes

- **Superseded cost figure.** An earlier version compared the system-layer analysis gate with the
  five-seed oracle sweep and reported a 2–18× (median 5.6×) premium. That compares different
  workloads. It survives only as the gate-vs-oracle table in §S33. The like-for-like Table 10 replaced
  it in round 8.
- **Where the gate premium peaks** (§S33, Table S29). The gate-to-oracle ratio is largest where
  the derived projection is dense. Enterprise is the maximum (17.7×): its 300 applications share 120
  topics, so Rule 1 derives a near-complete graph of 26,276 edges.
- **Why CDI is computed for every node.** Restricting CDI to articulation points leaves it
  identically zero wherever removal does not literally disconnect the graph. That drives the
  availability term to a near-constant in exactly the redundant multi-publisher topologies SaG
  targets.
- **Training cost** (one-off per model version, CPU, 60 fits per arm): 7.7 CPU-hours for four
  learned arms, about 0.22 kWh. The GPU sweep behind §S36 did not record per-fit durations.
- **Energy.** Figures are wall-clock time × the processor's 28 W base power: a nameplate estimate, not
  a RAPL/NVML measurement, and not a bound in either direction. The paper's energy argument rests on
  the break-even of a learned $I_\text{dyn}$ approximation (labels ≈ 355.6 Wh), not on differences
  between rankers.
- **Not implemented.** Incremental re-scoring of only the $k$-hop neighbourhood of a change. Table 10
  times full recomputation.
