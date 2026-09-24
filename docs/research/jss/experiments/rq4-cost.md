# RQ4 — Analysis cost and comparison with direct simulation

**Paper:** §7.4, Table 10, and the sustainability paragraph of §8.1. **Extended results:**
Supplement S28 (per-scenario gate vs. oracle, training cost).

## Reproduce

```bash
make -f reproduce/Makefile inference-latency       # per-stage latency; paper artifact: inference_latency_v3.json
PYTHONPATH=. python reproduce/oracle_timing.py \
    --repeats 3 --output results/oracle_timing_jss12.json   # oracle side + gate_oracle_ratio.json
```

The gate side of the comparison is `summary.gate_seconds` in
`detection_validation_timed_jss12.json`. The oracle and gate timings in the paper come from one
paired measurement session at one commit and corpus digest.

## Headline result

- The HGT forward pass takes 56 ms at 2,000 components, 0.02% of the ~4-minute end-to-end
  evaluation.
- Cost is dominated by the Connectivity Degradation Index (CDI), which is $O(|V|^2 + |V||E|)$.
- The analysis gate costs 2–18× (median 5.6×) the in-process cascade simulation, and it is more
  expensive on all twelve scenarios.

## Notes cut from the paper

- **Scaling.** From 249 to 1,998 components, wall-clock rises 138× against a 137× growth in
  $|V||E|$. That is an endpoint coincidence, not a tracked curve: between the middle rows $|V|$
  doubles while time rises 5.4×. The series varies $|V|$ and $|E|$ together; the per-scenario table
  (S28) separates them.
- **Why CDI is computed for every node.** Restricting CDI to articulation points leaves it identically
  zero wherever removal does not literally disconnect the graph. That drives $A(v)$ to a
  near-constant in exactly the redundant multi-publisher topologies SaG targets. CDI is also a
  predictor input, so gating it would change both pathways.
- **What predicts the premium.** The gate-to-oracle ratio tracks the size of the derived projection
  (ρ = 0.951 with $|E_\text{proj}|$) better than component count (0.792).
  - Enterprise is the maximum: its 300 applications share 120 topics, so Rule 1 derives a
    near-complete graph of 26,276 edges.
  - Across sessions the maxima range over 77–83 s (gate) and 4.5–4.8 s (oracle).
- **Training cost** (one-off per model version, CPU, 60 fits per arm): 0.6 h `GAT-S`, 0.9 h
  `GAT-S-w`, 1.4 h `HGT`, 4.9 h `HGT-QoS`, 7.7 CPU-hours in all. The GPU sweep behind Supplement Table S29 did
  not record per-fit durations.
- **Energy bound.** At 28 W base SoC power, one gate pass over the twelve scenarios costs ≤ 3.0 kJ
  (0.83 Wh), and training the four arms once costs about 0.78 MJ (0.22 kWh).
  - Both figures are upper bounds from wall-clock time, not RAPL/NVML measurements.
  - SaG's sustainability case is avoided staging infrastructure, not CPU time.
- **Not implemented.** Incremental re-scoring of only the $k$-hop neighbourhood of a change. Table 10
  times full recomputation.
