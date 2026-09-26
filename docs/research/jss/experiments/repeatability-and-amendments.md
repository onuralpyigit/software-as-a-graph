# Registered analysis plan, amendments, and repeatability

**Paper:** §6.3 (statistics and registration), §8.2 (threats). **Full text:**
[`../PREREGISTRATION.md`](../PREREGISTRATION.md). **Extended results:** Supplement S24 (amendment
log and omnibus Holm table).

## Registered, not pre-registered

The primary contrast was registered in this repository before the twelve-fold harness produced any
result: `HGT-QoS` vs `Topo-QoS` under LOSO, with five fixed seeds and the fold as the unit of
analysis. The protocol, statistic, unit and reporting commitment were all fixed at that point. We
call it *registered* rather than pre-registered for two reasons:
- the plan is a file under our own version control, with no third-party timestamp;
- an earlier, withdrawn eight-fold estimate of the same contrast predates it.

The registration shows that the analysis was not selected after the twelve-fold result was seen. It
does not show that the question was asked without any prior estimate.

## Amendments

| Entry | Written | What it did | Registered contrasts |
|---|---|---|---|
| Plan | Before any twelve-fold result | Primary and secondary contrasts vs `Topo-QoS` | 2 |
| A1 | Before any outer result | Two-fold inner holdout for selection (compute budget) | — |
| A2 | Before any control result | Capacity/channel/directionality-matched controls, label-side arm | 5 |
| A3 | Before any v5 result | Re-baseline as the v5 sweep; withdraw an unbacked paragraph | — |
| A4 | **After** the v5 results | Peer-review revision; every analysis it added is exploratory | — |
| A5 | Before any hybrid result | Hybrid-HGT | 2 |
| A6 | Before its result, after A2's | Hybrid-GAT, with a transfer criterion | 2 |

Dates, outcomes and the completeness statement are in Supplement S24. The sequence is adaptive:
Amendment 6 followed Amendment 2's result. Correcting within each family therefore does not bound the
error accumulated across the sequence. `make -f reproduce/Makefile omnibus` pools all thirteen
registered contrasts under one Holm correction (`omnibus_registered_holm.json`). The last two are
Amendment 2's directionality and capacity controls, both run after Amendment 7. Both hybrid
primaries survive it; no other registered contrast reaches α = 0.05.

## Repeatability

- **At fixed code, seeds and device**, every figure reproduces at its reported precision. A clean
  re-run of the hybrid sweep from a tagged commit changed no cell by more than 1.3×10⁻⁴, which is
  tie-breaking residue in QoS-weighted betweenness.
- **Training-free cells.** All 180 training-free cells of the main sweep reproduce across devices.
- **Learned cells** move across code revisions and devices: up to 0.172 in a fold mean between the
  last two sweeps, and 0.041 for `HGT-QoS`. There are three causes:
  - a since-fixed PyTorch Geometric device-placement issue;
  - stale checkpoint resumption;
  - non-deterministic CUDA reductions.
  
  Learned figures are therefore reported against the released artifacts, and every comparison is
  made within one sweep.
  - Measure the drift between two sweeps at the same corpus digest with
    `python reproduce/rerun_drift.py --before <sweep_a.json> --after <sweep_b.json>`
    (published: `rerun_drift.json`).
- **Zero-shot re-run.** Re-running the zero-shot `HGT-QoS` evaluation on CPU reproduced every
  published per-system value of Table 10.
- **Reconciliation.** `python reproduce/reconcile_manuscript.py --verbose` checks every reported
  table figure in the manuscript and supplement against its artifact. It flags stale-corpus or
  dirty-tree provenance and runs in CI with `--allow-missing`. It does **not** check numbers that
  appear only in prose.
