# Registered analysis plan, amendments and repeatability

**Paper:** §5.3 (`sec:6.3`, statistics, registration and status tiers), §7.4 (`sec:threats`,
internal and conclusion validity).
**Supplement:** §S27 (`supp:amendments`: amendment log Table S22, omnibus Holm Table S23).
**Full text:** [`../PREREGISTRATION.md`](../PREREGISTRATION.md).

## Registered, not pre-registered

The primary contrast was registered in this repository before the twelve-fold harness produced any
result: `HGT-QoS` vs `Topo-QoS` under LOSO, with five fixed seeds and the fold as the unit of
analysis. The protocol, statistic, unit and reporting commitment were all fixed at that point. We
call it *registered* rather than pre-registered for two reasons:
- the plan is a file under our own version control, with no third-party timestamp;
- an earlier, withdrawn eight-fold estimate of the same contrast predates it.

The registration shows that the analysis was not selected after the twelve-fold result was seen. It
does not show that the question was asked without any prior estimate.

**Status tiers** (§5.3):
- the plan's two co-primary contrasts are *confirmatory*;
- an amendment written before any result of its own arms is *registered secondary*;
- everything else is *exploratory*.

## Amendment log

"Before" means no outcome of the analysis the entry governs existed when it was committed. Contrast
counts are the decision-bearing contrasts the entry registered. Dates and outcomes match Table S22.

| Entry | Date | Written | What it did | Contrasts | Page |
|---|---|---|---|---|---|
| Plan | 2026-09-06 | before any twelve-fold result | co-primary `HGT-QoS` / `HGT` vs `Topo-QoS`; LOSO, five seeds; nested selection rule | 2 | [RQ1](rq1-ranking-accuracy.md) |
| A1 | 2026-09-06 | before any outer result | two-fold inner holdout for selection (compute budget); not applied | — | — |
| A2 | 2026-09-12 | before any control result | capacity-, channel- and directionality-matched controls; label-side arm | 7 | [RQ2](rq2-sources-of-performance.md) |
| A3 | 2026-09-13 | before any v5 result | re-baseline as the v5 sweep; withdraw an unbacked paragraph; declare `GBM-Feat` | — | [RQ1](rq1-ranking-accuracy.md) |
| A4 | 2026-09-20 | **after** the v5 results | peer-review revision; every analysis it added is exploratory | — | [RQ3](rq3-zero-shot-transfer.md), [RQ4](rq4-cost.md) |
| A5 | 2026-09-23 | before any hybrid result | Hybrid-HGT | 2 | [RQ1](rq1-ranking-accuracy.md#hybrids-amendments-5-and-6) |
| A6 | 2026-09-24 | before its result, after A2's | Hybrid-GAT, with a transfer criterion | 2 | [RQ1](rq1-ranking-accuracy.md#hybrids-amendments-5-and-6) |
| A7 | 2026-09-25 | before any result | training-free dependency counts; QoS-attribution controls | 4 | [a07](amendments/a07-training-free.md) |
| A8 | 2026-09-26 | **after** its results | attribution controls on the raw multigraph | — (5 exploratory) | [a08](amendments/a08-attribution-controls.md) |
| A9 | 2026-09-26 | before its learned arms' results | the same learners on the dependency graph | 12 | [a09](amendments/a09-dependency-graph-learning.md) |
| A10 | 2026-09-26 | before any result | the value of the dependency derivation | 3 | [a10](amendments/a10-derivation.md) |
| A11 | 2026-09-26 | before any result | full-population $I_\text{dyn}$; learned combination of dependency signals | 6 | [a11](amendments/a11-oracle-robust.md) |
| A12 | 2026-09-26 | mixed, per arm | round-7 referee analyses | 12 (R1) | [a12](amendments/a12-referee-round7.md) |
| A13 | 2026-09-26 | **after** all results | counts reclassified as references; no number changed | — | [a13](amendments/a13-reference-demotion.md) |
| Deviation | 2026-09-26 | **after** all results | the plan's selection rule was not applied to any published learned result | — | [a14](amendments/a14-round8.md) (arm N) |
| A14 | 2026-09-26 | before any result | round-8 arms: degree-free and GIN learners, $w_\text{in}$-held 2×2, nested selection, TOST | 19 | [a14](amendments/a14-round8.md) |
| A15 | 2026-09-29 | **after** all results | rate-weighted reference for $I_\text{dyn}$ (Eq. 7); input attribution | — | [a15](amendments/a15-rate-expansion.md) |
| A16 | 2026-09-30 | before any result | reverse-edge direction control; corrected-prior and `InDeg`-prior hybrids | 8 | [a16](amendments/a16-direction-control.md) |
| A17 | 2026-10-04 | before any result | oracle-aligned features removed; learning on top of Eq. 7; node-order permutation | see page | [a17](amendments/a17-round12.md) |
| A17b | 2026-10-05 | after F13 was seen | two more permutation seeds | — | [a17](amendments/a17-round12.md) |
| A18 | 2026-10-05 | before the relabel | payload-aware $I_\text{dyn}$; **not run** | — | [a18](amendments/a18-payload-oracle.md) |
| A19 | 2026-10-06 | before any result | round-14 controls; on branch `jss-revision-round14`, not yet on `main` | see page | [a19](amendments/a19-round14.md) |

Table S22 on `main` lists the plan through A16 and the deviation. A17 and A18 are recorded in
`PREREGISTRATION.md`, and A19 on its branch.

## Omnibus correction

Each registration Holm-corrects only its own contrasts. The sequence is adaptive: Amendment 6
followed Amendment 2's result, so correcting within each family does not bound the error accumulated
across the sequence. `make -f reproduce/Makefile omnibus` (`reproduce/omnibus_holm.py`) pools all
thirteen registered contrasts of the plan, A2, A5 and A6 under one Holm correction
(`omnibus_registered_holm.json`, Table S23). Both hybrid primaries survive it (p_omni 0.041 and
0.019); no other contrast reaches α = 0.05. Later families (A7 onward) are corrected within
themselves and do not join the omnibus.

## Repeatability

- **At fixed code, seeds and device**, every figure reproduces at its reported precision. A clean
  re-run of the hybrid sweep from a tagged commit changed no cell by more than 1.3×10⁻⁴, which is
  tie-breaking residue in QoS-weighted betweenness. Every amendment from A14 onward re-runs its
  comparators in the same invocation and gates on an exact match (G0).
- **Training-free cells** reproduce across devices.
- **Learned cells** move across code revisions and devices: up to 0.172 in a fold mean between two
  sweeps. There are three causes:
  - a since-fixed PyTorch Geometric device-placement issue;
  - stale checkpoint resumption;
  - non-deterministic CUDA reductions.

  Every learned-versus-learned contrast is therefore paired within one CPU sweep.
  `python reproduce/rerun_drift.py --before <sweep_a.json> --after <sweep_b.json>` measures the drift
  between two sweeps at the same corpus digest (published: `rerun_drift.json`).
- **Node order and configuration.** Permuting node order moves `GAT-P-QoS` by about 0.04 per fold
  ([A17](amendments/a17-round12.md)). The plan's nested selection moves learned rankers by up to
  ±0.055 ([A14](amendments/a14-round8.md)).
- **Zero-shot re-run.** Re-running the zero-shot `HGT-QoS` evaluation on CPU reproduced every
  published per-system value.
- **Reconciliation.** `python reproduce/reconcile_manuscript.py --verbose` checks every reported
  table figure in the manuscript and supplement against its artifact, and flags stale-corpus or
  dirty-tree provenance. It does **not** check numbers that appear only in prose, or the numbers on
  these experiment pages.
