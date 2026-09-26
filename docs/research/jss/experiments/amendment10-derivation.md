# Amendment 10: the value of the dependency derivation

**Paper:** §7.1 ("Counting dependents on the derived graph ranks impact well"), §8.3 (construct
validity); Supplementary S33 (Table S44).
**Registered:** [`PREREGISTRATION.md`](../PREREGISTRATION.md), Amendment 10 (commit `0c52329d`),
before any of its numbers existed. It is an exploratory family of three contrasts with decision
rules E1–E3.
**Run:** `PYTHONPATH=. python reproduce/training_free_suite.py derivation` at commit `ed7b2582`,
clean tree. It needs no GPU and no Neo4j.

## What it asks

The referee report of 2026-09-26 (M2) asked what SaG's `DEPENDS_ON` derivation contributes beyond
a subscriber count. It proposed two checks: `InDeg` on the raw multigraph (2-hop subscriber counts
through topics), and `InDeg` without Rule 5.

Both checks are **identities** for an Application, so they are recorded rather than run.
- **InDeg on the raw multigraph.** Projection `InDeg` equals the number of distinct subscribers
  of the topics v publishes.
- **InDeg without Rule 5.** Rule-5 edges point into Libraries, so removing them cannot change any
  Application's `InDeg`.
- **Rule-1 reach.** Reach over Rule-1 edges only is the raw publisher → topic → subscriber
  closure.

`tests/test_dependency_graph_substrate.py` pins all three identities on every committed graph.
The measurable questions are:
- Does the derived count beat the counts available on the raw multigraph without derivation?
- Does the derived library rule add to transitive reach?

## Arms

| Arm | Score for Application v |
|---|---|
| `Degree-raw` | total degree of v in the raw multigraph |
| `Pubs-raw` | number of topics v publishes to |
| `Reach-R1` | transitive dependents over Rule-1 edges only |
| `InDeg`, `Reach` | comparators from Amendment 7, recomputed |

## Results

LOSO mean ρ over the twelve held-out architectures:

| Arm | Mean ρ |
|---|---|
| `InDeg` | 0.764 |
| `Reach` | 0.732 |
| `Reach-R1` | 0.674 |
| `Pubs-raw` | 0.431 |
| `Degree-raw` | 0.199 |

Registered contrasts, Holm across the three:

| Contrast | Δρ | Folds won | Holm p |
|---|---|---|---|
| `InDeg` vs `Degree-raw` | +0.565 | 12/12 | 0.0015 |
| `InDeg` vs `Pubs-raw` | +0.334 | 12/12 | 0.0015 |
| `Reach` vs `Reach-R1` | +0.058 | 9/12 | 0.0068 |

**Decision rules.**
- **E1 applies.** Deriving topic-mediated dependencies is what makes the count predictive.
- **E2 applies.** The derived library rule adds to transitive reach.
- **E3 applies.** Every arm is reported in Supplementary Table S44.

**Checks.**
- The identity holds with max |Δ| = 0 on all seventeen graphs.
- `InDeg` and `Reach` match `tf_baselines.json` exactly.

On the five system models none of the Libraries publishes or subscribes, so `Reach-R1` equals
`Reach` (0.938) on every one of them.

## Artifact

`results/derivation_ablation.json` (committed, like Amendments 7 and 9). The supplementary table
is rendered from it by `reproduce/render_amendment9_tables.py` and checked byte for byte by
`reproduce/reconcile_manuscript.py`.
