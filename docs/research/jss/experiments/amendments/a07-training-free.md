# Amendment 7: dependency counts and QoS-attribution controls

**Paper:** §4.4 (`sec:4.4`, the references), §6.1 (`sec:rq1`, Table 6 reference block: `InDeg`,
`Reach`), §6.2 (`sec:rq2`, the closed-form QoS controls), §6.3 (Table 9 reference block), §6.4
(count cost), Figure 4.
**Supplement:** §S38 (`supp:amendment7`), §S44 (`supp:baselines`).
**Status:** registered secondary. Written 2026-09-25, before any of these numbers existed. Amendment
13 later reclassified `InDeg` and `Reach` as references that restate $I^*$'s rule, so their contrasts
no longer carry claim status in the main text ([A13](a13-reference-demotion.md)).
**Registration:** [`../../PREREGISTRATION.md`](../../PREREGISTRATION.md), Amendment 7 (commit
`4878cf28`). Decision rules R1, R2 and R2′ all applied. R3 (report every arm) was discharged by the
dependency-graph revision ([A9](a09-dependency-graph-learning.md)).

## What it asks

1. Does a training-free count on SaG's dependency projection already match the learned engines?
   The primary oracle is a cascade over the same dependency rules the projection encodes, which is
   why Amendment 13 later reported these counts as references.
2. Is the Topo → Topo-QoS gain (0.349 → 0.553) produced by the declared QoS contracts, or by the
   projection itself?
3. How much of every correlation is only the separation of inert components (I* = 0)?

## Arms

All arms read the Application–Library `DEPENDS_ON` projection (Rules 1 and 5):

| Arm | Score |
|---|---|
| `InDeg` | direct dependents |
| `Reach` | transitive dependents (`nx.ancestors`), normalised |
| `Reach-QoS` | sum over transitive dependents of the best-path product of `qos_weight` |
| `CDI` | Connectivity Degradation Index alone |
| `Topo-Mult` | Topo-QoS with every topic weight fixed at 0.5 |
| `Topo-QoS-Perm` | Topo-QoS with QoS profiles permuted across topics (20 permutations) |
| QoS-independent corpus | the twelve folds regenerated with `qos_affinity: false` |

The run also includes an oracle sweep over θ ∈ {0.1, 0.2, 0.3} × damping step ∈ {0.10, 0.15, 0.20},
the inert-vs-active rule "active iff the component has a dependent", and label-structure
descriptives.

## Reproduce (no GPU, no Neo4j)

```bash
PYTHONPATH=. python reproduce/training_free_suite.py all          # gate, baselines, controls, oracle, descriptives
PYTHONPATH=. python reproduce/training_free_suite.py make-variant  # QoS-independent corpus -> output/variants/qos_indep/
PYTHONPATH=. python reproduce/training_free_suite.py qos-indep
PYTHONPATH=. python reproduce/training_free_suite.py substrate     # what the registered Topo measured
PYTHONPATH=. python reproduce/training_free_suite.py cost
PYTHONPATH=. python reproduce/render_amendment7_tables.py          # latex/supp_amendment7.tex
```

**Before scoring anything new,** the harness regenerates I* with the published settings. It then
reproduces every published per-fold Topo-QoS value to three decimals
(`results/tf_reproduction_gate.json`). It re-executes itself with `PYTHONHASHSEED=0`, because the CDI
sample breaks degree ties in set-iteration order.

`reproduce/render_amendment7_figure.py` renders this amendment's original figure. That figure is no
longer in the paper, and the script still writes `latex/figures/Figure_5`, which is now the recall
figure (`reproduce/render_recall_figure.py`). Do not run it against the current manuscript.

## Artifacts

Tracked in git under `results/`, unlike most of the bundle:
- `tf_reproduction_gate.json`
- `qos_attribution_controls.json`
- `qos_indep_corpus.json`
- `topo_substrate_check.json`
- `oracle_param_sensitivity.json`
- `system_model_descriptives.json`
- `dependency_count_cost.json`

`tf_baselines.json`, which Table 6 and Table 9 read for `InDeg` and `Reach`, is not tracked. It
ships in the Zenodo bundle and regenerates with `training_free_suite.py all`.

## Outcome

- **InDeg** ranks at LOSO ρ 0.764 and beats Topo-QoS on 12/12 folds.
- **Reach** ranks at 0.732, and **Reach-QoS** at 0.714.
- **On the system models**, Reach reaches 0.938 and InDeg 0.863.
- **What drives Topo-QoS:**
  - Unweighted betweenness on the same projection scores 0.591.
  - Constant topic weights score 0.595, and permuted QoS scores 0.559.
  - On the QoS-independent corpus, QoS weighting changes the score by −0.035.
- **Inert-vs-active rule:** 94% balanced accuracy.

The full per-fold results are in §S38 (`supp:amendment7`).
