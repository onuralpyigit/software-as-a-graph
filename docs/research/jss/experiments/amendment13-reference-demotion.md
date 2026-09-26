# Amendment 13: dependency counts reclassified as oracle-proximal references

**Paper:** §1, §3.3 (Proposition 1), §4.4, §6.2 ("Reference rankings"), §7.1–7.3 (reference blocks of
Tables `tab:hybrid`, `tab:independent_oracles` and `tab:system_models_transfer`), §8.1 (`tab:guidance`),
§8.3, and the supplement's amendment log.
**Registered:** [`PREREGISTRATION.md`](../PREREGISTRATION.md), Amendment 13, written *after* all results
existed. It is a reporting deviation: no arm is run, re-run, dropped or added, and no number changes.
**Run:** nothing new. Figures are re-rendered from existing artifacts:
`PYTHONPATH=. python reproduce/render_headline_figure.py`,
`PYTHONPATH=. python reproduce/render_recall_figure.py` and
`PYTHONPATH=. python reproduce/render_graphical_abstract.py`.

## Why

The authors decided not to use `InDeg` and `Reach` as predictors, because they are circular with the
labelling simulator:
- By Proposition 1, the first propagation wave of the primary oracle `I*` reaches exactly the
  Applications that `InDeg` counts.
- The oracle's first-order expansion (Analytic-I*) reaches ρ = 0.808.
- `Reach` counts the transitive dependents a reachability cascade can visit, and it keeps no `I_dyn`
  signal beyond `I*` (partial ρ 0.058, CI includes 0; Amendment 12, R2).

A ranker that restates the oracle's rule measures how much of the oracle is that rule, not predictive
skill.

## Why the rows were kept

Deleting the rows would not remove the circularity. It would only hide how large it is:
- Every learned engine reads `in_degree_centrality` and `qos_weight_in`, a QoS-weighted in-degree
  (`saag/prediction/data_preparation.py`, feature columns 5 and 11), and is trained on `I*`.
- `GAT-P+InDeg` uses `InDeg` as its prior.
- `Topo-QoS` computes betweenness over the same subscriber→publisher arcs.

Without the reference rows, a GNN fed in-degree and trained on the oracle would have headlined the
paper. Seven referee rounds asked for these baselines, and deleting them after the results were known
would be selective reporting.

## What changed in the paper

| Item | Now |
|---|---|
| `InDeg`, `Reach`, Pubs-raw, Reach-R1 | Reported next to Analytic-I* in a "Reference" block. They show ρ, ρ>0 and Overlap@K, and have no Δρ/Won/p against `Topo-QoS`. They are absent from the title, abstract, highlights, conclusion, predictor taxonomy and guidance table. |
| `GAT-P+InDeg` | Supplement only, because its prior is a reference. |
| Partial ρ(·, `I_dyn` \| `I*`) | Kept, as the bound on how much of a reference survives outside the oracle it restates. |
| Recall headline | Now `GAT-P-QoS`, with 80% of the true top-20% recovered in the top 40%. The references are grey dashed curves in Figure 6. |
| Practical guidance | For `I*`, run the oracle directly. For `I_comp`, use `Topo-QoS` or Degree-raw. For `I_dyn`, no predictor reaches the reference. For unfamiliar topologies, use the dependency-graph GAT, with caveats. |
| Headline | The registered primary contrast is null, and the hybrids beat centrality on 11 of 12 folds. The best predictor, `GAT-P-QoS` (0.748), approaches `InDeg` (0.764) but never exceeds it on the registered per-seed statistic. Its five-seed ensemble (0.772) is level with `InDeg`, and it does exceed the `Reach` reference (0.732). |
| Title | "Software-as-a-Graph: Benchmarking Centrality and Graph Learning for Pre-Deployment Cascade-Impact Ranking in Publish–Subscribe Systems" |

## Verification

`reproduce/reconcile_manuscript.py` gained `check_reference_demotion`. It asserts that:
- the reference rankers are absent from the title, abstract, highlights, conclusion,
  `tab:predictor_taxonomy` and `tab:guidance`;
- no `GAT-P+InDeg` row appears in the main text;
- each results table's reference block holds exactly the expected rows, and those rows appear nowhere
  else in the table.

The existing table checks cover the reference rows' numbers. The `tab:hybrid` check also asserts that
the reference rows' contrast cells read `---`. Mutation-tested:
- `InDeg` reinserted into `tab:guidance`, the title or the taxonomy is caught;
- `Reach` added to the abstract is caught;
- a `GAT-P+InDeg` row restored is caught;
- a Δρ restored in a reference row is caught.

## Not done

- A learner without the in-degree and `w_in` columns, which would show whether message passing alone
  approaches the reference.
- Scoring the learned engines' partial correlation with `I_dyn` beyond `I*`.
