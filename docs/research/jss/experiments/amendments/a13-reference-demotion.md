# Amendment 13: dependency counts reclassified as references

**Paper:** §1, §3.3 (Remark 1), §4.4 (`sec:4.4`, the reference criterion), §5.2 (`sec:6.2`,
"References"), §6.1–6.3 (the reference blocks of Tables 5, 6 and 8: `tab:hybrid`,
`tab:independent_oracles`, `tab:system_models_transfer`), §7.3 (`tab:guidance`), §7.4.
**Supplement:** §S27 (row A13); `GAT-P+InDeg` is reported in §S38.
**Status:** reporting deviation, written *after* all results existed. No arm was run, re-run, dropped
or added, and no number changed.
**Registration:** [`../../PREREGISTRATION.md`](../../PREREGISTRATION.md), Amendment 13 (commit
`18f44f44`).
**Run:** nothing new. The figures are re-rendered from existing artifacts:
`PYTHONPATH=. python reproduce/render_headline_figure.py`,
`PYTHONPATH=. python reproduce/render_recall_figure.py` and
`PYTHONPATH=. python reproduce/render_graphical_abstract.py`.

## Why

The authors decided not to use `InDeg` and `Reach` as predictors, because they are circular with the
labelling simulator:
- By Proposition 1 (now Remark 1 in §3.3), the first propagation wave of the primary oracle `I*` reaches exactly the
  Applications that `InDeg` counts.
- The oracle's first-order expansion (Analytic-I*) reaches ρ = 0.808.
- `Reach` counts the transitive dependents a reachability cascade can visit, and it keeps no `I_dyn`
  signal beyond `I*` (partial ρ 0.058, CI includes 0; Amendment 12, R2, on the n = 30 sample). On the
  full population (Amendment 14) it keeps 0.117 [0.055, 0.184], which excludes zero but is the
  smallest of the four references in Table 6.

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

## What changed in the paper (at the time of the amendment)

| Item | Now |
|---|---|
| `InDeg`, `Reach`, Pubs-raw, Reach-R1 | Reported next to Analytic-I* in a "Reference" block. They show ρ, ρ>0 and Overlap@K, and have no Δρ/Won/p against `Topo-QoS`. They are absent from the title, abstract, highlights, conclusion, predictor taxonomy and guidance table. |
| `GAT-P+InDeg` | Supplement only, because its prior is a reference. |
| Partial ρ(·, `I_dyn` \| `I*`) | Kept, as the bound on how much of a reference survives outside the oracle it restates. |
| Recall headline | Now `GAT-P-QoS`, with 80% of the true top-20% recovered in the top 40%. The references are grey dashed curves in Figure 6. |
| Practical guidance | For `I*`, run the oracle directly. For `I_comp`, use `Topo-QoS` or Degree-raw. For `I_dyn`, no predictor reaches the reference. For unfamiliar topologies, use the dependency-graph GAT, with caveats. |
| Headline | The registered primary contrast is null, and the hybrids beat centrality on 11 of 12 folds. The best predictor, `GAT-P-QoS` (0.748), approaches `InDeg` (0.764) but never exceeds it on the registered per-seed statistic. Its five-seed ensemble (0.772) is level with `InDeg`, and it does exceed the `Reach` reference (0.732). |
| Title | "Software-as-a-Graph: Benchmarking Centrality and Graph Learning for Pre-Deployment Cascade-Impact Ranking in Publish–Subscribe Systems" |

**Since then.** Later revisions kept the reclassification but changed several rows above:
- **Title.** Now "Software-as-a-Graph: When Does Graph Learning Improve Cascade-Impact Ranking in
  Publish–Subscribe Systems?".
- **Reference criterion.** It is now formal (§4.4), and the rate-weighted expansion of $I_\text{dyn}$
  (Eq. 7, [A15](a15-rate-expansion.md)) joined the references. For $I_\text{dyn}$, `tab:guidance`
  now points to Eq. 7.
- **"Never exceeds".** Withdrawn by Amendment 14 (F3b). `GAT-P-QoS` is "not significantly
  different" from `InDeg`, and equivalence within ±0.05 is not established.
- **Recall headline.** It now lives in §6.1 and Figure 5.

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

## Left open at the time, since done

- **A learner without the in-degree and `w_in` columns.** Run by Amendment 14 (F1, F2) and
  Amendment 17 (F11).
- **The learned rankers' partial correlation with `I_dyn` beyond `I*`.** Reported in Table 6 since
  Amendment 14.
