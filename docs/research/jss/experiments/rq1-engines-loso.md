# RQ1 — Single engines under leave-one-scenario-out (LOSO)

> **Amendment 7 update.** Training-free dependency counts on the same projection outrank every engine on this page: InDeg reaches 0.764 and Reach 0.732. The Topo → Topo-QoS gain (0.349 → 0.553) is attributed to the projection substrate, not to QoS content: the registered Topo read the analysis stage's app-layer betweenness, and unweighted betweenness on the projection scores 0.591. See [amendment7-training-free.md](amendment7-training-free.md).

**Paper:** §7.1, Table 7 (CPU runs of the six engines); the registered GPU sweep described here is Supplement Table S29 (Section S30). **Extended results:** Supplement S25 (active stratum), S23 (per-fold,
CPU re-runs), S22 (articulation term), S17 (in-distribution per-scenario).

## Question

How accurately do SaG's closed-form engine (`Topo-QoS`) and learned engines (`HGT-QoS`, `GAT-S-w`)
rank Applications by simulated cascade impact $I^*(v)$ on architectures they have never seen,
compared with unweighted centrality (`Topo`) and untyped GNNs?

## Protocol

- **Folds.** There are twelve synthetic scenarios. Each fold trains on eleven and tests zero-shot on
  the twelfth.
- **Evaluation set.** Every predictor is scored on the same Application node set (26–300 nodes per
  fold), with $K = \mathrm{round}(0.2\,|V_\text{app}|)$, so $K$ runs from 5 to 60.
- **Seeds.** Five seeds $\{42, 123, 456, 789, 2024\}$. The fold score is the mean over seeds.
- **Learned arms.** All learned arms are trained on the native typed multigraph, with the same
  training set, depth (3 layers) and inner-split early stopping (patience 30, up to 300 epochs).
  - HGT uses $D = 64$, $H = 4$ heads, dropout 0.10.
  - Optimizer: AdamW, lr $3\times10^{-4}$, weight decay $10^{-4}$.
  - Schedule: cosine warm restarts with $T_0 = 75$, $T_\text{mult} = 2$, $\eta_\text{min} = 3\times10^{-6}$.
  - Loss: MSE + 0.5·aux MSE + 0.3·ListMLE + 0.1·pairwise margin, with $\gamma = 0.05$.
  - None of these were tuned on any evaluation split.
- **Closed-form scores.** Both are computed on the Application–Library `DEPENDS_ON` projection
  (Rules 1 and 5) as $0.6\cdot\text{BT} + 0.4\cdot\text{AP}$.
  - The AP term reads zero in the evaluated implementation, so the reported scores are pure
    betweenness and pure QoS-weighted betweenness.
  - Restoring AP lowers both scores (Supplement S22).
  - The Rule-5 edge on this projection carries the graph's median topic weight, so the closed-form
    scores consume no code metrics.
- **Statistics.** Paired two-sided Wilcoxon over folds and a bootstrap 95% CI ($B = 2000$) over
  folds.
  - The registered confirmatory family is `HGT-QoS` vs `Topo-QoS` and `HGT` vs `Topo-QoS`,
    Holm-corrected across the two.
  - Folds share 10 of 11 training scenarios, so every $p$ is nominal.

## Reproduce

```bash
make -f reproduce/Makefile cache
make -f reproduce/Makefile table4          # loso_all_variants.json + loso_significance.json
```

Supplement Table S29 is the v5 GPU sweep: artifacts `loso_all_variants_v5.json` and
`loso_significance_v5.json`. Learned cells move across devices and code revisions (see
[repeatability-and-amendments.md](repeatability-and-amendments.md)), so compare new runs within
one sweep, not against the published cells.

## Headline result

`Topo-QoS` improves on `Topo` on all twelve folds (ρ 0.553 vs 0.349). `HGT-QoS` has the highest
single-engine mean (0.638), but the registered contrast against `Topo-QoS` is not significant.

## Notes cut from the paper

- **Label-noise ceiling.** Re-running the oracle across seeds gives a test–retest ρ of 0.811–1.000
  (median 0.982).
  - Top-K sets are noisier: cross-seed Jaccard has median 0.847 and falls to 0.370 on Logistics
    Fleet.
  - This is why Overlap@K margins are less stable than ρ margins.
  - The least reproducible fold (Microservices) is not one the learned engine loses.
  - Artifact: `label_stability.json` (`reproduce/label_stability_check.py`).
- **In-distribution results** (60/20/20 node splits) are in Supplement S17. They are not compared
  across model families, because `GAT-S-P`/`GAT-S-P-w` read the flow projection while HGT reads the native
  multigraph (Supplement S29).
- **Where the engines differ.** `HGT-QoS` wins most on the folds where `Topo-QoS` is weakest
  (Microservices, ATM) and loses on its strongest (Enterprise, Telecom RAN). Neither graph size,
  density nor prediction dispersion predicts the winner in advance.

## Where learned engines help: engine regimes

Manuscript §8.2 (Table 12) and the supplement's Engine Regimes section. The analysis is post hoc and
exploratory and trains nothing.

- **Command:** `make -f reproduce/Makefile jss-regimes`, which runs `reproduce/engine_regimes.py`.
- **Artifact:** `engine_regimes.json`, checked by `reconcile_manuscript.py::check_engine_regimes`.

Each row below is a hypothesis for future work, not a finding. Across 96 descriptor–gain correlations
over the twelve folds, none survives Benjamini–Hochberg correction (smallest q = 0.32).

| Factor | Candidate mechanism | Evidence on this corpus |
|---|---|---|
| Closed-form fit | Learned engines are steady across architectures; `Topo-QoS` is not, so the winner is set by how well centrality fits | By `Topo-QoS` tercile: `HGT-QoS` 0.604 / 0.589 / 0.672 vs `Topo-QoS` 0.324 / 0.561 / 0.775 |
| QoS inputs | Declared coupling weights tell the model how strongly components couple | +0.073 at matched capacity, carried by the 3 QoS node columns, not the edge channel ([rq2-attribution-controls.md](rq2-attribution-controls.md)) |
| Message passing | None found: the accuracy comes from graph-derived node features | `GBM-Feat` (no graph) 0.642 ≈ `GAT-QoS` 0.635; no message reaches an Application in the GATs; `HGT-QoS-U` 0.632 vs `HGT-QoS` 0.622 |
| Topology and symmetry | Symmetric stars create betweenness ties for closed-form scores | Learned engines win all 4 weakest-centrality folds; EdgeX closed-form Overlap@K = 0 |
| Scale | Learned gain over `Topo-QoS` shrinks with size, while the hybrid correction grows | Spearman −0.48 (apps), −0.62 (libraries); hybrid +0.43; accuracy is untested above 300 apps |
| Typing × scale | Typed parameters may stand in for undeclared QoS in large multi-broker systems | Without QoS, HGT − GAT correlates +0.63 / +0.66 / +0.77 with apps / topics / brokers; +0.03 with QoS |
| Receptive field | **Refuted** as an explanation | HGT-QoS RF share (35–59%) vs per-fold ρ: +0.12; vs gain over `Topo-QoS`: −0.06 |
| Original system's paradigm | RPC failure semantics (synchronous calls) are not modelled | Pure learned ρ>0 0.18–0.83 on the 3 pub-sub-derived models; −0.19 to +0.16 on the 2 RPC-derived ones |
| Inert-node base rates | Zero-impact components are part of what full-population ρ rewards | 21–52% of Applications carry $I^* = 0$; $\rho_{>0}/\rho \approx$ 49–56% (Supplement S25) |
