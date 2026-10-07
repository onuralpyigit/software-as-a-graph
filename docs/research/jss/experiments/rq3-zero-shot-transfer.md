# RQ3 — Zero-shot transfer to five open-source system models

**Paper:** §6.3 (`sec:rq3`), Table 9 (`tab:system_models_transfer`); corpus in §5.1 (`sec:6.1`); the
single-modeler threat in §7.4 (`sec:threats`).
**Supplement:** §S32 (`supp:transfer-active`: bootstrap intervals and active stratum), §S18
(`supp:identification`: PR-AUC, F1@τ, nDCG), §S10 (`supp:rq4`: explanation layer on the same models),
§S35 (`supp:taxonomy`: the 2-layer configuration), Table S77 (`tab:supp-moved-systems`: unweighted
`Topo`).
**Status:** descriptive. Five systems are too few for inference; the intervals are percentile
bootstraps over systems. This is the plan's RQ4.
**Registration:** [`../PREREGISTRATION.md`](../PREREGISTRATION.md), the plan (RQ4) and Amendment 4
(the common 3-layer, 300-epoch protocol).

## The five system models

The five models are hand-authored typed multigraphs, written by one author from public documentation. They are not
extracted from manifests. Loader: [`saag/adapters/realworld_adapter.py`](../../../../saag/adapters/realworld_adapter.py);
data: `data/scenarios/realworld_*.json`.

| Model | Original paradigm | Notes |
|---|---|---|
| Autoware.universe (ROS 2) | pub-sub | |
| EdgeX Foundry | pub-sub | Symmetric adapter-to-broker stars; betweenness ties collapse closed-form triage |
| Home Assistant | pub-sub | |
| Online Boutique (pub-sub model) | gRPC | Modelled as a 22-application pub-sub mesh with four brokers; the original has about eleven gRPC services and no broker |
| Train-Ticket booking mesh | RPC | The service-discovery server is modelled as a broker |

No model contains a synchronous call edge. Brokers, QoS profiles, code metrics and host
specifications are partly assumed. Where a system declares no QoS manifest, standard defaults
(ROS 2 Best-Effort/Reliable, MQTT QoS 0/1) are applied uniformly to every predictor.

## Protocol

Train on all twelve synthetic scenarios, then score each system zero-shot against $I^*(v)$ on its
Applications. No system graph contributes gradients or checkpoint selection. The primary protocol is
3 layers and 300 epochs, identical to every LOSO result, over five seeds.

## Reproduce

```bash
# Real-world cache goes in its own directory: output/loso_cache is read as the LOSO fold list.
CACHE_DIR=output/realworld_cache bash scripts/populate_loso_cache.sh \
    realworld_autoware_ros2 realworld_cloud_microservices \
    realworld_trainticket realworld_homeassistant realworld_edgex

# The script defaults to 2 layers / 150 epochs; the paper's primary protocol is 3 / 300.
PYTHONPATH=. python reproduce/realworld_zeroshot.py --variant hgl_qos --layers 3 --epochs 300
```

`--variant` also accepts `hgl`, `hgl_qos_prior`, `gl_full_qos16_cap` and `gl_qos16_prior` (see
`--help`). The published artifacts are:
- `realworld_zeroshot_{hgl_qos,hgl_qos_prior,gl_full_qos16_cap,gl_qos16_prior}_cpu.json`: the
  `HGT-QoS`, Hybrid-HGT, `GAT-QoS` and Hybrid-GAT rows of Table 9, and the `Topo-QoS` row (from the
  `hgl_qos` artifact's bootstrap block);
- `realworld_zeroshot_gl_proj_qos16_cap_dependency_graph.json`: the `GAT-P-QoS` row
  ([A9](amendments/a09-dependency-graph-learning.md));
- `tf_baselines.json`: the `InDeg` and `Reach` reference rows ([A7](amendments/a07-training-free.md)).

Zero-shot results of the control arms are on their amendment pages: [A8](amendments/a08-attribution-controls.md)
(`GAT`, `GBM-Feat`), [A14](amendments/a14-round8.md), [A16](amendments/a16-direction-control.md)
(`GAT-QoS-R`, AP hybrids) and [A17](amendments/a17-round12.md).

## Headline result

- **Learned rankers transfer better than the training-free baseline.** `GAT-P-QoS` reaches ρ = 0.806
  and `GAT-QoS` 0.805, against 0.526 for `Topo-QoS`.
- **The references rank higher still.** `Reach` reaches 0.938 and `InDeg` 0.863. The models are
  structurally easy for `Reach`: 51% of their Applications are inert, against 31% in the folds, and
  their fan-in is more concentrated (Gini 0.65 vs 0.50).
- **Active stratum.** On components with $I^* > 0$, every predictor is weak: learned rankers score
  0.185–0.342 and the baseline is negative. `Reach` keeps 0.871.
- **The baseline prior costs transfer.** Both hybrids fall below their base learners (0.695 and 0.662
  against 0.760 and 0.805), and the same holds with the corrected prior (0.702 and 0.668).

## Notes

- **Configuration sensitivity.** An earlier configuration used 2 layers and 150 epochs, chosen
  because these meshes are small ($|V_\text{app}| \le 41$). That choice appealed to a property of
  the test systems, so it is not reported as primary. It is uniformly slightly stronger and changes
  no conclusion (§S35).
- **The 3–2 split.** The raw-multigraph learned rankers lead on the three models of originally
  pub-sub systems. On the two originally RPC systems, their $\rho_{>0}$ is only −0.19 to +0.16
  (§S37, Table S39 `tab:supp-regimes-zs`).
  - Both RPC-derived models are encoded as pub-sub graphs and labelled by the same
    forward-reachability oracle, so the split cannot be attributed to call-tree semantics.
  - Testing that needs synchronous edges in the schema and a backward-propagating oracle.
- **Second-modeler check (open).** No second modeler has re-derived any model (§7.4).
  [`reproduce/model_agreement.py`](../../../../reproduce/model_agreement.py) implements the
  re-modeling protocol: the second modeler gets the sources but not the original, and renames go
  only through an alias file. It also computes per-type entity and per-relation edge Jaccard, and is
  ready for when one or two systems are re-modelled.
