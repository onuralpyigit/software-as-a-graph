# RQ3 — Zero-shot transfer to five open-source system models

**Paper:** §7.3, Table 10 (and the transfer columns of Table 8). **Extended results:** Supplement
S27 (bootstrap intervals and active stratum), S15 (PR-AUC, F1@τ, nDCG), S7 (explanation layer on
the same models), S29 (2-layer configuration).

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
- `realworld_zeroshot_v7.json` (`HGT-QoS`, Table 10);
- `realworld_zeroshot_*_cpu.json` (the CPU rows of Table 8).

## Headline result

Learned engines rank zero-shot at ρ = 0.760 (`HGT-QoS`) and 0.805 (`GAT-QoS`), against
0.511–0.526 for every training-free score. They roughly double top-K overlap. On the active stratum
(components with $I^* > 0$), every interval spans zero at five systems, so that comparison is
unresolved.

## Notes cut from the paper

- **Configuration sensitivity.** An earlier configuration used 2 layers and 150 epochs, chosen
  because these meshes are small ($|V_\text{app}| \le 41$). That choice appealed to a property of
  the test systems, so it is not reported as primary. It is uniformly slightly stronger and changes
  no conclusion (Supplement S29).
- **The 3–2 split.** $\rho_{>0}$ is positive on the three pub-sub-derived models and non-positive on
  the two RPC-derived ones.
  - Both RPC-derived models are encoded as pub-sub graphs and labelled by the same
    forward-reachability oracle, so the split cannot be attributed to call-tree semantics.
  - Testing that needs synchronous edges in the schema and a backward-propagating oracle.
- **Second-modeler check (open).** No second modeler has re-derived any model.
  [`reproduce/model_agreement.py`](../../../../reproduce/model_agreement.py) implements the
  re-modeling protocol: the second modeler gets the sources but not the original, and renames go
  only through an alias file. It also computes per-type entity and per-relation edge Jaccard, and is
  ready for when one or two systems are re-modelled.
