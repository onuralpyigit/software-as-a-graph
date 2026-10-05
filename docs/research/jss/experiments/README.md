# JSS experiments: protocols, commands and artifacts

This folder is the experiment companion to the *Journal of Systems and Software* paper
**"Software-as-a-Graph: When Does Graph Learning Improve Cascade-Impact Ranking in
Publish–Subscribe Systems?"**. The paper reports the headline results and the evidence each one
needs. Each page here documents one experiment:
- the protocol and hyperparameters that the paper summarizes in a sentence;
- the command that reproduces it;
- the artifacts it writes;
- where its extended results live.

**Where the numbers live.** Numeric result tables are in the LaTeX manuscript
([`../latex/sections/`](../latex/sections/)) and supplement
([`../latex/supplementary.tex`](../latex/supplementary.tex)). There
[`reproduce/reconcile_manuscript.py`](../../../../reproduce/reconcile_manuscript.py) checks every
reported figure against the artifact that produced it. These pages deliberately do **not** copy those
tables, since a hand-copied table is exactly the kind of number that drifts. They quote only
abstract-level headline values, and point to the table that carries the full result.

**Where the artifacts live.** `results/` is not tracked in git, except the Amendment 7, 9 and 10
artifacts; later artifacts are tracked under `data/benchmarks/`. The JSON artifacts named below ship
in the Zenodo replication package ([10.5281/zenodo.23045204](https://doi.org/10.5281/zenodo.23045204))
as a dated bundle `SaG_JSS_Results_<stamp>` with a `MANIFEST.json` of SHA-256 digests, commit hashes
and corpus provenance. Every `make` target below writes into `results/` when run from the repository
root.

## Index

| Paper | Experiment | Page | Command | Extended results |
|---|---|---|---|---|
| §6.1, Table 5 | LOSO ranking of learned rankers against the training-free baseline | [rq1-engines-loso.md](rq1-engines-loso.md) | `make -f reproduce/Makefile table4` | Supp. S28, S35 |
| §6.1, Table 5 | Hybrid rankers (Hybrid-HGT, Hybrid-GAT) | [rq1-hybrid.md](rq1-hybrid.md) | `make -f reproduce/Makefile rq-hybrid rq-hybrid-gat` | Supp. S25 |
| §6.2 | Capacity- and channel-matched typing × QoS control | [rq2-matched-control.md](rq2-matched-control.md) | `make -f reproduce/Makefile rq2-matched` | Supp. S18, S23, S29, S30 |
| §6.2, §7.2 | Attribution controls: receptive field, QoS node columns vs. edge channel, feature-only GBM (Amendment 8, exploratory) | [rq2-attribution-controls.md](rq2-attribution-controls.md) | `make -f reproduce/Makefile rq-attribution` | Supp. S34 |
| §6.3, Table 9 | Zero-shot transfer to five open-source system models | [rq3-zero-shot-transfer.md](rq3-zero-shot-transfer.md) | `python reproduce/realworld_zeroshot.py` | Supp. S15, S31 |
| §6.4, Table 10 | Analysis cost against direct simulation | [rq4-cost.md](rq4-cost.md) | `make -f reproduce/Makefile inference-latency rq-cost-reconcile` | Supp. S32, S33 |
| §4.3, §7.5 | Simulation oracles, convergent validity, parameter sensitivity | [oracles-and-sensitivity.md](oracles-and-sensitivity.md) | `make -f reproduce/Makefile convergent-validity` | Supp. S4, S11 |
| §6.1, Table 5, Fig. 4 | Dependency counts and QoS-attribution controls (Amendment 7) | [amendment7-training-free.md](amendment7-training-free.md) | `python reproduce/training_free_suite.py all` | Supp. S37 |
| §6.1–6.3, Tables 5 and 9, §7.2 | Graph learning on the dependency graph vs. analytical references (Amendment 9, exploratory) | [amendment9-dependency-graph-learning.md](amendment9-dependency-graph-learning.md) | `make -f reproduce/Makefile rq-dependency-graph` | Supp. S38 |
| §3.3, §6.1 | Value of the dependency derivation: Rule 5 in transitive reach (Amendment 10, exploratory) | [amendment10-derivation.md](amendment10-derivation.md) | `python reproduce/training_free_suite.py derivation` | Supp. S38 |
| §6.1, Table 6 | Learned approximation of the queue-flow oracle (GBM on dependency features; Amendment 11) | — | `make -f reproduce/Makefile rq-oracle-robust` | Supp. S27 |
| §6.1, Tables 5–6, Fig. 5 | Referee analyses: raw-graph rankers, partial ρ beyond I*, learned rankers on all oracles, recall@k, latency (Amendment 12, exploratory) | [amendment12-referee.md](amendment12-referee.md) | `make -f reproduce/Makefile rq-referee-round7` | Supp. S39 |
| §4.4, §5.2, §6 | Dependency counts reported as references that restate I*, not as predictors (Amendment 13, reporting change, no new runs) | [amendment13-reference-demotion.md](amendment13-reference-demotion.md) | none | Supp. S27 |
| §6.1–6.2, Tables 6 and 8 | Degree-free and GIN learners, w_in-held 2×2, GNN trained on I_dyn, nested selection, full-population I_dyn (Amendment 14) | [amendment14-round8.md](amendment14-round8.md) | `make -f reproduce/Makefile rq-amendment14 rq-referee-round8` | Supp. S40 |
| §5.2, §6.1, Table 6 | Rate-weighted reference for I_dyn (Eq. 7) and input attribution of the learned approximation (Amendment 15, exploratory) | — | `make -f reproduce/Makefile rq-rate-expansion` | Supp. S41 |
| §3.3, §6.1–6.2, Table `tab:a17`, §7.5 | Oracle-aligned features removed, learners started from Eq. 7, node-order permutations (Amendments 17 and 17b) | [amendment17-round12.md](amendment17-round12.md) | `make -f reproduce/Makefile rq-amendment17 rq-amendment17b` | — |
| §4.3, §7.5 | Payload-aware queue-flow oracle `I_dyn-size` (Amendment 18): **registered, not run**; the round-13 revision was text-only, so no `I_dyn-size` number exists | — | `python reproduce/oracle_robust_ltr.py labels --payload-model size` (not executed) | — |
| §5.3 | Registered plan, amendments, omnibus correction, repeatability | [repeatability-and-amendments.md](repeatability-and-amendments.md) | `make -f reproduce/Makefile omnibus` | Supp. S27 |

Section, table and figure numbers are those of the compiled manuscript and supplement at submission.
The LaTeX sources refer to them by label (`sec:rq1`, `tab:hybrid`, …), so `manuscript.aux` and
`supplementary.aux` are the authoritative mapping if they drift.

## Predictor names

A name gives the architecture and then what distinguishes it:
- `-QoS` means QoS-weighted distances for `Topo`, and the 16-D QoS edge vector for a GNN.
- `-w` means a scalar QoS edge weight.
- `-S` means a small GAT (28k parameters).
- `-P` means the Application–Library `DEPENDS_ON` dependency graph rather than the native multigraph.
- `Hybrid-X` means model X corrected by the `Topo-QoS` prior.
- `→dyn` means trained on queue-flow (`I_dyn`) labels instead of `I*`.

Unsuffixed `GAT` and `GAT-QoS` are matched to HGT's parameter budget.

The registered plan and the result artifacts use the original labels. The map below mirrors
`LEGACY_LABELS` in [`saag/evaluation/variant_registry.py`](../../../../saag/evaluation/variant_registry.py).
Internal variant ids never changed.

| Current | Original label | Variant id |
|---|---|---|
| `GAT-S` / `GAT-S-w` | `GAT-N` / `GAT-N-QoS` | `gl_full` / `gl_full_qos` (`gl` / `gl_qos` under LOSO) |
| `GAT-S-P` / `GAT-S-P-w` | `GAT` / `GAT-QoS` (in-distribution only) | `gl` / `gl_qos` |
| `GAT` / `GAT-w` / `GAT-QoS` | `GAT-N-C` / `GAT-N-QoS-C` / `GAT-N-QoS16-C` | `gl_full_cap` / `gl_full_qos_cap` / `gl_full_qos16_cap` |
| `Hybrid-HGT` / `Hybrid-GAT` | `SaG-Hybrid` / `SaG-Hybrid-GAT` | `hgl_qos_prior` / `gl_qos16_prior` |

## Research-question numbering

The paper answers four research questions. The registered analysis plan
([`../PREREGISTRATION.md`](../PREREGISTRATION.md)) numbered five:

| Paper | Registered plan |
|---|---|
| RQ1: ranking accuracy of analytical, hybrid and learned rankers | RQ1, plus the hybrid Amendments 5–6 |
| RQ2: sources of predictive performance (representation, degree, typing, QoS, model family) | RQ2, plus the QoS-ablation part of RQ3 |
| RQ3: zero-shot transfer | RQ4 |
| RQ4: analysis cost | RQ5 |

The robustness part of the plan's RQ3 (sensitivity sweeps, oracle agreement) is reported in the
paper's threats to validity and in the supplement; see
[oracles-and-sensitivity.md](oracles-and-sensitivity.md).

## Common setup

```bash
pip install -e .                               # or: uv sync
make -f reproduce/Makefile block0              # go/no-go gate; run before anything else
make -f reproduce/Makefile cache               # rebuild output/loso_cache from data/scenarios
python reproduce/reconcile_manuscript.py -v    # check every reported figure against results/
```

The twelve synthetic scenarios regenerate byte-identically from `data/scenarios/scenario_*.yaml`
(`tests/test_scenario_corpus.py`). A stale `output/loso_cache/` silently outranks the datasets, so
rebuild it after any change to the generator. [`../../../../reproduce/README.md`](../../../../reproduce/README.md)
and [`../../../../reproduce/EXPERIMENTS.md`](../../../../reproduce/EXPERIMENTS.md) describe the
harness internals.
