# JSS experiments: protocols, commands and artifacts

This folder is the experiment companion to the *Journal of Systems and Software* paper
**"Software-as-a-Graph: Pre-Deployment Dependability Analysis of Publish–Subscribe Systems with
QoS-Aware Architecture Graphs and Hybrid Learning"**. The paper keeps the headline results and the
evidence each one needs. Each page here documents one experiment:
- the protocol and hyperparameters that the paper summarises in a sentence;
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

**Where the artifacts live.** `results/` is not tracked in git. The JSON artifacts named below ship
in the Zenodo replication package ([10.5281/zenodo.14922108](https://doi.org/10.5281/zenodo.14922108))
as a dated bundle `SaG_JSS_Results_<stamp>` with a `MANIFEST.json` of SHA-256 digests, commit hashes
and corpus provenance. Every `make` target below writes into `results/` when run from the repository
root.

## Index

| Paper | Experiment | Page | Command | Extended results |
|---|---|---|---|---|
| §7.1.1, Table 7 | LOSO ranking of single engines vs. baselines | [rq1-engines-loso.md](rq1-engines-loso.md) | `make -f reproduce/Makefile table4` | Supp. S23, S25 |
| §7.1.2, Table 8 | Hybrid-HGT and Hybrid-GAT | [rq1-hybrid.md](rq1-hybrid.md) | `make -f reproduce/Makefile rq-hybrid rq-hybrid-gat` | Supp. S23, S24 |
| §7.2, Table 9 | Capacity- and channel-matched typing × QoS control | [rq2-matched-control.md](rq2-matched-control.md) | `make -f reproduce/Makefile rq2-matched` | Supp. S16, S21, S26 |
| §7.3, Table 10 | Zero-shot transfer to five open-source system models | [rq3-zero-shot-transfer.md](rq3-zero-shot-transfer.md) | `python reproduce/realworld_zeroshot.py` | Supp. S7, S15, S27, S29 |
| §7.4, Table 11 | Analysis cost and comparison with direct simulation | [rq4-cost.md](rq4-cost.md) | `make -f reproduce/Makefile inference-latency` | Supp. S28 |
| §4.3, §8.2 | Oracles, label QoS content, convergent validity, parameter sensitivity | [oracles-and-sensitivity.md](oracles-and-sensitivity.md) | `make -f reproduce/Makefile convergent-validity` | Supp. S1–S4, S9 |
| §6.3, §8.2 | Registered plan, amendments, omnibus correction, repeatability | [repeatability-and-amendments.md](repeatability-and-amendments.md) | `make -f reproduce/Makefile omnibus` | Supp. S24 |

Section and table numbers are those of the compiled manuscript at the submission tag. The LaTeX
sources refer to them by label (`sec:rq1`, `tab:7`, …), so `manuscript.aux` is the authoritative
mapping if they drift.

## Predictor names

A name gives the architecture and then what distinguishes it:
- `-QoS` means QoS-weighted distances for `Topo`, and the 16-D QoS edge vector for a GNN.
- `-w` means a scalar QoS edge weight.
- `-S` means a small GAT (28k parameters).
- `-P` means the flow projection rather than the native multigraph.
- `Hybrid-X` means engine X corrected by the `Topo-QoS` prior.

Unsuffixed `GAT` and `GAT-QoS` are matched to HGT's parameter budget.

The registered plan and the result artifacts use earlier labels. The map below mirrors
`LEGACY_LABELS` in [`saag/evaluation/variant_registry.py`](../../../../saag/evaluation/variant_registry.py).
Internal variant ids never changed.

| Current | Earlier label | Variant id |
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
| RQ1: ranking accuracy of closed-form, learned and hybrid engines | RQ1, plus the hybrid Amendments 5–6 |
| RQ2: what learned engines need (typing vs. QoS channel) | RQ2, plus the QoS-ablation part of RQ3 |
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
