# JSS experiments: protocols, commands and artifacts

This folder is the experiment companion to the *Journal of Systems and Software* paper
**"Software-as-a-Graph: When Does Graph Learning Improve Cascade-Impact Ranking in
Publish–Subscribe Systems?"**. The paper reports the results and the evidence each one needs. These
pages record what the paper summarizes in a sentence:
- the protocol and hyperparameters;
- the command that reproduces each result;
- the artifacts it writes;
- the decision rules and their outcomes;
- where the extended results live in the supplement.

## Layout

- **One page per research question**, matching the paper's §6.1–6.4, plus one page for the oracles
  and one for the registration. Each RQ page states the current result; the amendments that produced
  it are linked, not repeated.
- **[`amendments/`](amendments/)**: one page per analysis-plan amendment from A7 on, in the order
  they were registered. Each page is the record of that amendment: question, arms, command,
  artifacts, decision-rule outcome, deviations and corrections. Amendments 1–6 have no page of their
  own: they are covered by the RQ pages and [repeatability-and-amendments.md](repeatability-and-amendments.md).

Every page opens with the same header: **Paper** (section, table and figure numbers plus their LaTeX
labels), **Supplement**, **Status** (confirmatory, registered secondary or exploratory, and whether
it was written before or after its results) and **Registration**.

## Index by paper section

| Paper | Page | Main commands |
|---|---|---|
| §6.1, Tables 5–6, Figs. 4–5 | [RQ1: ranking accuracy](rq1-ranking-accuracy.md) (registered primary, hybrids, references, three oracles) | `make -f reproduce/Makefile rq-hybrid rq-hybrid-gat rq-dependency-graph rq-referee-round8` |
| §6.2, Table 7 | [RQ2: sources of predictive performance](rq2-sources-of-performance.md) (matched 2×2 and every control family) | `make -f reproduce/Makefile rq2-matched` plus the amendment targets |
| §6.3, Table 8 | [RQ3: zero-shot transfer](rq3-zero-shot-transfer.md) | `python reproduce/realworld_zeroshot.py` |
| §6.4, Table 9 | [RQ4: cost](rq4-cost.md) | `make -f reproduce/Makefile rq-cost-reconcile` |
| §4.3–4.4, §7.4 | [Oracles and sensitivity](oracles-and-sensitivity.md) | `make -f reproduce/Makefile convergent-validity` |
| §5.3, §7.4 | [Registration, amendments, repeatability](repeatability-and-amendments.md) | `make -f reproduce/Makefile omnibus` |

## Amendment log

| Amendment | Status | Question | Page | Command | Supplement |
|---|---|---|---|---|---|
| A7 | registered secondary | Do dependency counts match the learned rankers? Is the `Topo-QoS` gain QoS content? | [a07](amendments/a07-training-free.md) | `python reproduce/training_free_suite.py all` | §S37 |
| A8 | exploratory (post hoc) | QoS node columns vs edge channel; gradient boosting vs GNN on the raw multigraph | [a08](amendments/a08-attribution-controls.md) | `make -f reproduce/Makefile rq-attribution` | §S27 |
| A9 | registered secondary | The same learners on the dependency graph | [a09](amendments/a09-dependency-graph-learning.md) | `make -f reproduce/Makefile rq-dependency-graph` | §S38 |
| A10 | registered secondary | What the derivation adds beyond raw-graph counts | [a10](amendments/a10-derivation.md) | `python reproduce/training_free_suite.py derivation` | §S38 |
| A11 | registered secondary | Full-population $I_\text{dyn}$; learned combination of dependency signals across oracles | [a11](amendments/a11-oracle-robust.md) | `make -f reproduce/Makefile rq-oracle-robust` | §S27, §S40 |
| A12 | mixed | Round-7 referee analyses: raw-graph rankers, partial ρ, recall, latency | [a12](amendments/a12-referee-round7.md) | `make -f reproduce/Makefile rq-referee-round7` | §S39 |
| A13 | reporting deviation (post hoc) | Counts reclassified as references; no run | [a13](amendments/a13-reference-demotion.md) | none | §S27 |
| A14 | registered secondary | Degree features, GIN, $w_\text{in}$-held 2×2, $I_\text{dyn}$-trained GNN, nested selection, like-for-like cost | [a14](amendments/a14-round8.md) | `make -f reproduce/Makefile rq-amendment14 rq-referee-round8` | §S40, §S42 |
| A15 | exploratory (post hoc) | Rate-weighted reference for $I_\text{dyn}$ (Eq. 7); input attribution | [a15](amendments/a15-rate-expansion.md) | `make -f reproduce/Makefile rq-rate-expansion` | §S41 |
| A16 | registered secondary | Reverse-edge direction control; corrected-prior hybrids | [a16](amendments/a16-direction-control.md) | `make -f reproduce/Makefile rq-amendment16` | §S42 |
| A17, 17b | registered secondary | Oracle-aligned features removed; learning on top of Eq. 7; node-order permutation | [a17](amendments/a17-round12.md) | `make -f reproduce/Makefile rq-amendment17 rq-amendment17b` | §S42 |
| A18 | registered, **not run** | Payload-aware queue-flow oracle | [a18](amendments/a18-payload-oracle.md) | not executed | — |
| A19 | registered secondary; **on branch `jss-revision-round14`** | Sum aggregation on the raw multigraph, rate-fed GNNs, tie-aware loss, learning curve | [a19](amendments/a19-round14.md) | on that branch only | on that branch |

## Conventions

**Where the numbers live.** The full result tables are in the LaTeX manuscript
([`../latex/sections/`](../latex/sections/)) and supplement
([`../latex/supplementary.tex`](../latex/supplementary.tex)).
[`reproduce/reconcile_manuscript.py`](../../../../reproduce/reconcile_manuscript.py) checks every
table figure there against the artifact that produced it. These pages quote only headline values and
the contrasts a decision rule reads, and point to the table that carries the rest. The reconciler
does not check these pages, so each quoted value was cross-checked by hand against the LaTeX
sources, `PREREGISTRATION.md` or the named artifact. Where a page and the paper disagree, the paper
and its artifact are authoritative.

**Section and table numbers.** Numbers are those of the compiled manuscript and supplement on
`main` (`latex/manuscript.aux`, `latex/supplementary.aux`, checked 2026-10-07). Each is given with
its LaTeX label (`tab:hybrid`, `supp:controls`, …), because labels survive renumbering. To re-check
after a rebuild:

```bash
grep -oE '\\newlabel\{(sec|tab|fig|supp)[^}@]*\}\{\{[^}]*\}' docs/research/jss/latex/manuscript.aux docs/research/jss/latex/supplementary.aux
```

The supplement numbers sections and tables with the same S prefix. These pages write sections as
"§S41" and tables as "Table S66".

**Where the artifacts live.** Most of `results/` is not tracked in git. The Amendment 7, 9 and 10
artifacts are tracked (see their pages), and later artifacts are tracked under `data/benchmarks/`.
Every JSON artifact named here ships in the Zenodo replication package
([10.5281/zenodo.23045204](https://doi.org/10.5281/zenodo.23045204)) as a dated bundle
`SaG_JSS_Results_<stamp>`, with a `MANIFEST.json` of SHA-256 digests, commit hashes and corpus
provenance. Every `make` target writes into `results/` or `data/benchmarks/` when run from the
repository root.

**The published link.** The manuscript's experiment-pages URL (`\sagexperimentsurl`) is pinned to
commit `56d9bff8`, so it shows these pages as they were at that revision.

## Predictor names

A name gives the architecture and then what distinguishes it:
- `-QoS` means QoS-weighted distances for `Topo`. For a GNN it means the 16-D QoS edge vector plus
  three QoS node columns.
- `-w` means a scalar QoS edge weight.
- `-S` means a small GAT (28k parameters).
- `-P` means the model reads the Application–Library `DEPENDS_ON` dependency graph rather than the
  raw multigraph.
- `Hybrid-X` means model X corrected by the `Topo-QoS` prior.
- `→dyn` means trained on queue-flow (`I_dyn`) labels instead of `I*`.

Control suffixes:
- `-R`: every raw edge also reversed;
- `-deg`, `-min`, `-const`: feature removal;
- `-AP`: corrected prior;
- `-perm`: permuted node order;
- `-U`: no reverse pass;
- `+InDeg`: `InDeg` prior.

Unsuffixed `GAT` and `GAT-QoS` are matched to HGT's parameter budget.

The registered plan and the early result artifacts use the original labels. The map below mirrors
`LEGACY_LABELS` in [`saag/evaluation/variant_registry.py`](../../../../saag/evaluation/variant_registry.py)
and Table S32 (`tab:supp-names`). Internal variant ids never changed.

| Current | Original label | Variant id |
|---|---|---|
| `GAT-S` / `GAT-S-w` | `GAT-N` / `GAT-N-QoS` | `gl_full` / `gl_full_qos` (`gl` / `gl_qos` under LOSO) |
| `GAT-S-P` / `GAT-S-P-w` | `GAT` / `GAT-QoS` (in-distribution only) | `gl` / `gl_qos` |
| `GAT` / `GAT-w` / `GAT-QoS` | `GAT-N-C` / `GAT-N-QoS-C` / `GAT-N-QoS16-C` | `gl_full_cap` / `gl_full_qos_cap` / `gl_full_qos16_cap` |
| `Hybrid-HGT` / `Hybrid-GAT` | `SaG-Hybrid` / `SaG-Hybrid-GAT` | `hgl_qos_prior` / `gl_qos16_prior` |

Arms added after the plan, which were registered under their current names:

| Name | Variant id | Amendment |
|---|---|---|
| `GAT-P` / `GAT-P-QoS` / `HGT-P-QoS` | `gl_proj_cap` / `gl_proj_qos16_cap` / `hgl_proj_qos` | A9 |
| `GAT-P+InDeg` (registered as Hybrid-GAT-P) | `gl_proj_qos16_indeg_prior` | A9 |
| `HGT-QoS-U` | `hgl_qos_uni` | A2 |
| `GAT-QoS-nf`, `GBM-Feat`, `GBM-Feat-QoS` | `gl_full_qos16_nfmask`, `tab_gbm`, `tab_gbm_qos` | A8 |
| `GBM-P-QoS→dyn` | `gbm_dep_qos_dyn` | A11 |
| `GAT-QoS-R`, Hybrid-GAT-AP, Hybrid-HGT-AP | `gl_full_qos16_cap_rev`, `gl_qos16_prior_ap`, `hgl_qos_prior_ap` | A16 |

## Research-question numbering

The paper answers four research questions. The registered plan
([`../PREREGISTRATION.md`](../PREREGISTRATION.md)) numbered five:

| Paper | Registered plan |
|---|---|
| RQ1: ranking accuracy of analytical, hybrid and learned rankers | RQ1, plus Amendments 5–6 |
| RQ2: sources of predictive performance | RQ2, plus the QoS-ablation part of RQ3 |
| RQ3: zero-shot transfer | RQ4 |
| RQ4: cost | RQ5 |

The robustness part of the plan's RQ3 (sensitivity sweeps, oracle agreement) is reported in the
paper's threats to validity and in the supplement; see
[oracles-and-sensitivity.md](oracles-and-sensitivity.md).

## Common setup

```bash
pip install -e .                               # or: uv sync
make -f reproduce/Makefile block0              # go/no-go gate; run before anything else
make -f reproduce/Makefile cache               # rebuild output/loso_cache from data/scenarios
python reproduce/reconcile_manuscript.py -v    # check every reported table figure against its artifact
```

The twelve synthetic scenarios regenerate byte-identically from `data/scenarios/scenario_*.yaml`
(`tests/test_scenario_corpus.py`). A stale `output/loso_cache/` silently outranks the datasets, so
rebuild it after any change to the generator. Run LOSO sweeps from the main checkout, or export
`LOSO_CACHE_DIR`. Run them from a clean tree, because artifacts record `dirty` and the reconciler
refuses dirty artifacts. [`../../../../reproduce/README.md`](../../../../reproduce/README.md) and
[`../../../../reproduce/EXPERIMENTS.md`](../../../../reproduce/EXPERIMENTS.md) describe the harness
internals.
