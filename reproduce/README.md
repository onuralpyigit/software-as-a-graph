# Reproducing Software-as-a-Graph (SaG)

> **Software-as-a-Graph: When Does Graph Learning Improve Cascade-Impact Ranking in
> Publish–Subscribe Systems?**
> Submitted to the *Journal of Systems and Software* Special Issue **VSI:AI4MSS** (AI Techniques for
> Performance, Reliability, and Sustainability of Modern Software Systems).
> The authoritative manuscript sources are in [`docs/research/jss/latex/`](../docs/research/jss/latex/);
> the per-experiment protocols, commands and artifacts are indexed in
> [`docs/research/jss/experiments/`](../docs/research/jss/experiments/README.md).

This directory contains the harnesses that reproduce the paper's results, tables and figures.
A Docker image is provided for exact environment replication.

---

## Hardware Requirements

| Configuration | Time estimate |
|---|---|
| CPU-only (8 cores, 32 GB RAM) | ~6–12 h (full, 5 seeds) |
| GPU (CUDA 11.8+, ≥8 GB VRAM) | ~1–2 h |
| Smoke-test (50 epochs, 2 seeds) | ~15–30 min CPU |
| Diagnostic & sensitivity sweeps (supplement) | seconds to minutes — pure graph computation & simulation |

### Measurement Hardware & Energy Estimation Baseline

The latency profiles and execution benchmarks reported in the paper were measured on the following hardware platform:
- **CPU:** 13th Gen Intel(R) Core(TM) i7-1370P (14 cores / 20 threads: 6 P-cores, 8 E-cores; Raptor Lake architecture)
- **Base Package Power (PBP / TDP):** 28.0 W nominal specification (vendor published rating)
- **RAM:** 32 GB LPDDR5
- **OS:** Linux x86_64

Energy consumption figures reported in the manuscript (Sections 6.4 and 7.4) represent theoretical upper-bound proxies calculated by `reproduce/energy_estimate.py` using this 28.0W TDP base. Hardware-level variations (e.g. DVFS, DRAM/disk power) are not captured, and physical meters (RAPL/external wattmeter) were not instrumented.

---

## Quick Start — Docker (recommended)

```bash
# 1. Build the image (≈5 min first run, cached afterwards)
docker build -t sag-jss -f reproduce/Dockerfile .

# 2. Full pipeline (~6-12 h)
docker run --rm -v $(pwd)/results:/workspace/results sag-jss

# 3. Smoke-test only (~15-30 min)
docker run --rm -v $(pwd)/results:/workspace/results \
    sag-jss make -C /workspace/reproduce smoke-test
```

---

## Quick Start — Local

### Prerequisites

```bash
python --version   # project requires >=3.9 (pyproject.toml); CI builds on 3.11
pip install -e ".[all]"   # installs from pyproject.toml: base + neo4j + gnn (PyTorch Geometric) + api extras
```

There is no separate `requirements.txt`; `pyproject.toml` at the repo root is the single source of
truth for dependencies (see its `[project.optional-dependencies]` table for the `neo4j`/`gnn`/`api`/
`dev` extras, or use `all` as above to install everything this package needs).

## Reproducing the paper

The experiment index, [`docs/research/jss/experiments/README.md`](../docs/research/jss/experiments/README.md),
maps every table and section of the paper to the command that produces it and to the supplement
section with its extended results. It is the source of truth for that mapping; this file covers setup.

### Step 1 — Go/no-go gate (~10 s)

```bash
make -f reproduce/Makefile block0
```

### Step 2 — Corpus and caches

```bash
make -f reproduce/Makefile scenarios   # regenerate the twelve synthetic scenarios (byte-identical)
make -f reproduce/Makefile cache       # rebuild output/loso_cache from data/scenarios
```

A stale `output/loso_cache/` silently outranks the datasets, so rebuild it after any change to the
generator.

### Step 3 — Experiments

| Research question | Commands |
|---|---|
| RQ1: ranking accuracy (Tables 5–6) | `make -f reproduce/Makefile table4 rq-hybrid rq-hybrid-gat rq-dependency-graph rq-referee-round7 rq-oracle-robust rq-rate-expansion` |
| RQ2: sources of predictive performance (Tables 7–8) | `make -f reproduce/Makefile rq2-matched rq-attribution rq-amendment14 rq-referee-round8` |
| RQ3: zero-shot transfer (Table 9) | `python reproduce/realworld_zeroshot.py` |
| RQ4: cost (Table 10) | `make -f reproduce/Makefile inference-latency rq-cost-reconcile` |
| Oracles and sensitivity (§4.3, §7.5) | `make -f reproduce/Makefile convergent-validity` |

### Step 4 — Figures

```bash
make -f reproduce/Makefile jss-figures   # manuscript figures, supplement figures, graphical abstract
```

### Step 5 — Reconcile and bundle

```bash
python reproduce/reconcile_manuscript.py   # every reported figure against the artifact that produced it
make -f reproduce/Makefile bundle          # reconcile, then cut the results bundle that ships with the paper
```

### Smoke test (~15–30 min)

```bash
make -f reproduce/Makefile smoke-test EPOCHS=50
```

Predictor names and the original labels used by the registered plan and the result artifacts are
mapped in the experiment index (section "Predictor names").

---

## Seed Lock

All experiments use seeds `[42, 123, 456, 789, 2024]` for reproducibility.
The Go/No-Go test (`make block0`) verifies determinism via `test_prediction_delta_is_deterministic`.

---

## Partial Replication (Selected Scenarios)

To reproduce results for a subset of scenarios:

```bash
python reproduce/main_table.py \
    --scenarios av_system iot_smart_city_system \
    --seeds 42,123 \
    --epochs 150 \
    --output results/partial_table.json

python reproduce/render_table.py \
    --table3 results/partial_table.json \
    --output-dir results/
```

---

## Reproducible GPU Rerun Protocol

`main_table.py`, `loso_all_variants.py`, and `kfold_all_variants.py` stamp every result JSON with a
`"provenance"` block (`reproduce/_provenance.py`: commit SHA, working-tree dirty flag, run config) —
this project has previously shipped a table that "reproduced from no commit"; the stamp exists so
that failure shows up in the artifact instead of only on the next rerun. Follow this sequence for a
rerun whose results can be trusted and diffed against the manuscript, e.g. on a Colab GPU runtime
(`notebooks/train_gnn_colab.ipynb`):

1. **Pin the commit.** `git status --porcelain` must be empty; record `git rev-parse HEAD`. A dirty
   tree makes every result's `provenance.dirty` come back `true`, which defeats the point.
2. **Regenerate the corpus.** `make -f reproduce/Makefile scenarios`, then verify with
   `pytest tests/test_scenario_corpus.py`.
3. **Rebuild the LOSO cache locally** (needs a live Neo4j — this step cannot run on Colab):
   `make -f reproduce/Makefile cache`.
4. **Package and upload the cache** for the GPU runtime to fetch:
   `tar -czf output/loso_cache.tar.gz -C output loso_cache`, then upload it to wherever the runtime's
   setup cell reads it from (e.g. Google Drive).
5. **Pin the remote clone to the same commit** — after cloning on the GPU runtime, run
   `git checkout <the SHA from step 1>` before installing dependencies or running anything.
6. **Run the experiments** (Block-0 gate → smoke test → `table3` → `table4` → `kfold`) on the GPU
   runtime.
7. **Bring the results back** — copy `results/` and `output/gnn_checkpoints/` from the GPU runtime
   back to a local machine (or durable storage).
8. **Reconcile.** `python reproduce/reconcile_manuscript.py --verbose` against the restored
   `results/` — each JSON's `provenance.commit` should match the step-1 SHA, and the tool's
   freshness/value checks compare those results against the numbers currently hardcoded in
   `docs/research/jss/latex/sections/sec7_results.tex`.
9. **Hand-edit the manuscript** only for the deltas `reconcile_manuscript.py` reports — table numbers
   in `sec7_results.tex` are hand-copied, not `\input`-ed from `results/*.tex`, so this stays a manual
   step. The tool does not check prose-only numbers, so also skim the prose around the tables it
   flags.

Durably archiving the resulting `results/` + `output/gnn_checkpoints/` bundle (e.g. to Zenodo) is a
manual step performed outside this repo — there is no in-repo tooling or git-tag convention for it.

---

## File Structure

```
reproduce/
├── Makefile                  — orchestration targets for the JSS experiments, tables and figures
├── Dockerfile                — exact environment (Python 3.11, PyG CPU)
├── README.md                 — this file
├── EXPERIMENTS.md            — technical notes on the harness internals and metrics
├── reconcile_manuscript.py   — checks every reported figure against its artifact
├── render_manuscript_md.py   — regenerates the Markdown rendering of the manuscript
└── *.py                      — one harness or renderer per experiment; see the experiment index
```

---

## Citation

```bibtex
@article{sag2026jss,
  author  = {Yigit, Ibrahim Onuralp and Buzluca, Feza},
  title   = {Software-as-a-Graph: When Does Graph Learning Improve Cascade-Impact Ranking in Publish--Subscribe Systems?},
  journal = {Journal of Systems and Software},
  note    = {Special Issue: AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems (VSI:AI4MSS). Under submission.},
  year    = {2026}
}
```
