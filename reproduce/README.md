# Reproducing HGL-QoS

> **Heterogeneous Graph Learning for Cascade Impact Prediction in Distributed Publish-Subscribe Middleware**
> Originally submitted to ACM Middleware 2026 (rejected); revised and extended for submission to the
> *Journal of Systems and Software* Special Issue **VSI:AI4MSS** (AI Techniques for Performance,
> Reliability, and Sustainability of Modern Software Systems). See
> `docs/research/jss/si_middleware_extension.md` for the paper text this package reproduces.

This directory contains everything needed to reproduce the paper's results from scratch.
A Docker image is provided for exact environment replication.

---

## Hardware Requirements

| Configuration | Time estimate |
|---|---|
| CPU-only (8 cores, 32 GB RAM) | ~6–12 h (full, 5 seeds) |
| GPU (CUDA 11.8+, ≥8 GB VRAM) | ~1–2 h |
| Smoke-test (50 epochs, 2 seeds) | ~15–30 min CPU |
| Reversed-projection ablation / hardening-budget analysis (Tables 8–9) | seconds — no GNN training, pure graph computation |

---

## Quick Start — Docker (recommended)

```bash
# 1. Build the image (≈5 min first run, cached afterwards)
docker build -t qhgl-mw26 -f reproduce/Dockerfile .

# 2. Full pipeline (~6-12 h)
docker run --rm -v $(pwd)/results:/workspace/results qhgl-mw26

# 3. Smoke-test only (~15-30 min)
docker run --rm -v $(pwd)/results:/workspace/results \
    qhgl-mw26 make -C /workspace/reproduce smoke-test
```

---

## Quick Start — Local

### Prerequisites

```bash
python --version   # requires 3.10 or 3.11
pip install -e ".[all]"   # installs from pyproject.toml: base + neo4j + gnn (PyTorch Geometric) + api extras
```

There is no separate `requirements.txt`; `pyproject.toml` at the repo root is the single source of
truth for dependencies (see its `[project.optional-dependencies]` table for the `neo4j`/`gnn`/`api`/
`dev` extras, or use `all` as above to install everything this package needs).

### Step 1 — W1 Gate (sanity check, ~10 s)

```bash
make -f reproduce/Makefile block0
# Expected: 26/26 tests pass
```

### Step 2 — Main results table (~2–6 h CPU)

```bash
make -f reproduce/Makefile table3
# Output: results/table3_main_results.tex  /  .csv  /  .md
```

> **Table-number mapping.** This harness's internal name "Table 3" predates the JSS revision. In
> the current paper (`si_middleware_extension.md`), this output corresponds to **Table 4** (Global
> Ranking Performance) and **Table 5** (Identification and Regression Metrics) — table numbers
> shifted when Table 2b/3 (scenario characterization) were inserted earlier in the paper during the
> JSS extension. The underlying evaluation design is unchanged: 7 scenarios × 6 variants × 5 seeds
> = 210 cells (140 GNN + 70 structural), matching the paper's Table 1 factorial design exactly.

### Step 3 — Table 4 → paper's Table 7 (LOSO inductive, ~3–8 h CPU)

```bash
make -f reproduce/Makefile table4
# Output: results/table4_loso_results.tex  /  .md
```

> Same table-number caveat as Step 2: this harness's "Table 4" is the paper's **Table 7**
> (Leave-One-Scenario-Out Cross-Validation Results).

### Step 4 — Figures (run after Tables 3+4)

```bash
make -f reproduce/Makefile figure4   # Stratified ρ (instantaneous, reads JSON)
make -f reproduce/Makefile figure5   # ATM attention subgraph (~10 min)
```

### Step 5 — Reversed-Projection Ablation & Hardening-Budget Analysis (paper's Tables 8–9, JSS revision only)

These two experiments were added during the JSS revision and do not require GNN training — they
reuse the same `FaultInjector` ground-truth engine as every other number in the paper, run in
seconds:

```bash
python reproduce/reversed_projection_ablation.py
# Output: output/reversed_projection_ablation.json — feeds paper's Table 8

python reproduce/hardening_budget.py
# Output: output/hardening_budget_experiment.json — feeds paper's Table 9
# (imports build_projection from reversed_projection_ablation.py — run that one first,
#  or just run both from this directory; Python adds each script's own directory to sys.path)
```

### All at once

```bash
make -f reproduce/Makefile all EPOCHS=300 SEEDS=42,123,456,789,2024
```

### Smoke-test (fast sanity check, ~15–30 min)

```bash
make -f reproduce/Makefile smoke-test EPOCHS=50
```

---

## Expected Outputs

| File | Content |
|---|---|
| `results/table3_main_results.tex` | LaTeX table — Spearman ρ 7×6 (paper Table 4) |
| `results/table3_main_results.csv` | CSV version for Excel/R |
| `results/table4_loso_results.tex` | LaTeX table — LOSO Δρ (paper Table 7) |
| `results/figure4_stratified_rho.pdf` | Figure — per-node-type ρ |
| `output/atm_case_study/attention_subgraph.pdf` | Figure — HGT attention (ATM running example) |
| `results/loso_all_variants.json` | Raw LOSO Δρ data |
| `output/reversed_projection_ablation.json` | Paper Table 8 — reversed-projection ablation |
| `output/hardening_budget_experiment.json` | Paper Table 9 — hardening-budget risk-mass coverage |

---

## Architecture Variants (Table columns)

These identifiers are used directly by `middleware26_main_table.py`, `loso_all_variants.py`, and the
two Table 8/9 scripts, and match the paper's Table 1 naming exactly (unlike some other CLI entry
points in this repo, e.g. `cli/train_graph.py`, which predate the paper and use a different internal
naming scheme — see that file's `--variant` help text for the mapping if you need it):

| Variant flag | Description |
|---|---|
| `hgl_qos` | **HGL-QoS (Proposed)** — Heterogeneous GAT with 7-d edge features |
| `hgl` | **HGL** — Heterogeneous GAT with QoS attributes masked |
| `gl_qos` | **GL-QoS** — Homogeneous GAT with scalar QoS weight per edge |
| `gl` | **GL** — Homogeneous GAT with no edge weighting |
| `topo_qos` | **Topo-QoS** — QoS-weighted structural centrality baseline |
| `topo_baseline` | **Topo-BL** — Unweighted structural centrality baseline |

---

## Seed Lock

All experiments use seeds `[42, 123, 456, 789, 2024]` for reproducibility.
The Go/No-Go test (`make block0`) verifies determinism via `test_prediction_delta_is_deterministic`.

---

## Partial Replication (Selected Scenarios)

To reproduce results for a subset of scenarios:

```bash
python reproduce/middleware26_main_table.py \
    --scenarios av_system iot_smart_city_system \
    --seeds 42,123 \
    --epochs 150 \
    --output results/partial_table.json

python reproduce/render_table.py \
    --table3 results/partial_table.json \
    --output-dir results/
```

---

## File Structure

```
reproduce/
├── Makefile           — orchestration targets
├── Dockerfile         — exact environment (Python 3.11, PyG CPU)
├── README.md          — this file
├── EXPERIMENTS.md      — technical deep-dive on the harness internals and metrics
├── __init__.py        — package initialization
│
│   Core harness — this paper's Tables 1, 4-7:
├── middleware26_main_table.py   — 7×6×5 evaluation matrix (paper Tables 4-5)
├── loso_all_variants.py         — LOSO × 4 variants (paper Table 7)
├── render_table.py              — LaTeX/CSV/MD table renderer
├── render_stratified_figure.py  — per-node-type ρ figure
├── extract_attention.py         — HGT attention extraction (ATM running example)
├── render_attention_subgraph.py — attention figure renderer
│
│   New for the JSS revision — this paper's Tables 8-9:
├── reversed_projection_ablation.py  — inverted DEPENDS_ON direction ablation (Table 8)
├── hardening_budget.py              — risk-mass coverage by top-K selection method (Table 9)
│
│   Auxiliary / support scripts for this paper:
├── qos_pipeline_inspect.py      — stage-by-stage QoS attribute trace (source data for a figure)
├── pilot_hgl_native.py          — Go/No-Go sanity pilot for the HGL-native variant
├── recalibrate_main_table.py    — post-hoc F1 recalibration utility for main_table.json
├── run_experiment.py            — topology-only-vs-QoS-aware ablation harness (requires a
│                                   pre-populated Neo4j-backed cli/ pipeline; not offline-only)
│
│   NOT used by this paper — tooling for the separate, deprioritized SaG flagship paper
│   (docs/research/jss/draft.md; see docs/research/middleware2026/middleware26_revision_plan.md
│   for why this paper extends the Middleware submission instead of that draft):
├── run_expert_study.py          — expert-panel Fleiss' Kappa / Kendall's Tau calculations for
│                                   the flagship draft's §9; that expert panel was never convened
│                                   and is explicitly withdrawn in that draft — do not cite results
│                                   from this script as part of this paper
└── run_prescribe_all.py         — prescriptive-remediation (SRI) batch evaluation for the
                                    flagship draft's §6; this paper makes no remediation claims
```

---

## Citation

If you use this code, please cite the JSS submission (once accepted) or, until then, the original
conference submission it extends:

```bibtex
@article{qhgl2026jss,
  title   = {Heterogeneous Graph Learning for Cascade Impact Prediction in Distributed Publish-Subscribe Middleware},
  journal = {Journal of Systems and Software},
  note    = {Special Issue: AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems (VSI:AI4MSS). Under submission.},
  year    = {2026}
}

@inproceedings{qhgl2026,
  title     = {Heterogeneous Graph Learning for Cascade Impact Prediction in Distributed Publish-Subscribe Middleware},
  booktitle = {Proceedings of the 27th ACM/IFIP International Middleware Conference},
  year      = {2026},
  note      = {Middleware 2026 submission; rejected, revised and extended into the JSS submission above}
}
```
