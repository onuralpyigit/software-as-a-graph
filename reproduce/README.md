# Reproducing Software-as-a-Graph (SaG)

> **Software-as-a-Graph: Heterogeneous Graph Learning for Pre-Deployment Reliability and Dependability Analysis of Complex Distributed Systems**
> Submitted to the *Journal of Systems and Software* Special Issue **VSI:AI4MSS** (AI Techniques for
> Performance, Reliability, and Sustainability of Modern Software Systems). See
> `docs/research/jss/draft.md` for the Markdown version of the paper, `docs/research/jss/latex/`
> for the authoritative Elsevier LaTeX submission sources, and
> `docs/research/jss/methodology_revision_findings.md` for the pre-submission audit that regenerated
> its numbers.

This directory contains everything needed to reproduce the paper's experimental results, tables, and figures from scratch.
A Docker image is provided for exact environment replication.

---

## Hardware Requirements

| Configuration | Time estimate |
|---|---|
| CPU-only (8 cores, 32 GB RAM) | ~6–12 h (full, 5 seeds) |
| GPU (CUDA 11.8+, ≥8 GB VRAM) | ~1–2 h |
| Smoke-test (50 epochs, 2 seeds) | ~15–30 min CPU |
| Diagnostic & sensitivity sweeps (Tables 10–13) | seconds to minutes — pure graph computation & simulation |

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
python --version   # requires 3.10 or 3.11
pip install -e ".[all]"   # installs from pyproject.toml: base + neo4j + gnn (PyTorch Geometric) + api extras
```

There is no separate `requirements.txt`; `pyproject.toml` at the repo root is the single source of
truth for dependencies (see its `[project.optional-dependencies]` table for the `neo4j`/`gnn`/`api`/
`dev` extras, or use `all` as above to install everything this package needs).

### Step 1 — W1 Gate (sanity check, ~10 s)

```bash
make -f reproduce/Makefile block0
# Expected: 32/32 tests pass
```

### Step 2 — Main in-distribution results (JSS Tables 6 & 7, ~2–6 h CPU)

```bash
make -f reproduce/Makefile table3
# Output: results/table3_main_results.tex  /  .csv  /  .md
```

> **Table-number mapping.** This harness's internal target `table3` feeds **JSS Table 6 (`tab:5`)**
> (In-distribution held-out Spearman $\rho$ across 7 scenarios × 6 variants × 5 seeds = 210 cells)
> and **JSS Table 7 (`tab:6`)** (Paired Wilcoxon signed-rank tests across scenarios).

### Step 3 — Inductive LOSO cross-validation (JSS Table 8, ~3–8 h CPU)

```bash
make -f reproduce/Makefile table4
# Output: results/table4_loso_results.tex  /  .md
```

> Feeds **JSS Table 8 (`tab:7`)** (Inductive Leave-One-Scenario-Out evaluation across 8 folds).

### Step 3b — Per-domain k-fold (primary intra-scenario validation, ~8–20 h CPU)

```bash
make -f reproduce/Makefile kfold
# Output: results/table4_kfold_results.tex  /  .md
```

Runs `reproduce/kfold_all_variants.py` for all 5 variants (`hgl_qos`, `hgl`, `gl_qos`, `gl`,
`topology_rm`), each evaluated via repeated stratified k-fold (`k=5`, 5 seeds) *independently
within* each of the 7 cached scenarios. Confirmed result: HGL-QoS reaches mean cross-scenario
$\rho=0.587$ ($\sigma=0.146$), $F_1@K=0.505$, positive in all seven scenarios individually.

### Step 4 — Figures (JSS Figures 1–5)

To generate all 5 manuscript figures into `docs/research/jss/latex/figures/`:

```bash
make -f reproduce/Makefile jss-figures
```

Individual figures can also be generated:
- `make -f reproduce/Makefile jss-fig1`: Figure 1 — SaG Dual-Pathway Architecture (`Figure_1.pdf`)
- `make -f reproduce/Makefile jss-fig2`: Figure 2 — Running Example Multigraph (`Figure_2.pdf`)
- `make -f reproduce/Makefile jss-fig3`: Figure 3 — HGT Attention Case Study (`Figure_3.pdf`)
- `make -f reproduce/Makefile jss-fig4`: Figure 4 — AHP Shrinkage Sensitivity (`Figure_4.pdf`)
- `make -f reproduce/Makefile jss-fig5`: Figure 5 — Results at a Glance (`Figure_5.pdf`)

### Step 5 — Sensitivity & Diagnostic Sweeps (JSS Tables 5, 10–13, 15)

```bash
# Table 5: Generative parameters (tab_genparams.tex)
make -f reproduce/Makefile jss-tables

# Table 10: Topic-weight coefficient sensitivity (beta, alpha, psi)
make -f reproduce/Makefile topic-weight-sensitivity

# Table 11: AHP shrinkage parameter lambda sweep
python reproduce/ahp_sensitivity.py

# Table 12: Global Morris screening & Dirichlet sampling (k=10)
make -f reproduce/Makefile weight-global-sensitivity

# Table 13: Three-oracle rank agreement (I*, I_comp, I_dyn)
make -f reproduce/Makefile convergent-validity

# Table 15: Per-stage inference latency vs. system size
make -f reproduce/Makefile inference-latency
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

## Expected Outputs & JSS Paper Mapping

| Script / Target | Generated Artifact | JSS Paper Output | Content |
|---|---|---|---|
| `main_table.py` (`make table3`) | `results/table3_main_results.tex` | **Table 6 (`tab:5`)**, **Table 7 (`tab:6`)** | In-distribution held-out $\rho$, Wilcoxon tests |
| `loso_all_variants.py` (`make table4`) | `results/table4_loso_results.tex` | **Table 8 (`tab:7`)** | Inductive LOSO cross-validation (8 folds) |
| `scenario_param_table.py` (`make jss-tables`) | `tab_genparams.tex` | **Table 5 (`tab:genparams`)** | Scenario generation parameters |
| `topic_weight_sensitivity.py` | `results/topic_weight_sensitivity.json` | **Table 10 (`tab:8b`)** | Sensitivity of topic weights $(\beta, \alpha, \psi)$ |
| `ahp_sensitivity.py` | `results/ahp_shrinkage_sweep.json` | **Table 11 (`tab:8`)** | AHP shrinkage $\lambda$ sweep |
| `weight_global_sensitivity.py` | `results/weight_global_sensitivity.json` | **Table 12 (`tab:8e`)** | Global Morris screening + Dirichlet sampling |
| `convergent_validity.py` | `results/convergent_validity.json` | **Table 13 (`tab:8c`)** | Inter-oracle agreement ($I^*, I_{\text{comp}}, I_{\text{dyn}}$) |
| `inference_latency.py` | `results/inference_latency.json` | **Table 15 (`tab:scale`)** | Per-stage inference latency |
| `figure1_pipeline.dot` (`make jss-fig1`) | `docs/research/jss/latex/figures/Figure_1.pdf` | **Figure 1 (`fig:1`)** | Architecture flowchart |
| `figure2_running_example.dot` (`make jss-fig2`) | `docs/research/jss/latex/figures/Figure_2.pdf` | **Figure 2 (`fig:2`)** | Running example graph |
| `extract_attention.py` + `render_attention_subgraph.py` (`make jss-fig3`) | `docs/research/jss/latex/figures/Figure_3.pdf` | **Figure 3 (`fig:3`)** | HGT attention case study |
| `render_shrinkage_figure.py` (`make jss-fig4`) | `docs/research/jss/latex/figures/Figure_4.pdf` | **Figure 4 (`fig:4`)** | AHP shrinkage curve |
| `render_results_figure.py` (`make jss-fig5`) | `docs/research/jss/latex/figures/Figure_5.pdf` | **Figure 5 (`fig:5`)** | Results at a glance |

---

## Architecture Variants (Table columns)

These identifiers are used directly by `main_table.py`, `loso_all_variants.py`, and the evaluation suite:

| Variant flag | Description |
|---|---|
| `hgl_qos` | **HGL-QoS (Proposed)** — Heterogeneous Graph Transformer (HGTConv) with 16-D continuous-categorical edge features |
| `hgl` | **HGL** — Heterogeneous Graph Transformer (HGTConv) with QoS attributes masked |
| `gl_qos` | **GL-QoS** — Homogeneous GAT with scalar QoS weight per edge |
| `gl` | **GL** — Homogeneous GAT with no edge weighting |
| `topo_qos` | **Topo-QoS** — QoS-weighted structural centrality baseline |
| `topo_baseline` | **Topo-BL** — Unweighted structural centrality baseline |
| `topology_rm` | **RM / $Q(v)$** — Diagnostic reference score from the ISO/IEC explanation layer |

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

## File Structure

```
reproduce/
├── Makefile           — orchestration targets for JSS tables and figures
├── Dockerfile         — exact environment (Python 3.11, PyG CPU)
├── README.md          — this file
├── EXPERIMENTS.md      — technical deep-dive on the harness internals and metrics
├── __init__.py        — package initialization
│
│   Core empirical harness (JSS Tables 6, 7, 8):
├── main_table.py                — 7×6×5 evaluation matrix (JSS Tables 6-7)
├── loso_all_variants.py         — LOSO 8 folds (JSS Table 8)
├── kfold_all_variants.py        — Stratified k-fold evaluation
├── render_table.py              — LaTeX/CSV/MD table renderer
│
│   JSS manuscript figures (Figures 1–5):
├── extract_attention.py         — HGT attention extraction for ATM System (Figure 3)
├── render_attention_subgraph.py — Figure 3 renderer (Figure_3.{pdf,png})
├── render_shrinkage_figure.py   — Figure 4 renderer (Figure_4.{pdf,png})
├── render_results_figure.py     — Figure 5 renderer (Figure_5.{pdf,png})
│
│   JSS sensitivity & diagnostic sweeps (Tables 5, 10, 11, 12, 13, 15):
├── scenario_param_table.py      — Generative scenario parameter table (Table 5)
├── topic_weight_sensitivity.py  — Topic weight coefficient sensitivity (Table 10)
├── ahp_sensitivity.py           — AHP shrinkage parameter lambda sweep (Table 11)
├── weight_global_sensitivity.py — Joint Morris screening & Dirichlet sampling (Table 12)
├── convergent_validity.py       — Inter-oracle rank agreement I*, I_comp, I_dyn (Table 13)
├── inference_latency.py         — Per-stage inference latency vs scale (Table 15)
│
│   Auxiliary validation & diagnostic utilities:
├── qos_pipeline_inspect.py      — Stage-by-stage QoS attribute trace
├── recalibrate_main_table.py    — Post-hoc F1 recalibration utility
├── run_prescribe_all.py         — Closed-loop counterfactual remediation verification
├── reversed_projection_ablation.py  — Dependency projection direction ablation
└── hardening_budget.py          — Risk-mass coverage by top-K selection
```

---

## Citation

```bibtex
@article{sag2026jss,
  author  = {Yigit, Onuralp and Collaborators},
  title   = {Software-as-a-Graph: Heterogeneous Graph Learning for Pre-Deployment Reliability and Dependability Analysis of Complex Distributed Systems},
  journal = {Journal of Systems and Software},
  note    = {Special Issue: AI Techniques for Performance, Reliability, and Sustainability of Modern Software Systems (VSI:AI4MSS). Under submission.},
  year    = {2026}
}
```
