# Reproducing HGL-QoS (Middleware 2026)

> **Heterogeneous Graph Learning for Cascade Impact Prediction in Distributed Publish-Subscribe Middleware**
> Submitted to ACM Middleware 2026

This directory contains everything needed to reproduce the paper's results from scratch.
A Docker image is provided for exact environment replication.

---

## Hardware Requirements

| Configuration | Time estimate |
|---|---|
| CPU-only (8 cores, 32 GB RAM) | ~6–12 h (full, 5 seeds) |
| GPU (CUDA 11.8+, ≥8 GB VRAM) | ~1–2 h |
| Smoke-test (50 epochs, 2 seeds) | ~15–30 min CPU |

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
pip install -r requirements.txt
```

> **PyTorch Geometric**: If not in `requirements.txt`, install via:
> ```bash
> pip install torch-geometric
> pip install torch-scatter torch-sparse  # optional but recommended
> ```

### Step 1 — W1 Gate (sanity check, ~10 s)

```bash
make -f reproduce/Makefile block0
# Expected: 26/26 tests pass
```

### Step 2 — Table 3 (main results, ~2–6 h CPU)

```bash
make -f reproduce/Makefile table3
# Output: results/table3_main_results.tex  /  .csv  /  .md
```

### Step 3 — Table 4 (LOSO inductive, ~3–8 h CPU)

```bash
make -f reproduce/Makefile table4
# Output: results/table4_loso_results.tex  /  .md
```

### Step 4 — Figures (run after Tables 3+4)

```bash
make -f reproduce/Makefile figure4   # Stratified ρ (instantaneous, reads JSON)
make -f reproduce/Makefile figure5   # ATM attention subgraph (~10 min)
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
| `results/table3_main_results.tex` | LaTeX Table 3 — Spearman ρ 8×6 |
| `results/table3_main_results.csv` | CSV version for Excel/R |
| `results/table4_loso_results.tex` | LaTeX Table 4 — LOSO Δρ |
| `results/figure4_stratified_rho.pdf` | Figure 4 — per-node-type ρ |
| `output/atm_case_study/attention_subgraph.pdf` | Figure 5 — HGT attention |
| `results/loso_all_variants.json` | Raw LOSO Δρ data |

---

## Architecture Variants (Table 3 / Table 4 columns)

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
    --scenarios atm_system av_system iot_smart_city_system \
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
├── __init__.py        — package initialization
├── middleware26_main_table.py   — Block C: 8×6×5 evaluation matrix
├── loso_all_variants.py         — Block E: LOSO × 4 variants
├── render_table.py              — Block C+E: LaTeX/CSV/MD table renderer
├── render_stratified_figure.py  — Block F: Figure 4 generator
├── extract_attention.py         — Block G: HGT attention extraction
└── render_attention_subgraph.py — Block G: Figure 5 renderer
```

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{qhgl2026,
  title     = {Heterogeneous Graph Learning for Cascade Impact Prediction in Distributed Publish-Subscribe Middleware},
  booktitle = {Proceedings of the 27th ACM/IFIP International Middleware Conference},
  year      = {2026},
  note      = {Middleware 2026}
}
```
