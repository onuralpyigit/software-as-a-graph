# Middleware 2026: Experimental Evaluation Overview

This document summarizes the methodology, experimental harness, and evaluation suite for the paper **"QoS-Aware Heterogeneous Graph Learning for Architectural Criticality Prediction"**.

---

## 1. Experimental Methodology

The core contribution of this work is the **Q-HGL** model—a QoS-aware Heterogeneous Graph Attention Network (HeteroGAT) designed to predict component criticality in large-scale distributed systems.

### A. Graph Representation
- **Nodes**: Applications, Libraries, Topics.
- **Edges**: `PUBLISHES`, `CONSUMES`, `USES`, and the derived `DEPENDS_ON`.
- **Edge Features**: 16-dimensional vectors containing QoS metadata (Latency, Reliability, Throughput) and topological weights.

### B. Training Matrix (Block C)
To ensure statistical significance, we evaluate across:
- **8 Diverse Scenarios**: ATM, AV System, Enterprise, Financial Trading, Healthcare, Hub-and-Spoke, IoT Smart City, and Microservices.
- **4 Model Variants**: `hetero_qos` (Q-HGL), `homo_scalar`, `homo_unweighted`, and `topology_rmav`.
- **5 Independent Seeds**: Ensuring results are not driven by initialization luck.

---

## 2. Experimental Harness (`middleware26_main_table.py`)

The harness provides a specialized environment for executing the 160-run matrix:

1.  **Topology Refinement**: Implements Rule 1 & 5 to derive logical dependencies from raw pub-sub relationships.
2.  **Label Remapping**: Maps simulation ground-truth (impact scores) to the refined graph topology.
3.  **GNN Service Integration**: Orchestrates the `saag` GNN pipeline, including stratified node splits and multi-head attention training.
4.  **Automated Wilcoxon**: Performs paired statistical testing between Q-HGL and all baselines per scenario.

---

## 3. Evaluation Suite

We evaluate models using two primary lenses: **Ranking** and **Identification**.

### A. Global Ranking (Spearman ρ)
Measures the correlation between predicted and actual criticality ranks.
- **Target**: $\rho \geq 0.80$ for dense systems.
- **Verification**: 95% Bootstrap Confidence Intervals (CI).

### B. Critical Component Identification (F1, Prec, Rec, Top-K)
Measures the model's ability to act as a binary classifier for "Critical" vs. "Safe" components.
- **F1 Score**: Harmonic mean of Precision and Recall (Threshold = 0.5).
- **Top-5/10 Overlap**: Intersection of the most critical components found by the model vs. simulation.
- **NDCG@10**: Normalized Discounted Cumulative Gain to reward high-accuracy at the top of the list.

---

## 4. Key Performance Highlights

Based on the validated 8x4x2 evaluation run:

| Dimension | Q-HGL Result | Interpretation |
|---|---|---|
| **Best-in-Class** | Enterprise (ρ=0.840) | Superior on complex hierarchical systems. |
| **Safety Margin** | Recall $\approx$ 1.000 | Effectively zero false-negatives for critical nodes. |
| **Classification** | F1 > 0.80 | Reliable for automated deployment gating. |
| **Statistical Sig.** | $p < 0.05$ | Statistically superior to RMAV and Scalar-GNN baselines. |

---

## 5. Reproducibility

The full suite is containerized and available via:
```bash
bash scripts/run_main_table.sh --epochs 300
```
Detailed technical documentation on the harness internals can be found in `reproduce/EXPERIMENTS.md`.
