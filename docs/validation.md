# Step 5: Validation

**Statistically prove that topology-based predictions agree with simulation-derived cascade impact.**

← [Step 4: Simulation](failure-simulation.md) | → [Step 6: Visualization](visualization.md)

---

## Table of Contents

1. [What This Step Does](#1-what-this-step-does)
2. [Validation Pipeline Overview](#2-validation-pipeline-overview)
3. [Ground Truth: I(v) from Cascade Simulation](#3-ground-truth-iv-from-cascade-simulation)
4. [RMAV Prediction: Q(v)](#4-rmav-prediction-qv)
5. [Statistical Battery](#5-statistical-battery)
   - 5.1 [Rank Correlation — Spearman ρ and Kendall τ](#51-rank-correlation--spearman-ρ-and-kendall-τ)
   - 5.2 [Bootstrap Confidence Interval](#52-bootstrap-confidence-interval)
   - 5.3 [Classification Metrics — Precision, Recall, F1 @ K](#53-classification-metrics--precision-recall-f1--k)
   - 5.4 [SPOF-F1](#54-spof-f1)
   - 5.5 [Specialist Metrics — ICR@K, BCE, FTR, PG](#55-specialist-metrics--icrk-bce-ftr-pg)
   - 5.6 [Wilcoxon Signed-Rank Test](#56-wilcoxon-signed-rank-test)
6. [Node-Type Stratified Reporting](#6-node-type-stratified-reporting)
7. [Topology-Class Gate System](#7-topology-class-gate-system)
8. [Multi-Seed Stability Sweep](#8-multi-seed-stability-sweep)
9. [Ablation Study: Topology-Only vs. QoS-Enriched](#9-ablation-study-topology-only-vs-qos-enriched)
10. [CLI Reference](#10-cli-reference)
11. [Output Schema](#11-output-schema)
12. [Interpreting Results](#12-interpreting-results)
13. [What Comes Next](#13-what-comes-next)

---

## 1. What This Step Does

Step 5 closes the methodological loop. It aligns two independently derived signals for every
component in the system:

| Signal | Source | What it represents |
|--------|--------|-------------------|
| **Q(v)** | RMAV formula applied to topology (Step 3) | *Predicted* criticality — computed from graph structure alone, before any runtime data |
| **I(v)** | Stochastic cascade simulation (Step 4) | *Proxy ground truth* — normalised damage score obtained by injecting each node as the failure origin |

High statistical agreement between Q(v) and I(v) is empirical evidence that **topology alone predicts
failure impact** — the central claim of the Software-as-a-Graph thesis.

```
   Graph (Step 1)
        │
   ┌────┴──────────────────────────────┐
   │ Step 3: RMAV                      │  Step 4: Cascade Simulation
   │   Q(v) = w·R + w·M + w·A + w·V   │    I(v) = mean impact over N_repeats
   └────┬──────────────────────────────┘             simulation seeds
        │                   │
        └────────┬──────────┘
                 │
          Statistical Battery
          ─────────────────────
          Spearman ρ,  Kendall τ
          Bootstrap 95% CI
          F1@K, SPOF-F1, FTR
          ICR@K, BCE, PG
          Wilcoxon vs. degree baseline
                 │
          Gate Evaluation
          (topology-class adaptive)
                 │
          PASS / FAIL verdict
```

A compound test is used because no single metric is sufficient:
- **ρ** confirms the global rank ordering is preserved.
- **F1@K** confirms the top-K critical components are correctly identified.
- **PG** confirms Q(v) outperforms the naïve degree-centrality baseline.
- **SPOF-F1** confirms structural SPOFs are correctly caught.

---

## 2. Validation Pipeline Overview

`bin/validate_graph.py` implements the full pipeline as a single self-contained CLI.  
It loads a graph JSON, derives I(v) by running the `FaultInjector` for every node, computes Q(v) via
the central `QualityAnalyzer`, then runs the statistical battery.

```
load_graph(system.json)
    │
    ├── compute_rmav(G, qos=False|True)   →  Q(v), R(v), M(v), A(v), V(v)
    │       uses QualityAnalyzer + StructuralMetrics
    │
    └── derive_ground_truth(G, seed, n_repeats=5)  →  I(v)
            uses FaultInjector._inject_node() per node
            I(v) = mean of n_repeats stochastic cascade runs
            │
run_statistical_tests(node_scores, top_k)
    │
    ├── Rank correlation  →  ρ, τ, CI
    ├── Classification    →  Precision, Recall, F1, FTR
    ├── SPOF-F1
    ├── ICR@K, BCE, PG
    └── Wilcoxon vs. degree centrality
    │
stratified_metrics(by node type)
classify_topology(G)    →   "sparse" | "medium" | "dense" | "hub_spoke"
evaluate_gates(vr, topo_class)
```

> **Independence guarantee.** Q(v) uses only the graph structure (PageRank, betweenness, degree,
> articulation points) and optionally QoS contract attributes. I(v) is produced by a stochastic
> forward simulation that has no access to Q(v). The two pipelines are strictly independent.

---

## 3. Ground Truth: I(v) from Cascade Simulation

### Simulation Mechanics

For each node v, the `FaultInjector` runs a two-phase cascade:

**Phase A — Direct propagation:**  
Failure spreads from v along `DEPENDS_ON` and `USES` edges stochastically.  
Propagation probability decays with depth: `p(k) = p₀ × (1 − β)^k`.

**Phase B — Pub-sub orphaning:**  
If the failed node is a publisher of a topic, and it is the last surviving publisher for that topic,
all subscribers of that topic receive a downstream failure signal with dampened probability (0.5×
the current propagation factor). This correctly models the broadcast semantics of pub-sub systems.

### Ground Truth Derivation

```python
rng_seeds = [seed + i × 37  for i in range(n_repeats)]   # default n_repeats = 5

for each node v:
    impacts = []
    for s in rng_seeds:
        imp, depth, affected = FaultInjector._inject_node(v, seed=s)
        impacts.append(imp)
    I(v) = mean(impacts)
```

Averaging across `n_repeats` seeds dampens stochastic variance and yields a stable mean impact
estimate. This is the value compared against Q(v) in all subsequent statistical tests.

### What I(v) Represents

`I(v)` is the **normalised cascade impact score** ∈ [0, 1].  
It measures: *how much of the system becomes unreachable or impaired when node v fails?*

---

## 4. RMAV Prediction: Q(v)

Q(v) is computed by `QualityAnalyzer` using a four-dimensional formula:

```
Q(v) = w_A × A(v)  +  w_R × R(v)  +  w_M × M(v)  +  w_V × V(v)
```

**AHP-derived weights (default):**

| Dimension | Weight |
|-----------|:------:|
| Availability (A) | 0.43 |
| Reliability (R) | 0.24 |
| Maintainability (M) | 0.17 |
| Vulnerability (V) | 0.16 |

**Latest formula versions (Middleware 2026):**

*Reliability R(v) — v7:*
```
R(v) = 0.60 × PR(v) × (1 + MPCI(v))  +  0.40 × DG_in(v)
```

*Availability A(v) — v4:*
```
A(v) = AP_c_directed(v) × (1 + 0.30 × QSPOF(v))  +  0.20 × PSPOF(v)
```

Where:
- **PR(v)** — PageRank (global connectivity importance)
- **MPCI(v)** — Multi-Path Coupling Intensity (betweenness-derived amplifier)
- **DG_in(v)** — normalised in-degree (direct dependent count)
- **AP_c_directed(v)** — Directed Articulation Point score (1.0 if structural SPOF, else 0)
- **QSPOF(v)** — `AP_c_directed × w(v)` (QoS-amplified SPOF severity)
- **PSPOF(v)** — Publisher SPOF score (> 0 if this node is the *sole* publisher of a topic with subscribers)

See [docs/prediction.md](prediction.md) for the complete formula reference.

**Topology-only vs. QoS-enriched modes:**

| Mode | `--qos` flag | PSPOF contribution |
|------|:-----------:|--------------------|
| Topology-only baseline | off (default) | `PSPOF = 0` for all nodes |
| QoS-enriched | on | `PSPOF` computed from pub-sub topology |

The ablation study (`compare` subcommand) measures the predictive lift from the QoS-enriched mode.

---

## 5. Statistical Battery

All statistics are computed on **Application-type nodes** by default, falling back to all nodes only
when fewer than 4 Application nodes exist. This matches the thesis claim: topology predicts
*application-layer* cascade criticality.

### 5.1 Rank Correlation — Spearman ρ and Kendall τ

**Spearman ρ** is the primary gate metric. It measures whether the *rank ordering* of components by
Q(v) matches the rank ordering by I(v).

```
ρ = 1  −  (6 × Σ dᵢ²) / (n × (n² − 1))

where  dᵢ = rank(Q(vᵢ)) − rank(I(vᵢ))
```

Significance test: two-tailed t-distribution with df = n − 2.

| ρ Range | Interpretation |
|---------|---------------|
| ≥ 0.85 | Very strong agreement |
| 0.80–0.85 | Strong — primary gate passes |
| 0.75–0.80 | Moderate — gate fails for dense/hub-spoke classes |
| < 0.75 | Weak — investigation required |

**Kendall τ** is the conservative cross-check:

```
τ = (C − D) / √((C + D + T_Q)(C + D + T_I))
```

A large |ρ − τ| gap (> 0.15) indicates that agreement is driven by a few extreme outliers. Inspect
the top 2–3 CRITICAL components in that case.

### 5.2 Bootstrap Confidence Interval

Non-parametric bootstrap CI for Spearman ρ (B = 2000 resamples, seed 42):

```
for b in 1..B:
    idx  = sample with replacement from [0..n-1]
    ρ_b  = spearmanr(Q[idx], I[idx])

CI_95 = [percentile(ρ_b, 2.5),  percentile(ρ_b, 97.5)]
```

A CI that does not cross the gate threshold provides stronger evidence than a point estimate alone.
When variance is zero (constant arrays), the CI degenerates to `[0, 0]` and a warning is emitted.

### 5.3 Classification Metrics — Precision, Recall, F1 @ K

`K` defaults to **20% of total node count** (minimum 3, maximum n). Override with `--top-k`.

```
gt_top_k  = top K nodes by I(v)   (ground truth critical set)
pred_top_k = top K nodes by Q(v)  (predicted critical set)

TP = |gt_top_k ∩ pred_top_k|
FP = |pred_top_k − gt_top_k|
FN = |gt_top_k  − pred_top_k|

Precision@K = TP / (TP + FP)
Recall@K    = TP / (TP + FN)
F1@K        = 2 × Precision × Recall / (Precision + Recall)
```

### 5.4 SPOF-F1

Measures the quality of articulation-point detection as an availability indicator.

```
SPOF-actual   = {v : is_articulation_point(v)  AND  I(v) > 0.3}
SPOF-predicted = {v : is_articulation_point(v)}

SPOF-F1 = harmonic mean of SPOF-precision and SPOF-recall
```

A low SPOF-F1 with a high overall ρ means the *global* ordering is correct but the binary SPOF
classification threshold is misaligned with the simulation threshold (0.3). See
[Interpreting Results](#12-interpreting-results).

### 5.5 Specialist Metrics — ICR@K, BCE, FTR, PG

**ICR@K — In-Cluster Recall at K:**

```
window = max(1, K // 2)
ICR@K  = |{v ∈ gt_top_k : |rank_Q(v) − rank_I(v)| ≤ window}| / K
```

Measures whether true-critical components appear in the *neighbourhood* of their Q-rank, even if not
exactly in the top-K. A high ICR with a low F1 indicates predictions are close but shifted by a few
positions.

**BCE — Binary Classification Error:**

```
y_true[v] = 1 if v ∈ gt_top_k  else 0
y_pred[v] = 1 if v ∈ pred_top_k else 0
BCE = mean(y_true ≠ y_pred)
```

The fraction of all nodes misclassified as critical or non-critical. Lower is better; 0 is perfect.

**FTR — False Top Rate:**

```
FTR = FP / K
```

The fraction of the predicted top-K that are *false alarms* (predicted critical, actually safe). A
high FTR is dangerous operationally: engineers would spend hardening effort on low-risk components.

**PG — Predictive Gain over degree-centrality baseline:**

```
ρ_DC     = spearmanr(degree_centrality(v), I(v))
ρ_Q      = spearmanr(Q(v), I(v))
PG       = |ρ_Q| − |ρ_DC|
```

PG > 0 means RMAV outperforms a naïve PageRank baseline. PG ≥ 0.03 is the gate threshold for
confirming that the multi-dimensional RMAV formula adds genuine predictive value.

### 5.6 Wilcoxon Signed-Rank Test

Tests whether Q(v) ranks nodes *better* than degree centrality against ground truth I(v):

```
diff_scores = |Q(v) − I(v)| − |DC(v) − I(v)|     for all v

Wilcoxon signed-rank test (one-sided: alternative='less', α=0.05)
```

Significance (`p < 0.05`) means Q(v) is statistically closer to I(v) than degree centrality is.
Requires at least 10 nodes for reliable results; otherwise the test is skipped and `p = 1.0` is
reported.

---

## 6. Node-Type Stratified Reporting

Spearman ρ and F1@K are computed independently for each node type:

```
Application   →  primary validation layer
Broker        →  secondary (broker-layer analysis)
Topic         →  expected zero-variance signal (cascade simulation does not
                 propagate *from* topics); reported with a note
InfraNode     →  infrastructure layer; smaller population
Library       →  library coupling layer; fewer nodes → noisier ρ
```

Strata with fewer than 4 nodes report `"too few nodes for ρ"`.  
Strata with constant I(v) (std < 1e-9) report `"constant signal (not a primary failure type)"`.

**Typical stratification output:**
```
  Application       n=  26  ρ= 0.8320  F1=0.7143
  Broker            n=   5  constant signal (not a primary failure type)
  Topic             n=  27  constant signal (not a primary failure type)
  InfraNode         n=   8  constant signal (not a primary failure type)
  Library           n=   8  constant signal (not a primary failure type)
```

Topics and Brokers are expected to show constant signal: the cascade simulation triggers from a
*source* node's failure, not from a topic. Topic-layer reliability is captured through pub-sub
orphaning (Phase B), but the score accrues to the publisher application, not the topic node itself.

---

## 7. Topology-Class Gate System

Gates are **adaptive** — a sparse 12-node system faces less stringent thresholds than a dense 80-node
hub-spoke architecture.

### Topology Classification

```python
density   = edges / (nodes × (nodes − 1))
hub_ratio = max_degree / mean_degree

"hub_spoke" if hub_ratio > 10  and density < 0.10
"sparse"    if density < 0.05
"dense"     if density > 0.20
"medium"    otherwise
```

### Gate Thresholds

| Class | ρ ≥ | F1 ≥ | SPOF-F1 ≥ | FTR ≤ | PG ≥ |
|-------|:---:|:----:|:---------:|:-----:|:----:|
| `sparse` | 0.75 | 0.65 | 0.60 | 0.30 | 0.02 |
| `medium` | 0.80 | 0.70 | 0.65 | 0.25 | 0.03 |
| `dense` | 0.82 | 0.72 | 0.65 | 0.25 | 0.03 |
| `hub_spoke` | 0.85 | 0.75 | 0.70 | 0.20 | 0.03 |

All five gates must pass for `overall_pass = True`. The exit code is 0 on PASS and 1 on FAIL,
enabling use in CI pipelines.

---

## 8. Multi-Seed Stability Sweep

A single seed run is not sufficient evidence. The `sweep` and `report` subcommands run the full
pipeline across multiple seeds to measure **stability**.

**Default seeds:** `42, 123, 456, 789, 2024`

**Aggregate metrics:**

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| `rho_mean` | mean(ρ across seeds) | Average predictive power |
| `rho_std` | std(ρ across seeds) | Stability; target σ ≤ 0.05 |
| `rho_min / rho_max` | range | Worst and best seed |
| `f1_mean` | mean(F1@K) | Average classification quality |
| `pg_mean` | mean(PG) | Average predictive gain |
| `rcr` | 1 − mean(normalised Kendall distance between seed pairs) | Rank Consistency Rate |
| `all_gates_pass_rate` | fraction of seeds that pass all gates | Reliability of PASS verdict |

**RCR — Rank Consistency Rate:**

```
For each pair of seeds (i, j):
    compute normalised Kendall distance d_{ij} = (1 − τ_{ij}) / 2  ∈ [0, 1]

RCR = 1 − mean(d_{ij})
```

RCR = 1.0 means identical rankings across all seeds. Target: RCR ≥ 0.90 for a stable methodology.

---

## 9. Ablation Study: Topology-Only vs. QoS-Enriched

The `compare` subcommand runs two full sweeps back-to-back — one without QoS (topology-only
baseline) and one with QoS enrichment — and reports the pair-wise deltas.

**Primary claim:** Δρ = ρ(Q_QoS, I) − ρ(Q_topo, I) > 0, *p* < 0.05

This is the evidence that incorporating QoS contract topology (singleton publisher detection, weight
amplification) adds predictive signal beyond purely structural graph metrics.

**Statistical test:** Paired t-test on seed-level ρ series (`alternative='greater'`).  
For fewer than 3 seeds, significance is approximated as `Δρ > 0.01`.

**LaTeX export:**

The `--latex` flag generates a ready-to-paste IEEE booktabs table (`ablation_table.tex`):

```latex
\begin{table}[t]
\centering
\caption{Ablation Study: Topology-Only vs.\ QoS-Enriched Prediction}
\label{tab:ablation}
\begin{tabular}{@{}lSSS@{}}
\toprule
Metric & {Topo-Only} & {QoS-Enr.} & {$\Delta$} \\
\midrule
Spearman $\rho$ (mean) & 0.8123 & 0.8467 & +0.0344 \\
Spearman $\rho$ (std)  & 0.0231 & 0.0198 & -0.0033 \\
F1 @ $K$               & 0.7200 & 0.7600 & +0.0400 \\
Predictive Gain (PG)   & 0.0410 & 0.0620 & +0.0210 \\
RCR                    & 0.9340 & 0.9410 & +0.0070 \\
\midrule
\multicolumn{4}{l}{\small QoS $\rho$-lift: $p < \alpha$, significant} \\
\bottomrule
\end{tabular}
\end{table}
```

The `compare` subcommand exits with code 0 only when Δρ > 0 *and* the test is significant — suitable
for CI gates guarding research claims.

---

## 10. CLI Reference

### Subcommands

| Subcommand | Description |
|-----------|-------------|
| `single` | One-seed run (uses the first seed in the list) |
| `sweep` | Multi-seed stability sweep |
| `report` | Full sweep + per-seed detail + gate JSON output |
| `compare` | Ablation study: topology-only vs. QoS-enriched |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input PATH` | *(required)* | Path to `system.json` or dataset JSON |
| `--qos` | off | Enable QoS-weighted RMAV scoring (adds PSPOF) |
| `--top-k INT` | 20% of nodes | K for classification and specialist metrics |
| `--seeds INTS` | `42,123,456,789,2024` | Comma-separated seed list |
| `--cascade INT` | `5` | Cascade simulation depth limit |
| `--bootstrap INT` | `2000` | Bootstrap resamples for CI |
| `--alpha FLOAT` | `0.05` | Significance level for Wilcoxon test |
| `--output PATH` | *(none)* | Write JSON report to path |
| `--csv` | off | Also write per-node CSV table |
| `--latex` | off | Write LaTeX ablation table (requires `compare` subcommand) |
| `--verbose` | off | Print per-node Q(v)/I(v) scores ranked by Q |
| `--no-color` | off | Disable ANSI colours in console output |

### Examples

```bash
# ── Quick sanity check (topology-only, single seed) ──────────────────────────
python bin/validate_graph.py single --input input/scenarios/atm_system.json

# ── QoS-enriched single run ───────────────────────────────────────────────────
python bin/validate_graph.py single --input input/scenarios/atm_system.json --qos --verbose

# ── Multi-seed stability sweep ────────────────────────────────────────────────
python bin/validate_graph.py sweep --input input/scenarios/atm_system.json --qos

# ── Full report with JSON output ──────────────────────────────────────────────
python bin/validate_graph.py report \
    --input input/scenarios/atm_system.json \
    --output output/atm_validation.json \
    --qos

# ── Ablation study + LaTeX table ──────────────────────────────────────────────
python bin/validate_graph.py compare \
    --input input/scenarios/atm_system.json \
    --output output/ablation.json \
    --seeds 42,123,456,789,2024 \
    --latex

# ── Custom K, deeper cascade, finer CI ────────────────────────────────────────
python bin/validate_graph.py report \
    --input input/scenarios/atm_system.json \
    --top-k 10 --cascade 10 --bootstrap 5000 \
    --qos --output output/fine_report.json

# ── CI-friendly exit codes ────────────────────────────────────────────────────
# Exit 0 = PASS;  Exit 1 = FAIL
python bin/validate_graph.py single --input input/scenarios/atm_system.json --qos \
    && echo "Validation passed"
```

---

## 11. Output Schema

### Single / Report JSON

```json
{
  "topology_class": "sparse",
  "validation": {
    "seed": 42,
    "qos_enabled": true,
    "n_nodes": 74,
    "n_app_nodes": 26,

    "spearman_rho":    0.8320,
    "spearman_p":      0.0009,
    "kendall_tau":     0.6101,
    "kendall_p":       0.0023,
    "bootstrap_ci_lo": 0.6841,
    "bootstrap_ci_hi": 0.9241,

    "top_k":           14,
    "precision_at_k":  0.7143,
    "recall_at_k":     0.7143,
    "f1_at_k":         0.7143,
    "spof_f1":         0.6250,
    "ftr":             0.2857,

    "icr_at_k":        0.7857,
    "bce":             0.2432,
    "pg":              0.0512,

    "wilcoxon_stat":   164.00,
    "wilcoxon_p":      0.0312,
    "wilcoxon_significant": true,

    "strata": {
      "Application": { "n": 26, "spearman_rho": 0.8320, "spearman_p": 0.0009, "f1_at_k": 0.7143, "k_used": 5 },
      "Broker":      { "n": 5,  "note": "constant signal (not a primary failure type)" },
      "Topic":       { "n": 27, "note": "constant signal (not a primary failure type)" }
    },

    "gates_passed": {
      "rho >= 0.75":      true,
      "f1 >= 0.65":       true,
      "spof_f1 >= 0.60":  true,
      "ftr <= 0.30":      true,
      "pg >= 0.02":       true
    },
    "overall_pass": true
  }
}
```

### Sweep / Report JSON

```json
{
  "sweep": {
    "qos_enabled": true,
    "seeds": [42, 123, 456, 789, 2024],
    "rho_mean":  0.8217,
    "rho_std":   0.0341,
    "rho_min":   0.7810,
    "rho_max":   0.8640,
    "f1_mean":   0.6974,
    "pg_mean":   0.0438,
    "rcr":       0.9231,
    "all_gates_pass_rate": 0.80,
    "per_seed":  [ ... ]
  },
  "topology_class": "sparse",
  "gate_thresholds": [0.75, 0.65, 0.60, 0.30, 0.02]
}
```

### CSV Output (`--csv` flag)

When `--csv` is specified, a file `<output>_nodes.csv` is written with one row per node, ranked
by Q(v) descending:

```
rank, node_id, node_type, Q, R, M, A, V, I, cascade_depth, nodes_affected, is_articulation_point, degree_centrality
1, ConflictDetector, Application, 0.8421, 0.7200, 0.6500, 0.9100, 0.4200, 0.8102, 4, 18, True, 0.0321
...
```

---

## 12. Interpreting Results

### ρ is high (≥ gate) but F1@K fails

The global rank ordering is correct, but the binary threshold for "critical" is misaligned.
Check the distribution of Q(v) and I(v):

- If IQR(I) is very small (most nodes have similar impact), the top-K boundary is unstable.
- Consider increasing `--top-k` to use a larger critical set, or reviewing whether the gate
  threshold is appropriate for this system's size.

### ρ is negative

This is the **Inverse Criticality** effect, observed in mission-critical systems where the most
central nodes are the most heavily hardened. High PageRank components have redundant publishers and
backup routes, so their failure impact is *lower* than less-connected components.

Possible causes:
- The system was explicitly designed for resilience (ATM, financial HFT) — high-centrality nodes
  are protected by design intent.
- The cascade simulation depth limit (`--cascade`) is too shallow to propagate through well-connected
  hubs; increase to `--cascade 10` or `--cascade 15`.
- The `--qos` flag is off: PSPOF is zero, so sole-publisher nodes are not penalised.

### PG ≤ 0 (RMAV no better than degree centrality)

RMAV is not adding value beyond the naïve baseline:
- If the graph has very few Application nodes (< 10), the statistical test has high variance.
- Check whether `ap_c_directed` and `mpci` are being populated in `StructuralMetrics` (they default
  to 0 if the upstream analyzer did not run Step 2 first).
- Run `--verbose` to inspect per-node Q(v) vs I(v) to identify systematic mispredictions.

### Wilcoxon not significant with high ρ

This is expected when n < 10. The Wilcoxon test requires at least 10 nodes for reliable results. On
small graphs, treat ρ as the primary evidence and Wilcoxon as inconclusive.

### Strata show "constant signal" for Topics and Brokers

This is the expected and correct behaviour. Topics and Brokers do not generate cascade failures as
*origin* nodes in the `FaultInjector` simulation — their impact accrues to connected Application
nodes via Phase B orphaning. Constant I(v) = 0 for these types does not indicate a bug.

### Large |ρ − τ| gap (> 0.15)

Agreement is driven by a small number of extreme outliers. Inspect the top 2–3 CRITICAL components:
one may be a global hub with disproportionate cascade reach. This is not necessarily a failure — it
often reflects correct detection of a "God Component" in the architecture.

---

## 13. What Comes Next

The validation report produced by Step 5 is the empirical evidence base for the central thesis
claim. Step 6 (Visualization) renders these results into an interactive HTML dashboard:

- **Q(v) vs. I(v) scatter plot** — with quadrant highlighting (TP, TN, FP, FN).
- **Delta heatmap** — topology graph coloured by |Q(v) − I(v)|; high-delta nodes are "architectural
  surprises" where structural intent diverges from simulated behaviour.
- **RMAV radar charts** — per-node multi-dimensional breakdown for critical components.
- **Ranked component table** — with Q, R, M, A, V, I columns and pass/fail gate badge.

For research-grade LaTeX artifacts, run `compare --latex` to generate the ablation table directly
from the pipeline results:

```bash
python bin/validate_graph.py compare \
    --input input/scenarios/atm_system.json \
    --seeds 42,123,456,789,2024 \
    --output output/ablation_final.json \
    --latex
```

This writes `output/ablation_final_table.tex` — a booktabs-formatted table ready for IEEE/ACM
double-column layout.

---

← [Step 4: Simulation](failure-simulation.md) | → [Step 6: Visualization](visualization.md)