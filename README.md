# Software-as-a-Graph (SaG)

**Predict which components in a distributed system will cause the most damage when they fail — using only its architecture, before it is deployed.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Next.js 14](https://img.shields.io/badge/Next.js-14-black)](https://nextjs.org/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB)](https://react.dev/)
[![Neo4j 5.x](https://img.shields.io/badge/neo4j-5.x-green.svg)](https://neo4j.com/)
[![Docker](https://img.shields.io/badge/docker-compose-blue)](https://www.docker.com/)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

[Architecture](ARCHITECTURE.md) | [Development Guide](CLAUDE.md) | [Methodology docs](docs/) | [Reproduction package](reproduce/)

---

## Table of Contents

1. [The Problem](#the-problem)
2. [The Seven Capabilities](#the-seven-capabilities)
3. [Quick Start](#quick-start)
4. [Reproducing the Published Results](#reproducing-the-published-results)
5. [Validation Gates](#validation-gates)
6. [Empirical Results](#empirical-results)
7. [Applying SaG to Your Own System](#applying-sag-to-your-own-system)
8. [RM Quality Model in Brief](#rm-quality-model-in-brief)
9. [Anti-Pattern Detection in Brief](#anti-pattern-detection-in-brief)
10. [Python SDK](#python-sdk)
11. [Further Reading](#further-reading)
12. [License](#license)

---

## The Problem

In distributed publish-subscribe systems (ROS 2/DDS, Apache Kafka, MQTT), some components are structurally far more critical than others — when they fail, failures cascade. Finding these weak points traditionally requires either runtime monitoring, which adds latency to systems that are often safety-critical, or waiting for a production incident, by which point the data loss or service disruption has already happened.

> [!IMPORTANT]
> **Core insight:** a component's position in the dependency graph reliably predicts its real-world failure impact — with no runtime data.

SaG operationalizes that insight as a framework: it models the architecture as a weighted directed graph, ranks components by predicted failure impact, verifies the ranking against simulated ground truth, and prescribes fixes — all before deployment.

---

## The Seven Capabilities

Each capability is a pipeline stage with its own CLI script, SDK method, and methodology document. `Generate` is an offline prep step used for experiments and benchmarks; real deployments start at **Model**.

| # | Capability | What it does | Key output | Docs |
|:--|:---|:---|:---|:---|
| — | *Generate (prep)* | Synthesizes a pub-sub topology for experiments, benchmarks, and CI regression | Topology JSON | [graph-generation.md](docs/graph-generation.md) |
| 1 | **Model** | Imports topology JSON into Neo4j as a weighted directed graph $G = (V, E, \tau_V, \tau_E, w)$; derives logical `DEPENDS_ON` edges via six rules; computes QoS-derived weights | $G_{\text{structural}}$, $G_{\text{analysis}}(l)$ | [graph-model.md](docs/graph-model.md) |
| 2 | **Analyze** | Deterministic, closed-form. Computes the 11 Tier-1 structural metrics $M(v)$ — and nothing else | $M(v)$ metric vector | [structural-analysis.md](docs/structural-analysis.md) |
| 3 | **Predict** | Maps $M(v)$ to RM dimension scores and $Q^*(v)$ via declared-weight formulas (always); blends in a heterogeneous GNN when a trained checkpoint exists; detects anti-patterns; generates a natural-language explanation | RM/$Q^*(v)$ scores, 5-level classification, GNN ranks, anti-pattern report | [prediction.md](docs/prediction.md) |
| 4 | **Simulate** | Injects faults and propagates cascades over the raw structural graph to obtain ground-truth impact — training labels for stage 3 and the oracle for stage 5 | $I^*(v)$ composite and per-dimension $I_R, I_M$ (itself $\alpha\cdot I_{FT}+(1-\alpha)\cdot I_A$) | [failure-simulation.md](docs/failure-simulation.md) |
| 5 | **Validate** | Correlates predictions against simulated ground truth: Spearman $\rho$, Kendall $\tau$, F1, predictive gain, bootstrap CIs, Wilcoxon — scored against seven gates | Statistical evidence of predictive validity | [validation.md](docs/validation.md) |
| 6 | **Prescribe** | Generates architectural edits (topic splits, host reallocations, QoS upgrades) and accepts each only if a closed-loop counterfactual simulation confirms the improvement | `PrescribeResult` with baseline vs. mutated SRI | [prescription.md](docs/prescription.md) |
| 7 | **Visualize** | Renders network graphs, dependency matrices, cascade heatmaps, and RM radar charts | Self-contained `dashboard.html` | [visualization.md](docs/visualization.md) |

```
 topology JSON ──▶ 1 Model ──▶ 2 Analyze ──┬──▶ 3 Predict ──┐
                                           │                 ├──▶ 5 Validate ──▶ 6 Prescribe ──▶ 7 Visualize
                                           └──▶ 4 Simulate ──┘
```

Stages 3 and 4 are independent by construction — that separation is what makes the validation in stage 5 meaningful, and it is enforced by tests. The full DAG, the first-run sequencing rule, and the package boundaries are in [ARCHITECTURE.md](ARCHITECTURE.md#system-pipeline--data-flow).

---

## Quick Start

### Prerequisites

- **Python 3.11** — the only version exercised in CI. `pyproject.toml` declares `>=3.9`, but that is untested.
- **Neo4j 5.x** — required for the Model stage and anything downstream of it. No GDS or APOC plugins are needed.
- **Node.js 18+** — only if you want the SMART web dashboard.

### 1. Install

The `torch-scatter` / `torch-sparse` wheels must come from PyG's index before the extras are installed. This is the sequence CI uses ([tests.yml](.github/workflows/tests.yml)):

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 \
  -f https://data.pyg.org/whl/torch-2.5.0+cpu.html

pip install -e ".[all]"
```

### 2. Start Neo4j

```bash
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password neo4j:5.23.0
```

> [!WARNING]
> `neo4j` / `password` are local-development defaults. Change them before any shared deployment. Note that the Python CLI reads the `NEO4J_URI`, `NEO4J_USERNAME` (falling back to `NEO4J_USER`), and `NEO4J_PASSWORD` environment variables directly — it does not load the root `.env` file, which is consumed only by Docker Compose and Next.js.

### 3. Run the pipeline

```bash
# All stages (Model → Analyze → Simulate → Predict → Validate → Prescribe → Visualize)
python cli/run.py --all --layer system

# Or one stage at a time; see the CLI table in ARCHITECTURE.md
python cli/import_graph.py --input data/system.json --clear
python cli/analyze_graph.py --layer system
python cli/predict_graph.py --layer system
python cli/visualize_graph.py --layer system --output output/dashboard.html --open
```

On a first run there is no GNN checkpoint, so `run.py` disables the Predict stage's ML path and falls back to deterministic RM scoring. To enable the GNN, run Simulate to produce labels, train, then pass the checkpoint **directory**:

```bash
python cli/train_graph.py --layer system
python cli/predict_graph.py --layer system --gnn-model models/gnn_checkpoints
```

### 4. Optional — the full stack in Docker

```bash
docker compose up --build
```

One container publishes the SMART dashboard on `:7000`, the FastAPI server on `:8000` (`/docs` for OpenAPI), and Neo4j on `:7474` / `:7687`.

### 5. Run the tests

```bash
pytest                       # full suite; needs no live Neo4j
pytest -m "not integration"  # skip anything requiring a database
```

---

## Reproducing the Published Results

[`reproduce/`](reproduce/) is a self-contained package that regenerates every table and figure in the paper. It has its own [README](reproduce/README.md), [EXPERIMENTS.md](reproduce/EXPERIMENTS.md), pinned [Dockerfile](reproduce/Dockerfile), and [Makefile](reproduce/Makefile).

```bash
# Exact environment (recommended)
docker build -t saag-repro -f reproduce/Dockerfile .
docker run --rm -v $(pwd)/results:/workspace/results saag-repro

# Or locally, after the install above
make -f reproduce/Makefile smoke-test   # ~15-30 min, 50 epochs, 2 seeds
make -f reproduce/Makefile all          # ~6-12 h CPU, 5 seeds
```

| Target | Produces | Notes |
|:---|:---|:---|
| `block0` | W1 QoS pipeline audit | **Go/no-go gate** — must pass before `table3`. Runs `tests/test_qos_pipeline_audit.py` and `tests/test_baselines.py` |
| `table3` | `results/table3_main_results.tex` | Main results: scenarios × 4 variants × 5 seeds. Depends on `block0`; `--resume` makes it restartable |
| `table4` | `results/table4_loso_results.tex` | **LOSO** — leave-one-scenario-out, measures the cross-domain generalization gap |
| `kfold` | `results/table4_kfold_results.tex` | **Per-domain repeated k-fold** — the in-domain protocol. Opt-in, not part of `all`: ~5× more model fits than LOSO for the same seed count |
| `figure4` | `results/figure4_stratified_rho.pdf` | Per-node-type stratified $\rho$ |
| `figure5` | `output/atm_case_study/attention_subgraph.pdf` | ATM attention-weight case study |
| `jss-figures` | `docs/research/jss/latex/figures/Figure_1..6.pdf` | The six figures for the JSS manuscript ([docs/research/jss/latex/](docs/research/jss/latex/)) |
| `all` | everything above except `kfold` | |

Budget roughly 6–12 h on 8 CPU cores, or 1–2 h on a CUDA GPU. Ablations (`reproduce/qos_label_ablation.py`, `threshold_sensitivity.py`, `ahp_sensitivity.py`, `reversed_projection_ablation.py`, `hardening_budget.py`, `convergent_validity.py`) run separately and are documented in `reproduce/EXPERIMENTS.md`.

Shell orchestrators for longer sweeps live in [`scripts/`](scripts/): `verify_pipeline.sh` (end-to-end smoke), `run_main_table.sh`, `populate_loso_cache.sh`, `train_all_variants.sh`, `run_production_pipeline.sh`, `recalibrate.sh`.

---

## Validation Gates

Stage 5 scores each run against seven gates (G1–G6, G8; G7 and G9 were retired along with the Vulnerability/Security dimension both were built to gate — the numbering gap is intentional). Thresholds are defined in [`ValidationTargets`](saag/validation/models.py#L10) and evaluated in [`ValidationService`](saag/validation/service.py#L713); full definitions are in [validation.md](docs/validation.md).

| Tier | Gate | Threshold |
|:---|:---|:---|
| 1 | G1 — Spearman $\rho$ | $\ge 0.70$ |
| 1 | G2 — F1 | $\ge 0.75$ |
| 1 | G3 — Precision | $\ge 0.80$ |
| 1 | G4 — Top-5 overlap | $\ge 0.60$ |
| 2 | G5 — Predictive gain over degree baseline | $> 0.03$ |
| 2 | G6 — Weighted $\kappa_{\text{CTA}}$ (maintainability) | $\ge 0.70$ |
| 3 | G8 — Bottleneck precision | $\ge 0.70$ |

These are the **per-dimension** gates. The stricter **composite** targets for $Q^*(v)$ against $I^*(v)$ are separate: $\rho \ge 0.85$, F1 $\ge 0.90$, Top-5 $\ge 0.80$.

---

## Empirical Results

Validated across the seven-scenario evaluation suite in [`data/scenarios/`](data/scenarios/) — autonomous vehicles, IoT smart city, high-frequency trading, healthcare, hub-and-spoke, microservices, and enterprise-scale pub-sub — plus an ATM/air-traffic case study. See [docs/scenario.md](docs/scenario.md) for the corpus, its provenance manifest, and which scenario backs which result.

Figures below are re-run under the RM model (`make -f reproduce/Makefile table3`, 210 runs — 7 scenarios × 6 variants × 5 seeds); see [`results/table3_main_results.md`](results/table3_main_results.md) for the full table and [`results/table4_loso_results.md`](results/table4_loso_results.md) for the LOSO cross-scenario generalisation results.

| Metric | Target | Achieved (best single run) |
|:---|:---:|:---:|
| Composite Spearman $\rho(Q^*, I^*)$ | $\ge 0.85$ | **0.928** (av_system, Topo-QoS) |
| Composite $\rho$ at large scale (300 apps, enterprise_system) | — | **0.920** (HGL) |
| Composite F1 | $\ge 0.90$ | **1.00** (multiple HGL runs) |
| Predictive gain vs. degree baseline | $> 0.03$ | measured per run, see LOSO/table3 reports |
| Best variant | — | HGL-QoS (mean ρ = 0.652 LOSO, 0.631 table3) — heterogeneous + QoS-aware beats topology-only baselines throughout |
| Scale effect | — | Accuracy improves with system size (Enterprise, 300 apps, outperforms smaller scenarios on mean ρ) |

---

## Applying SaG to Your Own System

Real deployments skip `Generate` and start at **Model** with a topology JSON describing your architecture: `nodes`, `brokers`, `topics` (with QoS policy), `applications`, and `libraries`. The schema and a worked example are in [graph-model.md](docs/graph-model.md); [`data/system.json`](data/system.json) is a 50-application reference instance, and [`examples/run_worked_example.py`](examples/run_worked_example.py) verifies weight and `DEPENDS_ON` derivation end to end in memory.

The graph model maps to any pub-sub middleware:

| Graph concept | ROS 2 / DDS | Apache Kafka | MQTT |
|:---|:---|:---|:---|
| **Application** | ROS node | Producer / consumer | MQTT client |
| **Topic** | ROS topic | Kafka topic | MQTT topic |
| **Broker** | DDS participant | Kafka broker | MQTT broker |
| **Infrastructure Node** | Host / container | Broker host | Broker server |
| **Library** | ROS package dependency | Maven artifact | Paho client library |

---

## RM Quality Model in Brief

Criticality here is a **Quality-in-Use** construct in the ISO/IEC 25019:2023 (SQuaRE) sense: the degree to which the failure, latency or degradation of a component — directly or transitively — reduces the system's capacity to enable its stakeholders to achieve their goals. It is a *consequence*, carrying no estimate of how likely that failure is. The formal definitions (D1–D4), and how this differs from FMECA criticality, assigned integrity levels such as SIL/ASIL, and topological critical-node detection, are in [criticality.md](docs/criticality.md).

The construct spans all three SQuaRE quality views, and they are kept apart deliberately: criticality is **computed** from internal quality evidence (topology plus static code metrics), **validated** against simulated external quality (service delivered under fault), and **defined** on Quality-in-Use ([three views](docs/criticality.md#30-three-quality-views-internal-external-and-quality-in-use)).

It decomposes into two ISO/IEC 25010:2023 characteristics computed on the derived dependency graph, where edges point from *dependent* to *dependency*. Reliability is **hierarchical**: its Fault Tolerance and Availability sub-characteristics are scored individually and combined via a declared blend. Each characteristic/sub-characteristic identifies the failure **mechanism**; the Quality-in-Use characteristic that attribute's loss threatens is the **harm** ([full binding](docs/criticality.md#35-how-the-dimensions-bind-to-external-quality-dependability-and-quality-in-use)).

The same dimensions score components and relationships alike: the dimension fixes the harm, the scope fixes the mechanism.

| Dimension | For a component | For a relationship | External quality attribute | Quality-in-Use harm | Stakeholder |
|:---|:---|:---|:---|:---|:---|
| **R — Reliability** (hierarchical) | Combines the two rows below via α=0.36 | Combines the two rows below | Reliability | Efficiency, Effectiveness, Satisfaction | Reliability engineer / DevOps / SRE |
| ↳ **FT — Fault Tolerance** | Its failure spreads to transitive dependents | It carries the spread between endpoints | Reliability → fault tolerance | Efficiency, then Satisfaction | Reliability engineer |
| ↳ **A — Availability** | Its loss partitions the dependency graph | It is the only route, both endpoints healthy | Reliability → availability | Effectiveness, Freedom from risk | DevOps / SRE |
| **M — Maintainability** | It resists safe change | It forces both sides to change together | Maintainability → modularity, modifiability | Efficiency (engineering) | Software architect |

Vulnerability/Security was scored as a third dimension in an earlier revision of this framework and has been **retired outright** — not folded into another dimension — because its ground-truth evidence was the weakest of the (then) four dimensions and no fault-model instrument could validate it by construction ([full rationale](docs/criticality.md#35-how-the-dimensions-bind-to-external-quality-dependability-and-quality-in-use)). These now map onto two of the standard dependability attributes; **safety is not covered** — no hazard class or functional integrity field exists in the schema, so these scores locate structural exposure and cannot discharge a safety argument.

Each cell traces to one of the sub-definitions D1.R, D1.FT, D1.A, D1.M (components) and their D2 counterparts (relationships) in [criticality.md](docs/criticality.md#43-the-rm-model), where each is stated as D1 or D2 restricted to a single mechanism.

The composite is $R(v) = 0.36\, FT(v) + 0.64\, A(v)$, then $Q(v) = 0.80\, R(v) + 0.20\, M(v)$. These are DECLARED constants — not AHP output — algebraically re-derived from the retired 4-D composite (A=0.43, R=0.24, M=0.17, V=0.16) by dropping Vulnerability and renormalising ([full derivation](docs/structural-analysis.md#composite-score-qv)). An equal-weight baseline ($w_R=w_M=0.5$, $\alpha=0.5$) is available via `--equal-weights` for comparison.

Scores map to five tiers using adaptive box-plot thresholds derived from the system's own score distribution: **CRITICAL** above the upper fence $Q_3 + 1.5 \cdot IQR$, then **HIGH** above $Q_3$, **MEDIUM** above the median, **LOW** above $Q_1$, **MINIMAL** at or below $Q_1$. Below 12 components the classifier falls back to fixed percentiles (top 10% → CRITICAL).

The per-dimension formulas, all Tier-1 metric definitions, and the composite's re-parameterisation derivation are in [structural-analysis.md](docs/structural-analysis.md) and [criticality.md](docs/criticality.md).

---

## Anti-Pattern Detection in Brief

Stage 3 audits its own RM output against a catalog of **19** structural anti-patterns — 5 CRITICAL (SPOF, SYSTEMIC_RISK, GOD_COMPONENT, FAILURE_HUB, COMPOUND_RISK), 5 HIGH (CYCLE, BRIDGE_EDGE, BOTTLENECK_EDGE, BROKER_OVERLOAD, DEEP_PIPELINE), and 9 MEDIUM. (TARGET and EXPOSURE, keyed on the now-retired Security dimension, were retired along with it — the catalog was 21 before that.) Trigger conditions and remediations for each are specified in [antipatterns.md](docs/antipatterns.md) and [remediation.md](docs/remediation.md); the catalog itself lives in [`antipattern_detector.py`](saag/analysis/antipattern_detector.py#L35).

```bash
python cli/detect_antipatterns.py --layer system --output output/antipatterns.json
```

> [!TIP]
> Exit codes make this a pre-merge gate: **0** clean, **1** warnings/smells, **2** critical or high patterns detected.

---

## Python SDK

`saag.Pipeline` is a fluent builder; stages execute in dependency order regardless of the order you chain them.

```python
import saag

result = (
    saag.Pipeline.from_json("data/system.json", clear=True)
        .analyze(layer="app")                      # structural metrics only
        .simulate(layer="app", mode="exhaustive")  # ground-truth labels
        .predict()                                 # RM (+ GNN if a checkpoint exists)
        .validate()                                # statistical validation
        .prescribe()                               # counterfactual-verified remediations
        .visualize(output="output/report.html")
        .run()
)

for layer, v in result.validation.layers.items():
    print(f"{layer}: rho={v.spearman_rho:.3f}  F1={v.f1_score:.3f}")

if result.prescription:
    print(f"SRI improvement = {result.prescription.sri_improvement:.4f}")
```

| Class | Import path | Purpose |
|:---|:---|:---|
| [Pipeline](saag/pipeline.py#L12) | `saag.Pipeline` | Fluent builder that sequences and runs the stages |
| [Client](saag/client.py#L9) | `saag.Client` | Step-by-step service façade for finer control |
| [AnalysisResult](saag/models.py#L102) | `saag.AnalysisResult` | Stage 2 — structural metrics |
| [PredictionResult](saag/models.py#L190) | `saag.PredictionResult` | Stage 3 — RM/GNN scores, anti-patterns, explanation |
| [ValidationResult](saag/models.py#L396) | `saag.ValidationResult` | Stage 5 — per-layer correlations and gate outcomes |
| [PrescribeResult](saag/prescription/models.py#L212) | `saag.prescription.PrescribeResult` | Stage 6 — accepted policy, per-edit verdicts and SRI delta |

More runnable examples — including a round-trip persistence check and per-stage ATM walkthroughs — are in [`examples/`](examples/).

---

## Further Reading

**Framework internals** — [ARCHITECTURE.md](ARCHITECTURE.md) for package boundaries and data flow, [CLAUDE.md](CLAUDE.md) for conventions and invariants.

**Methodology, per stage** — [graph-model.md](docs/graph-model.md) · [graph-generation.md](docs/graph-generation.md) · [structural-analysis.md](docs/structural-analysis.md) · [criticality.md](docs/criticality.md) · [prediction.md](docs/prediction.md) · [antipatterns.md](docs/antipatterns.md) · [failure-simulation.md](docs/failure-simulation.md) · [validation.md](docs/validation.md) · [prescription.md](docs/prescription.md) · [remediation.md](docs/remediation.md) · [statistics.md](docs/statistics.md) · [visualization.md](docs/visualization.md) · [scenario.md](docs/scenario.md) · [cli-pipeline-guide.md](docs/cli-pipeline-guide.md)

**Formal specifications** — [SRS](docs/requirements/SRS.md) · [SAD](docs/design/SAD.md) · [SDD](docs/design/SDD.md) · [STD](docs/tests/STD.md) · [SAR](docs/tests/SAR.md) · [User Manual](docs/user/SUM.md)

**Research** — [reproduce/](reproduce/) for the reproduction package, [docs/research/](docs/research/) for paper sources.

The primary research contribution is the demonstration that topological graph metrics reliably predict real-world failure impact without runtime instrumentation. Supporting contributions: the six dependency derivation rules; the RM decomposition; the MPCI (Multi-Path Coupling Index) metric; the directed $AP_c$ single-point-of-failure score; adaptive box-plot classification; and the structural independence guarantee separating the predictor from the simulation oracle.

---

## License

Apache-2.0. See [`LICENSE`](LICENSE).

> [!WARNING]
> The `LICENSE` file currently contains **GPL-3.0** text and `pyproject.toml` declares `MIT` — neither matches the intended Apache-2.0 license. Both need to be corrected; this is tracked as a known discrepancy, not a documentation error.
