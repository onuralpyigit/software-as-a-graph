# CLAUDE.md Development Guide

**Quick-reference guide for LLM coding guidelines, build commands, test suites, and project conventions.**

[README](README.md) | [Architecture](ARCHITECTURE.md)

---

## Table of Contents

1. [LLM Coding Guidelines](#llm-coding-guidelines)
2. [Command Line Reference](#command-line-reference)
3. [Project Conventions & Standards](#project-conventions--standards)
4. [Invariants](#invariants)
5. [Known Gotchas](#known-gotchas)

---

## LLM Coding Guidelines

These guidelines represent best-practice rules for AI coding assistants working on the repository. They prioritize caution, correctness, and simplicity.

### Think Before Coding
- **State assumptions explicitly** — if any requirements are ambiguous, clarify them with the user before writing code.
- **Surface tradeoffs** — present alternative implementations or designs rather than making a choice silently.
- **Push back when warranted** — if a simpler solution or a pre-existing function can achieve the task, explain it.
- **Stop when confused** — identify what is unclear, stop execution, and ask the user for clarification.

### Simplicity First
- **Avoid speculative features** — implement the minimum code required to solve the task. Do not add abstractions for single-use logic.
- **Keep it concise** — avoid adding unrequested configurability or overcomplicated error handling for impossible scenarios.
- **Review for complexity** — if a file or function becomes unnecessarily verbose, refactor it to its simplest equivalent form.

### Surgical Changes
- **Minimize diff blast-radius** — touch only the lines of code directly related to the user request.
- **Avoid unrelated styling or cleanup** — do not "improve" adjacent formatting, docstrings, or logic that is not broken.
- **Observe style parity** — match the established conventions, naming patterns, and file structure of the code you are editing.
- **Prune orphans** — clean up unused imports, variables, and functions introduced by your changes. Do not delete pre-existing dead code unless explicitly requested.

### Goal-Driven Execution
- **Establish success criteria** — transform requirements into testable validation steps (e.g. reproducing a bug in a test first).
- **Run verification loops** — execute tests before, during, and after implementing changes to verify that existing features do not break.

---

## Command Line Reference

Install, Docker, and frontend commands are documented once, in [README §Quick Start](README.md#quick-start) — do not re-derive them here; link instead.

### Pipeline Entry Points
Eight CLI scripts have console-script entry points (`pyproject.toml`); the rest run as `python cli/<script>.py` (or `PYTHONPATH=. python cli/<script>.py` if not installed editable):

```bash
saag --all --layer system              # cli/run.py — full pipeline
saag-import  --input data/system.json --clear
saag-analyze --layer system
saag-predict --layer system
saag-simulate fault-inject --input data/system.json --export-json
saag-validate report --input data/system.json --qos
saag-visualize --layer system --output output/dashboard.html

# No entry point — invoke directly:
python cli/train_graph.py --layer system
python cli/prescribe_graph.py --layer system
python cli/benchmark.py
```

Full flag reference for every script: [docs/cli-pipeline-guide.md](docs/cli-pipeline-guide.md). Package/service boundaries: [ARCHITECTURE.md](ARCHITECTURE.md).

### Pytest Testing Commands
Tests run from the project root and do not require a live Neo4j instance unless marked `integration`:

```bash
pytest                        # full suite
pytest -x                     # halt on first failure
pytest -k "reliability"       # pattern match
pytest tests/test_analysis_service.py tests/test_prescription.py
pytest -m "not integration"   # skip anything needing a live database
pytest -m slow                # run only the slow-marked tests
```

### Reproduction Package
Experiment sweeps and paper tables/figures live under `reproduce/`, with their own Makefile:

```bash
make -f reproduce/Makefile block0       # go/no-go gate — run before anything else
make -f reproduce/Makefile smoke-test   # fast sanity check, ~15-30 min
make -f reproduce/Makefile table3       # main results table
make -f reproduce/Makefile table4       # LOSO cross-domain generalization
make -f reproduce/Makefile kfold        # per-domain k-fold (opt-in, not part of `all`)
```

See [reproduce/README.md](reproduce/README.md) and [reproduce/EXPERIMENTS.md](reproduce/EXPERIMENTS.md) for the full protocol.

---

## Project Conventions & Standards

### Architecture & Design Patterns
- **Hexagonal Architecture** — The core SDK in [saag/](saag/) isolates business logic from databases and clients. Use Cases in [saag/usecases/](saag/usecases/) act as the boundary interface between presentation layers (FastAPI, CLI) and core domain services.
- **Repository Pattern** — Data operations are mapped to the `IGraphRepository` interface. Production runs use [Neo4jRepository](saag/infrastructure/neo4j_repo.py), whereas unit tests use the mock [MemoryRepository](saag/infrastructure/memory_repo.py). Core services must never depend directly on Neo4j classes.
- **REST Presenters (target state, partially applied)** — API routers in [api/routers/](api/routers/) should delegate serialization to presenters in [api/presenters/](api/presenters/). `graph.py`, `simulation.py`, `analysis.py`, `prediction.py`, and `validation.py` do this fully; `components.py` does it partially (only `serialize_critical_edge`). `classification.py`, `statistics.py`, `traffic.py`, and `health.py` still serialize inline. When touching one of those four, move it to a presenter rather than adding more inline serialization — but don't treat the gap as a bug to fix opportunistically across the whole API.

### Layer Projections & Dependency Derivation
- **Analysis Layers** — The codebase uses four canonical layers (`app`, `infra`, `mw`, `system`) defined in [saag/core/layers.py](saag/core/layers.py). The `app` layer includes both `Application` and `Library` nodes.
- **Logical Dependencies** — Physical pub-sub linkages are transformed into `DEPENDS_ON` edges using the six rules implemented in `Neo4jRepository`. In the dependency graph, edge direction points from the *dependent* to its *dependency*.
- **Simulation Substrate** — Failure simulations in [saag/simulation/](saag/simulation/) operate strictly on raw structural edges ($G_{\text{structural}}$), not derived dependency edges ($G_{\text{analysis}}$).

### Documentation Hyperlinking Standards
- **Relative Portability** — When referencing files, folders, or code elements inside documentation, always use standard relative Markdown links. Avoid using absolute `file:///` URLs to maintain portability across different workspaces and Git hosting systems.
- **Line Ranges** — Link directly to symbol line boundaries (e.g. `[Pipeline](saag/pipeline.py#L12)`) when referencing code elements to assist LLM navigation.

---

## Invariants

These are enforced by tests — breaking one will fail CI, not just review:

- **Simulation never reads derived edges.** `saag/simulation/` operates strictly on raw structural edges, never `DEPENDS_ON`. Enforced by `tests/test_independence_guarantee.py`.
- **`FaultInjector` and `FailureSimulator` are not interchangeable.** `FaultInjector` is the Predict-stage labeler; `FailureSimulator` is the Validate-stage oracle. Never substitute one for the other within a stage. Enforced by `tests/test_groundtruth_contract.py`.
- **Predict does not import Simulate.** The dependency runs the other way at the pipeline level (Simulate produces training labels; Predict consumes them from disk), not as a live import. Enforced by `tests/test_predict_simulate_separation.py`.
- **Layer names are exactly `app` / `infra` / `mw` / `system`.** Defined once in [saag/core/layers.py](saag/core/layers.py); `app` includes both `Application` and `Library` nodes.
- **Committed datasets are exactly what their configs generate.** Every `data/scenarios/*_system.json` must regenerate byte-identically from its `scenario_*.yaml`, and `data/scenarios/MANIFEST.json` must match what is on disk. Enforced by `tests/test_scenario_corpus.py`. After changing the generator, regenerate the corpus, refresh the manifest, and rebuild the caches (`make -f reproduce/Makefile cache`) — a stale `output/loso_cache/` silently outranks the datasets in `reproduce/main_table.py` and has published wrong numbers before.
- **A quality-in-use scalarisation is a reweighting of RM, never an independent score.** `saag/core/quality_model.py`'s `QIU_PROJECTION` is row-stochastic by construction, so any domain-weighted projection onto quality-in-use harm is algebraically identical to scoring the same RM vector under different composite weights (`M^T omega`). Do not compute a "quality-in-use score" and report it as a quantity distinct from a reweighted `Q(v)`. Enforced by `tests/test_quality_model.py::TestQiuCollapseEquivalence`.
- **The RM composite is a re-parameterisation of the retired 4-D AHP composite, not an independently invented weighting.** `r_alpha=0.36` and `(q_reliability=0.80, q_maintainability=0.20)` are DECLARED constants algebraically derived from the old AHP vector (A=0.43, R=0.24, M=0.17, V=0.16) by dropping Vulnerability and renormalising — `w_R*r_alpha ≈ 0.24/0.84`, `w_R*(1-r_alpha) ≈ 0.43/0.84`, `w_M ≈ 0.17/0.84`, each within `abs=0.003` of the rounded 2-s.f. shipped values. Do not re-tune these independently of that derivation. Enforced by `tests/test_quality_model.py::TestCompositeReparameterisation`.
- **The Topic QoS sub-weights are AHP-consistent, not arbitrary.** `QoSPolicy.W_RELIABILITY=0.30`/`W_DURABILITY=0.40`/`W_PRIORITY=0.30` in [saag/core/models.py](saag/core/models.py) (the runtime source of truth for `w(topic)`) must match the geometric-mean priority vector of `AHPMatrices.criteria_topic_qos`'s Saaty pairwise-comparison matrix within `abs=0.01`, and that matrix's consistency ratio must stay below 0.10. Do not edit either side (the constants or the matrix) without keeping the other in sync. Enforced by `tests/test_ahp_shrinkage.py::test_topic_qos_matrix_reproduces_shipped_weights`.

## Known Gotchas

Documented so they aren't mistaken for correctly-working code; not fixed in this pass:

| Symptom | Cause |
|:---|:---|
| Editing `.env` doesn't change local `python cli/*.py` behavior | Nothing in the Python code loads `.env` — it is consumed only by Docker Compose and Next.js |
| Every Library in a real-world scenario graph scores `CQP = 0.70` (near-maximal Maintainability penalty) despite carrying no `code_metrics` | `StructuralAnalyzer._compute_code_quality_metrics`'s min-max helper returns `1.0` for any zero-variance population — correct for a genuine single real-valued node, but indistinguishable from "many nodes, uniformly absent data." All six real-world scenario files (Autoware, Train-Ticket, Cloud-microservices) hit this; measured to flip the Maintainability tier of every affected Library and, via the shared box-plot fence, ~20–27% of every other component in those graphs, while leaving `ρ(Q, I*)` on the Application population unchanged (Δρ = 0.0000). See [criticality.md §3.0](docs/criticality.md#30-three-quality-views-internal-external-and-quality-in-use). Not fixed here — needs the guard to distinguish the two cases, which the current `_mm` signature can't do from normalised values alone. |
| `cm_avg_cbo` and `cm_avg_rfc` show up in the UI and in exported graphs but never move a criticality score | Ingested, flattened and persisted, but no scoring code reads them — `CQP(v)` only consumes `loc`, `cyclomatic_complexity`, `instability_code` (from `avg_fanin`/`avg_fanout`), and `lcom`. See [structural-analysis.md §11.2](docs/structural-analysis.md#112-rm-formulas). For the full scoring/detection/GNN-feature/descriptive role of every `StructuralMetrics` field, see [`saag/core/metric_registry.py`](saag/core/metric_registry.py) rather than re-deriving it by hand. |

REV and RCL (Reverse Eigenvector, Reverse Closeness) were `V(v)`'s other two terms and are no longer computed at all — not a gotcha, just gone. Do not add `w_in`/QoS-weighted In-Degree to this table: `dependency_weight_in` was retired from its `V(v)`/QADS role but was repurposed, not orphaned — it is a live Topic-only `FT(v)` input (`saag/analysis/analyzer.py`'s Topic branch of `_compute_rm`). See [structural-analysis.md §9.13](docs/structural-analysis.md#913-qos-weighted-in-degree-w_in).
