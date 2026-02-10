# Software and System Test Document

## Software-as-a-Graph

### Graph-Based Critical Component Prediction for Distributed Publish-Subscribe Systems

**Version 2.0** · **February 2026**

Istanbul Technical University, Computer Engineering Department

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Test Strategy](#2-test-strategy)
3. [Test Environment](#3-test-environment)
4. [Unit Tests](#4-unit-tests)
5. [Integration Tests](#5-integration-tests)
6. [System Tests](#6-system-tests)
7. [Performance and Scalability Tests](#7-performance-and-scalability-tests)
8. [Validation Tests](#8-validation-tests)
9. [Acceptance Criteria](#9-acceptance-criteria)
10. [Traceability Matrix](#10-traceability-matrix)
11. [Appendices](#11-appendices)

---

## 1. Introduction

### 1.1 Purpose

This document specifies how the Software-as-a-Graph framework is tested. It defines the test strategy, test cases, pass criteria, and procedures for verifying that the system meets the requirements in the SRS — from individual function correctness up through end-to-end pipeline accuracy.

### 1.2 Scope

Testing spans six levels, each targeting a different concern:

| Level | What It Verifies | Section |
|-------|-----------------|---------|
| Unit | Individual functions compute correct results | [§4](#4-unit-tests) |
| Integration | Modules compose correctly through the pipeline | [§5](#5-integration-tests) |
| System | End-to-end pipeline produces expected outputs | [§6](#6-system-tests) |
| Performance | Analysis completes within time budgets at each scale | [§7](#7-performance-and-scalability-tests) |
| Validation | Predictions statistically match simulation ground truth | [§8](#8-validation-tests) |
| Acceptance | User-facing requirements are satisfied | [§9](#9-acceptance-criteria) |

### 1.3 References

| Document | Description |
|----------|-------------|
| SRS v2.0 | Software Requirements Specification |
| SDD v2.0 | Software Design Description |
| IEEE 829-2008 | Standard for Software Test Documentation |
| IEEE 1012-2016 | Standard for System and Software Verification and Validation |

### 1.4 Glossary

| Term | Definition |
|------|------------|
| Fixture | Predefined test data created before a test runs |
| Mock | Simulated object that isolates the code under test |
| SUT | System Under Test |
| TP / FP / TN / FN | True/False Positive/Negative (classification outcomes) |

---

## 2. Test Strategy

### 2.1 Test Pyramid

The project follows the standard test pyramid: many fast unit tests at the base, fewer integration and system tests above, and targeted validation and acceptance tests at the top.

| Level | Distribution | Speed | Infrastructure |
|-------|-------------|-------|---------------|
| Unit | ~70% of tests | Milliseconds | None (pure Python) |
| Integration | ~20% of tests | Seconds | Neo4j (Docker) |
| System | ~8% of tests | Seconds–minutes | Neo4j + full CLI |
| Acceptance | ~2% of tests | Minutes | Full environment |

### 2.2 Entry and Exit Criteria

**Tests can begin when:**
code compiles without errors, all unit tests pass locally, and the test Neo4j instance is available.

**Release is approved when:**
all planned tests are executed, no critical or high-severity defects remain open, unit test coverage ≥ 80%, all primary validation targets pass at the application layer, and performance benchmarks are met.

### 2.3 Test Schedule

| Phase | When | Duration |
|-------|------|----------|
| Unit tests | Continuous (TDD) | Ongoing |
| Integration tests | After module completion | 1 week |
| System tests | After integration pass | 1 week |
| Performance + Validation | After system stability | 3 days each |
| Acceptance | Before release | 2 days |

---

## 3. Test Environment

### 3.1 Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 8 GB | 16 GB |
| Storage | 10 GB SSD | 50 GB SSD |

### 3.2 Software Stack

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.9+ | Runtime |
| pytest | 7.0+ | Test framework |
| pytest-cov | 4.0+ | Coverage reporting |
| pytest-timeout | 2.0+ | Timeout handling |
| Neo4j | 5.x (Community) | Graph database |
| Docker | 20.10+ | Test database isolation |

### 3.3 pytest Configuration

```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --cov=src --cov-report=html
timeout = 120

markers =
    slow: marks tests as slow (skip with -m "not slow")
    integration: marks integration tests (requires Neo4j)
    performance: marks performance benchmarks
```

### 3.4 Test Database

A dedicated Neo4j instance runs on a different port to prevent interference with development data:

```yaml
# docker-compose.test.yml
services:
  neo4j-test:
    image: neo4j:5-community
    ports:
      - "7688:7687"    # Bolt (test port)
      - "7475:7474"    # HTTP (test port)
    environment:
      NEO4J_AUTH: neo4j/testpassword
      NEO4J_PLUGINS: '["graph-data-science"]'
```

### 3.5 Running Tests

```bash
# Unit tests (fast, no infrastructure)
pytest tests/ -m "not integration" -v

# Unit tests with coverage
pytest tests/ --cov=src --cov-report=html

# Integration tests (requires Neo4j)
docker-compose -f docker-compose.test.yml up -d
pytest tests/ -m integration -v
docker-compose -f docker-compose.test.yml down

# Specific module or pattern
pytest tests/test_analysis.py -v
pytest tests/ -k "test_quality" -v

# Skip slow tests
pytest tests/ -m "not slow" -v
```

---

## 4. Unit Tests

Unit tests verify individual functions and classes in isolation, without database access. They use deterministic fixtures (known graph topologies with predictable metric values) and target ≥ 80% line coverage per module.

### 4.1 Core Module — QoS Weight Calculation

Tests that QoS attributes produce correct edge weights, which drive the entire downstream analysis.

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| UT-CORE-01 | Default QoS (VOLATILE, BEST_EFFORT, MEDIUM) | Weight ≈ 0.10 |
| UT-CORE-02 | RELIABLE adds +0.30 | Weight ≈ 0.40 |
| UT-CORE-03 | PERSISTENT adds +0.40 | Weight ≈ 0.50 |
| UT-CORE-04 | URGENT adds +0.30 | Weight ≈ 0.40 |
| UT-CORE-05 | All maximum QoS settings combined | Weight ≈ 1.00 |
| UT-CORE-06 | Size score: min(log₂(1 + size/1024) / 10, 1.0) | Correct for 1KB, 8KB, 64KB, 1MB |
| UT-CORE-07 | Roundtrip: QoSPolicy → dict → QoSPolicy | All fields preserved |

```python
class TestQoSPolicy:
    def test_default_qos_weight(self):
        policy = QoSPolicy()
        assert policy.calculate_weight() == pytest.approx(0.1, abs=0.01)

    def test_highest_qos_weight(self):
        policy = QoSPolicy(
            reliability="RELIABLE", durability="PERSISTENT",
            transport_priority="URGENT"
        )
        assert policy.calculate_weight() == pytest.approx(1.0, abs=0.01)

    def test_size_score_formula(self):
        cases = [(1024, 0.1), (8192, 0.32), (65536, 0.60), (1048576, 1.0)]
        for size, expected in cases:
            score = min(math.log2(1 + size / 1024) / 10, 1.0)
            assert score == pytest.approx(expected, abs=0.05)
```

### 4.2 Analysis Module — Structural Metrics

Tests that centrality metrics are computed correctly on graphs with known properties.

**Common fixtures:**

```python
@pytest.fixture
def linear_graph():
    """A → B → C. Predictable PageRank and betweenness ordering."""
    return GraphData(
        components=[ComponentData("A", "Application"),
                    ComponentData("B", "Application"),
                    ComponentData("C", "Application")],
        edges=[EdgeData("A", "B", ..., "app_to_app", 1.0),
               EdgeData("B", "C", ..., "app_to_app", 2.0)]
    )

@pytest.fixture
def ap_graph():
    """A—B—C, B—D. B is the sole articulation point."""
    ...
```

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| UT-ANAL-01 | Metrics computed on linear graph | All 13 metrics present per component |
| UT-ANAL-02 | PageRank ordering: A → B → C | PR(C) ≥ PR(A) (downstream accumulates) |
| UT-ANAL-03 | Reverse PageRank ordering | RPR(A) ≥ RPR(C) (upstream accumulates) |
| UT-ANAL-04 | Articulation point detection | B identified as AP in bridge graph |
| UT-ANAL-05 | Bridge detection | Linear graph edges are bridges |
| UT-ANAL-06 | Empty graph | Empty result, no crash |
| UT-ANAL-07 | Single node | Zero betweenness, not an AP |
| UT-ANAL-08 | Metric normalization | All continuous metrics ∈ [0, 1] |

### 4.3 Analysis Module — Quality Scoring and Classification

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| UT-ANAL-10 | RMAV scores computed from structural metrics | R, M, A, V, Q all in [0, 1] |
| UT-ANAL-11 | Hub component has high R(v) | PageRank-dominant node scores highest R |
| UT-ANAL-12 | AP component has high A(v) | Articulation point scores highest A |
| UT-ANAL-13 | AHP weights differ from defaults | Different weights → different Q ordering |
| UT-ANAL-14 | AHP consistency check | CR > 0.10 triggers warning |
| UT-ANAL-15 | Box-plot classification | 5 levels assigned (CRITICAL through MINIMAL) |
| UT-ANAL-16 | Small sample fallback | n < 12 uses percentile thresholds |
| UT-ANAL-17 | Layer filtering | App layer includes only Application components |

```python
def test_layer_analysis_filters_correctly(self, multi_layer_graph):
    analyzer = StructuralAnalyzer()
    res_app = analyzer.analyze(multi_layer_graph, layer=AnalysisLayer.APP)
    for comp in res_app.components.values():
        assert comp.type == "Application"

    res_infra = analyzer.analyze(multi_layer_graph, layer=AnalysisLayer.INFRA)
    for comp in res_infra.components.values():
        assert comp.type == "Node"
```

### 4.4 Simulation Module

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| UT-SIM-01 | Load graph from GraphData | All components tracked as ACTIVE |
| UT-SIM-02 | Fail a component → state changes | Status becomes FAILED |
| UT-SIM-03 | Pub-sub path detection | Publisher → Topic → Subscriber paths found |
| UT-SIM-04 | Graph reset restores all state | All components ACTIVE after reset |
| UT-SIM-10 | Event simulation: messages flow | Subscribers receive messages from publishers |
| UT-SIM-20 | Failure simulation: basic cascade | Cascade propagates from target |
| UT-SIM-21 | Physical cascade: Node → hosted Apps | Hosted applications fail when Node fails |
| UT-SIM-22 | Cascade count accuracy | |F| matches number of failed components |
| UT-SIM-23 | Impact calculation: I(v) computed | Composite score ∈ [0, 1] |
| UT-SIM-24 | Custom impact weights | Different w_r, w_f, w_t → different I(v) |

```python
@pytest.fixture
def raw_graph_data():
    """App1 publishes to Topic1, App2 subscribes. App1 runs on Node1."""
    return GraphData(
        components=[ComponentData("App1", "Application"),
                    ComponentData("App2", "Application"),
                    ComponentData("Topic1", "Topic"),
                    ComponentData("Node1", "Node")],
        edges=[EdgeData("App1", "Topic1", ..., "PUBLISHES_TO", ...),
               EdgeData("App2", "Topic1", ..., "SUBSCRIBES_TO", ...),
               EdgeData("App1", "Node1", ..., "RUNS_ON", ...)]
    )

def test_physical_cascade(raw_graph_data):
    graph = SimulationGraph(graph_data=raw_graph_data)
    sim = FailureSimulator(graph)
    res = sim.simulate(FailureScenario("Node1", "test"))
    assert "App1" in res.cascaded_failures

def test_configurable_impact_weights(raw_graph_data):
    m1 = ImpactMetrics(reachability_loss=1.0, fragmentation=0.0, throughput_loss=0.0)
    m2 = ImpactMetrics(reachability_loss=1.0, fragmentation=0.0, throughput_loss=0.0,
                       impact_weights={"reachability": 1.0, "fragmentation": 0.0,
                                       "throughput": 0.0})
    assert m1.composite_impact == pytest.approx(0.4, abs=0.01)   # default weights
    assert m2.composite_impact == pytest.approx(1.0, abs=0.01)   # custom weights
```

### 4.5 Validation Module

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| UT-VAL-01 | Perfect positive correlation | ρ = 1.0, p < 0.05 |
| UT-VAL-02 | Perfect inverse correlation | ρ = −1.0 |
| UT-VAL-03 | Identical predictions | RMSE = 0.0, MAE = 0.0 |
| UT-VAL-04 | Matching top elements | Top-K overlap high, NDCG > 0.9 |
| UT-VAL-05 | Mismatched top elements | NDCG < 1.0 |
| UT-VAL-06 | Empty input | No crash, empty result returned |
| UT-VAL-07 | Single element | Warning issued, valid result |
| UT-VAL-08 | Pass/fail logic | Correct determination against targets |

```python
class TestCorrelationMetrics:
    def test_perfect_correlation(self):
        x = [0.1, 0.2, 0.3, 0.4, 0.5]
        y = [0.1, 0.2, 0.3, 0.4, 0.5]
        rho, p = spearman_correlation(x, y)
        assert rho == pytest.approx(1.0, abs=0.01)
        assert p < 0.05

    def test_inverse_correlation(self):
        rho, _ = spearman_correlation([0.1, 0.2, 0.3], [0.3, 0.2, 0.1])
        assert rho == pytest.approx(-1.0, abs=0.01)

    def test_rmse_zero_for_identical(self):
        metrics = calculate_error_metrics([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert metrics.rmse == pytest.approx(0.0, abs=0.001)
```

### 4.6 Visualization Module

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| UT-VIZ-01 | Dashboard generator produces valid HTML | Output parses as HTML, contains expected sections |
| UT-VIZ-02 | KPI cards render with correct values | Component count, critical count match input |
| UT-VIZ-03 | Network graph data serialized correctly | Nodes and edges in vis.js JSON format |
| UT-VIZ-04 | Default color theme values | CRITICAL = #E74C3C, HIGH = #E67E22, etc. |
| UT-VIZ-05 | Custom theme overrides | Overridden values applied, defaults preserved |

### 4.7 Coverage Targets

| Module | Unit Tests | Target Coverage |
|--------|-----------|----------------|
| src/domain/models | ~15 | 85% |
| src/domain/services | ~35 | 82% |
| src/infrastructure (simulation) | ~15 | 78% |
| src/infrastructure (validation) | ~12 | 80% |
| src/infrastructure (visualization) | ~10 | 75% |
| **Total** | **~87** | **≥ 80%** |

---

## 5. Integration Tests

Integration tests verify that modules compose correctly when data flows between them. These tests use the `@pytest.mark.integration` marker and require a running Neo4j instance.

### 5.1 Analysis Pipeline

Tests that StructuralAnalyzer → QualityAnalyzer → ProblemDetector produces correct end-to-end results.

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| IT-ANAL-01 | StructuralAnalyzer → QualityAnalyzer | RMAV scores computed from structural metrics |
| IT-ANAL-02 | QualityAnalyzer → ProblemDetector | Architectural problems identified |
| IT-ANAL-03 | Full analysis pipeline | LayerAnalysisResult with components, edges, problems |
| IT-ANAL-04 | Multi-layer analysis (all 5 layers) | MultiLayerAnalysisResult with per-layer results |

```python
@pytest.mark.integration
def test_full_analysis_pipeline(multi_layer_graph):
    structural = StructuralAnalyzer()
    quality = QualityAnalyzer()
    detector = ProblemDetector()

    struct_result = structural.analyze(multi_layer_graph)
    assert len(struct_result.components) > 0

    qual_result = quality.analyze(struct_result)
    assert len(qual_result.components) > 0
    assert all(c.scores.overall >= 0 for c in qual_result.components)

    problems = detector.detect(qual_result.components, qual_result.edges, qual_result)
    assert isinstance(problems, list)
```

### 5.2 Simulation Pipeline

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| IT-SIM-01 | Event simulation + failure simulation on same graph | Both produce valid results |
| IT-SIM-02 | Exhaustive simulation across all components | One FailureResult per component |
| IT-SIM-03 | Layer-specific simulation | Only layer components simulated |

### 5.3 Cross-Pipeline (Analysis → Simulation → Validation)

The most important integration test: does the full prediction-vs-reality loop work?

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| IT-VAL-01 | Analysis Q(v) compared against simulation I(v) | ValidationGroupResult with all metrics |
| IT-VAL-02 | Matched component count > 0 | Predicted and actual sets overlap |
| IT-VAL-03 | Pass/fail reflects validation targets | Correct pass/fail determination |

```python
@pytest.mark.integration
def test_full_validation_pipeline(multi_layer_graph):
    # Analysis
    struct_result = StructuralAnalyzer().analyze(multi_layer_graph)
    qual_result = QualityAnalyzer().analyze(struct_result)

    # Simulation
    sim_graph = SimulationGraph(graph_data=multi_layer_graph)
    sim_results = FailureSimulator(sim_graph).simulate_exhaustive()

    # Validation
    predicted = {c.id: c.scores.overall for c in qual_result.components}
    actual = {r.target_id: r.impact.composite_impact for r in sim_results}
    result = Validator().validate(predicted, actual)

    assert result.matched_count > 0
    assert -1.0 <= result.correlation.spearman <= 1.0
```

### 5.4 Neo4j Integration

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| IT-NEO-01 | Import → Export roundtrip | All entities and edges preserved |
| IT-NEO-02 | DEPENDS_ON derivation | Derived edges exist with correct types |
| IT-NEO-03 | QoS weight propagation | Non-zero weights on topics and edges |
| IT-NEO-04 | Layer extraction query | Returns only components of the requested layer |

---

## 6. System Tests

System tests exercise the complete pipeline through the CLI tools, verifying end-to-end behavior at multiple scales.

### 6.1 End-to-End Pipeline

| Test ID | Scale | Description | Pass Criteria |
|---------|-------|-------------|---------------|
| ST-E2E-01 | Small | Full pipeline (~10–25 components) | All 6 steps complete, dashboard generated |
| ST-E2E-02 | Medium | Full pipeline (~30–50 components) | All 6 steps complete, validation passes |
| ST-E2E-03 | Large | Full pipeline (~60–100 components) | All 6 steps complete within time budget |

**Procedure (example: small scale):**

```bash
python bin/generate_graph.py --scale small --output /tmp/test_data.json
python bin/import_graph.py --input /tmp/test_data.json --clear
python bin/analyze_graph.py --layer system --use-ahp --output /tmp/analysis.json
python bin/simulate_graph.py failure --layer system --exhaustive --output /tmp/simulation.json
python bin/validate_graph.py --layer system --output /tmp/validation.json
python bin/visualize_graph.py --layer system --output /tmp/dashboard.html
```

Or as a single command:

```bash
python bin/run.py --all --layer system --scale small
```

### 6.2 CLI Command Tests

Each CLI tool is tested independently with its most common options.

| Test ID | Command | Tested Options | Pass Criteria |
|---------|---------|---------------|---------------|
| ST-CLI-01 | `generate_graph.py` | `--scale tiny/small/medium/large/xlarge` | JSON output for each scale |
| ST-CLI-02 | `import_graph.py` | `--input FILE --clear` | Import completes, entity counts logged |
| ST-CLI-03 | `analyze_graph.py` | `--layer app/infra/system`, `--use-ahp`, `--output` | Analysis JSON produced |
| ST-CLI-04 | `simulate_graph.py` | `failure --exhaustive`, `--monte-carlo` | Simulation results produced |
| ST-CLI-05 | `validate_graph.py` | `--layer app`, `--output` | Validation JSON with pass/fail |
| ST-CLI-06 | `visualize_graph.py` | `--output FILE`, `--open` | Valid HTML dashboard |
| ST-CLI-07 | `run.py` | `--all --layer system` | Full pipeline completes |
| ST-CLI-08 | `benchmark.py` | `--scales small,medium --runs 3` | Benchmark report generated |

### 6.3 Error Handling

| Test ID | Scenario | Expected Behavior |
|---------|----------|-------------------|
| ST-ERR-01 | Non-existent input file | Graceful error message, non-zero exit code |
| ST-ERR-02 | Malformed JSON | Parse error with file/line info |
| ST-ERR-03 | Neo4j unavailable | Connection error message, no stack trace |
| ST-ERR-04 | Empty input topology | Warning logged, empty result returned |
| ST-ERR-05 | Invalid layer name | Error listing valid layer options |

---

## 7. Performance and Scalability Tests

Performance tests verify that analysis completes within time budgets and that prediction accuracy improves with system scale (a key thesis contribution).

### 7.1 Timing Targets

| Scale | Components | Max Analysis Time | Max Simulation Time | Max Dashboard Time |
|-------|------------|-------------------|--------------------|--------------------|
| Tiny | 5–10 | < 0.5 s | < 1 s | < 5 s |
| Small | 10–25 | < 1 s | < 2 s | < 5 s |
| Medium | 30–50 | < 5 s | < 10 s | < 10 s |
| Large | 60–100 | < 10 s | < 20 s | < 10 s |
| XLarge | 150–300 | < 30 s | < 60 s | < 10 s |

### 7.2 Resource Targets

| Metric | Target |
|--------|--------|
| Peak memory (large scale) | < 2 GB |
| Database import (medium) | < 5 s |
| Peak memory (xlarge) | < 4 GB |

### 7.3 Benchmark Execution

The `benchmark.py` tool runs the full pipeline at each scale × layer combination, repeating N times with different random seeds to measure variance:

```bash
# Quick benchmark (1 run per configuration)
python bin/benchmark.py --scales small,medium,large --runs 1

# Full benchmark suite (5 runs for statistical stability)
python bin/benchmark.py --scales tiny,small,medium,large,xlarge \
    --layers app,infra,system --runs 5 --output results/benchmark
```

Outputs: `benchmark_data.csv` (raw records), `benchmark_results.json` (aggregated), `benchmark_report.md` (human-readable).

Each benchmark record captures: timing per pipeline step, graph statistics (nodes, edges, density), all validation metrics (Spearman ρ, F1, precision, recall, RMSE, Top-K), and pass/fail status.

---

## 8. Validation Tests

Validation tests are the most important tests for the research contribution. They verify that topology-based predictions (Q(v)) correlate with actual failure impact (I(v)), demonstrating that the methodology works.

### 8.1 Primary Validation Targets

| Metric | Target | Priority | Rationale |
|--------|--------|----------|-----------|
| Spearman ρ | ≥ 0.70 | Primary | Predicted and actual rankings agree |
| p-value | ≤ 0.05 | Primary | Correlation is statistically significant |
| F1-Score | ≥ 0.80 | Primary | Balanced precision and recall |
| Top-5 Overlap | ≥ 40% | Primary | Agreement on the most critical components |
| RMSE | ≤ 0.25 | Secondary | Bounded prediction error |
| Precision | ≥ 0.80 | Reported | Minimized false alarms |
| Recall | ≥ 0.80 | Reported | All critical components caught |
| Cohen's κ | ≥ 0.60 | Reported | Chance-corrected agreement |
| Top-10 Overlap | ≥ 50% | Reported | Extended critical set agreement |
| MAE | ≤ 0.20 | Reported | Bounded absolute error |

### 8.2 Validation Matrix (Layer × Scale)

| Test ID | Layer | Scale | Target ρ | Target F1 |
|---------|-------|-------|----------|-----------|
| VT-APP-01 | Application | Small | ≥ 0.75 | ≥ 0.75 |
| VT-APP-02 | Application | Medium | ≥ 0.80 | ≥ 0.80 |
| VT-APP-03 | Application | Large | ≥ 0.85 | ≥ 0.83 |
| VT-INF-01 | Infrastructure | Small | ≥ 0.50 | ≥ 0.65 |
| VT-INF-02 | Infrastructure | Medium | ≥ 0.52 | ≥ 0.66 |
| VT-INF-03 | Infrastructure | Large | ≥ 0.54 | ≥ 0.68 |
| VT-SYS-01 | System | Small | ≥ 0.70 | ≥ 0.75 |
| VT-SYS-02 | System | Medium | ≥ 0.75 | ≥ 0.80 |
| VT-SYS-03 | System | Large | ≥ 0.80 | ≥ 0.83 |

Note: Application layer targets are higher because application-level dependencies are more directly captured by topology. Infrastructure layer targets are lower — an expected limitation discussed in the thesis.

### 8.3 Achieved Results

**By layer (large scale):**

| Metric | Application | Infrastructure | Target |
|--------|-------------|----------------|--------|
| Spearman ρ | **0.85** ✓ | 0.54 | ≥ 0.70 |
| F1-Score | **0.83** ✓ | 0.68 | ≥ 0.80 |
| Precision | **0.86** ✓ | 0.71 | ≥ 0.80 |
| Recall | **0.80** ✓ | 0.65 | ≥ 0.80 |
| Top-5 | **62%** ✓ | 40% | ≥ 40% |

**By scale (application layer):**

| Scale | Components | Spearman ρ | F1-Score |
|-------|------------|------------|----------|
| Small | 10–25 | 0.78 | 0.75 |
| Medium | 30–50 | 0.82 | 0.80 |
| Large | 60–100 | 0.85 | 0.83 |
| XLarge | 150–300 | 0.88 | 0.85 |

**Key finding:** prediction accuracy improves with system scale. Larger systems produce more stable centrality distributions, leading to more reliable correlation.

### 8.4 Validation Procedure

```bash
# Deterministic validation with fixed seed
python bin/generate_graph.py --scale medium --seed 42 --output test_data.json
python bin/import_graph.py --input test_data.json --clear
python bin/analyze_graph.py --layer app --use-ahp --output analysis.json
python bin/simulate_graph.py failure --layer app --exhaustive --output simulation.json
python bin/validate_graph.py --layer app --output validation_result.json

# Check pass/fail
cat validation_result.json | jq '.overall.passed'
# Expected: true
```

---

## 9. Acceptance Criteria

Each user-facing capability has specific acceptance criteria. All automated criteria are verified by system tests; manual criteria are verified during acceptance testing.

### 9.1 Feature Acceptance

#### Graph Model Construction

| ID | Criterion | Method | Pass If |
|----|-----------|--------|---------|
| AC-01 | Import JSON topology | Auto | All entities appear in Neo4j |
| AC-02 | Derive DEPENDS_ON edges | Auto | Derived edges exist with correct types |
| AC-03 | Calculate QoS weights | Auto | Topic weights > 0 for non-default QoS |
| AC-04 | Support all preset scales | Auto | tiny through xlarge generate without error |

#### Structural Analysis

| ID | Criterion | Method | Pass If |
|----|-----------|--------|---------|
| AC-05 | Compute all 13 metrics | Auto | Non-zero values for non-trivial graphs |
| AC-06 | Identify articulation points | Auto | Known SPOFs detected on test graph |
| AC-07 | Multi-layer support | Auto | All 5 layers produce results |
| AC-08 | Export results to JSON | Auto | Valid JSON with expected schema |

#### Quality Scoring

| ID | Criterion | Method | Pass If |
|----|-----------|--------|---------|
| AC-09 | RMAV scores present | Auto | All 4 dimensions + overall in output |
| AC-10 | AHP weights differ from defaults | Auto | Custom matrix produces different Q ordering |
| AC-11 | 5-level classification | Auto | CRITICAL, HIGH, MEDIUM, LOW, MINIMAL assigned |
| AC-12 | Critical components detected | Auto | At least 1 CRITICAL in medium+ scale |

#### Failure Simulation

| ID | Criterion | Method | Pass If |
|----|-----------|--------|---------|
| AC-13 | Cascade propagation | Auto | |F| > 1 when Node fails |
| AC-14 | Impact score computed | Auto | I(v) ∈ [0, 1] for all components |
| AC-15 | Exhaustive mode | Auto | One result per component in layer |
| AC-16 | Results sorted by impact | Auto | Descending I(v) order |

#### Validation

| ID | Criterion | Method | Pass If |
|----|-----------|--------|---------|
| AC-17 | Spearman ρ computed | Auto | Value ∈ [−1, 1] |
| AC-18 | F1-Score computed | Auto | Value ∈ [0, 1] |
| AC-19 | Pass/fail determined | Auto | Boolean result matches target comparison |
| AC-20 | Accuracy targets met | Auto | ρ ≥ 0.70 and F1 ≥ 0.80 at app layer |

#### Visualization

| ID | Criterion | Method | Pass If |
|----|-----------|--------|---------|
| AC-21 | HTML dashboard generated | Auto | Valid HTML file with expected sections |
| AC-22 | KPI cards correct | Manual | Counts match analysis results |
| AC-23 | Charts render | Manual | Pie and bar charts visible in browser |
| AC-24 | Network graph interactive | Manual | vis.js: hover, click, drag, zoom work |

### 9.2 Acceptance Checklist

| ID | Requirement | Criteria | Status |
|----|-------------|----------|--------|
| ACC-01 | Import JSON topology | AC-01, AC-02, AC-03 | ☐ |
| ACC-02 | Compute all metrics | AC-05, AC-06 | ☐ |
| ACC-03 | RMAV quality scoring | AC-09, AC-10, AC-11 | ☐ |
| ACC-04 | Failure simulation | AC-13, AC-14, AC-15 | ☐ |
| ACC-05 | Validation accuracy | AC-17 – AC-20 | ☐ |
| ACC-06 | Dashboard generation | AC-21 – AC-24 | ☐ |
| ACC-07 | Performance (< 20 s for large) | §7.1 timing targets | ☐ |
| ACC-08 | Multi-layer analysis | AC-07 | ☐ |
| ACC-09 | CLI usability | ST-CLI-01 – ST-CLI-08 | ☐ |
| ACC-10 | Documentation complete | All docs reviewed | ☐ |

---

## 10. Traceability Matrix

Each SRS requirement maps to one or more test cases. Requirements use IDs from SRS v2.0.

| Requirement | Description | Test IDs |
|-------------|-------------|----------|
| REQ-GM-01 | Accept JSON topology | UT-CORE-07, ST-CLI-02 |
| REQ-GM-02 | Create 5 vertex types | IT-NEO-01 |
| REQ-GM-04 | Derive DEPENDS_ON edges | IT-NEO-02 |
| REQ-GM-05 | Compute QoS weights | UT-CORE-01 – UT-CORE-06 |
| REQ-GM-07 | Layer projection | UT-ANAL-17, IT-NEO-04 |
| REQ-SA-01 | Compute PageRank | UT-ANAL-02 |
| REQ-SA-02 | Compute Reverse PageRank | UT-ANAL-03 |
| REQ-SA-08 | Articulation point detection | UT-ANAL-04 |
| REQ-SA-10 | Normalize to [0, 1] | UT-ANAL-08 |
| REQ-QS-01 – 04 | Compute RMAV scores | UT-ANAL-10, UT-ANAL-11, UT-ANAL-12 |
| REQ-QS-07 | Support AHP weights | UT-ANAL-13, UT-ANAL-14 |
| REQ-QS-09 | Box-plot classification | UT-ANAL-15, UT-ANAL-16 |
| REQ-FS-01 | Simulate CRASH mode | UT-SIM-20 |
| REQ-FS-03 | Cascade propagation rules | UT-SIM-21, UT-SIM-22 |
| REQ-FS-07 | Composite impact I(v) | UT-SIM-23, UT-SIM-24 |
| REQ-VA-01 | Spearman ρ | UT-VAL-01, UT-VAL-02, IT-VAL-01 |
| REQ-VA-02 | F1-Score | UT-VAL-08, IT-VAL-03 |
| REQ-VZ-01 | HTML dashboard | UT-VIZ-01, ST-CLI-06 |
| REQ-VZ-03 | Interactive network graph | UT-VIZ-03, AC-24 |
| REQ-PERF-01 – 03 | Analysis timing | §7.1 timing targets, ST-E2E-01 – 03 |
| REQ-ACC-01 – 02 | Accuracy targets | VT-APP-01 – VT-SYS-03 |

---

## 11. Appendices

### Appendix A: Synthetic Graph Scale Specifications

| Scale | Apps | Brokers | Topics | Nodes | Libraries | Total Components |
|-------|------|---------|--------|-------|-----------|-----------------|
| Tiny | 5–8 | 1 | 3–5 | 2–3 | 2 | 13–19 |
| Small | 10–15 | 2 | 8–12 | 3–4 | 3 | 26–36 |
| Medium | 20–35 | 3 | 15–25 | 5–8 | 5 | 48–76 |
| Large | 50–80 | 5 | 30–50 | 8–12 | 8 | 101–155 |
| XLarge | 100–200 | 10 | 60–100 | 15–25 | 15 | 200–350 |

### Appendix B: CI/CD Pipeline

```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: { python-version: '3.9' }
      - run: pip install -r requirements.txt -r requirements-test.txt
      - run: pytest tests/ -m "not integration" --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v3

  integration-tests:
    runs-on: ubuntu-latest
    services:
      neo4j:
        image: neo4j:5-community
        ports: ['7687:7687']
        env: { NEO4J_AUTH: 'neo4j/testpassword' }
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with: { python-version: '3.9' }
      - run: pip install -r requirements.txt -r requirements-test.txt
      - run: sleep 30 && pytest tests/ -m integration -v
        env:
          NEO4J_URI: bolt://localhost:7687
          NEO4J_PASSWORD: testpassword

  benchmark:
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests]
    services:
      neo4j:
        image: neo4j:5-community
        ports: ['7687:7687']
        env: { NEO4J_AUTH: 'neo4j/testpassword' }
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: python bin/benchmark.py --scales small,medium --runs 3
      - uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: output/benchmark_*.json
```

### Appendix C: Defect Severity Classification

| Severity | Definition | Examples |
|----------|------------|---------|
| Critical | System unusable | Crash on startup, data loss, silent wrong results |
| High | Major feature broken | Analysis fails on valid input, cascade logic error |
| Medium | Feature partially works | Minor calculation error, formatting issue in dashboard |
| Low | Cosmetic | UI alignment, log message typos |

---

*Software-as-a-Graph Framework v2.0 · February 2026*