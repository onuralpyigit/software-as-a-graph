# Software and System Test Document

## Software-as-a-Graph

### Graph-Based Critical Component Prediction for Distributed Publish-Subscribe Systems

**Version 2.1** · **February 2026**

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

This document specifies how the Software-as-a-Graph framework is tested. It defines the test strategy, test cases, pass criteria, and procedures for verifying that the system meets the requirements in the SRS — from individual function correctness up through end-to-end pipeline accuracy and Genieus web application behaviour.

### 1.2 Scope

Testing spans six levels, each targeting a different concern:

| Level | What It Verifies | Section |
|-------|-----------------|---------|
| Unit | Individual functions compute correct results | [§4](#4-unit-tests) |
| Integration | Modules compose correctly through the pipeline | [§5](#5-integration-tests) |
| System | End-to-end pipeline and web application produce expected outputs | [§6](#6-system-tests) |
| Performance | Analysis completes within time budgets at each scale | [§7](#7-performance-and-scalability-tests) |
| Validation | Predictions statistically match simulation ground truth | [§8](#8-validation-tests) |
| Acceptance | All user-facing requirements are satisfied | [§9](#9-acceptance-criteria) |

Coverage spans both delivery mechanisms: the **CLI pipeline** (`bin/`) and the **Genieus web application** (FastAPI backend + Next.js frontend).

### 1.3 References

| Document | Description |
|----------|-------------|
| SRS v2.1 | Software Requirements Specification |
| SDD v2.1 | Software Design Description |
| IEEE 829-2008 | Standard for Software Test Documentation |
| IEEE 1012-2016 | Standard for System and Software Verification and Validation |
| IEEE RASSE 2025 | Published methodology paper (doi: 10.1109/RASSE64831.2025.11315354) |

### 1.4 Document Conventions

- Test IDs follow the pattern `<LEVEL>-<MODULE>-<NN>` (e.g., `UT-ANAL-01`, `IT-NEO-01`, `ST-E2E-01`, `VT-APP-01`, `AC-01`).
- The marker `@pytest.mark.<tag>` indicates the pytest marker used to select or exclude the test.
- Pass criteria use **shall** language matching the SRS requirement they verify.
- Requirement cross-references use IDs from SRS v2.1 (e.g., REQ-GM-01).

### 1.5 Document Overview

Section 2 describes the overall test strategy and schedule. Section 3 defines the test environment including software stack, database setup, and API test configuration. Sections 4–6 specify unit, integration, and system tests respectively. Section 7 covers performance and benchmark tests. Section 8 is the validation test suite — the most important section for the research contribution. Section 9 defines acceptance criteria for all user-facing capabilities including the Genieus web application. Section 10 provides the full SRS-to-test traceability matrix. Appendices cover scale specifications, CI/CD configuration, and defect severity classification.

### 1.6 Glossary

| Term | Definition |
|------|------------|
| AP\_c | Continuous articulation point score — fraction of graph fragmented upon vertex removal |
| BR | Bridge Ratio — fraction of a vertex's incident edges that are bridges |
| CR | Consistency Ratio in AHP (must be < 0.10) |
| Fixture | Predefined test data created before a test runs |
| Mock | Simulated object that isolates the code under test |
| NDCG | Normalized Discounted Cumulative Gain — ranking quality metric |
| ρ | Spearman rank correlation coefficient |
| RMAV | Reliability, Maintainability, Availability, Vulnerability |
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
| System | ~8% of tests | Seconds–minutes | Neo4j + full CLI / Docker stack |
| Acceptance | ~2% of tests | Minutes | Full environment (CLI + Web) |

### 2.2 Entry and Exit Criteria

**Tests can begin when:**
code compiles without errors, all unit tests pass locally, and the test Neo4j instance is reachable.

**Release is approved when:**
all planned tests are executed, no Critical or High-severity defects remain open, unit test coverage ≥ 80% per module, all primary validation targets pass at the application layer, performance benchmarks are met, and all web application acceptance criteria pass.

### 2.3 Test Schedule

| Phase | Milestone Trigger | Duration | Output |
|-------|------------------|----------|--------|
| Unit tests | Continuous (TDD) | Ongoing | Coverage report, CI badge |
| Integration tests | After each module completes | 1 week | Integration test report |
| System tests | After integration suite passes | 1 week | End-to-end test report |
| Performance + Validation | After system stability | 3 days each | Benchmark CSV + validation JSON |
| Acceptance | Before ICSA 2026 submission | 2 days | Signed acceptance checklist |
| Regression | After each significant change | Per CI run | Regression report |

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
| Python | 3.9+ | Runtime and test execution |
| pytest | 7.0+ | Test framework |
| pytest-cov | 4.0+ | Coverage reporting |
| pytest-timeout | 2.0+ | Timeout enforcement |
| pytest-asyncio | 0.21+ | Async FastAPI endpoint tests |
| httpx | 0.24+ | HTTP client for REST API tests |
| Node.js | 20+ | Next.js frontend build and testing |
| Neo4j | 5.x Community | Graph database |
| Docker | 20.10+ | Test database and full stack isolation |
| Docker Compose | 2.x | Full stack orchestration |

### 3.3 pytest Configuration

```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
timeout = 120

markers =
    slow: marks tests as slow (skip with --quick)
    integration: marks integration tests
```

### 3.4 Test Database

A dedicated Neo4j instance runs on separate ports to prevent interference with development data:

> **Note:** `docker-compose.test.yml` is not yet present in the repository; create this file before running integration tests.

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
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:7474"]
      interval: 10s
      timeout: 5s
      retries: 10
```

### 3.5 Running Tests

```bash
# Unit tests (fast, no infrastructure required)
pytest tests/ -m "not integration and not api" -v

# Unit tests with coverage report
pytest tests/ -m "not integration and not api" --cov=src --cov-report=html

# Integration tests (requires Neo4j)
docker compose -f docker-compose.test.yml up -d
pytest tests/ -m integration -v
docker compose -f docker-compose.test.yml down

# REST API tests (requires full Docker stack)
docker compose up -d --build
pytest tests/ -m api -v
docker compose down

# Specific module or pattern
pytest tests/test_analysis_service.py -v
pytest tests/ -k "test_quality" -v

# Skip slow tests
pytest tests/ -m "not slow" -v
```

### 3.6 API Test Environment

REST API tests target the FastAPI backend running in the full Docker stack. A shared `httpx.AsyncClient` fixture is configured for all `@pytest.mark.api` tests:

```python
# tests/conftest.py
import pytest
import httpx

BASE_URL = "http://localhost:8000"

@pytest.fixture(scope="session")
async def api_client():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=30.0) as client:
        # Wait for backend to be ready
        for _ in range(10):
            try:
                r = await client.get("/health")
                if r.status_code == 200:
                    break
            except httpx.ConnectError:
                await asyncio.sleep(2)
        yield client
```

All API tests use `@pytest.mark.asyncio` and `@pytest.mark.api`.

---

## 4. Unit Tests

Unit tests verify individual functions and classes in isolation, without database access. They use deterministic fixtures (known graph topologies with predictable metric values) and target ≥ 80% line coverage per module.

### 4.1 Core Module — QoS Weight Calculation

Tests that QoS attributes produce correct edge weights, which drive all downstream analysis.

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| UT-CORE-01 | Default QoS (VOLATILE, BEST\_EFFORT, LOW) | Weight ≈ 0.00 |
| UT-CORE-02 | RELIABLE adds +0.30 | Weight ≈ 0.30 |
| UT-CORE-03 | PERSISTENT adds +0.40 | Weight ≈ 0.40 |
| UT-CORE-04 | URGENT adds +0.30 | Weight ≈ 0.30 |
| UT-CORE-05 | All maximum QoS settings combined | Weight ≈ 1.00 |
| UT-CORE-06 | Size score formula: min(log₂(1+size/1024)/10, 1.0) | Correct for 1 KB, 8 KB, 64 KB, 1 MB |
| UT-CORE-07 | JSON topology roundtrip: parse → re-serialize | All fields preserved, IDs unchanged |
| UT-CORE-08 | TRANSIENT\_LOCAL durability → +0.20 increment | Weight = 0.20 (durability only) |
| UT-CORE-09 | HIGH priority → +0.20, MEDIUM → +0.10 | Correct intermediate increments |

```python
class TestQoSPolicy:
    def test_default_qos_weight(self):
        policy = QoSPolicy()  # VOLATILE, BEST_EFFORT, LOW
        assert policy.calculate_weight() == pytest.approx(0.0, abs=0.01)

    def test_highest_qos_weight(self):
        policy = QoSPolicy(
            reliability="RELIABLE", durability="PERSISTENT",
            transport_priority="URGENT"
        )
        assert policy.calculate_weight() == pytest.approx(1.0, abs=0.01)

    def test_size_score_formula(self):
        cases = [(1024, 0.1), (8192, 0.32), (65536, 0.60), (1_048_576, 1.0)]
        for size, expected in cases:
            score = min(math.log2(1 + size / 1024) / 10, 1.0)
            assert score == pytest.approx(expected, abs=0.05)

    def test_intermediate_durability(self):
        policy = QoSPolicy(durability="TRANSIENT_LOCAL")
        assert policy.calculate_weight() == pytest.approx(0.20, abs=0.01)
```

### 4.2 Analysis Module — Structural Metrics

Tests that centrality and resilience metrics are computed correctly on graphs with known topological properties.

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
    """A — B — C, B — D. B is the sole articulation point."""
    # Removing B disconnects {A} from {C, D}

@pytest.fixture
def star_graph():
    """Hub H connected to A, B, C, D. H has maximum betweenness."""
```

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| UT-ANAL-01 | Metrics computed on linear graph | All 16 fields of StructuralMetrics present per component |
| UT-ANAL-02 | PageRank ordering on A→B→C | PR(C) ≥ PR(B) ≥ PR(A) (downstream accumulates) |
| UT-ANAL-03 | Reverse PageRank ordering | RPR(A) ≥ RPR(B) ≥ RPR(C) (upstream accumulates) |
| UT-ANAL-04 | Betweenness: B is bottleneck on A→B→C | BT(B) > BT(A) and BT(B) > BT(C) |
| UT-ANAL-05 | Closeness: hub scores highest | CL(H) > CL(leaf) in star graph |
| UT-ANAL-06 | Eigenvector: connected-to-hubs scores higher | EV(connected\_to\_hub) > EV(leaf) |
| UT-ANAL-07 | Articulation point detection (binary) | B identified as AP in ap\_graph; A, C, D are not |
| UT-ANAL-08 | AP\_c continuous score: B in ap\_graph | AP\_c(B) > 0; AP\_c(A) = 0 |
| UT-ANAL-09 | AP\_c magnitude: half-split graph | AP\_c(cut\_vertex) ≈ 0.5 on balanced bipartition |
| UT-ANAL-10 | Bridge detection on linear graph | All edges are bridges (BR = 1.0 for all vertices) |
| UT-ANAL-11 | Bridge detection on cycle graph | No bridges (BR = 0.0 for all vertices) |
| UT-ANAL-12 | Empty graph | Empty result, no exception raised |
| UT-ANAL-13 | Single-node graph | Zero betweenness, is\_articulation\_point = False, AP\_c = 0 |
| UT-ANAL-14 | Min-max normalization | All continuous metrics ∈ [0, 1] after normalization |
| UT-ANAL-15 | Uniform distribution edge case | All identical raw values → all normalized to 0.0 |
| UT-ANAL-16 | Graph-level summary statistics | vertex\_count, edge\_count, density, diameter all present and correct |

```python
def test_ap_c_continuous_score(self, ap_graph):
    analyzer = StructuralAnalyzer()
    result = analyzer.analyze(ap_graph)
    B = result.components["B"]
    A = result.components["A"]
    assert B.ap_score > 0, "B is an AP; ap_score must be positive"
    assert A.ap_score == pytest.approx(0.0, abs=0.001), "A is not an AP"
    assert B.is_articulation_point is True
    assert A.is_articulation_point is False

def test_bridge_ratio_linear(self, linear_graph):
    result = StructuralAnalyzer().analyze(linear_graph)
    for comp in result.components.values():
        assert comp.bridge_ratio == pytest.approx(1.0, abs=0.01), \
            f"{comp.id}: all edges in linear graph are bridges"

def test_graph_summary_present(self, linear_graph):
    result = StructuralAnalyzer().analyze(linear_graph)
    s = result.graph_summary
    assert s.vertex_count == 3
    assert s.edge_count == 2
    assert 0.0 <= s.density <= 1.0
```

### 4.3 Analysis Module — Quality Scoring and Classification

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| UT-ANAL-20 | RMAV scores computed from structural metrics | R, M, A, V, Q all ∈ [0, 1] |
| UT-ANAL-21 | Hub component has high R(v) | PageRank-dominant node scores highest R |
| UT-ANAL-22 | Articulation point has high A(v) | Node with AP\_c > 0 scores highest A |
| UT-ANAL-23 | High-betweenness node has high M(v) | Bottleneck node scores highest M |
| UT-ANAL-24 | AHP weights produce different Q ordering | Custom matrix → different top-ranked component |
| UT-ANAL-25 | AHP consistency check: CR > 0.10 → abort | Exception raised, not just a warning |
| UT-ANAL-26 | AHP consistency check: CR ≤ 0.10 → weights accepted | No exception; weights sum to 1.0 |
| UT-ANAL-27 | Box-plot classification on n ≥ 12 | 5 levels assigned; CRITICAL count ≥ 0 |
| UT-ANAL-28 | Small sample (n < 12) uses percentile fallback | Percentile thresholds applied, no IQR computation |
| UT-ANAL-29 | Layer filtering: app layer | Only Application-type components in result |
| UT-ANAL-30 | Layer filtering: infra layer | Only Node-type components in result |
| UT-ANAL-31 | NDCG@K computed correctly | NDCG = 1.0 for perfect ranking; < 1.0 for imperfect |
| UT-ANAL-32 | NDCG@K zero case | NDCG = 1.0 by convention when all relevance scores are 0 |

```python
def test_ahp_inconsistency_raises(self):
    """CR > 0.10 must abort analysis, not just warn (REQ-QS-08)."""
    inconsistent_matrix = [
        [1.0, 9.0, 1.0/9.0],
        [1.0/9.0, 1.0, 9.0],
        [9.0, 1.0/9.0, 1.0]
    ]
    with pytest.raises(AHPInconsistencyError) as exc_info:
        AHPProcessor().compute_weights(inconsistent_matrix)
    assert "CR" in str(exc_info.value)
    assert "0.10" in str(exc_info.value)

def test_ndcg_perfect_ranking(self):
    predicted = [0.9, 0.7, 0.5, 0.3, 0.1]
    actual    = [0.9, 0.7, 0.5, 0.3, 0.1]
    assert compute_ndcg(predicted, actual, k=5) == pytest.approx(1.0, abs=0.001)

def test_ndcg_imperfect_ranking(self):
    predicted = [0.1, 0.3, 0.5, 0.7, 0.9]  # reversed
    actual    = [0.9, 0.7, 0.5, 0.3, 0.1]
    assert compute_ndcg(predicted, actual, k=5) < 1.0
```

### 4.4 Simulation Module

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| UT-SIM-01 | Load graph from GraphData | All components tracked as ACTIVE |
| UT-SIM-02 | Fail a component → state changes | Status becomes FAILED |
| UT-SIM-03 | Pub-sub path detection | Publisher → Topic → Subscriber paths found |
| UT-SIM-04 | Graph reset restores all state | All components ACTIVE after reset |
| UT-SIM-10 | Event simulation: messages flow | Subscribers receive messages from publishers |
| UT-SIM-20 | Failure simulation: CRASH cascade propagates | \|F\| > 1 when connected node removed |
| UT-SIM-21 | Physical cascade: Node → hosted Apps | Hosted applications added to F when Node fails |
| UT-SIM-22 | Logical cascade: Broker → exclusive Topics | Topics die when sole routing Broker fails |
| UT-SIM-23 | Application cascade: Publisher → starved Subscribers | Subscriber added to F when all publishers fail |
| UT-SIM-24 | Cascade count accuracy | \|cascaded\_failures\| matches number of newly failed components |
| UT-SIM-25 | Impact calculation: I(v) ∈ [0, 1] | Composite impact score in valid range |
| UT-SIM-26 | Custom impact weights (w\_r, w\_f, w\_t) | Different weights → different I(v) for same failure |
| UT-SIM-27 | DEGRADED failure mode | Component degrades, partial cascade (< full CRASH) |
| UT-SIM-28 | PARTITION failure mode | Network partition splits reachability |
| UT-SIM-29 | OVERLOAD failure mode | Throughput loss computed; structural paths unaffected |
| UT-SIM-30 | Monte Carlo: N trials produce distribution | Mean and variance of I(v) computed over N trials |
| UT-SIM-31 | Monte Carlo: p=1.0 equals exhaustive CRASH | Results identical within numerical tolerance |

```python
def test_degraded_failure_is_less_severe_than_crash(self, raw_graph_data):
    sim = FailureSimulator(SimulationGraph(graph_data=raw_graph_data))
    crash_result = sim.simulate(FailureScenario("Node1", mode="CRASH"))
    degraded_result = sim.simulate(FailureScenario("Node1", mode="DEGRADED"))
    assert degraded_result.impact.composite_impact <= crash_result.impact.composite_impact

def test_configurable_impact_weights(self, raw_graph_data):
    default_result = FailureSimulator(
        SimulationGraph(raw_graph_data)
    ).simulate(FailureScenario("App1", "CRASH"))
    custom_result = FailureSimulator(
        SimulationGraph(raw_graph_data),
        impact_weights={"reachability": 1.0, "fragmentation": 0.0, "throughput": 0.0}
    ).simulate(FailureScenario("App1", "CRASH"))
    assert default_result.impact.composite_impact != custom_result.impact.composite_impact
```

### 4.5 Anti-Pattern Detection Module

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| UT-PAT-01 | SPOF detected: known articulation point | Component with AP\_c > 0 flagged as SPOF |
| UT-PAT-02 | SPOF not detected: no AP | No SPOF pattern on fully redundant ring graph |
| UT-PAT-03 | God Component detected: high Q and high degree | Pattern with severity HIGH returned |
| UT-PAT-04 | Bottleneck Edge: bridge edge with high betweenness | Pattern returned with severity HIGH |
| UT-PAT-05 | Systemic Risk: ≥ 3 CRITICAL components in clique | SYSTEMIC\_RISK pattern returned |
| UT-PAT-06 | Empty graph | Empty list returned, no exception |
| UT-PAT-07 | DetectedProblem fields populated | pattern\_type, severity, component\_ids, description, recommendation all non-empty |

```python
def test_spof_detection(self, ap_graph_with_scores):
    detector = ProblemDetector()
    problems = detector.detect(ap_graph_with_scores)
    spof_ids = [p.component_ids for p in problems if p.pattern_type == "SPOF"]
    assert any("B" in ids for ids in spof_ids), "B is the sole AP; must be flagged as SPOF"

def test_no_spof_on_ring(self, ring_graph_with_scores):
    problems = ProblemDetector().detect(ring_graph_with_scores)
    assert all(p.pattern_type != "SPOF" for p in problems)
```

### 4.6 Validation Module

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| UT-VAL-01 | Perfect positive correlation | ρ = 1.0, p < 0.05 |
| UT-VAL-02 | Perfect inverse correlation | ρ = −1.0 |
| UT-VAL-03 | Identical predictions | RMSE = 0.0, MAE = 0.0 |
| UT-VAL-04 | Matching top elements | Top-K overlap = 100%, NDCG = 1.0 |
| UT-VAL-05 | Mismatched top elements | NDCG < 1.0 |
| UT-VAL-06 | Empty input | No crash; empty result returned |
| UT-VAL-07 | Single element | Warning issued; valid result with undefined ρ |
| UT-VAL-08 | Pass/fail logic for primary gates | Correct boolean given threshold comparison |
| UT-VAL-09 | Cohen's κ: perfect agreement | κ = 1.0 |
| UT-VAL-10 | Cohen's κ: random agreement | κ ≈ 0.0 |
| UT-VAL-11 | Cohen's κ: below-chance agreement | κ < 0.0 |

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

class TestCohensKappa:
    def test_perfect_agreement(self):
        predicted = ["CRITICAL", "HIGH", "LOW"]
        actual    = ["CRITICAL", "HIGH", "LOW"]
        assert cohens_kappa(predicted, actual) == pytest.approx(1.0, abs=0.01)

    def test_below_chance_agreement(self):
        predicted = ["CRITICAL", "CRITICAL", "CRITICAL"]
        actual    = ["MINIMAL",  "MINIMAL",  "MINIMAL"]
        assert cohens_kappa(predicted, actual) < 0.0
```

### 4.7 Visualization Module

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| UT-VIZ-01 | Dashboard generator produces valid HTML | Output parses as HTML; contains expected section IDs |
| UT-VIZ-02 | KPI cards render with correct values | Component count and critical count match input data |
| UT-VIZ-03 | Network graph data serialized to vis.js format | Nodes array and edges array present in embedded JSON |
| UT-VIZ-04 | Default color theme values | CRITICAL = #E74C3C, HIGH = #E67E22 |
| UT-VIZ-05 | Custom theme overrides | Overridden values applied; defaults preserved for unspecified keys |

### 4.8 Error Handling and Logging

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| UT-ERR-01 | Invalid JSON topology at import | `TopologyParseError` raised; error message includes offending field |
| UT-ERR-02 | Dangling edge (endpoint not in component list) | `TopologyValidationError` with edge ID in message |
| UT-ERR-03 | Duplicate component ID | `TopologyValidationError` with duplicate ID in message |
| UT-ERR-04 | Disconnected graph at import | Warning logged (not exception); analysis proceeds per component |
| UT-ERR-05 | Eigenvector centrality non-convergence | Falls back to in-degree; WARNING log entry emitted |
| UT-ERR-06 | Pipeline step logs start time and completion time | Two log entries at INFO level per step |
| UT-ERR-07 | Validation pass/fail logged with metric and threshold | Log entry contains metric value and target threshold |
| UT-ERR-08 | Neo4j credentials not in source code | `os.environ.get()` or config file used; no hardcoded credentials in module |

```python
def test_dangling_edge_raises(self):
    bad_topology = {
        "applications": [{"id": "app1", ...}],
        "relationships": {"subscribes_to": [{"source": "app1", "target": "nonexistent"}]}
    }
    with pytest.raises(TopologyValidationError, match="nonexistent"):
        TopologyParser().parse(bad_topology)

def test_step_completion_logged(self, caplog, linear_graph):
    with caplog.at_level(logging.INFO):
        StructuralAnalyzer().analyze(linear_graph)
    assert any("completed" in r.message.lower() for r in caplog.records)
    assert any("start" in r.message.lower() for r in caplog.records)
```

### 4.9 Coverage Targets

| Module | Unit Tests | Target Coverage |
|--------|-----------|----------------|
| `src/core/models.py` | ~15 | 85% |
| `src/core/neo4j_repo.py` | ~10 (mocked) | 75% |
| `src/analysis/structural_analyzer.py` | ~20 | 85% |
| `src/analysis/quality_analyzer.py` | ~15 | 82% |
| `src/simulation/failure_simulator.py` | ~15 | 80% |
| `src/validation/validator.py` | ~12 | 82% |
| `src/visualization/dashboard.py` | ~8 | 75% |
| **Total** | **~95** | **≥ 80%** |

---

## 5. Integration Tests

Integration tests verify that modules compose correctly when data flows between them. These tests use the `@pytest.mark.integration` marker and require a running Neo4j instance on port 7688 (test port).

### 5.1 Analysis Pipeline

Tests that StructuralAnalyzer → QualityAnalyzer → ProblemDetector produces correct end-to-end results.

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| IT-ANAL-01 | StructuralAnalyzer → QualityAnalyzer | RMAV scores computed from structural metrics |
| IT-ANAL-02 | QualityAnalyzer → ProblemDetector | Architectural problems identified |
| IT-ANAL-03 | Full analysis pipeline | LayerAnalysisResult with components, edges, problems |
| IT-ANAL-04 | Multi-layer analysis (all 4 layers) | Distinct results per layer; app layer has only Application components |

```python
@pytest.mark.integration
def test_full_analysis_pipeline(multi_layer_graph):
    structural = StructuralAnalyzer()
    quality = QualityAnalyzer()
    detector = ProblemDetector()

    struct_result = structural.analyze(multi_layer_graph)
    assert len(struct_result.components) > 0

    qual_result = quality.analyze(struct_result)
    assert all(0.0 <= c.scores.overall <= 1.0 for c in qual_result.components)

    problems = detector.detect(qual_result)
    assert isinstance(problems, list)
```

### 5.2 Simulation Pipeline

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| IT-SIM-01 | Event simulation + failure simulation on same graph | Both produce valid results without state contamination |
| IT-SIM-02 | Exhaustive simulation across all components | Exactly one FailureResult per component in layer |
| IT-SIM-03 | Layer-specific simulation | Only layer components simulated; cross-layer cascade still propagates |

### 5.3 Cross-Pipeline (Analysis → Simulation → Validation)

The most important integration test — verifies the full prediction-vs-reality loop.

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| IT-VAL-01 | Q(v) compared against I(v) | ValidationGroupResult with all 11 metrics present |
| IT-VAL-02 | Matched component count > 0 | Predicted and actual component sets overlap |
| IT-VAL-03 | Pass/fail reflects validation targets | Boolean result matches threshold comparison |

```python
@pytest.mark.integration
def test_full_validation_pipeline(multi_layer_graph):
    struct_result = StructuralAnalyzer().analyze(multi_layer_graph)
    qual_result = QualityAnalyzer().analyze(struct_result)

    sim_graph = SimulationGraph(graph_data=multi_layer_graph)
    sim_results = FailureSimulator(sim_graph).simulate_exhaustive()

    predicted = {c.id: c.scores.overall for c in qual_result.components}
    actual = {r.target_id: r.impact.composite_impact for r in sim_results}
    result = Validator().validate(predicted, actual)

    assert result.matched_count > 0
    assert -1.0 <= result.correlation.spearman <= 1.0
    assert 0.0 <= result.classification.f1 <= 1.0
```

### 5.4 Neo4j Integration

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| IT-NEO-01 | Import JSON → Export roundtrip | All entities and edges preserved with correct properties |
| IT-NEO-02 | DEPENDS\_ON derivation (all 4 rules) | Derived edges exist for app\_to\_app, app\_to\_broker, node\_to\_node, node\_to\_broker |
| IT-NEO-03 | QoS weight propagation | Non-zero weights on Topics and DEPENDS\_ON edges |
| IT-NEO-04 | Layer extraction query | Returns only components matching the requested dependency type |
| IT-NEO-05 | Uniqueness constraints enforced | Duplicate component ID raises Neo4j constraint error |
| IT-NEO-06 | GraphML import → same result as JSON import | Equivalent topology produces identical DEPENDS\_ON edges |

### 5.5 REST API Integration

Tests that the FastAPI backend correctly invokes domain services and returns properly structured responses. These tests require the full Docker stack (`@pytest.mark.api`).

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| IT-API-01 | `GET /health` | `{"status": "ok"}`, HTTP 200 |
| IT-API-02 | `GET /api/v1/graph/search-nodes` with no data | HTTP 200; empty node list |
| IT-API-03 | `POST /api/v1/graph/import` with valid JSON | HTTP 200; entity counts match topology |
| IT-API-04 | `POST /api/v1/graph/import` with invalid JSON | HTTP 422; error body contains offending field |
| IT-API-05 | `POST /api/v1/analysis/layer/{layer}` after import | HTTP 200; all components have RMAV scores |
| IT-API-06 | `POST /api/v1/simulation/failure` | HTTP 200; one FailureResult per component |
| IT-API-07 | `POST /api/v1/validation/run-pipeline` | HTTP 200; Spearman ρ ∈ [−1, 1] |
| IT-API-08 | `GET /api/v1/validation/layers` | HTTP 200; result contains layer entries |
| IT-API-09 | `POST /api/v1/components` with unknown layer | HTTP 404 or HTTP 422 with descriptive message |

```python
@pytest.mark.api
@pytest.mark.asyncio
async def test_health_endpoint(api_client):
    r = await api_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

@pytest.mark.api
@pytest.mark.asyncio
async def test_import_valid_topology(api_client, sample_topology_json):
    r = await api_client.post("/api/v1/graph/import", json=sample_topology_json)
    assert r.status_code == 200
    body = r.json()
    assert body["applications"] > 0

@pytest.mark.api
@pytest.mark.asyncio
async def test_analysis_endpoint_returns_scores(api_client):
    r = await api_client.post("/api/v1/analysis/layer/app")
    assert r.status_code == 200
    components = r.json()["components"]
    assert len(components) > 0
    assert all("scores" in c for c in components)
    assert all(0.0 <= c["scores"]["overall"] <= 1.0 for c in components)
```

---

## 6. System Tests

System tests exercise the complete pipeline through CLI tools and the Docker stack, verifying end-to-end behaviour at multiple scales.

### 6.1 End-to-End Pipeline (CLI)

| Test ID | Scale | Description | Pass Criteria |
|---------|-------|-------------|---------------|
| ST-E2E-01 | Small | Full CLI pipeline (~10–25 components) | All 6 steps complete; dashboard generated |
| ST-E2E-02 | Medium | Full CLI pipeline (~30–50 components) | All 6 steps complete; validation passes |
| ST-E2E-03 | Large | Full CLI pipeline (~60–100 components) | All 6 steps complete within time budget |

**Procedure (small scale):**

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
| ST-CLI-01 | `generate_graph.py` | `--scale tiny/small/medium/large/xlarge` | Valid JSON output for each scale |
| ST-CLI-02 | `import_graph.py` | `--input FILE --clear` | Import completes; entity counts logged |
| ST-CLI-03 | `import_graph.py` | `--input FILE.graphml` | GraphML import produces same DEPENDS\_ON edges as equivalent JSON |
| ST-CLI-04 | `analyze_graph.py` | `--layer app/infra/system`, `--use-ahp`, `--output` | Analysis JSON produced |
| ST-CLI-05 | `simulate_graph.py` | `failure --exhaustive`, `--monte-carlo` | Simulation results produced |
| ST-CLI-06 | `validate_graph.py` | `--layer app`, `--output` | Validation JSON with pass/fail |
| ST-CLI-07 | `visualize_graph.py` | `--output FILE`, `--open` | Valid HTML dashboard |
| ST-CLI-08 | `run.py` | `--all --layer system` | Full pipeline completes without error |
| ST-CLI-09 | `benchmark.py` | `--scales small,medium --runs 3` | Benchmark report generated |

### 6.3 Error Handling

| Test ID | Scenario | Expected Behavior |
|---------|----------|-------------------|
| ST-ERR-01 | Non-existent input file | Graceful error message; non-zero exit code |
| ST-ERR-02 | Malformed JSON topology | Parse error with file/line reference |
| ST-ERR-03 | Neo4j unavailable | Connection error with URI hint; no stack trace in output |
| ST-ERR-04 | Empty input topology | Warning logged; empty result returned |
| ST-ERR-05 | Invalid layer name | Error listing valid layer options |
| ST-ERR-06 | Inconsistent AHP matrix (CR > 0.10) | Abort with CR value and which dimension failed |
| ST-ERR-07 | GraphML topology with unknown element | Parse error with element name; import halted |

### 6.4 Genieus Web Application System Tests

These tests verify the full Docker stack. They require `docker compose up --build` to be running.

| Test ID | Description | Pass Criteria |
|---------|-------------|---------------|
| ST-WEB-01 | Docker stack starts successfully | The genieus service healthy within 60 s |
| ST-WEB-02 | Genieus frontend accessible | HTTP 200 on `http://localhost:7000` |
| ST-WEB-03 | FastAPI docs accessible | HTTP 200 on `http://localhost:8000/docs` |
| ST-WEB-04 | Neo4j Browser accessible | HTTP 200 on `http://localhost:7474` |
| ST-WEB-05 | Full API pipeline via REST | Import → Analyze → Simulate → Validate all return HTTP 200 |
| ST-WEB-06 | Docker health check passes | `docker inspect --format='{{.State.Health.Status}}'` = "healthy" |
| ST-WEB-07 | Graceful shutdown | `docker compose down` stops all services within 30 s; no data corruption |

```bash
# ST-WEB-01: verify stack startup
docker compose up --build -d
sleep 60
docker compose ps  # all services should show "running"

# ST-WEB-02: frontend check
curl -sf http://localhost:7000 | grep -q "Genieus"

# ST-WEB-05: full API pipeline
TOPO=$(cat input/sample_small.json)
curl -sf -X POST http://localhost:8000/api/v1/graph/import \
     -H "Content-Type: application/json" -d "$TOPO"
curl -sf -X POST http://localhost:8000/api/v1/analysis/layer/app
curl -sf -X POST http://localhost:8000/api/v1/simulation/failure \
     -H "Content-Type: application/json" -d '{"layer":"app"}'
curl -sf -X POST http://localhost:8000/api/v1/validation/run-pipeline \
     -H "Content-Type: application/json" -d '{"layer":"app"}'
```

---

## 7. Performance and Scalability Tests

Performance tests verify that analysis completes within the time budgets specified in SRS REQ-PERF-01–04, and that prediction accuracy improves with system scale — a key thesis contribution.

### 7.1 Timing Targets

The scale bands and timing limits below are taken directly from SRS REQ-PERF-01–04. Note that the SRS "large" scale (~600 components) differs from the validation test scale bands in §8 (which use smaller systems for controlled experiments).

| SRS Req | Scale | Representative Component Count | Max Analysis Time |
|---------|-------|-------------------------------|-------------------|
| REQ-PERF-01 | Small | ~30 components | < 1 s |
| REQ-PERF-02 | Medium | ~100 components | < 5 s |
| REQ-PERF-03 | Large | ~600 components | < 20 s |
| REQ-PERF-04 | Dashboard generation | Any scale | < 10 s |

Simulation and import time targets (aligned with §7.2 resource targets):

| Scale | Max Simulation Time | Max Import Time |
|-------|--------------------|--------------------|
| ~30 components | < 2 s | < 1 s |
| ~100 components | < 10 s | < 5 s |
| ~600 components | < 60 s | < 10 s |

### 7.2 Resource Targets

| Metric | Target |
|--------|--------|
| Peak memory at ~100 components | < 2 GB |
| Peak memory at ~600 components | < 4 GB |
| Database import at ~100 components | < 5 s |

### 7.3 Benchmark Execution

The `benchmark.py` tool runs the full pipeline at each scale × layer combination, repeating N times with different random seeds to measure variance:

```bash
# Quick benchmark (1 run per configuration)
python bin/benchmark.py --scales small,medium,large --runs 1

# Full benchmark suite (5 runs for statistical stability)
python bin/benchmark.py --scales tiny,small,medium,large,xlarge \
    --layers app,infra,system --runs 5 --output results/benchmark
```

**Outputs:**
- `benchmark_data.csv` — raw timing and metric records per run
- `benchmark_results.json` — aggregated mean ± std per configuration
- `benchmark_report.md` — human-readable summary with pass/fail against targets

Each benchmark record captures: timing per pipeline step, graph statistics (nodes, edges, density), all 11 validation metrics (Spearman ρ, Kendall τ, Pearson r, F1, Precision, Recall, Cohen's κ, Top-5/10 overlap, RMSE, MAE), and pass/fail status.

---

## 8. Validation Tests

Validation tests are the most important tests for the research contribution. They verify that topology-based predictions Q(v) correlate with actual failure impact I(v), demonstrating that the methodology works. All validation tests use fixed random seeds for reproducibility.

### 8.1 Primary Validation Targets

| Metric | Target | Gate Level | Rationale |
|--------|--------|-----------|-----------|
| Spearman ρ | ≥ 0.70 | Primary | Predicted and actual rankings agree monotonically |
| p-value | ≤ 0.05 | Primary | Correlation is statistically significant |
| F1-Score | ≥ 0.80 | Primary | Balanced precision and recall |
| Top-5 Overlap | ≥ 40% | Primary | Agreement on the most critical components |
| RMSE | ≤ 0.25 | Secondary | Bounded prediction error |
| Precision | ≥ 0.80 | Reported | Minimized false alarms |
| Recall | ≥ 0.80 | Reported | All critical components caught |
| Cohen's κ | ≥ 0.60 | Reported | Chance-corrected agreement |
| Top-10 Overlap | ≥ 50% | Reported | Extended critical set agreement |
| MAE | ≤ 0.20 | Reported | Bounded absolute error |

All targets apply to the **application layer**. Infrastructure layer results are reported for informational purposes; see §8.3 for expected infrastructure layer performance.

### 8.2 Validation Matrix (Layer × Scale)

| Test ID | Layer | Scale | Target ρ | Target F1 | Seed |
|---------|-------|-------|----------|-----------|------|
| VT-APP-01 | Application | Small | ≥ 0.75 | ≥ 0.75 | 42 |
| VT-APP-02 | Application | Medium | ≥ 0.80 | ≥ 0.80 | 42 |
| VT-APP-03 | Application | Large | ≥ 0.85 | ≥ 0.83 | 42 |
| VT-INF-01 | Infrastructure | Small | ≥ 0.50 | ≥ 0.65 | 42 |
| VT-INF-02 | Infrastructure | Medium | ≥ 0.52 | ≥ 0.66 | 42 |
| VT-INF-03 | Infrastructure | Large | ≥ 0.54 | ≥ 0.68 | 42 |
| VT-SYS-01 | System | Small | ≥ 0.70 | ≥ 0.75 | 42 |
| VT-SYS-02 | System | Medium | ≥ 0.75 | ≥ 0.80 | 42 |
| VT-SYS-03 | System | Large | ≥ 0.80 | ≥ 0.83 | 42 |

**Rationale for lower infrastructure targets:** Application-level dependencies are directly captured by pub-sub topology. Infrastructure dependencies are inferred through hosting relationships, which are more loosely coupled to the actual failure propagation model. This is a documented limitation discussed in the thesis.

### 8.3 Achieved Results

**By layer (large scale, seed 42):**

| Metric | Application | Infrastructure | Target |
|--------|-------------|----------------|--------|
| Spearman ρ | **0.85** ✓ | 0.54 | ≥ 0.70 |
| F1-Score | **0.83** ✓ | 0.68 | ≥ 0.80 |
| Precision | **0.86** ✓ | 0.71 | ≥ 0.80 |
| Recall | **0.80** ✓ | 0.65 | ≥ 0.80 |
| Top-5 Overlap | **62%** ✓ | 40% | ≥ 40% |

**By scale (application layer):**

| Scale | Components | Spearman ρ | F1-Score | Analysis Time |
|-------|------------|------------|----------|---------------|
| Tiny | 5–10 | 0.72 | 0.70 | < 0.5 s |
| Small | 10–25 | 0.78 | 0.75 | < 1 s |
| Medium | 30–50 | 0.82 | 0.80 | ~2 s |
| Large | 60–100 | 0.85 | 0.83 | ~5 s |
| XLarge | 150–300 | 0.88 | 0.85 | ~20 s |

**Key finding:** Prediction accuracy improves with system scale — larger systems produce more stable centrality distributions, leading to more reliable correlation. This trend is a key thesis contribution (REQ-ACC-05).

### 8.4 Validation Procedure

```bash
# Deterministic validation with fixed seed
python bin/generate_graph.py --scale medium --seed 42 --output test_data.json
python bin/import_graph.py --input test_data.json --clear
python bin/analyze_graph.py --layer app --use-ahp --output analysis.json
python bin/simulate_graph.py failure --layer app --exhaustive --output simulation.json
python bin/validate_graph.py --layer app --output validation_result.json

# Inspect pass/fail
cat validation_result.json | jq '.overall.passed'
# Expected: true

# Run full validation matrix across all scales and layers
python bin/benchmark.py --scales small,medium,large \
    --layers app,infra,system --runs 1 --seed 42 \
    --output results/validation_matrix
```

---

## 9. Acceptance Criteria

Each user-facing capability has specific acceptance criteria. Automated criteria are verified by system tests; manual criteria are verified during acceptance testing. All criteria must pass before ICSA 2026 paper submission.

### 9.1 Feature Acceptance

#### Graph Model Construction

| ID | Criterion | Method | Pass If |
|----|-----------|--------|---------|
| AC-01 | Import JSON topology | Auto | All entities appear in Neo4j with correct properties |
| AC-02 | Import GraphML topology | Auto | Equivalent DEPENDS\_ON edges produced as JSON import *(not yet implemented)* |
| AC-03 | Derive DEPENDS\_ON edges | Auto | All 4 derivation types present with correct weights |
| AC-04 | Calculate QoS weights | Auto | Topic weights > 0 for non-default QoS settings |
| AC-05 | Support all preset scales | Auto | tiny through xlarge generate without error |

#### Structural Analysis

| ID | Criterion | Method | Pass If |
|----|-----------|--------|---------|
| AC-06 | Compute all 16 metrics | Auto | All 16 fields of StructuralMetrics populated for non-trivial graphs |
| AC-07 | Compute graph-level summary | Auto | S(G) fields present: vertex\_count, edge\_count, density, diameter, etc. |
| AC-08 | Identify articulation points | Auto | Known SPOFs detected on test graph |
| AC-09 | Multi-layer support | Auto | All 4 layers produce non-empty results |
| AC-10 | Export results to JSON | Auto | Valid JSON matching the SDD §8.3 output schema |

#### Quality Scoring

| ID | Criterion | Method | Pass If |
|----|-----------|--------|---------|
| AC-11 | RMAV scores present | Auto | All 4 dimensions + overall score in output |
| AC-12 | AHP weights produce different ordering | Auto | Custom matrix → different top-ranked component vs. defaults |
| AC-13 | AHP inconsistency aborts analysis | Auto | CR > 0.10 raises error with diagnostic; analysis does not complete |
| AC-14 | 5-level classification assigned | Auto | CRITICAL, HIGH, MEDIUM, LOW, MINIMAL all present in medium+ scale |

#### Failure Simulation

| ID | Criterion | Method | Pass If |
|----|-----------|--------|---------|
| AC-15 | Cascade propagation (CRASH) | Auto | \|F\| > 1 when a connected Node fails |
| AC-16 | DEGRADED mode less severe than CRASH | Auto | DEGRADED I(v) ≤ CRASH I(v) for same target |
| AC-17 | Impact score computed | Auto | I(v) ∈ [0, 1] for all components |
| AC-18 | Exhaustive mode | Auto | Exactly one result per component in layer |
| AC-19 | Results sorted by impact | Auto | I(v) values in descending order |

#### Validation

| ID | Criterion | Method | Pass If |
|----|-----------|--------|---------|
| AC-20 | Spearman ρ computed | Auto | Value ∈ [−1, 1]; p-value present |
| AC-21 | F1-Score computed | Auto | Value ∈ [0, 1] |
| AC-22 | Cohen's κ computed | Auto | Value present in output |
| AC-23 | NDCG@K computed | Auto | Value ∈ [0, 1] for K = 5 and K = 10 |
| AC-24 | Pass/fail determined | Auto | Boolean result matches threshold comparison |
| AC-25 | Accuracy targets met at app layer | Auto | ρ ≥ 0.70 and F1 ≥ 0.80 |

#### Static HTML Dashboard

| ID | Criterion | Method | Pass If |
|----|-----------|--------|---------|
| AC-26 | HTML dashboard generated | Auto | Valid HTML file with expected section IDs |
| AC-27 | KPI cards correct | Manual | Counts match analysis result JSON |
| AC-28 | Charts render in browser | Manual | Pie and bar charts visible without errors |
| AC-29 | Network graph interactive | Manual | vis.js: hover, click, drag, zoom all work |
| AC-30 | Correlation scatter plot present | Manual | Q(v) vs. I(v) plot visible with data points |

#### Genieus Web Application

| ID | Criterion | Method | Pass If |
|----|-----------|--------|---------|
| AC-31 | Docker stack starts cleanly | Auto | The genieus service healthy within 60 s |
| AC-32 | Dashboard tab loads | Manual | KPI cards and criticality chart visible |
| AC-33 | Graph Explorer: layer filter works | Manual | Switching layers updates node set |
| AC-34 | Graph Explorer: node click shows detail | Manual | Side panel opens with RMAV scores |
| AC-35 | Analysis tab triggers analysis | Manual | Scores refresh after clicking Analyze |
| AC-36 | Simulation tab shows cascade | Manual | Failed components highlighted on graph |
| AC-37 | Settings tab persists Neo4j config | Manual | Config survives page reload |
| AC-38 | REST API responds to all endpoints | Auto | IT-API-01 through IT-API-09 all pass |

#### Security and Logging

| ID | Criterion | Method | Pass If |
|----|-----------|--------|---------|
| AC-39 | Neo4j credentials not in source | Auto | `grep -r "password" src/` finds no hardcoded values |
| AC-40 | Credentials read from environment | Auto | `NEO4J_PASSWORD` env var controls connection |
| AC-41 | Logging verbosity configurable | Auto | `--verbose` produces DEBUG entries; `--quiet` suppresses INFO |
| AC-42 | Each pipeline step logs timing | Auto | INFO log entries with start/end times per step |

### 9.2 Acceptance Checklist

| ID | Requirement Group | Criteria | Status |
|----|-------------------|----------|--------|
| ACC-01 | JSON topology import | AC-01, AC-03, AC-04 | ☐ |
| ACC-02 | GraphML topology import | AC-02 | ☐ |
| ACC-03 | Compute all 16 metrics | AC-06, AC-07 | ☐ |
| ACC-04 | RMAV quality scoring | AC-11, AC-12, AC-13, AC-14 | ☐ |
| ACC-05 | Failure simulation | AC-15, AC-16, AC-17, AC-18 | ☐ |
| ACC-06 | Validation accuracy | AC-20 – AC-25 | ☐ |
| ACC-07 | Static HTML dashboard | AC-26 – AC-30 | ☐ |
| ACC-08 | Genieus web application | AC-31 – AC-38 | ☐ |
| ACC-09 | Security and logging | AC-39 – AC-42 | ☐ |
| ACC-10 | Performance (all SRS targets) | §7.1 timing targets | ☐ |
| ACC-11 | Multi-layer analysis | AC-09 | ☐ |
| ACC-12 | CLI usability | ST-CLI-01 – ST-CLI-09 | ☐ |
| ACC-13 | API usability | IT-API-01 – IT-API-09 | ☐ |
| ACC-14 | Documentation complete | SRS, SDD, STD all reviewed | ☐ |

---

## 10. Traceability Matrix

Each SRS v2.1 requirement maps to one or more test cases. Requirements without explicit mapping are flagged as gaps.

| Requirement | Description | Test IDs |
|-------------|-------------|----------|
| REQ-GM-01 | Accept JSON topology | UT-CORE-07, ST-CLI-02, IT-NEO-01, AC-01 |
| REQ-GM-02 | Accept GraphML topology | ST-CLI-03, IT-NEO-06, AC-02 |
| REQ-GM-03 | Create 6 structural edge types | IT-NEO-01 (roundtrip verifies all edge types) |
| REQ-GM-04 | Derive DEPENDS\_ON edges | IT-NEO-02, AC-03 |
| REQ-GM-05 | Compute QoS edge weights | UT-CORE-01–09, IT-NEO-03 |
| REQ-GM-06 | *(see SRS §4.1)* | *(gap — no test mapped)* |
| REQ-GM-07, REQ-ML-01–04 | Layer projection (all 4 layers) | UT-ANAL-29, UT-ANAL-30, IT-NEO-04, AC-09 |
| REQ-SA-01 | Compute PageRank | UT-ANAL-02 |
| REQ-SA-02 | Compute Reverse PageRank | UT-ANAL-03 |
| REQ-SA-03 | Compute Betweenness Centrality | UT-ANAL-04 |
| REQ-SA-04 | Compute Closeness Centrality | UT-ANAL-05 |
| REQ-SA-05 | Compute Eigenvector Centrality | UT-ANAL-06 |
| REQ-SA-06 | Compute In-Degree and Out-Degree | UT-ANAL-01 (all 16 fields present) |
| REQ-SA-07 | Compute Clustering Coefficient | UT-ANAL-01 |
| REQ-SA-08 | Articulation point + AP\_c score | UT-ANAL-07, UT-ANAL-08, UT-ANAL-09, AC-08 |
| REQ-SA-09 | Bridge detection + Bridge Ratio | UT-ANAL-10, UT-ANAL-11 |
| REQ-SA-10 | QoS weight aggregates (w, w\_in, w\_out) | UT-ANAL-01 |
| REQ-SA-11 | Normalize to [0, 1] | UT-ANAL-14, UT-ANAL-15 |
| REQ-SA-12 | Graph-level summary statistics | UT-ANAL-16, AC-07 |
| REQ-QS-01–04 | Compute RMAV scores | UT-ANAL-20–23 |
| REQ-QS-05 | Compute composite Q(v) | UT-ANAL-20, IT-ANAL-01 |
| REQ-QS-07 | Support AHP weights | UT-ANAL-24 |
| REQ-QS-08 | AHP consistency check → abort | UT-ANAL-25, UT-ANAL-26, AC-13 |
| REQ-QS-09 | Box-plot classification | UT-ANAL-27 |
| REQ-QS-10 | Small-sample percentile fallback | UT-ANAL-28 |
| REQ-FS-01 | Simulate CRASH mode | UT-SIM-20, AC-15 |
| REQ-FS-02 | DEGRADED, PARTITION, OVERLOAD modes | UT-SIM-27, UT-SIM-28, UT-SIM-29, AC-16 |
| REQ-FS-03 | Cascade propagation rules (3 rules) | UT-SIM-21, UT-SIM-22, UT-SIM-23 |
| REQ-FS-04–06 | Reachability, Fragmentation, Throughput | UT-SIM-25 (I(v) composition) |
| REQ-FS-07 | Composite impact I(v) with configurable weights | UT-SIM-25, UT-SIM-26 |
| REQ-FS-08 | Exhaustive simulation | ST-CLI-05, IT-SIM-02, AC-18 |
| REQ-FS-09 | Monte Carlo mode | UT-SIM-30, UT-SIM-31, ST-CLI-05 |
| REQ-VA-01 | Spearman ρ | UT-VAL-01, UT-VAL-02, IT-VAL-01, AC-20 |
| REQ-VA-02 | Precision, Recall, F1-Score | UT-VAL-08, IT-VAL-03, AC-21 |
| REQ-VA-03 | Top-K overlap (K=5, K=10) | UT-VAL-04, UT-VAL-05 |
| REQ-VA-04 | NDCG@K | UT-ANAL-31, UT-ANAL-32, AC-23 |
| REQ-VA-05 | RMSE and MAE | UT-VAL-03 |
| REQ-VA-06 | Cohen's κ | UT-VAL-09, UT-VAL-10, UT-VAL-11, AC-22 |
| REQ-VA-07 | Pass/fail against configurable targets | UT-VAL-08, IT-VAL-03, AC-24 |
| REQ-VZ-01 | HTML dashboard with KPI cards | UT-VIZ-01, UT-VIZ-02, ST-CLI-07, AC-26 |
| REQ-VZ-02 | Criticality pie + ranking bar charts | UT-VIZ-01, AC-28 |
| REQ-VZ-03 | Interactive network graph | UT-VIZ-03, AC-29 |
| REQ-VZ-04 | Sortable/filterable component table | AC-27 (manual inspection) |
| REQ-VZ-05 | Correlation scatter plot | AC-30 |
| REQ-VZ-06 | Dependency matrix heatmap | AC-27 (manual inspection) |
| REQ-VZ-07 | Validation metrics with pass/fail | UT-VIZ-01, AC-27 |
| REQ-VZ-08 | Multi-layer comparison views | AC-09, AC-33 |
| REQ-CLI-01 | Individual CLI commands per step | ST-CLI-01–09 |
| REQ-CLI-02 | Pipeline orchestrator (`run.py --all`) | ST-CLI-08, ST-E2E-01–03 |
| REQ-CLI-03 | `--layer` flag | ST-CLI-04, AC-09 |
| REQ-CLI-04 | `--output` flag for JSON export | ST-CLI-04, ST-CLI-06, AC-10 |
| REQ-PERF-01–04 | Analysis and dashboard timing | §7.1 targets, ST-E2E-01–03 |
| REQ-SCAL-01–02 | Up to 1,000 components / 10,000 edges | §7.1 (large scale benchmarks) |
| REQ-ACC-01–04 | Accuracy targets at application layer | VT-APP-01–03, AC-25 |
| REQ-ACC-05 | Accuracy improves with scale | §8.3 scale table, VT-APP-01–03 |
| REQ-REL-01 | Graceful invalid-input handling | UT-ERR-01–03, ST-ERR-01–07 |
| REQ-REL-02 | Neo4j connection failure recovery | ST-ERR-03 |
| REQ-REL-03 | AHP consistency validated before weights used | UT-ANAL-25, UT-ANAL-26, AC-13 |
| REQ-SEC-01 | Neo4j authentication via credentials | UT-ERR-08, AC-40 |
| REQ-SEC-02 | Credentials not in source code | AC-39 |
| REQ-SEC-03 | Encrypted Bolt connections supported | *(gap — no test mapped)* |
| REQ-LOG-01 | Configurable verbosity (DEBUG/INFO/WARNING/ERROR) | UT-ERR-06, AC-41 |
| REQ-LOG-02 | Per-step start/end time logging | UT-ERR-06, AC-42 |
| REQ-LOG-03 | Validation pass/fail logged with values | UT-ERR-07, AC-42 |
| REQ-DV-01 | Unique component IDs | UT-ERR-03, IT-NEO-05 |
| REQ-DV-02 | Dangling edge validation | UT-ERR-02 |
| REQ-DV-03 | QoS values within valid ranges | UT-CORE-08, UT-CORE-09 |
| REQ-DV-04 | Weakly connected graph check | UT-ERR-04, ST-ERR-04 |
| REQ-HW-01–03 | Memory within targets per scale | §7.2 resource targets |
| REQ-PORT-01–02 | Platform and architecture support | CI/CD matrix in Appendix B |
| REQ-MAINT-01–03 | Code style, docstrings, 80% coverage | §4.9 coverage targets, Appendix B (linting job) |

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

> **Relationship to SRS performance targets:** The SRS defines "large" as ~600 components (REQ-PERF-03). The STD validation scale bands above use smaller systems for controlled experiments where ground truth is tractable. The benchmark suite (§7.3) generates systems matching the SRS definition; the STD scale bands are used for the validation matrix (§8.2) where exhaustive simulation must complete in reasonable time.

### Appendix B: CI/CD Pipeline

```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install flake8 mypy
      - run: flake8 src/ bin/ --max-line-length=100
      - run: mypy src/ --ignore-missing-imports

  unit-tests:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.11']
        os: [ubuntu-latest, macos-latest]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '${{ matrix.python-version }}' }
      - run: pip install -r requirements.txt -r requirements-test.txt
      - run: pytest tests/ -m "not integration and not api" --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v4

  integration-tests:
    runs-on: ubuntu-latest
    services:
      neo4j:
        image: neo4j:5-community
        ports: ['7687:7687']
        env: { NEO4J_AUTH: 'neo4j/testpassword' }
        options: >-
          --health-cmd "wget -q --spider http://localhost:7474 || exit 1"
          --health-interval 10s --health-timeout 5s --health-retries 10
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt -r requirements-test.txt
      - run: pytest tests/ -m integration -v
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
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r requirements.txt
      - run: python bin/benchmark.py --scales small,medium --runs 3 --seed 42
        env:
          NEO4J_URI: bolt://localhost:7687
          NEO4J_PASSWORD: testpassword
      - uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: output/benchmark_*.json
```

### Appendix C: Defect Severity Classification

| Severity | Definition | Examples |
|----------|------------|---------|
| Critical | System unusable or produces silent wrong results | Crash on startup, data loss, I(v) computed with wrong formula |
| High | Major feature broken | Analysis fails on valid input, cascade logic skips a rule, NDCG not computed |
| Medium | Feature partially works or produces cosmetically wrong output | Minor metric error within 5%, dashboard section missing |
| Low | Cosmetic or documentation | UI alignment, log message typo |

---

*Software-as-a-Graph Framework v2.1 · February 2026*
*Istanbul Technical University, Computer Engineering Department*