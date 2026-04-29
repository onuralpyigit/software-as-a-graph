# Software and System Test Document

## Software-as-a-Graph

### Graph-Based Critical Component Prediction for Distributed Publish-Subscribe Systems

**Version 2.3** · **March 2026**

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

Coverage spans both delivery mechanisms: the **CLI pipeline** (`cli/`) and the **Genieus web application** (FastAPI backend + Next.js frontend).

### 1.3 References

| Document | Description |
|----------|-------------|
| SRS v2.2 | Software Requirements Specification |
| SDD v2.2 | Software Design Description |
| IEEE 829-2008 | Standard for Software Test Documentation |
| IEEE 1012-2016 | Standard for System and Software Verification and Validation |
| IEEE RASSE 2025 | Published methodology paper (doi: 10.1109/RASSE64831.2025.11315354) |

### 1.4 Document Conventions

- Test IDs follow the pattern `<LEVEL>-<MODULE>-<NN>` (e.g., `UT-ANAL-01`, `IT-NEO-01`, `ST-E2E-01`, `VT-APP-01`, `AC-01`).
- The marker `@pytest.mark.<tag>` indicates the pytest marker used to select or exclude the test.
- Pass criteria use **shall** language matching the SRS requirement they verify.
- Requirement cross-references use IDs from SRS v2.2 (e.g., REQ-GM-01).

### 1.5 Document Overview

Section 2 describes the overall test strategy and schedule. Section 3 defines the test environment including software stack, database setup, and API test configuration. Sections 4–6 specify unit, integration, and system tests respectively. Section 7 covers performance and benchmark tests. Section 8 is the validation test suite — the most important section for the research contribution. Section 9 defines acceptance criteria for all user-facing capabilities including the Genieus web application. Section 10 provides the full SRS-to-test traceability matrix. Appendices cover scale specifications, CI/CD configuration, and defect severity classification.

### 1.6 Glossary

| Term | Definition |
|------|------------|
| AP\_c | Continuous articulation point score — fraction of graph fragmented upon vertex removal |
| AP\_c\_directed | Directed variant of AP\_c — `max(AP_c_out, AP_c_in)` on the directed graph |
| BR | Bridge Ratio — fraction of a vertex's incident edges that are bridges |
| CDI | Connectivity Degradation Index — normalised path elongation upon vertex removal |
| CDPot_enh | Enhanced Cascade Depth Potential — derived depth signal: `CDPot_base × (1 + MPCI)` |
| CouplingRisk_enh | `min(1.0, (1 - |2·Instability - 1|) * (1 + Δ·path_complexity))` — enriched imbalance score |
| CR | Consistency Ratio in AHP (must be < 0.10) |
| Fixture | Predefined test data created before a test runs |
| Mock | Simulated object that isolates the code under test |
| NDCG | Normalized Discounted Cumulative Gain — ranking quality metric |
| QSPOF | QoS-weighted SPOF Severity — `AP_c_dir(v) × w(v)` |
| RCL | Reverse Closeness Centrality — closeness computed on G^T |
| REV | Reverse Eigenvector Centrality — eigenvector centrality computed on G^T |
| ρ | Spearman rank correlation coefficient |
| RMAV | Reliability, Maintainability, Availability, Vulnerability |
| SUT | System Under Test |
| TP / FP / TN / FN | True/False Positive/Negative (classification outcomes) |

### 1.7 Change History

| Version | Date | Summary of Changes |
|---------|------|--------------------|
| 2.1 | February 2026 | Initial release |
| 2.3 | March 2026 | Refactored backend architecture with thinner routers, presenters, and dependency injection; updated quality formulas to include CDPot_enh, CQP, QSPOF, and QADS; aligned weights with AHP v2.3; updated glossary and test case formula references |
| 2.2 | February 2026 | Updated references to SRS/SDD v2.2; added CDPot, CouplingRisk, QSPOF, AP_c_directed, CDI, REV, RCL to glossary (§1.6); corrected UT-ANAL-21 formula reference from PR to RPR; added unit tests for new derived terms (§4.3 UT-ANAL-33–43); added `api` marker to pytest config (§3.3); corrected IT-API-09 from POST to GET; updated coverage table (§4.9); raised validation primary targets to match SRS v2.2 (§8.1, §8.2, AC-25); updated achieved results to IEEE RASSE 2025 published figures (§8.3); extended traceability matrix for SRS v2.2 requirements (§10) |
| 2.3 | March 2026 | Added `tests/test_api_graph.py` for comprehensive Graph module API verification; updated §3.5 with specific test command; refined REST API integration test descriptions (§5.5) |
| 2.3 | March 2026 | Added `tests/test_api_graph.py` for comprehensive Graph module API verification; updated §3.5 with specific test command; refined REST API integration test descriptions (§5.5); removed legacy `analyzer.py` shim |

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
    integration: marks tests requiring a running Neo4j instance
    api: marks tests requiring the full Docker stack (FastAPI + Neo4j)
```

> **Marker scope:** `integration` tests require only Neo4j on port 7688. `api` tests require the full `docker compose up` stack including FastAPI on port 8000 and Next.js on port 7000. `slow` marks any test taking > 10 s to allow quick exclusion with `pytest -m "not slow"`.

### 3.4 Test Database

A dedicated Neo4j instance runs on separate ports to prevent interference with development data:

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

> **Note:** `docker-compose.test.yml` must be created in the repository root before running integration tests. The file content above serves as the authoritative template.

### 3.5 Running Tests

```bash
# Unit tests (fast, no infrastructure required)
pytest tests/ -m "not integration and not api" -v

# Unit tests with coverage report
pytest tests/ -m "not integration and not api" --cov=src --cov-report=html

# Integration tests (requires Neo4j on port 7688)
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
import asyncio
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
| UT-CORE-06 | Size score formula: min(log₂(1+size/1024)/50, 0.20) | Correct for 1 KB, 8 KB, 64 KB, 1 MB |
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
        cases = [(1024, 0.02), (8192, 0.06), (1_048_576, 0.20)]
        for size, expected in cases:
             # Using calculate_weight with size contribution logic from Topic class
             size_kb = size / 1024
             score = min(math.log2(1 + size_kb) / 50, 0.20)
             assert score == pytest.approx(expected, abs=0.01)

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
    """A → B → C, B → D. B is the sole articulation point.
    Removing B disconnects {A} from {C, D}."""

@pytest.fixture
def star_graph():
    """Hub H → A, H → B, H → C, H → D.
    H has maximum betweenness and highest reverse eigenvector."""

@pytest.fixture
def cycle_graph():
    """A → B → C → A. No bridges; no articulation points."""

@pytest.fixture
def absorber_graph():
    """Many inputs → X → few outputs. X has DG_in >> DG_out; high CDPot."""
```

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| UT-ANAL-01 | Metrics computed on linear graph | All 16 fields of StructuralMetrics present per component |
| UT-ANAL-02 | PageRank ordering on A→B→C | PR(C) ≥ PR(B) ≥ PR(A) (downstream accumulates) |
| UT-ANAL-03 | Reverse PageRank ordering on A→B→C | RPR(A) ≥ RPR(B) ≥ RPR(C) (upstream accumulates) |
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

### 4.3 Analysis Module — Prediction and Classification

Tests that the RMAV formula inputs are correctly resolved, derived terms are computed, and the composite Q(v) and classification are correct.

**RMAV formula inputs reference (SDD v2.2 §6.19–§6.23):**
- R(v) = 0.45 × RPR + 0.30 × DG_in + 0.25 × CDPot_enh
- M(v) = 0.35 × BT + 0.30 × w_out + 0.15 × CQP + 0.12 × CouplingRisk + 0.08 × (1 − CC)
- A(v) = 0.35 × AP_c_directed + 0.25 × QSPOF + 0.25 × BR + 0.10 × CDI + 0.05 × w(v)
- V(v) = 0.40 × REV + 0.35 × RCL + 0.25 × QADS

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| UT-ANAL-20 | RMAV scores computed from structural metrics | R, M, A, V, Q all ∈ [0, 1] |
| UT-ANAL-21 | High-RPR node has high R(v) | Node with highest reverse\_pagerank scores highest R(v) (not PR) |
| UT-ANAL-22 | Articulation point has high A(v) | Node with AP\_c\_directed > 0 scores highest A(v) via QSPOF |
| UT-ANAL-23 | High-betweenness node has high M(v) | Bottleneck node scores highest M(v) |
| UT-ANAL-24 | AHP weights produce different Q ordering | Custom matrix → different top-ranked component vs. defaults |
| UT-ANAL-25 | AHP consistency check: CR > 0.10 → abort | `AHPConsistencyError` raised; analysis does not complete |
| UT-ANAL-26 | AHP consistency check: CR ≤ 0.10 → accepted | No exception; weights sum to 1.0 ± 1e-6 |
| UT-ANAL-27 | Box-plot classification on n ≥ 12 | 5 levels assigned; CRITICAL count ≥ 0 |
| UT-ANAL-28 | Small sample (n < 12) uses percentile fallback | Percentile thresholds applied; no IQR computation |
| UT-ANAL-29 | Layer filtering: app layer | Only Application-type components in result |
| UT-ANAL-30 | Layer filtering: infra layer | Only Node-type components in result |
| UT-ANAL-31 | NDCG@K computed correctly | NDCG = 1.0 for perfect ranking; < 1.0 for imperfect |
| UT-ANAL-32 | NDCG@K zero case | NDCG = 1.0 by convention when all relevance scores are 0 |
| UT-ANAL-33 | CDPot: absorber node (DG\_in >> DG\_out) | CDPot > 0.5; CDPot of fan-out hub (DG\_out >> DG\_in) ≈ 0.0 |
| UT-ANAL-34 | CDPot: fan-out hub | CDPot ≈ 0.0 (DG\_out/DG\_in ≥ 1.0, term clipped to 1) |
| UT-ANAL-35 | CDPot: division-by-zero guard | CDPot = 0.0 when DG\_in = 0 (ε denominator guard) |
| UT-ANAL-36 | CouplingRisk_enh: balanced node (Instability ≈ 0.5) | CouplingRisk_enh peak at 1.0 (capped) |
| UT-ANAL-37 | CouplingRisk_enh: pure source (DG\_in\_raw = 0) | CouplingRisk_enh = 0.0 |
| UT-ANAL-38 | CouplingRisk_enh: pure sink (DG\_out\_raw = 0) | CouplingRisk_enh = 0.0 |
| UT-ANAL-44 | CouplingRisk_enh: 1.0 cap enforcement | Value never exceeds 1.0 despite high path\_complexity |
| UT-ANAL-39 | QSPOF: AP node × high QoS weight | QSPOF > 0.0; non-AP node QSPOF = 0.0 |
| UT-ANAL-40 | AP\_c\_directed: directed removal | AP\_c\_directed ≥ undirected AP\_c on graphs with asymmetric reachability |
| UT-ANAL-41 | CDI: path elongation upon removal | CDI(bottleneck) > CDI(leaf) on path graph |
| UT-ANAL-42 | REV: computed on G^T | REV(source\_hub) > REV(sink) (roles reversed vs. EV) |
| UT-ANAL-43 | RCL: computed on G^T | RCL(easily\_reached\_node) > RCL(isolated) |

```python
def test_ahp_inconsistency_raises(self):
    """CR > 0.10 must abort analysis, not just warn (REQ-QS-08)."""
    inconsistent_matrix = [
        [1.0, 9.0, 1.0],
        [1/9, 1.0, 9.0],
        [1.0, 1/9, 1.0],
    ]
    with pytest.raises(AHPConsistencyError) as exc_info:
        AHPProcessor().compute_weights(inconsistent_matrix)
    assert "CR" in str(exc_info.value)

def test_r_uses_rpr_not_pr(self, linear_graph_with_scores):
    """R(v) formula uses RPR, DG_in, CDPot — not PR (SDD §6.19)."""
    quality_result = QualityAnalyzer().analyze(linear_graph_with_scores)
    # In A→B→C, C has highest PR but A has highest RPR.
    # R(C_component) should be lower than R(A_component).
    a = quality_result.component("A")
    c = quality_result.component("C")
    assert a.scores.reliability > c.scores.reliability, \
        "A is the source; its failure propagates to B and C; R(A) > R(C)"

def test_cdpot_absorber_vs_fanout(self):
    absorber = MockMetrics(rpr=0.8, dg_in=0.9, dg_out=0.1)
    fanout   = MockMetrics(rpr=0.8, dg_in=0.1, dg_out=0.9)
    assert compute_cdpot(absorber) > 0.5
    assert compute_cdpot(fanout)   == pytest.approx(0.0, abs=0.01)

def test_coupling_risk_balanced(self):
    # DG_in = DG_out → Instability = 0.5 → CouplingRisk = 1.0
    assert compute_coupling_risk(dg_in_raw=5, dg_out_raw=5) == pytest.approx(1.0, abs=0.01)

def test_coupling_risk_pure_source(self):
    # DG_in_raw = 0 → Instability = 1.0 → CouplingRisk_enh = 0.0
    assert compute_coupling_risk(dg_in_raw=0, dg_out_raw=5) == pytest.approx(0.0, abs=0.01)

def test_coupling_risk_cap_enforcement(self):
    # Base CR = 1.0, path_complexity = 10.0 -> Enriched = 1.0 * (1 + 0.1*10) = 2.0 -> Capped at 1.0
    assert compute_coupling_risk(dg_in_raw=5, dg_out_raw=5, path_complexity=10.0) == 1.0

def test_rev_roles_reversed_vs_ev(self, star_graph):
    """REV is eigenvector on G^T; source hubs in G become sink hubs in G^T."""
    metrics = StructuralAnalyzer().analyze(star_graph)
    hub = metrics.components["H"]
    assert hub.eigenvector > 0   # H is a hub in G (high EV)
    # REV is computed inside QualityAnalyzer; verify V(v) is highest for leaves
    # which are the hubs of G^T
    quality = QualityAnalyzer().analyze(metrics)
    leaf_v = max(
        (c for c in quality.components if c.id != "H"),
        key=lambda c: c.scores.vulnerability
    )
    assert leaf_v.scores.vulnerability > quality.component("H").scores.vulnerability, \
        "Leaves in G are hubs in G^T; they should have higher V(v)"
```

### 4.4 Simulation Module

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| UT-SIM-20 | CRASH mode: physical cascade | \|F\| > 1 when a Node hosting multiple applications fails |
| UT-SIM-21 | Physical cascade rule: Node → hosted Apps/Brokers fail | All hosted components in failed set |
| UT-SIM-22 | Logical cascade rule: Broker → exclusively-routed Topics die | Exclusively-routed topics marked unreachable |
| UT-SIM-23 | Application cascade: Publisher → starved Subscribers fail | Subscribers with no remaining publisher added to failed set |
| UT-SIM-24 | Fixed-point termination | Cascade terminates when no new failures in iteration |
| UT-SIM-25 | I(v) composition: ReachabilityLoss, Fragmentation, ThroughputLoss | composite\_impact = weighted combination of three components |
| UT-SIM-26 | Configurable I(v) weights | Different w\_r, w\_f, w\_t produce different composite\_impact ordering |
| UT-SIM-27 | DEGRADED mode less severe than CRASH | DEGRADED I(v) ≤ CRASH I(v) for same target component |
| UT-SIM-28 | PARTITION mode | Graph splitting reduces reachability; Fragmentation > 0 |
| UT-SIM-29 | OVERLOAD mode | ThroughputLoss > 0; cascade depth limited vs. CRASH |
| UT-SIM-30 | Monte Carlo mode: N trials | Returns N impact samples per component; mean and variance present |
| UT-SIM-31 | Monte Carlo deterministic with fixed seed | Identical results across two runs with same seed |

### 4.5 Anti-Pattern Detection Module

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| UT-PAT-01 | No anti-patterns on simple graph | Empty list returned |
| UT-PAT-02 | SPOF: component with AP\_c > 0 | SPOF pattern returned with severity CRITICAL |
| UT-PAT-03 | God Component: Q(v) > Q3 + 1.5×IQR and high degree | GOD\_COMPONENT pattern returned with severity HIGH |
| UT-PAT-04 | Bottleneck Edge: bridge edge with high betweenness | BOTTLENECK\_EDGE pattern returned with severity HIGH |
| UT-PAT-05 | Systemic Risk: ≥ 3 CRITICAL components in clique | SYSTEMIC\_RISK pattern returned with severity CRITICAL |
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
| UT-VAL-07 | n < 5 components | Warning issued; result contains diagnostic, no ρ computed |
| UT-VAL-08 | Pass/fail logic for primary gates | Correct boolean given threshold comparison |
| UT-VAL-09 | Cohen's κ: perfect agreement | κ = 1.0 |
| UT-VAL-10 | Cohen's κ: random agreement | κ ≈ 0.0 |
| UT-VAL-11 | Cohen's κ: below-chance agreement | κ < 0.0 |

```python
class TestCorrelationMetrics:
    def test_perfect_correlation(self):
        x = [0.1, 0.2, 0.3, 0.4, 0.5]
        y = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = Validator().validate(dict(zip("abcde", x)), dict(zip("abcde", y)))
        assert result.correlation.spearman == pytest.approx(1.0, abs=0.01)
        assert result.correlation.p_value < 0.05

    def test_minimum_n_warning(self):
        """Fewer than 5 paired samples should not crash but emit a warning."""
        x = {"a": 0.5, "b": 0.3}
        y = {"a": 0.4, "b": 0.2}
        with pytest.warns(UserWarning, match="n < 5"):
            result = Validator().validate(x, y)
        assert result is not None
```

### 4.7 Visualization Module

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| UT-VIZ-01 | HTML dashboard generated | Valid HTML string containing expected section IDs |
| UT-VIZ-02 | KPI values embedded | Node count, edge count, CRITICAL count in HTML |
| UT-VIZ-03 | vis.js network data embedded | JSON node/edge arrays present in `<script>` block |
| UT-VIZ-04 | Self-contained (no external CDN required for offline) | Critical vis.js and Chart.js included inline or bundled |
| UT-VIZ-05 | Empty analysis result | Produces valid HTML (empty state); no exception |

### 4.8 Error and Logging Tests

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| UT-ERR-01 | Neo4j connection failure | `ConnectionError` raised after 3 retries with URI in message |
| UT-ERR-02 | Dangling edge (target ID not in component list) | `TopologyValidationError` with edge ID in message |
| UT-ERR-03 | Duplicate component ID | `TopologyValidationError` with duplicate ID in message |
| UT-ERR-04 | Disconnected graph at import | Warning logged (not exception); analysis proceeds per component |
| UT-ERR-05 | Eigenvector centrality non-convergence | Falls back to in-degree; WARNING log entry emitted |
| UT-ERR-06 | Pipeline step logs start time and completion time | Two log entries at INFO level per step |
| UT-ERR-07 | Validation pass/fail logged with metric and threshold | Log entry contains metric value and target threshold |
| UT-ERR-08 | Neo4j credentials not in source code | `os.environ.get()` or config file used; no hardcoded credentials in module |

```python
def test_dangling_edge_raises(self):
    bad_topology = {
        "applications": [{"id": "app1", "name": "App"}],
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
| `saag/core/models.py` | ~15 | 85% |
| `saag/core/neo4j_repo.py` | ~10 (mocked) | 75% |
| `saag/core/memory_repo.py` | ~8 | 80% |
| `saag/analysis/structural_analyzer.py` | ~20 | 85% |
| `saag/analysis/quality_analyzer.py` | ~20 | 85% |
| `saag/simulation/failure_simulator.py` | ~15 | 80% |
| `saag/validation/validator.py` | ~12 | 82% |
| `saag/visualization/dashboard.py` | ~8 | 75% |
| `tools/benchmark/service.py` | ~5 | 70% |
| **Total** | **~118** | **≥ 80%** |

> **Note on path prefix:** Most modules are in `saag/`, while dev utilities are in `tools/`. Both packages are importable if the project root and `./` are in the Python path.

---

## 5. Integration Tests

Integration tests verify that modules compose correctly when data flows between them. These tests use the `@pytest.mark.integration` marker and require a running Neo4j instance on port 7688 (test port).

### 5.1 Analysis Pipeline

Tests that StructuralAnalyzer → QualityAnalyzer → ProblemDetector produces correct end-to-end results.

| Test ID | Description | Expected Result |
|---------|-------------|-----------------|
| IT-ANAL-01 | StructuralAnalyzer → PredictionEngine | RMAV scores computed from structural metrics; R uses RPR not PR |
| IT-ANAL-02 | PredictionEngine → ProblemDetector | Architectural problems identified from prediction result |
| IT-ANAL-03 | Full analysis pipeline | LayerAnalysisResult with components, edges, problems |
| IT-ANAL-04 | Multi-layer analysis (all 4 layers: app, infra, mw, system) | Distinct results per layer; app layer has only Application components |

```python
@pytest.mark.integration
def test_full_analysis_pipeline(multi_layer_graph):
    structural = StructuralAnalyzer()
    quality    = QualityAnalyzer()
    detector   = ProblemDetector()

    struct_result = structural.analyze(multi_layer_graph)
    assert len(struct_result.components) > 0

    qual_result = quality.analyze(struct_result)
    assert all(0.0 <= c.scores.overall <= 1.0 for c in qual_result.components)
    # All four RMAV dimensions must be present
    for c in qual_result.components:
        assert 0.0 <= c.scores.reliability     <= 1.0
        assert 0.0 <= c.scores.maintainability <= 1.0
        assert 0.0 <= c.scores.availability    <= 1.0
        assert 0.0 <= c.scores.vulnerability   <= 1.0

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
    qual_result   = QualityAnalyzer().analyze(struct_result)

    sim_graph    = SimulationGraph(graph_data=multi_layer_graph)
    sim_results  = FailureSimulator(sim_graph).simulate_exhaustive()

    predicted = {c.id: c.scores.overall for c in qual_result.components}
    actual    = {r.target_id: r.impact.composite_impact for r in sim_results}
    result    = Validator().validate(predicted, actual)

    assert result.matched_count > 0
    assert -1.0 <= result.correlation.spearman <= 1.0
    assert  0.0 <= result.classification.f1    <= 1.0
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
| IT-API-09 | `GET /api/v1/components` with unknown layer | HTTP 404 or HTTP 422 with descriptive message |
| IT-API-10 | `POST /api/v1/graph/generate` with valid payload | HTTP 200; returns generated graph data & stats |
| IT-API-10 | `POST /api/v1/graph/generate` with valid payload | HTTP 200; returns generated graph data & stats |

> **Endpoint method note:** IT-API-09 uses `GET /api/v1/components` (not POST). The `components` endpoint accepts the layer as a query parameter. See SDD v2.2 §8.2.

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
    # All four RMAV dimensions must be present
    for c in components:
        for dim in ("reliability", "maintainability", "availability", "vulnerability"):
            assert dim in c["scores"], f"Missing {dim} in component {c.get('id')}"
```

---

## 6. System Tests

System tests exercise the complete pipeline through CLI tools and the Docker stack, verifying end-to-end behaviour at multiple scales.

### 6.1 End-to-End Pipeline (CLI)

| Test ID | Scale | Description | Pass Criteria |
|---------|-------|-------------|---------------|
| ST-E2E-01 | Small | Full CLI pipeline (~10–25 components) | All 6 methodology steps complete (Import → Analyze → Predict → Simulate → Validate → Visualize); dashboard generated. Generate (Stage 0) is pre-pipeline and excluded from this count. |
| ST-E2E-02 | Medium | Full CLI pipeline (~30–50 components) | All 6 methodology steps complete; validation passes |
| ST-E2E-03 | Large | Full CLI pipeline (~60–100 components) | All 6 methodology steps complete within time budget |

**Procedure (small scale):**

```bash
PYTHONPATH=. python cli/generate_graph.py --scale small --output /tmp/test_data.json
PYTHONPATH=. python cli/import_graph.py --input /tmp/test_data.json --clear
PYTHONPATH=. python cli/analyze_graph.py --layer system --use-ahp --output /tmp/analysis.json
PYTHONPATH=. python cli/simulate_graph.py failure --layer system --exhaustive --output /tmp/simulation.json
PYTHONPATH=. python cli/validate_graph.py --layer system --output /tmp/validation.json
PYTHONPATH=. python cli/visualize_graph.py --layer system --output /tmp/dashboard.html
```

Or as a single command:

```bash
PYTHONPATH=. python cli/run.py --all --layer system --scale small
```

### 6.2 CLI Command Tests

Each CLI tool is tested independently with its most common options.

| Test ID | Command | Pass Criteria |
|---------|---------|---------------|
| ST-CLI-01 | `cli/import_graph.py --input <json>` | Exit 0; entities present in Neo4j |
| ST-CLI-02 | `cli/import_graph.py --input <json>` (JSON format) | Correct DEPENDS\_ON edges derived |
| ST-CLI-03 | `cli/import_graph.py --input <graphml>` | Equivalent topology as JSON import |
| ST-CLI-04 | `cli/analyze_graph.py --layer app` | Non-empty JSON output; RMAV scores present |
| ST-CLI-05 | `cli/simulate_graph.py failure --exhaustive` | One I(v) per component; sorted by impact |
| ST-CLI-06 | `cli/validate_graph.py --layer app` | JSON with Spearman ρ, F1, pass/fail flag |
| ST-CLI-07 | `cli/visualize_graph.py --layer system` | Valid HTML file; vis.js network included |
| ST-CLI-08 | `cli/generate_graph.py --scale medium --seed 42` | Deterministic output; same topology on re-run |
| ST-CLI-09 | `cli/run.py --all --layer system --open` | Full pipeline; browser launch attempted (mocked) |
| ST-CLI-10 | `cli/run.py --all --layer system --verbose` | DEBUG log entries visible; timing logged per step |
| ST-CLI-11 | `cli/run.py --generate --layer system --scale large` | Topology generated at large scale; import succeeds |
| ST-CLI-12 | `cli/benchmark.py --scales small,medium --runs 3` | JSON benchmark output; timing within budget |

> **ST-CLI-10 and ST-CLI-11** cover REQ-CLI-07 (`--verbose`) and REQ-CLI-05 (`--generate`) + REQ-CLI-06 (`--scale`) from SRS v2.2.

### 6.3 Web Application System Tests

| Test ID | Description | Pass Criteria |
|---------|-------------|---------------|
| ST-WEB-01 | `docker compose up` builds and starts cleanly | All four services healthy within 60 s |
| ST-WEB-02 | Frontend reachable at port 7000 | HTTP 200 on `http://localhost:7000` |
| ST-WEB-03 | Backend reachable at port 8000 | `/health` returns `{"status": "ok"}` |
| ST-WEB-04 | Dashboard tab loads data | KPI cards populated after analysis |
| ST-WEB-05 | Graph Explorer layer filter | Switching `app`/`infra`/`mw`/`system` updates node set |
| ST-WEB-06 | Analysis tab triggers analysis | POST to `/api/v1/analysis/layer/{layer}` completes; UI updates |
| ST-WEB-07 | Container health check passes | Docker reports container status `healthy` |

---

## 7. Performance and Scalability Tests

### 7.1 Analysis Timing Targets

All timing targets apply to the **application layer** on the designated scale, single-threaded (no parallelism), measured on the recommended hardware configuration (4 cores, 16 GB RAM).

| Scale | Components | Target: Analysis | Target: Simulation | Target: Full Pipeline |
|-------|------------|-----------------|--------------------|-----------------------|
| Tiny | 5–10 | < 1 s | < 2 s | < 5 s |
| Small | 10–25 | < 5 s | < 10 s | < 20 s |
| Medium | 30–50 | < 10 s | < 30 s | < 60 s |
| Large | 60–100 | < 30 s | < 120 s | < 180 s |
| XLarge | 150–300 | < 120 s | < 600 s | < 900 s |

> **REST API latency (REQ-PERF-05):** Each individual API endpoint (`/api/v1/analysis/...`, `/api/v1/simulation/...`, `/api/v1/validation/...`) must complete within **30 seconds** for medium-scale systems (30–50 components) under normal load. The health endpoint must respond within **100 ms**.

### 7.2 Scalability Tests

| Test ID | Description | Pass Criteria |
|---------|-------------|---------------|
| PT-SCAL-01 | Monotonic time growth | Analysis time increases sub-quadratically with component count |
| PT-SCAL-02 | Memory bounded at XLarge | Peak RSS ≤ 8 GB during XLarge analysis |
| PT-SCAL-03 | CDI sampled APSP at enterprise scale | CDI computation completes in ≤ 120 s for 300-node graph |
| PT-SCAL-04 | 1,000-node graph import | Import completes within 5 minutes; no OOM error |

### 7.3 Benchmark Procedure

```bash
# Run the full benchmark suite across all scales
PYTHONPATH=. python cli/benchmark.py \
    --scales tiny,small,medium,large,xlarge \
    --layers app,system \
    --runs 3 \
    --seed 42 \
    --output benchmarks/benchmark_$(date +%Y%m%d).json

# Verify timing targets
PYTHONPATH=. python cli/benchmark.py --check-targets --results benchmarks/benchmark_latest.json
```

The benchmark outputs a JSON file with `mean`, `std`, `min`, and `max` timing for each scale/layer combination across the specified number of runs.

---

## 8. Validation Tests

Validation tests are the most important tests for the research contribution. They verify that topology-based predictions Q(v) correlate with actual failure impact I(v), demonstrating that the methodology works. All validation tests use fixed random seeds for reproducibility.

### 8.1 Primary Validation Targets

These targets are aligned with SRS v2.2 §4.2, which raised all primary thresholds to reflect the empirically demonstrated performance published at IEEE RASSE 2025.

| Metric | Target | Gate Level | Rationale |
|--------|--------|-----------|-----------|
| Spearman ρ | ≥ 0.80 | Primary | Predicted and actual rankings agree monotonically |
| p-value | ≤ 0.05 | Primary | Correlation is statistically significant |
| F1-Score | ≥ 0.90 | Primary | Balanced precision and recall |
| Top-5 Overlap | ≥ 60% | Primary | Agreement on the most critical components |
| RMSE | ≤ 0.25 | Secondary | Bounded prediction error |
| Precision | ≥ 0.80 | Reported | Minimized false alarms |
| Recall | ≥ 0.80 | Reported | All critical components caught |
| Cohen's κ | ≥ 0.60 | Reported | Chance-corrected agreement |
| Top-10 Overlap | ≥ 60% | Reported | Extended critical set agreement |
| MAE | ≤ 0.20 | Reported | Bounded absolute error |

All primary targets apply to the **application layer**. Infrastructure layer results are reported for informational purposes; see §8.3 for expected infrastructure layer performance.

> **Threshold rationale:** SRS v2.1 used conservative targets (ρ ≥ 0.70, F1 ≥ 0.80, Top-5 ≥ 40%) set at project inception. IEEE RASSE 2025 published results (ρ = 0.876, F1 = 0.943, Top-5 = 80%) demonstrated the methodology substantially exceeds those thresholds. SRS v2.2 raises the targets to ρ ≥ 0.80, F1 ≥ 0.90, Top-5 ≥ 60% to reflect the demonstrated capability.

### 8.2 Validation Matrix (Layer × Scale)

Scale-specific targets are set below the aggregate primary targets to account for statistical instability at small sample sizes, while ensuring the overall primary gates are met at medium/large scale.

| Test ID | Layer | Scale | Target ρ | Target F1 | Target Top-5 | Seed |
|---------|-------|-------|----------|-----------|--------------|------|
| VT-APP-01 | Application | Small | ≥ 0.75 | ≥ 0.80 | ≥ 50% | 42 |
| VT-APP-02 | Application | Medium | ≥ 0.80 | ≥ 0.88 | ≥ 60% | 42 |
| VT-APP-03 | Application | Large | ≥ 0.85 | ≥ 0.90 | ≥ 70% | 42 |
| VT-INF-01 | Infrastructure | Small | ≥ 0.50 | ≥ 0.65 | ≥ 30% | 42 |
| VT-INF-02 | Infrastructure | Medium | ≥ 0.52 | ≥ 0.66 | ≥ 35% | 42 |
| VT-INF-03 | Infrastructure | Large | ≥ 0.54 | ≥ 0.68 | ≥ 40% | 42 |
| VT-SYS-01 | System | Small | ≥ 0.70 | ≥ 0.78 | ≥ 50% | 42 |
| VT-SYS-02 | System | Medium | ≥ 0.75 | ≥ 0.85 | ≥ 60% | 42 |
| VT-SYS-03 | System | Large | ≥ 0.80 | ≥ 0.88 | ≥ 65% | 42 |

**Rationale for lower infrastructure targets:** Application-level dependencies are directly captured by pub-sub topology. Infrastructure dependencies are inferred through hosting relationships, which are more loosely coupled to the actual failure propagation model. This is a documented limitation discussed in the thesis.

### 8.3 Achieved Results

**By layer (large scale, seed 42) — IEEE RASSE 2025 published results:**

| Metric | Application | Infrastructure | Primary Target |
|--------|-------------|----------------|----------------|
| Spearman ρ | **0.876** ✓ | 0.54 | ≥ 0.80 |
| F1-Score | **0.943** ✓ | 0.68 | ≥ 0.90 |
| Precision | **0.95** ✓ | 0.71 | ≥ 0.80 |
| Recall | **0.93** ✓ | 0.65 | ≥ 0.80 |
| Top-5 Overlap | **80%** ✓ | 40% | ≥ 60% |

**By scale (application layer) — empirical validation runs:**

| Scale | Components | Spearman ρ | F1-Score | Analysis Time |
|-------|------------|------------|----------|---------------|
| Tiny | 5–10 | 0.72 | 0.70 | < 0.5 s |
| Small | 10–25 | 0.787 ± 0.092 | 0.78 | < 1 s |
| Medium | 30–50 | 0.847 ± 0.067 | 0.85 | ~2 s |
| Large | 60–100 | 0.858 ± 0.025 | 0.90 | ~5 s |
| XLarge | 150–300 | 0.876 (aggregate) | 0.943 | ~20 s |

**Key finding:** Prediction accuracy improves with system scale — larger systems produce more stable centrality distributions, leading to more reliable correlation. This trend is a key thesis contribution (REQ-ACC-05).

### 8.4 Validation Procedure

```bash
# Deterministic validation with fixed seed
PYTHONPATH=. python cli/generate_graph.py --scale medium --seed 42 --output test_data.json
PYTHONPATH=. python cli/import_graph.py --input test_data.json --clear
PYTHONPATH=. python cli/analyze_graph.py --layer app --use-ahp --output analysis.json
PYTHONPATH=. python cli/simulate_graph.py failure --layer app --exhaustive --output simulation.json
PYTHONPATH=. python cli/validate_graph.py --layer app --output validation_result.json

# Inspect pass/fail
cat validation_result.json | jq '.overall.passed'
# Expected: true

# Run full validation matrix across all scales and layers
PYTHONPATH=. python cli/benchmark.py --scales small,medium,large \
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
| AC-02 | Import GraphML topology | Auto | Equivalent DEPENDS\_ON edges produced as JSON import |
| AC-03 | Derive DEPENDS\_ON edges | Auto | All 4 derivation types present with correct weights |
| AC-04 | Calculate QoS weights | Auto | Topic weights > 0 for non-default QoS settings |
| AC-05 | Support all preset scales | Auto | `tiny` through `xlarge` generate without error |

#### Structural Analysis

| ID | Criterion | Method | Pass If |
|----|-----------|--------|---------|
| AC-06 | Compute all 16 metrics | Auto | All 16 fields of StructuralMetrics populated for non-trivial graphs |
| AC-07 | Compute graph-level summary | Auto | S(G) fields present: vertex\_count, edge\_count, density, diameter |
| AC-08 | Identify articulation points | Auto | Known SPOFs detected on test graph |
| AC-09 | Multi-layer support | Auto | All 4 layers (app, infra, mw, system) produce non-empty results |
| AC-10 | Export results to JSON | Auto | Valid JSON matching the SDD v2.2 §8.3 output schema |

#### Prediction

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
| AC-25 | Accuracy targets met at app layer | Auto | ρ ≥ 0.80 **and** F1 ≥ 0.90 (SRS v2.2 targets) |

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

#### CLI Extended Flags

| ID | Criterion | Method | Pass If |
|----|-----------|--------|---------|
| AC-43 | `--generate` flag in orchestrator | Auto | ST-CLI-11 passes |
| AC-44 | `--scale` flag accepted | Auto | `tiny` through `xlarge` all generate without error |
| AC-45 | `--verbose` / `--quiet` flags | Auto | DEBUG entries present with `--verbose`; suppressed with `--quiet` |
| AC-46 | `--open` flag | Auto | No crash; browser launch mocked in CI |

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
| ACC-06 | Validation accuracy (v2.2 targets) | AC-20 – AC-25 | ☐ |
| ACC-07 | Static HTML dashboard | AC-26 – AC-30 | ☐ |
| ACC-08 | Genieus web application | AC-31 – AC-38 | ☐ |
| ACC-09 | Security and logging | AC-39 – AC-42 | ☐ |
| ACC-10 | Performance (all SRS v2.2 targets) | §7.1 timing targets | ☐ |
| ACC-11 | Multi-layer analysis | AC-09 | ☐ |
| ACC-12 | CLI usability (including extended flags) | ST-CLI-01 – ST-CLI-12, AC-43 – AC-46 | ☐ |
| ACC-13 | API usability | IT-API-01 – IT-API-09 | ☐ |
| ACC-14 | Documentation complete | SRS v2.2, SDD v2.2, STD v2.2 all reviewed | ☐ |

---

## 10. Traceability Matrix

Each SRS v2.2 requirement maps to one or more test cases. Requirements without explicit mapping are flagged.

| Requirement | Description | Test IDs |
|-------------|-------------|----------|
| REQ-GM-01 | Accept JSON topology | UT-CORE-07, ST-CLI-01, IT-NEO-01, AC-01 |
| REQ-GM-02 | Accept GraphML topology | ST-CLI-03, IT-NEO-06, AC-02 |
| REQ-GM-03 | Create 5 vertex types | IT-NEO-01 (roundtrip verifies all vertex types) |
| REQ-GM-04 | Create 6 structural edge types | IT-NEO-01 |
| REQ-GM-05 | Derive DEPENDS\_ON edges | IT-NEO-02, AC-03 |
| REQ-GM-06 | Compute QoS edge weights | UT-CORE-01–09, IT-NEO-03 |
| REQ-GM-07 | Propagate QoS weights to DEPENDS\_ON | IT-NEO-03 |
| REQ-GM-08, REQ-ML-01–04 | Layer projection (all 4 layers) | UT-ANAL-29, UT-ANAL-30, IT-NEO-04, AC-09 |
| REQ-SA-01 | Compute PageRank | UT-ANAL-02 |
| REQ-SA-02 | Compute Reverse PageRank | UT-ANAL-03 |
| REQ-SA-03 | Compute Betweenness Centrality | UT-ANAL-04 |
| REQ-SA-04 | Compute Closeness Centrality | UT-ANAL-05 |
| REQ-SA-05 | Compute Eigenvector Centrality | UT-ANAL-06 |
| REQ-SA-06 | Compute In-Degree and Out-Degree | UT-ANAL-01 (all 16 fields present) |
| REQ-SA-07 | Compute Clustering Coefficient | UT-ANAL-01 |
| REQ-SA-08 | Articulation point + AP\_c + AP\_c\_directed | UT-ANAL-07, UT-ANAL-08, UT-ANAL-09, UT-ANAL-40, AC-08 |
| REQ-SA-09 | Bridge detection + Bridge Ratio | UT-ANAL-10, UT-ANAL-11 |
| REQ-SA-10 | QoS weight aggregates (w, w\_in, w\_out) | UT-ANAL-01 |
| REQ-SA-11 | Normalize to [0, 1] | UT-ANAL-14, UT-ANAL-15 |
| REQ-SA-12 | Graph-level summary statistics | UT-ANAL-16, AC-07 |
| REQ-QS-01 | Compute Reliability R(v) | UT-ANAL-20, UT-ANAL-21, UT-ANAL-33, UT-ANAL-34, UT-ANAL-35 |
| REQ-QS-02 | Compute Maintainability M(v) | UT-ANAL-20, UT-ANAL-23, UT-ANAL-36, UT-ANAL-37, UT-ANAL-38 |
| REQ-QS-03 | Compute Availability A(v) | UT-ANAL-20, UT-ANAL-22, UT-ANAL-39, UT-ANAL-40, UT-ANAL-41 |
| REQ-QS-04 | Compute Vulnerability V(v) | UT-ANAL-20, UT-ANAL-42, UT-ANAL-43 |
| REQ-QS-05 | Compute composite Q(v) | UT-ANAL-20, IT-ANAL-01 |
| REQ-QS-06 | Classify into 5 criticality levels | UT-ANAL-27, AC-14 |
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
| REQ-VL-01 | Spearman ρ | UT-VAL-01, UT-VAL-02, IT-VAL-01, AC-20 |
| REQ-VL-02 | Precision, Recall, F1-Score | UT-VAL-08, IT-VAL-03, AC-21 |
| REQ-VL-03 | Top-K overlap (K=5, K=10) | UT-VAL-04, UT-VAL-05 |
| REQ-VL-04 | NDCG@K | UT-ANAL-31, UT-ANAL-32, AC-23 |
| REQ-VL-05 | RMSE and MAE | UT-VAL-03 |
| REQ-VL-06 | Cohen's κ | UT-VAL-09, UT-VAL-10, UT-VAL-11, AC-22 |
| REQ-VL-07 | Pass/fail against configurable targets | UT-VAL-08, IT-VAL-03, AC-24 |
| REQ-VL-08 | Minimum n ≥ 5 guard | UT-VAL-07 |
| REQ-VZ-01 | HTML dashboard with KPI cards | UT-VIZ-01, UT-VIZ-02, ST-CLI-07, AC-26 |
| REQ-VZ-02 | Criticality pie + ranking bar charts | UT-VIZ-01, AC-28 |
| REQ-VZ-03 | Interactive network graph | UT-VIZ-03, AC-29 |
| REQ-VZ-04 | Sortable/filterable component table | AC-27 (manual inspection) |
| REQ-VZ-05 | Correlation scatter plot | AC-30 |
| REQ-VZ-06 | Dependency matrix heatmap | AC-27 (manual inspection) |
| REQ-VZ-07 | Validation metrics with pass/fail | UT-VIZ-01, AC-27 |
| REQ-VZ-08 | Multi-layer comparison views | AC-09, AC-33 |
| REQ-CLI-01 | Individual CLI commands per step | ST-CLI-01–09 |
| REQ-CLI-02 | Pipeline orchestrator (`run.py`) | ST-E2E-01–03 |
| REQ-CLI-03 | `--layer` flag | UT-ANAL-29, UT-ANAL-30, ST-CLI-04 |
| REQ-CLI-04 | `--output` flag | ST-CLI-06 |
| REQ-CLI-05 | `--generate` flag | ST-CLI-11, AC-43 |
| REQ-CLI-06 | `--scale` flag | ST-CLI-08, ST-CLI-11, AC-44 |
| REQ-CLI-07 | `--verbose` / `--quiet` flags | ST-CLI-10, AC-41, AC-45 |
| REQ-CLI-08 | `--open` flag | ST-CLI-09, AC-46 |
| REQ-API-01 | `GET /health` | IT-API-01, AC-38 |
| REQ-API-02 | `GET /api/v1/graph/summary` | IT-API-03 (counts verified in response) |
| REQ-API-03 | `POST /api/v1/graph/import` | IT-API-03, IT-API-04, AC-01 |
| REQ-API-04 | `GET /api/v1/graph/search-nodes` | IT-API-02 |
| REQ-API-05 | `POST /api/v1/analysis/layer/{layer}` | IT-API-05, AC-11 |
| REQ-API-06 | `POST /api/v1/simulation/failure` | IT-API-06, AC-17 |
| REQ-API-07 | `POST /api/v1/validation/run-pipeline` | IT-API-07, AC-20 |
| REQ-API-08 | `GET /api/v1/validation/layers` | IT-API-08 |
| REQ-API-09 | `GET /api/v1/components` | IT-API-09 |
| REQ-API-10 | HTTP 422 error handling | IT-API-04 |
| REQ-API-11 | Swagger UI at `/docs` | ST-WEB-03 (manual navigation) |
| REQ-WEB-01 | Dashboard tab with KPI cards | ST-WEB-04, AC-32 |
| REQ-WEB-02 | Graph Explorer with layer filter | ST-WEB-05, AC-33 |
| REQ-WEB-03 | Node click shows detail panel | AC-34 |
| REQ-WEB-04 | Analysis tab triggers analysis | ST-WEB-06, AC-35 |
| REQ-WEB-05 | Simulation tab shows cascade | AC-36 |
| REQ-WEB-06 | Settings tab persists Neo4j config | AC-37 |
| REQ-WEB-07 | 2D / 3D graph toggle | AC-33 (extended manual check) |
| REQ-WEB-08–10 | Additional web UI requirements | AC-31 – AC-38 |
| REQ-SEC-01–03 | Credentials not hardcoded; env var controlled | UT-ERR-08, AC-39, AC-40 |
| REQ-LOG-01–03 | Logging verbosity; timing; pass/fail logged | UT-ERR-06, UT-ERR-07, AC-41, AC-42 |
| REQ-PERF-01–04 | Analysis timing within budget | §7.1 timing targets, PT-SCAL-01 |
| REQ-PERF-05 | REST API endpoint latency ≤ 30 s at medium scale | IT-API-05 (timed), §7.1 |
| REQ-SCAL-01–02 | Memory bounded; sub-quadratic growth | PT-SCAL-01, PT-SCAL-02 |
| REQ-PORT-01–02 | Docker container; platform-agnostic | ST-WEB-01, ST-WEB-07 |
| REQ-PORT-03 | Multi-stage Docker build | ST-WEB-01 (full stack build) |
| REQ-MAINT-01–03 | Composition/strategy/repository patterns | IT-ANAL-03 (strategy swap), UT-ANAL-24 |
| REQ-ACC-01–05 | Validation accuracy targets | §8.1–§8.3 validation tests |

---

## 11. Appendices

### Appendix A: Scale Definitions

| Scale | Applications | Topics | Brokers | Nodes | Libraries | Typical Use |
|-------|-------------|--------|---------|-------|-----------|-------------|
| Tiny | 5–8 | 3–5 | 1 | 2–3 | 1–2 | Unit tests, quick debugging |
| Small | 10–15 | 8–12 | 2 | 3–4 | 2–4 | Integration tests, development |
| Medium | 20–35 | 15–25 | 3–5 | 5–8 | 4–6 | System tests, PR validation |
| Large | 50–80 | 30–50 | 5–8 | 8–12 | 6–10 | Performance baseline |
| XLarge | 100–200 | 60–100 | 8–15 | 15–25 | 10–20 | Scalability benchmarks |

> **Note on scale bands:** The STD validation scale bands above use smaller systems for controlled experiments where ground truth is tractable. The benchmark suite (§7.3) generates systems matching the SRS v2.2 definition; the STD scale bands are used for the validation matrix (§8.2) where exhaustive simulation must complete in reasonable time.

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
      - run: flake8 saag/ cli/ --max-line-length=100
      - run: mypy saag/ --ignore-missing-imports

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
      - run: pip install -r ./requirements.txt -r ./requirements-test.txt
      - run: cd backend && pytest tests/ -m "not integration and not api" --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v4

  integration-tests:
    runs-on: ubuntu-latest
    services:
      neo4j:
        image: neo4j:5-community
        ports: ['7688:7687', '7475:7474']
        env: { NEO4J_AUTH: 'neo4j/testpassword' }
        options: >-
          --health-cmd "wget -q --spider http://localhost:7474 || exit 1"
          --health-interval 10s --health-timeout 5s --health-retries 10
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r ./requirements.txt -r ./requirements-test.txt
      - run: cd backend && pytest tests/ -m integration -v
        env:
          NEO4J_URI: bolt://localhost:7688
          NEO4J_PASSWORD: testpassword

  benchmark:
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests]
    services:
      neo4j:
        image: neo4j:5-community
        ports: ['7688:7687', '7475:7474']
        env: { NEO4J_AUTH: 'neo4j/testpassword' }
        options: >-
          --health-cmd "wget -q --spider http://localhost:7474 || exit 1"
          --health-interval 10s --health-timeout 5s --health-retries 10
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install -r ./requirements.txt
      - run: PYTHONPATH=. python cli/benchmark.py --scales small,medium --runs 3 --seed 42
        env:
          NEO4J_URI: bolt://localhost:7688
          NEO4J_PASSWORD: testpassword
      - uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: output/benchmark_*.json
```

### Appendix C: Defect Severity Classification

| Severity | Definition | Examples |
|----------|------------|---------|
| Critical | System unusable or produces silent wrong results | Crash on startup, data loss, I(v) computed with wrong formula, R(v) using PR instead of RPR |
| High | Major feature broken | Analysis fails on valid input, cascade logic skips a rule, NDCG not computed, CDPot always 0 |
| Medium | Feature partially works or produces cosmetically wrong output | Minor metric error within 5%, dashboard section missing, CDI not computed at enterprise scale |
| Low | Cosmetic or documentation | UI alignment, log message typo, incorrect SRS version cited in output |

---

*Software-as-a-Graph Framework v2.2 · February 2026*
*Istanbul Technical University, Computer Engineering Department*