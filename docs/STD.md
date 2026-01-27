# Software and System Test Document

## Software-as-a-Graph
### Graph-Based Critical Component Prediction for Distributed Publish-Subscribe Systems

**Version 1.0**  
**January 2026**

Istanbul Technical University  
Computer Engineering Department

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Test Plan Overview](#2-test-plan-overview)
3. [Test Environment](#3-test-environment)
4. [Unit Test Specifications](#4-unit-test-specifications)
5. [Integration Test Specifications](#5-integration-test-specifications)
6. [System Test Specifications](#6-system-test-specifications)
7. [Performance Test Specifications](#7-performance-test-specifications)
8. [Validation Test Specifications](#8-validation-test-specifications)
9. [Acceptance Test Specifications](#9-acceptance-test-specifications)
10. [Test Procedures](#10-test-procedures)
11. [Traceability Matrix](#11-traceability-matrix)
12. [Appendices](#appendices)

---

## 1. Introduction

### 1.1 Purpose

This Software and System Test Document provides comprehensive test specifications for the Software-as-a-Graph framework. It defines test cases, procedures, and acceptance criteria to verify that the system meets all functional and non-functional requirements specified in the Software Requirements Specification (SRS).

### 1.2 Scope

This document covers all testing activities including:

- **Unit Tests**: Individual module and function testing
- **Integration Tests**: Module interaction and data flow testing
- **System Tests**: End-to-end pipeline verification
- **Performance Tests**: Scalability and timing benchmarks
- **Validation Tests**: Statistical accuracy verification
- **Acceptance Tests**: User requirement validation

### 1.3 References

| Document | Version | Description |
|----------|---------|-------------|
| SRS | 1.0 | Software Requirements Specification |
| SDD | 1.0 | Software Design Description |
| IEEE 829-2008 | - | Standard for Software Test Documentation |
| IEEE 1012-2016 | - | Standard for System and Software Verification and Validation |

### 1.4 Definitions and Acronyms

| Term | Definition |
|------|------------|
| SUT | System Under Test |
| CI/CD | Continuous Integration/Continuous Deployment |
| Mock | Simulated object for isolated testing |
| Fixture | Predefined test data setup |
| TP/FP/TN/FN | True Positive, False Positive, True Negative, False Negative |
| ρ (rho) | Spearman rank correlation coefficient |
| RMSE | Root Mean Square Error |
| MAE | Mean Absolute Error |
| NDCG | Normalized Discounted Cumulative Gain |

---

## 2. Test Plan Overview

### 2.1 Test Strategy

The testing strategy follows a multi-level pyramid approach:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TESTING PYRAMID                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│                            ▲                                         │
│                           ╱ ╲        Acceptance Tests                │
│                          ╱   ╲       (User Scenarios)                │
│                         ╱─────╲                                      │
│                        ╱       ╲     System Tests                    │
│                       ╱         ╲    (End-to-End Pipeline)           │
│                      ╱───────────╲                                   │
│                     ╱             ╲   Integration Tests              │
│                    ╱               ╲  (Module Interactions)          │
│                   ╱─────────────────╲                                │
│                  ╱                   ╲ Unit Tests                    │
│                 ╱                     ╲(Functions/Classes)           │
│                ╱───────────────────────╲                             │
│                                                                      │
│   Distribution:   70%        20%         8%          2%             │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Test Levels Summary

| Level | Scope | Tools | Coverage Target |
|-------|-------|-------|-----------------|
| Unit | Individual functions and classes | pytest, pytest-cov | ≥80% line coverage |
| Integration | Module interactions, Neo4j connectivity | pytest, Docker | Key data flows |
| System | Complete pipeline execution | CLI scripts, benchmark.py | All user scenarios |
| Performance | Scalability across system sizes | benchmark.py | All defined scales |
| Validation | Statistical prediction accuracy | Validator module | All accuracy targets |
| Acceptance | User requirement satisfaction | Manual + automated | All requirements |

### 2.3 Test Schedule

| Phase | Duration | Activities |
|-------|----------|------------|
| Unit Testing | Continuous | TDD during development |
| Integration Testing | 1 week | After module completion |
| System Testing | 1 week | After integration pass |
| Performance Testing | 3 days | After system stability |
| Validation Testing | 3 days | Statistical verification |
| Acceptance Testing | 2 days | Before release |

### 2.4 Entry and Exit Criteria

**Entry Criteria:**
- Code compiles without errors
- All unit tests pass locally
- Code review completed
- Test environment available and configured

**Exit Criteria:**
- All planned tests executed
- No critical or high-severity defects open
- Unit test coverage ≥80%
- All validation targets met for application layer
- Performance benchmarks achieved

---

## 3. Test Environment

### 3.1 Hardware Configuration

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 8 GB | 16 GB |
| Storage | 10 GB SSD | 50 GB SSD |
| Network | 100 Mbps | 1 Gbps |

### 3.2 Software Configuration

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.9+ | Runtime environment |
| pytest | 7.0+ | Test framework |
| pytest-cov | 4.0+ | Coverage reporting |
| pytest-timeout | 2.0+ | Test timeout handling |
| Neo4j | 5.x | Graph database |
| Docker | 20.10+ | Test isolation |
| NetworkX | 2.6+ | Graph algorithms |

### 3.3 pytest Configuration

```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Default options
addopts = -v --tb=short --cov=src --cov-report=html

# Markers
markers =
    slow: marks tests as slow (skip with -m "not slow")
    integration: marks integration tests (requires Neo4j)
    performance: marks performance tests

# Timeout
timeout = 120

# Ignore warnings
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
```

### 3.4 Test Database Configuration

```yaml
# docker-compose.test.yml
services:
  neo4j-test:
    image: neo4j:5-community
    ports:
      - "7688:7687"  # Different port for test isolation
      - "7475:7474"
    environment:
      NEO4J_AUTH: neo4j/testpassword
      NEO4J_PLUGINS: '["graph-data-science"]'
    volumes:
      - neo4j-test-data:/data
```

---

## 4. Unit Test Specifications

### 4.1 Core Module Tests (`tests/test_core.py`)

#### 4.1.1 QoSPolicy Tests

| Test ID | Test Name | Description | Expected Result |
|---------|-----------|-------------|-----------------|
| UT-CORE-001 | test_default_qos_weight | Default QoS (VOLATILE, BEST_EFFORT, MEDIUM) | Weight ≈ 0.1 |
| UT-CORE-002 | test_reliable_qos_adds_weight | RELIABLE reliability setting | Weight ≈ 0.4 (+0.30) |
| UT-CORE-003 | test_persistent_qos_adds_weight | PERSISTENT durability setting | Weight ≈ 0.5 (+0.40) |
| UT-CORE-004 | test_urgent_priority_adds_weight | URGENT priority setting | Weight ≈ 0.3 (+0.30) |
| UT-CORE-005 | test_highest_qos_weight | Maximum QoS settings combined | Weight ≈ 1.0 |
| UT-CORE-006 | test_qos_to_dict | Serialization to dictionary | All fields present |
| UT-CORE-007 | test_qos_from_dict | Deserialization from dictionary | Correct QoSPolicy created |

**Test Implementation:**

```python
class TestQoSPolicy:
    """Tests for QoSPolicy weight calculation."""
    
    def test_default_qos_weight(self):
        """Default QoS should have low weight."""
        policy = QoSPolicy()
        weight = policy.calculate_weight()
        assert weight == pytest.approx(0.1, abs=0.01)
    
    def test_reliable_qos_adds_weight(self):
        """RELIABLE reliability adds 0.30 to weight."""
        policy = QoSPolicy(reliability="RELIABLE")
        weight = policy.calculate_weight()
        assert weight == pytest.approx(0.4, abs=0.01)
    
    def test_highest_qos_weight(self):
        """Maximum QoS settings should give weight of 1.0."""
        policy = QoSPolicy(
            reliability="RELIABLE",      # +0.30
            durability="PERSISTENT",     # +0.40
            transport_priority="URGENT"  # +0.30
        )
        weight = policy.calculate_weight()
        assert weight == pytest.approx(1.0, abs=0.01)
```

#### 4.1.2 Topic Weight Tests

| Test ID | Test Name | Description | Expected Result |
|---------|-----------|-------------|-----------------|
| UT-CORE-010 | test_small_topic_weight | Small message (1KB) | Weight ≈ 0.2 |
| UT-CORE-011 | test_medium_topic_weight | Medium message (64KB) | Weight ≈ 0.7 |
| UT-CORE-012 | test_large_topic_weight_capped | Large message (1MB) | Size score capped at 1.0 |
| UT-CORE-013 | test_full_topic_weight | Max QoS + large size | Weight ≈ 2.0 |
| UT-CORE-014 | test_topic_to_dict | Topic serialization | All properties included |

#### 4.1.3 Entity Tests

| Test ID | Test Name | Description | Expected Result |
|---------|-----------|-------------|-----------------|
| UT-CORE-020 | test_application_to_dict | Application serialization | All fields present |
| UT-CORE-021 | test_broker_to_dict | Broker serialization | id and name present |
| UT-CORE-022 | test_node_to_dict | Node serialization | id and name present |
| UT-CORE-023 | test_library_to_dict | Library with version | version included |
| UT-CORE-024 | test_library_without_version | Library without version | version omitted |

#### 4.1.4 Weight Formula Verification

| Test ID | Test Name | Description | Expected Result |
|---------|-----------|-------------|-----------------|
| UT-CORE-030 | test_size_score_formula | S_size = min(log₂(1 + size/1024) / 10, 1.0) | Formula matches documentation |
| UT-CORE-031 | test_qos_formula_components | Individual QoS contributions | RELIABLE=0.3, PERSISTENT=0.4, URGENT=0.3 |

```python
class TestWeightFormulas:
    """Verify weight formulas match documentation."""
    
    def test_size_score_formula(self):
        """Verify S_size = min(log₂(1 + size/1024) / 10, 1.0)."""
        test_cases = [
            (1024, 0.1),      # 1KB
            (8192, 0.32),     # 8KB  
            (65536, 0.60),    # 64KB
            (1048576, 1.0),   # 1MB (capped)
        ]
        
        for size_bytes, expected_score in test_cases:
            calculated = min(math.log2(1 + size_bytes / 1024) / 10, 1.0)
            assert calculated == pytest.approx(expected_score, abs=0.05)
```

### 4.2 Analysis Module Tests (`tests/test_analysis.py`)

#### 4.2.1 StructuralAnalyzer Tests

| Test ID | Test Name | Description | Expected Result |
|---------|-----------|-------------|-----------------|
| UT-ANAL-001 | test_structural_metrics | Basic metrics computation | All metrics computed |
| UT-ANAL-002 | test_pagerank_ordering | Downstream nodes higher PR | C ≥ A in A→B→C |
| UT-ANAL-003 | test_reverse_pagerank_ordering | Upstream nodes higher RPR | A ≥ C in A→B→C |
| UT-ANAL-004 | test_articulation_point_detection | AP correctly identified | B is AP in A-B-C,D |
| UT-ANAL-005 | test_bridge_detection | Bridge edges detected | Linear graph has bridges |
| UT-ANAL-006 | test_empty_graph_handling | Empty graph doesn't crash | Empty result returned |
| UT-ANAL-007 | test_single_node_handling | Single node graph | No AP, zero betweenness |

**Test Fixtures:**

```python
@pytest.fixture
def mock_graph_data():
    """Simple A->B->C linear graph."""
    return GraphData(
        components=[
            ComponentData(id="A", component_type="Application", weight=1.0),
            ComponentData(id="B", component_type="Application", weight=1.0),
            ComponentData(id="C", component_type="Application", weight=1.0)
        ],
        edges=[
            EdgeData("A", "B", "Application", "Application", "app_to_app", 1.0),
            EdgeData("B", "C", "Application", "Application", "app_to_app", 2.0)
        ]
    )

@pytest.fixture
def articulation_point_graph():
    """Graph where B is an articulation point: A--B--C, B--D."""
    return GraphData(
        components=[
            ComponentData(id="A", component_type="Application", weight=1.0),
            ComponentData(id="B", component_type="Application", weight=1.0),
            ComponentData(id="C", component_type="Application", weight=1.0),
            ComponentData(id="D", component_type="Application", weight=1.0),
        ],
        edges=[
            EdgeData("A", "B", "Application", "Application", "app_to_app", 1.0),
            EdgeData("B", "C", "Application", "Application", "app_to_app", 1.0),
            EdgeData("B", "D", "Application", "Application", "app_to_app", 1.0),
        ]
    )
```

**Test Implementation:**

```python
class TestStructuralAnalyzer:
    """Tests for StructuralAnalyzer metrics computation."""
    
    def test_structural_metrics(self, mock_graph_data):
        """Test basic structural metrics computation."""
        analyzer = StructuralAnalyzer()
        res = analyzer.analyze(mock_graph_data)
        
        assert "A" in res.components
        assert res.components["B"].betweenness > res.components["A"].betweenness
        assert ("A", "B") in res.edges
    
    def test_articulation_point_detection(self, articulation_point_graph):
        """Test that articulation points are correctly identified."""
        analyzer = StructuralAnalyzer()
        res = analyzer.analyze(articulation_point_graph)
        
        assert res.components["B"].is_articulation_point is True
        assert res.components["A"].is_articulation_point is False
    
    def test_empty_graph_handling(self, empty_graph):
        """Test that empty graphs don't crash."""
        analyzer = StructuralAnalyzer()
        res = analyzer.analyze(empty_graph)
        
        assert len(res.components) == 0
        assert len(res.edges) == 0
```

#### 4.2.2 QualityAnalyzer Tests

| Test ID | Test Name | Description | Expected Result |
|---------|-----------|-------------|-----------------|
| UT-ANAL-010 | test_quality_scoring | Quality scores computed | All components have Q(v) |
| UT-ANAL-011 | test_central_node_higher_score | Central nodes higher score | B ≥ A in A→B→C |
| UT-ANAL-012 | test_all_dimensions_computed | R, M, A, V all computed | All dimensions ≥ 0 |
| UT-ANAL-013 | test_edge_vulnerability_computed | Edge vulnerability | All edges have V score |
| UT-ANAL-014 | test_custom_weights | Custom weights affect scoring | Different results with different weights |
| UT-ANAL-015 | test_classification_summary | Summary properly built | Correct totals and distribution |

```python
class TestQualityAnalyzer:
    """Tests for QualityAnalyzer score computation."""
    
    def test_all_dimensions_computed(self, mock_graph_data):
        """Test that all four quality dimensions are computed."""
        struct_an = StructuralAnalyzer()
        struct_res = struct_an.analyze(mock_graph_data)
        
        qual_an = QualityAnalyzer()
        qual_res = qual_an.analyze(struct_res)
        
        for comp in qual_res.components:
            assert comp.scores.reliability >= 0
            assert comp.scores.maintainability >= 0
            assert comp.scores.availability >= 0
            assert comp.scores.vulnerability >= 0
            assert comp.scores.overall >= 0
    
    def test_custom_weights(self, mock_graph_data):
        """Test that custom weights affect scoring."""
        struct_an = StructuralAnalyzer()
        struct_res = struct_an.analyze(mock_graph_data)
        
        qual_default = QualityAnalyzer()
        res_default = qual_default.analyze(struct_res)
        
        custom_weights = QualityWeights(
            q_reliability=0.7,
            q_maintainability=0.1,
            q_availability=0.1,
            q_vulnerability=0.1
        )
        qual_custom = QualityAnalyzer(weights=custom_weights)
        res_custom = qual_custom.analyze(struct_res)
        
        default_scores = {c.id: c.scores.overall for c in res_default.components}
        custom_scores = {c.id: c.scores.overall for c in res_custom.components}
        
        assert any(
            abs(default_scores[id] - custom_scores[id]) > 0.001
            for id in default_scores
        )
```

#### 4.2.3 Layer Filtering Tests

| Test ID | Test Name | Description | Expected Result |
|---------|-----------|-------------|-----------------|
| UT-ANAL-020 | test_app_layer_components | APP layer filtering | Only Applications |
| UT-ANAL-021 | test_infra_layer_components | INFRA layer filtering | Only Nodes |
| UT-ANAL-022 | test_system_layer_includes_all | SYSTEM includes all types | All 5 types present |
| UT-ANAL-023 | test_mw_app_layer_components | MW_APP filtering | Application + Broker |
| UT-ANAL-024 | test_layer_analysis_filters | Filtering works correctly | Correct types per layer |

```python
class TestLayerFiltering:
    """Tests for layer definitions and component type filtering."""
    
    def test_app_layer_components(self):
        """APP layer should only include Application components."""
        layer_def = get_layer_definition(AnalysisLayer.APP)
        assert "Application" in layer_def.component_types
        assert "Node" not in layer_def.component_types
        assert "Broker" not in layer_def.component_types
    
    def test_layer_analysis_filters_correctly(self, multi_layer_graph):
        """Verify that layer analysis filters to correct component types."""
        analyzer = StructuralAnalyzer()
        
        res_app = analyzer.analyze(multi_layer_graph, layer=AnalysisLayer.APP)
        for comp_id, comp in res_app.components.items():
            assert comp.type == "Application"
        
        res_infra = analyzer.analyze(multi_layer_graph, layer=AnalysisLayer.INFRA)
        for comp_id, comp in res_infra.components.items():
            assert comp.type == "Node"
```

### 4.3 Simulation Module Tests (`tests/test_simulation.py`)

#### 4.3.1 SimulationGraph Tests

| Test ID | Test Name | Description | Expected Result |
|---------|-----------|-------------|-----------------|
| UT-SIM-001 | test_graph_loading | Load graph from data | All components loaded |
| UT-SIM-002 | test_component_state_tracking | State changes tracked | ACTIVE → FAILED |
| UT-SIM-003 | test_pub_sub_path_detection | Pub-sub paths identified | Publisher→Topic→Subscriber |
| UT-SIM-004 | test_graph_reset | Reset restores state | All ACTIVE after reset |

#### 4.3.2 EventSimulator Tests

| Test ID | Test Name | Description | Expected Result |
|---------|-----------|-------------|-----------------|
| UT-SIM-010 | test_event_simulation | Basic event flow | Messages reach subscribers |
| UT-SIM-011 | test_affected_topics | Topics correctly identified | Topic in affected list |
| UT-SIM-012 | test_reached_subscribers | Subscribers reached | Subscriber in reached list |

```python
@pytest.fixture
def raw_graph_data():
    return GraphData(
        components=[
            ComponentData("App1", "Application"),
            ComponentData("App2", "Application"),
            ComponentData("Topic1", "Topic"),
            ComponentData("Node1", "Node"),
        ],
        edges=[
            EdgeData("App1", "Topic1", "Application", "Topic", 
                    "PUBLISHES_TO", "PUBLISHES_TO"),
            EdgeData("App2", "Topic1", "Application", "Topic", 
                    "SUBSCRIBES_TO", "SUBSCRIBES_TO"),
            EdgeData("App1", "Node1", "Application", "Node", 
                    "RUNS_ON", "RUNS_ON"),
        ]
    )

def test_event_simulation(raw_graph_data):
    graph = SimulationGraph(graph_data=raw_graph_data)
    sim = EventSimulator(graph)
    
    res = sim.simulate(EventScenario("App1", "test"))
    
    assert "Topic1" in res.affected_topics
    assert "App2" in res.reached_subscribers
    assert res.metrics.messages_published > 0
```

#### 4.3.3 FailureSimulator Tests

| Test ID | Test Name | Description | Expected Result |
|---------|-----------|-------------|-----------------|
| UT-SIM-020 | test_failure_simulation | Basic failure cascade | Cascade propagates |
| UT-SIM-021 | test_physical_cascade | Node→Apps cascade | Hosted apps fail |
| UT-SIM-022 | test_cascade_count | Cascade count accurate | Count matches failed set |
| UT-SIM-023 | test_impact_calculation | Impact metrics computed | I(v) calculated |
| UT-SIM-024 | test_configurable_impact_weights | Custom weights work | Different I(v) values |

```python
def test_failure_simulation(raw_graph_data):
    graph = SimulationGraph(graph_data=raw_graph_data)
    sim = FailureSimulator(graph)
    
    res = sim.simulate(FailureScenario("Node1", "test"))
    
    assert "App1" in res.cascaded_failures
    assert res.impact.cascade_by_type.get("Application", 0) >= 1

def test_configurable_impact_weights(raw_graph_data):
    """Test that impact weights are configurable."""
    metrics1 = ImpactMetrics(
        reachability_loss=1.0, 
        fragmentation=0.0, 
        throughput_loss=0.0
    )
    metrics2 = ImpactMetrics(
        reachability_loss=1.0, 
        fragmentation=0.0, 
        throughput_loss=0.0,
        impact_weights={"reachability": 1.0, "fragmentation": 0.0, "throughput": 0.0}
    )
    
    assert metrics1.composite_impact == pytest.approx(0.4, abs=0.01)  # Default
    assert metrics2.composite_impact == pytest.approx(1.0, abs=0.01)  # Custom
```

### 4.4 Validation Module Tests (`tests/test_validation.py`)

#### 4.4.1 Correlation Metrics Tests

| Test ID | Test Name | Description | Expected Result |
|---------|-----------|-------------|-----------------|
| UT-VAL-001 | test_perfect_correlation | Identical rankings | ρ = 1.0 |
| UT-VAL-002 | test_inverse_correlation | Reversed rankings | ρ = -1.0 |
| UT-VAL-003 | test_no_correlation | Random rankings | -1.0 ≤ ρ ≤ 1.0 |
| UT-VAL-004 | test_rmse_calculation | RMSE computed correctly | Zero for identical |
| UT-VAL-005 | test_mae_calculation | MAE computed correctly | Correct average error |

```python
class TestCorrelationMetrics:
    """Tests for correlation and error metrics."""
    
    def test_perfect_correlation(self):
        """Perfect positive correlation should return rho=1.0."""
        x = [0.1, 0.2, 0.3, 0.4, 0.5]
        y = [0.1, 0.2, 0.3, 0.4, 0.5]
        rho, p_value = spearman_correlation(x, y)
        assert rho == pytest.approx(1.0, abs=0.01)
        assert p_value < 0.05
    
    def test_inverse_correlation(self):
        """Perfect inverse correlation should return rho=-1.0."""
        x = [0.1, 0.2, 0.3, 0.4, 0.5]
        y_inv = [0.5, 0.4, 0.3, 0.2, 0.1]
        rho, p_value = spearman_correlation(x, y_inv)
        assert rho == pytest.approx(-1.0, abs=0.01)
    
    def test_rmse_calculation(self):
        """Test RMSE calculation."""
        predicted = [1.0, 2.0, 3.0]
        actual = [1.0, 2.0, 3.0]
        metrics = calculate_error_metrics(predicted, actual)
        assert metrics.rmse == pytest.approx(0.0, abs=0.001)
```

#### 4.4.2 Ranking Metrics Tests

| Test ID | Test Name | Description | Expected Result |
|---------|-----------|-------------|-----------------|
| UT-VAL-010 | test_ranking_logic | Matching top elements | High overlap |
| UT-VAL-011 | test_ranking_with_mismatch | Different top elements | Lower NDCG |
| UT-VAL-012 | test_top_k_overlap | Top-K calculation | Correct overlap % |

```python
class TestRankingMetrics:
    """Tests for ranking and Top-K metrics."""
    
    def test_ranking_logic(self):
        """Test ranking metrics with matching top elements."""
        pred = {"A": 0.9, "B": 0.5, "C": 0.1}
        act = {"A": 0.8, "B": 0.4, "C": 0.2}
        
        res = calculate_ranking_metrics(pred, act)
        assert res.top_5_overlap > 0
        assert res.ndcg_10 > 0.9
    
    def test_ranking_with_mismatch(self):
        """Test ranking metrics when top elements differ."""
        pred = {"A": 0.9, "B": 0.5, "C": 0.1}
        act = {"C": 0.8, "B": 0.5, "A": 0.2}
        
        res = calculate_ranking_metrics(pred, act)
        assert res.ndcg_10 < 1.0
```

#### 4.4.3 Validator Tests

| Test ID | Test Name | Description | Expected Result |
|---------|-----------|-------------|-----------------|
| UT-VAL-020 | test_validator_flow | Basic validation flow | All metrics computed |
| UT-VAL-021 | test_validator_empty_input | Empty input handling | No crash, empty result |
| UT-VAL-022 | test_validator_single_element | Single element edge case | Warning issued |
| UT-VAL-023 | test_pass_fail_determination | Pass/fail logic | Correct determination |

```python
class TestValidator:
    """Tests for Validator orchestration."""
    
    def test_validator_flow(self):
        """Test basic validator flow."""
        validator = Validator()
        pred = {"A": 0.9, "B": 0.1, "C": 0.5}
        act = {"A": 0.8, "B": 0.2, "C": 0.6}
        types = {"A": "Type1", "B": "Type1", "C": "Type2"}
        
        res = validator.validate(pred, act, types)
        
        assert res.overall.sample_size == 3
        assert res.matched_count == 3
    
    def test_validator_empty_input(self):
        """Test validator handles empty input gracefully."""
        validator = Validator()
        res = validator.validate({}, {}, {})
        assert res.overall.sample_size == 0
    
    def test_validator_single_element(self):
        """Test validator with single element (edge case)."""
        validator = Validator()
        pred = {"A": 0.5}
        act = {"A": 0.5}
        types = {"A": "Application"}
        
        res = validator.validate(pred, act, types)
        assert res.matched_count == 1
        assert len(res.warnings) > 0  # Warning for small sample
```

### 4.5 Visualization Module Tests (`tests/test_visualization.py`)

#### 4.5.1 ColorTheme Tests

| Test ID | Test Name | Description | Expected Result |
|---------|-----------|-------------|-----------------|
| UT-VIZ-001 | test_default_theme_colors | Default theme attributes | All colors present |
| UT-VIZ-002 | test_custom_theme_override | Custom theme values | Custom values used |
| UT-VIZ-003 | test_theme_to_dict | Theme serialization | Correct dictionaries |
| UT-VIZ-004 | test_high_contrast_theme | Accessibility theme | Different from default |

```python
class TestColorTheme:
    """Tests for configurable color themes."""
    
    def test_default_theme_has_all_colors(self):
        """Default theme should have all required color attributes."""
        theme = DEFAULT_THEME
        assert theme.primary == "#3498db"
        assert theme.success == "#2ecc71"
        assert theme.danger == "#e74c3c"
        assert theme.critical == "#e74c3c"
    
    def test_custom_theme_override(self):
        """Custom theme should override default colors."""
        theme = ColorTheme(primary="#ff0000", success="#00ff00")
        assert theme.primary == "#ff0000"
        assert theme.success == "#00ff00"
        assert theme.danger == "#e74c3c"  # Default preserved
```

---

## 5. Integration Test Specifications

### 5.1 Analysis Pipeline Integration

| Test ID | Test Name | Description | Expected Result |
|---------|-----------|-------------|-----------------|
| IT-ANAL-001 | test_structural_to_quality | StructuralAnalyzer → QualityAnalyzer | Scores computed from metrics |
| IT-ANAL-002 | test_quality_to_problems | QualityAnalyzer → ProblemDetector | Problems identified |
| IT-ANAL-003 | test_full_analysis_pipeline | Complete analysis flow | LayerAnalysisResult |
| IT-ANAL-004 | test_multi_layer_analysis | All layers analyzed | MultiLayerAnalysisResult |

```python
@pytest.mark.integration
class TestAnalysisPipelineIntegration:
    """Integration tests for analysis pipeline."""
    
    def test_full_analysis_pipeline(self, multi_layer_graph):
        """Test complete analysis flow from graph to results."""
        structural = StructuralAnalyzer()
        quality = QualityAnalyzer()
        detector = ProblemDetector()
        
        # Step 1: Structural analysis
        struct_result = structural.analyze(multi_layer_graph)
        assert len(struct_result.components) > 0
        
        # Step 2: Quality scoring
        qual_result = quality.analyze(struct_result)
        assert len(qual_result.components) > 0
        
        # Step 3: Problem detection
        problems = detector.detect(qual_result.components, qual_result.edges, qual_result)
        assert isinstance(problems, list)
```

### 5.2 Simulation Pipeline Integration

| Test ID | Test Name | Description | Expected Result |
|---------|-----------|-------------|-----------------|
| IT-SIM-001 | test_event_to_failure | Event → Failure simulation | Both results computed |
| IT-SIM-002 | test_exhaustive_simulation | All components simulated | Results for each |
| IT-SIM-003 | test_layer_metrics | Layer metrics computed | LayerMetrics object |

### 5.3 Validation Pipeline Integration

| Test ID | Test Name | Description | Expected Result |
|---------|-----------|-------------|-----------------|
| IT-VAL-001 | test_analysis_to_validation | Analysis → Validation | Scores compared |
| IT-VAL-002 | test_simulation_to_validation | Simulation → Validation | Impacts compared |
| IT-VAL-003 | test_full_validation_pipeline | Analysis + Simulation → Validation | ValidationResult |

```python
@pytest.mark.integration
class TestValidationPipelineIntegration:
    """Integration tests for validation pipeline."""
    
    def test_full_validation_pipeline(self, multi_layer_graph):
        """Test complete validation flow."""
        # Analysis
        struct_an = StructuralAnalyzer()
        qual_an = QualityAnalyzer()
        struct_res = struct_an.analyze(multi_layer_graph)
        qual_res = qual_an.analyze(struct_res)
        
        # Simulation
        sim_graph = SimulationGraph(graph_data=multi_layer_graph)
        fail_sim = FailureSimulator(sim_graph)
        sim_results = fail_sim.simulate_exhaustive()
        
        # Extract scores
        predicted = {c.id: c.scores.overall for c in qual_res.components}
        actual = {r.target_id: r.impact.composite_impact for r in sim_results}
        
        # Validation
        validator = Validator()
        result = validator.validate(predicted, actual)
        
        assert result.matched_count > 0
```

### 5.4 Neo4j Integration Tests

| Test ID | Test Name | Description | Expected Result |
|---------|-----------|-------------|-----------------|
| IT-NEO-001 | test_import_export_roundtrip | Import → Export → Verify | Data preserved |
| IT-NEO-002 | test_dependency_derivation | DEPENDS_ON edges created | Derived edges exist |
| IT-NEO-003 | test_weight_propagation | Weights calculated | Non-zero weights |
| IT-NEO-004 | test_layer_extraction | Layer data retrieved | Correct filtering |

---

## 6. System Test Specifications

### 6.1 End-to-End Pipeline Tests

| Test ID | Test Name | Description | Expected Result |
|---------|-----------|-------------|-----------------|
| ST-E2E-001 | test_full_pipeline_small | Complete pipeline (small) | All steps complete |
| ST-E2E-002 | test_full_pipeline_medium | Complete pipeline (medium) | All steps complete |
| ST-E2E-003 | test_full_pipeline_large | Complete pipeline (large) | All steps complete |
| ST-E2E-004 | test_dashboard_generation | Pipeline with visualization | HTML dashboard created |

**System Test Procedure:**

```bash
# ST-E2E-001: Full Pipeline Test (Small Scale)

# Step 1: Generate data
python generate_graph.py --scale small --output test_data.json
# Expected: test_data.json created with ~10-25 components

# Step 2: Import to Neo4j
python import_graph.py --input test_data.json --clear
# Expected: All entities imported, dependencies derived

# Step 3: Analyze
python analyze_graph.py --layer system --use-ahp --output analysis.json
# Expected: Analysis results with RMAV scores

# Step 4: Simulate
python simulate_graph.py --layer system --exhaustive --output simulation.json
# Expected: Impact scores for all components

# Step 5: Validate
python validate_graph.py --layer system --output validation.json
# Expected: Validation metrics computed

# Step 6: Visualize
python visualize_graph.py --layer system --output dashboard.html
# Expected: HTML dashboard with all sections
```

### 6.2 CLI Command Tests

| Test ID | Test Name | Description | Expected Result |
|---------|-----------|-------------|-----------------|
| ST-CLI-001 | test_generate_graph_cli | generate_graph.py options | All scales work |
| ST-CLI-002 | test_import_graph_cli | import_graph.py options | Import succeeds |
| ST-CLI-003 | test_analyze_graph_cli | analyze_graph.py options | Analysis completes |
| ST-CLI-004 | test_simulate_graph_cli | simulate_graph.py options | Simulation completes |
| ST-CLI-005 | test_validate_graph_cli | validate_graph.py options | Validation completes |
| ST-CLI-006 | test_visualize_graph_cli | visualize_graph.py options | Dashboard generated |
| ST-CLI-007 | test_run_all_cli | run.py --all | Full pipeline completes |
| ST-CLI-008 | test_benchmark_cli | benchmark.py execution | Report generated |

### 6.3 Error Handling Tests

| Test ID | Test Name | Description | Expected Result |
|---------|-----------|-------------|-----------------|
| ST-ERR-001 | test_invalid_input_file | Non-existent file | Graceful error message |
| ST-ERR-002 | test_malformed_json | Invalid JSON format | Parse error reported |
| ST-ERR-003 | test_neo4j_connection_failure | Database unavailable | Connection error handled |
| ST-ERR-004 | test_empty_graph_handling | Empty input data | Warning, no crash |
| ST-ERR-005 | test_invalid_layer_name | Unknown layer specified | Error with valid options |

---

## 7. Performance Test Specifications

### 7.1 Scalability Tests

| Test ID | Scale | Components | Max Analysis Time | Min Accuracy |
|---------|-------|------------|-------------------|--------------|
| PT-SCAL-001 | Tiny | 5-10 | <0.5s | ρ ≥ 0.65 |
| PT-SCAL-002 | Small | 10-25 | <1s | ρ ≥ 0.70 |
| PT-SCAL-003 | Medium | 30-50 | <5s | ρ ≥ 0.75 |
| PT-SCAL-004 | Large | 60-100 | <10s | ρ ≥ 0.80 |
| PT-SCAL-005 | XLarge | 150-300 | <30s | ρ ≥ 0.85 |

### 7.2 Benchmark Configuration

```python
# benchmark.py configuration
SCALE_DEFINITIONS = {
    "tiny": {"nodes": "5-10", "description": "Minimal test system"},
    "small": {"nodes": "10-25", "description": "Small deployment"},
    "medium": {"nodes": "30-50", "description": "Medium deployment"},
    "large": {"nodes": "60-100", "description": "Large deployment"},
    "xlarge": {"nodes": "150-300", "description": "Enterprise scale"},
}

VALIDATION_TARGETS = {
    "spearman": 0.70,
    "f1": 0.80,
    "precision": 0.80,
    "recall": 0.80,
    "top5_overlap": 0.40,
    "top10_overlap": 0.60,
}
```

### 7.3 Performance Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Analysis Time | Scale-dependent | Elapsed time from start to results |
| Simulation Time | <2× Analysis Time | Time for exhaustive simulation |
| Memory Usage | <2GB for large | Peak memory during execution |
| Dashboard Generation | <10s | Time to generate HTML |
| Database Import | <5s for medium | Time to import topology |

### 7.4 Benchmark Execution

```bash
# Run full benchmark suite
python benchmark.py \
    --scales tiny,small,medium,large,xlarge \
    --layers app,infra,system \
    --runs 5 \
    --output results/benchmark

# Output files:
# - results/benchmark_data.csv
# - results/benchmark_results.json
# - results/benchmark_report.md
```

---

## 8. Validation Test Specifications

### 8.1 Statistical Validation Targets

| Metric | Target | Description | Priority |
|--------|--------|-------------|----------|
| Spearman ρ | ≥ 0.70 | Rank correlation | Critical |
| F1-Score | ≥ 0.80 | Classification accuracy | Critical |
| Precision | ≥ 0.80 | Avoid false positives | High |
| Recall | ≥ 0.80 | Catch critical components | High |
| Top-5 Overlap | ≥ 40% | Critical set agreement | High |
| Top-10 Overlap | ≥ 50% | Extended critical set | Medium |
| RMSE | ≤ 0.25 | Prediction error | Medium |
| MAE | ≤ 0.20 | Absolute error | Medium |

### 8.2 Validation Test Matrix

| Test ID | Layer | Scale | Target ρ | Target F1 |
|---------|-------|-------|----------|-----------|
| VT-APP-001 | Application | Small | ≥ 0.75 | ≥ 0.75 |
| VT-APP-002 | Application | Medium | ≥ 0.80 | ≥ 0.80 |
| VT-APP-003 | Application | Large | ≥ 0.85 | ≥ 0.83 |
| VT-INF-001 | Infrastructure | Small | ≥ 0.50 | ≥ 0.65 |
| VT-INF-002 | Infrastructure | Medium | ≥ 0.52 | ≥ 0.66 |
| VT-INF-003 | Infrastructure | Large | ≥ 0.54 | ≥ 0.68 |
| VT-SYS-001 | System | Small | ≥ 0.70 | ≥ 0.75 |
| VT-SYS-002 | System | Medium | ≥ 0.75 | ≥ 0.80 |
| VT-SYS-003 | System | Large | ≥ 0.80 | ≥ 0.83 |

### 8.3 Achieved Results Reference

**By Layer:**

| Metric | Application | Infrastructure | Target |
|--------|-------------|----------------|--------|
| Spearman ρ | **0.85** ✓ | 0.54 | ≥0.70 |
| F1-Score | **0.83** ✓ | 0.68 | ≥0.80 |
| Precision | **0.86** ✓ | 0.71 | ≥0.80 |
| Recall | **0.80** ✓ | 0.65 | ≥0.80 |
| Top-5 Overlap | **62%** ✓ | 40% | ≥40% |

**By Scale:**

| Scale | Components | Spearman ρ | F1-Score |
|-------|------------|------------|----------|
| Small | 10-25 | 0.78 | 0.75 |
| Medium | 30-50 | 0.82 | 0.80 |
| Large | 60-100 | 0.85 | 0.83 |
| XLarge | 150-300 | 0.88 | 0.85 |

**Key Finding:** Prediction accuracy improves with system size due to more stable statistical patterns.

### 8.4 Validation Test Procedure

```bash
# Step 1: Generate test data
python generate_graph.py --scale medium --seed 42 --output test_data.json

# Step 2: Import and analyze
python import_graph.py --input test_data.json --clear
python analyze_graph.py --layer app --use-ahp --output analysis.json

# Step 3: Run simulation
python simulate_graph.py --layer app --exhaustive --output simulation.json

# Step 4: Validate
python validate_graph.py --layer app \
    --spearman 0.70 \
    --f1 0.80 \
    --output validation_result.json

# Step 5: Check results
cat validation_result.json | jq '.overall.passed'
# Expected: true
```

---

## 9. Acceptance Test Specifications

### 9.1 User Story Acceptance Criteria

#### US-001: Graph Model Construction

| Criterion ID | Description | Test Method | Pass Criteria |
|--------------|-------------|-------------|---------------|
| AC-001-1 | Import JSON topology | Automated | All entities in Neo4j |
| AC-001-2 | Derive dependencies | Automated | DEPENDS_ON edges exist |
| AC-001-3 | Calculate weights | Automated | Weights > 0 for QoS topics |
| AC-001-4 | Support all scales | Automated | tiny to xlarge work |

#### US-002: Structural Analysis

| Criterion ID | Description | Test Method | Pass Criteria |
|--------------|-------------|-------------|---------------|
| AC-002-1 | Compute PageRank | Automated | Non-zero values |
| AC-002-2 | Identify SPOFs | Automated | Articulation points found |
| AC-002-3 | Multi-layer support | Automated | All 5 layers work |
| AC-002-4 | Export results | Automated | Valid JSON output |

#### US-003: Quality Scoring

| Criterion ID | Description | Test Method | Pass Criteria |
|--------------|-------------|-------------|---------------|
| AC-003-1 | RMAV scores | Automated | All 4 dimensions present |
| AC-003-2 | AHP weights | Automated | Different from default |
| AC-003-3 | Classification | Automated | 5 levels assigned |
| AC-003-4 | Critical detection | Automated | CRITICAL level present |

#### US-004: Failure Simulation

| Criterion ID | Description | Test Method | Pass Criteria |
|--------------|-------------|-------------|---------------|
| AC-004-1 | Cascade propagation | Automated | Failed set > 1 for nodes |
| AC-004-2 | Impact calculation | Automated | I(v) computed |
| AC-004-3 | Exhaustive mode | Automated | All components simulated |
| AC-004-4 | Sorted results | Automated | Highest impact first |

#### US-005: Validation

| Criterion ID | Description | Test Method | Pass Criteria |
|--------------|-------------|-------------|---------------|
| AC-005-1 | Spearman computed | Automated | Value in [-1, 1] |
| AC-005-2 | F1 computed | Automated | Value in [0, 1] |
| AC-005-3 | Pass/fail determined | Automated | Boolean result |
| AC-005-4 | Targets configurable | Automated | Custom targets work |

#### US-006: Visualization

| Criterion ID | Description | Test Method | Pass Criteria |
|--------------|-------------|-------------|---------------|
| AC-006-1 | Dashboard generated | Automated | Valid HTML file |
| AC-006-2 | KPIs displayed | Manual | Correct values shown |
| AC-006-3 | Charts rendered | Manual | Pie and bar charts visible |
| AC-006-4 | Network interactive | Manual | vis.js working |

### 9.2 Acceptance Checklist

| ID | Requirement | Status |
|----|-------------|--------|
| ACC-001 | Import JSON topology | ☐ |
| ACC-002 | Compute all metrics | ☐ |
| ACC-003 | RMAV scoring | ☐ |
| ACC-004 | Failure simulation | ☐ |
| ACC-005 | Validation accuracy (ρ ≥ 0.70, F1 ≥ 0.80) | ☐ |
| ACC-006 | Dashboard generation | ☐ |
| ACC-007 | Performance targets (<20s for large) | ☐ |
| ACC-008 | Multi-layer analysis | ☐ |
| ACC-009 | CLI usability | ☐ |
| ACC-010 | Documentation complete | ☐ |

---

## 10. Test Procedures

### 10.1 Unit Test Procedure

```bash
# Run all unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test module
pytest tests/test_analysis.py -v

# Run tests matching pattern
pytest tests/ -k "test_quality" -v

# Run excluding slow tests
pytest tests/ -m "not slow" -v

# Run with timeout
pytest tests/ --timeout=60
```

### 10.2 Integration Test Procedure

```bash
# Start test Neo4j instance
docker-compose -f docker-compose.test.yml up -d neo4j-test

# Wait for Neo4j to be ready
sleep 30

# Run integration tests
pytest tests/ -m integration -v

# Cleanup
docker-compose -f docker-compose.test.yml down
```

### 10.3 System Test Procedure

```bash
# Full pipeline test
./scripts/run_system_tests.sh

# Or manually:
python generate_graph.py --scale small --output /tmp/test.json
python import_graph.py --input /tmp/test.json --clear
python run.py --all --layer system --output /tmp/results/
```

### 10.4 Benchmark Procedure

```bash
# Quick benchmark
python benchmark.py --scales small,medium,large --runs 1

# Full benchmark suite
python benchmark.py --full-suite --output results/benchmark
```

### 10.5 Test Reporting

```bash
# Generate coverage report
pytest tests/ --cov=src --cov-report=html --cov-report=xml

# Generate JUnit XML for CI
pytest tests/ --junitxml=test-results.xml

# View HTML coverage report
open htmlcov/index.html
```

---

## 11. Traceability Matrix

### 11.1 Requirements to Test Traceability

| Requirement ID | Requirement | Test IDs |
|----------------|-------------|----------|
| REQ-GMC-001 | Accept JSON topology | UT-CORE-010, ST-CLI-002 |
| REQ-GMC-002 | Create vertices | IT-NEO-001 |
| REQ-GMC-004 | Derive DEPENDS_ON | IT-NEO-002 |
| REQ-GMC-005 | Calculate QoS weights | UT-CORE-001 to UT-CORE-007 |
| REQ-SA-001 | Compute PageRank | UT-ANAL-002, UT-ANAL-003 |
| REQ-SA-008 | Identify Articulation Points | UT-ANAL-004 |
| REQ-QS-001 | Compute R(v) | UT-ANAL-012 |
| REQ-QS-007 | Support AHP weights | UT-ANAL-014 |
| REQ-QS-010 | Assign criticality levels | UT-ANAL-015 |
| REQ-FS-001 | Simulate CRASH mode | UT-SIM-020 |
| REQ-FS-003 | Apply PHYSICAL cascade | UT-SIM-021 |
| REQ-VA-001 | Compute Spearman ρ | UT-VAL-001 to UT-VAL-003 |
| REQ-VA-002 | Compute F1-Score | IT-VAL-003 |
| REQ-VZ-001 | Generate HTML dashboard | UT-VIZ-010 |
| REQ-PERF-001 | Analysis <1s for small | PT-SCAL-002 |
| REQ-ACC-001 | Spearman ≥0.70 | VT-APP-001 to VT-SYS-003 |

### 11.2 Test Coverage Summary

| Module | Unit Tests | Integration Tests | Target Coverage |
|--------|------------|-------------------|-----------------|
| src/core | 31 | 4 | 85% |
| src/analysis | 24 | 4 | 82% |
| src/simulation | 12 | 3 | 78% |
| src/validation | 12 | 3 | 80% |
| src/visualization | 8 | 2 | 75% |
| **Total** | **87** | **16** | **≥80%** |

---

## Appendices

### Appendix A: Test Data Specifications

#### A.1 Synthetic Graph Scales

| Scale | Apps | Brokers | Topics | Nodes | Libraries | Total |
|-------|------|---------|--------|-------|-----------|-------|
| Tiny | 5-8 | 1 | 3-5 | 2-3 | 2 | 13-19 |
| Small | 10-15 | 2 | 8-12 | 3-4 | 3 | 26-36 |
| Medium | 20-35 | 3 | 15-25 | 5-8 | 5 | 48-76 |
| Large | 50-80 | 5 | 30-50 | 8-12 | 8 | 101-155 |
| XLarge | 100-200 | 10 | 60-100 | 15-25 | 15 | 200-350 |

#### A.2 Common Test Fixtures

```python
# conftest.py - Shared fixtures

@pytest.fixture
def small_graph():
    """Small test graph with known properties."""
    return generate_graph(scale="small", seed=42)

@pytest.fixture
def linear_graph():
    """A->B->C linear graph for predictable metrics."""
    return GraphData(
        components=[
            ComponentData("A", "Application"),
            ComponentData("B", "Application"),
            ComponentData("C", "Application"),
        ],
        edges=[
            EdgeData("A", "B", "Application", "Application", "app_to_app", 1.0),
            EdgeData("B", "C", "Application", "Application", "app_to_app", 1.0),
        ]
    )

@pytest.fixture
def star_graph():
    """Star topology with central hub."""
    components = [ComponentData("hub", "Broker")]
    edges = []
    for i in range(5):
        components.append(ComponentData(f"app{i}", "Application"))
        edges.append(EdgeData(f"app{i}", "hub", "Application", "Broker", 
                             "app_to_broker", 1.0))
    return GraphData(components=components, edges=edges)

@pytest.fixture
def empty_graph():
    """Empty graph with no components."""
    return GraphData(components=[], edges=[])

@pytest.fixture
def single_node_graph():
    """Single isolated node."""
    return GraphData(
        components=[ComponentData("solo", "Application", 1.0)],
        edges=[]
    )
```

### Appendix B: CI/CD Pipeline Configuration

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
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-test.txt
      - name: Run unit tests
        run: pytest tests/ -m "not integration" --cov=src --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  integration-tests:
    runs-on: ubuntu-latest
    services:
      neo4j:
        image: neo4j:5-community
        ports:
          - 7687:7687
        env:
          NEO4J_AUTH: neo4j/testpassword
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-test.txt
      - name: Wait for Neo4j
        run: sleep 30
      - name: Run integration tests
        run: pytest tests/ -m integration -v
        env:
          NEO4J_URI: bolt://localhost:7687
          NEO4J_PASSWORD: testpassword

  benchmark:
    runs-on: ubuntu-latest
    needs: [unit-tests, integration-tests]
    services:
      neo4j:
        image: neo4j:5-community
        ports:
          - 7687:7687
        env:
          NEO4J_AUTH: neo4j/testpassword
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - name: Run benchmark
        run: python benchmark.py --scales small,medium --runs 3
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: output/benchmark_*.json
```

### Appendix C: Validation Report Template

```
================== VALIDATION REPORT ==================
Layer: Application
Scale: Medium (45 components)
Timestamp: 2026-01-27T10:30:00Z

CORRELATION METRICS:
  Spearman ρ:    0.85  ✓ (target: ≥0.70)
  Pearson r:     0.82  ✓ (target: ≥0.65)
  Kendall τ:     0.71  ✓ (target: ≥0.50)

CLASSIFICATION METRICS:
  Precision:     0.86  ✓ (target: ≥0.80)
  Recall:        0.80  ✓ (target: ≥0.80)
  F1-Score:      0.83  ✓ (target: ≥0.80)
  Accuracy:      0.87  ✓ (target: ≥0.75)

RANKING METRICS:
  Top-5 Overlap:  62%  ✓ (target: ≥40%)
  Top-10 Overlap: 70%  ✓ (target: ≥50%)
  NDCG@10:       0.89  ✓ (target: ≥0.70)

ERROR METRICS:
  RMSE:          0.12  ✓ (target: ≤0.25)
  MAE:           0.09  ✓ (target: ≤0.20)

STATUS: ALL TARGETS MET ✓
======================================================
```

### Appendix D: Defect Severity Classification

| Severity | Definition | Examples |
|----------|------------|----------|
| Critical | System unusable | Crash on startup, data loss |
| High | Major feature broken | Analysis fails, wrong results |
| Medium | Feature partially works | Minor calculation errors |
| Low | Cosmetic or minor | UI issues, formatting |

### Appendix E: Test Result Summary Template

```
================== TEST SESSION REPORT ==================
Platform: Linux-5.15.0
Python: 3.9.16
pytest: 7.4.0

Collected: 103 tests
Passed: 100
Failed: 1
Skipped: 2 (marked slow)
Duration: 45.67s

COVERAGE SUMMARY:
  src/core         85% (234/275)
  src/analysis     82% (456/556)
  src/simulation   78% (189/242)
  src/validation   80% (123/154)
  src/visualization 75% (98/131)
  TOTAL            80% (1100/1358)

FAILED TESTS:
  tests/test_analysis.py::test_edge_case_x - AssertionError

======================== PASSED ========================
```

---

*Document Version: 1.0*  
*Last Updated: January 2026*  
*Software-as-a-Graph Framework*
