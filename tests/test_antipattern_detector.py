"""
Tests for the consolidated AntiPatternDetector.
"""
import pytest
from types import SimpleNamespace
from saag.analysis.antipattern_detector import CATALOG, AntiPatternDetector
from saag.prediction.models import QualityAnalysisResult, DetectedProblem
from saag.core.metrics import ComponentQuality, EdgeQuality, EdgeMetrics, QualityScores, QualityLevels, StructuralMetrics
from saag.core.criticality import CriticalityLevel


def _comp(id_, type_="Application", scores=None, levels=None, structural_kwargs=None):
    """Small helper to build a ComponentQuality with sane defaults for tests."""
    scores = scores or {}
    levels = levels or {}
    structural_kwargs = structural_kwargs or {}
    return ComponentQuality(
        id=id_,
        type=type_,
        scores=QualityScores(**scores),
        levels=QualityLevels(**levels),
        structural=StructuralMetrics(id=id_, name=id_, type=type_, **structural_kwargs),
    )


def _edge(source, target, dependency_type="app_to_app", overall_score=0.0, weight=1.0, betweenness=0.0):
    return EdgeQuality(
        source=source, target=target,
        source_type="Application", target_type="Application",
        dependency_type=dependency_type,
        scores=QualityScores(overall=overall_score),
        structural=EdgeMetrics(
            source=source, target=target,
            source_type="Application", target_type="Application",
            dependency_type=dependency_type, betweenness=betweenness,
        ),
    )


def _shim(quality_result):
    return SimpleNamespace(
        quality=quality_result,
        components=quality_result.components,
        edges=quality_result.edges,
    )

@pytest.fixture
def mock_quality_result():
    """Create a mock QualityAnalysisResult with a SPOF and a God Component."""
    # Component A: SPOF (articulation point)
    comp_a = ComponentQuality(
        id="A",
        type="Application",
        scores=QualityScores(overall=0.8, reliability=0.8, maintainability=0.8, availability=0.8, security=0.2),
        levels=QualityLevels(overall=CriticalityLevel.HIGH, reliability=CriticalityLevel.HIGH, maintainability=CriticalityLevel.HIGH, availability=CriticalityLevel.HIGH, security=CriticalityLevel.LOW),
        structural=StructuralMetrics(id="A", name="A", type="Application", is_articulation_point=True, betweenness=0.4, pagerank=0.2, in_degree_raw=5, out_degree_raw=5)
    )
    
    # Component B: Normal
    comp_b = ComponentQuality(
        id="B",
        type="Application",
        scores=QualityScores(overall=0.4, reliability=0.4, maintainability=0.4, availability=0.4, security=0.2),
        levels=QualityLevels(overall=CriticalityLevel.MEDIUM, reliability=CriticalityLevel.MEDIUM, maintainability=CriticalityLevel.MEDIUM, availability=CriticalityLevel.MEDIUM, security=CriticalityLevel.LOW),
        structural=StructuralMetrics(id="B", name="B", type="Application", is_articulation_point=False, betweenness=0.1, pagerank=0.1, in_degree_raw=1, out_degree_raw=1)
    )
    
    return QualityAnalysisResult(
        timestamp="2026-03-22T00:00:00",
        layer="system",
        context="test",
        components=[comp_a, comp_b],
        edges=[],
        classification_summary=SimpleNamespace(total_components=2, component_distribution={"high": 1, "medium": 1})
    )

def test_detector_spof(mock_quality_result):
    """Verify SPOF detection."""
    detector = AntiPatternDetector()
    shim = SimpleNamespace(quality=mock_quality_result, components=mock_quality_result.components)
    problems = detector.detect(shim, "system")
    
    spofs = [p for p in problems if "Single Point of Failure" in p.name]
    assert len(spofs) == 1
    assert spofs[0].entity_id == "A"
    assert spofs[0].severity in ["CRITICAL", "HIGH"]

def test_detector_spof_availability_fence_fallback():
    """docs/antipatterns.md §5.1: SPOF(v) <-> AP_c(v) > 0 OR A(v) > upper_fence(A).

    A component with an outlier availability score, but that is NOT an
    articulation point, must still be flagged via the fence fallback.
    """
    detector = AntiPatternDetector(active_patterns=["SPOF"])
    comps = [_comp(f"N{i}", scores={"availability": 0.1}) for i in range(9)]
    comps.append(_comp("HOT", scores={"availability": 0.9}))
    result = QualityAnalysisResult(
        timestamp="t", layer="system", context="test", components=comps, edges=[],
        classification_summary=SimpleNamespace(total_components=len(comps), component_distribution={}),
    )
    problems = detector.detect(_shim(result), "system")
    assert len(problems) == 1
    assert problems[0].entity_id == "HOT"
    assert problems[0].evidence["trigger"] == "availability_fence"


def test_detector_god_component(mock_quality_result):
    """Verify God Component detection."""
    detector = AntiPatternDetector()
    # Modify comp_a to be CRITICAL and high betweenness
    mock_quality_result.components[0].levels.maintainability = CriticalityLevel.CRITICAL
    mock_quality_result.components[0].structural.betweenness = 0.5

    shim = SimpleNamespace(quality=mock_quality_result, components=mock_quality_result.components)
    problems = detector.detect(shim, "system")

    gods = [p for p in problems if "God Component" in p.name]
    assert len(gods) == 1
    assert gods[0].entity_id == "A"


def test_detector_god_component_regression_high_betweenness_only():
    """Regression: high betweenness alone (non-CRITICAL maintainability) must not flag GOD_COMPONENT."""
    detector = AntiPatternDetector(active_patterns=["GOD_COMPONENT"])
    comp = _comp(
        "X", levels={"maintainability": CriticalityLevel.MEDIUM},
        structural_kwargs={"betweenness": 0.9},
    )
    result = QualityAnalysisResult(
        timestamp="t", layer="system", context="test", components=[comp], edges=[],
        classification_summary=SimpleNamespace(total_components=1, component_distribution={}),
    )
    problems = detector.detect(_shim(result), "system")
    assert problems == []


def test_detector_systemic_risk_ratio():
    """Regression: SYSTEMIC_RISK is a population-ratio check, not a clique/adjacency check."""
    detector = AntiPatternDetector(active_patterns=["SYSTEMIC_RISK"])
    # 2 of 5 components CRITICAL overall = 40% > 20% threshold, with no edges between them.
    comps = [
        _comp("A", levels={"overall": CriticalityLevel.CRITICAL}),
        _comp("B", levels={"overall": CriticalityLevel.CRITICAL}),
        _comp("C", levels={"overall": CriticalityLevel.MEDIUM}),
        _comp("D", levels={"overall": CriticalityLevel.MEDIUM}),
        _comp("E", levels={"overall": CriticalityLevel.MEDIUM}),
    ]
    result = QualityAnalysisResult(
        timestamp="t", layer="system", context="test", components=comps, edges=[],
        classification_summary=SimpleNamespace(total_components=5, component_distribution={}),
    )
    problems = detector.detect(_shim(result), "system")
    assert len(problems) == 1
    assert problems[0].name == "Systemic Risk Pattern"


def test_bottleneck_edge_category():
    assert CATALOG["BOTTLENECK_EDGE"].category == "Availability"


def test_bottleneck_edge_severity_is_high():
    """docs/antipatterns.md §5.5 documents BOTTLENECK_EDGE as HIGH severity."""
    assert CATALOG["BOTTLENECK_EDGE"].severity == "HIGH"


def test_detector_bottleneck_edge_adaptive_fence():
    """BOTTLENECK_EDGE uses an adaptive Q3+1.5*IQR fence over edge betweenness,
    not a hardcoded constant, so only the true outlier edge is flagged."""
    detector = AntiPatternDetector(active_patterns=["BOTTLENECK_EDGE"])
    chain = [chr(ord("A") + i) for i in range(11)]
    comps = [_comp(cid) for cid in chain]
    edges = [_edge(chain[i], chain[i + 1], betweenness=0.05) for i in range(9)]
    edges.append(_edge(chain[9], chain[10], betweenness=0.5))
    result = QualityAnalysisResult(
        timestamp="t", layer="system", context="test", components=comps, edges=edges,
        classification_summary=SimpleNamespace(total_components=len(chain), component_distribution={}),
    )
    problems = detector.detect(_shim(result), "system")
    assert len(problems) == 1
    assert problems[0].entity_id == f"{chain[9]}->{chain[10]}"


def test_catalog_names_avoid_prescription_collision():
    """Guard: prescription/service.py substring-matches PatternSpec.name for 'Bottleneck'/'Hub'.

    Scoped to the 7 patterns newly added in this change — FAILURE_HUB is a pre-existing
    pattern whose name already contains "Hub" (a known, pre-existing collision, not
    introduced here) and is intentionally excluded from this guard.
    """
    new_patterns = {
        "BROKER_OVERLOAD", "TOPIC_FANOUT", "QOS_MISMATCH",
        "DEEP_PIPELINE", "CHATTY_PAIR", "ORPHANED_TOPIC", "UNSTABLE_INTERFACE",
    }
    for pid in new_patterns:
        spec = CATALOG[pid]
        assert "Bottleneck" not in spec.name, f"{pid} name collides with prescription god_components matching"
        assert "Hub" not in spec.name, f"{pid} name collides with prescription god_components matching"


def test_detector_broker_overload_sole_broker():
    detector = AntiPatternDetector(active_patterns=["BROKER_OVERLOAD"])
    broker = _comp("B1", type_="Broker", scores={"availability": 0.5})
    result = QualityAnalysisResult(
        timestamp="t", layer="mw", context="test", components=[broker], edges=[],
        classification_summary=SimpleNamespace(total_components=1, component_distribution={}),
    )
    problems = detector.detect(_shim(result), "mw")
    assert len(problems) == 1
    assert problems[0].entity_id == "B1"
    assert problems[0].evidence["sole_broker"] is True


def test_detector_broker_overload_imbalanced():
    detector = AntiPatternDetector(active_patterns=["BROKER_OVERLOAD"])
    brokers = [
        _comp("B1", type_="Broker", scores={"availability": 0.8}),
        _comp("B2", type_="Broker", scores={"availability": 0.2}),
        _comp("B3", type_="Broker", scores={"availability": 0.2}),
    ]
    result = QualityAnalysisResult(
        timestamp="t", layer="mw", context="test", components=brokers, edges=[],
        classification_summary=SimpleNamespace(total_components=3, component_distribution={}),
    )
    problems = detector.detect(_shim(result), "mw")
    assert len(problems) == 1
    assert problems[0].entity_id == "B1"


def test_detector_topic_fanout():
    detector = AntiPatternDetector(active_patterns=["TOPIC_FANOUT"])
    topics = [
        _comp(f"T{i}", type_="Topic", structural_kwargs={"topic_subscriber_count": 2})
        for i in range(6)
    ] + [_comp("T-hot", type_="Topic", structural_kwargs={"topic_subscriber_count": 50})]
    result = QualityAnalysisResult(
        timestamp="t", layer="system", context="test", components=topics, edges=[],
        classification_summary=SimpleNamespace(total_components=7, component_distribution={}),
    )
    problems = detector.detect(_shim(result), "system")
    assert len(problems) == 1
    assert problems[0].entity_id == "T-hot"


def test_detector_orphaned_topic_publisher_only():
    detector = AntiPatternDetector(active_patterns=["ORPHANED_TOPIC"])
    topic = _comp("T1", type_="Topic", structural_kwargs={"topic_subscriber_count": 0, "topic_publisher_count": 1})
    result = QualityAnalysisResult(
        timestamp="t", layer="system", context="test", components=[topic], edges=[],
        classification_summary=SimpleNamespace(total_components=1, component_distribution={}),
    )
    problems = detector.detect(_shim(result), "system")
    assert len(problems) == 1
    assert problems[0].evidence["orphan_type"] == "publisher_only"


def test_detector_orphaned_topic_subscriber_only():
    detector = AntiPatternDetector(active_patterns=["ORPHANED_TOPIC"])
    topic = _comp("T1", type_="Topic", structural_kwargs={"topic_subscriber_count": 3, "topic_publisher_count": 0})
    result = QualityAnalysisResult(
        timestamp="t", layer="system", context="test", components=[topic], edges=[],
        classification_summary=SimpleNamespace(total_components=1, component_distribution={}),
    )
    problems = detector.detect(_shim(result), "system")
    assert len(problems) == 1
    assert problems[0].evidence["orphan_type"] == "subscriber_only"


def test_detector_orphaned_topic_fully_disconnected_not_double_flagged():
    """A topic with zero publishers AND zero subscribers is left to ISOLATED, not ORPHANED_TOPIC."""
    detector = AntiPatternDetector(active_patterns=["ORPHANED_TOPIC"])
    topic = _comp("T1", type_="Topic", structural_kwargs={"topic_subscriber_count": 0, "topic_publisher_count": 0})
    result = QualityAnalysisResult(
        timestamp="t", layer="system", context="test", components=[topic], edges=[],
        classification_summary=SimpleNamespace(total_components=1, component_distribution={}),
    )
    problems = detector.detect(_shim(result), "system")
    assert problems == []


def test_detector_qos_mismatch():
    detector = AntiPatternDetector(active_patterns=["QOS_MISMATCH"])
    pub = _comp("PUB", structural_kwargs={"weight": 0.1})
    sub = _comp("SUB", structural_kwargs={"weight": 0.9})
    edge = _edge("PUB", "SUB")
    result = QualityAnalysisResult(
        timestamp="t", layer="system", context="test", components=[pub, sub], edges=[edge],
        classification_summary=SimpleNamespace(total_components=2, component_distribution={}),
    )
    problems = detector.detect(_shim(result), "system")
    assert len(problems) == 1
    assert problems[0].entity_id == "PUB->SUB"


def test_detector_qos_mismatch_no_gap():
    detector = AntiPatternDetector(active_patterns=["QOS_MISMATCH"])
    pub = _comp("PUB", structural_kwargs={"weight": 0.8})
    sub = _comp("SUB", structural_kwargs={"weight": 0.9})
    edge = _edge("PUB", "SUB")
    result = QualityAnalysisResult(
        timestamp="t", layer="system", context="test", components=[pub, sub], edges=[edge],
        classification_summary=SimpleNamespace(total_components=2, component_distribution={}),
    )
    problems = detector.detect(_shim(result), "system")
    assert problems == []


def test_detector_chatty_pair():
    detector = AntiPatternDetector(active_patterns=["CHATTY_PAIR"])
    a, b = _comp("A"), _comp("B")
    edges = [_edge("A", "B", overall_score=0.6), _edge("B", "A", overall_score=0.6)]
    result = QualityAnalysisResult(
        timestamp="t", layer="app", context="test", components=[a, b], edges=edges,
        classification_summary=SimpleNamespace(total_components=2, component_distribution={}),
    )
    problems = detector.detect(_shim(result), "app")
    assert len(problems) == 1


def test_detector_chatty_pair_one_directional():
    detector = AntiPatternDetector(active_patterns=["CHATTY_PAIR"])
    a, b = _comp("A"), _comp("B")
    edges = [_edge("A", "B", overall_score=0.9)]
    result = QualityAnalysisResult(
        timestamp="t", layer="app", context="test", components=[a, b], edges=edges,
        classification_summary=SimpleNamespace(total_components=2, component_distribution={}),
    )
    problems = detector.detect(_shim(result), "app")
    assert problems == []


def test_detector_deep_pipeline():
    detector = AntiPatternDetector(active_patterns=["DEEP_PIPELINE"])
    chain = ["A", "B", "C", "D", "E", "F"]
    comps = [_comp(cid) for cid in chain]
    edges = [_edge(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]
    result = QualityAnalysisResult(
        timestamp="t", layer="app", context="test", components=comps, edges=edges,
        classification_summary=SimpleNamespace(total_components=len(chain), component_distribution={}),
    )
    problems = detector.detect(_shim(result), "app")
    assert len(problems) == 1
    assert problems[0].evidence["hops"] == 5


def test_detector_deep_pipeline_short_chain_not_flagged():
    detector = AntiPatternDetector(active_patterns=["DEEP_PIPELINE"])
    chain = ["A", "B", "C"]
    comps = [_comp(cid) for cid in chain]
    edges = [_edge(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]
    result = QualityAnalysisResult(
        timestamp="t", layer="app", context="test", components=comps, edges=edges,
        classification_summary=SimpleNamespace(total_components=len(chain), component_distribution={}),
    )
    problems = detector.detect(_shim(result), "app")
    assert problems == []


def test_detector_unstable_interface():
    detector = AntiPatternDetector(active_patterns=["UNSTABLE_INTERFACE"])
    comp = _comp(
        "X", scores={"maintainability": 0.9},
        structural_kwargs={"coupling_risk_enh": 0.9},
    )
    result = QualityAnalysisResult(
        timestamp="t", layer="app", context="test", components=[comp], edges=[],
        classification_summary=SimpleNamespace(total_components=1, component_distribution={}),
    )
    problems = detector.detect(_shim(result), "app")
    assert len(problems) == 1


def test_detector_unstable_interface_low_coupling_not_flagged():
    detector = AntiPatternDetector(active_patterns=["UNSTABLE_INTERFACE"])
    comp = _comp(
        "X", scores={"maintainability": 0.9},
        structural_kwargs={"coupling_risk_enh": 0.2},
    )
    result = QualityAnalysisResult(
        timestamp="t", layer="app", context="test", components=[comp], edges=[],
        classification_summary=SimpleNamespace(total_components=1, component_distribution={}),
    )
    problems = detector.detect(_shim(result), "app")
    assert problems == []

def test_problem_detector_wrapper(mock_quality_result):
    """Verify the backward-compatible ProblemDetector wrapper."""
    from saag.prediction.problem_detector import ProblemDetector
    
    wrapper = ProblemDetector()
    problems = wrapper.detect(mock_quality_result)
    
    assert len(problems) > 0
    assert isinstance(problems[0], DetectedProblem)
    
    summary = wrapper.summarize(problems)
    assert summary.total_problems == len(problems)
    assert summary.affected_components >= 1
