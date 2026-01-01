"""
Anti-Pattern Detector - Version 5.0

Detects common architectural anti-patterns in pub-sub systems.

Anti-Patterns Detected:
- God Topic: Topic with too many publishers/subscribers
- Single Point of Failure: Component with no redundancy
- Bottleneck Broker: Broker handling too much traffic
- Chatty Application: App with excessive messaging
- Hub-and-Spoke: Over-centralized topology
- Circular Dependency: Components depending on each other in a cycle

Each detection includes severity, impact assessment, and remediation guidance.

Author: Software-as-a-Graph Research Project
Version: 5.0
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Set

from .gds_client import GDSClient
from .classifier import BoxPlotClassifier, CriticalityLevel


# =============================================================================
# Enums
# =============================================================================

class AntiPatternType(Enum):
    """Types of anti-patterns that can be detected"""
    GOD_TOPIC = "god_topic"
    SINGLE_POINT_OF_FAILURE = "spof"
    BOTTLENECK_BROKER = "bottleneck_broker"
    CHATTY_APPLICATION = "chatty_app"
    HUB_AND_SPOKE = "hub_and_spoke"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    ORPHAN_COMPONENT = "orphan_component"
    TOPIC_EXPLOSION = "topic_explosion"
    
    @property
    def description(self) -> str:
        return {
            AntiPatternType.GOD_TOPIC: "Topic with too many connections",
            AntiPatternType.SINGLE_POINT_OF_FAILURE: "Component with no redundancy",
            AntiPatternType.BOTTLENECK_BROKER: "Broker handling too much traffic",
            AntiPatternType.CHATTY_APPLICATION: "App with excessive messaging",
            AntiPatternType.HUB_AND_SPOKE: "Over-centralized topology",
            AntiPatternType.CIRCULAR_DEPENDENCY: "Circular dependency chain",
            AntiPatternType.ORPHAN_COMPONENT: "Component with no connections",
            AntiPatternType.TOPIC_EXPLOSION: "Too many similar topics",
        }[self]


class PatternSeverity(Enum):
    """Severity levels for anti-patterns"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    
    @property
    def numeric(self) -> int:
        return {
            PatternSeverity.CRITICAL: 4,
            PatternSeverity.HIGH: 3,
            PatternSeverity.MEDIUM: 2,
            PatternSeverity.LOW: 1,
        }[self]
    
    @property
    def color(self) -> str:
        return {
            PatternSeverity.CRITICAL: "\033[91m",
            PatternSeverity.HIGH: "\033[93m",
            PatternSeverity.MEDIUM: "\033[94m",
            PatternSeverity.LOW: "\033[92m",
        }[self]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class AntiPattern:
    """
    A detected anti-pattern instance.
    """
    pattern_type: AntiPatternType
    severity: PatternSeverity
    affected_components: List[str]
    description: str
    impact: str
    recommendation: str
    quality_attributes: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.pattern_type.value,
            "severity": self.severity.value,
            "affected_components": self.affected_components,
            "description": self.description,
            "impact": self.impact,
            "recommendation": self.recommendation,
            "quality_attributes": self.quality_attributes,
            "metrics": self.metrics,
        }


@dataclass
class AntiPatternResult:
    """
    Complete result from anti-pattern detection.
    """
    timestamp: str
    patterns: List[AntiPattern]
    by_type: Dict[AntiPatternType, List[AntiPattern]]
    by_severity: Dict[PatternSeverity, List[AntiPattern]]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "patterns": [p.to_dict() for p in self.patterns],
            "summary": self.summary,
            "by_type": {
                t.value: [p.to_dict() for p in patterns]
                for t, patterns in self.by_type.items()
            },
            "by_severity": {
                s.value: [p.to_dict() for p in patterns]
                for s, patterns in self.by_severity.items()
            },
        }
    
    @property
    def critical_count(self) -> int:
        return len(self.by_severity.get(PatternSeverity.CRITICAL, []))
    
    @property
    def total_count(self) -> int:
        return len(self.patterns)


# =============================================================================
# Anti-Pattern Detector
# =============================================================================

class AntiPatternDetector:
    """
    Detects architectural anti-patterns in pub-sub systems.
    
    Uses graph analysis to identify common anti-patterns that
    affect reliability, maintainability, and scalability.
    
    Example:
        with GDSClient(uri, user, password) as gds:
            detector = AntiPatternDetector(gds)
            result = detector.detect_all()
            
            for pattern in result.patterns:
                print(f"[{pattern.severity.value}] {pattern.pattern_type.value}")
                print(f"  Components: {pattern.affected_components}")
    """

    def __init__(
        self,
        gds_client: GDSClient,
        god_topic_threshold: float = 0.3,
        bottleneck_ratio: float = 0.5,
        k_factor: float = 1.5,
    ):
        """
        Initialize detector.
        
        Args:
            gds_client: Connected GDS client
            god_topic_threshold: Fraction of total connections for god topic
            bottleneck_ratio: Ratio threshold for bottleneck detection
            k_factor: Box-plot k factor
        """
        self.gds = gds_client
        self.god_topic_threshold = god_topic_threshold
        self.bottleneck_ratio = bottleneck_ratio
        self.classifier = BoxPlotClassifier(k_factor=k_factor)
        self.logger = logging.getLogger(__name__)

    def detect_all(self) -> AntiPatternResult:
        """
        Run all anti-pattern detections.
        
        Returns:
            AntiPatternResult with all detected patterns
        """
        timestamp = datetime.now().isoformat()
        patterns: List[AntiPattern] = []
        
        self.logger.info("Starting anti-pattern detection")
        
        # Run all detections
        patterns.extend(self._detect_god_topics())
        patterns.extend(self._detect_bottleneck_brokers())
        patterns.extend(self._detect_chatty_applications())
        patterns.extend(self._detect_hub_and_spoke())
        patterns.extend(self._detect_orphan_components())
        
        # Organize results
        by_type = self._group_by_type(patterns)
        by_severity = self._group_by_severity(patterns)
        summary = self._generate_summary(patterns, by_type, by_severity)
        
        return AntiPatternResult(
            timestamp=timestamp,
            patterns=patterns,
            by_type=by_type,
            by_severity=by_severity,
            summary=summary,
        )

    def _detect_god_topics(self) -> List[AntiPattern]:
        """
        Detect God Topic anti-pattern.
        
        A God Topic is a topic with too many publishers or subscribers,
        creating a bottleneck and single point of failure.
        """
        patterns = []
        
        query = """
        MATCH (t:Topic)
        OPTIONAL MATCH (t)<-[:PUBLISHES_TO]-(pub:Application)
        OPTIONAL MATCH (t)-[:SUBSCRIBES_TO]->(sub:Application)
        WITH t, 
             count(DISTINCT pub) AS publishers,
             count(DISTINCT sub) AS subscribers,
             count(DISTINCT pub) + count(DISTINCT sub) AS total_connections
        WHERE total_connections > 0
        RETURN t.id AS topic_id,
               publishers,
               subscribers,
               total_connections
        ORDER BY total_connections DESC
        """
        
        with self.gds.session() as session:
            results = list(session.run(query))
        
        if not results:
            return patterns
        
        # Calculate total and classify
        total_all = sum(r["total_connections"] for r in results)
        
        items = [
            {
                "id": r["topic_id"],
                "type": "Topic",
                "score": r["total_connections"],
                "publishers": r["publishers"],
                "subscribers": r["subscribers"],
            }
            for r in results
        ]
        
        classification = self.classifier.classify(items, metric_name="topic_connections")
        
        for item in classification.get_critical():
            fraction = item.score / total_all if total_all > 0 else 0
            
            if fraction >= self.god_topic_threshold or item.is_outlier:
                severity = PatternSeverity.CRITICAL if fraction >= 0.5 else PatternSeverity.HIGH
                
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.GOD_TOPIC,
                    severity=severity,
                    affected_components=[item.id],
                    description=(
                        f"Topic '{item.id}' handles {item.score:.0f} connections "
                        f"({fraction:.1%} of total), making it a god topic."
                    ),
                    impact=(
                        "Single point of failure, scalability bottleneck, "
                        "difficult to maintain and evolve"
                    ),
                    recommendation=(
                        "Split into multiple domain-specific topics, "
                        "use topic hierarchies, or implement message routing"
                    ),
                    quality_attributes=["reliability", "maintainability", "scalability"],
                    metrics={
                        "total_connections": item.score,
                        "publishers": item.metadata.get("publishers", 0),
                        "subscribers": item.metadata.get("subscribers", 0),
                        "fraction_of_total": fraction,
                    },
                ))
        
        return patterns

    def _detect_bottleneck_brokers(self) -> List[AntiPattern]:
        """
        Detect Bottleneck Broker anti-pattern.
        
        A broker that routes disproportionately more traffic than others.
        """
        patterns = []
        
        query = """
        MATCH (b:Broker)
        OPTIONAL MATCH (b)-[:ROUTES]->(t:Topic)
        WITH b, count(DISTINCT t) AS routed_topics
        RETURN b.id AS broker_id,
               routed_topics
        ORDER BY routed_topics DESC
        """
        
        with self.gds.session() as session:
            results = list(session.run(query))
        
        if len(results) < 2:
            return patterns
        
        total_topics = sum(r["routed_topics"] for r in results)
        
        if total_topics == 0:
            return patterns
        
        items = [
            {"id": r["broker_id"], "type": "Broker", "score": r["routed_topics"]}
            for r in results
        ]
        classification = self.classifier.classify(items, metric_name="broker_topics")
        
        for item in classification.get_critical():
            fraction = item.score / total_topics
            
            if fraction >= self.bottleneck_ratio:
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.BOTTLENECK_BROKER,
                    severity=PatternSeverity.HIGH,
                    affected_components=[item.id],
                    description=(
                        f"Broker '{item.id}' routes {item.score:.0f} topics "
                        f"({fraction:.1%} of total), creating a bottleneck."
                    ),
                    impact=(
                        "Performance bottleneck, reduced availability, "
                        "single point of failure for many topics"
                    ),
                    recommendation=(
                        "Distribute topics across multiple brokers, "
                        "implement broker clustering, or use topic-based routing"
                    ),
                    quality_attributes=["availability", "performance", "scalability"],
                    metrics={
                        "routed_topics": item.score,
                        "fraction_of_total": fraction,
                    },
                ))
        
        return patterns

    def _detect_chatty_applications(self) -> List[AntiPattern]:
        """
        Detect Chatty Application anti-pattern.
        
        An application that publishes/subscribes to too many topics.
        """
        patterns = []
        
        query = """
        MATCH (a:Application)
        OPTIONAL MATCH (a)-[:PUBLISHES_TO]->(pub_topic:Topic)
        OPTIONAL MATCH (a)-[:SUBSCRIBES_TO]->(sub_topic:Topic)
        WITH a,
             count(DISTINCT pub_topic) AS publish_count,
             count(DISTINCT sub_topic) AS subscribe_count
        WITH a, publish_count, subscribe_count,
             publish_count + subscribe_count AS total_topics
        WHERE total_topics > 0
        RETURN a.id AS app_id,
               publish_count,
               subscribe_count,
               total_topics
        ORDER BY total_topics DESC
        """
        
        with self.gds.session() as session:
            results = list(session.run(query))
        
        if not results:
            return patterns
        
        items = [
            {
                "id": r["app_id"],
                "type": "Application",
                "score": r["total_topics"],
                "publish_count": r["publish_count"],
                "subscribe_count": r["subscribe_count"],
            }
            for r in results
        ]
        classification = self.classifier.classify(items, metric_name="app_topics")
        
        for item in classification.get_critical():
            patterns.append(AntiPattern(
                pattern_type=AntiPatternType.CHATTY_APPLICATION,
                severity=PatternSeverity.MEDIUM,
                affected_components=[item.id],
                description=(
                    f"Application '{item.id}' interacts with {item.score:.0f} topics "
                    f"(publishes: {item.metadata.get('publish_count', 0)}, "
                    f"subscribes: {item.metadata.get('subscribe_count', 0)})"
                ),
                impact=(
                    "High coupling, difficult to test and deploy independently, "
                    "potential performance issues"
                ),
                recommendation=(
                    "Split into focused microservices, use aggregation patterns, "
                    "or implement message batching"
                ),
                quality_attributes=["maintainability", "testability"],
                metrics={
                    "total_topics": item.score,
                    "publish_count": item.metadata.get("publish_count", 0),
                    "subscribe_count": item.metadata.get("subscribe_count", 0),
                },
            ))
        
        return patterns

    def _detect_hub_and_spoke(self) -> List[AntiPattern]:
        """
        Detect Hub-and-Spoke anti-pattern.
        
        Over-centralized topology where most traffic goes through one node.
        """
        patterns = []
        
        # Create a temporary projection
        projection_name = "antipattern_hub_spoke"
        
        try:
            self.gds.create_projection(
                projection_name,
                dependency_types=["app_to_app", "node_to_node"],
            )
            
            bc_results = self.gds.betweenness(projection_name)
            
            if not bc_results:
                return patterns
            
            items = [
                {"id": r.node_id, "type": r.node_type, "score": r.score}
                for r in bc_results
            ]
            classification = self.classifier.classify(items, metric_name="betweenness")
            
            # Check if one node dominates
            if classification.items:
                top_score = classification.items[0].score
                total_score = sum(item.score for item in classification.items)
                
                if total_score > 0:
                    concentration = top_score / total_score
                    
                    if concentration > 0.5:  # One node has >50% of betweenness
                        top_item = classification.items[0]
                        
                        patterns.append(AntiPattern(
                            pattern_type=AntiPatternType.HUB_AND_SPOKE,
                            severity=PatternSeverity.HIGH,
                            affected_components=[top_item.id],
                            description=(
                                f"System has hub-and-spoke topology centered on '{top_item.id}'. "
                                f"This component handles {concentration:.1%} of all shortest paths."
                            ),
                            impact=(
                                "Single point of failure, performance bottleneck, "
                                "poor fault tolerance"
                            ),
                            recommendation=(
                                "Introduce mesh topology, add redundant paths, "
                                "distribute routing across multiple hubs"
                            ),
                            quality_attributes=["reliability", "availability", "scalability"],
                            metrics={
                                "betweenness_concentration": concentration,
                                "hub_betweenness": top_score,
                            },
                        ))
        
        finally:
            self.gds.drop_projection(projection_name)
        
        return patterns

    def _detect_orphan_components(self) -> List[AntiPattern]:
        """
        Detect Orphan Component anti-pattern.
        
        Components with no connections (dead code or misconfiguration).
        """
        patterns = []
        
        query = """
        MATCH (n)
        WHERE (n:Application OR n:Broker OR n:Node)
        AND NOT (n)-[:DEPENDS_ON]-()
        AND NOT (n)-[:PUBLISHES_TO|SUBSCRIBES_TO|ROUTES|RUNS_ON|CONNECTS_TO]-()
        RETURN n.id AS component_id, labels(n)[0] AS component_type
        """
        
        orphans = []
        with self.gds.session() as session:
            for record in session.run(query):
                orphans.append({
                    "id": record["component_id"],
                    "type": record["component_type"],
                })
        
        if orphans:
            patterns.append(AntiPattern(
                pattern_type=AntiPatternType.ORPHAN_COMPONENT,
                severity=PatternSeverity.LOW,
                affected_components=[o["id"] for o in orphans],
                description=(
                    f"Found {len(orphans)} orphan component(s) with no connections: "
                    f"{', '.join(o['id'] for o in orphans[:5])}"
                    f"{'...' if len(orphans) > 5 else ''}"
                ),
                impact="Dead code, wasted resources, potential misconfiguration",
                recommendation=(
                    "Remove unused components or connect them to the system. "
                    "Verify configuration is correct."
                ),
                quality_attributes=["maintainability"],
                metrics={
                    "orphan_count": len(orphans),
                    "orphans": orphans[:10],  # Limit to first 10
                },
            ))
        
        return patterns

    def _group_by_type(
        self, 
        patterns: List[AntiPattern]
    ) -> Dict[AntiPatternType, List[AntiPattern]]:
        """Group patterns by type"""
        result: Dict[AntiPatternType, List[AntiPattern]] = {}
        for pattern in patterns:
            if pattern.pattern_type not in result:
                result[pattern.pattern_type] = []
            result[pattern.pattern_type].append(pattern)
        return result

    def _group_by_severity(
        self, 
        patterns: List[AntiPattern]
    ) -> Dict[PatternSeverity, List[AntiPattern]]:
        """Group patterns by severity"""
        result: Dict[PatternSeverity, List[AntiPattern]] = {
            sev: [] for sev in PatternSeverity
        }
        for pattern in patterns:
            result[pattern.severity].append(pattern)
        return result

    def _generate_summary(
        self,
        patterns: List[AntiPattern],
        by_type: Dict[AntiPatternType, List[AntiPattern]],
        by_severity: Dict[PatternSeverity, List[AntiPattern]],
    ) -> Dict[str, Any]:
        """Generate summary statistics"""
        return {
            "total_patterns": len(patterns),
            "by_type": {t.value: len(p) for t, p in by_type.items()},
            "by_severity": {s.value: len(p) for s, p in by_severity.items()},
            "critical_count": len(by_severity.get(PatternSeverity.CRITICAL, [])),
            "affected_components": list(set(
                comp for p in patterns for comp in p.affected_components
            )),
        }
