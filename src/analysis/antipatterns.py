"""
Anti-Pattern Detector - Version 4.0

Detects architectural anti-patterns in pub-sub systems that affect:
- Reliability: SPOFs, cascade risks
- Maintainability: God topics, tight coupling, circular dependencies  
- Availability: Bottlenecks, partition risks

Anti-Patterns Detected:
1. God Topic - Topic with excessive publishers/subscribers
2. Single Point of Failure (SPOF) - Articulation points
3. Bottleneck Broker - Broker with disproportionate load
4. Chatty Publisher - App publishing to too many topics
5. Single Consumer/Producer - Topic with only one subscriber/publisher
6. Star Topology - Central hub with many peripherals
7. Deep Dependency Chain - Excessively long dependency paths

Author: Software-as-a-Graph Research Project
Version: 4.0
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional
from collections import defaultdict


# =============================================================================
# Enums
# =============================================================================

class AntiPatternType(Enum):
    """Types of architectural anti-patterns"""
    GOD_TOPIC = "god_topic"
    SINGLE_POINT_OF_FAILURE = "spof"
    BOTTLENECK_BROKER = "bottleneck_broker"
    CHATTY_PUBLISHER = "chatty_publisher"
    SINGLE_CONSUMER = "single_consumer"
    SINGLE_PRODUCER = "single_producer"
    STAR_TOPOLOGY = "star_topology"
    DEEP_DEPENDENCY_CHAIN = "deep_chain"
    CIRCULAR_DEPENDENCY = "circular_dependency"


class PatternSeverity(Enum):
    """Severity of detected anti-patterns"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

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
    """A detected anti-pattern"""
    pattern_type: AntiPatternType
    severity: PatternSeverity
    affected_components: List[str]
    description: str
    impact: str
    recommendation: str
    quality_attributes: List[str]
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
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
    """Result from anti-pattern detection"""
    patterns: List[AntiPattern]
    summary: Dict[str, int]
    recommendations: List[str]

    def to_dict(self) -> Dict:
        return {
            "total_patterns": len(self.patterns),
            "summary": self.summary,
            "patterns": [p.to_dict() for p in self.patterns],
            "recommendations": self.recommendations,
        }

    def by_severity(self, severity: PatternSeverity) -> List[AntiPattern]:
        """Get patterns by severity"""
        return [p for p in self.patterns if p.severity == severity]

    def by_type(self, pattern_type: AntiPatternType) -> List[AntiPattern]:
        """Get patterns by type"""
        return [p for p in self.patterns if p.pattern_type == pattern_type]


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "god_topic_threshold": 8,          # Total connections
    "bottleneck_ratio": 3.0,           # vs average
    "chatty_threshold": 6,             # Topics per publisher
    "deep_chain_threshold": 4,         # Dependency depth
    "star_center_ratio": 4.0,          # Center degree vs periphery
}


# =============================================================================
# Anti-Pattern Detector
# =============================================================================

class AntiPatternDetector:
    """
    Detects architectural anti-patterns using Neo4j queries.
    
    Queries the graph database directly for pattern detection,
    avoiding the need to load the entire graph into memory.
    """

    def __init__(self, gds_client, config: Optional[Dict] = None):
        self.gds = gds_client
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.logger = logging.getLogger(__name__)

    def detect_all(self) -> AntiPatternResult:
        """
        Detect all anti-patterns.
        
        Returns:
            AntiPatternResult with all detected patterns
        """
        self.logger.info("Starting anti-pattern detection...")
        
        patterns = []
        
        # Run all detectors
        patterns.extend(self._detect_god_topics())
        patterns.extend(self._detect_bottleneck_brokers())
        patterns.extend(self._detect_chatty_publishers())
        patterns.extend(self._detect_single_consumer_producer())
        patterns.extend(self._detect_star_topology())
        patterns.extend(self._detect_circular_dependencies())
        
        # Generate summary
        summary = self._generate_summary(patterns)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(patterns)
        
        self.logger.info(f"Detected {len(patterns)} anti-patterns")
        
        return AntiPatternResult(
            patterns=patterns,
            summary=summary,
            recommendations=recommendations,
        )

    def _detect_god_topics(self) -> List[AntiPattern]:
        """Detect topics with excessive connections"""
        patterns = []
        threshold = self.config["god_topic_threshold"]
        
        query = """
        MATCH (t:Topic)
        OPTIONAL MATCH (pub:Application)-[:PUBLISHES_TO]->(t)
        OPTIONAL MATCH (sub:Application)-[:SUBSCRIBES_TO]->(t)
        WITH t, 
             count(DISTINCT pub) AS publishers,
             count(DISTINCT sub) AS subscribers,
             collect(DISTINCT pub.id) AS pub_ids,
             collect(DISTINCT sub.id) AS sub_ids
        WHERE publishers + subscribers >= $threshold
        RETURN t.id AS topic, t.name AS name,
               publishers, subscribers,
               pub_ids, sub_ids
        ORDER BY publishers + subscribers DESC
        """
        
        with self.gds.session() as session:
            for record in session.run(query, threshold=threshold):
                total = record["publishers"] + record["subscribers"]
                severity = PatternSeverity.CRITICAL if total >= threshold * 2 else PatternSeverity.HIGH
                
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.GOD_TOPIC,
                    severity=severity,
                    affected_components=[record["topic"]],
                    description=f"Topic '{record['name']}' has {total} connections "
                               f"({record['publishers']} publishers, {record['subscribers']} subscribers)",
                    impact="Single point of failure; changes affect many components; "
                           "hard to evolve without breaking dependents",
                    recommendation="Split into focused topics by domain or message type",
                    quality_attributes=["reliability", "maintainability"],
                    metrics={
                        "total_connections": total,
                        "publishers": record["publishers"],
                        "subscribers": record["subscribers"],
                    },
                ))
        
        return patterns

    def _detect_bottleneck_brokers(self) -> List[AntiPattern]:
        """Detect brokers with disproportionate load"""
        patterns = []
        ratio_threshold = self.config["bottleneck_ratio"]
        
        query = """
        MATCH (b:Broker)-[:ROUTES]->(t:Topic)
        WITH b, count(t) AS topic_count
        WITH b, topic_count, avg(topic_count) AS avg_topics
        WHERE topic_count > avg_topics * $ratio
        RETURN b.id AS broker, b.name AS name,
               topic_count, avg_topics
        ORDER BY topic_count DESC
        """
        
        with self.gds.session() as session:
            for record in session.run(query, ratio=ratio_threshold):
                ratio = record["topic_count"] / record["avg_topics"] if record["avg_topics"] > 0 else 0
                severity = PatternSeverity.HIGH if ratio > ratio_threshold * 1.5 else PatternSeverity.MEDIUM
                
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.BOTTLENECK_BROKER,
                    severity=severity,
                    affected_components=[record["broker"]],
                    description=f"Broker '{record['name']}' routes {record['topic_count']} topics "
                               f"({ratio:.1f}x average)",
                    impact="Performance bottleneck; single point of failure for message routing",
                    recommendation="Distribute topics across multiple brokers",
                    quality_attributes=["reliability", "availability"],
                    metrics={
                        "topic_count": record["topic_count"],
                        "average": record["avg_topics"],
                        "ratio": ratio,
                    },
                ))
        
        return patterns

    def _detect_chatty_publishers(self) -> List[AntiPattern]:
        """Detect applications publishing to too many topics"""
        patterns = []
        threshold = self.config["chatty_threshold"]
        
        query = """
        MATCH (a:Application)-[:PUBLISHES_TO]->(t:Topic)
        WITH a, count(t) AS topic_count, collect(t.id) AS topics
        WHERE topic_count >= $threshold
        RETURN a.id AS app, a.name AS name,
               topic_count, topics
        ORDER BY topic_count DESC
        """
        
        with self.gds.session() as session:
            for record in session.run(query, threshold=threshold):
                severity = PatternSeverity.MEDIUM if record["topic_count"] < threshold * 2 else PatternSeverity.HIGH
                
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.CHATTY_PUBLISHER,
                    severity=severity,
                    affected_components=[record["app"]],
                    description=f"App '{record['name']}' publishes to {record['topic_count']} topics",
                    impact="High coupling; difficult to understand responsibilities; "
                           "changes may have wide impact",
                    recommendation="Split into focused microservices with clear boundaries",
                    quality_attributes=["maintainability"],
                    metrics={
                        "topic_count": record["topic_count"],
                        "topics": record["topics"][:10],
                    },
                ))
        
        return patterns

    def _detect_single_consumer_producer(self) -> List[AntiPattern]:
        """Detect topics with only one subscriber or publisher"""
        patterns = []
        
        # Single consumer
        query_consumer = """
        MATCH (t:Topic)
        OPTIONAL MATCH (sub:Application)-[:SUBSCRIBES_TO]->(t)
        WITH t, count(sub) AS sub_count
        WHERE sub_count = 1
        MATCH (sub:Application)-[:SUBSCRIBES_TO]->(t)
        RETURN t.id AS topic, t.name AS name, sub.id AS subscriber
        """
        
        with self.gds.session() as session:
            for record in session.run(query_consumer):
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.SINGLE_CONSUMER,
                    severity=PatternSeverity.LOW,
                    affected_components=[record["topic"], record["subscriber"]],
                    description=f"Topic '{record['name']}' has only one subscriber",
                    impact="No redundancy; if subscriber fails, messages are lost",
                    recommendation="Consider adding consumer groups or dead-letter queues",
                    quality_attributes=["reliability"],
                    metrics={"subscriber": record["subscriber"]},
                ))
        
        # Single producer
        query_producer = """
        MATCH (t:Topic)
        OPTIONAL MATCH (pub:Application)-[:PUBLISHES_TO]->(t)
        WITH t, count(pub) AS pub_count
        WHERE pub_count = 1
        MATCH (pub:Application)-[:PUBLISHES_TO]->(t)
        RETURN t.id AS topic, t.name AS name, pub.id AS publisher
        """
        
        with self.gds.session() as session:
            for record in session.run(query_producer):
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.SINGLE_PRODUCER,
                    severity=PatternSeverity.LOW,
                    affected_components=[record["topic"], record["publisher"]],
                    description=f"Topic '{record['name']}' has only one publisher",
                    impact="Single point of failure for data production",
                    recommendation="Consider redundant producers for critical data",
                    quality_attributes=["reliability", "availability"],
                    metrics={"publisher": record["publisher"]},
                ))
        
        return patterns

    def _detect_star_topology(self) -> List[AntiPattern]:
        """Detect star topology in dependencies"""
        patterns = []
        ratio = self.config["star_center_ratio"]
        
        query = """
        MATCH (center)-[d:DEPENDS_ON]-()
        WITH center, count(d) AS degree
        WHERE degree >= 5
        WITH center, degree, avg(degree) AS avg_degree
        WHERE degree > avg_degree * $ratio
        RETURN center.id AS center, labels(center)[0] AS type, degree, avg_degree
        ORDER BY degree DESC
        LIMIT 5
        """
        
        with self.gds.session() as session:
            for record in session.run(query, ratio=ratio):
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.STAR_TOPOLOGY,
                    severity=PatternSeverity.HIGH,
                    affected_components=[record["center"]],
                    description=f"{record['type']} '{record['center']}' is hub with {record['degree']} connections",
                    impact="Central point of failure; bottleneck for communication",
                    recommendation="Distribute responsibilities to reduce centralization",
                    quality_attributes=["reliability", "availability"],
                    metrics={
                        "degree": record["degree"],
                        "avg_degree": record["avg_degree"],
                    },
                ))
        
        return patterns

    def _detect_circular_dependencies(self) -> List[AntiPattern]:
        """Detect circular dependencies"""
        patterns = []
        
        query = """
        MATCH path = (a)-[:DEPENDS_ON*2..5]->(a)
        WITH nodes(path) AS cycle
        WHERE size(cycle) > 1
        RETURN [n IN cycle | n.id] AS nodes
        LIMIT 10
        """
        
        with self.gds.session() as session:
            for record in session.run(query):
                nodes = record["nodes"]
                patterns.append(AntiPattern(
                    pattern_type=AntiPatternType.CIRCULAR_DEPENDENCY,
                    severity=PatternSeverity.HIGH,
                    affected_components=nodes,
                    description=f"Circular dependency: {' -> '.join(nodes[:5])}{'...' if len(nodes) > 5 else ''}",
                    impact="Potential deadlocks; hard to reason about; difficult testing",
                    recommendation="Break cycle using events, callbacks, or dependency inversion",
                    quality_attributes=["maintainability", "reliability"],
                    metrics={"cycle_length": len(nodes), "nodes": nodes},
                ))
        
        return patterns

    def _generate_summary(self, patterns: List[AntiPattern]) -> Dict[str, int]:
        """Generate summary statistics"""
        by_severity = defaultdict(int)
        by_type = defaultdict(int)
        
        for p in patterns:
            by_severity[p.severity.value] += 1
            by_type[p.pattern_type.value] += 1
        
        return {
            "total": len(patterns),
            "by_severity": dict(by_severity),
            "by_type": dict(by_type),
        }

    def _generate_recommendations(self, patterns: List[AntiPattern]) -> List[str]:
        """Generate prioritized recommendations"""
        recommendations = []
        
        critical = [p for p in patterns if p.severity == PatternSeverity.CRITICAL]
        high = [p for p in patterns if p.severity == PatternSeverity.HIGH]
        
        if critical:
            recommendations.append(f"🔴 URGENT: Address {len(critical)} critical anti-patterns")
        
        if high:
            recommendations.append(f"⚠️ Address {len(high)} high-severity patterns")
        
        # Type-specific
        type_counts = defaultdict(int)
        for p in patterns:
            type_counts[p.pattern_type] += 1
        
        if type_counts[AntiPatternType.GOD_TOPIC] > 0:
            recommendations.append(f"Split {type_counts[AntiPatternType.GOD_TOPIC]} god topics")
        
        if type_counts[AntiPatternType.CIRCULAR_DEPENDENCY] > 0:
            recommendations.append(f"Break {type_counts[AntiPatternType.CIRCULAR_DEPENDENCY]} circular dependencies")
        
        if type_counts[AntiPatternType.SINGLE_POINT_OF_FAILURE] > 0:
            recommendations.append(f"Add redundancy for {type_counts[AntiPatternType.SINGLE_POINT_OF_FAILURE]} SPOFs")
        
        if not recommendations:
            recommendations.append("✅ No significant anti-patterns detected")
        
        return recommendations
