"""
QoS-Aware Component Analyzer

Analyzes Quality of Service policies and their impact on system criticality.
Supports DDS-style QoS policies including durability, reliability, deadline,
lifespan, transport priority, and history depth.
"""

import networkx as nx
from typing import Dict, Optional, List, Tuple
import logging
from dataclasses import dataclass


@dataclass
class QoSAnalysisResult:
    """Result of QoS analysis for a component"""
    component: str
    component_type: str
    qos_score: float
    durability: Optional[str] = None
    reliability: Optional[str] = None
    deadline_ms: Optional[float] = None
    lifespan_ms: Optional[float] = None
    transport_priority: Optional[int] = None
    history_depth: Optional[int] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'component': self.component,
            'component_type': self.component_type,
            'qos_score': round(self.qos_score, 3),
            'durability': self.durability,
            'reliability': self.reliability,
            'deadline_ms': self.deadline_ms,
            'lifespan_ms': self.lifespan_ms,
            'transport_priority': self.transport_priority,
            'history_depth': self.history_depth
        }


class QoSAnalyzer:
    """
    Analyzer for QoS policies in pub-sub systems
    
    Analyzes QoS policies at multiple levels:
    - Topic-level: Direct QoS policies
    - Application-level: Derived from published/subscribed topics
    - Broker-level: Aggregated from routed topics
    - Node-level: Aggregated from hosted components
    """
    
    def __init__(self, 
                 durability_weight: float = 0.20,
                 reliability_weight: float = 0.25,
                 deadline_weight: float = 0.20,
                 lifespan_weight: float = 0.10,
                 transport_priority_weight: float = 0.15,
                 history_weight: float = 0.10):
        """
        Initialize QoS analyzer with configurable weights
        
        Args:
            durability_weight: Weight for durability policy (0-1)
            reliability_weight: Weight for reliability policy (0-1)
            deadline_weight: Weight for deadline policy (0-1)
            lifespan_weight: Weight for lifespan policy (0-1)
            transport_priority_weight: Weight for transport priority (0-1)
            history_weight: Weight for history depth (0-1)
        """
        # Validate weights sum to 1.0
        total = (durability_weight + reliability_weight + deadline_weight + 
                lifespan_weight + transport_priority_weight + history_weight)
        
        if not (0.99 <= total <= 1.01):  # Allow small floating point error
            raise ValueError(f"QoS weights must sum to 1.0, got {total}")
        
        self.weights = {
            'durability': durability_weight,
            'reliability': reliability_weight,
            'deadline': deadline_weight,
            'lifespan': lifespan_weight,
            'transport_priority': transport_priority_weight,
            'history': history_weight
        }
        
        self.logger = logging.getLogger(__name__)
        
        # Caches
        self._topic_scores = {}
        self._app_scores = {}
    
    def analyze_graph(self, 
                     graph: nx.DiGraph,
                     model=None) -> Dict:
        """
        Analyze QoS policies across the entire graph
        
        Args:
            graph: NetworkX directed graph
            model: Optional GraphModel with enhanced QoS data
        
        Returns:
            Dictionary with comprehensive QoS analysis results
        """
        self.logger.info("Starting QoS analysis...")
        
        # Clear caches
        self._topic_scores.clear()
        self._app_scores.clear()
        
        results = {
            'component_scores': {},
            'high_priority_topics': [],
            'high_priority_applications': [],
            'qos_violations': [],
            'compatibility_issues': [],
            'recommendations': [],
            'summary': {}
        }
        
        # Step 1: Analyze topics
        topic_results = self._analyze_topics(graph, model)
        results['component_scores'].update(topic_results)
        
        # Step 2: Analyze applications based on their topics
        app_results = self._analyze_applications(graph)
        results['component_scores'].update(app_results)
        
        # Step 3: Analyze brokers based on routed topics
        broker_results = self._analyze_brokers(graph)
        results['component_scores'].update(broker_results)
        
        # Step 4: Analyze nodes based on hosted components
        node_results = self._analyze_nodes(graph)
        results['component_scores'].update(node_results)
        
        # Step 5: Identify high priority components
        results['high_priority_topics'] = self._identify_high_priority_topics(graph)
        results['high_priority_applications'] = self._identify_high_priority_applications(graph)
        
        # Step 6: Check QoS compatibility
        results['compatibility_issues'] = self._check_qos_compatibility(graph)
        
        # Step 7: Generate recommendations
        results['recommendations'] = self._generate_qos_recommendations(graph, results)
        
        # Step 8: Generate summary
        results['summary'] = self._generate_summary(results)
        
        self.logger.info(f"QoS analysis complete: {len(results['component_scores'])} components analyzed")
        
        return results
    
    def _analyze_topics(self, graph: nx.DiGraph, model=None) -> Dict[str, float]:
        """Analyze QoS policies for all topics"""
        
        topic_scores = {}
        
        for node, data in graph.nodes(data=True):
            if data.get('type') != 'Topic':
                continue
            
            # Get QoS score from node attributes (if already calculated)
            if 'qos_score' in data:
                score = data['qos_score']
            else:
                # Calculate from individual QoS policies
                score = self._calculate_topic_qos_score(data)
            
            topic_scores[node] = score
            self._topic_scores[node] = score
        
        return topic_scores
    
    def _calculate_topic_qos_score(self, topic_data: Dict) -> float:
        """
        Calculate QoS criticality score for a topic
        
        Args:
            topic_data: Dictionary with QoS policy attributes
        
        Returns:
            QoS score [0, 1]
        """
        import math
        
        # Durability score
        durability_map = {
            'VOLATILE': 0.2,
            'TRANSIENT_LOCAL': 0.5,
            'TRANSIENT': 0.7,
            'PERSISTENT': 1.0
        }
        durability = topic_data.get('durability', 'VOLATILE')
        durability_score = durability_map.get(durability, 0.2)
        
        # Reliability score
        reliability = topic_data.get('reliability', 'BEST_EFFORT')
        reliability_score = 1.0 if reliability == 'RELIABLE' else 0.3
        
        # Deadline score (inverse exponential - shorter deadline = more critical)
        deadline_ms = topic_data.get('deadline_ms')
        if deadline_ms is None or deadline_ms == float('inf'):
            deadline_score = 0.1
        else:
            deadline_score = 1.0 - math.exp(-1000 / max(deadline_ms, 1))
        
        # Lifespan score (log transformation - longer lifespan = more critical)
        lifespan_ms = topic_data.get('lifespan_ms')
        if lifespan_ms is None or lifespan_ms == float('inf'):
            lifespan_score = 0.1
        else:
            lifespan_score = math.log(lifespan_ms + 1) / math.log(86400000)  # Normalized to 24h
        
        # Transport priority score (normalized to 0-1)
        transport_priority = topic_data.get('transport_priority', 0)
        transport_score = min(1.0, transport_priority / 100)
        
        # History depth score (log scale)
        history_depth = topic_data.get('history_depth', 1)
        history_score = min(1.0, math.log(history_depth + 1) / math.log(100))
        
        # Composite score
        score = (
            self.weights['durability'] * durability_score +
            self.weights['reliability'] * reliability_score +
            self.weights['deadline'] * deadline_score +
            self.weights['lifespan'] * lifespan_score +
            self.weights['transport_priority'] * transport_score +
            self.weights['history'] * history_score
        )
        
        return round(score, 3)
    
    def _analyze_applications(self, graph: nx.DiGraph) -> Dict[str, float]:
        """
        Analyze applications based on topics they publish/subscribe to
        
        Application QoS score = 0.7 * avg(published_topics) + 0.3 * avg(subscribed_topics)
        """
        
        app_scores = {}
        
        for node, data in graph.nodes(data=True):
            if data.get('type') != 'Application':
                continue
            
            # Find published topics
            published_topics = []
            subscribed_topics = []
            
            for source, target, edge_data in graph.out_edges(node, data=True):
                edge_type = edge_data.get('type')
                
                if edge_type == 'PUBLISHES_TO':
                    published_topics.append(target)
                elif edge_type == 'SUBSCRIBES_TO':
                    subscribed_topics.append(target)
            
            # Calculate application score
            pub_scores = [self._topic_scores.get(t, 0.0) for t in published_topics]
            sub_scores = [self._topic_scores.get(t, 0.0) for t in subscribed_topics]
            
            app_score = 0.0
            if pub_scores:
                app_score += 0.7 * (sum(pub_scores) / len(pub_scores))
            if sub_scores:
                app_score += 0.3 * (sum(sub_scores) / len(sub_scores))
            
            app_scores[node] = round(app_score, 3)
            self._app_scores[node] = app_scores[node]
        
        return app_scores
    
    def _analyze_brokers(self, graph: nx.DiGraph) -> Dict[str, float]:
        """
        Analyze brokers based on topics they route
        
        Broker QoS score = weighted average of routed topics
        """
        
        broker_scores = {}
        
        for node, data in graph.nodes(data=True):
            if data.get('type') != 'Broker':
                continue
            
            # Find routed topics
            routed_topics = []
            
            for source, target, edge_data in graph.out_edges(node, data=True):
                if edge_data.get('type') == 'ROUTES':
                    routed_topics.append(target)
            
            # Calculate broker score as average of topic scores
            if routed_topics:
                topic_scores = [self._topic_scores.get(t, 0.0) for t in routed_topics]
                broker_score = sum(topic_scores) / len(topic_scores)
            else:
                broker_score = 0.0
            
            broker_scores[node] = round(broker_score, 3)
        
        return broker_scores
    
    def _analyze_nodes(self, graph: nx.DiGraph) -> Dict[str, float]:
        """
        Analyze infrastructure nodes based on hosted components
        
        Node QoS score = weighted average of hosted applications and brokers
        """
        
        node_scores = {}
        
        for node, data in graph.nodes(data=True):
            if data.get('type') != 'Node':
                continue
            
            # Find hosted applications and brokers
            hosted_apps = []
            hosted_brokers = []
            
            for source, target, edge_data in graph.in_edges(node, data=True):
                if edge_data.get('type') == 'RUNS_ON':
                    source_type = graph.nodes[source].get('type')
                    if source_type == 'Application':
                        hosted_apps.append(source)
                    elif source_type == 'Broker':
                        hosted_brokers.append(source)
            
            # Calculate node score
            all_scores = []
            
            # Get application scores (weight 0.6)
            for app in hosted_apps:
                if app in self._app_scores:
                    all_scores.append(self._app_scores[app] * 0.6)
            
            # Get broker scores (weight 0.4)
            for broker in hosted_brokers:
                broker_topics = [
                    target for source, target, edge_data in graph.out_edges(broker, data=True)
                    if edge_data.get('type') == 'ROUTES'
                ]
                if broker_topics:
                    broker_topic_scores = [self._topic_scores.get(t, 0.0) for t in broker_topics]
                    broker_score = sum(broker_topic_scores) / len(broker_topic_scores)
                    all_scores.append(broker_score * 0.4)
            
            node_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
            node_scores[node] = round(node_score, 3)
        
        return node_scores
    
    def _identify_high_priority_topics(self, graph: nx.DiGraph) -> List[Dict]:
        """Identify topics with high QoS requirements"""
        
        high_priority = []
        
        for node, data in graph.nodes(data=True):
            if data.get('type') != 'Topic':
                continue
            
            score = self._topic_scores.get(node, 0.0)
            
            if score > 0.7:  # High priority threshold
                high_priority.append({
                    'name': node,
                    'score': score,
                    'durability': data.get('durability', 'VOLATILE'),
                    'reliability': data.get('reliability', 'BEST_EFFORT'),
                    'deadline_ms': data.get('deadline_ms'),
                    'transport_priority': data.get('transport_priority', 0)
                })
        
        # Sort by score (highest first)
        high_priority.sort(key=lambda x: x['score'], reverse=True)
        
        return high_priority
    
    def _identify_high_priority_applications(self, graph: nx.DiGraph) -> List[Dict]:
        """Identify applications with high QoS requirements"""
        
        high_priority = []
        
        for node, data in graph.nodes(data=True):
            if data.get('type') != 'Application':
                continue
            
            score = self._app_scores.get(node, 0.0)
            
            if score > 0.6:  # High priority threshold for apps
                # Find published topics
                pub_topics = [
                    target for source, target, edge_data in graph.out_edges(node, data=True)
                    if edge_data.get('type') == 'PUBLISHES_TO'
                ]
                
                high_priority.append({
                    'name': node,
                    'score': score,
                    'app_type': data.get('app_type', 'Unknown'),
                    'published_topics': pub_topics,
                    'business_domain': data.get('business_domain', 'Unknown')
                })
        
        # Sort by score (highest first)
        high_priority.sort(key=lambda x: x['score'], reverse=True)
        
        return high_priority
    
    def _check_qos_compatibility(self, graph: nx.DiGraph) -> List[Dict]:
        """
        Check for QoS compatibility issues between publishers and subscribers
        
        Returns list of potential compatibility issues
        """
        
        issues = []
        
        # For each topic, check publisher/subscriber QoS compatibility
        for topic, topic_data in graph.nodes(data=True):
            if topic_data.get('type') != 'Topic':
                continue
            
            topic_durability = topic_data.get('durability', 'VOLATILE')
            topic_reliability = topic_data.get('reliability', 'BEST_EFFORT')
            
            # Find publishers and subscribers
            publishers = [
                source for source, target, edge_data in graph.in_edges(topic, data=True)
                if edge_data.get('type') == 'PUBLISHES_TO'
            ]
            
            subscribers = [
                source for source, target, edge_data in graph.out_edges(topic, data=True)
                if edge_data.get('type') == 'SUBSCRIBES_TO'
            ]
            
            # Check for common issues
            
            # Issue 1: PERSISTENT topic with no RELIABLE subscribers
            if topic_durability == 'PERSISTENT' and topic_reliability == 'RELIABLE':
                for sub in subscribers:
                    # In a real implementation, check subscriber's QoS expectations
                    # For now, flag topics with high QoS and many subscribers
                    if len(subscribers) > 5:
                        issues.append({
                            'type': 'HIGH_QOS_MANY_SUBSCRIBERS',
                            'topic': topic,
                            'durability': topic_durability,
                            'reliability': topic_reliability,
                            'subscriber_count': len(subscribers),
                            'severity': 'MEDIUM',
                            'description': f'High-QoS topic {topic} has many subscribers which may cause performance issues'
                        })
                        break
            
            # Issue 2: BEST_EFFORT reliability but PERSISTENT durability (inconsistent)
            if topic_durability in ['PERSISTENT', 'TRANSIENT'] and topic_reliability == 'BEST_EFFORT':
                issues.append({
                    'type': 'INCONSISTENT_QOS',
                    'topic': topic,
                    'durability': topic_durability,
                    'reliability': topic_reliability,
                    'severity': 'LOW',
                    'description': f'Topic {topic} has durable policy but best-effort reliability'
                })
        
        return issues
    
    def _generate_qos_recommendations(self, graph: nx.DiGraph, results: Dict) -> List[Dict]:
        """Generate QoS-related recommendations"""
        
        recommendations = []
        
        # Recommendation 1: High-priority topics without dedicated resources
        for topic_info in results['high_priority_topics'][:5]:  # Top 5
            recommendations.append({
                'priority': 'HIGH',
                'type': 'QoS_Monitoring',
                'component': topic_info['name'],
                'issue': f"High-priority topic (QoS score: {topic_info['score']:.2f})",
                'recommendation': f"Ensure dedicated monitoring and resources for {topic_info['name']}",
                'details': {
                    'durability': topic_info['durability'],
                    'reliability': topic_info['reliability'],
                    'deadline_ms': topic_info['deadline_ms']
                }
            })
        
        # Recommendation 2: QoS compatibility issues
        for issue in results['compatibility_issues']:
            if issue['severity'] == 'HIGH':
                priority = 'HIGH'
            elif issue['severity'] == 'MEDIUM':
                priority = 'MEDIUM'
            else:
                priority = 'LOW'
            
            recommendations.append({
                'priority': priority,
                'type': 'QoS_Compatibility',
                'component': issue['topic'],
                'issue': issue['description'],
                'recommendation': 'Review and align QoS policies',
                'details': issue
            })
        
        # Recommendation 3: High-priority applications
        for app_info in results['high_priority_applications'][:3]:  # Top 3
            recommendations.append({
                'priority': 'MEDIUM',
                'type': 'QoS_Application',
                'component': app_info['name'],
                'issue': f"Application with high QoS requirements (score: {app_info['score']:.2f})",
                'recommendation': f"Ensure {app_info['name']} has adequate resources and monitoring",
                'details': {
                    'published_topics': app_info['published_topics'],
                    'business_domain': app_info['business_domain']
                }
            })
        
        return recommendations
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate summary statistics"""
        
        scores = list(results['component_scores'].values())
        
        if not scores:
            return {
                'total_components': 0,
                'avg_qos_score': 0.0,
                'high_priority_count': 0,
                'compatibility_issues': 0
            }
        
        return {
            'total_components': len(scores),
            'avg_qos_score': round(sum(scores) / len(scores), 3),
            'max_qos_score': round(max(scores), 3),
            'min_qos_score': round(min(scores), 3),
            'high_priority_topics': len(results['high_priority_topics']),
            'high_priority_applications': len(results['high_priority_applications']),
            'compatibility_issues': len(results['compatibility_issues']),
            'recommendations': len(results['recommendations'])
        }
    
    def analyze_topic(self, topic_name: str, graph: nx.DiGraph) -> Optional[QoSAnalysisResult]:
        """
        Analyze a specific topic's QoS policies
        
        Args:
            topic_name: Name of the topic
            graph: NetworkX graph containing the topic
        
        Returns:
            QoSAnalysisResult or None if topic not found
        """
        
        if topic_name not in graph.nodes:
            return None
        
        topic_data = graph.nodes[topic_name]
        
        if topic_data.get('type') != 'Topic':
            return None
        
        score = self._calculate_topic_qos_score(topic_data)
        
        return QoSAnalysisResult(
            component=topic_name,
            component_type='Topic',
            qos_score=score,
            durability=topic_data.get('durability'),
            reliability=topic_data.get('reliability'),
            deadline_ms=topic_data.get('deadline_ms'),
            lifespan_ms=topic_data.get('lifespan_ms'),
            transport_priority=topic_data.get('transport_priority'),
            history_depth=topic_data.get('history_depth')
        )
    
    def compare_topics(self, topic1: str, topic2: str, graph: nx.DiGraph) -> Dict:
        """
        Compare QoS policies of two topics
        
        Args:
            topic1: First topic name
            topic2: Second topic name
            graph: NetworkX graph
        
        Returns:
            Comparison results
        """
        
        result1 = self.analyze_topic(topic1, graph)
        result2 = self.analyze_topic(topic2, graph)
        
        if not result1 or not result2:
            return {'error': 'One or both topics not found'}
        
        return {
            'topic1': result1.to_dict(),
            'topic2': result2.to_dict(),
            'score_difference': abs(result1.qos_score - result2.qos_score),
            'more_critical': topic1 if result1.qos_score > result2.qos_score else topic2,
            'differences': {
                'durability': result1.durability != result2.durability,
                'reliability': result1.reliability != result2.reliability,
                'has_deadline': (result1.deadline_ms is not None) != (result2.deadline_ms is not None)
            }
        }
