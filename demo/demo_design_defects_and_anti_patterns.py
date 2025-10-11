import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class GodTopicDetector:
    """
    Detect topics that have become central hubs with too many responsibilities
    """
    
    def detect_god_topics(self, graph, threshold_percentile=90):
        """
        Identify topics with excessive connections indicating god topic anti-pattern
        """
        god_topics = []
        
        # Calculate degree for all topics
        topic_degrees = {}
        for node in graph.nodes():
            if graph.nodes[node].get('type') == 'topic':
                in_degree = graph.in_degree(node)   # Publishers
                out_degree = graph.out_degree(node)  # Subscribers
                total_degree = in_degree + out_degree
                topic_degrees[node] = {
                    'total': total_degree,
                    'publishers': in_degree,
                    'subscribers': out_degree
                }
        
        # Calculate threshold
        all_degrees = [d['total'] for d in topic_degrees.values()]
        threshold = np.percentile(all_degrees, threshold_percentile)
        
        for topic, degrees in topic_degrees.items():
            if degrees['total'] > threshold:
                # Analyze the nature of connections
                publishers = [n for n in graph.predecessors(topic) 
                            if graph.nodes[n].get('type') == 'application']
                subscribers = [n for n in graph.successors(topic) 
                             if graph.nodes[n].get('type') == 'application']
                
                # Check for mixed responsibilities
                message_types = self._analyze_message_types(graph, topic, publishers)
                
                god_topics.append({
                    'topic': topic,
                    'severity': self._calculate_god_topic_severity(degrees, len(message_types)),
                    'total_connections': degrees['total'],
                    'publishers': publishers,
                    'subscribers': subscribers,
                    'message_type_count': len(message_types),
                    'recommendation': self._generate_god_topic_recommendation(topic, message_types)
                })
        
        return sorted(god_topics, key=lambda x: x['severity'], reverse=True)
    
    def _analyze_message_types(self, graph, topic, publishers):
        """
        Infer different message types being published to the same topic
        """
        message_types = set()
        for pub in publishers:
            # Infer from publisher name/type
            pub_type = graph.nodes[pub].get('business_domain', '')
            if pub_type:
                message_types.add(pub_type)
        return message_types
    
    def _calculate_god_topic_severity(self, degrees, message_type_count):
        """
        Calculate severity of god topic anti-pattern
        """
        # High connection count + multiple message types = severe problem
        connection_score = min(1.0, degrees['total'] / 20)
        diversity_score = min(1.0, message_type_count / 5)
        return (connection_score * 0.6) + (diversity_score * 0.4)
    
    def _generate_god_topic_recommendation(self, topic, message_types):
        """
        Generate specific recommendations for god topic refactoring
        """
        if len(message_types) > 1:
            return f"Split '{topic}' into domain-specific topics: {', '.join([f'{mt}.{topic}' for mt in message_types])}"
        else:
            return f"Consider partitioning '{topic}' or implementing topic hierarchies"
        
class ChattyCommunicationDetector:
    """
    Detect excessive fine-grained communication between services
    """
    
    def detect_chatty_communication(self, graph, time_window_ms=1000):
        """
        Identify service pairs with excessive message exchange
        """
        chatty_patterns = []
        
        # Analyze service-to-service communication patterns
        service_pairs = self._find_service_pairs(graph)
        
        for (service1, service2), paths in service_pairs.items():
            # Calculate message exchange metrics
            total_msg_rate = 0
            avg_msg_size = []
            round_trips = 0
            
            for path in paths:
                # Analyze each communication path
                if len(path) == 3:  # Service -> Topic -> Service
                    topic = path[1]
                    edge_data = graph.edges[(path[0], topic)]
                    msg_rate = edge_data.get('msg_rate_per_sec', 0)
                    msg_size = edge_data.get('avg_msg_size_kb', 0)
                    
                    total_msg_rate += msg_rate
                    avg_msg_size.append(msg_size)
                    
                    # Check for request-response pattern
                    reverse_path = self._find_reverse_path(graph, service2, service1)
                    if reverse_path:
                        round_trips += 1
            
            # Detect chatty pattern
            if total_msg_rate > 50 and np.mean(avg_msg_size) < 1:  # Many small messages
                chatty_patterns.append({
                    'services': (service1, service2),
                    'severity': self._calculate_chatty_severity(total_msg_rate, avg_msg_size, round_trips),
                    'total_msg_rate': total_msg_rate,
                    'avg_msg_size_kb': np.mean(avg_msg_size),
                    'round_trips': round_trips,
                    'paths': paths,
                    'recommendation': self._generate_chatty_recommendation(service1, service2, round_trips)
                })
        
        return sorted(chatty_patterns, key=lambda x: x['severity'], reverse=True)
    
    def _find_service_pairs(self, graph):
        """
        Find all service pairs that communicate
        """
        service_pairs = {}
        
        for node in graph.nodes():
            if graph.nodes[node].get('type') == 'application':
                # Find all services this service communicates with
                for topic in graph.successors(node):
                    if graph.nodes[topic].get('type') == 'topic':
                        for target_service in graph.successors(topic):
                            if graph.nodes[target_service].get('type') == 'application':
                                pair = (node, target_service)
                                if pair not in service_pairs:
                                    service_pairs[pair] = []
                                service_pairs[pair].append([node, topic, target_service])
        
        return service_pairs
    
    def _calculate_chatty_severity(self, msg_rate, msg_sizes, round_trips):
        """
        Calculate severity of chatty communication
        """
        rate_score = min(1.0, msg_rate / 100)
        size_score = 1.0 - min(1.0, np.mean(msg_sizes))  # Smaller messages = higher score
        round_trip_score = min(1.0, round_trips / 5)
        
        return (rate_score * 0.4) + (size_score * 0.3) + (round_trip_score * 0.3)
    
    def _generate_chatty_recommendation(self, service1, service2, round_trips):
        """
        Generate recommendations for chatty communication
        """
        if round_trips > 2:
            return f"Implement BFF (Backend for Frontend) or aggregate gateway between {service1} and {service2}"
        else:
            return f"Batch messages or implement message aggregation between {service1} and {service2}"
        
class ChattyCommunicationDetector:
    """
    Detect excessive fine-grained communication between services
    """
    
    def detect_chatty_communication(self, graph, time_window_ms=1000):
        """
        Identify service pairs with excessive message exchange
        """
        chatty_patterns = []
        
        # Analyze service-to-service communication patterns
        service_pairs = self._find_service_pairs(graph)
        
        for (service1, service2), paths in service_pairs.items():
            # Calculate message exchange metrics
            total_msg_rate = 0
            avg_msg_size = []
            round_trips = 0
            
            for path in paths:
                # Analyze each communication path
                if len(path) == 3:  # Service -> Topic -> Service
                    topic = path[1]
                    edge_data = graph.edges[(path[0], topic)]
                    msg_rate = edge_data.get('msg_rate_per_sec', 0)
                    msg_size = edge_data.get('avg_msg_size_kb', 0)
                    
                    total_msg_rate += msg_rate
                    avg_msg_size.append(msg_size)
                    
                    # Check for request-response pattern
                    reverse_path = self._find_reverse_path(graph, service2, service1)
                    if reverse_path:
                        round_trips += 1
            
            # Detect chatty pattern
            if total_msg_rate > 50 and np.mean(avg_msg_size) < 1:  # Many small messages
                chatty_patterns.append({
                    'services': (service1, service2),
                    'severity': self._calculate_chatty_severity(total_msg_rate, avg_msg_size, round_trips),
                    'total_msg_rate': total_msg_rate,
                    'avg_msg_size_kb': np.mean(avg_msg_size),
                    'round_trips': round_trips,
                    'paths': paths,
                    'recommendation': self._generate_chatty_recommendation(service1, service2, round_trips)
                })
        
        return sorted(chatty_patterns, key=lambda x: x['severity'], reverse=True)
    
    def _find_service_pairs(self, graph):
        """
        Find all service pairs that communicate
        """
        service_pairs = {}
        
        for node in graph.nodes():
            if graph.nodes[node].get('type') == 'application':
                # Find all services this service communicates with
                for topic in graph.successors(node):
                    if graph.nodes[topic].get('type') == 'topic':
                        for target_service in graph.successors(topic):
                            if graph.nodes[target_service].get('type') == 'application':
                                pair = (node, target_service)
                                if pair not in service_pairs:
                                    service_pairs[pair] = []
                                service_pairs[pair].append([node, topic, target_service])
        
        return service_pairs
    
    def _find_reverse_path(self, graph, service_from, service_to):
        """
        Check if there is a reverse communication path from service_from to service_to
        """
        for topic in graph.successors(service_from):
            if graph.nodes[topic].get('type') == 'topic':
                for target_service in graph.successors(topic):
                    if target_service == service_to:
                        return [service_from, topic, service_to]
        return None
    
    def _calculate_chatty_severity(self, msg_rate, msg_sizes, round_trips):
        """
        Calculate severity of chatty communication
        """
        rate_score = min(1.0, msg_rate / 100)
        size_score = 1.0 - min(1.0, np.mean(msg_sizes))  # Smaller messages = higher score
        round_trip_score = min(1.0, round_trips / 5)
        
        return (rate_score * 0.4) + (size_score * 0.3) + (round_trip_score * 0.3)
    
    def _generate_chatty_recommendation(self, service1, service2, round_trips):
        """
        Generate recommendations for chatty communication
        """
        if round_trips > 2:
            return f"Implement BFF (Backend for Frontend) or aggregate gateway between {service1} and {service2}"
        else:
            return f"Batch messages or implement message aggregation between {service1} and {service2}"
        
class CircularDependencyDetector:
    """
    Detect circular dependencies in pub-sub message flows
    """
    
    def detect_circular_dependencies(self, graph):
        """
        Find cycles in the service dependency graph
        """
        # Build service dependency graph
        service_graph = self._build_service_dependency_graph(graph)
        
        # Find all simple cycles
        cycles = list(nx.simple_cycles(service_graph))
        
        circular_dependencies = []
        for cycle in cycles:
            if len(cycle) > 1:  # Non-trivial cycle
                cycle_analysis = self._analyze_cycle(graph, service_graph, cycle)
                
                circular_dependencies.append({
                    'cycle': cycle,
                    'length': len(cycle),
                    'severity': cycle_analysis['severity'],
                    'message_flows': cycle_analysis['flows'],
                    'is_synchronous': cycle_analysis['is_synchronous'],
                    'can_deadlock': cycle_analysis['can_deadlock'],
                    'recommendation': self._generate_cycle_recommendation(cycle, cycle_analysis)
                })
        
        return sorted(circular_dependencies, key=lambda x: x['severity'], reverse=True)
    
    def _build_service_dependency_graph(self, graph):
        """
        Build a graph showing only service-to-service dependencies
        """
        service_graph = nx.DiGraph()
        
        for node in graph.nodes():
            if graph.nodes[node].get('type') == 'application':
                service_graph.add_node(node)
        
        # Add edges based on pub-sub relationships
        for service in service_graph.nodes():
            # Find what topics this service publishes to
            for topic in graph.successors(service):
                if graph.nodes[topic].get('type') == 'topic':
                    # Find subscribers to this topic
                    for subscriber in graph.successors(topic):
                        if graph.nodes[subscriber].get('type') == 'application':
                            service_graph.add_edge(service, subscriber, via_topic=topic)
        
        return service_graph
    
    def _analyze_cycle(self, graph, service_graph, cycle):
        """
        Analyze the nature and impact of a cycle
        """
        flows = []
        is_synchronous = False
        can_deadlock = False
        
        for i in range(len(cycle)):
            source = cycle[i]
            target = cycle[(i + 1) % len(cycle)]
            
            if service_graph.has_edge(source, target):
                edge_data = service_graph.edges[(source, target)]
                topic = edge_data.get('via_topic')
                
                # Check if synchronous
                if topic and graph.has_edge(source, topic):
                    topic_edge = graph.edges[(source, topic)]
                    if topic_edge.get('is_synchronous', False):
                        is_synchronous = True
                
                flows.append({
                    'from': source,
                    'to': target,
                    'topic': topic
                })
        
        # Check deadlock potential
        if is_synchronous and len(cycle) > 2:
            can_deadlock = True
        
        # Calculate severity
        severity = 0.5
        if can_deadlock:
            severity = 0.9
        elif is_synchronous:
            severity = 0.7
        severity += len(cycle) * 0.05  # Longer cycles are worse
        
        return {
            'flows': flows,
            'is_synchronous': is_synchronous,
            'can_deadlock': can_deadlock,
            'severity': min(1.0, severity)
        }
    
    def _generate_cycle_recommendation(self, cycle, analysis):
        """
        Generate recommendations for breaking cycles
        """
        if analysis['can_deadlock']:
            return f"CRITICAL: Break cycle immediately - potential deadlock. Consider event-driven async pattern or introduce mediator service"
        elif analysis['is_synchronous']:
            return f"Convert synchronous calls to asynchronous in cycle: {' -> '.join(cycle)}"
        else:
            return f"Review business logic to eliminate cycle or introduce event sourcing pattern"
        
class HiddenCouplingDetector:
    """
    Detect hidden coupling through shared topics and implicit dependencies
    """
    
    def detect_hidden_coupling(self, graph):
        """
        Find services that are implicitly coupled through pub-sub patterns
        """
        hidden_couplings = []
        
        # Find services sharing topics (both publishing or both subscribing)
        topic_publishers = {}
        topic_subscribers = {}
        
        for topic in graph.nodes():
            if graph.nodes[topic].get('type') == 'topic':
                publishers = [n for n in graph.predecessors(topic) 
                            if graph.nodes[n].get('type') == 'application']
                subscribers = [n for n in graph.successors(topic) 
                             if graph.nodes[n].get('type') == 'application']
                
                topic_publishers[topic] = publishers
                topic_subscribers[topic] = subscribers
        
        # Detect multiple publishers to same topic (hidden coupling)
        for topic, publishers in topic_publishers.items():
            if len(publishers) > 1:
                coupling_analysis = self._analyze_publisher_coupling(graph, topic, publishers)
                
                if coupling_analysis['is_problematic']:
                    hidden_couplings.append({
                        'type': 'shared_publisher_topic',
                        'topic': topic,
                        'coupled_services': publishers,
                        'severity': coupling_analysis['severity'],
                        'issues': coupling_analysis['issues'],
                        'recommendation': coupling_analysis['recommendation']
                    })
        
        # Detect temporal coupling
        temporal_couplings = self._detect_temporal_coupling(graph)
        hidden_couplings.extend(temporal_couplings)
        
        # Detect schema coupling
        schema_couplings = self._detect_schema_coupling(graph)
        hidden_couplings.extend(schema_couplings)
        
        return sorted(hidden_couplings, key=lambda x: x['severity'], reverse=True)
    
    def _analyze_publisher_coupling(self, graph, topic, publishers):
        """
        Analyze if multiple publishers create problematic coupling
        """
        issues = []
        severity = 0.3
        
        # Check if publishers are from different domains
        domains = set()
        for pub in publishers:
            domain = graph.nodes[pub].get('business_domain', 'unknown')
            domains.add(domain)
        
        if len(domains) > 1:
            issues.append("Cross-domain publishing to same topic")
            severity += 0.3
        
        # Check for ordering conflicts
        topic_ordering = graph.nodes[topic].get('ordering', 'none')
        if topic_ordering == 'strict' and len(publishers) > 1:
            issues.append("Multiple publishers with strict ordering requirement")
            severity += 0.4
        
        is_problematic = len(issues) > 0
        
        recommendation = ""
        if is_problematic:
            if len(domains) > 1:
                recommendation = f"Split topic '{topic}' by domain or introduce domain-specific topics"
            else:
                recommendation = f"Implement publisher coordinator or switch to single publisher pattern"
        
        return {
            'is_problematic': is_problematic,
            'severity': min(1.0, severity),
            'issues': issues,
            'recommendation': recommendation
        }
    
    def _detect_temporal_coupling(self, graph):
        """
        Detect services that must operate in specific time windows
        """
        temporal_couplings = []
        
        # Find services with time-based dependencies
        for node in graph.nodes():
            if graph.nodes[node].get('type') == 'application':
                # Check for time-sensitive subscriptions
                time_sensitive_deps = []
                
                for topic in graph.predecessors(node):
                    if graph.nodes[topic].get('type') == 'topic':
                        edge_data = graph.edges[(topic, node)]
                        
                        if edge_data.get('time_window_ms'):
                            time_sensitive_deps.append({
                                'topic': topic,
                                'window': edge_data['time_window_ms']
                            })
                
                if len(time_sensitive_deps) > 2:
                    temporal_couplings.append({
                        'type': 'temporal_coupling',
                        'service': node,
                        'severity': min(1.0, len(time_sensitive_deps) * 0.2),
                        'time_dependencies': time_sensitive_deps,
                        'recommendation': 'Implement event store with temporal queries or CQRS pattern'
                    })
        
        return temporal_couplings
    
    def _detect_schema_coupling(self, graph):
        """
        Detect services tightly coupled through message schemas
        """
        schema_couplings = []
        
        # Analyze message schemas for topics
        topic_schemas = {}
        
        for topic in graph.nodes():
            if graph.nodes[topic].get('type') == 'topic':
                schema = graph.nodes[topic].get('message_schema', {})
                topic_schemas[topic] = schema
        
        # Find services sharing topics with incompatible schemas
        for topic, schema in topic_schemas.items():
            subscribers = [n for n in graph.successors(topic) 
                         if graph.nodes[n].get('type') == 'application']
            
            if len(subscribers) > 1:
                incompatible_services = []
                
                for sub in subscribers:
                    sub_schema = graph.nodes[sub].get('expected_schemas', {}).get(topic, {})
                    if not self._are_schemas_compatible(schema, sub_schema):
                        incompatible_services.append(sub)
                
                if incompatible_services:
                    schema_couplings.append({
                        'type': 'schema_coupling',
                        'topic': topic,
                        'affected_services': incompatible_services,
                        'severity': min(1.0, len(incompatible_services) * 0.3),
                        'recommendation': f"Implement schema versioning or use schema registry for topic '{topic}'"
                    })
        
        return schema_couplings
    
    def _are_schemas_compatible(self, schema1, schema2):
        """
        Simple check for schema compatibility (placeholder logic)
        """
        if not schema1 or not schema2:
            return True  # If one is empty, assume compatible
        
        # Check if all fields in schema2 are in schema1
        for field in schema2.get('fields', []):
            if field not in schema1.get('fields', []):
                return False
        return True
    
class TopicSprawlDetector:
    """
    Detect uncontrolled proliferation of topics
    """
    
    def detect_topic_sprawl(self, graph):
        """
        Identify symptoms of topic sprawl
        """
        sprawl_indicators = {
            'orphaned_topics': [],
            'redundant_topics': [],
            'underutilized_topics': [],
            'similar_topics': [],
            'namespace_chaos': []
        }
        
        topics = [n for n in graph.nodes() if graph.nodes[n].get('type') == 'topic']
        
        # Detect orphaned topics (no publishers or subscribers)
        for topic in topics:
            publishers = list(graph.predecessors(topic))
            subscribers = list(graph.successors(topic))
            
            if not publishers or not subscribers:
                sprawl_indicators['orphaned_topics'].append({
                    'topic': topic,
                    'has_publishers': bool(publishers),
                    'has_subscribers': bool(subscribers),
                    'severity': 0.9
                })
        
        # Detect underutilized topics
        for topic in topics:
            total_msg_rate = 0
            for pub in graph.predecessors(topic):
                if graph.has_edge(pub, topic):
                    total_msg_rate += graph.edges[(pub, topic)].get('msg_rate_per_sec', 0)
            
            if total_msg_rate < 0.1:  # Less than 1 message per 10 seconds
                sprawl_indicators['underutilized_topics'].append({
                    'topic': topic,
                    'msg_rate': total_msg_rate,
                    'severity': 0.6
                })
        
        # Detect similar topics (by name similarity and subscriber overlap)
        for i, topic1 in enumerate(topics):
            for topic2 in topics[i+1:]:
                similarity = self._calculate_topic_similarity(graph, topic1, topic2)
                
                if similarity > 0.7:
                    sprawl_indicators['similar_topics'].append({
                        'topics': (topic1, topic2),
                        'similarity': similarity,
                        'severity': similarity
                    })
        
        # Detect namespace chaos
        namespace_analysis = self._analyze_topic_namespaces(topics)
        if namespace_analysis['chaos_level'] > 0.5:
            sprawl_indicators['namespace_chaos'] = namespace_analysis
        
        return self._summarize_sprawl(sprawl_indicators)
    
    def _summarize_sprawl(self, indicators):
        """
        Summarize sprawl indicators into a flat list with severity
        """
        sprawl_issues = []
        
        for key, items in indicators.items():
            if isinstance(items, list):
                for item in items:
                    sprawl_issues.append({
                        'type': key,
                        **item
                    })
            elif isinstance(items, dict) and items:
                sprawl_issues.append({
                    'type': key,
                    **items
                })
        
        return sorted(sprawl_issues, key=lambda x: x.get('severity', 0), reverse=True)
    
    def _calculate_topic_similarity(self, graph, topic1, topic2):
        """
        Calculate similarity between two topics
        """
        # Name similarity (using simple string matching)
        name_similarity = self._string_similarity(topic1, topic2)
        
        # Subscriber overlap
        subs1 = set(graph.successors(topic1))
        subs2 = set(graph.successors(topic2))
        
        if subs1 or subs2:
            subscriber_overlap = len(subs1 & subs2) / len(subs1 | subs2)
        else:
            subscriber_overlap = 0
        
        # Publisher overlap
        pubs1 = set(graph.predecessors(topic1))
        pubs2 = set(graph.predecessors(topic2))
        
        if pubs1 or pubs2:
            publisher_overlap = len(pubs1 & pubs2) / len(pubs1 | pubs2)
        else:
            publisher_overlap = 0
        
        return (name_similarity * 0.3) + (subscriber_overlap * 0.4) + (publisher_overlap * 0.3)
    
    def _string_similarity(self, s1, s2):
        """
        Calculate string similarity
        """
        from difflib import SequenceMatcher
        return SequenceMatcher(None, s1, s2).ratio()
    
    def _analyze_topic_namespaces(self, topics):
        """
        Analyze topic naming conventions and namespace organization
        """
        namespaces = {}
        inconsistencies = []
        
        for topic in topics:
            parts = topic.split('.')
            if len(parts) > 1:
                namespace = parts[0]
                namespaces[namespace] = namespaces.get(namespace, 0) + 1
            else:
                inconsistencies.append(topic)
        
        # Calculate chaos level
        chaos_level = 0
        if inconsistencies:
            chaos_level += len(inconsistencies) / len(topics) * 0.5
        
        if len(namespaces) > len(topics) * 0.3:  # Too many namespaces
            chaos_level += 0.5
        
        return {
            'namespaces': namespaces,
            'inconsistent_topics': inconsistencies,
            'chaos_level': chaos_level,
            'severity': chaos_level
        }
    
class ReliabilityAntiPatternDetector:
    """
    Detect patterns that compromise system reliability
    """
    
    def detect_reliability_antipatterns(self, graph):
        """
        Identify reliability issues in the pub-sub design
        """
        antipatterns = []
        
        # Fire and Forget anti-pattern (no acknowledgment or confirmation)
        fire_forget = self._detect_fire_and_forget(graph)
        antipatterns.extend(fire_forget)
        
        # Poison Message Vulnerability
        poison_vulnerable = self._detect_poison_message_vulnerability(graph)
        antipatterns.extend(poison_vulnerable)
        
        # Missing Dead Letter Queue
        missing_dlq = self._detect_missing_dlq(graph)
        antipatterns.extend(missing_dlq)
        
        # Unbounded Retry
        unbounded_retry = self._detect_unbounded_retry(graph)
        antipatterns.extend(unbounded_retry)
        
        return sorted(antipatterns, key=lambda x: x['severity'], reverse=True)
    
    def _detect_fire_and_forget(self, graph):
        """
        Detect critical flows without confirmation
        """
        fire_forget_patterns = []
        
        for source, target, attrs in graph.edges(data=True):
            if attrs.get('relationship') == 'PUBLISHES_TO':
                topic = target
                
                # Check if critical message flow
                is_critical = (
                    attrs.get('business_flow') in ['payment_processing', 'order_processing'] or
                    attrs.get('transaction_boundary', False) or
                    graph.nodes[source].get('criticality') in ['critical', 'high']
                )
                
                if is_critical:
                    # Check for acknowledgment pattern
                    has_confirmation = attrs.get('confirmation_required', False)
                    has_reply_topic = self._check_reply_topic(graph, source, topic)
                    
                    if not has_confirmation and not has_reply_topic:
                        fire_forget_patterns.append({
                            'type': 'fire_and_forget',
                            'source': source,
                            'topic': topic,
                            'severity': 0.8,
                            'business_flow': attrs.get('business_flow'),
                            'recommendation': 'Implement request-reply pattern or acknowledgment mechanism'
                        })
        
        return fire_forget_patterns
    
    def _check_reply_topic(self, graph, service, topic):
        """
        Check if there is a reply topic associated with the service-topic pair
        """
        for succ in graph.successors(topic):
            if graph.nodes[succ].get('type') == 'application' and succ != service:
                edge_data = graph.edges[(topic, succ)]
                if edge_data.get('is_reply', False):
                    return True
        return False
    
    def _detect_poison_message_vulnerability(self, graph):
        """
        Detect lack of poison message handling
        """
        vulnerable_nodes = []
        
        for node in graph.nodes():
            if graph.nodes[node].get('type') == 'application':
                # Check if node has error handling
                has_error_handling = graph.nodes[node].get('error_handling', False)
                has_dlq = graph.nodes[node].get('dead_letter_queue', False)
                
                # Check criticality
                is_critical = graph.nodes[node].get('criticality') in ['critical', 'high']
                
                if is_critical and not has_error_handling:
                    vulnerable_nodes.append({
                        'type': 'poison_message_vulnerable',
                        'service': node,
                        'severity': 0.7,
                        'has_dlq': has_dlq,
                        'recommendation': 'Implement poison message detection and quarantine'
                    })
        
        return vulnerable_nodes
    
    def _detect_missing_dlq(self, graph):
        """
        Detect critical topics without dead letter queues
        """
        missing_dlq_issues = []
        
        for topic in graph.nodes():
            if graph.nodes[topic].get('type') == 'topic':
                # Check if topic is critical
                is_critical = graph.nodes[topic].get('criticality') in ['critical', 'high']
                
                if is_critical:
                    has_dlq = graph.nodes[topic].get('dead_letter_queue', False)
                    if not has_dlq:
                        missing_dlq_issues.append({
                            'type': 'missing_dead_letter_queue',
                            'topic': topic,
                            'severity': 0.6,
                            'recommendation': 'Add dead letter queue for critical topic'
                        })
        
        return missing_dlq_issues
    
    def _detect_unbounded_retry(self, graph):
        """
        Detect services with unbounded retry mechanisms
        """
        unbounded_retry_issues = []
        
        for node in graph.nodes():
            if graph.nodes[node].get('type') == 'application':
                has_retry = graph.nodes[node].get('retry_policy', None)
                
                if has_retry and has_retry.get('max_retries', None) is None:
                    is_critical = graph.nodes[node].get('criticality') in ['critical', 'high']
                    severity = 0.5
                    if is_critical:
                        severity += 0.3
                    
                    unbounded_retry_issues.append({
                        'type': 'unbounded_retry',
                        'service': node,
                        'severity': min(1.0, severity),
                        'recommendation': 'Implement bounded retry with exponential backoff'
                    })
        
        return unbounded_retry_issues
    
class DesignDefectAnalyzer:
    """
    Main analyzer for detecting all design defects and anti-patterns
    """
    
    def __init__(self, graph):
        self.graph = graph
        self.detectors = {
            'god_topic': GodTopicDetector(),
            'chatty': ChattyCommunicationDetector(),
            'circular': CircularDependencyDetector(),
            'hidden_coupling': HiddenCouplingDetector(),
            'sprawl': TopicSprawlDetector(),
            'reliability': ReliabilityAntiPatternDetector()
        }
    
    def analyze_design_defects(self):
        """
        Run comprehensive design defect analysis
        """
        defects = {}
        
        # Run each detector
        defects['god_topics'] = self.detectors['god_topic'].detect_god_topics(self.graph)
        defects['chatty_communication'] = self.detectors['chatty'].detect_chatty_communication(self.graph)
        defects['circular_dependencies'] = self.detectors['circular'].detect_circular_dependencies(self.graph)
        defects['hidden_coupling'] = self.detectors['hidden_coupling'].detect_hidden_coupling(self.graph)
        defects['topic_sprawl'] = self.detectors['sprawl'].detect_topic_sprawl(self.graph)
        defects['reliability_issues'] = self.detectors['reliability'].detect_reliability_antipatterns(self.graph)
        
        # Calculate overall design quality score
        quality_score = self._calculate_design_quality_score(defects)
        
        return {
            'defects': defects,
            'quality_score': quality_score,
            'summary': self._generate_summary(defects),
            'recommendations': self._prioritize_recommendations(defects)
        }
    
    def _calculate_design_quality_score(self, defects):
        """
        Calculate overall design quality score (0-100)
        """
        base_score = 100
        
        # Deduct points for each type of defect
        penalties = {
            'god_topics': 5,
            'chatty_communication': 3,
            'circular_dependencies': 10,
            'hidden_coupling': 4,
            'topic_sprawl': 2,
            'reliability_issues': 6
        }
        
        for defect_type, defect_list in defects.items():
            if defect_list:
                penalty = penalties.get(defect_type, 2)
                base_score -= len(defect_list) * penalty
        
        return max(0, base_score)
    
    def _generate_summary(self, defects):
        """
        Generate human-readable summary of defects
        """
        summary = {
            'critical_issues': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Categorize by severity
        for defect_type, defect_list in defects.items():
            for defect in defect_list:
                severity = defect.get('severity', 0.5)
                
                if severity > 0.8:
                    summary['critical_issues'].append({
                        'type': defect_type,
                        'description': self._describe_defect(defect),
                        'impact': self._assess_impact(defect)
                    })
                elif severity > 0.5:
                    summary['warnings'].append({
                        'type': defect_type,
                        'description': self._describe_defect(defect)
                    })
                else:
                    summary['suggestions'].append({
                        'type': defect_type,
                        'description': self._describe_defect(defect)
                    })
        
        return summary
    
    def _prioritize_recommendations(self, defects):
        """
        Generate prioritized list of improvements
        """
        all_recommendations = []
        
        for defect_type, defect_list in defects.items():
            for defect in defect_list:
                if 'recommendation' in defect:
                    all_recommendations.append({
                        'priority': self._calculate_priority(defect),
                        'type': defect_type,
                        'action': defect['recommendation'],
                        'effort': self._estimate_effort(defect),
                        'impact': self._estimate_impact(defect)
                    })
        
        # Sort by priority
        return sorted(all_recommendations, key=lambda x: x['priority'], reverse=True)[:10]
    
    def _describe_defect(self, defect):
        """
        Create a brief description of the defect
        """
        if 'topic' in defect:
            return f"Topic '{defect['topic']}' involved in {defect.get('type', 'unknown issue')}"
        elif 'service' in defect:
            return f"Service '{defect['service']}' involved in {defect.get('type', 'unknown issue')}"
        elif 'services' in defect:
            return f"Services '{', '.join(defect['services'])}' involved in {defect.get('type', 'unknown issue')}"
        else:
            return f"Defect of type '{defect.get('type', 'unknown')}' detected"
        
    def _assess_impact(self, defect):
        """
        Assess potential impact of the defect
        """
        severity = defect.get('severity', 0.5)
        if severity > 0.8:
            return "High risk of system failure or major issues"
        elif severity > 0.5:
            return "Moderate risk, may lead to performance or reliability issues"
        else:
            return "Low risk, but worth addressing for better design"
        
    def _calculate_priority(self, defect):
        """
        Calculate priority score for addressing the defect
        """
        severity = defect.get('severity', 0.5)
        effort = self._estimate_effort(defect)
        impact = self._estimate_impact(defect)
        
        return (severity * 0.5) + (impact * 0.3) - (effort * 0.2)
    
    def _estimate_effort(self, defect):
        """
        Estimate effort to fix the defect (0-1 scale)
        """
        if defect.get('type') in ['god_topics', 'circular_dependencies']:
            return 0.8  # High effort
        elif defect.get('type') in ['chatty_communication', 'hidden_coupling']:
            return 0.5  # Medium effort
        else:
            return 0.3  # Low effort
        
    def _estimate_impact(self, defect):
        """
        Estimate impact of fixing the defect (0-1 scale)
        """
        severity = defect.get('severity', 0.5)
        return severity  # Directly proportional to severity
    
def visualize_design_defects(graph, defect_analysis):
    """
    Visualize the graph with design defects highlighted
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Subplot 1: God Topics
    visualize_god_topics(graph, defect_analysis['defects']['god_topics'], ax1)
    
    # Subplot 2: Circular Dependencies
    visualize_circular_dependencies(graph, defect_analysis['defects']['circular_dependencies'], ax2)
    
    # Subplot 3: Hidden Coupling
    visualize_hidden_coupling(graph, defect_analysis['defects']['hidden_coupling'], ax3)
    
    # Subplot 4: Overall Design Quality Heatmap
    visualize_design_quality_heatmap(graph, defect_analysis, ax4)
    
    plt.suptitle(f"Design Defect Analysis - Quality Score: {defect_analysis['quality_score']}/100", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def visualize_god_topics(graph, god_topics, ax):
    """
    Visualize god topic anti-pattern
    """
    pos = nx.spring_layout(graph, k=2, iterations=50)
    
    # Color nodes based on god topic status
    node_colors = []
    node_sizes = []
    
    god_topic_names = {gt['topic'] for gt in god_topics}
    
    for node in graph.nodes():
        if node in god_topic_names:
            node_colors.append('#FF0000')  # Red for god topics
            node_sizes.append(1000)
        elif graph.nodes[node].get('type') == 'topic':
            node_colors.append('#87CEEB')  # Light blue for normal topics
            node_sizes.append(400)
        else:
            node_colors.append('#90EE90')  # Light green for applications
            node_sizes.append(500)
    
    nx.draw(graph, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            with_labels=True,
            font_size=8,
            arrows=True,
            edge_color='gray',
            alpha=0.7)
    
    ax.set_title("God Topic Anti-Pattern Detection", fontsize=14, fontweight='bold')
    
    # Add legend
    legend_elements = [
        plt.scatter([], [], c='#FF0000', s=100, label='God Topics'),
        plt.scatter([], [], c='#87CEEB', s=60, label='Normal Topics'),
        plt.scatter([], [], c='#90EE90', s=70, label='Applications')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

def visualize_circular_dependencies(graph, circular_deps, ax):
    """
    Visualize circular dependencies in the system
    """
    # Create service dependency graph
    service_graph = nx.DiGraph()
    
    for node in graph.nodes():
        if graph.nodes[node].get('type') == 'application':
            service_graph.add_node(node)
    
    # Add edges for service dependencies
    for service in service_graph.nodes():
        for topic in graph.successors(service):
            if graph.nodes[topic].get('type') == 'topic':
                for subscriber in graph.successors(topic):
                    if graph.nodes[subscriber].get('type') == 'application':
                        service_graph.add_edge(service, subscriber)
    
    pos = nx.circular_layout(service_graph)
    
    # Color edges based on whether they're part of a cycle
    edge_colors = []
    edge_widths = []
    
    cycles_edges = set()
    for cycle_info in circular_deps:
        cycle = cycle_info['cycle']
        for i in range(len(cycle)):
            edge = (cycle[i], cycle[(i + 1) % len(cycle)])
            cycles_edges.add(edge)
    
    for edge in service_graph.edges():
        if edge in cycles_edges:
            edge_colors.append('#FF0000')  # Red for cycles
            edge_widths.append(3)
        else:
            edge_colors.append('#808080')  # Gray for normal
            edge_widths.append(1)
    
    nx.draw(service_graph, pos, ax=ax,
            node_color='#FFD700',
            node_size=800,
            with_labels=True,
            font_size=10,
            font_weight='bold',
            arrows=True,
            edge_color=edge_colors,
            width=edge_widths,
            arrowsize=15)
    
    ax.set_title("Circular Dependencies", fontsize=12, fontweight='bold')
    
    # Add text showing cycles
    if circular_deps:
        cycle_text = "Detected Cycles:\n"
        for i, cycle_info in enumerate(circular_deps[:3], 1):
            cycle_text += f"{i}. {' → '.join(cycle_info['cycle'])} → ...\n"
        ax.text(0.02, 0.98, cycle_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
def visualize_hidden_coupling(graph, hidden_couplings, ax):
    """
    Visualize hidden coupling through shared topics
    """
    pos = nx.spring_layout(graph, k=2, iterations=50)
    
    # Color nodes based on hidden coupling status
    node_colors = []
    node_sizes = []
    
    coupled_services = set()
    for hc in hidden_couplings:
        if hc['type'] == 'shared_publisher_topic':
            coupled_services.update(hc['coupled_services'])
    
    for node in graph.nodes():
        if node in coupled_services:
            node_colors.append('#FF4500')  # OrangeRed for coupled services
            node_sizes.append(800)
        elif graph.nodes[node].get('type') == 'topic':
            node_colors.append('#87CEEB')  # Light blue for topics
            node_sizes.append(400)
        else:
            node_colors.append('#90EE90')  # Light green for applications
            node_sizes.append(500)
    
    nx.draw(graph, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            with_labels=True,
            font_size=8,
            arrows=True,
            edge_color='gray',
            alpha=0.7)
    
    ax.set_title("Hidden Coupling Detection", fontsize=14, fontweight='bold')
    
    # Add legend
    legend_elements = [
        plt.scatter([], [], c='#FF4500', s=80, label='Coupled Services'),
        plt.scatter([], [], c='#87CEEB', s=60, label='Topics'),
        plt.scatter([], [], c='#90EE90', s=70, label='Applications')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

def visualize_design_quality_heatmap(graph, defect_analysis, ax):
    """
    Visualize overall design quality as a heatmap
    """
    pos = nx.spring_layout(graph, k=2, iterations=50)
    
    # Color nodes based on overall quality score
    quality_score = defect_analysis['quality_score']
    
    def score_to_color(score):
        if score > 80:
            return '#90EE90'  # Light green
        elif score > 60:
            return '#FFFF00'  # Yellow
        elif score > 40:
            return '#FFA500'  # Orange
        else:
            return '#FF0000'  # Red
    
    node_colors = []
    node_sizes = []
    
    for node in graph.nodes():
        if graph.nodes[node].get('type') == 'topic':
            node_colors.append('#87CEEB')  # Light blue for topics
            node_sizes.append(400)
        else:
            node_colors.append(score_to_color(quality_score))
            node_sizes.append(500)
    
    nx.draw(graph, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            with_labels=True,
            font_size=8,
            arrows=True,
            edge_color='gray',
            alpha=0.7)
    ax.set_title("Overall Design Quality Heatmap", fontsize=14, fontweight='bold')

    # Add legend
    legend_elements = [
        plt.scatter([], [], c='#90EE90', s=70, label='Good Quality (>80)'),
        plt.scatter([], [], c='#FFFF00', s=70, label='Moderate Quality (61-80)'),
        plt.scatter([], [], c='#FFA500', s=70, label='Poor Quality (41-60)'),
        plt.scatter([], [], c='#FF0000', s=70, label='Critical Quality (≤40)'),
        plt.scatter([], [], c='#87CEEB', s=60, label='Topics')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

def demonstrate_design_defect_detection():
    """
    Complete demonstration of design defect detection
    """
    print("="*80)
    print("DESIGN DEFECT AND ANTI-PATTERN ANALYSIS")
    print("="*80)
    
    # Create a sample graph with various anti-patterns
    graph = create_problematic_pubsub_graph()
    
    # Initialize analyzer
    analyzer = DesignDefectAnalyzer(graph)
    
    # Run analysis
    analysis_results = analyzer.analyze_design_defects()
    
    # Display results
    print(f"\nOVERALL DESIGN QUALITY SCORE: {analysis_results['quality_score']}/100")
    print("="*80)
    
    # Display summary
    summary = analysis_results['summary']
    
    if summary['critical_issues']:
        print("\n🔴 CRITICAL ISSUES:")
        for issue in summary['critical_issues']:
            print(f"  - [{issue['type']}] {issue['description']}")
            print(f"    Impact: {issue['impact']}")
    
    if summary['warnings']:
        print("\n🟡 WARNINGS:")
        for warning in summary['warnings']:
            print(f"  - [{warning['type']}] {warning['description']}")
    
    if summary['suggestions']:
        print("\n🟢 SUGGESTIONS:")
        for suggestion in summary['suggestions'][:5]:  # Top 5
            print(f"  - [{suggestion['type']}] {suggestion['description']}")
    
    # Display detailed defects by category
    print("\n" + "="*80)
    print("DETAILED DEFECT ANALYSIS")
    print("="*80)
    
    defects = analysis_results['defects']
    
    # 1. God Topics
    if defects['god_topics']:
        print("\n1. GOD TOPIC ANTI-PATTERN")
        print("-" * 40)
        for god_topic in defects['god_topics']:
            print(f"  Topic: {god_topic['topic']}")
            print(f"  Severity: {god_topic['severity']:.2f}")
            print(f"  Total Connections: {god_topic['total_connections']}")
            print(f"  Publishers: {len(god_topic['publishers'])}")
            print(f"  Subscribers: {len(god_topic['subscribers'])}")
            print(f"  Message Types: {god_topic['message_type_count']}")
            print(f"  Recommendation: {god_topic['recommendation']}")
            print()
    
    # 2. Circular Dependencies
    if defects['circular_dependencies']:
        print("\n2. CIRCULAR DEPENDENCIES")
        print("-" * 40)
        for cycle in defects['circular_dependencies']:
            print(f"  Cycle: {' → '.join(cycle['cycle'])} → {cycle['cycle'][0]}")
            print(f"  Severity: {cycle['severity']:.2f}")
            print(f"  Length: {cycle['length']} services")
            print(f"  Synchronous: {cycle['is_synchronous']}")
            print(f"  Can Deadlock: {cycle['can_deadlock']}")
            print(f"  Recommendation: {cycle['recommendation']}")
            print()
    
    # 3. Chatty Communication
    if defects['chatty_communication']:
        print("\n3. CHATTY COMMUNICATION ANTI-PATTERN")
        print("-" * 40)
        for chatty in defects['chatty_communication']:
            print(f"  Services: {chatty['services'][0]} ↔ {chatty['services'][1]}")
            print(f"  Severity: {chatty['severity']:.2f}")
            print(f"  Message Rate: {chatty['total_msg_rate']} msg/sec")
            print(f"  Avg Message Size: {chatty['avg_msg_size_kb']:.2f} KB")
            print(f"  Round Trips: {chatty['round_trips']}")
            print(f"  Recommendation: {chatty['recommendation']}")
            print()
    
    # 4. Hidden Coupling
    if defects['hidden_coupling']:
        print("\n4. HIDDEN COUPLING")
        print("-" * 40)
        for coupling in defects['hidden_coupling'][:5]:  # Top 5
            print(f"  Type: {coupling['type']}")
            print(f"  Severity: {coupling['severity']:.2f}")
            if coupling['type'] == 'shared_publisher_topic':
                print(f"  Topic: {coupling['topic']}")
                print(f"  Coupled Services: {', '.join(coupling['coupled_services'])}")
                print(f"  Issues: {', '.join(coupling['issues'])}")
            print(f"  Recommendation: {coupling['recommendation']}")
            print()
    
    # Display prioritized recommendations
    print("\n" + "="*80)
    print("PRIORITIZED IMPROVEMENT RECOMMENDATIONS")
    print("="*80)
    
    for i, rec in enumerate(analysis_results['recommendations'], 1):
        print(f"\n{i}. Priority Score: {rec['priority']:.2f}")
        print(f"   Type: {rec['type']}")
        print(f"   Action: {rec['action']}")
        print(f"   Effort: {rec['effort']}")
        print(f"   Expected Impact: {rec['impact']}")
    
    return analysis_results

def create_problematic_pubsub_graph():
    """
    Create a graph with various design defects for demonstration
    """
    G = nx.DiGraph()
    
    # Add services
    services = [
        # Core services
        {'id': 'user-service', 'type': 'application', 'criticality': 'high', 'business_domain': 'user'},
        {'id': 'order-service', 'type': 'application', 'criticality': 'critical', 'business_domain': 'order'},
        {'id': 'payment-service', 'type': 'application', 'criticality': 'critical', 'business_domain': 'payment'},
        {'id': 'inventory-service', 'type': 'application', 'criticality': 'high', 'business_domain': 'inventory'},
        {'id': 'notification-service', 'type': 'application', 'criticality': 'low', 'business_domain': 'notification'},
        {'id': 'analytics-service', 'type': 'application', 'criticality': 'low', 'business_domain': 'analytics'},
        {'id': 'shipping-service', 'type': 'application', 'criticality': 'medium', 'business_domain': 'shipping'},
        {'id': 'recommendation-service', 'type': 'application', 'criticality': 'low', 'business_domain': 'recommendation'},
        {'id': 'audit-service', 'type': 'application', 'criticality': 'high', 'business_domain': 'audit'},
        {'id': 'report-service', 'type': 'application', 'criticality': 'medium', 'business_domain': 'report'}
    ]
    
    for service in services:
        G.add_node(service['id'], **service)
    
    # Add topics (including problematic ones)
    topics = [
        # God topic (everything goes here)
        {'id': 'events.all', 'type': 'topic', 'ordering': 'none'},
        
        # Normal topics
        {'id': 'orders.created', 'type': 'topic', 'ordering': 'strict'},
        {'id': 'orders.updated', 'type': 'topic', 'ordering': 'strict'},
        {'id': 'payments.processed', 'type': 'topic', 'ordering': 'strict'},
        {'id': 'inventory.updated', 'type': 'topic', 'ordering': 'partial'},
        
        # Similar/redundant topics (topic sprawl)
        {'id': 'user.updated', 'type': 'topic', 'ordering': 'none'},
        {'id': 'users.modified', 'type': 'topic', 'ordering': 'none'},
        {'id': 'user-changes', 'type': 'topic', 'ordering': 'none'},
        
        # Orphaned topics
        {'id': 'deprecated.events', 'type': 'topic', 'ordering': 'none'},
        {'id': 'test.topic', 'type': 'topic', 'ordering': 'none'},
        
        # Topics for circular dependency
        {'id': 'notifications.requested', 'type': 'topic', 'ordering': 'none'},
        {'id': 'analytics.processed', 'type': 'topic', 'ordering': 'none'}
    ]
    
    for topic in topics:
        G.add_node(topic['id'], **topic)
    
    # Add problematic edges
    
    # 1. God Topic Pattern - Everyone publishes to events.all
    god_topic_edges = [
        ('user-service', 'events.all', {'msg_rate_per_sec': 50}),
        ('order-service', 'events.all', {'msg_rate_per_sec': 100}),
        ('payment-service', 'events.all', {'msg_rate_per_sec': 80}),
        ('inventory-service', 'events.all', {'msg_rate_per_sec': 200}),
        ('shipping-service', 'events.all', {'msg_rate_per_sec': 60}),
        ('events.all', 'analytics-service', {'msg_rate_per_sec': 490}),
        ('events.all', 'audit-service', {'msg_rate_per_sec': 490}),
        ('events.all', 'report-service', {'msg_rate_per_sec': 490}),
    ]
    
    # 2. Circular Dependencies
    circular_edges = [
        ('order-service', 'orders.created', {'msg_rate_per_sec': 100, 'is_synchronous': True}),
        ('orders.created', 'payment-service', {'msg_rate_per_sec': 100}),
        ('payment-service', 'payments.processed', {'msg_rate_per_sec': 95, 'is_synchronous': True}),
        ('payments.processed', 'notification-service', {'msg_rate_per_sec': 95}),
        ('notification-service', 'notifications.requested', {'msg_rate_per_sec': 150}),
        ('notifications.requested', 'analytics-service', {'msg_rate_per_sec': 150}),
        ('analytics-service', 'analytics.processed', {'msg_rate_per_sec': 200}),
        ('analytics.processed', 'order-service', {'msg_rate_per_sec': 200}),  # Closes the circle
    ]
    
    # 3. Chatty Communication Pattern
    chatty_edges = [
        # Many small messages between user and recommendation service
        ('user-service', 'user.updated', {'msg_rate_per_sec': 100, 'avg_msg_size_kb': 0.1}),
        ('user.updated', 'recommendation-service', {'msg_rate_per_sec': 100}),
        ('recommendation-service', 'users.modified', {'msg_rate_per_sec': 80, 'avg_msg_size_kb': 0.2}),
        ('users.modified', 'user-service', {'msg_rate_per_sec': 80}),
    ]
    
    # 4. Hidden Coupling - Multiple publishers to same topic
    coupling_edges = [
        ('order-service', 'inventory.updated', {'msg_rate_per_sec': 50}),  # Order service updating inventory!
        ('inventory-service', 'inventory.updated', {'msg_rate_per_sec': 100}),
        ('shipping-service', 'inventory.updated', {'msg_rate_per_sec': 30}),  # Shipping also updates inventory
    ]
    
    # 5. Topic Sprawl - Similar topics
    sprawl_edges = [
        ('user-service', 'user-changes', {'msg_rate_per_sec': 0.01}),  # Very low usage
        ('user-service', 'users.modified', {'msg_rate_per_sec': 0.02}),  # Very low usage
    ]
    
    # Add all edges
    for edges in [god_topic_edges, circular_edges, chatty_edges, coupling_edges, sprawl_edges]:
        for source, target, attrs in edges:
            G.add_edge(source, target, relationship='PUBLISHES_TO' if G.nodes[target].get('type') == 'topic' else 'SUBSCRIBES_TO', **attrs)
    
    return G

class DesignQualityMetrics:
    """
    Calculate comprehensive design quality metrics
    """
    
    def __init__(self, graph):
        self.graph = graph
    
    def calculate_metrics(self):
        """
        Calculate all design quality metrics
        """
        metrics = {
            'coupling_metrics': self._calculate_coupling_metrics(),
            'cohesion_metrics': self._calculate_cohesion_metrics(),
            'complexity_metrics': self._calculate_complexity_metrics(),
            'modularity_metrics': self._calculate_modularity_metrics(),
            'evolvability_metrics': self._calculate_evolvability_metrics()
        }
        
        # Calculate aggregate scores
        metrics['overall_score'] = self._calculate_overall_score(metrics)
        metrics['maturity_level'] = self._determine_maturity_level(metrics['overall_score'])
        
        return metrics
    
    def _calculate_coupling_metrics(self):
        """
        Calculate coupling-related metrics
        """
        # Afferent coupling (incoming dependencies)
        # Efferent coupling (outgoing dependencies)
        coupling = {}
        
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('type') == 'application':
                afferent = len(list(self.graph.predecessors(node)))
                efferent = len(list(self.graph.successors(node)))
                
                coupling[node] = {
                    'afferent': afferent,
                    'efferent': efferent,
                    'instability': efferent / (afferent + efferent) if (afferent + efferent) > 0 else 0
                }
        
        # Calculate average coupling
        avg_coupling = np.mean([c['afferent'] + c['efferent'] for c in coupling.values()])
        
        return {
            'service_coupling': coupling,
            'average_coupling': avg_coupling,
            'coupling_score': max(0, 100 - (avg_coupling * 5))  # Lower coupling is better
        }
    
    def _calculate_cohesion_metrics(self):
        """
        Calculate cohesion within service boundaries
        """
        # Use community detection to measure cohesion
        communities = nx.community.louvain_communities(self.graph.to_undirected())
        
        modularity = nx.community.modularity(self.graph.to_undirected(), communities)
        
        return {
            'modularity': modularity,
            'community_count': len(communities),
            'cohesion_score': modularity * 100  # Higher modularity indicates better cohesion
        }
    
    def _calculate_complexity_metrics(self):
        """
        Calculate system complexity metrics
        """
        # Cyclomatic complexity (based on graph structure)
        edges = self.graph.number_of_edges()
        nodes = self.graph.number_of_nodes()
        components = nx.number_weakly_connected_components(self.graph)
        
        cyclomatic_complexity = edges - nodes + (2 * components)
        
        # Depth of dependency tree
        max_depth = 0
        for node in self.graph.nodes():
            if self.graph.nodes[node].get('type') == 'application':
                depth = self._calculate_dependency_depth(node)
                max_depth = max(max_depth, depth)
        
        return {
            'cyclomatic_complexity': cyclomatic_complexity,
            'max_dependency_depth': max_depth,
            'complexity_score': max(0, 100 - (cyclomatic_complexity * 2) - (max_depth * 5))
        }
    
    def _calculate_dependency_depth(self, node, visited=None):
        """
        Calculate maximum dependency depth from a node
        """
        if visited is None:
            visited = set()
        
        if node in visited:
            return 0
        
        visited.add(node)
        
        max_depth = 0
        for successor in self.graph.successors(node):
            if self.graph.nodes[successor].get('type') == 'topic':
                for sub in self.graph.successors(successor):
                    if self.graph.nodes[sub].get('type') == 'application':
                        depth = 1 + self._calculate_dependency_depth(sub, visited.copy())
                        max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _calculate_modularity_metrics(self):
        """
        Calculate modularity metrics
        """
        # Reuse cohesion modularity as a proxy
        cohesion = self._calculate_cohesion_metrics()
        
        return {
            'modularity': cohesion['modularity'],
            'modularity_score': cohesion['cohesion_score']
        }
    

    def _calculate_evolvability_metrics(self):
        """
        Calculate evolvability metrics
        """
        # Measure number of services and topics
        service_count = len([n for n in self.graph.nodes() if self.graph.nodes[n].get('type') == 'application'])
        topic_count = len([n for n in self.graph.nodes() if self.graph.nodes[n].get('type') == 'topic'])
        
        # Measure average change impact (simulated)
        avg_change_impact = np.mean([len(list(self.graph.successors(n))) for n in self.graph.nodes() if self.graph.nodes[n].get('type') == 'application'] + [0])
        
        return {
            'service_count': service_count,
            'topic_count': topic_count,
            'avg_change_impact': avg_change_impact,
            'evolvability_score': max(0, 100 - (avg_change_impact * 10))  # Lower impact is better
        }
    
    def _calculate_overall_score(self, metrics):
        """
        Calculate weighted overall score
        """
        weights = {
            'coupling': 0.25,
            'cohesion': 0.25,
            'complexity': 0.25,
            'modularity': 0.15,
            'evolvability': 0.10
        }
        
        score = (
            metrics['coupling_metrics']['coupling_score'] * weights['coupling'] +
            metrics['cohesion_metrics']['cohesion_score'] * weights['cohesion'] +
            metrics['complexity_metrics']['complexity_score'] * weights['complexity'] +
            metrics['modularity_metrics']['modularity_score'] * weights['modularity'] +
            metrics['evolvability_metrics']['evolvability_score'] * weights['evolvability']
        )
        
        return round(score, 2)
    
    def _determine_maturity_level(self, score):
        """
        Determine architecture maturity level
        """
        if score >= 90:
            return "Level 5: Optimized"
        elif score >= 75:
            return "Level 4: Managed"
        elif score >= 60:
            return "Level 3: Defined"
        elif score >= 40:
            return "Level 2: Repeatable"
        else:
            return "Level 1: Initial/Chaotic"
        
if __name__ == "__main__":
    # Run complete design defect analysis
    analysis_results = demonstrate_design_defect_detection()
    
    # Create visualization
    graph = create_problematic_pubsub_graph()
    visualize_design_defects(graph, analysis_results)
    
    # Calculate quality metrics
    metrics_calculator = DesignQualityMetrics(graph)
    quality_metrics = metrics_calculator.calculate_metrics()
    
    print("\n" + "="*80)
    print("DESIGN QUALITY METRICS")
    print("="*80)
    print(f"Overall Quality Score: {quality_metrics['overall_score']}/100")
    print(f"Maturity Level: {quality_metrics['maturity_level']}")
    print(f"\nDetailed Metrics:")
    print(f"  Coupling Score: {quality_metrics['coupling_metrics']['coupling_score']:.1f}")
    print(f"  Cohesion Score: {quality_metrics['cohesion_metrics']['cohesion_score']:.1f}")
    print(f"  Complexity Score: {quality_metrics['complexity_metrics']['complexity_score']:.1f}")