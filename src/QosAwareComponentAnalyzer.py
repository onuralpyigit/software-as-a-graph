import networkx as nx

from src.GraphExporter import GraphExporter

class QosAwareComponentAnalyzer:
    def __init__(self, exporter: GraphExporter):
        self.exporter = exporter

    def calculate_topic_qos_score(self, topic, durability_weight = 0.6, reliability_weight = 0.4 ):
        # Define ordering of reliability and durability policies
        reliability_order = {
            'BEST_EFFORT': 0,
            'RELIABLE': 1
        }

        durability_order = {
            'VOLATILE': 0.0,
            'TRANSIENT_LOCAL': 0.50,
            'TRANSIENT': 0.75,
            'PERSISTENT': 1.0
        }

        # Get topic attributes
        durability = topic.get('durability')
        reliability = topic.get('reliability')

        # Validate attributes
        if durability not in durability_order:
            durability = 'VOLATILE'  # Default to VOLATILE if not specified
        
        if reliability not in reliability_order:
            reliability = 'BEST_EFFORT' # Default to BEST_EFFORT if not specified
        
        # Calculate criticality values  
        durability_criticality = durability_order.get(durability, 0)
        reliability_criticality = reliability_order.get(reliability, 0) 

        return (durability_weight * durability_criticality) + (reliability_weight * reliability_criticality)  
    
    def calculate_application_qos_score(self, published_topics, subscribed_topics, pub_weight=0.7, sub_weight=0.3):
        # Calculate QoS scores for published topics
        total_published_score = sum(self.calculate_topic_qos_score(topic) for topic in published_topics)

        # Calculate QoS scores for subscribed topics
        total_subscribed_score = sum(self.calculate_topic_qos_score(topic) for topic in subscribed_topics)

        # Normalize the composite score
        pub_avg = total_published_score / len(published_topics) if published_topics else 0
        sub_avg = total_subscribed_score / len(subscribed_topics) if subscribed_topics else 0
        composite_score = pub_weight * pub_avg + sub_weight * sub_avg

        return composite_score
    
    def calculate_node_qos_score(self, running_apps_qos_scores):
        # Calculate the average QoS score for all running applications on the node
        if not running_apps_qos_scores:
            return 0.0
        
        total_score = sum(running_apps_qos_scores)
        average_score = total_score / len(running_apps_qos_scores)
        
        return average_score
    
    def calculate_broker_qos_score(self, routed_topics):
        # Calculate the average QoS score for all routed topics
        if not routed_topics:
            return 0.0
        
        total_score = sum(self.calculate_topic_qos_score(topic) for topic in routed_topics)
        average_score = total_score / len(routed_topics)
        
        return average_score
    
    def analye_topic(self, topic_name):
        # Export the graph for the specific topic
        topic_graph = self.exporter.export_graph_by_topic(topic_name)
        if not topic_graph:
            print(f"Topic {topic_name} not found in the graph.")
            return 0.0
        
        # Get topic attributes
        topic = topic_graph.nodes[topic_name]

        # Check if topic has required attributes
        if 'durability' not in topic or 'reliability' not in topic:
            print(f"Topic {topic_name} does not have required attributes (durability, reliability).")
            return 0.0
        
        # Calculate QoS score for the topic
        qos_score = self.calculate_topic_qos_score(topic)
        print(f"Topic {topic_name} - Durability: {topic['durability']}, Reliability: {topic['reliability']}, QoS Score: {qos_score:.3f}")
        return qos_score
    
    def analyze_topics(self):
        # Export the graph for all topics
        topics_graph = self.exporter.export_graph_for_topics()
        if not topics_graph:
            print("No topics found in the graph.")
            return {}

        # Calculate QoS scores for all topics
        topic_scores = {}
        for topic in topics_graph.nodes(data=True):
            topic_name = topic[0]
            qos_score = self.analye_topic(topic_name)
            topic_scores[topic_name] = qos_score

        return topic_scores
    
    def analyze_application(self, app_name):
        # Export the graph for the specific application
        app_graph = self.exporter.export_graph_by_application(app_name)
        if not app_graph:
            print(f"Application {app_name} not found in the graph.")
            return 0.0
        
        # Get published and subscribed topics
        published_topics = [app_graph.nodes[edge[1]] for edge in app_graph.out_edges(app_name, data=True) 
                            if edge[2].get('type') == 'PUBLISHES_TO']
        subscribed_topics = [app_graph.nodes[edge[1]] for edge in app_graph.out_edges(app_name, data=True) 
                             if edge[2].get('type') == 'SUBSCRIBES_TO'] 
        
        # Calculate QoS score for the application
        qos_score = self.calculate_application_qos_score(published_topics, subscribed_topics)
        return qos_score
    
    def analyze_applications(self):
        # Export the graph for all applications
        apps_graph = self.exporter.export_graph_for_applications()
        if not apps_graph:
            print("No applications found in the graph.")
            return {}

        # Calculate QoS scores for all applications
        app_scores = {}
        for app in apps_graph.nodes(data=True):
            app_name = app[0]
            qos_score = self.analyze_application(app_name)
            app_scores[app_name] = qos_score

        return app_scores
    
    def analyze_node(self, node_name):
        # Export the graph for the specific node
        node_graph = self.exporter.export_graph_by_node(node_name)
        if not node_graph:
            print(f"Node {node_name} not found in the graph.")
            return 0.0
        
        # Get all applications running on this node
        running_apps = [n for n, attr in node_graph.nodes(data=True) 
                        if attr.get('type') == 'Application']
        
        # Calculate QoS scores for all running applications
        running_apps_qos_scores = []
        for app in running_apps:
            app_score = self.analyze_application(app)
            running_apps_qos_scores.append(app_score)

        # Calculate QoS score for the node
        node_qos_score = self.calculate_node_qos_score(running_apps_qos_scores)
        return node_qos_score
    
    def analyze_nodes(self):
        # Export the graph for all nodes
        nodes_graph = self.exporter.export_graph_for_nodes()
        if not nodes_graph:
            print("No nodes found in the graph.")
            return {}

        # Calculate QoS scores for all nodes
        node_scores = {}
        for node in nodes_graph.nodes(data=True):
            node_name = node[0]
            qos_score = self.analyze_node(node_name)
            node_scores[node_name] = qos_score

        return node_scores
    
    def analyze_broker(self, broker_name):
        # Export the graph for the specific broker
        broker_graph = self.exporter.export_graph_by_broker(broker_name)
        if not broker_graph:
            print(f"Broker {broker_name} not found in the graph.")
            return 0.0

        # Get all routed topics for this broker
        routed_topics = [broker_graph.nodes[edge[1]] for edge in broker_graph.out_edges(broker_name, data=True) 
                         if edge[2].get('type') == 'ROUTES']
        
        # Calculate QoS score for the broker
        broker_qos_score = self.calculate_broker_qos_score(routed_topics)
        print(f"Broker {broker_name} - QoS Score: {broker_qos_score:.3f}")
        return broker_qos_score
    
    def analyze_brokers(self):
        # Export the graph for all brokers
        brokers_graph = self.exporter.export_graph_for_brokers()
        if not brokers_graph:
            print("No brokers found in the graph.")
            return {}

        # Calculate QoS scores for all brokers
        broker_scores = {}
        for broker in brokers_graph.nodes(data=True):
            broker_name = broker[0]
            qos_score = self.analyze_broker(broker_name)
            broker_scores[broker_name] = qos_score

        return broker_scores

    def analyze_critical_components_by_qos_score(self):
        # Analyze topics
        topic_scores = self.analyze_topics()
        
        # Analyze applications
        app_scores = self.analyze_applications()
        
        # Analyze nodes
        node_scores = self.analyze_nodes()
        
        # Analyze brokers
        broker_scores = self.analyze_brokers()

        # Combine all scores into a single dictionary
        critical_components_scores = {
            'topics': topic_scores,
            'applications': app_scores,
            'nodes': node_scores,
            'brokers': broker_scores
        }

        return critical_components_scores