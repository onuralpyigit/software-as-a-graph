from neo4j import GraphDatabase
import networkx as nx
 
class GraphExporter:    
    # Connection details for Neo4j
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "password"
    driver = GraphDatabase.driver(uri, auth=(user, password))

    def export_graph(self):
        with self.driver.session() as session:
            # Get the graph nodes
            results = session.run("""
            MATCH (n)
            RETURN n.name AS nodeName, labels(n) AS nodeType, n.reliability AS reliability, n.durability AS durability
            """)
            g = nx.DiGraph()
            for record in results:
                g.add_node(record["nodeName"], 
                        type=record["nodeType"][0], 
                        reliability=record["reliability"], 
                        durability=record["durability"])

            # Get the graph edges
            results = session.run("""
            MATCH (n1)-[r]->(n2)
            RETURN n1.name AS fromNode, n2.name AS toNode, type(r) AS edgeType
            """)
            for record in results:
                g.add_edge(record["fromNode"], record["toNode"], type=record["edgeType"])
            return g

    def export_graph_without_derived_relationships(self):
        with self.driver.session() as session:
            # Get the graph nodes
            results = session.run("""
            MATCH (n)
            RETURN n.name AS nodeName, labels(n) AS nodeType
            """)
            g = nx.DiGraph()
            for record in results:
                g.add_node(record["nodeName"], type=record["nodeType"][0])

            # Get the graph edges without derived relationships
            results = session.run("""
            MATCH (n1)-[r]->(n2)
            WHERE NOT type(r) IN ['DEPENDS_ON', 'CONNECTS_TO']
            RETURN n1.name AS fromNode, n2.name AS toNode, type(r) AS edgeType
            """)
            for record in results:
                g.add_edge(record["fromNode"], record["toNode"], type=record["edgeType"])

            return g

    def export_graph_for_application_level_analysis(self):
        with self.driver.session() as session:
            results = session.run("""
            MATCH (a1:Application)-[:DEPENDS_ON]->(a2:Application)
            RETURN a1.name AS fromApp, a2.name AS toApp
            """)
            g = nx.DiGraph()
            for record in results:
                g.add_node(record["fromApp"], type='Application')
                g.add_node(record["toApp"], type='Application')
                g.add_edge(record["fromApp"], record["toApp"], type="DEPENDS_ON")
  
            return g

    def export_graph_for_infrastructure_level_analysis(self):
        with self.driver.session() as session:
            results = session.run("""
            MATCH (n1:Node)-[:CONNECTS_TO]->(n2:Node)
            RETURN n1.name AS fromNode, n2.name AS toNode
            """)
            g = nx.DiGraph()
            for record in results:
                g.add_node(record["fromNode"], type='Node')
                g.add_node(record["toNode"], type='Node')
                g.add_edge(record["fromNode"], record["toNode"], type="CONNECTS_TO")
            
            return g
        
    def export_graph_for_topics(self):
        with self.driver.session() as session:
            results = session.run("""
            MATCH (t:Topic)
            RETURN t.name AS topicName, t.durability AS durability, t.reliability AS reliability
            """)
            g = nx.DiGraph()
            for record in results:
                g.add_node(record["topicName"], type='Topic', 
                           durability=record["durability"], 
                           reliability=record["reliability"])
            
            return g
        
    def export_graph_by_topic(self, topic_name):
        with self.driver.session() as session:
            results = session.run("""
            MATCH (a:Application)-[r:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic {name: $topicName})
            RETURN a.name AS appName, type(r) AS relationshipType, t.name AS topicName, t.durability AS topicDurability, t.reliability AS topicReliability
            """, topicName=topic_name)
            g = nx.DiGraph()
            for record in results:
                g.add_node(record["appName"], type='Application')
                g.add_node(record["topicName"], type='Topic', durability=record["topicDurability"], reliability=record["topicReliability"])
                g.add_edge(record["appName"], record["topicName"], type=record["relationshipType"])
            
            return g

    def export_graph_for_applications(self):
        with self.driver.session() as session:
            results = session.run("""
            MATCH (a:Application)
            RETURN a.name AS appName
            """)
            g = nx.DiGraph()
            for record in results:
                g.add_node(record["appName"], type='Application')
            
            return g

    def export_graph_by_application(self, app_name):
        with self.driver.session() as session:
            results = session.run("""
            MATCH (a:Application {name: $appName})-[r:PUBLISHES_TO|SUBSCRIBES_TO]->(t:Topic)
            RETURN a.name AS appName, type(r) AS relationshipType, t.name AS topicName, t.durability AS topicDurability, t.reliability AS topicReliability
            """, appName=app_name)
            g = nx.DiGraph()
            for record in results:
                g.add_node(record["appName"], type='Application')
                g.add_node(record["topicName"], type='Topic', durability=record["topicDurability"], reliability=record["topicReliability"])
                g.add_edge(record["appName"], record["topicName"], type=record["relationshipType"])
            
            return g
        
    def export_graph_for_nodes(self):
        with self.driver.session() as session:
            results = session.run("""
            MATCH (n:Node)
            RETURN n.name AS nodeName
            """)
            g = nx.DiGraph()
            for record in results:
                g.add_node(record["nodeName"], type='Node')
            
            return g
        
    def export_graph_by_node(self, node_name):
        with self.driver.session() as session:
            results = session.run("""
            MATCH (n:Node {name: $nodeName})<-[r:RUNS_ON]-(a:Application)
            RETURN n.name AS nodeName, type(r) AS relationshipType, a.name AS appName
            """, nodeName=node_name)
            g = nx.DiGraph()
            for record in results:
                g.add_node(record["nodeName"], type='Node')
                g.add_node(record["appName"], type='Application')
                g.add_edge(record["nodeName"], record["appName"], type=record["relationshipType"])
            
            return g
        
    def export_graph_for_brokers(self):
        with self.driver.session() as session:
            results = session.run("""
            MATCH (b:Broker)
            RETURN b.name AS brokerName
            """)
            g = nx.DiGraph()
            for record in results:
                g.add_node(record["brokerName"], type='Broker')
            
            return g
        
    def export_graph_by_broker(self, broker_name):
        with self.driver.session() as session:
            results = session.run("""
            MATCH (b:Broker {name: $brokerName})-[r:ROUTES]->(t:Topic)
            RETURN b.name AS brokerName, type(r) AS relationshipType, t.name AS topicName, t.durability AS topicDurability, t.reliability AS topicReliability
            """, brokerName=broker_name)
            g = nx.DiGraph()
            for record in results:
                g.add_node(record["brokerName"], type='Broker')
                g.add_node(record["topicName"], type='Topic', durability=record["topicDurability"], reliability=record["topicReliability"])
                g.add_edge(record["brokerName"], record["topicName"], type=record["relationshipType"])
            
            return g
        
    def print_graph(self, g):
        print(f"Number of nodes: {g.number_of_nodes()}")
        print(f"Number of edges: {g.number_of_edges()}")
        
        # Count node types
        node_types = {}
        for node, attrs in g.nodes(data=True):
            node_type = attrs.get('type')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        print("\nNode Distribution:")
        for node_type, count in node_types.items():
            print(f"  {node_type}: {count}")
        
        # Count edge types
        edge_types = {}
        for u, v, attrs in g.edges(data=True):
            edge_type = attrs.get('type')
            edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
        
        print("\nEdge Distribution:")
        for edge_type, count in edge_types.items():
            print(f"  {edge_type}: {count}") 

    def summarize_graph(self, g):
        summary = {
            'num_nodes': g.number_of_nodes(),
            'num_edges': g.number_of_edges(),
            'node_types': {},
            'edge_types': {}
        }
        
        # Count node types
        for node, attrs in g.nodes(data=True):
            node_type = attrs.get('type')
            summary['node_types'][node_type] = summary['node_types'].get(node_type, 0) + 1
        
        # Count edge types
        for u, v, attrs in g.edges(data=True):
            edge_type = attrs.get('type')
            summary['edge_types'][edge_type] = summary['edge_types'].get(edge_type, 0) + 1
        
        return summary

    # Function to export the graph to a JSON format compatible with D3.js visualization
    def export_graph_to_d3js(self, g):
        # Create nodes with all attributes
        nodes = []
        for node_id in g.nodes():
            node_data = g.nodes[node_id].copy()
            node_data['id'] = node_id
            nodes.append(node_data)
        
        # Create links with all attributes
        links = []
        for source, target, data in g.edges(data=True):
            link_data = data.copy()
            link_data['source'] = source
            link_data['target'] = target
            links.append(link_data)
        
        # Create the graph structure
        graph = {
            'nodes': nodes,
            'links': links
        }
        
        return graph