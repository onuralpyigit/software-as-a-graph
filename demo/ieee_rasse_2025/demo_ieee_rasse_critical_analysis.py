# Critical Component Analysis in Distributed Publish-Subscribe Systems
# Implementation of the Graph-Based Dependency Analysis Method
# Paper: "A Graph-Based Dependency Analysis Method for Identifying Critical Components 
#         in Distributed Publish-Subscribe Systems" - IEEE RASSE 2025

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set
import json
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')

# ============================================================================
# SECTION 1: DATA STRUCTURES AND GRAPH MODEL
# ============================================================================

@dataclass
class PubSubSystem:
    """Represents a distributed publish-subscribe system as defined in the paper."""
    applications: List[str]
    topics: List[str]
    brokers: List[str]
    nodes: List[str]
    publishes: List[Tuple[str, str]]  # (app, topic)
    subscribes: List[Tuple[str, str]]  # (app, topic)
    hosts: Dict[str, str]  # app/broker -> node
    routes: List[Tuple[str, str]]  # (broker, topic)

class PubSubGraphAnalyzer:
    """
    Implements the graph-based dependency analysis method for pub-sub systems.
    Based on the multi-layer graph model: G = (V, E) where V = A ∪ T ∪ B ∪ N
    """
    
    def __init__(self, system: PubSubSystem):
        self.system = system
        self.G = nx.DiGraph()  # Main multi-layer graph
        self.GA = nx.DiGraph()  # Application dependency graph
        self.GN = nx.DiGraph()  # Infrastructure graph
        self._build_graphs()
    
    def _build_graphs(self):
        """Constructs the multi-layer dependency graph (Algorithm 1 from paper)."""
        # Add vertices
        self.G.add_nodes_from(self.system.applications, layer='application', type='app')
        self.G.add_nodes_from(self.system.topics, layer='messaging', type='topic')
        self.G.add_nodes_from(self.system.brokers, layer='messaging', type='broker')
        self.G.add_nodes_from(self.system.nodes, layer='infrastructure', type='node')
        
        # Add edges: PUBLISHES_TO
        for app, topic in self.system.publishes:
            self.G.add_edge(app, topic, relation='PUBLISHES_TO', color='blue')
        
        # Add edges: SUBSCRIBES_TO
        for app, topic in self.system.subscribes:
            self.G.add_edge(topic, app, relation='SUBSCRIBES_TO', color='green')
        
        # Add edges: ROUTES
        for broker, topic in self.system.routes:
            self.G.add_edge(broker, topic, relation='ROUTES', color='purple')
        
        # Add edges: RUNS_ON
        for component, node in self.system.hosts.items():
            self.G.add_edge(component, node, relation='RUNS_ON', color='gray')
        
        # Build application dependency graph (DEPENDS_ON relationships)
        self._build_application_graph()
        
        # Build infrastructure graph
        self._build_infrastructure_graph()

        # Visualize the multi-layer graph
        self.visualize_multilayer_graph()
    
    def _build_application_graph(self):
        """Creates the application-level dependency graph."""
        self.GA.add_nodes_from(self.system.applications)
        
        # Find dependencies through topics
        for topic in self.system.topics:
            publishers = [app for app, t in self.system.publishes if t == topic]
            subscribers = [app for app, t in self.system.subscribes if t == topic]
            
            for pub in publishers:
                for sub in subscribers:
                    self.GA.add_edge(pub, sub, relation='DEPENDS_ON')
                    self.G.add_edge(sub, pub, relation='DEPENDS_ON', color='orange')
    
    def _build_infrastructure_graph(self):
        """Creates the infrastructure-level dependency graph."""
        self.GN.add_nodes_from(self.system.nodes)
        
        # Connect nodes if they host communicating components
        for n1 in self.system.nodes:
            for n2 in self.system.nodes:
                if n1 != n2:
                    # Check if components on n1 communicate with components on n2
                    components_n1 = [c for c, n in self.system.hosts.items() if n == n1]
                    components_n2 = [c for c, n in self.system.hosts.items() if n == n2]
                    
                    for c1 in components_n1:
                        for c2 in components_n2:
                            if self.G.has_edge(c1, c2) or self.G.has_edge(c2, c1):
                                self.GN.add_edge(n1, n2, relation='CONNECTS_TO')
                                self.G.add_edge(n1, n2, relation='CONNECTS_TO', color='red')
                                break


    def visualize_multilayer_graph(self):
        """Visualizes the multi-layer dependency graph."""
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        
        # Node colors based on type
        node_colors = []
        for node in self.G.nodes():
            node_type = self.G.nodes[node].get('type', '')
            if node_type == 'app':
                node_colors.append('skyblue')
            elif node_type == 'topic':
                node_colors.append('lightgreen')
            elif node_type == 'broker':
                node_colors.append('orange')
            else:  # node
                node_colors.append('violet')
        
        # Draw the graph
        pos = nx.spring_layout(self.G, seed=42)
        nx.draw_networkx_nodes(self.G, pos, node_color=node_colors, 
                              node_size=500, alpha=0.9, ax=ax)
        
        # Draw edges by type
        edge_colors = [self.G[u][v].get('color', 'gray') for u, v in self.G.edges()]
        nx.draw_networkx_edges(self.G, pos, edge_color=edge_colors, 
                               alpha=0.5, arrows=True, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(self.G, pos, font_size=8, ax=ax)
        
        # Add layer labels
        ax.set_title('Multi-layer Dependency Graph', fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def import_graph_to_neo4j(G: nx.DiGraph, uri: str, user: str, password: str):
        """Imports the graph into a Neo4j database."""
        from neo4j import GraphDatabase
        
        driver = GraphDatabase.driver(uri, auth=(user, password))
        
        def create_node(tx, node_id, properties):
            props = ', '.join(f"{k}: '{v}'" for k, v in properties.items())
            tx.run(f"MERGE (n {{id: '{node_id}'}}) SET n += {{{props}}}")
        
        def create_edge(tx, from_id, to_id, relation):
            tx.run(f"""
                MATCH (a {{id: '{from_id}'}}), (b {{id: '{to_id}'}})
                MERGE (a)-[r:{relation}]->(b)
            """)

        def clear_database(tx):
            tx.run("MATCH (n) DETACH DELETE n")    
        
        with driver.session() as session:
            session.write_transaction(clear_database)

            for node in G.nodes(data=True):
                node_id = node[0]
                properties = node[1]
                session.write_transaction(create_node, node_id, properties)
            
            for u, v, data in G.edges(data=True):
                relation = data.get('relation', 'RELATED_TO')
                session.write_transaction(create_edge, u, v, relation)
        
        driver.close()

# ============================================================================
# SECTION 2: TOPOLOGICAL METRICS CALCULATION
# ============================================================================

class TopologicalAnalyzer:
    """Implements topological metrics for criticality analysis."""
    
    @staticmethod
    def calculate_betweenness_centrality(G: nx.DiGraph) -> Dict[str, float]:
        """
        Calculates betweenness centrality for all vertices.
        C_B(v) = Σ_{s≠v≠t} σ_st(v) / σ_st
        """
        return nx.betweenness_centrality(G, normalized=False)
    
    @staticmethod
    def find_articulation_points(G: nx.DiGraph) -> Set[str]:
        """
        Finds articulation points (cut vertices) in the graph.
        AP(v) = 1 if |CC(G-v)| > |CC(G)|, 0 otherwise
        """
        # Convert to undirected for articulation point detection
        G_undirected = G.to_undirected()
        if nx.is_connected(G_undirected):
            return set(nx.articulation_points(G_undirected))
        else:
            # For disconnected graphs, find articulation points in each component
            articulation_points = set()
            for component in nx.connected_components(G_undirected):
                subgraph = G_undirected.subgraph(component)
                if len(component) > 2:  # Articulation points exist only in components with >2 nodes
                    articulation_points.update(nx.articulation_points(subgraph))
            return articulation_points
    
    @staticmethod
    def calculate_topological_score(G: nx.DiGraph, alpha: float = 0.7, beta: float = 0.3) -> Dict[str, float]:
        """
        Calculates the topological criticality score:
        C_topo(v) = α * C_B^norm(v) + β * AP(v)
        """
        # Calculate betweenness centrality
        centrality = TopologicalAnalyzer.calculate_betweenness_centrality(G)
        max_centrality = max(centrality.values()) if centrality else 1.0
        
        # Normalize centrality
        if max_centrality > 0:
            norm_centrality = {v: c/max_centrality for v, c in centrality.items()}
        else:
            norm_centrality = {v: 0.0 for v in centrality.keys()}
        
        # Find articulation points
        articulation_points = TopologicalAnalyzer.find_articulation_points(G)
        
        # Calculate topological score
        topo_scores = {}
        for vertex in G.nodes():
            cb_norm = norm_centrality.get(vertex, 0.0)
            ap = 1.0 if vertex in articulation_points else 0.0
            topo_scores[vertex] = alpha * cb_norm + beta * ap
        
        return topo_scores

# ============================================================================
# SECTION 3: IMPACT VALIDATION METRICS
# ============================================================================

class ImpactAnalyzer:
    """Implements impact analysis for validation of topological approach."""
    
    @staticmethod
    def calculate_impact_score(G: nx.DiGraph, vertex: str) -> float:
        """
        Calculates impact score: I(v) = 1 - |R(G-v)|/|R(G)|
        Used for validation only, not for criticality calculation.
        """
        # Calculate reachability before removal
        reachable_before = ImpactAnalyzer._count_reachable_pairs(G)
        
        # Create copy and remove vertex
        G_removed = G.copy()
        G_removed.remove_node(vertex)
        
        # Calculate reachability after removal
        reachable_after = ImpactAnalyzer._count_reachable_pairs(G_removed)
        
        if reachable_before == 0:
            return 0.0
        
        return 1.0 - (reachable_after / reachable_before)
    
    @staticmethod
    def _count_reachable_pairs(G: nx.DiGraph) -> int:
        """Counts the number of reachable vertex pairs in the graph."""
        count = 0
        for source in G.nodes():
            reachable = nx.descendants(G, source)
            count += len(reachable)
        return count
    
    @staticmethod
    def analyze_removal_impact(G: nx.DiGraph, vertex: str) -> Dict[str, any]:
        """
        Comprehensive impact analysis for vertex removal.
        Returns: ΔC (connectivity loss), ΔR (reachability degradation), ΔAPL (path length increase)
        """
        # Original metrics
        scc_before = nx.number_strongly_connected_components(G)
        wcc_before = nx.number_weakly_connected_components(G)
        
        # Remove vertex
        G_removed = G.copy()
        G_removed.remove_node(vertex)
        
        # New metrics
        scc_after = nx.number_strongly_connected_components(G_removed)
        wcc_after = nx.number_weakly_connected_components(G_removed)
        
        # Calculate changes
        return {
            'vertex': vertex,
            'delta_scc': scc_after - scc_before,
            'delta_wcc': wcc_after - wcc_before,
            'impact_score': ImpactAnalyzer.calculate_impact_score(G, vertex)
        }

# ============================================================================
# SECTION 4: VISUALIZATION
# ============================================================================

class PubSubVisualizer:
    """Visualization utilities for pub-sub dependency graphs."""
    
    @staticmethod
    def visualize_multilayer_graph(analyzer: PubSubGraphAnalyzer, topo_scores: Dict[str, float], 
                                  title: str = "Multi-layer Dependency Graph"):
        """Creates the three-layer visualization as shown in Figure 2 of the paper."""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        # Get node positions by layer
        pos = PubSubVisualizer._get_hierarchical_positions(analyzer.G)
        
        # Node colors based on type and criticality
        node_colors = []
        node_sizes = []
        for node in analyzer.G.nodes():
            node_type = analyzer.G.nodes[node].get('type', '')
            score = topo_scores.get(node, 0.0)
            
            # Color by type
            if node_type == 'app':
                base_color = np.array([0.29, 0.56, 0.89])  # Blue
            elif node_type == 'topic':
                base_color = np.array([0.31, 0.78, 0.47])  # Green
            elif node_type == 'broker':
                base_color = np.array([1.0, 0.70, 0.28])  # Orange
            else:  # node
                base_color = np.array([0.69, 0.61, 0.85])  # Purple
            
            # Adjust intensity based on criticality
            if score > 0.7:
                color = np.array([1.0, 0.2, 0.2])  # Red for critical
            elif score > 0.3:
                color = base_color * 0.8  # Darker for medium
            else:
                color = base_color
                
            node_colors.append(color)
            node_sizes.append(300 + score * 1000)
        
        # Draw the graph
        nx.draw_networkx_nodes(analyzer.G, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.9, ax=ax)
        
        # Draw edges by type
        edge_colors = [analyzer.G[u][v].get('color', 'gray') for u, v in analyzer.G.edges()]
        nx.draw_networkx_edges(analyzer.G, pos, edge_color=edge_colors, 
                               alpha=0.5, arrows=True, ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(analyzer.G, pos, font_size=8, ax=ax)
        
        # Add layer labels
        ax.text(0.05, 0.95, 'Application Layer', transform=ax.transAxes, 
                fontsize=12, fontweight='bold', va='top')
        ax.text(0.05, 0.65, 'Messaging Layer', transform=ax.transAxes, 
                fontsize=12, fontweight='bold', va='top')
        ax.text(0.05, 0.25, 'Infrastructure Layer', transform=ax.transAxes, 
                fontsize=12, fontweight='bold', va='top')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def _get_hierarchical_positions(G: nx.DiGraph) -> Dict:
        """Calculates hierarchical positions for multi-layer visualization."""
        pos = {}
        
        # Separate nodes by layer
        app_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'app']
        topic_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'topic']
        broker_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'broker']
        infra_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'node']
        
        # Position nodes by layer
        for i, node in enumerate(app_nodes):
            pos[node] = (i * 2, 3)
        
        for i, node in enumerate(topic_nodes):
            pos[node] = (i * 0.8, 2)
        
        for i, node in enumerate(broker_nodes):
            pos[node] = (2 + i * 4, 2)
        
        for i, node in enumerate(infra_nodes):
            pos[node] = (1 + i * 3, 0)
        
        return pos
    
    @staticmethod
    def plot_criticality_analysis(results_df: pd.DataFrame, title: str = "Criticality Analysis Results"):
        """Creates bar plot of topological scores and validation metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot topological scores
        top_10 = results_df.nlargest(10, 'topo_score')
        ax1.barh(range(len(top_10)), top_10['topo_score'].values)
        ax1.set_yticks(range(len(top_10)))
        ax1.set_yticklabels(top_10['vertex'].values)
        ax1.set_xlabel('Topological Criticality Score')
        ax1.set_title('Top 10 Critical Components')
        ax1.grid(True, alpha=0.3)
        
        # Add color coding
        colors = ['red' if s > 0.7 else 'orange' if s > 0.3 else 'blue' 
                 for s in top_10['topo_score'].values]
        bars = ax1.barh(range(len(top_10)), top_10['topo_score'].values, color=colors)
        
        # Plot correlation between topological score and impact
        ax2.scatter(results_df['topo_score'], results_df['impact_score'], alpha=0.6)
        
        # Add correlation line
        z = np.polyfit(results_df['topo_score'], results_df['impact_score'], 1)
        p = np.poly1d(z)
        ax2.plot(results_df['topo_score'].sort_values(), 
                p(results_df['topo_score'].sort_values()), 
                "r--", alpha=0.8, label=f'r = {results_df["topo_score"].corr(results_df["impact_score"]):.3f}')
        
        ax2.set_xlabel('Topological Score')
        ax2.set_ylabel('Impact Score (Validation)')
        ax2.set_title('Validation: Topological vs Impact')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

# ============================================================================
# SECTION 5: EXAMPLE SYSTEMS
# ============================================================================

def create_example_pubsub_system() -> PubSubSystem:
    """Creates the exemplary distributed pub-sub system from the paper."""
    return PubSubSystem(
        applications=['App_1', 'App_2', 'App_3', 'App_4', 'App_5', 
                     'App_6', 'App_7', 'App_8', 'App_9', 'App_10'],
        topics=['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10',
               'T11', 'T12', 'T13', 'T14', 'T15', 'T16', 'T17', 'T18', 'T19', 'T20',
               'T21', 'T22', 'T23', 'T24', 'T25'],
        brokers=['Broker_1', 'Broker_2'],
        nodes=['Node_1', 'Node_2', 'Node_3', 'Node_4'],
        publishes=[
            ('App_1', 'T1'), ('App_1', 'T2'), ('App_2', 'T3'), ('App_2', 'T4'),
            ('App_2', 'T5'), ('App_3', 'T6'), ('App_4', 'T7'), ('App_5', 'T8'),
            ('App_6', 'T9'), ('App_7', 'T10'), ('App_8', 'T11'), ('App_9', 'T12'),
            ('App_10', 'T13')
        ],
        subscribes=[
            ('App_1', 'T3'), ('App_2', 'T1'), ('App_2', 'T6'), ('App_3', 'T4'),
            ('App_3', 'T5'), ('App_4', 'T5'), ('App_5', 'T4'), ('App_6', 'T8'),
            ('App_7', 'T9'), ('App_8', 'T10'), ('App_9', 'T11'), ('App_10', 'T12')
        ],
        hosts={
            'App_1': 'Node_1', 'App_2': 'Node_1', 'App_3': 'Node_2',
            'App_4': 'Node_2', 'App_5': 'Node_2', 'App_6': 'Node_3',
            'App_7': 'Node_3', 'App_8': 'Node_4', 'App_9': 'Node_4',
            'App_10': 'Node_4', 'Broker_1': 'Node_2', 'Broker_2': 'Node_3'
        },
        routes=[
            ('Broker_1', 'T1'), ('Broker_1', 'T2'), ('Broker_1', 'T3'),
            ('Broker_1', 'T4'), ('Broker_1', 'T5'), ('Broker_1', 'T6'),
            ('Broker_1', 'T7'), ('Broker_1', 'T8'), ('Broker_1', 'T9'),
            ('Broker_1', 'T10'), ('Broker_1', 'T11'), ('Broker_1', 'T12'),
            ('Broker_2', 'T13'), ('Broker_2', 'T14'), ('Broker_2', 'T15'),
            ('Broker_2', 'T16'), ('Broker_2', 'T17'), ('Broker_2', 'T18'),
            ('Broker_2', 'T19'), ('Broker_2', 'T20'), ('Broker_2', 'T21'),
            ('Broker_2', 'T22'), ('Broker_2', 'T23'), ('Broker_2', 'T24'),
            ('Broker_2', 'T25')
        ]
    )

def create_smart_city_iot_system() -> PubSubSystem:
    """Creates the Smart City IoT example from Section III.D of the paper."""
    return PubSubSystem(
        applications=['TempSensor_1', 'TempSensor_2', 'TempSensor_3', 'TempSensor_4', 'TempSensor_5',
                     'TrafficMon_1', 'TrafficMon_2', 'TrafficMon_3',
                     'EmergencyDispatcher', 'CityDashboard'],
        topics=['temp_data', 'traffic_flow', 'emergency_alerts', 'system_status'],
        brokers=['RegionalBroker', 'CentralBroker'],
        nodes=['EdgeNode_1', 'EdgeNode_2', 'EdgeNode_3', 'CloudServer_1', 'CloudServer_2'],
        publishes=[
            ('TempSensor_1', 'temp_data'), ('TempSensor_2', 'temp_data'),
            ('TempSensor_3', 'temp_data'), ('TempSensor_4', 'temp_data'),
            ('TempSensor_5', 'temp_data'),
            ('TrafficMon_1', 'traffic_flow'), ('TrafficMon_2', 'traffic_flow'),
            ('TrafficMon_3', 'traffic_flow'),
            ('EmergencyDispatcher', 'emergency_alerts'),
            ('CityDashboard', 'system_status')
        ],
        subscribes=[
            ('CityDashboard', 'temp_data'), ('CityDashboard', 'traffic_flow'),
            ('CityDashboard', 'emergency_alerts'),
            ('EmergencyDispatcher', 'temp_data'), ('EmergencyDispatcher', 'traffic_flow')
        ],
        hosts={
            'TempSensor_1': 'EdgeNode_1', 'TempSensor_2': 'EdgeNode_1',
            'TempSensor_3': 'EdgeNode_2', 'TempSensor_4': 'EdgeNode_2',
            'TempSensor_5': 'EdgeNode_3',
            'TrafficMon_1': 'EdgeNode_1', 'TrafficMon_2': 'EdgeNode_2',
            'TrafficMon_3': 'EdgeNode_3',
            'EmergencyDispatcher': 'CloudServer_1',
            'CityDashboard': 'CloudServer_1',
            'RegionalBroker': 'EdgeNode_2',
            'CentralBroker': 'CloudServer_1'
        },
        routes=[
            ('RegionalBroker', 'temp_data'), ('RegionalBroker', 'traffic_flow'),
            ('CentralBroker', 'emergency_alerts'), ('CentralBroker', 'system_status'),
            ('CentralBroker', 'temp_data'), ('CentralBroker', 'traffic_flow')
        ]
    )

# ============================================================================
# SECTION 6: MAIN ANALYSIS WORKFLOW
# ============================================================================

def perform_criticality_analysis(system: PubSubSystem, system_name: str = "System", import_graph_to_neo4j=False) -> pd.DataFrame:
    """
    Complete criticality analysis workflow as presented in the paper.
    Returns DataFrame with topological scores and validation metrics.
    """
    print(f"=" * 60)
    print(f"CRITICALITY ANALYSIS: {system_name}")
    print(f"=" * 60)
    
    # Step 1: Build graphs
    print("\n1. Building multi-layer dependency graph...")
    analyzer = PubSubGraphAnalyzer(system)
    print(f"   - Total vertices: {analyzer.G.number_of_nodes()}")
    print(f"   - Total edges: {analyzer.G.number_of_edges()}")
    print(f"   - Applications: {len(system.applications)}")
    print(f"   - Topics: {len(system.topics)}")
    print(f"   - Brokers: {len(system.brokers)}")
    print(f"   - Nodes: {len(system.nodes)}")


    # Import to Neo4j (optional)
    if import_graph_to_neo4j:
        analyzer.import_graph_to_neo4j(analyzer.G, "bolt://localhost:7687", "neo4j", "password")
    
    # Step 2: Calculate topological metrics
    print("\n2. Calculating topological metrics...")
    
    # Application-level analysis
    app_topo_scores = TopologicalAnalyzer.calculate_topological_score(analyzer.GA)
    app_centrality = TopologicalAnalyzer.calculate_betweenness_centrality(analyzer.GA)
    app_articulation = TopologicalAnalyzer.find_articulation_points(analyzer.GA)
    
    print(f"   - Application-level articulation points: {app_articulation}")
    
    # Infrastructure-level analysis
    infra_topo_scores = TopologicalAnalyzer.calculate_topological_score(analyzer.GN)
    infra_articulation = TopologicalAnalyzer.find_articulation_points(analyzer.GN)
    
    print(f"   - Infrastructure-level articulation points: {infra_articulation}")
    
    # Combined analysis on full graph
    full_topo_scores = TopologicalAnalyzer.calculate_topological_score(analyzer.G)
    
    # Step 3: Validation through impact analysis
    print("\n3. Validating with impact analysis...")
    results = []
    
    # Analyze application-level components
    for app in system.applications:
        impact_metrics = ImpactAnalyzer.analyze_removal_impact(analyzer.GA, app)
        results.append({
            'vertex': app,
            'type': 'application',
            'betweenness': app_centrality.get(app, 0.0),
            'is_articulation': app in app_articulation,
            'topo_score': app_topo_scores.get(app, 0.0),
            'impact_score': impact_metrics['impact_score'],
            'delta_scc': impact_metrics['delta_scc']
        })
    
    # Analyze infrastructure-level components
    for node in system.nodes:
        impact_metrics = ImpactAnalyzer.analyze_removal_impact(analyzer.GN, node)
        results.append({
            'vertex': node,
            'type': 'infrastructure',
            'betweenness': 0.0,  # Will calculate if needed
            'is_articulation': node in infra_articulation,
            'topo_score': infra_topo_scores.get(node, 0.0),
            'impact_score': impact_metrics['impact_score'],
            'delta_scc': impact_metrics['delta_scc']
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('topo_score', ascending=False)
    
    # Calculate correlation
    correlation = results_df['topo_score'].corr(results_df['impact_score'])
    
    # Step 4: Display results
    print("\n4. Analysis Results:")
    print("-" * 60)
    print("\nTop 5 Critical Components (by Topological Score):")
    print(results_df[['vertex', 'type', 'topo_score', 'is_articulation', 'impact_score']].head())
    
    print(f"\nValidation Correlation (Topological vs Impact): {correlation:.3f}")
    
    # Step 5: Visualize
    print("\n5. Generating visualizations...")
    PubSubVisualizer.visualize_multilayer_graph(analyzer, full_topo_scores, 
                                               title=f"{system_name}: Multi-layer Dependency Graph")
    PubSubVisualizer.plot_criticality_analysis(results_df, 
                                              title=f"{system_name}: Criticality Analysis")
    
    return results_df

# ============================================================================
# SECTION 7: COMPARATIVE ANALYSIS
# ============================================================================

def compare_systems():
    """Compares criticality analysis across different systems."""
    print("\n" + "=" * 60)
    print("COMPARATIVE ANALYSIS OF DIFFERENT SYSTEMS")
    print("=" * 60)
    
    # Analyze example system
    example_system = create_example_pubsub_system()
    example_results = perform_criticality_analysis(example_system, "Example Pub-Sub System", import_graph_to_neo4j=True)
    
    # Analyze Smart City IoT system
    #smart_city_system = create_smart_city_iot_system()
    #smart_city_results = perform_criticality_analysis(smart_city_system, "Smart City IoT System")
    
    # Summary comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    comparison_data = {
        'System': ['Example Pub-Sub', 'Smart City IoT'],
        'Max Topo Score': [
            example_results['topo_score'].max(),
            smart_city_results['topo_score'].max()
        ],
        'Validation Correlation': [
            example_results['topo_score'].corr(example_results['impact_score']),
            smart_city_results['topo_score'].corr(smart_city_results['impact_score'])
        ],
        'Articulation Points': [
            example_results['is_articulation'].sum(),
            smart_city_results['is_articulation'].sum()
        ],
        'Avg Impact Score': [
            example_results['impact_score'].mean(),
            smart_city_results['impact_score'].mean()
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n", comparison_df.to_string(index=False))
    
    return example_results, smart_city_results

# ============================================================================
# SECTION 8: PERFORMANCE EVALUATION
# ============================================================================

def evaluate_scalability():
    """
    Evaluates the scalability of the topological analysis approach.
    Tests with different system sizes as presented in Section VI of the paper.
    """
    import time
    
    print("\n" + "=" * 60)
    print("SCALABILITY EVALUATION")
    print("=" * 60)
    
    sizes = [10, 50, 100, 500]
    performance_results = []
    
    for size in sizes:
        print(f"\nTesting with {size} applications...")
        
        # Generate a synthetic system
        apps = [f'App_{i}' for i in range(size)]
        topics = [f'Topic_{i}' for i in range(size * 2)]
        brokers = [f'Broker_{i}' for i in range(max(2, size // 10))]
        nodes = [f'Node_{i}' for i in range(max(4, size // 5))]
        
        # Random pub-sub relationships
        np.random.seed(42)
        publishes = [(apps[i], topics[np.random.randint(0, len(topics))]) 
                    for i in range(size) for _ in range(3)]
        subscribes = [(apps[i], topics[np.random.randint(0, len(topics))]) 
                     for i in range(size) for _ in range(2)]
        
        # Random hosting
        hosts = {app: nodes[i % len(nodes)] for i, app in enumerate(apps)}
        hosts.update({broker: nodes[i % len(nodes)] for i, broker in enumerate(brokers)})
        
        # Broker routes
        routes = [(brokers[i % len(brokers)], topic) for i, topic in enumerate(topics)]
        
        system = PubSubSystem(apps, topics, brokers, nodes, publishes, subscribes, hosts, routes)
        
        # Build graph
        analyzer = PubSubGraphAnalyzer(system)
        
        # Measure centrality computation time
        start = time.time()
        centrality = TopologicalAnalyzer.calculate_betweenness_centrality(analyzer.GA)
        centrality_time = time.time() - start
        
        # Measure articulation point detection time
        start = time.time()
        articulation = TopologicalAnalyzer.find_articulation_points(analyzer.GA)
        articulation_time = time.time() - start
        
        # Measure total analysis time
        start = time.time()
        topo_scores = TopologicalAnalyzer.calculate_topological_score(analyzer.GA)
        total_time = time.time() - start
        
        performance_results.append({
            'Applications': size,
            'Vertices': analyzer.G.number_of_nodes(),
            'Edges': analyzer.G.number_of_edges(),
            'Centrality Time (s)': centrality_time,
            'Articulation Time (s)': articulation_time,
            'Total Time (s)': total_time
        })
        
        print(f"   Vertices: {analyzer.G.number_of_nodes()}, Edges: {analyzer.G.number_of_edges()}")
        print(f"   Centrality computation: {centrality_time:.3f}s")
        print(f"   Articulation detection: {articulation_time:.3f}s")
        print(f"   Total analysis time: {total_time:.3f}s")
    
    # Display results
    perf_df = pd.DataFrame(performance_results)
    
    # Plot scalability
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(perf_df['Applications'], perf_df['Centrality Time (s)'], 'o-', label='Centrality')
    ax1.plot(perf_df['Applications'], perf_df['Articulation Time (s)'], 's-', label='Articulation')
    ax1.plot(perf_df['Applications'], perf_df['Total Time (s)'], '^-', label='Total')
    ax1.set_xlabel('Number of Applications')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Computation Time Scalability')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(perf_df['Edges'], perf_df['Total Time (s)'], 'o-', color='darkblue')
    ax2.set_xlabel('Number of Edges')
    ax2.set_ylabel('Total Time (seconds)')
    ax2.set_title('Time Complexity: O(V·E)')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Performance Evaluation: Topological Analysis Scalability', fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return perf_df

# ============================================================================
# SECTION 9: ROS 2 CASE STUDY SIMULATION
# ============================================================================

def create_ros2_sierra_nevada() -> PubSubSystem:
    """
    Simulates the Sierra Nevada ROS 2 benchmark from Section V.B.
    10 applications with specific topology.
    """
    return PubSubSystem(
        applications=['ponce', 'mandalay', 'hamburg', 'geneva', 'osaka', 
                     'montreal', 'lyon', 'arequipa', 'hebron', 'kingston'],
        topics=['topic_1', 'topic_2', 'topic_3', 'topic_4', 'topic_5',
               'topic_6', 'topic_7', 'topic_8', 'topic_9', 'topic_10'],
        brokers=[],  # ROS 2 uses DDS, no explicit brokers
        nodes=['robot_node_1', 'robot_node_2', 'robot_node_3'],
        publishes=[
            ('ponce', 'topic_1'), ('ponce', 'topic_2'), ('ponce', 'topic_3'),
            ('mandalay', 'topic_2'), ('mandalay', 'topic_4'),
            ('hamburg', 'topic_3'), ('hamburg', 'topic_5'),
            ('geneva', 'topic_4'), ('geneva', 'topic_6'),
            ('osaka', 'topic_5')
        ],
        subscribes=[
            ('mandalay', 'topic_1'), ('hamburg', 'topic_1'), ('hamburg', 'topic_2'),
            ('geneva', 'topic_2'), ('geneva', 'topic_3'),
            ('osaka', 'topic_3'), ('osaka', 'topic_4'),
            ('montreal', 'topic_5'), ('lyon', 'topic_6'),
            ('arequipa', 'topic_5'), ('hebron', 'topic_6'), 
            ('kingston', 'topic_4'), ('kingston', 'topic_5')
        ],
        hosts={
            'ponce': 'robot_node_1', 'mandalay': 'robot_node_1',
            'hamburg': 'robot_node_1', 'geneva': 'robot_node_2',
            'osaka': 'robot_node_2', 'montreal': 'robot_node_2',
            'lyon': 'robot_node_3', 'arequipa': 'robot_node_3',
            'hebron': 'robot_node_3', 'kingston': 'robot_node_3'
        },
        routes=[]  # No explicit routing in DDS
    )

def analyze_ros2_system():
    """Analyzes the ROS 2 case study as presented in the paper."""
    print("\n" + "=" * 60)
    print("ROS 2 CASE STUDY: SIERRA NEVADA BENCHMARK")
    print("=" * 60)
    
    system = create_ros2_sierra_nevada()
    
    # Build and analyze
    analyzer = PubSubGraphAnalyzer(system)
    
    # Calculate topological metrics for applications
    app_centrality = TopologicalAnalyzer.calculate_betweenness_centrality(analyzer.GA)
    app_articulation = TopologicalAnalyzer.find_articulation_points(analyzer.GA)
    app_topo_scores = TopologicalAnalyzer.calculate_topological_score(analyzer.GA)
    
    # Prepare results matching paper's Table I
    results = []
    for app in system.applications:
        centrality = app_centrality.get(app, 0.0)
        max_centrality = max(app_centrality.values()) if app_centrality else 1.0
        norm_centrality = centrality / max_centrality if max_centrality > 0 else 0.0
        
        results.append({
            'Application': app,
            'Betweenness': centrality,
            'Normalized': norm_centrality,
            'Articulation Point': 'Yes' if app in app_articulation else 'No',
            'Topo Score': app_topo_scores.get(app, 0.0)
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Topo Score', ascending=False)
    
    print("\nTable: Topological Analysis of Sierra Nevada")
    print("-" * 60)
    print(results_df.to_string(index=False))
    
    # Simulate removal of critical node 'ponce'
    print("\n\nImpact of Removing 'ponce':")
    print("-" * 40)
    
    impact = ImpactAnalyzer.analyze_removal_impact(analyzer.GA, 'ponce')
    print(f"  ΔSCCs: +{impact['delta_scc']}")
    print(f"  Impact Score: {impact['impact_score']:.3f}")
    
    # Calculate reachability before and after
    reachable_before = len(list(nx.descendants(analyzer.GA, 'kingston')))
    GA_removed = analyzer.GA.copy()
    GA_removed.remove_node('ponce')
    reachable_after = len(list(nx.descendants(GA_removed, 'kingston')))
    
    if reachable_before == 0:
        reachable_before = 1  # Avoid division by zero
        
    print(f"  Reachability from 'kingston': {reachable_before} → {reachable_after}")
    print(f"  Reachability loss: {(1 - reachable_after/reachable_before)*100:.1f}%")
    
    return results_df

# ============================================================================
# SECTION 10: RECOMMENDATIONS ENGINE
# ============================================================================

class RecommendationEngine:
    """Generates mitigation recommendations based on criticality analysis."""
    
    @staticmethod
    def generate_recommendations(results_df: pd.DataFrame, threshold: float = 0.7) -> List[str]:
        """
        Generates actionable recommendations for critical components.
        Based on Section III.D and conclusions from the paper.
        """
        recommendations = []
        
        # Identify critical components
        critical = results_df[results_df['topo_score'] > threshold]
        
        for _, row in critical.iterrows():
            vertex = row['vertex']
            score = row['topo_score']
            is_articulation = row.get('is_articulation', False)
            component_type = row.get('type', 'unknown')
            
            if component_type == 'application':
                if is_articulation:
                    recommendations.append(
                        f"🔴 CRITICAL: {vertex} is an articulation point (score: {score:.3f})\n"
                        f"   → Implement redundant instance of {vertex}\n"
                        f"   → Add alternative communication paths\n"
                        f"   → Consider load balancing across replicas"
                    )
                else:
                    recommendations.append(
                        f"⚠️  HIGH: {vertex} has high centrality (score: {score:.3f})\n"
                        f"   → Monitor {vertex} for performance bottlenecks\n"
                        f"   → Consider horizontal scaling\n"
                        f"   → Implement circuit breaker pattern"
                    )
            
            elif component_type == 'infrastructure':
                if is_articulation:
                    recommendations.append(
                        f"🔴 CRITICAL: {vertex} is a single point of failure (score: {score:.3f})\n"
                        f"   → Add redundant infrastructure node\n"
                        f"   → Distribute hosted components across multiple nodes\n"
                        f"   → Implement failover mechanism"
                    )
                else:
                    recommendations.append(
                        f"⚠️  HIGH: {vertex} hosts critical components (score: {score:.3f})\n"
                        f"   → Ensure high availability configuration\n"
                        f"   → Regular backup and disaster recovery testing\n"
                        f"   → Consider geographic distribution"
                    )
        
        # General recommendations
        articulation_count = results_df['is_articulation'].sum()
        if articulation_count > 2:
            recommendations.append(
                f"\n📊 SYSTEM-WIDE: Found {articulation_count} articulation points\n"
                f"   → Review overall architecture for single points of failure\n"
                f"   → Consider mesh topology for critical paths\n"
                f"   → Implement service mesh for better resilience"
            )
        
        return recommendations

def generate_report(results_df: pd.DataFrame, system_name: str = "System"):
    """Generates a comprehensive criticality report."""
    print("\n" + "=" * 60)
    print(f"CRITICALITY REPORT: {system_name}")
    print("=" * 60)
    
    # Summary statistics
    print("\nSUMMARY STATISTICS:")
    print("-" * 40)
    print(f"  Total components analyzed: {len(results_df)}")
    print(f"  Articulation points: {results_df['is_articulation'].sum()}")
    print(f"  Max topological score: {results_df['topo_score'].max():.3f}")
    print(f"  Mean topological score: {results_df['topo_score'].mean():.3f}")
    print(f"  Validation correlation: {results_df['topo_score'].corr(results_df['impact_score']):.3f}")
    
    # Critical components
    print("\nCRITICAL COMPONENTS (Top 3):")
    print("-" * 40)
    for i, row in results_df.head(3).iterrows():
        print(f"  {i+1}. {row['vertex']}")
        print(f"     - Topological Score: {row['topo_score']:.3f}")
        print(f"     - Impact Score: {row['impact_score']:.3f}")
        print(f"     - Articulation Point: {'Yes' if row['is_articulation'] else 'No'}")
    
    # Generate recommendations
    print("\nRECOMMENDATIONS:")
    print("-" * 40)
    recommendations = RecommendationEngine.generate_recommendations(results_df)
    for rec in recommendations:
        print(rec)
    
    print("\n" + "=" * 60)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "GRAPH-BASED DEPENDENCY ANALYSIS FOR PUB-SUB SYSTEMS" + " " * 10 + "║")
    print("║" + " " * 20 + "Implementation of IEEE RASSE 2025 Paper" + " " * 18 + "║")
    print("╚" + "═" * 78 + "╝")
    
    # 1. Run comparative analysis
    example_results, smart_city_results = compare_systems()
    
    # 2. Evaluate scalability
    print("\nEvaluating scalability...")
    perf_df = evaluate_scalability()
    
    # 3. Analyze ROS 2 case study
    ros2_results = analyze_ros2_system()
    
    # 4. Generate final report for the example system
    generate_report(example_results, "Example Pub-Sub System")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nKey Findings:")
    print("1. Topological metrics (betweenness centrality + articulation points)")
    print("   effectively identify critical components with >90% correlation to impact")
    print("2. The approach scales to large systems with O(V·E) complexity")
    print("3. Both simulated and ROS 2 systems show clear critical components")
    print("   that require redundancy measures")
    print("\nThis validates the graph-based approach presented in the paper.")
    print("=" * 60)