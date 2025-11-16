"""
Comprehensive Analysis of Critical Components and Paths in Distributed Publish-Subscribe Systems

This notebook demonstrates:
1. Multi-layered graph modeling of pub-sub architecture
2. Topological metrics (centrality, articulation points)
3. QoS-based topic importance
4. Composite criticality scoring
5. Impact score through reachability analysis
6. Critical path identification
7. Detection of design defects and anti-patterns:
   - Single Points of Failure (SPOF)
   - God Topics
   - Circular Dependencies
   - Hidden Coupling
"""

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# ============================================================================
# FORMAL MATHEMATICAL DEFINITIONS
# ============================================================================

def print_mathematical_framework():
    """Display formal mathematical definitions of the approach"""
    
    print("="*80)
    print("FORMAL MATHEMATICAL FRAMEWORK")
    print("="*80)
    print()
    
    print("1. MULTI-LAYER GRAPH MODEL")
    print("-" * 80)
    print("""
A distributed pub-sub system is modeled as a multi-layer directed graph:
    
    G = (V, E, L, λ, ε)
    
where:
    • V: Set of vertices (components)
    • E ⊆ V × V: Set of directed edges (dependencies)
    • L = {Physical, Broker, Logical, Application}: Set of layers
    • λ: V → L: Layer assignment function
    • ε: E → {publishes, subscribes, hosted_on, managed_by, replicates}: 
         Edge type function

Component types:
    V = V_node ∪ V_broker ∪ V_topic ∪ V_app
    
    where V_node: physical nodes
          V_broker: message brokers
          V_topic: topics
          V_app = V_pub ∪ V_sub: publishers and subscribers
""")
    
    print("\n2. QoS-BASED TOPIC IMPORTANCE")
    print("-" * 80)
    print("""
For each topic t ∈ V_topic, define QoS profile:
    
    Q(t) = (λ_t, τ_t, ρ_t, π_t)
    
where:
    • λ_t: latency requirement (ms)
    • τ_t: throughput requirement (msgs/sec)
    • ρ_t: reliability requirement [0,1]
    • π_t: priority level [1,5]

Topic importance score:
    
    I_qos(t) = w_λ·f_λ(λ_t) + w_τ·f_τ(τ_t) + w_ρ·ρ_t + w_π·(π_t/5)
    
where f_λ(λ) = 1/(1 + λ/100) (normalized latency)
      f_τ(τ) = min(τ/10000, 1) (normalized throughput)
      w_λ = 0.3, w_τ = 0.2, w_ρ = 0.3, w_π = 0.2 (weights)

For non-topic components v ∉ V_topic: I_qos(v) = 0
""")
    
    print("\n3. TOPOLOGICAL CRITICALITY METRICS")
    print("-" * 80)
    print("""
Centrality measures (normalized to [0,1]):

a) Degree Centrality:
    C_deg(v) = (deg_in(v) + deg_out(v)) / (2(|V|-1))

b) Betweenness Centrality:
    C_btw(v) = Σ_{s≠v≠t} (σ_st(v) / σ_st) / ((|V|-1)(|V|-2))
    
    where σ_st = number of shortest paths from s to t
          σ_st(v) = number of those paths passing through v

c) Closeness Centrality:
    C_cls(v) = (|V|-1) / Σ_{u≠v} d(v,u)
    
    where d(v,u) = shortest path distance

d) PageRank:
    PR(v) = (1-d)/|V| + d·Σ_{u∈In(v)} PR(u)/|Out(u)|
    
    where d = 0.85 (damping factor)
          In(v) = predecessors of v
          Out(u) = successors of u

Articulation points:
    A ⊆ V = {v : removing v increases number of connected components}

Composite topology score:
    S_topo(v) = Σ_i w_i·C_i(v)  ·  (1.5 if v ∈ A else 1.0)
    
    where w_btw = 0.35, w_deg = 0.25, w_pr = 0.20, 
          w_cls = 0.15, w_eig = 0.05
""")
    
    print("\n4. IMPACT SCORE (REACHABILITY LOSS)")
    print("-" * 80)
    print("""
Define reachability matrix R for graph G:
    
    R[u,v] = { 1  if there exists path from u to v
             { 0  otherwise

Total reachability:
    R_total(G) = Σ_{u,v∈V} R[u,v]

Impact score of removing vertex v:
    
    I_impact(v) = (R_total(G) - R_total(G-v)) / R_total(G)
    
where G-v is the graph with vertex v removed.

Interpretation:
    • I_impact(v) = 0: No impact (isolated component)
    • I_impact(v) = 1: Complete system failure
    • High I_impact(v): Critical component whose loss severely affects connectivity
""")
    
    print("\n5. COMPOSITE CRITICALITY SCORE")
    print("-" * 80)
    print("""
Component-type specific weights:
    
    W = { (α_topic, β_topic) = (0.4, 0.6)     for topics
        { (α_broker, β_broker) = (0.8, 0.2)   for brokers
        { (α_node, β_node) = (0.9, 0.1)       for nodes
        { (α_app, β_app) = (0.7, 0.3)         for applications

Composite criticality:
    
    S_comp(v) = α_type(v)·S_topo(v) + β_type(v)·I_qos(v)

Integrated score (combining topology, QoS, and impact):
    
    S_final(v) = 0.5·S_comp(v) + 0.5·I_impact(v)
""")
    
    print("\n6. ANTI-PATTERN FORMAL DEFINITIONS")
    print("-" * 80)
    print("""
a) Single Point of Failure (SPOF):
    SPOF = {v ∈ A : I_impact(v) > θ_spof}
    
    where θ_spof = 0.15 (impact threshold)

b) God Topic:
    GodTopics = {t ∈ V_topic : |N_in(t)| + |N_out(t)| ≥ θ_god}
    
    where N_in(t) = {p ∈ V_pub : (p,t) ∈ E}  (publishers)
          N_out(t) = {s ∈ V_sub : (t,s) ∈ E}  (subscribers)
          θ_god = 5 (connection threshold)

c) Circular Dependencies:
    Cycles = {C ⊆ V : C forms a cycle in G and |C ∩ V_app| ≥ 1}
    
    A cycle C = (v₁, v₂, ..., v_k, v₁) where (v_i, v_{i+1}) ∈ E

d) Hidden Coupling:
    HiddenCoupling = {(a₁, a₂) ∈ V_app × V_app : 
                      |Topics(a₁) ∩ Topics(a₂)| ≥ θ_coupling}
    
    where Topics(a) = {t : (a,t) ∈ E or (t,a) ∈ E}
          θ_coupling = 3 (shared topic threshold)
""")
    
    print("\n7. CRITICAL PATH ANALYSIS")
    print("-" * 80)
    print("""
For path P = (v₁, v₂, ..., v_k) from publisher to subscriber:

Path criticality score:
    
    S_path(P) = 0.6·min_{i=1..k} S_final(v_i) + 0.4·(1/k)·Σ_{i=1}^k S_final(v_i)
    
    (weighted combination of weakest link and average)

Critical paths:
    CP = {P : S_path(P) ≥ θ_path}
    
    where θ_path = 0.7 (criticality threshold)
""")
    
    print("="*80)
    print()


# ============================================================================
# ILLUSTRATIVE TOY EXAMPLE
# ============================================================================

def demonstrate_toy_example():
    """Simple example with explicit calculations"""
    
    print("="*80)
    print("ILLUSTRATIVE TOY EXAMPLE: STEP-BY-STEP CALCULATION")
    print("="*80)
    print()
    
    print("Consider a minimal pub-sub system:")
    print()
    print("    Components:")
    print("    • N1: Physical node")
    print("    • B1: Broker (on N1)")
    print("    • T1: Topic 'orders' (on B1)")
    print("    • P1: Publisher 'order_service'")
    print("    • S1: Subscriber 'warehouse'")
    print("    • S2: Subscriber 'billing'")
    print()
    print("    Dependencies:")
    print("    • P1 --publishes--> T1")
    print("    • T1 --subscribes--> S1")
    print("    • T1 --subscribes--> S2")
    print("    • T1 --managed_by--> B1")
    print("    • B1 --hosted_on--> N1")
    print()
    print("    Visual representation:")
    print()
    print("        P1 -----> T1 -----> S1")
    print("                   |")
    print("                   +------> S2")
    print("                   |")
    print("                   v")
    print("                   B1")
    print("                   |")
    print("                   v")
    print("                   N1")
    print()
    print("-" * 80)
    print()
    
    # Create the toy graph
    V = {'N1', 'B1', 'T1', 'P1', 'S1', 'S2'}
    E = [('P1','T1'), ('T1','S1'), ('T1','S2'), ('T1','B1'), ('B1','N1')]
    
    print("STEP 1: Topological Metrics")
    print("-" * 80)
    print()
    
    # Manually calculate degree centrality
    print("a) Degree Centrality (for directed graph):")
    print("   Formula: C_deg(v) = (deg_in(v) + deg_out(v)) / (2(|V|-1))")
    print(f"   |V| = {len(V)}, so denominator = 2(6-1) = 10")
    print()
    
    degrees = {
        'P1': (0, 1, 1/10),
        'T1': (1, 3, 4/10),
        'S1': (1, 0, 1/10),
        'S2': (1, 0, 1/10),
        'B1': (1, 1, 2/10),
        'N1': (1, 0, 1/10)
    }
    
    for v, (deg_in, deg_out, score) in degrees.items():
        print(f"   {v}: deg_in={deg_in}, deg_out={deg_out} → C_deg = {score:.2f}")
    print()
    
    print("b) Betweenness Centrality:")
    print("   Formula: C_btw(v) = Σ(σ_st(v)/σ_st) / ((|V|-1)(|V|-2))")
    print("   Denominator = (6-1)(6-2) = 20")
    print()
    
    print("   Shortest paths analysis:")
    print("   • Paths from P1 to S1: P1→T1→S1 (T1 is on this path)")
    print("   • Paths from P1 to S2: P1→T1→S2 (T1 is on this path)")
    print("   • Paths from P1 to B1: P1→T1→B1 (T1 is on this path)")
    print("   • Paths from P1 to N1: P1→T1→B1→N1 (T1, B1 on this path)")
    print("   • Paths from T1 to N1: T1→B1→N1 (B1 is on this path)")
    print("   • etc...")
    print()
    
    btw = {
        'P1': 0.0,
        'T1': 0.6,  # Many paths go through T1
        'S1': 0.0,
        'S2': 0.0,
        'B1': 0.3,  # Some paths go through B1
        'N1': 0.0
    }
    
    for v, score in btw.items():
        print(f"   {v}: C_btw = {score:.2f}")
    print()
    
    print("c) Articulation Points:")
    print("   An articulation point is a vertex whose removal disconnects the graph.")
    print()
    print("   Testing each vertex:")
    print("   • Remove P1: {T1,S1,S2,B1,N1} still connected → NOT articulation point")
    print("   • Remove T1: {P1},{S1},{S2},{B1,N1} → DISCONNECTED → IS articulation point")
    print("   • Remove S1: {P1,T1,S2,B1,N1} still connected → NOT articulation point")
    print("   • Remove S2: {P1,T1,S1,B1,N1} still connected → NOT articulation point")
    print("   • Remove B1: {P1,T1,S1,S2},{N1} → DISCONNECTED → IS articulation point")
    print("   • Remove N1: {P1,T1,S1,S2,B1} still connected → NOT articulation point")
    print()
    print("   Articulation points A = {T1, B1}")
    print()
    
    print("d) Composite Topology Score:")
    print("   S_topo(v) = 0.35·C_btw + 0.25·C_deg + ... (simplified)")
    print("   With 1.5× multiplier for articulation points")
    print()
    
    s_topo = {
        'P1': 0.10,
        'T1': 0.65 * 1.5,  # Articulation point bonus
        'S1': 0.10,
        'S2': 0.10,
        'B1': 0.40 * 1.5,  # Articulation point bonus
        'N1': 0.15
    }
    
    for v, score in s_topo.items():
        marker = " ★ (articulation point)" if v in ['T1', 'B1'] else ""
        print(f"   {v}: S_topo = {score:.2f}{marker}")
    print()
    
    print("-" * 80)
    print()
    print("STEP 2: QoS Importance (for T1 only)")
    print("-" * 80)
    print()
    print("   Topic T1 QoS profile:")
    print("   • Latency requirement: λ = 50 ms")
    print("   • Throughput requirement: τ = 5000 msgs/sec")
    print("   • Reliability: ρ = 0.99")
    print("   • Priority: π = 5")
    print()
    print("   Calculating importance:")
    print("   f_λ(50) = 1/(1+50/100) = 1/1.5 = 0.67")
    print("   f_τ(5000) = min(5000/10000, 1) = 0.50")
    print("   π_norm = 5/5 = 1.00")
    print()
    print("   I_qos(T1) = 0.3×0.67 + 0.2×0.50 + 0.3×0.99 + 0.2×1.00")
    print("             = 0.20 + 0.10 + 0.30 + 0.20")
    print("             = 0.80")
    print()
    print("   For non-topic components: I_qos = 0")
    print()
    
    i_qos = {'P1': 0, 'T1': 0.80, 'S1': 0, 'S2': 0, 'B1': 0, 'N1': 0}
    
    print("-" * 80)
    print()
    print("STEP 3: Impact Score (Reachability Loss)")
    print("-" * 80)
    print()
    
    print("   Original reachability matrix:")
    print("      | P1 T1 S1 S2 B1 N1")
    print("   ---+-------------------")
    print("   P1 | 1  1  1  1  1  1")
    print("   T1 | 0  1  1  1  1  1")
    print("   S1 | 0  0  1  0  0  0")
    print("   S2 | 0  0  0  1  0  0")
    print("   B1 | 0  0  0  0  1  1")
    print("   N1 | 0  0  0  0  0  1")
    print()
    print("   R_total(G) = 6+5+1+1+2+1 = 16")
    print()
    
    print("   Testing impact of removing T1:")
    print("   After removing T1, P1 cannot reach S1, S2, B1, N1")
    print("   New reachable pairs = 6-4 = 12")
    print("   I_impact(T1) = (16-12)/16 = 4/16 = 0.25")
    print()
    
    print("   Testing impact of removing B1:")
    print("   After removing B1, we lose: T1→B1, P1→B1, and paths to N1")
    print("   New reachable pairs ≈ 13")
    print("   I_impact(B1) = (16-13)/16 = 3/16 = 0.19")
    print()
    
    i_impact = {
        'P1': 0.31,  # Loses all downstream
        'T1': 0.25,
        'S1': 0.0,
        'S2': 0.0,
        'B1': 0.19,
        'N1': 0.0
    }
    
    for v, score in i_impact.items():
        print(f"   I_impact({v}) = {score:.2f}")
    print()
    
    print("-" * 80)
    print()
    print("STEP 4: Composite Criticality Score")
    print("-" * 80)
    print()
    
    print("   Component-specific weights:")
    print("   • Topic: α=0.4 (topology), β=0.6 (QoS)")
    print("   • Broker: α=0.8 (topology), β=0.2 (QoS)")
    print("   • Others: α=0.7 (topology), β=0.3 (QoS)")
    print()
    
    weights = {
        'P1': (0.7, 0.3), 'T1': (0.4, 0.6), 'S1': (0.7, 0.3),
        'S2': (0.7, 0.3), 'B1': (0.8, 0.2), 'N1': (0.9, 0.1)
    }
    
    print("   Composite scores:")
    for v in V:
        alpha, beta = weights[v]
        s_comp = alpha * s_topo[v] + beta * i_qos[v]
        print(f"   S_comp({v}) = {alpha}×{s_topo[v]:.2f} + {beta}×{i_qos[v]:.2f} = {s_comp:.2f}")
    print()
    
    print("-" * 80)
    print()
    print("STEP 5: Final Integrated Score")
    print("-" * 80)
    print()
    print("   S_final(v) = 0.5×S_comp(v) + 0.5×I_impact(v)")
    print()
    
    s_final = {}
    for v in V:
        alpha, beta = weights[v]
        s_comp = alpha * s_topo[v] + beta * i_qos[v]
        s_final[v] = 0.5 * s_comp + 0.5 * i_impact[v]
        print(f"   S_final({v}) = 0.5×{s_comp:.2f} + 0.5×{i_impact[v]:.2f} = {s_final[v]:.2f}")
    print()
    
    print("-" * 80)
    print()
    print("STEP 6: Anti-Pattern Detection")
    print("-" * 80)
    print()
    
    print("a) Single Points of Failure:")
    print("   SPOF = {v ∈ A : I_impact(v) > 0.15}")
    print(f"   A = {{T1, B1}} (articulation points)")
    print(f"   I_impact(T1) = {i_impact['T1']:.2f} > 0.15 ✓")
    print(f"   I_impact(B1) = {i_impact['B1']:.2f} > 0.15 ✓")
    print("   → SPOF detected: {T1, B1}")
    print()
    
    print("b) God Topics:")
    print("   Connections to T1: P1 (pub) + S1,S2 (sub) = 3 connections")
    print("   3 < 5 (threshold) → No god topic detected")
    print()
    
    print("c) Circular Dependencies:")
    print("   No cycles in this simple graph")
    print()
    
    print("d) Hidden Coupling:")
    print("   S1 and S2 both subscribe to T1 (1 shared topic)")
    print("   1 < 3 (threshold) → No hidden coupling detected")
    print()
    
    print("="*80)
    print()
    print("FINAL RANKING BY CRITICALITY:")
    print("-" * 80)
    sorted_components = sorted(s_final.items(), key=lambda x: x[1], reverse=True)
    for rank, (v, score) in enumerate(sorted_components, 1):
        marker = " [SPOF]" if v in ['T1', 'B1'] else ""
        print(f"{rank}. {v}: {score:.2f}{marker}")
    print()
    print("Conclusion: T1 (topic) is the most critical component, followed by B1 (broker).")
    print("Both are single points of failure and should be prioritized for redundancy.")
    print("="*80)
    print()


# Call demonstration functions at module level
print_mathematical_framework()
demonstrate_toy_example()

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class QoSProfile:
    """Quality of Service profile for topics"""
    latency_requirement: float  # ms
    throughput_requirement: float  # msgs/sec
    reliability: float  # 0-1
    priority: int  # 1-5
    
    def importance_score(self) -> float:
        """Calculate topic importance from QoS parameters"""
        # Normalize and weight QoS factors
        norm_latency = 1.0 / (1.0 + self.latency_requirement / 100)
        norm_throughput = min(self.throughput_requirement / 10000, 1.0)
        norm_priority = self.priority / 5.0
        
        return (0.3 * norm_latency + 
                0.2 * norm_throughput + 
                0.3 * self.reliability + 
                0.2 * norm_priority)


@dataclass
class Component:
    """Base component in pub-sub system"""
    id: str
    type: str  # 'topic', 'application', 'broker', 'node'
    metadata: Dict = None


@dataclass
class DesignDefect:
    """Represents a design defect or anti-pattern"""
    defect_type: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    component: str
    description: str
    metrics: Dict
    recommendations: List[str]


# ============================================================================
# MULTI-LAYER GRAPH MODEL
# ============================================================================

class PubSubGraphModel:
    """Multi-layered graph model for pub-sub systems"""
    
    def __init__(self):
        # Separate layers
        self.physical_layer = nx.DiGraph()  # Nodes and network
        self.broker_layer = nx.DiGraph()    # Broker topology
        self.logical_layer = nx.DiGraph()   # Topics and subscriptions
        self.application_layer = nx.DiGraph()  # Applications
        
        # Composite graph
        self.composite_graph = nx.DiGraph()
        
        # Metadata
        self.components = {}
        self.topic_qos = {}
        
    def add_physical_node(self, node_id: str, metadata: Dict = None):
        """Add physical infrastructure node"""
        self.physical_layer.add_node(node_id, layer='physical', **metadata or {})
        self.composite_graph.add_node(node_id, layer='physical', type='node')
        self.components[node_id] = Component(node_id, 'node', metadata)
        
    def add_broker(self, broker_id: str, node_id: str, metadata: Dict = None):
        """Add broker on a physical node"""
        self.broker_layer.add_node(broker_id, layer='broker', **metadata or {})
        self.composite_graph.add_node(broker_id, layer='broker', type='broker')
        self.composite_graph.add_edge(broker_id, node_id, relation='hosted_on')
        self.components[broker_id] = Component(broker_id, 'broker', metadata)
        
    def add_topic(self, topic_id: str, broker_id: str, qos: QoSProfile):
        """Add topic managed by broker"""
        self.logical_layer.add_node(topic_id, layer='logical', 
                                    importance=qos.importance_score())
        self.composite_graph.add_node(topic_id, layer='logical', type='topic',
                                     importance=qos.importance_score())
        self.composite_graph.add_edge(topic_id, broker_id, relation='managed_by')
        self.topic_qos[topic_id] = qos
        self.components[topic_id] = Component(topic_id, 'topic', {'qos': qos})
        
    def add_application(self, app_id: str, app_type: str, metadata: Dict = None):
        """Add publisher or subscriber application"""
        self.application_layer.add_node(app_id, layer='application', 
                                       app_type=app_type, **metadata or {})
        self.composite_graph.add_node(app_id, layer='application', 
                                     type='application', app_type=app_type)
        self.components[app_id] = Component(app_id, 'application', metadata)
        
    def add_publish_relation(self, publisher_id: str, topic_id: str, rate: float = 1.0):
        """Publisher publishes to topic"""
        self.application_layer.add_edge(publisher_id, topic_id, 
                                       relation='publishes', rate=rate)
        self.composite_graph.add_edge(publisher_id, topic_id, 
                                     relation='publishes', rate=rate)
        
    def add_subscribe_relation(self, topic_id: str, subscriber_id: str):
        """Subscriber subscribes to topic"""
        self.application_layer.add_edge(topic_id, subscriber_id, 
                                       relation='subscribes')
        self.composite_graph.add_edge(topic_id, subscriber_id, 
                                     relation='subscribes')
        
    def add_broker_link(self, broker1: str, broker2: str, bidirectional: bool = True):
        """Add connection between brokers (replication/federation)"""
        self.broker_layer.add_edge(broker1, broker2, relation='replicates')
        self.composite_graph.add_edge(broker1, broker2, relation='replicates')
        if bidirectional:
            self.broker_layer.add_edge(broker2, broker1, relation='replicates')
            self.composite_graph.add_edge(broker2, broker1, relation='replicates')


# ============================================================================
# TOPOLOGICAL METRICS
# ============================================================================

class TopologicalAnalyzer:
    """Compute centrality metrics and structural properties"""
    
    @staticmethod
    def compute_centrality_metrics(G: nx.Graph) -> pd.DataFrame:
        """Compute multiple centrality metrics"""
        metrics = {}
        
        # Degree centrality
        metrics['degree_centrality'] = nx.degree_centrality(G)
        
        # Betweenness centrality
        metrics['betweenness_centrality'] = nx.betweenness_centrality(G)
        
        # Closeness centrality
        try:
            metrics['closeness_centrality'] = nx.closeness_centrality(G)
        except:
            metrics['closeness_centrality'] = {n: 0 for n in G.nodes()}
        
        # Eigenvector centrality (if graph is strongly connected)
        try:
            metrics['eigenvector_centrality'] = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            metrics['eigenvector_centrality'] = {n: 0 for n in G.nodes()}
        
        # PageRank (works for directed graphs)
        metrics['pagerank'] = nx.pagerank(G)
        
        # Convert to DataFrame
        df = pd.DataFrame(metrics).fillna(0)
        
        # Normalize each metric to 0-1
        for col in df.columns:
            max_val = df[col].max()
            if max_val > 0:
                df[col] = df[col] / max_val
                
        return df
    
    @staticmethod
    def find_articulation_points(G: nx.Graph) -> Set[str]:
        """Find articulation points (cut vertices)"""
        # Convert to undirected for articulation point analysis
        G_undirected = G.to_undirected()
        return set(nx.articulation_points(G_undirected))
    
    @staticmethod
    def find_bridges(G: nx.Graph) -> Set[Tuple[str, str]]:
        """Find bridges (cut edges)"""
        G_undirected = G.to_undirected()
        return set(nx.bridges(G_undirected))


# ============================================================================
# CRITICALITY SCORING
# ============================================================================

class CriticalityScorer:
    """Compute composite criticality scores"""
    
    def __init__(self, model: PubSubGraphModel):
        self.model = model
        self.topology_metrics = None
        self.articulation_points = None
        
    def compute_topology_score(self) -> Dict[str, float]:
        """Compute topology-based criticality"""
        analyzer = TopologicalAnalyzer()
        
        # Get centrality metrics
        self.topology_metrics = analyzer.compute_centrality_metrics(
            self.model.composite_graph
        )
        
        # Find articulation points
        self.articulation_points = analyzer.find_articulation_points(
            self.model.composite_graph
        )
        
        # Weighted combination of centrality metrics
        weights = {
            'betweenness_centrality': 0.35,
            'degree_centrality': 0.25,
            'pagerank': 0.20,
            'closeness_centrality': 0.15,
            'eigenvector_centrality': 0.05
        }
        
        topology_scores = {}
        for node in self.model.composite_graph.nodes():
            score = sum(
                self.topology_metrics.loc[node, metric] * weight
                for metric, weight in weights.items()
            )
            
            # Bonus for articulation points
            if node in self.articulation_points:
                score *= 1.5
                
            topology_scores[node] = min(score, 1.0)
            
        return topology_scores
    
    def compute_qos_importance(self) -> Dict[str, float]:
        """Get QoS-based importance for topics"""
        importance = {}
        for node in self.model.composite_graph.nodes():
            if node in self.model.topic_qos:
                importance[node] = self.model.topic_qos[node].importance_score()
            else:
                importance[node] = 0.0
        return importance
    
    def compute_composite_score(self) -> pd.DataFrame:
        """Compute final composite criticality score"""
        # Get individual scores
        topology_scores = self.compute_topology_score()
        qos_scores = self.compute_qos_importance()
        
        # Component-specific weights
        component_weights = {
            'topic': {'topology': 0.4, 'qos': 0.6},
            'broker': {'topology': 0.8, 'qos': 0.2},
            'node': {'topology': 0.9, 'qos': 0.1},
            'application': {'topology': 0.7, 'qos': 0.3}
        }
        
        results = []
        for node in self.model.composite_graph.nodes():
            node_data = self.model.composite_graph.nodes[node]
            component_type = node_data.get('type', 'unknown')
            
            weights = component_weights.get(component_type, {'topology': 0.5, 'qos': 0.5})
            
            composite_score = (
                weights['topology'] * topology_scores.get(node, 0) +
                weights['qos'] * qos_scores.get(node, 0)
            )
            
            results.append({
                'component': node,
                'type': component_type,
                'topology_score': topology_scores.get(node, 0),
                'qos_score': qos_scores.get(node, 0),
                'composite_score': composite_score,
                'is_articulation_point': node in self.articulation_points
            })
        
        df = pd.DataFrame(results)
        return df.sort_values('composite_score', ascending=False)


# ============================================================================
# IMPACT ANALYSIS (REACHABILITY LOSS)
# ============================================================================

class ImpactAnalyzer:
    """Analyze impact through reachability loss"""
    
    def __init__(self, model: PubSubGraphModel):
        self.model = model
        
    def compute_reachability_matrix(self) -> Dict[Tuple[str, str], bool]:
        """Compute which nodes can reach which other nodes"""
        G = self.model.composite_graph
        reachability = {}
        
        for source in G.nodes():
            reachable = nx.descendants(G, source)
            reachable.add(source)
            for target in G.nodes():
                reachability[(source, target)] = target in reachable
                
        return reachability
    
    def compute_impact_score(self, node: str) -> float:
        """Compute impact of removing a node (reachability loss)"""
        G = self.model.composite_graph
        
        # Original reachability
        original_reachability = self.compute_reachability_matrix()
        total_pairs = len(G.nodes()) * (len(G.nodes()) - 1)
        original_reachable = sum(original_reachability.values())
        
        # Create graph without the node
        G_reduced = G.copy()
        G_reduced.remove_node(node)
        
        # New reachability
        new_reachability = {}
        for source in G_reduced.nodes():
            reachable = nx.descendants(G_reduced, source)
            reachable.add(source)
            for target in G_reduced.nodes():
                new_reachability[(source, target)] = target in reachable
        
        new_reachable = sum(new_reachability.values())
        
        # Impact = fraction of reachability lost
        if original_reachable > 0:
            impact = (original_reachable - new_reachable) / original_reachable
        else:
            impact = 0.0
            
        return impact
    
    def compute_all_impacts(self) -> pd.DataFrame:
        """Compute impact scores for all components"""
        results = []
        
        for node in self.model.composite_graph.nodes():
            node_data = self.model.composite_graph.nodes[node]
            impact = self.compute_impact_score(node)
            
            results.append({
                'component': node,
                'type': node_data.get('type', 'unknown'),
                'impact_score': impact
            })
        
        df = pd.DataFrame(results)
        return df.sort_values('impact_score', ascending=False)


# ============================================================================
# CRITICAL PATH ANALYSIS
# ============================================================================

class CriticalPathAnalyzer:
    """Identify critical paths in the system"""
    
    def __init__(self, model: PubSubGraphModel):
        self.model = model
        
    def find_publisher_to_subscriber_paths(self) -> List[Dict]:
        """Find all paths from publishers to subscribers"""
        G = self.model.composite_graph
        paths = []
        
        # Identify publishers and subscribers
        publishers = [n for n in G.nodes() 
                     if G.nodes[n].get('app_type') == 'publisher']
        subscribers = [n for n in G.nodes() 
                      if G.nodes[n].get('app_type') == 'subscriber']
        
        # Find all simple paths
        for pub in publishers:
            for sub in subscribers:
                try:
                    all_paths = list(nx.all_simple_paths(G, pub, sub, cutoff=10))
                    for path in all_paths:
                        paths.append({
                            'publisher': pub,
                            'subscriber': sub,
                            'path': path,
                            'length': len(path)
                        })
                except nx.NetworkXNoPath:
                    continue
                    
        return paths
    
    def score_path_criticality(self, path: List[str], 
                               component_scores: Dict[str, float]) -> float:
        """Score path criticality based on component scores"""
        # Use minimum score along path (weakest link)
        min_score = min(component_scores.get(node, 0) for node in path)
        # Use average score
        avg_score = np.mean([component_scores.get(node, 0) for node in path])
        
        # Combined metric
        return 0.6 * min_score + 0.4 * avg_score
    
    def identify_critical_paths(self, component_scores: Dict[str, float], 
                                top_k: int = 10) -> pd.DataFrame:
        """Identify most critical paths"""
        paths = self.find_publisher_to_subscriber_paths()
        
        results = []
        for path_info in paths:
            path = path_info['path']
            criticality = self.score_path_criticality(path, component_scores)
            
            results.append({
                'publisher': path_info['publisher'],
                'subscriber': path_info['subscriber'],
                'path': ' -> '.join(path),
                'path_length': len(path),
                'criticality_score': criticality,
                'components': path
            })
        
        df = pd.DataFrame(results)
        return df.sort_values('criticality_score', ascending=False).head(top_k)


# ============================================================================
# ANTI-PATTERN DETECTION
# ============================================================================

class AntiPatternDetector:
    """Detect design defects and anti-patterns"""
    
    def __init__(self, model: PubSubGraphModel):
        self.model = model
        self.defects = []
        
    def detect_single_points_of_failure(self, impact_scores: Dict[str, float]) -> List[DesignDefect]:
        """Detect Single Points of Failure (SPOF)"""
        spof_defects = []
        G = self.model.composite_graph
        
        # Find articulation points
        analyzer = TopologicalAnalyzer()
        articulation_points = analyzer.find_articulation_points(G)
        
        # Check each articulation point
        for node in articulation_points:
            node_data = G.nodes[node]
            component_type = node_data.get('type', 'unknown')
            impact = impact_scores.get(node, 0)
            
            # Determine severity based on impact and component type
            if impact > 0.5:
                severity = 'critical'
            elif impact > 0.3:
                severity = 'high'
            elif impact > 0.15:
                severity = 'medium'
            else:
                severity = 'low'
            
            # Check if there's redundancy
            predecessors = list(G.predecessors(node))
            successors = list(G.successors(node))
            
            defect = DesignDefect(
                defect_type='SPOF',
                severity=severity,
                component=node,
                description=f"{component_type.upper()} '{node}' is a single point of failure. "
                           f"Its removal disconnects {int(impact*100)}% of the system.",
                metrics={
                    'impact_score': impact,
                    'predecessors_count': len(predecessors),
                    'successors_count': len(successors),
                    'is_articulation_point': True
                },
                recommendations=[
                    f"Add redundancy for {component_type} '{node}'",
                    "Implement failover mechanisms",
                    "Consider active-active or active-passive replication",
                    "Add health checks and automatic recovery"
                ]
            )
            spof_defects.append(defect)
        
        # Also check critical brokers/nodes without redundancy
        for node in G.nodes():
            node_data = G.nodes[node]
            if node_data.get('type') in ['broker', 'node']:
                # Check if there are alternative paths around this component
                impact = impact_scores.get(node, 0)
                if impact > 0.3 and node not in articulation_points:
                    # High impact but not formally an articulation point
                    # Still potentially problematic
                    defect = DesignDefect(
                        defect_type='SPOF',
                        severity='medium',
                        component=node,
                        description=f"{node_data.get('type').upper()} '{node}' has high impact "
                                   f"({int(impact*100)}%) but limited redundancy.",
                        metrics={'impact_score': impact, 'is_articulation_point': False},
                        recommendations=[
                            f"Consider adding redundancy for '{node}'",
                            "Evaluate backup strategies"
                        ]
                    )
                    spof_defects.append(defect)
        
        return spof_defects
    
    def detect_god_topics(self, threshold_connections: int = 5) -> List[DesignDefect]:
        """Detect God Topics (topics with excessive connections)"""
        god_topics = []
        G = self.model.composite_graph
        
        # Analyze topics
        for node in G.nodes():
            node_data = G.nodes[node]
            if node_data.get('type') == 'topic':
                # Count publishers and subscribers
                publishers = [p for p in G.predecessors(node) 
                            if G.nodes[p].get('app_type') == 'publisher']
                subscribers = [s for s in G.successors(node) 
                             if G.nodes[s].get('app_type') == 'subscriber']
                
                total_connections = len(publishers) + len(subscribers)
                
                if total_connections >= threshold_connections:
                    # Calculate coupling index
                    coupling_index = total_connections / threshold_connections
                    
                    # Determine severity
                    if coupling_index >= 3:
                        severity = 'critical'
                    elif coupling_index >= 2:
                        severity = 'high'
                    elif coupling_index >= 1.5:
                        severity = 'medium'
                    else:
                        severity = 'low'
                    
                    defect = DesignDefect(
                        defect_type='God Topic',
                        severity=severity,
                        component=node,
                        description=f"Topic '{node}' has {len(publishers)} publishers and "
                                   f"{len(subscribers)} subscribers (total: {total_connections}). "
                                   f"This creates a central bottleneck and high coupling.",
                        metrics={
                            'publishers_count': len(publishers),
                            'subscribers_count': len(subscribers),
                            'total_connections': total_connections,
                            'coupling_index': coupling_index
                        },
                        recommendations=[
                            "Split topic into multiple specialized topics",
                            "Apply topic partitioning or sharding",
                            "Use hierarchical topic organization",
                            "Consider domain-driven design principles",
                            "Implement topic filtering at broker level"
                        ]
                    )
                    god_topics.append(defect)
        
        return god_topics
    
    def detect_circular_dependencies(self) -> List[DesignDefect]:
        """Detect circular dependencies in the system"""
        circular_deps = []
        G = self.model.composite_graph
        
        # Find all simple cycles
        try:
            cycles = list(nx.simple_cycles(G))
        except:
            cycles = []
        
        # Filter cycles that involve applications and topics
        relevant_cycles = []
        for cycle in cycles:
            # Check if cycle involves applications and topics
            has_app = any(G.nodes[n].get('type') == 'application' for n in cycle)
            has_topic = any(G.nodes[n].get('type') == 'topic' for n in cycle)
            
            if has_app and has_topic and len(cycle) > 2:
                relevant_cycles.append(cycle)
        
        # Group cycles by components involved
        cycle_groups = defaultdict(list)
        for cycle in relevant_cycles:
            key = tuple(sorted(cycle))
            cycle_groups[key].append(cycle)
        
        # Create defects for each unique cycle
        for cycle in cycle_groups.keys():
            cycle_list = list(cycle)
            cycle_length = len(cycle_list)
            
            # Determine severity based on cycle length and components
            if cycle_length <= 3:
                severity = 'critical'
            elif cycle_length <= 5:
                severity = 'high'
            else:
                severity = 'medium'
            
            # Identify applications in cycle
            apps_in_cycle = [n for n in cycle_list 
                           if G.nodes[n].get('type') == 'application']
            topics_in_cycle = [n for n in cycle_list 
                             if G.nodes[n].get('type') == 'topic']
            
            defect = DesignDefect(
                defect_type='Circular Dependency',
                severity=severity,
                component=' -> '.join(cycle_list),
                description=f"Circular dependency detected: {' -> '.join(cycle_list)} -> {cycle_list[0]}. "
                           f"This can cause cascading failures and deadlocks.",
                metrics={
                    'cycle_length': cycle_length,
                    'applications_involved': len(apps_in_cycle),
                    'topics_involved': len(topics_in_cycle),
                    'cycle_path': cycle_list
                },
                recommendations=[
                    "Break the circular dependency by introducing event aggregation",
                    "Use request-response pattern instead of pub-sub for some interactions",
                    "Implement circuit breakers to prevent cascading failures",
                    "Consider saga pattern for distributed transactions",
                    "Add dead letter topics for failed message processing"
                ]
            )
            circular_deps.append(defect)
        
        return circular_deps
    
    def detect_hidden_coupling(self, threshold_shared: int = 3) -> List[DesignDefect]:
        """Detect hidden coupling through shared dependencies"""
        hidden_couplings = []
        G = self.model.composite_graph
        
        # Find applications
        applications = [n for n in G.nodes() 
                       if G.nodes[n].get('type') == 'application']
        
        # For each pair of applications, check shared dependencies
        for i, app1 in enumerate(applications):
            for app2 in applications[i+1:]:
                # Find shared topics
                app1_topics = set()
                app2_topics = set()
                
                # Topics app1 publishes to or subscribes from
                for neighbor in list(G.predecessors(app1)) + list(G.successors(app1)):
                    if G.nodes[neighbor].get('type') == 'topic':
                        app1_topics.add(neighbor)
                
                # Topics app2 publishes to or subscribes from
                for neighbor in list(G.predecessors(app2)) + list(G.successors(app2)):
                    if G.nodes[neighbor].get('type') == 'topic':
                        app2_topics.add(neighbor)
                
                shared_topics = app1_topics & app2_topics
                
                if len(shared_topics) >= threshold_shared:
                    # Check for shared brokers
                    shared_brokers = set()
                    for topic in shared_topics:
                        for broker in G.successors(topic):
                            if G.nodes[broker].get('type') == 'broker':
                                shared_brokers.add(broker)
                    
                    # Determine severity
                    coupling_degree = len(shared_topics) / threshold_shared
                    if coupling_degree >= 2:
                        severity = 'high'
                    elif coupling_degree >= 1.5:
                        severity = 'medium'
                    else:
                        severity = 'low'
                    
                    defect = DesignDefect(
                        defect_type='Hidden Coupling',
                        severity=severity,
                        component=f"{app1} <-> {app2}",
                        description=f"Applications '{app1}' and '{app2}' share {len(shared_topics)} "
                                   f"topics and {len(shared_brokers)} brokers. "
                                   f"This creates implicit coupling not visible in direct dependencies.",
                        metrics={
                            'shared_topics': list(shared_topics),
                            'shared_brokers': list(shared_brokers),
                            'coupling_degree': len(shared_topics),
                            'app1': app1,
                            'app2': app2
                        },
                        recommendations=[
                            "Reduce shared dependencies where possible",
                            "Make coupling explicit through well-defined interfaces",
                            "Consider separate topic namespaces for different domains",
                            "Document shared dependencies in system architecture",
                            "Implement bulkheads to isolate failures"
                        ]
                    )
                    hidden_couplings.append(defect)
        
        return hidden_couplings
    
    def detect_all_anti_patterns(self, impact_scores: Dict[str, float]) -> pd.DataFrame:
        """Detect all anti-patterns and return summary"""
        print("Detecting anti-patterns...")
        
        # Detect each type
        spof = self.detect_single_points_of_failure(impact_scores)
        god_topics = self.detect_god_topics()
        circular = self.detect_circular_dependencies()
        hidden = self.detect_hidden_coupling()
        
        # Combine all defects
        all_defects = spof + god_topics + circular + hidden
        self.defects = all_defects
        
        # Create summary DataFrame
        if not all_defects:
            return pd.DataFrame()
        
        summary = []
        for defect in all_defects:
            summary.append({
                'defect_type': defect.defect_type,
                'severity': defect.severity,
                'component': defect.component,
                'description': defect.description
            })
        
        df = pd.DataFrame(summary)
        
        # Sort by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        df['severity_rank'] = df['severity'].map(severity_order)
        df = df.sort_values('severity_rank').drop('severity_rank', axis=1)
        
        return df


# ============================================================================
# INTEGRATED ANALYSIS
# ============================================================================

class IntegratedAnalysis:
    """Combine all analysis methods"""
    
    def __init__(self, model: PubSubGraphModel):
        self.model = model
        self.criticality_scorer = CriticalityScorer(model)
        self.impact_analyzer = ImpactAnalyzer(model)
        self.path_analyzer = CriticalPathAnalyzer(model)
        self.anti_pattern_detector = AntiPatternDetector(model)
        
    def run_complete_analysis(self) -> Dict[str, pd.DataFrame]:
        """Run all analyses and combine results"""
        print("Computing composite criticality scores...")
        criticality_df = self.criticality_scorer.compute_composite_score()
        
        print("Computing impact scores (reachability loss)...")
        impact_df = self.impact_analyzer.compute_all_impacts()
        
        print("Analyzing critical paths...")
        component_scores = dict(zip(criticality_df['component'], 
                                   criticality_df['composite_score']))
        critical_paths_df = self.path_analyzer.identify_critical_paths(component_scores)
        
        # Merge criticality and impact
        combined_df = criticality_df.merge(
            impact_df[['component', 'impact_score']], 
            on='component',
            how='left'
        )
        
        # Calculate final integrated score
        combined_df['integrated_score'] = (
            0.5 * combined_df['composite_score'] + 
            0.5 * combined_df['impact_score']
        )
        
        combined_df = combined_df.sort_values('integrated_score', ascending=False)
        
        # Detect anti-patterns
        print("Detecting design defects and anti-patterns...")
        impact_scores = dict(zip(impact_df['component'], impact_df['impact_score']))
        anti_patterns_df = self.anti_pattern_detector.detect_all_anti_patterns(impact_scores)
        
        return {
            'component_analysis': combined_df,
            'critical_paths': critical_paths_df,
            'topology_metrics': self.criticality_scorer.topology_metrics,
            'anti_patterns': anti_patterns_df,
            'defects': self.anti_pattern_detector.defects
        }


# ============================================================================
# VISUALIZATION
# ============================================================================

class Visualizer:
    """Visualization utilities"""
    
    @staticmethod
    def plot_component_scores(analysis_results: pd.DataFrame, top_n: int = 15):
        """Plot component criticality scores"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        top_components = analysis_results.head(top_n)
        
        # Topology score
        axes[0].barh(top_components['component'], 
                    top_components['topology_score'],
                    color='steelblue')
        axes[0].set_xlabel('Topology Score')
        axes[0].set_title('Topological Criticality')
        axes[0].invert_yaxis()
        
        # QoS importance
        axes[1].barh(top_components['component'], 
                    top_components['qos_score'],
                    color='coral')
        axes[1].set_xlabel('QoS Score')
        axes[1].set_title('QoS-based Importance')
        axes[1].invert_yaxis()
        
        # Impact score
        axes[2].barh(top_components['component'], 
                    top_components['impact_score'],
                    color='darkred')
        axes[2].set_xlabel('Impact Score')
        axes[2].set_title('Reachability Impact')
        axes[2].invert_yaxis()
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_integrated_scores(analysis_results: pd.DataFrame, top_n: int = 15):
        """Plot integrated criticality scores"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        top_components = analysis_results.head(top_n)
        
        x = np.arange(len(top_components))
        width = 0.35
        
        ax.barh(x - width/2, top_components['composite_score'], 
               width, label='Composite Score', color='steelblue', alpha=0.8)
        ax.barh(x + width/2, top_components['integrated_score'], 
               width, label='Integrated Score', color='darkgreen', alpha=0.8)
        
        ax.set_ylabel('Component')
        ax.set_xlabel('Score')
        ax.set_title(f'Top {top_n} Critical Components')
        ax.set_yticks(x)
        ax.set_yticklabels(top_components['component'])
        ax.legend()
        ax.invert_yaxis()
        
        # Highlight articulation points
        for i, (idx, row) in enumerate(top_components.iterrows()):
            if row['is_articulation_point']:
                ax.plot(1.05, i, 'r*', markersize=15)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_anti_pattern_summary(anti_patterns_df: pd.DataFrame, defects: List[DesignDefect]):
        """Visualize anti-pattern detection results"""
        if anti_patterns_df.empty:
            print("No anti-patterns detected!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Count by type
        type_counts = anti_patterns_df['defect_type'].value_counts()
        axes[0, 0].bar(type_counts.index, type_counts.values, color='coral')
        axes[0, 0].set_title('Anti-Patterns by Type', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Count by severity
        severity_counts = anti_patterns_df['severity'].value_counts()
        severity_order = ['critical', 'high', 'medium', 'low']
        severity_counts = severity_counts.reindex([s for s in severity_order if s in severity_counts.index])
        colors = {'critical': 'darkred', 'high': 'orangered', 'medium': 'orange', 'low': 'gold'}
        bar_colors = [colors.get(s, 'gray') for s in severity_counts.index]
        axes[0, 1].bar(severity_counts.index, severity_counts.values, color=bar_colors)
        axes[0, 1].set_title('Anti-Patterns by Severity', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Count')
        
        # Severity distribution by type
        pivot = anti_patterns_df.groupby(['defect_type', 'severity']).size().unstack(fill_value=0)
        pivot = pivot.reindex(columns=[s for s in severity_order if s in pivot.columns], fill_value=0)
        pivot.plot(kind='barh', stacked=True, ax=axes[1, 0], 
                  color=[colors.get(s, 'gray') for s in pivot.columns])
        axes[1, 0].set_title('Severity Distribution by Type', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Count')
        axes[1, 0].legend(title='Severity', loc='best')
        
        # Impact metrics for SPOF
        spof_defects = [d for d in defects if d.defect_type == 'SPOF']
        if spof_defects:
            components = [d.component for d in spof_defects[:10]]
            impacts = [d.metrics.get('impact_score', 0) for d in spof_defects[:10]]
            axes[1, 1].barh(components, impacts, color='darkred')
            axes[1, 1].set_title('SPOF Impact Scores (Top 10)', fontsize=14, fontweight='bold')
            axes[1, 1].set_xlabel('Impact Score')
            axes[1, 1].invert_yaxis()
        else:
            axes[1, 1].text(0.5, 0.5, 'No SPOF detected', 
                          ha='center', va='center', fontsize=12)
            axes[1, 1].set_title('SPOF Impact Scores', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_graph_with_defects(model: PubSubGraphModel, scores: Dict[str, float], 
                                defects: List[DesignDefect]):
        """Visualize graph highlighting defects"""
        G = model.composite_graph
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        
        # Identify defect components
        spof_components = set()
        god_topics = set()
        circular_components = set()
        hidden_coupling_apps = set()
        
        for defect in defects:
            if defect.defect_type == 'SPOF':
                spof_components.add(defect.component)
            elif defect.defect_type == 'God Topic':
                god_topics.add(defect.component)
            elif defect.defect_type == 'Circular Dependency':
                cycle = defect.metrics.get('cycle_path', [])
                circular_components.update(cycle)
            elif defect.defect_type == 'Hidden Coupling':
                hidden_coupling_apps.add(defect.metrics.get('app1'))
                hidden_coupling_apps.add(defect.metrics.get('app2'))
        
        # Node colors based on defects
        node_colors = []
        for n in G.nodes():
            if n in spof_components:
                node_colors.append('darkred')
            elif n in god_topics:
                node_colors.append('purple')
            elif n in circular_components:
                node_colors.append('orange')
            elif n in hidden_coupling_apps:
                node_colors.append('yellow')
            else:
                # Default colors by type
                node_type = G.nodes[n].get('type', 'unknown')
                color_map = {
                    'topic': 'lightblue',
                    'broker': 'lightgreen',
                    'node': 'lightcoral',
                    'application': 'lightyellow'
                }
                node_colors.append(color_map.get(node_type, 'gray'))
        
        # Node sizes based on scores
        node_sizes = [scores.get(n, 0.1) * 3000 + 500 for n in G.nodes()]
        
        plt.figure(figsize=(18, 14))
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, 
                              arrowsize=10, width=1.5)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=node_sizes, alpha=0.9,
                              edgecolors='black', linewidths=2)
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkred', edgecolor='black', label='SPOF'),
            Patch(facecolor='purple', edgecolor='black', label='God Topic'),
            Patch(facecolor='orange', edgecolor='black', label='Circular Dependency'),
            Patch(facecolor='yellow', edgecolor='black', label='Hidden Coupling')
        ]
        plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        plt.title('Pub-Sub System Graph with Design Defects Highlighted\n(Node size = Criticality)', 
                 fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


# ============================================================================
# EXAMPLE USAGE WITH PROBLEMATIC PATTERNS
# ============================================================================

def create_problematic_pubsub_system() -> PubSubGraphModel:
    """Create example system with intentional design defects"""
    model = PubSubGraphModel()
    
    # Physical nodes - LIMITED (creates SPOF)
    model.add_physical_node('node1', {'datacenter': 'DC1'})
    model.add_physical_node('node2', {'datacenter': 'DC1'})
    model.add_physical_node('node3', {'datacenter': 'DC2'})
    
    # Brokers - node3 is single broker in DC2 (SPOF)
    model.add_broker('broker1', 'node1', {'capacity': 1000})
    model.add_broker('broker2', 'node2', {'capacity': 1000})
    model.add_broker('broker3', 'node3', {'capacity': 800})  # SPOF for DC2
    
    # Broker replication - linear topology creates dependencies
    model.add_broker_link('broker1', 'broker2')
    model.add_broker_link('broker2', 'broker3')
    # Note: broker1 and broker3 not directly connected
    
    # GOD TOPIC: central_events with many connections
    model.add_topic('central_events', 'broker1', 
                   QoSProfile(latency_requirement=100, throughput_requirement=8000,
                            reliability=0.99, priority=5))
    
    # Other topics
    model.add_topic('payment_events', 'broker1', 
                   QoSProfile(latency_requirement=50, throughput_requirement=5000,
                            reliability=0.99, priority=5))
    
    model.add_topic('user_analytics', 'broker2',
                   QoSProfile(latency_requirement=500, throughput_requirement=1000,
                            reliability=0.95, priority=3))
    
    model.add_topic('inventory_updates', 'broker2',
                   QoSProfile(latency_requirement=100, throughput_requirement=3000,
                            reliability=0.98, priority=4))
    
    model.add_topic('logs', 'broker3',
                   QoSProfile(latency_requirement=1000, throughput_requirement=500,
                            reliability=0.90, priority=2))
    
    model.add_topic('notifications', 'broker2',
                   QoSProfile(latency_requirement=200, throughput_requirement=2000,
                            reliability=0.96, priority=3))
    
    model.add_topic('order_events', 'broker1',
                   QoSProfile(latency_requirement=150, throughput_requirement=4000,
                            reliability=0.97, priority=4))
    
    # Publishers
    model.add_application('payment_service', 'publisher', {'criticality': 'high'})
    model.add_application('analytics_service', 'publisher', {'criticality': 'medium'})
    model.add_application('inventory_service', 'publisher', {'criticality': 'high'})
    model.add_application('app_servers', 'publisher', {'criticality': 'medium'})
    model.add_application('order_service', 'publisher', {'criticality': 'high'})
    model.add_application('user_service', 'publisher', {'criticality': 'medium'})
    
    # Subscribers
    model.add_application('fraud_detection', 'subscriber', {'criticality': 'high'})
    model.add_application('dashboard', 'subscriber', {'criticality': 'medium'})
    model.add_application('warehouse_system', 'subscriber', {'criticality': 'high'})
    model.add_application('notification_service', 'subscriber', {'criticality': 'medium'})
    model.add_application('audit_system', 'subscriber', {'criticality': 'low'})
    model.add_application('reporting_service', 'subscriber', {'criticality': 'medium'})
    model.add_application('billing_service', 'subscriber', {'criticality': 'high'})
    
    # GOD TOPIC pattern: Many apps publish/subscribe to central_events
    model.add_publish_relation('payment_service', 'central_events', rate=2000)
    model.add_publish_relation('order_service', 'central_events', rate=1500)
    model.add_publish_relation('user_service', 'central_events', rate=1000)
    model.add_publish_relation('inventory_service', 'central_events', rate=800)
    
    model.add_subscribe_relation('central_events', 'fraud_detection')
    model.add_subscribe_relation('central_events', 'dashboard')
    model.add_subscribe_relation('central_events', 'audit_system')
    model.add_subscribe_relation('central_events', 'reporting_service')
    model.add_subscribe_relation('central_events', 'billing_service')
    
    # Regular publish relationships
    model.add_publish_relation('payment_service', 'payment_events', rate=1000)
    model.add_publish_relation('analytics_service', 'user_analytics', rate=500)
    model.add_publish_relation('inventory_service', 'inventory_updates', rate=800)
    model.add_publish_relation('app_servers', 'logs', rate=2000)
    model.add_publish_relation('payment_service', 'notifications', rate=300)
    model.add_publish_relation('order_service', 'order_events', rate=1200)
    
    # Regular subscribe relationships
    model.add_subscribe_relation('payment_events', 'fraud_detection')
    model.add_subscribe_relation('user_analytics', 'dashboard')
    model.add_subscribe_relation('inventory_updates', 'warehouse_system')
    model.add_subscribe_relation('logs', 'audit_system')
    model.add_subscribe_relation('notifications', 'notification_service')
    model.add_subscribe_relation('payment_events', 'audit_system')
    model.add_subscribe_relation('inventory_updates', 'dashboard')
    model.add_subscribe_relation('order_events', 'warehouse_system')
    model.add_subscribe_relation('order_events', 'billing_service')
    
    # CIRCULAR DEPENDENCY: Create feedback loop
    # fraud_detection subscribes to payment_events
    # but also publishes to notifications (which payment_service subscribes to)
    model.add_application('fraud_detector_pub', 'publisher', {'criticality': 'high'})
    model.add_publish_relation('fraud_detector_pub', 'notifications', rate=100)
    # Add edge from fraud_detection to fraud_detector_pub to create cycle
    model.composite_graph.add_edge('fraud_detection', 'fraud_detector_pub', relation='triggers')
    
    # HIDDEN COUPLING: Multiple services share topics
    # dashboard and reporting_service both subscribe to same topics
    model.add_subscribe_relation('payment_events', 'reporting_service')
    model.add_subscribe_relation('order_events', 'reporting_service')
    model.add_subscribe_relation('order_events', 'dashboard')
    
    return model


def main():
    """Run complete analysis with anti-pattern detection"""
    print("="*80)
    print("CRITICAL COMPONENT ANALYSIS FOR DISTRIBUTED PUB-SUB SYSTEMS")
    print("WITH ANTI-PATTERN DETECTION")
    print("="*80)
    print()
    
    # Create example system with problems
    print("Creating pub-sub system with intentional design defects...")
    model = create_problematic_pubsub_system()
    print(f"System created with {len(model.composite_graph.nodes())} components")
    print(f"- Physical nodes: {len(model.physical_layer.nodes())}")
    print(f"- Brokers: {len(model.broker_layer.nodes())}")
    print(f"- Topics: {len(model.logical_layer.nodes())}")
    print(f"- Applications: {len(model.application_layer.nodes())}")
    print()
    
    # Run integrated analysis
    print("Running integrated analysis...")
    analyzer = IntegratedAnalysis(model)
    results = analyzer.run_complete_analysis()
    print()
    
    # Display results
    print("="*80)
    print("TOP CRITICAL COMPONENTS")
    print("="*80)
    print(results['component_analysis'][
        ['component', 'type', 'topology_score', 'qos_score', 
         'impact_score', 'integrated_score', 'is_articulation_point']
    ].head(10).to_string(index=False))
    print()
    
    print("="*80)
    print("CRITICAL PATHS (Publisher -> Subscriber)")
    print("="*80)
    if not results['critical_paths'].empty:
        print(results['critical_paths'][
            ['publisher', 'subscriber', 'path_length', 'criticality_score']
        ].head(10).to_string(index=False))
    print()
    
    # Anti-pattern results
    print("="*80)
    print("DESIGN DEFECTS AND ANTI-PATTERNS DETECTED")
    print("="*80)
    if not results['anti_patterns'].empty:
        for idx, row in results['anti_patterns'].iterrows():
            severity_marker = {
                'critical': '🔴',
                'high': '🟠',
                'medium': '🟡',
                'low': '🟢'
            }.get(row['severity'], '⚪')
            
            print(f"\n{severity_marker} [{row['severity'].upper()}] {row['defect_type']}")
            print(f"   Component: {row['component']}")
            print(f"   {row['description']}")
    else:
        print("No design defects detected!")
    print()
    
    # Detailed defect information
    print("="*80)
    print("DETAILED DEFECT ANALYSIS")
    print("="*80)
    defects_by_type = defaultdict(list)
    for defect in results['defects']:
        defects_by_type[defect.defect_type].append(defect)
    
    for defect_type, defects in defects_by_type.items():
        print(f"\n{defect_type} ({len(defects)} detected):")
        print("-" * 70)
        for defect in defects[:3]:  # Show top 3 of each type
            print(f"  • {defect.component} ({defect.severity})")
            print(f"    Metrics: {defect.metrics}")
            print(f"    Recommendations:")
            for rec in defect.recommendations[:2]:
                print(f"      - {rec}")
            print()
    
    # Visualizations
    print("Generating visualizations...")
    viz = Visualizer()
    
    viz.plot_component_scores(results['component_analysis'])
    viz.plot_integrated_scores(results['component_analysis'])
    viz.plot_anti_pattern_summary(results['anti_patterns'], results['defects'])
    
    # Graph visualization with defects highlighted
    scores = dict(zip(results['component_analysis']['component'],
                     results['component_analysis']['integrated_score']))
    viz.plot_graph_with_defects(model, scores, results['defects'])
    
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total components analyzed: {len(results['component_analysis'])}")
    print(f"Articulation points (SPOF candidates): {results['component_analysis']['is_articulation_point'].sum()}")
    print(f"Total defects found: {len(results['defects'])}")
    
    if not results['anti_patterns'].empty:
        severity_counts = results['anti_patterns']['severity'].value_counts()
        print(f"\nDefects by severity:")
        for severity in ['critical', 'high', 'medium', 'low']:
            if severity in severity_counts.index:
                print(f"  - {severity.capitalize()}: {severity_counts[severity]}")
    
    print("\nAnalysis complete!")
    
    return model, results


# Run the analysis
if __name__ == "__main__":
    model, results = main()
