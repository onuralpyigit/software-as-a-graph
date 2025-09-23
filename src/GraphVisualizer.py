import networkx as nx
import random
import matplotlib.pyplot as plt

class GraphVisualizer:    
    def __init__(self):
        self.output_file_dir = "output/"
        
    def visualize_graph(self, graph):
        """Visualizes the graph using NetworkX and matplotlib"""
        # Create a new figure with a specific size
        plt.figure(figsize=(16, 12))
        
        # Define positions using a force-directed layout
        pos = nx.spring_layout(graph, k=0.5, iterations=50)
        
        # Define node colors based on node type
        color_map = {
            "Node": "skyblue",
            "Broker": "red",
            "Application": "green",
            "Topic": "yellow"
        }
        
        # Extract node types and assign colors
        node_colors = [color_map[data["type"]] for _, data in graph.nodes(data=True)]
        
        # Draw nodes
        nx.draw_networkx_nodes(graph, pos, node_size=500, node_color=node_colors, alpha=0.8)
        
        # Draw edges with different colors based on relationship type
        edge_colors = {
            "RUNS_ON": "gray",
            "PUBLISHES_TO": "blue",
            "SUBSCRIBES_TO": "green",
            "ROUTES": "red",
            "DEPENDS_ON": "purple",
            "CONNECTS_TO": "black"
        }
        
        # Draw edges by type
        for edge_type, color in edge_colors.items():
            edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("type") == edge_type]
            nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color=color, width=1.5, alpha=0.7, 
                                  arrowsize=15)
        
        # Draw labels with smaller font
        nx.draw_networkx_labels(graph, pos, labels={n: n for n in graph.nodes()}, 
                               font_size=8, font_weight="bold")
        
        # Create legend
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                     markersize=10, label=node_type) 
                         for node_type, color in color_map.items()]
        legend_elements += [plt.Line2D([0], [0], color=color, lw=2, label=edge_type) 
                          for edge_type, color in edge_colors.items()]
        
        plt.legend(handles=legend_elements, loc='upper right')
        plt.title("Publish-Subscribe System Graph Model", fontsize=16)
        plt.axis('off')
        output_file = self.output_file_dir + "Publish-Subscribe System Graph Model.png"
        plt.savefig(output_file)

    def visualize_layer(self, graph, node_types, edge_types, title, layout=None, scale_factor=1.0):
        """
        Visualize a specific layer of the system graph
        
        Parameters:
        - G: NetworkX graph object
        - node_types: List of node types to include
        - edge_types: List of edge types to include
        - title: Title for the visualization
        - layout: Optional pre-computed layout
        - scale_factor: Factor to scale visualization elements (for larger graphs)
        """
        # Create a subgraph with selected node and edge types
        SG = nx.DiGraph()
        
        # Add nodes of specified types
        for node, attrs in graph.nodes(data=True):
            if attrs.get('type') in node_types:
                SG.add_node(node, **attrs)
        
        # Add edges of specified types
        for u, v, attrs in graph.edges(data=True):
            if attrs.get('type') in edge_types and u in SG.nodes and v in SG.nodes:
                SG.add_edge(u, v, **attrs)
        
        # Skip empty graphs
        if len(SG) == 0:
            print(f"No nodes in {title} layer")
            return
        
        # Set figure size based on graph size
        node_count = len(SG)
        fig_size_base = 10
        fig_size = max(8, min(fig_size_base, fig_size_base * scale_factor * (node_count / 20)))
        plt.figure(figsize=(fig_size, fig_size))
        
        # Use specified layout or determine best layout based on graph properties
        if layout is None:
            # For small graphs, spring layout works well
            if node_count < 50:
                pos = nx.spring_layout(SG, seed=42)
            # For larger graphs, use faster algorithms
            else:
                try:
                    pos = nx.kamada_kawai_layout(SG)
                except:
                    # Fallback to spring layout with reduced iterations
                    pos = nx.spring_layout(SG, seed=42, iterations=50)
        else:
            pos = layout
        
        # Scale node size inversely with graph size
        node_size_base = 800
        node_size = max(100, node_size_base * (scale_factor * (50 / max(node_count, 1))))
        
        # Color nodes by type
        color_map = {
            'Application': 'lightblue',
            'Broker': 'orange',
            'Topic': 'lightgreen', 
            'Node': 'pink'
        }
        
        node_colors = [color_map[SG.nodes[node]['type']] for node in SG.nodes]
        
        # Draw nodes
        nx.draw_networkx_nodes(SG, pos, node_size=node_size, node_color=node_colors, alpha=0.8)
        
        # Draw edges with different colors by type
        edge_colors = {
            'RUNS_ON': 'gray',
            'PUBLISHES_TO': 'green',
            'SUBSCRIBES_TO': 'blue',
            'ROUTES': 'orange',
            'DEPENDS_ON': 'red',
            'CONNECTS_TO': 'black'
        }
        
        # Scale edge properties based on graph size
        edge_width = max(0.5, 1.5 * scale_factor)
        arrow_size = max(5, 15 * scale_factor)
        
        for edge_type in edge_types:
            edges = [(u, v) for u, v, attrs in SG.edges(data=True) if attrs.get('type') == edge_type]
            if edges:
                nx.draw_networkx_edges(SG, pos, edgelist=edges, 
                                    edge_color=edge_colors[edge_type], 
                                    width=edge_width, 
                                    arrowsize=arrow_size,
                                    connectionstyle='arc3,rad=0.1')
        
        # Adjust label size based on graph size
        font_size_base = 10
        font_size = max(6, font_size_base * scale_factor)
        
        # For large graphs, limit labels or use a different approach
        if node_count > 100:
            # Only label important nodes
            important_nodes = {}
            for node, attrs in SG.nodes(data=True):
                if attrs.get('type') in ['Broker', 'Node']:
                    important_nodes[node] = node
                elif len(list(SG.predecessors(node))) > 3 or len(list(SG.successors(node))) > 3:
                    important_nodes[node] = node
            
            nx.draw_networkx_labels(SG, pos, labels=important_nodes, font_size=font_size, font_weight='bold')
        else:
            # Label all nodes
            nx.draw_networkx_labels(SG, pos, font_size=font_size, font_weight='bold')
        
        # Create legend for node types
        node_patches = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=node_type) 
                    for node_type, color in color_map.items() if node_type in node_types]
        
        # Create legend for edge types
        edge_patches = [plt.Line2D([0], [0], color=color, lw=2, label=edge_type) 
                    for edge_type, color in edge_colors.items() if edge_type in edge_types]
        
        # Add legends
        plt.legend(handles=node_patches + edge_patches, loc='upper right')
        
        plt.title(f"{title} ({node_count} nodes)")
        plt.axis('off')
        plt.tight_layout()
        
        # Save figure with size info
        filename = self.output_file_dir + title + ".png"
        plt.savefig(filename, dpi=300)
    
    def visualize_layers(self, graph):
        """Visualize different layers of the system graph"""
        # Application Layer (Applications and their dependencies)
        self.visualize_layer(graph, 
            node_types=['Application'], 
            edge_types=['DEPENDS_ON'],
            title='Application Level Layer')
        
        # Infrastructure Layer (Nodes and their connections)
        self.visualize_layer(graph,  
            node_types=['Node'], 
            edge_types=['CONNECTS_TO'],
            title='Infrastructure Level Layer')
        
        # Application-Infrastructure Layer (Applications, Brokers, and their hosting)
        self.visualize_layer(graph,  
            node_types=['Application', 'Broker', 'Node'], 
            edge_types=['RUNS_ON', 'CONNECTS_TO'],
            title='Application-Infrastructure Layer')
        
        # Messaging Layer (Applications, Topics, and Brokers)
        self.visualize_layer(graph,  
            node_types=['Application', 'Topic', 'Broker'], 
            edge_types=['PUBLISHES_TO', 'SUBSCRIBES_TO', 'ROUTES'],
            title='Messaging Layer')
        
        # Complete System View (All elements)
        self.visualize_layer(graph,  
            node_types=['Application', 'Topic', 'Broker', 'Node'], 
            edge_types=['PUBLISHES_TO', 'SUBSCRIBES_TO', 'ROUTES', 'RUNS_ON', 'DEPENDS_ON', 'CONNECTS_TO'],
            title='Complete System View')

def generate_example_graph():
    # Create a sample pub-sub system graph
    graph = nx.DiGraph()
    
    # Add nodes
    # Physical nodes
    for i in range(1, 5):  # 4 nodes
        graph.add_node(f"node{i}", type="Node", name=f"Node {i}")
    
    # Brokers
    for i in range(1, 3):  # 2 brokers
        graph.add_node(f"broker{i}", type="Broker", name=f"Broker {i}")
        graph.add_edge(f"broker{i}", f"node{i}", type="RUNS_ON")
    
    # Applications
    for i in range(1, 11):  # 10 applications
        graph.add_node(f"app{i}", type="Application", name=f"App {i}")
        # Assign to a random node
        node_id = f"node{random.randint(1, 4)}"
        graph.add_edge(f"app{i}", node_id, type="RUNS_ON")
    
    # Topics
    for i in range(1, 26):  # 25 topics
        graph.add_node(f"topic{i}", type="Topic", name=f"Topic {i}")
        # Assign to a broker
        broker_id = f"broker{random.randint(1, 2)}"
        graph.add_edge(broker_id, f"topic{i}", type="ROUTES")
    
    # Create publish relationships
    for i in range(1, 11):  # For each app
        # Publish to 1-5 random topics
        num_topics = random.randint(1, 6)
        topic_ids = random.sample(range(1, 26), num_topics)
        for topic_num in topic_ids:
            graph.add_edge(f"app{i}", f"topic{topic_num}", type="PUBLISHES_TO")
    
    # Create subscribe relationships
    for i in range(1, 11):  # For each app
        # Subscribe to 1-8 random topics
        num_topics = random.randint(1, 9)
        topic_ids = random.sample(range(1, 26), num_topics)
        for topic_num in topic_ids:
            graph.add_edge(f"app{i}", f"topic{topic_num}", type="SUBSCRIBES_TO")
    
    # Create node connections (CONNECTS_TO)
    for i in range(1, 5):
        for j in range(i + 1, 5):
            graph.add_edge(f"node{i}", f"node{j}", type="CONNECTS_TO")
    
    return graph
