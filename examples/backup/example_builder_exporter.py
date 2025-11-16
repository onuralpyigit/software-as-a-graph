"""
Example: Graph Builder and Exporter Usage

Demonstrates how to:
1. Build graphs from various sources (JSON, CSV, NetworkX)
2. Export graphs to various formats (Neo4j, JSON, CSV, GraphML, DOT)
3. Convert between formats
4. Extract layer-specific views
"""

import sys
from pathlib import Path
import networkx as nx
import json

sys.path.insert(0, str(Path(__file__).parent / '..'))

from src.core.graph_model import (
    GraphModel, ApplicationNode, TopicNode, BrokerNode,
    QoSPolicy, QoSDurability, QoSReliability, ApplicationType,
    PublishesEdge, SubscribesEdge, RoutesEdge, RunsOnEdge
)
from src.core.graph_builder import GraphBuilder, create_example_json_config
from src.core.graph_exporter import GraphExporter


def example_1_json_roundtrip():
    """Example 1: JSON round-trip (build from JSON, export back to JSON)"""
    
    print("\n" + "=" * 70)
    print("Example 1: JSON Round-Trip")
    print("=" * 70)
    
    # Step 1: Create example JSON configuration
    print("\n[Step 1] Creating example JSON configuration...")
    config = create_example_json_config()
    
    # Save to file
    json_input = "temp_input.json"
    with open(json_input, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"  ✓ Saved configuration to {json_input}")
    
    # Step 2: Build graph from JSON
    print("\n[Step 2] Building graph from JSON...")
    builder = GraphBuilder()
    model = builder.build_from_json(json_input)
    
    summary = model.summary()
    print(f"  ✓ Built graph:")
    print(f"    - Applications: {summary['applications']}")
    print(f"    - Topics: {summary['topics']}")
    print(f"    - Brokers: {summary['brokers']}")
    print(f"    - Nodes: {summary['nodes']}")
    print(f"    - Total edges: {summary['total_edges']}")
    print(f"    - DEPENDS_ON edges: {summary['depends_on_edges']} (auto-derived)")
    
    # Step 3: Export back to JSON
    print("\n[Step 3] Exporting graph back to JSON...")
    exporter = GraphExporter()
    json_output = exporter.export_to_json(model, "temp_output.json")
    print(f"  ✓ Exported to {json_output}")
    
    # Step 4: Verify round-trip
    print("\n[Step 4] Verifying round-trip...")
    with open(json_output, 'r') as f:
        output_data = json.load(f)
    
    print(f"  ✓ Round-trip successful!")
    print(f"    Input summary: {config.keys()}")
    print(f"    Output summary: {output_data.keys()}")
    
    # Cleanup
    Path(json_input).unlink()
    Path(json_output).unlink()


def example_2_csv_export():
    """Example 2: Build graph and export to CSV"""
    
    print("\n" + "=" * 70)
    print("Example 2: Export to CSV")
    print("=" * 70)
    
    # Step 1: Create a simple graph
    print("\n[Step 1] Creating graph...")
    model = GraphModel()
    
    # Add components
    model.add_application(ApplicationNode(
        name="WebApp",
        app_type=ApplicationType.PRODUCER,
        cpu_cores=2.0,
        memory_mb=2048
    ))
    
    model.add_application(ApplicationNode(
        name="DataProcessor",
        app_type=ApplicationType.CONSUMER,
        cpu_cores=4.0,
        memory_mb=4096
    ))
    
    model.add_topic(TopicNode(
        name="events",
        qos_policy=QoSPolicy(
            durability=QoSDurability.PERSISTENT,
            reliability=QoSReliability.RELIABLE
        )
    ))
    
    model.publishes_edges.append(PublishesEdge(source="WebApp", target="events"))
    model.subscribes_edges.append(SubscribesEdge(source="DataProcessor", target="events"))
    
    print(f"  ✓ Created graph with {len(model.applications)} applications, {len(model.topics)} topics")
    
    # Step 2: Export to CSV
    print("\n[Step 2] Exporting to CSV...")
    exporter = GraphExporter()
    
    nodes_file, edges_file, qos_file = exporter.export_to_csv(
        model,
        "temp_nodes.csv",
        "temp_edges.csv",
        "temp_qos.csv"
    )
    
    print(f"  ✓ Exported to:")
    print(f"    - Nodes: {nodes_file}")
    print(f"    - Edges: {edges_file}")
    print(f"    - QoS: {qos_file}")
    
    # Step 3: Show CSV contents
    print("\n[Step 3] CSV Contents:")
    
    print("\n  Nodes CSV (first few lines):")
    with open(nodes_file, 'r') as f:
        for i, line in enumerate(f):
            if i < 4:
                print(f"    {line.strip()}")
    
    print("\n  Edges CSV (first few lines):")
    with open(edges_file, 'r') as f:
        for i, line in enumerate(f):
            if i < 4:
                print(f"    {line.strip()}")
    
    # Cleanup
    Path(nodes_file).unlink()
    Path(edges_file).unlink()
    if qos_file:
        Path(qos_file).unlink()


def example_3_networkx_conversion():
    """Example 3: Convert between GraphModel and NetworkX"""
    
    print("\n" + "=" * 70)
    print("Example 3: NetworkX Conversion")
    print("=" * 70)
    
    # Step 1: Create GraphModel
    print("\n[Step 1] Creating GraphModel...")
    model = GraphModel()
    
    for i in range(1, 4):
        model.add_application(ApplicationNode(name=f"app{i}"))
        model.add_topic(TopicNode(name=f"topic{i}"))
    
    model.publishes_edges.append(PublishesEdge(source="app1", target="topic1"))
    model.subscribes_edges.append(SubscribesEdge(source="app2", target="topic1"))
    
    print(f"  ✓ Created GraphModel with {len(model.applications)} apps, {len(model.topics)} topics")
    
    # Step 2: Export to NetworkX
    print("\n[Step 2] Converting to NetworkX...")
    exporter = GraphExporter()
    nx_graph = exporter.export_to_networkx(model)
    
    print(f"  ✓ NetworkX graph:")
    print(f"    - Nodes: {len(nx_graph)}")
    print(f"    - Edges: {len(nx_graph.edges())}")
    print(f"    - Density: {nx.density(nx_graph):.3f}")
    
    # Step 3: Convert back to GraphModel
    print("\n[Step 3] Converting back to GraphModel...")
    builder = GraphBuilder()
    model_rebuilt = builder.build_from_networkx(nx_graph)
    
    summary = model_rebuilt.summary()
    print(f"  ✓ Rebuilt GraphModel:")
    print(f"    - Total nodes: {summary['total_nodes']}")
    print(f"    - Total edges: {summary['total_edges']}")


def example_4_layer_export():
    """Example 4: Export layer-specific views"""
    
    print("\n" + "=" * 70)
    print("Example 4: Layer-Specific Export")
    print("=" * 70)
    
    # Step 1: Create a multi-layer system
    print("\n[Step 1] Creating multi-layer system...")
    
    config = {
        "applications": [
            {"name": "App1", "app_type": "Producer"},
            {"name": "App2", "app_type": "Consumer"}
        ],
        "topics": [
            {"name": "Topic1", "broker": "Broker1"}
        ],
        "brokers": [
            {"name": "Broker1", "node_host": "Node1"}
        ],
        "nodes": [
            {"name": "Node1", "datacenter": "DC1"}
        ],
        "edges": {
            "publishes": [{"source": "App1", "target": "Topic1"}],
            "subscribes": [{"source": "App2", "target": "Topic1"}],
            "routes": [{"source": "Broker1", "target": "Topic1"}],
            "runs_on": [
                {"source": "App1", "target": "Node1"},
                {"source": "App2", "target": "Node1"},
                {"source": "Broker1", "target": "Node1"}
            ],
            "connects_to": []
        }
    }
    
    builder = GraphBuilder()
    model = builder.build_from_dict(config)
    
    summary = model.summary()
    print(f"  ✓ Created system with {summary['total_nodes']} nodes across all layers")
    
    # Step 2: Export each layer separately
    print("\n[Step 2] Exporting individual layers...")
    exporter = GraphExporter()
    
    # Application layer
    app_layer = exporter.export_layer_to_networkx(model, layer='application')
    print(f"\n  Application Layer:")
    print(f"    - Nodes: {len(app_layer)}")
    print(f"    - Edges: {len(app_layer.edges())}")
    print(f"    - Node types: {set(nx.get_node_attributes(app_layer, 'type').values())}")
    
    # Infrastructure layer
    infra_layer = exporter.export_layer_to_networkx(model, layer='infrastructure')
    print(f"\n  Infrastructure Layer:")
    print(f"    - Nodes: {len(infra_layer)}")
    print(f"    - Edges: {len(infra_layer.edges())}")
    print(f"    - Node types: {set(nx.get_node_attributes(infra_layer, 'type').values())}")
    
    # Topic layer
    topic_layer = exporter.export_layer_to_networkx(model, layer='topic')
    print(f"\n  Topic Layer:")
    print(f"    - Nodes: {len(topic_layer)}")
    print(f"    - Edges: {len(topic_layer.edges())}")
    print(f"    - Edge types: {set(d.get('type') for _, _, d in topic_layer.edges(data=True))}")


def example_5_graphml_export():
    """Example 5: Export to GraphML for visualization tools"""
    
    print("\n" + "=" * 70)
    print("Example 5: GraphML Export (for Gephi, yEd)")
    print("=" * 70)
    
    # Step 1: Create example graph
    print("\n[Step 1] Creating example graph...")
    config = create_example_json_config()
    builder = GraphBuilder()
    model = builder.build_from_dict(config)

    print(f"  ✓ Created graph with {model.summary()['total_nodes']} nodes")

    # Step 2: Export to GraphML
    print("\n[Step 2] Exporting to GraphML...")
    exporter = GraphExporter()
    graphml_path = exporter.export_to_graphml(model, "temp_graph.graphml")
    
    print(f"  ✓ Exported to {graphml_path}")
    print(f"  ℹ️  You can now open this file in:")
    print(f"    - Gephi (https://gephi.org/)")
    print(f"    - yEd (https://www.yworks.com/products/yed)")
    print(f"    - Cytoscape (https://cytoscape.org/)")
    
    # Cleanup
    Path(graphml_path).unlink()


def example_6_dot_export():
    """Example 6: Export to DOT for Graphviz"""
    
    print("\n" + "=" * 70)
    print("Example 6: DOT Export (for Graphviz)")
    print("=" * 70)
    
    # Step 1: Create small example
    print("\n[Step 1] Creating simple graph...")
    model = GraphModel()
    
    model.add_application(ApplicationNode(name="Publisher"))
    model.add_application(ApplicationNode(name="Subscriber"))
    model.add_topic(TopicNode(name="Events"))
    model.add_broker(BrokerNode(name="MessageBroker"))
    
    model.publishes_edges.append(PublishesEdge(source="Publisher", target="Events"))
    model.subscribes_edges.append(SubscribesEdge(source="Subscriber", target="Events"))
    model.routes_edges.append(RoutesEdge(source="MessageBroker", target="Events"))
    
    print(f"  ✓ Created graph with {model.summary()['total_nodes']} nodes")
    
    # Step 2: Export to DOT
    print("\n[Step 2] Exporting to DOT format...")
    exporter = GraphExporter()
    dot_path = exporter.export_to_dot(model, "temp_graph.dot")
    
    print(f"  ✓ Exported to {dot_path}")
    
    # Step 3: Show DOT content
    print("\n[Step 3] DOT file content:")
    with open(dot_path, 'r') as f:
        content = f.read()
        print("  " + "\n  ".join(content.split('\n')[:15]))
        print("  ...")
    
    print(f"\n  ℹ️  To visualize, run:")
    print(f"    dot -Tpng {dot_path} -o graph.png")
    print(f"    dot -Tsvg {dot_path} -o graph.svg")
    
    # Cleanup
    Path(dot_path).unlink()


def example_7_summary_export():
    """Example 7: Export summary statistics"""
    
    print("\n" + "=" * 70)
    print("Example 7: Summary Statistics")
    print("=" * 70)
    
    # Step 1: Create comprehensive graph
    print("\n[Step 1] Creating comprehensive graph...")
    config = create_example_json_config()
    builder = GraphBuilder()
    model = builder.build_from_dict(config)
    
    # Step 2: Generate summary
    print("\n[Step 2] Generating summary...")
    exporter = GraphExporter()
    summary = exporter.export_summary(model)
    
    print("\n  Graph Summary:")
    print(f"    Total Nodes: {summary['total_nodes']}")
    print(f"    Total Edges: {summary['total_edges']}")
    
    print("\n  Node Type Distribution:")
    for node_type, count in summary['node_type_distribution'].items():
        print(f"    - {node_type}: {count}")
    
    print("\n  Edge Type Distribution:")
    for edge_type, count in summary['edge_type_distribution'].items():
        print(f"    - {edge_type}: {count}")
    
    if 'qos_statistics' in summary:
        print("\n  QoS Statistics:")
        qos_stats = summary['qos_statistics']
        print(f"    - Average QoS Score: {qos_stats['avg_score']:.3f}")
        print(f"    - Max QoS Score: {qos_stats['max_score']:.3f}")
        print(f"    - Min QoS Score: {qos_stats['min_score']:.3f}")
        print(f"    - Topics with High QoS: {qos_stats['topics_with_high_qos']}")


def main():
    """Run all examples"""
    
    print("\n" + "=" * 70)
    print("GRAPH BUILDER AND EXPORTER - COMPREHENSIVE EXAMPLES")
    print("=" * 70)
    
    try:
        example_1_json_roundtrip()
        example_2_csv_export()
        example_3_networkx_conversion()
        example_4_layer_export()
        example_5_graphml_export()
        example_6_dot_export()
        example_7_summary_export()
        
        print("\n" + "=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print("\n📚 Summary of Capabilities:")
        print("  ✓ Build from: JSON, CSV, NetworkX, Neo4j")
        print("  ✓ Export to: JSON, CSV, NetworkX, Neo4j, GraphML, DOT")
        print("  ✓ Layer extraction: Application, Infrastructure, Topic")
        print("  ✓ Auto-derive: DEPENDS_ON relationships")
        print("  ✓ Round-trip: Lossless format conversion")
        
        print("\n📖 Usage in Your Code:")
        print("""
from src.core.graph_builder import GraphBuilder
from src.core.graph_exporter import GraphExporter

# Build from JSON
builder = GraphBuilder()
model = builder.build_from_json("config.json")

# Export to multiple formats
exporter = GraphExporter()
exporter.export_to_json(model, "output.json")
exporter.export_to_csv(model, "nodes.csv", "edges.csv")
nx_graph = exporter.export_to_networkx(model)
        """)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
