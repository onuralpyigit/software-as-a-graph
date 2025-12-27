"""
Software-as-a-Graph Core Module - Version 4.0

A simplified, refactored core module for pub-sub system graph modeling,
generation, and Neo4j import with QoS-aware dependency analysis.

Graph Model:
    Vertices: Application, Broker, Topic, Node
    Edges: PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON, CONNECTS_TO
    Derived: DEPENDS_ON (app_to_app, node_to_node, app_to_broker, node_to_broker)

Usage:
    # Generate a graph
    from src.core import generate_graph
    graph = generate_graph(scale="medium", scenario="iot")
    
    # Import into Neo4j
    from src.core import GraphImporter
    with GraphImporter(uri="bolt://localhost:7687", password="secret") as importer:
        importer.import_graph(graph)
        stats = importer.get_statistics()

    # Work with graph model
    from src.core import GraphModel
    model = GraphModel.from_dict(graph)
    print(model.summary())

Author: Software-as-a-Graph Research Project
Version: 4.0
"""

# Graph Model - Data structures
from .graph_model import (
    # Enums
    VertexType,
    EdgeType,
    DependencyType,
    Durability,
    Reliability,
    Priority,
    # QoS
    QoSPolicy,
    # Vertices
    Application,
    Broker,
    Topic,
    Node,
    # Edges
    Edge,
    DependsOnEdge,
    # Model
    GraphModel,
)

# Graph Generator - Create test graphs
from .graph_generator import (
    GraphConfig,
    GraphGenerator,
    generate_graph,
)

# Graph Importer - Neo4j integration
from .graph_importer import (
    GraphImporter,
)

__all__ = [
    # Enums
    "VertexType",
    "EdgeType",
    "DependencyType",
    "Durability",
    "Reliability",
    "Priority",
    # QoS
    "QoSPolicy",
    # Vertices
    "Application",
    "Broker",
    "Topic",
    "Node",
    # Edges
    "Edge",
    "DependsOnEdge",
    # Model
    "GraphModel",
    # Generator
    "GraphConfig",
    "GraphGenerator",
    "generate_graph",
    # Importer
    "GraphImporter",
]

__version__ = "4.0.0"