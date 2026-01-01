"""
Software-as-a-Graph Core Module - Version 5.0

A simplified, refactored core module for pub-sub system graph modeling,
generation, and Neo4j import with QoS-aware dependency analysis.

Graph Model:
    Vertices: Application, Broker, Topic, Node (all with weight property)
    Edges: PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON, CONNECTS_TO
    Derived: DEPENDS_ON (app_to_app, node_to_node, app_to_broker, node_to_broker)

Weight Properties:
    - DEPENDS_ON edges: weight based on topic count, QoS, and message size
    - Components: weight based on sum of incoming/outgoing dependency weights

Usage:
    # Generate a graph
    from src.core import generate_graph
    graph = generate_graph(scale="medium", scenario="iot")
    
    # Import into Neo4j with weight calculation
    from src.core import GraphImporter
    with GraphImporter(uri="bolt://localhost:7687", password="secret") as importer:
        stats = importer.import_graph(graph)
        importer.show_analytics()

    # Work with graph model
    from src.core import GraphModel
    model = GraphModel.from_dict(graph)
    model.calculate_component_weights()
    print(model.summary())

Author: Software-as-a-Graph Research Project
Version: 5.0
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

# Graph Importer - Neo4j integration
from .graph_importer import (
    GraphImporter,
)

# Try to import graph generator if available
try:
    from .graph_generator import (
        GraphConfig,
        GraphGenerator,
        generate_graph,
    )
    _HAS_GENERATOR = True
except ImportError:
    _HAS_GENERATOR = False
    GraphConfig = None
    GraphGenerator = None
    generate_graph = None

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
    # Importer
    "GraphImporter",
]

# Add generator exports if available
if _HAS_GENERATOR:
    __all__.extend(["GraphConfig", "GraphGenerator", "generate_graph"])

__version__ = "5.0.0"