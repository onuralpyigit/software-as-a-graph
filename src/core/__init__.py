"""
Software-as-a-Graph Core Module - Simplified Version 3.0

Graph Model:
- Vertices: Application, Broker, Topic, Node
- Edges: PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, RUNS_ON, CONNECTS_TO

Exports:
    - GraphGenerator, GraphConfig: Generate graphs
    - GraphModel: Container for graph data
    - GraphBuilder: Build GraphModel from sources
    - GraphExporter: Export GraphModel to formats
    - Vertex types: Application, Broker, Topic, Node
    - Edge type: Edge
    - QoSPolicy: QoS configuration
"""

from .graph_generator import GraphGenerator, GraphConfig, create_graph
from .graph_model import (
    GraphModel,
    Application,
    Broker,
    Topic,
    Node,
    Edge,
    QoSPolicy,
    ApplicationRole,
    DurabilityPolicy,
    ReliabilityPolicy,
    TransportPriority,
    EdgeType,
    VertexType
)
from .graph_builder import GraphBuilder, ValidationResult, GraphDiffResult
from .graph_exporter import GraphExporter

__all__ = [
    # Generator
    'GraphGenerator',
    'GraphConfig',
    'create_graph',
    
    # Model
    'GraphModel',
    'Application',
    'Broker',
    'Topic',
    'Node',
    'Edge',
    'QoSPolicy',
    
    # Enums
    'ApplicationRole',
    'DurabilityPolicy',
    'ReliabilityPolicy',
    'TransportPriority',
    'EdgeType',
    'VertexType',
    
    # Builder & Validation
    'GraphBuilder',
    'ValidationResult',
    'GraphDiffResult',
    
    # Exporter
    'GraphExporter'
]