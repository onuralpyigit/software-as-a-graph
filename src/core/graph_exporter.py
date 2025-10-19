"""
Graph Exporter

Exports GraphModel instances to various formats:
- Neo4j database
- NetworkX DiGraph
- JSON files
- CSV files
- GraphML
- DOT (Graphviz)

Supports full round-trip with GraphBuilder for lossless conversion.
"""

import json
import csv
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging

from .graph_model import GraphModel


class GraphExporter:
    """
    Exports GraphModel instances to various formats
    
    Supports:
    - Neo4j database (create/update nodes and relationships)
    - NetworkX DiGraph (for analysis)
    - JSON (for configuration storage)
    - CSV (for data exchange)
    - GraphML (for Gephi, yEd)
    - DOT (for Graphviz visualization)
    """
    
    def __init__(self):
        """Initialize the graph exporter"""
        self.logger = logging.getLogger(__name__)
    
    def export_to_neo4j(self,
                       model: GraphModel,
                       uri: str,
                       auth: Tuple[str, str],
                       database: str = "neo4j",
                       clear_existing: bool = False) -> Dict[str, int]:
        """
        Export graph to Neo4j database
        
        Args:
            model: GraphModel to export
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            auth: Tuple of (username, password)
            database: Database name
            clear_existing: If True, clear all existing data first
        
        Returns:
            Dictionary with counts of created nodes and relationships
        """
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError("neo4j package required. Install with: pip install neo4j")
        
        self.logger.info(f"Exporting graph to Neo4j: {uri}")
        
        driver = GraphDatabase.driver(uri, auth=auth)
        counts = {'nodes': 0, 'relationships': 0}
        
        with driver.session(database=database) as session:
            # Clear existing data if requested
            if clear_existing:
                session.run("MATCH (n) DETACH DELETE n")
                self.logger.info("Cleared existing data from Neo4j")
            
            # Create constraints and indexes
            self._create_neo4j_constraints(session)
            
            # Export nodes
            counts['nodes'] += self._export_nodes_to_neo4j(session, model)
            
            # Export relationships
            counts['relationships'] += self._export_edges_to_neo4j(session, model)
        
        driver.close()
        
        self.logger.info(f"Export complete: {counts['nodes']} nodes, {counts['relationships']} relationships")
        return counts
    
    def export_to_networkx(self, model: GraphModel) -> nx.DiGraph:
        """
        Export graph to NetworkX DiGraph
        
        Args:
            model: GraphModel to export
        
        Returns:
            NetworkX directed graph
        """
        self.logger.info("Exporting graph to NetworkX...")
        
        graph = nx.DiGraph()
        
        # Add all nodes with attributes
        all_nodes = model.get_all_nodes()
        for name, attrs in all_nodes.items():
            graph.add_node(name, **attrs)
        
        # Add all edges with attributes
        all_edges = model.get_all_edges()
        for edge_dict in all_edges:
            source = edge_dict.pop('source')
            target = edge_dict.pop('target')
            graph.add_edge(source, target, **edge_dict)
        
        self.logger.info(f"Export complete: {len(graph)} nodes, {len(graph.edges())} edges")
        return graph
    
    def export_to_json(self, 
                      model: GraphModel,
                      filepath: str,
                      indent: int = 2) -> Path:
        """
        Export graph to JSON file
        
        Args:
            model: GraphModel to export
            filepath: Output file path
            indent: JSON indentation (default: 2)
        
        Returns:
            Path to created file
        """
        self.logger.info(f"Exporting graph to JSON: {filepath}")
        
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'metadata': {
                'format_version': '1.0',
                'created_at': model.applications[list(model.applications.keys())[0]].last_updated.isoformat() if model.applications and list(model.applications.values())[0].last_updated else None,
                'summary': model.summary()
            },
            'applications': [app.to_dict() for app in model.applications.values()],
            'topics': [topic.to_dict() for topic in model.topics.values()],
            'brokers': [broker.to_dict() for broker in model.brokers.values()],
            'nodes': [node.to_dict() for node in model.nodes.values()],
            'edges': {
                'publishes': [self._edge_to_dict(edge) for edge in model.publishes_edges],
                'subscribes': [self._edge_to_dict(edge) for edge in model.subscribes_edges],
                'routes': [self._edge_to_dict(edge) for edge in model.routes_edges],
                'runs_on': [self._edge_to_dict(edge) for edge in model.runs_on_edges],
                'connects_to': [self._edge_to_dict(edge) for edge in model.connects_to_edges],
                'depends_on': [self._edge_to_dict(edge) for edge in model.depends_on_edges]
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=indent, default=str)
        
        self.logger.info(f"Export complete: {output_path}")
        return output_path
    
    def export_to_csv(self,
                     model: GraphModel,
                     nodes_file: str,
                     edges_file: str,
                     qos_file: Optional[str] = None) -> Tuple[Path, Path, Optional[Path]]:
        """
        Export graph to CSV files
        
        Args:
            model: GraphModel to export
            nodes_file: Output path for nodes CSV
            edges_file: Output path for edges CSV
            qos_file: Optional output path for QoS policies CSV
        
        Returns:
            Tuple of (nodes_path, edges_path, qos_path)
        """
        self.logger.info(f"Exporting graph to CSV: {nodes_file}, {edges_file}")
        
        nodes_path = Path(nodes_file)
        edges_path = Path(edges_file)
        nodes_path.parent.mkdir(parents=True, exist_ok=True)
        edges_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export nodes
        all_nodes = model.get_all_nodes()
        if all_nodes:
            # Get all possible keys
            all_keys = set()
            for node_dict in all_nodes.values():
                all_keys.update(node_dict.keys())
            
            fieldnames = ['name'] + sorted(list(all_keys - {'name'}))
            
            with open(nodes_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for name, attrs in all_nodes.items():
                    row = {'name': name, **attrs}
                    writer.writerow(row)
        
        # Export edges
        all_edges = model.get_all_edges()
        if all_edges:
            # Get all possible keys
            all_keys = set()
            for edge_dict in all_edges:
                all_keys.update(edge_dict.keys())
            
            fieldnames = ['source', 'target', 'type'] + sorted(list(all_keys - {'source', 'target', 'type'}))
            
            with open(edges_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for edge_dict in all_edges:
                    writer.writerow(edge_dict)
        
        # Export QoS policies if requested
        qos_path = None
        if qos_file and model.topics:
            qos_path = Path(qos_file)
            qos_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(qos_path, 'w', newline='') as f:
                fieldnames = ['topic', 'durability', 'reliability', 'deadline_ms', 
                             'lifespan_ms', 'transport_priority', 'history_depth', 'qos_score']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for topic in model.topics.values():
                    topic_dict = topic.to_dict()
                    row = {
                        'topic': topic.name,
                        'durability': topic_dict.get('durability'),
                        'reliability': topic_dict.get('reliability'),
                        'deadline_ms': topic_dict.get('deadline_ms'),
                        'lifespan_ms': topic_dict.get('lifespan_ms'),
                        'transport_priority': topic_dict.get('transport_priority'),
                        'history_depth': topic_dict.get('history_depth'),
                        'qos_score': topic_dict.get('qos_score')
                    }
                    writer.writerow(row)
        
        self.logger.info(f"Export complete: nodes={nodes_path}, edges={edges_path}, qos={qos_path}")
        return nodes_path, edges_path, qos_path
    
    def export_to_graphml(self, model: GraphModel, filepath: str) -> Path:
        """
        Export graph to GraphML format (for Gephi, yEd, etc.)
        
        Args:
            model: GraphModel to export
            filepath: Output file path
        
        Returns:
            Path to created file
        """
        self.logger.info(f"Exporting graph to GraphML: {filepath}")
        
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to NetworkX first
        graph = self.export_to_networkx(model)

        # Remove nodes with None attributes to avoid GraphML issues
        for node in list(graph.nodes()):
            for key, value in dict(graph.nodes[node]).items():
                if value is None:
                    print(f"Removing None attribute '{key}' from node '{node}' for GraphML export")
                    del graph.nodes[node][key]

        for edge in list(graph.edges()):
            for key, value in dict(graph.edges[edge]).items():
                if value is None:
                    print(f"Removing None attribute '{key}' from edge '{edge}' for GraphML export")
                    del graph.edges[edge][key]
    
        # Export to GraphML
        nx.write_graphml(graph, str(output_path))
        
        self.logger.info(f"Export complete: {output_path}")
        return output_path
    
    def export_to_dot(self, model: GraphModel, filepath: str) -> Path:
        """
        Export graph to DOT format (for Graphviz)
        
        Args:
            model: GraphModel to export
            filepath: Output file path
        
        Returns:
            Path to created file
        """
        self.logger.info(f"Exporting graph to DOT: {filepath}")
        
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write('digraph G {\n')
            f.write('  rankdir=TB;\n')
            f.write('  node [shape=box];\n\n')
            
            # Write nodes with styling based on type
            all_nodes = model.get_all_nodes()
            for name, attrs in all_nodes.items():
                node_type = attrs.get('type', 'Unknown')
                
                # Style based on type
                if node_type == 'Application':
                    style = 'shape=box, style=filled, fillcolor=lightblue'
                elif node_type == 'Topic':
                    style = 'shape=ellipse, style=filled, fillcolor=lightgreen'
                elif node_type == 'Broker':
                    style = 'shape=diamond, style=filled, fillcolor=lightyellow'
                elif node_type == 'Node':
                    style = 'shape=box3d, style=filled, fillcolor=lightgray'
                else:
                    style = 'shape=box'
                
                f.write(f'  "{name}" [{style}];\n')
            
            f.write('\n')
            
            # Write edges with styling based on type
            all_edges = model.get_all_edges()
            for edge_dict in all_edges:
                source = edge_dict.get('source')
                target = edge_dict.get('target')
                edge_type = edge_dict.get('type', '')
                
                # Style based on type
                if edge_type == 'PUBLISHES_TO':
                    style = 'color=blue, label="pub"'
                elif edge_type == 'SUBSCRIBES_TO':
                    style = 'color=green, label="sub"'
                elif edge_type == 'ROUTES':
                    style = 'color=orange, label="route"'
                elif edge_type == 'RUNS_ON':
                    style = 'color=gray, style=dashed, label="runs_on"'
                elif edge_type == 'CONNECTS_TO':
                    style = 'color=black, style=dotted, label="connects"'
                elif edge_type == 'DEPENDS_ON':
                    style = 'color=red, style=bold, label="depends"'
                else:
                    style = ''
                
                f.write(f'  "{source}" -> "{target}" [{style}];\n')
            
            f.write('}\n')
        
        self.logger.info(f"Export complete: {output_path}")
        return output_path
    
    def export_layer_to_networkx(self, 
                                 model: GraphModel,
                                 layer: str = 'application') -> nx.DiGraph:
        """
        Export a specific layer to NetworkX
        
        Args:
            model: GraphModel to export
            layer: Layer to export ('application', 'infrastructure', 'topic')
        
        Returns:
            NetworkX directed graph for the specified layer
        """
        self.logger.info(f"Exporting {layer} layer to NetworkX...")
        
        graph = nx.DiGraph()
        
        if layer == 'application':
            # Application layer: applications and their DEPENDS_ON relationships
            for app in model.applications.values():
                graph.add_node(app.name, **app.to_dict())
            
            for edge in model.depends_on_edges:
                if edge.source in model.applications and edge.target in model.applications:
                    graph.add_edge(edge.source, edge.target, **edge.to_dict())
        
        elif layer == 'infrastructure':
            # Infrastructure layer: nodes, brokers, and their connections
            for node in model.nodes.values():
                graph.add_node(node.name, **node.to_dict())
            
            for broker in model.brokers.values():
                graph.add_node(broker.name, **broker.to_dict())
            
            for edge in model.connects_to_edges:
                graph.add_edge(edge.source, edge.target, **edge.to_dict())
            
            # Add RUNS_ON edges to show hosting
            for edge in model.runs_on_edges:
                if edge.target in model.nodes:
                    graph.add_edge(edge.source, edge.target, **edge.to_dict())
        
        elif layer == 'topic':
            # Topic layer: topics and their relationships
            for topic in model.topics.values():
                graph.add_node(topic.name, **topic.to_dict())
            
            for app in model.applications.values():
                graph.add_node(app.name, **app.to_dict())
            
            for edge in model.publishes_edges:
                graph.add_edge(edge.source, edge.target, **edge.to_dict())
            
            for edge in model.subscribes_edges:
                graph.add_edge(edge.source, edge.target, **edge.to_dict())
        
        self.logger.info(f"Export complete: {len(graph)} nodes, {len(graph.edges())} edges")
        return graph
    
    def _create_neo4j_constraints(self, session):
        """Create Neo4j constraints and indexes"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Application) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Topic) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Broker) REQUIRE n.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.name IS UNIQUE"
        ]
        
        for constraint in constraints:
            try:
                session.run(constraint)
            except Exception as e:
                self.logger.warning(f"Could not create constraint: {e}")
    
    def _export_nodes_to_neo4j(self, session, model: GraphModel) -> int:
        """Export all nodes to Neo4j"""
        count = 0
        
        # Export applications
        for app in model.applications.values():
            query = """
                MERGE (n:Application {name: $name})
                SET n += $props
            """
            session.run(query, name=app.name, props=app.to_dict())
            count += 1
        
        # Export topics
        for topic in model.topics.values():
            query = """
                MERGE (n:Topic {name: $name})
                SET n += $props
            """
            session.run(query, name=topic.name, props=topic.to_dict())
            count += 1
        
        # Export brokers
        for broker in model.brokers.values():
            query = """
                MERGE (n:Broker {name: $name})
                SET n += $props
            """
            session.run(query, name=broker.name, props=broker.to_dict())
            count += 1
        
        # Export nodes
        for node in model.nodes.values():
            query = """
                MERGE (n:Node {name: $name})
                SET n += $props
            """
            session.run(query, name=node.name, props=node.to_dict())
            count += 1
        
        return count
    
    def _export_edges_to_neo4j(self, session, model: GraphModel) -> int:
        """Export all edges to Neo4j"""
        count = 0
        
        # Export PUBLISHES_TO edges
        for edge in model.publishes_edges:
            query = """
                MATCH (source {name: $source})
                MATCH (target {name: $target})
                MERGE (source)-[r:PUBLISHES_TO]->(target)
                SET r += $props
            """
            session.run(query, source=edge.source, target=edge.target, props=edge.to_dict())
            count += 1
        
        # Export SUBSCRIBES_TO edges
        for edge in model.subscribes_edges:
            query = """
                MATCH (source {name: $source})
                MATCH (target {name: $target})
                MERGE (source)-[r:SUBSCRIBES_TO]->(target)
                SET r += $props
            """
            session.run(query, source=edge.source, target=edge.target, props=edge.to_dict())
            count += 1
        
        # Export ROUTES edges
        for edge in model.routes_edges:
            query = """
                MATCH (source {name: $source})
                MATCH (target {name: $target})
                MERGE (source)-[r:ROUTES]->(target)
                SET r += $props
            """
            session.run(query, source=edge.source, target=edge.target, props=edge.to_dict())
            count += 1
        
        # Export RUNS_ON edges
        for edge in model.runs_on_edges:
            query = """
                MATCH (source {name: $source})
                MATCH (target {name: $target})
                MERGE (source)-[r:RUNS_ON]->(target)
                SET r += $props
            """
            session.run(query, source=edge.source, target=edge.target, props=edge.to_dict())
            count += 1
        
        # Export CONNECTS_TO edges
        for edge in model.connects_to_edges:
            query = """
                MATCH (source {name: $source})
                MATCH (target {name: $target})
                MERGE (source)-[r:CONNECTS_TO]->(target)
                SET r += $props
            """
            session.run(query, source=edge.source, target=edge.target, props=edge.to_dict())
            count += 1
        
        # Export DEPENDS_ON edges
        for edge in model.depends_on_edges:
            query = """
                MATCH (source {name: $source})
                MATCH (target {name: $target})
                MERGE (source)-[r:DEPENDS_ON]->(target)
                SET r += $props
            """
            session.run(query, source=edge.source, target=edge.target, props=edge.to_dict())
            count += 1
        
        return count
    
    def _edge_to_dict(self, edge) -> Dict:
        """Convert edge to dictionary"""
        edge_dict = edge.to_dict()
        edge_dict['source'] = edge.source
        edge_dict['target'] = edge.target
        return edge_dict
    
    def export_summary(self, model: GraphModel) -> Dict[str, Any]:
        """
        Generate a summary of the graph
        
        Args:
            model: GraphModel to summarize
        
        Returns:
            Dictionary with summary statistics
        """
        summary = model.summary()
        
        # Add additional statistics
        all_nodes = model.get_all_nodes()
        
        # Count by type
        type_counts = {}
        for node_dict in all_nodes.values():
            node_type = node_dict.get('type', 'Unknown')
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        
        summary['node_type_distribution'] = type_counts
        
        # Edge type distribution
        edge_type_counts = {
            'PUBLISHES_TO': len(model.publishes_edges),
            'SUBSCRIBES_TO': len(model.subscribes_edges),
            'ROUTES': len(model.routes_edges),
            'RUNS_ON': len(model.runs_on_edges),
            'CONNECTS_TO': len(model.connects_to_edges),
            'DEPENDS_ON': len(model.depends_on_edges)
        }
        summary['edge_type_distribution'] = edge_type_counts
        
        # QoS statistics
        if model.topics:
            qos_scores = [t.get_qos_criticality() for t in model.topics.values()]
            summary['qos_statistics'] = {
                'avg_score': sum(qos_scores) / len(qos_scores),
                'max_score': max(qos_scores),
                'min_score': min(qos_scores),
                'topics_with_high_qos': sum(1 for s in qos_scores if s > 0.7)
            }
        
        return summary
