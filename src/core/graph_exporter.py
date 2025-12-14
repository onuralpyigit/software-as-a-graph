"""
Graph Exporter - Simplified Version 3.0

Exports GraphModel instances to various formats:
- JSON
- GraphML (for Gephi, yEd, NetworkX)
- GEXF (for Gephi, Sigma.js)
- DOT/GraphViz
- CSV (vertices and edges)
- Pickle (Python serialization)
- Cypher (Neo4j import script)
- NetworkX (direct graph object)

Author: Research Team
Version: 3.0
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

from .graph_model import GraphModel, QoSPolicy


class GraphExporter:
    """
    Exports GraphModel instances to various formats
    """
    
    def __init__(self):
        """Initialize the graph exporter"""
        self.logger = logging.getLogger(__name__)
    
    def export_to_json(self, model: GraphModel, filepath: str, indent: int = 2) -> str:
        """Export GraphModel to JSON file"""
        self.logger.info(f"Exporting to JSON: {filepath}")
        
        data = model.to_dict()
        data['_export'] = {
            'format': 'json',
            'exported_at': datetime.utcnow().isoformat() + 'Z',
            'exporter_version': '3.0'
        }
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, default=str)
        
        return str(path)
    
    def export_to_pickle(self, model: GraphModel, filepath: str) -> str:
        """Export GraphModel to pickle file"""
        self.logger.info(f"Exporting to pickle: {filepath}")
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        
        return str(path)
    
    def load_from_pickle(self, filepath: str) -> GraphModel:
        """Load GraphModel from pickle file"""
        self.logger.info(f"Loading from pickle: {filepath}")
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def export_to_graphml(self, model: GraphModel, filepath: str) -> str:
        """Export GraphModel to GraphML format"""
        self.logger.info(f"Exporting to GraphML: {filepath}")
        
        try:
            import networkx as nx
            G = self._to_networkx(model)
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            nx.write_graphml(G, str(path))
            return str(path)
        except ImportError:
            return self._export_graphml_manual(model, filepath)
    
    def _export_graphml_manual(self, model: GraphModel, filepath: str) -> str:
        """Manually generate GraphML without NetworkX"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')
            
            # Define attributes
            f.write('  <key id="label" for="node" attr.name="label" attr.type="string"/>\n')
            f.write('  <key id="type" for="node" attr.name="type" attr.type="string"/>\n')
            f.write('  <key id="role" for="node" attr.name="role" attr.type="string"/>\n')
            f.write('  <key id="size" for="node" attr.name="size" attr.type="int"/>\n')
            f.write('  <key id="edge_type" for="edge" attr.name="edge_type" attr.type="string"/>\n')
            
            f.write('  <graph id="G" edgedefault="directed">\n')
            
            # Write vertices
            for app_id, app in model.applications.items():
                f.write(f'    <node id="{self._escape_xml(app_id)}">\n')
                f.write(f'      <data key="label">{self._escape_xml(app.name)}</data>\n')
                f.write(f'      <data key="type">APPLICATION</data>\n')
                f.write(f'      <data key="role">{self._escape_xml(app.role)}</data>\n')
                f.write('    </node>\n')
            
            for broker_id, broker in model.brokers.items():
                f.write(f'    <node id="{self._escape_xml(broker_id)}">\n')
                f.write(f'      <data key="label">{self._escape_xml(broker.name)}</data>\n')
                f.write(f'      <data key="type">BROKER</data>\n')
                f.write('    </node>\n')
            
            for topic_id, topic in model.topics.items():
                f.write(f'    <node id="{self._escape_xml(topic_id)}">\n')
                f.write(f'      <data key="label">{self._escape_xml(topic.name)}</data>\n')
                f.write(f'      <data key="type">TOPIC</data>\n')
                f.write(f'      <data key="size">{topic.size}</data>\n')
                f.write('    </node>\n')
            
            for node_id, node in model.nodes.items():
                f.write(f'    <node id="{self._escape_xml(node_id)}">\n')
                f.write(f'      <data key="label">{self._escape_xml(node.name)}</data>\n')
                f.write(f'      <data key="type">NODE</data>\n')
                f.write('    </node>\n')
            
            # Write edges
            for i, edge in enumerate(model.edges):
                f.write(f'    <edge id="e{i}" source="{self._escape_xml(edge.source)}" target="{self._escape_xml(edge.target)}">\n')
                f.write(f'      <data key="edge_type">{self._escape_xml(edge.edge_type)}</data>\n')
                f.write('    </edge>\n')
            
            f.write('  </graph>\n')
            f.write('</graphml>\n')
        
        return str(path)
    
    def export_to_gexf(self, model: GraphModel, filepath: str) -> str:
        """Export GraphModel to GEXF format"""
        self.logger.info(f"Exporting to GEXF: {filepath}")
        
        try:
            import networkx as nx
            G = self._to_networkx(model)
            path = Path(filepath)
            path.parent.mkdir(parents=True, exist_ok=True)
            nx.write_gexf(G, str(path))
            return str(path)
        except ImportError:
            return self._export_gexf_manual(model, filepath)
    
    def _export_gexf_manual(self, model: GraphModel, filepath: str) -> str:
        """Manually generate GEXF without NetworkX"""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.utcnow().isoformat()[:10]
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">\n')
            f.write(f'  <meta lastmodifieddate="{timestamp}">\n')
            f.write('    <creator>Software-as-a-Graph Exporter v3.0</creator>\n')
            f.write('  </meta>\n')
            f.write('  <graph mode="static" defaultedgetype="directed">\n')
            
            # Attributes
            f.write('    <attributes class="node">\n')
            f.write('      <attribute id="0" title="type" type="string"/>\n')
            f.write('      <attribute id="1" title="role" type="string"/>\n')
            f.write('      <attribute id="2" title="size" type="integer"/>\n')
            f.write('    </attributes>\n')
            
            # Nodes
            f.write('    <nodes>\n')
            for app_id, app in model.applications.items():
                f.write(f'      <node id="{self._escape_xml(app_id)}" label="{self._escape_xml(app.name)}">\n')
                f.write(f'        <attvalues><attvalue for="0" value="APPLICATION"/><attvalue for="1" value="{app.role}"/></attvalues>\n')
                f.write('      </node>\n')
            
            for broker_id, broker in model.brokers.items():
                f.write(f'      <node id="{self._escape_xml(broker_id)}" label="{self._escape_xml(broker.name)}">\n')
                f.write('        <attvalues><attvalue for="0" value="BROKER"/></attvalues>\n')
                f.write('      </node>\n')
            
            for topic_id, topic in model.topics.items():
                f.write(f'      <node id="{self._escape_xml(topic_id)}" label="{self._escape_xml(topic.name)}">\n')
                f.write(f'        <attvalues><attvalue for="0" value="TOPIC"/><attvalue for="2" value="{topic.size}"/></attvalues>\n')
                f.write('      </node>\n')
            
            for node_id, node in model.nodes.items():
                f.write(f'      <node id="{self._escape_xml(node_id)}" label="{self._escape_xml(node.name)}">\n')
                f.write('        <attvalues><attvalue for="0" value="NODE"/></attvalues>\n')
                f.write('      </node>\n')
            
            f.write('    </nodes>\n')
            
            # Edges
            f.write('    <edges>\n')
            for i, edge in enumerate(model.edges):
                f.write(f'      <edge id="{i}" source="{self._escape_xml(edge.source)}" target="{self._escape_xml(edge.target)}" label="{edge.edge_type}"/>\n')
            f.write('    </edges>\n')
            
            f.write('  </graph>\n')
            f.write('</gexf>\n')
        
        return str(path)
    
    def export_to_dot(self, model: GraphModel, filepath: str) -> str:
        """Export GraphModel to DOT/GraphViz format"""
        self.logger.info(f"Exporting to DOT: {filepath}")
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Colors and shapes
        styles = {
            'APPLICATION': ('component', '#8E24AA'),
            'BROKER': ('hexagon', '#43A047'),
            'TOPIC': ('ellipse', '#FB8C00'),
            'NODE': ('box3d', '#1E88E5')
        }
        
        edge_colors = {
            'PUBLISHES_TO': '#E53935',
            'SUBSCRIBES_TO': '#1E88E5',
            'ROUTES': '#43A047',
            'RUNS_ON': '#757575',
            'CONNECTS_TO': '#9E9E9E'
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write('digraph PubSubSystem {\n')
            f.write('  rankdir=LR;\n')
            f.write('  node [fontname="Arial", fontsize=10];\n')
            f.write('  edge [fontname="Arial", fontsize=8];\n\n')
            
            # Applications
            f.write('  subgraph cluster_apps {\n')
            f.write('    label="Applications"; style=dashed; color=gray;\n')
            for app_id, app in model.applications.items():
                shape, color = styles['APPLICATION']
                label = f"{app.name}\\n[{app.role}]"
                f.write(f'    "{app_id}" [label="{label}", shape={shape}, fillcolor="{color}", style=filled, fontcolor=white];\n')
            f.write('  }\n\n')
            
            # Brokers
            f.write('  subgraph cluster_brokers {\n')
            f.write('    label="Brokers"; style=dashed; color=gray;\n')
            for broker_id, broker in model.brokers.items():
                shape, color = styles['BROKER']
                f.write(f'    "{broker_id}" [label="{broker.name}", shape={shape}, fillcolor="{color}", style=filled, fontcolor=white];\n')
            f.write('  }\n\n')
            
            # Topics
            f.write('  subgraph cluster_topics {\n')
            f.write('    label="Topics"; style=dashed; color=gray;\n')
            for topic_id, topic in model.topics.items():
                shape, color = styles['TOPIC']
                label = topic.name[:25] + '...' if len(topic.name) > 25 else topic.name
                f.write(f'    "{topic_id}" [label="{label}", shape={shape}, fillcolor="{color}", style=filled];\n')
            f.write('  }\n\n')
            
            # Nodes
            f.write('  subgraph cluster_nodes {\n')
            f.write('    label="Infrastructure"; style=dashed; color=gray;\n')
            for node_id, node in model.nodes.items():
                shape, color = styles['NODE']
                f.write(f'    "{node_id}" [label="{node.name}", shape={shape}, fillcolor="{color}", style=filled, fontcolor=white];\n')
            f.write('  }\n\n')
            
            # Edges
            for edge in model.edges:
                color = edge_colors.get(edge.edge_type, 'black')
                style = 'dashed' if edge.edge_type == 'SUBSCRIBES_TO' else 'solid'
                f.write(f'  "{edge.source}" -> "{edge.target}" [color="{color}", style={style}];\n')
            
            f.write('}\n')
        
        return str(path)
    
    def export_to_csv(self, model: GraphModel, output_dir: str, prefix: str = '') -> Dict[str, str]:
        """Export GraphModel to CSV files"""
        import csv
        
        self.logger.info(f"Exporting to CSV: {output_dir}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        prefix = f"{prefix}_" if prefix else ""
        exported = {}
        
        # Export vertices
        vertices_file = output_path / f"{prefix}vertices.csv"
        with open(vertices_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'name', 'type', 'attributes'])
            
            for app_id, app in model.applications.items():
                writer.writerow([app_id, app.name, 'APPLICATION', json.dumps({'role': app.role})])
            
            for broker_id, broker in model.brokers.items():
                writer.writerow([broker_id, broker.name, 'BROKER', '{}'])
            
            for topic_id, topic in model.topics.items():
                attrs = {'size': topic.size, 'qos': topic.qos.to_dict() if hasattr(topic.qos, 'to_dict') else topic.qos}
                writer.writerow([topic_id, topic.name, 'TOPIC', json.dumps(attrs)])
            
            for node_id, node in model.nodes.items():
                writer.writerow([node_id, node.name, 'NODE', '{}'])
        
        exported['vertices'] = str(vertices_file)
        
        # Export edges
        edges_file = output_path / f"{prefix}edges.csv"
        with open(edges_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['from', 'to', 'type'])
            
            for edge in model.edges:
                writer.writerow([edge.source, edge.target, edge.edge_type])
        
        exported['edges'] = str(edges_file)
        
        return exported
    
    def export_to_cypher(self, model: GraphModel, filepath: str) -> str:
        """Export GraphModel to Cypher script for Neo4j import"""
        self.logger.info(f"Exporting to Cypher: {filepath}")
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write("// Neo4j Import Script - Generated by Software-as-a-Graph\n")
            f.write(f"// Generated at: {datetime.utcnow().isoformat()}Z\n\n")
            
            # Constraints
            f.write("// Create constraints\n")
            f.write("CREATE CONSTRAINT app_id IF NOT EXISTS FOR (a:Application) REQUIRE a.id IS UNIQUE;\n")
            f.write("CREATE CONSTRAINT broker_id IF NOT EXISTS FOR (b:Broker) REQUIRE b.id IS UNIQUE;\n")
            f.write("CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.id IS UNIQUE;\n")
            f.write("CREATE CONSTRAINT node_id IF NOT EXISTS FOR (n:Node) REQUIRE n.id IS UNIQUE;\n\n")
            
            # Applications
            f.write("// Create Applications\n")
            for app in model.applications.values():
                f.write(f"MERGE (a:Application {{id: '{app.id}'}}) SET a.name = '{self._escape_cypher(app.name)}', a.role = '{app.role}';\n")
            f.write("\n")
            
            # Brokers
            f.write("// Create Brokers\n")
            for broker in model.brokers.values():
                f.write(f"MERGE (b:Broker {{id: '{broker.id}'}}) SET b.name = '{self._escape_cypher(broker.name)}';\n")
            f.write("\n")
            
            # Topics
            f.write("// Create Topics\n")
            for topic in model.topics.values():
                qos = topic.qos.to_dict() if isinstance(topic.qos, QoSPolicy) else topic.qos
                f.write(f"MERGE (t:Topic {{id: '{topic.id}'}}) SET t.name = '{self._escape_cypher(topic.name)}', ")
                f.write(f"t.size = {topic.size}, t.durability = '{qos.get('durability', 'VOLATILE')}', ")
                f.write(f"t.reliability = '{qos.get('reliability', 'BEST_EFFORT')}', ")
                f.write(f"t.transport_priority = '{qos.get('transport_priority', 'MEDIUM')}';\n")
            f.write("\n")
            
            # Nodes
            f.write("// Create Nodes\n")
            for node in model.nodes.values():
                f.write(f"MERGE (n:Node {{id: '{node.id}'}}) SET n.name = '{self._escape_cypher(node.name)}';\n")
            f.write("\n")
            
            # Edges
            edge_templates = {
                'PUBLISHES_TO': "MATCH (a:Application {{id: '{from_id}'}}), (t:Topic {{id: '{to_id}'}}) MERGE (a)-[:PUBLISHES_TO]->(t);",
                'SUBSCRIBES_TO': "MATCH (a:Application {{id: '{from_id}'}}), (t:Topic {{id: '{to_id}'}}) MERGE (a)-[:SUBSCRIBES_TO]->(t);",
                'ROUTES': "MATCH (b:Broker {{id: '{from_id}'}}), (t:Topic {{id: '{to_id}'}}) MERGE (b)-[:ROUTES]->(t);",
                'RUNS_ON': "MATCH (c {{id: '{from_id}'}}), (n:Node {{id: '{to_id}'}}) WHERE c:Application OR c:Broker MERGE (c)-[:RUNS_ON]->(n);",
                'CONNECTS_TO': "MATCH (n1:Node {{id: '{from_id}'}}), (n2:Node {{id: '{to_id}'}}) MERGE (n1)-[:CONNECTS_TO]->(n2);"
            }
            
            for edge_type, template in edge_templates.items():
                edges = model.get_edges_by_type(edge_type)
                if edges:
                    f.write(f"// Create {edge_type} relationships\n")
                    for edge in edges:
                        f.write(template.format(from_id=edge.source, to_id=edge.target) + "\n")
                    f.write("\n")
        
        return str(path)
    
    def export_to_networkx(self, model: GraphModel) -> Any:
        """Export GraphModel to NetworkX DiGraph"""
        return self._to_networkx(model)
    
    def _to_networkx(self, model: GraphModel) -> Any:
        """Convert model to NetworkX graph"""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("networkx package required. Install with: pip install networkx")
        
        G = nx.DiGraph()
        
        # Add vertices
        for app_id, app in model.applications.items():
            G.add_node(app_id, label=app.name, type='APPLICATION', role=app.role)
        
        for broker_id, broker in model.brokers.items():
            G.add_node(broker_id, label=broker.name, type='BROKER')
        
        for topic_id, topic in model.topics.items():
            qos = topic.qos.to_dict() if hasattr(topic.qos, 'to_dict') else topic.qos
            G.add_node(topic_id, label=topic.name, type='TOPIC', size=topic.size, **qos)
        
        for node_id, node in model.nodes.items():
            G.add_node(node_id, label=node.name, type='NODE')
        
        # Add edges
        for edge in model.edges:
            G.add_edge(edge.source, edge.target, edge_type=edge.edge_type)
        
        return G
    
    def _escape_xml(self, text: str) -> str:
        """Escape XML special characters"""
        if not isinstance(text, str):
            text = str(text)
        return (text
                .replace('&', '&amp;')
                .replace('<', '&lt;')
                .replace('>', '&gt;')
                .replace('"', '&quot;')
                .replace("'", '&apos;'))
    
    def _escape_cypher(self, text: str) -> str:
        """Escape Cypher string special characters"""
        if not isinstance(text, str):
            text = str(text)
        return text.replace("'", "\\'").replace("\\", "\\\\")