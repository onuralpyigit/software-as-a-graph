# Integrated Graph Analysis System for PhD Thesis
# Combines all components into a unified analysis framework

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import json
import yaml
from scipy.stats import skew, kurtosis, normaltest
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class IntegratedPubSubAnalyzer:
    """
    Complete integrated system for analyzing distributed publish-subscribe systems
    Combines all analysis components into a unified framework
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the integrated analyzer
        
        Args:
            config_path: Path to configuration file (YAML/JSON)
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path) if config_path else self._default_config()
        
        # Initialize components (assuming you have these implemented)
        self.graph = None
        self.analysis_results = {}
        self.recommendations = []
        self.validation_results = {}
        
        # Output directory
        self.output_dir = Path(self.config.get('output_dir', 'analysis_output'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis timestamp
        self.analysis_timestamp = datetime.now()
        
    def _default_config(self) -> Dict:
        """Default configuration for the analyzer"""
        return {
            'output_dir': 'analysis_output',
            'weights': {
                'centrality': {
                    'degree': 0.15,
                    'betweenness': 0.20,
                    'closeness': 0.15,
                    'pagerank': 0.15,
                    'eigenvector': 0.15,
                    'articulation': 0.20
                },
                'qos': {
                    'durability': 0.20,
                    'reliability': 0.25,
                    'priority': 0.15,
                    'deadline': 0.20,
                    'lifespan': 0.10,
                    'history': 0.10
                }
            },
            'thresholds': {
                'criticality': {
                    'very_high': 0.9,
                    'high': 0.75,
                    'medium': 0.5,
                    'low': 0.25
                }
            },
            'analysis': {
                'enable_ml_features': True,
                'enable_cascade_analysis': True,
                'enable_flow_analysis': True,
                'enable_visualization': True
            }
        }
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        path = Path(config_path)
        if path.suffix == '.yaml' or path.suffix == '.yml':
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        elif path.suffix == '.json':
            with open(path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
    
    def load_graph_from_neo4j(self, connection_params: Dict) -> nx.DiGraph:
        """
        Load graph from Neo4j database
        
        Args:
            connection_params: Neo4j connection parameters
            
        Returns:
            NetworkX directed graph
        """
        try:
            from neo4j import GraphDatabase
            
            driver = GraphDatabase.driver(
                connection_params['uri'],
                auth=(connection_params['user'], connection_params['password'])
            )
            
            with driver.session() as session:
                # Query to get all nodes
                nodes_query = """
                MATCH (n)
                RETURN n.id AS id, labels(n) AS labels, properties(n) AS props
                """
                nodes = session.run(nodes_query)
                
                # Query to get all relationships
                rels_query = """
                MATCH (n)-[r]->(m)
                RETURN n.id AS source, m.id AS target, type(r) AS type, properties(r) AS props
                """
                relationships = session.run(rels_query)
                
                # Build graph
                G = nx.DiGraph()
                
                for node in nodes:
                    G.add_node(
                        node['id'],
                        labels=node['labels'],
                        **node['props']
                    )
                
                for rel in relationships:
                    G.add_edge(
                        rel['source'],
                        rel['target'],
                        type=rel['type'],
                        **rel['props']
                    )
                
            driver.close()
            self.graph = G
            self.logger.info(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            return G
            
        except ImportError:
            self.logger.error("Neo4j driver not installed. Please install neo4j-python-driver")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load graph from Neo4j: {e}")
            raise
    
    def load_graph_from_file(self, filepath: str, format: str = 'graphml') -> nx.DiGraph:
        """
        Load graph from file
        
        Args:
            filepath: Path to graph file
            format: File format (graphml, gexf, json, pickle)
            
        Returns:
            NetworkX directed graph
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Graph file not found: {filepath}")
        
        if format == 'graphml':
            G = nx.read_graphml(path)
        elif format == 'gexf':
            G = nx.read_gexf(path)
        elif format == 'json':
            with open(path, 'r') as f:
                data = json.load(f)
                G = nx.node_link_graph(data)
        elif format == 'pickle':
            G = nx.read_gpickle(path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.graph = G.to_directed() if not G.is_directed() else G
        self.logger.info(f"Loaded graph from {filepath}")
        return self.graph
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """
        Execute complete analysis pipeline
        
        Returns:
            Comprehensive analysis results
        """
        if self.graph is None:
            raise ValueError("No graph loaded. Please load a graph first.")
        
        self.logger.info("="*60)
        self.logger.info("Starting Complete Graph Analysis Pipeline")
        self.logger.info("="*60)
        
        # Step 1: Basic Graph Statistics
        self.logger.info("Step 1: Computing basic graph statistics...")
        self.analysis_results['statistics'] = self._compute_graph_statistics()
        
        # Step 2: Centrality Analysis
        self.logger.info("Step 2: Computing centrality metrics...")
        self.analysis_results['centralities'] = self._compute_centralities()
        
        # Step 3: Vulnerability Analysis
        self.logger.info("Step 3: Analyzing structural vulnerabilities...")
        self.analysis_results['vulnerabilities'] = self._analyze_vulnerabilities()
        
        # Step 4: QoS Analysis
        self.logger.info("Step 4: Performing QoS-aware analysis...")
        self.analysis_results['qos'] = self._analyze_qos()
        
        # Step 5: Criticality Scoring
        self.logger.info("Step 5: Computing composite criticality scores...")
        self.analysis_results['criticality'] = self._compute_criticality_scores()
        
        # Step 6: Flow Analysis
        if self.config['analysis']['enable_flow_analysis']:
            self.logger.info("Step 6: Analyzing message flows...")
            self.analysis_results['flows'] = self._analyze_message_flows()
        
        # Step 7: Cascade Risk Analysis
        if self.config['analysis']['enable_cascade_analysis']:
            self.logger.info("Step 7: Analyzing cascade risks...")
            self.analysis_results['cascade'] = self._analyze_cascade_risks()
        
        # Step 8: Generate Recommendations
        self.logger.info("Step 8: Generating recommendations...")
        self.recommendations = self._generate_recommendations()
        
        # Step 9: ML Feature Generation
        if self.config['analysis']['enable_ml_features']:
            self.logger.info("Step 9: Generating ML features...")
            self.analysis_results['ml_features'] = self._generate_ml_features()
        
        # Step 10: Visualization
        if self.config['analysis']['enable_visualization']:
            self.logger.info("Step 10: Creating visualizations...")
            self._create_visualizations()
        
        # Save results
        self._save_results()
        
        self.logger.info("="*60)
        self.logger.info("Analysis Complete!")
        self.logger.info("="*60)
        
        return self.analysis_results
    
    def _compute_graph_statistics(self) -> Dict:
        """Compute basic graph statistics"""
        G = self.graph
        
        stats = {
            'nodes': {
                'total': G.number_of_nodes(),
                'by_type': {}
            },
            'edges': {
                'total': G.number_of_edges(),
                'by_type': {}
            },
            'density': nx.density(G),
            'is_connected': nx.is_weakly_connected(G),
            'number_of_components': nx.number_weakly_connected_components(G),
            'diameter': None,
            'average_degree': np.mean([d for n, d in G.degree()]),
            'average_clustering': nx.average_clustering(G.to_undirected())
        }
        
        # Count nodes by type
        for node, attrs in G.nodes(data=True):
            node_type = attrs.get('type', 'Unknown')
            stats['nodes']['by_type'][node_type] = stats['nodes']['by_type'].get(node_type, 0) + 1
        
        # Count edges by type
        for u, v, attrs in G.edges(data=True):
            edge_type = attrs.get('type', 'Unknown')
            stats['edges']['by_type'][edge_type] = stats['edges']['by_type'].get(edge_type, 0) + 1
        
        # Calculate diameter if connected
        if nx.is_weakly_connected(G):
            largest_cc = G.subgraph(max(nx.weakly_connected_components(G), key=len))
            stats['diameter'] = nx.diameter(largest_cc.to_undirected())
        
        return stats
    
    def _compute_centralities(self) -> Dict:
        """Compute all centrality metrics"""
        G = self.graph
        
        centralities = {
            'degree': nx.degree_centrality(G),
            'in_degree': nx.in_degree_centrality(G),
            'out_degree': nx.out_degree_centrality(G),
            'betweenness': nx.betweenness_centrality(G),
            'closeness': nx.closeness_centrality(G),
            'pagerank': nx.pagerank(G)
        }
        
        try:
            centralities['eigenvector'] = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            self.logger.warning("Eigenvector centrality computation failed, using degree as fallback")
            centralities['eigenvector'] = centralities['degree']
        
        return centralities
    
    def _analyze_vulnerabilities(self) -> Dict:
        """Analyze structural vulnerabilities"""
        G = self.graph
        G_undirected = G.to_undirected()
        
        vulnerabilities = {
            'articulation_points': list(nx.articulation_points(G_undirected)),
            'bridges': list(nx.bridges(G_undirected)),
            'node_connectivity': nx.node_connectivity(G_undirected) if nx.is_connected(G_undirected) else 0,
            'edge_connectivity': nx.edge_connectivity(G_undirected) if nx.is_connected(G_undirected) else 0,
            'single_points_of_failure': []
        }
        
        # Identify single points of failure
        for node in vulnerabilities['articulation_points']:
            node_type = G.nodes[node].get('type', 'Unknown')
            if node_type in ['Broker', 'Node']:
                vulnerabilities['single_points_of_failure'].append({
                    'node': node,
                    'type': node_type,
                    'risk': 'HIGH'
                })
        
        return vulnerabilities
    
    def _analyze_qos(self) -> Dict:
        """Analyze QoS aspects"""
        # This would integrate with your QosAwareComponentAnalyzer
        qos_results = {
            'critical_topics': [],
            'high_reliability_requirements': [],
            'strict_deadline_components': []
        }
        
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get('type') == 'Topic':
                # Check for QoS policies
                if 'qos_policy' in attrs:
                    policy = attrs['qos_policy']
                    
                    # Evaluate criticality based on QoS
                    criticality_score = self._evaluate_qos_criticality(policy)
                    
                    if criticality_score > 0.7:
                        qos_results['critical_topics'].append({
                            'topic': node,
                            'score': criticality_score
                        })
        
        return qos_results
    
    def _evaluate_qos_criticality(self, policy) -> float:
        """Evaluate QoS policy to determine criticality"""
        # Simple weighted scoring - you can make this more sophisticated
        weights = self.config['weights']['qos']
        
        score = 0.0
        if hasattr(policy, 'to_dict'):
            policy_dict = policy.to_dict()
            for attr, weight in weights.items():
                score += weight * policy_dict.get(attr, 0)
        
        return score
    
    def _compute_criticality_scores(self) -> pd.DataFrame:
        """Compute composite criticality scores for all components"""
        results = []
        
        centralities = self.analysis_results['centralities']
        vulnerabilities = self.analysis_results['vulnerabilities']
        
        for node in self.graph.nodes():
            # Aggregate centrality scores
            centrality_score = 0.0
            weights = self.config['weights']['centrality']
            
            centrality_score += weights['degree'] * centralities['degree'].get(node, 0)
            centrality_score += weights['betweenness'] * centralities['betweenness'].get(node, 0)
            centrality_score += weights['closeness'] * centralities['closeness'].get(node, 0)
            centrality_score += weights['pagerank'] * centralities['pagerank'].get(node, 0)
            centrality_score += weights['eigenvector'] * centralities['eigenvector'].get(node, 0)
            
            # Add articulation point weight
            if node in vulnerabilities['articulation_points']:
                centrality_score += weights['articulation']
            
            # Classify criticality
            level = self._classify_criticality_level(centrality_score)
            
            results.append({
                'node': node,
                'type': self.graph.nodes[node].get('type', 'Unknown'),
                'criticality_score': centrality_score,
                'criticality_level': level,
                'is_articulation_point': node in vulnerabilities['articulation_points'],
                'degree': self.graph.degree(node)
            })
        
        df = pd.DataFrame(results)
        return df.sort_values('criticality_score', ascending=False)
    
    def _classify_criticality_level(self, score: float) -> str:
        """Classify criticality level based on score"""
        thresholds = self.config['thresholds']['criticality']
        
        if score >= thresholds['very_high']:
            return 'VERY_HIGH'
        elif score >= thresholds['high']:
            return 'HIGH'
        elif score >= thresholds['medium']:
            return 'MEDIUM'
        elif score >= thresholds['low']:
            return 'LOW'
        else:
            return 'VERY_LOW'
    
    def _analyze_message_flows(self) -> Dict:
        """Analyze message flows in the pub-sub system"""
        flows = {
            'topic_flows': {},
            'critical_paths': [],
            'bottlenecks': []
        }
        
        # Analyze each topic
        for node, attrs in self.graph.nodes(data=True):
            if attrs.get('type') == 'Topic':
                publishers = [n for n, m in self.graph.in_edges(node) 
                            if self.graph.edges[n, m].get('type') == 'PUBLISHES_TO']
                subscribers = [m for n, m in self.graph.out_edges(node)
                             if self.graph.edges[n, m].get('type') == 'SUBSCRIBES_TO']
                
                flow_info = {
                    'publishers': publishers,
                    'subscribers': subscribers,
                    'fan_out': len(subscribers) / max(len(publishers), 1),
                    'is_bottleneck': len(publishers) > 3 and len(subscribers) > 5
                }
                
                flows['topic_flows'][node] = flow_info
                
                if flow_info['is_bottleneck']:
                    flows['bottlenecks'].append(node)
        
        return flows
    
    def _analyze_cascade_risks(self) -> List[Dict]:
        """Analyze cascade failure risks"""
        cascade_risks = []
        
        # Get top critical components
        critical_df = self.analysis_results['criticality']
        top_critical = critical_df[critical_df['criticality_level'].isin(['HIGH', 'VERY_HIGH'])]
        
        for _, row in top_critical.iterrows():
            node = row['node']
            
            # Analyze impact of node failure
            impact = self._simulate_node_failure(node)
            
            cascade_risks.append({
                'node': node,
                'type': row['type'],
                'direct_impact': impact['direct'],
                'indirect_impact': impact['indirect'],
                'total_impact': impact['total'],
                'risk_level': 'HIGH' if impact['total'] > 0.3 else 'MEDIUM'
            })
        
        return sorted(cascade_risks, key=lambda x: x['total_impact'], reverse=True)
    
    def _simulate_node_failure(self, node: str) -> Dict:
        """Simulate failure of a node and calculate impact"""
        G_copy = self.graph.copy()
        
        # Count affected connections
        direct_impact = len(list(G_copy.neighbors(node)))
        
        # Remove node and check connectivity
        G_copy.remove_node(node)
        
        # Calculate indirect impact
        if nx.is_weakly_connected(self.graph):
            was_connected = True
            is_connected = nx.is_weakly_connected(G_copy)
        else:
            was_connected = False
            is_connected = False
        
        indirect_impact = 0
        if was_connected and not is_connected:
            # Graph got disconnected
            components = list(nx.weakly_connected_components(G_copy))
            indirect_impact = len(components) - 1
        
        total_nodes = self.graph.number_of_nodes()
        
        return {
            'direct': direct_impact / total_nodes,
            'indirect': indirect_impact / total_nodes,
            'total': (direct_impact + indirect_impact) / total_nodes
        }
    
    def _generate_recommendations(self) -> List[Dict]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Based on articulation points
        if self.analysis_results['vulnerabilities']['articulation_points']:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Redundancy',
                'issue': f"Found {len(self.analysis_results['vulnerabilities']['articulation_points'])} single points of failure",
                'recommendation': 'Add redundant components for articulation points',
                'affected_components': self.analysis_results['vulnerabilities']['articulation_points'][:5]
            })
        
        # Based on bottlenecks
        if 'flows' in self.analysis_results and self.analysis_results['flows']['bottlenecks']:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Performance',
                'issue': f"Identified {len(self.analysis_results['flows']['bottlenecks'])} potential bottleneck topics",
                'recommendation': 'Consider partitioning or load balancing for high-traffic topics',
                'affected_components': self.analysis_results['flows']['bottlenecks']
            })
        
        # Based on connectivity
        if self.analysis_results['vulnerabilities']['node_connectivity'] < 2:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Reliability',
                'issue': 'Low node connectivity detected',
                'recommendation': 'Increase redundant connections between critical components',
                'affected_components': []
            })
        
        return recommendations
    
    def _generate_ml_features(self) -> np.ndarray:
        """Generate ML features for all nodes"""
        features_list = []
        
        centralities = self.analysis_results['centralities']
        
        for node in self.graph.nodes():
            features = []
            
            # Centrality features
            for metric in ['degree', 'betweenness', 'closeness', 'pagerank', 'eigenvector']:
                features.append(centralities[metric].get(node, 0))
            
            # Structural features
            features.append(self.graph.in_degree(node) / self.graph.number_of_nodes())
            features.append(self.graph.out_degree(node) / self.graph.number_of_nodes())
            
            # Vulnerability features
            features.append(1.0 if node in self.analysis_results['vulnerabilities']['articulation_points'] else 0.0)
            
            features_list.append(features)
        
        return np.array(features_list)
    
    def _create_visualizations(self):
        """Create comprehensive visualizations"""
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Criticality Distribution
        ax1 = plt.subplot(3, 3, 1)
        critical_df = self.analysis_results['criticality']
        critical_df['criticality_score'].hist(bins=20, ax=ax1, edgecolor='black')
        ax1.set_title('Criticality Score Distribution')
        ax1.set_xlabel('Score')
        ax1.set_ylabel('Count')
        
        # 2. Component Type Distribution
        ax2 = plt.subplot(3, 3, 2)
        type_counts = critical_df['type'].value_counts()
        type_counts.plot(kind='bar', ax=ax2)
        ax2.set_title('Components by Type')
        ax2.set_xlabel('Type')
        ax2.set_ylabel('Count')
        
        # 3. Criticality Level Distribution
        ax3 = plt.subplot(3, 3, 3)
        level_counts = critical_df['criticality_level'].value_counts()
        colors = {'VERY_HIGH': 'red', 'HIGH': 'orange', 'MEDIUM': 'yellow', 
                 'LOW': 'lightgreen', 'VERY_LOW': 'green'}
        level_colors = [colors.get(level, 'gray') for level in level_counts.index]
        level_counts.plot(kind='pie', ax=ax3, colors=level_colors, autopct='%1.1f%%')
        ax3.set_title('Criticality Level Distribution')
        
        # 4. Top 10 Critical Components
        ax4 = plt.subplot(3, 3, 4)
        top_10 = critical_df.head(10)
        ax4.barh(range(len(top_10)), top_10['criticality_score'].values)
        ax4.set_yticks(range(len(top_10)))
        ax4.set_yticklabels(top_10['node'].values)
        ax4.set_title('Top 10 Critical Components')
        ax4.set_xlabel('Criticality Score')
        
        # 5. Centrality Correlations
        ax5 = plt.subplot(3, 3, 5)
        centrality_data = pd.DataFrame(self.analysis_results['centralities'])
        correlation = centrality_data.corr()
        sns.heatmap(correlation, annot=True, fmt='.2f', ax=ax5, cmap='coolwarm')
        ax5.set_title('Centrality Metrics Correlation')
        
        # 6. Box plot of criticality by type
        ax6 = plt.subplot(3, 3, 6)
        critical_df.boxplot(column='criticality_score', by='type', ax=ax6)
        ax6.set_title('Criticality by Component Type')
        ax6.set_xlabel('Type')
        ax6.set_ylabel('Criticality Score')
        
        # 7. Cascade Risk
        if 'cascade' in self.analysis_results:
            ax7 = plt.subplot(3, 3, 7)
            cascade_df = pd.DataFrame(self.analysis_results['cascade'])
            if not cascade_df.empty:
                cascade_df = cascade_df.head(10)
                ax7.scatter(cascade_df['direct_impact'], 
                          cascade_df['indirect_impact'],
                          s=cascade_df['total_impact']*1000,
                          alpha=0.6)
                ax7.set_title('Cascade Risk Analysis')
                ax7.set_xlabel('Direct Impact')
                ax7.set_ylabel('Indirect Impact')
        
        # 8. Network Statistics
        ax8 = plt.subplot(3, 3, 8)
        stats = self.analysis_results['statistics']
        stats_text = f"""
        Nodes: {stats['nodes']['total']}
        Edges: {stats['edges']['total']}
        Density: {stats['density']:.3f}
        Avg Degree: {stats['average_degree']:.2f}
        Avg Clustering: {stats['average_clustering']:.3f}
        Components: {stats['number_of_components']}
        """
        ax8.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        ax8.set_title('Graph Statistics')
        ax8.axis('off')
        
        # 9. Recommendations Summary
        ax9 = plt.subplot(3, 3, 9)
        rec_text = "RECOMMENDATIONS\n" + "-"*20 + "\n"
        for i, rec in enumerate(self.recommendations[:3], 1):
            rec_text += f"{i}. [{rec['priority']}] {rec['category']}\n"
            rec_text += f"   {rec['issue'][:40]}...\n"
        ax9.text(0.1, 0.5, rec_text, fontsize=10, verticalalignment='center')
        ax9.set_title('Top Recommendations')
        ax9.axis('off')
        
        plt.suptitle(f'Publish-Subscribe System Analysis - {self.analysis_timestamp.strftime("%Y-%m-%d %H:%M")}', 
                    fontsize=16)
        plt.tight_layout()
        
        # Save figure
        output_file = self.output_dir / f"analysis_visualization_{self.analysis_timestamp.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualizations saved to {output_file}")
    
    def _save_results(self):
        """Save all analysis results to files"""
        timestamp = self.analysis_timestamp.strftime('%Y%m%d_%H%M%S')
        
        # Save criticality scores as CSV
        if 'criticality' in self.analysis_results:
            csv_file = self.output_dir / f"criticality_scores_{timestamp}.csv"
            self.analysis_results['criticality'].to_csv(csv_file, index=False)
            self.logger.info(f"Criticality scores saved to {csv_file}")
        
        # Save complete analysis as JSON
        json_file = self.output_dir / f"complete_analysis_{timestamp}.json"
        
        # Convert non-serializable objects
        json_results = {}
        for key, value in self.analysis_results.items():
            if key == 'criticality':
                json_results[key] = value.to_dict('records')
            elif key == 'ml_features':
                json_results[key] = value.tolist() if isinstance(value, np.ndarray) else value
            else:
                json_results[key] = value
        
        with open(json_file, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        self.logger.info(f"Complete analysis saved to {json_file}")
        
        # Save recommendations
        rec_file = self.output_dir / f"recommendations_{timestamp}.json"
        with open(rec_file, 'w') as f:
            json.dump(self.recommendations, f, indent=2)
        
        self.logger.info(f"Recommendations saved to {rec_file}")
        
        # Generate and save report
        report = self._generate_report()
        report_file = self.output_dir / f"analysis_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        self.logger.info(f"Report saved to {report_file}")
    
    def _generate_report(self) -> str:
        """Generate comprehensive text report"""
        report = []
        
        # Header
        report.append("="*80)
        report.append("DISTRIBUTED PUBLISH-SUBSCRIBE SYSTEM ANALYSIS REPORT")
        report.append("="*80)
        report.append(f"Generated: {self.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-"*40)
        
        stats = self.analysis_results['statistics']
        report.append(f"System Scale: {stats['nodes']['total']} nodes, {stats['edges']['total']} edges")
        
        if 'criticality' in self.analysis_results:
            critical_df = self.analysis_results['criticality']
            very_high = len(critical_df[critical_df['criticality_level'] == 'VERY_HIGH'])
            high = len(critical_df[critical_df['criticality_level'] == 'HIGH'])
            report.append(f"Critical Components: {very_high} VERY_HIGH, {high} HIGH")
        
        vuln = self.analysis_results['vulnerabilities']
        report.append(f"Single Points of Failure: {len(vuln['articulation_points'])}")
        report.append("")
        
        # System Statistics
        report.append("SYSTEM STATISTICS")
        report.append("-"*40)
        for node_type, count in stats['nodes']['by_type'].items():
            report.append(f"  {node_type}: {count}")
        report.append(f"  Graph Density: {stats['density']:.3f}")
        report.append(f"  Average Degree: {stats['average_degree']:.2f}")
        report.append(f"  Connected: {stats['is_connected']}")
        report.append("")
        
        # Critical Components
        report.append("TOP CRITICAL COMPONENTS")
        report.append("-"*40)
        if 'criticality' in self.analysis_results:
            top_10 = self.analysis_results['criticality'].head(10)
            for idx, row in top_10.iterrows():
                report.append(f"  {row['node']:<20} Type: {row['type']:<12} "
                            f"Score: {row['criticality_score']:.3f} "
                            f"Level: {row['criticality_level']}")
        report.append("")
        
        # Vulnerabilities
        report.append("STRUCTURAL VULNERABILITIES")
        report.append("-"*40)
        report.append(f"Articulation Points: {', '.join(vuln['articulation_points'][:5])}")
        if len(vuln['articulation_points']) > 5:
            report.append(f"  ... and {len(vuln['articulation_points']) - 5} more")
        report.append(f"Bridges: {len(vuln['bridges'])}")
        report.append(f"Node Connectivity: {vuln['node_connectivity']}")
        report.append(f"Edge Connectivity: {vuln['edge_connectivity']}")
        report.append("")
        
        # Message Flows
        if 'flows' in self.analysis_results:
            report.append("MESSAGE FLOW ANALYSIS")
            report.append("-"*40)
            flows = self.analysis_results['flows']
            report.append(f"Total Topics: {len(flows['topic_flows'])}")
            report.append(f"Bottleneck Topics: {len(flows['bottlenecks'])}")
            if flows['bottlenecks']:
                report.append(f"  Bottlenecks: {', '.join(flows['bottlenecks'][:5])}")
            report.append("")
        
        # Cascade Risks
        if 'cascade' in self.analysis_results and self.analysis_results['cascade']:
            report.append("CASCADE RISK ANALYSIS")
            report.append("-"*40)
            top_risks = self.analysis_results['cascade'][:5]
            for risk in top_risks:
                report.append(f"  {risk['node']:<20} "
                            f"Direct: {risk['direct_impact']:.1%} "
                            f"Indirect: {risk['indirect_impact']:.1%} "
                            f"Total: {risk['total_impact']:.1%}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-"*40)
        for i, rec in enumerate(self.recommendations, 1):
            report.append(f"{i}. [{rec['priority']}] {rec['category']}")
            report.append(f"   Issue: {rec['issue']}")
            report.append(f"   Action: {rec['recommendation']}")
            if rec['affected_components']:
                components = ', '.join(rec['affected_components'][:3])
                report.append(f"   Affected: {components}")
            report.append("")
        
        # Footer
        report.append("="*80)
        report.append("END OF REPORT")
        report.append("="*80)
        
        return "\n".join(report)
    
    def validate_analysis(self, ground_truth: Optional[Dict] = None) -> Dict:
        """
        Validate analysis results against ground truth or historical data
        
        Args:
            ground_truth: Optional ground truth data for validation
            
        Returns:
            Validation metrics
        """
        validation_results = {
            'internal_consistency': {},
            'statistical_validation': {},
            'ground_truth_comparison': {}
        }
        
        # Internal consistency checks
        validation_results['internal_consistency'] = {
            'articulation_points_valid': self._validate_articulation_points(),
            'centrality_bounds': self._validate_centrality_bounds(),
            'score_distribution': self._validate_score_distribution()
        }
        
        # Statistical validation
        if 'criticality' in self.analysis_results:
            scores = self.analysis_results['criticality']['criticality_score'].values
            validation_results['statistical_validation'] = {
                'normality_test': normaltest(scores),
                'outlier_fraction': np.sum(scores > np.percentile(scores, 95)) / len(scores),
                'coefficient_of_variation': np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0
            }
        
        # Ground truth comparison if available
        if ground_truth:
            validation_results['ground_truth_comparison'] = self._compare_with_ground_truth(ground_truth)
        
        return validation_results
    
    def _validate_articulation_points(self) -> bool:
        """Validate articulation points are correct"""
        G_undirected = self.graph.to_undirected()
        calculated_aps = set(self.analysis_results['vulnerabilities']['articulation_points'])
        actual_aps = set(nx.articulation_points(G_undirected))
        return calculated_aps == actual_aps
    
    def _validate_centrality_bounds(self) -> bool:
        """Check if centrality values are within valid bounds"""
        for metric, values in self.analysis_results['centralities'].items():
            for node, value in values.items():
                if value < 0 or value > 1:
                    return False
        return True
    
    def _validate_score_distribution(self) -> Dict:
        """Validate score distribution properties"""
        if 'criticality' not in self.analysis_results:
            return {}
        
        scores = self.analysis_results['criticality']['criticality_score'].values
        
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'skewness': skew(scores),
            'kurtosis': kurtosis(scores)
        }
    
    def _compare_with_ground_truth(self, ground_truth: Dict) -> Dict:
        """Compare analysis results with ground truth"""
        comparison = {}
        
        if 'critical_components' in ground_truth and 'criticality' in self.analysis_results:
            # Get top predicted critical components
            predicted = set(
                self.analysis_results['criticality'][
                    self.analysis_results['criticality']['criticality_level'].isin(['HIGH', 'VERY_HIGH'])
                ]['node'].values
            )
            
            actual = set(ground_truth['critical_components'])
            
            # Calculate metrics
            true_positives = len(predicted & actual)
            false_positives = len(predicted - actual)
            false_negatives = len(actual - predicted)
            
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            comparison['critical_components'] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives
            }
        
        return comparison
    
    def export_for_gnn(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Export graph data in format suitable for GNN training
        
        Returns:
            node_features: Node feature matrix
            edge_index: Edge connectivity matrix
            labels: Node labels/classifications
        """
        if 'ml_features' not in self.analysis_results:
            self.analysis_results['ml_features'] = self._generate_ml_features()
        
        # Node features
        node_features = self.analysis_results['ml_features']
        
        # Edge index (COO format for PyTorch Geometric)
        edge_index = []
        node_to_idx = {node: idx for idx, node in enumerate(self.graph.nodes())}
        
        for src, dst in self.graph.edges():
            edge_index.append([node_to_idx[src], node_to_idx[dst]])
        
        edge_index = np.array(edge_index).T if edge_index else np.array([[], []])
        
        # Labels (criticality levels)
        labels = {}
        if 'criticality' in self.analysis_results:
            for idx, row in self.analysis_results['criticality'].iterrows():
                node_idx = node_to_idx[row['node']]
                labels[node_idx] = row['criticality_level']
        
        return node_features, edge_index, labels


# ================== USAGE EXAMPLE ==================

def main():
    """
    Example usage of the integrated analyzer
    """
    # Initialize analyzer
    analyzer = IntegratedPubSubAnalyzer()
    
    # Create sample graph for demonstration
    G = nx.DiGraph()
    
    # Add nodes
    # Applications
    for i in range(1, 11):
        G.add_node(f'app{i}', type='Application')
    
    # Brokers
    for i in range(1, 4):
        G.add_node(f'broker{i}', type='Broker')
    
    # Topics
    for i in range(1, 16):
        G.add_node(f'topic{i}', type='Topic')
    
    # Physical nodes
    for i in range(1, 5):
        G.add_node(f'node{i}', type='Node')
    
    # Add edges
    import random
    random.seed(42)
    
    # Applications publish to topics
    for app in range(1, 11):
        num_topics = random.randint(1, 4)
        topics = random.sample(range(1, 16), num_topics)
        for topic in topics:
            G.add_edge(f'app{app}', f'topic{topic}', type='PUBLISHES_TO')
    
    # Applications subscribe to topics
    for app in range(1, 11):
        num_topics = random.randint(2, 5)
        topics = random.sample(range(1, 16), num_topics)
        for topic in topics:
            G.add_edge(f'topic{topic}', f'app{app}', type='SUBSCRIBES_TO')
    
    # Brokers route topics
    for topic in range(1, 16):
        broker = random.randint(1, 3)
        G.add_edge(f'broker{broker}', f'topic{topic}', type='ROUTES')
    
    # Deployment
    for app in range(1, 11):
        node = random.randint(1, 4)
        G.add_edge(f'app{app}', f'node{node}', type='RUNS_ON')
    
    for broker in range(1, 4):
        node = random.randint(1, 4)
        G.add_edge(f'broker{broker}', f'node{node}', type='RUNS_ON')
    
    # Set the graph
    analyzer.graph = G
    
    # Run complete analysis
    print("Starting integrated analysis pipeline...")
    results = analyzer.run_complete_analysis()
    
    # Validate results
    print("\nValidating analysis results...")
    validation = analyzer.validate_analysis()
    
    print("\nValidation Results:")
    print(f"  Articulation points valid: {validation['internal_consistency']['articulation_points_valid']}")
    print(f"  Centrality bounds valid: {validation['internal_consistency']['centrality_bounds']}")
    
    # Export for GNN
    print("\nExporting data for GNN training...")
    node_features, edge_index, labels = analyzer.export_for_gnn()
    print(f"  Node features shape: {node_features.shape}")
    print(f"  Edge index shape: {edge_index.shape}")
    print(f"  Number of labeled nodes: {len(labels)}")
    
    print("\n" + "="*80)
    print("Analysis complete! Check the 'analysis_output' directory for results.")
    print("="*80)
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()