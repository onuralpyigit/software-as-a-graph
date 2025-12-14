#!/usr/bin/env python3
"""
Comprehensive Pub-Sub System Graph Analyzer
============================================

Main entry point for analyzing distributed publish-subscribe systems using
graph-based methods. This script integrates multiple analysis approaches:

1. DEPENDENCY ANALYSIS - Multi-layer DEPENDS_ON relationship analysis
2. GRAPH ALGORITHMS - Direct application of centrality, structural algorithms
3. ANTI-PATTERN DETECTION - Automatic identification of architectural issues
4. CRITICALITY ASSESSMENT - Multiple perspectives on component importance

Usage:
    # Basic analysis with dependency graph
    python analyze_graph.py --input system.json
    
    # Full analysis with all features
    python analyze_graph.py --input system.json --full --export-all
    
    # Specific analysis modes
    python analyze_graph.py --input system.json --mode dependency
    python analyze_graph.py --input system.json --mode algorithms
    python analyze_graph.py --input system.json --mode antipatterns
    
    # Export options
    python analyze_graph.py --input system.json --export-json --export-html --export-csv
    
    # With Neo4j integration
    python analyze_graph.py --input system.json --neo4j --neo4j-uri bolt://localhost:7687

Author: Software-as-a-Graph Research Project
Version: 2.0
"""

import argparse
import json
import logging
import sys
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: NetworkX not available. Install with: pip install networkx")

# Import our analysis modules
from src.analysis.dependency_analyzer import (
    MultiLayerDependencyAnalyzer,
    DependencyGraphBuilder,
    GraphAlgorithmAnalyzer,
    AntiPatternDetector,
    CriticalComponentIdentifier,
    ReportGenerator,
    AnalysisResult,
    CriticalNode,
    CriticalEdge,
    AntiPattern,
    AntiPatternType,
    CriticalityReason,
    Layer,
    DependencyType
)

from src.analysis.graph_algorithms import (
    CentralityAnalyzer,
    StructuralAnalyzer,
    CommunityDetector,
    PathFlowAnalyzer,
    SimilarityCouplingAnalyzer,
    RobustnessAnalyzer,
    LayerAwareAnalyzer,
    ComprehensiveGraphAnalyzer,
    get_algorithm_recommendations
)


# ============================================================================
# Configuration and Constants
# ============================================================================

class AnalysisMode(Enum):
    """Available analysis modes"""
    FULL = "full"                    # All analyses
    DEPENDENCY = "dependency"        # Dependency graph analysis only
    ALGORITHMS = "algorithms"        # Graph algorithms only
    ANTIPATTERNS = "antipatterns"   # Anti-pattern detection only
    CRITICALITY = "criticality"     # Critical component identification
    QUICK = "quick"                 # Fast basic analysis


@dataclass
class AnalysisConfig:
    """Configuration for analysis"""
    # Input/Output
    input_file: str
    output_dir: str = "output"
    output_prefix: str = "analysis"
    
    # Analysis modes
    mode: AnalysisMode = AnalysisMode.FULL
    
    # Algorithm settings
    percentile_threshold: int = 90  # For "high" value detection
    
    # Anti-pattern thresholds
    god_topic_threshold: int = 10
    hub_overload_threshold: int = 15
    long_chain_threshold: int = 5
    tight_coupling_min_size: int = 4
    
    # Export options
    export_json: bool = True
    export_html: bool = True
    export_csv: bool = False
    export_text: bool = False
    
    # Neo4j integration
    use_neo4j: bool = False
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    
    # Logging
    verbose: bool = False
    quiet: bool = False


# ============================================================================
# Main Analyzer Class
# ============================================================================

class PubSubGraphAnalyzer:
    """
    Comprehensive analyzer for pub-sub systems.
    
    Integrates multiple analysis approaches:
    - Dependency graph construction and analysis
    - Graph algorithm application
    - Anti-pattern detection
    - Critical component identification
    """
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Analysis results
        self.graph: Optional[nx.DiGraph] = None
        self.raw_data: Optional[Dict] = None
        self.dependency_result: Optional[AnalysisResult] = None
        self.algorithm_results: Optional[Dict] = None
        
        # Timing
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('pub_sub_analyzer')
        
        if self.config.quiet:
            level = logging.WARNING
        elif self.config.verbose:
            level = logging.DEBUG
        else:
            level = logging.INFO
        
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        return logger
    
    def load_data(self) -> Dict[str, Any]:
        """Load pub-sub system data from file"""
        self.logger.info(f"Loading data from: {self.config.input_file}")
        
        input_path = Path(self.config.input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.config.input_file}")
        
        with open(input_path) as f:
            self.raw_data = json.load(f)
        
        # Validate basic structure
        required_keys = ['applications', 'topics']
        for key in required_keys:
            if key not in self.raw_data:
                self.logger.warning(f"Missing expected key in input: {key}")
        
        # Log summary
        self.logger.info(f"Loaded: {len(self.raw_data.get('nodes', []))} nodes, "
                        f"{len(self.raw_data.get('brokers', []))} brokers, "
                        f"{len(self.raw_data.get('topics', []))} topics, "
                        f"{len(self.raw_data.get('applications', []))} applications")
        
        return self.raw_data
    
    def analyze(self) -> Dict[str, Any]:
        """
        Run comprehensive analysis based on configuration.
        
        Returns:
            Dict containing all analysis results
        """
        self.start_time = datetime.now()
        self.logger.info(f"Starting analysis in {self.config.mode.value} mode...")
        
        results = {
            'metadata': {
                'analysis_time': self.start_time.isoformat(),
                'mode': self.config.mode.value,
                'input_file': self.config.input_file,
                'version': '2.0'
            }
        }
        
        # Load data if not already loaded
        if self.raw_data is None:
            self.load_data()
        
        # Run analyses based on mode
        if self.config.mode == AnalysisMode.FULL:
            results.update(self._run_full_analysis())
        elif self.config.mode == AnalysisMode.DEPENDENCY:
            results.update(self._run_dependency_analysis())
        elif self.config.mode == AnalysisMode.ALGORITHMS:
            results.update(self._run_algorithm_analysis())
        elif self.config.mode == AnalysisMode.ANTIPATTERNS:
            results.update(self._run_antipattern_analysis())
        elif self.config.mode == AnalysisMode.CRITICALITY:
            results.update(self._run_criticality_analysis())
        elif self.config.mode == AnalysisMode.QUICK:
            results.update(self._run_quick_analysis())
        
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        results['metadata']['duration_seconds'] = duration
        
        self.logger.info(f"Analysis complete in {duration:.2f} seconds")
        
        return results
    
    def _run_full_analysis(self) -> Dict[str, Any]:
        """Run complete analysis with all features"""
        self.logger.info("Running FULL analysis...")
        
        results = {}
        
        # 1. Dependency Analysis (includes algorithms, anti-patterns, criticality)
        self.logger.info("[1/4] Running dependency analysis...")
        dep_analyzer = MultiLayerDependencyAnalyzer()
        self.dependency_result = dep_analyzer.analyze(self.raw_data)
        
        results['dependency_analysis'] = self._serialize_dependency_result(self.dependency_result)
        
        # Store graph for further analysis
        builder = DependencyGraphBuilder()
        self.graph = builder.build_from_pubsub_data(self.raw_data)
        
        # 2. Extended Algorithm Analysis
        self.logger.info("[2/4] Running extended algorithm analysis...")
        results['extended_algorithms'] = self._run_extended_algorithms()
        
        # 3. Robustness Analysis
        self.logger.info("[3/4] Running robustness analysis...")
        results['robustness'] = self._run_robustness_analysis()
        
        # 4. Generate Summary
        self.logger.info("[4/4] Generating summary...")
        results['summary'] = self._generate_comprehensive_summary(results)
        
        return results
    
    def _run_dependency_analysis(self) -> Dict[str, Any]:
        """Run dependency-focused analysis"""
        self.logger.info("Running DEPENDENCY analysis...")
        
        dep_analyzer = MultiLayerDependencyAnalyzer()
        self.dependency_result = dep_analyzer.analyze(self.raw_data)
        
        # Store graph
        builder = DependencyGraphBuilder()
        self.graph = builder.build_from_pubsub_data(self.raw_data)
        
        return {
            'dependency_analysis': self._serialize_dependency_result(self.dependency_result)
        }
    
    def _run_algorithm_analysis(self) -> Dict[str, Any]:
        """Run graph algorithm-focused analysis"""
        self.logger.info("Running ALGORITHM analysis...")
        
        # Build graph
        builder = DependencyGraphBuilder()
        self.graph = builder.build_from_pubsub_data(self.raw_data)
        
        # Run comprehensive algorithms
        comprehensive = ComprehensiveGraphAnalyzer(self.graph)
        self.algorithm_results = comprehensive.run_comprehensive_analysis()
        
        return {
            'algorithm_analysis': self._serialize_algorithm_results(self.algorithm_results)
        }
    
    def _run_antipattern_analysis(self) -> Dict[str, Any]:
        """Run anti-pattern detection focused analysis"""
        self.logger.info("Running ANTIPATTERN analysis...")
        
        # Build graph and run algorithms (needed for anti-pattern detection)
        builder = DependencyGraphBuilder()
        self.graph = builder.build_from_pubsub_data(self.raw_data)
        
        algo_analyzer = GraphAlgorithmAnalyzer(self.graph)
        algo_results = algo_analyzer.run_all_algorithms()
        
        # Detect anti-patterns
        detector = AntiPatternDetector(self.graph, algo_results)
        detector.config.update({
            'god_topic_threshold': self.config.god_topic_threshold,
            'hub_overload_threshold': self.config.hub_overload_threshold,
            'long_chain_threshold': self.config.long_chain_threshold,
            'tight_coupling_min_size': self.config.tight_coupling_min_size,
        })
        
        anti_patterns = detector.detect_all()
        
        return {
            'anti_patterns': [self._serialize_antipattern(ap) for ap in anti_patterns],
            'summary': {
                'total': len(anti_patterns),
                'by_severity': self._count_by_severity(anti_patterns),
                'by_type': self._count_by_type(anti_patterns)
            }
        }
    
    def _run_criticality_analysis(self) -> Dict[str, Any]:
        """Run critical component identification focused analysis"""
        self.logger.info("Running CRITICALITY analysis...")
        
        # Build graph and run algorithms
        builder = DependencyGraphBuilder()
        self.graph = builder.build_from_pubsub_data(self.raw_data)
        
        algo_analyzer = GraphAlgorithmAnalyzer(self.graph)
        algo_results = algo_analyzer.run_all_algorithms()
        
        # Identify critical components
        identifier = CriticalComponentIdentifier(self.graph, algo_results)
        critical_nodes = identifier.identify_critical_nodes()
        critical_edges = identifier.identify_critical_edges()
        
        return {
            'critical_nodes': [self._serialize_critical_node(cn) for cn in critical_nodes],
            'critical_edges': [self._serialize_critical_edge(ce) for ce in critical_edges],
            'summary': {
                'total_critical_nodes': len(critical_nodes),
                'total_critical_edges': len(critical_edges),
                'by_layer': self._count_nodes_by_layer(critical_nodes),
                'by_reason': self._count_nodes_by_reason(critical_nodes)
            }
        }
    
    def _run_quick_analysis(self) -> Dict[str, Any]:
        """Run fast basic analysis"""
        self.logger.info("Running QUICK analysis...")
        
        # Build graph
        builder = DependencyGraphBuilder()
        self.graph = builder.build_from_pubsub_data(self.raw_data)
        
        # Basic metrics only
        results = {
            'graph_metrics': {
                'nodes': len(self.graph.nodes()),
                'edges': len(self.graph.edges()),
                'density': nx.density(self.graph),
                'is_connected': nx.is_weakly_connected(self.graph),
            },
            'layer_counts': self._get_layer_counts(),
            'basic_centrality': self._get_basic_centrality(),
            'structural_quick': self._get_structural_quick()
        }
        
        return results
    
    def _run_extended_algorithms(self) -> Dict[str, Any]:
        """Run extended algorithm analysis beyond dependency analyzer"""
        if self.graph is None:
            return {}
        
        results = {}
        
        # Community analysis with more detail
        try:
            community_detector = CommunityDetector(self.graph)
            louvain = community_detector.louvain_communities()
            composition = community_detector.analyze_community_composition(louvain)
            
            results['communities'] = {
                'num_communities': louvain.statistics['num_communities'],
                'modularity': louvain.modularity,
                'composition': composition['communities']
            }
        except Exception as e:
            self.logger.warning(f"Community analysis failed: {e}")
        
        # Path flow analysis
        try:
            path_analyzer = PathFlowAnalyzer(self.graph)
            results['message_flow'] = path_analyzer.message_flow_analysis()
            results['bottlenecks'] = path_analyzer.find_bottlenecks()
        except Exception as e:
            self.logger.warning(f"Path flow analysis failed: {e}")
        
        # Coupling analysis
        try:
            coupling_analyzer = SimilarityCouplingAnalyzer(self.graph)
            results['coupling'] = coupling_analyzer.coupling_analysis()
        except Exception as e:
            self.logger.warning(f"Coupling analysis failed: {e}")
        
        return results
    
    def _run_robustness_analysis(self) -> Dict[str, Any]:
        """Run robustness/resilience analysis"""
        if self.graph is None:
            return {}
        
        try:
            robustness = RobustnessAnalyzer(self.graph)
            comparison = robustness.compare_robustness()
            
            return {
                'vulnerability_to_targeted_attacks': comparison['vulnerability_to_targeted_attacks'],
                'comparison': {
                    name: {
                        'critical_threshold': r['critical_threshold'],
                        'auc_robustness': r['auc_robustness']
                    }
                    for name, r in comparison['comparison'].items()
                }
            }
        except Exception as e:
            self.logger.warning(f"Robustness analysis failed: {e}")
            return {}
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _get_layer_counts(self) -> Dict[str, int]:
        """Get node counts by layer"""
        counts = defaultdict(int)
        for node, data in self.graph.nodes(data=True):
            layer = data.get('layer', 'unknown')
            counts[layer] += 1
        return dict(counts)
    
    def _get_basic_centrality(self) -> Dict[str, Any]:
        """Get basic centrality measures"""
        bc = nx.betweenness_centrality(self.graph)
        pr = nx.pagerank(self.graph)
        
        # Top 5 by each measure
        top_bc = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:5]
        top_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'top_betweenness': [{'node': n, 'score': round(s, 4)} for n, s in top_bc],
            'top_pagerank': [{'node': n, 'score': round(s, 4)} for n, s in top_pr]
        }
    
    def _get_structural_quick(self) -> Dict[str, Any]:
        """Get quick structural analysis"""
        G_undirected = self.graph.to_undirected()
        
        aps = list(nx.articulation_points(G_undirected))
        bridges = list(nx.bridges(G_undirected))
        
        return {
            'articulation_points': aps,
            'articulation_point_count': len(aps),
            'bridge_count': len(bridges),
            'has_cycles': len(list(nx.simple_cycles(self.graph))) > 0 if len(self.graph) < 100 else 'not_computed'
        }
    
    def _generate_comprehensive_summary(self, results: Dict) -> Dict[str, Any]:
        """Generate comprehensive summary of all analyses"""
        summary = {
            'overall_health': 'unknown',
            'key_findings': [],
            'recommendations': [],
            'metrics': {}
        }
        
        # Extract key metrics
        if 'dependency_analysis' in results:
            dep = results['dependency_analysis']
            summary['metrics']['critical_nodes'] = dep.get('summary', {}).get('total_critical_nodes', 0)
            summary['metrics']['anti_patterns'] = dep.get('summary', {}).get('total_anti_patterns', 0)
            
            # Key findings from anti-patterns
            for ap in dep.get('anti_patterns', [])[:5]:
                if ap.get('severity') in ['critical', 'high']:
                    summary['key_findings'].append({
                        'type': ap.get('pattern_type'),
                        'severity': ap.get('severity'),
                        'description': ap.get('description')
                    })
        
        # Robustness assessment
        if 'robustness' in results:
            vuln = results['robustness'].get('vulnerability_to_targeted_attacks', 0)
            summary['metrics']['vulnerability_score'] = round(vuln, 4)
            
            if vuln > 0.5:
                summary['recommendations'].append(
                    "High vulnerability to targeted attacks. Add redundancy to critical components."
                )
        
        # Health assessment
        critical_count = summary['metrics'].get('critical_nodes', 0)
        antipattern_count = summary['metrics'].get('anti_patterns', 0)
        
        if critical_count == 0 and antipattern_count == 0:
            summary['overall_health'] = 'excellent'
        elif critical_count <= 3 and antipattern_count <= 5:
            summary['overall_health'] = 'good'
        elif critical_count <= 10 and antipattern_count <= 15:
            summary['overall_health'] = 'fair'
        else:
            summary['overall_health'] = 'needs_attention'
        
        return summary
    
    # ========================================================================
    # Serialization Methods
    # ========================================================================
    
    def _serialize_dependency_result(self, result: AnalysisResult) -> Dict:
        """Serialize AnalysisResult to JSON-compatible dict"""
        return {
            'graph_metrics': result.graph_metrics,
            'layer_metrics': result.layer_metrics,
            'critical_nodes': [self._serialize_critical_node(cn) for cn in result.critical_nodes],
            'critical_edges': [self._serialize_critical_edge(ce) for ce in result.critical_edges],
            'anti_patterns': [self._serialize_antipattern(ap) for ap in result.anti_patterns],
            'summary': result.summary
        }
    
    def _serialize_critical_node(self, node: CriticalNode) -> Dict:
        """Serialize CriticalNode"""
        return {
            'node_id': node.node_id,
            'layer': node.layer,
            'node_type': node.node_type,
            'reasons': [r.value for r in node.reasons],
            'metrics': {k: round(v, 4) if isinstance(v, float) else v 
                       for k, v in node.metrics.items()},
            'impact_description': node.impact_description,
            'recommendation': node.recommendation
        }
    
    def _serialize_critical_edge(self, edge: CriticalEdge) -> Dict:
        """Serialize CriticalEdge"""
        return {
            'source': edge.source,
            'target': edge.target,
            'dependency_type': edge.dependency_type,
            'reasons': [r.value for r in edge.reasons],
            'metrics': {k: round(v, 4) if isinstance(v, float) else v 
                       for k, v in edge.metrics.items()},
            'impact_description': edge.impact_description,
            'recommendation': edge.recommendation
        }
    
    def _serialize_antipattern(self, pattern: AntiPattern) -> Dict:
        """Serialize AntiPattern"""
        return {
            'pattern_type': pattern.pattern_type.value,
            'severity': pattern.severity,
            'affected_components': pattern.affected_components,
            'description': pattern.description,
            'impact': pattern.impact,
            'recommendation': pattern.recommendation,
            'metrics': pattern.metrics
        }
    
    def _serialize_algorithm_results(self, results: Dict) -> Dict:
        """Serialize algorithm results"""
        serialized = {}
        
        for key, value in results.items():
            if hasattr(value, '__dict__'):
                serialized[key] = self._serialize_object(value)
            elif isinstance(value, dict):
                serialized[key] = {str(k): self._serialize_object(v) 
                                  for k, v in value.items()}
            else:
                serialized[key] = value
        
        return serialized
    
    def _serialize_object(self, obj) -> Any:
        """Generic object serialization"""
        if isinstance(obj, Enum):
            return obj.value
        elif hasattr(obj, '__dict__'):
            return {k: self._serialize_object(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {str(k): self._serialize_object(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_object(i) for i in obj]
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, float):
            if obj == float('inf'):
                return "inf"
            return round(obj, 6)
        return obj
    
    def _count_by_severity(self, patterns: List[AntiPattern]) -> Dict[str, int]:
        """Count anti-patterns by severity"""
        counts = defaultdict(int)
        for p in patterns:
            counts[p.severity] += 1
        return dict(counts)
    
    def _count_by_type(self, patterns: List[AntiPattern]) -> Dict[str, int]:
        """Count anti-patterns by type"""
        counts = defaultdict(int)
        for p in patterns:
            counts[p.pattern_type.value] += 1
        return dict(counts)
    
    def _count_nodes_by_layer(self, nodes: List[CriticalNode]) -> Dict[str, int]:
        """Count critical nodes by layer"""
        counts = defaultdict(int)
        for n in nodes:
            counts[n.layer] += 1
        return dict(counts)
    
    def _count_nodes_by_reason(self, nodes: List[CriticalNode]) -> Dict[str, int]:
        """Count critical nodes by reason"""
        counts = defaultdict(int)
        for n in nodes:
            for r in n.reasons:
                counts[r.value] += 1
        return dict(counts)
    
    # ========================================================================
    # Export Methods
    # ========================================================================
    
    def export_results(self, results: Dict[str, Any]):
        """Export results in configured formats"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        prefix = self.config.output_prefix
        
        if self.config.export_json:
            self._export_json(results, output_dir / f"{prefix}_results.json")
        
        if self.config.export_html:
            self._export_html(results, output_dir / f"{prefix}_report.html")
        
        if self.config.export_csv:
            self._export_csv(results, output_dir / f"{prefix}")
        
        if self.config.export_text:
            self._export_text(results, output_dir / f"{prefix}_report.txt")
        
        self.logger.info(f"Results exported to: {output_dir}")
    
    def _export_json(self, results: Dict, filepath: Path):
        """Export to JSON"""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"  ✓ JSON: {filepath}")
    
    def _export_html(self, results: Dict, filepath: Path):
        """Export to HTML report"""
        # Use ReportGenerator if we have dependency results
        if self.dependency_result:
            html = ReportGenerator.generate_html_report(self.dependency_result)
        else:
            html = self._generate_basic_html(results)
        
        with open(filepath, 'w') as f:
            f.write(html)
        self.logger.info(f"  ✓ HTML: {filepath}")
    
    def _export_csv(self, results: Dict, filepath_prefix: Path):
        """Export to CSV files"""
        # Export critical nodes
        if 'dependency_analysis' in results:
            dep = results['dependency_analysis']
            
            # Critical nodes CSV
            if 'critical_nodes' in dep:
                csv_path = Path(f"{filepath_prefix}_critical_nodes.csv")
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['node_id', 'layer', 'type', 'reasons', 'impact', 'recommendation'])
                    for node in dep['critical_nodes']:
                        writer.writerow([
                            node['node_id'],
                            node['layer'],
                            node['node_type'],
                            ';'.join(node['reasons']),
                            node['impact_description'],
                            node['recommendation']
                        ])
                self.logger.info(f"  ✓ CSV: {csv_path}")
            
            # Anti-patterns CSV
            if 'anti_patterns' in dep:
                csv_path = Path(f"{filepath_prefix}_anti_patterns.csv")
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['type', 'severity', 'components', 'description', 'recommendation'])
                    for ap in dep['anti_patterns']:
                        writer.writerow([
                            ap['pattern_type'],
                            ap['severity'],
                            ';'.join(ap['affected_components'][:5]),
                            ap['description'],
                            ap['recommendation']
                        ])
                self.logger.info(f"  ✓ CSV: {csv_path}")
    
    def _export_text(self, results: Dict, filepath: Path):
        """Export to text report"""
        if self.dependency_result:
            text = ReportGenerator.generate_text_report(self.dependency_result)
        else:
            text = self._generate_basic_text(results)
        
        with open(filepath, 'w') as f:
            f.write(text)
        self.logger.info(f"  ✓ Text: {filepath}")
    
    def _generate_basic_html(self, results: Dict) -> str:
        """Generate basic HTML when full dependency result not available"""
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Graph Analysis Report</title>
    <style>
        body {{ font-family: sans-serif; padding: 20px; background: #1a1a2e; color: #eee; }}
        .container {{ max-width: 1000px; margin: 0 auto; }}
        pre {{ background: #2d2d44; padding: 15px; border-radius: 5px; overflow-x: auto; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Graph Analysis Report</h1>
    <p>Generated: {datetime.now().isoformat()}</p>
    <h2>Results</h2>
    <pre>{json.dumps(results, indent=2, default=str)}</pre>
</div>
</body>
</html>"""
    
    def _generate_basic_text(self, results: Dict) -> str:
        """Generate basic text report"""
        lines = [
            "=" * 60,
            "  GRAPH ANALYSIS REPORT",
            "=" * 60,
            f"\nGenerated: {datetime.now().isoformat()}",
            "\nResults:",
            json.dumps(results, indent=2, default=str)
        ]
        return "\n".join(lines)


# ============================================================================
# CLI Interface
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Pub-Sub System Graph Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python analyze_graph.py --input system.json
  
  # Full analysis with all exports
  python analyze_graph.py --input system.json --mode full --export-all
  
  # Quick analysis
  python analyze_graph.py --input system.json --mode quick
  
  # Anti-pattern detection only
  python analyze_graph.py --input system.json --mode antipatterns
  
  # Custom thresholds
  python analyze_graph.py --input system.json --god-topic-threshold 5
        """
    )
    
    # Input/Output
    parser.add_argument('--input', '-i', required=True,
                       help='Input JSON file with pub-sub system definition')
    parser.add_argument('--output-dir', '-o', default='output',
                       help='Output directory (default: output)')
    parser.add_argument('--output-prefix', default='analysis',
                       help='Prefix for output files (default: analysis)')
    
    # Analysis Mode
    parser.add_argument('--mode', '-m', 
                       choices=['full', 'dependency', 'algorithms', 'antipatterns', 'criticality', 'quick'],
                       default='full',
                       help='Analysis mode (default: full)')
    
    # Export Options
    parser.add_argument('--export-json', action='store_true', default=True,
                       help='Export results as JSON (default: True)')
    parser.add_argument('--export-html', action='store_true', default=True,
                       help='Export results as HTML report (default: True)')
    parser.add_argument('--export-csv', action='store_true',
                       help='Export results as CSV files')
    parser.add_argument('--export-text', action='store_true',
                       help='Export results as text report')
    parser.add_argument('--export-all', action='store_true',
                       help='Export in all formats')
    parser.add_argument('--no-export', action='store_true',
                       help='Do not export results (print to stdout)')
    
    # Anti-pattern Thresholds
    parser.add_argument('--god-topic-threshold', type=int, default=10,
                       help='Connections threshold for god topic detection (default: 10)')
    parser.add_argument('--hub-overload-threshold', type=int, default=15,
                       help='Out-degree threshold for hub overload (default: 15)')
    parser.add_argument('--long-chain-threshold', type=int, default=5,
                       help='Depth threshold for long dependency chains (default: 5)')
    parser.add_argument('--tight-coupling-size', type=int, default=4,
                       help='Minimum clique size for tight coupling (default: 4)')
    
    # Algorithm Settings
    parser.add_argument('--percentile-threshold', type=int, default=90,
                       help='Percentile for "high" value detection (default: 90)')
    
    # Neo4j Integration
    parser.add_argument('--neo4j', action='store_true',
                       help='Enable Neo4j integration')
    parser.add_argument('--neo4j-uri', default='bolt://localhost:7687',
                       help='Neo4j connection URI')
    parser.add_argument('--neo4j-user', default='neo4j',
                       help='Neo4j username')
    parser.add_argument('--neo4j-password', default='',
                       help='Neo4j password')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode (warnings only)')
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Build configuration
    config = AnalysisConfig(
        input_file=args.input,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        mode=AnalysisMode(args.mode),
        percentile_threshold=args.percentile_threshold,
        god_topic_threshold=args.god_topic_threshold,
        hub_overload_threshold=args.hub_overload_threshold,
        long_chain_threshold=args.long_chain_threshold,
        tight_coupling_min_size=args.tight_coupling_size,
        export_json=args.export_json and not args.no_export,
        export_html=args.export_html and not args.no_export,
        export_csv=args.export_csv or args.export_all,
        export_text=args.export_text or args.export_all,
        use_neo4j=args.neo4j,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        verbose=args.verbose,
        quiet=args.quiet
    )
    
    if args.export_all:
        config.export_json = True
        config.export_html = True
        config.export_csv = True
        config.export_text = True
    
    # Print banner
    if not config.quiet:
        print("""
╔══════════════════════════════════════════════════════════════════╗
║     PUB-SUB SYSTEM GRAPH ANALYZER v2.0                          ║
║     Graph-Based Analysis of Distributed Systems                  ║
╚══════════════════════════════════════════════════════════════════╝
        """)
    
    try:
        # Create analyzer and run
        analyzer = PubSubGraphAnalyzer(config)
        results = analyzer.analyze()
        
        # Export results
        if not args.no_export:
            analyzer.export_results(results)
        else:
            # Print to stdout
            print(json.dumps(results, indent=2, default=str))
        
        # Print summary
        if not config.quiet:
            print("\n" + "=" * 60)
            print("  ANALYSIS SUMMARY")
            print("=" * 60)
            
            if 'summary' in results:
                summary = results['summary']
                print(f"\n  Overall Health: {summary.get('overall_health', 'unknown').upper()}")
                
                metrics = summary.get('metrics', {})
                print(f"  Critical Nodes: {metrics.get('critical_nodes', 'N/A')}")
                print(f"  Anti-Patterns: {metrics.get('anti_patterns', 'N/A')}")
                
                if 'key_findings' in summary and summary['key_findings']:
                    print("\n  Key Findings:")
                    for finding in summary['key_findings'][:3]:
                        print(f"    • [{finding.get('severity', '').upper()}] {finding.get('type')}")
                
                if 'recommendations' in summary and summary['recommendations']:
                    print("\n  Recommendations:")
                    for rec in summary['recommendations'][:3]:
                        print(f"    • {rec}")
            
            print("\n" + "=" * 60)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
