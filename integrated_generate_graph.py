#!/usr/bin/env python3
"""
Enhanced generate_graph.py that integrates all improvements
Supports enhanced dataset import, validation, and benchmarking
"""

import argparse
import time
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

# Import your existing modules and the enhanced ones
from src.GraphBuilder import GraphBuilder
from src.GraphExporter import GraphExporter
from src.DatasetGenerator import DatasetGenerator

# Import enhanced modules (you can place these in src/ directory)
from src.EnhancedGraphBuilder import EnhancedGraphBuilder
from src.EnhancedGraphValidator import EnhancedGraphValidator

class SystemConfig:
    """Configuration for the graph generation system"""
    def __init__(self, **kwargs):
        self.num_nodes = kwargs.get('num_nodes', 5)
        self.num_apps = kwargs.get('num_apps', 25)
        self.num_topics = kwargs.get('num_topics', 50)
        self.num_brokers = kwargs.get('num_brokers', 2)
        self.dataset_file = kwargs.get('dataset_file', '')
        self.generate_dataset = kwargs.get('generate_dataset', False)
        self.ros2_data_file = kwargs.get('ros2_data_file', '')
        self.print_graph = kwargs.get('verbose', False)
        self.import_graph_from_file = kwargs.get('input', False)
        self.export_graph_to_file = kwargs.get('output', False)
        self.test_scalability = kwargs.get('test_scalability', False)
        self.validate = kwargs.get('validate', False)
        self.enhanced = kwargs.get('enhanced', False)
        self.benchmark = kwargs.get('benchmark', False)
        self.test_all = kwargs.get('test_all', False)

def parse_args():
    """Parse command line arguments with enhanced options"""
    parser = argparse.ArgumentParser(
        description='Enhanced Publish-Subscribe System Graph Model Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate and import small scale dataset with validation
  python generate_graph.py --import_dataset dataset_small_scale.json --enhanced --validate
  
  # Run scalability tests with benchmarking
  python generate_graph.py --test_scalability --benchmark
  
  # Generate synthetic dataset with custom parameters
  python generate_graph.py --nodes 10 --apps 50 --topics 100 --enhanced --output
        """
    )
    
    # Basic parameters
    parser.add_argument('--nodes', type=int, default=5, 
                       help='Number of nodes/machines (default: 5)')
    parser.add_argument('--apps', type=int, default=25, 
                       help='Number of applications (default: 25)')
    parser.add_argument('--topics', type=int, default=50, 
                       help='Number of topics (default: 50)')
    parser.add_argument('--brokers', type=int, default=2, 
                       help='Number of brokers (default: 2)')
    
    # Data source options
    parser.add_argument('--import_dataset', type=str, default='', 
                       help='Path to dataset JSON file to import')
    parser.add_argument('--generate_dataset', action='store_true', 
                       help='Generate synthetic datasets for all scales')
    parser.add_argument('--import_from_ros2', type=str, default='', 
                       help='Path to ROS2 JSON data file')
    
    # Processing options
    parser.add_argument('--enhanced', action='store_true',
                       help='Use enhanced graph builder with complete properties')
    parser.add_argument('--validate', action='store_true',
                       help='Run comprehensive validation after import')
    parser.add_argument('--benchmark', action='store_true',
                       help='Run performance benchmarking')
    
    # Output options
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose output')
    parser.add_argument('--input', action='store_true', 
                       help='Import graph from file instead of generating')
    parser.add_argument('--output', action='store_true', 
                       help='Export graph to file after generation')
    
    # Testing options
    parser.add_argument('--test_scalability', action='store_true', 
                       help='Run scalability tests across different scales')
    parser.add_argument('--test_all', action='store_true',
                       help='Run all datasets with validation and benchmarking')
    
    args = parser.parse_args()
    
    return SystemConfig(
        num_nodes=args.nodes,
        num_apps=args.apps,
        num_topics=args.topics,
        num_brokers=args.brokers,
        dataset_file=args.import_dataset,
        generate_dataset=args.generate_dataset,
        ros2_data_file=args.import_from_ros2,
        verbose=args.verbose,
        input=args.input,
        output=args.output,
        test_scalability=args.test_scalability,
        validate=args.validate,
        enhanced=args.enhanced,
        benchmark=args.benchmark,
        test_all=args.test_all
    )

def get_graph_builder(config: SystemConfig, uri: str, user: str, password: str):
    """Get appropriate graph builder based on configuration"""
    if config.enhanced:
        print("🚀 Using Enhanced Graph Builder with complete properties")
        return EnhancedGraphBuilder(uri, user, password)
    else:
        print("Using standard Graph Builder")
        return GraphBuilder(uri, user, password)

def import_and_validate_dataset(graph_builder, dataset_file: str, config: SystemConfig) -> Dict:
    """Import dataset and run validation if requested"""
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_file}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Import dataset
    if hasattr(graph_builder, 'import_enhanced_dataset'):
        # Enhanced import with validation
        graph_builder.import_enhanced_dataset(dataset_file)
    else:
        # Standard import
        graph_builder.import_dataset_from_file(dataset_file)
    
    import_time = time.time() - start_time
    
    results = {
        'dataset': dataset_file,
        'import_time': import_time
    }
    
    # Run validation if requested
    if config.validate:
        validator = EnhancedGraphValidator(graph_builder)
        validation_results = validator.validate_complete()
        results['validation'] = validation_results['summary']
    
    # Run benchmarking if requested
    if config.benchmark:
        benchmark_results = run_benchmarks(graph_builder)
        results['benchmark'] = benchmark_results
    
    # Export if requested
    if config.export_graph_to_file:
        output_dir = f"output/{Path(dataset_file).stem}"
        os.makedirs(output_dir, exist_ok=True)
        graph_builder.export_graph(output_dir)
        results['export_path'] = output_dir
    
    return results

def run_benchmarks(graph_builder) -> Dict:
    """Run performance benchmarks"""
    print("\n⏱️ Running Performance Benchmarks...")
    
    benchmarks = {
        'import_metrics': {},
        'query_performance': {},
        'graph_metrics': {}
    }
    
    # Graph size metrics
    size_metrics = graph_builder.execute_cypher("""
        MATCH (n) WITH count(n) as total_nodes
        MATCH ()-[r]->() WITH total_nodes, count(r) as total_edges
        RETURN total_nodes, total_edges, 
               toFloat(total_edges) / (total_nodes * (total_nodes - 1)) as density
    """)[0]
    benchmarks['graph_metrics'] = size_metrics
    
    # Critical queries
    queries = {
        'shortest_path': """
            MATCH (a1:Application {type: 'PRODUCER'}), (a2:Application {type: 'CONSUMER'})
            WITH a1, a2 LIMIT 1
            MATCH path = shortestPath((a1)-[:DEPENDS_ON*]-(a2))
            RETURN length(path) as path_length
        """,
        'centrality': """
            MATCH (a:Application)
            OPTIONAL MATCH (a)-[:DEPENDS_ON]->(other)
            WITH a, count(other) as out_degree
            OPTIONAL MATCH (other)-[:DEPENDS_ON]->(a)
            WITH a, out_degree, count(other) as in_degree
            RETURN a.name, out_degree + in_degree as degree_centrality
            ORDER BY degree_centrality DESC
            LIMIT 10
        """,
        'bottleneck': """
            MATCH (t:Topic)
            WHERE t.fanout_ratio > 5
            RETURN t.name, t.fanout_ratio, t.criticality_score
            ORDER BY t.fanout_ratio DESC
            LIMIT 10
        """,
        'critical_path': """
            MATCH path = (a1:Application)-[:DEPENDS_ON*1..5]->(a2:Application)
            WHERE a1.criticality_score > 0.8 AND a2.criticality_score > 0.8
            RETURN count(path) as critical_paths
        """
    }
    
    for name, query in queries.items():
        start_time = time.time()
        try:
            result = graph_builder.execute_cypher(query)
            execution_time = time.time() - start_time
            benchmarks['query_performance'][name] = {
                'time': execution_time,
                'status': 'success',
                'result_count': len(result) if result else 0
            }
        except Exception as e:
            benchmarks['query_performance'][name] = {
                'time': time.time() - start_time,
                'status': 'error',
                'error': str(e)
            }
    
    return benchmarks

def test_all_scales(config: SystemConfig):
    """Test all three dataset scales with validation and benchmarking"""
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "password"
    
    datasets = [
        "input/dataset_small_scale.json",
        "input/dataset_medium_scale.json",
        "input/dataset_large_scale.json"
    ]
    
    all_results = []
    
    for dataset in datasets:
        if not os.path.exists(dataset):
            print(f"⚠️ Dataset {dataset} not found, skipping...")
            continue
        
        graph_builder = get_graph_builder(config, uri, user, password)
        
        try:
            results = import_and_validate_dataset(graph_builder, dataset, config)
            all_results.append(results)
            
            # Print summary
            print(f"\n✅ {dataset} processed successfully")
            print(f"   Import time: {results['import_time']:.2f}s")
            
            if 'validation' in results:
                val_summary = results['validation']
                print(f"   Validation: {val_summary['passed']}/{val_summary['total_checks']} passed")
            
            if 'benchmark' in results:
                avg_query_time = sum(
                    q['time'] for q in results['benchmark']['query_performance'].values()
                ) / len(results['benchmark']['query_performance'])
                print(f"   Avg query time: {avg_query_time:.3f}s")
        
        except Exception as e:
            print(f"❌ Error processing {dataset}: {str(e)}")
            all_results.append({
                'dataset': dataset,
                'error': str(e)
            })
        
        finally:
            graph_builder.close()
    
    # Save comprehensive results
    with open('output/test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)
    print(f"Results saved to test_results.json")
    
    # Print comparison table
    print("\n📊 Scalability Comparison:")
    print(f"{'Dataset':<20} {'Import (s)':<12} {'Validation':<15} {'Avg Query (s)':<15}")
    print("-" * 62)
    
    for result in all_results:
        if 'error' not in result:
            dataset_name = Path(result['dataset']).stem.replace('dataset_', '')
            import_time = f"{result['import_time']:.2f}"
            
            if 'validation' in result:
                val = result['validation']
                validation = f"{val['passed']}/{val['total_checks']}"
            else:
                validation = "N/A"
            
            if 'benchmark' in result:
                query_times = [q['time'] for q in result['benchmark']['query_performance'].values()]
                avg_query = f"{sum(query_times)/len(query_times):.3f}"
            else:
                avg_query = "N/A"
            
            print(f"{dataset_name:<20} {import_time:<12} {validation:<15} {avg_query:<15}")

def main():
    """Main execution function"""
    config = parse_args()
    
    # Neo4j connection settings
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    
    # Print configuration
    print("🔧 System Configuration:")
    if config.enhanced:
        print("  Mode: ENHANCED (with complete properties)")
    else:
        print("  Mode: Standard")
    
    if config.validate:
        print("  Validation: ENABLED")
    if config.benchmark:
        print("  Benchmarking: ENABLED")
    
    # Handle different execution modes
    if config.test_all or (config.test_scalability and config.enhanced):
        # Test all scales with validation
        config.validate = True
        config.benchmark = True
        test_all_scales(config)
    
    elif config.generate_dataset:
        # Generate all synthetic datasets
        from DatasetGenerator import DatasetGenerator
        generator = DatasetGenerator()
        
        print("Generating synthetic datasets...")
        datasets = {
            'small': generator.generate_small_scale(),
            'medium': generator.generate_medium_scale_complete(),
            'large': generator.generate_large_scale_complete()
        }
        
        for scale, dataset in datasets.items():
            filename = f'dataset_{scale}_scale.json'
            with open(filename, 'w') as f:
                json.dump(dataset, f, indent=2)
            print(f"✅ Generated {filename}")
    
    elif config.dataset_file:
        # Import specific dataset
        graph_builder = get_graph_builder(config, uri, user, password)
        
        try:
            results = import_and_validate_dataset(graph_builder, config.dataset_file, config)
            
            print("\n" + "="*60)
            print("PROCESSING COMPLETE")
            print("="*60)
            print(f"✅ Import time: {results['import_time']:.2f}s")
            
            if 'validation' in results:
                val = results['validation']
                health_score = (val['passed'] / val['total_checks']) * 100
                print(f"✅ Health Score: {health_score:.1f}%")
            
        finally:
            graph_builder.close()
    
    else:
        # Generate synthetic graph with specified parameters
        graph_builder = get_graph_builder(config, uri, user, password)
        
        try:
            print(f"\nGenerating synthetic graph:")
            print(f"  Nodes: {config.num_nodes}")
            print(f"  Applications: {config.num_apps}")
            print(f"  Topics: {config.num_topics}")
            print(f"  Brokers: {config.num_brokers}")
            
            graph_config = {
                "num_nodes": config.num_nodes,
                "num_apps": config.num_apps,
                "num_topics": config.num_topics,
                "num_brokers": config.num_brokers
            }
            
            start_time = time.time()
            graph_builder.generate_synthetic_dataset(graph_config)
            graph_builder.import_dataset_to_neo4j()
            
            if config.validate:
                validator = EnhancedGraphValidator(graph_builder)
                validator.validate_complete()
            
            print(f"\n✅ Graph generation completed in {time.time() - start_time:.2f}s")
            
        finally:
            graph_builder.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)