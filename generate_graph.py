#!/usr/bin/env python3
import argparse
import time
import json
from src.GraphBuilder import GraphBuilder
from src.GraphExporter import GraphExporter
from src.DatasetGenerator import DatasetGenerator

# Configuration parameters with defaults
class SystemConfig:
    def __init__(self, num_nodes=4, num_apps=10, num_topics=25, num_brokers=1,  dataset_file="", generate_dataset=False, ros2_data_file="", print_graph=False, import_graph_from_file=False, export_graph_to_file=False, test_scalability=False):
        self.num_nodes = num_nodes
        self.num_apps = num_apps
        self.num_topics = num_topics
        self.num_brokers = num_brokers
        self.dataset_file = dataset_file
        self.generate_dataset = generate_dataset
        self.ros2_data_file = ros2_data_file
        self.print_graph = print_graph
        self.import_graph_from_file = import_graph_from_file
        self.export_graph_to_file = export_graph_to_file  
        self.test_scalability = test_scalability
        
        # Validate configuration
        if num_nodes < 1:
            raise ValueError("System must have at least 1 node")
        if num_apps < 1:
            raise ValueError("System must have at least 1 application")
        if num_topics < 1:
            raise ValueError("System must have at least 1 topic")
        if num_brokers < 0:
            raise ValueError("Number of brokers cannot be negative")
            
        # Print configuration summary
        print("System Configuration:")
        if test_scalability:
            print("  Running scalability tests...")
        elif generate_dataset:
            print("  Generating synthetic datasets...")
            print("  ------------------------")
        else:
            if dataset_file:
                print("  Using provided dataset file.")
                print(f"  Dataset File: {dataset_file if dataset_file else 'N/A'}")
            elif ros2_data_file:
                print("  Using provided ROS2 data file.")
                print(f"  ROS2 Data File: {ros2_data_file if ros2_data_file else 'N/A'}")
            elif import_graph_from_file:
                print("  Importing graph from file.")
            else:
                print("  Generating synthetic dataset.")
                print("  ------------------------")
                print(f"  Nodes: {num_nodes}")
                print(f"  Applications: {num_apps}")
                print(f"  Topics: {num_topics}")
                print(f"  Brokers: {num_brokers}")
            print(f"  Verbose Output: {print_graph}")
            print(f"  Export Graph to files: {export_graph_to_file}")
            print("  ------------------------")

# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Publish-Subscribe System Graph Model')
    parser.add_argument('--nodes', type=int, default=5, help='Number of nodes/machines (default: 5)')
    parser.add_argument('--apps', type=int, default=25, help='Number of applications (default: 25)')
    parser.add_argument('--topics', type=int, default=50, help='Number of topics (default: 50)')
    parser.add_argument('--brokers', type=int, default=2, help='Number of brokers (default: 2)')
    parser.add_argument('--import_dataset', type=str, default="", help='Path to the dataset file (if any)') 
    parser.add_argument('--generate_dataset', action='store_true', help='Generate a synthetic dataset (default: False)') 
    parser.add_argument('--import_from_ros2', type=str, default="", help='Path to the ROS2 JSON data file')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output (default: False)')
    parser.add_argument('--input', action='store_true', help='Import graph from file instead of generating it')
    parser.add_argument('--output', action='store_true', help='Export graph to file after generation')
    parser.add_argument('--test_scalability', action='store_true', help='Run scalability tests')
    args = parser.parse_args()
    
    return SystemConfig(
        num_nodes=args.nodes,
        num_apps=args.apps,
        num_topics=args.topics,
        num_brokers=args.brokers,
        dataset_file=args.import_dataset,
        generate_dataset=args.generate_dataset,
        ros2_data_file=args.import_from_ros2,
        print_graph=args.verbose,
        import_graph_from_file=args.input,
        export_graph_to_file=args.output,
        test_scalability=args.test_scalability
    )

def test_scalability(graph_builder, graph_exporter):
    scale_tests = [
        (10, 50, 100, 5),    # Small
        (50, 200, 500, 10),   # Medium
        (100, 500, 1000, 20), # Large
        (200, 1000, 2000, 40) # Very Large
    ]
    
    for num_nodes, num_apps, num_topics, num_brokers in scale_tests:
        start_time = time.time()
        graph_config = {
            "num_nodes": num_nodes,
            "num_apps": num_apps,
            "num_topics": num_topics,
            "num_brokers": num_brokers
            }
        graph_builder.generate_synthetic_dataset(graph_config)
        execution_time = time.time() - start_time
        print(f"Scale: {num_nodes}N/{num_apps}A/{num_topics}T/{num_brokers}B - Time: {execution_time:.2f}s")

        # Import dataset to Neo4j
        start_time = time.time()
        graph_builder.import_dataset_to_neo4j()
        execution_time = time.time() - start_time
        print(f"Import Time: {execution_time:.2f}s")

        # Export and measure memory usage
        graph = graph_exporter.export_graph()
        print(f"Graph size: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

        # Clean up for next test
        graph_builder.clear_graph()



def generate_all_datasets():
    """Generate all three scale datasets"""
    generator = DatasetGenerator()
    
    # Generate datasets
    datasets = {
        'small': generator.generate_small_scale(),
        'medium': generator.generate_medium_scale(),
        'large': generator.generate_large_scale()
    }
    
    # Save to files
    for scale, dataset in datasets.items():
        filename = f'output/dataset_{scale}_scale.json'
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"\n{scale.upper()} SCALE DATASET:")
        print(f"  Scenario: {dataset['metadata']['scenario']}")
        print(f"  Nodes: {dataset['metadata']['statistics']['total_nodes']}")
        print(f"  Applications: {dataset['metadata']['statistics']['total_applications']}")
        print(f"  Topics: {dataset['metadata']['statistics']['total_topics']}")
        print(f"  Brokers: {dataset['metadata']['statistics']['total_brokers']}")
        print(f"  Saved to: {filename}")
    
    return datasets

def test_specific_configuration(graph_builder, graph_exporter, config):
    # Define the file path and import/export options
    input_file_dir = "input/"
    import_graph_from_file = config.import_graph_from_file
    output_file_dir = "output/"
    export_graph_to_file = config.export_graph_to_file
    
    # Start timer for graph generation
    print("Starting graph generation...")
    start_time = time.time()

    try:
        if config.dataset_file:
            graph_builder.import_dataset_from_file(config.dataset_file)
        elif import_graph_from_file:
            graph_builder.import_graph_from_file(input_file_dir)
        elif config.ros2_data_file:
                graph_builder.build_graph_from_ros2_data_file(config.ros2_data_file)
        else:
            graph_config = {
                "num_nodes": config.num_nodes,
                "num_apps": config.num_apps,
                "num_topics": config.num_topics,
                "num_brokers": config.num_brokers
            }
            graph_builder.generate_synthetic_dataset(graph_config)
            graph_builder.import_dataset_to_neo4j()

        # Print the graph structure if verbose output is enabled
        if config.print_graph:
            graph_builder.print_graph()
        
        # Export the graph to files if the option is enabled
        if export_graph_to_file:
            print("Exporting graph to file...")
            graph_builder.export_graph(output_file_dir)
    finally:
        graph_builder.close()

    # End timer and print duration
    end_time = time.time()
    print(f"Graph generation completed in {end_time - start_time:.2f} seconds.")
    print("Graph generation process finished successfully.")

    # Export the graph
    graph = graph_exporter.export_graph()

    # Print the graph
    print("Generated Graph:")
    graph_exporter.print_graph(graph)

if __name__ == "__main__":
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "password"
    graph_builder = GraphBuilder(uri, user, password)
    graph_exporter = GraphExporter()
    
    # Parse command line arguments
    config = parse_args()  # Default: 4 nodes, 10 applications, 25 topics

    if hasattr(config, 'test_scalability') and config.test_scalability:
        test_scalability(graph_builder, graph_exporter)
    elif hasattr(config, 'generate_dataset') and config.generate_dataset:
            generate_all_datasets()
            print("Dataset generation complete!")
            print("Files created:")
            print("  - dataset_small_scale.json")
            print("  - dataset_medium_scale.json")
            print("  - dataset_large_scale.json")
    else:
        test_specific_configuration(graph_builder, graph_exporter, config) 
    