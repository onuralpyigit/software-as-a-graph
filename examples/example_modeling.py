import sys
from pathlib import Path
import logging
import json

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / '..'))

from src.core.graph_builder import GraphBuilder

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example: Build from JSON
    print("Example: Building graph from dataset.json")
    print("=" * 60)
    
    builder = GraphBuilder()
    
    try:
        # Build from JSON file
        model = builder.build_from_json('../input/dataset.json')
        
        # Print summary
        summary = model.summary()
        print("\n" + "=" * 60)
        print("GRAPH MODEL SUMMARY")
        print("=" * 60)
        print(f"Total Nodes: {summary['total_nodes']}")
        print(f"  - Applications: {summary['applications']}")
        print(f"  - Topics: {summary['topics']}")
        print(f"  - Brokers: {summary['brokers']}")
        print(f"  - Infrastructure: {summary['infrastructure_nodes']}")
        print(f"\nTotal Edges: {summary['total_edges']}")
        print(f"  - Publishes: {summary['publishes']}")
        print(f"  - Subscribes: {summary['subscribes']}")
        print(f"  - Routes: {summary['routes']}")
        print(f"  - Runs On: {summary['runs_on']}")
        print(f"  - Connects To: {summary['connects_to']}")
        print(f"  - Depends On: {summary['depends_on']}")
        
        # Show errors/warnings
        if builder.errors:
            print(f"\n⚠️  {len(builder.errors)} errors occurred during build")
        if builder.warnings:
            print(f"⚠️  {len(builder.warnings)} warnings occurred during build")
        
        print("\n✅ Graph built successfully!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please ensure input/dataset.json exists")
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON Error: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()