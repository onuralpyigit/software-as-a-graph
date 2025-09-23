#!/usr/bin/env python3
import argparse
import logging

from src.SystemAnalyzer import SystemAnalyzer

if __name__ == "__main__":
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Unified Pub-Sub System Analyzer')
    parser.add_argument('--visualize', action='store_true', help='Enable visualization')
    parser.add_argument('--simulate', action='store_true', default=True, 
                       help='Enable failure simulation')
    parser.add_argument('--output', type=str, default='output/', 
                       help='Output directory for results')
    parser.add_argument('--export', action='store_true', default=True,
                       help='Export results to files')
    
    args = parser.parse_args()
    
    # Initialize and run analyzer
    analyzer = SystemAnalyzer(output_dir=args.output)
    
    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis(
        enable_visualization=args.visualize,
        enable_simulation=args.simulate,
        export_results=args.export
    )
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Total execution time: {results['execution_time']:.2f} seconds")
    print(f"System resilience score: {results['reachability_analysis'].get('resilience_score', 'N/A'):.2f}")
    print(f"Critical components identified: {len(results['criticality_analysis']['critical_components'])}")
    print(f"Recommendations generated: {len(results['recommendations'])}")
    print(f"\nResults exported to: {args.output}")
    print("="*60)