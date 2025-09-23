#!/usr/bin/env python3
import time
import pandas as pd
from src.GraphExporter import GraphExporter
from src.QosAwareComponentAnalyzer import QosAwareComponentAnalyzer
from src.GraphAnalyzer import GraphAnalyzer
from src.ThresholdCalculator import ThresholdCalculator
from src.ReachabilityAnalyzer import ReachabilityAnalyzer
from src.CriticalComponentValidator import ValidationOrchestrator

# Example usage
if __name__ == "__main__":
   # Initialize components
   exporter = GraphExporter()
   graph = exporter.export_graph()
   
   qos_analyzer = QosAwareComponentAnalyzer(exporter)
   graph_analyzer = GraphAnalyzer(qos_analyzer)
   reachability_analyzer = ReachabilityAnalyzer(graph)
   
   # Run criticality analysis
   criticality_scores = graph_analyzer.analyze_comprehensive_criticality(graph)
   
   # Run validation
   orchestrator = ValidationOrchestrator(graph_analyzer, reachability_analyzer)
   validation_results = orchestrator.run_complete_validation(
       graph,
       criticality_scores,
       use_synthetic_data=True  # Use synthetic data for demonstration
   )
   
   # Print summary
   print("\n" + "="*60)
   print("VALIDATION COMPLETE")
   print("="*60)
   print(f"Overall Confidence: {validation_results['overall_confidence']:.1%}")
   print("\nValidation Methods Completed:")
   for method in validation_results['validation_methods']:
       print(f"- {method['method']}: {method['status']}")
   print("\nTop Recommendations:")
   for rec in validation_results['recommendations'][:3]:
       print(f"- {rec}")
   print("="*60)