#!/usr/bin/env python3
"""
Full Pipeline Demo: Software-as-a-Graph
---------------------------------------
This script demonstrates the complete lifecycle of the framework:
1. GENERATE: Creates a realistic 'Financial Trading' pub-sub system.
2. ANALYZE: Identifies critical nodes (SPOFs) and bridges using Graph Theory.
3. SIMULATE: Runs traffic simulation, injects a failure, and measures impact.
4. VISUALIZE: Generates interactive dashboards and static topology maps.
"""

import os
import sys
import json
import logging
import time
import asyncio
import networkx as nx
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent / '..'))

# Import Core Modules
from src.core.graph_generator import GraphGenerator, GraphConfig
from src.core.graph_model import GraphModel, ComponentType
from src.orchestration.analysis_orchestrator import AnalysisOrchestrator
from src.simulation.lightweight_dds_simulator import LightweightDDSSimulator
from src.simulation.enhanced_failure_simulator import FailureSimulator
from src.visualization.graph_visualizer import GraphVisualizer, VisualizationConfig, LayoutAlgorithm, ColorScheme
from src.visualization.metrics_dashboard import MetricsDashboard

# Configuration
OUTPUT_DIR = Path("demo_output")
SYSTEM_FILE = OUTPUT_DIR / "financial_system.json"
ANALYSIS_FILE = OUTPUT_DIR / "analysis_results.json"
SIM_RESULTS_FILE = OUTPUT_DIR / "simulation_results.json"

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DEMO")

def setup_dirs():
    if OUTPUT_DIR.exists():
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def step_1_generate():
    print("\n" + "="*60)
    print("STEP 1: GENERATING SYSTEM MODEL")
    print("="*60)
    
    config = GraphConfig(
        scale='medium',           # ~15 nodes, 50 apps
        scenario='financial',     # Trading patterns (Market Data -> Algo -> Execution)
        num_nodes=10,
        num_applications=40,
        num_topics=20,
        num_brokers=3,
        antipatterns=['tight_coupling'], # Inject some flaws for analysis to find
        seed=12345
    )
    
    generator = GraphGenerator(config)
    graph_dict = generator.generate()
    
    with open(SYSTEM_FILE, 'w') as f:
        json.dump(graph_dict, f, indent=2)
        
    logger.info(f"Generated system with {len(graph_dict['nodes'])} nodes and {len(graph_dict['applications'])} apps.")
    logger.info(f"Saved to {SYSTEM_FILE}")
    return graph_dict

def graph_dict_to_networkx(data):
    """Helper to convert JSON dict to NetworkX for Analysis"""
    G = nx.DiGraph()
    
    # Nodes
    for n in data.get('nodes', []): G.add_node(n['id'], type='Node', **n)
    for a in data.get('applications', []): G.add_node(a['id'], **a)
    for t in data.get('topics', []): G.add_node(t['id'], type='Topic', **t)
    for b in data.get('brokers', []): G.add_node(b['id'], type='Broker', **b)
        
    # Edges
    rels = data.get('relationships', {})
    for r in rels.get('runs_on', []): G.add_edge(r['from'], r['to'], type='RUNS_ON')
    for r in rels.get('publishes_to', []): G.add_edge(r['from'], r['to'], type='PUBLISHES_TO', **r)
    for r in rels.get('subscribes_to', []): G.add_edge(r['from'], r['to'], type='SUBSCRIBES_TO', **r)
    for r in rels.get('routes', []): G.add_edge(r['from'], r['to'], type='ROUTES', **r)
    
    return G

def step_2_analyze(graph_dict):
    print("\n" + "="*60)
    print("STEP 2: ANALYZING CRITICALITY & STRUCTURE")
    print("="*60)
    
    G = graph_dict_to_networkx(graph_dict)
    
    # Initialize Orchestrator
    orchestrator = AnalysisOrchestrator(
        output_dir=str(OUTPUT_DIR),
        enable_qos=True
    )
    
    results = orchestrator.analyze_graph(G)
    orchestrator.export_results("analysis_results.json")
    
    # Extract key insights
    critical_nodes = results['criticality_scores']['summary']['critical_count']
    bridges = results['structural_analysis'].get('bridges', []) # Assuming structure analyzer returns list
    
    logger.info(f"Analysis Complete: Found {critical_nodes} critical components.")
    
    # Extract scores for visualization later
    scores = {}
    raw_scores = results['criticality_scores']['top_critical_components']
    for score_data in raw_scores:
        scores[score_data['component']] = score_data['composite_score']
            
    return scores, results

async def step_3_simulate():
    print("\n" + "="*60)
    print("STEP 3: SIMULATING TRAFFIC & FAILURE")
    print("="*60)
    
    sim = LightweightDDSSimulator()
    sim.load_from_json(str(SYSTEM_FILE))
    
    # 1. Baseline Run
    logger.info("Running Baseline Simulation (5s)...")
    baseline_stats = await sim.run_simulation(5)
    sim.print_summary(baseline_stats)
    
    # 2. Failure Run (Fail Broker B1)
    logger.info("Injecting Failure: Broker B1 crashes...")
    
    # Reset simulator for clean run
    sim = LightweightDDSSimulator()
    sim.load_from_json(str(SYSTEM_FILE))
    
    fail_sim = FailureSimulator()
    # Schedule B1 failure at T+2s
    
    async def run_failure_scenario():
        sim_task = asyncio.create_task(sim.run_simulation(10))
        
        await asyncio.sleep(2)
        logger.warning(">>> INJECTING FAILURE NOW <<<")
        fail_sim.inject_failure(sim, "B1", ComponentType.BROKER, "complete")
        
        await sim_task
        return sim_task.result()
        
    failure_stats = await run_failure_scenario()
    sim.print_summary(failure_stats)
    
    return baseline_stats, failure_stats

def step_4_visualize(graph_dict, analysis_scores, sim_stats):
    print("\n" + "="*60)
    print("STEP 4: VISUALIZING RESULTS")
    print("="*60)
    
    G = graph_dict_to_networkx(graph_dict)
    
    # 1. Visualizer
    viz = GraphVisualizer(logger)
    config = VisualizationConfig(
        layout=LayoutAlgorithm.SPRING,
        color_scheme=ColorScheme.CRITICALITY
    )
    
    viz.visualize_graph(
        G, 
        str(OUTPUT_DIR / "topology_criticality.png"),
        config=config,
        criticality_scores=analysis_scores,
        title="Financial System - Criticality Heatmap"
    )
    
    # 2. Dashboard
    dashboard = MetricsDashboard()
    dashboard.create_dashboard(
        G, 
        {'criticality': analysis_scores}, 
        str(OUTPUT_DIR / "interactive_dashboard.html")
    )
    
    logger.info(f"Visualizations saved to {OUTPUT_DIR}")

def main():
    setup_dirs()
    
    # 1. Generate
    graph_dict = step_1_generate()
    
    # 2. Analyze
    scores, analysis_results = step_2_analyze(graph_dict)
    
    # 3. Simulate
    baseline, failure = asyncio.run(step_3_simulate())
    
    # 4. Visualize
    step_4_visualize(graph_dict, scores, failure)
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print(f"Artifacts available in: {OUTPUT_DIR.absolute()}")
    print("="*60)

if __name__ == "__main__":
    main()