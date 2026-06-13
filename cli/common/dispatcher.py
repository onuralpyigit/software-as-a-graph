"""
Dispatcher module for in-process execution of pipeline stages.
Provides a unified interface for both CLI scripts and the orchestrator.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

from saag.infrastructure import create_repository
from .console import ConsoleDisplay

# Import services and use cases lazily to avoid circular imports and heavy start-up costs
# if only one stage is needed.

def dispatch_generate(args: argparse.Namespace) -> Dict[str, Any]:
    """Dispatch graph generation stage."""
    from tools.generation import GenerationService, load_config, generate_graph
    
    graph_data = {}
    connection_density = getattr(args, 'connection_density', None)
    if hasattr(args, 'config') and args.config:
        config = load_config(Path(args.config))
        if connection_density is not None:
            config.connection_density = connection_density
        service = GenerationService(config=config)
        graph_data = service.generate()
    else:
        scale = getattr(args, 'scale', 'medium') or 'medium'
        seed = getattr(args, 'seed', 42)
        domain = getattr(args, 'domain', None)
        scenario = getattr(args, 'scenario', None)
        graph_data = generate_graph(
            scale=scale, 
            seed=seed,
            domain=domain,
            scenario=scenario,
            connection_density=connection_density
        )
    
    if hasattr(args, 'output') and args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(graph_data, f, indent=2)
            
    return graph_data


def dispatch_import(repo, args: argparse.Namespace, graph_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Dispatch graph import stage."""
    from saag.usecases import ModelGraphUseCase
    
    if graph_data is None:
        if not hasattr(args, 'input') or not args.input:
            raise ValueError("No input data or file specified for import.")
        input_path = Path(args.input)
        with open(input_path) as f:
            graph_data = json.load(f)
            
    use_case = ModelGraphUseCase(repo)
    clear_db = getattr(args, 'clear', False) or getattr(args, 'clean', False)
    stats_obj = use_case.execute(graph_data, clear=clear_db)
    
    return stats_obj.details if stats_obj.details else {}


def dispatch_analyze(repo, args: argparse.Namespace):
    """Dispatch structural analysis stage."""
    from saag.usecases.multi_layer_analysis import MultiLayerAnalysisUseCase
    
    layers_arg = getattr(args, 'layers', None)
    if layers_arg:
        layers = [l.strip() for l in layers_arg.split(",")]
    elif getattr(args, 'all', False):
        layers = ["app", "infra", "mw", "system"]
    else:
        layers = [getattr(args, 'layer', 'system')]

    use_case = MultiLayerAnalysisUseCase(repo)
    analysis_result = use_case.execute(
        layers=layers,
        use_ahp=getattr(args, 'use_ahp', False),
        normalization_method=getattr(args, 'norm', 'robust'),
        winsorize=getattr(args, 'winsorize', True),
        winsorize_limit=getattr(args, 'winsorize_limit', 0.05),
        gnn_model=getattr(args, 'gnn_model', None),
        equal_weights=getattr(args, 'equal_weights', False),
        ahp_shrinkage=getattr(args, 'ahp_shrinkage', 0.7),
        run_sensitivity=getattr(args, 'sensitivity', False)
    )

    if getattr(args, 'output', None):
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(analysis_result.to_dict(), f, indent=2, default=str)
            
    return analysis_result


def dispatch_predict(repo, args: argparse.Namespace):
    """Dispatch GNN prediction stage."""
    from saag.prediction import GNNService, extract_structural_metrics_dict, extract_rmav_scores_dict
    from saag.usecases import AnalyzeGraphUseCase, PredictGraphUseCase

    def load_json(path):
        if not path: return None
        p = Path(path)
        if not p.exists(): return None
        with open(p) as f: return json.load(f)

    structural_raw = load_json(getattr(args, 'structural', None))
    rmav_raw = load_json(getattr(args, 'rmav', None))
    simulation_raw = load_json(getattr(args, 'simulated', None))
    
    # Flattening logic omitted for brevity, but should be here if needed from cli/predict_graph.py
    # For now, we assume standard pipeline usage.

    layer = getattr(args, 'layer', 'app')
    checkpoint = getattr(args, 'checkpoint', None)
    if not checkpoint:
        raise ValueError("No checkpoint specified for prediction.")

    # If no pre-computed structural analysis, run it
    if not structural_raw:
        analyze_uc = AnalyzeGraphUseCase(repo)
        predict_uc = PredictGraphUseCase(repo)
        layer_result = analyze_uc.execute(layer)
        nx_graph = layer_result.graph
        structural_dict = extract_structural_metrics_dict(layer_result.structural)
        
        if not rmav_raw:
            quality_res, _ = predict_uc.execute(layer)
            rmav_dict = extract_rmav_scores_dict(quality_res)
        else:
            rmav_dict = extract_rmav_scores_dict(rmav_raw)
    else:
        raise ValueError(
            "GNN prediction requires a real NetworkX graph source containing actual "
            "topology and relationships. Running prediction solely from pre-computed "
            "structural/RMAV JSON files is not supported because GNN message passing "
            "requires edge connections and node types."
        )

    service = GNNService.from_checkpoint(checkpoint, graph=nx_graph)
    result = service.predict(
        graph=nx_graph,
        structural_metrics=structural_dict,
        rmav_scores=rmav_dict,
        simulation_results=simulation_raw
    )

    if getattr(args, 'output', None):
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
            
    return result


def dispatch_simulate(repo, args: argparse.Namespace):
    """Dispatch simulation stage."""
    from saag.usecases import SimulateGraphUseCase, SimulationMode
    from saag.simulation.models import FailureMode
    
    use_case = SimulateGraphUseCase(repo)
    command = getattr(args, 'command', 'report')
    
    if command == "report":
        layers = [l.strip() for l in getattr(args, 'layers', 'app,infra,mw,system').split(",")]
        result = use_case.execute(
            mode=SimulationMode.REPORT,
            layers=layers,
            classify_edges=getattr(args, 'edges', False)
        )
    elif command == "event":
        source = getattr(args, 'source', 'all')
        if source == 'all':
             result = use_case.execute(
                mode=SimulationMode.EVENT,
                source_app="all",
                num_messages=getattr(args, 'messages', 100),
                duration=getattr(args, 'duration', 10.0),
                layer=getattr(args, 'layer', 'system')
            )
        else:
            result = use_case.execute(
                mode=SimulationMode.EVENT,
                source_app=source,
                num_messages=getattr(args, 'messages', 100),
                duration=getattr(args, 'duration', 10.0)
            )
    elif command == "failure":
        if getattr(args, 'exhaustive', False):
            result = use_case.execute(
                layer=getattr(args, 'layer', 'system'),
                mode=SimulationMode.EXHAUSTIVE
            )
        elif getattr(args, 'pairwise', False):
             result = use_case.execute(
                mode=SimulationMode.PAIRWISE,
                layer=getattr(args, 'layer', 'system'),
                cascade_probability=getattr(args, 'cascade_prob', 1.0),
                failure_mode=FailureMode[getattr(args, 'failure_mode', 'CRASH')]
            )
        else:
             result = use_case.execute(
                target_id=getattr(args, 'target', None),
                layer=getattr(args, 'layer', 'system'),
                mode=SimulationMode.SINGLE
            )
    elif command == "classify":
        result = use_case.execute(
            mode=SimulationMode.CLASSIFY,
            layer=getattr(args, 'layer', 'system'),
            edges=getattr(args, 'edges', False),
            k_factor=getattr(args, 'k_factor', 1.5)
        )
    else:
        raise ValueError(f"Unknown simulation command: {command}")

    if getattr(args, 'output', None):
        with open(args.output, "w") as f:
            if isinstance(result, list):
                json.dump([r.to_dict() if hasattr(r, 'to_dict') else r for r in result], f, indent=2)
            else:
                json.dump(result.to_dict(), f, indent=2)
                
    return result


def dispatch_validate(repo, args: argparse.Namespace):
    """Dispatch validation stage."""
    from saag.usecases import ValidateGraphUseCase
    from saag.validation import ValidationTargets
    
    use_case = ValidateGraphUseCase(repo)
    layers = getattr(args, 'layer', 'app,infra,mw,system')
    if layers:
        layers_to_validate = [l.strip() for l in layers.split(",")]
    else:
        layers_to_validate = ["app", "infra", "mw", "system"]
        
    result = use_case.execute(layers=layers_to_validate)
    
    if getattr(args, 'output', None):
        with open(args.output, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
            
    return result


def dispatch_visualize(repo, args: argparse.Namespace):
    """Dispatch visualization stage."""
    from saag.usecases import VisualizeGraphUseCase, VisOptions
    from saag.visualization import LAYER_DEFINITIONS
    
    use_case = VisualizeGraphUseCase(repo)
    
    if getattr(args, 'all', False):
        layers = list(LAYER_DEFINITIONS.keys())
    elif getattr(args, 'layer', None):
        layers = [args.layer]
    else:
        layers = [l.strip() for l in getattr(args, 'layers', 'app,infra,system').split(",")]
        
    options = VisOptions(
        include_network=not getattr(args, 'no_network', False),
        include_matrix=not getattr(args, 'no_matrix', False),
        include_validation=not getattr(args, 'no_validation', False),
        antipatterns_file=getattr(args, 'antipatterns', None),
        multi_seed=getattr(args, 'multi_seed', 0)
    )
    
    output_path = use_case.execute(
        output_file=getattr(args, 'output', 'dashboard.html'),
        layers=layers,
        options=options
    )
    
    return str(output_path)
