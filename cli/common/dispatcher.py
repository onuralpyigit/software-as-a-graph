"""
Dispatcher module for in-process execution of pipeline stages.
Provides a unified interface for both CLI scripts and the orchestrator.

Only dispatch_generate is live — cli/generate_graph.py and cli/run.py's
--generate/--all(+--config/--scale) path both call it. The former
dispatch_import/dispatch_analyze/dispatch_predict/dispatch_simulate/
dispatch_visualize had no callers anywhere in the codebase (cli/run.py and
the per-stage CLI scripts call saag.Pipeline/saag.Client directly instead),
and several would have raised on first use: dispatch_predict/dispatch_simulate/
dispatch_visualize constructed their use cases with a bare repository where
the use case's __init__ requires a service (AttributeError on first
self.service.* call), and dispatch_predict's only reachable branch always
raised ValueError. Removed rather than repaired, since nothing calls them.
"""
import json
from pathlib import Path
from typing import Dict, Any

# Import services and use cases lazily to avoid circular imports and heavy start-up costs
# if only one stage is needed.

def dispatch_generate(args) -> Dict[str, Any]:
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
