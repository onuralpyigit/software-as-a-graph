"""
Complexity-derived processing latency processor.

Derives per-Application-subscriber processing latency from SonarQube static
analysis metrics stored in ComponentInfo.properties.

Rationale
---------------------
The event simulator previously applied a single uniform ``subscribe_latency``
to every DELIVER event, regardless of how complex the receiving application
actually is.  This module replaces that uniform constant with a
**complexity-derived per-component value**:

    pt(v) = base_latency * (1 + alpha * c_norm(v))

where:

  base_latency            scenario.base_processing_latency  [seconds]
  alpha                   scenario.complexity_scale_factor  (default 1.0)
  c_norm(v) in [0, 1]     normalised cyclomatic complexity of v

The normalisation is performed **once per simulation run** over the set of
Application subscribers that have a ``complexity`` property, so the hot path
(inside _handle_deliver) is a plain dict lookup.

Library contribution
---------------------
Libraries are not message-flow endpoints and do not appear in the DELIVER
chain.  However, their complexity still burdens the Application that USES
them.  The processor optionally folds library complexity into the Application's
effective processing time via an additive library penalty:

    pt_eff(v) = pt(v) + beta * sum(c_norm(lib))   for lib in USES(v)

where beta = scenario.library_complexity_weight (default 0.3).
This keeps Library nodes off the event queue while still reflecting their
runtime cost.

The processor degrades gracefully:
  - If ``complexity`` is absent from a component's properties, ``base_latency``
    is used unchanged (same as the previous uniform behaviour).
  - If *no* Application has a complexity property, the processor is effectively
    a no-op; ``enrich_processing_time`` can stay True without side effects.
"""

from typing import Dict, Any
from .graph import SimulationGraph
from .models import EventScenario


class ComplexityProcessor:
    """Processes component complexity to derive processing latencies."""
    
    def __init__(self, graph: SimulationGraph, scenario: EventScenario):
        self.graph = graph
        self.scenario = scenario

    def process(self) -> None:
        """Calculate and inject processing_latency into Application properties."""
        if not self.scenario.enrich_processing_time:
            return
            
        app_complexities = {}
        lib_complexities = {}

        # 1. Gather all complexities
        for comp_id, comp in self.graph.components.items():
            if 'complexity' in comp.properties:
                try:
                    comp_val = float(comp.properties['complexity'])
                    if comp.type == 'Application':
                        app_complexities[comp_id] = comp_val
                    elif comp.type == 'Library':
                        lib_complexities[comp_id] = comp_val
                except (ValueError, TypeError):
                    pass

        # If no Application has complexity, gracefully degrade to no-op
        if not app_complexities:
            return

        # 2. Normalize Application complexities
        app_min = min(app_complexities.values())
        app_max = max(app_complexities.values())
        app_range = app_max - app_min if app_max > app_min else 1.0
        app_norm = {k: (v - app_min) / app_range for k, v in app_complexities.items()}

        # 3. Normalize Library complexities
        lib_norm = {}
        if lib_complexities:
            lib_min = min(lib_complexities.values())
            lib_max = max(lib_complexities.values())
            lib_range = lib_max - lib_min if lib_max > lib_min else 1.0
            lib_norm = {k: (v - lib_min) / lib_range for k, v in lib_complexities.items()}

        # 4. Compute effective processing time and inject into properties
        base_latency = self.scenario.base_processing_latency
        alpha = self.scenario.complexity_scale_factor
        beta = self.scenario.library_complexity_weight

        for app_id, c_norm in app_norm.items():
            comp = self.graph.components[app_id]
            pt = base_latency * (1 + alpha * c_norm)

            lib_penalty = 0.0
            if beta > 0.0:
                out_edges = self.graph.out_edges.get(app_id, {})
                for target_id, relation in out_edges.items():
                    if relation == "USES" and target_id in lib_norm:
                        lib_penalty += lib_norm[target_id]

            pt_eff = pt + (beta * lib_penalty)
            comp.properties['processing_latency'] = pt_eff
