from typing import Protocol, Dict, Any, List

class SimulationDataProvider(Protocol):
    """Port for simulation graph data retrieval."""
    
    def load_simulation_graph(self, layer: str = "system") -> Dict[str, Any]:
        """
        Load graph data suitable for simulation.
        Returns a dictionary containing nodes, relationships, and metadata.
        """
        ...
