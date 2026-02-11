"""
Graph Generator Application Service

Orchestrates graph generation using domain services.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import yaml

from src.domain.models.statistics import GraphConfig
from src.domain.services.generator import StatisticalGraphGenerator


class GenerationService:
    """Application service for generating system graphs."""
    
    def __init__(
        self,
        scale: str = "medium",
        seed: int = 42,
        config: Optional[GraphConfig] = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = GraphConfig.from_scale(scale, seed)
            
        self.generator = StatisticalGraphGenerator(self.config)

    def generate(self) -> Dict[str, Any]:
        """Generate a complete graph."""
        return self.generator.generate()


def load_config(path: Path) -> GraphConfig:
    """Load graph configuration from a YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return GraphConfig.from_yaml(data)


def generate_graph(scale: str = "medium", **kwargs: Any) -> Dict[str, Any]:
    """Convenience function to generate a graph."""
    # check if 'config' is in kwargs, if so handle it
    config = kwargs.get('config')
    seed = kwargs.get('seed', 42)
        
    service = GenerationService(scale=scale, seed=seed, config=config)
    return service.generate()
