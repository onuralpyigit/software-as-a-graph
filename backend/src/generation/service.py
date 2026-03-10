"""
Graph Generation Service
"""
from typing import Dict, Any, Optional
from pathlib import Path
import yaml

from .models import GraphConfig
from .generator import StatisticalGraphGenerator


class GenerationService:
    """Service for generating system graphs."""
    
    def __init__(
        self,
        scale: str = "medium",
        seed: int = 42,
        config: Optional[GraphConfig] = None,
        domain: Optional[str] = None,
        scenario: Optional[str] = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = GraphConfig.from_scale(scale, seed)
            
        if domain is not None:
            self.config.domain = domain
        if scenario is not None:
            self.config.scenario = scenario
            
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
    config = kwargs.get('config')
    seed = kwargs.get('seed', 42)
    domain = kwargs.get('domain')
    scenario = kwargs.get('scenario')
        
    service = GenerationService(scale=scale, seed=seed, config=config, domain=domain, scenario=scenario)
    return service.generate()
