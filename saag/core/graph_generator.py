"""
Compatibility shim: src.core.graph_generator

Maps the old GraphGenerator API to the new GenerationService.
"""
from tools.generation.service import GenerationService as GraphGenerator

__all__ = ["GraphGenerator"]
