"""
Compatibility shim: src.core.graph_generator

Maps the old GraphGenerator API to the new GenerationService.
"""
from src.application.services.generation_service import GenerationService as GraphGenerator

__all__ = ["GraphGenerator"]
