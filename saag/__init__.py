"""
SoftwareAsAGraph (saag) Public SDK Facade
"""
import sys
from pathlib import Path

# Inject the backend directory into sys.path so that internal usecase imports resolve automatically.
# This prevents developers from needing PYTHONPATH hacks just to import `saag`.
_backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from .pipeline import Pipeline
from .client import Client
from .models import AnalysisResult, PredictionResult, ValidationResult

__all__ = ["Pipeline", "Client", "AnalysisResult", "PredictionResult", "ValidationResult"]
