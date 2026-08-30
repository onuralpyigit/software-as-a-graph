"""
saag/usecases/triage_graph.py

Backward-compatibility shim for TriageGraphUseCase.
Real implementation lives in saag.usecases.triage.
"""
from saag.usecases.triage import TriageUseCase, TriageGraphUseCase, TriageResult, TriageEntry

__all__ = [
    "TriageUseCase",
    "TriageGraphUseCase",
    "TriageResult",
    "TriageEntry",
]
