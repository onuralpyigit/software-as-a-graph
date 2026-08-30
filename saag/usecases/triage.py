"""
saag/usecases/triage.py

The Triage Bridge Use Case (Pathway B ──(Triage)──► Pathway A).
Coordinates high-throughput blast-radius filtering from Pathway B (HGL)
with targeted, deep root-cause attribution from Pathway A (ISO/IEC RM).
"""
from typing import Any, List, Optional, Sequence, Tuple
from saag.analysis.triage import triage, select_top_k, TriageResult, TriageEntry


class TriageUseCase:
    """
    Use Case for the Triage Bridge:
    Top-K Critical Components Shortlist ──(Triage)──► Architectural Root-Cause Profile.

    INDEPENDENCE GUARANTEE:
    Accepts prediction and diagnostic results and evaluates them without accessing
    runtime simulation data.
    """

    def execute(
        self,
        prediction_result: Any,
        k: int = 10,
        layer: str = "system",
        node_types: Optional[Sequence[str]] = None,
    ) -> TriageResult:
        """
        Scope Pathway A's root-cause attribution to Pathway B's Top-K most critical components.

        Args:
            prediction_result: Output from Pathway B (GNN result or RM result in cold start)
            k: Number of high-risk components to shortlist
            layer: Subgraph layer (default 'system')
            node_types: Optional filter on component types

        Returns:
            TriageResult containing annotated Top-K components with patterns and remediation roles.
        """
        return triage(prediction_result, k=k, layer=layer, node_types=node_types)

    def extract_shortlist(
        self,
        prediction_result: Any,
        k: int = 10,
        node_types: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float]]:
        """
        Extract raw Top-K (component_id, score) pairs.
        """
        return select_top_k(prediction_result, k=k, node_types=node_types)


# Backward-compatible alias
TriageGraphUseCase = TriageUseCase
