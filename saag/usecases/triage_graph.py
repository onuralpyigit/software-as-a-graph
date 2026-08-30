from typing import Any, Optional, Sequence
from saag.analysis.triage import triage, TriageResult


class TriageGraphUseCase:
    """
    Use case for the Triage bridge: scopes Pathway A's (RM) root-cause
    diagnosis to the components Pathway B (GNN, or RM in cold start)
    flagged as Top-K.

    Accepts a prediction result (Predict stage output) as a parameter and
    never reads raw runtime data from the repository, matching
    PredictGraphUseCase's independence guarantee.
    """

    def execute(
        self,
        prediction_result: Any,
        k: int = 10,
        layer: str = "system",
        node_types: Optional[Sequence[str]] = None,
    ) -> TriageResult:
        return triage(prediction_result, k=k, layer=layer, node_types=node_types)
