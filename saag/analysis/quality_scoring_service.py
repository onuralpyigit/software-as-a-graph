"""
Backward-compatibility shim.

QualityScoringService moved to saag.prediction.quality_scoring_service as part
of consolidating rule-based (RMAV) and ML (GNN) scoring into a single
Prediction Step (Step 3). Import from saag.prediction instead.
"""
from saag.prediction.quality_scoring_service import QualityScoringService

__all__ = ["QualityScoringService"]
