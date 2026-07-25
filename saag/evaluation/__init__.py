"""Shared evaluation contract for every predictor variant.

Every reported number in the paper — Table 3 (in-distribution), Table 4 (LOSO /
k-fold) and Table 5 (per-node-type) — must come from one implementation applied
to one declared node population. This package owns that implementation so
``reproduce/main_table.py``, ``cli/loso_evaluate.py`` and ``cli/kfold_evaluate.py``
cannot drift apart.
"""

from saag.evaluation.metrics import (
    EVAL_POPULATIONS,
    UNDEFINED,
    compute_inductive_metrics,
    resolve_eval_keys,
)

__all__ = [
    "EVAL_POPULATIONS",
    "UNDEFINED",
    "compute_inductive_metrics",
    "resolve_eval_keys",
]
