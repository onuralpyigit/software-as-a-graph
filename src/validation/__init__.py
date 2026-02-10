"""
Compatibility shim: src.validation

Re-exports validation classes from their new locations in the hexagonal architecture.
Classes that no longer exist (ValidationPipeline, QuickValidator) are stubbed
to raise descriptive errors instead of ImportErrors.
"""
from src.domain.models.validation.metrics import ValidationTargets
from src.domain.config.layers import LAYER_DEFINITIONS


class ValidationPipeline:
    """Stub — full pipeline not yet ported to new architecture."""

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "ValidationPipeline has not been ported to the new hexagonal architecture. "
            "Use ValidationService instead."
        )


class QuickValidator:
    """Stub — quick validator not yet ported to new architecture."""

    def __init__(self, **kwargs):
        raise NotImplementedError(
            "QuickValidator has not been ported to the new hexagonal architecture. "
            "Use ValidationService instead."
        )


__all__ = [
    "ValidationPipeline", "QuickValidator",
    "ValidationTargets", "LAYER_DEFINITIONS",
]
