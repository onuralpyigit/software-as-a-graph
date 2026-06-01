"""
QosConverter: Converts QoS values between custom and DDS formats.

This module is used for converting QoS and criticality values to string format.
QoS risk weights are defined here as DDS-standard constants.
Custom-to-DDS string mappings come from runtime.yaml (aggregator.qos_mappings).
"""

from typing import Dict, Optional, Tuple, Any

from common.runtime_config import get_runtime_config

_QOS_RISK_WEIGHTS: Dict[str, Dict[str, float]] = {
    "durability": {
        "PERSISTENT": 3.0,
        "TRANSIENT_LOCAL": 2.0,
        "TRANSIENT": 1.5,
        "VOLATILE": 1.0,
    },
    "reliability": {
        "RELIABLE": 2.0,
        "BEST_EFFORT": 1.0,
    },
    "transport_priority": {
        "URGENT": 4.0,
        "HIGH": 3.0,
        "MEDIUM": 2.0,
        "LOW": 1.0,
    },
}

_QOS_DEFAULT_WEIGHT = 1.0


def qos_risk_weight(dim: str, value: str, *, dds_mask: bool = True) -> float:
    """Return the risk weight for a QoS dimension value.

    When *dds_mask* is True the value is already in DDS format and is
    looked up directly.  When False the value is a custom string that
    is first resolved through ``aggregator.qos_mappings`` before lookup.
    """
    if not dds_mask:
        mappings = get_runtime_config().aggregator.qos_mappings
        dim_map = mappings.get(dim, {})
        value = dim_map.get(value, value)

    weight_map = _QOS_RISK_WEIGHTS.get(dim)
    if weight_map is None:
        return _QOS_DEFAULT_WEIGHT
    w = weight_map.get(value)
    return w if w is not None else _QOS_DEFAULT_WEIGHT


class QosConverter:
    """Converter class that transforms QoS values between custom and DDS formats."""
    
    @staticmethod
    def convert_qos(
        dur: Optional[Any] = None,
        rel: Optional[Any] = None,
        pri: Optional[Any] = None,
        boolean: Optional[bool] = None
    ) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """
        Converts Gateway QoS values to DDS format.
        
        Args:
            dur: Durability value
            rel: Reliability value
            pri: Transport priority value
            boolean: Boolean value (for criticality)
        
        Returns:
            (durability, reliability, transport_priority, boolean) tuple, all as strings
        """
        mappings = get_runtime_config().aggregator.qos_mappings

        dur_map = mappings.get("durability", {})
        rel_map = mappings.get("reliability", {})
        pri_map = mappings.get("transport_priority", {})

        converted_dur = dur_map.get(str(dur), str(dur)) if dur is not None else None
        converted_rel = rel_map.get(str(rel), str(rel)) if rel is not None else None
        converted_pri = pri_map.get(str(pri), str(pri)) if pri is not None else None

        bool_map = mappings.get("boolean", {})
        converted_bool = bool_map.get(str(boolean), boolean) if boolean is not None else None

        return converted_dur, converted_rel, converted_pri, converted_bool
