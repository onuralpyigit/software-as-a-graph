"""
Checks between a label artifact on disk and a training run.

Predict reads labels from disk and never imports Simulate, so a label file is the
whole contract between the two stages. These checks refuse the files that would
otherwise train without complaint: labels of another labeler, labels of another
topology (scenario ids such as ``A0`` overlap, so nothing fails on its own), and
labels too noisy to agree with themselves across seeds.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Mapping, Optional

from saag.core.graph_io import topology_fingerprint

logger = logging.getLogger(__name__)

FAULT_INJECTOR = "FaultInjector"
RUNTIME_LABELER = "MessageFlowSimulator/runtime"

#: `FaultInjector`'s scalar I*(v).
DEFAULT_LABEL = "impact_score"

#: Lowest worst-pair Spearman between seeds at which a label may be trained on.
#: A judgement, not a derived constant: it clears every FaultInjector label set in
#: the corpus (lowest 0.81, microservices) and refuses runtime labels whose ranking
#: does not reproduce (crash i_delivery measured -0.011 on one scenario).
MIN_LABEL_RETEST = 0.70


class LabelArtifactError(ValueError):
    """A label artifact that must not be trained on."""


def select_labels(raw: Mapping[str, Any], label: str = DEFAULT_LABEL) -> Dict[str, Dict[str, float]]:
    """Flatten one label of an artifact to ``{node_id: {dimension: value}}``.

    A runtime label fills only the composite dimension: it says nothing about
    reliability or maintainability, and an absent key is what marks a dimension
    unmeasured (see ``networkx_to_hetero_data``). A node whose label is ``None``
    was not measured, so it stays out rather than becoming a zero.
    """
    from saag.prediction.data_preparation import extract_simulation_dict

    # Artifacts older than schema 2.1 declare no labeler; only FaultInjector wrote them.
    labeler = raw.get("labeler", FAULT_INJECTOR)
    if labeler == FAULT_INJECTOR:
        if label != DEFAULT_LABEL:
            raise LabelArtifactError(
                f"{FAULT_INJECTOR} artifacts carry only {DEFAULT_LABEL!r}, not {label!r}.")
        return extract_simulation_dict(dict(raw))
    if labeler != RUNTIME_LABELER:
        raise LabelArtifactError(f"{labeler!r} is not a Predict-stage labeler.")

    available = raw.get("labeled_dimensions", [])
    if label == DEFAULT_LABEL:
        raise LabelArtifactError(
            f"A runtime-label artifact has no {DEFAULT_LABEL!r}: choose one with --label "
            f"(available: {', '.join(available)}).")
    labels = {
        str(node): {"composite": float(record[label]["mean"])}
        for node, record in raw.get("records", {}).items()
        if record.get(label) is not None
    }
    if not labels:
        raise LabelArtifactError(
            f"No node has a measured {label!r} (available: {', '.join(available)}).")
    return labels


def check_topology(raw: Mapping[str, Any], graph) -> None:
    """Refuse labels measured on a different topology than the one being trained on."""
    stored = raw.get("topology_fingerprint")
    if not stored:
        logger.warning("Label artifact has no topology fingerprint (schema < 2.2): "
                       "cannot verify it belongs to this topology.")
        return
    current = topology_fingerprint(graph)
    if stored != current:
        raise LabelArtifactError(
            f"Labels were measured on topology {stored} (graph_id "
            f"{raw.get('graph_id') or 'unknown'!r}), but this topology is {current}. "
            "Re-run the labeler on this topology.")


def label_retest(raw: Mapping[str, Any], label: str = DEFAULT_LABEL) -> Optional[float]:
    """The label's worst pairwise Spearman across seeds, or ``None`` if never measured."""
    block = raw.get("label_stability") or {}
    value = (block.get("test_retest_spearman") if label == DEFAULT_LABEL
             else (block.get(label) or {}).get("worst_spearman"))
    if value is None or math.isnan(value):
        return None
    return float(value)


def check_stability(
    raw: Mapping[str, Any],
    label: str = DEFAULT_LABEL,
    min_retest: float = MIN_LABEL_RETEST,
    allow_unstable: bool = False,
) -> Optional[float]:
    """Refuse a label that does not reproduce across seeds; return its retest.

    An unmeasured retest is not coerced to a pass or a fail. FaultInjector labels
    without one only warn: a single-seed run is a documented way to produce them,
    and their cascade is close to deterministic. Runtime labels are noisy enough
    that an unmeasured one is refused.
    """
    retest = label_retest(raw, label)
    if retest is None:
        message = (f"Label {label!r} has a stability that was never measured: "
                   "label with at least two seeds.")
        if raw.get("labeler", FAULT_INJECTOR) == FAULT_INJECTOR or allow_unstable:
            logger.warning(message)
            return None
        raise LabelArtifactError(f"{message} Pass --allow-unstable-labels to train anyway.")
    if retest < min_retest:
        message = (f"Label {label!r} agrees with itself across seeds at only rho = {retest:.3f}, "
                   f"below the minimum of {min_retest:.2f}.")
        if not allow_unstable:
            raise LabelArtifactError(
                f"{message} Use longer runs or more seeds, or pass --allow-unstable-labels.")
        logger.warning(message)
    return retest


def label_provenance(raw: Mapping[str, Any], label: str = DEFAULT_LABEL) -> Dict[str, Any]:
    """What a checkpoint was trained on, for writing next to it."""
    return {
        "labeler": raw.get("labeler", FAULT_INJECTOR),
        "label": label,
        "schema_version": raw.get("schema_version"),
        "graph_id": raw.get("graph_id"),
        "topology_fingerprint": raw.get("topology_fingerprint"),
        "seeds": raw.get("seeds_used", raw.get("seeds")),
        "retest": label_retest(raw, label),
        # FaultInjector states its settings here; a runtime sweep in these two.
        "labeler_config": raw.get("labeler_config"),
        "settings": raw.get("settings"),
        "fault": raw.get("fault"),
    }
