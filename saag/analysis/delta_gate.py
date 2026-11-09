"""Delta-aware CI gate: block only what a change introduces.

Absolute gating blocks every scenario in the corpus on pre-existing debt
(AuSE §7.2). This gate compares the candidate's anti-pattern findings with the
merge-base's and blocks only on a new CRITICAL/HIGH finding -- one whose
(pattern, entity) the merge-base did not have -- unless a waiver covers it.

A finding is keyed on (pattern, entity type, entity id). Severity is fixed per
pattern, so a "worsening severity" is a new key: the entity gained a
higher-severity pattern.

Fence margin. Fence-triggered patterns record ``fence_ratio`` (metric over the
threshold it crossed) in their evidence. Global metrics such as edge
betweenness move on edges a change never touched, so a one-edge change pushes
near-fence edges across: on the corpus a plain key diff still blocked 40-100% of
single-element edits, all through BOTTLENECK_EDGE, and most of those edges had
sat within 10% of the fence. A new finding with ``fence_ratio <= 1 + margin`` is
therefore a warning, not a block. Findings without a ratio (articulation
points, bridges, cycles) block whenever new.

Warn-only patterns. Even with the margin, BOTTLENECK_EDGE still blocked most
single-element edits (it flags 10-15% of all edges, so a new edge often lands
above the fence, and far edges still drift across it). In delta mode a new
BOTTLENECK_EDGE is a warning whatever its ratio; absolute mode is unchanged.

Exit codes follow the absolute gate: 2 block, 1 warning, 0 clean.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from saag.analysis.antipattern_detector import CATALOG

BLOCKING_SEVERITIES = frozenset({"CRITICAL", "HIGH"})
#: Patterns whose new findings warn but never block in delta mode (see module docstring).
WARN_ONLY_PATTERNS = frozenset({"BOTTLENECK_EDGE"})
_PATTERN_BY_NAME = {spec.name: pid for pid, spec in CATALOG.items()}

Key = Tuple[str, str, str]


@dataclass(frozen=True)
class Waiver:
    """An accepted risk: who accepted it, why, and until when."""
    pattern: str
    entity: str
    expires: date
    reason: str
    approved_by: str

    def covers(self, key: Key) -> bool:
        return (self.pattern, self.entity) == (key[0], key[2])


def load_waivers(path: Path) -> List[Waiver]:
    """Read a waiver register (YAML or JSON): a list, or ``{waivers: [...]}``.

    Every entry needs ``pattern``, ``entity``, ``expires`` (YYYY-MM-DD),
    ``reason`` and ``approved_by``; a register that lacks one is rejected rather
    than read as a waiver of everything or nothing.
    """
    import yaml  # JSON is a subset of YAML

    data = yaml.safe_load(Path(path).read_text()) or []
    entries = data.get("waivers", []) if isinstance(data, dict) else data
    out = []
    for i, e in enumerate(entries):
        missing = [k for k in ("pattern", "entity", "expires", "reason", "approved_by") if not e.get(k)]
        if missing:
            raise ValueError(f"waiver #{i + 1} in {path} lacks {', '.join(missing)}")
        if e["pattern"] not in CATALOG:
            raise ValueError(f"waiver #{i + 1} in {path}: unknown pattern {e['pattern']!r}")
        expires = e["expires"] if isinstance(e["expires"], date) else date.fromisoformat(str(e["expires"]))
        out.append(Waiver(e["pattern"], str(e["entity"]), expires, str(e["reason"]), str(e["approved_by"])))
    return out


def finding_key(finding: Mapping[str, Any]) -> Key:
    return (_PATTERN_BY_NAME[finding["name"]], finding["entity_type"], finding["entity_id"])


@dataclass
class DeltaGateResult:
    blocking: List[Dict[str, Any]] = field(default_factory=list)
    near_fence: List[Dict[str, Any]] = field(default_factory=list)
    new_advisory: List[Dict[str, Any]] = field(default_factory=list)
    waived: List[Dict[str, Any]] = field(default_factory=list)
    carried: int = 0
    resolved: List[Dict[str, Any]] = field(default_factory=list)
    expired_waivers: List[Waiver] = field(default_factory=list)

    @property
    def exit_code(self) -> int:
        if self.blocking:
            return 2
        return 1 if (self.near_fence or self.new_advisory) else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exit_code": self.exit_code,
            "blocking": self.blocking,
            "near_fence": self.near_fence,
            "new_advisory": self.new_advisory,
            "waived": self.waived,
            "carried": self.carried,
            "resolved": self.resolved,
            "expired_waivers": [
                {"pattern": w.pattern, "entity": w.entity, "expires": w.expires.isoformat(),
                 "reason": w.reason, "approved_by": w.approved_by}
                for w in self.expired_waivers
            ],
        }


def delta_gate(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
    waivers: Sequence[Waiver] = (),
    margin: float = 0.10,
    today: Optional[date] = None,
) -> DeltaGateResult:
    """Classify the candidate's findings against the merge-base's."""
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin!r}")
    today = today or date.today()
    active = [w for w in waivers if w.expires >= today]
    result = DeltaGateResult(expired_waivers=[w for w in waivers if w.expires < today])

    base_keys = {finding_key(f) for f in baseline}
    cand_keys = set()
    for f in candidate:
        key = finding_key(f)
        cand_keys.add(key)
        if key in base_keys:
            result.carried += 1
        elif any(w.covers(key) for w in active):
            result.waived.append(dict(f))
        elif f["severity"] not in BLOCKING_SEVERITIES or key[0] in WARN_ONLY_PATTERNS:
            result.new_advisory.append(dict(f))
        else:
            ratio = (f.get("evidence") or {}).get("fence_ratio")
            if ratio is not None and ratio <= 1.0 + margin:
                result.near_fence.append(dict(f))
            else:
                result.blocking.append(dict(f))
    result.resolved = [dict(f) for f in baseline if finding_key(f) not in cand_keys]
    return result
