"""
Visualization Colour Palette

The single source of truth for every colour the dashboard renders — chart
series, Cytoscape nodes, D3 matrix cells, badges and hierarchy dots all read
from here so they cannot drift apart.

Criticality ramp is accessibility-tested; RMAV colours carry AHP dimension
semantics (R=purple structural authority, M=teal maintainability,
A=coral operational risk, S=pink exposure).
"""
from typing import Dict

# ── Criticality levels ────────────────────────────────────────────────────────
CRITICALITY_COLORS: Dict[str, str] = {
    "CRITICAL": "#A32D2D",   # red-800
    "HIGH":     "#854F0B",   # amber-800
    "MEDIUM":   "#185FA5",   # blue-800
    "LOW":      "#3B6D11",   # green-800
    "MINIMAL":  "#5F5E5A",   # gray-800
}

# Muted background / saturated foreground pairs for criticality badges.
CRITICALITY_BADGE_COLORS: Dict[str, tuple] = {
    "CRITICAL": ("#FCEBEB", "#791F1F"),
    "HIGH":     ("#FAEEDA", "#633806"),
    "MEDIUM":   ("#E6F1FB", "#0C447C"),
    "LOW":      ("#EAF3DE", "#27500A"),
    "MINIMAL":  ("#F1EFE8", "#444441"),
}

DEFAULT_COLOR = CRITICALITY_COLORS["MINIMAL"]

# ── RMAV quality dimensions ───────────────────────────────────────────────────
RMAS_COLORS: Dict[str, str] = {
    "reliability":     "#534AB7",   # purple
    "maintainability": "#0F6E56",   # teal
    "availability":    "#993C1D",   # coral
    "security":        "#993556",   # pink
}

# AHP dimension weights — bar segments are scaled by these so a stacked
# RMAV bar sums to Q(v).
AHP_WEIGHTS: Dict[str, float] = {
    "availability":    0.43,
    "reliability":     0.24,
    "maintainability": 0.17,
    "security":        0.16,
}

# ── Component types ───────────────────────────────────────────────────────────
TYPE_COLORS: Dict[str, str] = {
    "Application": "#534AB7",
    "Broker":      "#0F6E56",
    "Node":        "#185FA5",
    "Topic":       "#993C1D",
    "Library":     "#3B6D11",
}

# ── MIL-STD-498 hierarchy levels: (background, accent) ─────────────────────────
HIERARCHY_COLORS: Dict[str, tuple] = {
    "CSS":  ("#EEEDFE", "#534AB7"),
    "CSCI": ("#E1F5EE", "#0F6E56"),
    "CSC":  ("#E6F1FB", "#185FA5"),
    "CSU":  ("#F1EFE8", "#5F5E5A"),
}

# ── Accents used outside the semantic ramps ───────────────────────────────────
BRAND_PURPLE = "#534AB7"     # primary accent: matrix cells, QoS-enriched bars
NEUTRAL_GREY = "#B4B2A9"     # baseline / comparison series

# Qualitative fallback ramp for charts with arbitrary category counts.
CATEGORICAL_PALETTE = [
    "#534AB7", "#0F6E56", "#993C1D", "#993556",
    "#185FA5", "#3B6D11", "#854F0B",
]


def criticality_badge_css() -> str:
    """Render the `.badge-<level>` rules from CRITICALITY_BADGE_COLORS."""
    return "\n".join(
        f"    .badge-{level.lower():<9} {{ background: {bg}; color: {fg}; }}"
        for level, (bg, fg) in CRITICALITY_BADGE_COLORS.items()
    )
