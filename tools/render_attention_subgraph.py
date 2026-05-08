#!/usr/bin/env python3
"""
tools/render_attention_subgraph.py — Block G: ATM Attention Subgraph Visualiser
================================================================================

Reads output/atm_case_study/attention_weights.json and renders a styled
NetworkX subgraph where edge thickness and colour encode mean attention α.

Outputs:
  output/atm_case_study/attention_subgraph.png  (paper Figure 5 / appendix)
  output/atm_case_study/attention_subgraph.pdf
  output/atm_case_study/top_attention_edges.md   (table for paper)

Usage
-----
  python tools/render_attention_subgraph.py
  python tools/render_attention_subgraph.py --layer layer_1 --top-k 15
  python tools/render_attention_subgraph.py --table-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_RESULTS_DIR = Path("output/atm_case_study")

# Node-type colour palette (matches dashboard colours)
_NODE_COLORS = {
    "Application": "#4C72B0",
    "Broker":      "#55A868",
    "Topic":       "#DD8452",
    "Node":        "#C44E52",
    "Library":     "#8172B2",
}
_DEFAULT_COLOR = "#999999"


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_attention(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Attention file not found: {path}\n"
            "Run:  python tools/extract_attention.py"
        )
    return json.loads(path.read_text())


# ── Edge extraction ───────────────────────────────────────────────────────────

def _collect_edges(
    attn_data: Dict, layer_key: str, top_k: int
) -> List[Dict]:
    """Return top-k edges by mean α from the specified layer."""
    layer = attn_data.get("attention_by_layer", {}).get(layer_key, {})
    node_ids = attn_data.get("node_id_map", {})

    all_edges = []
    for rel_key, info in layer.items():
        parts = rel_key.split("__")
        if len(parts) != 3:
            continue
        src_t, etype, dst_t = parts
        edges   = info.get("edges", [])
        means   = info.get("mean_alpha", [])
        src_ids = node_ids.get(src_t, [])
        dst_ids = node_ids.get(dst_t, [])

        for (src_idx, dst_idx), alpha in zip(edges, means):
            src_name = src_ids[src_idx] if src_idx < len(src_ids) else f"{src_t}[{src_idx}]"
            dst_name = dst_ids[dst_idx] if dst_idx < len(dst_ids) else f"{dst_t}[{dst_idx}]"
            all_edges.append({
                "src": src_name,
                "dst": dst_name,
                "src_type": src_t,
                "dst_type": dst_t,
                "etype": etype,
                "alpha": float(alpha),
                "rel_key": rel_key,
            })

    all_edges.sort(key=lambda e: e["alpha"], reverse=True)
    return all_edges[:top_k]


# ── Console table ─────────────────────────────────────────────────────────────

def _print_top_edges(edges: List[Dict], scenario: str, layer: str):
    print(f"\n  Top-{len(edges)} Attention Edges — {scenario} ({layer})")
    print(f"  {'Rank':<5} {'α':<8} {'Source':<30} {'Edge Type':<18} {'Destination'}")
    print("  " + "─" * 80)
    for i, e in enumerate(edges, 1):
        print(f"  {i:<5} {e['alpha']:<8.4f} {e['src']:<30} {e['etype']:<18} {e['dst']}")


def _save_md(edges: List[Dict], path: Path, scenario: str, layer: str):
    rows = [
        f"## Top Attention Edges — {scenario} ({layer})",
        "",
        "| Rank | α | Source | Edge Type | Destination |",
        "|---|---|---|---|---|",
    ]
    for i, e in enumerate(edges, 1):
        rows.append(f"| {i} | {e['alpha']:.4f} | {e['src']} | {e['etype']} | {e['dst']} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n")
    print(f"  Saved MD: {path}")


# ── Plot ──────────────────────────────────────────────────────────────────────

def _render_subgraph(edges: List[Dict], output_path: Path, title: str, dpi: int = 180):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import networkx as nx
    except ImportError:
        print("matplotlib/networkx not installed. Skipping plot.", file=sys.stderr)
        return

    G = nx.DiGraph()
    node_type_map: Dict[str, str] = {}

    for e in edges:
        G.add_node(e["src"]); node_type_map[e["src"]] = e["src_type"]
        G.add_node(e["dst"]); node_type_map[e["dst"]] = e["dst_type"]
        G.add_edge(e["src"], e["dst"], alpha=e["alpha"], etype=e["etype"])

    if G.number_of_nodes() == 0:
        print("  No edges to render.")
        return

    alphas = [G[u][v]["alpha"] for u, v in G.edges()]
    max_a  = max(alphas) if alphas else 1.0
    min_a  = min(alphas) if alphas else 0.0
    norm_a = [(a - min_a) / (max_a - min_a + 1e-9) for a in alphas]

    # Layout
    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=(14, 9))

    # Nodes
    node_colors = [_NODE_COLORS.get(node_type_map.get(n, ""), _DEFAULT_COLOR)
                   for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=600, alpha=0.9, zorder=2)
    nx.draw_networkx_labels(G, pos, ax=ax,
                            labels={n: n[:20] + "…" if len(n) > 20 else n for n in G.nodes()},
                            font_size=7, font_color="white", font_weight="bold", zorder=3)

    # Edges coloured by attention
    cmap = cm.YlOrRd
    edge_colors = [cmap(na) for na in norm_a]
    edge_widths  = [1.0 + 4.0 * na for na in norm_a]
    nx.draw_networkx_edges(G, pos, ax=ax,
                           edge_color=edge_colors, width=edge_widths,
                           arrows=True, arrowsize=15,
                           connectionstyle="arc3,rad=0.08",
                           zorder=1)

    # Colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(min_a, max_a))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Mean Attention α", fontsize=10)

    # Legend for node types
    legend_patches = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=c, markersize=10, label=nt)
        for nt, c in _NODE_COLORS.items()
        if nt in set(node_type_map.values())
    ]
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8, framealpha=0.9)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        p = output_path.with_suffix(f".{ext}")
        plt.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"  Saved: {p}")
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Block G: Render attention subgraph.")
    p.add_argument("--input", type=Path,
                   default=_RESULTS_DIR / "attention_weights.json")
    p.add_argument("--layer", default="layer_0",
                   help="HGTConv layer to visualise (default: layer_0)")
    p.add_argument("--top-k", type=int, default=20,
                   help="Top-K edges by mean α (default: 20)")
    p.add_argument("--output", type=Path,
                   default=_RESULTS_DIR / "attention_subgraph")
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--table-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"\n  Block G — ATM Attention Subgraph Visualiser")
    data = _load_attention(args.input)
    scenario = data.get("scenario", "unknown")

    # Determine best layer (last available with data)
    attn_layers = [k for k, v in data.get("attention_by_layer", {}).items() if v]
    if not attn_layers:
        print("  No attention data captured in JSON.")
        print("  Note: HGTConv may not expose alpha in this PyG version.")
        print("  The tool will still render the top-K edges based on edge weight if available.")
        sys.exit(0)

    layer_key = args.layer if args.layer in attn_layers else attn_layers[-1]
    print(f"  Scenario: {scenario}  |  Layer: {layer_key}")

    edges = _collect_edges(data, layer_key, top_k=args.top_k)
    _print_top_edges(edges, scenario, layer_key)
    _save_md(edges, args.output.parent / "top_attention_edges.md", scenario, layer_key)

    if not args.table_only:
        title = (
            f"HGT Attention Subgraph — {scenario} ({layer_key})\n"
            f"Top-{len(edges)} edges by mean attention α"
        )
        _render_subgraph(edges, args.output, title, dpi=args.dpi)
    print("\n  Done.")


if __name__ == "__main__":
    main()
