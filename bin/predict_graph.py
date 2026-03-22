#!/usr/bin/env python3
"""
bin/predict_graph.py — Run GNN inference on a new graph
======================================================
Loads a pre-trained GNN and predicts component and relationship criticality 
for a new system topology.

Usage
-----
  python bin/predict_graph.py --layer app --checkpoint output/gnn_checkpoints/
  python bin/predict_graph.py --layer app --checkpoint output/gnn_checkpoints/ --compare-rmav
  python bin/predict_graph.py --layer app --checkpoint output/gnn_checkpoints/ --show-edges
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# Ensure repo root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from src.cli.console import ConsoleDisplay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("predict_graph")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GNN inference — predict criticality on a new system graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--layer", default="app",
        help="Layer(s) to process (comma-separated: app,infra,mw,system)"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint directory from train_graph.py")

    neo4j = parser.add_argument_group("Neo4j (live pipeline)")
    neo4j.add_argument("--uri", default=None)
    neo4j.add_argument("--user", default=None)
    neo4j.add_argument("--password", default=None)

    inputs = parser.add_argument_group("Pre-computed inputs")
    inputs.add_argument("--structural", type=str, default=None)
    inputs.add_argument("--rmav", type=str, default=None)
    inputs.add_argument("--simulated", type=str, default=None)

    display_opt = parser.add_argument_group("Display options")
    display_opt.add_argument("--top-n", type=int, default=15, help="Number of nodes")
    display_opt.add_argument("--compare-rmav", action="store_true", help="Side-by-side vs RMAV")
    display_opt.add_argument("--show-edges", action="store_true", help="Display edge scores")
    display_opt.add_argument("--output", type=str, default=None, help="Save to JSON")

    return parser.parse_args()


from src.cli.dispatcher import dispatch_predict


def load_json(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        logger.error(f"File not found: {path}")
        sys.exit(1)
    with open(p) as f:
        return json.load(f)


def main() -> None:
    args = parse_args()
    display = ConsoleDisplay()
    display.print_header(f"GNN Prediction: {args.layer.upper()} Layer")

    from src.infrastructure import create_repository
    conn_kwargs = {k: v for k, v in [("uri", args.uri), ("user", args.user), 
                                     ("password", args.password)] if v}
    repo = create_repository(**conn_kwargs)

    try:
        result = dispatch_predict(repo, args)

        # ── Results ─────────────────────────────────────────────────────────────
        summary = result.summary()
        display.print_subheader(f"System Summary ({summary['total_components']} components)")
        
        for lvl in ["critical", "high", "medium", "low", "minimal"]:
            count = summary[lvl]
            bar = "█" * min(count * 2, 40)
            color = display.level_color(lvl)
            print(f"  {lvl.upper():<9} {count:>3}  {display.colored(bar, color)}")

        top_nodes = result.top_critical_nodes(args.top_n)
        source_label = "Ensemble" if result.ensemble_scores else "GNN"
        display.print_subheader(f"Top {args.top_n} Critical Components ({source_label})")
        
        header = f"  {'#':<4} {'Component':<30} {'Score':>7} {'Level':<10} {'R':>6} {'M':>6} {'A':>6} {'V':>6}"
        print(display.colored(header, display.Colors.WHITE, bold=True))
        print("  " + "-" * 78)
        for i, s in enumerate(top_nodes, 1):
            color = display.level_color(s.criticality_level)
            print(
                f"  {i:<4} {s.component[:29]:<30} "
                f"{display.colored(f'{s.composite_score:>7.4f}', color)} {display.colored(s.criticality_level[:10], color):<10} "
                f"{s.reliability_score:>6.3f} {s.maintainability_score:>6.3f} "
                f"{s.availability_score:>6.3f} {s.vulnerability_score:>6.3f}"
            )

        # Side-by-side comparison logic (could be moved to dispatcher but kept for now)
        if args.compare_rmav and getattr(args, 'rmav', None):
            rmav_dict = load_json(args.rmav)
            # Simplified comparison logic
            pass

        if args.show_edges and result.edge_scores:
            top_edges = result.top_critical_edges(args.top_n)
            display.print_subheader(f"Top {args.top_n} Critical Relationships")
            print(f"  {'#':<4} {'Source':<20} {'→':<3} {'Target':<20} {'Type':<15} {'Score':>7}")
            print("  " + "-" * 74)
            for i, e in enumerate(top_edges, 1):
                color = display.level_color(e.criticality_level)
                print(
                    f"  {i:<4} {e.source_node[:19]:<20} → {e.target_node[:19]:<20} "
                    f"{display.colored(e.edge_type[:14], display.Colors.CYAN):<15} "
                    f"{display.colored(f'{e.composite_score:>7.4f}', color)}"
                )

        if args.output:
            print(f"\n  {display.colored('Predictions exported:', display.Colors.GREEN)} {args.output}")

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        repo.close()

    print(f"\n  {display.colored('Done.', display.Colors.GREEN)}")


if __name__ == "__main__":
    main()
