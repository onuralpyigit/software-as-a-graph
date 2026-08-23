#!/usr/bin/env python3
"""
Prescription Optimization CLI

Generates prescriptive refactoring recommendations (logical topic splitting,
physical host anti-affinity container reallocations, and transport QoS contract hardening)
and evaluates resilience improvements in a closed-loop simulation on a mutated graph.
"""
import argparse
import sys
import json
from pathlib import Path

# Add project root to sys.path to support direct execution (python cli/prescribe_graph.py)
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saag import Client
from saag.analysis.antipattern_detector import CATALOG
from reproduce.detection_validation import DEFAULT_EXCLUDED_PATTERNS
from cli.common.arguments import add_neo4j_arguments, add_common_arguments, setup_logging
from cli.common.console import ConsoleDisplay

#: Verdicts listed before the output is truncated.
VERDICT_LIMIT = 20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prescriptive architecture optimization and closed-loop validation."
    )

    # Specific args
    parser.add_argument("--gnn-checkpoint", type=str, help="Path to trained GNN model checkpoint (optional)")
    parser.add_argument("--input", "-i", type=str, help="Path to system topology JSON file to run in-memory (optional)")

    # Per-edit acceptance filter
    parser.add_argument("--kappa", type=float, default=1.0,
                        help="Acceptance multiple: an edit is kept only when its mean impact reduction "
                             "exceeds kappa * sigma_seed at every propagation threshold (default 1.0).")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Simulation seeds for the noise estimate (default: PrescribeService default).")
    parser.add_argument("--thresholds", nargs="+", type=float, default=None,
                        help="Propagation thresholds the edit must clear at every value.")

    add_neo4j_arguments(parser)
    add_common_arguments(parser)
    return parser


def build_client(args, console: ConsoleDisplay) -> Client:
    if args.input:
        from saag.infrastructure.memory_repo import MemoryRepository
        console.print_step(f"Loading graph JSON in-memory: {args.input}...")
        repo = MemoryRepository()
        with open(args.input) as f:
            data = json.load(f)
        repo.save_graph(data, clear=True)
        repo.derive_dependencies()
        return Client(repo=repo)

    console.print_step(f"Connecting to Neo4j at {args.uri}...")
    return Client(neo4j_uri=args.uri, user=args.user, password=args.password)


def print_candidate_policy(console: ConsoleDisplay, res) -> None:
    """Everything the rules proposed, before verification."""
    console.print_header("Candidate Optimization Policy Delta(G)")
    candidate = res.candidate_policy
    if candidate is None or candidate.is_empty():
        print("No candidate edits compiled.")
        return

    print(f"Logical Topic Splits       : {len(candidate.topic_splits)}")
    print(f"Physical Host Reallocations: {len(candidate.node_reallocations)}")
    print(f"Transport QoS Upgrades     : {len(candidate.qos_upgrades)}")


def print_verdicts(console: ConsoleDisplay, res, kappa: float) -> None:
    """Per-edit filter outcomes — what was kept, what was declined and why."""
    if not res.edit_verdicts:
        return

    console.print_header("Per-Edit Acceptance Filter")
    print(f"{res.n_accepted}/{len(res.edit_verdicts)} candidate edits accepted (kappa={kappa}).")
    print()

    for verdict in res.edit_verdicts[:VERDICT_LIMIT]:
        mark = (
            console.colored("[OK]", console.Colors.GREEN) if verdict.accepted
            else console.colored("[--]", console.Colors.RED)
        )
        # Margin is the decision quantity (delta minus the noise bar); delta is
        # the interpretable one. Showing only delta makes the reason line, which
        # names the threshold with the smallest *margin*, look inconsistent.
        print(f"  {mark} {verdict.kind:<19} {verdict.target:<20} "
              f"worst delta={verdict.worst_delta:+.6f}  margin={verdict.worst_margin:+.6f}")
        if verdict.reason:
            print(f"       {verdict.reason}")

    remaining = len(res.edit_verdicts) - VERDICT_LIMIT
    if remaining > 0:
        print(f"  ... and {remaining} more")


def print_applied_policy(console: ConsoleDisplay, res) -> None:
    """The accepted subset, exactly as it was applied to the graph."""
    console.print_header("Applied Policy")
    if not res.applied_changes:
        print("Nothing applied — the graph is unchanged.")
        return
    for change in res.applied_changes:
        print(f"  - {change}")


def print_resilience_summary(console: ConsoleDisplay, res, kappa: float) -> None:
    console.print_header("Closed-Loop Resilience Summary")
    print(f"Baseline System Risk Index (SRI): {res.original_sri:.4f}")
    print(f"Mutated System Risk Index (SRI) : {res.mutated_sri:.4f}")

    mean_reduction = res.mean_cascade_impact_reduction
    reduction_str = f"{mean_reduction * 100:+.2f}%" if mean_reduction is not None else "n/a"
    print(f"Mean cascade-impact reduction   : {reduction_str} "
          f"(over {len(res.remediated_component_impact_deltas)} remediated components)")

    improvement = res.sri_improvement
    if improvement > 0:
        console.print_success(f"Resilience Improved by: {improvement:.4f}")
    elif improvement < 0:
        console.print_error(f"Resilience Degraded by: {abs(improvement):.4f}")
    else:
        print("No change in System Risk Index.")

    # The accept/reject gate only means something once something was applied.
    if not res.edit_verdicts:
        console.print_warning(
            "No candidate edits were compiled — no CRITICAL/HIGH components or smells found."
        )
    elif res.n_accepted == 0:
        console.print_warning(
            f"No candidate edit cleared the acceptance filter (kappa={kappa}); the graph is "
            "unchanged. That is the filter working, not a failure."
        )
    elif res.accepted:
        console.print_success("Policy Status: ACCEPTED")
    else:
        console.print_error("Policy Status: REJECTED - the applied subset did not reduce mutated risk")


def main():
    args = build_parser().parse_args()
    setup_logging(args)
    console = ConsoleDisplay()

    console.print_header("Prescriptive Architecture Optimization (Stage 6)")

    try:
        client = build_client(args, console)

        console.print_step(f"Running baseline analysis on layer: {args.layer}...")
        analysis = client.analyze(layer=args.layer)

        # Predict (RM + anti-patterns) so the rules compiler sees the same
        # CRITICAL/HIGH risk set Stage 6 is documented to consume -- without
        # a prediction_result, compile_policy() sees an empty critical/spof/
        # god set and only topic splitting can ever fire. DEEP_PIPELINE is
        # excluded: it does not terminate in practical time at these scales.
        console.print_step("Scoring criticality and detecting anti-patterns...")
        prediction = client.predict(
            analysis,
            gnn_checkpoint=args.gnn_checkpoint,
            active_patterns=[pid for pid in CATALOG if pid not in DEFAULT_EXCLUDED_PATTERNS],
        )

        console.print_step("Generating refactoring recommendations and validating in closed-loop...")
        filter_kwargs = {"kappa": args.kappa}
        if args.seeds is not None:
            filter_kwargs["seeds"] = args.seeds
        if args.thresholds is not None:
            filter_kwargs["thresholds"] = args.thresholds

        res = client.prescribe(
            analysis_result=analysis,
            prediction_result=prediction,
            layer=args.layer,
            gnn_checkpoint=args.gnn_checkpoint,
            **filter_kwargs,
        )

        console.print_success("Prescriptive Remediation Generated!")

        print_candidate_policy(console, res)
        print_verdicts(console, res, args.kappa)
        print_applied_policy(console, res)
        print_resilience_summary(console, res, args.kappa)

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w") as f:
                json.dump(res.to_dict(), f, indent=2, default=str)
            console.print_success(f"Prescription optimization result saved to {args.output}")

    except Exception as e:
        console.print_error(f"Prescription generation failed: {e}")
        if getattr(args, 'verbose', False):
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
