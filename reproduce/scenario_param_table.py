#!/usr/bin/env python3
"""
reproduce/scenario_param_table.py — the generator parameters, read from the configs
==================================================================================

Writes ``results/scenario_parameters.{tex,md}``: one row per evaluation scenario,
giving the inputs a third party needs to regenerate the corpus.

The manuscript previously said only that the scenarios were "generated using
statistical topology generators", which is not a specification. Every parameter
is committed in ``data/scenarios/scenario_*.yaml``, so the table is *read* from
those files rather than transcribed — a hand-copied table drifts the first time a
config changes, and silently.

Columns are chosen to be the ones that determine the topology's shape rather than
its size alone: the seed, the entity counts, the mean publish/subscribe fan-out
per application, the mean applications per host, and the modal QoS profile with
the share of topics that carry it.

The modal QoS column is computed from the committed topology
(``data/scenarios/<name>.json``'s ``topics``), not from the config's
``qos_stats`` block. Domain-driven topic QoS (``get_qos_for_topic`` in
``tools/generation/datasets.py``) takes precedence over the pool sampled from
``qos_stats`` whenever a scenario declares a ``domain`` -- true of every
scenario in this corpus -- so ``qos_stats.category_counts`` does not describe
what was actually generated for domains whose ``QOS_MAPPINGS`` entry is a
single ``"default"`` triple (``hub-and-spoke``, ``microservices``,
``enterprise``; ``av`` collapses four of five patterns to one triple). An
earlier version of this script read ``qos_stats`` directly and reported that
un-realized target distribution.

Usage
-----
    PYTHONPATH=. python reproduce/scenario_param_table.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SCENARIOS_DIR = Path("data/scenarios")
RESULTS_DIR = Path("results")

#: The seven evaluation scenarios, in the order the manuscript's corpus table
#: uses. Keyed by the dataset name so the manifest can be cross-checked.
_EVAL_ORDER = [
    ("av_system", "Autonomous Vehicle (AV)"),
    ("enterprise_system", "Enterprise Pub-Sub"),
    ("financial_trading_system", "Financial Trading"),
    ("healthcare_system", "Healthcare Integration"),
    ("hub_and_spoke_system", "Hub-and-Spoke"),
    ("iot_smart_city_system", "IoT Smart City"),
    ("microservices_system", "Microservices Mesh"),
]


def _mean(block: Optional[Dict[str, Any]]) -> Optional[float]:
    return None if not isinstance(block, dict) else block.get("mean")


def _modal_qos_from_topics(topics: List[Dict[str, Any]]) -> str:
    """Modal (reliability, durability, priority) triple and the topic share,
    computed from the committed topology rather than the config's declared
    (and not always realized) ``qos_stats`` distribution -- see module
    docstring."""
    import collections

    if not topics:
        return "---"
    parts, pcts = [], []
    for field in ("reliability", "durability", "transport_priority"):
        counts = collections.Counter(
            str((t.get("qos") or {}).get(field, "")).upper() for t in topics
        )
        counts.pop("", None)
        if not counts:
            continue
        mode, mode_n = counts.most_common(1)[0]
        parts.append(mode.replace("_", r"\_"))
        pcts.append(100.0 * mode_n / len(topics))
    if not parts:
        return "---"
    share = f" ({min(pcts):.0f}--{max(pcts):.0f}\\%)" if pcts else ""
    return "/".join(parts) + share


def collect(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    import yaml

    rows = []
    datasets = manifest.get("datasets", {})
    for name, label in _EVAL_ORDER:
        entry = datasets.get(name)
        if not entry:
            continue
        cfg_path = SCENARIOS_DIR / entry["config"]
        if not cfg_path.exists():
            continue
        g = (yaml.safe_load(cfg_path.read_text()) or {}).get("graph", {})
        counts = g.get("counts", {})
        app = g.get("application_stats", {}) or {}
        node = (g.get("node_stats", {}) or {}).get("applications_per_node")

        data_path = SCENARIOS_DIR / f"{name}.json"
        topics = json.loads(data_path.read_text()).get("topics", []) if data_path.exists() else []

        rows.append({
            "dataset": name,
            "label": label,
            "config": entry["config"],
            "seed": g.get("seed", entry.get("seed")),
            "counts": counts,
            "mean_publish": _mean(app.get("direct_publish_count")),
            "mean_subscribe": _mean(app.get("direct_subscribe_count")),
            "mean_apps_per_host": _mean(node),
            "modal_qos": _modal_qos_from_topics(topics),
            "sha256": entry.get("sha256", "")[:12],
        })
    return rows


def _fmt(v, spec="{:.1f}"):
    return "---" if v is None else spec.format(v)


def render_tex(rows: List[Dict[str, Any]], path: Path) -> None:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Generative parameters of the seven synthetic evaluation scenarios. "
        r"Counts, seed and fan-out figures are read directly from the committed "
        r"configurations. The modal QoS column gives the most common "
        r"reliability/durability/priority value and the range of topic shares carrying "
        r"them, computed from the committed topology rather than the config's declared "
        r"QoS targets, which domain-driven assignment does not always realize "
        r"(Section~\ref{sec:6.1}).}",
        r"\label{tab:genparams}",
        r"\begin{tabular}{llrrrrl}",
        r"\toprule",
        r"\textbf{Scenario} & \textbf{Config} & \textbf{Seed} & \textbf{Counts} & "
        r"\textbf{Pub} & \textbf{Sub} & \textbf{Modal QoS (R/D/P)} \\",
        r"\midrule",
    ]
    for r in rows:
        c = r["counts"]
        counts = "/".join(str(c.get(k, "--")) for k in
                          ("applications", "topics", "brokers", "nodes", "libraries"))
        cfg = r["config"].replace("_", r"\_").replace(".yaml", "")
        lines.append(
            rf"\textbf{{{r['label']}}} & \texttt{{{cfg}}} & {r['seed']} & {counts} & "
            rf"{_fmt(r['mean_publish'])} & {_fmt(r['mean_subscribe'])} & {r['modal_qos']} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"  Saved LaTeX: {path}")


def render_md(rows: List[Dict[str, Any]], path: Path) -> None:
    out = [
        "| Scenario | Config | Seed | Apps/Topics/Brokers/Hosts/Libs | Mean pub | "
        "Mean sub | Mean apps/host | Modal QoS | sha256 |",
        "|---|---|---:|---|---:|---:|---:|---|---|",
    ]
    for r in rows:
        c = r["counts"]
        counts = "/".join(str(c.get(k, "--")) for k in
                          ("applications", "topics", "brokers", "nodes", "libraries"))
        out.append(
            f"| {r['label']} | `{r['config']}` | {r['seed']} | {counts} | "
            f"{_fmt(r['mean_publish'])} | {_fmt(r['mean_subscribe'])} | "
            f"{_fmt(r['mean_apps_per_host'])} | {r['modal_qos'].replace(chr(92), '')} | "
            f"`{r['sha256']}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out) + "\n")
    print(f"  Saved Markdown: {path}")


def parse_args():
    p = argparse.ArgumentParser(description="Generator parameter table from the configs")
    p.add_argument("--manifest", type=Path, default=SCENARIOS_DIR / "MANIFEST.json")
    p.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    p.add_argument(
        "--latex-copy", type=Path,
        default=Path("docs/research/jss/latex/sections/tab_genparams.tex"),
        help="Second copy written inside the LaTeX tree, so the manuscript can "
             "\\input it with a relative path that survives being zipped for "
             "submission. Pass an empty path to skip.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = collect(manifest)
    if not rows:
        raise SystemExit("no scenarios resolved; check the manifest and config paths")

    print(f"Generator parameters for {len(rows)} evaluation scenarios")
    for r in rows:
        print(f"  {r['label']:26s} seed={r['seed']:<6} {r['modal_qos']}")

    render_tex(rows, args.output_dir / "scenario_parameters.tex")
    render_md(rows, args.output_dir / "scenario_parameters.md")
    if str(args.latex_copy):
        render_tex(rows, args.latex_copy)


if __name__ == "__main__":
    main()
