#!/usr/bin/env python3
"""
Example script that walks through Step 4, Diagnose (docs/diagnosis.md), on the
15-component mini-ATM topology and builds an interactive HTML page.

For every component it shows how the deterministic quality model turns
structural metrics into fault tolerance, availability and maintainability,
rolls them up into reliability and Q(v), assigns a level from the box-plot
fences, and attaches an explanation. No trained model and no simulation are
involved.
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

# Add project root to sys.path to support direct execution
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from saag import Client
from saag.analysis.analyzer import COUPLING_PATH_DELTA, QualityAnalyzer
from saag.infrastructure.memory_repo import MemoryRepository

from walkthrough_page import write_page

SCRIPT_DIR = Path(__file__).parent
TOPOLOGY = SCRIPT_DIR / "mini_atm.json"
TEMPLATE = SCRIPT_DIR / "diagnose_walkthrough.template.html"
DIMENSIONS = ("fault_tolerance", "availability", "reliability", "maintainability", "overall")


def r4(x):
    return None if x is None else round(float(x), 4)


def terms_of(m, norm, w) -> dict:
    """Each sub-score as (label, normalised input, weight) terms.

    QualityAnalyzer._compute_rm returns only the totals, so the terms are
    retraced here from the same normalisation table and weights, and main()
    checks that they sum to the engine's own FT, A and M.
    """
    def n(value, key):
        entry = norm.get(key)
        if isinstance(entry, dict):
            return entry.get(m.id, 0.0)
        return value / entry if entry else 0.0

    rpr, in_n, qw = n(m.reverse_pagerank, "reverse_pagerank"), n(m.in_degree_raw, "in_degree"), n(m.weight, "weight")
    if m.type == "Topic":
        cdpot = m.fan_out_criticality * (1.0 - min(n(m.dependency_weight_in, "w_in"), 1.0))
        ft = [("Fan-out criticality", m.fan_out_criticality, w.ft_topic_foc),
              ("Cascade depth potential", cdpot, w.ft_topic_cdpot)]
    else:
        base = ((rpr + in_n) / 2.0) * (1.0 - min(m.out_degree_raw / max(float(m.in_degree_raw), 1e-9), 1.0))
        ft = [("Reverse PageRank", rpr, w.ft_reverse_pagerank), ("Direct dependents", in_n, w.ft_in_degree),
              ("Cascade depth potential", min(base * (1.0 + m.mpci), 1.0), w.ft_cdpot)]
    availability = [("Cut-vertex score", m.ap_c_directed, w.a_ap_c_directed),
                    ("QoS-weighted cut vertex", m.ap_c_directed * qw, w.a_qspof),
                    ("Bridge ratio", m.bridge_ratio, w.a_bridge_ratio),
                    ("Connectivity loss", m.cdi, w.a_cdi), ("QoS weight", qw, w.a_qos_weight)]
    instability = m.out_degree_raw / (float(m.in_degree_raw + m.out_degree_raw) + 1e-9)
    coupling = min(1.0, (1.0 - abs(2.0 * instability - 1.0)) * (1.0 + COUPLING_PATH_DELTA * m.path_complexity))
    maintainability = [("Betweenness", n(m.betweenness, "betweenness"), w.m_betweenness),
                       ("Outgoing dependency weight", n(m.dependency_weight_out, "w_out"), w.m_w_out),
                       ("Code-quality penalty", m.code_quality_penalty, w.m_code_quality_penalty),
                       ("Coupling risk", coupling, w.m_coupling_risk),
                       ("Low clustering", 1.0 - m.clustering_coefficient, w.m_clustering)]
    return {k: [[label, r4(v), r4(weight)] for label, v, weight in rows]
            for k, rows in (("fault_tolerance", ft), ("availability", availability), ("maintainability", maintainability))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "diagnose_walkthrough.html",
                        help="Where to write the HTML walkthrough")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    logging.disable(logging.WARNING)

    repo = MemoryRepository()
    repo.save_graph(json.loads(TOPOLOGY.read_text()), clear=True)
    repo.derive_dependencies()
    client = Client(repo=repo)
    analysis = client.analyze(layer="system")

    # Record the normalisation table the analyzer scores with, without changing what it does.
    seen, original = {}, QualityAnalyzer._compute_rm

    def recording(self, m, norm):
        seen["norm"], seen["weights"] = norm, self.weights
        return original(self, m, norm)

    QualityAnalyzer._compute_rm = recording
    try:
        quality = client.diagnose(analysis).raw
    finally:
        QualityAnalyzer._compute_rm = original
    weights, norm = seen["weights"], seen["norm"]

    components = {}
    for c in quality.components:
        terms = terms_of(c.structural, norm, weights)
        for dim, rows in terms.items():
            total = sum(v * w for _, v, w in rows)
            assert abs(total - getattr(c.scores, dim)) < 1e-3, (c.id, dim, total, getattr(c.scores, dim))
        components[c.id] = {"type": c.type, "terms": terms, "level": c.levels.overall.value,
                            "levels": {d: getattr(c.levels, d).value for d in DIMENSIONS},
                            **{d: r4(getattr(c.scores, d)) for d in DIMENSIONS}}

    # The fences behind the five levels, recomputed with the analyzer's own classifier and checked.
    classifier = QualityAnalyzer().classifier
    stats = classifier.compute_stats([c.scores.overall for c in quality.components])
    for c in quality.components:
        assert classifier.classify_score(c.scores.overall, stats) == c.levels.overall, c.id
    fences = {"k": classifier.k_factor, **{k: r4(getattr(stats, k)) for k in ("q1", "median", "q3", "iqr", "upper_fence")}}

    report = quality.explanation.to_dict()
    declared = QualityAnalyzer().weights
    data = {
        "components": components,
        "weights": {k: r4(getattr(weights, k)) for k in ("r_alpha", "q_reliability", "q_maintainability")},
        "declared": {k: r4(getattr(declared, k)) for k in ("q_reliability", "q_maintainability")},
        "qos_profile": analysis.raw.structural.qos_profile,
        "fences": fences,
        "distribution": quality.classification_summary.component_distribution,
        "explain": {e["component_id"]: {k: e[k] for k in ("pattern", "level", "one_line", "top_risk", "dimensions",
                                                           "priority_action", "anti_patterns", "stakeholders")}
                    for e in report["component_explanations"]},
        "problems": [{k: p.to_dict()[k] for k in ("entity_id", "severity", "name", "category", "description",
                                                  "recommendation")} for p in (quality.problems or [])],
        "report": {k: report[k] for k in ("critical_count", "high_count", "deployment_blocked", "reason")},
    }

    print(f"{'ID':<4}{'FT':>7}{'A':>7}{'R':>7}{'M':>7}{'Q':>7}  level")
    for cid, c in components.items():
        print(f"{cid:<4}{c['fault_tolerance']:>7.3f}{c['availability']:>7.3f}{c['reliability']:>7.3f}"
              f"{c['maintainability']:>7.3f}{c['overall']:>7.3f}  {c['level']}")
    print(f"\n{len(data['problems'])} anti-pattern findings; deployment blocked: {data['report']['deployment_blocked']}")

    write_page(TEMPLATE, args.output, data)
    print(f"Walkthrough written to {args.output}")


if __name__ == "__main__":
    main()
