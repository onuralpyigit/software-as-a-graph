"""
Structural analysis service module.

Orchestrates the full structural analysis pipeline:
1. Build internal data model from raw sources (CSV + SYSTEM_REPO + TypeSupport)
2. Calculate structural metrics
3. Detect anomaly patterns via relative quartile interpretation
4. Compute combined anomaly scores
5. Save structural analysis results to JSON
6. Generate Markdown report
"""

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

from common.logger import setup_logger, log_info, log_error

from .metrics import calculate_all_metrics, AllMetrics
from .patterns import detect_patterns, PatternResults
from .scoring import calculate_scores, ScoringResults, DEFAULT_TAU, DEFAULT_LAMBDA
from .reporter import generate_markdown_report, log_report_summary

LOG_LEVEL = "INFO"


@dataclass
class StructuralConfig:
    """Configuration for the structural analysis service.

    Attributes:
        root_dir: Root directory of cloned projects (for TypeSupportParser).
        projects_dir: Directory containing analyzed CSV outputs.
        output_dir: Directory where outputs will be written.
        logs_dir: Directory where log files will be written.
        platform_name: Platform name for output file naming.
        config_dir: Directory containing configuration files (SYSTEM_REPO, etc.).
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR).
        dds_mask: Enable DDS QoS conversion.
        k: Upper bound for low-connectivity classification (LCR metric).
        tau: Cap for single-dimension contribution in scoring.
        lam: Weight for single-dimension term in combined score.
    """
    root_dir: str
    projects_dir: str
    output_dir: str
    logs_dir: str
    platform_name: str
    config_dir: str = ""
    log_level: str = LOG_LEVEL
    dds_mask: bool = True
    k: int = 3
    tau: float = DEFAULT_TAU
    lam: float = DEFAULT_LAMBDA


def _build_data_model(
    root_dir: str,
    projects_dir: str,
    platform_name: str,
    config_dir: str,
    dds_mask: bool,
) -> Dict[str, Any]:
    """Build internal data model from raw sources.

    Re-uses aggregator parsing functions to construct the same entity-relationship
    graph that aggregate would produce, so structural analysis can run independently.

    Returns:
        Data dict with nodes, topics, applications, libraries, relationships.
    """
    from pipeline.aggregator.parsers import SystemRepoParser, TypeSupportParser
    from pipeline.aggregator.service import (
        _read_csv_files,
        _create_nodes,
        _create_topics,
        _create_apps_libs_and_relations,
    )

    system_repo_parser = SystemRepoParser(platform_name, config_dir)
    app_node_rel = system_repo_parser.get_app_node_relation()
    app_role_map = system_repo_parser.get_app_role_relation()
    app_criticality_map = system_repo_parser.get_app_criticality_relation()

    topic_set = TypeSupportParser(root_dir).get_topic_list()

    platform_projects_dir = os.path.join(projects_dir, platform_name)
    valid_app_names = {rel[0] for rel in app_node_rel}
    csv_data = _read_csv_files(platform_projects_dir, valid_app_names)

    nodes, node_map = _create_nodes(app_node_rel)
    topics, topic_map = _create_topics(topic_set, dds_mask=dds_mask)
    apps, app_map, pub_to, sub_to, libs, uses = _create_apps_libs_and_relations(
        app_node_rel, csv_data, topic_map, app_role_map, app_criticality_map,
        dds_mask=dds_mask,
    )
    runs_on = [
        {"from": app_map[app], "to": node_map[node]}
        for app, node in app_node_rel
        if app in app_map and node in node_map
    ]

    return {
        "metadata": {"scale": str({
            "apps": len(apps), "topics": len(topics),
            "nodes": len(nodes), "libraries": len(libs), "brokers": 0,
        })},
        "nodes": nodes,
        "topics": topics,
        "applications": apps,
        "libraries": libs,
        "relationships": {
            "runs_on": runs_on,
            "publishes_to": pub_to,
            "subscribes_to": sub_to,
            "uses": uses,
        },
    }


def _save_structural_results(
    metrics: AllMetrics,
    patterns: PatternResults,
    scores: ScoringResults,
    output_dir: str,
    platform_name: str,
) -> str:
    """Save structural analysis results to a standalone JSON file.

    The JSON is keyed by component **name** so that the aggregate module can
    look up results without relying on ID assignments.

    Returns:
        Path to the written JSON file.
    """

    def _matched_patterns(comp_id: str, pattern_dict: Dict[str, list]) -> List[str]:
        matched = []
        for pname, match_list in pattern_dict.items():
            if any(m.id == comp_id for m in match_list):
                matched.append(pname)
        return matched

    def _build_entries(metric_list, score_list, pattern_dict):
        s_map = {s.id: s.to_dict() for s in score_list}
        entries: Dict[str, Any] = {}
        for m in metric_list:
            md = m.to_dict()
            name = md.get("name", m.id)
            sd = s_map.get(m.id, {})
            entries[name] = {
                "metrics": {k: v for k, v in md.items() if k not in ("id", "name")},
                "patterns": _matched_patterns(m.id, pattern_dict),
                "scores": {
                    "pattern_score": sd.get("pattern_score", 0),
                    "uni_score": sd.get("uni_score", 0),
                    "total_score": sd.get("total_score", 0),
                },
            }
        return entries

    result = {
        "parameters": scores.parameters,
        "quartiles": patterns.to_dict().get("quartiles", {}),
        "pattern_summary": {
            "applications": {k: len(v) for k, v in patterns.app_patterns.items()},
            "topics": {k: len(v) for k, v in patterns.topic_patterns.items()},
            "nodes": {k: len(v) for k, v in patterns.node_patterns.items()},
            "libraries": {k: len(v) for k, v in patterns.lib_patterns.items()},
        },
        "applications": _build_entries(
            metrics.applications, scores.applications, patterns.app_patterns,
        ),
        "topics": _build_entries(
            metrics.topics, scores.topics, patterns.topic_patterns,
        ),
        "nodes": _build_entries(
            metrics.nodes, scores.nodes, patterns.node_patterns,
        ),
        "libraries": _build_entries(
            metrics.libraries, scores.libraries, patterns.lib_patterns,
        ),
    }

    platform_dir = os.path.join(output_dir, platform_name)
    os.makedirs(platform_dir, exist_ok=True)
    output_path = os.path.join(platform_dir, f"{platform_name}_structural_analysis.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return output_path


def analyze(
    root_dir: str,
    projects_dir: str,
    output_dir: str,
    logs_dir: str,
    platform_name: str,
    config_dir: str = "",
    log_level: str = LOG_LEVEL,
    dds_mask: bool = True,
    k: int = 3,
    tau: float = DEFAULT_TAU,
    lam: float = DEFAULT_LAMBDA,
) -> tuple:
    """
    Execute the structural analysis pipeline.

    Args:
        root_dir: Root directory of cloned projects.
        projects_dir: Directory containing analyzed CSV outputs.
        output_dir: Output directory for results.
        logs_dir: Directory for log files.
        platform_name: Platform name for file naming.
        config_dir: Directory containing configuration files (SYSTEM_REPO, etc.).
        log_level: Logging level.
        dds_mask: Enable DDS QoS conversion.
        k: LCR low-connectivity threshold.
        tau: Single-dimension cap.
        lam: Single-dimension weight.

    Returns:
        Tuple of (AllMetrics, PatternResults, ScoringResults).
    """
    # Setup logging
    os.makedirs(logs_dir, exist_ok=True)
    setup_logger(logs_dir, platform_name, stage="structural", log_level=log_level)

    log_info("=" * 60)
    log_info("STAGE 3: STRUCTURAL ANALYSIS")
    log_info("=" * 60)

    log_info(f"Root dir: {root_dir}")
    log_info(f"Projects dir: {projects_dir}")
    log_info(f"Output dir: {output_dir}")
    log_info(f"Parameters: k={k}, tau={tau}, lambda={lam}")

    # Step 1: Build data model from raw sources
    log_info("Building data model from raw sources...")
    data = _build_data_model(root_dir, projects_dir, platform_name, config_dir, dds_mask)
    scale_str = data.get("metadata", {}).get("scale", "{}")
    log_info(f"Data scale: {scale_str}")

    # Step 2: Calculate metrics
    log_info("Calculating structural metrics...")
    metrics = calculate_all_metrics(data, k=k)
    log_info(
        f"  Apps={len(metrics.applications)}, Topics={len(metrics.topics)}, "
        f"Nodes={len(metrics.nodes)}, Libs={len(metrics.libraries)}"
    )

    # Step 3: Detect patterns
    log_info("Detecting structural anomaly patterns...")
    patterns = detect_patterns(metrics)

    # Step 4: Compute scores
    log_info("Computing combined anomaly scores...")
    scores = calculate_scores(metrics, patterns, tau=tau, lam=lam)

    # Step 5: Save structural analysis results
    log_info("Saving structural analysis results...")
    result_path = _save_structural_results(metrics, patterns, scores, output_dir, platform_name)
    log_info(f"Structural analysis JSON: {result_path}")

    # Step 6: Generate Markdown report
    log_info("Generating Markdown report...")
    generate_markdown_report(metrics, patterns, scores, output_dir, platform_name)

    # Log summary
    log_report_summary(metrics, patterns, scores)

    log_info("Structural analysis completed successfully.")

    return metrics, patterns, scores


class StructuralService:
    """Service class for structural analysis of publish-subscribe systems."""

    def __init__(self, config: StructuralConfig):
        """Initialize the structural analysis service.

        Args:
            config: StructuralConfig instance with analysis parameters.
        """
        self.config = config

    def run(self) -> int:
        """Execute the structural analysis pipeline.

        Returns:
            Exit code (0 for success, non-zero for failure).
        """
        try:
            analyze(
                root_dir=self.config.root_dir,
                projects_dir=self.config.projects_dir,
                output_dir=self.config.output_dir,
                logs_dir=self.config.logs_dir,
                platform_name=self.config.platform_name,
                config_dir=self.config.config_dir,
                log_level=self.config.log_level,
                dds_mask=self.config.dds_mask,
                k=self.config.k,
                tau=self.config.tau,
                lam=self.config.lam,
            )
            return 0
        except FileNotFoundError as e:
            log_error(f"Input not found: {e}")
            return 1
        except Exception as e:
            log_error(f"Structural analysis failed: {e}")
            import traceback
            log_error(traceback.format_exc())
            return 1
