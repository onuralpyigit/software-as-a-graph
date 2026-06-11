"""
Aggregator service module.

This module aggregates platform relationships into JSON format by combining
CSV outputs from projects in specified directories with information from
parser modules.
"""

import os
import csv
import json
import re
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Set, Optional

from .models import Topic
from .parsers import SystemRepoParser, TypeSupportParser
from .converter import QosConverter
from common.logger import setup_logger, log_info, log_warning, log_error

LOG_LEVEL = "INFO"

SYSTEM_HIERARCHY_FIELDS = (
    "csc_name",
    "csci_name",
    "css_name",
    "csms_name",
)

# ---------------------------------------------------------------------------
# QoS-derived topic attributes (mirrored from saag/core/models.py)
# ---------------------------------------------------------------------------
# These constants and helpers are copied locally so the analyzer has no
# dependency on the saag package while producing the same frequency and
# criticality values it would.

#: Topic frequency bins (Hz) indexed by reliability×priority score [0,1].
TOPIC_FREQUENCY_HZ: List[float] = [
    1.0, 1.0, 5.0, 10.0, 10.0, 20.0, 20.0, 50.0,
    50.0, 100.0, 100.0, 150.0, 150.0, 200.0, 200.0, 200.0,
]

#: Thresholds for classifying a QoS weight score into criticality levels.
CRITICALITY_THRESHOLDS: List[Tuple[float, str]] = [
    (0.00, "minimal"),
    (0.19, "low"),
    (0.43, "medium"),
    (0.64, "high"),
    (1.00, "critical"),
]

#: QoS dimension scores (DDS string -> [0,1] score).
_RELIABILITY_SCORES: Dict[str, float] = {"BEST_EFFORT": 0.0, "RELIABLE": 1.0}
_DURABILITY_SCORES: Dict[str, float] = {
    "VOLATILE": 0.0,
    "TRANSIENT_LOCAL": 0.5,
    "TRANSIENT": 0.6,
    "PERSISTENT": 1.0,
}
_PRIORITY_SCORES: Dict[str, float] = {
    "LOW": 0.0,
    "MEDIUM": 0.33,
    "HIGH": 0.66,
    "URGENT": 1.0,
}

#: AHP-derived QoS weights: Durability(0.4) > Reliability(0.3) = Priority(0.3).
_W_RELIABILITY: float = 0.30
_W_DURABILITY: float = 0.40
_W_PRIORITY: float = 0.30

#: Random options for application attributes.
_APP_PRIORITY_OPTIONS: List[str] = ["LOW", "MEDIUM", "HIGH"]
_APP_HOTSTANDBY_OPTIONS: List[bool] = [False, True]


def _derive_topic_frequency(reliability: str, transport_priority: str) -> float:
    """Derive topic frequency (Hz) from reliability × priority score."""
    r = _RELIABILITY_SCORES.get(reliability, 0.0)
    p = _PRIORITY_SCORES.get(transport_priority, 0.0)
    combined = r * p  # range [0, 1]
    bin_idx = int(combined * len(TOPIC_FREQUENCY_HZ))
    bin_idx = max(0, min(bin_idx, len(TOPIC_FREQUENCY_HZ) - 1))
    return float(TOPIC_FREQUENCY_HZ[bin_idx])


def _derive_topic_criticality(
    durability: str, reliability: str, transport_priority: str
) -> str:
    """Derive topic criticality label from the AHP-weighted QoS score."""
    qos_score = (
        _W_RELIABILITY * _RELIABILITY_SCORES.get(reliability, 0.0)
        + _W_DURABILITY * _DURABILITY_SCORES.get(durability, 0.0)
        + _W_PRIORITY * _PRIORITY_SCORES.get(transport_priority, 0.0)
    )
    for threshold, label in CRITICALITY_THRESHOLDS:
        if qos_score <= threshold:
            return label
    return "critical"


@dataclass
class AggregatorConfig:
    """Configuration for the aggregator service.
    
    Attributes:
        root_dir: Root directory containing projects.
        projects_dir: Main directory containing CSV outputs.
        output_dir: Directory where JSON output will be created.
        logs_dir: Directory where log files will be written.
        platform_name: Platform name.
        config_dir: Directory containing configuration files (SYSTEM_REPO, etc.).
        structural_dir: Directory containing structural analysis results.
        dds_mask: Enable DDS QoS conversion (True) or skip conversion (False).
    """
    root_dir: str
    projects_dir: str
    output_dir: str
    logs_dir: str
    platform_name: str
    config_dir: str = ""
    structural_dir: str = ""
    log_level: str = LOG_LEVEL
    dds_mask: bool = True


def _read_csv_files(projects_dir: str, valid_app_names: Set[str]) -> List[Tuple[List[str], str, str]]:
    """
    Read and parse all CSV files in the specified directory.

    Args:
        projects_dir: Directory containing CSV files
        valid_app_names: Valid application names to process

    Returns:
        List of (csv_row, version, app_type) tuples for each row
    """
    csv_rows = []
    if not os.path.isdir(projects_dir):
        log_warning(f"Project directory not found: {projects_dir}")
        return csv_rows

    for file in os.listdir(projects_dir):
        if not file.endswith('.csv'):
            continue

        file_base = file[:-4] if file.endswith('.csv') else file

        app_name = file_base
        version = ""
        app_type = ""

        # Split at the last underscore followed by a semver-like version (digits.digits)
        version_match = re.search(r'_(?=\d+\.\d+)', file_base)
        if version_match:
            app_name = file_base[:version_match.start()]
            version = file_base[version_match.start() + 1:]
        
        # app_type is the last underscore-separated part of app_name
        app_parts = app_name.split('_')
        if len(app_parts) >= 2:
            app_type = app_parts[-1]

        # Only process CSVs for valid apps and libraries
        if app_name not in valid_app_names and not app_name.endswith('_lib'):
            continue

        file_path = os.path.join(projects_dir, file)
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            for line_num, row in enumerate(reader, 1):
                if not row:
                    continue
                if len(row) != 3:
                    log_warning(f"{file}:{line_num}: Expected 3 columns, got {len(row)} — skipping row")
                    continue
                csv_rows.append((row, version, app_type))
    
    return csv_rows


def _normalize_csv_value(value: str) -> str:
    """Normalize CSV cell values and replace blanks with NOT_FOUND."""
    normalized = value.strip().strip('"').strip()
    return normalized if normalized else "NOT_FOUND"


def _default_system_hierarchy() -> Dict[str, str]:
    """Return a default system hierarchy payload with NOT_FOUND values."""
    return {field: "NOT_FOUND" for field in SYSTEM_HIERARCHY_FIELDS}


def _read_system_hierarchy(config_dir: str) -> Dict[str, Dict[str, str]]:
    """Read system hierarchy metadata keyed by csu_name from config/csci_info.csv."""
    system_hierarchy_map: Dict[str, Dict[str, str]] = {}
    csci_info_path = os.path.join(config_dir, "csci_info.csv")

    if not config_dir or not os.path.isfile(csci_info_path):
        log_warning(f"System hierarchy info file not found: {csci_info_path}")
        return system_hierarchy_map

    with open(csci_info_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, skipinitialspace=True)
        rows = list(reader)

    if not rows:
        log_warning(f"System hierarchy info file is empty: {csci_info_path}")
        return system_hierarchy_map

    # Build column index from header row
    header = [_normalize_csv_value(h) for h in rows[0]]
    col_index: Dict[str, int] = {name: i for i, name in enumerate(header)}

    # Validate required columns exist in header
    required_cols = {"csu_name"} | set(SYSTEM_HIERARCHY_FIELDS)
    missing_cols = required_cols - set(col_index.keys())
    if missing_cols:
        log_warning(f"csci_info.csv: missing required columns: {missing_cols}. "
                    f"Available columns: {list(col_index.keys())}")

    if "csu_name" not in col_index:
        log_warning("csci_info.csv: 'csu_name' column not found — cannot read system hierarchy.")
        return system_hierarchy_map

    expected_col_count = len(header)

    for row_num, row in enumerate(rows[1:], 2):
        if not row:
            continue

        if len(row) != expected_col_count:
            log_warning(f"csci_info.csv:{row_num}: Expected {expected_col_count} columns, got {len(row)}")

        normalized_row = [_normalize_csv_value(value) for value in row]
        csu_idx = col_index["csu_name"]
        csu_name = normalized_row[csu_idx] if csu_idx < len(normalized_row) else "NOT_FOUND"
        if csu_name == "NOT_FOUND":
            continue

        entry: Dict[str, str] = {}
        for field_name in SYSTEM_HIERARCHY_FIELDS:
            idx = col_index.get(field_name)
            entry[field_name] = normalized_row[idx] if idx is not None and idx < len(normalized_row) else "NOT_FOUND"
        system_hierarchy_map[csu_name] = entry

    log_info(f"Loaded system hierarchy info for {len(system_hierarchy_map)} CSU entries.")
    return system_hierarchy_map


def _create_nodes(
    app_node_relations: List[Tuple[str, str]]
) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    """
    Create node list and name-ID mapping.

    A node name of ``"NOT_FOUND"`` indicates that the SYSTEM_REPO does not
    know which node hosts the application.  Such placeholders must not
    appear in the dataset as a real node, so they are filtered out here.

    Args:
        app_node_relations: List of (app_name, node_name) tuples

    Returns:
        (nodes list, {node_name: node_id} mapping)
    """
    node_names: Set[str] = {
        rel[1] for rel in app_node_relations if rel[1] and rel[1] != "NOT_FOUND"
    }
    sorted_node_names = sorted(list(node_names))
    node_map = {name: f"N{i}" for i, name in enumerate(sorted_node_names)}

    nodes = [{"id": node_map[name], "name": name} for name in sorted_node_names]
    return nodes, node_map


def _create_topics(
    topic_set: Set[Topic],
    dds_mask: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Create topic list and name-ID mapping.
    
    Args:
        topic_set: Set of Topic objects
        dds_mask: Enable DDS QoS conversion (True) or skip conversion (False)
    
    Returns:
        (topics list, {topic_name: topic_id} mapping)
    """
    topics = []
    topic_map = {}
    
    # Process topics sorted by name
    sorted_topics = sorted(topic_set, key=lambda inf: inf.name)
    
    for i, inf in enumerate(sorted_topics):
        topic_id = f"T{i}"
        topic_name = inf.name
        
        topic_data = {
            "id": topic_id,
            "name": topic_name,
            "size": inf.size if hasattr(inf, "size") else -1
        }
        
        # Check QoS fields, pass through converter, and add with NOT_FOUND default
        dur = inf.durability if hasattr(inf, "durability") else None
        rel = inf.reliability if hasattr(inf, "reliability") else None
        pri = inf.transport_priority if hasattr(inf, "transport_priority") else None
        
        # Pass QoS values through converter only if dds_mask is enabled
        if dds_mask:
            converted_dur, converted_rel, converted_pri, _ = QosConverter.convert_qos(dur, rel, pri)
        else:
            # Skip conversion, use raw values as strings
            converted_dur = str(dur) if dur is not None else None
            converted_rel = str(rel) if rel is not None else None
            converted_pri = str(pri) if pri is not None else None
        
        qos_dur = converted_dur if converted_dur is not None else "NOT_FOUND"
        qos_rel = converted_rel if converted_rel is not None else "NOT_FOUND"
        qos_pri = converted_pri if converted_pri is not None else "NOT_FOUND"

        topic_data["qos"] = {
            "durability": qos_dur,
            "reliability": qos_rel,
            "transport_priority": qos_pri
        }

        # Derive QoS-based attributes (mirrors saag/core/models.py Topic).
        topic_data["frequency"] = _derive_topic_frequency(qos_rel, qos_pri)
        topic_data["criticality"] = _derive_topic_criticality(qos_dur, qos_rel, qos_pri)

        topics.append(topic_data)
        topic_map[topic_name] = topic_id
    
    log_info(f"{len(topics)} topics created.")
    return topics, topic_map


def _create_apps_libs_and_relations(
    app_node_relations: List[Tuple[str, str]],
    csv_data: List[Tuple[List[str], str, str]],
    topic_map: Dict[str, str],
    app_role_map: Dict[str, List[str]],
    app_criticality_map: Dict[str, bool],
    system_hierarchy_map: Optional[Dict[str, Dict[str, str]]] = None,
    dds_mask: bool = True
) -> Tuple[List[Dict], Dict[str, str], List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    Create application and library lists, mappings, and relationships.
    
    Args:
        app_node_relations: List of (app_name, node_name) tuples
        csv_data: (row, version, app_type) tuples read from CSV
        topic_map: {topic_name: topic_id} mapping
        app_role_map: {app_name: [role, ...]} mapping from SystemRepoParser
        app_criticality_map: {app_name: criticality} mapping from SystemRepoParser
        system_hierarchy_map: Optional {csu_name: system_hierarchy} mapping from csci_info.csv
        dds_mask: Enable DDS QoS conversion (True) or skip conversion (False)
    
    Returns:
        (applications, app_map, publishes_to, subscribes_to, libs, uses)
    """
    if system_hierarchy_map is None:
        system_hierarchy_map = {}

    # App isimlerini get_app_node_relation'dan al
    app_names: Set[str] = {rel[0] for rel in app_node_relations}
    lib_names: Set[str] = set()
    
    # Metadata dictleri
    app_versions: Dict[str, str] = {}
    app_types: Dict[str, str] = {}
    lib_versions: Dict[str, str] = {}
    
    # Collect metadata from CSV data
    for row_data, version, app_type in csv_data:
        if not row_data or len(row_data) == 0:
            continue
            
        entity_name = row_data[0]
        
        # Classify entities ending with _lib as libraries
        if entity_name.endswith('_lib'):
            lib_names.add(entity_name)
            if version:
                lib_versions[entity_name] = version
        # Collect metadata for valid apps (don't add to app_names, only from SystemRepoParser)
        elif entity_name in app_names:
            if version:
                app_versions[entity_name] = version
            if app_type:
                app_types[entity_name] = app_type

    # Create app and lib ID mappings
    sorted_app_names = sorted(app_names)
    app_map = {name: f"A{i}" for i, name in enumerate(sorted_app_names)}

    # Add libraries from uses relationships of valid apps and libs
    # (Libs can also use other libs)
    for row, _, _ in csv_data:
        if len(row) == 3 and row[2].lower() == 'uses':
            source = row[0]
            target = row[1]
            # If source is a valid app or _lib, add target only if it is a _lib
            if (source in app_names or source.endswith('_lib')) and target.endswith('_lib'):
                lib_names.add(target)
    
    sorted_lib_names = sorted(lib_names)
    lib_map = {name: f"L{i}" for i, name in enumerate(sorted_lib_names)}
    
    # Create lib list
    libs = []
    for name in sorted_lib_names:
        lib_data = {
            "id": lib_map[name],
            "name": name,
            "version": lib_versions.get(name, "NOT_FOUND"),
        }
        if system_hierarchy_map:
            lib_data["system_hierarchy"] = dict(system_hierarchy_map.get(name, _default_system_hierarchy()))
        libs.append(lib_data)

    # Create relationships
    publishes_to, subscribes_to, uses = [], [], []
    
    for row, _, _ in csv_data:
        if len(row) != 3:
            continue
        
        source_entity, target_entity, action = row[0], row[1], row[2].lower()
        
        # Find source entity ID (can be app or lib)
        source_id = app_map.get(source_entity) or lib_map.get(source_entity)
        if not source_id:
            continue

        # Pub/Sub relationships
        if action in ("pub", "sub") and target_entity in topic_map:
            topic_id = topic_map[target_entity]
            relation = {"from": source_id, "to": topic_id}
            if action == "pub":
                publishes_to.append(relation)
            else:
                subscribes_to.append(relation)
        
        # Uses relationships (target can be lib or app)
        elif action == "uses":
            target_id = lib_map.get(target_entity) or app_map.get(target_entity)
            if target_id:
                uses.append({"from": source_id, "to": target_id})

    # Create applications list
    applications = []
    for name in sorted_app_names:
        # Criticality is stored as a plain boolean in the dataset JSON.
        # Parser returns a custom string (e.g. "custom_true").  Resolve it
        # through the boolean mapping to get "True"/"False", then convert
        # to a real Python bool.
        raw_crit = app_criticality_map.get(name)
        if raw_crit is None:
            criticality_value = False
        else:
            _, _, _, resolved = QosConverter.convert_qos(boolean=raw_crit)
            criticality_value = str(resolved).strip().lower() == "true"
        
        # app_type is the suffix after the last underscore in the app name
        # (derived directly from SYSTEM_REPO; does not require a CSV file).
        name_parts = name.rsplit('_', 1)
        derived_app_type = name_parts[1] if len(name_parts) == 2 and name_parts[1] else "NOT_FOUND"

        # Random priority/hotstandby, seeded by app name for reproducible reruns.
        app_rng = random.Random(name)
        priority_value = app_rng.choice(_APP_PRIORITY_OPTIONS)
        hotstandby_value = app_rng.choice(_APP_HOTSTANDBY_OPTIONS)

        app_data = {
            "id": app_map[name],
            "name": name,
            "version": app_versions.get(name, "NOT_FOUND"),
            "app_type": app_types.get(name, derived_app_type),
            "role": list(app_role_map.get(name, ["NOT_FOUND"])),
            "criticality": criticality_value,
            "priority": priority_value,
            "hotstandby": hotstandby_value,
        }
        if system_hierarchy_map:
            app_data["system_hierarchy"] = dict(system_hierarchy_map.get(name, _default_system_hierarchy()))
        
        applications.append(app_data)

    return applications, app_map, publishes_to, subscribes_to, libs, uses


def _read_metrics_data(projects_dir: str) -> Dict[str, Any]:
    r"""
    Read code metrics analysis results from ``<versioned_name>_metrics.json`` files.

    Scans the projects directory for ``*_metrics.json`` files. The filename
    is parsed as ``<app_name>_<version>_metrics.json`` (the same
    ``_(?=\d+\.\d+)`` rule used elsewhere in the pipeline). When a version
    cannot be detected the whole stem is treated as the app name (legacy
    layout). Symlinks pointing into the shared store are followed
    transparently.

    Args:
        projects_dir: Directory containing ``_metrics.json`` files
                      (same directory as CSV files: output/analyzed/<platform>/)

    Returns:
        Dictionary mapping app_name -> metrics dict (latest version wins
        if multiple versions of the same app are present).
    """
    metrics_data: Dict[str, Any] = {}

    if not os.path.isdir(projects_dir):
        return metrics_data

    for file in os.listdir(projects_dir):
        if not file.endswith('_metrics.json'):
            continue

        stem = file[:-len('_metrics.json')]
        version_match = re.search(r'_(?=\d+\.\d+)', stem)
        if version_match:
            app_name = stem[:version_match.start()]
        else:
            app_name = stem

        file_path = os.path.join(projects_dir, file)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if data:  # Non-empty dict means successful scan
                metrics_data[app_name] = data
                log_info(f"Loaded code metrics data for: {app_name} ({len(data)} sections)")
            else:
                log_warning(f"Empty code metrics data for: {app_name}")
        except (json.JSONDecodeError, IOError) as e:
            log_warning(f"Failed to read metrics data for {app_name}: {e}")

    log_info(f"Code metrics data loaded for {len(metrics_data)} applications.")
    return metrics_data


def _inject_structural_results(
    dataset: Dict[str, Any],
    structural_dir: str,
    platform_name: str,
) -> None:
    """Read structural analysis JSON and inject results into dataset.

    Looks for ``<platform>_structural_analysis.json`` produced by the structural
    stage.  When found, each component (application, topic, node, library)
    receives a ``structural_analysis`` field, and a top-level summary is added.
    """
    structural_file = os.path.join(
        structural_dir, platform_name, f"{platform_name}_structural_analysis.json"
    )
    if not os.path.isfile(structural_file):
        log_info("No structural analysis results found, skipping injection.")
        return

    try:
        with open(structural_file, "r", encoding="utf-8") as f:
            sa = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        log_warning(f"Failed to read structural analysis: {e}")
        return

    for key in ("applications", "topics", "nodes", "libraries"):
        sa_entries = sa.get(key, {})
        for item in dataset.get(key, []):
            name = item.get("name", "")
            if name in sa_entries:
                item["structural_analysis"] = sa_entries[name]

    dataset["structural_analysis"] = {
        "parameters": sa.get("parameters", {}),
        "quartiles": sa.get("quartiles", {}),
        "pattern_summary": sa.get("pattern_summary", {}),
    }

    log_info(f"Structural analysis injected from: {structural_file}")


def aggregate(
    root_dir: str, 
    projects_dir: str, 
    output_dir: str, 
    logs_dir: str,
    platform_name: str,
    config_dir: str = "",
    structural_dir: str = "",
    log_level: str = LOG_LEVEL,
    dds_mask: bool = True
) -> None:
    """
    Collect and process all data, then write to JSON file.
    
    Args:
        root_dir: Root directory containing projects
        projects_dir: Main directory containing CSV outputs
        output_dir: Directory where JSON output will be created
        logs_dir: Directory where log files will be written
        platform_name: Platform name (projects subdirectory)
        config_dir: Directory containing configuration files (SYSTEM_REPO, etc.)
        log_level: Logging level
        dds_mask: Enable DDS QoS conversion (True) or skip conversion (False)
    """
    # 0. Setup Logging
    os.makedirs(logs_dir, exist_ok=True)
    setup_logger(logs_dir, platform_name, stage="aggregate", log_level=log_level)

    log_info("=" * 60)
    log_info("STAGE 3: AGGREGATE")
    log_info("=" * 60)

    log_info("Aggregator started.")

    # 1. Data Collection
    log_info("Data collection started.")
    
    # Get data from SystemRepoParser
    system_repo_parser = SystemRepoParser(platform_name, config_dir)
    
    try:
        app_node_rel = system_repo_parser.get_app_node_relation()
        log_info(f"{len(app_node_rel)} app-node relationships retrieved.")
    except Exception as e:
        log_error(f"Failed to retrieve app-node relationships: {e}")
        app_node_rel = []
    
    try:
        app_role_map = system_repo_parser.get_app_role_relation()
        log_info(f"{len(app_role_map)} app-role mappings retrieved.")
    except Exception as e:
        log_error(f"Failed to retrieve app-role map: {e}")
        app_role_map = {}
    
    try:
        app_criticality_map = system_repo_parser.get_app_criticality_relation()
        log_info(f"{len(app_criticality_map)} app-criticality mappings retrieved.")
    except Exception as e:
        log_error(f"Failed to retrieve app-criticality map: {e}")
        app_criticality_map = {}
    
    # Get topic set from TypeSupportParser
    try:
        topic_set = TypeSupportParser(root_dir).get_topic_list()
        log_info(f"{len(topic_set)} topics retrieved.")
    except Exception as e:
        log_error(f"Failed to retrieve topic set: {e}")
        topic_set = set()
    
    # Read CSV files
    platform_projects_dir = os.path.join(projects_dir, platform_name)
    valid_app_names = {rel[0] for rel in app_node_rel}
    csv_data = _read_csv_files(platform_projects_dir, valid_app_names)
    log_info(f"{len(csv_data)} CSV rows read.")

    # Read code metrics analysis results
    metrics_data = _read_metrics_data(platform_projects_dir)
    if metrics_data:
        log_info(f"Code metrics data available for {len(metrics_data)} applications.")
    else:
        log_info("No code metrics data found. Applications will have code_metrics=null.")

    system_hierarchy_map = _read_system_hierarchy(config_dir)

    # 2. Process Data
    nodes, node_map = _create_nodes(app_node_rel)
    topics, topic_map = _create_topics(topic_set, dds_mask=dds_mask)
    apps, app_map, pub_to, sub_to, libs, uses = _create_apps_libs_and_relations(
        app_node_rel,
        csv_data,
        topic_map,
        app_role_map,
        app_criticality_map,
        system_hierarchy_map,
        dds_mask=dds_mask
    )
    # Apps with node == "NOT_FOUND" stay in the graph but produce no runs_on
    # relation (the placeholder node is intentionally absent from node_map).
    runs_on = [
        {"from": app_map[app], "to": node_map[node]}
        for app, node in app_node_rel
        if app in app_map and node in node_map
    ]
    apps_without_node = sorted({
        app for app, node in app_node_rel
        if app in app_map and (not node or node == "NOT_FOUND")
    })
    if apps_without_node:
        log_info(
            f"{len(apps_without_node)} application(s) have no host node "
            f"(NOT_FOUND); included without runs_on relation: "
            f"{', '.join(apps_without_node)}"
        )
    log_info("All data processed and relationships created.")

    # Enrich applications and libraries with code metrics data
    # Strip per-class detail lists (kept in per-app _metrics.json, too verbose for aggregated output)
    _CLASS_DETAIL_KEYS = {"high_complexity_classes", "high_coupling_classes", "low_cohesion_classes"}

    def _strip_detail_keys(metrics_dict):
        """Return metrics dict with per-class detail lists removed."""
        return {
            section: {k: v for k, v in values.items() if k not in _CLASS_DETAIL_KEYS}
            if isinstance(values, dict) else values
            for section, values in metrics_dict.items()
        }

    for app in apps:
        app_name = app["name"]
        if app_name in metrics_data:
            app["code_metrics"] = _strip_detail_keys(metrics_data[app_name])
        else:
            app["code_metrics"] = None

    for lib in libs:
        lib_name = lib["name"]
        if lib_name in metrics_data:
            lib["code_metrics"] = _strip_detail_keys(metrics_data[lib_name])
        else:
            lib["code_metrics"] = None

    # 3. Create and Write JSON Output
    dataset = {
        "metadata": {
            "scale": {
                "apps": len(apps),
                "topics": len(topics),
                "nodes": len(nodes),
                "libraries": len(libs),
                "brokers": 0
            }
        },
        "nodes": nodes,
        "brokers": [],
        "topics": topics,
        "applications": apps,
        "libraries": libs,
        "relationships": {
            "runs_on": runs_on,
            "routes": [],
            "publishes_to": pub_to,
            "subscribes_to": sub_to,
            "uses": uses
        }
    }

    # Inject structural analysis results if available
    if structural_dir:
        _inject_structural_results(dataset, structural_dir, platform_name)

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{platform_name}_relations.json"
    output_path = os.path.join(output_dir, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2)

    success_msg = f"Dataset successfully created: {output_path}"
    log_info(success_msg)


class AggregatorService:
    """Service class for aggregating project data."""

    def __init__(self, config: AggregatorConfig):
        """Initialize the aggregator service.
        
        Args:
            config: AggregatorConfig instance.
        """
        self.config = config

    def run(self) -> int:
        """Execute the aggregator pipeline.
        
        Returns:
            Exit code (0 for success, non-zero for failure).
        """
        try:
            aggregate(
                root_dir=self.config.root_dir,
                projects_dir=self.config.projects_dir,
                output_dir=self.config.output_dir,
                logs_dir=self.config.logs_dir,
                platform_name=self.config.platform_name,
                config_dir=self.config.config_dir,
                structural_dir=self.config.structural_dir,
                log_level=self.config.log_level,
                dds_mask=self.config.dds_mask
            )
            return 0
        except Exception as e:
            log_error(f"Aggregation failed: {e}")
            return 1
