"""
Data analyzer module.

This module analyzes the aggregated JSON data and produces
statistics for nodes, apps, libs, and topics.
"""

from typing import Dict, List, Any, Set, Union
from dataclasses import dataclass, field

from .statistics import calculate_descriptive_stats, calculate_categorical_stats, sort_entities_by_metric, DescriptiveStats, CategoricalStats
from .metric_ids import (
    NODE_APPLICATION_COUNT,
    NODE_DOMAIN_HIERARCHY_DIVERSITY_COUNT,
    APP_DIRECT_PUBLISH_COUNT,
    APP_DIRECT_SUBSCRIBE_COUNT,
    APP_TOTAL_PUBLISH_COUNT,
    APP_TOTAL_SUBSCRIBE_COUNT,
    APP_ROLE_DISTRIBUTION,
    APP_CRITICALITY_DISTRIBUTION,
    APP_HIERARCHY_COMPONENT_DISTRIBUTION,
    APP_HIERARCHY_CONFIG_ITEM_DISTRIBUTION,
    APP_HIERARCHY_DOMAIN_DISTRIBUTION,
    APP_HIERARCHY_SYSTEM_DISTRIBUTION,
    APP_HIERARCHY_DOMAIN_AVG_DIRECT_PUBLISH_COUNT,
    APP_HIERARCHY_DOMAIN_AVG_DIRECT_SUBSCRIBE_COUNT,
    APP_HIERARCHY_DOMAIN_TOPIC_VARIETY_COUNT,
    APP_HIERARCHY_CONFIG_ITEM_AVG_DIRECT_PUBLISH_COUNT,
    APP_HIERARCHY_CONFIG_ITEM_AVG_DIRECT_SUBSCRIBE_COUNT,
    APP_HIERARCHY_CONFIG_ITEM_TOPIC_VARIETY_COUNT,
    LIB_APPLICATION_USAGE_COUNT,
    LIB_DIRECT_PUBLISH_COUNT,
    LIB_DIRECT_SUBSCRIBE_COUNT,
    LIB_TOTAL_PUBLISH_COUNT,
    LIB_TOTAL_SUBSCRIBE_COUNT,
    LIB_HIERARCHY_CONFIG_ITEM_DISTRIBUTION,
    LIB_HIERARCHY_DOMAIN_DISTRIBUTION,
    LIB_HIERARCHY_COMPLETENESS_PERCENT,
    TOPIC_SIZE_BYTES,
    TOPIC_PUBLISHER_APPLICATION_COUNT,
    TOPIC_SUBSCRIBER_APPLICATION_COUNT,
    TOPIC_QOS_DURABILITY_DISTRIBUTION,
    TOPIC_QOS_RELIABILITY_DISTRIBUTION,
    TOPIC_QOS_TRANSPORT_PRIORITY_DISTRIBUTION,
    STRUCTURAL_TOP_APPS,
    STRUCTURAL_TOP_TOPICS,
    STRUCTURAL_TOP_NODES,
    STRUCTURAL_TOP_LIBS,
    USES_CYCLE_DISTRIBUTION,
)


SYSTEM_HIERARCHY_FIELDS = (
    "csc_name",
    "csci_name",
    "css_name",
    "csms_name",
)


@dataclass
class ComponentAnalysis:
    """Analysis result for a single metric."""
    metric_name: str
    ranked_list: List[Dict[str, Any]]  # [{id, name, value}, ...]
    stats: Union[DescriptiveStats, CategoricalStats]
    description: str = ""
    is_categorical: bool = False  # True for distribution metrics
    show_entities: bool = True
    show_charts: bool = True
    show_statistics: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric_name,
            "description": self.description,
            "statistics": self.stats.to_dict(),
            "ranked_list": self.ranked_list,
            "is_categorical": self.is_categorical,
            "show_entities": self.show_entities,
            "show_charts": self.show_charts,
            "show_statistics": self.show_statistics,
        }


@dataclass 
class AnalysisReport:
    """Complete analysis report."""
    platform_name: str
    node_analysis: List[ComponentAnalysis] = field(default_factory=list)
    app_analysis: List[ComponentAnalysis] = field(default_factory=list)
    lib_analysis: List[ComponentAnalysis] = field(default_factory=list)
    topic_analysis: List[ComponentAnalysis] = field(default_factory=list)
    structural_analysis: List[ComponentAnalysis] = field(default_factory=list)
    extras_analysis: List[ComponentAnalysis] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)  # Raw JSON for cross-cutting charts
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform_name,
            "nodes": [a.to_dict() for a in self.node_analysis],
            "applications": [a.to_dict() for a in self.app_analysis],
            "libraries": [a.to_dict() for a in self.lib_analysis],
            "topics": [a.to_dict() for a in self.topic_analysis],
            "structural": [a.to_dict() for a in self.structural_analysis],
            "extras": [a.to_dict() for a in self.extras_analysis]
        }


def analyze_data(data: Dict[str, Any], platform_name: str, *, dds_mask: bool = True) -> AnalysisReport:
    """
    Analyze the aggregated JSON data.
    
    Args:
        data: Parsed JSON from aggregator output.
        platform_name: Platform name.
        dds_mask: When True display DDS standard QoS names, when False display custom names.
    
    Returns:
        AnalysisReport with all statistics.
    """
    report = AnalysisReport(platform_name=platform_name)
    
    # Store raw data for cross-cutting charts
    report.raw_data = data
    
    # Extract components
    nodes = data.get("nodes", [])
    apps = data.get("applications", [])
    libs = data.get("libraries", [])
    topics = data.get("topics", [])
    relationships = data.get("relationships", {})
    
    # Build lookup maps
    runs_on = relationships.get("runs_on", [])
    publishes_to = relationships.get("publishes_to", [])
    subscribes_to = relationships.get("subscribes_to", [])
    uses = relationships.get("uses", [])
    
    # Build dependency graph for recursive load calculation
    uses_graph = _build_uses_graph(uses)
    
    # Build pub/sub counts for all entities (apps + libs)
    entity_pub_count, entity_sub_count = _build_entity_pubsub_counts(
        apps, libs, publishes_to, subscribes_to
    )
    
    # Build entity names map for library breakdown reporting
    entity_names: Dict[str, str] = {}
    entity_versions: Dict[str, str] = {}
    for app in apps:
        entity_names[app["id"]] = app.get("name", app["id"])
        entity_versions[app["id"]] = app.get("version", "")
    for lib in libs:
        entity_names[lib["id"]] = lib.get("name", lib["id"])
        entity_versions[lib["id"]] = lib.get("version", "")
    
    # 1. NODE ANALYSIS: App count per node
    report.node_analysis = _analyze_nodes(nodes, runs_on, apps)
    
    # 2. APP ANALYSIS: Pub/Sub counts + Total Load (recursive)
    report.app_analysis = _analyze_apps(
        apps, publishes_to, subscribes_to, 
        uses_graph, entity_pub_count, entity_sub_count, entity_names, entity_versions
    )
    
    # 3. LIB ANALYSIS: Usage count, Pub/Sub counts + Total Load (recursive)
    report.lib_analysis = _analyze_libs(
        libs, uses, publishes_to, subscribes_to,
        uses_graph, entity_pub_count, entity_sub_count, entity_names, entity_versions
    )
    
    # 4. TOPIC  ANALYSIS: Size and Pub/Sub counts
    report.topic_analysis = _analyze_topics(topics, publishes_to, subscribes_to, dds_mask=dds_mask)
    
    # 5. STRUCTURAL ANALYSIS: Top scores from structural module
    report.structural_analysis = _analyze_structural(data)
    
    # 6. EXTRAS ANALYSIS: Cross-cutting analyses
    report.extras_analysis = [
        _build_uses_cycle_distribution_analysis(uses_graph, entity_names, entity_versions),
    ]
    
    return report


def _build_uses_graph(uses: List[Dict]) -> Dict[str, Set[str]]:
    """Build a graph of uses relationships: entity -> set of used libs."""
    graph: Dict[str, Set[str]] = {}
    for rel in uses:
        from_id = rel.get("from")
        to_id = rel.get("to")
        if from_id and to_id:
            if from_id not in graph:
                graph[from_id] = set()
            graph[from_id].add(to_id)
    return graph


def _build_entity_pubsub_counts(
    apps: List[Dict],
    libs: List[Dict],
    publishes_to: List[Dict],
    subscribes_to: List[Dict]
) -> tuple:
    """Build pub/sub counts for all entities (apps + libs)."""
    all_ids = {a["id"] for a in apps} | {lib["id"] for lib in libs}
    
    pub_count: Dict[str, int] = {eid: 0 for eid in all_ids}
    sub_count: Dict[str, int] = {eid: 0 for eid in all_ids}
    
    for rel in publishes_to:
        entity_id = rel.get("from")
        if entity_id in pub_count:
            pub_count[entity_id] += 1
    
    for rel in subscribes_to:
        entity_id = rel.get("from")
        if entity_id in sub_count:
            sub_count[entity_id] += 1
    
    return pub_count, sub_count


def _get_system_hierarchy_value(entity: Dict[str, Any], field_name: str) -> str:
    """Return a normalized system hierarchy value for the requested field."""
    hierarchy = entity.get("system_hierarchy") or {}
    value = hierarchy.get(field_name, "NOT_FOUND")
    if value is None:
        return "NOT_FOUND"

    normalized = str(value).strip()
    return normalized if normalized else "NOT_FOUND"


def _calculate_system_hierarchy_completeness(entity: Dict[str, Any]) -> float:
    """Return completeness percentage across system hierarchy fields."""
    present_count = sum(
        1 for field in SYSTEM_HIERARCHY_FIELDS
        if _get_system_hierarchy_value(entity, field) != "NOT_FOUND"
    )
    return (present_count / len(SYSTEM_HIERARCHY_FIELDS)) * 100


def _build_hierarchy_distribution_analysis(
    entities: List[Dict[str, Any]],
    hierarchy_field: str,
    metric_name: str,
    description: str,
) -> ComponentAnalysis:
    """Build a categorical distribution analysis for a hierarchy field."""
    counts: Dict[str, int] = {}

    for entity in entities:
        hierarchy_value = _get_system_hierarchy_value(entity, hierarchy_field)
        counts[hierarchy_value] = counts.get(hierarchy_value, 0) + 1

    ranked = [
        {
            "id": value,
            "name": value,
            "value": count,
        }
        for value, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)
    ]

    return ComponentAnalysis(
        metric_name=metric_name,
        description=description,
        ranked_list=ranked,
        stats=calculate_categorical_stats(counts),
        is_categorical=True,
        show_entities=False,
    )


def _build_hierarchy_completeness_analysis(
    entities: List[Dict[str, Any]],
    metric_name: str,
    description: str,
    entity_versions: Dict[str, str] = None,
) -> ComponentAnalysis:
    """Build a numeric completeness analysis for system hierarchy coverage."""
    if entity_versions is None:
        entity_versions = {}

    enriched = []
    for entity in entities:
        entity_id = entity.get("id", "")
        entity_name = entity.get("name", entity_id)
        completeness = _calculate_system_hierarchy_completeness(entity)
        enriched.append({
            "id": entity_id,
            "name": entity_name,
            "version": entity_versions.get(entity_id, ""),
            "completeness": completeness,
        })

    sorted_items = sort_entities_by_metric(enriched, "completeness", descending=True)
    values = [value for _, _, value in sorted_items]
    ranked = [
        {
            "id": entity_id,
            "name": entity_name,
            "version": entity_versions.get(entity_id, ""),
            "value": value,
        }
        for entity_id, entity_name, value in sorted_items
    ]

    return ComponentAnalysis(
        metric_name=metric_name,
        description=description,
        ranked_list=ranked,
        stats=calculate_descriptive_stats(values),
    )


def _build_group_average_analysis(
    entities: List[Dict[str, Any]],
    value_by_entity_id: Dict[str, int],
    hierarchy_field: str,
    metric_name: str,
    description: str,
) -> ComponentAnalysis:
    """Build a numeric analysis of average value per hierarchy bucket."""
    grouped_values: Dict[str, List[int]] = {}

    for entity in entities:
        entity_id = entity.get("id", "")
        group_name = _get_system_hierarchy_value(entity, hierarchy_field)
        if group_name not in grouped_values:
            grouped_values[group_name] = []
        grouped_values[group_name].append(value_by_entity_id.get(entity_id, 0))

    enriched = []
    for group_name, values in grouped_values.items():
        average_value = (sum(values) / len(values)) if values else 0.0
        enriched.append({
            "id": group_name,
            "name": group_name,
            "average_value": average_value,
        })

    sorted_items = sort_entities_by_metric(enriched, "average_value", descending=True)
    values = [value for _, _, value in sorted_items]
    ranked = [
        {"id": entity_id, "name": entity_name, "value": value}
        for entity_id, entity_name, value in sorted_items
    ]

    return ComponentAnalysis(
        metric_name=metric_name,
        description=description,
        ranked_list=ranked,
        stats=calculate_descriptive_stats(values),
    )


def _build_group_topic_variety_analysis(
    entities: List[Dict[str, Any]],
    publishes_to: List[Dict],
    subscribes_to: List[Dict],
    hierarchy_field: str,
    metric_name: str,
    description: str,
) -> ComponentAnalysis:
    """Build a numeric analysis of distinct topics touched by each hierarchy group."""
    entity_ids = {entity.get("id", "") for entity in entities}
    entity_topics: Dict[str, Set[str]] = {entity_id: set() for entity_id in entity_ids}

    for rel in publishes_to:
        entity_id = rel.get("from")
        topic_id = rel.get("to")
        if entity_id in entity_topics and topic_id:
            entity_topics[entity_id].add(topic_id)

    for rel in subscribes_to:
        entity_id = rel.get("from")
        topic_id = rel.get("to")
        if entity_id in entity_topics and topic_id:
            entity_topics[entity_id].add(topic_id)

    grouped_topics: Dict[str, Set[str]] = {}
    for entity in entities:
        entity_id = entity.get("id", "")
        group_name = _get_system_hierarchy_value(entity, hierarchy_field)
        if group_name not in grouped_topics:
            grouped_topics[group_name] = set()
        grouped_topics[group_name].update(entity_topics.get(entity_id, set()))

    enriched = []
    for group_name, topics in grouped_topics.items():
        enriched.append({
            "id": group_name,
            "name": group_name,
            "topic_variety": len(topics),
        })

    sorted_items = sort_entities_by_metric(enriched, "topic_variety", descending=True)
    values = [value for _, _, value in sorted_items]
    ranked = [
        {"id": entity_id, "name": entity_name, "value": value}
        for entity_id, entity_name, value in sorted_items
    ]

    return ComponentAnalysis(
        metric_name=metric_name,
        description=description,
        ranked_list=ranked,
        stats=calculate_descriptive_stats(values),
    )


def _format_entity_name(name: str, version: str = "") -> str:
    """Format entity name with version when available."""
    if version and version != "NOT_FOUND":
        return f"{name} ({version})"
    return str(name)


def _get_all_dependencies_recursive(entity_id: str, uses_graph: Dict[str, Set[str]]) -> Set[str]:
    """Get all dependencies recursively (transitive closure)."""
    all_deps: Set[str] = set()
    visited: Set[str] = {entity_id}
    
    def _recurse(eid: str):
        direct_deps = uses_graph.get(eid, set())
        for dep in direct_deps:
            if dep not in visited:
                visited.add(dep)
                all_deps.add(dep)
                _recurse(dep)
    
    _recurse(entity_id)
    return all_deps


def _calculate_total_load(
    entity_id: str,
    uses_graph: Dict[str, Set[str]],
    entity_pub_count: Dict[str, int],
    entity_sub_count: Dict[str, int],
    entity_names: Dict[str, str] = None
) -> tuple:
    """Calculate total pub/sub load including all recursive dependencies.
    
    Returns:
        tuple: (total_pub, total_sub, pub_breakdown, sub_breakdown)
        pub_breakdown and sub_breakdown are lists of (lib_name, count) tuples
        for libraries that contribute to the total.
    """
    # Get all transitive dependencies
    all_deps = _get_all_dependencies_recursive(entity_id, uses_graph)
    
    # Sum own + all dependencies
    total_pub = entity_pub_count.get(entity_id, 0)
    total_sub = entity_sub_count.get(entity_id, 0)
    
    # Track library contributions
    pub_breakdown = []
    sub_breakdown = []
    
    if entity_names is None:
        entity_names = {}
    
    # Include entity's own contribution in breakdown
    entity_name = entity_names.get(entity_id, entity_id)
    own_pub = entity_pub_count.get(entity_id, 0)
    own_sub = entity_sub_count.get(entity_id, 0)
    
    if own_pub > 0:
        pub_breakdown.append((f"{entity_name} (kendisi)", own_pub))
    if own_sub > 0:
        sub_breakdown.append((f"{entity_name} (kendisi)", own_sub))
    
    for dep_id in all_deps:
        dep_pub = entity_pub_count.get(dep_id, 0)
        dep_sub = entity_sub_count.get(dep_id, 0)
        dep_name = entity_names.get(dep_id, dep_id)
        
        total_pub += dep_pub
        total_sub += dep_sub
        
        if dep_pub > 0:
            pub_breakdown.append((dep_name, dep_pub))
        if dep_sub > 0:
            sub_breakdown.append((dep_name, dep_sub))
    
    # Sort breakdowns by count descending
    pub_breakdown.sort(key=lambda x: x[1], reverse=True)
    sub_breakdown.sort(key=lambda x: x[1], reverse=True)
    
    return total_pub, total_sub, pub_breakdown, sub_breakdown


def _find_strongly_connected_components(graph: Dict[str, Set[str]]) -> List[Set[str]]:
    """Return strongly connected components for a directed graph."""
    index = 0
    stack: List[str] = []
    on_stack: Set[str] = set()
    indices: Dict[str, int] = {}
    lowlinks: Dict[str, int] = {}
    components: List[Set[str]] = []

    def _strongconnect(node_id: str) -> None:
        nonlocal index
        indices[node_id] = index
        lowlinks[node_id] = index
        index += 1
        stack.append(node_id)
        on_stack.add(node_id)

        for neighbor_id in graph.get(node_id, set()):
            if neighbor_id not in indices:
                _strongconnect(neighbor_id)
                lowlinks[node_id] = min(lowlinks[node_id], lowlinks[neighbor_id])
            elif neighbor_id in on_stack:
                lowlinks[node_id] = min(lowlinks[node_id], indices[neighbor_id])

        if lowlinks[node_id] == indices[node_id]:
            component: Set[str] = set()
            while stack:
                member_id = stack.pop()
                on_stack.remove(member_id)
                component.add(member_id)
                if member_id == node_id:
                    break
            components.append(component)

    for node_id in graph:
        if node_id not in indices:
            _strongconnect(node_id)

    return components


def _find_cycle_path_in_component(component: Set[str], graph: Dict[str, Set[str]]) -> List[str]:
    """Return one explicit closed cycle path inside the component, with start=end."""
    if not component:
        return []

    sorted_members = sorted(component)

    for start_id in sorted_members:
        if start_id in graph.get(start_id, set()):
            return [start_id, start_id]

    def _dfs(current_id: str, start_id: str, path: List[str], seen_in_path: Set[str]) -> List[str]:
        for next_id in sorted(graph.get(current_id, set()) & component):
            if next_id == start_id and len(path) > 1:
                return path + [start_id]
            if next_id in seen_in_path:
                continue

            cycle_path = _dfs(next_id, start_id, path + [next_id], seen_in_path | {next_id})
            if cycle_path:
                return cycle_path

        return []

    for start_id in sorted_members:
        cycle_path = _dfs(start_id, start_id, [start_id], {start_id})
        if cycle_path:
            return cycle_path

    return sorted_members


def _canonical_cycle_key(cycle_nodes: List[str]) -> tuple:
    """Return a rotation-invariant key for a directed cycle."""
    if not cycle_nodes:
        return tuple()

    rotations = [tuple(cycle_nodes[index:] + cycle_nodes[:index]) for index in range(len(cycle_nodes))]
    return min(rotations)


def _find_all_simple_cycles(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """Return all simple directed cycles as closed paths where start=end."""
    unique_cycles: Dict[tuple, List[str]] = {}
    sorted_nodes = sorted(graph)

    def _dfs(start_id: str, current_id: str, path: List[str], visited: Set[str]) -> None:
        for next_id in sorted(graph.get(current_id, set())):
            if next_id == start_id:
                cycle_nodes = path[:]
                cycle_key = _canonical_cycle_key(cycle_nodes)
                unique_cycles.setdefault(cycle_key, cycle_nodes + [start_id])
                continue

            if next_id in visited:
                continue

            _dfs(start_id, next_id, path + [next_id], visited | {next_id})

    for start_id in sorted_nodes:
        _dfs(start_id, start_id, [start_id], {start_id})

    return sorted(
        unique_cycles.values(),
        key=lambda cycle_path: (len(cycle_path) - 1, tuple(cycle_path)),
    )


def _build_uses_cycle_distribution_analysis(
    uses_graph: Dict[str, Set[str]],
    entity_names: Dict[str, str],
    entity_versions: Dict[str, str],
) -> ComponentAnalysis:
    """Build an analysis of all explicit cyclic uses chains across all entities."""
    ranked = []
    counts: Dict[str, int] = {}

    def _member_sort_key(member_id: str) -> str:
        return _format_entity_name(
            entity_names.get(member_id, member_id),
            entity_versions.get(member_id, ""),
        )

    all_cycles = _find_all_simple_cycles(uses_graph)

    for index_value, cycle_path_ids in enumerate(all_cycles, 1):
        cycle_path_names = [_member_sort_key(member_id) for member_id in cycle_path_ids]
        cycle_path_text = " -> ".join(cycle_path_names)
        unique_lib_count = len(cycle_path_ids) - 1
        counts[cycle_path_text] = unique_lib_count
        ranked.append({
            "id": f"cycle_{index_value}",
            "name": cycle_path_text,
            "value": unique_lib_count,
        })

    return ComponentAnalysis(
        metric_name=USES_CYCLE_DISTRIBUTION,
        description="Döngüsel bağımlılık zincirleri",
        ranked_list=ranked,
        stats=calculate_categorical_stats(counts),
        is_categorical=True,
        show_entities=False,
        show_charts=False,
        show_statistics=False,
    )


def _analyze_nodes(nodes: List[Dict], runs_on: List[Dict], apps: List[Dict]) -> List[ComponentAnalysis]:
    """Analyze nodes by application count and hierarchy diversity."""
    # Count apps per node
    node_app_count: Dict[str, int] = {n["id"]: 0 for n in nodes}
    node_domain_values: Dict[str, Set[str]] = {n["id"]: set() for n in nodes}
    app_map = {app["id"]: app for app in apps}

    for rel in runs_on:
        node_id = rel.get("to")
        if node_id in node_app_count:
            node_app_count[node_id] += 1

        app_id = rel.get("from")
        app_data = app_map.get(app_id)
        if node_id in node_domain_values and app_data:
            css_value = _get_system_hierarchy_value(app_data, "css_name")
            if css_value != "NOT_FOUND":
                node_domain_values[node_id].add(css_value)
    
    # Enrich nodes with app_count
    enriched = []
    for node in nodes:
        enriched.append({
            "id": node["id"],
            "name": node.get("name", node["id"]),
            "app_count": node_app_count.get(node["id"], 0)
        })
    
    # Sort and calculate stats
    sorted_list = sort_entities_by_metric(enriched, "app_count", descending=True)
    values = [v for _, _, v in sorted_list]
    stats = calculate_descriptive_stats(values)
    
    ranked = [{"id": id, "name": name, "value": val} for id, name, val in sorted_list]
    
    return [ComponentAnalysis(
        metric_name=NODE_APPLICATION_COUNT,
        description="Node üzerinde çalışan uygulama sayısı",
        ranked_list=ranked,
        stats=stats
    ), ComponentAnalysis(
        metric_name=NODE_DOMAIN_HIERARCHY_DIVERSITY_COUNT,
        description="Node üzerinde bulunan farklı Alan hiyerarşi sayısı",
        ranked_list=[
            {"id": entity_id, "name": entity_name, "value": value}
            for entity_id, entity_name, value in sort_entities_by_metric([
                {
                    "id": node["id"],
                    "name": node.get("name", node["id"]),
                    "hierarchy_diversity": len(node_domain_values.get(node["id"], set())),
                }
                for node in nodes
            ], "hierarchy_diversity", descending=True)
        ],
        stats=calculate_descriptive_stats([
            len(node_domain_values.get(node["id"], set())) for node in nodes
        ])
    )]


def _analyze_apps(
    apps: List[Dict], 
    publishes_to: List[Dict], 
    subscribes_to: List[Dict],
    uses_graph: Dict[str, Set[str]],
    entity_pub_count: Dict[str, int],
    entity_sub_count: Dict[str, int],
    entity_names: Dict[str, str] = None,
    entity_versions: Dict[str, str] = None
) -> List[ComponentAnalysis]:
    """Analyze apps by pub/sub counts and total load (recursive)."""
    if entity_versions is None:
        entity_versions = {}
    # Count pub/sub per app (direct)
    app_pub_count: Dict[str, int] = {a["id"]: 0 for a in apps}
    app_sub_count: Dict[str, int] = {a["id"]: 0 for a in apps}
    
    for rel in publishes_to:
        app_id = rel.get("from")
        if app_id in app_pub_count:
            app_pub_count[app_id] += 1
    
    for rel in subscribes_to:
        app_id = rel.get("from")
        if app_id in app_sub_count:
            app_sub_count[app_id] += 1
    
    # Enrich apps with direct and total load
    enriched_pub = []
    enriched_sub = []
    enriched_total_pub = []
    enriched_total_sub = []
    
    for app in apps:
        app_id = app["id"]
        app_name = app.get("name", app_id)
        app_version = entity_versions.get(app_id, "")
        
        # Direct counts
        direct_pub = app_pub_count.get(app_id, 0)
        direct_sub = app_sub_count.get(app_id, 0)
        
        # Total load (recursive) with library breakdown
        total_pub, total_sub, pub_breakdown, sub_breakdown = _calculate_total_load(
            app_id, uses_graph, entity_pub_count, entity_sub_count, entity_names
        )
        
        enriched_pub.append({"id": app_id, "name": app_name, "version": app_version, "pub_count": direct_pub})
        enriched_sub.append({"id": app_id, "name": app_name, "version": app_version, "sub_count": direct_sub})
        enriched_total_pub.append({"id": app_id, "name": app_name, "version": app_version, "total_pub_load": total_pub, "lib_breakdown": pub_breakdown})
        enriched_total_sub.append({"id": app_id, "name": app_name, "version": app_version, "total_sub_load": total_sub, "lib_breakdown": sub_breakdown})
    
    results = []
    
    # Direct Pub analysis
    sorted_pub = sort_entities_by_metric(enriched_pub, "pub_count", descending=True)
    pub_values = [v for _, _, v in sorted_pub]
    pub_stats = calculate_descriptive_stats(pub_values)
    enriched_pub_map = {item["id"]: item.get("version", "") for item in enriched_pub}
    pub_ranked = [{"id": id, "name": name, "version": enriched_pub_map.get(id, ""), "value": val} for id, name, val in sorted_pub]
    results.append(ComponentAnalysis(
        metric_name=APP_DIRECT_PUBLISH_COUNT,
        description="Doğrudan publish sayısı",
        ranked_list=pub_ranked,
        stats=pub_stats
    ))
    
    # Direct Sub analysis
    sorted_sub = sort_entities_by_metric(enriched_sub, "sub_count", descending=True)
    sub_values = [v for _, _, v in sorted_sub]
    sub_stats = calculate_descriptive_stats(sub_values)
    enriched_sub_map = {item["id"]: item.get("version", "") for item in enriched_sub}
    sub_ranked = [{"id": id, "name": name, "version": enriched_sub_map.get(id, ""), "value": val} for id, name, val in sorted_sub]
    results.append(ComponentAnalysis(
        metric_name=APP_DIRECT_SUBSCRIBE_COUNT,
        description="Doğrudan subscribe sayısı",
        ranked_list=sub_ranked,
        stats=sub_stats
    ))
    
    # Total Pub Load analysis (recursive)
    sorted_total_pub = sort_entities_by_metric(enriched_total_pub, "total_pub_load", descending=True)
    total_pub_values = [v for _, _, v in sorted_total_pub]
    total_pub_stats = calculate_descriptive_stats(total_pub_values)
    # Build ranked list with lib_breakdown
    app_enriched_total_pub_map = {item["id"]: (item.get("lib_breakdown", []), item.get("version", "")) for item in enriched_total_pub}
    total_pub_ranked = [{"id": id, "name": name, "version": app_enriched_total_pub_map.get(id, ([], ""))[1], "value": val, "lib_breakdown": app_enriched_total_pub_map.get(id, ([], ""))[0]} for id, name, val in sorted_total_pub]
    results.append(ComponentAnalysis(
        metric_name=APP_TOTAL_PUBLISH_COUNT,
        description="Toplam publish sayısı (kullanılan kütüphaneler dahil)",
        ranked_list=total_pub_ranked,
        stats=total_pub_stats
    ))
    
    # Total Sub Load analysis (recursive)
    sorted_total_sub = sort_entities_by_metric(enriched_total_sub, "total_sub_load", descending=True)
    total_sub_values = [v for _, _, v in sorted_total_sub]
    total_sub_stats = calculate_descriptive_stats(total_sub_values)
    # Build ranked list with lib_breakdown
    app_enriched_total_sub_map = {item["id"]: (item.get("lib_breakdown", []), item.get("version", "")) for item in enriched_total_sub}
    total_sub_ranked = [{"id": id, "name": name, "version": app_enriched_total_sub_map.get(id, ([], ""))[1], "value": val, "lib_breakdown": app_enriched_total_sub_map.get(id, ([], ""))[0]} for id, name, val in sorted_total_sub]
    results.append(ComponentAnalysis(
        metric_name=APP_TOTAL_SUBSCRIBE_COUNT,
        description="Toplam subscribe sayısı (kullanılan kütüphaneler dahil)",
        ranked_list=total_sub_ranked,
        stats=total_sub_stats
    ))
    
    # Role distribution analysis
    # An application may carry multiple roles; each role is counted once
    # per app, so totals can exceed the number of applications.
    role_counts: Dict[str, int] = {}
    role_entities: Dict[str, List[str]] = {}  # role -> [app_name, ...]
    for app in apps:
        raw_roles = app.get("role", ["NOT_FOUND"])
        if isinstance(raw_roles, str):
            roles = [raw_roles]
        else:
            roles = list(raw_roles) if raw_roles else ["NOT_FOUND"]
        app_name = app.get("name", app["id"])
        app_version = entity_versions.get(app["id"], "")
        display_name = f"{app_name} ({app_version})" if app_version and app_version != "NOT_FOUND" else app_name
        for role in roles:
            role_counts[role] = role_counts.get(role, 0) + 1
            if role not in role_entities:
                role_entities[role] = []
            role_entities[role].append(display_name)
    
    role_ranked = [{"id": k, "name": k, "value": v, "entities": role_entities.get(k, [])} for k, v in sorted(role_counts.items(), key=lambda x: x[1], reverse=True)]
    role_stats = calculate_categorical_stats(role_counts)
    results.append(ComponentAnalysis(
        metric_name=APP_ROLE_DISTRIBUTION,
        description="Uygulama rol dağılımı (her rolün kaç uygulamada kullanıldığı)",
        ranked_list=role_ranked,
        stats=role_stats,
        is_categorical=True
    ))
    
    # Criticality distribution analysis
    criticality_counts: Dict[str, int] = {}
    criticality_entities: Dict[str, List[str]] = {}  # criticality -> [app_name, ...]
    for app in apps:
        raw_crit = app.get("criticality", False)
        criticality = "Kritik" if raw_crit is True else "Kritik Değil"
        app_name = app.get("name", app["id"])
        app_version = entity_versions.get(app["id"], "")
        display_name = f"{app_name} ({app_version})" if app_version and app_version != "NOT_FOUND" else app_name
        criticality_counts[criticality] = criticality_counts.get(criticality, 0) + 1
        if criticality not in criticality_entities:
            criticality_entities[criticality] = []
        criticality_entities[criticality].append(display_name)
    
    criticality_ranked = [{"id": k, "name": k, "value": v, "entities": criticality_entities.get(k, [])} for k, v in sorted(criticality_counts.items(), key=lambda x: x[1], reverse=True)]
    criticality_stats = calculate_categorical_stats(criticality_counts)
    results.append(ComponentAnalysis(
        metric_name=APP_CRITICALITY_DISTRIBUTION,
        description="Uygulama kritiklik dağılımı (her kritiklik seviyesinin kaç uygulamada kullanıldığı)",
        ranked_list=criticality_ranked,
        stats=criticality_stats,
        is_categorical=True
    ))

    results.append(_build_hierarchy_distribution_analysis(
        apps,
        "csc_name",
        APP_HIERARCHY_COMPONENT_DISTRIBUTION,
        "Uygulamaların CSC hiyerarşi dağılımı",
    ))
    results.append(_build_hierarchy_distribution_analysis(
        apps,
        "csci_name",
        APP_HIERARCHY_CONFIG_ITEM_DISTRIBUTION,
        "Uygulamaların CSCI hiyerarşi dağılımı",
    ))
    results.append(_build_hierarchy_distribution_analysis(
        apps,
        "css_name",
        APP_HIERARCHY_DOMAIN_DISTRIBUTION,
        "Uygulamaların CSS hiyerarşi dağılımı",
    ))
    results.append(_build_hierarchy_distribution_analysis(
        apps,
        "csms_name",
        APP_HIERARCHY_SYSTEM_DISTRIBUTION,
        "Uygulamaların CSMS hiyerarşi dağılımı",
    ))

    results.append(_build_group_average_analysis(
        apps,
        app_pub_count,
        "css_name",
        APP_HIERARCHY_DOMAIN_AVG_DIRECT_PUBLISH_COUNT,
        "CSS bazında uygulama başına ortalama doğrudan publish sayısı",
    ))
    results.append(_build_group_average_analysis(
        apps,
        app_sub_count,
        "css_name",
        APP_HIERARCHY_DOMAIN_AVG_DIRECT_SUBSCRIBE_COUNT,
        "CSS bazında uygulama başına ortalama doğrudan subscribe sayısı",
    ))
    results.append(_build_group_topic_variety_analysis(
        apps,
        publishes_to,
        subscribes_to,
        "css_name",
        APP_HIERARCHY_DOMAIN_TOPIC_VARIETY_COUNT,
        "CSS bazında kullanılan benzersiz topic çeşidi sayısı",
    ))
    results.append(_build_group_average_analysis(
        apps,
        app_pub_count,
        "csci_name",
        APP_HIERARCHY_CONFIG_ITEM_AVG_DIRECT_PUBLISH_COUNT,
        "CSCI bazında uygulama başına ortalama doğrudan publish sayısı",
    ))
    results.append(_build_group_average_analysis(
        apps,
        app_sub_count,
        "csci_name",
        APP_HIERARCHY_CONFIG_ITEM_AVG_DIRECT_SUBSCRIBE_COUNT,
        "CSCI bazında uygulama başına ortalama doğrudan subscribe sayısı",
    ))
    results.append(_build_group_topic_variety_analysis(
        apps,
        publishes_to,
        subscribes_to,
        "csci_name",
        APP_HIERARCHY_CONFIG_ITEM_TOPIC_VARIETY_COUNT,
        "CSCI bazında kullanılan benzersiz topic çeşidi sayısı",
    ))
    
    return results


def _analyze_libs(
    libs: List[Dict],
    uses: List[Dict],
    publishes_to: List[Dict],
    subscribes_to: List[Dict],
    uses_graph: Dict[str, Set[str]],
    entity_pub_count: Dict[str, int],
    entity_sub_count: Dict[str, int],
    entity_names: Dict[str, str] = None,
    entity_versions: Dict[str, str] = None
) -> List[ComponentAnalysis]:
    """Analyze libs by usage count, pub/sub counts, and total load (recursive)."""
    if entity_versions is None:
        entity_versions = {}
    lib_ids = {lib["id"] for lib in libs}
    
    # Count usage (how many entities use this lib)
    lib_usage_count: Dict[str, int] = {lib["id"]: 0 for lib in libs}
    for rel in uses:
        lib_id = rel.get("to")
        if lib_id in lib_usage_count:
            lib_usage_count[lib_id] += 1
    
    # Count pub/sub for libs (direct)
    lib_pub_count: Dict[str, int] = {lib["id"]: 0 for lib in libs}
    lib_sub_count: Dict[str, int] = {lib["id"]: 0 for lib in libs}
    
    for rel in publishes_to:
        entity_id = rel.get("from")
        if entity_id in lib_ids:
            lib_pub_count[entity_id] += 1
    
    for rel in subscribes_to:
        entity_id = rel.get("from")
        if entity_id in lib_ids:
            lib_sub_count[entity_id] += 1
    
    # Enrich libs
    enriched_usage = []
    enriched_pub = []
    enriched_sub = []
    enriched_total_pub = []
    enriched_total_sub = []
    
    for lib in libs:
        lib_id = lib["id"]
        lib_name = lib.get("name", lib_id)
        lib_version = entity_versions.get(lib_id, "")
        
        # Direct counts
        direct_pub = lib_pub_count.get(lib_id, 0)
        direct_sub = lib_sub_count.get(lib_id, 0)
        
        # Total load (recursive) with library breakdown
        total_pub, total_sub, pub_breakdown, sub_breakdown = _calculate_total_load(
            lib_id, uses_graph, entity_pub_count, entity_sub_count, entity_names
        )
        
        enriched_usage.append({"id": lib_id, "name": lib_name, "version": lib_version, "usage_count": lib_usage_count.get(lib_id, 0)})
        enriched_pub.append({"id": lib_id, "name": lib_name, "version": lib_version, "pub_count": direct_pub})
        enriched_sub.append({"id": lib_id, "name": lib_name, "version": lib_version, "sub_count": direct_sub})
        enriched_total_pub.append({"id": lib_id, "name": lib_name, "version": lib_version, "total_pub_load": total_pub, "lib_breakdown": pub_breakdown})
        enriched_total_sub.append({"id": lib_id, "name": lib_name, "version": lib_version, "total_sub_load": total_sub, "lib_breakdown": sub_breakdown})
    
    results = []
    
    # Usage analysis
    sorted_usage = sort_entities_by_metric(enriched_usage, "usage_count", descending=True)
    usage_values = [v for _, _, v in sorted_usage]
    usage_stats = calculate_descriptive_stats(usage_values)
    enriched_usage_map = {item["id"]: item.get("version", "") for item in enriched_usage}
    usage_ranked = [{"id": id, "name": name, "version": enriched_usage_map.get(id, ""), "value": val} for id, name, val in sorted_usage]
    results.append(ComponentAnalysis(
        metric_name=LIB_APPLICATION_USAGE_COUNT,
        description="Bu kütüphaneyi kullanan uygulama sayısı",
        ranked_list=usage_ranked,
        stats=usage_stats
    ))
    
    # Pub analysis
    sorted_pub = sort_entities_by_metric(enriched_pub, "pub_count", descending=True)
    pub_values = [v for _, _, v in sorted_pub]
    pub_stats = calculate_descriptive_stats(pub_values)
    enriched_pub_map = {item["id"]: item.get("version", "") for item in enriched_pub}
    pub_ranked = [{"id": id, "name": name, "version": enriched_pub_map.get(id, ""), "value": val} for id, name, val in sorted_pub]
    results.append(ComponentAnalysis(
        metric_name=LIB_DIRECT_PUBLISH_COUNT,
        description="Doğrudan publish sayısı",
        ranked_list=pub_ranked,
        stats=pub_stats
    ))
    
    # Sub analysis
    sorted_sub = sort_entities_by_metric(enriched_sub, "sub_count", descending=True)
    sub_values = [v for _, _, v in sorted_sub]
    sub_stats = calculate_descriptive_stats(sub_values)
    enriched_sub_map = {item["id"]: item.get("version", "") for item in enriched_sub}
    sub_ranked = [{"id": id, "name": name, "version": enriched_sub_map.get(id, ""), "value": val} for id, name, val in sorted_sub]
    results.append(ComponentAnalysis(
        metric_name=LIB_DIRECT_SUBSCRIBE_COUNT,
        description="Doğrudan subscribe sayısı",
        ranked_list=sub_ranked,
        stats=sub_stats
    ))
    
    # Total Pub Load analysis (recursive)
    sorted_total_pub = sort_entities_by_metric(enriched_total_pub, "total_pub_load", descending=True)
    total_pub_values = [v for _, _, v in sorted_total_pub]
    total_pub_stats = calculate_descriptive_stats(total_pub_values)
    # Build ranked list with lib_breakdown
    lib_enriched_total_pub_map = {item["id"]: (item.get("lib_breakdown", []), item.get("version", "")) for item in enriched_total_pub}
    total_pub_ranked = [{"id": id, "name": name, "version": lib_enriched_total_pub_map.get(id, ([], ""))[1], "value": val, "lib_breakdown": lib_enriched_total_pub_map.get(id, ([], ""))[0]} for id, name, val in sorted_total_pub]
    results.append(ComponentAnalysis(
        metric_name=LIB_TOTAL_PUBLISH_COUNT,
        description="Toplam publish sayısı (kullanılan kütüphaneler dahil)",
        ranked_list=total_pub_ranked,
        stats=total_pub_stats
    ))
    
    # Total Sub Load analysis (recursive)
    sorted_total_sub = sort_entities_by_metric(enriched_total_sub, "total_sub_load", descending=True)
    total_sub_values = [v for _, _, v in sorted_total_sub]
    total_sub_stats = calculate_descriptive_stats(total_sub_values)
    # Build ranked list with lib_breakdown
    lib_enriched_total_sub_map = {item["id"]: (item.get("lib_breakdown", []), item.get("version", "")) for item in enriched_total_sub}
    total_sub_ranked = [{"id": id, "name": name, "version": lib_enriched_total_sub_map.get(id, ([], ""))[1], "value": val, "lib_breakdown": lib_enriched_total_sub_map.get(id, ([], ""))[0]} for id, name, val in sorted_total_sub]
    results.append(ComponentAnalysis(
        metric_name=LIB_TOTAL_SUBSCRIBE_COUNT,
        description="Toplam subscribe sayısı (kullanılan kütüphaneler dahil)",
        ranked_list=total_sub_ranked,
        stats=total_sub_stats
    ))

    results.append(_build_hierarchy_distribution_analysis(
        libs,
        "csci_name",
        LIB_HIERARCHY_CONFIG_ITEM_DISTRIBUTION,
        "Kütüphanelerin CSCI hiyerarşi dağılımı",
    ))
    results.append(_build_hierarchy_distribution_analysis(
        libs,
        "css_name",
        LIB_HIERARCHY_DOMAIN_DISTRIBUTION,
        "Kütüphanelerin CSS hiyerarşi dağılımı",
    ))
    results.append(_build_hierarchy_completeness_analysis(
        libs,
        LIB_HIERARCHY_COMPLETENESS_PERCENT,
        "Kütüphane bazında system hierarchy tamlık oranı (%)",
        entity_versions,
    ))
    
    return results


def _analyze_topics(
    topics: List[Dict],
    publishes_to: List[Dict],
    subscribes_to: List[Dict],
    *,
    dds_mask: bool = True,
) -> List[ComponentAnalysis]:
    """Analyze topics by size, pub/sub counts, and QoS properties."""
    # Count pub/sub per topic
    topic_pub_count: Dict[str, int] = {t["id"]: 0 for t in topics}
    topic_sub_count: Dict[str, int] = {t["id"]: 0 for t in topics}
    
    for rel in publishes_to:
        topic_id = rel.get("to")
        if topic_id in topic_pub_count:
            topic_pub_count[topic_id] += 1
    
    for rel in subscribes_to:
        topic_id = rel.get("to")
        if topic_id in topic_sub_count:
            topic_sub_count[topic_id] += 1
    
    # Enrich topics
    enriched_size = []
    enriched_pub = []
    enriched_sub = []
    
    # QoS distribution counters and entity maps
    durability_counts: Dict[str, int] = {}
    durability_entities: Dict[str, List[str]] = {}  # durability -> [topic_name, ...]
    reliability_counts: Dict[str, int] = {}
    reliability_entities: Dict[str, List[str]] = {}  # reliability -> [topic_name, ...]
    transport_priority_entities: Dict[str, List[str]] = {}  # tp_value -> [topic_name, ...]
    
    # QoS enrichment lists
    enriched_durability = []
    enriched_reliability = []
    enriched_transport_priority = []

    # Build QoS name mapping based on dds_mask
    from common.runtime_config import get_runtime_config
    raw_mappings = get_runtime_config().aggregator.qos_mappings  # custom -> DDS
    qos_name_map: Dict[str, Dict[str, str]] = {}
    for dim_key in ("durability", "reliability", "transport_priority"):
        dim_map = raw_mappings.get(dim_key, {})
        if dds_mask:
            # custom -> DDS  (ensure DDS names are displayed)
            qos_name_map[dim_key] = {str(k): str(v) for k, v in dim_map.items()}
        else:
            # DDS -> custom  (ensure custom names are displayed)
            qos_name_map[dim_key] = {str(v): str(k) for k, v in dim_map.items()}

    def _map_qos(dim_key: str, value: str) -> str:
        return qos_name_map.get(dim_key, {}).get(value, value)
    
    for topic in topics:
        topic_id = topic["id"]
        topic_name = topic.get("name", topic_id)
        size = topic.get("size", 0)
        if size is None or size < 0:
            size = 0
        
        enriched_size.append({"id": topic_id, "name": topic_name, "size": size})
        enriched_pub.append({"id": topic_id, "name": topic_name, "publisher_count": topic_pub_count.get(topic_id, 0)})
        enriched_sub.append({"id": topic_id, "name": topic_name, "subscriber_count": topic_sub_count.get(topic_id, 0)})
        
        # Extract QoS properties and map names according to dds_mask
        qos = topic.get("qos", {})
        durability = _map_qos("durability", qos.get("durability", "NOT_FOUND"))
        reliability = _map_qos("reliability", qos.get("reliability", "NOT_FOUND"))
        transport_priority = _map_qos("transport_priority", qos.get("transport_priority", "NOT_FOUND"))
        
        # Count durability distribution and collect entities
        durability_counts[durability] = durability_counts.get(durability, 0) + 1
        if durability not in durability_entities:
            durability_entities[durability] = []
        durability_entities[durability].append(topic_name)
        enriched_durability.append({"id": topic_id, "name": topic_name, "durability": durability})
        
        # Count reliability distribution and collect entities
        reliability_counts[reliability] = reliability_counts.get(reliability, 0) + 1
        if reliability not in reliability_entities:
            reliability_entities[reliability] = []
        reliability_entities[reliability].append(topic_name)
        enriched_reliability.append({"id": topic_id, "name": topic_name, "reliability": reliability})
        
        # Transport priority - collect entities (categorical only)
        tp_str = str(transport_priority)
        if tp_str not in transport_priority_entities:
            transport_priority_entities[tp_str] = []
        transport_priority_entities[tp_str].append(topic_name)
        enriched_transport_priority.append({"id": topic_id, "name": topic_name, "transport_priority": transport_priority})
    
    results = []
    
    # Size analysis
    sorted_size = sort_entities_by_metric(enriched_size, "size", descending=True)
    size_values = [v for _, _, v in sorted_size]
    size_stats = calculate_descriptive_stats(size_values)
    size_ranked = [{"id": id, "name": name, "value": val} for id, name, val in sorted_size]
    results.append(ComponentAnalysis(
        metric_name=TOPIC_SIZE_BYTES,
        description="İnfogram boyutu (byte)",
        ranked_list=size_ranked,
        stats=size_stats
    ))
    
    # Publisher count analysis
    sorted_pub = sort_entities_by_metric(enriched_pub, "publisher_count", descending=True)
    pub_values = [v for _, _, v in sorted_pub]
    pub_stats = calculate_descriptive_stats(pub_values)
    pub_ranked = [{"id": id, "name": name, "value": val} for id, name, val in sorted_pub]
    results.append(ComponentAnalysis(
        metric_name=TOPIC_PUBLISHER_APPLICATION_COUNT,
        description="Bu topica publish yapan uygulama sayısı",
        ranked_list=pub_ranked,
        stats=pub_stats
    ))
    
    # Subscriber count analysis
    sorted_sub = sort_entities_by_metric(enriched_sub, "subscriber_count", descending=True)
    sub_values = [v for _, _, v in sorted_sub]
    sub_stats = calculate_descriptive_stats(sub_values)
    sub_ranked = [{"id": id, "name": name, "value": val} for id, name, val in sorted_sub]
    results.append(ComponentAnalysis(
        metric_name=TOPIC_SUBSCRIBER_APPLICATION_COUNT,
        description="Bu topica subscribe olan uygulama sayısı",
        ranked_list=sub_ranked,
        stats=sub_stats
    ))
    
    # QoS Durability distribution analysis
    durability_ranked = [{"id": k, "name": k, "value": v, "entities": durability_entities.get(k, [])} for k, v in sorted(durability_counts.items(), key=lambda x: x[1], reverse=True)]
    durability_stats = calculate_categorical_stats(durability_counts)
    results.append(ComponentAnalysis(
        metric_name=TOPIC_QOS_DURABILITY_DISTRIBUTION,
        description="QoS Durability dağılımı (her değerin kaç topicda kullanıldığı)",
        ranked_list=durability_ranked,
        stats=durability_stats,
        is_categorical=True
    ))
    
    # QoS Reliability distribution analysis
    reliability_ranked = [{"id": k, "name": k, "value": v, "entities": reliability_entities.get(k, [])} for k, v in sorted(reliability_counts.items(), key=lambda x: x[1], reverse=True)]
    reliability_stats = calculate_categorical_stats(reliability_counts)
    results.append(ComponentAnalysis(
        metric_name=TOPIC_QOS_RELIABILITY_DISTRIBUTION,
        description="QoS Reliability dağılımı (her değerin kaç topicda kullanıldığı)",
        ranked_list=reliability_ranked,
        stats=reliability_stats,
        is_categorical=True
    ))
    
    # QoS Transport Priority distribution analysis (categorical)
    transport_priority_counts: Dict[str, int] = {}
    for item in enriched_transport_priority:
        tp_val = str(item["transport_priority"])
        transport_priority_counts[tp_val] = transport_priority_counts.get(tp_val, 0) + 1
    
    tp_dist_ranked = [{"id": k, "name": k, "value": v, "entities": transport_priority_entities.get(k, [])} for k, v in sorted(transport_priority_counts.items(), key=lambda x: x[1], reverse=True)]
    tp_dist_stats = calculate_categorical_stats(transport_priority_counts)
    results.append(ComponentAnalysis(
        metric_name=TOPIC_QOS_TRANSPORT_PRIORITY_DISTRIBUTION,
        description="QoS Transport Priority dağılımı (her değerin kaç topicda kullanıldığı)",
        ranked_list=tp_dist_ranked,
        stats=tp_dist_stats,
        is_categorical=True
    ))
    
    return results


def _analyze_structural(data: Dict[str, Any]) -> List[ComponentAnalysis]:
    """Extract top structural scores from aggregated entity data.
    
    Reads the ``structural_analysis.scores.total_score`` field that the
    structural module attaches to each entity during aggregation.
    Returns one ComponentAnalysis per entity type (apps, topics, nodes, libs)
    with entities ranked by total_score descending, limited to top 10.
    """
    entity_groups = [
        (STRUCTURAL_TOP_APPS, "En yüksek yapısal anomali skoruna sahip 10 uygulama", data.get("applications", [])),
        (STRUCTURAL_TOP_TOPICS, "En yüksek yapısal anomali skoruna sahip 10 topic", data.get("topics", [])),
        (STRUCTURAL_TOP_NODES, "En yüksek yapısal anomali skoruna sahip 10 düğüm", data.get("nodes", [])),
        (STRUCTURAL_TOP_LIBS, "En yüksek yapısal anomali skoruna sahip 10 kütüphane", data.get("libraries", [])),
    ]

    results: List[ComponentAnalysis] = []
    for metric_name, description, entities in entity_groups:
        ranked_items: List[Dict[str, Any]] = []
        all_scores: List[float] = []

        for entity in entities:
            structural_data = entity.get("structural_analysis") or {}
            score_data = structural_data.get("scores") or {}
            total_score = score_data.get("total_score")
            if total_score is None:
                continue

            score_val = float(total_score)
            all_scores.append(score_val)
            ranked_items.append({
                "id": entity.get("id", ""),
                "name": entity.get("name", entity.get("id", "?")),
                "version": entity.get("version", ""),
                "value": score_val,
            })

        ranked_items.sort(key=lambda x: x["value"], reverse=True)
        top_10 = ranked_items[:10]
        top_10_values = [item["value"] for item in top_10]
        stats = calculate_descriptive_stats(top_10_values)

        results.append(ComponentAnalysis(
            metric_name=metric_name,
            description=description,
            ranked_list=top_10,
            stats=stats,
            show_charts=False,
            show_statistics=False,
        ))

    return results
