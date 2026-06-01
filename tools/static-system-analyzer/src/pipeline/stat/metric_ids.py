"""Canonical metric identifiers for stat analysis outputs.

Naming convention:
- prefix with the report section: node_, app_, lib_, topic_
- continue with the measurement domain
- end with the concrete measure or aggregation

This keeps metric IDs unique across sections and makes reporter logic rely on
stable identifiers rather than substring matching.
"""

NODE_APPLICATION_COUNT = "node_application_count"
NODE_DOMAIN_HIERARCHY_DIVERSITY_COUNT = "node_domain_hierarchy_diversity_count"

APP_DIRECT_PUBLISH_COUNT = "app_direct_publish_count"
APP_DIRECT_SUBSCRIBE_COUNT = "app_direct_subscribe_count"
APP_TOTAL_PUBLISH_COUNT = "app_total_publish_count"
APP_TOTAL_SUBSCRIBE_COUNT = "app_total_subscribe_count"
APP_ROLE_DISTRIBUTION = "app_role_distribution"
APP_CRITICALITY_DISTRIBUTION = "app_criticality_distribution"
APP_HIERARCHY_COMPONENT_DISTRIBUTION = "app_hierarchy_component_distribution"
APP_HIERARCHY_CONFIG_ITEM_DISTRIBUTION = "app_hierarchy_config_item_distribution"
APP_HIERARCHY_DOMAIN_DISTRIBUTION = "app_hierarchy_domain_distribution"
APP_HIERARCHY_SYSTEM_DISTRIBUTION = "app_hierarchy_system_distribution"
APP_HIERARCHY_DOMAIN_AVG_DIRECT_PUBLISH_COUNT = "app_hierarchy_domain_avg_direct_publish_count"
APP_HIERARCHY_DOMAIN_AVG_DIRECT_SUBSCRIBE_COUNT = "app_hierarchy_domain_avg_direct_subscribe_count"
APP_HIERARCHY_DOMAIN_TOPIC_VARIETY_COUNT = "app_hierarchy_domain_topic_variety_count"
APP_HIERARCHY_CONFIG_ITEM_AVG_DIRECT_PUBLISH_COUNT = "app_hierarchy_config_item_avg_direct_publish_count"
APP_HIERARCHY_CONFIG_ITEM_AVG_DIRECT_SUBSCRIBE_COUNT = "app_hierarchy_config_item_avg_direct_subscribe_count"
APP_HIERARCHY_CONFIG_ITEM_TOPIC_VARIETY_COUNT = "app_hierarchy_config_item_topic_variety_count"

LIB_APPLICATION_USAGE_COUNT = "lib_application_usage_count"
LIB_DIRECT_PUBLISH_COUNT = "lib_direct_publish_count"
LIB_DIRECT_SUBSCRIBE_COUNT = "lib_direct_subscribe_count"
LIB_TOTAL_PUBLISH_COUNT = "lib_total_publish_count"
LIB_TOTAL_SUBSCRIBE_COUNT = "lib_total_subscribe_count"
LIB_HIERARCHY_CONFIG_ITEM_DISTRIBUTION = "lib_hierarchy_config_item_distribution"
LIB_HIERARCHY_DOMAIN_DISTRIBUTION = "lib_hierarchy_domain_distribution"
LIB_HIERARCHY_COMPLETENESS_PERCENT = "lib_hierarchy_completeness_percent"

TOPIC_SIZE_BYTES = "topic_size_bytes"
TOPIC_PUBLISHER_APPLICATION_COUNT = "topic_publisher_application_count"
TOPIC_SUBSCRIBER_APPLICATION_COUNT = "topic_subscriber_application_count"
TOPIC_QOS_DURABILITY_DISTRIBUTION = "topic_qos_durability_distribution"
TOPIC_QOS_RELIABILITY_DISTRIBUTION = "topic_qos_reliability_distribution"
TOPIC_QOS_TRANSPORT_PRIORITY_DISTRIBUTION = "topic_qos_transport_priority_distribution"

STRUCTURAL_TOP_APPS = "structural_top_apps"
STRUCTURAL_TOP_TOPICS = "structural_top_topics"
STRUCTURAL_TOP_NODES = "structural_top_nodes"
STRUCTURAL_TOP_LIBS = "structural_top_libs"

USES_CYCLE_DISTRIBUTION = "uses_cycle_distribution"

TOTAL_METRIC_IDS = {
    APP_TOTAL_PUBLISH_COUNT,
    APP_TOTAL_SUBSCRIBE_COUNT,
    LIB_TOTAL_PUBLISH_COUNT,
    LIB_TOTAL_SUBSCRIBE_COUNT,
}


def is_total_metric(metric_id: str) -> bool:
    """Return whether the metric includes recursive library load breakdowns."""
    return metric_id in TOTAL_METRIC_IDS