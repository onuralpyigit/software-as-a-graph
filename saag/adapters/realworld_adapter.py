"""
Real-World Software Architecture Adapters for Software-as-a-Graph (SaG)

Provides adapters and parsers for converting real-world open-source software topologies
(e.g., ROS 2 Autoware.universe autonomous driving architecture, Cloud-Native Pub-Sub microservice stacks)
into canonical SaG typed multigraphs.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional


#: Maps the flat relation-list `type` string used when authoring topologies in this module to the
#: canonical relation-keyed dict key expected by MemoryRepository/Neo4jRepository (`save_graph`)
#: and every other dataset in data/scenarios/.
_RELATION_TYPE_TO_KEY = {
    "PUBLISHES_TO": "publishes_to",
    "SUBSCRIBES_TO": "subscribes_to",
    "ROUTES": "routes",
    "RUNS_ON": "runs_on",
    "CONNECTS_TO": "connects_to",
    "USES": "uses",
}


def _to_canonical_relationships(flat_relationships: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group a flat `[{"source", "target", "type"}, ...]` list into the canonical
    `{"publishes_to": [{"from", "to", "weight"}], ...}` shape.

    Raises on an unrecognized `type` rather than silently misrouting the edge.
    """
    canonical: Dict[str, List[Dict[str, Any]]] = {key: [] for key in _RELATION_TYPE_TO_KEY.values()}
    for rel in flat_relationships:
        raw_type = rel["type"]
        key = _RELATION_TYPE_TO_KEY.get(raw_type)
        if key is None:
            raise ValueError(f"Unrecognized relationship type {raw_type!r} in real-world topology")
        canonical[key].append({
            "from": rel["source"],
            "to": rel["target"],
            "weight": rel.get("weight", 1.0),
        })
    return canonical


class RealWorldAdapter:
    """Adapter for importing real-world distributed pub-sub software system architectures into SaG multigraph format."""

    @staticmethod
    def create_autoware_ros2_topology() -> Dict[str, Any]:
        """
        Creates an authentic ROS 2 pub-sub architecture graph for Autoware.universe,
        the open-source autonomous driving software platform.

        Topology overview:
        - 32 Applications across Perception, Sensing, Localization, Planning, Control, Vehicle Interface, and System
        - 24 Topics with explicit DDS QoS profiles (Reliable/BestEffort, TransientLocal/Volatile, Priority)
        - 3 Brokers (DDS Middleware instances: Eclipse CycloneDDS, eProsima FastDDS, Zenoh Router)
        - 6 Deployment Nodes (ECU-Main-Brain, ECU-Perception-GPU, ECU-Sensing-FPGA, ECU-Vehicle-Actuation, ECU-Teleop, ECU-Gateway)
        - 10 Shared C++/ROS 2 Libraries (autoware_universe_utils, tier4_autoware_utils, motion_utils, rclcpp, etc.)
        - Realistic SonarQube code quality metrics (cm_*) for applications and libraries
        """
        nodes = [
            {"id": "N0", "name": "ECU-Main-Brain", "type": "Node", "spec": "AMD EPYC / 64 Core"},
            {"id": "N1", "name": "ECU-Perception-GPU", "type": "Node", "spec": "NVIDIA Orin AGX 64GB"},
            {"id": "N2", "name": "ECU-Sensing-FPGA", "type": "Node", "spec": "Xilinx Zynq UltraScale+"},
            {"id": "N3", "name": "ECU-Vehicle-Actuation", "type": "Node", "spec": "Infineon Aurix TC397 (Safety MCU)"},
            {"id": "N4", "name": "ECU-Gateway", "type": "Node", "spec": "ARM Cortex-A72 Gateway"},
            {"id": "N5", "name": "ECU-Teleop-HMI", "type": "Node", "spec": "Intel NUC / Cockpit Display"}
        ]

        brokers = [
            {"id": "B0", "name": "cyclone-dds-router", "type": "Broker", "protocol": "DDS-RTPS"},
            {"id": "B1", "name": "fast-dds-discovery", "type": "Broker", "protocol": "DDS-RTPS"},
            {"id": "B2", "name": "zenoh-bridge-router", "type": "Broker", "protocol": "Zenoh"}
        ]

        topics = [
            {"id": "T0", "name": "/sensing/lidar/top/pointcloud", "size": 8192, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "CRITICAL"}, "frequency": 10.0, "criticality": "HIGH"},
            {"id": "T1", "name": "/sensing/camera/front/image_raw", "size": 4096, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "HIGH"}, "frequency": 30.0, "criticality": "HIGH"},
            {"id": "T2", "name": "/sensing/gnss/pose", "size": 512, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "CRITICAL"}, "frequency": 50.0, "criticality": "HIGH"},
            {"id": "T3", "name": "/sensing/imu/tamagawa/data", "size": 256, "qos": {"durability": "VOLATILE", "reliability": "RELIABLE", "transport_priority": "CRITICAL"}, "frequency": 100.0, "criticality": "HIGH"},
            {"id": "T4", "name": "/perception/object_recognition/objects", "size": 2048, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "CRITICAL"}, "frequency": 10.0, "criticality": "HIGH"},
            {"id": "T5", "name": "/perception/object_recognition/tracking/objects", "size": 2048, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "CRITICAL"}, "frequency": 10.0, "criticality": "HIGH"},
            {"id": "T6", "name": "/perception/occupancy_grid_map/map", "size": 16384, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 5.0, "criticality": "HIGH"},
            {"id": "T7", "name": "/localization/kinematic_state", "size": 1024, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "CRITICAL"}, "frequency": 50.0, "criticality": "HIGH"},
            {"id": "T8", "name": "/localization/pose_estimator/pose", "size": 512, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "CRITICAL"}, "frequency": 50.0, "criticality": "HIGH"},
            {"id": "T9", "name": "/map/vector_map", "size": 32768, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "MEDIUM"}, "frequency": 0.1, "criticality": "HIGH"},
            {"id": "T10", "name": "/planning/mission_planning/route", "size": 2048, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "CRITICAL"}, "frequency": 1.0, "criticality": "HIGH"},
            {"id": "T11", "name": "/planning/scenario_planning/trajectory", "size": 4096, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "CRITICAL"}, "frequency": 10.0, "criticality": "HIGH"},
            {"id": "T12", "name": "/control/command/control_cmd", "size": 256, "qos": {"durability": "VOLATILE", "reliability": "RELIABLE", "transport_priority": "HIGHEST"}, "frequency": 30.0, "criticality": "HIGH"},
            {"id": "T13", "name": "/control/command/emergency_cmd", "size": 128, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "HIGHEST"}, "frequency": 100.0, "criticality": "HIGH"},
            {"id": "T14", "name": "/vehicle/status/velocity_status", "size": 256, "qos": {"durability": "VOLATILE", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 50.0, "criticality": "HIGH"},
            {"id": "T15", "name": "/vehicle/status/steering_status", "size": 256, "qos": {"durability": "VOLATILE", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 50.0, "criticality": "HIGH"},
            {"id": "T16", "name": "/tf", "size": 1024, "qos": {"durability": "VOLATILE", "reliability": "RELIABLE", "transport_priority": "CRITICAL"}, "frequency": 100.0, "criticality": "HIGH"},
            {"id": "T17", "name": "/tf_static", "size": 2048, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 0.05, "criticality": "HIGH"},
            {"id": "T18", "name": "/system/fail_safe/mrm_state", "size": 128, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "HIGHEST"}, "frequency": 10.0, "criticality": "HIGH"},
            {"id": "T19", "name": "/system/emergency/hazard_status", "size": 256, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "HIGHEST"}, "frequency": 10.0, "criticality": "HIGH"},
            {"id": "T20", "name": "/teleop/remote_command", "size": 512, "qos": {"durability": "VOLATILE", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 20.0, "criticality": "HIGH"},
            {"id": "T21", "name": "/diagnostics", "size": 1024, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "LOW"}, "frequency": 1.0, "criticality": "LOW"},
            {"id": "T22", "name": "/api/autoware/get/status", "size": 512, "qos": {"durability": "VOLATILE", "reliability": "RELIABLE", "transport_priority": "MEDIUM"}, "frequency": 2.0, "criticality": "MEDIUM"},
            {"id": "T23", "name": "/sensing/radar/front/objects", "size": 1024, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "HIGH"}, "frequency": 20.0, "criticality": "HIGH"}
        ]

        libraries = [
            {"id": "L0", "name": "autoware_universe_utils", "version": "1.0.0", "description": "Core algorithm & math utilities"},
            {"id": "L1", "name": "tier4_autoware_utils", "version": "1.2.0", "description": "Tier4 specialized ROS 2 helpers"},
            {"id": "L2", "name": "motion_utils", "version": "0.9.1", "description": "Trajectory generation and path smoothing math"},
            {"id": "L3", "name": "signal_processing", "version": "0.5.0", "description": "Pointcloud and sensor filtering library"},
            {"id": "L4", "name": "lanelet2_extension", "version": "1.4.0", "description": "HD Vector map query & spatial indexing library"},
            {"id": "L5", "name": "vehicle_info_util", "version": "1.1.0", "description": "Vehicle dimensions & kinematics model library"},
            {"id": "L6", "name": "sensor_pointcloud_filter", "version": "0.8.2", "description": "CUDA/PCL accelerated pointcloud filter"},
            {"id": "L7", "name": "autoware_health_checker", "version": "2.0.0", "description": "System diagnostic monitoring library"},
            {"id": "L8", "name": "rclcpp_core", "version": "16.0.1", "description": "ROS 2 C++ client library runtime"},
            {"id": "L9", "name": "tf2_ros_utils", "version": "0.25.0", "description": "Coordinate frame transform tree client"}
        ]

        applications = [
            {"id": "A0", "name": "velodyne_node_container", "app_type": "sensor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 2840, "total_classes": 14, "total_methods": 95}, "complexity": {"avg_wmc": 18.4}, "cohesion": {"avg_lcom": 18.2}, "coupling": {"avg_cbo": 9.2, "avg_rfc": 28.5}}},
            {"id": "A1", "name": "pointcloud_preprocessor", "app_type": "sensor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 3410, "total_classes": 18, "total_methods": 112}, "complexity": {"avg_wmc": 21.0}, "cohesion": {"avg_lcom": 22.1}, "coupling": {"avg_cbo": 11.5, "avg_rfc": 34.0}}},
            {"id": "A2", "name": "camera_driver_node", "app_type": "sensor", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 1450, "total_classes": 8, "total_methods": 45}, "complexity": {"avg_wmc": 12.0}, "cohesion": {"avg_lcom": 10.5}, "coupling": {"avg_cbo": 6.1, "avg_rfc": 18.2}}},
            {"id": "A3", "name": "gnss_poser_node", "app_type": "sensor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 1120, "total_classes": 6, "total_methods": 38}, "complexity": {"avg_wmc": 9.5}, "cohesion": {"avg_lcom": 8.1}, "coupling": {"avg_cbo": 4.8, "avg_rfc": 14.5}}},
            {"id": "A4", "name": "imu_corrector_node", "app_type": "sensor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 890, "total_classes": 5, "total_methods": 29}, "complexity": {"avg_wmc": 7.8}, "cohesion": {"avg_lcom": 6.0}, "coupling": {"avg_cbo": 3.9, "avg_rfc": 11.2}}},
            {"id": "A5", "name": "radar_tracks_msgs_converter", "app_type": "sensor", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 940, "total_classes": 6, "total_methods": 32}, "complexity": {"avg_wmc": 8.1}, "cohesion": {"avg_lcom": 7.2}, "coupling": {"avg_cbo": 4.1, "avg_rfc": 12.0}}},

            {"id": "A6", "name": "lidar_centerpoint_node", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 6800, "total_classes": 32, "total_methods": 210}, "complexity": {"avg_wmc": 32.5}, "cohesion": {"avg_lcom": 35.8}, "coupling": {"avg_cbo": 16.4, "avg_rfc": 52.1}}},
            {"id": "A7", "name": "multi_object_tracker", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 5900, "total_classes": 28, "total_methods": 185}, "complexity": {"avg_wmc": 28.1}, "cohesion": {"avg_lcom": 31.0}, "coupling": {"avg_cbo": 14.2, "avg_rfc": 46.0}}},
            {"id": "A8", "name": "detected_object_validation", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 2100, "total_classes": 11, "total_methods": 68}, "complexity": {"avg_wmc": 14.2}, "cohesion": {"avg_lcom": 12.0}, "coupling": {"avg_cbo": 7.8, "avg_rfc": 22.4}}},
            {"id": "A9", "name": "occupancy_grid_map_node", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 3800, "total_classes": 19, "total_methods": 120}, "complexity": {"avg_wmc": 19.8}, "cohesion": {"avg_lcom": 19.5}, "coupling": {"avg_cbo": 10.1, "avg_rfc": 31.0}}},

            {"id": "A10", "name": "ndt_scan_matcher", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 4600, "total_classes": 22, "total_methods": 145}, "complexity": {"avg_wmc": 25.4}, "cohesion": {"avg_lcom": 26.2}, "coupling": {"avg_cbo": 13.0, "avg_rfc": 41.5}}},
            {"id": "A11", "name": "ekf_localizer", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 3200, "total_classes": 15, "total_methods": 98}, "complexity": {"avg_wmc": 16.5}, "cohesion": {"avg_lcom": 15.1}, "coupling": {"avg_cbo": 8.9, "avg_rfc": 29.0}}},
            {"id": "A12", "name": "stop_filter_node", "app_type": "processor", "criticality": "LOW", "hotstandby": False, "code_metrics": {"size": {"total_loc": 650, "total_classes": 4, "total_methods": 21}, "complexity": {"avg_wmc": 5.4}, "cohesion": {"avg_lcom": 4.1}, "coupling": {"avg_cbo": 2.9, "avg_rfc": 9.1}}},

            {"id": "A13", "name": "lanelet2_map_loader", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 2900, "total_classes": 14, "total_methods": 88}, "complexity": {"avg_wmc": 15.0}, "cohesion": {"avg_lcom": 14.2}, "coupling": {"avg_cbo": 7.5, "avg_rfc": 24.0}}},
            {"id": "A14", "name": "mission_planner", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 4100, "total_classes": 20, "total_methods": 132}, "complexity": {"avg_wmc": 22.0}, "cohesion": {"avg_lcom": 23.4}, "coupling": {"avg_cbo": 11.8, "avg_rfc": 36.2}}},
            {"id": "A15", "name": "behavior_path_planner", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 9500, "total_classes": 45, "total_methods": 290}, "complexity": {"avg_wmc": 41.2}, "cohesion": {"avg_lcom": 44.0}, "coupling": {"avg_cbo": 21.0, "avg_rfc": 68.0}}},
            {"id": "A16", "name": "behavior_velocity_planner", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 8200, "total_classes": 38, "total_methods": 255}, "complexity": {"avg_wmc": 36.8}, "cohesion": {"avg_lcom": 39.1}, "coupling": {"avg_cbo": 19.1, "avg_rfc": 59.4}}},
            {"id": "A17", "name": "obstacle_avoidance_planner", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 7400, "total_classes": 35, "total_methods": 230}, "complexity": {"avg_wmc": 34.0}, "cohesion": {"avg_lcom": 36.5}, "coupling": {"avg_cbo": 17.5, "avg_rfc": 55.0}}},
            {"id": "A18", "name": "scenario_selector", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 1800, "total_classes": 9, "total_methods": 54}, "complexity": {"avg_wmc": 11.2}, "cohesion": {"avg_lcom": 10.0}, "coupling": {"avg_cbo": 5.9, "avg_rfc": 17.8}}},

            {"id": "A19", "name": "trajectory_follower_controller", "app_type": "actuator", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 5200, "total_classes": 24, "total_methods": 160}, "complexity": {"avg_wmc": 27.5}, "cohesion": {"avg_lcom": 28.0}, "coupling": {"avg_cbo": 13.8, "avg_rfc": 43.1}}},
            {"id": "A20", "name": "latlon_controller", "app_type": "actuator", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 3100, "total_classes": 14, "total_methods": 92}, "complexity": {"avg_wmc": 16.0}, "cohesion": {"avg_lcom": 14.8}, "coupling": {"avg_cbo": 8.0, "avg_rfc": 25.4}}},
            {"id": "A21", "name": "shift_decider", "app_type": "actuator", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 1200, "total_classes": 6, "total_methods": 36}, "complexity": {"avg_wmc": 7.5}, "cohesion": {"avg_lcom": 6.8}, "coupling": {"avg_cbo": 4.1, "avg_rfc": 12.5}}},
            {"id": "A22", "name": "vehicle_cmd_gate", "app_type": "actuator", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 4400, "total_classes": 21, "total_methods": 138}, "complexity": {"avg_wmc": 23.1}, "cohesion": {"avg_lcom": 24.5}, "coupling": {"avg_cbo": 12.0, "avg_rfc": 38.0}}},

            {"id": "A23", "name": "raw_vehicle_cmd_converter", "app_type": "actuator", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 2300, "total_classes": 11, "total_methods": 72}, "complexity": {"avg_wmc": 13.0}, "cohesion": {"avg_lcom": 11.9}, "coupling": {"avg_cbo": 6.8, "avg_rfc": 21.0}}},
            {"id": "A24", "name": "can_bus_interface_node", "app_type": "actuator", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 3800, "total_classes": 18, "total_methods": 115}, "complexity": {"avg_wmc": 19.0}, "cohesion": {"avg_lcom": 18.0}, "coupling": {"avg_cbo": 9.9, "avg_rfc": 31.2}}},

            {"id": "A25", "name": "mrm_emergency_stop_operator", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 2700, "total_classes": 13, "total_methods": 82}, "complexity": {"avg_wmc": 14.5}, "cohesion": {"avg_lcom": 13.8}, "coupling": {"avg_cbo": 7.2, "avg_rfc": 22.8}}},
            {"id": "A26", "name": "emergency_handler", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 3300, "total_classes": 16, "total_methods": 102}, "complexity": {"avg_wmc": 17.2}, "cohesion": {"avg_lcom": 16.5}, "coupling": {"avg_cbo": 8.8, "avg_rfc": 27.9}}},
            {"id": "A27", "name": "system_error_monitor", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 2200, "total_classes": 11, "total_methods": 69}, "complexity": {"avg_wmc": 12.1}, "cohesion": {"avg_lcom": 11.0}, "coupling": {"avg_cbo": 6.0, "avg_rfc": 19.5}}},

            {"id": "A28", "name": "robot_state_publisher", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 1900, "total_classes": 9, "total_methods": 58}, "complexity": {"avg_wmc": 10.4}, "cohesion": {"avg_lcom": 9.5}, "coupling": {"avg_cbo": 5.2, "avg_rfc": 16.8}}},
            {"id": "A29", "name": "static_transform_publisher", "app_type": "processor", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 720, "total_classes": 4, "total_methods": 23}, "complexity": {"avg_wmc": 5.8}, "cohesion": {"avg_lcom": 4.5}, "coupling": {"avg_cbo": 3.1, "avg_rfc": 9.8}}},

            {"id": "A30", "name": "zenoh_teleop_bridge", "app_type": "processor", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 2600, "total_classes": 13, "total_methods": 80}, "complexity": {"avg_wmc": 13.8}, "cohesion": {"avg_lcom": 13.1}, "coupling": {"avg_cbo": 7.0, "avg_rfc": 22.0}}},
            {"id": "A31", "name": "autoware_web_hmi_gateway", "app_type": "processor", "criticality": "LOW", "hotstandby": False, "code_metrics": {"size": {"total_loc": 1750, "total_classes": 9, "total_methods": 51}, "complexity": {"avg_wmc": 9.2}, "cohesion": {"avg_lcom": 8.8}, "coupling": {"avg_cbo": 4.9, "avg_rfc": 15.2}}}
        ]

        relationships = [
            # Publishers to Topics
            {"source": "A0", "target": "T0", "type": "PUBLISHES_TO"},
            {"source": "A1", "target": "T0", "type": "PUBLISHES_TO"},
            {"source": "A2", "target": "T1", "type": "PUBLISHES_TO"},
            {"source": "A3", "target": "T2", "type": "PUBLISHES_TO"},
            {"source": "A4", "target": "T3", "type": "PUBLISHES_TO"},
            {"source": "A5", "target": "T23", "type": "PUBLISHES_TO"},
            {"source": "A6", "target": "T4", "type": "PUBLISHES_TO"},
            {"source": "A7", "target": "T5", "type": "PUBLISHES_TO"},
            {"source": "A9", "target": "T6", "type": "PUBLISHES_TO"},
            {"source": "A10", "target": "T8", "type": "PUBLISHES_TO"},
            {"source": "A11", "target": "T7", "type": "PUBLISHES_TO"},
            {"source": "A13", "target": "T9", "type": "PUBLISHES_TO"},
            {"source": "A14", "target": "T10", "type": "PUBLISHES_TO"},
            {"source": "A15", "target": "T11", "type": "PUBLISHES_TO"},
            {"source": "A16", "target": "T11", "type": "PUBLISHES_TO"},
            {"source": "A17", "target": "T11", "type": "PUBLISHES_TO"},
            {"source": "A19", "target": "T12", "type": "PUBLISHES_TO"},
            {"source": "A22", "target": "T12", "type": "PUBLISHES_TO"},
            {"source": "A23", "target": "T14", "type": "PUBLISHES_TO"},
            {"source": "A23", "target": "T15", "type": "PUBLISHES_TO"},
            {"source": "A25", "target": "T18", "type": "PUBLISHES_TO"},
            {"source": "A26", "target": "T19", "type": "PUBLISHES_TO"},
            {"source": "A27", "target": "T21", "type": "PUBLISHES_TO"},
            {"source": "A28", "target": "T16", "type": "PUBLISHES_TO"},
            {"source": "A29", "target": "T17", "type": "PUBLISHES_TO"},
            {"source": "A30", "target": "T20", "type": "PUBLISHES_TO"},
            {"source": "A31", "target": "T22", "type": "PUBLISHES_TO"},

            # Subscribers to Topics
            {"source": "A1", "target": "T0", "type": "SUBSCRIBES_TO"},
            {"source": "A6", "target": "T0", "type": "SUBSCRIBES_TO"},
            {"source": "A7", "target": "T4", "type": "SUBSCRIBES_TO"},
            {"source": "A7", "target": "T23", "type": "SUBSCRIBES_TO"},
            {"source": "A8", "target": "T5", "type": "SUBSCRIBES_TO"},
            {"source": "A9", "target": "T0", "type": "SUBSCRIBES_TO"},
            {"source": "A9", "target": "T5", "type": "SUBSCRIBES_TO"},
            {"source": "A10", "target": "T0", "type": "SUBSCRIBES_TO"},
            {"source": "A10", "target": "T9", "type": "SUBSCRIBES_TO"},
            {"source": "A11", "target": "T2", "type": "SUBSCRIBES_TO"},
            {"source": "A11", "target": "T3", "type": "SUBSCRIBES_TO"},
            {"source": "A11", "target": "T8", "type": "SUBSCRIBES_TO"},
            {"source": "A12", "target": "T7", "type": "SUBSCRIBES_TO"},
            {"source": "A14", "target": "T9", "type": "SUBSCRIBES_TO"},
            {"source": "A14", "target": "T22", "type": "SUBSCRIBES_TO"},
            {"source": "A15", "target": "T7", "type": "SUBSCRIBES_TO"},
            {"source": "A15", "target": "T9", "type": "SUBSCRIBES_TO"},
            {"source": "A15", "target": "T10", "type": "SUBSCRIBES_TO"},
            {"source": "A15", "target": "T5", "type": "SUBSCRIBES_TO"},
            {"source": "A16", "target": "T7", "type": "SUBSCRIBES_TO"},
            {"source": "A16", "target": "T5", "type": "SUBSCRIBES_TO"},
            {"source": "A17", "target": "T7", "type": "SUBSCRIBES_TO"},
            {"source": "A17", "target": "T6", "type": "SUBSCRIBES_TO"},
            {"source": "A19", "target": "T7", "type": "SUBSCRIBES_TO"},
            {"source": "A19", "target": "T11", "type": "SUBSCRIBES_TO"},
            {"source": "A20", "target": "T7", "type": "SUBSCRIBES_TO"},
            {"source": "A20", "target": "T11", "type": "SUBSCRIBES_TO"},
            {"source": "A22", "target": "T12", "type": "SUBSCRIBES_TO"},
            {"source": "A22", "target": "T18", "type": "SUBSCRIBES_TO"},
            {"source": "A22", "target": "T19", "type": "SUBSCRIBES_TO"},
            {"source": "A23", "target": "T12", "type": "SUBSCRIBES_TO"},
            {"source": "A24", "target": "T12", "type": "SUBSCRIBES_TO"},
            {"source": "A25", "target": "T19", "type": "SUBSCRIBES_TO"},
            {"source": "A26", "target": "T21", "type": "SUBSCRIBES_TO"},
            {"source": "A27", "target": "T21", "type": "SUBSCRIBES_TO"},
            {"source": "A22", "target": "T20", "type": "SUBSCRIBES_TO"},

            # Shared Library dependencies (USES)
            {"source": "A0", "target": "L6", "type": "USES"},
            {"source": "A0", "target": "L8", "type": "USES"},
            {"source": "A1", "target": "L3", "type": "USES"},
            {"source": "A1", "target": "L6", "type": "USES"},
            {"source": "A1", "target": "L8", "type": "USES"},
            {"source": "A6", "target": "L0", "type": "USES"},
            {"source": "A6", "target": "L1", "type": "USES"},
            {"source": "A6", "target": "L3", "type": "USES"},
            {"source": "A6", "target": "L8", "type": "USES"},
            {"source": "A7", "target": "L0", "type": "USES"},
            {"source": "A7", "target": "L1", "type": "USES"},
            {"source": "A7", "target": "L8", "type": "USES"},
            {"source": "A10", "target": "L0", "type": "USES"},
            {"source": "A10", "target": "L1", "type": "USES"},
            {"source": "A10", "target": "L8", "type": "USES"},
            {"source": "A11", "target": "L0", "type": "USES"},
            {"source": "A11", "target": "L1", "type": "USES"},
            {"source": "A11", "target": "L8", "type": "USES"},
            {"source": "A13", "target": "L4", "type": "USES"},
            {"source": "A13", "target": "L8", "type": "USES"},
            {"source": "A14", "target": "L4", "type": "USES"},
            {"source": "A14", "target": "L8", "type": "USES"},
            {"source": "A15", "target": "L0", "type": "USES"},
            {"source": "A15", "target": "L1", "type": "USES"},
            {"source": "A15", "target": "L2", "type": "USES"},
            {"source": "A15", "target": "L4", "type": "USES"},
            {"source": "A15", "target": "L5", "type": "USES"},
            {"source": "A15", "target": "L8", "type": "USES"},
            {"source": "A16", "target": "L0", "type": "USES"},
            {"source": "A16", "target": "L1", "type": "USES"},
            {"source": "A16", "target": "L2", "type": "USES"},
            {"source": "A16", "target": "L4", "type": "USES"},
            {"source": "A16", "target": "L8", "type": "USES"},
            {"source": "A17", "target": "L0", "type": "USES"},
            {"source": "A17", "target": "L1", "type": "USES"},
            {"source": "A17", "target": "L2", "type": "USES"},
            {"source": "A17", "target": "L8", "type": "USES"},
            {"source": "A19", "target": "L0", "type": "USES"},
            {"source": "A19", "target": "L2", "type": "USES"},
            {"source": "A19", "target": "L5", "type": "USES"},
            {"source": "A19", "target": "L8", "type": "USES"},
            {"source": "A22", "target": "L0", "type": "USES"},
            {"source": "A22", "target": "L7", "type": "USES"},
            {"source": "A22", "target": "L8", "type": "USES"},
            {"source": "A25", "target": "L7", "type": "USES"},
            {"source": "A25", "target": "L8", "type": "USES"},
            {"source": "A26", "target": "L7", "type": "USES"},
            {"source": "A26", "target": "L8", "type": "USES"},
            {"source": "A28", "target": "L9", "type": "USES"},
            {"source": "A28", "target": "L8", "type": "USES"},

            # Node hosting (RUNS_ON)
            {"source": "A0", "target": "N2", "type": "RUNS_ON"},
            {"source": "A1", "target": "N2", "type": "RUNS_ON"},
            {"source": "A2", "target": "N2", "type": "RUNS_ON"},
            {"source": "A3", "target": "N2", "type": "RUNS_ON"},
            {"source": "A4", "target": "N2", "type": "RUNS_ON"},
            {"source": "A5", "target": "N2", "type": "RUNS_ON"},
            {"source": "A6", "target": "N1", "type": "RUNS_ON"},
            {"source": "A7", "target": "N1", "type": "RUNS_ON"},
            {"source": "A8", "target": "N1", "type": "RUNS_ON"},
            {"source": "A9", "target": "N1", "type": "RUNS_ON"},
            {"source": "A10", "target": "N0", "type": "RUNS_ON"},
            {"source": "A11", "target": "N0", "type": "RUNS_ON"},
            {"source": "A12", "target": "N0", "type": "RUNS_ON"},
            {"source": "A13", "target": "N0", "type": "RUNS_ON"},
            {"source": "A14", "target": "N0", "type": "RUNS_ON"},
            {"source": "A15", "target": "N0", "type": "RUNS_ON"},
            {"source": "A16", "target": "N0", "type": "RUNS_ON"},
            {"source": "A17", "target": "N0", "type": "RUNS_ON"},
            {"source": "A18", "target": "N0", "type": "RUNS_ON"},
            {"source": "A19", "target": "N0", "type": "RUNS_ON"},
            {"source": "A20", "target": "N0", "type": "RUNS_ON"},
            {"source": "A21", "target": "N0", "type": "RUNS_ON"},
            {"source": "A22", "target": "N3", "type": "RUNS_ON"},
            {"source": "A23", "target": "N3", "type": "RUNS_ON"},
            {"source": "A24", "target": "N3", "type": "RUNS_ON"},
            {"source": "A25", "target": "N3", "type": "RUNS_ON"},
            {"source": "A26", "target": "N3", "type": "RUNS_ON"},
            {"source": "A27", "target": "N0", "type": "RUNS_ON"},
            {"source": "A28", "target": "N0", "type": "RUNS_ON"},
            {"source": "A29", "target": "N0", "type": "RUNS_ON"},
            {"source": "A30", "target": "N4", "type": "RUNS_ON"},
            {"source": "A31", "target": "N5", "type": "RUNS_ON"},

            {"source": "B0", "target": "N0", "type": "RUNS_ON"},
            {"source": "B1", "target": "N1", "type": "RUNS_ON"},
            {"source": "B2", "target": "N4", "type": "RUNS_ON"},

            # Topic Routing (ROUTES)
            {"source": "B0", "target": "T0", "type": "ROUTES"},
            {"source": "B0", "target": "T2", "type": "ROUTES"},
            {"source": "B0", "target": "T3", "type": "ROUTES"},
            {"source": "B0", "target": "T4", "type": "ROUTES"},
            {"source": "B0", "target": "T5", "type": "ROUTES"},
            {"source": "B0", "target": "T6", "type": "ROUTES"},
            {"source": "B0", "target": "T7", "type": "ROUTES"},
            {"source": "B0", "target": "T8", "type": "ROUTES"},
            {"source": "B0", "target": "T9", "type": "ROUTES"},
            {"source": "B0", "target": "T10", "type": "ROUTES"},
            {"source": "B0", "target": "T11", "type": "ROUTES"},
            {"source": "B0", "target": "T12", "type": "ROUTES"},
            {"source": "B0", "target": "T13", "type": "ROUTES"},
            {"source": "B0", "target": "T14", "type": "ROUTES"},
            {"source": "B0", "target": "T15", "type": "ROUTES"},
            {"source": "B0", "target": "T16", "type": "ROUTES"},
            {"source": "B0", "target": "T17", "type": "ROUTES"},
            {"source": "B0", "target": "T18", "type": "ROUTES"},
            {"source": "B0", "target": "T19", "type": "ROUTES"},
            {"source": "B1", "target": "T0", "type": "ROUTES"},
            {"source": "B1", "target": "T1", "type": "ROUTES"},
            {"source": "B1", "target": "T4", "type": "ROUTES"},
            {"source": "B1", "target": "T23", "type": "ROUTES"},
            {"source": "B2", "target": "T20", "type": "ROUTES"},
            {"source": "B2", "target": "T22", "type": "ROUTES"},

            # Hardware Interconnects (CONNECTS_TO)
            {"source": "N1", "target": "N0", "type": "CONNECTS_TO"},
            {"source": "N2", "target": "N0", "type": "CONNECTS_TO"},
            {"source": "N2", "target": "N1", "type": "CONNECTS_TO"},
            {"source": "N0", "target": "N3", "type": "CONNECTS_TO"},
            {"source": "N0", "target": "N4", "type": "CONNECTS_TO"},
            {"source": "N4", "target": "N5", "type": "CONNECTS_TO"}
        ]

        metadata = {
            "scale": {
                "apps": len(applications),
                "topics": len(topics),
                "brokers": len(brokers),
                "nodes": len(nodes),
                "libs": len(libraries)
            },
            "seed": 2026,
            "generation_mode": "realworld_open_source",
            "domain": "autoware_ros2",
            "scenario": "autoware_universe_ros2_autonomous_driving",
            "description": "Authentic Autoware.universe (ROS 2) autonomous driving software architecture containing perception, sensing, planning, control, and system nodes."
        }

        return {
            "metadata": metadata,
            "nodes": nodes,
            "brokers": brokers,
            "topics": topics,
            "applications": applications,
            "libraries": libraries,
            "relationships": _to_canonical_relationships(relationships)
        }

    @staticmethod
    def create_cloud_microservices_topology() -> Dict[str, Any]:
        """
        Creates an authentic Cloud-Native Pub-Sub Microservices mesh topology graph,
        based on production-grade microservice benchmark architectures (e.g. Google Online Boutique / E-Commerce Stack).

        Topology overview:
        - 22 Applications across Frontend, Order Processing, Payment, Inventory, Recommendations, Analytics, Notifications, and Search
        - 20 Topics with explicit Message Broker QoS configurations (Kafka, RabbitMQ, Redis PubSub)
        - 4 Brokers (Apache Kafka cluster, RabbitMQ exchange, Redis PubSub instance, NATS stream)
        - 6 Deployment Nodes (k8s-master, k8s-worker-1, k8s-worker-2, k8s-worker-3, cloud-db-node, edge-ingress)
        - 8 Shared Libraries (shared-auth-client, kafka-common-producer, grpc-telemetry-sdk, redis-cache-utils, etc.)
        """
        nodes = [
            {"id": "N0", "name": "k8s-control-plane", "type": "Node", "spec": "AWS c6i.4xlarge"},
            {"id": "N1", "name": "k8s-worker-node-1", "type": "Node", "spec": "AWS c6i.8xlarge"},
            {"id": "N2", "name": "k8s-worker-node-2", "type": "Node", "spec": "AWS c6i.8xlarge"},
            {"id": "N3", "name": "k8s-worker-node-3", "type": "Node", "spec": "AWS m6i.8xlarge"},
            {"id": "N4", "name": "cloud-managed-db-cluster", "type": "Node", "spec": "AWS r6g.4xlarge"},
            {"id": "N5", "name": "edge-ingress-gateway", "type": "Node", "spec": "AWS c6i.2xlarge"}
        ]

        brokers = [
            {"id": "B0", "name": "apache-kafka-cluster", "type": "Broker", "protocol": "Kafka"},
            {"id": "B1", "name": "rabbitmq-broker", "type": "Broker", "protocol": "AMQP"},
            {"id": "B2", "name": "redis-pubsub", "type": "Broker", "protocol": "Redis"},
            {"id": "B3", "name": "nats-jetstream", "type": "Broker", "protocol": "NATS"}
        ]

        topics = [
            {"id": "T0", "name": "order.checkout.events", "size": 1024, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "CRITICAL"}, "frequency": 50.0, "criticality": "HIGH"},
            {"id": "T1", "name": "payment.transaction.completed", "size": 512, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "CRITICAL"}, "frequency": 50.0, "criticality": "HIGH"},
            {"id": "T2", "name": "payment.transaction.failed", "size": 512, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "CRITICAL"}, "frequency": 5.0, "criticality": "HIGH"},
            {"id": "T3", "name": "inventory.stock.updated", "size": 256, "qos": {"durability": "TRANSIENT", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 100.0, "criticality": "HIGH"},
            {"id": "T4", "name": "inventory.stock.reservation_failed", "size": 256, "qos": {"durability": "TRANSIENT", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 2.0, "criticality": "HIGH"},
            {"id": "T5", "name": "user.notification.email", "size": 2048, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "MEDIUM"}, "frequency": 20.0, "criticality": "MEDIUM"},
            {"id": "T6", "name": "user.notification.push", "size": 1024, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "MEDIUM"}, "frequency": 30.0, "criticality": "MEDIUM"},
            {"id": "T7", "name": "analytics.user.clicks", "size": 512, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "LOW"}, "frequency": 500.0, "criticality": "LOW"},
            {"id": "T8", "name": "analytics.product.views", "size": 512, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "LOW"}, "frequency": 800.0, "criticality": "LOW"},
            {"id": "T9", "name": "recommendation.user.vector_update", "size": 4096, "qos": {"durability": "TRANSIENT", "reliability": "RELIABLE", "transport_priority": "MEDIUM"}, "frequency": 10.0, "criticality": "MEDIUM"},
            {"id": "T10", "name": "shipping.label.generated", "size": 1024, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 15.0, "criticality": "HIGH"},
            {"id": "T11", "name": "shipping.carrier.dispatched", "size": 1024, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 15.0, "criticality": "HIGH"},
            {"id": "T12", "name": "audit.security.logs", "size": 512, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 200.0, "criticality": "HIGH"},
            {"id": "T13", "name": "search.index.sync", "size": 2048, "qos": {"durability": "TRANSIENT", "reliability": "RELIABLE", "transport_priority": "MEDIUM"}, "frequency": 25.0, "criticality": "MEDIUM"},
            {"id": "T14", "name": "fraud.detection.alerts", "size": 512, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "HIGHEST"}, "frequency": 1.0, "criticality": "HIGH"},
            {"id": "T15", "name": "cart.session.sync", "size": 256, "qos": {"durability": "VOLATILE", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 100.0, "criticality": "HIGH"},
            {"id": "T16", "name": "ad.impression.stream", "size": 256, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "LOW"}, "frequency": 300.0, "criticality": "LOW"},
            {"id": "T17", "name": "currency.rate.refresh", "size": 128, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "MEDIUM"}, "frequency": 0.01, "criticality": "MEDIUM"},
            {"id": "T18", "name": "system.health.heartbeat", "size": 128, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "LOW"}, "frequency": 1.0, "criticality": "LOW"},
            {"id": "T19", "name": "user.registration.event", "size": 512, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 5.0, "criticality": "HIGH"}
        ]

        libraries = [
            {"id": "L0", "name": "shared-auth-jwt-client", "version": "3.1.0", "description": "JWT authentication verification client"},
            {"id": "L1", "name": "kafka-common-producer", "version": "2.8.4", "description": "Resilient Kafka producer with retry & circuit breaker"},
            {"id": "L2", "name": "grpc-telemetry-sdk", "version": "1.4.2", "description": "OpenTelemetry tracing & metrics exporter"},
            {"id": "L3", "name": "redis-cache-utils", "version": "4.0.1", "description": "Redis connection pooling & serialization helper"},
            {"id": "L4", "name": "event-schema-registry", "version": "5.2.0", "description": "Protobuf/Avro event schema definition library"},
            {"id": "L5", "name": "payment-gateway-sdk", "version": "1.9.0", "description": "Stripe/PayPal encrypted client wrapper"},
            {"id": "L6", "name": "spring-cloud-circuitbreaker", "version": "3.0.2", "description": "Resilience4j fault tolerance library"},
            {"id": "L7", "name": "elasticsearch-client-util", "version": "7.17.0", "description": "Elasticsearch client connection manager"}
        ]

        applications = [
            {"id": "A0", "name": "frontend-web-ui", "app_type": "sensor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 4500, "total_classes": 25, "total_methods": 140}, "complexity": {"avg_wmc": 16.5}, "cohesion": {"avg_lcom": 18.0}, "coupling": {"avg_cbo": 9.0, "avg_rfc": 30.0}}},
            {"id": "A1", "name": "api-gateway-service", "app_type": "sensor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 5200, "total_classes": 29, "total_methods": 170}, "complexity": {"avg_wmc": 22.1}, "cohesion": {"avg_lcom": 24.0}, "coupling": {"avg_cbo": 12.5, "avg_rfc": 41.2}}},
            {"id": "A2", "name": "auth-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 3100, "total_classes": 16, "total_methods": 95}, "complexity": {"avg_wmc": 15.0}, "cohesion": {"avg_lcom": 14.5}, "coupling": {"avg_cbo": 7.9, "avg_rfc": 25.0}}},
            
            {"id": "A3", "name": "cart-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 2800, "total_classes": 14, "total_methods": 88}, "complexity": {"avg_wmc": 13.5}, "cohesion": {"avg_lcom": 12.8}, "coupling": {"avg_cbo": 6.8, "avg_rfc": 22.1}}},
            {"id": "A4", "name": "checkout-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 6100, "total_classes": 32, "total_methods": 195}, "complexity": {"avg_wmc": 26.4}, "cohesion": {"avg_lcom": 28.1}, "coupling": {"avg_cbo": 14.0, "avg_rfc": 45.0}}},
            {"id": "A5", "name": "order-processor-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 7800, "total_classes": 40, "total_methods": 240}, "complexity": {"avg_wmc": 31.0}, "cohesion": {"avg_lcom": 34.0}, "coupling": {"avg_cbo": 16.8, "avg_rfc": 53.0}}},
            {"id": "A6", "name": "payment-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 4900, "total_classes": 24, "total_methods": 150}, "complexity": {"avg_wmc": 20.8}, "cohesion": {"avg_lcom": 21.9}, "coupling": {"avg_cbo": 11.2, "avg_rfc": 35.8}}},
            {"id": "A7", "name": "inventory-reservation-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 5400, "total_classes": 26, "total_methods": 165}, "complexity": {"avg_wmc": 23.0}, "cohesion": {"avg_lcom": 25.0}, "coupling": {"avg_cbo": 12.0, "avg_rfc": 39.0}}},
            
            {"id": "A8", "name": "email-notification-worker", "app_type": "actuator", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 1900, "total_classes": 10, "total_methods": 58}, "complexity": {"avg_wmc": 9.5}, "cohesion": {"avg_lcom": 8.8}, "coupling": {"avg_cbo": 4.5, "avg_rfc": 15.0}}},
            {"id": "A9", "name": "push-notification-worker", "app_type": "actuator", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 1600, "total_classes": 8, "total_methods": 49}, "complexity": {"avg_wmc": 8.2}, "cohesion": {"avg_lcom": 7.9}, "coupling": {"avg_cbo": 4.0, "avg_rfc": 13.5}}},
            {"id": "A10", "name": "shipping-fulfillment-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 3900, "total_classes": 20, "total_methods": 122}, "complexity": {"avg_wmc": 17.0}, "cohesion": {"avg_lcom": 18.2}, "coupling": {"avg_cbo": 9.1, "avg_rfc": 28.5}}},
            
            {"id": "A11", "name": "clickstream-analytics-consumer", "app_type": "processor", "criticality": "LOW", "hotstandby": False, "code_metrics": {"size": {"total_loc": 2400, "total_classes": 12, "total_methods": 75}, "complexity": {"avg_wmc": 11.0}, "cohesion": {"avg_lcom": 10.5}, "coupling": {"avg_cbo": 5.8, "avg_rfc": 18.0}}},
            {"id": "A12", "name": "product-recommendation-engine", "app_type": "processor", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 6500, "total_classes": 30, "total_methods": 190}, "complexity": {"avg_wmc": 28.0}, "cohesion": {"avg_lcom": 30.5}, "coupling": {"avg_cbo": 15.0, "avg_rfc": 47.0}}},
            {"id": "A13", "name": "fraud-detection-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 5100, "total_classes": 25, "total_methods": 155}, "complexity": {"avg_wmc": 24.5}, "cohesion": {"avg_lcom": 26.0}, "coupling": {"avg_cbo": 13.1, "avg_rfc": 40.0}}},

            {"id": "A14", "name": "search-indexing-worker", "app_type": "processor", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 3200, "total_classes": 16, "total_methods": 98}, "complexity": {"avg_wmc": 14.8}, "cohesion": {"avg_lcom": 15.0}, "coupling": {"avg_cbo": 7.8, "avg_rfc": 24.1}}},
            {"id": "A15", "name": "catalog-search-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 4100, "total_classes": 21, "total_methods": 130}, "complexity": {"avg_wmc": 18.2}, "cohesion": {"avg_lcom": 19.1}, "coupling": {"avg_cbo": 9.8, "avg_rfc": 31.0}}},
            {"id": "A16", "name": "currency-conversion-service", "app_type": "processor", "criticality": "LOW", "hotstandby": False, "code_metrics": {"size": {"total_loc": 1100, "total_classes": 5, "total_methods": 34}, "complexity": {"avg_wmc": 6.8}, "cohesion": {"avg_lcom": 5.9}, "coupling": {"avg_cbo": 3.4, "avg_rfc": 10.5}}},
            {"id": "A17", "name": "audit-logging-collector", "app_type": "actuator", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 2300, "total_classes": 11, "total_methods": 71}, "complexity": {"avg_wmc": 10.9}, "cohesion": {"avg_lcom": 10.0}, "coupling": {"avg_cbo": 5.1, "avg_rfc": 16.5}}},

            {"id": "A18", "name": "ad-service", "app_type": "sensor", "criticality": "LOW", "hotstandby": False, "code_metrics": {"size": {"total_loc": 1500, "total_classes": 7, "total_methods": 44}, "complexity": {"avg_wmc": 7.5}, "cohesion": {"avg_lcom": 6.8}, "coupling": {"avg_cbo": 3.9, "avg_rfc": 12.0}}},
            {"id": "A19", "name": "user-profile-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 3600, "total_classes": 18, "total_methods": 110}, "complexity": {"avg_wmc": 16.0}, "cohesion": {"avg_lcom": 17.0}, "coupling": {"avg_cbo": 8.5, "avg_rfc": 27.0}}},
            {"id": "A20", "name": "prometheus-metrics-exporter", "app_type": "actuator", "criticality": "LOW", "hotstandby": False, "code_metrics": {"size": {"total_loc": 950, "total_classes": 5, "total_methods": 28}, "complexity": {"avg_wmc": 5.1}, "cohesion": {"avg_lcom": 4.2}, "coupling": {"avg_cbo": 2.8, "avg_rfc": 8.9}}},
            {"id": "A21", "name": "tracing-agent-daemon", "app_type": "actuator", "criticality": "LOW", "hotstandby": False, "code_metrics": {"size": {"total_loc": 1250, "total_classes": 6, "total_methods": 36}, "complexity": {"avg_wmc": 6.2}, "cohesion": {"avg_lcom": 5.5}, "coupling": {"avg_cbo": 3.2, "avg_rfc": 10.1}}}
        ]

        relationships = [
            # Publishers
            {"source": "A0", "target": "T7", "type": "PUBLISHES_TO"},
            {"source": "A0", "target": "T8", "type": "PUBLISHES_TO"},
            {"source": "A0", "target": "T15", "type": "PUBLISHES_TO"},
            {"source": "A1", "target": "T12", "type": "PUBLISHES_TO"},
            {"source": "A1", "target": "T18", "type": "PUBLISHES_TO"},
            {"source": "A2", "target": "T19", "type": "PUBLISHES_TO"},
            {"source": "A4", "target": "T0", "type": "PUBLISHES_TO"},
            {"source": "A5", "target": "T3", "type": "PUBLISHES_TO"},
            {"source": "A5", "target": "T10", "type": "PUBLISHES_TO"},
            {"source": "A6", "target": "T1", "type": "PUBLISHES_TO"},
            {"source": "A6", "target": "T2", "type": "PUBLISHES_TO"},
            {"source": "A7", "target": "T3", "type": "PUBLISHES_TO"},
            {"source": "A7", "target": "T4", "type": "PUBLISHES_TO"},
            {"source": "A10", "target": "T11", "type": "PUBLISHES_TO"},
            {"source": "A12", "target": "T9", "type": "PUBLISHES_TO"},
            {"source": "A13", "target": "T14", "type": "PUBLISHES_TO"},
            {"source": "A14", "target": "T13", "type": "PUBLISHES_TO"},
            {"source": "A18", "target": "T16", "type": "PUBLISHES_TO"},

            # Subscribers
            {"source": "A3", "target": "T15", "type": "SUBSCRIBES_TO"},
            {"source": "A5", "target": "T0", "type": "SUBSCRIBES_TO"},
            {"source": "A6", "target": "T0", "type": "SUBSCRIBES_TO"},
            {"source": "A7", "target": "T0", "type": "SUBSCRIBES_TO"},
            {"source": "A5", "target": "T1", "type": "SUBSCRIBES_TO"},
            {"source": "A5", "target": "T4", "type": "SUBSCRIBES_TO"},
            {"source": "A8", "target": "T0", "type": "SUBSCRIBES_TO"},
            {"source": "A8", "target": "T5", "type": "SUBSCRIBES_TO"},
            {"source": "A8", "target": "T19", "type": "SUBSCRIBES_TO"},
            {"source": "A9", "target": "T6", "type": "SUBSCRIBES_TO"},
            {"source": "A10", "target": "T10", "type": "SUBSCRIBES_TO"},
            {"source": "A11", "target": "T7", "type": "SUBSCRIBES_TO"},
            {"source": "A11", "target": "T8", "type": "SUBSCRIBES_TO"},
            {"source": "A12", "target": "T8", "type": "SUBSCRIBES_TO"},
            {"source": "A12", "target": "T9", "type": "SUBSCRIBES_TO"},
            {"source": "A13", "target": "T0", "type": "SUBSCRIBES_TO"},
            {"source": "A13", "target": "T1", "type": "SUBSCRIBES_TO"},
            {"source": "A14", "target": "T3", "type": "SUBSCRIBES_TO"},
            {"source": "A15", "target": "T13", "type": "SUBSCRIBES_TO"},
            {"source": "A17", "target": "T12", "type": "SUBSCRIBES_TO"},

            # Shared Library dependencies (USES)
            {"source": "A0", "target": "L0", "type": "USES"},
            {"source": "A0", "target": "L2", "type": "USES"},
            {"source": "A0", "target": "L3", "type": "USES"},
            {"source": "A1", "target": "L0", "type": "USES"},
            {"source": "A1", "target": "L2", "type": "USES"},
            {"source": "A1", "target": "L6", "type": "USES"},
            {"source": "A2", "target": "L0", "type": "USES"},
            {"source": "A2", "target": "L2", "type": "USES"},
            {"source": "A4", "target": "L0", "type": "USES"},
            {"source": "A4", "target": "L1", "type": "USES"},
            {"source": "A4", "target": "L2", "type": "USES"},
            {"source": "A4", "target": "L4", "type": "USES"},
            {"source": "A4", "target": "L6", "type": "USES"},
            {"source": "A5", "target": "L0", "type": "USES"},
            {"source": "A5", "target": "L1", "type": "USES"},
            {"source": "A5", "target": "L2", "type": "USES"},
            {"source": "A5", "target": "L4", "type": "USES"},
            {"source": "A5", "target": "L6", "type": "USES"},
            {"source": "A6", "target": "L0", "type": "USES"},
            {"source": "A6", "target": "L1", "type": "USES"},
            {"source": "A6", "target": "L2", "type": "USES"},
            {"source": "A6", "target": "L4", "type": "USES"},
            {"source": "A6", "target": "L5", "type": "USES"},
            {"source": "A6", "target": "L6", "type": "USES"},
            {"source": "A7", "target": "L0", "type": "USES"},
            {"source": "A7", "target": "L1", "type": "USES"},
            {"source": "A7", "target": "L2", "type": "USES"},
            {"source": "A7", "target": "L3", "type": "USES"},
            {"source": "A7", "target": "L4", "type": "USES"},
            {"source": "A10", "target": "L1", "type": "USES"},
            {"source": "A10", "target": "L2", "type": "USES"},
            {"source": "A10", "target": "L4", "type": "USES"},
            {"source": "A13", "target": "L1", "type": "USES"},
            {"source": "A13", "target": "L2", "type": "USES"},
            {"source": "A13", "target": "L4", "type": "USES"},
            {"source": "A14", "target": "L7", "type": "USES"},
            {"source": "A15", "target": "L7", "type": "USES"},

            # Node Hosting (RUNS_ON)
            {"source": "A0", "target": "N5", "type": "RUNS_ON"},
            {"source": "A1", "target": "N5", "type": "RUNS_ON"},
            {"source": "A2", "target": "N1", "type": "RUNS_ON"},
            {"source": "A3", "target": "N1", "type": "RUNS_ON"},
            {"source": "A4", "target": "N1", "type": "RUNS_ON"},
            {"source": "A5", "target": "N1", "type": "RUNS_ON"},
            {"source": "A6", "target": "N2", "type": "RUNS_ON"},
            {"source": "A7", "target": "N2", "type": "RUNS_ON"},
            {"source": "A8", "target": "N3", "type": "RUNS_ON"},
            {"source": "A9", "target": "N3", "type": "RUNS_ON"},
            {"source": "A10", "target": "N2", "type": "RUNS_ON"},
            {"source": "A11", "target": "N3", "type": "RUNS_ON"},
            {"source": "A12", "target": "N3", "type": "RUNS_ON"},
            {"source": "A13", "target": "N2", "type": "RUNS_ON"},
            {"source": "A14", "target": "N3", "type": "RUNS_ON"},
            {"source": "A15", "target": "N3", "type": "RUNS_ON"},
            {"source": "A16", "target": "N1", "type": "RUNS_ON"},
            {"source": "A17", "target": "N0", "type": "RUNS_ON"},
            {"source": "A18", "target": "N3", "type": "RUNS_ON"},
            {"source": "A19", "target": "N1", "type": "RUNS_ON"},
            {"source": "A20", "target": "N0", "type": "RUNS_ON"},
            {"source": "A21", "target": "N0", "type": "RUNS_ON"},

            {"source": "B0", "target": "N1", "type": "RUNS_ON"},
            {"source": "B1", "target": "N2", "type": "RUNS_ON"},
            {"source": "B2", "target": "N1", "type": "RUNS_ON"},
            {"source": "B3", "target": "N3", "type": "RUNS_ON"},

            # Broker Routing (ROUTES)
            {"source": "B0", "target": "T0", "type": "ROUTES"},
            {"source": "B0", "target": "T1", "type": "ROUTES"},
            {"source": "B0", "target": "T2", "type": "ROUTES"},
            {"source": "B0", "target": "T3", "type": "ROUTES"},
            {"source": "B0", "target": "T4", "type": "ROUTES"},
            {"source": "B0", "target": "T7", "type": "ROUTES"},
            {"source": "B0", "target": "T8", "type": "ROUTES"},
            {"source": "B0", "target": "T9", "type": "ROUTES"},
            {"source": "B0", "target": "T10", "type": "ROUTES"},
            {"source": "B0", "target": "T11", "type": "ROUTES"},
            {"source": "B0", "target": "T12", "type": "ROUTES"},
            {"source": "B0", "target": "T13", "type": "ROUTES"},
            {"source": "B0", "target": "T14", "type": "ROUTES"},
            {"source": "B0", "target": "T19", "type": "ROUTES"},
            {"source": "B1", "target": "T5", "type": "ROUTES"},
            {"source": "B1", "target": "T6", "type": "ROUTES"},
            {"source": "B1", "target": "T10", "type": "ROUTES"},
            {"source": "B2", "target": "T15", "type": "ROUTES"},
            {"source": "B3", "target": "T16", "type": "ROUTES"},
            {"source": "B3", "target": "T17", "type": "ROUTES"},
            {"source": "B3", "target": "T18", "type": "ROUTES"},

            # Hardware Interconnects (CONNECTS_TO)
            {"source": "N5", "target": "N1", "type": "CONNECTS_TO"},
            {"source": "N1", "target": "N2", "type": "CONNECTS_TO"},
            {"source": "N1", "target": "N3", "type": "CONNECTS_TO"},
            {"source": "N2", "target": "N4", "type": "CONNECTS_TO"},
            {"source": "N0", "target": "N1", "type": "CONNECTS_TO"},
            {"source": "N0", "target": "N2", "type": "CONNECTS_TO"}
        ]

        metadata = {
            "scale": {
                "apps": len(applications),
                "topics": len(topics),
                "brokers": len(brokers),
                "nodes": len(nodes),
                "libs": len(libraries)
            },
            "seed": 2026,
            "generation_mode": "realworld_open_source",
            "domain": "cloud_microservices",
            "scenario": "production_ecommerce_cloud_microservices_mesh",
            "description": "Authentic production-grade cloud-native microservices mesh containing order processing, payment, inventory, recommendations, analytics, and notification services."
        }

        return {
            "metadata": metadata,
            "nodes": nodes,
            "brokers": brokers,
            "topics": topics,
            "applications": applications,
            "libraries": libraries,
            "relationships": _to_canonical_relationships(relationships)
        }

    @staticmethod
    def create_trainticket_microservices_topology() -> Dict[str, Any]:
        """
        Creates an authentic Train-Ticket Microservices Benchmark topology graph,
        based on the de facto academic microservice benchmark (Fudan University Train-Ticket).

        Topology overview:
        - 41 Applications across Frontend, Auth, User, Travel, Order, Route, Seat, Payment, Food, Security, and Admin
        - 30 Topics / Event Streams / API Endpoints with explicit Message Broker QoS configurations
        - 3 Brokers (RabbitMQ Cluster, Redis PubSub EventBus, Spring Cloud Eureka Discovery Server)
        - 8 Deployment Nodes (k8s-control-plane, 3 k8s workers, MySQL cluster, MongoDB cluster, Redis cluster, Edge Ingress)
        - 8 Shared Libraries (spring-boot-starter-web, eureka-client, ts-common-dto, mybatis, etc.)
        - Realistic SonarQube code quality metrics for applications
        """
        nodes = [
            {"id": "N0", "name": "k8s-control-plane", "type": "Node", "spec": "AWS c6i.4xlarge"},
            {"id": "N1", "name": "k8s-worker-node-1", "type": "Node", "spec": "AWS c6i.8xlarge"},
            {"id": "N2", "name": "k8s-worker-node-2", "type": "Node", "spec": "AWS c6i.8xlarge"},
            {"id": "N3", "name": "k8s-worker-node-3", "type": "Node", "spec": "AWS m6i.8xlarge"},
            {"id": "N4", "name": "mysql-db-cluster", "type": "Node", "spec": "AWS r6g.4xlarge"},
            {"id": "N5", "name": "mongo-db-cluster", "type": "Node", "spec": "AWS r6g.4xlarge"},
            {"id": "N6", "name": "redis-cache-cluster", "type": "Node", "spec": "AWS r6g.2xlarge"},
            {"id": "N7", "name": "edge-ingress-gateway", "type": "Node", "spec": "AWS c6i.2xlarge"}
        ]

        brokers = [
            {"id": "B0", "name": "rabbitmq-broker-cluster", "type": "Broker", "protocol": "AMQP"},
            {"id": "B1", "name": "redis-pubsub-eventbus", "type": "Broker", "protocol": "Redis"},
            {"id": "B2", "name": "spring-eureka-naming-server", "type": "Broker", "protocol": "HTTP-Discovery"}
        ]

        topics = [
            {"id": "T0", "name": "ts.order.create.event", "size": 1024, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "CRITICAL"}, "frequency": 80.0, "criticality": "HIGH"},
            {"id": "T1", "name": "ts.order.cancel.event", "size": 512, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 10.0, "criticality": "HIGH"},
            {"id": "T2", "name": "ts.travel.query.request", "size": 2048, "qos": {"durability": "VOLATILE", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 150.0, "criticality": "HIGH"},
            {"id": "T3", "name": "ts.ticket.preserve.command", "size": 1024, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "CRITICAL"}, "frequency": 60.0, "criticality": "HIGH"},
            {"id": "T4", "name": "ts.seat.allocate.request", "size": 512, "qos": {"durability": "VOLATILE", "reliability": "RELIABLE", "transport_priority": "CRITICAL"}, "frequency": 100.0, "criticality": "HIGH"},
            {"id": "T5", "name": "ts.payment.pay.event", "size": 512, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "CRITICAL"}, "frequency": 50.0, "criticality": "HIGH"},
            {"id": "T6", "name": "ts.payment.refund.event", "size": 512, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 5.0, "criticality": "HIGH"},
            {"id": "T7", "name": "ts.consign.record.event", "size": 256, "qos": {"durability": "TRANSIENT", "reliability": "BEST_EFFORT", "transport_priority": "LOW"}, "frequency": 15.0, "criticality": "LOW"},
            {"id": "T8", "name": "ts.food.order.event", "size": 512, "qos": {"durability": "TRANSIENT", "reliability": "RELIABLE", "transport_priority": "MEDIUM"}, "frequency": 25.0, "criticality": "MEDIUM"},
            {"id": "T9", "name": "ts.notification.email.send", "size": 2048, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "LOW"}, "frequency": 40.0, "criticality": "LOW"},
            {"id": "T10", "name": "ts.notification.sms.send", "size": 512, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "MEDIUM"}, "frequency": 30.0, "criticality": "MEDIUM"},
            {"id": "T11", "name": "ts.security.check.event", "size": 512, "qos": {"durability": "VOLATILE", "reliability": "RELIABLE", "transport_priority": "CRITICAL"}, "frequency": 120.0, "criticality": "HIGH"},
            {"id": "T12", "name": "ts.user.login.audit", "size": 256, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "MEDIUM"}, "frequency": 45.0, "criticality": "MEDIUM"},
            {"id": "T13", "name": "ts.user.register.event", "size": 512, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 10.0, "criticality": "HIGH"},
            {"id": "T14", "name": "ts.route.update.event", "size": 1024, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 2.0, "criticality": "HIGH"},
            {"id": "T15", "name": "ts.train.status.update", "size": 512, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 5.0, "criticality": "HIGH"},
            {"id": "T16", "name": "ts.price.config.sync", "size": 256, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "MEDIUM"}, "frequency": 1.0, "criticality": "MEDIUM"},
            {"id": "T17", "name": "ts.station.list.sync", "size": 2048, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "MEDIUM"}, "frequency": 0.5, "criticality": "MEDIUM"},
            {"id": "T18", "name": "ts.voucher.issue.event", "size": 512, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "LOW"}, "frequency": 8.0, "criticality": "LOW"},
            {"id": "T19", "name": "ts.contacts.sync.event", "size": 512, "qos": {"durability": "VOLATILE", "reliability": "RELIABLE", "transport_priority": "MEDIUM"}, "frequency": 20.0, "criticality": "MEDIUM"},
            {"id": "T20", "name": "ts.admin.basic.update", "size": 1024, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "MEDIUM"}, "frequency": 0.2, "criticality": "MEDIUM"},
            {"id": "T21", "name": "ts.admin.travel.update", "size": 1024, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "MEDIUM"}, "frequency": 0.2, "criticality": "MEDIUM"},
            {"id": "T22", "name": "ts.admin.order.update", "size": 1024, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "MEDIUM"}, "frequency": 0.2, "criticality": "MEDIUM"},
            {"id": "T23", "name": "ts.admin.user.update", "size": 1024, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "MEDIUM"}, "frequency": 0.2, "criticality": "MEDIUM"},
            {"id": "T24", "name": "ts.telemetry.metrics.stream", "size": 512, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "LOW"}, "frequency": 500.0, "criticality": "LOW"},
            {"id": "T25", "name": "ts.audit.access.log", "size": 512, "qos": {"durability": "PERSISTENT", "reliability": "BEST_EFFORT", "transport_priority": "LOW"}, "frequency": 300.0, "criticality": "LOW"},
            {"id": "T26", "name": "ts.cache.invalidation.event", "size": 256, "qos": {"durability": "VOLATILE", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 100.0, "criticality": "HIGH"},
            {"id": "T27", "name": "ts.assurance.purchase.event", "size": 256, "qos": {"durability": "TRANSIENT", "reliability": "RELIABLE", "transport_priority": "MEDIUM"}, "frequency": 12.0, "criticality": "MEDIUM"},
            {"id": "T28", "name": "ts.inside.payment.event", "size": 512, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "CRITICAL"}, "frequency": 40.0, "criticality": "HIGH"},
            {"id": "T29", "name": "ts.wait.order.process", "size": 512, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 15.0, "criticality": "HIGH"}
        ]

        libraries = [
            {"id": "L0", "name": "spring-boot-starter-web", "version": "3.2.1", "description": "Spring MVC REST web framework starter"},
            {"id": "L1", "name": "spring-cloud-starter-netflix-eureka-client", "version": "4.1.0", "description": "Eureka service discovery client"},
            {"id": "L2", "name": "ts-common-dto-library", "version": "1.0.0", "description": "Train-Ticket common Data Transfer Objects & Response wrappers"},
            {"id": "L3", "name": "mybatis-spring-boot-starter", "version": "3.0.3", "description": "MyBatis SQL mapping ORM library"},
            {"id": "L4", "name": "spring-boot-starter-data-redis", "version": "3.2.1", "description": "Redis cache & Session management client"},
            {"id": "L5", "name": "spring-boot-starter-amqp", "version": "3.2.1", "description": "RabbitMQ message listener & template wrapper"},
            {"id": "L6", "name": "jjwt-api-auth", "version": "0.12.3", "description": "JSON Web Token auth & verification library"},
            {"id": "L7", "name": "spring-boot-starter-actuator", "version": "3.2.1", "description": "Prometheus & Micrometer production telemetry SDK"}
        ]

        applications = [
            {"id": "A0", "name": "ts-ui-dashboard", "app_type": "sensor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 4800, "total_classes": 26, "total_methods": 142}, "complexity": {"avg_wmc": 16.8}, "cohesion": {"avg_lcom": 18.2}, "coupling": {"avg_cbo": 9.2, "avg_rfc": 30.5}}},
            {"id": "A1", "name": "ts-gateway-service", "app_type": "sensor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 5800, "total_classes": 31, "total_methods": 182}, "complexity": {"avg_wmc": 24.0}, "cohesion": {"avg_lcom": 25.5}, "coupling": {"avg_cbo": 13.5, "avg_rfc": 43.0}}},
            {"id": "A2", "name": "ts-auth-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 3400, "total_classes": 18, "total_methods": 105}, "complexity": {"avg_wmc": 16.2}, "cohesion": {"avg_lcom": 15.0}, "coupling": {"avg_cbo": 8.5, "avg_rfc": 27.0}}},
            {"id": "A3", "name": "ts-user-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 2900, "total_classes": 15, "total_methods": 92}, "complexity": {"avg_wmc": 14.1}, "cohesion": {"avg_lcom": 13.2}, "coupling": {"avg_cbo": 7.1, "avg_rfc": 23.0}}},
            {"id": "A4", "name": "ts-verification-code-service", "app_type": "processor", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 1200, "total_classes": 6, "total_methods": 38}, "complexity": {"avg_wmc": 7.2}, "cohesion": {"avg_lcom": 6.1}, "coupling": {"avg_cbo": 3.8, "avg_rfc": 12.0}}},
            {"id": "A5", "name": "ts-contacts-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 2600, "total_classes": 14, "total_methods": 84}, "complexity": {"avg_wmc": 12.8}, "cohesion": {"avg_lcom": 12.0}, "coupling": {"avg_cbo": 6.5, "avg_rfc": 21.0}}},

            {"id": "A6", "name": "ts-order-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 8200, "total_classes": 42, "total_methods": 260}, "complexity": {"avg_wmc": 33.5}, "cohesion": {"avg_lcom": 36.0}, "coupling": {"avg_cbo": 18.0, "avg_rfc": 58.0}}},
            {"id": "A7", "name": "ts-order-other-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 7100, "total_classes": 37, "total_methods": 225}, "complexity": {"avg_wmc": 29.0}, "cohesion": {"avg_lcom": 31.5}, "coupling": {"avg_cbo": 15.5, "avg_rfc": 49.0}}},
            {"id": "A8", "name": "ts-preserve-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 9400, "total_classes": 48, "total_methods": 310}, "complexity": {"avg_wmc": 42.0}, "cohesion": {"avg_lcom": 45.0}, "coupling": {"avg_cbo": 22.0, "avg_rfc": 71.0}}},
            {"id": "A9", "name": "ts-preserve-other-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 8500, "total_classes": 44, "total_methods": 275}, "complexity": {"avg_wmc": 37.5}, "cohesion": {"avg_lcom": 40.0}, "coupling": {"avg_cbo": 19.5, "avg_rfc": 63.0}}},
            {"id": "A10", "name": "ts-travel-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 9100, "total_classes": 46, "total_methods": 295}, "complexity": {"avg_wmc": 40.0}, "cohesion": {"avg_lcom": 43.0}, "coupling": {"avg_cbo": 21.0, "avg_rfc": 67.0}}},
            {"id": "A11", "name": "ts-travel2-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 8100, "total_classes": 41, "total_methods": 260}, "complexity": {"avg_wmc": 35.0}, "cohesion": {"avg_lcom": 38.0}, "coupling": {"avg_cbo": 18.5, "avg_rfc": 59.0}}},
            {"id": "A12", "name": "ts-travel-plan-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 6400, "total_classes": 33, "total_methods": 200}, "complexity": {"avg_wmc": 27.5}, "cohesion": {"avg_lcom": 29.0}, "coupling": {"avg_cbo": 14.2, "avg_rfc": 46.0}}},

            {"id": "A13", "name": "ts-route-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 4500, "total_classes": 23, "total_methods": 145}, "complexity": {"avg_wmc": 19.5}, "cohesion": {"avg_lcom": 20.8}, "coupling": {"avg_cbo": 10.2, "avg_rfc": 33.0}}},
            {"id": "A14", "name": "ts-route-plan-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 5200, "total_classes": 27, "total_methods": 165}, "complexity": {"avg_wmc": 22.8}, "cohesion": {"avg_lcom": 24.0}, "coupling": {"avg_cbo": 11.8, "avg_rfc": 38.5}}},
            {"id": "A15", "name": "ts-train-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 3600, "total_classes": 19, "total_methods": 118}, "complexity": {"avg_wmc": 15.8}, "cohesion": {"avg_lcom": 16.5}, "coupling": {"avg_cbo": 8.1, "avg_rfc": 26.0}}},
            {"id": "A16", "name": "ts-station-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 2800, "total_classes": 15, "total_methods": 90}, "complexity": {"avg_wmc": 12.2}, "cohesion": {"avg_lcom": 12.8}, "coupling": {"avg_cbo": 6.2, "avg_rfc": 20.5}}},
            {"id": "A17", "name": "ts-seat-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 5900, "total_classes": 30, "total_methods": 190}, "complexity": {"avg_wmc": 26.0}, "cohesion": {"avg_lcom": 27.5}, "coupling": {"avg_cbo": 13.8, "avg_rfc": 44.0}}},
            {"id": "A18", "name": "ts-price-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 3300, "total_classes": 17, "total_methods": 105}, "complexity": {"avg_wmc": 14.5}, "cohesion": {"avg_lcom": 15.2}, "coupling": {"avg_cbo": 7.5, "avg_rfc": 24.0}}},
            {"id": "A19", "name": "ts-config-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 2200, "total_classes": 11, "total_methods": 70}, "complexity": {"avg_wmc": 9.8}, "cohesion": {"avg_lcom": 10.0}, "coupling": {"avg_cbo": 5.0, "avg_rfc": 16.0}}},

            {"id": "A20", "name": "ts-payment-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 5100, "total_classes": 26, "total_methods": 160}, "complexity": {"avg_wmc": 22.0}, "cohesion": {"avg_lcom": 23.0}, "coupling": {"avg_cbo": 11.5, "avg_rfc": 37.0}}},
            {"id": "A21", "name": "ts-inside-payment-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 4600, "total_classes": 24, "total_methods": 148}, "complexity": {"avg_wmc": 20.2}, "cohesion": {"avg_lcom": 21.0}, "coupling": {"avg_cbo": 10.5, "avg_rfc": 34.0}}},
            {"id": "A22", "name": "ts-execute-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 3900, "total_classes": 20, "total_methods": 125}, "complexity": {"avg_wmc": 17.0}, "cohesion": {"avg_lcom": 18.0}, "coupling": {"avg_cbo": 8.8, "avg_rfc": 28.0}}},
            {"id": "A23", "name": "ts-cancel-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 4400, "total_classes": 22, "total_methods": 140}, "complexity": {"avg_wmc": 19.0}, "cohesion": {"avg_lcom": 19.8}, "coupling": {"avg_cbo": 9.9, "avg_rfc": 32.0}}},
            {"id": "A24", "name": "ts-rebook-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 4900, "total_classes": 25, "total_methods": 155}, "complexity": {"avg_wmc": 21.5}, "cohesion": {"avg_lcom": 22.4}, "coupling": {"avg_cbo": 11.0, "avg_rfc": 35.5}}},

            {"id": "A25", "name": "ts-assurance-service", "app_type": "processor", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 2100, "total_classes": 11, "total_methods": 65}, "complexity": {"avg_wmc": 9.2}, "cohesion": {"avg_lcom": 9.5}, "coupling": {"avg_cbo": 4.8, "avg_rfc": 15.0}}},
            {"id": "A26", "name": "ts-consign-service", "app_type": "processor", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 2700, "total_classes": 14, "total_methods": 82}, "complexity": {"avg_wmc": 11.8}, "cohesion": {"avg_lcom": 12.0}, "coupling": {"avg_cbo": 6.0, "avg_rfc": 19.5}}},
            {"id": "A27", "name": "ts-consign-price-service", "app_type": "processor", "criticality": "LOW", "hotstandby": False, "code_metrics": {"size": {"total_loc": 1500, "total_classes": 8, "total_methods": 45}, "complexity": {"avg_wmc": 6.5}, "cohesion": {"avg_lcom": 6.8}, "coupling": {"avg_cbo": 3.4, "avg_rfc": 11.0}}},
            {"id": "A28", "name": "ts-food-service", "app_type": "processor", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 3200, "total_classes": 16, "total_methods": 98}, "complexity": {"avg_wmc": 13.8}, "cohesion": {"avg_lcom": 14.0}, "coupling": {"avg_cbo": 7.2, "avg_rfc": 23.0}}},
            {"id": "A29", "name": "ts-food-map-service", "app_type": "processor", "criticality": "LOW", "hotstandby": False, "code_metrics": {"size": {"total_loc": 1800, "total_classes": 9, "total_methods": 54}, "complexity": {"avg_wmc": 8.0}, "cohesion": {"avg_lcom": 8.2}, "coupling": {"avg_cbo": 4.0, "avg_rfc": 13.0}}},

            {"id": "A30", "name": "ts-notification-service", "app_type": "actuator", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 2300, "total_classes": 12, "total_methods": 72}, "complexity": {"avg_wmc": 10.2}, "cohesion": {"avg_lcom": 10.5}, "coupling": {"avg_cbo": 5.2, "avg_rfc": 16.5}}},
            {"id": "A31", "name": "ts-security-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 4100, "total_classes": 21, "total_methods": 130}, "complexity": {"avg_wmc": 18.0}, "cohesion": {"avg_lcom": 19.0}, "coupling": {"avg_cbo": 9.5, "avg_rfc": 30.0}}},
            {"id": "A32", "name": "ts-voucher-service", "app_type": "processor", "criticality": "LOW", "hotstandby": False, "code_metrics": {"size": {"total_loc": 1900, "total_classes": 10, "total_methods": 58}, "complexity": {"avg_wmc": 8.5}, "cohesion": {"avg_lcom": 8.8}, "coupling": {"avg_cbo": 4.2, "avg_rfc": 13.8}}},

            {"id": "A33", "name": "ts-admin-user-service", "app_type": "processor", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 2400, "total_classes": 12, "total_methods": 74}, "complexity": {"avg_wmc": 10.5}, "cohesion": {"avg_lcom": 11.0}, "coupling": {"avg_cbo": 5.5, "avg_rfc": 17.5}}},
            {"id": "A34", "name": "ts-admin-travel-service", "app_type": "processor", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 2600, "total_classes": 13, "total_methods": 80}, "complexity": {"avg_wmc": 11.2}, "cohesion": {"avg_lcom": 11.8}, "coupling": {"avg_cbo": 5.8, "avg_rfc": 18.5}}},
            {"id": "A35", "name": "ts-admin-order-service", "app_type": "processor", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 2800, "total_classes": 14, "total_methods": 86}, "complexity": {"avg_wmc": 12.0}, "cohesion": {"avg_lcom": 12.5}, "coupling": {"avg_cbo": 6.2, "avg_rfc": 19.8}}},
            {"id": "A36", "name": "ts-admin-route-service", "app_type": "processor", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 2300, "total_classes": 11, "total_methods": 70}, "complexity": {"avg_wmc": 10.0}, "cohesion": {"avg_lcom": 10.2}, "coupling": {"avg_cbo": 5.1, "avg_rfc": 16.2}}},
            {"id": "A37", "name": "ts-admin-basic-info-service", "app_type": "processor", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 2500, "total_classes": 12, "total_methods": 76}, "complexity": {"avg_wmc": 10.8}, "cohesion": {"avg_lcom": 11.2}, "coupling": {"avg_cbo": 5.6, "avg_rfc": 17.8}}},

            {"id": "A38", "name": "ts-delivery-service", "app_type": "processor", "criticality": "LOW", "hotstandby": False, "code_metrics": {"size": {"total_loc": 1600, "total_classes": 8, "total_methods": 48}, "complexity": {"avg_wmc": 7.0}, "cohesion": {"avg_lcom": 7.2}, "coupling": {"avg_cbo": 3.6, "avg_rfc": 11.5}}},
            {"id": "A39", "name": "ts-wait-order-service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 3700, "total_classes": 19, "total_methods": 115}, "complexity": {"avg_wmc": 16.0}, "cohesion": {"avg_lcom": 16.8}, "coupling": {"avg_cbo": 8.2, "avg_rfc": 26.5}}},
            {"id": "A40", "name": "ts-news-service", "app_type": "processor", "criticality": "LOW", "hotstandby": False, "code_metrics": {"size": {"total_loc": 1100, "total_classes": 5, "total_methods": 32}, "complexity": {"avg_wmc": 5.2}, "cohesion": {"avg_lcom": 5.5}, "coupling": {"avg_cbo": 2.8, "avg_rfc": 9.0}}}
        ]

        relationships = [
            # Ingress & Gateway Publishers
            {"source": "A0", "target": "T12", "type": "PUBLISHES_TO"},
            {"source": "A0", "target": "T13", "type": "PUBLISHES_TO"},
            {"source": "A1", "target": "T2", "type": "PUBLISHES_TO"},
            {"source": "A1", "target": "T3", "type": "PUBLISHES_TO"},
            {"source": "A1", "target": "T11", "type": "PUBLISHES_TO"},
            {"source": "A1", "target": "T25", "type": "PUBLISHES_TO"},

            # Auth, User & Security
            {"source": "A2", "target": "T12", "type": "PUBLISHES_TO"},
            {"source": "A3", "target": "T13", "type": "PUBLISHES_TO"},
            {"source": "A5", "target": "T19", "type": "PUBLISHES_TO"},
            {"source": "A31", "target": "T11", "type": "PUBLISHES_TO"},

            # Preservation & Travel Pipeline
            {"source": "A8", "target": "T0", "type": "PUBLISHES_TO"},
            {"source": "A8", "target": "T4", "type": "PUBLISHES_TO"},
            {"source": "A8", "target": "T27", "type": "PUBLISHES_TO"},
            {"source": "A9", "target": "T0", "type": "PUBLISHES_TO"},
            {"source": "A10", "target": "T2", "type": "PUBLISHES_TO"},
            {"source": "A11", "target": "T2", "type": "PUBLISHES_TO"},

            # Order & Payment Pipeline
            {"source": "A6", "target": "T0", "type": "PUBLISHES_TO"},
            {"source": "A6", "target": "T1", "type": "PUBLISHES_TO"},
            {"source": "A7", "target": "T0", "type": "PUBLISHES_TO"},
            {"source": "A20", "target": "T5", "type": "PUBLISHES_TO"},
            {"source": "A20", "target": "T6", "type": "PUBLISHES_TO"},
            {"source": "A21", "target": "T28", "type": "PUBLISHES_TO"},
            {"source": "A23", "target": "T1", "type": "PUBLISHES_TO"},
            {"source": "A23", "target": "T6", "type": "PUBLISHES_TO"},

            # Ancillary Services (Food, Consign, Voucher, Notification)
            {"source": "A26", "target": "T7", "type": "PUBLISHES_TO"},
            {"source": "A28", "target": "T8", "type": "PUBLISHES_TO"},
            {"source": "A30", "target": "T9", "type": "PUBLISHES_TO"},
            {"source": "A30", "target": "T10", "type": "PUBLISHES_TO"},
            {"source": "A32", "target": "T18", "type": "PUBLISHES_TO"},
            {"source": "A39", "target": "T29", "type": "PUBLISHES_TO"},

            # Admin & Infrastructure Management
            {"source": "A13", "target": "T14", "type": "PUBLISHES_TO"},
            {"source": "A15", "target": "T15", "type": "PUBLISHES_TO"},
            {"source": "A18", "target": "T16", "type": "PUBLISHES_TO"},
            {"source": "A16", "target": "T17", "type": "PUBLISHES_TO"},
            {"source": "A33", "target": "T23", "type": "PUBLISHES_TO"},
            {"source": "A34", "target": "T21", "type": "PUBLISHES_TO"},
            {"source": "A35", "target": "T22", "type": "PUBLISHES_TO"},
            {"source": "A37", "target": "T20", "type": "PUBLISHES_TO"},

            # Topic Subscriptions (SUBSCRIBES_TO)
            {"source": "A1", "target": "T12", "type": "SUBSCRIBES_TO"},
            {"source": "A2", "target": "T13", "type": "SUBSCRIBES_TO"},
            {"source": "A6", "target": "T0", "type": "SUBSCRIBES_TO"},
            {"source": "A6", "target": "T4", "type": "SUBSCRIBES_TO"},
            {"source": "A6", "target": "T19", "type": "SUBSCRIBES_TO"},
            {"source": "A8", "target": "T2", "type": "SUBSCRIBES_TO"},
            {"source": "A8", "target": "T11", "type": "SUBSCRIBES_TO"},
            {"source": "A10", "target": "T14", "type": "SUBSCRIBES_TO"},
            {"source": "A10", "target": "T15", "type": "SUBSCRIBES_TO"},
            {"source": "A17", "target": "T4", "type": "SUBSCRIBES_TO"},
            {"source": "A20", "target": "T0", "type": "SUBSCRIBES_TO"},
            {"source": "A21", "target": "T5", "type": "SUBSCRIBES_TO"},
            {"source": "A23", "target": "T0", "type": "SUBSCRIBES_TO"},
            {"source": "A30", "target": "T0", "type": "SUBSCRIBES_TO"},
            {"source": "A30", "target": "T5", "type": "SUBSCRIBES_TO"},
            {"source": "A31", "target": "T12", "type": "SUBSCRIBES_TO"},
            {"source": "A39", "target": "T0", "type": "SUBSCRIBES_TO"},

            # Shared Library dependencies (USES)
            {"source": "A0", "target": "L0", "type": "USES"},
            {"source": "A0", "target": "L7", "type": "USES"},
            {"source": "A1", "target": "L0", "type": "USES"},
            {"source": "A1", "target": "L1", "type": "USES"},
            {"source": "A1", "target": "L6", "type": "USES"},
            {"source": "A1", "target": "L7", "type": "USES"},
            {"source": "A2", "target": "L0", "type": "USES"},
            {"source": "A2", "target": "L2", "type": "USES"},
            {"source": "A2", "target": "L6", "type": "USES"},
            {"source": "A6", "target": "L0", "type": "USES"},
            {"source": "A6", "target": "L1", "type": "USES"},
            {"source": "A6", "target": "L2", "type": "USES"},
            {"source": "A6", "target": "L3", "type": "USES"},
            {"source": "A6", "target": "L4", "type": "USES"},
            {"source": "A6", "target": "L5", "type": "USES"},
            {"source": "A8", "target": "L0", "type": "USES"},
            {"source": "A8", "target": "L1", "type": "USES"},
            {"source": "A8", "target": "L2", "type": "USES"},
            {"source": "A8", "target": "L5", "type": "USES"},
            {"source": "A10", "target": "L0", "type": "USES"},
            {"source": "A10", "target": "L1", "type": "USES"},
            {"source": "A10", "target": "L2", "type": "USES"},
            {"source": "A10", "target": "L3", "type": "USES"},
            {"source": "A10", "target": "L4", "type": "USES"},
            {"source": "A17", "target": "L0", "type": "USES"},
            {"source": "A17", "target": "L2", "type": "USES"},
            {"source": "A17", "target": "L4", "type": "USES"},
            {"source": "A20", "target": "L0", "type": "USES"},
            {"source": "A20", "target": "L2", "type": "USES"},
            {"source": "A20", "target": "L3", "type": "USES"},
            {"source": "A30", "target": "L0", "type": "USES"},
            {"source": "A30", "target": "L5", "type": "USES"},
            {"source": "A31", "target": "L0", "type": "USES"},
            {"source": "A31", "target": "L6", "type": "USES"},

            # Node Hosting (RUNS_ON)
            {"source": "A0", "target": "N7", "type": "RUNS_ON"},
            {"source": "A1", "target": "N7", "type": "RUNS_ON"},
            {"source": "A2", "target": "N1", "type": "RUNS_ON"},
            {"source": "A3", "target": "N1", "type": "RUNS_ON"},
            {"source": "A4", "target": "N1", "type": "RUNS_ON"},
            {"source": "A5", "target": "N1", "type": "RUNS_ON"},
            {"source": "A6", "target": "N1", "type": "RUNS_ON"},
            {"source": "A7", "target": "N1", "type": "RUNS_ON"},
            {"source": "A8", "target": "N1", "type": "RUNS_ON"},
            {"source": "A9", "target": "N1", "type": "RUNS_ON"},
            {"source": "A10", "target": "N2", "type": "RUNS_ON"},
            {"source": "A11", "target": "N2", "type": "RUNS_ON"},
            {"source": "A12", "target": "N2", "type": "RUNS_ON"},
            {"source": "A13", "target": "N2", "type": "RUNS_ON"},
            {"source": "A14", "target": "N2", "type": "RUNS_ON"},
            {"source": "A15", "target": "N2", "type": "RUNS_ON"},
            {"source": "A16", "target": "N2", "type": "RUNS_ON"},
            {"source": "A17", "target": "N2", "type": "RUNS_ON"},
            {"source": "A18", "target": "N2", "type": "RUNS_ON"},
            {"source": "A19", "target": "N2", "type": "RUNS_ON"},
            {"source": "A20", "target": "N3", "type": "RUNS_ON"},
            {"source": "A21", "target": "N3", "type": "RUNS_ON"},
            {"source": "A22", "target": "N3", "type": "RUNS_ON"},
            {"source": "A23", "target": "N3", "type": "RUNS_ON"},
            {"source": "A24", "target": "N3", "type": "RUNS_ON"},
            {"source": "A25", "target": "N3", "type": "RUNS_ON"},
            {"source": "A26", "target": "N3", "type": "RUNS_ON"},
            {"source": "A27", "target": "N3", "type": "RUNS_ON"},
            {"source": "A28", "target": "N3", "type": "RUNS_ON"},
            {"source": "A29", "target": "N3", "type": "RUNS_ON"},
            {"source": "A30", "target": "N3", "type": "RUNS_ON"},
            {"source": "A31", "target": "N1", "type": "RUNS_ON"},
            {"source": "A32", "target": "N3", "type": "RUNS_ON"},
            {"source": "A33", "target": "N0", "type": "RUNS_ON"},
            {"source": "A34", "target": "N0", "type": "RUNS_ON"},
            {"source": "A35", "target": "N0", "type": "RUNS_ON"},
            {"source": "A36", "target": "N0", "type": "RUNS_ON"},
            {"source": "A37", "target": "N0", "type": "RUNS_ON"},
            {"source": "A38", "target": "N3", "type": "RUNS_ON"},
            {"source": "A39", "target": "N1", "type": "RUNS_ON"},
            {"source": "A40", "target": "N3", "type": "RUNS_ON"},

            {"source": "B0", "target": "N1", "type": "RUNS_ON"},
            {"source": "B1", "target": "N6", "type": "RUNS_ON"},
            {"source": "B2", "target": "N0", "type": "RUNS_ON"},

            # Broker Routing (ROUTES)
            {"source": "B0", "target": "T0", "type": "ROUTES"},
            {"source": "B0", "target": "T1", "type": "ROUTES"},
            {"source": "B0", "target": "T5", "type": "ROUTES"},
            {"source": "B0", "target": "T6", "type": "ROUTES"},
            {"source": "B0", "target": "T7", "type": "ROUTES"},
            {"source": "B0", "target": "T8", "type": "ROUTES"},
            {"source": "B0", "target": "T9", "type": "ROUTES"},
            {"source": "B0", "target": "T10", "type": "ROUTES"},
            {"source": "B0", "target": "T27", "type": "ROUTES"},
            {"source": "B0", "target": "T28", "type": "ROUTES"},
            {"source": "B0", "target": "T29", "type": "ROUTES"},
            {"source": "B1", "target": "T4", "type": "ROUTES"},
            {"source": "B1", "target": "T19", "type": "ROUTES"},
            {"source": "B1", "target": "T26", "type": "ROUTES"},
            {"source": "B2", "target": "T2", "type": "ROUTES"},
            {"source": "B2", "target": "T3", "type": "ROUTES"},
            {"source": "B2", "target": "T11", "type": "ROUTES"},
            {"source": "B2", "target": "T14", "type": "ROUTES"},
            {"source": "B2", "target": "T15", "type": "ROUTES"},
            {"source": "B2", "target": "T16", "type": "ROUTES"},
            {"source": "B2", "target": "T17", "type": "ROUTES"},

            # Hardware Interconnects (CONNECTS_TO)
            {"source": "N7", "target": "N1", "type": "CONNECTS_TO"},
            {"source": "N1", "target": "N2", "type": "CONNECTS_TO"},
            {"source": "N1", "target": "N3", "type": "CONNECTS_TO"},
            {"source": "N2", "target": "N4", "type": "CONNECTS_TO"},
            {"source": "N2", "target": "N5", "type": "CONNECTS_TO"},
            {"source": "N3", "target": "N6", "type": "CONNECTS_TO"},
            {"source": "N0", "target": "N1", "type": "CONNECTS_TO"},
            {"source": "N0", "target": "N2", "type": "CONNECTS_TO"}
        ]

        metadata = {
            "scale": {
                "apps": len(applications),
                "topics": len(topics),
                "brokers": len(brokers),
                "nodes": len(nodes),
                "libs": len(libraries)
            },
            "seed": 2026,
            "generation_mode": "realworld_open_source",
            "domain": "trainticket_microservices",
            "scenario": "production_trainticket_benchmark_microservices_mesh",
            "description": "Authentic production-grade Train-Ticket benchmark microservices mesh (Fudan University benchmark) containing order, travel, preserve, route, seat, payment, food, security, and admin services."
        }

        return {
            "metadata": metadata,
            "nodes": nodes,
            "brokers": brokers,
            "topics": topics,
            "applications": applications,
            "libraries": libraries,
            "relationships": _to_canonical_relationships(relationships)
        }

    @staticmethod
    def create_homeassistant_topology() -> Dict[str, Any]:
        """
        Creates an authentic Home Assistant smart home automation topology graph,
        based on the open-source Home Assistant ecosystem (AsyncIO EventBus, MQTT, WebSockets).

        Topology overview:
        - 24 Applications across Core, Automation, Devices/Integrations, Climate, Safety, and Frontend
        - 22 Topics with explicit event bus and MQTT QoS configurations
        - 3 Brokers (AsyncIO in-memory event bus, Mosquitto MQTT broker, WebSocket stream router)
        - 6 Deployment Nodes (Appliance host, Zigbee radio, Z-Wave hub, TrueNAS storage, ESPHome bridge, Wall tablet)
        - 8 Shared Libraries (aiohttp, voluptuous, sqlalchemy, paho-mqtt, zeroconf, webrtc, etc.)
        - Realistic SonarQube code quality metrics for applications
        """
        nodes = [
            {"id": "N0", "name": "ha-host-appliance", "type": "Node", "spec": "ODROID-N2+ ARM64 / Home Assistant Green"},
            {"id": "N1", "name": "zigbee-matter-radio", "type": "Node", "spec": "Silicon Labs EFR32MG21 USB Coordinator"},
            {"id": "N2", "name": "zwave-controller-hub", "type": "Node", "spec": "Aeotec Z-Stick Gen7 / 700 Series Controller"},
            {"id": "N3", "name": "edge-nas-storage", "type": "Node", "spec": "Local TrueNAS Storage Host / NVMe"},
            {"id": "N4", "name": "edge-sensor-bridge", "type": "Node", "spec": "ESP32 ESPHome Distributed Gateway Hub"},
            {"id": "N5", "name": "mobile-tablet-wallmount", "type": "Node", "spec": "Android Wall Display / Lovelace Kiosk"}
        ]

        brokers = [
            {"id": "B0", "name": "hass-core-eventbus", "type": "Broker", "protocol": "AsyncIO-EventBus"},
            {"id": "B1", "name": "mosquitto-mqtt-broker", "type": "Broker", "protocol": "MQTT"},
            {"id": "B2", "name": "websocket-push-router", "type": "Broker", "protocol": "WebSocket"}
        ]

        topics = [
            {"id": "T0", "name": "event.state_changed", "size": 512, "qos": {"durability": "VOLATILE", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 100.0, "criticality": "HIGH"},
            {"id": "T1", "name": "event.call_service", "size": 256, "qos": {"durability": "VOLATILE", "reliability": "RELIABLE", "transport_priority": "CRITICAL"}, "frequency": 50.0, "criticality": "HIGH"},
            {"id": "T2", "name": "alarm.hazard.fire_smoke", "size": 128, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "HIGHEST"}, "frequency": 1.0, "criticality": "HIGH"},
            {"id": "T3", "name": "alarm.hazard.water_leak", "size": 128, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "HIGHEST"}, "frequency": 1.0, "criticality": "HIGH"},
            {"id": "T4", "name": "alarm.security.intrusion", "size": 256, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "HIGHEST"}, "frequency": 2.0, "criticality": "HIGH"},
            {"id": "T5", "name": "device.zigbee.raw_state", "size": 256, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "HIGH"}, "frequency": 60.0, "criticality": "HIGH"},
            {"id": "T6", "name": "device.zwave.raw_state", "size": 256, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "HIGH"}, "frequency": 40.0, "criticality": "HIGH"},
            {"id": "T7", "name": "device.esphome.telemetry", "size": 512, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "MEDIUM"}, "frequency": 30.0, "criticality": "MEDIUM"},
            {"id": "T8", "name": "command.actuate.valve", "size": 128, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "HIGHEST"}, "frequency": 0.5, "criticality": "HIGH"},
            {"id": "T9", "name": "command.actuate.siren", "size": 128, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "HIGHEST"}, "frequency": 1.0, "criticality": "HIGH"},
            {"id": "T10", "name": "command.actuate.switch", "size": 128, "qos": {"durability": "VOLATILE", "reliability": "RELIABLE", "transport_priority": "MEDIUM"}, "frequency": 20.0, "criticality": "MEDIUM"},
            {"id": "T11", "name": "climate.thermostat.status", "size": 256, "qos": {"durability": "VOLATILE", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 10.0, "criticality": "HIGH"},
            {"id": "T12", "name": "climate.hvac.command", "size": 128, "qos": {"durability": "VOLATILE", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 5.0, "criticality": "HIGH"},
            {"id": "T13", "name": "recorder.db.write_queue", "size": 2048, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "MEDIUM"}, "frequency": 80.0, "criticality": "MEDIUM"},
            {"id": "T14", "name": "energy.meter.consumption", "size": 512, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "LOW"}, "frequency": 10.0, "criticality": "LOW"},
            {"id": "T15", "name": "energy.solar.production", "size": 512, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "LOW"}, "frequency": 10.0, "criticality": "LOW"},
            {"id": "T16", "name": "presence.occupancy.detected", "size": 256, "qos": {"durability": "VOLATILE", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 15.0, "criticality": "HIGH"},
            {"id": "T17", "name": "notification.push.alert", "size": 1024, "qos": {"durability": "VOLATILE", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 5.0, "criticality": "HIGH"},
            {"id": "T18", "name": "media.playback.status", "size": 256, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "LOW"}, "frequency": 2.0, "criticality": "LOW"},
            {"id": "T19", "name": "camera.motion.event", "size": 1024, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "HIGH"}, "frequency": 5.0, "criticality": "HIGH"},
            {"id": "T20", "name": "system.health.heartbeat", "size": 128, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "LOW"}, "frequency": 1.0, "criticality": "LOW"},
            {"id": "T21", "name": "automation.trigger.event", "size": 512, "qos": {"durability": "VOLATILE", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 30.0, "criticality": "HIGH"}
        ]

        libraries = [
            {"id": "L0", "name": "aiohttp", "version": "3.9.5", "description": "Async HTTP client & WebSocket server runtime"},
            {"id": "L1", "name": "voluptuous", "version": "0.15.2", "description": "Data schema validation and typing library"},
            {"id": "L2", "name": "sqlalchemy", "version": "2.0.30", "description": "Database abstraction layer and ORM"},
            {"id": "L3", "name": "paho-mqtt", "version": "2.1.0", "description": "MQTT client protocol implementation"},
            {"id": "L4", "name": "zeroconf", "version": "0.132.2", "description": "Pure Python mDNS service discovery"},
            {"id": "L5", "name": "webrtc-streamer", "version": "0.8.4", "description": "Live camera low-latency WebRTC client helper"},
            {"id": "L6", "name": "pyroute2", "version": "0.7.12", "description": "Linux network routing and netlink socket client"},
            {"id": "L7", "name": "homeassistant-frontend", "version": "20240710.0", "description": "Lovelace reactive UI web client asset pack"}
        ]

        applications = [
            # Core & Automation
            {"id": "A0", "name": "core_state_machine", "app_type": "sensor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 5800, "total_classes": 26, "total_methods": 175}, "complexity": {"avg_wmc": 24.5}, "cohesion": {"avg_lcom": 22.0}, "coupling": {"avg_cbo": 12.0, "avg_rfc": 39.5}}},
            {"id": "A1", "name": "automation_engine", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 6400, "total_classes": 30, "total_methods": 190}, "complexity": {"avg_wmc": 28.0}, "cohesion": {"avg_lcom": 26.5}, "coupling": {"avg_cbo": 14.2, "avg_rfc": 45.0}}},
            {"id": "A2", "name": "script_execution_service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 3200, "total_classes": 15, "total_methods": 95}, "complexity": {"avg_wmc": 16.0}, "cohesion": {"avg_lcom": 14.2}, "coupling": {"avg_cbo": 8.1, "avg_rfc": 25.4}}},
            {"id": "A3", "name": "safety_hazard_monitor", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 2900, "total_classes": 14, "total_methods": 88}, "complexity": {"avg_wmc": 15.5}, "cohesion": {"avg_lcom": 13.0}, "coupling": {"avg_cbo": 7.5, "avg_rfc": 23.0}}},
            {"id": "A4", "name": "alarm_control_panel", "app_type": "actuator", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 3400, "total_classes": 16, "total_methods": 105}, "complexity": {"avg_wmc": 17.2}, "cohesion": {"avg_lcom": 15.8}, "coupling": {"avg_cbo": 8.9, "avg_rfc": 28.1}}},
            {"id": "A5", "name": "emergency_actuation_manager", "app_type": "actuator", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 2100, "total_classes": 11, "total_methods": 65}, "complexity": {"avg_wmc": 12.0}, "cohesion": {"avg_lcom": 10.5}, "coupling": {"avg_cbo": 6.2, "avg_rfc": 19.5}}},

            # Device Integrations
            {"id": "A6", "name": "zha_zigbee_gateway", "app_type": "sensor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 4800, "total_classes": 24, "total_methods": 150}, "complexity": {"avg_wmc": 21.4}, "cohesion": {"avg_lcom": 20.2}, "coupling": {"avg_cbo": 11.0, "avg_rfc": 35.0}}},
            {"id": "A7", "name": "zwave_js_supervisor", "app_type": "sensor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 4300, "total_classes": 21, "total_methods": 135}, "complexity": {"avg_wmc": 19.8}, "cohesion": {"avg_lcom": 18.5}, "coupling": {"avg_cbo": 10.2, "avg_rfc": 32.4}}},
            {"id": "A8", "name": "esphome_device_manager", "app_type": "sensor", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 3100, "total_classes": 16, "total_methods": 92}, "complexity": {"avg_wmc": 14.5}, "cohesion": {"avg_lcom": 13.0}, "coupling": {"avg_cbo": 7.4, "avg_rfc": 23.8}}},
            {"id": "A9", "name": "mqtt_event_bridge", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 2600, "total_classes": 13, "total_methods": 80}, "complexity": {"avg_wmc": 13.0}, "cohesion": {"avg_lcom": 11.8}, "coupling": {"avg_cbo": 6.8, "avg_rfc": 21.5}}},
            {"id": "A10", "name": "camera_stream_manager", "app_type": "sensor", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 3500, "total_classes": 17, "total_methods": 110}, "complexity": {"avg_wmc": 16.8}, "cohesion": {"avg_lcom": 15.0}, "coupling": {"avg_cbo": 8.5, "avg_rfc": 27.0}}},
            {"id": "A11", "name": "matter_thread_bridge", "app_type": "sensor", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 3900, "total_classes": 19, "total_methods": 125}, "complexity": {"avg_wmc": 18.2}, "cohesion": {"avg_lcom": 16.9}, "coupling": {"avg_cbo": 9.4, "avg_rfc": 30.1}}},

            # Actuation & Environment
            {"id": "A12", "name": "climate_hvac_controller", "app_type": "actuator", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 2800, "total_classes": 14, "total_methods": 85}, "complexity": {"avg_wmc": 14.0}, "cohesion": {"avg_lcom": 12.5}, "coupling": {"avg_cbo": 7.1, "avg_rfc": 22.8}}},
            {"id": "A13", "name": "light_switch_controller", "app_type": "actuator", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 2200, "total_classes": 11, "total_methods": 70}, "complexity": {"avg_wmc": 11.0}, "cohesion": {"avg_lcom": 9.8}, "coupling": {"avg_cbo": 5.8, "avg_rfc": 18.2}}},
            {"id": "A14", "name": "presence_detection_engine", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 2500, "total_classes": 12, "total_methods": 78}, "complexity": {"avg_wmc": 12.5}, "cohesion": {"avg_lcom": 11.2}, "coupling": {"avg_cbo": 6.5, "avg_rfc": 20.4}}},
            {"id": "A15", "name": "media_player_coordinator", "app_type": "actuator", "criticality": "LOW", "hotstandby": False, "code_metrics": {"size": {"total_loc": 1900, "total_classes": 10, "total_methods": 60}, "complexity": {"avg_wmc": 9.5}, "cohesion": {"avg_lcom": 8.5}, "coupling": {"avg_cbo": 5.0, "avg_rfc": 15.8}}},

            # Storage, Analytics & Notifications
            {"id": "A16", "name": "recorder_persistence_worker", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 4100, "total_classes": 20, "total_methods": 130}, "complexity": {"avg_wmc": 19.0}, "cohesion": {"avg_lcom": 17.5}, "coupling": {"avg_cbo": 9.8, "avg_rfc": 31.0}}},
            {"id": "A17", "name": "energy_analytics_worker", "app_type": "processor", "criticality": "LOW", "hotstandby": False, "code_metrics": {"size": {"total_loc": 2700, "total_classes": 13, "total_methods": 82}, "complexity": {"avg_wmc": 13.5}, "cohesion": {"avg_lcom": 12.0}, "coupling": {"avg_cbo": 6.9, "avg_rfc": 22.0}}},
            {"id": "A18", "name": "logbook_history_service", "app_type": "processor", "criticality": "LOW", "hotstandby": False, "code_metrics": {"size": {"total_loc": 1800, "total_classes": 9, "total_methods": 55}, "complexity": {"avg_wmc": 9.0}, "cohesion": {"avg_lcom": 8.0}, "coupling": {"avg_cbo": 4.8, "avg_rfc": 15.0}}},
            {"id": "A19", "name": "notification_dispatcher", "app_type": "actuator", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 2400, "total_classes": 12, "total_methods": 75}, "complexity": {"avg_wmc": 12.0}, "cohesion": {"avg_lcom": 10.8}, "coupling": {"avg_cbo": 6.2, "avg_rfc": 19.8}}},

            # Ingress & UI
            {"id": "A20", "name": "websocket_api_server", "app_type": "sensor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 3600, "total_classes": 18, "total_methods": 115}, "complexity": {"avg_wmc": 17.0}, "cohesion": {"avg_lcom": 15.5}, "coupling": {"avg_cbo": 8.8, "avg_rfc": 28.0}}},
            {"id": "A21", "name": "http_api_ingress", "app_type": "sensor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 3100, "total_classes": 15, "total_methods": 96}, "complexity": {"avg_wmc": 15.2}, "cohesion": {"avg_lcom": 13.8}, "coupling": {"avg_cbo": 7.9, "avg_rfc": 25.0}}},
            {"id": "A22", "name": "lovelace_ui_dashboard", "app_type": "actuator", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 4600, "total_classes": 23, "total_methods": 145}, "complexity": {"avg_wmc": 20.5}, "cohesion": {"avg_lcom": 19.0}, "coupling": {"avg_cbo": 10.5, "avg_rfc": 33.5}}},
            {"id": "A23", "name": "cloud_remote_agent", "app_type": "sensor", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 2300, "total_classes": 11, "total_methods": 72}, "complexity": {"avg_wmc": 11.5}, "cohesion": {"avg_lcom": 10.2}, "coupling": {"avg_cbo": 5.9, "avg_rfc": 18.6}}}
        ]

        relationships = [
            # EventBus routing (B0 routes internal topics)
            {"source": "B0", "target": "T0", "type": "ROUTES"},
            {"source": "B0", "target": "T1", "type": "ROUTES"},
            {"source": "B0", "target": "T2", "type": "ROUTES"},
            {"source": "B0", "target": "T3", "type": "ROUTES"},
            {"source": "B0", "target": "T4", "type": "ROUTES"},
            {"source": "B0", "target": "T11", "type": "ROUTES"},
            {"source": "B0", "target": "T13", "type": "ROUTES"},
            {"source": "B0", "target": "T14", "type": "ROUTES"},
            {"source": "B0", "target": "T15", "type": "ROUTES"},
            {"source": "B0", "target": "T16", "type": "ROUTES"},
            {"source": "B0", "target": "T17", "type": "ROUTES"},
            {"source": "B0", "target": "T20", "type": "ROUTES"},
            {"source": "B0", "target": "T21", "type": "ROUTES"},

            # Mosquitto MQTT broker routing (B1 routes device & field topics)
            {"source": "B1", "target": "T5", "type": "ROUTES"},
            {"source": "B1", "target": "T6", "type": "ROUTES"},
            {"source": "B1", "target": "T7", "type": "ROUTES"},
            {"source": "B1", "target": "T8", "type": "ROUTES"},
            {"source": "B1", "target": "T9", "type": "ROUTES"},
            {"source": "B1", "target": "T10", "type": "ROUTES"},
            {"source": "B1", "target": "T12", "type": "ROUTES"},

            # WebSocket router (B2 routes UI & push streams)
            {"source": "B2", "target": "T0", "type": "ROUTES"},
            {"source": "B2", "target": "T18", "type": "ROUTES"},
            {"source": "B2", "target": "T19", "type": "ROUTES"},

            # Device and integration publishers
            {"source": "A6", "target": "T5", "type": "PUBLISHES_TO"},
            {"source": "A7", "target": "T6", "type": "PUBLISHES_TO"},
            {"source": "A8", "target": "T7", "type": "PUBLISHES_TO"},
            {"source": "A9", "target": "T0", "type": "PUBLISHES_TO"},
            {"source": "A6", "target": "T0", "type": "PUBLISHES_TO"},
            {"source": "A7", "target": "T0", "type": "PUBLISHES_TO"},
            {"source": "A10", "target": "T19", "type": "PUBLISHES_TO"},
            {"source": "A11", "target": "T0", "type": "PUBLISHES_TO"},
            {"source": "A14", "target": "T16", "type": "PUBLISHES_TO"},
            {"source": "A8", "target": "T14", "type": "PUBLISHES_TO"},
            {"source": "A8", "target": "T15", "type": "PUBLISHES_TO"},

            # Core state machine & automation publish/subscribe
            {"source": "A0", "target": "T0", "type": "SUBSCRIBES_TO"},
            {"source": "A0", "target": "T1", "type": "SUBSCRIBES_TO"},
            {"source": "A0", "target": "T20", "type": "PUBLISHES_TO"},
            {"source": "A1", "target": "T0", "type": "SUBSCRIBES_TO"},
            {"source": "A1", "target": "T16", "type": "SUBSCRIBES_TO"},
            {"source": "A1", "target": "T21", "type": "PUBLISHES_TO"},
            {"source": "A1", "target": "T1", "type": "PUBLISHES_TO"},
            {"source": "A2", "target": "T1", "type": "SUBSCRIBES_TO"},
            {"source": "A2", "target": "T10", "type": "PUBLISHES_TO"},
            {"source": "A2", "target": "T12", "type": "PUBLISHES_TO"},

            # Safety and hazard detection loop (Critical SPOF path)
            {"source": "A3", "target": "T0", "type": "SUBSCRIBES_TO"},
            {"source": "A3", "target": "T2", "type": "PUBLISHES_TO"},
            {"source": "A3", "target": "T3", "type": "PUBLISHES_TO"},
            {"source": "A3", "target": "T4", "type": "PUBLISHES_TO"},
            {"source": "A4", "target": "T4", "type": "SUBSCRIBES_TO"},
            {"source": "A4", "target": "T9", "type": "PUBLISHES_TO"},
            {"source": "A4", "target": "T17", "type": "PUBLISHES_TO"},
            {"source": "A5", "target": "T3", "type": "SUBSCRIBES_TO"},
            {"source": "A5", "target": "T2", "type": "SUBSCRIBES_TO"},
            {"source": "A5", "target": "T8", "type": "PUBLISHES_TO"},
            {"source": "A5", "target": "T17", "type": "PUBLISHES_TO"},

            # Climate & Lighting actuation
            {"source": "A12", "target": "T11", "type": "SUBSCRIBES_TO"},
            {"source": "A12", "target": "T12", "type": "SUBSCRIBES_TO"},
            {"source": "A12", "target": "T0", "type": "PUBLISHES_TO"},
            {"source": "A13", "target": "T10", "type": "SUBSCRIBES_TO"},
            {"source": "A13", "target": "T0", "type": "PUBLISHES_TO"},
            {"source": "A15", "target": "T18", "type": "PUBLISHES_TO"},

            # Storage & Analytics subscribers
            {"source": "A16", "target": "T0", "type": "SUBSCRIBES_TO"},
            {"source": "A16", "target": "T13", "type": "PUBLISHES_TO"},
            {"source": "A17", "target": "T14", "type": "SUBSCRIBES_TO"},
            {"source": "A17", "target": "T15", "type": "SUBSCRIBES_TO"},
            {"source": "A18", "target": "T0", "type": "SUBSCRIBES_TO"},
            {"source": "A19", "target": "T17", "type": "SUBSCRIBES_TO"},

            # Web / Ingress / UI
            {"source": "A20", "target": "T0", "type": "SUBSCRIBES_TO"},
            {"source": "A20", "target": "T1", "type": "PUBLISHES_TO"},
            {"source": "A21", "target": "T1", "type": "PUBLISHES_TO"},
            {"source": "A22", "target": "T0", "type": "SUBSCRIBES_TO"},
            {"source": "A22", "target": "T1", "type": "PUBLISHES_TO"},
            {"source": "A23", "target": "T1", "type": "PUBLISHES_TO"},
            {"source": "A23", "target": "T17", "type": "SUBSCRIBES_TO"},

            # Deployment (RUNS_ON)
            {"source": "A0", "target": "N0", "type": "RUNS_ON"},
            {"source": "A1", "target": "N0", "type": "RUNS_ON"},
            {"source": "A2", "target": "N0", "type": "RUNS_ON"},
            {"source": "A3", "target": "N0", "type": "RUNS_ON"},
            {"source": "A4", "target": "N0", "type": "RUNS_ON"},
            {"source": "A5", "target": "N0", "type": "RUNS_ON"},
            {"source": "B0", "target": "N0", "type": "RUNS_ON"},
            {"source": "B1", "target": "N0", "type": "RUNS_ON"},
            {"source": "B2", "target": "N0", "type": "RUNS_ON"},

            {"source": "A6", "target": "N1", "type": "RUNS_ON"},
            {"source": "A11", "target": "N1", "type": "RUNS_ON"},
            {"source": "A7", "target": "N2", "type": "RUNS_ON"},
            {"source": "A16", "target": "N3", "type": "RUNS_ON"},
            {"source": "A17", "target": "N3", "type": "RUNS_ON"},
            {"source": "A18", "target": "N3", "type": "RUNS_ON"},
            {"source": "A8", "target": "N4", "type": "RUNS_ON"},
            {"source": "A9", "target": "N0", "type": "RUNS_ON"},
            {"source": "A10", "target": "N0", "type": "RUNS_ON"},
            {"source": "A12", "target": "N0", "type": "RUNS_ON"},
            {"source": "A13", "target": "N0", "type": "RUNS_ON"},
            {"source": "A14", "target": "N0", "type": "RUNS_ON"},
            {"source": "A15", "target": "N0", "type": "RUNS_ON"},
            {"source": "A19", "target": "N0", "type": "RUNS_ON"},
            {"source": "A20", "target": "N0", "type": "RUNS_ON"},
            {"source": "A21", "target": "N0", "type": "RUNS_ON"},
            {"source": "A22", "target": "N5", "type": "RUNS_ON"},
            {"source": "A23", "target": "N0", "type": "RUNS_ON"},

            # Library Dependencies (USES)
            {"source": "A0", "target": "L1", "type": "USES"},
            {"source": "A1", "target": "L1", "type": "USES"},
            {"source": "A9", "target": "L3", "type": "USES"},
            {"source": "A16", "target": "L2", "type": "USES"},
            {"source": "A17", "target": "L2", "type": "USES"},
            {"source": "A18", "target": "L2", "type": "USES"},
            {"source": "A20", "target": "L0", "type": "USES"},
            {"source": "A21", "target": "L0", "type": "USES"},
            {"source": "A10", "target": "L5", "type": "USES"},
            {"source": "A6", "target": "L4", "type": "USES"},
            {"source": "A8", "target": "L4", "type": "USES"},
            {"source": "A22", "target": "L7", "type": "USES"},
            {"source": "A0", "target": "L6", "type": "USES"},

            # Hardware Interconnects (CONNECTS_TO)
            {"source": "N0", "target": "N1", "type": "CONNECTS_TO"},
            {"source": "N0", "target": "N2", "type": "CONNECTS_TO"},
            {"source": "N0", "target": "N3", "type": "CONNECTS_TO"},
            {"source": "N0", "target": "N4", "type": "CONNECTS_TO"},
            {"source": "N0", "target": "N5", "type": "CONNECTS_TO"}
        ]

        metadata = {
            "scale": {
                "apps": len(applications),
                "topics": len(topics),
                "brokers": len(brokers),
                "nodes": len(nodes),
                "libs": len(libraries)
            },
            "seed": 2026,
            "generation_mode": "realworld_open_source",
            "domain": "homeassistant_iot",
            "scenario": "production_homeassistant_smarthome_mesh",
            "description": "Authentic production-grade Home Assistant smart home IoT topology (open-source local-first home automation) featuring async event bus, MQTT, and safety hazard actuation."
        }

        return {
            "metadata": metadata,
            "nodes": nodes,
            "brokers": brokers,
            "topics": topics,
            "applications": applications,
            "libraries": libraries,
            "relationships": _to_canonical_relationships(relationships)
        }

    @staticmethod
    def create_edgex_foundry_topology() -> Dict[str, Any]:
        """
        Creates an authentic EdgeX Foundry industrial IoT (IIoT) edge computing topology graph,
        based on the Linux Foundation LF Edge reference architecture.

        Topology overview:
        - 22 Applications across Southbound Device services, Core services, eKuiper rules, Support, and Security
        - 24 Topics with explicit industrial message bus and MQTT QoS configurations
        - 3 Brokers (Redis Streams message bus, Mosquitto OT field broker, eKuiper stream bus)
        - 6 Deployment Nodes (Main edge gateway, Cell A PLC, Cell B PLC, SCADA historian, DMZ connector, Vision accelerator)
        - 8 Shared Go Libraries (go-mod-core-contracts, go-mod-messaging, go-mod-bootstrap, etc.)
        - Realistic SonarQube code quality metrics for applications
        """
        nodes = [
            {"id": "N0", "name": "industrial-edge-gateway-main", "type": "Node", "spec": "Advantech UNO Xeon x86_64 / 32GB RAM"},
            {"id": "N1", "name": "plc-controller-cell-a", "type": "Node", "spec": "Siemens SIMATIC IOT2050 / ARM64"},
            {"id": "N2", "name": "plc-controller-cell-b", "type": "Node", "spec": "Advantech Edge PLC / ARM64 Cortex-A72"},
            {"id": "N3", "name": "ot-scada-historian-node", "type": "Node", "spec": "Local SCADA Host / RAID-10 NVMe"},
            {"id": "N4", "name": "cloud-connector-gateway", "type": "Node", "spec": "Edge DMZ Security Appliance / Dual NIC"},
            {"id": "N5", "name": "vision-inspection-accelerator", "type": "Node", "spec": "NVIDIA Jetson AGX Orin 64GB Industrial"}
        ]

        brokers = [
            {"id": "B0", "name": "edgex-redis-streams-bus", "type": "Broker", "protocol": "Redis-Streams"},
            {"id": "B1", "name": "mosquitto-ot-mqtt-broker", "type": "Broker", "protocol": "MQTT"},
            {"id": "B2", "name": "ekuiper-rule-engine-bus", "type": "Broker", "protocol": "ZeroMQ-eKuiper"}
        ]

        topics = [
            {"id": "T0", "name": "edgex.events.device.modbus_readings", "size": 512, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "MEDIUM"}, "frequency": 100.0, "criticality": "MEDIUM"},
            {"id": "T1", "name": "edgex.events.device.mqtt_telemetry", "size": 512, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "MEDIUM"}, "frequency": 80.0, "criticality": "MEDIUM"},
            {"id": "T2", "name": "edgex.events.device.gpio_digital", "size": 256, "qos": {"durability": "VOLATILE", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 50.0, "criticality": "HIGH"},
            {"id": "T3", "name": "edgex.events.device.vision_anomaly", "size": 4096, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "CRITICAL"}, "frequency": 10.0, "criticality": "HIGH"},
            {"id": "T4", "name": "edgex.core.data.normalized", "size": 1024, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 200.0, "criticality": "HIGH"},
            {"id": "T5", "name": "edgex.core.metadata.device_sync", "size": 2048, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 1.0, "criticality": "HIGH"},
            {"id": "T6", "name": "edgex.commands.request.actuate_valve", "size": 256, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "CRITICAL"}, "frequency": 5.0, "criticality": "HIGH"},
            {"id": "T7", "name": "edgex.commands.request.emergency_stop", "size": 128, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "HIGHEST"}, "frequency": 1.0, "criticality": "HIGH"},
            {"id": "T8", "name": "edgex.commands.request.speed_override", "size": 256, "qos": {"durability": "VOLATILE", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 10.0, "criticality": "HIGH"},
            {"id": "T9", "name": "edgex.commands.response.status", "size": 256, "qos": {"durability": "VOLATILE", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 20.0, "criticality": "HIGH"},
            {"id": "T10", "name": "edgex.alerts.critical_safety", "size": 512, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "HIGHEST"}, "frequency": 2.0, "criticality": "HIGH"},
            {"id": "T11", "name": "edgex.alerts.equipment_fault", "size": 512, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "CRITICAL"}, "frequency": 5.0, "criticality": "HIGH"},
            {"id": "T12", "name": "edgex.alerts.maintenance_warning", "size": 512, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "LOW"}, "frequency": 1.0, "criticality": "LOW"},
            {"id": "T13", "name": "edgex.rules.stream.filtered", "size": 1024, "qos": {"durability": "VOLATILE", "reliability": "RELIABLE", "transport_priority": "MEDIUM"}, "frequency": 50.0, "criticality": "MEDIUM"},
            {"id": "T14", "name": "edgex.export.aws_iot_core", "size": 2048, "qos": {"durability": "TRANSIENT", "reliability": "BEST_EFFORT", "transport_priority": "LOW"}, "frequency": 10.0, "criticality": "LOW"},
            {"id": "T15", "name": "edgex.export.azure_iot_hub", "size": 2048, "qos": {"durability": "TRANSIENT", "reliability": "BEST_EFFORT", "transport_priority": "LOW"}, "frequency": 10.0, "criticality": "LOW"},
            {"id": "T16", "name": "edgex.export.historian_db", "size": 4096, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 50.0, "criticality": "HIGH"},
            {"id": "T17", "name": "edgex.scheduler.cron_heartbeat", "size": 128, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "LOW"}, "frequency": 1.0, "criticality": "LOW"},
            {"id": "T18", "name": "edgex.notifications.sms_pager", "size": 256, "qos": {"durability": "TRANSIENT_LOCAL", "reliability": "RELIABLE", "transport_priority": "HIGHEST"}, "frequency": 1.0, "criticality": "HIGH"},
            {"id": "T19", "name": "edgex.notifications.email_digest", "size": 1024, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "LOW"}, "frequency": 0.1, "criticality": "LOW"},
            {"id": "T20", "name": "edgex.security.token_exchange", "size": 512, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 5.0, "criticality": "HIGH"},
            {"id": "T21", "name": "edgex.support.system_metrics", "size": 512, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "LOW"}, "frequency": 2.0, "criticality": "LOW"},
            {"id": "T22", "name": "edgex.events.device.snmp_network", "size": 512, "qos": {"durability": "VOLATILE", "reliability": "BEST_EFFORT", "transport_priority": "LOW"}, "frequency": 1.0, "criticality": "LOW"},
            {"id": "T23", "name": "edgex.core.keeper.config_update", "size": 1024, "qos": {"durability": "PERSISTENT", "reliability": "RELIABLE", "transport_priority": "HIGH"}, "frequency": 0.2, "criticality": "HIGH"}
        ]

        libraries = [
            {"id": "L0", "name": "go-mod-core-contracts", "version": "3.1.0", "description": "EdgeX canonical Go struct DTOs and data models"},
            {"id": "L1", "name": "go-mod-messaging", "version": "3.1.1", "description": "Multi-bus message client abstraction for Redis Streams and MQTT"},
            {"id": "L2", "name": "go-mod-bootstrap", "version": "3.1.0", "description": "Microservice bootstrap lifecycle and dependency injector"},
            {"id": "L3", "name": "go-mod-secrets", "version": "3.1.0", "description": "HashiCorp Vault secret store interface helper"},
            {"id": "L4", "name": "ekuiper-go-sdk", "version": "1.11.2", "description": "LF Edge eKuiper stream SQL processor extension client"},
            {"id": "L5", "name": "paho-mqtt-golang", "version": "1.4.3", "description": "Eclipse Paho MQTT client runtime for Go"},
            {"id": "L6", "go_mod": "go-redis-v9", "name": "go-redis-v9", "version": "9.5.1", "description": "Redis Streams client connection pool manager"},
            {"id": "L7", "name": "opentelemetry-go", "version": "1.24.0", "description": "OpenTelemetry distributed tracing and metrics SDK"}
        ]

        applications = [
            # Southbound Device Layer
            {"id": "A0", "name": "device_modbus_service", "app_type": "sensor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 4200, "total_classes": 22, "total_methods": 135}, "complexity": {"avg_wmc": 19.5}, "cohesion": {"avg_lcom": 18.0}, "coupling": {"avg_cbo": 9.5, "avg_rfc": 31.0}}},
            {"id": "A1", "name": "device_mqtt_service", "app_type": "sensor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 3600, "total_classes": 18, "total_methods": 115}, "complexity": {"avg_wmc": 16.8}, "cohesion": {"avg_lcom": 15.5}, "coupling": {"avg_cbo": 8.5, "avg_rfc": 27.5}}},
            {"id": "A2", "name": "device_gpio_service", "app_type": "sensor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 2800, "total_classes": 14, "total_methods": 88}, "complexity": {"avg_wmc": 14.0}, "cohesion": {"avg_lcom": 12.5}, "coupling": {"avg_cbo": 7.0, "avg_rfc": 22.0}}},
            {"id": "A3", "name": "device_camera_vision_service", "app_type": "sensor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 5200, "total_classes": 26, "total_methods": 160}, "complexity": {"avg_wmc": 23.5}, "cohesion": {"avg_lcom": 22.0}, "coupling": {"avg_cbo": 11.5, "avg_rfc": 37.0}}},
            {"id": "A4", "name": "device_snmp_service", "app_type": "sensor", "criticality": "LOW", "hotstandby": False, "code_metrics": {"size": {"total_loc": 2100, "total_classes": 11, "total_methods": 65}, "complexity": {"avg_wmc": 10.5}, "cohesion": {"avg_lcom": 9.5}, "coupling": {"avg_cbo": 5.2, "avg_rfc": 16.5}}},

            # Core Services Layer
            {"id": "A5", "name": "core_data_service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 6100, "total_classes": 29, "total_methods": 185}, "complexity": {"avg_wmc": 26.0}, "cohesion": {"avg_lcom": 24.5}, "coupling": {"avg_cbo": 13.0, "avg_rfc": 42.0}}},
            {"id": "A6", "name": "core_metadata_service", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 5400, "total_classes": 25, "total_methods": 165}, "complexity": {"avg_wmc": 22.5}, "cohesion": {"avg_lcom": 21.0}, "coupling": {"avg_cbo": 11.2, "avg_rfc": 36.0}}},
            {"id": "A7", "name": "core_command_service", "app_type": "actuator", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 4800, "total_classes": 23, "total_methods": 145}, "complexity": {"avg_wmc": 20.0}, "cohesion": {"avg_lcom": 18.8}, "coupling": {"avg_cbo": 10.0, "avg_rfc": 32.5}}},
            {"id": "A8", "name": "core_keeper_registry", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 3900, "total_classes": 19, "total_methods": 120}, "complexity": {"avg_wmc": 17.5}, "cohesion": {"avg_lcom": 16.0}, "coupling": {"avg_cbo": 8.8, "avg_rfc": 28.5}}},

            # Northbound App & Analytics Services
            {"id": "A9", "name": "app_service_configurable", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 5900, "total_classes": 28, "total_methods": 175}, "complexity": {"avg_wmc": 25.0}, "cohesion": {"avg_lcom": 23.5}, "coupling": {"avg_cbo": 12.5, "avg_rfc": 40.5}}},
            {"id": "A10", "name": "app_rules_engine_ekuiper", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 7200, "total_classes": 34, "total_methods": 215}, "complexity": {"avg_wmc": 31.0}, "cohesion": {"avg_lcom": 29.5}, "coupling": {"avg_cbo": 15.0, "avg_rfc": 49.0}}},
            {"id": "A11", "name": "app_safety_interlock", "app_type": "actuator", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 3100, "total_classes": 15, "total_methods": 95}, "complexity": {"avg_wmc": 15.5}, "cohesion": {"avg_lcom": 13.8}, "coupling": {"avg_cbo": 7.8, "avg_rfc": 24.5}}},
            {"id": "A12", "name": "app_cloud_exporter_aws", "app_type": "processor", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 3300, "total_classes": 16, "total_methods": 100}, "complexity": {"avg_wmc": 15.0}, "cohesion": {"avg_lcom": 13.5}, "coupling": {"avg_cbo": 7.5, "avg_rfc": 24.0}}},
            {"id": "A13", "name": "app_cloud_exporter_azure", "app_type": "processor", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 3200, "total_classes": 15, "total_methods": 98}, "complexity": {"avg_wmc": 14.8}, "cohesion": {"avg_lcom": 13.2}, "coupling": {"avg_cbo": 7.2, "avg_rfc": 23.5}}},
            {"id": "A14", "name": "app_historian_persister", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 3500, "total_classes": 17, "total_methods": 108}, "complexity": {"avg_wmc": 16.2}, "cohesion": {"avg_lcom": 14.8}, "coupling": {"avg_cbo": 8.0, "avg_rfc": 26.0}}},

            # Support & Security Services
            {"id": "A15", "name": "support_notifications", "app_type": "actuator", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 3700, "total_classes": 18, "total_methods": 112}, "complexity": {"avg_wmc": 17.0}, "cohesion": {"avg_lcom": 15.2}, "coupling": {"avg_cbo": 8.4, "avg_rfc": 27.2}}},
            {"id": "A16", "name": "support_scheduler", "app_type": "sensor", "criticality": "MEDIUM", "hotstandby": False, "code_metrics": {"size": {"total_loc": 2400, "total_classes": 12, "total_methods": 76}, "complexity": {"avg_wmc": 12.2}, "cohesion": {"avg_lcom": 11.0}, "coupling": {"avg_cbo": 6.1, "avg_rfc": 19.5}}},
            {"id": "A17", "name": "security_proxy_kong", "app_type": "sensor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 4400, "total_classes": 21, "total_methods": 132}, "complexity": {"avg_wmc": 19.2}, "cohesion": {"avg_lcom": 17.8}, "coupling": {"avg_cbo": 9.8, "avg_rfc": 31.5}}},
            {"id": "A18", "name": "security_secretstore_vault", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 4100, "total_classes": 20, "total_methods": 125}, "complexity": {"avg_wmc": 18.5}, "cohesion": {"avg_lcom": 16.8}, "coupling": {"avg_cbo": 9.2, "avg_rfc": 29.8}}},
            {"id": "A19", "name": "security_bootstrapper", "app_type": "processor", "criticality": "HIGH", "hotstandby": True, "code_metrics": {"size": {"total_loc": 2600, "total_classes": 13, "total_methods": 82}, "complexity": {"avg_wmc": 13.0}, "cohesion": {"avg_lcom": 11.5}, "coupling": {"avg_cbo": 6.6, "avg_rfc": 21.0}}},
            {"id": "A20", "name": "system_management_agent", "app_type": "sensor", "criticality": "LOW", "hotstandby": False, "code_metrics": {"size": {"total_loc": 2200, "total_classes": 11, "total_methods": 68}, "complexity": {"avg_wmc": 11.0}, "cohesion": {"avg_lcom": 9.8}, "coupling": {"avg_cbo": 5.5, "avg_rfc": 17.5}}},
            {"id": "A21", "name": "ui_management_dashboard", "app_type": "actuator", "criticality": "LOW", "hotstandby": False, "code_metrics": {"size": {"total_loc": 3800, "total_classes": 19, "total_methods": 118}, "complexity": {"avg_wmc": 17.5}, "cohesion": {"avg_lcom": 16.0}, "coupling": {"avg_cbo": 8.7, "avg_rfc": 28.0}}}
        ]

        relationships = [
            # Message bus routing
            # Redis Streams (B0 routes core and app event streams)
            {"source": "B0", "target": "T4", "type": "ROUTES"},
            {"source": "B0", "target": "T5", "type": "ROUTES"},
            {"source": "B0", "target": "T6", "type": "ROUTES"},
            {"source": "B0", "target": "T7", "type": "ROUTES"},
            {"source": "B0", "target": "T8", "type": "ROUTES"},
            {"source": "B0", "target": "T9", "type": "ROUTES"},
            {"source": "B0", "target": "T10", "type": "ROUTES"},
            {"source": "B0", "target": "T11", "type": "ROUTES"},
            {"source": "B0", "target": "T12", "type": "ROUTES"},
            {"source": "B0", "target": "T13", "type": "ROUTES"},
            {"source": "B0", "target": "T16", "type": "ROUTES"},
            {"source": "B0", "target": "T18", "type": "ROUTES"},
            {"source": "B0", "target": "T23", "type": "ROUTES"},

            # OT Mosquitto broker (B1 routes field device data)
            {"source": "B1", "target": "T0", "type": "ROUTES"},
            {"source": "B1", "target": "T1", "type": "ROUTES"},
            {"source": "B1", "target": "T2", "type": "ROUTES"},
            {"source": "B1", "target": "T6", "type": "ROUTES"},
            {"source": "B1", "target": "T7", "type": "ROUTES"},
            {"source": "B1", "target": "T22", "type": "ROUTES"},

            # eKuiper ZeroMQ bus (B2 routes stream rules & external clouds)
            {"source": "B2", "target": "T3", "type": "ROUTES"},
            {"source": "B2", "target": "T10", "type": "ROUTES"},
            {"source": "B2", "target": "T13", "type": "ROUTES"},
            {"source": "B2", "target": "T14", "type": "ROUTES"},
            {"source": "B2", "target": "T15", "type": "ROUTES"},
            {"source": "B2", "target": "T17", "type": "ROUTES"},
            {"source": "B2", "target": "T21", "type": "ROUTES"},

            # Device Services publishing
            {"source": "A0", "target": "T0", "type": "PUBLISHES_TO"},
            {"source": "A1", "target": "T1", "type": "PUBLISHES_TO"},
            {"source": "A2", "target": "T2", "type": "PUBLISHES_TO"},
            {"source": "A3", "target": "T3", "type": "PUBLISHES_TO"},
            {"source": "A4", "target": "T22", "type": "PUBLISHES_TO"},

            # Core Data ingestion & normalization
            {"source": "A5", "target": "T0", "type": "SUBSCRIBES_TO"},
            {"source": "A5", "target": "T1", "type": "SUBSCRIBES_TO"},
            {"source": "A5", "target": "T2", "type": "SUBSCRIBES_TO"},
            {"source": "A5", "target": "T3", "type": "SUBSCRIBES_TO"},
            {"source": "A5", "target": "T4", "type": "PUBLISHES_TO"},

            # Core Metadata & Registry
            {"source": "A6", "target": "T5", "type": "PUBLISHES_TO"},
            {"source": "A8", "target": "T23", "type": "PUBLISHES_TO"},
            {"source": "A0", "target": "T5", "type": "SUBSCRIBES_TO"},
            {"source": "A1", "target": "T5", "type": "SUBSCRIBES_TO"},

            # Command execution loop (Northbound -> Command -> Device actuation)
            {"source": "A7", "target": "T6", "type": "SUBSCRIBES_TO"},
            {"source": "A7", "target": "T7", "type": "SUBSCRIBES_TO"},
            {"source": "A7", "target": "T8", "type": "SUBSCRIBES_TO"},
            {"source": "A7", "target": "T9", "type": "PUBLISHES_TO"},
            {"source": "A0", "target": "T6", "type": "SUBSCRIBES_TO"},
            {"source": "A2", "target": "T7", "type": "SUBSCRIBES_TO"},

            # App Services & Rules Engine
            {"source": "A9", "target": "T4", "type": "SUBSCRIBES_TO"},
            {"source": "A9", "target": "T13", "type": "PUBLISHES_TO"},
            {"source": "A10", "target": "T4", "type": "SUBSCRIBES_TO"},
            {"source": "A10", "target": "T10", "type": "PUBLISHES_TO"},
            {"source": "A10", "target": "T11", "type": "PUBLISHES_TO"},
            {"source": "A10", "target": "T12", "type": "PUBLISHES_TO"},

            # Safety Interlock (Highest priority emergency circuit)
            {"source": "A11", "target": "T10", "type": "SUBSCRIBES_TO"},
            {"source": "A11", "target": "T7", "type": "PUBLISHES_TO"},
            {"source": "A11", "target": "T18", "type": "PUBLISHES_TO"},

            # Cloud Export & Historian
            {"source": "A12", "target": "T13", "type": "SUBSCRIBES_TO"},
            {"source": "A12", "target": "T14", "type": "PUBLISHES_TO"},
            {"source": "A13", "target": "T13", "type": "SUBSCRIBES_TO"},
            {"source": "A13", "target": "T15", "type": "PUBLISHES_TO"},
            {"source": "A14", "target": "T4", "type": "SUBSCRIBES_TO"},
            {"source": "A14", "target": "T16", "type": "PUBLISHES_TO"},

            # Support & Security
            {"source": "A15", "target": "T18", "type": "SUBSCRIBES_TO"},
            {"source": "A15", "target": "T19", "type": "SUBSCRIBES_TO"},
            {"source": "A16", "target": "T17", "type": "PUBLISHES_TO"},
            {"source": "A17", "target": "T20", "type": "PUBLISHES_TO"},
            {"source": "A18", "target": "T20", "type": "SUBSCRIBES_TO"},
            {"source": "A20", "target": "T21", "type": "PUBLISHES_TO"},
            {"source": "A21", "target": "T4", "type": "SUBSCRIBES_TO"},

            # Deployment (RUNS_ON)
            {"source": "A5", "target": "N0", "type": "RUNS_ON"},
            {"source": "A6", "target": "N0", "type": "RUNS_ON"},
            {"source": "A7", "target": "N0", "type": "RUNS_ON"},
            {"source": "A8", "target": "N0", "type": "RUNS_ON"},
            {"source": "A9", "target": "N0", "type": "RUNS_ON"},
            {"source": "A10", "target": "N0", "type": "RUNS_ON"},
            {"source": "A11", "target": "N0", "type": "RUNS_ON"},
            {"source": "B0", "target": "N0", "type": "RUNS_ON"},
            {"source": "B1", "target": "N0", "type": "RUNS_ON"},
            {"source": "B2", "target": "N0", "type": "RUNS_ON"},

            {"source": "A0", "target": "N1", "type": "RUNS_ON"},
            {"source": "A2", "target": "N1", "type": "RUNS_ON"},
            {"source": "A1", "target": "N2", "type": "RUNS_ON"},
            {"source": "A4", "target": "N2", "type": "RUNS_ON"},
            {"source": "A3", "target": "N5", "type": "RUNS_ON"},
            {"source": "A14", "target": "N3", "type": "RUNS_ON"},
            {"source": "A12", "target": "N4", "type": "RUNS_ON"},
            {"source": "A13", "target": "N4", "type": "RUNS_ON"},
            {"source": "A17", "target": "N4", "type": "RUNS_ON"},
            {"source": "A18", "target": "N0", "type": "RUNS_ON"},
            {"source": "A19", "target": "N0", "type": "RUNS_ON"},
            {"source": "A15", "target": "N0", "type": "RUNS_ON"},
            {"source": "A16", "target": "N0", "type": "RUNS_ON"},
            {"source": "A20", "target": "N0", "type": "RUNS_ON"},
            {"source": "A21", "target": "N0", "type": "RUNS_ON"},

            # Library Dependencies (USES)
            {"source": "A0", "target": "L0", "type": "USES"},
            {"source": "A1", "target": "L0", "type": "USES"},
            {"source": "A1", "target": "L5", "type": "USES"},
            {"source": "A5", "target": "L0", "type": "USES"},
            {"source": "A5", "target": "L1", "type": "USES"},
            {"source": "A5", "target": "L6", "type": "USES"},
            {"source": "A7", "target": "L0", "type": "USES"},
            {"source": "A7", "target": "L1", "type": "USES"},
            {"source": "A9", "target": "L1", "type": "USES"},
            {"source": "A10", "target": "L4", "type": "USES"},
            {"source": "A18", "target": "L3", "type": "USES"},
            {"source": "A5", "target": "L7", "type": "USES"},
            {"source": "A9", "target": "L2", "type": "USES"},

            # Hardware Interconnects (CONNECTS_TO)
            {"source": "N0", "target": "N1", "type": "CONNECTS_TO"},
            {"source": "N0", "target": "N2", "type": "CONNECTS_TO"},
            {"source": "N0", "target": "N3", "type": "CONNECTS_TO"},
            {"source": "N0", "target": "N4", "type": "CONNECTS_TO"},
            {"source": "N0", "target": "N5", "type": "CONNECTS_TO"},
            {"source": "N1", "target": "N2", "type": "CONNECTS_TO"}
        ]

        metadata = {
            "scale": {
                "apps": len(applications),
                "topics": len(topics),
                "brokers": len(brokers),
                "nodes": len(nodes),
                "libs": len(libraries)
            },
            "seed": 2026,
            "generation_mode": "realworld_open_source",
            "domain": "edgex_foundry_iiot",
            "scenario": "production_edgex_industrial_iot_mesh",
            "description": "Authentic production-grade EdgeX Foundry industrial IoT (IIoT) edge computing topology (Linux Foundation LF Edge) with Redis Streams, southbound Modbus/GPIO device services, and eKuiper stream rules."
        }

        return {
            "metadata": metadata,
            "nodes": nodes,
            "brokers": brokers,
            "topics": topics,
            "applications": applications,
            "libraries": libraries,
            "relationships": _to_canonical_relationships(relationships)
        }


