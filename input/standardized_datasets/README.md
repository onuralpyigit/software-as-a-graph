# Distributed System Topology Datasets

This directory contains topology datasets converted to a standardized JSON format for distributed system analysis.

## Sources

The datasets were extracted from the following open-source projects:

| Dataset | Source |
|---------|--------|
| `dataset_sierra_nevada.json` | [irobot-ros/ros2-performance](https://github.com/irobot-ros/ros2-performance/tree/rolling/irobot_benchmark/topology) |
| `dataset_cedar.json` | [irobot-ros/ros2-performance](https://github.com/irobot-ros/ros2-performance/tree/rolling/irobot_benchmark/topology) |
| `dataset_mont_blanc.json` | [irobot-ros/ros2-performance](https://github.com/irobot-ros/ros2-performance/tree/rolling/irobot_benchmark/topology) |
| `dataset_white_mountain.json` | [irobot-ros/ros2-performance](https://github.com/irobot-ros/ros2-performance/tree/rolling/irobot_benchmark/topology) |
| `dataset_rti_medtech.json` | [rticonnextdds-medtech-reference-architecture](https://github.com/rticommunity/rticonnextdds-medtech-reference-architecture) |
| `dataset_rti_automotive.json` | [rticonnextdds-automotive-cases-window-controller](https://github.com/rticommunity/rticonnextdds-automotive-cases-window-controller) |

## Data Extraction

### From ROS2 Topology Files
- **Applications**: Extracted from `node_name` fields
- **Topics**: Extracted from `publishers[].topic_name` and `subscribers[].topic_name`
- **Pub/Sub relationships**: Directly mapped from original topology

### From RTI ConnextDDS Projects
- **Applications**: Extracted from DomainParticipant definitions in XML configuration files
- **Topics**: Extracted from topic registrations in `DomainLibrary.xml` and IDL files
- **Pub/Sub relationships**: Extracted from DataWriter/DataReader definitions in `ParticipantLibrary.xml`
- **QoS profiles**: Extracted from `Qos.xml` (Streaming, Status, Command, Heartbeat profiles)

## Generated/Inferred Fields

The following fields were not present in the original sources and were assigned using default rules:

| Field | Assignment Rule |
|-------|-----------------|
| **nodes** | Auto-generated based on application count (roughly √n nodes) |
| **libraries** | Auto-generated placeholder libraries |
| **runs_on** | Round-robin distribution of applications across nodes |
| **connects_to** | Ring topology for node connections |
| **uses** | Balanced distribution of library dependencies |

### Topic Size
- **ROS datasets**: Estimated from `msg_type` field (e.g., `stamped4_int32` → 24 bytes) or used explicit `msg_size` when available
- **RTI datasets**: Calculated from struct definitions in IDL/XML type files

### QoS Settings
- **ROS datasets**: Default minimal values (`VOLATILE`, `BEST_EFFORT`, `LOW` priority) since QoS info was not available
- **RTI datasets**: Extracted from QoS profile definitions (e.g., `TRANSIENT_LOCAL` for status topics, `RELIABLE` for commands)

### Criticality
- **ROS datasets**: All set to `false` (no criticality information in source)
- **RTI datasets**: Assigned based on application semantics:
  - `true`: Controllers, sensors, orchestrators (safety-critical functions)
  - `false`: UI/monitoring applications

## Dataset Format

All datasets follow this schema:

```json
{
  "metadata": { "scale", "seed", "source" },
  "nodes": [{ "id", "name" }],
  "brokers": [],
  "topics": [{ "id", "name", "size", "qos" }],
  "applications": [{ "id", "name", "role", "app_type", "criticality", "version" }],
  "libraries": [{ "id", "name", "version" }],
  "relationships": {
    "runs_on": [],
    "routes": [],
    "publishes_to": [],
    "subscribes_to": [],
    "connects_to": [],
    "uses": []
  }
}
```

Note: `brokers` and `routes` are empty since DDS uses peer-to-peer communication without message brokers.
