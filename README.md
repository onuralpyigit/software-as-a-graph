# Software-as-a-Graph

**Graph-based quality assessment framework for distributed publish-subscribe systems.**

This project implements a comprehensive methodology for modeling software architectures as graphs, analyzing them using topological metrics, and validating the results against failure simulations. It focuses on assessing **Reliability**, **Maintainability**, and **Availability** (RMA).

---

## 🚀 Key Features

* **Multi-Layer Graph Modeling**: Maps distributed components (Applications, Topics, Brokers, Infrastructure Nodes) and their dependencies into a directed graph.
* **Smart Dependency Derivation**: Automatically derives weighted dependencies (e.g., `App -> App` via shared Topics) directly within Neo4j, factoring in **QoS** (Durability, Reliability, Priority) and message size.
* **Quality Analysis (RMA)**: Uses graph algorithms (PageRank, Betweenness, etc.) to compute composite quality scores.
* **Failure Simulation**: Simulates cascading failures and message events to measure actual system resilience.
* **Statistical Validation**: Validates analytical predictions against simulation results using Spearman correlation.
* **Visualization**: Generates HTML dashboards for model insights and analysis results.
* **Neo4j Integration**: Uses Neo4j as the central graph database for storage and analysis.

---

## 📊 Graph Modeling Approach

We treat the software system as a directed graph $G = (V, E)$, where nodes represent software/hardware components and edges represent dependencies.

### 1. Component Types (Nodes)
* **Application**: Services or microservices that publish/subscribe to data.
* **Topic**: Logical channels for message exchange.
* **Broker**: Middleware components managing message distribution.
* **Node**: Physical or virtual infrastructure hosting the components.

### 2. Dependency Derivation
The framework automatically derives a **DEPENDS_ON** relationship layer to enable architectural analysis. This is done via Cypher queries during import:

| Dependency Type | Direction | Logic |
|-----------------|-----------|-------|
| **app_to_app** | Subscriber → Publisher | Derived if Subscriber subscribes to a Topic that Publisher publishes to. |
| **app_to_broker** | App → Broker | Derived if an App uses a Topic routed by a Broker. |
| **node_to_node** | Node A → Node B | Derived if an App on Node A depends on an App on Node B. |
| **node_to_broker** | Node → Broker | Derived if an App on Node depends on that Broker. |

### 3. Weight Calculation
Dependencies are not equal. We assign a **Weight** to every `DEPENDS_ON` edge based on the criticality of the data flow:
* **Formula**: $Weight = \text{TopicCount} + \sum(\text{QoS Score} + \text{Size Factor})$
* **QoS Score**: Higher for `PERSISTENT`, `RELIABLE`, and `URGENT` topics.
* **Size Factor**: Higher for larger message payloads.

---

## 🧮 Analysis Framework

The core of the project is the **Quality Analyzer**, which computes composite scores based on topological metrics.

### 1. Reliability (R)
Measures the system's robustness against faults.
* **Formula**: $R(v) = w_{pr} \cdot \text{PageRank} + w_{fp} \cdot \text{FailureProp} + w_{id} \cdot \text{InDegree}$
* **Logic**: Components with high influence (PageRank) or that propagate failures widely are critical for reliability.

### 2. Maintainability (M)
Measures the ease of modification and modularity.
* **Formula**: $M(v) = w_{bt} \cdot \text{Betweenness} + w_{cc} \cdot (1 - \text{Clustering}) + w_{dc} \cdot \text{Degree}$
* **Logic**: High betweenness indicates a "bottleneck" component (hard to maintain). Low clustering implies poor modularity.

### 3. Availability (A)
Measures the risk of service interruption.
* **Formula**: $A(v) = w_{ap} \cdot \text{ArticulationPoint} + w_{br} \cdot \text{BridgeRatio} + w_{cr} \cdot \text{Criticality}$
* **Logic**: Articulation points are Single Points of Failure (SPOFs).

### 4. Overall Quality (Q)
A weighted composite of the three attributes:
* **Formula**: $Q(v) = 0.35 \cdot R(v) + 0.30 \cdot M(v) + 0.35 \cdot A(v)$

---

## 🔄 End-to-End Pipeline

The project is orchestrated via `run.py`, which executes the following steps sequentially:

1.  **Generate** (`generate_graph.py`): Creates synthetic graph data (nodes/edges) with rich QoS properties if no input is provided.
2.  **Import** (`import_graph.py`): 
    * Loads data into Neo4j.
    * **Derives** `DEPENDS_ON` relationships using Cypher.
    * **Calculates** edge and component weights.
3.  **Analyze** (`analyze_graph.py`): 
    * Retrieves the graph from Neo4j.
    * Computes Structural Metrics (NetworkX).
    * Calculates RMA Quality Scores.
    * Detects specific problems (SPOFs, bottlenecks).
4.  **Simulate** (`simulate_graph.py`): 
    * **Failure Sim**: Injects faults and traces cascading impact.
    * **Event Sim**: Traces message reachability.
    * Stores "Actual Impact" scores.
5.  **Validate** (`validate_graph.py`): 
    * Compares **Predicted Scores** (from Analysis) vs. **Actual Impact** (from Simulation).
    * Calculates correlation coefficients (Spearman, Pearson).
6.  **Visualize** (`visualize_graph.py`): Generates an HTML dashboard with charts and tables.

---

## 💻 Usage

### Prerequisites
* Python 3.8+
* Neo4j Database (running locally or remotely)
* Dependencies: `pip install -r requirements.txt`

### Quick Start
Run the full pipeline with a single command:
```bash
python run.py --all --uri bolt://localhost:7687 --password your_password