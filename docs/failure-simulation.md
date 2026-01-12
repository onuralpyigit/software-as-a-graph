# Step 4: Failure Simulation

**Validating predictions through empirical Fault Injection and Event-Driven Analysis.**

After predicting critical components using topological metrics (Step 2 & 3), Step 4 moves from *theory* to *practice*. We use the `simulate_graph.py` CLI to perform **Failure Impact Assessment**—injecting faults and simulating message flows to measure the *actual* impact on the system.

This step provides the "Ground Truth" () used to validate our predictive Criticality Scores ().

---

## 1. The Dual Simulation Approach

Distributed pub-sub systems fail in two distinct ways: structurally (disconnections) and behaviorally (message loss/latency). Therefore, the framework provides two simulation engines:

| Simulation Type | Engine | Focus | Key Question |
| --- | --- | --- | --- |
| **Failure Simulation** | `FailureSimulator` | **Structural** | "If Component X vanishes, how many parts of the system become unreachable?" |
| **Event Simulation** | `EventSimulator` | **Behavioral** | "If App A publishes 1,000 messages, how many arrive, and how fast?" |

---

## 2. Running Failure Simulations (Structural)

This mode simulates the complete removal of a component (crash failure) and measures the cascading effects on the graph topology. It handles physical cascades (Node crash  App crash) and logical cascades (Broker crash  Topic unreachable).

### CLI Usage

To simulate the failure of a specific component:

```bash
python simulate_graph.py --failure <COMPONENT_ID>

```

**Example:**

```bash
python simulate_graph.py --failure main_broker --layer system

```

### Understanding the Output

The CLI provides a color-coded impact assessment:

* **Composite Impact:** A score (0.0 - 1.0) combining reachability loss, fragmentation, and throughput loss.
* `> 0.5`: **CRITICAL** (Red)
* `> 0.2`: **HIGH** (Yellow)


* **Reachability Loss:** Percentage of broken Pub-Sub paths.
* **Cascade Analysis:** A trace of other components that failed due to the initial target (e.g., Apps failing because their host Node crashed).

---

## 3. Running Event-Driven Simulations (Behavioral)

This mode launches a discrete event simulation (DES) that models the flow of messages from a Publisher through Topics and Brokers to Subscribers. It enforces QoS policies (Reliability, Priority) and tracks runtime metrics.

### CLI Usage

To simulate traffic generation from a specific source application:

```bash
python simulate_graph.py --event <APP_ID> --messages 100 --duration 10.0

```

**Example:**

```bash
# Simulate 500 messages from 'sensing_node' over 20 seconds
python simulate_graph.py --event sensing_node --messages 500 --duration 20.0

```

### Understanding the Output

The simulator reports runtime performance metrics:

* **Delivery Rate:** % of messages successfully acknowledged.
* **Drop Rate:** % of messages lost (due to queue overflows, timeouts, or lack of routes).
* **Latency Metrics:** Min, Max, Average, P50, and **P99 Latency**.
* **Path Analysis:** Which Brokers and Topics were utilized during transmission.

---

## 4. Exhaustive Analysis & Reporting

To validate the entire system model, you can run simulations on **all components** or generate a comprehensive system report.

### Exhaustive Failure Analysis

This runs a failure simulation for *every single component* in the specified layer and ranks them by impact. This is the "Ground Truth" list of critical components.

```bash
python simulate_graph.py --exhaustive --layer system

```

**Output:**

```text
Exhaustive Failure Analysis
  Total Components Analyzed: 45
  
  Top 15 Components by Impact:
  Component            Type         Impact      Cascade    Reach Loss
  -----------------------------------------------------------------
  main_broker          Broker       0.9125      4          85.0%
  sensor_fusion        Application  0.8420      2          60.0%
  ...

```

### Full Simulation Report

Generates a summary combining event metrics (throughput/health) and failure resilience stats for all layers.

```bash
python simulate_graph.py --report --layers app,infra,system --output results/sim_report.json

```

---

## 5. Metrics Definition

The `simulate_graph.py` tool calculates specific metrics to quantify "Impact" ():

### Structural Metrics (Failure Sim)

* **Reachability Loss:** The percentage of pub-sub paths that no longer exist.
* **Fragmentation:** The increase in isolated graph islands (connected components).
* **Cascade Depth:** How many "hops" the failure propagated (e.g., Node  Broker  Topic  Subscriber).

### Runtime Metrics (Event Sim)

* **Throughput:** Messages delivered per second.
* **End-to-End Latency:** Time from `publish` event to `deliver` event.
* **Drop Reasons:** Specific categorization of failures (e.g., `no_subscribers`, `broker_failed`, `delivery_timeout`).

---

## Next Step

Once you have the **Impact Scores** () from this step, you compare them against the **Predicted Quality Scores** () from Step 3 to validate the model's accuracy.

* **See:** `docs/validation-comparison.md`
* **Run:** `python validate_graph.py`