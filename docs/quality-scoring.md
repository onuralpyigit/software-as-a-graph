# Step 3: Quality Scoring & Problem Detection

Once the graph is constructed (Step 1) and raw metrics are computed (Step 2), the system performs **Quality Scoring**. This step acts as an automated "Health Check" for your distributed architecture. It translates abstract topological metrics into concrete warnings about system stability, maintainability, and reliability.

## 1. The Goal: Automated Diagnostics

We don't just want to know *which* node has the highest "Betweenness Centrality"; we want to know *what that means* for the system.

This step answers:

* **Where will the system break?** (Availability/SPOF)
* **Which components are dangerous to touch?** (Maintainability/Coupling)
* **If X crashes, does the whole system go down?** (Reliability/Propagation)

## 2. Detecting Problems (`analyze_graph.py`)

The `analyze_graph.py` CLI script runs the **Problem Detector**, which scans the graph for specific anti-patterns and risks.

### **Usage**

To generate a diagnostic report for the entire system:

```bash
python analyze_graph.py --layer system

```

For a detailed JSON report (useful for CI/CD pipelines):

```bash
python analyze_graph.py --layer system --output results/health_check.json

```

### **What It Detects**

The detector looks for specific structural "smells" in three categories:

#### 🚨 Availability Risks (SPOFs)

* **Single Point of Failure**: A component identified as an *Articulation Point*. If it fails, the network partitions.
* *Symptom*: "Removing this component disconnects the dependency graph."


* **Bridge Edges**: Critical links that have no backup paths.
* *Symptom*: "Dependency A → B is the only path connecting these clusters."



#### ⚠️ Maintainability Risks (Coupling)

* **God Component**: A node with extreme *Betweenness Centrality*. It knows too much and does too much.
* *Symptom*: "Lies on 80% of all shortest paths. Coupling hotspot."


* **Hub-and-Spoke**: A central hub connected to many isolated neighbors (Low *Clustering Coefficient*).
* *Symptom*: "Neighbors don't communicate directly; highly dependent on the central hub."



#### 📉 Reliability Risks (Cascades)

* **Propagation Hub**: A component with high *Reverse PageRank*.
* *Symptom*: "A crash here cascades to >50% of the system."


* **Concentration Risk**: When a small percentage of components account for the majority of the system's importance.

---

## 3. Interpreting the Output

When you run the command, look for the **Detected Problems** section in the terminal output:

```text
>> Detected Problems
──────────────────────────────────────────────────────────────────────────────
  [CRITICAL] Single Point of Failure (SPOF)
           Entity: sensor_fusion (Application)
           Category: Availability
           Issue: Component is an articulation point. Removal partitions the system.
           Fix: Add redundant instance or decouple using an event bus.

  [CRITICAL] God Component / Central Bottleneck
           Entity: main_broker (Broker)
           Category: Maintainability
           Issue: Lies on 92% of shortest paths. High coupling risk.
           Fix: Split responsibilities or introduce a facade pattern.

  [HIGH]     Fragile Topology
           Entity: SYSTEM
           Category: Architecture
           Issue: System has 5 articulation points (15% of components).
           Fix: Increase mesh connectivity.

```

---

---

# Documentation Rewrite

Below is the rewritten, simplified, and improved **Quality Formulations** documentation.

---

# 📐 Quality Formulations Reference

**The Math Behind the Criticality Scores**

This document explains how we calculate the three dimensions of software quality: **Reliability (R)**, **Maintainability (M)**, and **Availability (A)**.

Instead of arbitrary guesses, we use **weighted composite scores** derived from graph theory.

## 1. The Three Pillars (R-M-A)

We assess every component using these three questions:

| Score | Name | The Question | Why it matters |
| --- | --- | --- | --- |
| **R** | **Reliability** | *If this breaks, does it spread?* | Focuses on **Runtime Stability** and **Cascade Risk**. |
| **M** | **Maintainability** | *Is this hard to change?* | Focuses on **Coupling**, **Complexity**, and **Tech Debt**. |
| **A** | **Availability** | *Is this indispensable?* | Focuses on **Redundancy** and **Single Points of Failure**. |

---

## 2. Component Scoring Formulas

We normalize all raw metrics to a  scale before combining them.

### 🛡️ Reliability Score 

**Measures: Fault Propagation Risk**

* **Influence (PageRank):** How many *important* components depend on this?
* **Impact (Reverse PageRank):** If this fails, how far downstream does the error travel?
* **Dependents (In-Degree):** How many direct components break immediately?

> **Interpretation:** A high  score means a component is a "Load-Bearing Wall." If it cracks, the building shakes.

---

### 🛠️ Maintainability Score 

**Measures: Coupling & Complexity**

* **Bottleneck (Betweenness):** Does information *have* to pass through here? High betweenness = Traffic Jam.
* **Tangling (1 - Clustering):** Are neighbors isolated from each other? (We invert Clustering because *low* clustering implies a dependency hub that is hard to refactor).
* **Complexity (Degree):** How many total connections does it manage?

> **Interpretation:** A high  score means a "God Object." Touching it is risky because it's connected to everything.

---

### ⏱️ Availability Score 

**Measures: Single Point of Failure (SPOF) Risk**

* **SPOF (Articulation Point):** Binary (0 or 1). If removed, does the network split into pieces?
* **Fragility (Bridge Ratio):** What % of its links are the *only* path to somewhere?
* **Importance (Criticality):** Is it both highly connected and highly influential?

> **Interpretation:** A high  score means the component is a "Choke Point." You likely need a backup instance (redundancy).

---

## 3. The Final Score: 

We combine the three pillars into one **Overall Criticality Score**.

* **Reliability & Availability (35% each):** Weighted higher because they impact the live system immediately.
* **Maintainability (30%):** Weighted slightly lower as it impacts development velocity primarily.

---

## 4. How We Classify (The Box-Plot Method)

We **do not** use static thresholds (e.g., "Anything above 0.8 is critical"). Every system is different.
Instead, we use **Adaptive Statistical Classification**.

We look at the distribution of scores in *your specific system* and calculate the **Interquartile Range (IQR)**.

| Level | Rule | Meaning |
| --- | --- | --- |
| 🔴 **CRITICAL** | Score >  | **Statistical Outlier**. This component is abnormally risky compared to the rest of your system. |
| 🟠 **HIGH** | Score >  (Top 25%) | **Priority**. The most critical of the "normal" components. |
| 🟡 **MEDIUM** | Score > Median | **Average**. Standard system components. |
| 🟢 **LOW** | Score ≤ Median | **Safe**. Leaves or simple components. |

---

## 5. Edge Quality (Dependencies)

We also score the connections (edges) themselves.

* **Edge Reliability:** Is this a heavy data pipe () between two critical nodes?
* **Edge Availability:** Is this a **Bridge Edge**? (i.e., The only cable connecting two islands).

> **Use Case:** A "Critical Edge" warning suggests you should add a second pathway or backup link between those two components.