# Step 6: Prescribe

**Rule-based architectural refactoring recommendations and closed-loop resilience validation.**

← [Step 5: Validate](validation.md) | → [Step 7: Visualize](visualization.md)

---

## Table of Contents

1. [What This Step Does](#1-what-this-step-does)
2. [Preservation & Remediation Rules](#2-preservation--remediation-rules)
   - 2.1 [Logical Subgraph Refactoring (Topic Splitting)](#21-logical-subgraph-refactoring-topic-splitting)
   - 2.2 [Physical Locality Anti-Affinity Rules](#22-physical-locality-anti-affinity-rules)
   - 2.3 [Middleware Transport Contract Hardening](#23-middleware-transport-contract-hardening)
3. [Closed-Loop Verification Mechanics](#3-closed-loop-verification-mechanics)
4. [Programmatic API Reference](#4-programmatic-api-reference)
   - 4.1 [Using Pipeline](#41-using-pipeline)
   - 4.2 [Using Client](#42-using-client)
5. [Data Schema & Output Models](#5-data-schema--output-models)

---

## 1. What This Step Does

Step 6 (Prescribe) completes the closed-loop optimization cycle. Once high-risk components and topological anti-patterns (such as Single Points of Failure, god components, and bottleneck topic hubs) are identified (Step 2/3) and validated against cascade simulation ground truths (Step 5), the Prescribe stage compiles an optimization policy $\Delta(G)$ consisting of three rule-based transformations.

```
       Validated Risks (Step 5)
                 │
                 ▼
       Refactoring Policies Δ(G)
     ┌───────────────────────────────┐
     │ 1. Topic Splitting            │
     │ 2. Physical Host Isolation    │
     │ 3. Transport QoS Hardening    │
     └───────────┬───────────────────┘
                 │
                 ▼
       In-Memory Mutated Graph G'
     ┌───────────────────────────────┐
     │  Export G to JSON             │
     │  Apply Δ(G) refactoring       │
     │  Load into MemoryRepository   │
     └───────────┬───────────────────┘
                 │
                 ▼
       Closed-Loop Validation
     ┌───────────────────────────────┐
     │  Run Simulate(G')             │
     │  Calculate Mutated SRI        │
     └───────────┬───────────────────┘
                 │
                 ▼
       Accept / Reject Gate
     ┌───────────────────────────────┐
     │  accepted = (ΔSRI > 0)        │
     └───────────┬───────────────────┘
                 │
                 ▼
       Remediation Blueprint
       (Baseline SRI -> Mutated SRI, accepted: true/false)
```

To prevent target database contamination or transaction overhead, these refactoring rules are applied directly in-memory to a JSON representation of the graph. The mutated topology $G'$ is then validated in a closed-loop simulation sweep, comparing the baseline System Risk Index (SRI) against the mutated SRI. Lower SRI indicates lower system risk (better health); higher SRI indicates greater structural risk. The result is marked `accepted` when the mutation reduces risk, but a rejected policy is still returned in full for inspection — it is not automatically discarded or retried (see §3).

---

## 2. Preservation & Remediation Rules

Remediations target components categorized as `CRITICAL`/`HIGH` risk by the adaptive box-plot filter; node reallocation (§2.2) additionally considers components flagged as SPOF or god-component smells by the `AntiPatternDetector`.

### 2.1 Logical Subgraph Refactoring (Topic Splitting)

**Problem**: A central topic hub connected to multiple publishers and subscribers becomes a logical bottleneck and a high-risk failure propagator.
**Remediation**: Split the congested topic $T$ into dedicated sub-topics per publisher:
* For each publisher $P_i$ publishing to $T$, create a new sub-topic $T_{P_i}$.
* Re-route the publish relationship from $P_i \rightarrow T$ to $P_i \rightarrow T_{P_i}$.
* Re-route all subscribers $S_j$ subscribing to $T$ to subscribe to the set of all sub-topics: $S_j \rightarrow T_{P_i}$ for all $P_i$.
* Duplicate broker routing links for all sub-topics.

This bounds failure propagation, separating independent logical communication channels.

### 2.2 Physical Locality Anti-Affinity Rules

**Problem**: Multiple processes (Applications or Brokers) co-located on a single physical host node $N$ that is flagged as a Single Point of Failure (SPOF) or critical risk. If $N$ fails, all hosted components fail simultaneously.
**Remediation**: Establish scheduling anti-affinity constraints to isolate co-located processes:
* Identify critical hosts $N$ where the host or any hosted component is `CRITICAL`/`HIGH` risk or a SPOF, and $N$ hosts multiple processes.
* Allocate new separate physical node instances $N_{C_i}$ for each co-located process $C_i$ (except the first).
* Update `RUNS_ON` relationships to reallocate $C_i$ to $N_{C_i}$.
* Duplicate host-to-host `CONNECTS_TO` connections to ensure network reachability is preserved.

### 2.3 Middleware Transport Contract Hardening

**Problem**: Critical communication channels utilizing unreliable or volatile transport configurations (e.g. ROS 2 `BEST_EFFORT` reliability or `VOLATILE` durability).
**Remediation**: Harden the transport properties to reliable and transient-local contracts:
* For any topic $T$ that is `CRITICAL`/`HIGH` risk or connects to a critical component:
  * Set `qos_reliability = "RELIABLE"`.
  * Set `qos_durability = "TRANSIENT"`.

---

## 3. Closed-Loop Verification Mechanics

The verification engine executes the following programmatic loop:
1. **Export Original Topology**: Export the source graph as a flat JSON schema.
2. **Apply Policy $\Delta(G)$**: Apply the compiled splits, reallocations, and QoS upgrades to the JSON structure.
3. **Instantiate Temporary Sandbox**: Seed a temporary `MemoryRepository` with the mutated JSON and derive its logical dependency edges.
4. **Evaluate Mutated Graph**: Run the full Analysis, Simulation, and Validation suite on the sandbox repository.
5. **Compute SRI Improvement**:
   $$\Delta \text{SRI} = \text{SRI}_{\text{baseline}} - \text{SRI}_{\text{mutated}}$$
   Since SRI is a risk index (lower is better), $\Delta \text{SRI} > 0$ means the mutated topology carries less structural risk than the baseline.
6. **Accept/Reject Gate**: If $\Delta \text{SRI} > 0$, the policy is marked `accepted = true`. Otherwise it is marked `accepted = false` and returned as-is for inspection — the policy is **not** automatically discarded or retried. This is a whole-policy gate only: per-edit filtering of individual mutation rules within a policy (rejecting just the one operator that hurt, while keeping the others) is not yet implemented and is tracked as future work.

---

## 4. Programmatic API Reference

### 4.1 Using Pipeline

The easiest way to trigger the Prescribe stage is using the fluent pipeline builder API:

```python
from saag import Pipeline

result = (
    Pipeline.from_json("data/system.json", clear=True)
        .analyze(layer="system")
        .simulate(layer="system")
        .validate()
        .prescribe()
        .run()
)

if result.prescription:
    print(f"Baseline SRI: {result.prescription.original_sri:.4f}")
    print(f"Mutated SRI : {result.prescription.mutated_sri:.4f}")
    print(f"Improvement : {result.prescription.sri_improvement:.4f}")
    print(f"Accepted    : {result.prescription.accepted}")
```

### 4.2 Using Client

For low-level execution control, instantiate the `Client` facade:

```python
from saag.client import Client

client = Client()

# Run baseline analysis
analysis = client.analyze(layer="system")

# Generate prescriptive optimizations and run closed-loop simulation
prescription = client.prescribe(
    analysis_result=analysis,
    layer="system"
)

# Access compiled policies
print("Topic splits compiled:", len(prescription.policy.topic_splits))
print("Accepted:", prescription.accepted)
print("Applied modifications:")
for change in prescription.applied_changes:
    print(f" - {change}")
```

---

## 5. Data Schema & Output Models

### 5.1 PrescriptionPolicy Schema

```json
{
  "topic_splits": [
    {
      "topic": "T1",
      "publishers": ["AppA", "AppC"],
      "subscribers": ["AppB", "AppD"]
    }
  ],
  "node_reallocations": [
    {
      "component": "AppB",
      "from_node": "NodeMain",
      "to_node": "NodeMain_AppB"
    }
  ],
  "qos_upgrades": [
    {
      "topic": "T1",
      "original_reliability": "BEST_EFFORT",
      "original_durability": "VOLATILE",
      "target_reliability": "RELIABLE",
      "target_durability": "TRANSIENT"
    }
  ]
}
```

### 5.2 PrescribeResult Schema

```json
{
  "original_sri": 0.4352,
  "mutated_sri": 0.3120,
  "sri_improvement": 0.1232,
  "original_metrics": {
    "sri": 0.4352,
    "avg_reachability_loss": 0.5230,
    "avg_fragmentation": 0.1540,
    "avg_throughput_loss": 0.3250
  },
  "mutated_metrics": {
    "sri": 0.3120,
    "avg_reachability_loss": 0.3840,
    "avg_fragmentation": 0.0820,
    "avg_throughput_loss": 0.2130
  },
  "policy": { ... },
  "applied_changes": [
    "Split topic 'T1' into sub-topics per publisher: AppA, AppC",
    "Moved process 'AppB' from SPOF node 'NodeMain' to isolated node 'NodeMain_AppB'",
    "Hardened QoS on topic 'T1': Reliability -> RELIABLE, Durability -> TRANSIENT"
  ],
  "accepted": true
}
```
