# Step 6: Prescribe

**Rule-based architectural refactoring recommendations, verified edit by edit in closed-loop simulation.**

← [Step 5: Validate](validation.md) | → [Step 7: Visualize](visualization.md)

---

## Table of Contents

1. [What This Step Does](#1-what-this-step-does)
2. [Preservation & Remediation Rules](#2-preservation--remediation-rules)
   - 2.1 [Logical Subgraph Refactoring (Topic Splitting)](#21-logical-subgraph-refactoring-topic-splitting)
   - 2.2 [Physical Locality Anti-Affinity Rules](#22-physical-locality-anti-affinity-rules)
   - 2.3 [Middleware Transport Contract Hardening](#23-middleware-transport-contract-hardening)
3. [Closed-Loop Verification Mechanics](#3-closed-loop-verification-mechanics)
   - 3.1 [The Acceptance Rule](#31-the-acceptance-rule)
   - 3.2 [Cost Model](#32-cost-model)
4. [Programmatic API Reference](#4-programmatic-api-reference)
   - 4.1 [Using Pipeline](#41-using-pipeline)
   - 4.2 [Using Client](#42-using-client)
   - 4.3 [Filter Parameters](#43-filter-parameters)
5. [Data Schema & Output Models](#5-data-schema--output-models)
   - 5.1 [PrescriptionPolicy Schema](#51-prescriptionpolicy-schema)
   - 5.2 [PrescribeResult Schema](#52-prescriberesult-schema)
6. [Commands](#6-commands)
7. [Known Limitations](#7-known-limitations)
8. [What Comes Next](#8-what-comes-next)

---

## 1. What This Step Does

Step 6 (Prescribe) closes the optimization loop. Once high-risk components and topological
anti-patterns (Single Points of Failure, god components, bottleneck topic hubs) have been
scored by the Predict stage and validated against cascade-simulation ground truth by the
Validate stage, Prescribe compiles an optimization policy $\Delta(G)$ from three rule-based
transformations, verifies each proposed edit on its own, and applies only those that survive.

```
   Validated risks (Step 5)
             │
             ▼
   Compile candidate Δ(G)          rules.py
   1. Topic splitting
   2. Physical host isolation
   3. Transport QoS hardening
             │
             ▼
   Per-edit acceptance filter      verifier.py
   for each candidate edit e:
     simulate G+{e} alone over
     thresholds × seeds, paired
     against the baseline at the
     same (θ, seed)
     keep iff ΔI > κ·σ at EVERY θ
             │
             ▼
   Apply the accepted subset       mutator.py
   to an in-memory copy of G
             │
             ▼
   Closed-loop validation (ΔSRI)   evaluator.py
             │
             ▼
   Remediation blueprint
   (per-edit verdicts + baseline SRI → mutated SRI)
```

To avoid contaminating the target database or paying transaction overhead, the refactoring
rules are applied in memory to a JSON export of the graph, which is then loaded into a
throwaway `MemoryRepository` for evaluation.

**Each candidate edit is verified on its own before anything is applied.** A compiled policy
is a *candidate* set. [`EditVerifier.verify`](../saag/prescription/verifier.py) builds a
counterfactual graph containing that edit alone, simulates it across the propagation-threshold
sweep and the seed set, and keeps it only when its mean impact reduction clears `κ · σ_seed` at
**every** threshold. Requiring it everywhere stops an edit being accepted because it happened to
help at the canonical 0.2 default; requiring it to beat σ stops simulator noise being read as
improvement.

> **Why this exists.** The policy used to be applied wholesale and judged by a single end-state
> SRI check, so an edit that made the system worse could ride along with edits that made it
> better — the aggregate improved while individual components regressed. With the filter, a
> regressing edit is rejected individually and never reaches the mutated graph.
>
> **An empty result is a valid result.** It is legitimate for *no* candidate to clear the bar,
> particularly on small topologies. `PrescribeResult` then reports `applied_changes == []` and
> `sri_improvement == 0.0`. That is the filter working, not a failure. `edit_verdicts` carries
> every candidate with its per-threshold statistics and rejection `reason`, so a run always
> states what it declined and why.

### Module layout

| Module | Responsibility |
|:--|:--|
| [rules.py](../saag/prescription/rules.py) | Compiles $\Delta(G)$ from analysis + prediction. Pure — no repository, no simulation. |
| [mutator.py](../saag/prescription/mutator.py) | `apply_policy(json, policy) -> json`. Pure JSON rewriting. |
| [evaluator.py](../saag/prescription/evaluator.py) | Runs Analyze → Predict → Simulate → Validate over a repository. |
| [verifier.py](../saag/prescription/verifier.py) | The per-edit acceptance filter and its arithmetic. |
| [service.py](../saag/prescription/service.py) | Orchestrates the five steps above. |
| [models.py](../saag/prescription/models.py) | `TopicSplit`, `NodeReallocation`, `QosUpgrade`, `EditVerdict`, `PrescriptionPolicy`, `PrescribeResult`. |

---

## 2. Preservation & Remediation Rules

Remediations target components the Predict stage scored `CRITICAL`/`HIGH` on any RMAV dimension
(reliability, maintainability, availability, security, or the aggregate). Node reallocation
([§2.2](#22-physical-locality-anti-affinity-rules)) additionally considers components flagged as
SPOF or god-component smells among `prediction_result.problems`.

**Automation coverage.** [antipatterns.md](antipatterns.md) documents remediation guidance for
all 21 catalog anti-patterns, but that guidance is advisory unless it maps to one of the three
operators below. Only 5 catalog IDs feed an operator, and they do so through a substring match
on the detected problem's *display name* rather than a dedicated pattern-ID field:

| Catalog ID | `PatternSpec.name` | Matched substring | Feeds |
|:--|:--|:--|:--|
| `SPOF` | Single Point of Failure (SPOF) | `SPOF` | §2.2 Node reallocation |
| `GOD_COMPONENT` | God Component / Central Bottleneck | `God Component` | §2.1 Topic splitting |
| `BOTTLENECK_EDGE` | Bottleneck Dependency | `Bottleneck` | §2.1 Topic splitting |
| `FAILURE_HUB` | Critical Failure Propagation Hub | `Hub` | §2.1 Topic splitting |
| `HUB_AND_SPOKE` | Hub-and-Spoke Anti-Pattern | `Hub` | §2.1 Topic splitting |

Every operator can also fire independently off the generic RMAV `CRITICAL`/`HIGH` tier,
regardless of which (if any) specific anti-pattern was detected. The remaining 16 catalog IDs
have no automated operator at all. See [remediation.md](remediation.md) for the full
advisory-versus-automated breakdown.

> Because the link is a name substring match, renaming a `PatternSpec` silently unlinks the
> operator it feeds. `tests/test_antipattern_detector.py::test_catalog_names_avoid_prescription_collision`
> guards the reverse direction — a new pattern accidentally acquiring "Bottleneck" or "Hub".

### 2.1 Logical Subgraph Refactoring (Topic Splitting)

**Problem**: A central topic multiplexing several publishers into several subscribers is a
logical bottleneck and a high-risk failure propagator.

**Trigger**: a topic $T$ matching any of three disjunctive conditions —

* $T$ has more than one publisher **and** more than one subscriber (congestion); or
* $T$ is itself `CRITICAL`/`HIGH` **and** has more than one publisher; or
* $T$ has more than one publisher **and** any of its publishers or subscribers is a
  god-component smell.

**Remediation**: split $T$ into dedicated sub-topics per publisher.

* For each publisher $P_i$ publishing to $T$, create a new sub-topic $T_{P_i}$.
* Re-route $P_i \rightarrow T$ to $P_i \rightarrow T_{P_i}$.
* Re-route every subscriber $S_j$ to the full set of sub-topics: $S_j \rightarrow T_{P_i}$ for
  all $P_i$, so no subscriber loses a channel it previously had.
* Duplicate broker `ROUTES` links across all sub-topics.

This bounds failure propagation by separating independent logical communication channels.

### 2.2 Physical Locality Anti-Affinity Rules

**Problem**: Multiple processes (Applications or Brokers) co-located on a single physical host
$N$ flagged as a SPOF or critical risk. If $N$ fails, all hosted components fail together.

**Trigger**: $N$ hosts more than one process **and** $N$ itself, or any process it hosts, is a
SPOF smell or scored `CRITICAL`/`HIGH`.

**Remediation**: establish scheduling anti-affinity to isolate the co-located processes.

* Allocate a new node instance $N_{C_i}$ for each co-located process $C_i$ *except the first*.
  Hosted ids are sorted, so "first" means lexicographically first and the choice is
  reproducible across runs.
* Update `RUNS_ON` to reallocate $C_i$ to $N_{C_i}$.
* Duplicate the host's `CONNECTS_TO` links in both directions so network reachability is
  preserved for the relocated process.

### 2.3 Middleware Transport Contract Hardening

**Problem**: Critical communication channels running on unreliable or volatile transport
configurations (e.g. ROS 2 `BEST_EFFORT` reliability or `VOLATILE` durability).

**Trigger**: a topic $T$ that is `CRITICAL`/`HIGH`, or has any publisher or subscriber that is,
**and** is not already hardened — i.e. `qos_reliability != "RELIABLE"` or
`qos_durability == "VOLATILE"`. An already-hardened topic produces no upgrade, so a compiled
upgrade always changes something.

**Remediation**: set `qos_reliability = "RELIABLE"` and `qos_durability = "TRANSIENT"`, written
to both the nested `qos` block and the flat properties the layer projections read.

---

## 3. Closed-Loop Verification Mechanics

The verification engine executes the following loop:

1. **Compile candidates.** `rules.py` produces the candidate $\Delta(G)$ from the raw graph,
   the analysis result and the prediction result.
2. **Evaluate the baseline.** `GraphEvaluator.evaluate` runs Analyze → Predict → Simulate →
   Validate on the source repository at the canonical point ($\theta = 0.2$, seed 42), yielding
   $\text{SRI}_{\text{baseline}}$, the aggregate metrics, and the per-component impact map
   $I(v)$. One simulation sweep feeds all three, so the reported SRI and the impact map
   provably describe the same run.
3. **Sweep the baseline grid.** `EditVerifier` measures the *unmutated* impact map at every
   $(\theta, \text{seed})$ point of the sweep. Computed once and shared by every edit.
4. **Filter per edit.** For each candidate: export $G$ to JSON, apply that edit *alone*, load
   the result into a throwaway `MemoryRepository`, and measure its impact map across the same
   grid. Each mutated run is differenced against the baseline at its **own**
   $(\theta, \text{seed})$ — see [§3.1](#31-the-acceptance-rule).
5. **Apply the accepted subset.** `PrescriptionPolicy.from_edits` rebuilds a policy from the
   passing edits only; `mutator.apply_policy` writes them into a fresh JSON copy. If nothing
   passed, the run returns the unchanged baseline with a full set of verdicts.
6. **Evaluate the mutated graph** the same way as step 2, and compute

   $$\Delta \text{SRI} = \text{SRI}_{\text{baseline}} - \text{SRI}_{\text{mutated}}$$

   SRI is a risk index (lower is better), so $\Delta \text{SRI} > 0$ means the mutated topology
   carries less structural risk.
7. **Whole-policy gate.** If $\Delta \text{SRI} > 0$ the result is marked `accepted = true`.
   Otherwise `accepted = false` and the policy is still returned in full for inspection — it is
   not discarded or retried. Because each edit already passed its own counterfactual test, a
   negative $\Delta \text{SRI}$ here means the *interaction* between individually-verified edits
   regressed, not that an unverified edit was applied.

### 3.1 The Acceptance Rule

An edit is **accepted** iff its sweep completed at every threshold and, for **every**
$\theta \in \text{thresholds}$:

$$\operatorname{mean}_{s}\!\left[\Delta I(\theta, s)\right] > \kappa \cdot \operatorname{stdev}_{s}\!\left[\Delta I(\theta, s)\right]$$

where

$$\Delta I(\theta, s) = \operatorname{mean}_{v \in V_{\text{common}}}\!\left[I_{\text{baseline}}(v; \theta, s) - I_{\text{mutated}}(v; \theta, s)\right]$$

Three properties this rule depends on:

* **Differences are paired.** Baseline and mutated impact are always taken at the *same*
  $(\theta, s)$. Differencing against a single fixed point would fold the threshold's own large
  effect into every delta, swamping the effect of the edit being judged.
* **σ is per-threshold.** The standard deviation is taken across seeds *within* one threshold.
  Pooling seeds and thresholds into one σ measures threshold sensitivity, not simulator noise,
  and inflates the bar accordingly.
* **$V_{\text{common}}$ is the intersection.** Only components present before *and* after are
  differenced, because a topic split renames its target and leaves no stable counterpart.

With a single seed — or with several seeds that agree exactly, which is common because the
simulator is deterministic for many edits — $\sigma = 0$ and the rule degenerates to a strictly
positive mean regardless of `kappa` (see [L5](#7-known-limitations)). The binding threshold,
$\arg\min_\theta (\text{mean} - \kappa\sigma)$, is named in the rejection `reason` so a report
says which threshold actually blocked the edit.

### 3.2 Cost Model

Verification is the expensive part of Stage 6. It costs one exhaustive failure-simulation sweep
per grid point:

| Phase | Sweeps |
|:--|:--|
| Baseline evaluation (SRI + metrics + $I(v)$) | 1 |
| Baseline grid, shared across all edits | `len(thresholds) × len(seeds)` = 9 |
| Per candidate edit | `len(thresholds) × len(seeds)` = 9 |
| Mutated evaluation (only if something passed) | 1 |

So a scenario with $n$ candidate edits costs roughly $9n + 11$ exhaustive sweeps at the
defaults. Reduce `--seeds` or `--thresholds` to trade statistical strength for runtime.

---

## 4. Programmatic API Reference

### 4.1 Using Pipeline

The easiest way to trigger the Prescribe stage is the fluent pipeline builder:

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
    p = result.prescription
    print(f"Baseline SRI: {p.original_sri:.4f}")
    print(f"Mutated SRI : {p.mutated_sri:.4f}")
    print(f"Improvement : {p.sri_improvement:.4f}")
    print(f"Edits       : {p.n_accepted} accepted / {p.n_rejected} rejected")
    print(f"Accepted    : {p.accepted}")
```

### 4.2 Using Client

For low-level execution control, instantiate the `Client` facade:

```python
from saag.client import Client

client = Client()

analysis = client.analyze(layer="system")
prediction = client.predict(analysis)

prescription = client.prescribe(
    analysis_result=analysis,
    prediction_result=prediction,
    layer="system",
    kappa=1.0,
)

print("Candidate splits:", len(prescription.candidate_policy.topic_splits))
print("Applied modifications:")
for change in prescription.applied_changes:
    print(f" - {change}")

for verdict in prescription.edit_verdicts:
    if not verdict.accepted:
        print(f" declined {verdict.kind}/{verdict.target}: {verdict.reason}")
```

> Passing `prediction_result` matters. Without it the rules see no GNN criticality scores and
> no anti-pattern smells, and fire only off whatever criticality the analysis result carries.

### 4.3 Filter Parameters

`kappa`, `seeds` and `thresholds` are forwarded through `Client.prescribe(**kwargs)` and
`PrescribeGraphUseCase.execute` to `PrescribeService.prescribe`. Defaults live in
[verifier.py](../saag/prescription/verifier.py).

| Parameter | Default | Meaning |
|:--|:--|:--|
| `kappa` | `1.0` | Acceptance multiple. An edit must beat `κ · σ_seed` at every threshold, not merely register a positive delta. Raise it to demand a larger margin over simulator noise. |
| `seeds` | `(42, 123, 456)` | Simulation seeds used to estimate σ. Three rather than five: because baseline and mutated runs are paired on `(θ, seed)`, shared noise largely cancels, and a third seed already pins the residual spread. |
| `thresholds` | `(0.1, 0.2, 0.5)` | Propagation thresholds the edit must clear at *every* value, so acceptance is not an artifact of the 0.2 default. |

The canonical evaluation point for reported SRI and metrics is $\theta = 0.2$, seed 42, set
explicitly in `evaluator.CANONICAL_THRESHOLD` / `CANONICAL_SEED`.

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
  "policy": { "...": "the accepted subset, §5.1 shape" },
  "candidate_policy": { "...": "the full candidate set before filtering, §5.1 shape" },
  "applied_changes": [
    "Split topic 'T1' into sub-topics per publisher: AppA, AppC",
    "Moved process 'AppB' from SPOF node 'NodeMain' to isolated node 'NodeMain_AppB'",
    "Hardened QoS on topic 'T1': Reliability -> RELIABLE, Durability -> TRANSIENT"
  ],
  "remediated_component_impact_deltas": {
    "AppB": { "before": 0.4210, "after": 0.2870, "reduction_frac": 0.3183 }
  },
  "mean_cascade_impact_reduction": 0.3183,
  "accepted": true,
  "n_candidate_edits": 12,
  "n_accepted_edits": 3,
  "n_rejected_edits": 9,
  "edit_verdicts": [
    {
      "schema": 2,
      "kind": "qos_upgrade",
      "target": "T1",
      "kappa": 1.0,
      "accepted": false,
      "reason": "delta 0.000012 <= kappa*sigma 0.000031 at threshold 0.5",
      "worst_delta": 0.000012,
      "per_threshold": {
        "0.1": { "mean_delta": 0.000180, "sigma_seed": 0.000021 },
        "0.2": { "mean_delta": 0.000094, "sigma_seed": 0.000019 },
        "0.5": { "mean_delta": 0.000012, "sigma_seed": 0.000031 }
      }
    }
  ]
}
```

`kind` is one of `topic_split`, `node_reallocation`, `qos_upgrade`. `worst_delta` is the
smallest `mean_delta` over the sweep; acceptance is decided on
`min(mean_delta - kappa * sigma_seed)`, which is exposed on the Python object as
`EditVerdict.worst_margin`.

> **`edit_verdicts` schema 2.** The scalar `delta_impact` and `sigma_seed` fields of schema 1
> are gone: there is no longer a single σ, because σ is computed per threshold. `per_threshold`
> values changed from bare floats to `{mean_delta, sigma_seed}` objects.

---

## 6. Commands

```bash
# ─── As part of the full pipeline ─────────────────────────────────────────────

saag --prescribe --layer system

# ─── Standalone (no console-script entry point; invoke the module directly) ────

PYTHONPATH=. python cli/prescribe_graph.py --input data/scenarios/atm_system.json --layer system
PYTHONPATH=. python cli/prescribe_graph.py --layer system --output output/prescribe.json

# Tune the per-edit acceptance filter
PYTHONPATH=. python cli/prescribe_graph.py --input data/scenarios/atm_system.json \
    --kappa 1.0 --thresholds 0.1 0.2 0.5 --seeds 42 123 456

# ─── Batch across the seven benchmark scenarios ───────────────────────────────

PYTHONPATH=. python reproduce/run_prescribe_all.py --kappa 1.0 --output results/prescribe_all.json
```

| Flag | Meaning |
|:--|:--|
| `--input`, `-i` | Run against a topology JSON in memory instead of Neo4j. |
| `--layer` | Analysis layer (`app`, `infra`, `mw`, `system`). |
| `--gnn-checkpoint` | Trained GNN checkpoint directory, forwarded to the evaluator's Predict stage. |
| `--kappa` | Acceptance multiple (§4.3). |
| `--seeds` | Simulation seeds for the noise estimate (§4.3). |
| `--thresholds` | Propagation thresholds the edit must clear at every value (§4.3). |
| `--output` | Write the full `PrescribeResult` JSON (§5.2). |

Full flag reference: [cli-pipeline-guide.md](cli-pipeline-guide.md).

---

## 7. Known Limitations

Documented so they aren't mistaken for working code.

| # | Limitation | Impact |
|:--|------------|--------|
| L1 | **Topic splits are excluded from `mean_cascade_impact_reduction`.** A split replaces the original topic id with per-publisher sub-topics. | There is no stable before/after counterpart to difference, so split targets are omitted from `remediated_component_impact_deltas` rather than approximated. On a policy of splits only, `mean_cascade_impact_reduction` is `null`. |
| L2 | **Anti-patterns link to operators by display-name substring**, not by pattern ID ([§2](#2-preservation--remediation-rules)). | Renaming a `PatternSpec` silently unlinks its operator. Only 5 of 21 catalog IDs are linked at all. |
| L3 | **§2.3 has no link to `QOS_MISMATCH`** despite the conceptual overlap. QoS hardening fires only from the generic RMAV criticality tier. | A detected `QOS_MISMATCH` does not by itself produce an upgrade. |
| L4 | **The whole-policy gate is reported, not enforced** ([§3](#3-closed-loop-verification-mechanics) step 7). A policy with `accepted = false` is still returned in full, and the mutation is not rolled back in the result object. | Callers must check `accepted` themselves. Nothing is written to the source repository either way — the mutation only ever exists in a sandbox. |
| L5 | **`kappa` has no effect on an edit whose seed spread is exactly zero.** The simulator is deterministic for many edits, giving $\sigma = 0$ at every threshold. | The bar $\kappa \cdot \sigma$ collapses to 0 and the rule degenerates to "mean reduction > 0" no matter how large `kappa` is. Raising `kappa` only filters edits whose measured deltas actually vary across seeds. |
| L6 | **Verification cost is linear in candidate count** ([§3.2](#32-cost-model)). | Large scenarios with tens of candidates run hundreds of exhaustive sweeps. |
| L7 | **No REST surface.** Stage 6 has no router in `api/routers/` and no presenter. | Prescribe is reachable from the SDK and the CLI only. |

---

## 8. What Comes Next

Prescribe consumes the validated risk scores from Step 5 and the cascade oracle from Step 4; it
produces a remediation blueprint rather than a change applied to the source graph. Feed the
accepted policy back into the topology by hand, or re-import the mutated JSON and re-run the
pipeline to confirm the improvement independently.

→ [Step 7: Visualize](visualization.md)

---

← [Step 5: Validate](validation.md) | → [Step 7: Visualize](visualization.md)
