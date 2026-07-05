# Detecting Architectural Anti-Patterns and Prescriptive Refactoring

**How Software-as-a-Graph goes from a flagged structural smell to a verified remediation.**

This document bridges [Step 2: Analyze](structural-analysis.md) (anti-pattern detection) and [Step 6: Prescribe](prescription.md) (closed-loop remediation). For the full formal specification of each anti-pattern, see [antipatterns.md](antipatterns.md); for the full prescription API and schema, see [prescription.md](prescription.md). This page focuses on how the two connect — and where they don't.

---

## 1. Detection: the 21-pattern catalog

`AntiPatternDetector` (`saag/analysis/antipattern_detector.py`) evaluates the 13-element structural metric vector $M(v)$ (see [structural-analysis.md](structural-analysis.md)) and the derived RMAV criticality scores against a catalog of **21** anti-patterns, each with a severity tier and a formal detection rule.

A key design property: thresholds are **population-relative, not universal**. Most detectors compare a component's metric against an adaptive box-plot fence (`Q3 + 1.5 × IQR`) computed over the *current system's own* metric distribution, not a fixed constant. A 300-component enterprise system and a 15-component ROS 2 stack get different absolute cutoffs for the same pattern, because "anomalous" is defined relative to each system's own population.

| Severity | Patterns |
|---|---|
| **CRITICAL** | `SPOF`, `SYSTEMIC_RISK`, `GOD_COMPONENT`, `FAILURE_HUB`, `TARGET`, `COMPOUND_RISK` |
| **HIGH** | `CYCLE`, `BRIDGE_EDGE`, `BOTTLENECK_EDGE`, `BROKER_OVERLOAD`, `DEEP_PIPELINE`, `EXPOSURE` |
| **MEDIUM** | `CONCENTRATION_RISK`, `TOPIC_FANOUT`, `CHATTY_PAIR`, `QOS_MISMATCH`, `ORPHANED_TOPIC`, `UNSTABLE_INTERFACE`, `HUB_AND_SPOKE`, `CHAIN`, `ISOLATED` |

Every entry carries a `PatternSpec.recommendation` string — narrative remediation guidance, reproduced in each pattern's `### 5.N` section of [antipatterns.md](antipatterns.md). That guidance exists for all 21 patterns. It is advice for a human, not code that runs — that distinction is the subject of the next section.

---

## 2. Prescription: three automated operators

`PrescribeService` (`saag/prescription/service.py`) compiles a mutation policy $\Delta(G)$ from exactly **three** rule-based operators:

| Operator | Trigger | What it does |
|---|---|---|
| **1. Logical topic splitting** | Topic is congested (>1 publisher and >1 subscriber), or CRITICAL/HIGH with >1 publisher, or connected to a component whose detected-problem name matches `"God Component"`, `"Bottleneck"`, or `"Hub"` | Splits the topic into per-publisher sub-topics, rewiring `publishes_to`/`subscribes_to`/routing |
| **2. Physical anti-affinity reallocation** | A Node (or something it hosts) is CRITICAL/HIGH or matches a detected-problem name containing `"SPOF"`/`"Single Point of Failure"`, and the node hosts >1 process | Moves all but the first hosted component to a newly cloned node, duplicating `CONNECTS_TO` links for reachability |
| **3. Transport QoS hardening** | A topic is CRITICAL/HIGH or connects to a CRITICAL/HIGH component, and its transport uses non-`RELIABLE` reliability or `VOLATILE` durability | Upgrades the contract to `RELIABLE` reliability / `TRANSIENT` durability |

### 2.1 Automation coverage is narrower than it looks

Two separate signals feed these triggers, and only one of them ties back to specific catalog IDs:

- **Generic criticality tier** — any component classified `CRITICAL`/`HIGH` on the RMAV dimensional scale can trigger any operator, regardless of which (if any) specific anti-pattern was detected on it.
- **Detected-problem name matching** (`service.py:164-171`) — the only channel that links back to particular catalog entries, and it works by substring-matching `DetectedProblem.name` (the human-readable `PatternSpec.name`), not a dedicated pattern-ID field.

Following that name-matching channel through to the catalog, only **5 of the 21** patterns are directly wired into an operator:

| Catalog ID | Operator reached | How |
|---|---|---|
| `SPOF` | 2 (anti-affinity) | name contains `"SPOF"` / `"Single Point of Failure"` |
| `GOD_COMPONENT` | 1 (topic split) | name contains `"God Component"` / `"Bottleneck"` |
| `BOTTLENECK_EDGE` | 1 (topic split) | name contains `"Bottleneck"` |
| `FAILURE_HUB` | 1 (topic split) | name contains `"Hub"` |
| `HUB_AND_SPOKE` | 1 (topic split) | name contains `"Hub"` |

Notably, **`QOS_MISMATCH` has no link to Operator 3** despite the obvious conceptual overlap — QoS hardening fires only from the generic criticality tier, never from the `QOS_MISMATCH` detection itself. The remaining 16 patterns (`BRIDGE_EDGE`, `BROKER_OVERLOAD`, `CONCENTRATION_RISK`, `DEEP_PIPELINE`, `TOPIC_FANOUT`, `QOS_MISMATCH`, `CHATTY_PAIR`, `ORPHANED_TOPIC`, `UNSTABLE_INTERFACE`, `TARGET`, `EXPOSURE`, `CYCLE`, `CHAIN`, `ISOLATED`, `SYSTEMIC_RISK`, `COMPOUND_RISK`) have **no automated operator at all** — their `PatternSpec.recommendation` text in [antipatterns.md](antipatterns.md) is advisory-only, for a human to act on (interface extraction, mediator components, stage merging, cycle-breaking via events, redundancy injection, and similar remediations that require semantic — not purely topological — judgment).

This is a principled boundary, not an oversight: the three operators only automate remediations expressible as pure topology/QoS mutations. Remediations that require understanding *what* a component does (breaking a cycle correctly, deciding which pipeline stages are safe to merge) stay advisory.

---

## 3. The in-silico trial: closed-loop verification

Every compiled policy is verified before it is ever reported as viable — and it is *never* applied to the live system:

1. **Baseline** — the source graph runs through analyze → simulate → validate, producing a baseline System Risk Index (SRI).
2. **Mutate in memory** — the graph is exported to flat JSON, $\Delta(G)$ is applied to that JSON (never to the production Neo4j graph), producing $G'$.
3. **Sandbox reload** — $G'$ is loaded into a temporary `MemoryRepository`; dependency edges are re-derived from scratch.
4. **Re-run the full suite** — analyze → simulate → validate re-executes on $G'$, under the same fault scenarios and seeds as the baseline.
5. **Accept/reject gate** — $\Delta\text{SRI} = \text{SRI}_{\text{baseline}} - \text{SRI}_{\text{mutated}}$; the policy is marked `accepted = true` iff $\Delta\text{SRI} > 0$.

This is the criterion actually implemented (`service.py:78-79`) and the one documented in [prescription.md](prescription.md#3-closed-loop-verification-mechanics) — a **whole-policy** gate. There is no per-edit acceptance filter: all compiled operators in a policy are applied together and evaluated as one unit, so a policy can be accepted even if one of its operators is individually harmful, so long as the net effect is a lower SRI. A stricter per-edit filter — rejecting each mutation individually unless its improvement clears a margin over seed noise ($\Delta I > \kappa \cdot \sigma_{\text{seed}}$) — is a known, explicitly tracked future-work item, not a currently-enforced criterion.

A rejected policy is still returned in full, with its before/after metrics, for the architect to inspect — nothing is silently discarded.

---

## 4. From blueprint to deployment

The output of a `prescribe()` call is a remediation blueprint: an itemized `applied_changes` list plus before/after metrics (reachability loss, fragmentation, throughput loss), surfaced in the SMART dashboard ([visualization.md](visualization.md)). The architect is the one who turns this into real deployment artifacts — topic redesign in middleware config, Kubernetes anti-affinity scheduling constraints, DDS/MQTT QoS profile changes. The framework diagnoses and simulates the treatment; it never administers it to the live system.

---

## 5. Summary table: diagnosis → automated vs. advisory remediation

| Catalog ID | Severity | Remediation |
|---|---|---|
| `SPOF` | CRITICAL | **Automated** (Operator 2) + advisory redundancy/failover guidance beyond reallocation |
| `GOD_COMPONENT` | CRITICAL | **Automated** (Operator 1) |
| `FAILURE_HUB` | CRITICAL | **Automated** (Operator 1) |
| `BOTTLENECK_EDGE` | HIGH | **Automated** (Operator 1) |
| `HUB_AND_SPOKE` | MEDIUM | **Automated** (Operator 1) |
| `SYSTEMIC_RISK`, `TARGET`, `COMPOUND_RISK` | CRITICAL | Advisory only |
| `CYCLE`, `BRIDGE_EDGE`, `BROKER_OVERLOAD`, `DEEP_PIPELINE`, `EXPOSURE` | HIGH | Advisory only |
| `CONCENTRATION_RISK`, `TOPIC_FANOUT`, `CHATTY_PAIR`, `QOS_MISMATCH`, `ORPHANED_TOPIC`, `UNSTABLE_INTERFACE`, `CHAIN`, `ISOLATED` | MEDIUM | Advisory only |

See [antipatterns.md](antipatterns.md) for each pattern's full specification and remediation narrative, and [prescription.md](prescription.md) for the operator implementation, schema, and API.
