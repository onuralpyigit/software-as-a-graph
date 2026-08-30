# Detecting Architectural Anti-Patterns and Prescriptive Refactoring

**How Software-as-a-Graph goes from a flagged structural smell to a verified remediation.**

This document bridges [Step 3: Predict](prediction.md) (anti-pattern detection) and [Step 6: Prescribe](prescription.md) (closed-loop remediation). For the full formal specification of each anti-pattern, see [antipatterns.md](antipatterns.md); for the full prescription API and schema, see [prescription.md](prescription.md). This page focuses on how the two connect — and where they don't.

---

## 1. Detection: the 19-pattern catalog

`AntiPatternDetector` (`saag/analysis/antipattern_detector.py`) evaluates the 11 Tier-1 structural metric vector $M(v)$ (see [structural-analysis.md](structural-analysis.md)) and the derived RM criticality scores against a catalog of **19** anti-patterns, each with a severity tier and a formal detection rule.

A key design property: thresholds are **population-relative, not universal**. Most detectors compare a component's metric against an adaptive box-plot fence (`Q3 + 1.5 × IQR`) computed over the *current system's own* metric distribution, not a fixed constant. A 300-component enterprise system and a 15-component ROS 2 stack get different absolute cutoffs for the same pattern, because "anomalous" is defined relative to each system's own population.

| Severity | Patterns |
|---|---|
| **CRITICAL** | `SPOF`, `SYSTEMIC_RISK`, `GOD_COMPONENT`, `FAILURE_HUB`, `COMPOUND_RISK` |
| **HIGH** | `CYCLE`, `BRIDGE_EDGE`, `BOTTLENECK_EDGE`, `BROKER_OVERLOAD`, `DEEP_PIPELINE` |
| **MEDIUM** | `CONCENTRATION_RISK`, `TOPIC_FANOUT`, `CHATTY_PAIR`, `QOS_MISMATCH`, `ORPHANED_TOPIC`, `UNSTABLE_INTERFACE`, `HUB_AND_SPOKE`, `CHAIN`, `ISOLATED` |

Every entry carries a `PatternSpec.recommendation` string — narrative remediation guidance, reproduced in each pattern's `### 5.N` section of [antipatterns.md](antipatterns.md). That guidance exists for all 19 patterns. It is advice for a human, not code that runs — that distinction is the subject of the next section.

---

## 2. Prescription: three automated operators

`compile_policy` (`saag/prescription/rules.py`) compiles a mutation policy $\Delta(G)$ from exactly **three** rule-based operators:

| Operator | Trigger | What it does |
|---|---|---|
| **1. Logical topic splitting** | Topic is congested (>1 publisher and >1 subscriber), or CRITICAL/HIGH with >1 publisher, or connected to a component whose detected-problem name matches `"God Component"`, `"Bottleneck"`, or `"Hub"` | Splits the topic into per-publisher sub-topics, rewiring `publishes_to`/`subscribes_to`/routing |
| **2. Physical anti-affinity reallocation** | A Node (or something it hosts) is CRITICAL/HIGH or matches a detected-problem name containing `"SPOF"`/`"Single Point of Failure"`, and the node hosts >1 process | Moves all but the first hosted component to a newly cloned node, duplicating `CONNECTS_TO` links for reachability |
| **3. Transport QoS hardening** | A topic is CRITICAL/HIGH or connects to a CRITICAL/HIGH component, and its transport uses non-`RELIABLE` reliability or `VOLATILE` durability | Upgrades the contract to `RELIABLE` reliability / `TRANSIENT` durability |

### 2.1 Automation coverage is narrower than it looks

Two separate signals feed these triggers, and only one of them ties back to specific catalog IDs:

- **Generic criticality tier** — any component classified `CRITICAL`/`HIGH` on the RM dimensional scale can trigger any operator, regardless of which (if any) specific anti-pattern was detected on it.
- **Detected-problem name matching** (`rules.py`, `_smells`) — the only channel that links back to particular catalog entries, and it works by substring-matching `DetectedProblem.name` (the human-readable `PatternSpec.name`), not a dedicated pattern-ID field.

Following that name-matching channel through to the catalog, only **5 of the 19** patterns are directly wired into an operator:

| Catalog ID | Operator reached | How |
|---|---|---|
| `SPOF` | 2 (anti-affinity) | name contains `"SPOF"` / `"Single Point of Failure"` |
| `GOD_COMPONENT` | 1 (topic split) | name contains `"God Component"` / `"Bottleneck"` |
| `BOTTLENECK_EDGE` | 1 (topic split) | name contains `"Bottleneck"` |
| `FAILURE_HUB` | 1 (topic split) | name contains `"Hub"` |
| `HUB_AND_SPOKE` | 1 (topic split) | name contains `"Hub"` |

Notably, **`QOS_MISMATCH` has no link to Operator 3** despite the obvious conceptual overlap — QoS hardening fires only from the generic criticality tier, never from the `QOS_MISMATCH` detection itself. The remaining 14 patterns (`BRIDGE_EDGE`, `BROKER_OVERLOAD`, `CONCENTRATION_RISK`, `DEEP_PIPELINE`, `TOPIC_FANOUT`, `QOS_MISMATCH`, `CHATTY_PAIR`, `ORPHANED_TOPIC`, `UNSTABLE_INTERFACE`, `CYCLE`, `CHAIN`, `ISOLATED`, `SYSTEMIC_RISK`, `COMPOUND_RISK`) have **no automated operator at all** — their `PatternSpec.recommendation` text in [antipatterns.md](antipatterns.md) is advisory-only, for a human to act on (interface extraction, mediator components, stage merging, cycle-breaking via events, redundancy injection, and similar remediations that require semantic — not purely topological — judgment).

This is a principled boundary, not an oversight: the three operators only automate remediations expressible as pure topology/QoS mutations. Remediations that require understanding *what* a component does (breaking a cycle correctly, deciding which pipeline stages are safe to merge) stay advisory.

---

## 3. The Triage Bridge: Stakeholder-Oriented Remediation Routing

The **Triage Bridge** (`saag.analysis.triage.triage()` / `TriageUseCase` / `triage_presenter.py`) connects Step 3's high-risk shortlist (Top-$K$ ranked components) to targeted architectural root causes, routing recommendations to three distinct stakeholder groups:

| Stakeholder Role | Primary Focus & Domain | Associated Patterns & Metrics | Concrete Remediation Actions |
|:---|:---|:---|:---|
| **DevOps / SRE** | Infrastructure locality & broker resilience | `SPOF`, `BROKER_OVERLOAD`, $AP_c^{\text{dir}}$, host co-location | Configure Kubernetes pod anti-affinity, replicate message brokers, reallocate co-located high-risk services |
| **System Architect** | Pub-sub topology & transport contracts | `GOD_COMPONENT`, `FAILURE_HUB`, `BOTTLENECK_EDGE`, `HUB_AND_SPOKE`, `CYCLE`, $CDI$ | Apply automated Operator 1 (Topic Splitting), Operator 3 (Transport QoS Hardening), and insert circuit breakers or event bridges |
| **Software Developer** | Code complexity & component coupling | `CYCLIC_DEPENDENCY`, High $CQP$, High $MPCI$, High $PC$ | Refactor god classes, decompose high-cyclomatic-complexity methods, prune redundant transitive library imports |

---

## 4. The in-silico trial: closed-loop verification

Every compiled policy is verified before it is ever reported as viable — and it is *never* applied to the live system:

1. **Baseline** — the source graph runs through analyze → simulate → validate, producing a baseline System Risk Index (SRI) and a per-component cascade impact map $I(v)$.
2. **Per-edit acceptance filter** — every candidate operator is applied *alone* to a counterfactual copy of the graph and simulated across a propagation-threshold sweep and a seed set. It is kept only if its mean impact reduction clears a margin over seed noise, $\Delta I > \kappa \cdot \sigma_{\text{seed}}$, at **every** threshold. Rejected operators never reach the mutated graph.
3. **Mutate in memory** — the graph is exported to flat JSON, the *accepted subset* of $\Delta(G)$ is applied to that JSON (never to the production Neo4j graph), producing $G'$.
4. **Sandbox reload** — $G'$ is loaded into a temporary `MemoryRepository`; dependency edges are re-derived from scratch.
5. **Re-run the full suite** — analyze → simulate → validate re-executes on $G'$, under the same fault scenarios and seeds as the baseline.
6. **Accept/reject gate** — $\Delta\text{SRI} = \text{SRI}_{\text{baseline}} - \text{SRI}_{\text{mutated}}$; the policy is marked `accepted = true` iff $\Delta\text{SRI} > 0$.

Both criteria are implemented — see [prescription.md](prescription.md#3-closed-loop-verification-mechanics) for the exact acceptance rule and its parameters. The per-edit filter (`saag/prescription/verifier.py`) is what stops an individually harmful operator riding along with helpful ones; the whole-policy gate in step 6 is what catches a regression in the *interaction* between operators that each passed on their own.

A rejected policy is still returned in full, with its before/after metrics, for the architect to inspect — nothing is silently discarded.

---

## 5. From blueprint to deployment

The output of a `prescribe()` call is a remediation blueprint: an itemized `applied_changes` list, per-edit verdicts for everything that was declined, and before/after metrics (reachability loss, fragmentation, throughput loss). It is reachable from the SDK and the CLI — Stage 6 has no REST router and is not rendered in the SMART dashboard. The architect is the one who turns this into real deployment artifacts — topic redesign in middleware config, Kubernetes anti-affinity scheduling constraints, DDS/MQTT QoS profile changes. The framework diagnoses and simulates the treatment; it never administers it to the live system.

---

## 6. Summary table: diagnosis → automated vs. advisory remediation

| Catalog ID | Severity | Remediation | Primary Stakeholder |
|---|---|---|---|
| `SPOF` | CRITICAL | **Automated** (Operator 2: Anti-Affinity) + advisory failover | DevOps / SRE |
| `GOD_COMPONENT` | CRITICAL | **Automated** (Operator 1: Topic Splitting) | System Architect |
| `FAILURE_HUB` | CRITICAL | **Automated** (Operator 1: Topic Splitting) | System Architect |
| `BOTTLENECK_EDGE` | HIGH | **Automated** (Operator 1: Topic Splitting) | System Architect |
| `HUB_AND_SPOKE` | MEDIUM | **Automated** (Operator 1: Topic Splitting) | System Architect |
| `BROKER_OVERLOAD` | HIGH | Advisory (Cluster replication / Broker scale-out) | DevOps / SRE |
| `QOS_MISMATCH` | MEDIUM | **Automated** (Operator 3: Transport QoS Hardening) via Tier | System Architect |
| `SYSTEMIC_RISK`, `COMPOUND_RISK` | CRITICAL | Advisory only | System Architect & SRE |
| `CYCLE`, `BRIDGE_EDGE`, `DEEP_PIPELINE` | HIGH | Advisory only | System Architect |
| `CONCENTRATION_RISK`, `TOPIC_FANOUT`, `CHATTY_PAIR`, `ORPHANED_TOPIC`, `UNSTABLE_INTERFACE`, `CHAIN`, `ISOLATED` | MEDIUM | Advisory only | Software Developer & Architect |

See [antipatterns.md](antipatterns.md) for each pattern's full specification and remediation narrative, and [prescription.md](prescription.md) for the operator implementation, schema, and API.
