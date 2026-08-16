# The Reliability and Maintainability Quality Models

**The layer-by-layer reference for the RM quality model: every attribute, the measure that operationalizes it, its coefficient, its provenance, and its code home — for components and for relationships alike.**

[README](../README.md) | [criticality.md](criticality.md) | [structural-analysis.md](structural-analysis.md)

---

## Table of Contents

1. [Overview](#1-overview)
2. [The Four-Layer Model](#2-the-four-layer-model)
3. [Layer 0 — Quality Measure Elements (ISO/IEC 25021)](#3-layer-0--quality-measure-elements-isoiec-25021)
   - 3.1 [Topological Measure Elements](#31-topological-measure-elements)
   - 3.2 [Code Measure Elements](#32-code-measure-elements)
   - 3.3 [Declared QoS Measure Elements](#33-declared-qos-measure-elements)
4. [Layer 1 — Internal Quality Measures (ISO/IEC 25023)](#4-layer-1--internal-quality-measures-isoiec-25023)
   - 4.1 [Two Sources of Internal Evidence](#41-two-sources-of-internal-evidence)
   - 4.2 [The Nineteen Scoring Measures](#42-the-nineteen-scoring-measures)
   - 4.3 [Derived Composites](#43-derived-composites)
   - 4.4 [Normalisation](#44-normalisation)
   - 4.5 [What Layer 1 Leaves Out](#45-what-layer-1-leaves-out)
5. [Layer 2 — External Quality Attributes (ISO/IEC 25010:2023)](#5-layer-2--external-quality-attributes-isoiec-250102023)
   - 5.1 [Reliability](#51-reliability)
   - 5.2 [Maintainability](#52-maintainability)
   - 5.3 [The Composite Q(v)](#53-the-composite-qv)
   - 5.4 [Weight Provenance and AHP Shrinkage](#54-weight-provenance-and-ahp-shrinkage)
   - 5.5 [The Oracle: IR(v) and IM(v)](#55-the-oracle-irv-and-imv)
   - 5.6 [QoS-Profile Adaptation](#56-qos-profile-adaptation)
6. [Layer 3 — Quality-in-Use Weighting (ISO/IEC 25019:2023)](#6-layer-3--quality-in-use-weighting-isoiec-250192023)
   - 6.1 [The Three Characteristics](#61-the-three-characteristics)
   - 6.2 [The Projection Matrix](#62-the-projection-matrix)
   - 6.3 [Domain Context Vectors](#63-domain-context-vectors)
   - 6.4 [The Collapse Invariant](#64-the-collapse-invariant)
   - 6.5 [What the Layer Buys, Measured](#65-what-the-layer-buys-measured)
7. [The Edge Quality Model, Layer by Layer](#7-the-edge-quality-model-layer-by-layer)
   - 7.1 [Layers 0 and 1 — Edge Measure Elements](#71-layers-0-and-1--edge-measure-elements)
   - 7.2 [Layer 2 — Edge External Quality Attributes](#72-layer-2--edge-external-quality-attributes)
   - 7.3 [The Edge Oracle](#73-the-edge-oracle)
   - 7.4 [Layer 3 for Edges](#74-layer-3-for-edges)
8. [Coverage and Declared Gaps](#8-coverage-and-declared-gaps)
   - 8.1 [Unmodelled Sub-Characteristics](#81-unmodelled-sub-characteristics)
   - 8.2 [Instrument Scope Notes](#82-instrument-scope-notes)
9. [Constant Register](#9-constant-register)
10. [Where This Fits in the Pipeline](#10-where-this-fits-in-the-pipeline)
11. [References](#11-references)

---

## 1. Overview

Two ISO/IEC 25010:2023 product-quality characteristics are scored in this project: **Reliability** — hierarchical over its Fault Tolerance and Availability sub-characteristics — and **Maintainability**. Together they are the RM model, and they are the quantity every criticality score is built from, for a component and for a relationship.

This document is the **layer-by-layer reference** for those two models. It walks the four SQuaRE layers declared in [`saag/core/quality_model.py`](../saag/core/quality_model.py) in computation order — quality measure elements, internal quality measures, external quality attributes, quality-in-use weighting — and at each layer names the attributes that exist there, the measures that operationalize them, the coefficients that combine them, and the provenance of each. Where a coefficient is a design judgement rather than a measurement, this document says so at the point of use rather than in a footnote.

It does not re-argue the construct. The one-sentence framing everything below sits inside is stated once, and it belongs to [criticality.md §1](criticality.md#1-overview):

> **Criticality is *computed* from internal quality evidence, *validated* against simulated external quality, and *defined* on Quality-in-Use.**

What this document owns, and what it defers to:

| Question | Where it is answered |
|:---|:---|
| What attribute lives at which layer, with which coefficient and which provenance | **Here** |
| What each RM dimension means for a *stakeholder*, and D1–D4 | [criticality.md §3–§5](criticality.md#3-quality-grounding-square) |
| The formal definition of each individual structural metric ($RPR$, $MPCI$, $AP_c$, …) | [structural-analysis.md §9](structural-analysis.md#9-formal-definitions) |
| Where the raw graph and the QoS weights come from | [graph-model.md §4](graph-model.md#4-construction-phases) |
| How the simulation oracle is constructed and what it observes | [failure-simulation.md](failure-simulation.md), [validation.md](validation.md) |
| How the learned (GNN) predictor differs from the rule-based path | [prediction.md](prediction.md) |
| Whether the construct is *valid*, and what it does not establish | [criticality.md §7](criticality.md#7-validity-of-the-construct) |

Every formula printed below is the formula the code executes, cross-checked against [`saag/analysis/analyzer.py`](../saag/analysis/analyzer.py) and against the statement of the same formula in [structural-analysis.md §11.2](structural-analysis.md#112-rm-formulas). Where the two documents state a formula, they state the same one; this is a reference, not a second opinion.

---

## 2. The Four-Layer Model

The construct is expressible as four layers, each with a declared epistemic status. The layer stack is declared — not computed — in [`saag/core/quality_model.py`](../saag/core/quality_model.py) as the `Layer` enum and `LAYER_SPEC`; the computation lives in [`saag/analysis/`](../saag/analysis/).

| Layer | Name | Standard | Provenance | Oracle |
|:---|:---|:---|:---|:---|
| **0** | Quality measure elements | ISO/IEC 25021 | **MEASURED** | — |
| **1** | Internal quality measures | ISO/IEC 25023 (internal measures) | **DERIVED** | — |
| **2** | External quality attributes ($R$, $M$) | ISO/IEC 25010:2023 | **DERIVED** | $IR(v)$ / $IM(v)$ |
| **3** | Quality-in-use weighting | ISO/IEC 25019:2023 | **DECLARED** | — |

**Provenance is a three-valued vocabulary, and the distinction is load-bearing.** `Provenance` ([`quality_model.py`](../saag/core/quality_model.py)) declares it and states the rule in one line — *a DECLARED quantity must never be reported as though it were MEASURED*:

| Provenance | Meaning |
|:---|:---|
| **MEASURED** | Read directly off the artifact; no modelling step. |
| **DERIVED** | Computed from measured inputs by a stated, deterministic formula. |
| **DECLARED** | Asserted as a design judgement. Not fitted to, or validated against, any observation available in this project. |

Serialised output carrying a Layer 3 result carries this tag with it, so a downstream consumer can tell a derivation from a judgement without reading this document.

**Layer 2 is the only layer with an oracle**, and that fact organizes everything else. Layers 0 and 1 are definitional — there is no ground truth for "is this component's betweenness right", only "is it computed as defined". Layer 3 is a declared reweighting and, as [§6.4](#64-the-collapse-invariant) shows, *cannot* be a prediction stage. Validation therefore attaches at Layer 2 and nowhere else ([`saag/validation/dimensions.py`](../saag/validation/dimensions.py)'s `DIMENSION_SPECS`).

Two ordering facts worth fixing before the details:

- **Layers 0→2 run per component on every analysis.** They are the rule-based scoring path, `StructuralAnalyzer` then `QualityAnalyzer._compute_rm`.
- **Layer 3 is opt-in.** `QualityAnalyzer(domain_weights=...)` derives the composite weighting from a declared domain; the default (`domain_weights=None`) leaves the static weighting untouched, and is what every published number in this repo uses. When both are active, domain derivation runs **first** and QoS-profile adaptation ([§5.6](#56-qos-profile-adaptation)) second — a fixed, documented order.

---

## 3. Layer 0 — Quality Measure Elements (ISO/IEC 25021)

**Provenance: MEASURED.** A quality measure element is a quantity read off the artifact with no modelling step interposed. Everything at this layer is either counted from the topology, ingested from a static-analysis tool, or declared in the input topology as a QoS policy. Nothing here is normalised, weighted, or combined — that is Layer 1's job.

### 3.1 Topological Measure Elements

Counted directly from the typed multigraph $G$ produced by Step 1 ([graph-model.md](graph-model.md)). These are incidence and connectivity facts about an element's position, not judgements about it:

| Measure element | What is counted |
|:---|:---|
| `in_degree_raw`, `out_degree_raw` | Immediate dependents and dependencies of $v$ in $G_{\text{analysis}}(l)$ |
| `bridge_count` | Incident edges that are graph bridges (cut-edges) |
| `is_articulation_point`, `is_directed_ap` | Whether removing $v$ disconnects the undirected / directed projection |
| `topic_publisher_count`, `topic_subscriber_count` | Publisher and subscriber counts on a Topic |
| `path_count(e)` | Distinct topics (or shared nodes) establishing one derived `DEPENDS_ON` edge |
| `topic_frequency_hz` | Publication rate assigned to a Topic from its QoS band ([`TOPIC_FREQUENCY_HZ`](../saag/core/models.py)) |
| `size` | Topic payload size, in bytes |

Centralities ($RPR$, $BT$, $CC$, …) are *not* Layer 0: they are computed by an algorithm over the whole graph and are treated as Layer 1 internal measures ([§4.2](#42-the-nineteen-scoring-measures)).

### 3.2 Code Measure Elements

Ingested from SonarQube-style static analysis onto Application and Library vertices only, under the `cm_*` naming of [graph-model.md §4.1](graph-model.md#41-phase-1--entity-modeling). Five reach a score, through `CQP` and nothing else:

| Measure element | Alias read by scoring code | Reaches a score? |
|:---|:---|:---|
| `cm_total_loc` | `loc` | Yes — via `loc_norm` |
| `cm_avg_wmc` | `cyclomatic_complexity` | Yes — via `complexity_norm` |
| `cm_avg_lcom` | `lcom` | Yes — via `lcom_norm` |
| `cm_avg_fanin` | `coupling_afferent` | Yes — via `instability_code` |
| `cm_avg_fanout` | `coupling_efferent` | Yes — via `instability_code` |
| `cm_avg_cbo` | — | **No** — ingested, flattened, persisted, shown in the UI, read by no scoring code |
| `cm_avg_rfc` | — | **No** — same |

`sqale_debt_ratio`, `bugs`, `vulnerabilities` and `duplicated_lines_density` are likewise ingested and exported but move no score ([§4.5](#45-what-layer-1-leaves-out)).

### 3.3 Declared QoS Measure Elements

A QoS policy is not a property of the code — it is a statement about how the system must behave while it executes. In SQuaRE terms these are **declared external quality requirements**, not internal quality measures ([graph-model.md §4.3](graph-model.md#43-phase-3--intrinsic-weight-computation)); they enter the model at Layer 0 because they are read off the artifact without a modelling step, and they are what lets two topologically identical components score differently.

`QoSPolicy` ([`saag/core/models.py`](../saag/core/models.py)) maps each declared enum onto a score:

| Policy | Levels → score |
|:---|:---|
| **Reliability** | `BEST_EFFORT` 0.0 · `RELIABLE` 1.0 |
| **Durability** | `VOLATILE` 0.0 · `TRANSIENT_LOCAL` 0.5 · `TRANSIENT` 0.6 · `PERSISTENT` 1.0 |
| **Transport priority** | `LOW` 0.0 · `MEDIUM` 0.33 · `HIGH` 0.66 · `URGENT` / `CRITICAL` / `HIGHEST` 1.0 |

The three combine into the topic's QoS score, and that into its weight $w(t)$:

```
QoS(t)  = 0.30 × Rel(t)  +  0.40 × Dur(t)  +  0.30 × Pri(t)

w(t)    = max(0.01, 0.85 × QoS(t) + 0.15 × min(log2(1 + size_kb) / 50, 1.0))
```

**The 0.30 / 0.40 / 0.30 split is AHP-consistent, not arbitrary.** `QoSPolicy.W_RELIABILITY` / `W_DURABILITY` / `W_PRIORITY` are the runtime source of truth for $w(t)$, and `AHPMatrices.criteria_topic_qos` ([`saag/analysis/weight_calculator.py`](../saag/analysis/weight_calculator.py)) exists solely to demonstrate that a consistent Saaty pairwise-comparison matrix reproduces them — the matrix is *not* fed back into graph construction. Its geometric-mean priority vector is $(0.3002, 0.3996, 0.3002)$ at a consistency ratio of $\approx 0$, matching the shipped constants within $0.01$. Editing either side without the other breaks a CLAUDE.md invariant, enforced by `tests/test_ahp_shrinkage.py::test_topic_qos_matrix_reproduces_shipped_weights`. The substantive claim behind the ordering — Durability > Reliability = Priority — is that in DDS-family middleware durability governs *state survival*, while reliability and priority govern transient delivery quality.

$\beta = 0.85$ (`TOPIC_QOS_WEIGHT_BETA`) makes QoS semantics the primary signal and payload size a secondary amplifier; `MIN_TOPIC_WEIGHT` $= 0.01$ is a floor preventing zero-importance topics. Component and edge weights $w(v)$, $w(e)$ are propagated from topic weights in Phases 5 and 5b ([graph-model.md §4.5](graph-model.md#45-phase-5--aggregate-weight-propagation)).

---

## 4. Layer 1 — Internal Quality Measures (ISO/IEC 25023)

**Provenance: DERIVED.** Deterministic, definitional composites over Layer 0 elements: centralities, articulation scores, normalised code composites, and the enriched derived terms. This is the layer the existing four-layer table names but never enumerates; the enumeration is below.

### 4.1 Two Sources of Internal Evidence

Internal evidence arrives from two instruments at two granularities, and they contribute very unequally:

| Source | Granularity | Instrument | What it measures |
|:---|:---|:---|:---|
| **SSA — static system analysis** | System-level, all node types | The topology import itself | An element's *position*: centralities, directed articulation, bridges, coupling, QoS-derived weights |
| **SCA — static code analysis** | Code-level, Application and Library only | Ingested `cm_*` metrics ([§3.2](#32-code-measure-elements)) | A component's *internals*: size, complexity, cohesion, coupling |

In the rule-based path, SCA feeds exactly one composite ($CQP$), which feeds exactly one attribute (Maintainability), at coefficient $0.15$. Fault Tolerance and Availability are purely topological plus declared QoS — **zero code-derived inputs**. With $w_M = 0.20$, the effective share of code evidence in $Q(v)$ is $0.20 \times 0.15 = 3\%$. Edge dimensions are entirely code-free ([§7.2](#72-layer-2--edge-external-quality-attributes)). The learned path is structurally different — the same five code features sit on every Application/Library node vector and a shared encoder feeds both RM heads, so code evidence reaches $R$ as well as $M$ and propagates onto Broker/Node/Topic nodes by message passing. [criticality.md §3.0](criticality.md#30-three-quality-views-internal-external-and-quality-in-use) develops that asymmetry.

### 4.2 The Nineteen Scoring Measures

Of the ~50 fields on `StructuralMetrics` ([`saag/core/metrics.py`](../saag/core/metrics.py)), exactly **19 move $Q(v)$**. The set is declared once, against verified call sites, in [`saag/core/metric_registry.py`](../saag/core/metric_registry.py) as `SCORING_METRICS`, and is enforced by `tests/test_metric_registry.py`. Grouped by the Layer 2 attribute that consumes each:

| # | Measure | Symbol | Consumed by | Coefficient | Definition |
|:---|:---|:---|:---|:---|:---|
| 1 | `reverse_pagerank` | $RPR$ | **FT** (non-Topic) | 0.45 | [§9.1](structural-analysis.md#91-reverse-pagerank-rpr) |
| 2 | `in_degree_raw` | $DG_{in}$ | **FT** (non-Topic) | 0.30 | [§9.2](structural-analysis.md#92-in-degree-dg_in) |
| 3 | `out_degree_raw` | $DG_{out}$ | **FT** (via $CDPot$), **M** (via instability) | derived term | [§11.3](structural-analysis.md#113-derived-terms) |
| 4 | `mpci` | $MPCI$ | **FT** (via $CDPot_{enh}$ multiplier) | derived term | [§9.3](structural-analysis.md#93-multi-path-coupling-index-mpci) |
| 5 | `fan_out_criticality` | $FOC$ | **FT** (Topic branch, both terms) | 0.50 / 0.50 | [§9.4](structural-analysis.md#94-fan-out-criticality-foc) |
| 6 | `dependency_weight_in` | $w_{in}$ | **FT** (Topic branch only, as `publisher_norm`) | redundancy discount | [§9.13](structural-analysis.md#913-qos-weighted-in-degree-w_in) |
| 7 | `ap_c_directed` | $AP_{c}^{\text{dir}}$ | **A** (direct term + $QSPOF$) | 0.35 | [§9.8](structural-analysis.md#98-directed-ap-score-ap_c_directed) |
| 8 | `bridge_ratio` | $BR$ | **A** | 0.25 | [§9.9](structural-analysis.md#99-bridge-ratio-br) |
| 9 | `cdi` | $CDI$ | **A** | 0.10 | [§9.10](structural-analysis.md#910-connectivity-degradation-index-cdi) |
| 10 | `weight` | $w(v)$ | **A** (direct term + $QSPOF$) | 0.05 | [graph-model.md §4.5](graph-model.md#45-phase-5--aggregate-weight-propagation) |
| 11 | `betweenness` | $BT$ | **M** | 0.35 | [§9.5](structural-analysis.md#95-betweenness-centrality-bt) |
| 12 | `dependency_weight_out` | $w_{out}$ | **M** | 0.30 | [§9.6](structural-analysis.md#96-qos-weighted-out-degree-w_out) |
| 13 | `code_quality_penalty` | $CQP$ | **M** | 0.15 | [§11.2](structural-analysis.md#112-rm-formulas) |
| 14 | `path_complexity` | $PC$ | **M** (via $CouplingRisk_{enh}$) | $\delta = 0.10$ | [§9.14](structural-analysis.md#914-path-complexity-pc) |
| 15 | `clustering_coefficient` | $CC$ | **M**, as $(1 - CC)$ | 0.08 | [§9.7](structural-analysis.md#97-clustering-coefficient-cc) |
| 16 | `loc_norm` | — | **M** via $CQP$ | 0.10 | [§11.2](structural-analysis.md#112-rm-formulas) |
| 17 | `complexity_norm` | — | **M** via $CQP$ | 0.35 | [§11.2](structural-analysis.md#112-rm-formulas) |
| 18 | `instability_code` | — | **M** via $CQP$ | 0.30 | [§11.2](structural-analysis.md#112-rm-formulas) |
| 19 | `lcom_norm` | — | **M** via $CQP$ | 0.25 | [§11.2](structural-analysis.md#112-rm-formulas) |

**Metric orthogonality is a real discipline, not a slogan.** Each raw measure feeds **exactly one** sub-characteristic — no measure appears in two of FT, A, M. What that does *not* mean is that FT and A are independent attributes: both are constituents of the single Reliability characteristic, so a component scoring high on both has one characteristic degraded through two mechanisms, not two unrelated problems ([criticality.md §3.5](criticality.md#35-how-the-dimensions-bind-to-external-quality-dependability-and-quality-in-use)).

### 4.3 Derived Composites

Four Layer 1 composites are computed from the measures above and consumed directly by Layer 2. All are deterministic and definitional.

**Code Quality Penalty** — the only route by which code evidence reaches any score. Weights are `_CQP_WEIGHTS = (0.10, 0.35, 0.30, 0.25)` in [`saag/analysis/structural_analyzer.py`](../saag/analysis/structural_analyzer.py):

```
CQP(v) = 0.10 × loc_norm(v) + 0.35 × complexity_norm(v) + 0.30 × instability_code(v) + 0.25 × lcom_norm(v)
```

Each `*_norm` term is min-max normalised over **Application and Library as separate populations**, since their LOC and complexity scales differ; Broker, Node and Topic are skipped entirely and carry $CQP = 0$. The zero-variance branch of that normalisation has a known interaction, stated in [§8.2](#82-instrument-scope-notes).

**Enhanced Cascade Depth Potential** — a depth signal derived from existing quantities rather than a new traversal. High in-degree with low out-degree marks an absorber; $MPCI$ amplifies it when the incoming channels are multi-path:

```
CDPot_base(v) = ((RPR(v) + DG_in(v)) / 2) × (1 − min(out_degree_raw / max(in_degree_raw, ε), 1))
CDPot_enh(v)  = min(CDPot_base(v) × (1 + MPCI(v)), 1.0)
```

**Coupling Risk with path complexity** — an afferent/efferent imbalance signal, maximal at instability $0.5$ (equal fan-in and fan-out) and enriched by how many distinct channels the outgoing dependencies span. $\delta = $ `COUPLING_PATH_DELTA` $= 0.10$:

```
instability(v)       = out_degree_raw / (in_degree_raw + out_degree_raw + ε)
CouplingRisk(v)      = 1 − |2 × instability(v) − 1|
CouplingRisk_enh(v)  = min(1.0, CouplingRisk(v) × (1 + 0.10 × path_complexity(v)))
```

`_compute_rm` writes `CouplingRisk_enh` back onto the metrics object, where the `UNSTABLE_INTERFACE` detector reads it ([antipatterns.md](antipatterns.md)); it is a scoring input *and* a persisted diagnostic.

**QoS-weighted SPOF severity** — the one place where a declared external requirement amplifies a structural fact at the node level:

```
QSPOF(v) = AP_c_directed(v) × w(v)
```

Full derivations for all three enriched terms are in [structural-analysis.md §11.3](structural-analysis.md#113-derived-terms).

### 4.4 Normalisation

Most Layer 1 measures reach Layer 2 through `_normalize_robust`, an **average-rank** normalisation: values are ranked, ties take the mean rank, and the result is $\text{avg\_rank}/(n-1)$, so every normalised population is bounded in $[0,1]$ and outliers cannot dominate. Eight metrics are additionally winsorized at the $1-\text{limit}$ percentile (default $5\%$) before ranking.

Rank-normalised: `reverse_pagerank`, `betweenness`, `in_degree_raw`, `out_degree_raw`, `weight`, `dependency_weight_in`, `dependency_weight_out`.

**Read raw, not rank-normalised:** `ap_c_directed`, `cdi`, `mpci`, `fan_out_criticality`, `clustering_coefficient`, `code_quality_penalty`. These are already bounded in $[0,1]$ by their own definitions, and re-ranking them would destroy the absolute reading that makes, for instance, $AP_c = 0$ mean "not an articulation point at all" rather than "lowest in this population". [structural-analysis.md §8](structural-analysis.md#8-normalization) states the contract and its caveats in full.

### 4.5 What Layer 1 Leaves Out

`MetricRole` ([`saag/core/metric_registry.py`](../saag/core/metric_registry.py)) partitions every `StructuralMetrics` field by what actually reads it. The roles are not mutually exclusive — most SCORING metrics are also GNN features — and the partition exists because, before it, four different counts of "how many metrics does this model use" were published across four documents, none enumerating the same set.

| Role | Read by | Moves $Q(v)$? |
|:---|:---|:---|
| **SCORING** | `QualityAnalyzer._compute_rm` | **Yes** — the 19 of [§4.2](#42-the-nineteen-scoring-measures) |
| **DETECTION** | `AntiPatternDetector` only | No |
| **GNN_FEATURE** | `saag.prediction.data_preparation` only (metrics that are *only* a GNN input) | No, in the rule-based path |
| **DESCRIPTIVE** | Exported, serialised, and at most rendered in the dashboard | No |

A DESCRIPTIVE metric is not dead weight to someone browsing a graph — but it moves no score, and should never be described as though it does. `blast_radius`, `cascade_depth`, `pubsub_betweenness`, `broker_exposure`, `sqale_debt_ratio`, `bugs`, `vulnerabilities` and `duplicated_lines_density` are all in this class.

---

## 5. Layer 2 — External Quality Attributes (ISO/IEC 25010:2023)

**Provenance: DERIVED. The only layer with an oracle.** Here the internal measures become estimates of two named product-quality characteristics. The names come from ISO/IEC 25010:2023; the attributes are *externally* observable in principle (Maintainability excepted, [§5.2](#52-maintainability)), which is what makes them validatable — but they are computed here from internal evidence alone, which is what makes them available pre-deployment.

All node-level scoring happens in `QualityAnalyzer._compute_rm` ([`saag/analysis/analyzer.py`](../saag/analysis/analyzer.py)).

### 5.1 Reliability

ISO/IEC 25010:2023 gives Reliability four sub-characteristics. This framework's scoring hierarchy mirrors that composition directly rather than mapping onto it:

| Sub-characteristic | Status here | Why |
|:---|:---|:---|
| **Faultlessness** | **Excluded by definitional choice** | The likelihood-bearing sub-characteristic — how *often* the system fails. [D3](criticality.md#23-consequence-not-risk) ("criticality is a consequence, not a risk") *is* this exclusion, restated in the standard's own vocabulary. |
| **Fault tolerance** | **Modelled as $FT(v)$** | Operating despite faults ⇒ how far an error propagates through dependents before containment. |
| **Availability** | **Modelled as $A(v)$** | Readiness for correct service ⇒ structural partition, the state where the task stops outright. |
| **Recoverability** | **Absent — a declared data gap** | Needs MTTR, restart semantics, or replication state; no such field exists in the schema. Every structural SPOF scores alike regardless of how fast it would actually be restored. |

#### Fault Tolerance — $FT(v)$

Two branches, because a Topic's failure semantics are not a component's. For Application, Broker, Node and Library:

```
FT(v) = 0.45 × RPR(v) + 0.30 × DG_in(v) + 0.25 × CDPot_enh(v)
```

Reverse PageRank is the leading term for a structural reason, not a stylistic one: `DEPENDS_ON` edges point *dependent → dependency*, so failure propagates **against** edge direction, and PageRank on $G^{\mathsf T}$ is the correct cascade-reach estimator.

For Topic nodes:

```
FT_topic(v)    = 0.50 × FOC(v) + 0.50 × CDPot_topic(v)
CDPot_topic(v) = FOC(v) × (1 − min(publisher_norm(v), 1))
```

`publisher_norm` is the rank-normalised `dependency_weight_in` — the QoS-weighted sum of a topic's incoming publisher edges. It acts as a **redundancy discount**: a topic with many publishers loses less when one of them fails than a sole-publisher topic does. The field's naming history is worth knowing, since it reads oddly otherwise — it was the Tier-1 input to the retired Vulnerability dimension and was repurposed rather than retired with it ([structural-analysis.md §9.13](structural-analysis.md#913-qos-weighted-in-degree-w_in)).

The Topic branch's $0.50/0.50$ split is a hard-coded literal, not a `QualityWeights` field; [§8.2](#82-instrument-scope-notes) states what follows from that.

#### Availability — $A(v)$

A five-term additive score over structural-partition evidence, with one QoS-amplified term:

```
A(v) = 0.35 × AP_c_directed(v) + 0.25 × QSPOF(v) + 0.25 × BR(v) + 0.10 × CDI(v) + 0.05 × w(v)
```

$AP_c^{\text{dir}}$ is continuous, not a binary articulation flag: it is $1 - |\text{largest component after removing } v| / (n-1)$, taken as the worse of the outbound and transposed views, so "how badly does removing this fragment the graph" is graded rather than thresholded. $CDI$ catches the non-articulation case where removal does not partition the graph but does lengthen every path through it.

#### The hierarchical blend

```
R(v) = r_alpha × FT(v) + (1 − r_alpha) × A(v)          r_alpha = 0.36
```

$FT$ and $A$ are reported individually as sub-characteristic diagnostics as well as blended, so a component carries a profile rather than one number. `CriticalityProfile` reads the $(\textit{ft\_crit}, \textit{a\_crit}, \textit{m\_crit})$ triple into named patterns — *SPOF* (high A alone), *Fault-Tolerance Hub* (high FT alone), *Bottleneck* (high M alone), *Total Hub*, *Fragile Hub*, *Fragile Bottleneck*, defaulting to *Composite Risk*. The profile, not the composite, is what identifies which remedy applies.

**In dependability terms the two are stages of one causal chain**, which is why they blend rather than stand as peers: the *fault* is the injected failure, the *error* is what propagates along `DEPENDS_ON` to dependents (what $FT$ measures), and the *failure* is the resulting loss of service (what $A$ measures when the loss is total).

### 5.2 Maintainability

ISO/IEC 25010:2023 gives Maintainability five sub-characteristics. Two are addressed:

| Sub-characteristic | Status here |
|:---|:---|
| **Modularity** | **Modelled** — via $BT$, $w_{out}$, $CouplingRisk_{enh}$, $(1-CC)$ |
| **Modifiability** | **Modelled** — same terms, plus $CQP$ |
| **Reusability** | **Unmodelled** |
| **Analysability** | **Unmodelled** |
| **Testability** | **Unmodelled** |

```
M(v) = 0.35 × BT(v)
     + 0.30 × w_out(v)
     + 0.15 × CQP(v)
     + 0.12 × CouplingRisk_enh(v)
     + 0.08 × (1 − CC(v))
```

$BT$ leads because a structural bottleneck is the position from which change is most expensive: everything routes through it. $w_{out}$ is QoS-weighted efferent coupling — a component that depends on many strongly-guaranteed flows cannot be changed without renegotiating each of them. $(1 - CC)$ enters inverted: a low clustering coefficient means a component's neighbours do not talk to each other, so it is the sole integration point between them.

> **Maintainability is the one attribute in this model that is not externally observable at all.** No amount of watching a system execute reveals what a change to it would cost. $M$ is therefore an internal-quality estimate of an internal-quality attribute — the only RM attribute where the "internal evidence, external estimand" structure of [§1](#1-overview) does not hold. This is also why the simulation oracles cannot measure it behaviourally, and why $IM(v)$ ([§5.5](#55-the-oracle-irv-and-imv)) is a change-propagation traversal rather than a fault injection.

### 5.3 The Composite Q(v)

```
Q(v) = w_R × R(v) + w_M × M(v)          w_R = 0.80, w_M = 0.20
```

Three qualifications travel with these numbers, and all three matter when citing them.

**They are DECLARED constants, not AHP output.** With only two composite terms, a $2\times2$ Saaty matrix is consistent by construction ($CR = 0$ for $n \le 2$) and would contribute nothing. `AHPMatrices` stores no composite matrix at all, and `use_ahp=True` does not change $w_R$, $w_M$ or $r_{\alpha}$ — it changes only the intra-dimension FT/M/A/Impact vectors. Do not describe $(w_R, w_M)$ as "the AHP weights" under any configuration.

**They are a re-parameterisation, not an independent invention.** $(w_R, w_M, r_\alpha) = (0.80, 0.20, 0.36)$ are algebraically derived from the retired 4-D AHP composite $(A{=}0.43,\ R{=}0.24,\ M{=}0.17,\ V{=}0.16)$ by dropping $V$ and renormalising:

```
r_alpha = 0.24 / (0.24 + 0.43) = 0.3582 → 0.36
w_R     = (0.24 + 0.43) / 0.84 = 0.7976 → 0.80
w_M     = 0.17 / 0.84          = 0.2024 → 0.20
```

At exact values, $w_R \cdot r_\alpha = 0.24/0.84$, $w_R(1 - r_\alpha) = 0.43/0.84$ and $w_M = 0.17/0.84$ — so the hierarchical composite recovers the retired composite's $R$, $A$ and $M$ shares exactly. Rounding to 2 s.f. is not free, but the drift is bounded and pinned at `abs=0.003` by `tests/test_quality_model.py::TestCompositeReparameterisation`, a CLAUDE.md invariant.

**No accuracy claim attaches to the *composite* weighting, under any configuration.** $(w_R, w_M)$ is $\lambda$-invariant by construction — every row of the shrinkage sweep used the identical $0.80/0.20$ — so no sensitivity result bears on it either way. What the sweep does characterise is the **intra-dimension** FT/M/A/Impact vectors, and there the current measured result mildly favours the calibrated judgement: the curve is monotone increasing in $\lambda$, the raw AHP judgement at $\lambda=1$ beats uniform intra-dimension weights by $+0.032\ \rho$, and the full sweep spans only $0.045\ \rho$ with every value near-zero-to-slightly-negative ([structural-analysis.md §11.6](structural-analysis.md#116-weight-shrinkage-strategy)). Read that as a small, consistent direction rather than a ranking argument — the decomposition is retained as an **attribution** mechanism, since Reliability and Maintainability have distinct remedies and distinct owners, not as a ranking-improvement device. An equal-weight baseline ($w_R = w_M = 0.5$, $\alpha = 0.5$) is available via `--equal-weights`.

Scores are then mapped onto five tiers by adaptive box-plot thresholding relative to the system's own distribution, independently per dimension and for the composite, falling back to fixed percentiles below `MIN_BOXPLOT_SAMPLE` = 12 components ([structural-analysis.md §11.7](structural-analysis.md#117-criticality-classification), [criticality.md §4.6](criticality.md#46-criticality-classification)).

### 5.4 Weight Provenance and AHP Shrinkage

Coefficients within a dimension come from Saaty pairwise-comparison matrices in `AHPMatrices`, resolved by the geometric-mean (approximate eigenvector) method and then **shrunk toward a uniform prior**:

$$
w_{\text{final}} = \lambda \cdot w_{\text{AHP}} + (1 - \lambda)\cdot \tfrac{1}{n}, \qquad \lambda = 0.70
$$

Shrinkage reconciles an elicited judgement with an uninformative prior, reducing the leverage any single pairwise comparison has on the result. The distinction between pre- and post-shrinkage vectors is easy to lose, so both are stated:

| Dimension | Matrix | Pre-shrinkage (the judgement) | Post-shrinkage at $\lambda = 0.70$ |
|:---|:---|:---|:---|
| **FT** | $3\times3$ — RPR, DG_in, CDPot | 0.45, 0.30, 0.25 | 0.422, 0.323, 0.255 |
| **M** | $5\times5$ — BT, w_out, CQP, CR, (1−CC) | 0.35, 0.30, 0.15, 0.12, 0.08 | 0.305, 0.270, 0.165, 0.144, 0.116 |
| **A** | $5\times5$ — AP_c_dir, QSPOF, BR, CDI, w | 0.35, 0.25, 0.25, 0.10, 0.05 | 0.305, 0.235, 0.235, 0.130, 0.095 |
| **Impact** | $4\times4$ — RL, FR, TL, FD | 0.393, 0.25, 0.25, 0.107 | 0.347, 0.254, 0.254, 0.145 |

**The pre-shrinkage column is what the formulas in [§5.1](#51-reliability)–[§5.2](#52-maintainability) print, and that is deliberate: those are the `QualityWeights` dataclass defaults, which is what runs when `use_ahp=False` — the default configuration, and the one behind every published number.** The post-shrinkage column is what `AHPProcessor.compute_weights()` returns when AHP is switched on. Every matrix is essentially perfectly consistent — $|CR| < 0.003$ for all five (FT $+0.0028$, M $+0.0005$, A $-0.0008$, Topic QoS $-0.0014$, Impact $+0.0011$), far below the conventional $0.10$ acceptance threshold. That is a weak result rather than a strong one, and worth reading as such: these matrices were constructed *backwards* from intended weight vectors rather than elicited from an expert, so near-zero $CR$ confirms the construction was arithmetically sound, not that the judgement was independently corroborated. $CR$ is defined as $0$ for $n \le 2$ by construction, which is why no composite matrix exists ([§5.3](#53-the-composite-qv)).

$r_\alpha$, $w_R$ and $w_M$ pass through `compute_weights()` untouched — they are DECLARED at every configuration ([§5.3](#53-the-composite-qv)).

### 5.5 The Oracle: IR(v) and IM(v)

Layer 2 is the only layer that can be wrong in a way an experiment could detect, because it is the only layer with a ground truth to be wrong against. `DIMENSION_SPECS` ([`saag/validation/dimensions.py`](../saag/validation/dimensions.py)) binds each attribute to its simulated impact:

| Attribute | Oracle | Composition |
|:---|:---|:---|
| **Reliability** | $IR(v)$ | $r_\alpha \cdot IFT(v) + (1 - r_\alpha)\cdot IA(v)$ — mirrors the scoring-side hierarchy exactly |
| ↳ Fault tolerance | $IFT(v)$ | $0.45 \cdot \textit{cascade\_reach} + 0.35 \cdot \textit{weighted\_cascade\_impact} + 0.20 \cdot \textit{normalized\_depth}$ |
| ↳ Availability | $IA(v)$ | $0.50 \cdot \textit{weighted\_reachability} + 0.35 \cdot \textit{weighted\_fragmentation} + 0.15 \cdot \textit{path\_breaking\_throughput}$ |
| **Maintainability** | $IM(v)$ | $0.45 \cdot \textit{change\_reach} + 0.35 \cdot \textit{weighted\_change\_impact} + 0.20 \cdot \textit{normalized\_change\_depth}$ |

$IA(v)$ is validated and reported as a sub-characteristic diagnostic but **excluded from the composite gates** ($I^*$, predictive gain, orthogonality) — including it there would double-count Reliability, since $IA$ already feeds $IR$'s blend.

**The oracle is an external-quality instrument, not a quality-in-use one.** What the simulator observes — delivery rate, fragmentation, latency, contract conformance — are external product-quality measures in the ISO/IEC 25023 sense, made on a *model* of the executing system rather than on a deployment. That naming is what lets [criticality.md §7.1](criticality.md#71-the-validation-chain-has-three-links) state precisely which link of the validation chain has been measured and which two have not. $IM(v)$ is a special case: since maintainability is not externally observable ([§5.2](#52-maintainability)), $IM$ is a change-propagation traversal over $G^{\mathsf T}$, not a fault injection.

### 5.6 QoS-Profile Adaptation

`adapt_qos_weights` is **on by default**, so the *effective* composite is per-system rather than fixed at $(0.80, 0.20)$. `_derive_qos_weights` reads the topic set's QoS profile and shifts weight between the two attributes:

```
rel_signal = (persistent_frac + reliable_frac + critical_frac) / 3

rel_signal ≥ 0.6  →  delta = min(0.15, (rel_signal − 0.5) × 0.30),  shifted to w_R
rel_signal ≤ 0.4  →  delta = min(0.15, (0.5 − rel_signal) × 0.30),  shifted to w_M
otherwise         →  defaults unchanged
```

Both weights are then floored at $0.05$ and renormalised to sum to $1.0$. The reasoning is that a system whose topics are overwhelmingly `PERSISTENT`/`RELIABLE`/`CRITICAL` is mission-critical and should weight Reliability harder, while a high-churn `VOLATILE`/`BEST_EFFORT` system should weight the cost of change harder. $r_\alpha$ is untouched — the FT-vs-A split is a separate, stakeholder-declared lever ([§6](#6-layer-3--quality-in-use-weighting-isoiec-250192023)).

When Layer 3 domain derivation is also active, the order is fixed: **domain derivation first, QoS adaptation second**, with the analyzer's own weights restored afterwards so no caller-owned object is mutated.

---

## 6. Layer 3 — Quality-in-Use Weighting (ISO/IEC 25019:2023)

**Provenance: DECLARED. No oracle, and — provably — no capacity to be one.** Nothing in this project measures quality-in-use: there is no user study, no expert elicitation, and no incident record anywhere in the repository. This layer therefore does not observe quality-in-use; it *re-expresses* an RM profile in stakeholder vocabulary, and derives a composite weighting from declared stakeholder priorities.

### 6.1 The Three Characteristics

ISO/IEC 25019:2023 structures quality-in-use into three primary characteristics:

| Characteristic | Sub-characteristics | Meaning in stakeholder terms |
|:---|:---|:---|
| **Beneficialness** | Usability (effectiveness, efficiency, satisfaction, trust, comfort, transparency), Accessibility, Suitability | The system delivers positive utility and lets stakeholders reach operational goals accurately and efficiently |
| **Freedom from risk** | Economic, health, human-life, environmental & societal risk | The system limits potential harm during operational failure |
| **Acceptability** | Experience, Trustworthiness, Compliance | Stakeholders respond favourably and keep confidence in the system's operation and regulatory standing |

[criticality.md §3.1–§3.2](criticality.md#31-what-quality-in-use-is) develops the stakeholder taxonomy these characteristics are defined against, and [§4.5](criticality.md#45-mapping-rm-to-external-quality-and-quality-in-use) reads the RM↔harm correspondence in the reverse direction.

### 6.2 The Projection Matrix

`QIU_PROJECTION` maps the RM vector $\mathbf{s}_{\text{RM}}(v) = [R(v), M(v)]^{\mathsf T}$ onto the three-characteristic harm vector:

$$
\mathbf{h}_{\text{QiU}}(v) \;=\; \mathbf{M}_{\text{RM} \to \text{QiU}} \cdot \mathbf{s}_{\text{RM}}(v), \qquad
\mathbf{M}_{\text{RM} \to \text{QiU}} \;=\;
\begin{bmatrix}
0.75 & 0.25 \\
0.80 & 0.20 \\
0.60 & 0.40
\end{bmatrix}
$$

Rows are Beneficialness, Freedom from risk, Acceptability; columns are $R$, $M$. `qiu_harm(scores)` computes this.

| Row | Coefficients | Derivation |
|:---|:---|:---|
| **Beneficialness** | $(0.75, 0.25)$ | **Mechanical.** An unchanged fold of the retired 4-D model's $A$ column into $R$; $V$'s coefficient in this row was already $0.00$, so $(0.35 + 0.40,\ 0.25)$ sums to $1.0$ with no new judgement. |
| **Freedom from risk** | $(0.80, 0.20)$ | **Re-declared.** $M$'s $0.20$ is maintainability's *MTTR channel*: slow repair prolongs hazard exposure. |
| **Acceptability** | $(0.60, 0.40)$ | **Re-declared.** $M$'s $0.40$ is maintainability's *evolvability channel* into perceived trust. |

**Why rows 2 and 3 were re-declared rather than folded.** The mechanical fold alone would have collapsed both to $(1.00, 0.00)$, making the matrix **rank-1 in $\vec\omega$**: the three-dimensional domain-priority vector would reduce to a single degree of freedom, and any two domains sharing a Beneficialness weight would tie exactly regardless of their risk posture — healthcare and enterprise, both $\omega_{\text{Ben}} = 0.40$, are the concrete case. Keeping the matrix rank-2 is what makes domain context expressible at all, and `tests/test_quality_model.py::TestEffectiveRmWeights::test_no_two_distinct_omegas_tie` pins it.

> **What the six coefficients are and are not.** They are a stated design judgement, not an estimated or validated quantity: nothing in this project measures quality-in-use, so nothing here could have fitted them. Their value is that they make the many-to-many RM↔harm correspondence arithmetic rather than rhetorical. Cite them as an operationalization proposal.

### 6.3 Domain Context Vectors

`DOMAIN_PRIORITIES` assigns each deployment domain an $\vec\omega = [\omega_{\text{Ben}}, \omega_{\text{Risk}}, \omega_{\text{Acc}}]$ summing to $1.0$. [criticality.md §3.3](criticality.md#33-context-of-use-and-domain-context-vector) states these as *ordinal* constraints; the numbers below are the smallest assignment satisfying those orderings. **The ordering, not the magnitude, is what the source table asserts.** Keys are the values that actually appear in scenario `metadata.domain`.

| Domain keys | $\vec\omega$ (Ben, Risk, Acc) | Dominant characteristic | Effective $(w_R, w_M)$ |
|:---|:---|:---|:---|
| `autoware_ros2`, `av` | $(0.20,\ 0.65,\ 0.15)$ | Freedom from risk (health & life) | $(0.760,\ 0.240)$ |
| `finance` | $(0.35,\ 0.50,\ 0.15)$ | Freedom from risk (economic) | $(0.753,\ 0.248)$ |
| `healthcare` | $(0.40,\ 0.45,\ 0.15)$ | Risk ≈ Beneficialness | $(0.750,\ 0.250)$ |
| `air_traffic_management` | $(0.50,\ 0.30,\ 0.20)$ | Beneficialness (effectiveness) | $(0.735,\ 0.265)$ |
| `iot` | $(0.50,\ 0.20,\ 0.30)$ | Beneficialness (efficiency) | $(0.715,\ 0.285)$ |
| `enterprise`, `microservices`, `cloud_microservices`, `trainticket_microservices` | $(0.40,\ 0.20,\ 0.40)$ | Beneficialness ≈ Acceptability | $(0.700,\ 0.300)$ |

`hub-and-spoke` is **deliberately absent**: it names a topology class, not a deployment domain, so it has no stakeholder priority and takes the fallback. `derive_rm_weights(domain)` looks up case- and whitespace-insensitively and, on an unknown or absent domain, returns the static defaults unchanged with `derived=False` — it never invents a vector. Only the two composite $q_*$ weights change; every intra-dimension weight and $r_\alpha$ are left alone.

### 6.4 The Collapse Invariant

**Every row of $\mathbf{M}_{\text{RM}\to\text{QiU}}$ sums to $1.0$.** That is a property of the matrix's *shape*, not of its coefficients, and it settles what this layer can be. For any domain weights $\vec\omega$:

$$
Q_{\text{QiU}} \;=\; \vec{\omega} \cdot (\mathbf{M}\,\mathbf{s}) \;=\; (\mathbf{M}^{\mathsf T}\vec{\omega}) \cdot \mathbf{s}, \qquad \textstyle\sum(\mathbf{M}^{\mathsf T}\vec{\omega}) \;=\; \sum\vec{\omega} \;=\; 1
$$

> **A quality-in-use scalarisation is algebraically identical to scoring the same RM vector under a different composite weighting.** There is no such thing as a "quality-in-use score" that ranks components differently from *some* RM weighting. This project's code does not compute one and report it as an independent quantity, and must not start: `effective_rm_weights(omega)` *is* $\mathbf{M}^{\mathsf T}\vec\omega$, returned as the composite pair $(w_R, w_M)$. This is a CLAUDE.md invariant, enforced by `tests/test_quality_model.py::TestQiuCollapseEquivalence`.

**Stated positively, Layer 3 is a principled generator of context-dependent composite weights.** It offers an alternative to the DECLARED static $(0.80, 0.20)$ that is traceable to a named domain and a named standard rather than to an elicitation — and, unlike an AHP vector, directly testable, since every scenario in the corpus already carries `metadata.domain`.

### 6.5 What the Layer Buys, Measured

Measured over 7 synthetic scenarios plus the 3 real-world RQ4 graphs, Spearman $\rho$ against $I^*(v)$ ([`reproduce/domain_weight_comparison.py`](../reproduce/domain_weight_comparison.py), `results/domain_weight_comparison.json`):

| Weighting | Mean $\rho$ | vs. static | vs. equal |
|:---|:---|:---|:---|
| Static default ($w_R = 0.80$) | 0.1223 | — | — |
| Equal ($w_R = w_M = 0.5$) | 0.2585 | — | — |
| **Domain-derived** ($w_R \in [0.70, 0.76]$) | **0.1488** | **+0.0265** | **−0.1097** |

**The headline is the mean Kendall $\tau$ of 0.9677** between domain-derived and static component rankings: across all six declared $\vec\omega$ values, domain-derived $w_R$ sits within $0.10$ of the static default, so the two rank components almost identically before $\rho$ enters the picture. Domain-derived weighting edges past the static default and is clearly beaten by equal weights. The full 1-D sweep of $\rho(Q(w_R), I^*)$ over the entire $[0,1]$ range is monotonically decreasing — $0.339$ at $w_R = 0$ down to $-0.031$ at $w_R = 1$ — so no weighting confined to the domain-derived sub-range could move $\rho$ much either way.

Read plainly: **the domain derivation's value is attributional — explaining criticality in stakeholder terms — not a ranking-improvement device.** That has always been the correct way to scope it, and these numbers make the point more starkly rather than less.

---

## 7. The Edge Quality Model, Layer by Layer

A relationship is scored on the **same** four layers and the **same** two attributes as a component, scoped to the link rather than the endpoint. Using one model for both is deliberate: it makes nodes and edges comparable in a single ranking, and it lets a remediation owner read node and edge findings in the same vocabulary — an SRE reads $A$ on both, an architect reads $M$ on both. The stakeholder-facing definition (D2) is in [criticality.md §5.1](criticality.md#51-definition); this section states the measurement stack.

The discriminating clause is that **both endpoints remain operational**. A node failure is a total outage of a capability; an edge failure is a *partial* outage — the component is up, its dashboards are green, its other consumers are fine, but one data flow has stopped.

### 7.1 Layers 0 and 1 — Edge Measure Elements

Carried by `EdgeMetrics` in [`saag/core/metrics.py`](../saag/core/metrics.py) and computed in [`saag/analysis/structural_analyzer.py`](../saag/analysis/structural_analyzer.py):

| Measure | Layer | What it is |
|:---|:---|:---|
| `is_bridge` | 0 | Whether the edge is a cut-edge (`nx.bridges()` over the undirected projection). Removing it disconnects a subgraph. |
| `betweenness` | 1 | Edge betweenness over the **inverted-weight** graph, each edge's length taken as $1/w(e)$ |
| `weight` — $w(e)$ | 0 | The QoS-derived guarantee crossing the link |
| `path_count` | 0 | How many distinct topics (or shared nodes) establish this one derived edge |

**Why betweenness is computed over inverted weights.** Taking edge length as $1/w(e)$ makes strongly-guaranteed dependencies *short*, so they attract shortest paths rather than repelling them — the links the system leans on hardest end up with the highest betweenness, which is the intended reading.

**Edge weight is the worst case, not an average.** For a structural edge $w(e)$ is inherited from the topic it attaches to; for a derived `DEPENDS_ON` edge it is $\max w(t)$ over the mediating topics. Taking the maximum is the conservative reading D2 requires: if *any* strongly-guaranteed flow crosses this link, losing the link breaks that guarantee no matter how many weak flows it also carries. `path_count` is deliberately kept out of $w(e)$ so that $w(e) \in [0,1]$ is preserved, and acts as a separate coupling-intensity signal.

Edges use $w(e)$ **directly, un-normalised** — the Step 1 contract already bounds it — unlike the node path, which rank-normalises $w(v)$.

### 7.2 Layer 2 — Edge External Quality Attributes

Computed in `_score_and_classify_edges` ([`saag/analysis/analyzer.py`](../saag/analysis/analyzer.py#L451-L529)), blending edge-intrinsic signals with endpoint context. Coefficients are the `e_*` fields of `QualityWeights`: `e_betweenness=0.35`, `e_bridge=0.30`, `e_endpoint=0.20`, `e_qos_weight=0.15`.

```
FT(u,v) = 0.35 × betweenness + 0.30 × w(e) + 0.20 × max(source.FT, target.FT)

A(u,v)  = 0.30 × is_bridge + 0.20 × min(source.A, target.A)

M(u,v)  = 0.35 × betweenness + 0.30 × is_bridge + 0.15 × w(e)

R(u,v)      = 0.36 × FT(u,v) + 0.64 × A(u,v)
overall(u,v) = 0.80 × R(u,v) + 0.20 × M(u,v)
```

The hierarchy mirrors the node model exactly — the *identical* `r_alpha`, `q_reliability` and `q_maintainability` constants, read from the same `QualityWeights` instance, not re-declared.

| Attribute | Question answered for a link | Endpoint term | Weight-bearing term |
|:---|:---|:---|:---|
| **FT** | How much does this link contribute to fault propagation? | $\max(\textit{src}.FT, \textit{tgt}.FT)$ — an edge is no more fault-tolerant than its riskiest end | $w(e)$ at 0.30 — a link conducts faults in proportion to what it promises to deliver |
| **A** | Does losing this link partition the graph? | $\min(\textit{src}.A, \textit{tgt}.A)$ — no more available than its weakest end | **None** |
| **M** | How much does this link add to change cost? | **None** | $w(e)$ at 0.15 — a high-guarantee contract is expensive to renegotiate, both sides must move together |

**Availability carries no $w(e)$ term, and that is a quality-model decision rather than an oversight.** Availability asks only *is this link replaceable?*, and redundancy is a topological fact: a bridge is a bridge whether it carries safety telemetry or debug logs. The two inputs sit on different SQuaRE views — replaceability is a structural property of the system (internal quality evidence), while $w(e)$ is a *declared external quality requirement* of one flow. Multiplying them would let a declared requirement change a topological fact. QoS amplification still reaches edge $A$, indirectly, through the endpoints' own $QSPOF$.

Three properties a reader needs before comparing edge numbers:

- **Edge dimensions are entirely code-free.** Edge $M$ carries no endpoint-$M$ term at all, so $CQP$ has no path to any edge score. A link's maintainability is blind to how coupled its endpoints already are — and `path_count` reaches an edge only indirectly, through the endpoints' $FT$, via the `max(src.FT, tgt.FT)` term.
- **The dimensions draw on unequal coefficient mass** — $FT$ sums to $0.85$ of a possible $1.0$, $M$ to $0.80$, $A$ to $0.50$. Raw edge scores are comparable *within* a dimension but not *across* them. Since classification is box-plot relative within the edge set, per-dimension rankings and tiers are unaffected; only the raw magnitudes are. **Read an edge's dimension tiers, not its absolute dimension values.**
- **Two node-level metrics are easy to mistake for edge scores.** Bridge Ratio $BR(v)$ is the fraction of a *node's* connections that are bridges, and $MPCI(v)$ counts redundant shared channels feeding a *node*. Both are edge-derived and neither is a per-edge score.

Vulnerability/Security was scored as a fourth edge dimension in an earlier revision and has been **retired outright** — not folded into another edge dimension — for the same reason as the component-level $V(v)$: it presumed an adversary rather than a fault, so no fault-removal oracle in this project could ever validate it.

### 7.3 The Edge Oracle

`FailureSimulator.simulate_edge_removal` ([`saag/simulation/failure_simulator.py`](../saag/simulation/failure_simulator.py)) severs one relationship with **both endpoints alive and no cascade run**, then recomputes the same reachability / fragmentation / throughput / flow quantities that back node impact, on the same $0.35/0.25/0.25/0.15$ composite weighting so node and edge labels stay comparable. No cascade is run because the question an edge label answers is *what does this link carry*, not *what else breaks afterwards* — cascade effects belong to the node labels.

Two properties of this oracle are easy to misread:

- **Every quantity is differenced against a null observation.** `_calculate_impact` is not zero on a pristine graph: topics that already lack a publisher or a subscriber count as lost throughput regardless of what failed. On `av_system` that floor is composite $0.0061$. Without differencing, edges that cost nothing — `RUNS_ON`, which this cascade model does not route traffic over — would all report the floor as if it were signal.
- **`evaluated=False` is not a measured zero.** Only a candidate set is swept (bridges plus the top-50 by edge betweenness); everything else is unmeasured, and `EdgeCriticality` records that distinction explicitly rather than defaulting it to $0$.

Per-edge impact is banded absolutely before any per-scenario reclassification: $\ge 0.50$ critical, $\ge 0.25$ high, $\ge 0.10$ medium, $> 10^{-6}$ low, otherwise minimal.

A third, independent edge score exists in the learned path — `GNNEdgeCriticalityScore` ([prediction.md §5](prediction.md#5-edge-criticality)) — predicted rather than computed or measured. Three numbers, three provenances: do not conflate them.

### 7.4 Layer 3 for Edges

Layer 3 applies to edges **unchanged**, because `overall(u,v)` uses the identical $(w_R, w_M)$ pair the node composite does. A domain-derived weighting therefore reweights node and edge composites consistently, and the collapse invariant of [§6.4](#64-the-collapse-invariant) holds for edges for exactly the same reason it holds for nodes: it is a property of the projection matrix's row sums, and is indifferent to what the RM vector describes.

---

## 8. Coverage and Declared Gaps

### 8.1 Unmodelled Sub-Characteristics

RM addresses **two of the nine** ISO/IEC 25010:2023 characteristics, and only partially. Stating that precisely is a requirement of using the standard's names.

| Not covered | Level | Why |
|:---|:---|:---|
| **Faultlessness** | Reliability sub-characteristic | Excluded *by definitional choice* — it is the likelihood-bearing sub-characteristic, and [D3](criticality.md#23-consequence-not-risk) is exactly its exclusion. Not a gap; a scoping decision. |
| **Recoverability** | Reliability sub-characteristic | A declared data gap: needs MTTR, restart semantics, or replication state, none of which exists in the schema. Every structural SPOF scores alike regardless of how fast it would be restored. |
| **Reusability, Analysability, Testability** | Maintainability sub-characteristics | Unmodelled. $M$ covers Modularity and Modifiability only — 2 of 5. |
| **Safety** | Whole characteristic | The schema carries no functional integrity class, no hazard catalogue, no safety-criticality field. |
| **Performance efficiency** | Whole characteristic | The discrete-event engine already records time behaviour and capacity ([failure-simulation.md §9](failure-simulation.md#9-what-the-simulator-measures-in-quality-model-terms)), but no RM dimension consumes them. |
| **Functional suitability, Compatibility, Interaction capability, Flexibility, Security** | Whole characteristics | Out of scope entirely. |

> **The Safety row has a consequence worth stating rather than burying.** The ROS 2 / autonomous-vehicle domain of [§6.3](#63-domain-context-vectors) — whose dominant characteristic is freedom from *health and life* risk — is the one domain whose primary harm no dimension estimates directly. A safety-critical deployment can use these scores to *find structural exposure*; it cannot use them to discharge a safety argument, and nothing in the tiering should be read as though it could. Assigned integrity levels (SIL/ASIL/DAL) remain the complementary instrument.

Vulnerability/Security was scored as a third dimension in an earlier revision and was **retired outright, not folded into R or M** — its ground-truth evidence was the weakest of the then-four dimensions, and a fault injector is the wrong instrument for an adversary by construction. [criticality.md §3.5](criticality.md#35-how-the-dimensions-bind-to-external-quality-dependability-and-quality-in-use) carries the full rationale, and [§7.3](criticality.md#73-characteristic-coverage) states coverage per characteristic on both the external-quality and quality-in-use axes.

### 8.2 Instrument Scope Notes

Three properties of the current implementation limit how far a number from this model can be read. They are stated here so they are not mistaken for correctly-working behaviour.

**$CQP$ is synthesised, not computed, for a population with no code metrics.** The min-max helper in `_compute_code_quality_metrics` returns $1.0$ for a zero-variance population — correct for a genuine single node with real values, but indistinguishable from "many nodes, uniformly absent data". Every Library in all six real-world scenario graphs lacks `code_metrics` entirely and therefore receives $CQP = 0.70$, a near-maximal penalty derived from no data. Measured: this does **not** move $\rho(Q, I^*)$ on the Application population ($\Delta\rho = 0.0000$ in each graph, since the shift is a uniform per-type offset and Spearman depends only on within-population order), but it does make the reported *tier* of every real-world Library, and — through the shared box-plot fence — a meaningful minority of every other component in those graphs, a fabricated rather than computed signal. The full accounting is in [criticality.md §3.0](criticality.md#30-three-quality-views-internal-external-and-quality-in-use); a correct fix requires distinguishing the two cases, which the current guard cannot do from normalised values alone.

**Two ingested code fields never move a score.** `cm_avg_cbo` and `cm_avg_rfc` are ingested, flattened onto the vertex, persisted, and shown in the UI — but $CQP$ consumes only `loc`, `cyclomatic_complexity`, `instability_code` and `lcom` ([§3.2](#32-code-measure-elements)). [`saag/core/metric_registry.py`](../saag/core/metric_registry.py) is the authority on which fields reach a score; do not re-derive it by hand.

**The Topic branch of $FT(v)$ is outside the weight system.** Its $0.50/0.50$ split is a hard-coded literal in `_compute_rm`, not a `QualityWeights` field. Three consequences follow: it appears in no AHP matrix, `_perturb_weights` never perturbs it, and `--equal-weights` does not flatten it. **A weight-sensitivity result therefore does not cover Topic nodes' Fault Tolerance term**, and should not be cited as though it did. Every other RM coefficient in this document is a `QualityWeights` field and is covered.

---

## 9. Constant Register

Every constant that moves an RM score, in one place. **Provenance** follows [§2](#2-the-four-layer-model): MEASURED / DERIVED / DECLARED.

**Test coverage** is stated at its actual strength, which is not uniform: *exact* means a test asserts the numeric value (or a formula containing it) and would fail if it changed; *structure* means the test asserts only that the weights are present, positive, ordered, and sum to $\approx 1$, so an individual coefficient could move without failing anything; *none* means no test reads the constant at all. Do not read a `structure` or `none` row as a stable contract.

| Constant | Value | Provenance | Home | Test coverage |
|:---|:---|:---|:---|:---|
| `r_alpha` | 0.36 | **DECLARED** (re-parameterised from retired AHP composite) | [`weight_calculator.py`](../saag/analysis/weight_calculator.py) · re-exported as `RELIABILITY_ALPHA` | **exact** — `test_quality_model.py::TestCompositeReparameterisation` (`abs=0.003`) |
| `q_reliability` ($w_R$) | 0.80 | **DECLARED** (same derivation) | [`weight_calculator.py`](../saag/analysis/weight_calculator.py) | **exact** — same test |
| `q_maintainability` ($w_M$) | 0.20 | **DECLARED** (same derivation) | [`weight_calculator.py`](../saag/analysis/weight_calculator.py) | **exact** — same test |
| FT weights (RPR, DG_in, CDPot) | 0.45 / 0.30 / 0.25 | **DERIVED** (AHP $3\times3$, geometric mean) | [`weight_calculator.py`](../saag/analysis/weight_calculator.py) | *structure* — `test_reliability_dimension.py::TestQualityWeightsV4` (positive, sum $\approx 1$) |
| M weights (BT, w_out, CQP, CR, 1−CC) | 0.35 / 0.30 / 0.15 / 0.12 / 0.08 | **DERIVED** (AHP $5\times5$) | [`weight_calculator.py`](../saag/analysis/weight_calculator.py) | *structure* — `test_maintainability_dimension.py::TestQualityWeightsV5` (positive, sum $\approx 1$, BT > w_out > CR > CC) |
| A weights (AP_c_dir, QSPOF, BR, CDI, w) | 0.35 / 0.25 / 0.25 / 0.10 / 0.05 | **DERIVED** (AHP $5\times5$) | [`weight_calculator.py`](../saag/analysis/weight_calculator.py) | *structure* — `test_availability_dimension.py` (fields present, positive) |
| `_CQP_WEIGHTS` (loc, complexity, instability, LCOM) | 0.10 / 0.35 / 0.30 / 0.25 | **DECLARED** | [`structural_analyzer.py`](../saag/analysis/structural_analyzer.py) | *none* — `test_code_quality_attributes.py` exercises $CQP$'s behaviour, not its coefficients |
| Topic FT split (FOC, CDPot_topic) | 0.50 / 0.50 | **DECLARED**, hard-coded outside `QualityWeights` ([§8.2](#82-instrument-scope-notes)) | [`analyzer.py`](../saag/analysis/analyzer.py) | *none* |
| Edge weights (`e_betweenness`, `e_bridge`, `e_endpoint`, `e_qos_weight`) | 0.35 / 0.30 / 0.20 / 0.15 | **DECLARED** | [`weight_calculator.py`](../saag/analysis/weight_calculator.py) | *none* — no test reads any `e_*` field |
| Impact weights (RL, FR, TL, FD) | 0.35 / 0.25 / 0.25 / 0.15 | **DERIVED** (AHP $4\times4$, post-shrinkage) | [`simulation/models.py`](../saag/simulation/models.py) | *structure* — `test_reliability_dimension.py::TestImpactMetricsIRv` asserts the formula shape via `impact_weights[...]`, not the literals |
| $IFT$ weights (reach, weighted, depth) | 0.45 / 0.35 / 0.20 | **DECLARED** | [`simulation/models.py`](../saag/simulation/models.py) | **exact** — `test_reliability_dimension.py::TestImpactMetricsIRv::test_fault_tolerance_impact_formula` |
| $IA$ weights (reach, fragmentation, throughput) | 0.50 / 0.35 / 0.15 | **DECLARED** | [`simulation/models.py`](../saag/simulation/models.py) | **exact** — `test_availability_dimension.py::TestImpactMetricsIAv::test_availability_impact_formula` |
| $IM$ weights (reach, weighted, depth) | 0.45 / 0.35 / 0.20 | **DECLARED** | [`simulation/models.py`](../saag/simulation/models.py) | **exact** — `test_maintainability_dimension.py::TestImpactMetricsIMv::test_maintainability_impact_formula` |
| `AHP_SHRINKAGE_LAMBDA` ($\lambda$) | 0.70 | **DECLARED** | [`core/models.py`](../saag/core/models.py) | **exact** — `test_ahp_shrinkage.py::test_shrinkage` |
| `COUPLING_PATH_DELTA` ($\delta$) | 0.10 | **DECLARED** | [`core/models.py`](../saag/core/models.py) | **exact** — `test_coupling_risk_hardening.py` |
| `TOPIC_QOS_WEIGHT_BETA` ($\beta$) | 0.85 | **DECLARED** | [`core/models.py`](../saag/core/models.py) | **exact** — `test_domain_model.py` topic-weight cases |
| `MIN_TOPIC_WEIGHT` | 0.01 | **DECLARED** | [`core/models.py`](../saag/core/models.py) | **exact** — `test_domain_model.py` |
| `W_RELIABILITY` / `W_DURABILITY` / `W_PRIORITY` | 0.30 / 0.40 / 0.30 | **DERIVED** (AHP $3\times3$, $CR \approx 0$) | [`core/models.py`](../saag/core/models.py) | **exact** — `test_ahp_shrinkage.py::test_topic_qos_matrix_reproduces_shipped_weights` (`abs=0.01`) |
| `MIN_BOXPLOT_SAMPLE` | 12 | **DECLARED** | [`analyzer.py`](../saag/analysis/analyzer.py) | *none* — referenced nowhere outside the analyzer |
| `QIU_PROJECTION` | $(0.75,0.25),(0.80,0.20),(0.60,0.40)$ | **DECLARED** | [`quality_model.py`](../saag/core/quality_model.py) | **exact** on the row sums and rank — `test_quality_model.py::TestProjectionIsRowStochastic`, `::TestEffectiveRmWeights`; individual coefficients are free to move as long as each row still sums to 1 |
| `DOMAIN_PRIORITIES` | six $\vec\omega$ vectors ([§6.3](#63-domain-context-vectors)) | **DECLARED** | [`quality_model.py`](../saag/core/quality_model.py) | *structure* — `test_quality_model.py::TestDomainPriorities` (sum to 1, declared orderings hold, `hub-and-spoke` absent) |

Three of these are CLAUDE.md invariants — the composite re-parameterisation, the QiU collapse equivalence, and the Topic QoS sub-weights' AHP consistency. Breaking any one fails CI, not just review.

> **The `none` and `structure` rows are the ones to watch when changing a weight.** The four edge coefficients, the four $CQP$ coefficients, the Topic FT split and `MIN_BOXPLOT_SAMPLE` can all be edited without a single test failing — and the FT/M/A dimension vectors can be edited within their ordering constraints. That is a statement about current test coverage, not a licence: those coefficients are load-bearing for every published edge score and every Maintainability score, and changing one silently changes results that no gate would catch.

---

## 10. Where This Fits in the Pipeline

| Stage | What it contributes to the quality model |
|:---|:---|
| **Step 1 — Model** ([graph-model.md](graph-model.md)) | Produces every Layer 0 element: the typed multigraph, the `cm_*` code metrics, and the QoS-derived weights $w(t)$, $w(v)$, $w(e)$ |
| **Step 2 — Analyze** ([structural-analysis.md](structural-analysis.md)) | Computes Layers 1 and 2 for components and edges: the 19 internal measures, then $FT$, $A$, $M$, $R$, $Q$ and the five-tier classification |
| **Step 3 — Predict** ([prediction.md](prediction.md)) | The learned path to the same Layer 2 attributes — different evidence structure, same attributes ([§4.1](#41-two-sources-of-internal-evidence)) |
| **Step 4 — Simulate** ([failure-simulation.md](failure-simulation.md)) | Produces the Layer 2 oracle: $IFT$, $IA$, $IM$, and per-edge severance impact |
| **Step 5 — Validate** ([validation.md](validation.md)) | Attaches the oracle to Layer 2 and nowhere else — `DIMENSION_SPECS`, the statistical battery, and the gates |
| **Step 6 — Prescribe** ([prescription.md](prescription.md)) | Acts on the attributes: each refactoring operator targets one mechanism, and the counterfactual is scored on the same $Q(v)$ |
| **Layer 3** | Opt-in, at analysis time, via `QualityAnalyzer(domain_weights=...)` — a reweighting, never a further stage ([§6.4](#64-the-collapse-invariant)) |

---

## 11. References

**The quality models themselves:**

- ISO/IEC 25010:2023, *Systems and software engineering — SQuaRE — Product quality model*. (Reliability and its four sub-characteristics; Maintainability and its five — the source of every attribute name in [§5](#5-layer-2--external-quality-attributes-isoiec-250102023))
- ISO/IEC 25019:2023, *Systems and software engineering — SQuaRE — Quality-in-use model*. (Beneficialness, Freedom from Risk, Acceptability — the three characteristics of [§6.1](#61-the-three-characteristics))
- ISO/IEC 25021:2012, *Systems and software engineering — SQuaRE — Quality measure elements*. (the Layer 0 notion of [§3](#3-layer-0--quality-measure-elements-isoiec-25021))
- ISO/IEC 25023:2016, *Systems and software engineering — SQuaRE — Measurement of system and software product quality*. (the internal-measure / external-measure distinction separating Layers 1 and 2)
- ISO/IEC 25022:2016, *Systems and software engineering — SQuaRE — Measurement of quality in use*. (the measurement approach nothing in this project performs, cf. [§6](#6-layer-3--quality-in-use-weighting-isoiec-250192023))

**Dependability taxonomy (the fault→error→failure chain that separates $FT$ from $A$, [§5.1](#51-reliability)):**

- Avizienis, A., Laprie, J.-C., Randell, B., & Landwehr, C. (2004). *Basic concepts and taxonomy of dependable and secure computing*. IEEE Transactions on Dependable and Secure Computing, 1(1), 11–33.

**Weight derivation ([§5.4](#54-weight-provenance-and-ahp-shrinkage)):**

- Saaty, T. L. (1980). *The Analytic Hierarchy Process*. McGraw-Hill. (pairwise comparison scale, priority vector, consistency ratio)
- Saaty, T. L. (1990). *How to make a decision: The Analytic Hierarchy Process*. European Journal of Operational Research, 48(1), 9–26. (the random-index table `_calculate_consistency_ratio` uses)

Definitions D1–D4, the stakeholder taxonomy, the literature positioning of "criticality", and the validity discussion are in [criticality.md](criticality.md) and [research/methodology/criticality-construct.md](research/methodology/criticality-construct.md).

---

← [criticality.md](criticality.md) | → [structural-analysis.md §11](structural-analysis.md#11-analyze-stage--rule-based-rm-scoring)
