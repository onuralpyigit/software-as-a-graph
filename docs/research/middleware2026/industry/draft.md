# Graph-Based Pre-Deployment Architecture Verification for Mission-Critical Pub/Sub Middleware: Requirements, Prototype, and Deployment Experience

**Track:** ACM Middleware 2026 — Industrial Track (6 pages)
**Submission deadline:** August 24, 2026

**Authors:** `TODO(data)` — the Industrial Track requires at least one author with an industry
affiliation. See [data/clearance.md](data/clearance.md).

<!--
PAGE BUDGET (ACM sigconf, 6 pages incl. references)
  §1 Introduction ................................. 0.75
  §2 System overview & graph model ................ 1.00
  §3 Verification engine (incl. §3.4 gap table) ... 1.25
  §4 CI/CD gating ................................. 0.75
  §5 Evaluation ................................... 1.75
  §6 Related work + §7 Conclusion ................. 0.50

PROVENANCE RULE — enforced throughout §5:
  Every number carries a tag: [I] = measured in the industrial deployment (cleared, see data/),
  [P] = measured on the open prototype (regenerable from a committed artifact).
  No number ships without an n, a window, and a tag.

CAPABILITY RULE — enforced throughout §2-§4:
  Present tense ("SaaG detects X") is reserved for behaviour implemented in the open prototype.
  Everything specified but not built lives in §3.4 and is written in the requirements voice.
-->

---

## Abstract

Mission-critical distributed systems built on publish/subscribe middleware (DDS, ROS 2, enterprise
message buses) are deployed across multi-core processing units, operator consoles, and networked
execution nodes. Their most damaging defects are not local: overlapping CPU core allocations,
incompatible Quality-of-Service (QoS) contracts between a publisher and its subscribers, topics that
lost their last consumer during a refactor, and circular package dependencies all pass unit and
integration testing and surface as field failures. Reproducing them requires the assembled target
hardware, which is exactly the resource a continuous-delivery pipeline cannot hold.

This paper reports on **System as a Graph (SaaG)**, a pre-deployment architecture verification
approach built for a mission-critical pub/sub program. We contribute three things. First, a
**requirements baseline**: 112 verifiable requirements that specify what static architecture
verification must cover for such a program, derived with its engineering organization and reusable
by others facing the same gap. Second, an **open prototype** implementing the graph model —
five component classes, six structural relation types, and six typed projection rules that derive
logical dependencies from physical pub/sub linkage — together with the subset of checks built so
far: orphaned topics, dependency cycles, and cascade-impact simulation. Third, **deployment
experience**: measurements from the program's continuous-delivery pipeline, and an honest account
of the distance between the specification and the prototype. We report the prototype's static
analysis cost from 29 to 520 components across a corpus of eleven generated and three transcribed
real-world architectures, repeated over five oracle seeds, and the detection catalog's agreement
with a cascade oracle: near-chance Cohen's $\kappa$ on the generated corpus, widening to fair
agreement on the transcribed architectures — a small-sample (n=3) but mechanistically explained
split we report as found rather than smoothing into one number. We argue that the gap between what
such a system is specified to check and what any implementation actually checks is the most useful
thing an industrial report can make explicit.

**Keywords:** pre-deployment verification, pub/sub middleware, QoS contracts, architecture
verification, CI/CD quality gating, DDS.

---

## 1. Introduction

<!-- 0.75 pages -->

Continuous delivery has made building, testing, and shipping software artifacts routine. It has not
made *deploying them into an assembled distributed system* routine. In programs governed by pub/sub
middleware — Data Distribution Service (DDS), ROS 2, enterprise message buses — the scope of what a
pipeline can test remains confined to a software unit's own logic and its immediate interfaces.

### 1.1 The pre-deployment verification gap

In a mission-critical deployment, applications run on multi-core processing nodes, operator
consoles, and networked execution units. Introducing one updated software unit creates non-local
interactions that no unit test observes:

* **Core contention.** Two latency-sensitive units are pinned to overlapping cores, or a node's
  allocations exceed its physical core count. The build is green; the timing violation appears only
  under field load.
* **QoS contract incompatibility.** A subscriber requests `TRANSIENT_LOCAL` durability from a
  publisher offering `VOLATILE`, or requests `RELIABLE` delivery from a `BEST_EFFORT` writer. In
  DDS these are not errors — the endpoints simply never match, and the subscriber receives nothing.
  Silence is the failure mode.
* **Silent topic disconnection.** A refactor removes the last subscriber of a topic, or two units
  bind the same topic name to divergent payload definitions. Both compile.
* **Circular package dependency.** A dependency edge added for convenience closes a cycle that only
  manifests as an initialization deadlock on the target.

Standing up the full target hardware for every candidate build is prohibitively expensive, so teams
fall back on partial staging environments — which is precisely where non-local defects hide. The
result is a class of defects that is cheap to detect statically and expensive to detect any other
way.

### 1.2 What this paper reports

We describe SaaG, an approach that models the system as a typed graph and audits it before
installation. The paper is deliberately structured around three artifacts at different maturity
levels, and is explicit about which is which:

1. **A requirements baseline (§2, §3.4).** 112 verifiable requirements covering model construction,
   scenario generation, field-record management, and design verification, developed with the
   program's engineering organization. We report it because the requirements themselves — not any
   one implementation — are what generalize.
2. **An open prototype (§2.3, §3.1–§3.3).** A working implementation of the graph model and a
   subset of the specified checks, with a CI-usable command-line gate.
3. **Deployment experience (§5).** Measurements from the program's pipeline, paired with
   reproducible measurements on the open prototype.

Our central observation is the one we would have wanted before starting: **the expensive part is not
building the graph, it is the semantic depth of the checks.** Graph construction and structural
audits are cheap and were finished early. The checks that catch the defects in §1.1 — QoS contract
matching with request/offer semantics, core-allocation conformance, design-versus-runtime drift —
require domain models that the graph alone does not supply. §3.4 states exactly which of these are
specified and not yet built, and why.

---

## 2. System Overview and Graph Model

<!-- 1.00 pages -->

### 2.1 Two artifacts, one name

Distinguishing these matters for reading §5, so we state it once, plainly:

| | **SaaG-D** (deployed) | **SaaG-P** (prototype) |
|---|---|---|
| What it is | The verification capability specified by the 112 requirements and integrated into the program's delivery pipeline | An open implementation of the graph model and a subset of the checks |
| Data sources | Configuration management database, source repositories, package repository, network topology descriptors | System descriptor files (JSON), a synthetic scenario generator, transcribed open-source architectures |
| Evaluation in §5 | §5.1, tagged **[I]** | §5.2, tagged **[P]** |
| Availability | Program-internal | Replication package `TODO(url)` |

Measurements from one are never presented as evidence for the other. Where SaaG-D implements a check
that SaaG-P does not, §3.4 says so.

### 2.2 Model setup data generation

The specification requires model setup data to be produced in a controlled, traceable way from four
external sources: the configuration management database (project, platform, and system version
metadata, with the effective version marked), the source and installation-script repositories, the
software package repository, and the network topology descriptor. Each acquired dataset is tagged
with project, platform, and system version, and every mandatory-field or schema failure is recorded
against its source (SSS Req 1.1–1.19).

The candidate-evaluation path matters most for gating: the software unit version inventory is
rebuilt by substituting the *candidate* version of one unit into the otherwise-unchanged inventory
of the target system version (SSS Req 1.11), so the audited graph is the system as it would be after
installation.

In SaaG-P, repository-side extraction is implemented — cloning, XML/IDL-adjacent topic parsing, and
per-unit code metrics — but configuration-management and network-topology ingestion are not; those
inputs are supplied as descriptor files. §3.4 records this.

### 2.3 Graph model

SaaG represents the system as a typed, weighted, directed multigraph $G = (V, E, \tau_V, \tau_E, w)$
over five component classes: **applications** (deployable software units), **libraries** (shared
modules), **topics** (pub/sub channels carrying QoS contracts), **brokers** (middleware routers and
DDS participants), and **nodes** (processing units and operator consoles).

Six relation types are imported directly from the descriptors: `PUBLISHES_TO`, `SUBSCRIBES_TO`
(component ↔ topic), `ROUTES` (broker → topic), `RUNS_ON` (component → node), `CONNECTS_TO`
(node ↔ node), and `USES` (component → library).

**Derived dependencies.** Physical linkage is not dependency. A subscriber depends on the publishers
of the topics it consumes, but the message flows the other way. SaaG therefore derives a
`DEPENDS_ON` relation by six typed projection rules, oriented *from dependent to dependency* —
against the direction of data flow. The rules cover application-to-application coupling through
shared topics (including transitive library chains), application-to-broker coupling, the lifting of
both to node level, library usage, and broker colocation. Where two components share several topics,
one edge is emitted carrying the maximum topic weight and the number of shared topics as a coupling
count.

This direction convention is worth stating twice, because pub/sub diagrams conventionally draw
arrows as data flow: **data flows publisher → subscriber; dependency points subscriber → publisher.**

**QoS-derived weights.** Each topic carries an intrinsic weight combining its QoS contract
(reliability, durability, transport priority) with payload size; component weights propagate upward
from the topics they touch, with library weights amplified by consumer fan-out to reflect blast
radius. The weighting scheme and its calibration are the subject of a companion manuscript
[Anon-A] and are not re-derived here; for this paper it is enough that edges and components carry a
criticality scalar in $[0,1]$ used to rank findings.

### 2.4 Candidate isolation

Concurrent pipeline runs must not corrupt one another or the baseline. Each candidate evaluation
constructs a process-specific model from the candidate unit plus the target system version's other
units, under an independent operation identifier, and analysis never mutates the baseline model
(SSS Req 5.18–5.20, 6.15).

---

## 3. Verification Engine

<!-- 1.25 pages -->

### 3.1 What the prototype checks today

SaaG-P implements a catalog of structural and criticality-weighted detectors over the derived
dependency graph. Those bearing directly on the defect classes of §1.1:

* **Orphaned topics.** A topic with publishers but no subscribers, or subscribers but no publisher,
  is flagged — the structural signature of the refactor described in §1.1.
* **Dependency cycles.** Strongly connected components over the application dependency subgraph
  identify circular package dependencies.
* **Single points of failure and bottlenecks.** Articulation points and bridge edges in the
  dependency graph, ranked by propagated QoS weight.
* **Structural concentration.** Broker overload, topic fan-out, hub-and-spoke concentration, deep
  pipelines, and god components.

Detectors emit findings with a severity (CRITICAL / HIGH / MEDIUM), the affected components, the
rule that fired, and a recommendation.

### 3.2 Cascade impact simulation

Beyond static rules, SaaG-P simulates component failure to estimate blast radius before deployment.
Failure is injected at a component and propagated along structural pub/sub edges — never along the
derived `DEPENDS_ON` edges, an independence property enforced by the prototype's test suite so that
simulation cannot silently consume its own derived output. The result ranks components by cascade
impact, which is what §5.2 uses as a reference oracle when scoring the detector catalog.

### 3.3 Scenario generation

Verification requires system models that do not yet exist in the field. SaaG-P includes a scenario
generator producing synthetic topologies with controlled scale, QoS mix, and clustering, plus a
curated corpus of eleven committed scenarios spanning autonomous vehicles, IoT, financial trading,
healthcare, air traffic management, and enterprise deployments. Each is hash-pinned against the
config that generates it, so published numbers remain reproducible. Three architectures transcribed
from open-source systems (§5.2.3) share the same descriptor schema and load through the identical
pipeline, letting §5.2 compare generated and transcribed topologies directly rather than describing
the latter only qualitatively.

### 3.4 Specified but not yet implemented

The following are required of SaaG-D and are **not** implemented in SaaG-P. We list them because the
gap is the most transferable finding in this paper: each is a check whose value is obvious and whose
cost is dominated by acquiring a domain model the graph does not carry.

| Capability | SSS Req | Why it is not yet built |
|---|---|---|
| **QoS contract conformance** — request/offer matching of durability, reliability, and transport priority between each topic's writers and readers | 6.20 | The prototype flags a QoS *anomaly* — a criticality gap between coupled components — not a contract violation. True conformance requires per-endpoint QoS at reader/writer granularity, whereas the prototype's model attaches QoS to the topic. Closing this means extending the schema below topic level. |
| **Payload schema consistency** — same topic name, divergent content definition | 6.21.3 | Requires parsing and comparing IDL/message definitions across units; the prototype's extractor does not yet resolve type definitions. |
| **Core allocation conformance** — capacity, conflicting pinning, dedicated cores for high-performance units | 6.24–6.27 | Core counts and memory sizes are carried as node attributes but no rule evaluates them. Needs the deployment's core-assignment descriptors, which live outside the sources the prototype ingests. |
| **OS and runtime configuration audit** — OS settings and runtime memory parameters against allocation | 6.25–6.26 | Same cause: the configuration is not in the prototype's input set. |
| **Architectural drift** — designed vs. observed topology from field telemetry | 6.37–6.39 | Requires a field-record store and a graph-diff over observed communication edges. Neither exists in the prototype. |
| **Installation suitability scoring** — four evaluation headings, per-rule weights, blocking flags, aggregate score | 6.51–6.54 | The prototype's gate is severity-based, not score-based (§4). The scoring model is specified in §4.2 as a requirement, not reported as a result. |
| **CMDB and network topology ingestion** | 1.2 | Prototype consumes descriptor files instead. |

Note what the table does *not* say: none of these are blocked on graph technology. Four of the seven
are blocked on **input acquisition** — the information exists in the program but not in a form the
model ingests. That ratio is our main lesson for teams attempting the same thing.

---

## 4. CI/CD Pipeline Integration

<!-- 0.75 pages -->

```
 Developer      Source Repo       CI Server        SaaG           Target Env
    |                |                |              |                |
    |-- push ------->|                |              |                |
    |                |-- webhook ---->|              |                |
    |                |                |-- build candidate unit        |
    |                |                |-- invoke gate (CLI) --------->|
    |                |                |              |-- build candidate graph
    |                |                |              |-- run detectors
    |                |                |<-- exit code + JSON findings  |
    |                |          [exit 0 or 1] -------------------------> deploy
    |                |          [exit 2] -- abort, report to developer |
```
*Figure 2: Gate integration. The prototype's decision is the process exit code.*

### 4.1 The implemented gate

SaaG-P integrates as a command-line invocation. It constructs the candidate graph, runs the
detectors, optionally writes the full finding set as JSON for downstream tooling, and encodes its
decision in the process exit status: **2** if any CRITICAL or HIGH finding is present, **1** if any
finding is present, **0** if none. Standard CI runners treat non-zero as failure, so a two-line
pipeline stage gets absolute severity gating with no plugin.

Two limitations are worth stating because they are the difference between a usable gate and a
tolerated one:

* **The gate is absolute, not delta-aware.** It scores the candidate graph on its own, not against
  the merge base. A codebase carrying pre-existing HIGH findings therefore fails every build until
  they are cleared — which in practice means teams disable the gate. This is not hypothetical on our
  corpus: at CRITICAL/HIGH severity, **all eleven scenarios we evaluated — eight generated, three
  transcribed — return exit code 2 at every one of five oracle seeds tested.** An absolute gate
  deployed as-is would block every build in this corpus, which is precisely the failure mode that
  gets a gate disabled rather than heeded. Delta semantics (fail only on findings the candidate
  *introduces*) and a waiver register for accepted findings are the first changes we would make.
* **There is no severity budget.** One HIGH finding and three hundred are the same decision.

### 4.2 The specified scoring model

SaaG-D is specified to replace the exit-code gate with a scored evaluation over four headings —
structural and architectural conformance; interface, topic, and communication conformance;
dependency and integration conformance; and resource and performance sufficiency (SSS Req 6.51).
Each rule carries an identifier, heading, severity, weight, acceptance criterion, and blocking flag
(SSS Req 6.52). A finding at critical severity, or a violation of any rule marked blocking, yields a
non-conforming installation decision **independently of the aggregate score** (SSS Req 6.53) — the
score is for reporting and trend analysis; blocking is categorical. Multi-unit evaluations run under
independent operation identifiers and return per-unit decisions plus an aggregate result in a
machine-processable form (SSS Req 6.54).

We report this as specification, not as measurement: it is not implemented in SaaG-P, and §5 makes
no claim about its behaviour.

### 4.3 Findings and reports

Each finding carries an identifier, type, description, affected entity, the rule or acceptance
criterion it derives from, supporting evidence, and a severity level (SSS Req 6.44). This
evidence-bearing structure is what makes a failing gate actionable rather than merely obstructive:
the developer receives the offending component and the rule, not a build log.

---

## 5. Evaluation

<!-- 1.75 pages — the largest section, per Industrial Track emphasis -->

Two evaluations, kept separate. §5.1 reports the deployment (**[I]**); §5.2 reports the open
prototype (**[P]**) and is fully reproducible. Neither substitutes for the other: the deployment
numbers describe real engineering impact but cannot be independently verified; the prototype numbers
can be rerun by any reader but describe a subset of the capability.

### 5.1 Deployment experience `TODO(data)`

> **Status: awaiting cleared data.** Every figure below is collected per
> [data/README.md](data/README.md) and must carry an $n$, a window, and provenance. Categories
> appear only where the corresponding check actually ran in the deployment. This subsection is not
> written until the CSVs are filled; no placeholder numbers appear in the draft.

* **5.1.1 System scale** — entity counts for the audited baseline, at the granularity clearance
  permits (`data/system_scale.csv`), plus the anonymization statement from `data/clearance.md`.
* **5.1.2 Verification cost in the pipeline** — per-stage wall-clock with $n$ runs, using the stage
  names the deployed pipeline emits (`data/audit_latency.csv`). The claim to support is that
  verification is negligible against build time — which requires stating build time.
* **5.1.3 Defects caught before deployment** — findings by category over a stated build count and
  window, with each category mapped to the rule and SSS requirement that produced it
  (`data/defect_detection.csv`). Where confirmation status is tracked, report true and false
  positives; where it is not, report findings and say so.
* **5.1.4 Production incidents** — middleware-related incidents before and after gating, normalized
  per release, with window lengths and **explicitly listed confounders**
  (`data/incident_comparison.csv`). A raw percentage drop without the confounder list will not
  survive review, and should not.

### 5.2 Prototype measurements `[P]`

All figures regenerate from a single committed artifact,
`results/detection_seed_sweep.json`, produced by `reproduce/detection_seed_sweep.py` over the
committed corpus — eight generated scenarios (29 to 520 components) plus three architectures
transcribed from open-source systems (60 to 90 components) — at five oracle seeds each
($\{42, 123, 456, 789, 2024\}$), 55 runs total, zero failures.

#### 5.2.1 Verification cost by system scale

End-to-end cost of one gate invocation — graph analysis plus all detectors — against component
count, mean $\pm$ std over the five seeds:

| Scenario | Corpus | Components | Analysis (s) | Detection (s) | Gate total (s) |
|---|---|---:|---:|---:|---:|
| tiny | generated | 29 | 0.02 ± 0.00 | 0.00 ± 0.00 | 0.02 ± 0.00 |
| cloud microservices | transcribed | 60 | 0.03 ± 0.00 | 0.00 ± 0.00 | 0.03 ± 0.00 |
| Autoware.universe | transcribed | 75 | 0.04 ± 0.00 | 0.00 ± 0.00 | 0.04 ± 0.00 |
| Train-Ticket | transcribed | 90 | 0.04 ± 0.00 | 0.00 ± 0.00 | 0.04 ± 0.00 |
| healthcare | generated | 98 | 0.27 ± 0.05 | 0.00 ± 0.00 | 0.27 ± 0.05 |
| financial trading | generated | 124 | 0.60 ± 0.03 | 0.01 ± 0.00 | 0.61 ± 0.03 |
| hub-and-spoke | generated | 139 | 1.21 ± 0.03 | 0.03 ± 0.00 | 1.24 ± 0.03 |
| autonomous vehicle | generated | 152 | 0.78 ± 0.05 | 0.01 ± 0.00 | 0.79 ± 0.05 |
| microservices | generated | 186 | 0.63 ± 0.03 | 0.01 ± 0.00 | 0.64 ± 0.03 |
| IoT smart city | generated | 326 | 1.04 ± 0.05 | 0.04 ± 0.03 | 1.08 ± 0.04 |
| enterprise | generated | 520 | 26.56 ± 0.32 | 0.18 ± 0.01 | 26.74 ± 0.32 |

*Table 2: Gate cost, mean ± std over 5 oracle seeds, layer = system. Source:
`results/detection_seed_sweep.json`.*

Three observations. First, **detection is still cheap**: even on the largest graph the full detector
suite costs under 0.2 s, and summed across the whole corpus the costliest single detector (dependency-
chain enumeration) averages 0.047 s. Rule evaluation is not the cost centre to optimize.

Second, **topology drives cost more than size does, and this is now a robust finding rather than a
single-run curiosity**: hub-and-spoke (139 components) costs more to analyze than IoT smart city
(326 components) — 1.21 s against 1.04 s — and more than autonomous vehicle (152 components) at
0.78 s. Repeated five times, both orderings hold at every seed; this is not measurement noise.

Third, **analysis cost is sharply non-linear at the top of our range, but the absolute number is
sensitive to host load and should be treated that way.** From IoT smart city (326) to enterprise
(520) — a 1.6× increase in components — mean analysis time rises from 1.04 s to 26.56 s, a
$\approx$25× increase, and is tight across seeds ($\pm$0.32 s on enterprise). An earlier single-seed
run of the identical scenario, seed, and code five days prior recorded 51.35 s — roughly double this
sweep's mean — with no change to the analysis path in between. We report the swept figure as
authoritative because it is the repeated one, and flag the discrepancy explicitly: absolute
wall-clock analysis time on this host is contended by other load, so a team sizing CI capacity from
this table should benchmark on its own runner rather than trust the absolute seconds. The scaling
*shape* — cheap until roughly 300 components, then a steep, likely superlinear climb consistent with
the exact betweenness computation in the analysis path — is the portable finding.

#### 5.2.2 Detector catalog against a cascade oracle

We scored the catalog's flagged set against components in the top quintile of simulated cascade
impact. Catalog output is a deterministic function of the graph, and — at this propagation threshold
(0.2) and these graph sizes — the oracle's critical-set membership is nearly seed-invariant too:
every metric was bit-identical across all 5 seeds for all 8 generated scenarios and for two of the
three transcribed ones; Train-Ticket's critical set changed by one component at seed 2024
(recall 0.846 → 0.786). The seed sweep therefore confirms the single-seed measurement was not a
lucky draw, but it does not supply independent repeated samples — the honest unit of analysis is one
value per architecture, not per architecture-seed pair:

| Corpus | $n$ (architectures) | Precision | Recall | $F_1$ | Cohen's $\kappa$ | % scored components flagged |
|---|---:|---:|---:|---:|---:|---:|
| Generated | 8 | 0.237 ± 0.015 | 0.887 ± 0.063 | 0.374 ± 0.023 | −0.036 ± 0.053 | 93.7% |
| Transcribed | 3 | 0.402 ± 0.073 | 0.865 ± 0.126 | 0.546 ± 0.077 | 0.299 ± 0.126 | 56.3% |

*Table 3: Rule-based detection vs. simulated cascade impact. Mean ± std across architectures within
each corpus (5-seed values collapsed to one per architecture, as justified above). Source:
`results/detection_seed_sweep.json`.*

Recall is high and precision is low in both corpora, which is the correct shape for a pre-deployment
gate: missing a genuinely critical component is worse than asking an engineer to dismiss a finding.
**On the generated corpus, Cohen's $\kappa \approx 0$ is the honest headline**: the catalog's
flagging agrees with cascade criticality no better than chance would, achieving recall by flagging
93.7% of scored components — the catalog is barely more informative than flagging everything.

**On the three transcribed architectures, agreement is materially better** — mean $\kappa = 0.30$,
conventionally "fair" agreement, driven by a catalog that is *more selective* on real topologies
(56.3% flagged, not 93.7%) rather than by a more lenient oracle. This ran opposite to what we
expected: we anticipated the generator's cleaner, more regular topology would be *easier* for a
rule-based catalog to rank, not harder. We report it as a small-sample finding — three architectures
is not enough to claim the effect is general — but it is mechanistically explained (over-flagging is
lower, not that criticality shifted) and consistent across all three architectures individually
(range $\kappa \in [0.18, 0.43]$, no overlap with the generated corpus's range), which is more than
we expected from three data points and is worth confirming on a larger transcribed corpus before
treating it as established.

We draw the conclusion the generated corpus forces, with the transcribed result as a qualifier
rather than a rebuttal: **structural rules alone are adequate for *locating* defects of known shape
(an orphaned topic is an orphaned topic) and, on synthetic topologies generated by our own
procedure, inadequate for *ranking* components by consequence.** Severity in the catalog is a
property of the rule, not of the component's position in the system, and the gate inherits that on
the corpus where we can measure it at scale. Whether that inadequacy is intrinsic to rule-based
detection or an artifact of our generator's structural priors is exactly what the transcribed split
leaves open. This is the strongest argument we have for the impact simulation of §3.2 being part of
the gate rather than an offline analysis — and, separately, for the criticality prediction work
reported in the companion manuscript [Anon-A].

#### 5.2.3 Coverage on transcribed open-source architectures

To check that the model is not overfitted to its own generator, we made the three transcribed
architectures loadable through the same pipeline as the generated corpus (they previously required a
bespoke script) and ran the full gate against each:

| Architecture | Apps | Topics | Brokers | Nodes | Libs | Relations | Findings | Gate exit code |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Autoware.universe (ROS 2) | 32 | 24 | 3 | 6 | 10 | 179 | 67 | 2 (all 5 seeds) |
| Train-Ticket benchmark | 41 | 30 | 3 | 8 | 8 | 162 | 78 | 2 (all 5 seeds) |
| Cloud microservices mesh | 22 | 20 | 4 | 6 | 8 | 128 | 50 | 2 (all 5 seeds) |

*Table 4: Transcribed architectures — scale, measured findings, and gate outcome. Findings are
deterministic (std = 0 across seeds). Source: `data/scenarios/realworld_*.json`,
`results/detection_seed_sweep.json`.*

**These are hand-built models of real architectures, not harvested artifacts.** The topologies were
transcribed from published component structures into the model's schema; no repository crawling or
runtime introspection produced them. What changed in this revision is that they moved from a
single example script's inline schema translation into the model's canonical schema, so the same
gate, the same detector catalog, and the same oracle now run on them unmodified — which is what
makes Table 3's transcribed row and this table's findings counts and gate outcomes actual
measurements rather than a claim that the pipeline merely "runs." They remain three architectures
transcribed by us, not independently sourced or extraction-verified, and we do not present them as
evidence of extraction accuracy.

### 5.3 Threats to validity

* **Two systems, one name.** §5.1 and §5.2 measure different artifacts (§2.1). The prototype
  implements a subset of the deployed capability; prototype cost figures are lower bounds on
  deployed cost.
* **Simulated oracle.** §5.2.2 scores detection against a simulator, not against observed field
  failures. The simulator is a model of cascade propagation, and agreement with it is not the same
  as agreement with reality. A second independent oracle in the same codebase agrees with this one
  at only $\rho \approx 0.40$, which bounds how far any figure measured on one transfers to the
  other.
* **Wall-clock is host-dependent.** Table 2's enterprise-scenario analysis time (26.56 s, repeated
  five times) came in at roughly half an isolated single-seed measurement of the identical
  scenario, seed, and code taken five days earlier (51.35 s). The five-seed repetition confirms the
  swept figure is not itself a fluke, but neither figure should be read as a hardware-independent
  constant; only the scaling shape is portable.
* **Effective sample size in Table 3 is 8 and 3 architectures, not 40 and 15 seed-pairs.** Catalog
  output is deterministic given the graph, and the oracle's critical-set membership was seed-invariant
  for 10 of the 11 scenarios (Train-Ticket's shifted by one component at one of five seeds). The seed
  sweep therefore rules out "we got lucky with seed 42" but does not add independent statistical
  power beyond one measurement per architecture — a caveat we apply to our own $\kappa = 0.30$ finding
  on the transcribed corpus, which rests on $n=3$.
* **Corpus composition.** Eight of eleven scenarios scored in §5.2.2 are generator-produced; the
  transcribed architectures narrow but do not close the gap to a field distribution, and the
  direction of their difference (higher $\kappa$, lower over-flagging) is itself only a 3-point
  observation.
* **Confounders in §5.1.4.** Gating was not the only change in the observation window; the listed
  confounders bound the causal claim.

---

## 6. Related Work

<!-- 0.25 pages -->

**Architecture description languages.** AADL and SysML model hardware/software interaction and
support analysis of the same properties we target [Feiler & Gluch]. Their practical obstacle in a
delivery pipeline is maintenance: the model is authored separately from the implementation and
drifts. SaaG's model is reconstructed per candidate build from the artifacts the build already
produces, which trades expressiveness for currency.

**Static code analysis.** SonarQube, Coverity, and similar platforms analyze source and local
control flow. They are structurally unable to see topic bindings, QoS contracts, core assignments,
or node topology — the system exists only in the composition of the units they analyze
individually. SaaG complements them at the architectural boundary rather than competing.

**Runtime observability.** Prometheus, Dynatrace, and distributed tracing observe the deployed
system with fidelity no static model matches, but only after the non-conforming artifact is running.
The specified drift detection of §3.4 is precisely an attempt to route that observation back into
pre-deployment verification.

**Middleware dependability.** Pub/sub dependability has been studied at the protocol, broker
overlay, and runtime levels. Our contribution is neither a protocol nor a runtime mechanism but a
pipeline-integrated static check over the deployment's own descriptors.

<!-- TODO(refs): expand from 6 to ~18 citations before submission. Reuse the bibliography assembled
     in docs/research/middleware2026/research/middleware26_revision_plan.md §A1 (Eugster, Carzaniga,
     DDS spec, MQTT, Freeman, Brandes). Add: DDS QoS conformance literature; deployment/config
     verification (e.g. configuration error studies); CI quality-gate practice. The prior main-track
     rejection cited a reference-free introduction as a weakness — §1 needs citations woven in. -->

---

## 7. Conclusion

<!-- 0.25 pages -->

We reported a graph-based approach to pre-deployment architecture verification for mission-critical
pub/sub middleware: a 112-requirement baseline developed with an industrial program, an open
prototype implementing the graph model and a subset of the specified checks, and measurements from
both the deployment and the prototype.

The findings we would most want carried forward are the negative ones, plus one we did not expect.
Building the graph and running structural rules is cheap — all twenty detectors cost under 0.2 s on
a 520-component system, confirmed across five oracle seeds. Ranking findings by consequence is not,
on the synthetic corpus we generate ourselves: rule-based severity agrees with simulated cascade
impact at $\kappa \approx 0$, so a catalog of structural rules locates defects well and prioritizes
them badly there. But on three architectures transcribed from real open-source systems, agreement
rose to fair ($\kappa \approx 0.30$) — a small-sample result we report rather than smooth over,
because it points at our synthetic generator's structural priors, not at rule-based detection in
general, as the likely source of the near-chance result. And of the seven specified checks we have
not built, four are blocked not on analysis technique but on **getting the configuration data into
the model at all** — core assignments, OS settings, and runtime parameters exist in the program and
not in the pipeline's reach. Teams planning similar work should budget for data acquisition, not for
graph algorithms.

**Future work.** Delta-aware gating against the merge base with a waiver register (§4.1); QoS
conformance at reader/writer granularity, which requires extending the schema below topic level
(§3.4); and drift detection, which requires the field-record path the specification already
describes.

---

## References

<!-- TODO(refs): convert to ACM format; expand per the note in §6. -->

1. Object Management Group. *Data Distribution Service (DDS) Specification*, Version 1.4, 2015.
2. P. H. Feiler and D. P. Gluch. *Model-Based Engineering with AADL*. Addison-Wesley, 2012.
3. J. Humble and D. Farley. *Continuous Delivery*. Addison-Wesley, 2010.
4. L. Bass, P. Clements, and R. Kazman. *Software Architecture in Practice*, 4th ed.
   Addison-Wesley, 2021.
5. P. Th. Eugster, P. A. Felber, R. Guerraoui, and A.-M. Kermarrec. The many faces of
   publish/subscribe. *ACM Computing Surveys*, 35(2), 2003.
6. A. Carzaniga, D. S. Rosenblum, and A. L. Wolf. Design and evaluation of a wide-area event
   notification service. *ACM TOCS*, 19(3), 2001.
7. U. Brandes. A faster algorithm for betweenness centrality. *Journal of Mathematical Sociology*,
   25(2), 2001.
8. [Anon-A] Companion manuscript on QoS-weighted criticality prediction. Under review.
