# An Architectural Digital Twin for Pre-Deployment Verification and CI/CD Gating in Distributed Middleware Systems

**Track:** ACM Middleware 2026 — Industrial Track (6 pages)
**Submission deadline:** August 24, 2026

**Authors:** `TODO(data)` — the Industrial Track requires at least one author with an industry
affiliation. See [data/clearance.md](data/clearance.md).

<!--
PAGE BUDGET (ACM sigconf, 6 pages incl. references)
  §1 Introduction ................................. 0.75
  §2 System overview & graph model ................ 1.00
  §3 Verification engine (incl. §3.4 gap table) ... 1.10
  §4 CI/CD gating .................................. 0.90
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

Distributed systems built on publish/subscribe and broker-mediated middleware (DDS, ROS 2,
enterprise message buses) are deployed across multi-core processing units, operator consoles, and
networked execution nodes. Their most damaging defects are not local: overlapping CPU core
allocations, incompatible Quality-of-Service (QoS) contracts between a publisher and its
subscribers, topics that lost their last consumer during a refactor, and circular package
dependencies all pass unit and integration testing and surface as field failures. Reproducing them
requires the assembled target hardware, which is exactly the resource a continuous-delivery pipeline
cannot hold.

This paper reports on **System as a Graph (SaaG)**, an architectural digital twin for pre-deployment
verification and CI/CD gating, built for a mission-critical pub/sub program. The twin is *static*: a
typed graph reconstructed fresh from the candidate system version for each pipeline run, not a
continuously live-synchronized model — the dynamic half of a full digital twin, telemetry overlay
and drift detection against the design, is specified but not yet built, and we say so explicitly
rather than blur the line. We contribute three things. First, the **digital twin model and its
requirements baseline**: five component classes, six structural relation types, and six typed
projection rules that derive logical dependencies from physical pub/sub linkage, backed by 112
verifiable requirements that specify what static architecture verification must cover for a program
like this one, derived with its engineering organization and reusable by others facing the same gap.
Second, a **CI/CD gate**: an implemented severity-based exit-code gate, paired with a scored
suitability model specified but not yet built, over a checks catalog covering orphaned topics,
dependency cycles, and cascade-impact simulation. Third, **deployment experience**: measurements
from the program's continuous-delivery pipeline, and an honest account of the distance between the
specification and the prototype. We evaluate the prototype primarily on a family of five systems in
the deployment's own domain — air traffic management — spanning 29 to 444 components with QoS mix,
fan-out shape, and criticality proportion held fixed across scale, isolating scale as the sole
independent variable: verification cost grows gently (0.01 s to 1.01 s end-to-end) but the detector
catalog's agreement with a cascade oracle declines monotonically from slight to chance as the same
system grows (Cohen's $\kappa$: 0.12 at 29 components to $-0.05$ at 444). A secondary check across
six other synthetic domains plus three architectures transcribed from real open-source systems finds
the same near-chance pattern on other large synthetic graphs, but materially better (fair) agreement
on the transcribed ones — a small-sample (n=3) split, unresolved by this paper's evidence, over
whether scale or synthetic generation is the better explanation. We argue that the gap between what
such a system is specified to check and what any implementation actually checks — and the discovery
that verification quality is not scale-invariant even within one domain — are the most useful things
an industrial report can make explicit.

**Keywords:** architectural digital twin, pre-deployment verification, pub/sub middleware, QoS
contracts, CI/CD gating, DDS.

---

## 1. Introduction

<!-- 0.75 pages -->

Continuous delivery has made building, testing, and shipping software artifacts routine. It has not
made *deploying them into an assembled distributed system* routine. In programs governed by pub/sub
middleware — Data Distribution Service (DDS), ROS 2, enterprise message buses — the scope of what a
pipeline can test remains confined to a software unit's own logic and its immediate interfaces.

### 1.1 The pre-deployment verification gap

In a large distributed pub/sub deployment, applications run on multi-core processing nodes,
operator consoles, and networked execution units. Introducing one updated software unit creates
non-local interactions that no unit test observes:

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

1. **A digital twin model and its requirements baseline (§2, §3.4).** The static graph model itself,
   plus 112 verifiable requirements covering model construction, scenario generation, field-record
   management, and design verification, developed with the program's engineering organization. We
   report the requirements because they — not any one implementation — are what generalize.
2. **A CI/CD gate: implemented and specified halves (§4).** A working exit-code gate over the
   detector catalog (§3.1–§3.3), and the scored installation-suitability model the specification
   calls for but the prototype does not yet build.
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

**What kind of digital twin this is.** "Digital twin" is used precisely here, not loosely: SaaG-P
builds a *static* twin — a graph reconstructed fresh for each candidate build (§2.4), not a model
kept continuously synchronized with the running system. A full digital twin has a dynamic half too:
overlaying live telemetry onto the model and detecting drift between the designed graph and what is
actually observed at runtime. That half is specified but not implemented in
SaaG-P; §3.4 lists it alongside the other specified-but-unbuilt checks rather than folding it
silently into what the prototype already does. Everything this paper reports as a capability of
SaaG-P is the static half.

### 2.2 Model setup data generation

The specification requires model setup data to be produced in a controlled, traceable way from four
external sources: the configuration management database (project, platform, and system version
metadata, with the effective version marked), the source and installation-script repositories, the
software package repository, and the network topology descriptor. Each acquired dataset is tagged
with project, platform, and system version, and every mandatory-field or schema failure is recorded
against its source.

The candidate-evaluation path matters most for gating: the software unit version inventory is
rebuilt by substituting the *candidate* version of one unit into the otherwise-unchanged inventory
of the target system version, so the audited graph is the system as it would be after
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
units, under an independent operation identifier, and analysis never mutates the baseline model.

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
curated corpus of fifteen committed scenarios: ten spanning autonomous vehicles, IoT, financial
trading, healthcare, and enterprise deployments at various scales, plus — the evaluation's primary
focus, §5.2 — a five-scenario family in the air-traffic-management domain that mirrors the deployment
motivating this paper (§2.2), spanning 29 to 444 components and sharing an identical QoS-mix,
fan-out-shape, and criticality-proportion profile so that scale is the only thing that varies between
them. Each is
hash-pinned against the config that generates it, so published numbers remain reproducible. Three
architectures transcribed from open-source systems (§5.2.3) share the same descriptor schema and load
through the identical pipeline, letting the secondary evaluation compare generated and transcribed
topologies directly rather than describing the latter only qualitatively.

### 3.4 Specified but not yet implemented

The following are required of SaaG-D and are **not** implemented in SaaG-P. We list them because the
gap is the most transferable finding in this paper: each is a check whose value is obvious and whose
cost is dominated by acquiring a domain model the graph does not carry.

| Capability | Why it is not yet built |
|---|---|
| **QoS contract conformance** — request/offer matching of durability, reliability, and transport priority between each topic's writers and readers | The prototype flags a QoS *anomaly* — a criticality gap between coupled components — not a contract violation. True conformance requires per-endpoint QoS at reader/writer granularity, whereas the prototype's model attaches QoS to the topic. Closing this means extending the schema below topic level. |
| **Payload schema consistency** — same topic name, divergent content definition | Requires parsing and comparing IDL/message definitions across units; the prototype's extractor does not yet resolve type definitions. |
| **Core allocation conformance** — capacity, conflicting pinning, dedicated cores for high-performance units | Core counts and memory sizes are carried as node attributes but no rule evaluates them. Needs the deployment's core-assignment descriptors, which live outside the sources the prototype ingests. |
| **OS and runtime configuration audit** — OS settings and runtime memory parameters against allocation | Same cause: the configuration is not in the prototype's input set. |
| **Architectural drift** — designed vs. observed topology from field telemetry (the dynamic half of the digital twin described in §2.1) | Requires a field-record store and a graph-diff over observed communication edges. Neither exists in the prototype. |
| **Installation suitability scoring** — four evaluation headings, per-rule weights, blocking flags, aggregate score | The prototype's gate is severity-based, not score-based (§4). The scoring model is specified in §4.2 as a requirement, not reported as a result. |
| **CMDB and network topology ingestion** | Prototype consumes descriptor files instead. |

Note what the table does *not* say: none of these are blocked on graph technology. Four of the seven
are blocked on **input acquisition** — the information exists in the program but not in a form the
model ingests. That ratio is our main lesson for teams attempting the same thing.

---

## 4. CI/CD Gating

<!-- 0.90 pages -->

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
  evaluation: at CRITICAL/HIGH severity, **all five ATM scale points (§5.2.1–§5.2.2), and all eleven
  scenarios in the secondary corpus (§5.2.3, eight other domains plus three transcribed) — sixteen
  systems in total — return exit code 2 at every one of five oracle seeds tested.** An absolute gate
  deployed as-is would block every build we evaluated, which is precisely the failure mode that
  gets a gate disabled rather than heeded. Delta semantics (fail only on findings the candidate
  *introduces*) and a waiver register for accepted findings are the first changes we would make.
  A learned, delta-aware, simulation-verified gate is exactly what the companion manuscript
  [Anon-A] reports — that paper's contribution is the gate this one identifies as missing, and we
  cite it as the direction rather than duplicating it here.
* **There is no severity budget.** One HIGH finding and three hundred are the same decision.

### 4.2 The specified scoring model

SaaG-D is specified to replace the exit-code gate with a scored evaluation over four headings —
structural and architectural conformance; interface, topic, and communication conformance;
dependency and integration conformance; and resource and performance sufficiency. Each rule carries
an identifier, heading, severity, weight, acceptance criterion, and blocking flag. A finding at
critical severity, or a violation of any rule marked blocking, yields a non-conforming installation
decision **independently of the aggregate score** — the score is for reporting and trend analysis;
blocking is categorical. Multi-unit evaluations run under independent operation identifiers and
return per-unit decisions plus an aggregate result in a machine-processable form.

We report this as specification, not as measurement: it is not implemented in SaaG-P, and §5 makes
no claim about its behaviour.

### 4.3 Findings and reports

Each finding carries an identifier, type, description, affected entity, the rule or acceptance
criterion it derives from, supporting evidence, and a severity level. This
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
  window, with each category mapped to the rule that produced it
  (`data/defect_detection.csv`). Where confirmation status is tracked, report true and false
  positives; where it is not, report findings and say so.
* **5.1.4 Production incidents** — middleware-related incidents before and after gating, normalized
  per release, with window lengths and **explicitly listed confounders**
  (`data/incident_comparison.csv`). A raw percentage drop without the confounder list will not
  survive review, and should not.

### 5.2 Prototype measurements `[P]`

Two studies. The **primary study** (§5.2.1–§5.2.2) evaluates a family of five systems in the
deployment's own domain — air traffic management — at five scales, regenerating from a single
committed artifact, `results/atm_scale_sweep.json`, produced by `reproduce/atm_scale_sweep.py`. The
**secondary study** (§5.2.3), carried forward unchanged from the prior revision, checks whether the
primary study's pattern generalizes across *other* domains rather than across scale within one.

The primary study's five systems — 29, 74 (the original `atm_system` scenario, unchanged),
148, 296, and 444 components — are generated from configs
(`data/scenarios/scenario_1{4,0,5,6,7}_atm_*.yaml`) that hold QoS mix, fan-out shape, and criticality
proportion identical across scales and vary only entity counts, so **scale is the sole independent
variable** — an improvement on the secondary study's corpus, where domain and scale changed together
and could not be separated. Each of the five is evaluated at five oracle seeds
($\{42, 123, 456, 789, 2024\}$), 25 runs total, zero failures.

#### 5.2.1 Verification cost by ATM scale

End-to-end cost of one gate invocation — graph analysis plus all detectors — against component
count, domain held fixed, mean $\pm$ std over the five seeds:

| Scale | Components | Analysis (s) | Detection (s) | Gate total (s) |
|---|---:|---:|---:|---:|
| tiny | 29 | 0.01 ± 0.00 | 0.00 ± 0.00 | 0.01 ± 0.00 |
| small (`atm_system`) | 74 | 0.05 ± 0.00 | 0.00 ± 0.00 | 0.05 ± 0.00 |
| medium | 148 | 0.13 ± 0.00 | 0.00 ± 0.00 | 0.13 ± 0.00 |
| large | 296 | 0.52 ± 0.02 | 0.01 ± 0.00 | 0.53 ± 0.02 |
| xlarge | 444 | 0.99 ± 0.03 | 0.02 ± 0.00 | 1.01 ± 0.03 |

*Table 2: Gate cost by ATM scale, mean ± std over 5 oracle seeds, layer = system. Source:
`results/atm_scale_sweep.json`.*

Two observations. First, **detection is cheap here too, more so than the secondary corpus**: even at
444 components the full detector suite costs 0.02 s, and summed across all five scales the costliest
single detector (cycle detection) averages 0.005 s — cheaper than the secondary corpus's already-cheap
0.047 s (§5.2.3), because this domain's fixed fan-out shape keeps the dependency graph sparse at every
scale.

Second, **this controlled sweep resolves an ambiguity the secondary corpus left open.** There,
analysis cost jumped sharply between 326 and 520 components (a 25× increase for a 1.6× size increase),
and it was unclear whether that reflected component count or the specific topology of that one
520-component scenario. Here, holding domain and topology shape fixed and scaling components alone
from 29 to 444 (a 15× increase) produces a **roughly proportional** cost increase — 0.01 s to 0.99 s,
about 100× — with no cliff. The secondary corpus's steep jump was therefore a property of that one
scenario's structure (or, per its own wall-clock caveat, of host load at measurement time), not a
general law that cost explodes past a few hundred components. For this paper's motivating domain, at
least, cost growth is gentle.

#### 5.2.2 Detector catalog against a cascade oracle, by ATM scale

We scored the catalog's flagged set against the oracle's critical set — box-plot top tier ($\ge Q_3$)
once a scale has enough scored components to estimate quartiles, a top-20% mask below that, the same
rule applied elsewhere in this codebase — at each of the five scales. Unlike §5.2.1's cost
measurement, this result is a genuine trend across scale rather than a single robustness check, so we
report one row per scale instead of pooling:

| Scale | Components | Precision | Recall | $F_1$ | Cohen's $\kappa$ | % scored flagged | Seed-invariant |
|---|---:|---:|---:|---:|---:|---:|---|
| tiny | 29 | 0.250 | 1.000 | 0.400 | 0.118 | 80.0% | Yes |
| small | 74 | 0.273 | 0.900 | 0.419 | 0.041 | 84.6% | Yes |
| medium | 148 | 0.269 | 0.900 | 0.414 | 0.031 | 85.9% | Yes |
| large | 296 | 0.250 | 0.949 | 0.396 | 0.000 | 94.9% | Yes |
| xlarge | 444 | 0.235 ± 0.002 | 0.868 ± 0.007 | 0.370 ± 0.003 | −0.045 ± 0.005 | 93.2% | No |

*Table 3: Rule-based detection vs. simulated cascade impact, by ATM scale. "Seed-invariant" scales
report a single value (all 5 seeds bit-identical); xlarge — the one exception — reports mean ± std
(4 of 5 seeds agree exactly; one seed's critical set differs by a single component). Source:
`results/atm_scale_sweep.json`.*

**Agreement declines monotonically as the same ATM system grows**, from $\kappa = 0.118$ (slight
agreement) at 29 components to $\kappa \approx -0.045$ (no better than chance) at 444, crossing zero
at the 296-component point. The mechanism is visible in the same table: the fraction of scored
components the catalog flags climbs from 80.0% to 93–95% as scale increases, so precision erodes
steadily (0.273 → 0.235) while recall stays high throughout (0.868–1.000). This is the same
over-flagging mechanism identified in the secondary corpus (§5.2.3) — but here it is a genuine trend
*within* one domain rather than a difference *between* domains: **as this ATM family grows, findings
volume stays cheap to compute (§5.2.1), but the catalog gets steadily less discriminating.**

**This is a mechanical consequence of how the catalog verdict is built, not evidence it fails to find
critical components.** Elsewhere in this evaluation, the RMAV composite score $Q(v)$ is deliberately
threshold-matched to the oracle's critical-set size on both sides, so its precision and recall
converge by construction. The catalog verdict has no such matching: it is a binary "flagged by at
least one of the catalog's rules" decision, and its flagged fraction is whatever the union of
independent rule firings happens to produce — not a calibrated decision boundary. Because of this,
precision here is fully determined by three quantities,
$\text{precision} = \text{base\_rate} \times \text{recall} \, / \, \text{flagged\_fraction}$,
where base rate is the oracle's critical-set share — a near-constant $\approx 25\%$ at every ATM
scale, since the threshold rule is the same one throughout. This identity reproduces the measured
precision to three decimal places at every scale in Table 3 (xlarge: $0.252 \times 0.864 / 0.932 =
0.234$, measured $0.235$). Once base rate and recall are both roughly fixed, **$F_1$'s entire trend
is the flagged-fraction trend read backwards** — the same 80.0%-to-93–95% climb driving both. A
reader expecting $F_1$ to behave like a calibrated classifier's score should not: the catalog was
never tuned to any particular flagged-fraction, so a low $F_1$ reflects that design choice, not a
failure to detect — recall, the metric that *does* directly measure missed detections, never drops
below 0.868.

We read the *decline itself* cautiously, independent of the mechanical point above. The generator's
domain skin recycles the same six named application roles
at every scale (§3.4): "444 components" means roughly seventeen additional copies of each of
`radar-tracker`, `flight-plan-processor`, `conflict-detector`, `weather-analyzer`, `clearance-router`,
and `trajectory-predictor`, not new subsystem types. Whether $\kappa$'s decline reflects something
structural about rule-based detection on denser, same-shaped graphs, or is an artifact of this
specific way of scaling a domain template, is exactly the kind of question a real, growing deployment
could answer and a synthetic scale sweep cannot settle alone. We report the trend because it is
measured and monotonic, not because we are confident it generalizes to how a real ATM system would
actually grow.

#### 5.2.3 External validity: generality beyond ATM

The primary study asks whether verification quality holds as *one* domain scales. A separate
question — does a pattern found on one domain generalize to *other* domains? — was the focus of the
prior revision's evaluation, retained here as a secondary check. It used eight generated scenarios
spanning six other domains (29 to 520 components, domain and scale varying together) plus three
architectures transcribed from open-source systems (Autoware.universe ROS 2, the Train-Ticket
microservices benchmark, and a cloud-native microservices mesh; 60–90 components), at the same five
oracle seeds:

| Corpus | $n$ (architectures) | Precision | Recall | $F_1$ | Cohen's $\kappa$ | % scored flagged |
|---|---:|---:|---:|---:|---:|---:|
| Generated (6 other domains) | 8 | 0.237 ± 0.015 | 0.887 ± 0.063 | 0.374 ± 0.023 | −0.036 ± 0.053 | 93.7% |
| Transcribed (real architectures) | 3 | 0.402 ± 0.073 | 0.865 ± 0.126 | 0.546 ± 0.077 | 0.299 ± 0.126 | 56.3% |

*Table 4: Rule-based detection vs. simulated cascade impact, cross-domain secondary corpus (unchanged
from the prior revision). Source: `results/detection_seed_sweep.json`.*

Agreement on the six other generated domains was near-chance ($\kappa \approx 0$) — the same
qualitative result as this paper's ATM finding at large scale — but agreement on the three
*transcribed, real* architectures was materially better (fair agreement, $\kappa \approx 0.30$),
driven by less over-flagging (56.3% vs. 93.7–94.9%) rather than a more lenient oracle. The same
mechanical identity verified in §5.2.2 ($\text{precision} = \text{base\_rate} \times \text{recall} \,
/ \, \text{flagged\_fraction}$) reproduces every one of this corpus's eleven precision values exactly
as well: with the oracle's base rate pinned near 25% and recall consistently $0.75$–$1.00$ across both
sub-corpora, the transcribed architectures' higher precision is arithmetically explained by their much
smaller flagged fraction — not by the catalog behaving differently, rule-for-rule, on real topologies.
Taken together
with §5.2.2's within-ATM trend, the pattern that best fits all the evidence in this paper is: **rule-
based flagging is least discriminating on large, synthetically-generated graphs regardless of domain,
and more discriminating on smaller and/or non-synthetic ones** — scale and "generated-ness" point the
same direction here, and this corpus's transcribed architectures (real, but also small: 60–90
components) cannot cleanly separate which of the two actually drives the effect. A real, growing ATM
deployment is the only clean way to settle that — exactly the kind of thing §5.1's pending deployment
data could speak to if the relevant categories are collected.

This secondary corpus carries its own already-established caveats, unchanged from the prior revision:
$n=8$ and $n=3$ architectures, not 40 and 15 seed-pairs (catalog output was seed-invariant for 10 of
11 scenarios there), and its enterprise-scenario timing showed a 2× wall-clock discrepancy against an
earlier isolated measurement. The transcribed architectures are hand-built models of real
architectures, not harvested artifacts — no repository crawling or runtime introspection produced
them — and are not evidence of extraction accuracy.

### 5.3 Threats to validity

* **Two systems, one name.** §5.1 and §5.2 measure different artifacts (§2.1). The prototype
  implements a subset of the deployed capability; prototype cost figures are lower bounds on
  deployed cost.
* **Simulated oracle.** §5.2.2 and §5.2.3 score detection against a simulator, not against observed
  field failures. The simulator is a model of cascade propagation, and agreement with it is not the
  same as agreement with reality. A second independent oracle in the same codebase agrees with this
  one at only $\rho \approx 0.40$, which bounds how far any figure measured on one transfers to the
  other.
* **Same six roles, recycled, not new subsystems.** The ATM generator's domain skin (§3.4) supplies
  six named application roles and a fixed five-cluster hierarchy regardless of scale; growing from 29
  to 444 components means more numbered copies of the same six roles, not new subsystem types.
  §5.2.2's scale trend describes what happens when *this generator's* ATM template is replicated,
  which may or may not resemble how a real ATM deployment adds capacity (new sensor modalities, new
  consoles, genuinely new subsystems).
* **Effective sample size in Table 3 is 5 scale points, not 25 scale-seed pairs.** Four of five are
  seed-confirmed exactly; the fifth (xlarge) to within std $\le 0.005$. The seed sweep rules out "we
  got lucky with one seed" at every scale, but one generator draw per scale is one measured
  trajectory, not several independent ones — the monotonic trend should be read as a single
  observation, not an averaged effect.
* **Corpus composition.** The primary study (§5.2.1–§5.2.2) is entirely single-domain and synthetic.
  The only cross-domain, and only partially non-synthetic, evidence in the paper is the secondary
  corpus (§5.2.3), which itself rests on $n=8$ and $n=3$.
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

**Static code analysis and runtime observability.** SonarQube, Coverity, and similar platforms
analyze source and local control flow; they are structurally unable to see topic bindings, QoS
contracts, core assignments, or node topology, since the system exists only in the composition of
the units they analyze individually. Observability platforms (Prometheus, Dynatrace, distributed
tracing) see the deployed system with fidelity no static model matches, but only after the
non-conforming artifact is already running. SaaG complements the former at the architectural
boundary and — through the specified drift detection of §3.4 — aims to route the latter's kind of
observation back into pre-deployment verification rather than after-the-fact alerting.

**Digital twins.** The term is established in manufacturing and Industry 4.0, where a digital twin
synchronizes a model against a physical asset via continuous sensor telemetry [Tao et al.]. The
object and the fidelity are both different here: the twin in this paper models a
*software architecture*, not a physical asset, and — as scoped in §2.1 — is *static*, reconstructed
per candidate build rather than continuously live. We use the term because the reconstruct-and-audit
pattern is the same one digital-twin systems apply to physical assets, and because the specified,
not-yet-built dynamic half (§3.4) is exactly the telemetry-synchronization step that literature
describes; we do not use it to claim a live-synchronized model we have not built.

**Middleware dependability.** Pub/sub dependability has been studied at the protocol, broker
overlay, and runtime levels. Our contribution is neither a protocol nor a runtime mechanism but a
pipeline-integrated static check over the deployment's own descriptors.

<!-- TODO(refs): expand from 6 to ~18 citations before submission. Reuse the bibliography assembled
     in docs/research/middleware2026/research/middleware26_revision_plan.md §A1 (Eugster, Carzaniga,
     DDS spec, MQTT, Freeman, Brandes). Add: DDS QoS conformance literature; deployment/config
     verification (e.g. configuration error studies); CI quality-gate practice; a manufacturing/
     Industry 4.0 digital-twin survey to back the "[Tao et al.]" placeholder in the new Digital
     Twins paragraph above (added when the title reverted to "architectural digital twin" — this
     citation is now load-bearing, not optional, since it's what lets the paper reuse the term
     against the literature that owns it). The prior main-track rejection cited a reference-free
     introduction as a weakness — §1 needs citations woven in. -->

---

## 7. Conclusion

<!-- 0.25 pages -->

We reported an architectural digital twin — reconstructed per candidate build, not continuously
live — for pre-deployment verification and CI/CD gating in distributed pub/sub middleware systems:
a 112-requirement baseline developed with an industrial program, an open prototype implementing the
graph model, the detector catalog, and a severity-based CI/CD gate, and measurements from both the
deployment and the prototype.

The findings we would most want carried forward are the negative ones, plus one that surprised us.
Building the graph and running structural rules is cheap — even the largest of five
air-traffic-management-domain systems we tested (444 components) costs 1.01 s end-to-end, confirmed
across five oracle seeds. Ranking findings by consequence is not, and it gets worse as the same
system grows: rule-based severity agrees with simulated cascade impact at $\kappa = 0.12$ (slight) on
the smallest ATM system we tested and at $\kappa \approx -0.05$ (chance) on the largest — same
domain, same generator template, only scale different. A secondary check across six other synthetic
domains found the same near-chance pattern, while three architectures transcribed from real
open-source systems showed materially better (fair, $\kappa \approx 0.30$) agreement — so scale and
synthetic generation both point toward the same failure mode, and this paper's evidence cannot
cleanly separate which one is the true cause. And of the seven specified checks we have not built,
four are blocked not on analysis technique but on **getting the configuration data into the model at
all** — core assignments, OS settings, and runtime parameters exist in the program and not in the
pipeline's reach. Teams planning similar work should budget for data acquisition, not for graph
algorithms.

**Future work.** Delta-aware gating against the merge base with a waiver register (§4.1); QoS
conformance at reader/writer granularity, which requires extending the schema below topic level
(§3.4); drift detection, which requires the field-record path the specification already describes;
and, highest priority given §5.2's open question, confirming the within-domain $\kappa$ decline
against a real, growing deployment rather than a synthetic scale sweep of one generator template.

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
