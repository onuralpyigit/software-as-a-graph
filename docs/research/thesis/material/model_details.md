# Model details cut from the JSS condensation: QoS weights, SCA ingestion, layer table

> **Provenance.** Verbatim from `docs/research/jss/draft.md` §3.2, §3.4, and the Table 6 portion of
> §3.5, as they stood at commit `f0cba41822820a79ebdab123d54a76072b8f1689` (the paper's authoritative
> pre-condensation text, also snapshotted in full at `../jss_draft_full.md`). The condensed JSS
> `draft.md` replaces §3.2's Tables 3–4 with two sentences of prose, drops §3.4 entirely, and
> compresses §3.5's Table 6 to one sentence naming the four layers. Nothing here has been reworded.

---

## 3.2 QoS-Aware Edge and Vertex Weights

Not all dependencies are equally consequential: a `RELIABLE`/`PERSISTENT` channel carrying critical
data couples its endpoints far more tightly than a `BEST_EFFORT`/`VOLATILE` one. Edge weights encode
this from the Quality-of-Service policy of each pub-sub relationship, via a two-stage computation:

$$\text{QoS\_score} = 0.30\,r + 0.40\,d + 0.30\,p,$$
$$\text{size\_norm} = \min\!\left(\frac{\log_2(1 + \text{size\_kb})}{50},\ 1.0\right),$$
$$w(e) = \beta\cdot\text{QoS\_score} + (1-\beta)\cdot\text{size\_norm}, \qquad \beta = 0.85,$$

where $r, d, p$ are the reliability, durability, and transport-priority scores of the mediating
topic, mapped from symbolic QoS values:

**Table 3. QoS symbolic-value to numeric-score mapping** used by the edge weight of §3.2.

| Dimension | Symbolic value → score |
|-----------|------------------------|
| Reliability $r$ | `RELIABLE` → 1.0; `BEST_EFFORT` → 0.0 |
| Durability $d$ | `PERSISTENT` → 1.0; `TRANSIENT` → 0.6; `TRANSIENT_LOCAL` → 0.5; `VOLATILE` → 0.0 |
| Priority $p$ | `URGENT`/`CRITICAL`/`HIGHEST` → 1.0; `HIGH` → 0.66; `MEDIUM` → 0.33; `LOW` → 0.0 |

The intra-QoS sub-weights are stated judgements checked for AHP consistency (§4.3): durability
(0.40) outweighs reliability and priority
(0.30 each) because durability governs message-state survival — the precondition for resilience —
whereas reliability and priority govern transient delivery quality. A floor of $w(e) = 0.01$ keeps
even zero-QoS components visible to attribution.

**Vertex weights** propagate QoS upward from incident edges, with type-specific aggregation that
reflects how each component type concentrates risk:

**Table 4. Type-specific vertex weight aggregation rules.**

| Type | $w_V$ |
|------|-------|
| Application | $0.80\cdot\max(w_{\text{topic}}) + 0.20\cdot\operatorname{mean}(w_{\text{topic}})$ |
| Broker | $0.70\cdot\max(w_{\text{topic}}) + 0.30\cdot\operatorname{mean}(w_{\text{topic}})$ |
| Node | $\max(w)$ over all hosted applications and brokers |
| Library | $\min\!\big(1.0,\ w_{\text{base}}\cdot(1 + \gamma\log_2(1 + \mathrm{DG\_in}))\big)$ (fan-out amplified) |

The library rule is deliberately fan-out amplified: a library's risk grows with the number of
applications that depend on it, anticipating the blast-radius mechanism of §3.3 and §5.

---

## 3.4 Ingestion of Code-Level SCA Metrics

To bridge the "Architecture-Code Gap," SaG does not operate in isolation from source code. Instead,
the framework integrates code-level quality attributes directly into the graph model. During the
model-import stage, SaG queries static code analysis (SCA) APIs (e.g., SonarQube's web API) or
parses local SCA report artifacts to extract modular metrics for executable `Application` and shared
`Library` components.

These metrics are stored as flat properties prefixed with `cm_*` on each component node:
- `cm_total_loc`: Total lines of code as reported by static analysis, providing a scale proxy.
- `cm_avg_wmc`: Average Weighted Methods per Class, representing cognitive complexity.
- `cm_avg_lcom`: Lack of Cohesion of Methods (on a raw [0, 1] scale), indicating how fragmented
  classes are.
- `cm_avg_cbo`: Coupling Between Objects, indicating intra-component code coupling.
- `cm_avg_rfc`: Response for a Class, measuring the number of methods invoked by a class.
- `sqale_debt_ratio`: Technical debt ratio as a percentage of estimated rewrite time.
- `bugs`: Count of static bugs identified in code.
- `vulnerabilities`: Count of code-level security issues.

These properties are normalized across the component population during structural analysis (§4.2)
and feed the **Code Quality Penalty (CQP)**, ensuring that local code defects are mathematically
combined with global structural dependencies.

---

## 3.5 (excerpt) Layer projection table

$G_{\text{analysis}}$ is filtered into four analytical layers, each isolating a component scope, a
dependency subset, and the quality dimension it most informs:

**Table 6. The four analytical layer projections** and the quality dimension each most informs.

| Layer | Projection | Vertices | Dependency types | Quality focus |
|-------|-----------|----------|------------------|---------------|
| Application | $\pi_{\text{app}}$ | App, Library | `app_to_app`, `app_to_lib` | Reliability |
| Infrastructure | $\pi_{\text{infra}}$ | Node | `node_to_node` | Availability |
| Middleware | $\pi_{\text{mw}}$ | Broker (in App/Node context) | `app_to_broker`, `node_to_broker`, `broker_to_broker` | Maintainability |
| System | $\pi_{\text{system}}$ | all five types | all six | Overall |

The middleware layer includes Application and Node vertices in the subgraph to preserve incoming
edges, but reports results only for Brokers. Components further aggregate along a MIL-STD-498 [52]
hierarchy — CSU → CSC → CSCI → CSS — so that criticality can be rolled up from a unit to a
configuration item to the whole system, supporting reporting at whatever granularity an
organization's software configuration management already uses.
