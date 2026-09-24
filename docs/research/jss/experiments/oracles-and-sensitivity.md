# Ground-truth oracles, label QoS content, and parameter sensitivity

**Paper:** §4.3 (oracles), §6.3 (population), §8.2 (threats). **Extended results:** Supplement
S1 (explanation-layer sensitivity: OFAT, Morris, $I_\text{comp}$ weights), S2 (zero-inflation),
S3 (domain weighting and thresholds), S4 (AHP matrices), S6 (anti-patterns and stratification),
S8 (HGT attention), S9 (convergent validity).

Simulation oracles run only on the raw structural graph $G_\text{structural}$. Predictors read only
the analysis graph $G_\text{analysis}$, and `tests/test_independence_guarantee.py` enforces the
split. The Predict-stage labeler (`FaultInjector`) and the Validate-stage oracle (`FailureSimulator`)
are never substituted for one another (`tests/test_groundtruth_contract.py`).

## The primary target $I^*(v)$

- **Mechanism.** Crash $v$, propagate by breadth-first traversal through dependent topics, brokers
  and links, and return the mean fractional feed loss over the intact graph's subscriber set.
  - Failed subscribers are retained in the denominator.
  - A topic's loss is the fraction of its publishers that failed; for a topic with no publisher, it
    is the fraction of its routers that failed.
  - Each subscriber contributes the unweighted mean over the topics it subscribes to.
- **QoS ladder.** The loss is scaled by ×1.2 for `RELIABLE`, ×1.15 for high/urgent priority and
  ×1.05 for medium priority, then clamped to [0, 1].
  - The factors are declared constants that encode a severity judgment, not a physical mechanism.
  - The implementation supports per-publisher rate weighting, but rates are declared per topic in
    this corpus, so it reduces exactly to the publisher fraction.
- **Seeds.** Five seeds break ties in propagation order, and $I^*$ is their mean. It is reproducible
  for a fixed seed set but not seed-invariant; see the label-noise ceiling in
  [rq1-engines-loso.md](rq1-engines-loso.md).
- **How much QoS the label carries** (artifact `qos_label_ablation.json`).
  - Disabling the ladder leaves the Application ordering at mean ρ = 0.965 against the ladder.
  - A durability-aware $w(t)$ scaling moves it less still.
  - The top-K set moves more (Jaccard 0.678).
  - $I^*$ is therefore a near-topological target. Oracles that express deadline misses, durability
    replay or priority inversion would let QoS encodings contribute contract semantics.

## Further oracles

- **$I_\text{comp}(v)$** is a severity mixture of reachability loss, fragmentation, throughput loss
  and flow disruption, weighted (0.35, 0.25, 0.25, 0.15).
  - The weights come from a rank-one AHP matrix and are not swept in the main analysis. A
    1,000-draw Dirichlet sweep over them is in S1.2.
  - This oracle labels the explanation layer's evaluation and Validate-stage gates only.
  - Declared topic criticality is masked out of its severity term, because it is a predictor input.
- **$I_\text{dyn}(v)$** is a SimPy discrete-event queue simulation that returns the drop in delivered
  message rate.
  - The Four Golden Signals (latency p50/p95/p99 split into queue wait and service time, traffic,
    errors, saturation) are recorded as diagnostics but deliberately kept out of $I_\text{dyn}$.
  - The reason is contention relief: crashing a high-rate publisher *reduces* surviving consumers'
    queue waits (ρ = −0.499 between delivery loss and tail-latency delta). An additive composite
    would cancel damage against relief.
- **$I_M(v)$** is a reverse `DEPENDS_ON` traversal, used only as a maintainability reference and
  never as a training label.

**Convergent validity** (`make -f reproduce/Makefile convergent-validity`, artifact
`convergent_validity.json`):
- $I_\text{dyn}$ and $I^*$ agree at mean ρ = 0.627 over the twelve folds (top-K Jaccard 0.370
  against 0.111 by chance). That is below $I^*$'s own test–retest of 0.811–1.000.
- Much of the agreement is the two oracles concurring on which components are *harmless*, and
  $I_\text{dyn}$ has its own noise floor.
- Results against one oracle are never transferred to another.

## Evaluation population

Every analysis is scored on one entity type, the Applications. Against $I_\text{comp}$, RM
correlates at ρ = 0.597 on Applications, 0.317 on Brokers and 0.138 on Execution Hosts, but only
0.217 pooled. That is aggregation bias, not a strict Simpson reversal. Across 1,000 Dirichlet draws
of $I_\text{comp}$'s weights, Application ρ exceeds pooled ρ on every draw (S1.2, S6).

## Parameter sensitivity

```bash
make -f reproduce/Makefile topic-weight-sensitivity    # (α, β, γ) split of w(t); artifact topic_weight_sensitivity*.json
make -f reproduce/Makefile weight-global-sensitivity   # joint Morris + Dirichlet over all 10 constants
```

- **Topic-weight split.** Across the full $(\alpha,\beta,\gamma)$ simplex of $w(t)$, orderings hold
  at ρ ≥ 0.919, with ranking changes of 0.031 (`Topo-QoS`) and 0.007 (RM). None of the three is
  influential under Morris ($\mu^* \le 0.025$).
- **Influential constants.** Only the AHP shrinkage λ and the FT/A blend $r_\text{FT}$ matter
  ($\mu^*$ = 0.134 and 0.132); the other eight have $\mu^* \le 0.025$. No setting of the topic-weight
  or QoS sub-weight constants changes any reported comparison.
- **Ranking with $Q(v)$.** If $Q(v)$ is used to rank, a uniform intra-dimension prior (λ = 0) beats
  the elicited weights (0.319 vs 0.200). The default λ = 0.70 is kept because the same constant
  parameterises $I_\text{comp}$.
- **AHP consistency.** Three of the five AHP matrices are rank-one (back-filled from a chosen
  vector), so their consistency ratios certify nothing (S4). The Topic-QoS matrix, $CR = 0.016$, and
  the Fault-Tolerance matrix carry genuine second-eigenvalue spread.
- **Anti-patterns.** A rule-based catalogue flags 93.4% of scored components and does not
  discriminate (S6), so critical-set identification is left to the continuous rankers.
- **Attention.** First-layer HGT attention on the ATM case study ranks `USES` (0.227) above pub-sub
  channels (0.163–0.176). The spread across relation types is narrow and driven by destination
  in-degree (S8).

## Artifacts without provenance stamps

Four supplementary artifacts predate provenance stamping and carry no commit or corpus digest. Their
correspondence to the corpus is asserted by the Zenodo bundle, not recorded in the file:
- `atm_scale_sweep_v3.json`
- `qos_label_ablation.json`
- `threshold_sensitivity_v3.json`
- `topic_weight_sensitivity_v3.json`
