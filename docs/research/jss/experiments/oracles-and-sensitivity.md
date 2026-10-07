# Simulation oracles, label QoS content and parameter sensitivity

**Paper:** §4.3 (`sec:4.3`, the three oracles), §4.4 (`sec:4.4`, input–label separation and the
reference criterion), §7.4 (`sec:threats`, construct validity and oracle circularity).
**Supplement:** §S11 (`supp:convergent`, convergent validity; `supp:istar`, $I^*$ pseudocode), §S3
(`supp:params`, explanation-layer sensitivity; §S3.1 `supp:icomp`, $I_\text{comp}$ weights), §S4
(`supp:zeroinfl`), §S6 (`supp:ahp`), §S8 (`supp:detection`), §S10 (`supp:attention`), §S41
(`supp:advisor-v6`, $I_\text{dyn}$ label reliability).
**Status:** the oracle definitions are fixed by the plan. The robustness analyses are the plan's RQ3
(robustness part) and are reported in the threats to validity and the supplement.

Simulation oracles run only on the raw structural graph $G_\text{structural}$. Rankers read only the
analysis graph, and `tests/test_independence_guarantee.py` enforces the split. The Predict-stage
labeler (`FaultInjector`) and the Validate-stage oracle (`FailureSimulator`) are never substituted
for one another (`tests/test_groundtruth_contract.py`). Procedural separation does not remove
construct overlap, which is what the reference criterion of §4.4 handles
([A13](amendments/a13-reference-demotion.md)).

## The primary oracle $I^*(v)$ (reachability cascade)

- **Mechanism.** Crash $v$ and propagate by breadth-first traversal through dependent topics, brokers
  and links. The label is the mean fractional feed loss over the intact graph's subscriber set.
  - Failed subscribers are retained in the denominator.
  - A topic's loss is the fraction of its publishers that failed; for a topic with no publisher, it
    is the fraction of its routers that failed.
  - A subscriber whose feed loss exceeds the propagation threshold (0.2) fails. It fails with
    probability one in the first wave, and the probability decays by 0.15 per later wave (floor 0.25).
- **QoS ladder.** The loss is scaled by ×1.2 for `RELIABLE`, ×1.15 for high/urgent priority and
  ×1.05 for medium priority, then clamped to [0, 1]. The factors are declared constants that encode
  a severity judgment, not a physical mechanism.
- **Seeds.** The label is the mean over five seeds. It is reproducible for a fixed seed set but not
  seed-invariant: test–retest ρ is 0.811–1.000 (`label_stability.json`).
- **How much QoS the label carries** (`qos_label_ablation.json`). Disabling the ladder leaves the
  Application ordering at mean ρ = 0.965, and durability-aware rescaling moves it less still (0.977).
  $I^*$ is predominantly a topological reachability metric.
- **First-order structure.** Its first wave reaches exactly the Applications `InDeg` counts
  (Remark 1). The first-order expansion (Eq. 6) recovers ρ = 0.808.

## The queue-flow oracle $I_\text{dyn}(v)$

- **Mechanism.** A SimPy discrete-event simulation of message rates, bounded subscriber queues and
  service contention, at target utilization 0.65. The label is the drop in delivered message rate
  for surviving consumers. It reads declared publication rates and QoS contracts (history depth,
  reliability, deadlines, lifespan, priority) that $I^*$ ignores.
- **What it does not model.**
  - Payload: every message has the same size ([A18](amendments/a18-payload-oracle.md)).
  - Brokers and network links.
  - Publisher blocking on full queues, and retries.

  A consumer that loses its inputs keeps publishing, so the oracle propagates no failure beyond one
  hop by construction.
- **Population and cost.** All 1,321 Applications, five seeds, 12.7 CPU-hours
  ([A11](amendments/a11-oracle-robust.md)). The earlier n = 30 lexical sample survives only as a
  sensitivity check (§S40).
- **Noise ceiling.** Two single seeds agree at ρ = 0.43–0.96 per fold. The Spearman–Brown reliability
  of the five-seed label is 0.79–0.99, which caps any ranker at $\sqrt{r}$ ≈ 0.89–0.996. The
  rate-weighted reference (Eq. 7) reaches 0.830 without training ([A15](amendments/a15-rate-expansion.md)).
- **Golden signals stay diagnostic.** Latency, traffic, errors and saturation are recorded but kept
  out of $I_\text{dyn}$. Crashing a high-rate publisher *reduces* surviving consumers' queue waits
  (contention relief), so an additive composite would cancel damage against relief.

## The multi-criteria oracle $I_\text{comp}(v)$

- **Mechanism.** It is the Validate-stage failure simulator:
  $I_\text{comp} = 0.35\,\text{RL} + 0.25\,\text{FR} + 0.25\,\text{TL} + 0.15\,\text{FD}$, over
  reachability loss, fragmentation, throughput loss and flow disruption. The weights are declared,
  not elicited. A 1,000-draw Dirichlet sweep over them is in §S3.1.
- **Never a training label.** Training on the Validate-stage oracle would break the
  `FaultInjector`/`FailureSimulator` separation.
- **Declared topic criticality is masked** out of its severity term, because it is a ranker input.

## Agreement between oracles

- **On the full population,** $I_\text{dyn}$ and $I^*$ agree at mean ρ = 0.711 (§4.3).
- **Convergent validity** (`make -f reproduce/Makefile convergent-validity`, artifact
  `convergent_validity.json`, §S11) was measured on the earlier n = 30 sample: ρ = 0.627, top-K
  Jaccard 0.370 against 0.111 by chance. Much of that agreement is the two oracles concurring on
  which components are *harmless*.
- **Results are never transferred from one oracle to another.** No ranker is best on all three
  (Table 6).

## Evaluation population

Every analysis is scored on Applications. Pooling entity types biases agreement: against
$I_\text{comp}$, the explanation layer's RM score correlates at ρ = 0.597 on Applications, 0.317 on
Brokers and 0.138 on Execution Hosts, but only 0.217 pooled. That is aggregation bias, not a strict
Simpson reversal (§S3.1, §S8).

## Parameter sensitivity

```bash
make -f reproduce/Makefile topic-weight-sensitivity    # (α, β, γ) split of w(t); topic_weight_sensitivity*.json
make -f reproduce/Makefile weight-global-sensitivity   # joint Morris + Dirichlet over all 10 constants
```

- **Topic-weight split.** Across the full $(\alpha,\beta,\gamma)$ simplex of $w(t)$, orderings hold at
  ρ ≥ 0.919, with ranking changes of 0.031 for `Topo-QoS`. None of the three weights is influential
  under Morris ($\mu^* \le 0.025$).
- **Oracle parameters.** Sweeping $I^*$'s threshold θ ∈ {0.1, 0.2, 0.3} × damping step
  ∈ {0.10, 0.15, 0.20} is part of Amendment 7 (`oracle_param_sensitivity.json`,
  [A7](amendments/a07-training-free.md)).
- **Explanation layer (proposed, not evaluated; §S26).** Only the AHP shrinkage λ and the FT/A blend
  $r_\text{FT}$ are influential ($\mu^*$ = 0.134 and 0.132). Three of the five AHP matrices are
  rank-one, back-filled from a chosen vector, so their consistency ratios certify nothing (§S6). The
  rule-based anti-pattern catalogue flags 93.4% of scored components and does not discriminate (§S8).
- **Attention.** First-layer HGT attention on the ATM case study ranks `USES` (0.227) above pub-sub
  channels (0.163–0.176). The spread is narrow and driven by destination in-degree (§S10).

## Artifacts without provenance stamps

Four supplementary artifacts predate provenance stamping and carry no commit or corpus digest. Their
correspondence to the corpus is asserted by the Zenodo bundle, not recorded in the file:
- `atm_scale_sweep_v3.json`
- `qos_label_ablation.json`
- `threshold_sensitivity_v3.json`
- `topic_weight_sensitivity_v3.json`
