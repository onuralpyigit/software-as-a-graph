# Amendment 15: the rate-weighted reference for $I_\text{dyn}$ (Eq. 7)

**Paper:** §5.2 (`sec:6.2`, Eq. 7 `eq:rate-expansion` and the reference block of
`tab:predictor_taxonomy`), §6.1 (`sec:rq1`, Table 7 `tab:independent_oracles`: the "Rate-weighted"
row; "What declared rates add on the queue-flow oracle"), §6.4 (cost), §7.3 (`tab:guidance`), Figure 5B.
**Supplement:** §S42 (`supp:advisor-v6`; Tables S67–S69: `tab:a15-folds`, `tab:a15-zeroshot`,
`tab:a15-contrasts`).
**Status:** exploratory. Post hoc: written 2026-09-29, after every Amendment 11 result was
published. The headline value (ρ = 0.830) was first computed ad hoc during the manuscript revision,
before this record existed. No registered arm is re-run, dropped or changed.
**Registration:** [`../../PREREGISTRATION.md`](../../PREREGISTRATION.md), Amendment 15 and its
clarifications (commit `2c5275f2`).

## Question

Does `GBM-P-QoS→dyn` (Amendment 11, 0.799 on $I_\text{dyn}$) beat the closed forms because it
learns? Or does it win because its Q columns carry the declared publication rates that $I_\text{dyn}$
reads and Analytic $I^*$ ignores? A fair comparison gives the closed form the same inputs.

## Quantities

- **Eq. 7.** $\hat I^\text{rate}_{\text{dyn},1}(v) = \sum_{t\in\text{pub}(v)} \frac{r_t}{|\text{pub}(t)|}\,|\text{sub}(t)|$,
  where $r_t$ is topic $t$'s declared publication rate. It weights each term of Analytic $I^*$ (Eq. 6)
  by $r_t$, which truncates $I_\text{dyn}$ to direct delivered-message loss. Under the reference
  criterion (§4.4, Amendment 13) it is a *reference* for $I_\text{dyn}$, not a predictor.
- **Variants:** rate × payload, and the bare publication rate of $v$.
- **Attribution arms.** Amendment 11's learner, seeds and LOSO protocol, trained on $I_\text{dyn}$,
  with feature set S plus:
  - (a) nothing;
  - (b) the rate and rate × payload columns;
  - (c) the seven QoS-derived columns;
  - (d) all of Q. This arm must reproduce `gbm_dep_qos_dyn` (gate G_A15).

  Three of the seven columns in (c) are $w(t)$-weighted scores, in which rate and size enter
  log-compressed at a quarter of the weight. The other four are pure policy shares.

## Reproduce

```bash
make -f reproduce/Makefile rq-rate-expansion    # reproduce/idyn_rate_expansion.py
```

It runs no simulator: it reads the Amendment 11 label caches.

## Artifact

`data/benchmarks/idyn_rate_expansion.json`. It is checked by
`reconcile_manuscript.py::check_rate_expansion`, which covers Table 7's Eq. 7 row, all three
supplement tables and the gates.

## Outcome

- **Gates.** Analytic $I^*$ reproduces Amendment 11's comparator on every fold, system and oracle
  (max |Δ| = 0). Arm (d) gives 0.799, the published value.
- **Eq. 7 on $I_\text{dyn}$:** LOSO ρ = 0.830 [0.778, 0.872]; zero-shot 0.893. On $I^*$ it reaches
  0.756, and on $I_\text{comp}$ 0.551. Computed in ≤ 1.2 ms per architecture.
- **Eq. 7 vs `GBM-P-QoS→dyn`:** +0.031 [+0.013, +0.049], 10/12 folds, Holm p = 0.009 (nominal) under
  LOSO; +0.093 on 5/5 systems zero-shot (Holm p = 0.19). The paper reads the LOSO margin as "the
  learned approximation does not exceed the formula", not as superiority, because it is comparable
  to the node-order spread.
- **Variants:** rate × payload 0.748; bare publication rate 0.786.
- **Attribution (LOSO, $I_\text{dyn}$):** S alone 0.704. Adding rate and payload gives 0.801
  (+0.097, 9/12, Holm 0.021). Adding the QoS-derived columns gives 0.700 (−0.005). The Amendment 11
  Family C gain is carried entirely by the rate columns.
- **Label reliability.** This was added in the clarifications. Two single $I_\text{dyn}$ seeds agree
  at 0.431–0.964 per fold. The Spearman–Brown projection to the five-seed label is 0.791–0.993, which
  caps any ranker at $\sqrt{r}$ ≈ 0.89–0.996.

## Corrections

- **Payload carries no signal.** Amendment 18 ([page](a18-payload-oracle.md)) found that the published
  oracle never reads payload size. The +0.097 attribution is therefore rate signal only, and the
  paper says "declared rates", not "rates and payload sizes".
- **Learning on top of Eq. 7.** Learners given Eq. 7 as an input or prior do not improve on it.
  This was tested by Amendment 17, F12 ([page](a17-round12.md)).
