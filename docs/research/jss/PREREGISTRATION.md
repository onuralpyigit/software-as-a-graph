# Pre-registration — HGL vs Topo-QoS under LOSO

Written **before** any number from the revised harness exists. Registered
2026-09-06. Anything not listed as primary or secondary below is exploratory and
must be reported as such.

## Motivation

`results/loso_all_variants.json` (the pre-revision artifact) reported HGT-QoS
ρ = 0.608 against the training-free Topo-QoS at 0.571 — +0.037 on 5/8 folds,
two-sided Wilcoxon p = 0.64. That artifact was subsequently shown to reproduce
from no commit in the repository and has been withdrawn; it is recorded here
only because it is what motivated this pass. The question it raised stands:
can a heterogeneous model beat Topo-QoS under LOSO by a margin that survives a
signed-rank test?

**Outcome (recorded after the fact, 2026-09-09; restated 2026-09-16).** No.
The 2026-09-09 run measured +0.127 (9/12, W = 16.0, p = 0.077) and was recorded
here as resting almost entirely on the ATM fold. That run has since been
superseded. The reported result is the twelve-fold re-run in
`results/loso_all_variants_v5.json` and `results/loso_significance_v5.json`:
**+0.085 (9/12, W = 20.0, p = 0.151, CI [-0.029, +0.194])**. The conclusion is
unchanged — the margin does not reach significance — but two details of the
earlier record no longer hold. The ATM dependency is gone: ATM contributes
+0.210, behind Healthcare (+0.401) and IoT Smart City (+0.360). The three
losing folds are Enterprise (-0.334), Telecom RAN (-0.169) and AV (-0.030);
Microservices, recorded as a loss in earlier drafts, is a +0.229 win. Reported
as registered, in Section 7.1 of the manuscript. The secondary contrast and the
typing comparisons are reported in the same section and in Section 7.2.

## Primary comparison

- **Contrast:** `hgl_qos` (HGT-QoS) vs `topo_qos` (Topo-QoS).
- **Protocol:** leave-one-scenario-out, `cli/loso_evaluate.py`.
- **Population:** `--eval-population application` (Application stratum only).
- **Statistic:** per-fold mean Spearman ρ over seeds `42,123,456,789,2024`;
  two-sided Wilcoxon signed-rank over folds; α = 0.05.
- **Unit of analysis:** the fold. Not the seed, and not the node. Pairing at
  (fold × seed) would be pseudo-replication and is forbidden.

## Secondary comparison

- `hgl` (HGT) vs `topo_qos`, identical protocol.
- Holm correction across the primary and secondary tests.

## Attainable-p floor

The signed-rank statistic is discrete. Report the floor alongside every p:

| n folds | best attainable two-sided p | 1 small loss | 2 small losses | 3 small losses |
|---:|---:|---:|---:|---:|
| 8 | 0.0078 | 0.0156 | 0.0391 | 0.1094 |
| 12 | 0.0005 | 0.0010 | 0.0024 | 0.0068 |

"Small loss" means the losing fold is among the smallest |Δρ| in the set. A
large loss costs far more rank mass than the table suggests.

## Selection rule

Every hyperparameter and representation switch is selected by **inner-LOSO mean
ρ over the N−1 training scenarios of the outer fold**. The outer holdout takes
no part in selection — not in early stopping, not in checkpoint choice, not in
config choice. `reproduce/nested_loso_search.py` asserts this.

Two classes of change are admissible outside the nested search, and only these:

1. **Harness defects** with a mechanism argued independently of the outer
   metric — the per-fold depth inconsistency, validation on the training
   distribution, and the GAT/HGT training-set asymmetry.
2. **Representation switches motivated by a diagnostic that predates any outer
   result** — rank normalization, motivated by the drift measured in
   `results/feature_shift_diagnostic.md` (mpci 115.7×, qos_weight_in 15.5×,
   betweenness 12.3×).

Both classes are still passed through the nested search where they have a free
parameter; the point is that their *motivation* is not the outer number.

## Reporting commitment

The result is reported as measured. If the margin does not reach significance,
the finding is that a training-free QoS-weighted structural score is competitive
with heterogeneous graph learning for cross-domain criticality ranking. That is
a publishable result, not a failed pass, and it will not be re-run with a
different configuration in search of a different answer.

---

## Amendment 1 — inner selection budget (2026-09-06, before any outer result)

**Status when written:** the 12-fold regeneration sweep was still running and no
outer ρ under the revised harness existed yet. The original artifact
(`results/loso_all_variants.json`) had already been shown not to reproduce from
any commit, so there was no outer number to tune toward even accidentally.

**Original rule:** configurations selected by inner-**LOSO** mean ρ over the N−1
training scenarios of each outer fold.

**Amended rule:** configurations selected by inner **holdout** — each
configuration is scored on `k = 2` inner folds (deterministically chosen, evenly
spaced by size, via `_inner_indices`) rather than all 11.

**Reason:** measured cost. The sweep paces at ~40 s per model fit, so full
inner-LOSO is 12 × 8 × 11 = 1,056 fits ≈ 12 hours for one variant's search. The
inner loop's job is to *rank* configurations, not to estimate their performance;
a 2-fold inner sample is a coarser ranking signal but a sufficient one, at 192
fits ≈ 2 hours.

**What is unchanged:** the outer holdout still takes no part in selection — it is
not in the inner set at all, which `reproduce/nested_loso_search.py` asserts per
outer fold. Inner runs use the same 300-epoch budget as the outer folds, so the
configurations are ranked under the budget they will be reported under.

**Cost of the amendment:** selection is noisier, so `c*(k)` may vary more across
outer folds than it would under inner-LOSO. That variation is reported per fold
rather than hidden, and it is itself a finding: a configuration that cannot be
picked stably from the training set is not one to recommend.

---

## Amendment 2 — RQ2 confound controls (2026-09-12, before any control result)

**Status when written:** the four control arms below have been implemented and
smoke-tested for capacity only (3 epochs, 3 folds, CPU — enough to confirm each
arm builds and trains at its intended parameter count, far too little to produce
a ρ worth reading). No control arm has been run at the reported budget on any
fold, so no outcome was known when this amendment was registered. The decision
rule in the last section is therefore a commitment, not a description.

**Why this amendment exists.** Section 7.2 attributes the typed-vs-untyped LOSO
margin (+0.114, 11/12 folds, p = 0.0122) to "relational typing rather than to
substrate, training set, depth, or selection rule". An internal audit found that
list incomplete. Three factors the comparison did not hold constant:

| Factor | As published |
|:---|:---|
| Parameter budget | HGT 434,620 vs GAT-N-QoS 28,168 — a 15.4× gap |
| Message-passing directionality | HGT carries a reverse HGTConv (103,725 params, 24% of the model); the homogeneous GAT propagates along native edge direction only, and `I*(v)` is a downstream-reachability functional |
| Edge-channel width | HGT-QoS reads all 16 edge-feature dimensions; GAT-N-QoS reads dimension 0 alone |

A fourth concerns the label rather than the model: `I*`'s QoS ladder keys on
reliability and transport priority, which are edge-feature dimensions 9 and 11 —
given only to the `-QoS` arms. The QoS ablation therefore compares a predictor
that can see the oracle's severity multipliers against one that cannot.

**These are post-hoc.** They were not part of the original pre-registration and
must be reported as exploratory throughout. Their Wilcoxon contrasts are
Holm-corrected within their own family, never pooled with the primary contrast.

**The arms** (`saag/evaluation/variant_registry.py`, family `control`):

| Arm | Label | Isolates | Parameters |
|:---|:---|:---|---:|
| `gl_full_qos_cap` | GAT-N-QoS-C | capacity | 439,272 (1.011×) |
| `gl_full_cap` | GAT-N-C | capacity, QoS-off replication | 437,496 (1.007×) |
| `gl_full_qos16_cap` | GAT-N-QoS16-C | edge-channel width | 429,992 (0.989×) |
| `hgl_qos_uni` | HGT-QoS-U | directionality | 330,895 |

Plus a label-side arm: the full LOSO sweep repeated against a cache built with
`QOS_FACTOR=none`, so the oracle's QoS ladder is disabled.

The edge-channel arm deliberately receives the 7-dimensional relation one-hot
(dims 2–8) along with the QoS block. That hands the untyped model relation type
*as a feature*, which is the strict form of the control: it separates typing as
an input signal from typing as relation-specific parameters. That distinction
must be stated wherever the arm is reported.

### Decision rule, committed before the results exist

| Outcome | What we will report |
|:---|:---|
| Margin survives all four (shrinks < 0.02, ≥ 10/12 folds, p < 0.05) | Section 7.2 is strengthened: the attribution sentence is rewritten to name capacity, directionality, edge-channel width and label overlap explicitly, with the controls table. |
| Margin shrinks but holds significance | The **controlled** margin becomes the headline in the abstract, Section 1, Section 7.2, Section 8 and Section 9. The uncontrolled +0.114 is reported alongside it, labelled as the naive comparison, with the parameter ratio that accounts for the difference. |
| Margin does not survive | Section 7.2's claim becomes "typed message passing does not outperform a capacity-matched untyped GAT under distribution shift". The abstract's +0.114 is withdrawn. The contribution shifts to per-relation edge criticalities and attention over heterogeneous schemas, which have no untyped counterpart — the argument Section 8.1 already makes as a secondary one. |
| Mixed (e.g. capacity survives, directionality does not) | Reported per confound in the controls table, with the claim narrowed to the factors actually controlled. Results are **not** aggregated into a single verdict. |

**Reporting commitment.** Every arm that is run is reported, whichever way it
comes out, with its parameter count in the table. An arm may be dropped only for
a stated technical failure, never for its result. Parameter counts belong in the
table rather than the prose: their absence is what allowed this confound to
survive internal review in the first place.

**Corpus and device.** The controls are run against the corpus committed on
2026-09-10 and are compared only against arms re-baselined on the same corpus
and the same device. Rows measured on CPU and rows measured on GPU are never
placed in the same comparison; `config.device` in each artifact records which.

---

## Amendment 3 — re-baseline, and a withdrawal (2026-09-13, before any v5 result)

**Status when written:** the sweep this amendment governs has not been run. No
number below is an outcome.

### What prompted it

`reproduce/reconcile_manuscript.py` reports **53 mismatches** between the
manuscript's Tables 7/7c/9b and the artifacts declared to back them. The cause
is artifact drift, not a reporting slip: the manuscript's LOSO figures match
`results/loso_all_variants_v3.json` (unstamped) to three decimals, while the
declared backing artifact is `results/loso_all_variants_v4.json`, stamped clean
at commit `99f2f45` against the current corpus digest. The v3→v4 gap is most
plausibly the 2026-09-12 corpus regeneration and `output/loso_cache/` rebuild.

The two runs do not merely differ in the third decimal. Under v4 the
typed-vs-untyped margin is **+0.011** (HGT-QoS 0.6216 vs GAT-N-QoS 0.6109),
against the published +0.114; HGT-QoS no longer leads on `F1@K`; and the
real-world active-stratum inversion on Cloud Microservices reverses sign. Under
v4's numbers the architecture contrasts run through `loso_significance.py` give
typing-with-QoS p = 0.791 and typing-unweighted p = 0.0024 — the opposite
pattern to the one published.

Neither v3 nor v4 is adopted. Both predate this amendment's tooling changes and
v4 is nine commits stale, so the sweep is re-run at a clean HEAD as **v5**, and
v5 is what the manuscript will report.

### Withdrawal

Section 7.2.1 as committed reports a capacity control arm — GAT-N-QoS-C at
439,041 parameters, ρ = 0.589, CI [0.501, 0.665], against GAT-N-QoS at 110,145
parameters — and calls it "definitive empirical evidence". **No artifact backs
any of those five numbers.** Neither v3 nor v4 contains a control arm;
`results/table_rq2_controls.md`, which `render_table.py` would emit, does not
exist; and the parameter counts contradict the values pinned in
`tests/test_baselines.py::TestControlArmCapacityParity` (HGT-QoS 434,620,
GAT-N-QoS 28,168). The paragraph entered the repository in a docs-only commit
(`f85271a`) one day after Amendment 2 recorded that no control arm had been run.

That paragraph and the three-arm announcement in Section 6.2.2 are withdrawn in
full. Amendment 2's reporting commitment stands unchanged: the arms are now
being run for the first time, and whatever they produce is what gets reported.

Section 6.2.2's announcement also mislabelled arm (ii) — 429,992 is
`gl_full_qos16_cap`, an *edge-channel* control. No bidirectional-homogeneous arm
exists in the codebase (`build_baseline` has no directionality parameter), and
none is claimed.

### New arms and tests, all post-hoc

| Addition | Purpose | Pre-registered? |
|:---|:---|:---|
| `tab_gbm` (GBM-Feat) | Gradient boosting on the identical typed node features, no message passing. Isolates whether any graph-learning margin is the aggregation or just the features — the features already contain betweenness, closeness, reverse PageRank and articulation scores, and `I*(v)` is a reachability functional over the same topology. | No |
| `ARCHITECTURE_CONTRASTS` in `loso_significance.py` | RQ2 (typed vs untyped) and RQ3 (QoS edge ablation) are variant-vs-variant comparisons, not comparisons against `topo_qos`. The exploratory block only ever paired against the baseline, so the p-values the manuscript reports for both — 0.0122 and 0.0093 — appear in **no** significance artifact under any version. Holm-corrected within their own family. | No |
| Bootstrap CI on every contrast's fold deltas | The LOSO table has never carried one; `EXPERIMENTS.md` §2.D's claim of bootstrap CIs is true only of the in-distribution table. | No |
| `broker` / `library` / `topic` / `node` evaluation populations | Section 1.3 argues message passing scores entity types no fault-injection sweep was configured for. That claim is only testable against a stratum the model was not trained to rank, and no current artifact scores one. | No |

### What is unchanged

The primary and secondary pre-registered contrasts, their unit of analysis (the
fold), the five fixed seeds, and the prohibition on pairing at (fold × seed).
Amendment 2's decision rule governs the control arms exactly as written.

### Reporting commitment

v5 is reported whichever way it comes out. If the typed-vs-untyped margin does
not survive the re-baseline, Amendment 2's "margin does not survive" row applies
and the abstract's +0.114 is withdrawn — with the added constraint, from this
amendment, that the replacement claim must name the artifact it reproduces from.
An artifact that does not reproduce from a commit is not reportable, which is
the rule v3 failed and the reason this amendment exists.

---

## Amendment 4 — peer-review revision (2026-09-20, after the v5 results existed)

**Status when written:** the twelve-fold v5 results and the manuscript built on
them existed. Everything recorded here is therefore a *post-hoc* change made in
response to reviewer comments, and nothing in it is pre-registered. It is written
down for the same reason the earlier amendments are: so that the difference
between what was planned and what was added afterwards stays legible.

### Terminology correction

This file was described in the manuscript as a *pre-registration*. It is not one
in the sense a reader would infer: it lives in the authors' own repository, has
no third-party timestamp, and — as its own Motivation section records — a prior
eight-fold run of the primary contrast predates it. Section 6.3 now calls it a
*registered analysis plan* and states both limitations. What the registration
establishes is that the analysis was fixed before the twelve-fold result existed;
what it cannot establish is that the question was asked in ignorance of any
earlier estimate.

### RQ4 protocol re-match

The real-world arm had been reported from a 2-layer, 150-epoch configuration
chosen to limit over-smoothing on the transcribed meshes. No target label or
gradient reached the model, but the *reasoning* appeals to a property of the
evaluation systems, so the arm was not blind to them. It is replaced as the
primary RQ4 result by a re-run at the 3-layer, 300-epoch budget used for every
other learned result in this paper (`results/realworld_zeroshot_v7.json`). The
matched protocol is uniformly slightly weaker (ρ 0.760 vs 0.792; ρ₊ +0.236 vs
+0.281; F1@K 0.470 vs 0.533) and changes no qualitative conclusion. The earlier
configuration is retained as a reported sensitivity, not as a headline.

### Analyses added after the fact, all exploratory

| Addition | Why | Pre-registered? |
|:---|:---|:---|
| Fisher-z recomputation of the 2×2 (`loso_significance.py --fisher-z`) | A difference of differences on bounded Spearman ρ can read as sub-additive through scale compression alone. The interaction survives and grows (−0.199 → −0.233), so the substitution claim is a property of the mechanisms, not the metric. | No |
| Bootstrap CIs over the five real systems | A five-point mean was being reported as if its error were negligible. The active-stratum interval spans zero for every predictor, which changes the RQ4 verdict from "qualified positive" to "unresolved". | No |
| Per-scenario gate:oracle ratio table | The cost claim was stated at the joint maximum (≈18×). The distribution is 2.0–17.7×, median 5.6×, and the premium tracks derived projection size (ρ = 0.95) rather than component count. | No |
| Revision-drift ledger (`reproduce/rerun_drift.py`) | Section 8.3 carried drift figures that reproduce from no committed artifact pair. They are replaced by a measured v4→v5 comparison at identical corpus digest. | No |
| Energy upper bound (`reproduce/energy_estimate.py`) | The sustainability argument was unquantified. Nameplate TDP × measured wall-clock is an upper bound, labelled as such, not a RAPL measurement. | No |

### What is unchanged

The primary and secondary contrasts, their unit of analysis, the five fixed
seeds, the prohibition on pairing at (fold × seed), and Amendment 2's decision
rule for the control arms — which remain implemented, registered, and unrun.

---

## Amendment 5 — hybrid engine (2026-09-23, before any hybrid result)

**Status when written:** the hybrid variant is not yet implemented. No hybrid
outcome exists, so the design and decision rule below are commitments, not
descriptions.

**Why it exists.** HGT-QoS's two largest LOSO losses to Topo-QoS (Enterprise,
Telecom RAN) fall on folds where the closed-form score is strongest, which
suggests the learned engine discards structural signal the closed-form engine
keeps. The hybrid gives the learned engine that signal explicitly and asks it
to learn only a correction to it.

**Design, fixed before any run. No tuning and no search.**

| Element | Choice |
|:---|:---|
| Variant id / label | `hgl_qos_prior` / SaG-Hybrid |
| Base model | HGT-QoS exactly as reported: 3 layers, D = 64, H = 4, 16-D QoS edge channel, bidirectional, same optimiser, schedule, loss (Eq. 5), epochs (300), early stopping |
| Prior $p(v)$ | The LOSO-path Topo-QoS score of each Application and Library, computed on each graph (training, validation and held-out alike) by the same code that produces the published Topo-QoS baseline, then rank-normalised to $[0, 1]$ within the graph (average ranks for ties). Every other entity type gets $p = 0$. |
| Input | $p(v)$ appended as one extra node-feature column, after feature rank normalisation |
| Output | $\hat{I}^*(v) = \sigma\big(z(v) + \alpha \cdot \operatorname{logit}(\operatorname{clip}(p(v), 0.01, 0.99))\big)$ for Applications and Libraries, where $z$ is HGT-QoS's composite-head logit and $\alpha$ is one learnable scalar initialised to 1.0 |

**Run.** One CPU invocation (`--device cpu`), 12 LOSO folds × 5 seeds
{42, 123, 456, 789, 2024}, Application population, containing `topo_baseline`,
`topo_qos`, `hgl_qos` and `hgl_qos_prior`. The rows are never compared with
the v5 GPU rows (Amendment 2, "Corpus and device").

**Contrasts.** Two-sided Wilcoxon signed-rank over folds, with Holm correction
across these two only:
- **Primary:** `hgl_qos_prior` vs `topo_qos`.
- **Secondary:** `hgl_qos_prior` vs `hgl_qos`.

The same model is also evaluated zero-shot on the five open-source system
models, under the protocol of `results/realworld_zeroshot_v7.json`.
That evaluation is descriptive, with bootstrap intervals and no test.

### Decision rule

| Outcome | What we will report |
|:---|:---|
| Primary contrast significant (Holm p < 0.05) | SaG-Hybrid becomes the headline learned engine in the abstract and §7, with this amendment cited. |
| Not significant | SaG-Hybrid is reported in its own subsection with the same numbers. HGT-QoS remains the headline learned engine, and no claim of superiority over Topo-QoS is made. |

**Reporting commitment.** The hybrid is reported whichever way it comes out.
It may be dropped only for a stated technical failure.

**Also recorded here:** the published Topo and Topo-QoS baselines never used
their articulation-point term. The cached `structural_metrics.json` carries no
`ap_c_score`, so `reproduce/main_table._parse_structural_metrics` sets it to 0
for every node. The registered comparator is left exactly as it ran. An
AP-corrected version is computed separately as a sensitivity analysis.

---

## Amendment 6 — hybrid on the untyped QoS engine (2026-09-24, before any result)

**Status when written:** the variant is not yet implemented. No outcome exists.

**Why it exists.** The matched control (Amendment 2) showed that relation-typed
weights add nothing at matched capacity. It also showed that the capacity-matched
untyped GAT with the 16-D QoS channel (`gl_full_qos16_cap`) transfers best
zero-shot. Amendment 5's hybrid was built on HGT. This amendment asks whether
the same correction, applied to the untyped engine, keeps the hybrid's LOSO
gain and also keeps the untyped engine's transfer.

**Design, fixed before any run. No tuning and no search.**

| Element | Choice |
|:---|:---|
| Variant id / label | `gl_qos16_prior` / SaG-Hybrid-GAT |
| Base model | `gl_full_qos16_cap` exactly as run in Amendment 2's sweep: untyped GAT, 288 hidden channels, 4 heads, 3 layers, 16-D edge channel, same trainer, loss, epochs (300) and early stopping |
| Prior | Identical to Amendment 5: the LOSO-path Topo-QoS score of each Application and Library, rank-normalised within the graph (average ranks); 0 for other types; appended as the last node-feature column |
| Output | $\hat{I}^*(v) = \sigma\big(z(v) + \alpha \cdot \operatorname{logit}(\operatorname{clip}(p(v), 0.01, 0.99))\big)$ for Applications and Libraries, with one learnable $\alpha$ initialised to 1.0 |

**Run.** One CPU invocation, 12 LOSO folds × 5 seeds
{42, 123, 456, 789, 2024}, Application population, containing `topo_qos`,
`gl_full_qos16_cap` and `gl_qos16_prior`. The same model is also evaluated
zero-shot on the five open-source system models under the Table 12 protocol.

**Contrasts.** Two-sided Wilcoxon over folds, with Holm correction across these
two only. This family is separate from Amendment 5's.
- **Primary:** `gl_qos16_prior` vs `topo_qos`.
- **Secondary:** `gl_qos16_prior` vs `gl_full_qos16_cap`.

The comparison with SaG-Hybrid (`hgl_qos_prior`) is descriptive only; its
Topo-QoS and HGT-QoS comparators are bit-identical across CPU sweeps.

### Decision rule

| Outcome | What we will report |
|:---|:---|
| Primary significant (Holm p < 0.05) | SaG-Hybrid-GAT is reported as a second engine that significantly outperforms closed-form ranking. It replaces SaG-Hybrid as the recommended hybrid only if it is also at least as good zero-shot (mean ρ on the five system models ≥ SaG-Hybrid's 0.695). |
| Not significant | Reported in the hybrid section with the same numbers. SaG-Hybrid remains the headline hybrid. |

**Reporting commitment.** Reported whichever way it comes out, in the manuscript
and supplement. It may be dropped only for a stated technical failure.

---

## Amendment 7 — training-free baselines and QoS-attribution controls (2026-09-25, before any result)

**Status when written:** none of the arms below has been implemented or run. The
per-fold values they will be compared against are already published (Supplementary
S22 for `Topo`/`Topo-QoS`, S23 for the CPU learned and hybrid engines), so this
amendment fixes the arms, the comparisons and what each outcome changes in the text
before any of the new numbers exist.

**Why it exists.** The referee report of 2026-09-25
(`docs/research/jss/reviews/review_2026-09-25.md`, M2, M3, M6, M7) raised three
questions that need no GNN to answer:
1. Does a trivial reachability or connectivity score, computed on the same
   projection, already match the learned engines? `I*(v)` is a cascade over the same
   dependency rules the projection encodes.
2. Is the `Topo` → `Topo-QoS` gain (+0.204) produced by the *content* of the declared
   QoS contracts, or by the *multiplicity* of shared topics that the probabilistic
   union rewards regardless of content? The generator also samples topology
   conditioned on QoS (`_APP_TYPE_QOS_AFFINITY`), which could couple the two.
3. How much of every full-population correlation is only the separation of inert
   components (`I* = 0`) from active ones?

**Labels.** `I*(v)` regenerated with the published settings: `FaultInjector`, seeds
{42, 123, 456, 789, 2024}, propagation threshold 0.2, unlimited cascade depth, QoS
ladder, node types Application/Broker/Library. Twelve LOSO scenarios and the five
system models. Scored on the Application population with the shared
`saag.evaluation.metrics.compute_inductive_metrics`.

**Reproduction gate.** Before anything else is reported, the rebuilt `Topo-QoS` must
reproduce the published per-fold values (Supplementary S22, "as run") to three
decimals. If it does not, nothing from this amendment is reported except the failure.

### Arms, fixed before any run

All arms are training-free and run on the Application–Library `DEPENDS_ON`
projection `Topo-QoS` uses (Rules 1 and 5), edges directed dependent → dependency.

| Arm | Score for component v |
|:---|:---|
| `Reach` | Number of transitive dependents of v (`nx.ancestors`), normalised by n − 1 |
| `Reach-QoS` | Sum over transitive dependents u of the best-path product of edge `qos_weight` from u to v |
| `CDI` | Connectivity Degradation Index alone, from `StructuralAnalyzer._compute_continuous_ap_scores` on the projection |
| `InDeg` | Number of direct dependents (in-degree on the projection) |
| `Topo-Mult` | `Topo-QoS` with every topic weight set to the constant 0.5, so that a Rule-1 edge weight depends only on how many topics join the pair |
| `Topo-QoS-Perm` | `Topo-QoS` with topic QoS profiles permuted uniformly across the topics of each scenario (20 permutations, seeds 0–19, mean ρ). Labels are **not** permuted. |
| `Topo` / `Topo-QoS` on a QoS-independent corpus | The twelve scenario configurations regenerated with a new opt-in generator switch `qos_affinity: false`, which removes QoS from topic selection and from criticality/hot-standby assignment; relabelled with the same oracle settings. The committed corpus is untouched. |

**Oracle sensitivity.** `I*` relabelled over propagation threshold θ ∈ {0.1, 0.2,
0.3} × depth-damping step ∈ {0.10, 0.15, 0.20} (floor 0.25). Reported: each
label's rank agreement with the shipped setting, and `Topo-QoS` ρ under each.

**Inert-vs-active rule.** Predict "active" (`I* > 0`) iff `Reach > 0`. Reported:
accuracy, balanced accuracy and F1 per fold.

### Contrasts

Exploratory family, Holm-corrected across the four new rankers: each of `Reach`,
`Reach-QoS`, `CDI`, `InDeg` against `Topo-QoS` (two-sided Wilcoxon over the twelve
folds, bootstrap 95% CI, B = 2,000). Each is also compared descriptively, per fold,
with the published CPU `HGT-QoS` values (S23).

### Decision rules

| Rule | Condition | What changes in the text |
|:---|:---|:---|
| R1 | The best of the four new rankers has LOSO mean ρ ≥ 0.622 (`HGT-QoS`, CPU) | Abstract, §1 and §9 say that learned engines do not beat a training-free reachability score on this oracle; the learned contribution is narrowed to transfer and identification. |
| R2 | `Topo-Mult` or `Topo-QoS-Perm` retains ≥ 50% of the published `Topo` → `Topo-QoS` gain (fold-mean ρ − 0.349) / 0.204 | The claim that *declared QoS contracts* produce the gain is replaced throughout by *QoS-weighted dependency multiplicity*; the QoS-content share is reported as what the controls leave. |
| R2′ | On the QoS-independent corpus, `Topo-QoS` − `Topo` < 50% of the published +0.204 | The generator coupling is reported as part of the mechanism of the gain. |
| R3 | Always | Every arm is reported in the manuscript or supplement, whichever way it comes out. |

**What is unchanged.** All registered contrasts of the plan and Amendments 1–6, their
families and their outcomes. No learned model is retrained.

---

## Amendment 8 — attribution controls (2026-09-26, after their results existed)

*Numbering note.* This amendment was first committed as "Amendment 7" on a parallel
branch. When the branches were merged, the record was put in date order: the
training-free amendment above (2026-09-25) keeps 7, and this one becomes 8. Its
text is otherwise unchanged.

**Status when written:** every run below is complete. This amendment is post hoc,
and everything it adds is exploratory. It registers no contrast and changes no
registered conclusion. It records why the manuscript's *interpretation* of RQ2
and RQ3 changed.

**Why it exists.** A receptive-field probe
(`reproduce/receptive_field_probe.py`) showed that no relation on the native
multigraph targets an Application. Every edge points from Application to Topic,
Node or Library, and `GATConv` aggregates from source to target only. The
untyped GAT arms of Amendment 2 (`gl_full_cap`, `gl_full_qos16_cap`) and
Amendment 6 (`gl_qos16_prior`) therefore score every Application from its own
features. On their trained checkpoints, deleting every edge changes no
Application prediction. HGT reaches Applications through its reverse pass.

Two consequences for the registered 2×2:
- It compared typed message passing with per-component learning, not two
  message-passing architectures.
- Its Q factor switched two inputs at once: the 16-D edge channel and three
  QoS node columns (`qos_weight`, `qos_weight_in`, `qos_weight_out`).

The registered arms are reported as registered. Nothing registered is re-run
or replaced.

**Added arms.**
- `tab_gbm` (GBM-Feat). It was declared post hoc in Amendment 3 and never run
  until now. Before its first run it was given the neural arms' per-graph label
  transform (`normalize_labels_robust`), because the earlier code fed it raw
  labels. A code comment claiming its node features are identical with and
  without QoS was wrong and was corrected.
- `tab_gbm_qos` (GBM-Feat-QoS): GBM-Feat reading the QoS-on node features.
- `gl_full_qos16_nfmask` (GAT-QoS-nf): `gl_full_qos16_cap` with GAT's node
  features and GAT-QoS's edge attributes. Each input is bit-identical to one
  parent arm.

**Run.** One CPU invocation (`make -f reproduce/Makefile rq-attribution`):
- LOSO arms: `topo_qos`, `gl_full_cap`, `gl_full_qos16_cap`,
  `gl_full_qos16_nfmask`, `tab_gbm` and `tab_gbm_qos`.
- Protocol: 12 folds × 5 seeds, Application population.
- Zero-shot on the five system models for the five learned arms, at 3 layers
  and 300 epochs.
- Checks: `gl_full_cap`, `gl_full_qos16_cap` and `topo_qos` reproduce their
  earlier rows bit for bit.

**Contrasts.** Five, as two-sided Wilcoxon tests over folds with Holm correction
across the five (`reproduce/attribution_contrasts.py`). No decision rule was
fixed in advance.

| Contrast | Δρ | Won | p | p_Holm |
|:---|---:|:---:|---:|---:|
| GAT-QoS vs GAT-QoS-nf (QoS node columns) | +0.095 | 11/12 | 0.0049 | 0.024 |
| GAT-QoS-nf vs GAT (QoS edge channel) | −0.023 | 3/12 | 0.064 | 0.192 |
| GBM-Feat-QoS vs GBM-Feat | −0.010 | 5/12 | 0.519 | 1.000 |
| GAT vs GBM-Feat | −0.079 | 1/12 | 0.0093 | 0.037 |
| GAT-QoS vs GBM-Feat-QoS | +0.003 | 7/12 | 1.000 | 1.000 |

**What changed in the manuscript.** RQ2 now asks where learned accuracy comes
from, and the answer is the per-component features:
- relation typing, message passing and the QoS edge encoding add nothing
  measurable;
- the untyped engine's QoS gain is carried by the three node columns.

RQ3 keeps its numbers and adds GAT (0.831) and GBM-Feat (0.757) to the transfer
table. The directionality control (`hgl_qos_uni`) is still unrun. Without its
reverse pass HGT would also score Applications per node, so that control now
tests whether HGT's message passing contributes at all.

## Results log — Amendment 2's directionality arm (2026-09-26)

This is not an amendment: it records a registered arm being run, under Amendment
2's reporting commitment ("every arm that is run is reported").

`hgl_qos_uni` (HGT-QoS-U: HGT-QoS without its reverse pass, 330,895 parameters)
was run in one CPU invocation with `topo_qos` and `hgl_qos`, using
`make -f reproduce/Makefile rq-directionality` at a clean commit. Both
comparators reproduce their published rows bit for bit. On this substrate the
reverse pass is HGT's only route into Applications (Amendment 8), so HGT-QoS-U
scores each Application from its own features.

- **Registered contrast** (Amendment 2 control family). HGT-QoS vs HGT-QoS-U:
  Δρ = −0.010 [−0.064, +0.036], 6/12 folds, W = 37, p = 0.910. Holm across
  Amendment 2's three run controls gives 1.000.
- **Zero-shot.** HGT-QoS-U scores 0.804 against HGT-QoS 0.760. It is higher on
  all five system models.
- **Omnibus.** The contrast joins the pooled family, which grows from 11 to 12.
  Hybrid-GAT p_omni = 0.018 (was 0.016) and Hybrid-HGT p_omni = 0.038 (was
  0.034); both remain significant. Amendment 2's other two controls move from
  a family p_Holm of 0.761 to 1.000.

Directionality therefore does not confound the typing result, and HGT's message
passing contributes nothing measurable on this target. The capacity-only control
`gl_full_qos_cap` (GAT-w) remains unrun.

## Results log — Amendment 2's capacity arm (2026-09-26)

`gl_full_qos_cap` (GAT-w: untyped GAT at HGT's budget, 439,272 parameters,
QoS-on node features and a scalar edge weight) was the last registered model
arm left unrun. It was run with `make -f reproduce/Makefile rq-capacity` at a
clean commit, in one CPU invocation with `topo_qos` and `hgl_qos`. Both
comparators reproduce bit for bit. Like every untyped arm, GAT-w scores
Applications per node.

- **Registered contrast** (Amendment 2 control family). HGT-QoS vs GAT-w:
  Δρ = −0.011 [−0.056, +0.031], 5/12 folds, W = 33, p = 0.677. Holm across all
  four controls gives 1.000.
- **Descriptive.** GAT-w scores 0.633. It is level with GAT-QoS (−0.002) and
  above GAT (+0.070, 10/12). Its zero-shot ρ is 0.794.
- **Omnibus.** The family grows to 13. Hybrid-GAT p_omni = 0.019 and Hybrid-HGT
  p_omni = 0.041; both remain significant.

Every model arm Amendment 2 registered has now been run. Its label-side arm (a
sweep against a cache with the oracle's QoS ladder disabled) was not run as a
sweep; the manuscript bounds the label's QoS content instead (§4.3).

## Amendment 9 — graph learning on the dependency graph (2026-09-26, before any result)

**Status when written:** none of the four learned arms below has been implemented
or run. The closed-form comparators (`InDeg`, `Reach`) are *not* unseen: Amendment 7
published them (`results/tf_baselines.json`; LOSO ρ 0.764 and 0.732; five system
models 0.863 and 0.938). The native counterparts are not unseen either
(Amendments 2, 6 and 8). Only the four learned arms' numbers do not exist yet.

**Why it exists.** Two facts, taken together, leave one question open.
1. Amendment 7: counting a component's dependents on the `DEPENDS_ON` projection
   (`InDeg`, `Reach`) beats every learned engine under LOSO (`InDeg` vs HGT-QoS
   +0.143, 10/12).
2. Amendment 8: every learned engine reads the native multigraph, where no relation
   targets an Application. The untyped GATs therefore score each Application from its
   own features, and HGT reaches Applications only through its reverse pass.

No learned model has ever been run under LOSO or zero-shot on the dependency graph
itself. On the Application–Library projection (Rules 1 and 5, dependent →
dependency), a directed `GATConv` aggregates each component's *dependents*. Three
layers therefore see its dependents up to three hops away, which makes it a learned
relative of `Reach`. The primary oracle is itself a damped reverse reachability over
the same subscriber → publisher relation (referee M2). This amendment asks two things:
- Does learning on the dependency graph beat counting dependents?
- Was the missing receptive field why the native engines lose to the counts?

One input fact is recorded before the run. Every learned arm already reads the
node feature `in_degree_centrality`, taken from the cached app-layer analysis. It
ranks Applications like projection `InDeg`:
- identically on ATM (Spearman 1.000);
- at 0.548–0.901 on the other eleven folds.

It is a correlate of `InDeg` computed on a different rule set, not `InDeg` itself.

### Arms, fixed before any run

Every arm keeps the native build's node features and labels bit for bit; only the
edges change. Every arm runs 3 layers, fixed (`--no-auto-layers`), and uses the
Amendment 2 arms' seeds {42, 123, 456, 789, 2024}, 300 epochs and inner-validation
protocol, on CPU.

The substrate is the projection built from the committed topology by
`derive_depends_on_edges`, the same edge set `InDeg` and `Reach` read. It keeps
Application and Library nodes, in native-graph order. Its relations are
(App|Lib, `DEPENDS_ON`, App|Lib): four triples on the synthetic corpus, and two on
the real system models, which have no Library → Application or Library → Library
Rule-1 edges. The edge scalar `w(e)` is the projection's `qos_weight`.

| id | Label | Edge channel | Width | Params | Native counterpart |
|:---|:---|:---|---:|---:|:---|
| `gl_proj_cap` | GAT-P | none (QoS masked everywhere) | 296 | 437,496 | `gl_full_cap` (GAT) |
| `gl_proj_qos16_cap` | GAT-P-QoS | 16-D QoS | 288 | 429,992 | `gl_full_qos16_cap` (GAT-QoS) |
| `gl_proj_qos16_indeg_prior` | Hybrid-GAT-P | 16-D QoS, plus the rank-normalised `InDeg` prior | 288 | 431,433 | `gl_qos16_prior` (Hybrid-GAT) |
| `hgl_proj_qos` | HGT-P-QoS | 16-D QoS, bidirectional | 100 | 430,680 | `hgl_qos` (HGT-QoS) |

Declared differences from the native arms:
- **HGT-P-QoS matches parameters, not width.** At width 64 it has 180,048
  parameters, because the projection has four relation triples where the native
  graph has ten. Width 100 gives 0.991× the native HGT's 434,620.
- **Hybrid-GAT-P's prior is `InDeg`, not `Topo-QoS`.** It is rank-normalised
  exactly as the Topo-QoS prior is, and the residual construction is unchanged
  (Amendment 5).
- **The training labels shrink.** Broker, Topic and Node labels are not in the
  projection, so every arm trains on Applications and Libraries only. The scored
  population is unchanged: Applications, scored with
  `compute_inductive_metrics(population="application")` against the native graph's
  labels.
- **Zero-shot.** Trained on the twelve synthetic folds and scored on the five system
  models, exactly as for the published arms (`reproduce/realworld_zeroshot.py`,
  3 layers).

**Comparator provenance.**
- `InDeg` and `Reach` are recomputed per fold on the same labels and must match
  `results/tf_baselines.json` within 1e-3. The result of that check is reported
  either way.
- Native counterparts are read from their clean artifacts, all on corpus digest
  `3afa81f0`:
  - `loso_attribution_cpu.json` (GAT, GAT-QoS);
  - `loso_hybrid_gat_cpu.json` (Hybrid-GAT);
  - `loso_directionality_cpu.json` (HGT-QoS).
- `topo_qos` and `gl_full_qos16_cap` are re-run in the new invocation. They must
  reproduce their earlier per-fold rows bit for bit; this checks that the routing
  change did not touch the native path.

A receptive-field probe on the new arms is reported beside the results
(`reproduce/receptive_field_probe.py --substrate projection`).

### Contrasts

A new exploratory family of twelve contrasts. It is Holm-corrected within itself and
does **not** join the manuscript's 13-contrast omnibus, which is unchanged. Each of
the four arms is compared with:
- `InDeg`;
- `Reach`;
- its native counterpart.

Each contrast is a two-sided Wilcoxon test over the twelve LOSO folds, with a
bootstrap 95% CI (B = 2,000).

Reported descriptively, without tests:
- Topo-QoS;
- GBM-Feat;
- `rho_active`;
- the zero-shot means over the five system models.

### Decision rules

| Rule | Condition | Conclusion |
|:---|:---|:---|
| D1 | Some arm beats `InDeg` at Holm p < 0.05 | Learning on the dependency graph adds value. That arm is recommended over the counts. |
| D2 | No arm differs from `InDeg` at Holm p < 0.05 | No measurable advantage. `InDeg` / `Reach` is recommended as the simpler predictor. |
| D3 | Every arm's mean Δρ vs `InDeg` is below 0 | "Learning on the dependency graph does not beat counting dependents" is stated without hedging. |
| M | A projection arm beats its native counterpart at Holm p < 0.05 | The missing receptive field is reported as a mechanism behind the native engines' deficit. |
| Z | The best arm's zero-shot mean is below `Reach`'s 0.938 | The deployment recommendation for unseen real systems stays `Reach`. |

**What is unchanged.**
- The manuscript, all registered contrasts of the plan and Amendments 1–8, and the
  13-contrast omnibus.
- No published arm is retrained or replaced.
- Amendment 7's reporting obligation (R3) is recorded here as still open: the
  current manuscript (v4) does not report its arms.

## Amendment 10 — value of the dependency derivation (2026-09-26, before any result)

**Status when written:** none of the arms below has been implemented or run. `InDeg`
and `Reach` are already published (Amendment 7).

**Why it exists.** The referee report of 2026-09-26 (`reviews/review_2026-09-26.md`, M2)
asks what SaG's `DEPENDS_ON` derivation contributes beyond a subscriber count. It
proposes two comparisons: `InDeg` computed on the raw multigraph (2-hop subscriber counts
through topics), and `InDeg` without Rule 5.

**Two identities, recorded rather than tested.** Both of the referee's comparisons are
identities by construction, so this amendment does not run them as experiments.
- For an Application, projection `InDeg` equals its raw 2-hop subscriber count.
  `derive_depends_on_edges` adds one Rule-1 edge per distinct subscriber of the topics
  v publishes, and it contains no other edges into an Application: Rule-5 edges point
  into Libraries. So removing Rule 5 does not change any Application's `InDeg`.
- On the raw multigraph, the transitive publisher → topic → subscriber closure is the
  same node set as `Reach` without Rule-5 edges.

A test pins both identities on every committed scenario
(`tests/test_dependency_graph_substrate.py`). The approved plan named `Reach-raw` and
`Reach-noR5` as two arms; because they are identical, they are one arm here, `Reach-R1`.

The manuscript will therefore say plainly that `InDeg` is publish–subscribe afferent
coupling (AIS; Martin's Ca), made computable by the derivation. The measurable
questions are two:
- Does deriving topic-mediated dependencies beat counting raw connections?
- Does the derived library rule (Rule 5) add to transitive reach?

### Arms, fixed before any run

All arms are training-free. They are scored with `training_free_suite.score` against
`labels_for` I*(v) (published settings) on the Application population of the twelve
LOSO folds and the five system models.

| Arm | Score for Application v |
|:---|:---|
| `Degree-raw` | Total degree of v in the raw multigraph (`build_graph_from_json`): every PUBLISHES_TO, SUBSCRIBES_TO, RUNS_ON and USES edge. This is the count available without deriving dependencies. |
| `Pubs-raw` | Number of topics v publishes to. This is the raw proxy for "has consumers". |
| `Reach-R1` | Transitive dependents of v over Rule-1 edges only (`nx.ancestors` on the projection with Rule-5 edges removed), normalised as `Reach`. |
| `InDeg`, `Reach` | Comparators, recomputed and checked against `tf_baselines.json` (max \|Δ\| ≤ 1e-3). |

### Contrasts

An exploratory family of three: two-sided Wilcoxon over the twelve folds, bootstrap
95% CI (B = 2,000), Holm across the three.
- `InDeg` vs `Degree-raw`
- `InDeg` vs `Pubs-raw`
- `Reach` vs `Reach-R1`

The five system models are reported descriptively.

### Decision rules

| Rule | Condition | What changes in the text |
|:---|:---|:---|
| E1 | `InDeg` beats both raw counts at Holm p < 0.05 | §3, §8 and the abstract may say that deriving topic-mediated dependencies is what makes the count predictive. |
| E1′ | Otherwise | Those sections say the derivation makes afferent coupling computable, and make no claim that it outperforms raw counts. |
| E2 | `Reach` beats `Reach-R1` at Holm p < 0.05 | §3 and §8 say the library rule adds to transitive reach. |
| E2′ | Otherwise | No Rule-level contribution is claimed. |
| E3 | Always | Every arm is reported in the supplement. |

**What is unchanged.** All earlier registered contrasts and the 13-contrast omnibus.

## Results log — Amendments 7, 9 and 10 in the manuscript (2026-09-26)

This is not an amendment. It records how the text consequences of three registered families were
applied when the manuscript was revised to lead with the dependency graph.

**Amendment 7.** R1, R2, R2′ and R3 all applied, but the v4 manuscript had not carried them out.
They are now carried out:
- **R1.** The abstract, §1 and §9 say that no learned engine beats the training-free dependency
  count on this oracle. The learners on the dependency graph *match* it (Amendment 9).
- **R2.** The Topo → Topo-QoS gain is attributed to "QoS-weighted dependency multiplicity" on the
  Application–Library graph, not to QoS contract content (§6.2, §7.1).
- **R2′.** The generator's QoS–topology coupling is reported as part of that mechanism (§7.1).
- **R3.** `InDeg` and `Reach` appear in Table 7. Every arm is in Supplementary S32, which is
  rendered from the committed artifacts and reconciled.

**Amendment 9.** D3, M and Z applied, and each is reported in §7.1–7.3 and §8.
- **D3:** no arm exceeds `InDeg`.
- **M:** the dependency-graph GATs beat their raw-multigraph counterparts.
- **Z:** `Reach` stays the recommendation for unseen systems.

All twelve contrasts are in Table 8 and Supplementary S33.

**Amendment 10.** E1, E2 and E3 applied.
- **E1:** §3, §7.1 and §8.3 say the derivation is what makes the count predictive.
- **E2:** §7.1 says the library rule adds to transitive reach.
- **E3:** every arm is in Supplementary S33.

**Unchanged.** The 13-contrast confirmatory omnibus, and every registered decision.
