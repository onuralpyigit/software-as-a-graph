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
