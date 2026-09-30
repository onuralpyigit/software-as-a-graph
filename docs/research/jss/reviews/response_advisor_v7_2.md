# Response to advisor notes v7_2 (notlar_v7_2.docx — Sections 3–5)

**Branch:** `jss-advisor-v7-2-revision` (from `main` @ `c6538adf`)
**Status:**

| Check | Result |
|---|---|
| Manuscript | 32 pp (31 before; the new §3–5 paragraphs) |
| Supplement | 44 pp |
| Build | No undefined references and no overfull boxes |
| `reproduce/reconcile_manuscript.py` | `1618 figures match` (1603 before). See Tooling for what was added. |
| `pytest -m "not integration"` | 1422 passed |

Line numbers in the "Note" column are those of the v7 PDF. The advisor's text was applied verbatim, with LaTeX cross-references substituted for hard-coded numbers. Departures are marked **Deviation**.

## Section 3 (`sections/sec3_sag_model.tex`)

| Note | Change |
|---|---|
| 215–218 | New intro paragraph (the dependency layer "makes runtime failure paths explicit"; controlled comparisons on identical dependency information). |
| Fig. 1 boxes | The top label is RANKING PATHWAY. The box is "Ranking Methods" with the Analytical / Hybrid / Learned bullets. The oracle box reads "Simulation Oracles (I*, I_dyn, I_comp)", "Run on raw G_structural", "Offline: training and evaluation labels". The edge label is "Evaluated against simulated labels". Source: `figures/src/figure1_pipeline.dot`; the figure width is unchanged (620 pt). |
| Fig. 1 caption | New caption applied. |
| 229–230 | "…projection from which all rankers, analytical and learned, are computed". |
| 253–261 | New QoS paragraph. |
| 263–266 | New §3.3 opening. |
| after 272 | Mechanism sentences inserted. |
| 289–290 | "including Rule 5 raises the agreement of transitive reach with I* by +0.058 (9 of 12 folds, Holm p = 0.0068; Section 6.1)". The reconciler now checks these three numbers against `derivation_ablation.json`. |
| Fig. 2 caption | "…all rankers, analytical and learned, read view (b)". |
| 295 | "All ranker inputs are computed on G_analysis". |
| after 298 | Raw-multigraph comparison-point sentence added at the end of §3.4. |
| after 312 | "Two roles" paragraph added as the last paragraph of §3. Also: §3.5's "predictive pathway" → "ranking pathway". |

## Section 4 (`sections/sec4_failure_impact_prediction.tex`)

| Note | Change |
|---|---|
| 313 | Title: "Ranking Methods, Simulation Oracles, and the Reference Criterion". |
| 314–321 | Replaced with the three-category paragraph. |
| Fig. 3 | Panel titles: "(a) Ranking methods on one analysis graph", "(b) Simulator labels and evaluation". Inside the figure: "Learned model", "Hybrid model", "no ranker reads", and an oracle box listing I*, I_dyn and I_comp (`reproduce/render_jss_diagrams.py`). Caption (b) applied. Caption (a): "ranking engines" → "ranking methods", "learned/hybrid engine" → "learned/hybrid model". |
| 330–331 | "The homogeneous Graph Attention Network (GAT-QoS, and GAT-P-QoS on the dependency graph)". |
| 340–341, 344 | "…because the simulator produces no maintainability labels" and "over the permutation π induced by the simulated labels". |
| 361–363 | Title "Simulation Oracles"; new opening. |
| 380–381 | Already changed in v7. **Deviation (kept from v7):** the sentence cites Supplementary Section S38 (Table S62), where the n = 30 check actually is, rather than "the replication repository". |
| 383–384 | **The claim is correct; text kept.** The reconciler recomputes `folds_above_seed_mean_bound` from `data/benchmarks/idyn_rate_expansion.json` on every run: no ranker in Table 6 exceeds its fold's √r bound on any fold. |
| 395 | Title "Input–Label Separation and the Reference Criterion". |
| 399–402 | The overlap caveat and **The reference criterion.** are now separate paragraphs. The criterion paragraph ends by defining "predictors" as all non-reference rankers. |
| 406 | "For I_comp, no ranking is a truncation;". |
| 409–410 | The learned-methods sentence was replaced and the "References are therefore reported…" sentence added. |
| 414 | "All results are therefore conditional on simulation fidelity." |

## Section 5 (`sections/sec6_experimental_setup.tex`)

| Note | Change |
|---|---|
| after 415 | New §5 intro. |
| after 423 | Corpus-separation sentence. |
| 436 | "modeling notes" was already fixed in v7. |
| 447–448 | See "Replication link" below. |
| 449–453 | Title "Rankers and References"; new opening. |
| Table 4 | Regrouped: Analytical baseline / Analytical references / Hybrid / Learned (dependency graph) / Learned (raw multigraph) / Learned approximations of I_dyn. Header "Predictor" → "Ranker". Params are `---` for training-free and tree rankers. The caption was replaced; details below. |
| 470 | Two sentences added before the references paragraph. |
| after 481 | Eq. 7 "natural reference" sentence. |
| 484–485 | Exploratory-exception sentence. |
| 499 | Effect-size sentence. |
| after 525 | "Taken together…" closing paragraph. |

**Table 4 details:**
- **Reference rows:** in addition to the rows the advisor listed, the table also includes **Rate-weighted (Eq. 7)** and **GAT-P-QoS→dyn**. Both appear in the main-text Table 6, so Table 4 now lists every ranker the results tables report.
- **Row details verified in the repository:**
  - GBM-P-QoS→dyn: 18 columns, 9 dependency counts plus 9 QoS/rate columns (`reproduce/oracle_robust_ltr.py`).
  - GAT-P-QoS→dyn: the same 288-channel, 16-D architecture as GAT-P-QoS (`gl_proj_qos16_cap_idyn`), so 429,992 parameters.
- **Deviation (caption):** his text said "ablation arms (GBM-Feat, GIN-P-QoS) are listed in Table 8". Only GIN-P-QoS is in Table 8; GBM-Feat is reported in the §6.2 text. The caption now reads "(GIN-P-QoS, Table 8; GBM-Feat) are reported in Section 6.2".
- **"engine X" wording:** "Hybrid-X: engine X" became "model X", following the decision to sweep out "engine".

## Terminology audit ("did we change all terms?")

**Removed from the main text** (manuscript, abstract, highlights, title page), as the final grep confirms:
- "ground truth" / "Ground-Truth" (9 sites);
- "predictive pathway" / PREDICTIVE PATHWAY;
- "ranking engines";
- every "engine(s)", now "ranker(s)", "method(s)" or "model(s)".

The only remaining string is the invisible LaTeX label `fig:engines`. "Ground truth" was also removed from the supplement (6 sites).

**"Predictor" is kept, deliberately, in 20 main-text places.** It now has a definition (§4.4): every non-reference ranker. Each remaining use contrasts predictors with references ("not predictors", "best predictor per column", "no predictor reaches 90%"). Uses that meant "any ranker" were changed:
- §1.2 "Rankers read only the analysis graph";
- §3 captions and §3.4;
- §5.3 "Every ranker is scored";
- §7.5 "Rankers consume G_analysis";
- the "Ranker" column headers of Tables 4, 5 and 9.

**Still open (supplement only):** the supplement still says "learned engine(s)" 39 times, and two section titles use the old terms: S32 "Predictor Taxonomy and Substrate Details" and S34 "Engine Regimes". The sweep was scoped to the main text; a supplement sweep is mechanical and can follow if wanted.

## Replication link (447–448)

- **The link.** `\sagexperiments` pointed to `reproduce/`, whose README described an older version of the paper: old title, stale Table 6–8 numbering, "3 figures", HGT-QoS labeled "Proposed", and a dead link. The paper's sentence describes per-experiment pages (protocol, hyperparameters, make target, artifacts). Those live in `docs/research/jss/experiments/`, so the link now points there.
- **`docs/research/jss/experiments/README.md`:**
  - current title;
  - the index was rebuilt against the compiled `manuscript.aux`/`supplementary.aux` (§/Table/Figure and Supp. S-numbers);
  - rows added for Amendment 11 (learned I_dyn approximation) and Amendment 15 (rate-weighted reference);
  - no "surrogate" or "engine" wording.
- **`reproduce/README.md`:**
  - current title and authors;
  - the stale Steps and mapping tables were replaced by a short RQ → make-target table that defers to the experiment index;
  - the dead link and the "Proposed" label were removed;
  - every make target it names exists in `reproduce/Makefile`.
- **Before submission:** once merged, open the URL and check that it renders as intended on GitHub.

## Tooling (`reproduce/reconcile_manuscript.py`)

- **References in Table 4.** Amendment 13's rule used to forbid InDeg/Reach anywhere in Table 4. It now requires them to appear **only** inside Table 4's "Analytical: references" block, with the block holding exactly {Analytic I*, Rate-weighted, InDeg, Reach}.
- **Topo-QoS-only rule.** The check keyed on the old "Training-free baseline" heading and would have been silently skipped. It now keys on the new heading and fails if the heading is missing.
- **New quotes.** Added a check of the §3.3 Rule-5 quote. Re-anchored the §6.4 "every learned ranker needs" quote.
- **Mutation-tested.** A reference row outside its block, a missing baseline heading and a wrong fold count each produce a finding.

## The three anticipated criticisms: options

The paper already names all three as threats (§4.3, §4.4, §7.5, §7.6). The question is what, if anything, to add before a reviewer asks. Options are ordered by cost. Nothing below has been run.

### 1. Simulator-dependent evaluation

**Already in the paper:**
- three oracles with different mechanisms (reachability, queue-flow, multi-criteria);
- the reference criterion, which separates restating from predicting;
- partial ρ beyond I* and beyond the first-order term;
- oracle-parameter sweeps (`reproduce/threshold_sensitivity.py`, `reproduce/icomp_sensitivity.py`);
- the explicit "conditional on simulation fidelity" scoping.

**Options:**
- **(a) Reframe, no cost.** State the claim as rank agreement with *declared* failure models. The contribution is then the methodology for deciding when learning adds value relative to a given simulator, and it transfers to any better simulator. §4.3's new opening already moves in this direction.
- **(b) Emulated oracle, days.** Deploy one system model as containers, for example the EdgeX or Home Assistant model on an MQTT broker via Docker Compose. Kill each Application in turn and measure delivered-message loss at the survivors. This gives a fourth oracle that is not a simulator we wrote. Even a single system, reported as a convergent-validity check against I*, I_dyn and Eq. 7, answers "your simulator encodes your assumptions".
- **(c) Report the disagreement.** The oracles agree only partially (I_dyn vs I* ρ = 0.711). Present this as evidence that the conclusions are not an artifact of one simulator, since the analytical-vs-learned ordering holds on both I* and I_dyn.

**Suggested rebuttal line:** "No pre-deployment method can be validated against failures of a system that has not been deployed. We therefore evaluate against three simulators with different propagation mechanisms, and report which conclusions hold across all of them."

### 2. Single-modeler transfer models

**Already in the paper:** the single-modeler threat is stated in §5.1, §6.3 and §7.5; RQ3 is scoped to "these models, not real systems"; and the active-stratum ρ>0 shows the transfer is weak where it matters.

**Options:**
- **(a) Second modeler, 1–2 days of someone else's time.** `reproduce/model_agreement.py` already implements the protocol: the second modeler gets the public sources, not the model, and agreement is measured with entity and edge Jaccard per type. It has never been run. Running it on the two smallest systems (Home Assistant, EdgeX) gives a real number where the paper currently says "not reported". A lab member or the second author who has not seen the models could do it.
- **(b) Perturbation robustness, hours, computational.** Drop or rewire 10–30% of the edges in each system model and check whether the RQ3 ordering (Reach > InDeg > learned > Topo-QoS; ρ>0 weak) survives. This bounds how much modeler discretion could change the conclusion. It needs a pre-registered amendment before running, as with Amendments 7–15.
- **(c) Mechanical extraction, days.** Derive one model automatically, for example Autoware's topic graph from its ROS 2 launch and node configuration, or EdgeX's from its compose and service configuration, and compare it with the hand model using (a)'s script.

**Suggested rebuttal line:** "We quantify the sensitivity of the transfer results to modeling choices by [a/b] and report inter-modeler agreement of J = …."

### 3. No validation against real failures

**Already in the paper:** stated as a limitation (§4.4 and §7.5), with all results explicitly conditional on simulation fidelity.

**Options:**
- **(a) Positioning, no cost.** The target is design time: before deployment there are no failures to observe, which is the setting SaG addresses. Real-failure validation belongs to a follow-up study with operational data.
- **(b) Emulation, days.** Criticism 1's option (b) also gives partial real-execution evidence: real middleware and real message loss, but injected faults.
- **(c) Incident triangulation, 1–2 days.** For the open-source systems, collect public issue-tracker or postmortem reports of cascading failures and check qualitatively whether the components involved rank highly. This is weak evidence, but concrete.
- **(d) Future work, no cost.** Make it an explicit item in §7.6 with a concrete design: an industrial partner, a deployment log, and the observed blast radius as the label.

**Suggested rebuttal line:** "SaG targets the pre-deployment setting, where no failure data exists; we state all conclusions as conditional on simulation fidelity and outline the operational validation study in §7.6."

**Recommendation.**
- **Before submission:** 1(a) and 3(a) are wording only.
- **If time allows before submission, or else for the first revision:** 2(a) is the most valuable addition per hour, because the tooling exists and it turns a stated weakness into a measured one. 2(b) is next.
- **Most convincing single addition:** the emulation, 1(b)/3(b), since it addresses two criticisms at once. It is also the most expensive.
