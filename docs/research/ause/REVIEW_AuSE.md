# Referee Report

**Manuscript:** "SaG-Prescribe: Unifying Diagnostic Pathways with Counterfactual Refactoring for Automated Code Review and Software Quality Evaluation in Distributed Publish–Subscribe Systems"

**Venue:** *Automated Software Engineering* (Springer) — Special Issue on Intelligent Techniques for Automated Code Review and Software Quality Evaluation

**Reviewer recommendation:** **Major Revision**

---

## 1. Summary

The paper argues that conventional static code analysis leaves an *Architecture-Code Gap* in publish–subscribe systems (ROS 2, DDS, MQTT), where a codebase can pass every file-level check while harbouring topology-level pathologies that only manifest as production cascades. It proposes SaG-Prescribe, which extracts a typed multi-layer dependency graph from Architecture-as-Code descriptors, folds file-level SCA metrics into a Code Quality Penalty, scores components on a hierarchical Reliability–Maintainability composite anchored to ISO/IEC 25010 and 25019, detects nineteen formalised pub-sub anti-patterns via adaptive box-plot thresholds, and compiles findings into three graph-mutation refactoring operators, each admitted only if a sandboxed discrete-event cascade simulation shows an impact reduction exceeding seed noise at every propagation threshold. Evaluation over eight synthetic scenarios reports that the composite does not out-rank degree centrality, that the catalog agrees with ground-truth criticality at chance level, that the Availability dimension is structurally degenerate on the layer measured, and that counterfactual verification admits 1128 of 1589 candidate edits with a significant System Risk Index reduction in all seven prescriptive scenarios.

## 2. Overall Impression & Assessment

Let me open with what is genuinely unusual about this submission, because it deserves to be said before the criticism. The authors report, in their own abstract, that their composite score loses to degree centrality; that their nineteen-pattern catalog achieves Cohen's κ = −0.002 against ground truth; that one of their two Reliability sub-dimensions is measured on a substrate where it cannot vary; and that a ρ = 0.94 correlation they themselves published at RASSE 2025 collapses to ρ = 0.485 under an independent oracle. §9.3's reconciliation of the two results — predictor and oracle sharing a substrate in the earlier work, so the correlation was partly structural — is a genuinely useful methodological contribution that generalises well beyond this paper, and I would like to see it survive revision intact. I reviewed this manuscript expecting the usual quiet burial of inconvenient numbers and did not find it. That is a real credit to the authors and it is why my recommendation is Major Revision rather than Reject.

I also verified the paper's arithmetic. Every column mean in Tables 5–8 recomputes correctly; the 6329/98/6231 finding-scope decomposition is internally consistent; 1128/1589 = 71.0%, and the three per-operator survival rates and per-scenario admission counts all reconcile to those totals. The numbers in this paper are, as far as I can check them, the numbers the experiments produced.

The problem is what remains standing afterwards.

**Originality.** The pub-sub anti-pattern catalog is, to my knowledge, the first of its kind, and it is a legitimate contribution as a *specification*. Counterfactual verification of refactoring candidates against a cascade simulator — verify-before-recommend as a discipline rather than a slogan — is a genuinely good idea and the most publishable core in the paper. Neither is as novel as the framing suggests: the catalog's specification structure is explicitly borrowed from Taibi et al., and closed-loop architecture optimisation against simulated quality attributes has a long lineage in the Palladio/PerOpteryx line of work that the paper does not cite.

**Significance.** This is where I have the most difficulty. After §§9.1–9.3, the diagnostic pathway has been shown not to rank better than a one-line centrality computation, and its catalog has been shown not to discriminate at the component level. The authors' response is to relocate the contribution: the value, they say, is *attribution* — the auditable decomposition into Fault Tolerance / Availability / Maintainability routed to Reliability Engineer / SRE / Architect. I am sympathetic to that argument in principle. But it is asserted in §1.3, §9.1, §10.1 and §11 and evidenced nowhere. There is no user study, no task-based evaluation, no developer survey, not even a worked qualitative example of a review comment the bot produces. A claim that explainable attribution helps practitioners is an empirical claim about practitioners, and this paper contains no data about practitioners. For a special issue on automated code review, that is the decisive gap.

**Methodological soundness.** Mixed, and the weakest link is load-bearing. The per-edit acceptance filter is sound: it uses an oracle genuinely independent of the RM score. The headline prescriptive result is not: ΔSRI is a reweighting of the very R and M the paper has just finished discrediting (Major Comment 1). And the paper's central methodological claim — that verification is what makes the recommendations trustworthy — is never tested against an unverified control (Major Comment 2). Beyond that, the formal model is not reproducible as written: two of its four core equations define functions by naming them (`FT(v) = f(...)`, `A(v) = g(...)`) without ever giving `f` or `g`.

**Suitability for a Q1 venue.** The topic fits the special issue squarely. The execution does not yet meet the bar, for three reasons: the corpus is a closed monoculture (the authors' generator feeds the authors' analyser, judged by the authors' simulator, with no real system anywhere in the loop); there is no comparison against any external tool; and the "automated code review" contribution has never been run on a commit, a diff, or a pull request. The paper is also non-compliant with the journal's submission format in ways that will be caught at desk check (§5 below).

My honest assessment: there is a good paper here, and it is smaller and sharper than the one submitted. It is about counterfactual verification of architectural refactorings, with the diagnostic pathway as the mechanism that generates candidates rather than as a co-equal contribution defended against evidence. The current manuscript spends a great deal of its length defending a diagnostic contribution its own results do not support, and comparatively little establishing the prescriptive contribution they arguably do.

## 3. Major Comments

### M1. The headline prescriptive result is measured in a unit the paper itself invalidates

§9.4's claim — every scenario achieves a significant net risk reduction, Wilcoxon *W* = 0, *p* = 0.0156 — is measured entirely in ΔSRI. §8.3 defines `SRI = 0.5(1−H_R) + 0.5(1−H_M)` and never defines `H_R` or `H_M`. Following the implementation, `H_d = 1 − (QoS-weighted mean of d over components)`, so

> SRI = 0.5·R̄_w + 0.5·M̄_w

That is, SRI is the QoS-weighted mean of the same R and M that constitute Q(v), differing from Q(v) only in its dimension weights (0.5/0.5 rather than 0.80/0.20). The paper should say this plainly, because as currently written the `H` indirection conceals it.

The consequence is severe. §9.1 has just shown that Q(v) ranks no better than degree centrality against the cascade oracle (ρ = 0.485 vs 0.519). §9.3 has just shown that A(v) — which carries 0.64 of R(v), which carries the larger half of SRI — has essentially no variance on the layer measured. The paper therefore establishes that its risk metric is weakly valid and then measures its principal contribution's success in that same metric. The prescriptive engine is being graded by a marker the paper has already told us is unreliable.

There is a further circularity: candidates are compiled from components flagged CRITICAL/HIGH by RM (§5), and the whole-policy gate then accepts the batch iff RM-derived SRI improves. Selection and evaluation share a scoring function. §6.4's claim of "complete view independence" holds for the per-edit filter — which uses the independent cascade simulator — but not for the second-level gate, and the manuscript does not distinguish them.

**What I need to see.** Report the primary prescriptive outcome in cascade-impact units — the same independent oracle `I_comp` used for the per-edit filter and for all of §9.1's validation. `results/prescribe_all.json` already carries `mean_cascade_impact_reduction` (0.0352); that quantity, aggregated per scenario with dispersion, is the defensible headline. Keep ΔSRI as a secondary, explicitly-labelled internal-consistency measure, and state the circularity where it is introduced rather than leaving a reader to derive it.

### M2. "Verify-before-recommend" is asserted but never demonstrated

This is the paper's central methodological claim and it currently rests on an invalid inference. §10.2 states that "an unverified engine would have applied 461 harmful or ineffective mutations (29.0%)". The 461 are the edits that failed `ΔĪ_θ > κ·σ_seed,θ` at some θ. Failing that test means *the measured improvement did not clear the noise floor at κ = 1.0*. It does not mean the edit was harmful, and it does not mean it was ineffective — the rejected set necessarily also contains edits that were beneficial but noisy, and edits that were neutral. Rewriting a "did not clear the margin" result as "harmful or ineffective" is exactly the kind of inferential slippage this paper elsewhere goes out of its way to avoid.

More fundamentally: the value of verification is never measured. There is no arm in which all 1589 candidates are applied, no random-edit control of matched size, and no comparison against a naïve "apply every candidate from a CRITICAL/HIGH component" policy. Without one of these, the reader cannot tell whether the filter is doing useful work or whether applying everything would have produced a comparable — or better — outcome. Given that the filter's own decision statistic is the independent cascade oracle, this experiment is cheap: re-run the seven scenarios applying (a) all candidates, (b) a size-matched random subset, (c) the admitted subset, and compare final impact. That single table would convert the paper's core claim from plausible to demonstrated, and I regard it as the most valuable addition available to this submission.

Relatedly, §9.4's Wilcoxon test is over *n* = 7, where *W* = 0 with all-positive differences yields *p* = 0.0156 — the smallest attainable value at that sample size. §10.5 concedes this, which I appreciate. But the deeper issue is unaddressed: the seven scenarios are configurations of one generator, not samples from any population, so the inferential frame the test presupposes does not exist. I would rather see per-scenario effect sizes with bootstrap intervals and no *p*-value than a significance claim the design cannot support.

### M3. The relocated contribution — "attribution, not accuracy" — has no supporting evidence

Once RQ1–RQ3 have been conceded, the paper's positive claim is that deterministic role-attributed decomposition is worth having even though it does not rank better. §1.3 states it, §9.1 finding 1 states it, §10.1 states it, §11 states it. None of them supports it.

What would support it: a controlled study in which practitioners triage findings with and without the RM decomposition and role routing, measured on time-to-diagnosis or diagnostic accuracy; or, at lower cost, a structured expert walkthrough on a handful of findings with independent raters assessing whether the attributed role matched the role that would actually own the fix. I note the repository contains `reproduce/run_expert_study.py`, which suggests some such instrument exists or was contemplated; if so, its results belong in this paper.

Absent human-subject data, the claim must be re-scoped honestly: the decomposition *is available* and *is auditable by construction*, which is a property of the design, not a demonstrated benefit. The paper may argue that this property is desirable and cite the explainability literature for why; it may not present it as an established contribution.

### M4. The automated code review contribution is a design sketch, not an evaluated system

The title, the venue and Contribution 5 all promise automated code review. The evaluation contains no commit, no diff, no pull request, and no repository. The gate is run on whole scenario snapshots in absolute mode; §7.2 reports that absolute mode returns Exit Code 2 (Block) on all eight scenarios, which the authors correctly identify as unusable in practice. The mode that would be usable — delta-aware gating against a merge base, with a waiver register — is stated in §7.2 to be "specified here for production integration but not yet implemented".

So the only implemented gating mode is one the paper demonstrates to be unusable, and the usable one does not exist. Contribution 5 cannot stand as written. Either implement delta-aware gating and evaluate it on a sequence of topology-modifying changes (which the scenario generator should make straightforward to synthesise), or demote it explicitly from a contribution to a described integration design and remove the corresponding claims from the abstract and §1.4.

I would add that even a single worked example — one PR-shaped change, the resulting findings, the review comment text emitted, and the exit code — would materially strengthen the paper's connection to its special issue. There is currently no qualitative artifact of any kind showing what this system says to a developer.

### M5. The formal model is not reproducible as specified

For a paper whose §3 is presented as a formalisation, the specification has gaps that would prevent independent reimplementation:

- Eq. (6): `FT(v) = f(RPR(v), DG_in(v), MPCI(v), FOC(v))`. `f` is never defined. The listed argument set is also wrong in a way that matters — in the implementation, FOC appears only in the Topic branch, and the non-Topic branch uses a `CDPot_enh` term built from RPR, in-/out-degree and MPCI. As written, the equation describes neither branch.
- Eq. (7): `A(v) = g(AP_c,directed(v), BR(v), CDI(v), w(v))`. `g` is never defined, and the implementation's A(v) is a *five*-term sum that includes an interaction term QSPOF = AP_c × w(v) not named here.
- Never defined anywhere: `MPCI`, `FOC`, `CDI`, `BR`, `PC`, `CouplingRisk_enh`, `H_R`, `H_M`.
- Definitions D1–D4 are stated and then never used. D2 (relationship criticality) defines a quantity the paper never computes or evaluates; edge scoring exists in the implementation but appears nowhere in the results.

These are all fixable from the authors' own documentation without new experiments, and they must be. A composite-score paper that does not state its composite is not reviewable on its merits.

### M6. Every weight in the model is unjustified

The manuscript specifies, without derivation: CQP's (0.10, 0.35, 0.30, 0.25); α = 0.36; M(v)'s (0.35, 0.30, 0.15, 0.12, 0.08); (w_R, w_M) = (0.80, 0.20); and the 3×2 Quality-in-Use projection matrix. The sole justification offered is a bare citation to Saaty (1980) attached to the composite weights. That citation names a method; it does not supply a pairwise comparison matrix, a priority vector, or a consistency ratio, and none appears in the paper.

Two specific requests:

1. **Provenance.** State where these numbers came from. If they descend from an earlier AHP elicitation — including one over a criteria set since revised — say so explicitly, give the original pairwise matrix and its consistency ratio in an appendix, and show the renormalisation. A declared derivation from a documented elicitation is defensible; an undeclared constant is not.
2. **Sensitivity.** Report how ρ(Q, I_comp) and the CRITICAL/HIGH tier assignments move under perturbation of α and (w_R, w_M). This matters more than usual here, because §9.3 establishes that one of the two components of R(v) is near-constant on the evaluated layer — which implies a large region of α-space produces near-identical scores, and the reader currently has no way to know how large.

**On the Quality-in-Use projection specifically.** The matrix `[[0.75, 0.25], [0.80, 0.20], [0.60, 0.40]]` maps a 2-vector to a 3-vector. It cannot add information; all three "stakeholder harm scores" are affine functions of the same (R, M) pair, and the three rows are near-collinear besides. Presenting `h_QiU` as three distinct harm dimensions (§3.4, Table 2) overstates what the model delivers — it is a reweighting of RM under three different weightings, not an independent assessment along three axes. Say so. Nothing is lost by saying so, and the alternative is a reader discovering it themselves and wondering what else was oversold.

### M7. CQP — the paper's motivating bridge — is defined but never evaluated

The Architecture-Code Gap opens the abstract, the introduction and the conclusion, and CQP is the mechanism that closes it. Yet:

- CQP enters only M(v), at weight 0.15, and M(v) enters Q(v) at 0.20. CQP therefore carries roughly **3%** of the composite. The paper never states this.
- There is no ablation. The reader cannot tell whether removing CQP entirely would change any reported result.
- Its inputs are synthetic, generated by the same harness that generates the topology, so even a favourable ablation would establish little about real code.
- §3.3 concedes that `sqale_debt_ratio` is ingested, persisted and exported but never scored — commendably honest, and simultaneously an indication of how thin the code-level layer is.

Either evaluate CQP (a with/without ablation on ρ and on tier assignment is cheap and would be sufficient) or substantially reduce the prominence of the Architecture-Code Gap framing. As it stands, the paper's central motivating claim is carried by 3% of its scoring function and zero of its experiments.

### M8. §5 presents an optimisation problem that the system does not solve

§5 states `min_Δ Σ_v I*_Δ(G)(v)` subject to `Cost(Δ) ≤ B`. Neither `Cost` nor `B` is ever defined, instantiated, or referenced again. No search over Δ is performed. The implemented procedure is a per-edit filter followed by a batch acceptance check — a screening rule, not an optimiser, and one with no notion of budget at all.

The formulation should either be removed or honestly reframed as the objective the filter *approximates*, with an explicit statement that no search is performed and no budget constraint is enforced. Retaining a decorative optimisation formulation invites exactly the comparison the paper cannot survive: against SBSE approaches (Harman et al., Aleti et al., NSGA-II), which the related work section cites and which do solve the stated problem.

### M9. No external baseline, and material gaps in the related work

RQ1's baselines are betweenness and degree — both computed inside the authors' own framework, on the authors' own projection. There is no comparison against any existing architectural analysis tool. This is defensible only if no comparable tool exists, and the paper does not make that argument.

Missing literature that a reviewer in this area will expect:

- **Architectural smell detection tooling and catalogs:** Arcan; Designite; DV8; Sonargraph. Several detect cyclic dependencies, hub-like dependency and unstable dependency — directly comparable to `CYCLE`, `HUB_AND_SPOKE` and `UNSTABLE_INTERFACE` in Table 3. At minimum, discuss why they are not applicable to pub-sub topologies; better, run one on the projected dependency graph as a baseline.
- **Architecture recovery and decay:** ARCADE (Garcia, Medvidović et al.); Mo et al. on architecture anti-patterns and hotspot patterns; Le and Medvidović on architectural decay. Mo et al. is especially relevant, being a catalog-plus-detection paper validated against maintenance outcomes.
- **Architectural smell surveys:** Herold et al.; Lenarduzzi et al.
- **The SonarQube quality-gate literature** (Lenarduzzi et al. on rule fault-proneness). This is the closest empirical analogue to §9.2's finding: a widely deployed rule-based screen whose flags correlate weakly with the outcome they purport to predict. The paper's κ ≈ 0 result is *less* surprising, and considerably better contextualised, against that background — and the authors are currently forgoing a supporting citation for their most contested finding.
- **Architecture optimisation against simulated quality attributes:** the Palladio / PerOpteryx line. The paper's §2.5 claims SBSE "operates open-loop, generating candidate architectures without empirically verifying resilience gains against simulated failure cascades". That claim is too strong as stated — simulation-in-the-loop architecture optimisation is precisely what that literature does — and it should be narrowed to what is actually distinctive here (discrete-event cascade semantics, per-edit noise-margin admission).
- **Chaos engineering:** Basiri et al. §2.1 dismisses chaos engineering in one sentence without a citation.

### M10. The corpus is a closed monoculture and the threats section does not say so

Eight scenarios are generated by the authors' generator, analysed by the authors' analyser, and scored against ground truth produced by the authors' simulator. No real system appears anywhere. §10.5 lists "Scenarios are synthetic" under External Validity, which understates the problem: the issue is not only that the systems are synthetic but that *every element of the measurement loop shares an author and a set of modelling assumptions*. If the generator and the cascade simulator embed a common view of how pub-sub failures propagate, then agreement between predictor and oracle is partly an artifact of that shared view — which is structurally the same criticism §9.3 makes, with admirable clarity, of the prior RASSE work.

Add this as a first-class construct-validity threat. And note the tension the paper should confront directly: §9.3 argues that a predictor validated against an oracle sharing its substrate will correlate well "almost by construction". That argument, applied consistently, also applies to the present evaluation — a weaker version of it, since `I_comp` genuinely does not share a computational substrate with Q(v), but not a null version, since both are downstream of the same generator's assumptions about topology and QoS.

### M11. Scale and latency claims are not supported by the reported data

- **The corpus scale is misstated.** The abstract and §1.4 claim "29–3325 vertices". Summing the scenario descriptors, the largest system (S07) has 300 + 50 + 120 + 10 + 40 = **520** vertices. Only the lower bound is right; the upper bound overstates scale by a factor of about 6.4. (S07's 3245 is its *edge* count, per Table 4 — I suspect a conflation.) This must be corrected everywhere it appears.
- **"Sub-second" and "real-time" are contradicted by the paper's own table.** §1.4 Contribution 5 claims a "sub-second pre-merge quality gate"; Figure 1's own diagram says "Sub-second latency"; the abstract calls the gate "real-time"; §9.5 and Table 5 report 20.98 s on S07. These cannot all be true.
- **The scaling is superlinear and uncharacterised.** 0.56 s at 200 applications → 20.98 s at 300 applications is roughly a 37× increase for a 1.5× increase in size. No complexity analysis is given for any stage. A reader considering CI/CD deployment on a system of realistic size cannot extrapolate from this, and the extrapolation implied by these two points is not encouraging. Give the asymptotic cost of the dominant stage (§9.5 attributes it to base metric computation — say which metric and why), and either report a scaling curve or withdraw the CI/CD feasibility claim for systems beyond the measured range.
- **"Ranking improves at scale" rests on n = 2.** §9.1 finding 2 compares mean ρ = 0.599 over two large scenarios against 0.447 over six smaller ones. The "large" group is S02 (0.718) and S07 (0.480); S07 — the largest system, and the one that matters for the extrapolation — sits essentially at the corpus mean. The entire effect is S02. This should be reported as a description of two scenarios, not as evidence about scaling behaviour.

## 4. Minor Comments

1. **Detector count contradiction.** §7.1 says the bot "executes the 19 anti-pattern detectors"; §9.5 says one (`DEEP_PIPELINE`) is excluded and reports 18 active. Reconcile.
2. **Scope taxonomy contradiction.** §4.3 states "three finding scopes — Component, Edge, and System", but Table 3's Scope column contains a fourth value, `Pair`, for `CHATTY_PAIR`.
3. **Table citation order.** Table 8 (`tab:operator_results`) is first cited in §6.3, ahead of Tables 5–7. The journal requires tables to be cited in consecutive numerical order.
4. **F₁ in Table 5 is near-vacuous and prominently placed.** §8.3 discloses that for Q(v), betweenness and degree, predicted and true critical sets are sized by the same rule, pinning precision = recall = F₁ by construction. Reporting F₁ as a headline column in the main detection table nonetheless invites misreading. Either drop the column or annotate it in the caption.
5. **§9.5's compositionality claim is unquantified.** "Two scenarios exhibit slight negative mean per-component changes among accepted edits" gives no numbers. (`results/prescribe_all.json` records `n_scenarios_regressed: 2`, so the claim is supportable — report the magnitudes.)
6. **Possible data error in Table 4.** S01 (152 vertices) and S05 (139 vertices) are both reported with exactly 797 edges. Please confirm this is not a copy error.
7. **Unpublished work in the reference list.** `sag_companion` is a manuscript under review at another journal, listed as `Authors` in the bibliography. The journal's guidelines state the reference list should contain only published or accepted works, with unpublished work mentioned in text only. Beyond formatting: two concurrent submissions sharing a graph model, a simulator and a benchmark corpus raise an overlap question the editor is entitled to assess. Disclose it explicitly in the cover letter and in Declarations, and state precisely which contributions are partitioned to which paper. §4.2's remark that forward centralities "feed only the learned GNN companion pathway of the JSS submission" indicates the boundary is well understood by the authors; it needs to be equally clear to the editor.
8. **Blinding is inconsistent and, in any case, broken.** The title block reads "Author Information Redacted for Double-Blind Review"; this journal is single-blind. `refs.bib` names Yigit and Buzluca in the `yigit2025graph` entry, and §1.5 identifies it as "our earlier conference paper", so the redaction achieves nothing. Restore a proper title page per the journal's Title Page section.
9. **No results figures.** The paper contains two schematic TikZ diagrams and no plot of any result. A Q1 empirical paper should show, at minimum, the Q vs. I_comp relationship, the per-type stratification underlying the Simpson's paradox finding (§9.3), and the ΔSRI or Δ-impact distribution. The repository already contains rendering scripts (`reproduce/render_stratified_figure.py`, `render_pooled_vs_pertype_figure.py`) and a rendered `results/figure4_stratified_rho.pdf`; §9.3's finding in particular is much more convincing shown than told.
10. **RQ2 is not well-formed.** "How well does the anti-pattern catalog's component-level agreement match its precision/recall profile" asks about the relationship between two of the authors' own metrics, not about a property of the system. Reformulate as a question about the catalog's screening behaviour.
11. **`BROKER_OVERLOAD` is self-defeating by construction.** §4.4 item 3 observes that a relative-to-median rule cannot fire when there are two equally overloaded brokers, since each *is* the median. Well spotted — but this is a specification defect in the catalog, not merely a benchmark artifact, and it should be labelled as such. The same critique applies to every within-population relative threshold in a small population.
12. **Domain count.** The abstract claims "six architectural domains" across eight scenarios, but seven distinct domains are named in Table 4 (S05 Hub-and-Spoke and S06 Microservices Mesh are presumably collapsed). Clarify.
13. **ISO/IEC 25019 grounding is thin.** The three harm dimensions are never named in full, never mapped to 25019's actual quality-in-use characteristics, and the projection matrix is unjustified (see M6). Either substantiate the mapping against the standard's own decomposition or describe it as "informed by" rather than "grounded in".
14. **Editorialising.** The manuscript repeatedly comments on its own virtue: "with complete candor", "promoted here to first-class status", "an instance of the more general substrate-adequacy issue ... stated as a modeling constraint rather than an ex post excuse", "We consider the re-measurement a strength of this submission's methodology rather than a weakness to minimize". The underlying candour is genuine and does not need advocacy — it is more persuasive stated flatly. This also costs real length.
15. **Redundancy.** Every principal finding is stated four or five times (abstract, §1.4 contributions, §9, §10, §11) at nearly full length. §§9–11 could lose 15–20% with no loss of content.
16. **Declarations.** Funding names TÜBİTAK without a grant number. Data Availability gives two shell commands but names no repository, archive or persistent identifier — a reviewer cannot act on it. The journal requires a data availability statement that explains how to *access* the data.

## 5. Compliance with the Author Guidelines

These will be caught at desk check and are worth fixing before resubmission:

| Requirement (Guide for Authors) | Manuscript | Action |
|---|---|---|
| Submission must be generated via LaTeX using `\documentclass{sn-jnl}` | `\documentclass[11pt,a4paper]{article}` | Port to the Springer Nature template |
| Cite by name and year; reference list alphabetised by first author | `\bibliographystyle{IEEEtran}`, numeric | Switch to author–year |
| Abstract 150–250 words | ~614 words | Cut by ~60% |
| 4–6 keywords | 10 | Cut to 6 |
| DOIs as full DOI links where available | none of 33 entries | Add |
| Reference list: published or accepted works only | `sag_companion` under review | Move to text mention |
| Data availability statement explaining how to access data | commands only, no repository | Rewrite |
| Single-blind review | title block claims double-blind | De-anonymise |
| Tables cited in consecutive order | Table 8 cited in §6.3 | Reorder |

## 6. Recommendation

### **Major Revision**

I want to be precise about why this is not a Reject, and not a Minor Revision.

It is not a Reject because the core is sound and the authors are trustworthy reporters of their own results. Counterfactual verification of architectural refactorings against a cascade oracle is a good contribution; the pub-sub anti-pattern catalog is a genuine first even if its detection performance is poor; and §9.3's substrate-adequacy analysis — a self-correction of a published result, with the mechanism identified rather than merely conceded — is a contribution to methodology that I would be sorry to see lost. Papers whose authors report negative results about their own prior work should be encouraged, not punished for the honesty that surfaced them.

It is not a Minor Revision because the paper's principal empirical claim is measured in a circular unit (M1), its central methodological claim is untested (M2), its relocated contribution is unevidenced (M3), its titular contribution is unimplemented (M4), and its formal model cannot be reimplemented as written (M5). Those are not presentation problems.

**What a successful revision looks like.** In descending order of importance:

1. **Re-anchor the prescriptive evaluation on the independent oracle** (M1) and **add the unverified-control arm** (M2). Together these are perhaps a day of compute against existing infrastructure, and together they convert the paper's strongest idea from asserted to demonstrated. If the authors do only two things, these are the two.
2. **Resolve the attribution claim** (M3) — either evidence it, or re-scope it to a design property and stop presenting it as a result.
3. **Decide what the code review contribution is** (M4). Implement delta-aware gating and evaluate it on synthesised change sequences, or demote it and adjust the title, abstract and contribution list accordingly. Add at least one worked example of the bot's output either way.
4. **Complete the formal specification** (M5) and **disclose weight provenance with a sensitivity analysis** (M6). Both are achievable from existing documentation and cheap experiments.
5. **Correct the factual errors** (M11) — particularly "29–3325 vertices", which materially misstates the evaluation's scale — and **reconcile the latency claims** with Table 5.
6. **Add the missing literature and at least one external baseline** (M9), and **strengthen the threats section** with the monoculture confound (M10).
7. **Fix compliance** (§5) and **cut length** (Minor 14–15).

I would be glad to review a revised version. I would also gently suggest that the authors consider whether the paper is better served by leading with the prescriptive contribution and presenting the diagnostic calibration findings as what they honestly are — a rigorous negative result about scalar architectural criticality scoring, which is publishable on its own terms and is currently buried under a defence of a contribution the data do not support.
