# Response to advisor notes v8 (notlar_v8.docx: Section 2 and paper-wide notes)

**Branch:** `jss-advisor-v8-revision`, from `main` at `370aad7f`.

**Status:**

| Check | Result |
|---|---|
| Manuscript | 35 pp, unchanged |
| Build | No undefined references; no overfull boxes |
| `reproduce/reconcile_manuscript.py` | 1711 figures match |
| `tests/test_reconcile_guards.py` | 4 passed |
| `scripts/check_doc_links.py` | OK |

The "Note" column uses line numbers from the v8 PDF. The advisor's text was applied verbatim, with LaTeX cross-references in place of hard-coded section numbers. Departures are marked **Deviation**, and points already resolved are marked **Already addressed**.

## Which version the notes were written against

`manuscript-v8.md` is byte-identical to `manuscript.md` at **`8dd4a536`**, the v7_2 revision. Three revisions have landed on `main` since then:
- **`f8e8c3c1`**: a hand edit.
- **`2cdb7b02`**: round-10 referee revision, text only.
- **`eb17a246`**: round-11 referee revision, which ran Amendment 16.

They are summarized in [What changed since the v8 you reviewed](#what-changed-since-the-v8-you-reviewed). The redline `jss_v8_to_v9_redline.html` marks every change from v8 to this revision, including these three.

The redline compares Markdown renderings. Reference numbers shift wherever a citation moved, so some changed paragraphs differ only in a bracketed number.

## Section 2 (`sections/sec2_related_work.tex`)

| Note | Change |
|---|---|
| 145 | §2.1 title: "Dependability and Failure-Impact Analysis of Distributed Systems". |
| after 166 | The "Rankings of failure impact are often evaluated against a failure model or simulator…" paragraph is appended at the end of §2.1. |
| 167 | §2.2 title: "Static Analysis and Architectural Dependency Metrics". |
| 170 | "advanced graph measures" → "complex graph measures". |
| after 183 | "These studies suggest… They leave open whether learned methods add predictive information…" is appended at the end of §2.2. |
| 190 | §2.4 title: "Graph Learning on Dependency Structures". |
| after 205 | "These studies show… fewer examine whether performance originates from the graph representation…" is appended to §2.4's first paragraph. |
| 206–214 | The last paragraph is replaced by a new **§2.5 Positioning of This Study** containing the advisor's text, including the optional closing sentence for the call. **Deviation:** the deleted paragraph was the only place Arcuri & Briand (2011) was cited. Round 11 (minor 7) had restricted that citation to the paired non-parametric testing it supports, so it now sits on the Wilcoxon sentence in §5.3. |

Round 10 changed §2 after v8, and that text is kept. It added the error-propagation paragraph to §2.1 (Abdelmoez, Popic, Cortellessa & Grassi, Hiller, HiP-HOPS, AADL EMA) and Eadro.

## Paper-wide notes

| Note | Change |
|---|---|
| 102–134, Contributions | Replaced by the five number-free contributions. The PDF section numbers became `\ref`s (3; 4.4; 5.2 and 6.1–6.3; 5.2 and 6.1 with Eq. (7); 5.1, 6.4 and 7.4), and [32] became `\cite{yigit2025graph}`. The RASSE paragraph and the section roadmap are unchanged. |
| Check, 63–64 (QoS sentence in *Findings in brief*) | **Deviation, restored.** After v8, `f8e8c3c1` replaced *Findings in brief* with a short "Organization of the empirical inquiry" paragraph, and round 10 cited that change as the answer to the referee's length comment (M11). The current version therefore had no *Findings in brief*: without restoring it, the number-free contributions would have left the introduction with no results. The paragraph is back, updated to the round-10/11 state, and the QoS sentence is in it verbatim. See *Findings in brief, what changed from v8* after this table. |
| 76–79 | The citations are split into two sentences as given. |
| 403–405, reliability bound | Applied: "$r = 0.79$–$0.99$ … no more than $\sqrt{r} \approx 0.89$–$0.996$ per fold". √0.9925 = 0.996, so the upper end is 0.996 rather than 0.99. **Deviation:** the claim that every ranker stays below its bound *had* been verified, and against the stricter $r$, including the rate-weighted reference. `data/benchmarks/idyn_rate_expansion.json` lists `folds_above_seed_mean_bound` as empty for all 15 rankers. The sentence therefore keeps it in a correct form: "Every ranker in Table 6 stays below even $r$ on every fold; per-fold values are in the replication package." The reconciler now checks all six numbers, √r included. |
| 482–483 and 1084, link | `\sagexperimentsurl` now points at `tree/jss-submission/docs/research/jss/experiments`. Round 10 had pointed it at a `v1.0-jss` tag, which exists on GitHub at `370aad7f`. The new tag is to be created on the final submission commit, at submission. |
| after 544 / 1016–1017, "matches" | The definition sentence is added to §5.3 Statistics. **Deviation:** its pointer goes to §7.5, where equivalence and the ±0.076 interval are discussed, instead of §6.1, which does not mention equivalence. The first sentence of §7.5 *Conclusion validity* is replaced as given. Round 10 had already reworded most "matches" claims to "not statistically distinguishable"; the few ranker-to-ranker uses that remain now follow the definition. |
| 635, "baseline priors" | **Already addressed** in round 11. The paragraph is now titled *Hybrid rankers versus the training-free baseline and their base learners*, and "baseline priors" appears nowhere in the manuscript or the supplement. |
| 666 | "Declared rates and payload sizes, rather than QoS policies, …" |
| Table 6 caption | The duplicated sentence about the rate-weighted expansion is deleted. |
| 871–873 | Replaced as given ("…its deficit is not a matter of missing inputs…"). |
| 924, energy | **Already addressed** in round 10. The text now reads "about 11 ms, an estimated 0.086 mWh, 0.31 J". That is correct, since 0.31 J / 3600 = 8.6×10⁻⁵ Wh = 0.086 mWh, so it was kept rather than replaced by "well under 0.001 Wh". If you prefer the shorter wording, it is a one-line change. |
| 948–954 | Replaced by the shorter example. Its "first priority" claim agrees with §7.6, whose item 1 is validation against real failures. "SAG" had already been corrected to "SaG". |
| 1034–1036 | Replaced as given. |
| 1081–1084, Data Availability | Appended: "The script `reproduce/reconcile_manuscript.py` mechanically verifies 1,711 reported figures against the released artifacts." The reconciler now fails if this number differs from its own count, so the number cannot go stale. |
| 1086–1088, AI declaration | Replaced as given. The rest of the sentence was turned into a semicolon list so it still parses: "…verified against the results; for LaTeX typesetting and formatting assistance; and for developing analysis scripts in the replication package. They used Grammarly for…". |

### *Findings in brief*: what changed from v8

Every number below comes from the current results text:

- the reverse-edge control (direction +0.041, not significant; the derived graph adds +0.072 beyond it, Holm p = 0.014);
- "not statistically distinguishable" in place of "match or exceed" (round-10 M6);
- the corrected-prior hybrids (+0.136 / +0.107), which do not differ from their base learners, and the InDeg-prior learners, which reproduce InDeg (round-11 M2);
- the closing sentence on simulator approximability, kept from the *Organization* paragraph;
- "about 40–45%" for 80% recall, since InDeg needs 45% and GAT-P-QoS needs 40%.

Restoring the paragraph partly reverses round-10 M11. The page count is unchanged at 35, so the length concern behind M11 does not apply.

## Abstract (follow-up request)

At the author's request, the abstract is now the v8 abstract (`8dd4a536`), not the round-11 version. Four items from the round-10/11 referee responses are woven into its second paragraph, and the rest is verbatim:

| Insertion | Commitment |
|---|---|
| "$+0.072$ above the same model with reverse edges on the raw multigraph" | Round 11, M1: the direction control. The reconciler quotes this phrase. |
| "whereas on that graph the registered primary contrast against a training-free baseline was null" | Round 10, M1/M7: the primary null is stated. It replaces "whereas pure learned models on the raw architecture graph do not outperform a training-free baseline". |
| "($\rho = 0.799$; exploratory)" | Round 11, M3: Eq. 7 is labeled exploratory. |
| "…the training-free baseline but not their base learners" | Round 11, M2: the hybrid gain belongs to the comparator. |

To stay within the 250-word limit (it is now exactly 250), "little additional predictive signal beyond that provided by explicit dependencies" became "little predictive signal beyond explicit dependencies". "Matched by counting direct dependents" stays, because it agrees with the definition of "matches" now in §5.3.

## Keywords (follow-up request)

At the author's request, the keywords are now the v8 list, minus "empirical study": *Dependency graphs; cascading failures; publish–subscribe; graph neural networks; graph learning; software architecture; dependability*. The v8 list has eight entries, but the JSS Guide for Authors allows 1 to 7. Round 10 had met that limit by merging the two learning terms into "graph representation learning". This version keeps both of the advisor's learning terms and drops instead the keyword that adds least for indexing.

## Highlights (follow-up request)

At the author's request, `highlights.tex` is now the v8 version (`8dd4a536`), verbatim. All five bullets are within the 85-character limit (75–80 characters), and every number still matches the current results (0.748, 0.635, 0.764, 0.830). Two referee-driven wordings are given up here, though both points remain in the abstract and the body:
- **Highlight 2.** Round 11 (M1) had restated it as the reverse-edge control, "+0.072". The v8 bullet's 0.748 vs 0.635 is the uncontrolled gain, but most of it survives the control: +0.072 of +0.113.
- **Highlight 5.** Round 10 (M4) had said learning beats "only a weak baseline".

Highlight 3's "matches" agrees with the definition now in §5.3.

## Cover letter and extension document (new)

- **`latex/cover_letter.tex`**, compiled to `cover_letter.pdf`. It uses your text, addressed to the VSI: AI4MSS guest editors, together with the RASSE extension sentence.
  - **Deviation:** "can replace 12.7 CPU-hours of simulation" became "a training-free approximation computed in milliseconds ranks components nearly as a queue-flow simulator does (Spearman ρ = 0.830), a simulator that takes 12.7 CPU-hours to label the corpus". The paper reports ρ = 0.830 as exploratory agreement, not as a substitute.
  - It also adds a sentence saying the manuscript is not under consideration elsewhere, plus the Zenodo DOI.
- **`latex/extension_statement.tex`**, compiled to `extension_statement.pdf`. This is the call's "document outlining the improvements" and has four parts:
  - what RASSE 2025 contained;
  - what the manuscript retains (the graph model, Rule 1, and the closed-form score family as the `Topo-QoS` baseline);
  - a section-by-section table of new content;
  - a statement that the present evaluation supersedes RASSE's implicit recommendation of the closed-form score, as §1.4 already says.
- Both files are in `make zip`.
- The IEEE PDF of the RASSE paper is **not** in the repository, because of IEEE copyright on a public repo. Upload it to Editorial Manager separately from `~/Documents/PhD/IEEE RASSE 2025/IEEE_RASSE_2025.pdf`.

## What changed since the v8 you reviewed

The full records are in [response_round10.md](response_round10.md) and [response_round11.md](response_round11.md). In brief:

- **`f8e8c3c1` (hand edit).**
  - The "Architecture–Code Gap" coinage is gone: §1.1 now names the gap descriptively.
  - *Findings in brief* was replaced by an *Organization* paragraph; this revision restores it, as described above.
  - §7.4 now describes the two-tier triage protocol.
  - "Predictive skill" became "statistically distinguishable" in the conclusion.
- **Round 10 (text only).**
  - The primary registered contrast is stated as null everywhere, and the headline findings are labeled registered secondary or exploratory.
  - All three simulators are said to be well approximated by low-order functions of the dependency graph, so no regime in which learning beats an aligned analytical ranking could appear.
  - "Matches" became "not statistically distinguishable; equivalence at ±0.05 not established" where the evidence is non-significance.
  - Hybrid claims now sit next to the facts that the baseline is weak, carries the articulation defect, and is nested in the hybrids, and that neither hybrid differs from its base learner.
  - §6.4 corpus-cost labels were corrected.
  - New §2 literature was added (error propagation, Eadro, Chen et al., Lv et al., Shchur et al., Huang et al., Arcuri & Briand).
  - The energy value became 0.086 mWh.
- **Round 11 (Amendment 16, two new experiments).**
  - **Reverse-edge control:** `GAT-QoS-R` reaches 0.676. Direction recovers +0.041, which is not significant, and the derived graph adds +0.072 beyond it (Holm p = 0.014). The representation claim survives the direction control, and this is now in the abstract, highlight 2, §6.2, the new Table `tab:a16`, §7.1, §7.5 and §8.
  - **Corrected-prior hybrids:** they still beat the corrected baseline but not their own base learners, so the hybrid gain belongs to the weak comparator.
  - **InDeg-prior learners:** they reproduce InDeg to within ±0.012.
  - **One definition of what rankers read:** separation is guaranteed between inputs and labels, not between graphs.
  - **Figure 5B:** now plots the rate-weighted expansion.
  - **Labels:** Eq. 7 is labeled exploratory throughout.

## Points from the email

- **Multiple modelers and real-system experiments.** Not feasible before submission. Both are §7.6 future-work items 1 and 3, and §7.5 and RQ3 state that the five system models come from a single modeler.
- **A full re-read for consistency.** The redline and the regenerated `manuscript.md` (v9) are prepared for exactly this, and the reconciler re-checks every quoted figure after each edit.
