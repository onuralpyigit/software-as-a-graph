# Response to advisor notes v9 (notlar_v9.docx, Abstract_v9, Introduction_v9, conclusion_v9)

**Branch:** `jss-advisor-v9-revision`, from `main` at `8bf4c05b`.

**Status:**

| Check | Result |
|---|---|
| Manuscript | 27 pp, unchanged |
| Build | No undefined references. The only overfull boxes are from the pinned-commit URL (round 13); none come from this revision |
| `reproduce/reconcile_manuscript.py` | 1,817 figures match (was 1,815: the new §7.4 and abstract sentences add two quoted figures; the Data Availability count is updated) |
| `tests/test_reconcile_guards.py` | 4 passed |
| `pytest -m "not integration"` | 1,440 passed |
| `scripts/check_doc_links.py` | OK |

The "Note" column uses line numbers from the v9 PDF. Departures are marked **Deviation**, and points already resolved are marked **Already addressed**.

## Which version the notes were written against

`manuscript-v9.md` is byte-identical to `manuscript.md` at **`583c9a16`** (2 Oct). The new Abstract, Introduction and Conclusion were therefore written against that text. Three revisions have landed on `main` since then:

- **Round 12** (`9bdb9842`, Amendment 17). Three controls were added:
  - **F11:** with every input feature that computes part of the reachability simulator removed, the dependency-graph GAT still beats an equally stripped reverse-edge control by **+0.231** (Holm p = 0.0068).
  - **F12:** learners started from Eq. 7 do not improve on it. As an extra input it leaves a gradient-boosted model at 0.830; as a prior it leaves a GAT at 0.812 (−0.018).
  - **F13:** node-order permutations have a per-fold spread of about 0.044.
- **Condensation** (`280f2e42`):
  - The body went from 32 to 24 pages. It is now 27 pages after round 13.
  - *Findings in Brief* became its own subsection (§1.3).
  - §2.3 (AHP) was folded into §3.2, and §7.3 (*Restatement versus Prediction*) was folded into §7.1–7.2. Threats and Limitations are now **§7.4 and §7.5**.
  - The control-arm tables moved to Supp. S42; the body keeps a 14-row digest, `tab:controls`.
- **Round 13** (`45c87045`, text only). It made four corrections found against the code, which matter for your rewrites:
  1. **I_dyn reads no payload.** Every message has the same size, so declared *rates* carry the signal, not "rates and payload sizes".
  2. **The Topo-QoS defect is a missing key, not an index mismatch.** The scorer reads the articulation flag under `ap_c_score`, which the cached metrics do not contain, so it falls back to zero.
  3. **Rules 2–4 and 6 (infrastructure) are formalized but not evaluated.** Contribution 1 now claims only Rules 1 and 5.
  4. **Scope.** All three simulators are dominated by first-order propagation by construction, and every learned result comes from one fixed configuration trained on eleven architectures. The claims are bounded accordingly.

  Round 13 also re-scoped several §1.1 citations that did not support their sentences. §2.1 now cites shortcut learning (Geirhos et al.) for the restatement problem.

The redline `~/Downloads/jss_v9_to_v10_redline.html` marks every change from your v9 to this version, including the three revisions above. `manuscript-v10.md` and `manuscript-v10.pdf` are the new reading copy.

## How the rewritten sections were merged

Your text is the base everywhere. Where it reverts a round-12/13 change, the current content is kept in compact form, and each such case is listed below.

### Abstract (rewritten; your version used)

Applied as written, with these changes:

| Change | Why |
|---|---|
| "+0.072 above the same model with reverse edges on the raw multigraph" re-inserted in the first result sentence | A v8 referee commitment (edge-direction control), and the reconciler checks it |
| "; learners that start from the formula do not improve on it" added after the 0.799 clause | Round-12 F12 result |
| Last paragraph opens "On these simulators, whose propagation is first-order by construction, …", replacing "In this simulated corpus" | Round-13 scope commitment |
| Removed to stay within 250 words (now exactly 250): ", exposing cascades through shared topics and libraries"; "from" (2nd); "therefore"; "strong"; "practical," | JSS limit |
| "Counting direct dependents reaches ρ = 0.764, not significantly different…" → "Counting direct dependents (ρ = 0.764) is not significantly different…" | Same meaning, shorter |

Not re-inserted, to respect the word limit: round 12's "its gain survives removing input features that compute part of the simulator". It remains in §1.3, §7.1 and the Conclusion.

**Keywords:** your list, 7 terms ("graph learning" replaced by "reliability").

### Introduction (tracked changes)

A word-diff showed that your Introduction is the `583c9a16` text plus exactly your tracked edits. Each edit was therefore applied to the current Introduction wherever its sentence still exists. Untracked passages that rounds 12/13 had rewritten keep the current text: the §1.1 citations, the RQ wording and contribution 1.

| Your edit | Status |
|---|---|
| §1.1 like → such as; move → propagate; depend → rely; needs → requires; long supported → been used for; arises → stems; derived from → based on | Applied where the sentence survives. The condensed §1.1 no longer has the "like" or "use/consume" sentences |
| "…architects must identify, from configuration manifests alone, the application services whose failures have the greatest cascading impact" | Applied |
| "SaG addresses this question by deriving explicit dependency graphs … before deployment" (end of §1.1) | Applied, replacing "SaG investigates this question" |
| §1.2 "…rather than predictors, so circular agreement is not mistaken for predictive skill" | Applied, merged with round 13's "analytical truncations of the simulation rule" |
| §1.2 one → a single; every → each; strictly removed; partial-correlation sentence | **Already addressed.** The condensation removed that paragraph; partial correlations are reported in §6.1 |
| *Findings in brief*, lead sentence | Applied as the opening line of §1.3 |
| "registered co-primary contrasts (… with and without the QoS channel, against the training-free baseline on that graph) were null (QoS-weighted: …)" | Applied |
| "statistically distinguishable" → "significantly different" | Applied, and made paper-wide (see below) |
| "Learning adds measurable value relative to the training-free baseline, but not beyond the strongest analytical references." | Applied as the third block's title |
| Hybrid sentence: "(+0.103 and +0.130, each on 11 of 12 folds, also when its articulation defect is corrected)… given the direct-dependent count as their prior, they reproduce that count" | Applied |
| "even higher", "the ordering" | Applied |
| "Each simulator is closely tracked by a low-order structural ranking, so a regime … did not arise in this benchmark" | Applied. **Deviation:** it is preceded by round 13's scope sentence (first-order by construction; eleven architectures, one fixed configuration) and followed by "whether it does under non-first-order propagation is not tested here (§7.2)" |
| "…for routine use: it needs no training and runs in milliseconds" | Applied |
| "Declared publication rates and payload sizes carry measurable signal…" | **Deviation:** not restored. The condensation removed this sentence, and the payload half is wrong (round-13 correction 1) |
| "QoS" expansion dropped in §1.4; "LOSO" spelled out; oracles → simulators | Applied. QoS is expanded only in §1.1, LOSO was already spelled out, and "second and third oracles" became "simulators" |
| Contribution 1: "…extending the Application-level rule of [32] with library-mediated and infrastructure dependencies" | **Deviation:** the round-13 wording is kept. Only Rules 1 and 5 are evaluated, and the infrastructure rules are formalized but not evaluated |
| Round-12 sentences on F11 (+0.231) and F12 (−0.018) | **Kept.** They are not in your version because they postdate it, and both are reconciler-checked |

### Conclusion (rewritten; your version used)

Applied as written, with these changes:

| Change | Why |
|---|---|
| "which are dominated by first-order propagation by construction" after "For the reachability and queue-flow simulators" | Round-13 scope |
| "a gain that survives controls for edge direction and for features that compute part of the simulator" | Round-12 F11 |
| "; learners that start from that approximation matched it at best" after the 0.799 clause | Round-12 F12 |
| "These conclusions hold for learners trained on eleven synthetic architectures in one fixed configuration, against simulators rather than observed failures." (end of ¶3) | Round-13 scope |
| "Sections 7.5 and 7.6" → `\ref`s, which now render as 7.4 and 7.5 | Renumbering after the condensation |
| "; re-testing these findings on a corpus generated after the analysis was frozen, under non-first-order simulators and against real outages, is the natural next step" | Round-13 commitment (§7.5 item 1) |

## Line-numbered notes (notlar_v9)

| Note | Change |
|---|---|
| Keywords | Applied: "reliability" added, and "graph learning" dropped to stay within the 7-keyword limit |
| 169–170 | **Already addressed.** Round 13 replaced the sentence with "The concern is not new: simulation methodology … learned models exploit shortcuts that encode the labeling rule [Geirhos]. What we add is an operational check for simulator-labeled ranking benchmarks." This answers the shortcut-learning objection directly. |
| 445–446 | Applied: "…stays below $r$, and therefore below $\sqrt{r}$, on every fold." The reconciler still verifies the claim from `idyn_rate_expansion.json` |
| 549–553 | Applied: "Due to an implementation defect … so every table reports QoS-weighted betweenness alone. The defect slightly favors the baseline: restoring the term lowers Topo-QoS from 0.553 to 0.533, so it does not bias comparisons toward the learned models or the references. The hybrids' prior p(v) inherits the defect." The corrected-prior hybrids now point to `tab:controls`, F9. **Deviation:** the cause is the missing `ap_c_score` key, not an index mismatch (round-13 correction 2) |
| Table 4 caption | Applied: "…so it acts as pure QoS-weighted betweenness; the indented row restores the term (ρ = 0.533), so the defect slightly favors the baseline, and no comparative conclusion changes." The same cause correction applies |
| 592–594 | **Already addressed.** The condensation removed that sentence. Your wording ("not established for any pair of independently derived rankers; it holds only for learners given a reference as their prior, which reproduce it") is used in §7.4 (note 1111–1113) |
| 644, Figure 4 caption, 903–905 | Applied: "not statistically distinguishable" / "not distinguishable" → "not significantly different" **throughout the main text**. That covers Abstract, §1.3, the Figure 4 caption, §7.1, the Table 10 row and §7.4; none remain. The §5.3 definition now reads: "'Not significantly different' does not mean statistically equivalent; equivalence is claimed only where a two one-sided test at ±0.05 passes, and where the paper says one ranker 'matches' another, it means the two are not significantly different." |
| 699–700 | Applied: "The hybrid gain therefore reflects the comparator, not the learned correction." |
| 819 | **Already addressed.** The condensation removed the sentence |
| 907–908 | Applied as "appears to be the principal source of predictive signal". The edge-direction clause already sits in the preceding sentence, which since round 12 also mentions the feature-removal control |
| 927–928 | **Already addressed.** Round 13 rewrote the paragraph ("…multi-hop, non-linear mechanisms that no low-order approximation captures… None of the three simulators implements them"), and "did not exercise" no longer appears |
| 977 | **Already addressed.** The condensation reduced the sentence to "publish–subscribe afferent coupling costs a fraction of that; its value is cost, not predictive skill" |
| 1050–1053 | Applied, as two sentences: "I* and I_dyn are well approximated by low-order functions of the dependency graph (a first-order expansion recovers 0.808 of I* and a rate-weighted one 0.830 of I_dyn), and I_comp's terms track degree and betweenness, which rank highest on it (0.70–0.72). The benchmark therefore did not exhibit a regime…". The values 0.70–0.72 are Topo-QoS 0.702 and raw total degree 0.719. Round 13's "first-order by construction" clause before it and "Its absence here is a property of these oracles…" after it are kept |
| 1111–1113 | Applied: "Twelve folds … underpowered for equivalence at ±0.05: it is not established for any pair of independently derived rankers, and holds only for learners given a reference as their prior, which reproduce it (Table `controls`, F10); on I_dyn, the interval of the learner started from Eq. 7 also lies within that margin. Where the paper says one ranker 'matches' another…". **Verified:** Supp. S42 reports the InDeg-prior hybrids "equivalent at ±0.05, TOST p < 0.001" |
| 1149 | Applied, in the Conclusion merge: "The registered co-primary contrasts were null…" |
| 1161 | Applied, in the Conclusion merge: "Learning outperformed the training-free baseline but not the strongest analytical references." |

**Highlights:** "hybrids beat a weak baseline" → "hybrids beat the baseline", following your softening of "weak comparator". The line stays within 85 characters.

## Open item: shortening notes

Your email mentions length suggestions up to page 12, but they were not in the attached files. Note that the body was condensed from 32 to 24 pages after the v9 you read; it is 27 pages after round 13. Some of the suggestions may therefore already be covered. Please send them, keyed to `manuscript-v10.pdf` if possible.
