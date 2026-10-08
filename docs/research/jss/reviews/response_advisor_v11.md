# Response to advisor notes v11 (e-mail of 2026-10-08, cover_letter.docx)

**Branch:** `jss-advisor-v11-revision`, from `main` at `1e2fb501`.

## Which version the notes were written against

`manuscript-v11.pdf` is byte-identical to `latex/manuscript.pdf` at `main` (`1e2fb501`), and `manuscript-v11.md` is byte-identical to `manuscript.md` at `0a36ccbc`. The notes were therefore written against the current text, and every line number maps directly. Nothing had to be merged.

`~/Downloads/jss_v11_to_v12_redline.html` marks every change. `manuscript-v12.md`, `manuscript-v12.pdf` and `cover_letter_v12.pdf` are the new reading copies.

**Status:**

| Check | Result |
|---|---|
| Manuscript | 30 pp, unchanged |
| Abstract | 248 words as rendered (a whitespace count, as Word does), 240 by the reconciler's count (math as one word); limit 250 |
| `reproduce/reconcile_manuscript.py` | 2,032 figures match. The abstract's direction-control quote was re-keyed to the new wording |
| `scripts/check_doc_links.py` | OK |
| Cover letter | 1 page (10 pt, 2 cm margins) |

## Notes

| Note (v11 line) | Change | Where |
|---|---|---|
| Abstract: "also under sum aggregation" | Adopted verbatim: "+0.072 above the same model passing raw-multigraph edges in both directions, and the dependency graph also wins under sum aggregation". This is accurate. The +0.072 is F8b (`GAT-P-QoS` vs `GAT-QoS-R`, where `-R` adds the reverse of every raw edge). The sum-aggregation claim is §6.2's +0.239 / +0.064 / +0.289 | Abstract |
| Abstract: "exploratory" was dropped | Restored: "(ρ = 0.799; exploratory)". Round 14 had removed it (M10, "over-hedging"); the advisor's request reverses that | Abstract |
| RQ terms missing from Contribution 3 | Adopted: "…QoS information, aggregator, model family and the number of training architectures." | §1.5 |
| "Baseline" for InDeg/Reach (81, 163, 198, 242) | Adopted in all four places: "analytical rankings", "structural metric", "analytical rankings", "structural metrics". "Baseline" now refers only to `Topo-QoS` in the main text, apart from §2.3's literature use ("homogeneous baselines") | §1.4, §2.2, §2.4, §3.3 |
| 526: payload sentence | Adopted verbatim | §6.1 |
| 701: "ground truth" | Changed to "as a label for cascade impact". The one remaining "ground truth" (§2.1) describes simulation methodology in general (Sargent), not our labels | §7.1 |
| 90: CI/CD defined late | Defined at first use, in RQ4: "continuous integration and delivery (CI/CD)". The later expansion at 674 becomes plain "CI/CD". The old expansion, "continuous integration (CI/CD)", also left out delivery | §1.4, §6.4 |
| 782: tense | "exceeds" → "exceeded" | §7.5 |
| 847: "the only ones" | Adopted verbatim | §7.5 item 3 |
| 878: "weak" | Adopted: "Learning outperformed only the registered training-free baseline, QoS-weighted betweenness, …" | §8 |
| 112: RASSE sentence | **Adopted with one factual deviation.** The note says the conference paper reports betweenness and articulation points separately, with no weighted combination. The RASSE PDF does combine them: Eq. (3), CS(v) = α·C_B(v) + β·AP(v), with α = 0.7 and β = 0.3, used for its Table I scores. The new sentence keeps the advisor's structure and the more accurate "illustrated by removal experiments on a synthetic example and two ROS 2 benchmarks". It adds "combined there in a weighted criticality score" and says Eq. (5) "is a QoS-weighted version of that combination". The following sentence's "the closed-form score family of [30]" became "this combination" | §1.5 |

**Length (follow-up).** The two notes above took the abstract to 251 rendered words (the author measured 256). To stay under 250 by any count, the direct-dependents sentence went back to the Abstract_v9 wording: "Counting direct dependents (ρ = 0.764) is not significantly different from the learned model." This drops "that is, afferent coupling" and "which approaches it as training architectures are added". Afferent coupling is still named in §1.3 and §8, and the learning-curve trend in §8.

**Advisor's abstract_v11_2.docx (follow-up).** This version is applied verbatim; the rendered text matches the .docx word for word. It makes three changes to the cut version:
- the first sentence reads "yet architects must identify components with the greatest cascading impact before deployment, without telemetry";
- "Counting direct dependents (afferent coupling, ρ = 0.764)";
- "not yet when it helps".

The reconciler's abstract quote now accepts the "afferent coupling," prefix.

### Abstract_v9 items not restored

The e-mail names two items, both done. Round 14 also changed three other parts of the Abstract_v9 text. These are left as round 14 has them, for the advisor to decide:

- the zero-shot sentence "Learned models transfer zero-shot better than the training-free baseline (ρ ≈ 0.81 vs. 0.53), although dependency counts rank higher", removed under round-14 M5 (no Topo-QoS comparison in the abstract);
- "Learning outperformed the training-free baseline but not the strongest analytical references", removed under the same item;
- "deployment manifests" → "declared publish–subscribe architecture models" and "hand-authored" → "stylized", under round-14 M8 (no manifest importer is evaluated).

## Cover letter

`latex/cover_letter.tex` now carries the advisor's `cover_letter.docx` text, with one change. His sentence "The journal version retains the multigraph model and the subscriber-to-publisher dependency rule. All other content is new" would describe the betweenness–articulation score as new. It now reads "…retains the multigraph model, the subscriber-to-publisher dependency rule and, as its training-free baseline, a QoS-weighted version of the conference paper's betweenness–articulation score. All other content is new…". This matches `extension_statement.tex`, which already lists that score as retained.

As he suggested, the letter is set in 10 pt with 2 cm margins and fits on one page. The date is `\today`, replacing "[Submission date]".

`extension_statement.tex` had two stale pointers: "reference [32]" (now [30]) and "Section 1.4" (the RASSE paragraph is in §1.5). Both are fixed.
