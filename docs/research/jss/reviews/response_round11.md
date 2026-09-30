# Response to the round-11 referee report

This answers [review_2026-09-30_round11.md](review_2026-09-30_round11.md). The revision is on branch
`jss-revision-round11`, based on `a0b4cc59`.

We ran the two experiments the referee asked for as **Amendment 16** in
[PREREGISTRATION.md](../PREREGISTRATION.md). It was committed (`c094c33c`) before anything was
trained, together with its contrasts and decision rules F8–F10.

| Item | Location |
|:---|:---|
| Sweep target | `make -f reproduce/Makefile rq-amendment16` |
| Analysis script | [reproduce/referee_round11.py](../../../../reproduce/referee_round11.py) |
| Artifact | `data/benchmarks/referee_round11_amendment16.json` |
| New table | main Table `tab:a16` |

M3 is handled in text, as the referee allowed.

We thank the referee. Both experiments settled their question, one in each direction:

- **Direction control (M1): the representation claim survives.** A matched-capacity GAT that passes every
  raw-graph edge in both directions (`GAT-QoS-R`, 0.676) recovers +0.041 of the +0.113 gain. That is not
  significant (Holm p = 0.064). `GAT-P-QoS` still beats the control by **+0.072 [+0.034, +0.113], 10/12
  folds, Holm p = 0.014** (rule F8a). The dependency graph therefore adds beyond edge direction. Reverse edges
  also *cost* transfer: `GAT-QoS-R` scores 0.744 zero-shot, against 0.805 for `GAT-QoS`.
- **Corrected-prior hybrids (M2): the hybrid gain is the comparator's.** With the articulation defect
  corrected, both hybrids still beat the corrected baseline: **+0.136 and +0.107**, Holm p = 0.0059 and
  0.0073, 11/12 folds each (rule F9a). Neither beats its own base learner (+0.034 and +0.018, Holm p = 0.94;
  F9c false for both). The paper now says plainly that the hybrid gain belongs to the weak comparator, not
  to the learned correction. With `InDeg` as the prior instead, both learners reproduce `InDeg` to within
  ±0.012 (0.763 and 0.759 against 0.764). A learner handed the reference adds nothing to it.

## Major comments

| # | Comment | What changed | Where |
|---|---|---|---|
| M1 | The representation claim rests on a confounded comparison | **Run.** Reverse-edge control `GAT-QoS-R` (Amendment 16, F8), with shared weights and 429,992 parameters. Result as above; rule F8a applies. Every place that named the confound now states the controlled result instead: abstract, highlight 2, contribution 1, §3.4, §6.2 (new paragraph and summary), §7.1, §7.5 and §8. §7.5 keeps one limit: the control shares weights across directions, so a direction-typed variant was not tested. | Abstract, highlights, §1.4, §3.4, §6.2, Table `tab:a16`, §7.1, §7.5, §8 |
| M2 | The hybrid result is headlined, and the corrected-prior retrain is cheap | **Run.** Both hybrids retrained with the AP-restored prior (F9), and both native learners given the `InDeg` prior (F10). Rules F9a, F9c (false) and F10 applied. The abstract now reads "outperform that baseline, even with its defect corrected, but not their own base learners". The §6.1 paragraph is retitled *Hybrid rankers versus the training-free baseline and their base learners*. Contribution 3, §7.2 and §8 say the gain belongs to the comparator. The corrected-prior hybrids also transfer worse than their base learners zero-shot (0.702 and 0.668 against 0.760 and 0.805; §6.3). | Abstract, §1.4, §6.1, Table `tab:a16`, §7.2, §8 |
| M3 | The Eq. 7 recommendation is post hoc | **Labelled, not confirmed.** Eq. 7 is marked "exploratory" in the abstract, in Table 11's ranker cell and in the Tier-1 protocol ("an exploratory result, not confirmed on new data"). A confirmation corpus remains future work (§7.6). | Abstract, §7.4, Table 11, §7.6 |
| M4 | Inconsistent definition of what rankers read | There is now one definition. $G_{\text{analysis}}$ is $G_{\text{structural}}$ plus the derived `DEPENDS_ON` edges and their node properties. Its App–Lib projection is what the analytical rankings and `-P` learners read. The raw-graph learners pass messages over $G_{\text{structural}}$ with features from $G_{\text{analysis}}$. §4.4 and §7.5 now say the guaranteed separation is between inputs and labels, not between graphs. The Figure 2 caption is fixed to match. | §1.2, §3.1, §3.4, Figure 2, §4.4, §7.5 |

## Minor comments

| # | What changed |
|---|---|
| 1 | §6.1 "match or exceed" becomes "are not exceeded by any learned model". |
| 2 | Removed the empty "; )" and "Arm N" from §6.2. "Proximate" becomes "term-aligned" in §7.5. |
| 3 | §3.2 now says the $I_{\text{dyn}}$ QoS null "may be informative", because its sensitivity to QoS was not measured. |
| 4 | Figure 5B now plots the rate-weighted expansion (Eq. 7). It recovers 0.81 of the critical set at 20% and 0.96 at 30%. §6.1 quotes both values and the reconciler checks them. The legend moved below the panels. |
| 5 | Dropped "upper bound" from the §6.4 energy estimate. |
| 6 | The abstract now reads "the best analytical ranking aligned with each simulator". |
| 7 | [78] is now cited only for the paired non-parametric testing it supports. Effect sizes are reported as Δρ with bootstrap CIs. |
| 8 | Deferrals now point to the supplement: notation (`supp:notation`), per-scenario composition (`supp:corpus`), identification metrics (`supp:identification`) and the Topo baseline (`supp:baselines`). |
| 9 | §2.1 now says model-based approaches need failure-mode annotations, propagation probabilities or operational profiles, which manifests do not carry. |
| 10 | §7.5 reports the negative-ρ seed runs: 1 of 60 `HGT-QoS` runs, 4 of 60 `HGT-P-QoS` runs, and none for either GAT. Per the plan, they stay in the per-seed means. |
| 11 | Fixed "1,321" in the Figure 5 caption. |
| 12 | Highlight 2 now reads "Derived dependency graphs beat reverse-edge raw graphs for graph attention (+0.072)." |

## Deviations recorded in Amendment 16's results log

- **Gate G0.** The Hybrid-HGT comparator differs from its published per-seed ρ by at most 1.2 × 10⁻⁴. Its
  means are unchanged at three decimals. It enters no registered contrast. The other five comparators
  reproduce exactly.
- **Zero-shot launch.** The first zero-shot launch failed at import because `PYTHONPATH` was unset. It was
  relaunched unchanged before any result existed.

## Not done, stated as limitations

| Requested | Why | Where |
|---|---|---|
| M3 confirmation corpus | Needs Neo4j cache builds and new $I_{\text{dyn}}$ labels; Eq. 7 is labelled exploratory instead | §7.6 |
| Direction-typed reverse-edge control | Would add parameters; the registered control shares weights | §7.5 |
| Real-system fault injection, second modeler (round 10) | Unchanged | §7.5, §7.6 |

## Checks

| Check | Result |
|---|---|
| `make` | 35 pages (supplement 44); 0 undefined references or citations |
| Abstract / highlights | 250 words; ≤ 85 characters each |
| `reproduce/reconcile_manuscript.py` | 1,709 figures match (new `check_amendment16`: Table `tab:a16` and the F8/F9/F10 prose; abstract and Figure 5B quotes) |
| `render_manuscript_md.py --check`, `check_doc_links.py docs/` | up to date; all links resolve |
| `pytest -m "not integration"` | 1,428 passed (3 new reverse-edge tests; 3 new arms in the capacity-parity test) |
