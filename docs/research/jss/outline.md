# Software-as-a-Graph — JSS submission reading map

> **[`latex/`](latex/) is the authoritative manuscript.** [`manuscript.md`](manuscript.md) and
> [`sections/`](sections/) are *generated* from it by
> [`reproduce/render_manuscript_md.py`](../../../reproduce/render_manuscript_md.py), which takes
> section, table, figure and citation numbers from the compiled `.aux`/`.bbl`. This file maps what
> each part of the paper argues and what backs it.
>
> **Regenerated after the 2026-09-25 revision** ([`reviews/review_2026-09-25.md`](reviews/review_2026-09-25.md))
> and Amendment 7 of the registered plan, and again after merging main's parallel revision (the
> Amendment 8 attribution controls). Earlier versions of this file carried the pre-Amendment-7
> thesis ("declared QoS contracts drive the gain"; "hybrids are the most accurate engines"), which
> the registered controls overturned.

* **Target journal:** Journal of Systems and Software (Elsevier), single-anonymised review
* **Target venue:** Special Issue "AI Techniques for Performance, Reliability, and Sustainability of
  Modern Software Systems" (VSI:AI4MSS); deadline 30 September 2026
* **Title:** *Software-as-a-Graph: Explicit Dependency Graphs Predict Cascading-Failure Impact in
  Publish–Subscribe Systems Before Deployment*
* **Scale:** 27 pages, 9 sections, 12 tables, 5 figures, 100 references; supplement S1–S33
  (30 pages). Take these from the build: `pdfinfo latex/manuscript.pdf`.
* **Build:** zero LaTeX errors, zero undefined references or citations. The reconciler
  (`reproduce/reconcile_manuscript.py`) checks every table figure against its artifact when the
  full results bundle is present (537 before the Amendment 8 merge); in a fresh clone it checks the
  committed corpus and the Amendment 7 artifacts (145 figures).

---

## The thesis

> Making the hidden dependencies of a publish–subscribe architecture explicit is what makes
> cascading-failure risk measurable before deployment. On SaG's dependency projection, a count of a
> component's dependents ranks its simulated impact better than any learned engine, transfers to
> independently authored system models without training, and costs well under a second.

Learned engines are evaluated for what they add on top: hybrids significantly beat the registered
betweenness engine, and models trained only on synthetic data transfer. Controls trace learned
accuracy to per-component features: typing, message passing and the QoS edge channel add nothing
measurable, and gradient boosting on the same features matches the neural engines. Registered controls attribute the closed-form gain to the
projection rather than to declared QoS contracts, so ranking does not require QoS profiles.

---

## Headline figures (all on the Application population, primary oracle I*)

| Quantity | Value | Where |
|---|---|---|
| **InDeg** (direct dependents), LOSO mean ρ | **0.764** [0.674, 0.840]; +0.211 vs Topo-QoS, 12/12, Holm p = 0.0020 | §7.1, Table 7 |
| Reach / Reach-QoS, LOSO mean ρ | 0.732 / 0.714 | §7.1, Table 7 |
| InDeg vs learned / hybrid engines | +0.143 (HGT-QoS), +0.129 (GAT-QoS), +0.108 (Hybrid-HGT), +0.081 (Hybrid-GAT), +0.122 (GBM-Feat); 10–11/12, p ≤ 0.005 | §7.1, Supp. S31 |
| Inert-vs-active rule ("has a dependent") | 94% balanced accuracy (LOSO), 97% (system models) | §7.1, Supp. S31 |
| Hybrids vs Topo-QoS (registered) | +0.103 / +0.130, 11/12, Holm p = 0.0068 / 0.0029; omnibus (13 contrasts) 0.041 / 0.019 | §7.1, §6.4 |
| Hybrids vs their own learned engines | +0.035 (p = 0.73) / +0.048 (p = 0.30) | §7.1 |
| Learned engines alone vs Topo-QoS | HGT-QoS 0.622 (+0.069, p = 0.27), GAT-QoS 0.635 (+0.082, p = 0.23) | §7.1 |
| Closed-form gain: registered Topo → Topo-QoS | 0.349 → 0.553 | §7.1.2 |
| …same projection, unweighted betweenness | 0.591 | §7.1.2 |
| …constant topic weight / permuted QoS | 0.595 / 0.559 | §7.1.2 |
| …QoS-independent corpus, QoS effect | −0.035 | §7.1.2 |
| Matched 2×2: typing / QoS inputs | −0.014 [−0.052, +0.023] / +0.073 (10/12, Holm p = 0.127) | §7.2, Table 8 |
| Attribution (A8, exploratory): QoS node columns / edge channel | +0.095 (11/12, Holm 0.024) / −0.023 | §7.2, Table 9 |
| Feature-only GBM-Feat; directionality HGT-QoS-U; capacity GAT-w | 0.642 [0.547, 0.725]; 0.632 (p = 0.91); 0.633 (p = 0.68) | §7.2, Supp. S32 |
| Zero-shot, learned models | GAT 0.831, GAT-QoS 0.805, HGT-QoS 0.760, GBM-Feat 0.757 vs 0.511–0.526 (betweenness) | §7.3, Table 10 |
| Zero-shot, dependency counts | Reach 0.938, Reach-QoS 0.933, InDeg 0.863 (Amendment 7 harness) | §7.3, Supp. S31 |
| Cost | projection + InDeg ≤ 0.06 s; Reach ≤ 0.15 s; simulator ≤ 4.5 s; analysis gate ≤ 79 s; HGT forward 56 ms at 2,000 nodes | §7.4 |

---

## Section map

| § | What it establishes |
|---|---|
| 1 | Motivation, why now (Architecture-as-Code), four RQs, **key findings at a glance**, five contributions |
| 2 | Reliability prediction, fault injection (Filibuster, LDFI), telemetry RCA, static architecture analysis (HAROS, ROSDiscover), graph learning |
| 3 | Typed multigraph, six `DEPENDS_ON` rules, dual views, typed node features |
| 4 | Engines; the primary oracle as implemented (threshold cascade, θ, damping) and its robustness; procedural independence guarantee |
| 5 | Explanation layer, presented as a design proposal |
| 6 | Corpus (with generator QoS–topology coupling disclosed), predictors, metrics, **§6.4 analysis plan and deviations** |
| 7 | Findings 1–5: dependency counts, where the closed-form gain comes from, where learned accuracy comes from (features, not typing or message passing), transfer, cost |
| 8 | Implications for practice and research, instrument guidance (Table 12), threats, research agenda |
| 9 | Conclusion |

## Registered decisions (PREREGISTRATION.md)

Plan and Amendments 1–8. Amendment 7 (2026-09-25, before any of its results) registered the
dependency counts and QoS-attribution controls with rules R1 (learned engines vs the best count),
R2 (QoS content vs multiplicity) and R2′ (generator coupling); all three applied, and the text
follows them. Amendment 8 (2026-09-26, written after its results, exploratory) records main's
attribution controls; it was first committed as "Amendment 7" on a parallel branch and renumbered in
date order at the merge. Every arm of Amendment 7 regenerates without GPU or database:

```bash
PYTHONPATH=. python reproduce/training_free_suite.py all        # gate + baselines + controls + oracle + descriptives
PYTHONPATH=. python reproduce/training_free_suite.py make-variant && \
PYTHONPATH=. python reproduce/training_free_suite.py qos-indep
PYTHONPATH=. python reproduce/training_free_suite.py substrate
PYTHONPATH=. python reproduce/training_free_suite.py cost
PYTHONPATH=. python reproduce/render_amendment7_tables.py        # Supplementary S31 tables
PYTHONPATH=. python reproduce/render_amendment7_figure.py        # Figure 5
```
