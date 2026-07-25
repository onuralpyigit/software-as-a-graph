# Methodology Revision — Findings and Required Draft Changes

**Status:** produced during the pre-submission methodology revision. Every number below comes from a
committed artifact under [results/](../../../results/) and is reproducible with the commands listed
in each section. This document is the change list for `draft.md`; it is not itself a paper section.

---

## Summary

Five defects were found and fixed. Two of them change the paper's headline conclusions, and three
produce new negative results that must be reported rather than omitted.

| # | Defect | Effect on the paper |
|---|---|---|
| 1 | Table 3 compared variants scored on different node sets and different samples | **Understated every learned model by ~0.2–0.35 ρ.** RQ1 conclusion strengthens, but only after correction |
| 2 | LOSO sweep reused stale checkpoints, skipping training entirely | **Published LOSO numbers were not produced by trained models** |
| 3 | The strongest non-learning baseline was absent from the LOSO table | **Topo-QoS ties the GNNs on LOSO ρ**; the "learning is required OOD" claim does not survive |
| 4 | The λ = 0.70 AHP default has no plateau and is beaten by equal weights | §8.3 sensitivity claim is contradicted by measurement |
| 5 | The two ground-truth oracles agree at ρ ≈ 0.25 | Bounds what §5.4/§5.5 can support; §8.2 must be qualified |

---

## 1. The evaluation population was not like-for-like (RQ1)

**What was wrong.** In `reproduce/main_table.py`, the training-free structural baselines were scored
on *every* node of the DEPENDS_ON projection, while the learned variants were scored on the *20 %
test split* of an `{Application, Library}` pool with both types collapsed into a single Spearman.
Two estimators measured on two different samples is not a comparison.

The type-pooling alone was decisive: on `av_system`, HGL scores ρ = 0.81 within Applications but
0.46 when Applications and Libraries are pooled — the Simpson pattern `docs/validation.md §5.0`
exists to guard against. The baselines were unaffected because their key set contained fewer than
three Library nodes, so that stratum was dropped rather than pooled.

**What changed.** All six variants now run through
[`saag.evaluation.metrics.compute_inductive_metrics`](../../../saag/evaluation/metrics.py) on one key
set resolved from the graph and labels only, and share one train/val/test split pinned by node id.
The headline figure is the held-out one; the transductive figure is retained under
`full_population`. Full contract: `docs/validation.md §5.0.1`.

**Corrected Table 3** — Spearman ρ, held-out, `eval_population = application`, mean of 5 seeds:

| Scenario | Topo-BL | Topo-QoS | GL | GL-QoS | HGL | HGL-QoS |
|---|---:|---:|---:|---:|---:|---:|
| AV System | 0.429 | 0.680 | 0.572 | 0.831 | 0.836 | 0.826 |
| Enterprise | 0.276 | 0.674 | 0.867 | 0.309 | 0.916 | 0.916 |
| Financial Trading | −0.067 | 0.430 | 0.623 | 0.604 | 0.504 | 0.550 |
| Healthcare | 0.513 | 0.732 | 0.601 | 0.489 | 0.591 | 0.559 |
| Hub-and-Spoke | −0.021 | 0.750 | 0.605 | 0.784 | 0.756 | 0.685 |
| IoT Smart City | 0.089 | 0.265 | 0.694 | 0.815 | 0.889 | 0.893 |
| Microservices | 0.163 | 0.438 | 0.706 | 0.816 | 0.863 | 0.781 |
| **Mean** | **0.197** | **0.567** | **0.667** | **0.664** | **0.765** | **0.744** |

**Against the previously published pooled means** (Topo-BL 0.233, Topo-QoS 0.612, GL 0.186,
GL-QoS 0.219, HGL 0.411, HGL-QoS 0.377): the baselines barely move, every learned variant gains
0.35–0.48. The published table said a training-free centrality beat every GNN by ~0.2 ρ; measured
correctly, HGL leads it by 0.20.

**Draft changes.** Replace the §8.1 in-distribution numbers. The qualitative RQ1 claim survives and
strengthens, but the reason given in the draft ("interpretable attribution suffices in-distribution")
no longer holds on this evidence — HGL beats Topo-QoS in 6 of 7 scenarios. Rewrite the conclusion to
match the corrected table rather than restating the old one.

Reproduce: `PYTHONPATH=. python reproduce/main_table.py --eval-population application`

---

## 2. The LOSO sweep was not training (critical)

**What was wrong.** `reproduce/loso_all_variants.py` did not clear the per-variant workspace between
runs. `GNNService` restores from any checkpoint it finds, so every run after the first **skipped
training** and scored a model fitted under a different fold configuration.

Measured on `hgl`, identical command, single seed:

| Workspace | Wall time | LOSO ρ |
|---|---:|---:|
| Dirty (stale checkpoints present) | 3.2 s | **−0.576** |
| Clean | 322.6 s | **+0.594** |

A 7-fold × 5-seed sweep finishing in 5 seconds per variant is the visible symptom; the published
LOSO table shows exactly that timing profile.

**What changed.** The workspace and previous `results.json` are removed before each variant runs, in
both `loso_all_variants.py` and `kfold_all_variants.py`.

**Draft changes.** Every LOSO figure in §8.1, §8.3 and the abstract must come from the clean re-run.
The draft currently quotes HGL-QoS ρ = 0.401 while the committed artifact said 0.2904 — both are
superseded.

---

## 3. Topo-QoS was missing from the LOSO table

`Topo-BL` and `Topo-QoS` need no training, so their in-distribution score **is** their
out-of-distribution score — they cannot overfit a scenario they were never fitted to. The published
§8.1 LOSO table omitted them and concluded "the non-learning and homogeneous baselines collapse",
having measured only the homogeneous ones.

**Corrected Table 4 (LOSO)**, after the clean re-run of §2 with all seven variants:

| Variant | Mean ρ | Std ρ | F1@K |
|---|---:|---:|---:|
| Topo-BL | 0.160 | 0.145 | 0.229 |
| **Topo-QoS** | **0.609** | 0.165 | 0.416 |
| RMAV baseline | 0.034 | 0.118 | 0.206 |
| GL | 0.434 | 0.164 | 0.478 |
| GL-QoS | 0.280 | 0.080 | 0.423 |
| **HGL** | **0.597** | 0.258 | **0.548** |
| HGL-QoS | 0.591 | 0.271 | 0.542 |

Against the previously published LOSO numbers (GL 0.010, GL-QoS 0.005, HGL 0.250, HGL-QoS 0.290 —
all produced without training, per §2), every learned variant roughly doubles.

**The headline finding is uncomfortable and must be reported.** `Topo-QoS`, a training-free
QoS-weighted centrality, reaches ρ = 0.609 — statistically indistinguishable from HGL (0.597) and
HGL-QoS (0.591), and it achieves that with no training, no labels and no transfer assumption. The
learned models do win on the critical-set metric (F1@K 0.548 vs 0.416), which is the more
practitioner-relevant quantity, and that is the defensible claim. The claim that heterogeneous
learning is *required* for out-of-distribution generalisation is not supported by this table.

One implementation note worth carrying into the paper: the baseline must be scored on the DEPENDS_ON
projection, not the raw pub-sub graph. Application nodes never route messages, so their betweenness
on the native graph is identically zero and the entire Application stratum becomes constant — its
pooled ρ would then be carried purely by between-type offsets.

**Draft changes.** Add both rows to the §8.1 LOSO table and reframe the RQ1/RQ3 discussion around the
strongest baseline rather than the weakest.

---

## 4. The AHP shrinkage default is unsupported (RQ3 / §8.3)

`docs/structural-analysis.md` claimed Spearman ρ "plateaus in the λ ∈ [0.65, 0.75] range". The
artifact that claim cited did not exist (`docs/internal/` was empty). Measured
([results/ahp_shrinkage_sweep.json](../../../results/ahp_shrinkage_sweep.json)):

| λ | 0.00 | 0.50 | 0.60 | 0.65 | **0.70** | 0.75 | 0.80 | 0.90 | 1.00 |
|---|---|---|---|---|---|---|---|---|---|
| mean ρ | **0.257** | 0.162 | 0.146 | 0.138 | **0.129** | 0.117 | 0.111 | 0.093 | 0.077 |

- ρ is **monotonically decreasing** in λ. There is no plateau anywhere, and the [0.65, 0.75] window
  varies by more than 0.01.
- **Equal weights beat the stated judgement** by 0.128 (λ = 0 vs the λ = 0.70 default).

Two related corrections in the same area:

- **Docs printed the wrong weights.** Shrinkage is applied to the intra-dimension vectors as well as
  the composite, so the shipped Q(v) weights are (0.395, 0.247, 0.193, 0.165), not the
  (0.43, 0.24, 0.17, 0.16) printed in every formula. Both are now tabulated in §11.2.
- **"AHP-derived" overstates the provenance.** CR ≈ 0.000–0.002 on 5×5 matrices indicates matrices
  written from a target weight vector, not elicited. Reframed as stated judgements with a
  consistency check.

**Draft changes.** §8.3's "the λ = 0.70 default is not a tuned artifact" must be replaced with the
measured result. This is a reportable negative finding, and the honest framing — a multi-criteria
decomposition whose *value is explanatory*, not accuracy-maximising — is defensible; the current
claim is not.

Reproduce: `PYTHONPATH=. python reproduce/ahp_sensitivity.py`

---

## 5. The two ground-truth oracles agree weakly

The paper draws Tables 3/4 from `I*(v)` (`FaultInjector`) and the library and stratified analyses
(§5.4, §5.5) from `I_comp(v)` (`FailureSimulator`). `docs/validation.md` said they "correlate only
loosely" without quantifying it. Measured
([results/convergent_validity.json](../../../results/convergent_validity.json)):

| | mean | min | worst scenario |
|---|---:|---:|---|
| Spearman ρ | **0.246** | 0.053 | healthcare_system |
| top-20 % Jaccard | **0.229** | 0.086 | microservices_system |

All seven correlations are positive, which is a weak convergent-validity argument. But at ρ ≈ 0.25
an argument established on one oracle does not transfer to a claim measured on the other. §5.4's
library result and §5.5's stratified ρ range are computed against `I_comp`, while §8.1's table is
`I*` — the draft currently reads as though they support each other.

**Draft changes.** Add the oracle to every results table; state the inter-oracle agreement in §7.5;
either re-run §5.4/§5.5 against `I*` or explicitly scope those claims to `I_comp`.

Reproduce: `PYTHONPATH=. python reproduce/convergent_validity.py`

---

## 6. Additional sensitivity results now committed

[results/threshold_sensitivity.json](../../../results/threshold_sensitivity.json) supplies the
robustness evidence §8.3 promised but never produced. Both sweeps are less flattering than the draft
assumes:

**Propagation threshold** (RMAV Q(v) vs `I*`, 7 scenarios):

| threshold | 0.00 | 0.10 | **0.20** | 0.35 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|---|---|
| mean ρ | −0.033 | 0.076 | **0.123** | 0.162 | 0.161 | 0.164 | 0.166 |

The conclusion *does* depend on the threshold: ρ spans 0.199 across the sweep and the canonical 0.2
default sits near the bottom. At 0.0 the correlation is slightly negative.

**Normalisation:**

| `--norm` | `robust` (rank, default) | `minmax` | `zscore` |
|---|---|---|---|
| mean ρ | **0.129** | **0.324** | **0.324** |

Rank normalisation — the default — costs ≈ 0.195 ρ against magnitude-preserving alternatives, because
it turns the RMAV weighted sum into something close to a Borda count over the Tier-1 metrics.

**Draft changes.** Report both sweeps as stated. Do not claim threshold-independence.

---

## 7. Edge criticality is now measured (closes L8 / G2)

`EdgeCriticality` was declared but never populated; edge labels were `I*(u) × {1.0, 0.1}`, a
projection of node labels through a hand-chosen multiplier.
`FailureSimulator.simulate_edge_removal` now severs each candidate edge — leaving both endpoints
active — and returns the delta against a no-op control.

The control subtraction is load-bearing: `_calculate_impact` is non-zero on a pristine graph
(composite 0.0061 on `av_system`) because topics that already lack a publisher or subscriber count as
lost throughput. Without it every edge inherits that floor as apparent signal.

Result on `av_system`: 4 of 40 candidates carry non-zero impact, all `PUBLISHES_TO`/`SUBSCRIBES_TO`.
`RUNS_ON` edges measure exactly 0.0 — the cascade routes no traffic over them (L5) — even though
bridge detection flags them as non-redundant. That zero means "this model cannot express that link's
failure", the same caveat as Topic/Node labels (L6), and should be reported as such.

**Draft changes.** Edge prediction can be claimed as validated against a measurement rather than a
heuristic, with the magnitudes and the L5 caveat stated. The honest reading of small magnitudes is a
finding: most individual links are replaceable.

---

## 8. Remediation now filters per edit (§6.4 / §6.7)

`PrescribeService` applied the whole compiled policy and judged it by one end-state SRI check, so a
regressing edit could ride along with improving ones — the mechanism behind the `+4.61 %` mean that
concealed `−31.67 %` regressions.

Each candidate is now simulated alone across the propagation-threshold sweep and the seed set, and
kept only when `ΔI > κ·σ_seed` at **every** threshold. `PrescribeResult` carries per-edit verdicts
with measured `delta_impact`, `sigma_seed` and rejection reasons.

An empty accepted set is a valid outcome and is reported as such rather than as a no-op mutation
with an unchanged SRI.

**Draft changes.** §6.4's "open implementation gap" disclosure can be replaced with the implemented
filter and its accept/reject counts. §6.7's mixed result must be re-run before it is quoted.

---

## Reproduction

```bash
PYTHONPATH=. python reproduce/main_table.py --eval-population application
PYTHONPATH=. python reproduce/loso_all_variants.py
PYTHONPATH=. python reproduce/kfold_all_variants.py
PYTHONPATH=. python reproduce/render_table.py
PYTHONPATH=. python reproduce/ahp_sensitivity.py
PYTHONPATH=. python reproduce/threshold_sensitivity.py
PYTHONPATH=. python reproduce/convergent_validity.py
PYTHONPATH=. python -m pytest tests/ -q
```
