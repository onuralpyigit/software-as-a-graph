# Outline — Middleware 2026 Industrial Track submission

**Paper:** An Architectural Digital Twin for Pre-Deployment Verification and CI/CD Gating in
Distributed Middleware Systems
**Format:** 6 pages, ACM `sigconf`, including references
**Deadline:** August 24, 2026
**Draft:** [draft.md](draft.md) · **Industrial data:** [data/](data/)

---

## Framing

The title names two things and the paper must earn both without overclaiming either:

- **"Architectural digital twin."** Scoped precisely in §2.1: the twin is *static* — a graph
  reconstructed fresh per candidate build, not continuously synchronized with the running system.
  The dynamic half (telemetry overlay, live drift detection) is specified in the requirements and
  explicitly listed as unbuilt in §3.4, not folded into what the prototype does. §6 adds a paragraph
  distinguishing this from the manufacturing/Industry 4.0 sense of the term. **This scoping is the
  one thing that must survive every future edit** — the term was dropped from an earlier draft
  precisely because it implied telemetry/drift capability that didn't exist; reintroducing it without
  the static/dynamic split would reopen that problem.
- **"CI/CD gating."** §4 (renamed from "CI/CD Pipeline Integration") covers the implemented
  severity-exit-code gate and the specified-but-unbuilt scored suitability model. §4.1 now carries
  one sentence distinguishing this gate from the companion JSS manuscript's delta-aware,
  simulation-verified gate — see JSS disjointness below, now higher-stakes since the title puts
  gating in its headline.

Underneath both, the same three artifacts from the prior revision, now relabeled to match:

1. **Digital twin model + requirements baseline (§2, §3.4).** 112 verifiable requirements developed
   with the program's engineering organization. The most transferable contribution.
2. **CI/CD gate — implemented and specified halves (§4).** Open prototype (SaaG-P) implements the
   graph model, detector catalog, and exit-code gate; everything in §5.2 is reproducible from it.
3. **Deployment experience (SaaG-D, §5.1).** Cleared measurements from the program's pipeline.

The paper's thesis is unchanged: the *gap between the specification and any implementation*, stated
explicitly in §3.4 rather than glossed over. Four of seven unbuilt checks — including the twin's own
dynamic half — are blocked on input data acquisition, not on analysis technique.

## Two non-negotiable rules

- **Provenance.** Every number tagged `[I]` (industrial deployment) or `[P]` (open prototype), and
  carries an $n$ and a window. Enforced in §5.
- **Capability voice.** Present tense only for behaviour implemented in the prototype. Everything
  specified-but-unbuilt lives in §3.4 or §4.2 and is written in the requirements voice. This now
  explicitly covers the twin's dynamic half, not only the CI/CD scoring model.

## Page budget

| § | Content | Pages |
|---|---|---|
| 1 | Introduction; the pre-deployment verification gap; what this paper reports | 0.75 |
| 2 | Two artifacts (§2.1); **what kind of digital twin this is**; model setup data; graph model; candidate isolation | 1.00 |
| 3 | Implemented checks; cascade simulation; scenario generation; **§3.4 specified-not-implemented** | 1.10 |
| 4 | **CI/CD Gating** — implemented exit-code gate + its two limitations incl. JSS division of labor; specified scoring model; findings format | 0.90 |
| 5 | **Evaluation** — §5.1 deployment `[I]`, §5.2 prototype `[P]`, threats | 1.75 |
| 6–7 | Related work (incl. new Digital Twins paragraph distinguishing from Industry 4.0 usage); conclusion | 0.50 |

Deliberately cut (companion-manuscript material, and needed for JSS disjointness): weight-propagation
derivations, layer projections, the six-rule derivation table, all GNN prediction claims.

## Evidence inventory

### Primary study — ATM at five scales (§5.2.1–§5.2.2)

Regenerate via `python reproduce/atm_scale_sweep.py`. Domain held fixed
(air_traffic_management); scale is the sole independent variable — five datasets
(`atm_system_tiny/atm_system/atm_system_medium/atm_system_large/atm_system_xlarge`, 29/74/148/296/444
components) built from configs that copy the QoS-mix/fan-out/criticality stats blocks verbatim and
vary only `graph.counts` (`data/scenarios/scenario_1{4,0,5,6,7}_atm_*.yaml`). 5 scales × 5 seeds
(`{42, 123, 456, 789, 2024}`) = 25 runs, 0 errors.

| Result | Source |
|---|---|
| Gate cost 0.01 s (29 comp) → 1.01 s ± 0.03 (444 comp), mean over 5 seeds — roughly proportional to scale, no cliff (contrast the secondary corpus's 25× jump for a 1.6× size increase) | `results/atm_scale_sweep.json` → `cost_by_scale` |
| Catalog vs. cascade oracle, **one row per scale, not pooled** — the trend is the result: κ declines monotonically from 0.118 (29 comp) to −0.045 ± 0.005 (444 comp), crossing zero at 296 comp; % scored-flagged climbs 80.0%→93–95% | same → `catalog_trend_by_scale` |
| Gate decision distribution: 5/5 ATM scales return exit code 2 at all 5 seeds | same → `gate_decisions` |

**Effective-$n$ note:** 4 of 5 scales gave bit-identical catalog metrics across all 5 seeds; xlarge
alone showed minor seed sensitivity (std ≤ 0.005). Honest n for the trend is 5 scale points (one
generator draw each, seed-confirmed), not 25 — same pattern as the secondary corpus below, restated
in §5.3 for the primary study specifically.

**Generator-honesty note:** the ATM domain skin (`tools/generation/datasets.py`) supplies six named
app roles and a fixed five-cluster hierarchy regardless of scale — "444 components" is ~17 numbered
copies of each of the same six roles, not new subsystem types. Stated as a threat in §5.3 and in
§5.2.2's own discussion; do not let it get lost if this section is trimmed for space.

### Secondary study — cross-domain generality (§5.2.3, unchanged from the prior revision)

Regenerate via `python reproduce/detection_seed_sweep.py`. Retained as evidence the ATM-scale
finding isn't the whole story — it answers "does the pattern hold across domains?" rather than
"across scale within one domain?", which is what the primary study now answers.

| Result | Source |
|---|---|
| Gate cost 0.02 s (29 comp) → 26.74 s ± 0.32 (520 comp), mean over 5 seeds; topology (not just size) drives cost — hub-and-spoke costs more than the 2.3× larger IoT scenario, stable across all 5 seeds | `results/detection_seed_sweep.json` → `table2_gate_cost` |
| Catalog vs. cascade oracle, split by corpus: generated (n=8 architectures) P 0.237 / R 0.887 / F₁ 0.374 / κ −0.036 ± 0.053; transcribed (n=3) P 0.402 / R 0.865 / F₁ 0.546 / κ 0.299 ± 0.126 | same → per-row `catalog`, aggregated per architecture (seed sweep confirmed near-zero seed sensitivity — see caveat below) |
| Gate decision distribution: 11/11 scenarios return exit code 2 at all 5 seeds | same → `gate_decisions` |
| Three transcribed architectures now load through the canonical pipeline (schema fix, see below) — findings and gate outcome measured, not just scale described | `data/scenarios/realworld_*.json`, same → `table4_transcribed` |

Corpus is 11 generated scenarios (hash-pinned in `MANIFEST.json`) + 3 transcribed, swept at 5 seeds
each (`{42, 123, 456, 789, 2024}`), 55 runs, 0 errors.

**Schema fix:** `data/scenarios/realworld_*.json` previously stored `relationships`
as a flat list, loadable only by one example script's inline translator (which silently misrouted
unrecognized types to `publishes_to`). Fixed at the producer —
[realworld_adapter.py](../../../../saag/adapters/realworld_adapter.py) now emits the canonical
relation-keyed dict via `_to_canonical_relationships()`, which raises on an unrecognized type instead.
Content-preserved: per-type component counts and total relation counts identical before/after
(verified by diff). This is what makes Table 4's (§5.2.3) transcribed row measurements rather than
descriptions.

**Effective-$n$ caveat:** catalog output is deterministic given the graph, and the
oracle's critical set was seed-invariant for 10 of 11 scenarios (Train-Ticket shifted by one
component at 1 of 5 seeds). The 5-seed sweep rules out a lucky-seed artifact but the honest sample
size is 8 and 3 architectures, not 40 and 15 seed-pairs — stated explicitly in the draft
rather than left in the aggregate's `n=` label.

**Blocked on collection (§5.1):** system scale, pipeline latency, defect categories, incident
comparison, clearance record — templates and ground rules in [data/README.md](data/README.md).

**Excluded:** `output/*_cascade.json` and `output/*_val_s{123,456}.json` are literals written by demo
scripts (`examples/run_autoware_ros2_pipeline.py:445-450` fabricated seed variance before that
revision removed the dead normalizer block it lived next to). Not citable.

### Corpus bookkeeping

The four new ATM-scale datasets are pinned in `scripts/write_scenario_manifest.py`'s `CORPUS` dict
with `role: "case_study"` (matching the existing `atm_system` entry) — deliberately **not**
`"evaluation"`, so they stay outside `test_evaluation_suite_matches_paper_population`'s pinned pooled
count (1545, unchanged), which belongs to the JSS companion paper's evaluation suite. Verified via
`pytest tests/test_scenario_corpus.py` (17 passed) after regenerating `MANIFEST.json`.

## Remaining work before submission

1. **Fill `data/*.csv` + `data/clearance.md`** and write §5.1. Hard blocker.
2. **Author block** with the industry-affiliated co-author — CFP requires at least one.
3. **References:** 8 → ~18. Reuse the bibliography in
   [middleware26_revision_plan.md](../research/middleware26_revision_plan.md) §A1; add DDS QoS
   conformance and configuration-verification literature. **Now also load-bearing, not optional:** a
   manufacturing/Industry 4.0 digital-twin survey to back the "[Tao et al.]" placeholder in §6's new
   Digital Twins paragraph — without it, the title's core term has no citation behind the
   distinction the paper draws. The prior main-track rejection flagged a reference-free
   introduction — weave citations into §1.
4. **ACM `sigconf` conversion** and page fitting (~half a day; no LaTeX skeleton exists yet).
5. **JSS disjointness:** §4.1 now carries one sentence pointing delta-aware gating at the companion
   manuscript by name, which is the industry paper's half of the division of labor. Still open: go to
   the JSS draft and narrow its §6 gating claim / confirm its §1.6 no-parallel-submission declaration
   still holds now that this paper's title puts "CI/CD Gating" in its headline — higher-stakes than
   before, not resolved by this pass.
6. **Commit the cited artifacts** — `results/` and `output/` are currently untracked, so nothing the
   paper cites is in version control.
