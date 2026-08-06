# Outline — Middleware 2026 Industrial Track submission

**Paper:** An Architectural Digital Twin for Pre-Deployment Verification and CI/CD Gating in
Distributed Middleware Systems
**Format:** 6 pages, ACM `sigconf`, including references
**Deadline:** August 24, 2026
**Draft:** [draft.md](draft.md) · **Requirements baseline:** [system_requirements.md](system_requirements.md) ·
**Industrial data:** [data/](data/)

---

## Framing

The title names two things and the paper must earn both without overclaiming either:

- **"Architectural digital twin."** Scoped precisely in §2.1: the twin is *static* — a graph
  reconstructed fresh per candidate build, not continuously synchronized with the running system.
  The dynamic half (telemetry overlay, live drift detection) is specified (SSS Req 6.37–6.39) and
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

**Available now, reproducible (§5.2) — regenerate via
`python reproduce/detection_seed_sweep.py`:**

| Result | Source |
|---|---|
| Gate cost 0.02 s (29 comp) → 26.74 s ± 0.32 (520 comp), mean over 5 seeds; topology (not just size) drives cost — hub-and-spoke costs more than the 2.3× larger IoT scenario, stable across all 5 seeds | `results/detection_seed_sweep.json` → `table2_gate_cost` |
| Catalog vs. cascade oracle, split by corpus: generated (n=8 architectures) P 0.237 / R 0.887 / F₁ 0.374 / κ −0.036 ± 0.053; transcribed (n=3) P 0.402 / R 0.865 / F₁ 0.546 / κ 0.299 ± 0.126 | same → per-row `catalog`, aggregated per architecture (seed sweep confirmed near-zero seed sensitivity — see caveat below) |
| Gate decision distribution: 11/11 scenarios return exit code 2 at all 5 seeds | same → `gate_decisions` |
| Three transcribed architectures now load through the canonical pipeline (schema fix, see below) — findings and gate outcome measured, not just scale described | `data/scenarios/realworld_*.json`, same → `table4_transcribed` |

Corpus is 11 generated scenarios (hash-pinned in `MANIFEST.json`) + 3 transcribed, swept at 5 seeds
each (`{42, 123, 456, 789, 2024}`), 55 runs, 0 errors.

**Schema fix (this revision):** `data/scenarios/realworld_*.json` previously stored `relationships`
as a flat list, loadable only by one example script's inline translator (which silently misrouted
unrecognized types to `publishes_to`). Fixed at the producer —
[realworld_adapter.py](../../../../saag/adapters/realworld_adapter.py) now emits the canonical
relation-keyed dict via `_to_canonical_relationships()`, which raises on an unrecognized type instead.
Content-preserved: per-type component counts and total relation counts identical before/after
(verified by diff). This is what makes Table 3's transcribed row and Table 4 measurements rather than
descriptions.

**Effective-$n$ caveat carried into §5.3:** catalog output is deterministic given the graph, and the
oracle's critical set was seed-invariant for 10 of 11 scenarios (Train-Ticket shifted by one
component at 1 of 5 seeds). The 5-seed sweep rules out a lucky-seed artifact but the honest sample
size for Table 3 is 8 and 3 architectures, not 40 and 15 seed-pairs — stated explicitly in the draft
rather than left in the aggregate's `n=` label.

**Blocked on collection (§5.1):** system scale, pipeline latency, defect categories, incident
comparison, clearance record — templates and ground rules in [data/README.md](data/README.md).

**Excluded:** `output/*_cascade.json` and `output/*_val_s{123,456}.json` are literals written by demo
scripts (`examples/run_autoware_ros2_pipeline.py:445-450` fabricated seed variance before this
revision removed the dead normalizer block it lived next to). Not citable.

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
