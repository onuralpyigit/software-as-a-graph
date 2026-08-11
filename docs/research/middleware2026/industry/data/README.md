# Industrial evaluation data — collection spec

Every number in §5 of [draft.md](../draft.md) must resolve to a row in one of the files in this
directory (industrial deployment) or to a committed artifact under `results/` (open prototype).
The draft marks each unfilled slot with `TODO(data)`; the paper is not submittable while any
remain.

## Ground rules

1. **No number without an `n`, a window, and a provenance tag.** A latency needs a sample count; a
   defect count needs an observation window; every table row needs `industrial` or `prototype`.
2. **Every reported category must correspond to a check that actually ran** in the deployed system.
   If core-pinning conformance is not deployed, there is no core-contention defect
   category — drop the row rather than estimating it.
3. **Stage names must be the ones the deployed system emits.** Do not retrofit the four-stage split
   that appeared in the earlier draft; record what the pipeline actually reports.
4. **Record confounders.** For the before/after incident comparison, list anything else that changed
   in the same window. A reviewer will ask, and the answer is stronger when volunteered.
5. **Anonymize once, at collection time.** Decide the redaction level (program name, absolute counts
   vs. rounded) before filling the files, and record it in `clearance.md`.

## Files

| File | Feeds | Status |
|---|---|---|
| `system_scale.csv` | §5.1 system-scale table | `TODO(data)` |
| `audit_latency.csv` | §5.2.1 pipeline overhead | `TODO(data)` |
| `defect_detection.csv` | §5.2.2 defects caught pre-deployment | `TODO(data)` |
| `incident_comparison.csv` | §5.2.3 before/after incidents | `TODO(data)` |
| `clearance.md` | §5 anonymization statement + acknowledgement | `TODO(data)` |

## What is already available without collection

These come from committed repository artifacts and need no clearance; they are the reproducible
half of §5:

| Quantity | Source |
|---|---|
| Static analysis latency, 29 → 520 components | `results/detection_validation.json` → `per_scenario[].timing` |
| Per-detector cost breakdown (20 detectors) | `results/detection_validation.json` → `summary.detector_seconds_total` |
| Catalog precision / recall / F1 / Cohen's κ vs. cascade oracle | `results/detection_validation.json` → `per_scenario[].catalog` |
| End-to-end run on three transcribed architectures | `output/autoware_ros2_validation_report.json`, `data/scenarios/realworld_*.json` |

**Do not** source anything from `output/*_cascade.json` or `output/*_val_s{123,456}.json` — those are
literals written by demo scripts, and `examples/run_autoware_ros2_pipeline.py:445-450` fabricates
seed variance by copying one result dict and overwriting the `spearman` field.
