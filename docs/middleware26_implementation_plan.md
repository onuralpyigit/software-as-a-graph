# Implementation Plan
## QoS-Aware Heterogeneous Graph Learning for Cascade Impact Prediction in Distributed Pub-Sub Middleware

**Target:** Middleware 2026 Cycle 2 — abstract May 29, full paper June 5
**Working window:** May 7 → June 5 (29 days, ~21 working days)
**Baseline framework:** SaG (Software-as-a-Graph) — existing PhD codebase
**Status:** Plan v1.0

---

## 1. Executive summary

The paper requires five empirical claims (E1–E5 from the §6 results section). Three are supported by infrastructure that already exists; two require new experimental harnesses. The single binding pre-condition — **W1, the QoS pipeline audit** — must clear in the first three working days. If it does not, the framing must be reconsidered before further investment.

**Critical path:** W1 audit → baseline model variants → main results table → QoS Gini sweep → LOSO inductive evaluation → paper assembly.

**Compute budget estimate:** ~720 GNN training runs across all experiments. At ~5 min per run on a single GPU, this is ~60 GPU-hours plus simulation regeneration time. Plan for parallelization or a shared cluster slot.

**Risk profile:** Three high-risk items (W1 outcome, QoS Gini scenario availability, LOSO numbers materially worse than transductive). Mitigations defined in §7.

---

## 2. Critical path and dependencies

```
W1 Audit ──┬─→ Block A (Baselines) ──┬─→ Block C (Main table) ──┐
           │                         │                          │
           ├─→ Block B (HGL fixes)   ├─→ Block D (Gini sweep) ───┤
           │                         │                          ├─→ Abstract → Paper
           └─→ Block E (LOSO orch.) ─┴─→ Block F (Stratified) ───┤
                                                                 │
                                       Block G (Case study) ─────┤
                                                                 │
                                       Block H (Reprod. bundle) ─┘
```

**Hard blockers** (work cannot start until cleared):
- W1 audit must complete and pass before Blocks A–H begin

**Soft dependencies** (can start in parallel but converge):
- Blocks A and B can run in parallel once W1 clears
- Blocks C, D, E, F all depend on A + B completing
- Blocks G and H can begin in parallel with C–F (different code paths)

---

## 3. Work breakdown structure

### Block 0 — W1 QoS pipeline audit (Days 1–3, hard blocker)

**Owner deliverable:** Verifiable evidence that QoS attributes flow from topology JSON through to the HeteroGAT message function, end-to-end.

| Task | File path | Action | Acceptance |
|---|---|---|---|
| 0.1 | `saag/core/graph_builder.py` | Trace QoS attribute extraction. Print every QoS field landed on each Topic/edge | Console log shows full QoS dict per topic, no defaults |
| 0.2 | Wherever `HeteroData` is built (likely `saag/prediction/data_preparation.py`) | Dump `edge_attr.shape` and `edge_attr[0]` for each relation triple | Each relation triple's edge_attr has > 1 dimension and non-zero values reflecting QoS |
| 0.3 | `saag/prediction/models/` (HeteroGAT class) | Confirm `GATv2Conv(edge_dim=...)` is set OR custom message function consumes edge_attr | Model parameters include a learnable `W_edge` weight per relation triple |
| 0.4 | `tests/test_qos_pipeline_audit.py` (NEW) | Write deterministic regression test: mutate one Topic's QoS profile, confirm prediction changes by > 0.01 | Test passes; rerunning with same seed produces same delta |
| 0.5 | `tools/qos_pipeline_inspect.py` (NEW) | Stage-by-stage report tool: takes scenario JSON, prints what QoS encoding exists at each stage | Generates Figure-2-source diagram for paper §3.2 |

**Go/No-Go gate (end of Day 3):** Test 0.4 passes for at least three scenarios (ATM, AV, IoT). If it fails, the QoS pipeline is broken and the paper claim collapses — escalate immediately.

**Failure mode:** If audit reveals QoS collapses to a default somewhere, allocate Days 4–7 to fix the pipeline. This pushes everything else by one week and likely forces withdrawal from Cycle 2.

---

### Block A — Baseline model variants (Days 4–8)

**Owner deliverable:** Two homogeneous GAT baselines that share the trainer and evaluation pipeline with the QoS-aware HeteroGAT, controlled for hyperparameters.

| Task | File path | Action | Acceptance |
|---|---|---|---|
| A.1 | `saag/prediction/models/baselines.py` (NEW) | Implement `HomogeneousGAT_Unweighted` (single GAT on flat DEPENDS_ON, no edge_attr) | Trains on ATM scenario, produces non-trivial predictions |
| A.2 | `saag/prediction/models/baselines.py` | Implement `HomogeneousGAT_ScalarWeighted` (single GAT, edge_attr = scalar w(e)) | Trains on ATM scenario; differs from A.1 in predictions |
| A.3 | `saag/prediction/trainer.py` | Add `model_variant: str` parameter routing to correct architecture | Same trainer instantiates all 3 variants without code duplication |
| A.4 | `cli/train_graph.py` | Add `--variant {topology_rmav,homo_unweighted,homo_scalar,hetero_qos}` flag | All four variants trainable from CLI |
| A.5 | `api/routers/prediction.py` | Add `variant` field to `TrainRequest` | API accepts variant; checkpoint dir embeds variant name |
| A.6 | `tests/test_baselines.py` (NEW) | Verify each variant's parameter count is reasonable; verify variants produce different predictions | All three variants distinguishable; no shared weights |

**Acceptance for Block A as a whole:** A single command — `bash scripts/train_all_variants.sh atm_system` — trains all four variants on the ATM scenario and produces four checkpoint directories, with reported Spearman ρ printed for each.

**Honest note:** Topology RMAV is "model variant 0" (the rule-based pipeline); it doesn't need a new model file but does need to participate in the comparison harness in Block C.

---

### Block B — HeteroGAT fixes from W1 findings (Days 4–7, parallel with A)

**Owner deliverable:** The QoS-aware HeteroGAT actually consumes the QoS attributes mathematically, not just structurally.

| Task | File path | Action | Acceptance |
|---|---|---|---|
| B.1 | `saag/prediction/models/hetero_gat.py` (or current location) | If `GATConv` is used, switch to `GATv2Conv(edge_dim=N)` per relation | Each relation triple's conv has `edge_dim` > 0 |
| B.2 | Same file | Confirm per-relation weight matrices via `to_hetero` or `HeteroConv`; not shared | Parameter inspection script (B.4) reports distinct weights |
| B.3 | `saag/prediction/data_preparation.py` | Expand edge_attr from current 8-d to ~16-d per the §3.2 spec (one-hot encodings + log-scaled numerics + relation type) | New edge_attr matches §3.2 specification; tests pass |
| B.4 | `tools/inspect_hetero_model.py` (NEW) | Load checkpoint, print param count per relation triple, attention head dimensions | Output is balanced; per-relation weights are distinct |

**Acceptance for Block B:** Mutating `/conflicts/alerts` from RELIABLE/100ms to BEST_EFFORT/no-deadline produces a measurable shift (> 0.05) in `ConflictDetector`'s predicted score on the ATM scenario.

---

### Block C — Main results table harness (Days 8–14)

**Owner deliverable:** A reproducible script that produces the paper's Table 3 (per-system × per-variant Spearman ρ).

| Task | File path | Action | Acceptance |
|---|---|---|---|
| C.1 | `tools/middleware26_main_table.py` (NEW) | Orchestrator: for each scenario × variant × seed, train + evaluate; emit JSON | Single command produces full results JSON |
| C.2 | Same file | Implement paired Wilcoxon signed-rank test across variants; bootstrap CI (B=2000) | Statistical tests output integrated into JSON |
| C.3 | `tools/render_table.py` (NEW) | Convert JSON to LaTeX booktabs table for §6.1 | Generates `results/main_table.tex` ready to include |
| C.4 | `scripts/run_main_table.sh` (NEW) | Shell wrapper running the experiment with sensible defaults | Single `bash scripts/run_main_table.sh` reproduces from clean checkout |

**Compute estimate:** 8 scenarios × 4 variants × 5 seeds = 160 training runs.

**Acceptance:** Generated `results/main_table.json` populates every cell of paper Table 3; Wilcoxon p-values are computed for each pairwise variant comparison; LaTeX renders without manual edit.

---

### Block D — QoS Gini monotonicity sweep (Days 10–17)

**Owner deliverable:** The paper's Figure 3 — Δρ as a function of QoS Gini coefficient.

| Task | File path | Action | Acceptance |
|---|---|---|---|
| D.1 | `tools/qos_gini_generator.py` (NEW) | Take base scenario JSON + target Gini, produce variant JSON by re-randomizing topic QoS profiles to hit Gini target | For 5 Gini levels {0.0, 0.2, 0.4, 0.6, 0.8}, produced variants measure within ±0.02 of target |
| D.2 | `tools/qos_gini_sweep.py` (NEW) | For each base scenario × Gini level × variant {topology_only, hetero_qos} × seed, train + evaluate; emit CSV | Single command produces full sweep CSV |
| D.3 | `tools/plot_gini_monotonicity.py` (NEW) | Render Figure 3: scatter of Δρ vs Gini with linear regression line + 95% CI | PDF figure ready for paper inclusion |
| D.4 | `data/scenarios/gini_variants/` (NEW directory) | Cache the generated variant JSONs for reproducibility | Variants are version-controlled; not regenerated per run |

**Compute estimate:** 8 base scenarios × 5 Gini levels × 2 variants × 5 seeds = 400 training runs.

**Acceptance:** Figure 3 PDF exists; the regression line slope is reported with R² and p-value; the CSV reproduces from cached variants.

**Honest disclosure:** The paper §5.1 must state explicitly that Gini variants are derived synthetic scenarios, not naturally observed.

---

### Block E — LOSO orchestration across variants (Days 12–18)

**Owner deliverable:** The paper's Table 5 — in-distribution vs LOSO held-out accuracy per variant.

| Task | File path | Action | Acceptance |
|---|---|---|---|
| E.1 | `cli/loso_evaluate.py` | Add `--variant` flag wiring to existing LOSO scaffolding | Existing LOSO runs with each of the 4 variants without code duplication |
| E.2 | `tools/loso_all_variants.py` (NEW) | Orchestrator: run LOSO for all variants, produce comparison JSON | Single command produces complete LOSO comparison |
| E.3 | `output/loso_cache/` | Pre-populate scenario cache (structural, RMAV, simulation outputs) for all 8 scenarios | Cache exists for all scenarios; LOSO runs skip pipeline regeneration |
| E.4 | `tools/render_loso_table.py` (NEW) | Render Table 5 LaTeX from JSON | Table renders cleanly |

**Compute estimate:** 8 folds × 4 variants × 5 seeds = 160 training runs (with caching).

**Acceptance:** Table 5 reports both in-distribution and LOSO numbers for all four variants. The retention claim — that QoS-aware HGL preserves its lift over baselines under inductive evaluation — is empirically testable from this output.

---

### Block F — Stratified per-node-type reporting (Days 14–17, parallel with D and E)

**Owner deliverable:** Paper Figure 4 — per-node-type Spearman ρ across variants.

| Task | File path | Action | Acceptance |
|---|---|---|---|
| F.1 | `saag/prediction/trainer.py` `evaluate()` function | Confirm per-node-type Spearman ρ is in the output (already in `cli/validate_graph.py`'s `strata` field) | Output JSON includes `per_node_type` dict |
| F.2 | `tools/render_stratified_figure.py` (NEW) | Extract per-node-type ρ from main_table.json, render grouped-bar Figure 4 | PDF figure exists |
| F.3 | Manual narrative review | In paper §6.4 prose, explicitly call out the Library stratum as the cleanest case for HGL contribution | Section text aligns with actual numbers |

**Acceptance:** Figure 4 shows per-node-type ρ for all 4 variants; the Library row is not empty; the contribution narrative is supported by the data.

---

### Block G — ATM case study (Days 18–22)

**Owner deliverable:** Paper §7 worked example with attention weights overlay (Figure 5).

| Task | File path | Action | Acceptance |
|---|---|---|---|
| G.1 | `tools/extract_attention.py` (NEW) | Load trained ATM HeteroGAT, extract attention weights for ConflictDetector's incoming edges | JSON output of edge → attention weight mapping |
| G.2 | `tools/render_attention_subgraph.py` (NEW) | Render subgraph PDF with edge thickness proportional to attention weight | Publication-quality PDF |
| G.3 | Paper §7 prose | Manual narrative: walk through ConflictDetector's predicted score across 4 variants, trace what HGL attends to, quantify operational impact | Narrative grounded in actual numbers from extracted attention |

**Acceptance:** Figure 5 PDF exists; §7 prose makes claims that are traceable to the extracted attention weights.

---

### Block H — Reproducibility bundle (Days 20–28, parallel)

**Owner deliverable:** A `reproduce/` directory that an external reviewer can run end-to-end.

| Task | File path | Action | Acceptance |
|---|---|---|---|
| H.1 | `reproduce/Makefile` | Single `make all` target that regenerates every paper table and figure | `make all` from clean checkout reproduces all artifacts |
| H.2 | `reproduce/README.md` | Hardware specs, runtime estimates, troubleshooting | Step-by-step instructions tested on a fresh machine |
| H.3 | `reproduce/Dockerfile` or `environment.yml` | Pinned dependencies | Container builds; experiments reproduce inside it |
| H.4 | `models/middleware26/` | Trained checkpoints with deterministic seed-tagged names | Checkpoints checked in (or hosted on Zenodo); paths documented |
| H.5 | `data/scenarios/` and `data/scenarios/gini_variants/` | All scenarios under version control | Scenarios load without errors |

**Acceptance:** Hand the `reproduce/` directory and a fresh machine to a non-author colleague; they reproduce Table 3 within 24 hours.

---

## 4. Week-by-week schedule

### Week 1 (May 8–14): Foundation

**Goal:** W1 audit clear; both baselines training; HGL fixes complete.

| Day | Activity | Deliverable | Gate |
|---|---|---|---|
| Thu May 8 | Block 0.1, 0.2 | QoS attribute trace logs | — |
| Fri May 9 | Block 0.3, 0.4 | Audit unit test passing | — |
| Mon May 12 | Block 0.5 + start A.1, B.1 | Audit tool ready; baseline implementations begun | **Go/No-Go: W1 audit must pass** |
| Tue May 13 | Block A.1–A.4 | Both homogeneous baselines training | — |
| Wed May 14 | Block B.2–B.4 | HGL fixes integrated; param inspection clean | End of Week 1 review |

**End-of-week artifact:** Single command trains all 4 variants on ATM scenario, prints per-variant Spearman ρ.

---

### Week 2 (May 15–21): Experimental machinery

**Goal:** Main table harness, Gini generator, LOSO wrapper all functional.

| Day | Activity | Deliverable |
|---|---|---|
| Thu May 15 | Block C.1–C.2 | Main table orchestrator runs end-to-end on 1 scenario |
| Fri May 16 | Block C.3, C.4 | LaTeX rendering + shell wrapper |
| Mon May 19 | Block D.1, D.2 | Gini generator + sweep orchestrator (functional, not full data) |
| Tue May 20 | Block E.1, E.2, E.3 | LOSO multi-variant + cache populated |
| Wed May 21 | Block F.1, F.2 | Stratified reporting wired up |

**End-of-week artifact:** All experimental harnesses functional. Begin running full experiments overnight Wed→Thu.

---

### Week 3 (May 22–28): Run experiments + draft writing + abstract

**Goal:** Generate all numbers, draft paper spine, submit abstract.

| Day | Activity | Deliverable |
|---|---|---|
| Thu May 22 | Run main table (Block C) — overnight | Table 3 numbers complete |
| Fri May 23 | Run Gini sweep (Block D) — long-running | Figure 3 data complete |
| Sat May 24 | Run LOSO (Block E) — long-running | Table 5 data complete |
| Sun May 25 | Begin writing §1, §3, §5 (the spine) | First-pass spine drafted |
| Mon May 26 | §6.1, §6.3 written from real numbers | Results section first draft |
| Tue May 27 | Block G case study + §7 prose | Case study figure + text |
| Wed May 28 | Final abstract polish; verify all §1 claims trace to §6 | Abstract submission ready |
| Thu May 29 | **Submit abstract to Middleware 2026 Cycle 2** | ✅ Abstract submitted |

**End-of-week artifact:** Abstract submitted; full paper spine drafted; results numbers populated.

---

### Week 4 (May 30 – June 5): Finalize and submit

**Goal:** Complete paper, polish, submit.

| Day | Activity | Deliverable |
|---|---|---|
| Fri May 30 | Write §6.2 (attribute ablation), §6.4 (stratified), §6.5 (LOSO) | §6 complete |
| Sat May 31 | Write §2 background, §4 architecture, §8 discussion | First full draft |
| Sun Jun 1 | Write §9 related work, §10 conclusion; build references | Full draft v1 |
| Mon Jun 2 | Internal review with advisor; fix gaps | Draft v2 |
| Tue Jun 3 | Block H reproducibility bundle finalization | `reproduce/` ready |
| Wed Jun 4 | Final pass: every claim in abstract verified against tables; references checked; figures captioned | Draft v3 |
| Thu Jun 5 | **Submit full paper** | ✅ Submitted |

**Reserved buffer:** None. Build slack into Week 4 by completing experimental runs by end of Week 3 — if any experiment is still running on June 1, the paper is at risk.

---

## 5. Compute and resource budget

| Resource | Estimate | Notes |
|---|---|---|
| GPU-hours (training) | ~60 | 720 runs × ~5 min/run; parallelize 4-way → ~15 wall-clock hours |
| GPU-hours (LOSO) | ~14 | Reuses cache; cheaper than full train-from-scratch |
| Simulation CPU-hours | ~20 | Cascade simulation per scenario × seeds; cached after first run |
| Disk space | ~5 GB | Checkpoints (~50 MB × 100 checkpoints) + cached intermediates |
| Memory | 16 GB peak | xlarge scenarios in HeteroData form |

**Recommendation:** Run experiments overnight where possible. Reserve a dedicated compute slot for May 22–24 (the heaviest experimental window).

---

## 6. Files to create vs. modify

### New files (~15)

```
saag/prediction/models/baselines.py
tests/test_qos_pipeline_audit.py
tests/test_baselines.py
tools/qos_pipeline_inspect.py
tools/inspect_hetero_model.py
tools/middleware26_main_table.py
tools/render_table.py
tools/qos_gini_generator.py
tools/qos_gini_sweep.py
tools/plot_gini_monotonicity.py
tools/loso_all_variants.py
tools/render_loso_table.py
tools/render_stratified_figure.py
tools/extract_attention.py
tools/render_attention_subgraph.py
scripts/train_all_variants.sh
scripts/run_main_table.sh
reproduce/Makefile
reproduce/README.md
reproduce/Dockerfile
```

### Modified files (~6)

```
saag/prediction/models/hetero_gat.py    # Edge-attribute message passing fix
saag/prediction/data_preparation.py     # Edge_attr expansion to 16-d
saag/prediction/trainer.py              # model_variant routing
cli/train_graph.py                      # --variant flag
cli/loso_evaluate.py                    # --variant flag
api/routers/prediction.py               # variant in TrainRequest
```

### New data directories

```
data/scenarios/gini_variants/           # Generated Gini-spectrum scenarios
output/loso_cache/                      # LOSO scenario cache
results/                                # All experimental output JSON/CSV
models/middleware26/                    # Final trained checkpoints
```

---

## 7. Risk register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| W1 audit fails (QoS doesn't flow end-to-end) | Medium | Critical | Days 4–7 reserved for fix; if not fixable in window, withdraw from Cycle 2 |
| Δρ on existing scenarios is too small for headline claim | Medium | High | Gini sweep may show stronger effect; pivot framing to "QoS lift is heterogeneity-conditional" |
| LOSO numbers materially worse than transductive | Medium | Medium | Report honestly with gap explanation; the *retention* claim survives even if absolute numbers drop |
| Compute budget overrun | Low | Medium | Reduce seed count from 5 to 3 for Gini sweep if needed; report wider CI |
| Existing scenarios lack QoS heterogeneity | High | Medium | Block D.1 (Gini generator) creates controlled variants; disclose in §5.1 |
| Co-author availability for review | Medium | Low | Schedule advisor review for Mon Jun 2 explicitly now |
| Page budget overrun in §6 | Medium | Low | Pre-budget 2.5pp; if over, push per-system breakdown to appendix |
| Reviewers reject as "ML paper, not middleware" | Medium | Medium | Cover letter foregrounds middleware contribution; §1 motivating example is middleware-first; §7 case study is operationally framed |

---

## 8. Decision points (Go/No-Go gates)

**Gate 1 — End of Day 3 (May 12):** W1 audit result.
- **Pass:** Proceed with full plan.
- **Fail and fixable in 4 days:** Use Week 1 to fix; compress Week 2 by deferring Block H.
- **Fail and not fixable:** Withdraw from Cycle 2; pivot to "Beyond Structural Centrality" framing for Cycle 1 of Middleware 2027.

**Gate 2 — End of Week 1 (May 14):** All baselines trainable, HGL fixes integrated.
- **Pass:** Proceed.
- **Fail:** One of A or B is incomplete. Triage: Block A is more time-sensitive (it gates Block C). Reschedule.

**Gate 3 — End of Week 2 (May 21):** All experimental harnesses functional.
- **Pass:** Proceed to overnight experimental runs.
- **Fail:** Identify which harness is blocking. If Block D (Gini sweep) is the issue, the paper still has §6.1, §6.4, §6.5 — drop §6.3 and adjust §1 to remove the monotonicity claim.

**Gate 4 — Tuesday May 27:** Are §6 numbers strong enough to commit the abstract?
- **Pass:** Submit abstract Wednesday May 28 evening.
- **Soft pass (numbers exist but weaker than expected):** Soften claims in abstract; submit anyway.
- **Fail (numbers contradict the framing):** Withdraw — do not commit a thesis that §6 won't support.

**Gate 5 — Tuesday June 3:** Is the paper review-ready?
- **Pass:** Submit by June 5.
- **Soft fail:** Push to last 48 hours; reserve final 6 hours for honest review of every claim.

---

## 9. Acceptance criteria for the paper

These map paper sections to artifacts that must exist before submission.

| Paper artifact | Source artifact | Owner block |
|---|---|---|
| §1 Figure 1 (motivating example) | Hand-drawn, traced from W1 audit findings | Block 0 + manual |
| §3 Figure 2 (architecture) | Hand-drawn from Block B inspection | Block B |
| §3.2 edge_attr specification | Verified by `tools/qos_pipeline_inspect.py` output | Block 0 |
| §6.1 Table 3 (main results) | `results/main_table.tex` | Block C |
| §6.2 Table 4 (attribute ablation) | `results/attribute_ablation.tex` | Block C variant |
| §6.3 Figure 3 (Gini monotonicity) | `results/figures/gini_monotonicity.pdf` | Block D |
| §6.4 Figure 4 (stratified) | `results/figures/stratified.pdf` | Block F |
| §6.5 Table 5 (LOSO) | `results/loso_table.tex` | Block E |
| §7 Figure 5 (attention overlay) | `results/figures/atm_attention.pdf` | Block G |
| Abstract numbers | All of §6 | Blocks C, D, E |
| Reproducibility statement | `reproduce/README.md` | Block H |

**No claim ships without traceability to one of these artifacts.**

---

## 10. Communication and tracking

- **Daily standup with advisor:** Recommended for Weeks 3–4 only; weekly check-ins suffice for Weeks 1–2
- **Issue tracker:** Use existing GitHub issues; tag each task with `middleware26`
- **Branch strategy:** `feat/middleware26-baselines`, `feat/middleware26-gini-sweep`, etc.; merge to `main` only after acceptance criteria met
- **Status summary:** End of each week, write a one-paragraph summary covering what shipped, what's blocked, what's at risk

---

*Plan v1.0 — May 7, 2026. Iterate before kickoff.*
