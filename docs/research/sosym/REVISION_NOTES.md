# Revision notes — SoSyM tool paper (November 2026)

This revision re-scopes the draft from a results paper into a **tool paper** and replaces every claim that the repository could not support. It records what changed, why, and where the evidence is.

## 1. Why the paper was re-scoped

**Overlap with the AuSE manuscript.** The previous draft's catalog, CQP, RM, refactoring, verifier and CI-gate results were those of the AuSE draft (`docs/research/ause`, unsubmitted and held in reserve):
- the same headline numbers (1128/1589, κ +0.296, ρ 0.485, NDCG@10 0.8295, …);
- about 30 byte-identical table rows;
- a declarations paragraph copied from AuSE that named an outdated JSS title and did not disclose AuSE at all.

The revised paper describes the *tool*: its modeling language, architecture, interfaces, extension points, application and lessons. It reports only measurements produced for it, and it discloses RASSE 2025, UYMS 2026 ×2, JSS and AuSE.

**Overlap check after revision.** Comparing 12-word shingles against `ause/latex/sections` and `ause/*.md`, the only shared text is:
- the RM formulas in Appendix A, which are disclosed as necessarily identical;
- TikZ source code;
- the AuSE title where the introduction cites it.

All other sections share 0 shingles.

## 2. Claims corrected against the code

| Draft claim | Reality | Evidence |
|---|---|---|
| SDK `SaagClient.from_config(...)`, `client.model.import_topology`, `client.diagnose.evaluate`, `client.prescribe.run_verified` | No such API exists. The real API is `saag.Pipeline` (fluent builder) and `saag.Client` (one method per step) | `saag/__init__.py`, `saag/client.py`, `saag/pipeline.py` |
| CLI subcommands `saag model / diagnose / prescribe / simulate / validate` | The CLI is `saag --<step>` flags plus `saag-import`, `saag-analyze`, `saag-diagnose`, … There is no `saag-prescribe`; prescribe runs as `python cli/prescribe_graph.py` | `pyproject.toml [project.scripts]`, `cli/run.py` |
| Classes `MultiLayerGraph`, `DiagnosticEngine`, `PrescriptionCompiler`, `CascadeSimulator`, `ClosedLoopVerifier` | These names do not exist. The real names are `QualityAnalyzer`, `StructuralAnalyzer`, `AntiPatternDetector`, `compile_policy`/`PrescribeService`, `FailureSimulator`/`FaultInjector`, `EditVerifier`/`GraphEvaluator`, and the use cases in `saag/usecases/` | grep |
| `MemoryRepository` uses "vectorized adjacency tables, thread-safe copy-on-write, microseconds" | It uses plain dicts. Verification works by deep-copying the JSON, loading it into a fresh `MemoryRepository`, re-deriving dependencies, and simulating with networkx in worker processes | `saag/infrastructure/memory_repo.py`, `saag/prescription/{mutator,evaluator,verifier}.py` |
| 5 vertex types, 6 edge types; four derivation rules listed | 6 vertex types (adds NetworkDevice), 7 edge types (adds derived DEPENDS_ON), 7 rules; all 7 are now in Table 2 | `docs/graph-model.md §6.1`, `saag/core/layers.py` |
| Delta-aware gating is "(Design)" | It is implemented, with a fence margin, warn-only BOTTLENECK_EDGE and a waiver register with expiry | `saag/analysis/delta_gate.py`, `cli/diagnose_graph.py --baseline/--waivers/--fence-margin` |
| 2D/3D force-directed views; cascade heatmaps | Only 2D views exist, in ECharts; react-force-graph is installed but never imported. The heatmap is a statistics heatmap | `smart/app/explorer/page.tsx`, `smart/package.json` |
| Apache 2.0 license; repo `github.com/software-as-a-graph/...` | The license is GPL-3.0 (LICENSE file; `pyproject.toml` said MIT and is now fixed). The repo is `github.com/onuralpyigit/software-as-a-graph` | `LICENSE`, `.zenodo.json`, `git remote -v` |
| Catalog table: BROKER_OVERLOAD, COMPOUND_RISK → reallocation; TOPIC_FANOUT → split | Operators are fed by SPOF (→ reallocation) and by GOD_COMPONENT / FAILURE_HUB / HUB_AND_SPOKE / BOTTLENECK_EDGE (→ split), matched by display name, plus criticality levels | `saag/prescription/rules.py` |
| "Substrate adequacy theorem" | Restated as Proposition 1 (an elementary observation) | §3.5 |
| Real-world corpus: "Apollo Cyber RT", "Cloud IoT Mesh (MQTT)", "transcribed from production" | Apollo has no data file. The cloud model is an Online-Boutique-style e-commerce mesh with Kafka/RabbitMQ/Redis/NATS. All five models are hand-authored "modelled after" the projects. EdgeX and Home Assistant are now included | `data/scenarios/realworld_*.json`, `docs/scenario.md`, `saag/adapters/realworld_adapter.py` |
| Lesson 5: persona observations from "early dissemination" | No evidence exists, so the claim was removed. Interfaces are now argued as design rationale; the lesson was replaced by lessons traced to real defects | — |

## 3. Numbers: all re-measured

The old §7 numbers came from gitignored runs in August 2026. Those runs predate:
- the September regeneration of the corpus (edge counts changed in every scenario);
- the September fix that gave the simulator all seven derivation rules.

Several old numbers had no surviving artifact at all: Simpson ρ = 0.028, "2 h 47 min", "> 14,000 sweeps", and the real-world edge counts 284/342/195.

All numbers in the revised paper are macros in `latex/numbers.tex`, generated by `scripts/make_numbers.py` from `results/sosym/*.json`. Every one of those files carries a provenance stamp (commit `58172b5f`, corpus digest `3afa81f0cddc…`). The stamp says `dirty: true` because the working tree contained the manuscript edits and the `pyproject.toml` license change, neither of which affects computation.

**Substantive changes in the re-measured results (these affect AuSE too — see §5):**
- **RM vs centralities, app layer.** Mean ρ(Q, I) is now 0.62, against 0.50 for betweenness and 0.49 for degree. Q beats degree in **8/8** scenarios (Wilcoxon p = 0.008). The old draft (and AuSE) reported a statistical tie (0.485 vs 0.519, p = 0.945).
- **No Simpson's paradox.** At the system layer, the pooled ρ (0.33) lies inside the per-type range: Application 0.64, Node 0.21, Broker 0.12.
- **Catalog on open-source vs synthetic models.** κ is +0.26 vs −0.03 (old: +0.296 vs −0.036), with 5 models instead of 3.
- **Diagnosis time.** Up to 79 s on the 520-component scenario (old: 20.98 s); detectors take ≤ 0.17 s.
- **Refactoring yield.** 1,028 of 1,510 candidate edits are admitted (68.1%), so 32% fail verification (old: 71.0% admitted). Survival by operator: reallocation 72.4%, QoS hardening 67.3%, topic split 55.3%. The combined policy reduces mean simulated impact in 4 of 7 scenarios and *increases* it in 3 (hub-and-spoke −13.4%, microservices −6.9%, enterprise −2.2%), although every edit was admitted individually. SRI improves in all 7, so SRI is not evidence of benefit. The full sweep, 13,653 simulation sweeps, took 2 h 36 min on 20 cores; the 520-component enterprise scenario alone took 2 h 05 min (old: "> 14,000 sweeps in < 3 h", no artifact).
- **CI scenario.** On 90 random single-subscription additions (10 per model under 150 components), the absolute gate blocks every change. The delta-aware gate blocks 26%, warns on 47% and passes 28%, and all of its blocks fall on the open-source models (23 of 50); none of the 40 synthetic-model changes are blocked. DEEP_PIPELINE had to be excluded here as well: the first run spent more than 8 minutes inside its path enumeration on the 124-component financial model without finishing a single diagnosis.
- **Mini-ATM dashboard.** The system layer has one CRITICAL component and three HIGH ones (ops-host, tracker, flight-plan). The earlier revision text implied two.

## 4. Defects found while revising (items 1–2b fixed; the rest are not fixed here)

1. **`Client.visualize` with a non-Neo4j repository** created no simulation service (`saag/client.py`, guarded by `hasattr(self.repo, 'driver')`). The HTML report then lost its simulation and validation sections and showed "Validation P 0.000" for a validation that never ran. **Fixed in this revision:** the service is now always created, and the regression test is `tests/test_client_visualize_memory.py`. The mini-ATM report now shows ρ = 0.685.
2. **The HTML report's "SPOFs detected" card read 0 on mini-ATM**, while the catalog reported a SPOF finding (the display). The card and the per-component badges counted undirected articulation points only. The SPOF detector also counts directed articulation points and the Availability fence. **Fixed in this revision:** the card and badges now count the catalog's SPOF findings (`saag/visualization/collector.py`). The regression test is in `tests/test_client_visualize_memory.py`, and `tests/test_visualization_service.py` now asserts the new contract.
2b. **The report's per-component anti-pattern lists were always empty when findings were detected on the fly.** Findings were mapped to components through `component_ids`/`components` keys, which `DetectedProblem` dicts do not have. **Fixed in this revision:** findings are normalised to their catalog ID and the components they name: the `entity_id`, or both endpoints of an Edge finding (`u->v`, or `u<->v` for a chatty pair). JSON-file findings work as before. The test is in `tests/test_client_visualize_memory.py`; on mini-ATM the display lists `SPOF`, each topic lists `ISOLATED`, and at the `mw` layer both brokers list `BRIDGE_EDGE`.
3. **At the system layer every Topic is reported as `ISOLATED`**, because topics are not endpoints of any DEPENDS_ON rule. The paper reports this as a known false positive (§4.2, §9.1).
4. **`cli/prescribe_graph.py` imports `reproduce.detection_validation`**, which is not packaged, so the script only works from a source checkout. The paper states this (§9.1).
5. **The UYMS visualization paper describes WebGL 2D/3D `react-force-graph` views** that the current front end does not use.

## 5. Items for the authors

- **AuSE.** The manuscript was not submitted to the AuSE Special Issue and is held in reserve. Its reported numbers are stale in the same way the old SoSyM numbers were. Most importantly, its RM-vs-degree "saturation" conclusion is reversed on the current release. Consider a correction or update when preparing for a future submission.
- **UYMS ADVENT case study.** The published Fleiss' κ range (0.65–0.76) is exactly the target range of `reproduce/run_expert_study.py`, which generates synthetic ratings by searching perturbations until κ lands in that window. Before any SaG paper cites the ADVENT expert evaluation as evidence, confirm that the published ratings were collected from experts. The revised SoSyM paper cites UYMS only for disclosure, not as evidence.
- **Release metadata.** Mint a Zenodo DOI for the tool release. Note that `.zenodo.json` currently describes the JSS dataset, not the tool.
- **Front matter.** Fill in affiliations, ORCIDs and the grant number, and switch to `sn-jnl`.
