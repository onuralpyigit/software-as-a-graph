# Middleware 2026 — Experimental Redesign (6-Variant Set)

**Study:** QoS-aware heterogeneous graph learning for pre-deployment critical-component prediction in distributed publish-subscribe middleware.

**Purpose:** Define the six model variants, the contrasts they support, and the evaluation protocol that keeps them comparable. The set is built so that the QoS effect is cleanly isolable at every architecture tier; read §3 for which contrasts are clean and which are bundled.

> **Naming note.** These names supersede earlier drafts. Mapping from prior terminology: `Q-Topo → Topo-QoS`; `Q-GL → GL-QoS`; `HGL-native(−QoS) → HGL`; `HGL-native → HGL-QoS`. **`HGL-proj` (heterogeneous on the DEPENDS_ON projection) has been dropped** — see the design note in §2. Note that `HGL` now means *heterogeneous on the raw pub-sub graph*, not on the projection.

---

## 1. Model Variants

| ID | Family | Substrate | Architecture | QoS | Trained? |
|----|--------|-----------|--------------|-----|----------|
| **Topo-BL** | Structural | `DEPENDS_ON` projection | centrality (untyped) | off | no (closed-form) |
| **Topo-QoS** | Structural | `DEPENDS_ON` projection | centrality (untyped) | QoS-weighted edges | no (closed-form) |
| **GL** | Graph learning | `DEPENDS_ON` projection | homogeneous GAT | off | yes |
| **GL-QoS** | Graph learning | `DEPENDS_ON` projection | homogeneous GAT | QoS-weighted | yes |
| **HGL** | Graph learning | raw pub-sub graph | heterogeneous GAT | off | yes |
| **HGL-QoS** | Graph learning | raw pub-sub graph | heterogeneous GAT | embedded on Topic nodes + pub-sub edges | yes |

**Definitions.**

- **Topo-BL** — betweenness + directed articulation points on the unweighted `DEPENDS_ON` graph. No QoS, no labels.
- **Topo-QoS** — same centrality battery on the QoS-weighted `DEPENDS_ON` graph (edge weight = QoS-derived weight).
- **GL** — homogeneous GAT over the `DEPENDS_ON` projection: one node type, one relation, no per-relation message functions. Trained on I*(v).
- **GL-QoS** — GL with QoS-weighted edges on the same projection. Trained on I*(v).
- **HGL** — heterogeneous GAT over the raw pub-sub graph: Application, Library, Topic, Broker, Node node types and the six structural relations (`PUBLISHES_TO`, `SUBSCRIBES_TO`, `ROUTES`, `RUNS_ON`, `CONNECTS_TO`, `USES`), per-relation message functions, **no QoS attributes**. Trained on I*(v).
- **HGL-QoS** — HGL with per-topic QoS embedded where it is natively heterogeneous: on Topic node features and on the `PUBLISHES_TO`/`SUBSCRIBES_TO` `edge_attr`, rather than max-pooled onto `DEPENDS_ON`. Trained on I*(v).

---

## 2. Factor Structure

The six variants form two substrate blocks, each with a QoS-off / QoS-on pair, plus the structural pair:

```
                         QoS off        QoS on
  Structural (proj):     Topo-BL   ──▶  Topo-QoS
  Homogeneous (proj):    GL        ──▶  GL-QoS
  Heterogeneous (native):HGL       ──▶  HGL-QoS
```

Vertical moves within the projection block (Topo→GL) add supervision. The move from the projection block to the native block (GL→HGL, GL-QoS→HGL-QoS) changes architecture **and** substrate together.

> **Design note — the bundled step.** Because `HGL-proj` (heterogeneous on the projection) is not in the set, there is no single-factor path that isolates *architecture* (homogeneous→heterogeneous) from *representation* (projection→native pub-sub). GL→HGL bundles both. This set therefore supports a clean **QoS** decomposition at every tier but **not** a clean architecture-vs-representation decomposition. If the latter is needed, reinsert `HGL-proj` (heterogeneous GAT on the `DEPENDS_ON` projection, QoS off) as a 7th variant; then GL→HGL-proj isolates architecture and HGL-proj→HGL isolates representation. Otherwise, present GL→HGL as a single "native heterogeneous modeling" treatment and do not attribute the gain to either factor alone.

---

## 3. Attributable Contrasts (analysis plan)

Report each as a paired delta over the 8 scenarios with a 95% bootstrap CI and a paired Wilcoxon p-value. Lead causal claims with the trained-vs-trained contrasts.

**Clean single-factor contrasts:**

| # | Contrast | Holds fixed | Isolates | Claim |
|---|----------|-------------|----------|-------|
| Q1 | Topo-BL → Topo-QoS | substrate, no learning | QoS at structural tier | "QoS carries criticality signal in structural centrality" |
| Q2 | GL → GL-QoS | substrate, homogeneous, supervision | QoS at homogeneous-projection tier | "QoS weighting added to projection-based learning" |
| **Q3** | **HGL → HGL-QoS** | substrate, heterogeneous, supervision | **QoS embedded natively** | **"embedding QoS into nodes/edges helps heterogeneous learning" ← QoS thesis** |
| S1 | Topo-BL → GL | substrate, QoS off | supervision | "learned predictor recovers the critical set better than fixed centrality (QoS off)" — confounded by supervision; frame as such |
| S2 | Topo-QoS → GL-QoS | substrate, QoS on | supervision | same, QoS on |

**Bundled (multi-factor) contrast — frame as an integrated treatment, not an isolated factor:**

| # | Contrast | Changes | Interpretation |
|---|----------|---------|----------------|
| N1 | GL → HGL | architecture **+** substrate | "switching to native heterogeneous modeling" (QoS off). **Cannot separate the two factors with this set.** |
| N1q | GL-QoS → HGL-QoS | architecture **+** substrate | same, QoS on |

**Interaction worth highlighting:** compare the QoS deltas across tiers — Q1 vs Q2 vs Q3. If QoS helps little when flattened onto the projection (small Q1/Q2) but substantially when embedded natively (large Q3), that is direct evidence the projection was destroying the QoS signal — a stronger, more interesting result than any single QoS delta in isolation.

---

## 4. Datasets

- The 8 existing synthetic scenarios (ATM, AV, HFT, healthcare, hub-and-spoke, microservices, enterprise-xlarge, IoT smart-city). Identical scenarios across all variants.
- **Disclose:** all scenarios come from one generator, so even leave-one-scenario-out shares the generator's inductive biases. This is the external-validity ceiling and partly subsumes the validation-circularity point.

---

## 5. Ground Truth

- I*(v) from the `FaultInjector` BFS cascade, multi-seed averaged. **Identical labels for all six variants** — any simulator bias differences out of the variant-to-variant deltas.
- **Check:** several records show `impact_score_std = 0.0`. If I*(v) is effectively deterministic given topology, drop the "stochastic simulation" framing or document the degenerate variance honestly. (Separate from the GNN training seeds in §7, which do carry variance.)

---

## 6. Prediction Target and Evaluation Node Set — *critical for comparability*

- **Prediction head and all metrics restricted to Application + Library nodes for every variant**, including HGL / HGL-QoS.
- HGL / HGL-QoS pass messages through Topic / Broker / Node nodes, but those nodes are **not scored**. Evaluating the native variants over a different node population than the projection variants would make F1/ρ incomparable and break every cross-block contrast.
- Mask the loss and the evaluation index to `τ_V(v) ∈ {Application, Library}` in all four GNN variants; compute structural-baseline scores on the same node set. Add a test asserting identical eval indices across variants per (scenario, seed).

---

## 7. Training & Evaluation Protocol (GNN variants: GL, GL-QoS, HGL, HGL-QoS)

**Within-scenario (primary table).**
- Per scenario, stratified-by-node-type split into train / validation; 5 seeds (seed controls split + init).
- **Metrics on held-out validation nodes only.** For small scenarios (tiny-regression: 12 apps + 4 libs), use repeated stratified k-fold rather than a single split.
- Structural baselines computed closed-form, evaluated on the same held-out indices per fold/seed.
- **Freeze one hyperparameter set across all four GNN variants** (hidden dim, heads, layers, optimizer, epochs, early-stopping). Per-variant tuning reintroduces a confound. HGL/HGL-QoS may need more layers for the longer app→topic→app paths — if so, hold that config constant across both native variants and disclose the projection/native config difference.

**Leave-One-Scenario-Out (generalization table).**
- Train on 7 scenarios, evaluate the held-out scenario's App+Library nodes. 8 folds × 5 seeds per GNN variant.
- Report relatively (which variants stay positive vs collapse); do not call low absolute ρ "strong generalization."
- Structural baselines do not train; their LOSO column equals their scenario-local score (state this).

**Aggregation & statistics.**
- Per cell: mean over 5 seeds + bootstrap 95% CI (B = 2000).
- Across scenarios: mean of per-scenario means; paired Wilcoxon per contrast in §3.
- **n = 8 is low-powered** — report effect sizes and CIs; describe non-significant deltas as "not distinguishable at n=8," not as confirmed nulls.

---

## 8. Metrics

- **Ranking:** Spearman ρ (composite) + per-node-type ρ (stratified, guards against Simpson's-paradox aggregation).
- **Identification:** F1 under rank-matched binarization (top-K predicted = critical, K = #{I*(v) > 0.5}); note P = R = F1 once, report F1.
- **Secondary:** NDCG@10, Top-5/Top-10 overlap.
- Keep a masking-specification table per GNN variant showing which attributes survive into node features / `edge_attr` — especially the QoS attributes for HGL-QoS.

---

## 9. Go / No-Go Gates

- **G0 — QoS pipeline audit for HGL-QoS (blocking).** Confirm per-topic QoS flows from topology JSON onto Topic node features and pub-sub `edge_attr` with expected dimensionality, and that mutating a topic's QoS profile produces a measurable prediction shift with non-zero gradient flow. (Native-substrate analogue of the W1 audit.)
- **G1 — HGL pilot (blocking).** Train HGL on 1–2 scenarios (one dense, one sparse). Confirm it trains to sane ρ/F1 and does **not** collapse from over-smoothing on long app→topic→app paths or hub-topic aggregation. The earlier over-smoothing note was specific to the `DEPENDS_ON` representation; this establishes whether the bipartite pub-sub graph behaves. **Do not build the full matrix until G1 passes.**
- **G2 — Full within-scenario matrix completes**, CIs finite, no degenerate cells.
- **G3 — LOSO completes** for all four GNN variants.

---

## 10. Risk Register

| Risk | Likelihood | Impact | Mitigation / fallback |
|------|-----------|--------|----------------------|
| HGL/HGL-QoS over-smooth on pub-sub graph | medium | high | G1 pilot first; if it fails, fall back to projection-only set (Topo-BL, Topo-QoS, GL, GL-QoS) and defer native modeling to journal extension |
| Eval node-set mismatch native vs projection | medium | high | App+Library mask (§6) + unit test on eval indices |
| QoS still null after native embedding (Q3 ≈ 0) | medium | medium | Q3 answers it either way; a clean native-QoS null is publishable and stronger than the projected-QoS null — and the Q1/Q2/Q3 comparison still tells the projection-vs-native story |
| Reviewer asks to separate architecture from representation | medium | medium | acknowledge bundling in N1; offer HGL-proj as the stated extension, or add it as a 7th variant if time permits |
| Per-variant tuning creep | medium | high | freeze one config across GNN variants; record in harness |
| Compute over budget | low–med | medium | only 4 GNN variants train; time the G1 pilot and extrapolate before LOSO |

---

## 11. Compute Estimate

- Trained variants: GL, GL-QoS, HGL, HGL-QoS (4).
- Within-scenario: 4 × 8 scenarios × 5 seeds = **160 GNN runs**.
- LOSO: 4 × 8 folds × 5 seeds = **160 GNN runs**.
- Structural baselines (Topo-BL, Topo-QoS): closed-form, negligible.
- **Total ≈ 320 GNN training runs.** Native variants (HGL, HGL-QoS) are heavier (larger graphs, more relations); estimate GPU-hours from the G1 pilot timing rather than assuming parity with the projection variants.

---

## 12. Implementation — file-level changes

*(Paths inferred from the current repo layout; confirm against your tree.)*

- `saag/prediction/data_preparation.py` — keep the `DEPENDS_ON` projection builder (Topo-*, GL, GL-QoS); add a native-pub-sub `HeteroData` builder (HGL, HGL-QoS) emitting five node stores and six structural edge stores. For HGL-QoS, attach per-topic QoS to Topic node features and `PUBLISHES_TO`/`SUBSCRIBES_TO` `edge_attr`; for HGL, build the same graph with QoS masked. Add `substrate ∈ {projection, native}` and `qos ∈ {off, on}` flags.
- `saag/prediction/models/` — resolve three model classes: homogeneous GAT (GL, GL-QoS), heterogeneous GAT on native graph (HGL, HGL-QoS). Parameterize relation set / input dims rather than fork. (No HGL-proj class unless the 7th variant is added.)
- Prediction head / loss — App+Library evaluation mask applied uniformly (§6).
- `tools/middleware26_main_table.py` — variant list = `{Topo-BL, Topo-QoS, GL, GL-QoS, HGL, HGL-QoS}`; wire substrate + QoS flags; assert identical eval node indices across variants per (scenario, seed).
- `tests/test_qos_pipeline_audit.py` — extend to native substrate for HGL-QoS (G0).
- New: `tools/pilot_hgl_native.py` — G1 pilot harness.
- `reproduce/EXPERIMENTS.md` — update variant table + run commands to the six names above.

---

## 13. Claim → Evidence Map

| Paper claim | Evidence | Supportable? |
|-------------|----------|--------------|
| QoS carries structural-level signal | Q1 | yes if Q1 > 0 |
| QoS added to projection-based learning helps / does not | Q2 | yes (either sign) |
| Embedding QoS natively helps heterogeneous learning | **Q3** | yes — this is the QoS thesis |
| QoS pays off more natively than under the projection | Q1 vs Q2 vs Q3 | yes — the interaction result |
| Learning beats fixed centrality | S1, S2 (confounded by supervision) | yes, framed as such |
| Native heterogeneous modeling beats homogeneous projection | N1, N1q (**bundled**) | yes as an integrated treatment; **cannot attribute to architecture or representation alone** |
| Architecture vs representation, separated | **not in this set** | **no** without adding HGL-proj |
| Inductive generalization (relative) | LOSO table | yes, framed relatively |

**Honest spine this set supports:** *a QoS-aware heterogeneous model over the native publish-subscribe graph recovers the simulator's critical set more accurately than homogeneous projection-based learning and fixed structural centrality; embedding QoS where it is natively heterogeneous (Topic nodes, pub-sub edges) contributes a measurable gain (Q3) that the dependency projection suppresses (Q1/Q2), while the architecture-plus-representation switch (N1) is reported as a single integrated treatment.* Every clause maps to a labeled contrast above.
