# Scenario Corpus

The reproduction reference for the datasets behind the paper's empirical results.

Everything in [`data/scenarios/`](../data/scenarios/) is a **generated artifact**. Each
`scenario_NN_*.yaml` is a versioned specification; each `*_system.json` is what the current
generator produces from it. The two are held together by three things:

1. [`MANIFEST.json`](../data/scenarios/MANIFEST.json) — per-dataset seed, entity counts, and
   canonical SHA-256.
2. [`tests/test_scenario_corpus.py`](../tests/test_scenario_corpus.py) — regenerates every dataset
   from its config and fails on any divergence from the committed bytes or the manifest.
3. The stale-cache guard in [`reproduce/main_table.py`](../reproduce/main_table.py) — refuses to
   evaluate against a `output/loso_cache/` topology that no longer matches its dataset.

That machinery exists because the corpus silently drifted once: published Table 3 and Table 4 numbers
were computed on cached topologies built by an older generator, and `make table3` reproduced the
paper only because the stale cache outranked the committed datasets. **Never hand-edit a
`*_system.json`.** Change the YAML, regenerate, refresh the manifest, rebuild the caches.

---

## 1. Corpus at a Glance

Counts are `apps / topics / brokers / nodes / libs`, read from the committed datasets.

### Evaluation suite — the seven scenarios behind every reported result

| Config | Dataset | Domain | Counts | Seed | SHA-256 |
|---|---|---|---|---:|---|
| `scenario_01_autonomous_vehicle.yaml` | `av_system.json` | av | 80 / 40 / 4 / 8 / 20 | 1001 | `f6566746ed86` |
| `scenario_02_iot_smart_city.yaml` | `iot_smart_city_system.json` | iot | 200 / 80 / 6 / 30 / 10 | 2002 | `19e97dd3e1e3` |
| `scenario_03_financial_trading.yaml` | `financial_trading_system.json` | finance | 60 / 35 / 5 / 6 / 18 | 3003 | `103f897ba3fb` |
| `scenario_04_healthcare.yaml` | `healthcare_system.json` | healthcare | 50 / 25 / 3 / 8 / 12 | 4004 | `187320d76f0b` |
| `scenario_05_hub_and_spoke.yaml` | `hub_and_spoke_system.json` | hub-and-spoke | 70 / 30 / 2 / 12 / 25 | 5005 | `5467d8c3c2d5` |
| `scenario_06_microservices.yaml` | `microservices_system.json` | microservices | 90 / 45 / 6 / 15 / 30 | 6006 | `497072b38a6d` |
| `scenario_07_enterprise_xlarge.yaml` | `enterprise_system.json` | enterprise | 300 / 120 / 10 / 40 / 50 | 7007 | `dbee39896904` |

**Pooled population: 1,545 nodes** — 850 Applications, 375 Topics, 165 Libraries, 119 Infrastructure
Nodes, 36 Brokers. This is the figure [draft.md §7.1](research/jss/draft.md) reports, and the
population underlying the per-type correlations of §5.5 and §8.2. It is asserted by
`test_evaluation_suite_matches_paper_population`; changing the corpus and the paper are one edit.

### Case study

| Config | Dataset | Domain | Counts | Seed | SHA-256 |
|---|---|---|---|---:|---|
| `scenario_10_atm_system.yaml` | `atm_system.json` | air_traffic_management | 26 / 27 / 5 / 8 / 8 | 42 | `8b0a313a3b17` |

ATM is **not** an evaluation fold. It is the worked case study (attention subgraph, expert study,
QoS pipeline trace) and is deliberately smaller and hand-shaped after the ICAO Global ATM Concept.

### Fixtures — not evaluation scenarios, not in any reported number

| Config | Dataset | Counts | Seed | SHA-256 | Used by |
|---|---|---|---:|---|---|
| `scenario_08_tiny_regression.yaml` | `tiny_system.json` | 12 / 8 / 2 / 3 / 4 | 8008 | `134aa536ae49` | generator golden-hash test; anti-pattern detection validation |
| `scenario_09_xlarge_stress.yaml` | `xlarge_system.json` | 500 / 300 / 10 / 50 / 100 | 9009 | `244a59dc505c` | pipeline scaling ceiling |

`tiny_system` is load-bearing despite its size: `scenario_08`'s canonical hash is pinned in
[`tests/test_generation_service.py`](../tests/test_generation_service.py) as the generator's
regression baseline, and the dataset is in `DETECTION_SCENARIOS` in
[`reproduce/detection_validation.py`](../reproduce/detection_validation.py). Do not delete either.

---

## 2. Which Scenario Backs Which Result

| Artifact | Scenarios used | Produced by |
|---|---|---|
| Table 3 — in-distribution ρ (§8.1) | all 7 evaluation | `make -f reproduce/Makefile table3` |
| Table 4 — LOSO × variants (§8.1) | all 7 evaluation (7 folds) | `make -f reproduce/Makefile table4` |
| Per-domain k-fold (§8.x) | all 7 evaluation | `make -f reproduce/Makefile kfold` |
| Figure 4 — stratified per-node-type ρ (§8.2) | all 7 evaluation | `make -f reproduce/Makefile figure4` |
| Figure 5 — attention subgraph | ATM only | `make -f reproduce/Makefile figure5` |
| Oracle agreement (§5.4–§5.5) | all 7 evaluation | `reproduce/convergent_validity.py` |
| Remediation SRI table (§6.7) | 6 evaluation — **Enterprise excluded** | `reproduce/run_prescribe_all.py` |
| Anti-pattern catalog efficacy (§6) | 7 evaluation + `tiny_system` | `reproduce/detection_validation.py` |
| Expert study (§9) | ATM only | `reproduce/run_expert_study.py` |

**Two scope exceptions, both stated in the paper.** Enterprise is excluded from §6.7 on measured cost
(≈8.7 h of serial per-edit verification for that scenario alone), and ATM is a case study rather than
an evaluation fold, so it appears in no LOSO or k-fold split.

---

## 3. Reproducing the Corpus

### Regenerate the datasets

```bash
# Whole corpus, in place, plus a manifest check
make -f reproduce/Makefile scenarios

# Equivalently, by hand:
PYTHONPATH=. python cli/generate_graph.py batch \
  --input-dir data/scenarios --output-dir data/scenarios --force
PYTHONPATH=. python scripts/write_scenario_manifest.py
```

`batch` writes `scenario_NN_*.json` and copies each to its canonical `<name>_system.json` through
`SCENARIO_SYSTEM_MAP` in [`cli/common/batch_generation.py`](../cli/common/batch_generation.py). The
`scenario_NN_*.json` intermediates are gitignored; only the `*_system.json` files are committed.

A single scenario:

```bash
PYTHONPATH=. python cli/generate_graph.py \
  --config data/scenarios/scenario_01_autonomous_vehicle.yaml \
  --output output/av_system.json
```

### Verify integrity

```bash
PYTHONPATH=. python scripts/write_scenario_manifest.py --check   # hashes only
PYTHONPATH=. python -m pytest tests/test_scenario_corpus.py -q   # + regeneration
PYTHONPATH=. python cli/generate_graph.py validate               # topology-class checks
```

### Rebuild the ground-truth caches

The evaluation harnesses read `output/loso_cache/<scenario>/` — topology, structural metrics,
simulated failure impact, and RMAV scores. **Regenerating datasets without rebuilding these leaves
the harnesses evaluating a graph that no longer exists**; the guard in `main_table.py` now raises
rather than letting that through.

```bash
make -f reproduce/Makefile cache      # rm -rf + repopulate; requires a live Neo4j
```

The `rm -rf` is not optional — every step in
[`scripts/populate_loso_cache.sh`](../scripts/populate_loso_cache.sh) is guarded by "skip if the file
exists", so a partial cache would otherwise survive and keep serving stale labels.

### Run the full pipeline on one scenario

```bash
PYTHONPATH=. python cli/run.py --all --input data/scenarios/av_system.json \
  --output-dir output/av_results
```

---

## 4. Design Rationale

### 4.1 Topology coverage

The evaluation suite spans five topology classes, chosen so that different criticality mechanisms
dominate in different scenarios:

1. **Fan-out dominated** (AV, IoT) — many subscribers per topic; broker and topic betweenness are the
   primary criticality driver.
2. **Dense pub-sub** (Financial Trading, Healthcare) — most apps both publish and subscribe;
   articulation-point detection and QoS weight are decisive.
3. **Anti-pattern / SPOF** (Hub-and-Spoke) — 2 brokers for 70 apps, structural vulnerability
   deliberately encoded; validates that the methodology catches what a reviewer would flag by eye.
4. **Sparse / well-distributed** (Microservices) — challenges the classifier to avoid over-flagging;
   the hardest precision test in the suite.
5. **Enterprise scale** (Enterprise, 300 apps) — the scalability benchmark, and the scenario whose
   per-edit verification cost forces its exclusion from §6.7.

The ATM case study adds a sixth regime — safety-critical real-time surveillance with ultra-reliable,
high-priority feeds — but contributes no evaluation fold.

### 4.2 QoS weight variation

Dominant QoS settings per scenario, from the `qos_stats` block of each config:

| Scenario | Durability | Reliability | Transport priority |
|---|---|---|---|
| 01 AV | TRANSIENT_LOCAL | RELIABLE | HIGH |
| 02 IoT | VOLATILE | BEST_EFFORT | LOW |
| 03 Financial Trading | PERSISTENT | RELIABLE | HIGH/CRITICAL |
| 04 Healthcare | PERSISTENT | RELIABLE | HIGH |
| 05 Hub-and-Spoke | TRANSIENT_LOCAL | RELIABLE | MEDIUM |
| 06 Microservices | TRANSIENT_LOCAL | RELIABLE | MEDIUM |
| 07 Enterprise | mixed | RELIABLE | MEDIUM |
| 10 ATM (case study) | VOLATILE | RELIABLE | HIGH/CRITICAL |
| 08 Tiny (fixture) | balanced | balanced | balanced |
| 09 XLarge (fixture) | mixed | RELIABLE | MEDIUM |

### 4.3 A note on `--scale` presets

The named presets in [`cli/generate_graph.py`](../cli/generate_graph.py) are convenience shorthands,
not the corpus. No evaluation scenario matches one:

| Preset | Apps | Topics | Brokers | Nodes | Libs |
|---|---:|---:|---:|---:|---:|
| `tiny` | 5 | 5 | 1 | 2 | 2 |
| `small` | 15 | 10 | 2 | 4 | 5 |
| `medium` | 50 | 30 | 3 | 8 | 10 |
| `large` | 150 | 100 | 6 | 20 | 30 |
| `jumbo` | 300 | 120 | 10 | 40 | 50 |
| `xlarge` | 500 | 300 | 10 | 50 | 100 |

`--scale` alone gives the preset's counts with uniform random QoS and topology; it does not reproduce
any scenario's statistical distributions. **Always reproduce the corpus with `--config`.**

---

## 5. Measured Outcomes

Measured on the corpus and caches described above, seed set `{42, 123, 456, 789, 2024}`, 300 epochs.
These are results, not targets — the pass/fail gates the paper applies are ρ ≥ 0.70 and F1 ≥ 0.80,
and several scenarios do not clear them. Regenerate with
`make -f reproduce/Makefile table3 table4 kfold`.

### 5.1 In-distribution Spearman ρ (Table 3)

| Scenario | Topo-BL | Topo-QoS | GL | GL-QoS | HGL | HGL-QoS |
|---|---:|---:|---:|---:|---:|---:|
| AV System | 0.308 | 0.772 | 0.760 | 0.759 | 0.713 | 0.692 |
| Enterprise | 0.393 | 0.815 | 0.853 | 0.621 | 0.885 | 0.883 |
| Financial Trading | 0.246 | 0.848 | 0.851 | 0.873 | 0.882 | **0.903** |
| Healthcare | −0.182 | 0.768 | 0.815 | 0.815 | 0.842 | 0.845 |
| Hub-and-Spoke | 0.299 | 0.473 | 0.494 | 0.438 | 0.537 | 0.557 |
| IoT Smart City | −0.063 | 0.100 | 0.674 | 0.548 | 0.891 | 0.883 |
| Microservices | 0.302 | 0.573 | 0.524 | 0.543 | 0.362 | 0.354 |
| **Mean** | **0.186** | **0.621** | **0.710** | **0.657** | **0.730** | **0.731** |

Microservices is the hardest scenario for the learned predictors (HGL 0.362), consistent with its
design intent: a sparse, low-centralisation topology with few genuine bottlenecks. Healthcare and IoT
are where the unweighted structural baseline fails outright (ρ ≤ 0).

### 5.2 Generalisation (Table 4, LOSO) and in-domain k-fold

| Variant | LOSO ρ | LOSO F1@K | k-fold ρ | k-fold F1@K |
|---|---:|---:|---:|---:|
| Topo-BL | 0.105 | 0.179 | 0.038 | 0.219 |
| Topo-QoS | 0.536 | 0.338 | 0.505 | 0.339 |
| RMAV / Q(v) | — | — | −0.123 | — |
| GL | 0.436 | 0.440 | 0.409 | 0.423 |
| GL-QoS | 0.430 | 0.435 | 0.397 | 0.446 |
| HGL | **0.608** | **0.465** | 0.666 | **0.491** |
| HGL-QoS | 0.595 | 0.461 | **0.693** | 0.479 |

### 5.3 Known measurement caveats

Two defects affect how these numbers should be read. Both predate this corpus rebuild — the archived
July logs show them too — and neither has been fixed here:

- **`Topo-QoS` silently degrades to `Topo-BL` on AV and IoT.** The harness logs "no QoS weights on
  graph; falling back to topology betweenness" for those scenarios, so the QoS-weighted baseline is
  in part an unweighted baseline under a QoS label. It carries the largest variance in both tables
  (σ = 0.295 LOSO, 0.322 k-fold).
- **The HGT attention extraction captures nothing.** `reproduce/extract_attention.py` reports "no
  attention captured — HGTConv version may not expose alpha", and the subgraph figure falls back to
  ranking edges by weight. Any attention-based interpretation of the ATM case study is unsupported by
  the current artifact.

---

## 6. Adding a Scenario

1. Copy an existing config and give `graph.seed` a value not already in `MANIFEST.json`.
2. Set `graph.counts` and adjust the `*_stats` distributions to your domain's topology.
3. Document the expected structural outcomes in the header comment — structural claims only. Measured
   correlations belong in §5 of this file, computed, not asserted in a YAML comment.
4. Register the config → dataset mapping in `SCENARIO_SYSTEM_MAP`
   ([`cli/common/batch_generation.py`](../cli/common/batch_generation.py)) and in `CORPUS`
   ([`scripts/write_scenario_manifest.py`](../scripts/write_scenario_manifest.py)), with its role.
5. Regenerate, refresh the manifest, rebuild the caches:
   `make -f reproduce/Makefile scenarios cache`.
6. Add it to §1 and §2 above.

Adding an **evaluation** scenario also changes the pooled population asserted by
`test_evaluation_suite_matches_paper_population` and reported in draft.md §7.1, and invalidates every
existing result — Table 3, Table 4, and the LOSO/k-fold splits all change shape. Add fixtures freely;
add evaluation scenarios only alongside a full re-run.
