# Scenario Corpus

The reproduction reference for the datasets behind the empirical results of the Journal of Systems and Software (JSS) paper.

Everything in [`data/scenarios/`](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/data/scenarios/) is a **versioned, reproducible artifact**. Each `scenario_NN_*.yaml` is a declarative specification; each `*_system.json` is what the generator deterministically produces from it. The corpus is held together by three integrity layers:

1. [`MANIFEST.json`](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/data/scenarios/MANIFEST.json) — per-dataset seed, entity counts, git commit, and canonical SHA-256 cryptographic digest.
2. [`tests/test_scenario_corpus.py`](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/tests/test_scenario_corpus.py) — continuous integration test that regenerates every dataset from its configuration and fails on any divergence from committed bytes or the manifest.
3. The stale-cache guard in [`reproduce/main_table.py`](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/reproduce/main_table.py) — refuses to evaluate against an `output/loso_cache/` topology that no longer matches its dataset.

That machinery exists because the corpus silently drifted early in development: published preliminary numbers were computed on cached topologies built by an older generator, and `make table3` reproduced the paper only because the stale cache outranked the committed datasets. **Never hand-edit a `*_system.json`.** Change the YAML configuration, regenerate via the CLI, refresh the manifest, and rebuild the caches.

---

## 1. Corpus at a Glance

The complete JSS evaluation corpus comprises **2,812 components and 11,618 edges across seventeen system architectures** (JSS Table 4), partitioned into twelve synthetic inductive scenarios, five hand-authored open-source system models, an ATM scaling suite for analysis cost benchmarking, and three regression/stress fixtures.

Entity counts denote `Applications / Topics / Brokers / Execution Hosts / Libraries` (`apps / topics / brokers / nodes / libs`), read directly from the committed datasets.

### 1.1 Synthetic Inductive Evaluation Suite — Twelve LOSO Folds

These twelve synthetic architectures form the basis of the **Leave-One-Scenario-Out (LOSO)** cross-validation protocol (JSS Table 7 and Supplementary Table S10). In each fold, one scenario is withheld as an unseen holdout architecture while models train on the remaining eleven.

| Config | Dataset | Domain / Architecture | Apps | Topics | Brokers | Hosts | Libs | $|V|$ | $|E|$ | Seed | Canonical SHA-256 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `scenario_01_autonomous_vehicle.yaml` | `av_system.json` | ROS 2 Cyber-Physical | 80 | 40 | 4 | 8 | 20 | 152 | 774 | 1001 | `93723cf417d8` |
| `scenario_02_iot_smart_city.yaml` | `iot_smart_city_system.json` | MQTT Telemetry Mesh | 200 | 80 | 6 | 30 | 10 | 326 | 1,188 | 2002 | `a1c4e05fbce3` |
| `scenario_03_financial_trading.yaml` | `financial_trading_system.json` | Low-Latency Pub-Sub | 60 | 35 | 5 | 6 | 18 | 124 | 631 | 3003 | `e31826279e33` |
| `scenario_04_healthcare.yaml` | `healthcare_system.json` | HL7/FHIR Event Mesh | 50 | 25 | 3 | 8 | 12 | 98 | 389 | 4004 | `c7200f062c27` |
| `scenario_05_hub_and_spoke.yaml` | `hub_and_spoke_system.json` | Broker Hub / ESB | 70 | 30 | 2 | 12 | 25 | 139 | 691 | 5005 | `74b126593ecd` |
| `scenario_06_microservices.yaml` | `microservices_system.json` | Cloud-Native Services | 90 | 45 | 6 | 15 | 30 | 186 | 678 | 6006 | `911f57d28b6a` |
| `scenario_07_enterprise_xlarge.yaml` | `enterprise_system.json` | Kafka Event Mesh | 300 | 120 | 10 | 40 | 50 | 520 | 3,216 | 7007 | `7550365ba09c` |
| `scenario_10_atm_system.yaml` | `atm_system.json` | ICAO ATM System | 26 | 27 | 5 | 8 | 8 | 74 | 261 | 42 | `8e8a46084939` |
| `scenario_18_telecom_ran.yaml` | `telecom_ran_system.json` | 5G Radio Access Network | 120 | 55 | 8 | 20 | 22 | 225 | 881 | 1801 | `659fe239ec19` |
| `scenario_19_industrial_scada.yaml` | `industrial_scada_system.json` | Plant SCADA Telemetry | 140 | 70 | 4 | 25 | 15 | 254 | 824 | 1902 | `e3efcce21b30` |
| `scenario_20_realtime_gaming.yaml` | `realtime_gaming_system.json` | Multiplayer State Sync | 75 | 38 | 5 | 12 | 28 | 158 | 630 | 2003 | `34f62d396976` |
| `scenario_21_logistics_fleet.yaml` | `logistics_fleet_system.json` | Vehicle Telematics Mesh | 110 | 50 | 7 | 18 | 20 | 205 | 755 | 2104 | `93595b70607c` |
| **Synthetic Subtotal (12 Folds)** | — | **12 Archetypes** | **1,321** | **615** | **65** | **202** | **258** | **2,461** | **10,918** | — | — |

* **11 Core Evaluation Scenarios Subtotal**: 2,387 components (1,295 Applications, 588 Topics, 60 Brokers, 194 Hosts, 250 Libraries) and 10,657 edges. Asserted by `test_evaluation_suite_matches_paper_population` in [`tests/test_scenario_corpus.py`](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/tests/test_scenario_corpus.py).
* **ATM System Case Study**: 74 components (26 Applications, 27 Topics, 5 Brokers, 8 Hosts, 8 Libraries) and 261 edges. Hand-shaped after the ICAO Global ATM Concept. In JSS, ATM serves a dual role: it is promoted to the **12th inductive LOSO fold** while retaining its function as the qualitative case study for relational attention subgraph inspection (JSS Figure 5) and the illustrative diagnostic remediation card (Supplementary Table S8).
* **Corpus Diversity Extension**: Four additional domains (`telecom_ran`, `industrial_scada`, `realtime_gaming`, `logistics_fleet`) were introduced to expand the inductive suite from 8 to 12 folds. This expansion provides sufficient statistical power for two-sided Wilcoxon signed-rank tests ($n=12$), avoiding the $p = 0.0078$ floor of $n=8$ (see [`docs/research/jss/PREREGISTRATION.md`](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/docs/research/jss/PREREGISTRATION.md)).

### 1.2 Open-Source Real-World System Models — Zero-Shot Transfer Suite

Five hand-authored architecture models of production open-source distributed systems, encoded in canonical SaG typed multigraph format via [`saag.adapters.realworld_adapter.RealWorldAdapter`](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/saag/adapters/realworld_adapter.py) and generated via [`cli/import_realworld_system.py`](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/cli/import_realworld_system.py). These models are **withheld from all training folds** and used exclusively for **zero-shot transfer evaluation** (RQ4, JSS Table 9b & 9c).

| Config | Dataset | Paradigm / Original System | Apps | Topics | Brokers | Hosts | Libs | $|V|$ | $|E|$ | Seed | Canonical SHA-256 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `scenario_11_realworld_autoware_ros2.yaml` | `realworld_autoware_ros2.json` | ROS 2 Autonomous Driving ([Autoware.universe](https://github.com/autowarefoundation/autoware.universe)) | 32 | 24 | 3 | 6 | 10 | 75 | 179 | 2026 | `e3cab5cc6d95` |
| `scenario_12_realworld_cloud_microservices.yaml` | `realworld_cloud_microservices.json` | Cloud Microservice Mesh (modelled after [Online Boutique](https://github.com/GoogleCloudPlatform/microservices-demo)) | 22 | 20 | 4 | 6 | 8 | 60 | 128 | 2026 | `a247c0c977a5` |
| `scenario_13_realworld_trainticket.yaml` | `realworld_trainticket.json` | Industrial Microservices (modelled after [Train-Ticket](https://github.com/FudanSELab/train-ticket)) | 41 | 30 | 3 | 8 | 8 | 90 | 162 | 2026 | `c6b4b6d3de98` |
| `scenario_14_realworld_homeassistant.yaml` | `realworld_homeassistant.json` | Local-First Smart Home ([Home Assistant](https://www.home-assistant.io/)) | 24 | 22 | 3 | 6 | 8 | 63 | 119 | 2026 | `eb571213c275` |
| `scenario_15_realworld_edgex.yaml` | `realworld_edgex.json` | Industrial Edge Computing ([EdgeX Foundry](https://www.edgexfoundry.org/)) | 22 | 24 | 3 | 6 | 8 | 63 | 112 | 2026 | `6c289dd666fc` |
| **Open-Source Subtotal (5 Systems)** | — | **5 Authentic Benchmarks** | **141** | **120** | **16** | **32** | **42** | **351** | **700** | — | — |

**Grand Total (Full JSS Corpus, 17 System Architectures):**
$$\mathbf{|V| = 2{,}812} \quad (1{,}462 \text{ Apps}, 735 \text{ Topics}, 81 \text{ Brokers}, 234 \text{ Hosts}, 300 \text{ Libs}), \quad \mathbf{|E| = 11{,}618}$$

#### Architectural Details of Open-Source Models

* **`realworld_autoware_ros2.json` (`autoware_ros2`)**: Authentic ROS 2 pub-sub architecture of Autoware.universe. Spans 32 Applications (sensing, perception, localization, planning, control, vehicle interface, emergency safety), 24 DDS Topics with explicit DDS QoS contracts (`VOLATILE`/`TRANSIENT_LOCAL` durability, `BEST_EFFORT`/`RELIABLE` reliability, `CRITICAL`/`HIGHEST`/`HIGH` priority), 3 Brokers (DDS middleware: Eclipse CycloneDDS, eProsima FastDDS, Zenoh Router), 6 Deployment ECUs (Main Brain EPYC, Perception GPU Orin AGX, Sensing FPGA, Vehicle Actuation Aurix MCU, Teleop HMI, Gateway), and 10 shared C++/ROS 2 libraries (`autoware_universe_utils`, `tier4_autoware_utils`, `motion_utils`, `rclcpp_core`). Tested via [`examples/run_autoware_ros2_pipeline.py`](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/examples/run_autoware_ros2_pipeline.py).
* **`realworld_cloud_microservices.json` (`cloud_microservices`)**: Cloud-native pub-sub microservice mesh modelled after production e-commerce stacks (Google Online Boutique reference architecture). Contains 22 Applications (frontend, API gateway, auth, cart, checkout, order processor, payment, inventory reservation, notification workers, fraud detection, recommendation engine), 20 Topics with message broker QoS profiles, 4 Brokers (Apache Kafka cluster, RabbitMQ exchange, Redis PubSub, NATS JetStream), 6 Infrastructure/K8s nodes, and 8 shared SDKs (`shared-auth-jwt-client`, `kafka-common-producer`, `grpc-telemetry-sdk`, `redis-cache-utils`, `spring-cloud-circuitbreaker`).
* **`realworld_trainticket.json` (`trainticket_microservices`)**: Industrial Train-Ticket microservices benchmark (Fudan University benchmark suite for SOA microservice management and fault diagnosis). Contains 41 Applications (order, travel, preserve, route, seat, payment, food, security, user, verification, admin, gateway services), 30 Pub-Sub topics for asynchronous event delivery and REST/gRPC message routes, 3 Message Brokers, 8 Deployment Nodes, and 8 Shared Libraries.
* **`realworld_homeassistant.json` (`homeassistant_iot`)**: Authentic Home Assistant smart home IoT topology. Spans 24 Applications (core state machine, automation rule engine, script runner, safety hazard monitor, alarm control panel, emergency actuation, ZHA Zigbee gateway, Z-Wave supervisor, ESPHome manager, MQTT bridge, camera stream manager, climate HVAC controller, recorder persistence, energy analytics, WebSocket server, HTTP ingress, Lovelace UI dashboard), 22 Topics, 3 Brokers (AsyncIO EventBus, Mosquitto MQTT, WebSocket push router), 6 Deployment Nodes, and 8 shared Python libraries (`aiohttp`, `voluptuous`, `sqlalchemy`, `paho-mqtt`, `zeroconf`).
* **`realworld_edgex.json` (`edgex_foundry_iiot`)**: Industrial EdgeX Foundry IoT edge computing topology. Contains 22 Applications across Southbound Device services (Modbus, MQTT, GPIO, Camera Vision, SNMP), Core services (Core Data, Metadata, Command, Keeper Registry), Northbound App & Analytics services (eKuiper stream rules engine, configurable app service, safety interlock, cloud exporters, SCADA historian), and Support/Security services (Kong proxy, Vault secret store, notifications, scheduler). Spans 24 Topics with industrial QoS configurations, 3 Brokers (Redis Streams bus, Mosquitto OT field broker, eKuiper stream bus), 6 Deployment Nodes, and 8 shared Go modules (`go-mod-core-contracts`, `go-mod-messaging`, `go-mod-bootstrap`).

### 1.3 ATM Scaling Suite — RQ5 Analysis Cost and Scalability Sweeps

The ATM system is parameterized into four scaled variants to measure runtime overhead, feature extraction bottlenecks, and neural inference scaling up to 1,998 components and 19,301 edges (JSS Table 10, Section 7.5).

| Config | Dataset | Apps | Topics | Brokers | Hosts | Libs | $|V|$ | $|E|$ | Seed | Canonical SHA-256 | Empirical Role |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `scenario_14_atm_tiny.yaml` | `atm_system_tiny.json` | 10 | 11 | 2 | 3 | 3 | 29 | 99 | 4214 | `c3ad075113d0` | Scalability sweep baseline |
| `scenario_15_atm_medium.yaml` | `atm_system_medium.json` | 52 | 54 | 10 | 16 | 16 | 148 | 540 | 4215 | `14ef3fdcee0d` | Intermediate scale checkpoint |
| `scenario_16_atm_large.yaml` | `atm_system_large.json` | 104 | 108 | 20 | 32 | 32 | 296 | 1,064 | 4216 | `36db65f8aadf` | High-load scale checkpoint |
| `scenario_17_atm_xlarge.yaml` | `atm_system_xlarge.json` | 156 | 162 | 30 | 48 | 48 | 444 | 1,723 | 4217 | `4b91d94675f4` | Stress-scale checkpoint |

In JSS Table 10, composite configurations scaling from 249 to 1,998 components benchmark the contrast between deterministic static analysis (dominated by the $O(|V|^2 + |V||E|)$ Connectivity Degradation Index, taking up to 239 s) and neural inference ($56\text{ ms}$ for HGT forward pass, an analyze:forward ratio of $4{,}259\times$).

### 1.4 Fixtures — Continuous Integration and Pipeline Limits

These three configurations serve regression testing and stress analysis. They are **not** evaluation scenarios and do not appear in any reported ranking correlation table:

| Config | Dataset | Apps | Topics | Brokers | Hosts | Libs | $|V|$ | $|E|$ | Seed | Canonical SHA-256 | Used By |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| `scenario_08_tiny_regression.yaml` | `tiny_system.json` | 12 | 8 | 2 | 3 | 4 | 29 | 96 | 8008 | `258e0b80dde4` | Generator golden-hash regression baseline; anti-pattern catalog validation |
| `scenario_09_xlarge_stress.yaml` | `xlarge_system.json` | 500 | 300 | 10 | 50 | 100 | 960 | 5,097 | 9009 | `e9357afba29f` | In-memory pipeline scaling ceiling and memory profiling |
| `scenario_11_integration_hub_migration.yaml` | `integration_hub_migration_system.json` | 40 | 20 | 4 | 6 | 10 | 80 | 240 | 1111 | `116e4be669ba` | Stage 5 corpus-diversity fixture (publisher-less topics, mixed QoS) |

* `tiny_system` is pinned in [`tests/test_generation_service.py`](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/tests/test_generation_service.py) as the generator regression baseline and in `DETECTION_SCENARIOS` in [`reproduce/detection_validation.py`](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/reproduce/detection_validation.py).
* `integration_hub_migration_system` deliberately targets two edge-case properties: publisher-less topics (2/20, both sole-routed) and genuinely balanced QoS (no durability/reliability category exceeds 55%). It intentionally omits `graph.domain` to bypass per-domain curated QoS lookups in `DomainDataset`.

---

## 2. Which Scenario Backs Which Result

| Paper Artifact / Table | Scenario Set Used | Count | Verification / Reproduction Target |
|---|---|:---:|---|
| **JSS Table 4** (Corpus Overview) | 12 Synthetic Scenarios + 5 Open-Source System Models | 17 | [`scripts/write_scenario_manifest.py`](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/scripts/write_scenario_manifest.py) |
| **JSS Table 5 & Supp. Table S11** (In-Distribution $\rho$ and Overlap@$K$) | 11 Evaluation Scenarios + ATM System | 12 | `make -f reproduce/Makefile table3` |
| **JSS Table 7** (Inductive LOSO Cross-Validation) | 12 Synthetic Inductive Folds | 12 | `make -f reproduce/Makefile table4` |
| **JSS Table 7c** (Active Stratum Ranking $\rho_{>0}$) | 12 Synthetic Inductive Folds ($n_{>0}$ active components) | 12 | `reproduce/main_table.py --active-stratum` |
| **JSS Table 6 / Table 11** (Capacity- and Channel-Matched Controls) | 12 Synthetic Inductive Folds (`GAT-N-C`, `GAT-N-QoS16-C`) | 12 | `reproduce/main_table.py --matched-controls` |
| **JSS Table 8 & Supp. Table S12** (`SaG-Hybrid` & `SaG-Hybrid-GAT`) | 12 Synthetic LOSO Folds + 5 Open-Source Models | 17 | `reproduce/main_table.py --hybrid` |
| **JSS Table 9b & 9c / Supp. Table S13** (Zero-Shot Transfer) | 5 Open-Source Systems (`autoware`, `boutique`, `trainticket`, `homeassistant`, `edgex`) | 5 | `reproduce/main_table.py --realworld-zeroshot` |
| **JSS Table 10 & Table 11 in §7.5** (Inference Cost & Scalability) | ATM Scaling Suite (249 to 1,998 nodes) + 12 Synthetic Scenarios | 16 | `reproduce/main_table.py --scale-sweep` |
| **JSS §7.3.2 / Supp. Table S9** (Cross-Oracle Convergent Validity) | 12 Synthetic Inductive Folds ($I^*$, $I_{\text{dyn}}$, $I_{\text{comp}}$) | 12 | `reproduce/convergent_validity.py` |
| **JSS §7.3.3 / Figure 4** (Node-Type Stratification & Anti-Patterns) | 12 Synthetic Scenarios ($I_{\text{comp}}$ Sensitivity) | 12 | `reproduce/detection_validation.py` |
| **JSS §7.3.4 / Figure 5 & Supp. Table S8** (Attention & Card) | ATM System Only | 1 | `reproduce/main_table.py --figure5` |
| **JSS §6.7** (Remediation / Prescriptive SRI Table) | 11 Evaluation Scenarios (**Enterprise excluded**) | 11 | `reproduce/run_prescribe_all.py` |

**Scope Exceptions:**
1. **Enterprise exclusion from remediation**: In §6.7, the Enterprise scenario (520 components, 3,216 edges, 26,276 projection edges) is excluded from automated prescriptive remediation verification due to measured computational cost ($\approx 8.7\text{ h}$ of serial per-edit simulation).
2. **ATM dual role**: In preliminary versions, ATM was only a qualitative case study; in the JSS submission, ATM is formally included as the 12th inductive LOSO fold while still providing the qualitative case study for attention visualization and remediation cards.

---

## 3. Reproducing the Corpus

### 3.1 Regenerate the Datasets

```bash
# Whole corpus, in place, plus manifest check
make -f reproduce/Makefile scenarios

# Equivalently, by direct CLI execution:
PYTHONPATH=. python cli/generate_graph.py batch \
  --input-dir data/scenarios --output-dir data/scenarios --force
PYTHONPATH=. python scripts/write_scenario_manifest.py

# Hand-authored open-source system topologies:
PYTHONPATH=. python cli/import_realworld_system.py
```

`batch` writes intermediate `scenario_NN_*.json` files and mirrors each to its canonical `<name>_system.json` via `SCENARIO_SYSTEM_MAP` in [`cli/common/batch_generation.py`](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/cli/common/batch_generation.py). The intermediate files are gitignored; only the `*_system.json` files are committed.

### 3.2 Verify Integrity

```bash
PYTHONPATH=. python scripts/write_scenario_manifest.py --check   # Check hashes against MANIFEST.json
PYTHONPATH=. python -m pytest tests/test_scenario_corpus.py -q   # Assert regeneration & population
PYTHONPATH=. python -m pytest tests/test_realworld_adapter.py -q # Assert open-source models
PYTHONPATH=. python cli/generate_graph.py validate               # Verify schema invariants
```

### 3.3 Rebuild Ground-Truth Caches

Harnesses read pre-computed topologies, structural metrics, failure reachability, and RM scores from `output/loso_cache/<scenario>/`:

```bash
make -f reproduce/Makefile cache      # Purges and repopulates caches (requires live Neo4j)
```

The `reproduce/main_table.py` runner asserts cache consistency against `data/scenarios/` and aborts if any cached graph diverges from the committed files.

---

## 4. Design Rationale and Domain Topology Coverage

### 4.1 Topology Coverage Across Paradigms

The 12 synthetic scenarios and 5 open-source systems span diverse operational communication structures, ensuring that different failure propagation mechanisms dominate across folds:

1. **Fan-Out Dominated Telemetry** (AV, IoT Smart City, Logistics Fleet) — High subscriber-to-topic ratios; broker betweenness and topic fan-out are the primary criticality drivers.
2. **Dense Symmetric Pub-Sub** (Financial Trading, Healthcare) — Most applications both publish and subscribe; articulation points and QoS transport priorities are decisive.
3. **Anti-Pattern / Single Point of Failure (SPOF)** (Enterprise Integration Hub / ESB) — 2 brokers for 70 applications, intentionally encoding centralized messaging bottlenecks to test whether topological methods match human architectural flags.
4. **Sparse / Well-Distributed Meshes** (Microservices) — Low centralization and high redundancy; tests the ranker against false-positive over-flagging (the hardest precision test).
5. **High-Density Enterprise Scale** (Enterprise Pub-Sub, 300 apps) — Evaluates graph projection density ($26{,}276$ derived edges) and scalability limits.
6. **Safety-Critical Surveillance** (Air Traffic Management) — Stringent real-time deadlines and ultra-reliable delivery contracts with minimal redundancy.
7. **5G Cellular Infrastructure** (Telecom RAN) — Asymmetric control-plane vs. data-plane split across 20 baseband execution units.
8. **Industrial Supervisory Control** (Industrial SCADA) — Star-and-tree sensor meshes feeding centralized historians and safety interlocks.
9. **Multiplayer State Synchronization** (Real-Time Gaming) — Ephemeral state updates with high volatile message rates.
10. **Authentic Cyber-Physical & Microservice Systems** (Autoware ROS 2, Online Boutique model, Train-Ticket model, Home Assistant, EdgeX Foundry) — Real-world DDS architectures, service discovery registries, and containerized pub-sub networks with heterogeneous middleware stacks.

### 4.2 Modal QoS Configurations

Dominant QoS settings per scenario from the `qos_stats` block of each configuration:

| Scenario / Architecture | Modal Durability | Modal Reliability | Transport Priority | QoS Variance Range |
|---|---|---|---|---|
| **01 Autonomous Vehicle** | `TRANSIENT_LOCAL` | `RELIABLE` | `HIGH` | 45%–80% modal share |
| **02 IoT Smart City** | `VOLATILE` | `BEST_EFFORT` | `LOW` | 56%–75% modal share |
| **03 Financial Trading** | `PERSISTENT` | `RELIABLE` | `HIGH` / `CRITICAL` | 40%–89% modal share |
| **04 Healthcare Integration** | `PERSISTENT` | `RELIABLE` | `HIGH` | 40%–88% modal share |
| **05 Enterprise Integration (ESB)** | `TRANSIENT_LOCAL` | `RELIABLE` | `MEDIUM` | 33%–67% modal share |
| **06 Microservices** | `TRANSIENT_LOCAL` | `RELIABLE` | `MEDIUM` | 40%–60% modal share |
| **07 Enterprise Pub-Sub** | Mixed | `RELIABLE` | `MEDIUM` | 29%–79% modal share |
| **10 ATM System** | `VOLATILE` | `RELIABLE` | `HIGH` / `CRITICAL` | 41%–81% modal share |
| **18 Telecom RAN** | `VOLATILE` | `RELIABLE` | `HIGH` | 36%–67% modal share |
| **19 Industrial SCADA** | `TRANSIENT` | `RELIABLE` | `HIGH` | 31%–80% modal share |
| **20 Real-Time Gaming** | `VOLATILE` | `BEST_EFFORT` | `HIGH` | 39%–71% modal share |
| **21 Logistics Fleet** | `TRANSIENT_LOCAL` | `RELIABLE` | `MEDIUM` | 40%–76% modal share |
| **Autoware.universe (ROS 2)** | `VOLATILE` / `TRANSIENT_LOCAL` | `RELIABLE` / `BEST_EFFORT` | `CRITICAL` / `HIGH` | Real DDS profiles |
| **Cloud Microservices (Boutique)** | `PERSISTENT` / `VOLATILE` | `RELIABLE` / `BEST_EFFORT` | `HIGH` / `MEDIUM` | Kafka/Redis/NATS profiles |
| **Train-Ticket Mesh** | `PERSISTENT` / `TRANSIENT` | `RELIABLE` | `HIGH` / `MEDIUM` | SOA event-bus profiles |
| **Home Assistant (IoT)** | `VOLATILE` | `BEST_EFFORT` / `RELIABLE` | `HIGH` / `LOW` | MQTT / EventBus profiles |
| **EdgeX Foundry (IIoT)** | `TRANSIENT` / `VOLATILE` | `RELIABLE` | `HIGH` / `MEDIUM` | Industrial broker profiles |
| *08 Tiny Regression (fixture)* | Balanced | Balanced | Balanced | Pinned golden fixture |
| *09 XLarge Stress (fixture)* | Mixed | `RELIABLE` | `MEDIUM` | Stress testing |
| *11 Integration Hub (fixture)* | Mixed (no cat > 55%) | `RELIABLE` / `BEST_EFFORT` | `LOW` .. `CRITICAL` | Publisher-less topics |

---

## 5. Measured Empirical Outcomes

Evaluated on the 17-architecture corpus, 5 seeds per arm (`{42, 123, 456, 789, 2024}`), 300 epochs. Critical-set identification uses $K = \text{round}(0.20 \cdot |V_{\text{app}}|)$.

### 5.1 In-Distribution Performance Across 12 Scenarios (JSS Supp. Table S11)

Held-out in-distribution evaluation where each scenario is evaluated with a 60/20/20 train/val/test split across 5 seeds:

| Scenario | Held-Out $n$ | Topo ($\rho$) | Topo ($F_1$) | Topo-QoS ($\rho$) | Topo-QoS ($F_1$) | GAT ($\rho$) | GAT ($F_1$) | GAT-QoS ($\rho$) | GAT-QoS ($F_1$) | HGT ($\rho$) | HGT ($F_1$) | HGT-QoS ($\rho$) | HGT-QoS ($F_1$) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **ATM System** | 5 | 0.538 | 0.400 | 0.557 | 0.400 | −0.393 | 0.000 | −0.080 | 0.000 | 0.492 | 0.600 | 0.348 | 0.400 |
| **AV System** | 16 | 0.188 | 0.333 | 0.797 | 0.533 | 0.816 | 0.400 | 0.465 | 0.267 | 0.637 | 0.533 | 0.558 | 0.533 |
| **Enterprise** | 60 | 0.443 | 0.600 | 0.793 | 0.600 | 0.779 | 0.583 | 0.481 | 0.433 | 0.861 | 0.600 | 0.878 | 0.600 |
| **Financial Trading** | 12 | 0.387 | 0.200 | 0.512 | 0.400 | 0.565 | 0.400 | 0.666 | 0.400 | 0.693 | 0.500 | 0.730 | 0.500 |
| **Healthcare** | 10 | 0.291 | 0.200 | 0.399 | 0.000 | 0.725 | 0.300 | 0.575 | 0.300 | 0.575 | 0.400 | 0.607 | 0.500 |
| **Enterprise Integration (ESB)** | 14 | 0.179 | 0.267 | 0.429 | 0.400 | 0.363 | 0.467 | −0.156 | 0.067 | 0.421 | 0.400 | 0.476 | 0.400 |
| **Industrial SCADA** | 28 | 0.601 | 0.533 | 0.710 | 0.533 | 0.656 | 0.533 | 0.478 | 0.500 | 0.787 | 0.667 | 0.839 | 0.633 |
| **IoT Smart City** | 40 | 0.320 | 0.350 | 0.397 | 0.350 | 0.580 | 0.450 | 0.538 | 0.425 | 0.849 | 0.650 | 0.850 | 0.650 |
| **Logistics Fleet** | 22 | 0.511 | 0.500 | 0.652 | 0.400 | 0.746 | 0.500 | 0.780 | 0.550 | 0.796 | 0.550 | 0.815 | 0.500 |
| **Microservices** | 18 | 0.219 | 0.150 | 0.344 | 0.250 | 0.351 | 0.400 | 0.363 | 0.450 | 0.141 | 0.300 | 0.664 | 0.600 |
| **Real-Time Gaming** | 15 | 0.360 | 0.533 | 0.802 | 0.533 | 0.464 | 0.333 | 0.471 | 0.400 | 0.651 | 0.533 | 0.641 | 0.400 |
| **Telecom RAN** | 24 | 0.402 | 0.480 | 0.422 | 0.280 | 0.608 | 0.320 | 0.350 | 0.360 | 0.591 | 0.280 | 0.526 | 0.320 |
| **Mean** | — | **0.370** | **0.379** | **0.568** | **0.390** | **0.522** | **0.391** | **0.411** | **0.346** | **0.624** | **0.501** | **0.661** | **0.503** |

### 5.2 Inductive Out-of-Distribution Generalization (LOSO, 12 Folds)

Leave-One-Scenario-Out cross-validation evaluating zero-shot prediction on completely unseen topologies (JSS Tables 7 & 8):

| Predictor | Substrate | Mean LOSO $\rho$ | 95% Bootstrap CI | $\Delta\rho$ vs. Topo-QoS | Overlap@$K$ | Requires Training? |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **RM / $Q(v)$** | Flow Projection ($G_{\text{analysis}}$) | 0.205 | $[0.092, 0.320]$ | −0.348 | 0.322 | No (Rule Attribution) |
| **Topo** | Flow Projection ($G_{\text{analysis}}$) | 0.349 | $[0.254, 0.452]$ | −0.204 | 0.366 | No |
| **Topo-QoS** | Flow Projection ($G_{\text{analysis}}$) | **0.553** | $[0.443, 0.657]$ | — (Reference) | 0.388 | No (Closed-Form) |
| **GAT-N** | Native Multigraph | 0.317 | $[0.254, 0.381]$ | −0.236 | 0.328 | Yes |
| **GAT-N-QoS** | Native Multigraph | 0.604 | $[0.538, 0.665]$ | +0.051 | 0.431 | Yes |
| **HGT** | Native Multigraph | 0.551 | $[0.474, 0.617]$ | −0.002 | 0.427 | Yes |
| **HGT-QoS** (GPU) | Native Multigraph | 0.638 | $[0.561, 0.710]$ | +0.085 | 0.424 | Yes |
| **HGT-QoS** (CPU) | Native Multigraph | 0.622 | $[0.547, 0.690]$ | +0.069 | 0.426 | Yes |
| **GAT-N-QoS16-C** | Native Multigraph | 0.635 | $[0.567, 0.696]$ | +0.082 | 0.438 | Yes (Matched Control) |
| **SaG-Hybrid** | Native Multigraph + Prior | **0.657** | $[0.572, 0.733]$ | **+0.103** ($p=0.0068$) | 0.435 | Yes (HGT + Topo-QoS) |
| **SaG-Hybrid-GAT** | Native Multigraph + Prior | **0.683** | $[0.603, 0.753]$ | **+0.130** ($p=0.0029$) | **0.450** | Yes (GAT + Topo-QoS) |

### 5.3 Per-Fold LOSO Breakdown Across All 12 Architectures (JSS Supp. Table S12)

Per-fold Spearman $\rho$ across the 12 holdouts, ordered by closed-form `Topo-QoS` score:

| Holdout Scenario | Topo-QoS | HGT-QoS (CPU) | SaG-Hybrid | GAT-N-QoS16-C | SaG-Hybrid-GAT |
|---|---:|---:|---:|---:|---:|
| **Real-Time Gaming** | 0.810 | 0.789 | **0.837** | 0.685 | 0.825 |
| **Enterprise Pub-Sub** | **0.795** | 0.426 | 0.735 | 0.407 | 0.768 |
| **AV System** | 0.753 | 0.704 | 0.782 | 0.732 | **0.793** |
| **Logistics Fleet** | 0.741 | 0.771 | 0.792 | 0.654 | **0.806** |
| **Industrial SCADA** | 0.650 | 0.684 | 0.758 | 0.721 | **0.768** |
| **Financial Trading** | 0.586 | 0.695 | 0.754 | 0.713 | **0.797** |
| **Telecom RAN** | 0.576 | 0.427 | 0.648 | 0.574 | **0.656** |
| **Enterprise Integration (ESB)** | 0.430 | 0.548 | 0.564 | **0.630** | 0.568 |
| **Healthcare Integration** | 0.369 | 0.730 | 0.625 | **0.798** | 0.686 |
| **IoT Smart City** | 0.351 | 0.688 | 0.590 | **0.720** | 0.654 |
| **ATM System** | 0.311 | **0.523** | 0.429 | 0.506 | 0.447 |
| **Microservices Mesh** | 0.265 | 0.475 | 0.366 | **0.479** | 0.429 |
| **Mean** | **0.553** | **0.622** | **0.657** | **0.635** | **0.683** |

**Empirical Pattern:**
* Pure learned models (`HGT-QoS`, `GAT-N-QoS16-C`) excel on low-centrality, decentralized topologies where closed-form betweenness struggles (Microservices $+0.21$, ATM $+0.21$, IoT $+0.37$, Healthcare $+0.43$).
* Conversely, pure learned models give up ground on highly centralized topologies dominated by single bottleneck brokers (Enterprise $0.426$ vs. $0.795$).
* The hybrid formulations (`SaG-Hybrid` and `SaG-Hybrid-GAT`) resolve this failure mode: by learning a residual correction over rank-normalized `Topo-QoS`, they lift Enterprise back to $0.735$–$0.768$ while retaining learned gains on decentralized topologies.

### 5.4 Zero-Shot Transfer to Open-Source System Models (JSS Table 9b & 9c)

Models trained on the 12 synthetic architectures evaluated zero-shot on 5 independently authored open-source architectures:

| System Model | $|V_{\text{app}}|$ | $n_{>0}$ | RM ($\rho$) | Topo ($\rho$) | Topo-QoS ($\rho$) | HGT-QoS ($\rho$) | GAT-N-QoS16-C ($\rho$) | SaG-Hybrid ($\rho$) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Autoware.universe (ROS 2)** | 32 | 19 | 0.357 | 0.307 | 0.378 | 0.716 | 0.751 | 0.648 |
| **EdgeX Foundry (IIoT)** | 22 | 10 | 0.470 | 0.534 | 0.534 | 0.793 | 0.812 | 0.731 |
| **Home Assistant (IoT)** | 24 | 17 | 0.265 | 0.297 | 0.289 | 0.864 | 0.887 | 0.729 |
| **Online Boutique (pub-sub model)** | 22 | 8 | 0.777 | **0.891** | 0.888 | 0.710 | 0.785 | 0.692 |
| **Train-Ticket Booking Mesh** | 41 | 14 | 0.713 | 0.528 | 0.541 | 0.717 | 0.790 | 0.675 |
| **Mean** | — | — | **0.516** | **0.511** | **0.526** | **0.760** | **0.805** | **0.695** |

* **Active Stratum Correlation ($\rho_{>0}$)**: Restricted to components that actually propagate failures ($n_{>0}$), pure learned models keep positive correlations ($\text{HGT-QoS} = +0.236$, $\text{GAT-N-QoS16-C} = +0.319$), whereas training-free baselines turn negative ($\text{Topo-QoS} = -0.092$, $\text{Topo} = -0.083$, $\text{RM} = -0.055$).
* **Transfer Trade-Off**: For systems outside the synthetic generator family, pure learned models (`GAT-N-QoS16-C` at $0.805$, `HGT-QoS` at $0.760$) outperform hybrids ($0.695$), as the structural prior pulls the model toward generator-specific betweenness heuristics.

### 5.5 Convergent Validity Across Simulation Oracles (JSS §7.3.2)

Across the 12 synthetic folds, three distinct failure impact oracles measure convergent validity:
1. **$I^*(v)$ (Topological Cascade Reachability)**: Deterministic, breadth-first reachability computation over degraded dependency paths. Test-retest reproducibility across seeds: $\rho \in [0.811, 1.000]$ (median $0.982$).
2. **$I_{\text{dyn}}(v)$ (Discrete-Event Queue Simulation)**: Stochastic SimPy discrete-event simulation tracking message delivery ratios, buffer saturation, and queuing latency under dynamic load.
3. **Agreement**: Mean rank correlation between $I^*$ and $I_{\text{dyn}}$ across the 12 folds is **$\rho = 0.627$** (top-$K$ Jaccard $0.370$ vs. $0.111$ expected by chance). This moderate-to-high agreement confirms convergent validity between independent topological and behavioral formulations while reflecting real behavioral dynamics (e.g., crashing a publisher reduces downstream queue contention, creating negative tail-latency correlations $\rho = -0.499$).

### 5.6 Provenance: Instrument Defects Discovered and Fixed

Two historical defects affected early pre-revision drafts and are permanently documented to ensure reproducibility:

1. **`Topo-QoS` Projection Weight Lookup**: QoS attributes are declared on Topic nodes, but `_derive_depends_on_edges` historically queried the pub-sub relationship, which emitted bare `{from, to}` pairs. The lookup failed silently, leaving all derived dependency edges at unit weight $w(e) = 1.0$ and causing `Topo-QoS` to collapse to plain `Topo`. The current engine correctly resolves $w(t)$ from the shared Topic via `topic_weight_from_node_attrs`, taking $\max$ across multiple shared topics.
2. **HGT Attention Extraction**: `HGTConv` in PyG $\ge 2.5$ does not expose `return_attention_weights`, causing the extractor to catch a `TypeError` and return an empty dictionary. Attention weights are now intercepted directly from the layer's internal softmax output during the forward pass, restoring attention subgraph extraction (JSS Figure 5).

---

## 6. Adding or Modifying a Scenario

To introduce an architecture or modify an existing one:

1. Create a `scenario_NN_<name>.yaml` configuration. Ensure `graph.seed` is unique and not already listed in [`data/scenarios/MANIFEST.json`](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/data/scenarios/MANIFEST.json).
2. Configure `graph.counts` and tailor the `*_stats` distributions to reflect the domain's communication topology and QoS contracts.
3. Register the mapping in `SCENARIO_SYSTEM_MAP` ([`cli/common/batch_generation.py`](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/cli/common/batch_generation.py)) and in `CORPUS` ([`scripts/write_scenario_manifest.py`](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/scripts/write_scenario_manifest.py)). If authoring an authentic open-source system model, define its adapter in [`saag/adapters/realworld_adapter.py`](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/saag/adapters/realworld_adapter.py) and hook it into [`cli/import_realworld_system.py`](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/cli/import_realworld_system.py).
4. Regenerate the datasets, refresh the cryptographic manifest, and repopulate the caches:
   ```bash
   make -f reproduce/Makefile scenarios cache
   ```
5. Run the regression suite to verify integrity:
   ```bash
   PYTHONPATH=. python -m pytest tests/test_scenario_corpus.py tests/test_realworld_adapter.py -q
   ```
6. **Note on Population Invariants**: Adding an *evaluation* scenario updates the pooled population asserted by `test_evaluation_suite_matches_paper_population` ($2{,}387$ nodes for the 11 evaluation scenarios, or $2{,}461$ for all 12 synthetic folds) and invalidates existing LOSO cross-validation tables. Add fixtures freely; add evaluation scenarios only alongside a full experiment sweep.
