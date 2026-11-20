# Software-as-a-Graph (SaG) — SoSyM Tool Paper
### Short Advisor Walkthrough

* **Venue:** *Software and Systems Modeling* (SoSyM, Springer) — **Tool Paper**
* **Title:** *Software-as-a-Graph: An Open-Source Tool for Multi-Layer Graph Modeling, Quality Diagnosis, and Simulation-Verified Refactoring of Publish–Subscribe Architectures*
* **Status:** Revised to strict Tool Paper scope; all claims & numbers synced with current release (`58172b5f`)

---

## 1. The Problem

* Modern robotics, automotive, IoT, and event-driven systems rely on **publish–subscribe middleware** (ROS 2/DDS, Kafka, MQTT).
* Pub–sub **decouples** components via topics, brokers, and QoS contracts—which **hides failure-propagation paths**.
* **Tooling gap:**
  * **ADLs (AADL, Palladio):** Rich analyses, but no first-class topics, brokers, or DDS-style QoS.
  * **Smell detectors (Arcan, Designite, DV8, SonarQube):** Operate on source-code graphs where pub–sub interactions are invisible.

> **Goal:** One open-source tool where architects can model pub–sub systems natively, diagnose structural risks, gate CI changes, and verify refactorings.

---

## 2. What SaG Does (The Pipeline)

```text
JSON Model ──► [1. Model] ──► [2. Analyze] ──► [3. Diagnose] ──► [4. Simulate] ──► [5. Prescribe]
              Derive 7       Explainable      19 Anti-Patterns   Structural      3 Refactoring Ops
              DEPENDS_ON     RM Quality       + Delta CI Gate    Failure         Filtered by
              rules          Model            + Waivers          Oracle          Simulation
```

* **5 Interfaces:** CLI, Python API (`Pipeline` / `Client`), FastAPI REST service, Next.js/React web dashboard, and self-contained HTML reports.
* **Open Source:** GNU GPL v3.0, `pip install -e .`, and `docker compose up`.

---

## 3. Modeling Language & Running Example

* **Metamodel:** Typed, attributed directed multigraph:
  * **6 vertex types:** `Application`, `Library`, `Topic`, `Broker`, `Node`, `NetworkDevice`
  * **7 edge types:** 6 declared structural relations + 1 derived `DEPENDS_ON`
* **7 Derivation Rules:** Automatically infer hidden dependencies (app-to-app, app-to-broker, host-to-host, etc.) weighted by QoS, message size, and frequency.
* **4 Layer Projections:** `app`, `infra`, `mw`, `system` — passed explicitly because cut-based metrics (SPOFs, bridges) are only meaningful on layers that actually have cut vertices (**Proposition 1**).
* **Running Example (`mini-ATM`):** A 15-element air-traffic system carried consistently across §§3–7 and 9 executable walkthroughs.

---

## 4. Key Analyses

1. **Explainable RM Quality Model (ISO/IEC 25010):**
   * Linear decomposition into **Fault Tolerance**, **Availability**, and **Maintainability** (including code metrics).
   * Tells *why* a component is risky (e.g., *ops-host* needs redundancy; *tracker* needs decoupling).
2. **19 Pub–Sub Anti-Patterns + Delta-Aware CI Gate:**
   * Includes pub–sub-specific patterns (`QOS_MISMATCH`, `CHATTY_PAIR`, `BROKER_OVERLOAD`).
   * **Delta mode (`--baseline`)** blocks only *new* CRITICAL/HIGH findings, with a 10% fence margin and expiring waivers.
3. **Simulation-Verified Refactoring (`Prescribe`):**
   * Compiles findings into 3 operators (**Topic split**, **Anti-affinity reallocation**, **QoS hardening**).
   * Admits an edit **only** if simulated impact drops beyond seed noise across all propagation thresholds ($\Theta = \{0.1, 0.2, 0.5\}$).

---

## 5. Architecture & Engineering Safeguards

* **Hexagonal Design:** Core logic depends only on `IGraphRepository`:
  * `Neo4jRepository` → persistent exploration & web UI.
  * `MemoryRepository` → originally a test double, now the fast isolated sandbox powering 13,653 parallel refactoring simulations.
* **~1,600 Automated Tests** enforcing core research invariants:
  * Simulation oracle **never** reads derived edges (strict independence).
  * Training labeler (`FaultInjector`) and validation oracle (`FailureSimulator`) cannot be swapped.
  * Unmeasured gates return `None`, never `0.0`.
  * Corpus regenerates byte-identically (SHA-256 manifest); every paper number is macro-generated from provenance-stamped JSON.

---

## 6. Headline Results

| Dimension | Key Finding |
| :--- | :--- |
| **Speed** | **0.05–0.11 s** on 5 open-source models (60–90 comps); **0.04–5.53 s** up to 326 comps (**78.9 s** at 520 comps). |
| **Open-Source Models** | **14–22%** of components directly flagged; better oracle agreement ($\kappa = +0.26$) than synthetic models ($\kappa = -0.03$). |
| **RM vs. Centrality** | On `app`, RM $Q$ achieves $\rho = 0.62$ vs. $0.49$ for degree—beating degree in **8 of 8** scenarios ($p = 0.008$). Weak on infra hosts/brokers. |
| **Verified Refactoring** | **32%** of 1,510 rule-generated edits fail verification (68.1% admitted). Combining admitted edits helps in **4/7** scenarios but regresses in **3/7**. |
| **Delta CI Gate** | On 90 random edits: absolute gate blocks **100%**; delta gate blocks **26%**, warns **47%**, passes **28%**. |

---

## 7. Scope & Overlap Cleanup (Revision Highlights)

* **Clean Tool-Paper Positioning:** Focused on the modeling language, architecture, interfaces, extension points, scenario applications, and lessons learnt.
* **Zero Overlap with Companion Work:**
  * Explicitly discloses RASSE 2025, UYMS 2026 (×2), the JSS learning manuscript (under review), and *SaG-Prescribe* (held in reserve).
  * **0 shared tables/figures** and **0 shared prose shingles** (beyond the RM formulas in Appendix A).
* **Every Claim Verified Against Code:** Removed stale API/CLI/3D-view claims, re-measured all numbers on the current release, and fixed 3 real visualization bugs surfaced during the audit.

---

## 8. Lessons Learnt (Strongest SoSyM Differentiator)

Eight defect-grounded lessons transferable to any modeling tool:

1. **Specify the view per metric** (`SPOF` has $F_1 = 0.01$ on `app` because topics remove cut vertices).
2. **Make "not measured" unrepresentable as `0.0`** (prevents silent fake failures).
3. **Test doubles can become production adapters** (`MemoryRepository` enabled counterfactual verification).
4. **Never trust a refactoring rule—or rule composition—without simulation** (32% fail; 3/7 compositions regress).
5. **Tie every number to versioned artefacts** (fixing the oracle reversed an earlier RM-vs-degree tie).
6. **Design CI gates for the merge decision** (delta diff + fence margin + waivers).
7. **Executable walkthroughs are self-testing docs** (caught code–doc drift missed by unit tests).
8. **Keep multigraph semantics end to end** (simple digraphs collapse parallel pub/sub edges).

---

## 9. Next Steps Before Submission

**Required for upload:**
* [ ] Fill in affiliations, ORCIDs, e-mail, and TÜBİTAK grant number (`manuscript.tex`, `declarations.tex`).
* [ ] Mint a Zenodo DOI for the tool release and replace `[TODO]` placeholders.
* [ ] Switch `article` class to Springer Nature `sn-jnl` template (`sn-basic.bst`).

**Optional discussion points with advisor:**
* **Known `ISOLATED` topic false positive:** Keep transparently documented as a limitation (§4.2, §9.1), or patch and re-run artifacts before submission?
* **Companion *SaG-Prescribe* (held in reserve):** Note that its earlier RM-vs-degree "saturation" result is reversed on the current release and will need re-measurement before any future submission.
