# Static System Analyzer

A pipeline tool for analyzing publish-subscribe distributed systems. It clones repositories, extracts topic metadata, aggregates relationships into JSON, computes descriptive statistics, and performs structural analysis with anomaly scoring.

---

## Pipeline Overview

```
┌─────────┐     ┌──────────┐     ┌ ─ ─ ─ ─ ┐     ┌──────────────┐     ┌───────────┐     ┌────────┐
│  CLONE  │ ──► │ ANALYZE  │ ──►  METRICS  ──► │  STRUCTURAL  │ ──► │ AGGREGATE │ ──► │  STAT  │
└─────────┘     └──────────┘     └ ─ ─ ─ ─ ┘     └──────────────┘     └───────────┘     └────────┘
  (HOST)          (HOST)         (DOCKER/opt)        (DOCKER)           (DOCKER)         (DOCKER)
```

| Stage | Input | Output |
|-------|-------|--------|
| **Clone** | `repo_names.txt`, CSV, `.env` | Cloned repos under `output/cloned/<platform>/` |
| **Analyze** | Cloned project folders | Per-project CSV files under `output/analyzed/<platform>/` |
| **Metrics** *(optional)* | Cloned Java project folders | Per-project `_metrics.json` under `output/analyzed/<platform>/` |
| **Structural** | Analyzed CSVs + SYSTEM_REPO + TypeSupport | `_structural_analysis.json` + `_structural.md` under `output/structural/<platform>/` |
| **Aggregate** | CSV files + `_metrics.json` + structural JSON | `<platform>_relations.json` (with `structural_analysis`) under `output/aggregated/` |
| **Stat** | Relations JSON | `<platform>_statistics.json` + `_report.pdf/.md` under `output/stat/` |

---

## Quick Start

```bash
cd src

# Full pipeline
python cmd/main.py --platform <name> --all

# Full pipeline with code metrics enabled
python cmd/main.py --platform <name> --all --enable-metrics

# Individual stages
python cmd/main.py --platform <name> --clone-only
python cmd/main.py --platform <name> --analyze-only
python cmd/main.py --platform <name> --metrics-only
python cmd/main.py --platform <name> --aggregate-only
python cmd/main.py --platform <name> --stat-only
python cmd/main.py --platform <name> --structural-only
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--platform, -p` | Platform name (required) |
| `--all, -a` | Run all stages |
| `--clone-only` | Clone repositories only |
| `--analyze-only` | Analyze cloned projects only |
| `--metrics-only` | Run code metrics scan only (ck, requires Docker) |
| `--enable-metrics` | Enable code metrics scan in full pipeline (default: disabled) |
| `--aggregate-only` | Aggregate CSVs to JSON only |
| `--stat-only` | Compute statistics only |
| `--structural-only` | Run structural analysis only |
| `--analysis-mode` | Analysis strategy: `manual` (default) or `codeql` |
| `--build` | Run `gmake regenerate_code` before analysis |
| `--skip-on-build-failure` | Skip project if build fails |
| `--verbose, -v` | Verbose output |
| `--log-level` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Configuration Files

Located in `config/`:

### 1. `.env`

A `.env.example` template is provided under `config/`. Fill in all required fields and rename it to `.env` before running the pipeline.

```bash
cp config/.env.example config/.env
# Edit config/.env with your actual credentials and settings
```

### 2. `runtime.yaml`

Environment-specific non-secret pipeline rules should live in `config/runtime.yaml`. A versionable template is provided as `config/runtime.example.yaml`.

```bash
cp config/runtime.example.yaml config/runtime.yaml
# Edit config/runtime.yaml with your environment-specific analyzer/aggregator rules
```

You can also point to an external file by setting `SSA_RUNTIME_CONFIG=/secure/path/runtime.yaml`.

### 3. `repo_names.txt`

List of repository names to clone (one per line). Lines starting with `#` are ignored.

```
flight_control_xy
navigation_yz
common_lib
```

### 4. `platform_pkg_version.csv`

Maps packages to versions per platform. Required columns:

| Column | Description |
|--------|-------------|
| `project_name` | Platform identifier (must match `--platform`) |
| `pkg_name` | Repository/package name |
| `pkg_version` | Git tag to clone |

```csv
project_name,pkg_name,pkg_version
avionics,flight_control_xy,v1.0.0
avionics,navigation_yz,v2.0.0
```

### 5. `csci_info.csv`

Maps each application or library name to its system hierarchy labels. Required columns:

| Column | Description |
|--------|-------------|
| `csu_name` | Application or library name matched against the aggregated entity name |
| `csc_name` | Top-level hierarchy name |
| `csci_name` | Intermediate hierarchy name |
| `css_name` | Subsystem hierarchy name |
| `csms_name` | Lowest-level hierarchy name |

```csv
csu_name,csc_name,csci_name,css_name,csms_name
flight_control_xy,Computer Science,Software Engineering,Software Design,Software Testing
common_lib,Computer Science,Software Engineering,Software Design,Shared Services
```

---

## Module Details

### 1. Cloner (`pipeline/cloner/`)

**Purpose:** Clone repositories from Bitbucket at specific tags.

**Clone Policy:**
- Reads `repo_names.txt` for target repos
- Filters `platform_pkg_version.csv` by `--platform` and `pkg_name`
- Only clones packages ending with the selected words.
- If multiple versions exist, clones **only the latest** (semver comparison)
- Performs **shallow clone** (`--depth 1`) for speed
- Removes `.git` directory after clone
- Clone timeout: 5 minutes per repo

**Output structure:**
```
output/cloned/<platform>/
├── flight_control_xy_v1.1.0/
├── navigation_yz_v2.0.0/
└── common_lib_v1.0.0/
```

---

### 2. Analyzer (`pipeline/analyzer/`)

**Purpose:** Extract topic (pub/sub topic) metadata and dependencies from each project.

The analyzer supports two analysis modes selected via `--analysis-mode`:

| Mode | Flag | Description |
|------|------|-------------|
| `manual` | `--analysis-mode manual` | XML parse + import parse. **Default.** |
| `codeql` | `--analysis-mode codeql` | CodeQL call-graph analysis. |

#### 2a. Manual Mode (default)

**Analysis Policy:**
- Scans each versioned folder under `output/cloned/<platform>/`
- **Valid project criteria:**
  - Has `<project_name>.xml` somewhere under `src/`
  - Has a `Makefile` containing `include/makefile.mk`
- Parses `<project_name>.xml` for `<topic name="..." role="...">` elements
- Ignores `DummyTopic` entries during parsing before any `pub` / `sub` / `pubsub` expansion
- Expands `pubsub` role into separate `pub` and `sub` entries
- Scans `.java` files recursively and extracts the first package segment after the configured domain prefix in `import` lines (for example `import a.b.c.<lib>...`) → creates `uses` relations
- Optionally runs `gmake regenerate_code` before parsing (with `--build` flag)

#### 2b. CodeQL Mode

**Purpose:** Starting from the `main` method, detect **write** (pub) and **read** (sub) calls reachable through any call chain. Extracts only the topics that are **actually reached at runtime**, not merely declared in XML.

**Reachability scope** — all of the following are included in the transitive call chain starting from `main`:
- Direct method calls
- Virtual dispatch (polymorphism)
- Interface dispatch
- Lambda / method reference
- Anonymous inner class
- Callback / event listener
- Reflection (`Method.invoke`)
- Nested / chained calls

**Topic name extraction:**
The topic name is not a string literal. It is extracted via regex from the **class type** of a specific argument:

```
custom_write(abcInstance)
  → abcInstance type: Abc_class
  → regex: "(.*)_class"
  → topic name: "Abc"
```

- The argument index (`topic_arg_index`) is specific to each write/read method definition.
- The topic name regex (`topic_name_pattern`) is global.

**Output rules:**

| Relationship | Rule |
|--------------|------|
| `pub` | Every write call reachable from `main` → application's pub |
| `sub` | Every read call reachable from `main` → application's sub |
| `uses` | If a **library's code** is traversed in the call chain leading to a write/read call |

- Write/read calls reached through library code are still the **application's** pub/sub (no distinction between direct and via-library).
- Importing alone is **not sufficient** for `uses`; library code must be **actually entered** in the call chain.
- Libraries also have a `main` method. When a library is analyzed independently, its own reachable write/read calls are extracted.

**CodeQL configuration (`runtime.yaml`):**

```yaml
analyzer:
  codeql:
    cli_path: "/usr/local/bin/codeql"
    java_home: "/opt/sdk/jdk21"
    build_command: "gmake clean all"
    main_method_name: "main"
    topic_name_pattern: "(.*)_class"
    write_methods:
      - class_name: "CustomWriter"
        method_name: "custom_write"
        topic_arg_index: 0
    read_methods:
      - class_name: "CustomReader"
        method_name: "custom_read"
        topic_arg_index: 0
```

| Key | Description |
|-----|-------------|
| `cli_path` | Path to the CodeQL CLI binary |
| `java_home` | Java home directory for CodeQL database creation |
| `build_command` | Build command executed during CodeQL database creation |
| `main_method_name` | Root method name for the call graph |
| `topic_name_pattern` | Regex (with capture group) to extract topic name from class name |
| `write_methods` | Method definitions that produce `pub` relationships (multiple allowed) |
| `read_methods` | Method definitions that produce `sub` relationships (multiple allowed) |
| `class_name` | Class containing the write/read method |
| `method_name` | Write/read method name |
| `topic_arg_index` | 0-based argument index containing the topic type |

#### Mode Comparison

| Feature | Manual | CodeQL |
|---------|--------|--------|
| Topic source | XML `<topic>` elements | Call-graph write/read calls |
| Uses source | Java import parse | Library code traversed in call chain |
| Evidence level | Declarative (what is defined) | Reachability (what is actually called) |
| Library inclusion | `uses` if imported | `uses` if library code is in call chain |
| Output format | Same CSV | Same CSV |

#### Output

One CSV per project (identical format for both modes):
```
output/analyzed/<platform>/<project_name>_<version>.csv
```

CSV format (no header):
```
folder_name,topic_name,role
flight_control_xy,FlightData,pub
flight_control_xy,NavigationCmd,sub
flight_control_xy,common_lib,uses
```

---

### 3. Code Metrics (`pipeline/analyzer/metrics_scanner.py`)

**Purpose:** Compute object-oriented code metrics for Java projects using [ck](https://github.com/mauricioaniche/ck) (Code Metrics for Java). A single tool provides all four metric categories — no server required, fully offline.

**How it works:**
1. Runs `java -jar ck.jar` on each project's source directory
2. Parses the generated `class.csv` output
3. Summarises per-class data into four categories: **size**, **complexity**, **cohesion**, **coupling**
4. Writes `_metrics.json` per application

**Metrics Collected:**

| Category | Metric | ck Column | Description |
|----------|--------|-----------|-------------|
| **Size** | `total_loc` | `loc` | Total lines of code across all classes |
| | `total_classes` | — | Number of classes analysed |
| | `total_methods` | `totalMethodsQty` | Sum of methods across all classes |
| | `total_fields` | `totalFieldsQty` | Sum of fields across all classes |
| **Complexity** | `total_wmc` | `wmc` | Sum of WMC (Weighted Methods per Class = Σ CCN) |
| | `avg_wmc` / `max_wmc` | `wmc` | Average / maximum WMC per class |
| | `high_complexity_classes` | `wmc` | Classes with WMC ≥ 50 (top 20, per-app JSON only) |
| **Cohesion** | `avg_lcom` / `max_lcom` | `lcom` | Average / maximum LCOM (Lack of Cohesion of Methods) |
| | `low_cohesion_classes` | `lcom` | Classes with LCOM ≥ 100 (top 20, per-app JSON only) |
| **Coupling** | `avg_cbo` / `max_cbo` | `cbo` | Average / maximum CBO (Coupling Between Objects) |
| | `avg_rfc` / `max_rfc` | `rfc` | Average / maximum RFC (Response For a Class) |
| | `avg_fanin` / `max_fanin` | `fanin` | Average / maximum Fan-in (incoming dependencies) |
| | `avg_fanout` / `max_fanout` | `fanout` | Average / maximum Fan-out (outgoing dependencies) |
| | `high_coupling_classes` | `cbo` | Classes with CBO ≥ 30 (top 20, per-app JSON only) |

> **CK compatibility note:** Stock ck does not emit comment-line totals. The analyzer therefore reports only metrics that are directly available from ck output.

> **Note:** Per-class detail lists (`high_complexity_classes`, `low_cohesion_classes`, `high_coupling_classes`) are included in per-app `_metrics.json` files but excluded from the aggregated `_relations.json` to keep it concise.

**Output:** One JSON per project:
```
output/analyzed/<platform>/<app_name>_metrics.json
```

---

### 4. Aggregator (`pipeline/aggregator/`)

**Purpose:** Merge all per-project CSVs into a single JSON representing the entire system graph.

**Data Sources:**
- Analyzed CSVs (pub/sub/uses relations)
- `SystemRepoParser`: app→node mappings, roles, criticality (placeholder implementation)
- `TypeSupportParser`: topic metadata (size, QoS) from TypeSupport/IDL files (placeholder)
- `config/csci_info.csv`: hierarchy metadata matched by entity name

**Output JSON structure:**
```json
{
  "metadata": { "scale": "{...}" },
  "nodes": [{ "id": "N0", "name": "Node-1", "structural_analysis": {...} }],
  "brokers": [],
  "topics": [{ "id": "T0", "name": "FlightData", "size": 1024, "qos": {...}, "structural_analysis": {...} }],
  "applications": [{ "id": "A0", "name": "flight_control_xy", "version": "v1.1.0", "system_hierarchy": { "csc_name": "Computer Science", "csci_name": "Software Engineering", "css_name": "Software Design", "csms_name": "Software Testing" }, "code_metrics": {...}, "structural_analysis": {...}, ... }],
  "libraries": [{ "id": "L0", "name": "common_lib", "version": "v1.0.0", "system_hierarchy": { "csc_name": "Computer Science", "csci_name": "Software Engineering", "css_name": "Software Design", "csms_name": "Shared Services" }, "code_metrics": null, "structural_analysis": {...} }],
  "relationships": {
    "runs_on": [{ "from": "A0", "to": "N0" }],
    "publishes_to": [{ "from": "A0", "to": "T0" }],
    "subscribes_to": [{ "from": "A1", "to": "T0" }],
    "uses": [{ "from": "A0", "to": "L0" }],
    "routes": []
  },
  "structural_analysis": {
    "parameters": {...},
    "quartiles": {...},
    "pattern_summary": {...}
  }
}
```

**QoS Conversion:** Uses `converter.py` to normalize durability, reliability, and transport_priority values.

When a matching row exists in `config/csci_info.csv`, each application and library includes a `system_hierarchy` object. Missing or blank hierarchy values are emitted as `NOT_FOUND`.

Applications and libraries include `code_metrics` when `_metrics.json` data is available; otherwise the field is `null`. When structural analysis output is available, aggregate output includes `structural_analysis` for matching nodes, topics, applications, and libraries, plus a top-level `structural_analysis` summary.

---

### 5. Structural Analysis (`pipeline/structural/`)

**Purpose:** Calculate structural metrics, detect anomaly patterns via relative quartile interpretation, and compute combined anomaly scores for all system components.

Structural analysis builds its own internal data model from raw sources (analyzed CSVs, SYSTEM_REPO, TypeSupport) independently of the aggregate module. Its results are saved to a standalone JSON file which the aggregate module then incorporates into the final relations JSON.

The analysis has three stages: *metric calculation*, *pattern detection*, and *anomaly scoring*.

#### Notation

| Symbol | Meaning |
|--------|---------|
| `PUB(a)` | Topics published by application `a` |
| `SUB(a)` | Topics subscribed by application `a` |
| `PUB(t)` | Applications publishing to topic `t` |
| `SUB(t)` | Applications subscribing to topic `t` |
| `RUNS(n)` | Applications running on node `n` |
| `USES(a)` | Libraries used by application `a` |
| `USES(l)` | Applications using library `l` |

#### Application-Level Metrics

| Metric | Name | Formula |
|--------|------|---------|
| **R(a)** | Reach | `\|{ a' ∈ A \ {a} \| (∃t ∈ PUB(a): a' ∈ SUB(t)) ∨ (∃t ∈ SUB(a): a' ∈ PUB(t)) }\|` |
| **AMP(a)** | Amplification | `R(a) / (\|PUB(a)\| + 1)` |
| **RA(a)** | Role Asymmetry | `(\|PUB(a)\| - \|SUB(a)\|) / (\|PUB(a)\| + \|SUB(a)\| + 1)` |
| **TC(a)** | Topic Context Diversity | `\|{ category(t) \| t ∈ PUB(a) ∪ SUB(a) }\|` |
| **LE(a)** | Library Exposure | `\|USES(a)\|` |

#### Topic-Level Metrics

| Metric | Name | Formula |
|--------|------|---------|
| **C(t)** | Coverage | `\|SUB(t)\| + \|PUB(t)\|` |
| **I(t)** | Imbalance | `\|\|SUB(t)\| - \|PUB(t)\|\| / (\|SUB(t)\| + \|PUB(t)\| + 1)` |
| **PS(t)** | Physical Spread | `\|{ n ∈ N \| ∃a ∈ SUB(t) ∪ PUB(t), a ∈ RUNS(n) }\|` |
| **LCR(t)** | Low Connectivity Ratio | `\|{a ∈ PUB(t) ∪ SUB(t) : \|PUB(a) ∪ SUB(a)\| ≤ k}\| / (\|PUB(t) ∪ SUB(t)\| + 1)` |

#### Node-Level Metrics

| Metric | Name | Formula |
|--------|------|---------|
| **ND(n)** | Node Density | `\|RUNS(n)\|` |
| **NID(n)** | Node Interaction Density | `\|{ (aᵢ,aⱼ) ⊆ RUNS(n) \| aᵢ ↔ aⱼ }\|` where `aᵢ ↔ aⱼ ⟺ ∃t: (aᵢ ∈ PUB(t) ∧ aⱼ ∈ SUB(t)) ∨ (aⱼ ∈ PUB(t) ∧ aᵢ ∈ SUB(t))` |

#### Library-Level Metrics

| Metric | Name | Formula |
|--------|------|---------|
| **LC(l)** | Library Coverage | `\|USES(l)\|` |
| **LCon(l)** | Library Concentration | `max over n ∈ N of \|RUNS(n) ∩ USES(l)\|` |

#### Relative Quartile Interpretation

Each metric is interpreted relative to its distribution across all components:

- **M(x)↑** — relatively high: `M(x) ≥ Q3(M)`
- **M(x)↓** — relatively low: `M(x) ≤ Q1(M)`
- When `Q1 = Q3` (degenerate): only absolute min/max extremes are evaluated.

#### Structural Anomaly Patterns

| Pattern | Level | Condition | Description |
|---------|-------|-----------|-------------|
| **WR** — Wide Reach | App | `R(a)↑ ∧ AMP(a)↑` | High reach with high amplification |
| **RS** — Role Skew | App | `RA(a)↑ ∨ RA(a)↓` | Strong producer or consumer imbalance |
| **CS** — Context Spread | App | `TC(a)↑` | Interacts across many topic categories |
| **SD** — Shared Dependency Exposure | App | `LE(a)↑` | Depends on many shared libraries |
| **CB** — Communication Backbone | Topic | `C(t)↑ ∧ I(t)↓` | High coverage, balanced pub/sub |
| **DC** — Directional Concentration | Topic | `I(t)↑` | Heavily skewed pub or sub direction |
| **PA** — Peripheral Aggregator | Topic | `LCR(t)↑` | Aggregates low-connectivity apps |
| **IH** — Interaction Hotspot | Node | `ND(n)↑ ∧ NID(n)↑` | Dense node with heavy internal interaction |
| **WUL** — Widely Used Library | Library | `LC(l)↑` | Used by many applications |
| **CL** — Concentrated Library | Library | `LCon(l)↑` | Usage concentrated on specific nodes |

#### Combined Anomaly Score

The final score combines two components:

1. **Pattern-based score** `OS^P(x)`: Rewards rare pattern matches more heavily.
   ```
   OS^P(x) = Σ_p (1 / |{x' : p(x')}|) · I[p(x)]
   ```

2. **Single-dimension contribution** `UNI(x)`: Bounded contribution from individual metric extremes.
   ```
   u_M(x) = (M(x) - Q3) / (max - Q3)    if M(x) > Q3
   c_M(x) = min(u_M(x), τ)
   UNI(x) = Σ_M c_M(x)
   ```

3. **Final score**: `Score(x) = OS^P(x) + λ · UNI(x)` (default: `τ=0.3`, `λ=0.1`)

**Outputs:**
- `<platform>_structural_analysis.json` — machine-readable results (in `output/structural/<platform>/`)
- `<platform>_structural.md` — human-readable Turkish markdown report (in `output/structural/<platform>/`)

---

### 6. Stat (`pipeline/stat/`)

**Purpose:** Compute descriptive statistics on the aggregated system model.

Analyze and aggregate stages keep only direct `uses` dependencies. Recursive "including libraries" metrics are computed in stat by traversing the direct `uses` graph transitively.

Metric IDs follow a consistent convention: `<section>_<domain>_<measure>`.
This keeps metrics unique across report sections and makes new metrics easy to add without relying on substring-based matching.

**Metrics Computed:**

| Category | Metric | Description |
|----------|--------|-------------|
| **Nodes** | `node_application_count` | How many apps run on each node |
| | `node_domain_hierarchy_diversity_count` | Number of distinct `css_name` groups colocated on each node |
| **Applications** | `app_direct_publish_count` | Direct pub count per app |
| | `app_direct_subscribe_count` | Direct sub count per app |
| | `app_total_publish_count` | Recursive pub (app + used libs) |
| | `app_total_subscribe_count` | Recursive sub (app + used libs) |
| | `app_role_distribution` | Role distribution (how many apps per role) |
| | `app_criticality_distribution` | Criticality distribution (how many apps per level) |
| | `app_hierarchy_component_distribution` | Application count per `csc_name` |
| | `app_hierarchy_config_item_distribution` | Application count per `csci_name` |
| | `app_hierarchy_domain_distribution` | Application count per `css_name` |
| | `app_hierarchy_system_distribution` | Application count per `csms_name` |
| | `app_hierarchy_domain_avg_direct_publish_count` | Average direct pub per app within each `css_name` |
| | `app_hierarchy_domain_avg_direct_subscribe_count` | Average direct sub per app within each `css_name` |
| | `app_hierarchy_domain_topic_variety_count` | Distinct topics touched by each `css_name` group |
| | `app_hierarchy_config_item_avg_direct_publish_count` | Average direct pub per app within each `csci_name` |
| | `app_hierarchy_config_item_avg_direct_subscribe_count` | Average direct sub per app within each `csci_name` |
| | `app_hierarchy_config_item_topic_variety_count` | Distinct topics touched by each `csci_name` group |
| **Libraries** | `lib_application_usage_count` | How many apps use this library |
| | `lib_direct_publish_count` | Direct pub count per library |
| | `lib_direct_subscribe_count` | Direct sub count per library |
| | `lib_total_publish_count` | Recursive pub (library + its used libs) |
| | `lib_total_subscribe_count` | Recursive sub (library + its used libs) |
| | `lib_hierarchy_config_item_distribution` | Library count per `csci_name` |
| | `lib_hierarchy_domain_distribution` | Library count per `css_name` |
| | `lib_hierarchy_completeness_percent` | System hierarchy completeness percentage per library |
| **Topics** | `topic_size_bytes` | Data size (bytes) |
| | `topic_publisher_application_count` | Publisher count |
| | `topic_subscriber_application_count` | Subscriber count |
| | `topic_qos_durability_distribution` | QoS Durability distribution |
| | `topic_qos_reliability_distribution` | QoS Reliability distribution |
| | `topic_qos_transport_priority_distribution` | QoS Transport Priority distribution |
| **Structural** | `structural_top_apps` | Top 10 apps by structural anomaly score |
| | `structural_top_topics` | Top 10 topics by structural anomaly score |
| | `structural_top_nodes` | Top 10 nodes by structural anomaly score |
| | `structural_top_libs` | Top 10 libraries by structural anomaly score |
| **Extras** | `uses_cycle_distribution` | Cyclic uses dependency chains (all simple directed cycles) |

**Statistics per metric:** count, mean, median, std, min, max, Q1, Q3, IQR

For `app_hierarchy_domain_topic_variety_count` and `app_hierarchy_config_item_topic_variety_count`, "topic variety" means the number of unique topics that the applications in that hierarchy bucket touch through direct publish or direct subscribe relationships.

**Extras (Cross-Cutting Charts):**

Each chart includes statistical outlier detection (IQR or z-score based) with visual markers and a separate top-10 outlier list.

| # | Chart | Type | Outlier Method |
|---|-------|------|----------------|
| 1 | Topic Size vs. Subscriber Count | Scatter | 2D z-score distance |
| 2 | Topic Size Distribution by QoS | Scatter (3 panels: Durability, Reliability, Transport Priority) | IQR per QoS group |
| 3 | Application Publish/Subscribe Balance | Scatter (with quadrant lines) | 2D z-score distance |
| 4 | Topic Fanout (Publisher vs. Subscriber) | Scatter | 2D z-score distance |
| 5 | Cross-Node Communication Heatmap | Heatmap | IQR on non-zero cells |
| 6 | Node Communication Load | Stacked horizontal bar | IQR on total load |
| 7 | Domain-to-Domain Communication Heatmap | Heatmap | IQR on non-zero cells |
| 8 | Criticality × I/O Load | Scatter | 2D z-score distance |
| 9 | Library Dependency Density | Bar (in-degree / out-degree) | IQR on total degree |
| 10 | Node Critical App Density | Stacked bar (critical vs normal) | IQR on total count |
| 11 | Domain App & Topic Diversity | Bubble (X=app count, Y=topic variety, size=total I/O) | 2D z-score distance |

**Outputs:**
- `<platform>_statistics.json` — machine-readable
- `<platform>_report.pdf` / `<platform>_report.md` — human-readable report (includes TOC, per-category tables, extras charts with outlier pages)

---

## Output Directory Structure

All outputs are under `output/`:

```
output/
├── cloned/<platform>/          # Cloned repos
├── analyzed/<platform>/        # Per-project CSVs + _metrics.json
├── structural/<platform>/      # Structural analysis JSON + Markdown report
├── aggregated/                 # <platform>_relations.json (includes structural_analysis)
├── stat/                       # Statistics JSON + PDF/MD report
└── logs/                       # Stage-specific log files
```

---

## Example Workflow

```bash
# 1. Configure credentials
cp config/.env.example config/.env
# Edit .env with your Bitbucket TOKEN, USERNAME, BASE_URL

# 2. List repos to clone
echo "flight_control_xy" >> config/repo_names.txt
echo "navigation_yz" >> config/repo_names.txt

# 3. Populate version CSV
# Ensure platform_pkg_version.csv has entries for your platform

# 4. Run full pipeline
python cmd/main.py --platform avionics --all --verbose

# 4b. Run full pipeline with code metrics
python cmd/main.py --platform avionics --all --verbose --enable-metrics

# 5. Check outputs
ls output/aggregated/
ls output/stat/
ls output/structural/
```

---

## Requirements

**Host Machine:**
- **Python 3.9+** (Required for orchestrator and clone stage)
- **Git** (Required for cloning)
- `python-dotenv` (Recommended for loading credentials from `.env`)
  ```bash
  pip install python-dotenv
  ```

**Docker Runtime:**
## Hybrid Execution Model

The application uses a hybrid execution model to balance ease of use with environment consistency:

1.  **Clone Stage (Runs on HOST)**:
    - Uses your host's `python3` and `git`.
    - Bypasses potential SSL/Certificate issues common in corporate Docker environments.
    - Uses your host's network and credentials directly.

2.  **Analyze/Aggregate/Stat Stages (Run in DOCKER)**:
    - Runs inside a controlled Docker container.
    - Ensures all complex dependencies (`pandas`, `matplotlib`) are present and consistent.
    - Mounts source code and output directories from the host.

### Build Runtime Image
Before running analysis, build the Docker image:
```bash
make docker-build
```

### Running the Pipeline
Use the `Makefile` to run the pipeline. It handles the switching between Host and Docker automatically.

```bash
# Run full pipeline (Clone on Host -> Others in Docker)
make all PLATFORM_NAME=avionics

# Run full pipeline with code metrics enabled
make all PLATFORM_NAME=avionics METRICS=1

# Run individual stages
make clone PLATFORM_NAME=avionics      # Runs on HOST
make analyze PLATFORM_NAME=avionics    # Runs in DOCKER
make stat PLATFORM_NAME=avionics       # Runs in DOCKER
make structural PLATFORM_NAME=avionics # Runs in DOCKER
```

### Interactive Shell
To open a bash shell inside the container (with your project files mounted) for debugging:
```bash
make docker-shell
```
*Note: The shell prompt will be `developer@static-analyzer:/app$`.*

---

## Offline Deployment

To deploy the application to an offline environment:

1.  **Create Distribution Package**:
    Run this command on a machine with internet access and Docker:
    ```bash
    make dist
    ```
    This will create `static-system-analyzer-offline.tar.gz`.

2.  **Transfer and Setup**:
    Copy the archive to the offline machine and extract it:
    ```bash
    tar -xzf static-system-analyzer-offline.tar.gz
    cd static-system-analyzer
    ```

3.  **Load Docker Image**:
    ```bash
    docker load -i image.tar
    ```

4.  **Run Application**:
    You can now use standard `make` commands, which will transparently use the Docker environment.
    ```bash
    make all PLATFORM_NAME=avionics
    ```
    
    **Note**: Since the source code and configuration are mounted from the host, any changes you make to files in `src/` or `config/` will be immediately effective without needing to rebuild the Docker image.
