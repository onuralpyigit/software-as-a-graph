# Software Installation & User Manual (SUM)
## Software-as-a-Graph (saag) Framework & Genieus Toolkit

**Standard Alignment:** Conforming to **ISO/IEC/IEEE 26511:2018** (Requirements for managers of user documentation) and aligned with the Software Operation and Maintenance processes of **ISO/IEC/IEEE 12207:2026**.

**Document Version:** 1.0  
**Release Date:** June 2026  
**Target Systems:** core SDK (`saag/`), REST API (`api/`), CLI Pipeline (`cli/`), and SMART Web Toolkit (`smart/`).

---

## 1. Introduction & System Overview

### 1.1 Document Purpose
This Software Installation & User Manual (SUM) provides instructions for installing, configuring, administering, and operating the Software-as-a-Graph (saag) framework, the command line interface (CLI) pipeline, the FastAPI REST API, and the Next.js web application (Genieus).

### 1.2 System Overview
The **Software-as-a-Graph (saag)** framework models pub-sub software system topologies as weighted directed graphs in Neo4j, executing a 7-stage analytical and simulation pipeline to predict critical components (Single Points of Failure, cascade hubs, bottleneck nodes) that present the highest risk of systemic failure if compromised or disrupted.

The system is structured as four core components:
1. **Core SDK (`saag/`):** Python library containing domain models, graph traversal heuristics, simulation engines, and GNN estimators.
2. **CLI Scripts (`cli/`):** Console pipeline entry points allowing researchers and operators to execute pipeline stages independently or as an orchestrated batch.
3. **REST API (`api/`):** FastAPI application acting as a gateway for programmatically triggering analysis, GNN training, and failure cascades.
4. **Genieus Web Toolkit (`smart/`):** A Next.js 16 + React 19 interactive single-page web dashboard offering force-directed graph rendering, metrics visualization, and validation reports.

---

## 2. System Prerequisites

Before installation, verify that the host system meets the following hardware and software requirements.

### 2.1 Hardware Requirements
- **Minimum:** 2 CPU cores, 8 GB RAM, 5 GB available disk space.
- **Recommended (GNN training on large graphs):** 4+ CPU cores, 16+ GB RAM, Nvidia GPU with CUDA support (optional; CPU execution is fully supported).

### 2.2 Operating System Requirements
- **Linux:** Ubuntu 22.04 LTS or newer (Fully verified, primary development OS).
- **macOS:** macOS Ventura (13.0) or newer (Apple Silicon and Intel).
- **Windows:** Windows 10/11 running via Windows Subsystem for Linux (WSL2 with Ubuntu 22.04).

### 2.3 Software Dependencies
Verify the installation of the following dependencies before proceeding with native deployment:

| Software | Required Version | Purpose | Verify Command |
|---|---|---|---|
| **Python** | `>= 3.9` (3.11 Pinned) | Core SDK, API, and GNN backend execution | `python3 --version` |
| **Node.js** | `>= 18.x` (20.x Rec.) | Next.js Frontend server compile & build | `node --version` |
| **npm** | `>= 9.x` | Next.js Frontend package management | `npm --version` |
| **Neo4j** | `5.x` | Graph Database store | `neo4j --version` |
| **Docker** | `>= 20.10.x` | Optional containerized runner | `docker --version` |
| **Docker Compose** | `>= v2.x` | Multi-container orchestration | `docker compose version` |

### 2.4 Neo4j Plugin Prerequisites
If utilizing a native, non-Dockerized Neo4j database, you must install the following plugins in your Neo4j instance:
1. **APOC** (Awesome Procedures on Cypher): Matching your Neo4j version.
2. **Graph Data Science (GDS)**: Matching your Neo4j version.
3. **Custom Graph Relationship Manager Plugin**: Pre-compiled custom JAR mapping transitive topology weights, located under `tools/neo4j-plugin/graph-relationship-manager/`.

---

## 3. Installation & Deployment

This section describes two installation methods: **Option A (Containerized Deployment)** for production-like environments or quick evaluations, and **Option B (Native Local Installation)** for development and local experimentation.

---

### 3.1 Option A: Containerized Deployment (Recommended)

The entire full-stack application (Next.js, FastAPI, Neo4j, and GNN libraries) is packaged into a single all-in-one Docker image, managed via Docker Compose.

```
       Host Machine Port Maps
 ┌────────────────────────────────────────────────────────┐
 │   7000 (HTTP)  -->  Next.js Frontend (Genieus)         │
 │   8000 (HTTP)  -->  FastAPI REST API                   │
 │   7474 (HTTP)  -->  Neo4j Browser Console              │
 │   7687 (Bolt)  -->  Neo4j Bolt Database Endpoint       │
 └────────────────────────────────────────────────────────┘
```

#### Step 1: Clone the Repository
Clone the repository and enter the root workspace directory:
```bash
git clone <repository_url> SoftwareAsAGraph
cd SoftwareAsAGraph
```

#### Step 2: Configure Environment Variables
Copy the default environment template into a `.env` file at the repository root:
```bash
cp .env.template .env
```
Ensure the default content matches the Docker network setup:
```ini
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_URI=bolt://localhost:7687
NEXT_PUBLIC_API_URL=http://localhost:8000
```

#### Step 3: Build and Start the Container
Build the single image and launch the stack in the background:
```bash
docker compose up --build -d
```

#### Step 4: Verify Deployment Health
Check the startup logs and verify container execution:
```bash
docker compose logs -f genieus
```
You should see output indicating that all ports (7000, 8000, 7474, 7687) are open and running. Test the API health endpoint:
```bash
curl http://localhost:8000/health
# Expected: {"status":"ok","database":"connected"}
```

#### Step 5: Stopping the Services
To shut down the containerized environment without deleting data:
```bash
docker compose down
```
To clear the Neo4j database volumes and perform a clean teardown:
```bash
docker compose down -v
```

---

### 3.2 Option B: Native Local Installation (Development)

Follow these steps to configure the backend virtual environment, build the custom Neo4j plugin, and compile the frontend packages natively.

#### Step 1: Configure Python Virtual Environment
Initialize a Python virtual environment and upgrade the package manager:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

#### Step 2: Install Python Dependencies & Core SDK
Install the core library along with database, API, and GNN dependencies:
```bash
pip install -e ".[all]"
```
> [!NOTE]
> This command installs PyTorch and PyTorch Geometric packages mapped to your system's hardware configuration (CPU/CUDA) as defined in [pyproject.toml](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/pyproject.toml).

#### Step 3: Build Custom Neo4j Plugin
Build the custom Neo4j plugin using Maven:
```bash
cd tools/neo4j-plugin/graph-relationship-manager
mvn clean package -DskipTests
```
Copy the compiled JAR file from `target/graph-relationship-manager-1.0.0.jar` to your local Neo4j plugins directory (typically `/var/lib/neo4j/plugins/` or `plugins/` inside your Neo4j installation home).

#### Step 4: Configure Local Neo4j Instance
Add the following configuration lines to your local `neo4j.conf` (usually `/etc/neo4j/neo4j.conf` or `conf/neo4j.conf`):
```ini
dbms.security.procedures.unrestricted=custom.*
dbms.security.procedures.allowlist=custom.*
```
Restart your local Neo4j instance to load the custom plugin:
```bash
sudo systemctl restart neo4j
```

#### Step 5: Install & Build Next.js Frontend
Enter the frontend project folder, install npm dependencies, and compile static assets:
```bash
cd smart
npm install
npm run build
```

---

## 4. Configuration Reference

The application is configured using environment variables defined in the root [.env](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/.env) file or passed directly to container environments.

### 4.1 Environment Variables Matrix

| Variable Name | Default Value | Description |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | Connection endpoint for the Neo4j database (use Bolt protocol). |
| `NEO4J_USERNAME` | `neo4j` | Database administrator username. |
| `NEO4J_PASSWORD` | `password` | Database administrator password. |
| `NEO4J_AUTH` | `neo4j/password` | Combined credentials used by Neo4j's Docker container. |
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Address of the FastAPI server consumed by the Next.js UI. |

---

## 5. Step-by-Step CLI User Guide

The CLI scripts provide granular, sequential access to all stages of the Software-as-a-Graph pipeline. All scripts must be run from the root workspace directory with the virtual environment activated.

```
┌───────────┐     ┌───────┐     ┌─────────┐     ┌─────────┐     ┌──────────┐     ┌──────────┐     ┌───────────┐
│ Generate  │ ──> │ Model │ ──> │ Analyze │ ──> │ Predict │ ──> │ Simulate │ ──> │ Validate │ ──> │ Visualize │
│  Step 0   │     │  Step 1   │     │  Step 2 │     │  Step 3 │     │  Step 4  │     │  Step 5  │     │  Step 6   │
└───────────┘     └───────┘     └─────────┘     └─────────┘     └──────────┘     └──────────┘     └───────────┘
```

---

### 5.1 Step 0: Synthetic Graph Generation
Generates synthetic topology JSON representing a pub-sub communication system (brokers, topics, applications, libraries).
- **Script:** [cli/generate_graph.py](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/cli/generate_graph.py) or `saag-generate`
- **Arguments:**
  - `--scale`: Preset scale (`tiny`, `small`, `medium`, `large`, `jumbo`, `xlarge`).
  - `--output`: File path to save the generated topology JSON.
- **Example:**
  ```bash
  python cli/generate_graph.py --scale medium --output data/synthetic_medium.json
  ```

---

### 5.2 Step 1: Model Import & Export
Imports a system topology JSON file into the Neo4j database, building nodes and deriving QoS-weighted `DEPENDS_ON` relationships.
- **Script:** [cli/import_graph.py](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/cli/import_graph.py) or `saag-import`
- **Arguments:**
  - `--input`: Path to topology JSON file.
  - `--clear`: Clear database before import.
- **Example:**
  ```bash
  python cli/import_graph.py --input data/synthetic_medium.json --clear
  ```

---

### 5.3 Step 2: Structural Analysis
Analyzes the imported graph in Neo4j to compute centrality metrics, anti-patterns, and baseline RMAV and Q(v) quality scores.
- **Script:** [cli/analyze_graph.py](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/cli/analyze_graph.py) or `saag-analyze`
- **Arguments:**
  - `--layer`: Targets specific system layers (`app`, `infra`, `mw`, `system`).
  - `--use-ahp`: Use AHP-derived dimension weights instead of uniform weights.
  - `--output`: Path to export analysis metrics JSON.
- **Example:**
  ```bash
  python cli/analyze_graph.py --layer system --use-ahp --output output/structural_metrics.json
  ```

---

### 5.4 Step 3: GNN Training & Prediction

#### GNN Model Training
Trains the GAT (Graph Attention Network) on simulated fault labels to predict component criticality.
- **Script:** [cli/train_graph.py](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/cli/train_graph.py)
- **Arguments:**
  - `--layer`: Targets specific system layer (`app`, `infra`, `mw`, `system`).
  - `--epochs`: Maximum training epochs (default: 300).
  - `--lr`: Learning rate (default: 3e-4).
  - `--checkpoint`: Directory to save the GNN model checkpoints (default: `output/gnn_checkpoints`).
  - `--variant`: Architecture baseline (`hetero_qos`, `homo_unweighted`, `homo_scalar`).
- **Example:**
  ```bash
  python cli/train_graph.py --layer system --epochs 200 --checkpoint models/system_checkpoints
  ```

#### GNN Model Prediction (Inference)
Executes inference on a Neo4j graph using a trained GNN checkpoint and combines it with structural scores via learnable ensemble blending.
- **Script:** [cli/predict_graph.py](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/cli/predict_graph.py) or `saag-predict`
- **Arguments:**
  - `--gnn-model`: Path to model checkpoint directory.
  - `--layer`: Targets specific system layer.
  - `--output`: Path to save prediction metrics JSON.
- **Example:**
  ```bash
  python cli/predict_graph.py --gnn-model models/system_checkpoints --layer system --output output/predictions.json
  ```

---

### 5.5 Step 4: Cascade Failure Simulation
Injects synthetic failures and evaluates cascade propagation, change propagation, and service disruptions to compute ground-truth impact labels.
- **Script:** [cli/simulate_graph.py](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/cli/simulate_graph.py) or `saag-simulate`
- **Arguments:**
  - `--layer`: Targets specific system layer.
  - `--sim-mode`: Simulation strategy (`exhaustive`, `monte_carlo`).
  - `--output`: Path to save simulation results JSON.
- **Example:**
  ```bash
  python cli/simulate_graph.py --layer system --sim-mode exhaustive --output output/simulation_results.json
  ```

---

### 5.6 Step 5: Statistical Validation
Compares analytical predictions (Step 3) against simulation ground-truth (Step 4) to verify validation gates (Spearman, F1-Score, NDCG@K, RMSE).
- **Script:** [cli/validate_graph.py](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/cli/validate_graph.py) or `saag-validate`
- **Arguments:**
  - `--structural`: Path to analysis metrics JSON.
  - `--simulated`: Path to simulation results JSON.
  - `--rmav`: Path to GNN prediction results JSON.
  - `--layer`: Layer to validate.
- **Example:**
  ```bash
  python cli/validate_graph.py --structural output/structural_metrics.json --simulated output/simulation_results.json --rmav output/predictions.json --layer system
  ```

---

### 5.7 Step 6: Interactive Dashboard Visualization
Generates an interactive, standalone HTML report containing system visualizations, metrics charts, and validation matrices.
- **Script:** [cli/visualize_graph.py](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/cli/visualize_graph.py) or `saag-visualize`
- **Arguments:**
  - `--output`: File path to save the generated HTML file (default: `dashboard.html`).
  - `--no-network`: Exclude force-directed network diagram (improves performance on large topologies).
- **Example:**
  ```bash
  python cli/visualize_graph.py --output output/dashboard.html --no-network
  ```

---

### 5.8 Full Pipeline Orchestration
Run the entire end-to-end pipeline from a single command:
- **Script:** [cli/run.py](file:///home/onuralpyigit/Workspace/SoftwareAsAGraph/cli/run.py) or `saag`
- **Arguments:**
  - `--all`: Runs all stages sequentially.
  - `--input`: Path to system topology JSON file.
  - `--clear`: Clean the database before running.
  - `--gnn-model`: Path to a pre-trained GNN model checkpoint.
- **Example:**
  ```bash
  python cli/run.py --all -i data/synthetic_medium.json --clear --gnn-model models/system_checkpoints
  ```

---

## 6. REST API User Guide

The REST API exposes the analytical and prediction pipeline as endpoints, allowing integrations with CI/CD gates and external tools.

### 6.1 Starting the API Server Locally
Activate the virtual environment and start the Uvicorn application server:
```bash
uvicorn api.main:app --reload --port 8000
```
The server will start on port `8000`. You can access the Swagger UI documentation at:
**[http://localhost:8000/docs](http://localhost:8000/docs)**

---

### 6.2 Primary REST API Endpoints

#### Graph Import `/api/v1/graph/import`
- **Method:** `POST`
- **Request Body (JSON):** Contains topology node and edge configurations.
- **Response:** Summary of imported components.

#### Execute Analysis `/api/v1/analysis/analyze`
- **Method:** `GET` or `POST`
- **Query Parameters:**
  - `layer` (string): `system` (default), `app`, `infra`, `mw`
  - `use_ahp` (boolean): `true` / `false`
- **Response:** JSON list of computed centrality metrics, anti-patterns, and RMAV scores per component.

#### GNN Criticality Prediction `/api/v1/prediction/predict`
- **Method:** `POST`
- **Request Body (JSON):**
  ```json
  {
    "layer": "system",
    "checkpoint_dir": "output/gnn_checkpoints"
  }
  ```
- **Response:** Ensembled GNN scores, edge attention mappings, and classification categories (e.g., `CRITICAL`, `HIGH`).

#### Failure Simulation `/api/v1/simulation/simulate`
- **Method:** `POST`
- **Request Body (JSON):**
  ```json
  {
    "layer": "system",
    "mode": "exhaustive"
  }
  ```
- **Response:** Node cascade reachability losses and service disruption metrics.

---

## 7. SMART Web Toolkit (Genieus) User Guide

Genieus is the web-based visual interface for interacting with the Software-as-a-Graph framework.

### 7.1 Launching the Web Application Natively
Navigate to the frontend directory and start the Next.js development server:
```bash
cd smart
npm run dev
```
Open your browser and navigate to:
**[http://localhost:7000](http://localhost:7000)**

---

### 7.2 Interface Navigation & Walkthrough

#### 1. Connection Panel
- Displays status indicators for the FastAPI backend and the Neo4j graph database.
- Allows users to select active namespaces, layers, and configuration presets.

#### 2. Topology Visualizer (2D & 3D Force-Directed Graphs)
- Displays structural dependencies and communications.
- Nodes are color-coded by their predicted criticality:
  - <span style="color:#d32f2f">●</span> **Red**: Critical Single Point of Failure (SPOF)
  - <span style="color:#f57c00">●</span> **Orange**: High Risk
  - <span style="color:#fbc02d">●</span> **Yellow**: Medium Risk
  - <span style="color:#388e3c">●</span> **Green**: Low / Minimal Risk
- Dragging, zooming, and clicking on a component opens a details panel containing code quality metrics (CQP, LCOM), degree centrality, and natural-language failure narratives.

#### 3. Analytical Dashboard
- **Anti-Pattern Audit**: Lists architectural smells (God Components, Failure Hubs, Exposure Paths, Cycles) ranked by severity.
- **RMAV Breakdown**: Charts detailing individual reliability, maintainability, availability, and vulnerability metrics for each node.

#### 4. Validation Panel
- Visualizes Spearman rank correlation scatter plots comparing predicted criticality against simulated ground truth.
- Displays F1-Scores and NDCG@K statistics across selected layers, highlighting passes or failures of standard compliance gates.

---

## 8. Troubleshooting & Diagnostics

Refer to the table below to diagnose and resolve issues encountered during installation and execution:

| Symptom / Error | Root Cause | Resolution |
|---|---|---|
| **Neo4j connection error:** `ServiceUnavailable: Cannot connect to ...` | Neo4j service is stopped or port is blocked by host firewall. | Check if Neo4j is running (`neo4j status` or docker log). Verify port bindings: port 7687 (Bolt) must be accessible. |
| **GDS procedure missing:** `Neo.ClientError.Procedure.ProcedureRegistrationFailed` | Graph Data Science library is not installed in Neo4j. | Verify `gds-*.jar` is in your Neo4j `plugins/` directory. Check config permissions. |
| **PyTorch Geometric error:** `ModuleNotFoundError: No module named 'torch_sparse'` | Missing binary extensions for PyTorch. | Re-run installation matching your exact PyTorch version: `pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cpu.html` |
| **Next.js server error:** `Next Router not mounted` | Port conflict or backend server unavailable on startup. | Confirm Uvicorn is running on port 8000. Check the frontend browser console to verify connection requests to the API. |
| **Validation fails:** `Spearman score < 0.70` | GNN training has not converged or incorrect model checkpoints were loaded. | Train model with more epochs or check validation dataset size. Validate with `--seeds` to confirm stability. |

---

### 8.1 Logs and Auditing
When running via Docker Compose, retrieve individual service logs:
```bash
# FastAPI Backend logs
docker compose exec genieus cat /tmp/backend.log

# Next.js Frontend logs
docker compose exec genieus cat /tmp/frontend.log

# Neo4j Database logs
docker compose exec genieus tail -n 100 /var/lib/neo4j/logs/neo4j.log
```

For native installations, standard log directories or console stdout contain execution details. The FastAPI server uses log records stored under `/tmp/backend.log` by default.
