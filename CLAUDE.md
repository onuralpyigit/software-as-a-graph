# CLAUDE.md Development Guide

**Quick-reference guide for LLM coding guidelines, build commands, test suites, and project conventions.**

[README](README.md) | [Architecture](ARCHITECTURE.md)

---

## Table of Contents

1. [LLM Coding Guidelines](#llm-coding-guidelines)
2. [Command Line Reference](#command-line-reference)
3. [Project Conventions & Standards](#project-conventions--standards)

---

## LLM Coding Guidelines

These guidelines represent best-practice rules for AI coding assistants working on the repository. They prioritize caution, correctness, and simplicity.

### Think Before Coding
- **State assumptions explicitly** — if any requirements are ambiguous, clarify them with the user before writing code.
- **Surface tradeoffs** — present alternative implementations or designs rather than making a choice silently.
- **Push back when warranted** — if a simpler solution or a pre-existing function can achieve the task, explain it.
- **Stop when confused** — identify what is unclear, stop execution, and ask the user for clarification.

### Simplicity First
- **Avoid speculative features** — implement the minimum code required to solve the task. Do not add abstractions for single-use logic.
- **Keep it concise** — avoid adding unrequested configurability or overcomplicated error handling for impossible scenarios.
- **Review for complexity** — if a file or function becomes unnecessarily verbose, refactor it to its simplest equivalent form.

### Surgical Changes
- **Minimize diff blast-radius** — touch only the lines of code directly related to the user request.
- **Avoid unrelated styling or cleanup** — do not "improve" adjacent formatting, docstrings, or logic that is not broken.
- **Observe style parity** — match the established conventions, naming patterns, and file structure of the code you are editing.
- **Prune orphans** — clean up unused imports, variables, and functions introduced by your changes. Do not delete pre-existing dead code unless explicitly requested.

### Goal-Driven Execution
- **Establish success criteria** — transform requirements into testable validation steps (e.g. reproducing a bug in a test first).
- **Run verification loops** — execute tests before, during, and after implementing changes to verify that existing features do not break.

---

## Command Line Reference

### Docker Stack Commands
Run the database, REST API, and web interface as a unified local container stack:

```bash
# Build and run the entire stack
docker compose up --build

# Stop the container services
docker compose down
```

The stack exposes:
- **SMART Dashboard** — `http://localhost:7000`
- **FastAPI API Server** — `http://localhost:8000/docs`
- **Neo4j Browser** — `http://localhost:7474` (default credentials: `neo4j` / `password`)

### Backend Local Commands
Commands must be run from the repository root:

```bash
# Install local package with all extras
pip install -e ".[all]"

# Start local FastAPI development server (auto-reloads)
uvicorn api.main:app --reload --port 8000

# Execute full pipeline script on a single layer
python cli/run.py --all --layer system

# Execute full pipeline scripts on all scenarios in batch
bash cli/run_scenarios.sh
```

For individual pipeline scripts, see the [cli/](cli/) directory.

### Frontend Local Commands
Commands must be run from the frontend directory:

```bash
cd smart

# Install frontend dependencies
npm install

# Start Next.js development server
npm run dev

# Regenerate API client types from backend OpenAPI JSON
npm run generate-client
```

### Pytest Testing Commands
Tests should be executed from the project root and do not require a live Neo4j instance:

```bash
# Run the complete test suite
pytest

# Halt testing on the first failure
pytest -x

# Execute a single test file
pytest tests/test_analysis_service.py

# Run tests matching a specific pattern
pytest -k "reliability"
```

---

## Project Conventions & Standards

### Architecture & Design Patterns
- **Hexagonal Architecture** — The core SDK in [saag/](saag/) isolates business logic from databases and clients. Use Cases in [saag/usecases/](saag/usecases/) act as the boundary interface between presentation layers (FastAPI, CLI) and core domain services.
- **Repository Pattern** — Data operations are mapped to the `IGraphRepository` interface. Production runs use [Neo4jRepository](saag/infrastructure/neo4j_repo.py), whereas unit tests use the mock [MemoryRepository](saag/infrastructure/memory_repo.py). Core services must never depend directly on Neo4j classes.
- **REST Presenters** — API routers in [api/routers/](api/routers/) must delegate serialization formatting to presenters in [api/presenters/](api/presenters/) to isolate response schemas from domain models.

### Layer Projections & Dependency Derivation
- **Analysis Layers** — The codebase uses four canonical layers (`app`, `infra`, `mw`, `system`) defined in [saag/core/layers.py](saag/core/layers.py). The `app` layer includes both `Application` and `Library` nodes.
- **Logical Dependencies** — Physical pub-sub linkages are transformed into `DEPENDS_ON` edges using the six rules implemented in `Neo4jRepository`. In the dependency graph, edge direction points from the *dependent* to its *dependency*.
- **Simulation Substrate** — Failure simulations in [saag/simulation/](saag/simulation/) operate strictly on raw structural edges ($G_{\text{structural}}$), not derived dependency edges ($G_{\text{analysis}}$).

### Documentation Hyperlinking Standards
- **Relative Portability** — When referencing files, folders, or code elements inside documentation, always use standard relative Markdown links. Avoid using absolute `file:///` URLs to maintain portability across different workspaces and Git hosting systems.
- **Line Ranges** — Link directly to symbol line boundaries (e.g. `[Pipeline](saag/pipeline.py#L12)`) when referencing code elements to assist LLM navigation.
