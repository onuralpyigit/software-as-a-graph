# =============================================================================
# Static System Analyzer - Makefile
# =============================================================================
# 
# Usage:
#   make all PLATFORM_NAME=<platform>        - Run full pipeline
#   make clone PLATFORM_NAME=<platform>      - Run only clone stage
#   make analyze PLATFORM_NAME=<platform>    - Run only analyze stage  
#   make metrics PLATFORM_NAME=<platform>    - Run only code metrics scan (ck)
#   make structural PLATFORM_NAME=<platform> - Run only structural analysis
#   make aggregate PLATFORM_NAME=<platform>  - Run only aggregate stage
#   make stat PLATFORM_NAME=<platform>       - Run only stat stage
#   make clean                               - Clean output directories
#   make help                                - Show this help
#
# Examples:
#   make all PLATFORM_NAME=avionics
#   make all PLATFORM_NAME=avionics BUILD=1 VERBOSE=1 METRICS=1
#   make analyze PLATFORM_NAME=avionics BUILD=1
#   make analyze PLATFORM_NAME=avionics ANALYSIS_MODE=codeql
# =============================================================================

# Configuration
SHELL := /bin/bash
.DEFAULT_GOAL := help

# Directories
ROOT_DIR := $(shell pwd)
SRC_DIR := src
OUTPUT_DIR := output
CONFIG_DIR := config

# Docker
DOCKER_IMAGE := static-system-analyzer
DOCKER_TAG := latest

# Docker Options
# Mount current directory to /app
# Mount timezone from host
# Set custom hostname and prompt to avoid "I have no name!" issues
# Run as current user to avoid permission issues
DOCKER_OPTS := --rm \
	--hostname static-analyzer \
	-v "$(ROOT_DIR):/app" \
	-v /etc/localtime:/etc/localtime:ro \
	-u "$$(id -u):$$(id -g)" \
	-w /app

# Run command (non-interactive by default)
DOCKER_RUN := docker run $(DOCKER_OPTS) $(DOCKER_IMAGE):$(DOCKER_TAG)

# Python Execution Environments
# PYTHON_HOST: Runs directly on the host machine (for Clone stage to use host git/certs)
PYTHON_HOST := python3.9
# PYTHON_DOCKER: Runs inside Docker container (for Analysis/Stats to use container deps)
PYTHON_DOCKER := $(DOCKER_RUN) python3

# Required parameter check
ifndef PLATFORM_NAME
    PLATFORM_MSG := "Error: PLATFORM_NAME is required. Usage: make <target> PLATFORM_NAME=<name>"
endif

# Optional parameters
BUILD ?= 0
VERBOSE ?= 0
SKIP_ON_BUILD_FAILURE ?= 0
METRICS ?= 0
LOG_LEVEL ?= INFO
STAT_FORMAT ?= markdown
PLATFORMS ?=
OP ?= stat
DDS_MASK ?= 1
ANALYSIS_MODE ?= manual

# Build flags
ifeq ($(BUILD),1)
    BUILD_FLAG := --build
else
    BUILD_FLAG :=
endif

ifeq ($(VERBOSE),1)
    VERBOSE_FLAG := --verbose
else
    VERBOSE_FLAG :=
endif

ifeq ($(SKIP_ON_BUILD_FAILURE),1)
    SKIP_FLAG := --skip-on-build-failure
else
    SKIP_FLAG :=
endif

ifeq ($(DDS_MASK),1)
    DDS_MASK_FLAG := --dds-mask
else
    DDS_MASK_FLAG :=
endif

ifeq ($(METRICS),1)
    METRICS_FLAG := --enable-metrics
else
    METRICS_FLAG :=
endif

LOG_LEVEL_FLAG := --log-level $(LOG_LEVEL)
STAT_FORMAT_FLAG := --stat-format $(STAT_FORMAT)
ANALYSIS_MODE_FLAG := --analysis-mode $(ANALYSIS_MODE)

# Config files
REPO_LIST := $(CONFIG_DIR)/repo_names.txt
CSV_FILE := $(CONFIG_DIR)/platform_pkg_version.csv
ENV_FILE := $(CONFIG_DIR)/.env

# Output directories
CLONED_DIR := $(OUTPUT_DIR)/cloned
ANALYZED_DIR := $(OUTPUT_DIR)/analyzed
AGGREGATED_DIR := $(OUTPUT_DIR)/aggregated
STAT_DIR := $(OUTPUT_DIR)/stat
STRUCTURAL_DIR := $(OUTPUT_DIR)/structural
LOGS_DIR := $(OUTPUT_DIR)/logs

# =============================================================================
# TARGETS
# =============================================================================

.PHONY: all clone analyze metrics aggregate stat structural clean help check-platform setup-dirs

## Run full pipeline (clone -> analyze -> metrics -> structural -> aggregate -> stat)
## Note: Clone and Analyze run on HOST, others run in DOCKER
all: check-platform setup-dirs
	@echo "============================================================"
	@echo "Running full pipeline for platform: $(PLATFORM_NAME)"
	@echo "============================================================"
	@$(MAKE) clone PLATFORM_NAME=$(PLATFORM_NAME) VERBOSE=$(VERBOSE) LOG_LEVEL=$(LOG_LEVEL)
	@$(MAKE) analyze PLATFORM_NAME=$(PLATFORM_NAME) BUILD=$(BUILD) VERBOSE=$(VERBOSE) SKIP_ON_BUILD_FAILURE=$(SKIP_ON_BUILD_FAILURE) LOG_LEVEL=$(LOG_LEVEL) ANALYSIS_MODE=$(ANALYSIS_MODE)
ifeq ($(METRICS),1)
	@$(MAKE) metrics PLATFORM_NAME=$(PLATFORM_NAME) LOG_LEVEL=$(LOG_LEVEL) || echo "[WARN] Code metrics scan failed, continuing pipeline..."
endif
	@$(MAKE) structural PLATFORM_NAME=$(PLATFORM_NAME) VERBOSE=$(VERBOSE) LOG_LEVEL=$(LOG_LEVEL)
	@$(MAKE) aggregate PLATFORM_NAME=$(PLATFORM_NAME) VERBOSE=$(VERBOSE) LOG_LEVEL=$(LOG_LEVEL) DDS_MASK=$(DDS_MASK)
	@$(MAKE) stat PLATFORM_NAME=$(PLATFORM_NAME) VERBOSE=$(VERBOSE) LOG_LEVEL=$(LOG_LEVEL) STAT_FORMAT=$(STAT_FORMAT)

## Run only clone stage (Runs on HOST)
clone: check-platform setup-dirs
	@echo "============================================================"
	@echo "Running CLONE stage for platform: $(PLATFORM_NAME)"
	@echo "============================================================"
	$(PYTHON_HOST) $(SRC_DIR)/cmd/main.py \
		--platform $(PLATFORM_NAME) \
		--clone-only \
		--workspace $(ROOT_DIR) \
		--repo-list $(REPO_LIST) \
		--csv-file $(CSV_FILE) \
		--env-file $(ENV_FILE) \
		$(VERBOSE_FLAG) \
		$(LOG_LEVEL_FLAG)

## Run only analyze stage (Step 1: Python analysis on HOST)
## If METRICS=1, also runs code metrics scan in Docker after analysis
analyze: check-platform setup-dirs
	@echo "============================================================"
	@echo "Running ANALYZE stage for platform: $(PLATFORM_NAME)"
	@echo "============================================================"
	$(PYTHON_HOST) $(SRC_DIR)/cmd/main.py \
		--platform $(PLATFORM_NAME) \
		--analyze-only \
		--workspace $(ROOT_DIR) \
		$(BUILD_FLAG) \
		$(VERBOSE_FLAG) \
		$(SKIP_FLAG) \
		$(LOG_LEVEL_FLAG) \
		$(ANALYSIS_MODE_FLAG)
ifeq ($(METRICS),1)
	@echo "------------------------------------------------------------"
	@echo "METRICS=1 detected, running code metrics scan after analysis..."
	@echo "------------------------------------------------------------"
	@$(MAKE) metrics PLATFORM_NAME=$(PLATFORM_NAME) LOG_LEVEL=$(LOG_LEVEL) || echo "[WARN] Code metrics scan failed, continuing..."
endif

## Run only code metrics scan stage (Step 2: ck in DOCKER)
metrics: check-platform setup-dirs
	@echo "============================================================"
	@echo "Running CODE METRICS stage for platform: $(PLATFORM_NAME)"
	@echo "============================================================"
	$(PYTHON_DOCKER) $(SRC_DIR)/cmd/main.py \
		--platform $(PLATFORM_NAME) \
		--metrics-only \
		--workspace $(ROOT_DIR) \
		$(LOG_LEVEL_FLAG)

## Run only aggregate stage (Runs in DOCKER)
aggregate: check-platform setup-dirs
	@echo "============================================================"
	@echo "Running AGGREGATE stage for platform: $(PLATFORM_NAME)"
	@echo "============================================================"
	$(PYTHON_DOCKER) $(SRC_DIR)/cmd/main.py \
		--platform $(PLATFORM_NAME) \
		--aggregate-only \
		--workspace $(ROOT_DIR) \
		$(VERBOSE_FLAG) \
		$(LOG_LEVEL_FLAG) \
		$(DDS_MASK_FLAG)

## Run only stat stage (Runs in DOCKER)
stat: check-platform setup-dirs
	@echo "============================================================"
	@echo "Running STAT stage for platform: $(PLATFORM_NAME)"
	@echo "============================================================"
	$(PYTHON_DOCKER) $(SRC_DIR)/cmd/main.py \
		--platform $(PLATFORM_NAME) \
		--stat-only \
		--workspace $(ROOT_DIR) \
		$(VERBOSE_FLAG) \
		$(LOG_LEVEL_FLAG) \
		$(STAT_FORMAT_FLAG) \
		$(DDS_MASK_FLAG)

## Run only structural analysis stage (Runs in DOCKER)
structural: check-platform setup-dirs
	@echo "============================================================"
	@echo "Running STRUCTURAL ANALYSIS stage for platform: $(PLATFORM_NAME)"
	@echo "============================================================"
	$(PYTHON_DOCKER) $(SRC_DIR)/cmd/main.py \
		--platform $(PLATFORM_NAME) \
		--structural-only \
		--workspace $(ROOT_DIR) \
		$(VERBOSE_FLAG) \
		$(LOG_LEVEL_FLAG)

## Run operation on all platforms
## Usage: make all_platforms OP=<operation> PLATFORMS="platform1 platform2 ..."
all_platforms: setup-dirs
ifeq ($(PLATFORMS),)
	@echo "Error: PLATFORMS is required. Usage: make all_platforms OP=<operation> PLATFORMS=\"platform1 platform2\""
	@exit 1
endif
	@echo "============================================================"
	@echo "Running $(OP) on all platforms: $(PLATFORMS)"
	@echo "============================================================"
	@for platform in $(PLATFORMS); do \
		echo ""; \
		echo "------------------------------------------------------------"; \
		echo "Processing platform: $$platform"; \
		echo "------------------------------------------------------------"; \
		$(MAKE) $(OP) PLATFORM_NAME=$$platform BUILD=$(BUILD) VERBOSE=$(VERBOSE) SKIP_ON_BUILD_FAILURE=$(SKIP_ON_BUILD_FAILURE) LOG_LEVEL=$(LOG_LEVEL) STAT_FORMAT=$(STAT_FORMAT) || exit 1; \
	done
	@echo ""
	@echo "============================================================"
	@echo "Completed $(OP) on all platforms: $(PLATFORMS)"
	@echo "============================================================"

## Setup output directories
setup-dirs:
	@mkdir -p $(CLONED_DIR)
	@mkdir -p $(ANALYZED_DIR)
	@mkdir -p $(AGGREGATED_DIR)
	@mkdir -p $(STAT_DIR)
	@mkdir -p $(STRUCTURAL_DIR)
	@mkdir -p $(LOGS_DIR)
	@mkdir -p $(CONFIG_DIR)

## Check if PLATFORM_NAME is provided
check-platform:
ifdef PLATFORM_MSG
	@echo $(PLATFORM_MSG)
	@exit 1
endif

## Clean all output directories
clean:
	@echo "Cleaning output directories and cache..."
	@rm -rf $(OUTPUT_DIR)
	@find . -name __pycache__ -type d -exec rm -rf {} +
	@find . -name '*.pyc' -delete
	@find . -name '*.pyo' -delete
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache
	@rm -rf .ruff_cache
	@rm -rf .coverage htmlcov
	@rm -rf $(SRC_DIR)/tmp/*
	@rm -rf dist build *.egg-info
	@rm -f static-system-analyzer-offline.tar.gz
	@echo "Done."

## Clean only specific platform output
clean-platform: check-platform
	@echo "Cleaning output for platform: $(PLATFORM_NAME)"
	@rm -rf $(CLONED_DIR)/$(PLATFORM_NAME)
	@rm -rf $(ANALYZED_DIR)/$(PLATFORM_NAME)
	@rm -rf $(AGGREGATED_DIR)/$(PLATFORM_NAME)_relations.json
	@rm -rf $(STAT_DIR)/$(PLATFORM_NAME)_statistics.json
	@rm -rf $(STAT_DIR)/$(PLATFORM_NAME)_report.txt
	@rm -rf $(STRUCTURAL_DIR)/$(PLATFORM_NAME)
	@rm -f $(LOGS_DIR)/*_$(PLATFORM_NAME).log
	@echo "Done."

## Build Docker runtime image
docker-build:
	@echo "Building Docker runtime image: $(DOCKER_IMAGE):$(DOCKER_TAG)"
	@docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

## Run interactive shell in Docker container (with source mounted)
docker-shell:
	@echo "Starting interactive shell in Docker container..."
	@docker run -it $(DOCKER_OPTS) --entrypoint /bin/bash $(DOCKER_IMAGE):$(DOCKER_TAG)

## Create offline distribution package
dist: docker-build
	@echo "Creating offline distribution package..."
	@mkdir -p dist/static-system-analyzer
	@echo "Saving Docker image..."
	@docker save $(DOCKER_IMAGE):$(DOCKER_TAG) -o dist/static-system-analyzer/image.tar
	@echo "Copying project files..."
	@cp -r src config Makefile README.md Dockerfile requirements.txt dist/static-system-analyzer/
	@echo "Creating archive..."
	@cd dist && tar -czf ../static-system-analyzer-offline.tar.gz static-system-analyzer
	@rm -rf dist
	@echo "Offline package created: static-system-analyzer-offline.tar.gz"

## Clean Docker image and dist files
docker-clean:
	@docker rmi $(DOCKER_IMAGE):$(DOCKER_TAG)
	@rm -f static-system-analyzer-offline.tar.gz

## Show help
help:
	@echo ""
	@echo "Static System Analyzer - Makefile"
	@echo "================================="
	@echo ""
	@echo "Usage:"
	@echo "  make <target> PLATFORM_NAME=<name> [OPTIONS]"
	@echo ""
	@echo "Targets:"
	@echo "  all            Run full pipeline (clone -> analyze -> [metrics] -> aggregate -> stat -> structural)"
	@echo "  clone          Run only the clone stage (HOST)"
	@echo "  analyze        Run only the analyze stage (HOST)"
	@echo "  metrics        Run only the code metrics scan stage (DOCKER, ck)"
	@echo "  aggregate      Run only the aggregate stage (DOCKER)"
	@echo "  stat           Run only the stat stage (statistical analysis)"
	@echo "  structural     Run only the structural analysis stage"
	@echo "  all_platforms  Run operation on multiple platforms"
	@echo "  clean          Clean all output directories"
	@echo "  clean-platform Clean output for specific platform"
	@echo "  help         Show this help message"
	@echo ""
	@echo "Required:"
	@echo "  PLATFORM_NAME  Platform name (e.g., avionics, maritime)"
	@echo ""
	@echo "Options:"
	@echo "  BUILD=1                  Run gmake regenerate_code during analysis"
	@echo "  VERBOSE=1                Enable verbose output"
	@echo "  SKIP_ON_BUILD_FAILURE=1  Skip project if build fails"
	@echo "  METRICS=1                  Enable code metrics scan in full pipeline (default: disabled)"
	@echo "  LOG_LEVEL=<level>        Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)"
	@echo "  STAT_FORMAT=<format>     Stat output format: markdown, pdf (default: markdown)"
	@echo "  PLATFORMS=<list>         Space-separated list of platforms for all_platforms target"
	@echo "  OP=<operation>           Operation for all_platforms: all, clone, analyze, aggregate, stat (default: stat)"
	@echo ""
	@echo "Examples:"
	@echo "  make all PLATFORM_NAME=avionics"
	@echo "  make all PLATFORM_NAME=avionics BUILD=1 VERBOSE=1"
	@echo "  make all PLATFORM_NAME=avionics METRICS=1"
	@echo "  make all PLATFORM_NAME=avionics LOG_LEVEL=DEBUG"
	@echo "  make clone PLATFORM_NAME=avionics"
	@echo "  make analyze PLATFORM_NAME=avionics BUILD=1"
	@echo "  make metrics PLATFORM_NAME=avionics"
	@echo "  make aggregate PLATFORM_NAME=avionics"
	@echo "  make stat PLATFORM_NAME=avionics"
	@echo "  make stat PLATFORM_NAME=avionics STAT_FORMAT=pdf"
	@echo "  make structural PLATFORM_NAME=avionics"
	@echo "  make all_platforms OP=stat PLATFORMS=\"avionics maritime ground\""
	@echo "  make all_platforms OP=all PLATFORMS=\"avionics maritime\" BUILD=1"
	@echo "  make clean-platform PLATFORM_NAME=avionics"
	@echo ""
	@echo "Directory Structure:"
	@echo "  output/"
	@echo "    ├── cloned/<platform>/      - Cloned repositories"
	@echo "    ├── analyzed/<platform>/    - Analysis CSV files"
	@echo "    ├── aggregated/             - JSON output files (includes structural_analysis)"
	@echo "    ├── stat/                   - Statistical analysis results"
	@echo "    ├── structural/             - Structural Markdown reports"
	@echo "    └── logs/                   - Log files"
	@echo ""
