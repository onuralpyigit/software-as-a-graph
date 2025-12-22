#!/bin/bash
#===============================================================================
#
#  Software-as-a-Graph: End-to-End Demo Pipeline
#  ==================================================
#
#  This script demonstrates the complete graph-based modeling and analysis
#  methodology for distributed publish-subscribe systems.
#
#  Pipeline Steps:
#    1. GENERATE  - Create realistic pub-sub system graph data
#    2. ANALYZE   - Calculate criticality scores using graph algorithms
#    3. SIMULATE  - Run failure simulations to get actual impact scores
#    4. VALIDATE  - Compare predictions with simulation (Spearman ≥0.7, F1 ≥0.9)
#    5. VISUALIZE - Generate interactive multi-layer visualizations
#
#  Usage:
#    ./run.sh                    # Run full demo with defaults
#    ./run.sh --scenario iot     # Specify scenario (iot, financial, ros2)
#    ./run.sh --scale large      # Specify scale (small, medium, large)
#    ./run.sh --quick            # Quick demo with small scale
#    ./run.sh --help             # Show help
#
#  Author: Ibrahim Onuralp Yigit
#  Research: Graph-Based Modeling and Analysis of Distributed Pub-Sub Systems
#  Publication: IEEE RASSE 2025
#
#===============================================================================

set -e  # Exit on error

#-------------------------------------------------------------------------------
# Configuration
#-------------------------------------------------------------------------------

# Default settings
SCENARIO="iot"
SCALE="medium"
OUTPUT_DIR="./output"
SEED=42

# Scoring weights (from research methodology)
ALPHA=0.25   # Betweenness centrality
BETA=0.30    # Articulation point
GAMMA=0.25   # Impact score
DELTA=0.10   # Degree centrality
EPSILON=0.10 # PageRank

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color
BOLD='\033[1m'

#-------------------------------------------------------------------------------
# Helper Functions
#-------------------------------------------------------------------------------

print_header() {
    echo ""
    echo -e "${PURPLE}${BOLD}======================================================================${NC}"
    echo -e "${PURPLE}${BOLD}  $1${NC}"
    echo -e "${PURPLE}${BOLD}======================================================================${NC}"
    echo ""
}

print_step() {
    echo ""
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}${BOLD}  STEP $1: $2${NC}"
    echo -e "${CYAN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}  ✓ $1${NC}"
}

print_error() {
    echo -e "${RED}  ✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}  ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}  ⚠ $1${NC}"
}

print_detail() {
    echo -e "${WHITE}    $1${NC}"
}

show_help() {
    cat << EOF
Software-as-a-Graph: End-to-End Demo Pipeline
==============================================

Usage: ./run.sh [OPTIONS]

Options:
    --scenario SCENARIO   System scenario: iot, financial, microservices, ros2
                          (default: iot)
    --scale SCALE         System scale: small, medium, large
                          (default: medium)
    --output-dir DIR      Output directory (default: ./output)
    --seed SEED           Random seed for reproducibility (default: 42)
    --quick               Quick demo with small scale
    --skip-generate       Skip graph generation (use existing file)
    --skip-validate       Skip validation step
    --input FILE          Use existing input file instead of generating
    --cascade             Enable cascade in simulation
    --antipatterns        Inject anti-patterns for testing
    --full-validation     Run full validation (sensitivity, bootstrap, CV)
    --verbose             Verbose output
    --help                Show this help message

Examples:
    # Full demo with IoT scenario
    ./run.sh --scenario iot --scale medium

    # Quick demo for testing
    ./run.sh --quick

    # Financial trading system, large scale
    ./run.sh --scenario financial --scale large

    # Use existing graph file
    ./run.sh --input my_system.json --skip-generate

    # Full demo with anti-patterns
    ./run.sh --scenario iot --antipatterns

Pipeline Steps:
    1. GENERATE  - Create realistic pub-sub system graph data
    2. ANALYZE   - Calculate criticality using graph algorithms  
    3. SIMULATE  - Run failure simulations for actual impact
    4. VALIDATE  - Compare predictions with simulation
    5. VISUALIZE - Generate interactive visualizations

Target Validation Metrics:
    - Spearman Correlation: ≥ 0.70
    - F1-Score: ≥ 0.90
    - Precision: ≥ 0.80
    - Recall: ≥ 0.80

EOF
}

#-------------------------------------------------------------------------------
# Parse Arguments
#-------------------------------------------------------------------------------

SKIP_GENERATE=false
SKIP_VALIDATE=false
INPUT_FILE=""
ENABLE_CASCADE=""
INJECT_ANTIPATTERNS=""
VERBOSE=""
FULL_VALIDATION=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --scenario)
            SCENARIO="$2"
            shift 2
            ;;
        --scale)
            SCALE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --quick)
            SCALE="small"
            shift
            ;;
        --skip-generate)
            SKIP_GENERATE=true
            shift
            ;;
        --skip-validate)
            SKIP_VALIDATE=true
            shift
            ;;
        --input)
            INPUT_FILE="$2"
            SKIP_GENERATE=true
            shift 2
            ;;
        --cascade)
            ENABLE_CASCADE="--cascade"
            shift
            ;;
        --antipatterns)
            INJECT_ANTIPATTERNS="--inject-antipatterns"
            shift
            ;;
        --full-validation)
            FULL_VALIDATION="--full-analysis"
            shift
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

#-------------------------------------------------------------------------------
# Setup
#-------------------------------------------------------------------------------

# Set input file path
if [ -z "$INPUT_FILE" ]; then
    INPUT_FILE="${OUTPUT_DIR}/system_${SCENARIO}_${SCALE}.json"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Timestamp
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

#-------------------------------------------------------------------------------
# Print Banner
#-------------------------------------------------------------------------------

clear 2>/dev/null || true
echo ""
echo -e "${PURPLE}${BOLD}"
cat << 'EOF'
  ╔═══════════════════════════════════════════════════════════════════════╗
  ║                                                                       ║
  ║   ███████╗ ██████╗ ███████╗████████╗██╗    ██╗ █████╗ ██████╗ ███████╗║
  ║   ██╔════╝██╔═══██╗██╔════╝╚══██╔══╝██║    ██║██╔══██╗██╔══██╗██╔════╝║
  ║   ███████╗██║   ██║█████╗     ██║   ██║ █╗ ██║███████║██████╔╝█████╗  ║
  ║   ╚════██║██║   ██║██╔══╝     ██║   ██║███╗██║██╔══██║██╔══██╗██╔══╝  ║
  ║   ███████║╚██████╔╝██║        ██║   ╚███╔███╔╝██║  ██║██║  ██║███████╗║
  ║   ╚══════╝ ╚═════╝ ╚═╝        ╚═╝    ╚══╝╚══╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝║
  ║                        AS A GRAPH                                     ║
  ║                                                                       ║
  ║   Graph-Based Modeling and Analysis of Distributed Pub-Sub Systems    ║
  ║                         Ibrahim Onuralp Yigit                         ║
  ║                                                                       ║
  ╚═══════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

echo -e "  ${WHITE}Timestamp:${NC}  $TIMESTAMP"
echo -e "  ${WHITE}Scenario:${NC}   $SCENARIO"
echo -e "  ${WHITE}Scale:${NC}      $SCALE"
echo -e "  ${WHITE}Output:${NC}     $OUTPUT_DIR"
echo -e "  ${WHITE}Seed:${NC}       $SEED"
echo ""

#-------------------------------------------------------------------------------
# Step 1: Generate Graph Data
#-------------------------------------------------------------------------------

if [ "$SKIP_GENERATE" = false ]; then
    print_step "1/5" "GENERATE - Create Pub-Sub System Graph"
    
    print_info "Generating $SCALE $SCENARIO system topology..."
    print_detail "This creates a realistic pub-sub graph with:"
    print_detail "  • Infrastructure nodes (servers, gateways)"
    print_detail "  • Message brokers (MQTT, Kafka, etc.)"
    print_detail "  • Topics/channels for message routing"
    print_detail "  • Applications (publishers and subscribers)"
    echo ""
    
    python generate_graph.py \
        --scenario "$SCENARIO" \
        --scale "$SCALE" \
        --output "$INPUT_FILE" \
        --seed "$SEED" \
        $INJECT_ANTIPATTERNS
    
    if [ $? -eq 0 ]; then
        print_success "Graph generated: $INPUT_FILE"
        
        # Show statistics
        python -c "
import json
with open('$INPUT_FILE') as f:
    data = json.load(f)
    stats = data.get('metadata', {}).get('statistics', {})
    print(f'    📊 Nodes: {stats.get(\"nodes\", \"N/A\")} | Edges: {stats.get(\"edges\", \"N/A\")}')
    print(f'    📱 Applications: {stats.get(\"applications\", \"N/A\")}')
    print(f'    📨 Topics: {stats.get(\"topics\", \"N/A\")}')
    print(f'    🔄 Brokers: {stats.get(\"brokers\", \"N/A\")}')
    print(f'    🖥️  Infrastructure: {stats.get(\"infrastructure\", \"N/A\")}')
" 2>/dev/null || true
    else
        print_error "Graph generation failed"
        exit 1
    fi

    # Import graph to Neo4j
    print_info "Importing graph into Neo4j database..."
    python import_graph.py \
        --input "$INPUT_FILE" \
        --clear \
        --analytics \
        $VERBOSE
else
    print_step "1/5" "GENERATE - Using Existing Graph"
    print_info "Input file: $INPUT_FILE"
    
    if [ ! -f "$INPUT_FILE" ]; then
        print_error "Input file not found: $INPUT_FILE"
        exit 1
    fi
    print_success "File exists and ready"
fi

#-------------------------------------------------------------------------------
# Step 2: Analyze Graph
#-------------------------------------------------------------------------------

print_step "2/5" "ANALYZE - Calculate Criticality Scores"

print_info "Running structural analysis using graph algorithms..."
print_detail "Quality attributes analyzed:"
print_detail "  • Reliability (SPOFs, redundancy, fault tolerance)"
print_detail "  • Maintainability (coupling, cohesion, modularity)"
print_detail "  • Availability (connectivity, reachability)"
echo ""
print_detail "Composite Criticality Score formula:"
print_detail "  C = α×BC + β×AP + γ×Impact + δ×DC + ε×PR"
print_detail "  Where: α=$ALPHA, β=$BETA, γ=$GAMMA, δ=$DELTA, ε=$EPSILON"
echo ""

python analyze_graph.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_DIR" \
    --format json html csv \
    --full \
    --antipatterns \
    --classify \
    $VERBOSE

if [ $? -eq 0 ]; then
    print_success "Analysis complete"
    print_detail "Results saved to: $OUTPUT_DIR/"
else
    print_error "Analysis failed"
    exit 1
fi

#-------------------------------------------------------------------------------
# Step 3: Simulate Failures
#-------------------------------------------------------------------------------

print_step "3/5" "SIMULATE - Run Failure Impact Analysis"

print_info "Simulating component failures to measure actual impact..."
print_detail "Simulation approach:"
print_detail "  • Test failure of each component individually"
print_detail "  • Measure reachability loss and service disruption"
print_detail "  • Calculate actual impact scores for validation"
echo ""

SIMULATION_OUTPUT="${OUTPUT_DIR}/simulation_results.json"

python simulate_graph.py \
    --input "$INPUT_FILE" \
    --campaign \
    --export-json "$SIMULATION_OUTPUT" \
    --seed "$SEED" \
    $ENABLE_CASCADE \
    $VERBOSE

if [ $? -eq 0 ]; then
    print_success "Simulation complete: $SIMULATION_OUTPUT"
    
    # Show top impacts
    python -c "
import json
with open('$SIMULATION_OUTPUT') as f:
    data = json.load(f)
    results = data.get('results', [])
    if results:
        sorted_results = sorted(results, key=lambda x: x.get('impact_score', 0), reverse=True)[:3]
        print('    Top 3 highest impact components:')
        for i, r in enumerate(sorted_results, 1):
            comp = r.get('failed_component', 'Unknown')
            impact = r.get('impact_score', 0)
            print(f'      {i}. {comp}: {impact:.3f}')
" 2>/dev/null || true
else
    print_error "Simulation failed"
    exit 1
fi

#-------------------------------------------------------------------------------
# Step 4: Validate Results
#-------------------------------------------------------------------------------

if [ "$SKIP_VALIDATE" = false ]; then
    print_step "4/5" "VALIDATE - Statistical Validation"
    
    print_info "Comparing predicted criticality with actual impact..."
    print_detail "Validation metrics and targets:"
    print_detail "  • Spearman Correlation ≥ 0.70 (rank correlation)"
    print_detail "  • F1-Score ≥ 0.90 (classification accuracy)"
    print_detail "  • Precision ≥ 0.80 (TP / (TP + FP))"
    print_detail "  • Recall ≥ 0.80 (TP / (TP + FN))"
    echo ""
    
    python validate_graph.py \
        --input "$INPUT_FILE" \
        --output "$OUTPUT_DIR" \
        --format json html csv \
        --alpha "$ALPHA" \
        --beta "$BETA" \
        --gamma "$GAMMA" \
        --delta "$DELTA" \
        --epsilon "$EPSILON" \
        --seed "$SEED" \
        $ENABLE_CASCADE \
        $FULL_VALIDATION \
        $VERBOSE
    
    if [ $? -eq 0 ]; then
        print_success "Validation complete"
        print_detail "Results saved to: ${OUTPUT_DIR}/"
    else
        print_warning "Validation completed (some targets may not be met)"
    fi
else
    print_step "4/5" "VALIDATE - Skipped"
    print_info "Use --skip-validate to enable validation"
fi

#-------------------------------------------------------------------------------
# Step 5: Visualize Results
#-------------------------------------------------------------------------------

print_step "5/5" "VISUALIZE - Generate Interactive Visualizations"

print_info "Creating multi-layer graph visualizations..."
print_detail "Visualization outputs:"
print_detail "  • Interactive network graph (vis.js)"
print_detail "  • Multi-layer architecture view"
print_detail "  • Criticality heatmap"
print_detail "  • Comprehensive dashboard"
echo ""

# Basic visualization
python visualize_graph.py \
    --input "$INPUT_FILE" \
    --output "${OUTPUT_DIR}/graph_visualization.html" \
    --run-analysis \
    $VERBOSE

print_success "Network graph: ${OUTPUT_DIR}/graph_visualization.html"

# Multi-layer visualization
python visualize_graph.py \
    --input "$INPUT_FILE" \
    --output "${OUTPUT_DIR}/multi_layer_view.html" \
    --multi-layer \
    --run-analysis \
    $VERBOSE

print_success "Multi-layer view: ${OUTPUT_DIR}/multi_layer_view.html"

# Criticality view
python visualize_graph.py \
    --input "$INPUT_FILE" \
    --output "${OUTPUT_DIR}/criticality_view.html" \
    --color-by criticality \
    --run-analysis \
    $VERBOSE

print_success "Criticality view: ${OUTPUT_DIR}/criticality_view.html"

# Dashboard
python visualize_graph.py \
    --input "$INPUT_FILE" \
    --output "${OUTPUT_DIR}/dashboard.html" \
    --dashboard \
    --run-analysis \
    $VERBOSE

print_success "Dashboard: ${OUTPUT_DIR}/dashboard.html"

#-------------------------------------------------------------------------------
# Summary
#-------------------------------------------------------------------------------

print_header "PIPELINE COMPLETE"

echo -e "${GREEN}${BOLD}  All 5 steps completed successfully!${NC}"
echo ""
echo -e "${WHITE}  Generated Files:${NC}"
echo ""
echo -e "  ${CYAN}Step 1 - Graph Data:${NC}"
echo -e "    📊 $INPUT_FILE"
echo ""
echo -e "  ${CYAN}Step 2 - Analysis:${NC}"
echo -e "    📈 ${OUTPUT_DIR}/analysis_results.json"
echo -e "    📄 ${OUTPUT_DIR}/analysis_report.html"
echo -e "    📋 ${OUTPUT_DIR}/criticality_scores.csv"
echo ""
echo -e "  ${CYAN}Step 3 - Simulation:${NC}"
echo -e "    💥 ${OUTPUT_DIR}/simulation_results.json"
echo ""
if [ "$SKIP_VALIDATE" = false ]; then
echo -e "  ${CYAN}Step 4 - Validation:${NC}"
echo -e "    ✅ ${OUTPUT_DIR}/validation_results.json"
echo -e "    📄 ${OUTPUT_DIR}/validation_report.html"
echo ""
fi
echo -e "  ${CYAN}Step 5 - Visualizations:${NC}"
echo -e "    🎨 ${OUTPUT_DIR}/graph_visualization.html"
echo -e "    🔀 ${OUTPUT_DIR}/multi_layer_view.html"
echo -e "    🎯 ${OUTPUT_DIR}/criticality_view.html"
echo -e "    📊 ${OUTPUT_DIR}/dashboard.html"
echo ""

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${WHITE}To view the dashboard, open in a browser:${NC}"
echo ""
echo -e "    ${GREEN}# macOS${NC}"
echo -e "    open ${OUTPUT_DIR}/dashboard.html"
echo ""
echo -e "    ${GREEN}# Linux${NC}"
echo -e "    xdg-open ${OUTPUT_DIR}/dashboard.html"
echo ""
echo -e "    ${GREEN}# Windows (WSL)${NC}"
echo -e "    explorer.exe ${OUTPUT_DIR}/dashboard.html"
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${GREEN}${BOLD}  Demo Complete! 🎉${NC}"
echo ""