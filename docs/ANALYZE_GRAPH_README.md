# analyze_graph.py - User Guide

## Overview

The `analyze_graph.py` script is the command-line interface for comprehensive pub-sub system analysis using the **modular architecture**.

### Architecture Flow

```
Input Source (JSON or Neo4j)
         ↓
    GraphBuilder
         ↓
    GraphModel
         ↓
   GraphExporter
         ↓
   NetworkX Graph
         ↓
AnalysisOrchestrator
    ├── StructuralAnalyzer
    ├── QoSAnalyzer
    ├── CompositeCriticalityScorer
    └── ReachabilityAnalyzer
         ↓
   Analysis Results
         ↓
Output (JSON/CSV/HTML)
```

## Features

✅ **Multiple Input Sources**
- JSON configuration files
- Neo4j database (live connection)

✅ **Comprehensive Analysis**
- Multi-metric centrality analysis
- QoS-aware criticality scoring
- Failure simulation and impact assessment
- Multi-layer dependency analysis

✅ **Flexible Configuration**
- Configurable criticality weights (α, β, γ)
- Enable/disable QoS analysis
- Enable/disable failure simulation

✅ **Multiple Output Formats**
- JSON (complete results + separate criticality scores)
- CSV (criticality scores as table)
- HTML (formatted report with styling)

✅ **Professional Logging**
- INFO and DEBUG levels
- Console and file logging
- Detailed error messages

## Installation

### Prerequisites

```bash
# Required packages
pip install networkx

# Optional for Neo4j
pip install neo4j

# Optional for CSV export
pip install pandas
```

### Quick Setup

```bash
# Clone or download the repository
cd your-project-directory

# Make script executable (Unix/Mac)
chmod +x analyze_graph.py

# Verify installation
python analyze_graph.py --help
```

## Usage

### Basic Usage

#### Analyze from JSON
```bash
python analyze_graph.py --input system.json
```

#### Analyze from Neo4j
```bash
python analyze_graph.py --neo4j \
    --uri bolt://localhost:7687 \
    --user neo4j \
    --password password
```

### Advanced Options

#### With Failure Simulation
```bash
python analyze_graph.py --input system.json --simulate
```

#### Custom Criticality Weights
```bash
python analyze_graph.py --input system.json \
    --alpha 0.5 \
    --beta 0.3 \
    --gamma 0.2
```

The weights determine how different metrics contribute to criticality:
- **α (alpha)**: Weight for betweenness centrality (default: 0.4)
- **β (beta)**: Weight for articulation points (default: 0.3)
- **γ (gamma)**: Weight for impact score (default: 0.3)

#### Multiple Export Formats
```bash
python analyze_graph.py --input system.json \
    --export-json \
    --export-csv \
    --export-html
```

#### Disable QoS Analysis
```bash
python analyze_graph.py --input system.json --no-qos
```

#### Verbose Logging
```bash
python analyze_graph.py --input system.json --verbose
```

#### Custom Output Directory
```bash
python analyze_graph.py --input system.json --output results/
```

## Command-Line Options

### Input Source (Required - Mutually Exclusive)

| Option | Description |
|--------|-------------|
| `--input`, `-i` | Input JSON file path |
| `--neo4j` | Load from Neo4j database |

### Neo4j Connection (When using --neo4j)

| Option | Default | Description |
|--------|---------|-------------|
| `--uri` | bolt://localhost:7687 | Neo4j connection URI |
| `--user` | neo4j | Neo4j username |
| `--password` | password | Neo4j password |
| `--database` | neo4j | Neo4j database name |

### Analysis Options

| Option | Default | Description |
|--------|---------|-------------|
| `--simulate` | False | Enable failure simulation |
| `--no-qos` | False | Disable QoS analysis |
| `--alpha` | 0.4 | Betweenness centrality weight |
| `--beta` | 0.3 | Articulation point weight |
| `--gamma` | 0.3 | Impact score weight |

### Output Options

| Option | Default | Description |
|--------|---------|-------------|
| `--output`, `-o` | output | Output directory |
| `--export-json` | True | Export as JSON |
| `--export-csv` | False | Export as CSV |
| `--export-html` | False | Export as HTML |
| `--no-summary` | False | Skip console summary |

### Logging Options

| Option | Description |
|--------|-------------|
| `--verbose`, `-v` | Enable DEBUG logging |

## Input File Format

### JSON Format

The JSON input should follow this structure:

```json
{
  "applications": [
    {
      "name": "app1",
      "type": "Publisher",
      "criticality": 0.8
    }
  ],
  "topics": [
    {
      "name": "topic1",
      "qos": {
        "durability": "PERSISTENT",
        "reliability": "RELIABLE"
      }
    }
  ],
  "brokers": [
    {
      "name": "broker1",
      "max_throughput": 10000
    }
  ],
  "nodes": [
    {
      "name": "node1",
      "zone": "us-east-1a"
    }
  ],
  "edges": {
    "publishes": [...],
    "subscribes": [...],
    "routes": [...],
    "runs_on": [...]
  }
}
```

See `examples/` directory for complete examples.

## Output Files

### JSON Output

#### analysis_results.json
Complete analysis results including:
- Graph summary
- Criticality scores for all components
- Structural analysis
- QoS analysis
- Layer analysis
- Failure simulation results
- Recommendations
- Execution time breakdown

#### criticality_scores.json
Separate file with just criticality scores for easy access:
```json
{
  "BrokerA": {
    "composite_score": 0.892,
    "level": "CRITICAL",
    "betweenness": 0.456,
    "is_articulation_point": true,
    "impact_score": 0.789
  }
}
```

### CSV Output (--export-csv)

**criticality_scores.csv**: Tabular format for Excel/spreadsheet import
```csv
component,composite_score,level,betweenness,is_articulation_point,impact_score
BrokerA,0.892,CRITICAL,0.456,True,0.789
ApplicationX,0.834,CRITICAL,0.378,True,0.712
```

### HTML Output (--export-html)

**analysis_report.html**: Formatted HTML report with:
- System overview metrics
- Critical components table (color-coded)
- QoS analysis summary
- Failure simulation results
- Recommendations (priority-ordered)
- Execution time breakdown
- Professional styling with responsive design

## Console Output

The script prints a formatted summary:

```
======================================================================
COMPLEX SYSTEM ANALYSIS
======================================================================

2024-11-05 10:30:00 - INFO - Loading graph from JSON: system.json
2024-11-05 10:30:01 - INFO - ✓ Loaded graph: 150 nodes, 320 edges
2024-11-05 10:30:01 - INFO - Graph loaded in 0.82s

2024-11-05 10:30:01 - INFO - Initializing AnalysisOrchestrator...
2024-11-05 10:30:01 - INFO - Running comprehensive analysis...
2024-11-05 10:30:09 - INFO - ✓ Analysis complete in 8.35s

======================================================================
ANALYSIS SUMMARY
======================================================================

📊 System Overview:
   Nodes:      150
   Edges:      320
   Density:    0.0145
   Connected:  True

⚠️  Critical Components:
   Total Analyzed:        150
   Critical (>0.7):       12
   High (0.5-0.7):        28
   Articulation Points:   5

   Top 5 Most Critical:
   1. BrokerA (Score: 0.892, Type: Broker)
   2. ApplicationX (Score: 0.834, Type: Application)
   3. TopicY (Score: 0.756, Type: Topic)
   4. NodeZ (Score: 0.723, Type: Node)
   5. BrokerB (Score: 0.698, Type: Broker)

🎯 QoS Analysis:
   High Priority Topics:       15
   High Priority Applications: 23
   Compatibility Issues:       3

💥 Failure Simulation:
   Resilience Score:          0.723
   Avg Components Affected:   12.3

💡 Top 3 Recommendations:
   1. [HIGH] Single Point of Failure
      Component: BrokerA
      Action: Add redundant broker or reroute critical topics
   2. [HIGH] QoS Compatibility
      Component: TopicX
      Action: Align QoS policies between publishers and subscribers
   3. [MEDIUM] Load Distribution
      Component: NodeY
      Action: Redistribute applications across multiple nodes

⏱️  Execution Time: 8.35s

======================================================================
```

## Integration with Architecture

### Component Usage

The script uses these modules:

```python
from src.core.graph_builder import GraphBuilder
from src.core.graph_exporter import GraphExporter
from src.core.graph_model import GraphModel
from src.orchestration.analysis_orchestrator import AnalysisOrchestrator
```

### Data Flow

1. **Loading Phase**
   ```python
   builder = GraphBuilder()
   model = builder.build_from_json(filepath)
   # or
   model = builder.build_from_neo4j(uri, auth, database)
   ```

2. **Conversion Phase**
   ```python
   exporter = GraphExporter()
   graph = exporter.export_to_networkx(model)
   ```

3. **Analysis Phase**
   ```python
   orchestrator = AnalysisOrchestrator(
       output_dir="output/",
       enable_qos=True,
       criticality_weights={'alpha': 0.4, 'beta': 0.3, 'gamma': 0.3}
   )
   
   results = orchestrator.analyze_graph(
       graph=graph,
       graph_model=model,
       enable_simulation=True
   )
   ```

4. **Output Phase**
   - JSON export
   - CSV export (optional)
   - HTML export (optional)
   - Console summary

## Examples

### Example 1: Quick Analysis

```bash
# Simplest usage - analyze a JSON file
python analyze_graph.py --input examples/small_system.json
```

### Example 2: Complete Analysis with All Outputs

```bash
# Full analysis with all export formats
python analyze_graph.py \
    --input data/production_system.json \
    --simulate \
    --export-json \
    --export-csv \
    --export-html \
    --output results/production_analysis/
```

### Example 3: Neo4j Live Analysis

```bash
# Analyze live Neo4j database
python analyze_graph.py \
    --neo4j \
    --uri bolt://production-db.example.com:7687 \
    --user analyst \
    --password secure_password \
    --database prod_graph \
    --simulate \
    --export-html
```

### Example 4: Custom Weights for Specific Use Case

```bash
# Emphasize betweenness centrality (flow criticality)
python analyze_graph.py \
    --input system.json \
    --alpha 0.6 \
    --beta 0.2 \
    --gamma 0.2 \
    --simulate
```

### Example 5: Fast Analysis (No Simulation, No QoS)

```bash
# Quick structural analysis only
python analyze_graph.py \
    --input system.json \
    --no-qos \
    --output quick_analysis/
```

## Troubleshooting

### Common Issues

#### 1. Graph is empty
```
ERROR - Graph is empty! Cannot perform analysis.
```
**Solution**: Check that your input file has valid nodes and edges.

#### 2. Neo4j connection failed
```
ERROR - Failed to connect to Neo4j
```
**Solutions**:
- Verify Neo4j is running: `docker ps` or check service status
- Check URI, username, and password
- Verify network connectivity
- Ensure neo4j package is installed: `pip install neo4j`

#### 3. Import errors
```
ERROR - Missing dependency: No module named 'neo4j'
```
**Solution**: Install missing packages
```bash
pip install neo4j networkx pandas
```

#### 4. Permission denied on output directory
```
ERROR - Permission denied: output/
```
**Solution**: Check write permissions or specify different output directory
```bash
python analyze_graph.py --input system.json --output ~/analysis_results/
```

### Debug Mode

For detailed debugging information:

```bash
python analyze_graph.py --input system.json --verbose
```

This enables:
- DEBUG level logging
- Full stack traces on errors
- Detailed component interactions
- Analysis progress indicators

Check `analysis.log` for complete logs.

## Performance Considerations

### Analysis Time

Typical execution times by graph size:

| Graph Size | Nodes | Edges | Time (no simulation) | Time (with simulation) |
|------------|-------|-------|---------------------|----------------------|
| Small      | <50   | <100  | 1-2s               | 2-4s                |
| Medium     | 50-200| 100-500| 3-8s              | 8-15s               |
| Large      | 200-1000| 500-2000| 10-30s          | 30-60s              |
| Very Large | >1000 | >2000 | 30-120s            | 2-5min              |

### Memory Usage

- Small graphs: <100 MB
- Medium graphs: 100-500 MB
- Large graphs: 500 MB - 2 GB
- Very large graphs: >2 GB

### Optimization Tips

1. **Disable simulation for faster analysis**
   ```bash
   python analyze_graph.py --input large_system.json
   # (simulation is off by default)
   ```

2. **Disable QoS analysis if not needed**
   ```bash
   python analyze_graph.py --input system.json --no-qos
   ```

3. **Skip CSV/HTML export for faster execution**
   ```bash
   python analyze_graph.py --input system.json
   # Only JSON export (default)
   ```

## Advanced Usage

### Batch Processing

```bash
#!/bin/bash
# Analyze multiple systems

for file in systems/*.json; do
    name=$(basename "$file" .json)
    python analyze_graph.py \
        --input "$file" \
        --output "results/$name/" \
        --export-html
done
```

### Integration with CI/CD

```yaml
# .github/workflows/analysis.yml
name: System Analysis

on: [push]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run analysis
        run: |
          python analyze_graph.py \
            --input system.json \
            --export-html \
            --output analysis_results/
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: analysis-report
          path: analysis_results/
```

### Python API Usage

You can also use the components programmatically:

```python
from src.core.graph_builder import GraphBuilder
from src.core.graph_exporter import GraphExporter
from src.orchestration.analysis_orchestrator import AnalysisOrchestrator

# Build graph
builder = GraphBuilder()
model = builder.build_from_json("system.json")

# Convert to NetworkX
exporter = GraphExporter()
graph = exporter.export_to_networkx(model)

# Analyze
orchestrator = AnalysisOrchestrator(
    output_dir="output/",
    enable_qos=True
)

results = orchestrator.analyze_graph(
    graph=graph,
    graph_model=model,
    enable_simulation=True
)

# Access results
print(f"Total nodes: {results['graph_summary']['total_nodes']}")
print(f"Critical components: {len(results['criticality_scores']['critical_components'])}")
```

## Contributing

When modifying the script:

1. Maintain backward compatibility with existing JSON formats
2. Add new CLI options as optional (with sensible defaults)
3. Update this documentation
4. Add examples for new features
5. Test with both JSON and Neo4j inputs

## Support

For issues or questions:
- Check the examples/ directory for sample usage
- Review analysis.log for detailed error information
- Consult the architecture documentation
- Open an issue with reproducible steps

## Version

Current version: 2.0.0

**Changes from 1.x**:
- Complete rewrite using modular architecture
- Added Neo4j support
- Added multiple export formats
- Configurable criticality weights
- Better error handling and logging
- Improved performance
