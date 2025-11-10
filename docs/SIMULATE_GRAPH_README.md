# simulate_graph.py - User Guide

## Overview

The `simulate_graph.py` script provides comprehensive simulation capabilities for pub-sub systems, including:

- **Message Traffic Simulation**: Realistic DDS message flow (1000x speedup)
- **Failure Injection**: Single, multiple, and cascading failures
- **Impact Analysis**: Performance degradation and component isolation
- **Real-time Monitoring**: Track simulation progress
- **Baseline Comparison**: Measure failure impact vs. healthy system
- **Multiple Export Formats**: JSON and HTML reports

## Architecture Integration

```
Input (JSON)
     ↓
LightweightDDSSimulator
     ↓
Message Traffic Simulation
     ↓
FailureSimulator (optional)
     ↓
Impact Analysis
     ↓
Reports & Recommendations
```

## Features

### 1. Traffic Simulation
- Realistic message publishing/subscribing
- QoS-aware delivery (reliability, deadline)
- Broker routing with configurable delays
- Performance metrics (latency, throughput, delivery rate)

### 2. Failure Injection
- **Complete failures**: Component unavailable
- **Partial failures**: Degraded performance
- **Cascading failures**: Propagation based on dependencies
- **Scheduled failures**: Time-based injection
- **Recovery simulation**: Component restoration

### 3. Impact Analysis
- Failed and affected components
- Isolated applications and topics
- Message loss and delays
- Cascade depth and width
- Performance degradation metrics

### 4. Monitoring & Reporting
- Real-time progress tracking
- Baseline comparison
- Comprehensive recommendations
- JSON and HTML export

## Installation

### Prerequisites

```bash
# Required packages
pip install networkx asyncio

# Optional for enhanced features
pip install pandas matplotlib
```

### File Structure

```
your-project/
├── simulate_graph.py          # Main simulation script
├── src/
│   ├── simulation/
│   │   ├── lightweight_dds_simulator.py
│   │   └── enhanced_failure_simulator.py
│   └── core/
│       ├── graph_builder.py
│       └── graph_model.py
├── examples/
│   └── example_graphs/
│       ├── small_system.json
│       └── iot_system.json
└── output/                    # Simulation results
```

## Usage

### Basic Commands

#### 1. Simple Traffic Simulation

```bash
# Simulate 60 seconds of message traffic
python simulate_graph.py --input system.json --duration 60
```

**Output:**
```
Messages Sent: 15,234
Messages Delivered: 14,987
Delivery Rate: 98.38%
Throughput: 249.8 msg/s
Avg Latency: 12.45 ms
```

#### 2. Single Component Failure

```bash
# Fail application A1 at 30 seconds
python simulate_graph.py --input system.json --duration 60 \
    --fail-component A1 --fail-type complete --fail-time 30
```

**Output:**
```
Failed Components: 1
Affected Components: 8
Messages Lost: 423
Cascade Depth: 0
```

#### 3. Multiple Simultaneous Failures

```bash
# Fail multiple components at once
python simulate_graph.py --input system.json --duration 60 \
    --fail-component "A1,B1,N1" --fail-time 30 --enable-cascade
```

#### 4. Predefined Scenarios

```bash
# Use cascading broker failure scenario
python simulate_graph.py --input system.json --duration 120 \
    --scenario cascading-broker
```

### Advanced Features

#### Baseline Comparison

```bash
# Run baseline first, then failure simulation
python simulate_graph.py --input system.json --duration 60 \
    --baseline --fail-component B1 --fail-time 30
```

**Output:**
```
Baseline Comparison:
  Latency Increase: +45.2% (+5.63 ms)
  Throughput Decrease: -23.4%
  Delivery Rate Decrease: -4.2%
  Additional Dropped: 1,234
```

#### Real-time Monitoring

```bash
# Monitor every 5 seconds
python simulate_graph.py --input system.json --duration 120 \
    --scenario stress-test --monitor --monitor-interval 5
```

**Output:**
```
[Monitor] T+5s: Delivered: 1,247, Dropped: 0, Failures: 0
[Monitor] T+35s: Delivered: 8,234, Dropped: 234, Failures: 1
[Monitor] T+50s: Delivered: 11,456, Dropped: 567, Failures: 3
```

#### Export Reports

```bash
# Export JSON and HTML reports
python simulate_graph.py --input system.json --duration 60 \
    --scenario cascading-broker \
    --export-json --export-html --output results/
```

**Generated files:**
```
results/
├── simulation_20241109_143022.json
└── simulation_20241109_143022.html
```

### Configuration Options

#### Failure Parameters

```bash
# Configure cascading behavior
python simulate_graph.py --input system.json --duration 60 \
    --fail-component B1 \
    --enable-cascade \
    --cascade-threshold 0.6 \
    --cascade-probability 0.8
```

**Parameters:**
- `--cascade-threshold`: Load threshold for cascade (0-1, default: 0.7)
- `--cascade-probability`: Chance of propagation (0-1, default: 0.6)

#### Recovery Simulation

```bash
# Enable automatic recovery
python simulate_graph.py --input system.json --duration 90 \
    --fail-component A1 --fail-time 30 \
    --recovery
```

## Predefined Scenarios

### 1. Single Application (`single-app`)
Single application failure to test isolation.

```bash
python simulate_graph.py --input system.json --duration 60 \
    --scenario single-app
```

### 2. Cascading Broker (`cascading-broker`)
Broker failure with cascading effects.

```bash
python simulate_graph.py --input system.json --duration 120 \
    --scenario cascading-broker
```

### 3. Node Failure (`node-failure`)
Infrastructure node failure affecting multiple apps.

```bash
python simulate_graph.py --input system.json --duration 90 \
    --scenario node-failure
```

### 4. Multiple Simultaneous (`multiple-simultaneous`)
Multiple components fail at the same time.

```bash
python simulate_graph.py --input system.json --duration 60 \
    --scenario multiple-simultaneous
```

### 5. Gradual Degradation (`gradual-degradation`)
System degrades over time with partial failures.

```bash
python simulate_graph.py --input system.json --duration 90 \
    --scenario gradual-degradation
```

### 6. Recovery Test (`recovery-test`)
Failure followed by recovery to test resilience.

```bash
python simulate_graph.py --input system.json --duration 90 \
    --scenario recovery-test
```

### 7. Stress Test (`stress-test`)
Multiple cascading failures to test resilience limits.

```bash
python simulate_graph.py --input system.json --duration 120 \
    --scenario stress-test
```

## Output Examples

### Console Output

```
======================================================================
SIMULATION SUMMARY
======================================================================

⏱️  Execution:
   Simulation Duration: 60s
   Real Time: 0.58s
   Speedup: 103.4x

📊 Message Traffic:
   Messages Sent: 15,234
   Messages Delivered: 13,456
   Messages Dropped: 1,778
   Delivery Rate: 88.32%
   Throughput: 224.3 msg/s
   Avg Latency: 18.67 ms
   Deadline Misses: 234

💥 Failure Impact:
   Failed Components: 1
   Affected Components: 12
   Isolated Applications: 3
   Unavailable Topics: 5
   Cascade Depth: 2
   Cascade Width: 4

📈 Baseline Comparison:
   Latency Increase: +6.22 ms (+50.0%)
   Throughput Decrease: -15.2%
   Delivery Rate Decrease: -11.68%
   Additional Dropped: 1,778

💡 Recommendations:

   REPLICATION:
   • Replicate B1 - failure affects 12 applications

   REDUNDANCY:
   • Add redundancy - cascade depth of 2 detected

   RECOVERY:
   • Implement automatic recovery - 3 applications isolated

======================================================================
```

### JSON Output Structure

```json
{
  "duration_seconds": 60,
  "execution_time": 0.58,
  "results": {
    "global_stats": {
      "messages_sent": 15234,
      "messages_delivered": 13456,
      "messages_dropped": 1778,
      "delivery_rate": 0.8832,
      "avg_latency_ms": 18.67,
      "throughput_msg_per_sec": 224.3,
      "deadline_misses": 234
    }
  },
  "failures": {
    "scheduled": [...],
    "events": [...],
    "active_count": 1
  },
  "impact": {
    "failed_components": ["B1"],
    "affected_components": [...],
    "isolated_applications": [...],
    "cascade_depth": 2,
    "cascade_width": 4,
    "latency_increase_pct": 50.0,
    "throughput_decrease_pct": 15.2
  },
  "recommendations": {
    "replication": [...],
    "load_balancing": [...],
    "redundancy": [...],
    "recovery": [...]
  },
  "baseline_comparison": {...}
}
```

### HTML Report

The HTML report includes:
- Executive summary with key metrics
- Interactive metric cards (color-coded)
- Performance degradation analysis
- Failure impact visualization
- Prioritized recommendations
- Professional styling

## Use Cases

### 1. System Validation

Test if your system meets reliability requirements:

```bash
# Run without failures to get baseline
python simulate_graph.py --input prod_system.json --duration 300

# Test critical component failures
python simulate_graph.py --input prod_system.json --duration 300 \
    --baseline --fail-component MainBroker --export-html
```

### 2. Resilience Testing

Evaluate system resilience under various failure conditions:

```bash
# Test all scenarios
for scenario in single-app cascading-broker node-failure stress-test; do
    python simulate_graph.py --input system.json --duration 120 \
        --scenario $scenario --export-json --output "results/$scenario/"
done
```

### 3. Performance Benchmarking

Compare different system configurations:

```bash
# Test baseline system
python simulate_graph.py --input baseline.json --duration 120 \
    --export-json --output baseline/

# Test with redundancy
python simulate_graph.py --input redundant.json --duration 120 \
    --export-json --output redundant/

# Compare results
```

### 4. Capacity Planning

Determine system limits and bottlenecks:

```bash
# Gradually increase load while monitoring
python simulate_graph.py --input system.json --duration 180 \
    --monitor --monitor-interval 10 --export-html
```

### 5. Failure Impact Analysis

Understand the blast radius of component failures:

```bash
# Test each critical component
for component in B1 B2 B3 MainApp CriticalNode; do
    python simulate_graph.py --input system.json --duration 60 \
        --baseline --fail-component $component \
        --export-html --output "impact/$component/"
done
```

## Integration Examples

### With Analysis Pipeline

```bash
# 1. Generate graph
python generate_graph.py --scale large --scenario iot --output iot.json

# 2. Analyze criticality
python analyze_graph.py --input iot.json --simulate --export-json

# 3. Simulate failures on critical components
python simulate_graph.py --input iot.json --duration 120 \
    --scenario stress-test --export-html
```

### Automated Testing

```bash
#!/bin/bash
# test_resilience.sh

SYSTEM="production_system.json"
DURATION=300
OUTPUT="resilience_report"

# Run baseline
python simulate_graph.py --input $SYSTEM --duration $DURATION \
    --export-json --output "$OUTPUT/baseline/"

# Test each scenario
for scenario in single-app cascading-broker node-failure stress-test; do
    echo "Testing scenario: $scenario"
    python simulate_graph.py --input $SYSTEM --duration $DURATION \
        --scenario $scenario --baseline \
        --export-json --export-html \
        --output "$OUTPUT/$scenario/"
done

echo "Resilience testing complete. Reports in $OUTPUT/"
```

## Troubleshooting

### Issue: Simulation runs slowly

```bash
# Reduce duration or check system resources
python simulate_graph.py --input system.json --duration 30 --verbose
```

### Issue: No failures detected

```bash
# Verify component IDs exist in graph
python simulate_graph.py --input system.json --duration 60 \
    --fail-component VALID_COMPONENT_ID --verbose
```

### Issue: Cascade not propagating

```bash
# Lower cascade threshold
python simulate_graph.py --input system.json --duration 60 \
    --fail-component B1 --enable-cascade \
    --cascade-threshold 0.5 --cascade-probability 0.9
```

## Performance Notes

- **Speedup**: Typically 100-1000x real-time
- **Memory**: Scales linearly with components (~10MB per 1000 components)
- **CPU**: Single-threaded, benefits from high clock speed
- **Disk**: JSON exports ~1-10MB depending on detail level

## Best Practices

1. **Always run baseline first** for meaningful comparisons
2. **Use appropriate duration** (60s for quick tests, 300s for thorough)
3. **Export results** for later analysis and documentation
4. **Test realistic scenarios** based on actual system failures
5. **Monitor long-running simulations** to catch issues early
6. **Validate results** against real-world data when available

## Advanced Configuration

### Custom Failure Schedule

Create a custom scenario programmatically:

```python
# custom_scenario.py
import asyncio
from simulate_graph import run_failure_simulation

failures = [
    {'time': 30, 'component': 'B1', 'component_type': 'broker',
     'failure_type': 'partial', 'severity': 0.5, 'enable_cascade': False},
    {'time': 60, 'component': 'A1', 'component_type': 'application',
     'failure_type': 'complete', 'severity': 1.0, 'enable_cascade': True},
    {'time': 90, 'component': 'B1', 'component_type': 'recovery',
     'failure_type': 'complete', 'severity': 0.0, 'enable_cascade': False}
]

# Then use with --scenario custom
```

## Support

For issues or questions:
- Check `simulation.log` for detailed execution logs
- Use `--verbose` flag for debug output
- Review examples in `examples/` directory
- Consult architecture documentation

## Version

Current version: 1.0.0

Compatible with:
- LightweightDDSSimulator v2.0+
- FailureSimulator v2.0+
- Refactored architecture v2.0+
