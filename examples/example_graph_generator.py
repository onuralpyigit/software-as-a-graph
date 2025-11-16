#!/usr/bin/env python3
"""
Quick Start Examples for Enhanced Graph Generator

This script demonstrates the key features and improvements of the
enhanced graph generation system through practical examples.

Run this to see the improvements in action!

Usage:
    python example_graph_generator.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / '..'))

from src.core.graph_generator import GraphGenerator, GraphConfig

def print_section(title):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70 + "\n")


def example_1_basic():
    """Example 1: Basic graph generation"""
    print_section("Example 1: Basic Small System")
    
    config = GraphConfig(
        scale='small',
        scenario='generic',
        num_nodes=5,
        num_applications=10,
        num_topics=8,
        num_brokers=2,
        edge_density=0.3,
        antipatterns=[],
        seed=42
    )
    
    generator = GraphGenerator(config)
    graph = generator.generate()
    
    print(f"✓ Generated {len(graph['nodes'])} nodes")
    print(f"✓ Generated {len(graph['applications'])} applications")
    print(f"✓ Generated {len(graph['topics'])} topics")
    print(f"✓ Generated {len(graph['brokers'])} brokers")
    
    # Show some interesting details
    print("\nSample Application:")
    app = graph['applications'][0]
    print(f"  ID: {app['id']}")
    print(f"  Name: {app['name']}")
    print(f"  Type: {app['type']}")
    
    print("\nSample Topic:")
    topic = graph['topics'][0]
    print(f"  ID: {topic['id']}")
    print(f"  Name: {topic['name']}")
    print(f"  QoS Durability: {topic['qos']['durability']}")
    print(f"  QoS Reliability: {topic['qos']['reliability']}")


def example_2_realistic_iot():
    """Example 2: Realistic IoT system with semantic matching"""
    print_section("Example 2: Realistic IoT System")
    
    config = GraphConfig(
        scale='small',
        scenario='iot',
        num_nodes=5,
        num_applications=10,
        num_topics=12,
        num_brokers=2,
        edge_density=0.3,
        antipatterns=[],
        seed=42
    )
    
    generator = GraphGenerator(config)
    graph = generator.generate()
    
    print("✓ Generated realistic IoT system")
    
    # Show realistic patterns
    print("\nIoT Application Types:")
    app_types = {}
    for app in graph['applications']:
        base = app['name'].split('_')[0]
        app_types[base] = app_types.get(base, 0) + 1
    for app_type, count in sorted(app_types.items()):
        print(f"  {app_type}: {count}")
    
    print("\nIoT Topic Patterns:")
    patterns = {}
    for topic in graph['topics']:
        pattern = topic['name'].split('/')[0] if '/' in topic['name'] else topic['name'].split('_')[0]
        patterns[pattern] = patterns.get(pattern, 0) + 1
    for pattern, count in sorted(patterns.items()):
        print(f"  {pattern}: {count}")
    
    # Show semantic matching
    print("\nSemantic Matching Examples:")
    for app in graph['applications'][:3]:
        app_name = app['name']
        app_topics = [
            t['name'] for rel in graph['relationships']['publishes_to']
            if rel['from'] == app['id']
            for t in graph['topics'] if t['id'] == rel['to']
        ]
        if app_topics:
            print(f"  {app_name} publishes to:")
            for topic in app_topics[:2]:
                print(f"    → {topic}")


def example_3_financial_ha():
    """Example 3: Financial system with HA and strict QoS"""
    print_section("Example 3: Financial Trading System (HA)")
    
    config = GraphConfig(
        scale='medium',
        scenario='financial',
        num_nodes=15,
        num_applications=50,
        num_topics=25,
        num_brokers=3,
        edge_density=0.3,
        antipatterns=[],
        seed=42
    )
    
    generator = GraphGenerator(config)
    graph = generator.generate()
    
    print("✓ Generated financial trading system with HA")
    
    # Show QoS strictness
    print("\nFinancial QoS Policies:")
    qos_stats = {'PERSISTENT': 0, 'RELIABLE': 0, 'URGENT': 0}
    for topic in graph['topics']:
        if topic['qos']['durability'] == 'PERSISTENT':
            qos_stats['PERSISTENT'] += 1
        if topic['qos']['reliability'] == 'RELIABLE':
            qos_stats['RELIABLE'] += 1
        if topic['qos']['transport_priority'] == 'URGENT':
            qos_stats['URGENT'] += 1
    
    print(f"  PERSISTENT durability: {qos_stats['PERSISTENT']}/{len(graph['topics'])}")
    print(f"  RELIABLE: {qos_stats['RELIABLE']}/{len(graph['topics'])}")
    print(f"  URGENT priority: {qos_stats['URGENT']}/{len(graph['topics'])}")
    
    # Show ultra-low latency
    low_latency = [t for t in graph['topics'] 
                   if t['qos'].get('deadline_ms') and t['qos']['deadline_ms'] <= 10]
    print(f"  Ultra-low latency (<10ms): {len(low_latency)} topics")


def example_4_antipatterns():
    """Example 4: System with multiple anti-patterns"""
    print_section("Example 4: System with Anti-Patterns")
    
    config = GraphConfig(
        scale='small',
        scenario='generic',
        num_nodes=5,
        num_applications=15,
        num_topics=10,
        num_brokers=2,
        edge_density=0.3,
        antipatterns=['spof', 'god_object', 'tight_coupling'],
        seed=42
    )
    
    generator = GraphGenerator(config)
    graph = generator.generate()
    
    print("✓ Generated system with anti-patterns: SPOF, God Object, Tight Coupling")
    
    # Find SPOF
    print("\n1. Single Point of Failure (SPOF):")
    spof_candidates = []
    for app in graph['applications']:
        runs_on = [r for r in graph['relationships']['runs_on'] if r['from'] == app['id']]
        if len(runs_on) == 1:
            spof_candidates.append(app)

    if spof_candidates:
        spof = spof_candidates[0]
        print(f"   App {spof['id']} ({spof['name']}):")
        
        # Count dependencies
        deps = [s for s in graph['relationships']['subscribes_to']
                if any(p['from'] == spof['id'] for p in graph['relationships']['publishes_to']
                      if p['to'] == s['to'])]
        print(f"     Dependents: {len(set(d['from'] for d in deps))} apps")
    
    # Find God Object
    print("\n2. God Object:")
    sub_counts = {}
    for sub in graph['relationships']['subscribes_to']:
        sub_counts[sub['from']] = sub_counts.get(sub['from'], 0) + 1
    
    max_subs = max(sub_counts.values()) if sub_counts else 0
    for app_id, count in sub_counts.items():
        if count == max_subs and count > len(graph['topics']) * 0.5:
            app = next(a for a in graph['applications'] if a['id'] == app_id)
            print(f"   App {app_id} ({app['name']}):")
            print(f"     Subscribes to: {count}/{len(graph['topics'])} topics ({count/len(graph['topics'])*100:.1f}%)")
    
    # Detect cycles (tight coupling)
    print("\n3. Tight Coupling (Circular Dependencies):")
    print("   Checking for cycles in pub-sub graph...")
    # Simple cycle detection
    pub_graph = {}
    for pub in graph['relationships']['publishes_to']:
        if pub['from'] not in pub_graph:
            pub_graph[pub['from']] = []
        # Find subscribers to this topic
        for sub in graph['relationships']['subscribes_to']:
            if sub['to'] == pub['to']:
                pub_graph[pub['from']].append(sub['from'])
    
    # Count edges suggesting cycles
    potential_cycles = sum(len(v) for v in pub_graph.values())
    print(f"   Found {potential_cycles} potential cycle edges")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print(" GRAPH GENERATOR - QUICK START EXAMPLES")
    print("="*70)
    print("\nDemonstrating key improvements and new features...\n")
    
    try:
        example_1_basic()
        input("\n[Press Enter to continue to Example 2...]")
        
        example_2_realistic_iot()
        input("\n[Press Enter to continue to Example 3...]")
        
        example_3_financial_ha()
        input("\n[Press Enter to continue to Example 4...]")
        
        example_4_antipatterns()
        
        print_section("Examples Complete!")
        print("✓ All examples executed successfully")
        print("\nNext steps:")
        print("  1. Try: python generate_graph.py --scale medium --scenario iot --output iot.json")
        print("  2. Try: python test_graph_generator.py --quick")
        print("  3. Read: GENERATE_GRAPH_README.md")
        
    except KeyboardInterrupt:
        print("\n\n✗ Examples interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
