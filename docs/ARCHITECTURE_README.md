# Software-as-a-Graph: Architecture

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER / APPLICATION                           │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   ORCHESTRATION LAYER                               │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │         AnalysisOrchestrator (orchestration/)              │     │
│  │  - Coordinates all analysis components                     │     │
│  │  - Manages execution pipeline                              │     │
│  │  - Aggregates results                                      │     │
│  │  - Generates reports                                       │     │
│  └────────────────────────────────────────────────────────────┘     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        ANALYSIS LAYER                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐    │
│  │ Criticality      │  │  Structural      │  │  Reachability   │    │
│  │ Scorer           │  │  Analyzer        │  │  Analyzer       │    │
│  │ • Betweenness    │  │ • Articulation   │  │ • Impact Calc   │    │
│  │ • Art. Points    │  │   Points         │  │ • Path Finding  │    │
│  │ • Impact Score   │  │ • Bridges        │  │ • Resilience    │    │
│  │ • Composite      │  │ • Cycles         │  │   Scoring       │    │
│  └──────────────────┘  └──────────────────┘  └─────────────────┘    │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐    │
│  │ QoS Analyzer     │  │  Centrality      │  │  Failure        │    │
│  │ • Policy Scoring │  │  Analyzer        │  │  Simulator      │    │
│  │ • Compatibility  │  │ • Degree         │  │ • Scenarios     │    │
│  │ • Topic Priority │  │ • Closeness      │  │ • Impact Sim    │    │
│  │                  │  │ • PageRank       │  │ • Validation    │    │
│  └──────────────────┘  └──────────────────┘  └─────────────────┘    │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         CORE LAYER                                  │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │                    GraphModel (core/)                      │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │     │
│  │  │ Application  │  │    Topic     │  │    Broker    │      │     │
│  │  │    Node      │  │     Node     │  │     Node     │      │     │
│  │  │ • Type       │  │ • QoS Policy │  │ • Capacity   │      │     │
│  │  │ • Resources  │  │ • Traffic    │  │ • Perf       │      │     │
│  │  │ • QoS Reqs   │  │ • Schema     │  │ • Config     │      │     │
│  │  └──────────────┘  └──────────────┘  └──────────────┘      │     │
│  │                                                            │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │     │
│  │  │Infrastructure│  │    Edges     │  │  QoS Policy  │      │     │
│  │  │     Node     │  │ • Publishes  │  │ • Durability │      │     │
│  │  │ • Location   │  │ • Subscribes │  │ • Reliability│      │     │
│  │  │ • Resources  │  │ • Routes     │  │ • Deadline   │      │     │
│  │  │ • Health     │  │ • Depends On │  │ • Priority   │      │     │
│  │  └──────────────┘  └──────────────┘  └──────────────┘      │     │
│  └────────────────────────────────────────────────────────────┘     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA SOURCES / SINKS                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │    Neo4j     │  │  NetworkX    │  │     JSON     │               │
│  │   Database   │  │    Graph     │  │    Files     │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │   ROS2 DDS   │  │    Kafka     │  │  CSV Files   │               │
│  │    Topics    │  │   Metadata   │  │              │               │
│  └──────────────┘  └──────────────┘  └──────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
┌──────────────┐
│ Input Data   │
│ (Neo4j, JSON)│
└──────┬───────┘
       │
       │ Load & Parse
       ▼
┌──────────────────┐
│   GraphModel     │ ◄─── Enhanced data model with complete properties
│ • Nodes          │
│ • Edges          │
│ • Metadata       │
└──────┬───────────┘
       │
       │ Convert
       ▼
┌──────────────────┐
│ NetworkX DiGraph │ ◄─── Standard graph representation
└──────┬───────────┘
       │
       │ Analyze
       ▼
┌──────────────────────────────────────────────┐
│         Analysis Pipeline                    │
│                                              │
│  1. Graph Structure Analysis                 │
│     ├─ Node/edge counts                      │
│     ├─ Density                               │
│     └─ Connected components                  │
│                                              │
│  2. QoS Analysis (if enabled)                │
│     ├─ Policy scoring                        │
│     ├─ Topic priorities                      │
│     └─ Compatibility checks                  │
│                                              │
│  3. Structural Analysis                      │
│     ├─ Articulation points                   │
│     ├─ Bridges                               │
│     └─ Cycles                                │
│                                              │
│  4. Criticality Scoring ◄───────────────┐    │
│     ├─ Betweenness centrality            │   │
│     ├─ Articulation point indicator      │   │
│     ├─ Impact score                      │   │
│     └─ C_score = α·BC + β·AP + γ·I ◄────┘    │
│                                              │
│  5. Layer Analysis                           │
│     ├─ Application layer                     │
│     ├─ Infrastructure layer                  │
│     └─ Cross-layer dependencies              │
│                                              │
│  6. Failure Simulation (if enabled)          │
│     ├─ Component removal                     │
│     ├─ Impact measurement                    │
│     └─ Connectivity changes                  │
│                                              │
│  7. Recommendations                          │
│     ├─ Critical components                   │
│     ├─ Vulnerabilities                       │
│     └─ Mitigation strategies                 │
└──────┬───────────────────────────────────────┘
       │
       │ Output
       ▼
┌──────────────────────────────────────────────┐
│            Results                           │
│  ┌────────────────────────────────────┐      │
│  │ JSON Export                        │      │
│  │ • All metrics                      │      │
│  │ • Scores per component             │      │
│  │ • Recommendations                  │      │
│  └────────────────────────────────────┘      │
│                                              │
│  ┌────────────────────────────────────┐      │
│  │ Console Output                     │      │
│  │ • Summary statistics               │      │
│  │ • Top critical components          │      │
│  │ • Key recommendations              │      │
│  └────────────────────────────────────┘      │
│                                              │
│  ┌────────────────────────────────────┐      │
│  │ Visualizations (future)            │      │
│  │ • Interactive graphs               │      │
│  │ • Layer views                      │      │
│  │ • Metrics heatmaps                 │      │
│  └────────────────────────────────────┘      │
└──────────────────────────────────────────────┘
```

## Criticality Score Calculation Flow

```
┌────────────────────────────────────────────────────────────────┐
│            Composite Criticality Score Calculation             │
│                                                                │
│  Input: NetworkX DiGraph G, Optional QoS Scores                │
└─────────────────────┬──────────────────────────────────────────┘
                      │
    ┌─────────────────┼─────────────────┐
    │                 │                 │
    ▼                 ▼                 ▼
┌───────────┐   ┌─────────────┐   ┌─────────────┐
│Betweenness│   │Articulation │   │   Impact    │
│Centrality │   │   Points    │   │    Score    │
│  C_B(v)   │   │   AP(v)     │   │    I(v)     │
└─────┬─────┘   └──────┬──────┘   └──────┬──────┘
      │                │                 │
      │ Normalize      │ Binary          │ Calculate
      │ to [0,1]       │ {0,1}           │ 1 - |R(G-v)|/|R(G)|
      │                │                 │
      ▼                ▼                 ▼
┌─────────┐   ┌──────────────┐   ┌─────────────┐
│C_B^norm │   │   AP(v)      │   │    I(v)     │
│ [0,1]   │   │   {0,1}      │   │   [0,1]     │
└─────┬───┘   └──────┬───────┘   └──────┬──────┘
      │              │                  │
      │    ┌─────────┼───────────┐      │
      │    │         │           │      │
      ▼    ▼         ▼           ▼      ▼
    ┌─────────────────────────────────────────┐
    │   C_score(v) = α·C_B^norm(v)            │
    │               + β·AP(v)                 │
    │               + γ·I(v)                  │
    │                                         │
    │   Default: α=0.4, β=0.3, γ=0.3          │
    │   (Configurable)                        │
    └──────────────────┬──────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────────┐
    │  Optional: QoS Adjustment (Topics)      │
    │  C_score' = C_score × (1 + qos_score/2) │
    │  Capped at 1.0                          │
    └──────────────────┬──────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────────┐
    │        Criticality Level                │
    │  ≥0.8: CRITICAL                         │
    │  ≥0.6: HIGH                             │
    │  ≥0.4: MEDIUM                           │
    │  ≥0.2: LOW                              │
    │  <0.2: MINIMAL                          │
    └──────────────────┬──────────────────────┘
                       │
                       ▼
    ┌─────────────────────────────────────────┐
    │     CompositeCriticalityScore           │
    │  • component: str                       │
    │  • component_type: str                  │
    │  • betweenness_centrality_norm: float   │
    │  • is_articulation_point: bool          │
    │  • impact_score: float                  │
    │  • composite_score: float               │
    │  • criticality_level: enum              │
    │  • qos_score: float                     │
    │  • components_affected: int             │
    │  • ...                                  │
    └─────────────────────────────────────────┘
```

## Component Interaction Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                    User Code                                   │
└─────────┬──────────────────────────────────────────────────────┘
          │
          │ create/load
          ▼
┌─────────────────┐
│   GraphModel    │
└────────┬────────┘
         │
         │ convert
         ▼
┌─────────────────┐       ┌──────────────────────────────┐
│ NetworkX Graph  │──────▶│  AnalysisOrchestrator        │
└─────────────────┘       └───────────┬──────────────────┘
                                      │
                     ┌────────────────┼────────────────┐
                     │                │                │
                     ▼                ▼                ▼
          ┌──────────────┐  ┌────────────┐  ┌──────────────┐
          │  QoSAnalyzer │  │Structural  │  │Reachability  │
          │              │  │Analyzer    │  │Analyzer      │
          └──────┬───────┘  └─────┬──────┘  └──────┬───────┘
                 │                │                │
                 │                │                │
                 └────────────────┼────────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ CompositeCriticalityScorer│
                    │                           │
                    │ Uses:                     │
                    │ • Centralities            │
                    │ • Articulation Points     │
                    │ • Impact Scores           │
                    │ • QoS Scores              │
                    └─────────┬─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Results Dict    │
                    │ • Scores         │
                    │ • Recommendations│
                    │ • Metrics        │
                    └──────────────────┘
```

## Class Hierarchy

```
GraphModel (core/graph_model.py)
├── ApplicationNode
│   └── to_dict() → Neo4j compatible
├── TopicNode
│   ├── QoSPolicy
│   │   └── get_criticality_score()
│   └── get_qos_criticality()
├── BrokerNode
│   └── get_capacity_utilization()
├── InfrastructureNode
│   └── get_resource_utilization()
└── Edge (base)
    ├── PublishesEdge
    ├── SubscribesEdge
    ├── RoutesEdge
    ├── RunsOnEdge
    ├── ConnectsToEdge
    └── DependsOnEdge

CompositeCriticalityScorer (analysis/criticality_scorer.py)
├── calculate_all_scores(graph) → Dict[CompositeCriticalityScore]
├── get_critical_components(scores, threshold)
├── get_top_critical(scores, n)
└── summarize_criticality(scores)

AnalysisOrchestrator (orchestration/analysis_orchestrator.py)
├── analyze_graph(graph, model, enable_simulation)
├── print_summary()
└── export_results(filename)

QoSAnalyzer (analysis/qos_analyzer.py)
├── analyze_graph(graph, model)
└── analyze_qos_compatibility(pub, sub, topic)

StructuralAnalyzer (analysis/structural_analyzer.py)
├── analyze(graph)
├── find_single_points_of_failure(graph)
└── analyze_redundancy(graph)

ReachabilityAnalyzer (analysis/reachability_analyzer.py)
├── analyze_impact(component)
├── find_critical_paths(source, target)
└── calculate_resilience_score()
```

## State Transitions

```
┌─────────────┐
│   Created   │  GraphModel initialized
└──────┬──────┘
       │ add nodes & edges
       ▼
┌─────────────┐
│  Populated  │  Model contains components
└──────┬──────┘
       │ convert to NetworkX
       ▼
┌─────────────┐
│  Graphified │  Ready for analysis
└──────┬──────┘
       │ run analysis
       ▼
┌─────────────┐
│  Analyzing  │  Computations in progress
└──────┬──────┘
       │ calculations complete
       ▼
┌─────────────┐
│  Analyzed   │  Results available
└──────┬──────┘
       │ export/visualize
       ▼
┌─────────────┐
│  Exported   │  Results saved/displayed
└─────────────┘
```

## Extensibility Points

```
1. New Component Types
   └─ Add new XxxNode class in graph_model.py
      └─ Implement to_dict()
      └─ Add to GraphModel

2. New Edge Types
   └─ Add new XxxEdge class in graph_model.py
      └─ Implement to_dict()
      └─ Add to GraphModel edge lists

3. New Analysis Metrics
   └─ Create new XxxAnalyzer in analysis/
      └─ Implement analyze(graph) method
      └─ Integrate with AnalysisOrchestrator

4. Custom Scoring Functions
   └─ Extend CompositeCriticalityScorer
      └─ Override _calculate_xxx methods
      └─ Adjust weight configuration

5. New Failure Scenarios
   └─ Implement in simulation/
      └─ Define scenario parameters
      └─ Calculate impact
      └─ Return structured results

6. Custom Visualizations
   └─ Create renderer in visualization/
      └─ Accept graph and scores
      └─ Generate visual output
      └─ Support interactive features
```

## Key Design Decisions

1. **Dataclasses Over Dicts**
   - Type safety
   - Auto-generated methods
   - IDE support

2. **NetworkX for Graph Operations**
   - Standard library
   - Extensive algorithms
   - Good performance

3. **Configurable Weights**
   - Different analysis priorities
   - Domain-specific tuning
   - Research flexibility

4. **Caching Strategy**
   - Expensive computations cached
   - Clear cache on graph changes
   - Transparent to users

5. **Modular Architecture**
   - Independent components
   - Easy testing
   - Simple extension

This architecture provides a solid foundation for analyzing distributed pub-sub systems with clear separation of concerns, extensibility, and performance optimization.
