"use client"

import { useState, useMemo } from "react"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import {
  BookMarked, Search, Network, GitBranch, BarChart3, Shield,
  Zap, CheckCircle2, Radio, Layers, Brain, Server, Database,
} from "lucide-react"

// ─── Types ────────────────────────────────────────────────────────────────────

type SectionKey =
  | "Component Types"
  | "Edge Types"
  | "Graph Structure"
  | "RMAV Scores"
  | "Structural Metrics"
  | "Anti-Patterns"
  | "Simulation"
  | "Validation"
  | "Pub-Sub & Statistics"
  | "Pipeline"
  | "ML & Training"
  | "Layers & Infrastructure"
  | "Neo4j Schema Fields"

interface Term {
  term: string
  definition: string
  formula?: string
}

interface Section {
  key: SectionKey
  textColor: string
  borderColor: string
  bgColor: string
  ringColor: string
  badgeClass: string
  filterActive: string
  filterIdle: string
  Icon: React.ElementType
  terms: Term[]
}

// ─── Data ─────────────────────────────────────────────────────────────────────

const SECTIONS: Section[] = [
  {
    key: "Component Types",
    textColor: "text-blue-400",
    borderColor: "border-blue-500/20",
    bgColor: "bg-blue-500/[0.06]",
    ringColor: "bg-blue-500/10",
    badgeClass: "bg-blue-500/10 text-blue-400 border-blue-500/20",
    filterActive: "bg-blue-500/15 border-blue-500/40 text-blue-300",
    filterIdle: "border-blue-500/30 text-blue-400",
    Icon: Network,
    terms: [
      { term: "Application", definition: "A software process that publishes or subscribes to topics. The primary actor in a pub-sub system." },
      { term: "Library", definition: "A shared software dependency used by one or more applications. Its failure can propagate to all consumers simultaneously." },
      { term: "Broker", definition: "A routing node that relays messages between publishers and subscribers across named topic channels." },
      { term: "Topic", definition: "A named message channel. Publishers write to it; subscribers read from it." },
      { term: "Node", definition: "A physical or virtual infrastructure host. Applications and brokers are deployed on nodes." },
      { term: "CSMS", definition: "Top-level system scope in the component hierarchy. Groups all logical segments of a deployment." },
      { term: "CSS", definition: "A logical domain or segment grouping related configuration items within the system." },
      { term: "CSCI", definition: "A configuration item — a named unit of software configuration within a domain." },
      { term: "CSC", definition: "A component grouping that contains deployable application units within a configuration item." },
      { term: "CSU", definition: "A deployable application unit — the leaf-level element of the component hierarchy." },
      { term: "Component", definition: "Any named unit in the system graph — Application, Library, Broker, Topic, or Node." },
      { term: "Publisher", definition: "An application that writes messages to a topic. It drives the upstream side of a data flow." },
      { term: "Subscriber", definition: "An application that reads messages from a topic. It drives the downstream side of a data flow." },
      { term: "Middleware", definition: "The messaging layer between applications — typically brokers and topic channels — that decouples producers from consumers." },
      { term: "Service", definition: "A named unit of functionality exposed by an application, often mapped to a set of topics it owns." },
      { term: "Microservice", definition: "A small, independently deployable application unit that owns a narrow slice of system behaviour." },
      { term: "Domain", definition: "A logical grouping of related components sharing a common operational concern or data boundary." },
      { term: "Segment", definition: "A named subset of the system used to scope analysis, visualisation, and communication statistics." },
      { term: "Subsystem", definition: "A self-contained group of components that together fulfil a distinct functional responsibility within the larger system." },
      { term: "Process", definition: "A running instance of an application on an infrastructure node. Multiple processes may share the same executable." },
      { term: "Container", definition: "A lightweight process-isolation unit. Applications and brokers are often containerised on infrastructure nodes." },
      { term: "Host", definition: "The physical or virtual machine on which components are deployed — equivalent to a Node in the graph model." },
      { term: "Endpoint", definition: "A network address or socket through which a component sends or receives messages." },
      { term: "Queue", definition: "A persistent or in-memory buffer within a broker holding undelivered messages for a topic." },
      { term: "Plugin", definition: "An optional extension loaded by a broker or application to add protocol support or processing logic." },
      { term: "Component Type", definition: "The class of a graph node — Application, Library, Broker, Topic, or Node. Governs which metrics and rules apply." },
      { term: "Role", definition: "An optional attribute on an application indicating its functional responsibility within the system." },
      { term: "Version", definition: "An optional attribute on a library node indicating the installed package version." },
      { term: "Broker Type", definition: "The protocol variant of a broker — e.g. DDS, MQTT, ROS2, Kafka. Governs routing semantics." },
    ],
  },
  {
    key: "Edge Types",
    textColor: "text-purple-400",
    borderColor: "border-purple-500/20",
    bgColor: "bg-purple-500/[0.06]",
    ringColor: "bg-purple-500/10",
    badgeClass: "bg-purple-500/10 text-purple-400 border-purple-500/20",
    filterActive: "bg-purple-500/15 border-purple-500/40 text-purple-300",
    filterIdle: "border-purple-500/30 text-purple-400",
    Icon: GitBranch,
    terms: [
      { term: "DEPENDS_ON", definition: "A derived dependency edge indicating component A requires component B to function normally." },
      { term: "PUBLISHES_TO", definition: "An edge from an application to a topic indicating the application sends messages to that channel." },
      { term: "SUBSCRIBES_TO", definition: "An edge from an application to a topic indicating the application receives messages from that channel." },
      { term: "RUNS_ON", definition: "An edge connecting an application or broker to the infrastructure node it is deployed on." },
      { term: "ROUTES", definition: "An edge from a broker to a topic indicating the broker is responsible for delivering messages on that channel." },
      { term: "USES", definition: "An edge from an application to a library indicating a runtime software dependency." },
      { term: "CONNECTS_TO", definition: "A physical or network connectivity edge between two infrastructure nodes." },
      { term: "Structural Edge", definition: "A raw topology edge from which dependency edges are derived. Simulation uses these directly." },
      { term: "app_to_app", definition: "A dependency subtype between two applications that share a common topic channel." },
      { term: "app_to_lib", definition: "A dependency subtype from an application to a shared library it depends on at runtime." },
      { term: "app_to_broker", definition: "A dependency subtype from an application to the broker routing its topics." },
      { term: "node_to_node", definition: "A lifted dependency between infrastructure nodes when their hosted applications share dependency chains." },
      { term: "node_to_broker", definition: "A lifted dependency from an infrastructure node to a broker serving a hosted application." },
      { term: "broker_to_broker", definition: "A symmetric shared-fate dependency between brokers co-located on the same physical node." },
      { term: "Edge Weight", definition: "A value between 0 and 1 on a dependency edge representing the maximum QoS severity of the relationship." },
      { term: "Path Count", definition: "Number of independent paths between two components. Higher counts indicate stronger coupling." },
      { term: "Directed Edge", definition: "An edge with a source and a target. Dependency edges are directed from dependent to dependency." },
      { term: "Undirected Edge", definition: "An edge with no inherent direction — used for symmetric relationships such as physical co-location." },
      { term: "Weighted Edge", definition: "An edge carrying a numeric weight that influences metric calculations downstream." },
      { term: "Derived Edge", definition: "An edge not present in the raw topology but inferred by applying dependency-derivation rules." },
      { term: "Lifted Dependency", definition: "A dependency promoted from application level to node level, reflecting infrastructure-level shared fate." },
      { term: "Transitive Edge", definition: "An edge that connects two nodes through intermediate hops, captured by dependency-derivation chains." },
      { term: "Co-location Edge", definition: "An implicit link between components that share a physical host, creating shared-fate failure risk." },
      { term: "Trust Edge", definition: "A dependency traversed during compromise propagation if its weight exceeds the trust threshold." },
      { term: "Dependency Subtype", definition: "A label on a DEPENDS_ON edge indicating which derivation rule produced it — e.g. app_to_app or node_to_broker." },
      { term: "Coupling Intensity", definition: "A measure of how tightly two components are bound, combining edge weight and path count." },
      { term: "Efferent Coupling", definition: "Outgoing dependencies — how many components this node depends on. High efferent coupling reduces stability." },
      { term: "Afferent Coupling", definition: "Incoming dependencies — how many components depend on this node. High afferent coupling raises reliability risk." },
      { term: "Coupling Width", definition: "Number of distinct neighbours a component has. Wide coupling increases change-impact surface." },
      { term: "Multi-hop Path", definition: "A dependency chain traversing two or more intermediate nodes. Longer hops amplify cascade depth." },
    ],
  },
  {
    key: "Graph Structure",
    textColor: "text-emerald-400",
    borderColor: "border-emerald-500/20",
    bgColor: "bg-emerald-500/[0.06]",
    ringColor: "bg-emerald-500/10",
    badgeClass: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
    filterActive: "bg-emerald-500/15 border-emerald-500/40 text-emerald-300",
    filterIdle: "border-emerald-500/30 text-emerald-400",
    Icon: BarChart3,
    terms: [
      { term: "Articulation Point", definition: "A node whose removal increases the number of disconnected graph components. A structural marker for single-point risk." },
      { term: "Bridge", definition: "An edge whose removal disconnects two previously connected parts of the graph." },
      { term: "SCC", definition: "Strongly Connected Component — a maximal subgraph where every node can reach every other. Cyclic dependencies form SCCs." },
      { term: "Graph Density", definition: "Ratio of actual edges to the maximum possible edges. Higher density means more interconnection." },
      { term: "Diameter", definition: "The longest shortest path in the graph — the maximum number of hops between any two nodes." },
      { term: "Average Path Length", definition: "Mean shortest-path distance across all node pairs. Lower values suggest a more compact, well-connected topology." },
      { term: "Clustering Coefficient", definition: "Likelihood that a node's neighbours are also connected. Higher values indicate local path redundancy." },
      { term: "Degree", definition: "Total number of edges connected to a node — a basic connectivity measure." },
      { term: "In-degree", definition: "Number of incoming edges — how many other components directly depend on this node." },
      { term: "Out-degree", definition: "Number of outgoing edges — how many components this node depends on." },
      { term: "Hub Node", definition: "A component with far more connections than average. Its failure affects many others." },
      { term: "Isolated Node", definition: "A component with no connections — neither a dependent nor a dependency." },
      { term: "Transitive Dependency", definition: "An indirect dependency through a chain of components. Failures can propagate unexpectedly through these chains." },
      { term: "Root Component", definition: "A component with no outgoing dependencies — a consumer at the top of the dependency chain." },
      { term: "Leaf Component", definition: "A component with no incoming dependencies — a foundational element everything else depends on." },
      { term: "Source Component", definition: "A component with outgoing dependencies but no incoming ones — an entry point." },
      { term: "Sink Component", definition: "A component with incoming dependencies but no outgoing ones — a foundational layer." },
      { term: "Network Density", definition: "Fraction of possible edges that are actually present. Sparse graphs are more vulnerable to targeted removal." },
      { term: "Node Weight", definition: "A component's operational priority weight between 0 and 1. Higher values mean more business-critical." },
      { term: "Num Components", definition: "Number of weakly connected subgraphs. Values above 1 mean the topology is fragmented." },
      { term: "Num Bridges", definition: "Count of edges that would disconnect the graph if removed. Each is a structural risk." },
      { term: "Avg Clustering", definition: "Average clustering coefficient across all nodes — a measure of local neighbourhood redundancy." },
      { term: "Avg Degree", definition: "Mean number of edges per node — a proxy for overall connectivity density." },
      { term: "Component Isolation", definition: "A component is isolated when it has no dependencies or no dependents in the graph." },
      { term: "Resilience Score", definition: "A composite redundancy measure based on alternative paths available if key components fail." },
      { term: "WCC", definition: "Weakly Connected Component — a subgraph where every pair of nodes is connected ignoring edge direction." },
      { term: "DAG", definition: "Directed Acyclic Graph — a graph with no cycles. Most healthy dependency graphs aim to approximate a DAG." },
      { term: "Cycle Detection", definition: "The process of identifying circular dependency chains in the graph that could cause coupled failures." },
      { term: "Shortest Path", definition: "The minimum number of hops between two nodes. Used in closeness, CDI, and path-breaking calculations." },
      { term: "Reachability", definition: "Whether a path exists from one node to another in the graph, following directed edges." },
      { term: "BFS", definition: "Breadth-First Search — a graph traversal algorithm used in cascade, change, and compromise simulators." },
      { term: "PageRank", definition: "An iterative centrality algorithm that scores nodes by the importance of nodes linking to them." },
      { term: "Eigenvector Centrality", definition: "A centrality measure that scores nodes by the scores of their neighbours — being connected to important nodes increases your own score." },
      { term: "Topological Sort", definition: "An ordering of nodes such that every dependency precedes its dependent. Applicable only to DAGs." },
      { term: "Vertex", definition: "A node in graph theory terms. Each component in the system maps to one vertex." },
      { term: "Directed Graph", definition: "A graph where edges have direction. The DEPENDS_ON graph is directed from dependent to dependency." },
      { term: "Multigraph", definition: "A graph where two nodes may be connected by more than one edge of different types or subtypes." },
      { term: "Weighted Graph", definition: "A graph where edges carry numeric weights used in metric calculations." },
      { term: "Sparse Graph", definition: "A graph with few edges relative to the number of possible edges — common in large distributed systems." },
      { term: "Dense Graph", definition: "A graph where most possible edges are present — indicates tight coupling and higher cascading risk." },
      { term: "Neighbourhood", definition: "The set of nodes directly connected to a given node by an edge." },
      { term: "Eccentricity", definition: "The maximum shortest path from a node to any other reachable node. Nodes with low eccentricity are centrally placed." },
      { term: "Radius", definition: "The minimum eccentricity across all nodes in the graph — a measure of how compact the topology is." },
      { term: "Graph Partition", definition: "Division of the graph into disjoint subsets. Partitioning events indicate fragmentation after a failure." },
      { term: "Connected Components", definition: "Maximal subgraphs where every node is reachable from every other when ignoring edge direction." },
      { term: "Edge Utilization", definition: "Fraction of maximum possible edges that are actually present — equivalent to network density." },
      { term: "Degree Distribution", definition: "Histogram of how many connections each component has — reveals hubs and isolated nodes." },
    ],
  },
  {
    key: "RMAV Scores",
    textColor: "text-violet-400",
    borderColor: "border-violet-500/20",
    bgColor: "bg-violet-500/[0.06]",
    ringColor: "bg-violet-500/10",
    badgeClass: "bg-violet-500/10 text-violet-400 border-violet-500/20",
    filterActive: "bg-violet-500/15 border-violet-500/40 text-violet-300",
    filterIdle: "border-violet-500/30 text-violet-400",
    Icon: Shield,
    terms: [
      { term: "RMAV", definition: "The four quality dimensions — Reliability, Maintainability, Availability, Vulnerability — scored for every component." },
      { term: "R(v) — Reliability", definition: "Fault-propagation risk — how far failures starting at this component will cascade through the system.", formula: "R(v) = 0.45·RPR + 0.30·DG_in + 0.25·CDPot" },
      { term: "M(v) — Maintainability", definition: "Change-impact risk — how costly a modification to this component is for the rest of the system.", formula: "M(v) = 0.35·BT + 0.30·w_out + 0.15·CQP + 0.12·CouplingRisk + 0.08·(1−CC)" },
      { term: "A(v) — Availability", definition: "Connectivity-disruption risk — how severely the network is partitioned when this component is removed.", formula: "A(v) = 0.35·AP_c + 0.25·QSPOF + 0.25·BR + 0.10·CDI + 0.05·w(v)" },
      { term: "V(v) — Vulnerability", definition: "Attack-surface risk — how attractive and reachable this component is for adversarial compromise.", formula: "V(v) = 0.40·REV + 0.35·RCL + 0.25·QADS" },
      { term: "Q(v) — Overall Quality", definition: "Weighted composite of all four RMAV dimensions. Higher values indicate greater predicted impact on failure.", formula: "Q(v) = 0.24·R + 0.17·M + 0.43·A + 0.16·V" },
      { term: "Q_ensemble", definition: "Blended quality score combining GNN predictions with rule-based RMAV scores.", formula: "Q_ensemble = α·Q_GNN + (1−α)·Q_RMAV" },
      { term: "CRITICAL", definition: "Highest risk tier — score is statistically extreme relative to all other components in the system." },
      { term: "HIGH", definition: "Score exceeds the 75th percentile — above-average risk that warrants architectural attention." },
      { term: "MEDIUM", definition: "Score exceeds the median — moderate risk worth monitoring over time." },
      { term: "LOW", definition: "Score exceeds the 25th percentile — below-average risk, relatively stable." },
      { term: "MINIMAL", definition: "Score at or below the 25th percentile — minimal predicted impact on the system." },
      { term: "IQR", definition: "Interquartile Range — spread between the 25th and 75th percentiles used to set CRITICAL thresholds." },
      { term: "Box-Plot Classification", definition: "A statistical method for assigning risk tiers using quartiles rather than fixed cut-off values." },
      { term: "AHP", definition: "Analytic Hierarchy Process — a method for deriving RMAV dimension weights from pairwise importance comparisons." },
      { term: "AHP Shrinkage", definition: "Blends AHP-derived weights with a uniform prior to prevent extreme weight concentrations." },
      { term: "DG_in", definition: "Normalised in-degree — direct dependent count used as an input to the Reliability score." },
      { term: "DG_out", definition: "Normalised out-degree — direct dependency count used in coupling and instability calculations." },
      { term: "w_out", definition: "QoS-weighted efferent coupling — sum of outgoing dependency weights. An input to Maintainability." },
      { term: "w(v)", definition: "Operational priority weight of component v, sourced from QoS attributes. Used in Availability and QSPOF." },
      { term: "w_in", definition: "Inbound dependency weight — sum of incoming dependency weights. Used as QADS in Vulnerability scoring." },
      { term: "Dimension Weight", definition: "The AHP-derived coefficient multiplying each RMAV dimension in the Q(v) formula." },
      { term: "Pairwise Comparison", definition: "An AHP step where two dimensions are compared for relative importance on a 1–9 scale." },
      { term: "Consistency Ratio", definition: "A measure of logical consistency in the AHP pairwise comparison matrix. Values below 0.10 are acceptable." },
      { term: "Score Normalisation", definition: "Scaling raw metric values to a common range before combining them into dimension scores." },
      { term: "Closed-Form Formula", definition: "An algebraic expression that computes a score directly from inputs without iterative optimisation." },
      { term: "Shrinkage Factor λ", definition: "The parameter controlling AHP-to-uniform blending. λ=0.7 means 70% AHP, 30% uniform prior." },
      { term: "Uniform Prior", definition: "An equal-weight baseline used during AHP shrinkage to prevent any single dimension dominating." },
      { term: "Score Distribution", definition: "The statistical spread of RMAV or Q(v) scores across all components in the system." },
      { term: "Percentile Threshold", definition: "A score cut-off derived from the population distribution, used for classification fallback in small graphs." },
      { term: "Population Stats", definition: "Summary statistics — mean, median, quartiles — computed over all component scores to anchor thresholds." },
      { term: "Q3", definition: "75th percentile of the score distribution — the lower bound of the HIGH tier and base of CRITICAL detection." },
      { term: "Median", definition: "50th percentile of the score distribution — the boundary between MEDIUM and LOW tiers." },
      { term: "Q1", definition: "25th percentile of the score distribution — the boundary between LOW and MINIMAL tiers." },
    ],
  },
  {
    key: "Structural Metrics",
    textColor: "text-cyan-400",
    borderColor: "border-cyan-500/20",
    bgColor: "bg-cyan-500/[0.06]",
    ringColor: "bg-cyan-500/10",
    badgeClass: "bg-cyan-500/10 text-cyan-400 border-cyan-500/20",
    filterActive: "bg-cyan-500/15 border-cyan-500/40 text-cyan-300",
    filterIdle: "border-cyan-500/30 text-cyan-400",
    Icon: BarChart3,
    terms: [
      { term: "RPR", definition: "Reverse PageRank — PageRank computed on the transposed graph. Captures how many components ultimately depend on this node." },
      { term: "Reverse PageRank", definition: "PageRank on the reversed dependency graph. Components with many dependents score higher." },
      { term: "Betweenness", definition: "Fraction of all shortest paths that pass through this component. High values signal a structural bottleneck." },
      { term: "Betweenness Centrality", definition: "How often a node lies on the shortest path between other nodes. A structural bottleneck measure." },
      { term: "Pubsub Betweenness", definition: "Betweenness centrality computed specifically on the publish-subscribe message routing graph." },
      { term: "REV", definition: "Reverse Eigenvector centrality — measures strategic importance for an attacker via eigenvector centrality on the transposed graph." },
      { term: "RCL", definition: "Reverse Closeness on the transposed graph — measures how quickly a compromise can spread from this node." },
      { term: "Closeness", definition: "Inverse of average shortest path length from a node to all others. Higher means faster propagation." },
      { term: "CDPot", definition: "Cascade Depth Potential — estimated maximum depth of failure propagation reachable from this component." },
      { term: "MPCI", definition: "Multi-Path Coupling Intensity — number of parallel failure paths that amplify the cascade depth estimate." },
      { term: "Bridge Ratio", definition: "Fraction of a component's incident edges that are bridges. Higher values mean most connections are single-path." },
      { term: "BR", definition: "Bridge Ratio — fraction of incident edges that are bridges at the edge level." },
      { term: "AP_c", definition: "Directional articulation-point score — maximum of the out-direction and in-direction removal impact scores." },
      { term: "QSPOF", definition: "QoS-Scaled SPOF Severity — articulation-point score multiplied by the component's operational priority weight.", formula: "QSPOF = AP_c × w(v)" },
      { term: "CDI", definition: "Connectivity Degradation Index — normalised increase in average path length when this component is removed." },
      { term: "QADS", definition: "QoS-weighted Attack-Dependent Surface — inbound dependency weight capturing upstream adversarial exposure." },
      { term: "CQP", definition: "Code Quality Penalty — composite of cyclomatic complexity, Martin instability, and LCOM. Non-zero only when source-code attributes exist.", formula: "CQP = 0.40·CC_norm + 0.35·Instability + 0.25·LCOM" },
      { term: "LCOM", definition: "Lack of Cohesion of Methods — measures how many unrelated responsibilities a class contains." },
      { term: "CouplingRisk", definition: "Peaks when a component is both heavily depended-upon and depends on many others simultaneously.", formula: "CouplingRisk = 1 − |2·Instability − 1|" },
      { term: "Martin Instability", definition: "Ratio of outgoing to total couplings. 0 means fully stable; 1 means fully unstable." },
      { term: "Blast Radius", definition: "The number or fraction of components reachable from a given component through dependency chains." },
      { term: "Bottleneck Score", definition: "A composite of betweenness, articulation-point status, and cascade depth. Higher means more structural risk." },
      { term: "LoC", definition: "Lines of Code — a rough size indicator for application or library nodes." },
      { term: "Cyclomatic CC", definition: "Cyclomatic Complexity — number of independent code paths. Higher values suggest harder-to-maintain modules." },
      { term: "Instability", definition: "Martin's software instability metric — fraction of a component's couplings that are outgoing." },
      { term: "Dependency Depth", definition: "Longest DEPENDS_ON chain leading to a component. Deeper dependencies amplify cascade effects." },
      { term: "Min-Max Normalisation", definition: "Scaling a metric to [0,1] using the population minimum and maximum. Used for complexity and LCOM inputs." },
      { term: "DG_in Norm", definition: "Normalised in-degree — fraction of the maximum observed in-degree across the component population." },
      { term: "DG_out Norm", definition: "Normalised out-degree — fraction of the maximum observed out-degree across the component population." },
      { term: "Path-Breaking", definition: "An edge or node is path-breaking if its removal eliminates all shortest paths between some node pair." },
      { term: "Weighted Reachability", definition: "Reachability weighted by the priority of reached nodes — losing high-weight components matters more." },
      { term: "Attack Surface", definition: "The total set of entry points through which an adversary could compromise a component or trigger propagation." },
      { term: "Trust Threshold", definition: "Minimum dependency weight required for a compromise to propagate along an edge during vulnerability simulation." },
      { term: "QoS Severity", definition: "Numeric priority derived from a topic's QoS reliability and durability settings. Feeds into edge and node weights." },
      { term: "Structural Metric", definition: "A graph-derived measure computed purely from topology, without simulation or ML." },
      { term: "Centrality", definition: "A family of metrics measuring how important or influential a node is based on its graph position." },
      { term: "Global Centrality", definition: "A centrality measure computed over the entire graph — as opposed to local neighbourhood-only measures." },
      { term: "Local Centrality", definition: "A centrality measure based only on immediate neighbours — such as degree or clustering coefficient." },
      { term: "Harmonic Centrality", definition: "Sum of inverse shortest-path lengths from a node to all others. Handles disconnected graphs more gracefully than closeness." },
      { term: "k-core", definition: "The maximal subgraph where every node has at least k neighbours. Higher k-cores contain denser, more robust clusters." },
      { term: "Impact Score", definition: "A numeric value summarising how much damage removing a component causes across the system." },
      { term: "Sensitivity", definition: "How much a final score changes when a single input metric changes — used in weight-sensitivity analysis." },
      { term: "Structural Fingerprint", definition: "The unique combination of metric values characterising a component's position and risk in the graph." },
    ],
  },
  {
    key: "Anti-Patterns",
    textColor: "text-red-400",
    borderColor: "border-red-500/20",
    bgColor: "bg-red-500/[0.06]",
    ringColor: "bg-red-500/10",
    badgeClass: "bg-red-500/10 text-red-400 border-red-500/20",
    filterActive: "bg-red-500/15 border-red-500/40 text-red-300",
    filterIdle: "border-red-500/30 text-red-400",
    Icon: Zap,
    terms: [
      { term: "SPOF", definition: "Single Point of Failure — a component whose removal alone causes significant system-wide disruption." },
      { term: "FAILURE_HUB", definition: "A component with critically high Reliability risk — too many downstream dependents would cascade-fail on its outage." },
      { term: "GOD_COMPONENT", definition: "A component that is both a structural bottleneck and heavily connected — a maintenance and reliability liability." },
      { term: "TARGET", definition: "A component with critically high Vulnerability — a high-value target for adversarial compromise." },
      { term: "BRIDGE_EDGE", definition: "An edge whose removal disconnects the graph — a structural weak link at the connection level." },
      { term: "EXPOSURE", definition: "A component with high Vulnerability and high closeness centrality — easy to reach and compromise." },
      { term: "CYCLE", definition: "A strongly connected component of size 2 or more — circular dependencies causing coupled failures." },
      { term: "HUB_AND_SPOKE", definition: "A low-clustering, high-degree node where everything routes through a single hub with little redundancy." },
      { term: "CHAIN", definition: "A long linear dependency path of four or more hops. Deep chains amplify cascade failure depth." },
      { term: "SYSTEMIC_RISK", definition: "More than 20% of system components are CRITICAL — a broad architectural fragility warning." },
      { term: "Tightly Coupled", definition: "Two components with high bidirectional dependency weight, making independent evolution or failure difficult." },
      { term: "Circular Dependency", definition: "A cycle in the dependency graph where A depends on B and B depends on A, directly or transitively." },
      { term: "Dead Component", definition: "A component with no active publishers or subscribers — a candidate for removal or decommissioning." },
      { term: "Phantom Dependency", definition: "A declared dependency that never carries message traffic, inflating coupling scores without real risk." },
      { term: "Implicit Coupling", definition: "Shared state or a shared resource creating a dependency that is not captured by explicit graph edges." },
      { term: "Megaservice", definition: "A single component that performs far too many responsibilities, creating wide coupling and high change risk." },
      { term: "Service Sprawl", definition: "An excessive number of micro-components with tangled fine-grained dependencies, increasing coordination overhead." },
      { term: "Dependency Inversion Violation", definition: "High-level components depending directly on low-level ones without an abstraction layer, increasing fragility." },
      { term: "Shared-Fate Failure", definition: "Multiple components that always fail together because they share a host, library, or broker." },
      { term: "Latent SPOF", definition: "A component that is not yet an articulation point but would become one if one existing redundant path were removed." },
      { term: "Bottleneck Accumulation", definition: "Multiple high-betweenness nodes forming a cluster, meaning the topology has several simultaneous structural bottlenecks." },
      { term: "Coupling Cluster", definition: "A group of tightly interdependent components whose collective failure probability is higher than the sum of individual risks." },
    ],
  },
  {
    key: "Simulation",
    textColor: "text-orange-400",
    borderColor: "border-orange-500/20",
    bgColor: "bg-orange-500/[0.06]",
    ringColor: "bg-orange-500/10",
    badgeClass: "bg-orange-500/10 text-orange-400 border-orange-500/20",
    filterActive: "bg-orange-500/15 border-orange-500/40 text-orange-300",
    filterIdle: "border-orange-500/30 text-orange-400",
    Icon: Zap,
    terms: [
      { term: "Cascade Failure", definition: "A chain reaction where the failure of one component causes others to fail in sequence." },
      { term: "Exhaustive Simulation", definition: "Removes every component one-by-one and records the resulting impact. Produces ground-truth criticality rankings." },
      { term: "Monte Carlo Simulation", definition: "Randomly samples failure scenarios to estimate impact distribution. Faster than exhaustive but less complete." },
      { term: "I(v) — Overall Impact", definition: "Composite simulation damage score from removing component v.", formula: "I = 0.35·reach + 0.25·frag + 0.25·thru + 0.15·flow" },
      { term: "IR(v) — Reliability Impact", definition: "Fault-propagation ground truth — measures cascade reach and depth from removing component v." },
      { term: "IM(v) — Maintainability Impact", definition: "Change-propagation ground truth — measures change-ripple reach on the transposed dependency graph." },
      { term: "IA(v) — Availability Impact", definition: "Connectivity-disruption ground truth — QoS-weighted reachability loss from removing component v." },
      { term: "IV(v) — Vulnerability Impact", definition: "Adversarial ground truth — compromise propagation reach via trusted dependency paths." },
      { term: "Reachability Loss", definition: "Fraction of node-to-node shortest paths that are broken when a component is removed." },
      { term: "Fragmentation", definition: "Percentage of components that become disconnected from the main cluster when a node is removed." },
      { term: "Throughput Loss", definition: "Estimated percentage drop in message-routing throughput due to a component's failure." },
      { term: "Flow Disruption", definition: "Fraction of event-simulation message flows interrupted by a given component's failure." },
      { term: "Cascade Probability", definition: "Likelihood that a failure at this component triggers at least one further failure in the network." },
      { term: "Cascade Reach", definition: "The number of components affected by a cascade starting at a given node." },
      { term: "Cascade Depth", definition: "Number of hops in the longest failure chain triggered by removing a component." },
      { term: "Delivery Rate", definition: "Percentage of published messages successfully delivered to at least one subscriber." },
      { term: "Drop Rate", definition: "Percentage of messages lost due to queue overflow, unreachable broker, or deadline expiry." },
      { term: "Avg Latency", definition: "Average end-to-end message delivery time in milliseconds across the simulation window." },
      { term: "p50 Latency", definition: "Median message latency — half of all messages were delivered within this time." },
      { term: "p99 Latency", definition: "99th-percentile latency — 99% of messages were delivered within this time." },
      { term: "Throughput", definition: "Number of messages successfully delivered per second during event simulation." },
      { term: "Single Failure Mode", definition: "A simulation run that injects a fault into exactly one component and measures the resulting impact." },
      { term: "Pairwise Failure", definition: "A simulation run removing two components simultaneously to capture correlated failure interactions." },
      { term: "N-k Contingency", definition: "Simultaneous removal of N components from a set of k candidates — tests resilience under multiple concurrent failures." },
      { term: "Fault Injection", definition: "The act of artificially removing a component or edge to observe what breaks downstream." },
      { term: "Fault Propagation", definition: "The spread of an initial failure through dependency edges to connected components." },
      { term: "Recovery Path", definition: "An alternative route available after a failure that allows the system to maintain partial connectivity." },
      { term: "MTBF", definition: "Mean Time Between Failures — average elapsed time between successive failures of a component." },
      { term: "MTTR", definition: "Mean Time To Recovery — average time to restore a component after a failure event." },
      { term: "Graceful Degradation", definition: "The ability of a system to maintain partial functionality when some components fail." },
      { term: "Fault Tolerance", definition: "The property that a system continues operating — perhaps with reduced capability — despite component failures." },
      { term: "Redundancy", definition: "Duplicate components or paths providing backup routes when a primary path fails." },
      { term: "Failover", definition: "Automatic re-routing of traffic to a backup component when the primary becomes unavailable." },
      { term: "Circuit Breaker", definition: "A pattern that stops sending messages to a failing component after repeated errors, preventing cascade overload." },
      { term: "Bulkhead", definition: "An isolation boundary preventing failures in one part of the system from spreading to others." },
      { term: "Backpressure", definition: "A mechanism that slows producers when consumers cannot keep up, preventing queue overflow." },
      { term: "Cascade Rule", definition: "A simulation rule governing how failures spread — e.g. LIBRARY (simultaneous blast) or PHYSICAL (co-located failure)." },
      { term: "Library Blast", definition: "Simultaneous failure of all applications that share a library dependency, triggered by the library's failure." },
      { term: "Physical Cascade", definition: "Simultaneous failure of all components co-located on a node, triggered by that node's failure." },
      { term: "Stop Condition", definition: "A rule that halts cascade propagation along an edge — e.g. loose coupling or stable interface." },
      { term: "Loose Coupling Threshold", definition: "Edge weight below which change-propagation stops during maintainability simulation. Default 0.20." },
      { term: "Stable Interface Threshold", definition: "Instability value below which change-propagation stops at a target node during maintainability simulation. Default 0.20." },
    ],
  },
  {
    key: "Validation",
    textColor: "text-green-400",
    borderColor: "border-green-500/20",
    bgColor: "bg-green-500/[0.06]",
    ringColor: "bg-green-500/10",
    badgeClass: "bg-green-500/10 text-green-400 border-green-500/20",
    filterActive: "bg-green-500/15 border-green-500/40 text-green-300",
    filterIdle: "border-green-500/30 text-green-400",
    Icon: CheckCircle2,
    terms: [
      { term: "Spearman ρ", definition: "Rank-order correlation between predicted scores and simulation ground truth. Target ≥ 0.87." },
      { term: "F1 Score", definition: "Harmonic mean of Precision and Recall for CRITICAL component classification. Target ≥ 0.90." },
      { term: "Precision", definition: "Fraction of predicted-CRITICAL components that are truly critical according to simulation." },
      { term: "Recall", definition: "Fraction of truly-critical components that were correctly predicted as CRITICAL." },
      { term: "NDCG@K", definition: "Normalised Discounted Cumulative Gain — measures ranking quality of the top-K predicted components." },
      { term: "Top-5 Overlap", definition: "Number of components appearing in both the top-5 predicted list and the top-5 simulated-impact list." },
      { term: "RMSE", definition: "Root Mean Squared Error between predicted and simulation ground-truth scores. Lower is better." },
      { term: "MAE", definition: "Mean Absolute Error between predicted scores and simulation ground truth." },
      { term: "CCR@5", definition: "Cascade Capture Rate at 5 — fraction of the top-5 cascade-critical components identified by R(v)." },
      { term: "COCR@5", definition: "Change Overlap Capture Rate at 5 — fraction of the top-5 change-impact components identified by M(v)." },
      { term: "SPOF_F1", definition: "SPOF classification F1 — how accurately A(v) identifies true network single-points-of-failure." },
      { term: "RRI", definition: "Robustness Rank Improvement — improvement over a random baseline in availability prediction ranking." },
      { term: "AHCR@5", definition: "Attack Hit Capture Rate at 5 — fraction of the top-5 adversarial targets correctly identified by V(v)." },
      { term: "FTR", definition: "False Trust Rate — fraction of components incorrectly classified as low-vulnerability." },
      { term: "APAR", definition: "Attack Path Agreement Rate — alignment between predicted vulnerability ranking and actual compromise propagation paths." },
      { term: "Weighted-κ CTA", definition: "Quadratic-weighted Cohen's kappa for Change-impact Tier Agreement between M(v) and simulation-derived tiers." },
      { term: "Kendall τ", definition: "Rank correlation measuring concordant vs discordant pairs. A complementary check alongside Spearman." },
      { term: "Cohen's Kappa", definition: "Agreement statistic corrected for chance — used for tier classification agreement." },
      { term: "AUC-ROC", definition: "Area Under the Receiver Operating Characteristic curve — measures binary classification quality independent of threshold." },
      { term: "Confusion Matrix", definition: "A table showing true positives, false positives, true negatives, and false negatives for classification results." },
      { term: "True Positive", definition: "A component correctly predicted as CRITICAL that is also CRITICAL according to simulation ground truth." },
      { term: "False Positive", definition: "A component incorrectly predicted as CRITICAL when simulation shows it is not critical." },
      { term: "True Negative", definition: "A component correctly predicted as non-CRITICAL that is also non-CRITICAL in ground truth." },
      { term: "False Negative", definition: "A component incorrectly predicted as non-CRITICAL when it is truly critical in ground truth." },
      { term: "Statistical Power", definition: "The probability that a validation test correctly detects a true relationship. Reported in power tables alongside Spearman." },
      { term: "p-value", definition: "Probability of observing the measured correlation by chance under the null hypothesis. Lower is more significant." },
      { term: "Confidence Interval", definition: "A range around the Spearman estimate indicating where the true correlation likely falls." },
      { term: "Rank Agreement", definition: "The degree to which two rankings of the same components are consistent — the core validation objective." },
      { term: "Calibration", definition: "How well predicted probability scores correspond to actual outcome frequencies." },
      { term: "Ground Truth Label", definition: "The simulation-derived impact tier assigned to a component and used as the validation reference." },
      { term: "CDCC", definition: "Cross-Dimensional Contamination Check — verifies that vulnerability scores are not contaminated by availability or reliability signals." },
      { term: "Top-K Overlap", definition: "Count of components appearing in both the top-K predicted list and the top-K ground-truth list." },
      { term: "Bottleneck Precision", definition: "Precision of M(v) in identifying true change-bottleneck components relative to simulation-derived change impact." },
    ],
  },
  {
    key: "Pub-Sub & Statistics",
    textColor: "text-amber-400",
    borderColor: "border-amber-500/20",
    bgColor: "bg-amber-500/[0.06]",
    ringColor: "bg-amber-500/10",
    badgeClass: "bg-amber-500/10 text-amber-400 border-amber-500/20",
    filterActive: "bg-amber-500/15 border-amber-500/40 text-amber-300",
    filterIdle: "border-amber-500/30 text-amber-400",
    Icon: Radio,
    terms: [
      { term: "Topic Bandwidth", definition: "Total estimated data throughput through a topic, based on publishing frequency and payload size." },
      { term: "App Balance", definition: "Ratio of publish to subscribe activity per application — indicates whether an app is producer-heavy, consumer-heavy, or balanced." },
      { term: "Topic Fanout", definition: "Number of distinct subscribers receiving messages from a topic. High fanout amplifies the impact of a topic failure." },
      { term: "Orphan Topic", definition: "A topic with publishers but no subscribers, or vice versa — a dead-end in the message flow." },
      { term: "Cross-Node Heatmap", definition: "Matrix of inter-node communication volume — off-diagonal values indicate coupled infrastructure nodes." },
      { term: "Node Load", definition: "Total message traffic routed through a physical infrastructure node per unit time." },
      { term: "Criticality I/O", definition: "Comparison of publish/subscribe activity between critical and non-critical components." },
      { term: "Library Deps", definition: "Inbound and outbound dependency counts for library components." },
      { term: "Node Density", definition: "Distribution of critical versus normal components across infrastructure nodes." },
      { term: "Segment Diversity", definition: "Variety of application and topic types within each logical system segment." },
      { term: "Bottlenecks", definition: "Components identified as structural bottlenecks based on betweenness, articulation-point status, and cascade depth." },
      { term: "QoS Weight", definition: "Operational priority weight between 0 and 1, derived from QoS settings. Higher means more critical." },
      { term: "QoS Reliability", definition: "A QoS setting controlling message delivery guarantees — RELIABLE or BEST_EFFORT." },
      { term: "QoS Durability", definition: "A QoS setting controlling whether messages are persisted for late-joining subscribers." },
      { term: "Frequency", definition: "How often a publisher sends messages, measured in Hz." },
      { term: "Deadline", definition: "Maximum allowed time between consecutive message deliveries, in milliseconds." },
      { term: "Queue Size", definition: "Maximum number of messages buffered in a topic queue before being discarded." },
      { term: "Payload Size", definition: "Size of each message in bytes. Affects bandwidth estimates but not message rate counts." },
      { term: "Publisher Count", definition: "Number of applications actively publishing messages to a topic." },
      { term: "Subscriber Count", definition: "Number of applications subscribed to receive messages from a topic." },
      { term: "Inbound Rate", definition: "Messages arriving at a broker per second from all publishers across its routed topics." },
      { term: "Outbound Rate", definition: "Fan-out deliveries leaving a broker per second — each inbound message is copied once per subscriber." },
      { term: "Fan-out Multiplier", definition: "Ratio of subscribers to publishers on a topic. High values mean one publisher drives large broker load." },
      { term: "Broker Load", definition: "Combined inbound and outbound traffic across all topics routed by a broker." },
      { term: "In (msg/s)", definition: "Inbound message rate at a broker — publisher count multiplied by publishing frequency." },
      { term: "Out (msg/s)", definition: "Outbound fan-out rate — inbound rate multiplied by subscriber count." },
      { term: "Network Bandwidth", definition: "Total estimated byte throughput across the simulation — sum of all topic bandwidths." },
      { term: "Simulation Frequency", definition: "Number of messages each publisher sends per second, applied per topic." },
      { term: "Simulation Duration", definition: "Simulated time window in seconds. Affects total message counts but not bandwidth rates." },
      { term: "Messages Published", definition: "Total messages sent by all publishers across the simulation window." },
      { term: "Messages Delivered", definition: "Total fan-out deliveries received by all subscribers — always at least equal to messages published when fanout exceeds 1." },
      { term: "Peak Topic Bandwidth", definition: "Highest single-topic bandwidth in the simulation — used as the reference for the bandwidth colour scale." },
      { term: "Segment Communication", definition: "Message flow volume between logical system segments." },
      { term: "1→N (One-to-Many)", definition: "A topic with one publisher and multiple subscribers — a broadcast communication pattern." },
      { term: "N→1 (Many-to-One)", definition: "A topic with multiple publishers and one subscriber — an aggregation communication pattern." },
      { term: "N→N (Many-to-Many)", definition: "A topic with multiple publishers and multiple subscribers — a mesh communication pattern." },
      { term: "Producers", definition: "Application nodes that publish messages to topics — sources in the message flow." },
      { term: "Consumers", definition: "Application nodes that subscribe to receive messages from topics — sinks in the message flow." },
      { term: "High I/O", definition: "Components with both significant publishing and subscribing activity — active participants in message flow." },
      { term: "Zero Activity", definition: "Components with no publish or subscribe activity detected during the analysis window." },
      { term: "At-Most-Once", definition: "A QoS delivery guarantee where messages may be dropped but never duplicated." },
      { term: "At-Least-Once", definition: "A QoS delivery guarantee where messages are retried until confirmed, but may be delivered more than once." },
      { term: "Exactly-Once", definition: "A QoS delivery guarantee ensuring each message is delivered precisely one time with no loss or duplication." },
      { term: "Durable Subscription", definition: "A subscription that persists across reconnections — the broker retains missed messages for the subscriber." },
      { term: "Transient Subscription", definition: "A subscription that only receives messages published while the subscriber is connected." },
      { term: "Message Ordering", definition: "Whether messages from a publisher arrive at subscribers in the order they were sent." },
      { term: "Retention Policy", definition: "Rules governing how long a broker keeps messages in a topic queue before discarding them." },
      { term: "Dead Letter Queue", definition: "A special queue holding messages that could not be delivered after exhausting retry attempts." },
      { term: "Replay", definition: "Re-delivering historical messages from a topic to a subscriber, used for state recovery after restart." },
      { term: "Partition", definition: "A horizontal slice of a topic's message stream, enabling parallel processing across multiple consumers." },
      { term: "Wildcard Subscription", definition: "A topic subscription pattern using wildcards to match multiple topic names simultaneously." },
      { term: "Session Affinity", definition: "Routing policy ensuring a subscriber always connects to the same broker instance for consistency." },
      { term: "Load Balancing", definition: "Distributing message traffic evenly across multiple broker instances to prevent hot spots." },
      { term: "DDS", definition: "Data Distribution Service — a publish-subscribe middleware standard commonly used in real-time systems and robotics." },
      { term: "MQTT", definition: "Message Queuing Telemetry Transport — a lightweight pub-sub protocol for IoT and constrained devices." },
      { term: "ROS2", definition: "Robot Operating System 2 — a pub-sub framework for robotics built on DDS middleware." },
      { term: "Kafka", definition: "A distributed log-based pub-sub platform optimised for high-throughput event streaming." },
      { term: "QoS Profile", definition: "A named set of QoS policies applied to a topic, governing reliability, durability, deadline, and liveliness." },
      { term: "Liveliness", definition: "A QoS attribute ensuring that a publisher signals it is still active at a defined heartbeat interval." },
      { term: "Topic Hierarchy", definition: "A structured naming scheme for topics using delimiters to group related channels by domain or function." },
      { term: "Message Schema", definition: "The structured definition of the data fields and types carried by messages on a topic." },
    ],
  },
  {
    key: "Pipeline",
    textColor: "text-indigo-400",
    borderColor: "border-indigo-500/20",
    bgColor: "bg-indigo-500/[0.06]",
    ringColor: "bg-indigo-500/10",
    badgeClass: "bg-indigo-500/10 text-indigo-400 border-indigo-500/20",
    filterActive: "bg-indigo-500/15 border-indigo-500/40 text-indigo-300",
    filterIdle: "border-indigo-500/30 text-indigo-400",
    Icon: Layers,
    terms: [
      { term: "Pipeline", definition: "The end-to-end analysis workflow: Generate → Import → Analyze → Predict → Simulate → Validate → Visualize." },
      { term: "Generate", definition: "Step 0 — produces a synthetic pub-sub topology for experiments and regression testing." },
      { term: "Import", definition: "Step 1 — converts a topology JSON into a weighted directed graph and derives DEPENDS_ON edges." },
      { term: "Analyze", definition: "Step 2 — computes deterministic RMAV scores and detects anti-patterns. Same graph always produces the same output." },
      { term: "Predict", definition: "Step 3 — optional GNN-based inference that blends learned scores with RMAV rule-based scores." },
      { term: "Simulate", definition: "Step 4 — injects faults, runs cascade and compromise simulators, and produces ground-truth impact labels." },
      { term: "Validate", definition: "Step 5 — compares predicted scores against simulation ground truth using Spearman, F1, and other metrics." },
      { term: "Visualize", definition: "Step 6 — generates interactive dashboards and static reports from analysis and validation results." },
      { term: "Topology", definition: "The structural arrangement of system components and their connections — the input to the pipeline." },
      { term: "Ground Truth", definition: "Simulation-derived impact scores used as the reference for validating predicted scores." },
      { term: "Stage", definition: "A discrete step in the pipeline, each with a dedicated CLI script, use case, and service class." },
      { term: "Artifact", definition: "A file produced by a pipeline stage — e.g. a trained model checkpoint, exported graph, or dashboard HTML." },
      { term: "Scenario", definition: "A domain-specific system topology configuration used as a benchmark or regression test case." },
      { term: "Seed", definition: "A random seed controlling synthetic graph generation. Same seed always produces the same topology." },
      { term: "Export", definition: "Serialising the in-memory or Neo4j graph to a JSON file for archiving or portability." },
      { term: "Round-Trip", definition: "Exporting and re-importing a graph to verify that no structural information is lost in serialisation." },
      { term: "CLI", definition: "Command-Line Interface — the set of scripts in cli/ for running individual pipeline stages." },
      { term: "SDK", definition: "Software Development Kit — the saag/ Python package providing programmatic access to all pipeline services." },
      { term: "UseCase", definition: "A class in saag/usecases/ that orchestrates one pipeline stage, acting as the boundary between API/CLI and service layer." },
      { term: "Repository Pattern", definition: "An abstraction over the data store — IGraphRepository — implemented by Neo4jRepository for production and MemoryRepository for tests." },
      { term: "Benchmark", definition: "A timed pipeline run across multiple graph scale presets to measure performance characteristics." },
      { term: "Run Scenarios", definition: "A batch script executing the full pipeline for all eight domain scenarios and collecting comparison results." },
      { term: "Orchestrator", definition: "The cli/run.py script that chains any combination of pipeline stages in a single invocation." },
      { term: "Config File", definition: "A YAML or JSON file specifying graph parameters, QoS profiles, or scenario overrides for a pipeline run." },
      { term: "Data File", definition: "A topology JSON in data/ that describes system components, edges, and QoS attributes." },
      { term: "Anti-Pattern Gate", definition: "A CI/CD check running detect_antipatterns.py that returns exit code 1 if HIGH issues and 2 if CRITICAL issues are found." },
    ],
  },
  {
    key: "ML & Training",
    textColor: "text-pink-400",
    borderColor: "border-pink-500/20",
    bgColor: "bg-pink-500/[0.06]",
    ringColor: "bg-pink-500/10",
    badgeClass: "bg-pink-500/10 text-pink-400 border-pink-500/20",
    filterActive: "bg-pink-500/15 border-pink-500/40 text-pink-300",
    filterIdle: "border-pink-500/30 text-pink-400",
    Icon: Brain,
    terms: [
      { term: "GNN", definition: "Graph Neural Network — learns node criticality directly from graph topology through message-passing between neighbours." },
      { term: "HeteroGAT", definition: "Heterogeneous Graph Attention Network — handles multiple node types and learns separate attention weights per edge type." },
      { term: "GAT", definition: "Graph Attention Network — uses learned attention weights to determine each neighbour's contribution." },
      { term: "Q_GNN", definition: "Criticality score predicted by the GNN through message-passing over the graph topology." },
      { term: "ensemble_alpha", definition: "Blending coefficient balancing GNN predictions versus rule-based RMAV scores in the ensemble." },
      { term: "Hidden Dim", definition: "Number of hidden features per node in each GNN layer. Larger values capture more complex patterns." },
      { term: "Attn Heads", definition: "Number of parallel attention heads in the GAT layers — more heads attend to different neighbourhood aspects." },
      { term: "GNN Layers", definition: "Number of message-passing hops. Each layer aggregates information one hop further in the graph." },
      { term: "Dropout", definition: "Fraction of neurons randomly deactivated during training to reduce overfitting." },
      { term: "Learning Rate", definition: "Step size for gradient descent during model training." },
      { term: "Early-stop Patience", definition: "Number of training epochs without validation improvement before stopping early." },
      { term: "Train Ratio", definition: "Fraction of components used for training the GNN. Remainder goes to validation and test sets." },
      { term: "Checkpoint", definition: "Saved model weights from the best training epoch — used to run inference without retraining." },
      { term: "Message Passing", definition: "The mechanism by which GNN layers aggregate information from neighbouring nodes in the graph." },
      { term: "Attention Weight", definition: "Learned scalar indicating how much a neighbour contributes to a node's updated representation." },
      { term: "Node Embedding", definition: "A dense vector representation of a component learned by the GNN from its structural context." },
      { term: "Feature Vector", definition: "A numerical representation of a node's input attributes — structural metrics — fed into the first GNN layer." },
      { term: "Readout", definition: "The final layer that maps per-node embeddings to scalar criticality scores." },
      { term: "Aggregation", definition: "The operation that combines neighbour embeddings into a single summary vector for each node." },
      { term: "Inductive Learning", definition: "Learning that generalises to unseen nodes or graphs without retraining — the GNN uses structural features, not node IDs." },
      { term: "Transductive Learning", definition: "Learning that is specific to the nodes seen during training and cannot directly generalise to new graphs." },
      { term: "Overfitting", definition: "When a model performs well on training data but poorly on unseen data because it memorised rather than generalised." },
      { term: "Loss Function", definition: "The objective minimised during training — measures how far predictions are from ground-truth labels." },
      { term: "Gradient", definition: "The derivative of the loss with respect to model parameters — used to update weights during backpropagation." },
      { term: "Backpropagation", definition: "Algorithm for computing gradients through all layers of the network by applying the chain rule." },
      { term: "Epoch", definition: "One complete pass through the training set during GNN training." },
      { term: "Validation Loss", definition: "Loss computed on the held-out validation set — used by early stopping to select the best checkpoint." },
      { term: "Hyperparameter", definition: "A configuration value — hidden dim, learning rate, dropout, heads — set before training, not learned." },
      { term: "Ensemble Blend", definition: "Combining GNN scores with RMAV rule-based scores using a weighted average to produce Q_ensemble." },
      { term: "Blending Coefficient α", definition: "The weight applied to Q_GNN in the ensemble. Typically 0.6–0.8 after training." },
      { term: "Heterogeneous Graph", definition: "A graph with multiple node and edge types — Applications, Libraries, Brokers, Topics, and Nodes all coexist." },
      { term: "Mini-Batch", definition: "A subset of training samples used for one gradient update step, enabling training on large graphs." },
      { term: "Inference", definition: "Running a trained model on new data to produce criticality predictions without further training." },
    ],
  },
  {
    key: "Layers & Infrastructure",
    textColor: "text-slate-400",
    borderColor: "border-slate-500/20",
    bgColor: "bg-slate-500/[0.06]",
    ringColor: "bg-slate-500/10",
    badgeClass: "bg-slate-500/10 text-slate-400 border-slate-500/20",
    filterActive: "bg-slate-500/15 border-slate-500/40 text-slate-300",
    filterIdle: "border-slate-500/30 text-slate-400",
    Icon: Server,
    terms: [
      { term: "Application Layer", definition: "Analysis view scoped to Application and Library nodes — focuses on software blast-radius risk." },
      { term: "Infrastructure Layer", definition: "Analysis view scoped to Node components — focuses on physical or virtual host failure risk." },
      { term: "Middleware Layer", definition: "Analysis view covering Applications, Nodes, and Brokers — captures full pub-sub communication risk." },
      { term: "System Layer", definition: "Full analysis across all component types — the most complete system-wide view." },
      { term: "Neo4j", definition: "Graph database used to persist the system topology. All nodes and edges are stored here." },
      { term: "Bolt Protocol", definition: "Binary network protocol used to communicate with Neo4j from the API server." },
      { term: "MemoryRepo", definition: "An in-memory repository used during testing. Implements the same interface as the production Neo4j repository." },
      { term: "APOC", definition: "Neo4j plugin providing advanced graph procedures used during topology import and analysis." },
      { term: "Graph Data Science", definition: "Neo4j plugin providing centrality, community detection, and path-finding algorithms." },
      { term: "OpenAPI", definition: "Machine-readable API specification describing all REST endpoints, request and response schemas." },
      { term: "Docker", definition: "A platform for packaging applications into portable containers that run consistently across environments." },
      { term: "Docker Compose", definition: "A tool for defining and running multi-service Docker applications from a single YAML file." },
      { term: "FastAPI", definition: "A high-performance Python web framework used to implement the REST API server on port 8000." },
      { term: "Uvicorn", definition: "An ASGI web server used to run the FastAPI application. Supports hot-reload during development." },
      { term: "Environment Variable", definition: "A runtime configuration value — e.g. NEO4J_URI, NEO4J_PASSWORD — passed to services at startup." },
      { term: "Port Mapping", definition: "A Docker directive forwarding traffic from the host machine to a container port — e.g. 8000:8000." },
      { term: "Volume", definition: "A Docker-managed persistent storage location mounted into a container for data that must survive restarts." },
      { term: "Healthcheck", definition: "A Docker instruction polling a container endpoint to determine whether the service is ready to accept traffic." },
      { term: "Dependency Injection", definition: "A pattern in api/dependencies.py providing services to route handlers without tight coupling to implementations." },
      { term: "Cypher", definition: "The declarative query language used to read and write data in Neo4j." },
      { term: "REST API", definition: "Representational State Transfer API — the HTTP interface exposed by the FastAPI server on port 8000." },
      { term: "Swagger UI", definition: "Interactive API documentation served at /docs — allows exploring and testing all endpoints in a browser." },
      { term: "CORS", definition: "Cross-Origin Resource Sharing — the HTTP policy configured in api/main.py to allow the Next.js frontend to call the API." },
      { term: "Presenter", definition: "A class in api/presenters/ that formats a service result into an HTTP response, decoupled from the route handler." },
    ],
  },
  {
    key: "Neo4j Schema Fields",
    textColor: "text-teal-400",
    borderColor: "border-teal-500/20",
    bgColor: "bg-teal-500/[0.06]",
    ringColor: "bg-teal-500/10",
    badgeClass: "bg-teal-500/10 text-teal-400 border-teal-500/20",
    filterActive: "bg-teal-500/15 border-teal-500/40 text-teal-300",
    filterIdle: "border-teal-500/30 text-teal-400",
    Icon: Database,
    terms: [
      // ── Shared / base ──────────────────────────────────────────────────────
      { term: "id", definition: "Unique string identifier for every graph entity. Used as the primary key in Neo4j uniqueness constraints." },
      { term: "name", definition: "Human-readable label for a graph entity — displayed throughout the dashboard and explorer." },
      { term: "weight", definition: "Computed QoS-derived priority weight in [0.01, 1.0] on any component. Higher means more operationally critical." },
      // ── Application / Library ──────────────────────────────────────────────
      { term: "app_type", definition: "Classification of an Application node — e.g. service, library, gateway. Governs which analysis rules apply." },
      { term: "criticality", definition: "Boolean flag on an Application node marking it as business-critical regardless of computed scores." },
      { term: "system_hierarchy", definition: "Nested object on Application/Library carrying the CSMS → CSS → CSCI → CSC decomposition hierarchy." },
      { term: "csms_name", definition: "Top-level system name in the system_hierarchy decomposition — the CSMS (Computer Software Management System) identifier." },
      { term: "css_name", definition: "Segment name within the system_hierarchy — the CSS (Computer Software Segment) identifier." },
      { term: "csci_name", definition: "Configuration-item name within the system_hierarchy — the CSCI (Computer Software Configuration Item) identifier." },
      { term: "csc_name", definition: "Component-group name within the system_hierarchy — the CSC (Computer Software Component) identifier." },
      { term: "code_metrics", definition: "Optional nested object on Application/Library holding raw OO source-code quality measurements grouped under size, complexity, cohesion, and coupling keys." },
      // ── code_metrics sub-fields ────────────────────────────────────────────
      { term: "cm_total_loc", definition: "Flattened Neo4j property: total lines of code for the component. Maps to code_metrics.size.total_loc." },
      { term: "cm_avg_wmc", definition: "Flattened Neo4j property: average Weighted Methods per Class. Input to cyclomatic complexity normalisation in CQP.", formula: "CQP uses CC_norm = normalise(cm_avg_wmc)" },
      { term: "cm_avg_lcom", definition: "Flattened Neo4j property: average Lack of Cohesion of Methods (SonarQube scale). Input to LCOM normalisation in CQP." },
      { term: "cm_avg_cbo", definition: "Flattened Neo4j property: average Coupling Between Objects — a class-level afferent+efferent coupling count from static analysis." },
      { term: "cm_avg_rfc", definition: "Flattened Neo4j property: average Response for a Class — number of methods potentially executed in response to a message." },
      { term: "cm_avg_fanin", definition: "Flattened Neo4j property: average afferent coupling fan-in from internal static analysis. Used for Library instability." },
      { term: "cm_avg_fanout", definition: "Flattened Neo4j property: average efferent coupling fan-out from internal static analysis. Used for Library instability." },
      // ── Topic ──────────────────────────────────────────────────────────────
      { term: "size", definition: "Payload size of a Topic's messages in bytes. Feeds into the size_norm component of the topic weight formula.", formula: "size_norm = min(log₂(1 + size_kb) / 50, 1.0)" },
      { term: "qos_reliability", definition: "QoS policy on a Topic: RELIABLE (score 1.0) or BEST_EFFORT (score 0.0). The primary determinant of topic weight." },
      { term: "qos_durability", definition: "QoS policy on a Topic: PERSISTENT (1.0), TRANSIENT (0.6), TRANSIENT_LOCAL (0.5), or VOLATILE (0.0)." },
      { term: "qos_transport_priority", definition: "QoS policy on a Topic: HIGHEST/CRITICAL/URGENT (1.0), HIGH (0.66), MEDIUM (0.33), LOW (0.0). Contributes 30% of QoS score." },
      { term: "subscriber_count", definition: "Computed Topic property: number of distinct Application/Library nodes with a SUBSCRIBES_TO edge. Written after Phase 2." },
      { term: "publisher_count", definition: "Computed Topic property: number of distinct Application/Library nodes with a PUBLISHES_TO edge. Written after Phase 2." },
      // ── DEPENDS_ON edge ────────────────────────────────────────────────────
      { term: "dependency_type", definition: "Property on every DEPENDS_ON edge naming the derivation rule that created it: app_to_app, app_to_lib, app_to_broker, node_to_node, node_to_broker, or broker_to_broker." },
      // ── Structural edge types ──────────────────────────────────────────────
      { term: "PUBLISHES_TO", definition: "Structural edge from Application/Library to Topic. Means this component sends messages to that topic." },
      { term: "SUBSCRIBES_TO", definition: "Structural edge from Application/Library to Topic. Means this component receives messages from that topic." },
      { term: "ROUTES", definition: "Structural edge from Broker to Topic. Means this broker is responsible for routing messages on that topic." },
      { term: "RUNS_ON", definition: "Structural edge from Application/Broker to Node. Means this component is deployed on that infrastructure host." },
      { term: "CONNECTS_TO", definition: "Structural edge between two Nodes representing direct network connectivity between infrastructure hosts." },
      { term: "USES", definition: "Structural edge from Application/Library to Library. Means this component depends on that shared code module." },
      { term: "DEPENDS_ON", definition: "Derived dependency edge between any two components. Points from dependent to dependency and carries weight and dependency_type." },
      // ── Constraint / schema ────────────────────────────────────────────────
      { term: "Uniqueness Constraint", definition: "A Neo4j schema constraint enforcing that no two nodes of the same label share the same id value. Created for all five vertex labels." },
      { term: "MERGE", definition: "Cypher keyword used during import to create a node or edge only if it does not already exist — preventing duplicates on re-import." },
      { term: "ON CREATE SET", definition: "Cypher clause that sets properties only when a MERGE creates a new node or edge — used for initial weight assignment." },
      { term: "ON MATCH SET", definition: "Cypher clause that updates properties when a MERGE matches an existing node or edge — used for max-preserving weight updates." },
      { term: "UNWIND", definition: "Cypher keyword for batching list parameters. All bulk imports use UNWIND $rows to process entity arrays efficiently." },
    ],
  },
]

const TOTAL_TERMS = SECTIONS.reduce((n, s) => n + s.terms.length, 0)

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function DictionaryPage() {
  const [search, setSearch] = useState("")
  const [active, setActive] = useState<SectionKey | "All">("All")

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    return SECTIONS.map(sec => ({
      ...sec,
      terms: sec.terms.filter(t =>
        (active === "All" || active === sec.key) &&
        (!q || t.term.toLowerCase().includes(q) || t.definition.toLowerCase().includes(q))
      ),
    })).filter(sec => sec.terms.length > 0)
  }, [search, active])

  return (
    <AppLayout title="Glossary" description="Definitions of terms used across the platform">
      <div className="space-y-5">

        {/* ── Search ──────────────────────────────────────────────────── */}
        <div className="relative max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search terms…"
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="pl-9 h-9"
          />
        </div>

        {/* ── Category filters ────────────────────────────────────────── */}
        <div className="flex flex-wrap gap-1.5">
          <button
            onClick={() => setActive("All")}
            className={`px-3 py-1 text-xs rounded-full border transition-colors font-medium ${
              active === "All"
                ? "bg-foreground/10 border-foreground/30 text-foreground"
                : "border-border text-muted-foreground hover:text-foreground hover:border-foreground/20"
            }`}
          >
            All
          </button>
          {SECTIONS.map(sec => (
            <button
              key={sec.key}
              onClick={() => setActive(active === sec.key ? "All" : sec.key)}
              className={`px-3 py-1 text-xs rounded-full border transition-colors font-medium bg-transparent ${
                active === sec.key ? sec.filterActive : `${sec.filterIdle} hover:opacity-80`
              }`}
            >
              {sec.key}
            </button>
          ))}
        </div>

        {/* ── Sections ────────────────────────────────────────────────── */}
        {filtered.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-slate-700 bg-slate-900/30 py-16 text-center">
            <BookMarked className="h-8 w-8 text-slate-500" />
            <p className="text-sm font-medium text-slate-300">No terms match your search</p>
            <p className="text-xs text-muted-foreground">Try a different keyword or clear the filter</p>
          </div>
        ) : (
          <div className="space-y-4">
            {filtered.map(sec => (
              <Card key={sec.key} className="bg-background">
                <CardHeader className="pb-2 flex flex-row items-center justify-between space-y-0">
                  <div className="flex items-center gap-2">
                    <div className={`rounded-lg ${sec.ringColor} p-1.5`}>
                      <sec.Icon className={`h-3.5 w-3.5 ${sec.textColor}`} />
                    </div>
                    <CardTitle className="text-[11px] text-muted-foreground uppercase tracking-widest">
                      {sec.key}
                    </CardTitle>
                  </div>
                  <Badge className={`${sec.badgeClass} text-[10px] px-2 py-0.5`}>
                    {sec.terms.length} {sec.terms.length === 1 ? "term" : "terms"}
                  </Badge>
                </CardHeader>
                <CardContent className="pt-0">
                  <div className="divide-y divide-border">
                    {sec.terms.map(term => (
                      <div key={term.term} className="flex gap-3 py-2.5 first:pt-0 last:pb-0">
                        <span className={`shrink-0 w-44 text-xs font-mono font-medium ${sec.textColor} leading-relaxed`}>
                          {term.term}
                        </span>
                        <span className="text-xs text-muted-foreground leading-relaxed">
                          {term.definition}
                          {term.formula && (
                            <code className="ml-2 text-[10px] font-mono bg-muted/60 rounded px-1.5 py-0.5 text-muted-foreground/80">
                              {term.formula}
                            </code>
                          )}
                        </span>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}

      </div>
    </AppLayout>
  )
}
