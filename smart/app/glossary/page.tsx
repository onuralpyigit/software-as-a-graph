"use client"

import { useState, useMemo, useRef, useEffect } from "react"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import { BookMarked, Search, Hash } from "lucide-react"

// ─── Types ────────────────────────────────────────────────────────────────────

interface Term {
  term: string
  definition: string
  formula?: string
}

interface Abbreviation {
  abbr: string
  expansion: string
  definition: string
}

// ─── Terms (Dashboard · Explorer · Statistics) ────────────────────────────────

const ALL_TERMS: Term[] = [
  // A
  { term: "Active Cell %", definition: "Fraction of all node-pair or segment-pair heatmap cells that have at least one message flow. High values mean tightly coupled physical infrastructure or subsystems.", formula: "nonzero_cells / total_cells × 100" },
  { term: "Active Cells", definition: "Heatmap matrix cells with non-zero inter-entity message traffic. High counts indicate tightly coupled physical hosts or segments." },
  { term: "Active Entities", definition: "Number of libraries (or apps) that have at least one dependency edge in the current analysis. Inactive entities are excluded from in-degree and out-degree calculations." },
  { term: "ap_c_directed", definition: "Directional articulation-point score — the maximum of the removal-impact scores computed in the outbound and inbound directions. Used as an input to the Bottleneck Score formula." },
  { term: "App Balance", definition: "Pub/sub ratio per application — reveals pure producers, pure consumers, and high-I/O hubs. Displayed in the Statistics App Balance tab.", formula: "I/O load = pub_count + sub_count" },
  { term: "app_type", definition: "Classification of an application's functional role (e.g., sensor, actuator, monitor, controller). Stored as a property on Application nodes in Neo4j; shown in bar-chart tooltip enrichment in Statistics." },
  { term: "Application", definition: "A software process that publishes or subscribes to topics. The primary actor in a pub-sub system. Shown as a node type in the Explorer." },
  { term: "Articulation Point", definition: "A node whose removal increases the number of disconnected graph components. A structural single-point-of-failure; flagged 🔴 in the Bottlenecks table." },
  { term: "Avg In-Degree", definition: "Mean number of applications consuming each library. Higher averages indicate more broadly shared libraries with larger collective blast radius.", formula: "mean(in-degree) across active libraries" },
  { term: "Avg Node Load", definition: "Average number of topic connections (publishes + subscribes) hosted on a single infrastructure node.", formula: "load(node) = Σ pub_count + Σ sub_count for hosted apps" },
  { term: "Avg Apps per Segment", definition: "Mean number of applications per segment. Very low values may indicate overly fragmented subsystems; very high values may signal monolithic segments.", formula: "total_apps / segment_count" },
  { term: "Avg I/O per Segment", definition: "Average combined publish + subscribe message load aggregated per segment. High-I/O segments are communication hubs whose degradation has the widest downstream reach.", formula: "mean(Σ pub + Σ sub per segment)" },
  { term: "Avg Topics per Segment", definition: "Mean number of topics owned or used per segment. Reflects how much communication surface each subsystem exposes to the rest of the architecture.", formula: "total_topics_touched / segment_count" },
  { term: "Avg Subscriber Bandwidth", definition: "Mean subscriber-side bandwidth per active topic. Measures how many bytes flow to all consumers per publish event.", formula: "bandwidth_sub = message_size × sub_count" },
  { term: "Analysis Layer", definition: "The view-scope filter applied to the graph in the Analysis page. Four layers: app (Application + Library nodes), infra (Node edges), mw (Broker edges), system (all component types). Determines which DEPENDS_ON subtypes are included in RMAS scoring." },
  { term: "Anti-Pattern", definition: "An architectural smell detected by the AntiPatternDetector after RMAS scoring. Severity levels: CRITICAL, HIGH, MEDIUM. Types: SPOF, BRIDGE_EDGE, BOTTLENECK_EDGE, BROKER_OVERLOAD, FAILURE_HUB, CONCENTRATION_RISK, SYSTEMIC_RISK, DEEP_PIPELINE, TOPIC_FANOUT, QOS_MISMATCH, GOD_COMPONENT, HUB_AND_SPOKE, CHATTY_PAIR, ORPHANED_TOPIC, UNSTABLE_INTERFACE, TARGET, EXPOSURE, CYCLE, CHAIN, ISOLATED, COMPOUND_RISK." },
  { term: "Availability Score A(v)", definition: "RMAS Availability dimension score measuring the structural impact of removing component v on graph connectivity. The dominant dimension in Q(v) with AHP weight 0.43.", formula: "A(v) = 0.35·AP_c_directed + 0.25·QSPOF + 0.25·BR + 0.10·CDI + 0.05·w(v)" },
  // B
  { term: "Bandwidth", definition: "Total estimated byte throughput for a topic per publish cycle, derived from message size and the number of connected subscribers or publishers." },
  { term: "Bandwidth Outliers", definition: "Topics whose bandwidth exceeds the IQR upper fence. A small number of such topics can dominate total network utilisation.", formula: "outlier if bandwidth > Q3 + 1.5 × IQR" },
  { term: "Betweenness", definition: "Fraction of all shortest paths in the graph that pass through a component. High values identify structural bottlenecks; used as an input to the Bottleneck Score." },
  { term: "Blast Radius", definition: "The number of components reachable from a given component through dependency chains — a measure of potential failure impact." },
  { term: "Blast Radius (Normalized)", definition: "Blast radius scaled to [0, 1] relative to the maximum observed across the system. Used as an input to the Bottleneck Score." },
  { term: "Bottleneck Score", definition: "Composite structural score combining betweenness, SPOF severity (ap_c_directed), blast radius, and bridge ratio. Higher scores identify the most critical single points of structural failure.", formula: "Score = w_bt×betweenness + w_ap×ap_c_directed + w_br×blast_radius_norm + w_bridge×bridge_ratio" },
  { term: "Bottlenecks", definition: "Components identified as lying on the most critical structural paths. Shown in the Statistics Bottlenecks tab with a ranked table and configurable formula weights." },
  { term: "Bridge", definition: "An edge whose removal disconnects two previously connected parts of the graph — a structural weak link." },
  { term: "Bridge Ratio", definition: "Fraction of a component's incident edges that are bridges. Higher values mean most connections are single-path with no redundancy.", formula: "bridge_ratio = bridge_edges / total_incident_edges" },
  { term: "Broadcast Topic", definition: "A topic with exactly one publisher and multiple subscribers (1→N pattern). A publisher failure on this topic simultaneously silences all consumers." },
  { term: "Broker", definition: "A routing node that relays messages between publishers and subscribers across named topic channels. Shown as a node type in the Explorer." },
  { term: "Broker Load", definition: "Sum of inbound + outbound traffic across all topics routed by a broker. A topic routed by multiple brokers is counted in full for each — each broker independently carries the full load for every topic it routes. Shown in the Traffic Simulator Broker Usage tab." },
  { term: "Broker Type", definition: "Classification of broker middleware (e.g., DDS, MQTT, custom). Stored as the type property on Broker nodes in Neo4j. Identifies the routing protocol family used by this broker." },
  { term: "bugs", definition: "Number of bugs detected by static analysis (e.g., SonarQube). An optional property on Application and Library nodes in Neo4j; contributes to maintainability risk assessment when present." },
  // C
  { term: "Cascade Depth", definition: "Number of hops in the longest failure chain triggered by removing a component. Used as an input to the Bottleneck Score." },
  { term: "Centrality", definition: "A family of metrics measuring how important or influential a node is based on its position in the graph. Includes betweenness, closeness, and eigenvector variants. Linked from the Dashboard statistics card." },
  { term: "Clustering", definition: "The likelihood that a node's neighbours are also directly connected to one another — a measure of local path redundancy. Linked from the Dashboard statistics card." },
  { term: "cm_avg_cbo", definition: "Average Coupling Between Objects per class in the component. Measures how many external classes each class depends on. High values increase change propagation risk and coupling instability. Stored on Application and Library nodes." },
  { term: "cm_avg_fanin", definition: "Average number of other classes that reference each class in the component (afferent coupling). High values indicate the class is a widely shared internal dependency. Stored on Application and Library nodes." },
  { term: "cm_avg_fanout", definition: "Average number of external classes each class in the component calls or depends on (efferent coupling). High values signal tightly coupled, hard-to-isolate code. Stored on Application and Library nodes." },
  { term: "cm_avg_lcom", definition: "Average Lack of Cohesion of Methods per class. Low cohesion means classes contain loosely related responsibilities. Input to the Code Quality Penalty (CQP) in M(v). Stored on Application and Library nodes." },
  { term: "cm_avg_rfc", definition: "Average Response for a Class — total number of methods callable by instances of each class. High values raise testing effort and maintenance cost. Stored on Application and Library nodes." },
  { term: "cm_avg_wmc", definition: "Average Weighted Methods per Class — average cyclomatic complexity across all classes in the component. Higher values indicate more complex, harder-to-maintain classes. Input to M(v) via CQP.", formula: "cm_avg_wmc = mean(cyclomatic_complexity per class)" },
  { term: "cm_max_cbo", definition: "Maximum Coupling Between Objects value across all classes in the component. Identifies the most tightly-coupled class and the worst-case coupling risk. Stored on Application and Library nodes." },
  { term: "cm_max_fanin", definition: "Maximum afferent coupling value across all classes in the component. Identifies the most-referenced internal class; a hot-spot for unintended shared-state coupling." },
  { term: "cm_max_fanout", definition: "Maximum efferent coupling value across all classes in the component. Identifies the class with the highest external dependency count." },
  { term: "cm_max_lcom", definition: "Maximum Lack of Cohesion of Methods across all classes in the component. Represents the worst-cohesion class — the class most in need of refactoring into focused responsibilities." },
  { term: "cm_max_rfc", definition: "Maximum Response for a Class value across all classes in the component. Identifies the class with the largest interface surface area." },
  { term: "cm_max_wmc", definition: "Maximum Weighted Methods per Class value across all classes. Identifies the most complex individual class in the component." },
  { term: "cm_total_classes", definition: "Total number of classes in the component. Used to contextualise WMC, LCOM, CBO, and RFC averages when comparing components of different structural sizes." },
  { term: "cm_total_fields", definition: "Total number of field (member variable) declarations across all classes in the component. A rough proxy for data complexity and encapsulation surface." },
  { term: "cm_total_loc", definition: "Total lines of code across all source files in the component as measured by static analysis. A scope-of-change proxy for maintainability and review effort." },
  { term: "cm_total_methods", definition: "Total number of method declarations across all classes in the component. Scales the testing and maintenance effort required." },
  { term: "cm_total_wmc", definition: "Sum of cyclomatic complexity across all methods in the component (total Weighted Methods per Class). Aggregate code-complexity for the entire module.", formula: "cm_total_wmc = Σ wmc per class" },
  { term: "Coefficient of Variation", definition: "Standard deviation of node load divided by the mean, expressed as a percentage. High values indicate uneven workload distribution across infrastructure nodes.", formula: "CV = std(load) / mean(load) × 100" },
  { term: "Component", definition: "Any named unit in the system graph: Application, Library, Broker, Topic, or Node. Used generically across Dashboard, Explorer, and Statistics." },
  { term: "Component Distribution", definition: "Breakdown of all graph nodes by component type — the donut chart on the Dashboard." },
  { term: "Config Item", definition: "A named unit of software configuration within a domain, corresponding to the CSCI level in the system hierarchy." },
  { term: "Connected Components", definition: "Maximal subgraphs where every node is reachable from every other when ignoring edge direction. Values above 1 indicate a fragmented topology." },
  { term: "Consumer-Only App", definition: "An application that subscribes above the system mean but publishes below it. Upstream failures cascade directly to these endpoints.", formula: "pub ≤ avg_pub AND sub > avg_sub" },
  { term: "Consumers", definition: "Application nodes that subscribe to receive messages from topics — sinks in the message flow. Shown in Statistics App Balance." },
  { term: "coupling_afferent", definition: "Afferent coupling Ca: number of external classes that depend on classes inside this component. High Ca means changes here propagate outward to many consumers. Stored on Application and Library nodes in Neo4j.", formula: "Ca = count(external classes that depend on this component)" },
  { term: "coupling_efferent", definition: "Efferent coupling Ce: number of external classes this component's classes depend on. High Ce indicates instability — many outside changes can break this component.", formula: "Ce = count(external classes this component depends on)" },
  { term: "cpu_cores", definition: "Number of CPU cores on an infrastructure Node. Stored as a property on Node vertices in Neo4j to model compute capacity in workload distribution analysis." },
  { term: "criticality", definition: "Mission-criticality flag on Application and Library nodes (e.g., 'critical'). Drives the Criticality I/O and Node Density Statistics tabs and feeds the RMAS scoring pipeline. Blank values are treated as non-critical." },
  { term: "Critical App Fraction", definition: "Share of applications flagged as mission-critical, expressed as a percentage. High values reduce fault-tolerance margin.", formula: "critical_apps / total_apps × 100" },
  { term: "Critical Avg I/O", definition: "Average number of pub/sub connections per critical application. High values compound failure impact — more dependencies are at risk when a critical component goes down.", formula: "mean(pub + sub) for critical apps" },
  { term: "Critical vs Normal I/O", definition: "Ratio of average I/O load for critical apps compared to normal apps. Values above 1× mean critical components are also the communication hotspots.", formula: "mean_io(critical) / mean_io(normal)" },
  { term: "Criticality I/O", definition: "Comparison of publish/subscribe activity between critical and non-critical applications. Shown in the Statistics Criticality I/O tab." },
  { term: "Cross-Node Heatmap", definition: "Matrix showing inter-node message flow volume. Off-diagonal cells indicate coupled infrastructure hosts. Shown in the Statistics Cross-Node tab." },
  { term: "Cross-Node Events", definition: "Topics that cross node boundaries — the sum of all off-diagonal entries in the cross-node heatmap. High counts raise network dependency risk and widen the blast radius of any single node failure.", formula: "Σ matrix[i][j] for i ≠ j" },
  { term: "Cross-Segment Events", definition: "Topic flows that cross segment boundaries — the sum of all off-diagonal entries in the segment-comm heatmap. High values indicate interdependency between subsystems and wider cascade paths under failure.", formula: "Σ matrix[i][j] for i ≠ j" },
  { term: "cyclomatic_complexity", definition: "McCabe cyclomatic complexity of the component's code — the number of linearly-independent execution paths. Higher values correlate with more defect-prone, harder-to-test code. Input to the Code Quality Penalty in M(v). Stored on Application and Library nodes." },
  { term: "CONNECTS_TO", definition: "Structural edge between two infrastructure Nodes representing a direct network link. Direction is Node → Node. Simulation uses this edge to detect when a network partition isolates a host and all components running on it.", formula: "(Node) -[:CONNECTS_TO]→ (Node)" },
  { term: "Cascade Depth Potential", definition: "Estimated maximum cascade depth for a component — the longest failure chain it can initiate. Enhanced variant multiplies by (1 + MPCI) to account for parallel cascade paths. Input to R(v).", formula: "CDPot_enh = CDPot_base × (1 + MPCI)" },
  { term: "Classification Level", definition: "Five-tier severity classification applied to each RMAS dimension score: CRITICAL (> Q3 + 0.75×IQR), HIGH (> Q3), MEDIUM (> Median), LOW (> Q1), MINIMAL (≤ Q1). For samples < 12 a fixed-percentile fallback is used." },
  { term: "Connectivity Degradation Index", definition: "Normalised increase in average shortest-path length after removing a component. Measures how much the remaining graph's routing efficiency degrades. Input to A(v).", formula: "CDI = (avg_path_after - avg_path_before) / avg_path_before" },
  // D
  { term: "Degree", definition: "Total number of edges connected to a node — a basic connectivity measure. Linked from the Dashboard statistics card." },
  { term: "DEPENDS_ON", definition: "Derived directed edge from a dependent component to the component it relies on. Never imported from the topology JSON — always inferred by the six derivation rules (app_to_app, app_to_broker, app_to_lib, node_to_node, node_to_broker, broker_to_broker) and carries weight and path_count properties.", formula: "(dependent) -[:DEPENDS_ON {weight, path_count, dependency_type}]→ (dependency)" },
  { term: "dependency_type", definition: "Property on each DEPENDS_ON edge classifying how the dependency was derived. One of six values: app_to_app, app_to_broker, app_to_lib, node_to_node, node_to_broker, broker_to_broker. Used to filter the graph by analysis layer (app, infra, mw, system)." },
  { term: "app_to_app", definition: "DEPENDS_ON subtype: subscriber App/Lib → publisher App/Lib, inferred via a shared topic. The foundational pub-sub dependency rule (Rule 1). Weight = max topic weight over all shared topics.", formula: "App_sub →[…Topic…]→ App_pub" },
  { term: "app_to_broker", definition: "DEPENDS_ON subtype: App/Lib → Broker that routes the topics it uses. Captures middleware-layer dependency (Rule 2). Weight = max topic weight over all routed topics.", formula: "App →[…Topic…ROUTES…]→ Broker" },
  { term: "app_to_lib", definition: "DEPENDS_ON subtype: App/Lib → Library connected via a USES edge (Rule 5). Models simultaneous blast semantics: a library failure hits all consuming apps at once, not sequentially. Weight = consuming app's weight." },
  { term: "broker_to_broker", definition: "DEPENDS_ON subtype: bidirectional colocation edge between two Brokers that share a physical Node (Rule 6). Models shared-fate risk — a node failure takes both brokers down simultaneously. Weight = hosting node's weight." },
  { term: "node_to_broker", definition: "DEPENDS_ON subtype: Node → Broker, lifted from app_to_broker edges on hosted apps (Rule 4). Captures the infrastructure-to-middleware dependency at the node level. Weight = max lifted edge weight." },
  { term: "node_to_node", definition: "DEPENDS_ON subtype: Node_B → Node_A, lifted from app_to_app and app_to_broker edges when hosted apps share those dependencies (Rule 3). Used in the infrastructure analysis layer." },
  { term: "Dependency Distribution", definition: "Breakdown of DEPENDS_ON edges by derivation subtype — the horizontal bar chart on the Dashboard." },
  { term: "Directed AP", definition: "A component that disconnects the directed reachable set when removed. Flagged 🟠 in the Bottlenecks table. May not appear in undirected articulation-point analysis." },
  { term: "Domain Communication", definition: "Message flow volume between architectural segments. High cross-segment traffic signals tight subsystem coupling. Shown in the Statistics Segment Comm tab." },
  { term: "duplicated_lines_density", definition: "Percentage of duplicated source lines detected by static analysis (e.g., SonarQube). High values indicate copy-paste coupling that amplifies change propagation when defects are found. Stored on Application and Library nodes." },
  // E
  { term: "Edge Types", definition: "Classification of graph connections. Includes six DEPENDS_ON subtypes and six structural relationship types. Shown on the Dashboard as a KPI tile." },
  // F
  { term: "Fan-out Multiplier", definition: "Ratio of subscriber count to publisher count per topic. A high multiplier (e.g. 10×) means one publisher drives significant broker outbound load. The delivered count in the summary reflects cumulative fan-out across all selected topics.", formula: "subscriber_count / publisher_count" },
  { term: "Fanout", definition: "The product of publisher count and subscriber count for a topic. High fanout amplifies the blast radius of a publisher failure.", formula: "fanout = pub_count × sub_count" },
  { term: "Flags", definition: "Visual indicators in the Bottlenecks table. 🔴 marks undirected articulation points, 🟠 marks directed articulation points, ⚡ marks score outliers." },
  // G
  { term: "GOD_COMPONENT", definition: "Anti-pattern flagged when M(v) ≥ CRITICAL and betweenness > 0.3. A single component acting as the primary structural bottleneck and hardest-to-change module. CRITICAL severity." },
  { term: "Graph Density", definition: "Ratio of actual edges to the maximum possible edges in the graph. Higher density means more interconnection. Shown as a KPI tile on the Dashboard." },
  // H
  { term: "High-Dependency Outliers", definition: "Libraries whose in-degree (number of consuming applications) is statistically extreme relative to the rest of the library population. Shown in the outlier table of the Statistics Library Deps tab." },
  { term: "High I/O App", definition: "An application that both publishes and subscribes above the system mean. A communication hub whose failure disrupts both upstream and downstream flows.", formula: "pub > avg_pub AND sub > avg_sub" },
  { term: "host", definition: "Hostname or network address where a Broker is deployed. Stored as the host property on Broker nodes in Neo4j; used in physical-layer analysis and heatmap drill-down enrichment." },
  // I
  { term: "Idle Nodes", definition: "Physical nodes hosting no communicating applications. May indicate orphaned infrastructure, pure compute nodes, or deployment imbalances.", formula: "count(nodes where load = 0)" },
  { term: "In (msg/s)", definition: "Inbound message rate arriving at the broker per topic: publisher_count × frequency_hz. Total messages over the window = In × duration_sec. Shown in the Traffic Simulator per-topic table." },
  { term: "In-Degree", definition: "Number of incoming edges to a component — how many other components directly depend on this node. Shown in Statistics Library Deps." },
  { term: "Inbound Rate", definition: "Messages arriving at a broker per second from all publishers across its routed topics. Formula: Σ(publisher_count × Hz) over all routed topics. Shown in the Broker Usage panel of the Traffic Simulator." },
  { term: "Inter-Entity Traffic", definition: "Message traffic flowing between different entities in the cross-node or segment communication heatmap — the off-diagonal cells." },
  { term: "Intra-Entity Traffic", definition: "Message traffic within the same entity in the heatmap — the diagonal cells representing self-routing." },
  { term: "Intra-Node Events", definition: "Topics where both publisher and subscriber run on the same physical node — the diagonal sum of the cross-node heatmap. This local traffic is unaffected by inter-node network failures.", formula: "Σ matrix[i][i]  (diagonal sum)" },
  { term: "Intra-Segment Events", definition: "Topic exchanges where publisher and subscriber belong to the same segment — the diagonal sum of the segment-comm heatmap. High intra-segment traffic signals well-encapsulated, loosely coupled subsystems.", formula: "Σ matrix[i][i]  (diagonal sum)" },
  { term: "Impact Score I(v)", definition: "Overall ground-truth impact of removing component v from the system, computed by the failure simulator.", formula: "I(v) = 0.35·reachability_loss + 0.25·fragmentation + 0.25·throughput_loss + 0.15·flow_disruption" },
  { term: "IA(v)", definition: "Availability ground-truth impact — QoS-weighted connectivity disruption from removing component v. Orthogonal to cascade-propagation (IR(v)).", formula: "IA(v) = 0.50·WeightedReachabilityLoss + 0.35·WeightedFragmentation + 0.15·PathBreakingThroughputLoss" },
  { term: "IM(v)", definition: "Maintainability ground-truth impact — change-propagation reach when v changes, computed via BFS on the transposed DEPENDS_ON graph G^T.", formula: "IM(v) = 0.45·ChangeReach + 0.35·WeightedChangeImpact + 0.20·NormalizedChangeDepth" },
  { term: "ip_address", definition: "IPv4 or IPv6 address of an infrastructure Node. Stored as a property on Node vertices in Neo4j to support physical-topology analysis and heatmap grouping by network location." },
  { term: "IR(v)", definition: "Reliability ground-truth impact — cascade failure reach and depth when component v fails. The simulation label consumed by the GNN Predict stage.", formula: "IR(v) = 0.45·CascadeReach + 0.35·WeightedCascadeImpact + 0.20·NormalizedCascadeDepth" },
  { term: "IS(v)", definition: "Security ground-truth impact — adversarial compromise propagation from v via trust-based dependency edges (θ_trust = 0.30).", formula: "IS(v) = 0.40·AttackReach + 0.35·WeightedAttackImpact + 0.25·HighValueContamination" },
  // L
  { term: "lcom", definition: "Lack of Cohesion of Methods scalar as reported by SonarQube static analysis. Values near 1 indicate poor cohesion; a single-class shorthand that feeds the Code Quality Penalty in M(v). Compare cm_avg_lcom (population average across all classes)." },
  { term: "Library", definition: "A shared software dependency used by one or more applications. Its failure propagates to all consumers simultaneously. Shown as a node type in the Explorer." },
  { term: "Library Deps", definition: "Inbound and outbound dependency counts for library components. High in-degree means a shared-fate risk. Shown in the Statistics Library Deps tab." },
  { term: "Load Variation", definition: "Coefficient of variation of node load. Measures how evenly workload is distributed across physical hosts. High values mean some nodes carry disproportionate traffic." },
  { term: "loc", definition: "Lines of Code as reported by SonarQube static analysis. May differ from cm_total_loc due to SonarQube's blank-line and comment exclusion rules. Input to the Code Quality Penalty in M(v). Stored on Application and Library nodes." },
  { term: "Low Activity", definition: "Applications with both publish and subscribe counts at or below the system mean. Neither heavy producers nor heavy consumers; they contribute little to overall message flow." },
  // M
  { term: "Maintainability Score M(v)", definition: "RMAS Maintainability dimension score measuring change-propagation risk and structural bottleneck position. AHP weight in Q(v): 0.17.", formula: "M(v) = 0.35·BT + 0.30·w_out + 0.15·CQP + 0.12·CouplingRisk + 0.08·(1−CC)" },
  { term: "max_connections", definition: "Maximum number of simultaneous client connections a Broker supports. Stored as a property on Broker nodes in Neo4j; may bound system throughput under high topic-fanout load." },
  { term: "Max Critical Apps per Node", definition: "Highest concentration of critical applications on a single physical node. A node with many critical apps is a blast-radius hotspot — losing it collapses multiple mission-critical flows.", formula: "max(critical_count per node)" },
  { term: "Max Fanout", definition: "Highest message multiplication factor in the system. One publish event on this topic is delivered to this many subscribers simultaneously." },
  { term: "Max Library In-Degree", definition: "The highest number of applications depending on a single library. A shared-fate risk — its failure or API change affects all dependents simultaneously." },
  { term: "Max Score", definition: "Highest composite Bottleneck Score in the system. Values above 0.5 indicate a severe structural single-point of failure whose removal would disrupt the largest fraction of the system." },
  { term: "memory_gb", definition: "Total RAM in gigabytes on an infrastructure Node. Stored as a property on Node vertices in Neo4j to model resource capacity in Node Density and workload analysis." },
  { term: "Message Size", definition: "Payload size per message in bytes. Sourced from the topic's size property in the graph; falls back to 1 024 B when absent. Affects bandwidth calculations only — not message rates." },
  { term: "Messages Delivered", definition: "Total fan-out deliveries received by all subscribers: Σ(published_per_topic × subscriber_count). Always ≥ Messages Published when any topic has multiple subscribers.", formula: "Σ(published_per_topic × subscriber_count)" },
  { term: "Messages Published", definition: "Total messages sent across all publishers during the simulation window: Σ(publisher_count × frequency_hz × duration_sec). This is the inbound side only.", formula: "Σ(publisher_count × Hz × duration_sec)" },
  // N
  { term: "Network Bandwidth", definition: "Total estimated byte throughput across the entire simulation: sum of all topic bandwidths. Each topic = (inbound + outbound) × message_size_bytes. Colour scale: green (<20% max) · yellow (20–50%) · orange (50–80%) · red (>80% max)." },
  { term: "Network Utilisation", definition: "Ratio of simulated total bandwidth to the configured network capacity, expressed as a percentage. Shown as a gauge on the Traffic Simulator results page. Colour-coded: green (<60%), orange (60–85%), red (>85%)." },
  { term: "Node", definition: "A physical or virtual infrastructure host on which applications and brokers are deployed. Shown as a node type in the Explorer." },
  { term: "Node Density", definition: "Distribution of critical versus normal applications across physical infrastructure nodes. Shown in the Statistics Node Density tab." },
  { term: "Node Load", definition: "Total message traffic (publish + subscribe connections) routed through a physical node. Shown in the Statistics Node Load tab." },
  { term: "Node Types", definition: "The five component types in the system graph: Application, Broker, Library, Topic, and Node. Shown as a KPI tile on the Dashboard." },
  { term: "Nodes Without Critical Apps", definition: "Physical nodes hosting no critical applications. These nodes have inherently lower individual failure impact on mission-critical system behaviour.", formula: "count(nodes where critical_count = 0)" },
  { term: "Normal Avg I/O", definition: "Average number of pub/sub connections per non-critical application. Used as the baseline denominator in the Crit/Normal Ratio metric.", formula: "mean(pub + sub) for non-critical apps" },
  // O
  { term: "Orphan Topic", definition: "A topic missing either a publisher or a subscriber — a dead-end in the message flow. Counted in the Statistics Topic Fanout tab.", formula: "count(pub_count = 0 OR sub_count = 0)" },
  { term: "os_type", definition: "Operating system type running on an infrastructure Node (e.g., Linux, Windows, QNX). Stored as a property on Node vertices in Neo4j for physical-layer segmentation and capacity analysis." },
  { term: "Out (msg/s)", definition: "Outbound fan-out rate leaving the broker per topic: In (msg/s) × subscriber_count. The broker delivers one copy to every subscriber. Shown in the Traffic Simulator per-topic table." },
  { term: "Out-Degree", definition: "Number of outgoing edges from a component — how many components this node depends on. Shown in Statistics Library Deps." },
  { term: "Outbound Rate", definition: "Fan-out deliveries leaving a broker per second. Each inbound message is copied once per subscriber. Formula: Σ(inbound × subscriber_count) over all routed topics. Shown in the Broker Usage panel of the Traffic Simulator." },
  { term: "Outlier", definition: "A data point that exceeds the IQR upper fence. Statistically extreme relative to the population; highlighted in charts and outlier tables.", formula: "outlier if value > Q3 + 1.5 × IQR" },
  { term: "Outlier Nodes", definition: "Infrastructure nodes whose combined pub/sub load is statistically extreme. Listed in the outlier table at the bottom of the Statistics Node Load tab." },
  { term: "Outlier Pairs", definition: "Node or segment pairs whose message-flow count is statistically extreme relative to all other pairs in the heatmap. Shown in the outlier table below the Cross-Node and Segment Comm heatmaps." },
  // P
  { term: "path_count", definition: "Number of shared topics (for app_to_app / app_to_broker) or shared nodes (for broker_to_broker) that establish a single DEPENDS_ON edge. Higher values indicate stronger coupling — multiple simultaneous failure vectors exist between the same component pair.", formula: "path_count = |shared topics or nodes between the pair|" },
  { term: "Path Length", definition: "Minimum number of hops between two nodes. Average path length is linked from the Dashboard statistics card." },
  { term: "Peak Topic Bandwidth", definition: "Highest single-topic bandwidth in the simulation set. The busiest topic is always rendered in red on the bandwidth colour scale. Useful as a reference for relative load across topics." },
  { term: "Producer-Only App", definition: "An application that publishes above the system mean but subscribes below it. Upstream failure causes downstream data loss.", formula: "pub > avg_pub AND sub ≤ avg_sub" },
  { term: "Producers", definition: "Application nodes that publish messages to topics — sources in the message flow. Shown in Statistics App Balance." },
  { term: "Publisher Count", definition: "Number of distinct applications actively publishing messages to a topic. Displayed in topic tooltip enrichment." },
  { term: "Pub/Sub Betweenness", definition: "Betweenness centrality computed specifically on the publish-subscribe message routing graph. Shown in Bottleneck item tooltips." },
  { term: "Publisher Bandwidth", definition: "Estimated byte throughput for a topic based on message size multiplied by the number of publishers. Shown as the Pub BW series in the Topic Bandwidth stacked bar chart.", formula: "bandwidth_pub = size × pub_count" },
  { term: "PUBLISHES_TO", definition: "Structural edge from an Application or Library to a Topic it writes messages to. Direction: App/Lib → Topic. Part of G_structural; weight is inherited from the target topic's QoS weight.", formula: "(Application|Library) -[:PUBLISHES_TO]→ (Topic)" },
  // Q
  { term: "QoS Durability", definition: "A QoS policy controlling whether messages are persisted for late-joining subscribers. Shown in topic chart tooltips in Statistics." },
  { term: "QoS Reliability", definition: "A QoS policy controlling message delivery guarantees — RELIABLE or BEST_EFFORT. Shown in topic chart tooltips in Statistics." },
  { term: "QoS Transport Priority", definition: "A QoS policy controlling the transmission priority level of messages — HIGHEST, HIGH, MEDIUM, or LOW. Shown in topic tooltips." },
  { term: "QoS Weight", definition: "Operational priority weight between 0 and 1 derived from QoS settings. Higher means more operationally critical. Shown in Bottleneck item tooltips." },
  { term: "Q(v)", definition: "Overall component quality score in the Analysis page, aggregating four RMAS dimensions via AHP weights (shrinkage λ=0.7).", formula: "Q(v) = 0.24·R(v) + 0.17·M(v) + 0.43·A(v) + 0.16·S(v)" },
  { term: "QADS", definition: "QoS-weighted Attack-Dependent Surface — inbound dependency weight measuring how many high-criticality components trust this node. Input to S(v)." },
  { term: "QSPOF", definition: "QoS-scaled SPOF Severity — product of the directional articulation-point score and the component's operational weight w(v). Amplifies SPOF severity for high-criticality components. Input to A(v).", formula: "QSPOF = AP_c_directed × w(v)" },
  { term: "Q_ensemble", definition: "Ensemble-blended quality score combining GNN inference with rule-based RMAS scoring in the Predict stage.", formula: "Q_ensemble(v) = α·Q_GNN + (1−α)·Q_RMAS" },
  // R
  { term: "Reliability Score R(v)", definition: "RMAS Reliability dimension score measuring fault-propagation reach from a component. AHP weight in Q(v): 0.24.", formula: "R(v) = 0.45·RPR + 0.30·DG_in + 0.25·CDPot_enh" },
  { term: "Reverse Closeness Centrality", definition: "Closeness centrality computed on the transposed graph G^T. Measures how quickly an adversarial compromise can propagate from this component to all others via trust-based dependency paths. Input to S(v)." },
  { term: "Reverse Eigenvector Centrality", definition: "Eigenvector centrality computed on the transposed graph G^T. Identifies components that are central in the reverse dependency network, indicating strategic attack reach. Input to S(v)." },
  { term: "Reverse PageRank", definition: "PageRank computed on the transposed graph G^T — measures fault-propagation reach from a failed component outward through reverse dependency edges. Primary input to R(v)." },
  { term: "RMAS", definition: "The four quality dimensions assessed per component in the Analysis page: Reliability (R), Maintainability (M), Availability (A), and Security (S). Each dimension has a dedicated score, ground-truth simulator, and validation metric set." },
  { term: "Role", definition: "An optional attribute on an Application node indicating its functional responsibility. Shown in bar-chart tooltips in Statistics and used to group components in the Simulator Roles tab." },
  { term: "ROUTES", definition: "Structural edge from a Broker to a Topic indicating that the broker is responsible for routing messages on that channel. Direction: Broker → Topic. Used in Rules 2 and 4 of DEPENDS_ON derivation.", formula: "(Broker) -[:ROUTES]→ (Topic)" },
  { term: "RUNS_ON", definition: "Structural edge from an Application or Broker to the infrastructure Node it is deployed on. Direction: App/Broker → Node. Used in Rules 3–6 to lift application-level dependencies to the node and broker levels.", formula: "(Application|Broker) -[:RUNS_ON]→ (Node)" },
  // S
  { term: "Saved Config", definition: "A named topic-selection and parameter set (frequency, duration, message size) stored in browser local storage. Can be reloaded to repeat previous Traffic Simulator runs without reconfiguring." },
  { term: "Score Outlier", definition: "A component whose bottleneck score exceeds the IQR upper fence. Flagged ⚡ in the Bottlenecks table.", formula: "score > Q3 + 1.5 × IQR" },
  { term: "Segment", definition: "A logical grouping of related components sharing a common operational concern or data boundary — the CSS level in the hierarchy." },
  { term: "Segment Communication", definition: "Volume of message flow between architectural segments. High cross-segment traffic signals tight subsystem coupling. Shown in the Statistics Segment Comm tab." },
  { term: "Segment Diversity", definition: "Variety of applications, topics, and I/O load per segment. Low diversity flags monolithic subsystems; high I/O flags communication hubs. Statistics Segment Diversity tab." },
  { term: "Shared Topics", definition: "The drill-down panel that appears when clicking a cell in the Cross-Node or Segment Comm heatmap. Lists topics where the row entity publishes and the column entity subscribes — showing the exact inter-entity data paths." },
  { term: "Simulation Duration", definition: "Simulated time window in seconds. Affects total message counts but not bandwidth rates.", formula: "total_messages = frequency_hz × duration_sec × publisher_count" },
  { term: "Simulation Frequency", definition: "How many messages each publisher sends per second (Hz). Can be set globally or overridden per topic in the Traffic Simulator. Typical ranges: 1 Hz (slow telemetry) · 10 Hz (moderate control loop) · 100 Hz (high-frequency sensor)." },
  { term: "size", definition: "Message payload size in bytes for a topic. Stored as the size property on Topic nodes in Neo4j. Used in the topic weight formula and all bandwidth calculations.", formula: "bandwidth = size × subscriber_count (or publisher_count)" },
  { term: "sqale_debt_ratio", definition: "SQALE technical debt ratio: estimated remediation effort as a percentage of total development cost, as reported by SonarQube. High values indicate significant accumulated code quality debt. Stored on Application and Library nodes." },
  { term: "SUBSCRIBES_TO", definition: "Structural edge from an Application or Library to a Topic it receives messages from. Direction: App/Lib → Topic. Part of G_structural; weight is inherited from the target topic's QoS weight. Fan-out is derived from this edge in Phase 2.", formula: "(Application|Library) -[:SUBSCRIBES_TO]→ (Topic)" },
  { term: "Subscriber Bandwidth", definition: "Estimated byte throughput for a topic based on message size multiplied by the number of subscribers. Shown as the Sub BW series in the Topic Bandwidth stacked bar chart.", formula: "bandwidth_sub = size × sub_count" },
  { term: "System Critical %", definition: "Percentage of all applications across the system that are marked as mission-critical. High values reduce redundancy headroom and increase vulnerability to targeted failures.", formula: "critical_apps / total_apps × 100" },
  { term: "Structural Relationships", definition: "Physical topology edges: RUNS_ON, PUBLISHES_TO, SUBSCRIBES_TO, ROUTES, USES, CONNECTS_TO. Shown in the Dashboard Structural Relationships chart." },
  { term: "Subscriber Count", definition: "Number of distinct applications subscribed to receive messages from a topic. Displayed in topic tooltip enrichment." },
  { term: "System Hierarchy", definition: "The five-level decomposition: CSMS → CSS → CSCI → CSC → CSU. Shown as tree levels in the Explorer hierarchy panel." },
  // T
  { term: "Topic", definition: "A named message channel. Publishers write to it; subscribers read from it. Shown as a node type in the Explorer." },
  { term: "Topic Bandwidth", definition: "Total estimated data throughput per topic based on message size and the number of connected publishers or subscribers. Statistics Topic Bandwidth tab.", formula: "bandwidth_sub = size × sub_count" },
  { term: "Topic Fanout", definition: "Publisher × subscriber count per topic. High fanout amplifies failure blast radius. Statistics Topic Fanout tab.", formula: "fanout = pub_count × sub_count" },
  { term: "Topic Weight", definition: "QoS-based operational priority weight ∈ [0, 1] assigned to a topic. Derived from its reliability and durability QoS settings. 0.0 = minimal priority · 1.0 = highest criticality. Feeds the RMAS scoring pipeline and Traffic Simulator weighting." },
  { term: "Total Dependencies", definition: "Total number of application-to-library USES edges in the system. Measures overall coupling density between the application and library layers.", formula: "count(USES relationships)" },
  { term: "Total Edges", definition: "Combined count of derived (DEPENDS_ON) and structural graph edges. Shown as a KPI tile on the Dashboard." },
  // U
  { term: "USES", definition: "Structural edge from an Application or Library to a Library it depends on. Direction: App/Lib → Library. The source of app_to_lib DEPENDS_ON edges (Rule 5). Also increments the library's DG_in, which amplifies its weight via the fan-out multiplier.", formula: "(Application|Library) -[:USES]→ (Library)" },
  // W
  { term: "version", definition: "Software version string for Application and Library nodes in Neo4j. Used for change-management tracking and dependency compatibility analysis across the system." },
  { term: "vulnerabilities", definition: "Number of known security vulnerabilities detected by static analysis (e.g., SonarQube). An optional property on Application and Library nodes that contributes to security risk scoring when present." },
  { term: "Security Score S(v)", definition: "RMAS Security dimension score measuring adversarial compromise risk via reverse-graph centrality metrics. AHP weight in Q(v): 0.16.", formula: "S(v) = 0.40·REV + 0.35·RCL + 0.25·QADS" },
  // W
  { term: "Weight", definition: "QoS-derived priority weight in [0, 1] assigned to components and edges. Higher values indicate more operational criticality. Shown in Bottleneck item tooltips." },
  // Z
  { term: "Zero Activity", definition: "Components with no publish or subscribe activity detected during the analysis window. Counted in Statistics App Balance." },
  { term: "Zero-Subscriber Topics", definition: "Topics that are published to but never consumed. Dead channels that waste publisher resources and typically indicate stale topic definitions.", formula: "count(topics where sub_count = 0)" },
]

// ─── Abbreviations ───────────────────────────────────────────────────────────

const ABBREVIATIONS: Abbreviation[] = [
  { abbr: "1→N",  expansion: "One-to-Many (Broadcast)",              definition: "A topic with one publisher and multiple subscribers. A publisher failure silences all consumers simultaneously." },
  { abbr: "AP",   expansion: "Articulation Point",                    definition: "A node whose removal disconnects the graph. The most severe bottleneck class; flagged 🔴 in the Bottlenecks table." },
  { abbr: "AHCR@K",expansion: "Attack Hit Capture Rate at K",          definition: "Fraction of the top-K simulated attack targets (by IS(v)) captured by the top-K security predictions S(v). Validation metric for the Security dimension." },
  { abbr: "AHP",  expansion: "Analytic Hierarchy Process",             definition: "Pairwise comparison method used to derive RMAS dimension weights. A shrinkage factor λ=0.7 blends AHP-derived weights with a uniform prior to prevent over-fitting to a single dimension.", },
  { abbr: "APAR", expansion: "Attack Path Agreement Rate",             definition: "Fraction of adversarial propagation paths that agree between IS(v) simulation and S(v) prediction. Validation metric for the Security dimension." },
  { abbr: "BFS",  expansion: "Breadth-First Search",                  definition: "Graph traversal algorithm used in cascade, change-propagation, and compromise simulators." },
  { abbr: "BR",   expansion: "Bridge Ratio",                          definition: "Fraction of incident edges that are bridges. Higher values mean most connections are single-path." },
  { abbr: "BT",   expansion: "Betweenness",                           definition: "Fraction of shortest paths passing through a node. Input coefficient in the Bottleneck Score formula." },
  { abbr: "Ca",   expansion: "Afferent Coupling",                     definition: "Number of external classes that depend on this component. Also stored as coupling_afferent on Application/Library nodes." },
  { abbr: "CBO",  expansion: "Coupling Between Objects",              definition: "Object-oriented coupling metric: number of other classes a class depends on. Stored in cm_avg_cbo / cm_max_cbo on Application and Library nodes." },
  { abbr: "Ce",   expansion: "Efferent Coupling",                     definition: "Number of external classes this component's classes depend on. Also stored as coupling_efferent on Application/Library nodes." },
  { abbr: "CCR@K",expansion: "Cascade Capture Rate at K",             definition: "Fraction of the top-K simulated cascade targets (by IR(v)) captured by the top-K reliability predictions R(v). Validation metric for the Reliability dimension." },
  { abbr: "CDI",  expansion: "Connectivity Degradation Index",         definition: "Normalised increase in average shortest-path length after removing a component. Input to A(v)." },
  { abbr: "CDPot",expansion: "Cascade Depth Potential",               definition: "Estimated maximum cascade depth for a component. Enhanced: CDPot_enh = CDPot_base × (1 + MPCI). Input to R(v)." },
  { abbr: "CME",  expansion: "Cascade Magnitude Error",               definition: "Mean absolute error between predicted cascade magnitude R(v) and simulated cascade reach IR(v). Validation metric for the Reliability dimension." },
  { abbr: "COCR@K",expansion: "Change Overlap Capture Rate at K",     definition: "Fraction of the top-K change-propagation targets (by IM(v)) captured by top-K M(v) predictions. Validation metric for the Maintainability dimension." },
  { abbr: "CQP",  expansion: "Code Quality Penalty",                  definition: "Composite code-quality score used in M(v): 0.40×complexity_norm + 0.35×instability_code + 0.25×lcom_norm. Falls back to 0 when code metrics are absent." },
  { abbr: "CSC",  expansion: "Computer Software Component",           definition: "A component grouping that contains deployable application units within a configuration item." },
  { abbr: "CSCI", expansion: "Computer Software Configuration Item",  definition: "A named unit of software configuration within a domain — the third level of the system hierarchy." },
  { abbr: "CSMS", expansion: "Computer Software Management System",   definition: "Top-level system scope in the component hierarchy. The root level shown in the Explorer tree." },
  { abbr: "CSS",  expansion: "Computer Software Segment",             definition: "A logical domain grouping related configuration items within the system — the second level of the hierarchy." },
  { abbr: "CSU",  expansion: "Computer Software Unit",                definition: "A deployable application unit — the leaf level of the component hierarchy shown in the Explorer." },
  { abbr: "CTA",  expansion: "Change Transfer Agreement",             definition: "Weighted Cohen's κ comparing M(v) classification tiers against IM(v) classification tiers. Measures ordinal rank agreement. Validation metric for the Maintainability dimension." },
  { abbr: "CV",   expansion: "Coefficient of Variation",              definition: "Standard deviation / mean of node load, as a percentage. High CV means uneven workload distribution." },
  { abbr: "DAG",  expansion: "Directed Acyclic Graph",                definition: "A graph with no cycles. Healthy dependency graphs aim to approximate a DAG." },
  { abbr: "DG_in",expansion: "Normalised In-Degree",                  definition: "In-degree of a component normalised to [0,1]. Direct dependent count; primary input to R(v) alongside RPR." },
  { abbr: "DG_out",expansion: "Normalised Out-Degree",                definition: "Out-degree of a component normalised to [0,1]. QoS-weighted efferent coupling w_out; input to M(v)." },
  { abbr: "DDS",  expansion: "Data Distribution Service",             definition: "Publish-subscribe middleware standard used in real-time systems and robotics." },
  { abbr: "FTR",  expansion: "False Trust Rate",                      definition: "Fraction of non-compromised components incorrectly predicted as vulnerable by S(v). Validation metric for the Security dimension." },
  { abbr: "GAT",  expansion: "Graph Attention Network",               definition: "The GNN architecture used in the Predict stage. Multi-head attention over heterogeneous component types learns criticality from simulated ground-truth labels." },
  { abbr: "GB/s", expansion: "Gigabytes per Second",                  definition: "Bandwidth unit in the Traffic Simulator. 1 GB/s = 1 000 MB/s. Displayed when total simulation throughput exceeds 1 000 MB/s." },
  { abbr: "GNN",  expansion: "Graph Neural Network",                  definition: "The machine-learning model in Step 3 (Predict). A heterogeneous GAT trained on simulate-derived ground-truth labels that generalises beyond closed-form RMAS scoring." },
  { abbr: "Hz",   expansion: "Hertz (messages per second)",           definition: "Simulation frequency unit. Specifies how many messages each publisher sends per second. Set globally or overridden per topic in the Traffic Simulator." },
  { abbr: "I(v)", expansion: "Overall Impact Score",                  definition: "Ground-truth aggregate impact of removing component v: 0.35·reachability_loss + 0.25·fragmentation + 0.25·throughput_loss + 0.15·flow_disruption. Produced by the failure simulator." },
  { abbr: "I/O",  expansion: "Input / Output",                        definition: "Combined publish and subscribe activity count for a component. Used to classify High I/O Apps in Statistics." },
  { abbr: "IA",   expansion: "Availability Impact IA(v)",             definition: "Ground-truth availability impact from removing v: QoS-weighted connectivity disruption. Orthogonal to cascade propagation (IR)." },
  { abbr: "IM",   expansion: "Maintainability Impact IM(v)",          definition: "Ground-truth change-propagation impact from BFS on G^T: ChangeReach, WeightedChangeImpact, NormalizedChangeDepth." },
  { abbr: "IR",   expansion: "Reliability Impact IR(v)",              definition: "Ground-truth cascade failure impact: CascadeReach, WeightedCascadeImpact, NormalizedCascadeDepth. Training label for the GNN Predict stage." },
  { abbr: "IS",   expansion: "Security Impact IS(v)",            definition: "Ground-truth adversarial compromise impact via trust-based BFS on G^T with θ_trust = 0.30." },
  { abbr: "IQR",  expansion: "Interquartile Range",                   definition: "Spread between Q1 (25th percentile) and Q3 (75th percentile). Used to detect outliers across all Statistics charts." },
  { abbr: "KB/s", expansion: "Kilobytes per Second",                  definition: "Bandwidth unit in the Traffic Simulator. 1 KB/s = 1 000 B/s. Displayed for low-throughput topics." },
  { abbr: "LCOM", expansion: "Lack of Cohesion of Methods",           definition: "Measures how unrelated the methods in a class are. Values near 1 indicate poor cohesion. Stored as lcom / cm_avg_lcom / cm_max_lcom on Application and Library nodes." },
  { abbr: "LOC",  expansion: "Lines of Code",                         definition: "Raw or SonarQube-measured source line count for a component. Stored as loc and cm_total_loc. Input to the Code Quality Penalty in M(v)." },
  { abbr: "MB/s", expansion: "Megabytes per Second",                  definition: "Primary bandwidth unit in the Traffic Simulator. 1 MB/s = 1 000 KB/s. Typical simulator result columns display values in MB/s." },
  { abbr: "MPCI", expansion: "Multi-Path Cascade Index",              definition: "Measures the number and combined weight of parallel cascade paths from a component. Enhances CDPot: CDPot_enh = CDPot_base × (1 + MPCI). Input to R(v) with weight 0.25." },
  { abbr: "msg/s",expansion: "Messages per Second",                   definition: "Message rate unit used in the Traffic Simulator. Inbound rate = publishers × Hz; outbound rate = inbound × subscribers (fan-out)." },
  { abbr: "MQTT", expansion: "Message Queuing Telemetry Transport",   definition: "Lightweight pub-sub protocol designed for IoT and constrained-network devices." },
  { abbr: "NDCG@K",expansion: "Normalised Discounted Cumulative Gain at K", definition: "Ranking quality metric comparing predicted vs simulated component rankings, discounted by position. Used in overall and per-dimension validation." },
  { abbr: "N→1",  expansion: "Many-to-One (Aggregation)",             definition: "A topic with multiple publishers and one subscriber. Any publisher failure reduces data feed completeness." },
  { abbr: "N→N",  expansion: "Many-to-Many (Mesh)",                   definition: "A topic with multiple publishers and multiple subscribers. Mesh communication pattern." },
  { abbr: "QoS",  expansion: "Quality of Service",                    definition: "Policies governing message delivery, durability, and transport priority. Determines component and edge weights." },
  { abbr: "QADS", expansion: "QoS-weighted Attack-Dependent Surface", definition: "Inbound dependency weight measuring high-criticality components that trust this node. Input to S(v) with weight 0.25." },
  { abbr: "QSPOF",expansion: "QoS-scaled SPOF Severity",              definition: "AP_c_directed × w(v) — amplifies articulation-point severity for high-criticality components. Input to A(v) with weight 0.25." },
  { abbr: "RCL",  expansion: "Reverse Closeness Centrality",          definition: "Closeness centrality on G^T. Measures adversarial propagation speed from a component. Input to S(v) with weight 0.35." },
  { abbr: "REV",  expansion: "Reverse Eigenvector Centrality",        definition: "Eigenvector centrality on G^T. Identifies components with the highest strategic attack reach. Input to S(v) with weight 0.40." },
  { abbr: "RMAS", expansion: "Reliability · Maintainability · Availability · Security", definition: "The four quality dimensions scored per component: Q(v) = 0.24R + 0.17M + 0.43A + 0.16S." },
  { abbr: "RPR",  expansion: "Reverse PageRank",                      definition: "PageRank on G^T — measures fault-propagation reach from a failed component. Input to R(v) with weight 0.45." },
  { abbr: "RRI",  expansion: "Robustness Rank Improvement",           definition: "Improvement in Spearman correlation when zero-impact components are excluded from A(v) vs IA(v) ranking. Validation metric for Availability." },
  { abbr: "ROS2", expansion: "Robot Operating System 2",              definition: "Pub-sub framework for robotics built on DDS middleware." },
  { abbr: "RFC",  expansion: "Response for a Class",                  definition: "Total number of methods callable by an instance of a class. High values indicate large interface surfaces and higher testing effort. Stored as cm_avg_rfc / cm_max_rfc." },
  { abbr: "SCC",  expansion: "Strongly Connected Component",          definition: "Maximal subgraph where every node can reach every other via directed edges. Cyclic dependencies form SCCs." },
  { abbr: "SQALE",expansion: "Software Quality Assessment based on Lifecycle Expectations", definition: "A technical-debt model used by SonarQube. The sqale_debt_ratio field expresses accumulated remediation cost as a percentage of total development cost." },
  { abbr: "SPOF", expansion: "Single Point of Failure",               definition: "A component whose removal alone causes significant system-wide disruption. Corresponds to an articulation point." },
  { abbr: "SPOF_F1",expansion: "SPOF Classification F1",             definition: "Harmonic mean of precision and recall when using A(v) ≥ CRITICAL to classify articulation points. Validation metric for the Availability dimension." },
  { abbr: "WCC",  expansion: "Weakly Connected Component",            definition: "Subgraph where every pair of nodes is connected when ignoring edge direction. Multiple WCCs mean a fragmented topology." },
  { abbr: "WMC",  expansion: "Weighted Methods per Class",            definition: "Sum (or average) of cyclomatic complexity across all methods in a class. Used as a code-complexity proxy. Stored as cm_avg_wmc, cm_max_wmc, cm_total_wmc on Application and Library nodes." },
]

// ─── Helpers ──────────────────────────────────────────────────────────────────

function groupByLetter(terms: Term[]): Record<string, Term[]> {
  const groups: Record<string, Term[]> = {}
  for (const t of terms) {
    const letter = t.term[0].toUpperCase()
    ;(groups[letter] ??= []).push(t)
  }
  return groups
}

const TOTAL_TERMS = ALL_TERMS.length
const TOTAL_ABBR  = ABBREVIATIONS.length

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function GlossaryPage() {
  const [search, setSearch]       = useState("")
  const [activeLetter, setActive] = useState<string>("All")
  const [mounted, setMounted]     = useState(false)
  const sectionRefs = useRef<Record<string, HTMLDivElement | null>>({})

  useEffect(() => { setMounted(true) }, [])

  const q = search.trim().toLowerCase()

  const filteredTerms = useMemo(() => {
    return ALL_TERMS.filter(t => {
      const letterOk = activeLetter === "All" || activeLetter === "Abbr" || t.term[0].toUpperCase() === activeLetter
      const searchOk = !q || t.term.toLowerCase().includes(q) || t.definition.toLowerCase().includes(q)
      return letterOk && searchOk
    })
  }, [q, activeLetter])

  const filteredAbbr = useMemo(() => {
    if (activeLetter !== "All" && activeLetter !== "Abbr") return []
    return ABBREVIATIONS.filter(a =>
      !q || a.abbr.toLowerCase().includes(q) || a.expansion.toLowerCase().includes(q) || a.definition.toLowerCase().includes(q)
    )
  }, [q, activeLetter])

  const grouped       = useMemo(() => groupByLetter(filteredTerms), [filteredTerms])
  const activeLetters = useMemo(() => Object.keys(grouped).sort(), [grouped])

  const existingLetters = useMemo(() => {
    const s = new Set<string>()
    for (const t of ALL_TERMS) s.add(t.term[0].toUpperCase())
    return s
  }, [])

  const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("")

  const showTerms = activeLetter !== "Abbr"
  const showAbbr  = activeLetter === "All" || activeLetter === "Abbr"
  const noResults = filteredTerms.length === 0 && filteredAbbr.length === 0

  function scrollTo(letter: string) {
    sectionRefs.current[letter]?.scrollIntoView({ behavior: "smooth", block: "start" })
  }

  if (!mounted) {
    return (
      <AppLayout title="Glossary" description="Key terms, metrics, and abbreviations used across the platform.">
        <div className="space-y-5">
          {/* Search skeleton */}
          <Skeleton className="h-9 w-72 rounded-md" />
          {/* Letter nav skeleton */}
          <div className="flex flex-wrap gap-1">
            {Array.from({ length: 28 }).map((_, i) => (
              <Skeleton key={i} className="h-6 w-8 rounded" />
            ))}
          </div>
          {/* Stats strip skeleton */}
          <div className="flex items-center gap-3">
            <Skeleton className="h-3 w-16" />
            <Skeleton className="h-3 w-2" />
            <Skeleton className="h-3 w-24" />
          </div>
          {/* Terms skeleton */}
          <div className="space-y-6">
            {Array.from({ length: 4 }).map((_, gi) => (
              <div key={gi} className="space-y-3">
                <Skeleton className="h-5 w-6 rounded" />
                {Array.from({ length: 4 }).map((_, i) => (
                  <div key={i} className="rounded-lg border border-border bg-muted/20 p-4 space-y-2">
                    <Skeleton className="h-4 w-32" />
                    <Skeleton className="h-3 w-full" />
                    <Skeleton className="h-3" style={{ width: `${60 + (i * 17) % 35}%` }} />
                  </div>
                ))}
              </div>
            ))}
          </div>
        </div>
      </AppLayout>
    )
  }

  return (
    <AppLayout
      title="Glossary"
      description="Key terms, metrics, and abbreviations used across the platform."
    >
      <div className="space-y-5">

        {/* ── Search ──────────────────────────────────────────────────── */}
        <div className="relative max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search terms or abbreviations…"
            value={search}
            onChange={e => { setSearch(e.target.value); setActive("All") }}
            className="pl-9 h-9"
          />
        </div>

        {/* ── Letter navigation ────────────────────────────────────────── */}
        <div className="flex flex-wrap gap-1">
          <button
            onClick={() => setActive("All")}
            className={`min-w-[2rem] px-2 py-0.5 text-xs rounded border font-medium transition-colors ${
              activeLetter === "All"
                ? "bg-foreground/10 border-foreground/30 text-foreground"
                : "border-border text-muted-foreground hover:text-foreground hover:border-foreground/20"
            }`}
          >
            All
          </button>

          {ALPHABET.map(l => {
            const hasTerms = existingLetters.has(l)
            const isActive = activeLetter === l
            return (
              <button
                key={l}
                disabled={!hasTerms}
                onClick={() => { setActive(isActive ? "All" : l); scrollTo(l) }}
                className={`min-w-[2rem] px-2 py-0.5 text-xs rounded border font-mono font-medium transition-colors ${
                  isActive
                    ? "bg-blue-500/15 border-blue-500/40 text-blue-300"
                    : hasTerms
                      ? "border-border text-muted-foreground hover:text-foreground hover:border-foreground/20"
                      : "border-border/30 text-muted-foreground/30 cursor-not-allowed"
                }`}
              >
                {l}
              </button>
            )
          })}

          <button
            onClick={() => setActive(activeLetter === "Abbr" ? "All" : "Abbr")}
            className={`px-2.5 py-0.5 text-xs rounded border font-medium transition-colors ${
              activeLetter === "Abbr"
                ? "bg-violet-500/15 border-violet-500/40 text-violet-300"
                : "border-violet-500/30 text-violet-400 hover:opacity-80"
            }`}
          >
            Abbr
          </button>
        </div>

        {/* ── Stats strip ─────────────────────────────────────────────── */}
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span>{TOTAL_TERMS} terms</span>
          <span className="text-border">·</span>
          <span>{TOTAL_ABBR} abbreviations</span>
          {q && (
            <>
              <span className="text-border">·</span>
              <span>{filteredTerms.length + filteredAbbr.length} results</span>
            </>
          )}
        </div>

        {/* ── Content ─────────────────────────────────────────────────── */}
        {noResults ? (
          <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-slate-700 bg-slate-900/30 py-16 text-center">
            <BookMarked className="h-8 w-8 text-slate-500" />
            <p className="text-sm font-medium text-slate-300">No results match your search</p>
            <p className="text-xs text-muted-foreground">Try a different keyword or clear the filter</p>
          </div>
        ) : (
          <div className="space-y-6">

            {/* ── Letter sections ─────────────────────────────────────── */}
            {showTerms && activeLetters.map(letter => (
              <div
                key={letter}
                ref={el => { sectionRefs.current[letter] = el }}
                className="scroll-mt-4"
              >
                <div className="flex items-center gap-3 mb-2">
                  <span className="text-2xl font-bold font-mono text-blue-400 leading-none w-8 shrink-0">
                    {letter}
                  </span>
                  <div className="flex-1 h-px bg-border" />
                  <Badge className="bg-blue-500/10 text-blue-400 border-blue-500/20 text-[10px] px-2 py-0.5 shrink-0">
                    {grouped[letter].length}
                  </Badge>
                </div>

                <Card className="bg-background">
                  <CardContent className="pt-3 pb-1">
                    <div className="divide-y divide-border">
                      {grouped[letter].map(t => (
                        <div key={t.term} className="flex gap-3 py-2.5 first:pt-0 last:pb-2">
                          <span className="shrink-0 w-52 text-xs font-mono font-semibold text-blue-300 leading-relaxed">
                            {t.term}
                          </span>
                          <span className="text-xs text-muted-foreground leading-relaxed">
                            {t.definition}
                            {t.formula && (
                              <code className="ml-2 text-[10px] font-mono bg-muted/60 rounded px-1.5 py-0.5 text-muted-foreground/80">
                                {t.formula}
                              </code>
                            )}
                          </span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            ))}

            {/* ── Abbreviations section ────────────────────────────────── */}
            {showAbbr && filteredAbbr.length > 0 && (
              <div
                ref={el => { sectionRefs.current["Abbr"] = el }}
                className="scroll-mt-4"
              >
                <div className="flex items-center gap-3 mb-2">
                  <div className="flex items-center gap-1.5 shrink-0">
                    <Hash className="h-5 w-5 text-violet-400" />
                    <span className="text-sm font-bold text-violet-400 leading-none">
                      Abbreviations
                    </span>
                  </div>
                  <div className="flex-1 h-px bg-border" />
                  <Badge className="bg-violet-500/10 text-violet-400 border-violet-500/20 text-[10px] px-2 py-0.5 shrink-0">
                    {filteredAbbr.length}
                  </Badge>
                </div>

                <Card className="bg-background">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-[11px] text-muted-foreground uppercase tracking-widest">
                      Abbreviations used in Dashboard · Explorer · Statistics
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="pt-0 pb-1">
                    <div className="divide-y divide-border">
                      {filteredAbbr.map(a => (
                        <div key={a.abbr} className="flex gap-3 py-2.5 first:pt-0 last:pb-2">
                          <div className="shrink-0 w-52 flex flex-col gap-0.5">
                            <span className="text-xs font-mono font-bold text-violet-300 leading-tight">
                              {a.abbr}
                            </span>
                            <span className="text-[10px] text-muted-foreground/70 leading-tight">
                              {a.expansion}
                            </span>
                          </div>
                          <span className="text-xs text-muted-foreground leading-relaxed">
                            {a.definition}
                          </span>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

          </div>
        )}

      </div>
    </AppLayout>
  )
}
