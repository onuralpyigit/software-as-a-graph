"use client"

import React, { useState, useEffect, useCallback, useMemo, useRef, memo, Suspense, useDeferredValue } from "react"
import { useSearchParams } from "next/navigation"
import dynamic from "next/dynamic"
import { useTheme } from "next-themes"
import { AppLayout } from "@/components/layout/app-layout"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { Skeleton } from "@/components/ui/skeleton"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip"
import { useConnection } from "@/lib/stores/connection-store"
import { useAnalysis } from "@/lib/stores/analysis-store"
import { apiClient } from "@/lib/api/client"
import { ReactFlow, ReactFlowProvider, Background, BackgroundVariant, Handle, Position, getBezierPath, applyNodeChanges, useViewport, useReactFlow, MarkerType, type NodeProps, type EdgeProps, type NodeChange } from "@xyflow/react"
import "@xyflow/react/dist/style.css"
import { cn } from "@/lib/utils"
import {
  ChevronDown,
  ChevronRight,
  Search,
  FolderOpen,
  Folder,
  Box,
  Cpu,
  Layers,
  Package,
  Network,
  List,
  X,
  Share2,
  Download,
  Eye,
  EyeOff,
} from "lucide-react"

const ReactECharts = dynamic(() => import("echarts-for-react"), { ssr: false })

// Polyfill GPUShaderStage to prevent errors when WebGPU is not available
if (typeof window !== 'undefined' && typeof (window as any).GPUShaderStage === 'undefined') {
  ;(window as any).GPUShaderStage = { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 }
}

// ── Types ─────────────────────────────────────────────────────────────────────

interface AppNode {
  id: string
  name?: string
  csc_name?: string
  csci_name?: string
  csms_name?: string
  css_name?: string
  csu?: string
  weight?: number
  [key: string]: unknown
}

// Hierarchy: System (CSMS) → Domain (CSS) → Config Item (CSCI) → Component (CSC) → App (CSU)
interface CscGroup  { name: string; apps: AppNode[] }
interface CsciGroup { name: string; csc: Record<string, CscGroup> }
interface CssGroup  { name: string; csci: Record<string, CsciGroup> }
interface CsmsGroup { name: string; css: Record<string, CssGroup> }

type SelectedKind = "csms" | "css" | "csci" | "csc" | "app" | "node" | "topic"
interface SelectedNode {
  kind: SelectedKind
  key: string
  label: string
  path: string[]
  payload: CsmsGroup | CssGroup | CsciGroup | CscGroup | AppNode
}

// Graph explorer types
type HGLevel = "csms" | "css" | "csci" | "csc" | "app"
interface HGNode {
  id: string
  name: string
  level: HGLevel
  nodeType?: string   // non-hierarchy node type (Topic, Node, Broker, Library …)
  appCount: number
  pathKey: string
  instanceKey?: string // full path within hierarchy tree: "app:csms/css/csci/csc/appId"
  appData?: AppNode   // raw AppNode data for app-level nodes (used in tooltips)
  // runtime fields added by force-graph simulation
  x?: number; y?: number; vx?: number; vy?: number; fx?: number; fy?: number
}
interface HGLink { source: string | HGNode; target: string | HGNode }

// ── Helpers ───────────────────────────────────────────────────────────────────

const OTHER = "(Other)"

function buildHierarchy(apps: AppNode[]): Record<string, CsmsGroup> {
  const root: Record<string, CsmsGroup> = {}
  for (const app of apps) {
    const csmsKey = app.csms_name?.trim() || OTHER
    const cssKey  = app.css_name?.trim()  || OTHER
    const csciKey = app.csci_name?.trim() || OTHER
    const cscKey  = app.csc_name?.trim()  || OTHER

    if (!root[csmsKey]) root[csmsKey] = { name: csmsKey, css: {} }
    const csms = root[csmsKey]
    if (!csms.css[cssKey]) csms.css[cssKey] = { name: cssKey, csci: {} }
    const css = csms.css[cssKey]
    if (!css.csci[csciKey]) css.csci[csciKey] = { name: csciKey, csc: {} }
    const csci = css.csci[csciKey]
    if (!csci.csc[cscKey]) csci.csc[cscKey] = { name: cscKey, apps: [] }
    const csc = csci.csc[cscKey]
    csc.apps.push(app)
  }
  return root
}

function matches(app: AppNode, q: string): boolean {
  if (!q) return true
  const f = (v?: string) => v?.toLowerCase().includes(q) ?? false
  return f(app.id) || f(app.name) || f(app.csc_name) || f(app.csci_name) ||
         f(app.csms_name) || f(app.css_name) || f(app.csu)
}

function sortKeys(keys: string[]): string[] {
  return keys.sort((a, b) => a === OTHER ? 1 : b === OTHER ? -1 : a.localeCompare(b))
}

// ── Graph helpers ─────────────────────────────────────────────────────────────

// Hierarchy node colors — distinct from connection node types to avoid confusion
const NODE_COLORS: Record<HGLevel, string> = {
  csms: "#ec4899", css: "#8b5cf6", csci: "#06b6d4", csc: "#14b8a6", app: "#4CBCD0",
}
const NODE_SIZES: Record<HGLevel, number> = {
  csms: 14, css: 10, csci: 8, csc: 6, app: 3.5,
}
const LEVEL_LABELS: Record<HGLevel, string> = {
  csms: "System (CSMS)", css: "Segment (CSS)", csci: "Config Item (CSCI)", csc: "Component (CSC)", app: "App (CSU)",
}

// Hierarchical connections-view layout: assign a y-layer per node type
const CONN_TYPE_LAYER: Record<string, number> = {
  Node: 0, Broker: 1, Application: 2, Topic: 3, Library: 4,
}
// Fraction of canvas height for each layer (top → bottom)
const CONN_LAYER_Y_FRACS = [0.10, 0.28, 0.50, 0.72, 0.90]
const CONN_LAYER_LABEL: Record<number, string> = {
  0: "Node", 1: "Broker", 2: "Application", 3: "Topic", 4: "Library",
}

// Node-type → color. Single source of truth shared with the Force Graph tab
// (FORCE_CATEGORIES) so item-type colours are consistent across List, Tree,
// and Force Graph views. Same values for both themes — Force Graph does not
// theme-switch, and consistency across tabs trumps per-theme contrast tuning.
const CONN_NODE_TYPE_COLORS_DARK: Record<string, string> = {
  Application: "#4CBCD0",
  Node:        "#C570CE",
  Broker:      "#EFC050",
  Topic:       "#7DAA7A",
  Library:     "#ECA088",
}
const CONN_NODE_TYPE_COLORS_LIGHT: Record<string, string> = {
  Application: "#4CBCD0",
  Node:        "#C570CE",
  Broker:      "#EFC050",
  Topic:       "#7DAA7A",
  Library:     "#ECA088",
}
// Link-type → color. Aligned with the Force Graph tab's EDGE_COLORS so
// relationship-type colours are consistent across List, Tree, and Force Graph.
// DEPENDS_ON / CONNECTS_TO are not part of EDGE_COLORS but are kept here for
// the Tree-tab connection groups; they reuse semantically related hues.
const CONN_LINK_TYPE_COLORS_DARK: Record<string, string> = {
  RUNS_ON:       "#64748b",
  PUBLISHES_TO:  "#22c55e",
  SUBSCRIBES_TO: "#f97316",
  USES:          "#06b6d4",
  ROUTES:        "#d946ef",
  DEPENDS_ON:    "#ef4444",
  CONNECTS_TO:   "#84cc16",
}
const CONN_LINK_TYPE_COLORS_LIGHT: Record<string, string> = {
  RUNS_ON:       "#64748b",
  PUBLISHES_TO:  "#22c55e",
  SUBSCRIBES_TO: "#f97316",
  USES:          "#06b6d4",
  ROUTES:        "#d946ef",
  DEPENDS_ON:    "#ef4444",
  CONNECTS_TO:   "#84cc16",
}
/** Deterministic fallback color for unknown types */
function hashTypeColor(type: string): string {
  let h = 0
  for (let i = 0; i < type.length; i++) h = (Math.imul(31, h) + type.charCodeAt(i)) | 0
  const hue = Math.abs(h) % 360
  return `hsl(${hue},65%,55%)`
}
function nodeTypeColor(type: string | undefined, isDark = true): string {
  if (!type) return isDark ? "#a1a1aa" : "#71717a"
  return (isDark ? CONN_NODE_TYPE_COLORS_DARK : CONN_NODE_TYPE_COLORS_LIGHT)[type] ?? hashTypeColor(type)
}
function linkTypeColor(type: string | undefined, isDark = true): string {
  if (!type) return isDark ? "#a1a1aa" : "#71717a"
  return (isDark ? CONN_LINK_TYPE_COLORS_DARK : CONN_LINK_TYPE_COLORS_LIGHT)[type] ?? hashTypeColor(type)
}

// ── Property descriptions (shown as tooltips on key labels) ─────────────────
const PROP_DESCS: Record<string, string> = {
  // ── System hierarchy ──────────────────────────────────────────────────────
  csms_name:               "Computer Software Management System — top-level system boundary this component belongs to",
  css_name:                "Computer Software Segment — functional grouping within the CSMS",
  csci_name:               "Computer Software Configuration Item — deployable unit within the CSS",
  csc_name:                "Computer Software Component — the immediate architectural parent of this application",
  // ── Identity / classification ──────────────────────────────────────────────
  id:                      "Unique identifier for this component in the graph",
  name:                    "Human-readable display name",
  type:                    "Component type (Application, Library, Node, Broker, Topic)",
  app_type:                "Application sub-type (e.g. publisher, subscriber, service)",
  role:                    "Functional role of this application within the system",
  criticality:             "Statically declared criticality tier (CRITICAL / HIGH / MEDIUM / LOW)",
  priority:                "Runtime priority tier of the application — HIGH, MEDIUM, or LOW",
  hotstandby:              "True when this application has a warm-standby replication pair deployed on two distinct infrastructure nodes (published as two RUNS_ON edges in Neo4j)",
  version:                 "Software version string for this component",
  layer:                   "Architectural layer this component belongs to (app, infra, mw, system)",
  // ── Infrastructure node ────────────────────────────────────────────────────
  ip_address:              "IP address of the physical or virtual host node",
  cpu_cores:               "Number of logical CPU cores available on this host",
  memory_gb:               "RAM capacity of this host in gigabytes",
  os_type:                 "Operating system type running on this host (e.g. Linux, Windows)",
  // ── Broker ────────────────────────────────────────────────────────────────
  host:                    "Hostname or IP address where this broker is reachable",
  max_connections:         "Maximum number of concurrent client connections this broker accepts",
  broker_type:             "Middleware protocol/type of this broker (e.g. MQTT, DDS, ROS2)",
  // ── Topic / QoS ───────────────────────────────────────────────────────────
  qos_reliability:         "QoS Reliability policy — BEST_EFFORT (0) or RELIABLE (1); drives delivery guarantee weight",
  qos_durability:          "QoS Durability policy — how long the broker retains messages for late-joining subscribers (VOLATILE < TRANSIENT_LOCAL < TRANSIENT < PERSISTENT)",
  qos_transport_priority:  "QoS Transport Priority — routing urgency level (LOW / MEDIUM / HIGH / URGENT)",
  subscriber_count:        "Number of active subscribers registered on this topic",
  // ── Operational weight ────────────────────────────────────────────────────
  weight:                  "Operational priority weight — QoS severity of topics routed through this component (0 = low, 1 = high)",
  path_count:              "Number of independent event-routing paths that share this dependency",
  dependency_weight_in:    "Sum of incoming dependency weights — total inbound QoS exposure",
  dependency_weight_out:   "Sum of outgoing dependency weights — total outbound QoS coupling",
  // ── Centrality ────────────────────────────────────────────────────────────
  pagerank:                "PageRank — global importance in the forward dependency graph; high value = many (important) dependents",
  reverse_pagerank:        "Reverse PageRank on G^T — fault-propagation reach; high value = failure spreads widely",
  betweenness:             "Betweenness centrality — fraction of shortest paths passing through this node; proxy for structural bottleneck",
  closeness:               "Closeness centrality — inverse average distance to all other nodes; high = fast change propagation",
  reverse_closeness:       "Reverse Closeness on G^T — adversarial propagation speed; how quickly a compromise reaches others",
  eigenvector:             "Eigenvector centrality — influence weighted by the importance of neighbours",
  reverse_eigenvector:     "Reverse Eigenvector on G^T — strategic attack reach; high value = many high-value nodes depend on this",
  // ── Degree ────────────────────────────────────────────────────────────────
  degree:                  "Normalised total degree (in + out) centrality",
  in_degree:               "Normalised in-degree — fraction of components that directly depend on this node",
  out_degree:              "Normalised out-degree — fraction of components this node directly depends on",
  in_degree_raw:           "Raw count of direct dependents (components that depend on this node)",
  out_degree_raw:          "Raw count of direct dependencies (components this node depends on)",
  // ── Resilience ────────────────────────────────────────────────────────────
  is_articulation_point:   "True if removing this node disconnects the graph — a structural Single Point of Failure (SPOF)",
  is_directed_ap:          "True if removing this node reduces the reachable set from source nodes — directed SPOF",
  is_isolated:             "True if this node has no connections at all",
  bridge_count:            "Number of bridge edges incident to this node (edges whose removal disconnects the graph)",
  bridge_ratio:            "Fraction of this node's incident edges that are bridges; high ratio = low redundancy",
  ap_c_directed:           "Directed articulation-point SPOF severity score ∈ [0,1]",
  cdi:                     "Connectivity Degradation Index — normalised increase in average path length when this node is removed",
  blast_radius:            "Number of nodes that become unreachable from the rest of the graph if this node fails",
  cascade_depth:           "Longest failure-propagation chain reachable from this node",
  clustering_coefficient:  "Local clustering coefficient — probability that two neighbours are also connected; higher = more redundant paths",
  publisher_spof:          "Sole-publisher risk — non-zero when this app is the only publisher on a high-weight topic",
  // ── Pub-sub topology ──────────────────────────────────────────────────────
  pubsub_degree:           "Number of topics this app publishes to or subscribes to in the pub-sub bipartite graph",
  pubsub_betweenness:      "Betweenness centrality computed on the pub-sub bipartite graph",
  broker_exposure:         "Number of distinct brokers routing topics touched by this component",
  fan_out_criticality:     "Topic fan-out criticality — number of distinct subscribers scaled by topic weight (Topic nodes only)",
  mpci:                    "Multi-Path Coupling Intensity — mean log-scaled path count across efferent dependencies; measures coupling via multiple routes",
  path_complexity:         "Efferent path count complexity: mean(log2(1 + path_count)) across outgoing dependencies",
  // ── Code size ─────────────────────────────────────────────────────────────
  loc:                     "Lines of Code — total source size of this component",
  loc_norm:                "Normalised Lines-of-Code within the component's type population (0 = smallest, 1 = largest)",
  cm_total_loc:            "Total lines of code across the component",
  cm_total_lines:          "Total lines of code across the component",
  cm_total_classes:        "Total number of classes defined in this component",
  cm_total_methods:        "Total number of methods across all classes in this component",
  cm_total_fields:         "Total number of fields (instance variables) across all classes in this component",
  duplicated_lines_density:"Percentage of lines that are duplicated within the codebase (SonarQube); higher = more copy-paste debt",
  // ── Complexity ────────────────────────────────────────────────────────────
  cyclomatic_complexity:   "McCabe's Cyclomatic Complexity — number of linearly independent paths through the code",
  complexity_norm:         "Normalised cyclomatic complexity within the component's type population (0 = simplest, 1 = most complex)",
  cm_avg_wmc:              "Average Weighted Methods per Class — sum of cyclomatic complexities of all methods, averaged across classes",
  cm_max_wmc:              "Maximum Weighted Methods per Class across all classes",
  cm_total_wmc:            "Total Weighted Methods per Class — aggregate cyclomatic complexity across the whole component",
  // ── Cohesion ──────────────────────────────────────────────────────────────
  lcom:                    "Raw Lack of Cohesion of Methods — measures how scattered the responsibilities of a class are (lower is better)",
  lcom_norm:               "Normalised Lack of Cohesion of Methods — how scattered responsibilities are (0 = cohesive, 1 = fully scattered)",
  cm_avg_lcom:             "Average Lack of Cohesion of Methods across classes — higher values indicate poorly cohesive classes",
  cm_max_lcom:             "Maximum Lack of Cohesion of Methods across all classes in this component",
  // ── Coupling ──────────────────────────────────────────────────────────────
  coupling_afferent:       "Afferent coupling (Ca) — number of external components that directly depend on this component; high = hard to change",
  coupling_efferent:       "Efferent coupling (Ce) — number of components this component directly depends on; high = many external dependencies",
  instability_code:        "Martin's Instability I = Ce/(Ca+Ce) — ratio of efferent to total couplings (0 = stable, 1 = unstable)",
  complexity_norm:         "Normalised cyclomatic complexity within the component's type population (0 = simplest, 1 = most complex)",
  coupling_risk:           "Coupling Risk = 1 − |2·Instability − 1| — maximised at 0.5 (deeply embedded on both sides); measures bi-directional coupling exposure",
  code_quality_penalty:    "Composite Code Quality Penalty (CQP) = 0.40·complexity + 0.35·instability + 0.25·LCOM; used as M(v) sub-score",
  cm_avg_cbo:              "Average Coupling Between Objects — number of classes this class is directly coupled to, averaged",
  cm_max_cbo:              "Maximum Coupling Between Objects across all classes",
  cm_avg_rfc:              "Average Response For Class — number of methods potentially executed in response to a message, averaged",
  cm_max_rfc:              "Maximum Response For Class across all classes",
  cm_avg_fanin:            "Average fan-in — average number of modules/classes that directly depend on each class",
  cm_max_fanin:            "Maximum fan-in across all classes in this component",
  cm_avg_fanout:           "Average fan-out — average number of modules/classes each class directly depends on",
  cm_max_fanout:           "Maximum fan-out across all classes in this component",
  // ── SonarQube quality ─────────────────────────────────────────────────────
  sqale_debt_ratio:        "SonarQube SQALE Debt Ratio — technical debt as a percentage of estimated time to rewrite the component from scratch (lower is better)",
  bugs:                    "Number of static bug issues detected by the static analyser (e.g. SonarQube)",
  vulnerabilities:         "Number of static security vulnerability issues detected by the static analyser",
  // ── Byte-sized fields ─────────────────────────────────────────────────────
  size:                    "Message payload size in bytes — used to compute the topic's contribution to the QoS weight (W_topic = 0.85·QoS + 0.15·size_norm)",
  message_size:            "Message payload size in bytes used in event-flow simulation",
  payload_size_bytes:      "Message payload size in bytes used in event-flow simulation",
  // ── Time fields ───────────────────────────────────────────────────────────
  deadline_ms:             "Maximum allowed end-to-end latency for this flow in milliseconds",
  latency_p50_ms:          "Median (p50) observed end-to-end message latency in milliseconds",
  latency_p95_ms:          "95th-percentile observed end-to-end message latency in milliseconds",
  latency_p99_ms:          "99th-percentile observed end-to-end message latency in milliseconds",
  // ── Queue / frequency ─────────────────────────────────────────────────────
  queue_size:              "Maximum number of messages this subscriber's queue can buffer before dropping",
  frequency:               "Publishing frequency in Hz (messages per second)",
  topic_frequency:         "Publishing frequency in Hz (messages per second)",
  criticality:             "Topic criticality level based on QoS weight",
  topic_criticality:       "Topic criticality level based on QoS weight",
}
const EDGE_DESCS: Record<string, string> = {
  weight:     "QoS severity weight [0–1] — maximum weight of topics routed through this dependency",
  path_count: "Number of independent event-routing paths that share this dependency",
}

// ── Contextual neighborhood scenarios (replaces raw depth 1-2-3) ─────────────
interface ConnScenario {
  id: string
  label: string
  tooltip: string
  depth: number
  /** If set, only edges of these types are shown */
  allowedEdgeTypes?: string[]
  /** If set, only nodes of these types are shown (center node always kept) */
  allowedNodeTypes?: string[]
  /**
   * When true, use a strict hop-by-hop BFS from the center so only nodes
   * reachable via the EXACT intended hop path are included. Prevents transitive
   * leakage where depth-2 queries pull in unrelated nodes via shared topics.
   */
  strictBFS?: boolean
}

const NODE_SCENARIOS: Record<string, ConnScenario[]> = {
  Application: [
    {
      id: "direct",  label: "Direct",
      tooltip: "Hosting node, pub/sub topics, and library dependencies",
      depth: 1,
      allowedNodeTypes: ["Application", "Node", "Topic", "Library"],
      allowedEdgeTypes: ["RUNS_ON", "PUBLISHES_TO", "SUBSCRIBES_TO", "USES"],
    },
    {
      id: "pubsub",  label: "Pub/Sub Context",
      tooltip: "Topics this app publishes/subscribes to, plus other apps sharing those topics",
      depth: 2,
      allowedNodeTypes: ["Application", "Topic"],
      allowedEdgeTypes: ["PUBLISHES_TO", "SUBSCRIBES_TO"],
      strictBFS: true,
    },
  ],
  Topic: [
    {
      id: "flows",  label: "Publishers & Subscribers",
      tooltip: "Applications that publish or subscribe to this topic",
      depth: 1,
      allowedNodeTypes: ["Application"],
      allowedEdgeTypes: ["PUBLISHES_TO", "SUBSCRIBES_TO"],
    },
    {
      id: "extended",  label: "Extended",
      tooltip: "Also includes infrastructure nodes of connected applications",
      depth: 2,
      allowedNodeTypes: ["Application", "Node", "Broker"],
      allowedEdgeTypes: ["PUBLISHES_TO", "SUBSCRIBES_TO", "RUNS_ON", "ROUTES"],
      strictBFS: true,
    },
  ],
  Node: [
    {
      id: "apps",  label: "Hosted Apps",
      tooltip: "Applications deployed on this infrastructure node",
      depth: 1,
      allowedNodeTypes: ["Application"],
      allowedEdgeTypes: ["RUNS_ON"],
    },
    {
      id: "ecosystem",  label: "App Ecosystem",
      tooltip: "Only apps that run on this node, plus topics those specific apps publish or subscribe to",
      depth: 2,
      allowedNodeTypes: ["Application", "Topic"],
      allowedEdgeTypes: ["RUNS_ON", "PUBLISHES_TO", "SUBSCRIBES_TO"],
      strictBFS: true,
    },
  ],
  Broker: [
    {
      id: "topics",  label: "Routed Topics",
      tooltip: "Topics routed through this message broker",
      depth: 1,
      allowedNodeTypes: ["Topic"],
      allowedEdgeTypes: ["ROUTES"],
    },
    {
      id: "context",  label: "App Connections",
      tooltip: "Routed topics plus applications that publish or subscribe to those topics",
      depth: 2,
      allowedNodeTypes: ["Topic", "Application"],
      allowedEdgeTypes: ["ROUTES", "PUBLISHES_TO", "SUBSCRIBES_TO"],
      strictBFS: true,
    },
  ],
  Library: [
    {
      id: "users",  label: "Dependent Apps",
      tooltip: "Applications that depend on this library",
      depth: 1,
      allowedNodeTypes: ["Application"],
      allowedEdgeTypes: ["USES"],
    },
    {
      id: "ecosystem",  label: "App Topics",
      tooltip: "Dependent apps plus the topics those specific apps publish or subscribe to",
      depth: 2,
      allowedNodeTypes: ["Application", "Topic"],
      allowedEdgeTypes: ["USES", "PUBLISHES_TO", "SUBSCRIBES_TO"],
      strictBFS: true,
    },
  ],
}
const DEFAULT_SCENARIOS: ConnScenario[] = [
  { id: "direct",   label: "Direct",   tooltip: "Direct neighbors (1 hop)",   depth: 1 },
  { id: "extended", label: "Extended", tooltip: "2-hop neighborhood",          depth: 2 },
]

function getScenariosForType(nodeType: string | undefined): ConnScenario[] {
  return NODE_SCENARIOS[nodeType ?? "Application"] ?? DEFAULT_SCENARIOS
}

/** Small ⓘ tooltip trigger shown inline after a property label */
function Tip({ text }: { text: string }) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          className="ml-1.5 inline-flex items-center justify-center w-3.5 h-3.5 rounded-full
            border border-muted-foreground/40 text-muted-foreground/60
            hover:border-foreground/50 hover:text-foreground/80 cursor-help transition-colors"
          style={{ fontSize: 8, lineHeight: 1, flexShrink: 0, verticalAlign: "middle" }}
        >i</span>
      </TooltipTrigger>
      <TooltipContent side="right" className="max-w-64 text-xs leading-relaxed">
        {text}
      </TooltipContent>
    </Tooltip>
  )
}

// ── Property unit suffixes & value formatting ─────────────────────────────────
function isBytesKey(key: string): boolean {
  return key === "size" || key === "message_size" || /bytes/.test(key)
}

/** Human-readable byte size: 41969 → "41.0 KB" */
function formatBytes(n: number): string {
  if (n < 1024)         return `${n} B`
  if (n < 1024 * 1024)  return `${(n / 1024).toFixed(1)} KB`
  if (n < 1024 ** 3)    return `${(n / 1024 ** 2).toFixed(2)} MB`
  return `${(n / 1024 ** 3).toFixed(2)} GB`
}

/** Formatted display value for a property (applies byte/etc formatting where meaningful) */
function propValue(key: string, v: unknown): string {
  const n = typeof v === "number" ? v : (typeof v === "string" && v !== "" && !isNaN(Number(v)) ? Number(v) : null)
  if (n !== null && isBytesKey(key)) return formatBytes(n)
  if (n !== null && !Number.isInteger(n)) return n.toFixed(2)
  return String(v)
}

function formatKey(key: string): string {
  return key
    .replace(/^cm_/, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, c => c.toUpperCase())
}

function propUnit(key: string): string {
  if (isBytesKey(key))                                return ""   // unit embedded by formatBytes

  // ── Strictly bounded [0–1] ─────────────────────────────────────────────────
  if (key === "weight")                               return "[0–1]"
  if (key === "lcom_norm")                            return "[0–1]"
  if (key === "instability_code")                     return "[0–1]"
  if (key === "complexity_norm")                      return "[0–1]"
  if (key === "loc_norm")                             return "[0–1]"
  if (key === "code_quality_penalty")                 return "[0–1]"
  if (/^coupling_risk/.test(key))                     return "[0–1]"
  if (/^coupling_/.test(key))                         return "[0–1]"
  if (key === "sqale_debt_ratio")                     return "%"
  if (key === "duplicated_lines_density")             return "%"

  // ── Topic fields ───────────────────────────────────────────────────────────
  if (key === "frequency" || key === "topic_frequency") return "Hz"

  // ── Centraliy / graph scores (all [0–1] normalised) ───────────────────────
  if (key === "pagerank")                             return "[0–1]"
  if (key === "reverse_pagerank")                     return "[0–1]"
  if (key === "betweenness")                          return "[0–1]"
  if (key === "closeness")                            return "[0–1]"
  if (key === "reverse_closeness")                    return "[0–1]"
  if (key === "eigenvector")                          return "[0–1]"
  if (key === "reverse_eigenvector")                  return "[0–1]"
  if (key === "degree")                               return "[0–1]"
  if (key === "in_degree")                            return "[0–1]"
  if (key === "out_degree")                           return "[0–1]"
  if (key === "bridge_ratio")                         return "[0–1]"
  if (key === "ap_c_directed")                        return "[0–1]"
  if (key === "cdi")                                  return "[0–1]"
  if (key === "clustering_coefficient")               return "[0–1]"
  if (key === "publisher_spof")                       return "[0–1]"
  if (key === "pubsub_betweenness")                   return "[0–1]"
  if (key === "mpci")                                 return "[0–1]"
  if (key === "path_complexity")                      return "[0–1]"
  if (key === "dependency_weight_in")                 return "[0–1]"
  if (key === "dependency_weight_out")                return "[0–1]"

  // ── Raw degree / topology counts ──────────────────────────────────────────
  if (key === "in_degree_raw")                        return "deps"
  if (key === "out_degree_raw")                       return "deps"
  if (key === "blast_radius")                         return "nodes"
  if (key === "cascade_depth")                        return "hops"
  if (key === "bridge_count")                         return "edges"
  if (key === "pubsub_degree")                        return "topics"
  if (key === "broker_exposure")                      return "brokers"
  if (key === "path_count")                           return "paths"
  if (key === "subscriber_count")                     return "subs"
  if (key === "fan_out_criticality")                  return "subs"
  if (key === "max_connections")                      return "conns"
  if (key === "cpu_cores")                            return "cores"
  if (key === "memory_gb")                            return "GB"

  // ── Time fields ───────────────────────────────────────────────────────────
  if (/_ms$/.test(key) || key === "deadline_ms")      return "ms"

  // ── Frequency ─────────────────────────────────────────────────────────────
  if (/_hz$/.test(key) || key === "frequency")        return "Hz"

  // ── Queue ─────────────────────────────────────────────────────────────────
  if (key === "queue_size")                           return "msgs"

  // ── Code size ─────────────────────────────────────────────────────────────
  if (key === "loc")                                  return "lines"
  if (key === "cyclomatic_complexity")                return "paths"
  if (/cm_total_(lines|loc)/.test(key))               return "lines"
  if (/cm_total_classes/.test(key))                   return "classes"
  if (/cm_total_methods/.test(key))                   return "methods"
  if (/cm_total_fields/.test(key))                    return "fields"
  if (/cm_total_wmc/.test(key))                       return "CC"
  if (/cm_(avg_|max_)wmc/.test(key))                  return "CC/class"

  // ── Cohesion ──────────────────────────────────────────────────────────────
  if (/cm_(avg_|max_)?lcom$|^lcom$/.test(key))        return "LCOM"

  // ── Coupling ──────────────────────────────────────────────────────────────
  if (/fanin/.test(key))                              return "in-deps"
  if (/fanout/.test(key))                             return "out-deps"
  if (/cbo/.test(key))                                return "classes"
  if (/rfc/.test(key))                                return "calls"
  if (key === "coupling_afferent")                    return "components"
  if (key === "coupling_efferent")                    return "components"

  // ── Issue counts ──────────────────────────────────────────────────────────
  if (key === "bugs")                                 return "issues"
  if (key === "vulnerabilities")                      return "issues"

  return ""
}

// ── Risk-range badge (good / bad judgement for [0–1] risk scores) ─────────────
/**
 * Returns true for keys whose value sits on a [0–1] risk scale where
 * higher = worse / more risky.  These keys get a colour-coded level badge.
 */
function isRiskKey(key: string): boolean {
  // RMAS quality / dimension scores
  if (/^(reliability|maintainability|availability|security|quality_score|rmav_score|rmas_score|overall)$/.test(key)) return true
  // Structural centrality metrics (all normalised to [0–1])
  if (/^(reverse_pagerank|betweenness_centrality|betweenness|bridge_ratio|bridge_score|reverse_eigenvector|reverse_closeness|ap_score|directed_ap_score|qspof)$/.test(key)) return true
  // Code quality [0–1] penalty inputs
  if (/^(lcom_norm|instability_code|complexity_norm|coupling_risk|weight)$/.test(key)) return true
  // Any coupling_ derived field
  if (/^coupling_/.test(key)) return true
  return false
}

const RISK_LEVELS = [
  { min: 0.75, label: "CRIT", cls: "bg-red-500/15 text-red-500" },
  { min: 0.50, label: "HIGH", cls: "bg-orange-500/15 text-orange-500" },
  { min: 0.25, label: "MOD",  cls: "bg-amber-500/15 text-amber-600 dark:text-amber-500" },
  { min: 0.00, label: "LOW",  cls: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-500" },
] as const

function getRiskLevel(n: number) {
  return RISK_LEVELS.find(r => n >= r.min) ?? RISK_LEVELS[RISK_LEVELS.length - 1]
}

function RiskBadge({ k, v }: { k: string; v: unknown }) {
  if (!isRiskKey(k)) return null
  const n = typeof v === "number" ? v
    : (typeof v === "string" && v !== "" && !isNaN(Number(v)) ? Number(v) : null)
  if (n === null || n < 0 || n > 1) return null
  const { label, cls } = getRiskLevel(n)
  return (
    <span className={`ml-2 inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-semibold tracking-wide ${cls}`}>
      {label}
    </span>
  )
}

// ── Shared ReactFlow prop constants (hoisted to avoid re-renders) ─────────────
const RF_NODE_ORIGIN: [number, number] = [0.5, 0.5]
const RF_FIT_VIEW_OPTIONS = { padding: 0.25 }
const RF_HIER_FIT_VIEW_OPTIONS = { padding: 0.12, minZoom: 0.25 }
const RF_STYLE_TRANSPARENT = { background: "transparent" }
const RF_STYLE_CONN = { background: "transparent", position: "relative" as const, zIndex: 1 }
const RF_PRO_OPTIONS = { hideAttribution: true }

// ── Graph Explorer ────────────────────────────────────────────────────────────

function buildDrillData(
  hierarchy: Record<string, CsmsGroup>,
  parent: HGNode | null,
): { nodes: HGNode[]; links: HGLink[] } {
  const nodes: HGNode[] = []
  const links: HGLink[] = []

  if (!parent) {
    // Root: show all CSMS nodes flat
    for (const [csmsKey, csms] of Object.entries(hierarchy)) {
      const appCount = Object.values(csms.css)
        .flatMap(c => Object.values(c.csci)).flatMap(ci => Object.values(ci.csc)).flatMap(c => c.apps).length
      nodes.push({ id: `csms:${csmsKey}`, name: csmsKey, level: "csms", appCount, pathKey: csmsKey })
    }
    return { nodes, links }
  }

  // Add the parent (anchor) node cleanly — strip d3 simulation fields (x/y/vx/vy)
  nodes.push({ id: parent.id, name: parent.name, level: parent.level, appCount: parent.appCount, pathKey: parent.pathKey })
  const p = parent.pathKey.split("/")

  switch (parent.level) {
    case "csms": {
      for (const [cssKey, css] of Object.entries(hierarchy[p[0]]?.css ?? {})) {
        const count = Object.values(css.csci).flatMap(ci => Object.values(ci.csc)).flatMap(c => c.apps).length
        const id = `css:${p[0]}/${cssKey}`
        nodes.push({ id, name: cssKey, level: "css", appCount: count, pathKey: `${p[0]}/${cssKey}` })
        links.push({ source: parent.id, target: id })
      }
      break
    }
    case "css": {
      for (const [csciKey, csci] of Object.entries(hierarchy[p[0]]?.css[p[1]]?.csci ?? {})) {
        const count = Object.values(csci.csc).flatMap(c => c.apps).length
        const id = `csci:${p[0]}/${p[1]}/${csciKey}`
        nodes.push({ id, name: csciKey, level: "csci", appCount: count, pathKey: `${p[0]}/${p[1]}/${csciKey}` })
        links.push({ source: parent.id, target: id })
      }
      break
    }
    case "csci": {
      for (const [cscKey, csc] of Object.entries(hierarchy[p[0]]?.css[p[1]]?.csci[p[2]]?.csc ?? {})) {
        const id = `csc:${p[0]}/${p[1]}/${p[2]}/${cscKey}`
        nodes.push({ id, name: cscKey, level: "csc", appCount: csc.apps.length, pathKey: `${p[0]}/${p[1]}/${p[2]}/${cscKey}` })
        links.push({ source: parent.id, target: id })
      }
      break
    }
    case "csc": {
      for (const app of hierarchy[p[0]]?.css[p[1]]?.csci[p[2]]?.csc[p[3]]?.apps ?? []) {
        const id = `app:${app.id}`
        nodes.push({ id, name: app.csu ?? app.name ?? app.id ?? "?", level: "app", appCount: 1, pathKey: app.id, appData: app })
        links.push({ source: parent.id, target: id })
      }
      break
    }
  }
  return { nodes, links }
}

// ── SVG-based connections graph (replaces react-force-graph-2d for swimlane view) ──────────────
function ConnSvgGraph({
  graphData, positions, dims, isDark, populatedLayers,
  selectedAppId, selectedLink, onNodeClick, onEdgeClick, onBackgroundClick,
}: {
  graphData: { nodes: any[]; links: any[] }
  positions: Map<string, { x: number; y: number }>
  dims: { width: number; height: number }
  isDark: boolean
  populatedLayers: Set<number>
  selectedAppId: string | null
  selectedLink: { link: any } | null
  onNodeClick: (node: any) => void
  onEdgeClick: (link: any, event: { clientX: number; clientY: number }) => void
  onBackgroundClick: () => void
}) {
  const W = dims.width || 800
  const H = dims.height || 600
  const svgRef = useRef<SVGSVGElement>(null)
  const [vp, setVp] = useState({ x: 0, y: 0, k: 1 })
  const dragRef = useRef<{ ox: number; oy: number; vx: number; vy: number } | null>(null)
  const movedRef = useRef(false)

  // Reset viewport when canvas size changes
  useEffect(() => { setVp({ x: 0, y: 0, k: 1 }) }, [W, H])

  // World coords (centered at 0,0) → screen pixel coords
  const sx = (wx: number) => wx + W / 2
  const sy = (wy: number) => wy + H / 2

  // Edge weight → stroke width
  const weightScale = useMemo(() => {
    const weights = graphData.links.map((l: any) => Number(l.weight ?? 1)).filter((w: number) => isFinite(w))
    if (!weights.length) return () => 1.5
    const lo = Math.min(...weights), hi = Math.max(...weights)
    if (lo === hi) return () => 2.0
    return (w: number) => 0.8 + ((w - lo) / (hi - lo)) * 3.2
  }, [graphData.links])

  // One SVG arrowhead marker per unique link color
  const markerColors = useMemo(() => {
    const s = new Set<string>()
    graphData.links.forEach((l: any) => s.add(linkTypeColor(l.type, isDark)))
    s.add("#f59e0b") // selected highlight colour
    return Array.from(s)
  }, [graphData.links, isDark])

  const mid = (color: string) => `arr-${color.replace(/[^0-9a-f]/gi, "x")}`

  function handleWheel(e: React.WheelEvent) {
    e.preventDefault()
    const rect = svgRef.current!.getBoundingClientRect()
    const mx = e.clientX - rect.left, my = e.clientY - rect.top
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12
    setVp(v => {
      const k = Math.max(0.15, Math.min(8, v.k * factor))
      return { k, x: mx - (mx - v.x) * (k / v.k), y: my - (my - v.y) * (k / v.k) }
    })
  }
  function handleMouseDown(e: React.MouseEvent) {
    movedRef.current = false
    dragRef.current = { ox: e.clientX, oy: e.clientY, vx: vp.x, vy: vp.y }
  }
  function handleMouseMove(e: React.MouseEvent) {
    if (!dragRef.current) return
    const dx = e.clientX - dragRef.current.ox, dy = e.clientY - dragRef.current.oy
    if (Math.abs(dx) + Math.abs(dy) > 4) movedRef.current = true
    setVp(v => ({ ...v, x: dragRef.current!.vx + dx, y: dragRef.current!.vy + dy }))
  }
  function handleMouseUp() { dragRef.current = null }

  return (
    <svg ref={svgRef} width={W} height={H}
      style={{ display: "block", cursor: dragRef.current ? "grabbing" : "grab" }}
      onWheel={handleWheel} onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}
      onClick={e => { if (!movedRef.current && e.target === svgRef.current) onBackgroundClick() }}
    >
      <defs>
        {markerColors.map(c => (
          <marker key={c} id={mid(c)} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <path d="M0,0.5 L0,6.5 L6,3.5 z" fill={c} />
          </marker>
        ))}
      </defs>
      <g transform={`translate(${vp.x},${vp.y}) scale(${vp.k})`}>
        {/* Swimlane bands */}
        {CONN_LAYER_Y_FRACS.map((yFrac, layer) => {
          if (!populatedLayers.has(layer)) return null
          const yPx = yFrac * H, bandH = H * 0.16
          const typeKey = Object.keys(CONN_TYPE_LAYER).find(t => CONN_TYPE_LAYER[t] === layer) ?? ""
          const tc = (isDark ? CONN_NODE_TYPE_COLORS_DARK : CONN_NODE_TYPE_COLORS_LIGHT)[typeKey] ?? "#888"
          return (
            <g key={layer}>
              <rect x={0} y={yPx - bandH / 2} width={W} height={bandH} fill={tc + "08"} />
              <line x1={0} y1={yPx - bandH / 2} x2={W} y2={yPx - bandH / 2} stroke={tc + "22"} strokeWidth={1} />
              <line x1={0} y1={yPx + bandH / 2} x2={W} y2={yPx + bandH / 2} stroke={tc + "22"} strokeWidth={1} />
              <text x={10} y={yPx} dominantBaseline="middle" fontSize={9} fontWeight="600"
                fill={tc + "aa"} style={{ userSelect: "none" as const, letterSpacing: "0.08em" }}>
                {String(CONN_LAYER_LABEL[layer] ?? "").toUpperCase()}
              </text>
            </g>
          )
        })}
        {/* Edges */}
        {graphData.links.map((link: any, i: number) => {
          const srcId = typeof link.source === "object" ? link.source.id : link.source
          const tgtId = typeof link.target === "object" ? link.target.id : link.target
          const sp = positions.get(srcId), ep = positions.get(tgtId)
          if (!sp || !ep) return null
          const x1 = sx(sp.x), y1 = sy(sp.y), x2 = sx(ep.x), y2 = sy(ep.y)
          const len = Math.hypot(x2 - x1, y2 - y1)
          if (len < 5) return null
          const ux = (x2 - x1) / len, uy = (y2 - y1) / len
          const sr = (srcId === selectedAppId ? 10 : 6) + 2
          const tr = (tgtId === selectedAppId ? 10 : 6) + 14
          const isHl = selectedLink?.link === link
          const color = isHl ? "#f59e0b" : linkTypeColor(link.type, isDark)
          const isAdj = srcId === selectedAppId || tgtId === selectedAppId
          const sw = isHl ? 5 : Math.min(isAdj ? weightScale(Number(link.weight ?? 1)) + 0.8 : weightScale(Number(link.weight ?? 1)), 5)
          return (
            <line key={i}
              x1={x1 + ux * sr} y1={y1 + uy * sr}
              x2={x2 - ux * tr} y2={y2 - uy * tr}
              stroke={color} strokeWidth={sw}
              markerEnd={`url(#${mid(color)})`}
              style={{ cursor: "pointer" }}
              onClick={e => { e.stopPropagation(); if (!movedRef.current) onEdgeClick(link, e) }}
            />
          )
        })}
        {/* Nodes */}
        {graphData.nodes.map((n: any) => {
          const pos = positions.get(n.id)
          if (!pos) return null
          const nx = sx(pos.x), ny = sy(pos.y)
          const isCenter = n.id === selectedAppId
          const r = isCenter ? 10 : 6
          const color = nodeTypeColor(n.type, isDark)
          const strokeColor = isCenter ? (isDark ? "#ffffff" : "#111111") : (isDark ? "rgba(255,255,255,0.3)" : "rgba(30,41,59,0.35)")
          const sw = isCenter ? 2.5 : 1
          let shape: React.ReactNode
          switch (n.type) {
            case "Node":
              shape = <rect x={nx - r} y={ny - r} width={r * 2} height={r * 2} fill={color} stroke={strokeColor} strokeWidth={sw} />
              break
            case "Topic":
              shape = <polygon points={`${nx},${ny - r} ${nx + r},${ny} ${nx},${ny + r} ${nx - r},${ny}`} fill={color} stroke={strokeColor} strokeWidth={sw} />
              break
            case "Library":
              shape = <polygon points={`${nx},${ny - r} ${nx + r},${ny + r * 0.6} ${nx - r},${ny + r * 0.6}`} fill={color} stroke={strokeColor} strokeWidth={sw} />
              break
            case "Broker": {
              const pts = Array.from({ length: 6 }, (_, i) => {
                const a = (Math.PI * 2 / 6) * i - Math.PI / 2
                return `${nx + r * Math.cos(a)},${ny + r * Math.sin(a)}`
              }).join(" ")
              shape = <polygon points={pts} fill={color} stroke={strokeColor} strokeWidth={sw} />
              break
            }
            default:
              shape = <>{isCenter && <circle cx={nx} cy={ny} r={r + 7} fill={color + "33"} />}<circle cx={nx} cy={ny} r={r} fill={color} stroke={strokeColor} strokeWidth={sw} /></>
          }
          const label = String(n.label ?? n.id ?? "?")
          const disp = label.length > 5 ? label.slice(0, 5) + "…" : label
          return (
            <g key={n.id} onClick={e => { e.stopPropagation(); if (!movedRef.current) onNodeClick(n) }} style={{ cursor: "pointer" }}>
              {shape}
              <text x={nx} y={ny + r + 11} textAnchor="middle" fontSize={10}
                fontWeight={isCenter ? "bold" : "normal"} fill={isDark ? "#e5e7eb" : "#374151"}>
                {disp}
              </text>
              <text x={nx} y={ny + r + 20} textAnchor="middle" fontSize={7} fill={isDark ? "#9ca3af" : "#6b7280"}>
                {n.type}
              </text>
              <circle cx={nx} cy={ny} r={r + 10} fill="transparent" />
            </g>
          )
        })}
      </g>
    </svg>
  )
}

// ── React Flow connections graph ─────────────────────────────────────────────
const ConnFlowNode = memo(function ConnFlowNode({ data }: NodeProps) {
  const { n, isDark, isCenter } = data as any
  const { zoom } = useViewport()
  // Scale labels inversely with zoom so they stay readable at any zoom level.
  // Never shrink below 1× (normal size when zoomed in), cap scale-up at 4×.
  const labelScale = Math.min(4, Math.max(1, 1 / Math.max(zoom, 0.25)))
  // Uniform canvas size for all shapes so labels align consistently
  const S  = isCenter ? 44 : 28   // bounding box side
  const C  = S / 2                // center coord
  const color = nodeTypeColor(n.type, isDark)
  const label = String(n.label ?? n.id ?? "?")
  const disp = label.length > 5 ? label.slice(0, 5) + "…" : label

  // Shared style tokens
  const fill   = color
  const stroke = isDark ? "rgba(255,255,255,0.18)" : "rgba(0,0,0,0.18)"
  const sw     = isCenter ? 2 : 1.5
  const glow   = color + (isDark ? "40" : "28")


  let shape: React.ReactNode
  switch (n.type) {

    // ── Application → Circle ─────────────────────────────────────────────────
    case "Application":
    default: {
      const r = C
      shape = <>
        <circle cx={C} cy={C} r={r + (isCenter ? 8 : 5)} fill={glow} />
        <circle cx={C} cy={C} r={r} fill={fill} stroke={stroke} strokeWidth={sw} />
      </>
      break
    }

    // ── Node → Square ────────────────────────────────────────────────────────
    case "Node": {
      // Squares appear optically larger than circles at same S — shrink by ~14%
      const sq  = S * 0.86
      const off = (S - sq) / 2
      const r   = isCenter ? 5 : 3
      shape = <>
        <rect x={off - 5} y={off - 5} width={sq + 10} height={sq + 10} rx={r + 3} fill={glow} />
        <rect x={off} y={off} width={sq} height={sq} rx={r} fill={fill} stroke={stroke} strokeWidth={sw} />
      </>
      break
    }

    // ── Topic → Rhombus ──────────────────────────────────────────────────────
    case "Topic": {
      shape = <>
        <polygon points={`${C},${-4} ${S + 4},${C} ${C},${S + 4} ${-4},${C}`} fill={glow} />
        <polygon points={`${C},0 ${S},${C} ${C},${S} 0,${C}`} fill={fill} stroke={stroke} strokeWidth={sw} />
      </>
      break
    }

    // ── Broker → Pentagon ────────────────────────────────────────────────────
    case "Broker": {
      const penta = (scale: number, ox = C, oy = C) =>
        Array.from({ length: 5 }, (_, i) => {
          const a = (Math.PI * 2 / 5) * i - Math.PI / 2
          return `${ox + scale * Math.cos(a)},${oy + scale * Math.sin(a)}`
        }).join(" ")
      shape = <>
        <polygon points={penta(C + 6)} fill={glow} />
        <polygon points={penta(C)} fill={fill} stroke={stroke} strokeWidth={sw} />
      </>
      break
    }

    // ── Library → Triangle ───────────────────────────────────────────────────
    case "Library": {
      const h = S * 0.866
      const yOff = (S - h) / 2
      shape = <>
        <polygon points={`${C},${yOff - 6} ${S + 7},${yOff + h + 4} ${-7},${yOff + h + 4}`} fill={glow} />
        <polygon points={`${C},${yOff} ${S},${yOff + h} 0,${yOff + h}`} fill={fill} stroke={stroke} strokeWidth={sw} />
      </>
      break
    }
  }

  const hs = { width: 1, height: 1, opacity: 0, border: "none", background: "transparent", minWidth: 0, minHeight: 0 }
  const textBg    = isDark ? "rgba(14,14,22,0.85)" : "rgba(255,255,255,0.90)"
  const textColor = isDark ? "#f1f5f9" : "#1e293b"
  const subColor  = isDark ? "#94a3b8" : "#64748b"
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", cursor: "pointer", userSelect: "none" }}>
      <Handle type="target" position={Position.Top}    id="t-top" style={hs} />
      <Handle type="source" position={Position.Top}    id="s-top" style={hs} />
      <svg width={S} height={S} overflow="visible" style={{ display: "block" }}>
        {shape}
        {isCenter && (
          <circle cx={C} cy={C} r={C + 10} fill="none"
            stroke={isDark ? "rgba(255,255,255,0.65)" : "rgba(0,0,0,0.50)"}
            strokeWidth={1.5} strokeDasharray="4 3" />
        )}
      </svg>
      <div style={{ transform: `scale(${labelScale})`, transformOrigin: "top center", display: "flex", flexDirection: "column", alignItems: "center" }}>
        <div style={{
          marginTop: 6,
          padding: "2px 9px",
          background: textBg,
          border: `1px solid ${color}44`,
          borderRadius: 20,
          backdropFilter: "blur(8px)",
          fontSize: isCenter ? 12 : 10,
          fontWeight: isCenter ? 700 : 500,
          color: textColor,
          whiteSpace: "nowrap",
          maxWidth: 160,
          overflow: "hidden",
          textOverflow: "ellipsis",
          lineHeight: "1.5",
          letterSpacing: "0.01em",
          boxShadow: `0 2px 6px ${isDark ? "rgba(0,0,0,0.6)" : "rgba(0,0,0,0.10)"}`,
        }}>{disp}</div>
        <div style={{
          marginTop: 2,
          fontSize: 8,
          fontWeight: 600,
          color: color,
          opacity: 0.75,
          whiteSpace: "nowrap",
          letterSpacing: "0.10em",
          textTransform: "uppercase",
        }}>{n.type}</div>
      </div>
      <Handle type="source" position={Position.Bottom} id="s-bot" style={hs} />
      <Handle type="target" position={Position.Bottom} id="t-bot" style={hs} />
    </div>
  )
})

// Swimlane background node — rendered inside RF so it moves with fitView
const ConnLaneNode = memo(function ConnLaneNode({ data }: NodeProps) {
  const { color, width, height, isDark } = data as any
  return (
    <div style={{
      width, height,
      borderTop: `1px solid ${color}1a`,
      borderBottom: `1px solid ${color}1a`,
      background: isDark
        ? `linear-gradient(90deg, ${color}18 0%, ${color}08 25%, transparent 70%)`
        : `linear-gradient(90deg, ${color}12 0%, ${color}05 25%, transparent 70%)`,
      pointerEvents: "none",
    }} />
  )
})
const cfNodeTypes = { conn: ConnFlowNode, lane: ConnLaneNode }

// Custom edge: smooth bezier + clean fixed-size filled arrowhead
const ConnFlowEdge = memo(function ConnFlowEdge({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, style, data }: EdgeProps) {
  // Fixed arrowhead dimensions — same size on every edge
  const AW = 5  // half-width
  const AH = 8  // height
  const color = (style?.stroke as string) ?? "#888"
  const sw = Number(style?.strokeWidth ?? 2)
  const opacity = Number(style?.opacity ?? 1)
  const dashArray = (style as any)?.strokeDasharray as string | undefined
  const curveOffset = Number((data as any)?.curveOffset ?? 0)
  // Handles are always top/bottom — bezier arrives vertically at target
  const goingDown = targetPosition === "top"
  const dir = goingDown ? 1 : -1
  // Shorten path so the line ends at arrowhead base, not tip — no overlap
  const adjustedTY = targetY - dir * AH
  // Build path: when curveOffset != 0 use a manually offset cubic bezier so
  // parallel edges between the same pair of nodes don't overlap.
  let edgePath: string
  if (curveOffset === 0) {
    ;[edgePath] = getBezierPath({
      sourceX, sourceY, sourcePosition,
      targetX, targetY: adjustedTY,
      targetPosition, curvature: 0.35,
    })
  } else {
    const midY = (sourceY + adjustedTY) / 2
    const bendY = Math.abs(adjustedTY - sourceY) * 0.35
    edgePath = [
      `M ${sourceX} ${sourceY}`,
      `C ${sourceX + curveOffset} ${sourceY + bendY}`,
      `  ${targetX + curveOffset} ${adjustedTY - bendY}`,
      `  ${targetX} ${adjustedTY}`,
    ].join(" ")
    void midY // suppress unused warning
  }
  const tipY  = targetY
  const baseY = adjustedTY
  const arrowPts = `${targetX},${tipY} ${targetX - AW},${baseY} ${targetX + AW},${baseY}`
  return (
    <g opacity={opacity}>
      <path d={edgePath} fill="none" stroke="transparent" strokeWidth={Math.max(sw + 10, 14)} />
      <path d={edgePath} fill="none" stroke={color} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round" strokeDasharray={dashArray} />
      <polygon points={arrowPts} fill={color} />
    </g>
  )
})
const cfEdgeTypes = { conn: ConnFlowEdge }

// ── Hierarchy graph using @xyflow/react ──────────────────────────────────────

const HIER_LEVEL_LABEL: Record<HGLevel, string> = {
  csms: "System", css: "Segment", csci: "Config Item", csc: "Component", app: "App",
}

const HierFlowNode = memo(function HierFlowNode({ data }: NodeProps) {
  const { n, isDark, isSelected, isParent } = data as any
  const hn = n as HGNode
  const color = NODE_COLORS[hn.level]
  const levelLabel = HIER_LEVEL_LABEL[hn.level] ?? hn.level
  const { zoom } = useViewport()
  // Keep labels readable at any zoom level — scale up when zoomed out, never shrink below 1×
  const labelScale = Math.min(4, Math.max(1, 1 / Math.max(zoom, 0.25)))

  // Parent (drilled-into) node is larger
  const pad = isParent ? "8px 18px" : "6px 14px"
  const fontSize = isParent ? 14 : 12
  const maxLabelWidth = isParent ? 220 : 180
  const dotSize = isParent ? 9 : 7

  const bgAlpha = isDark ? (isParent ? "30" : "1a") : (isParent ? "20" : "12")
  const borderAlpha = isSelected ? "" : (isParent ? "88" : "44")
  const borderColor = isSelected ? color : `${color}${borderAlpha}`
  const shadow = isSelected
    ? `0 0 0 2.5px ${color}50, 0 4px 16px ${color}30`
    : isParent
    ? `0 2px 12px ${color}25`
    : `0 1px 4px ${isDark ? "rgba(0,0,0,0.4)" : "rgba(0,0,0,0.08)"}`

  const hiddenHandle = { opacity: 0, width: 0, height: 0, minWidth: 0 }

  const nodeContent = (
    <div
      style={{
        display: "flex", flexDirection: "column", alignItems: "center", gap: 3,
        padding: pad,
        borderRadius: 12,
        background: isDark ? `${color}${bgAlpha}` : `${color}${bgAlpha}`,
        border: `1.5px solid ${borderColor}`,
        boxShadow: shadow,
        cursor: "pointer",
        userSelect: "none",
        whiteSpace: "nowrap",
        backdropFilter: "blur(8px)",
        transition: "box-shadow 0.15s, border-color 0.15s",
        transform: `scale(${labelScale})`,
        transformOrigin: "center center",
      }}
    >
      <Handle type="target" position={Position.Top} id="t" style={hiddenHandle} />
      {/* Level badge */}
      <span style={{
        fontSize: 8, fontWeight: 700, letterSpacing: "0.08em",
        textTransform: "uppercase", color: isDark ? `${color}99` : `${color}aa`,
        lineHeight: 1,
      }}>{levelLabel}</span>
      {/* Name row */}
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{
          width: dotSize, height: dotSize, borderRadius: "50%", background: color,
          flexShrink: 0, display: "inline-block",
          boxShadow: `0 0 6px ${color}66`,
        }} />
        <span style={{
          maxWidth: maxLabelWidth, overflow: "hidden", textOverflow: "ellipsis",
          fontSize, fontWeight: isParent ? 700 : isSelected ? 600 : 500,
          color: isDark ? "#f1f5f9" : "#1e293b",
          lineHeight: 1.3,
        }}>{hn.name}</span>
      </div>
      {/* Count badge */}
      {hn.appCount > 0 && hn.level !== "app" && (
        <span style={{
          fontSize: 9, fontWeight: 600,
          color: isDark ? "#94a3b8" : "#64748b",
          background: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.04)",
          borderRadius: 6, padding: "1px 6px",
          lineHeight: 1.4,
        }}>{hn.appCount} {hn.appCount === 1 ? "app" : "apps"}</span>
      )}
      <Handle type="source" position={Position.Bottom} id="s" style={hiddenHandle} />
    </div>
  )

  if (hn.level === "app" && hn.appData) {
    return nodeContent
  }
  return nodeContent
})

// Custom hierarchy edge with gradient stroke
const HierFlowEdge = memo(function HierFlowEdge({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, data }: EdgeProps) {
  const { isDark, sourceColor, targetColor } = (data ?? {}) as any
  const sc = sourceColor ?? (isDark ? "#ffffff" : "#000000")
  const tc = targetColor ?? (isDark ? "#ffffff" : "#000000")
  const gradId = `hg-${Math.round(sourceX)}-${Math.round(targetX)}`

  // Vertical tree edge: straight down from source, then curve to target
  const midY = sourceY + (targetY - sourceY) * 0.5
  const d = `M ${sourceX} ${sourceY} C ${sourceX} ${midY}, ${targetX} ${midY}, ${targetX} ${targetY}`

  return (
    <g>
      <defs>
        <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={sc} stopOpacity={isDark ? 0.5 : 0.35} />
          <stop offset="100%" stopColor={tc} stopOpacity={isDark ? 0.3 : 0.2} />
        </linearGradient>
      </defs>
      <path d={d} fill="none" stroke={`url(#${gradId})`} strokeWidth={2} strokeLinecap="round" />
    </g>
  )
})

const cfHierNodeTypes = { hier: HierFlowNode }
const cfHierEdgeTypes = { hier: HierFlowEdge }

const HierFlowGraph = memo(function HierFlowGraph({ graphData, dims, isDark, selectedNodeId, onNodeClick }: {
  graphData: { nodes: HGNode[]; links: HGLink[] }
  dims: { width: number; height: number }
  isDark: boolean
  selectedNodeId: string | null
  onNodeClick: (n: HGNode) => void
}) {
  const W = dims.width  || 800
  const H = dims.height || 600
  const LEVEL_ORDER: HGLevel[] = ["csms", "css", "csci", "csc", "app"]

  // Find which node is the parent (drilled-into) node — first node that has children
  const parentNodeId = useMemo(() => {
    if (graphData.nodes.length <= 1) return null
    const childIds = new Set(graphData.links.map(l => typeof l.target === "object" ? (l.target as any).id : l.target))
    return graphData.nodes.find(n => !childIds.has(n.id))?.id ?? graphData.nodes[0]?.id ?? null
  }, [graphData.nodes, graphData.links])

  // Tree layout: parent centered at top, children spaced evenly below
  const positions = useMemo(() => {
    const map = new Map<string, { x: number; y: number }>()
    const byLevel = new Map<HGLevel, HGNode[]>()
    for (const n of graphData.nodes) {
      if (!byLevel.has(n.level)) byLevel.set(n.level, [])
      byLevel.get(n.level)!.push(n)
    }

    // Collect present levels in order
    const presentLevels = LEVEL_ORDER.filter(l => byLevel.has(l))
    if (presentLevels.length === 0) return map

    // Vertical spacing: fixed minimum per tier so levels don't stack up on large trees
    const yStep = presentLevels.length === 1 ? 0 : Math.max(220, H / (presentLevels.length + 0.5))
    const yStart = presentLevels.length === 1 ? H * 0.4 : yStep * 0.75

    for (let li = 0; li < presentLevels.length; li++) {
      const level = presentLevels[li]
      const arr = byLevel.get(level) ?? []
      const y = yStart + li * yStep

      if (arr.length === 1) {
        // Single node (root or only child) — center it
        map.set(arr[0].id, { x: W / 2, y })
      } else {
        // Equal spacing with enough room for node cards (~200px wide)
        const minGap = Math.max(500, 400 + arr.length * 10)
        const naturalWidth = (arr.length - 1) * minGap
        const margin = Math.max(W * 0.12, 100)
        const usable = Math.max(W - 2 * margin, naturalWidth)
        // If children need more space than canvas, let fitView handle zoom
        const startX = (W - usable) / 2
        arr.forEach((n, i) => {
          const x = arr.length === 1 ? W / 2 : startX + (i / (arr.length - 1)) * usable
          map.set(n.id, { x, y })
        })
      }
    }
    return map
  }, [graphData.nodes, W, H])

  // Build node-to-color map for edge gradients
  const nodeColorMap = useMemo(() => {
    const m = new Map<string, string>()
    for (const n of graphData.nodes) m.set(n.id, NODE_COLORS[n.level] ?? "#888")
    return m
  }, [graphData.nodes])

  const initialHierNodes = useMemo(() => graphData.nodes.map(n => ({
    id: n.id,
    type: "hier" as const,
    position: positions.get(n.id) ?? { x: W / 2, y: H / 2 },
    data: { n, isDark, isSelected: n.id === selectedNodeId, isParent: n.id === parentNodeId },
    selectable: false,
    draggable: true,
  })), [graphData.nodes, positions, isDark, selectedNodeId, parentNodeId])

  // Only reset node positions when the set of nodes changes (new data loaded).
  // Dims/theme/selection changes must NOT reset positions — that would discard user drags.
  const graphNodesKeyHier = useMemo(() => graphData.nodes.map(n => n.id).join(","), [graphData.nodes])
  const lastHierKeyRef = useRef("")
  const [rfNodes, setRfNodes] = useState(initialHierNodes)
  useEffect(() => {
    if (graphNodesKeyHier !== lastHierKeyRef.current) {
      lastHierKeyRef.current = graphNodesKeyHier
      setRfNodes(initialHierNodes)
    }
  }, [graphNodesKeyHier, initialHierNodes])
  // Update theme/selection styling in-place without touching positions
  useEffect(() => {
    setRfNodes(nds => nds.map(nd => ({
      ...nd,
      data: { ...nd.data, isDark, isSelected: nd.data.n?.id === selectedNodeId, isParent: nd.data.n?.id === parentNodeId },
    })))
  }, [isDark, selectedNodeId, parentNodeId])
  const onNodesChange = useCallback(
    (changes: NodeChange[]) => setRfNodes(nds => applyNodeChanges(changes, nds) as any),
    [],
  )

  const rfEdges = useMemo(() => graphData.links.map((l: any, i: number) => {
    const srcId = typeof l.source === "object" ? l.source.id : l.source
    const tgtId = typeof l.target === "object" ? l.target.id : l.target
    return {
      id: `he${i}`,
      source: srcId,
      target: tgtId,
      type: "hier",
      data: { isDark, sourceColor: nodeColorMap.get(srcId), targetColor: nodeColorMap.get(tgtId) },
    }
  }), [graphData.links, isDark, nodeColorMap])

  const handleNodeClick = useCallback((_: any, rfNode: any) => onNodeClick(rfNode.data.n as HGNode), [onNodeClick])

  return (
    <ReactFlow
      nodes={rfNodes}
      edges={rfEdges}
      nodeTypes={cfHierNodeTypes}
      edgeTypes={cfHierEdgeTypes}
      nodeOrigin={RF_NODE_ORIGIN}
      fitView
      fitViewOptions={RF_HIER_FIT_VIEW_OPTIONS}
      minZoom={0.08}
      nodesDraggable
      onNodesChange={onNodesChange}
      onNodeClick={handleNodeClick}
      panOnDrag
      zoomOnScroll
      style={RF_STYLE_TRANSPARENT}
      proOptions={RF_PRO_OPTIONS}
    >
      <Background variant={BackgroundVariant.Dots} gap={24} size={1}
        color={isDark ? "#ffffff10" : "#00000010"} />
    </ReactFlow>
  )
})

// ── ECharts connections tree ──────────────────────────────────────────────────

// ── ECharts connections tree ──────────────────────────────────────────────────

const ConnEChartsGraph = memo(function ConnEChartsGraph({ graphData, dims, isDark, selectedAppId, onNodeClick, onBackgroundClick }: {
  graphData: { nodes: any[]; links: any[] }
  dims: { width: number; height: number }
  isDark: boolean
  selectedAppId: string | null
  selectedLink: { link: any } | null
  onNodeClick: (n: any) => void
  onEdgeClick: (link: any, event: React.MouseEvent) => void
  onBackgroundClick: () => void
}) {
  const W = dims.width || 800
  const H = dims.height || 600

  // Build edge-type-grouped tree:  root → [EdgeTypeGroup…] → target nodes
  const treeData = useMemo(() => {
    const nodes: any[] = graphData.nodes
    const links: any[] = graphData.links
    if (nodes.length === 0) return null

    const rootNode = nodes.find(n => n.id === selectedAppId) ?? nodes[0]
    const rootId = rootNode.id
    const rootColor = nodeTypeColor(rootNode.type, isDark)
    const rootLabel: string = rootNode.label ?? rootNode.name ?? rootId
    const trimRoot = rootLabel.length > 5 ? rootLabel.slice(0, 5) + "…" : rootLabel
    const nodeById = new Map(nodes.map(n => [n.id, n]))

    // Group links by edge type; each entry: { edgeType, peer node, direction }
    const edgeGroups = new Map<string, Array<{ node: any; dir: "out" | "in" }>>()
    for (const l of links) {
      const srcId = l.source?.id ?? l.source
      const tgtId = l.target?.id ?? l.target
      const type: string = l.type ?? "CONNECTED"

      if (srcId === rootId) {
        const peer = nodeById.get(tgtId)
        if (!peer) continue
        if (!edgeGroups.has(type)) edgeGroups.set(type, [])
        edgeGroups.get(type)!.push({ node: peer, dir: "out" })
      } else if (tgtId === rootId) {
        const peer = nodeById.get(srcId)
        if (!peer) continue
        if (!edgeGroups.has(type)) edgeGroups.set(type, [])
        edgeGroups.get(type)!.push({ node: peer, dir: "in" })
      }
    }

    // Sort groups: known types first (in CONN_LINK_TYPE_COLORS order), then unknown alphabetically
    const knownOrder = Object.keys(isDark ? CONN_LINK_TYPE_COLORS_DARK : CONN_LINK_TYPE_COLORS_LIGHT)
    const sortedEdgeTypes = Array.from(edgeGroups.keys()).sort((a, b) => {
      const ai = knownOrder.indexOf(a)
      const bi = knownOrder.indexOf(b)
      if (ai !== -1 && bi !== -1) return ai - bi
      if (ai !== -1) return -1
      if (bi !== -1) return 1
      return a.localeCompare(b)
    })

    const children = sortedEdgeTypes.map(edgeType => {
      const entries = edgeGroups.get(edgeType)!
      // De-duplicate peers (a node may appear via multiple edges of same type)
      const seen = new Set<string>()
      const unique = entries.filter(e => { if (seen.has(e.node.id)) return false; seen.add(e.node.id); return true })
      unique.sort((a, b) => (a.node.label ?? a.node.id ?? "").localeCompare(b.node.label ?? b.node.id ?? ""))

      const edgeColor = linkTypeColor(edgeType, isDark)

      const leafNodes = unique.map(({ node, dir }) => {
        const lbl: string = node.label ?? node.name ?? node.id ?? ""
        const trimLbl = lbl.length > 5 ? lbl.slice(0, 5) + "…" : lbl
        const nc = nodeTypeColor(node.type, isDark)
        return {
          name: trimLbl,
          value: node.id,
          _raw: node,
          _isGroup: false,
          _dir: dir,
          symbolSize: 10,
          itemStyle: { color: nc, borderWidth: 0 },
          lineStyle: { color: edgeColor + "88" },
          label: {
            fontSize: 9,
            fontWeight: 400,
            color: isDark ? "#a1a1aa" : "#6b7280",
          },
        }
      })

      return {
        name: `${edgeType}  (${unique.length})`,
        value: `__group__${edgeType}`,
        _raw: null,
        _isGroup: true,
        symbol: "none",
        symbolSize: 0,
        itemStyle: {
          color: edgeColor,
          borderWidth: 0,
        },
        label: {
          fontSize: 9,
          fontWeight: 700,
          color: edgeColor,
        },
        children: leafNodes,
      }
    })

    return {
      name: trimRoot,
      value: rootId,
      _raw: rootNode,
      _isGroup: false,
      symbolSize: 20,
      itemStyle: {
        color: rootColor,
        borderColor: isDark ? "#ffffff" : "#1e293b",
        borderWidth: 2.5,
        shadowBlur: 14,
        shadowColor: rootColor + "99",
      },
      label: {
        fontSize: 12,
        fontWeight: 700,
        color: isDark ? "#e5e7eb" : "#1e293b",
      },
      children,
    }
  }, [graphData, selectedAppId, isDark])

  const option = useMemo(() => {
    if (!treeData) return {}
    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "item",
        triggerOn: "mousemove",
        backgroundColor: isDark ? "#1c1c1e" : "#ffffff",
        borderColor: isDark ? "#3f3f46" : "#e4e4e7",
        textStyle: { color: isDark ? "#fafafa" : "#09090b", fontSize: 12 },
        formatter: (params: any) => {
          const d = params.data
          if (d?._isGroup) return `<b>${d.name}</b>`
          const n = d?._raw
          if (!n) return ""
          const typeLabel: string = n.type ?? "Node"
          const lbl = n.label ?? n.name ?? n.id ?? ""
          const get = (key: string) => n.properties?.[key] ?? n[key]
          let extra = ""
          if (typeLabel === "Application") {
            const role = get("role"); if (role != null && role !== "") extra += `<br/><span style="opacity:0.7">Role: ${role}</span>`
            const priority = get("priority"); if (priority != null && priority !== "") extra += `<br/><span style="opacity:0.7">Priority: ${priority}</span>`
            const hotstandby = get("hotstandby"); if (hotstandby) extra += `<br/><span style="opacity:0.7">Hot Standby: true</span>`
          } else if (typeLabel === "Topic") {
            const qr = get("qos_reliability"); if (qr != null && qr !== "") extra += `<br/><span style="opacity:0.7">Reliability: ${qr}</span>`
            const qd = get("qos_durability");  if (qd != null && qd !== "") extra += `<br/><span style="opacity:0.7">Durability: ${qd}</span>`
            const qt = get("qos_transport_priority"); if (qt != null && qt !== "") extra += `<br/><span style="opacity:0.7">Transport Priority: ${qt}</span>`
            const szRaw = get("message_size") ?? get("payload_size_bytes") ?? get("size")
            if (szRaw != null && szRaw !== "") { const szN = Number(szRaw); const szFmt = isFinite(szN) ? (szN >= 1048576 ? `${(szN/1048576).toFixed(2)} MB` : szN >= 1024 ? `${(szN/1024).toFixed(1)} KB` : `${szN} B`) : String(szRaw); extra += `<br/><span style="opacity:0.7">Size: ${szFmt}</span>` }
          } else if (typeLabel === "Library") {
            const ver = get("version"); if (ver != null && ver !== "") extra += `<br/><span style="opacity:0.7">Version: ${ver}</span>`
          } else if (typeLabel === "Broker") {
            const bt = get("broker_type"); if (bt != null && bt !== "") extra += `<br/><span style="opacity:0.7">Protocol: ${bt}</span>`
          }
          return `<div style="font-size:12px;line-height:1.7"><b>${lbl}</b><br/><span style="opacity:0.7">${typeLabel}</span>${extra}</div>`
        },
      },
      series: [{
        type: "tree",
        data: [treeData],
        top: "6%",
        left: "3%",
        bottom: "6%",
        right: "3%",
        orient: "TB",
        expandAndCollapse: true,
        initialTreeDepth: -1,
        roam: true,
        lineStyle: {
          width: 1.2,
          curveness: 0.4,
          color: isDark ? "rgba(255,255,255,0.18)" : "rgba(0,0,0,0.14)",
        },
        label: {
          position: "bottom",
          verticalAlign: "top",
          align: "center",
          fontSize: 9,
          color: isDark ? "#d4d4d8" : "#374151",
          formatter: (p: any) => {
            const name = p.name ?? ""
            return name.length > 5 ? name.slice(0, 5) + "…" : name
          },
        },
        leaves: {
          label: {
            position: "bottom",
            verticalAlign: "top",
            align: "center",
          },
        },
        emphasis: {
          focus: "descendant",
          itemStyle: { shadowBlur: 10 },
        },
        animationDuration: 350,
        animationDurationUpdate: 250,
      }],
    }
  }, [treeData, isDark])

  const onEvents = useMemo(() => ({
    click: (params: any) => {
      if (params.componentType === "series" && params.data?._raw && !params.data._isGroup) {
        onNodeClick(params.data._raw)
      } else if (!params.data || params.data._isGroup) {
        // group label click → collapse/expand handled by ECharts; no nav
      }
    },
  }), [onNodeClick])

  if (!treeData) return (
    <div style={{ width: W, height: H, display: "flex", alignItems: "center", justifyContent: "center" }}>
      <span style={{ fontSize: 13, color: isDark ? "#71717a" : "#a1a1aa" }}>Select a node to view connections</span>
    </div>
  )

  return (
    <div style={{ width: W, height: H, position: "relative" }}>
      {/* Legend */}
      <div style={{
        position: "absolute", bottom: 10, left: 10, zIndex: 10,
        display: "flex", flexWrap: "wrap", gap: "6px 12px",
        padding: "5px 10px",
        borderRadius: 8,
        background: isDark ? "rgba(15,15,20,0.70)" : "rgba(255,255,255,0.80)",
        backdropFilter: "blur(8px)",
        border: `1px solid ${isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)"}`,
        fontSize: 9,
        color: isDark ? "#94a3b8" : "#64748b",
        pointerEvents: "none",
      }}>
        {Object.keys(isDark ? CONN_LINK_TYPE_COLORS_DARK : CONN_LINK_TYPE_COLORS_LIGHT).map(type => (
          <span key={type} style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ width: 18, height: 2, background: linkTypeColor(type, isDark), flexShrink: 0, borderRadius: 1 }} />
            {type}
          </span>
        ))}
      </div>
      {/* Hint */}
      <div style={{
        position: "absolute", top: 8, right: 8, zIndex: 10,
        fontSize: 10, color: isDark ? "#52525b" : "#a1a1aa",
        pointerEvents: "none",
      }}>
        Click groups to collapse · Scroll to zoom · Drag to pan
      </div>
      <ReactECharts
        option={option}
        style={{ width: W, height: H }}
        onEvents={onEvents}
        notMerge={true}
        theme={isDark ? "dark" : undefined}
      />
    </div>
  )
})

const ConnFlowGraph = memo(function ConnFlowGraph({ graphData, positions, dims, isDark, populatedLayers, selectedAppId, selectedLink, onNodeClick, onEdgeClick, onBackgroundClick }: {
  graphData: { nodes: any[]; links: any[] }
  positions: Map<string, { x: number; y: number }>
  dims: { width: number; height: number }
  isDark: boolean
  populatedLayers: Set<number>
  selectedAppId: string | null
  selectedLink: { link: any } | null
  onNodeClick: (n: any) => void
  onEdgeClick: (link: any, event: React.MouseEvent) => void
  onBackgroundClick: () => void
}) {
  const W = dims.width  || 800
  const H = dims.height || 600
  const bandH = H * 0.20
  // Track viewport so sticky lane labels can be repositioned on pan/zoom
  const [vp, setVp] = useState({ x: 0, y: 0, zoom: 1 })

  const initialNodes = useMemo(() => {
    return graphData.nodes.map((n: any) => {
      const pos = positions.get(n.id) ?? { x: W / 2, y: H / 2 }
      return {
        id: n.id,
        type: "conn" as const,
        position: pos,
        data: { n, isDark, isCenter: n.id === selectedAppId },
        selectable: false,
        draggable: true,
        zIndex: 1,
      }
    })
  }, [graphData.nodes, positions, isDark, selectedAppId, dims, populatedLayers])

  // Only reset node positions when the set of nodes changes (new data loaded).
  // Dims/theme/selection changes must NOT reset positions — that would discard user drags.
  const graphNodesKeyConn = useMemo(() => graphData.nodes.map((n: any) => n.id).join(","), [graphData.nodes])
  const lastConnKeyRef = useRef("")
  const [rfNodes, setRfNodes] = useState(initialNodes)
  useEffect(() => {
    if (graphNodesKeyConn !== lastConnKeyRef.current) {
      lastConnKeyRef.current = graphNodesKeyConn
      setRfNodes(initialNodes)
    }
  }, [graphNodesKeyConn, initialNodes])
  // Update theme/selectedApp in-place without touching positions
  useEffect(() => {
    setRfNodes(nds => nds.map((nd: any) => ({
      ...nd,
      data: { ...nd.data, isDark, isCenter: nd.id === selectedAppId },
    })))
  }, [isDark, selectedAppId])
  const onNodesChange = useCallback(
    (changes: NodeChange[]) => setRfNodes(nds => applyNodeChanges(changes, nds) as any),
    [],
  )

  const rfEdges = useMemo(() => {
    const weights = graphData.links.map((l: any) => Number(l.weight ?? 1)).filter((w: number) => isFinite(w))
    const lo = weights.length ? Math.min(...weights) : 0
    const hi = weights.length ? Math.max(...weights) : 1
    const wScale = lo === hi ? () => 2.0 : (w: number) => 1.0 + ((w - lo) / (hi - lo)) * 2.5

    // Detect parallel edges (same node pair, any direction) and assign lateral offsets
    // so they fan out and don't overlap.
    const pairGroups = new Map<string, number[]>()
    graphData.links.forEach((l: any, i: number) => {
      const a = typeof l.source === "object" ? l.source.id : l.source
      const b = typeof l.target === "object" ? l.target.id : l.target
      const key = a < b ? `${a}||${b}` : `${b}||${a}`
      if (!pairGroups.has(key)) pairGroups.set(key, [])
      pairGroups.get(key)!.push(i)
    })
    const OFFSET_STEP = 28 // px between parallel edges

    return graphData.links.map((l: any, i: number) => {
      const srcId = typeof l.source === "object" ? l.source.id : l.source
      const tgtId = typeof l.target === "object" ? l.target.id : l.target
      const srcY = positions.get(srcId)?.y ?? 0
      const tgtY = positions.get(tgtId)?.y ?? 0
      const goingDown = tgtY >= srcY
      const sourceHandle = goingDown ? "s-bot" : "s-top"
      const targetHandle = goingDown ? "t-top" : "t-bot"
      const isHl = selectedLink?.link === l
      const isAdj = srcId === selectedAppId || tgtId === selectedAppId
      const isDerived = l.type === "DEPENDS_ON"
      const color = isHl ? "#f59e0b" : linkTypeColor(l.type, isDark)
      const sw = isHl ? 6 : isAdj ? Math.min(wScale(Number(l.weight ?? 1)) + 0.5, 4) : 1
      const opacity = isHl ? 1 : isAdj ? 0.88 : 0.2

      // Compute lateral curve offset for this edge within its parallel group
      const a = srcId < tgtId ? srcId : tgtId
      const b = srcId < tgtId ? tgtId : srcId
      const pairKey = `${a}||${b}`
      const group = pairGroups.get(pairKey) ?? [i]
      const groupIdx = group.indexOf(i)
      const n = group.length
      const curveOffset = n === 1 ? 0 : (groupIdx - (n - 1) / 2) * OFFSET_STEP

      return {
        id: `e${i}`,
        source: srcId,
        target: tgtId,
        sourceHandle,
        targetHandle,
        type: "conn",
        style: { stroke: color, strokeWidth: sw, opacity, ...(isDerived ? { strokeDasharray: "6 4" } : {}) },
        data: { link: l, curveOffset },
        selectable: false,
      }
    })
  }, [graphData.links, positions, isDark, selectedLink, selectedAppId])

  const handleNodeClick = useCallback((_: any, node: any) => onNodeClick((node.data as any).n), [onNodeClick])
  const handleEdgeClick = useCallback((event: any, edge: any) => onEdgeClick((edge.data as any).link, event), [onEdgeClick])

  return (
    <div style={{ width: dims.width, height: dims.height, position: "relative" }}>
      <ReactFlow
        nodes={rfNodes}
        edges={rfEdges as any}
        nodeTypes={cfNodeTypes}
        edgeTypes={cfEdgeTypes}
        nodeOrigin={RF_NODE_ORIGIN}
        fitView
        fitViewOptions={RF_FIT_VIEW_OPTIONS}
        nodesDraggable
        nodesConnectable={false}
        elementsSelectable={false}
        onNodesChange={onNodesChange}
        onNodeClick={handleNodeClick}
        onEdgeClick={handleEdgeClick}
        onPaneClick={onBackgroundClick}
        onViewportChange={setVp}
        style={RF_STYLE_CONN}
        proOptions={RF_PRO_OPTIONS}
      >
        <Background variant={BackgroundVariant.Dots} color={isDark ? "#3f3f46" : "#d4d4d8"} gap={28} size={1.5} />
      </ReactFlow>
      {/* Sticky swimlane bands + labels — rendered outside ReactFlow's transform
          so they stay fixed at the left edge regardless of pan/zoom.
          Band Y tracks zoom vertically; bands always span full width. */}
      <div style={{ position: "absolute", top: 0, left: 0, width: "100%", height: "100%", pointerEvents: "none", overflow: "hidden" }}>
        {CONN_LAYER_Y_FRACS.map((yFrac, layer) => {
          if (!populatedLayers.has(layer)) return null
          const typeKey = Object.keys(CONN_TYPE_LAYER).find(t => CONN_TYPE_LAYER[t] === layer) ?? ""
          const color = (isDark ? CONN_NODE_TYPE_COLORS_DARK : CONN_NODE_TYPE_COLORS_LIGHT)[typeKey] ?? "#888"
          // Convert world-space lane centre → screen-space Y
          const screenY = yFrac * H * vp.zoom + vp.y
          // Band height also scales with zoom so it always matches the node spacing
          const bandHScreen = H * 0.20 * vp.zoom
          return (
            <div key={layer} style={{
              position: "absolute",
              left: 0,
              width: "100%",
              top: screenY - bandHScreen / 2,
              height: bandHScreen,
              borderTop: `1px solid ${color}22`,
              borderBottom: `1px solid ${color}22`,
              background: isDark
                ? `linear-gradient(90deg, ${color}18 0%, ${color}08 25%, transparent 70%)`
                : `linear-gradient(90deg, ${color}12 0%, ${color}05 25%, transparent 70%)`,
            }}>
              {/* Label pinned to left edge */}
              <div style={{
                position: "absolute",
                left: 0,
                top: "50%",
                transform: "translateY(-50%)",
                display: "flex",
                alignItems: "center",
                gap: 5,
                paddingLeft: 10,
              }}>
                <div style={{ width: 3, height: 18, borderRadius: 2, background: color, opacity: 0.6, flexShrink: 0 }} />
                <span style={{
                  fontSize: 9, fontWeight: 700, letterSpacing: "0.12em",
                  textTransform: "uppercase", color, opacity: 0.75, userSelect: "none",
                  textShadow: isDark ? "0 1px 4px rgba(0,0,0,0.9)" : "0 1px 4px rgba(255,255,255,0.9)",
                }}>{CONN_LAYER_LABEL[layer] ?? ""}</span>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
})

// ── ECharts full-tree view ────────────────────────────────────────────────────

function buildEChartsTree(hierarchy: Record<string, CsmsGroup>): object {
  return {
    name: "System",
    itemStyle: { color: "#64748b" },
    label: { show: false },
    children: sortKeys(Object.keys(hierarchy)).map(csmsKey => {
      const csms = hierarchy[csmsKey]
      return {
        name: csms.name,
        itemStyle: { color: NODE_COLORS.csms },
        lineStyle: { color: NODE_COLORS.csms + "88" },
        children: sortKeys(Object.keys(csms.css)).map(cssKey => {
          const css = csms.css[cssKey]
          return {
            name: css.name,
            itemStyle: { color: NODE_COLORS.css },
            lineStyle: { color: NODE_COLORS.css + "88" },
            children: sortKeys(Object.keys(css.csci)).map(csciKey => {
              const csci = css.csci[csciKey]
              return {
                name: csci.name,
                itemStyle: { color: NODE_COLORS.csci },
                lineStyle: { color: NODE_COLORS.csci + "88" },
                children: sortKeys(Object.keys(csci.csc)).map(cscKey => {
                  const csc = csci.csc[cscKey]
                  return {
                    name: csc.name,
                    itemStyle: { color: NODE_COLORS.csc },
                    lineStyle: { color: NODE_COLORS.csc + "88" },
                    children: csc.apps.map(app => ({
                      name: app.csu ?? app.name ?? app.id ?? "?",
                      value: app.weight,
                      itemStyle: { color: NODE_COLORS.app },
                      lineStyle: { color: NODE_COLORS.app + "66" },
                      // carry raw app data for tooltip
                      _app: app,
                    })),
                  }
                }),
              }
            }),
          }
        }),
      }
    }),
  }
}

// ── Merged hierarchy + connections tree ──────────────────────────────────────

function buildConnSubtree(
  centerNodeId: string,
  connData: { nodes: any[]; links: any[] },
  expandedLeaves: Map<string, { nodes: any[]; links: any[] }>,
  isDark: boolean,
  parentPath: string = "",
): any[] {
  if (!centerNodeId || connData.nodes.length === 0) return []
  const nodeById = new Map(connData.nodes.map((n: any) => [n.id, n]))
  const groups = new Map<string, Array<{ node: any; dir: "out" | "in" }>>()
  for (const l of connData.links) {
    const src = l.source?.id ?? l.source
    const tgt = l.target?.id ?? l.target
    const type: string = l.type ?? "CONNECTED"
    if (src === centerNodeId) {
      const peer = nodeById.get(tgt); if (!peer) continue
      if (!groups.has(type)) groups.set(type, [])
      groups.get(type)!.push({ node: peer, dir: "out" })
    } else if (tgt === centerNodeId) {
      const peer = nodeById.get(src); if (!peer) continue
      if (!groups.has(type)) groups.set(type, [])
      groups.get(type)!.push({ node: peer, dir: "in" })
    }
  }
  const knownOrder = Object.keys(isDark ? CONN_LINK_TYPE_COLORS_DARK : CONN_LINK_TYPE_COLORS_LIGHT)
  const sortedTypes = Array.from(groups.keys()).sort((a, b) => {
    const ai = knownOrder.indexOf(a); const bi = knownOrder.indexOf(b)
    if (ai !== -1 && bi !== -1) return ai - bi
    if (ai !== -1) return -1; if (bi !== -1) return 1
    return a.localeCompare(b)
  })
  return sortedTypes.map(edgeType => {
    const entries = groups.get(edgeType)!
    const seen = new Set<string>()
    const unique = entries.filter(e => { if (seen.has(e.node.id)) return false; seen.add(e.node.id); return true })
    unique.sort((a, b) => (a.node.label ?? a.node.id ?? "").localeCompare(b.node.label ?? b.node.id ?? ""))
    const ec = linkTypeColor(edgeType, isDark)
    const groupPath = `${parentPath}/${edgeType}`
    const groupId = `cg:${groupPath}:${centerNodeId}`
    return {
      id: groupId,
      name: `${edgeType}  (${unique.length})\x00${groupId}`,
      value: `__cg__${edgeType}__${centerNodeId}`,
      _isConnGroup: true,
      collapsed: false,
      symbol: "none",
      symbolSize: 0,
      itemStyle: { color: ec, borderWidth: 0 },
      lineStyle: { color: ec + "88" },
      label: { fontSize: 11, fontWeight: 700, color: ec },
      children: unique.map(({ node, dir }) => {
        const lbl: string = node.label ?? node.name ?? node.id ?? ""
        const trimLbl = lbl.length > 26 ? lbl.slice(0, 24) + "…" : lbl
        const nc = nodeTypeColor(node.type, isDark)
        const instanceKey = `${groupPath}:${node.id}`
        const leafId = `cl:${instanceKey}`
        const leafData = expandedLeaves.get(instanceKey)
        return {
          id: leafId,
          name: trimLbl + `\x00${leafId}`,
          value: node.id,
          _raw: node,
          _isConnLeaf: true,
          _dir: dir,
          collapsed: !leafData,
          symbol: "circle",
          symbolSize: 10,
          itemStyle: { color: nc, borderWidth: leafData ? 1.5 : 0, ...(leafData ? { borderColor: isDark ? "#e5e7eb" : "#374151" } : {}) },
          lineStyle: { color: ec + "88" },
          label: { fontSize: 11, color: isDark ? "#a1a1aa" : "#6b7280" },
          ...(leafData ? { children: buildConnSubtree(node.id, leafData, expandedLeaves, isDark, instanceKey) } : {}),
        }
      }),
    }
  })
}

function buildMergedTree(
  hierarchy: Record<string, CsmsGroup>,
  selectedInstanceKey: string | null,
  selectedPathKey: string | null,
  connDataMap: Map<string, { nodes: any[]; links: any[] }>,
  expandedLeaves: Map<string, { nodes: any[]; links: any[] }>,
  isDark: boolean,
  focusHierPathKey?: string | null,
): object {
  const fp = focusHierPathKey?.split("/") ?? null
  // Find ancestor paths for ALL apps that have connection data loaded
  const ancMap = new Map<string, { csmsKey: string; cssKey: string; csciKey: string; cscKey: string }>()
  for (const [ck, csms] of Object.entries(hierarchy)) {
    for (const [sk, css] of Object.entries(csms.css)) {
      for (const [ik, csci] of Object.entries(css.csci)) {
        for (const [pk, csc] of Object.entries(csci.csc)) {
          for (const app of csc.apps) {
            const instKey = `app:${ck}/${sk}/${ik}/${pk}/${app.id}`
            const hasData = connDataMap.has(instKey)
            const isSelected = selectedInstanceKey ? instKey === selectedInstanceKey : app.id === selectedPathKey
            if (hasData || isSelected) {
              ancMap.set(instKey, { csmsKey: ck, cssKey: sk, csciKey: ik, cscKey: pk })
            }
          }
        }
      }
    }
  }

  return {
    id: "root",
    name: "System",
    symbol: "none",
    symbolSize: 0,
    itemStyle: { color: "transparent", borderWidth: 0, opacity: 0 },
    lineStyle: { opacity: 0 },
    label: { show: false },
    children: sortKeys(Object.keys(hierarchy)).map(csmsKey => {
      const csms = hierarchy[csmsKey]
      const csmsId = `csms:${csmsKey}`
      const isCsms = Array.from(ancMap.values()).some(a => a.csmsKey === csmsKey) || (fp != null && fp[0] === csmsKey)
      return {
        id: csmsId,
        name: csms.name + `\x00${csmsId}`,
        _level: "csms",
        ...(isCsms && { collapsed: false }),
        itemStyle: { color: NODE_COLORS.csms },
        // Hide the edge from the invisible synthetic root to this CSMS
        // (ECharts tree styles parent→child edges via the child's lineStyle).
        lineStyle: { opacity: 0, width: 0 },
        children: sortKeys(Object.keys(csms.css)).map(cssKey => {
          const css = csms.css[cssKey]
          const cssId = `css:${csmsKey}/${cssKey}`
          const isCss = isCsms && (Array.from(ancMap.values()).some(a => a.csmsKey === csmsKey && a.cssKey === cssKey) || (fp != null && fp[0] === csmsKey && fp[1] === cssKey))
          return {
            id: cssId,
            name: css.name + `\x00${cssId}`,
            _level: "css",
            ...(isCss && { collapsed: false }),
            itemStyle: { color: NODE_COLORS.css },
            lineStyle: { color: NODE_COLORS.css + "88" },
            children: sortKeys(Object.keys(css.csci)).map(csciKey => {
              const csci = css.csci[csciKey]
              const csciId = `csci:${csmsKey}/${cssKey}/${csciKey}`
              const isCsci = isCss && (Array.from(ancMap.values()).some(a => a.csmsKey === csmsKey && a.cssKey === cssKey && a.csciKey === csciKey) || (fp != null && fp[0] === csmsKey && fp[1] === cssKey && fp[2] === csciKey))
              return {
                id: csciId,
                name: csci.name + `\x00${csciId}`,
                _level: "csci",
                ...(isCsci && { collapsed: false }),
                itemStyle: { color: NODE_COLORS.csci },
                lineStyle: { color: NODE_COLORS.csci + "88" },
                children: sortKeys(Object.keys(csci.csc)).map(cscKey => {
                  const csc = csci.csc[cscKey]
                  const cscId = `csc:${csmsKey}/${cssKey}/${csciKey}/${cscKey}`
                  const isCsc = isCsci && (Array.from(ancMap.values()).some(a => a.csmsKey === csmsKey && a.cssKey === cssKey && a.csciKey === csciKey && a.cscKey === cscKey) || (fp != null && fp[0] === csmsKey && fp[1] === cssKey && fp[2] === csciKey && fp[3] === cscKey))
                  return {
                    id: cscId,
                    name: csc.name + `\x00${cscId}`,
                    _level: "csc",
                    ...(isCsc && { collapsed: false }),
                    itemStyle: { color: NODE_COLORS.csc },
                    lineStyle: { color: NODE_COLORS.csc + "88" },
                    children: csc.apps.map(app => {
                      const instancePathKey = `app:${csmsKey}/${cssKey}/${csciKey}/${cscKey}/${app.id}`
                      const isSel = selectedInstanceKey
                        ? instancePathKey === selectedInstanceKey
                        : selectedPathKey !== null && app.id === selectedPathKey
                      const appConnData = connDataMap.get(instancePathKey)
                      const appChildren = appConnData ? buildConnSubtree(app.id, appConnData, expandedLeaves, isDark, instancePathKey) : []
                      return {
                        id: instancePathKey,
                        name: (app.csu ?? app.name ?? app.id ?? "?") + `\x00${instancePathKey}`,
                        value: app.weight,
                        _app: app,
                        _level: "app",
                        ...(appConnData && { collapsed: false }),
                        itemStyle: {
                          color: NODE_COLORS.app,
                          ...(isSel ? { borderColor: isDark ? "#fff" : "#1e293b", borderWidth: 2, shadowBlur: 10, shadowColor: NODE_COLORS.app + "99" } : {}),
                        },
                        lineStyle: { color: NODE_COLORS.app + "66" },
                        ...(appChildren.length > 0 ? { children: appChildren } : {}),
                      }
                    }),
                  }
                }),
              }
            }),
          }
        }),
      }
    }),
  }
}

const MergedEChartsTree = memo(function MergedEChartsTree({
  hierarchy, connDataMap, expandedLeaves, selectedApp, dims, isDark, focusHierPathKey, onAppNodeClick, onConnNodeClick, exportFnRef,
}: {
  hierarchy: Record<string, CsmsGroup>
  connDataMap: Map<string, { nodes: any[]; links: any[] }>
  expandedLeaves: Map<string, { nodes: any[]; links: any[] }>
  selectedApp: HGNode | null
  dims: { width: number; height: number }
  isDark: boolean
  focusHierPathKey?: string | null
  onAppNodeClick: (app: AppNode, instanceKey: string) => void
  onConnNodeClick: (nodeId: string, instanceKey: string) => void
  exportFnRef?: React.MutableRefObject<(() => void) | null>
}) {
  const W = dims.width || 800
  const H = dims.height || 600

  // ECharts instance ref for PNG export (stores the echarts instance directly via onChartReady)
  const echartsRef = useRef<any>(null)
  useEffect(() => {
    if (!exportFnRef) return
    exportFnRef.current = () => {
      const ec = echartsRef.current
      if (!ec) return
      const dataUrl = ec.getDataURL({ type: "png", backgroundColor: isDark ? "#09090b" : "#ffffff", pixelRatio: 2 })
      const a = document.createElement("a")
      a.href = dataUrl
      a.download = "system-export.png"
      a.click()
    }
  }, [exportFnRef, isDark])

  // Horizontal spread multiplier. 1.0 = fits viewport with a 5% margin per side
  // (normal layout); >1 widens the tree beyond the viewport so spacing between
  // nodes grows. Panning enabled via roam: true. Controlled by the slider below.
  const [spread, setSpread] = useState(1)

  // Track which nodes have labels shown — only newly expanded nodes and their ancestors
  const [nodesToShowLabel, setNodesToShowLabel] = useState<Set<string>>(new Set())
  const prevExpandedLeavesRef = useRef(expandedLeaves)

  // Update nodesToShowLabel when expandedLeaves changes
  useEffect(() => {
    const prevKeys = new Set(prevExpandedLeavesRef.current.keys())
    const currKeys = new Set(expandedLeaves.keys())

    // Find newly expanded nodes (in current but not in previous)
    const newlyExpanded = new Set<string>()
    for (const key of currKeys) {
      if (!prevKeys.has(key)) {
        newlyExpanded.add(key)
      }
    }

    // If there are no new expansions, don't update state
    if (newlyExpanded.size === 0) {
      prevExpandedLeavesRef.current = expandedLeaves
      return
    }

    // For each newly expanded node, add it and its ancestors to the label set
    const toShow = new Set<string>(nodesToShowLabel)
    for (const nodeKey of newlyExpanded) {
      // Add the newly expanded leaf node (format: "parentPath:nodeId")
      const leafId = `cl:${nodeKey}`
      toShow.add(leafId)

      // Add ancestors by parsing the parent path
      // expandedLeaves keys are like "app:csms/css/csci/csc/appId:connNodeId"
      const parts = nodeKey.split(":")
      if (parts.length >= 2) {
        const parentPath = parts[0] === "app" ? parts[1] : parts.slice(0, -1).join(":")
        // parentPath is like "csms/css/csci/csc/appId" or "parentPath:nodeId"
        const pathSegments = parentPath.split("/")

        // Add each level of the hierarchy
        if (pathSegments.length >= 5) {
          // Full app path: csms/css/csci/csc/appId
          const [csmsKey, cssKey, csciKey, cscKey, appId] = pathSegments
          toShow.add(`app:${csmsKey}/${cssKey}/${csciKey}/${cscKey}/${appId}`)
          toShow.add(`csc:${csmsKey}/${cssKey}/${csciKey}/${cscKey}`)
          toShow.add(`csci:${csmsKey}/${cssKey}/${csciKey}`)
          toShow.add(`css:${csmsKey}/${cssKey}`)
          toShow.add(`csms:${csmsKey}`)
        } else if (parentPath.includes(":")) {
          // Nested connection leaf path: extract app id and add ancestors
          const segments = parentPath.split(":")[0].split("/")
          if (segments.length >= 5) {
            const [csmsKey, cssKey, csciKey, cscKey, appId] = segments
            toShow.add(`app:${csmsKey}/${cssKey}/${csciKey}/${cscKey}/${appId}`)
            toShow.add(`csc:${csmsKey}/${cssKey}/${csciKey}/${cscKey}`)
            toShow.add(`csci:${csmsKey}/${cssKey}/${csciKey}`)
            toShow.add(`css:${csmsKey}/${cssKey}`)
            toShow.add(`csms:${csmsKey}`)
          }
        }
      }
    }

    setNodesToShowLabel(toShow)
    prevExpandedLeavesRef.current = expandedLeaves
  }, [expandedLeaves])

  const treeData = useMemo(
    () => buildMergedTree(hierarchy, selectedApp?.instanceKey ?? null, selectedApp?.pathKey ?? null, connDataMap, expandedLeaves, isDark, focusHierPathKey),
    [hierarchy, selectedApp?.instanceKey, selectedApp?.pathKey, connDataMap, expandedLeaves, isDark, focusHierPathKey],
  )

  const option = useMemo(() => {
    const sharedProps = {
      type: "tree",
      orient: "TB",
      expandAndCollapse: true,
      initialTreeDepth: 2,
      roam: true,
      symbol: "circle",
      itemStyle: { borderWidth: 0 },
      lineStyle: { width: 1.2, curveness: 0.5, opacity: 0.6 },
      symbolSize: (_val: number, params: any) => {
        // treeAncestors includes the invisible "System" root — subtract 2 so CSMS = depth 0
        const depth = (params.treeAncestors?.length ?? 2) - 2
        // Monotonically decreasing top → bottom: CSMS, CSS, CSCI, CSC, App
        return ([40, 32, 26, 20, 16] as number[])[Math.min(Math.max(depth, 0), 4)] ?? 16
      },
      label: {
        position: "bottom",
        verticalAlign: "top",
        align: "center",
        fontSize: 14,
        color: isDark ? "#e5e7eb" : "#374151",
        show: false, // Hide by default
        formatter: (params: any) => {
          const raw: string = params.name ?? ""
          const name = raw.split("\x00")[0]
          return name
        },
      },
      leaves: {
        label: { position: "bottom", verticalAlign: "top", align: "center", fontSize: 11, show: false },
      },
      emphasis: { focus: "descendant", itemStyle: { shadowBlur: 8 } },
      animationDuration: 350,
      animationDurationUpdate: 250,
    }
    return {
      backgroundColor: "transparent",
      tooltip: {
        trigger: "item",
        backgroundColor: isDark ? "#1c1c1e" : "#ffffff",
        borderColor: isDark ? "#3f3f46" : "#e4e4e7",
        textStyle: { color: isDark ? "#fafafa" : "#09090b", fontSize: 12 },
        formatter: (params: any) => {
          const d = params.data
          const dispName = (s: string) => s?.split("\x00")[0] ?? s
          if (d?._isConnGroup) return dispName(d.name)
          if (d?._app) {
            const app = d._app
            const name = dispName(d.name)
            const levelLabel = LEVEL_LABELS[d._level] ?? "Application"
            const role = app.role ?? app.properties?.role
            const roleStr = (role != null && role !== "") ? `<br/><span style="opacity:0.7">Role: ${role}</span>` : ""
            const priority = app.priority ?? app.properties?.priority
            const priorityStr = (priority != null && priority !== "") ? `<br/><span style="opacity:0.7">Priority: ${priority}</span>` : ""
            const hotstandbyStr = app.hotstandby || app.properties?.hotstandby ? `<br/><span style="opacity:0.7">Hot Standby: true</span>` : ""
            return `<div style="font-size:12px;line-height:1.7"><b>${name}</b><br/><span style="opacity:0.7">${levelLabel}</span>${roleStr}${priorityStr}${hotstandbyStr}</div>`
          }
          if (d?._level && !d?._isConnLeaf) {
            const name = dispName(d.name)
            const levelLabel = LEVEL_LABELS[d._level] ?? d._level
            return `<div style="font-size:12px;line-height:1.7"><b>${name}</b><br/><span style="opacity:0.7">${levelLabel}</span></div>`
          }
          if (d?._isConnLeaf && d._raw) {
            const n = d._raw
            const name = dispName(d.name)
            const type: string = n.type ?? ""
            let extra = ""
            if (type === "Topic") {
              const qr = n.properties?.qos_reliability ?? n.qos_reliability
              const qd = n.properties?.qos_durability ?? n.qos_durability
              if (qr != null && qr !== "") extra += `<br/><span style="opacity:0.7">Reliability: ${qr}</span>`
              if (qd != null && qd !== "") extra += `<br/><span style="opacity:0.7">Durability: ${qd}</span>`
              const qt = n.properties?.qos_transport_priority ?? n.qos_transport_priority
              if (qt != null && qt !== "") extra += `<br/><span style="opacity:0.7">Transport Priority: ${qt}</span>`
              const szRaw = n.properties?.message_size ?? n.message_size ?? n.properties?.payload_size_bytes ?? n.payload_size_bytes ?? n.properties?.size ?? n.size
              if (szRaw != null && szRaw !== "") { const szN = Number(szRaw); const szFmt = isFinite(szN) ? (szN >= 1048576 ? `${(szN/1048576).toFixed(2)} MB` : szN >= 1024 ? `${(szN/1024).toFixed(1)} KB` : `${szN} B`) : String(szRaw); extra += `<br/><span style="opacity:0.7">Size: ${szFmt}</span>` }
            } else if (type === "Library") {
              const ver = n.properties?.version ?? n.version
              if (ver != null && ver !== "") extra += `<br/><span style="opacity:0.7">Version: ${ver}</span>`
            } else if (type === "Application") {
              const role = n.properties?.role ?? n.role
              if (role != null && role !== "") extra += `<br/><span style="opacity:0.7">Role: ${role}</span>`
              const priority = n.properties?.priority ?? n.priority
              if (priority != null && priority !== "") extra += `<br/><span style="opacity:0.7">Priority: ${priority}</span>`
              const hotstandby = n.properties?.hotstandby ?? n.hotstandby
              if (hotstandby) extra += `<br/><span style="opacity:0.7">Hot Standby: true</span>`
            } else if (type === "Broker") {
              const bt = n.properties?.broker_type ?? n.broker_type
              if (bt != null && bt !== "") extra += `<br/><span style="opacity:0.7">Protocol: ${bt}</span>`
            }
            return `<div style="font-size:12px;line-height:1.7"><b>${name}</b><br/><span style="opacity:0.7">${type}</span>${extra}</div>`
          }
          return dispName(d?.name ?? "")
        },
      },
      series: [{
        ...sharedProps,
        data: [treeData],
        // spread=1 → 5% margin per side (normal viewport fit). Increasing spread
        // grows the tree's bounding box beyond the viewport (negative margins),
        // which proportionally increases horizontal spacing between sibling nodes.
        left: `${50 - 45 * spread}%`,
        right: `${50 - 45 * spread}%`,
        top: "5%",
        bottom: "5%",
        // Override label rendering to show labels only for newly expanded nodes and their parents
        label: {
          ...sharedProps.label,
          show: (params: any) => {
            const nodeId = params.data?.id
            // Always show labels for connection leaves and groups
            if (params.data?._isConnLeaf || params.data?._isConnGroup) return true
            // Show labels for nodes marked to display
            return nodeId && nodesToShowLabel.has(nodeId)
          },
        },
      }],
    }
  }, [treeData, isDark, spread, nodesToShowLabel])

  const onEvents = useMemo(() => ({
    click: (params: any) => {
      const d = params.data
      if (!d) return
      if (d._isConnLeaf) {
        const rawKey: string = (d.name as string).includes('\x00') ? (d.name as string).split('\x00')[1] : `cl:${d.value}`
        // strip the "cl:" prefix that was added for ECharts node uniqueness
        const instanceKey = rawKey.startsWith('cl:') ? rawKey.slice(3) : rawKey
        onConnNodeClick(String(d.value ?? ''), instanceKey)
        return
      }
      if (d._isConnGroup) return
      if (d._app) {
        const instanceKey: string = (d.name as string).includes('\x00') ? (d.name as string).split('\x00')[1] : `app:${d._app.id}`
        onAppNodeClick(d._app, instanceKey)
        return
      }
    },
  }), [onAppNodeClick, onConnNodeClick])

  return (
    <div style={{ width: W, height: H, position: "relative" }}>
      {W > 0 && H > 0 && (
        <ReactECharts
          onChartReady={(instance: any) => { echartsRef.current = instance }}
          key={isDark ? "dark" : "light"}
          option={option}
          style={{ width: W, height: H }}
          onEvents={onEvents}
          notMerge={false}
          lazyUpdate
          theme={isDark ? "dark" : undefined}
        />
      )}
      {/* Spread slider — controls horizontal distance between nodes */}
      <div style={{
        position: "absolute", bottom: 12, right: 12, zIndex: 10,
        display: "flex", alignItems: "center", gap: 8,
        padding: "6px 10px",
        borderRadius: 8,
        width: 224,
        background: isDark ? "rgba(15,15,20,0.70)" : "rgba(255,255,255,0.80)",
        backdropFilter: "blur(8px)",
        border: `1px solid ${isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)"}`,
        fontSize: 11,
        color: isDark ? "#e4e4e7" : "#3f3f46",
        boxSizing: "border-box",
      }}>
        <span style={{ fontSize: 10, fontWeight: 600, opacity: 0.8, flexShrink: 0 }}>Spread</span>
        <input
          type="range"
          min={1}
          max={10}
          step={0.1}
          value={spread}
          onChange={(e) => setSpread(parseFloat(e.target.value))}
          style={{ flex: 1, minWidth: 0, accentColor: isDark ? "#a1a1aa" : "#71717a" }}
        />
        <span style={{ fontSize: 10, fontVariantNumeric: "tabular-nums", opacity: 0.7, minWidth: 28, textAlign: "right", flexShrink: 0 }}>
          {spread.toFixed(1)}×
        </span>
      </div>
    </div>
  )
})

// ── Legacy split-panel components (kept for reference) ────────────────────────

const HierEChartsTree = memo(function HierEChartsTree({
  hierarchy, dims, isDark, onNodeClick,
}: {
  hierarchy: Record<string, CsmsGroup>
  dims: { width: number; height: number }
  isDark: boolean
  onNodeClick?: (level: HGLevel, pathKey: string, name: string) => void
}) {
  const treeData = useMemo(() => buildEChartsTree(hierarchy), [hierarchy])

  const option = useMemo(() => ({
    backgroundColor: "transparent",
    tooltip: {
      trigger: "item",
      triggerOn: "mousemove",
      backgroundColor: isDark ? "#1c1c1e" : "#ffffff",
      borderColor: isDark ? "#3f3f46" : "#e4e4e7",
      textStyle: { color: isDark ? "#fafafa" : "#09090b", fontSize: 12 },
    },
    series: [{
      type: "tree",
      data: [treeData],
      top: "5%",
      left: "6%",
      bottom: "5%",
      right: "18%",
      symbolSize: (val: number, params: any) => {
        const depth = (params.treeAncestors?.length ?? 1) - 1
        const sizes = [0, 14, 10, 8, 6, 4]
        return sizes[depth] ?? 4
      },
      symbol: "circle",
      orient: "LR",
      expandAndCollapse: true,
      initialTreeDepth: 3,
      roam: true,
      label: {
        position: "right",
        rotate: 0,
        verticalAlign: "middle",
        align: "left",
        fontSize: (params: any) => {
          const depth = (params.treeAncestors?.length ?? 1) - 1
          return [0, 12, 11, 10, 9, 8][depth] ?? 8
        },
        color: isDark ? "#e5e7eb" : "#374151",
        formatter: (params: any) => {
          const name: string = params.name ?? ""
          return name.length > 5 ? name.slice(0, 5) + "…" : name
        },
      },
      leaves: {
        label: {
          position: "right",
          verticalAlign: "middle",
          align: "left",
          fontSize: 8,
          color: isDark ? "#a1a1aa" : "#6b7280",
        },
      },
      itemStyle: {
        borderWidth: 0,
      },
      lineStyle: {
        width: 1.2,
        curveness: 0.5,
        opacity: 0.6,
      },
      emphasis: {
        focus: "ancestor",
        itemStyle: { shadowBlur: 8 },
      },
      animationDuration: 350,
      animationDurationUpdate: 250,
    }],
  }), [treeData, isDark])

  const onEvents = useMemo(() => ({
    click: (params: any) => {
      if (!onNodeClick) return
      const depth = (params.treeAncestors?.length ?? 1) - 1
      const levels: HGLevel[] = ["csms", "csms", "css", "csci", "csc", "app"]
      const level = levels[depth] ?? "app"
      // Build path key from ancestor names
      const ancestors: string[] = (params.treeAncestors ?? []).map((a: any) => a.name).slice(1) // skip root "System"
      const name: string = params.name ?? ""
      const pathKey = [...ancestors, name].join("/")
      onNodeClick(level, pathKey, name)
    },
  }), [onNodeClick])

  return (
    <div style={{ width: dims.width, height: dims.height, position: "relative" }}>
      {/* Legend */}
      <div style={{
        position: "absolute", bottom: 10, left: 10, zIndex: 10,
        display: "flex", flexDirection: "column", gap: "6px",
        padding: "6px 10px",
        borderRadius: 8,
        background: isDark ? "rgba(15,15,20,0.70)" : "rgba(255,255,255,0.80)",
        backdropFilter: "blur(8px)",
        border: `1px solid ${isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)"}`,
        fontSize: 10,
        color: isDark ? "#94a3b8" : "#64748b",
        pointerEvents: "auto",
      }}>
        <span style={{ fontWeight: 600, marginBottom: 2, display: "block" }}>Legend</span>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "8px 14px" }}>
          {(["csms", "css", "csci", "csc", "app"] as const).map(lvl => (
            <span key={lvl} style={{ display: "flex", alignItems: "center", gap: 5 }}>
              <span style={{ width: 8, height: 8, borderRadius: "50%", background: NODE_COLORS[lvl], flexShrink: 0 }} />
              {LEVEL_LABELS[lvl]}
            </span>
          ))}
        </div>
      </div>
      {/* Hint */}
      <div style={{
        position: "absolute", top: 8, right: 8, zIndex: 10,
        fontSize: 10, color: isDark ? "#52525b" : "#a1a1aa",
        pointerEvents: "none",
      }}>
        Click nodes to collapse · Scroll to zoom · Drag to pan
      </div>
      <ReactECharts
        option={option}
        style={{ width: dims.width, height: dims.height }}
        onEvents={onEvents}
        notMerge={false}
        lazyUpdate
        theme={isDark ? "dark" : undefined}
      />
    </div>
  )
})

function HierarchyGraph({ hierarchy, extraNodes = [], initialNodeId = null, syncKey = null, onNodeSelect, onSelectInfo, exportFnRef }: { hierarchy: Record<string, CsmsGroup>; extraNodes?: any[]; initialNodeId?: string | null; syncKey?: string | null; onNodeSelect?: (key: string) => void; onSelectInfo?: (pathKey: string, name: string, nodeType?: string) => void; exportFnRef?: React.MutableRefObject<(() => void) | null> }) {
  const { theme, systemTheme } = useTheme()
  const isDark = (theme === "system" ? systemTheme : theme) === "dark"
  const textMuted = isDark ? "#71717a" : "#a1a1aa"

  const containerRef = useRef<HTMLDivElement>(null)
  const searchRef = useRef<HTMLInputElement>(null)
  const [dims, setDims] = useState({ width: 800, height: 580 })

  const [drillNode, setDrillNode] = useState<HGNode | null>(null)
  const [drillStack, setDrillStack] = useState<HGNode[]>([])
  const [focusHierPathKey, setFocusHierPathKey] = useState<string | null>(null)
  const [selectedApp, setSelectedApp] = useState<HGNode | null>(null)
  const [expandedLeaves, setExpandedLeaves] = useState<Map<string, { nodes: any[]; links: any[] }>>(new Map())
  const [connDataMap, setConnDataMap] = useState<Map<string, { nodes: any[]; links: any[] }>>(new Map())
  const [connData, setConnData] = useState<{ nodes: any[]; links: any[] } | null>(null)
  const [connLoading, setConnLoading] = useState(false)
  const [connError, setConnError] = useState<string | null>(null)
  const [connTab, setConnTab] = useState<"connections" | "props">("props")
  const [connScenario, setConnScenario] = useState("direct")
  const [connSort, setConnSort] = useState<{ col: "node" | "type" | "dir"; asc: boolean }>({ col: "type", asc: true })

  const isSyncingRef = useRef(false)



  const [selectedLink, setSelectedLink] = useState<{ link: any; x: number; y: number } | null>(null)

  const [appSearch, setAppSearch] = useState("")
  const [searchOpen, setSearchOpen] = useState(false)

  // Flat list of ALL hierarchy nodes for search (CSMS → CSS → CSCI → CSC → App)
  const flatNodes = useMemo<HGNode[]>(() => {
    const result: HGNode[] = []
    for (const [csmsKey, csms] of Object.entries(hierarchy)) {
      result.push({ id: `csms:${csmsKey}`, name: csms.name, level: "csms", appCount: Object.values(csms.css).flatMap(c => Object.values(c.csci)).flatMap(c => Object.values(c.csc)).flatMap(c => c.apps).length, pathKey: csmsKey })
      for (const [cssKey, css] of Object.entries(csms.css)) {
        result.push({ id: `css:${csmsKey}/${cssKey}`, name: css.name, level: "css", appCount: Object.values(css.csci).flatMap(c => Object.values(c.csc)).flatMap(c => c.apps).length, pathKey: `${csmsKey}/${cssKey}` })
        for (const [csciKey, csci] of Object.entries(css.csci)) {
          result.push({ id: `csci:${csmsKey}/${cssKey}/${csciKey}`, name: csci.name, level: "csci", appCount: Object.values(csci.csc).flatMap(c => c.apps).length, pathKey: `${csmsKey}/${cssKey}/${csciKey}` })
          for (const [cscKey, csc] of Object.entries(csci.csc)) {
            result.push({ id: `csc:${csmsKey}/${cssKey}/${csciKey}/${cscKey}`, name: csc.name, level: "csc", appCount: csc.apps.length, pathKey: `${csmsKey}/${cssKey}/${csciKey}/${cscKey}` })
            for (const app of csc.apps)
              result.push({ id: `app:${app.id}`, name: app.name ?? app.id, level: "app", appCount: 1, pathKey: app.id, instanceKey: `app:${csmsKey}/${cssKey}/${csciKey}/${cscKey}/${app.id}` })
          }
        }
      }
    }
    return result
  }, [hierarchy])

  const filteredNodes = useMemo(() => {
    const q = appSearch.trim().toLowerCase()
    if (!q) return []
    const hierResults = flatNodes.filter(n => n.name.toLowerCase().includes(q) || n.pathKey.toLowerCase().includes(q))
    const seen = new Set(hierResults.map(n => n.pathKey))
    const extraResults: HGNode[] = extraNodes
      .filter(n => {
        const name = (n.name ?? n.label ?? n.id ?? "").toLowerCase()
        const id = (n.id ?? "").toLowerCase()
        return !seen.has(n.id) && (name.includes(q) || id.includes(q))
      })
      .map(n => ({
        id: `extra:${n.id}`,
        name: n.name ?? n.label ?? n.id,
        level: "app" as HGLevel,
        nodeType: n.type,
        appCount: 0,
        pathKey: n.id,
      }))
    return [...hierResults, ...extraResults].slice(0, 25)
  }, [flatNodes, extraNodes, appSearch])

  const jumpToNode = useCallback((node: HGNode) => {
    setAppSearch("")
    setSearchOpen(false)
    const wasSyncing = isSyncingRef.current
    isSyncingRef.current = false
    if (node.level === "app") {
      setSelectedApp(node)
      setFocusHierPathKey(null)
      setConnTab("props")
      setConnData(null)
      onSelectInfo?.(node.pathKey, node.name, node.nodeType)
    } else {
      setSelectedApp(null)
      setFocusHierPathKey(node.pathKey)
      setConnData(null)
      setConnError(null)
      const parts = node.pathKey.split("/")
      const stack: HGNode[] = []
      if (parts[0]) {
        const csmsName = hierarchy[parts[0]]?.name ?? parts[0]
        if (parts.length > 1) stack.push({ id: `csms:${parts[0]}`, name: csmsName, level: "csms", appCount: 0, pathKey: parts[0] })
      }
      if (parts[1]) {
        const cssName = hierarchy[parts[0]]?.css[parts[1]]?.name ?? parts[1]
        if (parts.length > 2) stack.push({ id: `css:${parts[0]}/${parts[1]}`, name: cssName, level: "css", appCount: 0, pathKey: `${parts[0]}/${parts[1]}` })
      }
      if (parts[2]) {
        const csciName = hierarchy[parts[0]]?.css[parts[1]]?.csci[parts[2]]?.name ?? parts[2]
        if (parts.length > 3) stack.push({ id: `csci:${parts[0]}/${parts[1]}/${parts[2]}`, name: csciName, level: "csci", appCount: 0, pathKey: `${parts[0]}/${parts[1]}/${parts[2]}` })
      }
      setDrillStack(stack)
      setDrillNode({ id: node.id, name: node.name, level: node.level, appCount: node.appCount, pathKey: node.pathKey })
    }
    if (!wasSyncing) onNodeSelect?.(node.id)
  }, [hierarchy, onNodeSelect, onSelectInfo])

  // Sync selection when syncKey changes (cross-view sync or URL param)
  const prevSyncKeyRef = useRef<string | null | undefined>(undefined)
  useEffect(() => {
    // Resolve the effective key: prefer syncKey, fall back to initialNodeId (raw id)
    const effectiveKey = syncKey ?? (initialNodeId ? `__raw:${initialNodeId}` : null)
    if (effectiveKey === prevSyncKeyRef.current) return
    if (effectiveKey === undefined) return
    prevSyncKeyRef.current = effectiveKey
    if (!effectiveKey || flatNodes.length === 0) return
    isSyncingRef.current = true
    if (effectiveKey.startsWith('__raw:')) {
      // Legacy initialNodeId path: raw node id, search by pathKey
      const rawId = effectiveKey.slice(6)
      const hierNode = flatNodes.find(n => n.level === "app" && n.pathKey === rawId)
      if (hierNode) { jumpToNode(hierNode); return }
      const extra = extraNodes.find((n: any) => n.id === rawId)
      if (extra) { jumpToNode({ id: `extra:${extra.id}`, name: extra.name ?? extra.id, level: "app", nodeType: extra.type, appCount: 0, pathKey: extra.id }); return }
    } else if (effectiveKey.startsWith('extra:')) {
      const rawId = effectiveKey.slice(6)
      const extra = extraNodes.find((n: any) => n.id === rawId)
      if (extra) { jumpToNode({ id: effectiveKey, name: extra.name ?? extra.id, level: "app", nodeType: extra.type, appCount: 0, pathKey: rawId }); return }
    } else {
      // app:id, csms:key, css:k/k, csci:k/k/k, csc:k/k/k/k
      const hierNode = flatNodes.find(n => n.id === effectiveKey)
      if (hierNode) { jumpToNode(hierNode); return }
    }
    isSyncingRef.current = false
  }, [syncKey, initialNodeId, flatNodes, extraNodes, jumpToNode])

  const graphData = useMemo(() => buildDrillData(hierarchy, drillNode), [hierarchy, drillNode])

  const filteredGraphData = useMemo(() => graphData, [graphData])

  const connGraphData = useMemo(() => {
    if (!connData) return { nodes: [], links: [] }

    // Resolve active scenario filter
    const scenarios = getScenariosForType(selectedApp?.nodeType ?? "Application")
    const activeScenario = scenarios.find(s => s.id === connScenario) ?? scenarios[0]
    const centerPathKey = selectedApp?.pathKey ?? ""

    let links = [...connData.links]

    // 1. Scenario-based edge type filter
    if (activeScenario.allowedEdgeTypes) {
      const allowedSet = new Set(activeScenario.allowedEdgeTypes)
      links = links.filter(l => allowedSet.has(l.type))
    }

    // 2. Strict BFS hop-by-hop reachability (prevents transitive leakage)
    //    e.g. Node→App→Topic→OtherApp: OtherApp is NOT on this node and must be excluded
    if (activeScenario.strictBFS && centerPathKey) {
      const visited = new Set<string>([centerPathKey])
      let frontier = new Set<string>([centerPathKey])
      for (let hop = 0; hop < activeScenario.depth; hop++) {
        const nextFrontier = new Set<string>()
        for (const link of links) {
          const srcId = link.source?.id ?? link.source
          const tgtId = link.target?.id ?? link.target
          if (frontier.has(srcId) && !visited.has(tgtId)) {
            nextFrontier.add(tgtId)
            visited.add(tgtId)
          }
          if (frontier.has(tgtId) && !visited.has(srcId)) {
            nextFrontier.add(srcId)
            visited.add(srcId)
          }
        }
        frontier = nextFrontier
      }
      links = links.filter(l => visited.has(l.source?.id ?? l.source) && visited.has(l.target?.id ?? l.target))
    }

    // Keep only nodes referenced by surviving links or the center node
    const referencedIds = new Set<string>([
      ...(selectedApp ? [centerPathKey] : []),
      ...links.flatMap(l => [l.source?.id ?? l.source, l.target?.id ?? l.target]),
    ])
    let nodes = connData.nodes.filter(n => referencedIds.has(n.id))

    // 3. Scenario-based node type filter
    if (activeScenario.allowedNodeTypes) {
      const allowedSet = new Set(activeScenario.allowedNodeTypes)
      const removedIds = new Set(nodes.filter(n => !allowedSet.has(n.type) && n.id !== centerPathKey).map(n => n.id))
      if (removedIds.size > 0) {
        nodes = nodes.filter(n => !removedIds.has(n.id))
        links = links.filter(l => !removedIds.has(l.source?.id ?? l.source) && !removedIds.has(l.target?.id ?? l.target))
      }
    }

    return { nodes, links }
  }, [connData, selectedApp, connScenario])

  // Direct neighbor id sets (relative to selected app node)
  const directOutIds = useMemo(() =>
    new Set((connData?.links ?? [])
      .filter(l => (l.source?.id ?? l.source) === selectedApp?.pathKey)
      .map(l => l.target?.id ?? l.target)),
    [connData, selectedApp])
  const directInIds = useMemo(() =>
    new Set((connData?.links ?? [])
      .filter(l => (l.target?.id ?? l.target) === selectedApp?.pathKey)
      .map(l => l.source?.id ?? l.source)),
    [connData, selectedApp])

  // Pre-compute target (x, y) for hierarchical connections layout (2D only)
  const connTargetPositions = useMemo<Map<string, { x: number; y: number }>>(() => {
    const map = new Map<string, { x: number; y: number }>()
    if (connGraphData.nodes.length === 0) return map
    const W = dims.width || 800
    const H = dims.height || 600
    const margin = W * 0.20
    // Minimum pixel gap between node centres — prevents overlap for dense layers.
    // When the natural spread exceeds the canvas, positions extend beyond its
    // bounds and ReactFlow's fitView zooms out to show all nodes.
    const minGap = 140
    const centerType = connGraphData.nodes.find((n: any) => n.id === selectedApp?.pathKey)?.type ?? "Application"
    const centerLayer = CONN_TYPE_LAYER[centerType] ?? 2
    // Bucket non-center nodes per layer, sorted deterministically
    const buckets = new Map<number, string[]>()
    for (const n of connGraphData.nodes as any[]) {
      if (n.id === selectedApp?.pathKey) continue
      const layer = CONN_TYPE_LAYER[n.type] ?? 2
      if (!buckets.has(layer)) buckets.set(layer, [])
      buckets.get(layer)!.push(n.id)
    }
    buckets.forEach(arr => arr.sort())
    // Assign absolute pixel positions
    for (const n of connGraphData.nodes as any[]) {
      const isCenter = n.id === selectedApp?.pathKey
      const layer = CONN_TYPE_LAYER[n.type] ?? 2
      const fy = CONN_LAYER_Y_FRACS[layer] * H
      if (isCenter) {
        map.set(n.id, { x: W / 2, y: fy })
        continue
      }
      const bucket = buckets.get(layer) ?? []
      const idx = bucket.indexOf(n.id)
      const count = bucket.length
      let fx: number
      if (count === 1 && layer !== centerLayer) {
        fx = W / 2
      } else if (count === 1 && layer === centerLayer) {
        fx = W / 2 + W * 0.22
      } else {
        const naturalWidth = (count - 1) * minGap
        const usable = Math.max(W - 2 * margin, naturalWidth)
        const startX = (W - usable) / 2
        fx = startX + (idx / (count - 1)) * usable
        if (layer === centerLayer) {
          const gap = W * 0.15
          if (Math.abs(fx - W / 2) < gap) fx = fx < W / 2 ? W / 2 - gap : W / 2 + gap
        }
      }
      map.set(n.id, { x: fx, y: fy })
    }
    return map
  }, [connGraphData, selectedApp, dims])

  // Which layers are actually populated (for swimlane rendering)
  const connPopulatedLayers = useMemo(() => {
    const layers = new Set<number>()
    for (const n of connGraphData.nodes as any[]) {
      layers.add(CONN_TYPE_LAYER[n.type] ?? 2)
    }
    return layers
  }, [connGraphData])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    let rafId: number | null = null
    const ro = new ResizeObserver(() => {
      if (rafId) cancelAnimationFrame(rafId)
      rafId = requestAnimationFrame(() => {
        setDims({ width: el.clientWidth, height: el.clientHeight })
        rafId = null
      })
    })
    ro.observe(el)
    setDims({ width: el.clientWidth, height: el.clientHeight })
    return () => { ro.disconnect(); if (rafId) cancelAnimationFrame(rafId) }
  }, [])

  // Reset scenario when the selected node changes
  useEffect(() => {
    if (!selectedApp) return
    const scenarios = getScenariosForType(selectedApp.nodeType ?? "Application")
    setConnScenario(scenarios[0].id)
  }, [selectedApp?.pathKey]) // eslint-disable-line react-hooks/exhaustive-deps

  // Fetch connections whenever selected app or scenario changes
  useEffect(() => {
    if (!selectedApp) { setConnData(null); setConnError(null); return }
    const scenarios = getScenariosForType(selectedApp.nodeType ?? "Application")
    const activeScenario = scenarios.find(s => s.id === connScenario) ?? scenarios[0]
    const effectiveDepth = activeScenario.depth
    let cancelled = false
    setConnLoading(true)
    setConnError(null)
    setConnData(null)
    apiClient.getNodeConnectionsWithDepth(selectedApp.pathKey, true, effectiveDepth)
      .then(structural => {
        if (cancelled) return
        setConnData({ nodes: structural.nodes, links: structural.links })
        setConnDataMap(prev => new Map(prev).set(selectedApp.instanceKey ?? selectedApp.pathKey, { nodes: structural.nodes, links: structural.links }))
      })
      .catch(e => { if (!cancelled) setConnError(e instanceof Error ? e.message : String(e)) })
      .finally(() => { if (!cancelled) setConnLoading(false) })
    return () => { cancelled = true }
  }, [selectedApp, connScenario])

  const clearSelection = useCallback(() => {
    setSelectedApp(null)
    setConnData(null)
    setConnError(null)
    setSelectedLink(null)
    setExpandedLeaves(new Map())
    setConnDataMap(new Map())
  }, [])

  // Click handler — hierarchy mode
  const drillInto = useCallback((node: object) => {
    const n = node as HGNode
    if (drillNode && n.id === drillNode.id) return
    if (n.level === "app") {
      if (selectedApp?.id === n.id) { clearSelection(); return }
      setSelectedApp(n)
      setConnTab("props")
      onNodeSelect?.(n.id)
      return
    }
    clearSelection()
    const clean: HGNode = { id: n.id, name: n.name, level: n.level, appCount: n.appCount, pathKey: n.pathKey }
    setDrillStack(prev => drillNode ? [...prev, drillNode] : prev)
    setDrillNode(clean)
    onNodeSelect?.(n.id)
  }, [drillNode, selectedApp, clearSelection, onNodeSelect])

  // Click handler — connections mode (re-center on clicked node)
  const drillIntoConn = useCallback((node: object) => {
    const n = node as any
    if (!n || !n.id) return
    if (n.id === selectedApp?.pathKey) return
    setSelectedApp({ id: `app:${n.id}`, name: n.label ?? n.id, level: "app", nodeType: n.type, appCount: 1, pathKey: n.id })
    setConnTab("props")
    setConnData(null)
    const key = n.type === 'Application' ? `app:${n.id}` : `extra:${n.id}`
    onNodeSelect?.(key)
  }, [selectedApp, onNodeSelect])

  const onConnLeafClick = useCallback((nodeId: string, instanceKey: string) => {
    // Update right panel to show the clicked node
    const existingData = connDataMap.get(nodeId)
    // Use a "conn:" prefixed instanceKey so the hierarchy tree never highlights/expands this node
    const connInstanceKey = `conn:${nodeId}`
    const extraNode = extraNodes.find((n: any) => n.id === nodeId)
    const nodeName = extraNode?.name ?? extraNode?.label ?? nodeId
    const nodeType = extraNode?.type
    const hgNode: HGNode = { id: `app:${nodeId}`, name: nodeName, level: "app", appCount: 1, pathKey: nodeId, instanceKey: connInstanceKey, nodeType }
    setSelectedApp(hgNode)
    setConnTab("props")
    if (existingData) setConnData(existingData)

    // Notify parent so side panel and list view stay in sync
    onSelectInfo?.(nodeId, nodeName, nodeType)

    // Toggle expansion
    if (expandedLeaves.has(instanceKey)) {
      setExpandedLeaves(prev => { const next = new Map(prev); next.delete(instanceKey); return next })
      return
    }
    apiClient.getNodeConnectionsWithDepth(nodeId, true, 1)
      .then(structural => {
        setExpandedLeaves(prev => new Map(prev).set(instanceKey, { nodes: structural.nodes, links: structural.links }))
        setConnDataMap(prev => new Map(prev).set(nodeId, { nodes: structural.nodes, links: structural.links }))
        setConnData({ nodes: structural.nodes, links: structural.links })
        // Update name/type from fetched data and re-notify parent side panel
        const fetchedNode = structural.nodes.find((n: any) => n.id === nodeId)
        if (fetchedNode) {
          const fetchedName = (fetchedNode as any).label ?? (fetchedNode as any).name ?? nodeId
          const fetchedType = (fetchedNode as any).type
          setSelectedApp({ id: `app:${nodeId}`, name: fetchedName, level: "app", nodeType: fetchedType, appCount: 1, pathKey: nodeId, instanceKey: connInstanceKey })
          onSelectInfo?.(nodeId, fetchedName, fetchedType)
        }
      }).catch(() => {})
  }, [expandedLeaves, connDataMap, extraNodes, onNodeSelect, onSelectInfo])

  const drillTo = useCallback((idx: number) => {
    clearSelection()
    if (idx < 0) { setDrillNode(null); setDrillStack([]); return }
    const target = drillStack[idx]
    setDrillNode(target)
    setDrillStack(prev => prev.slice(0, idx))
    onNodeSelect?.(target.id)
  }, [drillStack, clearSelection, onNodeSelect])

  // Canvas painters
  const nodeCanvasObject = useCallback(
    (node: object, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const n = node as HGNode
      const isParent = drillNode !== null && n.id === drillNode.id
      const isSelectedApp = selectedApp?.id === n.id
      const r = isParent ? NODE_SIZES[n.level] * 1.5 : NODE_SIZES[n.level]
      const color = NODE_COLORS[n.level]
      if (isSelectedApp || isParent) {
        ctx.beginPath(); ctx.arc(n.x!, n.y!, r + 5, 0, 2 * Math.PI)
        ctx.fillStyle = color + "33"; ctx.fill()
      }
      ctx.beginPath(); ctx.arc(n.x!, n.y!, r, 0, 2 * Math.PI)
      ctx.fillStyle = color; ctx.fill()
      if (isParent) { ctx.strokeStyle = isDark ? "#fff" : "#111"; ctx.lineWidth = 2; ctx.stroke() }
      else if (isSelectedApp) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke() }
      const minScale: Record<HGLevel, number> = { csms: 0.15, css: 0.3, csci: 0.5, csc: 0.7, app: 0.9 }
      if (globalScale >= minScale[n.level]) {
        const base = n.level === "csms" ? 13 : n.level === "css" ? 11 : 9
        const fontSize = Math.max(2, base / globalScale)
        ctx.font = `${isParent || n.level === "csms" ? "bold " : ""}${fontSize}px sans-serif`
        ctx.fillStyle = isDark ? "#e5e7eb" : "#374151"; ctx.textAlign = "center"
        const label = n.name.length > 24 ? n.name.slice(0, 22) + "…" : n.name
        ctx.fillText(label, n.x!, n.y! + r + fontSize + 1)
        if (!isParent && n.level !== "app" && globalScale >= minScale[n.level] * 1.5) {
          ctx.font = `${Math.max(1.5, (base - 3) / globalScale)}px sans-serif`
          ctx.fillStyle = isDark ? "#6b7280" : "#9ca3af"
          ctx.fillText(`${n.appCount}`, n.x!, n.y! + r + fontSize * 2 + 2)
        }
      }
    }, [drillNode, selectedApp, isDark])

  const hierLinkColor = isDark ? "rgba(255,255,255,0.55)" : "rgba(0,0,0,0.30)"
  const bgColor = isDark ? "#09090b" : "#ffffff"
  const breadcrumbs = [
    { label: "Root", idx: -1 },
    ...drillStack.map((n, i) => ({ label: n.name, idx: i })),
  ]

  // Full ancestor path for the currently selected app, derived from hierarchy data
  const selectedAppPath = useMemo<string[]>(() => {
    if (!selectedApp) return []
    const appId = selectedApp.pathKey
    for (const [, csms] of Object.entries(hierarchy)) {
      for (const [, css] of Object.entries(csms.css)) {
        for (const [, csci] of Object.entries(css.csci)) {
          for (const [, csc] of Object.entries(csci.csc)) {
            if (csc.apps.some((a: any) => a.id === appId)) {
              return [csms.name, css.name, csci.name, csc.name]
            }
          }
        }
      }
    }
    return []
  }, [selectedApp, hierarchy])

  const outLinks = connGraphData.links.filter((l: any) => (l.source?.id ?? l.source) === selectedApp?.pathKey)
  const inLinks  = connGraphData.links.filter((l: any) => (l.target?.id ?? l.target) === selectedApp?.pathKey)
  const nodeById = useMemo(() => { const m = new Map<string, any>(); connData?.nodes.forEach(n => m.set(n.id, n)); return m }, [connData])
  const peerLabel = (nodeId: string) => nodeById.get(nodeId)?.label ?? nodeId
  const appNode = connData?.nodes.find(n => n.id === selectedApp?.pathKey)
  const appProps = appNode?.properties ? Object.entries(appNode.properties).filter(([, v]) => v !== undefined && v !== null && v !== "") : []

  return (
    <div className="flex flex-col gap-3 h-full">
      {/* Main row */}
      <div className="flex gap-4 flex-1 min-h-0">
        {/* Canvas — merged hierarchy + connections tree */}
        <div ref={containerRef} className="flex-1 overflow-hidden relative min-w-0" style={{ background: bgColor }}>
          {connLoading && !connData && (
            <div className="absolute inset-0 flex items-center justify-center z-10 bg-background/60">
              <LoadingSpinner className="h-8 w-8" />
            </div>
          )}
          {connError && (
            <div className="absolute inset-0 flex items-center justify-center z-10">
              <p className="text-sm text-destructive bg-background/90 px-4 py-2 rounded border">{connError}</p>
            </div>
          )}
          <MergedEChartsTree
            hierarchy={hierarchy}
            connDataMap={connDataMap}
            expandedLeaves={expandedLeaves}
            selectedApp={selectedApp}
            dims={dims}
            isDark={isDark}
            focusHierPathKey={focusHierPathKey}
            onAppNodeClick={(app: AppNode, instanceKey: string) => {
              const hgNode: HGNode = { id: `app:${app.id}`, name: app.csu ?? app.name ?? app.id ?? "?", level: "app", appCount: 1, pathKey: app.id, instanceKey, appData: app }
              // If this app is already expanded (has fetched connection data), single-click collapses it.
              const isExpanded = connDataMap.has(instanceKey) || connDataMap.has(app.id)
              if (isExpanded) {
                // Collapse just this app — keep other apps' expanded subtrees visible.
                const wasSelected = selectedApp?.instanceKey === instanceKey
                if (wasSelected) {
                  setSelectedApp(null)
                  setConnData(null)
                  setConnError(null)
                  setSelectedLink(null)
                }
                setConnDataMap(prev => {
                  const next = new Map(prev)
                  next.delete(instanceKey)
                  next.delete(app.id) // legacy key
                  return next
                })
                setExpandedLeaves(prev => {
                  const next = new Map(prev)
                  // expandedLeaves are keyed by `${parentPath}:${nodeId}` — drop any rooted at this app.
                  for (const k of Array.from(next.keys())) {
                    if (k === instanceKey || k.startsWith(`${instanceKey}:`) || k.startsWith(`${instanceKey}/`)) {
                      next.delete(k)
                    }
                  }
                  return next
                })
                return
              }
              setSelectedApp(hgNode)
              setConnTab("props")
              setConnData(null)
              onNodeSelect?.(hgNode.id)
              onSelectInfo?.(hgNode.pathKey, hgNode.name, hgNode.nodeType)
            }}
            onConnNodeClick={onConnLeafClick}
            exportFnRef={exportFnRef}
          />

          {/* Search overlay */}
          <div className="absolute top-12 left-3 z-20 w-56">
            <div className="relative flex items-center">
              <Search className="absolute left-2 h-3 w-3 text-muted-foreground/50 pointer-events-none" />
              <Input
                ref={searchRef}
                className="h-7 pl-6 pr-6 text-xs bg-background rounded-md border-border shadow-sm focus-visible:ring-1 focus-visible:ring-primary/50"
                placeholder="Search hierarchy…"
                value={appSearch}
                onChange={e => { setAppSearch(e.target.value); setSearchOpen(true) }}
                onFocus={() => setSearchOpen(true)}
                onBlur={() => setTimeout(() => setSearchOpen(false), 150)}
              />
              {appSearch && (
                <button
                  className="absolute right-2 text-muted-foreground/40 hover:text-muted-foreground transition-colors"
                  onMouseDown={e => { e.preventDefault(); setAppSearch(""); setSearchOpen(false) }}
                >
                  <X className="h-3 w-3" />
                </button>
              )}
            </div>
            {searchOpen && filteredNodes.length > 0 && (
              <div className="mt-1 rounded-md border border-border bg-background shadow-md overflow-hidden max-h-64 overflow-y-auto">
                {filteredNodes.map(r => (
                  <button
                    key={r.id}
                    className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-left hover:bg-muted transition-colors"
                    onMouseDown={e => { e.preventDefault(); jumpToNode(r) }}
                  >
                    <span
                      className="inline-block w-2 h-2 rounded-full shrink-0"
                      style={{ background: NODE_COLORS[r.level] }}
                    />
                    <span className="truncate">{r.name}</span>
                    <span className="ml-auto text-muted-foreground/50 shrink-0 pl-1">{LEVEL_LABELS[r.level]}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Unified Legend — single horizontal line (static, no toggling) */}
          <div className="absolute top-2 left-3 right-3 z-10 flex items-center gap-3 text-xs overflow-x-auto pointer-events-none shrink-0 select-none">
            <div className="flex items-center gap-1.5">
              <span className="text-xs text-muted-foreground mr-1 font-medium">Levels:</span>
              {(["csms", "css", "csci", "csc", "app"] as const).map(lvl => {
                const color = NODE_COLORS[lvl]
                return (
                  <span
                    key={lvl}
                    className="flex items-center gap-1 px-1.5 py-0.5 rounded-full border shrink-0 cursor-default"
                    style={{
                      borderColor: color,
                      color: color,
                      background: `${color}18`,
                    }}
                  >
                    <span className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
                    {LEVEL_LABELS[lvl].split(' ')[0]}
                  </span>
                )
              })}
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-xs text-muted-foreground mr-1 font-medium">Nodes:</span>
              {(Object.entries(CONN_NODE_TYPE_COLORS_DARK) as [string, string][]).map(([t, color]) => (
                <span
                  key={t}
                  className="flex items-center gap-1 px-1.5 py-0.5 rounded-full border shrink-0 cursor-default"
                  style={{
                    borderColor: color,
                    color: color,
                    background: `${color}18`,
                  }}
                >
                  <span className="w-1.5 h-1.5 rounded-full" style={{ background: color }} />
                  {t}
                </span>
              ))}
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-xs text-muted-foreground mr-1 font-medium">Edges:</span>
              {(Object.entries(CONN_LINK_TYPE_COLORS_DARK) as [string, string][])
                .filter(([type]) => type !== "DEPENDS_ON" && type !== "CONNECTS_TO")
                .map(([t, color]) => (
                  <span
                    key={t}
                    className="flex items-center gap-1 px-1.5 py-0.5 rounded-full border shrink-0 cursor-default"
                    style={{
                      borderColor: color,
                      color: color,
                      background: `${color}18`,
                    }}
                  >
                    <span className="w-3 h-px" style={{ background: color }} />
                    {t.replace(/_/g, " ")}
                  </span>
                ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
// ── Graph Tab Side Panel ─────────────────────────────────────────────────────

function GraphTabSidePanel({ selectedNode, links, nodeLabels, onSelect, loading = false }: {
  selectedNode: SelectedNode | null
  links: any[]
  nodeLabels: Map<string, { label: string; type: string }>
  onSelect: (id: string) => void
  loading?: boolean
}) {
  const [tab, setTab] = useState<"details" | "connections">("details")

  return (
    <div className="flex flex-col h-full">
      <div className="flex shrink-0 border-b border-border">
        {(["details", "connections"] as const).map(t => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={cn(
              "flex-1 py-2.5 text-[11px] font-medium transition-colors capitalize",
              tab === t
                ? "border-b-2 border-primary text-foreground"
                : "text-muted-foreground hover:text-foreground"
            )}
          >
            {t}
          </button>
        ))}
      </div>
      <div className="flex-1 min-h-0 overflow-auto">
        {tab === "details"
          ? (selectedNode ? <NodeDetailPanel node={selectedNode} /> : <EmptyDetailState />)
          : <ConnectionsColumn
              selectedNode={selectedNode}
              links={links}
              nodeLabels={nodeLabels}
              onSelect={(id) => { onSelect(id); setTab("details") }}
              loading={loading}
            />
        }
      </div>
    </div>
  )
}

// ── Detail Panel (left) ──────────────────────────────────────────────────────

const KIND_LABEL: Record<SelectedKind, string> = {
  csms: "System", css: "Segment", csci: "Config Item", csc: "Component", app: "Application", node: "Node", topic: "Topic",
}
// Hierarchy levels (csms/css/csci/csc) keep their grouping palette.
// Item-type kinds (app/node/topic) align with the Force Graph palette so
// the type badge in the List tab matches the Force Graph node colour.
const KIND_COLOR: Record<SelectedKind, string> = {
  csms: "#10b981", css: "#3b82f6", csci: "#f59e0b", csc: "#f97316", app: "#4CBCD0", node: "#C570CE", topic: "#7DAA7A",
}

function DetailTable({ headers, rows }: {
  headers: string[]
  rows: (string | number | undefined)[][]
}) {
  return (
    <table className="w-full text-sm">
      <thead className="sticky top-0 bg-background/95 backdrop-blur z-10">
        <tr className="border-b border-border/60">
          {headers.map((h) => (
            <th key={h} className="text-left px-4 py-2.5 text-[10px] font-semibold text-muted-foreground/60 uppercase tracking-widest whitespace-nowrap">
              {h}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr key={i} className="border-b border-border/40 hover:bg-muted/20 transition-colors">
            {row.map((cell, j) => (
              <td key={j} className={cn("px-4 py-2.5 text-xs", j === 0 ? "font-medium text-foreground" : "text-muted-foreground font-mono")}>
                {cell !== undefined && cell !== null && cell !== ""
                  ? String(cell)
                  : <span className="text-muted-foreground/40">—</span>}
              </td>
            ))}
          </tr>
        ))}
        {rows.length === 0 && (
          <tr>
            <td colSpan={headers.length} className="px-4 py-12 text-center text-muted-foreground/50 text-xs">
              No items
            </td>
          </tr>
        )}
      </tbody>
    </table>
  )
}

function NodeDetailPanel({ node }: { node: SelectedNode }) {
  let content: React.ReactNode

  if (node.kind === "csms") {
    const csms = node.payload as CsmsGroup
    const rows = sortKeys(Object.keys(csms.css)).map((k) => {
      const css = csms.css[k]
      const csciCount = Object.keys(css.csci).length
      const cscCount = Object.values(css.csci).reduce((s, ci) => s + Object.keys(ci.csc).length, 0)
      const appCount = Object.values(css.csci).flatMap(ci => Object.values(ci.csc)).flatMap(c => c.apps).length
      return [k, csciCount, cscCount, appCount]
    })
    content = <DetailTable headers={["Segment", "Config Item Groups", "Component Groups", "Apps"]} rows={rows} />
  } else if (node.kind === "css") {
    const css = node.payload as CssGroup
    const rows = sortKeys(Object.keys(css.csci)).map((k) => {
      const csci = css.csci[k]
      const cscCount = Object.keys(csci.csc).length
      const appCount = Object.values(csci.csc).flatMap(c => c.apps).length
      return [k, cscCount, appCount]
    })
    content = <DetailTable headers={["Config Item", "Component Groups", "Apps"]} rows={rows} />
  } else if (node.kind === "csci") {
    const csci = node.payload as CsciGroup
    const rows = sortKeys(Object.keys(csci.csc)).map((k) => [k, csci.csc[k].apps.length])
    content = <DetailTable headers={["Component", "Apps"]} rows={rows} />
  } else if (node.kind === "csc") {
    const csc = node.payload as CscGroup
    const rows = csc.apps.map((a) => [a.id, a.csu ?? a.name ?? "—", a.weight])
    content = <DetailTable headers={["ID", "Name / CSU", "Weight (QoS [0–1])"]} rows={rows} />
  } else {
    // app — grouped key-value table (same grouping as graph view Props panel)
    const app = node.payload as AppNode
    const entries = Object.entries(app).filter(([, v]) => v !== undefined && v !== null && v !== "")

    const HIER_KEYS = new Set(["csms_name","csc_name","csci_name","css_name"])
    const isCmSize     = (k: string) => /^cm_total_|^loc$|^duplicated_lines_density$/.test(k)
    const isCmComplex  = (k: string) => /^cm_(total_|avg_|max_)wmc$|^cyclomatic_complexity$/.test(k)
    const isCmCohesion = (k: string) => /^cm_(avg_|max_)lcom$|^lcom(_norm)?$/.test(k)
    const isCmCoupling = (k: string) => /^cm_(avg_|max_)(cbo|rfc|fanin|fanout)$|^coupling_/.test(k)
    const isCm         = (k: string) => /^cm_|^cyclomatic_complexity$|^coupling_|^loc$|^duplicated_lines_density$|^lcom(_norm)?$|^sqale_debt_ratio$/.test(k)

    const primitives = entries.filter(([k, v]) => typeof v !== "object" && !HIER_KEYS.has(k) && !isCm(k))
    const hierarchyEntries = entries.filter(([k]) => HIER_KEYS.has(k))
    const cmSize     = entries.filter(([k]) => isCmSize(k))
    const cmComplex  = entries.filter(([k]) => isCmComplex(k))
    const cmCohesion = entries.filter(([k]) => isCmCohesion(k))
    const cmCoupling = entries.filter(([k]) => isCmCoupling(k))
    const cmOther    = entries.filter(([k, v]) => isCm(k) && !isCmSize(k) && !isCmComplex(k) && !isCmCohesion(k) && !isCmCoupling(k))

    const PrimRow = ({ k, v, indent = false }: { k: string; v: unknown; indent?: boolean }) => {
      const unit = (typeof v === "number" || (typeof v === "string" && v !== "" && !isNaN(Number(v)))) ? propUnit(k) : ""
      const desc = PROP_DESCS[k]
      return (
        <tr className="border-b border-border/40 hover:bg-muted/20 transition-colors">
          <td className={`${indent ? "pl-8" : "px-4"} pr-4 py-2.5 w-2/5`}>
            <span className="inline-flex items-center gap-0 text-[11px] text-foreground">{k}{desc && <Tip text={desc} />}</span>
          </td>
          <td className="px-4 py-2.5 font-mono text-foreground break-all text-xs">
            {propValue(k, v as any)}{unit && <span className="ml-1 text-[9px] opacity-50 not-italic">{unit}</span>}
          </td>
        </tr>
      )
    }
    const GroupHeader = ({ label }: { label: string }) => (
      <tr>
        <td colSpan={2} className="px-4 pt-4 pb-1">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/60 whitespace-nowrap">{label}</span>
            <div className="flex-1 h-px bg-border/50" />
          </div>
        </td>
      </tr>
    )
    const SubHeader = ({ label }: { label: string }) => (
      <tr>
        <td colSpan={2} className="pl-8 pr-4 pt-2 pb-0.5">
          <span className="text-[10px] font-medium text-muted-foreground/50 uppercase tracking-wide">{label}</span>
        </td>
      </tr>
    )

    content = (
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-background/95 backdrop-blur z-10">
          <tr className="border-b border-border/60">
            <th className="text-left px-4 py-2.5 text-[10px] font-semibold text-muted-foreground/60 uppercase tracking-widest w-2/5">Property</th>
            <th className="text-left px-4 py-2.5 text-[10px] font-semibold text-muted-foreground/60 uppercase tracking-widest">Value</th>
          </tr>
        </thead>
        <tbody>
          {primitives.map(([k, v]) => <PrimRow key={k} k={k} v={v} />)}

          {hierarchyEntries.length > 0 && <>
            <GroupHeader label="System Hierarchy" />
            {hierarchyEntries.map(([k, v]) => <PrimRow key={k} k={k} v={v} indent />)}
          </>}

          {(cmSize.length + cmComplex.length + cmCohesion.length + cmCoupling.length + cmOther.length) > 0 && <>
            <GroupHeader label="Code Metrics" />
            {cmSize.length > 0 && <>
              <SubHeader label="Size" />
              {cmSize.map(([k, v]) => <PrimRow key={k} k={k} v={v} indent />)}
            </>}
            {cmComplex.length > 0 && <>
              <SubHeader label="Complexity" />
              {cmComplex.map(([k, v]) => <PrimRow key={k} k={k} v={v} indent />)}
            </>}
            {cmCohesion.length > 0 && <>
              <SubHeader label="Cohesion" />
              {cmCohesion.map(([k, v]) => <PrimRow key={k} k={k} v={v} indent />)}
            </>}
            {cmCoupling.length > 0 && <>
              <SubHeader label="Coupling" />
              {cmCoupling.map(([k, v]) => <PrimRow key={k} k={k} v={v} indent />)}
            </>}
            {cmOther.map(([k, v]) => <PrimRow key={k} k={k} v={v} indent />)}
          </>}

          {entries.length === 0 && (
            <tr><td colSpan={2} className="px-4 py-12 text-center text-muted-foreground text-sm">No properties</td></tr>
          )}
        </tbody>
      </table>
    )
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 pt-4 pb-3 border-b border-border/60 shrink-0">
        {node.path.length > 1 && (
          <p className="text-[10px] text-muted-foreground/50 mb-1.5 truncate leading-none">
            {node.path.join(" › ")}
          </p>
        )}
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-sm font-semibold text-foreground truncate">{node.label}</h2>
          {(() => {
            const rawType: string = node.kind === "node" ? ((node.payload as any).type ?? "Node") : KIND_LABEL[node.kind]
            const actualType = rawType.toLowerCase() === "mqtt" ? "Broker" : rawType
            const color = node.kind === "node"
              ? (CONN_NODE_TYPE_COLORS_DARK[actualType] ?? KIND_COLOR[node.kind])
              : KIND_COLOR[node.kind]
            return (
              <span
                className="inline-flex items-center rounded px-2 py-0.5 text-[10px] font-semibold shrink-0 border"
                style={{ color, borderColor: `${color}60`, background: `${color}12` }}
              >
                {actualType}
              </span>
            )
          })()}
        </div>
      </div>
      <div className="flex-1 overflow-auto">{content}</div>
    </div>
  )
}

function EmptyDetailState() {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-3 p-8 text-center">
      <div className="h-12 w-12 rounded-full bg-muted flex items-center justify-center">
        <Layers className="h-5 w-5 text-muted-foreground" />
      </div>
      <div className="space-y-1">
        <p className="text-sm font-medium">Select an item</p>
        <p className="text-xs text-muted-foreground max-w-52">
          Browse the hierarchy tree on the right and click any item to view its details here.
        </p>
      </div>
    </div>
  )
}

// ── Browse Tree (right panel) ─────────────────────────────────────────────────

interface TreeProps {
  selectedKey: string | null
  onSelect: (n: SelectedNode) => void
  openSet: Set<string>
  toggle: (k: string) => void
  q: string
}

function TreeRow({
  depth, icon, label, count, isSelected, isOpen, hasChildren, onClick, onToggle,
}: {
  depth: number
  icon: React.ReactNode
  label: string
  count?: number
  isSelected: boolean
  isOpen?: boolean
  hasChildren: boolean
  onClick: () => void
  onToggle?: (e: React.MouseEvent) => void
}) {
  return (
    <div
      className={cn(
        "flex items-center gap-1 py-[5px] pr-2 rounded cursor-pointer select-none transition-colors",
        isSelected ? "bg-accent text-accent-foreground" : "hover:bg-muted/50 text-foreground",
      )}
      style={{ paddingLeft: `${depth * 14 + 6}px` }}
      onClick={onClick}
    >
      <span
        className="shrink-0 w-4 h-4 flex items-center justify-center text-muted-foreground"
        onClick={hasChildren ? onToggle : undefined}
      >
        {hasChildren && (isOpen
          ? <ChevronDown className="h-3 w-3" />
          : <ChevronRight className="h-3 w-3" />
        )}
      </span>
      <span className="shrink-0">{icon}</span>
      <span className={cn("flex-1 truncate text-xs leading-tight", isSelected && "font-medium")}>
        {label}
      </span>
      {count !== undefined && (
        <span className={cn("text-[10px] shrink-0 tabular-nums ml-1", isSelected ? "text-accent-foreground/70" : "text-muted-foreground")}>
          {count}
        </span>
      )}
    </div>
  )
}

function AppTreeNode({ app, path, depth, selectedKey, onSelect }: {
  app: AppNode; path: string[]; depth: number
} & Pick<TreeProps, "selectedKey" | "onSelect">) {
  const label = app.csu ?? app.name ?? app.id ?? "?"
  const nodeKey = `app:${app.id}`
  return (
    <TreeRow
      depth={depth}
      icon={<Cpu className="h-3 w-3 text-violet-500" />}
      label={label}
      isSelected={selectedKey === nodeKey}
      hasChildren={false}
      onClick={() => onSelect({ kind: "app", key: nodeKey, label, path: [...path, label], payload: app })}
    />
  )
}

function CscTreeNode({ name, csc, path, depth, selectedKey, onSelect, openSet, toggle, q }: {
  name: string; csc: CscGroup; path: string[]; depth: number
} & TreeProps) {
  const myPath = [...path, name]
  const nodeKey = `csc:${myPath.join("/")}`
  const nameMatch = q ? name.toLowerCase().includes(q) : false
  const filtered = nameMatch ? csc.apps : csc.apps.filter(a => matches(a, q))
  if (!filtered.length && !nameMatch && q) return null
  const isOpen = openSet.has(nodeKey) || !!q
  return (
    <div>
      <TreeRow
        depth={depth}
        icon={<Box className="h-3 w-3 text-orange-500" />}
        label={name}
        count={filtered.length}
        isSelected={selectedKey === nodeKey}
        isOpen={isOpen}
        hasChildren={filtered.length > 0}
        onClick={() => { toggle(nodeKey); onSelect({ kind: "csc", key: nodeKey, label: name, path: myPath, payload: csc }) }}
        onToggle={(e) => { e.stopPropagation(); toggle(nodeKey) }}
      />
      {isOpen && filtered.map(app => (
        <AppTreeNode key={app.id} app={app} path={myPath} depth={depth + 1} selectedKey={selectedKey} onSelect={onSelect} />
      ))}
    </div>
  )
}

function CsciTreeNode({ name, csci, path, depth, selectedKey, onSelect, openSet, toggle, q }: {
  name: string; csci: CsciGroup; path: string[]; depth: number
} & TreeProps) {
  const myPath = [...path, name]
  const nodeKey = `csci:${myPath.join("/")}`
  const nameMatch = q ? name.toLowerCase().includes(q) : false
  const visibleCsc = sortKeys(Object.keys(csci.csc)).filter(k =>
    nameMatch || k.toLowerCase().includes(q) || csci.csc[k].apps.some(a => matches(a, q))
  )
  if (!visibleCsc.length && !nameMatch && q) return null
  const total = visibleCsc.reduce((s, k) => s + csci.csc[k].apps.length, 0)
  const isOpen = openSet.has(nodeKey) || !!q
  return (
    <div>
      <TreeRow
        depth={depth}
        icon={<Package className="h-3 w-3 text-amber-500" />}
        label={name}
        count={total || undefined}
        isSelected={selectedKey === nodeKey}
        isOpen={isOpen}
        hasChildren={visibleCsc.length > 0}
        onClick={() => { toggle(nodeKey); onSelect({ kind: "csci", key: nodeKey, label: name, path: myPath, payload: csci }) }}
        onToggle={(e) => { e.stopPropagation(); toggle(nodeKey) }}
      />
      {isOpen && visibleCsc.map(k => (
        <CscTreeNode key={k} name={k} csc={csci.csc[k]} path={myPath} depth={depth + 1}
          selectedKey={selectedKey} onSelect={onSelect} openSet={openSet} toggle={toggle} q={q} />
      ))}
    </div>
  )
}

function CssTreeNode({ name, css, path, depth, selectedKey, onSelect, openSet, toggle, q }: {
  name: string; css: CssGroup; path: string[]; depth: number
} & TreeProps) {
  const myPath = [...path, name]
  const nodeKey = `css:${myPath.join("/")}`
  const nameMatch = q ? name.toLowerCase().includes(q) : false
  const visibleCsci = sortKeys(Object.keys(css.csci)).filter(k =>
    nameMatch || k.toLowerCase().includes(q) ||
    Object.keys(css.csci[k].csc).some(ck => ck.toLowerCase().includes(q)) ||
    Object.values(css.csci[k].csc).some(c => c.apps.some(a => matches(a, q)))
  )
  if (!visibleCsci.length && !nameMatch && q) return null
  const total = visibleCsci.reduce((s, k) =>
    s + Object.values(css.csci[k].csc).flatMap(c => c.apps).length, 0)
  const isOpen = openSet.has(nodeKey) || !!q
  return (
    <div>
      <TreeRow
        depth={depth}
        icon={isOpen ? <FolderOpen className="h-3 w-3 text-blue-500" /> : <Folder className="h-3 w-3 text-blue-500" />}
        label={name}
        count={total || undefined}
        isSelected={selectedKey === nodeKey}
        isOpen={isOpen}
        hasChildren={visibleCsci.length > 0}
        onClick={() => { toggle(nodeKey); onSelect({ kind: "css", key: nodeKey, label: name, path: myPath, payload: css }) }}
        onToggle={(e) => { e.stopPropagation(); toggle(nodeKey) }}
      />
      {isOpen && visibleCsci.map(k => (
        <CsciTreeNode key={k} name={k} csci={css.csci[k]} path={myPath} depth={depth + 1}
          selectedKey={selectedKey} onSelect={onSelect} openSet={openSet} toggle={toggle} q={q} />
      ))}
    </div>
  )
}

function CsmsTreeNode({ name, csms, selectedKey, onSelect, openSet, toggle, q }: {
  name: string; csms: CsmsGroup
} & TreeProps) {
  const nodeKey = `csms:${name}`
  const nameMatch = q ? name.toLowerCase().includes(q) : false
  const visibleCss = sortKeys(Object.keys(csms.css)).filter(k =>
    nameMatch || k.toLowerCase().includes(q) ||
    Object.keys(csms.css[k].csci).some(ck => ck.toLowerCase().includes(q)) ||
    Object.values(csms.css[k].csci).flatMap(ci => Object.keys(ci.csc)).some(ck => ck.toLowerCase().includes(q)) ||
    Object.values(csms.css[k].csci).flatMap(ci => Object.values(ci.csc)).some(c => c.apps.some(a => matches(a, q)))
  )
  if (!visibleCss.length && !nameMatch && q) return null
  const total = visibleCss.reduce((s, k) =>
    s + Object.values(csms.css[k].csci).flatMap(ci => Object.values(ci.csc)).flatMap(c => c.apps).length, 0)
  const isOpen = openSet.has(nodeKey) || !!q
  return (
    <div>
      <TreeRow
        depth={0}
        icon={<Layers className="h-3.5 w-3.5 text-emerald-500" />}
        label={name}
        count={total || undefined}
        isSelected={selectedKey === nodeKey}
        isOpen={isOpen}
        hasChildren={visibleCss.length > 0}
        onClick={() => { toggle(nodeKey); onSelect({ kind: "csms", key: nodeKey, label: name, path: [name], payload: csms }) }}
        onToggle={(e) => { e.stopPropagation(); toggle(nodeKey) }}
      />
      {isOpen && visibleCss.map(k => (
        <CssTreeNode key={k} name={k} css={csms.css[k]} path={[name]} depth={1}
          selectedKey={selectedKey} onSelect={onSelect} openSet={openSet} toggle={toggle} q={q} />
      ))}
    </div>
  )
}

// ── Overview ECharts tab — full graph, thousands of nodes, no detail ──────────

// Re-export of shared palette (defined before ForceGraphEChart below)
const OVERVIEW_TYPE_COLORS: Record<string, string> = {
  Application: "#4CBCD0",
  Node:        "#C570CE",
  Topic:       "#7DAA7A",
  Broker:      "#EFC050",
  Library:     "#ECA088",
}
const OVERVIEW_EDGE_COLORS: Record<string, string> = {
  RUNS_ON:       "#64748b",
  PUBLISHES_TO:  "#22c55e",
  SUBSCRIBES_TO: "#f97316",
  USES:          "#06b6d4",
  ROUTES:        "#d946ef",
  DEPENDS_ON:    "#ef4444",
  CONNECTS_TO:   "#84cc16",
}

const OVERVIEW_TYPES = ["Application", "Node", "Broker", "Topic", "Library"] as const

const GraphOverviewEChart = memo(function GraphOverviewEChart({
  nodesList,
  appsList,
  topicsList,
  brokersList,
  libsList,
  graphLinks,
  linksLoading,
  exportFnRef,
}: {
  nodesList: any[]
  appsList: any[]
  topicsList: any[]
  brokersList: any[]
  libsList: any[]
  graphLinks: Array<{ source: string; target: string; type: string; weight?: number }>
  linksLoading: boolean
  exportFnRef?: React.MutableRefObject<(() => void) | null>
}) {
  const { theme, systemTheme } = useTheme()
  const isDark = (theme === "system" ? systemTheme : theme) === "dark"

  // ECharts instance ref — used to dispatch downplay on edge hover and for PNG export
  // Stores the echarts instance directly via onChartReady (avoids passing ref to LoadableComponent)
  const echartsRef = useRef<any>(null)
  useEffect(() => {
    if (!exportFnRef) return
    exportFnRef.current = () => {
      const ec = echartsRef.current
      if (!ec) return
      const dataUrl = ec.getDataURL({ type: "png", backgroundColor: isDark ? "#09090b" : "#ffffff", pixelRatio: 2 })
      const a = document.createElement("a")
      a.href = dataUrl
      a.download = "overview-export.png"
      a.click()
    }
  }, [exportFnRef, isDark])

  // Suppress focus effect when hovering an edge
  const onEvents = useMemo(() => ({
    mouseover: (params: any) => {
      if (params.dataType === "edge") {
        echartsRef.current?.dispatchAction({ type: "downplay" })
      }
    },
  }), [])

  // Legend filter — which node types to hide
  const [hiddenTypes, setHiddenTypes] = useState<Set<string>>(new Set())
  const toggleType = useCallback((t: string) =>
    setHiddenTypes(prev => { const n = new Set(prev); n.has(t) ? n.delete(t) : n.add(t); return n })
  , [])

  // Hidden edge types — DEPENDS_ON and CONNECTS_TO hidden by default (too noisy)
  const [hiddenEdgeTypes, setHiddenEdgeTypes] = useState<Set<string>>(new Set(["DEPENDS_ON", "CONNECTS_TO"]))
  const toggleEdgeType = useCallback((t: string) =>
    setHiddenEdgeTypes(prev => { const n = new Set(prev); n.has(t) ? n.delete(t) : n.add(t); return n })
  , [])

  // Layout mode: clustered (by type) or scattered (uniform disk)
  const [layoutMode, setLayoutMode] = useState<"clustered" | "scattered">("scattered")

  // Build per-type lists merged into one array, compute degree for node sizing
  const { eNodes, eEdges, edgeTypes, stats } = useMemo(() => {
    const typeArrays: Record<string, any[]> = {
      Application: appsList,
      Node:        nodesList,
      Broker:      brokersList,
      Topic:       topicsList,
      Library:     libsList,
    }

    // Degree map for sizing
    const degree = new Map<string, number>()
    const incDeg = (id: string) => degree.set(id, (degree.get(id) ?? 0) + 1)

    // Compute positions: place each type on its own arc segment in a big circle.
    // We pack arcs proportionally to node count.
    const allEntries: Array<{ id: string; name: string; typeName: string; raw: any }> = []
    for (const t of OVERVIEW_TYPES) {
      if (hiddenTypes.has(t)) continue
      for (const n of typeArrays[t]) {
        allEntries.push({ id: String(n.id), name: String(n.name ?? n.csu ?? n.id ?? ""), typeName: t, raw: n })
      }
    }

    // Per-type counts (used in stats / legend)
    const typeCounts: Record<string, number> = {}
    for (const { typeName } of allEntries) typeCounts[typeName] = (typeCounts[typeName] ?? 0) + 1

    // Filter edges first so we can compute degrees on visible nodes
    const visibleIds = new Set(allEntries.map(e => e.id))
    const candidateEdges = graphLinks.filter(l =>
      visibleIds.has(String(l.source)) &&
      visibleIds.has(String(l.target))
    )
    // All edge types present (used for stable legend buttons regardless of filter)
    const edgeTypes = [...new Set(candidateEdges.map(l => l.type ?? "").filter(Boolean))]

    const filteredEdges = candidateEdges.filter(l => !hiddenEdgeTypes.has(l.type ?? ""))
    for (const l of filteredEdges) { incDeg(String(l.source)); incDeg(String(l.target)) }

    // Deterministic seeded PRNG (mulberry32-style) — stable positions across re-renders
    let seed = 0x9e3779b9
    const rand = () => {
      seed = Math.imul(seed ^ (seed >>> 15), seed | 1)
      seed ^= seed + Math.imul(seed ^ (seed >>> 7), seed | 61)
      return ((seed ^ (seed >>> 14)) >>> 0) / 4294967296
    }

    // Place each type's cluster center evenly around a ring, then scatter
    // nodes uniformly inside a sub-disk around that center.
    // All math is O(n) — no simulation, no iteration.
    const visibleTypes = OVERVIEW_TYPES.filter(t => !hiddenTypes.has(t) && typeCounts[t])
    const numTypes = visibleTypes.length
    const ORBIT_R  = 620   // distance from origin to each cluster center
    const CLUSTER_R = 340  // radius of each type's sub-disk

    // Center of each type's cluster — evenly spaced on a circle
    const clusterCenter: Record<string, { cx: number; cy: number }> = {}
    visibleTypes.forEach((t, i) => {
      const a = (i / numTypes) * 2 * Math.PI - Math.PI / 2  // start from top
      clusterCenter[t] = { cx: ORBIT_R * Math.cos(a), cy: ORBIT_R * Math.sin(a) }
    })

    const eNodes: any[] = []
    for (const entry of allEntries) {
      const { id, name, typeName, raw } = entry
      const deg = degree.get(id) ?? 0
      const symbolSize = Math.max(3, Math.min(14, 3 + Math.log2(1 + deg) * 1.4))

      // Uniform disk sampling within the type's cluster sub-disk
      const { cx, cy } = clusterCenter[typeName] ?? { cx: 0, cy: 0 }
      const angle  = rand() * 2 * Math.PI
      const radius = Math.sqrt(rand()) * (layoutMode === "clustered" ? CLUSTER_R : 960)
      const finalCx = layoutMode === "clustered" ? cx : 0
      const finalCy = layoutMode === "clustered" ? cy : 0

      eNodes.push({
        id,
        name,
        category: OVERVIEW_TYPES.indexOf(typeName as any),
        x: finalCx + radius * Math.cos(angle),
        y: finalCy + radius * Math.sin(angle),
        symbolSize,
        itemStyle: { color: OVERVIEW_TYPE_COLORS[typeName] },
        label: { show: false },
        _raw: raw,
        _type: typeName,
      })
    }

    // Build ECharts edges
    const edgeOpacity = Math.max(0.04, Math.min(0.25, 0.12 - (filteredEdges.length / 100000) * 0.08))
    const eEdges = filteredEdges.map(l => ({
      source: String(l.source),
      target: String(l.target),
      edgeType: l.type ?? "",
      lineStyle: {
        color: OVERVIEW_EDGE_COLORS[l.type ?? ""] ?? "#94a3b8",
        opacity: edgeOpacity,
        width: 0.6,
      },
      // Suppress emphasis on edge hover — keep same opacity so nothing visually changes
      emphasis: {
        lineStyle: { opacity: edgeOpacity, width: 0.6 },
        disabled: false,
      },
    }))

    const stats = {
      nodes: eNodes.length,
      edges: eEdges.length,
      typeCounts,
    }

    return { eNodes, eEdges, edgeTypes, stats }
  }, [appsList, nodesList, brokersList, topicsList, libsList, graphLinks, hiddenTypes, hiddenEdgeTypes, layoutMode])

  // Name lookup for edge tooltip (id → display name)
  const nodeNameMap = useMemo(() => {
    const m = new Map<string, string>()
    eNodes.forEach(n => m.set(String(n.id), n.name || n.id))
    return m
  }, [eNodes])

  const option = useMemo(() => ({
    backgroundColor: "transparent",
    animation: false,
    tooltip: {
      show: true,
      trigger: "item",
      formatter: (params: any) => {
        if (params.dataType === "node") {
          const d = params.data
          const type: string = d._type ?? (OVERVIEW_TYPES[d.category] ?? "Unknown")
          const n = d._raw ?? {}
          const get = (k: string) => n.properties?.[k] ?? n[k]
          const name = d.name || d.id
          let extra = ""
          if (type === "Application") {
            const role = get("role"); if (role != null && role !== "") extra += `<br/><span style="opacity:0.7">Role: ${role}</span>`
            const priority = get("priority"); if (priority != null && priority !== "") extra += `<br/><span style="opacity:0.7">Priority: ${priority}</span>`
            const hotstandby = get("hotstandby"); if (hotstandby) extra += `<br/><span style="opacity:0.7">Hot Standby: true</span>`
          }
          if (type === "Topic") {
            const qr = get("qos_reliability"); if (qr != null && qr !== "") extra += `<br/><span style="opacity:0.7">Reliability: ${qr}</span>`
            const qd = get("qos_durability");  if (qd != null && qd !== "") extra += `<br/><span style="opacity:0.7">Durability: ${qd}</span>`
            const qt = get("qos_transport_priority"); if (qt != null && qt !== "") extra += `<br/><span style="opacity:0.7">Transport Priority: ${qt}</span>`
            const szRaw = get("message_size") ?? get("payload_size_bytes") ?? get("size")
            if (szRaw != null && szRaw !== "") { const szN = Number(szRaw); const szFmt = isFinite(szN) ? (szN >= 1048576 ? `${(szN/1048576).toFixed(2)} MB` : szN >= 1024 ? `${(szN/1024).toFixed(1)} KB` : `${szN} B`) : String(szRaw); extra += `<br/><span style="opacity:0.7">Size: ${szFmt}</span>` }
          }
          if (type === "Library") {
            const ver = get("version"); if (ver != null && ver !== "") extra += `<br/><span style="opacity:0.7">Version: ${ver}</span>`
          }
          if (type === "Broker") {
            const bt = get("broker_type"); if (bt != null && bt !== "") extra += `<br/><span style="opacity:0.7">Protocol: ${bt}</span>`
          }
          const typeStr = type ? `<br/><span style="opacity:0.7">${type}</span>` : ""
          return `<div style="font-size:12px;line-height:1.7"><b>${name}</b>${typeStr}${extra}</div>`
        }
        if (params.dataType === "edge") {
          const d = params.data
          const edgeType = d.edgeType ?? ""
          const sourceName = nodeNameMap.get(String(d.source)) ?? d.source
          const targetName = nodeNameMap.get(String(d.target)) ?? d.target
          return `<div style="font-size:12px;line-height:1.7"><b>${sourceName}</b><br/><span style="opacity:0.7">↓ ${edgeType}</span><br/><b>${targetName}</b></div>`
        }
        return ""
      },
      enterable: false,
      backgroundColor: isDark ? "#1c1c1e" : "#ffffff",
      borderColor: isDark ? "#3f3f46" : "#e4e4e7",
      textStyle: { color: isDark ? "#fafafa" : "#09090b", fontSize: 12 },
    },
    legend: { show: false },
    series: [{
      type: "graph",
      layout: "none",
      data: eNodes,
      links: eEdges,
      categories: OVERVIEW_TYPES.map(t => ({ name: t, itemStyle: { color: OVERVIEW_TYPE_COLORS[t] } })),
      roam: true,
      draggable: false,
      focusNodeAdjacency: false,
      progressive: 500,
      progressiveThreshold: 1000,
      label: { show: false },
      lineStyle: { curveness: 0 },
      emphasis: {
        focus: "adjacency",
        scale: true,
        lineStyle: { opacity: 0.7, width: 1.2 },
        label: {
          show: true,
          fontSize: 10,
          color: isDark ? "#fafafa" : "#09090b",
          backgroundColor: isDark ? "#1c1c1e" : "#ffffff",
          borderRadius: 4,
          padding: [2, 4],
        },
      },
      blur: {
        itemStyle: { opacity: 0.08 },
        lineStyle: { opacity: 0.03 },
      },
      zoom: 0.9,
    }],
  }), [eNodes, eEdges, isDark, nodeNameMap])

  const bg = isDark ? "#09090b" : "#fafaf9"
  const textMuted = isDark ? "#71717a" : "#a1a1aa"

  return (
    <div className="flex flex-col h-full" style={{ background: bg }}>
      {/* Toolbar */}
      <div className="flex items-center gap-3 px-3 py-2 shrink-0 flex-wrap">
        {/* Node type legend */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-xs text-muted-foreground mr-1">Nodes:</span>
          {OVERVIEW_TYPES.map(t => (
            <button
              key={t}
              onClick={() => toggleType(t)}
              className="flex items-center gap-1 px-2 py-0.5 rounded-full text-xs border transition-opacity"
              style={{
                borderColor: OVERVIEW_TYPE_COLORS[t],
                color: hiddenTypes.has(t) ? textMuted : OVERVIEW_TYPE_COLORS[t],
                opacity: hiddenTypes.has(t) ? 0.4 : 1,
                background: hiddenTypes.has(t) ? "transparent" : `${OVERVIEW_TYPE_COLORS[t]}18`,
              }}
            >
              <span className="w-2 h-2 rounded-full inline-block" style={{ background: OVERVIEW_TYPE_COLORS[t] }} />
              {t}
              {stats.typeCounts[t] ? <span className="opacity-60">({stats.typeCounts[t]})</span> : null}
            </button>
          ))}
        </div>

        {/* Edge type legend — always rendered so buttons don't disappear when all are toggled off */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-xs text-muted-foreground mr-1">Edges:</span>
          {edgeTypes.map(t => (
            <button
              key={t}
              onClick={() => toggleEdgeType(t)}
              className="flex items-center gap-1 px-2 py-0.5 rounded-full text-xs border transition-opacity"
              style={{
                borderColor: OVERVIEW_EDGE_COLORS[t] ?? "#94a3b8",
                color: hiddenEdgeTypes.has(t) ? textMuted : (OVERVIEW_EDGE_COLORS[t] ?? "#94a3b8"),
                opacity: hiddenEdgeTypes.has(t) ? 0.4 : 1,
                background: hiddenEdgeTypes.has(t) ? "transparent" : `${OVERVIEW_EDGE_COLORS[t] ?? "#94a3b8"}18`,
              }}
            >
              <span className="w-3 h-px inline-block" style={{ background: OVERVIEW_EDGE_COLORS[t] ?? "#94a3b8" }} />
              {t}
            </button>
          ))}
        </div>

        {/* Layout toggle + Stats */}
        <div className="ml-auto flex items-center gap-3 text-xs text-muted-foreground">
          {linksLoading && <span className="animate-pulse">Loading edges…</span>}
          <span>{stats.nodes.toLocaleString()} nodes</span>
          <span>{stats.edges.toLocaleString()} edges</span>
          <button
            onClick={() => setLayoutMode(m => m === "clustered" ? "scattered" : "clustered")}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded-md border border-border text-xs transition-colors hover:bg-accent"
            title={layoutMode === "clustered" ? "Switch to scattered layout" : "Switch to clustered layout"}
          >
            {layoutMode === "clustered" ? "Clustered" : "Scattered"}
          </button>
          <span className="hidden sm:inline opacity-60">Scroll to zoom · drag to pan</span>
        </div>
      </div>

      {/* Chart */}
      {eNodes.length === 0 ? (
        <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
          No graph data available.
        </div>
      ) : (
        <ReactECharts
          onChartReady={(instance: any) => { echartsRef.current = instance }}
          option={option}
          style={{ flex: 1, width: "100%", minHeight: 0 }}
          opts={{ renderer: "canvas" }}
          onEvents={onEvents}
          notMerge
        />
      )}
    </div>
  )
})

// ── Force-layout ECharts tab ──────────────────────────────────────────────────

type SwimlaneType = "Node" | "Application" | "Topic" | "Library" | "Broker"

const FORCE_CATEGORIES: { name: SwimlaneType; color: string }[] = [
  { name: "Application", color: "#4CBCD0" },
  { name: "Node",        color: "#C570CE" },
  { name: "Topic",       color: "#7DAA7A" },
  { name: "Broker",      color: "#EFC050" },
  { name: "Library",     color: "#ECA088" },
]

const EDGE_COLORS: Record<string, string> = {
  RUNS_ON:       "#64748b",
  PUBLISHES_TO:  "#22c55e",
  SUBSCRIBES_TO: "#f97316",
  USES:          "#06b6d4",
  ROUTES:        "#d946ef",
}
const EDGE_COLOR_FALLBACK = "#94a3b8"

const ForceGraphEChart = memo(function ForceGraphEChart({
  nodesList,
  appsList,
  topicsList,
  brokersList,
  libsList,
  graphLinks,
  linksLoading,
  selectedKey,
  onNodeClick,
  exportFnRef,
}: {
  nodesList: any[]
  appsList: any[]
  topicsList: any[]
  brokersList: any[]
  libsList: any[]
  graphLinks: Array<{ source: string; target: string; type: string; weight?: number }>
  linksLoading: boolean
  selectedKey: string | null
  onNodeClick: (rawNode: any, nodeType: SwimlaneType) => void
  exportFnRef?: React.MutableRefObject<(() => void) | null>
}) {
  const { theme, systemTheme } = useTheme()
  const isDark = (theme === "system" ? systemTheme : theme) === "dark"
  const selectedId = selectedKey ? selectedKey.replace(/^[^:]+:/, "") : null

  // ECharts instance ref for PNG export (stores the echarts instance directly via onChartReady)
  const echartsRef = useRef<any>(null)
  useEffect(() => {
    if (!exportFnRef) return
    exportFnRef.current = () => {
      const ec = echartsRef.current
      if (!ec) return
      const dataUrl = ec.getDataURL({ type: "png", backgroundColor: isDark ? "#09090b" : "#ffffff", pixelRatio: 2 })
      const a = document.createElement("a")
      a.href = dataUrl
      a.download = "graph-export.png"
      a.click()
    }
  }, [exportFnRef, isDark])

  // Search state
  const [graphSearch, setGraphSearch] = useState("")
  const [searchFocused, setSearchFocused] = useState(false)

  // Legend filter state
  const [hiddenNodeTypes, setHiddenNodeTypes] = useState<Set<string>>(new Set())
  const [hiddenEdgeTypes, setHiddenEdgeTypes] = useState<Set<string>>(new Set())
  const toggleNodeType = useCallback((name: string) =>
    setHiddenNodeTypes(prev => { const n = new Set(prev); n.has(name) ? n.delete(name) : n.add(name); return n }), [])
  const toggleEdgeType = useCallback((t: string) =>
    setHiddenEdgeTypes(prev => { const n = new Set(prev); n.has(t) ? n.delete(t) : n.add(t); return n }), [])

  // Track which nodes have been expanded (clicked) in the graph
  const [expandedIds, setExpandedIds] = useState<Set<string>>(() =>
    selectedId ? new Set([selectedId]) : new Set()
  )

  // Node spread/repulsion control
  const [spreadValue, setSpreadValue] = useState(100) // 0-200 scale, 100 is default

  // When selection changes from *outside* (List tab), reset expansion.
  // When the click originates from the graph itself we skip the reset.
  const skipResetRef = useRef(false)
  const prevSelectedIdRef = useRef<string | null>(null)
  useEffect(() => {
    if (selectedId !== prevSelectedIdRef.current) {
      prevSelectedIdRef.current = selectedId
      if (skipResetRef.current) {
        skipResetRef.current = false
      } else {
        setExpandedIds(selectedId ? new Set([selectedId]) : new Set())
      }
    }
  }, [selectedId])

  // Map nodeId → { raw, type } for click handler
  const nodeMap = useMemo(() => {
    const m = new Map<string, { raw: any; type: SwimlaneType }>()
    appsList.forEach(n  => m.set(String(n.id), { raw: n, type: "Application" }))
    nodesList.forEach(n => m.set(String(n.id), { raw: n, type: "Node" }))
    topicsList.forEach(n => m.set(String(n.id), { raw: n, type: "Topic" }))
    brokersList.forEach(n => m.set(String(n.id), { raw: n, type: "Broker" }))
    libsList.forEach(n  => m.set(String(n.id), { raw: n, type: "Library" }))
    return m
  }, [appsList, nodesList, topicsList, brokersList, libsList])

  // Search results derived from nodeMap
  const searchResults = useMemo(() => {
    const q = graphSearch.trim().toLowerCase()
    if (!q) return []
    const results: Array<{ id: string; name: string; type: SwimlaneType }> = []
    nodeMap.forEach(({ raw, type }, id) => {
      const name = raw.name ?? raw.csu ?? id
      if (String(name).toLowerCase().includes(q) || id.toLowerCase().includes(q))
        results.push({ id, name: String(name), type })
    })
    return results.slice(0, 20)
  }, [nodeMap, graphSearch])

  const eLinks = useMemo(() => {
    const filtered = graphLinks.filter(l => {
      if (l.type === "DEPENDS_ON" || l.type === "CONNECTS_TO") return false
      if (hiddenEdgeTypes.has(l.type ?? "")) return false
      const srcType = nodeMap.get(String(l.source))?.type
      const tgtType = nodeMap.get(String(l.target))?.type
      if (srcType && hiddenNodeTypes.has(srcType)) return false
      if (tgtType && hiddenNodeTypes.has(tgtType)) return false
      return true
    })
    const visible = !expandedIds.size
      ? filtered
      : filtered.filter(l => expandedIds.has(String(l.source)) || expandedIds.has(String(l.target)))

    // Count how many edges exist per unordered pair so we can spread them with curveness
    const pairCount = new Map<string, number>()
    const pairIndex = new Map<string, number>()
    for (const l of visible) {
      const key = [String(l.source), String(l.target)].sort().join("||")
      pairCount.set(key, (pairCount.get(key) ?? 0) + 1)
    }

    return visible.map(l => {
      const key = [String(l.source), String(l.target)].sort().join("||")
      const total = pairCount.get(key) ?? 1
      const idx   = pairIndex.get(key) ?? 0
      pairIndex.set(key, idx + 1)
      // Spread curves symmetrically: 0 edges → 0, 2 edges → ±0.3, 3 → -0.3/0/0.3 …
      const curveness = total === 1 ? 0 : (idx / (total - 1) - 0.5) * 0.6
      return {
        source: String(l.source),
        target: String(l.target),
        edgeType: l.type ?? "",
        label: { show: false, formatter: l.type ?? "", fontSize: 9 }, // Hide edge labels
        lineStyle: {
          opacity: 0.55,
          width: Math.max(0.5, (l.weight ?? 0.5) * 2),
          curveness,
          color: EDGE_COLORS[l.type ?? ""] ?? EDGE_COLOR_FALLBACK,
        },
      }
    })
  }, [graphLinks, expandedIds, hiddenEdgeTypes, hiddenNodeTypes, nodeMap])

  const visibleNodeIds = useMemo(() => {
    if (!expandedIds.size) return null
    const ids = new Set<string>(expandedIds)
    eLinks.forEach(l => { ids.add(String(l.source)); ids.add(String(l.target)) })
    return ids
  }, [expandedIds, eLinks])

  // First pass: build basic node list to determine complexity
  const baseNodeCount = useMemo(() => {
    const catIdx: Record<SwimlaneType, number> = {
      Application: 0, Node: 1, Topic: 2, Broker: 3, Library: 4,
    }
    const rows: [any[], SwimlaneType][] = [
      [appsList,   "Application"],
      [nodesList,  "Node"],
      [topicsList, "Topic"],
      [brokersList,"Broker"],
      [libsList,   "Library"],
    ]
    return rows.reduce((count, [list, type]) =>
      count + list.filter(n => !visibleNodeIds || visibleNodeIds.has(String(n.id)))
                  .filter(() => !hiddenNodeTypes.has(type as string)).length,
      0
    )
  }, [appsList, nodesList, topicsList, brokersList, libsList, visibleNodeIds, hiddenNodeTypes])

  // Detect graph complexity to adjust simulation parameters dynamically
  const graphComplexity = useMemo(() => {
    const nodeCount = baseNodeCount
    const edgeCount = eLinks.length
    const avgDegree = nodeCount > 0 ? edgeCount / nodeCount : 0
    // Classify: small (<50), medium (50-200), large (>200)
    return {
      nodeCount,
      edgeCount,
      avgDegree,
      isLarge: nodeCount > 200 || edgeCount > 500,
      isMedium: nodeCount > 50 || edgeCount > 150,
    }
  }, [baseNodeCount, eLinks])

  const eNodes = useMemo(() => {
    const catIdx: Record<SwimlaneType, number> = {
      Application: 0, Node: 1, Topic: 2, Broker: 3, Library: 4,
    }
    const rows: [any[], SwimlaneType][] = [
      [appsList,   "Application"],
      [nodesList,  "Node"],
      [topicsList, "Topic"],
      [brokersList,"Broker"],
      [libsList,   "Library"],
    ]
    // Node sizing for force graph
    const nodeSizeConfig = { selected: 60, normal: 28 }

    return rows.flatMap(([list, type]) =>
      list
        .filter(n => !visibleNodeIds || visibleNodeIds.has(String(n.id)))
        .filter(() => !hiddenNodeTypes.has(type as string))
        .map(n => {
          const id   = String(n.id)
          const name = n.name ?? n.csu ?? id
          const sel  = id === selectedId
          return {
            id,
            name,
            category: catIdx[type],
            symbolSize: sel ? nodeSizeConfig.selected : nodeSizeConfig.normal,
            itemStyle: (() => {
              const lvl = n.criticality_level?.toLowerCase() || "minimal"
              const CRIT_COLORS: Record<string, string> = {
                critical: '#ef4444', high: '#f97316', medium: '#eab308', low: '#22c55e', minimal: '#6b7280'
              }
              const isCritical = lvl === 'critical'
              const isHigh = lvl === 'high'

              let borderW = sel ? (graphComplexity.isLarge ? 2 : 4) : 0
              let borderC = sel ? "#f97316" : undefined
              let shadowB = sel ? (graphComplexity.isLarge ? 10 : 18) : 0
              let shadowC = sel ? "rgba(249,115,22,0.7)" : undefined

              if (isCritical || isHigh) {
                borderW = isCritical ? 2.5 : 1.5
                borderC = CRIT_COLORS[lvl]
                shadowB = isCritical ? 12 : 6
                shadowC = CRIT_COLORS[lvl] + '88'
              }

              if (sel) {
                borderW = graphComplexity.isLarge ? 3 : 5
                borderC = "#ffffff"
                shadowB = 20
                shadowC = isCritical ? "#ef4444" : "#f97316"
              }

              if (borderW > 0) {
                return {
                  borderWidth: borderW,
                  borderColor: borderC,
                  borderType: "solid" as const,
                  shadowBlur: shadowB,
                  shadowColor: shadowC,
                }
              }
              return undefined
            })(),
            nodeType: type as string,
            _lvl: n.criticality_level?.toLowerCase() || "minimal",
            ...(type === "Application" ? { _role: n.role ?? n.properties?.role ?? "", _priority: n.priority ?? n.properties?.priority ?? "", _hotstandby: n.hotstandby ?? n.properties?.hotstandby ?? false } : {}),
            ...(type === "Topic" ? {
              _qos_reliability:        n.qos_reliability            ?? n.properties?.qos_reliability            ?? "",
              _qos_durability:         n.qos_durability             ?? n.properties?.qos_durability             ?? "",
              _qos_transport_priority: n.qos_transport_priority     ?? n.properties?.qos_transport_priority     ?? "",
              _size:                   n.message_size ?? n.properties?.message_size ?? n.payload_size_bytes ?? n.properties?.payload_size_bytes ?? n.size ?? n.properties?.size ?? "",
            } : {}),
            ...(type === "Library" ? { _version: n.version ?? n.properties?.version ?? "" } : {}),
            ...(type === "Broker" ? { _broker_type: n.broker_type ?? n.properties?.broker_type ?? "" } : {}),
            label: {
              show: sel, // Show label only for selected node
              formatter: (p: any) => {
                const n = p.data?.name ?? p.name ?? ""
                return n // Show full name without truncation for selected node
              },
              fontSize: sel ? 11 : 9,
              fontWeight: sel ? 700 : 400,
              color: "#fff",
              position: "inside",
              overflow: "none",
            },
            z: sel ? 10 : 1,
            emphasis: { label: { show: sel } }, // Show label only for selected node on emphasis
          }
        })
    )
  }, [appsList, nodesList, topicsList, brokersList, libsList, selectedId, visibleNodeIds, hiddenNodeTypes, graphComplexity])

  // All edge types present in data (unfiltered) — used for legend so hidden items remain visible
  const edgeTypesInView = useMemo(() => {
    const types = graphLinks
      .filter(l => l.type !== "DEPENDS_ON" && l.type !== "CONNECTS_TO")
      .map(l => l.type)
      .filter(Boolean)
    return [...new Set(types)] as string[]
  }, [graphLinks])

  // All node types present in data (unfiltered) — used for legend
  const nodeTypesInView = useMemo(() => {
    const present = new Set<string>()
    nodeMap.forEach(({ type }) => present.add(type))
    return FORCE_CATEGORIES.filter(c => present.has(c.name))
  }, [nodeMap])

  const selectedConnections = useMemo(() => {
    if (!selectedId) return null
    const relevant = graphLinks.filter(l =>
      l.type !== "DEPENDS_ON" && l.type !== "CONNECTS_TO" &&
      !hiddenEdgeTypes.has(l.type ?? "")
    )
    const outgoing = relevant.filter(l => {
      if (String(l.source) !== selectedId) return false
      const targetType = nodeMap.get(String(l.target))?.type
      return !targetType || !hiddenNodeTypes.has(targetType)
    })
    const incoming = relevant.filter(l => {
      if (String(l.target) !== selectedId) return false
      const srcType = nodeMap.get(String(l.source))?.type
      return !srcType || !hiddenNodeTypes.has(srcType)
    })
    const getName = (id: string) => nodeMap.get(id)?.raw?.name ?? nodeMap.get(id)?.raw?.csu ?? id
    return { outgoing, incoming, getName }
  }, [selectedId, graphLinks, nodeMap, hiddenNodeTypes, hiddenEdgeTypes])

  const option = useMemo(() => {
    // Scale repulsion and edge length based on spread slider (0-200, default 100)
    const spreadFactor = spreadValue / 100
    const baseRepulsion = 400
    const baseEdgeLengthMin = 100
    const baseEdgeLengthMax = 200

    // Uniform force simulation parameters across all graph sizes — optimized for stability
    // Key: disable layoutAnimation to freeze positions once settled, use high friction to converge quickly
    const forceParams = {
      repulsion: baseRepulsion * spreadFactor,        // Scaled by spread slider
      gravity: 0.15 * (1 / spreadFactor),             // Reduced gravity inversely to repulsion
      edgeLength: [baseEdgeLengthMin * spreadFactor, baseEdgeLengthMax * spreadFactor], // Scaled edge spacing
      friction: 0.95,        // Very high damping to stop movement immediately
      layoutAnimation: false,// Disable continuous animation — freeze positions once calculated
      initLayout: "circular" as const, // Start from circular layout, converges faster
      iterations: 50,        // Run 50 iterations then stop completely
      // Note: ECharts will run physics for 50 frames then stop — no further movement
    }

    // Disable edge labels entirely (only show labels on selected node)
    const showEdgeLabels = false

    return {
      backgroundColor: "transparent",
      animation: false,        // Disable chart-level animation to prevent persistent movement
      legend: { show: false },
      tooltip: {
        trigger: "item",
        backgroundColor: isDark ? "#1c1c1e" : "#ffffff",
        borderColor: isDark ? "#3f3f46" : "#e4e4e7",
        textStyle: { color: isDark ? "#fafafa" : "#09090b", fontSize: 12 },
        formatter: (p: any) => {
          if (p.dataType === "node") {
            const d = p.data ?? {}
            const name = d.name ?? p.name ?? ""
            const type: string = d.nodeType ?? ""
            const lvl = d._lvl as string | undefined

            let html = `<div style="font-size:12px;line-height:1.7;max-width:260px">`
            html += `<b>${name}</b>`
            if (type) html += ` <span style="opacity:0.6;font-size:10px">(${type})</span>`

            if (lvl) {
              const CRIT_COLORS: Record<string, string> = {
                critical: '#ef4444', high: '#f97316', medium: '#eab308', low: '#22c55e', minimal: '#6b7280'
              }
              const col = CRIT_COLORS[lvl] || '#6b7280'
              html += `<br/><span style="font-size:9px;font-weight:700;padding:1px 5px;border-radius:4px;background:${col}20;color:${col};text-transform:uppercase">${lvl}</span>`
            }

            let extra = ""
            if (type === "Application" && d._role) extra += `<br/>Role: ${d._role}`
            if (type === "Application" && d._priority) extra += `<br/>Priority: ${d._priority}`
            if (type === "Topic") {
              if (d._qos_reliability) extra += `<br/>Reliability: ${d._qos_reliability}`
              if (d._qos_durability) extra += `<br/>Durability: ${d._qos_durability}`
              if (d._qos_transport_priority) extra += `<br/>Transport: ${d._qos_transport_priority}`
            }
            if (type === "Library" && d._version) extra += `<br/>Version: ${d._version}`
            if (type === "Broker" && d._broker_type) extra += `<br/>Protocol: ${d._broker_type}`
            html += `<br/><span style="opacity:0.7;font-size:11px">${extra}</span>`

            html += `</div>`
            return html
          }
          if (p.dataType === "edge") {
            const edgeType = p.data?.edgeType ?? ""
            const source = p.data?.source ?? ""
            const target = p.data?.target ?? ""
            const sourceName = nodeMap.get(source)?.raw?.name ?? nodeMap.get(source)?.raw?.csu ?? source
            const targetName = nodeMap.get(target)?.raw?.name ?? nodeMap.get(target)?.raw?.csu ?? target
            return `<div style="font-size:12px;line-height:1.7"><b>${sourceName}</b><br/><span style="opacity:0.7">↓ ${edgeType}</span><br/><b>${targetName}</b></div>`
          }
          return `${p.data?.source} → ${p.data?.target}`
        },
      },
      series: [{
        type:          "graph",
        layout:        "force",
        data:          eNodes,
        links:         eLinks,
        categories:    FORCE_CATEGORIES.map(c => ({
          name:      c.name,
          itemStyle: { color: c.color },
        })),
        roam:          true,  // Enable panning/zooming
        focusNodeAdjacency: true,
        label:         {
          show: false, // Labels controlled per-node; only selected node shows label
          position: "inside",
          fontSize: 9,
          formatter: (p: any) => {
            const name = p.data?.name ?? p.name ?? ""
            return name.length > 5 ? name.slice(0, 5) + "…" : name
          }
        },
        edgeSymbol:    ["none", "arrow"],
        edgeSymbolSize: graphComplexity.isLarge ? 4 : 8, // Smaller arrows for large graphs
        edgeLabel:     { show: false }, // Always hide edge labels
        lineStyle:     { opacity: 0.55 },
        force:         forceParams,
        emphasis: {
          focus:     "adjacency",
          lineStyle: { opacity: 0.7 },
          label:     { show: false }, // Hide edge labels on emphasis
        },
      }],
    }
  }, [eNodes, eLinks, isDark, graphComplexity, spreadValue])

  const onEvents = useMemo(() => ({
    click: (p: any) => {
      if (p.dataType !== "node") return
      const id = String(p.data?.id ?? "")
      // Expand this node's connections and make it selected
      setExpandedIds(prev => new Set([...prev, id]))
      const entry = nodeMap.get(id)
      if (entry) {
        skipResetRef.current = true
        onNodeClick(entry.raw, entry.type)
      }
    },
  }), [nodeMap, onNodeClick])

  return (
    <div className="relative w-full h-full">
      {linksLoading && (
        <div className="absolute top-2 right-3 z-10 text-xs text-muted-foreground">Loading edges…</div>
      )}
      {graphComplexity.isLarge && (
        <div className="absolute top-2 right-3 z-10 text-xs text-muted-foreground">
          Optimized mode: {graphComplexity.nodeCount} nodes, {graphComplexity.edgeCount} edges
        </div>
      )}

      {/* Search bar and spread slider overlay */}
      <div className="absolute top-12 left-3 z-20 w-64">
        <div className="relative flex items-center mb-2">
          <Search className="absolute left-2 h-3 w-3 text-muted-foreground/50 pointer-events-none" />
          <Input
            className="h-7 pl-6 pr-6 text-xs bg-background rounded-md border-border shadow-sm focus-visible:ring-1 focus-visible:ring-primary/50"
            placeholder="Search nodes…"
            value={graphSearch}
            onChange={e => setGraphSearch(e.target.value)}
            onFocus={() => setSearchFocused(true)}
            onBlur={() => setTimeout(() => setSearchFocused(false), 150)}
          />
          {graphSearch && (
            <button
              className="absolute right-2 text-muted-foreground/40 hover:text-muted-foreground transition-colors"
              onMouseDown={e => { e.preventDefault(); setGraphSearch("") }}
            >
              <X className="h-3 w-3" />
            </button>
          )}
        </div>
        {searchFocused && searchResults.length > 0 && (
          <div className="mt-1 rounded-md border border-border bg-background shadow-md overflow-hidden max-h-64 overflow-y-auto">
            {searchResults.map(r => (
              <button
                key={r.id}
                className="w-full flex items-center gap-2 px-3 py-1.5 text-xs text-left hover:bg-muted transition-colors"
                onMouseDown={e => {
                  e.preventDefault()
                  const entry = nodeMap.get(r.id)
                  if (!entry) return
                  setExpandedIds(prev => new Set([...prev, r.id]))
                  skipResetRef.current = true
                  onNodeClick(entry.raw, entry.type)
                  setGraphSearch("")
                }}
              >
                <span
                  className="inline-block w-2 h-2 rounded-full shrink-0"
                  style={{ background: FORCE_CATEGORIES.find(c => c.name === r.type)?.color ?? "#94a3b8" }}
                />
                <span className="truncate">{r.name}</span>
                <span className="ml-auto text-muted-foreground/50 shrink-0">{r.type}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Spread/Repulsion Slider — matches system tab styling, positioned bottom-right */}
      <div style={{
        position: "absolute", bottom: 12, right: 12, zIndex: 10,
        display: "flex", alignItems: "center", gap: 8,
        padding: "6px 10px",
        borderRadius: 8,
        width: 224,
        background: isDark ? "rgba(15,15,20,0.70)" : "rgba(255,255,255,0.80)",
        backdropFilter: "blur(8px)",
        border: `1px solid ${isDark ? "rgba(255,255,255,0.08)" : "rgba(0,0,0,0.08)"}`,
        fontSize: 11,
        color: isDark ? "#e4e4e7" : "#3f3f46",
        boxSizing: "border-box",
      }}>
        <span style={{ fontSize: 10, fontWeight: 600, opacity: 0.8, flexShrink: 0 }}>Spread</span>
        <input
          type="range"
          min="20"
          max="200"
          value={spreadValue}
          onChange={e => setSpreadValue(Number(e.target.value))}
          style={{ flex: 1, minWidth: 0, accentColor: isDark ? "#a1a1aa" : "#71717a" }}
        />
        <span style={{ fontSize: 10, fontVariantNumeric: "tabular-nums", opacity: 0.7, minWidth: 32, textAlign: "right", flexShrink: 0 }}>
          {spreadValue}%
        </span>
      </div>

      {/* Unified Legend — single horizontal line */}
      <div className="absolute top-2 left-3 right-3 z-10 flex items-center gap-3 text-xs overflow-x-auto pointer-events-auto shrink-0">
        {nodeTypesInView.length > 0 && (
          <div className="flex items-center gap-1.5">
            <span className="text-xs text-muted-foreground mr-1 font-medium">Nodes:</span>
            {nodeTypesInView.map(c => {
              const hidden = hiddenNodeTypes.has(c.name)
              return (
                <button key={c.name} onClick={() => toggleNodeType(c.name)}
                  className="flex items-center gap-1 px-1.5 py-0.5 rounded-full border transition-opacity shrink-0"
                  style={{
                    borderColor: c.color,
                    color: hidden ? (isDark ? "#71717a" : "#a1a1aa") : c.color,
                    opacity: hidden ? 0.4 : 1,
                    background: hidden ? "transparent" : `${c.color}18`,
                  }}>
                  <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ background: hidden ? "transparent" : c.color, border: hidden ? `1px solid ${c.color}` : "none" }} />
                  <span className={hidden ? "line-through" : ""}>{c.name}</span>
                </button>
              )
            })}
          </div>
        )}
        {edgeTypesInView.length > 0 && (
          <>
            {nodeTypesInView.length > 0 && <div className="w-px h-4 bg-border shrink-0" />}
            <div className="flex items-center gap-1.5">
              <span className="text-xs text-muted-foreground mr-1 font-medium">Relationships:</span>
              {edgeTypesInView.map(t => {
                const hidden = hiddenEdgeTypes.has(t)
                const color = EDGE_COLORS[t] ?? EDGE_COLOR_FALLBACK
                return (
                  <button key={t} onClick={() => toggleEdgeType(t)}
                    className="flex items-center gap-1 px-1.5 py-0.5 rounded-full border transition-opacity shrink-0"
                    style={{
                      borderColor: color,
                      color: hidden ? (isDark ? "#71717a" : "#a1a1aa") : color,
                      opacity: hidden ? 0.4 : 1,
                      background: hidden ? "transparent" : `${color}18`,
                    }}>
                    <span className="w-3 h-px shrink-0" style={{ background: hidden ? (isDark ? "#52525b" : "#d4d4d8") : color }} />
                    <span className={hidden ? "line-through" : ""}>{t}</span>
                  </button>
                )
              })}
            </div>
          </>
        )}
        <div className="w-px h-4 bg-border shrink-0" />
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-muted-foreground mr-1 font-medium">Criticality:</span>
          {(["critical", "high", "medium", "low", "minimal"] as const).map(lvl => {
            const CRIT_COLORS: Record<string, string> = {
              critical: '#ef4444', high: '#f97316', medium: '#eab308', low: '#22c55e', minimal: '#6b7280'
            }
            const col = CRIT_COLORS[lvl]
            return (
              <div key={lvl} className="flex items-center gap-1 px-1.5 py-0.5 rounded-full border border-border bg-background/50 shrink-0">
                <span className="w-1.5 h-1.5 rounded-full shrink-0" style={{ backgroundColor: col }} />
                <span className="capitalize text-muted-foreground">{lvl}</span>
              </div>
            )
          })}
        </div>
      </div>
      <ReactECharts
        onChartReady={(instance: any) => { echartsRef.current = instance }}
        option={option}
        onEvents={onEvents}
        style={{ width: "100%", height: "100%" }}
        theme={isDark ? "dark" : undefined}
        opts={{ renderer: "canvas", useDirtyRect: true }} // Enable incremental rendering
        notMerge={true} // Full re-render to handle selection/expansion properly
        lazyUpdate={false} // Ensure immediate updates for selection changes
      />

      {/* Connections panel removed — available in the Table view's third column */}
    </div>
  )
})

// ── Page ──────────────────────────────────────────────────────────────────────

// ── NodeExplanationPanel ──────────────────────────────────────────────────────
interface NodeExplanationPanelProps {
  componentId: string
  componentName: string
  componentAnalysis: any | null
}

function NodeExplanationPanel({ componentId, componentName, componentAnalysis }: NodeExplanationPanelProps) {
  if (!componentAnalysis) return null

  const CRIT_COLORS: Record<string, string> = {
    critical: 'bg-red-500/10 text-red-400 border-red-500/20',
    high: 'bg-orange-500/10 text-orange-400 border-orange-500/20',
    medium: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20',
    low: 'bg-green-500/10 text-green-400 border-green-500/20',
    minimal: 'bg-muted/30 text-muted-foreground border-border',
  }

  const CRIT_TEXT_COLORS: Record<string, string> = {
    critical: 'text-red-400',
    high: 'text-orange-400',
    medium: 'text-yellow-400',
    low: 'text-green-400',
    minimal: 'text-muted-foreground',
  }

  const lvl = componentAnalysis?.criticality_level?.toLowerCase() || "minimal"
  const scores = componentAnalysis?.scores

  return (
    <div className="flex flex-col gap-3 p-3 border-b border-border bg-muted/10 shrink-0">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h4 className="text-[11px] font-semibold truncate text-foreground">{componentName}</h4>
          <span className="text-[9px] text-muted-foreground truncate block">{componentId}</span>
        </div>
        <Badge className={`text-[8px] px-1 py-0 font-bold uppercase border ${CRIT_COLORS[lvl] ?? CRIT_COLORS.minimal}`}>
          {lvl}
        </Badge>
      </div>

      {scores && (
        <div className="space-y-1">
          <div className="text-[8px] font-semibold text-muted-foreground uppercase tracking-wider">RMAV Quality</div>
          {(['reliability', 'maintainability', 'availability', 'security'] as const).map(dim => {
            const val = scores[dim] ?? 0
            const pct = Math.round((1 - val) * 100)
            const dimLvl = val >= 0.8 ? "critical" : val >= 0.6 ? "high" : val >= 0.4 ? "medium" : val >= 0.2 ? "low" : "minimal"
            return (
              <div key={dim} className="space-y-0.5">
                <div className="flex justify-between text-[8.5px]">
                  <span className="capitalize text-muted-foreground">{dim}</span>
                  <span className={`font-semibold ${CRIT_TEXT_COLORS[dimLvl] ?? 'text-foreground'}`}>{pct}%</span>
                </div>
                <div className="h-1 w-full bg-muted rounded-full overflow-hidden border border-border/40">
                  <div
                    className={`h-full rounded-full transition-all duration-300 ${
                      dimLvl === 'critical' ? 'bg-red-500' :
                      dimLvl === 'high' ? 'bg-orange-500' :
                      dimLvl === 'medium' ? 'bg-yellow-500' :
                      dimLvl === 'low' ? 'bg-green-500' :
                      'bg-muted-foreground'
                    }`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────

function BrowserPageContent() {
  const searchParams = useSearchParams()
  const nodeId = searchParams?.get('node') ?? null
  const { status, initialLoadComplete } = useConnection()
  const isConnected = status === "connected"

  const [hierarchy, setHierarchy] = useState<Record<string, CsmsGroup>>({})
  const [totalApps, setTotalApps] = useState(0)
  const [nodesList, setNodesList] = useState<any[]>([])
  const [topicsList, setTopicsList] = useState<any[]>([])
  const [brokersList, setBrokersList] = useState<any[]>([])
  const [libsList, setLibsList] = useState<any[]>([])
  const [openSet, setOpenSet] = useState<Set<string>>(new Set())
  const [search, setSearch] = useState("")
  const [sideSearch, setSideSearch] = useState("")
  const [sideInitialTab, setSideInitialTab] = useState<"nodes" | "brokers" | "libs" | "apps" | "topics">("nodes")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedNode, setSelectedNode] = useState<SelectedNode | null>(null)
  const [hierSelectedNode, setHierSelectedNode] = useState<SelectedNode | null>(null)
  const [activeTab, setActiveTab] = useState<"browse" | "graph" | "forcegraph" | "overview">("browse")
  const [layerGraphLinks, setLayerGraphLinks] = useState<Array<{ source: string; target: string; type: string; weight?: number }>>([])
  const [layerLinksLoading, setLayerLinksLoading] = useState(false)
  const layerLinksLoadedRef = useRef(false)

  // RMAV analysis-store hooks
  const { cache: explanationsCache } = useAnalysis()
  const systemAnalysis = explanationsCache['layer:system']
  const selectedId = (selectedNode?.payload as any)?.id ? String((selectedNode.payload as any).id) : null
  const componentAnalysis = systemAnalysis?.components?.find(c => c.id === selectedId)

  // Export-to-PNG refs — populated by child graph components
  const forceGraphExportRef = useRef<(() => void) | null>(null)
  const systemGraphExportRef = useRef<(() => void) | null>(null)
  const overviewGraphExportRef = useRef<(() => void) | null>(null)

  const toggle = useCallback((key: string) => {
    setOpenSet((prev) => {
      const next = new Set(prev)
      next.has(key) ? next.delete(key) : next.add(key)
      return next
    })
  }, [])

  const fetchData = async (targetNodeId: string | null) => {
    setLoading(true)
    setError(null)
    try {
      const [apps, nodes, topics, brokers, libs] = await Promise.all([
        apiClient.getAllApps(),
        apiClient.getComponentsByType('Node'),
        apiClient.getComponentsByType('Topic'),
        apiClient.getComponentsByType('Broker'),
        apiClient.getComponentsByType('Library'),
      ])
      const h = buildHierarchy(apps)
      setHierarchy(h)
      setTotalApps(apps.length)
      setNodesList(nodes)
      setTopicsList(topics)
      setBrokersList(brokers)
      setLibsList(libs)

      // Auto-select node from URL ?node= param
      if (targetNodeId) {
        let found = false
        outer: for (const [csmsKey, csms] of Object.entries(h)) {
          for (const [cssKey, css] of Object.entries(csms.css)) {
            for (const [csciKey, csci] of Object.entries(css.csci)) {
              for (const [cscKey, csc] of Object.entries(csci.csc)) {
                const app = csc.apps.find((a: AppNode) => a.id === targetNodeId)
                if (app) {
                  const label = app.csu ?? app.name ?? app.id ?? "?"
                  const csmsOpenKey = `csms:${csmsKey}`
                  const cssOpenKey  = `css:${csmsKey}/${cssKey}`
                  const csciOpenKey = `csci:${csmsKey}/${cssKey}/${csciKey}`
                  const cscOpenKey  = `csc:${csmsKey}/${cssKey}/${csciKey}/${cscKey}`
                  setOpenSet(new Set([csmsOpenKey, cssOpenKey, csciOpenKey, cscOpenKey]))
                  setSelectedNode({ kind: "app", key: `app:${app.id}`, label, path: [csmsKey, cssKey, csciKey, cscKey, label], payload: app })
                  setSideInitialTab("apps")
                  found = true
                  break outer
                }
              }
            }
          }
        }
        if (!found) {
          // Fall back to nodesList / topicsList / brokersList / libsList
          const allExtra: any[] = [...nodes, ...topics, ...brokers, ...libs]
          const extra = allExtra.find((n: any) => n.id === targetNodeId)
          if (extra) {
            const kind: SelectedKind = topics.some((t: any) => t.id === targetNodeId) ? "topic" : "node"
            const label = extra.name ?? extra.id ?? "?"
            setSelectedNode({ kind, key: `${kind}:${extra.id}`, label, path: [label], payload: extra })
            setSideInitialTab(
              topics.some((t: any) => t.id === targetNodeId) ? "topics" :
              brokers.some((b: any) => b.id === targetNodeId) ? "brokers" :
              libs.some((l: any) => l.id === targetNodeId) ? "libs" : "nodes"
            )
          } else {
            // Unknown id — fall back to default first-CSMS selection
            const firstCsmsKey = sortKeys(Object.keys(h))[0]
            if (firstCsmsKey) {
              const firstCsms = h[firstCsmsKey]
              setOpenSet(new Set([`csms:${firstCsmsKey}`]))
              setSelectedNode({ kind: "csms", key: `csms:${firstCsmsKey}`, label: firstCsms.name, path: [firstCsms.name], payload: firstCsms })
              setSideInitialTab("nodes")
            }
          }
        }
      } else {
        setSideInitialTab("nodes")
        if (nodes.length > 0) {
          const first = nodes[0]
          setSelectedNode({ kind: "node", key: `node:${first.id}`, label: first.name ?? first.id ?? "?", path: [first.name ?? first.id ?? "?"], payload: first })
        }
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (isConnected) { fetchData(nodeId); fetchLayerLinks() }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isConnected, nodeId])

  const appsList = useMemo(() =>
    Object.values(hierarchy).flatMap(csms =>
      Object.values(csms.css).flatMap(css =>
        Object.values(css.csci).flatMap(csci =>
          Object.values(csci.csc).flatMap(csc => csc.apps)
        )
      )
    )
  , [hierarchy])

  // Flat list of all hierarchy nodes for the Browse tab jump-to dropdown
  type FlatBrowseEntry = { kind: SelectedKind; key: string; label: string; path: string[]; payload: CsmsGroup | CssGroup | CsciGroup | CscGroup | AppNode; appCount: number }
  const flatBrowseNodes = useMemo<FlatBrowseEntry[]>(() => {
    const result: FlatBrowseEntry[] = []
    for (const [csmsKey, csms] of Object.entries(hierarchy)) {
      const csmsPath = [csms.name]
      result.push({ kind: "csms", key: `csms:${csmsKey}`, label: csms.name, path: csmsPath, payload: csms, appCount: Object.values(csms.css).flatMap(c => Object.values(c.csci)).flatMap(c => Object.values(c.csc)).flatMap(c => c.apps).length })
      for (const [cssKey, css] of Object.entries(csms.css)) {
        const cssPath = [...csmsPath, css.name]
        result.push({ kind: "css", key: `css:${csmsKey}/${cssKey}`, label: css.name, path: cssPath, payload: css, appCount: Object.values(css.csci).flatMap(c => Object.values(c.csc)).flatMap(c => c.apps).length })
        for (const [csciKey, csci] of Object.entries(css.csci)) {
          const csciPath = [...cssPath, csci.name]
          result.push({ kind: "csci", key: `csci:${csmsKey}/${cssKey}/${csciKey}`, label: csci.name, path: csciPath, payload: csci, appCount: Object.values(csci.csc).flatMap(c => c.apps).length })
          for (const [cscKey, csc] of Object.entries(csci.csc)) {
            const cscPath = [...csciPath, csc.name]
            result.push({ kind: "csc", key: `csc:${csmsKey}/${cssKey}/${csciKey}/${cscKey}`, label: csc.name, path: cscPath, payload: csc, appCount: csc.apps.length })
            for (const app of csc.apps)
              result.push({ kind: "app", key: `app:${app.id}`, label: app.name ?? app.id, path: [...cscPath, app.name ?? app.id], payload: app, appCount: 1 })
          }
        }
      }
    }
    return result
  }, [hierarchy])

  const filteredBrowseNodes = useMemo(() => {
    const qb = search.trim().toLowerCase()
    if (!qb) return []
    return flatBrowseNodes.filter(n => n.label.toLowerCase().includes(qb) || n.key.toLowerCase().includes(qb)).slice(0, 25)
  }, [flatBrowseNodes, search])

  const jumpToBrowseNode = useCallback((entry: FlatBrowseEntry) => {
    setSelectedNode({ kind: entry.kind, key: entry.key, label: entry.label, path: entry.path, payload: entry.payload })
    if (entry.kind === 'app') {
      // App keys are "app:appId" — must search hierarchy to find ancestor CSMS/CSS/CSCI/CSC keys
      const appId = entry.key.slice(4)
      outer: for (const [csmsKey, csms] of Object.entries(hierarchy)) {
        for (const [cssKey, css] of Object.entries(csms.css)) {
          for (const [csciKey, csci] of Object.entries(css.csci)) {
            for (const [cscKey, csc] of Object.entries(csci.csc)) {
              if (csc.apps.some(a => a.id === appId)) {
                setOpenSet(prev => {
                  const next = new Set(prev)
                  next.add(`csms:${csmsKey}`)
                  next.add(`css:${csmsKey}/${cssKey}`)
                  next.add(`csci:${csmsKey}/${cssKey}/${csciKey}`)
                  next.add(`csc:${csmsKey}/${cssKey}/${csciKey}/${cscKey}`)
                  return next
                })
                break outer
              }
            }
          }
        }
      }
    } else {
      setOpenSet(prev => {
        const next = new Set(prev)
        const parts = entry.key.replace(/^[^:]+:/, "").split("/")
        const [csmsKey, cssKey, csciKey] = parts
        if (csmsKey) next.add(`csms:${csmsKey}`)
        if (cssKey)  next.add(`css:${csmsKey}/${cssKey}`)
        if (csciKey) next.add(`csci:${csmsKey}/${cssKey}/${csciKey}`)
        next.add(entry.key)
        return next
      })
    }
    setSearch("")
  }, [hierarchy])

  const expandPath = useCallback((keys: string[]) => {
    setOpenSet(prev => {
      const next = new Set(prev)
      keys.forEach(k => next.add(k))
      return next
    })
  }, [])

  const fetchLayerLinks = useCallback(async () => {
    if (layerLinksLoadedRef.current) return
    layerLinksLoadedRef.current = true
    setLayerLinksLoading(true)
    try {
      const data = await apiClient.getGraphData()
      setLayerGraphLinks(data.links.map(l => ({ source: String(l.source), target: String(l.target), type: l.type, weight: l.weight })))
    } catch {
      // edges are optional — silently ignore failures
    } finally {
      setLayerLinksLoading(false)
    }
  }, [])

  const handleTabChange = useCallback((tab: string) => {
    setActiveTab(tab as "browse" | "graph" | "forcegraph" | "overview")
    if (tab === "forcegraph" || tab === "browse" || tab === "overview") fetchLayerLinks()
  }, [fetchLayerLinks])

  const allNodeLabels = useMemo(() => {
    const m = new Map<string, { label: string; type: string }>()
    appsList.forEach(a => m.set(String(a.id), { label: a.csu ?? a.name ?? a.id ?? String(a.id), type: "Application" }))
    nodesList.forEach(n => m.set(String(n.id), { label: n.name ?? String(n.id), type: n.type ?? "Node" }))
    topicsList.forEach(t => m.set(String(t.id), { label: t.name ?? String(t.id), type: "Topic" }))
    brokersList.forEach(b => m.set(String(b.id), { label: b.name ?? String(b.id), type: "Broker" }))
    libsList.forEach(l => m.set(String(l.id), { label: l.name ?? String(l.id), type: "Library" }))
    return m
  }, [appsList, nodesList, topicsList, brokersList, libsList])

  const handleLayersNodeClick = useCallback((rawNode: any, nodeType: SwimlaneType) => {
    const id = String(rawNode.id ?? "")
    if (nodeType === "Application") {
      const app = appsList.find(a => a.id === id)
      if (app) {
        const label = app.csu ?? app.name ?? id
        setSelectedNode({ kind: "app", key: `app:${id}`, label, path: [label], payload: app })
        setSideInitialTab("apps")
        return
      }
    }
    const kindMap: Record<SwimlaneType, SelectedKind> = {
      Node: "node", Application: "app", Topic: "topic", Library: "node", Broker: "node",
    }
    const tabMap: Record<SwimlaneType, "nodes" | "brokers" | "libs" | "apps" | "topics"> = {
      Node: "nodes", Application: "apps", Topic: "topics", Library: "libs", Broker: "brokers",
    }
    const kind: SelectedKind = kindMap[nodeType] ?? "node"
    const label: string = rawNode.name ?? rawNode.id ?? "?"
    setSelectedNode({ kind, key: `${kind}:${id}`, label, path: [label], payload: rawNode })
    setSideInitialTab(tabMap[nodeType] ?? "nodes")
  }, [appsList])

  if (!initialLoadComplete || (isConnected && loading && nodesList.length === 0)) {
    return (
      <AppLayout title="Explorer" description="Browse your system topology">
        <div className="space-y-3">
          {/* Tabs skeleton */}
          <div className="flex items-center justify-between mb-2 shrink-0">
            <div className="bg-background border border-border inline-flex h-9 w-fit items-center justify-center rounded-lg p-[3px] gap-0.5">
              <div className="inline-flex items-center gap-1.5 h-[calc(100%-2px)] px-3 rounded-md bg-muted animate-pulse">
                <Skeleton className="h-4 w-4 rounded" />
                <Skeleton className="h-3.5 w-8" />
              </div>
              <div className="inline-flex items-center gap-1.5 h-[calc(100%-2px)] px-3 rounded-md">
                <Skeleton className="h-4 w-4 rounded" />
                <Skeleton className="h-3.5 w-10" />
              </div>
              <div className="inline-flex items-center gap-1.5 h-[calc(100%-2px)] px-3 rounded-md">
                <Skeleton className="h-4 w-4 rounded" />
                <Skeleton className="h-3.5 w-12" />
              </div>
              <div className="inline-flex items-center gap-1.5 h-[calc(100%-2px)] px-3 rounded-md">
                <Skeleton className="h-4 w-4 rounded" />
                <Skeleton className="h-3.5 w-14" />
              </div>
            </div>
          </div>
          {/* 3-column layout skeleton */}
          <div className="flex gap-0 border border-border rounded-lg overflow-hidden" style={{ minHeight: "520px" }}>
            {/* Left: list panel */}
            <div className="w-56 shrink-0 border-r border-border p-3 space-y-2">
              <Skeleton className="h-7 w-full rounded-md mb-3" />
              {Array.from({ length: 14 }).map((_, i) => (
                <div key={i} className="flex items-center gap-2 px-1">
                  <Skeleton className="h-3" style={{ width: `${50 + (i * 13) % 40}%` }} />
                </div>
              ))}
            </div>
            {/* Middle: detail panel */}
            <div className="flex-1 p-4 space-y-3 border-r border-border">
              <div className="flex items-center gap-3 pb-3 border-b border-border/60">
                <Skeleton className="h-4 w-40" />
                <Skeleton className="h-5 w-16 ml-auto rounded-full" />
              </div>
              {Array.from({ length: 10 }).map((_, i) => (
                <div key={i} className="flex items-center gap-4 py-1">
                  <Skeleton className="h-3" style={{ width: `${25 + (i * 17) % 30}%` }} />
                  <Skeleton className="h-3" style={{ width: `${20 + (i * 11) % 25}%` }} />
                </div>
              ))}
            </div>
            {/* Right: connections panel */}
            <div className="w-64 shrink-0 p-3 space-y-2">
              <Skeleton className="h-2.5 w-24 mb-1.5" />
              <Skeleton className="h-2.5 w-16 mb-3" />
              <Skeleton className="h-7 w-full rounded-md mb-2" />
              {Array.from({ length: 8 }).map((_, i) => (
                <div key={i} className="flex items-center gap-2 py-1.5 border-b border-border/40">
                  <Skeleton className="h-3" style={{ width: `${40 + (i * 19) % 40}%` }} />
                  <Skeleton className="h-4 w-14 rounded ml-auto" />
                </div>
              ))}
            </div>
          </div>
        </div>
      </AppLayout>
    )
  }
  if (!isConnected) {
    return <AppLayout title="Explorer" description="Browse your system topology"><NoConnectionInfo description="Connect to your Neo4j database to explore topology" /></AppLayout>
  }

  const q = search.toLowerCase()
  const csmsKeys = sortKeys(Object.keys(hierarchy))

  return (
    <AppLayout
      title="Explorer"
      description="Browse your system topology"
    >
      <div className="flex flex-col gap-5 h-full">
        {error && (
          <div className="rounded-md bg-destructive/10 border border-destructive/30 text-destructive px-4 py-3 text-sm shrink-0">
            {error}
          </div>
        )}

        <Tabs value={activeTab} onValueChange={handleTabChange} className="flex flex-col flex-1 min-h-0 gap-0">
          <div className="flex items-center justify-between mb-2 shrink-0">
            <TabsList className="bg-background border border-border">
              <TabsTrigger value="browse" className="flex items-center gap-2">
                <List className="h-4 w-4" />List
              </TabsTrigger>
              <TabsTrigger value="forcegraph" className="flex items-center gap-2">
                <Share2 className="h-4 w-4" />Graph
              </TabsTrigger>
              <TabsTrigger value="graph" className="flex items-center gap-2">
                <Network className="h-4 w-4" />Tree
              </TabsTrigger>
              <TabsTrigger value="overview" className="flex items-center gap-2">
                <Layers className="h-4 w-4" />Overview
              </TabsTrigger>
            </TabsList>

            {(activeTab === "forcegraph" || activeTab === "graph" || activeTab === "overview") && (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-8 gap-1.5 text-xs"
                    onClick={() => {
                      if (activeTab === "forcegraph") forceGraphExportRef.current?.()
                      else if (activeTab === "overview") overviewGraphExportRef.current?.()
                      else systemGraphExportRef.current?.()
                    }}
                  >
                    <Download className="h-3.5 w-3.5" />
                    Save as PNG
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Download current graph view as PNG</TooltipContent>
              </Tooltip>
            )}
          </div>

          {/* ── Browse tab ── */}
          <TabsContent forceMount value="browse" className="flex-1 min-h-0 mt-0 data-[state=inactive]:hidden">
            <div className="flex border border-border rounded-lg overflow-hidden h-full" style={{ minHeight: "520px" }}>
              {/* Left: Tabbed list panel */}
              <div className="w-72 flex flex-col overflow-hidden shrink-0 border-r border-border">
                <SideListPanel
                  nodesList={nodesList}
                  appsList={appsList}
                  topicsList={topicsList}
                  brokersList={brokersList}
                  libsList={libsList}
                  hierarchy={hierarchy}
                  openSet={openSet}
                  toggle={toggle}
                  expandPath={expandPath}
                  selectedKey={selectedNode?.key ?? null}
                  onSelect={setSelectedNode}
                  search={sideSearch}
                  onSearchChange={setSideSearch}
                  loading={loading}
                  initialTab={sideInitialTab}
                />
              </div>

              {/* Middle: Detail / table panel */}
              <div className="flex-1 overflow-auto min-w-0 border-r border-border">
                {loading && !csmsKeys.length
                  ? (
                    <div className="p-4 flex flex-col gap-3">
                      <div className="flex items-center gap-3 pb-3 border-b border-border/60">
                        <Skeleton className="h-4 w-40" />
                        <Skeleton className="h-5 w-16 ml-auto rounded-full" />
                      </div>
                      {Array.from({ length: 12 }).map((_, i) => (
                        <div key={i} className="flex items-center gap-4 py-1">
                          <Skeleton className="h-3" style={{ width: `${30 + (i * 17) % 30}%` }} />
                          <Skeleton className="h-3" style={{ width: `${20 + (i * 11) % 25}%` }} />
                        </div>
                      ))}
                    </div>
                  )
                  : selectedNode
                    ? <NodeDetailPanel node={selectedNode} />
                    : <EmptyDetailState />
                }
              </div>

              {/* Right: Connections column */}
              <div className="w-64 shrink-0 flex flex-col overflow-hidden">
                <ConnectionsColumn
                  selectedNode={selectedNode}
                  links={layerGraphLinks}
                  nodeLabels={allNodeLabels}
                  loading={layerLinksLoading}
                  onSelect={(id) => {
                    const info = allNodeLabels.get(id)
                    if (!info) return
                    const kind: SelectedKind = info.type === "Application" ? "app" : info.type === "Topic" ? "topic" : "node"
                    const app = appsList.find(a => String(a.id) === id)
                    const raw = app
                      ?? nodesList.find(n => String(n.id) === id)
                      ?? topicsList.find(t => String(t.id) === id)
                      ?? brokersList.find(b => String(b.id) === id)
                      ?? libsList.find(l => String(l.id) === id)
                    if (!raw) return
                    setSelectedNode({ kind, key: `${kind}:${id}`, label: info.label, path: [info.label], payload: raw })
                  }}
                />
              </div>
            </div>
          </TabsContent>

          {/* ── Graph tab ── */}
          <TabsContent forceMount value="graph" className="flex-1 min-h-0 mt-0 h-full data-[state=inactive]:hidden">
            {csmsKeys.length > 0
              ? (
                <div className="flex border border-border rounded-lg overflow-hidden h-full" style={{ minHeight: "520px" }}>
                  <div className="flex-1 min-w-0 overflow-hidden">
                    <HierarchyGraph
                      hierarchy={hierarchy}
                      extraNodes={[...nodesList, ...topicsList, ...brokersList, ...libsList]}
                      exportFnRef={systemGraphExportRef}
                      onSelectInfo={(pathKey, name, nodeType) => {
                        const info = allNodeLabels.get(pathKey)
                        const type = nodeType ?? info?.type ?? "Application"
                        const kind: SelectedKind = type === "Topic" ? "topic" : type === "Application" ? "app" : "node"
                        const label = info?.label ?? name
                        const raw = appsList.find(a => String(a.id) === pathKey)
                          ?? nodesList.find(n => String(n.id) === pathKey)
                          ?? topicsList.find(t => String(t.id) === pathKey)
                          ?? brokersList.find(b => String(b.id) === pathKey)
                          ?? libsList.find(l => String(l.id) === pathKey)
                        setHierSelectedNode({ kind, key: `${kind}:${pathKey}`, label, path: [label], payload: raw ?? { id: pathKey, name: label } })
                      }}
                    />
                  </div>
                  <div className="w-72 shrink-0 border-l border-border flex flex-col overflow-hidden">
                    <GraphTabSidePanel
                      selectedNode={hierSelectedNode}
                      links={layerGraphLinks}
                      nodeLabels={allNodeLabels}
                      loading={layerLinksLoading}
                      onSelect={(id) => {
                        const info = allNodeLabels.get(id)
                        if (!info) return
                        const kind: SelectedKind = info.type === "Application" ? "app" : info.type === "Topic" ? "topic" : "node"
                        const label = info.label
                        const raw = appsList.find(a => String(a.id) === id)
                          ?? nodesList.find(n => String(n.id) === id)
                          ?? topicsList.find(t => String(t.id) === id)
                          ?? brokersList.find(b => String(b.id) === id)
                          ?? libsList.find(l => String(l.id) === id)
                        setHierSelectedNode({ kind, key: `${kind}:${id}`, label, path: [label], payload: raw ?? { id, name: label } })
                      }}
                    />
                  </div>
                </div>
              )
              : <p className="text-center text-muted-foreground py-12 text-sm">No data loaded yet.</p>
            }
          </TabsContent>

          {/* ── Overview tab ── */}
          <TabsContent forceMount value="overview" className="flex-1 min-h-0 mt-0 h-full data-[state=inactive]:hidden">
            <div className="border border-border rounded-lg overflow-hidden h-full" style={{ minHeight: "520px" }}>
              <GraphOverviewEChart
                nodesList={nodesList}
                appsList={appsList}
                topicsList={topicsList}
                libsList={libsList}
                brokersList={brokersList}
                graphLinks={layerGraphLinks}
                linksLoading={layerLinksLoading}
                exportFnRef={overviewGraphExportRef}
              />
            </div>
          </TabsContent>

          {/* ── Force-graph tab ── */}
          <TabsContent forceMount value="forcegraph" className="flex-1 min-h-0 mt-0 h-full data-[state=inactive]:hidden">
            {(appsList.length > 0 || nodesList.length > 0 || topicsList.length > 0)
              ? (
                <div className="flex border border-border rounded-lg overflow-hidden h-full" style={{ minHeight: "520px" }}>
                  <div className="flex-1 min-w-0 overflow-hidden">
                    <ForceGraphEChart
                      nodesList={nodesList}
                      appsList={appsList}
                      topicsList={topicsList}
                      libsList={libsList}
                      brokersList={brokersList}
                      graphLinks={layerGraphLinks}
                      linksLoading={layerLinksLoading}
                      selectedKey={selectedNode?.key ?? null}
                      onNodeClick={handleLayersNodeClick}
                      exportFnRef={forceGraphExportRef}
                    />
                  </div>
                  <div className="w-64 shrink-0 border-l border-border flex flex-col overflow-hidden">
                    <NodeExplanationPanel
                      componentId={selectedId || ""}
                      componentName={selectedNode?.label || ""}
                      componentAnalysis={componentAnalysis}
                    />
                    <ConnectionsColumn
                      selectedNode={selectedNode}
                      links={layerGraphLinks}
                      nodeLabels={allNodeLabels}
                      loading={layerLinksLoading}
                      onSelect={(id) => {
                        const info = allNodeLabels.get(id)
                        if (!info) return
                        const kind: SelectedKind = info.type === "Application" ? "app" : info.type === "Topic" ? "topic" : "node"
                        const raw = appsList.find(a => String(a.id) === id)
                          ?? nodesList.find(n => String(n.id) === id)
                          ?? topicsList.find(t => String(t.id) === id)
                          ?? brokersList.find(b => String(b.id) === id)
                          ?? libsList.find(l => String(l.id) === id)
                        if (!raw) return
                        setSelectedNode({ kind, key: `${kind}:${id}`, label: info.label, path: [info.label], payload: raw })
                      }}
                    />
                  </div>
                </div>
              )
              : <p className="text-center text-muted-foreground py-12 text-sm">No data loaded yet.</p>
            }
          </TabsContent>
        </Tabs>
      </div>
    </AppLayout>
  )
}

export default function BrowserPage() {
  return (
    <Suspense fallback={null}>
      <BrowserPageContent />
    </Suspense>
  )
}

// ── Simple Virtual List ───────────────────────────────────────────────────────
// Renders only the rows visible in the scroll viewport — prevents thousands of
// DOM nodes when apps/nodes/topics lists are large.
function SimpleVirtualList({ items, renderItem, itemHeight = 34 }: {
  items: any[]
  renderItem: (item: any, index: number) => React.ReactNode
  itemHeight?: number
}) {
  const outerRef = useRef<HTMLDivElement>(null)
  const [scrollTop, setScrollTop] = useState(0)
  const [viewHeight, setViewHeight] = useState(400)
  const overscan = 8

  useEffect(() => {
    const el = outerRef.current
    if (!el) return
    setViewHeight(el.clientHeight)
    const ro = new ResizeObserver(() => setViewHeight(el.clientHeight))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const totalHeight = items.length * itemHeight
  const startIdx = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan)
  const endIdx   = Math.min(items.length, Math.ceil((scrollTop + viewHeight) / itemHeight) + overscan)

  return (
    <div ref={outerRef} style={{ overflowY: "auto", height: "100%" }}
      onScroll={e => setScrollTop((e.currentTarget).scrollTop)}>
      <div style={{ height: totalHeight, position: "relative" }}>
        <div style={{ position: "absolute", top: startIdx * itemHeight, width: "100%" }}>
          {items.slice(startIdx, endIdx).map((item, i) => renderItem(item, startIdx + i))}
        </div>
      </div>
    </div>
  )
}

// ── Connections Column ────────────────────────────────────────────────────────

const CONN_TYPE_COLORS: Record<string, string> = {
  app_to_app: "#4CBCD0", app_to_broker: "#f59e0b", app_to_lib: "#a78bfa",
  node_to_node: "#C570CE", node_to_broker: "#f97316", broker_to_broker: "#10b981",
}
function connTypeColor(type: string) { return CONN_TYPE_COLORS[type] ?? "#71717a" }

function ConnectionsColumn({ selectedNode, links, nodeLabels, onSelect, loading = false }: {
  selectedNode: SelectedNode | null
  links: Array<{ source: string; target: string; type: string; weight?: number }>
  nodeLabels: Map<string, { label: string; type: string }>
  onSelect: (id: string) => void
  loading?: boolean
}) {
  const nodeId = selectedNode ? selectedNode.key.replace(/^[^:]+:/, "") : null
  const [activeRelTypes, setActiveRelTypes] = useState<Set<string>>(new Set())
  const [activeNodeTypes, setActiveNodeTypes] = useState<Set<string>>(new Set())
  const [connSearch, setConnSearch] = useState("")

  const DERIVED_TYPES = new Set(["node_to_node", "node_to_broker", "DEPENDS_ON", "CONNECTS_TO"])

  const allConns = useMemo(() => {
    if (!nodeId) return []
    const out = links.filter(l => l.source === nodeId && !DERIVED_TYPES.has(l.type)).map(l => ({ peerId: l.target, type: l.type }))
    const inn = links.filter(l => l.target === nodeId && !DERIVED_TYPES.has(l.type)).map(l => ({ peerId: l.source, type: l.type }))
    return [...out, ...inn]
  }, [nodeId, links])

  const allRelTypes  = useMemo(() => Array.from(new Set(allConns.map(c => c.type))).sort(), [allConns])
  const allNodeTypes = useMemo(() => Array.from(new Set(allConns.map(c => nodeLabels.get(c.peerId)?.type ?? "Unknown"))).sort(), [allConns, nodeLabels])

  // Reset filters when selected node changes
  useEffect(() => { setActiveRelTypes(new Set()); setActiveNodeTypes(new Set()); setConnSearch("") }, [nodeId])

  const filtered = useMemo(() => {
    const q = connSearch.toLowerCase()
    return allConns.filter(c => {
      if (activeRelTypes.size > 0 && !activeRelTypes.has(c.type)) return false
      if (activeNodeTypes.size > 0 && !activeNodeTypes.has(nodeLabels.get(c.peerId)?.type ?? "Unknown")) return false
      if (q && !(nodeLabels.get(c.peerId)?.label ?? c.peerId).toLowerCase().includes(q)) return false
      return true
    })
  }, [allConns, activeRelTypes, activeNodeTypes, nodeLabels, connSearch])

  // Group by peer node type
  const grouped = useMemo(() => {
    const map = new Map<string, typeof filtered>()
    for (const c of filtered) {
      const peerType = nodeLabels.get(c.peerId)?.type ?? "Unknown"
      if (!map.has(peerType)) map.set(peerType, [])
      map.get(peerType)!.push(c)
    }
    return Array.from(map.entries()).sort(([a], [b]) => a.localeCompare(b))
  }, [filtered, nodeLabels])

  const toggleRelType  = (t: string) => setActiveRelTypes(prev => { const n = new Set(prev); n.has(t) ? n.delete(t) : n.add(t); return n })
  const toggleNodeType = (t: string) => setActiveNodeTypes(prev => { const n = new Set(prev); n.has(t) ? n.delete(t) : n.add(t); return n })

  const NODE_TYPE_COLORS: Record<string, string> = {
    Application: "#4CBCD0", Node: "#C570CE", Topic: "#7DAA7A", Broker: "#f59e0b", Library: "#a78bfa",
  }
  const nodeTypeColor = (t: string) => NODE_TYPE_COLORS[t] ?? "#71717a"

  const FilterChips = ({ items, active, toggle, colorFn }: {
    items: string[]; active: Set<string>; toggle: (t: string) => void; colorFn: (t: string) => string
  }) => (
    <div className="flex flex-wrap gap-1">
      {items.map(t => {
        const color = colorFn(t)
        const isActive = active.has(t)
        return (
          <button key={t} onClick={() => toggle(t)}
            className="text-[9px] font-medium px-1.5 py-0.5 rounded transition-all"
            style={isActive
              ? { color, background: `${color}28`, border: `1px solid ${color}80` }
              : { color: "#71717a", background: "transparent", border: "1px solid #3f3f46" }
            }
          >{t}</button>
        )
      })}
    </div>
  )

  const ConnRow = ({ peerId, type }: { peerId: string; type: string }) => {
    const label = nodeLabels.get(peerId)?.label ?? peerId
    const color = connTypeColor(type)
    return (
      <button
        onClick={() => onSelect(peerId)}
        className="w-full flex items-center gap-2 px-3 py-2 text-left hover:bg-muted/30 transition-colors border-b border-border/40 group"
      >
        <span className="flex-1 text-xs text-foreground group-hover:text-primary transition-colors break-words">{label}</span>
        <span
          className="shrink-0 text-[9px] font-medium px-1 py-px rounded whitespace-nowrap"
          style={{ color, background: `${color}18`, border: `1px solid ${color}40` }}
        >{type}</span>
      </button>
    )
  }

  if (!selectedNode) return (
    <div className="flex flex-col items-center justify-center h-full p-4 text-center">
      <Network className="h-5 w-5 text-muted-foreground/30 mb-2" />
      <p className="text-xs text-muted-foreground/50">Select a component to see what it connects to</p>
    </div>
  )

  if (loading && links.length === 0) return (
    <div className="flex flex-col h-full">
      <div className="px-3 pt-3 pb-2 border-b border-border/60 shrink-0">
        <Skeleton className="h-2.5 w-24 mb-1.5" />
        <Skeleton className="h-2.5 w-16" />
      </div>
      <div className="px-2 py-1.5 border-b border-border/60 shrink-0">
        <Skeleton className="h-7 w-full rounded-md" />
      </div>
      <div className="flex-1 overflow-y-auto p-1">
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="flex items-center gap-2 px-3 py-2 border-b border-border/40">
            <Skeleton className="h-3 flex-1" style={{ width: `${40 + (i * 19) % 40}%`, flex: "none" }} />
            <Skeleton className="h-4 w-14 rounded ml-auto" />
          </div>
        ))}
      </div>
    </div>
  )

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-3 pt-3 pb-2 border-b border-border/60 shrink-0">
        <p className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/60">Connections</p>
        <p className="text-[11px] text-muted-foreground mt-0.5">{allConns.length} structural</p>
      </div>

      {/* Search */}
      <div className="px-2 py-1.5 border-b border-border/60 shrink-0">
        <div className="relative flex items-center">
          <Search className="absolute left-2 h-3 w-3 text-muted-foreground/40 pointer-events-none" />
          <Input
            className="h-7 pl-6 pr-6 text-xs bg-background rounded-md border-border focus-visible:ring-1 focus-visible:ring-primary/50"
            placeholder="Search…"
            value={connSearch}
            onChange={e => setConnSearch(e.target.value)}
          />
          {connSearch && (
            <button className="absolute right-2 text-muted-foreground/40 hover:text-muted-foreground transition-colors" onClick={() => setConnSearch("")}>
              <X className="h-3 w-3" />
            </button>
          )}
        </div>
      </div>

      {/* Filters */}
      {(allNodeTypes.length > 1 || allRelTypes.length > 1) && (
        <div className="px-2 py-2 border-b border-border/60 shrink-0 space-y-1.5">
          {allNodeTypes.length > 1 && (
            <div>
              <p className="text-[9px] uppercase tracking-widest text-muted-foreground/40 mb-1">Node type</p>
              <FilterChips items={allNodeTypes} active={activeNodeTypes} toggle={toggleNodeType} colorFn={nodeTypeColor} />
            </div>
          )}
          {allRelTypes.length > 1 && (
            <div>
              <p className="text-[9px] uppercase tracking-widest text-muted-foreground/40 mb-1">Relationship type</p>
              <FilterChips items={allRelTypes} active={activeRelTypes} toggle={toggleRelType} colorFn={connTypeColor} />
            </div>
          )}
        </div>
      )}

      {/* Grouped list */}
      <div className="flex-1 overflow-y-auto">
        {filtered.length === 0 && (
          <p className="text-center text-muted-foreground/50 py-10 text-xs">No connections</p>
        )}
        {grouped.map(([peerType, conns]) => (
          <div key={peerType}>
            <div className="flex items-center gap-2 px-3 pt-3 pb-1">
              <span className="text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/60 whitespace-nowrap">{peerType}</span>
              <div className="flex-1 h-px bg-border/50" />
              <span className="text-[10px] text-muted-foreground/40">{conns.length}</span>
            </div>
            {conns.map((c, i) => <ConnRow key={i} {...c} />)}
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Skeleton list ─────────────────────────────────────────────────────────────

function SkeletonList({ count = 10 }: { count?: number }) {
  return (
    <div className="flex flex-col gap-0">
      {Array.from({ length: count }).map((_, i) => (
        <div key={i} className="flex items-center gap-3 px-3 h-8">
          <Skeleton className="h-3 flex-1" style={{ width: `${50 + (i * 13) % 40}%`, flex: "none" }} />
        </div>
      ))}
    </div>
  )
}

// ── Side List Panel ───────────────────────────────────────────────────────────

function SideListPanel({
  nodesList, appsList, topicsList, brokersList = [], libsList = [], hierarchy, openSet, toggle, expandPath, selectedKey, onSelect, search, onSearchChange, loading, initialTab,
}: {
  nodesList: any[]
  appsList: AppNode[]
  topicsList: any[]
  brokersList?: any[]
  libsList?: any[]
  hierarchy: Record<string, CsmsGroup>
  openSet: Set<string>
  toggle: (k: string) => void
  expandPath: (keys: string[]) => void
  selectedKey: string | null
  onSelect: (n: SelectedNode) => void
  search: string
  onSearchChange: (v: string) => void
  loading: boolean
  initialTab?: "nodes" | "brokers" | "libs" | "apps" | "topics"
}) {
  const [sideTab, setSideTab] = useState<"nodes" | "brokers" | "libs" | "apps" | "topics">(initialTab ?? "nodes")

  // Sync if parent changes the initial tab (e.g. via URL param auto-select)
  const prevInitialTab = useRef(initialTab)
  useEffect(() => {
    if (initialTab && initialTab !== prevInitialTab.current) {
      setSideTab(initialTab)
      prevInitialTab.current = initialTab
    }
  }, [initialTab])
  // Defer expensive filter/tree work so the search input stays responsive
  const deferredSearch = useDeferredValue(search)
  const q = deferredSearch.toLowerCase()

  const switchTab = (tab: "nodes" | "brokers" | "libs" | "apps" | "topics") => {
    setSideTab(tab)
    onSearchChange("")
  }

  const filteredNodes   = useMemo(() => nodesList.filter(n  => !q || (n.name ?? n.id ?? "").toLowerCase().includes(q) || (n.id ?? "").toLowerCase().includes(q)), [nodesList, q])
  const filteredBrokers = useMemo(() => brokersList.filter(b => !q || (b.name ?? b.id ?? "").toLowerCase().includes(q) || (b.id ?? "").toLowerCase().includes(q)), [brokersList, q])
  const filteredLibs    = useMemo(() => libsList.filter(l    => !q || (l.name ?? l.id ?? "").toLowerCase().includes(q) || (l.id ?? "").toLowerCase().includes(q)), [libsList, q])
  const filteredApps   = useMemo(() => appsList.filter(a  => !q || (a.csu ?? a.name ?? a.id ?? "").toLowerCase().includes(q) || (a.id ?? "").toLowerCase().includes(q)), [appsList, q])
  const filteredTopics = useMemo(() => topicsList.filter(t => !q || (t.name ?? t.id ?? "").toLowerCase().includes(q) || (t.id ?? "").toLowerCase().includes(q)), [topicsList, q])

  const makeRow = (item: any, kind: "node" | "app" | "topic") => {
    const label = kind === "app"
      ? ((item as AppNode).csu ?? item.name ?? item.id ?? "?")
      : (item.name ?? item.id ?? "?")
    const key = `${kind}:${item.id}`
    const isSelected = selectedKey === key
    return (
      <button
        key={key}
        className={cn(
          "w-full flex items-center gap-3 px-3 text-left text-xs transition-colors border-l-2 h-8",
          isSelected
            ? "bg-muted border-l-primary text-foreground font-medium"
            : "border-l-transparent text-muted-foreground hover:bg-muted/40 hover:text-foreground",
        )}
        onClick={() => onSelect({ kind, key, label, path: [label], payload: item })}
      >
        <span className="flex-1 truncate">{label}</span>
      </button>
    )
  }

  const csmsKeys = sortKeys(Object.keys(hierarchy))

  return (
    <>
      {/* Tab bar */}
      <div className="shrink-0">
        <div className="flex h-9 border-b border-border text-xs">
          {([
            { id: "nodes",   label: "Nodes",   count: nodesList.length },
            { id: "apps",    label: "Apps",    count: appsList.length },
            { id: "topics",  label: "Topics",  count: topicsList.length },
            { id: "libs",    label: "Libs",    count: libsList.length },
            { id: "brokers", label: "Brokers", count: brokersList.length },
          ] as const).map(tab => (
            <button
              key={tab.id}
              className={cn(
                "flex-1 px-1 transition-colors border-b-2 text-[11px] min-w-0",
                sideTab === tab.id
                  ? "border-primary text-foreground font-semibold"
                  : "border-transparent text-muted-foreground hover:text-foreground",
              )}
              onClick={() => switchTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </div>
        {/* Search */}
        <div className="px-3 border-b border-border bg-muted/20 h-10">
          <div className="relative flex items-center h-full">
            <Search className="absolute left-2 h-3 w-3 text-muted-foreground/40 pointer-events-none" />
            <Input
              className="h-7 pl-6 pr-6 text-xs bg-background rounded-md border-border focus-visible:ring-1 focus-visible:ring-primary/50"
              placeholder="Filter…"
              value={search}
              onChange={(e) => onSearchChange(e.target.value)}
            />
            {search && (
              <button
                className="absolute right-2 text-muted-foreground/40 hover:text-muted-foreground transition-colors"
                onClick={() => onSearchChange("")}
              >
                <X className="h-3 w-3" />
              </button>
            )}
          </div>
        </div>
      </div>

      {/* List */}
      <div className="flex-1 overflow-hidden">
        {sideTab === "nodes"   && (loading && !filteredNodes.length   ? <SkeletonList /> : filteredNodes.length   > 0 ? <SimpleVirtualList items={filteredNodes}   renderItem={n => makeRow(n, "node")}  itemHeight={32} /> : <p className="text-center text-muted-foreground/50 py-8 text-xs">No nodes.</p>)}
        {sideTab === "brokers" && (loading && !filteredBrokers.length ? <SkeletonList /> : filteredBrokers.length > 0 ? <SimpleVirtualList items={filteredBrokers} renderItem={b => makeRow(b, "node")}  itemHeight={32} /> : <p className="text-center text-muted-foreground/50 py-8 text-xs">No brokers.</p>)}
        {sideTab === "libs"    && (loading && !filteredLibs.length    ? <SkeletonList /> : filteredLibs.length    > 0 ? <SimpleVirtualList items={filteredLibs}    renderItem={l => makeRow(l, "node")}  itemHeight={32} /> : <p className="text-center text-muted-foreground/50 py-8 text-xs">No libraries.</p>)}
        {sideTab === "apps"    && (loading && !filteredApps.length    ? <SkeletonList /> : filteredApps.length    > 0 ? <SimpleVirtualList items={filteredApps}    renderItem={a => makeRow(a, "app")}   itemHeight={32} /> : <p className="text-center text-muted-foreground/50 py-8 text-xs">No apps.</p>)}
        {sideTab === "topics"  && (loading && !filteredTopics.length  ? <SkeletonList /> : filteredTopics.length  > 0 ? <SimpleVirtualList items={filteredTopics}  renderItem={t => makeRow(t, "topic")} itemHeight={32} /> : <p className="text-center text-muted-foreground/50 py-8 text-xs">No topics.</p>)}
      </div>

    </>
  )
}
