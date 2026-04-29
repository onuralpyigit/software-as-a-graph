"use client"

import { useState } from "react"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import {
  BookMarked,
  Search,
  Tag,
} from "lucide-react"

type TermCategory =
  | "Graph Model"
  | "RMAV Metrics"
  | "Simulation"
  | "Validation"
  | "Anti-Patterns"
  | "Pipeline"
  | "Infrastructure"

interface Term {
  term: string
  category: TermCategory
  definition: string
  formula?: string
}

const TERMS: Term[] = [
  // Graph Model
  {
    term: "Node (Component)",
    category: "Graph Model",
    definition:
      "A vertex in the system graph representing a deployable unit: Application, Library, Broker, or Infrastructure Node. Each node carries metadata such as type, weight (operational priority), and optional code-quality attributes.",
  },
  {
    term: "Edge (DEPENDS_ON)",
    category: "Graph Model",
    definition:
      "A directed edge from a dependent component to its dependency. Carries weight ∈ [0,1] (maximum QoS severity) and path_count (coupling intensity). Six derivation rules produce subtypes: app_to_app, app_to_lib, app_to_broker, node_to_node, node_to_broker, and broker_to_broker.",
  },
  {
    term: "Structural Edge",
    category: "Graph Model",
    definition:
      "A raw topology edge (PUBLISHES, SUBSCRIBES, USES, HOSTS, ROUTES) from which DEPENDS_ON edges are derived. Simulation operates on structural edges directly.",
  },
  {
    term: "Graph Layer",
    category: "Graph Model",
    definition:
      "A logical view of the system graph. Four layers are supported: app (Applications + Libraries), infra (Infrastructure Nodes), mw (Middleware / Brokers), and system (all node types combined).",
  },
  {
    term: "Bridge",
    category: "Graph Model",
    definition:
      "An edge whose removal disconnects the graph. Bridge Ratio (BR) is the fraction of a node's incident edges that are bridges and contributes to its Availability score.",
  },
  {
    term: "Articulation Point",
    category: "Graph Model",
    definition:
      "A node whose removal increases the number of connected components. It is the primary structural indicator for the SPOF anti-pattern and contributes directly to A(v).",
  },
  {
    term: "SCC (Strongly Connected Component)",
    category: "Graph Model",
    definition:
      "A maximal subgraph in which every node is reachable from every other node. SCCs of size ≥ 2 trigger the CYCLE anti-pattern.",
  },
  // RMAV Metrics
  {
    term: "R(v) — Reliability",
    category: "RMAV Metrics",
    definition:
      "Quantifies the fault-propagation risk of component v. Higher values indicate that failures originating at v are likely to cascade broadly and deeply.",
    formula: "R(v) = 0.45·RPR + 0.30·DG_in + 0.25·CDPot_enh",
  },
  {
    term: "M(v) — Maintainability",
    category: "RMAV Metrics",
    definition:
      "Quantifies how costly it is to change component v, considering its structural bottleneck role, coupling, and optional code-quality attributes.",
    formula: "M(v) = 0.35·BT + 0.30·w_out + 0.15·CQP + 0.12·CouplingRisk + 0.08·(1−CC)",
  },
  {
    term: "A(v) — Availability",
    category: "RMAV Metrics",
    definition:
      "Quantifies how severely the loss of component v disrupts connectivity and reachability for the rest of the system.",
    formula: "A(v) = 0.35·AP_c + 0.25·QSPOF + 0.25·BR + 0.10·CDI + 0.05·w(v)",
  },
  {
    term: "V(v) — Vulnerability",
    category: "RMAV Metrics",
    definition:
      "Quantifies how attractive and reachable component v is from an adversarial perspective, capturing both strategic position and inbound exposure.",
    formula: "V(v) = 0.40·REV + 0.35·RCL + 0.25·QADS",
  },
  {
    term: "Q(v) — Overall Quality",
    category: "RMAV Metrics",
    definition:
      "AHP-weighted composite quality score across all four RMAV dimensions. Higher values indicate greater criticality or risk.",
    formula: "Q(v) = 0.24·R(v) + 0.17·M(v) + 0.43·A(v) + 0.16·V(v)",
  },
  {
    term: "RPR (Reverse PageRank)",
    category: "RMAV Metrics",
    definition:
      "PageRank computed on the transposed graph G^T. Measures how many components ultimately depend on v (fault-propagation reach).",
  },
  {
    term: "BT (Betweenness Centrality)",
    category: "RMAV Metrics",
    definition:
      "Fraction of shortest paths between all node pairs that pass through v. High betweenness signals a structural bottleneck with high change-impact.",
  },
  {
    term: "REV (Reverse Eigenvector Centrality)",
    category: "RMAV Metrics",
    definition:
      "Eigenvector centrality computed on G^T. Captures strategic importance for an attacker — being depended on by other high-value nodes.",
  },
  {
    term: "CDPot (Cascade Depth Potential)",
    category: "RMAV Metrics",
    definition:
      "Maximum cascade depth reachable from v, enhanced by the Multi-Path Cascade Intensity (MPCI) factor to account for parallel failure paths.",
  },
  {
    term: "CQP (Code Quality Penalty)",
    category: "RMAV Metrics",
    definition:
      "Optional penalty derived from cyclomatic complexity, Martin instability, and Lack of Cohesion of Methods (LCOM). Applied only when code-quality attributes are present on the node.",
    formula: "CQP = 0.40·complexity_norm + 0.35·instability_code + 0.25·lcom_norm",
  },
  {
    term: "QSPOF",
    category: "RMAV Metrics",
    definition:
      "QoS-scaled SPOF severity: the articulation-point score multiplied by the node's operational priority weight w(v).",
    formula: "QSPOF = AP_c × w(v)",
  },
  {
    term: "AHP (Analytic Hierarchy Process)",
    category: "RMAV Metrics",
    definition:
      "Structured multi-criteria decision method used to derive dimension weights. A shrinkage factor λ=0.7 blends AHP-derived weights toward a uniform prior, preventing extreme weight concentration.",
  },
  {
    term: "Classification (Box-Plot)",
    category: "RMAV Metrics",
    definition:
      "Five-level criticality label assigned to each score. Thresholds are computed from the score distribution: CRITICAL (> Q3 + 0.75·IQR), HIGH (> Q3), MEDIUM (> Median), LOW (> Q1), MINIMAL (≤ Q1). For small samples (< 12 nodes) fixed percentile fallbacks are used.",
  },
  // Simulation
  {
    term: "Cascade Failure Simulation",
    category: "Simulation",
    definition:
      "Counterfactual experiment that injects a fault at each node, propagates failures through structural edges, and measures system-wide impact. Produces per-component ground-truth impact scores used for validation.",
  },
  {
    term: "I(v) — Overall Impact",
    category: "Simulation",
    definition:
      "Composite simulation ground truth for component v, weighting reachability loss, fragmentation, throughput loss, and flow disruption.",
    formula: "I(v) = 0.35·reachability_loss + 0.25·fragmentation + 0.25·throughput_loss + 0.15·flow_disruption",
  },
  {
    term: "IR(v) — Reliability Impact",
    category: "Simulation",
    definition:
      "Simulation ground truth for Reliability: how far and how deeply failures cascade from v.",
    formula: "IR(v) = 0.45·CascadeReach + 0.35·WeightedCascadeImpact + 0.20·NormalizedCascadeDepth",
  },
  {
    term: "IM(v) — Maintainability Impact",
    category: "Simulation",
    definition:
      "Simulation ground truth for Maintainability: scope of change propagation from v through the transposed dependency graph G^T.",
    formula: "IM(v) = 0.45·ChangeReach + 0.35·WeightedChangeImpact + 0.20·NormalizedChangeDepth",
  },
  {
    term: "IA(v) — Availability Impact",
    category: "Simulation",
    definition:
      "Simulation ground truth for Availability: QoS-weighted connectivity disruption caused by removing v.",
    formula: "IA(v) = 0.50·WeightedReachabilityLoss + 0.35·WeightedFragmentation + 0.15·PathBreakingThroughputLoss",
  },
  {
    term: "IV(v) — Vulnerability Impact",
    category: "Simulation",
    definition:
      "Simulation ground truth for Vulnerability: adversarial compromise propagation from v over trusted dependency paths.",
    formula: "IV(v) = 0.40·AttackReach + 0.35·WeightedAttackImpact + 0.25·HighValueContamination",
  },
  {
    term: "SPOF (Single Point of Failure)",
    category: "Simulation",
    definition:
      "A component whose failure alone can cause significant system-wide disruption. Identified structurally as an articulation point and empirically validated via the I(v) > 0.5 threshold.",
  },
  // Validation
  {
    term: "Spearman ρ",
    category: "Validation",
    definition:
      "Non-parametric rank correlation between predicted scores and simulation ground truth. The primary validation metric; target ρ > 0.87 across all dimensions.",
  },
  {
    term: "NDCG@K",
    category: "Validation",
    definition:
      "Normalized Discounted Cumulative Gain at rank K. Measures how well the top-K predicted critical components match the simulation-derived top-K, with higher ranks weighted more heavily.",
  },
  {
    term: "CCR@K (Cascade Capture Rate)",
    category: "Validation",
    definition:
      "Fraction of the top-K cascade-critical nodes (by IR) correctly identified in the top-K predicted Reliability scores.",
  },
  {
    term: "SPOF_F1",
    category: "Validation",
    definition:
      "F1 score for classifying nodes as SPOFs. Compares structural SPOF prediction (articulation points with high A(v)) against simulation-derived SPOF labels.",
  },
  {
    term: "AHCR@K (Attack Hit Capture Rate)",
    category: "Validation",
    definition:
      "Fraction of the top-K adversarially critical nodes (by IV) correctly identified in the top-K predicted Vulnerability scores.",
  },
  {
    term: "FTR (False Trust Rate)",
    category: "Validation",
    definition:
      "Fraction of nodes predicted as low-vulnerability that the simulation identifies as high adversarial-impact. A low FTR indicates the model correctly flags risky nodes.",
  },
  {
    term: "Weighted-κ CTA",
    category: "Validation",
    definition:
      "Quadratic-weighted Cohen's kappa for Change-impact Tier Agreement. Compares the ranked change-propagation tiers predicted by M(v) against simulation-derived IM(v) tiers.",
  },
  // Anti-Patterns
  {
    term: "FAILURE_HUB",
    category: "Anti-Patterns",
    definition:
      "A component with CRITICAL Reliability score. Indicates a hub whose failure triggers broad cascade propagation across the system.",
  },
  {
    term: "GOD_COMPONENT",
    category: "Anti-Patterns",
    definition:
      "A component with CRITICAL Maintainability score and betweenness > 0.3. A structural god object — too central to change safely.",
  },
  {
    term: "TARGET",
    category: "Anti-Patterns",
    definition:
      "A component with CRITICAL Vulnerability score. High adversarial value and exposure make it a prime attack target.",
  },
  {
    term: "EXPOSURE",
    category: "Anti-Patterns",
    definition:
      "A component with HIGH Vulnerability and closeness > 0.6. Broadly reachable by adversaries and not yet at CRITICAL severity.",
  },
  {
    term: "BRIDGE_EDGE",
    category: "Anti-Patterns",
    definition:
      "An edge that is a bridge (its removal disconnects the graph). Represents a single-link failure path with HIGH severity.",
  },
  {
    term: "CYCLE",
    category: "Anti-Patterns",
    definition:
      "A strongly connected component of size ≥ 2. Indicates circular dependencies that complicate failure isolation and change management.",
  },
  {
    term: "HUB_AND_SPOKE",
    category: "Anti-Patterns",
    definition:
      "A node with low clustering coefficient (< 0.1) and high degree (> 3). Centralized topology with little redundancy around the hub.",
  },
  {
    term: "CHAIN",
    category: "Anti-Patterns",
    definition:
      "A weakly connected sequential path of length ≥ 4. Long chains amplify cascade depth and create brittle dependency pipelines.",
  },
  {
    term: "SYSTEMIC_RISK",
    category: "Anti-Patterns",
    definition:
      "A system-level anti-pattern triggered when more than 20% of nodes are classified as CRITICAL. Indicates broad architectural fragility.",
  },
  // Pipeline
  {
    term: "Pipeline",
    category: "Pipeline",
    definition:
      "The six-stage analysis workflow: Import → Analyze → Predict → Simulate → Validate → Visualize. Each stage builds on the outputs of the previous.",
  },
  {
    term: "Import",
    category: "Pipeline",
    definition:
      "Stage 1: Converts a topology JSON file into a weighted directed graph stored in Neo4j. Derives DEPENDS_ON edges via six structural rules.",
  },
  {
    term: "Analyze",
    category: "Pipeline",
    definition:
      "Stage 2: Computes deterministic RMAV scores and Q(v) via closed-form AHP-weighted formulas. Also detects anti-patterns. Always produces the same output for the same graph.",
  },
  {
    term: "Predict",
    category: "Pipeline",
    definition:
      "Stage 3 (optional): Inductive GNN-based forecasting. A HeteroGAT learns non-linear multi-hop interactions and blends Q_GNN with Q_RMAV into Q_ensemble.",
  },
  {
    term: "Simulate",
    category: "Pipeline",
    definition:
      "Stage 4: Injects faults exhaustively or via Monte Carlo, runs four parallel simulators (cascade, change, connectivity, compromise), and produces per-RMAV ground-truth impact labels.",
  },
  {
    term: "Validate",
    category: "Pipeline",
    definition:
      "Stage 5: Compares predicted scores against simulation ground truth using Spearman, F1, NDCG@K, and dimension-specific metrics.",
  },
  {
    term: "Visualize",
    category: "Pipeline",
    definition:
      "Stage 6: Generates interactive web dashboards or static HTML reports from analysis and validation results.",
  },
  {
    term: "Q_ensemble",
    category: "Pipeline",
    definition:
      "Blended quality score combining GNN predictions and rule-based RMAV scores.",
    formula: "Q_ensemble(v) = α · Q_GNN + (1−α) · Q_RMAV",
  },
  // Infrastructure
  {
    term: "Neo4j",
    category: "Infrastructure",
    definition:
      "Graph database (v5.x) used to store the system topology. Accessed via the Bolt protocol. All structural and DEPENDS_ON edges are persisted here.",
  },
  {
    term: "MemoryRepo",
    category: "Infrastructure",
    definition:
      "In-memory repository implementation used during tests. Implements the same interface as Neo4jRepo so all business logic is testable without a live database.",
  },
  {
    term: "HeteroGAT",
    category: "Infrastructure",
    definition:
      "Heterogeneous Graph Attention Network. Operates on mixed-type node graphs (Application, Broker, Node) and learns per-type attention weights for message passing.",
  },
]

const CATEGORY_COLORS: Record<TermCategory, string> = {
  "Graph Model": "bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300",
  "RMAV Metrics": "bg-purple-100 text-purple-800 dark:bg-purple-900/40 dark:text-purple-300",
  "Simulation": "bg-orange-100 text-orange-800 dark:bg-orange-900/40 dark:text-orange-300",
  "Validation": "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300",
  "Anti-Patterns": "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
  "Pipeline": "bg-cyan-100 text-cyan-800 dark:bg-cyan-900/40 dark:text-cyan-300",
  "Infrastructure": "bg-slate-100 text-slate-800 dark:bg-slate-900/40 dark:text-slate-300",
}

const ALL_CATEGORIES: TermCategory[] = [
  "Graph Model",
  "RMAV Metrics",
  "Simulation",
  "Validation",
  "Anti-Patterns",
  "Pipeline",
  "Infrastructure",
]

export default function DictionaryPage() {
  const [search, setSearch] = useState("")
  const [activeCategory, setActiveCategory] = useState<TermCategory | "All">("All")

  const filtered = TERMS.filter((t) => {
    const matchesSearch =
      search.trim() === "" ||
      t.term.toLowerCase().includes(search.toLowerCase()) ||
      t.definition.toLowerCase().includes(search.toLowerCase())
    const matchesCategory =
      activeCategory === "All" || t.category === activeCategory
    return matchesSearch && matchesCategory
  })

  const grouped = ALL_CATEGORIES.reduce<Record<TermCategory, Term[]>>(
    (acc, cat) => {
      acc[cat] = filtered.filter((t) => t.category === cat)
      return acc
    },
    {} as Record<TermCategory, Term[]>
  )

  return (
    <AppLayout
      title="Dictionary"
      description="Definitions of terms used across the platform"
    >
      <div className="space-y-6">
        {/* Header banner */}
        <Card className="relative overflow-hidden border-0 shadow-xl">
          <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500">
            <div className="w-full h-full rounded-lg bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600" />
          </div>
          <CardContent className="p-8 relative text-white">
            <div className="flex items-center gap-3 mb-3">
              <BookMarked className="h-6 w-6" />
              <Badge className="bg-white/20 text-white border-white/30 hover:bg-white/30 px-3 py-1.5">
                Reference
              </Badge>
            </div>
            <h3 className="text-3xl font-bold mb-2">Platform Dictionary</h3>
            <p className="text-white/90 text-lg max-w-3xl">
              Glossary of all metrics, graph concepts, pipeline stages, anti-patterns, and
              infrastructure terms used throughout Genieus.
            </p>
            <p className="text-white/70 text-sm mt-2">{TERMS.length} terms across {ALL_CATEGORIES.length} categories</p>
          </CardContent>
        </Card>

        {/* Search + category filters */}
        <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-center">
          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search terms…"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-9"
            />
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setActiveCategory("All")}
              className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium border transition-colors ${
                activeCategory === "All"
                  ? "bg-foreground text-background border-foreground"
                  : "border-border text-muted-foreground hover:text-foreground hover:border-foreground/50"
              }`}
            >
              <Tag className="h-3 w-3" />
              All
            </button>
            {ALL_CATEGORIES.map((cat) => (
              <button
                key={cat}
                onClick={() => setActiveCategory(cat === activeCategory ? "All" : cat)}
                className={`rounded-full px-3 py-1 text-xs font-medium border transition-colors ${
                  activeCategory === cat
                    ? "bg-foreground text-background border-foreground"
                    : "border-border text-muted-foreground hover:text-foreground hover:border-foreground/50"
                }`}
              >
                {cat}
              </button>
            ))}
          </div>
        </div>

        {/* Term groups */}
        {filtered.length === 0 ? (
          <Card>
            <CardContent className="py-16 text-center text-muted-foreground">
              No terms match your search.
            </CardContent>
          </Card>
        ) : (
          ALL_CATEGORIES.map((cat) => {
            const terms = grouped[cat]
            if (terms.length === 0) return null
            return (
              <Card key={cat}>
                <CardHeader className="pb-3">
                  <CardTitle className="flex items-center gap-2 text-base">
                    <span
                      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${CATEGORY_COLORS[cat]}`}
                    >
                      {cat}
                    </span>
                    <span className="text-muted-foreground font-normal text-sm">
                      {terms.length} term{terms.length !== 1 ? "s" : ""}
                    </span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="divide-y divide-border">
                    {terms.map((t) => (
                      <div key={t.term} className="py-4 first:pt-0 last:pb-0">
                        <div className="flex flex-wrap items-center gap-2 mb-1">
                          <span className="font-semibold text-sm">{t.term}</span>
                        </div>
                        <p className="text-sm text-muted-foreground leading-relaxed">
                          {t.definition}
                        </p>
                        {t.formula && (
                          <p className="mt-2 font-mono text-xs bg-muted/60 rounded px-3 py-1.5 inline-block text-foreground">
                            {t.formula}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )
          })
        )}
      </div>
    </AppLayout>
  )
}
