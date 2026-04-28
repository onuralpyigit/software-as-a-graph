"use client"

import React, { useState } from "react"
import { useRouter } from "next/navigation"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import {
  Brain,
  Play,
  CheckCircle2,
  AlertTriangle,
  Loader2,
  ChevronDown,
  ChevronRight,
  BarChart3,
  Layers,
  Settings,
  Target,
  Zap,
  Shield,
  Wrench,
  Server,
  TrendingUp,
  Activity,
  Database,
} from "lucide-react"
import { useConnection } from "@/lib/stores/connection-store"
import { apiClient } from "@/lib/api/client"
import axios from "axios"
import { TermTooltip } from "@/components/ui/term-tooltip"

// ── Types ─────────────────────────────────────────────────────────────────────

interface GNNScore {
  component: string
  node_name: string
  composite_score: number
  reliability_score: number
  maintainability_score: number
  availability_score: number
  vulnerability_score: number
  criticality_level: string
  source: string
}

interface GNNEdgeScore {
  source: string
  source_name: string
  target: string
  target_name: string
  edge_type: string
  composite_score: number
  criticality_level: string
}

interface GNNMetrics {
  spearman_rho: number | null
  f1_score: number | null
  rmse: number | null
  mae: number | null
  ndcg_10: number | null
}

interface TrainSummary {
  total_components: number
  critical: number
  high: number
  medium: number
  low: number
  minimal: number
  critical_edges: number
}

interface TrainResult {
  success: boolean
  layer: string
  checkpoint_dir: string
  summary: TrainSummary
  gnn_metrics: GNNMetrics | null
  ensemble_metrics: GNNMetrics | null
  ensemble_alpha: number[] | null
  top_critical: GNNScore[]
  top_critical_edges: GNNEdgeScore[]
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function criticality_color(level: string): string {
  switch (level) {
    case "CRITICAL": return "text-red-600 dark:text-red-400"
    case "HIGH": return "text-orange-600 dark:text-orange-400"
    case "MEDIUM": return "text-yellow-600 dark:text-yellow-400"
    case "LOW": return "text-blue-600 dark:text-blue-400"
    default: return "text-muted-foreground"
  }
}

function criticality_badge(level: string) {
  const variants: Record<string, string> = {
    CRITICAL: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300",
    HIGH: "bg-orange-100 text-orange-800 dark:bg-orange-950 dark:text-orange-300",
    MEDIUM: "bg-yellow-100 text-yellow-800 dark:bg-yellow-950 dark:text-yellow-300",
    LOW: "bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300",
    MINIMAL: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
  }
  return (
    <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${variants[level] ?? variants["MINIMAL"]}`}>
      {level}
    </span>
  )
}

function ScoreBar({ value, dim }: { value: number; dim: string }) {
  const colors: Record<string, string> = {
    R: "bg-blue-500",
    M: "bg-purple-500",
    A: "bg-green-500",
    V: "bg-red-500",
  }
  const termMap: Record<string, string> = {
    R: "R(v)", M: "M(v)", A: "A(v)", V: "V(v)",
  }
  return (
    <div className="flex items-center gap-2">
      <span className="w-4 text-xs text-muted-foreground">
        <TermTooltip term={termMap[dim]}>{dim}</TermTooltip>
      </span>
      <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
        <div className={`h-full rounded-full ${colors[dim] ?? "bg-gray-400"}`} style={{ width: `${Math.round(value * 100)}%` }} />
      </div>
      <span className="w-10 text-right text-xs tabular-nums">{value.toFixed(3)}</span>
    </div>
  )
}

function PaginationBar({ page, total, pageSize, onPage }: { page: number; total: number; pageSize: number; onPage: (p: number) => void }) {
  const totalPages = Math.ceil(total / pageSize)
  if (totalPages <= 1) return null
  const start = (page - 1) * pageSize + 1
  const end = Math.min(page * pageSize, total)
  return (
    <div className="flex items-center justify-between pt-3 mt-1 border-t text-xs text-muted-foreground">
      <span>{start}–{end} of {total}</span>
      <div className="flex items-center gap-1">
        <Button variant="ghost" size="sm" className="h-7 px-2" disabled={page === 1} onClick={() => onPage(page - 1)}>
          ← Prev
        </Button>
        <span className="px-1">{page} / {totalPages}</span>
        <Button variant="ghost" size="sm" className="h-7 px-2" disabled={page === totalPages} onClick={() => onPage(page + 1)}>
          Next →
        </Button>
      </div>
    </div>
  )
}

// ── Page ──────────────────────────────────────────────────────────────────────

export default function TrainPage() {
  const router = useRouter()
  const { config, status } = useConnection()
  const isConnected = status === "connected"

  // Config form state
  const [layer, setLayer] = useState("app")
  const [checkpointName, setCheckpointName] = useState("")
  const [epochs, setEpochs] = useState(300)
  const [hidden, setHidden] = useState(64)
  const [heads, setHeads] = useState(4)
  const [gnnLayers, setGnnLayers] = useState(3)
  const [dropout, setDropout] = useState(0.2)
  const [lr, setLr] = useState(0.0003)
  const [patience, setPatience] = useState(30)
  const [trainRatio, setTrainRatio] = useState(0.6)
  const [valRatio, setValRatio] = useState(0.2)
  const [advancedOpen, setAdvancedOpen] = useState(false)

  // Execution state
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<TrainResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Pagination
  const PAGE_SIZE = 10
  const [compPage, setCompPage] = useState(1)
  const [edgePage, setEdgePage] = useState(1)

  const runTraining = async () => {
    if (!isConnected || !config) return

    setIsRunning(true)
    setResult(null)
    setError(null)
    setProgress(10)

    // Simulate progress during the long-running call
    const tick = setInterval(() => {
      setProgress(p => Math.min(p + 2, 90))
    }, 3000)

    try {
      const response = await axios.post(`${apiClient.getBaseURL()}/api/v1/prediction/train`, {
        credentials: config,
        layer,
        checkpoint_name: checkpointName.trim(),
        epochs,
        hidden,
        heads,
        layers: gnnLayers,
        dropout,
        lr,
        patience,
        train_ratio: trainRatio,
        val_ratio: valRatio,
      }, { timeout: 1_800_000 }) // 30 min max

      setProgress(100)
      setResult(response.data)
      setCompPage(1)
      setEdgePage(1)
    } catch (e: any) {
      const msg = e?.response?.data?.detail ?? e?.message ?? "Training failed"
      setError(msg)
    } finally {
      clearInterval(tick)
      setIsRunning(false)
    }
  }

  return (
    <AppLayout
      title="Train GNN Model"
      description="Train a Heterogeneous Graph Attention Network to predict component criticality"
    >
      {!isConnected ? (
        <NoConnectionInfo />
      ) : (
        <div className="space-y-6">

          {/* Section header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-gradient-to-br from-blue-600 via-purple-600 to-pink-600 p-3 shadow-lg">
                <Brain className="h-6 w-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold">Training Configuration</h2>
                <p className="text-sm text-muted-foreground">
                  Configure <TermTooltip term="HeteroGAT">HeteroGAT</TermTooltip> hyperparameters. Checkpoints auto-saved to{" "}
                  <code className="text-xs bg-muted px-1 py-0.5 rounded">output/gnn_checkpoints/YYYY-MM-DD_HH-MM-SS</code>
                </p>
              </div>
            </div>
            <Button
              onClick={runTraining}
              disabled={isRunning}
              size="lg"
              className="min-w-[180px] h-12 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 hover:from-blue-700 hover:via-purple-700 hover:to-pink-700 shadow-lg hover:shadow-xl transition-all text-base font-semibold"
            >
              {isRunning ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  Training…
                </>
              ) : (
                <>
                  <Play className="mr-2 h-5 w-5" />
                  Start Training
                </>
              )}
            </Button>
          </div>

          {/* Configuration card */}
          <Card>
            <CardContent className="space-y-6 pt-6">

              {/* Primary settings */}
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                <div className="space-y-1.5">
                  <Label htmlFor="layer">Layer</Label>
                  <Select value={layer} onValueChange={setLayer}>
                    <SelectTrigger id="layer">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="app">Application</SelectItem>
                      <SelectItem value="infra">Infrastructure</SelectItem>
                      <SelectItem value="mw">Middleware</SelectItem>
                      <SelectItem value="system">System</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-1.5 sm:col-span-2">
                  <Label htmlFor="ckpt-name">Checkpoint Name <span className="text-muted-foreground font-normal">(optional)</span></Label>
                  <Input
                    id="ckpt-name"
                    placeholder="e.g. my-model  (leave blank for auto datetime)"
                    value={checkpointName}
                    onChange={e => setCheckpointName(e.target.value)}
                  />
                </div>

                <div className="space-y-1.5">
                  <Label htmlFor="epochs">Max Epochs</Label>
                  <Input id="epochs" type="number" min={10} max={2000} value={epochs}
                    onChange={e => setEpochs(Number(e.target.value))} />
                </div>
              </div>

              {/* Advanced */}
              <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
                <CollapsibleTrigger asChild>
                  <button className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors">
                    {advancedOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                    Advanced hyperparameters
                  </button>
                </CollapsibleTrigger>
                <CollapsibleContent className="mt-4">
                  <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                    <div className="space-y-1.5">
                      <Label htmlFor="hidden"><TermTooltip term="Hidden Dim">Hidden Dim</TermTooltip></Label>
                      <Input id="hidden" type="number" min={8} max={512} value={hidden}
                        onChange={e => setHidden(Number(e.target.value))} />
                    </div>
                    <div className="space-y-1.5">
                      <Label htmlFor="heads"><TermTooltip term="Attn Heads">Attn Heads</TermTooltip></Label>
                      <Input id="heads" type="number" min={1} max={16} value={heads}
                        onChange={e => setHeads(Number(e.target.value))} />
                    </div>
                    <div className="space-y-1.5">
                      <Label><TermTooltip term="GNN Layers">GNN Layers</TermTooltip></Label>
                      <Input type="number" min={1} max={8} value={gnnLayers}
                        onChange={e => setGnnLayers(Number(e.target.value))} />
                    </div>
                    <div className="space-y-1.5">
                      <Label><TermTooltip term="Dropout">Dropout</TermTooltip></Label>
                      <Input type="number" step={0.05} min={0} max={0.8} value={dropout}
                        onChange={e => setDropout(Number(e.target.value))} />
                    </div>
                    <div className="space-y-1.5">
                      <Label><TermTooltip term="Learning Rate">Learning Rate</TermTooltip></Label>
                      <Input type="number" step={0.0001} min={0.00001} max={0.01} value={lr}
                        onChange={e => setLr(Number(e.target.value))} />
                    </div>
                    <div className="space-y-1.5">
                      <Label><TermTooltip term="Early-stop Patience">Early-stop Patience</TermTooltip></Label>
                      <Input type="number" min={5} max={200} value={patience}
                        onChange={e => setPatience(Number(e.target.value))} />
                    </div>
                    <div className="space-y-1.5">
                      <Label><TermTooltip term="Train Ratio">Train Ratio</TermTooltip></Label>
                      <Input type="number" step={0.05} min={0.3} max={0.8} value={trainRatio}
                        onChange={e => setTrainRatio(Number(e.target.value))} />
                    </div>
                    <div className="space-y-1.5">
                      <Label>Val Ratio</Label>
                      <Input type="number" step={0.05} min={0.1} max={0.4} value={valRatio}
                        onChange={e => setValRatio(Number(e.target.value))} />
                    </div>
                  </div>
                </CollapsibleContent>
              </Collapsible>

              {isRunning && (
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Running analysis → simulation → GNN training…</span>
                    <span>{progress}%</span>
                  </div>
                  <Progress value={progress} className="h-2" />
                </div>
              )}

              {error && (
                <div className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700 dark:border-red-800 dark:bg-red-950/30 dark:text-red-400">
                  <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
                  <span>{error}</span>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Results */}
          {result && (
            <>
              {/* Summary row */}
              <div className="grid grid-cols-3 gap-3 sm:grid-cols-6">
                {[
                  { label: "Critical", value: result.summary.critical, icon: AlertTriangle, color: "text-red-500" },
                  { label: "High", value: result.summary.high, icon: Zap, color: "text-orange-500" },
                  { label: "Medium", value: result.summary.medium, icon: Activity, color: "text-yellow-500" },
                  { label: "Low", value: result.summary.low, icon: Shield, color: "text-blue-500" },
                  { label: "Minimal", value: result.summary.minimal, icon: Server, color: "text-muted-foreground" },
                  { label: "Total", value: result.summary.total_components, icon: Database, color: "text-foreground" },
                ].map(({ label, value, icon: Icon, color }) => (
                  <Card key={label}>
                    <CardContent className="flex items-center gap-3 p-4">
                      <Icon className={`h-5 w-5 ${color}`} />
                      <div>
                        <p className="text-xl font-bold">{value}</p>
                        <p className="text-xs text-muted-foreground">{label}</p>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>

              {/* Metrics */}
              {(result.gnn_metrics || result.ensemble_metrics) && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-base">
                      <BarChart3 className="h-4 w-4 text-purple-500" />
                      Evaluation Metrics
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className={`grid grid-cols-1 gap-4 ${result.gnn_metrics && result.ensemble_metrics ? "sm:grid-cols-2" : ""}`}>
                      {[
                        { label: "GNN", metrics: result.gnn_metrics },
                        { label: "Ensemble (GNN + RMAV)", metrics: result.ensemble_metrics },
                      ].map(({ label, metrics }) => metrics && (
                        <div key={label} className="rounded-lg border p-4 space-y-2">
                          <p className="font-medium text-sm">{label}</p>
                          <div className="space-y-1 text-sm">
                            {[
                              ["Spearman ρ", metrics.spearman_rho],
                              ["F1", metrics.f1_score],
                              ["NDCG@10", metrics.ndcg_10],
                              ["RMSE", metrics.rmse],
                              ["MAE", metrics.mae],
                            ].map(([k, v]) => v != null && (
                              <div key={String(k)} className="flex items-center justify-between gap-4">
                                <span className="text-muted-foreground"><TermTooltip term={String(k)}>{k}</TermTooltip></span>
                                <span className="font-mono font-medium tabular-nums">{(v as number).toFixed(4)}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                    {result.ensemble_alpha && (
                      <p className="mt-3 text-xs text-muted-foreground">
                        Ensemble α (per RMAV dim):&nbsp;
                        {result.ensemble_alpha.map(a => a.toFixed(3)).join(", ")}
                      </p>
                    )}
                  </CardContent>
                </Card>
              )}

              {/* Checkpoint saved */}
              <div className="flex items-center gap-2 rounded-lg border border-green-200 bg-green-50 px-4 py-2.5 text-sm text-green-700 dark:border-green-800 dark:bg-green-950/30 dark:text-green-400">
                <CheckCircle2 className="h-4 w-4 shrink-0" />
                Model saved to <code className="font-mono mx-1">{result.checkpoint_dir}</code>
              </div>

              {/* Top critical components */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Target className="h-4 w-4 text-red-500" />
                    Top Critical Components (GNN)
                  </CardTitle>
                  <CardDescription>Components with highest predicted composite criticality score</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {result.top_critical.slice((compPage - 1) * PAGE_SIZE, compPage * PAGE_SIZE).map((s) => (
                      <div key={s.component} className="rounded-lg border p-3 space-y-2 cursor-pointer hover:bg-muted/40 transition-colors group" onClick={() => router.push(`/explorer?node=${encodeURIComponent(s.component)}`)}>
                        <div className="flex items-center justify-between">
                          <div className="flex flex-col min-w-0 max-w-[60%]">
                            <span className="font-medium text-sm truncate group-hover:underline">{s.node_name || s.component}</span>
                            <Badge variant="outline" className="text-xs font-mono">
                              {s.composite_score.toFixed(3)}
                            </Badge>
                          </div>
                        </div>
                        <div className="space-y-0.5">
                          <ScoreBar value={s.reliability_score} dim="R" />
                          <ScoreBar value={s.maintainability_score} dim="M" />
                          <ScoreBar value={s.availability_score} dim="A" />
                          <ScoreBar value={s.vulnerability_score} dim="V" />
                        </div>
                      </div>
                    ))}
                    <PaginationBar page={compPage} total={result.top_critical.length} pageSize={PAGE_SIZE} onPage={setCompPage} />
                  </div>
                </CardContent>
              </Card>

              {/* Top critical edges */}
              {result.top_critical_edges.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-base">
                      <TrendingUp className="h-4 w-4 text-orange-500" />
                      Top Critical Relationships (GNN)
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {result.top_critical_edges.slice((edgePage - 1) * PAGE_SIZE, edgePage * PAGE_SIZE).map((e, i) => (
                        <div key={i} className="flex items-center justify-between rounded border p-2 text-sm">
                          <div className="flex items-center gap-1.5 min-w-0">
                            <div className="flex flex-col min-w-0 cursor-pointer hover:underline" onClick={() => router.push(`/explorer?node=${encodeURIComponent(e.source)}`)}>
                              <span className="truncate font-medium">{e.source_name || e.source}</span>
                              {e.source_name && e.source_name !== e.source && (
                                <span className="text-xs text-muted-foreground font-mono truncate">{e.source}</span>
                              )}
                            </div>
                            <span className="text-muted-foreground text-xs shrink-0">→</span>
                            <div className="flex flex-col min-w-0 cursor-pointer hover:underline" onClick={() => router.push(`/explorer?node=${encodeURIComponent(e.target)}`)}>
                              <span className="truncate">{e.target_name || e.target}</span>
                              {e.target_name && e.target_name !== e.target && (
                                <span className="text-xs text-muted-foreground font-mono truncate">{e.target}</span>
                              )}
                            </div>
                          </div>
                          <div className="flex items-center gap-2 shrink-0 ml-2">
                            {criticality_badge(e.criticality_level)}
                            <span className="font-mono text-xs">{e.composite_score.toFixed(3)}</span>
                          </div>
                        </div>
                      ))}
                      <PaginationBar page={edgePage} total={result.top_critical_edges.length} pageSize={PAGE_SIZE} onPage={setEdgePage} />
                    </div>
                  </CardContent>
                </Card>
              )}
            </>
          )}
        </div>
      )}
    </AppLayout>
  )
}
