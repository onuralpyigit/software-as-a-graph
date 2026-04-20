"use client"

import React, { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import {
  Cpu,
  Play,
  AlertTriangle,
  Loader2,
  Target,
  TrendingUp,
  Info,
  Download,
  CheckCircle2,
  RefreshCw,
  FolderOpen,
  Package,
  Brain,
  GitBranch,
  Layers,
  Check,
  Network,
  Trash2,
} from "lucide-react"
import { useConnection } from "@/lib/stores/connection-store"
import { apiClient } from "@/lib/api/client"
import axios from "axios"
import { TermTooltip } from "@/components/ui/term-tooltip"
import { ScoreTooltip } from "@/components/ui/score-tooltip"

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

interface PredictSummary {
  total_components: number
  critical: number
  high: number
  medium: number
  low: number
  minimal: number
  critical_edges: number
}

interface PredictResult {
  success: boolean
  layer: string
  checkpoint_dir: string
  summary: PredictSummary
  scores: GNNScore[]
  edge_scores: GNNEdgeScore[]
}

interface CheckpointInfo {
  path: string
  name: string
  layer: string
  hidden_channels: number
  num_heads: number
  num_layers: number
  dropout: number
  predict_edges: boolean
  has_node_model: boolean
  has_edge_model: boolean
  has_ensemble: boolean
}

// ── Helpers ───────────────────────────────────────────────────────────────────

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
      <span className="w-10 text-right text-xs tabular-nums">
        <ScoreTooltip score={value} type="raw-risk" side="left">{value.toFixed(3)}</ScoreTooltip>
      </span>
    </div>
  )
}

// ── Checkpoint card ──────────────────────────────────────────────────────────

function CheckpointCard({
  ck,
  selected,
  onSelect,
  onDelete,
}: {
  ck: CheckpointInfo
  selected: boolean
  onSelect: () => void
  onDelete: () => void
}) {
  const [confirmDelete, setConfirmDelete] = React.useState(false)

  return (
    <div
      className={`rounded-lg border-2 transition-all ${
        selected
          ? "border-purple-500 bg-purple-50 dark:bg-purple-950/30"
          : "border-border hover:border-purple-300 hover:bg-muted/40"
      }`}
    >
      {/* Header row: name + layer badge + selected check + delete */}
      <div className="flex items-center justify-between gap-2 px-4 pt-4 pb-0">
        <button onClick={onSelect} className="flex items-center gap-2 min-w-0 flex-1 text-left">
          <FolderOpen className={`h-4 w-4 shrink-0 ${selected ? "text-purple-500" : "text-muted-foreground"}`} />
          <span className="font-semibold truncate">{ck.name}</span>
        </button>

        <div className="flex items-center gap-1.5 shrink-0">
          {ck.layer && (
            <span className="rounded-full bg-slate-100 px-2 py-0.5 text-[10px] font-bold text-slate-600 dark:bg-slate-800 dark:text-slate-300 uppercase tracking-wide">
              {{ app: "Application", infra: "Infrastructure", mw: "Middleware", system: "System" }[ck.layer] ?? ck.layer}
            </span>
          )}
          {selected && (
            <span className="rounded-full bg-purple-500 p-0.5">
              <Check className="h-3 w-3 text-white" />
            </span>
          )}
          {/* Delete */}
          {confirmDelete ? (
            <div className="flex items-center gap-1">
              <span className="text-xs text-red-600 dark:text-red-400 font-medium">Delete?</span>
              <button
                onClick={e => { e.stopPropagation(); onDelete() }}
                className="rounded px-1.5 py-0.5 text-xs font-medium bg-red-500 text-white hover:bg-red-600"
              >
                Yes
              </button>
              <button
                onClick={e => { e.stopPropagation(); setConfirmDelete(false) }}
                className="rounded px-1.5 py-0.5 text-xs font-medium bg-muted hover:bg-muted/80 text-foreground"
              >
                No
              </button>
            </div>
          ) : (
            <button
              onClick={e => { e.stopPropagation(); setConfirmDelete(true) }}
              className="rounded p-1 text-muted-foreground hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-950/30 transition-colors"
              title="Delete checkpoint"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Body — clicking selects the checkpoint */}
      <button onClick={onSelect} className="w-full text-left px-4 pt-2 pb-4">
        <div className="grid grid-cols-3 gap-x-4 gap-y-1 text-xs">
          <div className="flex items-center gap-1 text-muted-foreground">
            <Brain className="h-3 w-3" />
            <span>hidden={ck.hidden_channels}</span>
          </div>
          <div className="flex items-center gap-1 text-muted-foreground">
            <Network className="h-3 w-3" />
            <span>heads={ck.num_heads}</span>
          </div>
          <div className="flex items-center gap-1 text-muted-foreground">
            <Layers className="h-3 w-3" />
            <span>layers={ck.num_layers}</span>
          </div>
          <div className="flex items-center gap-1 text-muted-foreground">
            <GitBranch className="h-3 w-3" />
            <span>dropout={ck.dropout}</span>
          </div>
        </div>

        <div className="mt-2 flex flex-wrap gap-1">
          {ck.has_node_model && (
            <span className="rounded-full bg-blue-100 px-1.5 py-0.5 text-[10px] font-medium text-blue-700 dark:bg-blue-950 dark:text-blue-300">
              node model
            </span>
          )}
          {ck.has_edge_model && (
            <span className="rounded-full bg-green-100 px-1.5 py-0.5 text-[10px] font-medium text-green-700 dark:bg-green-950 dark:text-green-300">
              edge model
            </span>
          )}
          {ck.has_ensemble && (
            <span className="rounded-full bg-purple-100 px-1.5 py-0.5 text-[10px] font-medium text-purple-700 dark:bg-purple-950 dark:text-purple-300">
              ensemble
            </span>
          )}
          {ck.predict_edges && (
            <span className="rounded-full bg-orange-100 px-1.5 py-0.5 text-[10px] font-medium text-orange-700 dark:bg-orange-950 dark:text-orange-300">
              edge preds
            </span>
          )}
        </div>

        <p className="mt-2 text-[10px] text-muted-foreground font-mono truncate">{ck.path}</p>
      </button>
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

export default function PredictPage() {
  const router = useRouter()
  const { config, status } = useConnection()
  const isConnected = status === "connected"

  const [layer, setLayer] = useState("app")
  const [checkpointDir, setCheckpointDir] = useState("")
  const [filterLevel, setFilterLevel] = useState("ALL")

  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([])
  const [checkpointsLoading, setCheckpointsLoading] = useState(false)

  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [result, setResult] = useState<PredictResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  // Pagination
  const PAGE_SIZE = 10
  const [compPage, setCompPage] = useState(1)
  const [edgePage, setEdgePage] = useState(1)

  const loadCheckpoints = async () => {
    setCheckpointsLoading(true)
    try {
      const res = await axios.get(`${apiClient.getBaseURL()}/api/v1/prediction/checkpoints`)
      const list: CheckpointInfo[] = res.data.checkpoints ?? []
      setCheckpoints(list)
      if (list.length > 0 && !checkpointDir) {
        setCheckpointDir(list[0].path)
        if (list[0].layer) setLayer(list[0].layer)
      }
    } catch {
      // silently ignore — user can still type a path
    } finally {
      setCheckpointsLoading(false)
    }
  }

  useEffect(() => { loadCheckpoints() }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const deleteCheckpoint = async (ck: CheckpointInfo) => {
    try {
      await axios.delete(`${apiClient.getBaseURL()}/api/v1/prediction/checkpoints/${ck.name}`)
      if (checkpointDir === ck.path) setCheckpointDir("")
      await loadCheckpoints()
    } catch (e: any) {
      setError(e?.response?.data?.detail ?? e?.message ?? "Delete failed")
    }
  }

  const runPrediction = async () => {
    if (!isConnected || !config) return

    setIsRunning(true)
    setResult(null)
    setError(null)
    setProgress(15)

    const tick = setInterval(() => {
      setProgress(p => Math.min(p + 5, 90))
    }, 1500)

    try {
      const response = await axios.post(`${apiClient.getBaseURL()}/api/v1/prediction/predict`, {
        credentials: config,
        layer,
        checkpoint_dir: checkpointDir,
      }, { timeout: 600_000 }) // 10 min max

      setProgress(100)
      setResult(response.data)
      setCompPage(1)
      setEdgePage(1)
    } catch (e: any) {
      const msg = e?.response?.data?.detail ?? e?.message ?? "Prediction failed"
      setError(msg)
    } finally {
      clearInterval(tick)
      setIsRunning(false)
    }
  }

  const levels = ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW", "MINIMAL"]

  const filteredScores = result?.scores.filter(
    s => filterLevel === "ALL" || s.criticality_level === filterLevel
  ) ?? []

  const pagedScores = filteredScores.slice((compPage - 1) * PAGE_SIZE, compPage * PAGE_SIZE)
  const pagedEdges = (result?.edge_scores ?? []).slice((edgePage - 1) * PAGE_SIZE, edgePage * PAGE_SIZE)

  const downloadJSON = () => {
    if (!result) return
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `gnn_prediction_${result.layer}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <AppLayout
      title="Predict with GNN"
      description="Run inference with a trained GNN model to predict component criticality"
    >
      {!isConnected ? (
        <NoConnectionInfo />
      ) : (
        <div className="space-y-6">

          {/* Section header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-gradient-to-br from-purple-600 via-pink-600 to-rose-600 p-3 shadow-lg">
                <Cpu className="h-6 w-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold">Inference Configuration</h2>
                <p className="text-sm text-muted-foreground">Select a checkpoint and layer to run GNN inference</p>
              </div>
            </div>
            <Button
              onClick={runPrediction}
              disabled={isRunning}
              size="lg"
              className="min-w-[180px] h-12 bg-gradient-to-r from-purple-600 via-pink-600 to-rose-600 hover:from-purple-700 hover:via-pink-700 hover:to-rose-700 shadow-lg hover:shadow-xl transition-all text-base font-semibold"
            >
              {isRunning ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  Running inference…
                </>
              ) : (
                <>
                  <Play className="mr-2 h-5 w-5" />
                  Run Prediction
                </>
              )}
            </Button>
          </div>

          {/* Configuration card */}
          <Card>
            <CardContent className="space-y-4 pt-6">
              {/* Checkpoint picker */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Select Checkpoint</Label>
                  <button
                    onClick={loadCheckpoints}
                    disabled={checkpointsLoading}
                    className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
                  >
                    <RefreshCw className={`h-3 w-3 ${checkpointsLoading ? "animate-spin" : ""}`} />
                    Refresh
                  </button>
                </div>

                {checkpointsLoading ? (
                  <div className="flex items-center justify-center py-8 text-muted-foreground text-sm gap-2">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Scanning for checkpoints…
                  </div>
                ) : checkpoints.length === 0 ? (
                  <div className="rounded-lg border-2 border-dashed p-6 text-center">
                    <Package className="mx-auto h-8 w-8 text-muted-foreground/40 mb-2" />
                    <p className="text-sm text-muted-foreground font-medium">No checkpoints found</p>
                    <p className="text-xs text-muted-foreground mt-1">
                      Train a model first on the <strong>Train GNN</strong> page.
                    </p>
                  </div>
                ) : (
                  <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                    {checkpoints.map(ck => (
                      <CheckpointCard
                        key={ck.path}
                        ck={ck}
                        selected={checkpointDir === ck.path}
                        onSelect={() => {
                          setCheckpointDir(ck.path)
                          if (ck.layer) setLayer(ck.layer)
                        }}
                        onDelete={() => deleteCheckpoint(ck)}
                      />
                    ))}
                  </div>
                )}

                {checkpoints.length > 0 && (
                  <p className="text-xs text-muted-foreground flex items-center gap-1">
                    <Package className="h-3 w-3" />
                    {checkpoints.length} checkpoint{checkpoints.length !== 1 ? "s" : ""} found
                  </p>
                )}
              </div>

              {/* Layer selector */}
              {(() => {
                const selectedCk = checkpoints.find(c => c.path === checkpointDir)
                const lockedLayer = selectedCk?.layer
                return (
                  <div className="space-y-1.5">
                    <Label htmlFor="layer">Layer</Label>
                    {lockedLayer ? (
                      <div className="flex items-center gap-2">
                        <div className="flex h-9 w-48 items-center rounded-md border bg-muted px-3 text-sm font-medium">
                          {{ app: "Application", infra: "Infrastructure", mw: "Middleware", system: "System" }[lockedLayer] ?? lockedLayer}
                        </div>
                        <span className="text-xs text-muted-foreground">locked to checkpoint layer</span>
                      </div>
                    ) : (
                      <Select value={layer} onValueChange={setLayer}>
                        <SelectTrigger id="layer" className="w-48">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="app">Application</SelectItem>
                          <SelectItem value="infra">Infrastructure</SelectItem>
                          <SelectItem value="mw">Middleware</SelectItem>
                          <SelectItem value="system">System</SelectItem>
                        </SelectContent>
                      </Select>
                    )}
                  </div>
                )
              })()}

              <div className="flex items-start gap-2 rounded-lg border border-blue-200 bg-blue-50 p-3 text-xs text-blue-700 dark:border-blue-800 dark:bg-blue-950/30 dark:text-blue-400">
                <Info className="h-4 w-4 shrink-0 mt-0.5" />
                Structural analysis (Step 2) and RMAV scoring (Step 3) will run automatically to build node
                features for the inference pass.
              </div>

              {isRunning && (
                <div className="space-y-1.5">
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>Analysing graph → building features → GNN inference…</span>
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
                  { label: "Critical", value: result.summary.critical, color: "text-red-500" },
                  { label: "High", value: result.summary.high, color: "text-orange-500" },
                  { label: "Medium", value: result.summary.medium, color: "text-yellow-500" },
                  { label: "Low", value: result.summary.low, color: "text-blue-500" },
                  { label: "Minimal", value: result.summary.minimal, color: "text-muted-foreground" },
                  { label: "Total", value: result.summary.total_components, color: "text-foreground" },
                ].map(({ label, value, color }) => (
                  <Card key={label}>
                    <CardContent className="p-3 text-center">
                      <p className={`text-xl font-bold ${color}`}>{value}</p>
                      <p className="text-xs text-muted-foreground">{label}</p>
                    </CardContent>
                  </Card>
                ))}
              </div>

              {/* Actions bar */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Label className="text-sm whitespace-nowrap">Filter by level:</Label>
                  <Select value={filterLevel} onValueChange={v => { setFilterLevel(v); setCompPage(1) }}>
                    <SelectTrigger className="w-36">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {levels.map(l => (
                        <SelectItem key={l} value={l}>{l}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <Button variant="outline" size="sm" onClick={downloadJSON}>
                  <Download className="mr-2 h-4 w-4" />
                  Export JSON
                </Button>
              </div>

              {/* Scores table */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2 text-base">
                    <Target className="h-4 w-4 text-purple-500" />
                    Component Criticality Predictions
                    <Badge variant="secondary" className="ml-auto">{filteredScores.length} components</Badge>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {filteredScores.length === 0 ? (
                    <p className="text-sm text-muted-foreground text-center py-6">
                      No components match the selected filter.
                    </p>
                  ) : (
                    <div className="space-y-2">
                      {pagedScores.map((s) => (
                        <div key={s.component} className="rounded-lg border p-3 space-y-2 cursor-pointer hover:bg-muted/40 transition-colors group" onClick={() => router.push(`/explorer?node=${encodeURIComponent(s.component)}`)}>
                          <div className="flex items-center justify-between">
                            <div className="flex flex-col min-w-0">
                              <span className="font-medium text-sm truncate group-hover:underline">{s.node_name || s.component}</span>
                              {s.node_name && s.node_name !== s.component && (
                                <span className="text-xs text-muted-foreground font-mono truncate">{s.component}</span>
                              )}
                            </div>
                            <div className="flex items-center gap-2 shrink-0">
                              <Badge variant="outline" className="text-xs font-mono">{s.source}</Badge>
                              {criticality_badge(s.criticality_level)}
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
                      <PaginationBar page={compPage} total={filteredScores.length} pageSize={PAGE_SIZE} onPage={setCompPage} />
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Critical edges */}
              {result.edge_scores.length > 0 && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-base">
                      <TrendingUp className="h-4 w-4 text-orange-500" />
                      Critical Relationships
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      {pagedEdges.map((e, i) => (
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
                            <span className="text-xs text-muted-foreground shrink-0">({e.edge_type})</span>
                          </div>
                          <div className="flex items-center gap-2 shrink-0 ml-2">
                            {criticality_badge(e.criticality_level)}
                            <span className="font-mono text-xs">{e.composite_score.toFixed(3)}</span>
                          </div>
                        </div>
                      ))}
                      <PaginationBar page={edgePage} total={result.edge_scores.length} pageSize={PAGE_SIZE} onPage={setEdgePage} />
                    </div>
                  </CardContent>
                </Card>
              )}

              <div className="flex items-center gap-2 rounded-lg border border-green-200 bg-green-50 px-4 py-2.5 text-sm text-green-700 dark:border-green-800 dark:bg-green-950/30 dark:text-green-400">
                <CheckCircle2 className="h-4 w-4 shrink-0" />
                Prediction complete — checkpoint: <code className="font-mono mx-1">{result.checkpoint_dir}</code>, layer: <code className="font-mono">{result.layer}</code>
              </div>
            </>
          )}
        </div>
      )}
    </AppLayout>
  )
}
