"use client"

import { useState } from "react"
import { AppLayout } from "@/components/layout/app-layout"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Checkbox } from "@/components/ui/checkbox"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import {
  Database,
  Loader2,
  Settings,
  CheckCircle2,
  AlertTriangle,
  Trash2,
  Play,
  Upload,
  Download,
  Zap,
  HardDrive,
  BarChart3,
  Sliders,
  Layers,
} from "lucide-react"
import { useConnection } from "@/lib/stores/connection-store"
import { apiClient } from "@/lib/api/client"

const SCALE_PRESETS = {
  tiny:   { apps: 5,   topics: 5,   brokers: 1,  nodes: 2,  libs: 2   },
  small:  { apps: 15,  topics: 10,  brokers: 2,  nodes: 4,  libs: 5   },
  medium: { apps: 50,  topics: 30,  brokers: 3,  nodes: 8,  libs: 10  },
  large:  { apps: 150, topics: 100, brokers: 6,  nodes: 20, libs: 30  },
  xlarge: { apps: 500, topics: 300, brokers: 10, nodes: 50, libs: 100 },
} as const

const SCALE_LABELS: Record<string, string> = {
  tiny: "Tiny", small: "Small", medium: "Medium", large: "Large", xlarge: "X-Large"
}

const SCALES = (Object.entries(SCALE_PRESETS) as [string, { apps: number; topics: number; brokers: number; nodes: number; libs: number }][]).map(([value, c]) => {
  const total = c.apps + c.topics + c.brokers + c.nodes + c.libs
  return {
    value,
    label: SCALE_LABELS[value],
    description: `${total.toLocaleString()} total nodes`,
    count: `${c.apps} apps, ${c.topics} topics, ${c.brokers} broker${c.brokers !== 1 ? 's' : ''}, ${c.nodes} nodes, ${c.libs} libs`,
  }
})

export default function DataPage() {
  const { status, stats, initialLoadComplete } = useConnection()

  const [scale, setScale] = useState("small")
  const [clearFirst, setClearFirst] = useState(true)

  const [isGenerating, setIsGenerating] = useState(false)
  const [isClearing, setIsClearing] = useState(false)
  const [isImporting, setIsImporting] = useState(false)
  const [isDownloading, setIsDownloading] = useState(false)
  const [isDownloadingNeo4j, setIsDownloadingNeo4j] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  const isConnected = status === 'connected'

  const handleGenerate = async () => {
    if (!isConnected) return

    setIsGenerating(true)
    setError(null)
    setSuccess(null)

    try {
      const result = await apiClient.generateAndImport({
        scale,
        seed: 42,
        clear_first: clearFirst
      })

      setSuccess('Graph generated and imported successfully!')

      // Refresh connection stats
      setTimeout(() => {
        window.location.reload()
      }, 2000)
    } catch (error: any) {
      let errorMsg = 'Generation failed'
      if (error.response?.data?.detail) {
        if (typeof error.response.data.detail === 'string') {
          errorMsg = error.response.data.detail
        } else if (Array.isArray(error.response.data.detail)) {
          errorMsg = error.response.data.detail.map((e: any) => e.msg || JSON.stringify(e)).join(', ')
        } else {
          errorMsg = JSON.stringify(error.response.data.detail)
        }
      } else if (error.message) {
        errorMsg = error.message
      }
      setError(errorMsg)
    } finally {
      setIsGenerating(false)
    }
  }

  const handleClear = async () => {
    if (!isConnected) return
    if (!confirm('Are you sure you want to clear all data from the database? This action cannot be undone.')) {
      return
    }

    setIsClearing(true)
    setError(null)
    setSuccess(null)

    try {
      await apiClient.clearDatabase()
      setSuccess('Database cleared successfully')

      setTimeout(() => {
        window.location.reload()
      }, 1500)
    } catch (error: any) {
      let errorMsg = 'Clear operation failed'
      if (error.response?.data?.detail) {
        if (typeof error.response.data.detail === 'string') {
          errorMsg = error.response.data.detail
        } else if (Array.isArray(error.response.data.detail)) {
          errorMsg = error.response.data.detail.map((e: any) => e.msg || JSON.stringify(e)).join(', ')
        } else {
          errorMsg = JSON.stringify(error.response.data.detail)
        }
      } else if (error.message) {
        errorMsg = error.message
      }
      setError(errorMsg)
    } finally {
      setIsClearing(false)
    }
  }

  const handleImportFromFile = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!isConnected) return
    const file = event.target.files?.[0]
    if (!file) return

    setIsImporting(true)
    setError(null)
    setSuccess(null)

    try {
      const fileContent = await file.text()
      const graphData = JSON.parse(fileContent)
      
      await apiClient.importGraph(graphData, {
        clear_first: clearFirst
      })

      setSuccess('Graph imported successfully!')

      // Refresh connection stats
      setTimeout(() => {
        window.location.reload()
      }, 2000)
    } catch (error: any) {
      let errorMsg = 'Import failed'
      if (error.response?.data?.detail) {
        if (typeof error.response.data.detail === 'string') {
          errorMsg = error.response.data.detail
        } else if (Array.isArray(error.response.data.detail)) {
          errorMsg = error.response.data.detail.map((e: any) => e.msg || JSON.stringify(e)).join(', ')
        } else {
          errorMsg = JSON.stringify(error.response.data.detail)
        }
      } else if (error.message) {
        errorMsg = error.message
      }
      setError(errorMsg)
    } finally {
      setIsImporting(false)
      // Reset file input
      event.target.value = ''
    }
  }

  const handleDownloadGraph = async () => {
    setIsDownloading(true)
    setError(null)
    setSuccess(null)

    try {
      const blob = await apiClient.generateGraphFile({
        scale,
        scenario: 'generic',
        seed: 42
      })

      // Extract filename from response or create default
      const filename = `graph_${scale}.json`

      // Create download link and trigger download
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      link.style.display = 'none'
      
      document.body.appendChild(link)
      link.click()
      
      // Cleanup
      setTimeout(() => {
        document.body.removeChild(link)
        window.URL.revokeObjectURL(url)
      }, 100)

      setSuccess(`Graph file downloaded: ${filename}`)
    } catch (error: any) {
      let errorMsg = 'Download failed'
      
      // Handle blob error responses
      if (error.response?.data instanceof Blob) {
        try {
          const text = await error.response.data.text()
          const errorData = JSON.parse(text)
          if (errorData.detail) {
            if (typeof errorData.detail === 'string') {
              errorMsg = errorData.detail
            } else if (Array.isArray(errorData.detail)) {
              errorMsg = errorData.detail.map((e: any) => e.msg || JSON.stringify(e)).join(', ')
            } else {
              errorMsg = JSON.stringify(errorData.detail)
            }
          }
        } catch {
          errorMsg = 'Failed to download graph file'
        }
      } else if (error.response?.data?.detail) {
        if (typeof error.response.data.detail === 'string') {
          errorMsg = error.response.data.detail
        } else if (Array.isArray(error.response.data.detail)) {
          errorMsg = error.response.data.detail.map((e: any) => e.msg || JSON.stringify(e)).join(', ')
        } else {
          errorMsg = JSON.stringify(error.response.data.detail)
        }
      } else if (error.message) {
        errorMsg = error.message
      }
      
      setError(errorMsg)
    } finally {
      setIsDownloading(false)
    }
  }

  const handleDownloadNeo4jData = async () => {
    if (!isConnected) return

    setIsDownloadingNeo4j(true)
    setError(null)
    setSuccess(null)

    try {
      const blob = await apiClient.exportNeo4jData()

      // Create filename with timestamp
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5)
      const filename = `neo4j_export_${timestamp}.json`

      // Create download link and trigger download
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      link.style.display = 'none'
      
      document.body.appendChild(link)
      link.click()
      
      // Cleanup
      setTimeout(() => {
        document.body.removeChild(link)
        window.URL.revokeObjectURL(url)
      }, 100)

      setSuccess(`Neo4j data exported: ${filename}`)
    } catch (error: any) {
      let errorMsg = 'Export failed'
      
      // Handle blob error responses
      if (error.response?.data instanceof Blob) {
        try {
          const text = await error.response.data.text()
          const errorData = JSON.parse(text)
          if (errorData.detail) {
            if (typeof errorData.detail === 'string') {
              errorMsg = errorData.detail
            } else if (Array.isArray(errorData.detail)) {
              errorMsg = errorData.detail.map((e: any) => e.msg || JSON.stringify(e)).join(', ')
            } else {
              errorMsg = JSON.stringify(errorData.detail)
            }
          }
        } catch {
          errorMsg = 'Failed to export Neo4j data'
        }
      } else if (error.response?.data?.detail) {
        if (typeof error.response.data.detail === 'string') {
          errorMsg = error.response.data.detail
        } else if (Array.isArray(error.response.data.detail)) {
          errorMsg = error.response.data.detail.map((e: any) => e.msg || JSON.stringify(e)).join(', ')
        } else {
          errorMsg = JSON.stringify(error.response.data.detail)
        }
      } else if (error.message) {
        errorMsg = error.message
      }
      
      setError(errorMsg)
    } finally {
      setIsDownloadingNeo4j(false)
    }
  }

  // Loading State
  if (!initialLoadComplete || status === 'connecting') {
    return (
      <AppLayout title="Data Management" description="Generate and import graph data">
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" text={status === 'connecting' ? "Connecting to database..." : "Loading data management..."} />
        </div>
      </AppLayout>
    )
  }

  // Disconnected State - show only no connection component
  if (!isConnected) {
    return (
      <AppLayout title="Data Management" description="Generate and import graph data">
        <NoConnectionInfo description="Connect to your Neo4j database to manage graph data" />
      </AppLayout>
    )
  }

  return (
    <AppLayout title="Data Management" description="Generate and import graph data">
      <div className="space-y-5">

        {/* ── Database Overview KPIs ─────────────────────────────────────── */}
        {stats && (
          <div className="grid gap-3 grid-cols-2 sm:grid-cols-3 lg:grid-cols-5">
            {([
              { label: 'Components',  value: stats.total_nodes.toLocaleString(),   sub: 'Nodes in graph',        Icon: Database,     text: 'text-blue-400',    border: 'border-blue-500/20',    bg: 'bg-blue-500/[0.07]',    ring: 'bg-blue-500/10',    glow: 'bg-blue-500'    },
              { label: 'Total Edges', value: (stats.total_edges || 0).toLocaleString(), sub: 'Derived + structural', Icon: Zap,      text: 'text-purple-400',  border: 'border-purple-500/20',  bg: 'bg-purple-500/[0.07]',  ring: 'bg-purple-500/10',  glow: 'bg-purple-500'  },
              { label: 'Node Types',  value: String(stats.node_counts ? Object.keys(stats.node_counts).length : 0), sub: 'Unique component types', Icon: Layers, text: 'text-emerald-400', border: 'border-emerald-500/20', bg: 'bg-emerald-500/[0.07]', ring: 'bg-emerald-500/10', glow: 'bg-emerald-500' },
              { label: 'Edge Types',  value: String((stats.edge_counts ? Object.keys(stats.edge_counts).length : 0) + (stats.structural_edge_counts ? Object.keys(stats.structural_edge_counts).length : 0)), sub: 'Dependency + structural', Icon: Layers, text: 'text-cyan-400', border: 'border-cyan-500/20', bg: 'bg-cyan-500/[0.07]', ring: 'bg-cyan-500/10', glow: 'bg-cyan-500' },
              { label: 'DB Status',   value: stats.total_nodes === 0 ? 'Empty' : 'Active', sub: 'Current state', Icon: CheckCircle2, text: 'text-amber-400', border: 'border-amber-500/20', bg: 'bg-amber-500/[0.07]', ring: 'bg-amber-500/10', glow: 'bg-amber-500' },
            ] as const).map(({ label, value, sub, Icon, text, border, bg, ring, glow }) => (
              <div key={label} className={`relative overflow-hidden rounded-xl border ${border} ${bg} p-4`}>
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className="text-xs text-muted-foreground font-medium truncate">{label}</p>
                    <p className={`text-[1.65rem] font-bold leading-tight tracking-tight ${text}`}>{value}</p>
                    <p className="text-[11px] text-muted-foreground mt-0.5 truncate">{sub}</p>
                  </div>
                  <div className={`shrink-0 rounded-lg ${ring} p-2`}>
                    <Icon className={`h-4 w-4 ${text}`} />
                  </div>
                </div>
                <div className={`pointer-events-none absolute -bottom-5 -right-5 h-16 w-16 rounded-full blur-2xl opacity-20 ${glow}`} />
              </div>
            ))}
          </div>

        )}

        {/* ── Messages ──────────────────────────────────────────────────── */}
        {error && (
          <div className="flex items-start gap-3 rounded-xl border border-red-500/20 bg-red-500/[0.07] p-4 animate-in slide-in-from-top-2">
            <div className="shrink-0 rounded-lg bg-red-500/10 p-2">
              <AlertTriangle className="h-4 w-4 text-red-400" />
            </div>
            <div className="min-w-0">
              <p className="text-sm font-semibold text-red-400">Operation Failed</p>
              <p className="text-xs text-muted-foreground mt-0.5">{error}</p>
            </div>
          </div>
        )}

        {success && (
          <div className="flex items-start gap-3 rounded-xl border border-emerald-500/20 bg-emerald-500/[0.07] p-4 animate-in slide-in-from-top-2">
            <div className="shrink-0 rounded-lg bg-emerald-500/10 p-2">
              <CheckCircle2 className="h-4 w-4 text-emerald-400" />
            </div>
            <div className="min-w-0">
              <p className="text-sm font-semibold text-emerald-400">Success</p>
              <p className="text-xs text-muted-foreground mt-0.5">{success}</p>
            </div>
          </div>
        )}

        {/* ── Data Operations ─────────────────────────────────────────── */}
        <div className="space-y-3">
          <div className="flex items-center gap-2.5">
            <div className="shrink-0 rounded-lg bg-indigo-500/10 p-1.5">
              <Sliders className="h-4 w-4 text-indigo-400" />
            </div>
            <div>
              <p className="text-sm font-semibold">Data Operations</p>
              <p className="text-[11px] text-muted-foreground">Export, import, or erase your graph data</p>
            </div>
          </div>

          <div className="grid gap-3 grid-cols-2 lg:grid-cols-4">
            {/* Import Graph */}
            <div className="relative">
              <input
                type="file"
                id="import-file"
                accept=".json"
                onChange={handleImportFromFile}
                disabled={isGenerating || isImporting || isDownloading || isDownloadingNeo4j || isClearing}
                className="hidden"
              />
              <button
                onClick={() => document.getElementById('import-file')?.click()}
                disabled={isGenerating || isImporting || isDownloading || isDownloadingNeo4j || isClearing}
                className="text-left w-full disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <div className="rounded-xl border border-border p-4 transition-colors hover:border-purple-500/40 hover:bg-purple-500/[0.04]">
                  <div className="flex items-start justify-between gap-2 mb-3">
                    <div className="shrink-0 rounded-lg bg-purple-500/10 p-2">
                      {isImporting
                        ? <Loader2 className="h-4 w-4 text-purple-400 animate-spin" />
                        : <Upload className="h-4 w-4 text-purple-400" />}
                    </div>
                  </div>
                  <p className="text-lg font-bold leading-tight text-purple-400 mb-0.5">
                    {isImporting ? 'Importing…' : 'Import Graph'}
                  </p>
                  <p className="text-[11px] text-muted-foreground">
                    Upload a JSON topology file
                  </p>
                </div>
              </button>
            </div>

            {/* Export Database */}
            <button
              onClick={handleDownloadNeo4jData}
              disabled={isDownloadingNeo4j || isGenerating || isImporting || isDownloading || isClearing || !stats || stats.total_nodes === 0}
              className="text-left disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <div className="rounded-xl border border-border p-4 transition-colors hover:border-emerald-500/40 hover:bg-emerald-500/[0.04]">
                <div className="flex items-start justify-between gap-2 mb-3">
                  <div className="shrink-0 rounded-lg bg-emerald-500/10 p-2">
                    {isDownloadingNeo4j
                      ? <Loader2 className="h-4 w-4 text-emerald-400 animate-spin" />
                      : <HardDrive className="h-4 w-4 text-emerald-400" />}
                  </div>
                </div>
                <p className="text-lg font-bold leading-tight text-emerald-400 mb-0.5">
                  {isDownloadingNeo4j ? 'Exporting…' : 'Export Database'}
                </p>
                <p className="text-[11px] text-muted-foreground">
                  Download current Neo4j graph as JSON
                </p>
              </div>
            </button>

            {/* Generate Sample */}
            <button
              onClick={handleDownloadGraph}
              disabled={isGenerating || isImporting || isDownloading || isDownloadingNeo4j || isClearing}
              className="text-left disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <div className="rounded-xl border border-border p-4 transition-colors hover:border-blue-500/40 hover:bg-blue-500/[0.04]">
                <div className="flex items-start justify-between gap-2 mb-3">
                  <div className="shrink-0 rounded-lg bg-blue-500/10 p-2">
                    {isDownloading
                      ? <Loader2 className="h-4 w-4 text-blue-400 animate-spin" />
                      : <Download className="h-4 w-4 text-blue-400" />}
                  </div>
                </div>
                <p className="text-lg font-bold leading-tight text-blue-400 mb-0.5">
                  {isDownloading ? 'Generating…' : 'Generate Sample'}
                </p>
                <p className="text-[11px] text-muted-foreground">
                  Download a synthetic JSON graph
                </p>
              </div>
            </button>

            {/* Erase Database */}
            <button
              onClick={handleClear}
              disabled={isClearing || isGenerating || isImporting || isDownloading || isDownloadingNeo4j || !stats || stats.total_nodes === 0}
              className="text-left disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <div className="rounded-xl border border-border p-4 transition-colors hover:border-red-500/40 hover:bg-red-500/[0.04]">
                <div className="flex items-start justify-between gap-2 mb-3">
                  <div className="shrink-0 rounded-lg bg-red-500/10 p-2">
                    {isClearing
                      ? <Loader2 className="h-4 w-4 text-red-400 animate-spin" />
                      : <Trash2 className="h-4 w-4 text-red-400" />}
                  </div>
                </div>
                <p className="text-lg font-bold leading-tight text-red-400 mb-0.5">
                  {isClearing ? 'Erasing…' : 'Erase Database'}
                </p>
                <p className="text-[11px] text-muted-foreground">
                  Permanently remove all graph data
                </p>
              </div>
            </button>
          </div>
        </div>

        {/* ── Generate Graph ─────────────────────────────────────────── */}
        <div className="space-y-4">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center gap-2.5">
              <div className="shrink-0 rounded-lg bg-blue-500/10 p-1.5">
                <Settings className="h-4 w-4 text-blue-400" />
              </div>
              <div>
                <p className="text-sm font-semibold">Generate Graph</p>
                <p className="text-[11px] text-muted-foreground">Pick a scale preset and generate a synthetic topology</p>
              </div>
            </div>
            <Button
              onClick={handleGenerate}
              disabled={isGenerating || isImporting || isDownloading || isClearing}
              size="sm"
              className="shrink-0 bg-blue-500/15 text-blue-300 hover:bg-blue-500/25 border border-blue-500/30 transition-all"
            >
              {isGenerating ? (
                <><Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />Generating...</>
              ) : (
                <><Play className="mr-1.5 h-3.5 w-3.5" />Generate &amp; Import</>
              )}
            </Button>
          </div>

          {/* Scale preset tiles */}
          <div className="grid gap-3 grid-cols-2 sm:grid-cols-3 lg:grid-cols-5">
            {SCALES.map((s) => {
              const isSelected = scale === s.value
              const colorMap: Record<string, { text: string; selBorder: string; selBg: string; hover: string; ring: string; glow: string }> = {
                tiny:   { text: 'text-emerald-400', selBorder: 'border-emerald-500/40', selBg: 'bg-emerald-500/[0.04]', hover: 'hover:border-emerald-500/40 hover:bg-emerald-500/[0.04]', ring: 'bg-emerald-500/10', glow: 'bg-emerald-500' },
                small:  { text: 'text-blue-400',    selBorder: 'border-blue-500/40',    selBg: 'bg-blue-500/[0.04]',    hover: 'hover:border-blue-500/40 hover:bg-blue-500/[0.04]',       ring: 'bg-blue-500/10',    glow: 'bg-blue-500'    },
                medium: { text: 'text-purple-400',  selBorder: 'border-purple-500/40',  selBg: 'bg-purple-500/[0.04]',  hover: 'hover:border-purple-500/40 hover:bg-purple-500/[0.04]',   ring: 'bg-purple-500/10',  glow: 'bg-purple-500'  },
                large:  { text: 'text-amber-400',   selBorder: 'border-amber-500/40',   selBg: 'bg-amber-500/[0.04]',   hover: 'hover:border-amber-500/40 hover:bg-amber-500/[0.04]',     ring: 'bg-amber-500/10',   glow: 'bg-amber-500'   },
                xlarge: { text: 'text-rose-400',    selBorder: 'border-rose-500/40',    selBg: 'bg-rose-500/[0.04]',    hover: 'hover:border-rose-500/40 hover:bg-rose-500/[0.04]',       ring: 'bg-rose-500/10',    glow: 'bg-rose-500'    },
              }
              const c = colorMap[s.value] ?? colorMap.small
              return (
                <div
                  key={s.value}
                  onClick={() => setScale(s.value)}
                  className={`relative overflow-hidden rounded-xl border cursor-pointer transition-colors p-4 ${
                    isSelected ? `${c.selBorder} ${c.selBg}` : `border-border ${c.hover}`
                  }`}
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <div className="flex items-center gap-1.5">
                        <p className="text-xs text-muted-foreground font-medium truncate">{s.label}</p>
                        {isSelected && <span className={`text-[10px] font-bold ${c.text}`}>✓</span>}
                      </div>
                      <p className={`text-[1.65rem] font-bold leading-tight tracking-tight ${c.text}`}>
                        {s.description.split(' ')[0]}
                      </p>
                      <p className="text-[11px] text-muted-foreground mt-0.5 truncate">total nodes</p>
                    </div>
                    <div className={`shrink-0 rounded-lg ${c.ring} p-2`}>
                      <BarChart3 className={`h-4 w-4 ${c.text}`} />
                    </div>
                  </div>
                  <p className="text-[10px] text-muted-foreground mt-2 leading-relaxed truncate">{s.count}</p>
                  <div className={`pointer-events-none absolute -bottom-5 -right-5 h-16 w-16 rounded-full blur-2xl opacity-20 ${c.glow}`} />
                </div>
              )
            })}
          </div>

          {/* Clear-first option */}
          <div className="relative overflow-hidden rounded-xl border border-amber-500/20 bg-amber-500/[0.07] p-4">
            <div className="flex items-start gap-3">
              <div className="shrink-0 rounded-lg bg-amber-500/10 p-2 mt-0.5">
                <AlertTriangle className="h-4 w-4 text-amber-400" />
              </div>
              <div className="flex-1 space-y-1.5">
                <div className="flex items-center gap-2.5">
                  <Checkbox
                    id="clear-first"
                    checked={clearFirst}
                    onCheckedChange={(checked) => setClearFirst(checked as boolean)}
                    className="border-amber-500/50 data-[state=checked]:bg-amber-500 data-[state=checked]:border-amber-500"
                  />
                  <label htmlFor="clear-first" className="text-sm font-semibold leading-none cursor-pointer">
                    Clear database before generating
                  </label>
                </div>
                <p className="text-[11px] text-muted-foreground pl-6">
                  Removes all existing data before importing. Uncheck to merge with existing data.
                </p>
              </div>
            </div>
            <div className="pointer-events-none absolute -bottom-5 -right-5 h-16 w-16 rounded-full blur-2xl opacity-20 bg-amber-500" />
          </div>
        </div>

      </div>
    </AppLayout>
  )
}
