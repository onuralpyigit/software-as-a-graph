"use client"

import { useEffect, useState, useMemo, type ReactElement } from "react"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { useConnection } from "@/lib/stores/connection-store"
import { API_BASE_URL } from "@/lib/config/api"
import { apiClient } from "@/lib/api/client"
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, ResponsiveContainer,
} from "recharts"
import {
  ChartContainer, ChartTooltip, ChartTooltipContent,
} from "@/components/ui/chart"
import {
  Activity, AlertTriangle, BarChart3, Layers, Network, Radio,
  Shield, Box, Server, BookOpen, RefreshCw,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { TermTooltip } from "@/components/ui/term-tooltip"

// ── Types ──────────────────────────────────────────────────────────────

interface SummaryDict { [key: string]: number | string }

interface ExtrasStats {
  topic_bandwidth?: {
    labels: string[]
    ids: string[]
    sizes: number[]
    subs: number[]
    pubs: number[]
    bandwidth: number[]
    bandwidth_sub: number[]
    bandwidth_pub: number[]
    bandwidth_pubsub: number[]
    summary: SummaryDict
    outlier_indices: number[]
  }
  app_balance?: {
    labels: string[]
    ids: string[]
    pubs: number[]
    subs: number[]
    io_load: number[]
    summary: SummaryDict
    outlier_indices: number[]
  }
  topic_fanout?: {
    labels: string[]
    ids: string[]
    pubs: number[]
    subs: number[]
    fanout: number[]
    summary: SummaryDict
    outlier_indices: number[]
  }
  cross_node_heatmap?: {
    labels: string[]
    node_ids: string[]
    matrix: number[][]
    matrix_kb: number[][]
    summary: SummaryDict
    outlier_pairs: [string, string, number, number][]
    per_node?: Record<string, {
      label: string
      apps: { id: string; name: string }[]
      pub_topics: { id: string; name: string; size_kb: number }[]
      sub_topics: { id: string; name: string; size_kb: number }[]
    }>
  }
  node_comm_load?: {
    sorted_labels: string[]
    sorted_ids: string[]
    sorted_pub: number[]
    sorted_sub: number[]
    all_totals: number[]
    summary: SummaryDict
    outliers: [string, number, number, number, number][]
  }
  domain_comm?: {
    labels: string[]
    matrix: number[][]
    summary: SummaryDict
    outlier_pairs: [string, string, number, number][]
  }
  criticality_io?: {
    crit_labels: string[]
    crit_ids: string[]
    crit_pubs: number[]
    crit_subs: number[]
    norm_labels: string[]
    norm_ids: string[]
    norm_pubs: number[]
    norm_subs: number[]
    summary: SummaryDict
  }
  lib_dependency?: {
    labels: string[]
    display_ids: string[]
    in_vals: number[]
    out_vals: number[]
    summary: SummaryDict
    outliers: [string, number, number][]
  }
  node_critical_density?: {
    sorted_labels: string[]
    sorted_ids: string[]
    sorted_crit: number[]
    sorted_norm: number[]
    summary: SummaryDict
  }
  domain_diversity?: {
    labels: string[]
    app_counts: number[]
    topic_counts: number[]
    io_vals: number[]
    summary: SummaryDict
  }
}

// ── Helpers ─────────────────────────────────────────────────────────────

function fmtNum(v: number | string): string {
  return typeof v === "number"
    ? Number(v).toLocaleString(undefined, { maximumFractionDigits: 2 })
    : String(v)
}

function truncate(s: string, n: number = 18): string {
  return s.length > n ? s.slice(0, n) + "…" : s
}

const BAR_SLOT_WIDTH = 80
const CHART_MIN_WIDTH = 300
const CHART_Y_MARGIN = 60
const MAX_ITEMS = 20
const PAGE_SIZE = 20

// ── Pagination + Search ─────────────────────────────────────────────────

function usePaginatedSearch<T extends { name: string }>(items: T[]) {
  const [search, setSearch] = useState("")
  const [page, setPage] = useState(0)

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    return q ? items.filter((it) => it.name.toLowerCase().includes(q)) : items
  }, [items, search])

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
  const safePage = Math.min(page, totalPages - 1)
  const pageItems = filtered.slice(safePage * PAGE_SIZE, (safePage + 1) * PAGE_SIZE)

  // Reset to page 0 when search changes
  const handleSearch = (v: string) => { setSearch(v); setPage(0) }

  return { search, handleSearch, page: safePage, setPage, filtered, pageItems, totalPages }
}

function PaginationBar({
  search, onSearch, page, totalPages, onPage, totalItems, filteredItems,
}: {
  search: string; onSearch: (v: string) => void
  page: number; totalPages: number; onPage: (p: number) => void
  totalItems: number; filteredItems: number
}) {
  return (
    <div className="flex flex-wrap items-center gap-2 justify-between">
      <div className="flex items-center gap-1.5">
        <input
          type="search"
          placeholder="Search…"
          value={search}
          onChange={(e) => onSearch(e.target.value)}
          className="h-7 w-40 rounded-md border bg-background px-2 text-xs focus:outline-none focus:ring-1 focus:ring-ring"
        />
        <span className="text-xs text-muted-foreground">
          {search ? `${filteredItems} of ${totalItems}` : `${totalItems} items`}
        </span>
      </div>
      {totalPages > 1 && (
        <div className="flex items-center gap-1">
          <button onClick={() => onPage(0)} disabled={page === 0} className="h-6 w-6 text-xs rounded border disabled:opacity-30 hover:bg-muted transition-colors">«</button>
          <button onClick={() => onPage(page - 1)} disabled={page === 0} className="h-6 w-6 text-xs rounded border disabled:opacity-30 hover:bg-muted transition-colors">‹</button>
          <span className="text-xs px-1">{page + 1} / {totalPages}</span>
          <button onClick={() => onPage(page + 1)} disabled={page >= totalPages - 1} className="h-6 w-6 text-xs rounded border disabled:opacity-30 hover:bg-muted transition-colors">›</button>
          <button onClick={() => onPage(totalPages - 1)} disabled={page >= totalPages - 1} className="h-6 w-6 text-xs rounded border disabled:opacity-30 hover:bg-muted transition-colors">»</button>
        </div>
      )}
    </div>
  )
}

function goToExplorer(id: string | undefined) {
  if (id) window.open(`/explorer?node=${encodeURIComponent(id)}`, "_blank")
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function handleBarClick(e: any) {
  if (e?.activePayload?.[0]?.payload?.id) goToExplorer(e.activePayload[0].payload.id)
}

function SizedBarChart({ dataCount, config, children }: { dataCount: number; config: Record<string, { label: string; color: string }>; children: ReactElement }) {
  const innerWidth = Math.max(CHART_MIN_WIDTH, dataCount * BAR_SLOT_WIDTH + CHART_Y_MARGIN)
  return (
    <div className="overflow-x-auto">
      <ChartContainer config={config} className="h-[250px] sm:h-[300px] md:h-[350px] lg:h-[400px]" style={{ width: Math.min(innerWidth, 9999) }}>
        {children}
      </ChartContainer>
    </div>
  )
}

function SummaryCards({ summary, keys }: { summary: SummaryDict; keys: { key: string; label: string; format?: (v: number | string) => string }[] }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {keys.map(({ key, label, format }) => {
        const val = summary[key]
        if (val === undefined) return null
        return (
          <div key={key} className="rounded-lg border bg-card p-3">
            <p className="text-xs text-muted-foreground">{label}</p>
            <p className="text-lg font-semibold">{format ? format(val) : fmtNum(val)}</p>
          </div>
        )
      })}
    </div>
  )
}

function OutlierTable({ rows, headers }: { rows: (string | number)[][]; headers: string[] }) {
  if (!rows || rows.length === 0) return <p className="text-sm text-muted-foreground">No outliers detected</p>
  return (
    <div className="max-h-64 overflow-auto rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            {headers.map(h => <TableHead key={h}>{h}</TableHead>)}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.slice(0, MAX_ITEMS).map((row, i) => (
            <TableRow key={i}>
              {row.map((cell, j) => (
                <TableCell key={j}>{typeof cell === "number" ? fmtNum(cell) : cell}</TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}

function MetricInsightCard({
  label,
  value,
  description,
  formula,
  unit,
}: {
  label: string
  value: string | number
  description: string
  formula?: string
  unit?: string
}) {
  return (
    <div className="rounded-lg border bg-card p-4 flex flex-col gap-1.5">
      <p className="text-[11px] font-semibold text-muted-foreground uppercase tracking-widest leading-none">{label}</p>
      <p className="text-xl font-bold leading-tight">
        {typeof value === "number" ? fmtNum(value) : value}
        {unit && <span className="text-xs font-normal text-muted-foreground ml-1">{unit}</span>}
      </p>
      <p className="text-xs text-muted-foreground leading-relaxed">{description}</p>
      {formula && (
        <code className="text-[10px] font-mono bg-muted/80 rounded px-2 py-0.5 text-muted-foreground/90 mt-0.5 self-start">
          {formula}
        </code>
      )}
    </div>
  )
}

// ── Sections ────────────────────────────────────────────────────────────

type BandwidthMode = "sub" | "pub" | "pubsub"

const BANDWIDTH_MODE_CONFIG: Record<BandwidthMode, { label: string; multiplierLabel: string; avgKey: string; avgLabel: string }> = {
  sub:    { label: "Subscribers",  multiplierLabel: "Size × Subscribers",  avgKey: "sub_mean",  avgLabel: "Avg Subscribers" },
  pub:    { label: "Publishers",   multiplierLabel: "Size × Publishers",   avgKey: "pub_mean",  avgLabel: "Avg Publishers" },
  pubsub: { label: "Pub + Sub",    multiplierLabel: "Size × (Pub + Sub)",  avgKey: "sub_mean",  avgLabel: "Avg Subs" },
}

function TopicBandwidthSection({ data }: { data: ExtrasStats["topic_bandwidth"] }) {
  const [mode, setMode] = useState<BandwidthMode>("sub")

  const bwArray = mode === "pub" ? (data?.bandwidth_pub ?? data?.bandwidth ?? [])
                : mode === "pubsub" ? (data?.bandwidth_pubsub ?? data?.bandwidth ?? [])
                : (data?.bandwidth_sub ?? data?.bandwidth ?? [])

  const allItems = useMemo(() => (data?.labels ?? []).map((label, i) => ({
    name: label,
    id: data?.ids?.[i],
    bandwidth: bwArray[i] ?? 0,
  })).sort((a, b) => b.bandwidth - a.bandwidth), [data, mode]) // eslint-disable-line react-hooks/exhaustive-deps

  const { search, handleSearch, page, setPage, pageItems, totalPages, filtered } = usePaginatedSearch(allItems)

  if (!data) return null
  const cfg = BANDWIDTH_MODE_CONFIG[mode]

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        <MetricInsightCard
          label="Avg Subscriber Bandwidth"
          value={fmtNum(data.summary.bw_mean ?? 0)}
          unit="bytes"
          description="Mean subscriber-side bandwidth per active topic. Measures how many bytes flow to all consumers per publish event. High values signal topics that strain network capacity."
          formula="bandwidth_sub = size × sub_count"
        />
        <MetricInsightCard
          label="Zero-Subscriber Topics"
          value={data.summary.zero_sub_count ?? 0}
          description="Topics that are published to but never consumed. Dead channels waste publisher resources and typically indicate incomplete pub/sub wiring or stale topic definitions."
          formula="count(topics where sub_count = 0)"
        />
        <MetricInsightCard
          label="Bandwidth Outliers"
          value={data.summary.outlier_count ?? 0}
          description="Topics whose bandwidth exceeds the IQR upper fence. A small number of outlier topics can dominate total network utilisation."
          formula="outlier if bandwidth > Q3 + 1.5 × IQR"
        />
      </div>
      <SummaryCards summary={data.summary} keys={[
        { key: "total_topics", label: "Total Topics" },
        { key: "size_mean", label: "Avg Size", format: (v) => fmtNum(v) + " bytes" },
        { key: cfg.avgKey, label: cfg.avgLabel },
        { key: "outlier_count", label: "Outliers" },
      ]} />
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <CardTitle className="text-base">Topic Bandwidth ({cfg.multiplierLabel})</CardTitle>
            <div className="flex items-center gap-1 rounded-md border p-0.5 bg-muted/50">
              {(["sub", "pub", "pubsub"] as BandwidthMode[]).map((m) => (
                <button key={m} onClick={() => setMode(m)}
                  className={`px-2.5 py-0.5 text-xs rounded transition-colors ${mode === m ? "bg-background shadow font-medium" : "text-muted-foreground hover:text-foreground"}`}>
                  {BANDWIDTH_MODE_CONFIG[m].label}
                </button>
              ))}
            </div>
          </div>
          <PaginationBar search={search} onSearch={handleSearch} page={page} totalPages={totalPages} onPage={setPage} totalItems={allItems.length} filteredItems={filtered.length} />
        </CardHeader>
        <CardContent>
          <SizedBarChart dataCount={pageItems.length} config={{ bandwidth: { label: "Bandwidth", color: "#8b5cf6" } }}>
            <BarChart data={pageItems.map((d) => ({ ...d, name: truncate(d.name) }))} margin={{ bottom: 50 }} maxBarSize={48} onClick={handleBarClick} className="cursor-pointer">
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={70} tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Bar dataKey="bandwidth" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </SizedBarChart>
        </CardContent>
      </Card>
    </div>
  )
}

function AppBalanceSection({ data }: { data: ExtrasStats["app_balance"] }) {
  const allItems = useMemo(() => (!data ? [] : data.labels.map((label, i) => ({
    name: label,
    id: data.ids?.[i],
    publishes: data.pubs[i],
    subscribes: data.subs[i],
  })).sort((a, b) => (b.publishes + b.subscribes) - (a.publishes + a.subscribes))), [data])

  const { search, handleSearch, page, setPage, pageItems, totalPages, filtered } = usePaginatedSearch(allItems)

  if (!data) return null

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        <MetricInsightCard
          label="High I/O Apps"
          value={data.summary.q_high_io ?? 0}
          description="Applications that both publish and subscribe above the system mean — the busiest communication hubs. Their failure disrupts both upstream and downstream flows."
          formula="pub > avg_pub AND sub > avg_sub"
        />
        <MetricInsightCard
          label="Consumer-Only Apps"
          value={data.summary.q_consumer ?? 0}
          description="Apps that subscribe above average but publish below average. Pure consumers: data enters but nothing is emitted. Upstream failures cascade directly into these endpoints."
          formula="pub ≤ avg_pub AND sub > avg_sub"
        />
        <MetricInsightCard
          label="Producer-Only Apps"
          value={data.summary.q_producer ?? 0}
          description="Apps that publish above average but subscribe below average. Data sources whose failure causes downstream data loss across all dependent subscribers."
          formula="pub > avg_pub AND sub ≤ avg_sub"
        />
      </div>
      <SummaryCards summary={data.summary} keys={[
        { key: "total_apps", label: "Total Apps" },
        { key: "q_high_io", label: "High I/O" },
        { key: "q_consumer", label: "Consumers" },
        { key: "q_producer", label: "Producers" },
        { key: "q_low", label: "Low Activity" },
        { key: "zero_activity", label: "Zero Activity" },
        { key: "outlier_count", label: "Outliers" },
      ]} />
      <Card>
        <CardHeader>
          <CardTitle className="text-base"><TermTooltip term="Pub/Sub Balance">Application Pub/Sub Balance</TermTooltip></CardTitle>
          <PaginationBar search={search} onSearch={handleSearch} page={page} totalPages={totalPages} onPage={setPage} totalItems={allItems.length} filteredItems={filtered.length} />
        </CardHeader>
        <CardContent>
          <SizedBarChart dataCount={pageItems.length} config={{
            publishes: { label: "Publishes", color: "#3b82f6" },
            subscribes: { label: "Subscribes", color: "#10b981" },
          }}>
            <BarChart data={pageItems.map((d) => ({ ...d, name: truncate(d.name) }))} margin={{ bottom: 50 }} maxBarSize={48} onClick={handleBarClick} className="cursor-pointer">
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={70} tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Bar dataKey="publishes" fill="#3b82f6" radius={[4, 4, 0, 0]} stackId="a" />
              <Bar dataKey="subscribes" fill="#10b981" radius={[4, 4, 0, 0]} stackId="a" />
            </BarChart>
          </SizedBarChart>
        </CardContent>
      </Card>
    </div>
  )
}

function TopicFanoutSection({ data }: { data: ExtrasStats["topic_fanout"] }) {
  const allItems = useMemo(() => (!data ? [] : data.labels.map((label, i) => ({
    name: label,
    id: data.ids?.[i],
    fanout: data.fanout[i],
  })).sort((a, b) => b.fanout - a.fanout)), [data])

  const { search, handleSearch, page, setPage, pageItems, totalPages, filtered } = usePaginatedSearch(allItems)

  if (!data) return null

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        <MetricInsightCard
          label="Max Fanout"
          value={data.summary.fanout_max ?? 0}
          description="Highest message multiplication factor in the system. One publish event on this topic is delivered to this many subscribers simultaneously. High fanout amplifies failure blast radius."
          formula="fanout = pub_count × sub_count"
        />
        <MetricInsightCard
          label="Orphan Topics"
          value={data.summary.orphan ?? 0}
          description="Topics missing a publisher or subscriber — incomplete message flows that either produce data nobody receives, or await data that never arrives."
          formula="count(pub_count = 0 OR sub_count = 0)"
        />
        <MetricInsightCard
          label="Broadcast Topics (1→N)"
          value={data.summary.one_to_many ?? 0}
          description="Topics with a single publisher and multiple subscribers. A publisher failure on any of these simultaneously silences all downstream consumers."
          formula="count(pub_count = 1 AND sub_count > 1)"
        />
      </div>
      <SummaryCards summary={data.summary} keys={[
        { key: "total_topics", label: "Total Topics" },
        { key: "one_to_many", label: "1→N" },
        { key: "many_to_one", label: "N→1" },
        { key: "many_to_many", label: "N→N" },
        { key: "orphan", label: "Orphan" },
        { key: "fanout_max", label: "Max Fanout" },
        { key: "outlier_count", label: "Outliers" },
      ]} />
      <Card>
        <CardHeader>
          <CardTitle className="text-base"><TermTooltip term="Topic Fanout">Topic Fanout (Publishers × Subscribers)</TermTooltip></CardTitle>
          <PaginationBar search={search} onSearch={handleSearch} page={page} totalPages={totalPages} onPage={setPage} totalItems={allItems.length} filteredItems={filtered.length} />
        </CardHeader>
        <CardContent>
          <SizedBarChart dataCount={pageItems.length} config={{ fanout: { label: "Fanout", color: "#f59e0b" } }}>
            <BarChart data={pageItems.map((d) => ({ ...d, name: truncate(d.name) }))} margin={{ bottom: 50 }} maxBarSize={48} onClick={handleBarClick} className="cursor-pointer">
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={70} tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Bar dataKey="fanout" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </SizedBarChart>
        </CardContent>
      </Card>
    </div>
  )
}

type HeatmapMode = "pub" | "sub" | "pubsub"

function transposeMatrix(m: number[][]): number[][] {
  if (!m.length) return m
  return m[0].map((_, ci) => m.map((row) => row[ci]))
}

function combineMatrices(m: number[][]): number[][] {
  const t = transposeMatrix(m)
  return m.map((row, ri) => row.map((val, ci) => val + t[ri][ci]))
}

function HeatmapSection({ data, title, modeToggle, insights }: {
  data: {
    labels: string[]
    node_ids?: string[]
    matrix: number[][]
    matrix_kb?: number[][]
    summary: SummaryDict
    outlier_pairs: [string, string, number, number][]
    per_node?: Record<string, { label: string; apps: { id: string; name: string }[]; pub_topics: { id: string; name: string; size_kb: number }[]; sub_topics: { id: string; name: string; size_kb: number }[] }>
  } | undefined
  title: string
  modeToggle?: boolean
  insights?: React.ReactNode
}) {
  const [mode, setMode] = useState<HeatmapMode>("pub")
  const [showKb, setShowKb] = useState(false)
  const [selectedCell, setSelectedCell] = useState<{ rowId: string; colId: string; rowLabel: string; colLabel: string } | null>(null)
  const [search, setSearch] = useState("")
  const [page, setPage] = useState(0)

  if (!data || !data.labels.length) return <p className="text-sm text-muted-foreground">Not enough data for {title}</p>

  const totalLabels = data.labels.length

  // Sort all indices by row total descending (busiest entities first)
  const rowTotals = data.matrix.map((row) => row.reduce((s, v) => s + v, 0))
  const sortedIndices = rowTotals.map((_, i) => i).sort((a, b) => rowTotals[b] - rowTotals[a])

  // Filter by search term
  const q = search.trim().toLowerCase()
  const filteredIndices = q
    ? sortedIndices.filter((i) => data.labels[i].toLowerCase().includes(q))
    : sortedIndices

  const totalPages = Math.max(1, Math.ceil(filteredIndices.length / MAX_ITEMS))
  const safePage = Math.min(page, totalPages - 1)
  const indices = filteredIndices.slice(safePage * MAX_ITEMS, (safePage + 1) * MAX_ITEMS)
  const idxSet = new Set(indices)

  let labels = indices.map((i) => data.labels[i])
  let baseMatrix = indices.map((ri) => data.matrix[ri].filter((_, ci) => idxSet.has(ci)))
  let ids = data.node_ids ? indices.map((i) => data.node_ids![i]) : undefined

  const countMatrix = !modeToggle || mode === "pub" ? baseMatrix
               : mode === "sub" ? transposeMatrix(baseMatrix)
               : combineMatrices(baseMatrix)

  const baseMatrixKb = (showKb && data.matrix_kb) ? indices.map((ri) => data.matrix_kb!![ri].filter((_, ci) => idxSet.has(ci))) : null
  const kbMatrix = baseMatrixKb
    ? (!modeToggle || mode === "pub" ? baseMatrixKb
       : mode === "sub" ? transposeMatrix(baseMatrixKb)
       : combineMatrices(baseMatrixKb))
    : null

  const matrix = kbMatrix ?? countMatrix

  const rowAxisLabel = !modeToggle || mode === "pub" ? "Publisher →" : mode === "sub" ? "Subscriber →" : "Node →"
  const colAxisLabel = !modeToggle || mode === "pub" ? "→ Subscriber" : mode === "sub" ? "→ Publisher" : "→ Node"

  const maxVal = Math.max(1, ...matrix.flat())
  const n = labels.length
  const cellSize = n > 15 ? "min-w-[28px] h-[28px] p-0.5 text-[10px]" : n > 10 ? "min-w-[32px] h-[32px] p-1 text-[11px]" : "min-w-[40px] h-[40px] p-2 text-xs"
  const headerSize = n > 15 ? "text-[9px] p-0.5" : n > 10 ? "text-[10px] p-1" : "text-xs p-1"

  function handleCellClick(ri: number, ci: number) {
    if (!ids || !modeToggle || !data?.per_node) return
    const rowId = ids[ri], colId = ids[ci]
    const rowLabel = data.per_node[rowId]?.label ?? rowId
    const colLabel = data.per_node[colId]?.label ?? colId
    setSelectedCell((prev) =>
      prev?.rowId === rowId && prev?.colId === colId ? null : { rowId, colId, rowLabel, colLabel }
    )
  }

  // Compute directed topic lists based on current mode
  // pub:    row=publisher → col=subscriber
  // sub:    row=subscriber ← col=publisher
  // pubsub: both directions
  let pubTopics: { id: string; name: string }[] = []   // row→col direction
  let subTopics: { id: string; name: string }[] = []   // col→row direction
  if (selectedCell && data.per_node) {
    const pn = data.per_node
    if (mode === "pub" || mode === "pubsub") {
      // topics rowId publishes that colId subscribes
      const colSubIds = new Set(pn[selectedCell.colId]?.sub_topics.map((t) => t.id) ?? [])
      pubTopics = (pn[selectedCell.rowId]?.pub_topics ?? []).filter((t) => colSubIds.has(t.id))
    }
    if (mode === "sub" || mode === "pubsub") {
      // topics colId publishes that rowId subscribes
      const rowSubIds = new Set(pn[selectedCell.rowId]?.sub_topics.map((t) => t.id) ?? [])
      subTopics = (pn[selectedCell.colId]?.pub_topics ?? []).filter((t) => rowSubIds.has(t.id))
    }
  }

  return (
    <div className="space-y-4">
      {insights}
      <SummaryCards summary={data.summary} keys={[
        { key: "entity_count", label: "Entities" },
        { key: "nonzero_count", label: "Active Cells" },
        { key: "active_pct", label: "Active %", format: (v) => Number(v).toFixed(1) + "%" },
        { key: "intra_total", label: "Intra-entity" },
        { key: "inter_total", label: "Inter-entity" },
        { key: "outlier_count", label: "Outliers" },
      ]} />
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <CardTitle className="text-base">{title}</CardTitle>
            {modeToggle && (
              <div className="flex items-center gap-2 flex-wrap">
                <div className="flex items-center gap-1 rounded-md border p-0.5 bg-muted/50">
                  {(["pub", "sub", "pubsub"] as HeatmapMode[]).map((m) => (
                    <button
                      key={m}
                      onClick={() => { setMode(m); setSelectedCell(null) }}
                      className={`px-2.5 py-0.5 text-xs rounded transition-colors ${
                        mode === m ? "bg-background shadow font-medium" : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      {m === "pub" ? "Publishers" : m === "sub" ? "Subscribers" : "Pub + Sub"}
                    </button>
                  ))}
                </div>
                {data.matrix_kb && (
                  <div className="flex items-center gap-1 rounded-md border p-0.5 bg-muted/50">
                    {[false, true].map((kb) => (
                      <button
                        key={String(kb)}
                        onClick={() => setShowKb(kb)}
                        className={`px-2.5 py-0.5 text-xs rounded transition-colors ${
                          showKb === kb ? "bg-background shadow font-medium" : "text-muted-foreground hover:text-foreground"
                        }`}
                      >
                        {kb ? "bytes" : "Count"}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
          {modeToggle && (
            <p className="text-xs text-muted-foreground mt-1">
              {mode === "pub" ? "Row = publishing node, column = subscribing node. Click a cell to inspect shared topics." :
               mode === "sub" ? "Row = subscribing node, column = publishing node. Click a cell to inspect shared topics." :
               "Symmetric view: both directions summed. Click a cell to inspect shared topics."}
            </p>
          )}
          <PaginationBar
            search={search}
            onSearch={(v) => { setSearch(v); setPage(0); setSelectedCell(null) }}
            page={safePage}
            totalPages={totalPages}
            onPage={(p) => { setPage(p); setSelectedCell(null) }}
            totalItems={totalLabels}
            filteredItems={filteredIndices.length}
          />
        </CardHeader>
        <CardContent>
          <div className="overflow-auto">
            <table className="border-collapse">
              <thead>
                <tr>
                  <th className="sticky left-0 bg-background z-10 p-2 text-[10px] text-muted-foreground text-right">{rowAxisLabel}</th>
                  {labels.map((l, i) => (
                    <th
                      key={i}
                      className={`align-bottom text-left ${headerSize} ${ids ? "cursor-pointer hover:underline" : ""}`}
                      style={{ writingMode: "vertical-lr" }}
                      onClick={ids ? () => goToExplorer(ids[i]) : undefined}
                    >{l}</th>
                  ))}
                </tr>
                <tr>
                  <th className="sticky left-0 bg-background z-10 p-1 text-[10px] text-muted-foreground text-right">{colAxisLabel}</th>
                </tr>
              </thead>
              <tbody>
                {labels.map((rowLabel, ri) => (
                  <tr key={ri}>
                    <td
                      className={`sticky left-0 bg-background z-10 font-medium whitespace-nowrap ${headerSize} ${ids ? "cursor-pointer hover:underline" : ""}`}
                      onClick={ids ? () => goToExplorer(ids[ri]) : undefined}
                    >{rowLabel}</td>
                    {matrix[ri]?.map((val, ci) => {
                      const intensity = val / maxVal
                      const isSelected = !!(selectedCell && ids && selectedCell.rowId === ids[ri] && selectedCell.colId === ids[ci])
                      return (
                        <td
                          key={ci}
                          className={`text-center border ${cellSize} transition-all ${
                            modeToggle && ids && val > 0
                              ? "cursor-pointer hover:ring-2 hover:ring-violet-400"
                              : ""
                          } ${isSelected ? "ring-2 ring-violet-500 z-10 relative" : "border-border"}`}
                          style={{ backgroundColor: val > 0 ? `rgba(139, 92, 246, ${0.15 + intensity * 0.75})` : undefined }}
                          title={`${rowLabel} → ${labels[ci]}: ${showKb ? val.toFixed(1) + " bytes" : val}`}
                          onClick={val > 0 ? () => handleCellClick(ri, ci) : undefined}
                        >
                          {val > 0 ? (showKb ? (val >= 1_048_576 ? (val / 1_048_576).toFixed(1) + "M" : val >= 1024 ? (val / 1024).toFixed(1) + "K" : val.toFixed(0)) : val) : ""}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {selectedCell && data.per_node && (
        <Card className="border-violet-500/40">
          <CardHeader className="pb-2">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base flex items-center gap-2">
                <span className="text-violet-600 dark:text-violet-400 font-semibold">{selectedCell.rowLabel}</span>
                <span className="text-muted-foreground text-sm">{mode === "pub" ? "→" : mode === "sub" ? "←" : "↔"}</span>
                <span className="text-violet-600 dark:text-violet-400 font-semibold">{selectedCell.colLabel}</span>
              </CardTitle>
              <button onClick={() => setSelectedCell(null)} className="text-xs text-muted-foreground hover:text-foreground transition-colors px-1">✕</button>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            {(mode === "pub" || mode === "pubsub") && (
              <div>
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">
                  {selectedCell.rowLabel} publishes → {selectedCell.colLabel} subscribes ({pubTopics.length} topics)
                </p>
                {pubTopics.length === 0
                  ? <p className="text-xs text-muted-foreground">No published topics shared in this direction.</p>
                  : <ul className="columns-2 sm:columns-3 gap-2 space-y-0.5">
                      {pubTopics.map((t) => (
                        <li key={t.id} className="text-xs font-mono text-green-700 dark:text-green-400 truncate break-inside-avoid cursor-pointer hover:underline flex items-baseline gap-1" title={t.name} onClick={() => goToExplorer(t.id)}>
                          <span className="truncate">{t.name}</span>
                          {showKb && <span className="text-[10px] opacity-60 shrink-0">{t.size_kb ?? 0} bytes</span>}
                        </li>
                      ))}
                    </ul>
                }
              </div>
            )}
            {(mode === "sub" || mode === "pubsub") && (
              <div>
                <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide mb-1">
                  {selectedCell.rowLabel} subscribes ← {selectedCell.colLabel} publishes ({subTopics.length} topics)
                </p>
                {subTopics.length === 0
                  ? <p className="text-xs text-muted-foreground">No subscribed topics shared in this direction.</p>
                  : <ul className="columns-2 sm:columns-3 gap-2 space-y-0.5">
                      {subTopics.map((t) => (
                        <li key={t.id} className="text-xs font-mono text-purple-700 dark:text-purple-400 truncate break-inside-avoid cursor-pointer hover:underline flex items-baseline gap-1" title={t.name} onClick={() => goToExplorer(t.id)}>
                          <span className="truncate">{t.name}</span>
                          {showKb && <span className="text-[10px] opacity-60 shrink-0">{t.size_kb ?? 0} bytes</span>}
                        </li>
                      ))}
                    </ul>
                }
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {data.outlier_pairs.length > 0 && (
        <Card>
          <CardHeader><CardTitle className="text-base">Outlier Pairs</CardTitle></CardHeader>
          <CardContent>
            <OutlierTable headers={["Source", "Target", "Count", "Deviation"]} rows={data.outlier_pairs} />
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function NodeCommLoadSection({ data }: { data: ExtrasStats["node_comm_load"] }) {
  const allItems = useMemo(() => (!data ? [] : data.sorted_labels.map((label, i) => ({
    name: label,
    id: data.sorted_ids?.[i],
    publishes: data.sorted_pub[i],
    subscribes: data.sorted_sub[i],
  }))), [data])

  const { search, handleSearch, page, setPage, pageItems, totalPages, filtered } = usePaginatedSearch(allItems)

  if (!data) return null

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        <MetricInsightCard
          label="Avg Node Load"
          value={data.summary.load_mean ?? 0}
          description="Average number of topic connections (publishes + subscribes) hosted on a single node. Represents typical communication pressure across all hosts."
          formula="load(node) = Σ pub_count + Σ sub_count for hosted apps"
        />
        <MetricInsightCard
          label="Load Variation (CV)"
          value={`${Number(data.summary.cv ?? 0).toFixed(1)}%`}
          description="Coefficient of variation of node load. High CV means uneven workload distribution — some nodes carry disproportionate traffic while others are nearly idle."
          formula="CV = std(load) / mean(load) × 100"
        />
        <MetricInsightCard
          label="Idle Nodes"
          value={data.summary.zero_load ?? 0}
          description="Nodes hosting no communicating applications. May indicate orphaned infrastructure, pure compute nodes, or deployment imbalances worth investigating."
          formula="count(nodes where load = 0)"
        />
      </div>
      <SummaryCards summary={data.summary} keys={[
        { key: "node_count", label: "Nodes" },
        { key: "pub_total", label: "Total Pub" },
        { key: "sub_total", label: "Total Sub" },
        { key: "load_mean", label: "Avg Load" },
        { key: "cv", label: "CV%", format: (v) => Number(v).toFixed(1) + "%" },
        { key: "zero_load", label: "Zero Load" },
        { key: "outlier_count", label: "Outliers" },
      ]} />
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Node Communication Load</CardTitle>
          <PaginationBar search={search} onSearch={handleSearch} page={page} totalPages={totalPages} onPage={setPage} totalItems={allItems.length} filteredItems={filtered.length} />
        </CardHeader>
        <CardContent>
          <SizedBarChart dataCount={pageItems.length} config={{
            publishes: { label: "Publishes", color: "#3b82f6" },
            subscribes: { label: "Subscribes", color: "#ec4899" },
          }}>
            <BarChart data={pageItems.map((d) => ({ ...d, name: truncate(d.name) }))} margin={{ bottom: 50 }} maxBarSize={48} onClick={handleBarClick} className="cursor-pointer">
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={70} tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Bar dataKey="publishes" fill="#3b82f6" radius={[4, 4, 0, 0]} stackId="a" />
              <Bar dataKey="subscribes" fill="#ec4899" radius={[4, 4, 0, 0]} stackId="a" />
            </BarChart>
          </SizedBarChart>
        </CardContent>
      </Card>
      {data.outliers.length > 0 && (
        <Card>
          <CardHeader><CardTitle className="text-base">Outlier Nodes</CardTitle></CardHeader>
          <CardContent>
            <OutlierTable headers={["Node", "Pub", "Sub", "Total", "Deviation"]} rows={data.outliers} />
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function CriticalityIOSection({ data }: { data: ExtrasStats["criticality_io"] }) {
  const allItems = useMemo(() => (!data ? [] : data.crit_labels.map((label, i) => ({
    name: label,
    id: data.crit_ids?.[i],
    publishes: data.crit_pubs[i],
    subscribes: data.crit_subs[i],
  })).sort((a, b) => (b.publishes + b.subscribes) - (a.publishes + a.subscribes))), [data])

  const { search, handleSearch, page, setPage, pageItems, totalPages, filtered } = usePaginatedSearch(allItems)

  if (!data) return null

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        <MetricInsightCard
          label="Critical App Fraction"
          value={`${Number(data.summary.crit_pct ?? 0).toFixed(1)}%`}
          description="Share of applications flagged as mission-critical. A higher fraction reduces the system's fault-tolerance margin — more components failing means wider impact."
          formula="critical_apps / total_apps × 100"
        />
        <MetricInsightCard
          label="Critical vs Normal I/O"
          value={Number(data.summary.crit_norm_ratio ?? 0).toFixed(2)}
          unit="×"
          description="How much heavier critical apps' average I/O load is compared to normal apps. Values above 1× mean critical components are also the communication hotspots."
          formula="mean_io(critical) / mean_io(normal)"
        />
        <MetricInsightCard
          label="Critical Avg I/O"
          value={data.summary.crit_io_mean ?? 0}
          description="Average number of pub/sub connections per critical application. High values compound failure impact — more dependencies are at risk when a critical component goes down."
          formula="mean(pub + sub) for critical apps"
        />
      </div>
      <SummaryCards summary={data.summary} keys={[
        { key: "total_apps", label: "Total Apps" },
        { key: "crit_count", label: "Critical" },
        { key: "crit_pct", label: "Critical %", format: (v) => Number(v).toFixed(1) + "%" },
        { key: "crit_io_mean", label: "Crit Avg I/O" },
        { key: "norm_io_mean", label: "Normal Avg I/O" },
        { key: "crit_norm_ratio", label: "Crit/Normal Ratio" },
      ]} />
      {allItems.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Critical Applications I/O</CardTitle>
            <PaginationBar search={search} onSearch={handleSearch} page={page} totalPages={totalPages} onPage={setPage} totalItems={allItems.length} filteredItems={filtered.length} />
          </CardHeader>
          <CardContent>
            <SizedBarChart dataCount={pageItems.length} config={{
              publishes: { label: "Publishes", color: "#ef4444" },
              subscribes: { label: "Subscribes", color: "#f97316" },
            }}>
              <BarChart data={pageItems.map((d) => ({ ...d, name: truncate(d.name) }))} margin={{ bottom: 50 }} maxBarSize={48} onClick={handleBarClick} className="cursor-pointer">
                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                <XAxis dataKey="name" angle={-45} textAnchor="end" height={70} tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Bar dataKey="publishes" fill="#ef4444" radius={[4, 4, 0, 0]} stackId="a" />
                <Bar dataKey="subscribes" fill="#f97316" radius={[4, 4, 0, 0]} stackId="a" />
              </BarChart>
            </SizedBarChart>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function LibDependencySection({ data }: { data: ExtrasStats["lib_dependency"] }) {
  const allItems = useMemo(() => (!data ? [] : data.labels.map((label, i) => ({
    name: label,
    id: data.display_ids?.[i],
    inbound: data.in_vals[i],
    outbound: data.out_vals[i],
  }))), [data])

  const { search, handleSearch, page, setPage, pageItems, totalPages, filtered } = usePaginatedSearch(allItems)

  if (!data || !data.labels.length) return <p className="text-sm text-muted-foreground">No library dependency data</p>

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        <MetricInsightCard
          label="Max Library In-Degree"
          value={data.summary.in_max ?? 0}
          description="The highest number of applications depending on a single library. This library is a shared-fate risk — its failure or API change affects all dependents simultaneously."
          formula="max(count(apps depending on library))"
        />
        <MetricInsightCard
          label="Avg In-Degree"
          value={Number(data.summary.in_mean ?? 0).toFixed(1)}
          description="Mean number of applications consuming each library. Higher averages indicate broadly shared libraries with larger collective blast radius."
          formula="mean(in-degree) across active libraries"
        />
        <MetricInsightCard
          label="Total Dependencies"
          value={data.summary.total_relations ?? 0}
          description="Total application-to-library USES edges in the system. Measures overall coupling density between the application and library layers."
          formula="count(USES relationships)"
        />
      </div>
      <SummaryCards summary={data.summary} keys={[
        { key: "total_relations", label: "Total Relations" },
        { key: "active_count", label: "Active Entities" },
        { key: "app_count", label: "Apps" },
        { key: "lib_count", label: "Libraries" },
        { key: "in_mean", label: "Avg In-degree" },
        { key: "in_max", label: "Max In-degree" },
        { key: "outlier_count", label: "Outliers" },
      ]} />
      <Card>
        <CardHeader>
          <CardTitle className="text-base"><TermTooltip term="Library Dependency Density">Library Dependency Density</TermTooltip></CardTitle>
          <PaginationBar search={search} onSearch={handleSearch} page={page} totalPages={totalPages} onPage={setPage} totalItems={allItems.length} filteredItems={filtered.length} />
        </CardHeader>
        <CardContent>
          <SizedBarChart dataCount={pageItems.length} config={{
            inbound: { label: "In-degree (dependents)", color: "#8b5cf6" },
            outbound: { label: "Out-degree (dependencies)", color: "#14b8a6" },
          }}>
            <BarChart data={pageItems.map((d) => ({ ...d, name: truncate(d.name, 20) }))} margin={{ bottom: 50 }} maxBarSize={48} onClick={handleBarClick} className="cursor-pointer">
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={70} tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Bar dataKey="inbound" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              <Bar dataKey="outbound" fill="#14b8a6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </SizedBarChart>
        </CardContent>
      </Card>
      {data.outliers.length > 0 && (
        <Card>
          <CardHeader><CardTitle className="text-base">High-Dependency Outliers</CardTitle></CardHeader>
          <CardContent>
            <OutlierTable headers={["Entity", "In-degree", "Out-degree"]} rows={data.outliers} />
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function NodeCriticalDensitySection({ data }: { data: ExtrasStats["node_critical_density"] }) {
  const allItems = useMemo(() => (!data ? [] : data.sorted_labels.map((label, i) => ({
    name: label,
    id: data.sorted_ids?.[i],
    critical: data.sorted_crit[i],
    normal: data.sorted_norm[i],
  }))), [data])

  const { search, handleSearch, page, setPage, pageItems, totalPages, filtered } = usePaginatedSearch(allItems)

  if (!data) return null

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        <MetricInsightCard
          label="System Critical %"
          value={`${Number(data.summary.system_crit_pct ?? 0).toFixed(1)}%`}
          description="Percentage of all applications across the system that are marked critical. High values reduce redundancy headroom and make the system more vulnerable to targeted failures."
          formula="critical_apps / total_apps × 100"
        />
        <MetricInsightCard
          label="Max Critical Apps / Node"
          value={data.summary.crit_per_node_max ?? 0}
          description="Highest concentration of critical apps on a single physical node. A node with many critical apps is a blast-radius hotspot — losing it collapses multiple mission-critical flows."
          formula="max(critical_count per node)"
        />
        <MetricInsightCard
          label="Nodes Without Critical Apps"
          value={data.summary.zero_crit ?? 0}
          description="Physical nodes hosting no critical applications. These nodes have inherently lower individual failure impact on mission-critical system behaviour."
          formula="count(nodes where critical_count = 0)"
        />
      </div>
      <SummaryCards summary={data.summary} keys={[
        { key: "node_count", label: "Nodes" },
        { key: "total_crit", label: "Total Critical" },
        { key: "total_norm", label: "Total Normal" },
        { key: "system_crit_pct", label: "System Critical %", format: (v) => Number(v).toFixed(1) + "%" },
        { key: "crit_per_node_max", label: "Max Crit/Node" },
        { key: "zero_crit", label: "No Critical" },
      ]} />
      <Card>
        <CardHeader>
          <CardTitle className="text-base"><TermTooltip term="Node Critical Density">Node Critical Application Density</TermTooltip></CardTitle>
          <PaginationBar search={search} onSearch={handleSearch} page={page} totalPages={totalPages} onPage={setPage} totalItems={allItems.length} filteredItems={filtered.length} />
        </CardHeader>
        <CardContent>
          <SizedBarChart dataCount={pageItems.length} config={{
            critical: { label: "Critical", color: "#ef4444" },
            normal: { label: "Normal", color: "#3b82f6" },
          }}>
            <BarChart data={pageItems.map((d) => ({ ...d, name: truncate(d.name) }))} margin={{ bottom: 50 }} maxBarSize={48} onClick={handleBarClick} className="cursor-pointer">
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={70} tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Bar dataKey="critical" fill="#ef4444" radius={[4, 4, 0, 0]} stackId="a" />
              <Bar dataKey="normal" fill="#3b82f6" radius={[4, 4, 0, 0]} stackId="a" />
            </BarChart>
          </SizedBarChart>
        </CardContent>
      </Card>
    </div>
  )
}

function DomainDiversitySection({ data }: { data: ExtrasStats["domain_diversity"] }) {
  const allItems = useMemo(() => (!data ? [] : data.labels.map((label, i) => ({
    name: label,
    applications: data.app_counts[i],
    topics: data.topic_counts[i],
    io: data.io_vals[i],
  }))), [data])

  const { search, handleSearch, page, setPage, pageItems, totalPages, filtered } = usePaginatedSearch(allItems)

  if (!data || !data.labels.length) return <p className="text-sm text-muted-foreground">Insufficient domain data (need ≥ 2 domains)</p>

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        <MetricInsightCard
          label="Avg Apps per Domain"
          value={Number(data.summary.app_mean ?? 0).toFixed(1)}
          description="Mean number of applications per domain. Very low values may indicate overly fragmented subsystems; very high values may signal monolithic domains with unclear boundaries."
          formula="total_apps / domain_count"
        />
        <MetricInsightCard
          label="Avg Topics per Domain"
          value={Number(data.summary.topic_mean ?? 0).toFixed(1)}
          description="Mean number of topics owned or used per domain. Reflects how much communication surface each subsystem exposes to the rest of the architecture."
          formula="total_topics_touched / domain_count"
        />
        <MetricInsightCard
          label="Avg I/O per Domain"
          value={Number(data.summary.io_mean ?? 0).toFixed(1)}
          description="Average pub/sub message load aggregated per domain. High I/O domains are communication hubs whose degradation has the widest downstream reach."
          formula="mean(Σ pub + Σ sub per domain)"
        />
      </div>
      <SummaryCards summary={data.summary} keys={[
        { key: "css_count", label: "Domains" },
        { key: "app_mean", label: "Avg Apps/Domain" },
        { key: "app_max", label: "Max Apps" },
        { key: "topic_mean", label: "Avg Topics/Domain" },
        { key: "io_mean", label: "Avg I/O" },
        { key: "io_max", label: "Max I/O" },
      ]} />
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Domain Diversity</CardTitle>
          <PaginationBar search={search} onSearch={handleSearch} page={page} totalPages={totalPages} onPage={setPage} totalItems={allItems.length} filteredItems={filtered.length} />
        </CardHeader>
        <CardContent>
          <SizedBarChart dataCount={pageItems.length} config={{
            applications: { label: "Applications", color: "#3b82f6" },
            topics: { label: "Topics", color: "#10b981" },
            io: { label: "I/O Load", color: "#f59e0b" },
          }}>
            <BarChart data={pageItems.map((d) => ({ ...d, name: truncate(d.name) }))} margin={{ bottom: 50 }} maxBarSize={48}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
              <XAxis dataKey="name" angle={-45} textAnchor="end" height={70} tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 11 }} />
              <ChartTooltip content={<ChartTooltipContent />} />
              <Bar dataKey="applications" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              <Bar dataKey="topics" fill="#10b981" radius={[4, 4, 0, 0]} />
              <Bar dataKey="io" fill="#f59e0b" radius={[4, 4, 0, 0]} />
            </BarChart>
          </SizedBarChart>
        </CardContent>
      </Card>
    </div>
  )
}

// ── Tab config ──────────────────────────────────────────────────────────

const TAB_CONFIG = [
  { id: "topic_bandwidth", label: "Topic Bandwidth", icon: Radio, color: "text-violet-500", description: "Data throughput per topic based on message size and connection count. Switch between subscriber-side, publisher-side, or combined (pub + sub) bandwidth. High-bandwidth topics are potential bottlenecks." },
  { id: "app_balance", label: "App Balance", icon: Activity, color: "text-blue-500", description: "Publish/subscribe distribution across applications. Imbalances indicate uneven communication load or single-role components." },
  { id: "topic_fanout", label: "Topic Fanout", icon: Network, color: "text-amber-500", description: "Publisher-to-subscriber ratios per topic. Identifies 1-to-N broadcast, N-to-1 aggregation, and orphan patterns." },
  { id: "cross_node", label: "Cross-Node", icon: Server, color: "text-purple-500", description: "Inter-node communication intensity matrix. Reveals physical host coupling and network traffic hotspots." },
  { id: "node_load", label: "Node Load", icon: BarChart3, color: "text-pink-500", description: "Aggregate communication load per node. Highlights overloaded hosts and uneven workload distribution." },
  { id: "domain_comm", label: "Domain Comm", icon: Layers, color: "text-cyan-500", description: "Communication flows between domains. High cross-domain traffic signals tight coupling between subsystems." },
  { id: "criticality", label: "Criticality I/O", icon: AlertTriangle, color: "text-red-500", description: "I/O comparison of critical vs. normal applications. Shows whether critical components are also communication-heavy." },
  { id: "lib_deps", label: "Library Deps", icon: BookOpen, color: "text-teal-500", description: "Library dependency density and coupling. Libraries with high in-degree are shared-fate risks across multiple consumers." },
  { id: "node_density", label: "Node Density", icon: Shield, color: "text-green-500", description: "Distribution of critical applications across physical nodes. Concentration of critical apps on few nodes increases blast radius." },
  { id: "domain_div", label: "Domain Diversity", icon: Box, color: "text-orange-500", description: "Application and topic variety within each domain. Low diversity may indicate monolithic subsystems." },
] as const

type TabId = typeof TAB_CONFIG[number]["id"]

// Maps tab id → backend chart_id for the /chart/{chart_id} endpoint
const TAB_TO_CHART_ID: Record<TabId, keyof ExtrasStats> = {
  topic_bandwidth: "topic_bandwidth",
  app_balance: "app_balance",
  topic_fanout: "topic_fanout",
  cross_node: "cross_node_heatmap",
  node_load: "node_comm_load",
  domain_comm: "domain_comm",
  criticality: "criticality_io",
  lib_deps: "lib_dependency",
  node_density: "node_critical_density",
  domain_div: "domain_diversity",
}

// ── Main page ───────────────────────────────────────────────────────────

export default function StatisticsPage() {
  const { status, config, initialLoadComplete } = useConnection()
  const [tabData, setTabData] = useState<Partial<ExtrasStats>>({})
  const [tabLoading, setTabLoading] = useState<Partial<Record<TabId, boolean>>>({})
  const [tabError, setTabError] = useState<Partial<Record<TabId, string>>>({})
  const [activeTab, setActiveTab] = useState<TabId>("topic_bandwidth")

  const isConnected = status === "connected"

  const fetchTab = async (tabId: TabId, force = false) => {
    const chartKey = TAB_TO_CHART_ID[tabId]
    if (!force && tabData[chartKey] !== undefined) return
    if (!isConnected || !config) return

    setTabLoading((prev) => ({ ...prev, [tabId]: true }))
    setTabError((prev) => ({ ...prev, [tabId]: undefined }))
    try {
      const creds = apiClient.getCredentials()
      if (!creds) throw new Error("No credentials")
      const response = await fetch(`${API_BASE_URL}/api/v1/stats/chart/${chartKey}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(creds),
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const json = await response.json()
      if (json.success) {
        setTabData((prev) => ({ ...prev, [chartKey]: json.data }))
      } else {
        throw new Error(json.detail || "Failed")
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Failed to load statistics"
      setTabError((prev) => ({ ...prev, [tabId]: msg }))
    } finally {
      setTabLoading((prev) => ({ ...prev, [tabId]: false }))
    }
  }

  const handleTabChange = (value: string) => {
    const id = value as TabId
    setActiveTab(id)
    fetchTab(id)
  }

  const handleRefresh = () => {
    setTabData((prev) => {
      const chartKey = TAB_TO_CHART_ID[activeTab]
      const next = { ...prev }
      delete next[chartKey]
      return next
    })
    fetchTab(activeTab, true)
  }

  useEffect(() => {
    if (isConnected && config) {
      fetchTab("topic_bandwidth")
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isConnected, config])

  if (!initialLoadComplete || status === "connecting") {
    return (
      <AppLayout title="Statistics" description="Cross-cutting system statistics">
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" text={status === "connecting" ? "Connecting…" : "Loading…"} />
        </div>
      </AppLayout>
    )
  }

  return (
    <AppLayout title="Statistics" description="Cross-cutting system statistics">
      <div className="space-y-6">
        {!isConnected && <NoConnectionInfo />}

        {isConnected && (
          <>
            {/* Header */}
            <Card className="relative overflow-hidden border-0 shadow-xl">
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500">
                <div className="w-full h-full rounded-lg bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600" />
              </div>
              <CardContent className="p-6 relative text-white">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-2xl font-bold mb-1">System Statistics</h3>
                    <p className="text-white/85 text-sm">
                      Cross-cutting analysis: communication patterns, topic flow, node load, domain structure, and criticality distribution
                    </p>
                  </div>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={handleRefresh}
                    disabled={!!tabLoading[activeTab]}
                    className="shrink-0"
                  >
                    <RefreshCw className={`h-4 w-4 mr-1 ${tabLoading[activeTab] ? "animate-spin" : ""}`} />
                    Refresh
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Tabs value={activeTab} onValueChange={handleTabChange} className="w-full">
              <TabsList className="flex flex-wrap h-auto gap-1 bg-muted/50 p-1">
                {TAB_CONFIG.map(({ id, label, icon: Icon, color, description }) => (
                  <TabsTrigger key={id} value={id} className="text-xs gap-1 data-[state=active]:shadow-sm">
                    <Icon className={`h-3.5 w-3.5 ${color}`} />
                    <TermTooltip description={description}>{label}</TermTooltip>
                  </TabsTrigger>
                ))}
              </TabsList>

              {TAB_CONFIG.map(({ id, description }) => {
                const isLoading = !!tabLoading[id]
                const error = tabError[id]
                const chartKey = TAB_TO_CHART_ID[id]
                const data = tabData[chartKey] as ExtrasStats[typeof chartKey] | undefined
                return (
                  <TabsContent key={id} value={id} className="mt-4">
                    <p className="text-sm text-muted-foreground mb-4">{description}</p>
                    {isLoading && (
                      <div className="flex h-64 items-center justify-center">
                        <LoadingSpinner size="lg" text="Computing statistics…" />
                      </div>
                    )}
                    {!isLoading && error && (
                      <Card className="border-red-500/50 bg-red-500/5">
                        <CardContent className="p-4">
                          <p className="text-sm text-red-500">
                            <AlertTriangle className="inline h-4 w-4 mr-1" /> {error}
                          </p>
                        </CardContent>
                      </Card>
                    )}
                    {!isLoading && !error && data !== undefined && (
                      <>
                        {id === "topic_bandwidth" && <TopicBandwidthSection data={tabData.topic_bandwidth} />}
                        {id === "app_balance" && <AppBalanceSection data={tabData.app_balance} />}
                        {id === "topic_fanout" && <TopicFanoutSection data={tabData.topic_fanout} />}
                        {id === "cross_node" && <HeatmapSection data={tabData.cross_node_heatmap} title="Cross-Node Communication Heatmap" modeToggle insights={
                          tabData.cross_node_heatmap && (
                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                              <MetricInsightCard
                                label="Active Cell %"
                                value={`${Number(tabData.cross_node_heatmap.summary.active_pct ?? 0).toFixed(1)}%`}
                                description="Fraction of node pairs that exchange at least one topic. High density indicates tightly coupled physical infrastructure where many nodes share data paths."
                                formula="nonzero_cells / total_cells × 100"
                              />
                              <MetricInsightCard
                                label="Intra-Node Events"
                                value={tabData.cross_node_heatmap.summary.intra_total ?? 0}
                                description="Topics where both publisher and subscriber run on the same physical node. Local traffic is unaffected by inter-node network failures."
                                formula="Σ matrix[i][i]  (diagonal sum)"
                              />
                              <MetricInsightCard
                                label="Cross-Node Events"
                                value={tabData.cross_node_heatmap.summary.inter_total ?? 0}
                                description="Topics that cross node boundaries. High counts raise network dependency risk and increase the blast radius of any single node or link failure."
                                formula="Σ matrix[i][j]  for i ≠ j"
                              />
                            </div>
                          )
                        } />}
                        {id === "node_load" && <NodeCommLoadSection data={tabData.node_comm_load} />}
                        {id === "domain_comm" && <HeatmapSection data={tabData.domain_comm} title="Domain-to-Domain Communication" insights={
                          tabData.domain_comm && (
                            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                              <MetricInsightCard
                                label="Active Cell %"
                                value={`${Number(tabData.domain_comm.summary.active_pct ?? 0).toFixed(1)}%`}
                                description="Fraction of domain pairs with cross-domain message traffic. High values signal tight coupling between subsystems and potential dependency debt."
                                formula="nonzero_cells / total_cells × 100"
                              />
                              <MetricInsightCard
                                label="Intra-Domain Events"
                                value={tabData.domain_comm.summary.intra_total ?? 0}
                                description="Topic exchanges where publisher and subscriber belong to the same domain. High intra-domain traffic signals well-encapsulated, loosely coupled subsystems."
                                formula="Σ matrix[i][i]  (diagonal sum)"
                              />
                              <MetricInsightCard
                                label="Cross-Domain Events"
                                value={tabData.domain_comm.summary.inter_total ?? 0}
                                description="Topic flows that cross domain boundaries. High values indicate interdependency between subsystems and wider cascade paths under failure."
                                formula="Σ matrix[i][j]  for i ≠ j"
                              />
                            </div>
                          )
                        } />}
                        {id === "criticality" && <CriticalityIOSection data={tabData.criticality_io} />}
                        {id === "lib_deps" && <LibDependencySection data={tabData.lib_dependency} />}
                        {id === "node_density" && <NodeCriticalDensitySection data={tabData.node_critical_density} />}
                        {id === "domain_div" && <DomainDiversitySection data={tabData.domain_diversity} />}
                      </>
                    )}
                  </TabsContent>
                )
              })}
            </Tabs>
          </>
        )}
      </div>
    </AppLayout>
  )
}
