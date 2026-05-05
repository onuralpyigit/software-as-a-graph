"use client"

import { useEffect, useState, useMemo } from "react"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { Skeleton } from "@/components/ui/skeleton"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import { useConnection } from "@/lib/stores/connection-store"
import { API_BASE_URL } from "@/lib/config/api"
import { apiClient } from "@/lib/api/client"
import ReactECharts from "echarts-for-react"
import {
  Activity, AlertTriangle, BarChart3, Layers, Network, Radio,
  Shield, Box, Server, BookOpen, Zap, ChevronLeft,
} from "lucide-react"
import { TermTooltip } from "@/components/ui/term-tooltip"
import { ItemTooltip } from "@/components/ui/item-tooltip"
import { useTheme } from "next-themes"

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
    segment_ids?: string[]
    per_segment?: Record<string, { label: string; pub_topics: { id: string; name: string }[]; sub_topics: { id: string; name: string }[] }>
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
  bottleneck?: {
    items: {
      id: string
      name: string
      type: string
      bottleneck_score: number
      betweenness: number
      ap_c_directed: number
      blast_radius: number
      blast_radius_norm: number
      bridge_ratio: number
      is_articulation_point: boolean
      is_directed_ap: boolean
      cascade_depth: number
      pubsub_betweenness: number
      weight: number
    }[]
    outlier_indices: number[]
    summary: SummaryDict
  }
}

// ── Helpers ─────────────────────────────────────────────────────────────

function getPrimaryLength(data: unknown): number {
  if (!data || typeof data !== "object") return 0
  const d = data as Record<string, unknown>
  if (Array.isArray(d.labels)) return d.labels.length
  if (Array.isArray(d.sorted_labels)) return d.sorted_labels.length
  if (Array.isArray(d.crit_labels)) return d.crit_labels.length
  if (Array.isArray(d.items)) return d.items.length
  return 0
}

function EmptyDataState() {
  return (
    <div className="flex flex-col items-center justify-center py-16 gap-3 text-center">
      <BarChart3 className="h-7 w-7 text-muted-foreground" />
      <p className="text-sm font-semibold text-foreground">No data available</p>
      <p className="text-sm text-muted-foreground">Import a graph to populate this section.</p>
    </div>
  )
}

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

// ── Chart helpers ────────────────────────────────────────────────────────────

/**
 * Module-level node map, populated once when the statistics page mounts.
 * Keyed by node id → { type, properties }. Used by chart tooltips.
 */
const _nodeById = new Map<string, { type: string; properties: Record<string, unknown> }>()

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 ** 2).toFixed(2)} MB`
}

interface EBarSeries {
  key: string
  label: string
  color: string
  stack?: string
  fmt?: (v: number) => string
}

function EBarChart({
  items,
  series,
  onClickId,
  yDomain,
}: {
  items: Array<Record<string, unknown> & { name: string; id?: string }>
  series: EBarSeries[]
  onClickId?: (id: string | undefined) => void
  yDomain?: [number, number]
}) {
  const { resolvedTheme } = useTheme()
  const isDark = resolvedTheme !== "light"

  // Theme-aware color tokens
  const axisColor       = isDark ? "#94a3b8" : "#64748b"
  const gridColor       = isDark ? "rgba(148,163,184,0.15)" : "rgba(100,116,139,0.15)"
  const tooltipBg       = isDark ? "#1e293b" : "#ffffff"
  const tooltipBorder   = isDark ? "rgba(148,163,184,0.2)" : "rgba(100,116,139,0.25)"
  const tooltipText     = isDark ? "#e2e8f0" : "#1e293b"
  const tooltipMuted    = isDark ? "rgba(226,232,240,0.65)" : "rgba(71,85,105,0.8)"
  const zoomBg          = isDark ? "rgba(128,128,128,0.08)" : "rgba(0,0,0,0.03)"
  const zoomDataLine    = isDark ? "rgba(128,128,128,0.3)" : "rgba(100,116,139,0.25)"
  const zoomDataArea    = isDark ? "rgba(128,128,128,0.05)" : "rgba(100,116,139,0.05)"

  const showZoom = items.length > 20
  const zoomEnd = showZoom ? Math.round(20 / items.length * 100) : 100
  const labels = items.map(it => truncate(it.name, 18))
  const option = {
    tooltip: {
      trigger: "axis" as const,
      axisPointer: { type: "shadow" as const },
      backgroundColor: tooltipBg,
      borderColor: tooltipBorder,
      textStyle: { color: tooltipText },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      formatter: (params: any) => {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const ps = params as Array<{ dataIndex: number; seriesName: string; value: number; color: string }>
        if (!ps?.length) return ""
        const idx = ps[0].dataIndex
        const item = items[idx]
        const id = item.id as string | undefined
        const nodeEntry = id ? _nodeById.get(id) : undefined
        const type = nodeEntry?.type ?? ""
        const fullName = item.name
        let extra = ""
        if (nodeEntry?.properties) {
          const props = nodeEntry.properties
          const get = (k: string) => props[k]
          if (type === "Application") {
            const role = get("role"); if (role != null && role !== "") extra += `<br/><span style="color:${tooltipMuted}">Role: ${role}</span>`
          } else if (type === "Topic") {
            const qr = get("qos_reliability"); if (qr != null && qr !== "") extra += `<br/><span style="color:${tooltipMuted}">Reliability: ${qr}</span>`
            const qd = get("qos_durability");  if (qd != null && qd !== "") extra += `<br/><span style="color:${tooltipMuted}">Durability: ${qd}</span>`
            const qt = get("qos_transport_priority"); if (qt != null && qt !== "") extra += `<br/><span style="color:${tooltipMuted}">Transport Priority: ${qt}</span>`
            const szRaw = get("message_size") ?? get("payload_size_bytes") ?? get("size")
            if (szRaw != null && szRaw !== "") { const szN = typeof szRaw === "number" ? szRaw : Number(szRaw); extra += `<br/><span style="color:${tooltipMuted}">Size: ${isFinite(szN) ? fmtBytes(szN) : String(szRaw)}</span>` }
          } else if (type === "Library") {
            const ver = get("version"); if (ver != null && ver !== "") extra += `<br/><span style="color:${tooltipMuted}">Version: ${ver}</span>`
          } else if (type === "Broker") {
            const bt = get("broker_type"); if (bt != null && bt !== "") extra += `<br/><span style="color:${tooltipMuted}">Protocol: ${bt}</span>`
          } else if (type === "Node") {
            const ip = get("ip_address"); if (ip != null && ip !== "") extra += `<br/><span style="color:${tooltipMuted}">IP: ${ip}</span>`
          }
        }
        const serRows = ps.map(p => {
          const ser = series.find(s => s.label === p.seriesName)
          const val = ser?.fmt ? ser.fmt(p.value) : (p.value?.toLocaleString(undefined, { maximumFractionDigits: 3 }) ?? "—")
          return (
            `<div style="display:flex;justify-content:space-between;gap:12px;margin:2px 0">` +
            `<div style="display:flex;align-items:center;gap:6px">` +
            `<span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${p.color}"></span>` +
            `<span style="color:${tooltipMuted}">${p.seriesName}</span></div>` +
            `<span style="font-family:monospace;font-weight:600;color:${tooltipText}">${val}</span></div>`
          )
        }).join("")
        const typeStr = type ? `<br/><span style="color:${tooltipMuted}">${type}</span>` : ""
        return (
          `<div style="font-size:12px;line-height:1.7;min-width:160px;max-width:240px;color:${tooltipText}">` +
          `<b>${fullName}</b>${typeStr}${extra}` +
          (serRows ? `<hr style="margin:4px 0;border-color:rgba(128,128,128,0.3)"/>${serRows}` : "") +
          `</div>`
        )
      },
    },
    grid: { left: 80, right: 10, top: 10, bottom: showZoom ? 80 : 80 },
    xAxis: {
      type: "category" as const,
      data: labels,
      axisLabel: { rotate: -45, fontSize: 10, interval: 0, color: axisColor },
    },
    yAxis: {
      type: "value" as const,
      axisLabel: { fontSize: 11, color: axisColor },
      splitLine: { lineStyle: { color: gridColor } },
      ...(yDomain ? { min: yDomain[0], max: yDomain[1] } : {}),
    },
    series: series.map(s => {
      const isLastInStack = !s.stack || series.filter(x => x.stack === s.stack).at(-1) === s
      return {
        name: s.label,
        type: "bar" as const,
        stack: s.stack,
        data: items.map(it => it[s.key] as number),
        itemStyle: { color: s.color, borderRadius: isLastInStack ? [4, 4, 0, 0] : [0, 0, 0, 0] },
        cursor: onClickId ? "pointer" : "default",
      }
    }),
    ...(showZoom ? {
      dataZoom: [
        {
          type: "slider",
          xAxisIndex: 0,
          start: 0,
          end: zoomEnd,
          height: 24,
          bottom: 8,
          borderColor: "transparent",
          backgroundColor: zoomBg,
          fillerColor: "rgba(99,102,241,0.18)",
          handleStyle: { color: "#818cf8", borderColor: "#818cf8" },
          moveHandleStyle: { color: "#818cf8" },
          selectedDataBackground: { lineStyle: { color: "#818cf8" }, areaStyle: { color: "#818cf8" } },
          dataBackground: { lineStyle: { color: zoomDataLine }, areaStyle: { color: zoomDataArea } },
          textStyle: { color: axisColor, fontSize: 10 },
          brushSelect: false,
        },
        { type: "inside", xAxisIndex: 0, start: 0, end: zoomEnd },
      ],
    } : {}),
  }
  const events = onClickId
    ? { click: (params: { dataIndex: number }) => onClickId((items[params.dataIndex]?.id) as string | undefined) }
    : {}
  return (
    <ReactECharts
      option={option}
      notMerge={true}
      style={{ height: "350px", width: "100%" }}
      onEvents={events}
      opts={{ renderer: "canvas" }}
    />
  )
}

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

function useFilteredSearch<T extends { name: string }>(items: T[]) {
  const [search, setSearch] = useState("")
  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase()
    return q ? items.filter((it) => it.name.toLowerCase().includes(q)) : items
  }, [items, search])
  const handleSearch = (v: string) => setSearch(v)
  return { search, handleSearch, filtered }
}

function ChartSearchBar({ search, onSearch, count, total }: {
  search: string; onSearch: (v: string) => void; count: number; total: number
}) {
  return (
    <div className="flex items-center gap-2">
      <input
        type="search"
        placeholder="Search…"
        value={search}
        onChange={(e) => onSearch(e.target.value)}
        className="h-7 w-36 rounded-md border bg-background px-2 text-xs focus:outline-none focus:ring-1 focus:ring-ring"
      />
      <span className="text-xs text-muted-foreground">
        {search ? `${count} of ${total}` : `${total} items`}
      </span>
    </div>
  )
}

function goToExplorer(id: string | undefined) {
  if (id) window.open(`/explorer?node=${encodeURIComponent(id)}`, "_blank")
}

function SummaryCards({ summary, keys }: { summary: SummaryDict; keys: { key: string; label: string; format?: (v: number | string) => string }[] }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {keys.map(({ key, label, format }) => {
        const val = summary[key]
        if (val === undefined) return null
        return (
          <div key={key} className="rounded-lg border bg-background p-3">
            <p className="text-xs text-muted-foreground">{label}</p>
            <p className="text-lg font-semibold">{format ? format(val) : fmtNum(val)}</p>
          </div>
        )
      })}
    </div>
  )
}

function OutlierTable({ rows, headers, title }: { rows: (string | number)[][]; headers: string[]; title?: string }) {
  if (!rows || rows.length === 0) return null
  return (
    <div>
      {title && <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">{title}</p>}
      <div className="max-h-64 overflow-auto rounded-lg border border-border">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border">
              {headers.map((h, i) => (
                <th key={h} className={`px-3 py-2 text-xs font-medium text-muted-foreground uppercase tracking-wide ${i === 0 ? "text-left" : "text-right"}`}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, MAX_ITEMS).map((row, i) => (
              <tr key={i} className="border-b border-border/50 hover:bg-muted/30 transition-colors">
                {row.map((cell, j) => (
                  <td key={j} className={`px-3 py-2 ${j === 0 ? "text-left font-medium" : "text-right font-mono text-xs text-muted-foreground"}`}>
                    {typeof cell === "number" ? fmtNum(cell) : cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
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
    <div className="rounded-lg border bg-background p-4 flex flex-col gap-1.5">
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
  sub:    { label: "Sub",  multiplierLabel: "Size × Subscribers",  avgKey: "sub_mean",  avgLabel: "Avg Subscribers" },
  pub:    { label: "Pub",   multiplierLabel: "Size × Publishers",   avgKey: "pub_mean",  avgLabel: "Avg Publishers" },
  pubsub: { label: "Pub + Sub",    multiplierLabel: "Size × (Pub + Sub)",  avgKey: "sub_mean",  avgLabel: "Avg Subs" },
}

function TopicBandwidthSection({ data }: { data: ExtrasStats["topic_bandwidth"] }) {
  const [mode, setMode] = useState<BandwidthMode>("pubsub")

  const bwArray = mode === "pub" ? (data?.bandwidth_pub ?? data?.bandwidth ?? [])
                : mode === "pubsub" ? (data?.bandwidth_pubsub ?? data?.bandwidth ?? [])
                : (data?.bandwidth_sub ?? data?.bandwidth ?? [])

  const allItems = useMemo(() => (data?.labels ?? []).map((label, i) => ({
    name: label,
    id: data?.ids?.[i],
    bandwidth: bwArray[i] ?? 0,
    bandwidth_pub: data?.bandwidth_pub?.[i] ?? 0,
    bandwidth_sub: data?.bandwidth_sub?.[i] ?? 0,
  })).sort((a, b) => b.bandwidth - a.bandwidth), [data, mode]) // eslint-disable-line react-hooks/exhaustive-deps

  const { search, handleSearch, filtered } = useFilteredSearch(allItems)

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
      <Card className="bg-background pb-3">
        <CardHeader>
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <CardTitle className="text-[11px] text-muted-foreground uppercase tracking-widest">Topic Bandwidth</CardTitle>
            <div className="flex items-center gap-2">
              <ChartSearchBar search={search} onSearch={handleSearch} count={filtered.length} total={allItems.length} />
              <div className="flex items-center gap-1 rounded-md border p-0.5 bg-muted/50">
                {(["pubsub", "pub", "sub"] as BandwidthMode[]).map((m) => (
                  <button key={m} onClick={() => setMode(m)}
                    className={`px-2.5 py-0.5 text-xs rounded transition-colors ${mode === m ? "bg-background shadow font-medium" : "text-muted-foreground hover:text-foreground"}`}>
                    {BANDWIDTH_MODE_CONFIG[m].label}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {mode === "pubsub" ? (
            <EBarChart
              items={filtered}
              series={[
                { key: "bandwidth_pub", label: "Pub BW", color: "#818cf8", stack: "bw", fmt: fmtBytes },
                { key: "bandwidth_sub", label: "Sub BW", color: "#34d399", stack: "bw", fmt: fmtBytes },
              ]}
              onClickId={goToExplorer}
            />
          ) : (
            <EBarChart
              items={filtered}
              series={[{ key: "bandwidth", label: "Bandwidth", color: mode === "sub" ? "#34d399" : "#818cf8", fmt: fmtBytes }]}
              onClickId={goToExplorer}
            />
          )}
        </CardContent>
      </Card>
    </div>
  )
}

type AppBalanceMode = "pubsub" | "pub" | "sub"

function AppBalanceSection({ data }: { data: ExtrasStats["app_balance"] }) {
  const [mode, setMode] = useState<AppBalanceMode>("pubsub")

  const allItems = useMemo(() => (!data ? [] : data.labels.map((label, i) => ({
    name: label,
    id: data.ids?.[i],
    publishes: data.pubs[i],
    subscribes: data.subs[i],
    io: (data.pubs[i] ?? 0) + (data.subs[i] ?? 0),
  })).sort((a, b) => {
    if (mode === "pub") return b.publishes - a.publishes
    if (mode === "sub") return b.subscribes - a.subscribes
    return b.io - a.io
  })), [data, mode]) // eslint-disable-line react-hooks/exhaustive-deps

  const { search, handleSearch, filtered } = useFilteredSearch(allItems)

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
      <Card className="bg-background pb-3">
        <CardHeader>
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <CardTitle className="text-[11px] text-muted-foreground uppercase tracking-widest">App Balance</CardTitle>
            <div className="flex items-center gap-2">
              <ChartSearchBar search={search} onSearch={handleSearch} count={filtered.length} total={allItems.length} />
              <div className="flex items-center gap-1 rounded-md border p-0.5 bg-muted/50">
                {(["pubsub", "pub", "sub"] as AppBalanceMode[]).map((m) => (
                  <button key={m} onClick={() => setMode(m)}
                    className={`px-2.5 py-0.5 text-xs rounded transition-colors ${mode === m ? "bg-background shadow font-medium" : "text-muted-foreground hover:text-foreground"}`}>
                    {m === "pubsub" ? "Pub + Sub" : m === "pub" ? "Pub" : "Sub"}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {mode === "pubsub" ? (
            <EBarChart
              items={filtered}
              series={[
                { key: "publishes", label: "Publishes", color: "#818cf8", stack: "a" },
                { key: "subscribes", label: "Subscribes", color: "#34d399", stack: "a" },
              ]}
              onClickId={goToExplorer}
            />
          ) : mode === "pub" ? (
            <EBarChart
              items={filtered}
              series={[{ key: "publishes", label: "Publishes", color: "#818cf8" }]}
              onClickId={goToExplorer}
            />
          ) : (
            <EBarChart
              items={filtered}
              series={[{ key: "subscribes", label: "Subscribes", color: "#34d399" }]}
              onClickId={goToExplorer}
            />
          )}
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

  const { search, handleSearch, filtered } = useFilteredSearch(allItems)

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
      <Card className="bg-background pb-3">
        <CardHeader>
          <div className="flex items-center justify-between gap-2">
            <CardTitle className="text-[11px] text-muted-foreground uppercase tracking-widest">Topic Fanout</CardTitle>
            <ChartSearchBar search={search} onSearch={handleSearch} count={filtered.length} total={allItems.length} />
          </div>
        </CardHeader>
        <CardContent>
          <EBarChart
            items={filtered}
            series={[{ key: "fanout", label: "Fanout", color: "#fbbf24" }]}
            onClickId={goToExplorer}
          />
        </CardContent>
      </Card>
    </div>
  )
}


function HeatmapSection({ data, title, modeToggle, insights }: {
  data: {
    labels: string[]
    node_ids?: string[]
    segment_ids?: string[]
    matrix: number[][]
    matrix_kb?: number[][]
    summary: SummaryDict
    outlier_pairs: [string, string, number, number][]
    per_node?: Record<string, { label: string; apps: { id: string; name: string }[]; pub_topics: { id: string; name: string; size_kb?: number }[]; sub_topics: { id: string; name: string; size_kb?: number }[] }>
    per_segment?: Record<string, { label: string; pub_topics: { id: string; name: string }[]; sub_topics: { id: string; name: string }[] }>
  } | undefined
  title: string
  modeToggle?: boolean
  insights?: React.ReactNode
}) {
  const [showKb, setShowKb] = useState(false)
  const [search, setSearch] = useState("")
  const { resolvedTheme } = useTheme()
  const isDark = resolvedTheme !== "light"

  // Theme-aware color tokens
  const axisColor     = isDark ? "#94a3b8" : "#64748b"
  const tooltipBg     = isDark ? "#1e293b" : "#ffffff"
  const tooltipBorder = isDark ? "rgba(148,163,184,0.2)" : "rgba(100,116,139,0.25)"
  const tooltipText   = isDark ? "#e2e8f0" : "#1e293b"
  const tooltipMuted  = isDark ? "rgba(226,232,240,0.65)" : "rgba(71,85,105,0.8)"
  const splitAreaColors = isDark
    ? ["rgba(30,41,59,0.3)", "rgba(15,23,42,0.3)"]
    : ["rgba(241,245,249,0.6)", "rgba(226,232,240,0.4)"]
  const cellLabelColor = isDark ? "#e2e8f0" : "#1e293b"

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

  const indices = filteredIndices

  let labels = indices.map((i) => data.labels[i])
  let baseMatrix = indices.map((ri) => indices.map((ci) => data.matrix[ri][ci]))
  let ids = data.node_ids
    ? indices.map((i) => data.node_ids![i])
    : data.segment_ids
    ? indices.map((i) => data.segment_ids![i])
    : undefined

  const countMatrix = baseMatrix

  const baseMatrixKb = (showKb && data.matrix_kb) ? indices.map((ri) => indices.map((ci) => data.matrix_kb!![ri][ci])) : null
  const kbMatrix = baseMatrixKb ?? null

  const matrix = kbMatrix ?? countMatrix

  const maxVal = Math.max(1, ...matrix.flat())
  const n = labels.length
  const chartHeight = Math.max(260, n * 34 + 80)

  // Build flat [colIndex, rowIndex, value] data for ECharts heatmap
  const heatmapData: [number, number, number][] = []
  for (let ri = 0; ri < matrix.length; ri++) {
    for (let ci = 0; ci < (matrix[ri]?.length ?? 0); ci++) {
      heatmapData.push([ci, ri, matrix[ri][ci]])
    }
  }

  function fmtCellVal(v: number): string {
    if (v <= 0) return ""
    if (showKb) {
      if (v >= 1_048_576) return (v / 1_048_576).toFixed(1) + "M"
      if (v >= 1024) return (v / 1024).toFixed(1) + "K"
      return v.toFixed(0)
    }
    return String(v)
  }

  const perEntity = data.per_node ?? data.per_segment

  const heatmapOption = {
    backgroundColor: "transparent",
    tooltip: {
      position: "top" as const,
      backgroundColor: tooltipBg,
      borderColor: tooltipBorder,
      textStyle: { color: tooltipText },
      formatter: (params: { data: [number, number, number] }) => {
        const [ci, ri, val] = params.data
        const rLabel = labels[ri] ?? ri
        const cLabel = labels[ci] ?? ci
        const fmtVal = fmtCellVal(val) || "0"
        let topicsHtml = ""
        if (ids && perEntity && val > 0) {
          const rowId = ids[ri]
          const colId = ids[ci]
          const colSubIds = new Set(perEntity[colId]?.sub_topics.map((t) => t.id) ?? [])
          const shared = (perEntity[rowId]?.pub_topics ?? []).filter((t) => colSubIds.has(t.id))
          if (shared.length > 0) {
            const topicItems = shared.map((t) =>
              `<div style="font-family:monospace;font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:220px;color:${tooltipMuted}">${t.name}</div>`
            ).join("")
            topicsHtml = `<div style="margin-top:5px;border-top:1px solid ${tooltipBorder};padding-top:4px">${topicItems}</div>`
          }
        }
        return `<div style="font-size:12px;line-height:1.7;color:${tooltipText}"><b>${rLabel} → ${cLabel}</b><br/><span style="color:${tooltipMuted}">${fmtVal}</span>${topicsHtml}</div>`
      },
    },
    grid: { top: 20, right: 20, bottom: n > 10 ? 80 : 50, left: n > 10 ? 120 : 80 },
    xAxis: {
      type: "category" as const,
      data: labels,
      axisLabel: { rotate: 45, fontSize: 11, color: axisColor },
      axisLine: { lineStyle: { color: axisColor } },
      splitArea: { show: true, areaStyle: { color: splitAreaColors } },
    },
    yAxis: {
      type: "category" as const,
      data: labels,
      inverse: true,
      axisLabel: { fontSize: 11, color: axisColor },
      axisLine: { lineStyle: { color: axisColor } },
      splitArea: { show: true, areaStyle: { color: splitAreaColors } },
    },
    visualMap: {
      min: 0, max: maxVal, show: false,
      inRange: { color: isDark
        ? ["#0f172a", "#2e1065", "#6d28d9", "#7c3aed", "#a78bfa"]
        : ["#f1f5f9", "#ddd6fe", "#a78bfa", "#7c3aed", "#4c1d95"] },
    },
    series: [{
      type: "heatmap" as const,
      data: heatmapData,
      label: {
        show: n <= 20,
        formatter: (params: { data: [number, number, number] }) => fmtCellVal(params.data[2]),
        fontSize: n > 15 ? 9 : n > 10 ? 10 : 11,
        color: cellLabelColor,
      },
      emphasis: { itemStyle: { borderColor: "#818cf8", borderWidth: 2 } },
    }],
  }

  const heatmapEvents = {}

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
      <Card className="bg-background pb-3">
        <CardHeader>
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <CardTitle className="text-[11px] text-muted-foreground uppercase tracking-widest">{title}</CardTitle>
            <div className="flex items-center gap-2">
              <ChartSearchBar search={search} onSearch={(v) => setSearch(v)} count={filteredIndices.length} total={totalLabels} />
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
                      {kb ? "Bytes" : "Count"}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <ReactECharts
                option={heatmapOption}
                notMerge={true}
                style={{ height: `${chartHeight}px`, width: "100%" }}
                onEvents={heatmapEvents as Record<string, (params: { data?: [number, number, number] }) => void>}
              />
        </CardContent>
      </Card>

      {data.outlier_pairs.length > 0 && (
        <OutlierTable title="Outlier Pairs" headers={["Source", "Target", "Count", "Deviation"]} rows={data.outlier_pairs} />
      )}
    </div>
  )
}

type NodeLoadMode = "pubsub" | "pub" | "sub"

function NodeCommLoadSection({ data }: { data: ExtrasStats["node_comm_load"] }) {
  const [mode, setMode] = useState<NodeLoadMode>("pubsub")

  const allItems = useMemo(() => (!data ? [] : data.sorted_labels.map((label, i) => ({
    name: label,
    id: data.sorted_ids?.[i],
    publishes: data.sorted_pub[i],
    subscribes: data.sorted_sub[i],
    io: (data.sorted_pub[i] ?? 0) + (data.sorted_sub[i] ?? 0),
  })).sort((a, b) => {
    if (mode === "pub") return b.publishes - a.publishes
    if (mode === "sub") return b.subscribes - a.subscribes
    return b.io - a.io
  })), [data, mode]) // eslint-disable-line react-hooks/exhaustive-deps

  const { search, handleSearch, filtered } = useFilteredSearch(allItems)

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
      <Card className="bg-background pb-3">
        <CardHeader>
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <CardTitle className="text-[11px] text-muted-foreground uppercase tracking-widest">Node Load</CardTitle>
            <div className="flex items-center gap-2">
              <ChartSearchBar search={search} onSearch={handleSearch} count={filtered.length} total={allItems.length} />
              <div className="flex items-center gap-1 rounded-md border p-0.5 bg-muted/50">
                {(["pubsub", "pub", "sub"] as NodeLoadMode[]).map((m) => (
                  <button key={m} onClick={() => setMode(m)}
                    className={`px-2.5 py-0.5 text-xs rounded transition-colors ${mode === m ? "bg-background shadow font-medium" : "text-muted-foreground hover:text-foreground"}`}>
                    {m === "pubsub" ? "Pub + Sub" : m === "pub" ? "Pub" : "Sub"}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {mode === "pubsub" ? (
            <EBarChart
              items={filtered}
              series={[
                { key: "publishes", label: "Publishes", color: "#818cf8", stack: "a" },
                { key: "subscribes", label: "Subscribes", color: "#34d399", stack: "a" },
              ]}
              onClickId={goToExplorer}
            />
          ) : mode === "pub" ? (
            <EBarChart
              items={filtered}
              series={[{ key: "publishes", label: "Publishes", color: "#818cf8" }]}
              onClickId={goToExplorer}
            />
          ) : (
            <EBarChart
              items={filtered}
              series={[{ key: "subscribes", label: "Subscribes", color: "#34d399" }]}
              onClickId={goToExplorer}
            />
          )}
        </CardContent>
      </Card>
      {data.outliers.length > 0 && (
        <OutlierTable title="Outlier Nodes" headers={["Node", "Pub", "Sub", "Total", "Deviation"]} rows={data.outliers} />
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

  const { search, handleSearch, filtered } = useFilteredSearch(allItems)

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
        <Card className="bg-background pb-3">
          <CardHeader>
            <div className="flex items-center justify-between gap-2">
              <CardTitle className="text-[11px] text-muted-foreground uppercase tracking-widest">Criticality I/O</CardTitle>
              <ChartSearchBar search={search} onSearch={handleSearch} count={filtered.length} total={allItems.length} />
            </div>
          </CardHeader>
          <CardContent>
            <EBarChart
              items={filtered}
              series={[
                { key: "publishes", label: "Publishes", color: "#818cf8", stack: "a" },
                { key: "subscribes", label: "Subscribes", color: "#34d399", stack: "a" },
              ]}
              onClickId={goToExplorer}
            />
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

  const { search, handleSearch, filtered } = useFilteredSearch(allItems)

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
      <Card className="bg-background pb-3">
        <CardHeader>
          <div className="flex items-center justify-between gap-2">
            <CardTitle className="text-[11px] text-muted-foreground uppercase tracking-widest">Library Deps</CardTitle>
            <ChartSearchBar search={search} onSearch={handleSearch} count={filtered.length} total={allItems.length} />
          </div>
        </CardHeader>
        <CardContent>
          <EBarChart
            items={filtered}
            series={[
              { key: "inbound", label: "In-degree", color: "#a78bfa" },
              { key: "outbound", label: "Out-degree", color: "#2dd4bf" },
            ]}
            onClickId={goToExplorer}
          />
        </CardContent>
      </Card>
      {data.outliers.length > 0 && (
        <OutlierTable title="High-Dependency Outliers" headers={["Entity", "In-degree", "Out-degree"]} rows={data.outliers} />
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

  const { search, handleSearch, filtered } = useFilteredSearch(allItems)

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
      <Card className="bg-background pb-3">
        <CardHeader>
          <div className="flex items-center justify-between gap-2">
            <CardTitle className="text-[11px] text-muted-foreground uppercase tracking-widest">Node Density</CardTitle>
            <ChartSearchBar search={search} onSearch={handleSearch} count={filtered.length} total={allItems.length} />
          </div>
        </CardHeader>
        <CardContent>
          <EBarChart
            items={filtered}
            series={[
              { key: "critical", label: "Critical", color: "#fb7185", stack: "a" },
              { key: "normal", label: "Normal", color: "#94a3b8", stack: "a" },
            ]}
            onClickId={goToExplorer}
          />
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

  const { search, handleSearch, filtered } = useFilteredSearch(allItems)

  if (!data || !data.labels.length) return <p className="text-sm text-muted-foreground">Insufficient segment data (need ≥ 2 segments)</p>

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        <MetricInsightCard
          label="Avg Apps per Segment"
          value={Number(data.summary.app_mean ?? 0).toFixed(1)}
          description="Mean number of applications per segment. Very low values may indicate overly fragmented subsystems; very high values may signal monolithic segments with unclear boundaries."
          formula="total_apps / segment_count"
        />
        <MetricInsightCard
          label="Avg Topics per Segment"
          value={Number(data.summary.topic_mean ?? 0).toFixed(1)}
          description="Mean number of topics owned or used per segment. Reflects how much communication surface each subsystem exposes to the rest of the architecture."
          formula="total_topics_touched / segment_count"
        />
        <MetricInsightCard
          label="Avg I/O per Segment"
          value={Number(data.summary.io_mean ?? 0).toFixed(1)}
          description="Average pub/sub message load aggregated per segment. High I/O segments are communication hubs whose degradation has the widest downstream reach."
          formula="mean(Σ pub + Σ sub per segment)"
        />
      </div>
      <SummaryCards summary={data.summary} keys={[
        { key: "css_count", label: "Segments" },
        { key: "app_mean", label: "Avg Apps/Segment" },
        { key: "app_max", label: "Max Apps" },
        { key: "topic_mean", label: "Avg Topics/Segment" },
        { key: "io_mean", label: "Avg I/O" },
        { key: "io_max", label: "Max I/O" },
      ]} />
      <Card className="bg-background pb-3">
        <CardHeader>
          <div className="flex items-center justify-between gap-2">
            <CardTitle className="text-[11px] text-muted-foreground uppercase tracking-widest">Segment Diversity</CardTitle>
            <ChartSearchBar search={search} onSearch={handleSearch} count={filtered.length} total={allItems.length} />
          </div>
        </CardHeader>
        <CardContent>
          <EBarChart
            items={filtered}
            series={[
              { key: "applications", label: "Applications", color: "#818cf8" },
              { key: "topics", label: "Topics", color: "#34d399" },
              { key: "io", label: "I/O Load", color: "#fbbf24" },
            ]}
          />
        </CardContent>
      </Card>
    </div>
  )
}

function CoeffInput({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  return (
    <input
      type="number"
      min={0}
      max={1}
      step={0.01}
      value={value}
      onChange={(e) => {
        const v = parseFloat(e.target.value)
        if (!isNaN(v) && v >= 0) onChange(v)
      }}
      className="w-14 h-6 rounded border bg-background px-1 text-xs font-mono text-center text-foreground focus:outline-none focus:ring-1 focus:ring-ring"
    />
  )
}

const BOTTLENECK_DEFAULT_W = { bt: 0.40, ap: 0.25, br: 0.20, bridge: 0.15 }

function computeBottleneckItems(data: ExtrasStats["bottleneck"], w: typeof BOTTLENECK_DEFAULT_W) {
  if (!data) return []
  const t = (w.bt + w.ap + w.br + w.bridge) || 1
  const wBt = w.bt / t, wAp = w.ap / t, wBr = w.br / t, wBridge = w.bridge / t
  const scored = data.items
    .map((it) => ({
      ...it,
      bottleneck_score: Math.round((wBt * it.betweenness + wAp * it.ap_c_directed + wBr * it.blast_radius_norm + wBridge * it.bridge_ratio) * 10000) / 10000,
    }))
    .sort((a, b) => b.bottleneck_score - a.bottleneck_score)
  const scores = scored.map((it) => it.bottleneck_score).filter((s) => s > 0)
  const outlierIds = new Set<string>()
  if (scores.length >= 4) {
    const ss = [...scores].sort((a, b) => a - b)
    const q1 = ss[Math.floor(ss.length * 0.25)]
    const q3 = ss[Math.floor(ss.length * 0.75)]
    const upper = q3 + 1.5 * (q3 - q1)
    scored.forEach((it) => { if (it.bottleneck_score > upper) outlierIds.add(it.id) })
  }
  return scored.map((it) => ({ ...it, outlier: outlierIds.has(it.id) }))
}

function BottleneckSection({ data }: { data: ExtrasStats["bottleneck"] }) {
  const [pendingW, setPendingW] = useState(BOTTLENECK_DEFAULT_W)
  const [appliedW, setAppliedW] = useState(BOTTLENECK_DEFAULT_W)

  const allItems = useMemo(() => computeBottleneckItems(data, appliedW), [data, appliedW])

  const { search, handleSearch, filtered } = useFilteredSearch(allItems)

  if (!data) return null
  const s = data.summary

  const wTotal = appliedW.bt + appliedW.ap + appliedW.br + appliedW.bridge || 1
  const wNorm = {
    bt: appliedW.bt / wTotal,
    ap: appliedW.ap / wTotal,
    br: appliedW.br / wTotal,
    bridge: appliedW.bridge / wTotal,
  }
  const isDefault = Math.abs(appliedW.bt - BOTTLENECK_DEFAULT_W.bt) < 0.001 && Math.abs(appliedW.ap - BOTTLENECK_DEFAULT_W.ap) < 0.001 &&
    Math.abs(appliedW.br - BOTTLENECK_DEFAULT_W.br) < 0.001 && Math.abs(appliedW.bridge - BOTTLENECK_DEFAULT_W.bridge) < 0.001
  const isDirty = pendingW.bt !== appliedW.bt || pendingW.ap !== appliedW.ap ||
    pendingW.br !== appliedW.br || pendingW.bridge !== appliedW.bridge
  const recomputedOutlierCount = allItems.filter((it) => it.outlier).length
  const recomputedMaxScore = allItems.length > 0 ? allItems[0].bottleneck_score : 0

  function applyWeights() { setAppliedW(pendingW) }
  function resetWeights() { setPendingW(BOTTLENECK_DEFAULT_W); setAppliedW(BOTTLENECK_DEFAULT_W) }

  return (
    <div className="space-y-4">
      {/* Score formula with editable coefficients */}
      <div className="rounded-lg border bg-background px-4 py-3 text-sm text-muted-foreground">
        <div className="flex flex-wrap items-center gap-x-1.5 gap-y-2 font-mono text-xs">
          <span className="font-semibold text-foreground text-sm mr-0.5">Score =</span>
          <CoeffInput value={pendingW.bt} onChange={(v) => setPendingW((p) => ({ ...p, bt: v }))} />
          <span className="text-foreground">× betweenness</span>
          <span className="text-muted-foreground">+</span>
          <CoeffInput value={pendingW.ap} onChange={(v) => setPendingW((p) => ({ ...p, ap: v }))} />
          <span className="text-foreground">× ap_c_directed</span>
          <span className="text-muted-foreground">+</span>
          <CoeffInput value={pendingW.br} onChange={(v) => setPendingW((p) => ({ ...p, br: v }))} />
          <span className="text-foreground">× blast_radius_norm</span>
          <span className="text-muted-foreground">+</span>
          <CoeffInput value={pendingW.bridge} onChange={(v) => setPendingW((p) => ({ ...p, bridge: v }))} />
          <span className="text-foreground">× bridge_ratio</span>
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-3 text-xs">
          <button
            onClick={applyWeights}
            disabled={!isDirty}
            className="rounded border px-2.5 py-0.5 text-xs font-medium transition-colors disabled:opacity-40 disabled:cursor-not-allowed enabled:bg-primary enabled:text-primary-foreground enabled:hover:bg-primary/90"
          >
            Recalculate
          </button>
          {!isDefault && (
            <button
              onClick={resetWeights}
              className="rounded border px-2 py-0.5 text-xs text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
            >
              Reset defaults
            </button>
          )}
          {!isDefault && (
            <span className="text-muted-foreground">
              Effective (normalized):&nbsp;
              <span className="font-mono text-foreground">{wNorm.bt.toFixed(2)} / {wNorm.ap.toFixed(2)} / {wNorm.br.toFixed(2)} / {wNorm.bridge.toFixed(2)}</span>
              &nbsp;(sum = 1)
            </span>
          )}
          {isDefault && !isDirty && <span className="text-xs">Weights auto-normalize to sum = 1. Adjust then click Recalculate.</span>}
        </div>
        <p className="mt-1 text-xs">
          Components with the highest score lie on the most critical structural paths — their failure disrupts the largest fraction of the system.
          Articulation points (🔴) are graph-theoretic SPOFs: their removal disconnects the graph entirely.
        </p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MetricInsightCard
          label="Articulation Points"
          value={s.articulation_point_count ?? 0}
          description="Components whose removal disconnects the undirected graph. These are structural SPOFs — the most severe bottleneck class."
          formula="is_articulation_point = True"
        />
        <MetricInsightCard
          label="Directed APs"
          value={s.directed_ap_count ?? 0}
          description="Components that disconnect the directed reachable set. Directional SPOFs that may not appear in undirected analysis."
          formula="is_directed_ap = True"
        />
        <MetricInsightCard
          label="Score Outliers"
          value={recomputedOutlierCount}
          description="Components whose bottleneck score exceeds the IQR upper fence. These are statistically extreme relative to the rest of the system."
          formula="score > Q3 + 1.5 × IQR"
        />
        <MetricInsightCard
          label="Max Score"
          value={fmtNum(recomputedMaxScore)}
          description="Highest composite bottleneck score in the system. Values above 0.5 indicate a severe single-point bottleneck."
          formula={`max(${wNorm.bt.toFixed(2)}·BT + ${wNorm.ap.toFixed(2)}·AP + ${wNorm.br.toFixed(2)}·BR + ${wNorm.bridge.toFixed(2)}·bridge)`}
        />
      </div>

      {/* Bottleneck bar chart */}
      <Card className="bg-background pb-3">
        <CardHeader>
          <div className="flex items-center justify-between gap-2">
            <CardTitle className="text-[11px] text-muted-foreground uppercase tracking-widest">Bottlenecks</CardTitle>
            <ChartSearchBar search={search} onSearch={handleSearch} count={filtered.length} total={allItems.length} />
          </div>
        </CardHeader>
        <CardContent>
          <EBarChart
            items={filtered}
            series={[{ key: "bottleneck_score", label: "Bottleneck Score", color: "#fb7185" }]}
            onClickId={goToExplorer}
            yDomain={[0, 1]}
          />
        </CardContent>
      </Card>

      {/* Ranked table */}
      <div>
        <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">All Components Ranked by Bottleneck Score</p>
        <div className="max-h-[480px] overflow-auto rounded-lg border border-border">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground uppercase tracking-wide w-8">#</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground uppercase tracking-wide">Component</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground uppercase tracking-wide">Type</th>
                <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground uppercase tracking-wide">Score</th>
                <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground uppercase tracking-wide">Betweenness</th>
                <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground uppercase tracking-wide">Blast Radius</th>
                <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground uppercase tracking-wide">Bridge Ratio</th>
                <th className="px-3 py-2 text-center text-xs font-medium text-muted-foreground uppercase tracking-wide w-16">Flags</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((it, idx) => (
                <tr
                  key={it.id}
                  className={`border-b border-border/50 hover:bg-muted/30 transition-colors cursor-pointer ${it.outlier ? "bg-rose-500/5" : ""}`}
                  onClick={() => goToExplorer(it.id)}
                >
                  <td className="px-3 py-2 text-xs text-muted-foreground">{idx + 1}</td>
                  <td className="px-3 py-2 font-medium">
                    <ItemTooltip
                      data={{
                        type: it.type,
                        metrics: {
                          weight:              it.weight,
                          cascade_depth:       it.cascade_depth,
                          pubsub_betweenness:  it.pubsub_betweenness,
                          is_articulation_point: it.is_articulation_point,
                          is_directed_ap:      it.is_directed_ap,
                        },
                      }}
                      side="right"
                    >
                      <span>{it.name}</span>
                    </ItemTooltip>
                  </td>
                  <td className="px-3 py-2 text-xs text-muted-foreground">{it.type}</td>
                  <td className="px-3 py-2 text-right font-mono text-sm font-semibold">{fmtNum(it.bottleneck_score)}</td>
                  <td className="px-3 py-2 text-right font-mono text-xs text-muted-foreground">{fmtNum(it.betweenness)}</td>
                  <td className="px-3 py-2 text-right font-mono text-xs text-muted-foreground">{it.blast_radius}</td>
                  <td className="px-3 py-2 text-right font-mono text-xs text-muted-foreground">{fmtNum(it.bridge_ratio)}</td>
                  <td className="px-3 py-2 text-center text-sm w-16">
                    {it.is_articulation_point && <span title="Articulation point (undirected SPOF)">🔴</span>}
                    {it.is_directed_ap && !it.is_articulation_point && <span title="Directed articulation point">🟠</span>}
                    {it.outlier && <span title="Score outlier (IQR)">⚡</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

// ── Tab config ──────────────────────────────────────────────────────────

const TAB_CONFIG = [
  { id: "topic_bandwidth", label: "Topic Bandwidth", icon: Radio, color: "text-violet-500", description: "Bandwidth per topic (message size × connections). Spot the channels that dominate network load." },
  { id: "app_balance", label: "App Balance", icon: Activity, color: "text-blue-500", description: "Pub/sub ratio per application. Reveals pure producers, pure consumers, and high-I/O hubs." },
  { id: "topic_fanout", label: "Topic Fanout", icon: Network, color: "text-amber-500", description: "Publisher × subscriber count per topic. High fanout amplifies blast radius when a publisher fails." },
  { id: "cross_node", label: "Cross-Node", icon: Server, color: "text-purple-500", description: "Inter-node message flow matrix. Highlights tightly-coupled physical hosts and network hotspots." },
  { id: "node_load", label: "Node Load", icon: BarChart3, color: "text-pink-500", description: "Total pub/sub activity per physical node. Identifies overloaded hosts and deployment imbalances." },
  { id: "domain_comm", label: "Segment Communication", icon: Layers, color: "text-cyan-500", description: "Message flow between architectural segments. High cross-segment traffic signals tight subsystem coupling." },
  { id: "criticality", label: "Criticality I/O", icon: AlertTriangle, color: "text-red-500", description: "I/O load of critical vs. normal applications. Shows whether mission-critical components are also communication hotspots." },
  { id: "lib_deps", label: "Library Deps", icon: BookOpen, color: "text-teal-500", description: "In/out-degree per library. High in-degree means a shared-fate risk — one library failure hits many consumers." },
  { id: "node_density", label: "Node Density", icon: Shield, color: "text-green-500", description: "Critical application density per physical node. Concentrated critical apps mean a single host failure causes outsized impact." },
  { id: "domain_div", label: "Segment Diversity", icon: Box, color: "text-orange-500", description: "Apps, topics, and I/O load per segment. Low diversity flags monolithic subsystems; high I/O flags communication hubs." },
  { id: "bottleneck", label: "Bottlenecks", icon: Zap, color: "text-yellow-500", description: "Composite structural score: betweenness, SPOF severity, blast radius, and bridge ratio. The top scorers are your highest-risk single points of failure." },
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
  bottleneck: "bottleneck",
}

// ── Main page ───────────────────────────────────────────────────────────

export default function StatisticsPage() {
  const { status, config, initialLoadComplete } = useConnection()
  const isConnected = status === "connected"
  const [tabData, setTabData] = useState<Partial<ExtrasStats>>({})
  const [tabLoading, setTabLoading] = useState<Partial<Record<TabId, boolean>>>({})
  const [tabError, setTabError] = useState<Partial<Record<TabId, string>>>({})
  const [activeTab, setActiveTab] = useState<TabId>("topic_bandwidth")
  const [selectedSection, setSelectedSection] = useState<TabId | null>(null)

  // Populate the module-level node map for chart tooltip enrichment
  useEffect(() => {
    if (!isConnected || !config) return
    apiClient.getGraphData().then(graphData => {
      _nodeById.clear()
      graphData.nodes.forEach((n: any) => {
        _nodeById.set(n.id, { type: n.type ?? "Application", properties: n.properties ?? {} })
      })
    }).catch(() => { /* non-critical — tooltips just won't show node type */ })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isConnected, config])

  const fetchTab = async (tabId: TabId, force = false) => {
    const chartKey = TAB_TO_CHART_ID[tabId]
    if (!force && tabData[chartKey] !== undefined) return
    if (!isConnected || !config) return

    setTabLoading((prev) => ({ ...prev, [tabId]: true }))
    setTabError((prev) => ({ ...prev, [tabId]: undefined }))
    try {
      const creds = apiClient.getCredentials()
      if (!creds) throw new Error("No credentials")

      // Bottleneck tab uses a dedicated endpoint that runs structural analysis
      const url = tabId === "bottleneck"
        ? `${API_BASE_URL}/api/v1/stats/bottleneck`
        : `${API_BASE_URL}/api/v1/stats/chart/${chartKey}`

      const response = await fetch(url, {
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

  useEffect(() => {
    if (isConnected && config) {
      fetchTab("topic_bandwidth")
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isConnected, config])

  if (!initialLoadComplete || status === "connecting") {
    return (
      <AppLayout title="Statistics" description="Structural and communication metrics across topics, applications, nodes, and libraries">
        <div className="space-y-5">
          {/* Section card grid skeleton */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {Array.from({ length: 9 }).map((_, i) => (
              <div key={i} className="rounded-lg border border-border bg-muted/20 p-6 space-y-3">
                <Skeleton className="h-6 w-6 rounded-md" />
                <Skeleton className="h-4 w-32" />
                <Skeleton className="h-3 w-full" />
                <Skeleton className="h-3 w-4/5" />
              </div>
            ))}
          </div>
        </div>
      </AppLayout>
    )
  }

  if (!isConnected) {
    return (
      <AppLayout title="Statistics" description="Structural and communication metrics across topics, applications, nodes, and libraries">
        <NoConnectionInfo description="Connect to your Neo4j database to view statistics" />
      </AppLayout>
    )
  }

  return (
    <AppLayout title="Statistics" description="Structural and communication metrics across topics, applications, nodes, and libraries">
      <div className="space-y-6">
        <>
            {/* Card grid — hidden once a section is selected */}
            {!selectedSection && (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {TAB_CONFIG.map(({ id, label, icon: Icon, color, description }) => (
                  <button
                    key={id}
                    onClick={() => { handleTabChange(id); setSelectedSection(id) }}
                    className="text-left p-6 rounded-lg border border-border bg-background hover:border-primary/50 hover:bg-muted/50 transition-colors"
                  >
                    <Icon className={`h-6 w-6 ${color} mb-3`} />
                    <div className="font-semibold text-sm mb-2">{label}</div>
                    <p className="text-sm text-muted-foreground">{description}</p>
                  </button>
                ))}
              </div>
            )}

            {/* Breadcrumb + full-width content */}
            {selectedSection && (() => {
              const cfg = TAB_CONFIG.find(t => t.id === selectedSection)!
              const Icon = cfg.icon
              const isLoading = !!tabLoading[selectedSection]
              const error = tabError[selectedSection]
              const chartKey = TAB_TO_CHART_ID[selectedSection]
              const data = tabData[chartKey] as ExtrasStats[typeof chartKey] | undefined
              return (
                <div>
                  <button
                    onClick={() => setSelectedSection(null)}
                    className="flex items-center gap-1.5 text-sm text-muted-foreground hover:text-foreground transition-colors mb-4"
                  >
                    <ChevronLeft className="h-4 w-4" />
                    All statistics
                  </button>
                  <div className="flex items-center gap-2 mb-4">
                    <Icon className={`h-5 w-5 ${cfg.color}`} />
                    <h2 className="font-semibold text-base">{cfg.label}</h2>
                    <span className="text-sm text-muted-foreground">— {cfg.description}</span>
                  </div>
                  {isLoading && (
                    <div className="space-y-4">
                      {/* Summary cards skeleton */}
                      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
                        {Array.from({ length: 4 }).map((_, i) => (
                          <div key={i} className="rounded-lg border border-border bg-background p-4 space-y-2">
                            <Skeleton className="h-3 w-24" />
                            <Skeleton className="h-7 w-16" />
                            <Skeleton className="h-2.5 w-32" />
                          </div>
                        ))}
                      </div>
                      {/* Bar chart skeleton */}
                      <div className="rounded-lg border border-border bg-background p-4">
                        <Skeleton className="h-4 w-36 mb-4" />
                        <div className="flex items-end gap-2 h-52 pt-2">
                          {Array.from({ length: 16 }).map((_, i) => (
                            <Skeleton
                              key={i}
                              className="flex-1 rounded-sm"
                              style={{ height: `${20 + (i * 23 + 31) % 75}%` }}
                            />
                          ))}
                        </div>
                        <div className="flex gap-3 mt-3">
                          {Array.from({ length: 3 }).map((_, i) => (
                            <div key={i} className="flex items-center gap-1.5">
                              <Skeleton className="h-2.5 w-2.5 rounded-sm" />
                              <Skeleton className="h-2.5 w-14" />
                            </div>
                          ))}
                        </div>
                      </div>
                      {/* Table / ranked list skeleton */}
                      <div className="rounded-lg border border-border bg-background p-4">
                        <Skeleton className="h-4 w-48 mb-4" />
                        <div className="space-y-2.5">
                          {Array.from({ length: 7 }).map((_, i) => (
                            <div key={i} className="flex items-center gap-3">
                              <Skeleton className="h-3 shrink-0" style={{ width: `${55 + (i * 19) % 80}px` }} />
                              <Skeleton className="h-4 flex-1 rounded-sm" />
                              <Skeleton className="h-3 w-10 shrink-0" />
                            </div>
                          ))}
                        </div>
                      </div>
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
                    getPrimaryLength(data) === 0
                      ? <EmptyDataState />
                      : <>
                      {selectedSection === "topic_bandwidth" && <TopicBandwidthSection data={tabData.topic_bandwidth} />}
                      {selectedSection === "app_balance" && <AppBalanceSection data={tabData.app_balance} />}
                      {selectedSection === "topic_fanout" && <TopicFanoutSection data={tabData.topic_fanout} />}
                      {selectedSection === "cross_node" && <HeatmapSection data={tabData.cross_node_heatmap} title="Cross-Node" modeToggle insights={
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
                      {selectedSection === "node_load" && <NodeCommLoadSection data={tabData.node_comm_load} />}
                      {selectedSection === "domain_comm" && <HeatmapSection data={tabData.domain_comm} title="Segment Communication" modeToggle insights={
                        tabData.domain_comm && (
                          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
                            <MetricInsightCard
                              label="Active Cell %"
                              value={`${Number(tabData.domain_comm.summary.active_pct ?? 0).toFixed(1)}%`}
                              description="Fraction of segment pairs with cross-segment message traffic. High values signal tight coupling between subsystems and potential dependency debt."
                              formula="nonzero_cells / total_cells × 100"
                            />
                            <MetricInsightCard
                              label="Intra-Segment Events"
                              value={tabData.domain_comm.summary.intra_total ?? 0}
                              description="Topic exchanges where publisher and subscriber belong to the same segment. High intra-segment traffic signals well-encapsulated, loosely coupled subsystems."
                              formula="Σ matrix[i][i]  (diagonal sum)"
                            />
                            <MetricInsightCard
                              label="Cross-Segment Events"
                              value={tabData.domain_comm.summary.inter_total ?? 0}
                              description="Topic flows that cross segment boundaries. High values indicate interdependency between subsystems and wider cascade paths under failure."
                              formula="Σ matrix[i][j]  for i ≠ j"
                            />
                          </div>
                        )
                      } />}
                      {selectedSection === "criticality" && <CriticalityIOSection data={tabData.criticality_io} />}
                      {selectedSection === "lib_deps" && <LibDependencySection data={tabData.lib_dependency} />}
                      {selectedSection === "node_density" && <NodeCriticalDensitySection data={tabData.node_critical_density} />}
                      {selectedSection === "domain_div" && <DomainDiversitySection data={tabData.domain_diversity} />}
                      {selectedSection === "bottleneck" && <BottleneckSection data={tabData.bottleneck} />}
                    </>
                  )}
                </div>
              )
            })()}
        </>
      </div>
    </AppLayout>
  )
}
