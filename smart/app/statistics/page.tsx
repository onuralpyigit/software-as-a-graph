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
  Shield, Box, Server, BookOpen, Zap, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, Wifi,
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
  network_usage?: {
    sorted_labels: string[]
    sorted_ids: string[]
    sorted_out: number[]
    sorted_in: number[]
    sorted_total: number[]
    outliers: [string, number, number, number][]
    iqr_upper: number
    summary: SummaryDict
    topic_bandwidth?: Array<{
      id: string
      name: string
      frequency_hz: number
      size_bytes: number
      pub_count: number
      sub_count: number
      bw_out: number
      bw_in: number
      bw_total: number
    }>
    app_bandwidth?: Array<{
      id: string
      name: string
      node_id: string | null
      node_name: string | null
       role: string[] | null
      criticality: boolean
      bw_out: number
      bw_in: number
      bw_total: number
    }>
  }
  qos_distribution?: {
    total_topics: number
    durability: Record<string, number>
    reliability: Record<string, number>
    transport_priority: Record<string, number>
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
  if (d.total_topics !== undefined) return 1 // For qos_distribution
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
  if (n < 1024) return `${Number(n).toLocaleString(undefined, { maximumFractionDigits: 2 })} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(2)} KB`
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
  const tooltipBg       = isDark ? "#1c1c1e" : "#ffffff"
  const tooltipBorder   = isDark ? "#3f3f46" : "#e4e4e7"
  const tooltipText     = isDark ? "#fafafa" : "#09090b"
  const tooltipMuted    = isDark ? "rgba(250,250,250,0.7)" : "rgba(9,9,11,0.7)"
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
      textStyle: { color: tooltipText, fontSize: 12 },
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
          const val = ser?.fmt ? ser.fmt(p.value) : (p.value?.toLocaleString(undefined, { maximumFractionDigits: 2 }) ?? "—")
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
    <div className="grid grid-cols-[repeat(auto-fit,minmax(180px,1fr))] gap-3">
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

function TablePager({ page, totalPages, total, pageSize, label, onPage }: {
  page: number; totalPages: number; total: number; pageSize: number; label: string; onPage: (p: number) => void
}) {
  if (totalPages <= 1) return null
  // page is 0-based internally; display as 1-based
  const p1 = page + 1
  const from = page * pageSize + 1
  const to = Math.min(p1 * pageSize, total)
  return (
    <div className="flex items-center justify-between text-xs text-muted-foreground pt-1">
      <span>Showing {from}–{to} of {total} {label}</span>
      <div className="flex items-center gap-1">
        <button onClick={() => onPage(0)} disabled={page === 0} className="p-1 rounded hover:bg-muted disabled:opacity-30"><ChevronsLeft className="h-3.5 w-3.5" /></button>
        <button onClick={() => onPage(page - 1)} disabled={page === 0} className="p-1 rounded hover:bg-muted disabled:opacity-30"><ChevronLeft className="h-3.5 w-3.5" /></button>
        <span className="px-2">{p1} / {totalPages}</span>
        <button onClick={() => onPage(page + 1)} disabled={page >= totalPages - 1} className="p-1 rounded hover:bg-muted disabled:opacity-30"><ChevronRight className="h-3.5 w-3.5" /></button>
        <button onClick={() => onPage(totalPages - 1)} disabled={page >= totalPages - 1} className="p-1 rounded hover:bg-muted disabled:opacity-30"><ChevronsRight className="h-3.5 w-3.5" /></button>
      </div>
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

function PrimaryStatsCard({
  summary,
  prefix,
  label,
  format,
  description,
  formula,
}: {
  summary: SummaryDict
  prefix: string
  label: string
  format?: (v: number) => string
  description?: string
  formula?: string
}) {
  const mean = summary[`${prefix}_mean`]
  const median = summary[`${prefix}_median`]
  const max = summary[`${prefix}_max`]
  const min = summary[`${prefix}_min`]

  if (mean === undefined && median === undefined && max === undefined && min === undefined) {
    return null
  }

  const fmt = (val: number | string | undefined) => {
    if (val === undefined) return "—"
    return format ? format(Number(val)) : fmtNum(val)
  }

  return (
    <div className="rounded-lg border bg-background p-4 shadow-sm hover:shadow-md transition-shadow flex flex-col h-full">
      <p className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3 border-b border-border pb-2">
        {label}
      </p>
      <div className="grid grid-cols-2 gap-4 mb-3">
        <div>
          <p className="text-xs text-muted-foreground mb-1">Mean</p>
          <p className="text-lg font-bold text-foreground">{fmt(mean)}</p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground mb-1">Median</p>
          <p className="text-lg font-bold text-foreground">{fmt(median)}</p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground mb-1">Maximum</p>
          <p className="text-lg font-bold text-foreground">{fmt(max)}</p>
        </div>
        <div>
          <p className="text-xs text-muted-foreground mb-1">Minimum</p>
          <p className="text-lg font-bold text-foreground">{fmt(min)}</p>
        </div>
      </div>
      {description && (
        <p className="text-xs text-muted-foreground leading-relaxed mb-1 flex-grow">
          {description}
        </p>
      )}
      {formula && (
        <code className="text-[10px] font-mono bg-muted/80 rounded px-2 py-0.5 text-muted-foreground/90 self-start">
          {formula}
        </code>
      )}
    </div>
  )
}

function StatCountCard({
  label,
  value,
  description,
  formula,
}: {
  label: string
  value: number | string
  description: string
  formula?: string
}) {
  return (
    <div className="rounded-lg border bg-background p-4 shadow-sm hover:shadow-md transition-shadow flex flex-col justify-between h-full">
      <div>
        <p className="text-sm font-semibold text-muted-foreground uppercase tracking-wider mb-3 border-b border-border pb-2">
          {label}
        </p>
        <p className="text-3xl font-bold text-foreground mb-3">
          {typeof value === "number" ? fmtNum(value) : value}
        </p>
      </div>
      <div>
        <p className="text-xs text-muted-foreground leading-relaxed mb-2">
          {description}
        </p>
        {formula && (
          <code className="text-[10px] font-mono bg-muted/80 rounded px-2 py-0.5 text-muted-foreground/90 inline-block">
            {formula}
          </code>
        )}
      </div>
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
      <div className="grid grid-cols-[repeat(auto-fit,minmax(220px,1fr))] gap-4">
        <PrimaryStatsCard 
          summary={data.summary} 
          prefix="size" 
          label="Size" 
          format={fmtBytes} 
          description="Message payload size in bytes per topic."
          formula="size_bytes"
        />
        <PrimaryStatsCard 
          summary={data.summary} 
          prefix="pub" 
          label="Publishers" 
          description="Number of applications publishing to each topic."
          formula="count(publishers)"
        />
        <PrimaryStatsCard 
          summary={data.summary} 
          prefix="sub" 
          label="Subscribers" 
          description="Number of applications subscribing to each topic."
          formula="count(subscribers)"
        />
        <PrimaryStatsCard
          summary={data.summary}
          prefix={mode === "pub" ? "bw_pub" : mode === "pubsub" ? "bw_pubsub" : "bw_sub"}
          label={mode === "pub" ? "Publisher Bandwidth" : mode === "pubsub" ? "Pub+Sub Bandwidth" : "Subscriber Bandwidth"}
          format={fmtBytes}
          description={mode === "pub" ? "Total bytes per publish event produced by publishers." : mode === "pubsub" ? "Total bytes per publish event flowing through the topic." : "Total bytes per publish event consumed by subscribers."}
          formula={mode === "pub" ? "size × pub_count" : mode === "pubsub" ? "size × (pub_count + sub_count)" : "size × sub_count"}
        />
        <StatCountCard
          label="Outliers"
          value={data.summary.outlier_count ?? 0}
          description="Topics whose bandwidth exceeds the IQR upper fence (Q3 + 1.5 × IQR)."
        />
      </div>
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
      {data.outlier_indices && data.outlier_indices.length > 0 && (
        <OutlierTable
          title="Bandwidth Outliers"
          headers={["Topic", "Size (B)", "Publishers", "Subscribers", "Bandwidth (B/s)"]}
          rows={data.outlier_indices.map((i) => [
            data.labels[i],
            data.sizes[i],
            data.pubs[i],
            data.subs[i],
            data.bandwidth[i],
          ])}
        />
      )}
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
      <div className="grid grid-cols-[repeat(auto-fit,minmax(220px,1fr))] gap-4">
        <PrimaryStatsCard 
          summary={data.summary} 
          prefix="pub" 
          label="Publishers" 
          description="Number of topics each application publishes to."
          formula="count(published_topics)"
        />
        <PrimaryStatsCard 
          summary={data.summary} 
          prefix="sub" 
          label="Subscribers" 
          description="Number of topics each application subscribes to."
          formula="count(subscribed_topics)"
        />
        <PrimaryStatsCard 
          summary={data.summary} 
          prefix="io" 
          label="I/O Load" 
          description="Total combined publish and subscribe connections per application."
          formula="pub_count + sub_count"
        />
        <StatCountCard
          label="Outliers"
          value={data.summary.outlier_count ?? 0}
          description="Applications whose I/O load exceeds the IQR upper fence."
        />
      </div>
      <div className="grid grid-cols-[repeat(auto-fit,minmax(180px,1fr))] gap-3">
        <StatCountCard label="Total Apps" value={data.summary.total_apps ?? 0} description="Total number of applications registered in the system topology." formula="count(applications)" />
        <StatCountCard label="High I/O" value={data.summary.q_high_io ?? 0} description="Busiest communication hubs that publish and subscribe above average. Their failure disrupts both upstream and downstream flows." formula="pub > avg_pub AND sub > avg_sub" />
        <StatCountCard label="Consumers" value={data.summary.q_consumer ?? 0} description="Pure data sinks that subscribe above average but publish below average. Upstream failures cascade directly into these endpoints." formula="pub ≤ avg_pub AND sub > avg_sub" />
        <StatCountCard label="Producers" value={data.summary.q_producer ?? 0} description="Primary data sources that publish above average but subscribe below average. Their failure causes downstream data loss." formula="pub > avg_pub AND sub ≤ avg_sub" />
        <StatCountCard label="Low Activity" value={data.summary.q_low ?? 0} description="Applications with connection counts at or below the system mean, contributing minimally to overall message flow." formula="pub ≤ avg_pub AND sub ≤ avg_sub" />
        <StatCountCard label="Zero Activity" value={data.summary.zero_activity ?? 0} description="Applications with no publish or subscribe connections, potentially indicating stale or misconfigured components." formula="pub = 0 AND sub = 0" />
      </div>
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
      {data.outlier_indices && data.outlier_indices.length > 0 && (
        <OutlierTable
          title="High I/O Outliers"
          headers={["Application", "Publishes", "Subscribes", "I/O Load"]}
          rows={data.outlier_indices.map((i) => [
            data.labels[i],
            data.pubs[i],
            data.subs[i],
            data.io_load[i],
          ])}
        />
      )}
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
      <div className="grid grid-cols-[repeat(auto-fit,minmax(220px,1fr))] gap-4">
        <PrimaryStatsCard 
          summary={data.summary} 
          prefix="pub" 
          label="Publishers" 
          description="Number of applications publishing to each topic."
          formula="count(publishers)"
        />
        <PrimaryStatsCard 
          summary={data.summary} 
          prefix="sub" 
          label="Subscribers" 
          description="Number of applications subscribing to each topic."
          formula="count(subscribers)"
        />
        <PrimaryStatsCard 
          summary={data.summary} 
          prefix="fanout" 
          label="Fanout" 
          description="Message multiplication factor per topic. High fanout amplifies failure blast radius."
          formula="pub_count × sub_count"
        />
        <StatCountCard
          label="Outliers"
          value={data.summary.outlier_count ?? 0}
          description="Topics whose fanout exceeds the IQR upper fence."
        />
      </div>
      <div className="grid grid-cols-[repeat(auto-fit,minmax(180px,1fr))] gap-3">
        <StatCountCard label="Total Topics" value={data.summary.total_topics ?? 0} description="Total number of topics in the system." formula="count(topics)" />
        <StatCountCard label="1→N (Broadcast)" value={data.summary.one_to_many ?? 0} description="Single publisher, multiple subscribers. Publisher failure silences all downstream consumers." formula="pub = 1 AND sub > 1" />
        <StatCountCard label="N→1 (Aggregator)" value={data.summary.many_to_one ?? 0} description="Multiple publishers, single subscriber. Common in data aggregation or logging patterns." formula="pub > 1 AND sub = 1" />
        <StatCountCard label="N→N (Mesh)" value={data.summary.many_to_many ?? 0} description="Multiple publishers and multiple subscribers. Highly interconnected communication patterns." formula="pub > 1 AND sub > 1" />
        <StatCountCard label="Orphan" value={data.summary.orphan ?? 0} description="Topics missing a publisher or subscriber, indicating incomplete message flows." formula="pub = 0 OR sub = 0" />
      </div>
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
      {data.outlier_indices && data.outlier_indices.length > 0 && (
        <OutlierTable
          title="Fanout Outliers"
          headers={["Topic", "Publishers", "Subscribers", "Fanout"]}
          rows={data.outlier_indices.map((i) => [
            data.labels[i],
            data.pubs[i],
            data.subs[i],
            data.fanout[i],
          ])}
        />
      )}
    </div>
  )
}


function HeatmapSection({ data, title, modeToggle }: {
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
}) {
  const [showKb, setShowKb] = useState(false)
  const [search, setSearch] = useState("")
  const [selectedCell, setSelectedCell] = useState<{ rowLabel: string; colLabel: string; topics: string[] } | null>(null)
  const { resolvedTheme } = useTheme()
  const isDark = resolvedTheme !== "light"

  // Theme-aware color tokens
  const axisColor     = isDark ? "#94a3b8" : "#64748b"
  const tooltipBg     = isDark ? "#1c1c1e" : "#ffffff"
  const tooltipBorder = isDark ? "#3f3f46" : "#e4e4e7"
  const tooltipText   = isDark ? "#fafafa" : "#09090b"
  const tooltipMuted  = isDark ? "rgba(250,250,250,0.7)" : "rgba(9,9,11,0.7)"
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
      textStyle: { color: tooltipText, fontSize: 12 },
      confine: true,            // Keep tooltip within chart bounds
      hideDelay: 200,           // 200ms delay before hiding (allows scroll time)
      formatter: (params: { data: [number, number, number] }) => {
        const [ci, ri, val] = params.data
        const rLabel = labels[ri] ?? ri
        const cLabel = labels[ci] ?? ci
        const fmtVal = fmtCellVal(val) || "0"
        let topicCount = 0
        if (ids && perEntity && val > 0) {
          const rowId = ids[ri]
          const colId = ids[ci]
          const colSubIds = new Set(perEntity[colId]?.sub_topics.map((t) => t.id) ?? [])
          const shared = (perEntity[rowId]?.pub_topics ?? []).filter((t) => colSubIds.has(t.id))
          topicCount = shared.length
        }
        // Show click hint for all cells with topics
        const clickHint = topicCount > 0 ? `<br/><span style="color:${tooltipMuted};font-size:9px;font-style:italic">Click to view topics</span>` : ""
        return `<div style="font-size:12px;line-height:1.7;color:${tooltipText}"><b>${rLabel} → ${cLabel}</b><br/><span style="color:${tooltipMuted}">${fmtVal}</span>${clickHint}</div>`
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

  const handleHeatmapClick = (params: any) => {
    // Check if it's a heatmap cell click (seriesType will be "heatmap")
    if (params.seriesType !== "heatmap" || !params.data) return
    const [ci, ri, val] = params.data
    if (val <= 0 || !ids || !perEntity) return

    // ri and ci are indices into the filtered/current view
    const rowId = ids[ri]
    const colId = ids[ci]

    if (!rowId || !colId) return

    const colSubIds = new Set(perEntity[colId]?.sub_topics.map((t) => t.id) ?? [])
    const shared = (perEntity[rowId]?.pub_topics ?? []).filter((t) => colSubIds.has(t.id))

    if (shared.length > 0) {
      const rLabel = labels[ri] ?? ri
      const cLabel = labels[ci] ?? ci
      setSelectedCell({
        rowLabel: rLabel,
        colLabel: cLabel,
        topics: shared.map(t => t.name),
      })
    }
  }

  const heatmapEvents = {
    click: handleHeatmapClick,
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-[repeat(auto-fit,minmax(220px,1fr))] gap-4">
        <PrimaryStatsCard 
          summary={data.summary} 
          prefix="cell" 
          label="Connections" 
          description="Number of topics shared between entity pairs."
          formula="count(shared_topics)"
        />
        <StatCountCard label="Entities" value={data.summary.entity_count ?? 0} description="Total number of physical nodes or segments." formula="count(entities)" />
        <StatCountCard label="Active Cells" value={data.summary.nonzero_count ?? 0} description="Entity pairs that exchange at least one topic." formula="count(nonzero_cells)" />
        <StatCountCard label="Active %" value={`${Number(data.summary.active_pct ?? 0).toFixed(1)}%`} description="Fraction of total possible entity pairs that are actively communicating." formula="nonzero_cells / total_cells × 100" />
        <StatCountCard label="Intra-entity" value={data.summary.intra_total ?? 0} description="Topics where publisher and subscriber reside on the same entity." formula="Σ matrix[i][i]" />
        <StatCountCard label="Inter-entity" value={data.summary.inter_total ?? 0} description="Topics that cross entity boundaries, indicating interdependency." formula="Σ matrix[i][j] for i ≠ j" />
        <StatCountCard label="Outliers" value={data.summary.outlier_count ?? 0} description="Entity pairs whose connection count exceeds the IQR upper fence." formula="count > Q3 + 1.5 × IQR" />
      </div>
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

      {/* Topics detail modal */}
      {selectedCell && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4" onClick={() => setSelectedCell(null)}>
          <div className="bg-background border border-border rounded-lg shadow-lg max-w-3xl w-full h-[90vh] flex flex-col" onClick={(e) => e.stopPropagation()}>
            <div className="px-4 py-3 border-b border-border flex items-center justify-between flex-shrink-0">
              <h2 className="font-semibold text-sm">
                {selectedCell.rowLabel} → {selectedCell.colLabel}
              </h2>
              <button
                onClick={() => setSelectedCell(null)}
                className="text-muted-foreground hover:text-foreground transition-colors cursor-pointer text-lg"
              >
                ✕
              </button>
            </div>
            <div className="flex-1 min-h-0 overflow-y-auto p-4 space-y-1">
              {selectedCell.topics.length === 0 ? (
                <div className="text-xs text-muted-foreground">No topics found</div>
              ) : (
                selectedCell.topics.map((topicName, idx) => (
                  <div
                    key={idx}
                    className="font-mono text-xs p-2 rounded bg-muted/50 hover:bg-muted transition-colors break-all"
                    title={topicName}
                  >
                    {topicName}
                  </div>
                ))
              )}
            </div>
            <div className="px-4 py-3 border-t border-border text-xs text-muted-foreground text-right flex-shrink-0">
              {selectedCell.topics.length} topic{selectedCell.topics.length !== 1 ? 's' : ''}
            </div>
          </div>
        </div>
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
      <div className="grid grid-cols-[repeat(auto-fit,minmax(220px,1fr))] gap-4">
        <PrimaryStatsCard 
          summary={data.summary} 
          prefix="load" 
          label="Node Load" 
          description="Total combined publish and subscribe connections hosted per physical node."
          formula="Σ pub_count + Σ sub_count"
        />
        <StatCountCard label="Total Nodes" value={data.summary.node_count ?? 0} description="Total number of physical nodes in the system." formula="count(nodes)" />
        <StatCountCard label="Total Pub" value={data.summary.pub_total ?? 0} description="Aggregate number of publish connections across all nodes." formula="Σ pub_count" />
        <StatCountCard label="Total Sub" value={data.summary.sub_total ?? 0} description="Aggregate number of subscribe connections across all nodes." formula="Σ sub_count" />
        <StatCountCard label="Load Variation (CV)" value={`${Number(data.summary.cv ?? 0).toFixed(1)}%`} description="Coefficient of variation. High CV means uneven workload distribution across hosts." formula="std(load) / mean(load) × 100" />
        <StatCountCard label="Zero Load" value={data.summary.zero_load ?? 0} description="Nodes hosting no communicating applications, potentially indicating orphaned infrastructure." formula="count(load = 0)" />
        <StatCountCard label="Outliers" value={data.summary.outlier_count ?? 0} description="Nodes whose total load exceeds the IQR upper fence." formula="load > Q3 + 1.5 × IQR" />
      </div>
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
      <div className="grid grid-cols-[repeat(auto-fit,minmax(220px,1fr))] gap-4">
        <PrimaryStatsCard 
          summary={data.summary} 
          prefix="crit_io" 
          label="Critical I/O" 
          description="Total pub/sub connections per critical application. High values compound failure impact."
          formula="pub + sub for critical apps"
        />
        <PrimaryStatsCard 
          summary={data.summary} 
          prefix="norm_io" 
          label="Normal I/O" 
          description="Total pub/sub connections per non-critical application, serving as the baseline."
          formula="pub + sub for normal apps"
        />
        <StatCountCard label="Total Apps" value={data.summary.total_apps ?? 0} description="Total number of applications in the system." formula="count(apps)" />
        <StatCountCard label="Critical" value={data.summary.crit_count ?? 0} description="Applications flagged as mission-critical." formula="count(critical = true)" />
        <StatCountCard label="Critical %" value={`${Number(data.summary.crit_pct ?? 0).toFixed(1)}%`} description="Share of applications flagged as mission-critical. Higher fraction reduces fault-tolerance margin." formula="critical_apps / total_apps × 100" />
        <StatCountCard label="Crit/Normal Ratio" value={Number(data.summary.crit_norm_ratio ?? 0).toFixed(2)} description="How much heavier critical apps' average I/O load is compared to normal apps." formula="mean_io(critical) / mean_io(normal)" />
        <StatCountCard label="Outliers" value={data.summary.outlier_count ?? 0} description="Critical applications whose I/O load exceeds the IQR upper fence." formula="io > Q3 + 1.5 × IQR" />
      </div>
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
      {data.outliers && data.outliers.length > 0 && (
        <OutlierTable
          title="Critical I/O Outliers"
          headers={["Application", "Publishes", "Subscribes", "I/O Load"]}
          rows={data.outliers.map(([label, p, s, io]) => [label, p, s, io])}
        />
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
      <div className="grid grid-cols-[repeat(auto-fit,minmax(220px,1fr))] gap-4">
        <PrimaryStatsCard 
          summary={data.summary} 
          prefix="in" 
          label="In-Degree" 
          description="Number of applications depending on each library. High in-degree indicates a shared-fate risk."
          formula="count(apps depending on library)"
        />
        <PrimaryStatsCard 
          summary={data.summary} 
          prefix="out" 
          label="Out-Degree" 
          description="Number of libraries each application depends on."
          formula="count(libraries used by app)"
        />
        <StatCountCard label="Total Relations" value={data.summary.total_relations ?? 0} description="Total application-to-library USES edges in the system." formula="count(USES relationships)" />
        <StatCountCard label="Active Entities" value={data.summary.active_count ?? 0} description="Total number of apps and libraries involved in dependencies." formula="count(apps + libs with degree > 0)" />
        <StatCountCard label="Apps" value={data.summary.app_count ?? 0} description="Number of applications with library dependencies." formula="count(apps)" />
        <StatCountCard label="Libraries" value={data.summary.lib_count ?? 0} description="Number of libraries depended upon by applications." formula="count(libs)" />
        <StatCountCard label="Outliers" value={data.summary.outlier_count ?? 0} description="Libraries whose in-degree exceeds the IQR upper fence." formula="in_degree > Q3 + 1.5 × IQR" />
      </div>
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
      <div className="grid grid-cols-[repeat(auto-fit,minmax(220px,1fr))] gap-4">
        <PrimaryStatsCard 
          summary={data.summary} 
          prefix="crit_per_node" 
          label="Critical per Node" 
          description="Number of critical applications hosted on each physical node. High concentration creates blast-radius hotspots."
          formula="count(critical apps) per node"
        />
        <PrimaryStatsCard 
          summary={data.summary} 
          prefix="norm_per_node" 
          label="Normal per Node" 
          description="Number of non-critical applications hosted on each physical node."
          formula="count(normal apps) per node"
        />
        <StatCountCard label="Total Nodes" value={data.summary.node_count ?? 0} description="Total number of physical nodes in the system." formula="count(nodes)" />
        <StatCountCard label="Total Critical" value={data.summary.total_crit ?? 0} description="Total number of critical applications across all nodes." formula="Σ critical_count" />
        <StatCountCard label="Total Normal" value={data.summary.total_norm ?? 0} description="Total number of non-critical applications across all nodes." formula="Σ normal_count" />
        <StatCountCard label="System Critical %" value={`${Number(data.summary.system_crit_pct ?? 0).toFixed(1)}%`} description="Percentage of all applications marked critical. High values reduce redundancy headroom." formula="total_crit / total_all × 100" />
        <StatCountCard label="No Critical" value={data.summary.zero_crit ?? 0} description="Physical nodes hosting no critical applications, having lower individual failure impact." formula="count(critical_count = 0)" />
      </div>
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
      <div className="grid grid-cols-[repeat(auto-fit,minmax(220px,1fr))] gap-4">
        <PrimaryStatsCard 
          summary={data.summary} 
          prefix="app" 
          label="Apps per Segment" 
          description="Number of applications per segment. Low values may indicate fragmentation; high values may signal monolithic segments."
          formula="count(apps) per segment"
        />
        <PrimaryStatsCard 
          summary={data.summary} 
          prefix="topic" 
          label="Topics per Segment" 
          description="Number of topics owned or used per segment, reflecting communication surface exposure."
          formula="count(topics) per segment"
        />
        <PrimaryStatsCard 
          summary={data.summary} 
          prefix="io" 
          label="I/O per Segment" 
          description="Pub/sub message load aggregated per segment. High I/O segments are communication hubs."
          formula="Σ pub + Σ sub per segment"
        />
        <StatCountCard label="Segments" value={data.summary.css_count ?? 0} description="Total number of architectural segments (CSS) in the system." formula="count(segments)" />
      </div>
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

// ── Network Usage Section ─────────────────────────────────────────────────

type NetworkUsageMode = "total" | "out" | "in"
type NetworkUsageView = "nodes" | "apps" | "topics"
const NETWORK_TABLE_PAGE_SIZE = 10

function NetworkUsageSection({ data }: { data: ExtrasStats["network_usage"] }) {
  const [view, setView] = useState<NetworkUsageView>("nodes")
  const [mode, setMode] = useState<NetworkUsageMode>("total")
  const [tablePage, setTablePage] = useState(0)

  const nodeItems = useMemo(() => (!data ? [] : data.sorted_labels.map((label, i) => ({
    name: label,
    id: data.sorted_ids?.[i],
    outbound: data.sorted_out[i],
    inbound: data.sorted_in[i],
    total: data.sorted_total[i],
  })).sort((a, b) => {
    if (mode === "out") return b.outbound - a.outbound
    if (mode === "in") return b.inbound - a.inbound
    return b.total - a.total
  })), [data, mode]) // eslint-disable-line react-hooks/exhaustive-deps

  const appItems = useMemo(() => (!data?.app_bandwidth ? [] : data.app_bandwidth.map((a) => ({
    name: a.name,
    id: a.id,
    outbound: a.bw_out,
    inbound: a.bw_in,
    total: a.bw_total,
    node_name: a.node_name,
    role: a.role,
    criticality: a.criticality,
  })).sort((a, b) => {
    if (mode === "out") return b.outbound - a.outbound
    if (mode === "in") return b.inbound - a.inbound
    return b.total - a.total
  })), [data, mode]) // eslint-disable-line react-hooks/exhaustive-deps

  const topicItems = useMemo(() => (!data?.topic_bandwidth ? [] : data.topic_bandwidth.map((t) => ({
    name: t.name,
    id: t.id,
    outbound: t.bw_out,
    inbound: t.bw_in,
    total: t.bw_total,
    frequency_hz: t.frequency_hz,
    size_bytes: t.size_bytes,
    pub_count: t.pub_count,
    sub_count: t.sub_count,
  })).sort((a, b) => {
    if (mode === "out") return b.outbound - a.outbound
    if (mode === "in") return b.inbound - a.inbound
    return b.total - a.total
  })), [data, mode]) // eslint-disable-line react-hooks/exhaustive-deps

  const allItems = view === "nodes" ? nodeItems : view === "apps" ? appItems : topicItems
  const { search, handleSearch, filtered } = useFilteredSearch(allItems)

  // Reset to first page whenever the active view or search term changes
  useEffect(() => { setTablePage(0) }, [view, search])

  if (!data) return null

  const totalBw = Number(data.summary.total_bandwidth ?? 0)
  const totalOut = Number(data.summary.total_outbound ?? 0)
  const totalIn = Number(data.summary.total_inbound ?? 0)

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <PrimaryStatsCard 
          summary={data.summary} 
          prefix="bw" 
          label="Bandwidth per Node" 
          description="Sustained network bandwidth (bytes/s) consumed by the topology per physical node."
          format={fmtBytes}
          formula="Σ size × freq × (pub + sub) per node"
        />
        <StatCountCard label="Total Bandwidth" value={fmtBytes(totalBw) + "/s"} description="Aggregate sustained bandwidth flowing across the entire network." formula="Σ size(t) × freq(t) × (pub + sub)" />
        <StatCountCard label="Total Outbound" value={fmtBytes(totalOut) + "/s"} description="Sustained bytes per second produced by all application publishers." formula="Σ size(t) × freq(t) × pub_count(t)" />
        <StatCountCard label="Total Inbound" value={fmtBytes(totalIn) + "/s"} description="Sustained bytes per second consumed by all application subscribers." formula="Σ size(t) × freq(t) × sub_count(t)" />
        <StatCountCard label="System Size" value={`${data.summary.node_count ?? 0} Nodes, ${data.summary.topic_count ?? 0} Topics`} description="Total number of physical nodes and topics in the system topology." formula="count(nodes + topics)" />
        <StatCountCard label="Load Variation (CV)" value={`${Number(data.summary.cv ?? 0).toFixed(1)}%`} description="Coefficient of variation of node bandwidth. High CV means uneven network load distribution." formula="std(bw) / mean(bw) × 100" />
        <StatCountCard label="Zero-BW Nodes" value={data.summary.zero_bw_nodes ?? 0} description="Nodes with no network traffic, potentially indicating orphaned infrastructure." formula="count(bw = 0)" />
        <StatCountCard label="Outliers" value={data.summary.outlier_count ?? 0} description="Nodes whose total bandwidth exceeds the IQR upper fence." formula="bw > Q3 + 1.5 × IQR" />
      </div>
      <Card className="bg-background pb-3">
        <CardHeader>
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <div className="flex items-center gap-3">
              <CardTitle className="text-[11px] text-muted-foreground uppercase tracking-widest">
                {view === "nodes" ? "Network Usage per Node" : view === "apps" ? "Network Usage per App" : "Network Usage per Topic"}
              </CardTitle>
              <div className="flex items-center gap-1 rounded-md border p-0.5 bg-muted/50">
                {(["nodes", "apps", "topics"] as NetworkUsageView[]).map((v) => (
                  <button key={v} onClick={() => setView(v)}
                    className={`px-2.5 py-0.5 text-xs rounded transition-colors ${view === v ? "bg-background shadow font-medium" : "text-muted-foreground hover:text-foreground"}`}>
                    {v === "nodes" ? "Nodes" : v === "apps" ? "Apps" : "Topics"}
                  </button>
                ))}
              </div>
            </div>
            <div className="flex items-center gap-2">
              <ChartSearchBar search={search} onSearch={handleSearch} count={filtered.length} total={allItems.length} />
              <div className="flex items-center gap-1 rounded-md border p-0.5 bg-muted/50">
                {(["total", "out", "in"] as NetworkUsageMode[]).map((m) => (
                  <button key={m} onClick={() => setMode(m)}
                    className={`px-2.5 py-0.5 text-xs rounded transition-colors ${mode === m ? "bg-background shadow font-medium" : "text-muted-foreground hover:text-foreground"}`}>
                    {m === "total" ? "Total" : m === "out" ? "Outbound" : "Inbound"}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {mode === "total" ? (
            <EBarChart
              items={filtered}
              series={[
                { key: "outbound", label: "Outbound", color: "#818cf8", stack: "bw", fmt: fmtBytes },
                { key: "inbound", label: "Inbound", color: "#34d399", stack: "bw", fmt: fmtBytes },
              ]}
              onClickId={goToExplorer}
            />
          ) : mode === "out" ? (
            <EBarChart
              items={filtered}
              series={[{ key: "outbound", label: "Outbound", color: "#818cf8", fmt: fmtBytes }]}
              onClickId={goToExplorer}
            />
          ) : (
            <EBarChart
              items={filtered}
              series={[{ key: "inbound", label: "Inbound", color: "#34d399", fmt: fmtBytes }]}
              onClickId={goToExplorer}
            />
          )}
        </CardContent>
      </Card>
      {view === "apps" && appItems.length > 0 && (() => {
        const totalPages = Math.max(1, Math.ceil(filtered.length / NETWORK_TABLE_PAGE_SIZE))
        const page = Math.min(tablePage, totalPages - 1)
        const pageItems = (filtered as typeof appItems).slice(page * NETWORK_TABLE_PAGE_SIZE, (page + 1) * NETWORK_TABLE_PAGE_SIZE)
        return (
          <div>
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">App Bandwidth Detail</p>
            <div className="max-h-96 overflow-auto rounded-lg border border-border">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground uppercase tracking-wide">Application</th>
                    <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground uppercase tracking-wide">Node</th>
                    <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground uppercase tracking-wide">Role</th>
                    <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground uppercase tracking-wide">Outbound/s</th>
                    <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground uppercase tracking-wide">Inbound/s</th>
                    <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground uppercase tracking-wide">Total/s</th>
                  </tr>
                </thead>
                <tbody>
                  {pageItems.map((a) => (
                    <tr key={a.id} className="border-b border-border/50 hover:bg-muted/30 cursor-pointer transition-colors" onClick={() => goToExplorer(a.id)}>
                      <td className="px-3 py-2 text-left font-medium">
                        {a.criticality && <span className="mr-1 text-amber-400" title="Critical">●</span>}
                        {a.name}
                      </td>
                      <td className="px-3 py-2 text-right font-mono text-xs text-muted-foreground truncate max-w-[120px]">{a.node_name ?? "—"}</td>
                       <td className="px-3 py-2 text-right font-mono text-xs text-muted-foreground">{(a.role && a.role.length > 0) ? a.role.join(", ") : "—"}</td>
                      <td className="px-3 py-2 text-right font-mono text-xs text-muted-foreground">{fmtBytes(a.outbound)}/s</td>
                      <td className="px-3 py-2 text-right font-mono text-xs text-muted-foreground">{fmtBytes(a.inbound)}/s</td>
                      <td className="px-3 py-2 text-right font-mono text-xs text-muted-foreground">{fmtBytes(a.total)}/s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="px-3 py-2">
                <TablePager page={page} totalPages={totalPages} total={filtered.length} pageSize={NETWORK_TABLE_PAGE_SIZE} label="applications" onPage={setTablePage} />
              </div>
            </div>
          </div>
        )
      })()}
      {view === "topics" && topicItems.length > 0 && (() => {
        const totalPages = Math.max(1, Math.ceil(filtered.length / NETWORK_TABLE_PAGE_SIZE))
        const page = Math.min(tablePage, totalPages - 1)
        const pageItems = (filtered as typeof topicItems).slice(page * NETWORK_TABLE_PAGE_SIZE, (page + 1) * NETWORK_TABLE_PAGE_SIZE)
        return (
          <div>
            <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">Topic Bandwidth Detail</p>
            <div className="max-h-96 overflow-auto rounded-lg border border-border">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border">
                    <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground uppercase tracking-wide">Topic</th>
                    <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground uppercase tracking-wide">Freq (Hz)</th>
                    <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground uppercase tracking-wide">Size</th>
                    <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground uppercase tracking-wide">Pubs</th>
                    <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground uppercase tracking-wide">Subs</th>
                    <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground uppercase tracking-wide">Outbound/s</th>
                    <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground uppercase tracking-wide">Inbound/s</th>
                    <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground uppercase tracking-wide">Total/s</th>
                  </tr>
                </thead>
                <tbody>
                  {pageItems.map((t) => (
                    <tr key={t.id} className="border-b border-border/50 hover:bg-muted/30 cursor-pointer transition-colors" onClick={() => goToExplorer(t.id)}>
                      <td className="px-3 py-2 text-left font-medium truncate max-w-[180px]">{t.name}</td>
                      <td className="px-3 py-2 text-right font-mono text-xs text-muted-foreground">{t.frequency_hz.toFixed(1)}</td>
                      <td className="px-3 py-2 text-right font-mono text-xs text-muted-foreground">{fmtBytes(t.size_bytes)}</td>
                      <td className="px-3 py-2 text-right font-mono text-xs text-muted-foreground">{t.pub_count}</td>
                      <td className="px-3 py-2 text-right font-mono text-xs text-muted-foreground">{t.sub_count}</td>
                      <td className="px-3 py-2 text-right font-mono text-xs text-muted-foreground">{fmtBytes(t.outbound)}/s</td>
                      <td className="px-3 py-2 text-right font-mono text-xs text-muted-foreground">{fmtBytes(t.inbound)}/s</td>
                      <td className="px-3 py-2 text-right font-mono text-xs text-muted-foreground">{fmtBytes(t.total)}/s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="px-3 py-2">
                <TablePager page={page} totalPages={totalPages} total={filtered.length} pageSize={NETWORK_TABLE_PAGE_SIZE} label="topics" onPage={setTablePage} />
              </div>
            </div>
          </div>
        )
      })()}
      {data.outliers.length > 0 && (
        <OutlierTable
          title="High-Bandwidth Nodes"
          headers={["Node", "Outbound", "Inbound", "Total"]}
          rows={data.outliers.map(([label, out, inn, tot]) => [label, fmtBytes(out), fmtBytes(inn), fmtBytes(tot)])}
        />
      )}
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
      <div className="grid grid-cols-[repeat(auto-fit,minmax(220px,1fr))] gap-4">
        <PrimaryStatsCard 
          summary={s} 
          prefix="score" 
          label="Bottleneck Score" 
          description="Composite structural score identifying highest-risk single points of failure."
          formula={`${wNorm.bt.toFixed(2)}·BT + ${wNorm.ap.toFixed(2)}·AP + ${wNorm.br.toFixed(2)}·BR + ${wNorm.bridge.toFixed(2)}·bridge`}
        />
        <StatCountCard label="Articulation Points" value={s.articulation_point_count ?? 0} description="Components whose removal disconnects the undirected graph. Structural SPOFs." formula="is_articulation_point = True" />
        <StatCountCard label="Directed APs" value={s.directed_ap_count ?? 0} description="Components that disconnect the directed reachable set. Directional SPOFs." formula="is_directed_ap = True" />
        <StatCountCard label="Score Outliers" value={recomputedOutlierCount} description="Components whose bottleneck score exceeds the IQR upper fence." formula="score > Q3 + 1.5 × IQR" />
        <StatCountCard label="Total Components" value={s.total ?? 0} description="Total number of components analyzed for bottleneck risk." formula="count(components)" />
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
      {allItems.filter((it) => it.outlier).length > 0 && (
        <OutlierTable
          title="Bottleneck Score Outliers"
          headers={["Component", "Type", "Score", "Betweenness", "Blast Radius", "Bridge Ratio"]}
          rows={allItems.filter((it) => it.outlier).map((it) => [
            it.name,
            it.type,
            fmtNum(it.bottleneck_score),
            fmtNum(it.betweenness),
            it.blast_radius,
            fmtNum(it.bridge_ratio),
          ])}
        />
      )}
    </div>
  )
}

function QoSDistributionSection({ data }: { data: ExtrasStats["qos_distribution"] }) {
  if (!data) return null

  const { resolvedTheme } = useTheme()
  const isDark = resolvedTheme !== "light"

  const COLORS = ["#818cf8", "#34d399", "#fbbf24", "#fb7185", "#a78bfa", "#2dd4bf", "#f472b6", "#60a5fa", "#38bdf8", "#facc15"]

  const renderChart = (title: string, counts: Record<string, number>) => {
    const items = Object.entries(counts).map(([name, value]) => ({ name, value }))
    const option = {
      tooltip: {
        trigger: "item" as const,
        formatter: "{b}: {c} ({d}%)",
        backgroundColor: isDark ? "#1c1c1e" : "#ffffff",
        borderColor: isDark ? "#3f3f46" : "#e4e4e7",
        textStyle: { color: isDark ? "#fafafa" : "#09090b", fontSize: 12 },
      },
      legend: {
        orient: "vertical" as const,
        left: "left",
        textStyle: { color: isDark ? "#e2e8f0" : "#1e293b" },
      },
      series: [
        {
          name: title,
          type: "pie" as const,
          radius: "50%",
          data: items.map((item, index) => ({
            value: item.value,
            name: item.name,
            itemStyle: { color: COLORS[index % COLORS.length] },
          })),
          emphasis: {
            itemStyle: {
              shadowBlur: 10,
              shadowOffsetX: 0,
              shadowColor: "rgba(0, 0, 0, 0.5)",
            },
          },
        },
      ],
    }

    return (
      <Card className="bg-background">
        <CardHeader>
          <CardTitle className="text-[11px] text-muted-foreground uppercase tracking-widest">{title}</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="h-64">
              <ReactECharts option={option} notMerge={true} style={{ height: "100%", width: "100%" }} />
            </div>
            <div className="space-y-2">
              {items.map((item, index) => (
                <div key={item.name} className="flex justify-between items-center p-2 rounded bg-muted/50">
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-sm" style={{ backgroundColor: COLORS[index % COLORS.length] }} />
                    <span className="text-sm font-medium">{item.name}</span>
                  </div>
                  <span className="text-sm font-mono">{item.value}</span>
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-[repeat(auto-fit,minmax(220px,1fr))] gap-4">
        <StatCountCard label="Total Topics" value={data.total_topics ?? 0} description="Total number of topics in the system." formula="count(topics)" />
        <StatCountCard label="Durability Variants" value={Object.keys(data.durability || {}).length} description="Number of unique QoS durability values." formula="count(distinct durability)" />
        <StatCountCard label="Reliability Variants" value={Object.keys(data.reliability || {}).length} description="Number of unique QoS reliability values." formula="count(distinct reliability)" />
        <StatCountCard label="Transport Priority Variants" value={Object.keys(data.transport_priority || {}).length} description="Number of unique QoS transport priority values." formula="count(distinct transport_priority)" />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {renderChart("QoS Durability Distribution", data.durability || {})}
        {renderChart("QoS Reliability Distribution", data.reliability || {})}
        {renderChart("QoS Transport Priority Distribution", data.transport_priority || {})}
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
  { id: "network_usage", label: "Network Usage", icon: Wifi, color: "text-sky-500", description: "Sustained network bandwidth (bytes/s) consumed by the topology. Shows per-node outbound and inbound B/s load, factoring in topic frequency, so you can spot hosts that dominate raw network traffic." },
  { id: "qos_distribution", label: "QoS Distribution", icon: BarChart3, color: "text-indigo-500", description: "Distribution of QoS durability, reliability, and transport priority across all topics. Visualizes the policy landscape." },
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
  network_usage: "network_usage",
  qos_distribution: "qos_distribution",
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
      <AppLayout title="Statistics" description="System metrics across components">
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
      <AppLayout title="Statistics" description="System metrics across components">
        <NoConnectionInfo description="Connect to your Neo4j database to view statistics" />
      </AppLayout>
    )
  }

  return (
    <AppLayout title="Statistics" description="System metrics across components">
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
                      {selectedSection === "cross_node" && <HeatmapSection data={tabData.cross_node_heatmap} title="Cross-Node" modeToggle />}
                      {selectedSection === "node_load" && <NodeCommLoadSection data={tabData.node_comm_load} />}
                      {selectedSection === "domain_comm" && <HeatmapSection data={tabData.domain_comm} title="Segment Communication" modeToggle />}
                      {selectedSection === "criticality" && <CriticalityIOSection data={tabData.criticality_io} />}
                      {selectedSection === "lib_deps" && <LibDependencySection data={tabData.lib_dependency} />}
                      {selectedSection === "node_density" && <NodeCriticalDensitySection data={tabData.node_critical_density} />}
                      {selectedSection === "domain_div" && <DomainDiversitySection data={tabData.domain_diversity} />}
                      {selectedSection === "bottleneck" && <BottleneckSection data={tabData.bottleneck} />}
                      {selectedSection === "network_usage" && <NetworkUsageSection data={tabData.network_usage} />}
                      {selectedSection === "qos_distribution" && <QoSDistributionSection data={tabData.qos_distribution} />}
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
