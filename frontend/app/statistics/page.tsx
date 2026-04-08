"use client"

import { useEffect, useState, type ReactElement } from "react"
import { useRouter } from "next/navigation"
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

// ── Types ──────────────────────────────────────────────────────────────

interface SummaryDict { [key: string]: number | string }

interface ExtrasStats {
  topic_bandwidth?: {
    labels: string[]
    ids: string[]
    sizes: number[]
    subs: number[]
    bandwidth: number[]
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
    summary: SummaryDict
    outlier_pairs: [string, string, number, number][]
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

// ── Sections ────────────────────────────────────────────────────────────

function TopicBandwidthSection({ data }: { data: ExtrasStats["topic_bandwidth"] }) {
  if (!data) return null
  const chartData = data.labels.map((label, i) => ({
    name: truncate(label),
    id: data.ids?.[i],
    bandwidth: data.bandwidth[i],
  })).sort((a, b) => b.bandwidth - a.bandwidth)
  const total = chartData.length
  const capped = chartData.slice(0, MAX_ITEMS)

  return (
    <div className="space-y-4">
      <SummaryCards summary={data.summary} keys={[
        { key: "total_topics", label: "Total Topics" },
        { key: "size_mean", label: "Avg Size" },
        { key: "sub_mean", label: "Avg Subscribers" },
        { key: "outlier_count", label: "Outliers" },
      ]} />
      <Card>
        <CardHeader><CardTitle className="text-base">Topic Bandwidth (Size × Subscribers){total > MAX_ITEMS && <span className="text-xs font-normal text-muted-foreground ml-2">showing {MAX_ITEMS} of {total}</span>}</CardTitle></CardHeader>
        <CardContent>
          <SizedBarChart dataCount={capped.length} config={{ bandwidth: { label: "Bandwidth", color: "#8b5cf6" } }}>
            <BarChart data={capped} margin={{ bottom: 50 }} maxBarSize={48} onClick={handleBarClick} className="cursor-pointer">
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
  if (!data) return null
  const chartData = data.labels.map((label, i) => ({
    name: truncate(label),
    id: data.ids?.[i],
    publishes: data.pubs[i],
    subscribes: data.subs[i],
  })).sort((a, b) => (b.publishes + b.subscribes) - (a.publishes + a.subscribes))
  const total = chartData.length
  const capped = chartData.slice(0, MAX_ITEMS)

  return (
    <div className="space-y-4">
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
        <CardHeader><CardTitle className="text-base">Application Pub/Sub Balance{total > MAX_ITEMS && <span className="text-xs font-normal text-muted-foreground ml-2">showing {MAX_ITEMS} of {total}</span>}</CardTitle></CardHeader>
        <CardContent>
          <SizedBarChart dataCount={capped.length} config={{
            publishes: { label: "Publishes", color: "#3b82f6" },
            subscribes: { label: "Subscribes", color: "#10b981" },
          }}>
            <BarChart data={capped} margin={{ bottom: 50 }} maxBarSize={48} onClick={handleBarClick} className="cursor-pointer">
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
  if (!data) return null
  const chartData = data.labels.map((label, i) => ({
    name: truncate(label),
    id: data.ids?.[i],
    fanout: data.fanout[i],
  })).sort((a, b) => b.fanout - a.fanout)
  const total = chartData.length
  const capped = chartData.slice(0, MAX_ITEMS)

  return (
    <div className="space-y-4">
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
        <CardHeader><CardTitle className="text-base">Topic Fanout (Publishers × Subscribers){total > MAX_ITEMS && <span className="text-xs font-normal text-muted-foreground ml-2">showing {MAX_ITEMS} of {total}</span>}</CardTitle></CardHeader>
        <CardContent>
          <SizedBarChart dataCount={capped.length} config={{ fanout: { label: "Fanout", color: "#f59e0b" } }}>
            <BarChart data={capped} margin={{ bottom: 50 }} maxBarSize={48} onClick={handleBarClick} className="cursor-pointer">
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

function HeatmapSection({ data, title }: { data: { labels: string[]; node_ids?: string[]; matrix: number[][]; summary: SummaryDict; outlier_pairs: [string, string, number, number][] } | undefined; title: string }) {
  if (!data || !data.labels.length) return <p className="text-sm text-muted-foreground">Not enough data for {title}</p>

  const totalLabels = data.labels.length
  let labels = data.labels
  let matrix = data.matrix
  let ids = data.node_ids
  if (totalLabels > MAX_ITEMS) {
    const rowTotals = data.matrix.map((row) => row.reduce((s, v) => s + v, 0))
    const indices = rowTotals.map((_, i) => i).sort((a, b) => rowTotals[b] - rowTotals[a]).slice(0, MAX_ITEMS)
    const idxSet = new Set(indices)
    labels = indices.map((i) => data.labels[i])
    matrix = indices.map((ri) => data.matrix[ri].filter((_, ci) => idxSet.has(ci)))
    if (data.node_ids) ids = indices.map((i) => data.node_ids![i])
  }
  const maxVal = Math.max(1, ...matrix.flat())
  const n = labels.length
  const cellSize = n > 15 ? "min-w-[28px] h-[28px] p-0.5 text-[10px]" : n > 10 ? "min-w-[32px] h-[32px] p-1 text-[11px]" : "min-w-[40px] h-[40px] p-2 text-xs"
  const headerSize = n > 15 ? "text-[9px] p-0.5" : n > 10 ? "text-[10px] p-1" : "text-xs p-1"

  return (
    <div className="space-y-4">
      <SummaryCards summary={data.summary} keys={[
        { key: "entity_count", label: "Entities" },
        { key: "nonzero_count", label: "Active Cells" },
        { key: "active_pct", label: "Active %", format: (v) => Number(v).toFixed(1) + "%" },
        { key: "intra_total", label: "Intra-entity" },
        { key: "inter_total", label: "Inter-entity" },
        { key: "outlier_count", label: "Outliers" },
      ]} />
      <Card>
        <CardHeader><CardTitle className="text-base">{title}{totalLabels > MAX_ITEMS && <span className="text-xs font-normal text-muted-foreground ml-2">showing {MAX_ITEMS} of {totalLabels}</span>}</CardTitle></CardHeader>
        <CardContent>
          <div className="overflow-auto">
            <table className="border-collapse">
              <thead>
                <tr>
                  <th className="sticky left-0 bg-background z-10 p-2" />
                  {labels.map((l, i) => (
                    <th
                      key={i}
                      className={`align-bottom text-left ${headerSize} ${ids ? "cursor-pointer hover:underline" : ""}`}
                      style={{ writingMode: "vertical-lr" }}
                      onClick={ids ? () => goToExplorer(ids[i]) : undefined}
                    >{l}</th>
                  ))}
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
                      return (
                        <td
                          key={ci}
                          className={`text-center border border-border ${cellSize}`}
                          style={{ backgroundColor: val > 0 ? `rgba(139, 92, 246, ${0.15 + intensity * 0.75})` : undefined }}
                          title={`${rowLabel} → ${labels[ci]}: ${val}`}
                        >
                          {val > 0 ? val : ""}
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
  if (!data) return null
  const chartData = data.sorted_labels.map((label, i) => ({
    name: truncate(label),
    id: data.sorted_ids?.[i],
    publishes: data.sorted_pub[i],
    subscribes: data.sorted_sub[i],
  }))
  const total = chartData.length
  const capped = chartData.slice(0, MAX_ITEMS)

  return (
    <div className="space-y-4">
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
        <CardHeader><CardTitle className="text-base">Node Communication Load{total > MAX_ITEMS && <span className="text-xs font-normal text-muted-foreground ml-2">showing {MAX_ITEMS} of {total}</span>}</CardTitle></CardHeader>
        <CardContent>
          <SizedBarChart dataCount={capped.length} config={{
            publishes: { label: "Publishes", color: "#3b82f6" },
            subscribes: { label: "Subscribes", color: "#ec4899" },
          }}>
            <BarChart data={capped} margin={{ bottom: 50 }} maxBarSize={48} onClick={handleBarClick} className="cursor-pointer">
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
  if (!data) return null
  const critData = data.crit_labels.map((label, i) => ({
    name: truncate(label),
    id: data.crit_ids?.[i],
    publishes: data.crit_pubs[i],
    subscribes: data.crit_subs[i],
  })).sort((a, b) => (b.publishes + b.subscribes) - (a.publishes + a.subscribes))
  const total = critData.length
  const capped = critData.slice(0, MAX_ITEMS)

  return (
    <div className="space-y-4">
      <SummaryCards summary={data.summary} keys={[
        { key: "total_apps", label: "Total Apps" },
        { key: "crit_count", label: "Critical" },
        { key: "crit_pct", label: "Critical %", format: (v) => Number(v).toFixed(1) + "%" },
        { key: "crit_io_mean", label: "Crit Avg I/O" },
        { key: "norm_io_mean", label: "Normal Avg I/O" },
        { key: "crit_norm_ratio", label: "Crit/Normal Ratio" },
      ]} />
      {capped.length > 0 && (
        <Card>
          <CardHeader><CardTitle className="text-base">Critical Applications I/O{total > MAX_ITEMS && <span className="text-xs font-normal text-muted-foreground ml-2">showing {MAX_ITEMS} of {total}</span>}</CardTitle></CardHeader>
          <CardContent>
            <SizedBarChart dataCount={capped.length} config={{
              publishes: { label: "Publishes", color: "#ef4444" },
              subscribes: { label: "Subscribes", color: "#f97316" },
            }}>
              <BarChart data={capped} margin={{ bottom: 50 }} maxBarSize={48} onClick={handleBarClick} className="cursor-pointer">
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
  if (!data || !data.labels.length) return <p className="text-sm text-muted-foreground">No library dependency data</p>
  const chartData = data.labels.map((label, i) => ({
    name: truncate(label, 20),
    id: data.display_ids?.[i],
    inbound: data.in_vals[i],
    outbound: data.out_vals[i],
  }))
  const total = chartData.length
  const capped = chartData.slice(0, MAX_ITEMS)

  return (
    <div className="space-y-4">
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
        <CardHeader><CardTitle className="text-base">Library Dependency Density{total > MAX_ITEMS && <span className="text-xs font-normal text-muted-foreground ml-2">showing {MAX_ITEMS} of {total}</span>}</CardTitle></CardHeader>
        <CardContent>
          <SizedBarChart dataCount={capped.length} config={{
            inbound: { label: "In-degree (dependents)", color: "#8b5cf6" },
            outbound: { label: "Out-degree (dependencies)", color: "#14b8a6" },
          }}>
            <BarChart data={capped} margin={{ bottom: 50 }} maxBarSize={48} onClick={handleBarClick} className="cursor-pointer">
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
  if (!data) return null
  const chartData = data.sorted_labels.map((label, i) => ({
    name: truncate(label),
    id: data.sorted_ids?.[i],
    critical: data.sorted_crit[i],
    normal: data.sorted_norm[i],
  }))
  const total = chartData.length
  const capped = chartData.slice(0, MAX_ITEMS)

  return (
    <div className="space-y-4">
      <SummaryCards summary={data.summary} keys={[
        { key: "node_count", label: "Nodes" },
        { key: "total_crit", label: "Total Critical" },
        { key: "total_norm", label: "Total Normal" },
        { key: "system_crit_pct", label: "System Critical %", format: (v) => Number(v).toFixed(1) + "%" },
        { key: "crit_per_node_max", label: "Max Crit/Node" },
        { key: "zero_crit", label: "No Critical" },
      ]} />
      <Card>
        <CardHeader><CardTitle className="text-base">Node Critical Application Density{total > MAX_ITEMS && <span className="text-xs font-normal text-muted-foreground ml-2">showing {MAX_ITEMS} of {total}</span>}</CardTitle></CardHeader>
        <CardContent>
          <SizedBarChart dataCount={capped.length} config={{
            critical: { label: "Critical", color: "#ef4444" },
            normal: { label: "Normal", color: "#3b82f6" },
          }}>
            <BarChart data={capped} margin={{ bottom: 50 }} maxBarSize={48} onClick={handleBarClick} className="cursor-pointer">
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
  if (!data || !data.labels.length) return <p className="text-sm text-muted-foreground">Insufficient domain data (need ≥ 2 domains)</p>
  const chartData = data.labels.map((label, i) => ({
    name: truncate(label),
    applications: data.app_counts[i],
    topics: data.topic_counts[i],
    io: data.io_vals[i],
  }))
  const total = chartData.length
  const capped = chartData.slice(0, MAX_ITEMS)

  return (
    <div className="space-y-4">
      <SummaryCards summary={data.summary} keys={[
        { key: "css_count", label: "Domains" },
        { key: "app_mean", label: "Avg Apps/Domain" },
        { key: "app_max", label: "Max Apps" },
        { key: "topic_mean", label: "Avg Topics/Domain" },
        { key: "io_mean", label: "Avg I/O" },
        { key: "io_max", label: "Max I/O" },
      ]} />
      <Card>
        <CardHeader><CardTitle className="text-base">Domain Diversity{total > MAX_ITEMS && <span className="text-xs font-normal text-muted-foreground ml-2">showing {MAX_ITEMS} of {total}</span>}</CardTitle></CardHeader>
        <CardContent>
          <SizedBarChart dataCount={capped.length} config={{
            applications: { label: "Applications", color: "#3b82f6" },
            topics: { label: "Topics", color: "#10b981" },
            io: { label: "I/O Load", color: "#f59e0b" },
          }}>
            <BarChart data={capped} margin={{ bottom: 50 }} maxBarSize={48}>
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
  { id: "topic_bandwidth", label: "Topic Bandwidth", icon: Radio, color: "text-violet-500", description: "Data throughput per topic based on message size and subscriber count. High-bandwidth topics are potential bottlenecks." },
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

// ── Main page ───────────────────────────────────────────────────────────

export default function StatisticsPage() {
  const { status, config, initialLoadComplete } = useConnection()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [stats, setStats] = useState<ExtrasStats | null>(null)

  const isConnected = status === "connected"

  const fetchExtras = async () => {
    if (!isConnected || !config) return
    setLoading(true)
    setError(null)
    try {
      const creds = apiClient.getCredentials()
      if (!creds) throw new Error("No credentials")
      const response = await fetch(`${API_BASE_URL}/api/v1/stats/extras`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(creds),
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const json = await response.json()
      if (json.success) {
        setStats(json.stats)
      } else {
        throw new Error(json.detail || "Failed")
      }
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Failed to load statistics"
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (isConnected && config) {
      fetchExtras()
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
                    onClick={fetchExtras}
                    disabled={loading}
                    className="shrink-0"
                  >
                    <RefreshCw className={`h-4 w-4 mr-1 ${loading ? "animate-spin" : ""}`} />
                    Refresh
                  </Button>
                </div>
              </CardContent>
            </Card>

            {loading && !stats && (
              <div className="flex h-64 items-center justify-center">
                <LoadingSpinner size="lg" text="Computing statistics…" />
              </div>
            )}

            {error && (
              <Card className="border-red-500/50 bg-red-500/5">
                <CardContent className="p-4">
                  <p className="text-sm text-red-500">
                    <AlertTriangle className="inline h-4 w-4 mr-1" /> {error}
                  </p>
                </CardContent>
              </Card>
            )}

            {stats && (
              <Tabs defaultValue="topic_bandwidth" className="w-full">
                <TabsList className="flex flex-wrap h-auto gap-1 bg-muted/50 p-1">
                  {TAB_CONFIG.map(({ id, label, icon: Icon, color }) => (
                    <TabsTrigger key={id} value={id} className="text-xs gap-1 data-[state=active]:shadow-sm">
                      <Icon className={`h-3.5 w-3.5 ${color}`} />
                      {label}
                    </TabsTrigger>
                  ))}
                </TabsList>

                {TAB_CONFIG.map(({ id, description }) => (
                  <TabsContent key={id} value={id} className="mt-4">
                    <p className="text-sm text-muted-foreground mb-4">{description}</p>
                    {id === "topic_bandwidth" && <TopicBandwidthSection data={stats.topic_bandwidth} />}
                    {id === "app_balance" && <AppBalanceSection data={stats.app_balance} />}
                    {id === "topic_fanout" && <TopicFanoutSection data={stats.topic_fanout} />}
                    {id === "cross_node" && <HeatmapSection data={stats.cross_node_heatmap} title="Cross-Node Communication Heatmap" />}
                    {id === "node_load" && <NodeCommLoadSection data={stats.node_comm_load} />}
                    {id === "domain_comm" && <HeatmapSection data={stats.domain_comm} title="Domain-to-Domain Communication" />}
                    {id === "criticality" && <CriticalityIOSection data={stats.criticality_io} />}
                    {id === "lib_deps" && <LibDependencySection data={stats.lib_dependency} />}
                    {id === "node_density" && <NodeCriticalDensitySection data={stats.node_critical_density} />}
                    {id === "domain_div" && <DomainDiversitySection data={stats.domain_diversity} />}
                  </TabsContent>
                ))}
              </Tabs>
            )}
          </>
        )}
      </div>
    </AppLayout>
  )
}
