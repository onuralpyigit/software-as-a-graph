"use client"

import { useMemo } from "react"
import { useRouter } from "next/navigation"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { Skeleton } from "@/components/ui/skeleton"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import {
  Waypoints, Database, TrendingUp,
  Network, ArrowRight, Layers, BarChart3,
  ShieldAlert, CheckCircle2, Activity,
} from "lucide-react"
import { useConnection } from "@/lib/stores/connection-store"
import { useAnalysis } from "@/lib/stores/analysis-store"
import ReactECharts from "echarts-for-react"

// ── shared chart constants ────────────────────────────────────────────────────
const PALETTE   = ['#3b82f6','#a855f7','#22c55e','#f97316','#06b6d4','#6366f1','#ec4899','#14b8a6','#f59e0b','#ef4444']
const CHART_TXT = 'rgba(148,163,184,1)'
const GRID_LINE = 'rgba(148,163,184,0.08)'
const TIP_BG    = 'rgba(15,23,42,0.96)'
const TIP_BORD  = 'rgba(148,163,184,0.18)'

function formatKey(key: string): string {
  return key.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1).toLowerCase()).join(' ')
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function hBarOption(entries: [string, number][], color: string): any {
  return {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'none' },
      backgroundColor: TIP_BG,
      borderColor: TIP_BORD,
      textStyle: { color: '#e2e8f0', fontSize: 12 },
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      formatter: (p: any[]) =>
        `<span style="font-weight:600">${p[0].name}</span><br/>` +
        `<span style="color:${color};font-weight:700;font-size:14px">${(p[0].value as number).toLocaleString()}</span>`,
    },
    grid: { left: 120, right: 56, top: 16, bottom: 16, containLabel: false },
    xAxis: {
      type: 'value',
      axisLabel: { color: CHART_TXT, fontSize: 10 },
      splitLine: { lineStyle: { color: GRID_LINE } },
      axisLine: { show: false },
      axisTick: { show: false },
    },
    yAxis: {
      type: 'category',
      data: entries.map(([t]) => formatKey(t)),
      axisLabel: { color: CHART_TXT, fontSize: 11, align: 'right', width: 110, overflow: 'truncate' },
      axisTick: { show: false },
      axisLine: { lineStyle: { color: 'rgba(148,163,184,0.12)' } },
    },
    series: [{
      type: 'bar',
      barMaxWidth: 28,
      barMinHeight: 4,
      data: entries.map(([, c]) => ({
        value: c,
        itemStyle: {
          borderRadius: [0, 6, 6, 0],
          color: { type: 'linear', x: 0, y: 0, x2: 1, y2: 0,
            colorStops: [{ offset: 0, color: color + '28' }, { offset: 1, color: color }] },
        },
      })),
      label: {
        show: true, position: 'right', color: CHART_TXT, fontSize: 11,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        formatter: (p: any) => (p.value as number).toLocaleString(),
      },
    }],
  }
}

// ── page ──────────────────────────────────────────────────────────────────────
export default function DashboardPage() {
  const router = useRouter()
  const { status, stats, initialLoadComplete } = useConnection()
  const isConnected = status === 'connected'

  const { cache } = useAnalysis()
  const systemAnalysis = cache['layer:system']

  const rmavData = useMemo(() => {
    if (!systemAnalysis) return null
    const comps = systemAnalysis.components || []
    const avgRisk = comps.length > 0 
      ? comps.reduce((sum, c) => sum + (c.scores?.overall ?? 0), 0) / comps.length 
      : 0
    const qualityScale = (1 - avgRisk) * 100

    const criticalCount = systemAnalysis.summary?.critical_count ?? comps.filter(c => (c.criticality_level || '').toLowerCase() === 'critical').length
    const highCount = systemAnalysis.summary?.high_count ?? comps.filter(c => (c.criticality_level || '').toLowerCase() === 'high').length
    const antiPatternsCount = systemAnalysis.summary?.total_problems ?? systemAnalysis.problems?.length ?? 0
    const criticalProblems = systemAnalysis.summary?.critical_problems ?? systemAnalysis.problems?.filter(p => p.severity === 'CRITICAL').length ?? 0
    const isBlocked = criticalCount > 0 || criticalProblems > 0

    return {
      qualityScale,
      criticalCount,
      highCount,
      antiPatternsCount,
      isBlocked
    }
  }, [systemAnalysis])

  // derived
  const nodeEntries   = useMemo(() => Object.entries(stats?.node_counts   ?? {}), [stats])
  const edgeEntries   = useMemo(() => Object.entries(stats?.edge_counts   ?? {}).sort((a,b) => b[1]-a[1]), [stats])
  const structEntries = useMemo(() => Object.entries(stats?.structural_edge_counts ?? {}).sort((a,b) => b[1]-a[1]), [stats])

  // chart options (memoised so they don't rebuild on every render)
  const donutOption = useMemo(() => ({
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      backgroundColor: TIP_BG, borderColor: TIP_BORD,
      textStyle: { color: '#e2e8f0', fontSize: 12 },
      formatter: '{b}<br/><b style="font-size:14px">{c}</b> ({d}%)',
    },
    legend: {
      orient: 'vertical', right: 8, top: 'center',
      itemWidth: 9, itemHeight: 9, itemGap: 10,
      textStyle: { color: CHART_TXT, fontSize: 11 },
      formatter: (name: string) => `${name}  ${(stats?.node_counts?.[name] ?? 0).toLocaleString()}`,
    },
    series: [{
      type: 'pie', radius: ['48%', '72%'], center: ['38%', '50%'],
      avoidLabelOverlap: true,
      data: nodeEntries.map(([name, value], i) => ({
        name, value, itemStyle: { color: PALETTE[i % PALETTE.length] },
      })),
      label: { show: false }, labelLine: { show: false },
      emphasis: { scale: true, scaleSize: 6,
        itemStyle: { shadowBlur: 18, shadowColor: 'rgba(0,0,0,0.45)' } },
    }],
    graphic: [],
  }), [nodeEntries, stats])

  const edgeOption   = useMemo(() => hBarOption(edgeEntries,   '#a855f7'), [edgeEntries])
  const structOption = useMemo(() => hBarOption(structEntries, '#6366f1'), [structEntries])

  // loading
  if (!initialLoadComplete || status === 'connecting' || (isConnected && !stats)) {
    return (
      <AppLayout title="Dashboard" description="Distributed system at a glance">
        <div className="space-y-5">
          {/* KPI tiles skeleton */}
          <div className="grid gap-3 grid-cols-2 sm:grid-cols-3 lg:grid-cols-5">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="rounded-xl border border-border bg-muted/20 p-4 space-y-2">
                <Skeleton className="h-3 w-24" />
                <Skeleton className="h-8 w-16" />
                <Skeleton className="h-2.5 w-28" />
              </div>
            ))}
          </div>
          {/* CTA banners skeleton */}
          <div className="flex flex-col gap-3">
            {Array.from({ length: 2 }).map((_, i) => (
              <div key={i} className="rounded-xl border border-border bg-muted/20 px-5 py-4 flex items-center gap-3">
                <Skeleton className="h-9 w-9 rounded-lg shrink-0" />
                <div className="flex-1 space-y-1.5">
                  <Skeleton className="h-3.5 w-40" />
                  <Skeleton className="h-2.5 w-56" />
                </div>
                <Skeleton className="h-5 w-5 rounded shrink-0" />
              </div>
            ))}
          </div>
          {/* Chart cards skeleton */}
          <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {/* Donut chart */}
            <div className="rounded-xl border border-border bg-muted/20 p-4 space-y-3">
              <Skeleton className="h-4 w-36" />
              <div className="flex items-center justify-center gap-6 py-2">
                <Skeleton className="h-28 w-28 rounded-full" />
                <div className="space-y-2">
                  {Array.from({ length: 4 }).map((_, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <Skeleton className="h-2.5 w-2.5 rounded-sm shrink-0" />
                      <Skeleton className="h-2.5" style={{ width: `${50 + (i * 13) % 40}px` }} />
                    </div>
                  ))}
                </div>
              </div>
            </div>
            {/* Bar charts */}
            {Array.from({ length: 2 }).map((_, ci) => (
              <div key={ci} className="rounded-xl border border-border bg-muted/20 p-4 space-y-3">
                <Skeleton className="h-4 w-44" />
                <div className="space-y-2 pt-1">
                  {Array.from({ length: 6 }).map((_, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <Skeleton className="h-2.5 w-24 shrink-0" />
                      <Skeleton className="h-4 rounded-sm" style={{ width: `${30 + (i * 19 + ci * 7) % 55}%` }} />
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </AppLayout>
    )
  }

  const hasNodes  = nodeEntries.length > 0
  const hasEdges  = edgeEntries.length > 0
  const hasStruct = structEntries.length > 0
  const isEmpty   = !hasNodes && !hasEdges && !hasStruct

  // KPI tile definitions
  const kpis = [
    { label: 'Components',   value: stats?.total_nodes?.toLocaleString() ?? '—',
      sub: 'Nodes in graph',         Icon: Waypoints,
      text: 'text-blue-400',    border: 'border-blue-500/20',
      bg:   'bg-blue-500/[0.07]',    ring: 'bg-blue-500/10',   glow: 'bg-blue-500' },
    { label: 'Total Edges',  value: stats?.total_edges?.toLocaleString() ?? '—',
      sub: 'Derived + structural',   Icon: Network,
      text: 'text-purple-400',  border: 'border-purple-500/20',
      bg:   'bg-purple-500/[0.07]',  ring: 'bg-purple-500/10', glow: 'bg-purple-500' },
    { label: 'Node Types',   value: nodeEntries.length.toString(),
      sub: 'Unique component types', Icon: Layers,
      text: 'text-emerald-400', border: 'border-emerald-500/20',
      bg:   'bg-emerald-500/[0.07]', ring: 'bg-emerald-500/10',glow: 'bg-emerald-500' },
    { label: 'Edge Types',   value: (edgeEntries.length + structEntries.length).toString(),
      sub: 'Dependency + structural',Icon: TrendingUp,
      text: 'text-cyan-400',    border: 'border-cyan-500/20',
      bg:   'bg-cyan-500/[0.07]',    ring: 'bg-cyan-500/10',   glow: 'bg-cyan-500' },
    { label: 'Graph Density',value: stats?.density != null ? stats.density.toFixed(4) : '—',
      sub: 'Edge-to-node ratio',     Icon: BarChart3,
      text: 'text-amber-400',   border: 'border-amber-500/20',
      bg:   'bg-amber-500/[0.07]',   ring: 'bg-amber-500/10',  glow: 'bg-amber-500' },
  ] as const

  return (
    <AppLayout title="Dashboard" description="Distributed system at a glance">
      <div className="space-y-5">

        {!isConnected && <NoConnectionInfo />}

        {isConnected && (
        <>

        {/* ── KPI tiles ─────────────────────────────────────────────── */}
        <div className="grid gap-3 grid-cols-2 sm:grid-cols-3 lg:grid-cols-5">
          {kpis.map(({ label, value, sub, Icon, text, border, bg, ring, glow }) => (
            <div key={label}
              className={`relative overflow-hidden rounded-xl border ${border} ${bg} p-4`}>
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
              {/* ambient glow blob */}
              <div className={`pointer-events-none absolute -bottom-5 -right-5 h-16 w-16 rounded-full blur-2xl opacity-20 ${glow}`} />
            </div>
          ))}
        </div>

        {/* ── RMAV Quality KPI row ────────────────────────────────────── */}
        {rmavData ? (
          <div className="grid gap-3 grid-cols-2 sm:grid-cols-3 lg:grid-cols-5">
            {/* RMAV Quality */}
            <div className={`relative overflow-hidden rounded-xl border p-4 bg-background/50 backdrop-blur-sm ${
              rmavData.qualityScale >= 80 ? 'border-green-500/20 shadow-green-500/5' :
              rmavData.qualityScale >= 60 ? 'border-amber-500/20 shadow-amber-500/5' :
              'border-red-500/20 shadow-red-500/5'
            } shadow-sm`}>
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <p className="text-xs text-muted-foreground font-medium truncate">RMAV Quality</p>
                  <p className={`text-[1.65rem] font-bold leading-tight tracking-tight ${
                    rmavData.qualityScale >= 80 ? 'text-green-400' :
                    rmavData.qualityScale >= 60 ? 'text-amber-400' :
                    'text-red-400'
                  }`}>{rmavData.qualityScale.toFixed(0)}%</p>
                  <p className="text-[11px] text-muted-foreground mt-0.5 truncate">System overall health</p>
                </div>
                <div className={`shrink-0 rounded-lg p-2 ${
                  rmavData.qualityScale >= 80 ? 'bg-green-500/10' :
                  rmavData.qualityScale >= 60 ? 'bg-amber-500/10' :
                  'bg-red-500/10'
                }`}>
                  <Activity className={`h-4 w-4 ${
                    rmavData.qualityScale >= 80 ? 'text-green-400' :
                    rmavData.qualityScale >= 60 ? 'text-amber-400' :
                    'text-red-400'
                  }`} />
                </div>
              </div>
            </div>

            {/* Critical Components */}
            <div className={`relative overflow-hidden rounded-xl border p-4 bg-background/50 backdrop-blur-sm ${
              rmavData.criticalCount > 0 ? 'border-red-500/20 shadow-red-500/5' : 'border-border'
            } shadow-sm`}>
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <p className="text-xs text-muted-foreground font-medium truncate">Critical Components</p>
                  <p className={`text-[1.65rem] font-bold leading-tight tracking-tight ${
                    rmavData.criticalCount > 0 ? 'text-red-400' : 'text-foreground'
                  }`}>{rmavData.criticalCount}</p>
                  <p className="text-[11px] text-muted-foreground mt-0.5 truncate">Immediate failure risks</p>
                </div>
                <div className="shrink-0 rounded-lg bg-red-500/10 p-2">
                  <ShieldAlert className="h-4 w-4 text-red-400" />
                </div>
              </div>
            </div>

            {/* High Risk Components */}
            <div className={`relative overflow-hidden rounded-xl border p-4 bg-background/50 backdrop-blur-sm ${
              rmavData.highCount > 0 ? 'border-orange-500/20 shadow-orange-500/5' : 'border-border'
            } shadow-sm`}>
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <p className="text-xs text-muted-foreground font-medium truncate">High Risk</p>
                  <p className={`text-[1.65rem] font-bold leading-tight tracking-tight ${
                    rmavData.highCount > 0 ? 'text-orange-400' : 'text-foreground'
                  }`}>{rmavData.highCount}</p>
                  <p className="text-[11px] text-muted-foreground mt-0.5 truncate">Elevated failure risks</p>
                </div>
                <div className="shrink-0 rounded-lg bg-orange-500/10 p-2">
                  <ShieldAlert className="h-4 w-4 text-orange-400" />
                </div>
              </div>
            </div>

            {/* Active Anti-Patterns */}
            <div className={`relative overflow-hidden rounded-xl border p-4 bg-background/50 backdrop-blur-sm ${
              rmavData.antiPatternsCount > 0 ? 'border-amber-500/20 shadow-amber-500/5' : 'border-border'
            } shadow-sm`}>
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <p className="text-xs text-muted-foreground font-medium truncate">Anti-Patterns</p>
                  <p className={`text-[1.65rem] font-bold leading-tight tracking-tight ${
                    rmavData.antiPatternsCount > 0 ? 'text-amber-400' : 'text-foreground'
                  }`}>{rmavData.antiPatternsCount}</p>
                  <p className="text-[11px] text-muted-foreground mt-0.5 truncate">Smells & vulnerabilities</p>
                </div>
                <div className="shrink-0 rounded-lg bg-amber-500/10 p-2">
                  <Layers className="h-4 w-4 text-amber-400" />
                </div>
              </div>
            </div>

            {/* Deployment Status */}
            <div className={`relative overflow-hidden rounded-xl border p-4 bg-background/50 backdrop-blur-sm ${
              rmavData.isBlocked ? 'border-red-500/20 shadow-red-500/5' : 'border-green-500/20 shadow-green-500/5'
            } shadow-sm`}>
              <div className="flex items-start justify-between gap-2">
                <div className="min-w-0 flex-1">
                  <p className="text-xs text-muted-foreground font-medium truncate">Deployment Status</p>
                  <div className="mt-1">
                    <Badge variant={rmavData.isBlocked ? "destructive" : "default"} className={`text-xs px-2.5 py-0.5 font-bold ${
                      !rmavData.isBlocked ? "bg-green-500/15 text-green-400 border-green-500/35 hover:bg-green-500/25" : ""
                    }`}>
                      {rmavData.isBlocked ? "BLOCKED" : "CLEAR"}
                    </Badge>
                  </div>
                  <p className="text-[11px] text-muted-foreground mt-2 truncate">
                    {rmavData.isBlocked ? "Critical issues found" : "No blocking issues"}
                  </p>
                </div>
                <div className={`shrink-0 rounded-lg p-2 ${rmavData.isBlocked ? 'bg-red-500/10' : 'bg-green-500/10'}`}>
                  {rmavData.isBlocked ? (
                    <ShieldAlert className="h-4 w-4 text-red-400" />
                  ) : (
                    <CheckCircle2 className="h-4 w-4 text-green-400" />
                  )}
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="group flex items-center justify-between gap-4 rounded-xl border border-dashed border-border bg-muted/5 px-5 py-4">
            <div className="flex items-center gap-3 min-w-0">
              <div className="shrink-0 rounded-lg bg-muted p-2">
                <Activity className="h-5 w-5 text-muted-foreground" />
              </div>
              <div className="min-w-0">
                <p className="text-sm font-semibold text-muted-foreground leading-tight">No RMAV analysis data</p>
                <p className="text-xs text-muted-foreground mt-0.5 truncate">
                  Run a full system analysis to inspect risk profile, reliability, and deployment status.
                </p>
              </div>
            </div>
            <Button
              size="sm"
              variant="outline"
              className="shrink-0 border-dashed border-muted hover:border-foreground hover:bg-muted/10 transition-colors"
              onClick={() => router.push('/analysis')}
            >
              Analyze
              <ArrowRight className="ml-1.5 h-3.5 w-3.5" />
            </Button>
          </div>
        )}


        {/* ── Explorer & Statistics CTA banners ────────────────────── */}
        <div className="flex flex-col gap-3">
          <div
            className="group flex items-center justify-between gap-4 rounded-xl border border-blue-500/20 bg-blue-500/[0.06] px-5 py-4 cursor-pointer hover:bg-blue-500/10 transition-colors"
            onClick={() => router.push('/explorer')}
          >
            <div className="flex items-center gap-3 min-w-0">
              <div className="shrink-0 rounded-lg bg-blue-500/15 p-2">
                <Network className="h-5 w-5 text-blue-400" />
              </div>
              <div className="min-w-0">
                <p className="text-sm font-semibold text-blue-300 leading-tight">Explorer</p>
                <p className="text-xs text-muted-foreground mt-0.5 truncate">
                  topology · dependencies · component drill-down
                </p>
              </div>
            </div>
            <Button
              size="sm"
              className="shrink-0 bg-blue-500/15 text-blue-300 hover:bg-blue-500/25 border border-blue-500/30 group-hover:scale-105 transition-transform"
              onClick={e => { e.stopPropagation(); router.push('/explorer') }}
            >
              Explore
              <ArrowRight className="ml-1.5 h-3.5 w-3.5" />
            </Button>
          </div>

          <div
            className="group flex items-center justify-between gap-4 rounded-xl border border-violet-500/20 bg-violet-500/[0.06] px-5 py-4 cursor-pointer hover:bg-violet-500/10 transition-colors"
            onClick={() => router.push('/statistics')}
          >
            <div className="flex items-center gap-3 min-w-0">
              <div className="shrink-0 rounded-lg bg-violet-500/15 p-2">
                <BarChart3 className="h-5 w-5 text-violet-400" />
              </div>
              <div className="min-w-0">
                <p className="text-sm font-semibold text-violet-300 leading-tight">Statistics</p>
                <p className="text-xs text-muted-foreground mt-0.5 truncate">
                  degree · clustering · path length · centrality
                </p>
              </div>
            </div>
            <Button
              size="sm"
              className="shrink-0 bg-violet-500/15 text-violet-300 hover:bg-violet-500/25 border border-violet-500/30 group-hover:scale-105 transition-transform"
              onClick={e => { e.stopPropagation(); router.push('/statistics') }}
            >
              View
              <ArrowRight className="ml-1.5 h-3.5 w-3.5" />
            </Button>
          </div>
        </div>

        {/* ── Charts ────────────────────────────────────────────────── */}
        {isEmpty ? (
          <div className="flex flex-col items-center justify-center gap-4 rounded-xl border border-dashed border-slate-700 bg-slate-900/30 py-20 px-8 text-center">
            <div className="rounded-full border border-slate-700 bg-slate-800/60 p-5">
              <Database className="h-9 w-9 text-slate-500" />
            </div>
            <div className="space-y-1">
              <p className="font-semibold text-slate-300">No graph data yet</p>
              <p className="text-sm text-muted-foreground max-w-sm">
                Import or generate a system topology to see distribution charts and analytics.
              </p>
            </div>
            <Button onClick={() => router.push('/data')} variant="outline" className="mt-1">
              <Database className="mr-2 h-4 w-4" />
              Import Data
            </Button>
          </div>
        ) : (
          <div className="space-y-4">

            {/* Row 1 — Component donut + Dependency bar (or Structural if no deps) */}
            {(hasNodes || hasEdges) && (
              <div className="grid gap-4 lg:grid-cols-2">

                {hasNodes && (
                  <Card className="bg-background">
                    <CardHeader className="pb-1 flex flex-row items-center justify-between space-y-0">
                      <div className="flex items-center gap-2.5">
                        <div className="rounded-lg bg-blue-500/10 p-1.5">
                          <Waypoints className="h-4 w-4 text-blue-400" />
                        </div>
                        <div>
                          <CardTitle className="text-sm font-semibold">Component Distribution</CardTitle>
                          <p className="text-[11px] text-muted-foreground">By node type</p>
                        </div>
                      </div>
                      <Badge className="bg-blue-500/10 text-blue-400 border-blue-500/20 text-[11px] px-2">
                        {nodeEntries.length} types
                      </Badge>
                    </CardHeader>
                    <CardContent className="pt-1">
                      <ReactECharts
                        option={donutOption}
                        notMerge
                        style={{ height: '300px', width: '100%' }}
                        opts={{ renderer: 'canvas' }}
                      />
                    </CardContent>
                  </Card>
                )}

                {hasEdges ? (
                  <Card className="bg-background">
                    <CardHeader className="pb-1 flex flex-row items-center justify-between space-y-0">
                      <div className="flex items-center gap-2.5">
                        <div className="rounded-lg bg-purple-500/10 p-1.5">
                          <Database className="h-4 w-4 text-purple-400" />
                        </div>
                        <div>
                          <CardTitle className="text-sm font-semibold">Dependency Distribution</CardTitle>
                          <p className="text-[11px] text-muted-foreground">DEPENDS_ON edges by type</p>
                        </div>
                      </div>
                      <Badge className="bg-purple-500/10 text-purple-400 border-purple-500/20 text-[11px] px-2">
                        {edgeEntries.length} types
                      </Badge>
                    </CardHeader>
                    <CardContent className="pt-1">
                      <ReactECharts
                        option={edgeOption}
                        notMerge
                        style={{ height: '300px', width: '100%' }}
                        opts={{ renderer: 'canvas' }}
                      />
                    </CardContent>
                  </Card>
                ) : hasStruct && (
                  <Card className="bg-background">
                    <CardHeader className="pb-1 flex flex-row items-center justify-between space-y-0">
                      <div className="flex items-center gap-2.5">
                        <div className="rounded-lg bg-indigo-500/10 p-1.5">
                          <Waypoints className="h-4 w-4 text-indigo-400" />
                        </div>
                        <div>
                          <CardTitle className="text-sm font-semibold">Structural Relationships</CardTitle>
                          <p className="text-[11px] text-muted-foreground">Physical topology edges by type</p>
                        </div>
                      </div>
                      <Badge className="bg-indigo-500/10 text-indigo-400 border-indigo-500/20 text-[11px] px-2">
                        {structEntries.length} types
                      </Badge>
                    </CardHeader>
                    <CardContent className="pt-1">
                      <ReactECharts
                        option={structOption}
                        notMerge
                        style={{ height: '300px', width: '100%' }}
                        opts={{ renderer: 'canvas' }}
                      />
                    </CardContent>
                  </Card>
                )}

              </div>
            )}

            {/* Row 2 — Structural Relationships full width (only when deps also present) */}
            {hasStruct && hasEdges && (
              <Card className="bg-background">
                <CardHeader className="pb-1 flex flex-row items-center justify-between space-y-0">
                  <div className="flex items-center gap-2.5">
                    <div className="rounded-lg bg-indigo-500/10 p-1.5">
                      <Waypoints className="h-4 w-4 text-indigo-400" />
                    </div>
                    <div>
                      <CardTitle className="text-sm font-semibold">Structural Relationships</CardTitle>
                      <p className="text-[11px] text-muted-foreground">Physical topology edges by type</p>
                    </div>
                  </div>
                  <Badge className="bg-indigo-500/10 text-indigo-400 border-indigo-500/20 text-[11px] px-2">
                    {structEntries.length} types
                  </Badge>
                </CardHeader>
                <CardContent className="pt-1">
                  <ReactECharts
                    option={structOption}
                    notMerge
                    style={{ height: `${Math.max(220, structEntries.length * 46 + 32)}px`, width: '100%' }}
                    opts={{ renderer: 'canvas' }}
                  />
                </CardContent>
              </Card>
            )}

          </div>
        )}

        </>
        )}

      </div>
    </AppLayout>
  )
}
