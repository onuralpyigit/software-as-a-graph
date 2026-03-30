"use client"

import { useState, useEffect, useCallback, useMemo, useRef } from "react"
import dynamic from "next/dynamic"
import { useTheme } from "next-themes"
import { AppLayout } from "@/components/layout/app-layout"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { useConnection } from "@/lib/stores/connection-store"
import { apiClient } from "@/lib/api/client"
import { cn } from "@/lib/utils"
import {
  ChevronDown,
  ChevronRight,
  Search,
  RefreshCw,
  FolderOpen,
  Folder,
  Box,
  Cpu,
  Layers,
  Package,
  Network,
  List,
  X,
} from "lucide-react"

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), { ssr: false })

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

// Hierarchy: CSMS → CSS → CSCI → CSC → App (CSU)
interface CscGroup  { name: string; apps: AppNode[] }
interface CsciGroup { name: string; csc: Record<string, CscGroup> }
interface CssGroup  { name: string; csci: Record<string, CsciGroup> }
interface CsmsGroup { name: string; css: Record<string, CssGroup> }

type SelectedKind = "csms" | "css" | "csci" | "csc" | "app"
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
  appCount: number
  pathKey: string
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

const NODE_COLORS: Record<HGLevel, string> = {
  csms: "#10b981", css: "#3b82f6", csci: "#f59e0b", csc: "#f97316", app: "#8b5cf6",
}
const NODE_SIZES: Record<HGLevel, number> = {
  csms: 14, css: 10, csci: 8, csc: 6, app: 3.5,
}
const LEVEL_LABELS: Record<HGLevel, string> = {
  csms: "CSMS", css: "CSS", csci: "CSCI", csc: "CSC", app: "App (CSU)",
}

// Node-type → color (used in connection-graph mode)
// Colours are spaced ~30° apart on the hue wheel for maximum distinction.
const CONN_NODE_TYPE_COLORS: Record<string, string> = {
  Application: "#6366f1", // indigo  270°
  Node:        "#ef4444", // red       0°
  Broker:      "#f97316", // orange   30°
  Service:     "#0ea5e9", // sky      210°
  Library:     "#22c55e", // green   120°
  Database:    "#eab308", // yellow   60°
  Gateway:     "#ec4899", // pink    300°
  Queue:       "#14b8a6", // teal    180°
  Topic:       "#a855f7", // purple  280° — distinct from indigo by brightness
  Sensor:      "#84cc16", // lime     90°
  Actuator:    "#f43f5e", // rose    345°
}
// Link-type → color
const CONN_LINK_TYPE_COLORS: Record<string, string> = {
  PUBLISHES_TO:  "#3b82f6", // blue
  SUBSCRIBES_TO: "#f43f5e", // rose   (was orange — now distinct from USES)
  CALLS:         "#22c55e", // green
  USES:          "#eab308", // yellow (was amber — now distinct from SUBSCRIBES_TO)
  RUNS_ON:       "#f97316", // orange (explicit — was falling through to hash)
  CONTAINS:      "#94a3b8", // slate
  HOSTED_ON:     "#ec4899", // pink
  CONNECTS_TO:   "#0ea5e9", // sky
  PART_OF:       "#64748b", // gray
}
/** Deterministic fallback color for unknown types */
function hashTypeColor(type: string): string {
  let h = 0
  for (let i = 0; i < type.length; i++) h = (Math.imul(31, h) + type.charCodeAt(i)) | 0
  const hue = Math.abs(h) % 360
  return `hsl(${hue},65%,55%)`
}
function nodeTypeColor(type: string | undefined): string {
  if (!type) return "#6b7280"
  return CONN_NODE_TYPE_COLORS[type] ?? hashTypeColor(type)
}
function linkTypeColor(type: string | undefined): string {
  if (!type) return "#6b7280"
  return CONN_LINK_TYPE_COLORS[type] ?? hashTypeColor(type)
}

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
        nodes.push({ id, name: app.csu ?? app.name ?? app.id ?? "?", level: "app", appCount: 1, pathKey: app.id })
        links.push({ source: parent.id, target: id })
      }
      break
    }
  }
  return { nodes, links }
}

function HierarchyGraph({ hierarchy }: { hierarchy: Record<string, CsmsGroup> }) {
  const { theme, systemTheme } = useTheme()
  const isDark = (theme === "system" ? systemTheme : theme) === "dark"

  const containerRef = useRef<HTMLDivElement>(null)
  const fgRef = useRef<any>(null)
  const searchRef = useRef<HTMLInputElement>(null)
  const [dims, setDims] = useState({ width: 800, height: 580 })

  const [drillNode, setDrillNode] = useState<HGNode | null>(null)
  const [drillStack, setDrillStack] = useState<HGNode[]>([])
  const [selectedApp, setSelectedApp] = useState<HGNode | null>(null)
  const [viewMode, setViewMode] = useState<"hierarchy" | "connections">("hierarchy")

  const [connData, setConnData] = useState<{ nodes: any[]; links: any[] } | null>(null)
  const [connLoading, setConnLoading] = useState(false)
  const [connError, setConnError] = useState<string | null>(null)
  const [connTab, setConnTab] = useState<"out" | "in" | "props">("out")
  const [connDepth, setConnDepth] = useState(1)

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
              result.push({ id: `app:${app.id}`, name: app.name ?? app.id, level: "app", appCount: 1, pathKey: app.id })
          }
        }
      }
    }
    return result
  }, [hierarchy])

  const filteredNodes = useMemo(() => {
    const q = appSearch.trim().toLowerCase()
    if (!q) return []
    return flatNodes.filter(n => n.name.toLowerCase().includes(q) || n.pathKey.toLowerCase().includes(q)).slice(0, 25)
  }, [flatNodes, appSearch])

  const jumpToNode = useCallback((node: HGNode) => {
    setAppSearch("")
    setSearchOpen(false)
    if (node.level === "app") {
      setSelectedApp(node)
      setViewMode("connections")
      setConnTab("out")
      setConnData(null)
    } else {
      // Clear connection state and drill into the hierarchy node
      setSelectedApp(null)
      setViewMode("hierarchy")
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
  }, [hierarchy])

  const graphData = useMemo(() => buildDrillData(hierarchy, drillNode), [hierarchy, drillNode])
  const connGraphData = useMemo(() => {
    if (!connData) return { nodes: [], links: [] }
    const filteredLinks = connData.links.filter(l => l.type !== "DEPENDS_ON")
    // Keep only nodes that are still referenced by remaining links or are the selected app
    const referencedIds = new Set<string>([
      ...(selectedApp ? [selectedApp.pathKey] : []),
      ...filteredLinks.flatMap(l => [l.source?.id ?? l.source, l.target?.id ?? l.target]),
    ])
    const filteredNodes = connData.nodes.filter(n => referencedIds.has(n.id))
    return { nodes: filteredNodes, links: filteredLinks }
  }, [connData, selectedApp])

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

  // Forces — hierarchy
  useEffect(() => {
    if (viewMode !== "hierarchy") return
    const fg = fgRef.current
    if (!fg) return
    fg.d3Force("charge")?.strength(-120).distanceMax(200)
    fg.d3Force("link")?.distance(60)
  }, [graphData, viewMode])

  // Forces — connections
  useEffect(() => {
    if (viewMode !== "connections") return
    const fg = fgRef.current
    if (!fg) return
    fg.d3Force("charge")?.strength(-200).distanceMax(320)
    fg.d3Force("link")?.distance(90)
  }, [connGraphData, viewMode])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    const ro = new ResizeObserver(() => setDims({ width: el.clientWidth, height: el.clientHeight }))
    ro.observe(el)
    setDims({ width: el.clientWidth, height: el.clientHeight })
    return () => ro.disconnect()
  }, [])

  // Fetch connections whenever selected app or depth changes
  useEffect(() => {
    if (!selectedApp) { setConnData(null); setConnError(null); return }
    let cancelled = false
    setConnLoading(true)
    setConnError(null)
    setConnData(null)
    apiClient.getNodeConnectionsWithDepth(selectedApp.pathKey, true, connDepth)
      .then(d => { if (!cancelled) setConnData({ nodes: d.nodes, links: d.links }) })
      .catch(e => { if (!cancelled) setConnError(e instanceof Error ? e.message : String(e)) })
      .finally(() => { if (!cancelled) setConnLoading(false) })
    return () => { cancelled = true }
  }, [selectedApp, connDepth])

  const clearSelection = useCallback(() => {
    setSelectedApp(null)
    setViewMode("hierarchy")
    setConnData(null)
    setConnError(null)
  }, [])

  // Click handler — hierarchy mode
  const drillInto = useCallback((node: object) => {
    const n = node as HGNode
    if (drillNode && n.id === drillNode.id) return
    if (n.level === "app") {
      if (selectedApp?.id === n.id) { clearSelection(); return }
      setSelectedApp(n)
      setViewMode("connections")
      setConnTab("out")
      return
    }
    clearSelection()
    const clean: HGNode = { id: n.id, name: n.name, level: n.level, appCount: n.appCount, pathKey: n.pathKey }
    setDrillStack(prev => drillNode ? [...prev, drillNode] : prev)
    setDrillNode(clean)
  }, [drillNode, selectedApp, clearSelection])

  // Click handler — connections mode (re-center on clicked node)
  const drillIntoConn = useCallback((node: object) => {
    const n = node as any
    if (n.id === selectedApp?.pathKey) return
    setSelectedApp({ id: `app:${n.id}`, name: n.label ?? n.id, level: "app", appCount: 1, pathKey: n.id })
    setConnTab("out")
    setConnData(null)
  }, [selectedApp])

  const drillTo = useCallback((idx: number) => {
    clearSelection()
    if (idx < 0) { setDrillNode(null); setDrillStack([]) }
    else { setDrillNode(drillStack[idx]); setDrillStack(prev => prev.slice(0, idx)) }
  }, [drillStack, clearSelection])

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

  const nodeCanvasObjectConn = useCallback(
    (node: object, ctx: CanvasRenderingContext2D, globalScale: number) => {
      const n = node as any
      const isCenter = n.id === selectedApp?.pathKey
      const color = nodeTypeColor(n.type)
      const r = isCenter ? 10 : 6
      if (isCenter) {
        ctx.beginPath(); ctx.arc(n.x!, n.y!, r + 7, 0, 2 * Math.PI)
        ctx.fillStyle = color + "33"; ctx.fill()
      }
      ctx.beginPath(); ctx.arc(n.x!, n.y!, r, 0, 2 * Math.PI)
      ctx.fillStyle = color; ctx.fill()
      if (isCenter) { ctx.strokeStyle = isDark ? "#fff" : "#111"; ctx.lineWidth = 2.5; ctx.stroke() }
      if (globalScale >= 0.4) {
        const fontSize = Math.max(3, 10 / globalScale)
        ctx.font = `${isCenter ? "bold " : ""}${fontSize}px sans-serif`
        ctx.fillStyle = isDark ? "#e5e7eb" : "#374151"; ctx.textAlign = "center"
        const raw = n.label ?? n.id ?? "?"
        ctx.fillText(raw.length > 26 ? raw.slice(0, 24) + "…" : raw, n.x!, n.y! + r + fontSize + 1)
        if (n.type && globalScale >= 0.7) {
          ctx.font = `${Math.max(2, 7 / globalScale)}px sans-serif`
          ctx.fillStyle = isDark ? "#9ca3af" : "#6b7280"
          ctx.fillText(n.type, n.x!, n.y! + r + fontSize * 2 + 2)
        }
      }
    }, [selectedApp, isDark])

  const connLinkColor = useCallback((link: any) => linkTypeColor(link.type), [])

  const connLinkWidth = useCallback((link: any) => {
    const src = link.source?.id ?? link.source
    const tgt = link.target?.id ?? link.target
    return (src === selectedApp?.pathKey || tgt === selectedApp?.pathKey) ? 2.5 : 1.5
  }, [selectedApp])

  const hierLinkColor = isDark ? "rgba(255,255,255,0.55)" : "rgba(0,0,0,0.30)"
  const bgColor = isDark ? "#09090b" : "#ffffff"
  const breadcrumbs = [
    { label: "Root", idx: -1 },
    ...drillStack.map((n, i) => ({ label: n.name, idx: i })),
  ]

  const outLinks = (connData?.links ?? []).filter(l => (l.source?.id ?? l.source) === selectedApp?.pathKey)
  const inLinks  = (connData?.links ?? []).filter(l => (l.target?.id ?? l.target) === selectedApp?.pathKey)
  const nodeById = useMemo(() => { const m = new Map<string, any>(); connData?.nodes.forEach(n => m.set(n.id, n)); return m }, [connData])
  const peerLabel = (nodeId: string) => nodeById.get(nodeId)?.label ?? nodeId
  const appNode = connData?.nodes.find(n => n.id === selectedApp?.pathKey)
  const appProps = appNode?.properties ? Object.entries(appNode.properties).filter(([, v]) => v !== undefined && v !== null && v !== "") : []

  return (
    <div className="flex flex-col gap-3 h-[660px]">
      {/* Nav bar */}
      <div className="flex items-center gap-1 text-sm flex-wrap min-h-[28px]">
        {viewMode === "hierarchy" ? (
          <>
            {breadcrumbs.map((bc, i) => (
              <span key={bc.idx} className="flex items-center gap-1">
                {i > 0 && <ChevronRight className="h-3 w-3 text-muted-foreground shrink-0" />}
                <button className="text-muted-foreground hover:text-foreground transition-colors hover:underline underline-offset-2"
                  onClick={() => drillTo(bc.idx)}>{bc.label}</button>
              </span>
            ))}
            {drillNode && (
              <span className="flex items-center gap-1.5">
                <ChevronRight className="h-3 w-3 text-muted-foreground shrink-0" />
                <span className="h-2 w-2 rounded-full shrink-0" style={{ background: NODE_COLORS[drillNode.level] }} />
                <span className="font-semibold text-foreground">{drillNode.name}</span>
              </span>
            )}
            {!drillNode && <span className="ml-3 text-xs text-muted-foreground">click a node to drill in</span>}
          </>
        ) : (
          <>
            <button className="flex items-center gap-1 text-muted-foreground hover:text-foreground transition-colors"
              onClick={clearSelection}>
              <ChevronRight className="h-3 w-3 rotate-180" />
              <span className="text-xs">Hierarchy</span>
            </button>
            <ChevronRight className="h-3 w-3 text-muted-foreground shrink-0" />
            <span className="h-2 w-2 rounded-full shrink-0" style={{ background: NODE_COLORS.app }} />
            <span className="text-sm font-semibold text-foreground truncate max-w-64">{selectedApp?.name}</span>
            {connLoading && <LoadingSpinner className="h-3.5 w-3.5 ml-1 text-muted-foreground" />}
            {!connLoading && <span className="ml-2 text-xs text-muted-foreground">click a node to re-center</span>}
          </>
        )}

        {/* Search — always visible on the right */}
        <div className="ml-auto relative shrink-0">
          <div className="relative">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground pointer-events-none" />
            <input
              ref={searchRef}
              className="h-7 pl-7 pr-7 text-xs rounded-md border border-border bg-muted/40 focus:outline-none focus:ring-1 focus:ring-ring w-48 placeholder:text-muted-foreground"
              placeholder="Search apps…"
              value={appSearch}
              onChange={e => { setAppSearch(e.target.value); setSearchOpen(true) }}
              onFocus={() => setSearchOpen(true)}
              onBlur={() => setTimeout(() => setSearchOpen(false), 150)}
            />
            {appSearch && (
              <button className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                onMouseDown={e => { e.preventDefault(); setAppSearch(""); setSearchOpen(false) }}>
                <X className="h-3 w-3" />
              </button>
            )}
          </div>
          {searchOpen && filteredNodes.length > 0 && (
            <div className="absolute right-0 top-full mt-1 z-50 w-80 rounded-md border border-border bg-popover shadow-lg overflow-hidden">
              <div className="max-h-72 overflow-y-auto py-1">
                {filteredNodes.map(node => (
                  <button
                    key={node.id}
                    className="w-full flex items-center gap-2 px-3 py-1.5 text-xs hover:bg-accent hover:text-accent-foreground text-left transition-colors"
                    onMouseDown={e => { e.preventDefault(); jumpToNode(node) }}
                  >
                    <span className="h-2 w-2 rounded-full shrink-0" style={{ background: NODE_COLORS[node.level] }} />
                    <span className="font-medium truncate">{node.name}</span>
                    <span className="ml-auto flex items-center gap-1.5 shrink-0">
                      <span className="text-[10px] text-muted-foreground">{LEVEL_LABELS[node.level]}</span>
                      {node.level !== "app" && <span className="text-[10px] text-muted-foreground">{node.appCount} apps</span>}
                    </span>
                  </button>
                ))}
              </div>
              <div className="border-t border-border px-3 py-1.5 text-[10px] text-muted-foreground">
                {filteredNodes.length} result{filteredNodes.length !== 1 ? "s" : ""}{filteredNodes.length === 25 ? " (showing first 25)" : ""}
              </div>
            </div>
          )}
          {searchOpen && appSearch.trim() && filteredNodes.length === 0 && (
            <div className="absolute right-0 top-full mt-1 z-50 w-56 rounded-md border border-border bg-popover shadow-lg px-3 py-2.5 text-xs text-muted-foreground">
              No results for &ldquo;{appSearch.trim()}&rdquo;
            </div>
          )}
        </div>
      </div>

      {/* Main row */}
      <div className="flex gap-4 flex-1 min-h-0">
        {/* Legend */}
        <div className="flex flex-col gap-3 w-36 shrink-0">
          <div className="space-y-1 text-xs">
            <p className="font-medium text-muted-foreground uppercase tracking-wide text-[10px]">Legend</p>
            {viewMode === "hierarchy" ? (
              (["csms", "css", "csci", "csc", "app"] as HGLevel[]).map(lvl => (
                <span key={lvl} className="flex items-center gap-2">
                  <span className="h-2.5 w-2.5 rounded-full shrink-0" style={{ background: NODE_COLORS[lvl] }} />
                  <span>{LEVEL_LABELS[lvl]}</span>
                </span>
              ))
            ) : (
              <>
                <p className="font-medium text-muted-foreground uppercase tracking-wide text-[10px] mt-1">Nodes</p>
                {Array.from(new Set((connGraphData.nodes as any[]).map(n => n.type).filter(Boolean))).map(t => (
                  <span key={t} className="flex items-center gap-2">
                    <span className="h-2.5 w-2.5 rounded-full shrink-0" style={{ background: nodeTypeColor(t as string) }} />
                    <span className="truncate">{t as string}</span>
                  </span>
                ))}
                <p className="font-medium text-muted-foreground uppercase tracking-wide text-[10px] mt-2">Edges</p>
                {Array.from(new Set((connGraphData.links as any[]).map(l => l.type).filter(Boolean))).map(t => (
                  <span key={t} className="flex items-center gap-2">
                    <span className="h-0.5 w-4 shrink-0 rounded-full" style={{ background: linkTypeColor(t as string) }} />
                    <span className="truncate">{t as string}</span>
                  </span>
                ))}
              </>
            )}
          </div>
          <div className="mt-auto space-y-0.5 text-xs text-muted-foreground">
            {viewMode === "hierarchy" ? (
              <><p>{graphData.nodes.length} nodes</p><p>{graphData.links.length} edges</p></>
            ) : connData ? (
              <><p>{connData.nodes.length} nodes</p><p>{connData.links.length} edges</p></>
            ) : null}
          </div>
        </div>

        {/* Canvas */}
        <div ref={containerRef} className="flex-1 overflow-hidden relative" style={{ background: bgColor }}>
          {viewMode === "hierarchy" ? (
            <ForceGraph2D
              key="hierarchy"
              graphData={graphData as any}
              width={dims.width}
              height={dims.height}
              nodeCanvasObject={nodeCanvasObject}
              nodeCanvasObjectMode={() => "replace"}
              nodePointerAreaPaint={(node: object, color: string, ctx: CanvasRenderingContext2D) => {
                const n = node as HGNode
                const hitR = Math.max(NODE_SIZES[n.level] * 2.5, 14)
                ctx.beginPath(); ctx.arc(n.x!, n.y!, hitR, 0, 2 * Math.PI)
                ctx.fillStyle = color; ctx.fill()
              }}
              onNodeClick={drillInto}
              onBackgroundClick={() => { if (selectedApp) clearSelection() }}
              linkColor={() => hierLinkColor}
              linkWidth={2}
              linkDirectionalArrowLength={8}
              linkDirectionalArrowRelPos={0.85}
              linkDirectionalArrowColor={() => hierLinkColor}
              nodeRelSize={1}
              cooldownTicks={150}
              d3AlphaDecay={0.04}
              d3VelocityDecay={0.5}
              ref={fgRef}
            />
          ) : (
            <>
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
              <ForceGraph2D
                key={`conn-${selectedApp?.pathKey ?? ""}`}
                graphData={connGraphData as any}
                width={dims.width}
                height={dims.height}
                nodeCanvasObject={nodeCanvasObjectConn}
                nodeCanvasObjectMode={() => "replace"}
                nodePointerAreaPaint={(node: object, color: string, ctx: CanvasRenderingContext2D) => {
                  const n = node as any
                  const r = n.id === selectedApp?.pathKey ? 14 : 10
                  ctx.beginPath(); ctx.arc(n.x!, n.y!, r, 0, 2 * Math.PI)
                  ctx.fillStyle = color; ctx.fill()
                }}
                onNodeClick={drillIntoConn}
                onBackgroundClick={clearSelection}
                linkColor={connLinkColor}
                linkWidth={connLinkWidth}
                linkDirectionalArrowLength={9}
                linkDirectionalArrowRelPos={0.85}
                linkDirectionalArrowColor={connLinkColor}
                nodeRelSize={1}
                cooldownTicks={150}
                d3AlphaDecay={0.03}
                d3VelocityDecay={0.4}
                ref={fgRef}
              />
            </>
          )}
        </div>

        {/* Side panel */}
        {selectedApp && (
          <div className="w-72 shrink-0 border-l border-border bg-background flex flex-col overflow-hidden">
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-border shrink-0">
              <div className="flex items-center gap-2 min-w-0">
                <span className="inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-semibold text-white shrink-0"
                  style={{ background: NODE_COLORS.app }}>App</span>
                <span className="text-xs font-semibold text-foreground truncate">{selectedApp.name}</span>
              </div>
              <button onClick={clearSelection} className="text-muted-foreground hover:text-foreground transition-colors ml-1 shrink-0">
                <X className="h-3.5 w-3.5" />
              </button>
            </div>
            <div className="flex items-center gap-2 px-4 py-2 border-b border-border shrink-0">
              <span className="text-[11px] text-muted-foreground">Depth</span>
              {[1, 2, 3].map(d => (
                <button key={d} onClick={() => setConnDepth(d)}
                  className={cn("h-6 w-6 rounded text-xs font-medium transition-colors",
                    connDepth === d ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground hover:bg-muted/80"
                  )}>{d}</button>
              ))}
              {connData && !connLoading && (
                <span className="ml-auto text-[11px] text-muted-foreground">{connData.links.length} edges</span>
              )}
            </div>
            <div className="flex border-b border-border shrink-0">
              {(["out", "in", "props"] as const).map(tab => (
                <button key={tab} onClick={() => setConnTab(tab)}
                  className={cn("flex-1 py-1.5 text-[11px] font-medium transition-colors",
                    connTab === tab ? "border-b-2 border-primary text-foreground" : "text-muted-foreground hover:text-foreground"
                  )}>
                  {tab === "out" ? `Out (${outLinks.length})` : tab === "in" ? `In (${inLinks.length})` : "Props"}
                </button>
              ))}
            </div>
            <div className="flex-1 overflow-y-auto">
              {connLoading && <div className="flex items-center justify-center h-24"><LoadingSpinner className="h-5 w-5" /></div>}
              {connError && !connLoading && <p className="px-4 py-3 text-xs text-destructive">{connError}</p>}
              {!connLoading && !connError && connTab !== "props" && (
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-background z-10">
                    <tr className="border-b border-border">
                      <th className="text-left px-3 py-2 text-[10px] font-semibold text-muted-foreground uppercase tracking-wide">
                        {connTab === "out" ? "Target" : "Source"}
                      </th>
                      <th className="text-left px-3 py-2 text-[10px] font-semibold text-muted-foreground uppercase tracking-wide">Type</th>
                      <th className="text-right px-3 py-2 text-[10px] font-semibold text-muted-foreground uppercase tracking-wide">W</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(connTab === "out" ? outLinks : inLinks).map((link, i) => {
                      const peerId = connTab === "out" ? (link.target?.id ?? link.target) : (link.source?.id ?? link.source)
                      return (
                        <tr key={i} className="border-b border-border/50 hover:bg-muted/30 transition-colors">
                          <td className="px-3 py-2 font-medium text-foreground max-w-[110px] truncate" title={peerLabel(peerId)}>{peerLabel(peerId)}</td>
                          <td className="px-3 py-2 text-muted-foreground max-w-[80px] truncate text-[10px]">{link.type ?? "—"}</td>
                          <td className="px-3 py-2 text-right text-muted-foreground tabular-nums">
                            {link.weight != null ? Number(link.weight).toFixed(2) : "—"}
                          </td>
                        </tr>
                      )
                    })}
                    {(connTab === "out" ? outLinks : inLinks).length === 0 && (
                      <tr><td colSpan={3} className="px-3 py-8 text-center text-muted-foreground text-[11px]">
                        No {connTab === "out" ? "outbound" : "inbound"} connections
                      </td></tr>
                    )}
                  </tbody>
                </table>
              )}
              {!connLoading && !connError && connTab === "props" && (
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-background z-10">
                    <tr className="border-b border-border">
                      <th className="text-left px-3 py-2 text-[10px] font-semibold text-muted-foreground uppercase tracking-wide w-2/5">Key</th>
                      <th className="text-left px-3 py-2 text-[10px] font-semibold text-muted-foreground uppercase tracking-wide">Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    {appProps.map(([k, v]) => (
                      <tr key={k} className="border-b border-border/50 hover:bg-muted/30 transition-colors">
                        <td className="px-3 py-2 font-medium text-foreground">{k}</td>
                        <td className="px-3 py-2 font-mono text-muted-foreground break-all text-[10px]">
                          {typeof v === "object" ? JSON.stringify(v) : String(v)}
                        </td>
                      </tr>
                    ))}
                    {appProps.length === 0 && !connLoading && (
                      <tr><td colSpan={2} className="px-3 py-8 text-center text-muted-foreground text-[11px]">No properties available</td></tr>
                    )}
                  </tbody>
                </table>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Detail Panel (left) ──────────────────────────────────────────────────────

const KIND_LABEL: Record<SelectedKind, string> = {
  csms: "CSMS", css: "CSS", csci: "CSCI", csc: "CSC", app: "App (CSU)",
}
const KIND_COLOR: Record<SelectedKind, string> = {
  csms: "#10b981", css: "#3b82f6", csci: "#f59e0b", csc: "#f97316", app: "#8b5cf6",
}

function DetailTable({ headers, rows }: {
  headers: string[]
  rows: (string | number | undefined)[][]
}) {
  return (
    <table className="w-full text-sm">
      <thead className="sticky top-0 bg-background z-10">
        <tr className="border-b border-border">
          {headers.map((h) => (
            <th key={h} className="text-left px-4 py-2.5 text-xs font-semibold text-muted-foreground uppercase tracking-wide whitespace-nowrap">
              {h}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr key={i} className="border-b border-border/50 hover:bg-muted/30 transition-colors">
            {row.map((cell, j) => (
              <td key={j} className="px-4 py-2.5 text-sm first:font-medium first:text-foreground text-muted-foreground">
                {cell !== undefined && cell !== null && cell !== ""
                  ? String(cell)
                  : <span className="text-muted-foreground/40">—</span>}
              </td>
            ))}
          </tr>
        ))}
        {rows.length === 0 && (
          <tr>
            <td colSpan={headers.length} className="px-4 py-12 text-center text-muted-foreground text-sm">
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
    content = <DetailTable headers={["CSS", "CSCI Groups", "CSC Groups", "Apps"]} rows={rows} />
  } else if (node.kind === "css") {
    const css = node.payload as CssGroup
    const rows = sortKeys(Object.keys(css.csci)).map((k) => {
      const csci = css.csci[k]
      const cscCount = Object.keys(csci.csc).length
      const appCount = Object.values(csci.csc).flatMap(c => c.apps).length
      return [k, cscCount, appCount]
    })
    content = <DetailTable headers={["CSCI", "CSC Groups", "Apps"]} rows={rows} />
  } else if (node.kind === "csci") {
    const csci = node.payload as CsciGroup
    const rows = sortKeys(Object.keys(csci.csc)).map((k) => [k, csci.csc[k].apps.length])
    content = <DetailTable headers={["CSC", "Apps"]} rows={rows} />
  } else if (node.kind === "csc") {
    const csc = node.payload as CscGroup
    const rows = csc.apps.map((a) => [a.id, a.csu ?? a.name ?? "—", a.weight])
    content = <DetailTable headers={["ID", "Name / CSU", "Weight"]} rows={rows} />
  } else {
    // app — key-value table
    const app = node.payload as AppNode
    const entries = Object.entries(app).filter(([, v]) => v !== undefined && v !== null && v !== "")
    content = (
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-background z-10">
          <tr className="border-b border-border">
            <th className="text-left px-4 py-2.5 text-xs font-semibold text-muted-foreground uppercase tracking-wide w-2/5">Property</th>
            <th className="text-left px-4 py-2.5 text-xs font-semibold text-muted-foreground uppercase tracking-wide">Value</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([k, v]) => (
            <tr key={k} className="border-b border-border/50 hover:bg-muted/30 transition-colors">
              <td className="px-4 py-2.5 font-medium text-foreground">{k}</td>
              <td className="px-4 py-2.5 font-mono text-muted-foreground break-all text-xs">
                {typeof v === "object" ? JSON.stringify(v) : String(v)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    )
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-5 py-4 border-b border-border shrink-0">
        <div className="flex items-center gap-2 mb-1.5">
          <span
            className="inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold text-white"
            style={{ background: KIND_COLOR[node.kind] }}
          >
            {KIND_LABEL[node.kind]}
          </span>
        </div>
        <h2 className="text-base font-semibold text-foreground leading-tight">{node.label}</h2>
        {node.path.length > 1 && (
          <p className="text-xs text-muted-foreground mt-0.5 truncate">{node.path.join(" › ")}</p>
        )}
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
  const isOpen = openSet.has(nodeKey)
  const filtered = csc.apps.filter(a => matches(a, q))
  if (!filtered.length && q) return null
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
  const isOpen = openSet.has(nodeKey)
  const visibleCsc = sortKeys(Object.keys(csci.csc)).filter(k => csci.csc[k].apps.some(a => matches(a, q)))
  if (!visibleCsc.length && q) return null
  const total = visibleCsc.reduce((s, k) => s + csci.csc[k].apps.filter(a => matches(a, q)).length, 0)
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
  const isOpen = openSet.has(nodeKey)
  const visibleCsci = sortKeys(Object.keys(css.csci)).filter(k =>
    Object.values(css.csci[k].csc).some(c => c.apps.some(a => matches(a, q)))
  )
  if (!visibleCsci.length && q) return null
  const total = visibleCsci.reduce((s, k) =>
    s + Object.values(css.csci[k].csc).flatMap(c => c.apps).filter(a => matches(a, q)).length, 0)
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
  const isOpen = openSet.has(nodeKey)
  const visibleCss = sortKeys(Object.keys(csms.css)).filter(k =>
    Object.values(csms.css[k].csci).flatMap(ci => Object.values(ci.csc)).some(c => c.apps.some(a => matches(a, q)))
  )
  if (!visibleCss.length && q) return null
  const total = visibleCss.reduce((s, k) =>
    s + Object.values(csms.css[k].csci).flatMap(ci => Object.values(ci.csc)).flatMap(c => c.apps).filter(a => matches(a, q)).length, 0)
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

// ── Page ──────────────────────────────────────────────────────────────────────

export default function BrowserPage() {
  const { status, initialLoadComplete } = useConnection()
  const isConnected = status === "connected"

  const [hierarchy, setHierarchy] = useState<Record<string, CsmsGroup>>({})
  const [totalApps, setTotalApps] = useState(0)
  const [openSet, setOpenSet] = useState<Set<string>>(new Set())
  const [search, setSearch] = useState("")
  const [browseSearchOpen, setBrowseSearchOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedNode, setSelectedNode] = useState<SelectedNode | null>(null)

  const toggle = useCallback((key: string) => {
    setOpenSet((prev) => {
      const next = new Set(prev)
      next.has(key) ? next.delete(key) : next.add(key)
      return next
    })
  }, [])

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    try {
      const apps: AppNode[] = await apiClient.getAllApps()
      const h = buildHierarchy(apps)
      setHierarchy(h)
      setTotalApps(apps.length)
      const firstCsms = Object.keys(h)[0]
      if (firstCsms) {
        const firstCss = Object.keys(h[firstCsms].css)[0]
        setOpenSet(new Set([
          `csms:${firstCsms}`,
          ...(firstCss ? [`css:${firstCsms}/${firstCss}`] : []),
        ]))
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (isConnected) fetchData()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isConnected])

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
    setSearch("")
    setBrowseSearchOpen(false)
  }, [])

  if (!initialLoadComplete) {
    return <AppLayout><div className="flex items-center justify-center h-64"><LoadingSpinner /></div></AppLayout>
  }
  if (!isConnected) {
    return <AppLayout><NoConnectionInfo /></AppLayout>
  }

  const q = search.toLowerCase()
  const csmsKeys = sortKeys(Object.keys(hierarchy))

  return (
    <AppLayout>
      <div className="flex flex-col gap-5 p-6 h-full">
        {/* Header */}
        <div className="flex items-center justify-between shrink-0">
          <div>
            <h1 className="text-2xl font-bold">Apps Browser</h1>
            <p className="text-muted-foreground text-sm mt-1">CSMS → CSS → CSCI → CSC → App (CSU)</p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="secondary" className="px-3 py-1">
              <Layers className="h-3.5 w-3.5 mr-1.5" />{csmsKeys.length} CSMS
            </Badge>
            <Badge variant="secondary" className="px-3 py-1">
              <Cpu className="h-3.5 w-3.5 mr-1.5" />{totalApps} Apps
            </Badge>
            <Button variant="outline" size="sm" onClick={fetchData} disabled={loading}>
              {loading ? <LoadingSpinner className="h-4 w-4" /> : <RefreshCw className="h-4 w-4" />}
              <span className="ml-2">Refresh</span>
            </Button>
          </div>
        </div>

        {error && (
          <div className="rounded-md bg-destructive/10 border border-destructive/30 text-destructive px-4 py-3 text-sm shrink-0">
            {error}
          </div>
        )}

        <Tabs defaultValue="browse" className="flex flex-col flex-1 min-h-0">
          <TabsList className="mb-4 shrink-0">
            <TabsTrigger value="browse" className="flex items-center gap-2">
              <List className="h-4 w-4" />Browse
            </TabsTrigger>
            <TabsTrigger value="graph" className="flex items-center gap-2">
              <Network className="h-4 w-4" />Graph
            </TabsTrigger>
          </TabsList>

          {/* ── Browse tab ── */}
          <TabsContent value="browse" className="flex-1 min-h-0 mt-0">
            <div className="flex border border-border rounded-lg overflow-hidden h-full" style={{ minHeight: "520px" }}>
              {/* Left: Detail / table panel */}
              <div className="flex-1 overflow-auto border-r border-border min-w-0">
                {loading && !csmsKeys.length
                  ? <div className="flex items-center justify-center h-full"><LoadingSpinner /></div>
                  : selectedNode
                    ? <NodeDetailPanel node={selectedNode} />
                    : <EmptyDetailState />
                }
              </div>

              {/* Right: Tree panel */}
              <div className="w-72 flex flex-col overflow-hidden shrink-0">
                {/* Search bar */}
                <div className="px-3 py-2.5 border-b border-border shrink-0">
                  <div className="relative">
                    <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
                    <Input
                      className="pl-8 h-8 text-xs bg-muted/30"
                      placeholder="Search…"
                      value={search}
                      onChange={(e) => { setSearch(e.target.value); setBrowseSearchOpen(true) }}
                      onFocus={() => setBrowseSearchOpen(true)}
                      onBlur={() => setTimeout(() => setBrowseSearchOpen(false), 150)}
                    />
                    {search && (
                      <button
                        className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                        onClick={() => { setSearch(""); setBrowseSearchOpen(false) }}
                      >
                        <X className="h-3.5 w-3.5" />
                      </button>
                    )}
                    {/* Jump-to dropdown */}
                    {browseSearchOpen && filteredBrowseNodes.length > 0 && (
                      <div className="absolute left-0 right-0 top-full mt-1 z-50 rounded-md border border-border bg-popover shadow-lg overflow-hidden">
                        <div className="max-h-72 overflow-y-auto py-1">
                          {filteredBrowseNodes.map(node => (
                            <button
                              key={node.key}
                              className="w-full flex items-center gap-2 px-3 py-1.5 text-xs hover:bg-accent hover:text-accent-foreground text-left transition-colors"
                              onMouseDown={e => { e.preventDefault(); jumpToBrowseNode(node) }}
                            >
                              <span className="h-2 w-2 rounded-full shrink-0" style={{ background: NODE_COLORS[node.kind as HGLevel] }} />
                              <span className="font-medium truncate">{node.label}</span>
                              <span className="ml-auto flex items-center gap-1.5 shrink-0">
                                <span className="text-[10px] text-muted-foreground uppercase">{node.kind}</span>
                                {node.kind !== "app" && <span className="text-[10px] text-muted-foreground">{node.appCount}</span>}
                              </span>
                            </button>
                          ))}
                        </div>
                        <div className="border-t border-border px-3 py-1 text-[10px] text-muted-foreground">
                          {filteredBrowseNodes.length} result{filteredBrowseNodes.length !== 1 ? "s" : ""}{filteredBrowseNodes.length === 25 ? " (showing first 25)" : ""}
                        </div>
                      </div>
                    )}
                    {browseSearchOpen && search.trim() && filteredBrowseNodes.length === 0 && (
                      <div className="absolute left-0 right-0 top-full mt-1 z-50 rounded-md border border-border bg-popover shadow-lg px-3 py-2.5 text-xs text-muted-foreground">
                        No results for &ldquo;{search.trim()}&rdquo;
                      </div>
                    )}
                  </div>
                </div>

                {/* Tree */}
                <div className="flex-1 overflow-y-auto p-1.5">
                  {csmsKeys.map(k => (
                    <CsmsTreeNode
                      key={k}
                      name={k}
                      csms={hierarchy[k]}
                      selectedKey={selectedNode?.key ?? null}
                      onSelect={setSelectedNode}
                      openSet={openSet}
                      toggle={toggle}
                      q={q}
                    />
                  ))}
                  {!loading && csmsKeys.length === 0 && (
                    <p className="text-center text-muted-foreground py-10 text-xs px-4">
                      No data loaded. Import a graph first.
                    </p>
                  )}
                </div>

                {/* Legend footer */}
                <div className="border-t border-border px-3 py-2.5 shrink-0 space-y-1">
                  {([["csms", "#10b981", "CSMS"], ["css", "#3b82f6", "CSS"], ["csci", "#f59e0b", "CSCI"], ["csc", "#f97316", "CSC"], ["app", "#8b5cf6", "App"]] as const).map(([, color, lbl]) => (
                    <span key={lbl} className="flex items-center gap-1.5 text-[11px] text-muted-foreground">
                      <span className="h-2 w-2 rounded-full shrink-0" style={{ background: color }} />
                      {lbl}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </TabsContent>

          {/* ── Graph tab ── */}
          <TabsContent value="graph" className="mt-0">
            {csmsKeys.length > 0
              ? <HierarchyGraph hierarchy={hierarchy} />
              : <p className="text-center text-muted-foreground py-12 text-sm">No data loaded yet.</p>
            }
          </TabsContent>
        </Tabs>
      </div>
    </AppLayout>
  )
}
