"use client"

import React, { useState, useEffect, useCallback, useMemo, useRef, memo, Suspense, useDeferredValue } from "react"
import { useSearchParams } from "next/navigation"
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
import { forceCollide } from "d3-force-3d"
import { ReactFlow, Background, BackgroundVariant, Handle, Position, getBezierPath, applyNodeChanges, type NodeProps, type EdgeProps, type NodeChange } from "@xyflow/react"
import "@xyflow/react/dist/style.css"
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
const ForceGraph3D = dynamic(() => import("react-force-graph-3d"), { ssr: false })

// Polyfill GPUShaderStage to prevent errors when WebGPU is not available
if (typeof window !== 'undefined' && typeof (window as any).GPUShaderStage === 'undefined') {
  ;(window as any).GPUShaderStage = { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 }
}

// ── Types ─────────────────────────────────────────────────────────────────────

interface AppNode {
  id: string
  name?: string
  component_name?: string
  config_item_name?: string
  system_name?: string
  domain_name?: string
  csu?: string
  weight?: number
  [key: string]: unknown
}

// Hierarchy: System (CSMS) → Domain (CSS) → Config Item (CSCI) → Component (CSC) → App (CSU)
interface CscGroup  { name: string; apps: AppNode[] }
interface CsciGroup { name: string; csc: Record<string, CscGroup> }
interface CssGroup  { name: string; csci: Record<string, CsciGroup> }
interface CsmsGroup { name: string; css: Record<string, CssGroup> }

type SelectedKind = "csms" | "css" | "csci" | "csc" | "app" | "node" | "topic"
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
  nodeType?: string   // non-hierarchy node type (Topic, Node, Broker, Library …)
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
    const csmsKey = app.system_name?.trim() || OTHER
    const cssKey  = app.domain_name?.trim()  || OTHER
    const csciKey = app.config_item_name?.trim() || OTHER
    const cscKey  = app.component_name?.trim()  || OTHER

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
  return f(app.id) || f(app.name) || f(app.component_name) || f(app.config_item_name) ||
         f(app.system_name) || f(app.domain_name) || f(app.csu)
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
  csms: "System (CSMS)", css: "Domain (CSS)", csci: "Config Item (CSCI)", csc: "Component (CSC)", app: "App (CSU)",
}

// Hierarchical connections-view layout: assign a y-layer per node type
const CONN_TYPE_LAYER: Record<string, number> = {
  Node: 0, Broker: 1, Application: 2, Topic: 3, Library: 4,
}
// Fraction of canvas height for each layer (top → bottom)
const CONN_LAYER_Y_FRACS = [0.10, 0.28, 0.50, 0.72, 0.90]
const CONN_LAYER_LABEL: Record<number, string> = {
  0: "Node", 1: "Broker", 2: "Application", 3: "Topic", 4: "Library",
}

// Node-type → color (matches Explorer page — theme-aware via isDark flag passed at callsite)
// Dark variants used at runtime; light variants noted for reference.
const CONN_NODE_TYPE_COLORS_DARK: Record<string, string> = {
  Application: "#3b82f6",
  Node:        "#ef4444",
  Broker:      "#a1a1aa",
  Topic:       "#facc15",
  Library:     "#06b6d4",
}
const CONN_NODE_TYPE_COLORS_LIGHT: Record<string, string> = {
  Application: "#2563eb",
  Node:        "#dc2626",
  Broker:      "#71717a",
  Topic:       "#eab308",
  Library:     "#0891b2",
}
// Link-type → color (matches Explorer page)
const CONN_LINK_TYPE_COLORS_DARK: Record<string, string> = {
  RUNS_ON:       "#a855f7",
  PUBLISHES_TO:  "#22c55e",
  SUBSCRIBES_TO: "#f97316",
  DEPENDS_ON:    "#ef4444",
  CONNECTS_TO:   "#22c55e",
  ROUTES:        "#a1a1aa",
  USES:          "#06b6d4",
}
const CONN_LINK_TYPE_COLORS_LIGHT: Record<string, string> = {
  RUNS_ON:       "#9333ea",
  PUBLISHES_TO:  "#16a34a",
  SUBSCRIBES_TO: "#ea580c",
  DEPENDS_ON:    "#dc2626",
  CONNECTS_TO:   "#16a34a",
  ROUTES:        "#71717a",
  USES:          "#0891b2",
}
/** Deterministic fallback color for unknown types */
function hashTypeColor(type: string): string {
  let h = 0
  for (let i = 0; i < type.length; i++) h = (Math.imul(31, h) + type.charCodeAt(i)) | 0
  const hue = Math.abs(h) % 360
  return `hsl(${hue},65%,55%)`
}
function nodeTypeColor(type: string | undefined, isDark = true): string {
  if (!type) return isDark ? "#a1a1aa" : "#71717a"
  return (isDark ? CONN_NODE_TYPE_COLORS_DARK : CONN_NODE_TYPE_COLORS_LIGHT)[type] ?? hashTypeColor(type)
}
function linkTypeColor(type: string | undefined, isDark = true): string {
  if (!type) return isDark ? "#a1a1aa" : "#71717a"
  return (isDark ? CONN_LINK_TYPE_COLORS_DARK : CONN_LINK_TYPE_COLORS_LIGHT)[type] ?? hashTypeColor(type)
}

// ── Shared ReactFlow prop constants (hoisted to avoid re-renders) ─────────────
const RF_NODE_ORIGIN: [number, number] = [0.5, 0.5]
const RF_FIT_VIEW_OPTIONS = { padding: 0.25 }
const RF_STYLE_TRANSPARENT = { background: "transparent" }
const RF_STYLE_CONN = { background: "transparent", position: "relative" as const, zIndex: 1 }
const RF_PRO_OPTIONS = { hideAttribution: true }

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

// ── SVG-based connections graph (replaces react-force-graph-2d for swimlane view) ──────────────
function ConnSvgGraph({
  graphData, positions, dims, isDark, populatedLayers,
  selectedAppId, selectedLink, onNodeClick, onEdgeClick, onBackgroundClick,
}: {
  graphData: { nodes: any[]; links: any[] }
  positions: Map<string, { x: number; y: number }>
  dims: { width: number; height: number }
  isDark: boolean
  populatedLayers: Set<number>
  selectedAppId: string | null
  selectedLink: { link: any } | null
  onNodeClick: (node: any) => void
  onEdgeClick: (link: any, event: { clientX: number; clientY: number }) => void
  onBackgroundClick: () => void
}) {
  const W = dims.width || 800
  const H = dims.height || 600
  const svgRef = useRef<SVGSVGElement>(null)
  const [vp, setVp] = useState({ x: 0, y: 0, k: 1 })
  const dragRef = useRef<{ ox: number; oy: number; vx: number; vy: number } | null>(null)
  const movedRef = useRef(false)

  // Reset viewport when canvas size changes
  useEffect(() => { setVp({ x: 0, y: 0, k: 1 }) }, [W, H])

  // World coords (centered at 0,0) → screen pixel coords
  const sx = (wx: number) => wx + W / 2
  const sy = (wy: number) => wy + H / 2

  // Edge weight → stroke width
  const weightScale = useMemo(() => {
    const weights = graphData.links.map((l: any) => Number(l.weight ?? 1)).filter((w: number) => isFinite(w))
    if (!weights.length) return () => 1.5
    const lo = Math.min(...weights), hi = Math.max(...weights)
    if (lo === hi) return () => 2.0
    return (w: number) => 0.8 + ((w - lo) / (hi - lo)) * 3.2
  }, [graphData.links])

  // One SVG arrowhead marker per unique link color
  const markerColors = useMemo(() => {
    const s = new Set<string>()
    graphData.links.forEach((l: any) => s.add(linkTypeColor(l.type, isDark)))
    s.add("#f59e0b") // selected highlight colour
    return Array.from(s)
  }, [graphData.links, isDark])

  const mid = (color: string) => `arr-${color.replace(/[^0-9a-f]/gi, "x")}`

  function handleWheel(e: React.WheelEvent) {
    e.preventDefault()
    const rect = svgRef.current!.getBoundingClientRect()
    const mx = e.clientX - rect.left, my = e.clientY - rect.top
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12
    setVp(v => {
      const k = Math.max(0.15, Math.min(8, v.k * factor))
      return { k, x: mx - (mx - v.x) * (k / v.k), y: my - (my - v.y) * (k / v.k) }
    })
  }
  function handleMouseDown(e: React.MouseEvent) {
    movedRef.current = false
    dragRef.current = { ox: e.clientX, oy: e.clientY, vx: vp.x, vy: vp.y }
  }
  function handleMouseMove(e: React.MouseEvent) {
    if (!dragRef.current) return
    const dx = e.clientX - dragRef.current.ox, dy = e.clientY - dragRef.current.oy
    if (Math.abs(dx) + Math.abs(dy) > 4) movedRef.current = true
    setVp(v => ({ ...v, x: dragRef.current!.vx + dx, y: dragRef.current!.vy + dy }))
  }
  function handleMouseUp() { dragRef.current = null }

  return (
    <svg ref={svgRef} width={W} height={H}
      style={{ display: "block", cursor: dragRef.current ? "grabbing" : "grab" }}
      onWheel={handleWheel} onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}
      onClick={e => { if (!movedRef.current && e.target === svgRef.current) onBackgroundClick() }}
    >
      <defs>
        {markerColors.map(c => (
          <marker key={c} id={mid(c)} markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
            <path d="M0,0.5 L0,6.5 L6,3.5 z" fill={c} />
          </marker>
        ))}
      </defs>
      <g transform={`translate(${vp.x},${vp.y}) scale(${vp.k})`}>
        {/* Swimlane bands */}
        {CONN_LAYER_Y_FRACS.map((yFrac, layer) => {
          if (!populatedLayers.has(layer)) return null
          const yPx = yFrac * H, bandH = H * 0.16
          const typeKey = Object.keys(CONN_TYPE_LAYER).find(t => CONN_TYPE_LAYER[t] === layer) ?? ""
          const tc = (isDark ? CONN_NODE_TYPE_COLORS_DARK : CONN_NODE_TYPE_COLORS_LIGHT)[typeKey] ?? "#888"
          return (
            <g key={layer}>
              <rect x={0} y={yPx - bandH / 2} width={W} height={bandH} fill={tc + "08"} />
              <line x1={0} y1={yPx - bandH / 2} x2={W} y2={yPx - bandH / 2} stroke={tc + "22"} strokeWidth={1} />
              <line x1={0} y1={yPx + bandH / 2} x2={W} y2={yPx + bandH / 2} stroke={tc + "22"} strokeWidth={1} />
              <text x={10} y={yPx} dominantBaseline="middle" fontSize={9} fontWeight="600"
                fill={tc + "aa"} style={{ userSelect: "none" as const, letterSpacing: "0.08em" }}>
                {String(CONN_LAYER_LABEL[layer] ?? "").toUpperCase()}
              </text>
            </g>
          )
        })}
        {/* Edges */}
        {graphData.links.map((link: any, i: number) => {
          const srcId = typeof link.source === "object" ? link.source.id : link.source
          const tgtId = typeof link.target === "object" ? link.target.id : link.target
          const sp = positions.get(srcId), ep = positions.get(tgtId)
          if (!sp || !ep) return null
          const x1 = sx(sp.x), y1 = sy(sp.y), x2 = sx(ep.x), y2 = sy(ep.y)
          const len = Math.hypot(x2 - x1, y2 - y1)
          if (len < 5) return null
          const ux = (x2 - x1) / len, uy = (y2 - y1) / len
          const sr = (srcId === selectedAppId ? 10 : 6) + 2
          const tr = (tgtId === selectedAppId ? 10 : 6) + 14
          const isHl = selectedLink?.link === link
          const color = isHl ? "#f59e0b" : linkTypeColor(link.type, isDark)
          const isAdj = srcId === selectedAppId || tgtId === selectedAppId
          const sw = isHl ? 5 : Math.min(isAdj ? weightScale(Number(link.weight ?? 1)) + 0.8 : weightScale(Number(link.weight ?? 1)), 5)
          return (
            <line key={i}
              x1={x1 + ux * sr} y1={y1 + uy * sr}
              x2={x2 - ux * tr} y2={y2 - uy * tr}
              stroke={color} strokeWidth={sw}
              markerEnd={`url(#${mid(color)})`}
              style={{ cursor: "pointer" }}
              onClick={e => { e.stopPropagation(); if (!movedRef.current) onEdgeClick(link, e) }}
            />
          )
        })}
        {/* Nodes */}
        {graphData.nodes.map((n: any) => {
          const pos = positions.get(n.id)
          if (!pos) return null
          const nx = sx(pos.x), ny = sy(pos.y)
          const isCenter = n.id === selectedAppId
          const r = isCenter ? 10 : 6
          const color = nodeTypeColor(n.type, isDark)
          const strokeColor = isCenter ? (isDark ? "#ffffff" : "#111111") : (isDark ? "rgba(255,255,255,0.3)" : "rgba(30,41,59,0.35)")
          const sw = isCenter ? 2.5 : 1
          let shape: React.ReactNode
          switch (n.type) {
            case "Node":
              shape = <rect x={nx - r} y={ny - r} width={r * 2} height={r * 2} fill={color} stroke={strokeColor} strokeWidth={sw} />
              break
            case "Topic":
              shape = <polygon points={`${nx},${ny - r} ${nx + r},${ny} ${nx},${ny + r} ${nx - r},${ny}`} fill={color} stroke={strokeColor} strokeWidth={sw} />
              break
            case "Library":
              shape = <polygon points={`${nx},${ny - r} ${nx + r},${ny + r * 0.6} ${nx - r},${ny + r * 0.6}`} fill={color} stroke={strokeColor} strokeWidth={sw} />
              break
            case "Broker": {
              const pts = Array.from({ length: 6 }, (_, i) => {
                const a = (Math.PI * 2 / 6) * i - Math.PI / 2
                return `${nx + r * Math.cos(a)},${ny + r * Math.sin(a)}`
              }).join(" ")
              shape = <polygon points={pts} fill={color} stroke={strokeColor} strokeWidth={sw} />
              break
            }
            default:
              shape = <>{isCenter && <circle cx={nx} cy={ny} r={r + 7} fill={color + "33"} />}<circle cx={nx} cy={ny} r={r} fill={color} stroke={strokeColor} strokeWidth={sw} /></>
          }
          const label = String(n.label ?? n.id ?? "?")
          const disp = label.length > 22 ? label.slice(0, 20) + "…" : label
          return (
            <g key={n.id} onClick={e => { e.stopPropagation(); if (!movedRef.current) onNodeClick(n) }} style={{ cursor: "pointer" }}>
              {shape}
              <text x={nx} y={ny + r + 11} textAnchor="middle" fontSize={10}
                fontWeight={isCenter ? "bold" : "normal"} fill={isDark ? "#e5e7eb" : "#374151"}>
                {disp}
              </text>
              <text x={nx} y={ny + r + 20} textAnchor="middle" fontSize={7} fill={isDark ? "#9ca3af" : "#6b7280"}>
                {n.type}
              </text>
              <circle cx={nx} cy={ny} r={r + 10} fill="transparent" />
            </g>
          )
        })}
      </g>
    </svg>
  )
}

// ── React Flow connections graph ─────────────────────────────────────────────
const ConnFlowNode = memo(function ConnFlowNode({ data }: NodeProps) {
  const { n, isDark, isCenter } = data as any
  // Uniform canvas size for all shapes so labels align consistently
  const S  = isCenter ? 44 : 28   // bounding box side
  const C  = S / 2                // center coord
  const color = nodeTypeColor(n.type, isDark)
  const label = String(n.label ?? n.id ?? "?")
  const disp = label.length > 20 ? label.slice(0, 18) + "…" : label

  // Shared style tokens
  const fill   = color
  const stroke = isDark ? "rgba(255,255,255,0.18)" : "rgba(0,0,0,0.18)"
  const sw     = isCenter ? 2 : 1.5
  const glow   = color + (isDark ? "40" : "28")


  let shape: React.ReactNode
  switch (n.type) {

    // ── Application → Circle ─────────────────────────────────────────────────
    case "Application":
    default: {
      const r = C
      shape = <>
        <circle cx={C} cy={C} r={r + (isCenter ? 8 : 5)} fill={glow} />
        <circle cx={C} cy={C} r={r} fill={fill} stroke={stroke} strokeWidth={sw} />
      </>
      break
    }

    // ── Node → Square ────────────────────────────────────────────────────────
    case "Node": {
      // Squares appear optically larger than circles at same S — shrink by ~14%
      const sq  = S * 0.86
      const off = (S - sq) / 2
      const r   = isCenter ? 5 : 3
      shape = <>
        <rect x={off - 5} y={off - 5} width={sq + 10} height={sq + 10} rx={r + 3} fill={glow} />
        <rect x={off} y={off} width={sq} height={sq} rx={r} fill={fill} stroke={stroke} strokeWidth={sw} />
      </>
      break
    }

    // ── Topic → Rhombus ──────────────────────────────────────────────────────
    case "Topic": {
      shape = <>
        <polygon points={`${C},${-4} ${S + 4},${C} ${C},${S + 4} ${-4},${C}`} fill={glow} />
        <polygon points={`${C},0 ${S},${C} ${C},${S} 0,${C}`} fill={fill} stroke={stroke} strokeWidth={sw} />
      </>
      break
    }

    // ── Broker → Pentagon ────────────────────────────────────────────────────
    case "Broker": {
      const penta = (scale: number, ox = C, oy = C) =>
        Array.from({ length: 5 }, (_, i) => {
          const a = (Math.PI * 2 / 5) * i - Math.PI / 2
          return `${ox + scale * Math.cos(a)},${oy + scale * Math.sin(a)}`
        }).join(" ")
      shape = <>
        <polygon points={penta(C + 6)} fill={glow} />
        <polygon points={penta(C)} fill={fill} stroke={stroke} strokeWidth={sw} />
      </>
      break
    }

    // ── Library → Triangle ───────────────────────────────────────────────────
    case "Library": {
      const h = S * 0.866
      const yOff = (S - h) / 2
      shape = <>
        <polygon points={`${C},${yOff - 6} ${S + 7},${yOff + h + 4} ${-7},${yOff + h + 4}`} fill={glow} />
        <polygon points={`${C},${yOff} ${S},${yOff + h} 0,${yOff + h}`} fill={fill} stroke={stroke} strokeWidth={sw} />
      </>
      break
    }
  }

  const hs = { width: 1, height: 1, opacity: 0, border: "none", background: "transparent", minWidth: 0, minHeight: 0 }
  const textBg    = isDark ? "rgba(14,14,22,0.85)" : "rgba(255,255,255,0.90)"
  const textColor = isDark ? "#f1f5f9" : "#1e293b"
  const subColor  = isDark ? "#94a3b8" : "#64748b"
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", cursor: "pointer", userSelect: "none" }}>
      <Handle type="target" position={Position.Top}    id="t-top" style={hs} />
      <Handle type="source" position={Position.Top}    id="s-top" style={hs} />
      <svg width={S} height={S} overflow="visible" style={{ display: "block" }}>
        {shape}
        {isCenter && (
          <circle cx={C} cy={C} r={C + 10} fill="none"
            stroke={isDark ? "rgba(255,255,255,0.65)" : "rgba(0,0,0,0.50)"}
            strokeWidth={1.5} strokeDasharray="4 3" />
        )}
      </svg>
      <div style={{
        marginTop: 6,
        padding: "2px 9px",
        background: textBg,
        border: `1px solid ${color}44`,
        borderRadius: 20,
        backdropFilter: "blur(8px)",
        fontSize: isCenter ? 12 : 10,
        fontWeight: isCenter ? 700 : 500,
        color: textColor,
        whiteSpace: "nowrap",
        maxWidth: 160,
        overflow: "hidden",
        textOverflow: "ellipsis",
        lineHeight: "1.5",
        letterSpacing: "0.01em",
        boxShadow: `0 2px 6px ${isDark ? "rgba(0,0,0,0.6)" : "rgba(0,0,0,0.10)"}`,
      }}>{disp}</div>
      <div style={{
        marginTop: 2,
        fontSize: 8,
        fontWeight: 600,
        color: color,
        opacity: 0.75,
        whiteSpace: "nowrap",
        letterSpacing: "0.10em",
        textTransform: "uppercase",
      }}>{n.type}</div>
      <Handle type="source" position={Position.Bottom} id="s-bot" style={hs} />
      <Handle type="target" position={Position.Bottom} id="t-bot" style={hs} />
    </div>
  )
})

// Swimlane background node — rendered inside RF so it moves with fitView
const ConnLaneNode = memo(function ConnLaneNode({ data }: NodeProps) {
  const { label, color, width, height, isDark } = data as any
  return (
    <div style={{
      width, height,
      borderTop: `1px solid ${color}1a`,
      borderBottom: `1px solid ${color}1a`,
      background: isDark
        ? `linear-gradient(90deg, ${color}18 0%, ${color}08 25%, transparent 70%)`
        : `linear-gradient(90deg, ${color}12 0%, ${color}05 25%, transparent 70%)`,
      pointerEvents: "none",
      display: "flex",
      alignItems: "center",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 5, paddingLeft: 10 }}>
        <div style={{ width: 3, height: 18, borderRadius: 2, background: color, opacity: 0.55 }} />
        <span style={{ fontSize: 9, fontWeight: 700, letterSpacing: "0.12em",
          textTransform: "uppercase", color, opacity: 0.65, userSelect: "none" }}>
          {label}
        </span>
      </div>
    </div>
  )
})
const cfNodeTypes = { conn: ConnFlowNode, lane: ConnLaneNode }

// Custom edge: smooth bezier + clean fixed-size filled arrowhead
const ConnFlowEdge = memo(function ConnFlowEdge({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, style }: EdgeProps) {
  // Fixed arrowhead dimensions — same size on every edge
  const AW = 5  // half-width
  const AH = 8  // height
  const color = (style?.stroke as string) ?? "#888"
  const sw = Number(style?.strokeWidth ?? 2)
  const opacity = Number(style?.opacity ?? 1)
  const dashArray = (style as any)?.strokeDasharray as string | undefined
  // Handles are always top/bottom — bezier arrives vertically at target
  const goingDown = targetPosition === "top"
  const dir = goingDown ? 1 : -1
  // Shorten path so the line ends at arrowhead base, not tip — no overlap
  const adjustedTY = targetY - dir * AH
  const [edgePath] = getBezierPath({
    sourceX, sourceY, sourcePosition,
    targetX, targetY: adjustedTY,
    targetPosition, curvature: 0.35,
  })
  const tipY  = targetY
  const baseY = adjustedTY
  const arrowPts = `${targetX},${tipY} ${targetX - AW},${baseY} ${targetX + AW},${baseY}`
  return (
    <g opacity={opacity}>
      <path d={edgePath} fill="none" stroke="transparent" strokeWidth={Math.max(sw + 10, 14)} />
      <path d={edgePath} fill="none" stroke={color} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round" strokeDasharray={dashArray} />
      <polygon points={arrowPts} fill={color} />
    </g>
  )
})
const cfEdgeTypes = { conn: ConnFlowEdge }

// ── Hierarchy graph using @xyflow/react ──────────────────────────────────────

const HIER_LEVEL_LABEL: Record<HGLevel, string> = {
  csms: "System", css: "Domain", csci: "Config Item", csc: "Component", app: "App",
}

const HierFlowNode = memo(function HierFlowNode({ data }: NodeProps) {
  const { n, isDark, isSelected, isParent } = data as any
  const hn = n as HGNode
  const color = NODE_COLORS[hn.level]
  const levelLabel = HIER_LEVEL_LABEL[hn.level] ?? hn.level

  // Parent (drilled-into) node is larger
  const pad = isParent ? "8px 18px" : "6px 14px"
  const fontSize = isParent ? 14 : 12
  const maxLabelWidth = isParent ? 220 : 180
  const dotSize = isParent ? 9 : 7

  const bgAlpha = isDark ? (isParent ? "30" : "1a") : (isParent ? "20" : "12")
  const borderAlpha = isSelected ? "" : (isParent ? "88" : "44")
  const borderColor = isSelected ? color : `${color}${borderAlpha}`
  const shadow = isSelected
    ? `0 0 0 2.5px ${color}50, 0 4px 16px ${color}30`
    : isParent
    ? `0 2px 12px ${color}25`
    : `0 1px 4px ${isDark ? "rgba(0,0,0,0.4)" : "rgba(0,0,0,0.08)"}`

  const hiddenHandle = { opacity: 0, width: 0, height: 0, minWidth: 0 }

  return (
    <div
      style={{
        display: "flex", flexDirection: "column", alignItems: "center", gap: 3,
        padding: pad,
        borderRadius: 12,
        background: isDark ? `${color}${bgAlpha}` : `${color}${bgAlpha}`,
        border: `1.5px solid ${borderColor}`,
        boxShadow: shadow,
        cursor: "pointer",
        userSelect: "none",
        whiteSpace: "nowrap",
        backdropFilter: "blur(8px)",
        transition: "box-shadow 0.15s, border-color 0.15s",
      }}
    >
      <Handle type="target" position={Position.Top} id="t" style={hiddenHandle} />
      {/* Level badge */}
      <span style={{
        fontSize: 8, fontWeight: 700, letterSpacing: "0.08em",
        textTransform: "uppercase", color: isDark ? `${color}99` : `${color}aa`,
        lineHeight: 1,
      }}>{levelLabel}</span>
      {/* Name row */}
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{
          width: dotSize, height: dotSize, borderRadius: "50%", background: color,
          flexShrink: 0, display: "inline-block",
          boxShadow: `0 0 6px ${color}66`,
        }} />
        <span style={{
          maxWidth: maxLabelWidth, overflow: "hidden", textOverflow: "ellipsis",
          fontSize, fontWeight: isParent ? 700 : isSelected ? 600 : 500,
          color: isDark ? "#f1f5f9" : "#1e293b",
          lineHeight: 1.3,
        }}>{hn.name}</span>
      </div>
      {/* Count badge */}
      {hn.appCount > 0 && hn.level !== "app" && (
        <span style={{
          fontSize: 9, fontWeight: 600,
          color: isDark ? "#94a3b8" : "#64748b",
          background: isDark ? "rgba(255,255,255,0.06)" : "rgba(0,0,0,0.04)",
          borderRadius: 6, padding: "1px 6px",
          lineHeight: 1.4,
        }}>{hn.appCount} {hn.appCount === 1 ? "app" : "apps"}</span>
      )}
      <Handle type="source" position={Position.Bottom} id="s" style={hiddenHandle} />
    </div>
  )
})

// Custom hierarchy edge with gradient stroke
const HierFlowEdge = memo(function HierFlowEdge({ sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, data }: EdgeProps) {
  const { isDark, sourceColor, targetColor } = (data ?? {}) as any
  const sc = sourceColor ?? (isDark ? "#ffffff" : "#000000")
  const tc = targetColor ?? (isDark ? "#ffffff" : "#000000")
  const gradId = `hg-${Math.round(sourceX)}-${Math.round(targetX)}`

  // Vertical tree edge: straight down from source, then curve to target
  const midY = sourceY + (targetY - sourceY) * 0.5
  const d = `M ${sourceX} ${sourceY} C ${sourceX} ${midY}, ${targetX} ${midY}, ${targetX} ${targetY}`

  return (
    <g>
      <defs>
        <linearGradient id={gradId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={sc} stopOpacity={isDark ? 0.5 : 0.35} />
          <stop offset="100%" stopColor={tc} stopOpacity={isDark ? 0.3 : 0.2} />
        </linearGradient>
      </defs>
      <path d={d} fill="none" stroke={`url(#${gradId})`} strokeWidth={2} strokeLinecap="round" />
    </g>
  )
})

const cfHierNodeTypes = { hier: HierFlowNode }
const cfHierEdgeTypes = { hier: HierFlowEdge }

const HierFlowGraph = memo(function HierFlowGraph({ graphData, dims, isDark, selectedNodeId, onNodeClick }: {
  graphData: { nodes: HGNode[]; links: HGLink[] }
  dims: { width: number; height: number }
  isDark: boolean
  selectedNodeId: string | null
  onNodeClick: (n: HGNode) => void
}) {
  const W = dims.width  || 800
  const H = dims.height || 600
  const LEVEL_ORDER: HGLevel[] = ["csms", "css", "csci", "csc", "app"]

  // Find which node is the parent (drilled-into) node — first node that has children
  const parentNodeId = useMemo(() => {
    if (graphData.nodes.length <= 1) return null
    const childIds = new Set(graphData.links.map(l => typeof l.target === "object" ? (l.target as any).id : l.target))
    return graphData.nodes.find(n => !childIds.has(n.id))?.id ?? graphData.nodes[0]?.id ?? null
  }, [graphData.nodes, graphData.links])

  // Tree layout: parent centered at top, children spaced evenly below
  const positions = useMemo(() => {
    const map = new Map<string, { x: number; y: number }>()
    const byLevel = new Map<HGLevel, HGNode[]>()
    for (const n of graphData.nodes) {
      if (!byLevel.has(n.level)) byLevel.set(n.level, [])
      byLevel.get(n.level)!.push(n)
    }

    // Collect present levels in order
    const presentLevels = LEVEL_ORDER.filter(l => byLevel.has(l))
    if (presentLevels.length === 0) return map

    // Vertical spacing: use fraction of height per tier
    const yStep = presentLevels.length === 1 ? 0 : H / (presentLevels.length + 0.5)
    const yStart = presentLevels.length === 1 ? H * 0.4 : yStep * 0.75

    for (let li = 0; li < presentLevels.length; li++) {
      const level = presentLevels[li]
      const arr = byLevel.get(level) ?? []
      const y = yStart + li * yStep

      if (arr.length === 1) {
        // Single node (root or only child) — center it
        map.set(arr[0].id, { x: W / 2, y })
      } else {
        // Equal spacing with enough room for node cards (~200px wide)
        const minGap = 200
        const naturalWidth = (arr.length - 1) * minGap
        const margin = Math.max(W * 0.12, 100)
        const usable = Math.max(W - 2 * margin, naturalWidth)
        // If children need more space than canvas, let fitView handle zoom
        const startX = (W - usable) / 2
        arr.forEach((n, i) => {
          const x = arr.length === 1 ? W / 2 : startX + (i / (arr.length - 1)) * usable
          map.set(n.id, { x, y })
        })
      }
    }
    return map
  }, [graphData.nodes, W, H])

  // Build node-to-color map for edge gradients
  const nodeColorMap = useMemo(() => {
    const m = new Map<string, string>()
    for (const n of graphData.nodes) m.set(n.id, NODE_COLORS[n.level] ?? "#888")
    return m
  }, [graphData.nodes])

  const initialHierNodes = useMemo(() => graphData.nodes.map(n => ({
    id: n.id,
    type: "hier" as const,
    position: positions.get(n.id) ?? { x: W / 2, y: H / 2 },
    data: { n, isDark, isSelected: n.id === selectedNodeId, isParent: n.id === parentNodeId },
    selectable: false,
    draggable: true,
  })), [graphData.nodes, positions, isDark, selectedNodeId, parentNodeId])

  const [rfNodes, setRfNodes] = useState(initialHierNodes)
  useEffect(() => { setRfNodes(initialHierNodes) }, [initialHierNodes])
  const onNodesChange = useCallback(
    (changes: NodeChange[]) => setRfNodes(nds => applyNodeChanges(changes, nds) as any),
    [],
  )

  const rfEdges = useMemo(() => graphData.links.map((l: any, i: number) => {
    const srcId = typeof l.source === "object" ? l.source.id : l.source
    const tgtId = typeof l.target === "object" ? l.target.id : l.target
    return {
      id: `he${i}`,
      source: srcId,
      target: tgtId,
      type: "hier",
      data: { isDark, sourceColor: nodeColorMap.get(srcId), targetColor: nodeColorMap.get(tgtId) },
    }
  }), [graphData.links, isDark, nodeColorMap])

  const handleNodeClick = useCallback((_: any, rfNode: any) => onNodeClick(rfNode.data.n as HGNode), [onNodeClick])

  return (
    <ReactFlow
      nodes={rfNodes}
      edges={rfEdges}
      nodeTypes={cfHierNodeTypes}
      edgeTypes={cfHierEdgeTypes}
      nodeOrigin={RF_NODE_ORIGIN}
      fitView
      fitViewOptions={RF_FIT_VIEW_OPTIONS}
      nodesDraggable
      onNodesChange={onNodesChange}
      onNodeClick={handleNodeClick}
      panOnDrag
      zoomOnScroll
      style={RF_STYLE_TRANSPARENT}
      proOptions={RF_PRO_OPTIONS}
    >
      <Background variant={BackgroundVariant.Dots} gap={24} size={1}
        color={isDark ? "#ffffff10" : "#00000010"} />
    </ReactFlow>
  )
})

const ConnFlowGraph = memo(function ConnFlowGraph({ graphData, positions, dims, isDark, populatedLayers, selectedAppId, selectedLink, onNodeClick, onEdgeClick, onBackgroundClick }: {
  graphData: { nodes: any[]; links: any[] }
  positions: Map<string, { x: number; y: number }>
  dims: { width: number; height: number }
  isDark: boolean
  populatedLayers: Set<number>
  selectedAppId: string | null
  selectedLink: { link: any } | null
  onNodeClick: (n: any) => void
  onEdgeClick: (link: any, event: React.MouseEvent) => void
  onBackgroundClick: () => void
}) {
  const W = dims.width  || 800
  const H = dims.height || 600
  const bandH = H * 0.20

  const initialNodes = useMemo(() => {
    // Swimlane lane nodes (rendered behind, z-index -1 via RF zIndex)
    const laneNodes = CONN_LAYER_Y_FRACS.map((yFrac, layer) => {
      if (!populatedLayers.has(layer)) return null
      const typeKey = Object.keys(CONN_TYPE_LAYER).find(t => CONN_TYPE_LAYER[t] === layer) ?? ""
      const color = (isDark ? CONN_NODE_TYPE_COLORS_DARK : CONN_NODE_TYPE_COLORS_LIGHT)[typeKey] ?? "#888"
      return {
        id: `lane-${layer}`,
        type: "lane" as const,
        position: { x: 0, y: yFrac * H - bandH / 2 },
        origin: [0, 0] as [number, number],
        data: { label: CONN_LAYER_LABEL[layer] ?? "", color, width: W, height: bandH, isDark },
        selectable: false,
        draggable: false,
        zIndex: -1,
      }
    }).filter(Boolean) as any[]

    const connNodes = graphData.nodes.map((n: any) => {
      const pos = positions.get(n.id) ?? { x: W / 2, y: H / 2 }
      return {
        id: n.id,
        type: "conn" as const,
        position: pos,
        data: { n, isDark, isCenter: n.id === selectedAppId },
        selectable: false,
        draggable: true,
        zIndex: 1,
      }
    })
    return [...laneNodes, ...connNodes]
  }, [graphData.nodes, positions, isDark, selectedAppId, dims, populatedLayers])

  const [rfNodes, setRfNodes] = useState(initialNodes)
  useEffect(() => { setRfNodes(initialNodes) }, [initialNodes])
  const onNodesChange = useCallback(
    (changes: NodeChange[]) => setRfNodes(nds => applyNodeChanges(changes, nds) as any),
    [],
  )

  const rfEdges = useMemo(() => {
    const weights = graphData.links.map((l: any) => Number(l.weight ?? 1)).filter((w: number) => isFinite(w))
    const lo = weights.length ? Math.min(...weights) : 0
    const hi = weights.length ? Math.max(...weights) : 1
    const wScale = lo === hi ? () => 2.0 : (w: number) => 1.0 + ((w - lo) / (hi - lo)) * 2.5
    return graphData.links.map((l: any, i: number) => {
      const srcId = typeof l.source === "object" ? l.source.id : l.source
      const tgtId = typeof l.target === "object" ? l.target.id : l.target
      const srcY = positions.get(srcId)?.y ?? 0
      const tgtY = positions.get(tgtId)?.y ?? 0
      const goingDown = tgtY >= srcY
      const sourceHandle = goingDown ? "s-bot" : "s-top"
      const targetHandle = goingDown ? "t-top" : "t-bot"
      const isHl = selectedLink?.link === l
      const isAdj = srcId === selectedAppId || tgtId === selectedAppId
      const isDerived = l.type === "DEPENDS_ON"
      const color = isHl ? "#f59e0b" : linkTypeColor(l.type, isDark)
      const sw = isHl ? 6 : isAdj ? Math.min(wScale(Number(l.weight ?? 1)) + 0.5, 4) : 1
      const opacity = isHl ? 1 : isAdj ? 0.88 : 0.2
      return {
        id: `e${i}`,
        source: srcId,
        target: tgtId,
        sourceHandle,
        targetHandle,
        type: "conn",
        style: { stroke: color, strokeWidth: sw, opacity, ...(isDerived ? { strokeDasharray: "6 4" } : {}) },
        data: { link: l },
        selectable: false,
      }
    })
  }, [graphData.links, positions, isDark, selectedLink, selectedAppId])

  const handleNodeClick = useCallback((_: any, node: any) => onNodeClick((node.data as any).n), [onNodeClick])
  const handleEdgeClick = useCallback((event: any, edge: any) => onEdgeClick((edge.data as any).link, event), [onEdgeClick])

  return (
    <div style={{ width: dims.width, height: dims.height, position: "relative" }}>
      <ReactFlow
        nodes={rfNodes}
        edges={rfEdges as any}
        nodeTypes={cfNodeTypes}
        edgeTypes={cfEdgeTypes}
        nodeOrigin={RF_NODE_ORIGIN}
        fitView
        fitViewOptions={RF_FIT_VIEW_OPTIONS}
        nodesDraggable
        nodesConnectable={false}
        elementsSelectable={false}
        onNodesChange={onNodesChange}
        onNodeClick={handleNodeClick}
        onEdgeClick={handleEdgeClick}
        onPaneClick={onBackgroundClick}
        style={RF_STYLE_CONN}
        proOptions={RF_PRO_OPTIONS}
      >
        <Background variant={BackgroundVariant.Dots} color={isDark ? "#3f3f46" : "#d4d4d8"} gap={28} size={1.5} />
      </ReactFlow>
    </div>
  )
})

function HierarchyGraph({ hierarchy, extraNodes = [], initialNodeId = null, syncKey = null, onNodeSelect }: { hierarchy: Record<string, CsmsGroup>; extraNodes?: any[]; initialNodeId?: string | null; syncKey?: string | null; onNodeSelect?: (key: string) => void }) {
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
  const [is3D, setIs3D] = useState(false)
  const [threeReady, setThreeReady] = useState(false)

  // Load Three.js and SpriteText for 3D labels
  useEffect(() => {
    if (typeof window === 'undefined') return
    Promise.all([
      import('three-spritetext').then(m => { (window as any).__SpriteText = m.default }),
      import('three').then(m => { (window as any).__THREE = m }),
    ]).then(() => setThreeReady(true)).catch(() => {})
  }, [])

  const [connData, setConnData] = useState<{ nodes: any[]; links: any[] } | null>(null)
  const [connLoading, setConnLoading] = useState(false)
  const [connError, setConnError] = useState<string | null>(null)
  const [connTab, setConnTab] = useState<"connections" | "props">("props")
  const [connDepth, setConnDepth] = useState(1)
  const [connSort, setConnSort] = useState<{ col: "node" | "type" | "dir"; asc: boolean }>({ col: "type", asc: true })

  const isSyncingRef = useRef(false)

  const [hiddenLevels, setHiddenLevels] = useState<Set<HGLevel>>(new Set())
  const [hiddenNodeTypes, setHiddenNodeTypes] = useState<Set<string>>(new Set())
  const [hiddenEdgeTypes, setHiddenEdgeTypes] = useState<Set<string>>(new Set(["DEPENDS_ON"]))

  const toggleLevel = (lvl: HGLevel) => setHiddenLevels(prev => { const s = new Set(prev); s.has(lvl) ? s.delete(lvl) : s.add(lvl); return s })
  const toggleNodeType = (t: string) => setHiddenNodeTypes(prev => { const s = new Set(prev); s.has(t) ? s.delete(t) : s.add(t); return s })
  const toggleEdgeType = (t: string) => setHiddenEdgeTypes(prev => { const s = new Set(prev); s.has(t) ? s.delete(t) : s.add(t); return s })

  const [selectedLink, setSelectedLink] = useState<{ link: any; x: number; y: number } | null>(null)

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
    const hierResults = flatNodes.filter(n => n.name.toLowerCase().includes(q) || n.pathKey.toLowerCase().includes(q))
    const seen = new Set(hierResults.map(n => n.pathKey))
    const extraResults: HGNode[] = extraNodes
      .filter(n => {
        const name = (n.name ?? n.label ?? n.id ?? "").toLowerCase()
        const id = (n.id ?? "").toLowerCase()
        return !seen.has(n.id) && (name.includes(q) || id.includes(q))
      })
      .map(n => ({
        id: `extra:${n.id}`,
        name: n.name ?? n.label ?? n.id,
        level: "app" as HGLevel,
        nodeType: n.type,
        appCount: 0,
        pathKey: n.id,
      }))
    return [...hierResults, ...extraResults].slice(0, 25)
  }, [flatNodes, extraNodes, appSearch])

  const jumpToNode = useCallback((node: HGNode) => {
    setAppSearch("")
    setSearchOpen(false)
    const wasSyncing = isSyncingRef.current
    isSyncingRef.current = false
    if (node.level === "app") {
      setSelectedApp(node)
      setViewMode("connections")
      setConnTab("props")
      setConnData(null)
    } else {
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
    if (!wasSyncing) onNodeSelect?.(node.id)
  }, [hierarchy, onNodeSelect])

  // Sync selection when syncKey changes (cross-view sync or URL param)
  const prevSyncKeyRef = useRef<string | null | undefined>(undefined)
  useEffect(() => {
    // Resolve the effective key: prefer syncKey, fall back to initialNodeId (raw id)
    const effectiveKey = syncKey ?? (initialNodeId ? `__raw:${initialNodeId}` : null)
    if (effectiveKey === prevSyncKeyRef.current) return
    if (effectiveKey === undefined) return
    prevSyncKeyRef.current = effectiveKey
    if (!effectiveKey || flatNodes.length === 0) return
    isSyncingRef.current = true
    if (effectiveKey.startsWith('__raw:')) {
      // Legacy initialNodeId path: raw node id, search by pathKey
      const rawId = effectiveKey.slice(6)
      const hierNode = flatNodes.find(n => n.level === "app" && n.pathKey === rawId)
      if (hierNode) { jumpToNode(hierNode); return }
      const extra = extraNodes.find((n: any) => n.id === rawId)
      if (extra) { jumpToNode({ id: `extra:${extra.id}`, name: extra.name ?? extra.id, level: "app", nodeType: extra.type, appCount: 0, pathKey: extra.id }); return }
    } else if (effectiveKey.startsWith('extra:')) {
      const rawId = effectiveKey.slice(6)
      const extra = extraNodes.find((n: any) => n.id === rawId)
      if (extra) { jumpToNode({ id: effectiveKey, name: extra.name ?? extra.id, level: "app", nodeType: extra.type, appCount: 0, pathKey: rawId }); return }
    } else {
      // app:id, csms:key, css:k/k, csci:k/k/k, csc:k/k/k/k
      const hierNode = flatNodes.find(n => n.id === effectiveKey)
      if (hierNode) { jumpToNode(hierNode); return }
    }
    isSyncingRef.current = false
  }, [syncKey, initialNodeId, flatNodes, extraNodes, jumpToNode])

  const graphData = useMemo(() => buildDrillData(hierarchy, drillNode), [hierarchy, drillNode])

  const filteredGraphData = useMemo(() => {
    if (hiddenLevels.size === 0) return graphData
    const hiddenIds = new Set(graphData.nodes.filter(n => hiddenLevels.has(n.level)).map(n => n.id))
    return {
      nodes: graphData.nodes.filter(n => !hiddenLevels.has(n.level)),
      links: graphData.links.filter(l => !hiddenIds.has((l.source as any)?.id ?? l.source) && !hiddenIds.has((l.target as any)?.id ?? l.target)),
    }
  }, [graphData, hiddenLevels])

  const connGraphData = useMemo(() => {
    if (!connData) return { nodes: [], links: [] }
    let links = [...connData.links]
    if (hiddenEdgeTypes.size > 0) links = links.filter(l => !hiddenEdgeTypes.has(l.type))
    // Keep only nodes that are still referenced by remaining links or are the selected app
    const referencedIds = new Set<string>([
      ...(selectedApp ? [selectedApp.pathKey] : []),
      ...links.flatMap(l => [l.source?.id ?? l.source, l.target?.id ?? l.target]),
    ])
    let nodes = connData.nodes.filter(n => referencedIds.has(n.id))
    if (hiddenNodeTypes.size > 0) {
      const removedIds = new Set(nodes.filter(n => hiddenNodeTypes.has(n.type) && n.id !== selectedApp?.pathKey).map(n => n.id))
      nodes = nodes.filter(n => !removedIds.has(n.id))
      links = links.filter(l => !removedIds.has(l.source?.id ?? l.source) && !removedIds.has(l.target?.id ?? l.target))
    }
    return { nodes, links }
  }, [connData, selectedApp, hiddenEdgeTypes, hiddenNodeTypes])

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

  // Pre-compute target (x, y) for hierarchical connections layout (2D only)
  const connTargetPositions = useMemo<Map<string, { x: number; y: number }>>(() => {
    const map = new Map<string, { x: number; y: number }>()
    if (connGraphData.nodes.length === 0) return map
    const W = dims.width || 800
    const H = dims.height || 600
    const margin = W * 0.20
    const centerType = connGraphData.nodes.find((n: any) => n.id === selectedApp?.pathKey)?.type ?? "Application"
    const centerLayer = CONN_TYPE_LAYER[centerType] ?? 2
    // Bucket non-center nodes per layer, sorted deterministically
    const buckets = new Map<number, string[]>()
    for (const n of connGraphData.nodes as any[]) {
      if (n.id === selectedApp?.pathKey) continue
      const layer = CONN_TYPE_LAYER[n.type] ?? 2
      if (!buckets.has(layer)) buckets.set(layer, [])
      buckets.get(layer)!.push(n.id)
    }
    buckets.forEach(arr => arr.sort())
    // Assign absolute pixel positions
    for (const n of connGraphData.nodes as any[]) {
      const isCenter = n.id === selectedApp?.pathKey
      const layer = CONN_TYPE_LAYER[n.type] ?? 2
      const fy = CONN_LAYER_Y_FRACS[layer] * H
      if (isCenter) {
        map.set(n.id, { x: W / 2, y: fy })
        continue
      }
      const bucket = buckets.get(layer) ?? []
      const idx = bucket.indexOf(n.id)
      const count = bucket.length
      let fx: number
      if (count === 1 && layer !== centerLayer) {
        fx = W / 2
      } else if (count === 1 && layer === centerLayer) {
        fx = W / 2 + W * 0.22
      } else {
        fx = margin + (idx / (count - 1)) * (W - 2 * margin)
        if (layer === centerLayer) {
          const gap = W * 0.15
          if (Math.abs(fx - W / 2) < gap) fx = fx < W / 2 ? W / 2 - gap : W / 2 + gap
        }
      }
      map.set(n.id, { x: fx, y: fy })
    }
    return map
  }, [connGraphData, selectedApp, dims])

  // Which layers are actually populated (for swimlane rendering)
  const connPopulatedLayers = useMemo(() => {
    const layers = new Set<number>()
    for (const n of connGraphData.nodes as any[]) {
      layers.add(CONN_TYPE_LAYER[n.type] ?? 2)
    }
    return layers
  }, [connGraphData])

  // Forces — hierarchy
  useEffect(() => {
    if (viewMode !== "hierarchy") return
    const fg = fgRef.current
    if (!fg) return
    fg.d3Force("x", null)
    fg.d3Force("y", null)
    fg.d3Force("charge")?.strength(-600).distanceMax(800)
    fg.d3Force("link")?.distance(100)
    fg.d3Force("collide", forceCollide((node: any) => {
      const n = node as HGNode
      const base = NODE_SIZES[n.level] ?? 6
      const labelPad = { csms: 50, css: 40, csci: 35, csc: 30, app: 22 }[n.level as HGLevel] ?? 30
      return base * (n.id === drillNode?.id ? 1.5 : 1) + labelPad
    }).strength(1).iterations(4))
    fg.d3ReheatSimulation()
  }, [graphData, viewMode, drillNode])

  // Forces — connections 3D only (2D is handled by ConnFlowGraph)
  useEffect(() => {
    if (viewMode !== "connections" || !is3D) return
    const fg = fgRef.current
    if (!fg) return
    fg.d3Force("x", null)
    fg.d3Force("y", null)
    fg.d3Force("charge")?.strength(-800).distanceMax(800)
    fg.d3Force("link")?.distance(150)
    fg.d3Force("collide", forceCollide((node: any) => {
      return (node.id === selectedApp?.pathKey ? 10 : 4) + 28
    }).strength(1).iterations(4))
    fg.d3ReheatSimulation()
  }, [connGraphData, viewMode, selectedApp, is3D])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    let rafId: number | null = null
    const ro = new ResizeObserver(() => {
      if (rafId) cancelAnimationFrame(rafId)
      rafId = requestAnimationFrame(() => {
        setDims({ width: el.clientWidth, height: el.clientHeight })
        rafId = null
      })
    })
    ro.observe(el)
    setDims({ width: el.clientWidth, height: el.clientHeight })
    return () => { ro.disconnect(); if (rafId) cancelAnimationFrame(rafId) }
  }, [])

  // Fetch connections whenever selected app or depth changes
  // Two parallel calls: structural (fetch_structural=true) + derived DEPENDS_ON (fetch_structural=false)
  useEffect(() => {
    if (!selectedApp) { setConnData(null); setConnError(null); return }
    let cancelled = false
    setConnLoading(true)
    setConnError(null)
    setConnData(null)
    Promise.all([
      apiClient.getNodeConnectionsWithDepth(selectedApp.pathKey, true, connDepth),
      apiClient.getNodeConnectionsWithDepth(selectedApp.pathKey, false, connDepth),
    ]).then(([structural, derived]) => {
      if (cancelled) return
      // Merge nodes (deduplicate by id)
      const nodeMap = new Map<string, any>()
      for (const n of [...structural.nodes, ...derived.nodes]) nodeMap.set(n.id, n)
      // Merge links (structural first, then DEPENDS_ON from derived)
      const linkKey = (l: any) => `${l.source?.id ?? l.source}→${l.target?.id ?? l.target}→${l.type}`
      const linkMap = new Map<string, any>()
      for (const l of structural.links) linkMap.set(linkKey(l), l)
      for (const l of derived.links) { const k = linkKey(l); if (!linkMap.has(k)) linkMap.set(k, l) }
      setConnData({ nodes: Array.from(nodeMap.values()), links: Array.from(linkMap.values()) })
    })
      .catch(e => { if (!cancelled) setConnError(e instanceof Error ? e.message : String(e)) })
      .finally(() => { if (!cancelled) setConnLoading(false) })
    return () => { cancelled = true }
  }, [selectedApp, connDepth])

  const clearSelection = useCallback(() => {
    setSelectedApp(null)
    setViewMode("hierarchy")
    setConnData(null)
    setConnError(null)
    setSelectedLink(null)
  }, [])

  // ── 3D node objects with in-scene labels + selection ring ─────────────────
  const hierNodeThreeObj = useCallback((node: any) => {
    const n = node as HGNode
    const SpriteText = (window as any).__SpriteText
    const THREE = (window as any).__THREE
    if (!SpriteText || !THREE) return undefined

    const isParent = drillNode !== null && n.id === drillNode.id
    const isSelectedApp = selectedApp?.id === n.id
    const r = isParent ? NODE_SIZES[n.level] * 1.5 : NODE_SIZES[n.level]

    const group = new THREE.Group()

    const label = n.name.length > 24 ? n.name.slice(0, 22) + '\u2026' : n.name
    const sprite = new SpriteText(label)
    sprite.color = isDark ? '#e5e7eb' : '#374151'
    sprite.textHeight = Math.max(3, r * 0.85)
    sprite.position.set(0, r + sprite.textHeight + 2, 0)
    group.add(sprite)

    if (isParent || isSelectedApp) {
      const hexColor = isParent ? '#ffffff' : '#fbbf24'
      const tc = new THREE.Color(hexColor)
      const rc = Math.round(tc.r * 255), gc = Math.round(tc.g * 255), bc = Math.round(tc.b * 255)
      const cvs = document.createElement('canvas'); cvs.width = 128; cvs.height = 128
      const ctx2d = cvs.getContext('2d')!
      const grd = ctx2d.createRadialGradient(64, 64, r * 0.5, 64, 64, 64)
      grd.addColorStop(0,   `rgba(${rc},${gc},${bc},0.55)`)
      grd.addColorStop(0.4, `rgba(${rc},${gc},${bc},0.25)`)
      grd.addColorStop(1,   `rgba(${rc},${gc},${bc},0)`)
      ctx2d.fillStyle = grd; ctx2d.fillRect(0, 0, 128, 128)
      const glowSprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cvs), transparent: true, depthWrite: false }))
      const sz = (r + 2) * 5
      glowSprite.scale.set(sz, sz, 1)
      group.add(glowSprite)
    }

    return group
  }, [threeReady, drillNode, selectedApp, isDark])

  const connNodeThreeObj = useCallback((node: any) => {
    const n = node as any
    const SpriteText = (window as any).__SpriteText
    const THREE = (window as any).__THREE
    if (!SpriteText || !THREE) return undefined

    const isCenter = n.id === selectedApp?.pathKey
    const r = isCenter ? 10 : 4
    const color = nodeTypeColor(n.type, isDark)

    const group = new THREE.Group()

    const raw = n.label ?? n.id ?? '?'
    const labelText = raw.length > 26 ? raw.slice(0, 24) + '\u2026' : raw
    const sprite = new SpriteText(labelText)
    sprite.color = isDark ? '#e5e7eb' : '#374151'
    sprite.textHeight = Math.max(3, r * 0.8)
    sprite.position.set(0, r + sprite.textHeight + 2, 0)
    group.add(sprite)

    if (isCenter) {
      const tc = new THREE.Color(color)
      const rc = Math.round(tc.r * 255), gc = Math.round(tc.g * 255), bc = Math.round(tc.b * 255)
      const cvs = document.createElement('canvas'); cvs.width = 128; cvs.height = 128
      const ctx2d = cvs.getContext('2d')!
      const grd = ctx2d.createRadialGradient(64, 64, r * 0.5, 64, 64, 64)
      grd.addColorStop(0,   `rgba(${rc},${gc},${bc},0.6)`)
      grd.addColorStop(0.4, `rgba(${rc},${gc},${bc},0.28)`)
      grd.addColorStop(1,   `rgba(${rc},${gc},${bc},0)`)
      ctx2d.fillStyle = grd; ctx2d.fillRect(0, 0, 128, 128)
      const glowSprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(cvs), transparent: true, depthWrite: false }))
      const sz = (r + 2) * 4.5
      glowSprite.scale.set(sz, sz, 1)
      group.add(glowSprite)
    }

    return group
  }, [threeReady, selectedApp, isDark])

  // Click handler — hierarchy mode
  const drillInto = useCallback((node: object) => {
    const n = node as HGNode
    if (drillNode && n.id === drillNode.id) return
    if (n.level === "app") {
      if (selectedApp?.id === n.id) { clearSelection(); return }
      setSelectedApp(n)
      setViewMode("connections")
      setConnTab("props")
      onNodeSelect?.(n.id)
      return
    }
    clearSelection()
    const clean: HGNode = { id: n.id, name: n.name, level: n.level, appCount: n.appCount, pathKey: n.pathKey }
    setDrillStack(prev => drillNode ? [...prev, drillNode] : prev)
    setDrillNode(clean)
    onNodeSelect?.(n.id)
  }, [drillNode, selectedApp, clearSelection, onNodeSelect])

  // Click handler — connections mode (re-center on clicked node)
  const drillIntoConn = useCallback((node: object) => {
    const n = node as any
    if (n.id === selectedApp?.pathKey) return
    setSelectedApp({ id: `app:${n.id}`, name: n.label ?? n.id, level: "app", appCount: 1, pathKey: n.id })
    setConnTab("props")
    setConnData(null)
    const key = n.type === 'Application' ? `app:${n.id}` : `extra:${n.id}`
    onNodeSelect?.(key)
  }, [selectedApp, onNodeSelect])

  const drillTo = useCallback((idx: number) => {
    clearSelection()
    if (idx < 0) { setDrillNode(null); setDrillStack([]); return }
    const target = drillStack[idx]
    setDrillNode(target)
    setDrillStack(prev => prev.slice(0, idx))
    onNodeSelect?.(target.id)
  }, [drillStack, clearSelection, onNodeSelect])

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

  const connLinkColor = useCallback((link: any) => {
    if (selectedLink && link === selectedLink.link) return "#f59e0b"
    return linkTypeColor(link.type, isDark)
  }, [selectedLink, isDark])

  // Map edge weights → pixel widths, clamped to [MIN_LINK_W, MAX_LINK_W]
  const LINK_W_MIN = 0.8
  const LINK_W_MAX = 5.0
  const linkWeightScale = useMemo(() => {
    const weights = connGraphData.links.map((l: any) => Number(l.weight ?? 1)).filter(w => isFinite(w))
    if (weights.length === 0) return () => 1.5
    const lo = Math.min(...weights)
    const hi = Math.max(...weights)
    if (lo === hi) return () => (LINK_W_MIN + LINK_W_MAX) / 2
    return (w: number) => {
      const t = (w - lo) / (hi - lo)              // 0 → 1
      return LINK_W_MIN + t * (LINK_W_MAX - LINK_W_MIN)
    }
  }, [connGraphData.links])

  const connLinkWidth = useCallback((link: any) => {
    if (selectedLink && link === selectedLink.link) return 5.0
    const w = linkWeightScale(Number(link.weight ?? 1))
    const src = link.source?.id ?? link.source
    const tgt = link.target?.id ?? link.target
    const isAdjacentToCenter = src === selectedApp?.pathKey || tgt === selectedApp?.pathKey
    // Bump adjacent-to-center edges slightly but keep weight proportions
    return isAdjacentToCenter ? Math.min(w + 0.8, LINK_W_MAX) : w
  }, [selectedApp, selectedLink, linkWeightScale])

  const handleLinkClick = useCallback((link: any, event: MouseEvent) => {
    const rect = containerRef.current?.getBoundingClientRect()
    const x = rect ? event.clientX - rect.left : event.clientX
    const y = rect ? event.clientY - rect.top : event.clientY
    setSelectedLink(prev => (prev?.link === link ? null : { link, x, y }))
  }, [])

  const hierLinkColor = isDark ? "rgba(255,255,255,0.55)" : "rgba(0,0,0,0.30)"
  const bgColor = isDark ? "#09090b" : "#ffffff"
  const breadcrumbs = [
    { label: "Root", idx: -1 },
    ...drillStack.map((n, i) => ({ label: n.name, idx: i })),
  ]

  // Full ancestor path for the currently selected app, derived from hierarchy data
  const selectedAppPath = useMemo<string[]>(() => {
    if (!selectedApp) return []
    const appId = selectedApp.pathKey
    for (const [, csms] of Object.entries(hierarchy)) {
      for (const [, css] of Object.entries(csms.css)) {
        for (const [, csci] of Object.entries(css.csci)) {
          for (const [, csc] of Object.entries(csci.csc)) {
            if (csc.apps.some((a: any) => a.id === appId)) {
              return [csms.name, css.name, csci.name, csc.name]
            }
          }
        }
      }
    }
    return []
  }, [selectedApp, hierarchy])

  const outLinks = connGraphData.links.filter((l: any) => (l.source?.id ?? l.source) === selectedApp?.pathKey)
  const inLinks  = connGraphData.links.filter((l: any) => (l.target?.id ?? l.target) === selectedApp?.pathKey)
  const nodeById = useMemo(() => { const m = new Map<string, any>(); connData?.nodes.forEach(n => m.set(n.id, n)); return m }, [connData])
  const peerLabel = (nodeId: string) => nodeById.get(nodeId)?.label ?? nodeId
  const appNode = connData?.nodes.find(n => n.id === selectedApp?.pathKey)
  const appProps = appNode?.properties ? Object.entries(appNode.properties).filter(([, v]) => v !== undefined && v !== null && v !== "") : []

  return (
    <div className="flex flex-col gap-3 h-full">
      {/* Nav bar */}
      <div className="flex items-center gap-1 text-sm flex-wrap min-h-[32px] bg-muted/50 border border-border/60 rounded-lg px-3 py-1.5">
        {viewMode === "hierarchy" ? (
          <>
            {breadcrumbs.map((bc, i) => (
              <span key={bc.idx} className="flex items-center gap-1">
                {i > 0 && <ChevronRight className="h-3.5 w-3.5 text-muted-foreground shrink-0" />}
                <button className="text-foreground/60 hover:text-foreground transition-colors hover:underline underline-offset-2 text-xs font-medium"
                  onClick={() => drillTo(bc.idx)}>{bc.label}</button>
              </span>
            ))}
            {drillNode && (
              <span className="flex items-center gap-1.5">
                <ChevronRight className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                <span className="h-2 w-2 rounded-full shrink-0" style={{ background: NODE_COLORS[drillNode.level] }} />
                <span className="font-semibold text-foreground text-sm">{drillNode.name}</span>
              </span>
            )}
            {!drillNode && <span className="ml-3 text-xs text-muted-foreground">click a node to drill in</span>}
          </>
        ) : (
          <>
            {/* Reuse the same drillStack breadcrumbs as hierarchy mode — all clickable */}
            {breadcrumbs.map((bc, i) => (
              <span key={bc.idx} className="flex items-center gap-1">
                {i > 0 && <ChevronRight className="h-3.5 w-3.5 text-muted-foreground shrink-0" />}
                <button className="text-foreground/60 hover:text-foreground transition-colors hover:underline underline-offset-2 text-xs font-medium"
                  onClick={() => drillTo(bc.idx)}>{bc.label}</button>
              </span>
            ))}
            {/* drillNode (parent level) — clicking goes back to hierarchy at that level */}
            {drillNode && (
              <span className="flex items-center gap-1">
                <ChevronRight className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                <span className="h-2 w-2 rounded-full shrink-0" style={{ background: NODE_COLORS[drillNode.level] }} />
                <button className="text-foreground/60 hover:text-foreground transition-colors hover:underline underline-offset-2 text-xs font-medium"
                  onClick={clearSelection}>{drillNode.name}</button>
              </span>
            )}
            {/* Current selected app/node — not clickable */}
            <span className="flex items-center gap-1">
              <ChevronRight className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
              <span className="h-2 w-2 rounded-full shrink-0" style={{ background: nodeTypeColor(appNode?.type ?? selectedApp?.nodeType, isDark) }} />
              <span className="text-sm font-semibold text-foreground truncate max-w-64">{selectedApp?.name}</span>
            </span>
            {connLoading && <LoadingSpinner className="h-3.5 w-3.5 ml-1 text-muted-foreground" />}
            {!connLoading && <span className="ml-2 text-xs text-muted-foreground">click a node to re-center</span>}
          </>
        )}

        {/* Search — always visible on the right */}
        <div className="relative shrink-0 ml-auto">
          <div className="relative">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground pointer-events-none" />
            <input
              ref={searchRef}
              className="h-7 pl-7 pr-7 text-xs rounded-md border border-border bg-muted/40 focus:outline-none focus:ring-1 focus:ring-ring w-48 placeholder:text-muted-foreground"
              placeholder="Search nodes…"
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
                    <span className="h-2 w-2 rounded-full shrink-0" style={{ background: node.nodeType ? nodeTypeColor(node.nodeType, isDark) : NODE_COLORS[node.level] }} />
                    <span className="font-medium truncate">{node.name}</span>
                    <span className="ml-auto flex items-center gap-1.5 shrink-0">
                      <span className="text-[10px] text-muted-foreground">{node.nodeType ?? LEVEL_LABELS[node.level]}</span>
                      {node.level !== "app" && !node.nodeType && <span className="text-[10px] text-muted-foreground">{node.appCount} apps</span>}
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
        {/* Canvas */}
        <div ref={containerRef} className="flex-1 overflow-hidden relative" style={{ background: bgColor }}>
          {/* Filter overlay — connections mode only */}
          {viewMode === "connections" && <div className="absolute top-3 right-3 z-10 flex flex-col gap-2 rounded-lg border border-border/50 px-3.5 py-3 text-sm"
            style={{ background: isDark ? "rgba(15,15,20,0.75)" : "rgba(255,255,255,0.80)", backdropFilter: "blur(8px)", minWidth: 140 }}>
            <p className="font-semibold text-muted-foreground uppercase tracking-wide text-[10px]">Filter</p>
            {viewMode !== "hierarchy" && (
              <>
                <p className="font-semibold text-muted-foreground uppercase tracking-wide text-[10px] mt-0.5">Nodes</p>
                {Array.from(new Set((connData?.nodes as any[] ?? []).map(n => n.type).filter(Boolean))).map(t => (
                  <button key={t} onClick={() => toggleNodeType(t as string)}
                    className={cn("flex items-center gap-2.5 w-full text-left transition-opacity py-0.5", hiddenNodeTypes.has(t as string) ? "opacity-30" : "opacity-100")}>
                    <svg width="12" height="12" viewBox="0 0 10 10" className="shrink-0">
                      {t === "Node"    && <rect x="1" y="1" width="8" height="8" fill={nodeTypeColor(t, isDark)} />}
                      {t === "Topic"   && <polygon points="5,1 9,5 5,9 1,5" fill={nodeTypeColor(t, isDark)} />}
                      {t === "Library" && <polygon points="5,1 9,8.5 1,8.5" fill={nodeTypeColor(t, isDark)} />}
                      {t === "Broker"  && <polygon points="5,0.5 8.3,2.5 8.3,7.5 5,9.5 1.7,7.5 1.7,2.5" fill={nodeTypeColor(t, isDark)} />}
                      {!["Node","Topic","Library","Broker"].includes(t as string) && <circle cx="5" cy="5" r="4" fill={nodeTypeColor(t as string, isDark)} />}
                    </svg>
                    <span className={cn("truncate text-xs", hiddenNodeTypes.has(t as string) ? "line-through" : "")}>{t as string}</span>
                  </button>
                ))}
                <p className="font-semibold text-muted-foreground uppercase tracking-wide text-[10px] mt-1">Edges</p>
                {Array.from(new Set((connData?.links as any[] ?? []).map(l => l.type).filter(Boolean))).map(t => (
                  <button key={t} onClick={() => toggleEdgeType(t as string)}
                    className={cn("flex items-center gap-2.5 w-full text-left transition-opacity py-0.5", hiddenEdgeTypes.has(t as string) ? "opacity-30" : "opacity-100")}>
                    {t === "DEPENDS_ON" ? (
                      <svg width="18" height="6" viewBox="0 0 16 4" className="shrink-0">
                        <line x1="0" y1="2" x2="16" y2="2" stroke={linkTypeColor(t as string, isDark)} strokeWidth="1.5" strokeDasharray="4 2" />
                      </svg>
                    ) : (
                      <span className="h-0.5 w-4 shrink-0 rounded-full" style={{ background: linkTypeColor(t as string, isDark) }} />
                    )}
                    <span className={cn("truncate text-xs", hiddenEdgeTypes.has(t as string) ? "line-through" : "")}>{t as string}</span>
                  </button>
                ))}
              </>
            )}
            <div className="border-t border-border/40 mt-1 pt-1.5 space-y-0.5 text-xs text-muted-foreground">
              {connGraphData ? (
                <><p>{connGraphData.nodes.length} nodes</p><p>{connGraphData.links.length} edges</p></>
              ) : null}
            </div>
          </div>}
          {viewMode === "hierarchy" ? (
            <HierFlowGraph
              graphData={filteredGraphData}
              dims={dims}
              isDark={isDark}
              selectedNodeId={drillNode?.id ?? null}
              onNodeClick={drillInto}
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
              {is3D ? (
                <ForceGraph3D
                  graphData={connGraphData as any}
                  width={dims.width}
                  height={dims.height}
                  backgroundColor={bgColor}
                  nodeColor={(n: any) => nodeTypeColor(n.type, isDark)}
                  nodeVal={(n: any) => n.id === selectedApp?.pathKey ? 10 : 4}
                  nodeLabel=""
                  nodeThreeObject={threeReady ? connNodeThreeObj : undefined}
                  nodeThreeObjectExtend={true}
                  onNodeClick={(node: any) => { setSelectedLink(null); drillIntoConn(node) }}
                  onBackgroundClick={() => { setSelectedLink(null) }}
                  onLinkClick={handleLinkClick as any}
                  linkColor={connLinkColor}
                  linkWidth={connLinkWidth}
                  linkDirectionalArrowLength={9}
                  linkDirectionalArrowRelPos={0.85}
                  linkDirectionalArrowColor={connLinkColor}
                  cooldownTicks={200}
                  d3AlphaDecay={0.02}
                  d3VelocityDecay={0.3}
                  ref={fgRef}
                />
              ) : (
                <ConnFlowGraph
                  graphData={connGraphData}
                  positions={connTargetPositions}
                  dims={dims}
                  isDark={isDark}
                  populatedLayers={connPopulatedLayers}
                  selectedAppId={selectedApp?.pathKey ?? null}
                  selectedLink={selectedLink}
                  onNodeClick={(node: any) => { setSelectedLink(null); drillIntoConn(node) }}
                  onEdgeClick={(link: any, event: React.MouseEvent) => {
                    const rect = containerRef.current?.getBoundingClientRect()
                    const x = rect ? event.clientX - rect.left : event.clientX
                    const y = rect ? event.clientY - rect.top : event.clientY
                    setSelectedLink(prev => (prev?.link === link ? null : { link, x, y }))
                  }}
                  onBackgroundClick={() => { setSelectedLink(null) }}
                />
              )}
              {/* Edge props popover */}
              {selectedLink && (() => {
                const l = selectedLink.link
                const srcId = l.source?.id ?? l.source
                const tgtId = l.target?.id ?? l.target
                const srcLabel = nodeById.get(srcId)?.label ?? srcId
                const tgtLabel = nodeById.get(tgtId)?.label ?? tgtId
                const extraProps = l.properties ? Object.entries(l.properties as object).filter(([, v]) => v !== undefined && v !== null) : []
                const clampX = Math.min(selectedLink.x, dims.width - 224)
                const clampY = Math.min(selectedLink.y + 12, dims.height - 20)
                return (
                  <div className="absolute z-20 w-56 rounded-lg border border-border bg-popover shadow-xl text-xs"
                    style={{ left: clampX, top: clampY }}>
                    <div className="flex items-center justify-between px-3 py-2 border-b border-border">
                      <span className="font-semibold text-foreground" style={{ color: linkTypeColor(l.type, isDark) }}>{l.type ?? "Edge"}</span>
                      <button onClick={() => setSelectedLink(null)} className="text-muted-foreground hover:text-foreground"><X className="h-3 w-3" /></button>
                    </div>
                    <div className="px-3 py-2 space-y-1.5">
                      <div className="flex gap-1.5"><span className="text-muted-foreground w-10 shrink-0">From</span><span className="font-medium truncate">{srcLabel}</span></div>
                      <div className="flex gap-1.5"><span className="text-muted-foreground w-10 shrink-0">To</span><span className="font-medium truncate">{tgtLabel}</span></div>
                      {l.weight !== undefined && <div className="flex gap-1.5"><span className="text-muted-foreground w-10 shrink-0">Weight</span><span className="font-mono">{typeof l.weight === "number" ? l.weight.toFixed(4) : String(l.weight)}</span></div>}
                      {extraProps.map(([k, v]) => (
                        <div key={k} className="flex gap-1.5"><span className="text-muted-foreground w-10 shrink-0 truncate">{k}</span><span className="font-mono truncate">{String(v as any)}</span></div>
                      ))}
                    </div>
                  </div>
                )
              })()}
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
              {(["props", "connections"] as const).map(tab => (
                <button key={tab} onClick={() => setConnTab(tab)}
                  className={cn("flex-1 py-1.5 text-[11px] font-medium transition-colors",
                    connTab === tab ? "border-b-2 border-primary text-foreground" : "text-muted-foreground hover:text-foreground"
                  )}>
                  {tab === "connections" ? `Connections (${outLinks.length + inLinks.length})` : "Props"}
                </button>
              ))}
            </div>
            <div className="flex-1 overflow-y-auto">
              {connLoading && <div className="flex items-center justify-center h-24"><LoadingSpinner className="h-5 w-5" /></div>}
              {connError && !connLoading && <p className="px-4 py-3 text-xs text-destructive">{connError}</p>}
              {!connLoading && !connError && connTab === "connections" && (
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-background z-10">
                    <tr className="border-b border-border">
                      {(["node", "type", "dir"] as const).map(col => (
                        <th key={col}
                          onClick={() => setConnSort(s => ({ col, asc: s.col === col ? !s.asc : true }))}
                          className={cn(
                            "py-2 text-[10px] font-semibold uppercase tracking-wide cursor-pointer select-none transition-colors hover:text-foreground",
                            col === "dir" ? "text-right px-3" : "text-left px-3",
                            connSort.col === col ? "text-foreground" : "text-muted-foreground"
                          )}>
                          {col === "node" ? "Node" : col === "type" ? "Type" : "Dir"}
                          <span className={connSort.col === col ? "" : "opacity-30"}>{connSort.col === col ? (connSort.asc ? " ↑" : " ↓") : " ↑"}</span>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      ...outLinks.map(l => ({ link: l, dir: "out" as const })),
                      ...inLinks.map(l => ({ link: l, dir: "in" as const })),
                    ].sort((a, b) => {
                      let cmp = 0
                      if (connSort.col === "node") {
                        const aId = a.dir === "out" ? (a.link.target?.id ?? a.link.target) : (a.link.source?.id ?? a.link.source)
                        const bId = b.dir === "out" ? (b.link.target?.id ?? b.link.target) : (b.link.source?.id ?? b.link.source)
                        cmp = peerLabel(aId).localeCompare(peerLabel(bId))
                      } else if (connSort.col === "type") {
                        cmp = (a.link.type ?? "").localeCompare(b.link.type ?? "")
                      } else {
                        cmp = a.dir.localeCompare(b.dir)
                      }
                      return connSort.asc ? cmp : -cmp
                    }).map(({ link, dir }, i) => {
                      const peerId = dir === "out" ? (link.target?.id ?? link.target) : (link.source?.id ?? link.source)
                      return (
                        <tr key={i} className="border-b border-border/50 hover:bg-muted/30 transition-colors">
                          <td className="px-3 py-2 font-medium text-foreground max-w-[110px] truncate" title={peerLabel(peerId)}>{peerLabel(peerId)}</td>
                          <td className="px-3 py-2 text-muted-foreground max-w-[80px] truncate text-[10px]">{link.type ?? "—"}</td>
                          <td className="px-3 py-2 text-right">
                            <span className={cn(
                              "text-[9px] font-semibold px-1.5 py-0.5 rounded-full",
                              dir === "out"
                                ? "bg-blue-500/15 text-blue-400"
                                : "bg-orange-500/15 text-orange-400"
                            )}>{dir}</span>
                          </td>
                        </tr>
                      )
                    })}
                    {outLinks.length === 0 && inLinks.length === 0 && (
                      <tr><td colSpan={3} className="px-3 py-8 text-center text-muted-foreground text-[11px]">
                        No connections
                      </td></tr>
                    )}
                  </tbody>
                </table>
              )}
              {!connLoading && !connError && connTab === "props" && (
                <table className="w-full text-xs">
                  <tbody>
                    {appProps.length === 0 && (
                      <tr><td colSpan={2} className="px-3 py-8 text-center text-muted-foreground text-[11px]">No properties available</td></tr>
                    )}
                    {(() => {
                      const HIER_KEYS = new Set(["system_name","component_name","config_item_name","domain_name"])
                      const isCmSize     = (k: string) => /^cm_total_|^loc$/.test(k)
                      const isCmComplex  = (k: string) => /^cm_(total_|avg_|max_)wmc$|^cyclomatic_complexity$/.test(k)
                      const isCmCohesion = (k: string) => /^cm_(avg_|max_)lcom$/.test(k)
                      const isCmCoupling = (k: string) => /^cm_(avg_|max_)(cbo|rfc|fanin|fanout)$|^coupling_/.test(k)
                      const isCm         = (k: string) => /^cm_|^cyclomatic_complexity$|^coupling_|^loc$/.test(k)

                      const primitives = appProps.filter(([k, v]) => typeof v !== "object" && !HIER_KEYS.has(k) && !isCm(k))
                      const hierarchy  = appProps.filter(([k]) => HIER_KEYS.has(k))
                      const cmSize     = appProps.filter(([k]) => isCmSize(k))
                      const cmComplex  = appProps.filter(([k]) => isCmComplex(k))
                      const cmCohesion = appProps.filter(([k]) => isCmCohesion(k))
                      const cmCoupling = appProps.filter(([k]) => isCmCoupling(k))
                      const cmOther    = appProps.filter(([k, v]) => isCm(k) && !isCmSize(k) && !isCmComplex(k) && !isCmCohesion(k) && !isCmCoupling(k))

                      const PrimRow = ({ k, v, indent = false }: { k: string; v: unknown; indent?: boolean }) => (
                        <tr className="border-b border-border/50 hover:bg-muted/30 transition-colors">
                          <td className={`${indent ? "pl-6" : "px-3"} pr-3 py-2 font-medium text-foreground w-2/5`}>{k}</td>
                          <td className="px-3 py-2 font-mono text-muted-foreground break-all text-[10px]">{String(v as any)}</td>
                        </tr>
                      )
                      const GroupHeader = ({ label }: { label: string }) => (
                        <tr className="bg-muted/60 border-t border-border">
                          <td colSpan={2} className="px-3 py-1.5 text-[10px] font-semibold text-muted-foreground uppercase tracking-widest">{label}</td>
                        </tr>
                      )
                      const SubHeader = ({ label }: { label: string }) => (
                        <tr className="bg-muted/30 border-t border-border/50">
                          <td colSpan={2} className="pl-5 pr-3 py-1 text-[10px] font-medium text-muted-foreground/70 uppercase tracking-widest">{label}</td>
                        </tr>
                      )

                      return <>
                        {primitives.map(([k, v]) => <PrimRow key={k} k={k} v={v} />)}

                        {hierarchy.length > 0 && <>
                          <GroupHeader label="System Hierarchy" />
                          {hierarchy.map(([k, v]) => <PrimRow key={k} k={k} v={v} indent />)}
                        </>}

                        {(cmSize.length + cmComplex.length + cmCohesion.length + cmCoupling.length + cmOther.length) > 0 && <>
                          <GroupHeader label="Code Metrics" />
                          {cmSize.length > 0 && <>
                            <SubHeader label="Size" />
                            {cmSize.map(([k, v]) => <PrimRow key={k} k={k} v={v} indent />)}
                          </>}
                          {cmComplex.length > 0 && <>
                            <SubHeader label="Complexity" />
                            {cmComplex.map(([k, v]) => <PrimRow key={k} k={k} v={v} indent />)}
                          </>}
                          {cmCohesion.length > 0 && <>
                            <SubHeader label="Cohesion" />
                            {cmCohesion.map(([k, v]) => <PrimRow key={k} k={k} v={v} indent />)}
                          </>}
                          {cmCoupling.length > 0 && <>
                            <SubHeader label="Coupling" />
                            {cmCoupling.map(([k, v]) => <PrimRow key={k} k={k} v={v} indent />)}
                          </>}
                          {cmOther.map(([k, v]) => <PrimRow key={k} k={k} v={v} indent />)}
                        </>}
                      </>
                    })()}
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
  csms: "System (CSMS)", css: "Domain (CSS)", csci: "Config Item (CSCI)", csc: "Component (CSC)", app: "App (CSU)", node: "Node", topic: "Topic",
}
const KIND_COLOR: Record<SelectedKind, string> = {
  csms: "#10b981", css: "#3b82f6", csci: "#f59e0b", csc: "#f97316", app: "#8b5cf6", node: "#ef4444", topic: "#a855f7",
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
    content = <DetailTable headers={["Domain", "Config Item Groups", "Component Groups", "Apps"]} rows={rows} />
  } else if (node.kind === "css") {
    const css = node.payload as CssGroup
    const rows = sortKeys(Object.keys(css.csci)).map((k) => {
      const csci = css.csci[k]
      const cscCount = Object.keys(csci.csc).length
      const appCount = Object.values(csci.csc).flatMap(c => c.apps).length
      return [k, cscCount, appCount]
    })
    content = <DetailTable headers={["Config Item", "Component Groups", "Apps"]} rows={rows} />
  } else if (node.kind === "csci") {
    const csci = node.payload as CsciGroup
    const rows = sortKeys(Object.keys(csci.csc)).map((k) => [k, csci.csc[k].apps.length])
    content = <DetailTable headers={["Component", "Apps"]} rows={rows} />
  } else if (node.kind === "csc") {
    const csc = node.payload as CscGroup
    const rows = csc.apps.map((a) => [a.id, a.csu ?? a.name ?? "—", a.weight])
    content = <DetailTable headers={["ID", "Name / CSU", "Weight"]} rows={rows} />
  } else {
    // app — grouped key-value table (same grouping as graph view Props panel)
    const app = node.payload as AppNode
    const entries = Object.entries(app).filter(([, v]) => v !== undefined && v !== null && v !== "")

    const HIER_KEYS = new Set(["system_name","component_name","config_item_name","domain_name"])
    const isCmSize     = (k: string) => /^cm_total_|^loc$/.test(k)
    const isCmComplex  = (k: string) => /^cm_(total_|avg_|max_)wmc$|^cyclomatic_complexity$/.test(k)
    const isCmCohesion = (k: string) => /^cm_(avg_|max_)lcom$/.test(k)
    const isCmCoupling = (k: string) => /^cm_(avg_|max_)(cbo|rfc|fanin|fanout)$|^coupling_/.test(k)
    const isCm         = (k: string) => /^cm_|^cyclomatic_complexity$|^coupling_|^loc$/.test(k)

    const primitives = entries.filter(([k, v]) => typeof v !== "object" && !HIER_KEYS.has(k) && !isCm(k))
    const hierarchyEntries = entries.filter(([k]) => HIER_KEYS.has(k))
    const cmSize     = entries.filter(([k]) => isCmSize(k))
    const cmComplex  = entries.filter(([k]) => isCmComplex(k))
    const cmCohesion = entries.filter(([k]) => isCmCohesion(k))
    const cmCoupling = entries.filter(([k]) => isCmCoupling(k))
    const cmOther    = entries.filter(([k, v]) => isCm(k) && !isCmSize(k) && !isCmComplex(k) && !isCmCohesion(k) && !isCmCoupling(k))

    const PrimRow = ({ k, v, indent = false }: { k: string; v: unknown; indent?: boolean }) => (
      <tr className="border-b border-border/50 hover:bg-muted/30 transition-colors">
        <td className={`${indent ? "pl-8" : "px-4"} pr-4 py-2.5 font-medium text-foreground w-2/5`}>{k}</td>
        <td className="px-4 py-2.5 font-mono text-muted-foreground break-all text-xs">{String(v as any)}</td>
      </tr>
    )
    const GroupHeader = ({ label }: { label: string }) => (
      <tr className="bg-muted/60 border-t border-border">
        <td colSpan={2} className="px-4 py-1.5 text-[10px] font-semibold text-muted-foreground uppercase tracking-widest">{label}</td>
      </tr>
    )
    const SubHeader = ({ label }: { label: string }) => (
      <tr className="bg-muted/30 border-t border-border/50">
        <td colSpan={2} className="pl-7 pr-4 py-1 text-[10px] font-medium text-muted-foreground/70 uppercase tracking-widest">{label}</td>
      </tr>
    )

    content = (
      <table className="w-full text-sm">
        <thead className="sticky top-0 bg-background z-10">
          <tr className="border-b border-border">
            <th className="text-left px-4 py-2.5 text-xs font-semibold text-muted-foreground uppercase tracking-wide w-2/5">Property</th>
            <th className="text-left px-4 py-2.5 text-xs font-semibold text-muted-foreground uppercase tracking-wide">Value</th>
          </tr>
        </thead>
        <tbody>
          {primitives.map(([k, v]) => <PrimRow key={k} k={k} v={v} />)}

          {hierarchyEntries.length > 0 && <>
            <GroupHeader label="System Hierarchy" />
            {hierarchyEntries.map(([k, v]) => <PrimRow key={k} k={k} v={v} indent />)}
          </>}

          {(cmSize.length + cmComplex.length + cmCohesion.length + cmCoupling.length + cmOther.length) > 0 && <>
            <GroupHeader label="Code Metrics" />
            {cmSize.length > 0 && <>
              <SubHeader label="Size" />
              {cmSize.map(([k, v]) => <PrimRow key={k} k={k} v={v} indent />)}
            </>}
            {cmComplex.length > 0 && <>
              <SubHeader label="Complexity" />
              {cmComplex.map(([k, v]) => <PrimRow key={k} k={k} v={v} indent />)}
            </>}
            {cmCohesion.length > 0 && <>
              <SubHeader label="Cohesion" />
              {cmCohesion.map(([k, v]) => <PrimRow key={k} k={k} v={v} indent />)}
            </>}
            {cmCoupling.length > 0 && <>
              <SubHeader label="Coupling" />
              {cmCoupling.map(([k, v]) => <PrimRow key={k} k={k} v={v} indent />)}
            </>}
            {cmOther.map(([k, v]) => <PrimRow key={k} k={k} v={v} indent />)}
          </>}

          {entries.length === 0 && (
            <tr><td colSpan={2} className="px-4 py-12 text-center text-muted-foreground text-sm">No properties</td></tr>
          )}
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
  const nameMatch = q ? name.toLowerCase().includes(q) : false
  const filtered = nameMatch ? csc.apps : csc.apps.filter(a => matches(a, q))
  if (!filtered.length && !nameMatch && q) return null
  const isOpen = openSet.has(nodeKey) || !!q
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
  const nameMatch = q ? name.toLowerCase().includes(q) : false
  const visibleCsc = sortKeys(Object.keys(csci.csc)).filter(k =>
    nameMatch || k.toLowerCase().includes(q) || csci.csc[k].apps.some(a => matches(a, q))
  )
  if (!visibleCsc.length && !nameMatch && q) return null
  const total = visibleCsc.reduce((s, k) => s + csci.csc[k].apps.length, 0)
  const isOpen = openSet.has(nodeKey) || !!q
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
  const nameMatch = q ? name.toLowerCase().includes(q) : false
  const visibleCsci = sortKeys(Object.keys(css.csci)).filter(k =>
    nameMatch || k.toLowerCase().includes(q) ||
    Object.keys(css.csci[k].csc).some(ck => ck.toLowerCase().includes(q)) ||
    Object.values(css.csci[k].csc).some(c => c.apps.some(a => matches(a, q)))
  )
  if (!visibleCsci.length && !nameMatch && q) return null
  const total = visibleCsci.reduce((s, k) =>
    s + Object.values(css.csci[k].csc).flatMap(c => c.apps).length, 0)
  const isOpen = openSet.has(nodeKey) || !!q
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
  const nameMatch = q ? name.toLowerCase().includes(q) : false
  const visibleCss = sortKeys(Object.keys(csms.css)).filter(k =>
    nameMatch || k.toLowerCase().includes(q) ||
    Object.keys(csms.css[k].csci).some(ck => ck.toLowerCase().includes(q)) ||
    Object.values(csms.css[k].csci).flatMap(ci => Object.keys(ci.csc)).some(ck => ck.toLowerCase().includes(q)) ||
    Object.values(csms.css[k].csci).flatMap(ci => Object.values(ci.csc)).some(c => c.apps.some(a => matches(a, q)))
  )
  if (!visibleCss.length && !nameMatch && q) return null
  const total = visibleCss.reduce((s, k) =>
    s + Object.values(csms.css[k].csci).flatMap(ci => Object.values(ci.csc)).flatMap(c => c.apps).length, 0)
  const isOpen = openSet.has(nodeKey) || !!q
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

function BrowserPageContent() {
  const searchParams = useSearchParams()
  const nodeId = searchParams?.get('node') ?? null
  const { status, initialLoadComplete } = useConnection()
  const isConnected = status === "connected"

  const [hierarchy, setHierarchy] = useState<Record<string, CsmsGroup>>({})
  const [totalApps, setTotalApps] = useState(0)
  const [nodesList, setNodesList] = useState<any[]>([])
  const [topicsList, setTopicsList] = useState<any[]>([])
  const [brokersList, setBrokersList] = useState<any[]>([])
  const [libsList, setLibsList] = useState<any[]>([])
  const [openSet, setOpenSet] = useState<Set<string>>(new Set())
  const [search, setSearch] = useState("")
  const [sideSearch, setSideSearch] = useState("")
  const [browseSearchOpen, setBrowseSearchOpen] = useState(false)
  const [sideInitialTab, setSideInitialTab] = useState<"system" | "nodes" | "apps" | "topics">("system")
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

  const fetchData = async (targetNodeId: string | null) => {
    setLoading(true)
    setError(null)
    try {
      const [apps, nodes, topics, brokers, libs] = await Promise.all([
        apiClient.getAllApps(),
        apiClient.getComponentsByType('Node'),
        apiClient.getComponentsByType('Topic'),
        apiClient.getComponentsByType('Broker'),
        apiClient.getComponentsByType('Library'),
      ])
      const h = buildHierarchy(apps)
      setHierarchy(h)
      setTotalApps(apps.length)
      setNodesList(nodes)
      setTopicsList(topics)
      setBrokersList(brokers)
      setLibsList(libs)

      // Auto-select node from URL ?node= param
      if (targetNodeId) {
        let found = false
        outer: for (const [csmsKey, csms] of Object.entries(h)) {
          for (const [cssKey, css] of Object.entries(csms.css)) {
            for (const [csciKey, csci] of Object.entries(css.csci)) {
              for (const [cscKey, csc] of Object.entries(csci.csc)) {
                const app = csc.apps.find((a: AppNode) => a.id === targetNodeId)
                if (app) {
                  const label = app.csu ?? app.name ?? app.id ?? "?"
                  const csmsOpenKey = `csms:${csmsKey}`
                  const cssOpenKey  = `css:${csmsKey}/${cssKey}`
                  const csciOpenKey = `csci:${csmsKey}/${cssKey}/${csciKey}`
                  const cscOpenKey  = `csc:${csmsKey}/${cssKey}/${csciKey}/${cscKey}`
                  setOpenSet(new Set([csmsOpenKey, cssOpenKey, csciOpenKey, cscOpenKey]))
                  setSelectedNode({ kind: "app", key: `app:${app.id}`, label, path: [csmsKey, cssKey, csciKey, cscKey, label], payload: app })
                  setSideInitialTab("system")
                  found = true
                  break outer
                }
              }
            }
          }
        }
        if (!found) {
          // Fall back to nodesList / topicsList / brokersList / libsList
          const allExtra: any[] = [...nodes, ...topics, ...brokers, ...libs]
          const extra = allExtra.find((n: any) => n.id === targetNodeId)
          if (extra) {
            const kind: SelectedKind = topics.some((t: any) => t.id === targetNodeId) ? "topic" : "node"
            const label = extra.name ?? extra.id ?? "?"
            setSelectedNode({ kind, key: `${kind}:${extra.id}`, label, path: [label], payload: extra })
            setSideInitialTab(kind === "topic" ? "topics" : "nodes")
          } else {
            // Unknown id — fall back to default first-CSMS selection
            const firstCsmsKey = sortKeys(Object.keys(h))[0]
            if (firstCsmsKey) {
              const firstCsms = h[firstCsmsKey]
              setOpenSet(new Set([`csms:${firstCsmsKey}`]))
              setSelectedNode({ kind: "csms", key: `csms:${firstCsmsKey}`, label: firstCsms.name, path: [firstCsms.name], payload: firstCsms })
              setSideInitialTab("system")
            }
          }
        }
      } else {
        setSideInitialTab("system")
        const firstCsmsKey = sortKeys(Object.keys(h))[0]
        if (firstCsmsKey) {
          const firstCsms = h[firstCsmsKey]
          setOpenSet(new Set([`csms:${firstCsmsKey}`]))
          setSelectedNode({
            kind: "csms",
            key: `csms:${firstCsmsKey}`,
            label: firstCsms.name,
            path: [firstCsms.name],
            payload: firstCsms,
          })
        }
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (isConnected) fetchData(nodeId)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isConnected, nodeId])

  const appsList = useMemo(() =>
    Object.values(hierarchy).flatMap(csms =>
      Object.values(csms.css).flatMap(css =>
        Object.values(css.csci).flatMap(csci =>
          Object.values(csci.csc).flatMap(csc => csc.apps)
        )
      )
    )
  , [hierarchy])

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

  const expandPath = useCallback((keys: string[]) => {
    setOpenSet(prev => {
      const next = new Set(prev)
      keys.forEach(k => next.add(k))
      return next
    })
  }, [])

  // Derive the key in the graph's internal format for the currently selected list node
  const graphSyncKey = useMemo(() => {
    if (!selectedNode) return null
    if (selectedNode.kind === 'node' || selectedNode.kind === 'topic') {
      return `extra:${(selectedNode.payload as any).id}`
    }
    return selectedNode.key  // app:id, csms:k, css:k/k, csci:k/k/k, csc:k/k/k/k — already match graph ids
  }, [selectedNode])

  // Handle selection coming from the graph view — sync it into the list view
  const handleGraphNodeSelect = useCallback((key: string) => {
    if (!key) return
    if (key.startsWith('app:') || key.startsWith('csms:') || key.startsWith('css:') || key.startsWith('csci:') || key.startsWith('csc:')) {
      const entry = flatBrowseNodes.find(e => e.key === key)
      if (!entry) return
      if (selectedNode?.key === entry.key) return  // already selected
      setSelectedNode({ kind: entry.kind, key: entry.key, label: entry.label, path: entry.path, payload: entry.payload })
      // Expand tree ancestors
      setOpenSet(prev => {
        const next = new Set(prev)
        const rawPath = key.replace(/^[^:]+:/, "").split("/")
        const [k0, k1, k2] = rawPath
        if (k0) next.add(`csms:${k0}`)
        if (k1) next.add(`css:${k0}/${k1}`)
        if (k2) next.add(`csci:${k0}/${k1}/${k2}`)
        next.add(key)
        return next
      })
    } else if (key.startsWith('extra:')) {
      const rawId = key.slice(6)
      const allExtra = [...nodesList, ...topicsList, ...brokersList, ...libsList]
      const extra = allExtra.find((n: any) => n.id === rawId)
      if (!extra) return
      const kind: SelectedKind = topicsList.some((t: any) => t.id === rawId) ? 'topic' : 'node'
      const newKey = `${kind}:${rawId}`
      if (selectedNode?.key === newKey) return  // already selected
      setSelectedNode({ kind, key: newKey, label: extra.name ?? rawId, path: [extra.name ?? rawId], payload: extra })
      setSideInitialTab(kind === 'topic' ? 'topics' : 'nodes')
    }
  }, [flatBrowseNodes, nodesList, topicsList, brokersList, libsList, selectedNode])

  if (!initialLoadComplete) {
    return <AppLayout><div className="flex items-center justify-center h-64"><LoadingSpinner /></div></AppLayout>
  }
  if (!isConnected) {
    return <AppLayout><NoConnectionInfo /></AppLayout>
  }

  const q = search.toLowerCase()
  const csmsKeys = sortKeys(Object.keys(hierarchy))

  return (
    <AppLayout
      title="Explorer"
      description="System (CSMS) → Domain (CSS) → Config Item (CSCI) → Component (CSC) → App (CSU)"
    >
      <div className="flex flex-col gap-5 h-full">
        {error && (
          <div className="rounded-md bg-destructive/10 border border-destructive/30 text-destructive px-4 py-3 text-sm shrink-0">
            {error}
          </div>
        )}

        <Tabs defaultValue="browse" className="flex flex-col flex-1 min-h-0">
          <div className="flex items-center justify-between mb-4 shrink-0">
            <TabsList>
              <TabsTrigger value="browse" className="flex items-center gap-2">
                <List className="h-4 w-4" />List
              </TabsTrigger>
              <TabsTrigger value="graph" className="flex items-center gap-2">
                <Network className="h-4 w-4" />Graph
              </TabsTrigger>
            </TabsList>
            <div className="flex items-center gap-2">
              <Badge variant="secondary" className="px-3 py-1">
                <Layers className="h-3.5 w-3.5 mr-1.5" />{csmsKeys.length} CSMS
              </Badge>
              <Badge variant="secondary" className="px-3 py-1">
                <Cpu className="h-3.5 w-3.5 mr-1.5" />{totalApps} Apps
              </Badge>
              <Button variant="outline" size="sm" onClick={() => fetchData(selectedNode?.payload?.id ?? nodeId)} disabled={loading}>
                {loading ? <LoadingSpinner className="h-4 w-4" /> : <RefreshCw className="h-4 w-4" />}
                <span className="ml-2">Refresh</span>
              </Button>
            </div>
          </div>

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

              {/* Right: Tabbed list panel */}
              <div className="w-72 flex flex-col overflow-hidden shrink-0">
                <SideListPanel
                  nodesList={nodesList}
                  appsList={appsList}
                  topicsList={topicsList}
                  hierarchy={hierarchy}
                  openSet={openSet}
                  toggle={toggle}
                  expandPath={expandPath}
                  selectedKey={selectedNode?.key ?? null}
                  onSelect={setSelectedNode}
                  search={sideSearch}
                  onSearchChange={setSideSearch}
                  loading={loading}
                  initialTab={sideInitialTab}
                />
              </div>
            </div>
          </TabsContent>

          {/* ── Graph tab ── */}
          <TabsContent value="graph" className="flex-1 min-h-0 mt-0 h-full">
            {csmsKeys.length > 0
              ? <HierarchyGraph hierarchy={hierarchy} extraNodes={[...nodesList, ...topicsList, ...brokersList, ...libsList]} syncKey={graphSyncKey} onNodeSelect={handleGraphNodeSelect} />
              : <p className="text-center text-muted-foreground py-12 text-sm">No data loaded yet.</p>
            }
          </TabsContent>
        </Tabs>
      </div>
    </AppLayout>
  )
}

export default function BrowserPage() {
  return (
    <Suspense fallback={
      <AppLayout title="Explorer" description="System (CSMS) → Domain (CSS) → Config Item (CSCI) → Component (CSC) → App (CSU)">
        <div className="flex items-center justify-center h-64"><LoadingSpinner /></div>
      </AppLayout>
    }>
      <BrowserPageContent />
    </Suspense>
  )
}

// ── Simple Virtual List ───────────────────────────────────────────────────────
// Renders only the rows visible in the scroll viewport — prevents thousands of
// DOM nodes when apps/nodes/topics lists are large.
function SimpleVirtualList({ items, renderItem, itemHeight = 34 }: {
  items: any[]
  renderItem: (item: any, index: number) => React.ReactNode
  itemHeight?: number
}) {
  const outerRef = useRef<HTMLDivElement>(null)
  const [scrollTop, setScrollTop] = useState(0)
  const [viewHeight, setViewHeight] = useState(400)
  const overscan = 8

  useEffect(() => {
    const el = outerRef.current
    if (!el) return
    setViewHeight(el.clientHeight)
    const ro = new ResizeObserver(() => setViewHeight(el.clientHeight))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const totalHeight = items.length * itemHeight
  const startIdx = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan)
  const endIdx   = Math.min(items.length, Math.ceil((scrollTop + viewHeight) / itemHeight) + overscan)

  return (
    <div ref={outerRef} style={{ overflowY: "auto", height: "100%" }}
      onScroll={e => setScrollTop((e.currentTarget).scrollTop)}>
      <div style={{ height: totalHeight, position: "relative" }}>
        <div style={{ position: "absolute", top: startIdx * itemHeight, width: "100%" }}>
          {items.slice(startIdx, endIdx).map((item, i) => renderItem(item, startIdx + i))}
        </div>
      </div>
    </div>
  )
}

// ── Side List Panel ───────────────────────────────────────────────────────────

function SideListPanel({
  nodesList, appsList, topicsList, hierarchy, openSet, toggle, expandPath, selectedKey, onSelect, search, onSearchChange, loading, initialTab,
}: {
  nodesList: any[]
  appsList: AppNode[]
  topicsList: any[]
  hierarchy: Record<string, CsmsGroup>
  openSet: Set<string>
  toggle: (k: string) => void
  expandPath: (keys: string[]) => void
  selectedKey: string | null
  onSelect: (n: SelectedNode) => void
  search: string
  onSearchChange: (v: string) => void
  loading: boolean
  initialTab?: "system" | "nodes" | "apps" | "topics"
}) {
  const [sideTab, setSideTab] = useState<"system" | "nodes" | "apps" | "topics">(initialTab ?? "system")

  // Sync if parent changes the initial tab (e.g. via URL param auto-select)
  const prevInitialTab = useRef(initialTab)
  useEffect(() => {
    if (initialTab && initialTab !== prevInitialTab.current) {
      setSideTab(initialTab)
      prevInitialTab.current = initialTab
    }
  }, [initialTab])
  const [suggestOpen, setSuggestOpen] = useState(false)
  // Defer expensive filter/tree work so the search input stays responsive
  const deferredSearch = useDeferredValue(search)
  const q = deferredSearch.toLowerCase()

  const switchTab = (tab: "system" | "nodes" | "apps" | "topics") => {
    setSideTab(tab)
    onSearchChange("")
  }

  // Flat list of all hierarchy nodes for System tab suggestions
  const flatNodes = useMemo(() => {
    type Entry = { kind: SelectedKind; key: string; label: string; path: string[]; payload: CsmsGroup | CssGroup | CsciGroup | CscGroup | AppNode; treePath: string[]; tab: "system" | "nodes" | "apps" | "topics" }
    const result: Entry[] = []
    for (const [csmsKey, csms] of Object.entries(hierarchy)) {
      const csmsPath = [csms.name]
      const csmsOpenKey = `csms:${csmsKey}`
      result.push({ kind: "csms", key: csmsOpenKey, label: csms.name, path: csmsPath, payload: csms, treePath: [csmsOpenKey], tab: "system" })
      for (const [cssKey, css] of Object.entries(csms.css)) {
        const cssPath = [...csmsPath, css.name]
        const cssOpenKey = `css:${csmsKey}/${cssKey}`
        result.push({ kind: "css", key: cssOpenKey, label: css.name, path: cssPath, payload: css, treePath: [csmsOpenKey, cssOpenKey], tab: "system" })
        for (const [csciKey, csci] of Object.entries(css.csci)) {
          const csciPath = [...cssPath, csci.name]
          const csciOpenKey = `csci:${csmsKey}/${cssKey}/${csciKey}`
          result.push({ kind: "csci", key: csciOpenKey, label: csci.name, path: csciPath, payload: csci, treePath: [csmsOpenKey, cssOpenKey, csciOpenKey], tab: "system" })
          for (const [cscKey, csc] of Object.entries(csci.csc)) {
            const cscPath = [...csciPath, csc.name]
            const cscOpenKey = `csc:${csmsKey}/${cssKey}/${csciKey}/${cscKey}`
            result.push({ kind: "csc", key: cscOpenKey, label: csc.name, path: cscPath, payload: csc, treePath: [csmsOpenKey, cssOpenKey, csciOpenKey, cscOpenKey], tab: "system" })
            for (const app of csc.apps)
              result.push({ kind: "app", key: `app:${app.id}`, label: app.csu ?? app.name ?? app.id ?? "?", path: [...cscPath, app.csu ?? app.name ?? app.id ?? "?"], payload: app, treePath: [csmsOpenKey, cssOpenKey, csciOpenKey, cscOpenKey], tab: "system" })
          }
        }
      }
    }
    // Add nodes, topics as flat entries
    for (const n of nodesList)
      result.push({ kind: "node", key: `node:${n.id}`, label: n.name ?? n.id ?? "?", path: [n.name ?? n.id ?? "?"], payload: n, treePath: [], tab: "nodes" })
    for (const t of topicsList)
      result.push({ kind: "topic", key: `topic:${t.id}`, label: t.name ?? t.id ?? "?", path: [t.name ?? t.id ?? "?"], payload: t, treePath: [], tab: "topics" })
    return result
  }, [hierarchy, nodesList, topicsList])

  const suggestions = useMemo(() => {
    if (!q) return []
    return flatNodes.filter(n => n.label.toLowerCase().includes(q) || n.key.toLowerCase().includes(q)).slice(0, 30)
  }, [flatNodes, q])

  const handleJump = (entry: { kind: SelectedKind; key: string; label: string; path: string[]; payload: any; treePath: string[]; tab: "system" | "nodes" | "apps" | "topics" }) => {
    onSelect({ kind: entry.kind, key: entry.key, label: entry.label, path: entry.path, payload: entry.payload })
    if (entry.tab === "system") expandPath(entry.treePath)
    setSideTab(entry.tab)
    onSearchChange("")
    setSuggestOpen(false)
  }

  const filteredNodes  = useMemo(() => nodesList.filter(n => !q || (n.name ?? n.id ?? "").toLowerCase().includes(q) || (n.id ?? "").toLowerCase().includes(q)), [nodesList, q])
  const filteredApps   = useMemo(() => appsList.filter(a  => !q || (a.csu ?? a.name ?? a.id ?? "").toLowerCase().includes(q) || (a.id ?? "").toLowerCase().includes(q)), [appsList, q])
  const filteredTopics = useMemo(() => topicsList.filter(t => !q || (t.name ?? t.id ?? "").toLowerCase().includes(q) || (t.id ?? "").toLowerCase().includes(q)), [topicsList, q])

  const makeRow = (item: any, kind: "node" | "app" | "topic") => {
    const label = kind === "app"
      ? ((item as AppNode).csu ?? item.name ?? item.id ?? "?")
      : (item.name ?? item.id ?? "?")
    const key = `${kind}:${item.id}`
    const isSelected = selectedKey === key
    return (
      <button
        key={key}
        className={cn(
          "w-full flex items-center gap-2 px-3 py-2 text-left text-xs transition-colors",
          isSelected ? "bg-accent text-accent-foreground font-medium" : "hover:bg-muted/50 text-foreground",
        )}
        onClick={() => onSelect({ kind, key, label, path: [label], payload: item })}
      >
        <span className="h-2 w-2 rounded-full shrink-0" style={{ background: KIND_COLOR[kind] }} />
        <span className="flex-1 truncate">{label}</span>
        {item.weight != null && (
          <span className="text-[10px] text-muted-foreground shrink-0 tabular-nums">{typeof item.weight === "number" ? item.weight.toFixed(2) : item.weight}</span>
        )}
      </button>
    )
  }

  const csmsKeys = sortKeys(Object.keys(hierarchy))

  return (
    <>
      {/* Tab bar */}
      <div className="border-b border-border shrink-0">
        <div className="flex text-xs">
          {(["system", "nodes", "apps", "topics"] as const).map(tab => {
            const count = tab === "system" ? csmsKeys.length : tab === "nodes" ? nodesList.length : tab === "apps" ? appsList.length : topicsList.length
            const label = tab === "system" ? "System" : tab === "nodes" ? "Nodes" : tab === "apps" ? "Apps" : "Topics"
            return (
              <button
                key={tab}
                className={cn(
                  "flex-1 py-2 px-1 transition-colors border-b-2 text-[11px]",
                  sideTab === tab
                    ? "border-primary text-foreground font-medium"
                    : "border-transparent text-muted-foreground hover:text-foreground",
                )}
                onClick={() => switchTab(tab)}
              >
                {label}{" "}
                <span className={cn("text-[10px]", sideTab === tab ? "text-muted-foreground" : "text-muted-foreground/60")}>{count}</span>
              </button>
            )
          })}
        </div>
        {/* Search */}
        <div className="px-2 py-2">
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
            <Input
              className="pl-8 h-7 text-xs bg-muted/30"
              placeholder="Search…"
              value={search}
              onChange={(e) => { onSearchChange(e.target.value); setSuggestOpen(true) }}
              onFocus={() => setSuggestOpen(true)}
              onBlur={() => setTimeout(() => setSuggestOpen(false), 150)}
            />
            {search && (
              <button
                className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                onClick={() => { onSearchChange(""); setSuggestOpen(false) }}
              >
                <X className="h-3 w-3" />
              </button>
            )}
            {/* Global suggestions dropdown */}
            {suggestOpen && suggestions.length > 0 && (
              <div className="absolute left-0 right-0 top-full mt-1 z-50 rounded-md border border-border bg-popover shadow-lg overflow-hidden">
                <div className="max-h-64 overflow-y-auto py-1">
                  {suggestions.map(n => (
                    <button
                      key={n.key}
                      className="w-full flex items-center gap-2 px-3 py-1.5 text-xs hover:bg-accent hover:text-accent-foreground text-left transition-colors"
                      onMouseDown={e => { e.preventDefault(); handleJump(n) }}
                    >
                      <span className="h-2 w-2 rounded-full shrink-0" style={{ background: KIND_COLOR[n.kind] }} />
                      <span className="flex-1 truncate font-medium">{n.label}</span>
                      <span className="text-[10px] text-muted-foreground shrink-0 capitalize">{n.tab}</span>
                    </button>
                  ))}
                </div>
                <div className="border-t border-border px-3 py-1 text-[10px] text-muted-foreground">
                  {suggestions.length} result{suggestions.length !== 1 ? "s" : ""}{suggestions.length === 30 ? " (first 30)" : ""}
                </div>
              </div>
            )}
            {suggestOpen && search.trim() && suggestions.length === 0 && (
              <div className="absolute left-0 right-0 top-full mt-1 z-50 rounded-md border border-border bg-popover shadow-lg px-3 py-2.5 text-xs text-muted-foreground">
                No results for &ldquo;{search.trim()}&rdquo;
              </div>
            )}
          </div>
        </div>
      </div>

      {/* List / Tree */}
      <div className={`flex-1 ${sideTab === "system" ? "overflow-y-auto" : "overflow-hidden"}`}>
        {sideTab === "system" && (
          csmsKeys.length > 0
            ? csmsKeys.map(k => (
                <CsmsTreeNode
                  key={k}
                  name={k}
                  csms={hierarchy[k]}
                  selectedKey={selectedKey}
                  onSelect={onSelect}
                  openSet={openSet}
                  toggle={toggle}
                  q=""
                />
              ))
            : !loading && <p className="text-center text-muted-foreground py-8 text-xs px-4">No data loaded. Import a graph first.</p>
        )}
        {sideTab === "nodes"  && (filteredNodes.length  > 0 ? <SimpleVirtualList items={filteredNodes}  renderItem={n => makeRow(n, "node")}  itemHeight={34} /> : <p className="text-center text-muted-foreground py-8 text-xs px-4">No nodes found.</p>)}
        {sideTab === "apps"   && (filteredApps.length   > 0 ? <SimpleVirtualList items={filteredApps}   renderItem={a => makeRow(a, "app")}   itemHeight={34} /> : <p className="text-center text-muted-foreground py-8 text-xs px-4">No apps found.</p>)}
        {sideTab === "topics" && (filteredTopics.length > 0 ? <SimpleVirtualList items={filteredTopics} renderItem={t => makeRow(t, "topic")} itemHeight={34} /> : <p className="text-center text-muted-foreground py-8 text-xs px-4">No topics found.</p>)}
      </div>
      {/* Tree legend (System tab only) */}
      {sideTab === "system" && (
        <div className="border-t border-border px-3 py-2 shrink-0 flex flex-wrap gap-x-3 gap-y-1">
          {(["csms", "css", "csci", "csc", "app"] as const).map(k => (
            <span key={k} className="flex items-center gap-1 text-[10px] text-muted-foreground">
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: NODE_COLORS[k] }} />
              {KIND_LABEL[k]}
            </span>
          ))}
        </div>
      )}
    </>
  )
}
