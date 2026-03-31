"use client"

import React, { useState, useEffect, useCallback, useMemo, useRef, Suspense } from "react"
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

function HierarchyGraph({ hierarchy, extraNodes = [], initialNodeId = null }: { hierarchy: Record<string, CsmsGroup>; extraNodes?: any[]; initialNodeId?: string | null }) {
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
  const [connTab, setConnTab] = useState<"out" | "in" | "props">("props")
  const [connDepth, setConnDepth] = useState(1)

  const [hiddenLevels, setHiddenLevels] = useState<Set<HGLevel>>(new Set())
  const [hiddenNodeTypes, setHiddenNodeTypes] = useState<Set<string>>(new Set())
  const [hiddenEdgeTypes, setHiddenEdgeTypes] = useState<Set<string>>(new Set())

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
    if (node.level === "app") {
      setSelectedApp(node)
      setViewMode("connections")
      setConnTab("props")
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

  // Auto-select node from URL ?node= param once flatNodes is populated
  const autoJumpDone = useRef(false)
  useEffect(() => {
    if (!initialNodeId || autoJumpDone.current || flatNodes.length === 0) return
    // Try hierarchy nodes first (app:${id})
    const hierNode = flatNodes.find(n => n.level === "app" && n.pathKey === initialNodeId)
    if (hierNode) {
      autoJumpDone.current = true
      jumpToNode(hierNode)
      return
    }
    // Fall back to extraNodes (topics, infra nodes, brokers, etc.)
    const extra = extraNodes.find((n: any) => n.id === initialNodeId)
    if (extra) {
      autoJumpDone.current = true
      jumpToNode({ id: `extra:${extra.id}`, name: extra.name ?? extra.id, level: "app", nodeType: extra.type, appCount: 0, pathKey: extra.id })
    }
  }, [initialNodeId, flatNodes, extraNodes, jumpToNode])

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
    let links = connData.links.filter(l => l.type !== "DEPENDS_ON")
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
      const geo = new THREE.TorusGeometry(r + 3, 0.6, 6, 24)
      const mat = new THREE.MeshBasicMaterial({
        color: isParent ? 0xffffff : 0xfbbf24,
        transparent: true,
        opacity: 0.8,
      })
      group.add(new THREE.Mesh(geo, mat))
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
      const geo = new THREE.TorusGeometry(r + 3, 0.6, 6, 24)
      const mat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(color),
        transparent: true,
        opacity: 0.85,
      })
      group.add(new THREE.Mesh(geo, mat))
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
    setConnTab("props")
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
      const color = nodeTypeColor(n.type, isDark)
      const r = isCenter ? 10 : 6

      // Glow halo for center node
      if (isCenter) {
        ctx.beginPath(); ctx.arc(n.x!, n.y!, r + 7, 0, 2 * Math.PI)
        ctx.fillStyle = color + "33"; ctx.fill()
      }

      // Draw shape by type (matches Explorer)
      ctx.fillStyle = color
      ctx.beginPath()
      switch (n.type) {
        case "Node":
          ctx.rect(n.x! - r, n.y! - r, r * 2, r * 2)
          break
        case "Topic":
          ctx.moveTo(n.x!, n.y! - r); ctx.lineTo(n.x! + r, n.y!)
          ctx.lineTo(n.x!, n.y! + r); ctx.lineTo(n.x! - r, n.y!)
          ctx.closePath()
          break
        case "Library":
          ctx.moveTo(n.x!, n.y! - r)
          ctx.lineTo(n.x! + r, n.y! + r * 0.6)
          ctx.lineTo(n.x! - r, n.y! + r * 0.6)
          ctx.closePath()
          break
        case "Broker": {
          const a = (Math.PI * 2) / 6
          for (let i = 0; i < 6; i++) {
            const x = n.x! + r * Math.cos(a * i - Math.PI / 2)
            const y = n.y! + r * Math.sin(a * i - Math.PI / 2)
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
          }
          ctx.closePath()
          break
        }
        default: // Application + unknown → circle
          ctx.arc(n.x!, n.y!, r, 0, 2 * Math.PI)
          break
      }
      ctx.fill()

      // Border
      ctx.strokeStyle = isCenter ? (isDark ? "#fff" : "#111") : (isDark ? "rgba(255,255,255,0.3)" : "rgba(30,41,59,0.35)")
      ctx.lineWidth = isCenter ? 2.5 : 1 / globalScale
      ctx.stroke()

      // Labels
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

  const outLinks = (connData?.links ?? []).filter(l => (l.source?.id ?? l.source) === selectedApp?.pathKey)
  const inLinks  = (connData?.links ?? []).filter(l => (l.target?.id ?? l.target) === selectedApp?.pathKey)
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
            <button className="flex items-center gap-1 text-foreground/60 hover:text-foreground transition-colors"
              onClick={clearSelection}>
              <ChevronRight className="h-3.5 w-3.5 rotate-180" />
              <span className="text-xs font-medium">Hierarchy</span>
            </button>
            {selectedAppPath.length > 0
              ? selectedAppPath.map(segment => (
                  <span key={segment} className="flex items-center gap-1">
                    <ChevronRight className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                    <span className="text-xs text-foreground/60 font-medium">{segment}</span>
                  </span>
                ))
              : (appNode?.type ?? selectedApp?.nodeType) && (
                  <span className="flex items-center gap-1">
                    <ChevronRight className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
                    <span className="text-xs text-foreground/60 font-medium">{appNode?.type ?? selectedApp?.nodeType}</span>
                  </span>
                )
            }
            <ChevronRight className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
            <span className="h-2 w-2 rounded-full shrink-0" style={{ background: nodeTypeColor(appNode?.type ?? selectedApp?.nodeType, isDark) }} />
            <span className="text-sm font-semibold text-foreground truncate max-w-64">{selectedApp?.name}</span>
            {connLoading && <LoadingSpinner className="h-3.5 w-3.5 ml-1 text-muted-foreground" />}
            {!connLoading && <span className="ml-2 text-xs text-muted-foreground">click a node to re-center</span>}
          </>
        )}

        {/* 3D toggle */}
        <button
          onClick={() => setIs3D(v => !v)}
          className={cn(
            "ml-auto h-7 px-2.5 text-xs rounded-md border font-medium transition-colors shrink-0",
            is3D
              ? "bg-primary text-primary-foreground border-primary"
              : "bg-muted/40 border-border text-muted-foreground hover:bg-muted hover:text-foreground"
          )}
        >
          {is3D ? "2D" : "3D"}
        </button>

        {/* Search — always visible on the right */}
        <div className="relative shrink-0">
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
        {/* Legend */}
        <div className="flex flex-col gap-3 w-36 shrink-0">
          <div className="space-y-1 text-xs">
            <p className="font-medium text-muted-foreground uppercase tracking-wide text-[10px]">Filter</p>
            {viewMode === "hierarchy" ? (
              (["csms", "css", "csci", "csc", "app"] as HGLevel[]).map(lvl => (
                <button key={lvl} onClick={() => toggleLevel(lvl)}
                  className={cn("flex items-center gap-2 w-full text-left transition-opacity", hiddenLevels.has(lvl) ? "opacity-30" : "opacity-100")}>
                  <span className="h-2.5 w-2.5 rounded-full shrink-0" style={{ background: NODE_COLORS[lvl] }} />
                  <span className={hiddenLevels.has(lvl) ? "line-through" : ""}>{LEVEL_LABELS[lvl]}</span>
                </button>
              ))
            ) : (
              <>
                <p className="font-medium text-muted-foreground uppercase tracking-wide text-[10px] mt-1">Nodes</p>
                {Array.from(new Set((connData?.nodes as any[] ?? []).map(n => n.type).filter(Boolean))).map(t => (
                  <button key={t} onClick={() => toggleNodeType(t as string)}
                    className={cn("flex items-center gap-2 w-full text-left transition-opacity", hiddenNodeTypes.has(t as string) ? "opacity-30" : "opacity-100")}>
                    <svg width="10" height="10" viewBox="0 0 10 10" className="shrink-0">
                      {t === "Node"    && <rect x="1" y="1" width="8" height="8" fill={nodeTypeColor(t, isDark)} />}
                      {t === "Topic"   && <polygon points="5,1 9,5 5,9 1,5" fill={nodeTypeColor(t, isDark)} />}
                      {t === "Library" && <polygon points="5,1 9,8.5 1,8.5" fill={nodeTypeColor(t, isDark)} />}
                      {t === "Broker"  && <polygon points="5,0.5 8.3,2.5 8.3,7.5 5,9.5 1.7,7.5 1.7,2.5" fill={nodeTypeColor(t, isDark)} />}
                      {!["Node","Topic","Library","Broker"].includes(t as string) && <circle cx="5" cy="5" r="4" fill={nodeTypeColor(t as string, isDark)} />}
                    </svg>
                    <span className={cn("truncate", hiddenNodeTypes.has(t as string) ? "line-through" : "")}>{t as string}</span>
                  </button>
                ))}
                <p className="font-medium text-muted-foreground uppercase tracking-wide text-[10px] mt-2">Edges</p>
                {Array.from(new Set((connData?.links as any[] ?? []).filter(l => l.type !== "DEPENDS_ON").map(l => l.type).filter(Boolean))).map(t => (
                  <button key={t} onClick={() => toggleEdgeType(t as string)}
                    className={cn("flex items-center gap-2 w-full text-left transition-opacity", hiddenEdgeTypes.has(t as string) ? "opacity-30" : "opacity-100")}>
                    <span className="h-0.5 w-4 shrink-0 rounded-full" style={{ background: linkTypeColor(t as string, isDark) }} />
                    <span className={cn("truncate", hiddenEdgeTypes.has(t as string) ? "line-through" : "")}>{t as string}</span>
                  </button>
                ))}
              </>
            )}
          </div>
          <div className="mt-auto space-y-0.5 text-xs text-muted-foreground">
            {viewMode === "hierarchy" ? (
              <><p>{filteredGraphData.nodes.length} nodes</p><p>{filteredGraphData.links.length} edges</p></>
            ) : connGraphData ? (
              <><p>{connGraphData.nodes.length} nodes</p><p>{connGraphData.links.length} edges</p></>
            ) : null}
          </div>
        </div>

        {/* Canvas */}
        <div ref={containerRef} className="flex-1 overflow-hidden relative" style={{ background: bgColor }}>
          {viewMode === "hierarchy" ? (
            is3D ? (
              <ForceGraph3D
                key="hierarchy-3d"
                graphData={filteredGraphData as any}
                width={dims.width}
                height={dims.height}
                backgroundColor={bgColor}
                nodeColor={(n: any) => NODE_COLORS[(n as HGNode).level]}
                nodeVal={(n: any) => {
                  const hn = n as HGNode
                  return drillNode && hn.id === drillNode.id ? NODE_SIZES[hn.level] * 1.5 : NODE_SIZES[hn.level]
                }}
                nodeLabel=""
                nodeThreeObject={threeReady ? hierNodeThreeObj : undefined}
                nodeThreeObjectExtend={true}
                onNodeClick={drillInto}
                onBackgroundClick={() => { if (selectedApp) clearSelection() }}
                linkColor={() => hierLinkColor}
                linkWidth={2}
                linkDirectionalArrowLength={8}
                linkDirectionalArrowRelPos={0.85}
                linkDirectionalArrowColor={() => hierLinkColor}
                cooldownTicks={150}
                d3AlphaDecay={0.04}
                d3VelocityDecay={0.5}
                ref={fgRef}
              />
            ) : (
              <ForceGraph2D
                key="hierarchy"
                graphData={filteredGraphData as any}
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
            )
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
                  key={`conn-3d-${selectedApp?.pathKey ?? ""}`}
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
                  onBackgroundClick={() => { setSelectedLink(null); clearSelection() }}
                  onLinkClick={handleLinkClick as any}
                  linkColor={connLinkColor}
                  linkWidth={connLinkWidth}
                  linkDirectionalArrowLength={9}
                  linkDirectionalArrowRelPos={0.85}
                  linkDirectionalArrowColor={connLinkColor}
                  cooldownTicks={150}
                  d3AlphaDecay={0.03}
                  d3VelocityDecay={0.4}
                  ref={fgRef}
                />
              ) : (
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
                  onNodeClick={(node, event) => { setSelectedLink(null); drillIntoConn(node) }}
                  onBackgroundClick={() => { setSelectedLink(null); clearSelection() }}
                  onLinkClick={handleLinkClick}
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
              {(["props", "out", "in"] as const).map(tab => (
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
              <Button variant="outline" size="sm" onClick={fetchData} disabled={loading}>
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
                  appsList={Object.values(hierarchy).flatMap(csms =>
                    Object.values(csms.css).flatMap(css =>
                      Object.values(css.csci).flatMap(csci =>
                        Object.values(csci.csc).flatMap(csc => csc.apps)
                      )
                    )
                  )}
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
              ? <HierarchyGraph hierarchy={hierarchy} extraNodes={[...nodesList, ...topicsList, ...brokersList, ...libsList]} initialNodeId={nodeId} />
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
  const q = search.toLowerCase()

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

  const filteredNodes  = nodesList.filter(n => !q || (n.name ?? n.id ?? "").toLowerCase().includes(q) || (n.id ?? "").toLowerCase().includes(q))
  const filteredApps   = appsList.filter(a  => !q || (a.csu ?? a.name ?? a.id ?? "").toLowerCase().includes(q) || (a.id ?? "").toLowerCase().includes(q))
  const filteredTopics = topicsList.filter(t => !q || (t.name ?? t.id ?? "").toLowerCase().includes(q) || (t.id ?? "").toLowerCase().includes(q))

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
      <div className="flex-1 overflow-y-auto">
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
        {sideTab === "nodes"  && (filteredNodes.length  > 0 ? filteredNodes.map(n  => makeRow(n, "node"))  : <p className="text-center text-muted-foreground py-8 text-xs px-4">No nodes found.</p>)}
        {sideTab === "apps"   && (filteredApps.length   > 0 ? filteredApps.map(a   => makeRow(a, "app"))   : <p className="text-center text-muted-foreground py-8 text-xs px-4">No apps found.</p>)}
        {sideTab === "topics" && (filteredTopics.length > 0 ? filteredTopics.map(t => makeRow(t, "topic")) : <p className="text-center text-muted-foreground py-8 text-xs px-4">No topics found.</p>)}
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
