"use client"

import React, { useState, useEffect, useCallback, useMemo } from "react"
import { AppLayout } from "@/components/layout/app-layout"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Activity,
  BarChart2,
  Network,
  Server,
  Play,
  Save,
  Trash2,
  AlertCircle,
  X,
  Zap,
  LayoutList,
  MessageSquare,
  Search,
  ArrowUpDown,
  Pencil,
  ShieldAlert,
  Layers,
  Check,
  Cpu,
  Box,
  Database,
  Cloud,
  HardDrive,
  Router,
} from "lucide-react"
import { useConnection } from "@/lib/stores/connection-store"
import { trafficClient, type TopicInfo, type AppInfo, type TopicParams, type TrafficSimulationResult } from "@/lib/api/traffic-client"
import { simulationClient, type FailureResult } from "@/lib/api/simulation-client"
import { apiClient } from "@/lib/api/client"
import { TermTooltip } from "@/components/ui/term-tooltip"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import ReactECharts from "echarts-for-react"

// ============================================================================
// Types
// ============================================================================

interface SavedConfig {
  id: string
  name: string
  topic_ids: string[]
  app_ids?: string[]
  frequency_hz: number
  duration_sec: number
  message_size_bytes: number
  topic_params?: Record<string, TopicParams>
  created_at: string
}

// ============================================================================
// Helpers
// ============================================================================

function getBandwidthColor(bps: number, maxBps: number): string {
  if (maxBps === 0) return "text-gray-500"
  const ratio = bps / maxBps
  if (ratio > 0.8) return "text-red-500"
  if (ratio > 0.5) return "text-orange-500"
  if (ratio > 0.2) return "text-yellow-500"
  return "text-green-500"
}

const LS_CONFIGS_KEY = "traffic_sim_configs"

// ============================================================================
// Main Page
// ============================================================================

export default function TrafficSimulatorPage() {
  const { status, initialLoadComplete } = useConnection()

  // ---- Topics ----
  const [topics, setTopics] = useState<TopicInfo[]>([])
  const [topicsLoading, setTopicsLoading] = useState(false)
  const [topicsError, setTopicsError] = useState<string | null>(null)

  // ---- Apps ----
  const [apps, setApps] = useState<AppInfo[]>([])
  const [appsLoading, setAppsLoading] = useState(false)
  const [appsError, setAppsError] = useState<string | null>(null)
  const [appSearch, setAppSearch] = useState("")
  const [appSort, setAppSort] = useState<"name" | "weight" | "topics">("name")
  const [selectionTab, setSelectionTab] = useState<"topics" | "apps" | "roles">("roles")

  // ---- Selection ----
  const [selectedTopicIds, setSelectedTopicIds] = useState<string[]>([])
  const [selectedRoleKeys, setSelectedRoleKeys] = useState<Set<string>>(new Set())
  const [topicParams, setTopicParams] = useState<Record<string, TopicParams>>({})
  const [editingFreqId, setEditingFreqId] = useState<string | null>(null)
  const [topicSearch, setTopicSearch] = useState("")
  const [roleSearch, setRoleSearch] = useState("")
  const [topicQosFilter, setTopicQosFilter] = useState<string>("all")
  const [topicSort, setTopicSort] = useState<"name" | "pub" | "sub">("name")

  // ---- Parameters ----
  const [durationSec, setDurationSec] = useState<number>(60)
  const [messageSizeBytes, setMessageSizeBytes] = useState<number>(1024)

  // ---- Simulation ----
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<TrafficSimulationResult | null>(null)
  const [networkCapacityMbps, setNetworkCapacityMbps] = useState<number>(1000)

  // ---- Saved configs ----
  const [savedConfigs, setSavedConfigs] = useState<SavedConfig[]>([])
  const [newConfigName, setNewConfigName] = useState("")

  // ---- Simulation Mode ----
  const [simMode, setSimMode] = useState<"traffic" | "failure">("traffic")
  const [components, setComponents] = useState<Array<{ id: string; name: string; type: string }>>([])
  const [componentsLoading, setComponentsLoading] = useState(false)

  // ---- Failure Simulation ----
  const [failureTargetIds, setFailureTargetIds] = useState<string[]>([])
  const [failureSearch, setFailureSearch] = useState("")
  const [failureTypeTab, setFailureTypeTab] = useState<string>("")
  const [failureLayer, setFailureLayer] = useState<string>("system")
  const [failureCascadeProb, setFailureCascadeProb] = useState<number>(1.0)
  const [failureResults, setFailureResults] = useState<FailureResult[] | null>(null)
  const [failureLoading, setFailureLoading] = useState(false)
  const [failureError, setFailureError] = useState<string | null>(null)
  const [cascadedSearch, setCascadedSearch] = useState<string>("")

  // Group components by type for better UX
  const availableTypes = useMemo(() => {
    const types = new Set(components.map(c => c.type || "Unknown"))
    return Array.from(types).sort()
  }, [components])

  const groupedComponents = useMemo(() => {
    const groups: Record<string, typeof components> = {}
    for (const comp of components) {
      const type = comp.type || "Unknown"
      if (!groups[type]) {
        groups[type] = []
      }
      groups[type].push(comp)
    }
    return groups
  }, [components])

  // ------------------------------------------------------------------
  // Effects
  // ------------------------------------------------------------------

  // Load topics when connected
  useEffect(() => {
    if (status === "connected") {
      loadTopics()
      loadApps()
      loadComponents()
    } else {
      setTopics([])
      setApps([])
      setSelectedRoleKeys(new Set())
    }
  }, [status])

  // Restore saved configs from localStorage
  useEffect(() => {
    if (typeof window === "undefined") return
    const cfgs = localStorage.getItem(LS_CONFIGS_KEY)
    if (cfgs) {
      try { setSavedConfigs(JSON.parse(cfgs)) } catch { /* ignore */ }
    }
  }, [])

  // Persist saved configs
  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem(LS_CONFIGS_KEY, JSON.stringify(savedConfigs))
    }
  }, [savedConfigs])

  // ------------------------------------------------------------------
  // Actions
  // ------------------------------------------------------------------

  async function loadTopics() {
    setTopicsLoading(true)
    setTopicsError(null)
    try {
      const data = await trafficClient.listTopics()
      setTopics(data)
    } catch (err: any) {
      setTopicsError(err.message || "Failed to load topics")
    } finally {
      setTopicsLoading(false)
    }
  }

  async function loadApps() {
    setAppsLoading(true)
    setAppsError(null)
    try {
      const data = await trafficClient.listApps()
      setApps(data)
    } catch (err: any) {
      setAppsError(err.message || "Failed to load apps")
    } finally {
      setAppsLoading(false)
    }
  }

  async function loadComponents() {
    setComponentsLoading(true)
    try {
      const data = await apiClient.getGraphData()
      const comps = data.nodes.map((node: any) => ({
        id: node.id,
        name: node.label || node.id,
        type: node.type,
      }))
      setComponents(comps)
    } catch (err: any) {
      console.error("Failed to load components:", err)
    } finally {
      setComponentsLoading(false)
    }
  }

  function toggleFailureTarget(id: string) {
    setFailureTargetIds(prev => {
      if (prev.includes(id)) {
        return prev.filter(tid => tid !== id)
      }
      return [...prev, id]
    })
    setFailureResults(null)
    setFailureError(null)
  }

  async function runFailureSimulation() {
    if (failureTargetIds.length === 0) {
      setFailureError("Please select at least one target component.")
      return
    }
    setFailureLoading(true)
    setFailureError(null)
    setFailureResults(null)
    try {
      const results = await simulationClient.runFailureSimulation({
        target_ids: failureTargetIds,
        layer: failureLayer,
        cascade_probability: failureCascadeProb,
      })
      setFailureResults(results)
    } catch (err: any) {
      setFailureError(err.message || "Failure simulation failed")
    } finally {
      setFailureLoading(false)
    }
  }

  function getComponentName(id: string): string {
    const comp = components.find(c => c.id === id)
    return comp ? comp.name : id
  }

  function getComponentType(id: string): string {
    const comp = components.find(c => c.id === id)
    return comp ? comp.type : "Unknown"
  }

  function getImpactColor(impact: number): string {
    if (impact > 0.5) return "text-red-500"
    if (impact > 0.3) return "text-orange-500"
    if (impact > 0.1) return "text-yellow-500"
    return "text-green-500"
  }

  async function runSimulation() {
    if (selectedTopicIds.length === 0) {
      setError("Please select at least one topic.")
      return
    }
    setLoading(true)
    setError(null)
    try {
      const data = await trafficClient.simulate({
        topic_ids: selectedTopicIds,
        frequency_hz: 10, // Fallback default (not used when per_topic_params provided)
        duration_sec: durationSec,
        message_size_bytes: messageSizeBytes,
        per_topic_params: topicParams, // Always use per-topic frequencies
      })
      setResult(data)
    } catch (err: any) {
      setError(err.message || "Simulation failed")
    } finally {
      setLoading(false)
    }
  }

  function toggleTopic(id: string) {
    const topic = topicById[id]
    setSelectedTopicIds(prev => {
      if (prev.includes(id)) {
        // Remove topic and its params
        setTopicParams(p => {
          const next = { ...p }
          delete next[id]
          return next
        })
        return prev.filter(t => t !== id)
      }
      // Initialize topic params with topic's default frequency
      if (topic) {
        setTopicParams(p => ({
          ...p,
          [id]: {
            frequency_hz: topic.frequency ?? 10.0,
            duration_sec: durationSec,
          },
        }))
      }
      return [...prev, id]
    })
  }

  function toggleApp(app: AppInfo) {
    const appTopicIds = Array.from(new Set([...app.pub_topic_ids, ...app.sub_topic_ids]))
    const alreadyAllSelected = appTopicIds.length > 0 && appTopicIds.every(tid => selectedTopicIds.includes(tid))
    if (alreadyAllSelected) {
      setSelectedTopicIds(prev => prev.filter(tid => !appTopicIds.includes(tid)))
    } else {
      setSelectedTopicIds(prev => Array.from(new Set([...prev, ...appTopicIds])))
      // Initialize topic params for newly selected topics with their default frequencies
      setTopicParams(prev => {
        const next = { ...prev }
        for (const tid of appTopicIds) {
          if (!next[tid]) {
            const t = topicById[tid]
              next[tid] = {
                frequency_hz: t?.frequency ?? 10.0, // Fallback to 10 Hz if not available
              duration_sec: durationSec,
            }
          }
        }
        return next
      })
    }
  }

  function saveConfig() {
    const name = newConfigName.trim() || `Config ${savedConfigs.length + 1}`
    const cfg: SavedConfig = {
      id: Date.now().toString(),
      name,
      topic_ids: [...selectedTopicIds],
      frequency_hz: 0, // Deprecated - frequencies are now per-topic
      duration_sec: durationSec,
      message_size_bytes: messageSizeBytes,
      topic_params: { ...topicParams }, // Always save per-topic params
      created_at: new Date().toISOString(),
    }
    setSavedConfigs(prev => [cfg, ...prev])
    setNewConfigName("")
  }

  function loadConfig(cfg: SavedConfig) {
    setSelectedTopicIds(cfg.topic_ids)
    setDurationSec(cfg.duration_sec)
    setMessageSizeBytes(cfg.message_size_bytes)
    if (cfg.topic_params) {
      setTopicParams(cfg.topic_params)
    }
  }

  function deleteConfig(id: string) {
    setSavedConfigs(prev => prev.filter(c => c.id !== id))
  }

  // ------------------------------------------------------------------
  // Derived
  // ------------------------------------------------------------------

  const topicById = Object.fromEntries(topics.map(t => [t.id, t]))

  const filteredApps = React.useMemo(() => {
    let list = apps.filter(a =>
      a.name.toLowerCase().includes(appSearch.toLowerCase()) ||
      a.id.toLowerCase().includes(appSearch.toLowerCase())
    )
    list = [...list].sort((a, b) => {
      if (appSort === "name")   return a.name.localeCompare(b.name)
      if (appSort === "weight") return b.weight - a.weight
      if (appSort === "topics") {
        const aLen = a.pub_topic_ids.length + a.sub_topic_ids.length
        const bLen = b.pub_topic_ids.length + b.sub_topic_ids.length
        return bLen - aLen
      }
      return 0
    })
    return list
  }, [apps, appSearch, appSort])

  const filteredTopics = React.useMemo(() => {
    let list = topics.filter(t => {
      const matchesSearch =
        t.name.toLowerCase().includes(topicSearch.toLowerCase()) ||
        t.id.toLowerCase().includes(topicSearch.toLowerCase())
      const matchesQos =
        topicQosFilter === "all" ||
        (t.qos_reliability || "").toLowerCase() === topicQosFilter.toLowerCase() ||
        (t.qos_durability || "").toLowerCase() === topicQosFilter.toLowerCase() ||
        (t.qos_transport_priority || "").toLowerCase() === topicQosFilter.toLowerCase()
      return matchesSearch && matchesQos
    })
    list = [...list].sort((a, b) => {
      // Selected topics float to the top
      const aSelected = selectedTopicIds.includes(a.id)
      const bSelected = selectedTopicIds.includes(b.id)
      if (aSelected !== bSelected) return aSelected ? -1 : 1
      if (topicSort === "name")   return a.name.localeCompare(b.name)
      if (topicSort === "pub")    return b.publisher_count - a.publisher_count
      if (topicSort === "sub")    return b.subscriber_count - a.subscriber_count
      return 0
    })
    return list
  }, [topics, topicSearch, topicQosFilter, topicSort, selectedTopicIds])

  const qosOptions = React.useMemo(() => {
    const reliabilityVals = Array.from(new Set(topics.map(t => t.qos_reliability).filter(Boolean))) as string[]
    const durabilityVals = Array.from(new Set(topics.map(t => t.qos_durability).filter(Boolean))) as string[]
    const priorityVals = Array.from(new Set(topics.map(t => t.qos_transport_priority).filter(Boolean))) as string[]
    return { reliability: reliabilityVals, durability: durabilityVals, priority: priorityVals }
  }, [topics])

  function selectAllVisible() {
    const ids = filteredTopics.map(t => t.id)
    setSelectedTopicIds(prev => Array.from(new Set([...prev, ...ids])))
  }

  function deselectAllVisible() {
    const ids = new Set(filteredTopics.map(t => t.id))
    setSelectedTopicIds(prev => prev.filter(id => !ids.has(id)))
  }

  const allVisibleSelected =
    filteredTopics.length > 0 && filteredTopics.every(t => selectedTopicIds.includes(t.id))

  // ------------------------------------------------------------------
  // Render
  // ------------------------------------------------------------------

  // Loading state when page first opens
  if (!initialLoadComplete || status === 'connecting') {
    return (
      <AppLayout title="Simulator" description="Estimate pub-sub network and broker load">
        <div className="space-y-6">
          {/* Selection Card Skeleton */}
          <div className="rounded-xl border border-border bg-muted/20 p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex-1 space-y-3">
                <Skeleton className="h-6 w-6 rounded" />
                <Skeleton className="h-5 w-40" />
                <Skeleton className="h-4 w-56" />
              </div>
              <div className="flex items-center gap-2 shrink-0 flex-wrap justify-end">
                <Skeleton className="h-8 w-20" />
                <Skeleton className="h-8 w-32" />
                <Skeleton className="h-8 w-24" />
              </div>
            </div>
            {/* Tabs skeleton */}
            <div className="flex items-center gap-2 mb-4">
              <Skeleton className="h-9 w-20 rounded-md" />
              <Skeleton className="h-9 w-24 rounded-md" />
              <Skeleton className="h-9 w-20 rounded-md" />
            </div>
            {/* Selection list skeleton */}
            <div className="space-y-2">
              {Array.from({ length: 5 }).map((_, i) => (
                <div key={i} className="flex items-center gap-3 p-2">
                  <Skeleton className="h-4 w-4 rounded shrink-0" />
                  <Skeleton className="h-4 flex-1" style={{ width: `${50 + (i * 13) % 40}%`, flex: "none" }} />
                </div>
              ))}
            </div>
          </div>
        </div>
      </AppLayout>
    )
  }

  if (status !== "connected") {
    return (
      <AppLayout title="Simulator" description="Estimate pub-sub network and broker load">
        <NoConnectionInfo description="Connect to your Neo4j database to use the simulator" />
      </AppLayout>
    )
  }

  return (
    <AppLayout 
      title="Simulator" 
      description={simMode === "traffic" ? "Estimate pub-sub network and broker load for selected topics" : "Analyze the impact of component failures and cascades"}
    >
      <div className="space-y-6">

        {/* ── Simulation Mode Toggle ──────────────────────────────── */}
        <Tabs value={simMode} onValueChange={v => setSimMode(v as typeof simMode)}>
          <TabsList className="bg-background border border-border">
            <TabsTrigger value="traffic" className="flex items-center gap-2">
              <Network className="h-4 w-4" />Traffic Simulation
            </TabsTrigger>
            <TabsTrigger value="failure" className="flex items-center gap-2">
              <ShieldAlert className="h-4 w-4" />Failure Simulation
            </TabsTrigger>
          </TabsList>
        </Tabs>

        {/* ── Configuration Panel ─────────────────────────────────── */}
        <div className="space-y-6">

          {/* Traffic Simulation Configuration */}
          {simMode === "traffic" && (
          <Card className="border-border bg-background">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div>
                  <LayoutList className="h-6 w-6 text-blue-500 mb-3" />
                  <CardTitle className="font-semibold text-sm mb-1">
                    Selection
                    {selectedTopicIds.length > 0 && (
                      <Badge className="ml-2">{selectedTopicIds.length} topic{selectedTopicIds.length !== 1 ? "s" : ""} selected</Badge>
                    )}
                  </CardTitle>
                  <CardDescription className="text-sm">
                    Pick topics directly, or select applications to auto-include all their topics.
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2 shrink-0 flex-wrap justify-end">
                  {/* Global simulation params */}
                  <div className="flex items-center gap-1.5">
                    <Label className="text-xs text-muted-foreground whitespace-nowrap">Duration (s)</Label>
                    <Input
                      type="number" min={1} step={10}
                      value={durationSec}
                      onChange={e => setDurationSec(parseFloat(e.target.value) || 60)}
                      className="h-8 w-28 text-xs"
                    />
                  </div>
                  {selectedTopicIds.length > 0 && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-muted-foreground"
                      onClick={() => { setSelectedTopicIds([]); setSelectedRoleKeys(new Set()); setTopicParams({}) }}
                    >
                      <X className="h-3.5 w-3.5 mr-1" />
                      Clear all
                    </Button>
                  )}
                  <div className="flex items-center gap-1">
                    <Input
                      placeholder="Config name…"
                      value={newConfigName}
                      onChange={e => setNewConfigName(e.target.value)}
                      onKeyDown={e => { if (e.key === "Enter" && selectedTopicIds.length > 0) saveConfig() }}
                      className="h-8 text-xs w-32"
                      disabled={selectedTopicIds.length === 0}
                    />
                    <Button
                      onClick={saveConfig}
                      disabled={selectedTopicIds.length === 0}
                      size="sm"
                      variant="outline"
                    >
                      <Save className="h-4 w-4" />
                    </Button>
                  </div>
                  <Button
                    onClick={runSimulation}
                    disabled={loading || selectedTopicIds.length === 0}
                    size="sm"
                  >
                    {loading ? (
                      <><LoadingSpinner className="h-4 w-4 mr-2" />Simulating…</>
                    ) : (
                      <><Play className="h-4 w-4 mr-2" />Run Simulation</>
                    )}
                  </Button>
                </div>
              </div>
              {error && (
                <div className="flex items-start gap-2 text-destructive text-sm mt-2">
                  <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
                  {error}
                </div>
              )}
            </CardHeader>
            <CardContent className="space-y-3">
              {/* Saved configs list */}
              {savedConfigs.length > 0 && (
                <div className="space-y-1.5">
                  <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground px-0.5">
                    <Save className="h-3.5 w-3.5" />
                    Saved ({savedConfigs.length})
                  </div>
                <div className="rounded-md border border-border bg-muted/30 divide-y divide-border">
                  {savedConfigs.map(cfg => (
                    <div
                      key={cfg.id}
                      className="flex items-center justify-between px-3 py-2 hover:bg-muted/50 transition-colors"
                    >
                      <button
                        className="flex-1 min-w-0 text-left"
                        onClick={() => loadConfig(cfg)}
                        title={`Load: ${cfg.name}`}
                      >
                        <span className="text-sm font-medium text-foreground truncate block">{cfg.name}</span>
                        <span className="text-xs text-muted-foreground">
                          {cfg.topic_ids.length} topics · {cfg.duration_sec}s{cfg.topic_params && Object.keys(cfg.topic_params).length > 0 ? ' · per-topic frequencies' : ''}
                        </span>
                      </button>
                      <button
                        className="ml-3 shrink-0 text-muted-foreground hover:text-destructive transition-colors"
                        onClick={() => deleteConfig(cfg.id)}
                        title="Delete"
                      >
                        <X className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  ))}
                </div>
                </div>
              )}

              {/* Sub-tabs: Topics | Apps | Roles */}
              <Tabs value={selectionTab} onValueChange={v => setSelectionTab(v as typeof selectionTab)}>
                <TabsList className="bg-background border border-border">
                  <TabsTrigger value="roles" className="flex items-center gap-2">
                    <Zap className="h-4 w-4" />Roles
                    {apps.length > 0 && <span className="text-xs text-muted-foreground">({new Set(apps.flatMap(a => (a.role && a.role.length > 0) ? a.role : ["(unset)"])).size})</span>}
                  </TabsTrigger>
                  <TabsTrigger value="apps" className="flex items-center gap-2">
                    <Server className="h-4 w-4" />Applications
                    {apps.length > 0 && <span className="text-xs text-muted-foreground">({apps.length})</span>}
                  </TabsTrigger>
                  <TabsTrigger value="topics" className="flex items-center gap-2">
                    <MessageSquare className="h-4 w-4" />Topics
                    {topics.length > 0 && <span className="text-xs text-muted-foreground">({topics.length})</span>}
                  </TabsTrigger>
                </TabsList>
              </Tabs>

              {/* ── Topics panel ── */}
              {selectionTab === "topics" && (
                topicsLoading ? (
                  <div className="space-y-2 py-2">
                    {Array.from({ length: 6 }).map((_, i) => (
                      <div key={i} className="flex items-center gap-3 px-1 py-2 animate-pulse">
                        <div className="h-4 w-4 rounded bg-muted shrink-0" />
                        <div className="w-1 h-8 rounded-full bg-muted shrink-0" />
                        <div className="flex-1 space-y-1.5">
                          <div className="h-3 rounded bg-muted" style={{ width: `${55 + (i * 13) % 35}%` }} />
                          <div className="h-2.5 rounded bg-muted w-32" />
                        </div>
                        <div className="h-5 w-16 rounded-full bg-muted shrink-0" />
                      </div>
                    ))}
                  </div>
                ) : topicsError ? (
                  <div className="flex items-center gap-2 text-destructive text-sm">
                    <AlertCircle className="h-4 w-4" />
                    {topicsError}
                    <Button variant="ghost" size="sm" onClick={loadTopics}>Retry</Button>
                  </div>
                ) : (
                  <>
                    {/* Search + filter toolbar */}
                    <div className="flex flex-wrap gap-2">
                      <div className="relative flex-1 min-w-[180px]">
                        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
                        <Input
                          placeholder="Search topics…"
                          value={topicSearch}
                          onChange={e => setTopicSearch(e.target.value)}
                          className="pl-8 h-8 text-sm"
                        />
                        {topicSearch && (
                          <button
                            className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                            onClick={() => setTopicSearch("")}
                          >
                            <X className="h-3.5 w-3.5" />
                          </button>
                        )}
                      </div>

                      {/* QoS filter dropdown */}
                      <select
                        value={topicQosFilter}
                        onChange={e => setTopicQosFilter(e.target.value)}
                        className="h-8 text-xs rounded-md border border-input bg-background px-2 pr-6 focus:outline-none focus:ring-1 focus:ring-ring"
                      >
                        <option value="all">All QoS</option>
                        <optgroup label="Reliability">
                          {qosOptions.reliability.map(opt => (
                            <option key={opt} value={opt}>{opt}</option>
                          ))}
                        </optgroup>
                        <optgroup label="Durability">
                          {qosOptions.durability.map(opt => (
                            <option key={opt} value={opt}>{opt}</option>
                          ))}
                        </optgroup>
                        <optgroup label="Transport Priority">
                          {qosOptions.priority.map(opt => (
                            <option key={opt} value={opt}>{opt}</option>
                          ))}
                        </optgroup>
                      </select>

                      {/* Sort selector */}
                      <div className="flex items-center gap-1">
                        <ArrowUpDown className="h-3.5 w-3.5 text-muted-foreground" />
                        <select
                          value={topicSort}
                          onChange={e => setTopicSort(e.target.value as any)}
                          className="h-8 text-xs rounded-md border border-input bg-background px-2 pr-6 focus:outline-none focus:ring-1 focus:ring-ring"
                        >
                          <option value="name">Sort: Name</option>
                          <option value="pub">Sort: Publishers</option>
                          <option value="sub">Sort: Subscribers</option>
                        </select>
                      </div>
                    </div>

                    {/* Select-visible controls + count */}
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>
                        {filteredTopics.length} topic{filteredTopics.length !== 1 ? "s" : ""}
                        {topicSearch || topicQosFilter !== "all" ? ` (filtered from ${topics.length})` : ""}
                      </span>
                      <div className="flex gap-2">
                        <button
                          className="hover:text-foreground underline-offset-2 hover:underline disabled:opacity-40"
                          disabled={allVisibleSelected || filteredTopics.length === 0}
                          onClick={selectAllVisible}
                        >
                          Select all visible
                        </button>
                        <span>·</span>
                        <button
                          className="hover:text-foreground underline-offset-2 hover:underline disabled:opacity-40"
                          disabled={filteredTopics.every(t => !selectedTopicIds.includes(t.id))}
                          onClick={deselectAllVisible}
                        >
                          Deselect visible
                        </button>

                      </div>
                    </div>

                    {/* Topic list */}
                    <div className="border rounded-lg overflow-hidden">
                      {filteredTopics.length === 0 ? (
                        <div className="py-10 text-center text-sm text-muted-foreground">No topics match.</div>
                      ) : (
                        <div className="max-h-96 overflow-y-auto divide-y">
                          {filteredTopics.map(topic => {
                            const selected = selectedTopicIds.includes(topic.id)

                            return (
                              <div
                                key={topic.id}
                                className={`flex items-center gap-3 px-3 transition-colors ${selected ? "bg-primary/5 dark:bg-primary/10 py-2" : "py-2.5 hover:bg-muted/60"}`}
                              >
                                {/* Checkbox */}
                                <button
                                  onClick={() => toggleTopic(topic.id)}
                                  className="shrink-0"
                                >
                                  <div className={`h-4 w-4 rounded border-2 flex items-center justify-center transition-colors ${selected ? "bg-primary border-primary" : "border-muted-foreground/40"}`}>
                                    {selected && (
                                      <svg className="h-2.5 w-2.5 text-primary-foreground" fill="currentColor" viewBox="0 0 12 12">
                                        <path d="M10.28 2.28L3.989 8.575 1.695 6.28A1 1 0 00.28 7.695l3 3a1 1 0 001.414 0l7-7A1 1 0 0010.28 2.28z" />
                                      </svg>
                                    )}
                                  </div>
                                </button>

                                {/* Name + meta */}
                                <button
                                  onClick={() => toggleTopic(topic.id)}
                                  className="flex-1 min-w-0 text-left"
                                >
                                  <div className={`text-sm font-medium truncate ${selected ? "text-primary" : ""}`}>
                                    {topic.name}
                                  </div>
                                  <div className="flex items-center gap-2 mt-0.5">
                                    <span className="text-xs text-muted-foreground">
                                      {topic.publisher_count} pub · {topic.subscriber_count} sub{topic.size > 0 ? ` · ${topic.size >= 1024 ? `${(topic.size / 1024).toFixed(1)} KB` : `${topic.size} B`}` : ""}{topic.frequency != null && topic.frequency > 0 ? ` · ${Math.round(topic.frequency)} Hz` : ""}
                                    </span>
                                    {topic.broker_names.length > 0 && (
                                      <span className="text-xs text-muted-foreground hidden sm:inline">
                                        · {topic.broker_names.slice(0, 2).join(", ")}
                                        {topic.broker_names.length > 2 && ` +${topic.broker_names.length - 2}`}
                                      </span>
                                    )}
                                  </div>
                                </button>

                                {/* QoS badges */}
                                {(topic.qos_reliability || topic.qos_durability || topic.qos_transport_priority) && (
                                  <div className="flex items-center gap-1 shrink-0">
                                    {topic.qos_reliability && (
                                      <span className="inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-medium border border-green-500/40 text-green-600 dark:text-green-400 bg-green-500/5">
                                        {topic.qos_reliability}
                                      </span>
                                    )}
                                    {topic.qos_durability && (
                                      <span className="inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-medium border border-blue-500/40 text-blue-600 dark:text-blue-400 bg-blue-500/5">
                                        {topic.qos_durability}
                                      </span>
                                    )}
                                    {topic.qos_transport_priority && (
                                      <span className="inline-flex items-center rounded px-1.5 py-0.5 text-[10px] font-medium border border-amber-500/40 text-amber-600 dark:text-amber-400 bg-amber-500/5">
                                        TP {topic.qos_transport_priority}
                                      </span>
                                    )}
                                  </div>
                                )}

                                {/* Per-topic frequency (when selected) */}
                                {selected && (
                                  <div className="flex items-center gap-1 shrink-0" onClick={e => e.stopPropagation()}>
                                    {editingFreqId === topic.id ? (
                                      <>
                                        <Input
                                          type="number"
                                          min={0.001}
                                          step={1}
                                          autoFocus
                                          value={topicParams[topic.id]?.frequency_hz ?? (topic.frequency ?? 10.0)}
                                          onChange={e => {
                                            const val = parseFloat(e.target.value)
                                            if (!isNaN(val)) {
                                              setTopicParams(prev => ({
                                                ...prev,
                                                [topic.id]: {
                                                  frequency_hz: val,
                                                  duration_sec: topicParams[topic.id]?.duration_sec ?? durationSec,
                                                },
                                              }))
                                            }
                                          }}
                                          onBlur={() => setEditingFreqId(null)}
                                          onKeyDown={e => { if (e.key === "Enter" || e.key === "Escape") setEditingFreqId(null) }}
                                          className="h-6 w-16 text-xs px-1.5"
                                        />
                                        <span className="text-xs text-muted-foreground">Hz</span>
                                      </>
                                    ) : (
                                      <>
                                        <span className="text-xs font-medium text-foreground tabular-nums">
                                          {(topicParams[topic.id]?.frequency_hz ?? topic.frequency ?? 10.0).toFixed(1)}
                                        </span>
                                        <span className="text-xs text-muted-foreground">Hz</span>
                                        <button
                                          className="ml-0.5 text-muted-foreground hover:text-foreground transition-colors"
                                          onClick={() => setEditingFreqId(topic.id)}
                                          title="Edit frequency"
                                        >
                                          <Pencil className="h-3 w-3" />
                                        </button>
                                      </>
                                    )}
                                  </div>
                                )}
                              </div>
                            )
                          })}
                        </div>
                      )}
                    </div>
                  </>
                )
              )}

              {/* ── Apps panel ── */}
              {selectionTab === "apps" && (
                appsLoading ? (
                  <div className="space-y-2 py-2">
                    {Array.from({ length: 5 }).map((_, i) => (
                      <div key={i} className="flex items-center gap-3 px-1 py-2.5 animate-pulse">
                        <div className="h-4 w-4 rounded bg-muted shrink-0" />
                        <div className="flex-1 space-y-1.5">
                          <div className="h-3 rounded bg-muted" style={{ width: `${50 + (i * 17) % 40}%` }} />
                          <div className="h-2.5 rounded bg-muted w-40" />
                        </div>
                      </div>
                    ))}
                  </div>
                ) : appsError ? (
                  <div className="flex items-center gap-2 text-destructive text-sm">
                    <AlertCircle className="h-4 w-4" />
                    {appsError}
                    <Button variant="ghost" size="sm" onClick={loadApps}>Retry</Button>
                  </div>
                ) : (
                  <>
                    {/* Search + sort toolbar */}
                    <div className="flex flex-wrap gap-2">
                      <div className="relative flex-1 min-w-[180px]">
                        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
                        <Input
                          placeholder="Search applications…"
                          value={appSearch}
                          onChange={e => setAppSearch(e.target.value)}
                          className="pl-8 h-8 text-sm"
                        />
                        {appSearch && (
                          <button
                            className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                            onClick={() => setAppSearch("")}
                          >
                            <X className="h-3.5 w-3.5" />
                          </button>
                        )}
                      </div>
                      <div className="flex items-center gap-1">
                        <ArrowUpDown className="h-3.5 w-3.5 text-muted-foreground" />
                        <select
                          value={appSort}
                          onChange={e => setAppSort(e.target.value as any)}
                          className="h-8 text-xs rounded-md border border-input bg-background px-2 pr-6 focus:outline-none focus:ring-1 focus:ring-ring"
                        >
                          <option value="name">Sort: Name</option>
                          <option value="weight">Sort: Weight</option>
                          <option value="topics">Sort: Topics</option>
                        </select>
                      </div>
                    </div>

                    <div className="text-xs text-muted-foreground">
                      {filteredApps.length} application{filteredApps.length !== 1 ? "s" : ""}
                      {appSearch ? ` (filtered from ${apps.length})` : ""}
                      {" · Selecting an app includes all its topics in the simulation."}
                    </div>

                    {/* App list */}
                    <div className="border rounded-lg overflow-hidden">
                      {filteredApps.length === 0 ? (
                        <div className="py-10 text-center text-sm text-muted-foreground">
                          {apps.length === 0 ? "No Application nodes found in the graph." : "No applications match."}
                        </div>
                      ) : (
                        <div className="max-h-96 overflow-y-auto divide-y">
                          {[
                            ...filteredApps.filter(a => {
                              const ids = Array.from(new Set([...a.pub_topic_ids, ...a.sub_topic_ids]))
                              return ids.length > 0 && ids.every(tid => selectedTopicIds.includes(tid))
                            }),
                            ...filteredApps.filter(a => {
                              const ids = Array.from(new Set([...a.pub_topic_ids, ...a.sub_topic_ids]))
                              const count = ids.filter(tid => selectedTopicIds.includes(tid)).length
                              return count > 0 && count < ids.length
                            }),
                            ...filteredApps.filter(a => {
                              const ids = Array.from(new Set([...a.pub_topic_ids, ...a.sub_topic_ids]))
                              return ids.length === 0 || ids.every(tid => !selectedTopicIds.includes(tid))
                            }),
                          ].map(app => {
                            const allTopicIds = Array.from(new Set([...app.pub_topic_ids, ...app.sub_topic_ids]))
                            const selectedCount = allTopicIds.filter(tid => selectedTopicIds.includes(tid)).length
                            const isFullySelected = allTopicIds.length > 0 && selectedCount === allTopicIds.length
                            const isPartiallySelected = selectedCount > 0 && selectedCount < allTopicIds.length
                            const hasTopics = allTopicIds.length > 0

                            return (
                              <div
                                key={app.id}
                                className={`flex items-center gap-3 px-3 py-2.5 transition-colors ${isFullySelected ? "bg-primary/5 dark:bg-primary/10" : isPartiallySelected ? "bg-amber-50/50 dark:bg-amber-950/20" : "hover:bg-muted/60"}`}
                              >
                                {/* Checkbox */}
                                <button
                                  onClick={() => hasTopics && toggleApp(app)}
                                  className="shrink-0"
                                  disabled={!hasTopics}
                                >
                                  <div className={`h-4 w-4 rounded border-2 flex items-center justify-center transition-colors ${
                                    isFullySelected ? "bg-primary border-primary" : "border-muted-foreground/40"
                                  }`}>
                                    {isFullySelected && (
                                      <svg className="h-2.5 w-2.5 text-primary-foreground" fill="currentColor" viewBox="0 0 12 12">
                                        <path d="M10.28 2.28L3.989 8.575 1.695 6.28A1 1 0 00.28 7.695l3 3a1 1 0 001.414 0l7-7A1 1 0 0010.28 2.28z" />
                                      </svg>
                                    )}
                                  </div>
                                </button>

                                {/* Name */}
                                <button
                                  onClick={() => hasTopics && toggleApp(app)}
                                  className="flex-1 min-w-0 text-left"
                                  disabled={!hasTopics}
                                >
                                  <div className={`text-sm font-medium truncate ${isFullySelected ? "text-primary" : isPartiallySelected ? "text-amber-700 dark:text-amber-400" : ""}`}>
                                    {app.name}
                                  </div>
                                </button>

                                {/* Right: pub/sub counts + partial indicator */}
                                <div className="flex items-center gap-3 shrink-0">
                                  {isPartiallySelected && (
                                    <span className="text-xs text-amber-600 dark:text-amber-400 font-medium">
                                      {selectedCount}/{allTopicIds.length}
                                    </span>
                                  )}
                                  {hasTopics && (
                                    <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                                      {app.pub_topic_ids.length > 0 && (
                                        <span>{app.pub_topic_ids.length} pub</span>
                                      )}
                                      {app.pub_topic_ids.length > 0 && app.sub_topic_ids.length > 0 && (
                                        <span>·</span>
                                      )}
                                      {app.sub_topic_ids.length > 0 && (
                                        <span>{app.sub_topic_ids.length} sub</span>
                                      )}
                                    </div>
                                  )}
                                  {!hasTopics && (
                                    <span className="text-xs text-muted-foreground italic">no topics</span>
                                  )}
                                </div>
                              </div>
                            )
                          })}
                        </div>
                      )}
                    </div>
                  </>
                )
              )}

              {/* ── Roles panel ── */}
              {selectionTab === "roles" && (
                appsLoading ? (
                  <div className="space-y-2 py-2">
                    {Array.from({ length: 3 }).map((_, i) => (
                      <div key={i} className="flex items-center gap-3 px-1 py-2.5 animate-pulse">
                        <div className="h-4 w-4 rounded bg-muted shrink-0" />
                        <div className="flex-1 space-y-1.5">
                          <div className="h-3 rounded bg-muted" style={{ width: `${50 + (i * 17) % 40}%` }} />
                          <div className="h-2.5 rounded bg-muted w-40" />
                        </div>
                      </div>
                    ))}
                  </div>
                ) : appsError ? (
                  <div className="flex items-center gap-2 text-destructive text-sm">
                    <AlertCircle className="h-4 w-4" />
                    {appsError}
                    <Button variant="ghost" size="sm" onClick={loadApps}>Retry</Button>
                  </div>
                ) : (() => {
                  // Group apps by their individual roles; apps with multiple roles appear in multiple groups
                  const roleMap = new Map<string, AppInfo[]>()
                  for (const app of apps) {
                    const roles = (app.role && app.role.length > 0) ? app.role : ["(unset)"]
                    for (const r of roles) {
                      if (!roleMap.has(r)) roleMap.set(r, [])
                      roleMap.get(r)!.push(app)
                    }
                  }
                  const roleKeys = Array.from(roleMap.keys()).sort().filter(k =>
                    k.toLowerCase().includes(roleSearch.toLowerCase())
                  )

                  // A role is fully selected only when the user explicitly selected it.
                  function isRoleSelected(roleKey: string) {
                    return selectedRoleKeys.has(roleKey)
                  }

                  // Partial: some topics are selected (e.g. via the Apps/Topics tabs) but the
                  // role itself was not explicitly selected as a whole.
                  function isGroupPartiallySelected(roleKey: string, roleApps: AppInfo[]) {
                    if (selectedRoleKeys.has(roleKey)) return false
                    return roleApps.some(a => {
                      const ids = Array.from(new Set([...a.pub_topic_ids, ...a.sub_topic_ids]))
                      return ids.some(tid => selectedTopicIds.includes(tid))
                    })
                  }

                  function toggleGroup(roleKey: string, roleApps: AppInfo[]) {
                    const myIds = new Set(roleApps.flatMap(a => [...a.pub_topic_ids, ...a.sub_topic_ids]))
                    if (selectedRoleKeys.has(roleKey)) {
                      // Deselect: remove this role, keep topics still needed by other selected roles
                      setSelectedRoleKeys(prev => {
                        const next = new Set(prev)
                        next.delete(roleKey)
                        return next
                      })
                      const otherRoleTopics = new Set<string>()
                      for (const [otherKey, otherApps] of roleMap.entries()) {
                        if (otherKey === roleKey || !selectedRoleKeys.has(otherKey)) continue
                        for (const a of otherApps) {
                          for (const tid of [...a.pub_topic_ids, ...a.sub_topic_ids]) {
                            otherRoleTopics.add(tid)
                          }
                        }
                      }
                      setSelectedTopicIds(prev => prev.filter(tid => !myIds.has(tid) || otherRoleTopics.has(tid)))
                    } else {
                      // Select: add role and its topics
                      setSelectedRoleKeys(prev => new Set([...prev, roleKey]))
                      setSelectedTopicIds(prev => Array.from(new Set([...prev, ...myIds])))
                    }
                  }

                  return (
                    <div className="space-y-3">
                      {/* Search */}
                      <div className="relative flex-1 min-w-[180px]">
                        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
                        <Input
                          placeholder="Search roles..."
                          value={roleSearch}
                          onChange={e => setRoleSearch(e.target.value)}
                          className="pl-8 h-8 text-sm"
                        />
                        {roleSearch && (
                          <button
                            className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                            onClick={() => setRoleSearch("")}
                          >
                            <X className="h-3.5 w-3.5" />
                          </button>
                        )}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {roleKeys.length} role{roleKeys.length !== 1 ? "s" : ""}
                        {roleSearch ? ` (filtered from ${Array.from(roleMap.keys()).length})` : ""}
                        {" · Selecting a role includes every topic those apps publish to or subscribe to."}
                      </div>
                      {apps.length === 0 ? (
                        <div className="border rounded-lg overflow-hidden">
                          <div className="py-10 text-center text-sm text-muted-foreground">No Application nodes found in the graph.</div>
                        </div>
                      ) : roleKeys.length === 0 ? (
                        <div className="border rounded-lg overflow-hidden">
                          <div className="py-10 text-center text-sm text-muted-foreground">No role values set on Application nodes.</div>
                        </div>
                      ) : (
                        <div className="border rounded-lg overflow-hidden divide-y">
                          {[
                            ...roleKeys.filter(k => isRoleSelected(k)),
                            ...roleKeys.filter(k => isGroupPartiallySelected(k, roleMap.get(k)!)),
                            ...roleKeys.filter(k => !isRoleSelected(k) && !isGroupPartiallySelected(k, roleMap.get(k)!)),
                          ].map(roleKey => {
                            const roleApps = roleMap.get(roleKey)!
                            const fullySelected = isRoleSelected(roleKey)
                            const partiallySelected = isGroupPartiallySelected(roleKey, roleApps)
                            const topicCount = new Set(roleApps.flatMap(a => [...a.pub_topic_ids, ...a.sub_topic_ids])).size
                            return (
                              <div
                                key={roleKey}
                                className={`flex items-center gap-3 px-3 py-3 transition-colors ${fullySelected ? "bg-primary/5 dark:bg-primary/10" : partiallySelected ? "bg-amber-50/50 dark:bg-amber-950/20" : "hover:bg-muted/60"}`}
                              >
                                {/* Checkbox */}
                                <button onClick={() => toggleGroup(roleKey, roleApps)} className="shrink-0">
                                  <div className={`h-4 w-4 rounded border-2 flex items-center justify-center transition-colors ${
                                    fullySelected ? "bg-primary border-primary" : "border-muted-foreground/40"
                                  }`}>
                                    {fullySelected && (
                                      <svg className="h-2.5 w-2.5 text-primary-foreground" fill="currentColor" viewBox="0 0 12 12">
                                        <path d="M10.28 2.28L3.989 8.575 1.695 6.28A1 1 0 00.28 7.695l3 3a1 1 0 001.414 0l7-7A1 1 0 0010.28 2.28z" />
                                      </svg>
                                    )}
                                  </div>
                                </button>

                                {/* Label */}
                                <button
                                  onClick={() => toggleGroup(roleKey, roleApps)}
                                  className="flex-1 min-w-0 text-left"
                                >
                                  <span className={`text-sm font-medium ${fullySelected ? "text-primary" : partiallySelected ? "text-amber-700 dark:text-amber-400" : ""}`}>
                                    {roleKey}
                                  </span>
                                </button>

                                {/* Counts */}
                                <div className="flex items-center gap-1.5 shrink-0 text-xs text-muted-foreground">
                                  {partiallySelected && (
                                    <span className="text-amber-600 dark:text-amber-400 font-medium mr-1">partial</span>
                                  )}
                                  <span>{roleApps.length} app{roleApps.length !== 1 ? "s" : ""}</span>
                                  <span>·</span>
                                  <span>{topicCount} topic{topicCount !== 1 ? "s" : ""}</span>
                                </div>
                              </div>
                            )
                          })}
                        </div>
                      )}
                    </div>
                  )
                })()
              )}
            </CardContent>
          </Card>
          )}

          {/* Failure Simulation Configuration */}
          {simMode === "failure" && (
          <Card className="border-border bg-background">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div>
                  <ShieldAlert className="h-6 w-6 text-red-500 mb-3" />
                  <CardTitle className="font-semibold text-sm mb-1">
                    Selection
                    {failureTargetIds.length > 0 && (
                      <Badge className="ml-2">{failureTargetIds.length} component{failureTargetIds.length !== 1 ? "s" : ""} selected</Badge>
                    )}
                  </CardTitle>
                  <CardDescription className="text-sm">
                    Pick components to analyze their failure impact and cascade propagation.
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2 shrink-0 flex-wrap justify-end">
                  <div className="flex items-center gap-1.5">
                    <Label className="text-xs text-muted-foreground whitespace-nowrap">Layer</Label>
                    <Select 
                      value={failureLayer} 
                      onValueChange={setFailureLayer} 
                      disabled={failureLoading}
                    >
                      <SelectTrigger className="h-8 w-32 text-xs">
                        <SelectValue placeholder="Layer" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="app">Application</SelectItem>
                        <SelectItem value="infra">Infrastructure</SelectItem>
                        <SelectItem value="mw">Middleware</SelectItem>
                        <SelectItem value="system">System</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="flex items-center gap-1.5">
                    <Label className="text-xs text-muted-foreground whitespace-nowrap">Cascade</Label>
                    <Input
                      type="number"
                      min={0.0}
                      max={1.0}
                      step={0.1}
                      value={failureCascadeProb}
                      onChange={e => setFailureCascadeProb(parseFloat(e.target.value) || 1.0)}
                      disabled={failureLoading}
                      className="h-8 w-20 text-xs"
                    />
                  </div>
                  {failureTargetIds.length > 0 && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-muted-foreground"
                      onClick={() => {
                        setFailureTargetIds([])
                        setFailureResults(null)
                        setFailureError(null)
                      }}
                    >
                      <X className="h-3.5 w-3.5 mr-1" />
                      Clear all
                    </Button>
                  )}
                  <Button
                    onClick={runFailureSimulation}
                    disabled={failureLoading || failureTargetIds.length === 0}
                    size="sm"
                    className="bg-red-600 hover:bg-red-700 text-white"
                  >
                    {failureLoading ? (
                      <><LoadingSpinner className="h-4 w-4 mr-2" />Simulating…</>
                    ) : (
                      <><Play className="h-4 w-4 mr-2" />Run Simulation</>
                    )}
                  </Button>
                </div>
              </div>
              {failureError && (
                <div className="flex items-start gap-2 text-destructive text-sm mt-2">
                  <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
                  {failureError}
                </div>
              )}
            </CardHeader>
            <CardContent className="space-y-3">
              {/* Sub-tabs for component types */}
              <Tabs value={failureTypeTab || availableTypes[0]} onValueChange={setFailureTypeTab} className="w-full">
                <TabsList className="bg-background border border-border w-fit flex-wrap h-auto justify-start">
                  {availableTypes.map(type => {
                    const typeLower = type.toLowerCase()
                    let Icon = Box
                    if (typeLower.includes("app") || typeLower === "application") Icon = Server
                    else if (typeLower.includes("infra") || typeLower.includes("network")) Icon = Network
                    else if (typeLower.includes("mw") || typeLower.includes("middleware")) Icon = Layers
                    else if (typeLower.includes("system") || typeLower.includes("host") || typeLower.includes("node")) Icon = Cpu
                    else if (typeLower.includes("topic") || typeLower.includes("message")) Icon = MessageSquare
                    else if (typeLower.includes("broker") || typeLower.includes("database")) Icon = Database
                    else if (typeLower.includes("cloud") || typeLower.includes("aws") || typeLower.includes("azure")) Icon = Cloud
                    else if (typeLower.includes("hardware") || typeLower.includes("disk") || typeLower.includes("storage")) Icon = HardDrive
                    else if (typeLower.includes("router") || typeLower.includes("switch")) Icon = Router
                    
                    return (
                      <TabsTrigger key={type} value={type} className="flex items-center gap-2 capitalize">
                        <Icon className="h-4 w-4" />
                        {type.charAt(0).toUpperCase() + type.slice(1)}
                        <span className="text-xs text-muted-foreground">
                          ({(groupedComponents[type] || []).filter(c => 
                            c.name.toLowerCase().includes(failureSearch.toLowerCase()) || 
                            c.id.toLowerCase().includes(failureSearch.toLowerCase())
                          ).length})
                        </span>
                      </TabsTrigger>
                    )
                  })}
                </TabsList>

                <div>
                  {/* Search bar */}
                  <div className="flex flex-wrap gap-2">
                    <div className="relative flex-1 min-w-[180px]">
                      <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
                      <Input
                        placeholder="Search components…"
                        value={failureSearch}
                        onChange={e => setFailureSearch(e.target.value)}
                        className="pl-8 h-8 text-sm"
                      />
                      {failureSearch && (
                        <button
                          className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                          onClick={() => setFailureSearch("")}
                        >
                          <X className="h-3.5 w-3.5" />
                        </button>
                      )}
                    </div>
                  </div>

                  {availableTypes.map(type => {
                    const filtered = (groupedComponents[type] || []).filter(comp => 
                      comp.name.toLowerCase().includes(failureSearch.toLowerCase()) || 
                      comp.id.toLowerCase().includes(failureSearch.toLowerCase())
                    )
                    const total = (groupedComponents[type] || []).length
                    return (
                      <TabsContent key={type} value={type} className="mt-0">
                        <div className="text-xs text-muted-foreground py-3">
                          {filtered.length} component{filtered.length !== 1 ? "s" : ""}
                          {failureSearch && total !== filtered.length ? ` (filtered from ${total})` : ""}
                          {" · Selecting a component simulates its failure impact on the system."}
                        </div>
                        <div className="border rounded-lg overflow-hidden">
                          <div className="max-h-96 overflow-y-auto divide-y">
                            {filtered.map((comp) => {
                              const selected = failureTargetIds.includes(comp.id)
                              return (
                                <div
                                  key={comp.id}
                                  className={`flex items-center gap-3 px-3 transition-colors ${selected ? "bg-primary/5 dark:bg-primary/10 py-2" : "py-2.5 hover:bg-muted/60"}`}
                                >
                                  <button
                                    onClick={() => toggleFailureTarget(comp.id)}
                                    className="shrink-0"
                                  >
                                    <div className={`h-4 w-4 rounded border-2 flex items-center justify-center transition-colors ${selected ? "bg-primary border-primary" : "border-muted-foreground/40"}`}>
                                      {selected && (
                                        <svg className="h-2.5 w-2.5 text-primary-foreground" fill="currentColor" viewBox="0 0 12 12">
                                          <path d="M10.28 2.28L3.989 8.575 1.695 6.28A1 1 0 00.28 7.695l3 3a1 1 0 001.414 0l7-7A1 1 0 0010.28 2.28z" />
                                        </svg>
                                      )}
                                    </div>
                                  </button>
                                  <button
                                    onClick={() => toggleFailureTarget(comp.id)}
                                    className="flex-1 min-w-0 text-left"
                                  >
                                    <div className={`text-sm truncate ${selected ? "text-primary" : ""}`}>
                                      <span className={`font-medium ${selected ? "text-primary" : ""}`}>{comp.name}</span>
                                      <span className="text-xs text-muted-foreground ml-2">{comp.id}</span>
                                    </div>
                                  </button>
                                  {selected && (
                                    <Badge variant="secondary" className="shrink-0 text-xs">Selected</Badge>
                                  )}
                                </div>
                              )
                            })}
                          {filtered.length === 0 && (
                            <div className="py-8 text-center text-sm text-muted-foreground">
                              No components match your search.
                            </div>
                          )}
                        </div>
                        </div>
                      </TabsContent>
                    )
                  })}
                </div>
              </Tabs>
            </CardContent>
          </Card>
          )}
        </div>

        {/* ── Results ─────────────────────────────────────────────── */}
        {simMode === "traffic" && result && (() => {
          const usedMbps = result.summary.total_network_mbps
          const utilPct = Math.min((usedMbps / networkCapacityMbps) * 100, 100)
          const gaugeColor = utilPct > 85 ? "#ef4444" : utilPct > 60 ? "#f97316" : "#22c55e"

          const gaugeOption = {
            series: [{
              type: "gauge",
              startAngle: 210,
              endAngle: -30,
              min: 0,
              max: 100,
              radius: "85%",
              progress: { show: true, width: 14, itemStyle: { color: gaugeColor } },
              axisLine: { lineStyle: { width: 14, color: [[1, "#27272a"]] } },
              axisTick: { show: false },
              splitLine: { show: false },
              axisLabel: { show: false },
              pointer: { show: false },
              detail: {
                valueAnimation: true,
                formatter: (v: number) => `${v.toFixed(1)}%`,
                color: gaugeColor,
                fontSize: 22,
                fontWeight: "bold",
                offsetCenter: [0, "10%"],
              },
              title: { show: true, offsetCenter: [0, "38%"], fontSize: 12, color: "#71717a" },
              data: [{ value: parseFloat(utilPct.toFixed(2)), name: "Network Used" }],
            }],
            backgroundColor: "transparent",
          }

          const topicNames = result.per_topic.map(t => (t.topic_name ?? t.topic_id).length > 20 ? (t.topic_name ?? t.topic_id).slice(0, 18) + "…" : (t.topic_name ?? t.topic_id))
          const topicBwMbps = result.per_topic.map(t => parseFloat((t.bandwidth_total_bps / 1_000_000).toFixed(3)))
          const topicBarOption = {
            tooltip: { trigger: "axis", formatter: (p: any) => `${p[0].name}<br/>${p[0].value} MB/s` },
            grid: { top: 8, bottom: 40, left: 12, right: 12, containLabel: true },
            xAxis: { type: "category", data: topicNames, axisLabel: { fontSize: 11, color: "#a1a1aa", rotate: topicNames.length > 8 ? 30 : 0 } },
            yAxis: { type: "value", name: "MB/s", nameTextStyle: { color: "#a1a1aa", fontSize: 11 }, axisLabel: { color: "#a1a1aa", fontSize: 11 } },
            series: [{ type: "bar", data: topicBwMbps, itemStyle: { color: "#3b82f6", borderRadius: [3, 3, 0, 0] } }],
            backgroundColor: "transparent",
          }

          const brokerNames = result.broker_usage.map(b => b.broker_name.length > 16 ? b.broker_name.slice(0, 14) + "…" : b.broker_name)
          const brokerBwMbps = result.broker_usage.map(b => parseFloat(b.bandwidth_mbps.toFixed(3)))
          const brokerBarOption = {
            tooltip: { trigger: "axis", formatter: (p: any) => `${p[0].name}<br/>${p[0].value} MB/s` },
            grid: { top: 8, bottom: 40, left: 12, right: 12, containLabel: true },
            xAxis: { type: "category", data: brokerNames, axisLabel: { fontSize: 11, color: "#a1a1aa", rotate: brokerNames.length > 6 ? 30 : 0 } },
            yAxis: { type: "value", name: "MB/s", nameTextStyle: { color: "#a1a1aa", fontSize: 11 }, axisLabel: { color: "#a1a1aa", fontSize: 11 } },
            series: [{ type: "bar", data: brokerBwMbps, itemStyle: { color: "#a855f7", borderRadius: [3, 3, 0, 0] } }],
            backgroundColor: "transparent",
          }

          return (
            <Card className="border-border bg-background">
              <CardHeader className="pb-3">
                <div className="flex items-center justify-between flex-wrap gap-3">
                  <div>
                    <BarChart2 className="h-6 w-6 text-green-500 mb-3" />
                    <CardTitle className="font-semibold text-sm mb-1">Simulation Results</CardTitle>
                    <CardDescription className="text-sm">
                      Network utilisation for {result.summary.selected_topics} topic{result.summary.selected_topics !== 1 ? "s" : ""} · {result.summary.frequency_hz} Hz · {result.summary.duration_sec}s
                    </CardDescription>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <Label className="text-xs text-muted-foreground whitespace-nowrap">Network capacity</Label>
                    <Input
                      type="number"
                      min={1}
                      step={100}
                      value={networkCapacityMbps}
                      onChange={e => setNetworkCapacityMbps(parseFloat(e.target.value) || 1000)}
                      className="h-8 w-28 text-xs"
                    />
                    <span className="text-xs text-muted-foreground">MB/s</span>
                    <Button variant="ghost" size="sm" className="text-muted-foreground" onClick={() => setResult(null)}>
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Gauge + summary stats */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 items-center">
                  {/* Gauge */}
                  <div className="flex flex-col items-center">
                    <ReactECharts option={gaugeOption} style={{ height: 200, width: "100%" }} />
                    <div className="text-center -mt-4">
                      <div className="text-lg font-bold" style={{ color: gaugeColor }}>
                        {usedMbps.toFixed(2)} MB/s
                      </div>
                      <div className="text-xs text-muted-foreground">of {networkCapacityMbps} MB/s capacity</div>
                    </div>
                  </div>

                  {/* Stats */}
                  <div className="sm:col-span-2 grid grid-cols-2 gap-3">
                    {[
                      { label: "Total bandwidth", value: `${usedMbps.toFixed(2)} MB/s`, tip: "Combined inbound + outbound data rate across all selected topics." },
                      { label: "Total data volume", value: `${(result.summary.total_network_mbps * result.summary.duration_sec).toFixed(1)} MB`, tip: "Total data transferred over the entire simulation window. Equals bandwidth × duration." },
                      { label: "Published (total)", value: result.summary.total_msgs_published.toFixed(0), tip: `Total messages sent by all publishers over ${result.summary.duration_sec}s. Formula: publishers × Hz × duration.` },
                      { label: "Delivered (total)", value: result.summary.total_msgs_delivered.toFixed(0), tip: "Each published message is copied once per subscriber. Delivered = Published × subscriber count, so it is always ≥ Published." },
                      { label: "Throughput", value: `${result.per_topic.reduce((s, t) => s + t.msgs_total_per_sec, 0).toFixed(1)} msg/s`, tip: "Sum of msgs/s across all topics — combined message throughput rate." },
                      { label: "Peak topic", value: `${(result.summary.peak_topic_bps / 1_000_000).toFixed(2)} MB/s`, tip: "Highest per-topic bandwidth among all selected topics." },
                      { label: "Brokers involved", value: result.summary.brokers_involved.toString(), tip: "Number of distinct message brokers routing at least one selected topic." },
                      { label: "Topics simulated", value: `${result.summary.topics_found} / ${result.summary.selected_topics}`, tip: "Topics found in the graph out of the ones you selected." },
                    ].map(({ label, value, tip }) => (
                      <div key={label} className="rounded-md border border-border bg-muted/30 px-3 py-2">
                        <div className="flex items-center gap-1 text-xs text-muted-foreground">
                          {label}
                          <TermTooltip description={tip} iconOnly side="top" />
                        </div>
                        <div className="text-sm font-semibold mt-0.5">{value}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Per-topic bandwidth */}
                {result.per_topic.length > 0 && (
                  <div>
                    <div className="text-xs font-medium text-muted-foreground mb-2">Bandwidth per topic (MB/s)</div>
                    <ReactECharts option={topicBarOption} style={{ height: 200, width: "100%" }} />
                  </div>
                )}

                {/* Per-broker bandwidth */}
                {result.broker_usage.length > 0 && (
                  <div>
                    <div className="text-xs font-medium text-muted-foreground mb-2">Bandwidth per broker (MB/s)</div>
                    <ReactECharts option={brokerBarOption} style={{ height: 200, width: "100%" }} />
                  </div>
                )}
              </CardContent>
            </Card>
          )
        })()}

        {/* ── Failure Simulation Results ──────────────────────────── */}
        {simMode === "failure" && failureResults && failureResults.length > 0 && (() => {
          return (
            <>
              {failureResults.map((failureResult, index) => {
                const impactScore = failureResult.impact.composite_impact || 0;
                const gaugePct = Math.min(impactScore * 100, 100);
                const gaugeColor = impactScore > 0.5 ? "#ef4444" : impactScore > 0.25 ? "#f97316" : "#22c55e";

                const cascadeTypes = Object.keys(failureResult.impact.cascade?.by_type || {});
                const cascadeValues = Object.values(failureResult.impact.cascade?.by_type || {});

                const layerKeys = Object.keys(failureResult.layer_impacts || {});
                const layerValues = Object.values(failureResult.layer_impacts || {}).map((v: number) => parseFloat((v * 100).toFixed(1)));
                const sortedLayers = layerKeys.map((k, i) => ({ key: k, value: layerValues[i] })).sort((a, b) => b.value - a.value);
                const layerNames = sortedLayers.map(l => l.key.charAt(0).toUpperCase() + l.key.slice(1));
                const layerData = sortedLayers.map(l => l.value);

                const filteredCascaded = (failureResult.cascaded_failures || []).filter(id => {
                  const name = getComponentName(id).toLowerCase()
                  const type = getComponentType(id).toLowerCase()
                  const searchLower = cascadedSearch.toLowerCase()
                  return name.includes(searchLower) || type.includes(searchLower) || id.toLowerCase().includes(searchLower)
                })
                const getTypeColor = (type: string) => {
                  const t = type.toLowerCase()
                  if (t.includes("app")) return "border-blue-500/40 text-blue-600 dark:text-blue-400 bg-blue-500/5"
                  if (t.includes("broker")) return "border-purple-500/40 text-purple-600 dark:text-purple-400 bg-purple-500/5"
                  if (t.includes("host") || t.includes("node") || t.includes("system")) return "border-green-500/40 text-green-600 dark:text-green-400 bg-green-500/5"
                  return "border-gray-500/40 text-gray-600 dark:text-gray-400 bg-gray-500/5"
                }

                return (
                  <Card key={failureResult.target_id} className="border-border bg-background">
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between flex-wrap gap-3">
                        <div>
                          <ShieldAlert className="h-6 w-6 text-red-500 mb-3" />
                          <CardTitle className="font-semibold text-sm mb-1">Failure Simulation Results</CardTitle>
                          <CardDescription className="text-sm">
                            Impact analysis for {getComponentName(failureResult.target_id)} ({failureResult.target_type})
                          </CardDescription>
                        </div>
                        {failureResults.length === 1 && (
                          <Button variant="ghost" size="sm" className="text-muted-foreground" onClick={() => setFailureResults(null)}>
                            <X className="h-4 w-4" />
                          </Button>
                        )}
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-6">
                      {/* Gauge + summary stats */}
                      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 items-center">
                        {/* Gauge */}
                        <div className="flex flex-col items-center">
                          <ReactECharts
                            option={{
                              series: [{
                                type: "gauge",
                                startAngle: 210,
                                endAngle: -30,
                                min: 0,
                                max: 100,
                                radius: "85%",
                                progress: { show: true, width: 14, itemStyle: { color: gaugeColor } },
                                axisLine: { lineStyle: { width: 14, color: [[1, "#27272a"]] } },
                                axisTick: { show: false },
                                splitLine: { show: false },
                                axisLabel: { show: false },
                                pointer: { show: false },
                                detail: {
                                  valueAnimation: true,
                                  formatter: (v: number) => v.toFixed(1),
                                  color: gaugeColor,
                                  fontSize: 22,
                                  fontWeight: "bold",
                                  offsetCenter: [0, "10%"],
                                },
                                title: { show: true, offsetCenter: [0, "38%"], fontSize: 12, color: "#71717a" },
                                data: [{ value: parseFloat(gaugePct.toFixed(1)), name: "Composite Impact" }],
                              }],
                              backgroundColor: "transparent",
                            }}
                            style={{ height: 200, width: "100%" }}
                          />
                          <div className="text-center -mt-4">
                            <div className={`text-lg font-bold ${getImpactColor(impactScore)}`}>
                              {impactScore.toFixed(3)}
                            </div>
                            <div className="text-xs text-muted-foreground">normalized impact score</div>
                          </div>
                        </div>

                        {/* Stats */}
                        <div className="sm:col-span-2 grid grid-cols-2 gap-3">
                          {[
                            { label: "Reachability Loss", value: `${(failureResult.impact.reachability?.loss_percent || 0).toFixed(1)}%`, tip: "Percentage of broken pub-sub paths due to this failure." },
                            { label: "Throughput Loss", value: `${(failureResult.impact.throughput?.loss_percent || 0).toFixed(1)}%`, tip: "QoS-weighted reduction in message delivery capacity." },
                            { label: "Flow Disruption", value: `${(failureResult.impact.flow_disruption?.loss_percent || 0).toFixed(1)}%`, tip: "Percentage of data flow paths disrupted by this failure." },
                            { label: "Cascade Count", value: `${failureResult.impact.cascade?.count || 0}`, tip: "Number of additional components affected by the cascading failure." },
                            { label: "Max Cascade Depth", value: `${failureResult.impact.cascade?.depth || 0}`, tip: "Longest chain of dependent failures triggered by this component." },
                            { label: "Topics Affected", value: `${failureResult.impact.affected?.topics || 0}`, tip: "Number of pub/sub topics that lose connectivity or capacity." },
                            { label: "Publishers Affected", value: `${failureResult.impact.affected?.publishers || 0}`, tip: "Number of publisher applications disconnected or degraded." },
                            { label: "Subscribers Affected", value: `${failureResult.impact.affected?.subscribers || 0}`, tip: "Number of subscriber applications disconnected or degraded." },
                          ].map(({ label, value, tip }) => (
                            <div key={label} className="rounded-md border border-border bg-muted/30 px-3 py-2">
                              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                                {label}
                                <TermTooltip description={tip} iconOnly side="top" />
                              </div>
                              <div className="text-sm font-semibold mt-0.5">{value}</div>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Cascade by Component Type */}
                      {cascadeTypes.length > 0 && (
                        <div>
                          <div className="text-xs font-medium text-muted-foreground mb-2">Cascaded Failures by Component Type</div>
                          <ReactECharts
                            option={{
                              tooltip: { trigger: "axis", formatter: (p: any) => `${p[0].name}<br/>${p[0].value} components` },
                              grid: { top: 8, bottom: 40, left: 12, right: 12, containLabel: true },
                              xAxis: { type: "category", data: cascadeTypes, axisLabel: { fontSize: 11, color: "#a1a1aa", rotate: cascadeTypes.length > 5 ? 30 : 0 } },
                              yAxis: { type: "value", name: "Count", nameTextStyle: { color: "#a1a1aa", fontSize: 11 }, axisLabel: { color: "#a1a1aa", fontSize: 11 } },
                              series: [{ type: "bar", data: cascadeValues, itemStyle: { color: "#ef4444", borderRadius: [3, 3, 0, 0] } }],
                              backgroundColor: "transparent",
                            }}
                            style={{ height: 200, width: "100%" }}
                          />
                        </div>
                      )}

                      {/* Impact by Layer */}
                      {layerKeys.length > 0 && (
                        <div>
                          <div className="text-xs font-medium text-muted-foreground mb-2">Impact by Architectural Layer</div>
                          <ReactECharts
                            option={{
                              tooltip: { trigger: "axis", formatter: (p: any) => `${p[0].name}<br/>${p[0].value}% impact` },
                              grid: { top: 8, bottom: 8, left: 70, right: 12, containLabel: true },
                              xAxis: { type: "value", name: "Impact %", nameTextStyle: { color: "#a1a1aa", fontSize: 11 }, axisLabel: { color: "#a1a1aa", fontSize: 11 } },
                              yAxis: { type: "category", data: layerNames, axisLabel: { fontSize: 11, color: "#a1a1aa" } },
                              series: [{ 
                                type: "bar", 
                                data: layerData, 
                                itemStyle: { color: "#f97316", borderRadius: [0, 3, 3, 0] } 
                              }],
                              backgroundColor: "transparent",
                            }}
                            style={{ height: Math.max(150, layerNames.length * 40), width: "100%" }}
                          />
                        </div>
                      )}

                      {/* Cascaded Failures List */}
                      {failureResult.cascaded_failures && failureResult.cascaded_failures.length > 0 && (
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <div className="text-xs font-medium text-muted-foreground">
                              Detailed Cascaded Failures ({filteredCascaded.length}{(failureResult.cascaded_failures || []).length > filteredCascaded.length ? ` / ${(failureResult.cascaded_failures || []).length}` : ''})
                            </div>
                            <div className="relative w-48">
                              <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-muted-foreground pointer-events-none" />
                              <Input 
                                placeholder="Search failures..." 
                                value={cascadedSearch} 
                                onChange={e => setCascadedSearch(e.target.value)}
                                className="h-7 pl-7 text-xs"
                              />
                              {cascadedSearch && (
                                <button
                                  className="absolute right-1.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                                  onClick={() => setCascadedSearch("")}
                                >
                                  <X className="h-3 w-3" />
                                </button>
                              )}
                            </div>
                          </div>
                          <div className="max-h-64 overflow-y-auto rounded-md border border-border bg-muted/30 p-2 space-y-1.5">
                            {filteredCascaded.length === 0 ? (
                              <div className="text-xs text-muted-foreground text-center py-4">No matching failures found.</div>
                            ) : (
                              filteredCascaded.map((id) => {
                                const type = getComponentType(id)
                                const name = getComponentName(id)
                                return (
                                  <div key={id} className="flex items-center gap-2 p-2 rounded-md bg-background/50 border border-border/50 hover:border-border transition-colors">
                                    <Badge variant="outline" className={`text-[10px] h-5 px-1.5 font-mono uppercase shrink-0 ${getTypeColor(type)}`}>
                                      {type}
                                    </Badge>
                                    <span className="flex-1 text-sm font-medium truncate" title={name}>{name}</span>
                                    <span className="text-xs text-muted-foreground font-mono truncate max-w-[120px]" title={id}>{id}</span>
                                  </div>
                                )
                              })
                            )}
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                );
              })}
              {failureResults.length > 1 && (
                <div className="flex justify-center mt-4">
                  <Button variant="outline" size="sm" onClick={() => setFailureResults(null)}>
                    <X className="h-4 w-4 mr-2" />
                    Clear Results
                  </Button>
                </div>
              )}
            </>
          );
        })()}
      </div>
    </AppLayout>
  )
}
