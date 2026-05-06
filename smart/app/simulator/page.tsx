"use client"

import React, { useState, useEffect, useCallback } from "react"
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
} from "lucide-react"
import { useConnection } from "@/lib/stores/connection-store"
import { trafficClient, type TopicInfo, type AppInfo, type TopicParams, type TrafficSimulationResult } from "@/lib/api/traffic-client"
import { TermTooltip } from "@/components/ui/term-tooltip"
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
  const [topicParams, setTopicParams] = useState<Record<string, TopicParams>>({})
  const [topicSearch, setTopicSearch] = useState("")
  const [topicQosFilter, setTopicQosFilter] = useState<string>("all")
  const [topicSort, setTopicSort] = useState<"name" | "pub" | "sub">("name")

  // ---- Parameters ----
  const [frequencyHz, setFrequencyHz] = useState<number>(10)
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

  // ------------------------------------------------------------------
  // Effects
  // ------------------------------------------------------------------

  // Load topics when connected
  useEffect(() => {
    if (status === "connected") {
      loadTopics()
      loadApps()
    } else {
      setTopics([])
      setApps([])
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
        frequency_hz: frequencyHz,
        duration_sec: durationSec,
        message_size_bytes: messageSizeBytes,
      })
      setResult(data)
    } catch (err: any) {
      setError(err.message || "Simulation failed")
    } finally {
      setLoading(false)
    }
  }

  function toggleTopic(id: string) {
    setSelectedTopicIds(prev => {
      if (prev.includes(id)) {
        return prev.filter(t => t !== id)
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
    }
  }

  function saveConfig() {
    const name = newConfigName.trim() || `Config ${savedConfigs.length + 1}`
    const cfg: SavedConfig = {
      id: Date.now().toString(),
      name,
      topic_ids: [...selectedTopicIds],
      frequency_hz: frequencyHz,
      duration_sec: durationSec,
      message_size_bytes: messageSizeBytes,
      topic_params: Object.keys(topicParams).length > 0 ? { ...topicParams } : undefined,
      created_at: new Date().toISOString(),
    }
    setSavedConfigs(prev => [cfg, ...prev])
    setNewConfigName("")
  }

  function loadConfig(cfg: SavedConfig) {
    setSelectedTopicIds(cfg.topic_ids)
    setFrequencyHz(cfg.frequency_hz)
    setDurationSec(cfg.duration_sec)
    setMessageSizeBytes(cfg.message_size_bytes)
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
      if (topicSort === "name")   return a.name.localeCompare(b.name)
      if (topicSort === "pub")    return b.publisher_count - a.publisher_count
      if (topicSort === "sub")    return b.subscriber_count - a.subscriber_count
      return 0
    })
    return list
  }, [topics, topicSearch, topicQosFilter, topicSort])

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
        <NoConnectionInfo description="Connect to your Neo4j database to use the traffic simulator" />
      </AppLayout>
    )
  }

  return (
    <AppLayout title="Simulator" description="Estimate pub-sub network and broker load for selected topics">
      <div className="space-y-6">

        {/* ── Configuration Panel ─────────────────────────────────── */}
        <div className="space-y-6">

          {/* Topic/App Selection */}
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
                    <Label className="text-xs text-muted-foreground whitespace-nowrap">Hz</Label>
                    <Input
                      type="number" min={0.001} step={1}
                      value={frequencyHz}
                      onChange={e => setFrequencyHz(parseFloat(e.target.value) || 10)}
                      className="h-8 w-16 text-xs"
                    />
                  </div>
                  <div className="flex items-center gap-1.5">
                    <Label className="text-xs text-muted-foreground whitespace-nowrap">Duration (s)</Label>
                    <Input
                      type="number" min={1} step={10}
                      value={durationSec}
                      onChange={e => setDurationSec(parseFloat(e.target.value) || 60)}
                      className="h-8 w-20 text-xs"
                    />
                  </div>
                  {selectedTopicIds.length > 0 && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-muted-foreground"
                      onClick={() => { setSelectedTopicIds([]); setTopicParams({}) }}
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
                          {cfg.topic_ids.length} topics · {cfg.frequency_hz} Hz · {cfg.duration_sec}s
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
                    {apps.length > 0 && <span className="text-xs text-muted-foreground">({new Set(apps.map(a => a.role ?? "(unset)")).size})</span>}
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
                                      {topic.publisher_count} pub · {topic.subscriber_count} sub{topic.size > 0 ? ` · ${topic.size >= 1024 ? `${(topic.size / 1024).toFixed(1)} KB` : `${topic.size} B`}` : ""}
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
                          {filteredApps.map(app => {
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
                                    isFullySelected ? "bg-primary border-primary"
                                    : isPartiallySelected ? "bg-amber-400 border-amber-400"
                                    : "border-muted-foreground/40"
                                  }`}>
                                    {isFullySelected && (
                                      <svg className="h-2.5 w-2.5 text-primary-foreground" fill="currentColor" viewBox="0 0 12 12">
                                        <path d="M10.28 2.28L3.989 8.575 1.695 6.28A1 1 0 00.28 7.695l3 3a1 1 0 001.414 0l7-7A1 1 0 0010.28 2.28z" />
                                      </svg>
                                    )}
                                    {isPartiallySelected && (
                                      <svg className="h-2 w-2 text-white" fill="currentColor" viewBox="0 0 12 12">
                                        <rect x="1" y="5" width="10" height="2" rx="1" />
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
                  // Group apps by their `role` field; apps with null role go under "(unset)"
                  const roleMap = new Map<string, AppInfo[]>()
                  for (const app of apps) {
                    const key = app.role ?? "(unset)"
                    if (!roleMap.has(key)) roleMap.set(key, [])
                    roleMap.get(key)!.push(app)
                  }
                  const roleKeys = Array.from(roleMap.keys()).sort()

                  function isGroupFullySelected(roleApps: AppInfo[]) {
                    return roleApps.length > 0 && roleApps.every(a => {
                      const ids = Array.from(new Set([...a.pub_topic_ids, ...a.sub_topic_ids]))
                      return ids.length > 0 && ids.every(tid => selectedTopicIds.includes(tid))
                    })
                  }

                  function isGroupPartiallySelected(roleApps: AppInfo[]) {
                    return roleApps.some(a => {
                      const ids = Array.from(new Set([...a.pub_topic_ids, ...a.sub_topic_ids]))
                      return ids.some(tid => selectedTopicIds.includes(tid))
                    }) && !isGroupFullySelected(roleApps)
                  }

                  function toggleGroup(roleApps: AppInfo[]) {
                    const full = isGroupFullySelected(roleApps)
                    if (full) {
                      const allIds = new Set(roleApps.flatMap(a => [...a.pub_topic_ids, ...a.sub_topic_ids]))
                      setSelectedTopicIds(prev => prev.filter(tid => !allIds.has(tid)))
                    } else {
                      const allIds = Array.from(new Set(roleApps.flatMap(a => [...a.pub_topic_ids, ...a.sub_topic_ids])))
                      setSelectedTopicIds(prev => Array.from(new Set([...prev, ...allIds])))
                    }
                  }

                  return (
                    <div className="space-y-3">
                      <div className="text-xs text-muted-foreground">
                        Distinct <code className="font-mono">role</code> values from Application nodes. Selecting a role includes every topic those apps publish to or subscribe to.
                      </div>
                      {apps.length === 0 ? (
                        <div className="py-10 text-center text-sm text-muted-foreground">No Application nodes found in the graph.</div>
                      ) : roleKeys.length === 0 ? (
                        <div className="py-10 text-center text-sm text-muted-foreground">No role values set on Application nodes.</div>
                      ) : (
                        <div className="border rounded-lg overflow-hidden divide-y">
                          {roleKeys.map(roleKey => {
                            const roleApps = roleMap.get(roleKey)!
                            const fullySelected = isGroupFullySelected(roleApps)
                            const partiallySelected = isGroupPartiallySelected(roleApps)
                            const topicCount = new Set(roleApps.flatMap(a => [...a.pub_topic_ids, ...a.sub_topic_ids])).size
                            return (
                              <div
                                key={roleKey}
                                className={`flex items-center gap-3 px-3 py-3 transition-colors ${fullySelected ? "bg-primary/5 dark:bg-primary/10" : partiallySelected ? "bg-amber-50/50 dark:bg-amber-950/20" : "hover:bg-muted/60"}`}
                              >
                                {/* Checkbox */}
                                <button onClick={() => toggleGroup(roleApps)} className="shrink-0">
                                  <div className={`h-4 w-4 rounded border-2 flex items-center justify-center transition-colors ${
                                    fullySelected ? "bg-primary border-primary"
                                    : partiallySelected ? "bg-amber-400 border-amber-400"
                                    : "border-muted-foreground/40"
                                  }`}>
                                    {fullySelected && (
                                      <svg className="h-2.5 w-2.5 text-primary-foreground" fill="currentColor" viewBox="0 0 12 12">
                                        <path d="M10.28 2.28L3.989 8.575 1.695 6.28A1 1 0 00.28 7.695l3 3a1 1 0 001.414 0l7-7A1 1 0 0010.28 2.28z" />
                                      </svg>
                                    )}
                                    {partiallySelected && (
                                      <svg className="h-2 w-2 text-white" fill="currentColor" viewBox="0 0 12 12">
                                        <rect x="1" y="5" width="10" height="2" rx="1" />
                                      </svg>
                                    )}
                                  </div>
                                </button>

                                {/* Label */}
                                <button
                                  onClick={() => toggleGroup(roleApps)}
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
        </div>

        {/* ── Results ─────────────────────────────────────────────── */}
        {result && (() => {
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
      </div>
    </AppLayout>
  )
}
