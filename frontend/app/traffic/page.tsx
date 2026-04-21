"use client"

import React, { useState, useEffect, useCallback } from "react"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Activity,
  Network,
  Server,
  Wifi,
  Play,
  Save,
  Trash2,
  FolderOpen,
  AlertCircle,
  CheckCircle2,
  ChevronsUpDown,
  X,
  Zap,
  BarChart3,
  MessageSquare,
  Info,
  ChevronDown,
  ChevronUp,
  Search,
  ArrowUpDown,
} from "lucide-react"
import { useConnection } from "@/lib/stores/connection-store"
import { trafficClient, type TopicInfo, type TrafficSimulationResult, type TopicParams } from "@/lib/api/traffic-client"

// ============================================================================
// Types
// ============================================================================

interface SavedConfig {
  id: string
  name: string
  topic_ids: string[]
  frequency_hz: number
  duration_sec: number
  message_size_bytes: number
  topic_params?: Record<string, TopicParams>
  created_at: string
}

// ============================================================================
// Helpers
// ============================================================================

function formatBytes(bps: number): string {
  if (bps >= 1_000_000_000) return `${(bps / 1_000_000_000).toFixed(2)} GB/s`
  if (bps >= 1_000_000) return `${(bps / 1_000_000).toFixed(2)} MB/s`
  if (bps >= 1_000) return `${(bps / 1_000).toFixed(2)} KB/s`
  return `${bps.toFixed(2)} B/s`
}

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return n.toFixed(0)
}

function getBandwidthColor(bps: number, maxBps: number): string {
  if (maxBps === 0) return "text-gray-500"
  const ratio = bps / maxBps
  if (ratio > 0.8) return "text-red-500"
  if (ratio > 0.5) return "text-orange-500"
  if (ratio > 0.2) return "text-yellow-500"
  return "text-green-500"
}

const LS_RESULT_KEY = "traffic_sim_result"
const LS_CONFIGS_KEY = "traffic_sim_configs"

// ============================================================================
// Main Page
// ============================================================================

export default function TrafficSimulatorPage() {
  const { status } = useConnection()

  // ---- Topics ----
  const [topics, setTopics] = useState<TopicInfo[]>([])
  const [topicsLoading, setTopicsLoading] = useState(false)
  const [topicsError, setTopicsError] = useState<string | null>(null)

  // ---- Selection ----
  const [selectedTopicIds, setSelectedTopicIds] = useState<string[]>([])
  const [topicParams, setTopicParams] = useState<Record<string, TopicParams>>({})
  const [topicSearch, setTopicSearch] = useState("")
  const [topicQosFilter, setTopicQosFilter] = useState<string>("all")
  const [topicSort, setTopicSort] = useState<"name" | "weight" | "pub" | "sub">("weight")

  // ---- Parameters ----
  const [frequencyHz, setFrequencyHz] = useState<number>(10)
  const [durationSec, setDurationSec] = useState<number>(60)
  const [messageSizeBytes, setMessageSizeBytes] = useState<number>(1024)

  // ---- Simulation ----
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<TrafficSimulationResult | null>(null)

  // ---- Saved configs ----
  const [savedConfigs, setSavedConfigs] = useState<SavedConfig[]>([])
  const [newConfigName, setNewConfigName] = useState("")
  const [configsOpen, setConfigsOpen] = useState(true)

  // ---- How it works panel ----
  const [howItWorksOpen, setHowItWorksOpen] = useState(false)

  // ------------------------------------------------------------------
  // Effects
  // ------------------------------------------------------------------

  // Load topics when connected
  useEffect(() => {
    if (status === "connected") {
      loadTopics()
    } else {
      setTopics([])
    }
  }, [status])

  // Restore result + saved configs from localStorage
  useEffect(() => {
    if (typeof window === "undefined") return
    const saved = localStorage.getItem(LS_RESULT_KEY)
    if (saved) {
      try { setResult(JSON.parse(saved)) } catch { /* ignore */ }
    }
    const cfgs = localStorage.getItem(LS_CONFIGS_KEY)
    if (cfgs) {
      try { setSavedConfigs(JSON.parse(cfgs)) } catch { /* ignore */ }
    }
  }, [])

  // Persist result
  useEffect(() => {
    if (typeof window !== "undefined" && result) {
      localStorage.setItem(LS_RESULT_KEY, JSON.stringify(result))
    }
  }, [result])

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
        per_topic_params: Object.keys(topicParams).length > 0 ? topicParams : undefined,
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
        // Remove params when deselected
        setTopicParams(p => { const next = { ...p }; delete next[id]; return next })
        return prev.filter(t => t !== id)
      }
      // Init params from current global defaults when newly selected
      setTopicParams(p => ({
        ...p,
        [id]: { frequency_hz: frequencyHz, duration_sec: durationSec },
      }))
      return [...prev, id]
    })
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
    setTopicParams(cfg.topic_params ? { ...cfg.topic_params } : {})
    setConfigsOpen(false)
  }

  function deleteConfig(id: string) {
    setSavedConfigs(prev => prev.filter(c => c.id !== id))
  }

  function clearResult() {
    setResult(null)
    if (typeof window !== "undefined") localStorage.removeItem(LS_RESULT_KEY)
  }

  // ------------------------------------------------------------------
  // Derived
  // ------------------------------------------------------------------

  const topicById = Object.fromEntries(topics.map(t => [t.id, t]))

  const filteredTopics = React.useMemo(() => {
    let list = topics.filter(t => {
      const matchesSearch =
        t.name.toLowerCase().includes(topicSearch.toLowerCase()) ||
        t.id.toLowerCase().includes(topicSearch.toLowerCase())
      const matchesQos =
        topicQosFilter === "all" ||
        (t.qos_reliability || "").toLowerCase() === topicQosFilter.toLowerCase()
      return matchesSearch && matchesQos
    })
    list = [...list].sort((a, b) => {
      if (topicSort === "name")   return a.name.localeCompare(b.name)
      if (topicSort === "weight") return b.weight - a.weight
      if (topicSort === "pub")    return b.publisher_count - a.publisher_count
      if (topicSort === "sub")    return b.subscriber_count - a.subscriber_count
      return 0
    })
    return list
  }, [topics, topicSearch, topicQosFilter, topicSort])

  const qosOptions = React.useMemo(() => {
    const vals = Array.from(new Set(topics.map(t => t.qos_reliability).filter(Boolean))) as string[]
    return vals
  }, [topics])

  const maxWeight = React.useMemo(() => Math.max(...topics.map(t => t.weight), 0.01), [topics])

  function selectAllVisible() {
    const ids = filteredTopics.map(t => t.id)
    setSelectedTopicIds(prev => Array.from(new Set([...prev, ...ids])))
    setTopicParams(prev => {
      const next = { ...prev }
      for (const t of filteredTopics) {
        if (!next[t.id]) {
          next[t.id] = { frequency_hz: frequencyHz, duration_sec: durationSec }
        }
      }
      return next
    })
  }

  function deselectAllVisible() {
    const ids = new Set(filteredTopics.map(t => t.id))
    setSelectedTopicIds(prev => prev.filter(id => !ids.has(id)))
    setTopicParams(prev => {
      const next = { ...prev }
      for (const id of ids) delete next[id]
      return next
    })
  }

  const allVisibleSelected =
    filteredTopics.length > 0 && filteredTopics.every(t => selectedTopicIds.includes(t.id))

  const maxBrokerBps = result
    ? Math.max(...result.broker_usage.map(b => b.bandwidth_bps), 1)
    : 1

  // ------------------------------------------------------------------
  // Render
  // ------------------------------------------------------------------

  if (status !== "connected") {
    return (
      <AppLayout title="Traffic Simulator" description="Estimate pub-sub network and broker load">
        <div className="flex flex-col items-center justify-center h-64 gap-4">
          <AlertCircle className="h-12 w-12 text-muted-foreground" />
          <p className="text-muted-foreground">Connect to Neo4j to use the traffic simulator.</p>
        </div>
      </AppLayout>
    )
  }

  return (
    <AppLayout title="Traffic Simulator" description="Estimate pub-sub network and broker load for selected topics">
      <div className="space-y-6">

        {/* ── How it works ────────────────────────────────────────── */}
        <Card className="border-blue-200 dark:border-blue-900 bg-blue-50/50 dark:bg-blue-950/20">
          <CardHeader className="pb-2">
            <button
              className="flex w-full items-center justify-between text-left"
              onClick={() => setHowItWorksOpen(v => !v)}
            >
              <div className="flex items-center gap-2">
                <Info className="h-5 w-5 text-blue-500" />
                <CardTitle className="text-base text-blue-700 dark:text-blue-300">How this works</CardTitle>
              </div>
              {howItWorksOpen
                ? <ChevronUp className="h-4 w-4 text-blue-500" />
                : <ChevronDown className="h-4 w-4 text-blue-500" />
              }
            </button>
          </CardHeader>
          {howItWorksOpen && (
            <CardContent className="space-y-5 text-sm text-muted-foreground pt-0">

              <p>
                Pick one or more topics, set how fast messages are sent (Hz) and for how long,
                then hit <strong className="text-foreground">Run Simulation</strong>.
                The simulator reads your system's topology from the graph and instantly tells you
                how much traffic each topic and broker will carry — no guesswork needed.
              </p>

              <div className="grid sm:grid-cols-3 gap-4">
                <div className="rounded-lg border bg-background/60 p-4 space-y-1.5">
                  <div className="font-semibold text-foreground">① Messages in</div>
                  <p className="text-xs">
                    Every publisher on a topic sends at the frequency you set.
                    <br /><br />
                    <strong className="text-foreground">2 publishers × 10 Hz = 20 msg/s</strong> arriving at the broker.
                  </p>
                </div>
                <div className="rounded-lg border bg-background/60 p-4 space-y-1.5">
                  <div className="font-semibold text-foreground">② Fan-out</div>
                  <p className="text-xs">
                    The broker delivers a copy to <em>each</em> subscriber — so the outbound traffic multiplies.
                    <br /><br />
                    <strong className="text-foreground">20 msg/s × 5 subscribers = 100 msg/s</strong> leaving the broker.
                  </p>
                </div>
                <div className="rounded-lg border bg-background/60 p-4 space-y-1.5">
                  <div className="font-semibold text-foreground">③ Bandwidth</div>
                  <p className="text-xs">
                    Multiply message rate by the message size stored on each topic node.
                    <br /><br />
                    <strong className="text-foreground">120 msg/s × 1 KB = 120 KB/s</strong> total for that topic.
                  </p>
                </div>
              </div>

              <div className="space-y-2">
                <h4 className="font-semibold text-foreground">Good to know</h4>
                <ul className="space-y-1.5 list-disc list-inside text-xs">
                  <li>Each topic can have its own <strong className="text-foreground">Hz</strong> and <strong className="text-foreground">duration</strong> — just select it and edit the fields inline.</li>
                  <li>Message size comes from the topic node in the graph. The global fallback is used only when a topic has none set.</li>
                  <li>Broker load is the sum across all topics it routes — a busy broker shows up immediately.</li>
                  <li>Bandwidth colours are <strong className="text-green-600 dark:text-green-400">green → yellow → orange → red</strong> relative to the busiest topic/broker in your results.</li>
                  <li>No network simulation happens — results are calculated instantly from the topology.</li>
                </ul>
              </div>

            </CardContent>
          )}
        </Card>

        {/* ── Configuration Panel ─────────────────────────────────── */}
        <div className="space-y-6">

          {/* Topic Selection */}
          <Card>
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="flex items-center gap-2">
                    <MessageSquare className="h-5 w-5" />
                    Topic Selection
                    {selectedTopicIds.length > 0 && (
                      <Badge className="ml-1">{selectedTopicIds.length} selected</Badge>
                    )}
                  </CardTitle>
                  <CardDescription className="mt-1">
                    Pick one or more pub/sub topics to include in the simulation.
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2 shrink-0">
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

              {topicsLoading ? (
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

                    {/* QoS filter pills */}
                    <div className="flex items-center gap-1 flex-wrap">
                      {["all", ...qosOptions].map(opt => (
                        <button
                          key={opt}
                          onClick={() => setTopicQosFilter(opt)}
                          className={`px-2.5 py-1 rounded-full text-xs font-medium border transition-colors ${
                            topicQosFilter === opt
                              ? "bg-primary text-primary-foreground border-primary"
                              : "bg-background text-muted-foreground border-border hover:border-primary/50"
                          }`}
                        >
                          {opt === "all" ? "All QoS" : opt}
                        </button>
                      ))}
                    </div>

                    {/* Sort selector */}
                    <div className="flex items-center gap-1">
                      <ArrowUpDown className="h-3.5 w-3.5 text-muted-foreground" />
                      <select
                        value={topicSort}
                        onChange={e => setTopicSort(e.target.value as any)}
                        className="h-8 text-xs rounded-md border border-input bg-background px-2 pr-6 focus:outline-none focus:ring-1 focus:ring-ring"
                      >
                        <option value="weight">Sort: Weight</option>
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
                      {selectedTopicIds.length > 0 && (
                        <>
                          <span>·</span>
                          <button
                            className="hover:text-foreground underline-offset-2 hover:underline"
                            onClick={() => {
                              const reset: Record<string, TopicParams> = {}
                              for (const id of selectedTopicIds) reset[id] = { frequency_hz: frequencyHz, duration_sec: durationSec }
                              setTopicParams(reset)
                            }}
                          >
                            Reset params
                          </button>
                        </>
                      )}
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
                          const weightPct = (topic.weight / maxWeight) * 100
                          const p = topicParams[topic.id] ?? { frequency_hz: frequencyHz, duration_sec: durationSec }
                          const isDefault = p.frequency_hz === frequencyHz && p.duration_sec === durationSec

                          return (
                            <div
                              key={topic.id}
                              className={`flex items-center gap-3 px-3 transition-colors ${selected ? "bg-primary/5 dark:bg-primary/10 py-2" : "py-2.5 hover:bg-muted/60"}`}
                            >
                              {/* Checkbox — always clickable to toggle */}
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

                              {/* Weight bar */}
                              <div className="w-1 self-stretch rounded-full bg-muted overflow-hidden shrink-0">
                                <div
                                  className="w-full rounded-full bg-blue-400 dark:bg-blue-500 transition-all"
                                  style={{ height: `${weightPct}%`, marginTop: `${100 - weightPct}%` }}
                                />
                              </div>

                              {/* Name + meta — click to toggle */}
                              <button
                                onClick={() => toggleTopic(topic.id)}
                                className="flex-1 min-w-0 text-left"
                              >
                                <div className={`text-sm font-medium truncate ${selected ? "text-primary" : ""}`}>
                                  {topic.name}
                                </div>
                                <div className="flex items-center gap-2 mt-0.5">
                                  <span className="text-xs text-muted-foreground">
                                    {topic.publisher_count} pub · {topic.subscriber_count} sub{topic.size > 0 ? ` · ${topic.size}B` : ""}
                                  </span>
                                  {topic.broker_names.length > 0 && (
                                    <span className="text-xs text-muted-foreground hidden sm:inline">
                                      · {topic.broker_names.slice(0, 2).join(", ")}
                                      {topic.broker_names.length > 2 && ` +${topic.broker_names.length - 2}`}
                                    </span>
                                  )}
                                </div>
                              </button>

                              {/* Per-topic param inputs (selected only) */}
                              {selected ? (
                                <div className="flex items-center gap-1.5 shrink-0">
                                  <div className="flex items-center gap-1">
                                    <span className="text-xs text-muted-foreground">Hz</span>
                                    <Input
                                      type="number"
                                      min={0.001}
                                      step={1}
                                      value={p.frequency_hz}
                                      onChange={e => setTopicParams(prev => ({
                                        ...prev,
                                        [topic.id]: { ...p, frequency_hz: parseFloat(e.target.value) || frequencyHz },
                                      }))}
                                      className="h-6 w-16 text-xs px-1.5"
                                    />
                                  </div>
                                  <div className="flex items-center gap-1">
                                    <span className="text-xs text-muted-foreground">s</span>
                                    <Input
                                      type="number"
                                      min={1}
                                      step={10}
                                      value={p.duration_sec}
                                      onChange={e => setTopicParams(prev => ({
                                        ...prev,
                                        [topic.id]: { ...p, duration_sec: parseFloat(e.target.value) || durationSec },
                                      }))}
                                      className="h-6 w-16 text-xs px-1.5"
                                    />
                                  </div>
                                  {!isDefault && (
                                    <button
                                      title="Reset to defaults"
                                      className="text-muted-foreground hover:text-foreground"
                                      onClick={() => setTopicParams(prev => ({
                                        ...prev,
                                        [topic.id]: { frequency_hz: frequencyHz, duration_sec: durationSec },
                                      }))}
                                    >
                                      <X className="h-3.5 w-3.5" />
                                    </button>
                                  )}
                                </div>
                              ) : (
                                /* Badges (unselected) */
                                <div className="flex items-center gap-1.5 shrink-0">
                                  {topic.qos_reliability && (
                                    <Badge
                                      variant="outline"
                                      className={`text-xs ${topic.qos_reliability === "RELIABLE" ? "border-green-500/50 text-green-600 dark:text-green-400" : "border-amber-500/50 text-amber-600 dark:text-amber-400"}`}
                                    >
                                      {topic.qos_reliability}
                                    </Badge>
                                  )}
                                  <span className="text-xs font-mono text-muted-foreground w-12 text-right">
                                    {topic.weight.toFixed(3)}
                                  </span>
                                </div>
                              )}
                            </div>
                          )
                        })}
                      </div>
                    )}
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </div>

        {/* ── Save / Load Configurations ───────────────────────────── */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <Save className="h-5 w-5" />
                  Saved Configurations
                </CardTitle>
                <CardDescription>Save the current setup for quick re-use later.</CardDescription>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setConfigsOpen(v => !v)}
              >
                <FolderOpen className="h-4 w-4 mr-2" />
                {configsOpen ? "Hide" : `Show (${savedConfigs.length})`}
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Save row */}
            <div className="flex gap-2">
              <Input
                placeholder="Configuration name (optional)"
                value={newConfigName}
                onChange={e => setNewConfigName(e.target.value)}
                className="flex-1"
              />
              <Button
                variant="secondary"
                onClick={saveConfig}
                disabled={selectedTopicIds.length === 0}
              >
                <Save className="h-4 w-4 mr-2" />
                Save current
              </Button>
            </div>

            {/* Saved list */}
            {configsOpen && (
              savedConfigs.length === 0 ? (
                <p className="text-sm text-muted-foreground text-center py-4">No saved configurations yet.</p>
              ) : (
                <div className="space-y-2">
                  {savedConfigs.map(cfg => (
                    <div
                      key={cfg.id}
                      className="flex items-center justify-between rounded-lg border p-3 hover:bg-muted/50 transition-colors"
                    >
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-sm truncate">{cfg.name}</div>
                        <div className="text-xs text-muted-foreground">
                          {cfg.topic_ids.length} topics · {cfg.frequency_hz} Hz · {cfg.duration_sec}s · {cfg.message_size_bytes}B
                          <span className="ml-2 opacity-60">{new Date(cfg.created_at).toLocaleDateString()}</span>
                        </div>
                      </div>
                      <div className="flex gap-1 ml-3 shrink-0">
                        <Button variant="ghost" size="sm" onClick={() => loadConfig(cfg)}>
                          <FolderOpen className="h-4 w-4" />
                        </Button>
                        <Button variant="ghost" size="sm" className="text-destructive hover:text-destructive" onClick={() => deleteConfig(cfg.id)}>
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              )
            )}
          </CardContent>
        </Card>

        {/* ── Results ─────────────────────────────────────────────── */}
        {result && (
          <div className="space-y-6">
            {/* Summary cards */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center gap-2 text-muted-foreground mb-1">
                    <Network className="h-4 w-4" />
                    <span className="text-xs">Total bandwidth</span>
                  </div>
                  <div className="text-2xl font-bold">{formatBytes(result.summary.total_network_bps)}</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    {result.summary.total_network_mbps.toFixed(4)} MB/s
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center gap-2 text-muted-foreground mb-1">
                    <Activity className="h-4 w-4" />
                    <span className="text-xs">Messages published</span>
                  </div>
                  <div className="text-2xl font-bold">{formatNumber(result.summary.total_msgs_published)}</div>
                  <div className="text-xs text-muted-foreground mt-1">over {result.summary.duration_sec}s</div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center gap-2 text-muted-foreground mb-1">
                    <CheckCircle2 className="h-4 w-4" />
                    <span className="text-xs">Messages delivered</span>
                  </div>
                  <div className="text-2xl font-bold">{formatNumber(result.summary.total_msgs_delivered)}</div>
                  <div className="text-xs text-muted-foreground mt-1">fan-out total</div>
                </CardContent>
              </Card>

              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center gap-2 text-muted-foreground mb-1">
                    <Server className="h-4 w-4" />
                    <span className="text-xs">Brokers involved</span>
                  </div>
                  <div className="text-2xl font-bold">{result.summary.brokers_involved}</div>
                  <div className="text-xs text-muted-foreground mt-1">
                    {result.summary.topics_found} / {result.summary.selected_topics} topics
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Detail tabs */}
            <Tabs defaultValue="topics">
              <div className="flex items-center justify-between mb-4">
                <TabsList>
                  <TabsTrigger value="topics">
                    <MessageSquare className="h-4 w-4 mr-2" />
                    Per Topic ({result.per_topic.length})
                  </TabsTrigger>
                  <TabsTrigger value="brokers">
                    <Server className="h-4 w-4 mr-2" />
                    Broker Usage ({result.broker_usage.length})
                  </TabsTrigger>
                  <TabsTrigger value="overview">
                    <BarChart3 className="h-4 w-4 mr-2" />
                    Overview
                  </TabsTrigger>
                </TabsList>
                <Button variant="ghost" size="sm" className="text-muted-foreground" onClick={clearResult}>
                  <Trash2 className="h-4 w-4 mr-1" />
                  Clear
                </Button>
              </div>

              {/* Per-topic tab */}
              <TabsContent value="topics">
                <Card>
                  <CardContent className="p-0">
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead>
                          <tr className="border-b bg-muted/50">
                            <th className="text-left p-3 font-medium">Topic</th>
                            <th className="text-right p-3 font-medium" title="Number of publishers writing to this topic">Pubs</th>
                            <th className="text-right p-3 font-medium" title="Number of subscribers reading from this topic">Subs</th>
                            <th className="text-right p-3 font-medium" title="Effective frequency used for this topic (Hz)">Hz</th>
                            <th className="text-right p-3 font-medium" title="Inbound: publisher_count × frequency_hz">In (msg/s)</th>
                            <th className="text-right p-3 font-medium" title="Outbound fan-out: msgs_in/s × subscriber_count">Out (msg/s)</th>
                            <th className="text-right p-3 font-medium" title="Total bandwidth: (in + out) × message_size_bytes">Bandwidth</th>
                            <th className="text-left p-3 font-medium" title="Brokers that route this topic">Brokers</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.per_topic.map((t, i) => {
                            const maxBps = Math.max(...result.per_topic.map(x => x.bandwidth_total_bps), 1)
                            return (
                              <tr key={t.topic_id} className={i % 2 === 0 ? "bg-background" : "bg-muted/20"}>
                                <td className="p-3">
                                  <div className="font-medium">{t.topic_name}</div>
                                  <div className="text-xs text-muted-foreground">{t.topic_id}</div>
                                </td>
                                <td className="p-3 text-right">{t.publisher_count}</td>
                                <td className="p-3 text-right">{t.subscriber_count}</td>
                                <td className="p-3 text-right font-mono text-xs">{t.frequency_hz}</td>
                                <td className="p-3 text-right font-mono">{t.msgs_published_per_sec.toFixed(1)}</td>
                                <td className="p-3 text-right font-mono">{t.msgs_delivered_per_sec.toFixed(1)}</td>
                                <td className={`p-3 text-right font-mono font-medium ${getBandwidthColor(t.bandwidth_total_bps, maxBps)}`}>
                                  {formatBytes(t.bandwidth_total_bps)}
                                </td>
                                <td className="p-3">
                                  <div className="flex flex-wrap gap-1">
                                    {t.broker_names.length === 0
                                      ? <span className="text-muted-foreground text-xs">—</span>
                                      : t.broker_names.map(b => (
                                        <Badge key={b} variant="outline" className="text-xs">{b}</Badge>
                                      ))
                                    }
                                  </div>
                                </td>
                              </tr>
                            )
                          })}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              {/* Broker usage tab */}
              <TabsContent value="brokers">
                {result.broker_usage.length === 0 ? (
                  <Card>
                    <CardContent className="py-12 text-center text-muted-foreground">
                      No broker routing data found for the selected topics.
                    </CardContent>
                  </Card>
                ) : (
                  <div className="space-y-4">
                    <p className="text-sm text-muted-foreground">
                      Each broker's load is the sum of all topics it routes.
                      Inbound = messages arriving from publishers; Outbound = fan-out deliveries to subscribers.
                      A topic routed by multiple brokers is counted in full for each one.
                    </p>
                    <div className="grid gap-4 md:grid-cols-2">
                    {result.broker_usage.map(broker => {
                      const bwRatio = broker.bandwidth_bps / maxBrokerBps
                      return (
                        <Card key={broker.broker_id}>
                          <CardHeader className="pb-2">
                            <div className="flex items-center justify-between">
                              <CardTitle className="text-base flex items-center gap-2">
                                <Server className="h-4 w-4" />
                                {broker.broker_name}
                              </CardTitle>
                              <Badge variant="outline">{broker.topics_routed.length} topic{broker.topics_routed.length !== 1 ? "s" : ""}</Badge>
                            </div>
                          </CardHeader>
                          <CardContent className="space-y-3">
                            {/* Bandwidth bar */}
                            <div>
                              <div className="flex justify-between text-xs text-muted-foreground mb-1">
                                <span>Bandwidth</span>
                                <span className={`font-medium ${getBandwidthColor(broker.bandwidth_bps, maxBrokerBps)}`}>
                                  {formatBytes(broker.bandwidth_bps)}
                                </span>
                              </div>
                              <div className="h-2 bg-muted rounded-full overflow-hidden">
                                <div
                                  className={`h-full rounded-full transition-all ${bwRatio > 0.8 ? "bg-red-500" : bwRatio > 0.5 ? "bg-orange-500" : bwRatio > 0.2 ? "bg-yellow-500" : "bg-green-500"}`}
                                  style={{ width: `${Math.max(2, bwRatio * 100).toFixed(1)}%` }}
                                />
                              </div>
                            </div>

                            <div className="grid grid-cols-2 gap-3 text-sm">
                              <div>
                                <div className="text-xs text-muted-foreground">Inbound</div>
                                <div className="font-mono font-medium">{broker.msgs_inbound_per_sec.toFixed(1)} msg/s</div>
                              </div>
                              <div>
                                <div className="text-xs text-muted-foreground">Outbound</div>
                                <div className="font-mono font-medium">{broker.msgs_outbound_per_sec.toFixed(1)} msg/s</div>
                              </div>
                              <div>
                                <div className="text-xs text-muted-foreground">Total msg/s</div>
                                <div className="font-mono font-medium">{broker.msgs_total_per_sec.toFixed(1)}</div>
                              </div>
                              <div>
                                <div className="text-xs text-muted-foreground">Bandwidth</div>
                                <div className="font-mono font-medium">{broker.bandwidth_mbps.toFixed(4)} MB/s</div>
                              </div>
                            </div>

                            <div>
                              <div className="text-xs text-muted-foreground mb-1">Routed topics</div>
                              <div className="flex flex-wrap gap-1">
                                {broker.topics_routed.map(tid => {
                                  const t = topicById[tid]
                                  return (
                                    <Badge key={tid} variant="secondary" className="text-xs">
                                      {t ? t.name : tid}
                                    </Badge>
                                  )
                                })}
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      )
                    })}
                  </div>
                  </div>
                )}
              </TabsContent>

              {/* Overview tab */}
              <TabsContent value="overview">
                <Card>
                  <CardHeader>
                    <CardTitle>Simulation Summary</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <dl className="grid grid-cols-2 sm:grid-cols-3 gap-x-6 gap-y-4 text-sm">
                      {[
                        { label: "Topics selected", value: result.summary.selected_topics },
                        { label: "Topics found in graph", value: result.summary.topics_found },
                        { label: "Frequency", value: `${result.summary.frequency_hz} Hz` },
                        { label: "Duration", value: `${result.summary.duration_sec}s` },
                        { label: "Message size", value: `${result.summary.message_size_bytes} B` },
                        { label: "Brokers involved", value: result.summary.brokers_involved },
                        { label: "Total published", value: formatNumber(result.summary.total_msgs_published) + " msgs" },
                        { label: "Total delivered (fan-out)", value: formatNumber(result.summary.total_msgs_delivered) + " msgs" },
                        { label: "Total bandwidth", value: formatBytes(result.summary.total_network_bps) },
                        { label: "Peak topic bandwidth", value: formatBytes(result.summary.peak_topic_bps) },
                        { label: "Total (kbps)", value: result.summary.total_network_kbps.toFixed(2) + " KB/s" },
                        { label: "Total (mbps)", value: result.summary.total_network_mbps.toFixed(4) + " MB/s" },
                      ].map(item => (
                        <div key={item.label}>
                          <dt className="text-muted-foreground">{item.label}</dt>
                          <dd className="font-semibold">{item.value}</dd>
                        </div>
                      ))}
                    </dl>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
        )}
      </div>
    </AppLayout>
  )
}
