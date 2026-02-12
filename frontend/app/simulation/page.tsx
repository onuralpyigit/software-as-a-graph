"use client"

import React, { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command"
import {
  Activity,
  AlertTriangle,
  Zap,
  Database,
  TrendingDown,
  TrendingUp,
  Loader2,
  Play,
  AlertCircle,
  CheckCircle2,
  Clock,
  Gauge,
  Network,
  Server,
  MessageSquare,
  ShieldAlert,
  GitBranch,
  BarChart3,
  FileSpreadsheet,
  Info,
  Hash,
  Layers,
  XCircle,
  Check,
  ChevronsUpDown
} from "lucide-react"
import { useConnection } from "@/lib/stores/connection-store"
import { simulationClient } from "@/lib/api/simulation-client"
import { apiClient } from "@/lib/api/client"

// ============================================================================
// Types
// ============================================================================

interface EventMetrics {
  messages_published: number
  messages_delivered: number
  messages_dropped: number
  delivery_rate_percent: number
  drop_rate_percent: number
  avg_latency_ms: number
  min_latency_ms: number
  max_latency_ms: number
  p50_latency_ms: number
  p99_latency_ms: number
  throughput_per_sec: number
}

interface EventResult {
  source_app: string
  scenario: string
  duration?: number  // Legacy field name
  duration_sec?: number  // API returns this field name
  metrics: EventMetrics
  affected_topics: string[]
  brokers_used: string[]
  reached_subscribers: string[]
  drop_reasons: Record<string, number>
  component_impacts: Record<string, number>
}

interface FailureImpact {
  composite_impact: number
  reachability: {
    initial_paths: number
    remaining_paths: number
    loss_percent: number
  }
  infrastructure: {
    initial_components: number
    failed_components: number
    fragmentation_percent: number
  }
  throughput: {
    loss_percent: number
  }
  affected: {
    topics: number
    subscribers: number
    publishers: number
  }
  cascade: {
    count: number
    depth: number
    by_type: Record<string, number>
  }
}

interface FailureResult {
  target_id: string
  target_type: string
  scenario: string
  impact: FailureImpact
  cascaded_failures: string[]
  layer_impacts: Record<string, number>
}

interface LayerMetrics {
  layer: string
  event_metrics: {
    throughput: number
    delivery_rate_percent: number
    drop_rate_percent: number
    avg_latency_ms: number
  }
  failure_metrics: {
    avg_reachability_loss_percent: number
    avg_fragmentation_percent: number
    avg_throughput_loss_percent: number
    max_impact: number
  }
  criticality: {
    total_components: number
    critical: number
    high: number
    medium: number
    spof_count: number
  }
}

interface SimulationReport {
  timestamp: string
  graph_summary: Record<string, any>
  layer_metrics: Record<string, LayerMetrics>
  top_critical: Array<{
    id: string
    type: string
    level: string
    scores: {
      event_impact: number
      failure_impact: number
      combined_impact: number
    }
    metrics: {
      cascade_count: number
      message_throughput: number
      reachability_loss_percent: number
    }
  }>
  recommendations: string[]
}

// ============================================================================
// Helper Functions
// ============================================================================

function getCriticalityColor(level: string): string {
  const colors: Record<string, string> = {
    critical: "text-red-500",
    high: "text-orange-500",
    medium: "text-yellow-500",
    low: "text-blue-500",
    minimal: "text-gray-500",
  }
  return colors[level.toLowerCase()] || "text-gray-500"
}

function getCriticalityBadgeVariant(level: string): "default" | "secondary" | "destructive" | "outline" {
  const variants: Record<string, any> = {
    critical: "destructive",
    high: "destructive",
    medium: "default",
    low: "secondary",
    minimal: "outline",
  }
  return variants[level.toLowerCase()] || "outline"
}

function formatDropReason(reason: string): string {
  return reason
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ')
}

function getImpactColor(impact: number): string {
  if (impact > 0.5) return "text-red-500"
  if (impact > 0.3) return "text-orange-500"
  if (impact > 0.1) return "text-yellow-500"
  return "text-green-500"
}

// ============================================================================
// Main Page Component
// ============================================================================

export default function SimulationPage() {
  const router = useRouter()
  const { status } = useConnection()

  // UI State
  const [simulationMode, setSimulationMode] = useState<'event' | 'failure' | 'exhaustive' | 'report'>('event')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [stats, setStats] = useState<any>(null)
  const [statsLoading, setStatsLoading] = useState(true)
  const [components, setComponents] = useState<Array<{ id: string; name: string; type: string }>>([])  
  const [componentsLoading, setComponentsLoading] = useState(false)

  // Event Simulation State
  const [eventSourceApp, setEventSourceApp] = useState<string>("")
  const [eventSourceAppOpen, setEventSourceAppOpen] = useState(false)
  const [eventMessages, setEventMessages] = useState<number>(100)
  const [eventDuration, setEventDuration] = useState<number>(10)
  const [eventResult, setEventResult] = useState<EventResult | null>(null)

  // Failure Simulation State
  const [failureTargetId, setFailureTargetId] = useState<string>("")
  const [failureTargetOpen, setFailureTargetOpen] = useState(false)
  const [failureLayer, setFailureLayer] = useState<string>("system")
  const [failureCascadeProb, setFailureCascadeProb] = useState<number>(1.0)
  const [failureResult, setFailureResult] = useState<FailureResult | null>(null)

  // Exhaustive Analysis State
  const [exhaustiveLayer, setExhaustiveLayer] = useState<string>("system")
  const [exhaustiveResults, setExhaustiveResults] = useState<FailureResult[]>([])
  const [exhaustiveSummary, setExhaustiveSummary] = useState<any>(null)

  // Report State
  const [reportLayers, setReportLayers] = useState<string[]>(["application", "infrastructure", "system"])
  const [report, setReport] = useState<SimulationReport | null>(null)

  // Effect: Check connection and fetch stats
  useEffect(() => {
    if (status !== 'connected') {
      setError("Please connect to Neo4j database first")
      setStats(null)
      setStatsLoading(false)
    } else {
      setError(null)
      fetchStats()
    }
  }, [status])

  // Load results from localStorage on mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const savedEventResult = localStorage.getItem('simulation_eventResult')
      const savedFailureResult = localStorage.getItem('simulation_failureResult')
      const savedExhaustiveResults = localStorage.getItem('simulation_exhaustiveResults')
      const savedExhaustiveSummary = localStorage.getItem('simulation_exhaustiveSummary')
      const savedReport = localStorage.getItem('simulation_report')
      
      if (savedEventResult) setEventResult(JSON.parse(savedEventResult))
      if (savedFailureResult) setFailureResult(JSON.parse(savedFailureResult))
      if (savedExhaustiveResults) setExhaustiveResults(JSON.parse(savedExhaustiveResults))
      if (savedExhaustiveSummary) setExhaustiveSummary(JSON.parse(savedExhaustiveSummary))
      if (savedReport) setReport(JSON.parse(savedReport))
    }
  }, [])

  // Save results to localStorage when they change
  useEffect(() => {
    if (typeof window !== 'undefined' && eventResult) {
      localStorage.setItem('simulation_eventResult', JSON.stringify(eventResult))
    }
  }, [eventResult])

  useEffect(() => {
    if (typeof window !== 'undefined' && failureResult) {
      localStorage.setItem('simulation_failureResult', JSON.stringify(failureResult))
    }
  }, [failureResult])

  useEffect(() => {
    if (typeof window !== 'undefined' && exhaustiveResults && exhaustiveResults.length > 0) {
      localStorage.setItem('simulation_exhaustiveResults', JSON.stringify(exhaustiveResults))
    }
  }, [exhaustiveResults])

  useEffect(() => {
    if (typeof window !== 'undefined' && exhaustiveSummary) {
      localStorage.setItem('simulation_exhaustiveSummary', JSON.stringify(exhaustiveSummary))
    }
  }, [exhaustiveSummary])

  useEffect(() => {
    if (typeof window !== 'undefined' && report) {
      localStorage.setItem('simulation_report', JSON.stringify(report))
    }
  }, [report])

  const fetchStats = async () => {
    setStatsLoading(true)
    try {
      const data = await apiClient.getGraphStats()
      setStats(data)
      // Also fetch components for dropdowns
      fetchComponents()
    } catch (err: any) {
      console.error("Failed to fetch stats:", err)
    } finally {
      setStatsLoading(false)
    }
  }

  const fetchComponents = async () => {
    setComponentsLoading(true)
    try {
      const data = await apiClient.getGraphData()
      // Extract unique components with id, name, and type
      const componentsList = data.nodes.map(node => ({
        id: node.id,
        name: node.label || node.id,
        type: node.type
      }))
      setComponents(componentsList)
    } catch (err: any) {
      console.error("Failed to fetch components:", err)
    } finally {
      setComponentsLoading(false)
    }
  }

  // Helper function to get component name from ID
  const getComponentName = (id: string): string => {
    const comp = components.find(c => c.id === id)
    return comp ? comp.name : id
  }

  // ============================================================================
  // Event Simulation Handlers
  // ============================================================================

  const handleEventSimulation = async () => {
    if (!eventSourceApp.trim()) {
      setError("Please enter a source application ID")
      return
    }

    setLoading(true)
    setError(null)
    setEventResult(null)

    try {
      const result = await simulationClient.runEventSimulation({
        source_app: eventSourceApp,
        num_messages: eventMessages,
        duration: eventDuration,
      })

      setEventResult(result)
    } catch (err: any) {
      setError(err.message || "Event simulation failed")
      console.error("Event simulation error:", err)
    } finally {
      setLoading(false)
    }
  }

  // ============================================================================
  // Failure Simulation Handlers
  // ============================================================================

  const handleFailureSimulation = async () => {
    if (!failureTargetId.trim()) {
      setError("Please enter a target component ID")
      return
    }

    setLoading(true)
    setError(null)
    setFailureResult(null)

    try {
      const result = await simulationClient.runFailureSimulation({
        target_id: failureTargetId,
        layer: failureLayer,
        cascade_probability: failureCascadeProb,
      })

      setFailureResult({ ...result, layer: failureLayer })
    } catch (err: any) {
      setError(err.message || "Failure simulation failed")
      console.error("Failure simulation error:", err)
    } finally {
      setLoading(false)
    }
  }

  // ============================================================================
  // Exhaustive Analysis Handlers
  // ============================================================================

  const handleExhaustiveAnalysis = async () => {
    setLoading(true)
    setError(null)
    setExhaustiveResults([])
    setExhaustiveSummary(null)

    try {
      const { results, summary } = await simulationClient.runExhaustiveSimulation({
        layer: exhaustiveLayer,
        cascade_probability: 1.0,
      })

      console.log('Exhaustive summary received:', summary)
      setExhaustiveResults(results)
      setExhaustiveSummary(summary)
    } catch (err: any) {
      setError(err.message || "Exhaustive analysis failed")
      console.error("Exhaustive analysis error:", err)
    } finally {
      setLoading(false)
    }
  }

  // ============================================================================
  // Report Generation Handlers
  // ============================================================================

  const handleGenerateReport = async () => {
    setLoading(true)
    setError(null)
    setReport(null)

    try {
      const result = await simulationClient.generateReport({
        layers: reportLayers,
      })

      setReport(result)
    } catch (err: any) {
      setError(err.message || "Report generation failed")
      console.error("Report generation error:", err)
    } finally {
      setLoading(false)
    }
  }

  const clearEventResult = () => {
    localStorage.removeItem('simulation_eventResult')
    setEventResult(null)
  }

  const clearFailureResult = () => {
    localStorage.removeItem('simulation_failureResult')
    setFailureResult(null)
  }

  const clearExhaustiveResults = () => {
    localStorage.removeItem('simulation_exhaustiveResults')
    localStorage.removeItem('simulation_exhaustiveSummary')
    setExhaustiveResults([])
    setExhaustiveSummary(null)
  }

  const clearReport = () => {
    localStorage.removeItem('simulation_report')
    setReport(null)
  }

  // ============================================================================
  // Render: Connecting
  // ============================================================================

  if (status === 'connecting') {
    return (
      <AppLayout title="Simulation" description="Run simulations and analyze system behavior">
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" text="Connecting to database..." />
        </div>
      </AppLayout>
    )
  }

  // ============================================================================
  // Render: No Connection
  // ============================================================================

  if (status !== 'connected') {
    return (
      <AppLayout title="Simulation" description="Run simulations and analyze system behavior">
        <NoConnectionInfo description="Connect to your Neo4j database to run simulations" />
      </AppLayout>
    )
  }

  // ============================================================================
  // Render: Loading Stats
  // ============================================================================

  if (statsLoading || !stats) {
    return (
      <AppLayout title="Simulation" description="Run simulations and analyze system behavior">
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" text="Loading graph statistics..." />
        </div>
      </AppLayout>
    )
  }

  // ============================================================================
  // Render: Empty Database
  // ============================================================================

  if (stats.total_nodes === 0) {
    return (
      <AppLayout title="Simulation" description="Run simulations and analyze system behavior">
        <div className="w-full">
          <Card className="border-2 border-purple-500/50 dark:border-purple-500/50 bg-white/95 dark:bg-black/95 backdrop-blur-md shadow-2xl shadow-purple-500/20 hover:shadow-purple-500/30 hover:border-purple-500/70 transition-all duration-300 overflow-hidden">
            {/* Decorative top border */}
            <div className="h-1 w-full bg-gradient-to-r from-purple-500 via-pink-500 to-purple-500" />

            <CardHeader className="pb-6 pt-8 px-8">
              <div className="flex flex-col sm:flex-row items-start sm:items-center gap-5">
                {/* Icon with animated gradient */}
                <div className="relative group">
                  <div className="absolute inset-0 bg-gradient-to-br from-purple-500 to-purple-600 rounded-2xl blur-xl opacity-30 group-hover:opacity-50 transition-opacity duration-300" />
                  <div className="relative rounded-2xl bg-gradient-to-br from-purple-500/20 to-purple-600/20 dark:from-purple-500/30 dark:to-purple-600/30 p-4 ring-1 ring-purple-500/30 group-hover:ring-purple-500/50 transition-all duration-300">
                    <Database className="h-8 w-8 text-purple-600 dark:text-purple-400 group-hover:scale-110 transition-transform duration-300" />
                  </div>
                </div>

                {/* Title section */}
                <div className="flex-1 space-y-1.5">
                  <CardTitle className="text-2xl font-bold tracking-tight">Empty Database</CardTitle>
                  <CardDescription className="text-base text-muted-foreground">
                    No graph data available for simulation
                  </CardDescription>
                </div>
              </div>
            </CardHeader>

            <CardContent className="px-8 pb-8 space-y-6">\n              {/* Information box with steps */}
              <div className="rounded-2xl bg-gradient-to-br from-muted/40 via-muted/20 to-muted/10 border border-border/40 p-6 space-y-5">
                <div className="flex items-center gap-2.5">
                  <div className="rounded-xl bg-gradient-to-br from-blue-500/20 to-blue-600/20 dark:from-blue-500/30 dark:to-blue-600/30 p-2.5 ring-1 ring-blue-500/30">
                    <Info className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                  </div>
                  <h3 className="font-semibold text-base text-foreground">How to Populate Your Database</h3>
                </div>

                {/* Step-by-step list */}
                <div className="space-y-3 pl-1">
                  <div className="flex items-start gap-3.5">
                    <div className="flex-shrink-0 w-7 h-7 rounded-full bg-gradient-to-br from-purple-500/20 to-purple-600/20 flex items-center justify-center ring-1 ring-purple-500/30">
                      <span className="text-xs font-bold text-purple-600 dark:text-purple-400">1</span>
                    </div>
                    <p className="text-sm text-muted-foreground leading-relaxed pt-0.5">
                      Navigate to the Data Management page
                    </p>
                  </div>
                  <div className="flex items-start gap-3.5">
                    <div className="flex-shrink-0 w-7 h-7 rounded-full bg-gradient-to-br from-purple-500/20 to-purple-600/20 flex items-center justify-center ring-1 ring-purple-500/30">
                      <span className="text-xs font-bold text-purple-600 dark:text-purple-400">2</span>
                    </div>
                    <p className="text-sm text-muted-foreground leading-relaxed pt-0.5">
                      Generate and import graph data into the database
                    </p>
                  </div>
                </div>
              </div>

              {/* CTA Button */}
              <div className="pt-2">
                <Button
                  onClick={() => router.push('/data')}
                  size="lg"
                  className="w-full bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white shadow-lg hover:shadow-xl transition-all duration-300"
                >
                  <Database className="mr-2 h-4 w-4" />
                  Populate Database
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </AppLayout>
    )
  }

  // ============================================================================
  // Render: Main Content
  // ============================================================================

  return (
    <AppLayout title="Simulation" description="Run simulations and analyze system behavior">
      <div className="space-y-6">
        {/* Error Display */}
        {error && (
          <Card className="border-red-500 bg-red-50 dark:bg-red-950/10">
            <CardContent className="flex items-center gap-2 pt-6">
              <AlertCircle className="h-5 w-5 text-red-500" />
              <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
            </CardContent>
          </Card>
        )}

        {/* Configuration Section */}
        <div className="space-y-4">
          {/* Header Section */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-gradient-to-br from-blue-600 via-purple-600 to-pink-600 p-3 shadow-lg">
                <Zap className="h-6 w-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold">Simulation Configuration</h2>
                <p className="text-sm text-muted-foreground">Select your simulation type and run system behavior analysis</p>
              </div>
            </div>
            {/* Run Button - Changes based on simulation mode */}
            {simulationMode === 'event' && (
              <Button
                onClick={handleEventSimulation}
                disabled={loading || !eventSourceApp.trim()}
                size="lg"
                className="min-w-[200px] h-12 bg-gradient-to-r from-purple-500 via-pink-500 to-purple-600 hover:from-purple-600 hover:via-pink-600 hover:to-purple-700 shadow-lg hover:shadow-xl transition-all text-base font-semibold"
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                    Running...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-5 w-5" />
                    Run Event Simulation
                  </>
                )}
              </Button>
            )}
            {simulationMode === 'failure' && (
              <Button
                onClick={handleFailureSimulation}
                disabled={loading || !failureTargetId.trim()}
                size="lg"
                className="min-w-[200px] h-12 bg-gradient-to-r from-red-500 via-rose-500 to-pink-500 hover:from-red-600 hover:via-rose-600 hover:to-pink-600 shadow-lg hover:shadow-xl transition-all text-base font-semibold"
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                    Running...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-5 w-5" />
                    Run Failure Simulation
                  </>
                )}
              </Button>
            )}
            {simulationMode === 'exhaustive' && (
              <Button
                onClick={handleExhaustiveAnalysis}
                disabled={loading}
                size="lg"
                className="min-w-[200px] h-12 bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 hover:from-blue-600 hover:via-indigo-600 hover:to-purple-600 shadow-lg hover:shadow-xl transition-all text-base font-semibold"
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                    Running...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-5 w-5" />
                    Run Exhaustive Analysis
                  </>
                )}
              </Button>
            )}
            {simulationMode === 'report' && (
              <Button
                onClick={handleGenerateReport}
                disabled={loading || reportLayers.length === 0}
                size="lg"
                className="min-w-[200px] h-12 bg-gradient-to-r from-cyan-500 via-teal-500 to-emerald-500 hover:from-cyan-600 hover:via-teal-600 hover:to-emerald-600 shadow-lg hover:shadow-xl transition-all text-base font-semibold"
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-5 w-5" />
                    Generate Report
                  </>
                )}
              </Button>
            )}
          </div>

          {/* Simulation Mode Selection - Card Grid */}
          <div className="grid gap-6 md:grid-cols-4">
            {/* Event Simulation Card */}
            <Card
              className={`group relative cursor-pointer transition-all duration-300 ease-in-out overflow-hidden ${
                simulationMode === 'event'
                  ? 'border-0 shadow-2xl shadow-purple-500/25 scale-[1.02]'
                  : 'border-0 hover:shadow-xl hover:shadow-purple-500/25 hover:scale-[1.01]'
              }`}
              onClick={() => setSimulationMode('event')}
            >
              {/* Gradient border */}
              <div className={`absolute inset-0 rounded-lg p-[2px] transition-opacity duration-300 ${
                simulationMode === 'event'
                  ? 'bg-gradient-to-r from-purple-500 via-pink-500 to-purple-600 opacity-100'
                  : 'bg-gradient-to-br from-slate-200 via-slate-300 to-slate-200 dark:from-slate-700 dark:via-slate-800 dark:to-slate-700 opacity-100 group-hover:from-purple-500 group-hover:via-pink-500 group-hover:to-purple-600'
              }`}>
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              
              {/* Background gradient overlay */}
              <div className={`absolute inset-[2px] rounded-lg transition-opacity duration-300 ${
                simulationMode === 'event'
                  ? 'bg-gradient-to-br from-purple-500/15 via-pink-500/10 to-purple-500/5 opacity-100'
                  : 'bg-gradient-to-br from-purple-500/5 via-pink-500/3 to-transparent opacity-0 group-hover:opacity-100'
              }`} />
              
              <CardContent className="relative p-7">
                <div className="space-y-5">
                  {/* Radio button and Icon Section */}
                  <div className="flex items-start gap-4">
                    {/* Radio indicator */}
                    <div className={`relative flex-shrink-0 w-5 h-5 rounded-full border-2 transition-all duration-300 mt-1 ${
                      simulationMode === 'event'
                        ? 'border-purple-500 dark:border-purple-400'
                        : 'border-slate-300 dark:border-slate-600 group-hover:border-purple-400'
                    }`}>
                      {simulationMode === 'event' && (
                        <div className="absolute inset-0.5 rounded-full bg-purple-500 dark:bg-purple-400 animate-in zoom-in duration-200" />
                      )}
                    </div>
                    
                    {/* Icon */}
                    <div className={`relative rounded-2xl p-3.5 transition-all duration-300 ${
                      simulationMode === 'event'
                        ? 'bg-gradient-to-br from-purple-500 to-pink-600 shadow-lg shadow-purple-500/30'
                        : 'bg-gradient-to-br from-purple-100 to-pink-50 dark:from-purple-900/50 dark:to-pink-900/30 group-hover:scale-105'
                    }`}>
                      <MessageSquare className={`h-6 w-6 transition-all duration-300 ${
                        simulationMode === 'event'
                          ? 'text-white'
                          : 'text-purple-600 dark:text-purple-400'
                      }`} />
                      {simulationMode === 'event' && (
                        <div className="absolute inset-0 rounded-2xl bg-purple-500 animate-ping opacity-20" />
                      )}
                    </div>
                  </div>
                  
                  {/* Content Section */}
                  <div className="space-y-2.5 pl-9">
                    <div className="flex items-center gap-2">
                      <h3 className={`text-lg font-bold tracking-tight transition-colors duration-200 ${
                        simulationMode === 'event'
                          ? 'text-purple-700 dark:text-purple-300'
                          : 'text-foreground group-hover:text-purple-700 dark:group-hover:text-purple-300'
                      }`}>
                        Event
                      </h3>
                      {simulationMode === 'event' && (
                        <Badge className="bg-purple-500/10 text-purple-700 dark:text-purple-300 border-purple-500/20 text-xs">
                          Selected
                        </Badge>
                      )}
                    </div>
                    <p className="text-sm leading-relaxed text-muted-foreground/90">
                      Simulate message flow through the pub-sub system
                    </p>
                    {!simulationMode || simulationMode !== 'event' ? (
                      <p className="text-xs text-muted-foreground/60 flex items-center gap-1 mt-2">
                        <Info className="h-3 w-3" />
                        Click to select this type
                      </p>
                    ) : null}
                  </div>
                  
                  {/* Selection indicator bar */}
                  {simulationMode === 'event' && (
                    <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-purple-500 to-pink-600" />
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Failure Simulation Card */}
            <Card
              className={`group relative cursor-pointer transition-all duration-300 ease-in-out overflow-hidden ${
                simulationMode === 'failure'
                  ? 'border-0 shadow-2xl shadow-red-500/25 scale-[1.02]'
                  : 'border-0 hover:shadow-xl hover:shadow-red-500/25 hover:scale-[1.01]'
              }`}
              onClick={() => setSimulationMode('failure')}
            >
              {/* Gradient border */}
              <div className={`absolute inset-0 rounded-lg p-[2px] transition-opacity duration-300 ${
                simulationMode === 'failure'
                  ? 'bg-gradient-to-r from-red-500 via-rose-500 to-pink-500 opacity-100'
                  : 'bg-gradient-to-br from-slate-200 via-slate-300 to-slate-200 dark:from-slate-700 dark:via-slate-800 dark:to-slate-700 opacity-100 group-hover:from-red-500 group-hover:via-rose-500 group-hover:to-pink-500'
              }`}>
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              
              {/* Background gradient overlay */}
              <div className={`absolute inset-[2px] rounded-lg transition-opacity duration-300 ${
                simulationMode === 'failure'
                  ? 'bg-gradient-to-br from-red-500/15 via-rose-500/10 to-pink-500/5 opacity-100'
                  : 'bg-gradient-to-br from-red-500/5 via-rose-500/3 to-transparent opacity-0 group-hover:opacity-100'
              }`} />
              
              <CardContent className="relative p-7">
                <div className="space-y-5">
                  {/* Radio button and Icon Section */}
                  <div className="flex items-start gap-4">
                    {/* Radio indicator */}
                    <div className={`relative flex-shrink-0 w-5 h-5 rounded-full border-2 transition-all duration-300 mt-1 ${
                      simulationMode === 'failure'
                        ? 'border-red-500 dark:border-red-400'
                        : 'border-slate-300 dark:border-slate-600 group-hover:border-red-400'
                    }`}>
                      {simulationMode === 'failure' && (
                        <div className="absolute inset-0.5 rounded-full bg-red-500 dark:bg-red-400 animate-in zoom-in duration-200" />
                      )}
                    </div>
                    
                    {/* Icon */}
                    <div className={`relative rounded-2xl p-3.5 transition-all duration-300 ${
                      simulationMode === 'failure'
                        ? 'bg-gradient-to-br from-red-500 to-rose-600 shadow-lg shadow-red-500/30'
                        : 'bg-gradient-to-br from-red-100 to-rose-50 dark:from-red-900/50 dark:to-rose-900/30 group-hover:scale-105'
                    }`}>
                      <ShieldAlert className={`h-6 w-6 transition-all duration-300 ${
                        simulationMode === 'failure'
                          ? 'text-white'
                          : 'text-red-600 dark:text-red-400'
                      }`} />
                      {simulationMode === 'failure' && (
                        <div className="absolute inset-0 rounded-2xl bg-red-500 animate-ping opacity-20" />
                      )}
                    </div>
                  </div>
                  
                  {/* Content Section */}
                  <div className="space-y-2.5 pl-9">
                    <div className="flex items-center gap-2">
                      <h3 className={`text-lg font-bold tracking-tight transition-colors duration-200 ${
                        simulationMode === 'failure'
                          ? 'text-red-700 dark:text-red-300'
                          : 'text-foreground group-hover:text-red-700 dark:group-hover:text-red-300'
                      }`}>
                        Failure
                      </h3>
                      {simulationMode === 'failure' && (
                        <Badge className="bg-red-500/10 text-red-700 dark:text-red-300 border-red-500/20 text-xs">
                          Selected
                        </Badge>
                      )}
                    </div>
                    <p className="text-sm leading-relaxed text-muted-foreground/90">
                      Analyze impact of component failures
                    </p>
                    {!simulationMode || simulationMode !== 'failure' ? (
                      <p className="text-xs text-muted-foreground/60 flex items-center gap-1 mt-2">
                        <Info className="h-3 w-3" />
                        Click to select this type
                      </p>
                    ) : null}
                  </div>
                  
                  {/* Selection indicator bar */}
                  {simulationMode === 'failure' && (
                    <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-red-500 to-rose-600" />
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Exhaustive Simulation Card */}
            <Card
              className={`group relative cursor-pointer transition-all duration-300 ease-in-out overflow-hidden ${
                simulationMode === 'exhaustive'
                  ? 'border-0 shadow-2xl shadow-blue-500/25 scale-[1.02]'
                  : 'border-0 hover:shadow-xl hover:shadow-blue-500/25 hover:scale-[1.01]'
              }`}
              onClick={() => setSimulationMode('exhaustive')}
            >
              {/* Gradient border */}
              <div className={`absolute inset-0 rounded-lg p-[2px] transition-opacity duration-300 ${
                simulationMode === 'exhaustive'
                  ? 'bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 opacity-100'
                  : 'bg-gradient-to-br from-slate-200 via-slate-300 to-slate-200 dark:from-slate-700 dark:via-slate-800 dark:to-slate-700 opacity-100 group-hover:from-blue-500 group-hover:via-indigo-500 group-hover:to-purple-500'
              }`}>
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              
              {/* Background gradient overlay */}
              <div className={`absolute inset-[2px] rounded-lg transition-opacity duration-300 ${
                simulationMode === 'exhaustive'
                  ? 'bg-gradient-to-br from-blue-500/15 via-indigo-500/10 to-purple-500/5 opacity-100'
                  : 'bg-gradient-to-br from-blue-500/5 via-indigo-500/3 to-transparent opacity-0 group-hover:opacity-100'
              }`} />
              
              <CardContent className="relative p-7">
                <div className="space-y-5">
                  {/* Radio button and Icon Section */}
                  <div className="flex items-start gap-4">
                    {/* Radio indicator */}
                    <div className={`relative flex-shrink-0 w-5 h-5 rounded-full border-2 transition-all duration-300 mt-1 ${
                      simulationMode === 'exhaustive'
                        ? 'border-blue-500 dark:border-blue-400'
                        : 'border-slate-300 dark:border-slate-600 group-hover:border-blue-400'
                    }`}>
                      {simulationMode === 'exhaustive' && (
                        <div className="absolute inset-0.5 rounded-full bg-blue-500 dark:bg-blue-400 animate-in zoom-in duration-200" />
                      )}
                    </div>
                    
                    {/* Icon */}
                    <div className={`relative rounded-2xl p-3.5 transition-all duration-300 ${
                      simulationMode === 'exhaustive'
                        ? 'bg-gradient-to-br from-blue-500 to-indigo-600 shadow-lg shadow-blue-500/30'
                        : 'bg-gradient-to-br from-blue-100 to-indigo-50 dark:from-blue-900/50 dark:to-indigo-900/30 group-hover:scale-105'
                    }`}>
                      <BarChart3 className={`h-6 w-6 transition-all duration-300 ${
                        simulationMode === 'exhaustive'
                          ? 'text-white'
                          : 'text-blue-600 dark:text-blue-400'
                      }`} />
                      {simulationMode === 'exhaustive' && (
                        <div className="absolute inset-0 rounded-2xl bg-blue-500 animate-ping opacity-20" />
                      )}
                    </div>
                  </div>
                  
                  {/* Content Section */}
                  <div className="space-y-2.5 pl-9">
                    <div className="flex items-center gap-2">
                      <h3 className={`text-lg font-bold tracking-tight transition-colors duration-200 ${
                        simulationMode === 'exhaustive'
                          ? 'text-blue-700 dark:text-blue-300'
                          : 'text-foreground group-hover:text-blue-700 dark:group-hover:text-blue-300'
                      }`}>
                        Exhaustive
                      </h3>
                      {simulationMode === 'exhaustive' && (
                        <Badge className="bg-blue-500/10 text-blue-700 dark:text-blue-300 border-blue-500/20 text-xs">
                          Selected
                        </Badge>
                      )}
                    </div>
                    <p className="text-sm leading-relaxed text-muted-foreground/90">
                      Comprehensive failure analysis for a layer
                    </p>
                    {!simulationMode || simulationMode !== 'exhaustive' ? (
                      <p className="text-xs text-muted-foreground/60 flex items-center gap-1 mt-2">
                        <Info className="h-3 w-3" />
                        Click to select this type
                      </p>
                    ) : null}
                  </div>
                  
                  {/* Selection indicator bar */}
                  {simulationMode === 'exhaustive' && (
                    <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-blue-500 to-indigo-600" />
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Report Card */}
            <Card
              className={`group relative cursor-pointer transition-all duration-300 ease-in-out overflow-hidden ${
                simulationMode === 'report'
                  ? 'border-0 shadow-2xl shadow-cyan-500/25 scale-[1.02]'
                  : 'border-0 hover:shadow-xl hover:shadow-cyan-500/25 hover:scale-[1.01]'
              }`}
              onClick={() => setSimulationMode('report')}
            >
              {/* Gradient border */}
              <div className={`absolute inset-0 rounded-lg p-[2px] transition-opacity duration-300 ${
                simulationMode === 'report'
                  ? 'bg-gradient-to-r from-cyan-500 via-teal-500 to-emerald-500 opacity-100'
                  : 'bg-gradient-to-br from-slate-200 via-slate-300 to-slate-200 dark:from-slate-700 dark:via-slate-800 dark:to-slate-700 opacity-100 group-hover:from-cyan-500 group-hover:via-teal-500 group-hover:to-emerald-500'
              }`}>
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              
              {/* Background gradient overlay */}
              <div className={`absolute inset-[2px] rounded-lg transition-opacity duration-300 ${
                simulationMode === 'report'
                  ? 'bg-gradient-to-br from-cyan-500/15 via-teal-500/10 to-emerald-500/5 opacity-100'
                  : 'bg-gradient-to-br from-cyan-500/5 via-teal-500/3 to-transparent opacity-0 group-hover:opacity-100'
              }`} />
              
              <CardContent className="relative p-7">
                <div className="space-y-5">
                  {/* Radio button and Icon Section */}
                  <div className="flex items-start gap-4">
                    {/* Radio indicator */}
                    <div className={`relative flex-shrink-0 w-5 h-5 rounded-full border-2 transition-all duration-300 mt-1 ${
                      simulationMode === 'report'
                        ? 'border-cyan-500 dark:border-cyan-400'
                        : 'border-slate-300 dark:border-slate-600 group-hover:border-cyan-400'
                    }`}>
                      {simulationMode === 'report' && (
                        <div className="absolute inset-0.5 rounded-full bg-cyan-500 dark:bg-cyan-400 animate-in zoom-in duration-200" />
                      )}
                    </div>
                    
                    {/* Icon */}
                    <div className={`relative rounded-2xl p-3.5 transition-all duration-300 ${
                      simulationMode === 'report'
                        ? 'bg-gradient-to-br from-cyan-500 to-teal-600 shadow-lg shadow-cyan-500/30'
                        : 'bg-gradient-to-br from-cyan-100 to-teal-50 dark:from-cyan-900/50 dark:to-teal-900/30 group-hover:scale-105'
                    }`}>
                      <FileSpreadsheet className={`h-6 w-6 transition-all duration-300 ${
                        simulationMode === 'report'
                          ? 'text-white'
                          : 'text-cyan-600 dark:text-cyan-400'
                      }`} />
                      {simulationMode === 'report' && (
                        <div className="absolute inset-0 rounded-2xl bg-cyan-500 animate-ping opacity-20" />
                      )}
                    </div>
                  </div>
                  
                  {/* Content Section */}
                  <div className="space-y-2.5 pl-9">
                    <div className="flex items-center gap-2">
                      <h3 className={`text-lg font-bold tracking-tight transition-colors duration-200 ${
                        simulationMode === 'report'
                          ? 'text-cyan-700 dark:text-cyan-300'
                          : 'text-foreground group-hover:text-cyan-700 dark:group-hover:text-cyan-300'
                      }`}>
                        Report
                      </h3>
                      {simulationMode === 'report' && (
                        <Badge className="bg-cyan-500/10 text-cyan-700 dark:text-cyan-300 border-cyan-500/20 text-xs">
                          Selected
                        </Badge>
                      )}
                    </div>
                    <p className="text-sm leading-relaxed text-muted-foreground/90">
                      Generate comprehensive simulation report
                    </p>
                    {!simulationMode || simulationMode !== 'report' ? (
                      <p className="text-xs text-muted-foreground/60 flex items-center gap-1 mt-2">
                        <Info className="h-3 w-3" />
                        Click to select this type
                      </p>
                    ) : null}
                  </div>
                  
                  {/* Selection indicator bar */}
                  {simulationMode === 'report' && (
                    <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-cyan-500 to-teal-600" />
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Configuration Panel for Event Simulation */}
          {simulationMode === 'event' && (
            <div className="space-y-3">
              <Label className="text-base font-semibold flex items-center gap-2">
                <MessageSquare className="h-4 w-4" />
                Event Simulation Parameters
              </Label>
              <p className="text-sm text-muted-foreground/80">
                Configure message flow simulation settings
              </p>
              <div className="space-y-4 pt-2">
                <div className="grid gap-4 md:grid-cols-3">
                  <div className="space-y-2">
                    <Label htmlFor="event-source">Source Application</Label>
                    <Popover open={eventSourceAppOpen} onOpenChange={setEventSourceAppOpen}>
                      <PopoverTrigger asChild>
                        <Button
                          id="event-source"
                          variant="outline"
                          role="combobox"
                          aria-expanded={eventSourceAppOpen}
                          disabled={loading || componentsLoading}
                          className="w-full justify-between font-normal"
                        >
                          {eventSourceApp ? (
                            <div className="flex items-center justify-between gap-2 w-full">
                              <span className="truncate">
                                {components.find(c => c.id === eventSourceApp)?.name || eventSourceApp}
                              </span>
                              <span className="text-xs text-muted-foreground truncate">
                                {eventSourceApp}
                              </span>
                            </div>
                          ) : (
                            <span className="text-muted-foreground">
                              {componentsLoading ? "Loading..." : "Select an application"}
                            </span>
                          )}
                          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                        </Button>
                      </PopoverTrigger>
                      <PopoverContent className="w-[400px] p-0" align="start">
                        <Command>
                          <CommandInput placeholder="Search application..." />
                          <CommandList>
                            <CommandEmpty>No application found.</CommandEmpty>
                            <CommandGroup>
                              {components
                                .filter(c => c.type === 'Application')
                                .map((comp) => (
                                  <CommandItem
                                    key={comp.id}
                                    value={`${comp.name} ${comp.id}`}
                                    onSelect={() => {
                                      setEventSourceApp(comp.id)
                                      setEventSourceAppOpen(false)
                                    }}
                                  >
                                    <Check
                                      className={`mr-2 h-4 w-4 ${
                                        eventSourceApp === comp.id ? "opacity-100" : "opacity-0"
                                      }`}
                                    />
                                    <div className="flex items-center justify-between gap-2 flex-1">
                                      <span>{comp.name}</span>
                                      <span className="text-xs text-muted-foreground">{comp.id}</span>
                                    </div>
                                  </CommandItem>
                                ))}
                            </CommandGroup>
                          </CommandList>
                        </Command>
                      </PopoverContent>
                    </Popover>
                    <p className="text-xs text-muted-foreground">
                      The application that will publish messages to the system
                    </p>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="event-messages">Number of Messages</Label>
                    <Input
                      id="event-messages"
                      type="number"
                      min="1"
                      max="10000"
                      value={eventMessages}
                      onChange={(e) => setEventMessages(parseInt(e.target.value) || 100)}
                      disabled={loading}
                    />
                    <p className="text-xs text-muted-foreground">
                      Total messages to simulate (1-10,000)
                    </p>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="event-duration">Duration (seconds)</Label>
                    <Input
                      id="event-duration"
                      type="number"
                      min="0.1"
                      step="0.1"
                      value={eventDuration}
                      onChange={(e) => setEventDuration(parseFloat(e.target.value) || 10)}
                      disabled={loading}
                    />
                    <p className="text-xs text-muted-foreground">
                      Time period over which messages are sent
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

        {/* Event Results - Show only in event mode */}
        {simulationMode === 'event' && eventResult && (
          <>
            {/* Results Header */}
            <Card className="relative overflow-hidden border-0 shadow-xl">
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-purple-500 via-pink-500 to-purple-600">
                <div className="w-full h-full bg-gradient-to-r from-purple-600 via-pink-600 to-purple-700 rounded-lg" />
              </div>
              <CardContent className="relative p-6 text-white">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-4">
                    <div className="rounded-xl bg-white/20 p-3 backdrop-blur-sm">
                      <CheckCircle2 className="h-8 w-8" />
                    </div>
                    <div>
                      <h2 className="text-2xl font-bold mb-1">Event Simulation Complete</h2>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => router.push(`/explorer?node=${encodeURIComponent(eventResult.source_app)}`)}
                          className="text-base font-semibold hover:underline transition-all text-white/95 cursor-pointer"
                        >
                          {getComponentName(eventResult.source_app)}
                        </button>
                        <span className="text-white/70">·</span>
                        <span className="text-white/90">{eventResult.scenario}</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-start gap-6">
                    <div className="hidden md:flex items-center gap-4">
                      <div className="text-right">
                        <div className="text-3xl font-bold">{eventResult.metrics.messages_published || 0}</div>
                        <div className="text-sm text-white/80">Messages</div>
                      </div>
                      <div className="text-right">
                        <div className="text-3xl font-bold">{eventResult.reached_subscribers?.length || 0}</div>
                        <div className="text-sm text-white/80">Subscribers</div>
                      </div>
                      <div className="text-right">
                        <div className="text-3xl font-bold">{parseFloat((eventResult.metrics.delivery_rate_percent || 0).toFixed(1))}%</div>
                        <div className="text-sm text-white/80">Delivered</div>
                      </div>
                    </div>
                    <Button
                      onClick={clearEventResult}
                      variant="outline"
                      size="sm"
                      className="flex items-center gap-2 bg-white/10 hover:bg-white/20 border-white/30 text-white"
                    >
                      <XCircle className="h-4 w-4" />
                      Clear Results
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Key Metrics */}
            <div>
              <div className="mb-4 flex items-center gap-3">
                <div className="rounded-xl bg-gradient-to-br from-purple-600 via-pink-600 to-purple-700 p-2.5 shadow-lg">
                  <Gauge className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-bold">Message Flow Metrics</h2>
                  <p className="text-sm text-muted-foreground">
                    Performance and delivery statistics from {getComponentName(eventResult.source_app)}
                  </p>
                </div>
              </div>
              <div className="grid gap-4 md:grid-cols-3">
                <Card className="group relative overflow-hidden border-0 hover:shadow-xl hover:shadow-green-500/25 hover:scale-[1.02] transition-all duration-300">
                  {/* Gradient border */}
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  {/* Subtle background glow */}
                  <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-green-500/30 via-green-500/15 to-green-500/5" />
                  
                  <CardContent className="relative p-6">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="rounded-xl bg-gradient-to-br from-green-600 via-emerald-600 to-teal-600 p-2.5 shadow-lg">
                        <CheckCircle2 className="h-5 w-5 text-white" />
                      </div>
                      <h3 className="text-sm font-medium">Delivery Success</h3>
                    </div>
                    <div className="text-3xl font-bold text-green-600">
                      {parseFloat((eventResult.metrics.delivery_rate_percent || 0).toFixed(1))}%
                    </div>
                    <Progress value={eventResult.metrics.delivery_rate_percent || 0} className="h-2.5 mt-2 bg-slate-100 dark:bg-slate-800" />
                    <p className="text-xs text-muted-foreground mt-1">
                      {eventResult.metrics.messages_delivered || 0} / {eventResult.metrics.messages_published || 0} messages
                    </p>
                  </CardContent>
                </Card>

                <Card className="group relative overflow-hidden border-0 hover:shadow-xl hover:shadow-blue-500/25 hover:scale-[1.02] transition-all duration-300">
                  {/* Gradient border */}
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  {/* Subtle background glow */}
                  <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-blue-500/30 via-blue-500/15 to-blue-500/5" />
                  
                  <CardContent className="relative p-6">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="rounded-xl bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-600 p-2.5 shadow-lg">
                        <Gauge className="h-5 w-5 text-white" />
                      </div>
                      <h3 className="text-sm font-medium">Throughput</h3>
                    </div>
                    <div className="text-3xl font-bold text-blue-600">
                      {parseFloat((eventResult.metrics.throughput_per_sec || 0).toFixed(1))}
                    </div>
                    <div className="text-sm text-muted-foreground mt-2">messages/second</div>
                  </CardContent>
                </Card>

                <Card className="group relative overflow-hidden border-0 hover:shadow-xl hover:shadow-purple-500/25 hover:scale-[1.02] transition-all duration-300">
                  {/* Gradient border */}
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-purple-500 via-pink-500 to-purple-600">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  {/* Subtle background glow */}
                  <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-purple-500/30 via-purple-500/15 to-purple-500/5" />
                  
                  <CardContent className="relative p-6">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="rounded-xl bg-gradient-to-br from-purple-600 via-pink-600 to-purple-700 p-2.5 shadow-lg">
                        <Clock className="h-5 w-5 text-white" />
                      </div>
                      <h3 className="text-sm font-medium">Avg Latency</h3>
                    </div>
                    <div className="text-3xl font-bold text-purple-600">
                      {parseFloat((eventResult.metrics.avg_latency_ms || 0).toFixed(2))}
                    </div>
                    <div className="text-sm text-muted-foreground mt-2">milliseconds</div>
                  </CardContent>
                </Card>
              </div>
            </div>

            {/* Detailed Metrics */}
            <Card className="border-0 shadow-xl relative overflow-hidden">
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-purple-500 via-pink-500 to-purple-600">
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-purple-500/10 via-purple-500/5 to-transparent" />
              
              <CardHeader className="relative">
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Detailed Statistics
                </CardTitle>
                <CardDescription>
                  Comprehensive delivery and latency metrics
                </CardDescription>
              </CardHeader>
              <CardContent className="relative">
                <div className="grid gap-6 md:grid-cols-2">
                  {/* Message Flow */}
                  <div className="space-y-3">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="rounded-lg bg-gradient-to-br from-emerald-500 to-emerald-600 p-2">
                        <TrendingUp className="h-4 w-4 text-white" />
                      </div>
                      <h3 className="font-semibold text-base">Message Flow</h3>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between p-3 rounded-lg bg-muted/30 hover:bg-muted/40 transition-colors">
                        <span className="text-sm text-muted-foreground">Published</span>
                        <span className="font-semibold text-base">{eventResult.metrics.messages_published || 0}</span>
                      </div>
                      <div className="flex items-center justify-between p-3 rounded-lg bg-emerald-50 dark:bg-emerald-950/30 hover:bg-emerald-100 dark:hover:bg-emerald-950/40 transition-colors">
                        <span className="text-sm text-emerald-700 dark:text-emerald-300">Delivered</span>
                        <span className="font-semibold text-base text-emerald-600 dark:text-emerald-400">{eventResult.metrics.messages_delivered || 0}</span>
                      </div>
                      <div className="flex items-center justify-between p-3 rounded-lg bg-red-50 dark:bg-red-950/30 hover:bg-red-100 dark:hover:bg-red-950/40 transition-colors">
                        <span className="text-sm text-red-700 dark:text-red-300">Dropped</span>
                        <span className="font-semibold text-base text-red-600 dark:text-red-400">{eventResult.metrics.messages_dropped || 0}</span>
                      </div>
                      <div className="flex items-center justify-between p-3 rounded-lg border bg-card">
                        <span className="text-sm font-medium">Drop Rate</span>
                        <span className="font-bold text-base text-red-600 dark:text-red-400">{parseFloat((eventResult.metrics.drop_rate_percent || 0).toFixed(1))}%</span>
                      </div>
                    </div>
                  </div>

                  {/* Latency Distribution */}
                  <div className="space-y-3">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="rounded-lg bg-gradient-to-br from-blue-500 to-blue-600 p-2">
                        <Clock className="h-4 w-4 text-white" />
                      </div>
                      <h3 className="font-semibold text-base">Latency Distribution</h3>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between p-3 rounded-lg bg-muted/30 hover:bg-muted/40 transition-colors">
                        <span className="text-sm text-muted-foreground">Minimum</span>
                        <span className="font-mono text-sm font-semibold">{(eventResult.metrics.min_latency_ms || 0).toFixed(2)} ms</span>
                      </div>
                      <div className="flex items-center justify-between p-3 rounded-lg bg-blue-50 dark:bg-blue-950/30 hover:bg-blue-100 dark:hover:bg-blue-950/40 transition-colors">
                        <span className="text-sm text-blue-700 dark:text-blue-300">P50 (Median)</span>
                        <span className="font-mono text-sm font-semibold text-blue-600 dark:text-blue-400">{(eventResult.metrics.p50_latency_ms || 0).toFixed(2)} ms</span>
                      </div>
                      <div className="flex items-center justify-between p-3 rounded-lg bg-purple-50 dark:bg-purple-950/30 hover:bg-purple-100 dark:hover:bg-purple-950/40 transition-colors">
                        <span className="text-sm text-purple-700 dark:text-purple-300">P99</span>
                        <span className="font-mono text-sm font-semibold text-purple-600 dark:text-purple-400">{(eventResult.metrics.p99_latency_ms || 0).toFixed(2)} ms</span>
                      </div>
                      <div className="flex items-center justify-between p-3 rounded-lg bg-muted/30 hover:bg-muted/40 transition-colors">
                        <span className="text-sm text-muted-foreground">Maximum</span>
                        <span className="font-mono text-sm font-semibold">{(eventResult.metrics.max_latency_ms || 0).toFixed(2)} ms</span>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* System Topology */}
            <div>
              <div className="mb-4 flex items-center gap-3">
                <div className="rounded-xl bg-gradient-to-br from-cyan-600 via-teal-600 to-emerald-700 p-2.5 shadow-lg">
                  <Network className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-bold">System Topology</h2>
                  <p className="text-sm text-muted-foreground">
                    Components involved in message flow
                  </p>
                </div>
              </div>
              
              <div className="grid gap-4 md:grid-cols-3">
                {/* Affected Topics */}
                <Card className="border-0 shadow-lg relative overflow-hidden">
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-cyan-500 to-cyan-600">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  <CardHeader className="relative pb-3">
                    <CardTitle className="flex items-center gap-2 text-base">
                      <div className="rounded-lg bg-gradient-to-br from-cyan-500 to-cyan-600 p-2">
                        <Hash className="h-4 w-4 text-white" />
                      </div>
                      Topics
                      <Badge variant="secondary" className="ml-auto">{eventResult.affected_topics?.length || 0}</Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="relative">
                    {eventResult.affected_topics && eventResult.affected_topics.length > 0 ? (
                      <div className="space-y-2">
                        {eventResult.affected_topics.map((topic) => (
                          <div key={topic} className="flex items-center gap-2 p-3 rounded-lg bg-muted/30 hover:bg-muted/50 transition-colors cursor-pointer" onClick={() => router.push(`/explorer?node=${encodeURIComponent(topic)}`)}>
                            <div className="flex-1 min-w-0">
                              <div className="text-sm font-medium truncate hover:underline transition-all">{getComponentName(topic)}</div>
                              <div className="text-xs text-muted-foreground truncate">{topic}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground">None</p>
                    )}
                  </CardContent>
                </Card>

                {/* Brokers Used */}
                <Card className="border-0 shadow-lg relative overflow-hidden">
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-teal-500 to-teal-600">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  <CardHeader className="relative pb-3">
                    <CardTitle className="flex items-center gap-2 text-base">
                      <div className="rounded-lg bg-gradient-to-br from-teal-500 to-teal-600 p-2">
                        <Server className="h-4 w-4 text-white" />
                      </div>
                      Brokers
                      <Badge variant="secondary" className="ml-auto">{eventResult.brokers_used?.length || 0}</Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="relative">
                    {eventResult.brokers_used && eventResult.brokers_used.length > 0 ? (
                      <div className="space-y-2">
                        {eventResult.brokers_used.map((broker) => (
                          <div key={broker} className="flex items-center gap-2 p-3 rounded-lg bg-muted/30 hover:bg-muted/50 transition-colors cursor-pointer" onClick={() => router.push(`/explorer?node=${encodeURIComponent(broker)}`)}>
                            <div className="flex-1 min-w-0">
                              <div className="text-sm font-medium truncate hover:underline transition-all">{getComponentName(broker)}</div>
                              <div className="text-xs text-muted-foreground truncate">{broker}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground">None</p>
                    )}
                  </CardContent>
                </Card>

                {/* Reached Subscribers */}
                <Card className="border-0 shadow-lg relative overflow-hidden">
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-emerald-500 to-emerald-600">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  <CardHeader className="relative pb-3">
                    <CardTitle className="flex items-center gap-2 text-base">
                      <div className="rounded-lg bg-gradient-to-br from-emerald-500 to-emerald-600 p-2">
                        <Activity className="h-4 w-4 text-white" />
                      </div>
                      Subscribers
                      <Badge variant="secondary" className="ml-auto">{eventResult.reached_subscribers?.length || 0}</Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="relative">
                    {eventResult.reached_subscribers && eventResult.reached_subscribers.length > 0 ? (
                      <div className="space-y-2 max-h-[300px] overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-muted scrollbar-track-transparent">
                        {eventResult.reached_subscribers.map((sub) => (
                          <div key={sub} className="flex items-center gap-2 p-3 rounded-lg bg-emerald-50 dark:bg-emerald-950/30 hover:bg-emerald-100 dark:hover:bg-emerald-950/40 transition-colors cursor-pointer" onClick={() => router.push(`/explorer?node=${encodeURIComponent(sub)}`)}>
                            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-600 dark:text-emerald-400 flex-shrink-0" />
                            <div className="flex-1 min-w-0">
                              <div className="text-sm font-medium truncate hover:underline transition-all">{getComponentName(sub)}</div>
                              <div className="text-xs text-muted-foreground truncate">{sub}</div>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <p className="text-sm text-muted-foreground">None</p>
                    )}
                  </CardContent>
                </Card>
              </div>
              
              {/* Component Impacts */}
              {eventResult.component_impacts && Object.keys(eventResult.component_impacts).length > 0 && (
                <Card className="border-0 shadow-lg relative overflow-hidden mt-4">
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-blue-500 to-blue-600">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  <CardHeader className="relative pb-3">
                    <CardTitle className="flex items-center gap-2 text-base">
                      <div className="rounded-lg bg-gradient-to-br from-blue-500 to-blue-600 p-2">
                        <BarChart3 className="h-4 w-4 text-white" />
                      </div>
                      Component Impact Analysis
                      <Badge variant="secondary" className="ml-auto">Top 5</Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="relative">
                    <div className="space-y-2">
                      {Object.entries(eventResult.component_impacts)
                        .sort(([, a], [, b]) => (b as number) - (a as number))
                        .slice(0, 5)
                        .map(([componentId, impact], index) => {
                          const impactValue = impact as number
                          const impactPercent = parseFloat((impactValue * 100).toFixed(1))
                          return (
                            <div key={componentId} className="group p-3 rounded-lg bg-muted/30 hover:bg-muted/50 transition-all cursor-pointer" onClick={() => router.push(`/explorer?node=${encodeURIComponent(componentId)}`)}>
                              <div className="flex items-center gap-3 mb-2">
                                <div className="flex items-center justify-center w-6 h-6 rounded-full bg-primary/10 text-xs font-bold text-primary">
                                  {index + 1}
                                </div>
                                <div className="flex-1 min-w-0">
                                  <div className="font-medium text-sm truncate group-hover:underline transition-all">{getComponentName(componentId)}</div>
                                  <div className="text-xs text-muted-foreground truncate">{componentId}</div>
                                </div>
                                <div className="text-right">
                                  <div className="text-sm font-bold">{impactPercent}%</div>
                                </div>
                              </div>
                              <Progress value={impactValue * 100} className="h-1.5" />
                            </div>
                          )
                        })}
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>

                {/* Drop Reasons (if any) */}
                {eventResult.drop_reasons && Object.keys(eventResult.drop_reasons).length > 0 && (
                  <Card className="border-0 shadow-xl relative overflow-hidden">
                    <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-red-500 via-rose-500 to-pink-500">
                      <div className="w-full h-full bg-background rounded-lg" />
                    </div>
                    <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-red-500/10 via-red-500/5 to-transparent" />
                    
                    <CardHeader className="relative">
                      <CardTitle className="flex items-center gap-2">
                        <AlertTriangle className="h-5 w-5" />
                        Drop Analysis
                      </CardTitle>
                      <CardDescription>
                        Messages dropped during simulation with breakdown by reason
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="relative">
                      {/* Summary Stats */}
                      <div className="grid gap-6 md:grid-cols-2 mb-6">
                        <div className="space-y-3">
                          <div className="flex items-center gap-2 mb-4">
                            <div className="rounded-lg bg-gradient-to-br from-red-500 to-red-600 p-2">
                              <TrendingDown className="h-4 w-4 text-white" />
                            </div>
                            <h3 className="font-semibold text-base">Drop Summary</h3>
                          </div>
                          <div className="space-y-2">
                            <div className="flex items-center justify-between p-3 rounded-lg bg-red-50 dark:bg-red-950/30 hover:bg-red-100 dark:hover:bg-red-950/40 transition-colors">
                              <span className="text-sm text-red-700 dark:text-red-300">Total Dropped</span>
                              <span className="font-semibold text-base text-red-600 dark:text-red-400">
                                {Object.values(eventResult.drop_reasons).reduce((sum, count) => sum + count, 0)}
                              </span>
                            </div>
                            <div className="flex items-center justify-between p-3 rounded-lg border bg-card">
                              <span className="text-sm font-medium">Drop Rate</span>
                              <span className="font-bold text-base text-red-600 dark:text-red-400">
                                {eventResult.metrics.messages_published > 0 
                                  ? `${parseFloat(((Object.values(eventResult.drop_reasons).reduce((sum, count) => sum + count, 0) / eventResult.metrics.messages_published) * 100).toFixed(1))}%`
                                  : 'N/A'}
                              </span>
                            </div>
                            <div className="flex items-center justify-between p-3 rounded-lg bg-muted/30 hover:bg-muted/40 transition-colors">
                              <span className="text-sm text-muted-foreground">Unique Reasons</span>
                              <span className="font-semibold text-base">{Object.keys(eventResult.drop_reasons).length}</span>
                            </div>
                          </div>
                        </div>

                        <div className="space-y-3">
                          <div className="flex items-center gap-2 mb-4">
                            <div className="rounded-lg bg-gradient-to-br from-rose-500 to-rose-600 p-2">
                              <BarChart3 className="h-4 w-4 text-white" />
                            </div>
                            <h3 className="font-semibold text-base">Breakdown by Reason</h3>
                          </div>
                          <div className="space-y-2 max-h-[300px] overflow-y-auto scrollbar-thin">
                            {Object.entries(eventResult.drop_reasons)
                              .sort(([, a], [, b]) => b - a)
                              .slice(0, 5)
                              .map(([reason, count]) => {
                                const percentage = eventResult.metrics.messages_published > 0 
                                  ? parseFloat(((count / eventResult.metrics.messages_published) * 100).toFixed(1))
                                  : 0;
                                const totalDropped = Object.values(eventResult.drop_reasons).reduce((sum, c) => sum + c, 0);
                                const portionOfDrops = totalDropped > 0
                                  ? parseFloat(((count / totalDropped) * 100).toFixed(1))
                                  : 0;
                                
                                return (
                                  <div key={reason} className="p-3 rounded-lg bg-muted/30 hover:bg-muted/40 transition-colors">
                                    <div className="flex items-center justify-between mb-2">
                                      <span className="text-sm font-medium truncate">{formatDropReason(reason)}</span>
                                      <span className="font-semibold text-sm text-red-600 dark:text-red-400 ml-2">{count}</span>
                                    </div>
                                    <Progress 
                                      value={parseFloat(portionOfDrops)} 
                                      className="h-2 bg-red-500/10"
                                    />
                                    <div className="text-xs text-muted-foreground mt-1">
                                      {percentage}% of total • {portionOfDrops}% of drops
                                    </div>
                                  </div>
                                );
                              })}
                          </div>
                        </div>
                      </div>

                      {/* All Drop Reasons (if more than 5) */}
                      {Object.keys(eventResult.drop_reasons).length > 5 && (
                        <div className="pt-4 border-t">
                          <h4 className="text-sm font-semibold mb-3">All Drop Reasons</h4>
                          <div className="space-y-2">
                            {Object.entries(eventResult.drop_reasons)
                              .sort(([, a], [, b]) => b - a)
                              .slice(5)
                              .map(([reason, count]) => {
                                const percentage = eventResult.metrics.messages_published > 0 
                                  ? parseFloat(((count / eventResult.metrics.messages_published) * 100).toFixed(1))
                                  : 0;
                                
                                return (
                                  <div key={reason} className="flex items-center justify-between p-2 rounded-lg bg-muted/20 hover:bg-muted/30 transition-colors text-sm">
                                    <span className="truncate">{formatDropReason(reason)}</span>
                                    <div className="flex items-center gap-2 ml-2">
                                      <span className="text-xs text-muted-foreground">{percentage}%</span>
                                      <Badge variant="outline" className="text-xs">{count}</Badge>
                                    </div>
                                  </div>
                                );
                              })}
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                )}
            </>
          )}

          {/* Configuration Panel for Failure Simulation */}
          {simulationMode === 'failure' && (
            <div className="space-y-3">
              <Label className="text-base font-semibold flex items-center gap-2">
                <ShieldAlert className="h-4 w-4" />
                Failure Simulation Parameters
              </Label>
              <p className="text-sm text-muted-foreground/80">
                Configure component failure and cascade analysis
              </p>
              <div className="space-y-4 pt-2">
                <div className="grid gap-4 md:grid-cols-3">
                  <div className="space-y-2">
                    <Label htmlFor="failure-target">Target Component</Label>
                    <Popover open={failureTargetOpen} onOpenChange={setFailureTargetOpen}>
                      <PopoverTrigger asChild>
                        <Button
                          id="failure-target"
                          variant="outline"
                          role="combobox"
                          aria-expanded={failureTargetOpen}
                          disabled={loading || componentsLoading}
                          className="w-full justify-between font-normal"
                        >
                          {failureTargetId ? (
                            <div className="flex items-center justify-between gap-2 w-full">
                              <div className="flex items-center gap-2 min-w-0">
                                <Badge variant="outline" className="text-xs shrink-0">
                                  {components.find(c => c.id === failureTargetId)?.type || 'Unknown'}
                                </Badge>
                                <span className="truncate">
                                  {components.find(c => c.id === failureTargetId)?.name || failureTargetId}
                                </span>
                              </div>
                              <span className="text-xs text-muted-foreground truncate shrink-0">
                                {failureTargetId}
                              </span>
                            </div>
                          ) : (
                            <span className="text-muted-foreground">
                              {componentsLoading ? "Loading..." : "Select a component"}
                            </span>
                          )}
                          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                        </Button>
                      </PopoverTrigger>
                      <PopoverContent className="w-[450px] p-0" align="start">
                        <Command>
                          <CommandInput placeholder="Search component..." />
                          <CommandList>
                            <CommandEmpty>No component found.</CommandEmpty>
                            <CommandGroup>
                              {components.map((comp) => (
                                <CommandItem
                                  key={comp.id}
                                  value={`${comp.name} ${comp.id} ${comp.type}`}
                                  onSelect={() => {
                                    setFailureTargetId(comp.id)
                                    setFailureTargetOpen(false)
                                  }}
                                >
                                  <Check
                                    className={`mr-2 h-4 w-4 ${
                                      failureTargetId === comp.id ? "opacity-100" : "opacity-0"
                                    }`}
                                  />
                                  <div className="flex items-center justify-between gap-2 flex-1">
                                    <div className="flex items-center gap-2">
                                      <Badge variant="outline" className="text-xs">{comp.type}</Badge>
                                      <span>{comp.name}</span>
                                    </div>
                                    <span className="text-xs text-muted-foreground">{comp.id}</span>
                                  </div>
                                </CommandItem>
                              ))}
                            </CommandGroup>
                          </CommandList>
                        </Command>
                      </PopoverContent>
                    </Popover>
                    <p className="text-xs text-muted-foreground">Component to simulate failure for</p>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="failure-layer">Analysis Layer</Label>
                    <Select value={failureLayer} onValueChange={setFailureLayer} disabled={loading}>
                      <SelectTrigger id="failure-layer">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="application">Application</SelectItem>
                        <SelectItem value="infrastructure">Infrastructure</SelectItem>
                        <SelectItem value="middleware">Middleware</SelectItem>
                        <SelectItem value="system">System</SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-muted-foreground">Architectural layer to analyze failure impact</p>
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="failure-cascade">Cascade Probability</Label>
                    <Input
                      id="failure-cascade"
                      type="number"
                      min="0"
                      max="1"
                      step="0.1"
                      value={failureCascadeProb}
                      onChange={(e) => setFailureCascadeProb(parseFloat(e.target.value) || 1.0)}
                      disabled={loading}
                    />
                    <p className="text-xs text-muted-foreground">Probability of failure cascading to connected components (0.0-1.0)</p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Configuration Panel for Exhaustive Analysis */}
          {simulationMode === 'exhaustive' && (
            <div className="space-y-3">
              <Label className="text-base font-semibold flex items-center gap-2">
                <BarChart3 className="h-4 w-4" />
                Exhaustive Analysis Parameters
              </Label>
              <p className="text-sm text-muted-foreground/80">
                Analyze failure impact for all components in a layer
              </p>
              <div className="space-y-4 pt-2">
                <div className="space-y-2">
                  <Label htmlFor="exhaustive-layer">Analysis Layer</Label>
                  <Select value={exhaustiveLayer} onValueChange={setExhaustiveLayer} disabled={loading}>
                    <SelectTrigger id="exhaustive-layer">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="application">Application</SelectItem>
                      <SelectItem value="infrastructure">Infrastructure</SelectItem>
                      <SelectItem value="middleware">Middleware</SelectItem>
                      <SelectItem value="system">System</SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    Architectural layer to analyze failure impact across all components
                  </p>
                </div>
              </div>
            </div>
          )}

        {/* Failure Results - Show only in failure mode */}
        {simulationMode === 'failure' && failureResult && (
          <>
            {/* Results Header */}
            <Card className="relative overflow-hidden border-0 shadow-xl">
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-red-500 via-rose-500 to-pink-500">
                <div className="w-full h-full bg-gradient-to-r from-red-600 via-rose-600 to-pink-700 rounded-lg" />
              </div>
              <CardContent className="relative p-6 text-white">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-4">
                    <div className="rounded-xl bg-white/20 p-3 backdrop-blur-sm">
                      <CheckCircle2 className="h-8 w-8" />
                    </div>
                    <div>
                      <h2 className="text-2xl font-bold mb-1">Failure Simulation Complete</h2>
                      <div className="flex items-center gap-2">
                        <button
                          onClick={() => router.push(`/explorer?node=${encodeURIComponent(failureResult.target_id)}`)}
                          className="text-base font-semibold hover:underline transition-all text-white/95 cursor-pointer"
                        >
                          {getComponentName(failureResult.target_id)}
                        </button>
                        <span className="text-white/70">·</span>
                        <span className="text-white/90">{failureResult.target_type}</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-start gap-6">
                    <div className="hidden md:flex items-center gap-4">
                      <div className="text-right">
                        <div className="text-3xl font-bold">{(failureResult.impact.composite_impact || 0).toFixed(3)}</div>
                        <div className="text-sm text-white/80">Impact</div>
                      </div>
                      <div className="text-right">
                        <div className="text-3xl font-bold">{failureResult.cascaded_failures?.length || 0}</div>
                        <div className="text-sm text-white/80">Cascades</div>
                      </div>
                      <div className="text-right">
                        <div className="text-3xl font-bold">{parseFloat((failureResult.impact.reachability?.loss_percent || 0).toFixed(1))}%</div>
                        <div className="text-sm text-white/80">Path Loss</div>
                      </div>
                    </div>
                    <Button
                      onClick={clearFailureResult}
                      variant="outline"
                      size="sm"
                      className="flex items-center gap-2 bg-white/10 hover:bg-white/20 border-white/30 text-white"
                    >
                      <XCircle className="h-4 w-4" />
                      Clear Results
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Impact Metrics */}
            <div>
              <div className="mb-4 flex items-center gap-3">
                <div className="rounded-xl bg-gradient-to-br from-red-600 via-rose-600 to-pink-700 p-2.5 shadow-lg">
                  <AlertTriangle className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-bold">Failure Impact Metrics</h2>
                  <p className="text-sm text-muted-foreground">
                    System-wide impact analysis for {getComponentName(failureResult.target_id)}
                  </p>
                </div>
              </div>
              <div className="grid gap-4 md:grid-cols-3">
                <Card className="group relative overflow-hidden border-0 hover:shadow-xl hover:shadow-red-500/25 hover:scale-[1.02] transition-all duration-300">
                  {/* Gradient border */}
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-red-500 via-rose-500 to-pink-500">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  {/* Subtle background glow */}
                  <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-red-500/30 via-red-500/15 to-red-500/5" />
                  
                  <CardContent className="relative p-6">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="rounded-xl bg-gradient-to-br from-red-600 via-rose-600 to-pink-700 p-2.5 shadow-lg">
                        <AlertTriangle className="h-5 w-5 text-white" />
                      </div>
                      <h3 className="text-sm font-medium">Composite Impact</h3>
                    </div>
                    <div className={`text-3xl font-bold ${getImpactColor(failureResult.impact.composite_impact || 0)}`}>
                      {(failureResult.impact.composite_impact || 0).toFixed(4)}
                    </div>
                    <Progress value={(failureResult.impact.composite_impact || 0) * 100} className="h-2.5 mt-2 bg-slate-100 dark:bg-slate-800" />
                    <p className="text-xs text-muted-foreground mt-1">Overall system impact score</p>
                  </CardContent>
                </Card>

                <Card className="group relative overflow-hidden border-0 hover:shadow-xl hover:shadow-orange-500/25 hover:scale-[1.02] transition-all duration-300">
                  {/* Gradient border */}
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-orange-500 via-amber-500 to-yellow-500">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  {/* Subtle background glow */}
                  <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-orange-500/30 via-orange-500/15 to-orange-500/5" />
                  
                  <CardContent className="relative p-6">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="rounded-xl bg-gradient-to-br from-orange-600 via-amber-600 to-yellow-600 p-2.5 shadow-lg">
                        <Network className="h-5 w-5 text-white" />
                      </div>
                      <h3 className="text-sm font-medium">Reachability Loss</h3>
                    </div>
                    <div className="text-3xl font-bold text-orange-600">
                      {parseFloat((failureResult.impact.reachability?.loss_percent || 0).toFixed(1))}%
                    </div>
                    <Progress value={failureResult.impact.reachability?.loss_percent || 0} className="h-2.5 mt-2 bg-slate-100 dark:bg-slate-800" />
                    <p className="text-xs text-muted-foreground mt-1">
                      {failureResult.impact.reachability?.remaining_paths || 0} / {failureResult.impact.reachability?.initial_paths || 0} paths remaining
                    </p>
                  </CardContent>
                </Card>

                <Card className="group relative overflow-hidden border-0 hover:shadow-xl hover:shadow-purple-500/25 hover:scale-[1.02] transition-all duration-300">
                  {/* Gradient border */}
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-purple-500 via-pink-500 to-purple-600">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  {/* Subtle background glow */}
                  <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-purple-500/30 via-purple-500/15 to-purple-500/5" />
                  
                  <CardContent className="relative p-6">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="rounded-xl bg-gradient-to-br from-purple-600 via-pink-600 to-purple-700 p-2.5 shadow-lg">
                        <GitBranch className="h-5 w-5 text-white" />
                      </div>
                      <h3 className="text-sm font-medium">Cascaded Failures</h3>
                    </div>
                    <div className="text-3xl font-bold text-purple-600">
                      {failureResult.cascaded_failures?.length || 0}
                    </div>
                    <div className="text-sm text-muted-foreground mt-2">components affected</div>
                  </CardContent>
                </Card>
              </div>
            </div>

            {/* Detailed Analysis */}
            <Card className="border-0 shadow-xl relative overflow-hidden">
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-red-500 via-rose-500 to-pink-500">
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-red-500/10 via-red-500/5 to-transparent" />
              
              <CardHeader className="relative">
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Detailed Analysis
                </CardTitle>
                <CardDescription>
                  Infrastructure and throughput impact breakdown
                </CardDescription>
              </CardHeader>
              <CardContent className="relative">
                <div className="grid gap-6 md:grid-cols-2">
                  {/* Throughput Impact */}
                  <div className="space-y-3">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="rounded-lg bg-gradient-to-br from-red-500 to-red-600 p-2">
                        <TrendingDown className="h-4 w-4 text-white" />
                      </div>
                      <h3 className="font-semibold text-base">Throughput Impact</h3>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between p-3 rounded-lg bg-red-50 dark:bg-red-950/30 hover:bg-red-100 dark:hover:bg-red-950/40 transition-colors">
                        <span className="text-sm text-red-700 dark:text-red-300">Loss</span>
                        <span className="font-semibold text-base text-red-600 dark:text-red-400">{parseFloat((failureResult.impact.throughput?.loss_percent || 0).toFixed(1))}%</span>
                      </div>
                      <div className="flex items-center justify-between p-3 rounded-lg bg-muted/30 hover:bg-muted/40 transition-colors">
                        <span className="text-sm text-muted-foreground">Topics Affected</span>
                        <span className="font-semibold text-base">{failureResult.impact.affected?.topics || 0}</span>
                      </div>
                    </div>
                  </div>

                  {/* Affected Components */}
                  <div className="space-y-3">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="rounded-lg bg-gradient-to-br from-blue-500 to-blue-600 p-2">
                        <Activity className="h-4 w-4 text-white" />
                      </div>
                      <h3 className="font-semibold text-base">Affected Components</h3>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between p-3 rounded-lg bg-blue-50 dark:bg-blue-950/30 hover:bg-blue-100 dark:hover:bg-blue-950/40 transition-colors">
                        <span className="text-sm text-blue-700 dark:text-blue-300">Publishers</span>
                        <span className="font-semibold text-base text-blue-600 dark:text-blue-400">{failureResult.impact.affected?.publishers || 0}</span>
                      </div>
                      <div className="flex items-center justify-between p-3 rounded-lg bg-muted/30 hover:bg-muted/40 transition-colors">
                        <span className="text-sm text-muted-foreground">Subscribers</span>
                        <span className="font-semibold text-base">{failureResult.impact.affected?.subscribers || 0}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Cascade Analysis */}
            <div>
              <div className="mb-4 flex items-center gap-3">
                <div className="rounded-xl bg-gradient-to-br from-yellow-600 via-orange-600 to-amber-600 p-2.5 shadow-lg">
                  <GitBranch className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-bold">Cascade Analysis</h2>
                  <p className="text-sm text-muted-foreground">Failure propagation through the system</p>
                </div>
              </div>
              
              <div className="grid gap-4 md:grid-cols-3">
                {/* Cascade Count */}
                <Card className="border-0 shadow-lg relative overflow-hidden">
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-yellow-500 to-amber-600">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  <CardHeader className="relative pb-3">
                    <CardTitle className="flex items-center gap-2 text-base">
                      <div className="rounded-lg bg-gradient-to-br from-yellow-500 to-amber-600 p-2">
                        <Hash className="h-4 w-4 text-white" />
                      </div>
                      Cascade Count
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="relative">
                    <div className="text-3xl font-bold text-orange-600">
                      {failureResult.impact.cascade?.count || 0}
                    </div>
                    <p className="text-xs text-muted-foreground mt-2">
                      Total components affected by cascade
                    </p>
                  </CardContent>
                </Card>

                {/* Cascade Depth */}
                <Card className="border-0 shadow-lg relative overflow-hidden">
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-orange-500 to-orange-600">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  <CardHeader className="relative pb-3">
                    <CardTitle className="flex items-center gap-2 text-base">
                      <div className="rounded-lg bg-gradient-to-br from-orange-500 to-orange-600 p-2">
                        <Layers className="h-4 w-4 text-white" />
                      </div>
                      Cascade Depth
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="relative">
                    <div className="text-3xl font-bold text-yellow-600">
                      {failureResult.impact.cascade?.depth || 0}
                    </div>
                    <p className="text-xs text-muted-foreground mt-2">
                      Maximum propagation depth
                    </p>
                  </CardContent>
                </Card>

                {/* Affected Entities */}
                <Card className="border-0 shadow-lg relative overflow-hidden">
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-amber-500 to-amber-600">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  <CardHeader className="relative pb-3">
                    <CardTitle className="flex items-center gap-2 text-base">
                      <div className="rounded-lg bg-gradient-to-br from-amber-500 to-amber-600 p-2">
                        <Activity className="h-4 w-4 text-white" />
                      </div>
                      Affected Entities
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="relative">
                    <div className="space-y-2">
                      <div className="flex items-center justify-between p-3 rounded-lg bg-muted/30 hover:bg-muted/40 transition-colors">
                        <span className="text-sm text-muted-foreground">Publishers</span>
                        <span className="font-semibold text-base">{failureResult.impact.affected?.publishers || 0}</span>
                      </div>
                      <div className="flex items-center justify-between p-3 rounded-lg bg-muted/30 hover:bg-muted/40 transition-colors">
                        <span className="text-sm text-muted-foreground">Subscribers</span>
                        <span className="font-semibold text-base">{failureResult.impact.affected?.subscribers || 0}</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Cascade by Type */}
              {failureResult.impact.cascade?.by_type && Object.keys(failureResult.impact.cascade.by_type).length > 0 && (
                <Card className="border-0 shadow-lg relative overflow-hidden mt-4">
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-yellow-500 to-amber-600">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  <CardHeader className="relative pb-3">
                    <CardTitle className="flex items-center gap-2 text-base">
                      <div className="rounded-lg bg-gradient-to-br from-yellow-500 to-amber-600 p-2">
                        <BarChart3 className="h-4 w-4 text-white" />
                      </div>
                      Cascade by Type
                      <Badge variant="secondary" className="ml-auto">{Object.keys(failureResult.impact.cascade.by_type).length}</Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="relative">
                    <div className="space-y-2">
                      {Object.entries(failureResult.impact.cascade.by_type)
                        .sort(([, a], [, b]) => (b as number) - (a as number))
                        .map(([type, count]) => (
                          <div key={type} className="flex items-center justify-between p-3 rounded-lg bg-muted/30 hover:bg-muted/50 transition-colors">
                            <span className="text-sm font-medium">{type}</span>
                            <Badge variant="outline">{count}</Badge>
                          </div>
                        ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Cascaded Failures */}
              {failureResult.cascaded_failures && failureResult.cascaded_failures.length > 0 && (
                <Card className="border-0 shadow-lg relative overflow-hidden mt-4">
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-red-500 to-rose-600">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  <CardHeader className="relative pb-3">
                    <CardTitle className="flex items-center gap-2 text-base">
                      <div className="rounded-lg bg-gradient-to-br from-red-500 to-rose-600 p-2">
                        <AlertTriangle className="h-4 w-4 text-white" />
                      </div>
                      Cascaded Failures
                      <Badge variant="secondary" className="ml-auto">{failureResult.cascaded_failures.length}</Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="relative">
                    <div className="space-y-2 max-h-[300px] overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-muted scrollbar-track-transparent">
                      {failureResult.cascaded_failures.map((id) => (
                        <div key={id} className="flex items-center gap-2 p-3 rounded-lg bg-red-50 dark:bg-red-950/20 hover:bg-red-100 dark:hover:bg-red-950/30 transition-colors cursor-pointer" onClick={() => router.push(`/explorer?node=${encodeURIComponent(id)}`)}>
                          <AlertTriangle className="h-3.5 w-3.5 text-red-600 flex-shrink-0" />
                          <div className="flex-1 min-w-0">
                            <div className="text-sm font-medium truncate hover:underline transition-all">{getComponentName(id)}</div>
                            <div className="text-xs text-muted-foreground truncate">{id}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>

            {/* Per-Layer Impact */}
            {failureResult.layer_impacts && Object.keys(failureResult.layer_impacts).length > 0 && (
              <div>
                <div className="mb-4 flex items-center gap-3">
                  <div className="rounded-xl bg-gradient-to-br from-purple-600 via-violet-600 to-indigo-700 p-2.5 shadow-lg">
                    <Layers className="h-5 w-5 text-white" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold">Per-Layer Impact</h2>
                    <p className="text-sm text-muted-foreground">Impact breakdown across architectural layers</p>
                  </div>
                </div>

                <Card className="border-0 shadow-lg relative overflow-hidden">
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-purple-500 to-indigo-600">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  <CardHeader className="relative pb-3">
                    <CardTitle className="flex items-center gap-2 text-base">
                      <div className="rounded-lg bg-gradient-to-br from-purple-500 to-indigo-600 p-2">
                        <BarChart3 className="h-4 w-4 text-white" />
                      </div>
                      Layer Impact Scores
                      <Badge variant="secondary" className="ml-auto">{Object.keys(failureResult.layer_impacts).length} layers</Badge>
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="relative">
                    <div className="space-y-2">
                      {Object.entries(failureResult.layer_impacts)
                        .sort(([, a], [, b]) => (b as number) - (a as number))
                        .map(([layer, impact], index) => {
                          const impactValue = impact as number;
                          const impactPercent = (impactValue * 100).toFixed(1);
                          return (
                            <div key={layer} className="group p-3 rounded-lg bg-muted/30 hover:bg-muted/50 transition-all">
                              <div className="flex items-center gap-3 mb-2">
                                <div className="flex items-center justify-center w-6 h-6 rounded-full bg-primary/10 text-xs font-bold text-primary">
                                  {index + 1}
                                </div>
                                <div className="flex-1 min-w-0">
                                  <div className="font-medium text-sm">{layer.split(/[-_]/).map(word => word.charAt(0).toUpperCase() + word.slice(1)).join('-')}</div>
                                </div>
                                <div className="text-right">
                                  <div className={`text-sm font-bold ${getImpactColor(impactValue)}`}>
                                    {impactValue.toFixed(4)}
                                  </div>
                                  <div className="text-xs text-muted-foreground">
                                    {impactPercent}%
                                  </div>
                                </div>
                              </div>
                              <Progress value={impactValue * 100} className="h-1.5" />
                            </div>
                          );
                        })}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
              </>
            )}

        {/* Exhaustive Results - Show only in exhaustive mode */}
        {simulationMode === 'exhaustive' && exhaustiveSummary && (
          <>
            {/* Results Header */}
            <Card className="relative overflow-hidden border-0 shadow-xl">
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500">
                <div className="w-full h-full bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-700 rounded-lg" />
              </div>
              <CardContent className="relative p-6 text-white">
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-4">
                    <div className="rounded-xl bg-white/20 p-3 backdrop-blur-sm">
                      <BarChart3 className="h-8 w-8" />
                    </div>
                    <div>
                      <h2 className="text-2xl font-bold mb-1">Exhaustive Failure Analysis</h2>
                      <div className="text-base text-white/90">
                        Comprehensive failure impact analysis
                      </div>
                    </div>
                  </div>
                  <div className="flex items-start gap-6">
                    <div className="hidden md:flex items-center gap-4">
                      <div className="text-right">
                        <div className="text-3xl font-bold">{exhaustiveSummary.total_components || 0}</div>
                        <div className="text-sm text-white/80">Analyzed</div>
                      </div>
                      <div className="text-right">
                        <div className="text-3xl font-bold">{exhaustiveSummary.critical_count || 0}</div>
                        <div className="text-sm text-white/80">Critical</div>
                      </div>
                      <div className="text-right">
                        <div className="text-3xl font-bold">{exhaustiveSummary.high_count || 0}</div>
                        <div className="text-sm text-white/80">High</div>
                      </div>
                    </div>
                    <Button
                      onClick={clearExhaustiveResults}
                      variant="outline"
                      size="sm"
                      className="flex items-center gap-2 bg-white/10 hover:bg-white/20 border-white/30 text-white"
                    >
                      <XCircle className="h-4 w-4" />
                      Clear Results
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Overview Statistics */}
            <div>
              <div className="mb-4 flex items-center gap-3">
                <div className="rounded-xl bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-700 p-2.5 shadow-lg">
                  <Gauge className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-bold">Analysis Overview</h2>
                  <p className="text-sm text-muted-foreground">
                    Key metrics and impact distribution
                  </p>
                </div>
              </div>
              <div className="grid gap-4 md:grid-cols-3">
                <Card className="group relative overflow-hidden border-0 hover:shadow-xl hover:shadow-blue-500/25 hover:scale-[1.02] transition-all duration-300">
                  {/* Gradient border */}
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-cyan-500 to-teal-500">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  {/* Subtle background glow */}
                  <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-blue-500/30 via-blue-500/15 to-blue-500/5" />
                  
                  <CardContent className="relative p-6">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="rounded-xl bg-gradient-to-br from-blue-600 via-cyan-600 to-teal-600 p-2.5 shadow-lg">
                        <Server className="h-5 w-5 text-white" />
                      </div>
                      <h3 className="text-sm font-medium">Total Components</h3>
                    </div>
                    <div className="text-3xl font-bold text-blue-600">
                      {exhaustiveSummary.total_components || 0}
                    </div>
                    <p className="text-xs text-muted-foreground mt-2">
                      Analyzed components
                    </p>
                  </CardContent>
                </Card>

                <Card className="group relative overflow-hidden border-0 hover:shadow-xl hover:shadow-indigo-500/25 hover:scale-[1.02] transition-all duration-300">
                  {/* Gradient border */}
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-indigo-500 via-purple-500 to-violet-500">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  {/* Subtle background glow */}
                  <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-indigo-500/30 via-indigo-500/15 to-indigo-500/5" />
                  
                  <CardContent className="relative p-6">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="rounded-xl bg-gradient-to-br from-indigo-600 via-purple-600 to-violet-600 p-2.5 shadow-lg">
                        <Gauge className="h-5 w-5 text-white" />
                      </div>
                      <h3 className="text-sm font-medium">Average Impact</h3>
                    </div>
                    <div className="text-3xl font-bold text-indigo-600">
                      {(exhaustiveSummary.avg_impact || 0).toFixed(4)}
                    </div>
                    <p className="text-xs text-muted-foreground mt-2">
                      Mean composite score
                    </p>
                  </CardContent>
                </Card>

                <Card className="group relative overflow-hidden border-0 hover:shadow-xl hover:shadow-red-500/25 hover:scale-[1.02] transition-all duration-300">
                  {/* Gradient border */}
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-red-500 via-rose-500 to-pink-500">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  {/* Subtle background glow */}
                  <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-red-500/30 via-red-500/15 to-red-500/5" />
                  
                  <CardContent className="relative p-6">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="rounded-xl bg-gradient-to-br from-red-600 via-rose-600 to-pink-600 p-2.5 shadow-lg">
                        <AlertTriangle className="h-5 w-5 text-white" />
                      </div>
                      <h3 className="text-sm font-medium">Critical Components</h3>
                    </div>
                    <div className="text-3xl font-bold text-red-600">
                      {exhaustiveSummary.critical_count || 0}
                    </div>
                    <p className="text-xs text-muted-foreground mt-2">
                      Impact &gt; 0.5
                    </p>
                  </CardContent>
                </Card>
              </div>
            </div>

            {/* Impact Distribution */}
            <Card className="border-0 shadow-xl relative overflow-hidden">
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-violet-500 via-purple-500 to-fuchsia-600">
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-purple-500/10 via-purple-500/5 to-transparent" />
              
              <CardHeader className="relative">
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  Impact Distribution
                </CardTitle>
                <CardDescription>
                  Component severity classification
                </CardDescription>
              </CardHeader>
              <CardContent className="relative">
                <div className="grid gap-6 md:grid-cols-2">
                  {/* Critical & High Impact */}
                  <div className="space-y-3">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="rounded-lg bg-gradient-to-br from-red-500 to-red-600 p-2">
                        <AlertTriangle className="h-4 w-4 text-white" />
                      </div>
                      <h3 className="font-semibold text-base">High Severity</h3>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between p-3 rounded-lg bg-red-50 dark:bg-red-950/30 hover:bg-red-100 dark:hover:bg-red-950/40 transition-colors">
                        <span className="text-sm text-red-700 dark:text-red-300">Critical (&gt;0.5)</span>
                        <span className="font-semibold text-base text-red-600 dark:text-red-400">{exhaustiveSummary.critical_count || 0}</span>
                      </div>
                      <div className="flex items-center justify-between p-3 rounded-lg bg-orange-50 dark:bg-orange-950/30 hover:bg-orange-100 dark:hover:bg-orange-950/40 transition-colors">
                        <span className="text-sm text-orange-700 dark:text-orange-300">High (0.3-0.5)</span>
                        <span className="font-semibold text-base text-orange-600 dark:text-orange-400">{exhaustiveSummary.high_count || 0}</span>
                      </div>
                    </div>
                  </div>

                  {/* Medium & Low Impact */}
                  <div className="space-y-3">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="rounded-lg bg-gradient-to-br from-blue-500 to-blue-600 p-2">
                        <CheckCircle2 className="h-4 w-4 text-white" />
                      </div>
                      <h3 className="font-semibold text-base">Low Severity</h3>
                    </div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between p-3 rounded-lg bg-blue-50 dark:bg-blue-950/30 hover:bg-blue-100 dark:hover:bg-blue-950/40 transition-colors">
                        <span className="text-sm text-blue-700 dark:text-blue-300">Medium (0.1-0.3)</span>
                        <span className="font-semibold text-base text-blue-600 dark:text-blue-400">{exhaustiveSummary.medium_count || 0}</span>
                      </div>
                      <div className="flex items-center justify-between p-3 rounded-lg bg-green-50 dark:bg-green-950/30 hover:bg-green-100 dark:hover:bg-green-950/40 transition-colors">
                        <span className="text-sm text-green-700 dark:text-green-300">Low (≤0.1)</span>
                        <span className="font-semibold text-base text-green-600 dark:text-green-400">{exhaustiveSummary.low_count || 0}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Top Critical Components */}
            <div>
              <div className="mb-4 flex items-center gap-3">
                <div className="rounded-xl bg-gradient-to-br from-red-600 via-rose-600 to-red-700 p-2.5 shadow-lg">
                  <AlertTriangle className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h2 className="text-xl font-bold">Top Critical Components</h2>
                  <p className="text-sm text-muted-foreground">
                    Components with highest failure impact (showing top 20)
                  </p>
                </div>
              </div>
              
              <div className="rounded-md border">
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b bg-muted/50">
                        <th className="p-3 text-left text-sm font-medium">Rank</th>
                        <th className="p-3 text-left text-sm font-medium">Component</th>
                        <th className="p-3 text-left text-sm font-medium">Type</th>
                        <th className="p-3 text-right text-sm font-medium">Impact Score</th>
                        <th className="p-3 text-right text-sm font-medium">Cascades</th>
                        <th className="p-3 text-right text-sm font-medium">Path Loss</th>
                      </tr>
                    </thead>
                    <tbody>
                      {exhaustiveResults.slice(0, 20).map((result, index) => {
                        const impactValue = result.impact.composite_impact || 0;
                        const cascadeCount = result.impact.cascade?.count || 0;
                        const reachLoss = result.impact.reachability?.loss_percent || 0;
                        const componentName = getComponentName(result.target_id);
                        
                        return (
                          <tr 
                            key={result.target_id} 
                            className="border-b last:border-0 hover:bg-muted/30 transition-colors cursor-pointer group"
                            onClick={() => router.push(`/explorer?node=${encodeURIComponent(result.target_id)}`)}
                          >
                            <td className="p-3">
                              <div className="flex items-center gap-2">
                                <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                                  index === 0 ? 'bg-yellow-500/20 text-yellow-600 dark:text-yellow-400' :
                                  index === 1 ? 'bg-gray-400/20 text-gray-600 dark:text-gray-400' :
                                  index === 2 ? 'bg-orange-500/20 text-orange-600 dark:text-orange-400' :
                                  'bg-muted text-muted-foreground'
                                }`}>
                                  {index + 1}
                                </div>
                              </div>
                            </td>
                            <td className="p-3">
                              <div className="font-medium text-sm group-hover:underline transition-all">{componentName}</div>
                              <div className="text-xs text-muted-foreground font-mono">{result.target_id}</div>
                            </td>
                            <td className="p-3">
                              <Badge variant="outline" className="text-xs">
                                {result.target_type}
                              </Badge>
                            </td>
                            <td className="p-3 text-right">
                              <div className={`font-semibold ${getImpactColor(impactValue)}`}>
                                {impactValue.toFixed(4)}
                              </div>
                            </td>
                            <td className="p-3 text-right">
                              <div className="font-semibold text-orange-600 dark:text-orange-400">
                                {cascadeCount}
                              </div>
                            </td>
                            <td className="p-3 text-right">
                              <div className={`font-semibold ${
                                reachLoss > 50 ? 'text-red-600 dark:text-red-400' :
                                reachLoss > 25 ? 'text-orange-600 dark:text-orange-400' :
                                'text-blue-600 dark:text-blue-400'
                              }`}>
                                {reachLoss.toFixed(1)}%
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
              <p className="text-xs text-muted-foreground mt-3">
                These components are critical points of failure. Their unavailability would severely impact system functionality.
              </p>
            </div>
          </>
        )}

          {/* Configuration Panel for Report Generation */}
          {simulationMode === 'report' && (
            <div className="space-y-3">
              <Label className="text-base font-semibold flex items-center gap-2">
                <FileSpreadsheet className="h-4 w-4" />
                Report Generation Parameters
              </Label>
              <p className="text-sm text-muted-foreground/80">
                Generate comprehensive analysis report across layers
              </p>
              <div className="space-y-4 pt-2">
                <div className="space-y-2">
                  <Label>Select Layers</Label>
                  <div className="flex flex-wrap gap-2">
                    {["application", "infrastructure", "middleware", "system"].map((layer) => (
                      <Badge
                        key={layer}
                        variant={reportLayers.includes(layer) ? "default" : "outline"}
                        className="cursor-pointer"
                        onClick={() => {
                          setReportLayers((prev) =>
                            prev.includes(layer)
                              ? prev.filter((l) => l !== layer)
                              : [...prev, layer]
                          )
                        }}
                      >
                        {layer.split('-').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join('-')}
                      </Badge>
                    ))}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Choose one or more architectural layers to include in the comprehensive report
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Report Results - Show only in report mode */}
        {simulationMode === 'report' && report && (
          <>
            {/* Results Header */}
            {report.graph_summary && (
              <>
              <Card className="relative overflow-hidden border-0 shadow-xl">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-cyan-500 via-teal-500 to-emerald-500">
                  <div className="w-full h-full bg-gradient-to-r from-cyan-600 via-teal-600 to-emerald-700 rounded-lg" />
                </div>
                <CardContent className="relative p-6 text-white">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                      <div className="rounded-xl bg-white/20 p-3 backdrop-blur-sm">
                        <FileSpreadsheet className="h-8 w-8" />
                      </div>
                      <div>
                        <h2 className="text-2xl font-bold mb-1">Comprehensive Simulation Report</h2>
                        <p className="text-base text-white/90">
                          Complete analysis across selected architectural layers
                        </p>
                      </div>
                    </div>
                    <Button
                      onClick={clearReport}
                      variant="outline"
                      size="sm"
                      className="flex items-center gap-2 bg-white/10 hover:bg-white/20 border-white/30 text-white"
                    >
                      <XCircle className="h-4 w-4" />
                      Clear Results
                    </Button>
                  </div>
                </CardContent>
              </Card>

              {/* Graph Summary */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <div className="rounded-xl bg-gradient-to-br from-cyan-600 via-teal-600 to-emerald-700 p-2.5 shadow-lg">
                    <Network className="h-5 w-5 text-white" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold">Graph Overview</h2>
                    <p className="text-sm text-muted-foreground">
                      System topology and component statistics
                    </p>
                  </div>
                </div>
                <div className="grid gap-4 md:grid-cols-4">
                  <Card className="group relative overflow-hidden border-0 hover:shadow-xl hover:shadow-cyan-500/25 hover:scale-[1.02] transition-all duration-300">
                    {/* Gradient border */}
                    <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-cyan-500 via-sky-500 to-blue-500">
                      <div className="w-full h-full bg-background rounded-lg" />
                    </div>
                    {/* Subtle background glow */}
                    <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-cyan-500/30 via-cyan-500/15 to-cyan-500/5" />
                    
                    <CardContent className="relative p-6">
                      <div className="flex items-center gap-2 mb-4">
                        <div className="rounded-xl bg-gradient-to-br from-cyan-600 via-sky-600 to-blue-600 p-2.5 shadow-lg">
                          <Server className="h-5 w-5 text-white" />
                        </div>
                        <h3 className="text-sm font-medium">Total Nodes</h3>
                      </div>
                      <div className="text-3xl font-bold text-cyan-600">
                        {report.graph_summary.total_nodes || 0}
                      </div>
                      <p className="text-xs text-muted-foreground mt-2">
                        System components
                      </p>
                    </CardContent>
                  </Card>

                  <Card className="group relative overflow-hidden border-0 hover:shadow-xl hover:shadow-teal-500/25 hover:scale-[1.02] transition-all duration-300">
                    {/* Gradient border */}
                    <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-teal-500 via-emerald-500 to-green-500">
                      <div className="w-full h-full bg-background rounded-lg" />
                    </div>
                    {/* Subtle background glow */}
                    <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-teal-500/30 via-teal-500/15 to-teal-500/5" />
                    
                    <CardContent className="relative p-6">
                      <div className="flex items-center gap-2 mb-4">
                        <div className="rounded-xl bg-gradient-to-br from-teal-600 via-emerald-600 to-green-600 p-2.5 shadow-lg">
                          <Network className="h-5 w-5 text-white" />
                        </div>
                        <h3 className="text-sm font-medium">Total Edges</h3>
                      </div>
                      <div className="text-3xl font-bold text-teal-600">
                        {report.graph_summary.total_edges || 0}
                      </div>
                      <p className="text-xs text-muted-foreground mt-2">
                        Component connections
                      </p>
                    </CardContent>
                  </Card>

                  <Card className="group relative overflow-hidden border-0 hover:shadow-xl hover:shadow-emerald-500/25 hover:scale-[1.02] transition-all duration-300">
                    {/* Gradient border */}
                    <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-emerald-500 via-green-500 to-lime-500">
                      <div className="w-full h-full bg-background rounded-lg" />
                    </div>
                    {/* Subtle background glow */}
                    <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-emerald-500/30 via-emerald-500/15 to-emerald-500/5" />
                    
                    <CardContent className="relative p-6">
                      <div className="flex items-center gap-2 mb-4">
                        <div className="rounded-xl bg-gradient-to-br from-emerald-600 via-green-600 to-lime-600 p-2.5 shadow-lg">
                          <MessageSquare className="h-5 w-5 text-white" />
                        </div>
                        <h3 className="text-sm font-medium">Topics</h3>
                      </div>
                      <div className="text-3xl font-bold text-emerald-600">
                        {report.graph_summary.topics || 0}
                      </div>
                      <p className="text-xs text-muted-foreground mt-2">
                        Pub-sub topics
                      </p>
                    </CardContent>
                  </Card>

                  <Card className="group relative overflow-hidden border-0 hover:shadow-xl hover:shadow-green-500/25 hover:scale-[1.02] transition-all duration-300">
                    {/* Gradient border */}
                    <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-green-500 via-lime-500 to-yellow-500">
                      <div className="w-full h-full bg-background rounded-lg" />
                    </div>
                    {/* Subtle background glow */}
                    <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-green-500/30 via-green-500/15 to-green-500/5" />
                    
                    <CardContent className="relative p-6">
                      <div className="flex items-center gap-2 mb-4">
                        <div className="rounded-xl bg-gradient-to-br from-green-600 via-lime-600 to-yellow-600 p-2.5 shadow-lg">
                          <TrendingUp className="h-5 w-5 text-white" />
                        </div>
                        <h3 className="text-sm font-medium">Pub-Sub Paths</h3>
                      </div>
                      <div className="text-3xl font-bold text-green-600">
                        {report.graph_summary.pub_sub_paths || 0}
                      </div>
                      <p className="text-xs text-muted-foreground mt-2">
                        Communication paths
                      </p>
                    </CardContent>
                  </Card>
                </div>

                {/* Component Types */}
                {report.graph_summary.component_types && (
                  <Card className="border-0 shadow-xl relative overflow-hidden">
                    <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-cyan-500 via-teal-500 to-emerald-500">
                      <div className="w-full h-full bg-background rounded-lg" />
                    </div>
                    <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-cyan-500/10 via-teal-500/5 to-transparent" />
                    
                    <CardHeader className="relative">
                      <CardTitle className="flex items-center gap-2">
                        <Server className="h-5 w-5" />
                        Component Types
                      </CardTitle>
                      <CardDescription>
                        System component classification
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="relative">
                      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                        {Object.entries(report.graph_summary.component_types).map(([type, count]) => (
                          <div key={type} className="flex items-center justify-between p-3 rounded-lg bg-cyan-50 dark:bg-cyan-950/30 hover:bg-cyan-100 dark:hover:bg-cyan-950/40 transition-colors">
                            <span className="text-sm text-cyan-700 dark:text-cyan-300 font-medium">{type}</span>
                            <span className="font-semibold text-base text-cyan-600 dark:text-cyan-400">{count}</span>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                )}
              </div>
              </>
            )}

            {/* Layer Metrics */}
            {Object.entries(report.layer_metrics).map(([layerName, metrics]) => (
                  <div key={layerName} className="space-y-4">
                    <div className="flex items-center gap-2">
                      <div className="rounded-xl bg-gradient-to-br from-indigo-600 via-purple-600 to-pink-700 p-2.5 shadow-lg">
                        <Layers className="h-5 w-5 text-white" />
                      </div>
                      <div>
                        <h2 className="text-xl font-bold">Layer: {layerName.split(/[-_]/).map(word => word.charAt(0).toUpperCase() + word.slice(1)).join('-')}</h2>
                        <p className="text-sm text-muted-foreground">Performance and failure analysis metrics</p>
                      </div>
                    </div>
                    
                    <div className="grid gap-4 md:grid-cols-4">
                      {/* Throughput Card */}
                      <Card className="group relative overflow-hidden border-0 hover:shadow-xl hover:shadow-purple-500/25 hover:scale-[1.02] transition-all duration-300">
                        <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-purple-500 via-violet-500 to-fuchsia-500">
                          <div className="w-full h-full bg-background rounded-lg" />
                        </div>
                        <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-purple-500/30 via-purple-500/15 to-purple-500/5" />
                        
                        <CardContent className="relative p-6">
                          <div className="flex items-center gap-2 mb-4">
                            <div className="rounded-xl bg-gradient-to-br from-purple-600 via-violet-600 to-fuchsia-600 p-2.5 shadow-lg">
                              <MessageSquare className="h-5 w-5 text-white" />
                            </div>
                            <h3 className="text-sm font-medium">Throughput</h3>
                          </div>
                          <div className="text-3xl font-bold text-purple-600">
                            {metrics.event_metrics?.throughput || 0}
                          </div>
                          <p className="text-xs text-muted-foreground mt-2">
                            Messages processed
                          </p>
                        </CardContent>
                      </Card>

                      {/* Delivery Rate Card */}
                      <Card className="group relative overflow-hidden border-0 hover:shadow-xl hover:shadow-green-500/25 hover:scale-[1.02] transition-all duration-300">
                        <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500">
                          <div className="w-full h-full bg-background rounded-lg" />
                        </div>
                        <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-green-500/30 via-green-500/15 to-green-500/5" />
                        
                        <CardContent className="relative p-6">
                          <div className="flex items-center gap-2 mb-4">
                            <div className="rounded-xl bg-gradient-to-br from-green-600 via-emerald-600 to-teal-600 p-2.5 shadow-lg">
                              <CheckCircle2 className="h-5 w-5 text-white" />
                            </div>
                            <h3 className="text-sm font-medium">Delivery Rate</h3>
                          </div>
                          <div className="text-3xl font-bold text-green-600">
                            {parseFloat((metrics.event_metrics?.delivery_rate_percent || 0).toFixed(1))}%
                          </div>
                          <p className="text-xs text-muted-foreground mt-2">
                            Successful deliveries
                          </p>
                        </CardContent>
                      </Card>

                      {/* Avg Latency Card */}
                      <Card className="group relative overflow-hidden border-0 hover:shadow-xl hover:shadow-blue-500/25 hover:scale-[1.02] transition-all duration-300">
                        <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-cyan-500 to-sky-500">
                          <div className="w-full h-full bg-background rounded-lg" />
                        </div>
                        <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-blue-500/30 via-blue-500/15 to-blue-500/5" />
                        
                        <CardContent className="relative p-6">
                          <div className="flex items-center gap-2 mb-4">
                            <div className="rounded-xl bg-gradient-to-br from-blue-600 via-cyan-600 to-sky-600 p-2.5 shadow-lg">
                              <Clock className="h-5 w-5 text-white" />
                            </div>
                            <h3 className="text-sm font-medium">Avg Latency</h3>
                          </div>
                          <div className="text-3xl font-bold text-blue-600">
                            {(metrics.event_metrics?.avg_latency_ms || 0).toFixed(2)}
                          </div>
                          <p className="text-xs text-muted-foreground mt-2">
                            Milliseconds
                          </p>
                        </CardContent>
                      </Card>

                      {/* Max Impact Card */}
                      <Card className="group relative overflow-hidden border-0 hover:shadow-xl hover:shadow-red-500/25 hover:scale-[1.02] transition-all duration-300">
                        <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-red-500 via-rose-500 to-pink-500">
                          <div className="w-full h-full bg-background rounded-lg" />
                        </div>
                        <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-red-500/30 via-red-500/15 to-red-500/5" />
                        
                        <CardContent className="relative p-6">
                          <div className="flex items-center gap-2 mb-4">
                            <div className="rounded-xl bg-gradient-to-br from-red-600 via-rose-600 to-pink-600 p-2.5 shadow-lg">
                              <AlertTriangle className="h-5 w-5 text-white" />
                            </div>
                            <h3 className="text-sm font-medium">Max Impact</h3>
                          </div>
                          <div className="text-3xl font-bold text-red-600">
                            {(metrics.failure_metrics?.max_impact || 0).toFixed(4)}
                          </div>
                          <p className="text-xs text-muted-foreground mt-2">
                            Failure impact score
                          </p>
                        </CardContent>
                      </Card>
                    </div>

                    {/* Criticality Distribution */}
                    <Card className="border-0 shadow-xl relative overflow-hidden">
                      <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500">
                        <div className="w-full h-full bg-background rounded-lg" />
                      </div>
                      <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-indigo-500/10 via-purple-500/5 to-transparent" />
                      
                      <CardHeader className="relative">
                        <CardTitle className="flex items-center gap-2">
                          <BarChart3 className="h-5 w-5" />
                          Criticality Distribution
                        </CardTitle>
                        <CardDescription>
                          Component severity classification
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="relative">
                        <div className="grid grid-cols-5 gap-3">
                          <div className="flex flex-col items-center p-3 rounded-lg border bg-muted/30">
                            <div className="text-xs text-muted-foreground mb-1">Total</div>
                            <div className="text-2xl font-bold">{metrics.criticality?.total_components || 0}</div>
                          </div>
                          <div className="flex flex-col items-center p-3 rounded-lg border bg-red-500/10 border-red-500/20">
                            <div className="text-xs text-red-500 mb-1">Critical</div>
                            <div className="text-2xl font-bold text-red-500">{metrics.criticality?.critical || 0}</div>
                          </div>
                          <div className="flex flex-col items-center p-3 rounded-lg border bg-orange-500/10 border-orange-500/20">
                            <div className="text-xs text-orange-500 mb-1">High</div>
                            <div className="text-2xl font-bold text-orange-500">{metrics.criticality?.high || 0}</div>
                          </div>
                          <div className="flex flex-col items-center p-3 rounded-lg border bg-blue-500/10 border-blue-500/20">
                            <div className="text-xs text-blue-500 mb-1">Medium</div>
                            <div className="text-2xl font-bold text-blue-500">{metrics.criticality?.medium || 0}</div>
                          </div>
                          <div className="flex flex-col items-center p-3 rounded-lg border bg-purple-500/10 border-purple-500/20">
                            <div className="text-xs text-purple-500 mb-1">SPOFs</div>
                            <div className="text-2xl font-bold text-purple-500">{metrics.criticality?.spof_count || 0}</div>
                          </div>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                ))}

                {report.top_critical.length > 0 && (
                  <div>
                    <div className="mb-4 flex items-center gap-3">
                      <div className="rounded-xl bg-gradient-to-br from-red-600 via-rose-600 to-red-700 p-2.5 shadow-lg">
                        <AlertTriangle className="h-5 w-5 text-white" />
                      </div>
                      <div>
                        <h2 className="text-xl font-bold">Top Critical Components</h2>
                        <p className="text-sm text-muted-foreground">
                          Most critical components across all analyzed layers (showing top 10)
                        </p>
                      </div>
                    </div>
                    
                    <div className="rounded-md border">
                      <div className="overflow-x-auto">
                        <table className="w-full">
                          <thead>
                            <tr className="border-b bg-muted/50">
                              <th className="p-3 text-left text-sm font-medium">Rank</th>
                              <th className="p-3 text-left text-sm font-medium">Component</th>
                              <th className="p-3 text-left text-sm font-medium">Type</th>
                              <th className="p-3 text-right text-sm font-medium">Impact Score</th>
                              <th className="p-3 text-right text-sm font-medium">Cascades</th>
                              <th className="p-3 text-left text-sm font-medium">Level</th>
                            </tr>
                          </thead>
                          <tbody>
                            {report.top_critical.slice(0, 20).map((comp, index) => {
                              const impactValue = comp.scores?.combined_impact || 0;
                              const cascadeCount = comp.metrics?.cascade_count || 0;
                              const componentName = getComponentName(comp.id);
                              
                              return (
                                <tr 
                                  key={comp.id} 
                                  className="border-b last:border-0 hover:bg-muted/30 transition-colors cursor-pointer group"
                                  onClick={() => router.push(`/explorer?node=${encodeURIComponent(comp.id)}`)}
                                >
                                  <td className="p-3">
                                    <div className="flex items-center gap-2">
                                      <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                                        index === 0 ? 'bg-yellow-500/20 text-yellow-600 dark:text-yellow-400' :
                                        index === 1 ? 'bg-gray-400/20 text-gray-600 dark:text-gray-400' :
                                        index === 2 ? 'bg-orange-500/20 text-orange-600 dark:text-orange-400' :
                                        'bg-muted text-muted-foreground'
                                      }`}>
                                        {index + 1}
                                      </div>
                                    </div>
                                  </td>
                                  <td className="p-3">
                                    <div className="font-medium text-sm group-hover:underline transition-all">{componentName}</div>
                                    <div className="text-xs text-muted-foreground font-mono">{comp.id}</div>
                                  </td>
                                  <td className="p-3">
                                    <Badge variant="outline" className="text-xs">
                                      {comp.type}
                                    </Badge>
                                  </td>
                                  <td className="p-3 text-right">
                                    <div className={`font-semibold ${getImpactColor(impactValue)}`}>
                                      {impactValue.toFixed(4)}
                                    </div>
                                  </td>
                                  <td className="p-3 text-right">
                                    <div className="font-semibold text-orange-600 dark:text-orange-400">
                                      {cascadeCount}
                                    </div>
                                  </td>
                                  <td className="p-3">
                                    <Badge variant={getCriticalityBadgeVariant(comp.level)} className="text-xs">
                                      {comp.level.charAt(0).toUpperCase() + comp.level.slice(1)}
                                    </Badge>
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                    <p className="text-xs text-muted-foreground mt-3">
                      These components are critical points of failure. Their unavailability would severely impact system functionality.
                    </p>
                  </div>
                )}

                {/* Recommendations */}
                {/* Recommendations */}
                {report.recommendations.length > 0 && (
                  <div>
                    <div className="mb-4 flex items-center gap-3">
                      <div className="rounded-xl bg-gradient-to-br from-green-600 via-emerald-600 to-green-700 p-2.5 shadow-lg">
                        <CheckCircle2 className="h-5 w-5 text-white" />
                      </div>
                      <div>
                        <h2 className="text-xl font-bold">Recommendations</h2>
                        <p className="text-sm text-muted-foreground">Suggested improvements based on analysis results</p>
                      </div>
                    </div>
                    
                    <Card className="border-0 shadow-xl relative overflow-hidden">
                      <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500">
                        <div className="w-full h-full bg-background rounded-lg" />
                      </div>
                      <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-green-500/10 via-emerald-500/5 to-transparent" />
                      
                      <CardHeader className="relative">
                        <CardTitle className="flex items-center gap-2">
                          <CheckCircle2 className="h-5 w-5" />
                          Action Items
                        </CardTitle>
                        <CardDescription>
                          Prioritized recommendations for system improvement
                        </CardDescription>
                      </CardHeader>
                      <CardContent className="relative">
                        <div className="space-y-3">
                          {report.recommendations.map((rec, index) => (
                            <div key={index} className="flex gap-3 p-3 rounded-lg bg-green-50 dark:bg-green-950/30 hover:bg-green-100 dark:hover:bg-green-950/40 transition-colors">
                              <div className="flex items-center justify-center w-6 h-6 rounded-full bg-green-500/20 text-xs font-bold text-green-600 dark:text-green-400 shrink-0 mt-0.5">
                                {index + 1}
                              </div>
                              <span className="text-sm leading-relaxed">{rec}</span>
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                )}
          </>
        )}
      </div>
    </AppLayout>
  )
}
