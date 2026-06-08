"use client"

import React, { useState, useEffect, useMemo } from "react"
import { useRouter } from "next/navigation"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { Skeleton } from "@/components/ui/skeleton"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Input } from "@/components/ui/input"
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Database,
  TrendingDown,
  TrendingUp,
  Shield,
  Wrench,
  Server,
  FileSpreadsheet,
  Info,
  Zap,
  Gauge,
  ChevronRight,
  Layers,
  Box,
  Network,
  BarChart3,
  Lightbulb,
  Target,
  Filter,
  Clock,
  Search,
  X,
  XCircle,
  ChevronLeft,
  ChevronsLeft,
  ChevronsRight,
  MessageSquare,
  Tag,
  Hash,
  Download,
  LayoutGrid,
  Share2,
  Terminal,
  ChevronDown,
} from "lucide-react"
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { useConnection } from "@/lib/stores/connection-store"
import { useAnalysis } from "@/lib/stores/analysis-store"
import { apiClient } from "@/lib/api/client"
import dynamic from "next/dynamic"
import { TermTooltip } from "@/components/ui/term-tooltip"
import { ScoreTooltip } from "@/components/ui/score-tooltip"
import { ItemTooltip } from "@/components/ui/item-tooltip"

const ReactECharts = dynamic(() => import("echarts-for-react"), { ssr: false })

// Types for the new API structure
interface ComponentAnalysis {
  id: string
  name: string
  type: string
  criticality_level: string
  criticality_levels?: {
    reliability: string
    maintainability: string
    availability: string
    vulnerability: string
    overall: string
  }
  scores: {
    reliability: number
    maintainability: number
    availability: number
    vulnerability: number
    overall: number
  }
}

interface EdgeAnalysis {
  source: string
  target: string
  source_name?: string
  target_name?: string
  type: string
  criticality_level: string
  scores: {
    reliability: number
    maintainability: number
    availability: number
    vulnerability: number
    overall: number
  }
}

interface Problem {
  entity_id: string
  type: string
  category: string
  severity: string
  name: string
  description: string
  recommendation: string
}

interface AnalysisResult {
  context?: string
  summary: any
  stats: any
  components: ComponentAnalysis[]
  edges?: EdgeAnalysis[]
  problems: Problem[]
  logs?: string[]
}

export default function AnalysisPage() {
  const router = useRouter()
  const { status, stats, initialLoadComplete } = useConnection()
  const { getAnalysis, setAnalysis, clearAnalysis } = useAnalysis()

  // Layer selection
  const [selectedLayer, setSelectedLayer] = useState<string>('system')
  
  // Loading and error states
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [elapsedTime, setElapsedTime] = useState(0)
  const [startTime, setStartTime] = useState<number | null>(null)

  // Analysis log panel state
  const [logsOpen, setLogsOpen] = useState(false)

  // Track dark mode so charts re-render on theme switch
  const [isDark, setIsDark] = useState(() =>
    typeof document !== 'undefined' && document.documentElement.classList.contains('dark')
  )
  useEffect(() => {
    const observer = new MutationObserver(() =>
      setIsDark(document.documentElement.classList.contains('dark'))
    )
    observer.observe(document.documentElement, { attributeFilter: ['class'] })
    return () => observer.disconnect()
  }, [])
  
  // Estimated duration for progress calculation (in seconds)
  const estimatedDuration = 30

  // Issues pagination and filtering
  const [issuesPage, setIssuesPage] = useState(1)
  const [issuesPerPage] = useState(10)
  const [severityFilter, setSeverityFilter] = useState<string>('all')
  const [categoryFilter, setCategoryFilter] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState<string>('')

  // Critical components pagination and filtering
  const [componentsPage, setComponentsPage] = useState(1)
  const [componentsPerPage] = useState(10)
  const [compSearchQuery, setCompSearchQuery] = useState<string>('')
  const [compTypeFilter, setCompTypeFilter] = useState<string>('all')
  const [compLevelFilter, setCompLevelFilter] = useState<string>('all')

  // Critical edges pagination
  const [edgesPage, setEdgesPage] = useState(1)
  const [edgesPerPage] = useState(10)

  // Node properties map for tooltip enrichment: id → { type, properties }
  const [nodeById, setNodeById] = useState<Map<string, { type: string; properties: Record<string, unknown> }>>(new Map())

  const isConnected = status === 'connected'

  // Fetch raw node properties once when connected
  useEffect(() => {
    if (!isConnected) return
    apiClient.getGraphData().then(data => {
      const m = new Map<string, { type: string; properties: Record<string, unknown> }>()
      data.nodes.forEach((n: any) => m.set(n.id, { type: n.type ?? 'Application', properties: n.properties ?? {} }))
      setNodeById(m)
    }).catch(() => {})
  }, [isConnected])

  // Generate cache key for current analysis configuration
  const getCacheKey = () => `layer:${selectedLayer}`

  // Get current analysis data from cache
  const analysisData = getAnalysis(getCacheKey())

  // Track elapsed time during analysis
  useEffect(() => {
    if (isLoading && startTime) {
      const interval = setInterval(() => {
        setElapsedTime(Math.floor((Date.now() - startTime) / 1000))
      }, 1000)

      return () => clearInterval(interval)
    } else {
      setElapsedTime(0)
    }
  }, [isLoading, startTime])

  // Handle analysis based on selected mode
  const handleAnalyze = async () => {
    if (!isConnected || isLoading) return // Prevent multiple simultaneous calls

    setIsLoading(true)
    setStartTime(Date.now())
    setError(null)

    try {
      let response: any = null
      response = await apiClient.analyzeByLayer(selectedLayer)

      if (!response) {
        throw new Error('No response from analysis')
      }

      if (response.success && response.analysis) {
        const cacheKey = getCacheKey()
        setAnalysis(cacheKey, response.analysis)
      } else {
        throw new Error('Invalid response from server')
      }
    } catch (error: any) {
      const errorMsg = error.response?.data?.detail || error.message || 'Analysis failed'
      setError(errorMsg)
      console.error('Analysis error:', errorMsg)
    } finally {
      setIsLoading(false)
      setStartTime(null)
    }
  }
  const getScoreColor = (score: number) => {
    // score is raw 0-1 risk score (high = bad). Display is inverted: (1-score)*100 where high = good.
    const rawPct = score * 100
    if (rawPct >= 80) return "text-red-600 dark:text-red-500"   // Critical risk → low display score → red
    if (rawPct >= 60) return "text-yellow-600 dark:text-yellow-500"
    return "text-green-600 dark:text-green-500" // Low risk → high display score → green
  }

  const getSeverityVariant = (severity: string) => {
    switch (severity.toLowerCase()) {
      case 'critical': return 'destructive'
      case 'high': return 'default'
      case 'medium': return 'secondary'
      default: return 'outline'
    }
  }

  const getCriticalityColor = (level: string) => {
    switch (level.toLowerCase()) {
      case 'critical': return 'bg-red-500 text-white'
      case 'high': return 'bg-orange-500 text-white'
      case 'medium': return 'bg-yellow-500 text-white'
      case 'low': return 'bg-green-500 text-white'
      default: return 'bg-gray-500 text-white'
    }
  }

  const formatKey = (key: string): string => {
    return key
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ')
  }

  // Calculate aggregate scores
  const calculateAggregateScores = () => {
    if (!analysisData || !analysisData.components || analysisData.components.length === 0) {
      return { reliability: 0, maintainability: 0, availability: 0, vulnerability: 0, overall: 0 }
    }

    const components = analysisData.components
    const edges = analysisData.edges || []

    // Combine components and edges for overall system score
    const allEntities = [...components, ...edges]

    if (allEntities.length === 0) {
      return { reliability: 0, maintainability: 0, availability: 0, vulnerability: 0, overall: 0 }
    }

    // Invert raw risk scores so that high display score = good quality
    const avg = (key: keyof typeof allEntities[0]['scores']) =>
      (1 - allEntities.reduce((sum, e) => sum + (e.scores?.[key] ?? 0), 0) / allEntities.length) * 100

    return {
      reliability: avg('reliability'),
      maintainability: avg('maintainability'),
      availability: avg('availability'),
      vulnerability: avg('vulnerability'),
      overall: avg('overall'),
    }
  }

  // Get critical components
  const getCriticalComponents = () => {
    if (!analysisData?.components) return []
    return [...analysisData.components]
      .sort((a, b) => b.scores.overall - a.scores.overall)
  }

  // Get critical edges
  const getCriticalEdges = () => {
    if (!analysisData?.edges) return []
    return analysisData.edges
      .filter(e => e.criticality_level === 'critical' || e.criticality_level === 'high')
      .sort((a, b) => b.scores.overall - a.scores.overall)
  }

  // Get problems by category
  const getProblemsByCategory = (category: string) => {
    if (!analysisData?.problems) return []
    return analysisData.problems.filter(p => 
      p.category.toLowerCase() === category.toLowerCase()
    )
  }

  // Filtered and paginated issues
  const filteredIssues = useMemo(() => {
    if (!analysisData?.problems) return []
    
    let filtered = analysisData.problems

    // Apply severity filter
    if (severityFilter !== 'all') {
      filtered = filtered.filter(p => p.severity.toLowerCase() === severityFilter.toLowerCase())
    }

    // Apply category filter
    if (categoryFilter !== 'all') {
      filtered = filtered.filter(p => p.category.toLowerCase() === categoryFilter.toLowerCase())
    }

    // Apply search query
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase()
      filtered = filtered.filter(p => {
        // Search in problem fields
        const matchesProblemFields = 
          p.name.toLowerCase().includes(query) ||
          p.description.toLowerCase().includes(query) ||
          p.recommendation.toLowerCase().includes(query) ||
          p.entity_id.toLowerCase().includes(query)
        
        if (matchesProblemFields) return true

        // Search in component name(s)
        const isLink = p.entity_id.includes('->')
        
        if (isLink) {
          // For edges/cycles, search in all node names
          const nodeIds = p.entity_id.split('->').map(s => s.trim())
          return nodeIds.some(id => {
            const comp = analysisData.components?.find(c => c.id === id)
            return comp?.name && comp.name.toLowerCase().includes(query)
          })
        } else {
          // For single components, search in component name
          const component = analysisData.components?.find(c => c.id === p.entity_id)
          return component?.name && component.name.toLowerCase().includes(query)
        }
      })
    }

    return filtered
  }, [analysisData?.problems, analysisData?.components, analysisData?.edges, severityFilter, categoryFilter, searchQuery])

  // Paginated issues
  const paginatedIssues = useMemo(() => {
    const startIdx = (issuesPage - 1) * issuesPerPage
    const endIdx = startIdx + issuesPerPage
    return filteredIssues.slice(startIdx, endIdx)
  }, [filteredIssues, issuesPage, issuesPerPage])

  const totalIssuesPages = Math.ceil(filteredIssues.length / issuesPerPage)

  // Get unique severities and categories for filters
  const availableSeverities = useMemo(() => {
    if (!analysisData?.problems) return []
    const severities = new Set(analysisData.problems.map(p => p.severity))
    return Array.from(severities)
  }, [analysisData?.problems])

  const availableCategories = useMemo(() => {
    if (!analysisData?.problems) return []
    const categories = new Set(analysisData.problems.map(p => p.category))
    return Array.from(categories)
  }, [analysisData?.problems])

  // Reset to page 1 when filters change
  useEffect(() => {
    setIssuesPage(1)
  }, [severityFilter, categoryFilter, searchQuery])

  // Filtered + paginated critical components
  const allCriticalComponents = useMemo(() => {
    let comps = getCriticalComponents()
    if (compTypeFilter !== 'all') comps = comps.filter(c => c.type === compTypeFilter)
    if (compLevelFilter !== 'all') comps = comps.filter(c => c.criticality_level === compLevelFilter)
    if (compSearchQuery.trim()) {
      const q = compSearchQuery.toLowerCase()
      comps = comps.filter(c => (c.name || c.id).toLowerCase().includes(q) || c.type.toLowerCase().includes(q))
    }
    return comps
  }, [analysisData?.components, compTypeFilter, compLevelFilter, compSearchQuery])

  const availableCompTypes = useMemo(() => {
    if (!analysisData?.components) return []
    return Array.from(new Set(analysisData.components.map((c: any) => c.type)))
  }, [analysisData?.components])

  const availableCompLevels = useMemo(() => {
    const order = ['critical', 'high', 'medium', 'low', 'minimal']
    if (!analysisData?.components) return []
    const present = new Set(analysisData.components.map((c: any) => c.criticality_level))
    return order.filter(l => present.has(l))
  }, [analysisData?.components])

  // Reset pages when analysis data changes (new analysis run)
  useEffect(() => {
    setComponentsPage(1)
  }, [analysisData?.components])

  useEffect(() => {
    setIssuesPage(1)
  }, [analysisData?.problems])

  useEffect(() => {
    setEdgesPage(1)
  }, [analysisData?.edges])

  // Reset page when filters change (but not when analysisData changes, handled above)
  useEffect(() => {
    setComponentsPage(1)
  }, [compTypeFilter, compLevelFilter, compSearchQuery])

  const paginatedComponents = useMemo(() => {
    const startIdx = (componentsPage - 1) * componentsPerPage
    const endIdx = startIdx + componentsPerPage
    return allCriticalComponents.slice(startIdx, endIdx)
  }, [allCriticalComponents, componentsPage, componentsPerPage])

  const totalComponentsPages = Math.ceil(allCriticalComponents.length / componentsPerPage)

  // Paginated critical edges
  const allCriticalEdges = useMemo(() => getCriticalEdges(), [analysisData?.edges])
  
  const paginatedEdges = useMemo(() => {
    const startIdx = (edgesPage - 1) * edgesPerPage
    const endIdx = startIdx + edgesPerPage
    return allCriticalEdges.slice(startIdx, endIdx)
  }, [allCriticalEdges, edgesPage, edgesPerPage])

  const totalEdgesPages = Math.ceil(allCriticalEdges.length / edgesPerPage)

  // Export analysis results as JSON
  const exportAsJSON = () => {
    if (!analysisData) return

    // Format data to match analyze_graph.py export structure
    const exportData = {
      timestamp: new Date().toISOString(),
      layers: {
        [selectedLayer]: {
          layer: selectedLayer,
          layer_name: analysisData.context || 'System Analysis',
          description: analysisData.description || 'Quality analysis results',
          graph_summary: {
            nodes: analysisData.stats?.nodes || analysisData.components?.length || 0,
            edges: analysisData.stats?.edges || analysisData.edges?.length || 0,
            density: analysisData.stats?.density || 0,
            avg_degree: analysisData.stats?.avg_degree || 0,
            avg_clustering: analysisData.stats?.avg_clustering || 0,
            is_connected: analysisData.stats?.is_connected ?? true,
            num_components: analysisData.stats?.num_components || 1,
            num_articulation_points: analysisData.stats?.num_articulation_points || 0,
            num_bridges: analysisData.stats?.num_bridges || 0,
            diameter: analysisData.stats?.diameter || 0,
            avg_path_length: analysisData.stats?.avg_path_length || 0,
          },
          quality_analysis: {
            components: analysisData.components?.map(c => ({
              id: c.id,
              name: c.name,
              type: c.type,
              criticality_level: c.criticality_level,
              criticality_levels: c.criticality_levels || {
                reliability: c.criticality_level,
                maintainability: c.criticality_level,
                availability: c.criticality_level,
                vulnerability: c.criticality_level,
                overall: c.criticality_level,
              },
              scores: c.scores,
            })) || [],
            edges: analysisData.edges?.map(e => ({
              source: e.source,
              target: e.target,
              source_name: e.source_name,
              target_name: e.target_name,
              type: e.type,
              criticality_level: e.criticality_level,
              scores: e.scores,
            })) || [],
            classification_summary: analysisData.summary || {},
          },
          problems: analysisData.problems?.map(p => ({
            entity_id: p.entity_id,
            entity_type: p.type,
            category: p.category,
            severity: p.severity,
            name: p.name,
            description: p.description,
            recommendation: p.recommendation,
          })) || [],
          problem_summary: {
            total_problems: analysisData.problems?.length || 0,
            by_severity: analysisData.problems?.reduce((acc, p) => {
              acc[p.severity] = (acc[p.severity] || 0) + 1
              return acc
            }, {} as Record<string, number>) || {},
            by_category: analysisData.problems?.reduce((acc, p) => {
              acc[p.category] = (acc[p.category] || 0) + 1
              return acc
            }, {} as Record<string, number>) || {},
            requires_attention: analysisData.problems?.filter(p => 
              p.severity === 'CRITICAL' || p.severity === 'HIGH'
            ).length || 0,
          },
        },
      },
      cross_layer_insights: [],
    }

    // Create downloadable file
    const dataStr = JSON.stringify(exportData, null, 2)
    const blob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    
    // Generate filename with timestamp and layer
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5)
    link.download = `analysis-${selectedLayer}-${timestamp}.json`
    
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  // Loading State
  if (!initialLoadComplete || status === 'connecting') {
    return (
      <AppLayout
        title="Analysis"
        description="RMAV quality scoring by architectural layer"
      >
        <div className="space-y-6">
          {/* Layer selection card skeleton */}
          <div className="rounded-xl border border-border bg-muted/20 p-6 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex-1 space-y-3">
                <Skeleton className="h-6 w-6 rounded" />
                <Skeleton className="h-5 w-36" />
                <Skeleton className="h-4 w-64" />
              </div>
              <div className="flex items-center gap-2 shrink-0">
                <Skeleton className="h-8 w-16" />
                <Skeleton className="h-8 w-28" />
              </div>
            </div>
            {/* Layer buttons skeleton */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-1">
              {Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="rounded-lg border border-border p-4 space-y-3">
                  <Skeleton className="h-8 w-8 rounded-lg" />
                  <div className="space-y-1.5">
                    <Skeleton className="h-4 w-24" />
                    <Skeleton className="h-3 w-32" />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </AppLayout>
    )
  }

  // Disconnected State - show only no connection component
  if (!isConnected) {
    return (
      <AppLayout
        title="Analysis"
        description="RMAV quality scoring by architectural layer"
      >
        <NoConnectionInfo description="Connect to your Neo4j database to run quality analysis" />
      </AppLayout>
    )
  }

  return (
    <AppLayout
      title="Analysis"
      description="RMAV quality scoring by architectural layer"
    >
      <div className="space-y-6">

        {/* Layer Selection */}
        <Card className="border-border bg-background">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div>
                <Layers className="h-6 w-6 text-blue-500 mb-3" />
                <CardTitle className="font-semibold text-sm mb-1">
                  Layer Selection
                  <Badge className="ml-2 capitalize">
                    {selectedLayer === 'infrastructure' ? 'Infrastructure' :
                     selectedLayer === 'middleware' ? 'Middleware' :
                     selectedLayer === 'application' ? 'Application' : 'System'}
                  </Badge>
                </CardTitle>
                <CardDescription className="text-sm">
                  Choose an architectural layer to scope the quality analysis, then run it.
                </CardDescription>
              </div>
              <div className="flex items-center gap-2 shrink-0 flex-wrap justify-end">
                {analysisData && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-muted-foreground"
                    onClick={() => { clearAnalysis(getCacheKey()); setError(null) }}
                  >
                    <XCircle className="h-3.5 w-3.5 mr-1" />
                    Clear
                  </Button>
                )}
                <Button
                  onClick={handleAnalyze}
                  disabled={isLoading}
                  size="sm"
                >
                  {isLoading ? (
                    <><LoadingSpinner className="h-4 w-4 mr-2" />Analyzing…</>
                  ) : (
                    <><Activity className="h-4 w-4 mr-2" />Run Analysis</>
                  )}
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">

              {/* System */}
              <button
                onClick={() => setSelectedLayer('system')}
                className={`flex flex-col items-start gap-3 rounded-lg border p-4 text-left transition-all duration-200 cursor-pointer ${
                  selectedLayer === 'system'
                    ? 'border-emerald-500 bg-emerald-500/10 shadow-sm'
                    : 'border-border hover:bg-muted/50 hover:border-muted-foreground/30'
                }`}
              >
                <div className={`rounded-lg p-2 transition-colors ${selectedLayer === 'system' ? 'bg-emerald-500' : 'bg-muted'}`}>
                  <Layers className={`h-4 w-4 ${selectedLayer === 'system' ? 'text-white' : 'text-muted-foreground'}`} />
                </div>
                <div>
                  <div className={`font-semibold text-sm ${selectedLayer === 'system' ? 'text-emerald-700 dark:text-emerald-300' : 'text-foreground'}`}>
                    System
                  </div>
                  <div className="text-xs text-muted-foreground mt-0.5">Cross-layer full view</div>
                </div>
              </button>

              {/* Application */}
              <button
                onClick={() => setSelectedLayer('application')}
                className={`flex flex-col items-start gap-3 rounded-lg border p-4 text-left transition-all duration-200 cursor-pointer ${
                  selectedLayer === 'application'
                    ? 'border-blue-500 bg-blue-500/10 shadow-sm'
                    : 'border-border hover:bg-muted/50 hover:border-muted-foreground/30'
                }`}
              >
                <div className={`rounded-lg p-2 transition-colors ${selectedLayer === 'application' ? 'bg-blue-500' : 'bg-muted'}`}>
                  <Server className={`h-4 w-4 ${selectedLayer === 'application' ? 'text-white' : 'text-muted-foreground'}`} />
                </div>
                <div>
                  <div className={`font-semibold text-sm ${selectedLayer === 'application' ? 'text-blue-700 dark:text-blue-300' : 'text-foreground'}`}>
                    Application
                  </div>
                  <div className="text-xs text-muted-foreground mt-0.5">App-to-app dependencies</div>
                </div>
              </button>

              {/* Infrastructure */}
              <button
                onClick={() => setSelectedLayer('infrastructure')}
                className={`flex flex-col items-start gap-3 rounded-lg border p-4 text-left transition-all duration-200 cursor-pointer ${
                  selectedLayer === 'infrastructure'
                    ? 'border-slate-500 bg-slate-500/10 shadow-sm'
                    : 'border-border hover:bg-muted/50 hover:border-muted-foreground/30'
                }`}
              >
                <div className={`rounded-lg p-2 transition-colors ${selectedLayer === 'infrastructure' ? 'bg-slate-500' : 'bg-muted'}`}>
                  <Database className={`h-4 w-4 ${selectedLayer === 'infrastructure' ? 'text-white' : 'text-muted-foreground'}`} />
                </div>
                <div>
                  <div className={`font-semibold text-sm ${selectedLayer === 'infrastructure' ? 'text-slate-700 dark:text-slate-300' : 'text-foreground'}`}>
                    Infrastructure
                  </div>
                  <div className="text-xs text-muted-foreground mt-0.5">Node-to-node dependencies</div>
                </div>
              </button>

              {/* Middleware */}
              <button
                onClick={() => setSelectedLayer('middleware')}
                className={`flex flex-col items-start gap-3 rounded-lg border p-4 text-left transition-all duration-200 cursor-pointer ${
                  selectedLayer === 'middleware'
                    ? 'border-purple-500 bg-purple-500/10 shadow-sm'
                    : 'border-border hover:bg-muted/50 hover:border-muted-foreground/30'
                }`}
              >
                <div className={`rounded-lg p-2 transition-colors ${selectedLayer === 'middleware' ? 'bg-purple-500' : 'bg-muted'}`}>
                  <Network className={`h-4 w-4 ${selectedLayer === 'middleware' ? 'text-white' : 'text-muted-foreground'}`} />
                </div>
                <div>
                  <div className={`font-semibold text-sm ${selectedLayer === 'middleware' ? 'text-purple-700 dark:text-purple-300' : 'text-foreground'}`}>
                    Middleware
                  </div>
                  <div className="text-xs text-muted-foreground mt-0.5">Broker-to-broker dependencies</div>
                </div>
              </button>

            </div>
          </CardContent>
        </Card>

        {/* Error State */}
        {error && (
          <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-red-500/25 transition-all duration-300">
            {/* Gradient border */}
            <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-red-500 via-rose-500 to-pink-500">
              <div className="w-full h-full bg-background rounded-lg" />
            </div>

            {/* Background gradient overlay */}
            <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-red-500/30 via-red-500/15 to-red-500/5" />

            <CardContent className="p-6 relative">
              <div className="flex items-start gap-4">
                <div className="rounded-xl bg-red-500/10 p-3 flex-shrink-0">
                  <AlertTriangle className="h-6 w-6 text-red-500" />
                </div>
                <div className="flex-1 space-y-3">
                  <div>
                    <h3 className="text-lg font-bold">Analysis Failed</h3>
                    <p className="text-sm text-muted-foreground">Unable to complete analysis</p>
                  </div>
                  <div className="rounded-lg bg-red-500/10 border border-red-500/20 p-3.5">
                    <p className="text-sm text-red-700 dark:text-red-300 font-medium">{error}</p>
                  </div>
                  <Button onClick={handleAnalyze} className="w-full" variant="outline">
                    <Activity className="mr-2 h-4 w-4" />
                    Retry Analysis
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Loading State */}
        {isLoading && (() => {
          const STEPS = [
            { id: "1",   indent: false, label: "Deriving dependency edges from graph relationships",              after: 2  },
            { id: "2",   indent: false, label: "Running structural analysis",                                     after: 7  },
            { id: "2a",  indent: true,  label: "Building layer subgraph",                                         after: 7  },
            { id: "2b",  indent: true,  label: "PageRank & Reverse PageRank",                                     after: 8  },
            { id: "2c",  indent: true,  label: "Betweenness centrality",                                          after: 9  },
            { id: "2d",  indent: true,  label: "Harmonic closeness centrality",                                   after: 11 },
            { id: "2e",  indent: true,  label: "Eigenvector centrality",                                          after: 13 },
            { id: "2f",  indent: true,  label: "MPCI, fan-out criticality, path complexity",                      after: 14 },
            { id: "2g",  indent: true,  label: "Articulation points, blast radius, cascade depth",                after: 15 },
            { id: "2h",  indent: true,  label: "Clustering coefficients and bridges",                             after: 17 },
            { id: "2i",  indent: true,  label: "Edge betweenness centrality",                                     after: 19 },
            { id: "2j",  indent: true,  label: "Pub-sub topology metrics",                                        after: 21 },
            { id: "2k",  indent: true,  label: "Assembling component metrics & code quality normalisation",       after: 22 },
            { id: "2l",  indent: true,  label: "Edge metrics, RCM ordering & graph summary",                      after: 23 },
            { id: "3",   indent: false, label: "Scoring RMAV quality dimensions",                                 after: 24 },
            { id: "4",   indent: false, label: "Detecting architectural anti-patterns",                           after: 27 },
          ]
          const completedCount = STEPS.filter(s => elapsedTime > s.after).length
          const progressValue = Math.min((completedCount / STEPS.length) * 100, 95)
          const activeStep = [...STEPS].reverse().find(s => elapsedTime >= s.after)

          return (
            <div className="rounded-xl border border-border bg-muted/20 px-6 py-5 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <LoadingSpinner className="h-4 w-4" />
                  <span className="text-sm font-medium">
                    {activeStep ? activeStep.label : 'Starting…'}
                  </span>
                </div>
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <Clock className="h-3.5 w-3.5" />
                  <span>{elapsedTime}s</span>
                  <span className="text-muted-foreground/50">·</span>
                  <span>{completedCount}/{STEPS.length}</span>
                  <div className="w-32 ml-1">
                    <Progress value={progressValue} className="h-1.5" />
                  </div>
                </div>
              </div>
              {/* Animated step indicators */}
              <div className="space-y-1 pl-1">
                {STEPS.map(({ id, indent, label, after }) => {
                  const active = elapsedTime >= after
                  const current = elapsedTime >= after && elapsedTime < after + 3
                  return (
                    <div key={id} className={`flex items-center gap-2 text-xs transition-opacity duration-500 ${active ? 'opacity-100' : 'opacity-25'} ${indent ? 'pl-5' : ''}`}>
                      {current ? (
                        <LoadingSpinner className="h-3 w-3 shrink-0 text-blue-500" />
                      ) : active ? (
                        <CheckCircle2 className={`h-3 w-3 shrink-0 ${indent ? 'text-emerald-500' : 'text-green-500'}`} />
                      ) : (
                        <div className={`h-3 w-3 shrink-0 rounded-full border ${indent ? 'border-muted-foreground/25' : 'border-muted-foreground/40'}`} />
                      )}
                      <span className={`${active ? (indent ? 'text-muted-foreground' : 'text-foreground') : 'text-muted-foreground'} ${indent ? '' : 'font-medium'}`}>
                        {indent ? '' : `${id}/4 — `}{label}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>
          )
        })()}

        {/* Analysis Results */}
        {!isLoading && analysisData && (
          <>
            {/* Analysis Log Panel */}
            {analysisData.logs && analysisData.logs.length > 0 && (
              <div className="rounded-xl border border-border bg-muted/10 overflow-hidden">
                <button
                  onClick={() => setLogsOpen(o => !o)}
                  className="w-full flex items-center justify-between px-4 py-2.5 text-left hover:bg-muted/20 transition-colors"
                >
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <Terminal className="h-3.5 w-3.5" />
                    <span className="font-medium">Analysis Log</span>
                    <span className="tabular-nums">({analysisData.logs.length} entries)</span>
                  </div>
                  <ChevronDown className={`h-3.5 w-3.5 text-muted-foreground transition-transform duration-200 ${logsOpen ? 'rotate-180' : ''}`} />
                </button>
                {logsOpen && (
                  <div className="border-t border-border px-4 py-3">
                    <pre className="text-xs font-mono text-muted-foreground whitespace-pre-wrap leading-relaxed max-h-64 overflow-y-auto space-y-0.5">
                      {analysisData.logs.map((line, i) => {
                        const isError = line.startsWith('ERROR')
                        const isWarn = line.startsWith('WARNING')
                        const isStep = /Step \d+\/\d+/.test(line)
                        return (
                          <div
                            key={i}
                            className={`${isError ? 'text-red-500 dark:text-red-400' : isWarn ? 'text-yellow-600 dark:text-yellow-400' : isStep ? 'text-foreground' : ''}`}
                          >
                            {line}
                          </div>
                        )
                      })}
                    </pre>
                  </div>
                )}
              </div>
            )}

            {/* Classification Charts */}
            {analysisData.summary && (() => {
              const CRIT_COLORS: Record<string, string> = {
                critical: '#ef4444',
                high:     '#f97316',
                medium:   '#eab308',
                low:      '#22c55e',
                minimal:  '#64748b',
              }
              const order = ['critical', 'high', 'medium', 'low', 'minimal']
              const TIP_BG   = isDark ? 'rgba(15,23,42,0.92)'       : 'rgba(255,255,255,0.96)'
              const TIP_BORD = isDark ? 'rgba(100,116,139,0.3)'     : 'rgba(100,116,139,0.25)'
              const TIP_TEXT = isDark ? '#e2e8f0'                   : '#0f172a'

              const makeHalfDonut = (data: Record<string, number>, total: number) => ({
                backgroundColor: 'transparent',
                tooltip: {
                  trigger: 'item',
                  backgroundColor: TIP_BG,
                  borderColor: TIP_BORD,
                  textStyle: { color: TIP_TEXT, fontSize: 12 },
                  formatter: (p: { name: string; value: number; percent: number }) =>
                    `${p.name}: <b>${p.value}</b> (${p.percent.toFixed(0)}%)`,
                },
                legend: {
                  orient: 'horizontal',
                  bottom: 0,
                  left: 'center',
                  itemWidth: 8,
                  itemHeight: 8,
                  itemGap: 12,
                  textStyle: { color: '#94a3b8', fontSize: 11 },
                  formatter: (name: string) => `${name}  ${data[name.toLowerCase()] ?? 0}`,
                },
                series: [{
                  type: 'pie',
                  radius: ['52%', '78%'],
                  center: ['50%', '62%'],
                  startAngle: 180,
                  endAngle: 0,
                  data: order.map(level => ({
                    name: formatKey(level),
                    value: (data[level] as number) || 0,
                    itemStyle: { color: CRIT_COLORS[level] },
                  })),
                  label: { show: false },
                  labelLine: { show: false },
                  emphasis: { scale: true, scaleSize: 5, itemStyle: { shadowBlur: 14, shadowColor: 'rgba(0,0,0,0.4)' } },
                  graphic: [
                    { type: 'text', style: { text: String(total), fill: '#f1f5f9', fontSize: 22, fontWeight: 'bold' }, left: 'center', top: '44%' },
                    { type: 'text', style: { text: 'total', fill: '#64748b', fontSize: 11 }, left: 'center', top: '56%' },
                  ],
                }],
              })

              const compData  = analysisData.summary.components || {}
              const compTotal = Object.values(compData).reduce((s: number, v) => s + (v as number), 0)
              const edgeData  = analysisData.summary.edges || {}
              const edgeTotal = Object.values(edgeData).reduce((s: number, v) => s + (v as number), 0)
              const hasEdges  = analysisData.edges && analysisData.edges.length > 0

              const comps = analysisData.components || []
              const avgDim = (key: 'reliability' | 'maintainability' | 'availability' | 'vulnerability') =>
                comps.length ? parseFloat((comps.reduce((s: number, c: any) => s + (c.scores?.[key] ?? 0), 0) / comps.length).toFixed(3)) : 0
              const radarDims = [
                { name: 'Reliability',     value: avgDim('reliability') },
                { name: 'Maintainability', value: avgDim('maintainability') },
                { name: 'Availability',    value: avgDim('availability') },
                { name: 'Vulnerability',   value: avgDim('vulnerability') },
              ]
              const radarMax = parseFloat((Math.max(...radarDims.map(d => d.value)) * 1.25).toFixed(3))
              const radarOption = {
                backgroundColor: 'transparent',
                tooltip: {
                  trigger: 'item',
                  backgroundColor: TIP_BG,
                  borderColor: TIP_BORD,
                  borderWidth: 1,
                  textStyle: { color: TIP_TEXT, fontSize: 12 },
                  extraCssText: 'border-radius:8px;box-shadow:0 4px 16px rgba(0,0,0,0.15);',
                  formatter: (p: any) => radarDims.map((d, i) => `${d.name}: <b>${p.value[i]}</b>`).join('<br/>'),
                },
                radar: {
                  indicator: radarDims.map(d => ({ name: d.name, max: radarMax })),
                  shape: 'polygon',
                  splitNumber: 4,
                  axisName: { color: '#94a3b8', fontSize: 11 },
                  splitLine: { lineStyle: { color: 'rgba(148,163,184,0.15)' } },
                  splitArea: { show: false },
                  axisLine: { lineStyle: { color: 'rgba(148,163,184,0.2)' } },
                },
                series: [{
                  type: 'radar',
                  data: [{
                    value: radarDims.map(d => d.value),
                    name: 'Avg Score',
                    areaStyle: { color: 'rgba(99,102,241,0.15)' },
                    lineStyle: { color: '#6366f1', width: 2 },
                    itemStyle: { color: '#6366f1' },
                  }],
                }],
              }

              return (
                <div className={`grid gap-3 ${hasEdges ? 'md:grid-cols-2 xl:grid-cols-4' : 'md:grid-cols-3'}`}>
                  {/* Graph Summary — 1st card */}
                  {analysisData.stats && (() => {
                    const gs = analysisData.stats
                    const health: string = gs.connectivity_health ?? 'UNKNOWN'
                    const healthColor =
                      health === 'HEALTHY'  ? 'text-green-500'  :
                      health === 'MODERATE' ? 'text-yellow-500' :
                      health === 'AT_RISK'  ? 'text-red-500'    : 'text-muted-foreground'
                    const connectedColor = gs.is_connected ? 'text-green-500' : 'text-red-500'
                    const apColor = gs.num_articulation_points > 0 ? 'text-red-500' : 'text-green-500'
                    const brColor = gs.num_bridges > 0 ? 'text-red-500' : 'text-green-500'
                    const nodeTypes: Record<string, number> = gs.node_types || {}
                    const edgeTypes: Record<string, number> = gs.edge_types || {}
                    return (
                      <Card className="border-border bg-background">
                        <CardHeader className="py-2 px-4 flex flex-row items-center gap-3 space-y-0">
                          <BarChart3 className="h-6 w-6 text-blue-500" />
                          <CardTitle className="text-sm font-semibold">Graph Summary</CardTitle>
                        </CardHeader>
                        <CardContent className="px-4 pb-3 pt-1 space-y-1.5">
                          {[
                            { label: 'Nodes',            value: gs.nodes ?? '—',           cls: '' },
                            { label: 'Edges',            value: gs.edges ?? '—',           cls: '' },
                            { label: 'Density',          value: gs.density != null ? gs.density.toFixed(4) : '—', cls: '' },
                            { label: 'Avg Degree',       value: gs.avg_degree != null ? gs.avg_degree.toFixed(2) : '—', cls: '' },
                            { label: 'Avg Clustering',   value: gs.avg_clustering != null ? gs.avg_clustering.toFixed(4) : '—', cls: '' },
                            { label: 'Connected',        value: gs.is_connected ? 'Yes' : 'No', cls: connectedColor },
                            { label: 'Components',       value: gs.num_components ?? '—', cls: '' },
                            { label: 'Articulation Points', value: gs.num_articulation_points ?? '—', cls: apColor },
                            { label: 'Bridges',          value: gs.num_bridges ?? '—',    cls: brColor },
                            { label: 'Health',           value: health,                   cls: `font-bold ${healthColor}` },
                          ].map(({ label, value, cls }) => (
                            <div key={label} className="flex items-center justify-between text-xs">
                              <span className="text-muted-foreground">{label}</span>
                              <span className={`font-semibold tabular-nums ${cls}`}>{String(value)}</span>
                            </div>
                          ))}
                          {(Object.keys(nodeTypes).length > 0 || Object.keys(edgeTypes).length > 0) && (
                            <div className="pt-3 mt-2 border-t border-border space-y-1">
                              {Object.keys(nodeTypes).length > 0 && (
                                <div className="flex flex-wrap gap-1">
                                  {Object.entries(nodeTypes).map(([t, c]) => (
                                    <Badge key={t} variant="secondary" className="text-[10px] px-1 py-0">{t}: {c}</Badge>
                                  ))}
                                </div>
                              )}
                              {Object.keys(edgeTypes).length > 0 && (
                                <div className="flex flex-wrap gap-1">
                                  {Object.entries(edgeTypes).map(([t, c]) => (
                                    <Badge key={t} variant="secondary" className="text-[10px] px-1 py-0">{t}: {c}</Badge>
                                  ))}
                                </div>
                              )}
                            </div>
                          )}
                        </CardContent>
                      </Card>
                    )
                  })()}

                  {/* Radar chart */}
                  <Card className="border-border bg-background">
                    <CardHeader className="py-2 px-4 flex flex-row items-center gap-3 space-y-0">
                      <Activity className="h-6 w-6 text-indigo-500" />
                      <CardTitle className="text-sm font-semibold">Risk Profile</CardTitle>
                    </CardHeader>
                    <CardContent className="px-2 pb-2 pt-0">
                      <ReactECharts option={radarOption} notMerge style={{ height: 320, width: '100%' }} opts={{ renderer: 'canvas' }} />
                    </CardContent>
                  </Card>

                  {/* Components half-donut */}
                  <Card className="border-border bg-background">
                    <CardHeader className="py-2 px-4 flex flex-row items-center justify-between space-y-0">
                      <div className="flex items-center gap-3">
                        <LayoutGrid className="h-6 w-6 text-blue-500" />
                        <CardTitle className="text-sm font-semibold">Component Classification</CardTitle>
                      </div>
                      <Badge variant="secondary" className="text-xs font-bold">{compTotal}</Badge>
                    </CardHeader>
                    <CardContent className="px-2 pb-1 pt-0">
                      <ReactECharts
                        option={makeHalfDonut(compData, compTotal)}
                        notMerge
                        style={{ height: 320, width: '100%' }}
                        opts={{ renderer: 'canvas' }}
                      />
                    </CardContent>
                  </Card>

                  {/* Connections half-donut */}
                  {hasEdges && (
                    <Card className="border-border bg-background">
                      <CardHeader className="py-2 px-4 flex flex-row items-center justify-between space-y-0">
                        <div className="flex items-center gap-3">
                          <Share2 className="h-6 w-6 text-purple-500" />
                          <CardTitle className="text-sm font-semibold">Connection Classification</CardTitle>
                        </div>
                        <Badge variant="secondary" className="text-xs font-bold">{edgeTotal}</Badge>
                      </CardHeader>
                      <CardContent className="px-2 pb-1 pt-0">
                        <ReactECharts
                          option={makeHalfDonut(edgeData, edgeTotal)}
                          notMerge
                          style={{ height: 320, width: '100%' }}
                          opts={{ renderer: 'canvas' }}
                        />
                      </CardContent>
                    </Card>
                  )}
                </div>
              )
            })()}

            {(analysisData?.components?.length ?? 0) > 0 && (

              <div className="space-y-2">

                {/* Component filters */}
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide mr-1">Level</span>
                  {['all', ...availableCompLevels].map(l => {
                    const count = l === 'all' ? (analysisData?.components?.length ?? 0) : (analysisData?.components?.filter((c: any) => c.criticality_level === l).length ?? 0)
                    const active = compLevelFilter === l
                    const colors: Record<string, string> = { critical: 'bg-red-500 text-white', high: 'bg-orange-500 text-white', medium: 'bg-yellow-500 text-white', low: 'bg-green-500 text-white', minimal: 'bg-slate-500 text-white' }
                    return (
                      <button key={l} onClick={() => setCompLevelFilter(l)}
                        className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${active ? (colors[l] ?? 'bg-foreground text-background') : 'bg-muted text-muted-foreground hover:text-foreground'}`}>
                        {l === 'all' ? 'All' : formatKey(l)} ({count})
                      </button>
                    )
                  })}
                  <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide mx-1">Type</span>
                  {['all', ...availableCompTypes].map(t => {
                    const active = compTypeFilter === t
                    return (
                      <button key={t} onClick={() => setCompTypeFilter(t)}
                        className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${active ? 'bg-foreground text-background' : 'bg-muted text-muted-foreground hover:text-foreground'}`}>
                        {t === 'all' ? 'All' : t}
                      </button>
                    )
                  })}
                  <div className="relative ml-auto">
                    <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />
                    <Input placeholder="Search..." value={compSearchQuery} onChange={e => setCompSearchQuery(e.target.value)}
                      className="h-7 pl-7 pr-7 text-xs w-48 bg-muted border-border" />
                    {compSearchQuery && (
                      <button onClick={() => setCompSearchQuery('')} className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground">
                        <X className="h-3.5 w-3.5" />
                      </button>
                    )}
                  </div>
                </div>

                <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  {allCriticalComponents.length === (analysisData?.components?.length ?? 0)
                    ? `All ${allCriticalComponents.length} Components Ranked by Criticality`
                    : `${allCriticalComponents.length} of ${analysisData?.components?.length ?? 0} Components`}
                </p>

                <div className="rounded-lg border border-border">

                  <table className="w-full text-sm">

                    <thead>

                      <tr className="border-b border-border">

                        <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground uppercase tracking-wide w-8">#</th>

                        <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground uppercase tracking-wide">Component</th>

                        <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground uppercase tracking-wide">Type</th>

                        <th className="px-3 py-2 text-center text-xs font-medium text-muted-foreground uppercase tracking-wide">Level</th>

                        <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground uppercase tracking-wide">Overall</th>
                        <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground uppercase tracking-wide">Reliability</th>
                        <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground uppercase tracking-wide">Maintainability</th>
                        <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground uppercase tracking-wide">Availability</th>
                        <th className="px-3 py-2 text-right text-xs font-medium text-muted-foreground uppercase tracking-wide">Vulnerability</th>

                      </tr>

                    </thead>

                    <tbody>

                      {paginatedComponents.length === 0 ? (
                        <tr><td colSpan={9} className="px-3 py-6 text-center text-xs text-muted-foreground">No components match the current filters.</td></tr>
                      ) : paginatedComponents.map((component, idx) => (

                        <tr

                          key={component.id}

                          className="border-b border-border/50 hover:bg-muted/30 transition-colors cursor-pointer"

                          onClick={() => router.push(`/explorer?node=${encodeURIComponent(component.id)}`)}

                        >

                          <td className="px-3 py-2 text-xs text-muted-foreground">{(componentsPage - 1) * componentsPerPage + idx + 1}</td>

                          <td className="px-3 py-2 font-medium">{component.name || component.id}</td>

                          <td className="px-3 py-2 text-xs text-muted-foreground">{component.type}</td>

                          <td className="px-3 py-2 text-center">

                            <Badge className={`text-xs font-semibold capitalize ${getCriticalityColor(component.criticality_level)}`}>

                              {component.criticality_level}

                            </Badge>

                          </td>

                          <td className="px-3 py-2 text-right font-mono text-sm font-semibold">
                            <span className={getScoreColor(component.scores.overall)}>{component.scores.overall.toFixed(3)}</span>
                          </td>

                          <td className="px-3 py-2 text-right font-mono text-xs">
                            <span className={getScoreColor(component.scores.reliability)}>{component.scores.reliability.toFixed(3)}</span>
                          </td>

                          <td className="px-3 py-2 text-right font-mono text-xs">
                            <span className={getScoreColor(component.scores.maintainability)}>{component.scores.maintainability.toFixed(3)}</span>
                          </td>

                          <td className="px-3 py-2 text-right font-mono text-xs">
                            <span className={getScoreColor(component.scores.availability)}>{component.scores.availability.toFixed(3)}</span>
                          </td>

                          <td className="px-3 py-2 text-right font-mono text-xs">
                            <span className={getScoreColor(component.scores.vulnerability)}>{component.scores.vulnerability.toFixed(3)}</span>
                          </td>

                        </tr>

                      ))}

                    </tbody>

                  </table>

                </div>

                {/* Pagination */}
                {totalComponentsPages > 1 && (
                  <div className="flex items-center justify-between text-xs text-muted-foreground pt-1">
                    <span>Showing {(componentsPage - 1) * componentsPerPage + 1}–{Math.min(componentsPage * componentsPerPage, allCriticalComponents.length)} of {allCriticalComponents.length} components</span>
                    <div className="flex items-center gap-1">
                      <button onClick={() => setComponentsPage(1)} disabled={componentsPage === 1} className="p-1 rounded hover:bg-muted disabled:opacity-30"><ChevronsLeft className="h-3.5 w-3.5" /></button>
                      <button onClick={() => setComponentsPage(p => Math.max(1, p - 1))} disabled={componentsPage === 1} className="p-1 rounded hover:bg-muted disabled:opacity-30"><ChevronLeft className="h-3.5 w-3.5" /></button>
                      <span className="px-2">{componentsPage} / {totalComponentsPages}</span>
                      <button onClick={() => setComponentsPage(p => Math.min(totalComponentsPages, p + 1))} disabled={componentsPage === totalComponentsPages} className="p-1 rounded hover:bg-muted disabled:opacity-30"><ChevronRight className="h-3.5 w-3.5" /></button>
                      <button onClick={() => setComponentsPage(totalComponentsPages)} disabled={componentsPage === totalComponentsPages} className="p-1 rounded hover:bg-muted disabled:opacity-30"><ChevronsRight className="h-3.5 w-3.5" /></button>
                    </div>
                  </div>
                )}

              </div>

            )}

            {/* Identified Issues */}

            {analysisData.problems && analysisData.problems.length > 0 && (

              <div className="space-y-3">

                {/* Filters */}

                <div className="flex flex-wrap items-center gap-2">

                  <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide mr-1">Severity</span>

                  {['all', ...availableSeverities].map(s => {

                    const count = s === 'all' ? analysisData.problems.length : analysisData.problems.filter((p: any) => p.severity.toLowerCase() === s.toLowerCase()).length

                    const active = severityFilter === s

                    const colors: Record<string,string> = { critical:'bg-red-500 text-white', high:'bg-orange-500 text-white', medium:'bg-yellow-500 text-white', low:'bg-green-500 text-white' }

                    return (

                      <button key={s} onClick={() => setSeverityFilter(s)}

                        className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${active ? (colors[s] ?? 'bg-foreground text-background') : 'bg-muted text-muted-foreground hover:text-foreground'}`}>

                        {s === 'all' ? 'All' : s} ({count})

                      </button>

                    )

                  })}

                  <span className="text-xs font-medium text-muted-foreground uppercase tracking-wide mx-1">Category</span>

                  {['all', ...availableCategories].map(c => {

                    const active = categoryFilter === c

                    return (

                      <button key={c} onClick={() => setCategoryFilter(c)}

                        className={`px-2.5 py-1 rounded text-xs font-medium transition-colors ${active ? 'bg-foreground text-background' : 'bg-muted text-muted-foreground hover:text-foreground'}`}>

                        {c === 'all' ? 'All' : formatKey(c)}

                      </button>

                    )

                  })}

                  <div className="relative ml-auto">

                    <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground pointer-events-none" />

                    <Input placeholder="Search..." value={searchQuery} onChange={e => setSearchQuery(e.target.value)}

                      className="h-7 pl-7 pr-7 text-xs w-48 bg-muted border-border" />

                    {searchQuery && (

                      <button onClick={() => setSearchQuery('')} className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground">

                        <X className="h-3.5 w-3.5" />

                      </button>

                    )}

                  </div>

                </div>



                {/* Table */}

                <div>

                  <p className="text-xs font-medium text-muted-foreground uppercase tracking-wide mb-2">

                    {filteredIssues.length === analysisData.problems.length

                      ? `${analysisData.problems.length} issue${analysisData.problems.length !== 1 ? 's' : ''} detected`

                      : `${filteredIssues.length} of ${analysisData.problems.length} issues`}

                  </p>

                  <div className="rounded-lg border border-border">

                    {filteredIssues.length === 0 ? (

                      <div className="py-12 text-center text-sm text-muted-foreground">No issues match the current filters.</div>

                    ) : (

                      <table className="w-full text-sm">

                        <thead>

                          <tr className="border-b border-border">

                            <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground uppercase tracking-wide w-8">#</th>

                            <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground uppercase tracking-wide w-24">Severity</th>

                            <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground uppercase tracking-wide w-28">Category</th>

                            <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground uppercase tracking-wide w-40">Component</th>

                            <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground uppercase tracking-wide">Issue</th>

                            <th className="px-3 py-2 text-left text-xs font-medium text-muted-foreground uppercase tracking-wide">Recommendation</th>

                          </tr>

                        </thead>

                        <tbody>

                          {paginatedIssues.map((problem: any, idx: number) => {

                            const globalIdx = (issuesPage - 1) * issuesPerPage + idx

                            const sev = problem.severity.toLowerCase()

                            const sevColor: Record<string,string> = { critical:'bg-red-500 text-white', high:'bg-orange-500 text-white', medium:'bg-yellow-500 text-white', low:'bg-green-500 text-white' }

                            const isLink = problem.entity_id.includes('->')

                            let componentCell: React.ReactNode

                            if (isLink) {

                              const nodeIds = problem.entity_id.split('->').map((s: string) => s.trim())

                              const nodeNames = nodeIds.map((id: string) => analysisData.components?.find((c: any) => c.id === id)?.name || id)

                              componentCell = (

                                <span className="font-mono text-xs text-muted-foreground flex flex-wrap items-center gap-0.5">

                                  {nodeIds.map((id: string, i: number) => (
                                    <span key={id} className="flex items-center gap-0.5">
                                      {i > 0 && <span className="mx-0.5">{String.fromCharCode(0x2192)}</span>}
                                      <button onClick={() => router.push(`/explorer?node=${encodeURIComponent(id)}`)} className="hover:underline hover:text-foreground">{nodeNames[i]}</button>
                                    </span>
                                  ))}

                                </span>

                              )} else {

                              const comp = analysisData.components?.find((c: any) => c.id === problem.entity_id)

                              const label = comp?.name || problem.entity_id

                              componentCell = (

                                <button onClick={() => router.push(`/explorer?node=${encodeURIComponent(problem.entity_id)}`)}

                                  className="font-medium hover:underline hover:text-foreground text-left truncate max-w-[148px] block" title={label}>

                                  {label}

                                </button>

                              )

                            }

                            return (

                              <tr key={`${problem.entity_id}-${idx}`} className="border-b border-border/50 hover:bg-muted/30 transition-colors">

                                <td className="px-3 py-2 text-xs text-muted-foreground">{globalIdx + 1}</td>

                                <td className="px-3 py-2">

                                  <span className={`px-2 py-0.5 rounded text-xs font-semibold capitalize ${sevColor[sev] ?? 'bg-muted text-muted-foreground'}`}>{problem.severity}</span>

                                </td>

                                <td className="px-3 py-2 text-xs text-muted-foreground">{formatKey(problem.category)}</td>

                                <td className="px-3 py-2">{componentCell}</td>

                                <td className="px-3 py-2">

                                  <p className="font-medium text-xs mb-0.5">{problem.name}</p>

                                  <p className="text-xs text-muted-foreground line-clamp-2">{problem.description}</p>

                                </td>

                                <td className="px-3 py-2 text-xs text-muted-foreground line-clamp-2">{problem.recommendation}</td>

                              </tr>

                            )

                          })}

                        </tbody>

                      </table>

                    )}

                  </div>

                </div>



                {/* Pagination */}
                {totalIssuesPages > 1 && (
                  <div className="flex items-center justify-between text-xs text-muted-foreground pt-1">
                    <span>Showing {(issuesPage - 1) * issuesPerPage + 1}–{Math.min(issuesPage * issuesPerPage, filteredIssues.length)} of {filteredIssues.length} issues</span>
                    <div className="flex items-center gap-1">
                      <button onClick={() => setIssuesPage(1)} disabled={issuesPage === 1} className="p-1 rounded hover:bg-muted disabled:opacity-30"><ChevronsLeft className="h-3.5 w-3.5" /></button>
                      <button onClick={() => setIssuesPage(p => p - 1)} disabled={issuesPage === 1} className="p-1 rounded hover:bg-muted disabled:opacity-30"><ChevronLeft className="h-3.5 w-3.5" /></button>
                      <span className="px-2">{issuesPage} / {totalIssuesPages}</span>
                      <button onClick={() => setIssuesPage(p => p + 1)} disabled={issuesPage === totalIssuesPages} className="p-1 rounded hover:bg-muted disabled:opacity-30"><ChevronRight className="h-3.5 w-3.5" /></button>
                      <button onClick={() => setIssuesPage(totalIssuesPages)} disabled={issuesPage === totalIssuesPages} className="p-1 rounded hover:bg-muted disabled:opacity-30"><ChevronsRight className="h-3.5 w-3.5" /></button>
                    </div>
                  </div>
                )}

              </div>

            )}
            {(!analysisData.problems || analysisData.problems.length === 0) && (
              <div>
                <div className="mb-4 flex items-center gap-3">
                  <div className="rounded-xl bg-gradient-to-br from-green-500 to-emerald-600 p-2.5 shadow-md">
                    <CheckCircle2 className="h-5 w-5 text-white" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold">All Clear!</h2>
                    <p className="text-sm text-muted-foreground">No issues detected - your system health is excellent</p>
                  </div>
                </div>
                <Card className="border-green-200 dark:border-green-900 bg-gradient-to-br from-green-50/50 to-white dark:from-green-950/20 dark:to-background hover:shadow-md transition-all">
                  <CardContent className="py-12">
                    <div className="text-center">
                      <div className="rounded-full bg-green-100 dark:bg-green-900 p-4 w-fit mx-auto mb-4">
                        <CheckCircle2 className="h-12 w-12 text-green-600 dark:text-green-400" />
                      </div>
                      <h3 className="text-xl font-semibold mb-2">System Operating Optimally</h3>
                      <p className="text-sm text-muted-foreground max-w-md mx-auto">
                        All components and connections have passed quality checks. Your distributed system is healthy and well-architected.
                      </p>
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
