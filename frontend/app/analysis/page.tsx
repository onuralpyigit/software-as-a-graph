"use client"

import React, { useState, useEffect, useMemo } from "react"
import { useRouter } from "next/navigation"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Input } from "@/components/ui/input"
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Loader2,
  Settings,
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
  ChevronDown,
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
  Download
} from "lucide-react"
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { useConnection } from "@/lib/stores/connection-store"
import { useAnalysis } from "@/lib/stores/analysis-store"
import { apiClient } from "@/lib/api/client"

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
}

type AnalysisMode = 'full' | 'type' | 'layer'

export default function AnalysisPage() {
  const router = useRouter()
  const { status, stats, initialLoadComplete } = useConnection()
  const { getAnalysis, setAnalysis, clearAnalysis } = useAnalysis()
  
  // Analysis mode and filters
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>('full')
  const [selectedType, setSelectedType] = useState<string>('Application')
  const [selectedLayer, setSelectedLayer] = useState<string>('application')
  
  // Loading and error states
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [elapsedTime, setElapsedTime] = useState(0)
  const [startTime, setStartTime] = useState<number | null>(null)
  
  // Estimated duration for progress calculation (in seconds)
  const estimatedDuration = 30

  // Issues pagination and filtering
  const [issuesPage, setIssuesPage] = useState(1)
  const [issuesPerPage] = useState(10)
  const [severityFilter, setSeverityFilter] = useState<string>('all')
  const [categoryFilter, setCategoryFilter] = useState<string>('all')
  const [searchQuery, setSearchQuery] = useState<string>('')

  // Critical components pagination
  const [componentsPage, setComponentsPage] = useState(1)
  const [componentsPerPage] = useState(10)

  // Critical edges pagination
  const [edgesPage, setEdgesPage] = useState(1)
  const [edgesPerPage] = useState(10)

  const isConnected = status === 'connected'

  // Generate cache key for current analysis configuration
  const getCacheKey = () => {
    if (analysisMode === 'full') return 'full'
    if (analysisMode === 'type') return `type:${selectedType}`
    return `layer:${selectedLayer}`
  }

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
      
      // Run ONLY the selected analysis mode
      if (analysisMode === 'full') {
        console.log('Running full system analysis...')
        response = await apiClient.analyzeFullSystem()
      } else if (analysisMode === 'type') {
        console.log(`Running analysis for component type: ${selectedType}`)
        response = await apiClient.analyzeByType(selectedType)
      } else if (analysisMode === 'layer') {
        console.log(`Running analysis for layer: ${selectedLayer}`)
        response = await apiClient.analyzeByLayer(selectedLayer)
      } else {
        throw new Error('Invalid analysis mode selected')
      }

      if (!response) {
        throw new Error('No response from analysis')
      }

      if (response.success && response.analysis) {
        console.log(`Analysis complete. Found ${response.analysis.components?.length || 0} components`)
        // Cache the result with the current configuration key
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
    // Scores are 0-1, convert to 0-100 for display
    const displayScore = score * 100
    if (displayScore >= 80) return "text-red-600 dark:text-red-500" // High score = bad (high risk)
    if (displayScore >= 60) return "text-yellow-600 dark:text-yellow-500"
    return "text-green-600 dark:text-green-500" // Low score = good (low risk)
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

    const reliability = (allEntities.reduce((sum, e) => sum + e.scores.reliability, 0) / allEntities.length) * 100
    const maintainability = (allEntities.reduce((sum, e) => sum + e.scores.maintainability, 0) / allEntities.length) * 100
    const availability = (allEntities.reduce((sum, e) => sum + e.scores.availability, 0) / allEntities.length) * 100
    const vulnerability = (allEntities.reduce((sum, e) => sum + e.scores.vulnerability, 0) / allEntities.length) * 100
    const overall = (allEntities.reduce((sum, e) => sum + e.scores.overall, 0) / allEntities.length) * 100

    return { reliability, maintainability, availability, vulnerability, overall }
  }

  // Get critical components
  const getCriticalComponents = () => {
    if (!analysisData?.components) return []
    return analysisData.components
      .filter(c => c.criticality_level === 'critical' || c.criticality_level === 'high')
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
          // For edges, search in both source and target component names
          const [sourceId, targetId] = p.entity_id.split('->').map(s => s.trim())
          const edge = analysisData.edges?.find(e => `${e.source}->${e.target}` === p.entity_id)
          
          if (edge) {
            // Search in edge source_name and target_name
            return (
              (edge.source_name && edge.source_name.toLowerCase().includes(query)) ||
              (edge.target_name && edge.target_name.toLowerCase().includes(query))
            )
          } else {
            // Search in component names from components array
            const sourceComponent = analysisData.components?.find(c => c.id === sourceId)
            const targetComponent = analysisData.components?.find(c => c.id === targetId)
            return (
              (sourceComponent?.name && sourceComponent.name.toLowerCase().includes(query)) ||
              (targetComponent?.name && targetComponent.name.toLowerCase().includes(query))
            )
          }
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

  // Paginated critical components
  const allCriticalComponents = useMemo(() => getCriticalComponents(), [analysisData?.components])
  
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
        [analysisMode === 'layer' ? selectedLayer : 'system']: {
          layer: analysisMode === 'layer' ? selectedLayer : 'system',
          layer_name: analysisData.context || (analysisMode === 'type' ? `Type: ${selectedType}` : 'System Analysis'),
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
    
    // Generate filename with timestamp and analysis mode
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5)
    const modeLabel = analysisMode === 'type' ? selectedType.toLowerCase() : 
                      analysisMode === 'layer' ? selectedLayer : 'full-system'
    link.download = `analysis-${modeLabel}-${timestamp}.json`
    
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  // Loading State
  if (!initialLoadComplete || status === 'connecting') {
    return (
      <AppLayout
        title="Quality Analysis"
        description="Component-based quality attribute assessment"
      >
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" text={status === 'connecting' ? "Connecting to database..." : "Loading analysis..."} />
        </div>
      </AppLayout>
    )
  }

  // Disconnected State - show only no connection component
  if (!isConnected) {
    return (
      <AppLayout
        title="Quality Analysis"
        description="Component-based quality attribute assessment"
      >
        <NoConnectionInfo description="Connect to your Neo4j database to run quality analysis" />
      </AppLayout>
    )
  }

  // Empty database state
  if (stats && stats.total_nodes === 0) {
    return (
      <AppLayout
        title="Quality Analysis"
        description="Component-based quality attribute assessment"
      >
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
                    No graph data available
                  </CardDescription>
                </div>
              </div>
            </CardHeader>

            <CardContent className="px-8 pb-8 space-y-6">
              {/* Information box with steps */}
              <div className="rounded-2xl bg-gradient-to-br from-muted/40 via-muted/20 to-muted/10 border border-border/40 p-6 space-y-5">
                <div className="flex items-center gap-2.5">
                  <div className="rounded-xl bg-gradient-to-br from-blue-500/20 to-blue-600/20 dark:from-blue-500/30 dark:to-blue-600/30 p-2.5 ring-1 ring-blue-500/30">
                    <Info className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                  </div>
                  <h3 className="font-semibold text-base text-foreground">How to Populate Your Database</h3>
                </div>

                {/* Step-by-step list */}
                <div className="space-y-3 pl-1">
                  <div className="flex items-start gap-3.5 group/item">
                    <div className="flex-shrink-0 w-7 h-7 rounded-full bg-gradient-to-br from-purple-500/20 to-purple-600/20 dark:from-purple-500/25 dark:to-purple-600/25 flex items-center justify-center ring-1 ring-purple-500/30 group-hover/item:ring-purple-500/50 transition-all duration-200">
                      <span className="text-xs font-bold text-purple-600 dark:text-purple-400">1</span>
                    </div>
                    <p className="text-sm text-muted-foreground leading-relaxed pt-0.5 group-hover/item:text-foreground transition-colors duration-200">
                      Navigate to the Data Management page
                    </p>
                  </div>
                  <div className="flex items-start gap-3.5 group/item">
                    <div className="flex-shrink-0 w-7 h-7 rounded-full bg-gradient-to-br from-purple-500/20 to-purple-600/20 dark:from-purple-500/25 dark:to-purple-600/25 flex items-center justify-center ring-1 ring-purple-500/30 group-hover/item:ring-purple-500/50 transition-all duration-200">
                      <span className="text-xs font-bold text-purple-600 dark:text-purple-400">2</span>
                    </div>
                    <p className="text-sm text-muted-foreground leading-relaxed pt-0.5 group-hover/item:text-foreground transition-colors duration-200">
                      Upload your system architecture configuration files
                    </p>
                  </div>
                  <div className="flex items-start gap-3.5 group/item">
                    <div className="flex-shrink-0 w-7 h-7 rounded-full bg-gradient-to-br from-purple-500/20 to-purple-600/20 dark:from-purple-500/25 dark:to-purple-600/25 flex items-center justify-center ring-1 ring-purple-500/30 group-hover/item:ring-purple-500/50 transition-all duration-200">
                      <span className="text-xs font-bold text-purple-600 dark:text-purple-400">3</span>
                    </div>
                    <p className="text-sm text-muted-foreground leading-relaxed pt-0.5 group-hover/item:text-foreground transition-colors duration-200">
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
                  className="w-full bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white shadow-lg hover:shadow-xl transition-all duration-300 group"
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

  return (
    <AppLayout
      title="Quality Analysis"
      description="Component-based quality attribute assessment"
    >
      <div className="space-y-6">

        {/* Analysis Configuration */}
        <div className="space-y-4">
          {/* Header Section */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-gradient-to-br from-blue-600 via-purple-600 to-pink-600 p-3 shadow-lg">
                <Settings className="h-6 w-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold">Analysis Configuration</h2>
                <p className="text-sm text-muted-foreground">Select your analysis scope and run quality assessment</p>
              </div>
            </div>
            <Button
              onClick={handleAnalyze}
              disabled={isLoading}
              size="lg"
              className="min-w-[180px] h-12 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 hover:from-blue-700 hover:via-purple-700 hover:to-pink-700 shadow-lg hover:shadow-xl transition-all text-base font-semibold"
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Activity className="mr-2 h-5 w-5" />
                  Run Analysis
                </>
              )}
            </Button>
          </div>

          {/* Analysis Mode Selection */}
          <div className="grid gap-6 md:grid-cols-3">
            {/* Full System Card */}
            <Card
              className={`group relative cursor-pointer transition-all duration-300 ease-in-out overflow-hidden ${
                analysisMode === 'full'
                  ? 'border-0 shadow-2xl shadow-green-500/25 scale-[1.02]'
                  : 'border-0 hover:shadow-xl hover:shadow-green-500/25 hover:scale-[1.01]'
              }`}
              onClick={() => setAnalysisMode('full')}
            >
              {/* Gradient border */}
              <div className={`absolute inset-0 rounded-lg p-[2px] transition-opacity duration-300 ${
                analysisMode === 'full'
                  ? 'bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500 opacity-100'
                  : 'bg-gradient-to-br from-slate-200 via-slate-300 to-slate-200 dark:from-slate-700 dark:via-slate-800 dark:to-slate-700 opacity-100 group-hover:from-green-500 group-hover:via-emerald-500 group-hover:to-teal-500'
              }`}>
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              
              {/* Background gradient overlay */}
              <div className={`absolute inset-[2px] rounded-lg transition-opacity duration-300 ${
                analysisMode === 'full'
                  ? 'bg-gradient-to-br from-green-500/15 via-emerald-500/10 to-teal-500/5 opacity-100'
                  : 'bg-gradient-to-br from-green-500/5 via-emerald-500/3 to-transparent opacity-0 group-hover:opacity-100'
              }`} />
              
              <CardContent className="relative p-7">
                <div className="space-y-5">
                  {/* Radio button and Icon Section */}
                  <div className="flex items-start gap-4">
                    {/* Radio indicator */}
                    <div className={`relative flex-shrink-0 w-5 h-5 rounded-full border-2 transition-all duration-300 mt-1 ${
                      analysisMode === 'full'
                        ? 'border-green-500 dark:border-green-400'
                        : 'border-slate-300 dark:border-slate-600 group-hover:border-green-400'
                    }`}>
                      {analysisMode === 'full' && (
                        <div className="absolute inset-0.5 rounded-full bg-green-500 dark:bg-green-400 animate-in zoom-in duration-200" />
                      )}
                    </div>
                    
                    {/* Icon */}
                    <div className={`relative rounded-2xl p-3.5 transition-all duration-300 ${
                      analysisMode === 'full'
                        ? 'bg-gradient-to-br from-green-500 to-emerald-600 shadow-lg shadow-green-500/30'
                        : 'bg-gradient-to-br from-green-100 to-emerald-50 dark:from-green-900/50 dark:to-emerald-900/30 group-hover:scale-105'
                    }`}>
                      <Network className={`h-6 w-6 transition-all duration-300 ${
                        analysisMode === 'full'
                          ? 'text-white'
                          : 'text-green-600 dark:text-green-400'
                      }`} />
                      {analysisMode === 'full' && (
                        <div className="absolute inset-0 rounded-2xl bg-green-500 animate-ping opacity-20" />
                      )}
                    </div>
                  </div>
                  
                  {/* Content Section */}
                  <div className="space-y-2.5 pl-9">
                    <div className="flex items-center gap-2">
                      <h3 className={`text-lg font-bold tracking-tight transition-colors duration-200 ${
                        analysisMode === 'full'
                          ? 'text-green-700 dark:text-green-300'
                          : 'text-foreground group-hover:text-green-700 dark:group-hover:text-green-300'
                      }`}>
                        Full System
                      </h3>
                      {analysisMode === 'full' && (
                        <Badge className="bg-green-500/10 text-green-700 dark:text-green-300 border-green-500/20 text-xs">
                          Selected
                        </Badge>
                      )}
                    </div>
                    <p className="text-sm leading-relaxed text-muted-foreground/90">
                      Comprehensive analysis of all components and dependencies in your system
                    </p>
                    {!analysisMode || analysisMode !== 'full' ? (
                      <p className="text-xs text-muted-foreground/60 flex items-center gap-1 mt-2">
                        <Info className="h-3 w-3" />
                        Click to select this scope
                      </p>
                    ) : null}
                  </div>
                  
                  {/* Selection indicator bar */}
                  {analysisMode === 'full' && (
                    <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-green-500 to-emerald-600" />
                  )}
                </div>
              </CardContent>
            </Card>

            {/* By Component Type Card */}
            <Card
              className={`group relative cursor-pointer transition-all duration-300 ease-in-out overflow-hidden ${
                analysisMode === 'type'
                  ? 'border-0 shadow-2xl shadow-blue-500/25 scale-[1.02]'
                  : 'border-0 hover:shadow-xl hover:shadow-blue-500/25 hover:scale-[1.01]'
              }`}
              onClick={() => setAnalysisMode('type')}
            >
              {/* Gradient border */}
              <div className={`absolute inset-0 rounded-lg p-[2px] transition-opacity duration-300 ${
                analysisMode === 'type'
                  ? 'bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 opacity-100'
                  : 'bg-gradient-to-br from-slate-200 via-slate-300 to-slate-200 dark:from-slate-700 dark:via-slate-800 dark:to-slate-700 opacity-100 group-hover:from-blue-500 group-hover:via-indigo-500 group-hover:to-purple-500'
              }`}>
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              
              {/* Background gradient overlay */}
              <div className={`absolute inset-[2px] rounded-lg transition-opacity duration-300 ${
                analysisMode === 'type'
                  ? 'bg-gradient-to-br from-blue-500/15 via-indigo-500/10 to-violet-500/5 opacity-100'
                  : 'bg-gradient-to-br from-blue-500/5 via-indigo-500/3 to-transparent opacity-0 group-hover:opacity-100'
              }`} />
              
              <CardContent className="relative p-7">
                <div className="space-y-5">
                  {/* Radio button and Icon Section */}
                  <div className="flex items-start gap-4">
                    {/* Radio indicator */}
                    <div className={`relative flex-shrink-0 w-5 h-5 rounded-full border-2 transition-all duration-300 mt-1 ${
                      analysisMode === 'type'
                        ? 'border-blue-500 dark:border-blue-400'
                        : 'border-slate-300 dark:border-slate-600 group-hover:border-blue-400'
                    }`}>
                      {analysisMode === 'type' && (
                        <div className="absolute inset-0.5 rounded-full bg-blue-500 dark:bg-blue-400 animate-in zoom-in duration-200" />
                      )}
                    </div>
                    
                    {/* Icon */}
                    <div className={`relative rounded-2xl p-3.5 transition-all duration-300 ${
                      analysisMode === 'type'
                        ? 'bg-gradient-to-br from-blue-500 to-indigo-600 shadow-lg shadow-blue-500/30'
                        : 'bg-gradient-to-br from-blue-100 to-indigo-50 dark:from-blue-900/50 dark:to-indigo-900/30 group-hover:scale-105'
                    }`}>
                      <Box className={`h-6 w-6 transition-all duration-300 ${
                        analysisMode === 'type'
                          ? 'text-white'
                          : 'text-blue-600 dark:text-blue-400'
                      }`} />
                      {analysisMode === 'type' && (
                        <div className="absolute inset-0 rounded-2xl bg-blue-500 animate-ping opacity-20" />
                      )}
                    </div>
                  </div>
                  
                  {/* Content Section */}
                  <div className="space-y-2.5 pl-9">
                    <div className="flex items-center gap-2">
                      <h3 className={`text-lg font-bold tracking-tight transition-colors duration-200 ${
                        analysisMode === 'type'
                          ? 'text-blue-700 dark:text-blue-300'
                          : 'text-foreground group-hover:text-blue-700 dark:group-hover:text-blue-300'
                      }`}>
                        By Component Type
                      </h3>
                      {analysisMode === 'type' && (
                        <Badge className="bg-blue-500/10 text-blue-700 dark:text-blue-300 border-blue-500/20 text-xs">
                          Selected
                        </Badge>
                      )}
                    </div>
                    <p className="text-sm leading-relaxed text-muted-foreground/90">
                      Focus analysis on specific component types like applications or brokers
                    </p>
                    {!analysisMode || analysisMode !== 'type' ? (
                      <p className="text-xs text-muted-foreground/60 flex items-center gap-1 mt-2">
                        <Info className="h-3 w-3" />
                        Click to select this scope
                      </p>
                    ) : null}
                  </div>
                  
                  {/* Selection indicator bar */}
                  {analysisMode === 'type' && (
                    <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-blue-500 to-indigo-600" />
                  )}
                </div>
              </CardContent>
            </Card>

            {/* By Layer Card */}
            <Card
              className={`group relative cursor-pointer transition-all duration-300 ease-in-out overflow-hidden ${
                analysisMode === 'layer'
                  ? 'border-0 shadow-2xl shadow-purple-500/25 scale-[1.02]'
                  : 'border-0 hover:shadow-xl hover:shadow-purple-500/25 hover:scale-[1.01]'
              }`}
              onClick={() => setAnalysisMode('layer')}
            >
              {/* Gradient border */}
              <div className={`absolute inset-0 rounded-lg p-[2px] transition-opacity duration-300 ${
                analysisMode === 'layer'
                  ? 'bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 opacity-100'
                  : 'bg-gradient-to-br from-slate-200 via-slate-300 to-slate-200 dark:from-slate-700 dark:via-slate-800 dark:to-slate-700 opacity-100 group-hover:from-blue-500 group-hover:via-purple-500 group-hover:to-pink-500'
              }`}>
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              
              {/* Background gradient overlay */}
              <div className={`absolute inset-[2px] rounded-lg transition-opacity duration-300 ${
                analysisMode === 'layer'
                  ? 'bg-gradient-to-br from-purple-500/15 via-violet-500/10 to-fuchsia-500/5 opacity-100'
                  : 'bg-gradient-to-br from-purple-500/5 via-violet-500/3 to-transparent opacity-0 group-hover:opacity-100'
              }`} />
              
              <CardContent className="relative p-7">
                <div className="space-y-5">
                  {/* Radio button and Icon Section */}
                  <div className="flex items-start gap-4">
                    {/* Radio indicator */}
                    <div className={`relative flex-shrink-0 w-5 h-5 rounded-full border-2 transition-all duration-300 mt-1 ${
                      analysisMode === 'layer'
                        ? 'border-purple-500 dark:border-purple-400'
                        : 'border-slate-300 dark:border-slate-600 group-hover:border-purple-400'
                    }`}>
                      {analysisMode === 'layer' && (
                        <div className="absolute inset-0.5 rounded-full bg-purple-500 dark:bg-purple-400 animate-in zoom-in duration-200" />
                      )}
                    </div>
                    
                    {/* Icon */}
                    <div className={`relative rounded-2xl p-3.5 transition-all duration-300 ${
                      analysisMode === 'layer'
                        ? 'bg-gradient-to-br from-purple-500 to-violet-600 shadow-lg shadow-purple-500/30'
                        : 'bg-gradient-to-br from-purple-100 to-violet-50 dark:from-purple-900/50 dark:to-violet-900/30 group-hover:scale-105'
                    }`}>
                      <Layers className={`h-6 w-6 transition-all duration-300 ${
                        analysisMode === 'layer'
                          ? 'text-white'
                          : 'text-purple-600 dark:text-purple-400'
                      }`} />
                      {analysisMode === 'layer' && (
                        <div className="absolute inset-0 rounded-2xl bg-purple-500 animate-ping opacity-20" />
                      )}
                    </div>
                  </div>
                  
                  {/* Content Section */}
                  <div className="space-y-2.5 pl-9">
                    <div className="flex items-center gap-2">
                      <h3 className={`text-lg font-bold tracking-tight transition-colors duration-200 ${
                        analysisMode === 'layer'
                          ? 'text-purple-700 dark:text-purple-300'
                          : 'text-foreground group-hover:text-purple-700 dark:group-hover:text-purple-300'
                      }`}>
                        By Layer
                      </h3>
                      {analysisMode === 'layer' && (
                        <Badge className="bg-purple-500/10 text-purple-700 dark:text-purple-300 border-purple-500/20 text-xs">
                          Selected
                        </Badge>
                      )}
                    </div>
                    <p className="text-sm leading-relaxed text-muted-foreground/90">
                      Analyze components within specific architectural layers
                    </p>
                    {!analysisMode || analysisMode !== 'layer' ? (
                      <p className="text-xs text-muted-foreground/60 flex items-center gap-1 mt-2">
                        <Info className="h-3 w-3" />
                        Click to select this scope
                      </p>
                    ) : null}
                  </div>
                  
                  {/* Selection indicator bar */}
                  {analysisMode === 'layer' && (
                    <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-purple-500 to-violet-600" />
                  )}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Configuration Panel */}
          {analysisMode === 'type' && (
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="rounded-lg bg-blue-500 p-2">
                  <Box className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h4 className="font-semibold text-base">Select Component Type</h4>
                  <p className="text-sm text-muted-foreground">Choose which type of components to analyze</p>
                </div>
              </div>
              <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
                <Card
                  className={`group relative cursor-pointer transition-all duration-300 ease-in-out overflow-hidden ${
                    selectedType === 'Application'
                      ? 'border-0 shadow-xl shadow-blue-500/25'
                      : 'border-0 hover:shadow-lg hover:shadow-blue-500/20'
                  }`}
                  onClick={() => setSelectedType('Application')}
                >
                  {/* Gradient border */}
                  <div className={`absolute inset-0 rounded-lg p-[2px] transition-opacity duration-300 ${
                    selectedType === 'Application'
                      ? 'bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 opacity-100'
                      : 'bg-gradient-to-br from-slate-200 via-slate-300 to-slate-200 dark:from-slate-700 dark:via-slate-800 dark:to-slate-700 opacity-100 group-hover:from-blue-500 group-hover:via-indigo-500 group-hover:to-purple-500'
                  }`}>
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  
                  {/* Background gradient overlay */}
                  <div className={`absolute inset-[2px] rounded-lg transition-opacity duration-300 ${
                    selectedType === 'Application'
                      ? 'bg-gradient-to-br from-blue-500/10 via-indigo-500/5 to-transparent opacity-100'
                      : 'bg-gradient-to-br from-blue-500/5 via-indigo-500/3 to-transparent opacity-0 group-hover:opacity-100'
                  }`} />
                  
                  <CardContent className="relative p-4">
                    <div className="flex items-center gap-3">
                      <div className={`rounded-lg p-2 ${
                        selectedType === 'Application'
                          ? 'bg-blue-500'
                          : 'bg-blue-100 dark:bg-blue-900'
                      }`}>
                        <Server className={`h-5 w-5 ${
                          selectedType === 'Application'
                            ? 'text-white'
                            : 'text-blue-600 dark:text-blue-400'
                        }`} />
                      </div>
                      <div className="flex-1">
                        <div className="font-semibold text-sm">Application</div>
                        <div className="text-xs text-muted-foreground">Service applications</div>
                      </div>
                      {selectedType === 'Application' && (
                        <CheckCircle2 className="h-5 w-5 text-blue-500 shrink-0" />
                      )}
                    </div>
                  </CardContent>
                </Card>

                <Card
                  className={`group relative cursor-pointer transition-all duration-300 ease-in-out overflow-hidden ${
                    selectedType === 'Broker'
                      ? 'border-0 shadow-xl shadow-purple-500/25'
                      : 'border-0 hover:shadow-lg hover:shadow-purple-500/20'
                  }`}
                  onClick={() => setSelectedType('Broker')}
                >
                  {/* Gradient border */}
                  <div className={`absolute inset-0 rounded-lg p-[2px] transition-opacity duration-300 ${
                    selectedType === 'Broker'
                      ? 'bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 opacity-100'
                      : 'bg-gradient-to-br from-slate-200 via-slate-300 to-slate-200 dark:from-slate-700 dark:via-slate-800 dark:to-slate-700 opacity-100 group-hover:from-blue-500 group-hover:via-purple-500 group-hover:to-pink-500'
                  }`}>
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  
                  {/* Background gradient overlay */}
                  <div className={`absolute inset-[2px] rounded-lg transition-opacity duration-300 ${
                    selectedType === 'Broker'
                      ? 'bg-gradient-to-br from-purple-500/10 via-violet-500/5 to-transparent opacity-100'
                      : 'bg-gradient-to-br from-purple-500/5 via-violet-500/3 to-transparent opacity-0 group-hover:opacity-100'
                  }`} />
                  
                  <CardContent className="relative p-4">
                    <div className="flex items-center gap-3">
                      <div className={`rounded-lg p-2 ${
                        selectedType === 'Broker'
                          ? 'bg-purple-500'
                          : 'bg-purple-100 dark:bg-purple-900'
                      }`}>
                        <Network className={`h-5 w-5 ${
                          selectedType === 'Broker'
                            ? 'text-white'
                            : 'text-purple-600 dark:text-purple-400'
                        }`} />
                      </div>
                      <div className="flex-1">
                        <div className="font-semibold text-sm">Broker</div>
                        <div className="text-xs text-muted-foreground">Message brokers</div>
                      </div>
                      {selectedType === 'Broker' && (
                        <CheckCircle2 className="h-5 w-5 text-purple-500 shrink-0" />
                      )}
                    </div>
                  </CardContent>
                </Card>

                <Card
                  className={`group relative cursor-pointer transition-all duration-300 ease-in-out overflow-hidden ${
                    selectedType === 'Node'
                      ? 'border-0 shadow-xl shadow-green-500/25'
                      : 'border-0 hover:shadow-lg hover:shadow-green-500/20'
                  }`}
                  onClick={() => setSelectedType('Node')}
                >
                  {/* Gradient border */}
                  <div className={`absolute inset-0 rounded-lg p-[2px] transition-opacity duration-300 ${
                    selectedType === 'Node'
                      ? 'bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500 opacity-100'
                      : 'bg-gradient-to-br from-slate-200 via-slate-300 to-slate-200 dark:from-slate-700 dark:via-slate-800 dark:to-slate-700 opacity-100 group-hover:from-green-500 group-hover:via-emerald-500 group-hover:to-teal-500'
                  }`}>
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  
                  {/* Background gradient overlay */}
                  <div className={`absolute inset-[2px] rounded-lg transition-opacity duration-300 ${
                    selectedType === 'Node'
                      ? 'bg-gradient-to-br from-green-500/10 via-emerald-500/5 to-transparent opacity-100'
                      : 'bg-gradient-to-br from-green-500/5 via-emerald-500/3 to-transparent opacity-0 group-hover:opacity-100'
                  }`} />
                  
                  <CardContent className="relative p-4">
                    <div className="flex items-center gap-3">
                      <div className={`rounded-lg p-2 ${
                        selectedType === 'Node'
                          ? 'bg-green-500'
                          : 'bg-green-100 dark:bg-green-900'
                      }`}>
                        <Box className={`h-5 w-5 ${
                          selectedType === 'Node'
                            ? 'text-white'
                            : 'text-green-600 dark:text-green-400'
                        }`} />
                      </div>
                      <div className="flex-1">
                        <div className="font-semibold text-sm">Node</div>
                        <div className="text-xs text-muted-foreground">System nodes</div>
                      </div>
                      {selectedType === 'Node' && (
                        <CheckCircle2 className="h-5 w-5 text-green-500 shrink-0" />
                      )}
                    </div>
                  </CardContent>
                </Card>

                <Card
                  className={`group relative cursor-pointer transition-all duration-300 ease-in-out overflow-hidden ${
                    selectedType === 'Topic'
                      ? 'border-0 shadow-xl shadow-orange-500/25'
                      : 'border-0 hover:shadow-lg hover:shadow-orange-500/20'
                  }`}
                  onClick={() => setSelectedType('Topic')}
                >
                  {/* Gradient border */}
                  <div className={`absolute inset-0 rounded-lg p-[2px] transition-opacity duration-300 ${
                    selectedType === 'Topic'
                      ? 'bg-gradient-to-r from-orange-500 via-amber-500 to-yellow-500 opacity-100'
                      : 'bg-gradient-to-br from-slate-200 via-slate-300 to-slate-200 dark:from-slate-700 dark:via-slate-800 dark:to-slate-700 opacity-100 group-hover:from-orange-500 group-hover:via-amber-500 group-hover:to-yellow-500'
                  }`}>
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  
                  {/* Background gradient overlay */}
                  <div className={`absolute inset-[2px] rounded-lg transition-opacity duration-300 ${
                    selectedType === 'Topic'
                      ? 'bg-gradient-to-br from-orange-500/10 via-amber-500/5 to-transparent opacity-100'
                      : 'bg-gradient-to-br from-orange-500/5 via-amber-500/3 to-transparent opacity-0 group-hover:opacity-100'
                  }`} />
                  
                  <CardContent className="relative p-4">
                    <div className="flex items-center gap-3">
                      <div className={`rounded-lg p-2 ${
                        selectedType === 'Topic'
                          ? 'bg-orange-500'
                          : 'bg-orange-100 dark:bg-orange-900'
                      }`}>
                        <Activity className={`h-5 w-5 ${
                          selectedType === 'Topic'
                            ? 'text-white'
                            : 'text-orange-600 dark:text-orange-400'
                        }`} />
                      </div>
                      <div className="flex-1">
                        <div className="font-semibold text-sm">Topic</div>
                        <div className="text-xs text-muted-foreground">Message topics</div>
                      </div>
                      {selectedType === 'Topic' && (
                        <CheckCircle2 className="h-5 w-5 text-orange-500 shrink-0" />
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          )}

          {analysisMode === 'layer' && (
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="rounded-lg bg-purple-500 p-2">
                  <Layers className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h4 className="font-semibold text-base">Select Architectural Layer</h4>
                  <p className="text-sm text-muted-foreground">Choose which layer to analyze</p>
                </div>
              </div>
              <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
                {/* Application Layer */}
                <Card
                  className={`group relative cursor-pointer transition-all duration-300 ease-in-out overflow-hidden ${
                    selectedLayer === 'application'
                      ? 'border-0 shadow-xl shadow-blue-500/25'
                      : 'border-0 hover:shadow-lg hover:shadow-blue-500/20'
                  }`}
                  onClick={() => setSelectedLayer('application')}
                >
                  <div className={`absolute inset-0 rounded-lg p-[2px] transition-opacity duration-300 ${
                    selectedLayer === 'application'
                      ? 'bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500 opacity-100'
                      : 'bg-gradient-to-br from-slate-200 via-slate-300 to-slate-200 dark:from-slate-700 dark:via-slate-800 dark:to-slate-700 opacity-100 group-hover:from-blue-500 group-hover:via-indigo-500 group-hover:to-purple-500'
                  }`}>
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  <div className={`absolute inset-[2px] rounded-lg transition-opacity duration-300 ${
                    selectedLayer === 'application'
                      ? 'bg-gradient-to-br from-blue-500/10 via-indigo-500/5 to-transparent opacity-100'
                      : 'bg-gradient-to-br from-blue-500/5 via-indigo-500/3 to-transparent opacity-0 group-hover:opacity-100'
                  }`} />
                  <CardContent className="relative p-4">
                    <div className="flex items-center gap-3">
                      <div className={`rounded-lg p-2 ${
                        selectedLayer === 'application'
                          ? 'bg-blue-500'
                          : 'bg-blue-100 dark:bg-blue-900'
                      }`}>
                        <Server className={`h-5 w-5 ${
                          selectedLayer === 'application'
                            ? 'text-white'
                            : 'text-blue-600 dark:text-blue-400'
                        }`} />
                      </div>
                      <div className="flex-1">
                        <div className="font-semibold text-sm">Application</div>
                        <div className="text-xs text-muted-foreground">App-to-app dependencies</div>
                      </div>
                      {selectedLayer === 'application' && (
                        <CheckCircle2 className="h-5 w-5 text-blue-500 shrink-0" />
                      )}
                    </div>
                  </CardContent>
                </Card>

                {/* Infrastructure Layer */}
                <Card
                  className={`group relative cursor-pointer transition-all duration-300 ease-in-out overflow-hidden ${
                    selectedLayer === 'infrastructure'
                      ? 'border-0 shadow-xl shadow-slate-500/25'
                      : 'border-0 hover:shadow-lg hover:shadow-slate-500/20'
                  }`}
                  onClick={() => setSelectedLayer('infrastructure')}
                >
                  <div className={`absolute inset-0 rounded-lg p-[2px] transition-opacity duration-300 ${
                    selectedLayer === 'infrastructure'
                      ? 'bg-gradient-to-br from-slate-400 via-gray-500 to-zinc-600 opacity-100'
                      : 'bg-gradient-to-br from-slate-200 via-slate-300 to-slate-200 dark:from-slate-700 dark:via-slate-800 dark:to-slate-700 opacity-100 group-hover:from-slate-300 group-hover:via-gray-400 group-hover:to-zinc-500'
                  }`}>
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  <div className={`absolute inset-[2px] rounded-lg transition-opacity duration-300 ${
                    selectedLayer === 'infrastructure'
                      ? 'bg-gradient-to-br from-slate-500/10 via-gray-500/5 to-transparent opacity-100'
                      : 'bg-gradient-to-br from-slate-500/5 via-gray-500/3 to-transparent opacity-0 group-hover:opacity-100'
                  }`} />
                  <CardContent className="relative p-4">
                    <div className="flex items-center gap-3">
                      <div className={`rounded-lg p-2 ${
                        selectedLayer === 'infrastructure'
                          ? 'bg-slate-500'
                          : 'bg-slate-100 dark:bg-slate-900'
                      }`}>
                        <Database className={`h-5 w-5 ${
                          selectedLayer === 'infrastructure'
                            ? 'text-white'
                            : 'text-slate-600 dark:text-slate-400'
                        }`} />
                      </div>
                      <div className="flex-1">
                        <div className="font-semibold text-sm">Infrastructure</div>
                        <div className="text-xs text-muted-foreground">Node-to-node dependencies</div>
                      </div>
                      {selectedLayer === 'infrastructure' && (
                        <CheckCircle2 className="h-5 w-5 text-slate-500 shrink-0" />
                      )}
                    </div>
                  </CardContent>
                </Card>

                {/* Middleware Layer */}
                <Card
                  className={`group relative cursor-pointer transition-all duration-300 ease-in-out overflow-hidden ${
                    selectedLayer === 'middleware'
                      ? 'border-0 shadow-xl shadow-purple-500/25'
                      : 'border-0 hover:shadow-lg hover:shadow-purple-500/20'
                  }`}
                  onClick={() => setSelectedLayer('middleware')}
                >
                  <div className={`absolute inset-0 rounded-lg p-[2px] transition-opacity duration-300 ${
                    selectedLayer === 'middleware'
                      ? 'bg-gradient-to-r from-purple-500 via-violet-500 to-fuchsia-500 opacity-100'
                      : 'bg-gradient-to-br from-slate-200 via-slate-300 to-slate-200 dark:from-slate-700 dark:via-slate-800 dark:to-slate-700 opacity-100 group-hover:from-purple-500 group-hover:via-violet-500 group-hover:to-fuchsia-500'
                  }`}>
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  <div className={`absolute inset-[2px] rounded-lg transition-opacity duration-300 ${
                    selectedLayer === 'middleware'
                      ? 'bg-gradient-to-br from-purple-500/10 via-violet-500/5 to-transparent opacity-100'
                      : 'bg-gradient-to-br from-purple-500/5 via-violet-500/3 to-transparent opacity-0 group-hover:opacity-100'
                  }`} />
                  <CardContent className="relative p-4">
                    <div className="flex items-center gap-3">
                      <div className={`rounded-lg p-2 ${
                        selectedLayer === 'middleware'
                          ? 'bg-purple-500'
                          : 'bg-purple-100 dark:bg-purple-900'
                      }`}>
                        <Network className={`h-5 w-5 ${
                          selectedLayer === 'middleware'
                            ? 'text-white'
                            : 'text-purple-600 dark:text-purple-400'
                        }`} />
                      </div>
                      <div className="flex-1">
                        <div className="font-semibold text-sm">Middleware</div>
                        <div className="text-xs text-muted-foreground">Broker dependencies</div>
                      </div>
                      {selectedLayer === 'middleware' && (
                        <CheckCircle2 className="h-5 w-5 text-purple-500 shrink-0" />
                      )}
                    </div>
                  </CardContent>
                </Card>

                {/* System Layer */}
                <Card
                  className={`group relative cursor-pointer transition-all duration-300 ease-in-out overflow-hidden ${
                    selectedLayer === 'system'
                      ? 'border-0 shadow-xl shadow-emerald-500/25'
                      : 'border-0 hover:shadow-lg hover:shadow-emerald-500/20'
                  }`}
                  onClick={() => setSelectedLayer('system')}
                >
                  <div className={`absolute inset-0 rounded-lg p-[2px] transition-opacity duration-300 ${
                    selectedLayer === 'system'
                      ? 'bg-gradient-to-r from-emerald-500 via-green-500 to-teal-500 opacity-100'
                      : 'bg-gradient-to-br from-slate-200 via-slate-300 to-slate-200 dark:from-slate-700 dark:via-slate-800 dark:to-slate-700 opacity-100 group-hover:from-emerald-500 group-hover:via-green-500 group-hover:to-teal-500'
                  }`}>
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  <div className={`absolute inset-[2px] rounded-lg transition-opacity duration-300 ${
                    selectedLayer === 'system'
                      ? 'bg-gradient-to-br from-emerald-500/10 via-green-500/5 to-transparent opacity-100'
                      : 'bg-gradient-to-br from-emerald-500/5 via-green-500/3 to-transparent opacity-0 group-hover:opacity-100'
                  }`} />
                  <CardContent className="relative p-4">
                    <div className="flex items-center gap-3">
                      <div className={`rounded-lg p-2 ${
                        selectedLayer === 'system'
                          ? 'bg-emerald-500'
                          : 'bg-emerald-100 dark:bg-emerald-900'
                      }`}>
                        <Layers className={`h-5 w-5 ${
                          selectedLayer === 'system'
                            ? 'text-white'
                            : 'text-emerald-600 dark:text-emerald-400'
                        }`} />
                      </div>
                      <div className="flex-1">
                        <div className="font-semibold text-sm">System Layer</div>
                        <div className="text-xs text-muted-foreground">All dependencies (complete)</div>
                      </div>
                      {selectedLayer === 'system' && (
                        <CheckCircle2 className="h-5 w-5 text-emerald-500 shrink-0" />
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          )}
        </div>

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
        {isLoading && (
          <Card className="border-0 shadow-xl relative overflow-hidden">
            <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 animate-pulse">
              <div className="w-full h-full bg-background rounded-lg" />
            </div>
            <CardContent className="relative pt-8 pb-8">
              <div className="space-y-6">
                <div className="flex flex-col items-center gap-4">
                  <div className="relative">
                    <div className="absolute inset-0 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full blur-2xl opacity-30 animate-pulse" />
                    <div className="relative rounded-full bg-gradient-to-br from-blue-500/20 to-purple-600/20 p-6 ring-1 ring-blue-500/30">
                      <Loader2 className="h-10 w-10 animate-spin text-blue-600 dark:text-blue-400" />
                    </div>
                  </div>
                  <div className="text-center space-y-2">
                    <p className="text-xl font-bold">Running Quality Analysis...</p>
                    <p className="text-sm text-muted-foreground max-w-md">
                      Analyzing graph structure, computing quality metrics, and identifying potential issues
                    </p>
                  </div>
                </div>
                
                <div className="flex flex-col items-center gap-3">
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Clock className="h-4 w-4" />
                    <span className="text-sm font-medium">{elapsedTime}s elapsed</span>
                  </div>
                  <div className="w-full max-w-md">
                    <Progress value={Math.min((elapsedTime / estimatedDuration) * 100, 90)} className="w-full h-2" />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Analysis Results */}
        {!isLoading && analysisData && (
          <>
            {/* Results Header */}
            <Card className="relative overflow-hidden border-0 shadow-xl">
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500">
                <div className="w-full h-full bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 rounded-lg" />
              </div>
              <CardContent className="relative p-6 text-white">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="rounded-xl bg-white/20 p-3 backdrop-blur-sm">
                      <CheckCircle2 className="h-8 w-8" />
                    </div>
                    <div>
                      <h2 className="text-2xl font-bold mb-1">Analysis Complete</h2>
                      <p className="text-white/90">
                        {analysisMode === 'full' && 'Full System Analysis'}
                        {analysisMode === 'type' && `${selectedType} Components · ${analysisData.components?.length || 0} analyzed`}
                        {analysisMode === 'layer' && `${formatKey(selectedLayer)} · ${analysisData.components?.length || 0} components`}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="hidden md:flex items-center gap-4">
                    <div className="text-right">
                      <div className="text-3xl font-bold">{analysisData.components?.length || 0}</div>
                      <div className="text-sm text-white/80">Components</div>
                    </div>
                    {analysisData.edges && analysisData.edges.length > 0 && (
                      <div className="text-right">
                        <div className="text-3xl font-bold">{analysisData.edges.length}</div>
                        <div className="text-sm text-white/80">Connections</div>
                      </div>
                    )}
                    {analysisData.problems && (
                      <div className="text-right">
                        <div className="text-3xl font-bold">{analysisData.problems.length}</div>
                        <div className="text-sm text-white/80">Issues</div>
                      </div>
                    )}
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        onClick={exportAsJSON}
                        variant="outline"
                        size="sm"
                        className="flex items-center gap-2 bg-white/10 hover:bg-white/20 border-white/30 text-white"
                      >
                        <Download className="h-4 w-4" />
                        Export JSON
                      </Button>
                      <Button
                        onClick={() => {
                          clearAnalysis()
                          setError(null)
                        }}
                        variant="outline"
                        size="sm"
                        className="flex items-center gap-2 bg-white/10 hover:bg-white/20 border-white/30 text-white"
                      >
                        <XCircle className="h-4 w-4" />
                        Clear Results
                      </Button>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Quality Metrics Overview */}
            <div className="space-y-6">
              {/* Header Section with System Health Badge */}
              <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                  <div className="relative group">
                    <div className="absolute inset-0 bg-gradient-to-br from-blue-600 via-purple-600 to-pink-600 rounded-2xl blur-xl opacity-40 group-hover:opacity-60 transition-opacity duration-300" />
                    <div className="relative rounded-2xl bg-gradient-to-br from-blue-600 via-purple-600 to-pink-600 p-3 shadow-lg">
                      <Gauge className="h-6 w-6 text-white" />
                    </div>
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
                      Quality Metrics
                    </h2>
                    <p className="text-sm text-muted-foreground mt-0.5">
                      Comprehensive risk assessment across {analysisData.edges && analysisData.edges.length > 0
                        ? `${analysisData.components?.length || 0} components and ${analysisData.edges.length} connections`
                        : `${analysisData.components?.length || 0} analyzed components`}
                    </p>
                  </div>
                </div>
                
                {/* Overall System Health Badge */}
                {(() => {
                  const scores = calculateAggregateScores()
                  const avgScore = scores.overall
                  const healthStatus = avgScore < 40 ? 'Excellent' : avgScore < 60 ? 'Good' : avgScore < 80 ? 'Fair' : 'Critical'
                  const healthColor = avgScore < 40 ? 'from-green-500 to-emerald-600' : 
                                      avgScore < 60 ? 'from-blue-500 to-cyan-600' : 
                                      avgScore < 80 ? 'from-yellow-500 to-orange-600' : 'from-red-500 to-rose-600'
                  const healthBg = avgScore < 40 ? 'bg-green-500/10' : 
                                   avgScore < 60 ? 'bg-blue-500/10' : 
                                   avgScore < 80 ? 'bg-yellow-500/10' : 'bg-red-500/10'
                  const healthBorder = avgScore < 40 ? 'border-green-500/20' : 
                                       avgScore < 60 ? 'border-blue-500/20' : 
                                       avgScore < 80 ? 'border-yellow-500/20' : 'border-red-500/20'
                  
                  return (
                    <div className={`flex items-center gap-3 ${healthBg} ${healthBorder} rounded-2xl px-5 py-3 border`}>
                      <div className="text-center">
                        <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">System Health</div>
                        <div className={`text-2xl font-bold bg-gradient-to-r ${healthColor} bg-clip-text text-transparent`}>
                          {healthStatus}
                        </div>
                      </div>
                      <div className="h-12 w-px bg-border/40" />
                      <div className="text-center">
                        <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Overall Risk</div>
                        <div className={`text-2xl font-bold ${getScoreColor(avgScore / 100)}`}>
                          {avgScore.toFixed(1)}
                        </div>
                      </div>
                    </div>
                  )
                })()}
              </div>

              {/* Main Metrics Grid */}
              <div className="grid gap-6 lg:grid-cols-3">
                {/* Left Column: Metric Cards (2 columns on large screens) */}
                <div className="lg:col-span-2 grid gap-4 sm:grid-cols-2 auto-rows-fr content-start">
                  {(() => {
                    const scores = calculateAggregateScores()
                    const metrics = [
                      {
                        key: 'reliability',
                        name: 'Reliability Risk',
                        icon: Shield,
                        value: scores.reliability,
                        gradient: 'from-blue-500 via-indigo-500 to-purple-500',
                        description: 'System failure probability',
                      },
                      {
                        key: 'maintainability',
                        name: 'Maintainability Risk',
                        icon: Wrench,
                        value: scores.maintainability,
                        gradient: 'from-purple-500 via-violet-500 to-fuchsia-500',
                        description: 'Code complexity & technical debt',
                      },
                      {
                        key: 'availability',
                        name: 'Availability Risk',
                        icon: Zap,
                        value: scores.availability,
                        gradient: 'from-green-500 via-emerald-500 to-teal-500',
                        description: 'Service uptime vulnerability',
                      },
                      {
                        key: 'vulnerability',
                        name: 'Security Risk',
                        icon: AlertTriangle,
                        value: scores.vulnerability,
                        gradient: 'from-orange-500 via-red-500 to-rose-500',
                        description: 'Security exposure level',
                      },
                    ]

                    return metrics.map((metric) => {
                      const Icon = metric.icon
                      const riskLevel = metric.value < 40 ? 'Low' : metric.value < 60 ? 'Medium' : metric.value < 80 ? 'High' : 'Critical'
                      const circumference = 2 * Math.PI * 32
                      const offset = circumference - (metric.value / 100) * circumference
                      
                      return (
                        <Card 
                          key={metric.key}
                          className="group relative overflow-hidden border-0 shadow-lg hover:shadow-2xl transition-all duration-300 flex flex-col"
                        >
                          {/* Animated gradient border */}
                          <div className={`absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br ${metric.gradient} opacity-70 group-hover:opacity-100 transition-opacity duration-300`}>
                            <div className="w-full h-full bg-background rounded-lg" />
                          </div>
                          
                          {/* Background effects */}
                          <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br opacity-5 rounded-full blur-3xl group-hover:opacity-10 transition-opacity duration-500" style={{ background: `linear-gradient(to bottom right, ${metric.gradient})` }} />
                          
                          <CardHeader className="relative pb-4">
                            <div className="flex items-start justify-between">
                              <div className="flex items-center gap-3">
                                <div className={`relative rounded-xl bg-gradient-to-br ${metric.gradient} p-3 shadow-lg group-hover:scale-110 transition-transform duration-300`}>
                                  <Icon className="h-5 w-5 text-white" />
                                </div>
                                <div>
                                  <CardTitle className="text-base font-semibold">{metric.name}</CardTitle>
                                  <p className="text-xs text-muted-foreground mt-0.5">{metric.description}</p>
                                </div>
                              </div>
                            </div>
                          </CardHeader>
                          
                          <CardContent className="relative flex-1 flex flex-col justify-end space-y-4">
                            {/* Main Score Display with Circular Progress */}
                            <div className="flex items-end justify-between">
                              <div>
                                <div className="flex items-baseline gap-2">
                                  <span className={`text-4xl font-bold tabular-nums ${getScoreColor(metric.value / 100)}`}>
                                    {metric.value.toFixed(1)}
                                  </span>
                                  <span className="text-lg text-muted-foreground font-medium">/ 100</span>
                                </div>
                                <Badge 
                                  className={`mt-2 ${
                                    riskLevel === 'Low' ? 'bg-green-500/15 text-green-700 dark:text-green-300 border-green-500/30' :
                                    riskLevel === 'Medium' ? 'bg-yellow-500/15 text-yellow-700 dark:text-yellow-300 border-yellow-500/30' :
                                    riskLevel === 'High' ? 'bg-orange-500/15 text-orange-700 dark:text-orange-300 border-orange-500/30' :
                                    'bg-red-500/15 text-red-700 dark:text-red-300 border-red-500/30'
                                  }`}
                                >
                                  {riskLevel} Risk
                                </Badge>
                              </div>
                              
                              {/* Circular Progress Indicator */}
                              <div className="relative w-20 h-20">
                                <svg className="w-20 h-20 -rotate-90" viewBox="0 0 80 80">
                                  <circle
                                    cx="40"
                                    cy="40"
                                    r="32"
                                    fill="none"
                                    stroke="currentColor"
                                    strokeWidth="6"
                                    className="text-muted/20"
                                  />
                                  <circle
                                    cx="40"
                                    cy="40"
                                    r="32"
                                    fill="none"
                                    strokeWidth="6"
                                    strokeLinecap="round"
                                    strokeDasharray={circumference}
                                    strokeDashoffset={offset}
                                    className={`transition-all duration-1000 ease-out stroke-current`}
                                    style={{ 
                                      stroke: `url(#gradient-${metric.key})`,
                                      filter: 'drop-shadow(0 0 6px currentColor)'
                                    }}
                                  />
                                  <defs>
                                    <linearGradient id={`gradient-${metric.key}`} x1="0%" y1="0%" x2="100%" y2="100%">
                                      {metric.gradient.match(/(\w+-\d+)/g)?.map((color, i, arr) => (
                                        <stop key={i} offset={`${(i / (arr.length - 1)) * 100}%`} className={`text-${color}`} stopColor="currentColor" />
                                      ))}
                                    </linearGradient>
                                  </defs>
                                </svg>
                                <div className="absolute inset-0 flex items-center justify-center">
                                  <span className="text-xs font-bold text-muted-foreground">
                                    {Math.round(metric.value)}%
                                  </span>
                                </div>
                              </div>
                            </div>

                            {/* Progress Bar */}
                            <div className="space-y-2">
                              <div className="relative h-3 bg-muted/30 rounded-full overflow-hidden">
                                <div 
                                  className={`absolute inset-y-0 left-0 bg-gradient-to-r ${metric.gradient} rounded-full transition-all duration-1000 ease-out shadow-lg`}
                                  style={{ width: `${metric.value}%` }}
                                >
                                  <div className="absolute inset-0 bg-gradient-to-r from-white/20 to-transparent" />
                                </div>
                              </div>
                              <div className="flex items-center justify-between text-xs">
                                <span className="text-green-600 dark:text-green-400 font-medium">0 (Best)</span>
                                <span className="text-red-600 dark:text-red-400 font-medium">100 (Worst)</span>
                              </div>
                            </div>
                          </CardContent>
                        </Card>
                      )
                    })
                  })()}
                </div>

                {/* Right Column: Enhanced Radar Chart */}
                <Card className="border-0 shadow-2xl relative overflow-hidden bg-gradient-to-br from-background via-background to-muted/20">
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 opacity-70">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  
                  {/* Animated background elements */}
                  <div className="absolute top-10 right-10 w-32 h-32 bg-purple-500 rounded-full blur-3xl opacity-10 animate-pulse" />
                  <div className="absolute bottom-10 left-10 w-32 h-32 bg-blue-500 rounded-full blur-3xl opacity-10 animate-pulse" style={{ animationDelay: '1s' }} />
                  
                  <CardHeader className="relative pb-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="relative p-3 rounded-xl bg-gradient-to-br from-purple-500 via-blue-500 to-pink-500 shadow-lg">
                          <Target className="h-5 w-5 text-white" />
                        </div>
                        <div>
                          <CardTitle className="text-base font-semibold">Quality Overview</CardTitle>
                          <p className="text-xs text-muted-foreground mt-0.5">Risk distribution radar</p>
                        </div>
                      </div>
                    </div>
                  </CardHeader>
                  
                  <CardContent className="relative pb-6 flex flex-col items-center justify-center flex-1">
                    {(() => {
                      const scores = calculateAggregateScores()
                      const chartData = [
                        { attribute: "Reliability", value: scores.reliability, fullMark: 100 },
                        { attribute: "Maintainability", value: scores.maintainability, fullMark: 100 },
                        { attribute: "Availability", value: scores.availability, fullMark: 100 },
                        { attribute: "Security", value: scores.vulnerability, fullMark: 100 },
                      ]
                      
                      const maxValue = Math.max(...chartData.map(d => d.value))
                      const scaledMax = Math.ceil(maxValue / 10) * 10
                      const domain = scaledMax > 0 ? [0, Math.max(scaledMax, 10)] : [0, 100]
                      
                      const chartConfig = {
                        value: {
                          label: "Risk Score",
                          color: "hsl(var(--chart-1))",
                        },
                      }

                      return (
                        <>
                          <ChartContainer config={chartConfig} className="w-full h-full min-h-[320px]">
                            <RadarChart data={chartData} margin={{ top: 20, right: 30, bottom: 20, left: 30 }}>
                              <defs>
                                <radialGradient id="radarGradient" cx="50%" cy="50%">
                                  <stop offset="0%" stopColor="rgb(147 51 234)" stopOpacity="0.1" />
                                  <stop offset="40%" stopColor="rgb(139 92 246)" stopOpacity="0.25" />
                                  <stop offset="70%" stopColor="rgb(124 58 237)" stopOpacity="0.45" />
                                  <stop offset="100%" stopColor="rgb(109 40 217)" stopOpacity="0.65" />
                                </radialGradient>
                                <radialGradient id="radarGradientDark" cx="50%" cy="50%">
                                  <stop offset="0%" stopColor="rgb(196 181 253)" stopOpacity="0.15" />
                                  <stop offset="40%" stopColor="rgb(167 139 250)" stopOpacity="0.3" />
                                  <stop offset="70%" stopColor="rgb(139 92 246)" stopOpacity="0.5" />
                                  <stop offset="100%" stopColor="rgb(124 58 237)" stopOpacity="0.7" />
                                </radialGradient>
                                <filter id="radarGlow">
                                  <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
                                  <feMerge>
                                    <feMergeNode in="coloredBlur"/>
                                    <feMergeNode in="SourceGraphic"/>
                                  </feMerge>
                                </filter>
                                <linearGradient id="radarStroke" x1="0%" y1="0%" x2="100%" y2="100%">
                                  <stop offset="0%" stopColor="rgb(147 51 234)" />
                                  <stop offset="33%" stopColor="rgb(99 102 241)" />
                                  <stop offset="66%" stopColor="rgb(59 130 246)" />
                                  <stop offset="100%" stopColor="rgb(14 165 233)" />
                                </linearGradient>
                              </defs>
                              <PolarGrid 
                                strokeDasharray="3 3"
                                stroke="currentColor"
                                className="stroke-muted-foreground/25 dark:stroke-muted-foreground/35"
                                strokeWidth={1.5}
                              />
                              <PolarAngleAxis 
                                dataKey="attribute" 
                                tick={{ 
                                  fill: 'currentColor', 
                                  fontSize: 12, 
                                  fontWeight: 600,
                                }}
                                className="text-foreground [&_text]:drop-shadow-sm"
                                tickLine={false}
                              />
                              <PolarRadiusAxis 
                                angle={90} 
                                domain={domain}
                                tick={{ 
                                  fill: 'currentColor', 
                                  fontSize: 10,
                                  fontWeight: 500,
                                }}
                                className="text-muted-foreground"
                                axisLine={false}
                                tickCount={5}
                                stroke="currentColor"
                                strokeWidth={1}
                              />
                              <Radar
                                dataKey="value"
                                stroke="url(#radarStroke)"
                                strokeWidth={3}
                                fill="url(#radarGradient)"
                                fillOpacity={1}
                                filter="url(#radarGlow)"
                                className="dark:fill-[url(#radarGradientDark)] transition-all"
                              />
                              <ChartTooltip 
                                content={<ChartTooltipContent 
                                  formatter={(value) => `${Number(value).toFixed(1)}`}
                                  className="font-semibold"
                                  labelFormatter={(label) => `${label} Risk`}
                                />} 
                              />
                            </RadarChart>
                          </ChartContainer>
                          
                          {/* Legend */}
                          <div className="w-full mt-4 pt-4 border-t border-border/40">
                            <div className="grid grid-cols-2 gap-3 text-xs">
                              <div className="flex items-center gap-2">
                                <div className="w-3 h-3 rounded-full bg-gradient-to-r from-green-500 to-emerald-500" />
                                <span className="text-muted-foreground">0-40: Low Risk</span>
                              </div>
                              <div className="flex items-center gap-2">
                                <div className="w-3 h-3 rounded-full bg-gradient-to-r from-yellow-500 to-orange-500" />
                                <span className="text-muted-foreground">40-60: Medium</span>
                              </div>
                              <div className="flex items-center gap-2">
                                <div className="w-3 h-3 rounded-full bg-gradient-to-r from-orange-500 to-red-500" />
                                <span className="text-muted-foreground">60-80: High</span>
                              </div>
                              <div className="flex items-center gap-2">
                                <div className="w-3 h-3 rounded-full bg-gradient-to-r from-red-500 to-rose-600" />
                                <span className="text-muted-foreground">80-100: Critical</span>
                              </div>
                            </div>
                          </div>
                        </>
                      )
                    })()}
                  </CardContent>
                </Card>
              </div>
            </div>

            {/* Detailed Statistics */}
            {analysisData.summary && (
              <div className={`grid gap-3 ${analysisData.edges && analysisData.edges.length > 0 ? 'md:grid-cols-2' : 'md:grid-cols-1'}`}>
                {/* Components Card */}
                <Card className="group relative overflow-hidden border-0 hover:shadow-lg hover:shadow-blue-500/20 transition-all duration-300">
                    {/* Gradient border */}
                    <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500">
                      <div className="w-full h-full bg-background rounded-lg" />
                    </div>
                    {/* Subtle background glow */}
                    <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-blue-500/30 via-blue-500/15 to-blue-500/5" />
                    
                    <CardContent className="relative px-4 py-2">
                      {/* Header */}
                      <div className="flex items-center justify-between mb-2 pb-2 border-b border-border/30 px-2">
                        <div className="flex items-center gap-2">
                          <div className="rounded-xl bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-600 p-2.5 shadow-lg">
                            <Box className="h-5 w-5 text-white" />
                          </div>
                          <h3 className="text-sm font-medium">Components</h3>
                        </div>
                        <Badge variant="secondary" className="text-sm font-bold px-2.5 py-1">
                          {analysisData.components?.length || 0}
                        </Badge>
                      </div>

                      {/* Levels */}
                      <div className="space-y-1">
                        {(() => {
                          const order = ['critical', 'high', 'medium', 'low', 'minimal'];
                          const componentsData = analysisData.summary.components || {};
                          const componentTotal = Object.values(componentsData).reduce((sum: number, count) => sum + (count as number), 0);
                          
                          return order.map((level) => {
                            const componentCount = (componentsData[level] as number) || 0;
                            const componentPercentage = componentTotal > 0 ? (componentCount / componentTotal) * 100 : 0;
                            const isEmpty = componentCount === 0;
                            
                            return (
                              <div key={level} className={`group/item flex items-center gap-2.5 rounded-md px-2 py-1 transition-all hover:bg-muted/30`}>
                                <Badge className={`${getCriticalityColor(level)} text-xs font-semibold px-2.5 py-1 w-18 justify-center shrink-0`}>
                                  {formatKey(level)}
                                </Badge>
                                <div className="flex-1 relative h-2.5 bg-slate-200/60 dark:bg-slate-800/60 rounded-md overflow-hidden">
                                  <div 
                                    className={`absolute inset-y-0 left-0 transition-all duration-500 ease-out ${
                                      level === 'critical' ? 'bg-gradient-to-r from-red-500 to-red-600' :
                                      level === 'high' ? 'bg-gradient-to-r from-orange-500 to-orange-600' :
                                      level === 'medium' ? 'bg-gradient-to-r from-yellow-500 to-yellow-600' :
                                      level === 'low' ? 'bg-gradient-to-r from-green-500 to-green-600' :
                                      'bg-gradient-to-r from-slate-400 to-slate-500'
                                    }`}
                                    style={{ width: `${componentPercentage}%` }}
                                  />
                                </div>
                                <div className="flex items-center gap-1.5 shrink-0 min-w-[60px] justify-end">
                                  <span className="text-sm font-bold tabular-nums">{componentCount}</span>
                                  <span className="text-sm text-muted-foreground tabular-nums">({componentPercentage.toFixed(0)}%)</span>
                                </div>
                              </div>
                            );
                          });
                        })()}
                      </div>
                    </CardContent>
                  </Card>

                  {/* Connections Card */}
                  {analysisData.edges && analysisData.edges.length > 0 && (
                    <Card className="group relative overflow-hidden border-0 hover:shadow-lg hover:shadow-purple-500/20 transition-all duration-300">
                      {/* Gradient border */}
                      <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500">
                        <div className="w-full h-full bg-background rounded-lg" />
                      </div>
                      {/* Subtle background glow */}
                      <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-purple-500/30 via-purple-500/15 to-purple-500/5" />
                      
                      <CardContent className="relative px-4 py-2">
                        {/* Header */}
                        <div className="flex items-center justify-between mb-2 pb-2 border-b border-border/30 px-2">
                          <div className="flex items-center gap-2">
                            <div className="rounded-xl bg-gradient-to-br from-blue-600 via-purple-600 to-pink-600 p-2.5 shadow-lg">
                              <Network className="h-5 w-5 text-white" />
                            </div>
                            <h3 className="text-sm font-medium">Connections</h3>
                          </div>
                          <Badge variant="secondary" className="text-sm font-bold px-2.5 py-1">
                            {analysisData.edges?.length || 0}
                          </Badge>
                        </div>

                        {/* Levels */}
                        <div className="space-y-1">
                          {(() => {
                            const order = ['critical', 'high', 'medium', 'low', 'minimal'];
                            const edgesData = analysisData.summary.edges || {};
                            const edgeTotal = Object.values(edgesData).reduce((sum: number, count) => sum + (count as number), 0);
                            
                            return order.map((level) => {
                              const edgeCount = (edgesData[level] as number) || 0;
                              const edgePercentage = edgeTotal > 0 ? (edgeCount / edgeTotal) * 100 : 0;
                              const isEmpty = edgeCount === 0;
                              
                              return (
                                <div key={level} className={`group/item flex items-center gap-2.5 rounded-md px-2 py-1 transition-all hover:bg-muted/30`}>
                                  <Badge className={`${getCriticalityColor(level)} text-xs font-semibold px-2.5 py-1 w-18 justify-center shrink-0`}>
                                    {formatKey(level)}
                                  </Badge>
                                  <div className="flex-1 relative h-2.5 bg-slate-200/60 dark:bg-slate-800/60 rounded-md overflow-hidden">
                                    <div 
                                      className={`absolute inset-y-0 left-0 transition-all duration-500 ease-out ${
                                        level === 'critical' ? 'bg-gradient-to-r from-red-500 to-red-600' :
                                        level === 'high' ? 'bg-gradient-to-r from-orange-500 to-orange-600' :
                                        level === 'medium' ? 'bg-gradient-to-r from-yellow-500 to-yellow-600' :
                                        level === 'low' ? 'bg-gradient-to-r from-green-500 to-green-600' :
                                        'bg-gradient-to-r from-slate-400 to-slate-500'
                                      }`}
                                      style={{ width: `${edgePercentage}%` }}
                                    />
                                  </div>
                                  <div className="flex items-center gap-1.5 shrink-0 min-w-[60px] justify-end">
                                    <span className="text-sm font-bold tabular-nums">{edgeCount}</span>
                                    <span className="text-sm text-muted-foreground tabular-nums">({edgePercentage.toFixed(0)}%)</span>
                                  </div>
                                </div>
                              );
                            });
                          })()}
                        </div>
                      </CardContent>
                    </Card>
                  )}
                </div>
            )}

            {/* Graph Topology */}
            {analysisData.stats && (
              <div>
                <div className="mb-4 flex items-center gap-3">
                  <div className="rounded-xl bg-gradient-to-br from-slate-500 to-slate-600 p-2.5 shadow-md">
                    <Network className="h-5 w-5 text-white" />
                  </div>
                  <div>
                    <h2 className="text-xl font-bold">Graph Topology</h2>
                    <p className="text-sm text-muted-foreground">Structural properties of the analyzed system</p>
                  </div>
                </div>
                <div className="grid gap-4 md:grid-cols-2">
                  <Card className="group relative overflow-hidden border-0 hover:shadow-xl hover:shadow-blue-500/25 hover:scale-[1.02] transition-all duration-300">
                    {/* Gradient border */}
                    <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500">
                      <div className="w-full h-full bg-background rounded-lg" />
                    </div>
                    {/* Subtle background glow */}
                    <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-blue-500/30 via-blue-500/15 to-blue-500/5" />
                    
                    <CardHeader className="relative pb-3">
                      <div className="flex items-center gap-2">
                        <div className="rounded-xl bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-600 p-2.5 shadow-lg">
                          <Box className="h-5 w-5 text-white" />
                        </div>
                        <CardTitle className="text-sm font-medium">Total Nodes</CardTitle>
                      </div>
                    </CardHeader>
                    <CardContent className="relative">
                      <div className="text-3xl font-bold text-blue-600 dark:text-blue-400">
                        {analysisData.stats.nodes || 0}
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">Components in system</p>
                    </CardContent>
                  </Card>

                  <Card className="group relative overflow-hidden border-0 hover:shadow-xl hover:shadow-purple-500/25 hover:scale-[1.02] transition-all duration-300">
                    {/* Gradient border */}
                    <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500">
                      <div className="w-full h-full bg-background rounded-lg" />
                    </div>
                    {/* Subtle background glow */}
                    <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-purple-500/30 via-purple-500/15 to-purple-500/5" />
                    
                    <CardHeader className="relative pb-3">
                      <div className="flex items-center gap-2">
                        <div className="rounded-xl bg-gradient-to-br from-blue-600 via-purple-600 to-pink-600 p-2.5 shadow-lg">
                          <Network className="h-5 w-5 text-white" />
                        </div>
                        <CardTitle className="text-sm font-medium">Total Edges</CardTitle>
                      </div>
                    </CardHeader>
                    <CardContent className="relative">
                      <div className="text-3xl font-bold text-purple-600 dark:text-purple-400">
                        {analysisData.stats.edges || 0}
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">Dependencies tracked</p>
                    </CardContent>
                  </Card>
                </div>
              </div>
            )}

            {/* Critical Components */}
            {allCriticalComponents.length > 0 && (
              <div>
                <div className="mb-4 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="rounded-xl bg-gradient-to-br from-red-500 to-orange-600 p-2.5 shadow-lg">
                      <AlertTriangle className="h-5 w-5 text-white" />
                    </div>
                    <div>
                      <h2 className="text-xl font-bold">Critical Components</h2>
                      <p className="text-sm text-muted-foreground">
                        {allCriticalComponents.length} component{allCriticalComponents.length !== 1 ? 's' : ''} requiring attention
                      </p>
                    </div>
                  </div>
                  <Badge variant="destructive" className="gap-1.5 h-7 px-3 text-xs font-semibold">
                    <AlertTriangle className="h-3 w-3" />
                    {allCriticalComponents.length}
                  </Badge>
                </div>
                
                {/* Compact Table-like List */}
                <Card className="relative overflow-hidden border-0 p-0 shadow-lg">
                  {/* Gradient border */}
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-red-400 via-orange-500 to-red-600">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  {/* Subtle background glow */}
                  <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-red-500/35 via-red-500/20 to-red-500/5" />
                  
                  <div className="relative overflow-x-auto rounded-lg">
                    <table className="w-full border-separate border-spacing-0">
                      <thead>
                        <tr>
                          <th className="sticky top-0 text-left px-5 py-4 text-xs font-bold uppercase tracking-wider text-foreground/90 bg-gradient-to-b from-red-500/20 via-orange-500/15 to-red-500/10 backdrop-blur-sm border-b-2 border-red-500/40 w-14 first:rounded-tl-lg">
                            <div className="flex items-center justify-center gap-1.5">
                              <Hash className="h-3.5 w-3.5 text-red-600 dark:text-red-400" />
                            </div>
                          </th>
                          <th className="sticky top-0 text-left px-5 py-4 text-xs font-bold uppercase tracking-wider text-foreground/90 bg-gradient-to-b from-red-500/20 via-orange-500/15 to-red-500/10 backdrop-blur-sm border-b-2 border-red-500/40">
                            <div className="flex items-center gap-2">
                              <Box className="h-3.5 w-3.5 text-red-600 dark:text-red-400" />
                              Component
                            </div>
                          </th>
                          <th className="sticky top-0 text-left px-5 py-4 text-xs font-bold uppercase tracking-wider text-foreground/90 bg-gradient-to-b from-red-500/20 via-orange-500/15 to-red-500/10 backdrop-blur-sm border-b-2 border-red-500/40 w-36">
                            <div className="flex items-center gap-2">
                              <Tag className="h-3.5 w-3.5 text-orange-600 dark:text-orange-400" />
                              Type
                            </div>
                          </th>
                          <th className="sticky top-0 text-center px-5 py-4 text-xs font-bold uppercase tracking-wider text-foreground/90 bg-gradient-to-b from-red-500/20 via-orange-500/15 to-red-500/10 backdrop-blur-sm border-b-2 border-red-500/40 w-28">
                            <div className="flex items-center justify-center gap-2">
                              <AlertTriangle className="h-3.5 w-3.5 text-red-600 dark:text-red-400" />
                              Level
                            </div>
                          </th>
                          <th className="sticky top-0 text-center px-5 py-4 text-xs font-bold uppercase tracking-wider text-foreground/90 bg-gradient-to-b from-red-500/20 via-orange-500/15 to-red-500/10 backdrop-blur-sm border-b-2 border-red-500/40 w-28">
                            <div className="flex items-center justify-center gap-2">
                              <Gauge className="h-3.5 w-3.5 text-orange-600 dark:text-orange-400" />
                              Overall
                            </div>
                          </th>
                          <th className="sticky top-0 text-center px-5 py-4 text-xs font-bold uppercase tracking-wider text-foreground/90 bg-gradient-to-b from-red-500/20 via-orange-500/15 to-red-500/10 backdrop-blur-sm border-b-2 border-red-500/40 w-28">
                            <div className="flex items-center justify-center gap-2">
                              <Shield className="h-3.5 w-3.5 text-blue-600 dark:text-blue-400" />
                              Reliability
                            </div>
                          </th>
                          <th className="sticky top-0 text-center px-5 py-4 text-xs font-bold uppercase tracking-wider text-foreground/90 bg-gradient-to-b from-red-500/20 via-orange-500/15 to-red-500/10 backdrop-blur-sm border-b-2 border-red-500/40 w-28">
                            <div className="flex items-center justify-center gap-2">
                              <Wrench className="h-3.5 w-3.5 text-purple-600 dark:text-purple-400" />
                              Maintain
                            </div>
                          </th>
                          <th className="sticky top-0 text-center px-5 py-4 text-xs font-bold uppercase tracking-wider text-foreground/90 bg-gradient-to-b from-red-500/20 via-orange-500/15 to-red-500/10 backdrop-blur-sm border-b-2 border-red-500/40 w-28">
                            <div className="flex items-center justify-center gap-2">
                              <Zap className="h-3.5 w-3.5 text-green-600 dark:text-green-400" />
                              Availability
                            </div>
                          </th>
                          <th className="sticky top-0 text-center px-5 py-4 text-xs font-bold uppercase tracking-wider text-foreground/90 bg-gradient-to-b from-red-500/20 via-orange-500/15 to-red-500/10 backdrop-blur-sm border-b-2 border-red-500/40 w-28 last:rounded-tr-lg">
                            <div className="flex items-center justify-center gap-2">
                              <Shield className="h-3.5 w-3.5 text-cyan-600 dark:text-cyan-400" />
                              Vulnerability
                            </div>
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border/20">
                        {paginatedComponents.map((component, idx) => {
                          const globalIdx = (componentsPage - 1) * componentsPerPage + idx
                          return (
                          <tr 
                            key={idx} 
                            className="group hover:bg-gradient-to-r hover:from-red-500/[0.03] hover:via-orange-500/[0.02] hover:to-red-500/[0.03] transition-all duration-200"
                          >
                            {/* Rank */}
                            <td className="px-5 py-4">
                              <div className="flex items-center justify-center">
                                <div className="relative">
                                  <div className="absolute inset-0 bg-gradient-to-br from-red-500 to-orange-600 rounded-lg blur opacity-20 group-hover:opacity-40 transition-opacity" />
                                  <div className="relative w-8 h-8 rounded-lg bg-gradient-to-br from-red-500 to-orange-600 flex items-center justify-center shadow-md group-hover:shadow-lg group-hover:scale-110 transition-all duration-200">
                                    <span className="text-xs font-bold text-white">
                                      {globalIdx + 1}
                                    </span>
                                  </div>
                                </div>
                              </div>
                            </td>
                            
                            {/* Component Name & Icon */}
                            <td className="px-5 py-4">
                              <div className="flex items-center gap-3">
                                <div className="relative flex-shrink-0">
                                  <div className="absolute inset-0 bg-gradient-to-br from-red-500 to-orange-600 rounded-lg blur opacity-10 group-hover:opacity-25 transition-opacity" />
                                  <div className="relative rounded-lg bg-gradient-to-br from-red-500/15 to-orange-500/10 p-2 group-hover:from-red-500/25 group-hover:to-orange-500/20 transition-all duration-200">
                                    {component.type === 'Application' && <Server className="h-4 w-4 text-red-600 dark:text-red-400" />}
                                    {component.type === 'Broker' && <Network className="h-4 w-4 text-red-600 dark:text-red-400" />}
                                    {component.type === 'Node' && <Box className="h-4 w-4 text-red-600 dark:text-red-400" />}
                                    {component.type === 'Topic' && <MessageSquare className="h-4 w-4 text-red-600 dark:text-red-400" />}
                                    {!['Application', 'Broker', 'Node', 'Topic'].includes(component.type) && <Box className="h-4 w-4 text-red-600 dark:text-red-400" />}
                                  </div>
                                </div>
                                <div className="min-w-0 flex-1">
                                  {component.name && component.name !== component.id ? (
                                    <>
                                      <button
                                        onClick={() => router.push(`/explorer?node=${encodeURIComponent(component.id)}`)}
                                        className="text-sm font-semibold truncate hover:text-red-700 dark:hover:text-red-300 hover:underline transition-colors block w-full text-left"
                                        title={component.name}
                                      >
                                        {component.name}
                                      </button>
                                      <div className="text-xs text-muted-foreground/70 truncate font-mono mt-0.5" title={component.id}>
                                        {component.id}
                                      </div>
                                    </>
                                  ) : (
                                    <button
                                      onClick={() => router.push(`/explorer?node=${encodeURIComponent(component.id)}`)}
                                      className="text-sm font-semibold truncate hover:text-red-700 dark:hover:text-red-300 hover:underline transition-colors block w-full text-left"
                                      title={component.id}
                                    >
                                      {component.id}
                                    </button>
                                  )}
                                </div>
                              </div>
                            </td>
                            
                            {/* Type */}
                            <td className="px-5 py-4">
                              <Badge variant="outline" className="text-xs font-medium px-2.5 py-1 border-2 group-hover:border-orange-500/50 group-hover:bg-orange-500/5 transition-all duration-200">
                                {formatKey(component.type)}
                              </Badge>
                            </td>
                            
                            {/* Criticality Level */}
                            <td className="px-5 py-4">
                              <div className="flex justify-center">
                                <Badge className={`text-xs font-bold capitalize px-3 py-1.5 shadow-sm group-hover:shadow-md transition-all duration-200 ${getCriticalityColor(component.criticality_level)}`}>
                                  {component.criticality_level}
                                </Badge>
                              </div>
                            </td>
                            
                            {/* Overall Score */}
                            <td className="px-5 py-4">
                              <div className="flex flex-col items-center gap-2">
                                <div className="flex items-center gap-1.5">
                                  <span className={`text-base font-bold tabular-nums ${getScoreColor(component.scores.overall)}`}>
                                    {(component.scores.overall * 100).toFixed(0)}
                                  </span>
                                  <Gauge className={`h-3.5 w-3.5 ${getScoreColor(component.scores.overall)} group-hover:scale-110 transition-transform`} />
                                </div>
                                <div className="w-full max-w-[70px] h-1.5 bg-muted/50 rounded-full overflow-hidden shadow-inner">
                                  <div 
                                    className={`h-full rounded-full transition-all duration-500 shadow-sm ${
                                      component.scores.overall * 100 >= 80 
                                        ? 'bg-gradient-to-r from-red-500 to-red-600' 
                                        : component.scores.overall * 100 >= 60 
                                        ? 'bg-gradient-to-r from-yellow-500 to-yellow-600' 
                                        : 'bg-gradient-to-r from-green-500 to-green-600'
                                    }`}
                                    style={{ width: `${component.scores.overall * 100}%` }}
                                  />
                                </div>
                              </div>
                            </td>
                            
                            {/* Reliability Score */}
                            <td className="px-5 py-4">
                              <div className="flex flex-col items-center gap-2">
                                <span className={`text-sm font-semibold tabular-nums ${getScoreColor(component.scores.reliability)}`}>
                                  {(component.scores.reliability * 100).toFixed(0)}
                                </span>
                                <div className="w-full max-w-[60px] h-1.5 bg-muted/50 rounded-full overflow-hidden shadow-inner">
                                  <div 
                                    className={`h-full rounded-full transition-all duration-500 ${
                                      component.scores.reliability * 100 >= 80 
                                        ? 'bg-gradient-to-r from-red-500 to-red-600' 
                                        : component.scores.reliability * 100 >= 60 
                                        ? 'bg-gradient-to-r from-yellow-500 to-yellow-600' 
                                        : 'bg-gradient-to-r from-green-500 to-green-600'
                                    }`}
                                    style={{ width: `${component.scores.reliability * 100}%` }}
                                  />
                                </div>
                              </div>
                            </td>
                            
                            {/* Maintainability Score */}
                            <td className="px-5 py-4">
                              <div className="flex flex-col items-center gap-2">
                                <span className={`text-sm font-semibold tabular-nums ${getScoreColor(component.scores.maintainability)}`}>
                                  {(component.scores.maintainability * 100).toFixed(0)}
                                </span>
                                <div className="w-full max-w-[60px] h-1.5 bg-muted/50 rounded-full overflow-hidden shadow-inner">
                                  <div 
                                    className={`h-full rounded-full transition-all duration-500 ${
                                      component.scores.maintainability * 100 >= 80 
                                        ? 'bg-gradient-to-r from-red-500 to-red-600' 
                                        : component.scores.maintainability * 100 >= 60 
                                        ? 'bg-gradient-to-r from-yellow-500 to-yellow-600' 
                                        : 'bg-gradient-to-r from-green-500 to-green-600'
                                    }`}
                                    style={{ width: `${component.scores.maintainability * 100}%` }}
                                  />
                                </div>
                              </div>
                            </td>
                            
                            {/* Availability Score */}
                            <td className="px-5 py-4">
                              <div className="flex flex-col items-center gap-2">
                                <span className={`text-sm font-semibold tabular-nums ${getScoreColor(component.scores.availability)}`}>
                                  {(component.scores.availability * 100).toFixed(0)}
                                </span>
                                <div className="w-full max-w-[60px] h-1.5 bg-muted/50 rounded-full overflow-hidden shadow-inner">
                                  <div 
                                    className={`h-full rounded-full transition-all duration-500 ${
                                      component.scores.availability * 100 >= 80 
                                        ? 'bg-gradient-to-r from-red-500 to-red-600' 
                                        : component.scores.availability * 100 >= 60 
                                        ? 'bg-gradient-to-r from-yellow-500 to-yellow-600' 
                                        : 'bg-gradient-to-r from-green-500 to-green-600'
                                    }`}
                                    style={{ width: `${component.scores.availability * 100}%` }}
                                  />
                                </div>
                              </div>
                            </td>
                            
                            {/* Vulnerability Score */}
                            <td className="px-5 py-4">
                              <div className="flex flex-col items-center gap-2">
                                <span className={`text-sm font-semibold tabular-nums ${getScoreColor(component.scores.vulnerability)}`}>
                                  {(component.scores.vulnerability * 100).toFixed(0)}
                                </span>
                                <div className="w-full max-w-[60px] h-1.5 bg-muted/50 rounded-full overflow-hidden shadow-inner">
                                  <div 
                                    className={`h-full rounded-full transition-all duration-500 ${
                                      component.scores.vulnerability * 100 >= 80 
                                        ? 'bg-gradient-to-r from-red-500 to-red-600' 
                                        : component.scores.vulnerability * 100 >= 60 
                                        ? 'bg-gradient-to-r from-yellow-500 to-yellow-600' 
                                        : 'bg-gradient-to-r from-green-500 to-green-600'
                                    }`}
                                    style={{ width: `${component.scores.vulnerability * 100}%` }}
                                  />
                                </div>
                              </div>
                            </td>
                          </tr>
                        )})}
                      </tbody>
                    </table>
                  </div>
                </Card>

                {/* Pagination Card for Critical Components */}
                {totalComponentsPages > 1 && (
                  <Card className="relative overflow-hidden border-0 p-0 shadow-lg mt-4">
                    {/* Gradient border */}
                    <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-red-400 via-orange-500 to-red-600">
                      <div className="w-full h-full bg-background rounded-lg" />
                    </div>
                    {/* Subtle background glow */}
                    <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_center,var(--tw-gradient-stops))] from-red-500/10 via-transparent to-transparent" />
                    
                    <div className="relative px-6 py-4">
                        <div className="flex items-center justify-between gap-4">
                          {/* Page Info - Left Side */}
                          <div className="flex items-center gap-3">
                            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-background border border-border shadow-sm">
                              <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Page</span>
                              <span className="text-sm font-bold text-foreground">{componentsPage}</span>
                              <span className="text-xs text-muted-foreground">/</span>
                              <span className="text-sm font-bold text-foreground">{totalComponentsPages}</span>
                            </div>
                            <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-background/50 border border-border/50">
                              <span className="text-xs text-muted-foreground">Showing</span>
                              <span className="text-sm font-semibold text-foreground">
                                {(componentsPage - 1) * componentsPerPage + 1}-{Math.min(componentsPage * componentsPerPage, allCriticalComponents.length)}
                              </span>
                              <span className="text-xs text-muted-foreground">of</span>
                              <span className="text-sm font-semibold text-foreground">{allCriticalComponents.length}</span>
                            </div>
                          </div>
                        
                        {/* Navigation - Right Side */}
                        <div className="flex items-center gap-1.5">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setComponentsPage(1)}
                            disabled={componentsPage === 1}
                            className="h-9 w-9 p-0 rounded-lg border-2 hover:bg-red-500/10 hover:border-red-500/50 hover:shadow-md disabled:opacity-40 transition-all duration-200 hover:scale-105"
                            title="First page"
                          >
                            <ChevronsLeft className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setComponentsPage(p => p - 1)}
                            disabled={componentsPage === 1}
                            className="h-9 px-3.5 rounded-lg border-2 hover:bg-red-500/10 hover:border-red-500/50 hover:shadow-md disabled:opacity-40 transition-all duration-200 hover:scale-105"
                            title="Previous page"
                          >
                            <ChevronLeft className="h-4 w-4 mr-1" />
                            <span className="text-xs font-semibold">Prev</span>
                          </Button>
                          
                          {/* Page Numbers */}
                          <div className="hidden lg:flex items-center gap-1.5 mx-1">
                            {Array.from({ length: totalComponentsPages }, (_, i) => i + 1)
                              .filter(page => {
                                if (totalComponentsPages <= 7) return true
                                if (page === 1 || page === totalComponentsPages) return true
                                if (Math.abs(page - componentsPage) <= 1) return true
                                return false
                              })
                              .map((page, idx, arr) => (
                                <React.Fragment key={page}>
                                  {idx > 0 && arr[idx - 1] !== page - 1 && (
                                    <span className="px-2 text-muted-foreground text-xs font-bold">···</span>
                                  )}
                                  <Button
                                    variant={componentsPage === page ? 'default' : 'outline'}
                                    size="sm"
                                    onClick={() => setComponentsPage(page)}
                                    className={`h-9 min-w-9 px-2.5 text-xs font-bold rounded-lg border-2 transition-all duration-200 ${
                                      componentsPage === page
                                        ? 'bg-gradient-to-r from-red-600 to-orange-700 text-white shadow-lg shadow-red-500/30 hover:from-red-700 hover:to-orange-800 border-0 scale-110'
                                        : 'hover:bg-red-500/10 hover:border-red-500/50 hover:shadow-md hover:scale-105'
                                    }`}
                                  >
                                    {page}
                                  </Button>
                                </React.Fragment>
                              ))}
                          </div>
                          
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setComponentsPage(p => p + 1)}
                            disabled={componentsPage === totalComponentsPages}
                            className="h-9 px-3.5 rounded-lg border-2 hover:bg-red-500/10 hover:border-red-500/50 hover:shadow-md disabled:opacity-40 transition-all duration-200 hover:scale-105"
                            title="Next page"
                          >
                            <span className="text-xs font-semibold">Next</span>
                            <ChevronRight className="h-4 w-4 ml-1" />
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setComponentsPage(totalComponentsPages)}
                            disabled={componentsPage === totalComponentsPages}
                            className="h-9 w-9 p-0 rounded-lg border-2 hover:bg-red-500/10 hover:border-red-500/50 hover:shadow-md disabled:opacity-40 transition-all duration-200 hover:scale-105"
                            title="Last page"
                          >
                            <ChevronsRight className="h-4 w-4" />
                          </Button>
                        </div>
                        </div>
                    </div>
                  </Card>
                )}
              </div>
            )}

            {/* Critical Edges */}
            {allCriticalEdges.length > 0 && (
              <div>
                <div className="mb-4 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="rounded-xl bg-gradient-to-br from-purple-500 to-pink-600 p-2.5 shadow-lg">
                      <Network className="h-5 w-5 text-white" />
                    </div>
                    <div>
                      <h2 className="text-xl font-bold">Critical Connections</h2>
                      <p className="text-sm text-muted-foreground">
                        {allCriticalEdges.length} connection{allCriticalEdges.length !== 1 ? 's' : ''} requiring attention
                      </p>
                    </div>
                  </div>
                  <Badge variant="destructive" className="gap-1.5 h-7 px-3 text-xs font-semibold">
                    <Network className="h-3 w-3" />
                    {allCriticalEdges.length}
                  </Badge>
                </div>
                
                {/* Compact Table-like List */}
                <Card className="relative overflow-hidden border-0 p-0 shadow-lg">
                  {/* Gradient border */}
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-purple-400 via-fuchsia-500 to-pink-600">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  {/* Subtle background glow */}
                  <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-purple-500/35 via-purple-500/20 to-purple-500/5" />
                  
                  <div className="relative overflow-x-auto rounded-lg">
                    <table className="w-full border-separate border-spacing-0">
                      <thead>
                        <tr>
                          <th className="sticky top-0 text-left px-5 py-4 text-xs font-bold uppercase tracking-wider text-foreground/90 bg-gradient-to-b from-purple-500/20 via-fuchsia-500/15 to-purple-500/10 backdrop-blur-sm border-b-2 border-purple-500/40 w-14 first:rounded-tl-lg">
                            <div className="flex items-center justify-center gap-1.5">
                              <Hash className="h-3.5 w-3.5 text-purple-600 dark:text-purple-400" />
                            </div>
                          </th>
                          <th className="sticky top-0 text-left px-5 py-4 text-xs font-bold uppercase tracking-wider text-foreground/90 bg-gradient-to-b from-purple-500/20 via-fuchsia-500/15 to-purple-500/10 backdrop-blur-sm border-b-2 border-purple-500/40">
                            <div className="flex items-center gap-2">
                              <Network className="h-3.5 w-3.5 text-purple-600 dark:text-purple-400" />
                              Connection
                            </div>
                          </th>
                          <th className="sticky top-0 text-left px-5 py-4 text-xs font-bold uppercase tracking-wider text-foreground/90 bg-gradient-to-b from-purple-500/20 via-fuchsia-500/15 to-purple-500/10 backdrop-blur-sm border-b-2 border-purple-500/40 w-36">
                            <div className="flex items-center gap-2">
                              <Tag className="h-3.5 w-3.5 text-fuchsia-600 dark:text-fuchsia-400" />
                              Type
                            </div>
                          </th>
                          <th className="sticky top-0 text-center px-5 py-4 text-xs font-bold uppercase tracking-wider text-foreground/90 bg-gradient-to-b from-purple-500/20 via-fuchsia-500/15 to-purple-500/10 backdrop-blur-sm border-b-2 border-purple-500/40 w-28">
                            <div className="flex items-center justify-center gap-2">
                              <AlertTriangle className="h-3.5 w-3.5 text-purple-600 dark:text-purple-400" />
                              Level
                            </div>
                          </th>
                          <th className="sticky top-0 text-center px-5 py-4 text-xs font-bold uppercase tracking-wider text-foreground/90 bg-gradient-to-b from-purple-500/20 via-fuchsia-500/15 to-purple-500/10 backdrop-blur-sm border-b-2 border-purple-500/40 w-28">
                            <div className="flex items-center justify-center gap-2">
                              <Gauge className="h-3.5 w-3.5 text-fuchsia-600 dark:text-fuchsia-400" />
                              Overall
                            </div>
                          </th>
                          <th className="sticky top-0 text-center px-5 py-4 text-xs font-bold uppercase tracking-wider text-foreground/90 bg-gradient-to-b from-purple-500/20 via-fuchsia-500/15 to-purple-500/10 backdrop-blur-sm border-b-2 border-purple-500/40 w-28">
                            <div className="flex items-center justify-center gap-2">
                              <Shield className="h-3.5 w-3.5 text-blue-600 dark:text-blue-400" />
                              Reliability
                            </div>
                          </th>
                          <th className="sticky top-0 text-center px-5 py-4 text-xs font-bold uppercase tracking-wider text-foreground/90 bg-gradient-to-b from-purple-500/20 via-fuchsia-500/15 to-purple-500/10 backdrop-blur-sm border-b-2 border-purple-500/40 w-28">
                            <div className="flex items-center justify-center gap-2">
                              <Wrench className="h-3.5 w-3.5 text-purple-600 dark:text-purple-400" />
                              Maintain
                            </div>
                          </th>
                          <th className="sticky top-0 text-center px-5 py-4 text-xs font-bold uppercase tracking-wider text-foreground/90 bg-gradient-to-b from-purple-500/20 via-fuchsia-500/15 to-purple-500/10 backdrop-blur-sm border-b-2 border-purple-500/40 w-28">
                            <div className="flex items-center justify-center gap-2">
                              <Zap className="h-3.5 w-3.5 text-green-600 dark:text-green-400" />
                              Availability
                            </div>
                          </th>
                          <th className="sticky top-0 text-center px-5 py-4 text-xs font-bold uppercase tracking-wider text-foreground/90 bg-gradient-to-b from-purple-500/20 via-fuchsia-500/15 to-purple-500/10 backdrop-blur-sm border-b-2 border-purple-500/40 w-28 last:rounded-tr-lg">
                            <div className="flex items-center justify-center gap-2">
                              <Shield className="h-3.5 w-3.5 text-cyan-600 dark:text-cyan-400" />
                              Vulnerability
                            </div>
                          </th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border/20">
                        {paginatedEdges.map((edge, idx) => (
                          <tr 
                            key={idx} 
                            className="group hover:bg-gradient-to-r hover:from-purple-500/[0.03] hover:via-fuchsia-500/[0.02] hover:to-purple-500/[0.03] transition-all duration-200"
                          >
                            {/* Rank */}
                            <td className="px-5 py-4">
                              <div className="flex items-center justify-center">
                                <div className="relative">
                                  <div className="absolute inset-0 bg-gradient-to-br from-purple-500 to-fuchsia-600 rounded-lg blur opacity-20 group-hover:opacity-40 transition-opacity" />
                                  <div className="relative w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500 to-fuchsia-600 flex items-center justify-center shadow-md group-hover:shadow-lg group-hover:scale-110 transition-all duration-200">
                                    <span className="text-xs font-bold text-white">
                                      {(edgesPage - 1) * edgesPerPage + idx + 1}
                                    </span>
                                  </div>
                                </div>
                              </div>
                            </td>
                            
                            {/* Connection Name & Icon */}
                            <td className="px-5 py-4">
                              <div className="flex items-center gap-3">
                                <div className="relative flex-shrink-0">
                                  <div className="absolute inset-0 bg-gradient-to-br from-purple-500 to-fuchsia-600 rounded-lg blur opacity-10 group-hover:opacity-25 transition-opacity" />
                                  <div className="relative rounded-lg bg-gradient-to-br from-purple-500/15 to-fuchsia-500/10 p-2 group-hover:from-purple-500/25 group-hover:to-fuchsia-500/20 transition-all duration-200">
                                    <Network className="h-4 w-4 text-purple-600 dark:text-purple-400" />
                                  </div>
                                </div>
                                <div className="min-w-0 flex-1">
                                  <div className="flex items-center gap-1.5 text-sm font-semibold" title={`${edge.source} → ${edge.target}`}>
                                    <button
                                      onClick={() => router.push(`/explorer?node=${encodeURIComponent(edge.source)}`)}
                                      className="truncate hover:text-purple-700 dark:hover:text-purple-300 hover:underline transition-colors"
                                    >
                                      {edge.source_name || edge.source}
                                    </button>
                                    <span className="text-muted-foreground/60 shrink-0">→</span>
                                    <button
                                      onClick={() => router.push(`/explorer?node=${encodeURIComponent(edge.target)}`)}
                                      className="truncate hover:text-purple-700 dark:hover:text-purple-300 hover:underline transition-colors"
                                    >
                                      {edge.target_name || edge.target}
                                    </button>
                                  </div>
                                  {(edge.source_name || edge.target_name) && (
                                    <div className="flex items-center gap-1.5 text-xs text-muted-foreground/70 font-mono mt-0.5" title={`${edge.source} → ${edge.target}`}>
                                      <span className="truncate">{edge.source.length > 15 ? edge.source.substring(0, 15) + '...' : edge.source}</span>
                                      <span className="shrink-0">→</span>
                                      <span className="truncate">{edge.target.length > 15 ? edge.target.substring(0, 15) + '...' : edge.target}</span>
                                    </div>
                                  )}
                                </div>
                              </div>
                            </td>
                            
                            {/* Type */}
                            <td className="px-5 py-4">
                              <Badge variant="outline" className="text-xs font-medium px-2.5 py-1 border-2 group-hover:border-fuchsia-500/50 group-hover:bg-fuchsia-500/5 transition-all duration-200">
                                {formatKey(edge.type)}
                              </Badge>
                            </td>
                            
                            {/* Criticality Level */}
                            <td className="px-5 py-4">
                              <div className="flex justify-center">
                                <Badge className={`text-xs font-bold capitalize px-3 py-1.5 shadow-sm group-hover:shadow-md transition-all duration-200 ${getCriticalityColor(edge.criticality_level)}`}>
                                  {edge.criticality_level}
                                </Badge>
                              </div>
                            </td>
                            
                            {/* Overall Score */}
                            <td className="px-5 py-4">
                              <div className="flex flex-col items-center gap-2">
                                <div className="flex items-center gap-1.5">
                                  <span className={`text-base font-bold tabular-nums ${getScoreColor(edge.scores.overall)}`}>
                                    {(edge.scores.overall * 100).toFixed(0)}
                                  </span>
                                  <Gauge className={`h-3.5 w-3.5 ${getScoreColor(edge.scores.overall)} group-hover:scale-110 transition-transform`} />
                                </div>
                                <div className="w-full max-w-[70px] h-1.5 bg-muted/50 rounded-full overflow-hidden shadow-inner">
                                  <div 
                                    className={`h-full rounded-full transition-all duration-500 shadow-sm ${
                                      edge.scores.overall * 100 >= 80 
                                        ? 'bg-gradient-to-r from-red-500 to-red-600' 
                                        : edge.scores.overall * 100 >= 60 
                                        ? 'bg-gradient-to-r from-yellow-500 to-yellow-600' 
                                        : 'bg-gradient-to-r from-green-500 to-green-600'
                                    }`}
                                    style={{ width: `${edge.scores.overall * 100}%` }}
                                  />
                                </div>
                              </div>
                            </td>
                            
                            {/* Reliability Score */}
                            <td className="px-5 py-4">
                              <div className="flex flex-col items-center gap-2">
                                <span className={`text-sm font-semibold tabular-nums ${getScoreColor(edge.scores.reliability)}`}>
                                  {(edge.scores.reliability * 100).toFixed(0)}
                                </span>
                                <div className="w-full max-w-[60px] h-1.5 bg-muted/50 rounded-full overflow-hidden shadow-inner">
                                  <div 
                                    className={`h-full rounded-full transition-all duration-500 ${
                                      edge.scores.reliability * 100 >= 80 
                                        ? 'bg-gradient-to-r from-red-500 to-red-600' 
                                        : edge.scores.reliability * 100 >= 60 
                                        ? 'bg-gradient-to-r from-yellow-500 to-yellow-600' 
                                        : 'bg-gradient-to-r from-green-500 to-green-600'
                                    }`}
                                    style={{ width: `${edge.scores.reliability * 100}%` }}
                                  />
                                </div>
                              </div>
                            </td>
                            
                            {/* Maintainability Score */}
                            <td className="px-5 py-4">
                              <div className="flex flex-col items-center gap-2">
                                <span className={`text-sm font-semibold tabular-nums ${getScoreColor(edge.scores.maintainability)}`}>
                                  {(edge.scores.maintainability * 100).toFixed(0)}
                                </span>
                                <div className="w-full max-w-[60px] h-1.5 bg-muted/50 rounded-full overflow-hidden shadow-inner">
                                  <div 
                                    className={`h-full rounded-full transition-all duration-500 ${
                                      edge.scores.maintainability * 100 >= 80 
                                        ? 'bg-gradient-to-r from-red-500 to-red-600' 
                                        : edge.scores.maintainability * 100 >= 60 
                                        ? 'bg-gradient-to-r from-yellow-500 to-yellow-600' 
                                        : 'bg-gradient-to-r from-green-500 to-green-600'
                                    }`}
                                    style={{ width: `${edge.scores.maintainability * 100}%` }}
                                  />
                                </div>
                              </div>
                            </td>
                            
                            {/* Availability Score */}
                            <td className="px-5 py-4">
                              <div className="flex flex-col items-center gap-2">
                                <span className={`text-sm font-semibold tabular-nums ${getScoreColor(edge.scores.availability)}`}>
                                  {(edge.scores.availability * 100).toFixed(0)}
                                </span>
                                <div className="w-full max-w-[60px] h-1.5 bg-muted/50 rounded-full overflow-hidden shadow-inner">
                                  <div 
                                    className={`h-full rounded-full transition-all duration-500 ${
                                      edge.scores.availability * 100 >= 80 
                                        ? 'bg-gradient-to-r from-red-500 to-red-600' 
                                        : edge.scores.availability * 100 >= 60 
                                        ? 'bg-gradient-to-r from-yellow-500 to-yellow-600' 
                                        : 'bg-gradient-to-r from-green-500 to-green-600'
                                    }`}
                                    style={{ width: `${edge.scores.availability * 100}%` }}
                                  />
                                </div>
                              </div>
                            </td>
                            
                            {/* Vulnerability Score */}
                            <td className="px-5 py-4">
                              <div className="flex flex-col items-center gap-2">
                                <span className={`text-sm font-semibold tabular-nums ${getScoreColor(edge.scores.vulnerability)}`}>
                                  {(edge.scores.vulnerability * 100).toFixed(0)}
                                </span>
                                <div className="w-full max-w-[60px] h-1.5 bg-muted/50 rounded-full overflow-hidden shadow-inner">
                                  <div 
                                    className={`h-full rounded-full transition-all duration-500 ${
                                      edge.scores.vulnerability * 100 >= 80 
                                        ? 'bg-gradient-to-r from-red-500 to-red-600' 
                                        : edge.scores.vulnerability * 100 >= 60 
                                        ? 'bg-gradient-to-r from-yellow-500 to-yellow-600' 
                                        : 'bg-gradient-to-r from-green-500 to-green-600'
                                    }`}
                                    style={{ width: `${edge.scores.vulnerability * 100}%` }}
                                  />
                                </div>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>

                {/* Pagination Card for Critical Connections */}
                {totalEdgesPages > 1 && (
                  <Card className="relative overflow-hidden border-0 p-0 shadow-lg mt-4">
                    {/* Gradient border */}
                    <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-purple-400 via-fuchsia-500 to-pink-600">
                      <div className="w-full h-full bg-background rounded-lg" />
                    </div>
                    {/* Subtle background glow */}
                    <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_center,var(--tw-gradient-stops))] from-purple-500/10 via-transparent to-transparent" />
                    
                    <div className="relative px-6 py-4">
                        <div className="flex items-center justify-between gap-4">
                          {/* Page Info - Left Side */}
                          <div className="flex items-center gap-3">
                            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-background border border-border shadow-sm">
                              <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Page</span>
                              <span className="text-sm font-bold text-foreground">{edgesPage}</span>
                              <span className="text-xs text-muted-foreground">/</span>
                              <span className="text-sm font-bold text-foreground">{totalEdgesPages}</span>
                            </div>
                            <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-background/50 border border-border/50">
                              <span className="text-xs text-muted-foreground">Showing</span>
                              <span className="text-sm font-semibold text-foreground">
                                {(edgesPage - 1) * edgesPerPage + 1}-{Math.min(edgesPage * edgesPerPage, allCriticalEdges.length)}
                              </span>
                              <span className="text-xs text-muted-foreground">of</span>
                              <span className="text-sm font-semibold text-foreground">{allCriticalEdges.length}</span>
                            </div>
                          </div>
                        
                        {/* Navigation - Right Side */}
                        <div className="flex items-center gap-1.5">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setEdgesPage(1)}
                            disabled={edgesPage === 1}
                            className="h-9 w-9 p-0 rounded-lg border-2 hover:bg-purple-500/10 hover:border-purple-500/50 hover:shadow-md disabled:opacity-40 transition-all duration-200 hover:scale-105"
                            title="First page"
                          >
                            <ChevronsLeft className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setEdgesPage(p => p - 1)}
                            disabled={edgesPage === 1}
                            className="h-9 px-3.5 rounded-lg border-2 hover:bg-purple-500/10 hover:border-purple-500/50 hover:shadow-md disabled:opacity-40 transition-all duration-200 hover:scale-105"
                            title="Previous page"
                          >
                            <ChevronLeft className="h-4 w-4 mr-1" />
                            <span className="text-xs font-semibold">Prev</span>
                          </Button>
                          
                          {/* Page Numbers */}
                          <div className="hidden lg:flex items-center gap-1.5 mx-1">
                            {Array.from({ length: totalEdgesPages }, (_, i) => i + 1)
                              .filter(page => {
                                if (totalEdgesPages <= 7) return true
                                if (page === 1 || page === totalEdgesPages) return true
                                if (Math.abs(page - edgesPage) <= 1) return true
                                return false
                              })
                              .map((page, idx, arr) => (
                                <React.Fragment key={page}>
                                  {idx > 0 && arr[idx - 1] !== page - 1 && (
                                    <span className="px-2 text-muted-foreground text-xs font-bold">···</span>
                                  )}
                                  <Button
                                    variant={edgesPage === page ? 'default' : 'outline'}
                                    size="sm"
                                    onClick={() => setEdgesPage(page)}
                                    className={`h-9 min-w-9 px-2.5 text-xs font-bold rounded-lg border-2 transition-all duration-200 ${
                                      edgesPage === page
                                        ? 'bg-gradient-to-r from-purple-600 to-fuchsia-700 text-white shadow-lg shadow-purple-500/30 hover:from-purple-700 hover:to-fuchsia-800 border-0 scale-110'
                                        : 'hover:bg-purple-500/10 hover:border-purple-500/50 hover:shadow-md hover:scale-105'
                                    }`}
                                  >
                                    {page}
                                  </Button>
                                </React.Fragment>
                              ))}
                          </div>
                          
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setEdgesPage(p => p + 1)}
                            disabled={edgesPage === totalEdgesPages}
                            className="h-9 px-3.5 rounded-lg border-2 hover:bg-purple-500/10 hover:border-purple-500/50 hover:shadow-md disabled:opacity-40 transition-all duration-200 hover:scale-105"
                            title="Next page"
                          >
                            <span className="text-xs font-semibold">Next</span>
                            <ChevronRight className="h-4 w-4 ml-1" />
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setEdgesPage(totalEdgesPages)}
                            disabled={edgesPage === totalEdgesPages}
                            className="h-9 w-9 p-0 rounded-lg border-2 hover:bg-purple-500/10 hover:border-purple-500/50 hover:shadow-md disabled:opacity-40 transition-all duration-200 hover:scale-105"
                            title="Last page"
                          >
                            <ChevronsRight className="h-4 w-4" />
                          </Button>
                        </div>
                        </div>
                    </div>
                  </Card>
                )}
              </div>
            )}

            {/* Identified Issues */}
            {analysisData.problems && analysisData.problems.length > 0 && (
              <div className="space-y-4">
                {/* Header Section */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="rounded-xl bg-gradient-to-br from-orange-500 to-red-600 p-2.5 shadow-lg shadow-orange-500/30">
                      <AlertTriangle className="h-5 w-5 text-white" />
                    </div>
                    <div>
                      <h2 className="text-xl font-bold">Identified Issues</h2>
                      <p className="text-sm text-muted-foreground">
                        {filteredIssues.length === analysisData.problems.length 
                          ? `${analysisData.problems.length} total issue${analysisData.problems.length !== 1 ? 's' : ''} detected`
                          : `Showing ${filteredIssues.length} of ${analysisData.problems.length} issue${analysisData.problems.length !== 1 ? 's' : ''}`
                        }
                      </p>
                    </div>
                  </div>
                  <Badge className="gap-1.5 h-8 px-3 bg-gradient-to-r from-red-500 to-orange-600 text-white shadow-md">
                    <AlertTriangle className="h-3.5 w-3.5" />
                    {analysisData.problems.length}
                  </Badge>
                </div>

                {/* Search and Filters Card */}
                <Card className="relative overflow-hidden border-0 p-0 shadow-lg">
                    {/* Gradient border */}
                    <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-blue-400 via-purple-500 to-blue-600">
                      <div className="w-full h-full bg-background rounded-lg" />
                    </div>
                    {/* Subtle background glow */}
                    <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_top_left,var(--tw-gradient-stops))] from-blue-500/15 via-purple-500/10 to-transparent" />
                    
                    {/* Search and Filters Section */}
                    <div className="relative px-6 py-5">
                      <div className="space-y-4">
                        {/* Search Bar with Enhanced Design */}
                        <div className="relative group">
                          <div className="absolute inset-0 bg-gradient-to-r from-blue-500/5 via-purple-500/5 to-blue-500/5 rounded-xl blur-sm opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                          <div className="relative">
                            <div className="absolute left-4 top-1/2 -translate-y-1/2 flex items-center gap-2">
                              <div className="rounded-lg bg-gradient-to-br from-blue-500/15 to-purple-500/15 p-2 group-hover:from-blue-500/25 group-hover:to-purple-500/25 transition-all duration-200 shadow-sm">
                                <Search className="h-4 w-4 text-blue-600 dark:text-blue-400" />
                              </div>
                            </div>
                            <Input
                              placeholder="Search by name, description, recommendation, component ID..."
                              value={searchQuery}
                              onChange={(e) => setSearchQuery(e.target.value)}
                              className="h-12 pl-14 pr-12 text-sm border-2 border-border hover:border-blue-500/50 focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all bg-background shadow-sm rounded-xl font-medium"
                            />
                            {searchQuery && (
                              <button
                                onClick={() => setSearchQuery('')}
                                className="absolute right-4 top-1/2 -translate-y-1/2 rounded-lg bg-red-100 dark:bg-red-950/50 p-1.5 hover:bg-red-200 dark:hover:bg-red-900/50 transition-all duration-200 shadow-sm group/clear"
                                title="Clear search"
                              >
                                <X className="h-4 w-4 text-red-600 dark:text-red-400 group-hover/clear:scale-110 transition-transform" />
                              </button>
                            )}
                          </div>
                        </div>

                        {/* Filters Section */}
                        <div className="flex flex-col gap-3">
                          {/* Severity Filter Row */}
                          <div className="flex items-center gap-4">
                            <div className="flex items-center gap-2.5 min-w-[100px]">
                              <div className="rounded-lg bg-gradient-to-br from-red-500/15 to-orange-500/15 p-1.5 shadow-sm">
                                <AlertTriangle className="h-3.5 w-3.5 text-red-600 dark:text-red-400" />
                              </div>
                              <span className="text-xs font-bold text-foreground uppercase tracking-wide">Severity</span>
                            </div>
                            <div className="flex flex-wrap gap-2 flex-1">
                              <button
                                onClick={() => setSeverityFilter('all')}
                                className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all duration-200 ${
                                  severityFilter === 'all'
                                    ? 'bg-gradient-to-r from-slate-700 to-slate-800 dark:from-slate-600 dark:to-slate-700 text-white shadow-lg shadow-slate-500/30 ring-2 ring-slate-700/50 scale-105'
                                    : 'bg-slate-100 dark:bg-slate-900/50 text-slate-700 dark:text-slate-400 hover:bg-slate-200 dark:hover:bg-slate-800 hover:shadow-md hover:scale-105'
                                }`}
                              >
                                All <span className="font-normal opacity-80">({analysisData.problems.length})</span>
                              </button>
                              {availableSeverities.map((severity) => {
                                const count = analysisData.problems.filter(p => p.severity.toLowerCase() === severity.toLowerCase()).length
                                const isActive = severityFilter === severity.toLowerCase()
                                const severityStyles = {
                                  critical: {
                                    active: 'bg-gradient-to-r from-red-600 to-red-700 text-white shadow-md shadow-red-500/30 ring-1 ring-red-600/50',
                                    inactive: 'bg-red-100 dark:bg-red-950/30 text-red-700 dark:text-red-400 hover:bg-red-200 dark:hover:bg-red-950/50 hover:shadow-sm'
                                  },
                                  high: {
                                    active: 'bg-gradient-to-r from-orange-600 to-orange-700 text-white shadow-md shadow-orange-500/30 ring-1 ring-orange-600/50',
                                    inactive: 'bg-orange-100 dark:bg-orange-950/30 text-orange-700 dark:text-orange-400 hover:bg-orange-200 dark:hover:bg-orange-950/50 hover:shadow-sm'
                                  },
                                  medium: {
                                    active: 'bg-gradient-to-r from-yellow-600 to-yellow-700 text-white shadow-md shadow-yellow-500/30 ring-1 ring-yellow-600/50',
                                    inactive: 'bg-yellow-100 dark:bg-yellow-950/30 text-yellow-700 dark:text-yellow-400 hover:bg-yellow-200 dark:hover:bg-yellow-950/50 hover:shadow-sm'
                                  },
                                  low: {
                                    active: 'bg-gradient-to-r from-blue-600 to-blue-700 text-white shadow-md shadow-blue-500/30 ring-1 ring-blue-600/50',
                                    inactive: 'bg-blue-100 dark:bg-blue-950/30 text-blue-700 dark:text-blue-400 hover:bg-blue-200 dark:hover:bg-blue-950/50 hover:shadow-sm'
                                  }
                                }
                                const style = severityStyles[severity.toLowerCase() as keyof typeof severityStyles] || severityStyles.low
                                return (
                                  <button
                                    key={severity}
                                    onClick={() => setSeverityFilter(severity.toLowerCase())}
                                    className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all duration-200 ${
                                      isActive ? `${style.active} scale-105` : `${style.inactive} hover:scale-105`
                                    }`}
                                  >
                                    {severity} <span className="font-normal opacity-80">({count})</span>
                                  </button>
                                )
                              })}
                            </div>
                          </div>

                          {/* Category Filter Row */}
                          <div className="flex items-center gap-4">
                            <div className="flex items-center gap-2.5 min-w-[100px]">
                              <div className="rounded-lg bg-gradient-to-br from-purple-500/15 to-fuchsia-500/15 p-1.5 shadow-sm">
                                <Tag className="h-3.5 w-3.5 text-purple-600 dark:text-purple-400" />
                              </div>
                              <span className="text-xs font-bold text-foreground uppercase tracking-wide">Category</span>
                            </div>
                            <div className="flex flex-wrap gap-2 flex-1">
                              <button
                                onClick={() => setCategoryFilter('all')}
                                className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all duration-200 ${
                                  categoryFilter === 'all'
                                    ? 'bg-gradient-to-r from-purple-600 to-fuchsia-700 text-white shadow-lg shadow-purple-500/30 ring-2 ring-purple-600/50 scale-105'
                                    : 'bg-purple-100 dark:bg-purple-950/30 text-purple-700 dark:text-purple-400 hover:bg-purple-200 dark:hover:bg-purple-900/50 hover:shadow-md hover:scale-105'
                                }`}
                              >
                                All
                              </button>
                              {availableCategories.map((category) => {
                                const count = analysisData.problems.filter(p => p.category.toLowerCase() === category.toLowerCase()).length
                                const isActive = categoryFilter === category.toLowerCase()
                                return (
                                  <button
                                    key={category}
                                    onClick={() => setCategoryFilter(category.toLowerCase())}
                                    className={`px-3 py-1.5 rounded-lg text-xs font-bold transition-all duration-200 ${
                                      isActive
                                        ? 'bg-gradient-to-r from-purple-600 to-fuchsia-700 text-white shadow-lg shadow-purple-500/30 ring-2 ring-purple-600/50 scale-105'
                                        : 'bg-purple-100 dark:bg-purple-950/30 text-purple-700 dark:text-purple-400 hover:bg-purple-200 dark:hover:bg-purple-900/50 hover:shadow-md hover:scale-105'
                                    }`}
                                  >
                                    {formatKey(category)} <span className="font-normal opacity-80">({count})</span>
                                  </button>
                                )
                              })}
                            </div>
                          </div>
                        </div>

                        {/* Active Filters - Compact Badge Display */}
                        {(searchQuery || severityFilter !== 'all' || categoryFilter !== 'all') && (
                          <div className="flex items-center gap-3 pt-3 border-t border-orange-500/30">
                            <div className="flex items-center gap-2">
                              <div className="rounded-lg bg-gradient-to-br from-orange-500/15 to-red-500/15 p-1.5 shadow-sm">
                                <Filter className="h-3.5 w-3.5 text-orange-600 dark:text-orange-400" />
                              </div>
                              <span className="text-xs font-bold text-foreground uppercase tracking-wide">Active Filters</span>
                            </div>
                            <div className="flex flex-wrap gap-1.5 flex-1">
                              {searchQuery && (
                                <Badge className="gap-1.5 h-6 text-xs px-2.5 bg-blue-500 text-white border-0 shadow-sm hover:shadow-md transition-all">
                                  &quot;{searchQuery.substring(0, 12)}{searchQuery.length > 12 ? '...' : ''}&quot;
                                  <button onClick={() => setSearchQuery('')} className="hover:bg-blue-600 rounded p-0.5 transition-colors">
                                    <X className="h-2.5 w-2.5" />
                                  </button>
                                </Badge>
                              )}
                              {severityFilter !== 'all' && (
                                <Badge className="gap-1.5 h-6 text-xs px-2.5 bg-orange-500 text-white border-0 shadow-sm hover:shadow-md transition-all capitalize">
                                  {severityFilter}
                                  <button onClick={() => setSeverityFilter('all')} className="hover:bg-orange-600 rounded p-0.5 transition-colors">
                                    <X className="h-2.5 w-2.5" />
                                  </button>
                                </Badge>
                              )}
                              {categoryFilter !== 'all' && (
                                <Badge className="gap-1.5 h-6 text-xs px-2.5 bg-purple-500 text-white border-0 shadow-sm hover:shadow-md transition-all">
                                  {formatKey(categoryFilter)}
                                  <button onClick={() => setCategoryFilter('all')} className="hover:bg-purple-600 rounded p-0.5 transition-colors">
                                    <X className="h-2.5 w-2.5" />
                                  </button>
                                </Badge>
                              )}
                              <button
                                onClick={() => {
                                  setSearchQuery('')
                                  setSeverityFilter('all')
                                  setCategoryFilter('all')
                                }}
                                className="h-6 px-3 text-xs font-semibold text-muted-foreground hover:text-foreground hover:bg-red-100 dark:hover:bg-red-950/50 hover:text-red-600 dark:hover:text-red-400 rounded-lg transition-all shadow-sm hover:shadow-md"
                              >
                                Clear all
                              </button>
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                </Card>

                {/* Content Card */}
                <Card className="relative overflow-hidden border-0 p-0 gap-0 shadow-lg">
                    {/* Gradient border */}
                    <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-orange-400 via-orange-500 to-red-600">
                      <div className="w-full h-full bg-background rounded-lg" />
                    </div>
                    {/* Subtle background glow */}
                    <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-orange-500/35 via-orange-500/20 to-orange-500/5" />
                    
                    {/* Table or Empty State */}
                    {filteredIssues.length === 0 ? (
                      <div className="relative py-16 text-center">
                        <div className="relative w-fit mx-auto mb-6">
                          <div className="absolute inset-0 bg-slate-200 dark:bg-slate-800 rounded-full blur-xl opacity-50"></div>
                          <div className="relative rounded-2xl bg-slate-100 dark:bg-slate-900 p-6">
                            <Search className="h-12 w-12 text-slate-400" />
                          </div>
                        </div>
                        <h3 className="text-xl font-bold mb-2">No Issues Found</h3>
                        <p className="text-sm text-muted-foreground max-w-md mx-auto">
                          {searchQuery || severityFilter !== 'all' || categoryFilter !== 'all'
                            ? 'No issues match your current filters. Try adjusting your search criteria.'
                            : 'No issues match the current criteria'}
                        </p>
                      </div>
                    ) : (
                    <>
                    <div className="relative overflow-x-auto rounded-lg">
                      <table className="w-full border-separate border-spacing-0">
                        <thead>
                          <tr>
                            <th className="sticky top-0 text-left px-5 py-4 text-xs font-bold uppercase tracking-wider text-foreground/90 bg-gradient-to-b from-orange-500/20 via-red-500/15 to-orange-500/10 backdrop-blur-sm border-b-2 border-orange-500/40 w-14 first:rounded-tl-lg">
                              <div className="flex items-center justify-center gap-1.5">
                                <Hash className="h-3.5 w-3.5 text-orange-600 dark:text-orange-400" />
                              </div>
                            </th>
                            <th className="sticky top-0 text-left px-5 py-4 text-xs font-bold uppercase tracking-wider text-foreground/90 bg-gradient-to-b from-orange-500/20 via-red-500/15 to-orange-500/10 backdrop-blur-sm border-b-2 border-orange-500/40 w-32">
                              <div className="flex items-center gap-2">
                                <AlertTriangle className="h-3.5 w-3.5 text-red-600 dark:text-red-400" />
                                Severity
                              </div>
                            </th>
                            <th className="sticky top-0 text-left px-5 py-4 text-xs font-bold uppercase tracking-wider text-foreground/90 bg-gradient-to-b from-orange-500/20 via-red-500/15 to-orange-500/10 backdrop-blur-sm border-b-2 border-orange-500/40 w-32">
                              <div className="flex items-center gap-2">
                                <Tag className="h-3.5 w-3.5 text-orange-600 dark:text-orange-400" />
                                Category
                              </div>
                            </th>
                            <th className="sticky top-0 text-left px-5 py-4 text-xs font-bold uppercase tracking-wider text-foreground/90 bg-gradient-to-b from-orange-500/20 via-red-500/15 to-orange-500/10 backdrop-blur-sm border-b-2 border-orange-500/40 w-48">
                              <div className="flex items-center gap-2">
                                <Box className="h-3.5 w-3.5 text-orange-600 dark:text-orange-400" />
                                Component
                              </div>
                            </th>
                            <th className="sticky top-0 text-left px-5 py-4 text-xs font-bold uppercase tracking-wider text-foreground/90 bg-gradient-to-b from-orange-500/20 via-red-500/15 to-orange-500/10 backdrop-blur-sm border-b-2 border-orange-500/40">
                              <div className="flex items-center gap-2">
                                <AlertTriangle className="h-3.5 w-3.5 text-red-600 dark:text-red-400" />
                                Issue
                              </div>
                            </th>
                            <th className="sticky top-0 text-left px-5 py-4 text-xs font-bold uppercase tracking-wider text-foreground/90 bg-gradient-to-b from-orange-500/20 via-red-500/15 to-orange-500/10 backdrop-blur-sm border-b-2 border-orange-500/40 last:rounded-tr-lg">
                              <div className="flex items-center gap-2">
                                <Lightbulb className="h-3.5 w-3.5 text-yellow-600 dark:text-yellow-400" />
                                Recommendation
                              </div>
                            </th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-border/20">
                          {paginatedIssues.map((problem, idx) => {
                            const globalIdx = (issuesPage - 1) * issuesPerPage + idx
                            const severityConfig = {
                              critical: {
                                badge: 'bg-red-500 text-white shadow-md',
                                icon: 'bg-gradient-to-br from-red-500/15 to-orange-500/10',
                                iconColor: 'text-red-600 dark:text-red-400',
                                numberBg: 'from-red-500 to-red-600'
                              },
                              high: {
                                badge: 'bg-orange-500 text-white shadow-md',
                                icon: 'bg-gradient-to-br from-orange-500/15 to-yellow-500/10',
                                iconColor: 'text-orange-600 dark:text-orange-400',
                                numberBg: 'from-orange-500 to-orange-600'
                              },
                              medium: {
                                badge: 'bg-yellow-500 text-white shadow-md',
                                icon: 'bg-gradient-to-br from-yellow-500/15 to-amber-500/10',
                                iconColor: 'text-yellow-600 dark:text-yellow-400',
                                numberBg: 'from-yellow-500 to-yellow-600'
                              },
                              low: {
                                badge: 'bg-blue-500 text-white shadow-md',
                                icon: 'bg-gradient-to-br from-blue-500/15 to-cyan-500/10',
                                iconColor: 'text-blue-600 dark:text-blue-400',
                                numberBg: 'from-blue-500 to-blue-600'
                              }
                            }
                            const config = severityConfig[problem.severity.toLowerCase() as keyof typeof severityConfig] || severityConfig.low
                            
                            return (
                              <tr
                                key={`${problem.entity_id}-${idx}`}
                                className="group hover:bg-gradient-to-r hover:from-orange-500/[0.03] hover:via-red-500/[0.02] hover:to-orange-500/[0.03] transition-all duration-200"
                              >
                                {/* Rank */}
                                <td className="px-5 py-4">
                                  <div className="flex items-center justify-center">
                                    <div className="relative">
                                      <div className={`absolute inset-0 bg-gradient-to-br ${config.numberBg} rounded-lg blur opacity-20 group-hover:opacity-40 transition-opacity`} />
                                      <div className={`relative w-8 h-8 rounded-lg bg-gradient-to-br ${config.numberBg} flex items-center justify-center shadow-md group-hover:shadow-lg group-hover:scale-110 transition-all duration-200`}>
                                        <span className="text-xs font-bold text-white">
                                          {globalIdx + 1}
                                        </span>
                                      </div>
                                    </div>
                                  </div>
                                </td>

                                {/* Severity Column */}
                                <td className="px-5 py-4">
                                  <Badge className={`${config.badge} text-xs font-bold capitalize px-3 py-1.5 group-hover:shadow-lg transition-all duration-200`}>
                                    {problem.severity}
                                  </Badge>
                                </td>

                                {/* Category Column */}
                                <td className="px-5 py-4">
                                  <Badge variant="outline" className="text-xs font-medium px-2.5 py-1 border-2 group-hover:border-orange-500/50 group-hover:bg-orange-500/5 transition-all duration-200">
                                    {formatKey(problem.category)}
                                  </Badge>
                                </td>

                                {/* Component Column */}
                                <td className="px-5 py-4 max-w-xs">
                                  <div className="flex items-center gap-3 min-w-0">
                                    <div className="relative flex-shrink-0">
                                      <div className={`absolute inset-0 bg-gradient-to-br ${config.numberBg} rounded-lg blur opacity-10 group-hover:opacity-25 transition-opacity`} />
                                      <div className={`relative rounded-lg ${config.icon} p-2 group-hover:from-orange-500/25 group-hover:to-red-500/20 transition-all duration-200`}>
                                        <Box className={`h-4 w-4 ${config.iconColor}`} />
                                      </div>
                                    </div>
                                    <div className="min-w-0 flex-1">
                                      {(() => {
                                        // Check if entity_id is a link (contains "->")
                                        const isLink = problem.entity_id.includes('->')
                                        
                                        if (isLink) {
                                          // Parse the link to extract source and target
                                          const [sourceId, targetId] = problem.entity_id.split('->').map(s => s.trim())
                                          
                                          // Try to find the edge in analysisData
                                          const edge = analysisData.edges?.find(e => `${e.source}->${e.target}` === problem.entity_id)
                                          
                                          // If edge found, use its source_name and target_name
                                          if (edge) {
                                            return (
                                              <div className="space-y-0.5 min-w-0">
                                                <div className="flex items-center gap-1.5 text-sm font-semibold" title={`${edge.source} → ${edge.target}`}>
                                                  <button
                                                    onClick={() => router.push(`/explorer?node=${encodeURIComponent(edge.source)}`)}
                                                    className="truncate group-hover:text-orange-600 dark:group-hover:text-orange-400 hover:text-orange-700 dark:hover:text-orange-300 hover:underline transition-colors"
                                                  >
                                                    {edge.source_name || edge.source}
                                                  </button>
                                                  <span className="text-muted-foreground flex-shrink-0">→</span>
                                                  <button
                                                    onClick={() => router.push(`/explorer?node=${encodeURIComponent(edge.target)}`)}
                                                    className="truncate group-hover:text-orange-600 dark:group-hover:text-orange-400 hover:text-orange-700 dark:hover:text-orange-300 hover:underline transition-colors"
                                                  >
                                                    {edge.target_name || edge.target}
                                                  </button>
                                                </div>
                                                {(edge.source_name || edge.target_name) && (
                                                  <div className="flex items-center gap-1.5 text-xs text-muted-foreground/70 font-mono" title={`${edge.source} → ${edge.target}`}>
                                                    <span className="truncate">{edge.source.length > 15 ? edge.source.substring(0, 15) + '...' : edge.source}</span>
                                                    <span className="flex-shrink-0">→</span>
                                                    <span className="truncate">{edge.target.length > 15 ? edge.target.substring(0, 15) + '...' : edge.target}</span>
                                                  </div>
                                                )}
                                              </div>
                                            )
                                          } else {
                                            // Edge not found in edges array, but we can still parse it
                                            // Try to find component names from the components array
                                            const sourceComponent = analysisData.components?.find(c => c.id === sourceId)
                                            const targetComponent = analysisData.components?.find(c => c.id === targetId)
                                            const sourceName = sourceComponent?.name || sourceId
                                            const targetName = targetComponent?.name || targetId
                                            const showIds = (sourceComponent?.name && sourceComponent.name !== sourceId) || (targetComponent?.name && targetComponent.name !== targetId)
                                            
                                            return (
                                              <div className="space-y-0.5 min-w-0">
                                                <div className="flex items-center gap-1.5 text-sm font-semibold" title={`${sourceId} → ${targetId}`}>
                                                  <button
                                                    onClick={() => router.push(`/explorer?node=${encodeURIComponent(sourceId)}`)}
                                                    className="truncate group-hover:text-orange-600 dark:group-hover:text-orange-400 hover:text-orange-700 dark:hover:text-orange-300 hover:underline transition-colors"
                                                  >
                                                    {sourceName}
                                                  </button>
                                                  <span className="text-muted-foreground flex-shrink-0">→</span>
                                                  <button
                                                    onClick={() => router.push(`/explorer?node=${encodeURIComponent(targetId)}`)}
                                                    className="truncate group-hover:text-orange-600 dark:group-hover:text-orange-400 hover:text-orange-700 dark:hover:text-orange-300 hover:underline transition-colors"
                                                  >
                                                    {targetName}
                                                  </button>
                                                </div>
                                                {showIds && (
                                                  <div className="flex items-center gap-1.5 text-xs text-muted-foreground/70 font-mono" title={`${sourceId} → ${targetId}`}>
                                                    <span className="truncate">{sourceId.length > 15 ? sourceId.substring(0, 15) + '...' : sourceId}</span>
                                                    <span className="flex-shrink-0">→</span>
                                                    <span className="truncate">{targetId.length > 15 ? targetId.substring(0, 15) + '...' : targetId}</span>
                                                  </div>
                                                )}
                                              </div>
                                            )
                                          }
                                        } else {
                                          // Handle component/node
                                          const component = analysisData.components?.find(c => c.id === problem.entity_id)
                                          return component && component.name && component.name !== component.id ? (
                                            <div className="space-y-0.5 min-w-0">
                                              <button
                                                onClick={() => router.push(`/explorer?node=${encodeURIComponent(problem.entity_id)}`)}
                                                className="text-sm font-semibold truncate group-hover:text-orange-600 dark:group-hover:text-orange-400 hover:text-orange-700 dark:hover:text-orange-300 hover:underline transition-colors text-left w-full"
                                                title={component.name}
                                              >
                                                {component.name}
                                              </button>
                                              <div className="text-xs text-muted-foreground/70 truncate font-mono" title={problem.entity_id}>
                                                {problem.entity_id}
                                              </div>
                                            </div>
                                          ) : (
                                            <button
                                              onClick={() => router.push(`/explorer?node=${encodeURIComponent(problem.entity_id)}`)}
                                              className="text-sm font-semibold text-muted-foreground truncate font-mono group-hover:text-orange-600 dark:group-hover:text-orange-400 hover:text-orange-700 dark:hover:text-orange-300 hover:underline transition-colors text-left w-full"
                                              title={problem.entity_id}
                                            >
                                              {problem.entity_id}
                                            </button>
                                          )
                                        }
                                      })()}
                                    </div>
                                  </div>
                                </td>

                                {/* Issue Column */}
                                <td className="px-5 py-4 max-w-md">
                                  <div className="space-y-1">
                                    {problem.name && (
                                      <p className="text-sm font-semibold text-foreground truncate" title={problem.name}>
                                        {problem.name}
                                      </p>
                                    )}
                                    <p className="text-sm leading-relaxed text-muted-foreground line-clamp-2" title={problem.description}>
                                      {problem.description}
                                    </p>
                                  </div>
                                </td>

                                {/* Recommendation Column */}
                                <td className="px-5 py-4 max-w-md">
                                  <p className="text-sm text-muted-foreground leading-relaxed line-clamp-2" title={problem.recommendation}>
                                    {problem.recommendation}
                                  </p>
                                </td>
                              </tr>
                            )
                          })}
                        </tbody>
                      </table>
                    </div>
                    </>
                    )}
                  </Card>

                {/* Pagination Card */}
                {totalIssuesPages > 1 && (
                  <Card className="relative overflow-hidden border-0 p-0 shadow-lg">
                    {/* Gradient border */}
                    <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-slate-400 via-gray-500 to-slate-600">
                      <div className="w-full h-full bg-background rounded-lg" />
                    </div>
                    {/* Subtle background glow */}
                    <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_center,var(--tw-gradient-stops))] from-slate-500/10 via-transparent to-transparent" />
                    
                    <div className="relative px-6 py-4">
                        <div className="flex items-center justify-between gap-4">
                          {/* Page Info - Left Side */}
                          <div className="flex items-center gap-3">
                            <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-background border border-border shadow-sm">
                              <span className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">Page</span>
                              <span className="text-sm font-bold text-foreground">{issuesPage}</span>
                              <span className="text-xs text-muted-foreground">/</span>
                              <span className="text-sm font-bold text-foreground">{totalIssuesPages}</span>
                            </div>
                            <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-background/50 border border-border/50">
                              <span className="text-xs text-muted-foreground">Showing</span>
                              <span className="text-sm font-semibold text-foreground">
                                {(issuesPage - 1) * issuesPerPage + 1}-{Math.min(issuesPage * issuesPerPage, filteredIssues.length)}
                              </span>
                              <span className="text-xs text-muted-foreground">of</span>
                              <span className="text-sm font-semibold text-foreground">{filteredIssues.length}</span>
                            </div>
                          </div>
                        
                        {/* Navigation - Right Side */}
                        <div className="flex items-center gap-1.5">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setIssuesPage(1)}
                            disabled={issuesPage === 1}
                            className="h-9 w-9 p-0 rounded-lg border-2 hover:bg-slate-500/10 hover:border-slate-500/50 hover:shadow-md disabled:opacity-40 transition-all duration-200 hover:scale-105"
                            title="First page"
                          >
                            <ChevronsLeft className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setIssuesPage(p => p - 1)}
                            disabled={issuesPage === 1}
                            className="h-9 px-3.5 rounded-lg border-2 hover:bg-slate-500/10 hover:border-slate-500/50 hover:shadow-md disabled:opacity-40 transition-all duration-200 hover:scale-105"
                            title="Previous page"
                          >
                            <ChevronLeft className="h-4 w-4 mr-1" />
                            <span className="text-xs font-semibold">Prev</span>
                          </Button>
                          
                          {/* Page Numbers */}
                          <div className="hidden lg:flex items-center gap-1.5 mx-1">
                            {Array.from({ length: totalIssuesPages }, (_, i) => i + 1)
                              .filter(page => {
                                if (totalIssuesPages <= 7) return true
                                if (page === 1 || page === totalIssuesPages) return true
                                if (Math.abs(page - issuesPage) <= 1) return true
                                return false
                              })
                              .map((page, idx, arr) => (
                                <React.Fragment key={page}>
                                  {idx > 0 && arr[idx - 1] !== page - 1 && (
                                    <span className="px-2 text-muted-foreground text-xs font-bold">···</span>
                                  )}
                                  <Button
                                    variant={issuesPage === page ? 'default' : 'outline'}
                                    size="sm"
                                    onClick={() => setIssuesPage(page)}
                                    className={`h-9 min-w-9 px-2.5 text-xs font-bold rounded-lg border-2 transition-all duration-200 ${
                                      issuesPage === page
                                        ? 'bg-gradient-to-r from-slate-600 to-gray-700 text-white shadow-lg shadow-slate-500/30 hover:from-slate-700 hover:to-gray-800 border-0 scale-110'
                                        : 'hover:bg-slate-500/10 hover:border-slate-500/50 hover:shadow-md hover:scale-105'
                                    }`}
                                  >
                                    {page}
                                  </Button>
                                </React.Fragment>
                              ))}
                          </div>
                          
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setIssuesPage(p => p + 1)}
                            disabled={issuesPage === totalIssuesPages}
                            className="h-9 px-3.5 rounded-lg border-2 hover:bg-slate-500/10 hover:border-slate-500/50 hover:shadow-md disabled:opacity-40 transition-all duration-200 hover:scale-105"
                            title="Next page"
                          >
                            <span className="text-xs font-semibold">Next</span>
                            <ChevronRight className="h-4 w-4 ml-1" />
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => setIssuesPage(totalIssuesPages)}
                            disabled={issuesPage === totalIssuesPages}
                            className="h-9 w-9 p-0 rounded-lg border-2 hover:bg-slate-500/10 hover:border-slate-500/50 hover:shadow-md disabled:opacity-40 transition-all duration-200 hover:scale-105"
                            title="Last page"
                          >
                            <ChevronsRight className="h-4 w-4" />
                          </Button>
                        </div>
                        </div>
                    </div>
                  </Card>
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

        {/* Empty State - No Analysis Yet */}
        {!isLoading && !analysisData && !error && (
          <Card className="relative overflow-hidden border-2 border-dashed border-slate-300 dark:border-slate-700 bg-gradient-to-br from-slate-50/50 via-white to-blue-50/30 dark:from-slate-950/50 dark:via-background dark:to-blue-950/20 shadow-lg">
            {/* Animated background decoration */}
            <div className="absolute inset-0 opacity-5 dark:opacity-10">
              <div className="absolute top-10 right-10 w-32 h-32 bg-blue-500 rounded-full blur-3xl animate-pulse" />
              <div className="absolute bottom-10 left-10 w-40 h-40 bg-purple-500 rounded-full blur-3xl animate-pulse delay-700" />
            </div>
            
            <CardContent className="relative py-10 text-center">
              {/* Icon with gradient background */}
              <div className="relative w-fit mx-auto mb-6">
                <div className="absolute inset-0 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full blur-xl opacity-30 animate-pulse" />
                <div className="relative rounded-2xl bg-gradient-to-br from-blue-500 via-purple-500 to-pink-500 p-5 shadow-xl">
                  <Target className="h-8 w-8 text-white" />
                </div>
              </div>
              
              {/* Title with gradient */}
              <h3 className="text-2xl font-bold mb-2 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent">
                Ready to Analyze
              </h3>
              
              {/* Description */}
              <p className="text-sm text-muted-foreground/90 max-w-lg mx-auto mb-6 leading-relaxed">
                Configure your analysis scope above and click <span className="font-semibold text-foreground">&quot;Run Analysis&quot;</span> to evaluate your system&apos;s quality attributes
              </p>
              
              {/* Feature highlights */}
              <div className="grid gap-4 md:grid-cols-3 max-w-3xl mx-auto mt-10">
                <div className="flex flex-col items-center gap-2 p-4 rounded-xl bg-white/60 dark:bg-slate-900/40 border border-slate-200/60 dark:border-slate-800/60">
                  <div className="rounded-lg bg-blue-100 dark:bg-blue-900/50 p-2.5">
                    <Shield className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                  </div>
                  <div className="text-sm font-semibold text-foreground">Reliability Check</div>
                  <div className="text-xs text-muted-foreground text-center">Assess system stability</div>
                </div>
                
                <div className="flex flex-col items-center gap-2 p-4 rounded-xl bg-white/60 dark:bg-slate-900/40 border border-slate-200/60 dark:border-slate-800/60">
                  <div className="rounded-lg bg-purple-100 dark:bg-purple-900/50 p-2.5">
                    <Wrench className="h-5 w-5 text-purple-600 dark:text-purple-400" />
                  </div>
                  <div className="text-sm font-semibold text-foreground">Maintainability</div>
                  <div className="text-xs text-muted-foreground text-center">Evaluate code quality</div>
                </div>
                
                <div className="flex flex-col items-center gap-2 p-4 rounded-xl bg-white/60 dark:bg-slate-900/40 border border-slate-200/60 dark:border-slate-800/60">
                  <div className="rounded-lg bg-green-100 dark:bg-green-900/50 p-2.5">
                    <Zap className="h-5 w-5 text-green-600 dark:text-green-400" />
                  </div>
                  <div className="text-sm font-semibold text-foreground">Availability</div>
                  <div className="text-xs text-muted-foreground text-center">Check uptime metrics</div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </AppLayout>
  )
}
