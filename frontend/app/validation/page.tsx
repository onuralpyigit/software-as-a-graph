"use client"

import React, { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import {
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Loader2,
  Play,
  Clock,
  Layers,
  Target,
  BarChart3,
  Info,
  ShieldCheck,
  GitCompare,
  Sparkles,
  Settings,
  Activity,
  Network,
  Database,
  Box,
} from "lucide-react"
import { useConnection } from "@/lib/stores/connection-store"
import { validationClient, PipelineResult, LayerValidationResult, ValidationTargets, LayerDefinition } from "@/lib/api/validation-client"

// ============================================================================
// Types
// ============================================================================

interface ValidationState {
  isRunning: boolean
  result: PipelineResult | null
  error: string | null
  startTime: number | null
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Normalize validation result to ensure it has the expected structure
 * Handles both old and new API response formats
 */
function normalizeValidationResult(result: any): PipelineResult | null {
  if (!result) return null

  try {
    // If result already has summary property, ensure it has all required fields
    if (result.summary) {
      return {
        timestamp: result.timestamp || new Date().toISOString(),
        summary: {
          total_components: result.summary.total_components || 0,
          layers_validated: result.summary.layers_validated || (result.layers ? Object.keys(result.layers).length : 0),
          layers_passed: result.summary.layers_passed || 0,
          all_passed: result.summary.all_passed || false,
        },
        layers: result.layers || {},
        cross_layer_insights: result.cross_layer_insights || [],
        targets: result.targets || null,
      }
    }

    // Transform old format to new format
    return {
      timestamp: result.timestamp || new Date().toISOString(),
      summary: {
        total_components: result.total_components || 0,
        layers_validated: result.layers ? Object.keys(result.layers).length : 0,
        layers_passed: result.layers_passed || 0,
        all_passed: result.all_passed || false,
      },
      layers: result.layers || {},
      cross_layer_insights: result.warnings || result.cross_layer_insights || [],
      targets: result.targets || null,
    }
  } catch (error) {
    console.error('Error normalizing validation result:', error)
    return null
  }
}

/**
 * Clean layer description by removing π prefix notation
 * Converts "π_app: Analyse Applications..." to "Analyse Applications..."
 */
function cleanLayerDescription(description: string): string {
  return description.replace(/^π_\w+:\s*/, '')
}

// ============================================================================
// Main Component
// ============================================================================

export default function ValidationPage() {
  const router = useRouter()
  const { config, status, stats, initialLoadComplete } = useConnection()
  const isConnected = status === 'connected'
  
  const [state, setState] = useState<ValidationState>({
    isRunning: false,
    result: null,
    error: null,
    startTime: null,
  })
  
  const [selectedLayers, setSelectedLayers] = useState<string[]>(["application", "infrastructure", "system"])
  const [availableLayers, setAvailableLayers] = useState<Record<string, LayerDefinition>>({
    application: {
      name: "Application Layer",
      description: "Application components only",
      component_types: ["Application"],
    },
    infrastructure: {
      name: "Infrastructure Layer",
      description: "Infrastructure nodes only",
      component_types: ["Node"],
    },
    middleware: {
      name: "Middleware Layer",
      description: "Applications, Nodes and Brokers",
      component_types: ["Application", "Node", "Broker"],
    },
    system: {
      name: "Complete System",
      description: "All components",
      component_types: ["Application", "Broker", "Node"],
    },
  })
  const [validationTargets, setValidationTargets] = useState<ValidationTargets | null>(null)
  const [includeComparisons, setIncludeComparisons] = useState(true)
  
  // Ref for scrolling to results section
  const resultsRef = React.useRef<HTMLDivElement>(null)

  // Load persisted validation results on mount
  useEffect(() => {
    const savedResult = localStorage.getItem('validation-result')
    if (savedResult) {
      try {
        const parsed = JSON.parse(savedResult)
        const normalized = normalizeValidationResult(parsed)
        if (normalized) {
          setState(prev => ({ ...prev, result: normalized }))
        }
      } catch (error) {
        console.error('Failed to load saved validation result:', error)
        localStorage.removeItem('validation-result')
      }
    }
  }, [])

  // Load available layers and targets
  useEffect(() => {
    const loadMetadata = async () => {
      try {
        const [layers, targets] = await Promise.all([
          validationClient.getLayers(),
          validationClient.getTargets()
        ])
        setAvailableLayers(layers)
        setValidationTargets(targets)
      } catch (error) {
        console.error("Failed to load validation metadata:", error)
      }
    }
    
    loadMetadata()
  }, [])

  // Sync credentials with validation client
  useEffect(() => {
    if (config) {
      validationClient.setCredentials(config)
    }
  }, [config])

  const runValidation = async () => {
    if (!isConnected || !config) {
      return
    }

    setState({
      isRunning: true,
      result: null,
      error: null,
      startTime: Date.now(),
    })

    // Scroll to results section
    setTimeout(() => {
      resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 100)

    try {
      const result = await validationClient.runPipeline(selectedLayers, includeComparisons)
      
      // Normalize and save result to localStorage
      const normalized = normalizeValidationResult(result)
      if (normalized) {
        localStorage.setItem('validation-result', JSON.stringify(normalized))
        
        setState({
          isRunning: false,
          result: normalized,
          error: null,
          startTime: null,
        })
      } else {
        throw new Error('Invalid validation result structure')
      }
    } catch (error: any) {
      // Clear saved result on error
      localStorage.removeItem('validation-result')
      
      setState({
        isRunning: false,
        result: null,
        error: error.message || "Validation failed",
        startTime: null,
      })
    }
  }

  const clearResults = () => {
    localStorage.removeItem('validation-result')
    setState(prev => ({ ...prev, result: null, error: null }))
  }

  const toggleLayer = (layer: string) => {
    setSelectedLayers(prev => 
      prev.includes(layer) 
        ? prev.filter(l => l !== layer)
        : [...prev, layer]
    )
  }

  // Track elapsed time with a timer
  const [elapsedTime, setElapsedTime] = useState(0)
  
  // Estimated duration for progress calculation (in seconds)
  const estimatedDuration = 60

  useEffect(() => {
    if (state.isRunning && state.startTime) {
      const interval = setInterval(() => {
        setElapsedTime(Math.floor((Date.now() - state.startTime!) / 1000))
      }, 1000)

      return () => clearInterval(interval)
    } else {
      setElapsedTime(0)
    }
  }, [state.isRunning, state.startTime])

  // Loading State
  if (!initialLoadComplete || status === 'connecting') {
    return (
      <AppLayout
        title="Validation"
        description="Validate graph analysis predictions against failure simulation results"
      >
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" text={status === 'connecting' ? "Connecting to database..." : "Loading validation..."} />
        </div>
      </AppLayout>
    )
  }

  // Disconnected State
  if (!isConnected) {
    return (
      <AppLayout
        title="Validation"
        description="Validate graph analysis predictions against failure simulation results"
      >
        <NoConnectionInfo description="Connect to your Neo4j database to run validation" />
      </AppLayout>
    )
  }

  // Empty database state
  if (stats && stats.total_nodes === 0) {
    return (
      <AppLayout
        title="Validation"
        description="Validate graph analysis predictions against failure simulation results"
      >
        <div className="w-full">
          <Card className="border-2 border-purple-500/50 dark:border-purple-500/50 bg-white/95 dark:bg-black/95 backdrop-blur-md shadow-2xl shadow-purple-500/20 hover:shadow-purple-500/30 hover:border-purple-500/70 transition-all duration-300 overflow-hidden">
            <div className="h-1 w-full bg-gradient-to-r from-purple-500 via-pink-500 to-purple-500" />
            <CardHeader className="pb-6 pt-8 px-8">
              <div className="flex flex-col sm:flex-row items-start sm:items-center gap-5">
                <div className="relative group">
                  <div className="absolute inset-0 bg-gradient-to-br from-purple-500 to-purple-600 rounded-2xl blur-xl opacity-30 group-hover:opacity-50 transition-opacity duration-300" />
                  <div className="relative rounded-2xl bg-gradient-to-br from-purple-500/20 to-purple-600/20 dark:from-purple-500/30 dark:to-purple-600/30 p-4 ring-1 ring-purple-500/30 group-hover:ring-purple-500/50 transition-all duration-300">
                    <Database className="h-8 w-8 text-purple-600 dark:text-purple-400 group-hover:scale-110 transition-transform duration-300" />
                  </div>
                </div>
                <div className="flex-1 space-y-1.5">
                  <CardTitle className="text-2xl font-bold tracking-tight">Empty Database</CardTitle>
                  <CardDescription className="text-base text-muted-foreground">
                    No graph data available for validation
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="px-8 pb-8 space-y-6">
              <div className="rounded-2xl bg-gradient-to-br from-muted/40 via-muted/20 to-muted/10 border border-border/40 p-6 space-y-5">
                <div className="flex items-center gap-2.5">
                  <div className="rounded-xl bg-gradient-to-br from-blue-500/20 to-blue-600/20 dark:from-blue-500/30 dark:to-blue-600/30 p-2.5 ring-1 ring-blue-500/30">
                    <Info className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                  </div>
                  <h3 className="font-semibold text-base text-foreground">How to Populate Your Database</h3>
                </div>
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

  return (
    <AppLayout
      title="Validation"
      description="Validate graph analysis predictions against failure simulation results"
    >
      <div className="space-y-6">
        {/* Configuration Section */}
        <div className="space-y-4">
          {/* Header Section */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-gradient-to-br from-blue-600 via-purple-600 to-pink-600 p-3 shadow-lg">
                <Settings className="h-6 w-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold">Validation Configuration</h2>
                <p className="text-sm text-muted-foreground">Configure validation layers and run statistical validation</p>
              </div>
            </div>
            <Button
              onClick={runValidation}
              disabled={state.isRunning || selectedLayers.length === 0}
              size="lg"
              className="min-w-[180px] h-12 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 hover:from-blue-700 hover:via-purple-700 hover:to-pink-700 shadow-lg hover:shadow-xl transition-all text-base font-semibold"
            >
              {state.isRunning ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  Validating...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-5 w-5" />
                  Run Validation
                </>
              )}
            </Button>
          </div>

          {/* Layer Selection */}
          <div className="space-y-3">
            <Label className="text-base font-semibold flex items-center gap-2">
              <Layers className="h-4 w-4" />
              Select Validation Layers
            </Label>
            <p className="text-sm text-muted-foreground/80">
              Choose which architectural layers to validate. Each layer represents a different view of your system.
            </p>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 pt-2">
                  {Object.entries(availableLayers).map(([key, layer]) => {
                    const isSelected = selectedLayers.includes(key)
                    return (
                      <Card
                        key={key}
                        className={`group relative cursor-pointer transition-all duration-300 ease-in-out overflow-hidden ${
                          isSelected
                            ? 'border-0 shadow-2xl shadow-blue-500/25 scale-[1.02]'
                            : 'border-0 hover:shadow-xl hover:shadow-blue-500/25 hover:scale-[1.01]'
                        }`}
                        onClick={() => toggleLayer(key)}
                      >
                        {/* Gradient border */}
                        <div className={`absolute inset-0 rounded-lg p-[2px] transition-opacity duration-300 ${
                          isSelected
                            ? 'bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 opacity-100'
                            : 'bg-gradient-to-br from-slate-200 via-slate-300 to-slate-200 dark:from-slate-700 dark:via-slate-800 dark:to-slate-700 opacity-100 group-hover:from-blue-500 group-hover:via-purple-500 group-hover:to-pink-500'
                        }`}>
                          <div className="w-full h-full bg-background rounded-lg" />
                        </div>
                        
                        {/* Background gradient overlay */}
                        <div className={`absolute inset-[2px] rounded-lg transition-opacity duration-300 ${
                          isSelected
                            ? 'bg-gradient-to-br from-blue-500/15 via-purple-500/10 to-pink-500/5 opacity-100'
                            : 'bg-gradient-to-br from-blue-500/5 via-purple-500/3 to-transparent opacity-0 group-hover:opacity-100'
                        }`} />
                        
                        <CardContent className="relative p-6">
                          <div className="space-y-4">
                            {/* Checkbox and Icon Section */}
                            <div className="flex items-start gap-4">
                              {/* Checkbox indicator */}
                              <Checkbox
                                id={`layer-${key}`}
                                checked={isSelected}
                                onCheckedChange={() => toggleLayer(key)}
                                className="mt-1"
                              />
                              
                              {/* Icon */}
                              <div className={`relative rounded-2xl p-3 transition-all duration-300 ${
                                isSelected
                                  ? 'bg-gradient-to-br from-blue-500 to-purple-600 shadow-lg shadow-blue-500/30'
                                  : 'bg-gradient-to-br from-blue-100 to-purple-50 dark:from-blue-900/50 dark:to-purple-900/30 group-hover:scale-105'
                              }`}>
                                <Layers className={`h-5 w-5 transition-all duration-300 ${
                                  isSelected
                                    ? 'text-white'
                                    : 'text-blue-600 dark:text-blue-400'
                                }`} />
                                {isSelected && (
                                  <div className="absolute inset-0 rounded-2xl bg-blue-500 animate-ping opacity-20" />
                                )}
                              </div>
                            </div>
                            
                            {/* Content Section */}
                            <div className="space-y-2 pl-9">
                              <div className="flex items-center gap-2 flex-wrap">
                                <Label
                                  htmlFor={`layer-${key}`}
                                  className={`text-base font-bold tracking-tight cursor-pointer transition-colors duration-200 ${
                                    isSelected
                                      ? 'text-blue-700 dark:text-blue-300'
                                      : 'text-foreground group-hover:text-blue-700 dark:group-hover:text-blue-300'
                                  }`}
                                >
                                  {layer.name}
                                </Label>
                                {isSelected && (
                                  <Badge className="bg-blue-500/10 text-blue-700 dark:text-blue-300 border-blue-500/20 text-xs">
                                    Selected
                                  </Badge>
                                )}
                              </div>
                              <p className="text-sm leading-relaxed text-muted-foreground/90">
                                {cleanLayerDescription(layer.description)}
                              </p>
                              <div className="flex flex-wrap gap-1 pt-1">
                                {layer.component_types.map((type: string) => (
                                  <Badge key={type} variant="outline" className="text-xs">
                                    {type}
                                  </Badge>
                                ))}
                              </div>
                              {!isSelected && (
                                <p className="text-xs text-muted-foreground/60 flex items-center gap-1 mt-2">
                                  <Info className="h-3 w-3" />
                                  Click to select this layer
                                </p>
                              )}
                            </div>
                            
                            {/* Selection indicator bar */}
                            {isSelected && (
                              <div className="absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r from-blue-500 via-purple-600 to-pink-600" />
                            )}
                          </div>
                        </CardContent>
                      </Card>
                    )
                  })}
                </div>
              </div>

              {/* Options */}
              <div className="space-y-3">
                <Label className="text-base font-semibold flex items-center gap-2">
                  <Settings className="h-4 w-4" />
                  Additional Options
                </Label>
                <div className="space-y-3">
                  <div className="flex items-start space-x-3 p-4 rounded-lg border bg-muted/20 hover:bg-muted/30 transition-colors">
                    <Checkbox
                      id="include-comparisons"
                      checked={includeComparisons}
                      onCheckedChange={(checked) => setIncludeComparisons(!!checked)}
                      className="mt-0.5"
                    />
                    <div className="flex-1">
                      <Label
                        htmlFor="include-comparisons"
                        className="font-medium cursor-pointer text-sm"
                      >
                        Include detailed component comparisons
                      </Label>
                      <p className="text-xs text-muted-foreground mt-1">
                        Generate detailed component-by-component comparison data with predicted vs actual scores
                      </p>
                    </div>
                  </div>
                </div>
              </div>
          </div>

        {/* Results Section - with ref for scrolling */}
        <div ref={resultsRef}>
          {/* Running State */}
          {state.isRunning && (
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
                    <p className="text-xl font-bold">Running Validation Pipeline...</p>
                    <p className="text-sm text-muted-foreground max-w-md">
                      Analyzing graph structure, simulating failures, and computing statistical validation metrics
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

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-4">
                  <div className="flex items-center gap-3 p-4 rounded-lg bg-muted/30 border">
                    <div className="rounded-lg bg-blue-500/10 p-2">
                      <BarChart3 className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Step 1</p>
                      <p className="text-sm font-semibold">Graph Analysis</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3 p-4 rounded-lg bg-muted/30 border">
                    <div className="rounded-lg bg-purple-500/10 p-2">
                      <Activity className="h-5 w-5 text-purple-600 dark:text-purple-400" />
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Step 2</p>
                      <p className="text-sm font-semibold">Failure Simulation</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3 p-4 rounded-lg bg-muted/30 border">
                    <div className="rounded-lg bg-pink-500/10 p-2">
                      <Target className="h-5 w-5 text-pink-600 dark:text-pink-400" />
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Step 3</p>
                      <p className="text-sm font-semibold">Validation Metrics</p>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Error State */}
        {state.error && (
          <Card className="border-0 shadow-xl relative overflow-hidden">
            <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-red-500 via-rose-500 to-pink-500">
              <div className="w-full h-full bg-background rounded-lg" />
            </div>
            <div className="absolute inset-[2px] rounded-lg bg-gradient-to-br from-red-500/10 via-rose-500/5 to-pink-500/5" />
            <CardHeader className="relative">
              <div className="flex items-center gap-3">
                <div className="rounded-2xl bg-gradient-to-br from-red-500 to-rose-600 p-3 shadow-lg shadow-red-500/30">
                  <XCircle className="h-6 w-6 text-white" />
                </div>
                <div>
                  <CardTitle className="text-xl">Validation Failed</CardTitle>
                  <CardDescription>An error occurred during the validation process</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="relative">
              <div className="p-4 rounded-lg bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-900">
                <p className="text-sm text-red-800 dark:text-red-200 font-mono">{state.error}</p>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Results */}
        {state.result && state.result.summary && (
          <div className="space-y-4">
            {/* Overall Summary */}
            <Card className={`border-0 shadow-2xl relative overflow-hidden ${
              state.result.summary.all_passed
                ? "shadow-green-500/20"
                : "shadow-yellow-500/20"
            }`}>
              {/* Gradient border */}
              <div className={`absolute inset-0 rounded-lg p-[2px] ${
                state.result.summary.all_passed
                  ? "bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500"
                  : "bg-gradient-to-r from-yellow-500 via-amber-500 to-orange-500"
              }`}>
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              
              {/* Background gradient overlay */}
              <div className={`absolute inset-[2px] rounded-lg ${
                state.result.summary.all_passed
                  ? "bg-gradient-to-br from-green-500/10 via-emerald-500/5 to-teal-500/5"
                  : "bg-gradient-to-br from-yellow-500/10 via-amber-500/5 to-orange-500/5"
              }`} />
              
              <CardHeader className="relative">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    {state.result.summary.all_passed ? (
                      <div className="rounded-2xl bg-gradient-to-br from-green-500 to-emerald-600 p-3 shadow-lg shadow-green-500/30">
                        <CheckCircle2 className="h-7 w-7 text-white" />
                      </div>
                    ) : (
                      <div className="rounded-2xl bg-gradient-to-br from-yellow-500 to-amber-600 p-3 shadow-lg shadow-yellow-500/30">
                        <AlertTriangle className="h-7 w-7 text-white" />
                      </div>
                    )}
                    <div>
                      <CardTitle className="text-2xl font-bold">
                        {state.result.summary.all_passed ? "All Validations Passed ✓" : "Some Validations Failed"}
                      </CardTitle>
                      <CardDescription className="text-base">
                        {state.result.summary.layers_passed} of {state.result.summary.layers_validated} layers passed validation criteria
                      </CardDescription>
                    </div>
                  </div>
                  <Button
                    onClick={clearResults}
                    variant="outline"
                    size="sm"
                    className="flex items-center gap-2"
                  >
                    <XCircle className="h-4 w-4" />
                    Clear Results
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="relative">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="space-y-1 p-4 rounded-lg bg-muted/30">
                    <p className="text-sm text-muted-foreground">Total Components</p>
                    <p className="text-2xl font-bold">{state.result.summary.total_components}</p>
                  </div>
                  <div className="space-y-1 p-4 rounded-lg bg-muted/30">
                    <p className="text-sm text-muted-foreground">Layers Validated</p>
                    <p className="text-2xl font-bold">{state.result.summary.layers_validated}</p>
                  </div>
                  <div className="space-y-1 p-4 rounded-lg bg-muted/30">
                    <p className="text-sm text-muted-foreground">Layers Passed</p>
                    <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                      {state.result.summary.layers_passed}
                    </p>
                  </div>
                  <div className="space-y-1 p-4 rounded-lg bg-muted/30">
                    <p className="text-sm text-muted-foreground">Success Rate</p>
                    <p className="text-2xl font-bold">
                      {Math.round((state.result.summary.layers_passed / state.result.summary.layers_validated) * 100)}%
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Validation Targets */}
            {validationTargets && (
              <Card className="border-0 shadow-xl relative overflow-hidden">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-slate-200 via-slate-300 to-slate-200 dark:from-slate-700 dark:via-slate-800 dark:to-slate-700">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <CardHeader className="relative">
                  <CardTitle className="flex items-center gap-2">
                    <Target className="h-5 w-5" />
                    Validation Targets
                  </CardTitle>
                  <CardDescription>
                    Success criteria for validation metrics
                  </CardDescription>
                </CardHeader>
                <CardContent className="relative">
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="p-3 border rounded-lg bg-muted/20">
                      <p className="text-xs text-muted-foreground">Spearman ρ</p>
                      <p className="text-lg font-bold">≥ {validationTargets.spearman.toFixed(2)}</p>
                    </div>
                    <div className="p-3 border rounded-lg bg-muted/20">
                      <p className="text-xs text-muted-foreground">F1 Score</p>
                      <p className="text-lg font-bold">≥ {validationTargets.f1_score.toFixed(2)}</p>
                    </div>
                    <div className="p-3 border rounded-lg bg-muted/20">
                      <p className="text-xs text-muted-foreground">Precision</p>
                      <p className="text-lg font-bold">≥ {validationTargets.precision.toFixed(2)}</p>
                    </div>
                    <div className="p-3 border rounded-lg bg-muted/20">
                      <p className="text-xs text-muted-foreground">Recall</p>
                      <p className="text-lg font-bold">≥ {validationTargets.recall.toFixed(2)}</p>
                    </div>
                    <div className="p-3 border rounded-lg bg-muted/20">
                      <p className="text-xs text-muted-foreground">Top-5 Overlap</p>
                      <p className="text-lg font-bold">≥ {validationTargets.top_5_overlap.toFixed(2)}</p>
                    </div>
                    <div className="p-3 border rounded-lg bg-muted/20">
                      <p className="text-xs text-muted-foreground">RMSE</p>
                      <p className="text-lg font-bold">≤ {validationTargets.rmse_max.toFixed(2)}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Layer Results */}
            {state.result.layers && Object.entries(state.result.layers).map(([layerKey, layer]) => (
              <LayerResultCard
                key={layerKey}
                layer={layer}
                targets={validationTargets}
              />
            ))}

            {/* Cross-Layer Insights */}
            {state.result.cross_layer_insights && state.result.cross_layer_insights.length > 0 && (
              <Card className="border-0 shadow-xl relative overflow-hidden">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-slate-200 via-slate-300 to-slate-200 dark:from-slate-700 dark:via-slate-800 dark:to-slate-700">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <CardHeader className="relative">
                  <CardTitle className="flex items-center gap-2">
                    <Sparkles className="h-5 w-5" />
                    Cross-Layer Insights
                  </CardTitle>
                </CardHeader>
                <CardContent className="relative">
                  <ul className="space-y-2">
                    {state.result.cross_layer_insights.map((insight, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <Info className="h-4 w-4 mt-1 text-blue-500 flex-shrink-0" />
                        <span>{insight}</span>
                      </li>
                    ))}
                  </ul>
                </CardContent>
              </Card>
            )}
          </div>
        )}
        </div>
      </div>
    </AppLayout>
  )
}

// ============================================================================
// Layer Result Card Component
// ============================================================================

function LayerResultCard({ 
  layer, 
  targets 
}: { 
  layer: LayerValidationResult
  targets: ValidationTargets | null
}) {
  const passed = layer.summary.passed

  return (
    <Card className={`border-0 shadow-xl relative overflow-hidden`}>
      {/* Gradient border */}
      <div className={`absolute inset-0 rounded-lg p-[2px] ${
        passed
          ? 'bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500'
          : 'bg-gradient-to-r from-yellow-500 via-amber-500 to-orange-500'
      }`}>
        <div className="w-full h-full bg-background rounded-lg" />
      </div>
      
      {/* Background gradient overlay */}
      <div className={`absolute inset-[2px] rounded-lg ${
        passed
          ? 'bg-gradient-to-br from-green-500/5 via-emerald-500/3 to-transparent'
          : 'bg-gradient-to-br from-yellow-500/5 via-amber-500/3 to-transparent'
      }`} />
      
      <CardHeader className="relative">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`rounded-2xl p-3 ${
              passed
                ? "bg-gradient-to-br from-green-500 to-emerald-600 shadow-lg shadow-green-500/30"
                : "bg-gradient-to-br from-yellow-500 to-amber-600 shadow-lg shadow-yellow-500/30"
            }`}>
              <Layers className="h-5 w-5 text-white" />
            </div>
            <div>
              <CardTitle>{layer.layer_name}</CardTitle>
              <CardDescription>{layer.layer}</CardDescription>
            </div>
          </div>
          <Badge
            className={passed 
              ? "bg-green-600 hover:bg-green-700 text-white"
              : "bg-yellow-600 hover:bg-yellow-700 text-white"
            }
          >
            {passed ? "Passed" : "Failed"}
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="relative space-y-6">
        {/* Data Alignment */}
        <div>
          <h4 className="font-semibold mb-3 flex items-center gap-2">
            <GitCompare className="h-4 w-4" />
            Data Alignment
          </h4>
          <div className="grid grid-cols-3 gap-4">
            <div className="p-3 border rounded-lg bg-muted/20">
              <p className="text-xs text-muted-foreground">Predicted</p>
              <p className="text-xl font-bold">{layer.data.predicted_components}</p>
            </div>
            <div className="p-3 border rounded-lg bg-muted/20">
              <p className="text-xs text-muted-foreground">Simulated</p>
              <p className="text-xl font-bold">{layer.data.simulated_components}</p>
            </div>
            <div className="p-3 border rounded-lg bg-muted/20">
              <p className="text-xs text-muted-foreground">Matched</p>
              <p className="text-xl font-bold text-blue-600 dark:text-blue-400">
                {layer.data.matched_components}
              </p>
            </div>
          </div>
        </div>

        {/* Metrics Summary */}
        <div>
          <h4 className="font-semibold mb-3 flex items-center gap-2">
            <BarChart3 className="h-4 w-4" />
            Validation Metrics
          </h4>
          <div className="space-y-3">
            {targets && (
              <>
                <MetricRow
                  label="Spearman ρ"
                  value={layer.summary.spearman}
                  target={targets.spearman}
                  higherBetter={true}
                />
                <MetricRow
                  label="F1 Score"
                  value={layer.summary.f1_score}
                  target={targets.f1_score}
                  higherBetter={true}
                />
                <MetricRow
                  label="Precision"
                  value={layer.summary.precision}
                  target={targets.precision}
                  higherBetter={true}
                />
                <MetricRow
                  label="Recall"
                  value={layer.summary.recall}
                  target={targets.recall}
                  higherBetter={true}
                />
                <MetricRow
                  label="Top-5 Overlap"
                  value={layer.summary.top_5_overlap}
                  target={targets.top_5_overlap}
                  higherBetter={true}
                />
                <MetricRow
                  label="RMSE"
                  value={layer.summary.rmse}
                  target={targets.rmse_max}
                  higherBetter={false}
                />
              </>
            )}
          </div>
        </div>

        {/* Warnings */}
        {layer.warnings.length > 0 && (
          <div className="p-4 border border-yellow-300 dark:border-yellow-900 rounded-lg bg-yellow-50 dark:bg-yellow-950/20">
            <h4 className="font-semibold mb-2 flex items-center gap-2 text-yellow-900 dark:text-yellow-100">
              <AlertTriangle className="h-4 w-4" />
              Warnings
            </h4>
            <ul className="space-y-1 text-sm text-yellow-800 dark:text-yellow-200">
              {layer.warnings.map((warning, idx) => (
                <li key={idx}>• {warning}</li>
              ))}
            </ul>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

// ============================================================================
// Metric Row Component
// ============================================================================

function MetricRow({
  label,
  value,
  target,
  higherBetter,
}: {
  label: string
  value: number
  target: number
  higherBetter: boolean
}) {
  const passed = higherBetter ? value >= target : value <= target
  const percentage = (value / target) * 100

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium">{label}</span>
        <div className="flex items-center gap-2">
          <span className={passed ? "text-green-600 dark:text-green-400 font-bold" : "text-yellow-600 dark:text-yellow-400 font-bold"}>
            {value.toFixed(4)}
          </span>
          <span className="text-muted-foreground">
            {higherBetter ? "≥" : "≤"} {target.toFixed(2)}
          </span>
          {passed ? (
            <CheckCircle2 className="h-4 w-4 text-green-600 dark:text-green-400" />
          ) : (
            <XCircle className="h-4 w-4 text-yellow-600 dark:text-yellow-400" />
          )}
        </div>
      </div>
      <div className="relative h-2 bg-muted rounded-full overflow-hidden">
        <div
          className={`h-full transition-all ${
            passed
              ? "bg-gradient-to-r from-green-500 to-green-600"
              : "bg-gradient-to-r from-yellow-500 to-yellow-600"
          }`}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>
    </div>
  )
}
