"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ArrowLeft, Box, AlertTriangle, CheckCircle, Info, ArrowRight, ArrowLeftRight, GitFork } from "lucide-react"
import { useConnection } from "@/lib/stores/connection-store"
import { apiClient } from "@/lib/api/client"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"

interface Component {
  id: string
  name: string
  type: string
  in_degree: number
  out_degree: number
}

interface ComponentIsolationStats {
  total_components: number
  isolated_count: number
  source_count: number
  sink_count: number
  bidirectional_count: number
  isolated_percentage: number
  source_percentage: number
  sink_percentage: number
  bidirectional_percentage: number
  interpretation: string
  category: string
  health: string
  isolated_components: Component[]
  top_sources: Component[]
  top_sinks: Component[]
  top_bidirectional: Component[]
}

export default function ComponentIsolationPage() {
  const router = useRouter()
  const { status, stats: graphStats, initialLoadComplete } = useConnection()
  const [stats, setStats] = useState<ComponentIsolationStats | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const isConnected = status === 'connected'

  useEffect(() => {
    if (isConnected) {
      loadStats()
    }
  }, [isConnected])

  const loadStats = async () => {
    try {
      setLoading(true)
      setError(null)
      const result = await apiClient.getComponentIsolationStats()
      setStats(result.stats)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load statistics')
    } finally {
      setLoading(false)
    }
  }

  // Get interpretation color and icon
  const getInterpretation = () => {
    if (!stats) return { color: 'text-gray-500', icon: Info, label: 'Unknown' }
    
    switch (stats.health) {
      case 'good':
        return { 
          color: 'text-green-500', 
          icon: CheckCircle, 
          label: 'Healthy Architecture'
        }
      case 'fair':
        return { 
          color: 'text-blue-500', 
          icon: Info, 
          label: 'Fair Architecture'
        }
      case 'moderate':
        return { 
          color: 'text-yellow-500', 
          icon: AlertTriangle, 
          label: 'Moderate Concerns'
        }
      case 'poor':
        return { 
          color: 'text-red-500', 
          icon: AlertTriangle, 
          label: 'Needs Attention'
        }
      default:
        return { 
          color: 'text-gray-500', 
          icon: Info, 
          label: 'Unknown'
        }
    }
  }

  const interpretation = getInterpretation()
  const Icon = interpretation.icon

  // Loading State
  if (!initialLoadComplete || status === 'connecting') {
    return (
      <AppLayout title="Component Isolation" description="Analyze component connectivity patterns">
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" text="Connecting to database..." />
        </div>
      </AppLayout>
    )
  }

  // Disconnected State
  if (!isConnected) {
    return (
      <AppLayout title="Component Isolation" description="Analyze component connectivity patterns">
        <NoConnectionInfo />
      </AppLayout>
    )
  }

  return (
    <AppLayout title="Component Isolation" description="Analyze component connectivity patterns">
      <div className="space-y-6">
        {/* Header with Back Button */}
        <div className="flex items-center gap-4">
          <Button
            variant="outline"
            size="sm"
            onClick={() => router.push('/statistics')}
            className="gap-2"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Statistics
          </Button>
        </div>

        {/* Page Header */}
        <Card className="relative overflow-hidden border-0 shadow-xl hover:shadow-2xl hover:shadow-orange-500/25 transition-all duration-300">
          <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-orange-500 via-red-500 to-pink-500">
            <div className="w-full h-full rounded-lg bg-gradient-to-r from-orange-600 via-red-600 to-pink-600" />
          </div>
          
          <CardContent className="p-8 relative text-white">
            <div className="flex items-center gap-4 mb-4">
              <div className="rounded-xl bg-white/20 p-3">
                <Box className="h-8 w-8" />
              </div>
              <div className="flex-1">
                <h3 className="text-3xl font-bold">Component Isolation Analysis</h3>
                <p className="text-white/90 mt-2">
                  Understanding component connectivity and architectural roles
                </p>
              </div>
            </div>
            
            <div className="bg-white/10 rounded-lg p-4 mt-4">
              <div className="flex items-start gap-3">
                <Info className="h-5 w-5 mt-0.5 flex-shrink-0" />
                <div className="text-sm">
                  <p className="font-semibold mb-1">What is Component Isolation?</p>
                  <p className="text-white/90">
                    Component isolation analysis categorizes components based on their dependency patterns:
                    <strong> Isolated</strong> (no connections), 
                    <strong> Sources</strong> (only outgoing - entry points), 
                    <strong> Sinks</strong> (only incoming - foundations), and 
                    <strong> Bidirectional</strong> (both directions).
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Loading/Error States */}
        {loading && (
          <div className="flex items-center justify-center py-12">
            <LoadingSpinner size="lg" text="Analyzing component isolation..." />
          </div>
        )}

        {error && (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Stats Display */}
        {stats && !loading && (
          <>
            {/* Distribution Overview Cards */}
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              {/* Isolated Components */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-gray-500/25 transition-all duration-300 transform hover:scale-[1.02]">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-gray-500 via-slate-500 to-zinc-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-gray-500/30 via-gray-500/15 to-gray-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Isolated</CardTitle>
                  <div className="rounded-xl bg-gray-500/10 p-2.5">
                    <Box className="h-4 w-4 text-gray-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-gray-500">
                    {stats.isolated_count}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {stats.isolated_percentage.toFixed(1)}% - No connections
                  </p>
                  <Progress value={stats.isolated_percentage} className="h-1 mt-2" />
                </CardContent>
              </Card>

              {/* Source Components */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-blue-500/25 transition-all duration-300 transform hover:scale-[1.02]">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-blue-500/30 via-blue-500/15 to-blue-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Sources</CardTitle>
                  <div className="rounded-xl bg-blue-500/10 p-2.5">
                    <ArrowRight className="h-4 w-4 text-blue-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-blue-500">
                    {stats.source_count}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {stats.source_percentage.toFixed(1)}% - Entry points
                  </p>
                  <Progress value={stats.source_percentage} className="h-1 mt-2" />
                </CardContent>
              </Card>

              {/* Sink Components */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-green-500/25 transition-all duration-300 transform hover:scale-[1.02]">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-green-500/30 via-green-500/15 to-green-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Sinks</CardTitle>
                  <div className="rounded-xl bg-green-500/10 p-2.5">
                    <ArrowLeft className="h-4 w-4 text-green-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-green-500">
                    {stats.sink_count}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {stats.sink_percentage.toFixed(1)}% - Foundations
                  </p>
                  <Progress value={stats.sink_percentage} className="h-1 mt-2" />
                </CardContent>
              </Card>

              {/* Bidirectional Components */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-purple-500/25 transition-all duration-300 transform hover:scale-[1.02]">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-purple-500 via-fuchsia-500 to-pink-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-purple-500/30 via-purple-500/15 to-purple-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Bidirectional</CardTitle>
                  <div className="rounded-xl bg-purple-500/10 p-2.5">
                    <ArrowLeftRight className="h-4 w-4 text-purple-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-purple-500">
                    {stats.bidirectional_count}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {stats.bidirectional_percentage.toFixed(1)}% - Connected
                  </p>
                  <Progress value={stats.bidirectional_percentage} className="h-1 mt-2" />
                </CardContent>
              </Card>
            </div>

            {/* Health Assessment */}
            <Card className="relative overflow-hidden border-0 shadow-lg">
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-orange-500 via-red-500 to-pink-500">
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              <div className="absolute inset-[2px] rounded-lg bg-background" />
              
              <CardHeader className="relative">
                <div className="flex items-center gap-2">
                  <div className="rounded-xl bg-orange-500/10 p-2.5">
                    <GitFork className="h-5 w-5 text-orange-500" />
                  </div>
                  <div>
                    <CardTitle>Architecture Health Assessment</CardTitle>
                    <CardDescription>
                      Overall connectivity and structural health evaluation
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              
              <CardContent className="relative">
                <Alert className={`border-2 ${interpretation.color.replace('text-', 'border-')}`}>
                  <Icon className={`h-5 w-5 ${interpretation.color}`} />
                  <AlertTitle className="text-lg">
                    {interpretation.label}: {stats.category.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}
                  </AlertTitle>
                  <AlertDescription className="text-base">
                    {stats.interpretation}
                  </AlertDescription>
                </Alert>
              </CardContent>
            </Card>

            {/* Top Sources Table */}
            {stats.top_sources.length > 0 && (
              <Card className="relative overflow-hidden border-0 shadow-lg">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-background" />
                
                <CardHeader className="relative">
                  <div className="flex items-center gap-2">
                    <div className="rounded-xl bg-blue-500/10 p-2.5">
                      <ArrowRight className="h-5 w-5 text-blue-500" />
                    </div>
                    <div>
                      <CardTitle>Top Source Components</CardTitle>
                      <CardDescription>
                        Entry point components with only outgoing dependencies
                      </CardDescription>
                    </div>
                  </div>
                </CardHeader>
                
                <CardContent className="relative">
                  <div className="rounded-md border">
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead>
                          <tr className="border-b bg-muted/50">
                            <th className="p-3 text-left text-sm font-medium">Rank</th>
                            <th className="p-3 text-left text-sm font-medium">Component</th>
                            <th className="p-3 text-left text-sm font-medium">Type</th>
                            <th className="p-3 text-right text-sm font-medium">Dependencies</th>
                          </tr>
                        </thead>
                        <tbody>
                          {stats.top_sources.map((comp, index) => (
                            <tr key={comp.id} className="border-b last:border-0 hover:bg-muted/30 transition-colors cursor-pointer group" onClick={() => router.push(`/explorer?node=${encodeURIComponent(comp.id)}`)}>
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
                                <div className="font-medium text-sm group-hover:underline transition-all">{comp.name}</div>
                                <div className="text-xs text-muted-foreground font-mono">{comp.id}</div>
                              </td>
                              <td className="p-3">
                                <Badge variant="outline" className="text-xs">
                                  {comp.type}
                                </Badge>
                              </td>
                              <td className="p-3 text-right">
                                <div className="font-semibold text-blue-600 dark:text-blue-400">
                                  {comp.out_degree}
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Top Sinks Table */}
            {stats.top_sinks.length > 0 && (
              <Card className="relative overflow-hidden border-0 shadow-lg">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-background" />
                
                <CardHeader className="relative">
                  <div className="flex items-center gap-2">
                    <div className="rounded-xl bg-green-500/10 p-2.5">
                      <ArrowLeft className="h-5 w-5 text-green-500" />
                    </div>
                    <div>
                      <CardTitle>Top Sink Components</CardTitle>
                      <CardDescription>
                        Foundation components with only incoming dependencies
                      </CardDescription>
                    </div>
                  </div>
                </CardHeader>
                
                <CardContent className="relative">
                  <div className="rounded-md border">
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead>
                          <tr className="border-b bg-muted/50">
                            <th className="p-3 text-left text-sm font-medium">Rank</th>
                            <th className="p-3 text-left text-sm font-medium">Component</th>
                            <th className="p-3 text-left text-sm font-medium">Type</th>
                            <th className="p-3 text-right text-sm font-medium">Dependents</th>
                          </tr>
                        </thead>
                        <tbody>
                          {stats.top_sinks.map((comp, index) => (
                            <tr key={comp.id} className="border-b last:border-0 hover:bg-muted/30 transition-colors cursor-pointer group" onClick={() => router.push(`/explorer?node=${encodeURIComponent(comp.id)}`)}>
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
                                <div className="font-medium text-sm group-hover:underline transition-all">{comp.name}</div>
                                <div className="text-xs text-muted-foreground font-mono">{comp.id}</div>
                              </td>
                              <td className="p-3">
                                <Badge variant="outline" className="text-xs">
                                  {comp.type}
                                </Badge>
                              </td>
                              <td className="p-3 text-right">
                                <div className="font-semibold text-green-600 dark:text-green-400">
                                  {comp.in_degree}
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Isolated Components Table */}
            {stats.isolated_components.length > 0 && (
              <Card className="relative overflow-hidden border-0 shadow-lg">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-gray-500 via-slate-500 to-zinc-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-background" />
                
                <CardHeader className="relative">
                  <div className="flex items-center gap-2">
                    <div className="rounded-xl bg-gray-500/10 p-2.5">
                      <Box className="h-5 w-5 text-gray-500" />
                    </div>
                    <div>
                      <CardTitle>Isolated Components</CardTitle>
                      <CardDescription>
                        Components with no incoming or outgoing connections
                      </CardDescription>
                    </div>
                  </div>
                </CardHeader>
                
                <CardContent className="relative">
                  <div className="rounded-md border">
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead>
                          <tr className="border-b bg-muted/50">
                            <th className="p-3 text-left text-sm font-medium">#</th>
                            <th className="p-3 text-left text-sm font-medium">Component</th>
                            <th className="p-3 text-left text-sm font-medium">Type</th>
                            <th className="p-3 text-right text-sm font-medium">Status</th>
                          </tr>
                        </thead>
                        <tbody>
                          {stats.isolated_components.slice(0, 10).map((comp, index) => (
                            <tr key={comp.id} className="border-b last:border-0 hover:bg-muted/30 transition-colors cursor-pointer group" onClick={() => router.push(`/explorer?node=${encodeURIComponent(comp.id)}`)}>
                              <td className="p-3">
                                <div className="text-sm text-muted-foreground">{index + 1}</div>
                              </td>
                              <td className="p-3">
                                <div className="font-medium text-sm group-hover:underline transition-all">{comp.name}</div>
                                <div className="text-xs text-muted-foreground font-mono">{comp.id}</div>
                              </td>
                              <td className="p-3">
                                <Badge variant="outline" className="text-xs">
                                  {comp.type}
                                </Badge>
                              </td>
                              <td className="p-3 text-right">
                                <div className="text-sm text-muted-foreground">
                                  No Connections
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                  {stats.isolated_components.length > 10 && (
                    <p className="text-xs text-muted-foreground mt-3">
                      Showing 10 of {stats.isolated_components.length} isolated components
                    </p>
                  )}
                </CardContent>
              </Card>
            )}

            {/* Key Insights */}
            <Card className="relative overflow-hidden border-0 shadow-lg">
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-orange-500 via-red-500 to-pink-500">
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              <div className="absolute inset-[2px] rounded-lg bg-background" />
              
              <CardHeader className="relative">
                <div className="flex items-center gap-2">
                  <div className="rounded-xl bg-orange-500/10 p-2.5">
                    <Info className="h-5 w-5 text-orange-500" />
                  </div>
                  <CardTitle>Key Insights</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="relative space-y-4">
                <div className="space-y-3">
                  <div className="flex items-start gap-3">
                    <div className="rounded-full bg-gray-500/10 p-1 mt-0.5">
                      <div className="w-2 h-2 rounded-full bg-gray-500" />
                    </div>
                    <div className="flex-1">
                      <p className="font-medium">Isolated Components</p>
                      <p className="text-sm text-muted-foreground">
                        Components with no connections may indicate unused code, incomplete integration, 
                        or planned but not yet implemented features. Consider reviewing for removal or connection.
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start gap-3">
                    <div className="rounded-full bg-blue-500/10 p-1 mt-0.5">
                      <div className="w-2 h-2 rounded-full bg-blue-500" />
                    </div>
                    <div className="flex-1">
                      <p className="font-medium">Source Components (Entry Points)</p>
                      <p className="text-sm text-muted-foreground">
                        Components with only outgoing dependencies are system entry points. They initiate operations 
                        and coordinate workflows. High count suggests multiple system interfaces.
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start gap-3">
                    <div className="rounded-full bg-green-500/10 p-1 mt-0.5">
                      <div className="w-2 h-2 rounded-full bg-green-500" />
                    </div>
                    <div className="flex-1">
                      <p className="font-medium">Sink Components (Foundations)</p>
                      <p className="text-sm text-muted-foreground">
                        Components with only incoming dependencies are foundational utilities. Changes to these 
                        components have wide impact. They should be stable and well-tested.
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start gap-3">
                    <div className="rounded-full bg-purple-500/10 p-1 mt-0.5">
                      <div className="w-2 h-2 rounded-full bg-purple-500" />
                    </div>
                    <div className="flex-1">
                      <p className="font-medium">Bidirectional Components</p>
                      <p className="text-sm text-muted-foreground">
                        Components with both incoming and outgoing dependencies are intermediate layers. 
                        High percentage indicates good layering and modularity in the architecture.
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </AppLayout>
  )
}
