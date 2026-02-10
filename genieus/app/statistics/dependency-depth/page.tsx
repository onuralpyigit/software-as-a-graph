"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Layers, ArrowLeft, TrendingUp, AlertTriangle, CheckCircle, Info, AlertCircle, BarChart3 } from "lucide-react"
import { useConnection } from "@/lib/stores/connection-store"
import { apiClient } from "@/lib/api/client"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid
} from 'recharts'
import { ChartContainer, ChartTooltip, ChartTooltipContent, ChartLegend, ChartLegendContent } from "@/components/ui/chart"

interface ComponentInfo {
  id: string
  name: string
  type: string
  depth: number
  dependencies: number
  dependents: number
}

interface RootLeafNode {
  id: string
  name: string
  type: string
  dependencies?: number
  dependents?: number
}

interface DependencyDepthStats {
  avg_depth: number
  max_depth: number
  min_depth: number
  median_depth: number
  std_depth: number
  interpretation: string
  category: string
  shallow_count: number
  low_depth_count: number
  medium_depth_count: number
  high_depth_count: number
  total_nodes: number
  deepest_components: ComponentInfo[]
  root_nodes: RootLeafNode[]
  leaf_nodes: RootLeafNode[]
  depth_distribution: Record<string, number>
}

export default function DependencyDepthPage() {
  const router = useRouter()
  const { status, stats: graphStats, initialLoadComplete } = useConnection()
  const [stats, setStats] = useState<DependencyDepthStats | null>(null)
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
      const result = await apiClient.getDependencyDepthStats()
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
    
    switch (stats.category) {
      case 'isolated':
        return { 
          color: 'text-gray-500', 
          icon: Info, 
          label: 'Isolated',
          description: 'No dependencies between components - completely independent system'
        }
      case 'shallow':
        return { 
          color: 'text-green-500', 
          icon: CheckCircle, 
          label: 'Shallow Dependencies',
          description: 'Simple, flat architecture with minimal cascading - easy to understand and maintain'
        }
      case 'moderate':
        return { 
          color: 'text-blue-500', 
          icon: Layers, 
          label: 'Moderate Depth',
          description: 'Balanced architecture with reasonable layering - good separation of concerns'
        }
      case 'deep':
        return { 
          color: 'text-orange-500', 
          icon: AlertCircle, 
          label: 'Deep Dependencies',
          description: 'Complex architecture with significant layering - may impact change propagation'
        }
      case 'very_deep':
        return { 
          color: 'text-red-500', 
          icon: AlertTriangle, 
          label: 'Very Deep Dependencies',
          description: 'Highly complex with extensive cascading chains - high risk for changes and failures'
        }
      default:
        return { color: 'text-gray-500', icon: Info, label: 'Unknown' }
    }
  }

  const interpretation = getInterpretation()
  const Icon = interpretation.icon

  // Prepare distribution data for chart
  const distributionData = stats ? Object.entries(stats.depth_distribution)
    .sort((a, b) => parseInt(a[0]) - parseInt(b[0]))
    .map(([depth, count]) => ({
      depth: `Depth ${depth}`,
      count,
      depthNum: parseInt(depth)
    })) : []

  // Prepare depth category data for progress bars
  const categoryData = stats ? [
    { name: 'No Dependencies', value: stats.shallow_count, color: 'hsl(var(--muted))' },
    { name: 'Shallow (1-2)', value: stats.low_depth_count, color: 'hsl(142 76% 36%)' },
    { name: 'Moderate (3-5)', value: stats.medium_depth_count, color: 'hsl(217 91% 60%)' },
    { name: 'Deep (6+)', value: stats.high_depth_count, color: 'hsl(25 95% 53%)' },
  ].filter(item => item.value > 0) : []

  // Loading State
  if (!initialLoadComplete || status === 'connecting') {
    return (
      <AppLayout title="Dependency Depth" description="Analyze dependency chain depth and complexity">
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" text="Connecting to database..." />
        </div>
      </AppLayout>
    )
  }

  // Disconnected State
  if (!isConnected) {
    return (
      <AppLayout title="Dependency Depth" description="Analyze dependency chain depth and complexity">
        <NoConnectionInfo />
      </AppLayout>
    )
  }

  return (
    <AppLayout title="Dependency Depth" description="Analyze dependency chain depth and complexity">
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
        <Card className="relative overflow-hidden border-0 shadow-xl hover:shadow-2xl hover:shadow-amber-500/25 transition-all duration-300">
          <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-orange-500 via-amber-500 to-yellow-500">
            <div className="w-full h-full rounded-lg bg-gradient-to-r from-orange-600 via-amber-600 to-yellow-600" />
          </div>
          
          <CardContent className="p-8 relative text-white">
            <div className="flex items-center gap-4 mb-4">
              <div className="rounded-xl bg-white/20 p-3">
                <Layers className="h-8 w-8" />
              </div>
              <div className="flex-1">
                <h3 className="text-3xl font-bold">Dependency Depth Analysis</h3>
                <p className="text-white/90 mt-2">
                  Understanding dependency chain depth and architectural complexity
                </p>
              </div>
            </div>
            
            <div className="bg-white/10 rounded-lg p-4 mt-4">
              <div className="flex items-start gap-3">
                <Info className="h-5 w-5 mt-0.5 flex-shrink-0" />
                <div className="text-sm">
                  <p className="font-semibold mb-1">What is Dependency Depth?</p>
                  <p className="text-white/90">
                    Dependency depth measures how many layers deep the transitive dependency chains are. 
                    A depth of 3 means a component depends on another, which depends on another, forming a 3-level chain. 
                    Deeper chains increase complexity, failure propagation risk, and testing scope.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Loading/Error States */}
        {loading && (
          <div className="flex items-center justify-center py-12">
            <LoadingSpinner size="lg" text="Computing dependency depth..." />
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
            {/* Main Depth Metric */}
            <Card className="relative overflow-hidden border-0 shadow-lg">
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-orange-500 via-amber-500 to-yellow-500">
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              <div className="absolute inset-[2px] rounded-lg bg-background" />
              
              <CardHeader className="relative">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-2xl mb-2">Maximum Dependency Depth</CardTitle>
                    <CardDescription>Deepest dependency chain in the system</CardDescription>
                  </div>
                  <div className={`rounded-xl bg-orange-500/10 p-3 ${interpretation.color}`}>
                    <Icon className="h-8 w-8" />
                  </div>
                </div>
              </CardHeader>
              <CardContent className="relative space-y-4">
                <div className="flex items-baseline gap-3">
                  <span className="text-6xl font-bold text-orange-500">{stats.max_depth}</span>
                  <span className="text-2xl text-muted-foreground">levels</span>
                </div>
                
                <Alert className={`border-2 ${interpretation.color.replace('text-', 'border-')}`}>
                  <Icon className={`h-4 w-4 ${interpretation.color}`} />
                  <AlertTitle className="font-semibold">{interpretation.label}</AlertTitle>
                  <AlertDescription>{interpretation.description}</AlertDescription>
                </Alert>
                
                <div className="pt-2">
                  <p className="text-sm text-muted-foreground mb-2">Interpretation</p>
                  <p className="text-sm">{stats.interpretation}</p>
                </div>
              </CardContent>
            </Card>

            {/* Key Statistics Grid */}
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              {/* Average Depth */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-blue-500/25 transition-all duration-300">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 to-cyan-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-blue-500/30 via-blue-500/15 to-blue-500/5" />
                
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
                  <CardTitle className="text-sm font-medium">Average Depth</CardTitle>
                  <div className="rounded-xl bg-blue-500/10 p-2.5">
                    <TrendingUp className="h-4 w-4 text-blue-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-blue-500">{stats.avg_depth.toFixed(2)}</div>
                  <p className="text-xs text-muted-foreground mt-1">Mean chain depth</p>
                </CardContent>
              </Card>

              {/* Median Depth */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-purple-500/25 transition-all duration-300">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-purple-500 to-pink-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-purple-500/30 via-purple-500/15 to-purple-500/5" />
                
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
                  <CardTitle className="text-sm font-medium">Median Depth</CardTitle>
                  <div className="rounded-xl bg-purple-500/10 p-2.5">
                    <Layers className="h-4 w-4 text-purple-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-purple-500">{stats.median_depth.toFixed(2)}</div>
                  <p className="text-xs text-muted-foreground mt-1">Middle value</p>
                </CardContent>
              </Card>

              {/* Root Components */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-green-500/25 transition-all duration-300">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-green-500 to-emerald-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-green-500/30 via-green-500/15 to-green-500/5" />
                
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
                  <CardTitle className="text-sm font-medium">Root Components</CardTitle>
                  <div className="rounded-xl bg-green-500/10 p-2.5">
                    <CheckCircle className="h-4 w-4 text-green-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-green-500">{stats.root_nodes.length}</div>
                  <p className="text-xs text-muted-foreground mt-1">No dependencies</p>
                </CardContent>
              </Card>

              {/* Leaf Components */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-rose-500/25 transition-all duration-300">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-rose-500 to-pink-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-rose-500/30 via-rose-500/15 to-rose-500/5" />
                
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
                  <CardTitle className="text-sm font-medium">Leaf Components</CardTitle>
                  <div className="rounded-xl bg-rose-500/10 p-2.5">
                    <AlertCircle className="h-4 w-4 text-rose-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-rose-500">{stats.leaf_nodes.length}</div>
                  <p className="text-xs text-muted-foreground mt-1">Foundation layer</p>
                </CardContent>
              </Card>
            </div>

            {/* Distribution Charts */}
            <div className="grid gap-6 md:grid-cols-2">
              {/* Depth Distribution Bar Chart */}
              <Card className="relative overflow-hidden border-0 shadow-lg">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-orange-500 via-amber-500 to-yellow-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-background" />
                
                <CardHeader className="relative">
                  <div className="flex items-center gap-2">
                    <div className="rounded-xl bg-orange-500/10 p-2.5">
                      <BarChart3 className="h-5 w-5 text-orange-500" />
                    </div>
                    <div>
                      <CardTitle>Depth Distribution</CardTitle>
                      <CardDescription>Number of components at each dependency depth level</CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <ChartContainer
                    config={{
                      count: {
                        label: "Components",
                        theme: {
                          light: "hsl(25 95% 53%)",
                          dark: "hsl(25 95% 60%)"
                        }
                      },
                    }}
                    className="h-[300px] w-full"
                  >
                    <BarChart data={distributionData}>
                      <CartesianGrid vertical={false} />
                      <XAxis
                        dataKey="depth"
                        tickLine={false}
                        tickMargin={10}
                        axisLine={false}
                      />
                      <YAxis
                        tickLine={false}
                        axisLine={false}
                        tickMargin={8}
                      />
                      <ChartTooltip content={<ChartTooltipContent />} />
                      <Bar
                        dataKey="count"
                        fill="var(--color-count)"
                        radius={[8, 8, 0, 0]}
                      />
                    </BarChart>
                  </ChartContainer>
                </CardContent>
              </Card>

              {/* Category Distribution */}
              <Card className="relative overflow-hidden border-0 shadow-lg">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-amber-500 via-yellow-500 to-orange-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-background" />
                
                <CardHeader className="relative">
                  <div className="flex items-center gap-2">
                    <div className="rounded-xl bg-amber-500/10 p-2.5">
                      <Layers className="h-5 w-5 text-amber-500" />
                    </div>
                    <div>
                      <CardTitle>Depth Categories</CardTitle>
                      <CardDescription>Distribution of components by depth category</CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="space-y-4">
                    {categoryData.map((category, idx) => (
                      <div key={idx} className="space-y-2">
                        <div className="flex items-center justify-between text-sm">
                          <span className="font-medium">{category.name}</span>
                          <span className="text-muted-foreground">{category.value} components</span>
                        </div>
                        <Progress 
                          value={(category.value / stats.total_nodes) * 100} 
                          className="h-2"
                          style={{ 
                            ['--progress-background' as string]: category.color 
                          }}
                        />
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Deepest Components Table */}
            {stats.deepest_components.length > 0 && (
              <Card className="relative overflow-hidden border-0 shadow-lg">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-red-500 via-orange-500 to-amber-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-background" />
                
                <CardHeader className="relative">
                  <div className="flex items-center gap-2">
                    <div className="rounded-xl bg-red-500/10 p-2.5">
                      <AlertTriangle className="h-5 w-5 text-red-500" />
                    </div>
                    <div>
                      <CardTitle>Deepest Dependency Chains</CardTitle>
                      <CardDescription>Components with the longest transitive dependency paths</CardDescription>
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
                            <th className="p-3 text-right text-sm font-medium">Depth</th>
                            <th className="p-3 text-right text-sm font-medium">Direct Deps</th>
                            <th className="p-3 text-right text-sm font-medium">Dependents</th>
                          </tr>
                        </thead>
                        <tbody>
                          {stats.deepest_components.map((component, index) => (
                            <tr key={component.id} className="border-b last:border-0 hover:bg-muted/30 transition-colors cursor-pointer group" onClick={() => router.push(`/explorer?node=${encodeURIComponent(component.id)}`)}>
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
                                <div className="font-medium text-sm group-hover:underline transition-all">{component.name}</div>
                                <div className="text-xs text-muted-foreground font-mono">{component.id}</div>
                              </td>
                              <td className="p-3">
                                <Badge variant="outline" className="text-xs">
                                  {component.type}
                                </Badge>
                              </td>
                              <td className="p-3 text-right">
                                <div className="font-semibold text-orange-600 dark:text-orange-400">
                                  {component.depth}
                                </div>
                              </td>
                              <td className="p-3 text-right">
                                <div className="text-sm text-muted-foreground">
                                  {component.dependencies}
                                </div>
                              </td>
                              <td className="p-3 text-right">
                                <div className="text-sm text-muted-foreground">
                                  {component.dependents}
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground mt-3">
                    These components have the longest dependency chains. High depth may indicate complexity and testing challenges.
                  </p>
                </CardContent>
              </Card>
            )}

            {/* Root and Leaf Nodes */}
            <div className="grid gap-6 md:grid-cols-2">
              {/* Root Nodes */}
              <Card className="relative overflow-hidden border-0 shadow-lg">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-green-500 to-emerald-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-background" />
                
                <CardHeader className="relative">
                  <div className="flex items-center gap-2">
                    <div className="rounded-xl bg-green-500/10 p-2.5">
                      <CheckCircle className="h-5 w-5 text-green-500" />
                    </div>
                    <div>
                      <CardTitle>Top Root Components</CardTitle>
                      <CardDescription>
                        {stats.root_nodes.length > 0 && stats.root_nodes[0].dependents !== undefined && stats.root_nodes[0].dependents > 0
                          ? "Components with fewest dependents (closest to entry points)"
                          : "Components with no incoming dependencies (entry points)"}
                      </CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  {stats.root_nodes.length > 0 ? (
                    <div className="space-y-2">
                      {stats.root_nodes.map((node, idx) => (
                        <div key={idx} className="flex items-center justify-between p-3 rounded-lg border hover:bg-muted/30 transition-colors cursor-pointer group" onClick={() => router.push(`/explorer?node=${encodeURIComponent(node.id)}`)}>
                          <div className="flex-1">
                            <div className="font-medium text-sm group-hover:underline transition-all">{node.name}</div>
                            <div className="text-xs text-muted-foreground">{node.type}</div>
                          </div>
                          <div className="flex items-center gap-2">
                            <Badge className="bg-green-500/10 text-green-500 hover:bg-green-500/20">
                              {node.dependencies} deps
                            </Badge>
                            {node.dependents !== undefined && node.dependents > 0 && (
                              <Badge variant="outline" className="text-xs">
                                {node.dependents} users
                              </Badge>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-muted-foreground">
                      <Info className="h-8 w-8 mx-auto mb-2 opacity-50" />
                      <p className="text-sm">No root components found</p>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Leaf Nodes */}
              <Card className="relative overflow-hidden border-0 shadow-lg">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-rose-500 to-pink-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-background" />
                
                <CardHeader className="relative">
                  <div className="flex items-center gap-2">
                    <div className="rounded-xl bg-rose-500/10 p-2.5">
                      <AlertCircle className="h-5 w-5 text-rose-500" />
                    </div>
                    <div>
                      <CardTitle>Top Leaf Components</CardTitle>
                      <CardDescription>Foundation components with no outgoing dependencies</CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  {stats.leaf_nodes.length > 0 ? (
                    <div className="space-y-2">
                      {stats.leaf_nodes.map((node, idx) => (
                        <div key={idx} className="flex items-center justify-between p-3 rounded-lg border hover:bg-muted/30 transition-colors cursor-pointer group" onClick={() => router.push(`/explorer?node=${encodeURIComponent(node.id)}`)}>
                          <div className="flex-1">
                            <div className="font-medium text-sm group-hover:underline transition-all">{node.name}</div>
                            <div className="text-xs text-muted-foreground">{node.type}</div>
                          </div>
                          <Badge className="bg-rose-500/10 text-rose-500 hover:bg-rose-500/20">
                            {node.dependents} users
                          </Badge>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-muted-foreground">
                      <Info className="h-8 w-8 mx-auto mb-2 opacity-50" />
                      <p className="text-sm">No leaf components found</p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Insights and Recommendations */}
            <Card className="relative overflow-hidden border-0 shadow-lg">
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500">
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              <div className="absolute inset-[2px] rounded-lg bg-background" />
              
              <CardHeader className="relative">
                <div className="flex items-center gap-2">
                  <div className="rounded-xl bg-blue-500/10 p-2.5">
                    <Info className="h-5 w-5 text-blue-500" />
                  </div>
                  <div>
                    <CardTitle>Insights & Recommendations</CardTitle>
                    <CardDescription>Understanding the impact of dependency depth</CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="relative space-y-4">
                <div className="grid gap-4 md:grid-cols-2">
                  <div className="space-y-2">
                    <h4 className="font-semibold flex items-center gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      Why This Matters
                    </h4>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• Deep chains increase failure propagation risk</li>
                      <li>• Changes at low levels impact many components</li>
                      <li>• Testing requires understanding full dependency tree</li>
                      <li>• Refactoring becomes more complex and risky</li>
                    </ul>
                  </div>
                  <div className="space-y-2">
                    <h4 className="font-semibold flex items-center gap-2">
                      <TrendingUp className="h-4 w-4 text-blue-500" />
                      Best Practices
                    </h4>
                    <ul className="text-sm space-y-1 text-muted-foreground">
                      <li>• Aim for shallow, flat architectures (depth ≤ 3)</li>
                      <li>• Use dependency injection to reduce coupling</li>
                      <li>• Apply layered architecture patterns</li>
                      <li>• Consider breaking deep chains with abstractions</li>
                    </ul>
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
