"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Share2, ArrowLeft, GitBranch, TrendingUp, AlertTriangle, CheckCircle, Info, Network, CheckCircle2 } from "lucide-react"
import { useConnection } from "@/lib/stores/connection-store"
import { apiClient } from "@/lib/api/client"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface ClusteredComponent {
  id: string
  name: string
  type: string
  coefficient: number
  degree: number
  triangles: number
}

interface ClusteringCoefficientStats {
  avg_clustering_coefficient: number
  max_coefficient: number
  min_coefficient: number
  median_coefficient: number
  std_coefficient: number
  interpretation: string
  category: string
  zero_clustering_count: number
  low_clustering_count: number
  medium_clustering_count: number
  high_clustering_count: number
  total_nodes: number
  highly_clustered_components: ClusteredComponent[]
}

export default function ClusteringCoefficientPage() {
  const router = useRouter()
  const { status, stats: graphStats, initialLoadComplete } = useConnection()
  const [stats, setStats] = useState<ClusteringCoefficientStats | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedComponentType, setSelectedComponentType] = useState<string>('all')

  const isConnected = status === 'connected'
  
  // Get available component types from connection stats
  const componentTypes = graphStats?.node_counts 
    ? Object.keys(graphStats.node_counts).sort()
    : []

  useEffect(() => {
    if (isConnected) {
      loadStats()
    }
  }, [isConnected, selectedComponentType])

  const loadStats = async () => {
    try {
      setLoading(true)
      setError(null)
      const nodeType = selectedComponentType === 'all' ? undefined : selectedComponentType
      const result = await apiClient.getClusteringCoefficientStats(nodeType)
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
    
    if (stats.avg_clustering_coefficient < 0.1) {
      return { 
        color: 'text-blue-500', 
        icon: Info, 
        label: 'Low Clustering',
        description: 'Components operate independently with few tightly connected groups - typical of loosely coupled systems'
      }
    } else if (stats.avg_clustering_coefficient < 0.3) {
      return { 
        color: 'text-green-500', 
        icon: CheckCircle, 
        label: 'Moderate Clustering',
        description: 'Balanced structure with some component grouping - good modularity with reasonable local connectivity'
      }
    } else if (stats.avg_clustering_coefficient < 0.6) {
      return { 
        color: 'text-cyan-500', 
        icon: GitBranch, 
        label: 'High Clustering',
        description: 'Strong component grouping with modular structure - components form well-defined clusters'
      }
    } else {
      return { 
        color: 'text-purple-500', 
        icon: Share2, 
        label: 'Very High Clustering',
        description: 'Tightly connected groups with highly modular structure - strong local connectivity patterns'
      }
    }
  }

  const interpretation = getInterpretation()
  const Icon = interpretation.icon

  // Loading State
  if (!initialLoadComplete || status === 'connecting') {
    return (
      <AppLayout title="Clustering Coefficient" description="Analyze component grouping and modularity">
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" text="Connecting to database..." />
        </div>
      </AppLayout>
    )
  }

  // Disconnected State
  if (!isConnected) {
    return (
      <AppLayout title="Clustering Coefficient" description="Analyze component grouping and modularity">
        <NoConnectionInfo />
      </AppLayout>
    )
  }

  return (
    <AppLayout title="Clustering Coefficient" description="Analyze component grouping and modularity">
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
        <Card className="relative overflow-hidden border-0 shadow-xl hover:shadow-2xl hover:shadow-purple-500/25 transition-all duration-300">
          <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-purple-500 via-pink-500 to-rose-500">
            <div className="w-full h-full rounded-lg bg-gradient-to-r from-purple-600 via-pink-600 to-rose-600" />
          </div>
          
          <CardContent className="p-8 relative text-white">
            <div className="flex items-center gap-4 mb-4">
              <div className="rounded-xl bg-white/20 p-3">
                <Share2 className="h-8 w-8" />
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between gap-3 mb-2">
                  <h3 className="text-3xl font-bold">Clustering Coefficient Analysis</h3>
                  {/* Component Type Filter Dropdown */}
                  {componentTypes.length > 0 && (
                    <div className="flex items-center gap-2">
                      <Select value={selectedComponentType} onValueChange={setSelectedComponentType}>
                        <SelectTrigger className="w-[220px] bg-white/10 border-white/20 text-white hover:bg-white/20">
                          <SelectValue placeholder="Select type" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="all">
                            <div className="flex items-center gap-2">
                              <span className="font-medium">All Components</span>
                            </div>
                          </SelectItem>
                          {componentTypes.map(type => (
                            <SelectItem key={type} value={type}>
                              <div className="flex items-center justify-between gap-2">
                                <span>{type.split('_').map(word => 
                                  word.charAt(0).toUpperCase() + word.slice(1).toLowerCase()
                                ).join(' ')}</span>
                                {graphStats?.node_counts && (
                                  <Badge variant="outline" className="ml-2 text-xs">
                                    {graphStats.node_counts[type]}
                                  </Badge>
                                )}
                              </div>
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      {selectedComponentType !== 'all' && (
                        <Badge className="bg-white/20 text-white border-white/30">
                          <span className="flex items-center gap-1">
                            <CheckCircle2 className="h-3 w-3" />
                            Filtered
                          </span>
                        </Badge>
                      )}
                    </div>
                  )}
                </div>
                <p className="text-white/90 mt-2">
                  {selectedComponentType === 'all' 
                    ? 'Understanding component grouping and local connectivity patterns'
                    : `Analyzing ${selectedComponentType.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')} components only`
                  }
                </p>
              </div>
            </div>
            
            <div className="bg-white/10 rounded-lg p-4 mt-4">
              <div className="flex items-start gap-3">
                <Info className="h-5 w-5 mt-0.5 flex-shrink-0" />
                <div className="text-sm">
                  <p className="font-semibold mb-1">What is Clustering Coefficient?</p>
                  <p className="text-white/90">
                    The clustering coefficient measures the degree to which nodes tend to cluster together. 
                    For each node, it's the ratio of actual connections between its neighbors to the maximum possible connections. 
                    Higher values indicate stronger local grouping and modular structure.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Loading/Error States */}
        {loading && (
          <div className="flex items-center justify-center py-12">
            <LoadingSpinner size="lg" text="Computing clustering coefficient..." />
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
            {/* Main Clustering Metric */}
            <Card className="relative overflow-hidden border-0 shadow-lg">
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-purple-500 via-pink-500 to-rose-500">
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              <div className="absolute inset-[2px] rounded-lg bg-background" />
              
              <CardHeader className="relative">
                <div className="flex items-center gap-2">
                  <div className="rounded-xl bg-purple-500/10 p-2.5">
                    <Share2 className="h-5 w-5 text-purple-500" />
                  </div>
                  <div>
                    <CardTitle>Average Clustering Coefficient</CardTitle>
                    <CardDescription>
                      Measure of how nodes cluster together forming tight-knit groups
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              
              <CardContent className="relative space-y-6">
                {/* Large Coefficient Display */}
                <div className="text-center py-8">
                  <div className="text-7xl font-bold bg-gradient-to-r from-purple-500 via-pink-500 to-rose-500 bg-clip-text text-transparent">
                    {(stats.avg_clustering_coefficient * 100).toFixed(2)}%
                  </div>
                  <p className="text-muted-foreground mt-2">Clustering Coefficient</p>
                </div>

                {/* Progress Bar */}
                <div className="space-y-2">
                  <Progress value={stats.avg_clustering_coefficient * 100} className="h-3" />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>0% (No Clustering)</span>
                    <span>50%</span>
                    <span>100% (Perfect Clustering)</span>
                  </div>
                </div>

                {/* Interpretation */}
                <Alert className={`border-2 ${interpretation.color.replace('text-', 'border-')}`}>
                  <Icon className={`h-5 w-5 ${interpretation.color}`} />
                  <AlertTitle className="text-lg">
                    {interpretation.label}
                  </AlertTitle>
                  <AlertDescription className="text-base">
                    {interpretation.description}
                  </AlertDescription>
                </Alert>
              </CardContent>
            </Card>

            {/* Distribution Metrics */}
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              {/* Max Coefficient */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-purple-500/25 transition-all duration-300 transform hover:scale-[1.02]">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-purple-500 via-fuchsia-500 to-pink-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-purple-500/30 via-purple-500/15 to-purple-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Maximum</CardTitle>
                  <div className="rounded-xl bg-purple-500/10 p-2.5">
                    <TrendingUp className="h-4 w-4 text-purple-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-purple-500">
                    {(stats.max_coefficient * 100).toFixed(1)}%
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">Highest clustering</p>
                </CardContent>
              </Card>

              {/* Median Coefficient */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-pink-500/25 transition-all duration-300 transform hover:scale-[1.02]">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-pink-500 via-rose-500 to-red-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-pink-500/30 via-pink-500/15 to-pink-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Median</CardTitle>
                  <div className="rounded-xl bg-pink-500/10 p-2.5">
                    <GitBranch className="h-4 w-4 text-pink-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-pink-500">
                    {(stats.median_coefficient * 100).toFixed(1)}%
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">Middle value</p>
                </CardContent>
              </Card>

              {/* High Clustering Nodes */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-cyan-500/25 transition-all duration-300 transform hover:scale-[1.02]">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-cyan-500 via-blue-500 to-indigo-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-cyan-500/30 via-cyan-500/15 to-cyan-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">High Clustering</CardTitle>
                  <div className="rounded-xl bg-cyan-500/10 p-2.5">
                    <Share2 className="h-4 w-4 text-cyan-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-cyan-500">
                    {stats.high_clustering_count + stats.medium_clustering_count}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">Well-clustered nodes</p>
                </CardContent>
              </Card>

              {/* Zero Clustering */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-gray-500/25 transition-all duration-300 transform hover:scale-[1.02]">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-gray-500 via-slate-500 to-zinc-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-gray-500/30 via-gray-500/15 to-gray-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">No Clustering</CardTitle>
                  <div className="rounded-xl bg-gray-500/10 p-2.5">
                    <Network className="h-4 w-4 text-gray-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-gray-500">
                    {stats.zero_clustering_count}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">Isolated connections</p>
                </CardContent>
              </Card>
            </div>

            {/* Clustering Distribution */}
            <Card className="relative overflow-hidden border-0 shadow-lg">
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-purple-500 via-pink-500 to-rose-500">
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              <div className="absolute inset-[2px] rounded-lg bg-background" />
              
              <CardHeader className="relative">
                <div className="flex items-center gap-2">
                  <div className="rounded-xl bg-purple-500/10 p-2.5">
                    <GitBranch className="h-5 w-5 text-purple-500" />
                  </div>
                  <div>
                    <CardTitle>Clustering Distribution</CardTitle>
                    <CardDescription>
                      Breakdown of components by clustering level
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              
              <CardContent className="relative">
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">None (0%)</span>
                      <Badge variant="outline" className="bg-gray-500/10">
                        {stats.zero_clustering_count}
                      </Badge>
                    </div>
                    <Progress 
                      value={(stats.zero_clustering_count / stats.total_nodes) * 100} 
                      className="h-2 bg-gray-200 dark:bg-gray-800"
                    />
                    <p className="text-xs text-muted-foreground">
                      {((stats.zero_clustering_count / stats.total_nodes) * 100).toFixed(1)}% of nodes
                    </p>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Low (0-30%)</span>
                      <Badge variant="outline" className="bg-blue-500/10">
                        {stats.low_clustering_count}
                      </Badge>
                    </div>
                    <Progress 
                      value={(stats.low_clustering_count / stats.total_nodes) * 100} 
                      className="h-2"
                    />
                    <p className="text-xs text-muted-foreground">
                      {((stats.low_clustering_count / stats.total_nodes) * 100).toFixed(1)}% of nodes
                    </p>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">Medium (30-70%)</span>
                      <Badge variant="outline" className="bg-cyan-500/10">
                        {stats.medium_clustering_count}
                      </Badge>
                    </div>
                    <Progress 
                      value={(stats.medium_clustering_count / stats.total_nodes) * 100} 
                      className="h-2"
                    />
                    <p className="text-xs text-muted-foreground">
                      {((stats.medium_clustering_count / stats.total_nodes) * 100).toFixed(1)}% of nodes
                    </p>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">High (70%+)</span>
                      <Badge variant="outline" className="bg-purple-500/10">
                        {stats.high_clustering_count}
                      </Badge>
                    </div>
                    <Progress 
                      value={(stats.high_clustering_count / stats.total_nodes) * 100} 
                      className="h-2"
                    />
                    <p className="text-xs text-muted-foreground">
                      {((stats.high_clustering_count / stats.total_nodes) * 100).toFixed(1)}% of nodes
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Key Insights */}
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
                  <CardTitle>Key Insights</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="relative space-y-4">
                <div className="space-y-3">
                  <div className="flex items-start gap-3">
                    <div className="rounded-full bg-purple-500/10 p-1 mt-0.5">
                      <div className="w-2 h-2 rounded-full bg-purple-500" />
                    </div>
                    <div className="flex-1">
                      <p className="font-medium">Modularity</p>
                      <p className="text-sm text-muted-foreground">
                        {stats.avg_clustering_coefficient < 0.2 
                          ? "Low clustering suggests weak modularity - components don't form tight groups."
                          : stats.avg_clustering_coefficient < 0.4
                          ? "Moderate clustering indicates balanced modular structure with some component grouping."
                          : "High clustering reveals strong modularity - components form well-defined, tightly connected groups."}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start gap-3">
                    <div className="rounded-full bg-pink-500/10 p-1 mt-0.5">
                      <div className="w-2 h-2 rounded-full bg-pink-500" />
                    </div>
                    <div className="flex-1">
                      <p className="font-medium">Local Connectivity</p>
                      <p className="text-sm text-muted-foreground">
                        {stats.median_coefficient > 0.3
                          ? "Strong local connectivity - neighbors of components tend to be interconnected."
                          : "Weak local connectivity - components connect to diverse, non-interconnected neighbors."}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start gap-3">
                    <div className="rounded-full bg-cyan-500/10 p-1 mt-0.5">
                      <div className="w-2 h-2 rounded-full bg-cyan-500" />
                    </div>
                    <div className="flex-1">
                      <p className="font-medium">Component Groups</p>
                      <p className="text-sm text-muted-foreground">
                        {stats.high_clustering_count > 0
                          ? `${stats.high_clustering_count} components form tightly knit groups, suggesting functional or logical groupings.`
                          : "No highly clustered components - system has a more distributed, less grouped structure."}
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Highly Clustered Components */}
            {stats.highly_clustered_components && stats.highly_clustered_components.length > 0 && (
              <Card className="relative overflow-hidden border-0 shadow-lg">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-purple-500 via-pink-500 to-rose-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-background" />
                
                <CardHeader className="relative">
                  <div className="flex items-center gap-2">
                    <div className="rounded-xl bg-purple-500/10 p-2.5">
                      <Share2 className="h-5 w-5 text-purple-500" />
                    </div>
                    <div>
                      <CardTitle>Highly Clustered Components</CardTitle>
                      <CardDescription>
                        Components with strongest local grouping - their neighbors are highly interconnected
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
                            <th className="p-3 text-right text-sm font-medium">Coefficient</th>
                            <th className="p-3 text-right text-sm font-medium">Neighbors</th>
                            <th className="p-3 text-right text-sm font-medium">Triangles</th>
                          </tr>
                        </thead>
                        <tbody>
                          {stats.highly_clustered_components.map((component, index) => (
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
                                <div className="font-semibold text-purple-600 dark:text-purple-400">
                                  {(component.coefficient * 100).toFixed(1)}%
                                </div>
                              </td>
                              <td className="p-3 text-right">
                                <div className="text-sm text-muted-foreground">
                                  {component.degree}
                                </div>
                              </td>
                              <td className="p-3 text-right">
                                <div className="text-sm text-muted-foreground">
                                  {component.triangles}
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground mt-3">
                    Triangles represent completed triads where the component and two of its neighbors are all interconnected.
                  </p>
                </CardContent>
              </Card>
            )}

            {/* Recommendations */}
            <Card className="border-2 border-amber-500/20 bg-amber-500/5">
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-amber-600 dark:text-amber-400">
                  <AlertTriangle className="h-5 w-5" />
                  Recommendations
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm">
                  {stats.avg_clustering_coefficient < 0.2 ? (
                    <>
                      <li className="flex items-start gap-2">
                        <Info className="h-4 w-4 text-blue-500 mt-0.5 flex-shrink-0" />
                        <span>Low clustering may indicate opportunity for better component grouping and service boundaries</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <Info className="h-4 w-4 text-blue-500 mt-0.5 flex-shrink-0" />
                        <span>Consider organizing related components into modules or microservices for better cohesion</span>
                      </li>
                    </>
                  ) : stats.avg_clustering_coefficient < 0.4 ? (
                    <>
                      <li className="flex items-start gap-2">
                        <CheckCircle className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                        <span>Good balance - maintain modular structure while keeping flexibility</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <Info className="h-4 w-4 text-blue-500 mt-0.5 flex-shrink-0" />
                        <span>Monitor clustering in new features to preserve architectural patterns</span>
                      </li>
                    </>
                  ) : (
                    <>
                      <li className="flex items-start gap-2">
                        <CheckCircle className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                        <span>Excellent modularity - strong component grouping suggests well-defined boundaries</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <Info className="h-4 w-4 text-blue-500 mt-0.5 flex-shrink-0" />
                        <span>High clustering enables independent development and testing of component groups</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <Info className="h-4 w-4 text-blue-500 mt-0.5 flex-shrink-0" />
                        <span>Ensure inter-cluster communication remains efficient to avoid bottlenecks</span>
                      </li>
                    </>
                  )}
                </ul>
              </CardContent>
            </Card>
          </>
        )}
      </div>
    </AppLayout>
  )
}
