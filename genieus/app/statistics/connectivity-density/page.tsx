"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Activity, ArrowLeft, Network, TrendingUp, AlertTriangle, CheckCircle, Info, CheckCircle2 } from "lucide-react"
import { useConnection } from "@/lib/stores/connection-store"
import { apiClient } from "@/lib/api/client"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

interface DenseComponent {
  id: string
  name: string
  type: string
  degree: number
  density_contribution: number
}

interface ConnectivityDensityStats {
  density: number
  total_nodes: number
  total_edges: number
  max_possible_edges: number
  interpretation: string
  category: string
  most_dense_components: DenseComponent[]
}

export default function ConnectivityDensityPage() {
  const router = useRouter()
  const { status, stats: graphStats, initialLoadComplete } = useConnection()
  const [stats, setStats] = useState<ConnectivityDensityStats | null>(null)
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
      const result = await apiClient.getConnectivityDensityStats(nodeType)
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
    
    if (stats.density < 0.05) {
      return { 
        color: 'text-green-500', 
        icon: CheckCircle, 
        label: 'Sparse',
        description: 'Low coupling - components are loosely connected, promoting modularity and easier maintenance'
      }
    } else if (stats.density < 0.15) {
      return { 
        color: 'text-blue-500', 
        icon: Info, 
        label: 'Moderate',
        description: 'Balanced connectivity - reasonable interconnection without excessive coupling'
      }
    } else if (stats.density < 0.30) {
      return { 
        color: 'text-orange-500', 
        icon: AlertTriangle, 
        label: 'Dense',
        description: 'High coupling - many interdependencies may complicate changes and increase failure propagation'
      }
    } else {
      return { 
        color: 'text-red-500', 
        icon: AlertTriangle, 
        label: 'Very Dense',
        description: 'Very high coupling - excessive interconnections create complexity and maintenance challenges'
      }
    }
  }

  const interpretation = getInterpretation()
  const Icon = interpretation.icon

  // Loading State
  if (!initialLoadComplete || status === 'connecting') {
    return (
      <AppLayout title="Connectivity Density" description="Analyze system coupling and interconnection">
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" text="Connecting to database..." />
        </div>
      </AppLayout>
    )
  }

  // Disconnected State
  if (!isConnected) {
    return (
      <AppLayout title="Connectivity Density" description="Analyze system coupling and interconnection">
        <NoConnectionInfo />
      </AppLayout>
    )
  }

  return (
    <AppLayout title="Connectivity Density" description="Analyze system coupling and interconnection">
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
        <Card className="relative overflow-hidden border-0 shadow-xl hover:shadow-2xl hover:shadow-cyan-500/25 transition-all duration-300">
          <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-cyan-500 via-teal-500 to-emerald-500">
            <div className="w-full h-full rounded-lg bg-gradient-to-r from-cyan-600 via-teal-600 to-emerald-600" />
          </div>
          
          <CardContent className="p-8 relative text-white">
            <div className="flex items-center gap-4 mb-4">
              <div className="rounded-xl bg-white/20 p-3">
                <Activity className="h-8 w-8" />
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between gap-3 mb-2">
                  <h3 className="text-3xl font-bold">Connectivity Density Analysis</h3>
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
                    ? 'Understanding system coupling through connection density metrics'
                    : `Analyzing ${selectedComponentType.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')} components only`
                  }
                </p>
              </div>
            </div>
            
            <div className="bg-white/10 rounded-lg p-4 mt-4">
              <div className="flex items-start gap-3">
                <Info className="h-5 w-5 mt-0.5 flex-shrink-0" />
                <div className="text-sm">
                  <p className="font-semibold mb-1">What is Connectivity Density?</p>
                  <p className="text-white/90">
                    Density = (Actual Edges) / (Maximum Possible Edges). For a directed graph with N nodes, 
                    max edges = N × (N-1). Higher density indicates tighter coupling between components.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Loading/Error States */}
        {loading && (
          <div className="flex items-center justify-center py-12">
            <LoadingSpinner size="lg" text="Computing connectivity density..." />
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
            {/* Main Density Metric */}
            <Card className="relative overflow-hidden border-0 shadow-lg">
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-cyan-500 via-teal-500 to-emerald-500">
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              <div className="absolute inset-[2px] rounded-lg bg-background" />
              
              <CardHeader className="relative">
                <div className="flex items-center gap-2">
                  <div className="rounded-xl bg-cyan-500/10 p-2.5">
                    <Activity className="h-5 w-5 text-cyan-500" />
                  </div>
                  <div>
                    <CardTitle>Network Density Score</CardTitle>
                    <CardDescription>
                      Ratio of actual connections to maximum possible connections
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              
              <CardContent className="relative space-y-6">
                {/* Large Density Display */}
                <div className="text-center py-8">
                  <div className="text-7xl font-bold bg-gradient-to-r from-cyan-500 via-teal-500 to-emerald-500 bg-clip-text text-transparent">
                    {(stats.density * 100).toFixed(2)}%
                  </div>
                  <p className="text-muted-foreground mt-2">Connectivity Density</p>
                </div>

                {/* Progress Bar */}
                <div className="space-y-2">
                  <Progress value={stats.density * 100} className="h-3" />
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>0% (Sparse)</span>
                    <span>50%</span>
                    <span>100% (Complete Graph)</span>
                  </div>
                </div>

                {/* Interpretation */}
                <Alert className={`border-2 ${interpretation.color.replace('text-', 'border-')}`}>
                  <Icon className={`h-5 w-5 ${interpretation.color}`} />
                  <AlertTitle className="text-lg">
                    {interpretation.label} Connectivity
                  </AlertTitle>
                  <AlertDescription className="text-base">
                    {interpretation.description}
                  </AlertDescription>
                </Alert>
              </CardContent>
            </Card>

            {/* Detailed Metrics */}
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              {/* Total Nodes */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-blue-500/25 transition-all duration-300 transform hover:scale-[1.02]">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-blue-500/30 via-blue-500/15 to-blue-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Nodes</CardTitle>
                  <div className="rounded-xl bg-blue-500/10 p-2.5">
                    <Network className="h-4 w-4 text-blue-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-blue-500">
                    {stats.total_nodes.toLocaleString()}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">Components in system</p>
                </CardContent>
              </Card>

              {/* Actual Edges */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-cyan-500/25 transition-all duration-300 transform hover:scale-[1.02]">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-cyan-500 via-teal-500 to-emerald-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-cyan-500/30 via-cyan-500/15 to-cyan-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Actual Edges</CardTitle>
                  <div className="rounded-xl bg-cyan-500/10 p-2.5">
                    <TrendingUp className="h-4 w-4 text-cyan-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-cyan-500">
                    {stats.total_edges.toLocaleString()}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">Real connections</p>
                </CardContent>
              </Card>

              {/* Max Possible Edges */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-emerald-500/25 transition-all duration-300 transform hover:scale-[1.02]">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-emerald-500/30 via-emerald-500/15 to-emerald-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Max Possible</CardTitle>
                  <div className="rounded-xl bg-emerald-500/10 p-2.5">
                    <Network className="h-4 w-4 text-emerald-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-emerald-500">
                    {stats.max_possible_edges.toLocaleString()}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">Complete graph</p>
                </CardContent>
              </Card>

              {/* Utilization */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-purple-500/25 transition-all duration-300 transform hover:scale-[1.02]">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-purple-500 via-fuchsia-500 to-pink-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-purple-500/30 via-purple-500/15 to-purple-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Edge Utilization</CardTitle>
                  <div className="rounded-xl bg-purple-500/10 p-2.5">
                    <Activity className="h-4 w-4 text-purple-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-purple-500">
                    {((stats.total_edges / stats.max_possible_edges) * 100).toFixed(1)}%
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">Of max capacity</p>
                </CardContent>
              </Card>
            </div>

            {/* Insights */}
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
                    <div className="rounded-full bg-blue-500/10 p-1 mt-0.5">
                      <div className="w-2 h-2 rounded-full bg-blue-500" />
                    </div>
                    <div className="flex-1">
                      <p className="font-medium">Coupling Impact</p>
                      <p className="text-sm text-muted-foreground">
                        {stats.density < 0.05 
                          ? "Low coupling allows independent changes and reduces failure cascades."
                          : stats.density < 0.15
                          ? "Moderate coupling balances integration benefits with maintainability."
                          : "High coupling may lead to complex dependencies and change ripple effects."}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start gap-3">
                    <div className="rounded-full bg-cyan-500/10 p-1 mt-0.5">
                      <div className="w-2 h-2 rounded-full bg-cyan-500" />
                    </div>
                    <div className="flex-1">
                      <p className="font-medium">Maintainability</p>
                      <p className="text-sm text-muted-foreground">
                        {stats.density < 0.10
                          ? "Good modularity - changes are likely localized to individual components."
                          : "Higher interconnection requires careful change management and testing."}
                      </p>
                    </div>
                  </div>

                  <div className="flex items-start gap-3">
                    <div className="rounded-full bg-emerald-500/10 p-1 mt-0.5">
                      <div className="w-2 h-2 rounded-full bg-emerald-500" />
                    </div>
                    <div className="flex-1">
                      <p className="font-medium">System Architecture</p>
                      <p className="text-sm text-muted-foreground">
                        Average of {(stats.total_edges / stats.total_nodes).toFixed(1)} connections per component.
                        {stats.density < 0.10 
                          ? " System follows good separation of concerns."
                          : " Consider reviewing component boundaries."}
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Most Dense Components */}
            {stats.most_dense_components && stats.most_dense_components.length > 0 && (
              <Card className="relative overflow-hidden border-0 shadow-lg">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-cyan-500 via-teal-500 to-emerald-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-background" />
                
                <CardHeader className="relative">
                  <div className="flex items-center gap-2">
                    <div className="rounded-xl bg-cyan-500/10 p-2.5">
                      <Network className="h-5 w-5 text-cyan-500" />
                    </div>
                    <div>
                      <CardTitle>Most Connected Components</CardTitle>
                      <CardDescription>
                        Components with the highest number of connections - potential coupling hotspots
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
                            <th className="p-3 text-right text-sm font-medium">Connections</th>
                            <th className="p-3 text-right text-sm font-medium">% of Total</th>
                          </tr>
                        </thead>
                        <tbody>
                          {stats.most_dense_components.map((component, index) => (
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
                                <div className="font-semibold text-cyan-600 dark:text-cyan-400">
                                  {component.degree}
                                </div>
                              </td>
                              <td className="p-3 text-right">
                                <div className="text-sm text-muted-foreground">
                                  {component.density_contribution}%
                                </div>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                  <p className="text-xs text-muted-foreground mt-3">
                    These components have the most connections and may be coupling hotspots. Consider reviewing their responsibilities and dependencies.
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
                  {stats.density < 0.05 ? (
                    <>
                      <li className="flex items-start gap-2">
                        <CheckCircle className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                        <span>Excellent - maintain this low coupling through clear interfaces and bounded contexts</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <CheckCircle className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                        <span>Monitor for isolated components that may indicate missing integrations</span>
                      </li>
                    </>
                  ) : stats.density < 0.15 ? (
                    <>
                      <li className="flex items-start gap-2">
                        <Info className="h-4 w-4 text-blue-500 mt-0.5 flex-shrink-0" />
                        <span>Good balance - ensure new features don't unnecessarily increase coupling</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <Info className="h-4 w-4 text-blue-500 mt-0.5 flex-shrink-0" />
                        <span>Use degree distribution analysis to identify overly connected components</span>
                      </li>
                    </>
                  ) : (
                    <>
                      <li className="flex items-start gap-2">
                        <AlertTriangle className="h-4 w-4 text-orange-500 mt-0.5 flex-shrink-0" />
                        <span>Consider refactoring to reduce dependencies between components</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <AlertTriangle className="h-4 w-4 text-orange-500 mt-0.5 flex-shrink-0" />
                        <span>Introduce abstraction layers or use event-driven patterns to decouple</span>
                      </li>
                      <li className="flex items-start gap-2">
                        <AlertTriangle className="h-4 w-4 text-orange-500 mt-0.5 flex-shrink-0" />
                        <span>Review critical paths to reduce cascade failure risk</span>
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
