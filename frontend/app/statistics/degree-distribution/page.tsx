"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { 
  Network, 
  ArrowLeft, 
  TrendingUp, 
  Activity, 
  AlertTriangle,
  CheckCircle2,
  BarChart3,
  Zap,
  Info
} from "lucide-react"
import { useConnection } from "@/lib/stores/connection-store"
import { API_BASE_URL } from "@/lib/config/api"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid
} from 'recharts'
import { ChartContainer, ChartTooltip, ChartTooltipContent, ChartLegend, ChartLegendContent } from "@/components/ui/chart"
import { Progress } from "@/components/ui/progress"

interface DegreeStats {
  in_degree: {
    mean: number
    median: number
    max: number
    min: number
    std: number
  }
  out_degree: {
    mean: number
    median: number
    max: number
    min: number
    std: number
  }
  total_degree: {
    mean: number
    median: number
    max: number
    min: number
    std: number
  }
  hub_nodes: Array<{
    id: string
    name: string
    degree: number
    type: string
  }>
  isolated_nodes: number
  computation_time_ms: number
}

export default function DegreeDistributionPage() {
  const router = useRouter()
  const { status, config, stats: connectionStats } = useConnection()
  const [loading, setLoading] = useState(false)
  const [stats, setStats] = useState<DegreeStats | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [selectedComponentType, setSelectedComponentType] = useState<string>('all')

  const isConnected = status === 'connected'
  
  // Get available component types from connection stats
  const componentTypes = connectionStats?.node_counts 
    ? Object.keys(connectionStats.node_counts).sort()
    : []

  // Fetch degree distribution stats
  const fetchStats = async () => {
    if (!isConnected || !config) return

    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/stats/degree-distribution`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          uri: config.uri,
          user: config.user,
          password: config.password,
          database: config.database,
          node_type: selectedComponentType === 'all' ? undefined : selectedComponentType,
        }),
      })

      if (!response.ok) {
        throw new Error(`Failed to fetch stats: ${response.statusText}`)
      }

      const data = await response.json()
      setStats(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (isConnected) {
      fetchStats()
    }
  }, [isConnected, selectedComponentType])

  // Loading State
  if (loading || (isConnected && !stats && !error)) {
    return (
      <AppLayout title="Degree Distribution" description="Node connectivity analysis">
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" text="Computing degree statistics..." />
        </div>
      </AppLayout>
    )
  }

  // Prepare chart data
  const degreeComparisonData = stats ? [
    {
      metric: 'Mean',
      'In-Degree': stats.in_degree.mean,
      'Out-Degree': stats.out_degree.mean,
      'Total': stats.total_degree.mean
    },
    {
      metric: 'Median',
      'In-Degree': stats.in_degree.median,
      'Out-Degree': stats.out_degree.median,
      'Total': stats.total_degree.median
    },
    {
      metric: 'Max',
      'In-Degree': stats.in_degree.max,
      'Out-Degree': stats.out_degree.max,
      'Total': stats.total_degree.max
    },
    {
      metric: 'Std Dev',
      'In-Degree': stats.in_degree.std,
      'Out-Degree': stats.out_degree.std,
      'Total': stats.total_degree.std
    }
  ] : []

  const hubTypeDistribution = stats?.hub_nodes.reduce((acc, hub) => {
    acc[hub.type] = (acc[hub.type] || 0) + 1
    return acc
  }, {} as Record<string, number>)

  const hubTypePieData = hubTypeDistribution ? Object.entries(hubTypeDistribution).map(([name, value]) => ({
    name,
    value
  })) : []

  const COLORS = [
    '#8b5cf6', // Vibrant Purple
    '#3b82f6', // Bright Blue
    '#10b981', // Emerald Green
    '#f59e0b', // Amber
    '#ef4444', // Red
    '#ec4899', // Pink
    '#14b8a6', // Teal
    '#f97316', // Orange
    '#06b6d4', // Cyan
    '#a855f7', // Purple variant
  ]

  return (
    <AppLayout title="Degree Distribution" description="Node connectivity patterns analysis">
      <div className="space-y-6">
        {/* Back Button */}
        <Button
          variant="outline"
          size="sm"
          onClick={() => router.push('/statistics')}
          className="gap-2"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Statistics
        </Button>

        {/* No Connection Info Banner */}
        {!isConnected && <NoConnectionInfo />}

        {isConnected && (
          <>
            {/* Page Header */}
            <Card className="relative overflow-hidden border-0 shadow-xl hover:shadow-2xl hover:shadow-violet-500/25 transition-all duration-300">
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-violet-500 via-purple-500 to-indigo-500">
                <div className="w-full h-full rounded-lg bg-gradient-to-r from-violet-600 via-purple-600 to-indigo-600" />
              </div>
              
              <CardContent className="p-8 relative text-white">
                <div className="flex items-center gap-4 mb-4">
                  <div className="rounded-xl bg-white/20 p-3">
                    <Network className="h-8 w-8" />
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center justify-between gap-3 mb-2">
                      <div className="flex items-center gap-3">
                        <h2 className="text-3xl font-bold">Degree Distribution Analysis</h2>
                        {stats && (
                          <Badge className="bg-white/20 text-white border-white/30">
                            {stats.computation_time_ms}ms
                          </Badge>
                        )}
                      </div>
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
                                    {connectionStats?.node_counts && (
                                      <Badge variant="outline" className="ml-2 text-xs">
                                        {connectionStats.node_counts[type]}
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
                        ? 'Analyze node connectivity patterns and identify system hubs'
                        : `Analyzing ${selectedComponentType.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')} components only`
                      }
                    </p>
                  </div>
                </div>
                
                <div className="bg-white/10 rounded-lg p-4 mt-4">
                  <div className="flex items-start gap-3">
                    <Activity className="h-5 w-5 mt-0.5 flex-shrink-0" />
                    <div className="text-sm">
                      <p className="font-semibold mb-1">What is Degree Distribution?</p>
                      <p className="text-white/90">
                        Degree measures the number of connections each node has. Hub nodes (degree &gt; mean + 2×std) are critical connection points. 
                        High standard deviation indicates uneven connectivity, suggesting potential bottlenecks and single points of failure.
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Error State */}
            {error && (
              <Card className="border-red-500/50 bg-red-500/5">
                <CardHeader>
                  <div className="flex items-center gap-2">
                    <AlertTriangle className="h-5 w-5 text-red-500" />
                    <CardTitle className="text-red-500">Error Loading Statistics</CardTitle>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground">{error}</p>
                  <Button onClick={fetchStats} variant="outline" className="mt-4" size="sm">
                    Retry
                  </Button>
                </CardContent>
              </Card>
            )}

            {/* Statistics Content */}
            {stats && (
              <>
                {/* Key Metrics Overview */}
                <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
                  {/* Average Degree */}
                  <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-violet-500/25 transition-all duration-300">
                    <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-violet-500 to-purple-500">
                      <div className="w-full h-full bg-background rounded-lg" />
                    </div>
                    <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-violet-500/30 via-violet-500/15 to-violet-500/5" />
                    
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
                      <CardTitle className="text-sm font-medium">Average Degree</CardTitle>
                      <div className="rounded-xl bg-violet-500/10 p-2.5">
                        <Activity className="h-4 w-4 text-violet-500" />
                      </div>
                    </CardHeader>
                    <CardContent className="relative">
                      <div className="text-3xl font-bold text-violet-500">
                        {stats.total_degree.mean.toFixed(2)}
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        Median: {stats.total_degree.median}
                      </p>
                    </CardContent>
                  </Card>

                  {/* Max Degree (Hub) */}
                  <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-purple-500/25 transition-all duration-300">
                    <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-purple-500 to-fuchsia-500">
                      <div className="w-full h-full bg-background rounded-lg" />
                    </div>
                    <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-purple-500/30 via-purple-500/15 to-purple-500/5" />
                    
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
                      <CardTitle className="text-sm font-medium">Largest Hub</CardTitle>
                      <div className="rounded-xl bg-purple-500/10 p-2.5">
                        <TrendingUp className="h-4 w-4 text-purple-500" />
                      </div>
                    </CardHeader>
                    <CardContent className="relative">
                      <div className="text-3xl font-bold text-purple-500">
                        {stats.total_degree.max}
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        connections
                      </p>
                    </CardContent>
                  </Card>

                  {/* Hub Nodes Count */}
                  <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-indigo-500/25 transition-all duration-300">
                    <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-indigo-500 to-blue-500">
                      <div className="w-full h-full bg-background rounded-lg" />
                    </div>
                    <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-indigo-500/30 via-indigo-500/15 to-indigo-500/5" />
                    
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
                      <CardTitle className="text-sm font-medium">Hub Nodes</CardTitle>
                      <div className="rounded-xl bg-indigo-500/10 p-2.5">
                        <Network className="h-4 w-4 text-indigo-500" />
                      </div>
                    </CardHeader>
                    <CardContent className="relative">
                      <div className="text-3xl font-bold text-indigo-500">
                        {stats.hub_nodes.length}
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        degree {'>'} μ + 2σ
                      </p>
                    </CardContent>
                  </Card>

                  {/* Isolated Nodes */}
                  <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-amber-500/25 transition-all duration-300">
                    <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-amber-500 to-orange-500">
                      <div className="w-full h-full bg-background rounded-lg" />
                    </div>
                    <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-amber-500/30 via-amber-500/15 to-amber-500/5" />
                    
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
                      <CardTitle className="text-sm font-medium">Isolated Nodes</CardTitle>
                      <div className="rounded-xl bg-amber-500/10 p-2.5">
                        {stats.isolated_nodes === 0 ? (
                          <CheckCircle2 className="h-4 w-4 text-green-500" />
                        ) : (
                          <AlertTriangle className="h-4 w-4 text-amber-500" />
                        )}
                      </div>
                    </CardHeader>
                    <CardContent className="relative">
                      <div className={`text-3xl font-bold ${stats.isolated_nodes === 0 ? 'text-green-500' : 'text-amber-500'}`}>
                        {stats.isolated_nodes}
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        {stats.isolated_nodes === 0 ? 'All connected' : 'degree = 0'}
                      </p>
                    </CardContent>
                  </Card>
                </div>

                {/* Charts Section */}
                <div className="grid gap-6 lg:grid-cols-2">
                  {/* Degree Statistics Comparison */}
                  <Card className="relative overflow-hidden border-0 shadow-lg">
                    <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-violet-500 via-purple-500 to-indigo-500">
                      <div className="w-full h-full bg-background rounded-lg" />
                    </div>
                    <div className="absolute inset-[2px] rounded-lg bg-background" />
                    
                    <CardHeader className="relative">
                      <div className="flex items-center gap-2">
                        <div className="rounded-xl bg-violet-500/10 p-2.5">
                          <BarChart3 className="h-5 w-5 text-violet-500" />
                        </div>
                        <div>
                          <CardTitle>Degree Statistics</CardTitle>
                          <CardDescription>In-degree, Out-degree, and Total degree metrics</CardDescription>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="relative">
                      <ChartContainer
                        config={{
                          "In-Degree": {
                            label: "In-Degree",
                            color: "hsl(271 91% 65%)"
                          },
                          "Out-Degree": {
                            label: "Out-Degree",
                            color: "hsl(280 85% 70%)"
                          },
                          "Total": {
                            label: "Total",
                            color: "hsl(288 82% 75%)"
                          },
                        }}
                        className="h-[300px] w-full"
                      >
                        <BarChart data={degreeComparisonData}>
                          <CartesianGrid vertical={false} />
                          <XAxis
                            dataKey="metric"
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
                          <ChartLegend content={<ChartLegendContent />} />
                          <Bar
                            dataKey="In-Degree"
                            fill="var(--color-In-Degree)"
                            radius={[6, 6, 0, 0]}
                          />
                          <Bar
                            dataKey="Out-Degree"
                            fill="var(--color-Out-Degree)"
                            radius={[6, 6, 0, 0]}
                          />
                          <Bar
                            dataKey="Total"
                            fill="var(--color-Total)"
                            radius={[6, 6, 0, 0]}
                          />
                        </BarChart>
                      </ChartContainer>
                    </CardContent>
                  </Card>

                  {/* Hub Node Type Distribution */}
                  <Card className="relative overflow-hidden border-0 shadow-lg">
                    <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-purple-500 via-fuchsia-500 to-pink-500">
                      <div className="w-full h-full bg-background rounded-lg" />
                    </div>
                    <div className="absolute inset-[2px] rounded-lg bg-background" />
                    
                    <CardHeader className="relative">
                      <div className="flex items-center gap-2">
                        <div className="rounded-xl bg-purple-500/10 p-2.5">
                          <Network className="h-5 w-5 text-purple-500" />
                        </div>
                        <div>
                          <CardTitle>Hub Node Types</CardTitle>
                          <CardDescription>Distribution of hub nodes by component type</CardDescription>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent className="relative">
                      {hubTypePieData.length > 0 ? (
                        <div className="space-y-4">
                          {hubTypePieData.map((item, idx) => (
                            <div key={idx} className="space-y-2">
                              <div className="flex items-center justify-between text-sm">
                                <span className="font-medium">{item.name}</span>
                                <span className="text-muted-foreground">{item.value} hubs</span>
                              </div>
                              <Progress 
                                value={(item.value / hubTypePieData.reduce((sum, d) => sum + d.value, 0)) * 100} 
                                className="h-2"
                                style={{ 
                                  ['--progress-background' as string]: COLORS[idx % COLORS.length] 
                                }}
                              />
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="h-[300px] flex items-center justify-center text-muted-foreground">
                          No hub nodes detected
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </div>

                {/* Hub Nodes List */}
                {stats.hub_nodes.length > 0 && (
                  <Card className="relative overflow-hidden border-0 shadow-lg">
                    <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-indigo-500 via-blue-500 to-cyan-500">
                      <div className="w-full h-full bg-background rounded-lg" />
                    </div>
                    <div className="absolute inset-[2px] rounded-lg bg-background" />
                    
                    <CardHeader className="relative">
                      <div className="flex items-center gap-2">
                        <div className="rounded-xl bg-indigo-500/10 p-2.5">
                          <Zap className="h-5 w-5 text-indigo-500" />
                        </div>
                        <div>
                          <CardTitle>Identified Hub Nodes</CardTitle>
                          <CardDescription>
                            Nodes with degree greater than mean + 2 standard deviations (critical connection points)
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
                              </tr>
                            </thead>
                            <tbody>
                              {stats.hub_nodes.map((hub, index) => (
                                <tr key={hub.id} className="border-b last:border-0 hover:bg-muted/30 transition-colors cursor-pointer group" onClick={() => router.push(`/explorer?node=${encodeURIComponent(hub.id)}`)}>
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
                                    <div className="font-medium text-sm group-hover:underline transition-all">{hub.name}</div>
                                    <div className="text-xs text-muted-foreground font-mono">{hub.id}</div>
                                  </td>
                                  <td className="p-3">
                                    <Badge variant="outline" className="text-xs">
                                      {hub.type}
                                    </Badge>
                                  </td>
                                  <td className="p-3 text-right">
                                    <div className="font-semibold text-indigo-600 dark:text-indigo-400">
                                      {hub.degree}
                                    </div>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                      <p className="text-xs text-muted-foreground mt-3">
                        These hub nodes are critical connection points. Their failure would severely impact system functionality.
                      </p>
                    </CardContent>
                  </Card>
                )}

                {/* Insights Card */}
                <Card className="relative overflow-hidden border-0 shadow-lg">
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-slate-400 via-gray-500 to-slate-600">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  <div className="absolute inset-[2px] rounded-lg bg-background" />
                  
                  <CardHeader className="relative">
                    <CardTitle className="text-base">Interpretation & Insights</CardTitle>
                  </CardHeader>
                  <CardContent className="relative">
                    <div className="space-y-3 text-sm text-muted-foreground">
                      <div className="flex items-start gap-3">
                        <div className="rounded-lg bg-violet-500/10 p-2 mt-0.5">
                          <TrendingUp className="h-4 w-4 text-violet-500" />
                        </div>
                        <div>
                          <span className="font-semibold text-foreground">High degree nodes (hubs)</span> are critical connection points. 
                          Their failure would disconnect many components and severely impact system functionality.
                        </div>
                      </div>
                      <div className="flex items-start gap-3">
                        <div className="rounded-lg bg-purple-500/10 p-2 mt-0.5">
                          <Activity className="h-4 w-4 text-purple-500" />
                        </div>
                        <div>
                          <span className="font-semibold text-foreground">High standard deviation</span> indicates uneven connectivity distribution, 
                          suggesting potential bottlenecks and single points of failure.
                        </div>
                      </div>
                      <div className="flex items-start gap-3">
                        <div className="rounded-lg bg-amber-500/10 p-2 mt-0.5">
                          <AlertTriangle className="h-4 w-4 text-amber-500" />
                        </div>
                        <div>
                          <span className="font-semibold text-foreground">Isolated nodes</span> represent disconnected components 
                          that cannot communicate with the rest of the system and may indicate configuration issues.
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </>
            )}
          </>
        )}
      </div>
    </AppLayout>
  )
}
