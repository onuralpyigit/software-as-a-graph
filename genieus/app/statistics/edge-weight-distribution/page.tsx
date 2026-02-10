"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ArrowLeft, GitCompare, AlertTriangle, CheckCircle, Info, BarChart3, ArrowRight } from "lucide-react"
import { useConnection } from "@/lib/stores/connection-store"
import { apiClient } from "@/lib/api/client"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Progress } from "@/components/ui/progress"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Bar, BarChart, XAxis, YAxis, CartesianGrid } from "recharts"

interface Edge {
  source: string
  target: string
  source_name: string
  target_name: string
  type: string
  weight: number
}

interface TypeStats {
  count: number
  total_weight: number
  avg_weight: number
  median_weight: number
  min_weight: number
  max_weight: number
  std_weight: number
}

interface EdgeWeightStats {
  total_edges: number
  total_weight: number
  avg_weight: number
  median_weight: number
  min_weight: number
  max_weight: number
  std_weight: number
  weight_concentration: number
  interpretation: string
  category: string
  health: string
  very_high_count: number
  high_count: number
  medium_count: number
  low_count: number
  very_low_count: number
  top_edges: Edge[]
  type_stats: Record<string, TypeStats>
}

export default function EdgeWeightDistributionPage() {
  const router = useRouter()
  const { status, stats: graphStats, initialLoadComplete } = useConnection()
  const [stats, setStats] = useState<EdgeWeightStats | null>(null)
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
      const result = await apiClient.getEdgeWeightDistributionStats()
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
          label: 'Healthy Distribution'
        }
      case 'fair':
        return { 
          color: 'text-blue-500', 
          icon: Info, 
          label: 'Fair Distribution'
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

  // Format edge type for display
  const formatEdgeType = (type: string) => {
    return type
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ')
  }

  // Prepare chart data
  const distributionData = stats ? [
    { name: 'Very High', count: stats.very_high_count },
    { name: 'High', count: stats.high_count },
    { name: 'Medium', count: stats.medium_count },
    { name: 'Low', count: stats.low_count },
    { name: 'Very Low', count: stats.very_low_count }
  ] : []

  // Prepare type stats chart data
  const typeStatsData = stats ? Object.entries(stats.type_stats).map(([type, data]) => ({
    type: formatEdgeType(type),
    avgWeight: data.avg_weight,
    count: data.count
  })) : []

  // Loading State
  if (!initialLoadComplete || status === 'connecting') {
    return (
      <AppLayout title="Edge Weight Distribution" description="Analyze dependency strength distribution">
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" text="Connecting to database..." />
        </div>
      </AppLayout>
    )
  }

  // Disconnected State
  if (!isConnected) {
    return (
      <AppLayout title="Edge Weight Distribution" description="Analyze dependency strength distribution">
        <NoConnectionInfo />
      </AppLayout>
    )
  }

  return (
    <AppLayout title="Edge Weight Distribution" description="Analyze dependency strength distribution">
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
        <Card className="relative overflow-hidden border-0 shadow-xl hover:shadow-2xl hover:shadow-emerald-500/25 transition-all duration-300">
          <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-emerald-500 via-green-500 to-teal-500">
            <div className="w-full h-full rounded-lg bg-gradient-to-r from-emerald-600 via-green-600 to-teal-600" />
          </div>
          
          <CardContent className="p-8 relative text-white">
            <div className="flex items-center gap-4 mb-4">
              <div className="rounded-xl bg-white/20 p-3">
                <GitCompare className="h-8 w-8" />
              </div>
              <div className="flex-1">
                <h3 className="text-3xl font-bold">Edge Weight Distribution Analysis</h3>
                <p className="text-white/90 mt-2">
                  Understanding how dependency strength is distributed across connections
                </p>
              </div>
            </div>
            
            <div className="bg-white/10 rounded-lg p-4 mt-4">
              <div className="flex items-start gap-3">
                <Info className="h-5 w-5 mt-0.5 flex-shrink-0" />
                <div className="text-sm">
                  <p className="font-semibold mb-1">What is Edge Weight Distribution?</p>
                  <p className="text-white/90">
                    Edge weight represents dependency strength calculated from connection patterns and quality metrics.
                    This analysis reveals whether critical dependencies are concentrated in few connections or distributed evenly.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Loading/Error States */}
        {loading && (
          <div className="flex items-center justify-center py-12">
            <LoadingSpinner size="lg" text="Analyzing edge weight distribution..." />
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
            {/* Summary Cards */}
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              {/* Average Weight */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-green-500/25 transition-all duration-300 transform hover:scale-[1.02]">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-green-500/30 via-green-500/15 to-green-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Average Weight</CardTitle>
                  <div className="rounded-xl bg-green-500/10 p-2.5">
                    <BarChart3 className="h-4 w-4 text-green-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-green-500">
                    {stats.avg_weight.toFixed(3)}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Median: {stats.median_weight.toFixed(3)}
                  </p>
                </CardContent>
              </Card>

              {/* Weight Range */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-teal-500/25 transition-all duration-300 transform hover:scale-[1.02]">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-teal-500 via-cyan-500 to-blue-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-teal-500/30 via-teal-500/15 to-teal-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Weight Range</CardTitle>
                  <div className="rounded-xl bg-teal-500/10 p-2.5">
                    <GitCompare className="h-4 w-4 text-teal-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-teal-500">
                    {stats.min_weight.toFixed(3)} - {stats.max_weight.toFixed(3)}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Std Dev: {stats.std_weight.toFixed(3)}
                  </p>
                </CardContent>
              </Card>

              {/* Total Weight */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-emerald-500/25 transition-all duration-300 transform hover:scale-[1.02]">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-emerald-500 via-green-500 to-lime-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-emerald-500/30 via-emerald-500/15 to-emerald-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Weight</CardTitle>
                  <div className="rounded-xl bg-emerald-500/10 p-2.5">
                    <BarChart3 className="h-4 w-4 text-emerald-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-emerald-500">
                    {stats.total_weight.toFixed(2)}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {stats.total_edges} dependencies
                  </p>
                </CardContent>
              </Card>

              {/* Weight Concentration */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-lime-500/25 transition-all duration-300 transform hover:scale-[1.02]">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-lime-500 via-green-500 to-emerald-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-lime-500/30 via-lime-500/15 to-lime-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Concentration</CardTitle>
                  <div className="rounded-xl bg-lime-500/10 p-2.5">
                    <GitCompare className="h-4 w-4 text-lime-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-lime-500">
                    {stats.weight_concentration.toFixed(1)}%
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Top 20% of edges
                  </p>
                  <Progress value={stats.weight_concentration} className="h-1 mt-2" />
                </CardContent>
              </Card>
            </div>

            {/* Health Assessment */}
            <Card className="relative overflow-hidden border-0 shadow-lg">
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-emerald-500 via-green-500 to-teal-500">
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              <div className="absolute inset-[2px] rounded-lg bg-background" />
              
              <CardHeader className="relative">
                <div className="flex items-center gap-2">
                  <div className="rounded-xl bg-emerald-500/10 p-2.5">
                    <GitCompare className="h-5 w-5 text-emerald-500" />
                  </div>
                  <div>
                    <CardTitle>Distribution Health Assessment</CardTitle>
                    <CardDescription>
                      Overall dependency weight distribution and concentration analysis
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

            {/* Distribution Charts Side by Side */}
            <div className="grid gap-6 md:grid-cols-2">
              {/* Distribution Chart */}
              <Card className="relative overflow-hidden border-0 shadow-lg">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-background" />
                
                <CardHeader className="relative">
                  <CardTitle>Weight Distribution by Category</CardTitle>
                  <CardDescription>
                    Dependency count across different weight categories
                  </CardDescription>
                </CardHeader>
                
                <CardContent className="relative">
                  <ChartContainer
                    config={{
                      count: {
                        label: "Dependencies",
                        theme: {
                          light: "hsl(142 76% 36%)",
                          dark: "hsl(142 76% 50%)"
                        }
                      },
                    }}
                    className="h-[300px] w-full"
                  >
                    <BarChart data={distributionData}>
                      <CartesianGrid vertical={false} />
                      <XAxis
                        dataKey="name"
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

              {/* Type Statistics Chart */}
              {typeStatsData.length > 0 && (
                <Card className="relative overflow-hidden border-0 shadow-lg">
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-cyan-500 via-teal-500 to-green-500">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  <div className="absolute inset-[2px] rounded-lg bg-background" />
                  
                  <CardHeader className="relative">
                    <CardTitle>Average Weight by Dependency Type</CardTitle>
                    <CardDescription>
                      How different dependency types compare in strength
                    </CardDescription>
                  </CardHeader>
                  
                  <CardContent className="relative">
                    <ChartContainer
                      config={{
                        avgWeight: {
                          label: "Avg Weight",
                          theme: {
                            light: "hsl(158 64% 52%)",
                            dark: "hsl(158 64% 60%)"
                          }
                        },
                      }}
                      className="h-[300px] w-full"
                    >
                      <BarChart data={typeStatsData}>
                        <CartesianGrid vertical={false} />
                        <XAxis
                          dataKey="type"
                          tickLine={false}
                          tickMargin={10}
                          axisLine={false}
                        />
                        <YAxis
                          tickLine={false}
                          axisLine={false}
                          tickMargin={8}
                        />
                        <ChartTooltip
                          content={<ChartTooltipContent />}
                          formatter={(value: number) => value.toFixed(4)}
                        />
                        <Bar
                          dataKey="avgWeight"
                          fill="var(--color-avgWeight)"
                          radius={[8, 8, 0, 0]}
                        />
                      </BarChart>
                    </ChartContainer>
                  </CardContent>
                </Card>
              )}
            </div>

            {/* Top Edges Table */}
            {stats.top_edges.length > 0 && (
              <Card className="relative overflow-hidden border-0 shadow-lg">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-emerald-500 via-teal-500 to-cyan-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-background" />
                
                <CardHeader className="relative">
                  <CardTitle>Top Weighted Dependencies</CardTitle>
                  <CardDescription>
                    Dependencies with highest strength scores
                  </CardDescription>
                </CardHeader>
                
                <CardContent className="relative">
                  <div className="rounded-md border">
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead>
                          <tr className="border-b bg-muted/50">
                            <th className="p-3 text-left text-sm font-medium">Rank</th>
                            <th className="p-3 text-left text-sm font-medium">Source</th>
                            <th className="p-3 w-[50px]"></th>
                            <th className="p-3 text-left text-sm font-medium">Target</th>
                            <th className="p-3 text-left text-sm font-medium">Type</th>
                            <th className="p-3 text-right text-sm font-medium">Weight</th>
                          </tr>
                        </thead>
                        <tbody>
                          {stats.top_edges.map((edge, index) => (
                            <tr key={`${edge.source}-${edge.target}`} className="border-b last:border-0 hover:bg-muted/30 transition-colors group">
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
                              <td className="p-3 cursor-pointer" onClick={() => router.push(`/explorer?node=${encodeURIComponent(edge.source)}`)}>
                                <div className="font-medium text-sm group-hover:underline transition-all">{edge.source_name}</div>
                                <div className="text-xs text-muted-foreground font-mono">{edge.source}</div>
                              </td>
                              <td className="p-3">
                                <ArrowRight className="h-4 w-4 text-muted-foreground" />
                              </td>
                              <td className="p-3 cursor-pointer" onClick={() => router.push(`/explorer?node=${encodeURIComponent(edge.target)}`)}>
                                <div className="font-medium text-sm group-hover:underline transition-all">{edge.target_name}</div>
                                <div className="text-xs text-muted-foreground font-mono">{edge.target}</div>
                              </td>
                              <td className="p-3">
                                <Badge variant="outline" className="text-xs">
                                  {formatEdgeType(edge.type)}
                                </Badge>
                              </td>
                              <td className="p-3 text-right">
                                <div className="font-semibold text-emerald-600 dark:text-emerald-400">
                                  {edge.weight.toFixed(4)}
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
          </>
        )}
      </div>
    </AppLayout>
  )
}
