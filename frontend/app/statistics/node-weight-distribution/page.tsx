"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ArrowLeft, TrendingUp, AlertTriangle, CheckCircle, Info, BarChart3 } from "lucide-react"
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

interface Component {
  id: string
  name: string
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

interface NodeWeightStats {
  total_components: number
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
  top_components: Component[]
  type_stats: Record<string, TypeStats>
}

export default function NodeWeightDistributionPage() {
  const router = useRouter()
  const { status, stats: graphStats, initialLoadComplete } = useConnection()
  const [stats, setStats] = useState<NodeWeightStats | null>(null)
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
      const result = await apiClient.getNodeWeightDistributionStats()
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
    type,
    avgWeight: data.avg_weight,
    count: data.count
  })) : []

  // Loading State
  if (!initialLoadComplete || status === 'connecting') {
    return (
      <AppLayout title="Node Weight Distribution" description="Analyze component importance distribution">
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" text="Connecting to database..." />
        </div>
      </AppLayout>
    )
  }

  // Disconnected State
  if (!isConnected) {
    return (
      <AppLayout title="Node Weight Distribution" description="Analyze component importance distribution">
        <NoConnectionInfo />
      </AppLayout>
    )
  }

  return (
    <AppLayout title="Node Weight Distribution" description="Analyze component importance distribution">
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
        <Card className="relative overflow-hidden border-0 shadow-xl hover:shadow-2xl hover:shadow-indigo-500/25 transition-all duration-300">
          <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500">
            <div className="w-full h-full rounded-lg bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-600" />
          </div>
          
          <CardContent className="p-8 relative text-white">
            <div className="flex items-center gap-4 mb-4">
              <div className="rounded-xl bg-white/20 p-3">
                <TrendingUp className="h-8 w-8" />
              </div>
              <div className="flex-1">
                <h3 className="text-3xl font-bold">Node Weight Distribution Analysis</h3>
                <p className="text-white/90 mt-2">
                  Understanding how component importance is distributed across the system
                </p>
              </div>
            </div>
            
            <div className="bg-white/10 rounded-lg p-4 mt-4">
              <div className="flex items-start gap-3">
                <Info className="h-5 w-5 mt-0.5 flex-shrink-0" />
                <div className="text-sm">
                  <p className="font-semibold mb-1">What is Node Weight Distribution?</p>
                  <p className="text-white/90">
                    Node weight represents component importance calculated from structural metrics (degree, centrality) and quality scores.
                    This analysis reveals whether importance is concentrated in few critical components or spread evenly across the system.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Loading/Error States */}
        {loading && (
          <div className="flex items-center justify-center py-12">
            <LoadingSpinner size="lg" text="Analyzing node weight distribution..." />
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
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-purple-500/25 transition-all duration-300 transform hover:scale-[1.02]">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-purple-500 via-indigo-500 to-blue-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-purple-500/30 via-purple-500/15 to-purple-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Average Weight</CardTitle>
                  <div className="rounded-xl bg-purple-500/10 p-2.5">
                    <BarChart3 className="h-4 w-4 text-purple-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-purple-500">
                    {stats.avg_weight.toFixed(3)}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Median: {stats.median_weight.toFixed(3)}
                  </p>
                </CardContent>
              </Card>

              {/* Weight Range */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-blue-500/25 transition-all duration-300 transform hover:scale-[1.02]">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-cyan-500 to-teal-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-blue-500/30 via-blue-500/15 to-blue-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Weight Range</CardTitle>
                  <div className="rounded-xl bg-blue-500/10 p-2.5">
                    <TrendingUp className="h-4 w-4 text-blue-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-blue-500">
                    {stats.min_weight.toFixed(3)} - {stats.max_weight.toFixed(3)}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Std Dev: {stats.std_weight.toFixed(3)}
                  </p>
                </CardContent>
              </Card>

              {/* Total Weight */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-green-500/25 transition-all duration-300 transform hover:scale-[1.02]">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-green-500/30 via-green-500/15 to-green-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Weight</CardTitle>
                  <div className="rounded-xl bg-green-500/10 p-2.5">
                    <BarChart3 className="h-4 w-4 text-green-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-green-500">
                    {stats.total_weight.toFixed(2)}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {stats.total_components} components
                  </p>
                </CardContent>
              </Card>

              {/* Weight Concentration */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-orange-500/25 transition-all duration-300 transform hover:scale-[1.02]">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-orange-500 via-amber-500 to-yellow-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-orange-500/30 via-orange-500/15 to-orange-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Concentration</CardTitle>
                  <div className="rounded-xl bg-orange-500/10 p-2.5">
                    <TrendingUp className="h-4 w-4 text-orange-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-orange-500">
                    {stats.weight_concentration.toFixed(1)}%
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Top 20% of components
                  </p>
                  <Progress value={stats.weight_concentration} className="h-1 mt-2" />
                </CardContent>
              </Card>
            </div>

            {/* Health Assessment */}
            <Card className="relative overflow-hidden border-0 shadow-lg">
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500">
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              <div className="absolute inset-[2px] rounded-lg bg-background" />
              
              <CardHeader className="relative">
                <div className="flex items-center gap-2">
                  <div className="rounded-xl bg-indigo-500/10 p-2.5">
                    <TrendingUp className="h-5 w-5 text-indigo-500" />
                  </div>
                  <div>
                    <CardTitle>Distribution Health Assessment</CardTitle>
                    <CardDescription>
                      Overall weight distribution and concentration analysis
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
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-purple-500 via-pink-500 to-rose-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-background" />
                
                <CardHeader className="relative">
                  <CardTitle>Weight Distribution by Category</CardTitle>
                  <CardDescription>
                    Component count across different weight categories
                  </CardDescription>
                </CardHeader>
                
                <CardContent className="relative">
                  <ChartContainer
                    config={{
                      count: {
                        label: "Components",
                        theme: {
                          light: "hsl(271 91% 65%)",
                          dark: "hsl(271 91% 70%)"
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
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-cyan-500 via-blue-500 to-indigo-500">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  <div className="absolute inset-[2px] rounded-lg bg-background" />
                  
                  <CardHeader className="relative">
                    <CardTitle>Average Weight by Component Type</CardTitle>
                    <CardDescription>
                      How different component types compare in importance
                    </CardDescription>
                  </CardHeader>
                  
                  <CardContent className="relative">
                    <ChartContainer
                      config={{
                        avgWeight: {
                          label: "Avg Weight",
                          theme: {
                            light: "hsl(217 91% 60%)",
                            dark: "hsl(217 91% 65%)"
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

            {/* Top Components Table */}
            {stats.top_components.length > 0 && (
              <Card className="relative overflow-hidden border-0 shadow-lg">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-purple-500 via-fuchsia-500 to-pink-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-background" />
                
                <CardHeader className="relative">
                  <CardTitle>Top Weighted Components</CardTitle>
                  <CardDescription>
                    Components with highest importance scores
                  </CardDescription>
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
                            <th className="p-3 text-right text-sm font-medium">Weight</th>
                          </tr>
                        </thead>
                        <tbody>
                          {stats.top_components.map((component, index) => (
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
                                  {component.weight.toFixed(4)}
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
