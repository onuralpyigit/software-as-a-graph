"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ArrowLeft, Zap, AlertTriangle, CheckCircle, Info, TrendingUp, Activity, MessageSquare } from "lucide-react"
import { useConnection } from "@/lib/stores/connection-store"
import { apiClient } from "@/lib/api/client"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

interface Topic {
  id: string
  name: string
  publishers: number
  subscribers: number
  total_activity: number
  brokers: number
  pub_weight: number
  sub_weight: number
}

interface Broker {
  id: string
  name: string
  topics: number
  publishers: number
  subscribers: number
  total_load: number
}

interface Application {
  id: string
  name: string
  topics?: number
  subscriptions?: number
  publications?: number
}

interface MessageFlowStats {
  total_topics: number
  total_brokers: number
  total_applications: number
  active_applications: number
  avg_publishers_per_topic: number
  avg_subscribers_per_topic: number
  avg_topics_per_broker: number
  interpretation: string
  category: string
  health: string
  hot_topics: Topic[]
  broker_utilization: Broker[]
  isolated_applications: Application[]
  top_publishers: Application[]
  top_subscribers: Application[]
}

// Helper function to safely extract string from potentially complex data
const safeString = (value: any): string => {
  if (typeof value === 'string') return value
  if (typeof value === 'number') return String(value)
  if (value && typeof value === 'object') {
    // If it's an object with name or id, extract that
    return value.name || value.id || JSON.stringify(value)
  }
  return String(value || '')
}

export default function MessageFlowPatternsPage() {
  const router = useRouter()
  const { status, stats: graphStats, initialLoadComplete } = useConnection()
  const [stats, setStats] = useState<MessageFlowStats | null>(null)
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
      const result = await apiClient.getMessageFlowPatternsStats()
      
      // Ensure all nested objects have proper string values
      if (result?.stats) {
        setStats(result.stats)
      } else if (result?.total_topics !== undefined) {
        // Handle case where stats are at root level
        setStats(result)
      } else {
        setError('Invalid response format from server')
      }
    } catch (err) {
      console.error('Failed to load stats:', err)
      setError(err instanceof Error ? err.message : 'Failed to load statistics')
    } finally {
      setLoading(false)
    }
  }

  // Get health interpretation color and icon
  const getHealthInfo = () => {
    if (!stats) return { color: 'text-gray-500', icon: Info, label: 'Unknown' }
    
    switch (stats.health) {
      case 'good':
        return { 
          color: 'text-green-500', 
          icon: CheckCircle, 
          label: 'Healthy Communication'
        }
      case 'fair':
        return { 
          color: 'text-blue-500', 
          icon: Info, 
          label: 'Fair Communication'
        }
      case 'moderate':
        return { 
          color: 'text-yellow-500', 
          icon: AlertTriangle, 
          label: 'Moderate Activity'
        }
      case 'poor':
        return { 
          color: 'text-red-500', 
          icon: AlertTriangle, 
          label: 'Communication Issues'
        }
      default:
        return { 
          color: 'text-gray-500', 
          icon: Info, 
          label: 'Unknown'
        }
    }
  }

  const healthInfo = getHealthInfo()
  const HealthIcon = healthInfo.icon

  // Get category color
  const getCategoryColor = () => {
    if (!stats) return 'text-gray-500'
    switch (stats.category) {
      case 'balanced':
        return 'text-green-500'
      case 'bottleneck':
        return 'text-orange-500'
      case 'sparse':
        return 'text-blue-500'
      case 'isolated':
        return 'text-red-500'
      default:
        return 'text-gray-500'
    }
  }

  // Loading State
  if (!initialLoadComplete || status === 'connecting') {
    return (
      <AppLayout title="Message Flow Patterns" description="Analyze pub-sub communication patterns">
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" text="Connecting to database..." />
        </div>
      </AppLayout>
    )
  }

  // Disconnected State
  if (!isConnected) {
    return (
      <AppLayout title="Message Flow Patterns" description="Analyze pub-sub communication patterns">
        <NoConnectionInfo />
      </AppLayout>
    )
  }

  return (
    <AppLayout title="Message Flow Patterns" description="Analyze pub-sub communication patterns">
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
          <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-yellow-500 via-orange-500 to-red-500">
            <div className="w-full h-full rounded-lg bg-gradient-to-r from-yellow-600 via-orange-600 to-red-600" />
          </div>
          
          <CardContent className="p-8 relative text-white">
            <div className="flex items-center gap-4 mb-4">
              <div className="rounded-xl bg-white/20 p-3">
                <Zap className="h-8 w-8" />
              </div>
              <div className="flex-1">
                <h3 className="text-3xl font-bold">Message Flow Pattern Analysis</h3>
                <p className="text-white/90 mt-2">
                  Understanding pub-sub communication patterns, bottlenecks, and message flow
                </p>
              </div>
            </div>
            
            <div className="bg-white/10 rounded-lg p-4 mt-4">
              <div className="flex items-start gap-3">
                <Info className="h-5 w-5 mt-0.5 flex-shrink-0" />
                <div className="text-sm">
                  <p className="font-semibold mb-1">What are Message Flow Patterns?</p>
                  <p className="text-white/90">
                    Message flow pattern analysis examines <strong>pub-sub communication</strong> to identify 
                    <strong> hot topics</strong> (highly active message channels), 
                    <strong> broker utilization</strong> (load distribution across message brokers), and 
                    <strong> communication bottlenecks</strong>. It also identifies isolated applications 
                    that don't participate in message flow.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Loading/Error States */}
        {loading && (
          <div className="flex items-center justify-center py-12">
            <LoadingSpinner size="lg" text="Analyzing message flow patterns..." />
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
            {/* Overview Cards */}
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              {/* Total Topics */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-yellow-500/25 transition-all duration-300">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-yellow-500 to-orange-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-yellow-500/30 via-yellow-500/15 to-yellow-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Topics</CardTitle>
                  <div className="rounded-xl bg-yellow-500/10 p-2.5">
                    <MessageSquare className="h-4 w-4 text-yellow-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-yellow-500">
                    {stats.total_topics}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Active message channels
                  </p>
                </CardContent>
              </Card>

              {/* Active Applications */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-green-500/25 transition-all duration-300">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-green-500 to-emerald-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-green-500/30 via-green-500/15 to-green-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Active Apps</CardTitle>
                  <div className="rounded-xl bg-green-500/10 p-2.5">
                    <Activity className="h-4 w-4 text-green-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-green-500">
                    {stats.active_applications}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Out of {stats.total_applications} total
                  </p>
                </CardContent>
              </Card>

              {/* Isolated Applications */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-red-500/25 transition-all duration-300">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-red-500 to-rose-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-red-500/30 via-red-500/15 to-red-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Isolated Apps</CardTitle>
                  <div className="rounded-xl bg-red-500/10 p-2.5">
                    <AlertTriangle className="h-4 w-4 text-red-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-red-500">
                    {stats.isolated_applications.length}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {((stats.isolated_applications.length / stats.total_applications) * 100).toFixed(1)}% of total
                  </p>
                </CardContent>
              </Card>

              {/* Avg Activity */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-blue-500/25 transition-all duration-300">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 to-indigo-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-blue-500/30 via-blue-500/15 to-blue-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Avg Topic Activity</CardTitle>
                  <div className="rounded-xl bg-blue-500/10 p-2.5">
                    <TrendingUp className="h-4 w-4 text-blue-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-blue-500">
                    {(stats.avg_publishers_per_topic + stats.avg_subscribers_per_topic).toFixed(1)}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Pub+Sub per topic
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Health & Interpretation */}
            <Card className="relative overflow-hidden border-0 shadow-lg">
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500">
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              <div className="absolute inset-[2px] rounded-lg bg-background" />
              
              <CardHeader className="relative">
                <CardTitle className="flex items-center gap-2">
                  <HealthIcon className={`h-5 w-5 ${healthInfo.color}`} />
                  System Health: {healthInfo.label}
                </CardTitle>
                <CardDescription>Overall message flow pattern assessment</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4 relative">
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium">Category:</span>
                  <Badge variant="outline" className={getCategoryColor()}>
                    {stats.category.toUpperCase()}
                  </Badge>
                </div>
                <div className="flex items-start gap-3 p-4 bg-muted rounded-lg">
                  <Info className="h-5 w-5 mt-0.5 flex-shrink-0" />
                  <p className="text-sm">{stats.interpretation}</p>
                </div>
                <div className="grid grid-cols-2 gap-4 pt-2">
                  <div>
                    <p className="text-sm text-muted-foreground">Avg Publishers/Topic</p>
                    <p className="text-2xl font-bold">{stats.avg_publishers_per_topic.toFixed(2)}</p>
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground">Avg Subscribers/Topic</p>
                    <p className="text-2xl font-bold">{stats.avg_subscribers_per_topic.toFixed(2)}</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Hot Topics */}
            {stats.hot_topics && Array.isArray(stats.hot_topics) && stats.hot_topics.length > 0 && (
              <Card className="relative overflow-hidden border-0 shadow-lg">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-yellow-500 to-orange-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-background" />
                
                <CardHeader className="relative">
                  <CardTitle className="flex items-center gap-2">
                    <Zap className="h-5 w-5 text-yellow-500" />
                    Hot Topics
                  </CardTitle>
                  <CardDescription>
                    Most active message channels by publisher and subscriber count
                  </CardDescription>
                </CardHeader>
                <CardContent className="relative">
                  <div className="rounded-md border">
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead>
                          <tr className="border-b bg-muted/50">
                            <th className="p-3 text-left text-sm font-medium">Rank</th>
                            <th className="p-3 text-left text-sm font-medium">Topic</th>
                            <th className="p-3 text-right text-sm font-medium">Publishers</th>
                            <th className="p-3 text-right text-sm font-medium">Subscribers</th>
                            <th className="p-3 text-right text-sm font-medium">Total Activity</th>
                            <th className="p-3 text-right text-sm font-medium">Brokers</th>
                          </tr>
                        </thead>
                        <tbody>
                          {stats.hot_topics.map((topic, index) => (
                            <tr key={safeString(topic.id)} className="border-b last:border-0 hover:bg-muted/30 transition-colors cursor-pointer group" onClick={() => router.push(`/explorer?node=${encodeURIComponent(topic.id)}`)}>
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
                                <div className="font-medium text-sm group-hover:underline transition-all">{safeString(topic.name || topic.id)}</div>
                              </td>
                              <td className="p-3 text-right">
                                <div className="text-sm">{topic.publishers}</div>
                              </td>
                              <td className="p-3 text-right">
                                <div className="text-sm">{topic.subscribers}</div>
                              </td>
                              <td className="p-3 text-right">
                                <div className="font-semibold text-yellow-600 dark:text-yellow-400">
                                  {topic.total_activity}
                                </div>
                              </td>
                              <td className="p-3 text-right">
                                <div className="text-sm">{topic.brokers}</div>
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

            {/* Broker Utilization */}
            {stats.broker_utilization && Array.isArray(stats.broker_utilization) && stats.broker_utilization.length > 0 && (
              <Card className="relative overflow-hidden border-0 shadow-lg">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 to-indigo-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-background" />
                
                <CardHeader className="relative">
                  <CardTitle className="flex items-center gap-2">
                    <Activity className="h-5 w-5 text-blue-500" />
                    Broker Utilization
                  </CardTitle>
                  <CardDescription>
                    Load distribution across message brokers
                  </CardDescription>
                </CardHeader>
                <CardContent className="relative">
                  <div className="rounded-md border">
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead>
                          <tr className="border-b bg-muted/50">
                            <th className="p-3 text-left text-sm font-medium">Rank</th>
                            <th className="p-3 text-left text-sm font-medium">Broker</th>
                            <th className="p-3 text-right text-sm font-medium">Topics</th>
                            <th className="p-3 text-right text-sm font-medium">Publishers</th>
                            <th className="p-3 text-right text-sm font-medium">Subscribers</th>
                            <th className="p-3 text-right text-sm font-medium">Total Load</th>
                          </tr>
                        </thead>
                        <tbody>
                          {stats.broker_utilization.map((broker, index) => (
                            <tr key={safeString(broker.id)} className="border-b last:border-0 hover:bg-muted/30 transition-colors">
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
                                <div className="font-medium text-sm">{safeString(broker.name || broker.id)}</div>
                                <div className="text-xs text-muted-foreground font-mono">{safeString(broker.id)}</div>
                              </td>
                              <td className="p-3 text-right">
                                <div className="text-sm">{broker.topics}</div>
                              </td>
                              <td className="p-3 text-right">
                                <div className="text-sm">{broker.publishers}</div>
                              </td>
                              <td className="p-3 text-right">
                                <div className="text-sm">{broker.subscribers}</div>
                              </td>
                              <td className="p-3 text-right">
                                <div className="font-semibold text-blue-600 dark:text-blue-400">
                                  {broker.total_load}
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

            {/* Top Publishers and Subscribers */}
            <div className="grid gap-4 md:grid-cols-2">
              {/* Top Publishers */}
              {stats.top_publishers && Array.isArray(stats.top_publishers) && stats.top_publishers.length > 0 && (
                <Card className="relative overflow-hidden border-0 shadow-lg">
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-green-500 to-emerald-500">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  <div className="absolute inset-[2px] rounded-lg bg-background" />
                  
                  <CardHeader className="relative">
                    <CardTitle className="flex items-center gap-2">
                      <TrendingUp className="h-5 w-5 text-green-500" />
                      Top Publishers
                    </CardTitle>
                    <CardDescription>Most active message producers</CardDescription>
                  </CardHeader>
                  <CardContent className="relative">
                    <div className="rounded-md border">
                      <div className="overflow-x-auto">
                        <table className="w-full">
                          <thead>
                            <tr className="border-b bg-muted/50">
                              <th className="p-3 text-left text-sm font-medium">Rank</th>
                              <th className="p-3 text-left text-sm font-medium">Application</th>
                              <th className="p-3 text-right text-sm font-medium">Topics</th>
                            </tr>
                          </thead>
                          <tbody>
                            {stats.top_publishers.map((app, index) => (
                              <tr key={safeString(app.id)} className="border-b last:border-0 hover:bg-muted/30 transition-colors cursor-pointer group" onClick={() => router.push(`/explorer?node=${encodeURIComponent(app.id)}`)}>
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
                                  <div className="font-medium text-sm group-hover:underline transition-all">{safeString(app.name || app.id)}</div>
                                  <div className="text-xs text-muted-foreground font-mono">{safeString(app.id)}</div>
                                </td>
                                <td className="p-3 text-right">
                                  <div className="font-semibold text-green-600 dark:text-green-400">
                                    {app.topics}
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

              {/* Top Subscribers */}
              {stats.top_subscribers && Array.isArray(stats.top_subscribers) && stats.top_subscribers.length > 0 && (
                <Card className="relative overflow-hidden border-0 shadow-lg">
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-purple-500 to-fuchsia-500">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>
                  <div className="absolute inset-[2px] rounded-lg bg-background" />
                  
                  <CardHeader className="relative">
                    <CardTitle className="flex items-center gap-2">
                      <Activity className="h-5 w-5 text-purple-500" />
                      Top Subscribers
                    </CardTitle>
                    <CardDescription>Most active message consumers</CardDescription>
                  </CardHeader>
                  <CardContent className="relative">
                    <div className="rounded-md border">
                      <div className="overflow-x-auto">
                        <table className="w-full">
                          <thead>
                            <tr className="border-b bg-muted/50">
                              <th className="p-3 text-left text-sm font-medium">Rank</th>
                              <th className="p-3 text-left text-sm font-medium">Application</th>
                              <th className="p-3 text-right text-sm font-medium">Topics</th>
                            </tr>
                          </thead>
                          <tbody>
                            {stats.top_subscribers.map((app, index) => (
                              <tr key={safeString(app.id)} className="border-b last:border-0 hover:bg-muted/30 transition-colors cursor-pointer group" onClick={() => router.push(`/explorer?node=${encodeURIComponent(app.id)}`)}>
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
                                  <div className="font-medium text-sm group-hover:underline transition-all">{safeString(app.name || app.id)}</div>
                                  <div className="text-xs text-muted-foreground font-mono">{safeString(app.id)}</div>
                                </td>
                                <td className="p-3 text-right">
                                  <div className="font-semibold text-purple-600 dark:text-purple-400">
                                    {app.topics}
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
            </div>

            {/* Isolated Applications */}
            {stats.isolated_applications && Array.isArray(stats.isolated_applications) && stats.isolated_applications.length > 0 && (
              <Card className="relative overflow-hidden border-0 shadow-lg">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-red-500 to-rose-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-background" />
                
                <CardHeader className="relative">
                  <CardTitle className="flex items-center gap-2">
                    <AlertTriangle className="h-5 w-5 text-red-500" />
                    Isolated Applications
                  </CardTitle>
                  <CardDescription>
                    Applications not participating in message flow (no publishing or subscribing)
                  </CardDescription>
                </CardHeader>
                <CardContent className="relative">
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
                    {stats.isolated_applications.map((app) => (
                      <Badge key={safeString(app.id)} variant="outline" className="text-red-500 justify-center py-2">
                        {safeString(app.name || app.id)}
                      </Badge>
                    ))}
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
