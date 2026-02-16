"use client"

import { useRouter } from "next/navigation"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import {
  Waypoints,
  Database,
  Activity,
  TrendingUp,
  Settings,
  Sparkles,
  Zap,
  BarChart3,
  ArrowRight,
  Info,
  CheckCircle2
} from "lucide-react"
import { useConnection } from "@/lib/stores/connection-store"

// Helper function to format keys (snake_case to Title Case)
function formatKey(key: string): string {
  return key
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(' ')
}

export default function DashboardPage() {
  const router = useRouter()
  const { status, stats, config, initialLoadComplete } = useConnection()

  const isConnected = status === 'connected'

  // Loading State - show when connecting or when initial load hasn't completed
  if (!initialLoadComplete || status === 'connecting' || (isConnected && !stats)) {
    return (
      <AppLayout title="Dashboard" description="System overview and analytics">
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" text={status === 'connecting' ? "Connecting to database..." : "Loading dashboard data..."} />
        </div>
      </AppLayout>
    )
  }

  // Disconnected State - show the no connection banner but keep the normal layout
  const showNoConnection = !isConnected

  return (
    <AppLayout title="Dashboard" description="System overview and analytics">
      <div className="space-y-8">

        {/* Hero Banner with Gradient Background */}
        <div className="relative overflow-hidden border-0 min-h-[450px] flex items-center justify-center">
          {/* Animated gradient background */}
          <div className="absolute inset-0 bg-gradient-to-b from-purple-100 via-blue-100 to-indigo-100 dark:from-[#1a0b2e] dark:via-[#2d1b4e] dark:to-[#0f0520]" />
          
          {/* Glowing orbs */}
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-300/20 dark:bg-purple-600/30 rounded-full blur-3xl animate-pulse" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-300/15 dark:bg-blue-600/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
          
          {/* Bottom fade transition */}
          <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-b from-transparent via-background/50 to-background" />
          
          <div className="relative z-10 text-center px-6 md:px-12 py-12">
            <div className="max-w-5xl mx-auto">
              {/* Badge */}
              <div className="inline-flex items-center gap-2 mb-6">
                <Badge className="bg-purple-200/50 dark:bg-purple-500/20 text-purple-700 dark:text-purple-200 border-purple-300 dark:border-purple-400/30 hover:bg-purple-200/70 dark:hover:bg-purple-500/30 px-4 py-2 text-sm backdrop-blur-sm">
                  <Sparkles className="h-3.5 w-3.5 mr-1.5" />
                  Powered by NetworkX + Neo4j
                </Badge>
              </div>
              
              {/* Main Title */}
              <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold mb-5 leading-tight text-gray-900 dark:text-white">
                The Distributed System<br/>
                <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent">
                  Graph Analytics Framework
                </span>
              </h1>
              
              {/* Subtitle */}
              <p className="text-lg md:text-xl text-gray-700 dark:text-gray-300 mb-8 max-w-2xl mx-auto leading-relaxed">
                The fastest way to add reliability analysis and quality metrics to your distributed system architecture
              </p>
              
              {/* CTA Buttons */}
              <div className="flex flex-wrap gap-4 justify-center mb-10">
                <Button
                  size="lg"
                  className="bg-purple-600 hover:bg-purple-700 text-white dark:bg-white dark:text-purple-600 dark:hover:bg-gray-100 shadow-xl hover:shadow-2xl font-semibold px-8 py-6 text-lg rounded-full"
                  onClick={() => router.push('/analysis')}
                >
                  Get started
                </Button>
                <Button
                  size="lg"
                  variant="outline"
                  className="bg-white/10 dark:bg-transparent text-gray-900 dark:text-white border-2 border-gray-900/30 dark:border-white/30 hover:bg-white/20 dark:hover:bg-white/10 backdrop-blur-sm shadow-lg font-semibold px-8 py-6 text-lg rounded-full"
                  onClick={() => router.push('/tutorial')}
                >
                  Documentation
                </Button>
              </div>
              
              {/* Technology badges/logos */}
              <div className="flex flex-wrap items-center justify-center gap-8 text-gray-500 dark:text-gray-400 text-sm">
                <div className="flex items-center gap-2 opacity-70 hover:opacity-100 transition-opacity">
                  <Database className="h-5 w-5" />
                  <span>Neo4j</span>
                </div>
                <div className="flex items-center gap-2 opacity-70 hover:opacity-100 transition-opacity">
                  <Waypoints className="h-5 w-5" />
                  <span>NetworkX</span>
                </div>
                <div className="flex items-center gap-2 opacity-70 hover:opacity-100 transition-opacity">
                  <Zap className="h-5 w-5" />
                  <span>FastAPI</span>
                </div>
                <div className="flex items-center gap-2 opacity-70 hover:opacity-100 transition-opacity">
                  <BarChart3 className="h-5 w-5" />
                  <span>BoxPlot</span>
                </div>
                <div className="flex items-center gap-2 opacity-70 hover:opacity-100 transition-opacity">
                  <Activity className="h-5 w-5" />
                  <span>React</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* No Connection Info Banner - Show if not connected */}
        {showNoConnection && <NoConnectionInfo />}

        {/* System Metrics Section */}
        <div className="space-y-6">
          <div className="text-center">
            <h2 className="text-2xl md:text-3xl font-bold mb-2">
              <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent">
                System Metrics
              </span>
            </h2>
            <p className="text-muted-foreground">Real-time overview of your distributed system</p>
          </div>
          
          <div className="grid gap-4 md:grid-cols-4 lg:grid-cols-5">
          {/* Total Components Card */}
          <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-blue-500/25 transition-all duration-300 transform hover:scale-[1.02]">
            {/* Gradient border */}
            <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500">
              <div className="w-full h-full bg-background rounded-lg" />
            </div>
            
            {/* Background gradient overlay */}
            <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-blue-500/30 via-blue-500/15 to-blue-500/5" />
            
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3 relative">
                <CardTitle className="text-sm font-medium">Total Components</CardTitle>
                <div className="rounded-xl bg-blue-500/10 p-2.5">
                  <Waypoints className="h-4 w-4 text-blue-500" />
                </div>
              </CardHeader>
              <CardContent className="relative">
                <div className="text-3xl font-bold text-blue-500">{stats?.total_nodes?.toLocaleString() || 0}</div>
                <p className="text-xs text-muted-foreground mt-1">
                  System components in graph
                </p>
              </CardContent>
          </Card>

          {/* Total Edges Card */}
          <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-purple-500/25 transition-all duration-300 transform hover:scale-[1.02]">
            {/* Gradient border */}
            <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500">
              <div className="w-full h-full bg-background rounded-lg" />
            </div>
            
            {/* Background gradient overlay */}
            <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-purple-500/30 via-purple-500/15 to-purple-500/5" />
            
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3 relative">
                <CardTitle className="text-sm font-medium">Total Edges</CardTitle>
                <div className="rounded-xl bg-purple-500/10 p-2.5">
                  <Database className="h-4 w-4 text-purple-500" />
                </div>
              </CardHeader>
              <CardContent className="relative">
                <div className="text-3xl font-bold text-purple-500">
                  {((stats?.total_edges || 0) + (stats?.total_structural_edges || 0)).toLocaleString()}
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  Combined derived & structural edges
                </p>
              </CardContent>
          </Card>

          {/* Component Types Card */}
          <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-green-500/25 transition-all duration-300 transform hover:scale-[1.02]">
            {/* Gradient border */}
            <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500">
              <div className="w-full h-full bg-background rounded-lg" />
            </div>
            
            {/* Background gradient overlay */}
            <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-green-500/30 via-green-500/15 to-green-500/5" />
            
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3 relative">
                <CardTitle className="text-sm font-medium">Component Types</CardTitle>
                <div className="rounded-xl bg-green-500/10 p-2.5">
                  <TrendingUp className="h-4 w-4 text-green-500" />
                </div>
              </CardHeader>
              <CardContent className="relative">
                <div className="text-3xl font-bold text-green-500">
                  {stats?.node_counts ? Object.keys(stats.node_counts).length : 0}
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  Unique component types
                </p>
              </CardContent>
          </Card>

          {/* Edge Types Card */}
          <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-cyan-500/25 transition-all duration-300 transform hover:scale-[1.02]">
            {/* Gradient border */}
            <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-cyan-500 via-teal-500 to-emerald-500">
              <div className="w-full h-full bg-background rounded-lg" />
            </div>
            
            {/* Background gradient overlay */}
            <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-cyan-500/30 via-cyan-500/15 to-cyan-500/5" />
            
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3 relative">
                <CardTitle className="text-sm font-medium">Edge Types</CardTitle>
                <div className="rounded-xl bg-cyan-500/10 p-2.5">
                  <Database className="h-4 w-4 text-cyan-500" />
                </div>
              </CardHeader>
              <CardContent className="relative">
                <div className="text-3xl font-bold text-cyan-500">
                  {((stats?.edge_counts ? Object.keys(stats.edge_counts).length : 0) + 
                    (stats?.structural_edge_counts ? Object.keys(stats.structural_edge_counts).length : 0))}
                </div>
                <p className="text-xs text-muted-foreground mt-1">
                  Unique edge types
                </p>
              </CardContent>
          </Card>

          {/* System Health Card */}
          <Card 
            className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-orange-500/25 transition-all duration-300 transform hover:scale-[1.02] cursor-pointer group"
            onClick={() => router.push('/analysis')}
          >
            {/* Gradient border */}
            <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-orange-500 via-amber-500 to-yellow-500">
              <div className="w-full h-full bg-background rounded-lg" />
            </div>
            
            {/* Background gradient overlay */}
            <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-orange-500/30 via-orange-500/15 to-orange-500/5" />
            
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3 relative">
                <CardTitle className="text-sm font-medium">System Health</CardTitle>
                <div className="rounded-xl bg-orange-500/10 p-2.5 group-hover:scale-110 transition-transform">
                  <Activity className="h-4 w-4 text-orange-500" />
                </div>
              </CardHeader>
              <CardContent className="relative">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex-1">
                    <div className="text-3xl font-bold text-slate-400 dark:text-slate-600 group-hover:text-slate-500 dark:group-hover:text-slate-500 transition-colors">
                      --
                    </div>
                    <p className="text-xs text-muted-foreground mt-1 group-hover:text-orange-500 transition-colors">
                      Click to analyze
                    </p>
                  </div>
                  <Button
                    size="sm"
                    className="bg-orange-500/10 text-orange-500 hover:bg-orange-500/20 border border-orange-500/30 shrink-0 shadow-md hover:shadow-lg transition-all group-hover:scale-105"
                    onClick={(e) => {
                      e.stopPropagation()
                      router.push('/analysis')
                    }}
                  >
                    <Activity className="mr-1.5 h-3.5 w-3.5" />
                    Analyze
                  </Button>
                </div>
              </CardContent>
          </Card>
        </div>
        </div>

        {/* Node Distribution */}
        {stats?.node_counts && Object.keys(stats.node_counts).length > 0 ? (
          <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-blue-500/20 transition-all duration-300">
            {/* Gradient border */}
            <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-400 via-indigo-500 to-purple-600">
              <div className="w-full h-full bg-background rounded-lg" />
            </div>
            
            {/* Background gradient overlay */}
            <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-blue-500/35 via-blue-500/20 to-blue-500/5" />
              <CardHeader className="pb-4 relative">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="rounded-xl bg-blue-500/10 p-2.5">
                      <Waypoints className="h-5 w-5 text-blue-500" />
                    </div>
                    <div>
                      <CardTitle className="text-lg font-semibold">Component Distribution</CardTitle>
                      <CardDescription>Breakdown of components by type</CardDescription>
                    </div>
                  </div>
                  <Badge variant="secondary" className="text-sm px-3 py-1 bg-blue-500/10 text-blue-500 border-blue-500/20">
                    {Object.keys(stats.node_counts).length} types
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="relative">
                <div className="flex flex-wrap gap-5">
                  {Object.entries(stats.node_counts).map(([type, count]) => (
                    <div key={type} className="space-y-3 p-4 rounded-lg border border-blue-500/20 bg-blue-500/5 hover:bg-blue-500/10 hover:shadow-sm transition-all flex-1 min-w-[200px]">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-semibold">{type}</span>
                        <Badge variant="secondary" className="bg-blue-500/10 text-blue-500 px-2.5 py-0.5 border-blue-500/20">
                          {count.toLocaleString()}
                        </Badge>
                      </div>
                      <Progress
                        value={(count / stats.total_nodes) * 100}
                        className="h-2.5 bg-slate-200 dark:bg-slate-700"
                      />
                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span>{((count / stats.total_nodes) * 100).toFixed(1)}%</span>
                        <span>of {stats.total_nodes.toLocaleString()} total</span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
          </Card>
        ) : (
          <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-slate-500/20 transition-all duration-300">
            {/* Gradient border */}
            <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-slate-200 via-slate-300 to-slate-200 dark:from-slate-700 dark:via-slate-800 dark:to-slate-700">
              <div className="w-full h-full bg-background rounded-lg" />
            </div>
            
            {/* Background gradient overlay */}
            <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-slate-500/15 via-slate-500/8 to-transparent" />
            
            <CardHeader className="relative">
                <div className="flex items-center gap-2">
                  <div className="rounded-full bg-slate-500/10 p-2">
                    <Waypoints className="h-4 w-4 text-slate-500" />
                  </div>
                  <div>
                    <CardTitle>Component Distribution</CardTitle>
                    <CardDescription>No components in database yet</CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4 relative">
                <p className="text-sm text-muted-foreground">
                  Your database is empty. Import or generate graph data to see component distribution and analytics.
                </p>
                <Button onClick={() => router.push('/data')} className="w-full" variant="outline">
                  <Database className="mr-2 h-4 w-4" />
                  Import Data
                </Button>
              </CardContent>
          </Card>
        )}

        {/* Edge Distribution */}
        {stats?.edge_counts && Object.keys(stats.edge_counts).length > 0 ? (
          <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-purple-500/20 transition-all duration-300">
            {/* Gradient border */}
            <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-purple-400 via-fuchsia-500 to-pink-600">
              <div className="w-full h-full bg-background rounded-lg" />
            </div>
            
            {/* Background gradient overlay */}
            <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-purple-500/35 via-purple-500/20 to-purple-500/5" />
              <CardHeader className="pb-4 relative">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="rounded-xl bg-purple-500/10 p-2.5">
                      <Database className="h-5 w-5 text-purple-500" />
                    </div>
                    <div>
                      <CardTitle className="text-lg font-semibold">Dependency Distribution</CardTitle>
                      <CardDescription>How components depend on each other across system layers</CardDescription>
                    </div>
                  </div>
                  <Badge variant="secondary" className="text-sm px-3 py-1 bg-purple-500/10 text-purple-500 border-purple-500/20">
                    {Object.keys(stats.edge_counts).length} types
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="relative">
                <div className="flex flex-wrap gap-5">
                  {Object.entries(stats.edge_counts).map(([type, count]) => (
                    <div key={type} className="space-y-3 p-4 rounded-lg border border-purple-500/20 bg-purple-500/5 hover:bg-purple-500/10 hover:shadow-sm transition-all flex-1 min-w-[200px]">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-semibold">
                          {formatKey(type)}
                        </span>
                        <Badge variant="outline" className="border-purple-500/30 text-purple-500 px-2.5 py-0.5 bg-purple-500/5">
                          {count.toLocaleString()}
                        </Badge>
                      </div>
                      <Progress
                        value={(count / (stats.total_dependency_edges || 1)) * 100}
                        className="h-2.5 bg-slate-200 dark:bg-slate-700"
                      />
                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span>{((count / (stats.total_dependency_edges || 1)) * 100).toFixed(1)}%</span>
                        <span>of {(stats.total_dependency_edges || 0).toLocaleString()} total</span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
          </Card>
        ) : (
          <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-slate-500/20 transition-all duration-300">
            {/* Gradient border */}
            <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-slate-200 via-slate-300 to-slate-200 dark:from-slate-700 dark:via-slate-800 dark:to-slate-700">
              <div className="w-full h-full bg-background rounded-lg" />
            </div>
            
            {/* Background gradient overlay */}
            <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-slate-500/15 via-slate-500/8 to-transparent" />
            
            <CardHeader className="relative">
                <div className="flex items-center gap-2">
                  <div className="rounded-full bg-slate-500/10 p-2">
                    <Database className="h-4 w-4 text-slate-500" />
                  </div>
                  <div>
                    <CardTitle>Edge Distribution</CardTitle>
                    <CardDescription>No relationships in database yet</CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4 relative">
                <p className="text-sm text-muted-foreground">
                  Your database has no relationship data. Add graph data to analyze dependencies and connections.
                </p>
                <Button onClick={() => router.push('/data')} className="w-full" variant="outline">
                  <Database className="mr-2 h-4 w-4" />
                  Populate Database
                </Button>
              </CardContent>
          </Card>
        )}

        {/* Structural Relationships Distribution */}
        {stats?.structural_edge_counts && Object.keys(stats.structural_edge_counts).length > 0 ? (
          <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-indigo-500/20 transition-all duration-300">
            {/* Gradient border */}
            <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-indigo-400 via-blue-500 to-cyan-600">
              <div className="w-full h-full bg-background rounded-lg" />
            </div>
            
            {/* Background gradient overlay */}
            <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-indigo-500/35 via-indigo-500/20 to-indigo-500/5" />
              <CardHeader className="pb-4 relative">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="rounded-xl bg-indigo-500/10 p-2.5">
                      <Waypoints className="h-5 w-5 text-indigo-500" />
                    </div>
                    <div>
                      <CardTitle className="text-lg font-semibold">Structural Relationships</CardTitle>
                      <CardDescription>Physical topology and communication patterns between components</CardDescription>
                    </div>
                  </div>
                  <Badge variant="secondary" className="text-sm px-3 py-1 bg-indigo-500/10 text-indigo-500 border-indigo-500/20">
                    {Object.keys(stats.structural_edge_counts).length} types
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="relative">
                <div className="flex flex-wrap gap-5">
                  {Object.entries(stats.structural_edge_counts).map(([type, count]) => (
                    <div key={type} className="space-y-3 p-4 rounded-lg border border-indigo-500/20 bg-indigo-500/5 hover:bg-indigo-500/10 hover:shadow-sm transition-all flex-1 min-w-[200px]">
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-semibold">
                          {formatKey(type)}
                        </span>
                        <Badge variant="outline" className="border-indigo-500/30 text-indigo-500 px-2.5 py-0.5 bg-indigo-500/5">
                          {count.toLocaleString()}
                        </Badge>
                      </div>
                      <Progress
                        value={(count / (stats.total_structural_edges || 1)) * 100}
                        className="h-2.5 bg-slate-200 dark:bg-slate-700"
                      />
                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span>{((count / (stats.total_structural_edges || 1)) * 100).toFixed(1)}%</span>
                        <span>of {(stats.total_structural_edges || 0).toLocaleString()} total</span>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
          </Card>
        ) : null}

        {/* Spot Issues Faster Section */}
        <div className="relative overflow-hidden border-0">
          <div className="absolute inset-0 bg-gradient-to-b from-purple-100 via-blue-100 to-indigo-100 dark:from-[#1a0b2e] dark:via-[#2d1b4e] dark:to-[#0f0520]" />
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_20%,rgba(139,92,246,0.15),transparent_50%)] dark:bg-[radial-gradient(circle_at_30%_20%,rgba(139,92,246,0.2),transparent_50%)]" />
          
          {/* Bottom fade transition */}
          <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-b from-transparent via-background/50 to-background" />
          
          <div className="relative p-10 md:p-12 text-gray-900 dark:text-white">
            <div className="text-center mb-10">
              <h2 className="text-3xl md:text-4xl font-bold mb-4">
                <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent">
                  Spot issues faster
                </span>
              </h2>
              <p className="text-lg md:text-xl text-gray-700 dark:text-gray-300 max-w-2xl mx-auto">
                Identify bottlenecks, single points of failure, and architectural antipatterns before they impact your system
              </p>
            </div>
            
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4 max-w-6xl mx-auto">
              <div className="space-y-3 bg-white/50 dark:bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-gray-200 dark:border-white/10 hover:bg-white/70 dark:hover:bg-white/10 transition-colors">
                <div className="rounded-lg bg-purple-500/20 p-3 w-fit">
                  <Activity className="h-6 w-6 text-purple-500 dark:text-purple-300" />
                </div>
                <h4 className="font-semibold text-lg text-gray-900 dark:text-white">Centrality Analysis</h4>
                <p className="text-sm text-gray-600 dark:text-white/70">
                  Identify critical components using betweenness, closeness, and degree centrality
                </p>
              </div>
              
              <div className="space-y-3 bg-white/50 dark:bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-gray-200 dark:border-white/10 hover:bg-white/70 dark:hover:bg-white/10 transition-colors">
                <div className="rounded-lg bg-blue-500/20 p-3 w-fit">
                  <BarChart3 className="h-6 w-6 text-blue-500 dark:text-blue-300" />
                </div>
                <h4 className="font-semibold text-lg text-gray-900 dark:text-white">Quality Metrics</h4>
                <p className="text-sm text-gray-600 dark:text-white/70">
                  Evaluate reliability, maintainability, and availability with fuzzy logic scoring
                </p>
              </div>
              
              <div className="space-y-3 bg-white/50 dark:bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-gray-200 dark:border-white/10 hover:bg-white/70 dark:hover:bg-white/10 transition-colors">
                <div className="rounded-lg bg-green-500/20 p-3 w-fit">
                  <TrendingUp className="h-6 w-6 text-green-500 dark:text-green-300" />
                </div>
                <h4 className="font-semibold text-lg text-gray-900 dark:text-white">Statistical Analysis</h4>
                <p className="text-sm text-gray-600 dark:text-white/70">
                  Classify components with BoxPlot statistical methods for outlier detection
                </p>
              </div>
              
              <div className="space-y-3 bg-white/50 dark:bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-gray-200 dark:border-white/10 hover:bg-white/70 dark:hover:bg-white/10 transition-colors">
                <div className="rounded-lg bg-orange-500/20 p-3 w-fit">
                  <Zap className="h-6 w-6 text-orange-500 dark:text-orange-300" />
                </div>
                <h4 className="font-semibold text-lg text-gray-900 dark:text-white">Path Analysis</h4>
                <p className="text-sm text-gray-600 dark:text-white/70">
                  Discover shortest paths and identify communication bottlenecks
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Tool Features Grid */}
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {/* Quality Analysis Card */}
            <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-blue-500/20 transition-all duration-300 transform hover:scale-[1.02]">
              {/* Gradient border */}
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-400 via-indigo-500 to-violet-600">
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              
              {/* Background gradient overlay */}
              <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-blue-500/35 via-blue-500/20 to-blue-500/5" />
                <CardHeader className="pb-3 relative">
                  <div className="rounded-xl bg-blue-500/10 p-3 w-fit mb-3">
                    <BarChart3 className="h-6 w-6 text-blue-500" />
                  </div>
                  <CardTitle className="text-lg font-semibold">Quality Analysis</CardTitle>
                  <CardDescription className="text-sm">
                    Full system analysis with reliability, maintainability, and availability metrics. Analyze by component type or architectural layer.
                  </CardDescription>
                </CardHeader>
                <CardContent className="relative">
                  <Button
                    variant="outline"
                    className="w-full hover:bg-blue-500/10 hover:border-blue-500/30 transition-colors border-blue-500/20"
                    onClick={() => router.push('/analysis')}
                  >
                    View Analysis
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </CardContent>
            </Card>

            {/* Visualization Card */}
            <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-green-500/20 transition-all duration-300 transform hover:scale-[1.02]">
              {/* Gradient border */}
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-green-400 via-emerald-500 to-teal-600">
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              
              {/* Background gradient overlay */}
              <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-green-500/35 via-green-500/20 to-green-500/5" />
                <CardHeader className="pb-3 relative">
                  <div className="rounded-xl bg-green-500/10 p-3 w-fit mb-3">
                    <Waypoints className="h-6 w-6 text-green-500" />
                  </div>
                  <CardTitle className="text-lg font-semibold">Visualization</CardTitle>
                  <CardDescription className="text-sm">
                    Explore your system architecture with interactive 2D/3D force-directed graphs. Filter by type and relationship.
                  </CardDescription>
                </CardHeader>
                <CardContent className="relative">
                  <Button
                    variant="outline"
                    className="w-full hover:bg-green-500/10 hover:border-green-500/30 transition-colors border-green-500/20"
                    onClick={() => router.push('/explorer')}
                  >
                    Explore Graph
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </CardContent>
            </Card>

            {/* Data Management Card */}
            <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-orange-500/20 transition-all duration-300 transform hover:scale-[1.02]">
              {/* Gradient border */}
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-orange-400 via-amber-500 to-yellow-600">
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              
              {/* Background gradient overlay */}
              <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-orange-500/35 via-orange-500/20 to-orange-500/5" />
                <CardHeader className="pb-3 relative">
                  <div className="rounded-xl bg-orange-500/10 p-3 w-fit mb-3">
                    <Database className="h-6 w-6 text-orange-500" />
                  </div>
                  <CardTitle className="text-lg font-semibold">Data Management</CardTitle>
                  <CardDescription className="text-sm">
                    Generate synthetic graphs at any scale, import from files, export graph data, or clear the database.
                  </CardDescription>
                </CardHeader>
                <CardContent className="relative">
                  <Button
                    variant="outline"
                    className="w-full hover:bg-orange-500/10 hover:border-orange-500/30 transition-colors border-orange-500/20"
                    onClick={() => router.push('/data')}
                  >
                    Manage Data
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </CardContent>
            </Card>
        </div>

        {/* Getting Started Card */}
        <div className="relative overflow-hidden border-0">
          <div className="absolute inset-0 bg-gradient-to-br from-purple-50 via-blue-50 to-indigo-50 dark:from-[#1a0b2e] dark:via-[#2d1b4e] dark:to-[#0f0520]" />
          <div className="absolute inset-0 dark:bg-[radial-gradient(circle_at_30%_20%,rgba(147,51,234,0.2),transparent_50%)]" />
          
          <div className="relative text-center py-12 px-6">
            <div className="inline-flex items-center justify-center rounded-2xl bg-gradient-to-r from-purple-500 to-blue-600 p-3 mb-4">
              <Sparkles className="h-8 w-8 text-white" />
            </div>
            <h2 className="text-3xl md:text-4xl font-bold mb-3">
              <span className="bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent">
                Get Started in Minutes
              </span>
            </h2>
            <p className="text-lg text-gray-600 dark:text-gray-300 mb-10 max-w-2xl mx-auto">
              Four simple steps to comprehensive system analysis
            </p>
            
            <div className="grid md:grid-cols-4 gap-6 mb-10 max-w-6xl mx-auto">
              <div className="text-center space-y-3 p-6 rounded-2xl bg-white/50 dark:bg-white/5 backdrop-blur-sm border border-blue-200/50 dark:border-blue-500/20 hover:border-blue-500/40 transition-all hover:shadow-lg">
                <div className="inline-flex items-center justify-center rounded-full bg-blue-500/10 p-4 mb-2">
                  <Database className="h-8 w-8 text-blue-500" />
                </div>
                <h4 className="text-xl font-semibold text-gray-900 dark:text-white">1. Import Data</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Load your system data or generate synthetic graphs for testing
                </p>
              </div>
              
              <div className="text-center space-y-3 p-6 rounded-2xl bg-white/50 dark:bg-white/5 backdrop-blur-sm border border-purple-200/50 dark:border-purple-500/20 hover:border-purple-500/40 transition-all hover:shadow-lg">
                <div className="inline-flex items-center justify-center rounded-full bg-purple-500/10 p-4 mb-2">
                  <Activity className="h-8 w-8 text-purple-500" />
                </div>
                <h4 className="text-xl font-semibold text-gray-900 dark:text-white">2. Analyze</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Run comprehensive quality and centrality analysis on your system
                </p>
              </div>
              
              <div className="text-center space-y-3 p-6 rounded-2xl bg-white/50 dark:bg-white/5 backdrop-blur-sm border border-orange-200/50 dark:border-orange-500/20 hover:border-orange-500/40 transition-all hover:shadow-lg">
                <div className="inline-flex items-center justify-center rounded-full bg-orange-500/10 p-4 mb-2">
                  <Zap className="h-8 w-8 text-orange-500" />
                </div>
                <h4 className="text-xl font-semibold text-gray-900 dark:text-white">3. Simulate</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Test failure scenarios and analyze system resilience patterns
                </p>
              </div>
              
              <div className="text-center space-y-3 p-6 rounded-2xl bg-white/50 dark:bg-white/5 backdrop-blur-sm border border-green-200/50 dark:border-green-500/20 hover:border-green-500/40 transition-all hover:shadow-lg">
                <div className="inline-flex items-center justify-center rounded-full bg-green-500/10 p-4 mb-2">
                  <Waypoints className="h-8 w-8 text-green-500" />
                </div>
                <h4 className="text-xl font-semibold text-gray-900 dark:text-white">4. Visualize</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Explore your architecture with interactive 3D graph visualization
                </p>
              </div>
            </div>
            
            <div className="flex flex-col sm:flex-row gap-4 max-w-2xl mx-auto">
              <Button 
                size="lg"
                onClick={() => router.push('/data')} 
                className="flex-1 bg-white text-purple-600 hover:bg-gray-100 dark:bg-white dark:hover:bg-gray-100 shadow-lg hover:shadow-xl font-semibold rounded-full"
              >
                Get Started Now
              </Button>
              <Button 
                size="lg"
                onClick={() => router.push('/tutorial')} 
                variant="outline" 
                className="flex-1 border-2 border-purple-500/30 dark:border-white/30 hover:bg-purple-500/10 dark:hover:bg-white/10 font-semibold rounded-full"
              >
                View Tutorial
              </Button>
            </div>
          </div>

          {/* Bottom fade transition */}
          <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-b from-transparent via-background/50 to-background" />
        </div>
      </div>
    </AppLayout>
  )
}
