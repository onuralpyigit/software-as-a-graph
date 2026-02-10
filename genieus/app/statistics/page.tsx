"use client"

import { useRouter } from "next/navigation"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import {
  TrendingUp,
  BarChart3,
  Network,
  Gauge,
  GitCompare,
  Activity,
  ArrowRight,
  Waypoints,
  Zap,
  Box,
  Tags,
  Layers,
  Shield,
} from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { useConnection } from "@/lib/stores/connection-store"
import { PieChart, Pie, Cell } from 'recharts'
import { ChartContainer, ChartTooltip, ChartTooltipContent, ChartLegend, ChartLegendContent } from "@/components/ui/chart"

// Statistics option cards configuration
const statisticsOptions = [
  {
    id: 'centrality',
    title: 'Centrality Statistics',
    description: 'Degree, betweenness, closeness, and eigenvector centrality distributions and rankings',
    icon: Network,
    gradient: 'from-blue-500 via-indigo-500 to-purple-500',
    bgGradient: 'from-blue-500/35 via-blue-500/20 to-blue-500/5',
    iconBg: 'bg-blue-500/10',
    iconColor: 'text-blue-500',
    shadowColor: 'hover:shadow-blue-500/25',
    href: '/statistics/centrality'
  },
  {
    id: 'quality',
    title: 'Quality Metrics',
    description: 'Reliability, maintainability, availability scores and distributions across components',
    icon: Gauge,
    gradient: 'from-green-500 via-emerald-500 to-teal-500',
    bgGradient: 'from-green-500/35 via-green-500/20 to-green-500/5',
    iconBg: 'bg-green-500/10',
    iconColor: 'text-green-500',
    shadowColor: 'hover:shadow-green-500/25',
    href: '/statistics/quality'
  },
  {
    id: 'topology',
    title: 'Topology Statistics',
    description: 'Graph density, clustering coefficient, path lengths, and structural properties',
    icon: BarChart3,
    gradient: 'from-purple-500 via-fuchsia-500 to-pink-500',
    bgGradient: 'from-purple-500/35 via-purple-500/20 to-purple-500/5',
    iconBg: 'bg-purple-500/10',
    iconColor: 'text-purple-500',
    shadowColor: 'hover:shadow-purple-500/25',
    href: '/statistics/topology'
  },
  {
    id: 'performance',
    title: 'Performance Analytics',
    description: 'Component utilization, response time patterns, and system efficiency metrics',
    icon: Activity,
    gradient: 'from-orange-500 via-amber-500 to-yellow-500',
    bgGradient: 'from-orange-500/35 via-orange-500/20 to-orange-500/5',
    iconBg: 'bg-orange-500/10',
    iconColor: 'text-orange-500',
    shadowColor: 'hover:shadow-orange-500/25',
    href: '/statistics/performance'
  },
  {
    id: 'comparison',
    title: 'Layer Comparison',
    description: 'Compare statistics across application, middleware, and infrastructure layers',
    icon: GitCompare,
    gradient: 'from-cyan-500 via-sky-500 to-blue-500',
    bgGradient: 'from-cyan-500/35 via-cyan-500/20 to-cyan-500/5',
    iconBg: 'bg-cyan-500/10',
    iconColor: 'text-cyan-500',
    shadowColor: 'hover:shadow-cyan-500/25',
    href: '/statistics/comparison'
  },
  {
    id: 'trends',
    title: 'Temporal Trends',
    description: 'Historical analysis of metrics over time and change detection',
    icon: TrendingUp,
    gradient: 'from-rose-500 via-pink-500 to-fuchsia-500',
    bgGradient: 'from-rose-500/35 via-rose-500/20 to-rose-500/5',
    iconBg: 'bg-rose-500/10',
    iconColor: 'text-rose-500',
    shadowColor: 'hover:shadow-rose-500/25',
    href: '/statistics/trends'
  },
]

// Quick Statistics - Fast computation stats
const quickStatistics = [
  {
    id: 'degree-distribution',
    title: 'Degree Distribution',
    description: 'Analyze node connectivity patterns: hubs, isolated nodes, and degree statistics',
    why: 'Identifies critical hubs and connectivity imbalances that impact system resilience',
    icon: Network,
    gradient: 'from-violet-500 via-purple-500 to-indigo-500',
    bgGradient: 'from-violet-500/35 via-violet-500/20 to-violet-500/5',
    iconBg: 'bg-violet-500/10',
    iconColor: 'text-violet-500',
    shadowColor: 'hover:shadow-violet-500/25',
    href: '/statistics/degree-distribution',
    computeTime: '< 100ms',
    category: 'Structure'
  },
  {
    id: 'connectivity-density',
    title: 'Connectivity Density',
    description: 'Measure system coupling: how interconnected components are relative to maximum possible connections',
    why: 'High density indicates tight coupling, affecting maintainability and failure propagation',
    icon: Activity,
    gradient: 'from-cyan-500 via-teal-500 to-emerald-500',
    bgGradient: 'from-cyan-500/35 via-cyan-500/20 to-cyan-500/5',
    iconBg: 'bg-cyan-500/10',
    iconColor: 'text-cyan-500',
    shadowColor: 'hover:shadow-cyan-500/25',
    href: '/statistics/connectivity-density',
    computeTime: '< 50ms',
    category: 'Structure'
  },
  {
    id: 'clustering-coefficient',
    title: 'Clustering Coefficient',
    description: 'Measure component grouping: how neighbors of a node tend to be connected to each other',
    why: 'High clustering indicates modular structure with well-defined component groups',
    icon: Waypoints,
    gradient: 'from-purple-500 via-pink-500 to-rose-500',
    bgGradient: 'from-purple-500/35 via-purple-500/20 to-purple-500/5',
    iconBg: 'bg-purple-500/10',
    iconColor: 'text-purple-500',
    shadowColor: 'hover:shadow-purple-500/25',
    href: '/statistics/clustering-coefficient',
    computeTime: '< 200ms',
    category: 'Structure'
  },
  {
    id: 'dependency-depth',
    title: 'Dependency Depth',
    description: 'Measure dependency chain depth: how many layers deep transitive dependencies extend',
    why: 'Deep chains increase failure propagation risk, testing complexity, and change impact radius',
    icon: Layers,
    gradient: 'from-orange-500 via-amber-500 to-yellow-500',
    bgGradient: 'from-orange-500/35 via-orange-500/20 to-orange-500/5',
    iconBg: 'bg-orange-500/10',
    iconColor: 'text-orange-500',
    shadowColor: 'hover:shadow-orange-500/25',
    href: '/statistics/dependency-depth',
    computeTime: '< 150ms',
    category: 'Architecture'
  },
  {
    id: 'component-isolation',
    title: 'Component Isolation',
    description: 'Identify isolated, source, and sink components to understand architectural roles and dependencies',
    why: 'Reveals entry points, foundations, and unused components - critical for understanding system structure',
    icon: Box,
    gradient: 'from-red-500 via-orange-500 to-amber-500',
    bgGradient: 'from-red-500/35 via-red-500/20 to-red-500/5',
    iconBg: 'bg-red-500/10',
    iconColor: 'text-red-500',
    shadowColor: 'hover:shadow-red-500/25',
    href: '/statistics/component-isolation',
    computeTime: '< 100ms',
    category: 'Architecture'
  },
  {
    id: 'component-redundancy',
    title: 'Component Redundancy',
    description: 'Analyze system resilience by identifying single points of failure and measuring fault tolerance',
    why: 'Discovers critical components whose failure would disconnect the system - essential for reliability',
    icon: Shield,
    gradient: 'from-cyan-500 via-blue-500 to-indigo-500',
    bgGradient: 'from-cyan-500/35 via-cyan-500/20 to-cyan-500/5',
    iconBg: 'bg-cyan-500/10',
    iconColor: 'text-cyan-500',
    shadowColor: 'hover:shadow-cyan-500/25',
    href: '/statistics/component-redundancy',
    computeTime: '< 200ms',
    category: 'Resilience'
  },
  {
    id: 'message-flow-patterns',
    title: 'Message Flow Patterns',
    description: 'Analyze pub-sub communication patterns: hot topics, broker utilization, and message flow bottlenecks',
    why: 'Identifies communication hotspots, bottlenecks, and isolated applications in message flow',
    icon: Zap,
    gradient: 'from-yellow-500 via-orange-500 to-red-500',
    bgGradient: 'from-yellow-500/35 via-yellow-500/20 to-yellow-500/5',
    iconBg: 'bg-yellow-500/10',
    iconColor: 'text-yellow-500',
    shadowColor: 'hover:shadow-yellow-500/25',
    href: '/statistics/message-flow-patterns',
    computeTime: '< 150ms',
    category: 'Communication'
  },
  {
    id: 'node-weight-distribution',
    title: 'Node Weight Distribution',
    description: 'Analyze component importance distribution: how weight (criticality) is spread across components',
    why: 'Reveals concentration of critical components and importance hierarchy in the architecture',
    icon: TrendingUp,
    gradient: 'from-indigo-500 via-purple-500 to-pink-500',
    bgGradient: 'from-indigo-500/35 via-indigo-500/20 to-indigo-500/5',
    iconBg: 'bg-indigo-500/10',
    iconColor: 'text-indigo-500',
    shadowColor: 'hover:shadow-indigo-500/25',
    href: '/statistics/node-weight-distribution',
    computeTime: '< 100ms',
    category: 'Importance'
  },
  {
    id: 'edge-weight-distribution',
    title: 'Edge Weight Distribution',
    description: 'Analyze dependency strength distribution: how importance is spread across connections',
    why: 'Identifies critical dependencies and reveals coupling strength patterns in the system',
    icon: GitCompare,
    gradient: 'from-emerald-500 via-green-500 to-teal-500',
    bgGradient: 'from-emerald-500/35 via-emerald-500/20 to-emerald-500/5',
    iconBg: 'bg-emerald-500/10',
    iconColor: 'text-emerald-500',
    shadowColor: 'hover:shadow-emerald-500/25',
    href: '/statistics/edge-weight-distribution',
    computeTime: '< 100ms',
    category: 'Importance'
  },
]

export default function StatisticsPage() {
  const router = useRouter()
  const { status, stats, initialLoadComplete } = useConnection()

  const isConnected = status === 'connected'

  // Helper function to format type names
  const formatTypeName = (name: string): string => {
    return name
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ')
  }

  // Process component type distribution (from node_counts)
  const componentTypeData = stats?.node_counts ? Object.entries(stats.node_counts).map(([name, value]) => ({
    name: formatTypeName(name),
    value: value as number
  })) : []

  // Process edge type distribution (from edge_counts and structural_edge_counts)
  const edgeTypeData = (() => {
    const allEdgeCounts: Record<string, number> = {}
    
    // Add derived edges
    if (stats?.edge_counts) {
      Object.entries(stats.edge_counts).forEach(([type, count]) => {
        allEdgeCounts[type] = (count as number)
      })
    }
    
    // Add structural edges
    if (stats?.structural_edge_counts) {
      Object.entries(stats.structural_edge_counts).forEach(([type, count]) => {
        allEdgeCounts[type] = (count as number)
      })
    }
    
    return Object.entries(allEdgeCounts).map(([name, value]) => ({
      name: formatTypeName(name),
      value
    }))
  })()

  // Color palettes for charts - vibrant and distinct colors for better visualization
  const componentColors = [
    '#8b5cf6', // Vibrant Purple
    '#3b82f6', // Bright Blue
    '#10b981', // Emerald Green
    '#f59e0b', // Amber
    '#ef4444', // Red
    '#ec4899', // Pink
    '#14b8a6', // Teal
    '#8b5cf6', // Purple (repeat)
    '#f97316', // Orange
    '#06b6d4', // Cyan
    '#a855f7', // Purple variant
    '#84cc16', // Lime
  ]
  const edgeColors = [
    '#ec4899', // Hot Pink
    '#3b82f6', // Blue
    '#8b5cf6', // Purple
    '#14b8a6', // Teal
    '#f59e0b', // Amber
    '#ef4444', // Red
    '#06b6d4', // Cyan
    '#a855f7', // Violet
    '#10b981', // Green
    '#f97316', // Orange
    '#6366f1', // Indigo
    '#84cc16', // Lime
  ]

  // Loading State - show when connecting or when initial load hasn't completed
  if (!initialLoadComplete || status === 'connecting' || (isConnected && !stats)) {
    return (
      <AppLayout title="Statistics" description="Advanced statistical analysis and insights">
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" text={status === 'connecting' ? "Connecting to database..." : "Loading statistics..."} />
        </div>
      </AppLayout>
    )
  }

  // Disconnected State - show the no connection banner
  const showNoConnection = !isConnected

  return (
    <AppLayout title="Statistics" description="Advanced statistical analysis and insights">
      <div className="space-y-6">
        
        {/* No Connection Info Banner - Show if not connected */}
        {showNoConnection && <NoConnectionInfo />}

        {/* Show content only when connected */}
        {isConnected && (
          <>
            {/* Page Header */}
            <Card className="relative overflow-hidden border-0 shadow-xl hover:shadow-2xl hover:shadow-purple-500/25 transition-all duration-300">
              {/* Gradient border */}
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500">
                <div className="w-full h-full rounded-lg bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600" />
              </div>
              
              <CardContent className="p-8 relative text-white">
                <div className="flex items-center justify-between gap-6">
                  <div className="flex-1">
                    <h3 className="text-3xl font-bold mb-3">Statistical Analysis Dashboard</h3>
                    <p className="text-white/95 mb-3 max-w-3xl text-lg">
                      Fast-computing metrics and comprehensive insights into your distributed system
                    </p>
                    <p className="text-white/80 text-sm">
                      Select a statistics category below to view detailed analysis and visualizations
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Overview Statistics - KPIs */}
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {/* Total Components */}
          <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-blue-500/25 transition-all duration-300 transform hover:scale-[1.02]">
            <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500">
              <div className="w-full h-full bg-background rounded-lg" />
            </div>
            <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-blue-500/30 via-blue-500/15 to-blue-500/5" />
            
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
              <CardTitle className="text-sm font-medium">Total Components</CardTitle>
              <div className="rounded-xl bg-blue-500/10 p-2.5">
                <Waypoints className="h-4 w-4 text-blue-500" />
              </div>
            </CardHeader>
            <CardContent className="relative">
              <div className="text-3xl font-bold text-blue-500">{stats?.total_nodes?.toLocaleString() || 0}</div>
              <p className="text-xs text-muted-foreground mt-1">System nodes</p>
            </CardContent>
          </Card>

          {/* Total Connections */}
          <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-purple-500/25 transition-all duration-300 transform hover:scale-[1.02]">
            <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-purple-500 via-fuchsia-500 to-pink-500">
              <div className="w-full h-full bg-background rounded-lg" />
            </div>
            <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-purple-500/30 via-purple-500/15 to-purple-500/5" />
            
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
              <CardTitle className="text-sm font-medium">Total Connections</CardTitle>
              <div className="rounded-xl bg-purple-500/10 p-2.5">
                <Zap className="h-4 w-4 text-purple-500" />
              </div>
            </CardHeader>
            <CardContent className="relative">
              <div className="text-3xl font-bold text-purple-500">{stats?.total_edges?.toLocaleString() || 0}</div>
              <p className="text-xs text-muted-foreground mt-1">System edges</p>
            </CardContent>
          </Card>

          {/* Component Types */}
          <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-orange-500/25 transition-all duration-300 transform hover:scale-[1.02]">
            <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-orange-500 via-amber-500 to-yellow-500">
              <div className="w-full h-full bg-background rounded-lg" />
            </div>
            <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-orange-500/30 via-orange-500/15 to-orange-500/5" />
            
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
              <CardTitle className="text-sm font-medium">Component Types</CardTitle>
              <div className="rounded-xl bg-orange-500/10 p-2.5">
                <Box className="h-4 w-4 text-orange-500" />
              </div>
            </CardHeader>
            <CardContent className="relative">
              <div className="text-3xl font-bold text-orange-500">{componentTypeData.length}</div>
              <p className="text-xs text-muted-foreground mt-1">Unique types</p>
            </CardContent>
          </Card>

          {/* Connection Types */}
          <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-green-500/25 transition-all duration-300 transform hover:scale-[1.02]">
            <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500">
              <div className="w-full h-full bg-background rounded-lg" />
            </div>
            <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-green-500/30 via-green-500/15 to-green-500/5" />
            
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2 relative">
              <CardTitle className="text-sm font-medium">Connection Types</CardTitle>
              <div className="rounded-xl bg-green-500/10 p-2.5">
                <Tags className="h-4 w-4 text-green-500" />
              </div>
            </CardHeader>
            <CardContent className="relative">
              <div className="text-3xl font-bold text-green-500">{edgeTypeData.length}</div>
              <p className="text-xs text-muted-foreground mt-1">Edge categories</p>
            </CardContent>
          </Card>
        </div>

        {/* Distribution Charts */}
        <div className="grid gap-6 md:grid-cols-2">
          {/* Component Type Distribution */}
          <Card className="relative overflow-hidden border-0 shadow-lg">
            <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500">
              <div className="w-full h-full bg-background rounded-lg" />
            </div>
            <div className="absolute inset-[2px] rounded-lg bg-background" />
            
            <CardHeader className="relative">
              <div className="flex items-center gap-2">
                <div className="rounded-xl bg-blue-500/10 p-2.5">
                  <Box className="h-5 w-5 text-blue-500" />
                </div>
                <div>
                  <CardTitle>Component Type Distribution</CardTitle>
                  <CardDescription>Breakdown by component types</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="relative">
              {componentTypeData.length > 0 ? (
                <ChartContainer
                  config={componentTypeData.reduce((acc, item, idx) => ({
                    ...acc,
                    [item.name]: {
                      label: item.name,
                      color: componentColors[idx % componentColors.length],
                    },
                  }), {})}
                  className="h-[300px] w-full"
                >
                  <PieChart>
                    <ChartTooltip content={<ChartTooltipContent />} />
                    <ChartLegend content={<ChartLegendContent />} />
                    <Pie
                      data={componentTypeData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={90}
                      innerRadius={50}
                      paddingAngle={2}
                    >
                      {componentTypeData.map((entry, index) => (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={componentColors[index % componentColors.length]}
                        />
                      ))}
                    </Pie>
                  </PieChart>
                </ChartContainer>
              ) : (
                <div className="h-[300px] flex items-center justify-center text-muted-foreground">
                  No component data available
                </div>
              )}
            </CardContent>
          </Card>

          {/* Connection Type Distribution */}
          <Card className="relative overflow-hidden border-0 shadow-lg">
            <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-purple-500 via-fuchsia-500 to-pink-500">
              <div className="w-full h-full bg-background rounded-lg" />
            </div>
            <div className="absolute inset-[2px] rounded-lg bg-background" />
            
            <CardHeader className="relative">
              <div className="flex items-center gap-2">
                <div className="rounded-xl bg-purple-500/10 p-2.5">
                  <Zap className="h-5 w-5 text-purple-500" />
                </div>
                <div>
                  <CardTitle>Connection Type Distribution</CardTitle>
                  <CardDescription>Breakdown by edge types</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="relative">
              {edgeTypeData.length > 0 ? (
                <ChartContainer
                  config={edgeTypeData.reduce((acc, item, idx) => ({
                    ...acc,
                    [item.name]: {
                      label: item.name,
                      color: edgeColors[idx % edgeColors.length],
                    },
                  }), {})}
                  className="h-[300px] w-full"
                >
                  <PieChart>
                    <ChartTooltip content={<ChartTooltipContent />} />
                    <ChartLegend content={<ChartLegendContent />} />
                    <Pie
                      data={edgeTypeData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={90}
                      innerRadius={50}
                      paddingAngle={2}
                    >
                      {edgeTypeData.map((entry, index) => (
                        <Cell 
                          key={`cell-${index}`} 
                          fill={edgeColors[index % edgeColors.length]}
                        />
                      ))}
                    </Pie>
                  </PieChart>
                </ChartContainer>
              ) : (
                <div className="h-[300px] flex items-center justify-center text-muted-foreground">
                  No connection data available
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Statistics Section */}
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-gradient-to-r from-violet-500 to-purple-500 p-2.5">
              <Zap className="h-5 w-5 text-white" />
            </div>
            <div>
              <h3 className="text-xl font-bold">Quick Statistics</h3>
              <p className="text-sm text-muted-foreground">Fast-computing metrics for instant insights</p>
            </div>
          </div>

          {/* Statistics Grid */}
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {quickStatistics.map((stat) => {
            const Icon = stat.icon
            
            return (
              <Card
                key={stat.id}
                className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-[1.02] cursor-pointer group"
                onClick={() => router.push(stat.href)}
              >
                {/* Gradient border */}
                <div className={`absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r ${stat.gradient}`}>
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                
                {/* Background gradient overlay */}
                <div className={`absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] ${stat.bgGradient}`} />
                
                <CardHeader className="relative pb-3">
                  <div className="flex items-center justify-between mb-3">
                    <div className={`rounded-xl ${stat.iconBg} p-3`}>
                      <Icon className={`h-6 w-6 ${stat.iconColor}`} />
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge className={`${stat.iconBg} ${stat.iconColor} border-0 text-xs`}>
                        {stat.computeTime}
                      </Badge>
                      <ArrowRight className={`h-5 w-5 ${stat.iconColor} opacity-0 group-hover:opacity-100 transform translate-x-0 group-hover:translate-x-1 transition-all duration-200`} />
                    </div>
                  </div>
                  <CardTitle className="text-lg font-semibold">{stat.title}</CardTitle>
                  <CardDescription className="text-sm mt-2">
                    {stat.description}
                  </CardDescription>
                </CardHeader>
                
                <CardContent className="relative space-y-3">
                  <div className="flex items-start gap-2 text-xs text-muted-foreground">
                    <span className="font-semibold">Why Important:</span>
                    <span className="flex-1">{stat.why}</span>
                  </div>
                  <div className={`flex items-center gap-2 text-sm font-medium ${stat.iconColor}`}>
                    <span>View Analysis</span>
                    <ArrowRight className="h-4 w-4 transform translate-x-0 group-hover:translate-x-1 transition-transform duration-200" />
                  </div>
                </CardContent>
              </Card>
            )
          })}
          </div>
        </div>

        {/* Info Card */}
        <Card className="relative overflow-hidden border-0 shadow-lg">
          {/* Gradient border */}
          <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-slate-400 via-gray-500 to-slate-600">
            <div className="w-full h-full bg-background rounded-lg" />
          </div>
          
          {/* Background gradient overlay */}
          <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-slate-500/15 via-slate-500/8 to-transparent" />
          
          <CardHeader className="relative">
            <CardTitle className="text-base">About Statistics</CardTitle>
          </CardHeader>
          <CardContent className="relative">
            <p className="text-sm text-muted-foreground">
              Statistics pages provide in-depth quantitative analysis of your distributed system. 
              Each category offers specialized visualizations, distribution charts, and key metrics 
              to help you understand system characteristics and identify optimization opportunities.
            </p>
          </CardContent>
        </Card>
          </>
        )}
      </div>
    </AppLayout>
  )
}
