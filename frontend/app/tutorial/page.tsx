"use client"

import { useRouter } from "next/navigation"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  BookOpen,
  Database,
  Settings,
  BarChart3,
  Waypoints,
  Target,
  GitBranch,
  Zap,
  CheckCircle2,
  ArrowRight,
  Play,
  Download,
  Upload,
  Eye,
  Gauge,
  TrendingUp,
  Shield,
  Wrench,
  Activity,
  Sparkles,
  GraduationCap
} from "lucide-react"

export default function TutorialPage() {
  const router = useRouter()

  return (
    <AppLayout title="Tutorial" description="Learn how to use the Graph Analysis Platform">
      <div className="space-y-6">
        {/* Introduction Banner - Matching Dashboard Style */}
        <Card className="relative overflow-hidden border-0 shadow-xl hover:shadow-2xl hover:shadow-purple-500/25 transition-all duration-300 transform hover:scale-[1.01]">
          {/* Gradient border */}
          <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500">
            <div className="w-full h-full rounded-lg bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600" />
          </div>
          
          <CardContent className="p-8 relative text-white">
            <div className="flex items-center justify-between gap-6">
              <div className="flex-1">
                <div className="flex items-center gap-3 mb-3">
                  <GraduationCap className="h-6 w-6" />
                  <Badge className="bg-white/20 text-white border-white/30 hover:bg-white/30 px-3 py-1.5">
                    Interactive Tutorial
                  </Badge>
                </div>
                <h3 className="text-3xl font-bold mb-3">Welcome to Graph Analysis Platform</h3>
                <p className="text-white/95 mb-5 max-w-3xl text-lg">
                  A comprehensive tool for analyzing distributed system architectures through graph-based visualization and quality metrics
                </p>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="flex items-center gap-2.5">
                    <div className="rounded-lg bg-white/20 p-2">
                      <Waypoints className="h-4 w-4" />
                    </div>
                    <span className="text-sm font-medium">Graph Visualization</span>
                  </div>
                  <div className="flex items-center gap-2.5">
                    <div className="rounded-lg bg-white/20 p-2">
                      <BarChart3 className="h-4 w-4" />
                    </div>
                    <span className="text-sm font-medium">Quality Analysis</span>
                  </div>
                  <div className="flex items-center gap-2.5">
                    <div className="rounded-lg bg-white/20 p-2">
                      <Target className="h-4 w-4" />
                    </div>
                    <span className="text-sm font-medium">Classification</span>
                  </div>
                  <div className="flex items-center gap-2.5">
                    <div className="rounded-lg bg-white/20 p-2">
                      <Database className="h-4 w-4" />
                    </div>
                    <span className="text-sm font-medium">Neo4j Powered</span>
                  </div>
                </div>
              </div>
              <div className="hidden lg:flex flex-col gap-3">
                <Button
                  size="lg"
                  className="bg-white text-purple-600 hover:bg-white/90 shadow-lg"
                  onClick={() => router.push('/data')}
                >
                  <Database className="mr-2 h-5 w-5" />
                  Get Started
                </Button>
                <Button
                  size="lg"
                  variant="outline"
                  className="bg-white/15 text-white border-white/30 hover:bg-white/25 backdrop-blur-sm"
                  onClick={() => router.push('/dashboard')}
                >
                  View Dashboard
                  <ArrowRight className="ml-2 h-5 w-5" />
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Getting Started */}
        <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-indigo-500/20 transition-all duration-300">
          {/* Gradient border */}
          <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-400 via-indigo-500 to-purple-600">
            <div className="w-full h-full bg-background rounded-lg" />
          </div>
          
          {/* Background gradient overlay */}
          <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-indigo-500/35 via-indigo-500/20 to-indigo-500/5" />
          
          <CardHeader className="relative">
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-indigo-500/10 p-2.5">
                <Zap className="h-5 w-5 text-indigo-500" />
              </div>
              <div>
                <CardTitle className="text-lg font-semibold">Getting Started</CardTitle>
                <CardDescription>Follow these steps to start analyzing your system</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4 relative">
            {/* Step 1 */}
            <div className="flex gap-4 rounded-lg bg-blue-500/5 border border-blue-500/20 p-4 hover:bg-blue-500/10 transition-colors">
              <div className="flex-shrink-0">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-blue-500 text-white font-bold shadow-lg">
                  1
                </div>
              </div>
              <div className="flex-1 space-y-2">
                <div className="flex items-center gap-2">
                  <Settings className="h-4 w-4 text-blue-500" />
                  <h3 className="font-semibold text-blue-500">Connect to Neo4j Database</h3>
                </div>
                <p className="text-sm text-muted-foreground">
                  Navigate to the Settings page and configure your Neo4j database connection. You'll need the URI, username,
                  password, and database name. Once connected, the system will be ready to store and analyze your graph data.
                </p>
              </div>
            </div>

            {/* Step 2 */}
            <div className="flex gap-4 rounded-lg bg-purple-500/5 border border-purple-500/20 p-4 hover:bg-purple-500/10 transition-colors">
              <div className="flex-shrink-0">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-purple-500 text-white font-bold shadow-lg">
                  2
                </div>
              </div>
              <div className="flex-1 space-y-2">
                <div className="flex items-center gap-2">
                  <Database className="h-4 w-4 text-purple-500" />
                  <h3 className="font-semibold text-purple-500">Populate Your Database</h3>
                </div>
                <p className="text-sm text-muted-foreground">
                  Go to the Data page to generate sample graphs or import your own system architecture data. You can generate
                  graphs of different sizes (tiny, small, medium, large) with various scenarios and even inject anti-patterns
                  for testing.
                </p>
              </div>
            </div>

            {/* Step 3 */}
            <div className="flex gap-4 rounded-lg bg-green-500/5 border border-green-500/20 p-4 hover:bg-green-500/10 transition-colors">
              <div className="flex-shrink-0">
                <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-green-500 text-white font-bold shadow-lg">
                  3
                </div>
              </div>
              <div className="flex-1 space-y-2">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                  <h3 className="font-semibold text-green-500">Start Analyzing</h3>
                </div>
                <p className="text-sm text-muted-foreground">
                  Explore the Dashboard, run comprehensive analyses, visualize graphs, and classify components to gain insights
                  into your system architecture.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Pages Overview */}
        <div>
          <h2 className="text-2xl font-bold mb-4">Pages Overview</h2>
          <div className="grid gap-4 md:grid-cols-2">
            {/* Dashboard */}
            <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-blue-500/20 transition-all duration-300 transform hover:scale-[1.02] cursor-pointer group"
              onClick={() => router.push('/dashboard')}>
              {/* Gradient border */}
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-400 via-indigo-500 to-violet-600">
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              
              {/* Background gradient overlay */}
              <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-blue-500/35 via-blue-500/20 to-blue-500/5" />
              
              <CardHeader className="relative">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="rounded-xl bg-blue-500/10 p-2.5 group-hover:scale-110 transition-transform">
                      <BarChart3 className="h-5 w-5 text-blue-500" />
                    </div>
                    <div>
                      <CardTitle className="text-base">Dashboard</CardTitle>
                      <CardDescription>System overview and statistics</CardDescription>
                    </div>
                  </div>
                  <ArrowRight className="h-5 w-5 text-blue-500 opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
              </CardHeader>
              <CardContent className="space-y-3 relative">
                <p className="text-sm text-muted-foreground">
                  View key metrics about your graph database including node counts, edge distributions, and weight statistics.
                  Quick access to run comprehensive analysis.
                </p>
                <div className="space-y-1.5 text-xs text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-blue-500" />
                    <span>Total nodes and edges count</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-blue-500" />
                    <span>Node type distribution</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-blue-500" />
                    <span>Relationship statistics</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Data Management */}
            <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-purple-500/20 transition-all duration-300 transform hover:scale-[1.02] cursor-pointer group"
              onClick={() => router.push('/data')}>
              {/* Gradient border */}
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-purple-400 via-fuchsia-500 to-pink-600">
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              
              {/* Background gradient overlay */}
              <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-purple-500/35 via-purple-500/20 to-purple-500/5" />
              
              <CardHeader className="relative">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="rounded-xl bg-purple-500/10 p-2.5 group-hover:scale-110 transition-transform">
                      <Database className="h-5 w-5 text-purple-500" />
                    </div>
                    <div>
                      <CardTitle className="text-base">Data Management</CardTitle>
                      <CardDescription>Generate and import graph data</CardDescription>
                    </div>
                  </div>
                  <ArrowRight className="h-5 w-5 text-purple-500 opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
              </CardHeader>
              <CardContent className="space-y-3 relative">
                <p className="text-sm text-muted-foreground">
                  Generate synthetic graphs or import real system architectures. Configure graph size, scenario type, and
                  optionally inject anti-patterns for testing.
                </p>
                <div className="space-y-1.5 text-xs text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <Play className="h-3 w-3 text-purple-500" />
                    <span>Generate & Import - Create and load graphs directly</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Download className="h-3 w-3 text-purple-500" />
                    <span>Download Graph File - Save generated graphs</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Upload className="h-3 w-3 text-purple-500" />
                    <span>Import from File - Load existing JSON graphs</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Graph Visualization */}
            <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-green-500/20 transition-all duration-300 transform hover:scale-[1.02] cursor-pointer group"
              onClick={() => router.push('/explorer')}>
              {/* Gradient border */}
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-green-400 via-emerald-500 to-teal-600">
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              
              {/* Background gradient overlay */}
              <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-green-500/35 via-green-500/20 to-green-500/5" />
              
              <CardHeader className="relative">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="rounded-xl bg-green-500/10 p-2.5 group-hover:scale-110 transition-transform">
                      <Waypoints className="h-5 w-5 text-green-500" />
                    </div>
                    <div>
                      <CardTitle className="text-base">Graph Visualization</CardTitle>
                      <CardDescription>Interactive 2D/3D graph explorer</CardDescription>
                    </div>
                  </div>
                  <ArrowRight className="h-5 w-5 text-green-500 opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
              </CardHeader>
              <CardContent className="space-y-3 relative">
                <p className="text-sm text-muted-foreground">
                  Visualize your system architecture as an interactive force-directed graph. Switch between 2D and 3D views,
                  filter by node types, and explore subgraphs.
                </p>
                <div className="space-y-1.5 text-xs text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <Eye className="h-3 w-3 text-green-500" />
                    <span>Toggle 2D/3D visualization modes</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <GitBranch className="h-3 w-3 text-green-500" />
                    <span>View subgraphs centered on specific nodes</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-green-500" />
                    <span>Filter by node and relationship types</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Quality Analysis */}
            <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-orange-500/20 transition-all duration-300 transform hover:scale-[1.02] cursor-pointer group"
              onClick={() => router.push('/analysis')}>
              {/* Gradient border */}
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-orange-400 via-amber-500 to-yellow-600">
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              
              {/* Background gradient overlay */}
              <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-orange-500/35 via-orange-500/20 to-orange-500/5" />
              
              <CardHeader className="relative">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="rounded-xl bg-orange-500/10 p-2.5 group-hover:scale-110 transition-transform">
                      <Activity className="h-5 w-5 text-orange-500" />
                    </div>
                    <div>
                      <CardTitle className="text-base">Quality Analysis</CardTitle>
                      <CardDescription>System quality attribute evaluation</CardDescription>
                    </div>
                  </div>
                  <ArrowRight className="h-5 w-5 text-orange-500 opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
              </CardHeader>
              <CardContent className="space-y-3 relative">
                <p className="text-sm text-muted-foreground">
                  Run comprehensive quality analysis to evaluate your system across multiple dimensions including reliability,
                  maintainability, and availability.
                </p>
                <div className="space-y-1.5 text-xs text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <Shield className="h-3 w-3 text-orange-500" />
                    <span>Reliability - System fault tolerance</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Wrench className="h-3 w-3 text-orange-500" />
                    <span>Maintainability - Change impact analysis</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <TrendingUp className="h-3 w-3 text-orange-500" />
                    <span>Availability - Service uptime potential</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Classification */}
            <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-pink-500/20 transition-all duration-300 transform hover:scale-[1.02] cursor-pointer group"
              onClick={() => router.push('/analysis')}>
              {/* Gradient border */}
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-pink-400 via-rose-500 to-red-600">
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              
              {/* Background gradient overlay */}
              <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-pink-500/35 via-pink-500/20 to-pink-500/5" />
              
              <CardHeader className="relative">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="rounded-xl bg-pink-500/10 p-2.5 group-hover:scale-110 transition-transform">
                      <Target className="h-5 w-5 text-pink-500" />
                    </div>
                    <div>
                      <CardTitle className="text-base">Component Classification</CardTitle>
                      <CardDescription>Identify critical components</CardDescription>
                    </div>
                  </div>
                  <ArrowRight className="h-5 w-5 text-pink-500 opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
              </CardHeader>
              <CardContent className="space-y-3 relative">
                <p className="text-sm text-muted-foreground">
                  Classify components based on graph metrics like betweenness centrality, PageRank, and degree. Identify
                  critical, high, medium, and low importance components.
                </p>
                <div className="space-y-1.5 text-xs text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <Gauge className="h-3 w-3 text-pink-500" />
                    <span>Multiple metrics: betweenness, PageRank, degree</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-pink-500" />
                    <span>Statistical thresholds for classification</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-pink-500" />
                    <span>Export results as CSV</span>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Settings */}
            <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-violet-500/20 transition-all duration-300 transform hover:scale-[1.02] cursor-pointer group"
              onClick={() => router.push('/settings')}>
              {/* Gradient border */}
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-violet-400 via-indigo-500 to-blue-600">
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              
              {/* Background gradient overlay */}
              <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-violet-500/35 via-violet-500/20 to-violet-500/5" />
              
              <CardHeader className="relative">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="rounded-xl bg-violet-500/10 p-2.5 group-hover:scale-110 transition-transform">
                      <Settings className="h-5 w-5 text-violet-500" />
                    </div>
                    <div>
                      <CardTitle className="text-base">Settings</CardTitle>
                      <CardDescription>Database and API configuration</CardDescription>
                    </div>
                  </div>
                  <ArrowRight className="h-5 w-5 text-violet-500 opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
              </CardHeader>
              <CardContent className="space-y-3 relative">
                <p className="text-sm text-muted-foreground">
                  Configure your Neo4j database connection and backend API settings. Test connections and manage your
                  configuration.
                </p>
                <div className="space-y-1.5 text-xs text-muted-foreground">
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-violet-500" />
                    <span>Neo4j connection parameters</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-violet-500" />
                    <span>Backend API URL configuration</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="h-3 w-3 text-violet-500" />
                    <span>Connection health checks</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Key Concepts */}
        <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-green-500/20 transition-all duration-300">
          {/* Gradient border */}
          <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-green-400 via-emerald-500 to-teal-600">
            <div className="w-full h-full bg-background rounded-lg" />
          </div>
          
          {/* Background gradient overlay */}
          <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-green-500/35 via-green-500/20 to-green-500/5" />
          
          <CardHeader className="relative">
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-green-500/10 p-2.5">
                <GitBranch className="h-5 w-5 text-green-500" />
              </div>
              <div>
                <CardTitle className="text-lg font-semibold">Key Concepts</CardTitle>
                <CardDescription>Understanding the platform terminology</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4 relative">
            <div className="grid gap-6 md:grid-cols-2">
              <div className="space-y-3 rounded-lg bg-green-500/5 border border-green-500/20 p-4">
                <h4 className="font-semibold text-sm flex items-center gap-2 text-green-500">
                  <Waypoints className="h-4 w-4" />
                  Node Types
                </h4>
                <div className="space-y-2 text-xs text-muted-foreground">
                  <div className="flex items-start gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-green-500 mt-1.5 flex-shrink-0" />
                    <div><strong className="text-foreground">Application:</strong> Software components or services</div>
                  </div>
                  <div className="flex items-start gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-green-500 mt-1.5 flex-shrink-0" />
                    <div><strong className="text-foreground">Node:</strong> Physical or virtual machines</div>
                  </div>
                  <div className="flex items-start gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-green-500 mt-1.5 flex-shrink-0" />
                    <div><strong className="text-foreground">Broker:</strong> Message brokers (e.g., Kafka, RabbitMQ)</div>
                  </div>
                  <div className="flex items-start gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-green-500 mt-1.5 flex-shrink-0" />
                    <div><strong className="text-foreground">Topic:</strong> Message topics or queues</div>
                  </div>
                </div>
              </div>
              <div className="space-y-3 rounded-lg bg-emerald-500/5 border border-emerald-500/20 p-4">
                <h4 className="font-semibold text-sm flex items-center gap-2 text-emerald-500">
                  <GitBranch className="h-4 w-4" />
                  Relationship Types
                </h4>
                <div className="space-y-2 text-xs text-muted-foreground">
                  <div className="flex items-start gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-emerald-500 mt-1.5 flex-shrink-0" />
                    <div><strong className="text-foreground">RUNS_ON:</strong> Application deployed on node</div>
                  </div>
                  <div className="flex items-start gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-emerald-500 mt-1.5 flex-shrink-0" />
                    <div><strong className="text-foreground">PUBLISHES_TO:</strong> Publishing messages to topic</div>
                  </div>
                  <div className="flex items-start gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-emerald-500 mt-1.5 flex-shrink-0" />
                    <div><strong className="text-foreground">SUBSCRIBES_TO:</strong> Subscribing to topic messages</div>
                  </div>
                  <div className="flex items-start gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-emerald-500 mt-1.5 flex-shrink-0" />
                    <div><strong className="text-foreground">DEPENDS_ON:</strong> Component dependencies</div>
                  </div>
                  <div className="flex items-start gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-emerald-500 mt-1.5 flex-shrink-0" />
                    <div><strong className="text-foreground">CONNECTS_TO:</strong> Network connections</div>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Tips & Best Practices */}
        <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-amber-500/20 transition-all duration-300">
          {/* Gradient border */}
          <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-amber-400 via-orange-500 to-yellow-600">
            <div className="w-full h-full bg-background rounded-lg" />
          </div>
          
          {/* Background gradient overlay */}
          <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-amber-500/35 via-amber-500/20 to-amber-500/5" />
          
          <CardHeader className="relative">
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-amber-500/10 p-2.5">
                <Sparkles className="h-5 w-5 text-amber-500" />
              </div>
              <div>
                <CardTitle className="text-lg font-semibold">Tips & Best Practices</CardTitle>
                <CardDescription>Get the most out of the platform</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-3 relative">
            <div className="space-y-3 text-sm">
              <div className="flex items-start gap-3 rounded-lg bg-green-500/5 border border-green-500/20 p-3 hover:bg-green-500/10 transition-colors">
                <CheckCircle2 className="h-5 w-5 text-green-500 flex-shrink-0 mt-0.5" />
                <div>
                  <strong className="text-green-500">Start Small:</strong>
                  <p className="text-muted-foreground text-xs mt-1">Begin with tiny or small graphs to familiarize yourself with the tools before analyzing larger systems.</p>
                </div>
              </div>
              <div className="flex items-start gap-3 rounded-lg bg-blue-500/5 border border-blue-500/20 p-3 hover:bg-blue-500/10 transition-colors">
                <CheckCircle2 className="h-5 w-5 text-blue-500 flex-shrink-0 mt-0.5" />
                <div>
                  <strong className="text-blue-500">Use Anti-patterns:</strong>
                  <p className="text-muted-foreground text-xs mt-1">Generate graphs with anti-patterns to test how the analysis detects problematic architectures.</p>
                </div>
              </div>
              <div className="flex items-start gap-3 rounded-lg bg-purple-500/5 border border-purple-500/20 p-3 hover:bg-purple-500/10 transition-colors">
                <CheckCircle2 className="h-5 w-5 text-purple-500 flex-shrink-0 mt-0.5" />
                <div>
                  <strong className="text-purple-500">Explore Subgraphs:</strong>
                  <p className="text-muted-foreground text-xs mt-1">Click on nodes in the graph visualization to explore their immediate neighborhoods in detail.</p>
                </div>
              </div>
              <div className="flex items-start gap-3 rounded-lg bg-orange-500/5 border border-orange-500/20 p-3 hover:bg-orange-500/10 transition-colors">
                <CheckCircle2 className="h-5 w-5 text-orange-500 flex-shrink-0 mt-0.5" />
                <div>
                  <strong className="text-orange-500">Export Data:</strong>
                  <p className="text-muted-foreground text-xs mt-1">Use CSV exports from classification and analysis pages for further processing in spreadsheets or other tools.</p>
                </div>
              </div>
              <div className="flex items-start gap-3 rounded-lg bg-pink-500/5 border border-pink-500/20 p-3 hover:bg-pink-500/10 transition-colors">
                <CheckCircle2 className="h-5 w-5 text-pink-500 flex-shrink-0 mt-0.5" />
                <div>
                  <strong className="text-pink-500">Clear Database Option:</strong>
                  <p className="text-muted-foreground text-xs mt-1">The "Clear database before importing" option is checked by default to prevent data conflicts.</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </AppLayout>
  )
}
