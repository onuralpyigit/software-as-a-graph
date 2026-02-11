"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Checkbox } from "@/components/ui/checkbox"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import {
  Database,
  Loader2,
  Settings,
  CheckCircle2,
  AlertTriangle,
  Trash2,
  Play,
  Info,
  Upload,
  Download,
  Sparkles,
  Zap,
  FileJson,
  HardDrive,
  BarChart3,
  Sliders,
  Layers
} from "lucide-react"
import { useConnection } from "@/lib/stores/connection-store"
import { apiClient } from "@/lib/api/client"

const SCALES = [
  { value: "tiny", label: "Tiny", description: "25 total nodes", count: "10 apps, 8 topics, 1 broker, 3 nodes, 3 libs" },
  { value: "small", label: "Small", description: "100 total nodes", count: "40 apps, 30 topics, 3 brokers, 12 nodes, 15 libs" },
  { value: "medium", label: "Medium", description: "500 total nodes", count: "200 apps, 150 topics, 10 brokers, 60 nodes, 80 libs" },
  { value: "large", label: "Large", description: "1,000 total nodes", count: "400 apps, 300 topics, 15 brokers, 120 nodes, 165 libs" },
  { value: "xlarge", label: "X-Large", description: "10,000 total nodes", count: "4,100 apps, 3,000 topics, 50 brokers, 1,200 nodes, 1,650 libs" }
]

export default function DataPage() {
  const router = useRouter()
  const { status, stats, initialLoadComplete } = useConnection()

  const [scale, setScale] = useState("small")
  const [clearFirst, setClearFirst] = useState(true)

  const [isGenerating, setIsGenerating] = useState(false)
  const [isClearing, setIsClearing] = useState(false)
  const [isImporting, setIsImporting] = useState(false)
  const [isDownloading, setIsDownloading] = useState(false)
  const [isDownloadingNeo4j, setIsDownloadingNeo4j] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  const isConnected = status === 'connected'

  const handleGenerate = async () => {
    if (!isConnected) return

    setIsGenerating(true)
    setError(null)
    setSuccess(null)

    try {
      const result = await apiClient.generateAndImport({
        scale,
        seed: 42,
        clear_first: clearFirst
      })

      setSuccess('Graph generated and imported successfully!')

      // Refresh connection stats
      setTimeout(() => {
        window.location.reload()
      }, 2000)
    } catch (error: any) {
      let errorMsg = 'Generation failed'
      if (error.response?.data?.detail) {
        if (typeof error.response.data.detail === 'string') {
          errorMsg = error.response.data.detail
        } else if (Array.isArray(error.response.data.detail)) {
          errorMsg = error.response.data.detail.map((e: any) => e.msg || JSON.stringify(e)).join(', ')
        } else {
          errorMsg = JSON.stringify(error.response.data.detail)
        }
      } else if (error.message) {
        errorMsg = error.message
      }
      setError(errorMsg)
    } finally {
      setIsGenerating(false)
    }
  }

  const handleClear = async () => {
    if (!isConnected) return
    if (!confirm('Are you sure you want to clear all data from the database? This action cannot be undone.')) {
      return
    }

    setIsClearing(true)
    setError(null)
    setSuccess(null)

    try {
      await apiClient.clearDatabase()
      setSuccess('Database cleared successfully')

      setTimeout(() => {
        window.location.reload()
      }, 1500)
    } catch (error: any) {
      let errorMsg = 'Clear operation failed'
      if (error.response?.data?.detail) {
        if (typeof error.response.data.detail === 'string') {
          errorMsg = error.response.data.detail
        } else if (Array.isArray(error.response.data.detail)) {
          errorMsg = error.response.data.detail.map((e: any) => e.msg || JSON.stringify(e)).join(', ')
        } else {
          errorMsg = JSON.stringify(error.response.data.detail)
        }
      } else if (error.message) {
        errorMsg = error.message
      }
      setError(errorMsg)
    } finally {
      setIsClearing(false)
    }
  }

  const handleImportFromFile = async (event: React.ChangeEvent<HTMLInputElement>) => {
    if (!isConnected) return
    const file = event.target.files?.[0]
    if (!file) return

    setIsImporting(true)
    setError(null)
    setSuccess(null)

    try {
      const fileContent = await file.text()
      const graphData = JSON.parse(fileContent)
      
      await apiClient.importGraph(graphData, {
        clear_first: clearFirst
      })

      setSuccess('Graph imported successfully!')

      // Refresh connection stats
      setTimeout(() => {
        window.location.reload()
      }, 2000)
    } catch (error: any) {
      let errorMsg = 'Import failed'
      if (error.response?.data?.detail) {
        if (typeof error.response.data.detail === 'string') {
          errorMsg = error.response.data.detail
        } else if (Array.isArray(error.response.data.detail)) {
          errorMsg = error.response.data.detail.map((e: any) => e.msg || JSON.stringify(e)).join(', ')
        } else {
          errorMsg = JSON.stringify(error.response.data.detail)
        }
      } else if (error.message) {
        errorMsg = error.message
      }
      setError(errorMsg)
    } finally {
      setIsImporting(false)
      // Reset file input
      event.target.value = ''
    }
  }

  const handleDownloadGraph = async () => {
    setIsDownloading(true)
    setError(null)
    setSuccess(null)

    try {
      const blob = await apiClient.generateGraphFile({
        scale,
        scenario: 'generic',
        seed: 42
      })

      // Extract filename from response or create default
      const filename = `graph_${scale}.json`

      // Create download link and trigger download
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      link.style.display = 'none'
      
      document.body.appendChild(link)
      link.click()
      
      // Cleanup
      setTimeout(() => {
        document.body.removeChild(link)
        window.URL.revokeObjectURL(url)
      }, 100)

      setSuccess(`Graph file downloaded: ${filename}`)
    } catch (error: any) {
      let errorMsg = 'Download failed'
      
      // Handle blob error responses
      if (error.response?.data instanceof Blob) {
        try {
          const text = await error.response.data.text()
          const errorData = JSON.parse(text)
          if (errorData.detail) {
            if (typeof errorData.detail === 'string') {
              errorMsg = errorData.detail
            } else if (Array.isArray(errorData.detail)) {
              errorMsg = errorData.detail.map((e: any) => e.msg || JSON.stringify(e)).join(', ')
            } else {
              errorMsg = JSON.stringify(errorData.detail)
            }
          }
        } catch {
          errorMsg = 'Failed to download graph file'
        }
      } else if (error.response?.data?.detail) {
        if (typeof error.response.data.detail === 'string') {
          errorMsg = error.response.data.detail
        } else if (Array.isArray(error.response.data.detail)) {
          errorMsg = error.response.data.detail.map((e: any) => e.msg || JSON.stringify(e)).join(', ')
        } else {
          errorMsg = JSON.stringify(error.response.data.detail)
        }
      } else if (error.message) {
        errorMsg = error.message
      }
      
      setError(errorMsg)
    } finally {
      setIsDownloading(false)
    }
  }

  const handleDownloadNeo4jData = async () => {
    if (!isConnected) return

    setIsDownloadingNeo4j(true)
    setError(null)
    setSuccess(null)

    try {
      const blob = await apiClient.exportNeo4jData()

      // Create filename with timestamp
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5)
      const filename = `neo4j_export_${timestamp}.json`

      // Create download link and trigger download
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = filename
      link.style.display = 'none'
      
      document.body.appendChild(link)
      link.click()
      
      // Cleanup
      setTimeout(() => {
        document.body.removeChild(link)
        window.URL.revokeObjectURL(url)
      }, 100)

      setSuccess(`Neo4j data exported: ${filename}`)
    } catch (error: any) {
      let errorMsg = 'Export failed'
      
      // Handle blob error responses
      if (error.response?.data instanceof Blob) {
        try {
          const text = await error.response.data.text()
          const errorData = JSON.parse(text)
          if (errorData.detail) {
            if (typeof errorData.detail === 'string') {
              errorMsg = errorData.detail
            } else if (Array.isArray(errorData.detail)) {
              errorMsg = errorData.detail.map((e: any) => e.msg || JSON.stringify(e)).join(', ')
            } else {
              errorMsg = JSON.stringify(errorData.detail)
            }
          }
        } catch {
          errorMsg = 'Failed to export Neo4j data'
        }
      } else if (error.response?.data?.detail) {
        if (typeof error.response.data.detail === 'string') {
          errorMsg = error.response.data.detail
        } else if (Array.isArray(error.response.data.detail)) {
          errorMsg = error.response.data.detail.map((e: any) => e.msg || JSON.stringify(e)).join(', ')
        } else {
          errorMsg = JSON.stringify(error.response.data.detail)
        }
      } else if (error.message) {
        errorMsg = error.message
      }
      
      setError(errorMsg)
    } finally {
      setIsDownloadingNeo4j(false)
    }
  }

  // Loading State
  if (!initialLoadComplete || status === 'connecting') {
    return (
      <AppLayout title="Data Management" description="Generate and import graph data">
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" text={status === 'connecting' ? "Connecting to database..." : "Loading data management..."} />
        </div>
      </AppLayout>
    )
  }

  // Disconnected State - show only no connection component
  if (!isConnected) {
    return (
      <AppLayout title="Data Management" description="Generate and import graph data">
        <NoConnectionInfo description="Connect to your Neo4j database to manage graph data" />
      </AppLayout>
    )
  }

  return (
    <AppLayout title="Data Management" description="Generate and import graph data">
      <div className="space-y-6">

        {/* Current Database Status */}
        {stats && (
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-gradient-to-br from-slate-500 to-slate-700 p-3 shadow-lg">
                <HardDrive className="h-6 w-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold">Database Overview</h2>
                <p className="text-sm text-muted-foreground">Current state of your graph database</p>
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-5">
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-blue-500/25 transition-all duration-300 hover:scale-[1.02]">
                {/* Gradient border */}
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>

                {/* Background gradient overlay */}
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-blue-500/30 via-blue-500/15 to-blue-500/5" />

                <CardContent className="p-5 relative">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-sm font-medium">Total Components</h3>
                    <Database className="h-5 w-5 text-blue-500" />
                  </div>
                  <div className="text-4xl font-bold text-blue-500 mb-2">{stats.total_nodes.toLocaleString()}</div>
                  <p className="text-xs text-muted-foreground">System components in graph</p>
                </CardContent>
              </Card>

              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-purple-500/25 transition-all duration-300 hover:scale-[1.02]">
                {/* Gradient border */}
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>

                {/* Background gradient overlay */}
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-purple-500/30 via-purple-500/15 to-purple-500/5" />

                <CardContent className="p-5 relative">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-sm font-medium">Total Edges</h3>
                    <Zap className="h-5 w-5 text-purple-500" />
                  </div>
                  <div className="text-4xl font-bold text-purple-500 mb-2">
                    {((stats.total_edges || 0) + (stats.total_structural_edges || 0)).toLocaleString()}
                  </div>
                  <p className="text-xs text-muted-foreground">Combined derived & structural edges</p>
                </CardContent>
              </Card>

              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-green-500/25 transition-all duration-300 hover:scale-[1.02]">
                {/* Gradient border */}
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>

                {/* Background gradient overlay */}
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-green-500/30 via-green-500/15 to-green-500/5" />

                <CardContent className="p-5 relative">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-sm font-medium">Component Types</h3>
                    <Layers className="h-5 w-5 text-green-500" />
                  </div>
                  <div className="text-4xl font-bold text-green-500 mb-2">
                    {stats.node_counts ? Object.keys(stats.node_counts).length : 0}
                  </div>
                  <p className="text-xs text-muted-foreground">Unique component types</p>
                </CardContent>
              </Card>

              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-cyan-500/25 transition-all duration-300 hover:scale-[1.02]">
                {/* Gradient border */}
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-cyan-500 via-teal-500 to-emerald-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>

                {/* Background gradient overlay */}
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-cyan-500/30 via-cyan-500/15 to-cyan-500/5" />

                <CardContent className="p-5 relative">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-sm font-medium">Edge Types</h3>
                    <Layers className="h-5 w-5 text-cyan-500" />
                  </div>
                  <div className="text-4xl font-bold text-cyan-500 mb-2">
                    {((stats.edge_counts ? Object.keys(stats.edge_counts).length : 0) + 
                      (stats.structural_edge_counts ? Object.keys(stats.structural_edge_counts).length : 0))}
                  </div>
                  <p className="text-xs text-muted-foreground">Unique edge types</p>
                </CardContent>
              </Card>

              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-orange-500/25 transition-all duration-300 hover:scale-[1.02]">
                {/* Gradient border */}
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-orange-500 via-amber-500 to-yellow-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>

                {/* Background gradient overlay */}
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-orange-500/30 via-orange-500/15 to-orange-500/5" />

                <CardContent className="p-5 relative">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-sm font-medium">Database Status</h3>
                    <CheckCircle2 className="h-5 w-5 text-orange-500" />
                  </div>
                  <div className="text-4xl font-bold text-orange-500 mb-2">
                    {stats.total_nodes === 0 ? 'Empty' : 'Active'}
                  </div>
                  <p className="text-xs text-muted-foreground">Current state</p>
                </CardContent>
              </Card>
            </div>
          </div>
        )}

        {/* Generation Configuration */}
        <div className="space-y-4">
          {/* Messages */}
          {error && (
            <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-red-500/25 transition-all duration-300 animate-in slide-in-from-top-2">
              {/* Gradient border */}
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-red-500 via-rose-500 to-pink-500">
                <div className="w-full h-full bg-background rounded-lg" />
              </div>

              {/* Background gradient overlay */}
              <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-red-500/30 via-red-500/15 to-red-500/5" />

              <CardContent className="p-6 relative">
                <div className="flex items-start gap-4">
                  <div className="rounded-xl bg-red-500/10 p-3 flex-shrink-0">
                    <AlertTriangle className="h-6 w-6 text-red-500" />
                  </div>
                  <div className="flex-1 space-y-2">
                    <div>
                      <h3 className="text-lg font-bold">Operation Failed</h3>
                      <p className="text-sm text-muted-foreground">Unable to complete the requested operation</p>
                    </div>
                    <div className="rounded-lg bg-red-500/10 border border-red-500/20 p-3.5">
                      <p className="text-sm text-red-700 dark:text-red-300 font-medium">{error}</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {success && (
            <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-green-500/25 transition-all duration-300 animate-in slide-in-from-top-2">
              {/* Gradient border */}
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500">
                <div className="w-full h-full bg-background rounded-lg" />
              </div>

              {/* Background gradient overlay */}
              <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-green-500/30 via-green-500/15 to-green-500/5" />

              <CardContent className="p-6 relative">
                <div className="flex items-start gap-4">
                  <div className="rounded-xl bg-green-500/10 p-3 flex-shrink-0">
                    <CheckCircle2 className="h-6 w-6 text-green-500" />
                  </div>
                  <div className="flex-1 space-y-2">
                    <div>
                      <h3 className="text-lg font-bold">Success!</h3>
                      <p className="text-sm text-muted-foreground">Operation completed successfully</p>
                    </div>
                    <div className="rounded-lg bg-green-500/10 border border-green-500/20 p-3.5">
                      <p className="text-sm text-green-700 dark:text-green-300 font-medium">{success}</p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Header Section */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-gradient-to-br from-blue-600 via-purple-600 to-pink-600 p-3 shadow-lg">
                <Settings className="h-6 w-6 text-white" />
              </div>
              <div>
                <h2 className="text-2xl font-bold">Generation Configuration</h2>
                <p className="text-sm text-muted-foreground">Configure graph generation parameters and options</p>
              </div>
            </div>
            <Button
              onClick={handleGenerate}
              disabled={isGenerating || isImporting || isDownloading || isClearing}
              size="lg"
              className="min-w-[180px] h-12 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 hover:from-blue-700 hover:via-purple-700 hover:to-pink-700 shadow-lg hover:shadow-xl transition-all text-base font-semibold"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-5 w-5" />
                  Generate & Import
                </>
              )}
            </Button>
          </div>

          {/* Scale Selection */}
          <div className="space-y-3">
            <div className="grid gap-6 md:grid-cols-3 lg:grid-cols-5">
              {SCALES.map((s) => {
                const isSelected = scale === s.value
                const gradientColor = 
                  s.value === 'tiny' ? 'from-green-500 via-emerald-500 to-teal-500' :
                  s.value === 'small' ? 'from-blue-500 via-indigo-500 to-purple-500' :
                  s.value === 'medium' ? 'from-blue-500 via-purple-500 to-pink-500' :
                  s.value === 'large' ? 'from-orange-500 via-amber-500 to-yellow-500' :
                  'from-red-500 via-rose-500 to-pink-500'
                
                const iconColor =
                  s.value === 'tiny' ? 'from-green-600 to-emerald-600' :
                  s.value === 'small' ? 'from-blue-600 to-indigo-600' :
                  s.value === 'medium' ? 'from-purple-600 to-pink-600' :
                  s.value === 'large' ? 'from-orange-600 to-amber-600' :
                  'from-rose-600 to-pink-600'
                  
                const shadowColor =
                  s.value === 'tiny' ? 'shadow-green-500/30' :
                  s.value === 'small' ? 'shadow-blue-500/30' :
                  s.value === 'medium' ? 'shadow-purple-500/30' :
                  s.value === 'large' ? 'shadow-orange-500/30' :
                  'shadow-rose-500/30'
                  
                const hoverShadowColor =
                  s.value === 'tiny' ? 'hover:shadow-green-500/25' :
                  s.value === 'small' ? 'hover:shadow-blue-500/25' :
                  s.value === 'medium' ? 'hover:shadow-purple-500/25' :
                  s.value === 'large' ? 'hover:shadow-orange-500/25' :
                  'hover:shadow-rose-500/25'
                  
                const borderColor =
                  s.value === 'tiny' ? 'border-green-500 dark:border-green-400' :
                  s.value === 'small' ? 'border-blue-500 dark:border-blue-400' :
                  s.value === 'medium' ? 'border-purple-500 dark:border-purple-400' :
                  s.value === 'large' ? 'border-orange-500 dark:border-orange-400' :
                  'border-rose-500 dark:border-rose-400'
                  
                const hoverBorderColor =
                  s.value === 'tiny' ? 'group-hover:border-emerald-400' :
                  s.value === 'small' ? 'group-hover:border-blue-400' :
                  s.value === 'medium' ? 'group-hover:border-purple-400' :
                  s.value === 'large' ? 'group-hover:border-orange-400' :
                  'group-hover:border-rose-400'
                  
                const bgColor =
                  s.value === 'tiny' ? 'bg-emerald-500 dark:bg-emerald-400' :
                  s.value === 'small' ? 'bg-blue-500 dark:bg-blue-400' :
                  s.value === 'medium' ? 'bg-purple-500 dark:bg-purple-400' :
                  s.value === 'large' ? 'bg-orange-500 dark:bg-orange-400' :
                  'bg-rose-500 dark:bg-rose-400'
                  
                const textColor =
                  s.value === 'tiny' ? 'text-emerald-700 dark:text-emerald-300' :
                  s.value === 'small' ? 'text-blue-700 dark:text-blue-300' :
                  s.value === 'medium' ? 'text-purple-700 dark:text-purple-300' :
                  s.value === 'large' ? 'text-orange-700 dark:text-orange-300' :
                  'text-rose-700 dark:text-rose-300'
                  
                const hoverTextColor =
                  s.value === 'tiny' ? 'group-hover:text-emerald-700 dark:group-hover:text-emerald-300' :
                  s.value === 'small' ? 'group-hover:text-blue-700 dark:group-hover:text-blue-300' :
                  s.value === 'medium' ? 'group-hover:text-purple-700 dark:group-hover:text-purple-300' :
                  s.value === 'large' ? 'group-hover:text-orange-700 dark:group-hover:text-orange-300' :
                  'group-hover:text-rose-700 dark:group-hover:text-rose-300'
                  
                const badgeColor =
                  s.value === 'tiny' ? 'bg-emerald-500/10 text-emerald-700 dark:text-emerald-300 border-emerald-500/20' :
                  s.value === 'small' ? 'bg-blue-500/10 text-blue-700 dark:text-blue-300 border-blue-500/20' :
                  s.value === 'medium' ? 'bg-purple-500/10 text-purple-700 dark:text-purple-300 border-purple-500/20' :
                  s.value === 'large' ? 'bg-orange-500/10 text-orange-700 dark:text-orange-300 border-orange-500/20' :
                  'bg-rose-500/10 text-rose-700 dark:text-rose-300 border-rose-500/20'
                  
                const iconBgColor =
                  s.value === 'tiny' ? 'text-emerald-600 dark:text-emerald-400' :
                  s.value === 'small' ? 'text-blue-600 dark:text-blue-400' :
                  s.value === 'medium' ? 'text-purple-600 dark:text-purple-400' :
                  s.value === 'large' ? 'text-orange-600 dark:text-orange-400' :
                  'text-rose-600 dark:text-rose-400'
                  
                const iconBgLight =
                  s.value === 'tiny' ? 'bg-gradient-to-br from-emerald-100 to-green-50 dark:from-emerald-900/50 dark:to-green-900/30' :
                  s.value === 'small' ? 'bg-gradient-to-br from-blue-100 to-indigo-50 dark:from-blue-900/50 dark:to-indigo-900/30' :
                  s.value === 'medium' ? 'bg-gradient-to-br from-purple-100 to-violet-50 dark:from-purple-900/50 dark:to-violet-900/30' :
                  s.value === 'large' ? 'bg-gradient-to-br from-amber-100 to-orange-50 dark:from-amber-900/50 dark:to-orange-900/30' :
                  'bg-gradient-to-br from-rose-100 to-pink-50 dark:from-rose-900/50 dark:to-pink-900/30'
                  
                const pulseColor =
                  s.value === 'tiny' ? 'bg-emerald-500' :
                  s.value === 'small' ? 'bg-blue-500' :
                  s.value === 'medium' ? 'bg-purple-500' :
                  s.value === 'large' ? 'bg-orange-500' :
                  'bg-rose-500'
                  
                return (
                  <Card
                    key={s.value}
                    className={`group relative cursor-pointer transition-all duration-300 ease-in-out overflow-hidden ${
                      isSelected
                        ? `border-0 shadow-2xl ${shadowColor} scale-[1.02]`
                        : `border-0 hover:shadow-xl ${hoverShadowColor} hover:scale-[1.01]`
                    }`}
                    onClick={() => setScale(s.value)}
                  >
                    {/* Gradient border */}
                    <div className={`absolute inset-0 rounded-lg p-[2px] transition-all duration-300 ${
                      isSelected
                        ? `bg-gradient-to-r ${gradientColor} opacity-100`
                        : `bg-gradient-to-br from-slate-200 via-slate-300 to-slate-200 dark:from-slate-700 dark:via-slate-800 dark:to-slate-700 opacity-100 group-hover:from-slate-300 group-hover:via-slate-400 group-hover:to-slate-300 dark:group-hover:from-slate-600 dark:group-hover:via-slate-700 dark:group-hover:to-slate-600`
                    }`}>
                      <div className="w-full h-full bg-background rounded-lg" />
                    </div>
                    
                    {/* Background gradient overlay - radial from bottom right corner */}
                    <div className={`absolute inset-[2px] rounded-lg transition-opacity duration-300 $${
                      isSelected
                        ? `bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] ${gradientColor.replace('-400', '-500/35').replace('-500', '-500/20').replace('-600', '-500/5')} opacity-100`
                        : `bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] ${gradientColor.replace('-400', '-500/15').replace('-500', '-500/10').replace('-600', '-500/3')} opacity-0 group-hover:opacity-100`
                    }`} />
                    
                    <CardContent className="relative p-6">
                      <div className="space-y-4">
                        {/* Radio button and Icon Section */}
                        <div className="flex items-start gap-4">
                          {/* Radio indicator */}
                          <div className={`relative flex-shrink-0 w-5 h-5 rounded-full border-2 transition-all duration-300 mt-0.5 ${
                            isSelected
                              ? borderColor
                              : `border-slate-300 dark:border-slate-600 ${hoverBorderColor}`
                          }`}>
                            {isSelected && (
                              <div className={`absolute inset-0.5 rounded-full animate-in zoom-in duration-200 ${bgColor}`} />
                            )}
                          </div>
                          
                          {/* Icon */}
                          <div className={`relative rounded-2xl p-3 transition-all duration-300 ${
                            isSelected
                              ? `bg-gradient-to-br ${iconColor} shadow-lg ${shadowColor}`
                              : `${iconBgLight} group-hover:scale-105`
                          }`}>
                            <BarChart3 className={`h-5 w-5 transition-all duration-300 ${
                              isSelected
                                ? 'text-white'
                                : iconBgColor
                            }`} />
                            {isSelected && (
                              <div className={`absolute inset-0 rounded-2xl animate-ping opacity-20 ${pulseColor}`} />
                            )}
                          </div>
                        </div>
                        
                        {/* Content Section */}
                        <div className="space-y-2.5 pl-0">
                          <div className="flex items-center gap-2 flex-wrap">
                            <h3 className={`text-lg font-bold tracking-tight transition-colors duration-200 ${
                              isSelected
                                ? textColor
                                : `text-foreground ${hoverTextColor}`
                            }`}>
                              {s.label}
                            </h3>
                            {isSelected && (
                              <Badge className={`text-xs ${badgeColor}`}>
                                Selected
                              </Badge>
                            )}
                          </div>
                          <p className="text-sm leading-relaxed text-muted-foreground/90">
                            {s.description}
                          </p>
                          <div className="flex flex-wrap gap-1.5">
                            {s.count.split(', ').map((item, idx) => (
                              <Badge key={idx} variant="outline" className="font-mono text-xs">
                                {item}
                              </Badge>
                            ))}
                          </div>
                        </div>
                        
                        {/* Selection indicator bar */}
                        {isSelected && (
                          <div className={`absolute bottom-0 left-0 right-0 h-1.5 bg-gradient-to-r ${iconColor} shadow-sm`} />
                        )}
                      </div>
                    </CardContent>
                  </Card>
                )
              })}
            </div>
          </div>

          {/* Clear Database Option */}
          <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-amber-500/20 transition-all duration-300">
            {/* Gradient border */}
            <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-amber-400 via-orange-500 to-yellow-600">
              <div className="w-full h-full bg-background rounded-lg" />
            </div>

            {/* Background gradient overlay */}
            <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-amber-500/35 via-amber-500/20 to-amber-500/5" />

            <CardContent className="p-5 relative">
              <div className="flex items-start gap-4">
                <div className="rounded-xl bg-amber-500/10 p-2.5 flex-shrink-0 mt-0.5">
                  <AlertTriangle className="h-5 w-5 text-amber-500" />
                </div>
                <div className="flex-1 space-y-2">
                  <div className="flex items-center gap-3">
                    <Checkbox
                      id="clear-first"
                      checked={clearFirst}
                      onCheckedChange={(checked) => setClearFirst(checked as boolean)}
                      className="border-amber-500/50 data-[state=checked]:bg-amber-500 data-[state=checked]:border-amber-500"
                    />
                    <label
                      htmlFor="clear-first"
                      className="text-sm font-semibold leading-none cursor-pointer"
                    >
                      Clear database before generating new graph
                    </label>
                  </div>
                  <p className="text-xs text-muted-foreground pl-7">
                    This will remove all existing data before importing. Uncheck to merge with existing data.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Additional Actions */}
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 p-3 shadow-lg">
              <Sliders className="h-6 w-6 text-white" />
            </div>
            <div>
              <h2 className="text-2xl font-bold">Additional Actions</h2>
              <p className="text-sm text-muted-foreground">Download, import, or manage your graph data</p>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <button
              onClick={handleDownloadGraph}
              disabled={isGenerating || isImporting || isDownloading || isDownloadingNeo4j || isClearing}
              className="group disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:hover:shadow-none"
            >
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-blue-500/20 transition-all duration-300 group-hover:scale-[1.02] group-disabled:hover:scale-100">
                {/* Gradient border */}
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-400 via-indigo-500 to-violet-600">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>

                {/* Background gradient overlay */}
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-blue-500/35 via-blue-500/20 to-blue-500/5" />

                <CardContent className="p-6 relative">
                  <div className="flex flex-col items-center gap-4">
                    <div className="rounded-xl bg-blue-500/10 p-3 group-hover:bg-blue-500/20 transition-colors">
                      {isDownloading ? (
                        <Loader2 className="h-6 w-6 text-blue-500 animate-spin" />
                      ) : (
                        <Download className="h-6 w-6 text-blue-500" />
                      )}
                    </div>
                    <div className="text-center">
                      <h3 className="text-base font-bold mb-1">
                        {isDownloading ? 'Generating...' : 'Generate Sample'}
                      </h3>
                      <p className="text-xs text-muted-foreground">Create new synthetic graph</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </button>

            <button
              onClick={handleDownloadNeo4jData}
              disabled={isDownloadingNeo4j || isGenerating || isImporting || isDownloading || isClearing || !stats || stats.total_nodes === 0}
              className="group disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:hover:shadow-none"
            >
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-green-500/20 transition-all duration-300 group-hover:scale-[1.02] group-disabled:hover:scale-100">
                {/* Gradient border */}
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-green-400 via-emerald-500 to-teal-600">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>

                {/* Background gradient overlay */}
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-green-500/35 via-green-500/20 to-green-500/5" />

                <CardContent className="p-6 relative">
                  <div className="flex flex-col items-center gap-4">
                    <div className="rounded-xl bg-green-500/10 p-3 group-hover:bg-green-500/20 transition-colors">
                      {isDownloadingNeo4j ? (
                        <Loader2 className="h-6 w-6 text-green-500 animate-spin" />
                      ) : (
                        <HardDrive className="h-6 w-6 text-green-500" />
                      )}
                    </div>
                    <div className="text-center">
                      <h3 className="text-base font-bold mb-1">
                        {isDownloadingNeo4j ? 'Exporting...' : 'Export Database'}
                      </h3>
                      <p className="text-xs text-muted-foreground">Download current Neo4j data</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </button>

            <div className="relative">
              <input
                type="file"
                id="import-file"
                accept=".json"
                onChange={handleImportFromFile}
                disabled={isGenerating || isImporting || isDownloading || isDownloadingNeo4j || isClearing}
                className="hidden"
              />
              <button
                onClick={() => document.getElementById('import-file')?.click()}
                disabled={isGenerating || isImporting || isDownloading || isDownloadingNeo4j || isClearing}
                className="group w-full disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:hover:shadow-none"
              >
                <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-purple-500/20 transition-all duration-300 group-hover:scale-[1.02] group-disabled:hover:scale-100">
                  {/* Gradient border */}
                  <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-purple-400 via-fuchsia-500 to-pink-600">
                    <div className="w-full h-full bg-background rounded-lg" />
                  </div>

                  {/* Background gradient overlay */}
                  <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-purple-500/35 via-purple-500/20 to-purple-500/5" />

                  <CardContent className="p-6 relative">
                    <div className="flex flex-col items-center gap-4">
                      <div className="rounded-xl bg-purple-500/10 p-3 group-hover:bg-purple-500/20 transition-colors">
                        {isImporting ? (
                          <Loader2 className="h-6 w-6 text-purple-500 animate-spin" />
                        ) : (
                          <Upload className="h-6 w-6 text-purple-500" />
                        )}
                      </div>
                      <div className="text-center">
                        <h3 className="text-base font-bold mb-1">
                          {isImporting ? 'Importing...' : 'Import File'}
                        </h3>
                        <p className="text-xs text-muted-foreground">Upload graph data file</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </button>
            </div>

            <button
              onClick={handleClear}
              disabled={isClearing || isGenerating || isImporting || isDownloading || isDownloadingNeo4j || !stats || stats.total_nodes === 0}
              className="group disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 disabled:hover:shadow-none"
            >
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-red-500/20 transition-all duration-300 group-hover:scale-[1.02] group-disabled:hover:scale-100">
                {/* Gradient border */}
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-red-400 via-rose-500 to-pink-600">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>

                {/* Background gradient overlay */}
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-red-500/35 via-red-500/20 to-red-500/5" />

                <CardContent className="p-6 relative">
                  <div className="flex flex-col items-center gap-4">
                    <div className="rounded-xl bg-red-500/10 p-3 group-hover:bg-red-500/20 transition-colors">
                      {isClearing ? (
                        <Loader2 className="h-6 w-6 text-red-500 animate-spin" />
                      ) : (
                        <Trash2 className="h-6 w-6 text-red-500" />
                      )}
                    </div>
                    <div className="text-center">
                      <h3 className="text-base font-bold mb-1">
                        {isClearing ? 'Clearing...' : 'Clear Database'}
                      </h3>
                      <p className="text-xs text-muted-foreground">Remove all graph data</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </button>
          </div>
        </div>
      </div>
    </AppLayout>
  )
}
