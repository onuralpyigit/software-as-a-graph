"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ArrowLeft, Shield, AlertTriangle, CheckCircle, Info, ShieldAlert, ShieldCheck, GitBranch } from "lucide-react"
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

interface Component {
  id: string
  name: string
  type: string
  in_degree: number
  out_degree: number
  is_critical?: boolean
  bridge_score?: number
}

interface ComponentRedundancyStats {
  total_components: number
  spof_count: number
  spof_percentage?: number
  redundant_count: number
  redundancy_percentage?: number
  resilience_score?: number
  interpretation?: string
  category?: string
  health?: string
  single_points_of_failure?: Component[]
  bridge_components?: Component[]
}

export default function ComponentRedundancyPage() {
  const router = useRouter()
  const { status, stats: graphStats, initialLoadComplete } = useConnection()
  const [stats, setStats] = useState<ComponentRedundancyStats | null>(null)
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
      const result = await apiClient.getComponentRedundancyStats()
      setStats(result.stats)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load statistics')
    } finally {
      setLoading(false)
    }
  }

  // Get interpretation color and icon
  const getInterpretation = () => {
    if (!stats || !stats.health) return { color: 'text-gray-500', icon: Info, label: 'Unknown' }
    
    switch (stats.health) {
      case 'good':
        return { 
          color: 'text-green-500', 
          icon: CheckCircle, 
          label: 'Highly Resilient'
        }
      case 'fair':
        return { 
          color: 'text-blue-500', 
          icon: Info, 
          label: 'Moderately Resilient'
        }
      case 'moderate':
        return { 
          color: 'text-yellow-500', 
          icon: AlertTriangle, 
          label: 'Limited Resilience'
        }
      case 'poor':
        return { 
          color: 'text-red-500', 
          icon: AlertTriangle, 
          label: 'Low Resilience'
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

  // Get resilience score color
  const getResilienceColor = () => {
    if (!stats || stats.resilience_score === undefined) return 'text-gray-500'
    if (stats.resilience_score >= 70) return 'text-green-500'
    if (stats.resilience_score >= 50) return 'text-blue-500'
    if (stats.resilience_score >= 30) return 'text-yellow-500'
    return 'text-red-500'
  }

  // Loading State
  if (!initialLoadComplete || status === 'connecting') {
    return (
      <AppLayout title="Component Redundancy" description="Analyze system resilience and fault tolerance">
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" text="Connecting to database..." />
        </div>
      </AppLayout>
    )
  }

  // Disconnected State
  if (!isConnected) {
    return (
      <AppLayout title="Component Redundancy" description="Analyze system resilience and fault tolerance">
        <NoConnectionInfo />
      </AppLayout>
    )
  }

  return (
    <AppLayout title="Component Redundancy" description="Analyze system resilience and fault tolerance">
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
          <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-cyan-500 via-blue-500 to-indigo-500">
            <div className="w-full h-full rounded-lg bg-gradient-to-r from-cyan-600 via-blue-600 to-indigo-600" />
          </div>
          
          <CardContent className="p-8 relative text-white">
            <div className="flex items-center gap-4 mb-4">
              <div className="rounded-xl bg-white/20 p-3">
                <Shield className="h-8 w-8" />
              </div>
              <div className="flex-1">
                <h3 className="text-3xl font-bold">Component Redundancy Analysis</h3>
                <p className="text-white/90 mt-2">
                  Understanding system resilience through redundancy and fault tolerance
                </p>
              </div>
            </div>
            
            <div className="bg-white/10 rounded-lg p-4 mt-4">
              <div className="flex items-start gap-3">
                <Info className="h-5 w-5 mt-0.5 flex-shrink-0" />
                <div className="text-sm">
                  <p className="font-semibold mb-1">What is Component Redundancy?</p>
                  <p className="text-white/90">
                    Component redundancy analysis identifies <strong>Single Points of Failure (SPOFs)</strong> - 
                    critical components whose failure would disconnect parts of the system. It also measures 
                    <strong> redundant paths</strong> and calculates a <strong>resilience score</strong> to 
                    assess overall system fault tolerance.
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Loading/Error States */}
        {loading && (
          <div className="flex items-center justify-center py-12">
            <LoadingSpinner size="lg" text="Analyzing component redundancy..." />
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
              {/* Resilience Score */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-cyan-500/25 transition-all duration-300 transform hover:scale-[1.02]">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-cyan-500 via-blue-500 to-indigo-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-cyan-500/30 via-cyan-500/15 to-cyan-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Resilience Score</CardTitle>
                  <div className="rounded-xl bg-cyan-500/10 p-2.5">
                    <ShieldCheck className="h-4 w-4 text-cyan-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className={`text-3xl font-bold ${getResilienceColor()}`}>
                    {stats.resilience_score !== undefined ? stats.resilience_score.toFixed(1) : 'N/A'}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Out of 100 - Higher is better
                  </p>
                  <Progress value={stats.resilience_score ?? 0} className="h-1 mt-2" />
                </CardContent>
              </Card>

              {/* Single Points of Failure */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-red-500/25 transition-all duration-300 transform hover:scale-[1.02]">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-red-500 via-rose-500 to-pink-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-red-500/30 via-red-500/15 to-red-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">SPOFs</CardTitle>
                  <div className="rounded-xl bg-red-500/10 p-2.5">
                    <ShieldAlert className="h-4 w-4 text-red-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-red-500">
                    {stats.spof_count}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {stats.spof_percentage !== undefined ? stats.spof_percentage.toFixed(1) : '0.0'}% - Critical failures
                  </p>
                  <Progress value={stats.spof_percentage ?? 0} className="h-1 mt-2" />
                </CardContent>
              </Card>

              {/* Redundant Components */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-green-500/25 transition-all duration-300 transform hover:scale-[1.02]">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-green-500/30 via-green-500/15 to-green-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Redundant</CardTitle>
                  <div className="rounded-xl bg-green-500/10 p-2.5">
                    <GitBranch className="h-4 w-4 text-green-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-green-500">
                    {stats.redundant_count}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    {stats.redundancy_percentage !== undefined ? stats.redundancy_percentage.toFixed(1) : '0.0'}% - Multiple paths
                  </p>
                  <Progress value={stats.redundancy_percentage ?? 0} className="h-1 mt-2" />
                </CardContent>
              </Card>

              {/* Total Components */}
              <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-purple-500/25 transition-all duration-300 transform hover:scale-[1.02]">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-purple-500 via-fuchsia-500 to-pink-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-purple-500/30 via-purple-500/15 to-purple-500/5" />
                
                <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">Total Components</CardTitle>
                  <div className="rounded-xl bg-purple-500/10 p-2.5">
                    <Shield className="h-4 w-4 text-purple-500" />
                  </div>
                </CardHeader>
                <CardContent className="relative">
                  <div className="text-3xl font-bold text-purple-500">
                    {stats.total_components}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">
                    Components analyzed
                  </p>
                </CardContent>
              </Card>
            </div>

            {/* Health Assessment */}
            <Card className="relative overflow-hidden border-0 shadow-lg">
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-cyan-500 via-blue-500 to-indigo-500">
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              <div className="absolute inset-[2px] rounded-lg bg-background" />
              
              <CardHeader className="relative">
                <div className="flex items-center gap-2">
                  <div className="rounded-xl bg-cyan-500/10 p-2.5">
                    <Shield className="h-5 w-5 text-cyan-500" />
                  </div>
                  <div>
                    <CardTitle>Resilience Assessment</CardTitle>
                    <CardDescription>
                      Overall system fault tolerance and redundancy evaluation
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>
              
              <CardContent className="relative">
                <Alert className={`border-2 ${interpretation.color.replace('text-', 'border-')}`}>
                  <Icon className={`h-5 w-5 ${interpretation.color}`} />
                  <AlertTitle className="text-lg">
                    {interpretation.label}{stats.category ? `: ${stats.category.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}` : ''}
                  </AlertTitle>
                  <AlertDescription className="text-base">
                    {stats.interpretation || 'No interpretation available'}
                  </AlertDescription>
                </Alert>
              </CardContent>
            </Card>

            {/* Single Points of Failure Table */}
            {stats.single_points_of_failure && stats.single_points_of_failure.length > 0 && (
              <Card className="relative overflow-hidden border-0 shadow-lg">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-red-500 via-rose-500 to-pink-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-background" />
                
                <CardHeader className="relative">
                  <div className="flex items-center gap-2">
                    <div className="rounded-xl bg-red-500/10 p-2.5">
                      <ShieldAlert className="h-5 w-5 text-red-500" />
                    </div>
                    <div>
                      <CardTitle>Single Points of Failure</CardTitle>
                      <CardDescription>
                        Critical components whose failure would disconnect parts of the system
                      </CardDescription>
                    </div>
                  </div>
                </CardHeader>
                
                <CardContent className="relative">
                  <div className="rounded-lg border overflow-hidden">
                    <Table>
                      <TableHeader>
                        <TableRow className="bg-muted/50">
                          <TableHead className="font-semibold">Component</TableHead>
                          <TableHead className="font-semibold">Type</TableHead>
                          <TableHead className="text-right font-semibold">In-Degree</TableHead>
                          <TableHead className="text-right font-semibold">Out-Degree</TableHead>
                          <TableHead className="text-right font-semibold">Total</TableHead>
                          <TableHead className="text-center font-semibold">Status</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {stats.single_points_of_failure?.map((component, index) => (
                          <TableRow key={component.id} className="hover:bg-muted/50 cursor-pointer group" onClick={() => router.push(`/explorer?node=${encodeURIComponent(component.id)}`)}>
                            <TableCell className="font-medium max-w-[300px] truncate group-hover:underline transition-all" title={component.name}>
                              {component.name}
                            </TableCell>
                            <TableCell>
                              <Badge variant="outline" className="text-xs">
                                {component.type}
                              </Badge>
                            </TableCell>
                            <TableCell className="text-right">{component.in_degree}</TableCell>
                            <TableCell className="text-right">{component.out_degree}</TableCell>
                            <TableCell className="text-right font-semibold">
                              {component.in_degree + component.out_degree}
                            </TableCell>
                            <TableCell className="text-center">
                              <Badge variant="destructive" className="text-xs">
                                CRITICAL
                              </Badge>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Bridge Components Table */}
            {stats.bridge_components && stats.bridge_components.length > 0 && (
              <Card className="relative overflow-hidden border-0 shadow-lg">
                <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-amber-500 via-orange-500 to-yellow-500">
                  <div className="w-full h-full bg-background rounded-lg" />
                </div>
                <div className="absolute inset-[2px] rounded-lg bg-background" />
                
                <CardHeader className="relative">
                  <div className="flex items-center gap-2">
                    <div className="rounded-xl bg-amber-500/10 p-2.5">
                      <GitBranch className="h-5 w-5 text-amber-500" />
                    </div>
                    <div>
                      <CardTitle>Bridge Components</CardTitle>
                      <CardDescription>
                        Components through which many paths flow - critical for system connectivity
                      </CardDescription>
                    </div>
                  </div>
                </CardHeader>
                
                <CardContent className="relative">
                  <div className="rounded-lg border overflow-hidden">
                    <Table>
                      <TableHeader>
                        <TableRow className="bg-muted/50">
                          <TableHead className="font-semibold">Component</TableHead>
                          <TableHead className="font-semibold">Type</TableHead>
                          <TableHead className="text-right font-semibold">Bridge Score</TableHead>
                          <TableHead className="text-right font-semibold">In-Degree</TableHead>
                          <TableHead className="text-right font-semibold">Out-Degree</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {stats.bridge_components?.map((component, index) => (
                          <TableRow key={component.id} className="hover:bg-muted/50 cursor-pointer group" onClick={() => router.push(`/explorer?node=${encodeURIComponent(component.id)}`)}>
                            <TableCell className="font-medium max-w-[300px] truncate group-hover:underline transition-all" title={component.name}>
                              {component.name}
                            </TableCell>
                            <TableCell>
                              <Badge variant="outline" className="text-xs">
                                {component.type}
                              </Badge>
                            </TableCell>
                            <TableCell className="text-right">
                              <Badge variant="secondary" className="text-xs">
                                {component.bridge_score}
                              </Badge>
                            </TableCell>
                            <TableCell className="text-right">{component.in_degree}</TableCell>
                            <TableCell className="text-right">{component.out_degree}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
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
