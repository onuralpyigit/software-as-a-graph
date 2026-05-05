"use client"

import { AppLayout } from "@/components/layout/app-layout"
import { ConnectionForm } from "@/components/settings/connection-form"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { useState, useEffect } from "react"
import { 
  CheckCircle2, XCircle, Loader2, Server, Database, AlertCircle, 
  RefreshCw, Info, Activity, Package
} from "lucide-react"
import { apiClient } from "@/lib/api/client"
import { useConnection } from "@/lib/stores/connection-store"

export default function SettingsPage() {
  const { status: neo4jStatus, stats, error: neo4jError, disconnect: disconnectNeo4j } = useConnection()
  
  // Helper to get default API URL dynamically
  const getDefaultApiUrl = () => {
    if (typeof window !== 'undefined') {
      // Check localStorage first
      const saved = localStorage.getItem('api-base-url');
      if (saved) return saved;
      
      // Auto-detect: use same hostname as frontend with port 8000
      const hostname = window.location.hostname;
      const protocol = window.location.protocol;
      return `${protocol}//${hostname}:8000`;
    }
    return 'http://localhost:8000';
  };
  
  const [apiBaseUrl, setApiBaseUrl] = useState(getDefaultApiUrl())
  const [apiStatus, setApiStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking')
  const [apiError, setApiError] = useState<string | null>(null)
  const [lastCheckedUrl, setLastCheckedUrl] = useState<string>('')
  const [isTestingConnection, setIsTestingConnection] = useState(false)
  const [mounted, setMounted] = useState(false)

  // Check API connection on mount
  useEffect(() => {
    setMounted(true)
    checkApiConnection()
  }, [])

  const checkApiConnection = async () => {
    setApiStatus('checking')
    setApiError(null)

    try {
      apiClient.setBaseURL(apiBaseUrl)

      // Add timeout to the fetch request for better UX
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 10000) // 10 second timeout

      const response = await fetch(`${apiBaseUrl}/health`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
      })

      clearTimeout(timeoutId)

      if (response.ok) {
        setApiStatus('connected')
        setLastCheckedUrl(apiBaseUrl)
        setApiError(null)
      } else {
        setApiStatus('disconnected')
        setApiError(`API returned error status: ${response.status} ${response.statusText}`)
      }
    } catch (error: any) {
      setApiStatus('disconnected')

      // Provide more specific error messages
      if (error.name === 'AbortError') {
        setApiError('Connection timeout - API server is not responding')
      } else if (error.message?.includes('fetch')) {
        setApiError('Cannot reach API server - Check if the service is running')
      } else {
        setApiError(error.message || 'Connection failed')
      }
    }
  }

  const handleSaveApiUrl = async () => {
    setIsTestingConnection(true)

    // If URL changed and Neo4j was connected, disconnect it
    if (apiBaseUrl !== lastCheckedUrl && neo4jStatus === 'connected') {
      disconnectNeo4j()
    }

    localStorage.setItem('api-base-url', apiBaseUrl)
    await checkApiConnection()
    setIsTestingConnection(false)
  }

  // Detect if URL has changed since last successful connection
  const hasUrlChanged = apiBaseUrl !== lastCheckedUrl && lastCheckedUrl !== ''

  if (!mounted) {
    return (
      <AppLayout title="Settings" description="Configure your analysis environment">
        <div className="space-y-5">
          {/* Status tiles skeleton */}
          <div className="grid gap-3 grid-cols-1 sm:grid-cols-2">
            {Array.from({ length: 2 }).map((_, i) => (
              <div key={i} className="rounded-xl border border-border bg-muted/20 p-4 space-y-2">
                <Skeleton className="h-3 w-24" />
                <Skeleton className="h-8 w-28" />
                <Skeleton className="h-2.5 w-20" />
              </div>
            ))}
          </div>
          {/* Config card skeleton */}
          <div className="rounded-xl border border-border bg-muted/20 p-5 space-y-4">
            <div className="flex items-center gap-3">
              <Skeleton className="h-8 w-8 rounded-lg shrink-0" />
              <div className="space-y-1.5">
                <Skeleton className="h-4 w-36" />
                <Skeleton className="h-3 w-48" />
              </div>
            </div>
            <Skeleton className="h-9 w-full rounded-md" />
            <Skeleton className="h-9 w-32 rounded-md" />
          </div>
          {/* Neo4j config card skeleton */}
          <div className="rounded-xl border border-border bg-muted/20 p-5 space-y-4">
            <div className="flex items-center gap-3">
              <Skeleton className="h-8 w-8 rounded-lg shrink-0" />
              <div className="space-y-1.5">
                <Skeleton className="h-4 w-40" />
                <Skeleton className="h-3 w-44" />
              </div>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <Skeleton className="h-9 w-full rounded-md" />
              <Skeleton className="h-9 w-full rounded-md" />
            </div>
            <Skeleton className="h-9 w-28 rounded-md" />
          </div>
        </div>
      </AppLayout>
    )
  }

  return (
    <AppLayout title="Settings" description="Configure your analysis environment">
      <div className="space-y-5">

        {/* Connection Status Tiles */}
        <div className="grid gap-3 grid-cols-1 sm:grid-cols-2">

          {/* API Status Tile */}
          {(() => {
            const c = apiStatus === 'connected'
              ? { text: 'text-green-400', border: 'border-green-500/20', bg: 'bg-green-500/[0.07]', ring: 'bg-green-500/10', glow: 'bg-green-500' }
              : apiStatus === 'checking'
              ? { text: 'text-blue-400', border: 'border-blue-500/20', bg: 'bg-blue-500/[0.07]', ring: 'bg-blue-500/10', glow: 'bg-blue-500' }
              : { text: 'text-red-400', border: 'border-red-500/20', bg: 'bg-red-500/[0.07]', ring: 'bg-red-500/10', glow: 'bg-red-500' }
            return (
              <div className={`relative overflow-hidden rounded-xl border ${c.border} ${c.bg} p-4`}>
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className="text-xs text-muted-foreground font-medium">Analysis API</p>
                    <p className={`text-[1.65rem] font-bold leading-tight tracking-tight ${c.text}`}>
                      {apiStatus === 'connected' ? 'Connected' : apiStatus === 'checking' ? 'Checking...' : 'Offline'}
                    </p>
                    <p className="text-[11px] text-muted-foreground mt-0.5">Backend Service</p>
                  </div>
                  <div className={`shrink-0 rounded-lg ${c.ring} p-2`}>
                    {apiStatus === 'connected'
                      ? <CheckCircle2 className={`h-4 w-4 ${c.text}`} />
                      : apiStatus === 'checking'
                      ? <Loader2 className={`h-4 w-4 ${c.text} animate-spin`} />
                      : <XCircle className={`h-4 w-4 ${c.text}`} />}
                  </div>
                </div>
                <div className={`pointer-events-none absolute -bottom-5 -right-5 h-16 w-16 rounded-full blur-2xl opacity-20 ${c.glow}`} />
              </div>
            )
          })()}

          {/* Neo4j Status Tile */}
          {(() => {
            const c = neo4jStatus === 'connected'
              ? { text: 'text-green-400', border: 'border-green-500/20', bg: 'bg-green-500/[0.07]', ring: 'bg-green-500/10', glow: 'bg-green-500' }
              : neo4jStatus === 'connecting'
              ? { text: 'text-blue-400', border: 'border-blue-500/20', bg: 'bg-blue-500/[0.07]', ring: 'bg-blue-500/10', glow: 'bg-blue-500' }
              : neo4jStatus === 'error'
              ? { text: 'text-red-400', border: 'border-red-500/20', bg: 'bg-red-500/[0.07]', ring: 'bg-red-500/10', glow: 'bg-red-500' }
              : { text: 'text-slate-400', border: 'border-slate-500/20', bg: 'bg-slate-500/[0.07]', ring: 'bg-slate-500/10', glow: 'bg-slate-500' }
            return (
              <div className={`relative overflow-hidden rounded-xl border ${c.border} ${c.bg} p-4`}>
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className="text-xs text-muted-foreground font-medium">Neo4j Database</p>
                    <p className={`text-[1.65rem] font-bold leading-tight tracking-tight ${c.text}`}>
                      {neo4jStatus === 'connected' ? 'Connected' : neo4jStatus === 'connecting' ? 'Connecting...' : neo4jStatus === 'error' ? 'Error' : 'Not Connected'}
                    </p>
                    <p className="text-[11px] text-muted-foreground mt-0.5">Graph Storage</p>
                  </div>
                  <div className={`shrink-0 rounded-lg ${c.ring} p-2`}>
                    {neo4jStatus === 'connected'
                      ? <CheckCircle2 className={`h-4 w-4 ${c.text}`} />
                      : neo4jStatus === 'connecting'
                      ? <Loader2 className={`h-4 w-4 ${c.text} animate-spin`} />
                      : <XCircle className={`h-4 w-4 ${c.text}`} />}
                  </div>
                </div>
                <div className={`pointer-events-none absolute -bottom-5 -right-5 h-16 w-16 rounded-full blur-2xl opacity-20 ${c.glow}`} />
              </div>
            )
          })()}

        </div>

        {/* Data Stats — only when both connected */}
        {apiStatus === 'connected' && neo4jStatus === 'connected' && stats && (
          <Card className="bg-background">
            <CardHeader className="pb-1 flex flex-row items-center justify-between space-y-0">
              <div className="flex items-center gap-2.5">
                <div className="rounded-lg bg-purple-500/10 p-1.5">
                  <Database className="h-4 w-4 text-purple-400" />
                </div>
                <div>
                  <CardTitle className="text-sm font-semibold">Database Statistics</CardTitle>
                  <p className="text-[11px] text-muted-foreground">Current graph data overview</p>
                </div>
              </div>
            </CardHeader>
            <CardContent className="pt-1">
              <div className="grid gap-3 sm:grid-cols-2">
                <div className="relative overflow-hidden rounded-xl border border-blue-500/20 bg-blue-500/[0.07] p-4">
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <p className="text-xs text-muted-foreground font-medium">Components</p>
                      <p className="text-[1.65rem] font-bold leading-tight tracking-tight text-blue-400">{stats.total_nodes.toLocaleString()}</p>
                      <p className="text-[11px] text-muted-foreground mt-0.5">Nodes in graph</p>
                    </div>
                    <div className="shrink-0 rounded-lg bg-blue-500/10 p-2">
                      <Package className="h-4 w-4 text-blue-400" />
                    </div>
                  </div>
                  <div className="pointer-events-none absolute -bottom-5 -right-5 h-16 w-16 rounded-full blur-2xl opacity-20 bg-blue-500" />
                </div>
                <div className="relative overflow-hidden rounded-xl border border-purple-500/20 bg-purple-500/[0.07] p-4">
                  <div className="flex items-start justify-between gap-2">
                    <div>
                      <p className="text-xs text-muted-foreground font-medium">Relationships</p>
                      <p className="text-[1.65rem] font-bold leading-tight tracking-tight text-purple-400">{stats.total_edges.toLocaleString()}</p>
                      <p className="text-[11px] text-muted-foreground mt-0.5">Edges in graph</p>
                    </div>
                    <div className="shrink-0 rounded-lg bg-purple-500/10 p-2">
                      <Activity className="h-4 w-4 text-purple-400" />
                    </div>
                  </div>
                  <div className="pointer-events-none absolute -bottom-5 -right-5 h-16 w-16 rounded-full blur-2xl opacity-20 bg-purple-500" />
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Configuration Section */}
        <div className="space-y-4">

          {/* API Server Configuration */}
          <Card className="bg-background">
            <CardHeader className="pb-1 flex flex-row items-center justify-between space-y-0">
              <div className="flex items-center gap-2.5">
                <div className={`rounded-lg p-1.5 ${
                  apiStatus === 'connected' ? 'bg-green-500/10' : apiStatus === 'checking' ? 'bg-blue-500/10' : 'bg-red-500/10'
                }`}>
                  <Server className={`h-4 w-4 ${
                    apiStatus === 'connected' ? 'text-green-400' : apiStatus === 'checking' ? 'text-blue-400' : 'text-red-400'
                  }`} />
                </div>
                <div>
                  <CardTitle className="text-sm font-semibold">Graph Analysis API</CardTitle>
                  <p className="text-[11px] text-muted-foreground">Backend analysis service endpoint</p>
                </div>
              </div>
              {apiStatus === 'connected' && (
                <Badge className="bg-green-500/10 text-green-400 border-green-500/20 text-[11px] px-2">Connected</Badge>
              )}
            </CardHeader>
            <CardContent className="pt-1 space-y-4">
              <div className="space-y-2">
                <Label htmlFor="api-url">Service URL</Label>
                <div className="flex gap-2">
                  <Input
                    id="api-url"
                    placeholder="http://localhost:8000"
                    value={apiBaseUrl}
                    onChange={(e) => setApiBaseUrl(e.target.value)}
                    className={`font-mono text-sm ${
                      hasUrlChanged ? 'border-amber-500 focus-visible:ring-amber-500' : ''
                    }`}
                    onKeyDown={(e) => e.key === 'Enter' && handleSaveApiUrl()}
                  />
                  <Button
                    onClick={handleSaveApiUrl}
                    disabled={isTestingConnection || apiStatus === 'checking'}
                    size="default"
                  >
                    {isTestingConnection || apiStatus === 'checking' ? (
                      <><Loader2 className="mr-2 h-4 w-4 animate-spin" />Testing...</>
                    ) : (
                      <><RefreshCw className="mr-2 h-4 w-4" />{hasUrlChanged ? 'Test' : 'Save & Test'}</>
                    )}
                  </Button>
                </div>
                {hasUrlChanged && (
                  <p className="text-xs text-amber-600 dark:text-amber-500">
                    URL has changed - Click "Test" to verify the connection
                  </p>
                )}
                <p className="text-xs text-muted-foreground">
                  Backend service that processes graph generation, analysis, and query requests
                </p>
              </div>

              {apiError && (
                <div className="rounded-xl border border-red-500/20 bg-red-500/[0.07] p-4">
                  <div className="flex items-start gap-3">
                    <div className="rounded-lg bg-red-500/10 p-1.5 mt-0.5">
                      <AlertCircle className="h-4 w-4 text-red-400 flex-shrink-0" />
                    </div>
                    <div className="space-y-1 text-sm">
                      <p className="font-medium">Connection Failed</p>
                      <p className="text-muted-foreground">{apiError}</p>
                    </div>
                  </div>
                </div>
              )}
              {apiStatus === 'connected' && !hasUrlChanged && (
                <div className="rounded-xl border border-green-500/20 bg-green-500/[0.07] p-4">
                  <div className="flex items-start gap-3">
                    <div className="rounded-lg bg-green-500/10 p-1.5 mt-0.5">
                      <CheckCircle2 className="h-4 w-4 text-green-400 flex-shrink-0" />
                    </div>
                    <div className="space-y-1 text-sm">
                      <p className="font-medium">Connected successfully</p>
                      <p className="text-muted-foreground">You can now configure the database connection below.</p>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Warning when API not connected */}
          {apiStatus !== 'connected' && (
            <Card className="bg-background">
              <CardHeader className="pb-1 flex flex-row items-center justify-between space-y-0">
                <div className="flex items-center gap-2.5">
                  <div className="rounded-lg bg-purple-500/10 p-1.5">
                    <Database className="h-4 w-4 text-purple-400" />
                  </div>
                  <div>
                    <CardTitle className="text-sm font-semibold">Graph Database Connection</CardTitle>
                    <p className="text-[11px] text-muted-foreground">Connect to the analysis service first</p>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="pt-1 space-y-4">
                <div className="rounded-xl border border-purple-500/20 bg-purple-500/[0.07] p-4">
                  <div className="flex items-start gap-3">
                    <div className="rounded-lg bg-purple-500/10 p-1.5 mt-0.5">
                      <Info className="h-4 w-4 text-purple-400 flex-shrink-0" />
                    </div>
                    <div className="space-y-1 text-sm">
                      <p className="font-medium">Neo4j database configuration is currently unavailable</p>
                      <p className="text-muted-foreground">
                        The Neo4j database configuration requires an active connection to the analysis service.
                        Please configure and test the API connection above to proceed.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="space-y-2">
                  <p className="text-sm font-medium text-foreground">Required steps:</p>
                  <ol className="space-y-2 text-sm text-muted-foreground ml-1">
                    <li className="flex items-start gap-2">
                      <Badge className="mt-0.5 h-5 w-5 rounded-full flex items-center justify-center p-0 text-xs bg-purple-500/10 text-purple-400 border-purple-500/20">1</Badge>
                      <span>Enter the Graph Analysis API URL above (e.g., http://localhost:8000)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <Badge className="mt-0.5 h-5 w-5 rounded-full flex items-center justify-center p-0 text-xs bg-purple-500/10 text-purple-400 border-purple-500/20">2</Badge>
                      <span>Click "Save & Test" to verify the connection</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <Badge className="mt-0.5 h-5 w-5 rounded-full flex items-center justify-center p-0 text-xs bg-purple-500/10 text-purple-400 border-purple-500/20">3</Badge>
                      <span>Once connected, the Neo4j configuration form will appear below</span>
                    </li>
                  </ol>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Neo4j Connection Form - Only show when API connected */}
          {apiStatus === 'connected' && <ConnectionForm />}
        </div>
      </div>
    </AppLayout>
  )
}
