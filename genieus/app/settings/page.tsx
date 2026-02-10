"use client"

import { AppLayout } from "@/components/layout/app-layout"
import { ConnectionForm } from "@/components/settings/connection-form"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
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

  // Check API connection on mount
  useEffect(() => {
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

  return (
    <AppLayout title="Settings" description="Configure your analysis environment">
      <div className="space-y-6">
        
        {/* Connection Status Cards */}
        <div className="grid gap-4 md:grid-cols-2">
          {/* API Status Card */}
          <Card className={`relative overflow-hidden border-0 shadow-lg hover:shadow-xl transition-all duration-300 ${
            apiStatus === 'connected'
              ? 'hover:shadow-green-500/25'
              : apiStatus === 'checking'
              ? 'hover:shadow-blue-500/25'
              : 'hover:shadow-red-500/25'
          }`}>
            {/* Gradient border */}
            <div className={`absolute inset-0 rounded-lg p-[2px] ${
              apiStatus === 'connected'
                ? 'bg-gradient-to-r from-green-400 via-emerald-500 to-teal-600'
                : apiStatus === 'checking'
                ? 'bg-gradient-to-r from-blue-400 via-indigo-500 to-violet-600'
                : 'bg-gradient-to-r from-red-400 via-rose-500 to-pink-600'
            }`}>
              <div className="w-full h-full bg-background rounded-lg" />
            </div>
            
            {/* Background gradient overlay */}
            <div className={`absolute inset-[2px] rounded-lg ${
              apiStatus === 'connected'
                ? 'bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-green-500/35 via-green-500/20 to-green-500/5'
                : apiStatus === 'checking'
                ? 'bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-blue-500/35 via-blue-500/20 to-blue-500/5'
                : 'bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-red-500/35 via-red-500/20 to-red-500/5'
            }`} />
            
            <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-3">
              <CardTitle className="text-sm font-semibold">Analysis API Status</CardTitle>
              <div className={`rounded-xl p-2.5 transition-all duration-300 ${
                apiStatus === 'connected'
                  ? 'bg-green-100 dark:bg-green-900 shadow-green-200 dark:shadow-green-800 shadow-sm'
                  : apiStatus === 'checking'
                  ? 'bg-blue-100 dark:bg-blue-900 shadow-blue-200 dark:shadow-blue-800 shadow-sm'
                  : 'bg-red-100 dark:bg-red-900 shadow-red-200 dark:shadow-red-800 shadow-sm'
              }`}>
                <Server className={`h-4 w-4 transition-colors duration-300 ${
                  apiStatus === 'connected'
                    ? 'text-green-600 dark:text-green-400'
                    : apiStatus === 'checking'
                    ? 'text-blue-600 dark:text-blue-400'
                    : 'text-red-600 dark:text-red-400'
                }`} />
              </div>
            </CardHeader>
            <CardContent className="relative">
              <div className="flex items-center justify-between">
                <div>
                  <div className={`text-2xl font-bold transition-colors duration-300 ${
                    apiStatus === 'connected'
                      ? 'text-green-600 dark:text-green-400'
                      : apiStatus === 'checking'
                      ? 'text-blue-600 dark:text-blue-400'
                      : 'text-red-600 dark:text-red-400'
                  }`}>
                    {apiStatus === 'connected' ? 'Connected' : apiStatus === 'checking' ? 'Checking...' : 'Offline'}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1.5">Backend Service</p>
                </div>
                {apiStatus === 'connected' ? (
                  <CheckCircle2 className="h-9 w-9 text-green-500 animate-in zoom-in duration-300" />
                ) : apiStatus === 'checking' ? (
                  <Loader2 className="h-9 w-9 text-blue-500 animate-spin" />
                ) : (
                  <XCircle className="h-9 w-9 text-red-500 animate-in zoom-in duration-300" />
                )}
              </div>
            </CardContent>
          </Card>

          {/* Neo4j Status Card */}
          <Card className={`relative overflow-hidden border-0 shadow-lg hover:shadow-xl transition-all duration-300 ${
            neo4jStatus === 'connected'
              ? 'hover:shadow-green-500/25'
              : neo4jStatus === 'connecting'
              ? 'hover:shadow-blue-500/25'
              : neo4jStatus === 'disconnected' && neo4jError
              ? 'hover:shadow-red-500/25'
              : 'hover:shadow-slate-500/25'
          }`}>
            {/* Gradient border */}
            <div className={`absolute inset-0 rounded-lg p-[2px] ${
              neo4jStatus === 'connected'
                ? 'bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500'
                : neo4jStatus === 'connecting'
                ? 'bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500'
                : neo4jStatus === 'disconnected' && neo4jError
                ? 'bg-gradient-to-r from-red-500 via-rose-500 to-pink-500'
                : 'bg-gradient-to-br from-slate-200 via-slate-300 to-slate-200 dark:from-slate-700 dark:via-slate-800 dark:to-slate-700'
            }`}>
              <div className="w-full h-full bg-background rounded-lg" />
            </div>
            
            {/* Background gradient overlay */}
            <div className={`absolute inset-[2px] rounded-lg ${
              neo4jStatus === 'connected'
                ? 'bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-green-500/30 via-green-500/15 to-green-500/5'
                : neo4jStatus === 'connecting'
                ? 'bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-blue-500/30 via-blue-500/15 to-blue-500/5'
                : neo4jStatus === 'error'
                ? 'bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-red-500/30 via-red-500/15 to-red-500/5'
                : 'bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-slate-500/15 via-slate-500/8 to-transparent'
            }`} />
            
            <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-3">
              <CardTitle className="text-sm font-semibold">Neo4j Database Status</CardTitle>
              <div className={`rounded-xl p-2.5 transition-all duration-300 ${
                neo4jStatus === 'connected'
                  ? 'bg-green-100 dark:bg-green-900 shadow-green-200 dark:shadow-green-800 shadow-sm'
                  : neo4jStatus === 'connecting'
                  ? 'bg-blue-100 dark:bg-blue-900 shadow-blue-200 dark:shadow-blue-800 shadow-sm'
                  : neo4jStatus === 'error'
                  ? 'bg-red-100 dark:bg-red-900 shadow-red-200 dark:shadow-red-800 shadow-sm'
                  : 'bg-slate-100 dark:bg-slate-900'
              }`}>
                <Database className={`h-4 w-4 transition-colors duration-300 ${
                  neo4jStatus === 'connected'
                    ? 'text-green-600 dark:text-green-400'
                    : neo4jStatus === 'connecting'
                    ? 'text-blue-600 dark:text-blue-400'
                    : neo4jStatus === 'error'
                    ? 'text-red-600 dark:text-red-400'
                    : 'text-slate-600 dark:text-slate-400'
                }`} />
              </div>
            </CardHeader>
            <CardContent className="relative">
              <div className="flex items-center justify-between">
                <div>
                  <div className={`text-2xl font-bold transition-colors duration-300 ${
                    neo4jStatus === 'connected'
                      ? 'text-green-600 dark:text-green-400'
                      : neo4jStatus === 'connecting'
                      ? 'text-blue-600 dark:text-blue-400'
                      : neo4jStatus === 'error'
                      ? 'text-red-600 dark:text-red-400'
                      : 'text-slate-600 dark:text-slate-400'
                  }`}>
                    {neo4jStatus === 'connected' ? 'Connected' : neo4jStatus === 'connecting' ? 'Connecting...' : neo4jStatus === 'error' ? 'Error' : 'Not Connected'}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1.5">Graph Storage</p>
                </div>
                {neo4jStatus === 'connected' ? (
                  <CheckCircle2 className="h-9 w-9 text-green-500 animate-in zoom-in duration-300" />
                ) : neo4jStatus === 'connecting' ? (
                  <Loader2 className="h-9 w-9 text-blue-500 animate-spin" />
                ) : neo4jStatus === 'error' ? (
                  <XCircle className="h-9 w-9 text-red-500 animate-in zoom-in duration-300" />
                ) : (
                  <XCircle className="h-9 w-9 text-slate-400" />
                )}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Data Stats - Only show when both connected */}
        {apiStatus === 'connected' && neo4jStatus === 'connected' && stats && (
          <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-purple-500/25 transition-all duration-300">
            {/* Gradient border */}
            <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500">
              <div className="w-full h-full bg-background rounded-lg" />
            </div>
            
            {/* Background gradient overlay */}
            <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-purple-500/30 via-purple-500/15 to-purple-500/5" />
            
            <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-3">
              <div className="flex items-center gap-3">
                <div className="rounded-xl bg-purple-500/10 p-2.5">
                  <Database className="h-4 w-4 text-purple-500" />
                </div>
                <div>
                  <CardTitle className="text-sm font-semibold">Database Statistics</CardTitle>
                  <CardDescription className="text-xs mt-0.5">Current graph data overview</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="relative">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="flex items-center gap-3 p-4 rounded-lg border border-blue-200 dark:border-blue-900 bg-gradient-to-br from-blue-50/50 to-white dark:from-blue-950/30 dark:to-background hover:shadow-md transition-all duration-200">
                  <div className="rounded-lg bg-blue-100 dark:bg-blue-900 p-2.5">
                    <Package className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-blue-600 dark:text-blue-400">{stats.total_nodes.toLocaleString()}</p>
                    <p className="text-xs text-muted-foreground">Components</p>
                  </div>
                </div>
                <div className="flex items-center gap-3 p-4 rounded-lg border border-purple-200 dark:border-purple-900 bg-gradient-to-br from-purple-50/50 to-white dark:from-purple-950/30 dark:to-background hover:shadow-md transition-all duration-200">
                  <div className="rounded-lg bg-purple-100 dark:bg-purple-900 p-2.5">
                    <Activity className="h-5 w-5 text-purple-600 dark:text-purple-400" />
                  </div>
                  <div>
                    <p className="text-2xl font-bold text-purple-600 dark:text-purple-400">{stats.total_edges.toLocaleString()}</p>
                    <p className="text-xs text-muted-foreground">Relationships</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Configuration Section */}
        <div className="space-y-4">
          {/* API Server Configuration */}
          <Card className={`relative overflow-hidden border-0 shadow-lg hover:shadow-xl transition-all duration-300 ${
            apiStatus === 'connected'
              ? 'hover:shadow-green-500/25'
              : apiStatus === 'checking'
              ? 'hover:shadow-blue-500/25'
              : 'hover:shadow-red-500/25'
          }`}>
            {/* Gradient border */}
            <div className={`absolute inset-0 rounded-lg p-[2px] ${
              apiStatus === 'connected'
                ? 'bg-gradient-to-r from-green-500 via-emerald-500 to-teal-500'
                : apiStatus === 'checking'
                ? 'bg-gradient-to-r from-blue-500 via-indigo-500 to-purple-500'
                : 'bg-gradient-to-r from-red-500 via-rose-500 to-pink-500'
            }`}>
              <div className="w-full h-full bg-background rounded-lg" />
            </div>
            
            {/* Background gradient overlay */}
            <div className={`absolute inset-[2px] rounded-lg ${
              apiStatus === 'connected'
                ? 'bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-green-500/30 via-green-500/15 to-green-500/5'
                : apiStatus === 'checking'
                ? 'bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-blue-500/30 via-blue-500/15 to-blue-500/5'
                : 'bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-red-500/30 via-red-500/15 to-red-500/5'
            }`} />
            
            <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-3">
              <div className="flex items-center gap-3">
                <div className={`rounded-xl p-2.5 ${
                  apiStatus === 'connected'
                    ? 'bg-green-500/10'
                    : apiStatus === 'checking'
                    ? 'bg-blue-500/10'
                    : 'bg-red-500/10'
                }`}>
                  <Server className={`h-4 w-4 ${
                    apiStatus === 'connected'
                      ? 'text-green-500'
                      : apiStatus === 'checking'
                      ? 'text-blue-500'
                      : 'text-red-500'
                  }`} />
                </div>
                <div>
                  <CardTitle className="text-sm font-semibold">Graph Analysis API</CardTitle>
                  <CardDescription className="text-xs mt-0.5">Backend analysis service endpoint</CardDescription>
                </div>
              </div>
              {apiStatus === 'connected' && (
                <Badge variant="outline" className="text-green-600 border-green-600">
                  Connected
                </Badge>
              )}
            </CardHeader>
            <CardContent className="relative space-y-4">
              <div className="space-y-2">
                <Label htmlFor="api-url">Service URL</Label>
                <div className="flex gap-2">
                  <Input
                    id="api-url"
                    placeholder="http://localhost:8000"
                    value={apiBaseUrl}
                    onChange={(e) => setApiBaseUrl(e.target.value)}
                    className={`font-mono text-sm ${
                      hasUrlChanged
                        ? 'border-amber-500 focus-visible:ring-amber-500'
                        : ''
                    }`}
                    onKeyDown={(e) => e.key === 'Enter' && handleSaveApiUrl()}
                  />
                  <Button
                    onClick={handleSaveApiUrl}
                    disabled={isTestingConnection || apiStatus === 'checking'}
                    size="default"
                  >
                    {isTestingConnection || apiStatus === 'checking' ? (
                      <>
                        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        Testing...
                      </>
                    ) : (
                      <>
                        <RefreshCw className="mr-2 h-4 w-4" />
                        {hasUrlChanged ? 'Test' : 'Save & Test'}
                      </>
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
                <div className="rounded-xl bg-gradient-to-br from-red-50 to-pink-50 dark:from-red-950/30 dark:to-pink-950/30 border border-red-200 dark:border-red-900 p-4">
                  <div className="flex items-start gap-3">
                    <div className="rounded-lg bg-red-100 dark:bg-red-900 p-2 mt-0.5">
                      <AlertCircle className="h-4 w-4 text-red-600 dark:text-red-400 flex-shrink-0" />
                    </div>
                    <div className="space-y-2 text-sm">
                      <p className="font-medium text-red-900 dark:text-red-100">Connection Failed</p>
                      <p className="text-red-800 dark:text-red-200">{apiError}</p>
                    </div>
                  </div>
                </div>
              )}
              {apiStatus === 'connected' && !hasUrlChanged && (
                <div className="rounded-xl bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-950/30 dark:to-emerald-950/30 border border-green-200 dark:border-green-900 p-4">
                  <div className="flex items-start gap-3">
                    <div className="rounded-lg bg-green-100 dark:bg-green-900 p-2 mt-0.5">
                      <CheckCircle2 className="h-4 w-4 text-green-600 dark:text-green-400 flex-shrink-0" />
                    </div>
                    <div className="space-y-2 text-sm">
                      <p className="font-medium text-green-900 dark:text-green-100">
                        Connected successfully
                      </p>
                      <p className="text-green-800 dark:text-green-200">
                        You can now configure the database connection below.
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Warning when API not connected */}
          {apiStatus !== 'connected' && (
            <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-purple-500/25 transition-all duration-300">
              {/* Gradient border */}
              <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500">
                <div className="w-full h-full bg-background rounded-lg" />
              </div>
              
              {/* Background gradient overlay */}
              <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-purple-500/30 via-purple-500/15 to-purple-500/5" />
              
              <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-3">
                <div className="flex items-center gap-3">
                  <div className="rounded-xl bg-purple-500/10 p-2.5">
                    <Database className="h-4 w-4 text-purple-500" />
                  </div>
                  <div>
                    <CardTitle className="text-sm font-semibold">Graph Database Connection</CardTitle>
                    <CardDescription className="text-xs mt-0.5">Connect to the analysis service first</CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="relative space-y-4">
                <div className="rounded-xl bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-950/30 dark:to-pink-950/30 border border-purple-200 dark:border-purple-900 p-4">
                  <div className="flex items-start gap-3">
                    <div className="rounded-lg bg-purple-100 dark:bg-purple-900 p-2 mt-0.5">
                      <Info className="h-4 w-4 text-purple-600 dark:text-purple-400 flex-shrink-0" />
                    </div>
                    <div className="space-y-2 text-sm">
                      <p className="font-medium text-purple-900 dark:text-purple-100">Neo4j database configuration is currently unavailable</p>
                      <p className="text-purple-800 dark:text-purple-200">
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
                      <Badge variant="outline" className="mt-0.5 h-5 w-5 rounded-full flex items-center justify-center p-0 text-xs border-purple-500/30">1</Badge>
                      <span>Enter the Graph Analysis API URL above (e.g., http://localhost:8000)</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <Badge variant="outline" className="mt-0.5 h-5 w-5 rounded-full flex items-center justify-center p-0 text-xs border-purple-500/30">2</Badge>
                      <span>Click "Save & Test" to verify the connection</span>
                    </li>
                    <li className="flex items-start gap-2">
                      <Badge variant="outline" className="mt-0.5 h-5 w-5 rounded-full flex items-center justify-center p-0 text-xs border-purple-500/30">3</Badge>
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
