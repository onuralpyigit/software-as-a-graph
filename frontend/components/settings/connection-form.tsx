"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Loader2, CheckCircle2, XCircle, Database, AlertTriangle } from "lucide-react"
import { useConnection } from "@/lib/stores/connection-store"
import type { Neo4jConfig } from "@/lib/types/api"

export function ConnectionForm() {
  const { status, config, stats, error, connect, disconnect } = useConnection()

  // Helper to get default Neo4j URI dynamically
  const getDefaultNeo4jUri = () => {
    if (typeof window !== 'undefined') {
      const hostname = window.location.hostname;
      return `bolt://${hostname}:7687`;
    }
    return "bolt://localhost:7687";
  };

  const [formData, setFormData] = useState<Neo4jConfig>({
    uri: config?.uri || getDefaultNeo4jUri(),
    user: config?.user || "neo4j",
    password: config?.password || "password",
    database: config?.database || "neo4j",
  })

  const [isSubmitting, setIsSubmitting] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsSubmitting(true)

    try {
      await connect(formData)
    } catch (error) {
      // Error is handled in the store
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleDisconnect = () => {
    disconnect()
    setFormData(prev => ({ ...prev, password: "" }))
  }

  const isConnected = status === 'connected'
  const isConnecting = status === 'connecting' || isSubmitting

  return (
    <Card className={`relative overflow-hidden border-0 shadow-lg hover:shadow-xl transition-all duration-300 ${
      status === 'connected'
        ? 'hover:shadow-green-500/20'
        : status === 'connecting'
        ? 'hover:shadow-blue-500/20'
        : status === 'error'
        ? 'hover:shadow-red-500/20'
        : 'hover:shadow-slate-500/20'
    }`}>
      {/* Gradient border */}
      <div className={`absolute inset-0 rounded-lg p-[2px] ${
        status === 'connected'
          ? 'bg-gradient-to-r from-green-400 via-emerald-500 to-teal-600'
          : status === 'connecting'
          ? 'bg-gradient-to-r from-blue-400 via-indigo-500 to-violet-600'
          : status === 'error'
          ? 'bg-gradient-to-r from-red-400 via-rose-500 to-pink-600'
          : 'bg-gradient-to-br from-slate-200 via-slate-300 to-slate-200 dark:from-slate-700 dark:via-slate-800 dark:to-slate-700'
      }`}>
        <div className="w-full h-full bg-background rounded-lg" />
      </div>
      
      {/* Background gradient overlay */}
      <div className={`absolute inset-[2px] rounded-lg ${
        status === 'connected'
          ? 'bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-green-500/35 via-green-500/20 to-green-500/5'
          : status === 'connecting'
          ? 'bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-blue-500/25 via-blue-500/15 to-blue-500/3'
          : status === 'error'
          ? 'bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-red-500/35 via-red-500/20 to-red-500/5'
          : 'bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-slate-500/15 via-slate-500/8 to-transparent'
      }`} />
      
      <CardHeader className="relative flex flex-row items-center justify-between space-y-0 pb-3">
        <div className="flex items-center gap-3">
          <div className={`rounded-xl p-2.5 ${
            status === 'connected'
              ? 'bg-green-500/10'
              : status === 'connecting'
              ? 'bg-blue-500/10'
              : status === 'error'
              ? 'bg-red-500/10'
              : 'bg-slate-500/10'
          }`}>
            <Database className={`h-4 w-4 ${
              status === 'connected'
                ? 'text-green-500'
                : status === 'connecting'
                ? 'text-blue-500'
                : status === 'error'
                ? 'text-red-500'
                : 'text-slate-500'
            }`} />
          </div>
          <div>
            <CardTitle className="text-sm font-semibold">Graph Database</CardTitle>
            <CardDescription className="text-xs mt-0.5">
              {isConnected ? 'Connected to your system graph' : 'Configure Neo4j connection'}
            </CardDescription>
          </div>
        </div>
        <div>
          {status === 'connected' && (
            <Badge variant="outline" className="text-green-600 border-green-600">Connected</Badge>
          )}
          {status === 'disconnected' && (
            <Badge variant="outline">Disconnected</Badge>
          )}
          {status === 'error' && (
            <Badge variant="destructive">Error</Badge>
          )}
          {status === 'connecting' && (
            <Badge variant="secondary">
              <Loader2 className="h-3 w-3 mr-1 animate-spin" />
              Connecting...
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="relative space-y-4">{/* Error message */}
        {error && (
          <div className="rounded-xl bg-gradient-to-br from-red-50 to-pink-50 dark:from-red-950/30 dark:to-pink-950/30 border border-red-200 dark:border-red-900 p-4">
            <div className="flex items-start gap-3">
              <div className="rounded-lg bg-red-100 dark:bg-red-900 p-2 mt-0.5">
                <AlertTriangle className="h-4 w-4 text-red-600 dark:text-red-400 flex-shrink-0" />
              </div>
              <div className="space-y-2 text-sm">
                <p className="font-medium text-red-900 dark:text-red-100">Connection Failed</p>
                <p className="text-red-800 dark:text-red-200">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Connection Form */}
        <form onSubmit={handleSubmit} className="space-y-4 pt-2">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="uri">URI</Label>
              <Input
                id="uri"
                placeholder="bolt://localhost:7687"
                value={formData.uri}
                onChange={(e) => setFormData({ ...formData, uri: e.target.value })}
                disabled={isConnected}
                required
                className="font-mono text-sm"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="database">Database</Label>
              <Input
                id="database"
                placeholder="neo4j"
                value={formData.database}
                onChange={(e) => setFormData({ ...formData, database: e.target.value })}
                disabled={isConnected}
                required
              />
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="user">Username</Label>
              <Input
                id="user"
                placeholder="neo4j"
                value={formData.user}
                onChange={(e) => setFormData({ ...formData, user: e.target.value })}
                disabled={isConnected}
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                placeholder="Enter password"
                value={formData.password}
                onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                disabled={isConnected}
                required={!isConnected}
              />
            </div>
          </div>

          <div className="flex gap-2 pt-2">
            {!isConnected ? (
              <Button
                type="submit"
                disabled={isConnecting}
                className="w-full"
              >
                {isConnecting ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Connecting...
                  </>
                ) : (
                  'Connect'
                )}
              </Button>
            ) : (
              <Button
                type="button"
                variant="destructive"
                onClick={handleDisconnect}
                className="w-full"
              >
                Disconnect
              </Button>
            )}
          </div>
        </form>
      </CardContent>
    </Card>
  )
}
