"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
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
    <Card className={`bg-background`}>
      <CardHeader className="pb-1 flex flex-row items-center justify-between space-y-0">
        <div className="flex items-center gap-2.5">
          <div className={`rounded-lg p-1.5 ${
            status === 'connected' ? 'bg-green-500/10'
            : status === 'connecting' ? 'bg-blue-500/10'
            : status === 'error' ? 'bg-red-500/10'
            : 'bg-slate-500/10'
          }`}>
            <Database className={`h-4 w-4 ${
              status === 'connected' ? 'text-green-400'
              : status === 'connecting' ? 'text-blue-400'
              : status === 'error' ? 'text-red-400'
              : 'text-slate-400'
            }`} />
          </div>
          <div>
            <CardTitle className="text-sm font-semibold">Graph Database</CardTitle>
            <p className="text-[11px] text-muted-foreground">
              {isConnected ? 'Connected to your system graph' : 'Configure Neo4j connection'}
            </p>
          </div>
        </div>
        <div>
          {status === 'connected' && (
            <Badge className="bg-green-500/10 text-green-400 border-green-500/20 text-[11px] px-2">Connected</Badge>
          )}
          {status === 'disconnected' && (
            <Badge variant="outline" className="text-[11px] px-2">Disconnected</Badge>
          )}
          {status === 'error' && (
            <Badge variant="destructive" className="text-[11px] px-2">Error</Badge>
          )}
          {status === 'connecting' && (
            <Badge variant="secondary" className="text-[11px] px-2">
              <Loader2 className="h-3 w-3 mr-1 animate-spin" />
              Connecting...
            </Badge>
          )}
        </div>
      </CardHeader>
      <CardContent className="pt-1 space-y-4">
        {/* Error message */}
        {error && (
          <div className="rounded-xl border border-red-500/20 bg-red-500/[0.07] p-4">
            <div className="flex items-start gap-3">
              <div className="rounded-lg bg-red-500/10 p-1.5 mt-0.5">
                <AlertTriangle className="h-4 w-4 text-red-400 flex-shrink-0" />
              </div>
              <div className="space-y-1 text-sm">
                <p className="font-medium">Connection Failed</p>
                <p className="text-muted-foreground">{error}</p>
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
