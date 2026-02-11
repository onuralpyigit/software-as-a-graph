"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { X, Waypoints } from "lucide-react"
import { Button } from "@/components/ui/button"
import type { GraphNode } from "@/lib/types/api"

interface NodeDetailsPanelProps {
  node: GraphNode | null
  onClose: () => void
  onViewSubgraph?: (nodeId: string) => void
}

export function NodeDetailsPanel({ node, onClose, onViewSubgraph }: NodeDetailsPanelProps) {
  if (!node) return null

  const getCriticalityColor = (level?: string) => {
    switch (level) {
      case "critical":
        return "destructive"
      case "high":
        return "default"
      case "medium":
        return "secondary"
      case "low":
        return "outline"
      default:
        return "outline"
    }
  }

  // Get the property name and ID
  const nodeName = node.properties?.name
  const nodeId = node.properties?.id || node.id

  return (
    <Card className="relative gap-0 h-full max-h-full flex flex-col overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-blue-500/25 transition-all duration-300">
      <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-blue-500 via-indigo-500 to-purple-500 opacity-100 pointer-events-none">
        <div className="w-full h-full bg-background rounded-lg" />
      </div>
      <div className="relative flex flex-col h-full">
      <CardHeader className="pb-3 flex-shrink-0">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-3 flex-1">
            <div className="rounded-2xl bg-gradient-to-br from-blue-600 via-indigo-600 to-purple-600 p-3 shadow-lg flex-shrink-0">
              <Waypoints className="h-5 w-5 text-white" />
            </div>
            <div className="flex-1 min-w-0">
              <CardTitle className="text-base font-semibold truncate">{node.label}</CardTitle>
              <div className="mt-1.5">
                <Badge variant="outline" className="text-xs border-blue-400 dark:border-blue-500 bg-blue-100/60 dark:bg-blue-500/20 text-blue-700 dark:text-blue-300">{node.type}</Badge>
              </div>
            </div>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose} className="flex-shrink-0 h-8 w-8 hover:bg-slate-100 dark:hover:bg-slate-800">
            <X className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 overflow-y-auto flex-1">
        {/* Identification */}
        <div className="rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50/80 dark:bg-slate-800/50 p-3 hover:border-slate-300 dark:hover:border-slate-600 transition-all">
          <div className="text-xs font-semibold text-blue-600 dark:text-blue-400 uppercase mb-3">
            Identification
          </div>
          <div className="space-y-2">
            <div className="flex justify-between text-sm gap-3">
              <span className="text-slate-500 dark:text-slate-400">ID:</span>
              <span className="font-mono text-xs text-slate-900 dark:text-white break-all text-right">{nodeId}</span>
            </div>
            {nodeName && (
              <div className="flex justify-between text-sm gap-3">
                <span className="text-slate-500 dark:text-slate-400">Name:</span>
                <span className="font-mono text-xs text-slate-900 dark:text-white break-all text-right">{nodeName}</span>
              </div>
            )}
            <div className="flex justify-between text-sm gap-3">
              <span className="text-slate-500 dark:text-slate-400">Type:</span>
              <span className="font-medium text-slate-900 dark:text-white">{node.type}</span>
            </div>
            <div className="flex justify-between text-sm gap-3">
              <span className="text-slate-500 dark:text-slate-400">Label:</span>
              <span className="font-medium text-slate-900 dark:text-white break-all text-right">{node.label}</span>
            </div>
          </div>
        </div>

        {/* Analysis */}
        {((node.degree !== undefined && node.degree !== null) ||
          (node.betweenness !== undefined && node.betweenness !== null) ||
          (node.pagerank !== undefined && node.pagerank !== null) ||
          node.criticality_level) && (
          <div className="rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50/80 dark:bg-slate-800/50 p-3 hover:border-slate-300 dark:hover:border-slate-600 transition-all">
            <div className="text-xs font-semibold text-blue-600 dark:text-blue-400 uppercase mb-3">
              Analysis
            </div>
            <div className="space-y-3">
              {/* Criticality (shown first as most important) */}
              {node.criticality_level && (
                <>
                  <div>
                    <div className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-2">Criticality Assessment</div>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-slate-500 dark:text-slate-400">Level:</span>
                        <Badge variant={getCriticalityColor(node.criticality_level)}>
                          {node.criticality_level}
                        </Badge>
                      </div>
                      {(node.criticality_score !== undefined && node.criticality_score !== null) && (
                        <div className="flex justify-between text-sm">
                          <span className="text-slate-500 dark:text-slate-400">Score:</span>
                          <span className="font-medium text-slate-900 dark:text-white">{Number.isInteger(node.criticality_score) ? node.criticality_score : node.criticality_score.toFixed(2)}</span>
                        </div>
                      )}
                    </div>
                  </div>
                  {((node.degree !== undefined && node.degree !== null) ||
                    (node.betweenness !== undefined && node.betweenness !== null) ||
                    (node.pagerank !== undefined && node.pagerank !== null)) && (
                    <Separator className="bg-slate-700/50" />
                  )}
                </>
              )}

              {/* Centrality Metrics */}
              {((node.degree !== undefined && node.degree !== null) ||
                (node.betweenness !== undefined && node.betweenness !== null) ||
                (node.pagerank !== undefined && node.pagerank !== null)) && (
                <div>
                  <div className="text-xs font-medium text-slate-500 dark:text-slate-400 mb-2">Network Centrality</div>
                  <div className="space-y-2">
                    {(node.degree !== undefined && node.degree !== null) && (
                      <div className="flex justify-between text-sm">
                        <span className="text-slate-500 dark:text-slate-400">Degree:</span>
                        <span className="font-medium text-slate-900 dark:text-white">{node.degree}</span>
                      </div>
                    )}
                    {(node.betweenness !== undefined && node.betweenness !== null) && (
                      <div className="flex justify-between text-sm">
                        <span className="text-slate-500 dark:text-slate-400">Betweenness:</span>
                        <span className="font-medium text-slate-900 dark:text-white">{Number.isInteger(node.betweenness) ? node.betweenness : node.betweenness.toFixed(2)}</span>
                      </div>
                    )}
                    {(node.pagerank !== undefined && node.pagerank !== null) && (
                      <div className="flex justify-between text-sm">
                        <span className="text-slate-500 dark:text-slate-400">PageRank:</span>
                        <span className="font-medium text-slate-900 dark:text-white">{Number.isInteger(node.pagerank) ? node.pagerank : node.pagerank.toFixed(2)}</span>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Additional Properties */}
        {(() => {
          // Filter out 'id' and 'name' properties since they're shown above
          const filteredProps = Object.entries(node.properties).filter(
            ([key]) => key !== 'id' && key !== 'name'
          )
          return filteredProps.length > 0 && (
            <div className="rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50/80 dark:bg-slate-800/50 p-3 hover:border-slate-300 dark:hover:border-slate-600 transition-all">
              <div className="text-xs font-semibold text-blue-600 dark:text-blue-400 uppercase mb-3">
                Additional Properties
              </div>
              <div className="space-y-1">
                {filteredProps.map(([key, value]) => (
                  <div key={key} className="flex justify-between text-sm">
                    <span className="text-slate-500 dark:text-slate-400">{key}:</span>
                    <span className="font-medium font-mono text-xs text-slate-900 dark:text-white">
                      {typeof value === "object" ? JSON.stringify(value) : typeof value === "number" ? (Number.isInteger(value) ? value : value.toFixed(2)) : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )
        })()}
      </CardContent>
      </div>
    </Card>
  )
}
