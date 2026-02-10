"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { X, ArrowRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import type { GraphLink, GraphNode } from "@/lib/types/api"

interface LinkDetailsPanelProps {
  link: GraphLink | null
  nodes: GraphNode[]
  onClose: () => void
}

export function LinkDetailsPanel({ link, nodes, onClose }: LinkDetailsPanelProps) {
  if (!link) return null

  // Helper to get node by ID
  const getNodeById = (id: string) => {
    return nodes.find(n => n.id === id)
  }

  // Get source and target IDs
  const sourceId = typeof link.source === 'string' ? link.source : (link.source as any).id
  const targetId = typeof link.target === 'string' ? link.target : (link.target as any).id

  // Get node objects
  const sourceNode = getNodeById(sourceId)
  const targetNode = getNodeById(targetId)

  return (
    <Card className="relative gap-0 h-full max-h-full flex flex-col overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-green-500/20 transition-all duration-300">
      <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-green-400 via-emerald-500 to-emerald-600 opacity-100 pointer-events-none">
        <div className="w-full h-full bg-background rounded-lg" />
      </div>
      <div className="relative flex flex-col h-full">
      <CardHeader className="pb-3 flex-shrink-0">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-3 flex-1">
            <div className="rounded-2xl bg-gradient-to-br from-green-500 to-emerald-600 p-3 shadow-lg flex-shrink-0">
              <ArrowRight className="h-4 w-4 text-white" />
            </div>
            <div className="flex-1 min-w-0">
              <CardTitle className="text-base font-semibold">Link Details</CardTitle>
              <div className="mt-1.5">
                <Badge variant="outline" className="text-xs border-green-400 dark:border-green-500/50 bg-green-100/60 dark:bg-green-500/10 text-green-700 dark:text-green-400">{link.type}</Badge>
              </div>
            </div>
          </div>
          <Button variant="ghost" size="icon" onClick={onClose} className="flex-shrink-0 h-8 w-8 hover:bg-slate-100 dark:hover:bg-slate-800">
            <X className="h-4 w-4" />
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 overflow-y-auto flex-1">
        {/* Connection */}
        <div className="rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50/80 dark:bg-slate-800/50 p-3 hover:border-slate-300 dark:hover:border-slate-600 transition-all">
          <div className="text-xs font-semibold text-emerald-600 dark:text-emerald-400 uppercase mb-3">
            Connection
          </div>
          <div className="space-y-2">
            {/* Source Node */}
            <div className="bg-white/80 dark:bg-slate-900/50 rounded-md p-3 border border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600 transition-all">
              <div className="text-xs text-emerald-600 dark:text-emerald-400 font-semibold mb-1.5">Source</div>
              <div className="space-y-1.5">
                <div className="flex justify-between text-sm gap-2">
                  <span className="text-slate-500 dark:text-slate-400 text-xs">ID:</span>
                  <span className="font-mono text-xs text-slate-900 dark:text-white break-all text-right">{sourceNode?.properties?.id || sourceId}</span>
                </div>
                {sourceNode && (
                  <>
                    <div className="flex justify-between text-sm gap-2">
                      <span className="text-slate-500 dark:text-slate-400 text-xs">Label:</span>
                      <span className="font-medium text-sm text-slate-900 dark:text-white break-all text-right">{sourceNode.label}</span>
                    </div>
                    <div className="flex justify-between items-center text-sm gap-2">
                      <span className="text-slate-500 dark:text-slate-400 text-xs">Type:</span>
                      <Badge variant="outline" className="text-xs border-emerald-400 dark:border-emerald-500/50 bg-emerald-100/60 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400">
                        {sourceNode.type}
                      </Badge>
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Arrow */}
            <div className="flex justify-center py-1">
              <div className="rounded-full bg-gradient-to-br from-emerald-400 to-emerald-600 dark:from-emerald-500 dark:to-emerald-600 p-1.5">
                <ArrowRight className="h-4 w-4 text-white" />
              </div>
            </div>

            {/* Target Node */}
            <div className="bg-white/80 dark:bg-slate-900/50 rounded-md p-3 border border-slate-200 dark:border-slate-700 hover:border-slate-300 dark:hover:border-slate-600 transition-all">
              <div className="text-xs text-emerald-600 dark:text-emerald-400 font-semibold mb-1.5">Target</div>
              <div className="space-y-1.5">
                <div className="flex justify-between text-sm gap-2">
                  <span className="text-slate-500 dark:text-slate-400 text-xs">ID:</span>
                  <span className="font-mono text-xs text-slate-900 dark:text-white break-all text-right">{targetNode?.properties?.id || targetId}</span>
                </div>
                {targetNode && (
                  <>
                    <div className="flex justify-between text-sm gap-2">
                      <span className="text-slate-500 dark:text-slate-400 text-xs">Label:</span>
                      <span className="font-medium text-sm text-slate-900 dark:text-white break-all text-right">{targetNode.label}</span>
                    </div>
                    <div className="flex justify-between items-center text-sm gap-2">
                      <span className="text-slate-500 dark:text-slate-400 text-xs">Type:</span>
                      <Badge variant="outline" className="text-xs border-emerald-400 dark:border-emerald-500/50 bg-emerald-100/60 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400">
                        {targetNode.type}
                      </Badge>
                    </div>
                  </>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Link Metadata */}
        <div className="rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50/80 dark:bg-slate-800/50 p-3 hover:border-slate-300 dark:hover:border-slate-600 transition-all">
          <div className="text-xs font-semibold text-emerald-600 dark:text-emerald-400 uppercase mb-3">
            Link Metadata
          </div>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-slate-500 dark:text-slate-400">Type:</span>
              <span className="font-medium text-slate-900 dark:text-white">{link.type}</span>
            </div>
            {link.weight !== undefined && link.weight !== null && (
              <div className="flex justify-between text-sm">
                <span className="text-slate-500 dark:text-slate-400">Weight:</span>
                <span className="font-medium text-slate-900 dark:text-white">{typeof link.weight === "number" ? (Number.isInteger(link.weight) ? link.weight : link.weight.toFixed(2)) : link.weight}</span>
              </div>
            )}
          </div>
        </div>

        {/* Additional Properties */}
        {link.properties && Object.keys(link.properties).length > 0 && (
          <div className="rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50/80 dark:bg-slate-800/50 p-3 hover:border-slate-300 dark:hover:border-slate-600 transition-all">
            <div className="text-xs font-semibold text-emerald-600 dark:text-emerald-400 uppercase mb-3">
              Additional Properties
            </div>
            <div className="space-y-1">
              {Object.entries(link.properties).map(([key, value]) => (
                <div key={key} className="flex justify-between text-sm">
                  <span className="text-slate-500 dark:text-slate-400">{key}:</span>
                  <span className="font-medium font-mono text-xs text-slate-900 dark:text-white">
                    {typeof value === "object" ? JSON.stringify(value) : typeof value === "number" ? (Number.isInteger(value) ? value : value.toFixed(2)) : String(value)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
      </div>
    </Card>
  )
}
