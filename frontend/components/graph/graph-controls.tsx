"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Separator } from "@/components/ui/separator"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  RefreshCw,
  Box,
  Layers,
  Waypoints,
  AppWindow,
  Server,
  GitBranch,
  Workflow,
  Database,
  Shield,
  Search,
  Maximize2,
  Download,
  FileSpreadsheet,
  Settings,
  X,
  Filter,
  BarChart3,
  Zap
} from "lucide-react"
import type { GraphView } from "@/lib/types/graph-views"
import { GRAPH_VIEWS } from "@/lib/types/graph-views"
import type { ForceGraphData, GraphNode, GraphLink } from "@/lib/types/api"
import { NodeDetailsPanel } from "./node-details-panel"
import { LinkDetailsPanel } from "./link-details-panel"

interface GraphControlsProps {
  currentView: GraphView
  onViewChange: (view: GraphView) => void
  is3D: boolean
  onToggle3D: (value: boolean) => void
  colorByType: boolean
  onToggleColorByType: (value: boolean) => void
  onRefresh: () => void
  onFitGraph?: () => void
  onExportPNG?: () => void
  onExportCSV?: () => void
  nodeCount: number
  linkCount: number
  isLoading?: boolean
  nodeTypes?: string[]
  linkTypes?: string[]
  visibleNodeTypes?: Set<string>
  visibleLinkTypes?: Set<string>
  onToggleNodeType?: (type: string) => void
  onToggleLinkType?: (type: string) => void
  currentZoom?: number
  pubSubZoomThreshold?: number
  onPubSubZoomThresholdChange?: (threshold: number) => void
  isPubSubEnabled?: boolean
  onTogglePubSubEnabled?: (enabled: boolean) => void
  graphData?: ForceGraphData | null
  searchQuery?: string
  onSearchChange?: (query: string) => void
  searchResults?: Array<{id: string, label: string, type: string}>
  onSearchResultClick?: (nodeId: string) => void
  selectedNode?: GraphNode | null
  selectedLink?: GraphLink | null
  onCloseNodeDetails?: () => void
  onCloseLinkDetails?: () => void
  onViewSubgraph?: (nodeId: string) => void
  nodeConnections?: Array<{node: GraphNode, link: GraphLink, direction: 'outgoing' | 'incoming'}>
  nodeLimit?: number
  edgeLimit?: number
  useLimits?: boolean
  onNodeLimitChange?: (limit: number) => void
  onEdgeLimitChange?: (limit: number) => void
  onUseLimitsChange?: (enabled: boolean) => void
  onApplyLimits?: () => void
}

export function GraphControls({
  currentView,
  onViewChange,
  is3D,
  onToggle3D,
  colorByType,
  onToggleColorByType,
  onRefresh,
  onFitGraph,
  onExportPNG,
  onExportCSV,
  nodeCount,
  linkCount,
  isLoading = false,
  nodeTypes = [],
  linkTypes = [],
  visibleNodeTypes,
  visibleLinkTypes,
  onToggleNodeType,
  onToggleLinkType,
  currentZoom = 1,
  pubSubZoomThreshold = 2,
  onPubSubZoomThresholdChange,
  isPubSubEnabled = true,
  onTogglePubSubEnabled,
  graphData,
  searchQuery = '',
  onSearchChange,
  searchResults = [],
  onSearchResultClick,
  selectedNode,
  selectedLink,
  onCloseNodeDetails,
  onCloseLinkDetails,
  onViewSubgraph,
  nodeConnections = [],
  nodeLimit = 1000,
  edgeLimit = 3000,
  useLimits = true,
  onNodeLimitChange,
  onEdgeLimitChange,
  onUseLimitsChange,
  onApplyLimits,
}: GraphControlsProps) {
  const viewConfig = GRAPH_VIEWS[currentView]
  
  // Local state for connections search
  const [connectionsSearch, setConnectionsSearch] = useState('')
  const [selectedConnectionTypes, setSelectedConnectionTypes] = useState<Set<string>>(new Set())

  // Calculate type counts
  const nodeTypeCounts = graphData?.nodes.reduce((acc, node) => {
    acc[node.type] = (acc[node.type] || 0) + 1
    return acc
  }, {} as Record<string, number>) || {}

  const linkTypeCounts = graphData?.links.reduce((acc, link) => {
    acc[link.type] = (acc[link.type] || 0) + 1
    return acc
  }, {} as Record<string, number>) || {}

  // Get icon for each view
  const getViewIcon = (viewId: GraphView) => {
    switch (viewId) {
      case 'complete':
        return Waypoints
      case 'application':
        return AppWindow
      case 'infrastructure':
        return Server
      case 'middleware':
        return GitBranch
      default:
        return Waypoints
    }
  }

  // Define color mappings for node types
  const nodeTypeColors: Record<string, string> = {
    Application: "bg-blue-500",
    Node: "bg-red-500",
    Broker: "bg-zinc-400",
    Topic: "bg-yellow-400",
    Library: "bg-cyan-500",
    Unknown: "bg-zinc-400",
  }

  // Define color mappings for link types
  const linkTypeColors: Record<string, string> = {
    RUNS_ON: "bg-purple-500",
    PUBLISHES_TO: "bg-green-500",
    SUBSCRIBES_TO: "bg-orange-500",
    DEPENDS_ON: "bg-red-500",
    CONNECTS_TO: "bg-green-500",
    ROUTES: "bg-zinc-400",
    USES: "bg-cyan-500",
    app_to_app: "bg-red-500",
    node_to_node: "bg-red-500",
    app_to_broker: "bg-purple-500",
    node_to_broker: "bg-orange-500",
  }

  // Convert link type to human-friendly display name
  const formatLinkType = (linkType: string): string => {
    // Map specific types to cleaner names
    if (linkType === 'app_to_app' || linkType === 'node_to_node' || 
        linkType === 'app_to_broker' || linkType === 'node_to_broker') {
      return 'DEPENDS_ON'
    }
    // Convert underscores to spaces and title case
    return linkType
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ')
  }

  // Get unique link types (merge dependency types into DEPENDS_ON, but preserve the original for color)
  const uniqueLinkTypesWithColors = linkTypes.reduce((acc, type) => {
    if (type === 'app_to_app' || type === 'node_to_node' || 
        type === 'app_to_broker' || type === 'node_to_broker') {
      // For dependency types, use the original type for color lookup but display as DEPENDS_ON
      if (!acc.some(item => item.display === 'DEPENDS_ON')) {
        acc.push({ original: type, display: 'DEPENDS_ON' })
      }
    } else {
      if (!acc.some(item => item.display === type)) {
        acc.push({ original: type, display: type })
      }
    }
    return acc
  }, [] as Array<{ original: string; display: string }>)

  return (
    <div className="h-full overflow-auto">
      <div className="space-y-4">
      {/* Search - Always Visible */}
      {onSearchChange && (
        <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-blue-500/20 transition-all duration-300">
          <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-blue-400 via-indigo-500 to-indigo-600 opacity-100">
            <div className="w-full h-full bg-background rounded-lg" />
          </div>
          <div className="relative">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-3">
              <div className="rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 p-3 shadow-lg">
                <Search className="h-5 w-5 text-white" />
              </div>
              <div className="flex-1">
                <CardTitle className="text-base font-semibold">Search Nodes</CardTitle>
                <CardDescription className="text-xs mt-0.5">Find nodes by name or type</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
              <Input
                type="text"
                placeholder="Search by node name..."
                value={searchQuery}
                onChange={(e) => onSearchChange(e.target.value)}
                className="pl-9 pr-9 border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 placeholder:text-slate-400 dark:placeholder:text-slate-500 focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500 transition-all"
              />
              {searchQuery && (
                <button
                  onClick={() => onSearchChange('')}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 transition-colors p-1"
                >
                  <X className="h-4 w-4" />
                </button>
              )}
            </div>
            {searchQuery && searchResults.length > 0 && (
              <div className="space-y-1 max-h-48 overflow-y-auto rounded-lg border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 p-2">
                <p className="text-xs font-medium text-slate-600 dark:text-slate-400 px-2 py-1">
                  {searchResults.length} result{searchResults.length !== 1 ? 's' : ''} found
                </p>
                {searchResults.map((result) => (
                  <button
                    key={result.id}
                    onClick={() => onSearchResultClick?.(result.id)}
                    className="w-full text-left px-2 py-2 rounded-md hover:bg-blue-500/10 transition-colors group"
                  >
                    <div className="flex items-center gap-2">
                      <div className={`flex-shrink-0 w-2 h-2 rounded-full ${
                        nodeTypeColors[result.type] || 'bg-gray-500'
                      }`} />
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate group-hover:text-blue-600 dark:group-hover:text-blue-300">
                          {result.label}
                        </p>
                        <p className="text-xs text-muted-foreground">{result.type}</p>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            )}
            {searchQuery && searchResults.length === 0 && (
              <div className="rounded-lg border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 p-3 text-center">
                <p className="text-sm text-slate-600 dark:text-slate-400">No nodes found matching "{searchQuery}"</p>
              </div>
            )}
          </CardContent>
          </div>
        </Card>
      )}

      {/* Node Details Panel - Always visible when selected */}
      {selectedNode && onCloseNodeDetails && (
        <NodeDetailsPanel
          node={selectedNode}
          onClose={onCloseNodeDetails}
          onViewSubgraph={onViewSubgraph}
        />
      )}

      {/* Node Connections List - Always visible when node is selected */}
      {selectedNode && nodeConnections.length > 0 && (() => {
        // Get unique connection types from current connections
        const connectionTypes = Array.from(new Set(nodeConnections.map(conn => conn.link.type)))
        
        // Filter connections based on search query and type filter
        const filteredConnections = nodeConnections.filter(conn => {
          // Search filter
          if (connectionsSearch.trim()) {
            const searchLower = connectionsSearch.toLowerCase()
            const matchesSearch = (
              conn.node.label.toLowerCase().includes(searchLower) ||
              conn.node.id.toLowerCase().includes(searchLower) ||
              conn.node.type.toLowerCase().includes(searchLower) ||
              conn.link.type.toLowerCase().includes(searchLower)
            )
            if (!matchesSearch) return false
          }
          
          // Type filter
          if (selectedConnectionTypes.size > 0 && !selectedConnectionTypes.has(conn.link.type)) {
            return false
          }
          
          return true
        })
        
        return (
        <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-emerald-500/20 transition-all duration-300">
          <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-emerald-400 via-emerald-500 to-emerald-600 opacity-100">
            <div className="w-full h-full bg-background rounded-lg" />
          </div>
          <div className="relative">
            <CardHeader className="pb-3">
              <div className="flex items-center gap-3">
                <div className="rounded-2xl bg-gradient-to-br from-emerald-500 to-emerald-600 p-3 shadow-lg">
                  <GitBranch className="h-5 w-5 text-white" />
                </div>
                <div className="flex-1">
                  <CardTitle className="text-base font-semibold">Connections</CardTitle>
                  <CardDescription className="text-xs mt-0.5">
                    {filteredConnections.length} of {nodeConnections.length} connection{nodeConnections.length !== 1 ? 's' : ''}
                  </CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-3">
              {/* Connection Type Filter */}
              {connectionTypes.length > 1 && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <p className="text-xs font-semibold text-slate-700 dark:text-slate-300">Filter by Type</p>
                    {selectedConnectionTypes.size > 0 && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setSelectedConnectionTypes(new Set())}
                        className="h-6 text-xs px-2 text-slate-500 hover:text-slate-700 dark:hover:text-slate-300"
                      >
                        Clear
                      </Button>
                    )}
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {connectionTypes.map((type) => {
                      const isSelected = selectedConnectionTypes.has(type)
                      const typeColor = linkTypeColors[type] || 'bg-zinc-400'
                      
                      return (
                        <button
                          key={type}
                          onClick={() => {
                            const newSelected = new Set(selectedConnectionTypes)
                            if (isSelected) {
                              newSelected.delete(type)
                            } else {
                              newSelected.add(type)
                            }
                            setSelectedConnectionTypes(newSelected)
                          }}
                          className={`flex items-center gap-1.5 px-2 py-1 rounded-md text-xs font-medium transition-all ${
                            isSelected
                              ? 'bg-emerald-500 text-white hover:bg-emerald-600'
                              : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-200 dark:hover:bg-slate-700'
                          }`}
                        >
                          <div className={`h-1 w-3 rounded-full ${typeColor} ${isSelected ? 'opacity-100' : 'opacity-50'}`} />
                          {formatLinkType(type)}
                        </button>
                      )
                    })}
                  </div>
                </div>
              )}
              
              {/* Search Input */}
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400 dark:text-slate-500" />
                <Input
                  placeholder="Search connections..."
                  value={connectionsSearch}
                  onChange={(e) => setConnectionsSearch(e.target.value)}
                  className="pl-9 pr-9 border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 hover:bg-slate-100 dark:hover:bg-slate-800 hover:border-slate-300 dark:hover:border-slate-600 focus:border-emerald-500 dark:focus:border-emerald-500 focus:ring-2 focus:ring-emerald-500/50 transition-all"
                />
                {connectionsSearch && (
                  <button
                    onClick={() => setConnectionsSearch('')}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 transition-colors p-1"
                  >
                    <X className="h-4 w-4" />
                  </button>
                )}
              </div>
              
              {/* Connections List */}
              <div className="space-y-2 max-h-64 overflow-y-auto">
              {filteredConnections.length > 0 ? (
                filteredConnections.map((conn, idx) => {
                const isOutgoing = conn.direction === 'outgoing'
                const linkTypeColor = linkTypeColors[conn.link.type] || 'bg-zinc-400'
                
                return (
                  <button
                    key={`${conn.node.id}-${conn.link.type}-${idx}`}
                    onClick={() => onSearchResultClick?.(conn.node.id)}
                    className="w-full text-left rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 p-3 hover:bg-slate-100 dark:hover:bg-slate-800 hover:border-slate-300 dark:hover:border-slate-600 hover:shadow-md transition-all group"
                  >
                    <div className="flex items-start gap-3">
                      <div className="flex-shrink-0 mt-0.5">
                        <div className={`w-2 h-2 rounded-full ${nodeTypeColors[conn.node.type] || 'bg-zinc-400'}`} />
                      </div>
                      <div className="flex-1 min-w-0 space-y-1.5">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium text-slate-900 dark:text-white truncate group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors">
                            {conn.node.label}
                          </span>
                          <Badge variant="outline" className="text-[10px] px-1.5 py-0 h-4 border-slate-300 dark:border-slate-600 bg-slate-100 dark:bg-slate-700 text-slate-600 dark:text-slate-300 flex-shrink-0">
                            {conn.node.type}
                          </Badge>
                        </div>
                        <div className="flex items-center gap-2">
                          <div className={`flex items-center gap-1.5 text-[11px] px-2 py-0.5 rounded-md ${linkTypeColor} bg-opacity-20 dark:bg-opacity-30`}>
                            <span className="font-medium">{isOutgoing ? '→' : '←'}</span>
                            <span className="font-medium">{formatLinkType(conn.link.type)}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </button>
                )
              })) : (
                <div className="text-center py-6 text-sm text-muted-foreground">
                  <Search className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p>No connections match your search</p>
                </div>
              )}
              </div>
            </CardContent>
          </div>
        </Card>
        )
      })()}

      {/* Link Details Panel - Always visible when selected */}
      {selectedLink && onCloseLinkDetails && graphData && (
        <LinkDetailsPanel
          link={selectedLink}
          nodes={graphData.nodes}
          onClose={onCloseLinkDetails}
        />
      )}

      <Tabs defaultValue="view" className="w-full">
        <TabsList className="w-full grid grid-cols-3 bg-slate-100 dark:bg-slate-800/50 border border-slate-300 dark:border-slate-700">
          <TabsTrigger value="view" className="gap-1.5 data-[state=active]:bg-white dark:data-[state=active]:bg-slate-700">
            <Workflow className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">View</span>
          </TabsTrigger>
          <TabsTrigger value="filter" className="gap-1.5 data-[state=active]:bg-white dark:data-[state=active]:bg-slate-700">
            <Filter className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Filter</span>
          </TabsTrigger>
          <TabsTrigger value="actions" className="gap-1.5 data-[state=active]:bg-white dark:data-[state=active]:bg-slate-700">
            <Zap className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Actions</span>
          </TabsTrigger>
        </TabsList>

        {/* VIEW TAB */}
        <TabsContent value="view" className="space-y-4">
      {/* Graph View & Mode */}
      <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-purple-500/20 transition-all duration-300">
        <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-purple-400 via-purple-500 to-purple-600 opacity-100">
          <div className="w-full h-full bg-background rounded-lg" />
        </div>
        <div className="relative">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-3">
            <div className="rounded-2xl bg-gradient-to-br from-purple-500 to-purple-600 p-3 shadow-lg">
              <Workflow className="h-5 w-5 text-white" />
            </div>
            <div className="flex-1">
              <CardTitle className="text-base font-semibold">View Configuration</CardTitle>
              <CardDescription className="text-xs mt-0.5">Customize graph visualization</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs font-semibold text-slate-700 dark:text-slate-300">Graph Type</Label>
              <Badge variant="secondary" className="text-[10px] bg-slate-200 dark:bg-slate-800 border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-300">{viewConfig.name}</Badge>
            </div>
            <Select value={currentView} onValueChange={(value) => onViewChange(value as GraphView)}>
              <SelectTrigger className="w-full h-10 border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 hover:bg-slate-100 dark:hover:bg-slate-800 hover:shadow-md transition-all focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500">
                <SelectValue>
                  {(() => {
                    const Icon = getViewIcon(currentView)
                    return (
                      <div className="flex items-center gap-2">
                        <div className="rounded-lg bg-purple-500/20 p-1">
                          <Icon className="h-3 w-3" style={{ color: viewConfig.color }} />
                        </div>
                        <span className="font-medium text-sm text-slate-900 dark:text-white">{viewConfig.name}</span>
                      </div>
                    )
                  })()}
                </SelectValue>
              </SelectTrigger>
              <SelectContent className="border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900">
                {Object.values(GRAPH_VIEWS).map((view) => {
                  const Icon = getViewIcon(view.id)
                  return (
                    <SelectItem 
                      key={view.id} 
                      value={view.id}
                      className="hover:bg-purple-500/20 focus:bg-purple-500/20 text-slate-900 dark:text-white"
                    >
                      <div className="flex items-center gap-2">
                        <div className="rounded-lg bg-purple-500/20 p-1">
                          <Icon className="h-3 w-3" style={{ color: view.color }} />
                        </div>
                        <span className="font-medium text-sm">{view.name}</span>
                      </div>
                    </SelectItem>
                  )
                })}
              </SelectContent>
            </Select>
            <p className="text-xs text-slate-400 mt-1.5">
              {viewConfig.description}
            </p>
          </div>

          <Separator className="bg-slate-300 dark:bg-slate-700/50" />

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label className="text-xs font-semibold text-slate-700 dark:text-slate-300">View Mode</Label>
              <Badge variant="secondary" className="text-[10px] bg-slate-200 dark:bg-slate-800 border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-300">{is3D ? '3D' : '2D'}</Badge>
            </div>
            <div className="flex gap-2.5">
              <button
                onClick={() => onToggle3D(false)}
                className={`flex-1 h-9 rounded-lg border font-medium transition-all duration-200 flex items-center justify-center gap-2 ${
                  !is3D
                    ? 'bg-gradient-to-br from-green-500 to-emerald-600 border-green-500 text-white shadow-lg shadow-green-500/30'
                    : 'border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 hover:border-slate-400 dark:hover:border-slate-600'
                }`}
              >
                <div className={`rounded-lg p-1 ${
                  !is3D 
                    ? 'bg-white/20' 
                    : 'bg-slate-200 dark:bg-slate-700'
                }`}>
                  <Layers className="h-3.5 w-3.5" />
                </div>
                <span className="text-sm">2D</span>
              </button>
              <button
                onClick={() => onToggle3D(true)}
                className={`flex-1 h-9 rounded-lg border font-medium transition-all duration-200 flex items-center justify-center gap-2 ${
                  is3D
                    ? 'bg-gradient-to-br from-green-500 to-emerald-600 border-green-500 text-white shadow-lg shadow-green-500/30'
                    : 'border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 text-slate-600 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800 hover:border-slate-400 dark:hover:border-slate-600'
                }`}
              >
                <div className={`rounded-lg p-1 ${
                  is3D 
                    ? 'bg-white/20' 
                    : 'bg-slate-200 dark:bg-slate-700'
                }`}>
                  <Box className="h-3.5 w-3.5" />
                </div>
                <span className="text-sm">3D</span>
              </button>
            </div>
            <p className="text-xs text-slate-400 mt-1.5">
              {is3D 
                ? "Immersive 3D visualization with spatial depth and rotation"
                : "Flat 2D layout optimized for clarity and performance"
              }
            </p>
          </div>
        </CardContent>
        </div>
      </Card>

      {/* Performance Limits */}
      {onNodeLimitChange && onEdgeLimitChange && onUseLimitsChange && (
        <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-orange-500/20 transition-all duration-300">
          <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-orange-400 via-orange-500 to-orange-600 opacity-100">
            <div className="w-full h-full bg-background rounded-lg" />
          </div>
          <div className="relative">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-3">
              <div className="rounded-2xl bg-gradient-to-br from-orange-500 to-orange-600 p-3 shadow-lg">
                <Settings className="h-5 w-5 text-white" />
              </div>
              <div className="flex-1">
                <CardTitle className="text-base font-semibold">Performance Limits</CardTitle>
                <CardDescription className="text-xs mt-0.5">Control graph size for better performance</CardDescription>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <Label htmlFor="use-limits" className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                Enable Limits
              </Label>
              <Switch
                id="use-limits"
                checked={useLimits}
                onCheckedChange={onUseLimitsChange}
              />
            </div>
            
            {useLimits && (
              <>
                <Separator className="bg-slate-300 dark:bg-slate-700/50" />
                
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="node-limit" className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                      Node Limit
                    </Label>
                    <Badge variant="secondary" className="text-[10px] bg-slate-200 dark:bg-slate-800 border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-300">
                      {nodeLimit}
                    </Badge>
                  </div>
                  <Input
                    id="node-limit"
                    type="number"
                    min={10}
                    max={1000}
                    step={10}
                    defaultValue={nodeLimit}
                    onChange={(e) => {
                      const val = e.target.value
                      if (val === '') return // Allow clearing
                      const parsed = parseInt(val)
                      if (!isNaN(parsed) && parsed >= 10 && parsed <= 1000) {
                        onNodeLimitChange(parsed)
                      }
                    }}
                    onBlur={(e) => {
                      if (e.target.value === '' || parseInt(e.target.value) < 10) {
                        onNodeLimitChange(1000) // Reset to default on blur if empty or invalid
                        e.target.value = '1000'
                      }
                    }}
                    className="border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 focus:ring-2 focus:ring-orange-500/50 focus:border-orange-500"
                  />
                  <p className="text-xs text-slate-400">
                    Maximum nodes to fetch (sorted by weight)
                  </p>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label htmlFor="edge-limit" className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                      Edge Limit
                    </Label>
                    <Badge variant="secondary" className="text-[10px] bg-slate-200 dark:bg-slate-800 border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-300">
                      {edgeLimit > 0 ? edgeLimit : 'None'}
                    </Badge>
                  </div>
                  <Input
                    id="edge-limit"
                    type="number"
                    min={0}
                    max={5000}
                    step={50}
                    defaultValue={edgeLimit}
                    onChange={(e) => {
                      const val = e.target.value
                      if (val === '') return // Allow clearing
                      const parsed = parseInt(val)
                      if (!isNaN(parsed) && parsed >= 0 && parsed <= 5000) {
                        onEdgeLimitChange(parsed)
                      }
                    }}
                    onBlur={(e) => {
                      if (e.target.value === '' || parseInt(e.target.value) < 0) {
                        onEdgeLimitChange(3000) // Reset to default on blur if empty or invalid
                        e.target.value = '3000'
                      }
                    }}
                    placeholder="0 = No limit"
                    className="border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 focus:ring-2 focus:ring-orange-500/50 focus:border-orange-500"
                  />
                  <p className="text-xs text-slate-400">
                    Maximum edges between limited nodes (0 = no limit)
                  </p>
                </div>
                
                <Separator className="bg-slate-300 dark:bg-slate-700/50" />
                
                {onApplyLimits && (
                  <Button
                    onClick={onApplyLimits}
                    className="w-full bg-gradient-to-br from-orange-500 to-orange-600 hover:from-orange-600 hover:to-orange-700 text-white shadow-lg hover:shadow-xl transition-all"
                  >
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Apply Limits
                  </Button>
                )}
              </>
            )}
          </CardContent>
          </div>
        </Card>
      )}
        </TabsContent>

        {/* FILTER TAB */}
        <TabsContent value="filter" className="space-y-4">
      {/* Legend & Filtering */}
      {colorByType && (
        <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-pink-500/20 transition-all duration-300">
          <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-pink-400 via-rose-500 to-rose-600 opacity-100">
            <div className="w-full h-full bg-background rounded-lg" />
          </div>
          <div className="relative">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="rounded-2xl bg-gradient-to-br from-pink-500 to-rose-600 p-3 shadow-lg">
                  <Layers className="h-5 w-5 text-white" />
                </div>
                <div className="flex-1">
                  <CardTitle className="text-base font-semibold">Legend & Filtering</CardTitle>
                  <CardDescription className="text-xs mt-0.5">Click to toggle visibility</CardDescription>
                </div>
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-5">
            {/* Node Types */}
            {nodeTypes.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <p className="text-xs font-semibold text-slate-700 dark:text-slate-300">Node Types</p>
                  <Badge variant="secondary" className="text-[10px] bg-slate-200 dark:bg-slate-800 border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-300">
                    {nodeTypes.length} types
                  </Badge>
                </div>
                <div className="space-y-1.5">
                  {nodeTypes.map((nodeType) => {
                    const isVisible = visibleNodeTypes?.has(nodeType) ?? true

                    // Render different shapes based on node type - matching graph rendering exactly
                    const renderShape = () => {
                      const baseColor = nodeTypeColors[nodeType] || "bg-gray-500"
                      const containerClasses = `flex items-center justify-center w-5 h-5 transition-all duration-200 ${isVisible ? 'scale-100' : 'scale-75 opacity-50'}`

                      switch (nodeType) {
                        case 'Node':
                          // Square (matching ctx.rect in graph)
                          return (
                            <div className={containerClasses}>
                              <div className={`w-3.5 h-3.5 ${baseColor}`} />
                            </div>
                          )
                        case 'Application':
                          // Circle (matching ctx.arc in graph)
                          return (
                            <div className={containerClasses}>
                              <div className={`w-3.5 h-3.5 ${baseColor} rounded-full`} />
                            </div>
                          )
                        case 'Topic':
                          // Diamond (matching 4-point lineTo path in graph)
                          return (
                            <div className={containerClasses}>
                              <div
                                className={`w-3.5 h-3.5 ${baseColor}`}
                                style={{
                                  clipPath: 'polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)'
                                }}
                              />
                            </div>
                          )
                        case 'Broker':
                          // Hexagon (matching 6-point path in graph)
                          return (
                            <div className={containerClasses}>
                              <div
                                className={`w-3.5 h-3.5 ${baseColor}`}
                                style={{
                                  clipPath: 'polygon(50% 0%, 93% 25%, 93% 75%, 50% 100%, 7% 75%, 7% 25%)'
                                }}
                              />
                            </div>
                          )
                        case 'Library':
                          // Triangle (matching 3-point path in graph)
                          return (
                            <div className={containerClasses}>
                              <div
                                className={`w-3.5 h-3.5 ${baseColor}`}
                                style={{
                                  clipPath: 'polygon(50% 0%, 100% 100%, 0% 100%)'
                                }}
                              />
                            </div>
                          )
                        default:
                          // Circle for unknown types (matching default case in graph)
                          return (
                            <div className={containerClasses}>
                              <div className={`w-3.5 h-3.5 ${baseColor} rounded-full`} />
                            </div>
                          )
                      }
                    }

                    return (
                      <div
                        key={nodeType}
                        className={`flex items-center gap-2.5 cursor-pointer hover:bg-green-500/10 rounded-md px-2 py-2 transition-all duration-200 group ${!isVisible ? 'opacity-50' : 'opacity-100'} border border-transparent hover:border-green-500/30`}
                        onClick={() => onToggleNodeType?.(nodeType)}
                      >
                        <div className="flex-shrink-0">{renderShape()}</div>
                        <span className="flex-1 text-xs font-semibold group-hover:text-green-600 dark:group-hover:text-green-300 transition-colors">
                          {nodeType}
                        </span>
                        <div className={`flex-shrink-0 w-4 h-4 rounded-full border-2 flex items-center justify-center transition-all ${
                          isVisible 
                            ? 'border-green-500 bg-green-500' 
                            : 'border-slate-600 bg-transparent'
                        }`}>
                          {isVisible && (
                            <svg className="w-2.5 h-2.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                            </svg>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Link Types */}
            {uniqueLinkTypesWithColors.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <p className="text-xs font-semibold text-slate-700 dark:text-slate-300">Link Types</p>
                  <Badge variant="secondary" className="text-[10px] bg-slate-200 dark:bg-slate-800 border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-300">
                    {uniqueLinkTypesWithColors.length} types
                  </Badge>
                </div>
                <div className="space-y-1.5">
                  {uniqueLinkTypesWithColors.map((item) => {
                    const isVisible = visibleLinkTypes?.has(item.display) ?? true
                    return (
                      <div
                        key={item.display}
                        className={`flex items-center gap-2.5 cursor-pointer hover:bg-green-500/10 rounded-md px-2 py-2 transition-all duration-200 group ${!isVisible ? 'opacity-50' : 'opacity-100'} border border-transparent hover:border-green-500/30`}
                        onClick={() => onToggleLinkType?.(item.display)}
                      >
                        <div className="flex-shrink-0">
                          <div className={`h-1 w-5 rounded-full ${linkTypeColors[item.original] || "bg-gray-500"} transition-all duration-200 ${isVisible ? 'scale-100' : 'scale-75 opacity-50'}`} />
                        </div>
                        <span className="flex-1 text-xs font-semibold group-hover:text-green-600 dark:group-hover:text-green-300 transition-colors">
                          {formatLinkType(item.display)}
                        </span>
                        <div className={`flex-shrink-0 w-4 h-4 rounded-full border-2 flex items-center justify-center transition-all ${
                          isVisible 
                            ? 'border-green-500 bg-green-500' 
                            : 'border-slate-600 bg-transparent'
                        }`}>
                          {isVisible && (
                            <svg className="w-2.5 h-2.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                            </svg>
                          )}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Pub/Sub Visibility Controls */}
            <div className="space-y-3 pt-3">
              <div className="flex items-center justify-between">
                <Label htmlFor="pub-sub-toggle" className="text-xs font-semibold text-slate-700 dark:text-slate-300">Zoom-Based Filtering</Label>
                {onTogglePubSubEnabled && (
                  <Switch
                    id="pub-sub-toggle"
                    checked={isPubSubEnabled}
                    onCheckedChange={onTogglePubSubEnabled}
                    className="data-[state=checked]:bg-green-500"
                  />
                )}
              </div>
              <p className="text-[10px] text-slate-500 dark:text-slate-400 leading-tight">
                {isPubSubEnabled ? 'Pub/sub links use zoom threshold in complete view' : 'Pub/sub links always visible'}
              </p>
              
              {isPubSubEnabled && currentView === 'complete' && (
                <div className="space-y-2 pl-1">
                  <div className="flex items-center justify-between">
                    <Label className="text-xs text-slate-600 dark:text-slate-400">Zoom Threshold</Label>
                    <Badge 
                      variant={currentZoom >= pubSubZoomThreshold ? "default" : "secondary"}
                      className={`text-[10px] h-5 ${currentZoom >= pubSubZoomThreshold ? 'bg-green-500 border-green-500 text-white' : 'bg-amber-500 border-amber-500 text-white'}`}
                    >
                      {currentZoom >= pubSubZoomThreshold ? 'Visible' : 'Hidden'}
                    </Badge>
                  </div>
                  <p className="text-[10px] text-slate-500 dark:text-slate-400 leading-tight">
                    Pub/sub links appear only when zoomed beyond threshold
                  </p>

                  {onPubSubZoomThresholdChange && (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-slate-500 dark:text-slate-400">Threshold</span>
                        <span className="text-xs font-mono font-semibold text-slate-700 dark:text-slate-300">
                          {pubSubZoomThreshold.toFixed(1)}x
                        </span>
                      </div>
                      <input
                        id="zoom-threshold"
                        type="range"
                        min="1"
                        max="5"
                        step="0.1"
                        value={pubSubZoomThreshold}
                        onChange={(e) => onPubSubZoomThresholdChange(parseFloat(e.target.value))}
                        className="w-full h-2 rounded-lg appearance-none cursor-pointer bg-slate-700 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-slate-300 [&::-webkit-slider-thumb]:cursor-pointer [&::-webkit-slider-thumb]:transition-all [&::-webkit-slider-thumb]:hover:bg-white [&::-moz-range-thumb]:w-3 [&::-moz-range-thumb]:h-3 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-slate-300 [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:cursor-pointer [&::-moz-range-thumb]:transition-all [&::-moz-range-thumb]:hover:bg-white"
                      />
                    </div>
                  )}
                </div>
              )}
            </div>
          </CardContent>
          </div>
        </Card>
      )}

      {/* Graph Stats */}
      <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-purple-500/20 transition-all duration-300">
        <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-purple-400 via-pink-500 to-pink-600 opacity-100">
          <div className="w-full h-full bg-background rounded-lg" />
        </div>
        <div className="relative">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="rounded-2xl bg-gradient-to-br from-purple-500 to-pink-600 p-3 shadow-lg">
                <Layers className="h-5 w-5 text-white" />
              </div>
              <div className="flex-1">
                <CardTitle className="text-base font-semibold">Graph Statistics</CardTitle>
                <CardDescription className="text-xs mt-0.5">Current graph composition</CardDescription>
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="space-y-3">
            <div className="rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 p-3 hover:bg-slate-100 dark:hover:bg-slate-800 transition-all">
              <div className="flex items-center gap-2 mb-2">
                <div className="rounded-lg bg-purple-500/20 p-1.5">
                  <Layers className="h-3 w-3 text-purple-600 dark:text-purple-400" />
                </div>
                <span className="text-xs font-semibold">Nodes</span>
              </div>
              <p className="text-2xl font-bold text-purple-600 dark:text-purple-400 mb-2">{nodeCount}</p>
              {Object.keys(nodeTypeCounts).length > 0 && (
                <div className="space-y-1.5">
                  {Object.entries(nodeTypeCounts)
                    .sort(([, a], [, b]) => b - a)
                    .map(([type, count]) => (
                      <div key={type} className="flex items-center justify-between text-xs">
                        <span className="text-slate-600 dark:text-slate-400 truncate flex-1 font-medium">{type}</span>
                        <Badge variant="secondary" className="ml-2 text-[10px] px-1.5 py-0.5 h-5 font-semibold bg-emerald-500/20 text-emerald-800 dark:text-emerald-300 border-emerald-500/30">
                          {count}
                        </Badge>
                      </div>
                    ))}
                </div>
              )}
            </div>
            <div className="rounded-lg border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 p-3 hover:bg-slate-100 dark:hover:bg-slate-800 transition-all">
              <div className="flex items-center gap-2 mb-2">
                <div className="rounded-lg bg-purple-500/20 p-1.5">
                  <Box className="h-3 w-3 text-purple-600 dark:text-purple-400" />
                </div>
                <span className="text-xs font-semibold">Links</span>
              </div>
              <p className="text-2xl font-bold text-purple-600 dark:text-purple-400 mb-2">{linkCount}</p>
              {Object.keys(linkTypeCounts).length > 0 && (
                <div className="space-y-1.5">
                  {Object.entries(linkTypeCounts)
                    .sort(([, a], [, b]) => b - a)
                    .map(([type, count]) => (
                      <div key={type} className="flex items-center justify-between text-xs">
                        <span className="text-muted-foreground truncate flex-1 font-medium">{type}</span>
                        <Badge variant="secondary" className="ml-2 text-[10px] px-1.5 py-0.5 h-5 font-semibold bg-purple-500/20 text-purple-800 dark:text-purple-300 border-purple-500/30">
                          {count}
                        </Badge>
                      </div>
                    ))}
                </div>
              )}
            </div>
          </div>
          <p className="text-xs text-muted-foreground text-center">
            Breakdown by type in current view
          </p>
        </CardContent>
        </div>
      </Card>
        </TabsContent>

        {/* ACTIONS TAB */}
        <TabsContent value="actions" className="space-y-4">
      {/* Actions */}
      <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-slate-500/20 transition-all duration-300">
        <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-br from-slate-400 via-slate-600 to-slate-700 opacity-100">
          <div className="w-full h-full bg-background rounded-lg" />
        </div>
        <div className="relative">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-3">
            <div className="rounded-2xl bg-gradient-to-br from-slate-600 to-slate-700 p-3 shadow-lg">
              <Settings className="h-5 w-5 text-white" />
            </div>
            <div className="flex-1">
              <CardTitle className="text-base font-semibold">Actions</CardTitle>
              <CardDescription className="text-xs mt-0.5">Graph utilities and exports</CardDescription>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-2">
          <Button 
            onClick={onRefresh} 
            disabled={isLoading} 
            variant="outline" 
            className="w-full h-9 border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 hover:bg-slate-100 dark:hover:bg-slate-800 hover:border-slate-400 dark:hover:border-slate-600 hover:shadow-md transition-all"
          >
            <RefreshCw className={`mr-2 h-4 w-4 ${isLoading ? "animate-spin" : ""}`} />
            Refresh Graph
          </Button>
          {onFitGraph && (
            <Button 
              onClick={onFitGraph} 
              variant="outline" 
              className="w-full h-9 border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 hover:bg-slate-100 dark:hover:bg-slate-800 hover:border-slate-400 dark:hover:border-slate-600 hover:shadow-md transition-all"
            >
              <Maximize2 className="mr-2 h-4 w-4" />
              Fit to View
            </Button>
          )}
          {onExportPNG && (
            <Button 
              onClick={onExportPNG} 
              variant="outline" 
              className="w-full h-9 border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 hover:bg-slate-100 dark:hover:bg-slate-800 hover:border-slate-400 dark:hover:border-slate-600 hover:shadow-md transition-all"
            >
              <Download className="mr-2 h-4 w-4" />
              Export as PNG
            </Button>
          )}
          {onExportCSV && (
            <Button 
              onClick={onExportCSV} 
              variant="outline" 
              className="w-full h-9 border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-slate-800/50 hover:bg-slate-100 dark:hover:bg-slate-800 hover:border-slate-400 dark:hover:border-slate-600 hover:shadow-md transition-all"
            >
              <FileSpreadsheet className="mr-2 h-4 w-4" />
              Export as CSV
            </Button>
          )}
        </CardContent>
        </div>
      </Card>
        </TabsContent>
      </Tabs>
      </div>
    </div>
  )
}
