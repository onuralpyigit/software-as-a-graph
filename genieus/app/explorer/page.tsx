"use client"

import React, { useState, useRef, useMemo, useEffect, Suspense } from "react"
import { useTheme } from "next-themes"
import { useRouter, useSearchParams } from "next/navigation"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { ForceGraph2DWrapper, type ForceGraph2DRef } from "@/components/graph/force-graph-2d-wrapper"
import { ForceGraph3DWrapper, type ForceGraph3DRef } from "@/components/graph/force-graph-3d-wrapper"
import { GraphKeyboardShortcuts } from "@/components/graph/graph-keyboard-shortcuts"
import { KeyboardShortcutsHelp } from "@/components/graph/keyboard-shortcuts-help"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import { useConnection } from "@/lib/stores/connection-store"
import type { ForceGraphData, GraphNode, GraphLink } from "@/lib/types/api"
import type { GraphView } from "@/lib/types/graph-views"
import { GRAPH_VIEWS } from "@/lib/types/graph-views"
import { Button } from "@/components/ui/button"
import { Loader2, ChevronRight, X, Box, Grid3x3, Maximize2, RefreshCw, Download, Network, Layers, AppWindow, Server, GitBranch, Database, Info } from "lucide-react"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

// Hooks
import { useGraphData } from "./hooks/useGraphData"
import { useGraphFiltering } from "./hooks/useGraphFiltering"
import { useGraphSearch } from "./hooks/useGraphSearch"
import { useGraphDrillDown } from "./hooks/useGraphDrillDown"

// Components
import { NodeDetailsCard } from "./components/NodeDetailsCard"
import { LinkDetailsCard } from "./components/LinkDetailsCard"
import { SearchBar } from "./components/SearchBar"
import { BreadcrumbNavigation } from "./components/BreadcrumbNavigation"
import { HierarchicalTree } from "./components/HierarchicalTree"

// Utils
import { getNodeIcon, getLinkId, getNodeColorMap, getLinkColorMap } from "./utils"

function ExplorerContent() {
  const router = useRouter()
  const { status, initialLoadComplete } = useConnection()
  const isConnected = status === 'connected'
  const { theme, systemTheme } = useTheme()
  const currentTheme = theme === 'system' ? systemTheme : theme
  const searchParams = useSearchParams()

  // Set page title
  useEffect(() => {
    document.title = 'Explorer - Genieus'
  }, [])

  // View state
  const [currentView, setCurrentView] = useState<GraphView>('complete')

  // Graph data management
  const { graphData, isLoading, error, fetchTopologyData, setGraphData } = useGraphData(isConnected, currentView)
  const [viewGraphData, setViewGraphData] = useState<ForceGraphData | null>(null)
  
  // Selection state
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [selectedLink, setSelectedLink] = useState<GraphLink | null>(null)
  
  // Filtering
  const dataToFilter = viewGraphData || graphData
  const { filteredGraphData, hiddenNodeTypes, hiddenLinkTypes, uniqueNodeTypes, uniqueLinkTypes, toggleNodeType, toggleLinkType } = useGraphFiltering(dataToFilter, selectedNode)
  
  // Search
  const { searchQuery, searchResults, showSearchResults, isSearching, handleSearch, handleClearSearch, setShowSearchResults } = useGraphSearch(isConnected, currentView)
  
  // Drill-down navigation
  const { drillDownHistory, hierarchyData, breadcrumbPath, expandedNodes, isLoadingChildren, drillDown, navigateToBreadcrumb, reset, toggleNodeExpansion } = useGraphDrillDown(currentView)
  
  // 3D view toggle
  const [is3DView, setIs3DView] = useState(false)
  
  // Graph ref
  const graphRef = useRef<ForceGraph2DRef | ForceGraph3DRef>(null)

  // Track if URL parameter has been processed
  const [urlNodeProcessed, setUrlNodeProcessed] = useState<string | null>(null)

  // Handle URL parameter for initial node selection - act like search result click
  useEffect(() => {
    const nodeId = searchParams.get('node')
    if (nodeId && isConnected && !isLoading && nodeId !== urlNodeProcessed) {
      setUrlNodeProcessed(nodeId)
      
      // Reset view state
      setViewGraphData(null)
      setSelectedNode(null)
      setSelectedLink(null)
      
      // Drill into the node as a fresh root (like search result click)
      // The drillDown function will fetch the node from the API directly by ID
      const performDrillDown = async () => {
        try {
          const result = await drillDown(nodeId, graphData, true)
          if (result) {
            setViewGraphData(result)
          }
        } catch (error) {
          console.error('Failed to drill down to node from URL:', nodeId, error)
        }
      }
      performDrillDown()
    }
  }, [searchParams, isConnected, isLoading, drillDown, urlNodeProcessed, graphData])

  // Color mappings
  const nodeColorByType = useMemo(() => getNodeColorMap(currentTheme), [currentTheme])
  const linkColorByType = useMemo(() => getLinkColorMap(currentTheme), [currentTheme])
  
  // Get connection count for a node
  const getConnectionCount = (nodeId: string): number => {
    if (!dataToFilter) return 0
    return dataToFilter.links.filter(link => {
      const sourceId = typeof link.source === 'string' ? link.source : (link.source as any).id
      const targetId = typeof link.target === 'string' ? link.target : (link.target as any).id
      return sourceId === nodeId || targetId === nodeId
    }).length
  }

  // Get connection type between parent and child
  const getConnectionType = (parentId: string, childId: string): string | null => {
    const parentData = hierarchyData.get(parentId)
    if (parentData?.links) {
      const connection = parentData.links.find(link => {
        const sourceId = typeof link.source === 'string' ? link.source : (link.source as any).id
        const targetId = typeof link.target === 'string' ? link.target : (link.target as any).id
        return (sourceId === parentId && targetId === childId) || (sourceId === childId && targetId === parentId)
      })
      if (connection) return connection.type || null
    }
    
    if (!dataToFilter) return null
    const connection = dataToFilter.links.find(link => {
      const sourceId = typeof link.source === 'string' ? link.source : (link.source as any).id
      const targetId = typeof link.target === 'string' ? link.target : (link.target as any).id
      return (sourceId === parentId && targetId === childId) || (sourceId === childId && targetId === parentId)
    })
    return connection?.type || null
  }

  // Handle node click
  const handleNodeClick = async (node: GraphNode) => {
    setSelectedNode(node)
    setSelectedLink(null)
    
    // Check if node is in the current drill-down path
    const existingIndex = drillDownHistory.findIndex(h => h.nodeId === node.id)
    
    if (existingIndex >= 0) {
      // Node is already in the drill path - navigate back to it
      const result = navigateToBreadcrumb(node.id)
      if (result) {
        setViewGraphData(result)
      }
    } else if (drillDownHistory.length > 0) {
      // Check if node is a child of the current drill level
      const currentLevel = drillDownHistory[drillDownHistory.length - 1]
      const currentLevelData = hierarchyData.get(currentLevel.nodeId)
      const isChildOfCurrent = currentLevelData?.children.some(child => child.id === node.id)
      
      if (isChildOfCurrent) {
        // Node is a child of current level - drill down into it
        const result = await drillDown(node.id, graphData, false)
        if (result) {
          setViewGraphData(result)
          // Keep node selected for detail card
        }
      } else {
        // Node is not in current tree view - expand it in the explorer
        // by drilling down from it as a new root
        const result = await drillDown(node.id, graphData, true)
        if (result) {
          setViewGraphData(result)
          // Keep node selected for detail card
        }
      }
    } else {
      // No drill-down active - expand this node in the explorer
      const result = await drillDown(node.id, graphData, true)
      if (result) {
        setViewGraphData(result)
        // Keep node selected for detail card
      }
    }
  }

  // Handle link click
  const handleLinkClick = (link: GraphLink) => {
    setSelectedLink(link)
    setSelectedNode(null)
  }

  // Handle background click
  const handleBackgroundClick = () => {
    setSelectedNode(null)
    setSelectedLink(null)
    setShowSearchResults(false)
  }

  // Handle search result click
  const handleSearchResultClick = async (node: GraphNode) => {
    setShowSearchResults(false)
    
    // Reset view state
    setViewGraphData(null)
    setSelectedNode(null)
    setSelectedLink(null)
    
    // Drill into the searched node as a fresh root
    const result = await drillDown(node.id, graphData, true)
    if (result) {
      setViewGraphData(result)
    }
  }

  // Handle breadcrumb click
  const handleBreadcrumbClick = async (nodeId: string | null) => {
    if (nodeId === null) {
      handleReset()
      return
    }
    
    const result = navigateToBreadcrumb(nodeId)
    if (result) {
      setViewGraphData(result)
      setSelectedNode(null)
      setSelectedLink(null)
    }
    // If result is null, user clicked current breadcrumb - do nothing
  }

  // Handle reset
  const handleReset = () => {
    reset()
    setSelectedNode(null)
    setSelectedLink(null)
    setViewGraphData(null)
    // Don't fetch again if we already have the data
    if (!graphData) {
      fetchTopologyData()
    }
  }

  // Handle view change
  const handleViewChange = (view: GraphView) => {
    setCurrentView(view)
    // Reset drill-down state when changing views
    reset()
    setSelectedNode(null)
    setSelectedLink(null)
    setViewGraphData(null)
    // Clear search results when changing views
    handleClearSearch()
  }

  // Get icon for each view
  const getViewIcon = (viewId: GraphView) => {
    switch (viewId) {
      case 'complete':
        return Network
      case 'application':
        return AppWindow
      case 'infrastructure':
        return Server
      case 'middleware':
        return GitBranch
      default:
        return Network
    }
  }

  // Keyboard shortcut handlers
  const handleFitGraph = () => {
    if (graphRef.current) {
      graphRef.current.zoomToFit(800, 50)
    }
  }

  const handleToggle3D = () => {
    setIs3DView(prev => !prev)
  }

  const handleRefresh = () => {
    fetchTopologyData()
  }

  const handleExportPNG = () => {
    if (graphRef.current) {
      const now = new Date()
      const dateStr = now.toISOString().slice(0, 10)
      const timeStr = now.toTimeString().slice(0, 8).replace(/:/g, '-')
      const viewMode = is3DView ? '3d' : '2d'
      const filename = `topology-${viewMode}-${dateStr}_${timeStr}.png`
      graphRef.current.exportPNG(filename)
    }
  }

  const handleClearSelection = () => {
    setSelectedNode(null)
    setSelectedLink(null)
  }

  if (!initialLoadComplete || status === 'connecting') {
    return (
      <AppLayout title="Explorer" description="Explore and navigate your distributed system">
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" text={status === 'connecting' ? "Connecting to database..." : "Loading topology..."} />
        </div>
      </AppLayout>
    )
  }

  if (!isConnected) {
    return (
      <AppLayout title="Explorer" description="Explore and navigate your distributed system">
        <NoConnectionInfo />
      </AppLayout>
    )
  }

  // Empty database state
  if (graphData && graphData.nodes.length === 0) {
    return (
      <AppLayout title="Explorer" description="Explore and navigate your distributed system">
        <div className="w-full">
          <Card className="border-2 border-purple-500/50 dark:border-purple-500/50 bg-white/95 dark:bg-black/95 backdrop-blur-md shadow-2xl shadow-purple-500/20 hover:shadow-purple-500/30 hover:border-purple-500/70 transition-all duration-300 overflow-hidden">
            {/* Decorative top border */}
            <div className="h-1 w-full bg-gradient-to-r from-purple-500 via-pink-500 to-purple-500" />

            <CardHeader className="pb-6 pt-8 px-8">
              <div className="flex flex-col sm:flex-row items-start sm:items-center gap-5">
                {/* Icon with animated gradient */}
                <div className="relative group">
                  <div className="absolute inset-0 bg-gradient-to-br from-purple-500 to-purple-600 rounded-2xl blur-xl opacity-30 group-hover:opacity-50 transition-opacity duration-300" />
                  <div className="relative rounded-2xl bg-gradient-to-br from-purple-500/20 to-purple-600/20 dark:from-purple-500/30 dark:to-purple-600/30 p-4 ring-1 ring-purple-500/30 group-hover:ring-purple-500/50 transition-all duration-300">
                    <Database className="h-8 w-8 text-purple-600 dark:text-purple-400 group-hover:scale-110 transition-transform duration-300" />
                  </div>
                </div>

                {/* Title section */}
                <div className="flex-1 space-y-1.5">
                  <CardTitle className="text-2xl font-bold tracking-tight">Empty Database</CardTitle>
                  <CardDescription className="text-base text-muted-foreground">
                    No graph data available
                  </CardDescription>
                </div>
              </div>
            </CardHeader>

            <CardContent className="px-8 pb-8 space-y-6">
              {/* Information box with steps */}
              <div className="rounded-2xl bg-gradient-to-br from-muted/40 via-muted/20 to-muted/10 border border-border/40 p-6 space-y-5">
                <div className="flex items-center gap-2.5">
                  <div className="rounded-xl bg-gradient-to-br from-blue-500/20 to-blue-600/20 dark:from-blue-500/30 dark:to-blue-600/30 p-2.5 ring-1 ring-blue-500/30">
                    <Info className="h-5 w-5 text-blue-600 dark:text-blue-400" />
                  </div>
                  <h3 className="font-semibold text-base text-foreground">How to Populate Your Database</h3>
                </div>

                {/* Step-by-step list */}
                <div className="space-y-3 pl-1">
                  <div className="flex items-start gap-3.5 group/item">
                    <div className="flex-shrink-0 w-7 h-7 rounded-full bg-gradient-to-br from-purple-500/20 to-purple-600/20 dark:from-purple-500/25 dark:to-purple-600/25 flex items-center justify-center ring-1 ring-purple-500/30 group-hover/item:ring-purple-500/50 transition-all duration-200">
                      <span className="text-xs font-bold text-purple-600 dark:text-purple-400">1</span>
                    </div>
                    <p className="text-sm text-muted-foreground leading-relaxed pt-0.5 group-hover/item:text-foreground transition-colors duration-200">
                      Navigate to the Data Management page
                    </p>
                  </div>
                  <div className="flex items-start gap-3.5 group/item">
                    <div className="flex-shrink-0 w-7 h-7 rounded-full bg-gradient-to-br from-purple-500/20 to-purple-600/20 dark:from-purple-500/25 dark:to-purple-600/25 flex items-center justify-center ring-1 ring-purple-500/30 group-hover/item:ring-purple-500/50 transition-all duration-200">
                      <span className="text-xs font-bold text-purple-600 dark:text-purple-400">2</span>
                    </div>
                    <p className="text-sm text-muted-foreground leading-relaxed pt-0.5 group-hover/item:text-foreground transition-colors duration-200">
                      Upload your system architecture configuration files
                    </p>
                  </div>
                  <div className="flex items-start gap-3.5 group/item">
                    <div className="flex-shrink-0 w-7 h-7 rounded-full bg-gradient-to-br from-purple-500/20 to-purple-600/20 dark:from-purple-500/25 dark:to-purple-600/25 flex items-center justify-center ring-1 ring-purple-500/30 group-hover/item:ring-purple-500/50 transition-all duration-200">
                      <span className="text-xs font-bold text-purple-600 dark:text-purple-400">3</span>
                    </div>
                    <p className="text-sm text-muted-foreground leading-relaxed pt-0.5 group-hover/item:text-foreground transition-colors duration-200">
                      Generate and import graph data into the database
                    </p>
                  </div>
                </div>
              </div>

              {/* CTA Button */}
              <div className="pt-2">
                <Button
                  onClick={() => router.push('/data')}
                  size="lg"
                  className="w-full bg-gradient-to-r from-purple-600 to-purple-700 hover:from-purple-700 hover:to-purple-800 text-white shadow-lg hover:shadow-xl transition-all duration-300 group"
                >
                  <Database className="mr-2 h-4 w-4" />
                  Populate Database
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </AppLayout>
    )
  }

  return (
    <AppLayout title="Explorer" description="Explore and navigate your distributed system">
      <div className="flex h-full gap-4">
        {/* Main content */}
        <div className="flex flex-1 flex-col gap-4">
          {/* Graph */}
          <Card className="flex-1 relative bg-white dark:bg-black border-2 border-gray-200 dark:border-white/10 p-0 overflow-hidden">
            <CardContent className="h-full p-0 relative bg-white dark:bg-black">
              {/* Overlay Controls */}
              <div className="absolute top-4 left-4 right-4 z-10 flex flex-col gap-2">
                {/* Top Row - Navigation and Stats */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 bg-background/95 backdrop-blur-md border-2 border-border rounded-lg px-3 py-2 shadow-xl h-[42px]">
                    {viewGraphData && drillDownHistory.length > 0 ? (
                      <>
                        <BreadcrumbNavigation
                          breadcrumbPath={breadcrumbPath}
                          onBreadcrumbClick={handleBreadcrumbClick}
                          onReset={handleReset}
                        />
                      </>
                    ) : (
                      <span className="text-xs text-muted-foreground font-medium">
                        System explorer - Select a node to explore
                      </span>
                    )}
                  </div>
                  
                  {graphData && (
                    <div className="flex items-center gap-3">
                      {/* View Selector */}
                      <div className="bg-background/95 backdrop-blur-md border-2 border-border rounded-lg shadow-xl">
                        <Select value={currentView} onValueChange={(value) => handleViewChange(value as GraphView)}>
                          <SelectTrigger className="h-[38px] border-0 bg-transparent hover:bg-accent/50 transition-colors px-3">
                            <SelectValue>
                              {(() => {
                                const viewConfig = GRAPH_VIEWS[currentView]
                                const Icon = getViewIcon(currentView)
                                return (
                                  <div className="flex items-center gap-2">
                                    <Icon className="h-3.5 w-3.5" style={{ color: viewConfig.color }} />
                                    <span className="text-xs font-medium">{viewConfig.name}</span>
                                  </div>
                                )
                              })()}
                            </SelectValue>
                          </SelectTrigger>
                          <SelectContent className="border-2 border-border bg-background/95 backdrop-blur-md">
                            {Object.values(GRAPH_VIEWS).map((view) => {
                              const Icon = getViewIcon(view.id)
                              return (
                                <SelectItem 
                                  key={view.id} 
                                  value={view.id}
                                  className="hover:bg-accent/50 focus:bg-accent/50"
                                >
                                  <div className="flex items-center gap-2">
                                    <Icon className="h-3.5 w-3.5" style={{ color: view.color }} />
                                    <span className="text-xs font-medium">{view.name}</span>
                                  </div>
                                </SelectItem>
                              )
                            })}
                          </SelectContent>
                        </Select>
                      </div>

                      {/* Stats */}
                      <div className="flex items-center gap-3 bg-background/95 backdrop-blur-md border-2 border-border rounded-lg px-3 py-2 shadow-xl text-xs text-muted-foreground h-[42px]">
                        <span>Nodes: <span className="font-medium text-foreground">{filteredGraphData?.nodes.length || 0}</span></span>
                        <span>Edges: <span className="font-medium text-foreground">{filteredGraphData?.links.length || 0}</span></span>
                        
                        <div className="h-4 w-px bg-border mx-1" />
                        
                        {/* 2D/3D Toggle */}
                        <div className="flex items-center gap-1">
                          <Button
                            variant={!is3DView ? "default" : "ghost"}
                            size="sm"
                            onClick={() => setIs3DView(false)}
                            className="h-6 px-2"
                            title="2D View"
                          >
                            <Grid3x3 className="h-3 w-3" />
                          </Button>
                          <Button
                            variant={is3DView ? "default" : "ghost"}
                            size="sm"
                            onClick={() => setIs3DView(true)}
                            className="h-6 px-2"
                            title="3D View"
                          >
                            <Box className="h-3 w-3" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Search Bar */}
                <SearchBar
                  searchQuery={searchQuery}
                  onSearch={handleSearch}
                  onClear={handleClearSearch}
                  isSearching={isSearching}
                  searchResults={searchResults}
                  showSearchResults={showSearchResults}
                  onResultClick={handleSearchResultClick}
                  getConnectionCount={getConnectionCount}
                  getNodeIcon={getNodeIcon}
                  nodeColorByType={nodeColorByType}
                  isConnected={isConnected}
                />
              </div>

              {isLoading && (
                <div className="flex h-full items-center justify-center">
                  <div className="flex flex-col items-center gap-2">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                    <p className="text-sm text-muted-foreground">Loading topology...</p>
                  </div>
                </div>
              )}

              {error && (
                <div className="flex h-full items-center justify-center">
                  <div className="text-center">
                    <p className="text-sm text-destructive">{error}</p>
                    <Button
                      variant="outline"
                      size="sm"
                      className="mt-4"
                      onClick={() => fetchTopologyData()}
                    >
                      Retry
                    </Button>
                  </div>
                </div>
              )}

              {!isLoading && !error && graphData && filteredGraphData && (
                <div className="h-full w-full">
                  {is3DView ? (
                    <ForceGraph3DWrapper
                      ref={graphRef as any}
                      nodes={filteredGraphData.nodes}
                      links={filteredGraphData.links}
                      onNodeClick={handleNodeClick}
                      onLinkClick={handleLinkClick}
                      onBackgroundClick={handleBackgroundClick}
                      selectedNodeId={selectedNode?.id}
                      selectedLinkId={selectedLink ? getLinkId(selectedLink) : null}
                      autoCenterOnSelect={false}
                      colorByType={true}
                      visibleNodeTypes={new Set(uniqueNodeTypes.filter(t => !hiddenNodeTypes.has(t)))}
                      visibleLinkTypes={new Set(uniqueLinkTypes.filter(t => !hiddenLinkTypes.has(t)))}
                    />
                  ) : (
                    <ForceGraph2DWrapper
                      ref={graphRef as any}
                      nodes={filteredGraphData.nodes}
                      links={filteredGraphData.links}
                      onNodeClick={handleNodeClick}
                      onLinkClick={handleLinkClick}
                      onBackgroundClick={handleBackgroundClick}
                      selectedNodeId={selectedNode?.id}
                      selectedLinkId={selectedLink ? getLinkId(selectedLink) : null}
                      autoCenterOnSelect={false}
                      colorByType={true}
                      visibleNodeTypes={new Set(uniqueNodeTypes.filter(t => !hiddenNodeTypes.has(t)))}
                      visibleLinkTypes={new Set(uniqueLinkTypes.filter(t => !hiddenLinkTypes.has(t)))}
                    />
                  )}
                  
                  {/* Node Details Card */}
                  {selectedNode && (
                    <NodeDetailsCard
                      node={selectedNode}
                      onClose={() => setSelectedNode(null)}
                      getConnectionCount={getConnectionCount}
                      getNodeIcon={getNodeIcon}
                      nodeColorByType={nodeColorByType}
                    />
                  )}
                  
                  {/* Link Details Card */}
                  {selectedLink && (
                    <LinkDetailsCard
                      link={selectedLink}
                      graphData={filteredGraphData}
                      onClose={() => setSelectedLink(null)}
                      onNodeClick={handleNodeClick}
                      getNodeIcon={getNodeIcon}
                      nodeColorByType={nodeColorByType}
                      linkColorByType={linkColorByType}
                    />
                  )}
                  
                  {/* Legend */}
                  <div className="absolute bottom-4 left-4 z-10 bg-background/95 backdrop-blur-md border border-border rounded-lg px-2.5 py-2 shadow-lg max-w-[200px]">
                    <h4 className="text-[11px] font-semibold mb-1.5">Legend</h4>
                    
                    {/* Node Types */}
                    {uniqueNodeTypes.length > 0 && (
                      <div className="mb-1.5">
                        <div className="text-[9px] text-muted-foreground mb-1 font-medium">Node Types</div>
                        <div className="space-y-0.5">
                          {uniqueNodeTypes.map((type) => {
                            const color = nodeColorByType[type as keyof typeof nodeColorByType] || nodeColorByType.Unknown
                            const isHidden = hiddenNodeTypes.has(type)
                            return (
                              <button
                                key={type}
                                onClick={() => toggleNodeType(type)}
                                className={`flex w-full items-center gap-1.5 px-1 py-0.5 rounded hover:bg-accent/50 transition-colors ${
                                  isHidden ? 'opacity-40' : ''
                                }`}
                                title={isHidden ? `Click to show ${type}` : `Click to hide ${type}`}
                              >
                                <svg width="10" height="10" viewBox="0 0 10 10" className="flex-shrink-0">
                                  {type === 'Node' && <rect x="1" y="1" width="8" height="8" fill={color} />}
                                  {type === 'Application' && <circle cx="5" cy="5" r="4" fill={color} />}
                                  {type === 'Topic' && <polygon points="5,1 9,5 5,9 1,5" fill={color} />}
                                  {type === 'Library' && <polygon points="5,1 9,8 1,8" fill={color} />}
                                  {type === 'Broker' && <polygon points="5,0.5 8.3,2.5 8.3,7.5 5,9.5 1.7,7.5 1.7,2.5" fill={color} />}
                                  {!['Node', 'Application', 'Topic', 'Library', 'Broker'].includes(type) && <circle cx="5" cy="5" r="4" fill={color} />}
                                  {isHidden && <line x1="1" y1="1" x2="9" y2="9" stroke="currentColor" strokeWidth="1.5" opacity="0.6" />}
                                </svg>
                                <span className={`text-[10px] ${isHidden ? 'line-through' : ''}`}>{type}</span>
                              </button>
                            )
                          })}
                        </div>
                      </div>
                    )}
                    
                    {/* Edge Types */}
                    {uniqueLinkTypes.length > 0 && (
                      <div>
                        <div className="text-[9px] text-muted-foreground mb-1 font-medium">Edge Types</div>
                        <div className="space-y-0.5">
                          {uniqueLinkTypes.map((type) => {
                            const isHidden = hiddenLinkTypes.has(type)
                            return (
                              <button
                                key={type}
                                onClick={() => toggleLinkType(type)}
                                className={`flex w-full items-center gap-1.5 px-1 py-0.5 rounded hover:bg-accent/50 transition-colors ${
                                  isHidden ? 'opacity-40' : ''
                                }`}
                                title={isHidden ? `Click to show ${type}` : `Click to hide ${type}`}
                              >
                                <div className="relative flex-shrink-0">
                                  <div className="w-2.5 h-0.5" style={{ backgroundColor: linkColorByType[type as keyof typeof linkColorByType] || '#888' }} />
                                  {isHidden && (
                                    <div className="absolute inset-0 flex items-center justify-center">
                                      <div className="w-3 h-px bg-current transform rotate-45" />
                                    </div>
                                  )}
                                </div>
                                <span className={`text-[10px] ${isHidden ? 'line-through' : ''}`}>{type.replace(/_/g, ' ')}</span>
                              </button>
                            )
                          })}
                        </div>
                      </div>
                    )}
                  </div>
                  
                  {/* Keyboard Shortcuts */}
                  <GraphKeyboardShortcuts
                    onFitGraph={handleFitGraph}
                    onToggle3D={handleToggle3D}
                    onRefresh={handleRefresh}
                    onExportPNG={handleExportPNG}
                    onClearSelection={handleClearSelection}
                  />
                  
                  {/* Action Buttons */}
                  <div className="absolute bottom-4 left-[220px] z-50 flex items-center gap-2">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="rounded-full shadow-lg bg-background/90 backdrop-blur-sm border-2 border-border hover:bg-accent hover:scale-110 hover:shadow-xl transition-all duration-200"
                      onClick={handleFitGraph}
                      title="Fit graph to view (F)"
                      disabled={!graphData}
                    >
                      <Maximize2 className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="rounded-full shadow-lg bg-background/90 backdrop-blur-sm border-2 border-border hover:bg-accent hover:scale-110 hover:shadow-xl transition-all duration-200"
                      onClick={handleRefresh}
                      title="Refresh graph (R)"
                      disabled={isLoading}
                    >
                      <RefreshCw className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="rounded-full shadow-lg bg-background/90 backdrop-blur-sm border-2 border-border hover:bg-accent hover:scale-110 hover:shadow-xl transition-all duration-200"
                      onClick={handleExportPNG}
                      title="Export as PNG (Ctrl+S)"
                      disabled={!graphData}
                    >
                      <Download className="h-4 w-4" />
                    </Button>
                  </div>
                  
                  {/* Keyboard Shortcuts Help */}
                  <KeyboardShortcutsHelp className="bottom-4 left-[350px]" />
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Hierarchical Tree Sidebar */}
        <div className="w-80 flex-shrink-0">
          <Card className="h-full overflow-hidden bg-gradient-to-br from-background to-background/95 backdrop-blur-sm border-2 shadow-xl">
            <CardContent className="flex h-full flex-col px-5 py-0">
              <div className="mb-4">
                <div className="flex items-center justify-between pb-3 border-b border-border/50">
                  <div className="flex items-center gap-2.5">
                    <div className="p-1.5 rounded-lg bg-primary/10 ring-1 ring-primary/20">
                      <Network className="h-4 w-4 text-primary" />
                    </div>
                    <h3 className="text-sm font-semibold tracking-tight">Explorer</h3>
                  </div>
                  {drillDownHistory.length > 0 && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleReset}
                      className="h-6 text-xs"
                    >
                      <X className="h-3 w-3 mr-1" />
                      Reset
                    </Button>
                  )}
                </div>
              </div>
              
              <div 
                className="flex-1 overflow-y-auto pr-2 -mr-2" 
                role="tree" 
                aria-label="System component hierarchy tree"
              >
                {graphData && graphData.nodes.length > 0 ? (
                  <HierarchicalTree
                    graphData={graphData}
                    drillDownHistory={drillDownHistory}
                    hierarchyData={hierarchyData}
                    expandedNodes={expandedNodes}
                    selectedNode={selectedNode}
                    isLoadingChildren={isLoadingChildren}
                    onNodeClick={handleNodeClick}
                    onReset={handleReset}
                    toggleNodeExpansion={toggleNodeExpansion}
                    getConnectionCount={getConnectionCount}
                    getConnectionType={getConnectionType}
                    getNodeIcon={getNodeIcon}
                    nodeColorByType={nodeColorByType}
                    linkColorByType={linkColorByType}
                  />
                ) : (
                  <div className="flex items-center justify-center h-32 text-sm text-muted-foreground">
                    {isLoading ? (
                      <div className="flex flex-col items-center gap-2">
                        <Loader2 className="h-5 w-5 animate-spin" />
                        <span className="text-xs">Loading...</span>
                      </div>
                    ) : (
                      <div className="text-center">
                        <Box className="h-8 w-8 mx-auto mb-2 text-muted-foreground/50" />
                        <p className="text-xs">No nodes available</p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </AppLayout>
  )
}
export default function TopologyPage() {
  return (
    <Suspense fallback={
      <AppLayout>
        <div className="flex items-center justify-center h-screen">
          <LoadingSpinner />
        </div>
      </AppLayout>
    }>
      <ExplorerContent />
    </Suspense>
  )
}