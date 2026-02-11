"use client"

import { useState, useEffect, useRef, Suspense } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { AppLayout } from "@/components/layout/app-layout"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { ForceGraph2DWrapper, type ForceGraph2DRef } from "@/components/graph/force-graph-2d-wrapper"
import { ForceGraph3DWrapper, type ForceGraph3DRef } from "@/components/graph/force-graph-3d-wrapper"
import { GraphControls } from "@/components/graph/graph-controls"
import { NodeDetailsPanel } from "@/components/graph/node-details-panel"
import { LinkDetailsPanel } from "@/components/graph/link-details-panel"
import { GraphKeyboardShortcuts } from "@/components/graph/graph-keyboard-shortcuts"
import { KeyboardShortcutsHelp } from "@/components/graph/keyboard-shortcuts-help"
import { LoadingSpinner } from "@/components/ui/loading-spinner"
import { NoConnectionInfo } from "@/components/layout/no-connection-info"
import { ToastNotification } from "@/components/ui/toast-notification"
import { useConnection } from "@/lib/stores/connection-store"
import { apiClient } from "@/lib/api/client"
import type { ForceGraphData, GraphNode, GraphLink } from "@/lib/types/api"
import type { GraphView } from "@/lib/types/graph-views"
import { GRAPH_VIEWS } from "@/lib/types/graph-views"
import { Loader2, Database, Settings, AlertCircle, Info, Maximize2, MousePointer, Eye, Search, X } from "lucide-react"
import { Input } from "@/components/ui/input"

function GraphPageContent() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const { status } = useConnection()
  const isConnected = status === 'connected'
  const isConnecting = status === 'connecting'

  // Graph data state
  const [graphData, setGraphData] = useState<ForceGraphData | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // UI state
  const [currentView, setCurrentView] = useState<GraphView>('complete')
  const [is3D, setIs3D] = useState(false)
  const [isSwitchingView, setIsSwitchingView] = useState(false)
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  const [selectedLink, setSelectedLink] = useState<GraphLink | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<GraphNode[]>([])
  const [nodeConnections, setNodeConnections] = useState<Array<{node: GraphNode, link: GraphLink, direction: 'outgoing' | 'incoming'}>>([])
  const [autoCenterOnSelect, setAutoCenterOnSelect] = useState(true)

  // Data options
  const [includeMetrics, setIncludeMetrics] = useState(false)
  
  // Limit options
  const [nodeLimit, setNodeLimit] = useState<number>(1000)
  const [edgeLimit, setEdgeLimit] = useState<number>(3000)
  const [useLimits, setUseLimits] = useState(true)

  // Visual options
  const [colorByType, setColorByType] = useState(true)

  // Visibility options
  const [visibleNodeTypes, setVisibleNodeTypes] = useState<Set<string>>(new Set())
  const [visibleLinkTypes, setVisibleLinkTypes] = useState<Set<string>>(new Set())

  // Zoom-based filtering for pub/sub relationships
  const [currentZoom, setCurrentZoom] = useState(1)
  const [pubSubZoomThreshold, setPubSubZoomThreshold] = useState(2)
  const [isPubSubEnabled, setIsPubSubEnabled] = useState(true)

  // Subgraph state
  const [subgraphNodeId, setSubgraphNodeId] = useState<string | null>(null)
  const [isLoadingSubgraph, setIsLoadingSubgraph] = useState(false)

  // Node connections loading state
  const [isLoadingNodeConnections, setIsLoadingNodeConnections] = useState(false)

  // Graph refs for fit-to-view functionality
  const graph2DRef = useRef<ForceGraph2DRef>(null)
  const graph3DRef = useRef<ForceGraph3DRef>(null)
  const searchTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  // Fetch graph data based on selected view
  const fetchGraphData = async () => {
    if (!isConnected) return

    setIsLoading(true)
    setError(null)

    try {
      const viewConfig = GRAPH_VIEWS[currentView]
      
      // Determine if this is a structural view (complete) or derived view (app/infra layers)
      const isStructuralView = currentView === 'complete'
      
      console.log('🎨 [Graph Page] Fetching graph data:', {
        view: currentView,
        viewConfig,
        isStructuralView,
        useLimits,
        nodeLimit,
        edgeLimit
      });

      const data = await apiClient.getLimitedGraphData({
        node_limit: useLimits ? nodeLimit : undefined,
        edge_limit: useLimits && edgeLimit > 0 ? edgeLimit : undefined,
        fetch_structural: isStructuralView,
        relationship_types: viewConfig.relationshipTypes,
        dependency_types: viewConfig.dependencyTypes,
        node_types: useLimits ? viewConfig.nodeTypes : undefined,
      })
      
      console.log('✨ [Graph Page] Graph data received:', {
        nodes: data.nodes.length,
        links: data.links.length,
        nodeTypes: [...new Set(data.nodes.map(n => n.type))],
        linkTypes: [...new Set(data.links.map(l => l.type))],
        metadata: data.metadata
      });
      
      setGraphData(data)
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || err.message || 'Failed to fetch graph data'
      console.error('❌ [Graph Page] Error fetching graph data:', err);
      setError(errorMsg)
    } finally {
      setIsLoading(false)
    }
  }

  // Load graph on mount and when options change (but not when viewing a subgraph)
  useEffect(() => {
    if (isConnected && !subgraphNodeId) {
      fetchGraphData()
    }
  }, [isConnected, currentView, useLimits])

  // Reset selections and search when view changes
  useEffect(() => {
    setSelectedNode(null)
    setSelectedLink(null)
    setNodeConnections([])
    setSearchQuery('')
    setSearchResults([])
    setAutoCenterOnSelect(true)
  }, [currentView])

  // Reset selections and search when 2D/3D mode changes
  useEffect(() => {
    setSelectedNode(null)
    setSelectedLink(null)
    setNodeConnections([])
    setSearchQuery('')
    setSearchResults([])
    setAutoCenterOnSelect(true)
  }, [is3D])

  // Handle node selection from URL parameter
  useEffect(() => {
    const nodeId = searchParams.get('node')
    const linkSource = searchParams.get('linkSource')
    const linkTarget = searchParams.get('linkTarget')
    
    if (nodeId && graphData) {
      const node = graphData.nodes.find(n => n.id === nodeId)
      if (node) {
        // Disable auto-centering when coming from analysis page
        setAutoCenterOnSelect(false)
        // Call handleNodeClick to fetch connections and properly select the node
        // Pass shouldCenter=false to show full graph view instead of centering
        // This allows users to see the complete context when coming from analysis page
        handleNodeClick(node, false)
      } else {
        // Node not in current graph (due to limits), fetch it with connections
        handleSelectSearchResult(nodeId)
      }
    } else if (linkSource && linkTarget && graphData) {
      const link = graphData.links.find(l => {
        const sourceId = typeof l.source === 'string' ? l.source : (l.source as any).id
        const targetId = typeof l.target === 'string' ? l.target : (l.target as any).id
        return sourceId === linkSource && targetId === linkTarget
      })
      if (link) {
        setSelectedLink(link)
        setSelectedNode(null)
        
        // Don't center/zoom on the link - show full graph view instead
      }
    }
  }, [searchParams, graphData, is3D])

  // Initialize visible types when graph data loads
  useEffect(() => {
    if (graphData) {
      const allNodeTypes = new Set(graphData.nodes.map(n => n.type))
      // Get all link types directly (API already normalizes DEPENDS_ON)
      const allLinkTypes = new Set(graphData.links.map(l => l.type))

      // Show all node and link types by default
      const visibleNodes = new Set(allNodeTypes)
      const visibleLinks = new Set(allLinkTypes)

      setVisibleNodeTypes(visibleNodes)
      setVisibleLinkTypes(visibleLinks)
    }
  }, [graphData, subgraphNodeId, searchParams, currentView])

  // Cleanup search timeout on unmount
  useEffect(() => {
    return () => {
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current)
      }
    }
  }, [])

  // Handle color scheme toggle
  const handleToggleColorByType = (value: boolean) => {
    setColorByType(value)
  }

  // Handle search with debouncing for better performance
  const handleSearch = async (query: string) => {
    setSearchQuery(query)

    if (!query.trim()) {
      setSearchResults([])
      return
    }

    // Debounce search for performance
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current)
    }

    searchTimeoutRef.current = setTimeout(async () => {
      try {
        // Search across all nodes in database
        const results = await apiClient.searchNodes(query, 20)
        // Convert to GraphNode format
        const graphNodes: GraphNode[] = results.map(r => ({
          id: r.id,
          label: r.label,
          type: r.type,
          degree: 0,
          properties: {}
        }))
        setSearchResults(graphNodes)
      } catch (err) {
        console.error('Search failed:', err)
        setSearchResults([])
      }
    }, 300)
  }

  // Handle selecting a search result
  const handleSelectSearchResult = async (nodeId: string) => {
    // Check if node is already in graph
    const existingNode = graphData?.nodes.find(n => n.id === nodeId)
    if (existingNode) {
      // Node already in graph, just select it
      handleNodeClick(existingNode)
      return
    }

    // Node not in graph, fetch it with its connections
    try {
      setIsLoadingNodeConnections(true)
      const isStructuralView = currentView === 'complete'
      const connectionData = await apiClient.getNodeConnections(nodeId, isStructuralView)

      if (graphData && connectionData.nodes.length > 0) {
        // Find the searched node
        const searchedNode = connectionData.nodes.find(n => n.id === nodeId)
        
        // Merge into existing graph
        const existingNodeIds = new Set(graphData.nodes.map(n => n.id))
        const newNodes = connectionData.nodes.filter(n => !existingNodeIds.has(n.id))
        
        const existingLinkIds = new Set(graphData.links.map(l => {
          const sourceId = typeof l.source === 'string' ? l.source : (l.source as any).id
          const targetId = typeof l.target === 'string' ? l.target : (l.target as any).id
          return `${sourceId}-${l.type}-${targetId}`
        }))
        
        const newLinks = connectionData.links.filter(l => {
          const linkId = `${l.source}-${l.type}-${l.target}`
          return !existingLinkIds.has(linkId)
        })

        if (newNodes.length > 0 || newLinks.length > 0) {
          console.log(`✨ Added searched node ${nodeId} with ${newNodes.length} new nodes and ${newLinks.length} new links`)
          
          // Position new nodes around center of current graph
          const centerX = graphData.nodes.reduce((sum, n: any) => sum + (n.x || 0), 0) / graphData.nodes.length || 0
          const centerY = graphData.nodes.reduce((sum, n: any) => sum + (n.y || 0), 0) / graphData.nodes.length || 0
          
          const newNodesWithPositions = newNodes.map((n, i) => {
            const angle = (Math.PI * 2 * i) / newNodes.length
            const radius = 150
            return {
              ...n,
              x: centerX + radius * Math.cos(angle),
              y: centerY + radius * Math.sin(angle),
              vx: 0,
              vy: 0
            }
          })

          const updatedGraphData = {
            nodes: [...graphData.nodes, ...newNodesWithPositions],
            links: [...graphData.links, ...newLinks],
            metadata: {
              ...graphData.metadata,
              search_added_nodes: [...(graphData.metadata.search_added_nodes || []), nodeId]
            }
          }
          
          setGraphData(updatedGraphData)

          // Select the searched node after state update and center on it
          setTimeout(() => {
            const nodeToSelect = updatedGraphData.nodes.find(n => n.id === nodeId)
            if (nodeToSelect) {
              handleNodeClick(nodeToSelect)
              // Additional explicit centering after graph update
              setTimeout(() => {
                if (is3D) {
                  graph3DRef.current?.centerOnNode(nodeId)
                } else {
                  graph2DRef.current?.centerOnNode(nodeId)
                }
              }, 500)
            }
          }, 150)
        } else {
          // Node and connections already exist, just select it
          const nodeToSelect = graphData.nodes.find(n => n.id === nodeId)
          if (nodeToSelect) {
            handleNodeClick(nodeToSelect)
          }
        }
      } else if (connectionData.nodes.length > 0) {
        // No existing graph, set this as the initial graph
        setGraphData(connectionData)
        setTimeout(() => {
          const nodeToSelect = connectionData.nodes.find(n => n.id === nodeId)
          if (nodeToSelect) {
            handleNodeClick(nodeToSelect)
            // Additional explicit centering after graph initialization
            setTimeout(() => {
              if (is3D) {
                graph3DRef.current?.centerOnNode(nodeId)
              } else {
                graph2DRef.current?.centerOnNode(nodeId)
              }
            }, 500)
          }
        }, 150)
      }
    } catch (err) {
      console.error('Failed to fetch searched node:', err)
    } finally {
      setIsLoadingNodeConnections(false)
    }
  }

  // Handle subgraph view
  const handleViewSubgraph = async (nodeId: string) => {
    if (!isConnected) return

    setIsLoadingSubgraph(true)
    setError(null)

    try {
      const data = await apiClient.getNodeSubgraph(nodeId, {
        depth: 2,
        include_metrics: includeMetrics,
        direction: 'both',
      })

      // Store the current subgraph node ID
      setSubgraphNodeId(nodeId)

      // Replace the graph data with subgraph data
      setGraphData(data)
    } catch (err: any) {
      const errorMsg = err.response?.data?.detail || err.message || 'Failed to fetch subgraph'
      setError(errorMsg)
    } finally {
      setIsLoadingSubgraph(false)
    }
  }

  // Handle returning to full graph
  const handleReturnToFullGraph = () => {
    setSubgraphNodeId(null)
    fetchGraphData()
  }

  // Handle zoom changes from graph
  const handleZoomChange = (zoom: number) => {
    setCurrentZoom(zoom)
  }

  // Handle fit graph to view
  const handleFitGraph = () => {
    if (is3D) {
      graph3DRef.current?.zoomToFit(1000, 150)
    } else {
      graph2DRef.current?.zoomToFit(1000, 150)
    }
  }

  // Handle export as PNG
  const handleExportPNG = () => {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5)
    const viewName = currentView.replace('_', '-')
    const mode = is3D ? '3d' : '2d'
    const filename = `graph-${viewName}-${mode}-${timestamp}.png`

    if (is3D) {
      graph3DRef.current?.exportPNG(filename)
    } else {
      graph2DRef.current?.exportPNG(filename)
    }
  }

  // Handle export as CSV
  const handleExportCSV = () => {
    if (!graphData) return

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5)
    const viewName = currentView.replace('_', '-')

    // Export nodes
    const nodeHeaders = ['id', 'label', 'type', 'degree', 'betweenness', 'pagerank', 'criticality_level', 'criticality_score']
    const nodeRows = graphData.nodes.map(node => [
      node.id,
      node.label,
      node.type,
      node.degree ?? '',
      node.betweenness ?? '',
      node.pagerank ?? '',
      node.criticality_level ?? '',
      node.criticality_score ?? ''
    ])

    const nodesCSV = [
      nodeHeaders.join(','),
      ...nodeRows.map(row => row.map(val => `"${String(val).replace(/"/g, '""')}"`).join(','))
    ].join('\n')

    // Export links
    const linkHeaders = ['source', 'target', 'type', 'weight']
    const linkRows = graphData.links.map(link => [
      typeof link.source === 'string' ? link.source : (link.source as any).id,
      typeof link.target === 'string' ? link.target : (link.target as any).id,
      link.type,
      link.weight ?? ''
    ])

    const linksCSV = [
      linkHeaders.join(','),
      ...linkRows.map(row => row.map(val => `"${String(val).replace(/"/g, '""')}"`).join(','))
    ].join('\n')

    // Download nodes CSV
    const nodesBlob = new Blob([nodesCSV], { type: 'text/csv;charset=utf-8;' })
    const nodesUrl = URL.createObjectURL(nodesBlob)
    const nodesLink = document.createElement('a')
    nodesLink.href = nodesUrl
    nodesLink.download = `graph-nodes-${viewName}-${timestamp}.csv`
    nodesLink.click()
    URL.revokeObjectURL(nodesUrl)

    // Download links CSV
    const linksBlob = new Blob([linksCSV], { type: 'text/csv;charset=utf-8;' })
    const linksUrl = URL.createObjectURL(linksBlob)
    const linksLink = document.createElement('a')
    linksLink.href = linksUrl
    linksLink.download = `graph-links-${viewName}-${timestamp}.csv`
    linksLink.click()
    URL.revokeObjectURL(linksUrl)
  }

  // Connecting State - show loading indicator
  if (isConnecting) {
    return (
      <AppLayout title="Graph Visualization" description="Interactive system graph explorer">
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" text="Connecting to database..." />
        </div>
      </AppLayout>
    )
  }

  // Disconnected State - show only no connection component
  if (!isConnected) {
    return (
      <AppLayout title="Graph Visualization" description="Interactive system graph explorer">
        <NoConnectionInfo description="Connect to your Neo4j database to visualize your system graph" />
      </AppLayout>
    )
  }

  // Loading state
  if (isLoading && !graphData) {
    return (
      <AppLayout title="Graph Visualization" description="Interactive system graph explorer">
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" text="Loading graph data..." />
        </div>
      </AppLayout>
    )
  }

  // Error state
  if (error && !graphData) {
    return (
      <AppLayout title="Graph Visualization" description="Interactive system graph explorer">
        <div className="space-y-6">

          <Card className="relative overflow-hidden border-0 shadow-lg hover:shadow-xl hover:shadow-red-500/20 transition-all duration-300">
            {/* Gradient border */}
            <div className="absolute inset-0 rounded-lg p-[2px] bg-gradient-to-r from-red-400 via-rose-500 to-pink-600">
              <div className="w-full h-full bg-background rounded-lg" />
            </div>

            {/* Background gradient overlay */}
            <div className="absolute inset-[2px] rounded-lg bg-[radial-gradient(circle_at_bottom_right,var(--tw-gradient-stops))] from-red-500/35 via-red-500/20 to-red-500/5" />

            <CardContent className="p-6 relative">
              <div className="flex items-start gap-4">
                <div className="rounded-xl bg-red-500/10 p-3 flex-shrink-0">
                  <AlertCircle className="h-6 w-6 text-red-500" />
                </div>
                <div className="flex-1 space-y-3">
                  <div>
                    <h3 className="text-lg font-bold">Failed to Load Graph</h3>
                    <p className="text-sm text-muted-foreground">Unable to retrieve graph data</p>
                  </div>
                  <div className="rounded-lg bg-red-500/10 border border-red-500/20 p-3.5">
                    <p className="text-sm text-red-700 dark:text-red-300 font-medium">{error}</p>
                  </div>
                  <Button onClick={fetchGraphData} className="w-full" variant="outline">
                    <Loader2 className="mr-2 h-4 w-4" />
                    Retry
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </AppLayout>
    )
  }

  // Empty database state
  if (graphData && graphData.nodes.length === 0) {
    return (
      <AppLayout title="Graph Visualization" description="Interactive system graph explorer">
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

  // Extract unique node and link types from graph data
  const nodeTypes = graphData ? Array.from(new Set(graphData.nodes.map(n => n.type))).sort() : []
  const linkTypes = graphData ? Array.from(new Set(graphData.links.map(l => l.type))).sort() : []

  // Determine if pub/sub relationships should be visible
  // When toggle is OFF: show all pub/sub links regardless of zoom
  // When toggle is ON: use zoom-based filtering in complete view
  const isPubSubVisible = !isPubSubEnabled || !(currentView === 'complete' && currentZoom < pubSubZoomThreshold)

  // Helper to get link ID
  const getLinkId = (link: GraphLink) => {
    const sourceId = typeof link.source === 'string' ? link.source : (link.source as any).id
    const targetId = typeof link.target === 'string' ? link.target : (link.target as any).id
    return `${sourceId}-${link.type}-${targetId}`
  }

  // Handle node click - clear link selection
  const handleNodeClick = async (node: GraphNode, shouldCenter: boolean = true) => {
    setSelectedNode(node)
    setSelectedLink(null)
    
    // Enable auto-centering if shouldCenter is true (normal clicks), disable if false (from URL)
    if (shouldCenter) {
      setAutoCenterOnSelect(true)
      // Immediately center on the clicked node
      setTimeout(() => {
        if (is3D) {
          graph3DRef.current?.centerOnNode(node.id)
        } else {
          graph2DRef.current?.centerOnNode(node.id)
        }
      }, 50)
    }
    
    // Compute connections for the selected node
    if (graphData) {
      const connections: Array<{node: GraphNode, link: GraphLink, direction: 'outgoing' | 'incoming'}> = []
      
      graphData.links.forEach(link => {
        const sourceId = typeof link.source === 'string' ? link.source : (link.source as any).id
        const targetId = typeof link.target === 'string' ? link.target : (link.target as any).id
        
        if (sourceId === node.id) {
          const targetNode = graphData.nodes.find(n => n.id === targetId)
          if (targetNode) {
            connections.push({ node: targetNode, link, direction: 'outgoing' })
          }
        }
        if (targetId === node.id) {
          const sourceNode = graphData.nodes.find(n => n.id === sourceId)
          if (sourceNode) {
            connections.push({ node: sourceNode, link, direction: 'incoming' })
          }
        }
      })
      
      setNodeConnections(connections)
      
      // Fetch and merge all connections for this node
      setIsLoadingNodeConnections(true)
      try {
        const isStructuralView = currentView === 'complete'
        const connectionData = await apiClient.getNodeConnections(node.id, isStructuralView)
        
        // Merge new nodes and links with existing graph data
        const existingNodeIds = new Set(graphData.nodes.map(n => n.id))
        const newNodes = connectionData.nodes.filter(n => !existingNodeIds.has(n.id))
        
        const existingLinkIds = new Set(graphData.links.map(l => {
          const sourceId = typeof l.source === 'string' ? l.source : (l.source as any).id
          const targetId = typeof l.target === 'string' ? l.target : (l.target as any).id
          return `${sourceId}-${l.type}-${targetId}`
        }))
        
        const newLinks = connectionData.links.filter(l => {
          const linkId = `${l.source}-${l.type}-${l.target}`
          return !existingLinkIds.has(linkId)
        })
        
        if (newNodes.length > 0 || newLinks.length > 0) {
          console.log(`✨ Merged ${newNodes.length} new nodes and ${newLinks.length} new links for node ${node.id}`)
          const updatedGraphData = {
            nodes: [...graphData.nodes, ...newNodes],
            links: [...graphData.links, ...newLinks],
            metadata: {
              ...graphData.metadata,
              expanded_nodes: [...(graphData.metadata.expanded_nodes || []), node.id]
            }
          }
          setGraphData(updatedGraphData)
          
          // Recompute connections with the updated graph data to include newly fetched connections
          const updatedConnections: Array<{node: GraphNode, link: GraphLink, direction: 'outgoing' | 'incoming'}> = []
          
          updatedGraphData.links.forEach(link => {
            const sourceId = typeof link.source === 'string' ? link.source : (link.source as any).id
            const targetId = typeof link.target === 'string' ? link.target : (link.target as any).id
            
            if (sourceId === node.id) {
              const targetNode = updatedGraphData.nodes.find(n => n.id === targetId)
              if (targetNode) {
                updatedConnections.push({ node: targetNode, link, direction: 'outgoing' })
              }
            }
            if (targetId === node.id) {
              const sourceNode = updatedGraphData.nodes.find(n => n.id === sourceId)
              if (sourceNode) {
                updatedConnections.push({ node: sourceNode, link, direction: 'incoming' })
              }
            }
          })
          
          setNodeConnections(updatedConnections)
          
          // Center camera on the node after loading connections (only if requested)
          // Use centerOnNode which will continuously track the node during layout changes
          if (shouldCenter) {
            // Immediate center attempt
            setTimeout(() => {
              if (is3D) {
                graph3DRef.current?.centerOnNode(node.id)
              } else {
                graph2DRef.current?.centerOnNode(node.id)
              }
            }, 100)
            
            // Additional center attempt after physics has had time to run
            setTimeout(() => {
              if (is3D) {
                graph3DRef.current?.centerOnNode(node.id)
              } else {
                graph2DRef.current?.centerOnNode(node.id)
              }
            }, 1000)
          }
        }
      } catch (err) {
        console.error('Failed to fetch node connections:', err)
        // Continue showing existing connections even if fetch fails
      } finally {
        setIsLoadingNodeConnections(false)
      }
    }
  }

  // Handle link click - clear node selection
  const handleLinkClick = (link: GraphLink) => {
    setSelectedLink(link)
    setSelectedNode(null)
  }

  // Handle background click - clear all selections
  const handleBackgroundClick = () => {
    setSelectedNode(null)
    setSelectedLink(null)
    setNodeConnections([])
  }

  // Handle 3D mode toggle with loading state
  const handleToggle3D = (value: boolean) => {
    setIsSwitchingView(true)
    setIs3D(value)
    // Clear switching state after a short delay to show the loading spinner
    setTimeout(() => setIsSwitchingView(false), 300)
  }

  // Handle node type visibility toggle
  const handleToggleNodeType = (type: string) => {
    setVisibleNodeTypes(prev => {
      const newSet = new Set(prev)
      if (newSet.has(type)) {
        newSet.delete(type)
      } else {
        newSet.add(type)
      }
      return newSet
    })
  }

  // Handle link type visibility toggle
  const handleToggleLinkType = (type: string) => {
    setVisibleLinkTypes(prev => {
      const newSet = new Set(prev)
      if (newSet.has(type)) {
        newSet.delete(type)
      } else {
        newSet.add(type)
      }
      return newSet
    })
  }

  return (
    <AppLayout title="Graph Visualization" description="Interactive system graph explorer">
      <div className="relative flex h-full gap-4">
        {/* Graph Canvas */}
        <div className="flex-1 flex flex-col relative">
          <Card className="flex-1 bg-gradient-to-br from-slate-50 via-white to-slate-50 dark:from-neutral-950 dark:via-neutral-900 dark:to-neutral-950 relative overflow-hidden p-0 shadow-lg">
            <CardContent className="h-full p-0 bg-gradient-to-br from-slate-50 via-white to-slate-50 dark:from-neutral-950 dark:via-neutral-900 dark:to-neutral-950 relative">

              {/* Loading node connections indicator - top right overlay */}
              {isLoadingNodeConnections && (
                <div className="absolute bottom-4 right-4 z-20 bg-background/95 backdrop-blur-md border border-border rounded-lg px-3 py-2 text-sm font-medium shadow-lg transition-all duration-300 animate-in fade-in duration-200">
                  <div className="flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin text-primary" />
                    <span className="text-foreground">Loading connections...</span>
                  </div>
                </div>
              )}

              {/* Clear Selection button - top left overlay */}
              {(selectedNode || selectedLink) && (
                <div className="absolute top-4 left-4 z-10 animate-in slide-in-from-left-3 fade-in duration-300">
                  <Button
                    onClick={handleBackgroundClick}
                    variant="outline"
                    size="sm"
                    className="border-amber-300 dark:border-amber-700 bg-amber-50/95 dark:bg-amber-900/30 hover:bg-amber-100 dark:hover:bg-amber-900/40 hover:border-amber-400 dark:hover:border-amber-600 shadow-lg hover:shadow-xl backdrop-blur-sm transition-all"
                  >
                    <X className="mr-2 h-4 w-4" />
                    Clear Selection
                  </Button>
                </div>
              )}

              {/* Subgraph mode indicator - top right overlay */}
              {subgraphNodeId && (() => {
                // Find the center node from graph data to show its label
                const centerNode = graphData?.nodes.find(n => n.id === subgraphNodeId)
                const centerLabel = centerNode?.label || subgraphNodeId

                return (
                  <div className="absolute top-4 right-4 z-10 flex flex-col gap-2 items-end animate-in slide-in-from-right-3 fade-in duration-300">
                    <Card className="border-primary/50 bg-card/95 backdrop-blur-sm shadow-lg transition-all hover:shadow-xl">
                      <CardContent className="p-3">
                        <div className="flex items-center gap-3">
                          <div className="flex flex-col gap-1">
                            <Badge variant="outline" className="border-primary text-primary w-fit">
                              Subgraph View
                            </Badge>
                            <span className="text-xs text-muted-foreground">
                              Neighborhood of: <span className="font-medium">{centerLabel}</span>
                            </span>
                          </div>
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={handleReturnToFullGraph}
                            disabled={isLoading}
                          >
                            Exit
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                )
              })()}

              {/* Loading overlay with smooth transitions */}
              {(isLoading && graphData) || isSwitchingView || isLoadingSubgraph ? (
                <div className="absolute inset-0 bg-background/90 backdrop-blur-md z-20 flex items-center justify-center animate-in fade-in duration-300">
                  <LoadingSpinner
                    size="lg"
                    text={
                      isLoadingSubgraph
                        ? "Loading subgraph..."
                        : isSwitchingView
                        ? "Switching view..."
                        : "Loading graph data..."
                    }
                  />
                </div>
              ) : null}

              {graphData && (
                <>
                  {is3D ? (
                    <ForceGraph3DWrapper
                      ref={graph3DRef}
                      nodes={graphData.nodes}
                      links={graphData.links}
                      onNodeClick={handleNodeClick}
                      onLinkClick={handleLinkClick}
                      onBackgroundClick={handleBackgroundClick}
                      selectedNodeId={selectedNode?.id}
                      selectedLinkId={selectedLink ? getLinkId(selectedLink) : null}
                      autoCenterOnSelect={autoCenterOnSelect}
                      onZoomChange={handleZoomChange}
                      isPubSubVisible={isPubSubVisible}
                      visibleNodeTypes={visibleNodeTypes}
                      visibleLinkTypes={visibleLinkTypes}
                    />
                  ) : (
                    <ForceGraph2DWrapper
                      ref={graph2DRef}
                      nodes={graphData.nodes}
                      links={graphData.links}
                      onNodeClick={handleNodeClick}
                      onLinkClick={handleLinkClick}
                      onBackgroundClick={handleBackgroundClick}
                      selectedNodeId={selectedNode?.id}
                      selectedLinkId={selectedLink ? getLinkId(selectedLink) : null}
                      autoCenterOnSelect={autoCenterOnSelect}
                      onZoomChange={handleZoomChange}
                      isPubSubVisible={isPubSubVisible}
                      visibleNodeTypes={visibleNodeTypes}
                      visibleLinkTypes={visibleLinkTypes}
                    />
                  )}
                </>
              )}

              {/* Keyboard Shortcuts Help */}
              <KeyboardShortcutsHelp />
            </CardContent>
          </Card>
        </div>

        {/* Right Controls Panel */}
        <div className="w-72 flex-shrink-0">
          <GraphControls
            currentView={currentView}
            onViewChange={setCurrentView}
            is3D={is3D}
            onToggle3D={handleToggle3D}
            colorByType={colorByType}
            onToggleColorByType={handleToggleColorByType}
            onRefresh={fetchGraphData}
            onFitGraph={handleFitGraph}
            onExportPNG={handleExportPNG}
            onExportCSV={handleExportCSV}
            nodeCount={graphData?.nodes.length || 0}
            linkCount={graphData?.links.length || 0}
            isLoading={isLoading}
            nodeTypes={nodeTypes}
            linkTypes={linkTypes}
            visibleNodeTypes={visibleNodeTypes}
            visibleLinkTypes={visibleLinkTypes}
            onToggleNodeType={handleToggleNodeType}
            onToggleLinkType={handleToggleLinkType}
            currentZoom={currentZoom}
            pubSubZoomThreshold={pubSubZoomThreshold}
            onPubSubZoomThresholdChange={setPubSubZoomThreshold}
            isPubSubEnabled={isPubSubEnabled}
            onTogglePubSubEnabled={setIsPubSubEnabled}
            graphData={graphData}
            searchQuery={searchQuery}
            onSearchChange={handleSearch}
            searchResults={searchResults.map(node => ({
              id: node.id,
              label: node.label,
              type: node.type
            }))}
            onSearchResultClick={handleSelectSearchResult}
            selectedNode={selectedNode}
            selectedLink={selectedLink}
            onCloseNodeDetails={() => {
              setSelectedNode(null)
              setNodeConnections([])
            }}
            onCloseLinkDetails={() => setSelectedLink(null)}
            onViewSubgraph={handleViewSubgraph}
            nodeConnections={nodeConnections}
            nodeLimit={nodeLimit}
            edgeLimit={edgeLimit}
            useLimits={useLimits}
            onNodeLimitChange={setNodeLimit}
            onEdgeLimitChange={setEdgeLimit}
            onUseLimitsChange={setUseLimits}
            onApplyLimits={fetchGraphData}
          />
        </div>

        {/* Keyboard Shortcuts */}
        <GraphKeyboardShortcuts
          onFitGraph={handleFitGraph}
          onToggle3D={() => handleToggle3D(!is3D)}
          onRefresh={fetchGraphData}
          onExportPNG={handleExportPNG}
          onClearSelection={handleBackgroundClick}
        />

        {/* Graph Error Toast Notification */}
        {error && graphData && (
          <ToastNotification
            type="warning"
            message={error}
            onClose={() => setError(null)}
          />
        )}
      </div>
    </AppLayout>
  )
}

export default function GraphPage() {
  return (
    <Suspense fallback={
      <AppLayout title="Graph Visualization" description="Interactive system graph explorer">
        <div className="flex h-full items-center justify-center">
          <LoadingSpinner size="lg" text="Loading graph..." />
        </div>
      </AppLayout>
    }>
      <GraphPageContent />
    </Suspense>
  )
}
