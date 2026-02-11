"use client"

import { useEffect, useRef, useState, forwardRef, useImperativeHandle, useMemo, useCallback, memo } from "react"
import dynamic from "next/dynamic"
import { useTheme } from "next-themes"
import type { GraphNode, GraphLink } from "@/lib/types/api"

const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center">
      <div className="text-sm text-muted-foreground">Loading graph...</div>
    </div>
  ),
})

interface ForceGraph2DWrapperProps {
  nodes: GraphNode[]
  links: GraphLink[]
  onNodeClick?: (node: GraphNode) => void
  onLinkClick?: (link: GraphLink) => void
  onBackgroundClick?: () => void
  selectedNodeId?: string | null
  selectedLinkId?: string | null
  autoCenterOnSelect?: boolean
  colorByType?: boolean
  colorByCriticality?: boolean
  onZoomChange?: (zoom: number) => void
  isPubSubVisible?: boolean
  visibleNodeTypes?: Set<string>
  visibleLinkTypes?: Set<string>
}

export interface ForceGraph2DRef {
  zoomToFit: (duration?: number, padding?: number) => void
  exportPNG: (filename?: string) => void
  centerOnNode: (nodeId: string) => void
}

export const ForceGraph2DWrapper = forwardRef<ForceGraph2DRef, ForceGraph2DWrapperProps>(({
  nodes,
  links,
  onNodeClick,
  onLinkClick,
  onBackgroundClick,
  selectedNodeId,
  selectedLinkId,
  autoCenterOnSelect = true,
  colorByType = true,
  colorByCriticality = false,
  onZoomChange,
  isPubSubVisible = true,
  visibleNodeTypes,
  visibleLinkTypes,

}, ref) => {
  const graphRef = useRef<any>(null)
  const { theme, systemTheme } = useTheme()
  const [dimensions, setDimensions] = useState({ width: 800, height: 600 })
  const containerRef = useRef<HTMLDivElement>(null)
  const [currentZoom, setCurrentZoom] = useState(1)
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null)
  const [hoveredLink, setHoveredLink] = useState<GraphLink | null>(null)
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string } | null>(null)
  const [isRendering, setIsRendering] = useState(true)
  const zoomTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const pendingCenterNodeId = useRef<string | null>(null)
  const centeringFrameCount = useRef(0)

  // Get actual theme
  const currentTheme = theme === 'system' ? systemTheme : theme

  // OPTIMIZED: Single configuration on first tick
  const forcesConfiguredRef = useRef(false)

  // Update dimensions on mount and resize
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.offsetWidth,
          height: containerRef.current.offsetHeight,
        })
      }
    }

    updateDimensions()
    window.addEventListener("resize", updateDimensions)
    return () => window.removeEventListener("resize", updateDimensions)
  }, [])

  // OPTIMIZED: Removed redundant useEffect - forces configured in onEngineTick

  // Distinct colors for each node type - matching app color scheme with vibrant saturated colors
  const nodeColorByType: Record<string, string> = useMemo(() => ({
    Application: currentTheme === "dark" ? "#3b82f6" : "#2563eb",
    Node: currentTheme === "dark" ? "#ef4444" : "#dc2626",
    Broker: currentTheme === "dark" ? "#a1a1aa" : "#71717a",
    Topic: currentTheme === "dark" ? "#facc15" : "#eab308",
    Library: currentTheme === "dark" ? "#06b6d4" : "#0891b2",
    Unknown: currentTheme === "dark" ? "#a1a1aa" : "#71717a",
  }), [currentTheme])

  // Distinct colors for each link/relationship type
  const linkColorByType: Record<string, string> = useMemo(() => ({
    RUNS_ON: currentTheme === "dark" ? "#a855f7" : "#9333ea",
    PUBLISHES_TO: currentTheme === "dark" ? "#22c55e" : "#16a34a",
    SUBSCRIBES_TO: currentTheme === "dark" ? "#f97316" : "#ea580c",
    DEPENDS_ON: currentTheme === "dark" ? "#ef4444" : "#dc2626",
    CONNECTS_TO: currentTheme === "dark" ? "#22c55e" : "#16a34a",
    ROUTES: currentTheme === "dark" ? "#a1a1aa" : "#71717a",
    USES: currentTheme === "dark" ? "#06b6d4" : "#0891b2",
    app_to_app: currentTheme === "dark" ? "#ef4444" : "#dc2626",
    node_to_node: currentTheme === "dark" ? "#ef4444" : "#dc2626",
    default: currentTheme === "dark" ? "#94a3b8" : "#64748b",
  }), [currentTheme])

  // Criticality colors - clear visual hierarchy
  const nodeColorByCriticality = useCallback((level?: string) => {
    switch (level) {
      case "critical":
        return currentTheme === "dark" ? "#ef4444" : "#dc2626"
      case "high":
        return currentTheme === "dark" ? "#f59e0b" : "#d97706"
      case "medium":
        return currentTheme === "dark" ? "#60a5fa" : "#3b82f6"
      case "low":
        return currentTheme === "dark" ? "#34d399" : "#10b981"
      default:
        return currentTheme === "dark" ? "#6b7280" : "#9ca3af"
    }
  }, [currentTheme])

  // Track connected nodes for highlighting
  const [connectedNodeIds, setConnectedNodeIds] = useState<Set<string>>(new Set())

  // Compute connected nodes when selection changes
  useEffect(() => {
    if (!selectedNodeId) {
      setConnectedNodeIds(new Set())
      return
    }

    const connected = new Set<string>()
    links.forEach(link => {
      const sourceId = typeof link.source === 'string' ? link.source : (link.source as any).id
      const targetId = typeof link.target === 'string' ? link.target : (link.target as any).id
      
      if (sourceId === selectedNodeId) {
        connected.add(targetId)
      }
      if (targetId === selectedNodeId) {
        connected.add(sourceId)
      }
    })
    setConnectedNodeIds(connected)
  }, [selectedNodeId, links])

  // Memoize node color function to avoid recalculations
  const getNodeColor = useMemo(() => {
    const colorCache = new Map<string, string>()
    return (node: any) => {
      const n = node as GraphNode
      const isHovered = hoveredNode?.id === n.id
      const cacheKey = `${n.id}-${selectedNodeId}-${connectedNodeIds.has(n.id)}-${isHovered}`
      
      if (colorCache.has(cacheKey)) {
        return colorCache.get(cacheKey)!
      }
      
      const isNodeHighlighted = !selectedNodeId || n.id === selectedNodeId || connectedNodeIds.has(n.id)
      
      let baseColor: string
      if (colorByCriticality && n.criticality_level) {
        baseColor = nodeColorByCriticality(n.criticality_level)
      } else {
        baseColor = nodeColorByType[n.type] || nodeColorByType.Unknown
      }
      
      let result: string
      if (isHovered) {
        // Brighten hovered node
        const r = Math.min(255, parseInt(baseColor.slice(1, 3), 16) + 40)
        const g = Math.min(255, parseInt(baseColor.slice(3, 5), 16) + 40)
        const b = Math.min(255, parseInt(baseColor.slice(5, 7), 16) + 40)
        result = `rgb(${r}, ${g}, ${b})`
      } else if (selectedNodeId && !isNodeHighlighted) {
        const r = parseInt(baseColor.slice(1, 3), 16)
        const g = parseInt(baseColor.slice(3, 5), 16)
        const b = parseInt(baseColor.slice(5, 7), 16)
        result = `rgba(${r}, ${g}, ${b}, 0.2)`
      } else {
        result = baseColor
      }
      
      colorCache.set(cacheKey, result)
      return result
    }
  }, [selectedNodeId, connectedNodeIds, hoveredNode, colorByCriticality, nodeColorByType, nodeColorByCriticality])

  // Memoize node size function
  const getNodeSize = useMemo(() => {
    const sizeCache = new Map<string, number>()
    return (node: any) => {
      const n = node as GraphNode
      const isHovered = hoveredNode?.id === n.id
      const cacheKey = `${n.id}-${selectedNodeId}-${connectedNodeIds.has(n.id)}-${isHovered}`
      
      if (sizeCache.has(cacheKey)) {
        return sizeCache.get(cacheKey)!
      }
      
      let baseSize: number
      if (colorByCriticality && n.criticality_score) {
        baseSize = 3 + (n.criticality_score * 7)
      } else {
        switch (n.type) {
          case 'Node':
            baseSize = 8
            break
          case 'Application':
            baseSize = 6
            break
          case 'Topic':
          case 'Broker':
            baseSize = 4
            break
          default:
            baseSize = 5
        }
      }
      
      // Increase size on hover
      const result = isHovered ? baseSize * 1.3 : baseSize
      
      sizeCache.set(cacheKey, result)
      return result
    }
  }, [colorByCriticality, hoveredNode])

  // Update dimensions on mount and resize

  // Memoize node lookup map for faster link rendering
  const nodeMap = useMemo(() => {
    const map = new Map<string, GraphNode>()
    nodes.forEach(node => map.set(node.id, node))
    return map
  }, [nodes])

  // Memoize link color function with caching
  const getLinkColor = useMemo(() => {
    const colorCache = new Map<string, string>()
    return (link: any) => {
      const l = link as GraphLink
      const linkId = getLinkId(l)

      // Check cache first
      const cacheKey = `${linkId}-${selectedLinkId}-${selectedNodeId}`
      if (colorCache.has(cacheKey)) {
        return colorCache.get(cacheKey)!
      }

      const isSelected = selectedLinkId === linkId
      const color = linkColorByType[l.type] || linkColorByType.default

      // Get source and target nodes using map for O(1) lookup
      const sourceId = typeof l.source === 'string' ? l.source : (l.source as any).id
      const targetId = typeof l.target === 'string' ? l.target : (l.target as any).id
      const sourceNode = nodeMap.get(sourceId)
      const targetNode = nodeMap.get(targetId)

      // Check if source or target node type is visible
      const isSourceVisible = !visibleNodeTypes || !sourceNode || visibleNodeTypes.has(sourceNode.type)
      const isTargetVisible = !visibleNodeTypes || !targetNode || visibleNodeTypes.has(targetNode.type)

      // Check if link type is visible
      const isLinkTypeVisible = !visibleLinkTypes || visibleLinkTypes.has(l.type)

      // Check if this is a pub/sub link that should be hidden
      const isPubSubLink = l.type === 'PUBLISHES_TO' || l.type === 'SUBSCRIBES_TO'

      // Check if link connects to selected node (for highlighting)
      const isConnectedToSelected = selectedNodeId && (sourceId === selectedNodeId || targetId === selectedNodeId)

      // Set opacity based on visibility and selection
      let opacity = 0.4
      if (isSelected) {
        opacity = 0.9
      } else if (!isLinkTypeVisible || !isSourceVisible || !isTargetVisible) {
        opacity = 0 // Make invisible when link type is hidden or either node is hidden
      } else if (isPubSubLink && !isPubSubVisible && !isConnectedToSelected) {
        opacity = 0 // Make invisible when zoomed out (but always show if connected to selected node)
      } else if (selectedNodeId && !isConnectedToSelected) {
        opacity = 0.1 // Dim links not connected to selected node
      } else if (isConnectedToSelected) {
        opacity = 0.8 // Highlight links connected to selected node
      }

      const result = hexToRgba(color, opacity)
      colorCache.set(cacheKey, result)
      return result
    }
  }, [linkColorByType, selectedLinkId, selectedNodeId, visibleNodeTypes, visibleLinkTypes, isPubSubVisible, nodeMap])

  // Helper function to convert hex to rgba
  const hexToRgba = (hex: string, alpha: number) => {
    const r = parseInt(hex.slice(1, 3), 16)
    const g = parseInt(hex.slice(3, 5), 16)
    const b = parseInt(hex.slice(5, 7), 16)
    return `rgba(${r}, ${g}, ${b}, ${alpha})`
  }

  const handleNodeHover = useCallback((node: any) => {
    if (node) {
      const n = node as GraphNode
      setHoveredNode(n)
      const canvas = containerRef.current?.querySelector('canvas')
      if (canvas) {
        canvas.style.cursor = 'pointer'
        canvas.style.transition = 'opacity 0.2s ease-in-out'
      }
    } else {
      setHoveredNode(null)
      const canvas = containerRef.current?.querySelector('canvas')
      if (canvas) {
        canvas.style.cursor = 'default'
      }
    }
  }, [])

  const handleNodeClick = (node: any) => {
    if (onNodeClick) {
      const n = node as GraphNode
      // Check if node type is visible
      const isNodeTypeVisible = !visibleNodeTypes || visibleNodeTypes.has(n.type)
      if (!isNodeTypeVisible) {
        return // Don't handle click for invisible nodes
      }
      onNodeClick(n)
    }
  }

  const handleLinkClick = (link: any) => {
    if (onLinkClick) {
      const l = link as GraphLink
      // Get source and target nodes
      const sourceId = typeof l.source === 'string' ? l.source : (l.source as any).id
      const targetId = typeof l.target === 'string' ? l.target : (l.target as any).id
      const sourceNode = nodes.find(n => n.id === sourceId)
      const targetNode = nodes.find(n => n.id === targetId)

      // Check if source or target node type is visible
      const isSourceVisible = !visibleNodeTypes || !sourceNode || visibleNodeTypes.has(sourceNode.type)
      const isTargetVisible = !visibleNodeTypes || !targetNode || visibleNodeTypes.has(targetNode.type)

      // Check if link type is visible
      const isLinkTypeVisible = !visibleLinkTypes || visibleLinkTypes.has(l.type)

      // Check if this is a pub/sub link that should be hidden
      const isPubSubLink = l.type === 'PUBLISHES_TO' || l.type === 'SUBSCRIBES_TO'

      if (!isLinkTypeVisible || !isSourceVisible || !isTargetVisible || (isPubSubLink && !isPubSubVisible)) {
        return // Don't handle click for invisible links
      }
      onLinkClick(l)
    }
  }

  const handleBackgroundClick = (event: MouseEvent) => {
    if (onBackgroundClick) {
      onBackgroundClick()
    }
  }

  const handleNodeDragEnd = (node: any) => {
    const n = node as GraphNode & { fx?: number; fy?: number; x: number; y: number }
    // Fix node position after drag
    n.fx = n.x
    n.fy = n.y
  }

  // Expose methods to parent component
  useImperativeHandle(ref, () => ({
    zoomToFit: (duration = 1000, padding = 50) => {
      if (graphRef.current) {
        graphRef.current.zoomToFit(duration, padding)
      }
    },
    centerOnNode: (nodeId: string) => {
      // Set pending center node to continuously track it during layout changes
      pendingCenterNodeId.current = nodeId
      centeringFrameCount.current = 0
      
      const node = nodes.find(n => n.id === nodeId) as any
      if (node && graphRef.current) {
        // Immediately center if node has position
        if (node.x !== undefined && node.y !== undefined) {
          graphRef.current?.centerAt(node.x, node.y, 1000)
          graphRef.current?.zoom(3, 1000)
        }
      }
    },
    exportPNG: (filename = 'graph.png') => {
      if (containerRef.current) {
        const canvas = containerRef.current.querySelector('canvas') as HTMLCanvasElement
        if (canvas) {
          try {
            // Use toDataURL for reliable export across all browsers
            const dataURL = canvas.toDataURL('image/png')
            const link = document.createElement('a')
            link.href = dataURL
            link.download = filename
            link.click()
          } catch (error) {
            console.error('Failed to export 2D graph as PNG:', error)
          }
        }
      }
    }
  }))

  // Center camera on selected node - set pending node for continuous centering
  useEffect(() => {
    if (autoCenterOnSelect && selectedNodeId) {
      pendingCenterNodeId.current = selectedNodeId
      centeringFrameCount.current = 0
    } else {
      // Clear pending center when no selection, regardless of autoCenterOnSelect
      pendingCenterNodeId.current = null
      centeringFrameCount.current = 0
    }
  }, [autoCenterOnSelect, selectedNodeId])
  
  // Continuously center on pending node during physics simulation
  useEffect(() => {
    if (!pendingCenterNodeId.current || !graphRef.current) return
    
    const intervalId = setInterval(() => {
      const nodeId = pendingCenterNodeId.current
      if (!nodeId) return
      
      const node = nodes.find(n => n.id === nodeId) as any
      if (node && node.x !== undefined && node.y !== undefined) {
        // More aggressive centering with faster transitions initially
        const duration = centeringFrameCount.current < 5 ? 300 : 500
        graphRef.current?.centerAt(node.x, node.y, duration)
        
        // Apply zoom on first several frames to ensure it takes effect
        if (centeringFrameCount.current < 5) {
          graphRef.current?.zoom(3, duration)
        }
        
        centeringFrameCount.current++
        
        // Stop centering after 30 frames (about 6 seconds) to allow user interaction
        if (centeringFrameCount.current > 30) {
          pendingCenterNodeId.current = null
        }
      }
    }, 200)
    
    return () => clearInterval(intervalId)
  }, [nodes, autoCenterOnSelect])

  const getLinkId = useCallback((link: GraphLink) => {
    const sourceId = typeof link.source === 'string' ? link.source : (link.source as any).id
    const targetId = typeof link.target === 'string' ? link.target : (link.target as any).id
    return `${sourceId}-${link.type}-${targetId}`
  }, [])

  // Handle zoom changes with debouncing
  const handleZoom = useCallback((zoom: any) => {
    const newZoom = zoom?.k ?? 1

    // OPTIMIZED: Defer state update to avoid setState during render
    requestAnimationFrame(() => {
      setCurrentZoom(newZoom)
    })

    // Debounce zoom change callback for performance
    if (zoomTimeoutRef.current) {
      clearTimeout(zoomTimeoutRef.current)
    }
    zoomTimeoutRef.current = setTimeout(() => {
      if (onZoomChange) {
        onZoomChange(newZoom)
      }
    }, 100)
  }, [onZoomChange])

  // Background colors from shadcn/ui - use CSS variables for consistency
  const cardBgColor = currentTheme === "dark" ? "hsl(0 0% 3.9%)" : "hsl(0 0% 100%)"

  // Memoize graph data to prevent unnecessary re-renders
  const graphData = useMemo(() => ({ nodes, links }), [nodes, links])

  // Create link width cache with memoization
  const linkWidthCache = useRef(new Map<string, number>())
  const getLinkWidth = useCallback((link: any) => {
    const originalLink = link.__data ?? link
    const linkId = getLinkId(originalLink)
    const cacheKey = `${linkId}-${selectedLinkId}-${isPubSubVisible}`
    
    if (linkWidthCache.current.has(cacheKey)) {
      return linkWidthCache.current.get(cacheKey)!
    }

    const isSelected = selectedLinkId === linkId
    const sourceId = typeof originalLink.source === 'string' ? originalLink.source : (originalLink.source as any).id
    const targetId = typeof originalLink.target === 'string' ? originalLink.target : (originalLink.target as any).id
    const sourceNode = nodes.find(n => n.id === sourceId)
    const targetNode = nodes.find(n => n.id === targetId)

    const isSourceVisible = !visibleNodeTypes || !sourceNode || visibleNodeTypes.has(sourceNode.type)
    const isTargetVisible = !visibleNodeTypes || !targetNode || visibleNodeTypes.has(targetNode.type)
    const isLinkTypeVisible = !visibleLinkTypes || visibleLinkTypes.has(originalLink.type)
    
    if (!isLinkTypeVisible || !isSourceVisible || !isTargetVisible) {
      linkWidthCache.current.set(cacheKey, 0)
      return 0
    }

    const isPubSubLink = originalLink.type === 'PUBLISHES_TO' || originalLink.type === 'SUBSCRIBES_TO'
    const isConnectedToSelected = selectedNodeId && (sourceId === selectedNodeId || targetId === selectedNodeId)
    if (isPubSubLink && !isPubSubVisible && !isConnectedToSelected) {
      linkWidthCache.current.set(cacheKey, 0)
      return 0
    }

    const weight = originalLink.weight ?? originalLink.properties?.weight ?? originalLink.properties?.count
    const isDependsOn = originalLink.type === 'DEPENDS_ON'
    let baseWidth = 1.5
    if (weight && typeof weight === 'number' && weight > 0) {
      if (isDependsOn) {
        // More aggressive scaling for smaller weights, especially below 20
        if (weight < 20) {
          baseWidth = Math.max(1.5, Math.min(4, 1.5 + weight * 0.4))
        } else {
          baseWidth = Math.max(1.5, Math.min(4, 1.5 + Math.sqrt(weight) * 1.2))
        }
      } else {
        // More aggressive scaling for non-DEPENDS_ON links with small weights
        if (weight < 20) {
          baseWidth = Math.max(1.2, Math.min(3.5, 1.2 + weight * 0.35))
        } else {
          baseWidth = Math.max(1.2, Math.min(3.5, 1.2 + Math.sqrt(weight) * 0.8))
        }
      }
    }
    // Add hover effect
    const isHovered = hoveredLink && getLinkId(hoveredLink) === linkId
    const result = isSelected ? baseWidth * 2.5 : (isHovered ? baseWidth * 1.8 : baseWidth)
    linkWidthCache.current.set(cacheKey, result)
    return result
  }, [selectedLinkId, hoveredLink, isPubSubVisible, visibleNodeTypes, visibleLinkTypes])

  // Dynamic arrow length based on link width
  const getLinkArrowLength = useCallback((link: any) => {
    const originalLink = link.__data ?? link
    const width = getLinkWidth(originalLink)
    // Make arrows proportional to link width with a reasonable range
    return width > 0 ? Math.max(4, Math.min(8, 4 + width * 0.8)) : 0
  }, [getLinkWidth])

  // Clear cache when dependencies change
  useEffect(() => {
    linkWidthCache.current.clear()
  }, [selectedLinkId, isPubSubVisible, visibleNodeTypes, visibleLinkTypes])

  // Handle initial rendering state with smooth transition
  useEffect(() => {
    if (nodes.length > 0) {
      setIsRendering(true)
      const timer = setTimeout(() => setIsRendering(false), 500)
      return () => clearTimeout(timer)
    }
  }, [nodes.length])

  // Cleanup zoom timeout on unmount
  useEffect(() => {
    return () => {
      if (zoomTimeoutRef.current) {
        clearTimeout(zoomTimeoutRef.current)
      }
    }
  }, [])

  // Handle initial rendering state
  useEffect(() => {
    if (nodes.length > 0) {
      setIsRendering(true)
      const timer = setTimeout(() => setIsRendering(false), 500)
      return () => clearTimeout(timer)
    }
  }, [nodes.length])

  // Cleanup zoom timeout on unmount
  useEffect(() => {
    return () => {
      if (zoomTimeoutRef.current) {
        clearTimeout(zoomTimeoutRef.current)
      }
    }
  }, [])

  return (
    <div ref={containerRef} className="h-full w-full bg-neutral-950 dark:bg-neutral-950 bg-white relative">
      {/* Skeleton Loading Overlay */}
      {isRendering && (
        <div className="absolute inset-0 bg-background/95 backdrop-blur-sm z-30 flex items-center justify-center animate-in fade-in duration-200">
          <div className="flex flex-col items-center gap-4">
            <div className="flex gap-2">
              <div className="w-3 h-3 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
              <div className="w-3 h-3 bg-primary rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
              <div className="w-3 h-3 bg-primary rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
            </div>
            <p className="text-sm text-muted-foreground">Rendering graph...</p>
          </div>
        </div>
      )}
      
      <ForceGraph2D
        ref={graphRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={graphData}
        nodeId="id"
        nodeLabel=""
        nodeColor={getNodeColor}
        nodeVal={getNodeSize}
        nodeRelSize={1}
        linkColor={getLinkColor}
        linkWidth={getLinkWidth}
        linkDirectionalArrowLength={getLinkArrowLength}
        linkDirectionalArrowRelPos={0.92}
        linkDirectionalArrowColor={getLinkColor}
        linkDirectionalParticles={0}
        linkCurvature={0}
        onNodeClick={handleNodeClick}
        onNodeHover={handleNodeHover}
        onLinkClick={handleLinkClick}
        onBackgroundClick={handleBackgroundClick}
        onZoom={handleZoom}
        minZoom={0.2}
        maxZoom={15}
        nodePointerAreaPaint={(node: any, color: string, ctx: CanvasRenderingContext2D) => {
          const n = node as GraphNode & { x: number; y: number }
          const size = getNodeSize(n)
          
          // Check if node type is visible
          const isNodeTypeVisible = !visibleNodeTypes || visibleNodeTypes.has(n.type)
          
          // Don't paint pointer area for invisible nodes - this prevents them from being clickable
          if (!isNodeTypeVisible) return
          
          // Paint the clickable area
          ctx.fillStyle = color
          ctx.beginPath()
          ctx.arc(n.x, n.y, size, 0, 2 * Math.PI, false)
          ctx.fill()
        }}
        nodeCanvasObject={(node: any, ctx: any, globalScale: number) => {
          const n = node as GraphNode & { x: number; y: number }
          const label = n.label || ""
          const fontSize = Math.max(10, 12 / globalScale)
          const size = getNodeSize(n)

          // Check if node type is visible
          const isNodeTypeVisible = !visibleNodeTypes || visibleNodeTypes.has(n.type)
          const opacity = isNodeTypeVisible ? 1 : 0

          // Skip rendering if not visible
          if (opacity === 0) return

          // Save context and set global alpha for opacity
          ctx.save()
          ctx.globalAlpha = opacity

          // Add shadow for selected or hovered nodes
          const isSelected = selectedNodeId === n.id
          const isHovered = hoveredNode?.id === n.id
          
          if (isSelected) {
            ctx.shadowColor = currentTheme === "dark" ? "rgba(255, 255, 255, 1)" : "rgba(0, 0, 0, 0.8)"
            ctx.shadowBlur = 40
            ctx.shadowOffsetX = 0
            ctx.shadowOffsetY = 0
          } else if (isHovered) {
            ctx.shadowColor = currentTheme === "dark" ? "rgba(255, 255, 255, 0.6)" : "rgba(0, 0, 0, 0.5)"
            ctx.shadowBlur = 25
            ctx.shadowOffsetX = 0
            ctx.shadowOffsetY = 0
          }

          // Draw node - simplified shapes for performance
          ctx.fillStyle = getNodeColor(n)
          ctx.beginPath()

          switch (n.type) {
            case 'Node':
              // Simplified: Use square instead of rounded square
              ctx.rect(n.x - size, n.y - size, size * 2, size * 2)
              break
            case 'Application':
              // Circle (fast)
              ctx.arc(n.x, n.y, size, 0, 2 * Math.PI, false)
              break
            case 'Topic':
              // Simplified: Use simple diamond (4 lines) instead of rounded
              ctx.moveTo(n.x, n.y - size)
              ctx.lineTo(n.x + size, n.y)
              ctx.lineTo(n.x, n.y + size)
              ctx.lineTo(n.x - size, n.y)
              ctx.closePath()
              break
            case 'Library':
              // Triangle pointing up
              ctx.moveTo(n.x, n.y - size)
              ctx.lineTo(n.x + size, n.y + size * 0.6)
              ctx.lineTo(n.x - size, n.y + size * 0.6)
              ctx.closePath()
              break
            case 'Broker':
              // Simplified: Use hexagon without rounding
              const angle = (Math.PI * 2) / 6
              for (let i = 0; i < 6; i++) {
                const x = n.x + size * Math.cos(angle * i - Math.PI / 2)
                const y = n.y + size * Math.sin(angle * i - Math.PI / 2)
                if (i === 0) {
                  ctx.moveTo(x, y)
                } else {
                  ctx.lineTo(x, y)
                }
              }
              ctx.closePath()
              break
            default:
              // Circle for unknown types (fast)
              ctx.arc(n.x, n.y, size, 0, 2 * Math.PI, false)
              break
          }

          ctx.fill()

          // Reset shadow before drawing border
          ctx.shadowColor = 'transparent'
          ctx.shadowBlur = 0

          // Draw border for all nodes
          const borderColor = selectedNodeId === n.id 
            ? (currentTheme === "dark" ? "#ffffff" : "#1e293b")
            : (currentTheme === "dark" ? "rgba(255, 255, 255, 0.3)" : "rgba(30, 41, 59, 0.35)")
          ctx.strokeStyle = borderColor
          ctx.lineWidth = selectedNodeId === n.id ? 2.5 / globalScale : 1 / globalScale
          ctx.stroke()

          // Show labels at all zoom levels
          if (label) {
            const labelFontSize = Math.max(8, 10 / globalScale)
            ctx.font = `${labelFontSize}px sans-serif`
            ctx.textAlign = "center"
            ctx.textBaseline = "top"

            // Measure text for background
            const textMetrics = ctx.measureText(label)
            const textWidth = textMetrics.width
            const textHeight = labelFontSize
            const padding = 3
            const bgX = n.x - textWidth / 2 - padding
            const bgY = n.y + size + 4 - padding
            const bgWidth = textWidth + padding * 2
            const bgHeight = textHeight + padding * 2
            const borderRadius = 4

            // Draw semi-transparent rounded background
            ctx.fillStyle = currentTheme === "dark" ? "rgba(0, 0, 0, 0.7)" : "rgba(255, 255, 255, 0.8)"
            ctx.beginPath()
            ctx.roundRect(bgX, bgY, bgWidth, bgHeight, borderRadius)
            ctx.fill()

            // Draw label text
            ctx.fillStyle = currentTheme === "dark" ? "#ffffff" : "#000000"
            ctx.fillText(label, n.x, n.y + size + 4)
          }

          // Restore context
          ctx.restore()
        }}
        backgroundColor={cardBgColor}
        cooldownTicks={10}
        warmupTicks={0}
        d3AlphaDecay={0.03}
        d3VelocityDecay={0.4}
        d3AlphaMin={0.001}
        enableNodeDrag={true}
        enableZoomInteraction={true}
        enablePanInteraction={true}
        onEngineStop={() => {
          // Stop the simulation completely when it has settled
          if (graphRef.current) {
            graphRef.current.pauseAnimation()
          }
        }}
        onEngineTick={() => {
          // OPTIMIZED: Configure forces only once on first tick
          const fg = graphRef.current
          if (fg && !forcesConfiguredRef.current) {
            try {
              const linkForce = fg.d3Force('link')
              if (linkForce) {
                linkForce.strength((link: any) => {
                  const linkType = link.type || 'DEPENDS_ON'
                  switch(linkType) {
                    case 'RUNS_ON': return 1.0
                    case 'DEPENDS_ON': return 0.05
                    case 'PUBLISHES_TO': return 0.02
                    case 'SUBSCRIBES_TO': return 0.02
                    case 'ROUTES': return 0.02
                    default: return 0.03
                  }
                })
                linkForce.distance((link: any) => {
                  const linkType = link.type || 'DEPENDS_ON'
                  switch(linkType) {
                    case 'RUNS_ON': return 30
                    case 'DEPENDS_ON': return 80
                    case 'PUBLISHES_TO': return 120
                    case 'SUBSCRIBES_TO': return 120
                    default: return 100
                  }
                })
              }
              const chargeForce = fg.d3Force('charge')
              if (chargeForce) {
                chargeForce.strength(-300)
                chargeForce.distanceMax(400)
              }
              const centerForce = fg.d3Force('center')
              if (centerForce) {
                centerForce.strength(0.05)
              }
              forcesConfiguredRef.current = true
            } catch (e) {
              // Silently ignore
            }
          }
        }}
        onNodeDragEnd={handleNodeDragEnd}
        dagMode={undefined}
        dagLevelDistance={undefined}
        onRenderFramePre={(ctx: CanvasRenderingContext2D, globalScale: number) => {
          // Optimize canvas rendering based on zoom level
          ctx.imageSmoothingEnabled = globalScale > 1
          ctx.imageSmoothingQuality = 'high'
        }}
      />
      
      {/* Zoom Indicator with smooth transitions */}
      <div className="absolute bottom-4 right-4 bg-background/95 backdrop-blur-md border border-border rounded-lg px-3 py-2 text-sm font-medium shadow-lg transition-all duration-300 hover:scale-105 hover:shadow-xl">
        <div className="flex items-center gap-2">
          <span className="text-muted-foreground transition-colors">Zoom:</span>
          <span className="text-foreground font-mono tabular-nums transition-all">{(currentZoom * 100).toFixed(0)}%</span>
        </div>
      </div>

      {/* Hover Tooltip with smooth entrance animation */}
      {hoveredNode && (
        <div className="absolute bottom-16 right-4 bg-background/95 backdrop-blur-md border border-border rounded-lg p-3 shadow-2xl max-w-xs animate-in slide-in-from-right-2 fade-in duration-200 transition-all">
          <div className="space-y-1">
            <div className="font-bold text-foreground">{hoveredNode.label}</div>
            <div className="text-xs text-muted-foreground">{hoveredNode.type}</div>
            {hoveredNode.criticality_level && (
              <div className="text-xs">
                <span className="text-muted-foreground">Criticality: </span>
                <span className="font-medium capitalize">{hoveredNode.criticality_level}</span>
              </div>
            )}
            {hoveredNode.properties && Object.keys(hoveredNode.properties).length > 0 && (
              <div className="text-xs mt-2 pt-2 border-t border-border">
                {Object.entries(hoveredNode.properties).slice(0, 3).map(([key, value]) => {
                  // Format numbers: 2 decimals for floats, whole numbers for integers
                  let displayValue = String(value)
                  if (typeof value === 'number') {
                    displayValue = Number.isInteger(value) ? value.toString() : value.toFixed(2)
                  }
                  return (
                    <div key={key} className="flex justify-between gap-2">
                      <span className="text-muted-foreground truncate">{key}:</span>
                      <span className="font-medium text-right truncate">{displayValue}</span>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Interactive Guide with smooth entrance */}
      {!selectedNodeId && !hoveredNode && currentZoom === 1 && !isRendering && (
        <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-primary/10 backdrop-blur-sm border border-primary/20 rounded-lg px-4 py-2 text-sm shadow-lg animate-in fade-in slide-in-from-top-2 duration-500 delay-700">
          <span className="text-primary font-medium">💡 Click nodes to explore • Drag to reposition • Scroll to zoom</span>
        </div>
      )}
    </div>
  )
})

ForceGraph2DWrapper.displayName = 'ForceGraph2DWrapper'
