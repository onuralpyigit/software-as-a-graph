"use client"

import { useEffect, useRef, useState, forwardRef, useImperativeHandle, useMemo, useCallback, memo } from "react"
import dynamic from "next/dynamic"
import { useTheme } from "next-themes"
import type { GraphNode, GraphLink } from "@/lib/types/api"

// Polyfill GPUShaderStage to prevent errors when WebGPU is not available
if (typeof window !== 'undefined' && typeof (window as any).GPUShaderStage === 'undefined') {
  (window as any).GPUShaderStage = {
    VERTEX: 1,
    FRAGMENT: 2,
    COMPUTE: 4
  }
}

const ForceGraph3D = dynamic(() => import("react-force-graph-3d").catch(err => {
  console.warn('Failed to load 3D graph, falling back to 2D', err)
  return import("react-force-graph-2d")
}), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center">
      <div className="text-sm text-muted-foreground">Loading 3D graph...</div>
    </div>
  ),
})

interface ForceGraph3DWrapperProps {
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

export interface ForceGraph3DRef {
  zoomToFit: (duration?: number, padding?: number) => void
  exportPNG: (filename?: string) => void
  centerOnNode: (nodeId: string) => void
}

export const ForceGraph3DWrapper = forwardRef<ForceGraph3DRef, ForceGraph3DWrapperProps>(({
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
  const [isRendering, setIsRendering] = useState(true)
  const zoomTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const pendingCenterNodeId = useRef<string | null>(null)
  const centeringFrameCount = useRef(0)
  const [spriteTextLoaded, setSpriteTextLoaded] = useState(false)

  // Get actual theme - handle undefined systemTheme
  const currentTheme = theme === 'system' ? (systemTheme || 'dark') : theme

  // Configure d3 force link strength using fgRef - OPTIMIZED: Single configuration on first tick
  const forcesConfiguredRef = useRef(false)

  // Track connected nodes for highlighting
  const [connectedNodeIds, setConnectedNodeIds] = useState<Set<string>>(new Set())

  // Load SpriteText library dynamically on mount
  useEffect(() => {
    if (typeof window !== 'undefined') {
      import('three-spritetext').then(module => {
        (window as any).SpriteText = module.default
        setSpriteTextLoaded(true)
      }).catch(err => {
        console.warn('Failed to load SpriteText:', err)
      })
    }
  }, [])

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

  // Memoize colors for performance
  const nodeColorByType = useMemo(() => ({
    Application: currentTheme === "dark" ? "#3b82f6" : "#2563eb",
    Node: currentTheme === "dark" ? "#ef4444" : "#dc2626",
    Broker: currentTheme === "dark" ? "#a1a1aa" : "#71717a",
    Topic: currentTheme === "dark" ? "#facc15" : "#eab308",
    Library: currentTheme === "dark" ? "#06b6d4" : "#0891b2",
    Unknown: currentTheme === "dark" ? "#a1a1aa" : "#71717a",
  } as Record<string, string>), [currentTheme])

  const linkColorByType = useMemo(() => ({
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
  } as Record<string, string>), [currentTheme])

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

  // Memoize expensive functions with caching
  const getNodeColor = useMemo(() => {
    return (node: any) => {
      const n = node as GraphNode
      const isHovered = hoveredNode?.id === n.id
      const isNodeHighlighted = !selectedNodeId || n.id === selectedNodeId || connectedNodeIds.has(n.id)
      
      let baseColor: string
      if (colorByCriticality && n.criticality_level) {
        baseColor = nodeColorByCriticality(n.criticality_level)
      } else {
        baseColor = nodeColorByType[n.type] || nodeColorByType.Unknown
      }
      
      // Brighten hovered nodes
      if (isHovered) {
        const r = Math.min(255, parseInt(baseColor.slice(1, 3), 16) + 50)
        const g = Math.min(255, parseInt(baseColor.slice(3, 5), 16) + 50)
        const b = Math.min(255, parseInt(baseColor.slice(5, 7), 16) + 50)
        return `rgb(${r}, ${g}, ${b})`
      }
      
      if (selectedNodeId && !isNodeHighlighted) {
        const r = parseInt(baseColor.slice(1, 3), 16)
        const g = parseInt(baseColor.slice(3, 5), 16)
        const b = parseInt(baseColor.slice(5, 7), 16)
        return `rgba(${r}, ${g}, ${b}, 0.2)`
      }
      
      return baseColor
    }
  }, [selectedNodeId, connectedNodeIds, hoveredNode, colorByCriticality, nodeColorByType, nodeColorByCriticality])

  const getNodeSize = useMemo(() => {
    return (node: any) => {
      const n = node as GraphNode
      const isHovered = hoveredNode?.id === n.id
      
      let baseSize: number
      if (colorByCriticality && n.criticality_score) {
        baseSize = 2 + (n.criticality_score * 4)
      } else {
        switch (n.type) {
          case 'Node':
            baseSize = 6
            break
          case 'Application':
            baseSize = 4
            break
          case 'Topic':
          case 'Broker':
            baseSize = 2.5
            break
          default:
            baseSize = 3
        }
      }
      
      // Increase size on hover
      return isHovered ? baseSize * 1.4 : baseSize
    }
  }, [colorByCriticality, hoveredNode])

  // Helper function to convert hex to rgba
  const hexToRgba = useCallback((hex: string, alpha: number) => {
    const r = parseInt(hex.slice(1, 3), 16)
    const g = parseInt(hex.slice(3, 5), 16)
    const b = parseInt(hex.slice(5, 7), 16)
    return `rgba(${r}, ${g}, ${b}, ${alpha})`
  }, [])

  // Memoize node lookup map for faster link rendering
  const nodeMap = useMemo(() => {
    const map = new Map<string, GraphNode>()
    nodes.forEach(node => map.set(node.id, node))
    return map
  }, [nodes])

  // Define getLinkId first (needed by getLinkColor)
  const getLinkId = useCallback((link: GraphLink) => {
    const sourceId = typeof link.source === 'string' ? link.source : (link.source as any).id
    const targetId = typeof link.target === 'string' ? link.target : (link.target as any).id
    return `${sourceId}-${link.type}-${targetId}`
  }, [])

  // Simplified link color function - OPTIMIZED: Removed caching overhead
  const getLinkColor = useCallback((link: any) => {
    const l = link as GraphLink
    const linkId = getLinkId(l)

    const isSelected = selectedLinkId === linkId
    const isHovered = hoveredLink && getLinkId(hoveredLink) === linkId
    const color = linkColorByType[l.type] || linkColorByType.default

    // Get source and target nodes
    const sourceId = typeof l.source === 'string' ? l.source : (l.source as any).id
    const targetId = typeof l.target === 'string' ? l.target : (l.target as any).id
    const sourceNode = nodeMap.get(sourceId)
    const targetNode = nodeMap.get(targetId)

    // Visibility checks
    const isSourceVisible = !visibleNodeTypes || !sourceNode || visibleNodeTypes.has(sourceNode.type)
    const isTargetVisible = !visibleNodeTypes || !targetNode || visibleNodeTypes.has(targetNode.type)
    const isLinkTypeVisible = !visibleLinkTypes || visibleLinkTypes.has(l.type)
    const isPubSubLink = l.type === 'PUBLISHES_TO' || l.type === 'SUBSCRIBES_TO'
    const isConnectedToSelected = selectedNodeId && (sourceId === selectedNodeId || targetId === selectedNodeId)

    // Fast path for invisible links
    if (!isLinkTypeVisible || !isSourceVisible || !isTargetVisible ||
        (isPubSubLink && !isPubSubVisible && !isConnectedToSelected)) {
      return hexToRgba(color, 0)
    }

    if (isSelected) return '#ffffff'
    if (isHovered) {
      const r = Math.min(255, parseInt(color.slice(1, 3), 16) + 60)
      const g = Math.min(255, parseInt(color.slice(3, 5), 16) + 60)
      const b = Math.min(255, parseInt(color.slice(5, 7), 16) + 60)
      return `rgb(${r}, ${g}, ${b})`
    }
    if (selectedNodeId && !isConnectedToSelected) return hexToRgba(color, 0.1)
    if (isConnectedToSelected) return hexToRgba(color, 0.8)
    return color
  }, [linkColorByType, selectedLinkId, selectedNodeId, visibleNodeTypes, visibleLinkTypes, isPubSubVisible, hexToRgba, nodeMap, hoveredLink, getLinkId])

  // Memoize graph data
  const graphData = useMemo(() => ({ nodes, links }), [nodes, links])

  // Create link width cache for performance
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
    const sourceNode = nodeMap.get(sourceId)
    const targetNode = nodeMap.get(targetId)

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
    let baseWidth = 1.0
    if (weight && typeof weight === 'number' && weight > 0) {
      if (isDependsOn) {
        // Scale DEPENDS_ON edges more prominently based on weight
        baseWidth = Math.max(1.0, Math.min(3.5, 1.0 + Math.sqrt(weight) * 0.5))
      } else {
        baseWidth = Math.max(0.8, Math.min(3.0, 0.8 + Math.sqrt(weight) * 0.3))
      }
    }

    // Add hover effect
    const isHovered = hoveredLink && getLinkId(hoveredLink) === linkId
    const result = isSelected ? baseWidth * 3 : (isHovered ? baseWidth * 2 : baseWidth)
    linkWidthCache.current.set(cacheKey, result)
    return result
  }, [getLinkId, selectedLinkId, isPubSubVisible, nodeMap, visibleNodeTypes, visibleLinkTypes, hoveredLink])

  // Dynamic arrow length based on link width
  const getLinkArrowLength = useCallback((link: any) => {
    const originalLink = link.__data ?? link
    const width = getLinkWidth(originalLink)
    // Make arrows proportional to link width with a reasonable range for 3D
    return width > 0 ? Math.max(4, Math.min(10, 4 + width * 1.2)) : 0
  }, [getLinkWidth])

  // Clear cache when dependencies change
  useEffect(() => {
    linkWidthCache.current.clear()
  }, [selectedLinkId, isPubSubVisible, visibleNodeTypes, visibleLinkTypes])

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

  const handleLinkHover = useCallback((link: any) => {
    if (link) {
      const l = link as GraphLink
      setHoveredLink(l)
      const canvas = containerRef.current?.querySelector('canvas')
      if (canvas) {
        canvas.style.cursor = 'pointer'
      }
    } else {
      setHoveredLink(null)
      const canvas = containerRef.current?.querySelector('canvas')
      if (canvas) {
        canvas.style.cursor = 'default'
      }
    }
  }, [])

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

  const handleBackgroundClick = () => {
    if (onBackgroundClick) {
      onBackgroundClick()
    }
  }

  const handleNodeDragEnd = (node: any) => {
    const n = node as GraphNode & { fx?: number; fy?: number; fz?: number; x: number; y: number; z: number }
    // Fix node position after drag
    n.fx = n.x
    n.fy = n.y
    n.fz = n.z
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
        if (node.x !== undefined && node.y !== undefined && node.z !== undefined) {
          const distance = 300
          graphRef.current?.cameraPosition(
            { x: node.x, y: node.y, z: node.z + distance },
            node,
            1000
          )
        }
      }
    },
    exportPNG: (filename = 'graph-3d.png') => {
      if (containerRef.current) {
        const canvas = containerRef.current.querySelector('canvas') as HTMLCanvasElement
        if (canvas) {
          try {
            // For WebGL canvas (3D), use toDataURL which works more reliably
            const dataURL = canvas.toDataURL('image/png')
            const link = document.createElement('a')
            link.href = dataURL
            link.download = filename
            link.click()
          } catch (error) {
            console.error('Failed to export 3D graph as PNG:', error)
          }
        }
      }
    }
  }))

  // Track camera zoom/distance for 3D - OPTIMIZED: Reduced polling frequency
  useEffect(() => {
    if (!graphRef.current || !onZoomChange) return

    const intervalId = setInterval(() => {
      if (graphRef.current?.camera) {
        const camera = graphRef.current.camera()
        if (camera) {
          const position = camera.position
          const distance = Math.sqrt(position.x ** 2 + position.y ** 2 + position.z ** 2)
          const normalizedZoom = 1500 / Math.max(distance, 100)
          if (Math.abs(normalizedZoom - currentZoom) > 0.1) { // Increased threshold
            setCurrentZoom(normalizedZoom)
            onZoomChange(normalizedZoom)
          }
        }
      }
    }, 500) // OPTIMIZED: Reduced from 200ms to 500ms

    return () => clearInterval(intervalId)
  }, [onZoomChange, currentZoom])

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
      if (node && node.x !== undefined && node.y !== undefined && node.z !== undefined) {
        // More aggressive centering with faster transitions initially
        const duration = centeringFrameCount.current < 5 ? 300 : 500
        const distance = 300
        graphRef.current?.cameraPosition(
          { x: node.x, y: node.y, z: node.z + distance },
          node,
          duration
        )
        
        centeringFrameCount.current++
        
        // Stop centering after 30 frames (about 6 seconds) to allow user interaction
        if (centeringFrameCount.current > 30) {
          pendingCenterNodeId.current = null
        }
      }
    }, 200)
    
    return () => clearInterval(intervalId)
  }, [nodes, autoCenterOnSelect])

  // Background colors from shadcn/ui - use CSS variables for consistency
  const cardBgColor = currentTheme === "dark" ? "hsl(0 0% 3.9%)" : "hsl(0 0% 100%)"

  // Custom node renderer for text labels - extends default node rendering
  const nodeThreeObject = useCallback((node: any) => {
    const n = node as GraphNode
    const SpriteTextClass = (window as any).SpriteText
    
    // Only add sprite text if library is loaded
    if (!SpriteTextClass || !spriteTextLoaded) return undefined

    // Create text sprite label
    const sprite = new SpriteTextClass(n.label || n.id)
    sprite.color = getNodeColor(n)
    sprite.textHeight = 8
    sprite.position.y = getNodeSize(n) + 10 // Position above the node
    
    return sprite
  }, [getNodeColor, getNodeSize, spriteTextLoaded])

  // Handle initial rendering state with smooth transition
  useEffect(() => {
    if (nodes.length > 0) {
      setIsRendering(true)
      const timer = setTimeout(() => setIsRendering(false), 1000)
      return () => clearTimeout(timer)
    }
  }, [nodes.length])

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
            <p className="text-sm text-muted-foreground">Rendering 3D graph...</p>
          </div>
        </div>
      )}

      <ForceGraph3D
        ref={graphRef}
        width={dimensions.width}
        height={dimensions.height}
        graphData={graphData}
        nodeId="id"
        nodeLabel=""
        nodeColor={getNodeColor}
        nodeVal={getNodeSize}
        nodeRelSize={3}
        nodeVisibility={(node: any) => {
          const n = node as GraphNode
          return !visibleNodeTypes || visibleNodeTypes.has(n.type)
        }}
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
        onLinkHover={handleLinkHover}
        onBackgroundClick={handleBackgroundClick}
        onNodeDragEnd={handleNodeDragEnd}
        nodeThreeObject={nodeThreeObject}
        nodeThreeObjectExtend={true}
        backgroundColor={cardBgColor}
        showNavInfo={false}
        cooldownTicks={10}
        warmupTicks={0}
        d3AlphaDecay={0.04}
        d3VelocityDecay={0.7}
        d3AlphaMin={0.005}
        enableNodeDrag={true}
        enableNavigationControls={true}
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
                    case 'DEPENDS_ON': return 0.08
                    case 'PUBLISHES_TO': return 0.02
                    case 'SUBSCRIBES_TO': return 0.02
                    case 'ROUTES': return 0.02
                    default: return 0.03
                  }
                })
                linkForce.distance((link: any) => {
                  const linkType = link.type || 'DEPENDS_ON'
                  switch(linkType) {
                    case 'RUNS_ON': return 40
                    case 'DEPENDS_ON': return 100
                    case 'PUBLISHES_TO': return 180
                    case 'SUBSCRIBES_TO': return 180
                    default: return 120
                  }
                })
              }
              const chargeForce = fg.d3Force('charge')
              if (chargeForce) {
                chargeForce.strength(-300) // OPTIMIZED: Reduced from -400
                chargeForce.distanceMax(400) // OPTIMIZED: Reduced from 500
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
        dagMode={undefined}
        dagLevelDistance={undefined}
        rendererConfig={{
          antialias: false,
          alpha: false,
          powerPreference: 'high-performance',
          precision: 'lowp',
          stencil: false,
          depth: true,
          preserveDrawingBuffer: true
        }}
        extraRenderers={[]}
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

      {/* Navigation Help for 3D */}
      {!selectedNodeId && !hoveredNode && (
        <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-background/90 backdrop-blur-sm border border-border rounded-lg px-4 py-2 text-sm shadow-lg">
          <span className="text-muted-foreground">🖱️ Left click + drag to rotate • Right click + drag to pan • Scroll to zoom</span>
        </div>
      )}
    </div>
  )
})

ForceGraph3DWrapper.displayName = 'ForceGraph3DWrapper'
