import { useState } from 'react'
import { apiClient } from '@/lib/api/client'
import type { ForceGraphData, GraphNode, GraphLink } from '@/lib/types/api'
import type { GraphView } from '@/lib/types/graph-views'
import { GRAPH_VIEWS } from '@/lib/types/graph-views'

interface DrillDownLevel {
  nodeId: string
  nodeType: string
  nodeLabel?: string
  childNodes?: GraphNode[]
}

interface HierarchyData {
  node: GraphNode
  children: GraphNode[]
  links: GraphLink[]
}

export function useGraphDrillDown(currentView: GraphView = 'complete') {
  const [drillDownHistory, setDrillDownHistory] = useState<DrillDownLevel[]>([])
  const [hierarchyData, setHierarchyData] = useState<Map<string, HierarchyData>>(new Map())
  const [breadcrumbPath, setBreadcrumbPath] = useState<Array<{id: string | null, label: string}>>([])
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set())
  const [isLoadingChildren, setIsLoadingChildren] = useState(false)

  // Get view configuration for filtering
  const viewConfig = GRAPH_VIEWS[currentView]

  // Filter links based on current view
  const filterLinksByView = (links: GraphLink[]): GraphLink[] => {
    // For complete view, show all links
    if (currentView === 'complete') {
      return links
    }

    console.log('🔧 [Filter] View config:', {
      relationshipTypes: viewConfig.relationshipTypes,
      dependencyTypes: viewConfig.dependencyTypes,
      nodeTypes: viewConfig.nodeTypes
    })

    // For layer views, filter by relationship types and dependency types
    const filtered = links.filter(link => {
      // Check if relationship type matches
      if (!viewConfig.relationshipTypes.includes(link.type)) {
        return false
      }

      // Check dependency type if specified
      if (viewConfig.dependencyTypes && viewConfig.dependencyTypes.length > 0) {
        const depType = link.properties?.dependency_type
        if (!depType || !viewConfig.dependencyTypes.includes(depType)) {
          console.log('🚫 [Filter] Excluded link:', link.type, 'depType:', depType, 'from:', link.source, 'to:', link.target)
          return false
        }
      }

      return true
    })
    
    console.log('✅ [Filter] Kept', filtered.length, 'of', links.length, 'links')
    return filtered
  }

  const updateBreadcrumbPath = (history: DrillDownLevel[]) => {
    const newPath: Array<{id: string | null, label: string}> = [{id: null, label: 'System'}]
    history.forEach(h => {
      newPath.push({id: h.nodeId, label: h.nodeLabel || h.nodeId})
    })
    setBreadcrumbPath(newPath)
  }

  const drillDown = async (
    nodeId: string,
    graphData: ForceGraphData | null,
    startFresh: boolean = false
  ): Promise<ForceGraphData | null> => {
    try {
      setIsLoadingChildren(true)
      
      // Check if node is already in history - if so, navigate back to it (unless startFresh is true)
      const existingIndex = !startFresh ? drillDownHistory.findIndex(h => h.nodeId === nodeId) : -1
      if (existingIndex >= 0) {
        // Node is in current path - navigate back to it
        const newHistory = drillDownHistory.slice(0, existingIndex + 1)
        setDrillDownHistory(newHistory)
        updateBreadcrumbPath(newHistory)
        
        // Auto-expand nodes in the drill path
        const pathNodeIds = newHistory.map(h => h.nodeId)
        setExpandedNodes(new Set(pathNodeIds))
        
        const levelData = hierarchyData.get(nodeId)
        if (levelData) {
          const allNodes = [levelData.node, ...levelData.children]
          return {
            nodes: allNodes,
            links: levelData.links.filter(link => {
              const sourceId = typeof link.source === 'string' ? link.source : (link.source as any).id
              const targetId = typeof link.target === 'string' ? link.target : (link.target as any).id
              return allNodes.some(n => n.id === sourceId) && allNodes.some(n => n.id === targetId)
            }),
            metadata: {}
          }
        }
      }
      
      // For layer views (application, infrastructure, middleware), we need derived relationships (fetch_structural=false)
      // For complete view, we need structural relationships (fetch_structural=true)
      const fetchStructural = currentView === 'complete'
      
      console.log('🔍 [Drill Down] View:', currentView, 'fetchStructural:', fetchStructural, 'nodeId:', nodeId)
      
      const connections = await apiClient.getNodeConnectionsWithDepth(nodeId, fetchStructural, 1)
      
      console.log('📦 [Drill Down] Received:', connections.nodes.length, 'nodes,', connections.links.length, 'links')
      
      const transformedEdges = connections.links.map((edge: any) => ({
        ...edge,
        type: edge.relation_type || edge.type || 'default'
      }))

      // Filter links based on current view
      const filteredLinks = filterLinksByView(transformedEdges)
      
      console.log('🔗 [Drill Down] After view filter:', filteredLinks.length, 'links')

      // Only keep links that directly involve the clicked node
      const nodeSpecificLinks = filteredLinks.filter(link => {
        const sourceId = typeof link.source === 'string' ? link.source : (link.source as any).id
        const targetId = typeof link.target === 'string' ? link.target : (link.target as any).id
        return sourceId === nodeId || targetId === nodeId
      })
      
      console.log('🎯 [Drill Down] Node-specific links:', nodeSpecificLinks.length)

      // Get all node IDs that are connected to the clicked node via filtered links
      const connectedNodeIds = new Set<string>([nodeId])
      nodeSpecificLinks.forEach(link => {
        const sourceId = typeof link.source === 'string' ? link.source : (link.source as any).id
        const targetId = typeof link.target === 'string' ? link.target : (link.target as any).id
        connectedNodeIds.add(sourceId)
        connectedNodeIds.add(targetId)
      })

      // Filter nodes: only include nodes that are connected to the clicked node
      let filteredNodes = connections.nodes.filter(n => connectedNodeIds.has(n.id))

      // Additionally filter by node type if specified in view config
      if (currentView !== 'complete' && viewConfig.nodeTypes && viewConfig.nodeTypes.length > 0) {
        filteredNodes = filteredNodes.filter(node => viewConfig.nodeTypes!.includes(node.type))
      }
      
      console.log('✨ [Drill Down] Final result:', filteredNodes.length, 'nodes,', nodeSpecificLinks.length, 'links')
      
      const filteredData = {
        nodes: filteredNodes,
        links: nodeSpecificLinks,
        metadata: {}
      }
      
      const node = filteredNodes.find(n => n.id === nodeId)
      if (!node) return null
      
      const children = filteredNodes.filter(n => n.id !== nodeId)
      
      setHierarchyData(prev => {
        const newMap = startFresh ? new Map() : new Map(prev)
        newMap.set(nodeId, { node, children, links: nodeSpecificLinks })
        return newMap
      })
      
      // Add to history (or start fresh if requested)
      const newHistory = startFresh ? [{
        nodeId,
        nodeType: node.type,
        nodeLabel: node.label,
        childNodes: children
      }] : [...drillDownHistory, {
        nodeId,
        nodeType: node.type,
        nodeLabel: node.label,
        childNodes: children
      }]
      setDrillDownHistory(newHistory)
      updateBreadcrumbPath(newHistory)
      
      // Auto-expand all nodes in the drill path
      const pathNodeIds = newHistory.map(h => h.nodeId)
      setExpandedNodes(new Set(pathNodeIds))
      
      return filteredData
    } catch (err) {
      console.error('Failed to drill down:', err)
      return null
    } finally {
      setIsLoadingChildren(false)
    }
  }

  const navigateToBreadcrumb = (nodeId: string | null): ForceGraphData | null => {
    if (nodeId === null) {
      reset()
      return null
    }
    
    const clickedIndex = breadcrumbPath.findIndex(p => p.id === nodeId)
    if (clickedIndex === -1) return null
    
    // If clicking the current (last) breadcrumb, do nothing but return current data
    if (clickedIndex === breadcrumbPath.length - 1) {
      const levelData = hierarchyData.get(nodeId)
      if (levelData) {
        const allNodes = [levelData.node, ...levelData.children]
        return {
          nodes: allNodes,
          links: levelData.links.filter(link => {
            const sourceId = typeof link.source === 'string' ? link.source : (link.source as any).id
            const targetId = typeof link.target === 'string' ? link.target : (link.target as any).id
            return allNodes.some(n => n.id === sourceId) && allNodes.some(n => n.id === targetId)
          }),
          metadata: {}
        }
      }
      return null
    }
    
    const newPath = breadcrumbPath.slice(0, clickedIndex + 1)
    const newHistory = drillDownHistory.slice(0, clickedIndex)
    
    setBreadcrumbPath(newPath)
    setDrillDownHistory(newHistory)
    
    // Auto-expand nodes in the navigation path
    const pathNodeIds = newHistory.map(h => h.nodeId)
    setExpandedNodes(new Set(pathNodeIds))
    
    const levelData = hierarchyData.get(nodeId)
    if (levelData) {
      const allNodes = [levelData.node, ...levelData.children]
      return {
        nodes: allNodes,
        links: levelData.links.filter(link => {
          const sourceId = typeof link.source === 'string' ? link.source : (link.source as any).id
          const targetId = typeof link.target === 'string' ? link.target : (link.target as any).id
          return allNodes.some(n => n.id === sourceId) && allNodes.some(n => n.id === targetId)
        }),
        metadata: {}
      }
    }
    
    return null
  }

  const reset = () => {
    setDrillDownHistory([])
    setBreadcrumbPath([])
    setHierarchyData(new Map())
    setExpandedNodes(new Set())
  }

  const toggleNodeExpansion = (nodeId: string) => {
    setExpandedNodes(prev => {
      const newSet = new Set(prev)
      if (newSet.has(nodeId)) {
        newSet.delete(nodeId)
      } else {
        newSet.add(nodeId)
      }
      return newSet
    })
  }

  return {
    drillDownHistory,
    hierarchyData,
    breadcrumbPath,
    expandedNodes,
    isLoadingChildren,
    drillDown,
    navigateToBreadcrumb,
    reset,
    toggleNodeExpansion
  }
}
