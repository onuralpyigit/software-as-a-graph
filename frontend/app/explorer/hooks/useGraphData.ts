import { useState, useEffect, useCallback, useRef } from 'react'
import { apiClient } from '@/lib/api/client'
import type { ForceGraphData, GraphNode, GraphLink } from '@/lib/types/api'
import type { GraphView } from '@/lib/types/graph-views'
import { GRAPH_VIEWS } from '@/lib/types/graph-views'

export function useGraphData(isConnected: boolean, currentView: GraphView = 'complete') {
  const [graphData, setGraphData] = useState<ForceGraphData | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Whether the initial load for the current view has completed
  const hasLoadedRef = useRef(false)
  // Monotonically increasing counter to discard stale responses
  const requestIdRef = useRef(0)
  // Track previous view so we can reset on view change
  const prevViewRef = useRef(currentView)

  // Synchronously reset when the view changes (runs during render, before effects)
  if (prevViewRef.current !== currentView) {
    prevViewRef.current = currentView
    hasLoadedRef.current = false
  }

  const fetchTopologyData = useCallback(async (nodeId?: string) => {
    if (!isConnected) return null

    const isRefresh = hasLoadedRef.current
    const thisRequestId = ++requestIdRef.current

    if (isRefresh) {
      setIsRefreshing(true)
    } else {
      setIsLoading(true)
      setError(null)
    }

    try {
      const viewConfig = GRAPH_VIEWS[currentView]
      const isStructuralView = currentView === 'complete'
      
      let data: ForceGraphData
      
      if (isStructuralView) {
        // For complete view, use the topology endpoint
        const response = await apiClient.getTopologyData(nodeId, 1000)
        
        // Transform components to have consistent structure with properties field
        const transformedNodes = (response.components || []).map((c: any) => ({
          id: c.id,
          label: c.label || c.name || c.id,
          type: c.type,
          properties: c, // Store entire component as properties
          degree: 0,
          criticality_score: c.weight || 0,
        }));
        
        const transformedEdges = (response.edges || []).map((edge: any) => ({
          ...edge,
          type: edge.relation_type || edge.type || 'default'
        }))

        data = {
          nodes: transformedNodes,
          links: transformedEdges,
          metadata: {}
        }
      } else {
        // For layer views, use the getGraphData with appropriate filters
        data = await apiClient.getGraphData({
          relationship_types: viewConfig.relationshipTypes,
          dependency_types: viewConfig.dependencyTypes,
          node_types: viewConfig.nodeTypes,
          include_metrics: false,
          limit_nodes: 1000,
        })
      }
      
      // Discard if a newer request has been started
      if (thisRequestId !== requestIdRef.current) return null
      
      hasLoadedRef.current = true
      setGraphData(data)
      return data
    } catch (err) {
      // Discard stale errors
      if (thisRequestId !== requestIdRef.current) return null

      console.error('Failed to fetch topology data:', err)
      setError(err instanceof Error ? err.message : 'Failed to fetch topology data')
      return null
    } finally {
      // Only clear loading flags if this is still the latest request
      if (thisRequestId === requestIdRef.current) {
        setIsLoading(false)
        setIsRefreshing(false)
      }
    }
  }, [isConnected, currentView])

  useEffect(() => {
    if (isConnected) {
      fetchTopologyData()
    }
  }, [isConnected, currentView, fetchTopologyData])

  return { graphData, isLoading, isRefreshing, error, fetchTopologyData, setGraphData }
}
