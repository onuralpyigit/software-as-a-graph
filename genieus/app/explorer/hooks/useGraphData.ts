import { useState, useEffect } from 'react'
import { apiClient } from '@/lib/api/client'
import type { ForceGraphData, GraphNode, GraphLink } from '@/lib/types/api'
import type { GraphView } from '@/lib/types/graph-views'
import { GRAPH_VIEWS } from '@/lib/types/graph-views'

export function useGraphData(isConnected: boolean, currentView: GraphView = 'complete') {
  const [graphData, setGraphData] = useState<ForceGraphData | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchTopologyData = async (nodeId?: string) => {
    if (!isConnected) return

    setIsLoading(true)
    setError(null)

    try {
      const viewConfig = GRAPH_VIEWS[currentView]
      const isStructuralView = currentView === 'complete'
      
      console.log('📊 [useGraphData] Fetching topology for view:', currentView, 'isStructural:', isStructuralView)
      
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
        console.log('🔧 [useGraphData] Fetching with filters:', {
          relationship_types: viewConfig.relationshipTypes,
          dependency_types: viewConfig.dependencyTypes,
          node_types: viewConfig.nodeTypes
        })
        
        data = await apiClient.getGraphData({
          relationship_types: viewConfig.relationshipTypes,
          dependency_types: viewConfig.dependencyTypes,
          node_types: viewConfig.nodeTypes,
          include_metrics: false,
          limit_nodes: 1000,
        })
      }
      
      console.log('✅ [useGraphData] Received:', data.nodes.length, 'nodes,', data.links.length, 'links')
      
      setGraphData(data)
      return data
    } catch (err) {
      console.error('Failed to fetch topology data:', err)
      setError(err instanceof Error ? err.message : 'Failed to fetch topology data')
      return null
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    if (isConnected) {
      fetchTopologyData()
    }
  }, [isConnected, currentView])

  return { graphData, isLoading, error, fetchTopologyData, setGraphData }
}
