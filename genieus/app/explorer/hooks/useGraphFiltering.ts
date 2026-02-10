import { useState, useMemo } from 'react'
import type { ForceGraphData, GraphNode } from '@/lib/types/api'

export function useGraphFiltering(graphData: ForceGraphData | null, selectedNode: GraphNode | null) {
  const [hiddenNodeTypes, setHiddenNodeTypes] = useState<Set<string>>(new Set())
  const [hiddenLinkTypes, setHiddenLinkTypes] = useState<Set<string>>(new Set())

  const filteredGraphData = useMemo(() => {
    if (!graphData) return null
    
    const filteredNodes = graphData.nodes.filter(node => 
      !hiddenNodeTypes.has(node.type) || node.id === selectedNode?.id
    )
    const visibleNodeIds = new Set(filteredNodes.map(n => n.id))
    
    const filteredLinks = graphData.links.filter(link => {
      const sourceId = typeof link.source === 'string' ? link.source : (link.source as any).id
      const targetId = typeof link.target === 'string' ? link.target : (link.target as any).id
      
      if (selectedNode && (sourceId === selectedNode.id || targetId === selectedNode.id)) {
        return visibleNodeIds.has(sourceId) && visibleNodeIds.has(targetId)
      }
      
      if (hiddenLinkTypes.has(link.type)) return false
      return visibleNodeIds.has(sourceId) && visibleNodeIds.has(targetId)
    })
    
    return {
      ...graphData,
      nodes: filteredNodes,
      links: filteredLinks
    }
  }, [graphData, hiddenNodeTypes, hiddenLinkTypes, selectedNode])

  const uniqueNodeTypes = useMemo(() => {
    if (!graphData?.nodes.length) return []
    return [...new Set(graphData.nodes.map(n => n.type))]
  }, [graphData])

  const uniqueLinkTypes = useMemo(() => {
    if (!graphData?.links.length) return []
    return [...new Set(graphData.links.map(l => l.type))]
  }, [graphData])

  const toggleNodeType = (type: string) => {
    setHiddenNodeTypes(prev => {
      const newSet = new Set(prev)
      if (newSet.has(type)) {
        newSet.delete(type)
      } else {
        newSet.add(type)
      }
      return newSet
    })
  }

  const toggleLinkType = (type: string) => {
    setHiddenLinkTypes(prev => {
      const newSet = new Set(prev)
      if (newSet.has(type)) {
        newSet.delete(type)
      } else {
        newSet.add(type)
      }
      return newSet
    })
  }

  return {
    filteredGraphData,
    hiddenNodeTypes,
    hiddenLinkTypes,
    uniqueNodeTypes,
    uniqueLinkTypes,
    toggleNodeType,
    toggleLinkType
  }
}
