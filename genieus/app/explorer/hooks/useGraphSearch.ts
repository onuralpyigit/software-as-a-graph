import { useState, useRef } from 'react'
import { apiClient } from '@/lib/api/client'
import type { GraphNode } from '@/lib/types/api'
import type { GraphView } from '@/lib/types/graph-views'
import { GRAPH_VIEWS } from '@/lib/types/graph-views'

export function useGraphSearch(isConnected: boolean, currentView: GraphView = 'complete') {
  const [searchQuery, setSearchQuery] = useState<string>("")
  const [searchResults, setSearchResults] = useState<GraphNode[]>([])
  const [showSearchResults, setShowSearchResults] = useState(false)
  const [isSearching, setIsSearching] = useState(false)
  const searchTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  const handleSearch = (query: string) => {
    setSearchQuery(query)
    
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current)
    }
    
    if (!isConnected || !query.trim()) {
      setSearchResults([])
      setShowSearchResults(false)
      setIsSearching(false)
      return
    }

    setIsSearching(true)
    
    searchTimeoutRef.current = setTimeout(async () => {
      try {
        const results = await apiClient.searchNodes(query.trim(), 50)
        const mappedResults = results.map((r: any) => ({
          ...r,
          properties: r.properties || {}
        })) as GraphNode[]
        
        // Filter results based on current view's node types
        const viewConfig = GRAPH_VIEWS[currentView]
        let filteredResults = mappedResults
        
        if (currentView !== 'complete' && viewConfig.nodeTypes && viewConfig.nodeTypes.length > 0) {
          filteredResults = mappedResults.filter(node => viewConfig.nodeTypes!.includes(node.type))
          console.log('🔍 [Search] Filtered results for', currentView, ':', filteredResults.length, 'of', mappedResults.length)
        }
        
        setSearchResults(filteredResults)
        setShowSearchResults(true)
      } catch (err) {
        console.error('Search failed:', err)
        setSearchResults([])
        setShowSearchResults(true)
      } finally {
        setIsSearching(false)
      }
    }, 300)
  }

  const handleClearSearch = () => {
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current)
    }
    setSearchQuery("")
    setSearchResults([])
    setShowSearchResults(false)
    setIsSearching(false)
  }

  return {
    searchQuery,
    searchResults,
    showSearchResults,
    isSearching,
    handleSearch,
    handleClearSearch,
    setShowSearchResults
  }
}
