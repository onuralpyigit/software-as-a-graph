import React from 'react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Search, X, Loader2, ArrowRight, Link2 } from 'lucide-react'
import type { GraphNode } from '@/lib/types/api'

interface SearchBarProps {
  searchQuery: string
  onSearch: (query: string) => void
  onClear: () => void
  isSearching: boolean
  searchResults: GraphNode[]
  showSearchResults: boolean
  onResultClick: (node: GraphNode) => void
  getConnectionCount: (nodeId: string) => number
  getNodeIcon: (type: string) => React.ComponentType<any>
  nodeColorByType: Record<string, string>
  isConnected: boolean
}

export function SearchBar({
  searchQuery,
  onSearch,
  onClear,
  isSearching,
  searchResults,
  showSearchResults,
  onResultClick,
  getConnectionCount,
  getNodeIcon,
  nodeColorByType,
  isConnected
}: SearchBarProps) {
  return (
    <div className="flex items-center justify-center">
      <div className="relative w-[500px]">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            type="text"
            placeholder="Search all nodes by name, ID, or type..."
            value={searchQuery}
            onChange={(e) => onSearch(e.target.value)}
            onFocus={() => searchQuery && showSearchResults}
            className="pl-10 pr-10 bg-background/95 backdrop-blur-md shadow-xl border-2 border-border"
            disabled={!isConnected}
          />
          {isSearching && (
            <Loader2 className="absolute right-3 top-1/2 transform -translate-y-1/2 h-4 w-4 animate-spin text-muted-foreground" />
          )}
          {searchQuery && !isSearching && (
            <button
              onClick={onClear}
              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground"
            >
              <X className="h-4 w-4" />
            </button>
          )}
        </div>
        
        {/* Search Results Dropdown */}
        {showSearchResults && !isSearching && searchResults.length > 0 && (
          <div className="absolute top-full mt-2 w-full bg-background/98 backdrop-blur-md border-2 border-border rounded-lg shadow-2xl max-h-96 overflow-y-auto z-50">
            <div className="p-2">
              <div className="px-3 py-2 text-xs text-muted-foreground border-b border-border mb-2 flex items-center justify-between">
                <span className="font-medium">
                  {searchResults.length} result{searchResults.length !== 1 ? 's' : ''} found
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={onClear}
                  className="h-6 px-2 text-[10px]"
                >
                  Clear
                </Button>
              </div>
              <div className="space-y-1">
                {searchResults.map((node) => {
                  const NodeIcon = getNodeIcon(node.type)
                  const connectionCount = getConnectionCount(node.id)
                  return (
                    <button
                      key={node.id}
                      onClick={() => onResultClick(node)}
                      className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg hover:bg-accent transition-all text-left group border border-transparent hover:border-accent/50 hover:shadow-sm"
                    >
                      <NodeIcon className="h-4 w-4 flex-shrink-0" style={{ color: nodeColorByType[node.type] }} />
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-sm truncate text-left group-hover:text-accent-foreground">
                          {node.label || node.id}
                        </div>
                        <div className="text-[10px] text-muted-foreground truncate text-left font-mono">
                          {node.id}
                        </div>
                      </div>
                      <div className="flex items-center gap-1.5 flex-shrink-0">
                        {connectionCount > 0 && (
                          <Badge variant="outline" className="text-[8px] px-1.5 py-0 h-4 flex items-center gap-0.5">
                            <Link2 className="h-2.5 w-2.5" />
                            {connectionCount}
                          </Badge>
                        )}
                        <Badge 
                          variant="secondary" 
                          className="text-[9px] px-1.5 py-0.5 h-4"
                          style={{ backgroundColor: nodeColorByType[node.type] + '20', color: nodeColorByType[node.type] }}
                        >
                          {node.type}
                        </Badge>
                      </div>
                      <ArrowRight className="h-3.5 w-3.5 opacity-0 group-hover:opacity-100 transition-opacity text-muted-foreground" />
                    </button>
                  )
                })}
              </div>
            </div>
          </div>
        )}
        
        {/* Loading State */}
        {showSearchResults && isSearching && (
          <div className="absolute top-full mt-2 w-full bg-background/95 backdrop-blur-md border border-border rounded-lg shadow-xl p-4 text-center text-sm text-muted-foreground z-50">
            <Loader2 className="h-5 w-5 animate-spin mx-auto mb-2" />
            Searching...
          </div>
        )}
        
        {/* No Results Message */}
        {showSearchResults && !isSearching && searchQuery && searchResults.length === 0 && (
          <div className="absolute top-full mt-2 w-full bg-background/95 backdrop-blur-md border border-border rounded-lg shadow-xl p-4 text-center text-sm text-muted-foreground z-50">
            No nodes found matching "{searchQuery}"
          </div>
        )}
      </div>
    </div>
  )
}
