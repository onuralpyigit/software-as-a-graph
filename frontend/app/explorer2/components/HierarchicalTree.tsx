import React, { useRef, useEffect } from 'react'
import { Badge } from '@/components/ui/badge'
import { ChevronDown, ChevronRight, Loader2, ArrowRight, ArrowLeft } from 'lucide-react'
import type { GraphNode, GraphLink, ForceGraphData } from '@/lib/types/api'

interface HierarchicalTreeProps {
  graphData: ForceGraphData | null
  drillDownHistory: Array<{nodeId: string, nodeType: string, nodeLabel?: string, childNodes?: GraphNode[]}>
  hierarchyData: Map<string, {node: GraphNode, children: GraphNode[], links: GraphLink[]}>
  expandedNodes: Set<string>
  selectedNode: GraphNode | null
  isLoadingChildren: boolean
  onNodeClick: (node: GraphNode) => void
  onReset: () => void
  toggleNodeExpansion: (nodeId: string) => void
  getConnectionCount: (nodeId: string) => number
  getConnectionType: (parentId: string, childId: string) => string | null
  getConnectionTypeAndDirection: (parentId: string, childId: string) => { type: string, direction: 'outgoing' | 'incoming' } | null
  getNodeIcon: (type: string) => React.ComponentType<any>
  nodeColorByType: Record<string, string>
  linkColorByType: Record<string, string>
}

export function HierarchicalTree({
  graphData,
  drillDownHistory,
  hierarchyData,
  expandedNodes,
  selectedNode,
  isLoadingChildren,
  onNodeClick,
  onReset,
  toggleNodeExpansion,
  getConnectionCount,
  getConnectionType,
  getConnectionTypeAndDirection,
  getNodeIcon,
  nodeColorByType,
  linkColorByType
}: HierarchicalTreeProps) {
  const lastNodeRef = useRef<HTMLDivElement>(null)
  
  // Scroll to the newest drilled node only when a new drill happens
  useEffect(() => {
    if (drillDownHistory.length > 0 && lastNodeRef.current) {
      setTimeout(() => {
        lastNodeRef.current?.scrollIntoView({ 
          behavior: 'smooth', 
          block: 'center'
        })
      }, 150)
    }
  }, [drillDownHistory.length])
  
  // Render a hierarchical level with its children
  const renderLevel = (levelIndex: number): React.ReactNode => {
    if (levelIndex >= drillDownHistory.length) return null
    
    const level = drillDownHistory[levelIndex]
    const isLastLevel = levelIndex === drillDownHistory.length - 1
    const levelData = hierarchyData.get(level.nodeId)
    const children = levelData?.children || level.childNodes || []
    const nextLevelNodeId = !isLastLevel ? drillDownHistory[levelIndex + 1]?.nodeId : null
    const isExpanded = expandedNodes.has(level.nodeId)
    const NodeIcon = getNodeIcon(level.nodeType)
    
    const parentNodeId = levelIndex > 0 ? drillDownHistory[levelIndex - 1]?.nodeId : null
    const connectionToParent = parentNodeId ? getConnectionType(parentNodeId, level.nodeId) : null
    const connectionToParentWithDir = parentNodeId ? getConnectionTypeAndDirection(parentNodeId, level.nodeId) : null
    
    const parentNode = levelData?.node || graphData?.nodes.find(n => n.id === level.nodeId)
    const isParentSelected = selectedNode?.id === level.nodeId
    
    return (
      <div className="space-y-1">
        {/* Level Parent Node */}
        <div 
          className="relative"
          ref={isLastLevel ? lastNodeRef : null}
        >
          <button
            onClick={() => {
              if (parentNode) {
                onNodeClick(parentNode)
              }
            }}
            className={`flex w-full items-center gap-2 rounded px-2 h-9 text-xs transition-colors ${
              isLastLevel 
                ? 'bg-primary/10 font-semibold border border-primary/30' 
                : 'bg-muted/50 font-medium border border-transparent'
            } ${
              isParentSelected 
                ? 'border-primary' 
                : 'hover:bg-accent'
            }`}
          >
            {children.length > 0 && (
              <span
                onClick={(e) => {
                  e.stopPropagation()
                  toggleNodeExpansion(level.nodeId)
                }}
                className="hover:bg-background/50 rounded flex-shrink-0 cursor-pointer"
              >
                {isExpanded ? (
                  <ChevronDown className="h-3 w-3" />
                ) : (
                  <ChevronRight className="h-3 w-3" />
                )}
              </span>
            )}
            {children.length === 0 && <div className="w-3" />}
            <NodeIcon className="h-3.5 w-3.5 flex-shrink-0" style={{ color: nodeColorByType[level.nodeType] }} />
            <div className="flex-1 min-w-0">
              <div className="truncate text-left">{level.nodeLabel || level.nodeId}</div>
              {connectionToParentWithDir && (
                <div className="text-[10px] opacity-50 truncate text-left">
                  {connectionToParentWithDir.direction === 'outgoing' ? '→' : '←'} {connectionToParentWithDir.type.replace(/_/g, ' ')}
                </div>
              )}
            </div>
            <span className="text-[10px] opacity-60 flex-shrink-0">{level.nodeType}</span>
            {children.length > 0 && (
              <span className="text-[10px] opacity-40 flex-shrink-0">{children.length}</span>
            )}
          </button>
        </div>
        
        {/* Show children when expanded */}
        {isExpanded && children.length > 0 && (
          <div className="space-y-1 ml-4 pl-3 border-l border-border">
            {children.map(child => {
              const isNextLevel = child.id === nextLevelNodeId
              
              if (isNextLevel) {
                // Continue drill-down path
                return (
                  <div key={child.id} className="-ml-6">
                    {renderLevel(levelIndex + 1)}
                  </div>
                )
              }
              
              // Regular child node
              return <TreeChildNode key={child.id} child={child} parentId={level.nodeId} />
            })}
          </div>
        )}
        
        {children.length === 0 && isExpanded && (
          <div className="text-[10px] text-muted-foreground italic py-2 px-3 ml-4">
            {isLoadingChildren ? (
              <div className="flex items-center gap-2">
                <Loader2 className="h-3 w-3 animate-spin" />
                <span>Loading children...</span>
              </div>
            ) : (
              'No connected nodes'
            )}
          </div>
        )}
      </div>
    )
  }

  // Render a child node (not in drill path)
  const TreeChildNode = ({ child, parentId }: { child: GraphNode, parentId: string }) => {
    const ChildIcon = getNodeIcon(child.type)
    const childConnectionType = getConnectionType(parentId, child.id)
    const childConnectionWithDir = getConnectionTypeAndDirection(parentId, child.id)
    const connectionCount = getConnectionCount(child.id)
    
    return (
      <button
        onClick={() => onNodeClick(child)}
        className={`flex w-full items-center gap-2 rounded px-2 h-9 text-xs transition-colors ${
          selectedNode?.id === child.id
            ? "bg-primary/10 border border-primary/30"
            : "hover:bg-accent border border-transparent"
        }`}
      >
        <ChildIcon className="h-3 w-3 flex-shrink-0" style={{ color: nodeColorByType[child.type] }} />
        <div className="flex-1 min-w-0">
          <div className="truncate text-left">{child.label || child.id}</div>
          {childConnectionWithDir && (
            <div className="text-[10px] opacity-50 truncate text-left">
              {childConnectionWithDir.direction === 'outgoing' ? '→' : '←'} {childConnectionWithDir.type.replace(/_/g, ' ')}
            </div>
          )}
        </div>
        <span className="text-[10px] opacity-60 flex-shrink-0">{child.type}</span>
        {connectionCount > 0 && (
          <ChevronRight className="h-3 w-3 opacity-40 flex-shrink-0" />
        )}
      </button>
    )
  }

  // Render top-level nodes (no drill-down active)
  const renderTopLevelNodes = () => {
    const topLevelNodes = graphData?.nodes || []
    
    return (
      <div className="space-y-1">
        {topLevelNodes.map(node => {
          const NodeIcon = getNodeIcon(node.type)
          
          return (
            <button
              key={node.id}
              onClick={() => onNodeClick(node)}
              className={`flex w-full items-center gap-2 rounded px-2 h-9 text-xs transition-colors ${
                selectedNode?.id === node.id
                  ? "bg-primary/10 border border-primary/30"
                  : "hover:bg-accent border border-transparent"
              }`}
            >
              <NodeIcon className="h-3.5 w-3.5 flex-shrink-0" style={{ color: nodeColorByType[node.type] }} />
              <span className="truncate flex-1 text-left">{node.label || node.id}</span>
              <span className="text-[10px] opacity-60 flex-shrink-0" style={{
                  color: nodeColorByType[node.type]
                }}
              >
                {node.type}
              </span>
            </button>
          )
        })}
      </div>
    )
  }

  return (
    <div className="space-y-1 px-1">
      {drillDownHistory.length === 0 ? (
        // Show all top-level nodes when not drilled down
        renderTopLevelNodes()
      ) : (
        // Show only the drill-down path when drilled down
        renderLevel(0)
      )}
    </div>
  )
}
