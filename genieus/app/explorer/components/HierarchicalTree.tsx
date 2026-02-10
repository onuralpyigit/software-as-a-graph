import React, { useRef, useEffect } from 'react'
import { Badge } from '@/components/ui/badge'
import { ChevronDown, ChevronRight, Loader2 } from 'lucide-react'
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
            className={`flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-xs transition-all duration-200 ${
              isLastLevel 
                ? 'bg-gradient-to-r from-primary/15 to-primary/10 font-semibold ring-2 ring-primary/30 shadow-md' 
                : 'bg-gradient-to-r from-muted/60 to-muted/40 font-medium shadow-sm'
            } ${
              isParentSelected 
                ? 'ring-2 ring-primary scale-[1.02]' 
                : 'hover:shadow-md hover:scale-[1.01]'
            }`}
          >
            {children.length > 0 && (
              <span
                onClick={(e) => {
                  e.stopPropagation()
                  toggleNodeExpansion(level.nodeId)
                }}
                className="p-1 hover:bg-accent/80 rounded-md transition-all flex-shrink-0 cursor-pointer hover:scale-110"
              >
                {isExpanded ? (
                  <ChevronDown className="h-3 w-3" />
                ) : (
                  <ChevronRight className="h-3 w-3" />
                )}
              </span>
            )}
            {children.length === 0 && <div className="w-5" />}
            <div className="p-1 rounded-md" style={{ backgroundColor: `${nodeColorByType[level.nodeType]}15` }}>
              <NodeIcon className="h-3.5 w-3.5 flex-shrink-0" style={{ color: nodeColorByType[level.nodeType] }} />
            </div>
            <div className="flex-1 flex flex-col items-start justify-center min-w-0 gap-0.5">
              <span className="font-medium text-left truncate w-full">{level.nodeLabel || level.nodeId}</span>
              {connectionToParent && (
                <Badge 
                  variant="outline" 
                  className="text-[7px] px-1 py-0 h-3.5 font-semibold border"
                  style={{ 
                    borderColor: linkColorByType[connectionToParent] || 'currentColor',
                    backgroundColor: `${linkColorByType[connectionToParent] || 'currentColor'}10`,
                    color: linkColorByType[connectionToParent] || 'currentColor'
                  }}
                >
                  {connectionToParent.replace(/_/g, ' ')}
                </Badge>
              )}
            </div>
            <Badge 
              variant="outline" 
              className="text-[9px] px-2 py-0.5 h-5 font-semibold flex-shrink-0 border-2 shadow-sm"
              style={{ 
                backgroundColor: `${nodeColorByType[level.nodeType]}10`,
                borderColor: `${nodeColorByType[level.nodeType]}40`,
                color: nodeColorByType[level.nodeType]
              }}
            >
              {level.nodeType}
            </Badge>
            <Badge variant="secondary" className="text-[10px] px-1.5 py-0.5 h-5 font-medium">
              {children.length}
            </Badge>
          </button>
        </div>
        
        {/* Show children when expanded */}
        {isExpanded && children.length > 0 && (
          <div className="space-y-1 ml-6 pl-4 border-l-2 border-border/50 relative before:absolute before:left-0 before:top-0 before:bottom-0 before:w-0.5 before:bg-gradient-to-b before:from-primary/20 before:to-transparent">
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
          <div className="text-[10px] text-muted-foreground italic py-3 px-4 ml-6 bg-muted/20 rounded-lg border border-dashed border-muted-foreground/20">
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
    const connectionCount = getConnectionCount(child.id)
    
    return (
      <div className="flex w-full items-center gap-1">
        <button
          onClick={() => onNodeClick(child)}
          className={`flex flex-1 items-center gap-2.5 rounded-lg px-3 py-2 text-xs transition-all duration-200 ${
            selectedNode?.id === child.id
              ? "bg-gradient-to-r from-primary/15 to-primary/10 ring-2 ring-primary/30 shadow-sm scale-[1.02]"
              : "hover:bg-gradient-to-r hover:from-accent/80 hover:to-accent/40 hover:shadow-sm hover:scale-[1.01]"
          }`}
        >
          <div className="p-1 rounded-md" style={{ backgroundColor: `${nodeColorByType[child.type]}15` }}>
            <ChildIcon className="h-3 w-3 flex-shrink-0" style={{ color: nodeColorByType[child.type] }} />
          </div>
          <div className="flex-1 flex flex-col items-start justify-center min-w-0 gap-0.5">
            <span className="font-medium text-left truncate w-full">{child.label || child.id}</span>
            {childConnectionType && (
              <Badge 
                variant="outline" 
                className="text-[7px] px-1 py-0 h-3.5 font-semibold border"
                style={{ 
                  borderColor: linkColorByType[childConnectionType] || 'currentColor',
                  backgroundColor: `${linkColorByType[childConnectionType] || 'currentColor'}10`,
                  color: linkColorByType[childConnectionType] || 'currentColor'
                }}
              >
                {childConnectionType.replace(/_/g, ' ')}
              </Badge>
            )}
          </div>
          <Badge 
            variant="outline" 
            className="text-[9px] px-2 py-0.5 h-5 font-medium flex-shrink-0 border-2"
            style={{ 
              backgroundColor: `${nodeColorByType[child.type]}10`,
              borderColor: `${nodeColorByType[child.type]}40`,
              color: nodeColorByType[child.type]
            }}
          >
            {child.type}
          </Badge>
        </button>
        {connectionCount > 0 && (
          <button
            onClick={(e) => {
              e.stopPropagation()
              onNodeClick(child)
            }}
            className="p-1.5 hover:bg-accent rounded-md transition-all flex-shrink-0 group"
            title={`Expand ${connectionCount} connection${connectionCount === 1 ? '' : 's'}`}
          >
            <ChevronRight className="h-3.5 w-3.5 text-muted-foreground group-hover:text-primary group-hover:scale-110 transition-all" />
          </button>
        )}
      </div>
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
              className={`flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-xs transition-all duration-200 ${
                selectedNode?.id === node.id
                  ? "bg-gradient-to-r from-primary/15 to-primary/10 ring-2 ring-primary/30 shadow-sm scale-[1.02]"
                  : "hover:bg-gradient-to-r hover:from-accent/80 hover:to-accent/40 hover:shadow-sm hover:scale-[1.01]"
              }`}
            >
              <div className="p-1 rounded-md" style={{ backgroundColor: `${nodeColorByType[node.type]}15` }}>
                <NodeIcon className="h-3.5 w-3.5 flex-shrink-0" style={{ color: nodeColorByType[node.type] }} />
              </div>
              <span className="truncate flex-1 text-left font-medium">{node.label || node.id}</span>
              <Badge 
                variant="outline" 
                className="text-[9px] px-2 py-0.5 h-5 font-medium flex-shrink-0 border-2"
                style={{ 
                  backgroundColor: `${nodeColorByType[node.type]}10`,
                  borderColor: `${nodeColorByType[node.type]}40`,
                  color: nodeColorByType[node.type]
                }}
              >
                {node.type}
              </Badge>
            </button>
          )
        })}
      </div>
    )
  }

  return (
    <div className="space-y-1">
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
