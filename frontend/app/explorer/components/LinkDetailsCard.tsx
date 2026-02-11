import React from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { X, ArrowRight, Link2 } from 'lucide-react'
import type { GraphLink, GraphNode, ForceGraphData } from '@/lib/types/api'

interface LinkDetailsCardProps {
  link: GraphLink
  graphData: ForceGraphData | null
  onClose: () => void
  onNodeClick: (node: GraphNode) => void
  getNodeIcon: (type: string) => React.ComponentType<any>
  nodeColorByType: Record<string, string>
  linkColorByType: Record<string, string>
}

export function LinkDetailsCard({
  link,
  graphData,
  onClose,
  onNodeClick,
  getNodeIcon,
  nodeColorByType,
  linkColorByType
}: LinkDetailsCardProps) {
  const sourceId = typeof link.source === 'string' ? link.source : (link.source as any).id
  const targetId = typeof link.target === 'string' ? link.target : (link.target as any).id
  
  const sourceNode = graphData?.nodes.find(n => n.id === sourceId)
  const targetNode = graphData?.nodes.find(n => n.id === targetId)

  return (
    <Card className="border flex flex-col h-full bg-white dark:bg-black">
        <CardHeader className="pb-3 pt-4 px-4 flex-shrink-0 border-b bg-white dark:bg-black">
          <div className="flex items-start justify-between gap-2">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <Link2 className="h-4 w-4 flex-shrink-0" style={{ color: linkColorByType[link.type] }} />
                <CardTitle className="text-sm">Connection</CardTitle>
              </div>
              <div className="text-xs opacity-60">
                {link.type.replace(/_/g, ' ')}
              </div>
            </div>
            <Button 
              variant="ghost" 
              size="icon" 
              onClick={onClose}
              className="h-6 w-6 flex-shrink-0"
            >
              <X className="h-3 w-3" />
            </Button>
          </div>
        </CardHeader>
        <CardContent className="pt-3 pb-3 px-4 space-y-1.5 flex-1 overflow-y-auto">
          <div className="space-y-1.5">
            {/* Source Node */}
            {sourceNode && (
              <div className="p-2 bg-muted/30 rounded border">
                <div className="flex items-center gap-2 mb-1">
                  <div className="text-[10px] opacity-60">FROM</div>
                  {(() => {
                    const SourceIcon = getNodeIcon(sourceNode.type)
                    return (
                      <div className="flex items-center gap-1 text-[10px] opacity-60 ml-auto">
                        <SourceIcon className="h-2.5 w-2.5" style={{ color: nodeColorByType[sourceNode.type] }} />
                        <span>{sourceNode.type}</span>
                      </div>
                    )
                  })()}
                </div>
                <button
                  onClick={() => onNodeClick(sourceNode)}
                  className="text-sm truncate hover:text-primary transition-colors w-full text-left"
                >
                  {sourceNode.label || sourceId}
                </button>
                <div className="text-[10px] opacity-40 font-mono mt-1">
                  {sourceId}
                </div>
              </div>
            )}
            
            {/* Arrow */}
            <div className="flex justify-center">
              <div className="flex items-center gap-1.5 px-2 py-1 text-[10px] opacity-60">
                <ArrowRight className="h-3 w-3" />
                <span>{link.type.replace(/_/g, ' ')}</span>
              </div>
            </div>
            
            {/* Target Node */}
            {targetNode && (
              <div className="p-2 bg-muted/30 rounded border">
                <div className="flex items-center gap-2 mb-1">
                  <div className="text-[10px] opacity-60">TO</div>
                  {(() => {
                    const TargetIcon = getNodeIcon(targetNode.type)
                    return (
                      <div className="flex items-center gap-1 text-[10px] opacity-60 ml-auto">
                        <TargetIcon className="h-2.5 w-2.5" style={{ color: nodeColorByType[targetNode.type] }} />
                        <span>{targetNode.type}</span>
                      </div>
                    )
                  })()}
                </div>
                <button
                  onClick={() => onNodeClick(targetNode)}
                  className="text-sm truncate hover:text-primary transition-colors w-full text-left"
                >
                  {targetNode.label || targetId}
                </button>
                <div className="text-[10px] opacity-40 font-mono mt-1">
                  {targetId}
                </div>
              </div>
            )}
          </div>
          
          {/* Additional Properties */}
          {(() => {
            // Get all properties from link
            const allProperties: Record<string, any> = {
              ...link.properties,
              weight: link.weight,
              criticality: (link as any).criticality,
            };
            
            // Fields to exclude (already shown elsewhere or internal)
            const excludeFields = new Set([
              'source', 'target', 'source_id', 'target_id',
              'type', 'relation_type', 'dependency_type',
              'source_type', 'target_type',
              '__threeObj', '__lineObj', '__arrowObj', '__spriteObj', '__textObj' // Three.js internals
            ]);
            
            // Get all fields that exist and aren't excluded
            const displayFields = Object.entries(allProperties)
              .filter(([key, value]) => {
                if (value === undefined || value === null) return false;
                if (excludeFields.has(key)) return false;
                if (key.startsWith('__')) return false; // Exclude all internal fields starting with '__'
                if (typeof value === 'function') return false;
                if (typeof value === 'object' && value.isObject3D) return false; // Exclude Three.js objects
                return true;
              })
              .sort(([keyA], [keyB]) => {
                // Sort weight and criticality first, then alphabetically
                if (keyA === 'weight') return -1;
                if (keyB === 'weight') return 1;
                if (keyA === 'criticality') return -1;
                if (keyB === 'criticality') return 1;
                return keyA.localeCompare(keyB);
              });
            
            if (displayFields.length === 0) return null;
            
            // Format value for display
            const formatValue = (value: any): string => {
              if (typeof value === 'number') {
                return value % 1 === 0 ? value.toString() : value.toFixed(2);
              }
              if (typeof value === 'boolean') {
                return value ? 'Yes' : 'No';
              }
              if (typeof value === 'object') {
                return JSON.stringify(value);
              }
              return String(value);
            };
            
            // Format key for display
            const formatKey = (key: string): string => {
              return key
                .replace(/_/g, ' ')
                .split(' ')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ');
            };
            
            return (
              <div className="space-y-1.5">
                {displayFields.map(([key, value]) => (
                  <div 
                    key={key} 
                    className="p-2 rounded border bg-muted/30"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[10px] opacity-60">
                        {formatKey(key)}
                      </span>
                      <span className="text-xs text-right break-all">
                        {formatValue(value)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            );
          })()}
        </CardContent>
      </Card>
  )
}
