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
    <div className="absolute bottom-4 right-4 z-10 w-80 animate-in slide-in-from-right-3 fade-in duration-300">
      <Card className="bg-background/95 backdrop-blur-md border-2 shadow-xl">
        <CardHeader className="pb-3 pt-4 px-4">
          <div className="flex items-start justify-between gap-2">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-2">
                <Link2 className="h-4 w-4 flex-shrink-0" style={{ color: linkColorByType[link.type] }} />
                <CardTitle className="text-base">Connection</CardTitle>
              </div>
              <Badge 
                variant="secondary" 
                className="text-[10px] px-2 py-0.5"
                style={{ backgroundColor: linkColorByType[link.type] + '20', color: linkColorByType[link.type] }}
              >
                {link.type.replace(/_/g, ' ')}
              </Badge>
            </div>
            <Button 
              variant="ghost" 
              size="icon" 
              onClick={onClose}
              className="h-7 w-7 flex-shrink-0 hover:bg-destructive/10"
            >
              <X className="h-3.5 w-3.5" />
            </Button>
          </div>
        </CardHeader>
        <CardContent className="pt-0 pb-3 px-4 space-y-2">
          <div className="space-y-1.5">
            {/* Source Node */}
            {sourceNode && (
              <div className="p-2 bg-muted/30 rounded-md border border-border/50">
                <div className="flex items-center gap-2 mb-1">
                  <div className="text-[10px] text-muted-foreground font-medium">FROM</div>
                  {(() => {
                    const SourceIcon = getNodeIcon(sourceNode.type)
                    return (
                      <Badge 
                        variant="outline" 
                        className="text-[8px] px-1.5 py-0 h-3.5 ml-auto"
                        style={{ backgroundColor: nodeColorByType[sourceNode.type] + '15', borderColor: nodeColorByType[sourceNode.type] + '40' }}
                      >
                        <SourceIcon className="h-2.5 w-2.5 mr-1" style={{ color: nodeColorByType[sourceNode.type] }} />
                        {sourceNode.type}
                      </Badge>
                    )
                  })()}
                </div>
                <button
                  onClick={() => onNodeClick(sourceNode)}
                  className="text-sm font-semibold truncate hover:text-primary transition-colors w-full text-left"
                >
                  {sourceNode.label || sourceId}
                </button>
                <div className="text-[10px] text-muted-foreground font-mono mt-1">
                  {sourceId}
                </div>
              </div>
            )}
            
            {/* Arrow */}
            <div className="flex justify-center">
              <div className="flex items-center gap-2 px-3 py-1 bg-primary/10 rounded-full border border-primary/20">
                <ArrowRight className="h-3.5 w-3.5 text-primary" />
                <span className="text-[10px] font-medium text-primary">
                  {link.type.replace(/_/g, ' ')}
                </span>
              </div>
            </div>
            
            {/* Target Node */}
            {targetNode && (
              <div className="p-2 bg-muted/30 rounded-md border border-border/50">
                <div className="flex items-center gap-2 mb-1">
                  <div className="text-[10px] text-muted-foreground font-medium">TO</div>
                  {(() => {
                    const TargetIcon = getNodeIcon(targetNode.type)
                    return (
                      <Badge 
                        variant="outline" 
                        className="text-[8px] px-1.5 py-0 h-3.5 ml-auto"
                        style={{ backgroundColor: nodeColorByType[targetNode.type] + '15', borderColor: nodeColorByType[targetNode.type] + '40' }}
                      >
                        <TargetIcon className="h-2.5 w-2.5 mr-1" style={{ color: nodeColorByType[targetNode.type] }} />
                        {targetNode.type}
                      </Badge>
                    )
                  })()}
                </div>
                <button
                  onClick={() => onNodeClick(targetNode)}
                  className="text-sm font-semibold truncate hover:text-primary transition-colors w-full text-left"
                >
                  {targetNode.label || targetId}
                </button>
                <div className="text-[10px] text-muted-foreground font-mono mt-1">
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
              'source_type', 'target_type'
            ]);
            
            // Get all fields that exist and aren't excluded
            const displayFields = Object.entries(allProperties)
              .filter(([key, value]) => 
                value !== undefined && 
                value !== null && 
                !excludeFields.has(key)
              )
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
                    className={`p-2 rounded-md border ${
                      key === 'weight' 
                        ? 'bg-primary/5 border-primary/20' 
                        : key === 'criticality'
                        ? 'bg-destructive/5 border-destructive/20'
                        : 'bg-muted/30 border-border/50'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className={`text-[11px] font-medium ${
                        key === 'weight' 
                          ? 'text-muted-foreground' 
                          : key === 'criticality'
                          ? 'text-destructive' 
                          : 'text-muted-foreground'
                      }`}>
                        {formatKey(key)}
                      </span>
                      <span className={`text-xs font-bold text-right break-all ${
                        key === 'weight' 
                          ? 'text-primary' 
                          : key === 'criticality'
                          ? 'text-destructive'
                          : 'text-foreground'
                      }`}>
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
    </div>
  )
}
