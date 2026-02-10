import React from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { X } from 'lucide-react'
import type { GraphNode } from '@/lib/types/api'

interface NodeDetailsCardProps {
  node: GraphNode
  onClose: () => void
  getConnectionCount: (nodeId: string) => number
  getNodeIcon: (type: string) => React.ComponentType<any>
  nodeColorByType: Record<string, string>
}

export function NodeDetailsCard({
  node,
  onClose,
  getConnectionCount,
  getNodeIcon,
  nodeColorByType
}: NodeDetailsCardProps) {
  const NodeIcon = getNodeIcon(node.type)
  const connectionCount = getConnectionCount(node.id)

  return (
    <div className="absolute bottom-4 right-4 z-10 w-80 animate-in slide-in-from-right-3 fade-in duration-300">
      <Card className="bg-background/95 backdrop-blur-md border-2 shadow-xl">
        <CardHeader className="pb-3 pt-4 px-4">
          <div className="flex items-start justify-between gap-2">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-2">
                <NodeIcon className="h-4 w-4 flex-shrink-0" style={{ color: nodeColorByType[node.type] }} />
                <CardTitle className="text-base truncate">{node.label}</CardTitle>
              </div>
              <div className="flex items-center gap-2">
                <Badge 
                  variant="secondary" 
                  className="text-[10px] px-2 py-0.5"
                  style={{ backgroundColor: nodeColorByType[node.type] + '20', color: nodeColorByType[node.type] }}
                >
                  {node.type}
                </Badge>
                <Badge variant="outline" className="text-[10px] px-2 py-0.5">
                  {connectionCount} connections
                </Badge>
              </div>
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
          {/* Node Details */}
          {(() => {
            // Get all properties from node - they can be in node.properties or directly on node
            const allProperties: Record<string, any> = {
              ...(node.properties || {}),
              // Also include properties that might be directly on the node object
              ...(node as any),
            };
            
            // Fields to exclude (already shown elsewhere or internal/structural)
            const excludeFields = new Set([
              'id', 'label', 'type', 'degree', 'criticality_score',
              'criticality_level', 'criticality_levels', 'properties',
              'x', 'y', 'z', 'vx', 'vy', 'vz', 'fx', 'fy', 'fz', // Force graph internals
              '__indexColor', 'index' // D3 internals
            ]);
            
            // Get all fields that exist and aren't excluded
            const displayFields = Object.entries(allProperties)
              .filter(([key, value]) => 
                value !== undefined && 
                value !== null &&
                !excludeFields.has(key) &&
                typeof value !== 'function'
              )
              .sort(([keyA], [keyB]) => {
                // Sort weight first, then alphabetically
                if (keyA === 'weight') return -1;
                if (keyB === 'weight') return 1;
                return keyA.localeCompare(keyB);
              });
            
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
                {/* Always show Node ID first */}
                <div className="p-2 bg-muted/30 rounded-md border border-border/50">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[11px] text-muted-foreground font-medium">Node ID</span>
                    <span className="font-mono text-[10px] bg-background px-1.5 py-0.5 rounded border text-right break-all">
                      {node.id}
                    </span>
                  </div>
                </div>
                
                {/* Display all other fields */}
                {displayFields.length > 0 && displayFields.map(([key, value]) => (
                  <div 
                    key={key} 
                    className={`p-2 rounded-md border ${
                      key === 'weight' 
                        ? 'bg-primary/5 border-primary/20' 
                        : 'bg-muted/30 border-border/50'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[11px] font-medium text-muted-foreground">
                        {formatKey(key)}
                      </span>
                      <span className={`text-xs font-semibold text-right break-all max-w-[180px] ${
                        key === 'weight' 
                          ? 'text-primary' 
                          : 'text-foreground'
                      }`} title={formatValue(value)}>
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
