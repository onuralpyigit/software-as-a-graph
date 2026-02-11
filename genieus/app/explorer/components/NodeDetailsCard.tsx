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
    <Card className="border flex flex-col h-full bg-white dark:bg-black">
        <CardHeader className="pb-3 pt-4 px-4 flex-shrink-0 border-b bg-white dark:bg-black">
          <div className="flex items-start justify-between gap-2">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <NodeIcon className="h-4 w-4 flex-shrink-0" style={{ color: nodeColorByType[node.type] }} />
                <CardTitle className="text-sm truncate">{node.label}</CardTitle>
              </div>
              <div className="flex items-center gap-2 text-xs">
                <span className="opacity-60">{node.type}</span>
                <span className="opacity-40">•</span>
                <span className="opacity-60">{connectionCount} connections</span>
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
        <CardContent className="pt-3 pb-3 px-4 space-y-1 flex-1 overflow-y-auto">
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
              '__indexColor', 'index', // D3 internals
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
                <div className="p-2 bg-muted/30 rounded border">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-[10px] opacity-60">Node ID</span>
                    <span className="font-mono text-[10px] text-right break-all">
                      {node.id}
                    </span>
                  </div>
                </div>
                
                {/* Display all other fields */}
                {displayFields.length > 0 && displayFields.map(([key, value]) => (
                  <div 
                    key={key} 
                    className="p-2 rounded border bg-muted/30"
                  >
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-[10px] opacity-60">
                        {formatKey(key)}
                      </span>
                      <span className="text-xs text-right break-all max-w-[180px]" title={formatValue(value)}>
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
