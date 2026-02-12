export type GraphView = 'complete' | 'application' | 'infrastructure' | 'middleware'

export interface GraphViewConfig {
  id: GraphView
  name: string
  description: string
  relationshipTypes: string[]
  dependencyTypes?: string[]
  nodeTypes?: string[]
  color: string
}

export const GRAPH_VIEWS: Record<GraphView, GraphViewConfig> = {
  complete: {
    id: 'complete',
    name: 'Complete System',
    description: 'All system components and their interactions',
    relationshipTypes: ['RUNS_ON', 'PUBLISHES_TO', 'SUBSCRIBES_TO', 'ROUTES', 'CONNECTS_TO', 'USES'],
    nodeTypes: ['Application', 'Node', 'Broker', 'Topic', 'Library'],
    color: '#3b82f6', // blue
  },
  application: {
    id: 'application',
    name: 'Application Layer',
    description: 'Application-to-application dependencies',
    relationshipTypes: ['DEPENDS_ON'],
    dependencyTypes: ['app_to_app'],
    nodeTypes: ['Application'],
    color: '#10b981', // green
  },
  infrastructure: {
    id: 'infrastructure',
    name: 'Infrastructure Layer',
    description: 'Infrastructure node dependencies',
    relationshipTypes: ['DEPENDS_ON'],
    dependencyTypes: ['node_to_node'],
    nodeTypes: ['Node'],
    color: '#f59e0b', // amber
  },
  middleware: {
    id: 'middleware',
    name: 'Middleware Layer',
    description: 'Application and infrastructure to broker dependencies',
    relationshipTypes: ['DEPENDS_ON'],
    dependencyTypes: ['app_to_broker', 'node_to_broker'],
    nodeTypes: ['Application', 'Node', 'Broker'],
    color: '#a855f7', // purple
  },
}
