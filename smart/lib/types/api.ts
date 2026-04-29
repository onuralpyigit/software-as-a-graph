// API Response Types based on api.py

export interface Neo4jConfig {
  uri: string;
  user: string;
  password: string;
  database: string;
}

export interface HealthCheckResponse {
  status: "healthy" | "degraded";
  neo4j_connected: boolean;
  timestamp: string;
  version?: string;
  message?: string;
}

export interface GraphStatsResponse {
  total_nodes: number;
  total_edges: number;
  total_structural_edges?: number;
  total_dependency_edges?: number;
  density: number;
  node_counts: Record<string, number>;
  edge_counts: Record<string, number>;
  structural_edge_counts?: Record<string, number>;
  weight_statistics?: Record<string, any>;
  weight_by_dependency_type?: Record<string, any>;
}

export interface FindingResponse {
  severity: string;
  category: string;
  component: string;
  description: string;
  impact: string;
  recommendation: string;
  metrics: Record<string, any>;
}

export interface CriticalComponentResponse {
  id: string;
  type: string;
  score: number;
  quality_attribute: string;
  reasons: string[];
  metrics: Record<string, any>;
}

export interface QualityAnalysisResponse {
  quality_attribute: string;
  score: number;
  findings_count: number;
  critical_count: number;
  findings: FindingResponse[];
  critical_components: CriticalComponentResponse[];
  metrics: Record<string, any>;
  recommendations: string[];
  timestamp: string;
}

export interface ComprehensiveAnalysisResponse {
  overall_score: number;
  timestamp: string;
  graph_stats: GraphStatsResponse;
  reliability?: QualityAnalysisResponse;
  maintainability?: QualityAnalysisResponse;
  availability?: QualityAnalysisResponse;
  vulnerability?: QualityAnalysisResponse;
}

export interface AnalysisRequest {
  analyze_reliability?: boolean;
  analyze_maintainability?: boolean;
  analyze_availability?: boolean;
  analyze_vulnerability?: boolean;
  dependency_types?: string[];
  use_weights?: boolean;
  weight_property?: string;
}

// Graph Visualization Types
export interface GraphNode {
  id: string;
  label: string;
  type: string;
  properties: Record<string, any>;
  degree?: number;
  betweenness?: number;
  pagerank?: number;
  criticality_score?: number;
  criticality_level?: string;
  criticality_levels?: {
    reliability: string;
    maintainability: string;
    availability: string;
    vulnerability: string;
    overall: string;
  };
}

export interface GraphLink {
  source: string;
  target: string;
  type: string;
  weight?: number;
  properties: Record<string, any>;
  criticality?: number;
}

export interface ForceGraphData {
  nodes: GraphNode[];
  links: GraphLink[];
  metadata: Record<string, any>;
}

export interface GraphDataRequest {
  dependency_types?: string[];
  node_types?: string[];
  include_metrics?: boolean;
  include_criticality?: boolean;
  limit_nodes?: number;
}

// Classification Types
export interface ClassificationItemResponse {
  id: string;
  type: string;
  level: string;
  score: number;
  percentile: number;
  z_score: number;
  fuzzy_membership?: Record<string, number>;
}

export interface ClassificationStatsResponse {
  min_val: number;
  q1: number;
  median: number;
  q3: number;
  max_val: number;
  iqr: number;
  upper_fence: number;
  k_factor: number;
}

export interface SingleClassificationResponse {
  metric_name: string;
  statistics: ClassificationStatsResponse;
  distribution: Record<string, number>;
  items: ClassificationItemResponse[];
}

export interface ClassificationRequest {
  metrics?: string[];
  k_factor?: number;
  use_fuzzy?: boolean;
  fuzzy_width?: number;
  dependency_types?: string[];
  use_weights?: boolean;
  weight_property?: string;
}

export interface MergedClassificationItem {
  id: string;
  merged_score: number;
  dominant_level: string;
  scores_by_metric: Record<string, number>;
}

export interface ClassificationResponse {
  classifications: Record<string, SingleClassificationResponse>;
  timestamp: string;
  merged_ranking?: MergedClassificationItem[];
}
