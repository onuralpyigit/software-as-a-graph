import axios, { AxiosInstance } from 'axios';
import { API_BASE_URL } from '@/lib/config/api';
import type {
  Neo4jConfig,
  HealthCheckResponse,
  GraphStatsResponse,
  ComprehensiveAnalysisResponse,
  AnalysisRequest,
  QualityAnalysisResponse,
  ForceGraphData,
  ClassificationRequest,
  ClassificationResponse,
} from '@/lib/types/api';

class GraphAnalysisAPI {
  private client: AxiosInstance;
  private baseURL: string;
  private onConnectionError?: () => void;
  private credentials?: Neo4jConfig;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
    this.client = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 600000, // 10 minutes for long-running analyses
    });

    // Add response interceptor to detect connection errors
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        // Check if it's a network error or server unavailable
        if (!error.response || error.code === 'ECONNABORTED' || error.code === 'ERR_NETWORK') {
          // Notify connection error callback
          if (this.onConnectionError) {
            this.onConnectionError();
          }
        }
        return Promise.reject(error);
      }
    );
  }

  // Set callback for connection errors
  setConnectionErrorHandler(handler: () => void) {
    this.onConnectionError = handler;
  }

  // Set credentials for subsequent requests
  setCredentials(config: Neo4jConfig) {
    this.credentials = config;
  }

  // Get current credentials
  getCredentials(): Neo4jConfig | undefined {
    return this.credentials;
  }

  // Clear stored credentials
  clearCredentials() {
    this.credentials = undefined;
  }

  // Connection & Health (using new FastAPI endpoints)
  async connect(config: Neo4jConfig) {
    // Test connection using the new /api/v1/connect endpoint
    const response = await this.client.post('/api/v1/connect', config);
    
    if (response.data.success) {
      // Store credentials for future requests
      this.setCredentials(config);
    }
    
    return response.data;
  }

  async healthCheck(): Promise<HealthCheckResponse> {
    const response = await this.client.get<HealthCheckResponse>('/health');
    return response.data;
  }

  // Graph Statistics (using new FastAPI endpoints)
  async getGraphStats(): Promise<GraphStatsResponse> {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }
    
    const response = await this.client.post('/api/v1/stats', this.credentials);
    const data = response.data;
    
    // Return data from new API format
    return {
      total_nodes: data.stats.total_nodes || 0,
      total_edges: data.stats.total_edges || 0,
      total_structural_edges: data.stats.total_structural_edges || 0,
      density: 0, // Calculate if needed
      node_counts: data.stats.node_counts || {},
      edge_counts: data.stats.edge_counts || {},
      structural_edge_counts: data.stats.structural_edge_counts || {},
    };
  }

  // Analysis Endpoints (using new FastAPI endpoints)
  async analyzeComprehensive(request: AnalysisRequest): Promise<ComprehensiveAnalysisResponse> {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }
    
    // Use the new full analysis endpoint
    const response = await this.client.post('/api/v1/analysis/full', this.credentials);
    const data = response.data;
    
    if (!data.success || !data.analysis) {
      throw new Error('Analysis failed');
    }
    
    const analysis = data.analysis;
    
    // Transform to match old format
    const problems = analysis.problems || [];
    
    // Calculate scores from components
    const components = analysis.components || [];
    const reliabilityScores = components.map(c => c.scores.reliability * 100);
    const maintainabilityScores = components.map(c => c.scores.maintainability * 100);
    const availabilityScores = components.map(c => c.scores.availability * 100);
    
    const avgReliability = reliabilityScores.length > 0 
      ? reliabilityScores.reduce((a, b) => a + b, 0) / reliabilityScores.length 
      : 0;
    const avgMaintainability = maintainabilityScores.length > 0
      ? maintainabilityScores.reduce((a, b) => a + b, 0) / maintainabilityScores.length
      : 0;
    const avgAvailability = availabilityScores.length > 0
      ? availabilityScores.reduce((a, b) => a + b, 0) / availabilityScores.length
      : 0;
    
    const overallScore = (avgReliability + avgMaintainability + avgAvailability) / 3;
    
    // Transform problems to findings format
    const transformProblems = (category: string) => {
      return problems
        .filter(p => {
          const cat = p.category.toLowerCase();
          return cat === category.toLowerCase() || cat === 'architecture';
        })
        .map(p => ({
          severity: p.severity.toLowerCase(),
          category: p.category,
          component: p.entity_id,
          description: p.description,
          impact: '',
          recommendation: p.recommendation,
          metrics: {}
        }));
    };
    
    // Transform components to critical components format
    const transformCritical = (attr: string, scores: number[]) => {
      return components
        .filter(c => {
          // Check dimension-specific criticality if available, otherwise fall back to overall
          if (c.criticality_levels) {
            const level = c.criticality_levels[attr as keyof typeof c.criticality_levels];
            return level === 'critical' || level === 'high';
          }
          return c.criticality_level === 'critical' || c.criticality_level === 'high';
        })
        .map(c => ({
          id: c.id,
          type: c.type,
          score: c.scores[attr as keyof typeof c.scores] * 100,
          quality_attribute: attr,
          reasons: [
            c.criticality_levels 
              ? `${c.criticality_levels[attr as keyof typeof c.criticality_levels]} criticality for ${attr}`
              : `${c.criticality_level} criticality component`
          ],
          metrics: c.scores
        }))
        .slice(0, 10);
    };
    
    return {
      overall_score: overallScore,
      timestamp: new Date().toISOString(),
      graph_stats: await this.getGraphStats(),
      reliability: request.analyze_reliability !== false ? {
        quality_attribute: 'reliability',
        score: avgReliability,
        findings_count: transformProblems('reliability').length,
        critical_count: transformCritical('reliability', reliabilityScores).length,
        findings: transformProblems('reliability'),
        critical_components: transformCritical('reliability', reliabilityScores),
        metrics: {},
        recommendations: [],
        timestamp: new Date().toISOString()
      } : undefined,
      maintainability: request.analyze_maintainability !== false ? {
        quality_attribute: 'maintainability',
        score: avgMaintainability,
        findings_count: transformProblems('maintainability').length,
        critical_count: transformCritical('maintainability', maintainabilityScores).length,
        findings: transformProblems('maintainability'),
        critical_components: transformCritical('maintainability', maintainabilityScores),
        metrics: {},
        recommendations: [],
        timestamp: new Date().toISOString()
      } : undefined,
      availability: request.analyze_availability !== false ? {
        quality_attribute: 'availability',
        score: avgAvailability,
        findings_count: transformProblems('availability').length,
        critical_count: transformCritical('availability', availabilityScores).length,
        findings: transformProblems('availability'),
        critical_components: transformCritical('availability', availabilityScores),
        metrics: {},
        recommendations: [],
        timestamp: new Date().toISOString()
      } : undefined
    };
  }

  async analyzeReliability(params?: {
    use_weights?: boolean;
    weight_property?: string;
    dependency_types?: string[];
  }): Promise<QualityAnalysisResponse> {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }
    
    const response = await this.client.post('/api/v1/analysis/full', this.credentials);
    const data = response.data;
    
    if (!data.success || !data.analysis) {
      throw new Error('Reliability analysis failed');
    }
    
    const analysis = data.analysis;
    const components = analysis.components || [];
    const problems = analysis.problems || [];
    
    const reliabilityScores = components.map(c => c.scores.reliability * 100);
    const avgScore = reliabilityScores.length > 0
      ? reliabilityScores.reduce((a, b) => a + b, 0) / reliabilityScores.length
      : 0;
    
    const findings = problems
      .filter(p => {
        const cat = p.category.toLowerCase();
        return cat === 'reliability' || cat === 'architecture';
      })
      .map(p => ({
        severity: p.severity.toLowerCase(),
        category: p.category,
        component: p.entity_id,
        description: p.description,
        impact: '',
        recommendation: p.recommendation,
        metrics: {}
      }));
    
    const critical = components
      .filter(c => {
        // Check reliability-specific criticality if available
        if (c.criticality_levels) {
          const level = c.criticality_levels.reliability;
          return level === 'critical' || level === 'high';
        }
        return c.criticality_level === 'critical' || c.criticality_level === 'high';
      })
      .map(c => ({
        id: c.id,
        type: c.type,
        score: c.scores.reliability * 100,
        quality_attribute: 'reliability',
        reasons: [
          c.criticality_levels
            ? `${c.criticality_levels.reliability} criticality for reliability`
            : `${c.criticality_level} criticality`
        ],
        metrics: c.scores
      }))
      .slice(0, 10);
    
    return {
      quality_attribute: 'reliability',
      score: avgScore,
      findings_count: findings.length,
      critical_count: critical.length,
      findings,
      critical_components: critical,
      metrics: {},
      recommendations: [],
      timestamp: new Date().toISOString()
    };
  }

  async analyzeMaintainability(params?: {
    use_weights?: boolean;
    weight_property?: string;
    dependency_types?: string[];
  }): Promise<QualityAnalysisResponse> {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }
    
    const response = await this.client.post('/api/v1/analysis/full', this.credentials);
    const data = response.data;
    
    if (!data.success || !data.analysis) {
      throw new Error('Maintainability analysis failed');
    }
    
    const analysis = data.analysis;
    const components = analysis.components || [];
    const problems = analysis.problems || [];
    
    const scores = components.map(c => c.scores.maintainability * 100);
    const avgScore = scores.length > 0
      ? scores.reduce((a, b) => a + b, 0) / scores.length
      : 0;
    
    const findings = problems
      .filter(p => {
        const cat = p.category.toLowerCase();
        return cat === 'maintainability' || cat === 'architecture';
      })
      .map(p => ({
        severity: p.severity.toLowerCase(),
        category: p.category,
        component: p.entity_id,
        description: p.description,
        impact: '',
        recommendation: p.recommendation,
        metrics: {}
      }));
    
    const critical = components
      .filter(c => {
        // Check maintainability-specific criticality if available
        if (c.criticality_levels) {
          const level = c.criticality_levels.maintainability;
          return level === 'critical' || level === 'high';
        }
        return c.criticality_level === 'critical' || c.criticality_level === 'high';
      })
      .map(c => ({
        id: c.id,
        type: c.type,
        score: c.scores.maintainability * 100,
        quality_attribute: 'maintainability',
        reasons: [
          c.criticality_levels
            ? `${c.criticality_levels.maintainability} criticality for maintainability`
            : `${c.criticality_level} criticality`
        ],
        metrics: c.scores
      }))
      .slice(0, 10);
    
    return {
      quality_attribute: 'maintainability',
      score: avgScore,
      findings_count: findings.length,
      critical_count: critical.length,
      findings,
      critical_components: critical,
      metrics: {},
      recommendations: [],
      timestamp: new Date().toISOString()
    };
  }

  async analyzeAvailability(params?: {
    use_weights?: boolean;
    weight_property?: string;
    dependency_types?: string[];
  }): Promise<QualityAnalysisResponse> {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }
    
    const response = await this.client.post('/api/v1/analysis/full', this.credentials);
    const data = response.data;
    
    if (!data.success || !data.analysis) {
      throw new Error('Availability analysis failed');
    }
    
    const analysis = data.analysis;
    const components = analysis.components || [];
    const problems = analysis.problems || [];
    
    const scores = components.map(c => c.scores.availability * 100);
    const avgScore = scores.length > 0
      ? scores.reduce((a, b) => a + b, 0) / scores.length
      : 0;
    
    const findings = problems
      .filter(p => {
        const cat = p.category.toLowerCase();
        return cat === 'availability' || cat === 'architecture';
      })
      .map(p => ({
        severity: p.severity.toLowerCase(),
        category: p.category,
        component: p.entity_id,
        description: p.description,
        impact: '',
        recommendation: p.recommendation,
        metrics: {}
      }));
    
    const critical = components
      .filter(c => {
        // Check availability-specific criticality if available
        if (c.criticality_levels) {
          const level = c.criticality_levels.availability;
          return level === 'critical' || level === 'high';
        }
        return c.criticality_level === 'critical' || c.criticality_level === 'high';
      })
      .map(c => ({
        id: c.id,
        type: c.type,
        score: c.scores.availability * 100,
        quality_attribute: 'availability',
        reasons: [
          c.criticality_levels
            ? `${c.criticality_levels.availability} criticality for availability`
            : `${c.criticality_level} criticality`
        ],
        metrics: c.scores
      }))
      .slice(0, 10);
    
    return {
      quality_attribute: 'availability',
      score: avgScore,
      findings_count: findings.length,
      critical_count: critical.length,
      findings,
      critical_components: critical,
      metrics: {},
      recommendations: [],
      timestamp: new Date().toISOString()
    };
  }

  // Classification endpoint
  async classify(request: ClassificationRequest): Promise<ClassificationResponse> {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }
    
    // Build query string for POST request with query parameters
    const params = new URLSearchParams();
    
    if (request.metrics && request.metrics.length > 0) {
      request.metrics.forEach(m => params.append('metrics', m));
    }
    if (request.use_fuzzy !== undefined) {
      params.append('use_fuzzy', String(request.use_fuzzy));
    }
    if (request.use_weights !== undefined) {
      params.append('use_weights', String(request.use_weights));
    }
    if (request.dependency_types && request.dependency_types.length > 0) {
      request.dependency_types.forEach(dt => params.append('dependency_types', dt));
    }
    
    const response = await this.client.post(`/api/v1/classify?${params.toString()}`, this.credentials);
    
    const data = response.data;
    
    if (!data.success) {
      throw new Error('Classification failed');
    }
    
    // Transform the response to match our type structure
    const classifications: Record<string, SingleClassificationResponse> = {};
    
    for (const [metricName, classification] of Object.entries(data.classifications as any)) {
      const metricData = classification as any;
      classifications[metricName] = {
        metric_name: metricName,
        statistics: {
          min_val: metricData.statistics.min_val,
          q1: metricData.statistics.q1,
          median: metricData.statistics.median,
          q3: metricData.statistics.q3,
          max_val: metricData.statistics.max_val,
          iqr: metricData.statistics.iqr,
          upper_fence: metricData.statistics.upper_fence,
          k_factor: request.k_factor || 1.5
        },
        distribution: metricData.distribution,
        items: metricData.components.map((comp: any) => ({
          id: comp.id,
          level: comp.level,
          score: comp.score,
          percentile: 0, // Not provided by API
          z_score: 0, // Not provided by API
        }))
      };
    }
    
    return {
      classifications,
      timestamp: new Date().toISOString(),
      merged_ranking: data.merged_ranking || []
    };
  }

  // New Analysis Endpoints for component-based analysis
  async analyzeFullSystem(): Promise<any> {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }
    
    const response = await this.client.post('/api/v1/analysis/full', this.credentials);
    return response.data;
  }

  async analyzeByType(componentType: string): Promise<any> {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }
    
    const response = await this.client.post(`/api/v1/analysis/type/${componentType}`, this.credentials);
    return response.data;
  }

  async analyzeByLayer(layer: string): Promise<any> {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }
    
    const response = await this.client.post(`/api/v1/analysis/layer/${layer}`, this.credentials);
    return response.data;
  }

  // Graph Data for Visualization (using export and components endpoints)
  async getGraphData(params?: {
    relationship_types?: string[];
    dependency_types?: string[];
    node_types?: string[];
    include_metrics?: boolean;
    include_criticality?: boolean;
    limit_nodes?: number;
  }): Promise<ForceGraphData> {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }
    
    const response = await this.client.post('/api/v1/graph/export', this.credentials);
    const data = response.data;
    
    if (!data.success) {
      throw new Error('Failed to get graph data');
    }
    
    const components = data.components || [];
    const edges = data.edges || [];
    
    // Transform to ForceGraphData format
    const nodes = components.map((c: any) => ({
      id: c.id,
      label: c.name || c.id, // Use name as label, fallback to id if name doesn't exist
      type: c.component_type || c.type || 'Unknown',
      properties: c,
      degree: 0, // Will be calculated from links
      criticality_score: c.weight || 0,
    }));
    
    const links = edges.map((e: any) => ({
      source: e.source_id || e.source,
      target: e.target_id || e.target,
      type: e.relation_type || e.type || 'DEPENDS_ON',
      weight: e.weight || 1,
      properties: e,
    }));
    
    // Calculate node degrees
    const degreeMap = new Map<string, number>();
    links.forEach(link => {
      degreeMap.set(link.source, (degreeMap.get(link.source) || 0) + 1);
      degreeMap.set(link.target, (degreeMap.get(link.target) || 0) + 1);
    });
    nodes.forEach(node => {
      node.degree = degreeMap.get(node.id) || 0;
    });
    
    // Apply filters if specified
    let filteredNodes = nodes;
    let filteredLinks = links;
    
    if (params?.node_types && params.node_types.length > 0) {
      filteredNodes = nodes.filter(n => params.node_types!.includes(n.type));
      const nodeIds = new Set(filteredNodes.map(n => n.id));
      filteredLinks = links.filter(l => nodeIds.has(String(l.source)) && nodeIds.has(String(l.target)));
    }
    
    // Filter by relationship_types (e.g., DEPENDS_ON, RUNS_ON, PUBLISHES_TO)
    if (params?.relationship_types && params.relationship_types.length > 0) {
      filteredLinks = filteredLinks.filter(l => params.relationship_types!.includes(l.type));
    }
    
    // Filter by dependency_types (e.g., app_to_app, node_to_node) - checks the properties.dependency_type field
    if (params?.dependency_types && params.dependency_types.length > 0) {
      filteredLinks = filteredLinks.filter(l => {
        const depType = l.properties?.dependency_type;
        return depType && params.dependency_types!.includes(depType);
      });
    }
    
    if (params?.limit_nodes && filteredNodes.length > params.limit_nodes) {
      filteredNodes = filteredNodes.slice(0, params.limit_nodes);
      const nodeIds = new Set(filteredNodes.map(n => n.id));
      filteredLinks = filteredLinks.filter(l => nodeIds.has(String(l.source)) && nodeIds.has(String(l.target)));
    }
    
    return {
      nodes: filteredNodes,
      links: filteredLinks,
      metadata: {
        total_nodes: filteredNodes.length,
        total_links: filteredLinks.length,
      }
    };
  }

  async getLimitedGraphData(params?: {
    node_limit?: number;
    edge_limit?: number;
    fetch_structural?: boolean;
    relationship_types?: string[];
    dependency_types?: string[];
    node_types?: string[];
  }): Promise<ForceGraphData> {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }
    
    console.log('🔍 [getLimitedGraphData] Request params:', params);
    
    // Use the new limited export endpoint with Neo4j LIMIT
    const queryParams = new URLSearchParams();
    if (params?.node_limit !== undefined && params.node_limit !== null) {
      queryParams.append('node_limit', params.node_limit.toString());
    }
    if (params?.edge_limit !== undefined && params.edge_limit !== null && params.edge_limit > 0) {
      queryParams.append('edge_limit', params.edge_limit.toString());
    }
    if (params?.fetch_structural !== undefined) {
      queryParams.append('fetch_structural', params.fetch_structural.toString());
    }
    if (params?.node_types && params.node_types.length > 0) {
      params.node_types.forEach(type => {
        queryParams.append('node_types', type);
      });
    }
    
    const url = `/api/v1/graph/export-limited${queryParams.toString() ? '?' + queryParams.toString() : ''}`;
    console.log('🌐 [getLimitedGraphData] Request URL:', url);
    console.log('🔧 [getLimitedGraphData] Query params object:', Object.fromEntries(queryParams));
    
    const response = await this.client.post(url, this.credentials);
    const data = response.data;
    
    console.log('📦 [getLimitedGraphData] Raw response:', {
      success: data.success,
      componentCount: data.components?.length || 0,
      edgeCount: data.edges?.length || 0,
      metadata: data.metadata,
      stats: data.stats
    });
    
    if (!data.success) {
      throw new Error('Failed to get limited graph data');
    }
    
    const components = data.components || [];
    const edges = data.edges || [];
    
    console.log('📊 [getLimitedGraphData] Raw edges sample:', edges.slice(0, 5));
    
    // Transform to ForceGraphData format
    const nodes = components.map((c: any) => ({
      id: c.id,
      label: c.name || c.id,
      type: c.component_type || c.type || 'Unknown',
      properties: c,
      degree: 0,
      criticality_score: c.weight || 0,
    }));
    
    const links = edges.map((e: any) => ({
      source: e.source_id || e.source,
      target: e.target_id || e.target,
      type: e.relation_type || e.type || 'DEPENDS_ON',
      weight: e.weight || 1,
      properties: e,
    }));
    
    console.log('🔗 [getLimitedGraphData] Transformed links sample:', links.slice(0, 5));
    console.log('🔗 [getLimitedGraphData] All link types:', [...new Set(links.map(l => l.type))]);
    
    // Calculate node degrees
    const degreeMap = new Map<string, number>();
    links.forEach(link => {
      degreeMap.set(link.source, (degreeMap.get(link.source) || 0) + 1);
      degreeMap.set(link.target, (degreeMap.get(link.target) || 0) + 1);
    });
    nodes.forEach(node => {
      node.degree = degreeMap.get(node.id) || 0;
    });
    
    // Apply filters if specified
    let filteredNodes = nodes;
    let filteredLinks = links;
    
    console.log('🔧 [getLimitedGraphData] Starting filtering with params:', {
      node_types: params?.node_types,
      relationship_types: params?.relationship_types,
      dependency_types: params?.dependency_types,
      initial_links: links.length,
      initial_nodes: nodes.length
    });
    
    if (params?.node_types && params.node_types.length > 0) {
      filteredNodes = nodes.filter(n => params.node_types!.includes(n.type));
      const nodeIds = new Set(filteredNodes.map(n => n.id));
      filteredLinks = links.filter(l => nodeIds.has(String(l.source)) && nodeIds.has(String(l.target)));
      console.log('🎯 [getLimitedGraphData] After node_types filter:', {
        node_types: params.node_types,
        filteredNodes: filteredNodes.length,
        filteredLinks: filteredLinks.length
      });
    }
    
    // Skip relationship_types filter - backend already returns the correct type via fetch_structural
    // The fetch_structural parameter handles whether we get structural or derived relationships
    if (params?.relationship_types && params.relationship_types.length > 0) {
      console.log('⏭️  [getLimitedGraphData] Skipping relationship_types filter (handled by backend via fetch_structural)');
    }
    
    if (params?.dependency_types && params.dependency_types.length > 0) {
      const beforeCount = filteredLinks.length;
      filteredLinks = filteredLinks.filter(l => {
        const depType = l.properties?.dependency_type;
        return depType && params.dependency_types!.includes(depType);
      });
      console.log('🎯 [getLimitedGraphData] After dependency_types filter:', {
        dependency_types: params.dependency_types,
        before: beforeCount,
        after: filteredLinks.length,
        removed: beforeCount - filteredLinks.length
      });
    }
    
    const result = {
      nodes: filteredNodes,
      links: filteredLinks,
      metadata: {
        total_nodes: filteredNodes.length,
        total_links: filteredLinks.length,
        limited: true,
        node_limit: params?.node_limit || 1000,
        ...data.metadata
      }
    };
    
    console.log('✅ [getLimitedGraphData] Final result:', {
      nodes: result.nodes.length,
      links: result.links.length,
      linkTypes: [...new Set(result.links.map(l => l.type))],
      metadata: result.metadata
    });
    
    return result;
  }

  async getNodeSubgraph(
    nodeId: string,
    params?: {
      depth?: number;
      include_metrics?: boolean;
      direction?: 'incoming' | 'outgoing' | 'both';
    }
  ): Promise<ForceGraphData> {
    // Get full graph and filter for subgraph
    const fullGraph = await this.getGraphData({
      include_metrics: params?.include_metrics,
    });
    
    // Simple subgraph extraction (BFS)
    const depth = params?.depth || 1;
    const visited = new Set<string>([nodeId]);
    const queue: Array<{id: string, level: number}> = [{id: nodeId, level: 0}];
    
    while (queue.length > 0) {
      const current = queue.shift()!;
      if (current.level >= depth) continue;
      
      fullGraph.links.forEach(link => {
        let nextId: string | null = null;
        
        if (params?.direction === 'incoming' && link.target === current.id) {
          nextId = link.source;
        } else if (params?.direction === 'outgoing' && link.source === current.id) {
          nextId = link.target;
        } else if (params?.direction !== 'incoming' && params?.direction !== 'outgoing') {
          if (link.source === current.id) nextId = link.target;
          else if (link.target === current.id) nextId = link.source;
        }
        
        if (nextId && !visited.has(nextId)) {
          visited.add(nextId);
          queue.push({id: nextId, level: current.level + 1});
        }
      });
    }
    
    const nodes = fullGraph.nodes.filter(n => visited.has(n.id));
    const nodeIds = new Set(nodes.map(n => n.id));
    const links = fullGraph.links.filter(l => 
      nodeIds.has(l.source) && nodeIds.has(l.target)
    );
    
    return {
      nodes,
      links,
      metadata: {
        center_node: nodeId,
        depth,
        total_nodes: nodes.length,
        total_links: links.length,
      }
    };
  }

  async getNodeConnections(
    nodeId: string,
    fetch_structural: boolean = false
  ): Promise<ForceGraphData> {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }

    const queryParams = new URLSearchParams();
    queryParams.append('node_id', nodeId);
    queryParams.append('fetch_structural', fetch_structural.toString());

    const url = `/api/v1/graph/node-connections?${queryParams.toString()}`;
    const response = await this.client.post(url, this.credentials);
    const data = response.data;

    if (!data.success) {
      throw new Error('Failed to fetch node connections');
    }

    const components = data.components || [];
    const edges = data.edges || [];

    // Transform to ForceGraphData format
    const nodes = components.map((c: any) => ({
      id: c.id,
      label: c.name || c.id,
      type: c.type || 'Unknown',
      properties: c,
      degree: 0,
      criticality_score: c.weight || 0,
    }));

    const links = edges.map((e: any) => ({
      source: e.source,
      target: e.target,
      type: e.relation_type || 'DEPENDS_ON',
      weight: e.weight || 1,
      properties: e,
    }));

    // Calculate node degrees
    const degreeMap = new Map<string, number>();
    links.forEach((link: any) => {
      degreeMap.set(link.source, (degreeMap.get(link.source) || 0) + 1);
      degreeMap.set(link.target, (degreeMap.get(link.target) || 0) + 1);
    });
    nodes.forEach((node: any) => {
      node.degree = degreeMap.get(node.id) || 0;
    });

    return {
      nodes,
      links,
      metadata: {
        node_id: nodeId,
        total_nodes: nodes.length,
        total_links: links.length,
      }
    };
  }

  async getNodeConnectionsWithDepth(
    nodeId: string,
    fetch_structural: boolean = false,
    depth: number = 1
  ): Promise<ForceGraphData> {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }

    const queryParams = new URLSearchParams();
    queryParams.append('node_id', nodeId);
    queryParams.append('fetch_structural', fetch_structural.toString());
    queryParams.append('depth', depth.toString());

    const url = `/api/v1/graph/node-connections?${queryParams.toString()}`;
    const response = await this.client.post(url, this.credentials);
    const data = response.data;

    if (!data.success) {
      throw new Error('Failed to fetch node connections');
    }

    const components = data.components || [];
    const edges = data.edges || [];

    // Transform to ForceGraphData format
    const nodes = components.map((c: any) => ({
      id: c.id,
      label: c.name || c.id,
      type: c.type || 'Unknown',
      properties: c,
      degree: 0,
      criticality_score: c.weight || 0,
    }));

    const links = edges.map((e: any) => ({
      source: e.source,
      target: e.target,
      type: e.relation_type || 'DEPENDS_ON',
      weight: e.weight || 1,
      properties: e,
    }));

    // Calculate node degrees
    const degreeMap = new Map<string, number>();
    links.forEach((link: any) => {
      degreeMap.set(link.source, (degreeMap.get(link.source) || 0) + 1);
      degreeMap.set(link.target, (degreeMap.get(link.target) || 0) + 1);
    });
    nodes.forEach((node: any) => {
      node.degree = degreeMap.get(node.id) || 0;
    });

    return {
      nodes,
      links,
      metadata: {
        node_id: nodeId,
        depth,
        total_nodes: nodes.length,
        total_links: links.length,
      }
    };
  }

  async searchNodes(query: string, limit: number = 20): Promise<Array<{id: string, type: string, label: string, weight: number}>> {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }

    const queryParams = new URLSearchParams();
    queryParams.append('query', query);
    queryParams.append('limit', limit.toString());

    const url = `/api/v1/graph/search-nodes?${queryParams.toString()}`;
    const response = await this.client.get(url, {
      params: this.credentials
    });
    const data = response.data;

    if (!data.success) {
      throw new Error('Failed to search nodes');
    }

    return data.nodes || [];
  }

  // Graph Generation & Import (using new FastAPI endpoints)
  async importFromFile(file: File, params?: {
    batch_size?: number;
    clear_first?: boolean;
    derive_dependencies?: boolean;
  }) {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }
    
    // The new API doesn't have file upload, so we need to read and parse the file
    const text = await file.text();
    const graphData = JSON.parse(text);
    
    // Use the import endpoint
    const response = await this.client.post('/api/v1/graph/import', {
      credentials: this.credentials,
      graph_data: graphData,
      clear_database: params?.clear_first || false
    });
    
    return response.data;
  }

  async importGraph(graphData: any, params?: {
    clear_first?: boolean;
  }) {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }
    
    // Direct import of graph data
    const response = await this.client.post('/api/v1/graph/import', {
      credentials: this.credentials,
      graph_data: graphData,
      clear_database: params?.clear_first || false
    });
    
    return response.data;
  }

  async generateAndImport(request: {
    scale: string;
    scenario?: string;
    num_nodes?: number;
    num_applications?: number;
    num_topics?: number;
    num_brokers?: number;
    antipatterns?: string[];
    seed?: number;
    batch_size?: number;
    clear_first?: boolean;
    derive_dependencies?: boolean;
  }) {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }
    
    // Use the new generate-and-import endpoint
    const response = await this.client.post('/api/v1/graph/generate-and-import', this.credentials, {
      params: {
        scale: request.scale,
        seed: request.seed || 42,
        clear_database: request.clear_first || false
      }
    });
    
    return response.data;
  }

  async generateGraphFile(request: {
    scale: string;
    scenario?: string;
    num_nodes?: number;
    num_applications?: number;
    num_topics?: number;
    num_brokers?: number;
    antipatterns?: string[];
    seed?: number;
  }): Promise<Blob> {
    // Build request body with only defined values
    const requestBody: any = {
      scale: request.scale,
      scenario: request.scenario || 'generic',
      seed: request.seed || 42,
      antipatterns: request.antipatterns || []
    };

    // Only include optional numeric fields if they are defined
    if (request.num_nodes !== undefined) requestBody.num_nodes = request.num_nodes;
    if (request.num_applications !== undefined) requestBody.num_applications = request.num_applications;
    if (request.num_topics !== undefined) requestBody.num_topics = request.num_topics;
    if (request.num_brokers !== undefined) requestBody.num_brokers = request.num_brokers;

    // Use the generate-file endpoint that doesn't require credentials
    const response = await this.client.post('/api/v1/graph/generate-file', requestBody);
    
    // Extract graph_data from the response
    const graphData = response.data.graph_data;
    
    // Convert to blob for download
    const blob = new Blob([JSON.stringify(graphData, null, 2)], {
      type: 'application/json'
    });
    
    // Return the blob directly
    return blob;
  }

  async clearDatabase() {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }
    
    // Use POST instead of DELETE to avoid CORS issues
    const response = await this.client.post('/api/v1/graph/clear', this.credentials);
    return response.data;
  }

  async exportNeo4jData(): Promise<Blob> {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }
    
    // Call the export endpoint that returns data in input file format
    const response = await this.client.post('/api/v1/graph/export-neo4j-data', this.credentials);
    
    // Extract graph_data from the response
    const graphData = response.data.graph_data;
    
    // Convert to blob for download
    const blob = new Blob([JSON.stringify(graphData, null, 2)], {
      type: 'application/json'
    });
    
    return blob;
  }

  // Statistics Endpoints
  async getDegreeDistributionStats() {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }
    
    const response = await this.client.post('/api/v1/stats/degree-distribution', this.credentials);
    return response.data;
  }

  async getConnectivityDensityStats(nodeType?: string) {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }
    
    const requestBody = {
      ...this.credentials,
      node_type: nodeType
    };
    
    const response = await this.client.post('/api/v1/stats/connectivity-density', requestBody);
    return response.data;
  }

  async getClusteringCoefficientStats(nodeType?: string) {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }
    
    const requestBody = {
      ...this.credentials,
      node_type: nodeType
    };
    
    const response = await this.client.post('/api/v1/stats/clustering-coefficient', requestBody);
    return response.data;
  }

  async getDependencyDepthStats() {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }
    
    const response = await this.client.post('/api/v1/stats/dependency-depth', this.credentials);
    return response.data;
  }

  async getComponentIsolationStats() {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }
    
    const response = await this.client.post('/api/v1/stats/component-isolation', this.credentials);
    return response.data;
  }

  async getComponentRedundancyStats() {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }
    
    const response = await this.client.post('/api/v1/stats/component-redundancy', this.credentials);
    return response.data;
  }

  async getMessageFlowPatternsStats() {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }
    
    const response = await this.client.post('/api/v1/stats/message-flow-patterns', this.credentials);
    return response.data;
  }

  async getNodeWeightDistributionStats() {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }
    
    const response = await this.client.post('/api/v1/stats/node-weight-distribution', this.credentials);
    return response.data;
  }

  async getEdgeWeightDistributionStats() {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }
    
    const response = await this.client.post('/api/v1/stats/edge-weight-distribution', this.credentials);
    return response.data;
  }

  // Topology
  async getTopologyData(nodeId?: string, nodeLimit: number = 1000): Promise<any> {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }

    const params: any = { node_limit: nodeLimit };
    if (nodeId) {
      params.node_id = nodeId;
    }

    const response = await this.client.post('/api/v1/graph/topology', this.credentials, { params });
    return response.data;
  }

  // Update base URL
  setBaseURL(url: string) {
    this.baseURL = url;
    this.client.defaults.baseURL = url;
  }
}

// Export singleton instance
export const apiClient = new GraphAnalysisAPI();

// Export class for custom instances
export default GraphAnalysisAPI;
