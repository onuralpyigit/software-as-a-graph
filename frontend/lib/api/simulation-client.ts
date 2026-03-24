import axios, { AxiosInstance } from 'axios';
import { API_BASE_URL } from '@/lib/config/api';
import { apiClient } from './client';

// ============================================================================
// Types
// ============================================================================

interface EventMetrics {
  messages_published: number;
  messages_delivered: number;
  messages_dropped: number;
  delivery_rate_percent: number;
  drop_rate_percent: number;
  avg_latency_ms: number;
  min_latency_ms: number;
  p50_latency_ms: number;
  p99_latency_ms: number;
  max_latency_ms: number;
  throughput_per_sec: number;
}

interface EventResult {
  source_app: string;
  scenario: string;
  duration_sec: number;
  metrics: EventMetrics;
  affected_topics: string[];
  brokers_used: string[];
  reached_subscribers: string[];
  drop_reasons: Record<string, number>;
  component_impacts: Record<string, number>;
}

interface FailureImpact {
  composite_impact: number;
  reachability: {
    initial_paths: number;
    remaining_paths: number;
    loss_percent: number;
  };
  fragmentation: {
    fragmentation_percent: number;
  };
  throughput: {
    loss_percent: number;
  };
  flow_disruption: {
    loss_percent: number;
  };
  cascade: {
    count: number;
    depth: number;
    by_type: Record<string, number>;
  };
  affected: {
    topics: number;
    publishers: number;
    subscribers: number;
  };
}

interface FailureResult {
  target_id: string;
  target_type: string;
  scenario: string;
  impact: FailureImpact;
  cascaded_failures: string[];
  layer_impacts: Record<string, number>;
}

interface LayerMetrics {
  layer: string;
  event_metrics: {
    delivery_rate_percent: number;
    avg_latency_ms: number;
    throughput: number;
  };
  failure_metrics: {
    avg_reachability_loss_percent: number;
    max_impact: number;
  };
  criticality: {
    critical: number;
    high: number;
    medium: number;
    total_components: number;
    spof_count: number;
  };
}

interface SimulationReport {
  timestamp: string;
  graph_summary: Record<string, any>;
  layer_metrics: Record<string, LayerMetrics>;
  top_critical: Array<{
    id: string;
    type: string;
    level: string;
    scores: { combined_impact: number; event_impact: number; failure_impact: number };
    metrics: { cascade_count: number; message_throughput: number; reachability_loss_percent: number };
  }>;
  recommendations: string[];
}

interface EventSimulationRequest {
  source_app: string;
  num_messages?: number;
  duration?: number;
}

interface FailureSimulationRequest {
  target_id: string;
  layer?: string;
  cascade_probability?: number;
}

interface ExhaustiveSimulationRequest {
  layer?: string;
  cascade_probability?: number;
}

interface ReportRequest {
  layers?: string[];
}

// ============================================================================
// Simulation API Client
// ============================================================================

class SimulationAPI {
  private client: AxiosInstance;

  constructor(baseURL: string = API_BASE_URL) {
    this.client = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 600000, // 10 minutes for long-running simulations
    });
  }

  /**
   * Get credentials from the main API client
   */
  private getCredentials() {
    const credentials = apiClient.getCredentials();
    if (!credentials) {
      throw new Error('No Neo4j credentials set. Please connect to the database first.');
    }
    return { credentials };
  }

  /**
   * Run event simulation from a source application
   */
  async runEventSimulation(request: EventSimulationRequest): Promise<EventResult> {
    const credentials = this.getCredentials();
    
    const response = await this.client.post('/api/v1/simulation/event', {
      ...credentials,
      source_app: request.source_app,
      num_messages: request.num_messages || 100,
      duration: request.duration || 10.0,
    });

    if (!response.data.success) {
      throw new Error(response.data.message || 'Event simulation failed');
    }

    return response.data.result;
  }

  /**
   * Run failure simulation for a target component
   */
  async runFailureSimulation(request: FailureSimulationRequest): Promise<FailureResult> {
    const credentials = this.getCredentials();
    
    const response = await this.client.post('/api/v1/simulation/failure', {
      ...credentials,
      target_id: request.target_id,
      layer: request.layer || 'system',
      cascade_probability: request.cascade_probability !== undefined ? request.cascade_probability : 1.0,
    });

    if (!response.data.success) {
      throw new Error(response.data.message || 'Failure simulation failed');
    }

    return response.data.result;
  }

  /**
   * Run exhaustive failure analysis for all components in a layer
   */
  async runExhaustiveSimulation(
    request: ExhaustiveSimulationRequest
  ): Promise<{ results: FailureResult[]; summary: any }> {
    const credentials = this.getCredentials();
    
    const response = await this.client.post('/api/v1/simulation/exhaustive', {
      ...credentials,
      layer: request.layer || 'system',
      cascade_probability: request.cascade_probability !== undefined ? request.cascade_probability : 1.0,
    });

    if (!response.data.success) {
      throw new Error(response.data.message || 'Exhaustive simulation failed');
    }

    return {
      results: response.data.results,
      summary: response.data.summary,
    };
  }

  /**
   * Generate comprehensive simulation report
   */
  async generateReport(request: ReportRequest): Promise<SimulationReport> {
    const credentials = this.getCredentials();
    
    const response = await this.client.post('/api/v1/simulation/report', {
      ...credentials,
      layers: request.layers || ['application', 'infrastructure', 'system'],
    });

    if (!response.data.success) {
      throw new Error(response.data.message || 'Report generation failed');
    }

    return response.data.report;
  }
}

// Export singleton instance
export const simulationClient = new SimulationAPI();

// Export types
export type {
  EventMetrics,
  EventResult,
  FailureImpact,
  FailureResult,
  LayerMetrics,
  SimulationReport,
  EventSimulationRequest,
  FailureSimulationRequest,
  ExhaustiveSimulationRequest,
  ReportRequest,
};
