import axios, { AxiosInstance } from 'axios';
import { API_BASE_URL } from '@/lib/config/api';
import type { Neo4jConfig } from '@/lib/types/api';

// ============================================================================
// Types
// ============================================================================

export interface ValidationTargets {
  spearman: number;
  pearson: number;
  kendall: number;
  f1_score: number;
  precision: number;
  recall: number;
  top_5_overlap: number;
  top_10_overlap: number;
  rmse_max: number;
}

export interface CorrelationMetrics {
  spearman: number;
  pearson: number;
  kendall: number;
  spearman_pvalue: number;
  pearson_pvalue: number;
  kendall_pvalue: number;
}

export interface ErrorMetrics {
  rmse: number;
  mae: number;
  max_error: number;
  mean_error: number;
}

export interface ClassificationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  confusion_matrix: {
    tp: number;
    fp: number;
    tn: number;
    fn: number;
  };
}

export interface RankingMetrics {
  top_5_overlap: number;
  top_10_overlap: number;
  top_5_predicted: string[];
  top_5_actual: string[];
  top_5_common: string[];
  ndcg_at_5: number;
  ndcg_at_10: number;
}

export interface ComponentComparison {
  id: string;
  type: string;
  predicted: number;
  actual: number;
  error: number;
  predicted_critical: boolean;
  actual_critical: boolean;
  classification: string; // TP, FP, TN, FN
}

export interface ValidationGroupResult {
  group_name: string;
  sample_size: number;
  passed: boolean;
  metrics: {
    correlation: CorrelationMetrics;
    error: ErrorMetrics;
    classification: ClassificationMetrics;
    ranking: RankingMetrics;
  };
  summary: {
    spearman: number;
    f1: number;
    precision: number;
    recall: number;
    rmse: number;
    top5_overlap: number;
  };
}

export interface ValidationResult {
  timestamp: string;
  layer: string;
  context: string;
  targets: ValidationTargets;
  overall: ValidationGroupResult;
  by_type: Record<string, ValidationGroupResult>;
  predicted_count: number;
  actual_count: number;
  matched_count: number;
  warnings: string[];
  passed: boolean;
}

export interface LayerValidationResult {
  layer: string;
  layer_name: string;
  data: {
    predicted_components: number;
    simulated_components: number;
    matched_components: number;
  };
  summary: {
    passed: boolean;
    spearman: number;
    f1_score: number;
    precision: number;
    recall: number;
    top_5_overlap: number;
    rmse: number;
  };
  validation_result: ValidationResult | null;
  warnings: string[];
}

export interface PipelineResult {
  timestamp: string;
  summary: {
    total_components: number;
    layers_validated: number;
    layers_passed: number;
    all_passed: boolean;
  };
  layers: Record<string, LayerValidationResult>;
  cross_layer_insights: string[];
  targets: ValidationTargets;
}

export interface LayerDefinition {
  name: string;
  description: string;
  component_types: string[];
}

export interface ValidationRequest {
  credentials: Neo4jConfig;
  layers: string[];
  include_comparisons: boolean;
}

export interface QuickValidationRequest {
  credentials: Neo4jConfig;
  predicted_file?: string;
  actual_file?: string;
  predicted_data?: Record<string, number>;
  actual_data?: Record<string, number>;
}

// ============================================================================
// Validation API Client
// ============================================================================

class ValidationAPI {
  private client: AxiosInstance;
  private baseURL: string;
  private credentials?: Neo4jConfig;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
    this.client = axios.create({
      baseURL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 600000, // 10 minutes for long-running validation
    });
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

  /**
   * Run the full validation pipeline
   * 
   * Orchestrates:
   * 1. Graph analysis to get predicted criticality scores
   * 2. Failure simulation to get actual impact scores
   * 3. Statistical validation comparing predictions vs reality
   */
  async runPipeline(
    layers: string[] = ['application', 'infrastructure', 'system'],
    includeComparisons: boolean = true
  ): Promise<PipelineResult> {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }

    const response = await this.client.post('/api/v1/validation/run-pipeline', {
      credentials: this.credentials,
      layers,
      include_comparisons: includeComparisons,
    });

    if (!response.data.success) {
      throw new Error(response.data.detail || 'Validation pipeline failed');
    }

    return response.data.result;
  }

  /**
   * Quick validation from provided data
   * 
   * Compare predicted scores against actual scores using
   * statistical validation metrics without running the full pipeline.
   */
  async quickValidation(
    predictedData?: Record<string, number>,
    actualData?: Record<string, number>,
    predictedFile?: string,
    actualFile?: string
  ): Promise<ValidationResult> {
    if (!this.credentials) {
      throw new Error('No credentials set. Please connect first.');
    }

    const response = await this.client.post('/api/v1/validation/quick', {
      credentials: this.credentials,
      predicted_data: predictedData,
      actual_data: actualData,
      predicted_file: predictedFile,
      actual_file: actualFile,
    });

    if (!response.data.success) {
      throw new Error(response.data.detail || 'Quick validation failed');
    }

    return response.data.result;
  }

  /**
   * Get available validation layers and their definitions
   */
  async getLayers(): Promise<Record<string, LayerDefinition>> {
    const response = await this.client.get('/api/v1/validation/layers');

    if (!response.data.success) {
      throw new Error(response.data.detail || 'Failed to get validation layers');
    }

    return response.data.layers;
  }

  /**
   * Get default validation targets (success criteria)
   */
  async getTargets(): Promise<ValidationTargets> {
    const response = await this.client.get('/api/v1/validation/targets');

    if (!response.data.success) {
      throw new Error(response.data.detail || 'Failed to get validation targets');
    }

    return response.data.targets;
  }
}

// Export singleton instance
export const validationClient = new ValidationAPI();
