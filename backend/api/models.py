"""
Pydantic models for API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from enum import Enum


from src.adapters import config


class Neo4jCredentials(BaseModel):
    uri: str = Field(default_factory=config.get_default_uri, description="Neo4j connection URI")
    user: str = Field(default_factory=config.get_default_username, description="Neo4j username")
    password: str = Field(default_factory=config.get_default_password, description="Neo4j password")
    database: str = Field(default_factory=config.get_default_database, description="Neo4j database name")
    node_type: Optional[str] = Field(default=None, description="Filter by specific node type")

    def __init__(self, **data):
        super().__init__(**data)
        # Rewrite localhost URIs when running inside Docker
        object.__setattr__(self, 'uri', config.resolve_neo4j_uri(self.uri))


class GraphRequestWithCredentials(BaseModel):
    credentials: Neo4jCredentials = Field(..., description="Neo4j connection credentials")


class GenerateGraphRequest(GraphRequestWithCredentials):
    scale: str = Field(default="medium", description="Graph scale: tiny, small, medium, large, xlarge")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    domain: Optional[str] = Field(default=None, description="Domain dataset (e.g. e-commerce)")
    scenario: Optional[str] = Field(default=None, description="Topic QoS Scenario (e.g. transactions)")


class GenerateGraphFileRequest(BaseModel):
    """Request for generating a graph file without database credentials"""
    scale: str = Field(default="medium", description="Graph scale: tiny, small, medium, large, xlarge")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    domain: Optional[str] = Field(default=None, description="Domain dataset (e.g. e-commerce)")
    scenario: Optional[str] = Field(default=None, description="Topic QoS Scenario (e.g. transactions)")


class ImportGraphRequest(GraphRequestWithCredentials):
    graph_data: Dict[str, Any] = Field(..., description="Graph data structure to import")
    clear_database: bool = Field(default=False, description="Clear database before import")


class CriticalityLevelsModel(BaseModel):
    reliability: str
    maintainability: str
    availability: str
    vulnerability: str
    overall: str


class ScoresModel(BaseModel):
    reliability: float
    maintainability: float
    availability: float
    vulnerability: float
    overall: float


class ComponentResponse(BaseModel):
    id: str
    name: str
    type: str
    is_critical: bool
    rmav_score: float
    criticality_level: str
    criticality_levels: CriticalityLevelsModel
    scores: ScoresModel


class EdgeResponse(BaseModel):
    source: str
    target: str
    source_name: str
    target_name: str
    type: str
    criticality_level: str
    scores: ScoresModel


class ProblemResponse(BaseModel):
    entity_id: str
    type: str
    category: str
    severity: str
    name: str
    description: str
    recommendation: str


class AnalysisSummaryModel(BaseModel):
    total_components: int
    critical_count: int
    high_count: int
    total_problems: int
    critical_problems: int
    components: Dict[str, int]
    edges: Dict[str, int]


class AnalysisStatsModel(BaseModel):
    nodes: int
    edges: int
    density: float
    avg_degree: float


class AnalysisDetailModel(BaseModel):
    context: str
    description: str
    summary: AnalysisSummaryModel
    stats: AnalysisStatsModel
    components: List[ComponentResponse]
    edges: List[EdgeResponse]
    problems: List[ProblemResponse]


class AnalysisEnvelope(BaseModel):
    success: bool
    layer: str
    component_type: Optional[str] = None
    analysis: AnalysisDetailModel


# ── Simulation Response Models ──────────────────────────────────────────

class EventMetricsModel(BaseModel):
    messages_published: int
    messages_delivered: int
    messages_dropped: int
    delivery_rate_percent: float
    drop_rate_percent: float
    avg_latency_ms: float
    min_latency_ms: float
    p50_latency_ms: float
    p99_latency_ms: float
    max_latency_ms: float
    throughput_per_sec: float


class EventSimulationResultModel(BaseModel):
    source_app: str
    scenario: str
    duration_sec: float
    metrics: EventMetricsModel
    affected_topics: List[str]
    reached_subscribers: List[str]
    brokers_used: List[str]
    component_impacts: Dict[str, float]
    failed_components: List[str]
    drop_reasons: Dict[str, int]
    related_components: List[str]


class EventSimulationResponse(BaseModel):
    success: bool
    simulation_type: str
    result: EventSimulationResultModel


class ImpactDetail(BaseModel):
    initial_paths: Optional[int] = None
    remaining_paths: Optional[int] = None
    loss_percent: float


class FragmentationDetail(BaseModel):
    fragmentation_percent: float


class CascadeDetail(BaseModel):
    count: int
    depth: int
    by_type: Dict[str, int] = Field(default_factory=dict)


class ReliabilityImpactModel(BaseModel):
    cascade_reach: float
    weighted_cascade_impact: float
    normalized_cascade_depth: float
    reliability_impact: float


class MaintainabilityImpactModel(BaseModel):
    change_reach: float
    weighted_change_impact: float
    normalized_change_depth: float
    maintainability_impact: float


class AvailabilityImpactModel(BaseModel):
    weighted_reachability_loss: float
    weighted_fragmentation: float
    path_breaking_throughput_loss: float
    availability_impact: float
    ia_out: float
    ia_in: float


class VulnerabilityImpactModel(BaseModel):
    attack_reach: float
    weighted_attack_impact: float
    high_value_contamination: float
    vulnerability_impact: float


class AffectedDetail(BaseModel):
    topics: int
    publishers: int
    subscribers: int


class FailureImpactModel(BaseModel):
    reachability: ImpactDetail
    fragmentation: FragmentationDetail
    throughput: ImpactDetail
    flow_disruption: ImpactDetail
    cascade: CascadeDetail
    affected: AffectedDetail
    composite_impact: float
    reliability: ReliabilityImpactModel
    maintainability: MaintainabilityImpactModel
    availability: AvailabilityImpactModel
    vulnerability: VulnerabilityImpactModel


class CascadeSequenceEvent(BaseModel):
    id: str
    type: str
    cause: str
    depth: int


class FailureSimulationResultModel(BaseModel):
    target_id: str
    target_type: str
    scenario: str
    impact: FailureImpactModel
    cascaded_failures: List[str]
    cascade_sequence: List[CascadeSequenceEvent]
    layer_impacts: Dict[str, float]


class FailureSimulationResponse(BaseModel):
    success: bool
    simulation_type: str
    result: FailureSimulationResultModel


class ExhaustiveSummaryModel(BaseModel):
    total_components: int
    avg_impact: float
    max_impact: float
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    spof_count: int


class ExhaustiveSimulationResponse(BaseModel):
    success: bool
    simulation_type: str
    layer: str
    summary: ExhaustiveSummaryModel
    results: List[FailureSimulationResultModel]


class LayerEventMetricsModel(BaseModel):
    delivery_rate_percent: float
    avg_latency_ms: float
    throughput: float = 0.0


class LayerFailureMetricsModel(BaseModel):
    avg_reachability_loss_percent: float
    max_impact: float


class LayerCriticalitySummaryModel(BaseModel):
    critical: int
    high: int
    total_components: int = 0
    medium: int = 0
    spof_count: int = 0


class LayerMetricsResponseModel(BaseModel):
    layer: str
    event_metrics: LayerEventMetricsModel
    failure_metrics: LayerFailureMetricsModel
    criticality: LayerCriticalitySummaryModel


class TopCriticalComponentModel(BaseModel):
    id: str
    type: str
    level: str
    scores: Dict[str, float]
    metrics: Dict[str, Any]


class SimulationReportResponseModel(BaseModel):
    timestamp: str
    layer_metrics: Dict[str, LayerMetricsResponseModel]
    top_critical: List[TopCriticalComponentModel]
    graph_summary: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)


class SimulationReportResponse(BaseModel):
    success: bool
    report: SimulationReportResponseModel


class GraphStatsResponse(BaseModel):
    success: bool
    stats: Dict[str, Any]


# ── Statistics Response Models ──────────────────────────────────────────

class DistributionMetricsModel(BaseModel):
    mean: float
    median: float
    std: float
    min: float
    max: float


class DegreeDistributionModel(BaseModel):
    in_degree: DistributionMetricsModel
    out_degree: DistributionMetricsModel
    total_degree: DistributionMetricsModel
    hub_nodes: List[Dict[str, Any]]
    isolated_nodes: int
    total_nodes: int
    hub_threshold: float


class DegreeDistributionResponse(BaseModel):
    success: bool
    stats: DegreeDistributionModel
    computation_time_ms: float = 0.0


class DensityStatsModel(BaseModel):
    density: float
    total_nodes: int
    total_edges: int
    max_possible_edges: int
    interpretation: str
    category: str
    most_dense_components: List[Dict[str, Any]]


class ConnectivityDensityResponse(BaseModel):
    success: bool
    stats: DensityStatsModel
    computation_time_ms: float = 0.0


class ClusteringStatsModel(BaseModel):
    avg_clustering_coefficient: float
    global_clustering: float
    average_clustering: float
    max_coefficient: float
    median_coefficient: float
    min_coefficient: float
    std_coefficient: float
    high_clustering_count: int
    medium_clustering_count: int
    low_clustering_count: int
    zero_clustering_count: int
    total_nodes: int
    high_clustering_nodes: List[Dict[str, Any]]
    zero_clustering_nodes: List[Dict[str, Any]]


class ClusteringCoefficientResponse(BaseModel):
    success: bool
    stats: ClusteringStatsModel
    computation_time_ms: float = 0.0


class DepthStatsModel(BaseModel):
    max_depth: int
    avg_depth: float
    median_depth: float
    min_depth: int
    std_depth: float
    interpretation: str
    category: str
    depth_distribution: Dict[str, int]
    shallow_count: int
    low_depth_count: int
    medium_depth_count: int
    high_depth_count: int
    total_nodes: int
    root_nodes: List[Dict[str, Any]]
    leaf_nodes: List[Dict[str, Any]]
    deepest_components: List[Dict[str, Any]]


class DependencyDepthResponse(BaseModel):
    success: bool
    stats: DepthStatsModel
    computation_time_ms: float = 0.0


class IsolationStatsModel(BaseModel):
    isolated_count: int
    isolated_percentage: float
    source_count: int
    source_percentage: float
    sink_count: int
    sink_percentage: float
    bidirectional_count: int
    bidirectional_percentage: float
    category: str
    interpretation: str
    health: str
    top_sources: List[Dict[str, Any]]
    top_sinks: List[Dict[str, Any]]
    isolated_components: List[Dict[str, Any]]
    total_nodes: int


class ComponentIsolationResponse(BaseModel):
    success: bool
    stats: IsolationStatsModel
    computation_time_ms: float = 0.0


class MessageFlowStatsModel(BaseModel):
    total_topics: int
    total_brokers: int
    total_applications: int
    active_applications: int
    avg_publishers_per_topic: float
    avg_subscribers_per_topic: float
    avg_topics_per_broker: float
    interpretation: str
    category: str
    health: str
    hot_topics: List[Dict[str, Any]]
    broker_utilization: List[Dict[str, Any]]
    isolated_applications: List[Dict[str, Any]]
    top_publishers: List[Dict[str, Any]]
    top_subscribers: List[Dict[str, Any]]


class MessageFlowPatternsResponse(BaseModel):
    success: bool
    stats: MessageFlowStatsModel
    computation_time_ms: float = 0.0


class RedundancyStatsModel(BaseModel):
    total_components: int
    spof_count: int
    spof_percentage: float
    redundant_count: int
    redundancy_percentage: float
    resilience_score: float
    interpretation: str
    category: str
    health: str
    single_points_of_failure: List[Dict[str, Any]]
    bridge_components: List[Dict[str, Any]]


class ComponentRedundancyResponse(BaseModel):
    success: bool
    stats: RedundancyStatsModel
    computation_time_ms: float = 0.0


class WeightDistributionModel(BaseModel):
    total_components: Optional[int] = None
    total_edges: Optional[int] = None
    total_weight: float
    avg_weight: float
    median_weight: float
    min_weight: float
    max_weight: float
    std_weight: float
    weight_concentration: float
    interpretation: str
    category: str
    health: str
    very_high_count: int
    high_count: int
    medium_count: int
    low_count: int
    very_low_count: int
    top_components: Optional[List[Dict[str, Any]]] = None
    top_edges: Optional[List[Dict[str, Any]]] = None
    type_stats: Dict[str, Any]


class NodeWeightDistributionResponse(BaseModel):
    success: bool
    stats: WeightDistributionModel
    computation_time_ms: float = 0.0


class EdgeWeightDistributionResponse(BaseModel):
    success: bool
    stats: WeightDistributionModel
    computation_time_ms: float = 0.0


class GraphGenerationStats(BaseModel):
    nodes: int
    brokers: int
    topics: int
    applications: int


class GraphGenerationResponse(BaseModel):
    success: bool
    message: str
    metadata: Dict[str, Any]
    stats: GraphGenerationStats
    graph_data: Dict[str, Any]


class GenericSuccessResponse(BaseModel):
    success: bool
    message: str


class GraphImportResponse(BaseModel):
    success: bool
    message: str
    stats: Dict[str, Any]


class GraphGenerateImportResponse(BaseModel):
    success: bool
    message: str
    generation: Dict[str, Any]
    import_stats: Dict[str, Any]


class ExportFormat(str, Enum):
    """Format type for exported data."""
    ANALYSIS = "analysis"      # Flat components/edges format (visualization)
    PERSISTENCE = "persistence" # Nested nodes/brokers/topics format (re-import)


class GraphExportResponse(BaseModel):
    success: bool
    export_format: ExportFormat = Field(default=ExportFormat.ANALYSIS)
    components: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    stats: Dict[str, Any]


class LimitedGraphExportStats(BaseModel):
    component_count: int
    edge_count: int
    node_limit: int
    edge_limit: Optional[int]
    limited: bool


class LimitedGraphExportResponse(BaseModel):
    success: bool
    export_format: ExportFormat = Field(default=ExportFormat.ANALYSIS)
    components: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    stats: LimitedGraphExportStats


class Neo4jExportStats(BaseModel):
    nodes: int
    brokers: int
    topics: int
    applications: int
    libraries: int


class Neo4jExportResponse(BaseModel):
    success: bool
    export_format: ExportFormat = Field(default=ExportFormat.PERSISTENCE)
    message: str
    graph_data: Dict[str, Any]
    stats: Neo4jExportStats


class SearchNodesResponse(BaseModel):
    success: bool
    query: str
    count: int
    nodes: List[Dict[str, Any]]


class NodeConnectionsStats(BaseModel):
    connected_nodes: int
    edges: int


class NodeConnectionsResponse(BaseModel):
    success: bool
    node_id: str
    depth: int
    components: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    stats: NodeConnectionsStats


class TopologyResponse(BaseModel):
    success: bool
    node_id: Optional[str]
    components: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    stats: Dict[str, int]


class ComponentQueryParams(BaseModel):
    component_type: Optional[str] = Field(None, description="Filter by component type")
    min_weight: Optional[float] = Field(None, description="Minimum weight threshold")
    criticality_level: Optional[str] = Field(None, description="Filter by criticality level")


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    neo4j_connected: bool
    message: Optional[str] = None


class EventSimulationRequest(GraphRequestWithCredentials):
    source_app: str = Field(..., description="Source application ID for event simulation")
    num_messages: int = Field(default=100, description="Number of messages to simulate")
    duration: float = Field(default=10.0, description="Simulation duration in seconds")


class FailureSimulationRequest(GraphRequestWithCredentials):
    target_id: str = Field(..., description="Target component ID to simulate failure")
    layer: str = Field(default="system", description="Analysis layer: app, infra, mw, system (aliases supported)")
    cascade_probability: float = Field(default=1.0, description="Cascade propagation probability (0.0-1.0)")


class ExhaustiveSimulationRequest(GraphRequestWithCredentials):
    layer: str = Field(default="system", description="Analysis layer: app, infra, mw, system (aliases supported)")
    cascade_probability: float = Field(default=1.0, description="Cascade propagation probability (0.0-1.0)")


class ReportRequest(GraphRequestWithCredentials):
    layers: List[str] = Field(default=["app", "infra", "system"], description="Layers to include in report")
