"""
Validation Pipeline

Orchestrates the validation process for graph-based criticality prediction:
1. Analysis: Run graph analysis to get predicted criticality scores
2. Simulation: Run failure simulation to get actual impact scores  
3. Validation: Compare predictions vs actual using statistical metrics

Key Features:
- Unified layer definitions aligned with analysis module
- Proper component set alignment between analysis and simulation
- Support for JSON file and Neo4j database inputs
- Multi-layer validation (application, infrastructure, complete)
- Detailed diagnostic reporting
- Scatter plot data export for visualization

Author: Software-as-a-Graph Research Project
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

from .validator import Validator, ValidationResult, MultiLayerValidator
from .metrics import ValidationTargets, ValidationSummary

# Import analysis and simulation modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from analysis.analyzer import GraphAnalyzer
    from analysis.classifier import CriticalityLevel
    from analysis.structural_analyzer import AnalysisLayer, LAYER_DEFINITIONS as ANALYSIS_LAYERS
    HAS_ANALYSIS = True
except ImportError:
    try:
        from src.analysis.analyzer import GraphAnalyzer
        from src.analysis.classifier import CriticalityLevel
        from src.analysis.structural_analyzer import AnalysisLayer, LAYER_DEFINITIONS as ANALYSIS_LAYERS
        HAS_ANALYSIS = True
    except ImportError:
        HAS_ANALYSIS = False
        GraphAnalyzer = None
        ANALYSIS_LAYERS = {}

try:
    from simulation.simulator import Simulator
    from simulation.simulation_graph import SimulationGraph
    HAS_SIMULATION = True
except ImportError:
    try:
        from src.simulation.simulator import Simulator
        from src.simulation.simulation_graph import SimulationGraph
        HAS_SIMULATION = True
    except ImportError:
        HAS_SIMULATION = False
        Simulator = None


# =============================================================================
# Unified Layer Definitions (aligned with analysis module)
# =============================================================================

LAYER_DEFINITIONS = {
    "application": {
        "name": "Application Layer",
        "analysis_layer": "application",
        "component_types": {"Application"},  # Match analysis module
        "description": "Application-level service dependencies",
        "includes_brokers": False,
    },
    "infrastructure": {
        "name": "Infrastructure Layer",
        "analysis_layer": "infrastructure",
        "component_types": {"Node"},
        "description": "Infrastructure nodes and network topology",
        "includes_brokers": False,
    },
    "app_broker": {
        "name": "Application-Broker Layer",
        "analysis_layer": "app_broker",
        "component_types": {"Application", "Broker"},
        "description": "Applications and message brokers",
        "includes_brokers": True,
    },
    "complete": {
        "name": "Complete System",
        "analysis_layer": "complete",
        "component_types": {"Application", "Broker", "Node"},
        "description": "All system components across all layers",
        "includes_brokers": True,
    },
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ComponentComparison:
    """Comparison data for a single component."""
    id: str
    type: str
    predicted_score: float
    actual_impact: float
    error: float
    predicted_critical: bool
    actual_critical: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "predicted": round(self.predicted_score, 4),
            "actual": round(self.actual_impact, 4),
            "error": round(self.error, 4),
            "pred_critical": self.predicted_critical,
            "actual_critical": self.actual_critical,
        }


@dataclass
class LayerValidationResult:
    """Validation result for a single layer."""
    layer: str
    layer_name: str
    
    # Analysis results
    analysis_components: int
    predicted_scores: Dict[str, float]
    critical_count: int
    high_count: int
    
    # Simulation results
    simulation_components: int
    actual_scores: Dict[str, float]
    avg_impact: float
    max_impact: float
    
    # Validation results
    validation_result: ValidationResult
    component_comparisons: List[ComponentComparison]
    
    # Alignment info
    matched_components: int
    analysis_only: List[str]
    simulation_only: List[str]
    
    @property
    def passed(self) -> bool:
        return self.validation_result.passed
    
    def get_scatter_data(self) -> List[Tuple[str, float, float]]:
        """Get (id, predicted, actual) tuples for scatter plot."""
        return [(c.id, c.predicted_score, c.actual_impact) 
                for c in self.component_comparisons]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "layer_name": self.layer_name,
            "passed": self.passed,
            "analysis": {
                "components": self.analysis_components,
                "critical_count": self.critical_count,
                "high_count": self.high_count,
            },
            "simulation": {
                "components": self.simulation_components,
                "avg_impact": round(self.avg_impact, 4),
                "max_impact": round(self.max_impact, 4),
            },
            "alignment": {
                "matched": self.matched_components,
                "analysis_only": self.analysis_only,
                "simulation_only": self.simulation_only,
            },
            "validation": self.validation_result.to_dict(),
            "comparisons": [c.to_dict() for c in self.component_comparisons],
        }


@dataclass
class PipelineResult:
    """Complete pipeline result across all layers."""
    timestamp: str
    
    layers: Dict[str, LayerValidationResult]
    targets: ValidationTargets
    
    total_components: int = 0
    layers_passed: int = 0
    
    # Cross-layer analysis
    cross_layer_insights: List[str] = field(default_factory=list)
    
    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.layers.values())
    
    def get_all_scatter_data(self) -> Dict[str, List[Tuple[str, float, float]]]:
        """Get scatter data for all layers."""
        return {layer: result.get_scatter_data() 
                for layer, result in self.layers.items()}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total_components": self.total_components,
                "layers_validated": len(self.layers),
                "layers_passed": self.layers_passed,
                "all_passed": self.all_passed,
            },
            "targets": self.targets.to_dict(),
            "layers": {k: v.to_dict() for k, v in self.layers.items()},
            "cross_layer_insights": self.cross_layer_insights,
        }


# =============================================================================
# Validation Pipeline
# =============================================================================

class ValidationPipeline:
    """
    Orchestrates validation of graph-based criticality prediction.
    
    Pipeline stages:
    1. Load graph data (from JSON or Neo4j)
    2. Run graph analysis to get predicted scores
    3. Run failure simulation to get actual impact scores
    4. Compare predictions vs actual using statistical metrics
    5. Generate validation report with diagnostics
    
    Key design decisions:
    - Analysis drives component selection (predicted components)
    - Simulation provides ground truth for those components
    - Validation compares on intersection of both sets
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        targets: Optional[ValidationTargets] = None
    ):
        """
        Initialize validation pipeline.
        
        Args:
            uri: Neo4j connection URI
            user: Neo4j username
            password: Neo4j password
            targets: Validation success criteria
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.targets = targets or ValidationTargets()
        
        self.logger = logging.getLogger(__name__)

        # Validators
        self.validator = Validator(targets=self.targets)
        
        # Check dependencies
        if not HAS_ANALYSIS:
            raise ImportError("Analysis module not available. Install required dependencies.")
        if not HAS_SIMULATION:
            raise ImportError("Simulation module not available. Install required dependencies.")
    
    def run(
        self,
        layers: Optional[List[str]] = None,
        include_diagnostics: bool = True
    ) -> PipelineResult:
        """
        Run the full validation pipeline.
        
        Args:
            layers: List of layers to validate (default: application, infrastructure, complete)
            include_diagnostics: Whether to include detailed component comparisons
            
        Returns:
            PipelineResult with validation results for each layer
        """
        timestamp = datetime.now().isoformat()
        
        if layers is None:
            layers = ["application", "infrastructure", "complete"]
        
        # Validate layer names
        valid_layers = []
        for layer in layers:
            if layer in LAYER_DEFINITIONS:
                valid_layers.append(layer)
            else:
                self.logger.warning(f"Unknown layer '{layer}', skipping. Valid: {list(LAYER_DEFINITIONS.keys())}")
        
        self.logger.info(f"Starting validation pipeline")
        self.logger.info(f"Layers to validate: {valid_layers}")
        
        results = {}
        total_components = 0
        
        for layer in valid_layers:
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"VALIDATING LAYER: {layer.upper()}")
            self.logger.info(f"{'=' * 60}")
            
            try:
                layer_result = self._validate_layer(layer, include_diagnostics)
                results[layer] = layer_result
                total_components = max(total_components, layer_result.matched_components)
                
                status = "PASSED" if layer_result.passed else "FAILED"
                self.logger.info(f"Layer {layer}: {status}")
                self.logger.info(f"  Spearman: {layer_result.validation_result.overall.correlation.spearman:.4f}")
                self.logger.info(f"  F1 Score: {layer_result.validation_result.overall.classification.f1_score:.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed to validate layer {layer}: {e}")
                raise
        
        # Generate cross-layer insights
        insights = self._generate_insights(results)
        
        layers_passed = sum(1 for r in results.values() if r.passed)
        
        return PipelineResult(
            timestamp=timestamp,
            layers=results,
            targets=self.targets,
            total_components=total_components,
            layers_passed=layers_passed,
            cross_layer_insights=insights,
        )
    
    def _validate_layer(
        self, 
        layer: str,
        include_diagnostics: bool = True
    ) -> LayerValidationResult:
        """Validate a single layer."""
        layer_def = LAYER_DEFINITIONS[layer]
        
        # Phase 1: Analysis (Predictions)
        self.logger.info("[1/3] Running Graph Analysis...")
        analysis_data = self._run_analysis(layer)
        self.logger.info(f"      Analyzed {len(analysis_data['predicted_scores'])} components")
        self.logger.info(f"      Critical: {analysis_data['critical_count']}, High: {analysis_data['high_count']}")
        
        # Phase 2: Simulation (Ground Truth)
        self.logger.info("[2/3] Running Failure Simulation...")
        simulation_data = self._run_simulation(layer, analysis_data['component_types'])
        self.logger.info(f"      Simulated {len(simulation_data['actual_scores'])} components")
        self.logger.info(f"      Avg Impact: {simulation_data['avg_impact']:.4f}, Max: {simulation_data['max_impact']:.4f}")
        
        # Align component sets
        pred_ids = set(analysis_data['predicted_scores'].keys())
        actual_ids = set(simulation_data['actual_scores'].keys())
        common_ids = pred_ids & actual_ids
        analysis_only = sorted(pred_ids - actual_ids)
        simulation_only = sorted(actual_ids - pred_ids)
        
        self.logger.info(f"      Matched: {len(common_ids)} components")
        if analysis_only:
            self.logger.info(f"      Analysis-only: {analysis_only}")
        if simulation_only:
            self.logger.info(f"      Simulation-only: {simulation_only}")
        
        # Phase 3: Validation (Comparison)
        self.logger.info("[3/3] Running Statistical Validation...")
        
        # Filter to common components
        pred_scores = {k: analysis_data['predicted_scores'][k] for k in common_ids}
        actual_scores = {k: simulation_data['actual_scores'][k] for k in common_ids}
        comp_types = {k: analysis_data['component_types'][k] for k in common_ids}
        
        validation_result = self.validator.validate(
            predicted_scores=pred_scores,
            actual_scores=actual_scores,
            component_types=comp_types,
            layer=layer,
            context=f"{layer_def['name']} Validation",
        )
        
        # Build component comparisons
        comparisons = []
        if include_diagnostics:
            # Determine critical thresholds
            pred_values = list(pred_scores.values())
            actual_values = list(actual_scores.values())
            pred_threshold = self._percentile(pred_values, 75) if pred_values else 0
            actual_threshold = self._percentile(actual_values, 75) if actual_values else 0
            
            for comp_id in sorted(common_ids):
                pred = pred_scores[comp_id]
                actual = actual_scores[comp_id]
                comparisons.append(ComponentComparison(
                    id=comp_id,
                    type=comp_types[comp_id],
                    predicted_score=pred,
                    actual_impact=actual,
                    error=abs(pred - actual),
                    predicted_critical=pred >= pred_threshold,
                    actual_critical=actual >= actual_threshold,
                ))
        
        return LayerValidationResult(
            layer=layer,
            layer_name=layer_def["name"],
            analysis_components=len(analysis_data['predicted_scores']),
            predicted_scores=analysis_data['predicted_scores'],
            critical_count=analysis_data['critical_count'],
            high_count=analysis_data['high_count'],
            simulation_components=len(simulation_data['actual_scores']),
            actual_scores=simulation_data['actual_scores'],
            avg_impact=simulation_data['avg_impact'],
            max_impact=simulation_data['max_impact'],
            validation_result=validation_result,
            component_comparisons=comparisons,
            matched_components=len(common_ids),
            analysis_only=analysis_only,
            simulation_only=simulation_only,
        )
    
    def _run_analysis(self, layer: str) -> Dict[str, Any]:
        """
        Run graph analysis to get predicted criticality scores.
        
        Returns dict with:
        - predicted_scores: Dict[component_id, overall_score]
        - component_types: Dict[component_id, type]
        - critical_count, high_count
        """
        # Initialize analyzer
        analyzer_kwargs = {}
        analyzer_kwargs["uri"] = self.uri
        analyzer_kwargs["user"] = self.user
        analyzer_kwargs["password"] = self.password
        
        with GraphAnalyzer(**analyzer_kwargs) as analyzer:
            # Use analyze() which returns dict with "results" containing QualityAnalysisResult
            analysis = analyzer.analyze(layer=layer)
        
        # Extract scores from quality result
        quality_result = analysis["results"]
        
        predicted_scores = {}
        component_types = {}
        critical_count = 0
        high_count = 0
        
        # Get allowed types for this layer
        layer_types = LAYER_DEFINITIONS[layer]["component_types"]
        
        for comp in quality_result.components:
            # Filter by layer component types
            if comp.type not in layer_types:
                continue
            
            predicted_scores[comp.id] = comp.scores.overall
            component_types[comp.id] = comp.type
            
            if comp.levels.overall == CriticalityLevel.CRITICAL:
                critical_count += 1
            elif comp.levels.overall == CriticalityLevel.HIGH:
                high_count += 1
        
        return {
            "predicted_scores": predicted_scores,
            "component_types": component_types,
            "critical_count": critical_count,
            "high_count": high_count,
        }
    
    def _run_simulation(
        self, 
        layer: str,
        component_types: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Run failure simulation to get actual impact scores.
        
        Args:
            layer: Layer name
            component_types: Dict mapping component ID to type (from analysis)
            
        Returns dict with:
        - actual_scores: Dict[component_id, composite_impact]
        - avg_impact, max_impact
        """
        # Initialize simulator
        sim_kwargs = {}
        sim_kwargs["uri"] = self.uri
        sim_kwargs["user"] = self.user
        sim_kwargs["password"] = self.password
        
        with Simulator(**sim_kwargs) as simulator:
            # Run exhaustive failure simulation for the layer
            results = simulator.run_failure_simulation_exhaustive(layer=layer)
        
        # Extract actual impact scores
        actual_scores = {}
        impacts = []
        
        # Get allowed types for this layer
        layer_types = LAYER_DEFINITIONS[layer]["component_types"]
        
        for result in results:
            # Get component type from simulation or from analysis
            comp_type = getattr(result, 'target_type', None)
            if comp_type is None:
                comp_type = component_types.get(result.target_id, "Unknown")
            
            # Filter by layer component types
            if comp_type not in layer_types:
                continue
            
            actual_scores[result.target_id] = result.impact.composite_impact
            impacts.append(result.impact.composite_impact)
        
        avg_impact = sum(impacts) / len(impacts) if impacts else 0.0
        max_impact = max(impacts) if impacts else 0.0
        
        return {
            "actual_scores": actual_scores,
            "avg_impact": avg_impact,
            "max_impact": max_impact,
        }
    
    def _generate_insights(
        self, 
        results: Dict[str, LayerValidationResult]
    ) -> List[str]:
        """Generate cross-layer validation insights."""
        insights = []
        
        # Insight 1: Overall validation status
        passed_layers = [l for l, r in results.items() if r.passed]
        failed_layers = [l for l, r in results.items() if not r.passed]
        
        if len(passed_layers) == len(results):
            insights.append("All layers passed validation - the graph model accurately predicts criticality.")
        elif len(failed_layers) == len(results):
            insights.append("All layers failed validation - consider reviewing the quality formulas and weights.")
        else:
            insights.append(f"Mixed results: {passed_layers} passed, {failed_layers} failed.")
        
        # Insight 2: Correlation comparison
        correlations = {l: r.validation_result.overall.correlation.spearman 
                       for l, r in results.items()}
        if correlations:
            best_layer = max(correlations, key=correlations.get)
            worst_layer = min(correlations, key=correlations.get)
            insights.append(
                f"Best correlation in {best_layer} (ρ={correlations[best_layer]:.3f}), "
                f"worst in {worst_layer} (ρ={correlations[worst_layer]:.3f})."
            )
        
        # Insight 3: Classification accuracy comparison  
        f1_scores = {l: r.validation_result.overall.classification.f1_score 
                    for l, r in results.items()}
        if f1_scores:
            best_f1 = max(f1_scores, key=f1_scores.get)
            if f1_scores[best_f1] >= 0.8:
                insights.append(f"Critical component detection works best for {best_f1} layer (F1={f1_scores[best_f1]:.3f}).")
        
        # Insight 4: Component coverage
        for layer, result in results.items():
            if result.analysis_only:
                insights.append(
                    f"{layer}: {len(result.analysis_only)} components in analysis but not simulated "
                    f"({', '.join(result.analysis_only[:3])}{'...' if len(result.analysis_only) > 3 else ''})."
                )
        
        # Insight 5: Error analysis
        for layer, result in results.items():
            if result.component_comparisons:
                high_error = [c for c in result.component_comparisons if c.error > 0.4]
                if high_error:
                    worst = max(high_error, key=lambda c: c.error)
                    insights.append(
                        f"{layer}: High prediction error for {worst.id} "
                        f"(predicted={worst.predicted_score:.3f}, actual={worst.actual_impact:.3f})."
                    )
        
        return insights
    
    def export_report(
        self,
        result: PipelineResult,
        output_path: str,
        include_comparisons: bool = True,
        include_scatter_data: bool = True
    ) -> None:
        """
        Export validation report to JSON file.
        
        Args:
            result: Pipeline result to export
            output_path: Path to output JSON file
            include_comparisons: Include component-level comparisons
            include_scatter_data: Include scatter plot data
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = result.to_dict()
        
        # Optionally add scatter data
        if include_scatter_data:
            data["scatter_data"] = {}
            for layer, layer_result in result.layers.items():
                data["scatter_data"][layer] = [
                    {"id": c.id, "predicted": c.predicted_score, "actual": c.actual_impact}
                    for c in layer_result.component_comparisons
                ]
        
        # Optionally remove detailed comparisons
        if not include_comparisons:
            for layer_data in data.get("layers", {}).values():
                layer_data.pop("comparisons", None)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        self.logger.info(f"Report exported to: {path.absolute()}")
    
    def export_scatter_csv(
        self,
        result: PipelineResult,
        output_dir: str
    ) -> List[str]:
        """
        Export scatter plot data to CSV files (one per layer).
        
        Args:
            result: Pipeline result
            output_dir: Directory for CSV files
            
        Returns:
            List of created file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        files_created = []
        
        for layer, layer_result in result.layers.items():
            csv_path = output_path / f"validation_{layer}_scatter.csv"
            
            with open(csv_path, 'w') as f:
                f.write("component_id,type,predicted_score,actual_impact,error,pred_critical,actual_critical\n")
                for comp in layer_result.component_comparisons:
                    f.write(f"{comp.id},{comp.type},{comp.predicted_score:.4f},"
                           f"{comp.actual_impact:.4f},{comp.error:.4f},"
                           f"{comp.predicted_critical},{comp.actual_critical}\n")
            
            files_created.append(str(csv_path))
            self.logger.info(f"Scatter data exported to: {csv_path}")
        
        return files_created
    
    @staticmethod
    def _percentile(values: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        import math
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * percentile / 100.0
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_values[int(k)]
        return sorted_values[int(f)] * (c - k) + sorted_values[int(c)] * (k - f)


# =============================================================================
# Quick Validator (for pre-computed scores)
# =============================================================================

class QuickValidator:
    """
    Quick validation for pre-computed scores.
    
    Use when analysis and simulation results are already available.
    """
    
    def __init__(self, targets: Optional[ValidationTargets] = None):
        self.targets = targets or ValidationTargets()
        self.validator = Validator(targets=self.targets)
        self.logger = logging.getLogger(__name__)
    
    def validate_scores(
        self,
        predicted_scores: Dict[str, float],
        actual_scores: Dict[str, float],
        component_types: Optional[Dict[str, str]] = None,
        context: str = "Quick Validation"
    ) -> ValidationResult:
        """
        Validate pre-computed scores.
        
        Args:
            predicted_scores: Dict mapping component ID to predicted score
            actual_scores: Dict mapping component ID to actual impact score
            component_types: Optional dict mapping component ID to type
            context: Context description
            
        Returns:
            ValidationResult
        """
        return self.validator.validate(
            predicted_scores=predicted_scores,
            actual_scores=actual_scores,
            component_types=component_types,
            context=context,
        )
    
    def validate_from_files(
        self,
        predicted_file: str,
        actual_file: str,
        context: str = "File Validation"
    ) -> ValidationResult:
        """
        Validate scores from JSON files.
        
        Expected format: {"component_id": score, ...}
        """
        with open(predicted_file, 'r') as f:
            predicted_scores = json.load(f)
        
        with open(actual_file, 'r') as f:
            actual_scores = json.load(f)
        
        return self.validate_scores(
            predicted_scores=predicted_scores,
            actual_scores=actual_scores,
            context=context,
        )
    
    def validate_from_csv(
        self,
        csv_file: str,
        predicted_col: str = "predicted",
        actual_col: str = "actual",
        id_col: str = "id",
        type_col: Optional[str] = "type"
    ) -> ValidationResult:
        """
        Validate scores from CSV file.
        
        Args:
            csv_file: Path to CSV file
            predicted_col: Column name for predicted scores
            actual_col: Column name for actual scores
            id_col: Column name for component IDs
            type_col: Optional column name for component types
        """
        import csv
        
        predicted_scores = {}
        actual_scores = {}
        component_types = {}
        
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                comp_id = row[id_col]
                predicted_scores[comp_id] = float(row[predicted_col])
                actual_scores[comp_id] = float(row[actual_col])
                if type_col and type_col in row:
                    component_types[comp_id] = row[type_col]
        
        return self.validate_scores(
            predicted_scores=predicted_scores,
            actual_scores=actual_scores,
            component_types=component_types if component_types else None,
            context=f"CSV Validation: {csv_file}",
        )