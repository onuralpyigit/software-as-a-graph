"""
Validation Pipeline

Orchestrates the full validation workflow:
    1. Graph Analysis → Predicted criticality scores
    2. Failure Simulation → Actual impact scores
    3. Statistical Validation → Compare predictions vs reality

Retrieves graph data directly from Neo4j for validation.

Layers:
    - app: Application layer
    - infra: Infrastructure layer
    - mw-app: Middleware-Application layer
    - mw-infra: Middleware-Infrastructure layer
    - system: Complete system
"""

from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .metrics import ValidationTargets
from .validator import Validator, ValidationResult, ComponentComparison


# Layer definitions
LAYER_DEFINITIONS = {
    "app": {
        "name": "Application Layer",
        "description": "Application components only",
        "component_types": {"Application"},
    },
    "infra": {
        "name": "Infrastructure Layer",
        "description": "Infrastructure nodes only",
        "component_types": {"Node"},
    },
    "mw-app": {
        "name": "Middleware-Application Layer",
        "description": "Applications and Brokers",
        "component_types": {"Application", "Broker"},
    },
    "mw-infra": {
        "name": "Middleware-Infrastructure Layer",
        "description": "Nodes and Brokers",
        "component_types": {"Node", "Broker"},
    },
    "system": {
        "name": "Complete System",
        "description": "All components including Libraries",
        "component_types": {"Application", "Broker", "Node", "Library"},
    },
}


@dataclass
class LayerValidationResult:
    """Validation result for a single layer."""
    layer: str
    layer_name: str
    
    # Data counts
    predicted_components: int = 0
    simulated_components: int = 0
    matched_components: int = 0
    
    # Validation result
    validation_result: Optional[ValidationResult] = None
    
    # Summary metrics
    spearman: float = 0.0
    f1_score: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    top_5_overlap: float = 0.0
    rmse: float = 0.0
    
    # Pass/fail
    passed: bool = False
    
    # Component comparisons
    comparisons: List[ComponentComparison] = field(default_factory=list)
    
    # Warnings
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "layer_name": self.layer_name,
            "data": {
                "predicted_components": self.predicted_components,
                "simulated_components": self.simulated_components,
                "matched_components": self.matched_components,
            },
            "summary": {
                "passed": self.passed,
                "spearman": round(self.spearman, 4),
                "f1_score": round(self.f1_score, 4),
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
                "top_5_overlap": round(self.top_5_overlap, 4),
                "rmse": round(self.rmse, 4),
            },
            "validation_result": self.validation_result.to_dict() if self.validation_result else None,
            "warnings": self.warnings,
        }


@dataclass
class PipelineResult:
    """Complete validation pipeline result."""
    timestamp: str
    
    # Layer results
    layers: Dict[str, LayerValidationResult] = field(default_factory=dict)
    
    # Overall statistics
    total_components: int = 0
    layers_passed: int = 0
    all_passed: bool = False
    
    # Cross-layer insights
    cross_layer_insights: List[str] = field(default_factory=list)
    
    # Configuration
    targets: ValidationTargets = field(default_factory=ValidationTargets)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total_components": self.total_components,
                "layers_validated": len(self.layers),
                "layers_passed": self.layers_passed,
                "all_passed": self.all_passed,
            },
            "layers": {k: v.to_dict() for k, v in self.layers.items()},
            "cross_layer_insights": self.cross_layer_insights,
            "targets": self.targets.to_dict(),
        }
    
    def get_layer_summary(self) -> List[Dict[str, Any]]:
        """Get summary table for all layers."""
        summary = []
        for layer_name, result in self.layers.items():
            summary.append({
                "layer": layer_name,
                "n": result.matched_components,
                "spearman": result.spearman,
                "f1": result.f1_score,
                "precision": result.precision,
                "recall": result.recall,
                "top5": result.top_5_overlap,
                "passed": result.passed,
            })
        return summary


class ValidationPipeline:
    """
    Full validation pipeline: Analysis → Simulation → Validation.
    
    Retrieves graph data directly from Neo4j and orchestrates:
        1. Graph analysis to get predicted criticality scores
        2. Failure simulation to get actual impact scores
        3. Statistical validation comparing predictions vs reality
    
    Example:
        >>> pipeline = ValidationPipeline(uri="bolt://localhost:7687")
        >>> result = pipeline.run(layers=["app", "infra", "system"])
        >>> print(f"All passed: {result.all_passed}")
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
        
        # Validator instance
        self.validator = Validator(targets=self.targets)
        
        # Lazy-loaded modules
        self._analyzer = None
        self._simulator = None
    
    @property
    def analyzer(self):
        """Lazy-load analyzer."""
        if self._analyzer is None:
            try:
                from ..analysis import GraphAnalyzer
                self._analyzer = GraphAnalyzer(
                    uri=self.uri,
                    user=self.user,
                    password=self.password
                )
            except ImportError:
                raise ImportError("Analysis module not available")
        return self._analyzer
    
    @property
    def simulator(self):
        """Lazy-load simulator."""
        if self._simulator is None:
            try:
                from ..simulation import Simulator
                self._simulator = Simulator(
                    uri=self.uri,
                    user=self.user,
                    password=self.password
                )
            except ImportError:
                raise ImportError("Simulation module not available")
        return self._simulator
    
    def run(
        self,
        layers: Optional[List[str]] = None,
        include_comparisons: bool = True
    ) -> PipelineResult:
        """
        Run the full validation pipeline.
        
        Args:
            layers: Layers to validate (default: app, infra, system)
            include_comparisons: Include detailed component comparisons
            
        Returns:
            PipelineResult with validation results for each layer
        """
        timestamp = datetime.now().isoformat()
        
        if layers is None:
            layers = ["app", "infra", "system"]
        
        # Validate layer names
        valid_layers = [l for l in layers if l in LAYER_DEFINITIONS]
        if len(valid_layers) < len(layers):
            invalid = set(layers) - set(valid_layers)
            self.logger.warning(f"Invalid layers ignored: {invalid}")
        
        self.logger.info(f"Starting validation pipeline for layers: {valid_layers}")
        
        # Run validation for each layer
        layer_results: Dict[str, LayerValidationResult] = {}
        total_components = 0
        
        for layer in valid_layers:
            self.logger.info(f"Validating layer: {layer}")
            
            try:
                result = self._validate_layer(layer, include_comparisons)
                layer_results[layer] = result
                total_components += result.matched_components
            except Exception as e:
                self.logger.error(f"Failed to validate layer {layer}: {e}")
                layer_results[layer] = LayerValidationResult(
                    layer=layer,
                    layer_name=LAYER_DEFINITIONS[layer]["name"],
                    warnings=[f"Validation failed: {str(e)}"],
                )
        
        # Count passed layers
        layers_passed = sum(1 for r in layer_results.values() if r.passed)
        all_passed = layers_passed == len(valid_layers) and len(valid_layers) > 0
        
        # Generate cross-layer insights
        insights = self._generate_insights(layer_results)
        
        return PipelineResult(
            timestamp=timestamp,
            layers=layer_results,
            total_components=total_components,
            layers_passed=layers_passed,
            all_passed=all_passed,
            cross_layer_insights=insights,
            targets=self.targets,
        )
    
    def _validate_layer(
        self,
        layer: str,
        include_comparisons: bool
    ) -> LayerValidationResult:
        """Validate a single layer."""
        layer_def = LAYER_DEFINITIONS[layer]
        
        # Phase 1: Analysis (Predictions)
        self.logger.info(f"  [1/3] Running graph analysis...")
        analysis_data = self._run_analysis(layer)
        self.logger.info(f"        Analyzed {len(analysis_data['predicted'])} components")
        
        # Phase 2: Simulation (Ground Truth)
        self.logger.info(f"  [2/3] Running failure simulation...")
        simulation_data = self._run_simulation(layer)
        self.logger.info(f"        Simulated {len(simulation_data['actual'])} components")
        
        # Align component sets
        pred_ids = set(analysis_data['predicted'].keys())
        actual_ids = set(simulation_data['actual'].keys())
        common_ids = pred_ids & actual_ids
        
        self.logger.info(f"        Matched {len(common_ids)} components")
        
        # Phase 3: Validation (Comparison)
        self.logger.info(f"  [3/3] Running statistical validation...")
        
        # Filter to common components
        pred_scores = {k: analysis_data['predicted'][k] for k in common_ids}
        actual_scores = {k: simulation_data['actual'][k] for k in common_ids}
        comp_types = {k: analysis_data['types'].get(k, "Unknown") for k in common_ids}
        
        validation_result = self.validator.validate(
            predicted_scores=pred_scores,
            actual_scores=actual_scores,
            component_types=comp_types,
            layer=layer,
            context=f"{layer_def['name']} Validation",
        )
        
        # Extract summary metrics
        overall = validation_result.overall
        
        # Build layer result
        result = LayerValidationResult(
            layer=layer,
            layer_name=layer_def["name"],
            predicted_components=len(pred_ids),
            simulated_components=len(actual_ids),
            matched_components=len(common_ids),
            validation_result=validation_result,
            spearman=overall.correlation.spearman,
            f1_score=overall.classification.f1_score,
            precision=overall.classification.precision,
            recall=overall.classification.recall,
            top_5_overlap=overall.ranking.top_5_overlap,
            rmse=overall.error.rmse,
            passed=overall.passed,
            comparisons=overall.components if include_comparisons else [],
            warnings=validation_result.warnings,
        )
        
        return result
    
    def _run_analysis(self, layer: str) -> Dict[str, Any]:
        """Run graph analysis for a layer."""
        try:
            # Use the analysis module
            result = self.analyzer.analyze_layer(layer)
            
            # Extract predicted scores (overall quality score)
            predicted = {}
            types = {}
            
            for comp in result.quality.components:
                predicted[comp.id] = comp.scores.overall
                types[comp.id] = comp.type
            
            return {
                "predicted": predicted,
                "types": types,
            }
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {"predicted": {}, "types": {}}
    
    def _run_simulation(self, layer: str) -> Dict[str, Any]:
        """Run failure simulation for a layer."""
        try:
            # Use the simulation module
            results = self.simulator.run_failure_simulation_exhaustive(layer=layer)
            
            # Extract actual impact scores
            actual = {}
            for result in results:
                actual[result.target_id] = result.impact.composite_impact
            
            return {"actual": actual}
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            return {"actual": {}}
    
    def _generate_insights(
        self,
        layer_results: Dict[str, LayerValidationResult]
    ) -> List[str]:
        """Generate cross-layer insights."""
        insights = []
        
        if not layer_results:
            return insights
        
        # Best and worst performing layers
        valid_results = [(l, r) for l, r in layer_results.items() if r.matched_components > 0]
        
        if valid_results:
            best = max(valid_results, key=lambda x: x[1].spearman)
            worst = min(valid_results, key=lambda x: x[1].spearman)
            
            if best[0] != worst[0]:
                insights.append(
                    f"Best correlation: {best[0]} (ρ={best[1].spearman:.3f}), "
                    f"Worst: {worst[0]} (ρ={worst[1].spearman:.3f})"
                )
        
        # Overall pass rate
        passed = sum(1 for _, r in valid_results if r.passed)
        total = len(valid_results)
        
        if total > 0:
            if passed == total:
                insights.append(f"All {total} layers passed validation targets")
            else:
                insights.append(f"{passed}/{total} layers passed validation targets")
        
        # Classification consistency
        f1_scores = [r.f1_score for _, r in valid_results]
        if f1_scores:
            avg_f1 = sum(f1_scores) / len(f1_scores)
            if avg_f1 >= 0.8:
                insights.append(f"Strong classification accuracy (avg F1={avg_f1:.3f})")
            elif avg_f1 >= 0.6:
                insights.append(f"Moderate classification accuracy (avg F1={avg_f1:.3f})")
            else:
                insights.append(f"Weak classification accuracy (avg F1={avg_f1:.3f})")
        
        # Layer-specific insights
        for layer, result in layer_results.items():
            if result.spearman < self.targets.spearman:
                insights.append(
                    f"Layer '{layer}' has weak correlation - consider adjusting weights"
                )
            
            if result.precision < 0.5 and result.recall > 0.8:
                insights.append(
                    f"Layer '{layer}' over-predicts critical components (high FP rate)"
                )
            elif result.precision > 0.8 and result.recall < 0.5:
                insights.append(
                    f"Layer '{layer}' under-predicts critical components (high FN rate)"
                )
        
        return insights
    
    def export_result(self, result: PipelineResult, output_path: str) -> None:
        """Export result to JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        self.logger.info(f"Result exported to: {path}")


class QuickValidator:
    """
    Quick validation for pre-computed scores.
    
    Use when analysis and simulation results are already available.
    """
    
    def __init__(self, targets: Optional[ValidationTargets] = None):
        self.targets = targets or ValidationTargets()
        self.validator = Validator(targets=self.targets)
    
    def validate(
        self,
        predicted_scores: Dict[str, float],
        actual_scores: Dict[str, float],
        component_types: Optional[Dict[str, str]] = None,
        layer: str = "custom",
        context: str = "Quick Validation"
    ) -> ValidationResult:
        """
        Validate pre-computed scores.
        
        Args:
            predicted_scores: Dict mapping component ID to predicted score
            actual_scores: Dict mapping component ID to actual impact score
            component_types: Optional dict mapping component ID to type
            layer: Layer name
            context: Context description
            
        Returns:
            ValidationResult
        """
        return self.validator.validate(
            predicted_scores=predicted_scores,
            actual_scores=actual_scores,
            component_types=component_types,
            layer=layer,
            context=context,
        )
    
    def validate_from_files(
        self,
        predicted_file: str,
        actual_file: str,
        layer: str = "file"
    ) -> ValidationResult:
        """Validate scores from JSON files."""
        with open(predicted_file, 'r') as f:
            predicted = json.load(f)
        
        with open(actual_file, 'r') as f:
            actual = json.load(f)
        
        return self.validate(
            predicted_scores=predicted,
            actual_scores=actual,
            layer=layer,
            context=f"File validation: {predicted_file} vs {actual_file}",
        )