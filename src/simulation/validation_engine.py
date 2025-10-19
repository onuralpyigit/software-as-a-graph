"""
Validation Engine

Validates simulation results against historical data and real-world observations.
Helps tune simulation parameters and assess accuracy of impact predictions.

Supports:
- Historical incident validation
- Accuracy metrics calculation
- Parameter tuning recommendations
- Simulation calibration
- Confidence scoring
"""

import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
import statistics


class ValidationMetric(Enum):
    """Types of validation metrics"""
    ACCURACY = "accuracy"                   # How often predictions match reality
    PRECISION = "precision"                 # True positives / (true positives + false positives)
    RECALL = "recall"                       # True positives / (true positives + false negatives)
    F1_SCORE = "f1_score"                  # Harmonic mean of precision and recall
    MEAN_ABSOLUTE_ERROR = "mae"            # Average absolute difference
    ROOT_MEAN_SQUARE_ERROR = "rmse"        # Root of average squared differences


@dataclass
class HistoricalIncident:
    """Record of a historical incident"""
    incident_id: str
    timestamp: datetime
    failed_components: List[str]
    affected_components: List[str]
    actual_impact_score: float
    actual_downtime_minutes: float
    actual_users_affected: int
    actual_financial_impact_usd: float
    root_cause: str
    resolution_time_hours: float
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'incident_id': self.incident_id,
            'timestamp': self.timestamp.isoformat(),
            'failed_components': self.failed_components,
            'affected_components': self.affected_components,
            'actual_impact_score': self.actual_impact_score,
            'actual_downtime_minutes': self.actual_downtime_minutes,
            'actual_users_affected': self.actual_users_affected,
            'actual_financial_impact_usd': self.actual_financial_impact_usd,
            'root_cause': self.root_cause,
            'resolution_time_hours': self.resolution_time_hours,
            'metadata': self.metadata or {}
        }


@dataclass
class ValidationResult:
    """Result of validation against historical data"""
    incident_id: str
    predicted_impact: float
    actual_impact: float
    impact_error: float
    predicted_affected: int
    actual_affected: int
    affected_error: int
    predicted_downtime_hours: float
    actual_downtime_hours: float
    downtime_error_hours: float
    accuracy_score: float
    confidence_level: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'incident_id': self.incident_id,
            'predicted_impact': round(self.predicted_impact, 3),
            'actual_impact': round(self.actual_impact, 3),
            'impact_error': round(self.impact_error, 3),
            'predicted_affected': self.predicted_affected,
            'actual_affected': self.actual_affected,
            'affected_error': self.affected_error,
            'predicted_downtime_hours': round(self.predicted_downtime_hours, 2),
            'actual_downtime_hours': round(self.actual_downtime_hours, 2),
            'downtime_error_hours': round(self.downtime_error_hours, 2),
            'accuracy_score': round(self.accuracy_score, 3),
            'confidence_level': self.confidence_level
        }


@dataclass
class CalibrationReport:
    """Report on simulation calibration"""
    total_incidents: int
    validation_results: List[ValidationResult]
    overall_accuracy: float
    mean_absolute_error: float
    root_mean_square_error: float
    precision: float
    recall: float
    f1_score: float
    recommendations: List[str]
    calibration_quality: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'total_incidents': self.total_incidents,
            'validation_results': [v.to_dict() for v in self.validation_results],
            'overall_accuracy': round(self.overall_accuracy, 3),
            'mean_absolute_error': round(self.mean_absolute_error, 3),
            'root_mean_square_error': round(self.root_mean_square_error, 3),
            'precision': round(self.precision, 3),
            'recall': round(self.recall, 3),
            'f1_score': round(self.f1_score, 3),
            'recommendations': self.recommendations,
            'calibration_quality': self.calibration_quality
        }


class ValidationEngine:
    """
    Validates simulation results against historical data
    
    Capabilities:
    - Compare predictions vs actual incidents
    - Calculate accuracy metrics
    - Identify systematic biases
    - Recommend parameter adjustments
    - Generate confidence scores
    """
    
    def __init__(self,
                 acceptable_error_threshold: float = 0.2,
                 minimum_accuracy_threshold: float = 0.7):
        """
        Initialize validation engine
        
        Args:
            acceptable_error_threshold: Maximum acceptable error (0-1)
            minimum_accuracy_threshold: Minimum accuracy for good predictions (0-1)
        """
        self.logger = logging.getLogger(__name__)
        self.error_threshold = acceptable_error_threshold
        self.accuracy_threshold = minimum_accuracy_threshold
        self.historical_incidents: List[HistoricalIncident] = []
    
    def add_historical_incident(self, incident: HistoricalIncident):
        """
        Add a historical incident to the validation dataset
        
        Args:
            incident: Historical incident record
        """
        self.historical_incidents.append(incident)
        self.logger.info(f"Added incident {incident.incident_id} to validation dataset")
    
    def load_incidents_from_dict(self, incidents_data: List[Dict]):
        """
        Load historical incidents from list of dictionaries
        
        Args:
            incidents_data: List of incident dictionaries
        """
        for data in incidents_data:
            incident = HistoricalIncident(
                incident_id=data['incident_id'],
                timestamp=datetime.fromisoformat(data['timestamp']),
                failed_components=data['failed_components'],
                affected_components=data['affected_components'],
                actual_impact_score=data['actual_impact_score'],
                actual_downtime_minutes=data['actual_downtime_minutes'],
                actual_users_affected=data['actual_users_affected'],
                actual_financial_impact_usd=data['actual_financial_impact_usd'],
                root_cause=data['root_cause'],
                resolution_time_hours=data['resolution_time_hours'],
                metadata=data.get('metadata', {})
            )
            self.add_historical_incident(incident)
        
        self.logger.info(f"Loaded {len(incidents_data)} historical incidents")
    
    def validate_simulation(self,
                           graph: nx.DiGraph,
                           incident: HistoricalIncident,
                           simulation_result: Any) -> ValidationResult:
        """
        Validate a simulation result against a historical incident
        
        Args:
            graph: NetworkX directed graph
            incident: Historical incident
            simulation_result: Result from FailureSimulator
        
        Returns:
            ValidationResult comparing prediction vs reality
        """
        self.logger.info(f"Validating simulation for incident {incident.incident_id}")
        
        # Extract predicted values
        predicted_impact = simulation_result.impact_score
        predicted_affected = len(simulation_result.affected_components)
        predicted_downtime = self._estimate_downtime_from_impact(predicted_impact)
        
        # Extract actual values
        actual_impact = incident.actual_impact_score
        actual_affected = len(incident.affected_components)
        actual_downtime = incident.actual_downtime_minutes / 60.0  # Convert to hours
        
        # Calculate errors
        impact_error = abs(predicted_impact - actual_impact)
        affected_error = abs(predicted_affected - actual_affected)
        downtime_error = abs(predicted_downtime - actual_downtime)
        
        # Calculate accuracy score (inverse of normalized error)
        accuracy_score = 1.0 - min(1.0, impact_error)
        
        # Determine confidence level
        confidence = self._calculate_confidence_level(accuracy_score)
        
        return ValidationResult(
            incident_id=incident.incident_id,
            predicted_impact=predicted_impact,
            actual_impact=actual_impact,
            impact_error=impact_error,
            predicted_affected=predicted_affected,
            actual_affected=actual_affected,
            affected_error=affected_error,
            predicted_downtime_hours=predicted_downtime,
            actual_downtime_hours=actual_downtime,
            downtime_error_hours=downtime_error,
            accuracy_score=accuracy_score,
            confidence_level=confidence
        )
    
    def validate_batch(self,
                      graph: nx.DiGraph,
                      simulation_results: Dict[str, Any]) -> List[ValidationResult]:
        """
        Validate multiple simulation results against historical incidents
        
        Args:
            graph: NetworkX directed graph
            simulation_results: Dictionary of simulation results keyed by incident_id
        
        Returns:
            List of validation results
        """
        self.logger.info(f"Validating {len(simulation_results)} simulations")
        
        validation_results = []
        
        for incident in self.historical_incidents:
            if incident.incident_id in simulation_results:
                result = self.validate_simulation(
                    graph,
                    incident,
                    simulation_results[incident.incident_id]
                )
                validation_results.append(result)
        
        return validation_results
    
    def generate_calibration_report(self,
                                   validation_results: List[ValidationResult]) -> CalibrationReport:
        """
        Generate comprehensive calibration report
        
        Args:
            validation_results: List of validation results
        
        Returns:
            CalibrationReport with metrics and recommendations
        """
        self.logger.info("Generating calibration report...")
        
        if not validation_results:
            raise ValueError("No validation results provided")
        
        # Calculate overall metrics
        accuracy_scores = [v.accuracy_score for v in validation_results]
        overall_accuracy = statistics.mean(accuracy_scores)
        
        # Calculate MAE (Mean Absolute Error)
        impact_errors = [v.impact_error for v in validation_results]
        mae = statistics.mean(impact_errors)
        
        # Calculate RMSE (Root Mean Square Error)
        squared_errors = [e ** 2 for e in impact_errors]
        rmse = statistics.mean(squared_errors) ** 0.5
        
        # Calculate precision, recall, F1
        # Using impact error threshold to determine TP/FP/FN
        true_positives = sum(1 for v in validation_results if v.impact_error < self.error_threshold)
        false_positives = sum(1 for v in validation_results if v.predicted_impact > v.actual_impact + self.error_threshold)
        false_negatives = sum(1 for v in validation_results if v.predicted_impact < v.actual_impact - self.error_threshold)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Generate recommendations
        recommendations = self._generate_calibration_recommendations(
            overall_accuracy,
            mae,
            rmse,
            validation_results
        )
        
        # Determine calibration quality
        calibration_quality = self._assess_calibration_quality(overall_accuracy, mae)
        
        return CalibrationReport(
            total_incidents=len(validation_results),
            validation_results=validation_results,
            overall_accuracy=overall_accuracy,
            mean_absolute_error=mae,
            root_mean_square_error=rmse,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            recommendations=recommendations,
            calibration_quality=calibration_quality
        )
    
    def identify_systematic_bias(self,
                                validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Identify systematic biases in predictions
        
        Args:
            validation_results: List of validation results
        
        Returns:
            Dictionary describing identified biases
        """
        if not validation_results:
            return {'bias_detected': False}
        
        # Calculate average bias
        errors = [v.predicted_impact - v.actual_impact for v in validation_results]
        avg_bias = statistics.mean(errors)
        
        # Determine bias direction
        if avg_bias > 0.1:
            bias_direction = "OVERESTIMATION"
            bias_description = "Simulations consistently overestimate impact"
        elif avg_bias < -0.1:
            bias_direction = "UNDERESTIMATION"
            bias_description = "Simulations consistently underestimate impact"
        else:
            bias_direction = "NONE"
            bias_description = "No systematic bias detected"
        
        # Calculate bias magnitude
        bias_magnitude = abs(avg_bias)
        
        # Identify patterns
        overestimations = sum(1 for e in errors if e > 0.1)
        underestimations = sum(1 for e in errors if e < -0.1)
        
        return {
            'bias_detected': bias_magnitude > 0.1,
            'bias_direction': bias_direction,
            'bias_magnitude': round(bias_magnitude, 3),
            'bias_description': bias_description,
            'overestimation_count': overestimations,
            'underestimation_count': underestimations,
            'total_predictions': len(validation_results)
        }
    
    def recommend_parameter_adjustments(self,
                                       validation_results: List[ValidationResult]) -> List[Dict[str, Any]]:
        """
        Recommend parameter adjustments based on validation results
        
        Args:
            validation_results: List of validation results
        
        Returns:
            List of parameter adjustment recommendations
        """
        recommendations = []
        
        # Identify bias
        bias_info = self.identify_systematic_bias(validation_results)
        
        if bias_info['bias_detected']:
            if bias_info['bias_direction'] == 'OVERESTIMATION':
                recommendations.append({
                    'parameter': 'impact_weights',
                    'current_issue': 'Overestimating impact',
                    'suggested_action': 'Reduce impact weights by 10-20%',
                    'expected_improvement': 'More accurate impact predictions'
                })
            elif bias_info['bias_direction'] == 'UNDERESTIMATION':
                recommendations.append({
                    'parameter': 'impact_weights',
                    'current_issue': 'Underestimating impact',
                    'suggested_action': 'Increase impact weights by 10-20%',
                    'expected_improvement': 'More accurate impact predictions'
                })
        
        # Check downtime estimation accuracy
        downtime_errors = [v.downtime_error_hours for v in validation_results]
        avg_downtime_error = statistics.mean(downtime_errors)
        
        if avg_downtime_error > 1.0:
            recommendations.append({
                'parameter': 'recovery_time_model',
                'current_issue': f'Average downtime error: {avg_downtime_error:.2f} hours',
                'suggested_action': 'Review recovery time estimation model',
                'expected_improvement': 'Better downtime predictions'
            })
        
        # Check affected component accuracy
        affected_errors = [v.affected_error for v in validation_results]
        avg_affected_error = statistics.mean(affected_errors)
        
        if avg_affected_error > 3:
            recommendations.append({
                'parameter': 'cascade_propagation',
                'current_issue': f'Average affected component error: {avg_affected_error:.1f}',
                'suggested_action': 'Adjust cascade propagation parameters',
                'expected_improvement': 'Better prediction of affected components'
            })
        
        return recommendations
    
    def calculate_confidence_score(self,
                                  graph: nx.DiGraph,
                                  component: str) -> float:
        """
        Calculate confidence score for predictions about a component
        
        Args:
            graph: NetworkX directed graph
            component: Component to assess
        
        Returns:
            Confidence score (0.0-1.0)
        """
        # Base confidence from validation history
        base_confidence = 0.5
        
        # Adjust based on historical accuracy for similar components
        component_type = graph.nodes[component].get('type', 'Unknown')
        
        # Filter validation results for similar components
        relevant_validations = []
        for incident in self.historical_incidents:
            if component in incident.failed_components:
                # This component was involved in historical incident
                base_confidence += 0.1
        
        # Adjust based on graph connectivity (well-connected = higher confidence)
        degree = graph.degree(component)
        avg_degree = sum(dict(graph.degree()).values()) / len(graph) if len(graph) > 0 else 1
        connectivity_factor = min(1.0, degree / avg_degree)
        
        confidence = base_confidence * (0.7 + 0.3 * connectivity_factor)
        
        return min(1.0, confidence)
    
    def generate_validation_report(self,
                                  validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Generate comprehensive validation report
        
        Args:
            validation_results: List of validation results
        
        Returns:
            Detailed validation report
        """
        calibration_report = self.generate_calibration_report(validation_results)
        bias_info = self.identify_systematic_bias(validation_results)
        adjustments = self.recommend_parameter_adjustments(validation_results)
        
        # Calculate additional statistics
        high_accuracy_count = sum(1 for v in validation_results if v.accuracy_score > 0.8)
        low_accuracy_count = sum(1 for v in validation_results if v.accuracy_score < 0.5)
        
        return {
            'summary': {
                'total_validations': len(validation_results),
                'overall_accuracy': calibration_report.overall_accuracy,
                'calibration_quality': calibration_report.calibration_quality,
                'high_accuracy_predictions': high_accuracy_count,
                'low_accuracy_predictions': low_accuracy_count
            },
            'metrics': {
                'mean_absolute_error': calibration_report.mean_absolute_error,
                'root_mean_square_error': calibration_report.root_mean_square_error,
                'precision': calibration_report.precision,
                'recall': calibration_report.recall,
                'f1_score': calibration_report.f1_score
            },
            'bias_analysis': bias_info,
            'recommendations': adjustments,
            'detailed_results': [v.to_dict() for v in validation_results]
        }
    
    def _estimate_downtime_from_impact(self, impact_score: float) -> float:
        """Estimate downtime hours from impact score"""
        # Simple linear model: impact 0.5 = 1 hour, impact 1.0 = 4 hours
        return impact_score * 4.0
    
    def _calculate_confidence_level(self, accuracy_score: float) -> str:
        """Calculate confidence level from accuracy score"""
        if accuracy_score >= 0.9:
            return "VERY_HIGH"
        elif accuracy_score >= 0.75:
            return "HIGH"
        elif accuracy_score >= 0.6:
            return "MEDIUM"
        elif accuracy_score >= 0.4:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _generate_calibration_recommendations(self,
                                            accuracy: float,
                                            mae: float,
                                            rmse: float,
                                            results: List[ValidationResult]) -> List[str]:
        """Generate recommendations for calibration"""
        recommendations = []
        
        if accuracy < self.accuracy_threshold:
            recommendations.append(
                f"Overall accuracy ({accuracy:.1%}) is below threshold. "
                "Review impact calculation parameters."
            )
        
        if mae > self.error_threshold:
            recommendations.append(
                f"Mean absolute error ({mae:.3f}) exceeds acceptable threshold. "
                "Consider adjusting impact weights."
            )
        
        if rmse > self.error_threshold * 1.5:
            recommendations.append(
                f"High RMSE ({rmse:.3f}) indicates some large errors. "
                "Investigate outliers in predictions."
            )
        
        # Check for consistent patterns
        errors = [v.impact_error for v in results]
        if statistics.stdev(errors) > 0.3:
            recommendations.append(
                "High variability in prediction errors. "
                "Consider component-specific calibration."
            )
        
        if not recommendations:
            recommendations.append(
                "Calibration quality is good. Continue monitoring."
            )
        
        return recommendations
    
    def _assess_calibration_quality(self, accuracy: float, mae: float) -> str:
        """Assess overall calibration quality"""
        if accuracy >= 0.85 and mae < 0.15:
            return "EXCELLENT"
        elif accuracy >= 0.75 and mae < 0.25:
            return "GOOD"
        elif accuracy >= 0.65 and mae < 0.35:
            return "ACCEPTABLE"
        elif accuracy >= 0.5:
            return "POOR"
        else:
            return "INADEQUATE"
