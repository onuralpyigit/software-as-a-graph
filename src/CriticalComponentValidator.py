# CriticalityValidationFramework.py
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import logging
from dataclasses import dataclass

@dataclass
class ValidationMetrics:
    """Container for validation metrics"""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    false_positive_rate: float
    false_negative_rate: float
    auc_roc: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None

class CriticalityValidator:
    """
    Comprehensive validation framework for critical component identification
    in distributed publish-subscribe systems
    """
    
    def __init__(self, graph_analyzer, reachability_analyzer):
        self.graph_analyzer = graph_analyzer
        self.reachability_analyzer = reachability_analyzer
        self.logger = logging.getLogger(__name__)
        
        # Storage for validation results
        self.validation_results = {
            'historical': {},
            'simulation': {},
            'expert': {},
            'cross_validation': {},
            'sensitivity': {}
        }
    
    def validate_comprehensive(self, 
                              graph: nx.DiGraph,
                              criticality_scores: pd.DataFrame,
                              historical_failures: Optional[pd.DataFrame] = None,
                              expert_labels: Optional[Dict] = None,
                              simulation_count: int = 100) -> Dict[str, Any]:
        """
        Perform comprehensive validation of criticality scores
        
        Args:
            graph: System graph
            criticality_scores: DataFrame with criticality analysis results
            historical_failures: Historical failure data (if available)
            expert_labels: Expert-labeled critical components (if available)
            simulation_count: Number of failure simulations to run
            
        Returns:
            Comprehensive validation results
        """
        self.logger.info("Starting comprehensive validation...")
        
        # 1. Historical Validation (if data available)
        if historical_failures is not None:
            self.validation_results['historical'] = self.validate_against_historical_data(
                criticality_scores, historical_failures
            )
        
        # 2. Simulation-based Validation
        self.validation_results['simulation'] = self.validate_through_simulation(
            graph, criticality_scores, simulation_count
        )
        
        # 3. Expert Validation (if labels available)
        if expert_labels is not None:
            self.validation_results['expert'] = self.validate_against_expert_labels(
                criticality_scores, expert_labels
            )
        
        # 4. Cross-validation
        self.validation_results['cross_validation'] = self.perform_cross_validation(
            graph, criticality_scores
        )
        
        # 5. Sensitivity Analysis
        self.validation_results['sensitivity'] = self.perform_sensitivity_analysis(
            graph, criticality_scores
        )
        
        # 6. Generate validation report
        validation_report = self.generate_validation_report()
        
        return validation_report
    
    def validate_against_historical_data(self, 
                                        criticality_scores: pd.DataFrame,
                                        historical_failures: pd.DataFrame) -> Dict:
        """
        Validate criticality scores against historical failure data
        
        Args:
            criticality_scores: Predicted criticality scores
            historical_failures: Historical failure records with columns:
                - component: Component identifier
                - failure_time: Timestamp of failure
                - impact_score: Measured impact (0-1)
                - recovery_time: Time to recover (minutes)
                
        Returns:
            Validation metrics against historical data
        """
        self.logger.info("Validating against historical failure data...")
        
        # Merge predictions with historical data
        merged = pd.merge(
            criticality_scores,
            historical_failures,
            left_on='node',
            right_on='component',
            how='inner'
        )
        
        if len(merged) == 0:
            self.logger.warning("No matching components in historical data")
            return {}
        
        # Calculate correlation between predicted criticality and actual impact
        correlation = merged['composite_score'].corr(merged['impact_score'])
        
        # Define threshold for critical components (top 20%)
        threshold = criticality_scores['composite_score'].quantile(0.8)
        
        # Binary classification
        predicted_critical = (merged['composite_score'] >= threshold).astype(int)
        actual_critical = (merged['impact_score'] >= 0.5).astype(int)  # Impact > 0.5 considered critical
        
        # Calculate metrics
        metrics = self._calculate_validation_metrics(actual_critical, predicted_critical)
        
        # Analyze failure patterns
        failure_patterns = self._analyze_failure_patterns(merged)
        
        return {
            'correlation': correlation,
            'metrics': metrics.__dict__,
            'failure_patterns': failure_patterns,
            'sample_size': len(merged)
        }
    
    def _analyze_failure_patterns(self, merged_data: pd.DataFrame) -> Dict:
        """Analyze patterns in historical failures"""
        patterns = {}
        
        # Group by component type if available
        if 'type' in merged_data.columns:
            type_analysis = merged_data.groupby('type').agg({
                'composite_score': 'mean',
                'impact_score': 'mean',
                'recovery_time': 'mean'
            }).to_dict('index')
            patterns['by_type'] = type_analysis
        
        # Identify components with repeated failures
        failure_counts = merged_data['component'].value_counts()
        patterns['repeated_failures'] = failure_counts[failure_counts > 1].to_dict()
        
        # Correlation between criticality and recovery time
        if 'recovery_time' in merged_data.columns:
            recovery_correlation = merged_data['composite_score'].corr(
                merged_data['recovery_time']
            )
            patterns['recovery_correlation'] = recovery_correlation
        
        return patterns
    
    def validate_through_simulation(self,
                                   graph: nx.DiGraph,
                                   criticality_scores: pd.DataFrame,
                                   simulation_count: int = 100) -> Dict:
        """
        Validate criticality scores through failure simulation
        
        Args:
            graph: System graph
            criticality_scores: Predicted criticality scores
            simulation_count: Number of simulations to run
            
        Returns:
            Simulation-based validation results
        """
        self.logger.info(f"Running {simulation_count} failure simulations...")
        
        # Sort components by criticality score
        sorted_components = criticality_scores.sort_values(
            'composite_score', ascending=False
        )
        
        # Select top N critical and random components for comparison
        top_n = min(10, len(sorted_components) // 5)
        critical_components = sorted_components.head(top_n)['node'].tolist()
        random_components = sorted_components.sample(
            min(top_n, len(sorted_components))
        )['node'].tolist()
        
        # Run simulations
        critical_impacts = []
        random_impacts = []
        
        for _ in range(simulation_count):
            # Simulate critical component failure
            if critical_components:
                comp = np.random.choice(critical_components)
                impact = self._simulate_failure_impact(graph, comp)
                critical_impacts.append(impact)
            
            # Simulate random component failure
            if random_components:
                comp = np.random.choice(random_components)
                impact = self._simulate_failure_impact(graph, comp)
                random_impacts.append(impact)
        
        # Statistical comparison
        critical_mean = np.mean(critical_impacts) if critical_impacts else 0
        random_mean = np.mean(random_impacts) if random_impacts else 0
        
        # Perform t-test
        from scipy import stats
        if critical_impacts and random_impacts:
            t_stat, p_value = stats.ttest_ind(critical_impacts, random_impacts)
        else:
            t_stat, p_value = 0, 1
        
        # Calculate effect size (Cohen's d)
        if critical_impacts and random_impacts:
            pooled_std = np.sqrt((np.var(critical_impacts) + np.var(random_impacts)) / 2)
            effect_size = (critical_mean - random_mean) / pooled_std if pooled_std > 0 else 0
        else:
            effect_size = 0
        
        return {
            'critical_impact_mean': critical_mean,
            'random_impact_mean': random_mean,
            'impact_difference': critical_mean - random_mean,
            'statistical_significance': {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            },
            'effect_size': effect_size,
            'simulation_count': simulation_count
        }
    
    def _simulate_failure_impact(self, graph: nx.DiGraph, component: str) -> float:
        """Simulate failure impact for a single component"""
        if component not in graph:
            return 0.0
        
        # Use reachability analyzer for impact calculation
        original_reachability = self.reachability_analyzer.compute_reachability_matrix()
        new_reachability = self.reachability_analyzer.simulate_component_failure(
            component, 'complete'
        )
        
        impact = self.reachability_analyzer.calculate_reachability_impact(
            original_reachability, new_reachability
        )
        
        # Combine multiple impact factors
        reachability_loss = impact['reachability_loss'] / 100
        isolated_ratio = impact['isolated_apps'] / len(original_reachability) if len(original_reachability) > 0 else 0
        
        # Weighted impact score
        total_impact = 0.6 * reachability_loss + 0.4 * isolated_ratio
        
        return total_impact
    
    def validate_against_expert_labels(self,
                                      criticality_scores: pd.DataFrame,
                                      expert_labels: Dict[str, str]) -> Dict:
        """
        Validate against expert-provided labels
        
        Args:
            criticality_scores: Predicted criticality scores
            expert_labels: Dict mapping component to criticality level
                          ('CRITICAL', 'HIGH', 'MEDIUM', 'LOW')
        
        Returns:
            Validation metrics against expert labels
        """
        self.logger.info("Validating against expert labels...")
        
        # Map expert labels to binary
        expert_binary = {
            comp: 1 if level in ['CRITICAL', 'HIGH'] else 0
            for comp, level in expert_labels.items()
        }
        
        # Get predictions for labeled components
        labeled_scores = criticality_scores[
            criticality_scores['node'].isin(expert_binary.keys())
        ].copy()
        
        if len(labeled_scores) == 0:
            self.logger.warning("No matching components in expert labels")
            return {}
        
        # Define threshold
        threshold = labeled_scores['composite_score'].quantile(0.7)
        
        # Create binary predictions
        labeled_scores['predicted'] = (
            labeled_scores['composite_score'] >= threshold
        ).astype(int)
        labeled_scores['actual'] = labeled_scores['node'].map(expert_binary)
        
        # Calculate metrics
        metrics = self._calculate_validation_metrics(
            labeled_scores['actual'],
            labeled_scores['predicted']
        )
        
        # Calculate agreement by component type
        agreement_by_type = {}
        if 'type' in labeled_scores.columns:
            for comp_type in labeled_scores['type'].unique():
                type_data = labeled_scores[labeled_scores['type'] == comp_type]
                if len(type_data) > 0:
                    agreement = (type_data['predicted'] == type_data['actual']).mean()
                    agreement_by_type[comp_type] = agreement
        
        return {
            'metrics': metrics.__dict__,
            'agreement_by_type': agreement_by_type,
            'sample_size': len(labeled_scores),
            'threshold_used': threshold
        }
    
    def perform_cross_validation(self,
                                graph: nx.DiGraph,
                                criticality_scores: pd.DataFrame,
                                k_folds: int = 5) -> Dict:
        """
        Perform k-fold cross-validation
        
        Args:
            graph: System graph
            criticality_scores: Criticality scores
            k_folds: Number of folds
            
        Returns:
            Cross-validation results
        """
        self.logger.info(f"Performing {k_folds}-fold cross-validation...")
        
        # Prepare data
        components = criticality_scores['node'].values
        scores = criticality_scores['composite_score'].values
        
        # Generate synthetic labels based on impact simulation
        labels = []
        for comp in components:
            if comp in graph:
                impact = self._simulate_failure_impact(graph, comp)
                labels.append(1 if impact > 0.5 else 0)
            else:
                labels.append(0)
        
        labels = np.array(labels)
        
        # Perform k-fold cross-validation
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        cv_results = []
        for fold, (train_idx, test_idx) in enumerate(kf.split(scores)):
            # Split data
            train_scores = scores[train_idx]
            test_scores = scores[test_idx]
            train_labels = labels[train_idx]
            test_labels = labels[test_idx]
            
            # Find optimal threshold on training set
            threshold = self._find_optimal_threshold(train_scores, train_labels)
            
            # Predict on test set
            test_predictions = (test_scores >= threshold).astype(int)
            
            # Calculate metrics
            metrics = self._calculate_validation_metrics(test_labels, test_predictions)
            cv_results.append({
                'fold': fold + 1,
                'threshold': threshold,
                'metrics': metrics.__dict__
            })
        
        # Aggregate results
        aggregated = self._aggregate_cv_results(cv_results)
        
        return {
            'fold_results': cv_results,
            'aggregated': aggregated
        }
    
    def _find_optimal_threshold(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """Find optimal threshold using F1 score"""
        best_threshold = 0
        best_f1 = 0
        
        for threshold in np.percentile(scores, range(50, 95, 5)):
            predictions = (scores >= threshold).astype(int)
            metrics = self._calculate_validation_metrics(labels, predictions)
            if metrics.f1_score > best_f1:
                best_f1 = metrics.f1_score
                best_threshold = threshold
        
        return best_threshold
    
    def _aggregate_cv_results(self, cv_results: List[Dict]) -> Dict:
        """Aggregate cross-validation results"""
        metrics_list = [r['metrics'] for r in cv_results]
        
        aggregated = {}
        for metric in ['precision', 'recall', 'f1_score', 'accuracy']:
            values = [m[metric] for m in metrics_list]
            aggregated[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return aggregated
    
    def perform_sensitivity_analysis(self,
                                    graph: nx.DiGraph,
                                    criticality_scores: pd.DataFrame) -> Dict:
        """
        Perform sensitivity analysis on criticality scoring
        
        Args:
            graph: System graph
            criticality_scores: Original criticality scores
            
        Returns:
            Sensitivity analysis results
        """
        self.logger.info("Performing sensitivity analysis...")
        
        # Define weight variations
        weight_variations = [
            {'name': 'baseline', 'weights': self.graph_analyzer.criticality_weights.copy()},
            {'name': 'structure_focused', 'weights': {
                'degree_centrality': 0.20,
                'betweenness_centrality': 0.30,
                'closeness_centrality': 0.20,
                'pagerank': 0.20,
                'eigenvector_centrality': 0.10,
                'articulation_point': 0.00,
                'qos_score': 0.00
            }},
            {'name': 'qos_focused', 'weights': {
                'degree_centrality': 0.05,
                'betweenness_centrality': 0.10,
                'closeness_centrality': 0.05,
                'pagerank': 0.10,
                'eigenvector_centrality': 0.10,
                'articulation_point': 0.10,
                'qos_score': 0.50
            }},
            {'name': 'balanced', 'weights': {
                'degree_centrality': 0.14,
                'betweenness_centrality': 0.14,
                'closeness_centrality': 0.14,
                'pagerank': 0.14,
                'eigenvector_centrality': 0.14,
                'articulation_point': 0.15,
                'qos_score': 0.15
            }}
        ]
        
        sensitivity_results = []
        
        for variation in weight_variations:
            # Recalculate scores with new weights
            self.graph_analyzer.criticality_weights = variation['weights']
            new_scores = self.graph_analyzer.analyze_comprehensive_criticality(graph)
            
            # Compare rankings
            original_top10 = set(criticality_scores.head(10)['node'].values)
            new_top10 = set(new_scores.head(10)['node'].values)
            
            overlap = len(original_top10.intersection(new_top10))
            
            # Calculate rank correlation
            merged = pd.merge(
                criticality_scores[['node', 'composite_score']],
                new_scores[['node', 'composite_score']],
                on='node',
                suffixes=('_original', '_new')
            )
            
            rank_correlation = merged['composite_score_original'].corr(
                merged['composite_score_new'],
                method='spearman'
            )
            
            sensitivity_results.append({
                'variation': variation['name'],
                'weights': variation['weights'],
                'top10_overlap': overlap,
                'rank_correlation': rank_correlation
            })
        
        # Reset to original weights
        self.graph_analyzer.criticality_weights = weight_variations[0]['weights']
        
        return {
            'variations': sensitivity_results,
            'stability_score': np.mean([r['rank_correlation'] for r in sensitivity_results])
        }
    
    def _calculate_validation_metrics(self, 
                                     actual: np.ndarray,
                                     predicted: np.ndarray) -> ValidationMetrics:
        """Calculate comprehensive validation metrics"""
        # Basic metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual, predicted, average='binary', zero_division=0
        )
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(actual, predicted).ravel() if len(np.unique(actual)) > 1 else (0, 0, 0, len(actual))
        
        # Calculate rates
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # AUC-ROC if possible
        try:
            auc = roc_auc_score(actual, predicted) if len(np.unique(actual)) > 1 else None
        except:
            auc = None
        
        return ValidationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            false_positive_rate=fpr,
            false_negative_rate=fnr,
            auc_roc=auc
        )
    
    def generate_validation_report(self) -> Dict:
        """Generate comprehensive validation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_methods': []
        }
        
        # Historical validation
        if self.validation_results['historical']:
            hist = self.validation_results['historical']
            report['validation_methods'].append({
                'method': 'Historical Data Validation',
                'status': 'completed',
                'metrics': hist.get('metrics', {}),
                'correlation': hist.get('correlation', 0),
                'sample_size': hist.get('sample_size', 0),
                'insights': self._generate_historical_insights(hist)
            })
        
        # Simulation validation
        if self.validation_results['simulation']:
            sim = self.validation_results['simulation']
            report['validation_methods'].append({
                'method': 'Simulation-based Validation',
                'status': 'completed',
                'impact_difference': sim['impact_difference'],
                'statistical_significance': sim['statistical_significance'],
                'effect_size': sim['effect_size'],
                'insights': self._generate_simulation_insights(sim)
            })
        
        # Expert validation
        if self.validation_results['expert']:
            exp = self.validation_results['expert']
            report['validation_methods'].append({
                'method': 'Expert Label Validation',
                'status': 'completed',
                'metrics': exp.get('metrics', {}),
                'agreement_by_type': exp.get('agreement_by_type', {}),
                'insights': self._generate_expert_insights(exp)
            })
        
        # Cross-validation
        if self.validation_results['cross_validation']:
            cv = self.validation_results['cross_validation']
            report['validation_methods'].append({
                'method': 'Cross-validation',
                'status': 'completed',
                'aggregated_metrics': cv.get('aggregated', {}),
                'insights': self._generate_cv_insights(cv)
            })
        
        # Sensitivity analysis
        if self.validation_results['sensitivity']:
            sens = self.validation_results['sensitivity']
            report['validation_methods'].append({
                'method': 'Sensitivity Analysis',
                'status': 'completed',
                'stability_score': sens['stability_score'],
                'variations': sens['variations'],
                'insights': self._generate_sensitivity_insights(sens)
            })
        
        # Overall confidence score
        report['overall_confidence'] = self._calculate_overall_confidence()
        report['recommendations'] = self._generate_validation_recommendations()
        
        return report
    
    def _generate_historical_insights(self, results: Dict) -> List[str]:
        """Generate insights from historical validation"""
        insights = []
        
        if 'correlation' in results:
            corr = results['correlation']
            if corr > 0.7:
                insights.append(f"Strong correlation ({corr:.2f}) with historical failures")
            elif corr > 0.4:
                insights.append(f"Moderate correlation ({corr:.2f}) with historical failures")
            else:
                insights.append(f"Weak correlation ({corr:.2f}) with historical failures")
        
        if 'metrics' in results:
            metrics = results['metrics']
            if metrics.get('f1_score', 0) > 0.8:
                insights.append("Excellent prediction accuracy against historical data")
            elif metrics.get('f1_score', 0) > 0.6:
                insights.append("Good prediction accuracy against historical data")
        
        return insights
    
    def _generate_simulation_insights(self, results: Dict) -> List[str]:
        """Generate insights from simulation validation"""
        insights = []
        
        diff = results['impact_difference']
        if diff > 0.3:
            insights.append(f"Critical components have {diff:.1%} higher impact than random")
        
        if results['statistical_significance']['significant']:
            insights.append("Statistical significance confirmed (p < 0.05)")
        
        effect = results['effect_size']
        if abs(effect) > 0.8:
            insights.append("Large effect size indicates strong differentiation")
        elif abs(effect) > 0.5:
            insights.append("Medium effect size indicates moderate differentiation")
        
        return insights
    
    def _generate_expert_insights(self, results: Dict) -> List[str]:
        """Generate insights from expert validation"""
        insights = []
        
        if 'metrics' in results:
            metrics = results['metrics']
            agreement = metrics.get('accuracy', 0)
            if agreement > 0.8:
                insights.append(f"High agreement ({agreement:.1%}) with expert assessment")
            elif agreement > 0.6:
                insights.append(f"Moderate agreement ({agreement:.1%}) with expert assessment")
        
        if 'agreement_by_type' in results:
            best_type = max(results['agreement_by_type'].items(), 
                          key=lambda x: x[1], 
                          default=(None, 0))
            if best_type[0]:
                insights.append(f"Best agreement for {best_type[0]} components")
        
        return insights
    
    def _generate_cv_insights(self, results: Dict) -> List[str]:
        """Generate insights from cross-validation"""
        insights = []
        
        if 'aggregated' in results:
            agg = results['aggregated']
            f1_mean = agg.get('f1_score', {}).get('mean', 0)
            f1_std = agg.get('f1_score', {}).get('std', 0)
            
            if f1_mean > 0.7:
                insights.append(f"Consistent performance (F1: {f1_mean:.2f}±{f1_std:.2f})")
            
            if f1_std < 0.05:
                insights.append("Low variance indicates stable predictions")
        
        return insights
    
    def _generate_sensitivity_insights(self, results: Dict) -> List[str]:
        """Generate insights from sensitivity analysis"""
        insights = []
        
        stability = results['stability_score']
        if stability > 0.9:
            insights.append("Very stable rankings across weight variations")
        elif stability > 0.7:
            insights.append("Reasonably stable rankings across weight variations")
        else:
            insights.append("Rankings sensitive to weight changes")
        
        return insights
    
    def _calculate_overall_confidence(self) -> float:
        """Calculate overall confidence score for the validation"""
        confidence_scores = []
        
        # Historical validation confidence
        if self.validation_results['historical']:
            hist = self.validation_results['historical']
            if 'metrics' in hist:
                confidence_scores.append(hist['metrics'].get('f1_score', 0))
        
        # Simulation validation confidence
        if self.validation_results['simulation']:
            sim = self.validation_results['simulation']
            if sim['statistical_significance']['significant']:
                confidence_scores.append(min(1.0, abs(sim['effect_size'])))
        
        # Expert validation confidence
        if self.validation_results['expert']:
            exp = self.validation_results['expert']
            if 'metrics' in exp:
                confidence_scores.append(exp['metrics'].get('accuracy', 0))
        
        # Cross-validation confidence
        if self.validation_results['cross_validation']:
            cv = self.validation_results['cross_validation']
            if 'aggregated' in cv:
                f1_mean = cv['aggregated'].get('f1_score', {}).get('mean', 0)
                confidence_scores.append(f1_mean)
        
        # Sensitivity analysis confidence
        if self.validation_results['sensitivity']:
            sens = self.validation_results['sensitivity']
            confidence_scores.append(sens['stability_score'])
        
        # Return average confidence
        return np.mean(confidence_scores) if confidence_scores else 0.0
    
    def _generate_validation_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        confidence = self._calculate_overall_confidence()
        
        if confidence < 0.6:
            recommendations.append("Consider collecting more historical failure data")
            recommendations.append("Increase expert validation sample size")
            recommendations.append("Review and adjust criticality scoring weights")
        elif confidence < 0.8:
            recommendations.append("Methodology shows promise but needs refinement")
            recommendations.append("Consider domain-specific weight adjustments")
        else:
            recommendations.append("Methodology validated with high confidence")
            recommendations.append("Ready for production deployment")
        
        # Specific recommendations based on results
        if self.validation_results['sensitivity']:
            stability = self.validation_results['sensitivity']['stability_score']
            if stability < 0.7:
                recommendations.append("Consider using ensemble methods for more stable results")
        
        if self.validation_results['simulation']:
            effect = self.validation_results['simulation']['effect_size']
            if abs(effect) < 0.5:
                recommendations.append("Enhance differentiation between critical and non-critical components")
        
        return recommendations
    
    def visualize_validation_results(self, output_dir: str = "output/"):
        """Create visualizations for validation results"""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Criticality Validation Results', fontsize=16)
        
        # 1. Historical Validation Metrics
        if self.validation_results['historical'] and 'metrics' in self.validation_results['historical']:
            metrics = self.validation_results['historical']['metrics']
            ax = axes[0, 0]
            metrics_df = pd.DataFrame([metrics])
            metrics_df[['precision', 'recall', 'f1_score', 'accuracy']].plot(
                kind='bar', ax=ax, legend=False
            )
            ax.set_title('Historical Validation Metrics')
            ax.set_ylim([0, 1])
            ax.set_xticklabels(['Metrics'], rotation=0)
            ax.legend(loc='lower right')
        
        # 2. Simulation Impact Comparison
        if self.validation_results['simulation']:
            sim = self.validation_results['simulation']
            ax = axes[0, 1]
            impacts = [sim['critical_impact_mean'], sim['random_impact_mean']]
            labels = ['Critical', 'Random']
            ax.bar(labels, impacts, color=['red', 'blue'])
            ax.set_title('Average Failure Impact')
            ax.set_ylabel('Impact Score')
        
        # 3. Cross-validation Results
        if self.validation_results['cross_validation'] and 'fold_results' in self.validation_results['cross_validation']:
           cv = self.validation_results['cross_validation']
           ax = axes[0, 2]
           fold_metrics = pd.DataFrame([r['metrics'] for r in cv['fold_results']])
           fold_metrics['fold'] = range(1, len(fold_metrics) + 1)
           fold_metrics.plot(x='fold', y='f1_score', ax=ax, marker='o')
           ax.set_title('Cross-validation F1 Scores')
           ax.set_xlabel('Fold')
           ax.set_ylabel('F1 Score')
           ax.set_ylim([0, 1])
        
       
        # 4. Expert Agreement by Type
        if self.validation_results['expert'] and 'agreement_by_type' in self.validation_results['expert']:
            ax = axes[1, 0]
            agreement = self.validation_results['expert']['agreement_by_type']
            if agreement:
                pd.Series(agreement).plot(kind='bar', ax=ax)
                ax.set_title('Expert Agreement by Component Type')
                ax.set_ylabel('Agreement Rate')
                ax.set_ylim([0, 1])
                ax.tick_params(axis='x', rotation=45)
       
        # 5. Sensitivity Analysis
        if self.validation_results['sensitivity'] and 'variations' in self.validation_results['sensitivity']:
            ax = axes[1, 1]
            variations = self.validation_results['sensitivity']['variations']
            var_names = [v['variation'] for v in variations]
            correlations = [v['rank_correlation'] for v in variations]
            ax.bar(var_names, correlations)
            ax.set_title('Sensitivity to Weight Changes')
            ax.set_ylabel('Rank Correlation')
            ax.set_ylim([0, 1])
            ax.tick_params(axis='x', rotation=45)
        
        # 6. Overall Confidence
        ax = axes[1, 2]
        confidence = self._calculate_overall_confidence()
        ax.pie([confidence, 1-confidence], 
                labels=['Confident', 'Uncertain'],
                colors=['green', 'lightgray'],
                autopct='%1.1f%%',
                startangle=90)
        ax.set_title(f'Overall Confidence: {confidence:.1%}')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/validation_results.png", dpi=300, bbox_inches='tight')
        plt.show()
    
class RealWorldValidator:
   """
   Specialized validator for real-world system validation
   """
   
   def __init__(self):
       self.logger = logging.getLogger(__name__)
       
   def create_synthetic_failure_data(self, 
                                    graph: nx.DiGraph,
                                    num_failures: int = 100,
                                    time_period_days: int = 365) -> pd.DataFrame:
       """
       Create synthetic failure data for validation when historical data is unavailable
       
       Args:
           graph: System graph
           num_failures: Number of failures to generate
           time_period_days: Time period to simulate
           
       Returns:
           DataFrame with synthetic failure data
       """
       self.logger.info(f"Generating {num_failures} synthetic failures...")
       
       failures = []
       nodes = list(graph.nodes())
       
       # Get node types
       node_types = nx.get_node_attributes(graph, 'type')
       
       # Define failure probabilities by type
       failure_probs = {
           'Application': 0.3,
           'Broker': 0.2,
           'Node': 0.1,
           'Topic': 0.4  # Topics don't really fail but can have issues
       }
       
       # Generate failures
       base_time = datetime.now() - timedelta(days=time_period_days)
       
       for i in range(num_failures):
           # Select component based on type probabilities
           component = np.random.choice(nodes)
           comp_type = node_types.get(component, 'Unknown')
           
           # Skip with probability based on type
           if np.random.random() > failure_probs.get(comp_type, 0.5):
               continue
           
           # Calculate impact based on graph properties
           impact = self._calculate_synthetic_impact(graph, component)
           
           # Generate recovery time (correlated with impact)
           recovery_time = np.random.exponential(30 * (1 + impact))
           
           # Generate timestamp
           failure_time = base_time + timedelta(
               days=np.random.uniform(0, time_period_days)
           )
           
           failures.append({
               'component': component,
               'failure_time': failure_time,
               'impact_score': impact,
               'recovery_time': recovery_time,
               'type': comp_type
           })
       
       return pd.DataFrame(failures)
   
   def _calculate_synthetic_impact(self, graph: nx.DiGraph, component: str) -> float:
       """Calculate synthetic impact score for a component"""
       if component not in graph:
           return 0.0
       
       # Use various graph metrics to estimate impact
       try:
           degree = graph.degree(component)
           betweenness = nx.betweenness_centrality(graph).get(component, 0)
           
           # Check if articulation point
           undirected = graph.to_undirected()
           is_articulation = component in nx.articulation_points(undirected)
           
           # Calculate synthetic impact
           impact = (
               0.3 * min(degree / 10, 1.0) +  # Normalized degree
               0.4 * betweenness +
               0.3 * (1.0 if is_articulation else 0.0)
           )
           
           # Add noise
           impact += np.random.normal(0, 0.1)
           
           return max(0, min(1, impact))
       except:
           return np.random.uniform(0.1, 0.5)
   
   def generate_expert_labels(self,
                             graph: nx.DiGraph,
                             sample_size: int = 50) -> Dict[str, str]:
       """
       Generate synthetic expert labels for validation
       
       Args:
           graph: System graph
           sample_size: Number of components to label
           
       Returns:
           Dictionary of expert labels
       """
       self.logger.info(f"Generating synthetic expert labels for {sample_size} components...")
       
       # Calculate actual criticality for synthetic labeling
       centralities = nx.betweenness_centrality(graph)
       
       # Sample components
       all_nodes = list(graph.nodes())
       sample_nodes = np.random.choice(
           all_nodes, 
           min(sample_size, len(all_nodes)), 
           replace=False
       )
       
       labels = {}
       for node in sample_nodes:
           centrality = centralities.get(node, 0)
           
           # Assign label based on centrality with some noise
           noise = np.random.normal(0, 0.1)
           score = centrality + noise
           
           if score > 0.5:
               label = 'CRITICAL'
           elif score > 0.3:
               label = 'HIGH'
           elif score > 0.15:
               label = 'MEDIUM'
           else:
               label = 'LOW'
           
           labels[node] = label
       
       return labels
   
   def validate_against_monitoring_data(self,
                                       criticality_scores: pd.DataFrame,
                                       monitoring_data: pd.DataFrame) -> Dict:
       """
       Validate against real-time monitoring data
       
       Args:
           criticality_scores: Predicted criticality scores
           monitoring_data: Real monitoring data with columns:
               - component: Component identifier
               - metric: Metric name (cpu, memory, latency, etc.)
               - value: Metric value
               - timestamp: Timestamp
               
       Returns:
           Validation results against monitoring data
       """
       self.logger.info("Validating against monitoring data...")
       
       # Aggregate monitoring data by component
       component_stats = monitoring_data.groupby('component').agg({
           'value': ['mean', 'std', 'max']
       }).reset_index()
       
       component_stats.columns = ['component', 'mean_load', 'std_load', 'max_load']
       
       # Merge with criticality scores
       merged = pd.merge(
           criticality_scores,
           component_stats,
           left_on='node',
           right_on='component',
           how='inner'
       )
       
       if len(merged) == 0:
           return {}
       
       # Calculate correlations
       correlations = {
           'mean_load': merged['composite_score'].corr(merged['mean_load']),
           'max_load': merged['composite_score'].corr(merged['max_load']),
           'load_variance': merged['composite_score'].corr(merged['std_load'])
       }
       
       # Identify overloaded critical components
       critical_threshold = merged['composite_score'].quantile(0.8)
       load_threshold = merged['mean_load'].quantile(0.8)
       
       critical_overloaded = merged[
           (merged['composite_score'] >= critical_threshold) &
           (merged['mean_load'] >= load_threshold)
       ]
       
       return {
           'correlations': correlations,
           'critical_overloaded_count': len(critical_overloaded),
           'critical_overloaded_components': critical_overloaded['node'].tolist()
       }


class ValidationOrchestrator:
   """
   Orchestrates the complete validation process
   """
   
   def __init__(self, graph_analyzer, reachability_analyzer):
       self.graph_analyzer = graph_analyzer
       self.reachability_analyzer = reachability_analyzer
       self.validator = CriticalityValidator(graph_analyzer, reachability_analyzer)
       self.real_world_validator = RealWorldValidator()
       self.logger = logging.getLogger(__name__)
   
   def run_complete_validation(self,
                              graph: nx.DiGraph,
                              criticality_scores: pd.DataFrame,
                              use_synthetic_data: bool = True) -> Dict:
       """
       Run complete validation pipeline
       
       Args:
           graph: System graph
           criticality_scores: Criticality analysis results
           use_synthetic_data: Generate synthetic data if real data unavailable
           
       Returns:
           Complete validation results
       """
       self.logger.info("Starting complete validation pipeline...")
       
       validation_data = {}
       
       # Generate or load validation data
       if use_synthetic_data:
           self.logger.info("Using synthetic validation data...")
           validation_data['historical_failures'] = self.real_world_validator.create_synthetic_failure_data(
               graph, num_failures=200
           )
           validation_data['expert_labels'] = self.real_world_validator.generate_expert_labels(
               graph, sample_size=50
           )
       
       # Run comprehensive validation
       results = self.validator.validate_comprehensive(
           graph,
           criticality_scores,
           historical_failures=validation_data.get('historical_failures'),
           expert_labels=validation_data.get('expert_labels'),
           simulation_count=50
       )
       
       # Generate visualizations
       self.validator.visualize_validation_results()
       
       # Generate detailed report
       self._generate_detailed_report(results)
       
       return results
   
   def _generate_detailed_report(self, results: Dict):
       """Generate detailed validation report"""
       report_path = "output/validation_report.md"
       
       with open(report_path, 'w') as f:
           f.write("# Critical Component Identification Validation Report\n\n")
           f.write(f"Generated: {results['timestamp']}\n\n")
           
           # Executive Summary
           f.write("## Executive Summary\n\n")
           f.write(f"**Overall Confidence Score**: {results['overall_confidence']:.1%}\n\n")
           
           # Validation Methods
           f.write("## Validation Methods Results\n\n")
           for method in results['validation_methods']:
               f.write(f"### {method['method']}\n")
               f.write(f"**Status**: {method['status']}\n\n")
               
               # Method-specific results
               if 'metrics' in method and method['metrics']:
                   f.write("**Performance Metrics**:\n")
                   for metric, value in method['metrics'].items():
                       if isinstance(value, float):
                           f.write(f"- {metric}: {value:.3f}\n")
               
               if 'insights' in method:
                   f.write("\n**Key Insights**:\n")
                   for insight in method['insights']:
                       f.write(f"- {insight}\n")
               
               f.write("\n")
           
           # Recommendations
           f.write("## Recommendations\n\n")
           for rec in results['recommendations']:
               f.write(f"- {rec}\n")
           
           f.write("\n## Conclusion\n\n")
           confidence = results['overall_confidence']
           if confidence > 0.8:
               f.write("The critical component identification methodology has been validated with high confidence. ")
               f.write("The approach successfully identifies components that have significant impact on system reliability.")
           elif confidence > 0.6:
               f.write("The methodology shows good potential but may benefit from further refinement. ")
               f.write("Consider the recommendations above to improve accuracy.")
           else:
               f.write("The validation indicates that the methodology needs significant improvement. ")
               f.write("Additional data collection and parameter tuning are recommended.")
       
       self.logger.info(f"Validation report saved to {report_path}")
