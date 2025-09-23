# practical_validation_pipeline.py

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import networkx as nx

class ValidationPipeline:
    """Validation pipeline for your criticality analysis"""
    
    def __init__(self, graph_exporter, component_analyzer, graph_analyzer, reachability_analyzer):
        self.graph_exporter = graph_exporter
        self.component_analyzer = component_analyzer
        self.graph_analyzer = graph_analyzer
        self.reachability_analyzer = reachability_analyzer
        
    def run_complete_validation(self):
        """Run complete validation pipeline"""
        
        print("\n" + "="*70)
        print("CRITICAL COMPONENT VALIDATION PIPELINE")
        print("="*70)
        
        results = {}
        
        # 1. Basic Validation: Does high criticality = high impact?
        print("\n[1/5] Impact Correlation Validation")
        results['impact_validation'] = self.validate_impact_correlation()
        
        # 2. Consistency Validation: Are results stable?
        print("\n[2/5] Consistency Validation")
        results['consistency'] = self.validate_consistency()
        
        # 3. QoS Contribution Validation
        print("\n[3/5] QoS Contribution Validation")
        results['qos_validation'] = self.validate_qos_contribution()
        
        # 4. Comparative Validation
        print("\n[4/5] Comparative Analysis")
        results['comparison'] = self.compare_methods()
        
        # 5. Failure Cascade Validation
        print("\n[5/5] Cascade Prediction Validation")
        results['cascade'] = self.validate_cascade_predictions()
        
        # Generate summary
        self.print_validation_summary(results)
        
        return results
    
    def validate_impact_correlation(self):
        """Validate that criticality scores correlate with actual impact"""
        
        graph = self.graph_exporter.export_graph()
        
        # Get your criticality predictions
        components = self.graph_analyzer.analyze_critical_components(graph)
        classified = self.graph_analyzer.classify_critical_components(components)
        
        # Test top 10 critical components
        critical_components = [c for c, level in classified.items() 
                              if level in ['VERY_HIGH', 'HIGH']][:10]
        
        # Test bottom 10 components  
        non_critical = [c for c, level in classified.items() 
                       if level in ['LOW', 'VERY_LOW']][:10]
        
        # Measure actual impact
        critical_impacts = []
        for comp in critical_components:
            if comp in graph:
                impact = self._measure_component_impact(graph, comp)
                critical_impacts.append(impact)
        
        non_critical_impacts = []
        for comp in non_critical:
            if comp in graph:
                impact = self._measure_component_impact(graph, comp)
                non_critical_impacts.append(impact)
        
        # Statistical test
        from scipy.stats import mannwhitneyu
        
        if critical_impacts and non_critical_impacts:
            statistic, p_value = mannwhitneyu(critical_impacts, non_critical_impacts, 
                                             alternative='greater')
            
            validation_passed = p_value < 0.05
            
            result = {
                'critical_mean_impact': np.mean(critical_impacts),
                'non_critical_mean_impact': np.mean(non_critical_impacts),
                'impact_ratio': np.mean(critical_impacts) / np.mean(non_critical_impacts) if np.mean(non_critical_impacts) > 0 else float('inf'),
                'p_value': p_value,
                'validation_passed': validation_passed
            }
            
            print(f"  ✓ Critical components have {result['impact_ratio']:.2f}x higher impact")
            print(f"  ✓ Statistical significance: p={p_value:.4f}")
            
            return result
        
        return None
    
    def _measure_component_impact(self, graph, component):
        """Measure actual impact of removing a component"""
        
        # Create modified graph
        test_graph = graph.copy()
        test_graph.remove_node(component)
        
        # Measure various impacts
        impacts = {}
        
        # 1. Connectivity impact
        original_components = nx.number_weakly_connected_components(graph)
        new_components = nx.number_weakly_connected_components(test_graph)
        impacts['connectivity'] = (new_components - original_components) / original_components if original_components > 0 else 0
        
        # 2. Reachability impact (simplified)
        original_reachable = sum(1 for n in graph.nodes() 
                                for m in graph.nodes() 
                                if n != m and nx.has_path(graph, n, m))
        new_reachable = sum(1 for n in test_graph.nodes() 
                           for m in test_graph.nodes() 
                           if n != m and nx.has_path(test_graph, n, m))
        impacts['reachability'] = 1 - (new_reachable / original_reachable) if original_reachable > 0 else 0
        
        # 3. Service impact (based on node type)
        node_type = graph.nodes[component].get('type')
        if node_type == 'Broker':
            # Count affected topics
            affected = len([n for n in graph.neighbors(component) 
                          if graph.nodes[n].get('type') == 'Topic'])
            total_topics = len([n for n, d in graph.nodes(data=True) 
                              if d.get('type') == 'Topic'])
            impacts['service'] = affected / total_topics if total_topics > 0 else 0
        else:
            impacts['service'] = 0
        
        # Combined impact
        return np.mean(list(impacts.values()))
    
    def validate_consistency(self):
        """Validate consistency of results across different graph views"""
        
        results = {}
        
        # Test on different graph levels
        graphs = {
            'full': self.graph_exporter.export_graph(),
            'application': self.graph_exporter.export_graph_for_application_level_analysis(),
            'infrastructure': self.graph_exporter.export_graph_for_infrastructure_level_analysis()
        }
        
        all_scores = {}
        
        for level, graph in graphs.items():
            components = self.graph_analyzer.analyze_critical_components(graph)
            scores = {comp: data['criticality_score'] 
                     for comp, data in components}
            all_scores[level] = scores
        
        # Check consistency between levels
        if 'application' in all_scores and 'infrastructure' in all_scores:
            # Find common components
            common = set(all_scores['application'].keys()) & set(all_scores['infrastructure'].keys())
            
            if common:
                app_scores = [all_scores['application'][c] for c in common]
                infra_scores = [all_scores['infrastructure'][c] for c in common]
                
                from scipy.stats import pearsonr
                correlation, p_value = pearsonr(app_scores, infra_scores)
                
                results['cross_level_correlation'] = correlation
                results['consistency_validated'] = correlation > 0.7
                
                print(f"  ✓ Cross-level consistency: r={correlation:.3f}")
        
        # Test stability with repeated runs
        stability_scores = []
        for i in range(3):
            graph = self.graph_exporter.export_graph()
            components = self.graph_analyzer.analyze_critical_components(graph)
            scores = [data['criticality_score'] for _, data in components]
            stability_scores.append(scores)
        
        # Calculate variance
        if stability_scores:
            variances = []
            for i in range(len(stability_scores[0])):
                values = [run[i] for run in stability_scores if i < len(run)]
                if len(values) > 1:
                    variances.append(np.var(values))
            
            results['stability_variance'] = np.mean(variances)
            results['stable'] = results['stability_variance'] < 0.01
            
            print(f"  ✓ Stability variance: {results['stability_variance']:.6f}")
        
        return results
    
    def validate_qos_contribution(self):
        """Validate that QoS actually improves criticality identification"""
        
        graph = self.graph_exporter.export_graph()
        
        # Run with QoS
        components_with_qos = self.graph_analyzer.analyze_critical_components(graph)
        
        # Run without QoS (modify weights temporarily)
        original_method = self.graph_analyzer.calculate_criticality_score
        
        def no_qos_score(centrality_score, weight_centrality, 
                         articulation_point_score, weight_articulation, 
                         qos_score=None, weight_qos=0.0):
            return original_method(
                centrality_score, 0.67,  # Redistribute weight
                articulation_point_score, 0.33,
                None, 0.0  # No QoS
            )
        
        self.graph_analyzer.calculate_criticality_score = no_qos_score
        components_without_qos = self.graph_analyzer.analyze_critical_components(graph)
        self.graph_analyzer.calculate_criticality_score = original_method  # Restore
        
        # Get high-QoS components
        high_qos_components = []
        for node, data in graph.nodes(data=True):
            if data.get('type') == 'Topic':
                if (data.get('durability') == 'PERSISTENT' and 
                    data.get('reliability') == 'RELIABLE'):
                    high_qos_components.append(node)
        
        # Check if high-QoS components are ranked higher with QoS
        improvements = 0
        for comp in high_qos_components:
            score_with_qos = next((d['criticality_score'] for c, d in components_with_qos if c == comp), 0)
            score_without_qos = next((d['criticality_score'] for c, d in components_without_qos if c == comp), 0)
            
            if score_with_qos > score_without_qos:
                improvements += 1
        
        result = {
            'high_qos_components': len(high_qos_components),
            'improved_rankings': improvements,
            'improvement_rate': improvements / len(high_qos_components) if high_qos_components else 0,
            'qos_helps': improvements > len(high_qos_components) * 0.7
        }
        
        print(f"  ✓ QoS improved {result['improvement_rate']*100:.1f}% of high-QoS component rankings")
        
        return result
    
    def compare_methods(self):
        """Compare your method with baselines"""
        
        graph = self.graph_exporter.export_graph()
        
        # Establish simple ground truth (impact-based)
        ground_truth = {}
        sample_nodes = list(graph.nodes())[:20]  # Test on subset for efficiency
        
        for node in sample_nodes:
            impact = self._measure_component_impact(graph, node)
            ground_truth[node] = impact
        
        # Method 1: Your QoS-aware approach
        components_qos = self.graph_analyzer.analyze_critical_components(graph)
        scores_qos = {c: d['criticality_score'] for c, d in components_qos}
        
        # Method 2: PageRank only
        pagerank = nx.pagerank(graph)
        
        # Method 3: Degree centrality only
        degree = nx.degree_centrality(graph)
        
        # Method 4: Random baseline
        import random
        random_scores = {n: random.random() for n in graph.nodes()}
        
        # Evaluate each method
        methods = {
            'QoS-Aware (Your Method)': scores_qos,
            'PageRank Only': pagerank,
            'Degree Only': degree,
            'Random': random_scores
        }
        
        results = {}
        for method_name, scores in methods.items():
            # Calculate correlation with ground truth
            common = set(scores.keys()) & set(ground_truth.keys())
            if len(common) > 1:
                pred = [scores[c] for c in common]
                true = [ground_truth[c] for c in common]
                
                from scipy.stats import spearmanr
                correlation, p_value = spearmanr(pred, true)
                
                # Top-k accuracy
                k = 5
                top_k_pred = sorted(common, key=lambda x: scores[x], reverse=True)[:k]
                top_k_true = sorted(common, key=lambda x: ground_truth[x], reverse=True)[:k]
                top_k_accuracy = len(set(top_k_pred) & set(top_k_true)) / k
                
                results[method_name] = {
                    'correlation': correlation,
                    'p_value': p_value,
                    'top_k_accuracy': top_k_accuracy
                }
        
        # Print comparison
        print("\n  Method Comparison:")
        print("  " + "-"*50)
        for method, metrics in results.items():
            print(f"  {method:25s} | Corr: {metrics['correlation']:+.3f} | Top-5: {metrics['top_k_accuracy']:.2f}")
        
        return results
    
    def validate_cascade_predictions(self):
        """Validate cascade failure predictions"""
        
        graph = self.graph_exporter.export_graph()
        
        # Get critical components
        components = self.graph_analyzer.analyze_critical_components(graph)
        classified = self.graph_analyzer.classify_critical_components(components)
        
        critical = [c for c, level in classified.items() 
                   if level in ['VERY_HIGH', 'HIGH']][:5]
        
        cascade_results = []
        
        for component in critical:
            if component not in graph:
                continue
                
            # Predict cascade
            predicted_cascade = self._predict_cascade(graph, component)
            
            # Simulate actual cascade
            actual_cascade = self._simulate_cascade(graph, component)
            
            # Calculate accuracy
            if predicted_cascade and actual_cascade:
                overlap = len(set(predicted_cascade) & set(actual_cascade))
                precision = overlap / len(predicted_cascade) if predicted_cascade else 0
                recall = overlap / len(actual_cascade) if actual_cascade else 0
                
                cascade_results.append({
                    'component': component,
                    'predicted_size': len(predicted_cascade),
                    'actual_size': len(actual_cascade),
                    'precision': precision,
                    'recall': recall
                })
        
        if cascade_results:
            avg_precision = np.mean([r['precision'] for r in cascade_results])
            avg_recall = np.mean([r['recall'] for r in cascade_results])
            
            result = {
                'cascade_predictions': len(cascade_results),
                'avg_precision': avg_precision,
                'avg_recall': avg_recall,
                'cascade_validation_passed': avg_precision > 0.6
            }
            
            print(f"  ✓ Cascade prediction precision: {avg_precision:.2f}")
            print(f"  ✓ Cascade prediction recall: {avg_recall:.2f}")
            
            return result
        
        return None
    
    def _predict_cascade(self, graph, initial_failure):
        """Predict cascade based on dependencies"""
        
        # Get all dependent nodes
        dependents = nx.descendants(graph, initial_failure)
        
        # Filter for likely cascade (direct dependencies)
        likely_cascade = []
        for dep in dependents:
            # Check if this node heavily depends on the failed component
            paths = list(nx.all_simple_paths(graph, initial_failure, dep, cutoff=3))
            if len(paths) > 0:
                likely_cascade.append(dep)
        
        return likely_cascade
    
    def _simulate_cascade(self, graph, initial_failure, threshold=0.5):
        """Simulate actual cascade failure"""
        
        failed = {initial_failure}
        cascade = [initial_failure]
        
        while True:
            new_failures = set()
            
            for node in graph.nodes():
                if node in failed:
                    continue
                
                # Check if node would fail due to dependencies
                dependencies = list(graph.predecessors(node))
                if dependencies:
                    failed_deps = len([d for d in dependencies if d in failed])
                    failure_ratio = failed_deps / len(dependencies)
                    
                    if failure_ratio >= threshold:
                        new_failures.add(node)
            
            if not new_failures:
                break
                
            failed.update(new_failures)
            cascade.extend(new_failures)
        
        return cascade[1:]  # Exclude initial failure
    
    def print_validation_summary(self, results):
        """Print validation summary"""
        
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)
        
        # Overall validation score
        validations_passed = 0
        total_validations = 0
        
        if results.get('impact_validation'):
            total_validations += 1
            if results['impact_validation'].get('validation_passed'):
                validations_passed += 1
                print("✅ Impact Correlation: PASSED")
            else:
                print("❌ Impact Correlation: FAILED")
        
        if results.get('consistency'):
            total_validations += 1
            if results['consistency'].get('stable'):
                validations_passed += 1
                print("✅ Consistency: PASSED")
            else:
                print("❌ Consistency: FAILED")
        
        if results.get('qos_validation'):
            total_validations += 1
            if results['qos_validation'].get('qos_helps'):
                validations_passed += 1
                print("✅ QoS Contribution: VALIDATED")
            else:
                print("❌ QoS Contribution: NOT VALIDATED")
        
        if results.get('comparison'):
            total_validations += 1
            # Check if your method is best
            qos_corr = results['comparison'].get('QoS-Aware (Your Method)', {}).get('correlation', 0)
            best_corr = max(m.get('correlation', 0) for m in results['comparison'].values())
            if qos_corr == best_corr:
                validations_passed += 1
                print("✅ Method Comparison: BEST PERFORMER")
            else:
                print("❌ Method Comparison: NOT BEST")
        
        if results.get('cascade'):
            total_validations += 1
            if results['cascade'].get('cascade_validation_passed'):
                validations_passed += 1
                print("✅ Cascade Prediction: PASSED")
            else:
                print("❌ Cascade Prediction: FAILED")
        
        print("\n" + "-"*70)
        print(f"Overall Validation Score: {validations_passed}/{total_validations} "
              f"({validations_passed/total_validations*100:.0f}%)")
        
        if validations_passed >= total_validations * 0.8:
            print("\n🎉 VALIDATION SUCCESSFUL: Your criticality identification method is validated!")
        elif validations_passed >= total_validations * 0.6:
            print("\n⚠️ PARTIAL VALIDATION: Some aspects need improvement")
        else:
            print("\n❌ VALIDATION FAILED: Significant improvements needed")
        
        print("="*70)