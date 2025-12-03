#!/usr/bin/env python3
"""
Unified Fuzzy Criticality Scorer for Pub-Sub Systems

This module provides a complete replacement for the traditional composite 
criticality scoring with fuzzy logic-based scoring for both nodes and edges.

Key Benefits of Fuzzy Logic over Composite Scoring:
====================================================
1. SMOOTH TRANSITIONS: No "sharp boundary problem" where score 0.799 is HIGH 
   but 0.801 is CRITICAL. Instead, both might be 40% HIGH and 60% CRITICAL.

2. HANDLES UNCERTAINTY: Real systems have measurement noise and uncertainty.
   Fuzzy logic naturally accommodates this with membership degrees.

3. RICHER INSIGHTS: Instead of a single score, we get membership degrees in
   each criticality level, providing more nuanced decision support.

4. DOMAIN-ALIGNED RULES: Fuzzy rules map directly to expert knowledge like
   "IF betweenness is HIGH AND is_articulation_point THEN criticality is CRITICAL"

5. COMBINED FACTORS: Multiple indicators are naturally combined through
   fuzzy inference rather than arbitrary weighted sums.

Mathematical Foundation:
========================
Traditional Composite Score:
    C_score(v) = α·C_B^norm(v) + β·AP(v) + γ·I(v)

Fuzzy Logic Replacement:
    μ_level(v) = max_{r∈Rules} [min(μ_antecedent(r), w_r)]
    
    where:
    - μ_level(v) is the membership degree in criticality level
    - μ_antecedent(r) is the firing strength of rule r
    - w_r is the rule weight

The fuzzy score is obtained via defuzzification:
    C_fuzzy(v) = ∫ μ(x)·x dx / ∫ μ(x) dx  (centroid method)

Research Alignment:
==================
This implementation maintains compatibility with research validation targets:
- Spearman correlation ≥ 0.7 with failure simulations
- F1-score ≥ 0.9 for critical component identification  
- Precision ≥ 0.9, Recall ≥ 0.85

Author: Software-as-a-Graph Research Project
Version: 2.0 - Unified Fuzzy Logic Implementation
"""

import math
from typing import Dict, List, Tuple, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging

# Optional dependencies
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ============================================================================
# Enumerations
# ============================================================================

class FuzzyCriticalityLevel(Enum):
    """Fuzzy criticality levels with associated value ranges"""
    CRITICAL = "critical"    # [0.8, 1.0] - Immediate attention required
    HIGH = "high"            # [0.6, 0.8] - High priority monitoring
    MEDIUM = "medium"        # [0.4, 0.6] - Regular monitoring  
    LOW = "low"              # [0.2, 0.4] - Routine checks
    MINIMAL = "minimal"      # [0.0, 0.2] - Low concern


class MembershipType(Enum):
    """Types of membership functions"""
    TRIANGULAR = "triangular"
    TRAPEZOIDAL = "trapezoidal"
    GAUSSIAN = "gaussian"
    SIGMOID_LEFT = "sigmoid_left"
    SIGMOID_RIGHT = "sigmoid_right"


class DefuzzificationMethod(Enum):
    """Defuzzification methods for converting fuzzy output to crisp value"""
    CENTROID = "centroid"      # Center of gravity - most common
    BISECTOR = "bisector"      # Bisector of area
    MOM = "mom"                # Mean of maximum
    SOM = "som"                # Smallest of maximum
    LOM = "lom"                # Largest of maximum
    WEIGHTED_AVERAGE = "weighted_average"  # Weighted average of centers


# ============================================================================
# Core Fuzzy Logic Components (Pure Python - No Dependencies)
# ============================================================================

@dataclass
class FuzzySet:
    """
    Represents a fuzzy set with a membership function.
    
    A fuzzy set allows partial membership, where an element can belong
    to a set with a degree between 0 and 1, rather than binary 0 or 1.
    
    Supported membership functions:
    - Triangular: μ(x) = max(0, min((x-a)/(b-a), (c-x)/(c-b)))
    - Trapezoidal: μ(x) with flat top between b and c
    - Gaussian: μ(x) = exp(-((x-c)²)/(2σ²))
    - Sigmoid: μ(x) = 1/(1 + exp(-a(x-c)))
    """
    name: str
    membership_type: MembershipType
    params: Tuple[float, ...]  # Parameters depend on membership type
    
    def membership(self, x: float) -> float:
        """
        Calculate membership degree for value x.
        
        Args:
            x: Input value
            
        Returns:
            Membership degree in [0, 1]
        """
        if self.membership_type == MembershipType.TRIANGULAR:
            a, b, c = self.params
            if x <= a or x >= c:
                return 0.0
            elif x <= b:
                return (x - a) / (b - a) if b != a else 1.0
            else:
                return (c - x) / (c - b) if c != b else 1.0
                
        elif self.membership_type == MembershipType.TRAPEZOIDAL:
            a, b, c, d = self.params
            if x <= a or x >= d:
                return 0.0
            elif x <= b:
                return (x - a) / (b - a) if b != a else 1.0
            elif x <= c:
                return 1.0
            else:
                return (d - x) / (d - c) if d != c else 1.0
                
        elif self.membership_type == MembershipType.GAUSSIAN:
            center, sigma = self.params
            return math.exp(-((x - center) ** 2) / (2 * sigma ** 2))
            
        elif self.membership_type == MembershipType.SIGMOID_LEFT:
            # High on left, low on right
            center, steepness = self.params
            return 1.0 / (1.0 + math.exp(steepness * (x - center)))
            
        elif self.membership_type == MembershipType.SIGMOID_RIGHT:
            # Low on left, high on right
            center, steepness = self.params
            return 1.0 / (1.0 + math.exp(-steepness * (x - center)))
            
        return 0.0
    
    def get_center(self) -> float:
        """Get the center point of this fuzzy set"""
        if self.membership_type == MembershipType.TRIANGULAR:
            return self.params[1]  # b is center
        elif self.membership_type == MembershipType.TRAPEZOIDAL:
            return (self.params[1] + self.params[2]) / 2  # midpoint of flat top
        elif self.membership_type in (MembershipType.GAUSSIAN, 
                                       MembershipType.SIGMOID_LEFT,
                                       MembershipType.SIGMOID_RIGHT):
            return self.params[0]
        return 0.5


@dataclass
class FuzzyVariable:
    """
    Represents a fuzzy linguistic variable.
    
    A linguistic variable like "betweenness" can have linguistic terms
    like "low", "medium", "high", each with its own fuzzy set.
    """
    name: str
    universe: Tuple[float, float]  # (min, max) range
    fuzzy_sets: Dict[str, FuzzySet] = field(default_factory=dict)
    
    def add_set(self, name: str, membership_type: MembershipType, 
                params: Tuple[float, ...]) -> None:
        """Add a fuzzy set (linguistic term) to this variable"""
        self.fuzzy_sets[name] = FuzzySet(name, membership_type, params)
    
    def fuzzify(self, value: float) -> Dict[str, float]:
        """
        Fuzzify a crisp value into membership degrees for all terms.
        
        Args:
            value: Crisp input value
            
        Returns:
            Dictionary mapping term names to membership degrees
        """
        return {name: fs.membership(value) for name, fs in self.fuzzy_sets.items()}
    
    def get_dominant_term(self, value: float) -> Tuple[str, float]:
        """Get the term with highest membership for given value"""
        memberships = self.fuzzify(value)
        best_term = max(memberships, key=memberships.get)
        return best_term, memberships[best_term]


@dataclass 
class FuzzyRule:
    """
    Represents an IF-THEN fuzzy rule.
    
    Example: IF betweenness IS high AND is_articulation IS yes 
             THEN criticality IS critical
    
    The antecedent (IF part) is combined using AND (min) or OR (max).
    The consequent (THEN part) specifies the output fuzzy set.
    """
    antecedent: List[Tuple[str, str]]  # [(variable_name, term_name), ...]
    consequent: Tuple[str, str]        # (variable_name, term_name)
    weight: float = 1.0                # Rule weight/importance
    use_and: bool = True               # True for AND (min), False for OR (max)
    
    def evaluate(self, input_memberships: Dict[str, Dict[str, float]]) -> float:
        """
        Evaluate rule firing strength given input memberships.
        
        Args:
            input_memberships: {variable_name: {term_name: membership}}
            
        Returns:
            Rule firing strength in [0, 1]
        """
        strengths = []
        for var_name, term_name in self.antecedent:
            if var_name in input_memberships:
                membership = input_memberships[var_name].get(term_name, 0.0)
                strengths.append(membership)
            else:
                strengths.append(0.0)
        
        if not strengths:
            return 0.0
        
        # AND = min, OR = max
        if self.use_and:
            return min(strengths) * self.weight
        else:
            return max(strengths) * self.weight


class FuzzyInferenceSystem:
    """
    Mamdani-style Fuzzy Inference System.
    
    Implements the complete fuzzy inference process:
    1. Fuzzification - Convert crisp inputs to membership degrees
    2. Rule Evaluation - Apply fuzzy rules to get firing strengths
    3. Aggregation - Combine rule outputs
    4. Defuzzification - Convert fuzzy output to crisp value
    """
    
    def __init__(self, name: str):
        self.name = name
        self.input_variables: Dict[str, FuzzyVariable] = {}
        self.output_variables: Dict[str, FuzzyVariable] = {}
        self.rules: List[FuzzyRule] = []
        self.logger = logging.getLogger(f"FIS.{name}")
    
    def add_input(self, variable: FuzzyVariable) -> None:
        """Add an input variable"""
        self.input_variables[variable.name] = variable
    
    def add_output(self, variable: FuzzyVariable) -> None:
        """Add an output variable"""
        self.output_variables[variable.name] = variable
    
    def add_rule(self, rule: FuzzyRule) -> None:
        """Add a fuzzy rule"""
        self.rules.append(rule)
    
    def fuzzify_inputs(self, inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Fuzzify all crisp inputs.
        
        Args:
            inputs: {variable_name: crisp_value}
            
        Returns:
            {variable_name: {term_name: membership_degree}}
        """
        memberships = {}
        for var_name, value in inputs.items():
            if var_name in self.input_variables:
                memberships[var_name] = self.input_variables[var_name].fuzzify(value)
        return memberships
    
    def infer(self, inputs: Dict[str, float], 
              defuzz_method: DefuzzificationMethod = DefuzzificationMethod.CENTROID
              ) -> Dict[str, float]:
        """
        Perform fuzzy inference.
        
        Args:
            inputs: Crisp input values
            defuzz_method: Defuzzification method
            
        Returns:
            Crisp output values
        """
        # Step 1: Fuzzify inputs
        input_memberships = self.fuzzify_inputs(inputs)
        
        # Step 2 & 3: Evaluate rules and aggregate by output variable
        output_aggregations: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        for rule in self.rules:
            firing_strength = rule.evaluate(input_memberships)
            if firing_strength > 0:
                out_var, out_term = rule.consequent
                # Use max aggregation for same output term
                current = output_aggregations[out_var][out_term]
                output_aggregations[out_var][out_term] = max(current, firing_strength)
        
        # Step 4: Defuzzify each output
        outputs = {}
        for var_name, term_strengths in output_aggregations.items():
            if var_name in self.output_variables:
                outputs[var_name] = self._defuzzify(
                    self.output_variables[var_name],
                    term_strengths,
                    defuzz_method
                )
        
        return outputs
    
    def _defuzzify(self, variable: FuzzyVariable, 
                   term_strengths: Dict[str, float],
                   method: DefuzzificationMethod) -> float:
        """
        Defuzzify aggregated fuzzy output to crisp value.
        
        Args:
            variable: Output fuzzy variable
            term_strengths: {term_name: firing_strength}
            method: Defuzzification method
            
        Returns:
            Crisp output value
        """
        if not term_strengths:
            return (variable.universe[0] + variable.universe[1]) / 2
        
        if method == DefuzzificationMethod.WEIGHTED_AVERAGE:
            # Fast approximation using weighted average of centers
            numerator = 0.0
            denominator = 0.0
            for term_name, strength in term_strengths.items():
                if term_name in variable.fuzzy_sets and strength > 0:
                    center = variable.fuzzy_sets[term_name].get_center()
                    numerator += center * strength
                    denominator += strength
            return numerator / denominator if denominator > 0 else 0.5
        
        elif method == DefuzzificationMethod.CENTROID:
            # Centroid of aggregated membership function
            min_val, max_val = variable.universe
            num_points = 101
            step = (max_val - min_val) / (num_points - 1)
            
            numerator = 0.0
            denominator = 0.0
            
            for i in range(num_points):
                x = min_val + i * step
                # Get aggregated membership at x
                mu = 0.0
                for term_name, strength in term_strengths.items():
                    if term_name in variable.fuzzy_sets:
                        term_mu = variable.fuzzy_sets[term_name].membership(x)
                        # Clip membership by firing strength
                        clipped = min(term_mu, strength)
                        mu = max(mu, clipped)  # Max aggregation
                
                numerator += x * mu
                denominator += mu
            
            return numerator / denominator if denominator > 0 else 0.5
        
        elif method == DefuzzificationMethod.MOM:
            # Mean of Maximum - find region with max membership
            min_val, max_val = variable.universe
            num_points = 101
            step = (max_val - min_val) / (num_points - 1)
            
            max_mu = 0.0
            max_points = []
            
            for i in range(num_points):
                x = min_val + i * step
                mu = 0.0
                for term_name, strength in term_strengths.items():
                    if term_name in variable.fuzzy_sets:
                        term_mu = variable.fuzzy_sets[term_name].membership(x)
                        clipped = min(term_mu, strength)
                        mu = max(mu, clipped)
                
                if mu > max_mu:
                    max_mu = mu
                    max_points = [x]
                elif abs(mu - max_mu) < 1e-6:
                    max_points.append(x)
            
            return sum(max_points) / len(max_points) if max_points else 0.5
        
        # Default to weighted average
        return self._defuzzify(variable, term_strengths, 
                              DefuzzificationMethod.WEIGHTED_AVERAGE)


# ============================================================================
# Node Criticality Fuzzy System
# ============================================================================

def create_node_criticality_fis() -> FuzzyInferenceSystem:
    """
    Create Fuzzy Inference System for node criticality assessment.
    
    Input Variables (same as composite score formula):
    - betweenness: Normalized betweenness centrality C_B^norm(v) ∈ [0, 1]
    - articulation_point: AP indicator (extended to continuous) ∈ [0, 1]
    - impact: Impact score I(v) ∈ [0, 1]
    
    Output Variable:
    - criticality: Fuzzy criticality level ∈ [0, 1]
    
    Returns:
        Configured FIS for node criticality
    """
    fis = FuzzyInferenceSystem("NodeCriticality")
    
    # === Input Variable: Betweenness Centrality ===
    # C_B^norm(v) - how often node appears on shortest paths
    # Overlapping regions ensure smooth transitions
    betweenness = FuzzyVariable("betweenness", (0.0, 1.0))
    betweenness.add_set("very_low", MembershipType.TRAPEZOIDAL, (0.0, 0.0, 0.1, 0.25))
    betweenness.add_set("low", MembershipType.TRIANGULAR, (0.1, 0.25, 0.4))
    betweenness.add_set("medium", MembershipType.TRIANGULAR, (0.25, 0.45, 0.65))
    betweenness.add_set("high", MembershipType.TRIANGULAR, (0.5, 0.7, 0.85))
    betweenness.add_set("very_high", MembershipType.TRAPEZOIDAL, (0.7, 0.85, 1.0, 1.0))
    fis.add_input(betweenness)
    
    # === Input Variable: Articulation Point ===
    # Extended from binary to continuous for fuzzy handling
    # 0.0 = definitely not AP, 1.0 = definitely AP
    # Allows for "near-articulation points" with intermediate values
    articulation = FuzzyVariable("articulation_point", (0.0, 1.0))
    articulation.add_set("no", MembershipType.TRAPEZOIDAL, (0.0, 0.0, 0.2, 0.4))
    articulation.add_set("uncertain", MembershipType.TRIANGULAR, (0.3, 0.5, 0.7))
    articulation.add_set("yes", MembershipType.TRAPEZOIDAL, (0.6, 0.8, 1.0, 1.0))
    fis.add_input(articulation)
    
    # === Input Variable: Impact Score ===
    # I(v) = 1 - |R(G-v)| / |R(G)| - reachability loss
    impact = FuzzyVariable("impact", (0.0, 1.0))
    impact.add_set("negligible", MembershipType.TRAPEZOIDAL, (0.0, 0.0, 0.1, 0.2))
    impact.add_set("minor", MembershipType.TRIANGULAR, (0.1, 0.25, 0.4))
    impact.add_set("moderate", MembershipType.TRIANGULAR, (0.3, 0.45, 0.6))
    impact.add_set("major", MembershipType.TRIANGULAR, (0.5, 0.65, 0.8))
    impact.add_set("catastrophic", MembershipType.TRAPEZOIDAL, (0.7, 0.85, 1.0, 1.0))
    fis.add_input(impact)
    
    # === Output Variable: Criticality ===
    criticality = FuzzyVariable("criticality", (0.0, 1.0))
    criticality.add_set("minimal", MembershipType.TRAPEZOIDAL, (0.0, 0.0, 0.1, 0.25))
    criticality.add_set("low", MembershipType.TRIANGULAR, (0.15, 0.3, 0.45))
    criticality.add_set("medium", MembershipType.TRIANGULAR, (0.35, 0.5, 0.65))
    criticality.add_set("high", MembershipType.TRIANGULAR, (0.55, 0.7, 0.85))
    criticality.add_set("critical", MembershipType.TRAPEZOIDAL, (0.75, 0.85, 1.0, 1.0))
    fis.add_output(criticality)
    
    # === Fuzzy Rules ===
    # These rules encode expert knowledge about criticality
    
    # CRITICAL rules - any strong indicator combination
    # Articulation points are inherently critical
    fis.add_rule(FuzzyRule(
        [("articulation_point", "yes")],
        ("criticality", "critical"), weight=0.9
    ))
    fis.add_rule(FuzzyRule(
        [("betweenness", "very_high"), ("articulation_point", "yes")],
        ("criticality", "critical"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("betweenness", "very_high"), ("impact", "catastrophic")],
        ("criticality", "critical"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("articulation_point", "yes"), ("impact", "catastrophic")],
        ("criticality", "critical"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("betweenness", "high"), ("articulation_point", "yes"), ("impact", "major")],
        ("criticality", "critical"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("articulation_point", "yes"), ("impact", "major")],
        ("criticality", "critical"), weight=0.95
    ))
    
    # HIGH rules
    fis.add_rule(FuzzyRule(
        [("betweenness", "very_high"), ("impact", "major")],
        ("criticality", "high"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("betweenness", "high"), ("articulation_point", "yes")],
        ("criticality", "high"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("betweenness", "high"), ("impact", "major")],
        ("criticality", "high"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("articulation_point", "yes"), ("impact", "moderate")],
        ("criticality", "high"), weight=0.9
    ))
    fis.add_rule(FuzzyRule(
        [("betweenness", "medium"), ("articulation_point", "yes"), ("impact", "moderate")],
        ("criticality", "high"), weight=0.9
    ))
    fis.add_rule(FuzzyRule(
        [("betweenness", "very_high"), ("articulation_point", "uncertain")],
        ("criticality", "high"), weight=0.85
    ))
    # Additional rules for smooth transition coverage
    fis.add_rule(FuzzyRule(
        [("betweenness", "high")],
        ("criticality", "high"), weight=0.6
    ))
    fis.add_rule(FuzzyRule(
        [("betweenness", "very_high")],
        ("criticality", "high"), weight=0.75
    ))
    
    # MEDIUM rules
    fis.add_rule(FuzzyRule(
        [("betweenness", "high"), ("impact", "moderate")],
        ("criticality", "medium"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("betweenness", "medium"), ("impact", "major")],
        ("criticality", "medium"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("betweenness", "medium"), ("impact", "moderate")],
        ("criticality", "medium"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("betweenness", "medium"), ("articulation_point", "uncertain")],
        ("criticality", "medium"), weight=0.8
    ))
    fis.add_rule(FuzzyRule(
        [("betweenness", "low"), ("articulation_point", "yes")],
        ("criticality", "medium"), weight=0.9
    ))
    fis.add_rule(FuzzyRule(
        [("betweenness", "high"), ("articulation_point", "no"), ("impact", "minor")],
        ("criticality", "medium"), weight=0.85
    ))
    # Additional rules for coverage
    fis.add_rule(FuzzyRule(
        [("betweenness", "medium")],
        ("criticality", "medium"), weight=0.5
    ))
    
    # LOW rules
    fis.add_rule(FuzzyRule(
        [("betweenness", "low"), ("impact", "moderate")],
        ("criticality", "low"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("betweenness", "medium"), ("impact", "minor")],
        ("criticality", "low"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("betweenness", "low"), ("impact", "minor")],
        ("criticality", "low"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("betweenness", "low"), ("articulation_point", "no")],
        ("criticality", "low"), weight=0.8
    ))
    fis.add_rule(FuzzyRule(
        [("betweenness", "medium"), ("articulation_point", "no"), ("impact", "negligible")],
        ("criticality", "low"), weight=0.9
    ))
    # Additional rules for coverage
    fis.add_rule(FuzzyRule(
        [("betweenness", "low")],
        ("criticality", "low"), weight=0.6
    ))
    
    # MINIMAL rules
    fis.add_rule(FuzzyRule(
        [("betweenness", "very_low"), ("impact", "negligible")],
        ("criticality", "minimal"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("betweenness", "very_low"), ("articulation_point", "no"), ("impact", "minor")],
        ("criticality", "minimal"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("betweenness", "low"), ("articulation_point", "no"), ("impact", "negligible")],
        ("criticality", "minimal"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("betweenness", "very_low"), ("articulation_point", "no")],
        ("criticality", "minimal"), weight=0.9
    ))
    # Default low-end coverage for smooth transitions
    fis.add_rule(FuzzyRule(
        [("betweenness", "very_low")],
        ("criticality", "minimal"), weight=0.7
    ))
    fis.add_rule(FuzzyRule(
        [("betweenness", "low"), ("articulation_point", "no"), ("impact", "minor")],
        ("criticality", "low"), weight=0.8
    ))
    
    return fis


# ============================================================================
# Edge Criticality Fuzzy System
# ============================================================================

def create_edge_criticality_fis() -> FuzzyInferenceSystem:
    """
    Create Fuzzy Inference System for edge criticality assessment.
    
    Input Variables:
    - edge_betweenness: Normalized edge betweenness centrality ∈ [0, 1]
    - is_bridge: Bridge indicator (extended to continuous) ∈ [0, 1]
    - flow_importance: Message flow importance based on QoS/volume ∈ [0, 1]
    
    Output Variable:
    - edge_criticality: Fuzzy edge criticality level ∈ [0, 1]
    
    Returns:
        Configured FIS for edge criticality
    """
    fis = FuzzyInferenceSystem("EdgeCriticality")
    
    # === Input Variable: Edge Betweenness ===
    edge_betweenness = FuzzyVariable("edge_betweenness", (0.0, 1.0))
    edge_betweenness.add_set("very_low", MembershipType.TRAPEZOIDAL, (0.0, 0.0, 0.1, 0.25))
    edge_betweenness.add_set("low", MembershipType.TRIANGULAR, (0.1, 0.25, 0.4))
    edge_betweenness.add_set("medium", MembershipType.TRIANGULAR, (0.25, 0.45, 0.65))
    edge_betweenness.add_set("high", MembershipType.TRIANGULAR, (0.5, 0.7, 0.85))
    edge_betweenness.add_set("very_high", MembershipType.TRAPEZOIDAL, (0.7, 0.85, 1.0, 1.0))
    fis.add_input(edge_betweenness)
    
    # === Input Variable: Bridge Status ===
    # Extended from binary to continuous - stronger separation
    bridge = FuzzyVariable("is_bridge", (0.0, 1.0))
    bridge.add_set("no", MembershipType.TRAPEZOIDAL, (0.0, 0.0, 0.15, 0.35))
    bridge.add_set("uncertain", MembershipType.TRIANGULAR, (0.25, 0.5, 0.75))
    bridge.add_set("yes", MembershipType.TRAPEZOIDAL, (0.65, 0.85, 1.0, 1.0))
    fis.add_input(bridge)
    
    # === Input Variable: Flow Importance ===
    # Based on message volume, QoS requirements, subscriber count
    flow = FuzzyVariable("flow_importance", (0.0, 1.0))
    flow.add_set("negligible", MembershipType.TRAPEZOIDAL, (0.0, 0.0, 0.1, 0.2))
    flow.add_set("minor", MembershipType.TRIANGULAR, (0.1, 0.25, 0.4))
    flow.add_set("moderate", MembershipType.TRIANGULAR, (0.3, 0.5, 0.7))
    flow.add_set("significant", MembershipType.TRIANGULAR, (0.6, 0.75, 0.9))
    flow.add_set("critical", MembershipType.TRAPEZOIDAL, (0.8, 0.9, 1.0, 1.0))
    fis.add_input(flow)
    
    # === Output Variable: Edge Criticality ===
    criticality = FuzzyVariable("edge_criticality", (0.0, 1.0))
    criticality.add_set("minimal", MembershipType.TRAPEZOIDAL, (0.0, 0.0, 0.1, 0.25))
    criticality.add_set("low", MembershipType.TRIANGULAR, (0.15, 0.3, 0.45))
    criticality.add_set("medium", MembershipType.TRIANGULAR, (0.35, 0.5, 0.65))
    criticality.add_set("high", MembershipType.TRIANGULAR, (0.55, 0.7, 0.85))
    criticality.add_set("critical", MembershipType.TRAPEZOIDAL, (0.75, 0.85, 1.0, 1.0))
    fis.add_output(criticality)
    
    # === Fuzzy Rules for Edges ===
    
    # CRITICAL rules - bridges are inherently critical regardless of other factors
    fis.add_rule(FuzzyRule(
        [("is_bridge", "yes")],
        ("edge_criticality", "critical"), weight=0.95
    ))
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "very_high"), ("is_bridge", "yes")],
        ("edge_criticality", "critical"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("is_bridge", "yes"), ("flow_importance", "critical")],
        ("edge_criticality", "critical"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "very_high"), ("flow_importance", "critical")],
        ("edge_criticality", "critical"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "high"), ("is_bridge", "yes"), ("flow_importance", "significant")],
        ("edge_criticality", "critical"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("is_bridge", "yes"), ("flow_importance", "significant")],
        ("edge_criticality", "critical"), weight=0.95
    ))
    
    # HIGH rules
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "high"), ("is_bridge", "yes")],
        ("edge_criticality", "high"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "very_high"), ("flow_importance", "significant")],
        ("edge_criticality", "high"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("is_bridge", "yes"), ("flow_importance", "moderate")],
        ("edge_criticality", "high"), weight=0.9
    ))
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "high"), ("flow_importance", "significant")],
        ("edge_criticality", "high"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "very_high"), ("is_bridge", "uncertain")],
        ("edge_criticality", "high"), weight=0.85
    ))
    # Bridge-only rule for strong impact
    fis.add_rule(FuzzyRule(
        [("is_bridge", "yes")],
        ("edge_criticality", "high"), weight=0.8
    ))
    # Additional coverage rules
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "high")],
        ("edge_criticality", "high"), weight=0.55
    ))
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "very_high")],
        ("edge_criticality", "high"), weight=0.7
    ))
    
    # MEDIUM rules
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "medium"), ("is_bridge", "yes")],
        ("edge_criticality", "medium"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "high"), ("flow_importance", "moderate")],
        ("edge_criticality", "medium"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "medium"), ("flow_importance", "significant")],
        ("edge_criticality", "medium"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "medium"), ("flow_importance", "moderate")],
        ("edge_criticality", "medium"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "high"), ("is_bridge", "no"), ("flow_importance", "minor")],
        ("edge_criticality", "medium"), weight=0.85
    ))
    # Coverage rule
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "medium")],
        ("edge_criticality", "medium"), weight=0.5
    ))
    
    # LOW rules
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "low"), ("flow_importance", "moderate")],
        ("edge_criticality", "low"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "medium"), ("is_bridge", "no"), ("flow_importance", "minor")],
        ("edge_criticality", "low"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "low"), ("is_bridge", "no")],
        ("edge_criticality", "low"), weight=0.8
    ))
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "medium"), ("flow_importance", "minor")],
        ("edge_criticality", "low"), weight=0.9
    ))
    # Coverage rule
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "low")],
        ("edge_criticality", "low"), weight=0.6
    ))
    
    # MINIMAL rules
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "very_low"), ("is_bridge", "no")],
        ("edge_criticality", "minimal"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "very_low"), ("flow_importance", "negligible")],
        ("edge_criticality", "minimal"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "low"), ("is_bridge", "no"), ("flow_importance", "negligible")],
        ("edge_criticality", "minimal"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "very_low"), ("flow_importance", "minor")],
        ("edge_criticality", "minimal"), weight=0.9
    ))
    # Coverage rule
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "very_low")],
        ("edge_criticality", "minimal"), weight=0.7
    ))
    
    return fis


# ============================================================================
# Result Data Classes
# ============================================================================

@dataclass
class FuzzyNodeCriticalityScore:
    """
    Complete fuzzy criticality assessment for a node.
    
    Replaces CompositeCriticalityScore with fuzzy logic-based scoring.
    Maintains backward compatibility with the original interface while
    providing richer membership degree information.
    """
    # Node identification
    component: str
    component_type: str
    
    # Raw input values (same as original)
    betweenness_centrality_norm: float
    is_articulation_point: bool
    impact_score: float
    
    # Fuzzy output (replaces composite_score)
    fuzzy_score: float
    criticality_level: FuzzyCriticalityLevel
    
    # Rich fuzzy information (new in fuzzy version)
    membership_degrees: Dict[str, float] = field(default_factory=dict)
    input_memberships: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Additional metrics (for compatibility)
    components_affected: int = 0
    reachability_loss_percentage: float = 0.0
    degree_centrality: float = 0.0
    closeness_centrality: float = 0.0
    pagerank: float = 0.0
    qos_score: float = 0.0
    
    # Backward compatibility alias
    @property
    def composite_score(self) -> float:
        """Alias for fuzzy_score to maintain backward compatibility"""
        return self.fuzzy_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'component': self.component,
            'component_type': self.component_type,
            'fuzzy_score': round(self.fuzzy_score, 4),
            'composite_score': round(self.fuzzy_score, 4),  # Alias
            'criticality_level': self.criticality_level.value,
            'betweenness_centrality_norm': round(self.betweenness_centrality_norm, 4),
            'is_articulation_point': self.is_articulation_point,
            'impact_score': round(self.impact_score, 4),
            'components_affected': self.components_affected,
            'reachability_loss_percentage': round(self.reachability_loss_percentage, 2),
            'membership_degrees': {k: round(v, 4) for k, v in self.membership_degrees.items()},
            'dominant_memberships': self._get_dominant_memberships()
        }
    
    def _get_dominant_memberships(self, threshold: float = 0.1) -> List[Tuple[str, float]]:
        """Get membership levels above threshold, sorted by degree"""
        significant = [(k, v) for k, v in self.membership_degrees.items() if v >= threshold]
        return sorted(significant, key=lambda x: x[1], reverse=True)


@dataclass
class FuzzyEdgeCriticalityScore:
    """
    Complete fuzzy criticality assessment for an edge.
    
    Replaces EdgeCriticalityScore with fuzzy logic-based scoring.
    """
    # Edge identification
    source: str
    target: str
    edge_type: str
    
    # Raw input values
    edge_betweenness: float
    is_bridge: bool
    flow_importance: float
    
    # Additional impact metrics
    creates_disconnection: bool = False
    components_after_removal: int = 1
    
    # Fuzzy output
    fuzzy_score: float = 0.0
    criticality_level: FuzzyCriticalityLevel = FuzzyCriticalityLevel.MINIMAL
    
    # Rich fuzzy information
    membership_degrees: Dict[str, float] = field(default_factory=dict)
    input_memberships: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Backward compatibility alias
    @property
    def composite_score(self) -> float:
        """Alias for fuzzy_score to maintain backward compatibility"""
        return self.fuzzy_score
    
    @property
    def betweenness_centrality(self) -> float:
        """Alias for edge_betweenness to maintain backward compatibility"""
        return self.edge_betweenness
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'source': self.source,
            'target': self.target,
            'edge_type': self.edge_type,
            'fuzzy_score': round(self.fuzzy_score, 4),
            'composite_score': round(self.fuzzy_score, 4),  # Alias
            'criticality_level': self.criticality_level.value,
            'edge_betweenness': round(self.edge_betweenness, 4),
            'is_bridge': self.is_bridge,
            'flow_importance': round(self.flow_importance, 4),
            'creates_disconnection': self.creates_disconnection,
            'components_after_removal': self.components_after_removal,
            'membership_degrees': {k: round(v, 4) for k, v in self.membership_degrees.items()}
        }


# ============================================================================
# Unified Fuzzy Criticality Scorer
# ============================================================================

class FuzzyCriticalityScorer:
    """
    Unified Fuzzy Criticality Scorer for both nodes and edges.
    
    This class replaces both CriticalityScorer and EdgeCriticalityAnalyzer
    with a unified fuzzy logic-based approach.
    
    Key Features:
    - Single interface for node and edge criticality
    - Smooth transitions between criticality levels
    - Rich membership degree information
    - Backward compatible with existing interfaces
    - Configurable defuzzification methods
    
    Usage:
        scorer = FuzzyCriticalityScorer()
        
        # Analyze entire graph
        node_scores, edge_scores = scorer.analyze_graph(G)
        
        # Or analyze individual components
        node_score = scorer.score_node(G, "node_id")
        edge_score = scorer.score_edge(G, "source", "target")
    """
    
    def __init__(self, 
                 defuzz_method: DefuzzificationMethod = DefuzzificationMethod.CENTROID,
                 calculate_impact: bool = True):
        """
        Initialize the unified fuzzy criticality scorer.
        
        Args:
            defuzz_method: Method for converting fuzzy output to crisp value
            calculate_impact: Whether to calculate impact scores (slower but more accurate)
        """
        self.defuzz_method = defuzz_method
        self.calculate_impact = calculate_impact
        
        # Initialize fuzzy inference systems
        self.node_fis = create_node_criticality_fis()
        self.edge_fis = create_edge_criticality_fis()
        
        # Cache for graph metrics
        self._betweenness_cache: Dict[str, float] = {}
        self._edge_betweenness_cache: Dict[Tuple[str, str], float] = {}
        self._articulation_points: Set[str] = set()
        self._bridges: Set[Tuple[str, str]] = set()
        
        self.logger = logging.getLogger(__name__)
    
    def _score_to_level(self, score: float) -> FuzzyCriticalityLevel:
        """Convert fuzzy score to criticality level"""
        if score >= 0.8:
            return FuzzyCriticalityLevel.CRITICAL
        elif score >= 0.6:
            return FuzzyCriticalityLevel.HIGH
        elif score >= 0.4:
            return FuzzyCriticalityLevel.MEDIUM
        elif score >= 0.2:
            return FuzzyCriticalityLevel.LOW
        else:
            return FuzzyCriticalityLevel.MINIMAL
    
    def _get_output_memberships(self, score: float, fis: FuzzyInferenceSystem, 
                                 output_var: str) -> Dict[str, float]:
        """Get membership degrees in output fuzzy sets for given score"""
        if output_var not in fis.output_variables:
            return {}
        
        var = fis.output_variables[output_var]
        return {name: fs.membership(score) for name, fs in var.fuzzy_sets.items()}
    
    def _calculate_node_impact(self, G, node: str) -> Tuple[float, int, float]:
        """
        Calculate impact score for node removal.
        
        I(v) = 1 - |R(G-v)| / |R(G)|
        
        Returns:
            (impact_score, components_affected, reachability_loss_pct)
        """
        if not NETWORKX_AVAILABLE:
            return 0.0, 0, 0.0
        
        try:
            # Calculate original reachability
            original_pairs = sum(1 for _ in nx.all_pairs_shortest_path_length(G))
            if original_pairs == 0:
                return 0.0, 0, 0.0
            
            # Create graph without node
            G_minus = G.copy()
            neighbors = list(G.neighbors(node))
            G_minus.remove_node(node)
            
            # Calculate new reachability
            new_pairs = sum(1 for _ in nx.all_pairs_shortest_path_length(G_minus))
            
            # Calculate impact
            impact = 1.0 - (new_pairs / original_pairs) if original_pairs > 0 else 0.0
            loss_pct = impact * 100
            
            return impact, len(neighbors), loss_pct
            
        except Exception as e:
            self.logger.debug(f"Error calculating impact for {node}: {e}")
            return 0.0, 0, 0.0
    
    def _calculate_edge_impact(self, G, source: str, target: str) -> Tuple[bool, int]:
        """
        Calculate impact of edge removal.
        
        Returns:
            (creates_disconnection, components_after_removal)
        """
        if not NETWORKX_AVAILABLE:
            return False, 1
        
        try:
            original_components = nx.number_weakly_connected_components(G)
            
            G_minus = G.copy()
            G_minus.remove_edge(source, target)
            
            new_components = nx.number_weakly_connected_components(G_minus)
            
            return new_components > original_components, new_components
            
        except Exception as e:
            self.logger.debug(f"Error calculating edge impact: {e}")
            return False, 1
    
    def _precompute_graph_metrics(self, G) -> None:
        """Precompute expensive graph metrics"""
        if not NETWORKX_AVAILABLE:
            return
        
        self.logger.info("Precomputing graph metrics...")
        
        # Betweenness centrality for nodes
        bc = nx.betweenness_centrality(G, normalized=True)
        max_bc = max(bc.values()) if bc else 1.0
        self._betweenness_cache = {n: v/max_bc if max_bc > 0 else 0 for n, v in bc.items()}
        
        # Edge betweenness centrality
        ebc = nx.edge_betweenness_centrality(G, normalized=True)
        max_ebc = max(ebc.values()) if ebc else 1.0
        self._edge_betweenness_cache = {e: v/max_ebc if max_ebc > 0 else 0 for e, v in ebc.items()}
        
        # Articulation points and bridges
        try:
            undirected = G.to_undirected()
            self._articulation_points = set(nx.articulation_points(undirected))
            self._bridges = set(nx.bridges(undirected))
        except Exception as e:
            self.logger.warning(f"Could not compute articulation points/bridges: {e}")
            self._articulation_points = set()
            self._bridges = set()
    
    def score_node(self, G, node: str, 
                   precomputed: bool = False) -> FuzzyNodeCriticalityScore:
        """
        Calculate fuzzy criticality score for a single node.
        
        Args:
            G: NetworkX graph
            node: Node identifier
            precomputed: Whether graph metrics are already precomputed
            
        Returns:
            FuzzyNodeCriticalityScore with full fuzzy analysis
        """
        if not NETWORKX_AVAILABLE:
            raise RuntimeError("NetworkX is required for graph analysis")
        
        if not precomputed:
            self._precompute_graph_metrics(G)
        
        node_data = G.nodes.get(node, {})
        node_type = node_data.get('type', 'Unknown')
        
        # Get input values
        bc_norm = self._betweenness_cache.get(node, 0.0)
        is_ap = node in self._articulation_points
        
        # Calculate impact if enabled
        if self.calculate_impact:
            impact, affected, loss_pct = self._calculate_node_impact(G, node)
        else:
            # Estimate from degree
            degree = G.degree(node)
            max_degree = max(d for _, d in G.degree()) if G.number_of_nodes() > 0 else 1
            impact = degree / max_degree if max_degree > 0 else 0.0
            affected = degree
            loss_pct = impact * 100
        
        # Prepare fuzzy inputs
        inputs = {
            'betweenness': min(1.0, max(0.0, bc_norm)),
            'articulation_point': 1.0 if is_ap else 0.0,
            'impact': min(1.0, max(0.0, impact))
        }
        
        # Get input memberships for transparency
        input_memberships = self.node_fis.fuzzify_inputs(inputs)
        
        # Run fuzzy inference
        outputs = self.node_fis.infer(inputs, self.defuzz_method)
        fuzzy_score = outputs.get('criticality', 0.5)
        
        # Get output memberships
        output_memberships = self._get_output_memberships(
            fuzzy_score, self.node_fis, 'criticality'
        )
        
        # Determine level
        level = self._score_to_level(fuzzy_score)
        
        return FuzzyNodeCriticalityScore(
            component=node,
            component_type=node_type,
            betweenness_centrality_norm=bc_norm,
            is_articulation_point=is_ap,
            impact_score=impact,
            fuzzy_score=fuzzy_score,
            criticality_level=level,
            membership_degrees=output_memberships,
            input_memberships=input_memberships,
            components_affected=affected,
            reachability_loss_percentage=loss_pct
        )
    
    def score_edge(self, G, source: str, target: str,
                   flow_importance: Optional[float] = None,
                   precomputed: bool = False) -> FuzzyEdgeCriticalityScore:
        """
        Calculate fuzzy criticality score for a single edge.
        
        Args:
            G: NetworkX graph
            source: Source node
            target: Target node
            flow_importance: Optional flow importance [0,1], defaults to estimated
            precomputed: Whether graph metrics are already precomputed
            
        Returns:
            FuzzyEdgeCriticalityScore with full fuzzy analysis
        """
        if not NETWORKX_AVAILABLE:
            raise RuntimeError("NetworkX is required for graph analysis")
        
        if not precomputed:
            self._precompute_graph_metrics(G)
        
        edge = (source, target)
        edge_data = G.edges.get(edge, {})
        edge_type = edge_data.get('type', 'Unknown')
        
        # Get input values
        ebc = self._edge_betweenness_cache.get(edge, 0.0)
        is_bridge = edge in self._bridges or (target, source) in self._bridges
        
        # Calculate flow importance if not provided
        if flow_importance is None:
            # Estimate from edge attributes or connected node degrees
            qos = edge_data.get('qos_score', 0.0)
            weight = edge_data.get('weight', 1.0)
            src_degree = G.degree(source)
            tgt_degree = G.degree(target)
            max_degree = max(d for _, d in G.degree()) if G.number_of_nodes() > 0 else 1
            
            # Combine factors
            degree_factor = (src_degree + tgt_degree) / (2 * max_degree) if max_degree > 0 else 0.5
            flow_importance = (qos + degree_factor + (weight/10)) / 3
            flow_importance = min(1.0, max(0.0, flow_importance))
        
        # Calculate disconnection impact
        creates_disc, components_after = self._calculate_edge_impact(G, source, target)
        
        # Prepare fuzzy inputs
        inputs = {
            'edge_betweenness': min(1.0, max(0.0, ebc)),
            'is_bridge': 1.0 if is_bridge or creates_disc else 0.0,
            'flow_importance': min(1.0, max(0.0, flow_importance))
        }
        
        # Get input memberships
        input_memberships = self.edge_fis.fuzzify_inputs(inputs)
        
        # Run fuzzy inference
        outputs = self.edge_fis.infer(inputs, self.defuzz_method)
        fuzzy_score = outputs.get('edge_criticality', 0.5)
        
        # Get output memberships
        output_memberships = self._get_output_memberships(
            fuzzy_score, self.edge_fis, 'edge_criticality'
        )
        
        # Determine level
        level = self._score_to_level(fuzzy_score)
        
        return FuzzyEdgeCriticalityScore(
            source=source,
            target=target,
            edge_type=edge_type,
            edge_betweenness=ebc,
            is_bridge=is_bridge,
            flow_importance=flow_importance,
            creates_disconnection=creates_disc,
            components_after_removal=components_after,
            fuzzy_score=fuzzy_score,
            criticality_level=level,
            membership_degrees=output_memberships,
            input_memberships=input_memberships
        )
    
    def analyze_graph(self, G) -> Tuple[Dict[str, FuzzyNodeCriticalityScore],
                                         Dict[Tuple[str, str], FuzzyEdgeCriticalityScore]]:
        """
        Perform complete fuzzy criticality analysis on entire graph.
        
        This is the main entry point for graph-wide analysis.
        
        Args:
            G: NetworkX directed graph
            
        Returns:
            Tuple of (node_scores, edge_scores) dictionaries
        """
        if not NETWORKX_AVAILABLE:
            raise RuntimeError("NetworkX is required for graph analysis")
        
        self.logger.info(f"Analyzing graph with {G.number_of_nodes()} nodes "
                        f"and {G.number_of_edges()} edges using fuzzy logic...")
        
        # Precompute all metrics once
        self._precompute_graph_metrics(G)
        
        # Analyze all nodes
        node_scores = {}
        for node in G.nodes():
            node_scores[node] = self.score_node(G, node, precomputed=True)
        
        # Analyze all edges
        edge_scores = {}
        for source, target in G.edges():
            edge_scores[(source, target)] = self.score_edge(
                G, source, target, precomputed=True
            )
        
        self.logger.info(f"Analysis complete. Found {self._count_critical(node_scores)} "
                        f"critical nodes and {self._count_critical_edges(edge_scores)} "
                        f"critical edges.")
        
        return node_scores, edge_scores
    
    def _count_critical(self, scores: Dict[str, FuzzyNodeCriticalityScore]) -> int:
        """Count nodes at CRITICAL level"""
        return sum(1 for s in scores.values() 
                   if s.criticality_level == FuzzyCriticalityLevel.CRITICAL)
    
    def _count_critical_edges(self, scores: Dict[Tuple[str, str], FuzzyEdgeCriticalityScore]) -> int:
        """Count edges at CRITICAL level"""
        return sum(1 for s in scores.values() 
                   if s.criticality_level == FuzzyCriticalityLevel.CRITICAL)
    
    # ========================================================================
    # Utility Methods (Backward Compatible with Original Interfaces)
    # ========================================================================
    
    def get_top_critical_nodes(self, 
                               scores: Dict[str, FuzzyNodeCriticalityScore],
                               n: int = 10,
                               min_score: float = 0.0) -> List[FuzzyNodeCriticalityScore]:
        """Get top N most critical nodes"""
        filtered = [s for s in scores.values() if s.fuzzy_score >= min_score]
        return sorted(filtered, key=lambda x: x.fuzzy_score, reverse=True)[:n]
    
    def get_top_critical_edges(self,
                               scores: Dict[Tuple[str, str], FuzzyEdgeCriticalityScore],
                               n: int = 10,
                               min_score: float = 0.0) -> List[FuzzyEdgeCriticalityScore]:
        """Get top N most critical edges"""
        filtered = [s for s in scores.values() if s.fuzzy_score >= min_score]
        return sorted(filtered, key=lambda x: x.fuzzy_score, reverse=True)[:n]
    
    def get_critical_components(self,
                                scores: Dict[str, FuzzyNodeCriticalityScore],
                                threshold: float = 0.7) -> List[FuzzyNodeCriticalityScore]:
        """Get all nodes above criticality threshold"""
        return [s for s in scores.values() if s.fuzzy_score >= threshold]
    
    def get_bridges(self,
                    scores: Dict[Tuple[str, str], FuzzyEdgeCriticalityScore]
                    ) -> List[FuzzyEdgeCriticalityScore]:
        """Get all bridge edges"""
        return [s for s in scores.values() if s.is_bridge]
    
    def summarize_node_criticality(self, 
                                   scores: Dict[str, FuzzyNodeCriticalityScore]) -> Dict[str, Any]:
        """Generate summary statistics for node criticality"""
        if not scores:
            return {'total_nodes': 0}
        
        level_counts = defaultdict(int)
        fuzzy_scores = []
        ap_count = 0
        
        for score in scores.values():
            level_counts[score.criticality_level.value] += 1
            fuzzy_scores.append(score.fuzzy_score)
            if score.is_articulation_point:
                ap_count += 1
        
        return {
            'total_nodes': len(scores),
            'critical_count': level_counts['critical'],
            'high_count': level_counts['high'],
            'medium_count': level_counts['medium'],
            'low_count': level_counts['low'],
            'minimal_count': level_counts['minimal'],
            'articulation_points': ap_count,
            'avg_fuzzy_score': round(sum(fuzzy_scores) / len(fuzzy_scores), 4),
            'max_fuzzy_score': round(max(fuzzy_scores), 4),
            'min_fuzzy_score': round(min(fuzzy_scores), 4)
        }
    
    def summarize_edge_criticality(self,
                                   scores: Dict[Tuple[str, str], FuzzyEdgeCriticalityScore]
                                   ) -> Dict[str, Any]:
        """Generate summary statistics for edge criticality"""
        if not scores:
            return {'total_edges': 0}
        
        level_counts = defaultdict(int)
        fuzzy_scores = []
        bridge_count = 0
        
        for score in scores.values():
            level_counts[score.criticality_level.value] += 1
            fuzzy_scores.append(score.fuzzy_score)
            if score.is_bridge:
                bridge_count += 1
        
        return {
            'total_edges': len(scores),
            'critical_count': level_counts['critical'],
            'high_count': level_counts['high'],
            'medium_count': level_counts['medium'],
            'low_count': level_counts['low'],
            'minimal_count': level_counts['minimal'],
            'bridge_count': bridge_count,
            'bridge_percentage': round(100 * bridge_count / len(scores), 2),
            'avg_fuzzy_score': round(sum(fuzzy_scores) / len(fuzzy_scores), 4),
            'max_fuzzy_score': round(max(fuzzy_scores), 4),
            'min_fuzzy_score': round(min(fuzzy_scores), 4)
        }


# ============================================================================
# Comparison Utilities
# ============================================================================

def compare_with_composite_score(G,
                                 fuzzy_scorer: FuzzyCriticalityScorer,
                                 alpha: float = 0.4,
                                 beta: float = 0.3,
                                 gamma: float = 0.3) -> Dict[str, Any]:
    """
    Compare fuzzy scores with traditional composite scores.
    
    Useful for validation and understanding differences between approaches.
    
    Args:
        G: NetworkX graph
        fuzzy_scorer: Configured fuzzy scorer
        alpha, beta, gamma: Weights for composite score formula
        
    Returns:
        Comparison statistics and correlations
    """
    if not NETWORKX_AVAILABLE:
        raise RuntimeError("NetworkX is required")
    
    node_scores, _ = fuzzy_scorer.analyze_graph(G)
    
    comparisons = []
    
    for node, fuzzy_result in node_scores.items():
        # Calculate traditional composite score
        composite = (
            alpha * fuzzy_result.betweenness_centrality_norm +
            beta * (1.0 if fuzzy_result.is_articulation_point else 0.0) +
            gamma * fuzzy_result.impact_score
        )
        
        comparisons.append({
            'node': node,
            'fuzzy_score': fuzzy_result.fuzzy_score,
            'composite_score': composite,
            'difference': fuzzy_result.fuzzy_score - composite,
            'fuzzy_level': fuzzy_result.criticality_level.value,
            'membership_degrees': fuzzy_result.membership_degrees
        })
    
    # Calculate correlation if numpy available
    if NUMPY_AVAILABLE and len(comparisons) > 2:
        fuzzy_vals = np.array([c['fuzzy_score'] for c in comparisons])
        composite_vals = np.array([c['composite_score'] for c in comparisons])
        
        correlation = np.corrcoef(fuzzy_vals, composite_vals)[0, 1]
        
        # Calculate rank correlation (Spearman)
        fuzzy_ranks = np.argsort(np.argsort(-fuzzy_vals))
        composite_ranks = np.argsort(np.argsort(-composite_vals))
        rank_correlation = np.corrcoef(fuzzy_ranks, composite_ranks)[0, 1]
    else:
        correlation = None
        rank_correlation = None
    
    return {
        'comparisons': comparisons,
        'pearson_correlation': correlation,
        'spearman_correlation': rank_correlation,
        'avg_difference': sum(c['difference'] for c in comparisons) / len(comparisons),
        'max_difference': max(abs(c['difference']) for c in comparisons)
    }


# ============================================================================
# Main - Testing and Demonstration
# ============================================================================

def main():
    """Demonstrate fuzzy criticality scoring"""
    print("=" * 70)
    print("Unified Fuzzy Criticality Scorer - Demonstration")
    print("=" * 70)
    
    if not NETWORKX_AVAILABLE:
        print("❌ NetworkX not available. Install with: pip install networkx")
        return
    
    # Create sample graph
    print("\n📊 Creating sample pub-sub graph...")
    G = nx.DiGraph()
    
    # Add nodes
    nodes = [
        ('broker_1', {'type': 'Broker'}),
        ('broker_2', {'type': 'Broker'}),
        ('topic_orders', {'type': 'Topic', 'qos_score': 0.9}),
        ('topic_events', {'type': 'Topic', 'qos_score': 0.5}),
        ('app_producer', {'type': 'Application'}),
        ('app_consumer_1', {'type': 'Application'}),
        ('app_consumer_2', {'type': 'Application'}),
        ('node_1', {'type': 'Node'}),
        ('node_2', {'type': 'Node'}),
    ]
    G.add_nodes_from(nodes)
    
    # Add edges
    edges = [
        ('app_producer', 'topic_orders', {'type': 'PUBLISHES'}),
        ('topic_orders', 'app_consumer_1', {'type': 'SUBSCRIBES'}),
        ('topic_orders', 'app_consumer_2', {'type': 'SUBSCRIBES'}),
        ('app_producer', 'topic_events', {'type': 'PUBLISHES'}),
        ('topic_events', 'app_consumer_1', {'type': 'SUBSCRIBES'}),
        ('topic_orders', 'broker_1', {'type': 'HOSTED_ON'}),
        ('topic_events', 'broker_2', {'type': 'HOSTED_ON'}),
        ('broker_1', 'broker_2', {'type': 'CONNECTS'}),  # Bridge edge
        ('broker_1', 'node_1', {'type': 'RUNS_ON'}),
        ('broker_2', 'node_2', {'type': 'RUNS_ON'}),
    ]
    G.add_edges_from(edges)
    
    print(f"   Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Initialize scorer
    print("\n🔧 Initializing Fuzzy Criticality Scorer...")
    scorer = FuzzyCriticalityScorer(
        defuzz_method=DefuzzificationMethod.CENTROID,
        calculate_impact=True
    )
    
    # Analyze graph
    print("\n📈 Analyzing graph criticality...")
    node_scores, edge_scores = scorer.analyze_graph(G)
    
    # Print node results
    print("\n" + "=" * 70)
    print("NODE CRITICALITY RESULTS (Fuzzy Logic)")
    print("=" * 70)
    
    top_nodes = scorer.get_top_critical_nodes(node_scores, n=5)
    for i, score in enumerate(top_nodes, 1):
        print(f"\n{i}. {score.component} ({score.component_type})")
        print(f"   Fuzzy Score: {score.fuzzy_score:.4f}")
        print(f"   Level: {score.criticality_level.value.upper()}")
        print(f"   Inputs: BC={score.betweenness_centrality_norm:.3f}, "
              f"AP={score.is_articulation_point}, Impact={score.impact_score:.3f}")
        print(f"   Memberships: {score._get_dominant_memberships()}")
    
    # Print edge results
    print("\n" + "=" * 70)
    print("EDGE CRITICALITY RESULTS (Fuzzy Logic)")
    print("=" * 70)
    
    top_edges = scorer.get_top_critical_edges(edge_scores, n=5)
    for i, score in enumerate(top_edges, 1):
        print(f"\n{i}. {score.source} → {score.target} ({score.edge_type})")
        print(f"   Fuzzy Score: {score.fuzzy_score:.4f}")
        print(f"   Level: {score.criticality_level.value.upper()}")
        print(f"   Inputs: EBC={score.edge_betweenness:.3f}, Bridge={score.is_bridge}, "
              f"Flow={score.flow_importance:.3f}")
        print(f"   Memberships: {score.membership_degrees}")
    
    # Print summaries
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    node_summary = scorer.summarize_node_criticality(node_scores)
    print("\nNode Criticality Distribution:")
    for key, value in node_summary.items():
        print(f"   {key}: {value}")
    
    edge_summary = scorer.summarize_edge_criticality(edge_scores)
    print("\nEdge Criticality Distribution:")
    for key, value in edge_summary.items():
        print(f"   {key}: {value}")
    
    # Compare with composite score
    print("\n" + "=" * 70)
    print("COMPARISON WITH TRADITIONAL COMPOSITE SCORE")
    print("=" * 70)
    
    comparison = compare_with_composite_score(G, scorer)
    print(f"\nPearson Correlation: {comparison['pearson_correlation']:.4f}")
    print(f"Spearman Rank Correlation: {comparison['spearman_correlation']:.4f}")
    print(f"Average Score Difference: {comparison['avg_difference']:.4f}")
    print(f"Maximum Score Difference: {comparison['max_difference']:.4f}")
    
    print("\n" + "=" * 70)
    print("✓ Fuzzy criticality analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
