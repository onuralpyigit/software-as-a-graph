#!/usr/bin/env python3
"""
Fuzzy Logic Criticality Analyzer for Pub-Sub Systems

This module implements fuzzy logic techniques for criticality classification,
addressing the "sharp boundary problem" in traditional threshold-based approaches.

Key Features:
- Fuzzy membership functions (triangular, trapezoidal, gaussian)
- Mamdani fuzzy inference system
- Multi-input fuzzy rules for node and edge criticality
- Smooth transitions between criticality levels
- Handles measurement uncertainty naturally
- Supports both node and edge criticality analysis

Theory:
Instead of crisp boundaries (score >= 0.8 → Critical), fuzzy logic allows
partial membership in multiple categories. A node with betweenness=0.75 might be:
- 60% "High" criticality
- 40% "Critical" criticality

The final classification considers all membership degrees, providing more
nuanced and realistic criticality assessments.

Formula Integration:
The fuzzy system uses the same input metrics as the composite score:
- Betweenness Centrality (BC_norm)
- Articulation Point indicator (AP)
- Impact Score (I)

But instead of: C_score = α·BC + β·AP + γ·I
It applies fuzzy rules and defuzzification for smoother classification.

Author: Software-as-a-Graph Research Project
Version: 1.0
"""

import math
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import logging

# Check for optional numpy
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# Check for optional skfuzzy
try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    SKFUZZY_AVAILABLE = True
except ImportError:
    SKFUZZY_AVAILABLE = False


# ============================================================================
# Enums and Constants
# ============================================================================

class MembershipType(Enum):
    """Types of membership functions"""
    TRIANGULAR = "triangular"
    TRAPEZOIDAL = "trapezoidal"
    GAUSSIAN = "gaussian"
    SIGMOID = "sigmoid"


class FuzzyCriticalityLevel(Enum):
    """Fuzzy criticality levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


class DefuzzificationMethod(Enum):
    """Defuzzification methods"""
    CENTROID = "centroid"          # Center of gravity
    BISECTOR = "bisector"          # Bisector of area
    MOM = "mom"                    # Mean of maximum
    SOM = "som"                    # Smallest of maximum
    LOM = "lom"                    # Largest of maximum


# ============================================================================
# Pure Python Fuzzy Logic Implementation (No Dependencies)
# ============================================================================

@dataclass
class FuzzySet:
    """
    Represents a fuzzy set with a membership function
    
    Supports triangular, trapezoidal, and gaussian membership functions
    without requiring external dependencies.
    """
    name: str
    mf_type: MembershipType
    params: Tuple[float, ...]  # Parameters depend on membership type
    
    def membership(self, x: float) -> float:
        """
        Calculate membership degree for value x
        
        Args:
            x: Input value
            
        Returns:
            Membership degree in [0, 1]
        """
        if self.mf_type == MembershipType.TRIANGULAR:
            # params = (a, b, c) where a <= b <= c
            a, b, c = self.params
            if x <= a or x >= c:
                return 0.0
            elif a < x <= b:
                return (x - a) / (b - a) if b != a else 1.0
            else:  # b < x < c
                return (c - x) / (c - b) if c != b else 1.0
        
        elif self.mf_type == MembershipType.TRAPEZOIDAL:
            # params = (a, b, c, d) where a <= b <= c <= d
            a, b, c, d = self.params
            if x <= a or x >= d:
                return 0.0
            elif a < x < b:
                return (x - a) / (b - a) if b != a else 1.0
            elif b <= x <= c:
                return 1.0
            else:  # c < x < d
                return (d - x) / (d - c) if d != c else 1.0
        
        elif self.mf_type == MembershipType.GAUSSIAN:
            # params = (mean, sigma)
            mean, sigma = self.params
            if sigma == 0:
                return 1.0 if x == mean else 0.0
            return math.exp(-0.5 * ((x - mean) / sigma) ** 2)
        
        elif self.mf_type == MembershipType.SIGMOID:
            # params = (c, a) where c is center, a is slope
            c, a = self.params
            return 1.0 / (1.0 + math.exp(-a * (x - c)))
        
        return 0.0


@dataclass
class FuzzyVariable:
    """
    Represents a fuzzy linguistic variable
    
    A linguistic variable has multiple fuzzy sets representing
    linguistic terms (e.g., "low", "medium", "high").
    """
    name: str
    universe: Tuple[float, float]  # (min, max) range
    fuzzy_sets: Dict[str, FuzzySet] = field(default_factory=dict)
    
    def add_set(self, name: str, mf_type: MembershipType, params: Tuple[float, ...]):
        """Add a fuzzy set to this variable"""
        self.fuzzy_sets[name] = FuzzySet(name, mf_type, params)
    
    def fuzzify(self, value: float) -> Dict[str, float]:
        """
        Fuzzify a crisp value into membership degrees
        
        Args:
            value: Crisp input value
            
        Returns:
            Dictionary mapping set names to membership degrees
        """
        return {
            name: fs.membership(value)
            for name, fs in self.fuzzy_sets.items()
        }
    
    def get_membership(self, value: float, set_name: str) -> float:
        """Get membership degree for specific set"""
        if set_name in self.fuzzy_sets:
            return self.fuzzy_sets[set_name].membership(value)
        return 0.0


@dataclass
class FuzzyRule:
    """
    Represents a fuzzy IF-THEN rule
    
    Format: IF (antecedent1 AND antecedent2 ...) THEN consequent
    
    Antecedents: List of (variable_name, set_name) tuples
    Consequent: (variable_name, set_name) tuple
    """
    antecedents: List[Tuple[str, str]]  # [(var_name, set_name), ...]
    consequent: Tuple[str, str]         # (var_name, set_name)
    weight: float = 1.0                 # Rule weight
    operator: str = "AND"               # AND or OR for combining antecedents
    
    def evaluate(self, memberships: Dict[str, Dict[str, float]]) -> float:
        """
        Evaluate the rule given membership degrees
        
        Args:
            memberships: Dict of {var_name: {set_name: degree}}
            
        Returns:
            Rule firing strength
        """
        if not self.antecedents:
            return 0.0
        
        # Get membership degrees for all antecedents
        degrees = []
        for var_name, set_name in self.antecedents:
            if var_name in memberships and set_name in memberships[var_name]:
                degrees.append(memberships[var_name][set_name])
            else:
                degrees.append(0.0)
        
        # Combine using operator
        if self.operator == "AND":
            firing_strength = min(degrees)  # Minimum for AND (Mamdani)
        else:  # OR
            firing_strength = max(degrees)  # Maximum for OR
        
        return firing_strength * self.weight


class FuzzyInferenceSystem:
    """
    Mamdani Fuzzy Inference System
    
    Implements the complete fuzzy inference process:
    1. Fuzzification - Convert crisp inputs to fuzzy memberships
    2. Rule Evaluation - Apply fuzzy rules
    3. Aggregation - Combine rule outputs
    4. Defuzzification - Convert fuzzy output to crisp value
    """
    
    def __init__(self, name: str = "FIS"):
        self.name = name
        self.input_variables: Dict[str, FuzzyVariable] = {}
        self.output_variables: Dict[str, FuzzyVariable] = {}
        self.rules: List[FuzzyRule] = []
        self.logger = logging.getLogger(f'fuzzy.{name}')
    
    def add_input(self, variable: FuzzyVariable):
        """Add an input variable"""
        self.input_variables[variable.name] = variable
    
    def add_output(self, variable: FuzzyVariable):
        """Add an output variable"""
        self.output_variables[variable.name] = variable
    
    def add_rule(self, rule: FuzzyRule):
        """Add a fuzzy rule"""
        self.rules.append(rule)
    
    def fuzzify_inputs(self, inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Fuzzify all input values
        
        Args:
            inputs: Dict of {variable_name: crisp_value}
            
        Returns:
            Dict of {variable_name: {set_name: membership_degree}}
        """
        memberships = {}
        for var_name, value in inputs.items():
            if var_name in self.input_variables:
                memberships[var_name] = self.input_variables[var_name].fuzzify(value)
        return memberships
    
    def evaluate_rules(
        self, 
        memberships: Dict[str, Dict[str, float]]
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Evaluate all rules and get consequent activations
        
        Args:
            memberships: Fuzzified input memberships
            
        Returns:
            Dict of {output_var: [(set_name, activation), ...]}
        """
        activations = defaultdict(list)
        
        for rule in self.rules:
            firing_strength = rule.evaluate(memberships)
            
            if firing_strength > 0:
                output_var, output_set = rule.consequent
                activations[output_var].append((output_set, firing_strength))
        
        return dict(activations)
    
    def aggregate(
        self, 
        activations: Dict[str, List[Tuple[str, float]]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate rule outputs using maximum
        
        Args:
            activations: Rule activations by output variable
            
        Returns:
            Aggregated memberships for each output set
        """
        aggregated = {}
        
        for output_var, acts in activations.items():
            set_activations = defaultdict(float)
            for set_name, strength in acts:
                # Use maximum for aggregation
                set_activations[set_name] = max(
                    set_activations[set_name], 
                    strength
                )
            aggregated[output_var] = dict(set_activations)
        
        return aggregated
    
    def defuzzify(
        self, 
        aggregated: Dict[str, Dict[str, float]],
        method: DefuzzificationMethod = DefuzzificationMethod.CENTROID
    ) -> Dict[str, float]:
        """
        Defuzzify aggregated outputs to crisp values
        
        Args:
            aggregated: Aggregated output memberships
            method: Defuzzification method
            
        Returns:
            Dict of {output_var: crisp_value}
        """
        outputs = {}
        
        for output_var, set_activations in aggregated.items():
            if output_var not in self.output_variables:
                continue
            
            var = self.output_variables[output_var]
            
            if method == DefuzzificationMethod.CENTROID:
                outputs[output_var] = self._defuzzify_centroid(var, set_activations)
            elif method == DefuzzificationMethod.MOM:
                outputs[output_var] = self._defuzzify_mom(var, set_activations)
            else:
                outputs[output_var] = self._defuzzify_centroid(var, set_activations)
        
        return outputs
    
    def _defuzzify_centroid(
        self, 
        var: FuzzyVariable, 
        activations: Dict[str, float],
        resolution: int = 100
    ) -> float:
        """Centroid (center of gravity) defuzzification"""
        min_val, max_val = var.universe
        step = (max_val - min_val) / resolution
        
        numerator = 0.0
        denominator = 0.0
        
        for i in range(resolution + 1):
            x = min_val + i * step
            
            # Calculate aggregated membership at this point
            max_membership = 0.0
            for set_name, activation in activations.items():
                if set_name in var.fuzzy_sets:
                    # Clip membership by activation (implication)
                    membership = min(
                        var.fuzzy_sets[set_name].membership(x),
                        activation
                    )
                    max_membership = max(max_membership, membership)
            
            numerator += x * max_membership
            denominator += max_membership
        
        if denominator == 0:
            return (min_val + max_val) / 2  # Default to midpoint
        
        return numerator / denominator
    
    def _defuzzify_mom(
        self, 
        var: FuzzyVariable, 
        activations: Dict[str, float],
        resolution: int = 100
    ) -> float:
        """Mean of Maximum defuzzification"""
        min_val, max_val = var.universe
        step = (max_val - min_val) / resolution
        
        max_membership = 0.0
        max_points = []
        
        for i in range(resolution + 1):
            x = min_val + i * step
            
            # Calculate aggregated membership
            membership = 0.0
            for set_name, activation in activations.items():
                if set_name in var.fuzzy_sets:
                    m = min(var.fuzzy_sets[set_name].membership(x), activation)
                    membership = max(membership, m)
            
            if membership > max_membership:
                max_membership = membership
                max_points = [x]
            elif membership == max_membership and membership > 0:
                max_points.append(x)
        
        if not max_points:
            return (min_val + max_val) / 2
        
        return sum(max_points) / len(max_points)
    
    def infer(
        self, 
        inputs: Dict[str, float],
        method: DefuzzificationMethod = DefuzzificationMethod.CENTROID
    ) -> Dict[str, float]:
        """
        Complete fuzzy inference
        
        Args:
            inputs: Crisp input values
            method: Defuzzification method
            
        Returns:
            Crisp output values
        """
        # Step 1: Fuzzify inputs
        memberships = self.fuzzify_inputs(inputs)
        
        # Step 2: Evaluate rules
        activations = self.evaluate_rules(memberships)
        
        # Step 3: Aggregate
        aggregated = self.aggregate(activations)
        
        # Step 4: Defuzzify
        outputs = self.defuzzify(aggregated, method)
        
        return outputs
    
    def get_membership_degrees(self, inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """Get all membership degrees for inputs (for analysis)"""
        return self.fuzzify_inputs(inputs)


# ============================================================================
# Node Criticality Fuzzy System
# ============================================================================

def create_node_criticality_fis() -> FuzzyInferenceSystem:
    """
    Create Fuzzy Inference System for node criticality
    
    Input Variables:
    - betweenness: Normalized betweenness centrality [0, 1]
    - articulation_point: AP indicator as continuous [0, 1]
    - impact: Impact score from reachability analysis [0, 1]
    
    Output Variable:
    - criticality: Criticality level [0, 1]
    
    Returns:
        Configured FIS for node criticality
    """
    fis = FuzzyInferenceSystem("NodeCriticality")
    
    # === Input Variable: Betweenness Centrality ===
    betweenness = FuzzyVariable("betweenness", (0.0, 1.0))
    betweenness.add_set("very_low", MembershipType.TRAPEZOIDAL, (0.0, 0.0, 0.1, 0.2))
    betweenness.add_set("low", MembershipType.TRIANGULAR, (0.1, 0.25, 0.4))
    betweenness.add_set("medium", MembershipType.TRIANGULAR, (0.3, 0.5, 0.7))
    betweenness.add_set("high", MembershipType.TRIANGULAR, (0.6, 0.75, 0.9))
    betweenness.add_set("very_high", MembershipType.TRAPEZOIDAL, (0.8, 0.9, 1.0, 1.0))
    fis.add_input(betweenness)
    
    # === Input Variable: Articulation Point (continuous for fuzzy) ===
    # Even though AP is binary, we treat it as continuous for fuzzy processing
    # Values near 1 indicate high confidence of being AP
    ap = FuzzyVariable("articulation_point", (0.0, 1.0))
    ap.add_set("no", MembershipType.TRAPEZOIDAL, (0.0, 0.0, 0.2, 0.4))
    ap.add_set("uncertain", MembershipType.TRIANGULAR, (0.3, 0.5, 0.7))
    ap.add_set("yes", MembershipType.TRAPEZOIDAL, (0.6, 0.8, 1.0, 1.0))
    fis.add_input(ap)
    
    # === Input Variable: Impact Score ===
    impact = FuzzyVariable("impact", (0.0, 1.0))
    impact.add_set("negligible", MembershipType.TRAPEZOIDAL, (0.0, 0.0, 0.1, 0.2))
    impact.add_set("minor", MembershipType.TRIANGULAR, (0.1, 0.25, 0.4))
    impact.add_set("moderate", MembershipType.TRIANGULAR, (0.3, 0.5, 0.7))
    impact.add_set("major", MembershipType.TRIANGULAR, (0.6, 0.75, 0.9))
    impact.add_set("catastrophic", MembershipType.TRAPEZOIDAL, (0.8, 0.9, 1.0, 1.0))
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
    # Rule structure: IF (conditions) THEN criticality IS level
    
    # Critical rules - any strong indicator leads to critical
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
    
    # High criticality rules
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
    
    # Medium criticality rules
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
    
    # Low criticality rules
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
    
    # Minimal criticality rules
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
    
    return fis


# ============================================================================
# Edge Criticality Fuzzy System
# ============================================================================

def create_edge_criticality_fis() -> FuzzyInferenceSystem:
    """
    Create Fuzzy Inference System for edge criticality
    
    Input Variables:
    - edge_betweenness: Normalized edge betweenness centrality [0, 1]
    - is_bridge: Bridge indicator as continuous [0, 1]
    - flow_importance: Message flow importance [0, 1]
    
    Output Variable:
    - edge_criticality: Edge criticality level [0, 1]
    
    Returns:
        Configured FIS for edge criticality
    """
    fis = FuzzyInferenceSystem("EdgeCriticality")
    
    # === Input Variable: Edge Betweenness ===
    edge_betweenness = FuzzyVariable("edge_betweenness", (0.0, 1.0))
    edge_betweenness.add_set("very_low", MembershipType.TRAPEZOIDAL, (0.0, 0.0, 0.1, 0.2))
    edge_betweenness.add_set("low", MembershipType.TRIANGULAR, (0.1, 0.25, 0.4))
    edge_betweenness.add_set("medium", MembershipType.TRIANGULAR, (0.3, 0.5, 0.7))
    edge_betweenness.add_set("high", MembershipType.TRIANGULAR, (0.6, 0.75, 0.9))
    edge_betweenness.add_set("very_high", MembershipType.TRAPEZOIDAL, (0.8, 0.9, 1.0, 1.0))
    fis.add_input(edge_betweenness)
    
    # === Input Variable: Bridge Status ===
    bridge = FuzzyVariable("is_bridge", (0.0, 1.0))
    bridge.add_set("no", MembershipType.TRAPEZOIDAL, (0.0, 0.0, 0.2, 0.4))
    bridge.add_set("uncertain", MembershipType.TRIANGULAR, (0.3, 0.5, 0.7))
    bridge.add_set("yes", MembershipType.TRAPEZOIDAL, (0.6, 0.8, 1.0, 1.0))
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
    
    # Critical edge rules
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
    
    # High edge criticality rules
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "high"), ("is_bridge", "yes")],
        ("edge_criticality", "high"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "very_high"), ("flow_importance", "significant")],
        ("edge_criticality", "high"), weight=1.0
    ))
    fis.add_rule(FuzzyRule(
        [("is_bridge", "yes"), ("flow_importance", "significant")],
        ("edge_criticality", "high"), weight=0.9
    ))
    fis.add_rule(FuzzyRule(
        [("edge_betweenness", "high"), ("flow_importance", "significant")],
        ("edge_criticality", "high"), weight=1.0
    ))
    
    # Medium edge criticality rules
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
    
    # Low edge criticality rules
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
    
    # Minimal edge criticality rules
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
    
    return fis


# ============================================================================
# Fuzzy Criticality Analyzer
# ============================================================================

@dataclass
class FuzzyNodeCriticalityResult:
    """Result of fuzzy node criticality analysis"""
    component: str
    component_type: str
    
    # Raw input values
    betweenness_centrality_norm: float
    is_articulation_point: bool
    impact_score: float
    
    # Fuzzy analysis results
    fuzzy_criticality_score: float
    criticality_level: FuzzyCriticalityLevel
    
    # Membership degrees for transparency
    membership_degrees: Dict[str, float]
    
    # Input membership degrees
    input_memberships: Dict[str, Dict[str, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'component': self.component,
            'component_type': self.component_type,
            'betweenness_centrality_norm': round(self.betweenness_centrality_norm, 4),
            'is_articulation_point': self.is_articulation_point,
            'impact_score': round(self.impact_score, 4),
            'fuzzy_criticality_score': round(self.fuzzy_criticality_score, 4),
            'criticality_level': self.criticality_level.value,
            'membership_degrees': {
                k: round(v, 4) for k, v in self.membership_degrees.items()
            },
            'input_memberships': {
                var: {s: round(m, 4) for s, m in sets.items()}
                for var, sets in self.input_memberships.items()
            }
        }


@dataclass
class FuzzyEdgeCriticalityResult:
    """Result of fuzzy edge criticality analysis"""
    source: str
    target: str
    edge_type: str
    
    # Raw input values
    edge_betweenness_norm: float
    is_bridge: bool
    flow_importance: float
    
    # Fuzzy analysis results
    fuzzy_criticality_score: float
    criticality_level: FuzzyCriticalityLevel
    
    # Membership degrees
    membership_degrees: Dict[str, float]
    input_memberships: Dict[str, Dict[str, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'source': self.source,
            'target': self.target,
            'edge_type': self.edge_type,
            'edge_betweenness_norm': round(self.edge_betweenness_norm, 4),
            'is_bridge': self.is_bridge,
            'flow_importance': round(self.flow_importance, 4),
            'fuzzy_criticality_score': round(self.fuzzy_criticality_score, 4),
            'criticality_level': self.criticality_level.value,
            'membership_degrees': {
                k: round(v, 4) for k, v in self.membership_degrees.items()
            }
        }


class FuzzyCriticalityAnalyzer:
    """
    Comprehensive fuzzy criticality analyzer for pub-sub systems
    
    Applies fuzzy logic to both node and edge criticality analysis,
    providing smooth transitions between criticality levels and
    handling measurement uncertainty.
    """
    
    def __init__(
        self,
        defuzzification_method: DefuzzificationMethod = DefuzzificationMethod.CENTROID
    ):
        """
        Initialize the fuzzy analyzer
        
        Args:
            defuzzification_method: Method for converting fuzzy output to crisp value
        """
        self.defuzz_method = defuzzification_method
        self.node_fis = create_node_criticality_fis()
        self.edge_fis = create_edge_criticality_fis()
        self.logger = logging.getLogger('fuzzy.analyzer')
    
    def _score_to_level(self, score: float) -> FuzzyCriticalityLevel:
        """
        Convert fuzzy criticality score to level
        
        Uses the output membership functions to determine the level
        with highest membership.
        """
        # Get membership in each output category
        output_var = self.node_fis.output_variables.get("criticality")
        if not output_var:
            # Fallback
            if score >= 0.75:
                return FuzzyCriticalityLevel.CRITICAL
            elif score >= 0.55:
                return FuzzyCriticalityLevel.HIGH
            elif score >= 0.35:
                return FuzzyCriticalityLevel.MEDIUM
            elif score >= 0.15:
                return FuzzyCriticalityLevel.LOW
            else:
                return FuzzyCriticalityLevel.MINIMAL
        
        # Find level with maximum membership
        max_membership = 0.0
        best_level = FuzzyCriticalityLevel.MINIMAL
        
        level_mapping = {
            'critical': FuzzyCriticalityLevel.CRITICAL,
            'high': FuzzyCriticalityLevel.HIGH,
            'medium': FuzzyCriticalityLevel.MEDIUM,
            'low': FuzzyCriticalityLevel.LOW,
            'minimal': FuzzyCriticalityLevel.MINIMAL
        }
        
        for set_name, fuzzy_set in output_var.fuzzy_sets.items():
            membership = fuzzy_set.membership(score)
            if membership > max_membership:
                max_membership = membership
                best_level = level_mapping.get(set_name, FuzzyCriticalityLevel.MINIMAL)
        
        return best_level
    
    def _get_output_memberships(
        self, 
        score: float, 
        fis: FuzzyInferenceSystem,
        output_var_name: str
    ) -> Dict[str, float]:
        """Get membership degrees for output score"""
        output_var = fis.output_variables.get(output_var_name)
        if not output_var:
            return {}
        
        return {
            set_name: round(fuzzy_set.membership(score), 4)
            for set_name, fuzzy_set in output_var.fuzzy_sets.items()
        }
    
    def analyze_node(
        self,
        node_id: str,
        node_type: str,
        betweenness_norm: float,
        is_articulation_point: bool,
        impact_score: float
    ) -> FuzzyNodeCriticalityResult:
        """
        Analyze criticality of a single node using fuzzy logic
        
        Args:
            node_id: Node identifier
            node_type: Type of node (Application, Broker, etc.)
            betweenness_norm: Normalized betweenness centrality [0, 1]
            is_articulation_point: Whether node is an articulation point
            impact_score: Impact score from reachability analysis [0, 1]
        
        Returns:
            Fuzzy criticality analysis result
        """
        # Prepare inputs (convert boolean AP to continuous)
        inputs = {
            'betweenness': min(1.0, max(0.0, betweenness_norm)),
            'articulation_point': 1.0 if is_articulation_point else 0.0,
            'impact': min(1.0, max(0.0, impact_score))
        }
        
        # Get input memberships for transparency
        input_memberships = self.node_fis.fuzzify_inputs(inputs)
        
        # Run fuzzy inference
        outputs = self.node_fis.infer(inputs, self.defuzz_method)
        
        # Get criticality score
        fuzzy_score = outputs.get('criticality', 0.5)
        
        # Get output memberships
        output_memberships = self._get_output_memberships(
            fuzzy_score, self.node_fis, 'criticality'
        )
        
        # Determine level
        level = self._score_to_level(fuzzy_score)
        
        return FuzzyNodeCriticalityResult(
            component=node_id,
            component_type=node_type,
            betweenness_centrality_norm=betweenness_norm,
            is_articulation_point=is_articulation_point,
            impact_score=impact_score,
            fuzzy_criticality_score=fuzzy_score,
            criticality_level=level,
            membership_degrees=output_memberships,
            input_memberships=input_memberships
        )
    
    def analyze_edge(
        self,
        source: str,
        target: str,
        edge_type: str,
        edge_betweenness_norm: float,
        is_bridge: bool,
        flow_importance: float
    ) -> FuzzyEdgeCriticalityResult:
        """
        Analyze criticality of a single edge using fuzzy logic
        
        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Type of edge (PUBLISHES, SUBSCRIBES, etc.)
            edge_betweenness_norm: Normalized edge betweenness [0, 1]
            is_bridge: Whether edge is a bridge
            flow_importance: Importance of message flow [0, 1]
        
        Returns:
            Fuzzy edge criticality analysis result
        """
        # Prepare inputs
        inputs = {
            'edge_betweenness': min(1.0, max(0.0, edge_betweenness_norm)),
            'is_bridge': 1.0 if is_bridge else 0.0,
            'flow_importance': min(1.0, max(0.0, flow_importance))
        }
        
        # Get input memberships
        input_memberships = self.edge_fis.fuzzify_inputs(inputs)
        
        # Run fuzzy inference
        outputs = self.edge_fis.infer(inputs, self.defuzz_method)
        
        # Get criticality score
        fuzzy_score = outputs.get('edge_criticality', 0.5)
        
        # Get output memberships
        output_memberships = self._get_output_memberships(
            fuzzy_score, self.edge_fis, 'edge_criticality'
        )
        
        # Determine level
        level = self._score_to_level(fuzzy_score)
        
        return FuzzyEdgeCriticalityResult(
            source=source,
            target=target,
            edge_type=edge_type,
            edge_betweenness_norm=edge_betweenness_norm,
            is_bridge=is_bridge,
            flow_importance=flow_importance,
            fuzzy_criticality_score=fuzzy_score,
            criticality_level=level,
            membership_degrees=output_memberships,
            input_memberships=input_memberships
        )


# ============================================================================
# Integration with NetworkX Graph Analysis
# ============================================================================

def analyze_graph_with_fuzzy_logic(
    G,  # nx.DiGraph
    calculate_impact: bool = True,
    defuzz_method: DefuzzificationMethod = DefuzzificationMethod.CENTROID
) -> Tuple[Dict[str, FuzzyNodeCriticalityResult], Dict[Tuple[str, str], FuzzyEdgeCriticalityResult]]:
    """
    Analyze entire graph using fuzzy logic for both nodes and edges
    
    Args:
        G: NetworkX directed graph
        calculate_impact: Whether to calculate impact scores (slower)
        defuzz_method: Defuzzification method
    
    Returns:
        Tuple of (node_results, edge_results) dictionaries
    """
    import networkx as nx
    
    logger = logging.getLogger('fuzzy.graph_analysis')
    logger.info("Starting fuzzy logic graph analysis...")
    
    # Initialize analyzer
    analyzer = FuzzyCriticalityAnalyzer(defuzz_method)
    
    # Calculate metrics
    logger.info("Calculating betweenness centrality...")
    betweenness = nx.betweenness_centrality(G, normalized=True)
    max_bc = max(betweenness.values()) if betweenness else 1.0
    if max_bc == 0:
        max_bc = 1.0
    
    logger.info("Finding articulation points...")
    undirected = G.to_undirected()
    articulation_points = set(nx.articulation_points(undirected))
    
    logger.info("Calculating edge betweenness...")
    edge_betweenness = nx.edge_betweenness_centrality(G, normalized=True)
    max_ebc = max(edge_betweenness.values()) if edge_betweenness else 1.0
    if max_ebc == 0:
        max_ebc = 1.0
    
    logger.info("Finding bridges...")
    bridges = set(nx.bridges(undirected))
    
    # Analyze nodes
    logger.info("Analyzing nodes with fuzzy logic...")
    node_results = {}
    
    for node in G.nodes():
        node_data = G.nodes[node]
        node_type = node_data.get('type', 'Unknown')
        
        # Get betweenness (normalized)
        bc_norm = betweenness.get(node, 0.0) / max_bc
        
        # Check if articulation point
        is_ap = node in articulation_points
        
        # Calculate impact score if enabled
        if calculate_impact:
            impact = _calculate_node_impact(G, node)
        else:
            # Estimate from degree
            degree = G.degree(node)
            max_degree = max(d for _, d in G.degree()) if G.degree() else 1
            impact = degree / max_degree
        
        # Run fuzzy analysis
        result = analyzer.analyze_node(
            node_id=node,
            node_type=node_type,
            betweenness_norm=bc_norm,
            is_articulation_point=is_ap,
            impact_score=impact
        )
        
        node_results[node] = result
    
    # Analyze edges
    logger.info("Analyzing edges with fuzzy logic...")
    edge_results = {}
    
    for source, target, data in G.edges(data=True):
        edge_type = data.get('type', 'Unknown')
        
        # Get edge betweenness (normalized)
        ebc = edge_betweenness.get((source, target), 0.0)
        ebc_norm = ebc / max_ebc
        
        # Check if bridge
        is_bridge = (source, target) in bridges or (target, source) in bridges
        
        # Calculate flow importance
        flow = _calculate_flow_importance(G, source, target, data)
        
        # Run fuzzy analysis
        result = analyzer.analyze_edge(
            source=source,
            target=target,
            edge_type=edge_type,
            edge_betweenness_norm=ebc_norm,
            is_bridge=is_bridge,
            flow_importance=flow
        )
        
        edge_results[(source, target)] = result
    
    logger.info(f"Fuzzy analysis complete: {len(node_results)} nodes, {len(edge_results)} edges")
    
    return node_results, edge_results


def _calculate_node_impact(G, node: str) -> float:
    """Calculate impact score for a node based on reachability loss"""
    import networkx as nx
    
    if node not in G:
        return 0.0
    
    # Original reachability count
    original_reachable = sum(1 for n in G.nodes() if n != node 
                            for _ in nx.descendants(G, n))
    
    if original_reachable == 0:
        return 0.0
    
    # Remove node and recalculate
    G_copy = G.copy()
    G_copy.remove_node(node)
    
    new_reachable = sum(1 for n in G_copy.nodes() 
                       for _ in nx.descendants(G_copy, n))
    
    # Calculate loss ratio
    loss = (original_reachable - new_reachable) / original_reachable
    
    return min(1.0, max(0.0, loss))


def _calculate_flow_importance(G, source: str, target: str, edge_data: Dict) -> float:
    """
    Calculate flow importance for an edge
    
    Based on:
    - Edge type (PUBLISHES/SUBSCRIBES are more important)
    - Connected node degrees
    - QoS if available
    """
    edge_type = edge_data.get('type', 'Unknown')
    
    # Base importance by type
    type_importance = {
        'PUBLISHES': 0.8,
        'SUBSCRIBES': 0.8,
        'RUNS_ON': 0.5,
        'ROUTES_TO': 0.7
    }
    base = type_importance.get(edge_type, 0.5)
    
    # Adjust by node degrees
    source_degree = G.degree(source)
    target_degree = G.degree(target)
    max_degree = max(d for _, d in G.degree()) if G.degree() else 1
    
    degree_factor = (source_degree + target_degree) / (2 * max_degree)
    
    # QoS factor if available
    qos = edge_data.get('qos', {})
    reliability = qos.get('reliability', 'BEST_EFFORT')
    qos_factor = 1.0 if reliability == 'RELIABLE' else 0.7
    
    # Combine factors
    flow_importance = base * 0.5 + degree_factor * 0.3 + qos_factor * 0.2
    
    return min(1.0, max(0.0, flow_importance))


# ============================================================================
# Visualization Support for Fuzzy Analysis
# ============================================================================

def generate_fuzzy_membership_chart(
    fis: FuzzyInferenceSystem,
    variable_name: str,
    output_path: str,
    is_input: bool = True
):
    """
    Generate membership function chart for a variable
    
    Args:
        fis: Fuzzy inference system
        variable_name: Name of variable to plot
        output_path: Path to save chart
        is_input: Whether this is an input or output variable
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return
    
    variables = fis.input_variables if is_input else fis.output_variables
    if variable_name not in variables:
        print(f"Variable {variable_name} not found")
        return
    
    var = variables[variable_name]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate x values
    min_val, max_val = var.universe
    x = [min_val + i * (max_val - min_val) / 200 for i in range(201)]
    
    # Plot each fuzzy set
    colors = ['#e74c3c', '#f39c12', '#3498db', '#27ae60', '#9b59b6']
    
    for i, (set_name, fuzzy_set) in enumerate(var.fuzzy_sets.items()):
        y = [fuzzy_set.membership(xi) for xi in x]
        color = colors[i % len(colors)]
        ax.plot(x, y, label=set_name.replace('_', ' ').title(), 
               linewidth=2, color=color)
        ax.fill_between(x, y, alpha=0.2, color=color)
    
    ax.set_xlabel(variable_name.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel('Membership Degree', fontsize=12)
    ax.set_title(f'Fuzzy Membership Functions: {variable_name.replace("_", " ").title()}',
                fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(min_val, max_val)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved membership chart: {output_path}")


# ============================================================================
# Main Function for Testing
# ============================================================================

def main():
    """Test fuzzy logic system"""
    print("=" * 60)
    print("Fuzzy Logic Criticality Analyzer - Test")
    print("=" * 60)
    
    # Create analyzer
    analyzer = FuzzyCriticalityAnalyzer()
    
    # Test cases for nodes
    print("\n📊 Node Criticality Analysis:")
    print("-" * 50)
    
    test_nodes = [
        ("N1", "Node", 0.9, True, 0.8),    # High BC, AP, High impact -> Critical
        ("B1", "Broker", 0.7, True, 0.5),  # High BC, AP, Medium impact -> High/Critical
        ("T1", "Topic", 0.5, False, 0.6),  # Medium BC, not AP, Medium impact -> Medium
        ("A1", "Application", 0.2, False, 0.3),  # Low BC, not AP, Low impact -> Low
        ("A2", "Application", 0.1, False, 0.1),  # Very low everything -> Minimal
    ]
    
    for node_id, node_type, bc, is_ap, impact in test_nodes:
        result = analyzer.analyze_node(node_id, node_type, bc, is_ap, impact)
        print(f"\n  {node_id} ({node_type}):")
        print(f"    Inputs: BC={bc:.2f}, AP={is_ap}, Impact={impact:.2f}")
        print(f"    Fuzzy Score: {result.fuzzy_criticality_score:.4f}")
        print(f"    Level: {result.criticality_level.value.upper()}")
        print(f"    Memberships: {result.membership_degrees}")
    
    # Test cases for edges
    print("\n\n📊 Edge Criticality Analysis:")
    print("-" * 50)
    
    test_edges = [
        ("A1", "T1", "PUBLISHES", 0.9, True, 0.9),   # High everything -> Critical
        ("T1", "A2", "SUBSCRIBES", 0.6, True, 0.5),  # Medium BC, bridge -> High
        ("T2", "B1", "RUNS_ON", 0.4, False, 0.5),    # Medium everything -> Medium
        ("A3", "N1", "RUNS_ON", 0.2, False, 0.3),    # Low everything -> Low
        ("A4", "N2", "RUNS_ON", 0.1, False, 0.1),    # Very low -> Minimal
    ]
    
    for source, target, edge_type, ebc, is_bridge, flow in test_edges:
        result = analyzer.analyze_edge(source, target, edge_type, ebc, is_bridge, flow)
        print(f"\n  {source} -> {target} ({edge_type}):")
        print(f"    Inputs: EBC={ebc:.2f}, Bridge={is_bridge}, Flow={flow:.2f}")
        print(f"    Fuzzy Score: {result.fuzzy_criticality_score:.4f}")
        print(f"    Level: {result.criticality_level.value.upper()}")
        print(f"    Memberships: {result.membership_degrees}")
    
    print("\n" + "=" * 60)
    print("✓ Fuzzy logic tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
