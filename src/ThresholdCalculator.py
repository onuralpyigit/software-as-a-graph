from dataclasses import dataclass
from typing import List
from enum import Enum
import numpy as np
import networkx as nx

class CriticalityLevel(Enum):
    VERY_HIGH = "VERY_HIGH"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"

    def to_str(self):
        return self.value  # Return the string representation of the enum

@dataclass
class ComponentThresholds:
    """Thresholds for different component types"""
    very_high_threshold: float
    high_threshold: float
    medium_threshold: float
    low_threshold: float
    very_low_threshold: float
    
    def get_criticality(self, score: float) -> CriticalityLevel:
        """Get criticality level based on the score"""
        if score >= self.very_high_threshold:
            return CriticalityLevel.VERY_HIGH
        elif score >= self.high_threshold:
            return CriticalityLevel.HIGH
        elif score >= self.medium_threshold:
            return CriticalityLevel.MEDIUM
        elif score >= self.low_threshold:
            return CriticalityLevel.LOW
        else:
            return CriticalityLevel.VERY_LOW

class ThresholdCalculator:        
    def calculate_statistical_thresholds(self, scores: List[float]) -> ComponentThresholds:
        """Calculate thresholds using statistical methods"""
        if not scores:
            return ComponentThresholds(0.8, 0.5)  # Default fallback
            
        scores_array = np.array(scores)
        mean = np.mean(scores_array)
        std = np.std(scores_array)
        
        return ComponentThresholds(
            high_threshold=mean + std,  # High: Above 1 standard deviation
            medium_threshold=mean       # Medium: Above mean
        )
    
    def calculate_percentile_thresholds(self, scores: List[float]) -> ComponentThresholds:
        """Calculate thresholds using percentile method"""
        if not scores:
            return ComponentThresholds(0.9, 0.75, 0.5, 0.25, 0.1)  # Default fallback
        
        # Calculate Q1, Q2, Q3
        q1 = np.percentile(scores, 25)
        q2 = np.percentile(scores, 50)
        q3 = np.percentile(scores, 75)

        # Calculate IQR
        iqr = q3 - q1

        return ComponentThresholds(
            very_high_threshold=q3 + 1.5 * iqr,  # Very High: Above Q3 + 1.5 * IQR
            high_threshold=q3,                   # High: Above Q3
            medium_threshold=q2,                 # Medium: Above Q2
            low_threshold=q1,                    # Low: Above Q1
            very_low_threshold=q1 - 1.5 * iqr    # Very Low: Below Q1 - 1.5 * IQR
        )
    