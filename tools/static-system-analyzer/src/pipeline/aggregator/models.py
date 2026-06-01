"""
Data models for the aggregator module.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Topic:
    """Represents a topic/topic with its QoS properties.
    
    Attributes:
        name: Topic/topic name
        size: Data size in bytes (optional)
        durability: DDS durability QoS setting
        reliability: DDS reliability QoS setting
        transport_priority: DDS transport priority QoS setting
    """
    name: str
    size: Optional[int] = None
    durability: Optional[str] = None
    reliability: Optional[str] = None
    transport_priority: Optional[int] = None
    
    def __hash__(self):
        return hash(self.name)
    
    def __eq__(self, other):
        if isinstance(other, Topic):
            return self.name == other.name
        return False
