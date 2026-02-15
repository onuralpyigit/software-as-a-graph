import pytest
from src.core import Application, Topic, QoSPolicy
from src.generation import GenerationService

class TestDomainModels:
    def test_qos_policy_defaults(self):
        qos = QoSPolicy()
        assert qos.reliability == "BEST_EFFORT"
        assert qos.durability == "VOLATILE"
        assert qos.transport_priority == "MEDIUM"

    def test_qos_weight_calculation(self):
        # Default: MEDIUM (0.1) + BEST_EFFORT (0.0) + VOLATILE (0.0) = 0.1
        qos = QoSPolicy()
        assert qos.calculate_weight() == 0.1
        
        # Max scores
        qos_max = QoSPolicy(
            reliability="RELIABLE", 
            durability="PERSISTENT", 
            transport_priority="URGENT"
        )
        # 0.3 + 0.4 + 0.3 = 1.0
        assert qos_max.calculate_weight() == 1.0

    def test_topic_weight_calculation(self):
        topic = Topic(id="t1", name="test", size=1024)
        # Size score = log2(1 + 1) / 10 = 1/10 = 0.1
        # QoS score (defaults) = 0.1
        # Total = ~0.2
        assert 0.19 < topic.calculate_weight() < 0.21

class TestGenerationService:
    def test_generate_tiny(self):
        gen = GenerationService(scale="tiny", seed=1)
        data = gen.generate()
        
        assert len(data["nodes"]) == 2
        assert len(data["brokers"]) == 1
        assert len(data["topics"]) == 5
        assert len(data["applications"]) == 5
