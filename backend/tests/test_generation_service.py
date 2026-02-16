
import pytest
from src.generation import GenerationService

class TestGenerationService:
    def test_generate_tiny(self):
        gen = GenerationService(scale="tiny", seed=1)
        data = gen.generate()
        
        assert len(data["nodes"]) == 2
        assert len(data["brokers"]) == 1
        assert len(data["topics"]) == 5
        assert len(data["applications"]) == 5
