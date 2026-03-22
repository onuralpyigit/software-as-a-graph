import pytest
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from tools.generation import GenerationService

class TestGenerationService:
    def test_generate_tiny(self):
        gen = GenerationService(scale="tiny", seed=1)
        data = gen.generate()
        
        assert len(data["nodes"]) == 2
        assert len(data["brokers"]) == 1
        assert len(data["topics"]) == 5
        assert len(data["applications"]) == 5
