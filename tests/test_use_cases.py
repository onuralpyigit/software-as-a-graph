import pytest
from src.application.use_cases import ImportGraphUseCase
from src.adapters.persistence import InMemoryGraphRepository

class TestImportUseCase:
    def test_execute(self):
        repo = InMemoryGraphRepository()
        use_case = ImportGraphUseCase(repository=repo)
        
        data = {
            "applications": [{"id": "a1", "name": "app1"}],
            "nodes": [{"id": "n1", "name": "node1"}]
        }
        
        stats = use_case.execute(data)
        
        assert stats["application_count"] == 1
        assert stats["node_count"] == 1
