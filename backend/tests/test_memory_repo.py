from src.core.memory_repo import InMemoryGraphRepository
from src.core import Application

class TestInMemoryRepository:
    def test_save_and_retrieve(self):
        repo = InMemoryGraphRepository()
        
        data = {
            "applications": [
                {"id": "a1", "name": "App 1", "role": "pub", "app_type": "service", "criticality": False, "version": "1.0"},
            ],
            "relationships": {
                "runs_on": []
            } # Minimal Structure
        }
        
        repo.save_graph(data)
        
        # Retrieve - GraphData object
        result = repo.get_graph_data(component_types=["Application"])
        assert len(result.components) == 1
        assert result.components[0].id == "a1"
        assert result.components[0].component_type == "Application"
        
        # Stats
        stats = repo.get_statistics()
        assert stats["application_count"] == 1

