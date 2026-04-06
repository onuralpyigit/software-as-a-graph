import unittest
from unittest.mock import MagicMock, patch
from src.infrastructure.neo4j_repo import Neo4jRepository

class TestNeo4jImportExtended(unittest.TestCase):
    def setUp(self):
        self.mock_driver = MagicMock()
        self.repo = Neo4jRepository(uri="bolt://localhost:7687", user="neo4j", password="password")
        self.repo.driver = self.mock_driver
        self.repo.database = "neo4j"

    def test_import_nodes_extended(self):
        data = {
            "nodes": [
                {
                    "id": "node1",
                    "name": "Compute Node 1",
                    "ip_address": "192.168.1.1",
                    "cpu_cores": 8,
                    "memory_gb": 16,
                    "os_type": "ubuntu"
                }
            ]
        }
        
        # Mock session.run for _import_batch
        mock_session = self.mock_driver.session.return_value.__enter__.return_value
        
        self.repo._import_entities(data)
        
        # Check if _import_batch was called for nodes with correct data
        calls = mock_session.run.call_args_list
        found_nodes_call = False
        for call in calls:
            query, params = call.args
            if "MERGE (n:Node {id: row.id})" in query:
                found_nodes_call = True
                rows = params["rows"]
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["ip_address"], "192.168.1.1")
                self.assertEqual(rows[0]["cpu_cores"], 8)
                self.assertIn("n.ip_address = row.ip_address", query)
        
        self.assertTrue(found_nodes_call)

    def test_import_apps_quality_metrics(self):
        data = {
            "applications": [
                {
                    "id": "app1",
                    "name": "Service A",
                    "code_metrics": {
                        "quality": {
                            "sqale_debt_ratio": 0.05,
                            "bugs": 2,
                            "vulnerabilities": 1,
                            "duplicated_lines_density": 0.1
                        },
                        "size": {"total_loc": 1000},
                        "complexity": {"avg_wmc": 5.5},
                        "cohesion": {"avg_lcom": 0.2},
                        "coupling": {"avg_fanin": 3, "avg_fanout": 4}
                    }
                }
            ]
        }
        
        mock_session = self.mock_driver.session.return_value.__enter__.return_value
        self.repo._import_entities(data)
        
        calls = mock_session.run.call_args_list
        found_apps_call = False
        for call in calls:
            query, params = call.args
            if "MERGE (a:Application {id: row.id})" in query:
                found_apps_call = True
                rows = params["rows"]
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["sqale_debt_ratio"], 0.05)
                self.assertEqual(rows[0]["bugs"], 2)
                self.assertEqual(rows[0]["loc"], 1000)
                self.assertIn("a.sqale_debt_ratio = row.sqale_debt_ratio", query)
        
        self.assertTrue(found_apps_call)

if __name__ == "__main__":
    unittest.main()
