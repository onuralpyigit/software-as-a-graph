import unittest
from unittest.mock import MagicMock
from src.infrastructure.neo4j_repo import Neo4jRepository

class TestNeo4jOrdering(unittest.TestCase):
    def setUp(self):
        self.mock_driver = MagicMock()
        self.repo = Neo4jRepository(uri="bolt://localhost:7687", user="neo4j", password="password")
        self.repo.driver = self.mock_driver
        self.repo.database = "neo4j"

    def test_save_graph_phase_ordering(self):
        # This test ensures that save_graph calls the steps in order.
        # Actually, since I'm mostly worried about Rule 6 using the updated Node weight,
        # I'll mock the queries.
        
        mock_session = self.mock_driver.session.return_value.__enter__.return_value
        
        data = {"nodes": [{"id": "n1"}], "brokers": [{"id": "b1"}, {"id": "b2"}]}
        self.repo.save_graph(data)
        
        # Capture all run calls
        calls = mock_session.run.call_args_list
        queries = [call.args[0] for call in calls]
        
        # Check that Node weight is set (Phase 5 Step 4) BEFORE broker_to_broker weight is set (Phase 5 Step 6)
        node_weight_idx = -1
        b2b_weight_idx = -1
        
        for i, q in enumerate(queries):
            if "MATCH (n:Node)" in q and "SET n.weight =" in q:
                node_weight_idx = i
            if "dependency_type: 'broker_to_broker'" in q and "SET d.weight = coalesce(node_w" in q:
                b2b_weight_idx = i
        
        self.assertNotEqual(node_weight_idx, -1, "Node weight calculation not found")
        self.assertNotEqual(b2b_weight_idx, -1, "Broker-to-broker weight propagation not found")
        self.assertTrue(node_weight_idx < b2b_weight_idx, "Node weight must be computed before B2B propagation")
        
        # Also check that _derive_dependencies (Phase 4) runs before Phase 5
        derive_idx = -1
        for i, q in enumerate(queries):
            if "Rule 6: broker_to_broker" in q:
                derive_idx = i
                break
        
        self.assertTrue(derive_idx < node_weight_idx, "Derivation (Rule 6) must happen before Phase 5")

if __name__ == "__main__":
    unittest.main()
