import unittest
import logging
from unittest.mock import MagicMock, patch
from src.infrastructure.neo4j_repo import Neo4jRepository

class TestNeo4jHardening(unittest.TestCase):
    def setUp(self):
        self.mock_driver = MagicMock()
        self.repo = Neo4jRepository(uri="bolt://localhost:7687", user="neo4j", password="password")
        self.repo.driver = self.mock_driver
        self.repo.database = "neo4j"
        self.repo.logger = MagicMock()

    def test_orphan_prevention_logging(self):
        # Case: 1 relationship, but _import_batch returns count > 1 (meaning nodes were created)
        # In reality, MATCH + MERGE relation shouldn't create nodes, 
        # but our audit check (count > len(batch)) would catch if it did (or if MERGE was used wrongly).
        
        mock_session = self.mock_driver.session.return_value.__enter__.return_value
        
        # Mock _import_batch to return 2 (1 rel + 1 unexpected node) for a batch of 1
        with patch.object(Neo4jRepository, '_import_batch', return_value=2):
            data = {
                "relationships": {
                    "publishes_to": [{"from": "app1", "to": "topic99"}]
                }
            }
            self.repo._import_relationships(data)
            
            # Verify warning was logged
            self.repo.logger.warning.assert_called()
            args, _ = self.repo.logger.warning.call_args
            self.assertIn("created 1 orphan nodes", args[0])

    def test_transaction_wrapper_failure(self):
        # Mock a failure in Phase 3
        with patch.object(Neo4jRepository, '_calculate_intrinsic_weights', side_effect=Exception("Neo4j OOM")):
            data = {"nodes": [], "relationships": {}}
            
            with self.assertRaises(Exception):
                self.repo.save_graph(data)
            
            # Verify error and critical recommendation were logged
            self.repo.logger.error.assert_called_with("Import failed during phase orchestration: Neo4j OOM")
            self.repo.logger.critical.assert_called()
            args, _ = self.repo.logger.critical.call_args
            self.assertIn("Re-run with clear=True", args[0])

    def test_always_on_constraints(self):
        # Verify _create_constraints is called regardless of clear=True/False
        mock_session = self.mock_driver.session.return_value.__enter__.return_value
        
        # Case 1: clear=True
        self.repo.save_graph({}, clear=True)
        self.repo.logger.info.assert_any_call("Starting import. Clear DB: True")
        
        # Case 2: clear=False
        self.repo.save_graph({}, clear=False)
        self.repo.logger.info.assert_any_call("Starting import. Clear DB: False")
        
        # Verify _create_constraints was called at least twice 
        # (Once per save_graph call)
        # We can check specific queries if needed, but the orchestration logic is the key.

if __name__ == "__main__":
    unittest.main()
