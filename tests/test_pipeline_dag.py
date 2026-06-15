import pytest
from saag import Pipeline
from saag.infrastructure.memory_repo import MemoryRepository

def test_pipeline_fail_fast_no_checkpoint_no_simulate():
    """
    Verifies that calling predict() in a pipeline without a GNN checkpoint
    and without simulate() fails fast with RuntimeError.
    """
    repo = MemoryRepository()
    # Seed with some basic data
    repo.save_graph({
        "applications": [{"id": "AppA", "name": "App A"}],
        "relationships": {}
    })
    
    pipeline = Pipeline(repo=repo)
    pipeline.analyze().predict()  # predict requested, no simulate, no checkpoint
    
    with pytest.raises(RuntimeError) as exc_info:
        pipeline.run()
        
    assert "GNN Prediction requested but no trained GNN checkpoint was found" in str(exc_info.value)

def test_pipeline_reordered_simulate_before_predict(monkeypatch):
    """
    Verifies that Pipeline.run() executes Simulate before Predict.
    """
    repo = MemoryRepository()
    repo.save_graph({
        "applications": [{"id": "AppA", "name": "App A"}],
        "relationships": {}
    })
    
    execution_order = []
    
    # Monkeypatch client methods to track execution order
    monkeypatch.setattr(pipeline_client := Pipeline(repo=repo).client, "import_topology", lambda *args, **kwargs: execution_order.append("import"))
    monkeypatch.setattr(pipeline_client, "analyze", lambda *args, **kwargs: type('MockRes', (), {'raw': type('MockRaw', (), {'structural': type('MockStruct', (), {'graph': None})()})()})())
    
    # We mock predict to just track execution
    monkeypatch.setattr(pipeline_client, "predict", lambda *args, **kwargs: execution_order.append("predict"))
    # We mock simulate to track execution
    monkeypatch.setattr(pipeline_client, "simulate", lambda *args, **kwargs: execution_order.append("simulate"))
    
    pipeline = Pipeline(repo=repo)
    # Inject our mocked client
    pipeline.client = pipeline_client
    
    # Set analyze, predict, and simulate
    pipeline.analyze().predict().simulate()
    
    # Mock checkpoint check to succeed so we don't trigger the fail-fast guard
    monkeypatch.setattr("pathlib.Path.exists", lambda self: True)
    
    pipeline.run()
    
    # Ensure simulate ran before predict
    assert "simulate" in execution_order
    assert "predict" in execution_order
    assert execution_order.index("simulate") < execution_order.index("predict")

def test_gnn_independence_invariant_assertion():
    """
    Verifies that predict_from_data raises AssertionError if target label dimensions
    leak into features, violating the independence invariant.
    """
    from saag.prediction.gnn_service import GNNService
    from torch_geometric.data import HeteroData
    import torch
    
    # Initialize GNNService
    service = GNNService(hidden_channels=16, predict_edges=False)
    
    # Mock node model metadata
    class MockModel:
        node_types = ["Application"]
        edge_types = []
        def eval(self): pass
        
    service._node_model = MockModel()
    
    data = HeteroData()
    # Application node features: 5 dimensions (should trigger leak detection since App expects 23 dims)
    data["Application"].x = torch.zeros((10, 5))
    data["Application"].num_nodes = 10
    
    service._conversion_result = type('MockConv', (), {'node_id_map': {"Application": [f"App{i}" for i in range(10)]}})()
    
    with pytest.raises(AssertionError) as exc_info:
        service.predict_from_data(data)
        
    assert "Violation of Independence Guarantee" in str(exc_info.value)

