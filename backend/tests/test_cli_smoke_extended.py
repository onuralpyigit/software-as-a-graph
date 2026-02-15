import sys
import pytest
import importlib
import json
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

# Ensure project root is in path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "bin"))

def test_generate_graph_cli():
    """Test generate_graph.py main function with mocks."""
    mock_data = {"nodes": [{"id": "n1"}]}
    
    with patch.object(sys, 'argv', ['generate_graph.py', '--scale', 'tiny', '--output', 'test_output.json']), \
         patch('src.generation.generate_graph', return_value=mock_data) as mock_gen, \
         patch('builtins.open', mock_open()) as m_open:
        
        # Clear module from sys.modules to ensure clean import
        if 'generate_graph' in sys.modules:
            del sys.modules['generate_graph']
        import generate_graph
        
        generate_graph.main()
        
        mock_gen.assert_called_once()
        m_open.assert_called()

def test_import_graph_cli():
    """Test import_graph.py main function with mocks."""
    mock_repo = MagicMock()
    mock_repo.get_statistics.return_value = {"node_count": 10}
    
    # Mock data to read from file
    mock_data = {"nodes": []}
    
    with patch.object(sys, 'argv', ['import_graph.py', '--input', 'test.json', '--clear']), \
         patch('src.core.create_repository', return_value=mock_repo) as MockCreateRepo, \
         patch('pathlib.Path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=json.dumps(mock_data))):
        
        if 'import_graph' in sys.modules:
            del sys.modules['import_graph']
        import import_graph
        
        import_graph.main()
        
        MockCreateRepo.assert_called_once()
        mock_repo.save_graph.assert_called_once_with(mock_data, clear=True)
        mock_repo.get_statistics.assert_called_once()
        mock_repo.close.assert_called_once()

def test_export_graph_cli():
    """Test export_graph.py main function with mocks."""
    mock_repo = MagicMock()
    
    mock_data = {"nodes": [], "relationships": {}}
    mock_repo.export_json.return_value = mock_data
    
    with patch.object(sys, 'argv', ['export_graph.py', '--output', 'exported.json']), \
         patch('src.core.create_repository', return_value=mock_repo) as MockCreateRepo, \
         patch('builtins.open', mock_open()) as m_open:
        
        if 'export_graph' in sys.modules:
            del sys.modules['export_graph']
        import export_graph
        
        export_graph.main()
        
        MockCreateRepo.assert_called_once()
        mock_repo.export_json.assert_called_once()
        m_open.assert_called()
        mock_repo.close.assert_called_once()
