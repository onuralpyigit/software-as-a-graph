
import sys
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import importlib

# Ensure project root is in path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "bin"))

SIMULATE_GRAPH_PY = "simulate_graph.py"

def test_run_py_all_stages():
    """Test run.py with --all flag to ensure all sub-scripts are called with correct arguments."""
    with patch('subprocess.run') as mock_run, \
         patch.object(sys, 'argv', ['run.py', '--all', '--scale', 'tiny', '--clean']), \
         patch('sys.executable', 'python'):
        
        import run
        importlib.reload(run)
        
        mock_run.return_value.returncode = 0
        ret = run.main()
        
        assert ret == 0
        
        # Collect all commands executed
        commands = [call.args[0] for call in mock_run.call_args_list]
        
        # Verify Generate
        gen_cmd = next((cmd for cmd in commands if 'generate_graph.py' in cmd[1]), None)
        assert gen_cmd is not None
        assert '--scale' in gen_cmd
        assert 'tiny' in gen_cmd
        assert '--output' in gen_cmd
        
        # Verify Import
        imp_cmd = next((cmd for cmd in commands if 'import_graph.py' in cmd[1]), None)
        assert imp_cmd is not None
        assert '--clear' in imp_cmd # from --clean
        
        # Verify Analyze
        # Expect multiple calls or one depending on implementation. 
        # Current implementation iterates over layers: app, infra, mw
        an_cmds = [cmd for cmd in commands if 'analyze_graph.py' in cmd[1]]
        assert len(an_cmds) >= 1
        # Check layers are covered
        layers_covered = []
        for cmd in an_cmds:
            if '--layer' in cmd:
                idx = cmd.index('--layer')
                layers_covered.append(cmd[idx+1])
        
        for layer in ['app', 'infra', 'mw']:
            assert layer in layers_covered

        # Verify Simulate
        sim_cmd = next((cmd for cmd in commands if 'simulate_graph.py' in cmd[1]), None)
        assert sim_cmd is not None
        assert 'report' in sim_cmd
        
        # Verify Validate
        val_cmd = next((cmd for cmd in commands if 'validate_graph.py' in cmd[1]), None)
        assert val_cmd is not None
        
        # Verify Visualize
        viz_cmd = next((cmd for cmd in commands if 'visualize_graph.py' in cmd[1]), None)
        assert viz_cmd is not None

def test_run_py_specific_layers():
    """Test run.py with specific layers."""
    with patch('subprocess.run') as mock_run, \
         patch.object(sys, 'argv', ['run.py', '--analyze', '--layer', 'app,infra']), \
         patch('sys.executable', 'python'):
        
        import run
        importlib.reload(run)
        
        mock_run.return_value.returncode = 0
        ret = run.main()
        
        assert ret == 0
        
        commands = [call.args[0] for call in mock_run.call_args_list]
        an_cmds = [cmd for cmd in commands if 'analyze_graph.py' in str(cmd)]
        
        # Should be called for app and infra
        layers_covered = []
        for cmd in an_cmds:
            if '--layer' in cmd:
                idx = cmd.index('--layer')
                layers_covered.append(cmd[idx+1])
        
        assert 'app' in layers_covered
        assert 'infra' in layers_covered
        assert 'mw' not in layers_covered
