"""
test_discover_scenarios_min.py
────────────────────────────────
Pins `discover_scenarios(min_scenarios=...)`: LOSO needs >= 2 scenarios (no
"other" scenario to train on with fewer), but k-fold trains and tests within
one scenario's own graph and has no such requirement. cli/kfold_evaluate.py
passes min_scenarios=1 for exactly this reason — an ATM-only run must not
hit the LOSO-shaped "need >= 2 scenarios" guard.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from cli.loso_evaluate import ScenarioBundle, discover_scenarios


def _fake_bundle(scenario_id: str) -> ScenarioBundle:
    return ScenarioBundle(
        scenario_id=scenario_id, graph=None, structural={}, rm={}, simulation={},
        hetero_data=None, n_nodes=10, n_edges=10, n_labelled=5,
    )


def test_default_requires_at_least_two_scenarios(tmp_path):
    (tmp_path / "atm_system").mkdir()
    with patch("cli.loso_evaluate.load_scenario_bundle", return_value=_fake_bundle("atm_system")):
        with pytest.raises(ValueError, match="Need >= 2"):
            discover_scenarios(tmp_path, skip=[])


def test_min_scenarios_one_allows_single_scenario(tmp_path):
    (tmp_path / "atm_system").mkdir()
    with patch("cli.loso_evaluate.load_scenario_bundle", return_value=_fake_bundle("atm_system")):
        bundles = discover_scenarios(tmp_path, skip=[], min_scenarios=1)
    assert len(bundles) == 1
    assert bundles[0].scenario_id == "atm_system"


def test_min_scenarios_one_still_errors_on_zero_usable(tmp_path):
    (tmp_path / "atm_system").mkdir()
    with patch("cli.loso_evaluate.load_scenario_bundle", return_value=None):
        with pytest.raises(ValueError, match="Need >= 1"):
            discover_scenarios(tmp_path, skip=[], min_scenarios=1)
