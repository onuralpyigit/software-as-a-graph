"""reproduce/omnibus_holm.py pools exactly the registered contrasts."""

import json

import pytest

from reproduce.omnibus_holm import REGISTERED_FAMILY, collect, omnibus


def _row(p, label="X", baseline="Y"):
    return {"label": label, "baseline_label": baseline, "mean_delta": 0.1, "p": p, "p_holm": p}


@pytest.fixture
def results_dir(tmp_path):
    blocks = {
        ("loso_significance_v5.json", "preregistered"): [_row(0.15), _row(0.47)],
        ("loso_significance_rq2_matched.json", "factorial"): [
            {"quantity": q, "label": q, "mean_delta": 0.0, "p": p, "p_holm": p}
            for q, p in (("typing", 0.47), ("qos", 0.042), ("interaction", 0.73))],
        ("loso_significance_rq2_matched.json", "rq2_controls"): [_row(0.38), _row(0.57)],
        # Amendment 2's directionality control, run later in its own sweep.
        ("loso_significance_directionality_cpu.json", "rq2_controls"): [_row(0.91)],
        ("loso_significance_hybrid_cpu.json", "hybrid"): [_row(0.0034), _row(0.73)],
        ("loso_significance_hybrid_gat_cpu.json", "hybrid_gat"): [_row(0.0015), _row(0.30)],
    }
    files = {}
    for (artifact, block), rows in blocks.items():
        files.setdefault(artifact, {"exploratory": [_row(1e-6)]})[block] = rows
    for artifact, data in files.items():
        (tmp_path / artifact).write_text(json.dumps(data))
    return tmp_path


def test_family_is_the_registered_blocks_only(results_dir):
    family = collect(results_dir)
    assert len(family) == 12
    assert {r["registration"] for r in family} == {"plan", "amendment_2", "amendment_5", "amendment_6"}
    # Exploratory rows (planted with a tiny p) never enter the family.
    assert min(r["p"] for r in family) == 0.0015
    assert {b for _, _, b in REGISTERED_FAMILY} >= {"hybrid", "hybrid_gat"}


def test_omnibus_holm_matches_hand_computation(results_dir):
    rows = {r["p"]: r["p_holm_omnibus"] for r in omnibus(collect(results_dir))}
    assert rows[0.0015] == pytest.approx(12 * 0.0015)
    assert rows[0.0034] == pytest.approx(11 * 0.0034)
    assert rows[0.042] == pytest.approx(10 * 0.042)
    # Monotone and capped at 1.
    assert rows[0.73] == 1.0


def test_family_holm_spans_artifacts(results_dir):
    """A family split across sweeps is corrected as one; a single-artifact family is untouched."""
    family = {r["p"]: r["p_holm_family"] for r in collect(results_dir)}
    # Amendment 2's controls: 0.38 / 0.57 in one artifact, 0.91 in another -> Holm over 3.
    assert family[0.38] == pytest.approx(min(1.0, 3 * 0.38))
    assert family[0.91] == pytest.approx(max(min(1.0, 3 * 0.38), 0.91))
    # The plan's two contrasts live in one artifact: Holm over 2, as that artifact had it.
    assert family[0.15] == pytest.approx(2 * 0.15)
