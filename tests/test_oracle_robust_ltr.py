"""
tests/test_oracle_robust_ltr.py — the Amendment 11 harness
==========================================================

PREREGISTRATION.md Amendment 11 fixes, before any run, what these tests hold
``reproduce/oracle_robust_ltr.py`` to:

  * exactly the registered S (9) and Q (9) feature columns, rank-normalised within
    scenario, with the published training-free scores as the S/Q columns they name;
  * the four arms are the registered 2 x 2 of feature set x training oracle;
  * I_comp is never a training label (FailureSimulator is the Validate-stage oracle);
  * worst-case rho is the minimum over the three oracles.

Labels here are synthetic, so no simulator runs and no registered number is seen.
"""

from __future__ import annotations

import numpy as np
import pytest

from reproduce import oracle_robust_ltr as m
from reproduce.training_free_suite import _flow, indeg

FOLDS = ["atm_system", "healthcare_system"]


def test_registered_feature_sets():
    assert len(m.S_NAMES) == 9 and len(m.Q_NAMES) == 9
    assert not set(m.S_NAMES) & set(m.Q_NAMES)
    assert m.ARMS == {
        "gbm_dep": (("S",), "i_star"),
        "gbm_dep_qos": (("S", "Q"), "i_star"),
        "gbm_dep_dyn": (("S",), "i_dyn"),
        "gbm_dep_qos_dyn": (("S", "Q"), "i_dyn"),
    }


@pytest.mark.parametrize("name", FOLDS + ["realworld_edgex"])
def test_features_are_finite_and_match_published_rankers(name):
    topo = m._topology(name)
    apps, raw = m.raw_features(topo)
    assert apps == m.app_ids(topo)
    for key, names in (("S", m.S_NAMES), ("Q", m.Q_NAMES)):
        assert raw[key].shape == (len(apps), len(names))
        assert np.isfinite(raw[key]).all()
    dep = indeg(_flow(topo))
    assert raw["S"][:, m.S_NAMES.index("InDeg")].tolist() == [dep[a] for a in apps]
    shares = raw["Q"][:, [m.Q_NAMES.index(c) for c in
                          ("ReliableShare", "DurableShare", "DeadlineShare")]]
    assert ((shares >= 0) & (shares <= 1)).all()

    _, ranked = m.features(topo)
    for mat in ranked.values():
        assert ((mat > 0) & (mat <= 1)).all()


def test_icomp_is_never_a_training_label():
    with pytest.raises(ValueError, match="Validate-stage"):
        m.fit_predict(FOLDS[:1], FOLDS[1], ("S",), "i_comp", {}, 42, {})


def test_fit_predict_scores_every_holdout_application():
    # Synthetic labels (InDeg itself): exercises the fit path without any oracle.
    fake = {"i_star": {n: indeg(_flow(m._topology(n))) for n in FOLDS}}
    pred = m.fit_predict(FOLDS[:1], FOLDS[1], ("S", "Q"), "i_star", fake, 42, {})
    assert set(pred) == set(m.app_ids(m._topology(FOLDS[1])))


def test_worst_case_is_min_over_oracles():
    row = {"arms": {"x": {"i_star": {"rho": 0.8}, "i_dyn": {"rho": 0.3},
                          "i_comp": {"rho": 0.6}}},
           "comparators": {}}
    assert m.worst_case(row, "x") == 0.3


def test_contrasts_and_decisions_run_on_synthetic_folds():
    rng = np.random.default_rng(0)

    def block():
        return {o: {"rho": float(rng.uniform(0.2, 0.9))} for o in m.ORACLES}

    per_fold = {f: {"arms": {a: block() for a in m.ARMS},
                    "comparators": {r: block() for r in m.COMPARATORS}}
                for f in m.FOLDS}
    fam = m.contrasts(per_fold)
    assert {k for f in fam.values() for k in f} == {"A1", "A2", "B1", "B2", "C1", "C2"}
    assert all("p_holm" in c for f in fam.values() for c in f.values())
    zs = {"arm_means": {a: {"i_star": 0.5} for a in m.ARMS}, "reach_i_star": 0.938}
    dec = m.decisions(fam, zs)
    assert dec["A"][0] == "A" and dec["B"] in ("B", "B′") and dec["C"]
    assert dec["Z"]["applies"] is True
