"""Non-graph learned baseline: gradient boosting on the typed node features.

This arm exists to answer a question none of the other baselines can. The LOSO
table contrasts typed graph learning against untyped graph learning and against
closed-form centrality, so it can say whether *typing* helps and whether
*learning* beats a heuristic -- but not whether **message passing** contributes
anything over a plain regressor reading the same numbers.

That question has teeth here because of what the features already contain.
Indices 0--17 of every node vector (Section 3.4 of the manuscript) are
betweenness, closeness, reverse PageRank, clustering, articulation score and
bridge load -- the graph structure, already summarised into scalars by the
deterministic analysis stage. And the target ``I*(v)`` is itself a reachability
functional over the same topology. A gradient-boosted tree over those columns
therefore has access to most of what a GNN could aggregate, without aggregating
anything. If it matches the GNNs, the graph-learning contribution is not the
message passing.

The features are not merely *similar* to the ones the GNNs read: they are the
same tensors, taken from ``networkx_to_hetero_data`` output, so the comparison
carries no feature-construction confound. Only the model class differs.

One model is fitted per node type, because the feature blocks have different
widths (19--25 dims) and different meanings per type; pooling them would require
padding that invents columns. Types absent from the holdout, or with too few
labelled training rows, are skipped and simply carry no prediction.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

#: Below this many labelled training rows a per-type fit is not attempted. Three
#: is the same floor ``saag.evaluation.metrics`` uses before it will report a
#: Spearman correlation at all.
_MIN_TRAIN_ROWS = 3


def _labelled_rows(store) -> np.ndarray:
    """Boolean mask over a node store: which rows carry simulation ground truth.

    Mirrors ``data_preparation._labelled_index_mask`` -- preferring the explicit
    ``label_mask`` and falling back to the ``|y| > 0`` proxy -- so the tabular
    arm trains on exactly the rows the GNNs train on. Using the proxy alone
    would silently drop nodes the simulator targeted and scored 0.0, which is a
    real observation and not a missing one.
    """
    mask = getattr(store, "label_mask", None)
    if mask is not None:
        return mask.detach().cpu().numpy().astype(bool)
    y = store.y.detach().cpu().numpy()
    return np.abs(y[:, 0]) > 1e-6


def _xy(store) -> Tuple[np.ndarray, np.ndarray]:
    """``(features, composite target)`` for the labelled rows of one node store."""
    x = store.x.detach().cpu().numpy()
    y = store.y.detach().cpu().numpy()[:, 0]
    keep = _labelled_rows(store)
    return x[keep], y[keep]


def _build_regressor(seed: int):
    """The estimator, with a scikit-learn availability fallback.

    Gradient boosting is the intended model. Ridge is the fallback so that a
    thin install still produces *a* non-graph learned arm rather than skipping
    the control entirely -- but it is a materially weaker comparator, so the
    substitution is logged rather than made silently.
    """
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:                                   # pragma: no cover
        from sklearn.linear_model import Ridge
        logger.warning(
            "GradientBoostingRegressor unavailable; falling back to Ridge. "
            "The tabular control is weaker than intended -- say so if reported."
        )
        return Ridge(alpha=1.0, random_state=seed)

    # Deliberately close to scikit-learn's defaults. This arm is a control, and
    # tuning it while the GNN arms run at fixed conventional hyperparameters
    # (Section 4.1) would make the comparison asymmetric in the tabular model's
    # favour -- the mirror image of the confound the RQ2 control arms exist to
    # remove.
    return GradientBoostingRegressor(random_state=seed)


def fit_predict_tabular(
    train_data: Iterable[Any],
    holdout_data: Any,
    holdout_id_map: Dict[str, List[str]],
    seed: int = 42,
    node_types: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Fit per node type on the training graphs, predict on the holdout.

    Args:
        train_data: ``HeteroData`` for each of the N-1 training scenarios.
        holdout_data: ``HeteroData`` for the held-out scenario.
        holdout_id_map: ``node_id_map`` from the holdout's conversion result,
            mapping node type to the node ids its rows correspond to, in order.
        seed: Passed to the estimator, so the arm varies across seeds the way
            the learned arms do rather than reporting a spurious zero variance.
        node_types: Restrict to these types. ``None`` means every type the
            holdout carries.

    Returns:
        ``{node_id: predicted_impact}``. Empty when nothing could be fitted.
    """
    train_list = list(train_data)
    wanted = (set(node_types) if node_types is not None
              else {nt for nt in holdout_data.node_types})

    preds: Dict[str, float] = {}
    for node_type in sorted(wanted):
        if node_type not in holdout_data.node_types:
            continue
        holdout_store = holdout_data[node_type]
        if not hasattr(holdout_store, "x") or holdout_store.x.numel() == 0:
            continue

        xs, ys = [], []
        for data in train_list:
            if node_type not in data.node_types:
                continue
            store = data[node_type]
            if not hasattr(store, "y") or store.y.numel() == 0:
                continue
            x, y = _xy(store)
            if len(y):
                xs.append(x)
                ys.append(y)

        if not xs:
            continue
        x_train = np.vstack(xs)
        y_train = np.concatenate(ys)
        if len(y_train) < _MIN_TRAIN_ROWS:
            logger.info("  tabular: %s has %d labelled training rows; skipping",
                        node_type, len(y_train))
            continue

        x_holdout = holdout_store.x.detach().cpu().numpy()
        if x_holdout.shape[1] != x_train.shape[1]:
            # A type whose feature block differs in width between corpora is a
            # schema mismatch, not something to paper over by truncating.
            logger.warning(
                "  tabular: %s feature width %d on holdout vs %d in training; "
                "skipping", node_type, x_holdout.shape[1], x_train.shape[1]
            )
            continue

        # A constant target carries no gradient and no ranking; fitting it
        # produces a constant prediction that scores as an undefined rho
        # downstream, which is more honest reported as "not fitted".
        if float(np.ptp(y_train)) < 1e-9:
            logger.info("  tabular: %s target is constant; skipping", node_type)
            continue

        model = _build_regressor(seed)
        model.fit(x_train, y_train)
        scores = model.predict(x_holdout)

        ids = holdout_id_map.get(node_type, [])
        if len(ids) != len(scores):
            logger.warning(
                "  tabular: %s id map has %d entries for %d rows; skipping",
                node_type, len(ids), len(scores)
            )
            continue
        for node_id, score in zip(ids, scores):
            preds[str(node_id)] = float(score)

    return preds
