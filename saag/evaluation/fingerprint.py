"""
saag/evaluation/fingerprint.py — content fingerprints for resumable sweeps
==========================================================================

``reproduce/loso_all_variants.py --resume`` decides whether a previous run's
output can be reused. It used to decide on mtimes alone, and its own docstring
admits the hole: "Model code changes far more often than the cache does, and a
code change leaves no mtime on any input this can check." The mitigation was a
two-day age bound, which is a guess, not a check — and it is the wrong
granularity besides, since a variant is only reusable as a whole.

This module closes that by fingerprinting what a fit actually depends on:

  1. the hyperparameters and protocol flags it ran under,
  2. the *contents* of the cache artefacts feeding it, and
  3. the *contents* of the modules that define the model, the trainer, the
     feature construction and the harness itself.

A stored fit is reusable iff its fingerprint matches. A code edit, a
regenerated cache, or a changed flag all change the hash, so the stale-state
hazard that once published LOSO rho = -0.576 from a leftover checkpoint becomes
a mismatch rather than a silent reuse.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

#: Repository root, derived from this file's location (saag/evaluation/...).
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: Modules whose contents change what a fit produces. Deliberately explicit:
#: hashing the whole tree would invalidate every shard on a docs typo, and
#: hashing nothing is the hole this module exists to close.
CODE_FILES = (
    "saag/prediction/trainer.py",
    "saag/prediction/gnn_service.py",
    "saag/prediction/data_preparation.py",
    "saag/prediction/models/core.py",
    "saag/prediction/models/baselines.py",
    "saag/evaluation/metrics.py",
    "saag/evaluation/variant_registry.py",
    "cli/loso_evaluate.py",
)


def _hash_file(path: Path, h: "hashlib._Hash") -> None:
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)


@lru_cache(maxsize=1)
def code_digest() -> str:
    """SHA-256 over the contents of every file in :data:`CODE_FILES`.

    Cached: the files cannot change inside one process, and the hash is
    recomputed once per worker rather than once per fit.
    """
    h = hashlib.sha256()
    for rel in CODE_FILES:
        p = _REPO_ROOT / rel
        h.update(rel.encode())
        if p.exists():
            _hash_file(p, h)
        else:                       # absent is a state worth distinguishing
            h.update(b"\0missing")
    return h.hexdigest()


@lru_cache(maxsize=8)
def cache_digest(cache_dir: str) -> str:
    """SHA-256 over the contents of every JSON artefact under ``cache_dir``.

    Content, not mtime: ``make cache`` rewrites files that may be byte-identical,
    and a mtime-only check re-runs a 3-hour variant for no reason. The corpus is
    ~17 MB, so this costs well under a second.
    """
    h = hashlib.sha256()
    root = Path(cache_dir)
    for p in sorted(root.rglob("*.json")):
        h.update(str(p.relative_to(root)).encode())
        _hash_file(p, h)
    return h.hexdigest()


def fit_fingerprint(
    config: Dict[str, Any],
    cache_dir: Optional[str] = None,
    extra: Optional[Iterable[str]] = None,
) -> str:
    """Fingerprint one unit of work (a variant/fold/seed fit).

    ``config`` is any JSON-serialisable mapping of the parameters the fit ran
    under; keys are sorted, so callers need not.
    """
    h = hashlib.sha256()
    h.update(json.dumps(config, sort_keys=True, default=str).encode())
    h.update(code_digest().encode())
    if cache_dir is not None:
        h.update(cache_digest(str(cache_dir)).encode())
    for item in extra or ():
        h.update(str(item).encode())
    return h.hexdigest()
