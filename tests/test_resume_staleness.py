"""
test_resume_staleness.py
─────────────────────────
``reproduce/loso_all_variants.py --resume`` used to reuse any ``results.json``
that existed, at any age, against any cache. `make table4` passes ``--resume``,
so a published table could be assembled from variants trained weeks apart
against different caches with nothing saying so — and the one artifact checked
this way, ``results/loso_all_variants.json``, turned out to reproduce from no
commit at all.

These pin the guard: a result is reusable only when it is newer than every cache
artefact feeding it, and only while it is recent enough that the model code is
unlikely to have moved underneath it.
"""

import os
import time

import pytest

from reproduce.loso_all_variants import _RESUME_MAX_AGE_DAYS, _staleness


@pytest.fixture
def cache_dir(tmp_path):
    d = tmp_path / "loso_cache" / "scenario_a"
    d.mkdir(parents=True)
    (d / "topology.json").write_text("{}")
    (d / "quality_scores.json").write_text("{}")
    return tmp_path / "loso_cache"


def _touch(path, seconds_ago: float):
    ts = time.time() - seconds_ago
    os.utime(path, (ts, ts))


def test_missing_result_is_stale(tmp_path, cache_dir):
    assert _staleness(tmp_path / "nope.json", cache_dir) == "no results.json"


def test_fresh_result_newer_than_cache_is_reusable(tmp_path, cache_dir):
    rp = tmp_path / "results.json"
    rp.write_text("{}")
    for f in cache_dir.rglob("*.json"):
        _touch(f, 600)
    _touch(rp, 60)
    assert _staleness(rp, cache_dir) is None


def test_result_older_than_a_cache_artefact_is_stale(tmp_path, cache_dir):
    """The failure that actually happened: a cache rebuilt under old results."""
    rp = tmp_path / "results.json"
    rp.write_text("{}")
    _touch(rp, 600)
    for f in cache_dir.rglob("*.json"):
        _touch(f, 60)
    reason = _staleness(rp, cache_dir)
    assert reason is not None and "is newer" in reason


def test_old_result_is_stale_even_when_cache_is_older_still(tmp_path, cache_dir):
    """A code change leaves no mtime on any input, so age alone has to bound reuse."""
    rp = tmp_path / "results.json"
    rp.write_text("{}")
    too_old = (_RESUME_MAX_AGE_DAYS + 1) * 86400
    _touch(rp, too_old)
    for f in cache_dir.rglob("*.json"):
        _touch(f, too_old + 3600)
    reason = _staleness(rp, cache_dir)
    assert reason is not None and "days old" in reason


def test_empty_cache_dir_falls_back_to_the_age_bound(tmp_path):
    empty = tmp_path / "empty_cache"
    empty.mkdir()
    rp = tmp_path / "results.json"
    rp.write_text("{}")
    _touch(rp, 60)
    assert _staleness(rp, empty) is None
    _touch(rp, (_RESUME_MAX_AGE_DAYS + 1) * 86400)
    assert "days old" in _staleness(rp, empty)
