"""
reproduce/_provenance.py — what produced this artifact
======================================================

Every reproduce artifact should say which commit and which oracle configuration
produced it. This project has twice shipped a number whose provenance could not
be reconstructed afterwards: the LOSO table that "reproduces from no commit", and
a composite oracle silently missing one of its four criteria depending on which
script computed it. In both cases the artifact looked exactly like a correct one.

A stamp does not prevent either failure. It makes them visible in the artifact
rather than only in a re-run, which is the difference between a reviewer asking
a question and a reader trusting a stale figure.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

#: Committed corpus manifest, which carries a SHA-256 per scenario file.
_MANIFEST = Path(__file__).resolve().parent.parent / "data/scenarios/MANIFEST.json"


def _git(*args: str) -> Optional[str]:
    """Run a git command, returning None outside a repository or without git."""
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def corpus_digest() -> Optional[str]:
    """One fingerprint over the committed corpus, from the manifest's SHA-256s.

    Staleness was previously judged by comparing an artifact's mtime to the
    corpus files' mtimes. That is wrong in both directions: regenerating the
    corpus byte-identically (which CI asserts is possible, and which happens
    routinely) bumps every mtime and ages every artifact that is in fact still
    valid, while restoring an older corpus from a checkout leaves mtimes newer
    than the content they carry. Content answers the question mtime was standing
    in for: was this artifact computed against the corpus now on disk.

    Returns None when the manifest is missing or unreadable, which leaves the
    caller on the mtime fallback rather than silently asserting freshness.
    """
    try:
        datasets = json.loads(_MANIFEST.read_text())["datasets"]
    except (OSError, ValueError, KeyError):
        return None
    digests = sorted(
        f"{name}:{meta['sha256']}"
        for name, meta in datasets.items()
        if isinstance(meta, dict) and "sha256" in meta
    )
    if not digests:
        return None
    return hashlib.sha256("\n".join(digests).encode("utf-8")).hexdigest()


def stamp(**config: Any) -> Dict[str, Any]:
    """Provenance block for an artifact: commit state plus the run's own config.

    ``dirty`` matters as much as ``commit``: a figure produced from a modified
    working tree cannot be regenerated from the commit it names, and that is
    precisely the failure mode that cost this project a published table. Pass the
    oracle parameters the run actually used (seeds, priming, qos mode, depth
    limit) as keyword arguments — a commit alone does not pin a result whose
    script takes flags.
    """
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
        # Which corpus this artifact describes, by content rather than by
        # timestamp. reproduce/reconcile_manuscript.py compares it against the
        # corpus on disk; an artifact without one falls back to the weaker
        # mtime test.
        "corpus_digest": corpus_digest(),
        "config": dict(config),
    }
