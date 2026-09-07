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

import subprocess
from typing import Any, Dict, Optional


def _git(*args: str) -> Optional[str]:
    """Run a git command, returning None outside a repository or without git."""
    try:
        out = subprocess.run(
            ["git", *args], capture_output=True, text=True, timeout=10, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


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
        "config": dict(config),
    }
