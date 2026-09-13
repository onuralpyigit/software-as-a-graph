#!/usr/bin/env python3
"""Backwards-compatibility wrapper: delegates to reproduce/render_manuscript_md.py.

draft.md has been superseded by docs/research/jss/manuscript.md to mirror
manuscript.tex faithfully without drift.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reproduce.render_manuscript_md import main

if __name__ == "__main__":
    raise SystemExit(main())
