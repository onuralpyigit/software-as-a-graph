#!/usr/bin/env python3
"""
scripts/check_doc_links.py — do the Markdown cross-references still resolve?
===========================================================================

The documentation set is heavily cross-linked and the links are the only thing
tying a formula to the section that derives it. They rot silently: renumbering
one document's headings (``structural-analysis.md`` went from §9.x to §7.x and
§10 to §8) broke forty-one anchors across twelve files, and nothing noticed,
because a dead ``#anchor`` renders as a working link that lands at the top of
the page.

Checks two things for every ``[text](target)`` in every tracked ``.md``:

* the file target exists, and
* the ``#fragment``, if any, matches a heading in that file.

Fragments are resolved with GitHub's slug rules (lowercase; drop everything but
word characters, spaces and hyphens; spaces to hyphens; de-duplicate with a
numeric suffix), so ``$Q^*(v)$`` and ``§4.1 Phase 1: Entity Modeling`` slug the
way they do on GitHub rather than the way they read.

Usage
-----
    python scripts/check_doc_links.py              # whole repo, tracked .md only
    python scripts/check_doc_links.py docs/        # one subtree
    python scripts/check_doc_links.py --suggest    # print the nearest heading
"""
from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

#: ``[text](target)`` where target is not a URL and not an image.
LINK = re.compile(r"(?<!\!)\[(?:[^\]\[]|\[[^\]]*\])*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
#: ATX headings only; the docs use no Setext headings.
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$", re.M)
FENCE = re.compile(r"^\s*(```|~~~)")


def strip_code_fences(text: str) -> str:
    """Blank out fenced blocks so their contents are neither links nor headings."""
    out, in_fence, marker = [], False, ""
    for line in text.splitlines():
        m = FENCE.match(line)
        if m and not in_fence:
            in_fence, marker = True, m.group(1)
            out.append("")
            continue
        if in_fence:
            out.append("")
            if line.strip().startswith(marker):
                in_fence = False
            continue
        out.append(line)
    return "\n".join(out)


def slug(heading: str) -> str:
    """GitHub's heading slug, closely enough for this repo's headings."""
    s = heading.strip()
    s = re.sub(r"`([^`]*)`", r"\1", s)                  # inline code keeps its text
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)      # links keep their text
    # Only ``*`` and ``~`` are stripped as emphasis. ``_`` is NOT: GitHub keeps
    # underscores in slugs, so dropping them here turns every anchor naming a
    # metric key or detector (``ap_c_directed``, ``SYSTEMIC_RISK``) into a false
    # positive -- which is most of this repo's anchors.
    s = re.sub(r"[*~]", "", s)
    s = s.lower()
    s = re.sub(r"[^\w\s-]", "", s, flags=re.UNICODE)     # drop punctuation, $, §, ...
    s = s.replace(" ", "-")
    return s


def anchors_of(text: str) -> List[str]:
    seen: Dict[str, int] = defaultdict(int)
    out = []
    for _, title in HEADING.findall(strip_code_fences(text)):
        base = slug(title)
        n = seen[base]
        seen[base] += 1
        out.append(base if n == 0 else f"{base}-{n}")
    return out


def tracked_markdown(paths: List[str]) -> List[Path]:
    try:
        listed = subprocess.run(["git", "-C", str(ROOT), "ls-files", "*.md"],
                                capture_output=True, text=True, check=True).stdout.split()
        files = [ROOT / f for f in listed]
    except (OSError, subprocess.SubprocessError):
        files = [p for p in ROOT.rglob("*.md") if ".git" not in p.parts]
    if paths:
        roots = [(ROOT / p).resolve() for p in paths]
        files = [f for f in files
                 if any(f.resolve() == r or r in f.resolve().parents for r in roots)]
    return sorted(files)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="limit to these files or directories")
    ap.add_argument("--suggest", action="store_true",
                    help="for a broken fragment, print the closest real heading")
    args = ap.parse_args()

    files = tracked_markdown(args.paths)
    cache: Dict[Path, List[str]] = {}

    def anchors_for(p: Path) -> Optional[List[str]]:
        if p not in cache:
            try:
                cache[p] = anchors_of(p.read_text(encoding="utf8"))
            except (OSError, UnicodeDecodeError):
                return None
        return cache[p]

    bad_files: List[Tuple[Path, int, str]] = []
    bad_anchors: List[Tuple[Path, int, str, str]] = []
    n_links = 0

    for f in files:
        try:
            text = strip_code_fences(f.read_text(encoding="utf8"))
        except (OSError, UnicodeDecodeError):
            continue
        offsets = [0]
        for line in text.splitlines(keepends=True):
            offsets.append(offsets[-1] + len(line))

        def line_of(pos: int) -> int:
            lo, hi = 0, len(offsets) - 1
            while lo < hi - 1:
                mid = (lo + hi) // 2
                if offsets[mid] <= pos:
                    lo = mid
                else:
                    hi = mid
            return lo + 1

        for m in LINK.finditer(text):
            target = m.group(1)
            if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("//"):
                continue  # external scheme
            n_links += 1
            path_part, _, frag = target.partition("#")
            dest = f if not path_part else (f.parent / path_part)
            if path_part:
                if not dest.exists():
                    bad_files.append((f, line_of(m.start()), target))
                    continue
                if dest.is_dir() or dest.suffix.lower() != ".md":
                    continue  # directory link or non-markdown file: existence is enough
            if not frag:
                continue
            have = anchors_for(dest)
            if have is None:
                continue
            if frag.lower() not in have:
                hint = ""
                if args.suggest:
                    close = difflib.get_close_matches(frag.lower(), have, n=1, cutoff=0.5)
                    hint = f"  -> #{close[0]}" if close else "  -> (no close heading)"
                bad_anchors.append((f, line_of(m.start()), target, hint))

    print(f"\n  Checked {n_links} relative link(s) across {len(files)} markdown file(s).\n")

    if bad_files:
        print("  BROKEN FILE TARGETS:")
        for f, ln, t in bad_files:
            print(f"    ! {f.relative_to(ROOT)}:{ln}  {t}")
        print()

    if bad_anchors:
        print("  BROKEN ANCHORS:")
        for f, ln, t, hint in bad_anchors:
            print(f"    ! {f.relative_to(ROOT)}:{ln}  {t}{hint}")
        print()

    ok = not bad_files and not bad_anchors
    print("  OK — every relative link and anchor resolves.\n" if ok
          else f"  {len(bad_files)} broken file target(s), {len(bad_anchors)} broken anchor(s).\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
