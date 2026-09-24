#!/usr/bin/env python3
"""
reproduce/model_agreement.py — inter-modeler agreement for system models
=======================================================================

One author wrote each of the five open-source system models
(``data/scenarios/realworld_*.json``) from the public documentation. This
script measures how closely an independent second model of the same system
agrees with the original. The manuscript treats the missing check as a
construct-validity threat (Section 8.3).

Re-modeling protocol
--------------------
1. The second modeler receives the list of public sources the original cites
   for the system (``metadata.description`` and the system's supplementary row)
   and the schema of ``data/scenarios/realworld_edgex.json``. They do **not**
   receive the original model.
2. They write a complete model in that schema: applications, libraries,
   topics (with QoS), brokers, nodes, and all six relationship kinds.
3. Before comparing, entity names are reconciled only through an alias file
   (``{"their-name": "original-name"}``) that lists each rename with a
   justification from the documentation. No other edits are allowed.
   Entities are matched on normalised name, so IDs never need to agree.
4. Report the output of this script as it is, including the per-type rows.

Agreement is the Jaccard index |A ∩ B| / |A ∪ B|. For entities it is taken per
type over matched names. For edges it is taken per relationship kind over
(source name, target name) pairs. Edge weights and QoS values are not compared
here: structure comes first, and the parameter values are a separate question.

Usage:
    PYTHONPATH=. python reproduce/model_agreement.py \\
        data/scenarios/realworld_edgex.json path/to/edgex_second_modeler.json \\
        [--alias aliases.json] [--output results/model_agreement_edgex.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

ENTITY_TYPES = ("applications", "libraries", "topics", "brokers", "nodes")


def normalise(name: str) -> str:
    """Case- and punctuation-insensitive key: ``Device-Modbus_Service`` -> ``devicemodbusservice``."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _jaccard(a: Set[Any], b: Set[Any]) -> Optional[float]:
    union = a | b
    return len(a & b) / len(union) if union else None


def _names(model: Dict[str, Any]) -> Dict[str, Tuple[str, str]]:
    """id -> (entity type, normalised name)."""
    return {e["id"]: (etype, normalise(e["name"]))
            for etype in ENTITY_TYPES for e in model.get(etype, [])}


def agreement(original: Dict[str, Any], second: Dict[str, Any],
              alias: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Per-type entity and per-kind edge Jaccard between two models of one system."""
    alias = {normalise(k): v for k, v in (alias or {}).items()}
    # Aliases are applied to the second model only, so an alias key is
    # matched against the second modeler's own spelling of a name.
    names_a = _names(original)
    names_b = {i: (t, normalise(alias.get(n, n))) for i, (t, n) in _names(second).items()}

    entities = {}
    for etype in ENTITY_TYPES:
        a = {n for t, n in names_a.values() if t == etype}
        b = {n for t, n in names_b.values() if t == etype}
        entities[etype] = {"original": len(a), "second": len(b),
                           "shared": len(a & b), "jaccard": _jaccard(a, b)}

    kinds = sorted(set(original.get("relationships", {})) | set(second.get("relationships", {})))
    edges = {}
    for kind in kinds:
        a = {(names_a[r["from"]][1], names_a[r["to"]][1])
             for r in original.get("relationships", {}).get(kind, [])}
        b = {(names_b[r["from"]][1], names_b[r["to"]][1])
             for r in second.get("relationships", {}).get(kind, [])}
        edges[kind] = {"original": len(a), "second": len(b),
                       "shared": len(a & b), "jaccard": _jaccard(a, b)}

    def pooled(rows: Dict[str, Dict[str, Any]]) -> Optional[float]:
        union = sum(r["original"] + r["second"] - r["shared"] for r in rows.values())
        return sum(r["shared"] for r in rows.values()) / union if union else None

    return {"entities": entities, "edges": edges,
            "entity_jaccard": pooled(entities), "edge_jaccard": pooled(edges)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("original", type=Path)
    ap.add_argument("second", type=Path)
    ap.add_argument("--alias", type=Path, help="JSON map {second-modeler name: original name}")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()

    load = lambda p: json.loads(p.read_text(encoding="utf8"))  # noqa: E731
    result = agreement(load(args.original), load(args.second),
                       load(args.alias) if args.alias else None)
    result["inputs"] = {"original": str(args.original), "second": str(args.second),
                        "alias": str(args.alias) if args.alias else None}

    fmt = lambda x: "  n/a" if x is None else f"{x:.3f}"  # noqa: E731
    for section in ("entities", "edges"):
        print(f"{section}:")
        for k, r in result[section].items():
            print(f"  {k:14s} orig={r['original']:4d} second={r['second']:4d} "
                  f"shared={r['shared']:4d} jaccard={fmt(r['jaccard'])}")
    print(f"pooled entity Jaccard {fmt(result['entity_jaccard'])}, "
          f"pooled edge Jaccard {fmt(result['edge_jaccard'])}")
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf8")
        print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
