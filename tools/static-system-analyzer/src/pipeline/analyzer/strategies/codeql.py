"""CodeQL analysis strategy — call-graph based topic extraction."""

from __future__ import annotations

import csv
import json
import re
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .base import AnalysisStrategy
from ..models import TopicEntry
from ..file_finder import find_makefile_with_Makefile
from common.logger import log_info, log_warning, log_error, log_debug
from common.runtime_config import CodeQLRuntimeConfig, get_runtime_config


# ── QL query template ────────────────────────────────────────────────────────
# Placeholders are filled from runtime.yaml at execution time.
_QL_TEMPLATE = r"""
/**
 * @name TopicWriteReadReachability
 * @description Find write/read calls reachable from the entry-point method,
 *              extract the topic type of the specified argument, and track
 *              whether the call passes through library code.
 * @kind problem
 * @id ssa/topic-reachability
 */

import java

// ── helpers ──────────────────────────────────────────────────────────────────

/** A callable that is reachable from the entry-point (transitively), including the entry-point itself. */
predicate reachable(Callable entry, Callable target) {
  entry = target
  or
  entry.polyCalls(target)
  or
  exists(Callable mid | entry.polyCalls(mid) and reachable(mid, target))
}

/** The single entry-point method per compilation unit (restricted to src/ directory). */
class EntryPoint extends Method {
  EntryPoint() {
    this.hasName("$$MAIN_METHOD$$") and
    this.getLocation().getFile().getRelativePath().matches("src/%")
  }
}

// ── sink predicates (generated) ──────────────────────────────────────────────
$$WRITE_PREDICATES$$
$$READ_PREDICATES$$

// ── main query ───────────────────────────────────────────────────────────────
from EntryPoint entry, MethodCall sink, string role, string argType
where
  (
    $$WRITE_WHERE_CLAUSES$$
  )
  or
  (
    $$READ_WHERE_CLAUSES$$
  )
select sink, role + "|" + argType + "|" + sink.getEnclosingCallable().getDeclaringType().getPackage().getName()
"""


def _build_sink_predicate(idx: int, class_name: str, method_name: str,
                          arg_index: int, role: str) -> Tuple[str, str]:
    """Return (predicate_def, where_clause) for one write/read method."""
    pred_name = f"is{role.capitalize()}Sink_{idx}"
    pred = (
        f"predicate {pred_name}(MethodCall mc, string argType) {{\n"
        f"  mc.getMethod().hasName(\"{method_name}\") and\n"
        f"  mc.getMethod().getDeclaringType().hasName(\"{class_name}\") and\n"
        f"  argType = mc.getArgument({arg_index}).getType().getName()\n"
        f"}}\n"
    )
    where = (
        f"    reachable(entry, sink.getEnclosingCallable()) and\n"
        f"    {pred_name}(sink, argType) and role = \"{role}\""
    )
    return pred, where


def _render_query(cfg: CodeQLRuntimeConfig) -> str:
    """Fill the QL template with values from runtime config."""
    write_preds: List[str] = []
    write_wheres: List[str] = []
    read_preds: List[str] = []
    read_wheres: List[str] = []

    for i, wm in enumerate(cfg.write_methods):
        p, w = _build_sink_predicate(i, wm.class_name, wm.method_name,
                                     wm.topic_arg_index, "pub")
        write_preds.append(p)
        write_wheres.append(w)

    for i, rm in enumerate(cfg.read_methods):
        p, w = _build_sink_predicate(i, rm.class_name, rm.method_name,
                                     rm.topic_arg_index, "sub")
        read_preds.append(p)
        read_wheres.append(w)

    query = _QL_TEMPLATE
    query = query.replace("$$MAIN_METHOD$$", cfg.main_method_name)
    query = query.replace("$$WRITE_PREDICATES$$", "\n".join(write_preds))
    query = query.replace("$$READ_PREDICATES$$", "\n".join(read_preds))
    query = query.replace("$$WRITE_WHERE_CLAUSES$$",
                          "\n    or\n".join(write_wheres) if write_wheres else "    none()")
    query = query.replace("$$READ_WHERE_CLAUSES$$",
                          "\n    or\n".join(read_wheres) if read_wheres else "    none()")
    return query


# ── CodeQL CLI wrappers ──────────────────────────────────────────────────────

_DB_CREATE_TIMEOUT = 600   # seconds
_QUERY_TIMEOUT = 300


def _run_codeql(cli_path: str, args: List[str], cwd: Optional[Path] = None,
                timeout: int = _QUERY_TIMEOUT,
                java_home: str = "") -> Tuple[bool, str]:
    """Execute a codeql CLI command and return (success, combined_output)."""
    cmd = [cli_path] + args
    log_debug(f"Running: {' '.join(cmd)}")
    env = None
    if java_home:
        env = {**os.environ, "JAVA_HOME": java_home}
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd,
            env=env,
        )
        output = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0:
            log_warning(f"CodeQL exit {result.returncode}: {output[:500]}")
            return False, output
        return True, output
    except subprocess.TimeoutExpired:
        log_error(f"CodeQL timed out after {timeout}s: {' '.join(cmd)}")
        return False, "timeout"
    except FileNotFoundError:
        log_error(f"CodeQL CLI not found: {cli_path}")
        return False, "not_found"


def _create_database(cli_path: str, source_root: Path,
                     db_path: Path, build_command: str = "",
                     java_home: str = "") -> bool:
    """Create a CodeQL Java database for the project."""
    args = [
        "database", "create", str(db_path),
        "--language=java-kotlin",
        "--source-root", str(source_root),
        "--overwrite",
    ]
    if build_command:
        args += ["--command", build_command]
    else:
        args.append("--build-mode=none")
    ok, _ = _run_codeql(cli_path, args, cwd=source_root, timeout=_DB_CREATE_TIMEOUT,
                        java_home=java_home)
    return ok


def _run_query(cli_path: str, db_path: Path,
               query_path: Path, bqrs_path: Path,
               java_home: str = "") -> bool:
    """Run a QL query against a database."""
    ok, _ = _run_codeql(cli_path, [
        "query", "run",
        "--database", str(db_path),
        "--output", str(bqrs_path),
        str(query_path),
    ], timeout=_QUERY_TIMEOUT, java_home=java_home)
    return ok


def _decode_results(cli_path: str, bqrs_path: Path,
                    csv_path: Path, java_home: str = "") -> bool:
    """Decode BQRS to CSV."""
    ok, _ = _run_codeql(cli_path, [
        "bqrs", "decode",
        "--format=csv",
        "--output", str(csv_path),
        str(bqrs_path),
    ], java_home=java_home)
    return ok


# ── result parsing ───────────────────────────────────────────────────────────

def _extract_topic_name(type_name: str, pattern: str) -> Optional[str]:
    """Apply the topic_name_pattern regex to extract the topic name."""
    m = re.search(pattern, type_name)
    if m and m.lastindex and m.lastindex >= 1:
        return m.group(1)
    return None


def _identify_lib_package(package_name: str,
                          dependency_suffixes: Tuple[str, ...]) -> Optional[str]:
    """If the package belongs to a library, return the lib name."""
    parts = package_name.split(".")
    for part in parts:
        for suffix in dependency_suffixes:
            if part.endswith(suffix):
                return part
    return None


def _parse_codeql_csv(csv_path: Path, folder_name: str,
                      topic_pattern: str,
                      dependency_suffixes: Tuple[str, ...]) -> List[TopicEntry]:
    """Parse the decoded CSV and produce TopicEntry objects.

    Each CSV row produced by the query has a message column formatted as:
        role|argTypeName|packageName
    """
    entries: List[TopicEntry] = []
    libs_used: Set[str] = set()

    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return entries

    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            # codeql bqrs decode CSV has header row; skip it
            msg = row[-1] if row else ""
            parts = msg.split("|")
            if len(parts) < 2:
                continue
            role = parts[0].strip()
            arg_type = parts[1].strip()
            pkg_name = parts[2].strip() if len(parts) > 2 else ""

            if role not in ("pub", "sub"):
                continue

            topic_name = _extract_topic_name(arg_type, topic_pattern)
            if not topic_name:
                log_debug(f"Skipping unmatched type: {arg_type}")
                continue

            # app-level pub/sub
            entries.append(TopicEntry(
                source_folder=folder_name,
                name=topic_name,
                role=role,
            ))

            # detect library traversal → uses
            lib_name = _identify_lib_package(pkg_name, dependency_suffixes)
            if lib_name and lib_name != folder_name:
                libs_used.add(lib_name)

    # add uses entries for libraries found in the call chain
    for lib in sorted(libs_used):
        entries.append(TopicEntry(
            source_folder=folder_name,
            name=lib,
            role="uses",
        ))

    return entries


# ── strategy ─────────────────────────────────────────────────────────────────

class CodeQLStrategy(AnalysisStrategy):
    """Extract pub/sub/uses via CodeQL call-graph analysis."""

    def extract(self, folder_path: Path, folder_name: str) -> List[TopicEntry]:
        cfg = get_runtime_config().analyzer.codeql
        dep_suffixes = get_runtime_config().analyzer.dependency_suffixes

        if not cfg.write_methods and not cfg.read_methods:
            log_warning("No write/read methods configured for CodeQL — skipping")
            return []

        local_tmp = Path("/tmp") / f"codeql_{folder_name}"
        os.makedirs(local_tmp, exist_ok=True)
        try:
            return self._run(folder_path, folder_name, cfg, dep_suffixes,
                             local_tmp)
        finally:
            shutil.rmtree(local_tmp, ignore_errors=True)

    def _run(self, folder_path: Path, folder_name: str,
             cfg: CodeQLRuntimeConfig,
             dep_suffixes: Tuple[str, ...],
             tmp_dir: Path) -> List[TopicEntry]:

        db_path = tmp_dir / "db"
        query_path = tmp_dir / "query.ql"
        bqrs_path = tmp_dir / "results.bqrs"
        csv_path = tmp_dir / "results.csv"

        # 1. Render query
        query_text = _render_query(cfg)
        query_path.write_text(query_text, encoding="utf-8")
        log_debug(f"Generated QL query:\n{query_text}")

        # qlpack.yml so CodeQL can resolve `import java`
        qlpack_path = tmp_dir / "qlpack.yml"
        qlpack_path.write_text(
            "name: ssa/topic-query\n"
            "version: 0.0.1\n"
            "dependencies:\n"
            "  codeql/java-all: \"*\"\n",
            encoding="utf-8",
        )

        # 2. Create database
        build_cmd = cfg.build_command
        if build_cmd:
            makefile_path = find_makefile_with_Makefile(folder_path)
            if makefile_path:
                makefile_dir = makefile_path.parent
                # Insert -C <dir> between the tool name and targets
                parts = build_cmd.split(maxsplit=1)
                tool = parts[0]
                targets = parts[1] if len(parts) > 1 else ""
                build_cmd = f"{tool} -C {makefile_dir} {targets}".strip()
                log_info(f"CodeQL build command: {build_cmd}")
            else:
                log_warning(f"No Makefile with include pattern found for {folder_name}, using build command as-is")

        log_info(f"Creating CodeQL database for {folder_name}...")
        if not _create_database(cfg.cli_path, folder_path, db_path, build_cmd,
                                java_home=cfg.java_home):
            log_error(f"CodeQL database creation failed for {folder_name}")
            return []

        # 3. Run query
        log_info(f"Running CodeQL query for {folder_name}...")
        if not _run_query(cfg.cli_path, db_path, query_path, bqrs_path,
                          java_home=cfg.java_home):
            log_error(f"CodeQL query failed for {folder_name}")
            return []

        # 4. Decode results
        if not _decode_results(cfg.cli_path, bqrs_path, csv_path,
                               java_home=cfg.java_home):
            log_error(f"CodeQL result decode failed for {folder_name}")
            return []

        # 5. Parse CSV
        entries = _parse_codeql_csv(
            csv_path, folder_name, cfg.topic_name_pattern, dep_suffixes,
        )
        log_info(f"CodeQL extracted {len(entries)} entries for {folder_name}")
        return entries
