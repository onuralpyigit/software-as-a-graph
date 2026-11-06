"""
Keep docs/failure-simulation.md and docs/simulation-extensions.md honest.

The pages' examples had drifted from the code: calls to APIs that did not
exist, CLI defaults that had changed, JSON snippets with invented fields. This
test makes each kind of drift a failure:

* every ```python block runs, in a scratch directory whose ``data/`` points at
  the repository's (outputs never land in the repo);
* every ``python cli/simulate_graph.py`` / ``saag-simulate`` line in a ```bash
  block parses with the real argparse parser, and its ``data/...`` inputs exist;
* every ```json block preceded by ``<!-- schema: NAME -->`` uses only keys that
  a real result of that artifact has.

A python block preceded by ``<!-- doc-test: skip (reason) -->`` is not run.
"""

import json
import logging
import re
import shlex
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "docs" / "failure-simulation.md", ROOT / "docs" / "simulation-extensions.md"]

_FENCE = re.compile(r"(?:^<!--\s*(?P<marker>[^>]*?)\s*-->\s*\n)?^```(?P<lang>\w+)\n(?P<body>.*?)^```",
                    re.M | re.S)


def _blocks(lang):
    out = []
    for doc in DOCS:
        text = doc.read_text()
        for m in _FENCE.finditer(text):
            if m.group("lang") == lang:
                line = text[: m.start("body")].count("\n")
                out.append(pytest.param(m.group("body"), m.group("marker") or "",
                                        id=f"{doc.stem}:line{line}"))
    return out


# ── python blocks ────────────────────────────────────────────────────────────

@pytest.fixture
def scratch(tmp_path, monkeypatch):
    (tmp_path / "data").symlink_to(ROOT / "data")
    (tmp_path / "output" / "simulation").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    logging.disable(logging.WARNING)
    yield tmp_path
    logging.disable(logging.NOTSET)


@pytest.mark.parametrize("code, marker", _blocks("python"))
def test_python_example_runs(code, marker, scratch, capsys):
    if marker.startswith("doc-test: skip"):
        pytest.skip(marker)
    if "MessageFlowSimulator" in code or "runtime_labels" in code:
        pytest.importorskip("simpy")
    exec(compile(code, "<simulation docs>", "exec"), {"__name__": "__doc_example__"})


# ── CLI lines ────────────────────────────────────────────────────────────────

def _commands():
    cmds = []
    for p in _blocks("bash"):
        body = p.values[0].replace("\\\n", " ")
        for line in body.splitlines():
            line = line.strip()
            for prefix in ("python cli/simulate_graph.py", "saag-simulate"):
                if line.startswith(prefix):
                    cmds.append(pytest.param(shlex.split(line[len(prefix):]), id=line[:60]))
    return cmds


@pytest.mark.parametrize("argv", _commands())
def test_cli_example_parses(argv):
    from cli.simulate_graph import _build_parser

    args = _build_parser().parse_args(argv)
    path = getattr(args, "input", None)
    if path and path.startswith("data/"):
        assert (ROOT / path).exists(), f"example input {path} does not exist"


def test_the_pages_have_examples():
    assert len(_blocks("python")) >= 8 and len(_commands()) >= 8


# ── JSON snippets ────────────────────────────────────────────────────────────

#: Containers whose keys are data (node ids, seeds, topics), not schema.
_DYNAMIC = {"records", "seed_impact_scores", "measured_utilization", "per_topic_i_dyn",
            "per_seed", "topic_stats", "subscriber_stats", "service_time_s"}


def _real(name):
    from saag.core.graph_io import load_graph

    logging.disable(logging.WARNING)
    atm = load_graph(ROOT / "data" / "scenarios" / "atm_system.json")
    if name == "impact_scores":
        from saag.simulation.fault_injector import FaultInjector
        out = FaultInjector(atm, seeds=[42, 123]).run(node_types=["Application", "Broker"]).to_dict()
    elif name == "message_flow_results":
        pytest.importorskip("simpy")
        from saag.simulation.message_flow_simulator import MessageFlowSimulator
        out = MessageFlowSimulator(atm, duration=20.0, fault_node="A0", seed=42).run().to_dict()
    elif name == "runtime_labels":
        pytest.importorskip("simpy")
        from saag.simulation.runtime.labels import runtime_labels
        out = runtime_labels(load_graph(ROOT / "data" / "scenarios" / "tiny_system.json"),
                             seeds=[42], settings={"duration": 10.0})
    else:
        raise AssertionError(f"unknown schema {name!r}")
    return json.loads(json.dumps(out, default=str))


def _check(doc, real, path="$"):
    if isinstance(doc, dict):
        assert isinstance(real, dict), f"{path}: documented as an object"
        dynamic = path.rsplit(".", 1)[-1] in _DYNAMIC
        for k, v in doc.items():
            if dynamic:
                if real:
                    _check(v, next(iter(real.values())), f"{path}.<{k}>")
                continue
            assert k in real, f"{path}.{k}: not in the real output (keys: {sorted(real)})"
            _check(v, real[k], f"{path}.{k}")
    elif isinstance(doc, list) and doc and isinstance(doc[0], dict) and isinstance(real, list) and real:
        for item in doc:
            _check(item, real[0], f"{path}[]")


@pytest.mark.parametrize("body, marker", [p for p in _blocks("json")])
def test_json_snippet_matches_a_real_result(body, marker):
    if not marker.startswith("schema:"):
        pytest.skip("not a schema snippet")
    name = marker.split(":", 1)[1].strip()
    _check(json.loads(body), _real(name))
