"""
Tests for cli/common/arguments.py's Neo4j credential env-var resolution.
"""
import argparse

from cli.common.arguments import add_neo4j_arguments


def _parsed_user(monkeypatch, **env) -> str:
    for key in ("NEO4J_USERNAME", "NEO4J_USER"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    parser = argparse.ArgumentParser()
    add_neo4j_arguments(parser)
    return parser.parse_args([]).user


def test_neo4j_username_env_var_is_honoured(monkeypatch):
    """.env, docker-compose.yml, and the API all set NEO4J_USERNAME — the
    CLI used to read only NEO4J_USER, so a username set the way every other
    part of the stack expects had no effect on the CLI."""
    assert _parsed_user(monkeypatch, NEO4J_USERNAME="alice") == "alice"


def test_neo4j_user_env_var_still_works_as_fallback(monkeypatch):
    """Anyone already relying on the CLI-only NEO4J_USER name must not break."""
    assert _parsed_user(monkeypatch, NEO4J_USER="bob") == "bob"


def test_neo4j_username_takes_precedence_over_neo4j_user(monkeypatch):
    assert _parsed_user(monkeypatch, NEO4J_USERNAME="alice", NEO4J_USER="bob") == "alice"


def test_neo4j_user_defaults_to_neo4j(monkeypatch):
    assert _parsed_user(monkeypatch) == "neo4j"
