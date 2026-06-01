"""Runtime configuration loader for environment-specific pipeline rules."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


RUNTIME_CONFIG_ENV_VAR = "SSA_RUNTIME_CONFIG"


@dataclass(frozen=True)
class CodeQLMethodConfig:
    """Configuration for a single write/read method target."""

    class_name: str = ""
    method_name: str = ""
    topic_arg_index: int = 0


@dataclass(frozen=True)
class CodeQLRuntimeConfig:
    """CodeQL-specific analyzer configuration."""

    cli_path: str = "codeql"
    java_home: str = ""
    build_command: str = ""
    main_method_name: str = "main"
    topic_name_pattern: str = "(.*)"
    write_methods: Tuple[CodeQLMethodConfig, ...] = ()
    read_methods: Tuple[CodeQLMethodConfig, ...] = ()


@dataclass(frozen=True)
class AnalyzerRuntimeConfig:
    """Environment-specific analyzer rules."""

    import_domain_prefix: str = "a.b.c"
    dependency_suffixes: Tuple[str, ...] = ("_lib",)
    dummy_topic_names: Tuple[str, ...] = ("dummytopic",)
    makefile_include_patterns: Tuple[str, ...] = ("include/Makefile_java.mk",)
    custom_topic_name: str = "topic"
    codeql: CodeQLRuntimeConfig = CodeQLRuntimeConfig()


@dataclass(frozen=True)
class AggregatorRuntimeConfig:
    """Environment-specific aggregator rules."""

    dummy_topic_names: Tuple[str, ...] = ("dummytopic",)
    qos_mappings: Dict[str, Dict[str, str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.qos_mappings is None:
            object.__setattr__(self, "qos_mappings", {})


@dataclass(frozen=True)
class RuntimeConfig:
    """Top-level runtime configuration grouped by pipeline module."""

    analyzer: AnalyzerRuntimeConfig = AnalyzerRuntimeConfig()
    aggregator: AggregatorRuntimeConfig = AggregatorRuntimeConfig()


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _default_runtime_config_path() -> Path:
    return _project_root() / "config" / "runtime.yaml"


def _normalize_string_list(values: Any, default: Tuple[str, ...]) -> Tuple[str, ...]:
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
        return default

    normalized = []
    for value in values:
        text = str(value).strip()
        if text:
            normalized.append(text)

    return tuple(normalized) if normalized else default


def _read_yaml_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file) or {}

    if not isinstance(loaded, dict):
        raise ValueError(f"Runtime config must be a mapping: {config_path}")

    return loaded


def _build_codeql_config(codeql_data: Dict[str, Any]) -> CodeQLRuntimeConfig:
    """Build CodeQL config from parsed YAML data."""
    write_methods: List[CodeQLMethodConfig] = []
    for item in codeql_data.get("write_methods", []):
        if isinstance(item, dict):
            write_methods.append(CodeQLMethodConfig(
                class_name=str(item.get("class_name", "")),
                method_name=str(item.get("method_name", "")),
                topic_arg_index=int(item.get("topic_arg_index", 0)),
            ))

    read_methods: List[CodeQLMethodConfig] = []
    for item in codeql_data.get("read_methods", []):
        if isinstance(item, dict):
            read_methods.append(CodeQLMethodConfig(
                class_name=str(item.get("class_name", "")),
                method_name=str(item.get("method_name", "")),
                topic_arg_index=int(item.get("topic_arg_index", 0)),
            ))

    return CodeQLRuntimeConfig(
        cli_path=os.path.expandvars(str(codeql_data.get("cli_path", CodeQLRuntimeConfig.cli_path))),
        java_home=os.path.expandvars(str(codeql_data.get("java_home", CodeQLRuntimeConfig.java_home))),
        build_command=str(codeql_data.get("build_command", CodeQLRuntimeConfig.build_command)),
        main_method_name=str(codeql_data.get("main_method_name", CodeQLRuntimeConfig.main_method_name)),
        topic_name_pattern=str(codeql_data.get("topic_name_pattern", CodeQLRuntimeConfig.topic_name_pattern)),
        write_methods=tuple(write_methods),
        read_methods=tuple(read_methods),
    )


def _build_runtime_config(data: Dict[str, Any]) -> RuntimeConfig:
    analyzer_data = data.get("analyzer") if isinstance(data.get("analyzer"), dict) else {}
    aggregator_data = data.get("aggregator") if isinstance(data.get("aggregator"), dict) else {}

    codeql_data = analyzer_data.get("codeql") if isinstance(analyzer_data.get("codeql"), dict) else {}

    raw_qos_mappings = aggregator_data.get("qos_mappings")
    if isinstance(raw_qos_mappings, dict):
        qos_mappings: Dict[str, Dict[str, str]] = {}
        for dim, mapping in raw_qos_mappings.items():
            if isinstance(mapping, dict):
                qos_mappings[str(dim)] = {
                    str(k): str(v) for k, v in mapping.items()
                }
    else:
        qos_mappings = {}

    return RuntimeConfig(
        analyzer=AnalyzerRuntimeConfig(
            import_domain_prefix=str(
                analyzer_data.get("import_domain_prefix", AnalyzerRuntimeConfig.import_domain_prefix)
            ).strip() or AnalyzerRuntimeConfig.import_domain_prefix,
            dependency_suffixes=_normalize_string_list(
                analyzer_data.get("dependency_suffixes"),
                AnalyzerRuntimeConfig.dependency_suffixes,
            ),
            dummy_topic_names=_normalize_string_list(
                analyzer_data.get("dummy_topic_names"),
                AnalyzerRuntimeConfig.dummy_topic_names,
            ),
            makefile_include_patterns=_normalize_string_list(
                analyzer_data.get("makefile_include_patterns"),
                AnalyzerRuntimeConfig.makefile_include_patterns,
            ),
            custom_topic_name=str(
                analyzer_data.get("custom_topic_name", AnalyzerRuntimeConfig.custom_topic_name)
            ).strip() or AnalyzerRuntimeConfig.custom_topic_name,
            codeql=_build_codeql_config(codeql_data),
        ),
        aggregator=AggregatorRuntimeConfig(
            dummy_topic_names=_normalize_string_list(
                aggregator_data.get("dummy_topic_names"),
                AggregatorRuntimeConfig.dummy_topic_names,
            ),
            qos_mappings=qos_mappings,
        ),
    )


def _resolve_runtime_config_path(config_path: str | None = None) -> Path:
    candidate = config_path or os.getenv(RUNTIME_CONFIG_ENV_VAR)
    if candidate:
        return Path(candidate).expanduser().resolve()
    return _default_runtime_config_path()


@lru_cache(maxsize=4)
def load_runtime_config(config_path: str | None = None) -> RuntimeConfig:
    """Load runtime configuration from YAML, falling back to defaults."""
    resolved_path = _resolve_runtime_config_path(config_path)
    return _build_runtime_config(_read_yaml_config(resolved_path))


def get_runtime_config(config_path: str | None = None) -> RuntimeConfig:
    """Get cached runtime configuration for the active environment."""
    return load_runtime_config(str(_resolve_runtime_config_path(config_path)))