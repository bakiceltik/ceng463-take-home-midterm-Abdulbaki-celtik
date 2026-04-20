"""Helpers for loading simple YAML-like configs with inheritance."""

from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _strip_comments(line: str) -> str:
    return re.sub(r"\s+#.*$", "", line).rstrip("\n")


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()

    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None

    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(item.strip()) for item in inner.split(",")]

    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def _read_lines(path: Path) -> list[tuple[int, str]]:
    parsed_lines: list[tuple[int, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = _strip_comments(raw_line)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        parsed_lines.append((indent, line.strip()))
    return parsed_lines


def _parse_block(lines: list[tuple[int, str]], start: int, indent: int) -> tuple[Any, int]:
    if start >= len(lines):
        return {}, start

    current_indent, current_text = lines[start]
    if current_indent != indent:
        return {}, start

    if current_text.startswith("- "):
        items: list[Any] = []
        index = start
        while index < len(lines):
            line_indent, text = lines[index]
            if line_indent < indent:
                break
            if line_indent > indent:
                raise ValueError(f"Unexpected indentation in config near: {text}")
            if not text.startswith("- "):
                break

            payload = text[2:].strip()
            index += 1
            if payload:
                items.append(_parse_scalar(payload))
                continue

            nested, index = _parse_block(lines, index, indent + 2)
            items.append(nested)

        return items, index

    mapping: dict[str, Any] = {}
    index = start
    while index < len(lines):
        line_indent, text = lines[index]
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"Unexpected indentation in config near: {text}")
        if text.startswith("- "):
            break

        if ":" not in text:
            raise ValueError(f"Invalid config line: {text}")

        key, raw_value = text.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        index += 1

        if raw_value:
            mapping[key] = _parse_scalar(raw_value)
            continue

        if index < len(lines) and lines[index][0] > indent:
            nested, index = _parse_block(lines, index, lines[index][0])
            mapping[key] = nested
        else:
            mapping[key] = {}

    return mapping, index


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()

    lines = _read_lines(config_path)
    config, _ = _parse_block(lines, 0, 0)

    base_name = config.pop("extends", None)
    if base_name:
        base_path = Path(base_name)
        if not base_path.is_absolute():
            base_path = config_path.parent / base_path
        base_config = load_config(base_path)
        config = _merge_dicts(base_config, config)

    config["_config_path"] = str(config_path)
    return config
