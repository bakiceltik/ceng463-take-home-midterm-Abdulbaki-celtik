"""Run-directory bootstrap helpers for question pipelines."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

from src.common.io import ensure_dir, project_path, write_json, write_text


def _slugify(value: str) -> str:
    slug = []
    for char in value.lower():
        if char.isalnum():
            slug.append(char)
        elif char in {"-", "_"}:
            slug.append(char)
        else:
            slug.append("_")
    return "".join(slug).strip("_") or "run"


def _render_markdown_list(items: Iterable[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def bootstrap_run(
    question_slug: str,
    config: dict,
    summary: str,
    checklist: list[str],
    expected_artifacts: list[str],
) -> Path:
    experiment_name = _slugify(str(config.get("experiment_name", question_slug)))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = ensure_dir(project_path("outputs", question_slug, f"{timestamp}_{experiment_name}"))

    for child in ("figures", "tables", "logs", "models"):
        ensure_dir(run_dir / child)

    write_json(
        run_dir / "resolved_config.json",
        {
            "created_at": timestamp,
            "question_slug": question_slug,
            "config": config,
        },
    )

    overview = "\n".join(
        [
            f"# {question_slug}",
            "",
            summary,
            "",
            "## Starter Checklist",
            _render_markdown_list(checklist),
            "",
            "## Expected Artifacts",
            _render_markdown_list(expected_artifacts),
            "",
            "## Run Directories",
            "- `figures/` for plots and visualisations",
            "- `tables/` for metrics tables and CSV exports",
            "- `logs/` for console logs or training history",
            "- `models/` for checkpoints or serialized estimators",
        ]
    )
    write_text(run_dir / "README.md", overview)

    return run_dir
