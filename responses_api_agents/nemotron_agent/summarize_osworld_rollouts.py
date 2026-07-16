#!/usr/bin/env python3
"""Summarize OSWorld rollout JSONL files.

The Nemo-GYM OSWorld rollout path writes one JSON object per task with a
top-level reward. This helper keeps the aggregate calculation in one place so
Slurm runs, retries, and RFC reports use the same numbers.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def _domain(row: dict[str, Any]) -> str:
    if row.get("domain"):
        return str(row["domain"])
    metadata = row.get("verifier_metadata") or {}
    if metadata.get("domain"):
        return str(metadata["domain"])
    row_id = row.get("id")
    if isinstance(row_id, str) and "/" in row_id:
        return row_id.split("/", 1)[0]
    return "unknown"


def _model(row: dict[str, Any]) -> str | None:
    params = row.get("responses_create_params") or {}
    model = params.get("model")
    return str(model) if model else None


def _reward(row: dict[str, Any]) -> float:
    try:
        return float(row.get("reward") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def summarize(rollouts_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if rollouts_path.exists():
        with rollouts_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]

    by_domain: dict[str, dict[str, Any]] = collections.defaultdict(
        lambda: {"completed": 0, "success_sum": 0.0, "accuracy": 0.0, "errors": 0}
    )
    model_counts: collections.Counter[str] = collections.Counter()
    success_sum = 0.0
    errors = 0

    for row in rows:
        reward = _reward(row)
        domain = _domain(row)
        by_domain[domain]["completed"] += 1
        by_domain[domain]["success_sum"] += reward
        if row.get("error"):
            by_domain[domain]["errors"] += 1
            errors += 1
        model = _model(row)
        if model:
            model_counts[model] += 1
        success_sum += reward

    for stats in by_domain.values():
        completed = stats["completed"]
        stats["accuracy"] = stats["success_sum"] / completed if completed else 0.0

    return {
        "rollouts": str(rollouts_path),
        "completed": len(rows),
        "success_sum": success_sum,
        "accuracy": success_sum / len(rows) if rows else 0.0,
        "errors": errors,
        "models": dict(sorted(model_counts.items())),
        "by_domain": dict(sorted(by_domain.items())),
    }


def write_markdown(
    summary: dict[str, Any],
    report_path: Path,
    *,
    label: str | None,
    expected_accuracy: float | None,
) -> None:
    title = label or "OSWorld Rollout Summary"
    accuracy = float(summary["accuracy"])
    lines = [
        f"# {title}",
        "",
        f"- Rollouts: `{summary['rollouts']}`",
        f"- Completed: {summary['completed']}",
        f"- Success sum: {summary['success_sum']:.6f}",
        f"- Accuracy: {accuracy:.6f} ({accuracy * 100:.2f}%)",
        f"- Errors: {summary['errors']}",
    ]
    if expected_accuracy is not None:
        delta = accuracy - expected_accuracy
        lines.append(f"- Expected accuracy: {expected_accuracy:.6f} ({expected_accuracy * 100:.2f}%)")
        lines.append(f"- Delta from expected: {delta:+.6f} ({delta * 100:+.2f} pp)")

    if summary["models"]:
        lines.extend(["", "## Models"])
        for model, count in summary["models"].items():
            lines.append(f"- `{model}`: {count}")

    lines.extend(["", "## By Domain", "", "| Domain | Completed | Success Sum | Accuracy | Errors |"])
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for domain, stats in summary["by_domain"].items():
        lines.append(
            "| {domain} | {completed} | {success_sum:.6f} | {accuracy:.6f} | {errors} |".format(
                domain=domain,
                completed=stats["completed"],
                success_sum=stats["success_sum"],
                accuracy=stats["accuracy"],
                errors=stats["errors"],
            )
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts_jsonl", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--report-md", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--expected-accuracy", type=float)
    args = parser.parse_args()

    summary = summarize(args.rollouts_jsonl)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.summary_json:
        args.summary_json.write_text(text + "\n", encoding="utf-8")
    if args.report_md:
        write_markdown(
            summary,
            args.report_md,
            label=args.label,
            expected_accuracy=args.expected_accuracy,
        )


if __name__ == "__main__":
    main()
