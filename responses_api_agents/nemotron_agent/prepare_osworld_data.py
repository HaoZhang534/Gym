"""Create Nemo Gym JSONL rows from real OSWorld task metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
DEFAULT_EVALUATION_EXAMPLES_DIR = APP_DIR / "data" / "evaluation_examples"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-all-meta-path",
        default=str(DEFAULT_EVALUATION_EXAMPLES_DIR / "test_test.json"),
    )
    parser.add_argument(
        "--test-config-base-dir",
        default=str(DEFAULT_EVALUATION_EXAMPLES_DIR),
    )
    parser.add_argument("--output-jsonl", default=str(APP_DIR / "data" / "osworld_test_test.jsonl"))
    parser.add_argument("--agent-name", default="nemotron_agent")
    parser.add_argument("--model", default="nvidia/nemotron-vl")
    parser.add_argument("--domain", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta_path = Path(args.test_all_meta_path).resolve()
    base_dir = Path(args.test_config_base_dir).resolve()
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with meta_path.open("r", encoding="utf-8") as f:
        test_meta = json.load(f)

    rows = []
    for domain, example_ids in test_meta.items():
        if args.domain and domain != args.domain:
            continue
        for example_id in example_ids:
            task_path = base_dir / "examples" / domain / f"{example_id}.json"
            with task_path.open("r", encoding="utf-8") as f:
                example = json.load(f)
            try:
                task_config_path = str(task_path.relative_to(APP_DIR))
            except ValueError:
                task_config_path = str(task_path)

            rows.append(
                {
                    "id": f"{domain}/{example_id}",
                    "domain": domain,
                    "example_id": example_id,
                    "responses_create_params": {
                        "model": args.model,
                        "input": [
                            {
                                "role": "user",
                                "content": example["instruction"],
                            }
                        ],
                    },
                    "verifier_metadata": {
                        "domain": domain,
                        "example_id": example_id,
                        "task_config_path": task_config_path,
                        "instruction": example["instruction"],
                    },
                    "agent_ref": {
                        "type": "responses_api_agents",
                        "name": args.agent_name,
                    },
                }
            )

            if args.limit and len(rows) >= args.limit:
                break
        if args.limit and len(rows) >= args.limit:
            break

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    print(f"Wrote {len(rows)} real OSWorld rows to {out_path}")


if __name__ == "__main__":
    main()
