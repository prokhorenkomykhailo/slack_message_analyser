#!/usr/bin/env python3
"""
Calculate approximate token usage for the Step 6 comprehensive test report.

The script loads `output/step6_comprehensive_tests/comprehensive_test_report.json`,
counts tokens for the input message payload (channel, user, text, timestamp) and the
AI-generated output (metadata or discard reasoning), and prints an aggregated summary.

Token counting uses OpenAI's `tiktoken` package with the configurable encoding
(`cl100k_base` by default), which is a reasonable approximation for modern GPT-style models.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

try:
    import tiktoken
except ImportError as exc:  # pragma: no cover - defensive error reporting
    raise SystemExit(
        "tiktoken is required for token counting. "
        "Install it with `pip install tiktoken`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate token usage for Step 6 comprehensive tests."
    )
    default_report = os.path.join(
        "output", "step6_comprehensive_tests", "comprehensive_test_report.json"
    )
    default_output = os.path.join(
        "output", "step6_comprehensive_tests", "step6_token_usage.json"
    )

    parser.add_argument(
        "--report",
        default=default_report,
        help=f"Path to Step 6 comprehensive test report (default: {default_report})",
    )
    parser.add_argument(
        "--encoding",
        default="cl100k_base",
        help="Tokenizer encoding name (default: cl100k_base)",
    )
    parser.add_argument(
        "--output",
        default=default_output,
        help="Optional path to save the aggregated token stats as JSON.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write the JSON summary file (print to stdout only).",
    )
    return parser.parse_args()


def count_tokens(text: str, encoding) -> int:
    """Return token count for text using the provided encoding."""
    if not text:
        return 0
    return len(encoding.encode(text))


def build_input_payload(message: Dict[str, Any]) -> str:
    """Serialize the message payload to a JSON string for token counting."""
    payload = {
        "channel": message.get("channel"),
        "user": message.get("user"),
        "text": message.get("text"),
        "timestamp": message.get("timestamp"),
        "thread_ts": message.get("thread_ts"),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def build_output_payload(result: Dict[str, Any]) -> str:
    """
    Serialize the AI output portion (metadata or discard reasoning) to JSON.
    Only fields originating from AI generations are included to approximate output tokens.
    """
    action = result.get("action")
    output: Dict[str, Any] = {"action": action}

    if action in {"created", "updated"} and "metadata" in result:
        output["metadata"] = result["metadata"]
    elif action == "discarded":
        output["reason"] = result.get("reason")
        output["worthiness_evaluation"] = result.get("worthiness_evaluation")

    return json.dumps(output, ensure_ascii=False, separators=(",", ":"))


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.report):
        raise SystemExit(f"Report file not found: {args.report}")

    with open(args.report, "r") as fp:
        report = json.load(fp)

    test_results: List[Dict[str, Any]] = report.get("test_results", [])
    if not test_results:
        raise SystemExit("No test_results found in the provided report.")

    encoding = tiktoken.get_encoding(args.encoding)

    rows: List[Dict[str, Any]] = []
    total_input = 0
    total_output = 0

    for test in test_results:
        message = test.get("message", {})
        result = test.get("result", {})

        input_tokens = count_tokens(build_input_payload(message), encoding)
        output_tokens = count_tokens(build_output_payload(result), encoding)

        rows.append(
            {
                "test_id": test.get("test_id"),
                "category": test.get("category"),
                "passed": test.get("passed"),
                "expected_action": test.get("expected_action"),
                "actual_action": test.get("actual_action"),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
        )

        total_input += input_tokens
        total_output += output_tokens

    total_combined = total_input + total_output

    header = (
        f"{'Test':<6} {'Cat':<10} {'Input':>10} {'Output':>10} "
        f"{'Total':>10} {'Pass':>5} {'Exp→Act':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        exp_act = f"{row['expected_action']}→{row['actual_action']}"
        print(
            f"{row['test_id']:<6} "
            f"{row['category']:<10} "
            f"{row['input_tokens']:>10} "
            f"{row['output_tokens']:>10} "
            f"{row['total_tokens']:>10} "
            f"{'Y' if row['passed'] else 'N':>5} "
            f"{exp_act:>12}"
        )

    print("\nTotals")
    print("-" * 40)
    print(f"Total tests     : {len(rows)}")
    print(f"Total input tok : {total_input:,}")
    print(f"Total output tok: {total_output:,}")
    print(f"Combined tokens : {total_combined:,}")

    if not args.no_save:
        summary = {
            "report_path": os.path.abspath(args.report),
            "encoding": args.encoding,
            "total_tests": len(rows),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_combined,
            "per_test": rows,
        }

        output_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(output_dir, exist_ok=True)

        with open(args.output, "w") as fp:
            json.dump(summary, fp, indent=2)

        print(f"\n📁 Token summary saved to: {args.output}")


if __name__ == "__main__":
    main()
