#!/usr/bin/env python3
"""
Step 3 (Topic metadata extraction) — single-run script.

Reads refined topic clusters (e.g. Java Step 2 output event from `ai-topic-refined`)
and the Synthetic Slack CSV, then calls one LLM model to generate per-topic metadata.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Allow imports from phase_evaluation_engine/*
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_clients import call_model_with_retry  # noqa: E402
from step3.step3_utils import (  # noqa: E402
    default_out_path,
    extract_json_object,
    format_messages_for_prompt,
    iter_events_from_json_or_jsonl,
    read_messages,
    render_prompt,
    select_event,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Step 3 metadata extraction on refined clusters")
    p.add_argument(
        "--refined",
        required=True,
        help="Path to a refined event JSON/JSONL (e.g. logs/step2-ws-gemini-batch_51.json or a Kafka dump file).",
    )
    p.add_argument(
        "--csv",
        default=str(Path(__file__).resolve().parents[1] / "data" / "Synthetic_Slack_Messages.csv"),
        help="Path to Synthetic_Slack_Messages.csv",
    )
    p.add_argument("--provider", default="google", help="LLM provider (default: google)")
    p.add_argument("--model", default="gemini-2.0-flash-001", help="Model name (default: gemini-2.0-flash-001)")
    p.add_argument(
        "--batch-id",
        default=None,
        help="Optional batchId filter if the refined file contains multiple events (JSONL).",
    )
    p.add_argument(
        "--max-clusters",
        type=int,
        default=0,
        help="Limit number of clusters to process (0 = all).",
    )
    p.add_argument(
        "--max-messages-per-cluster",
        type=int,
        default=60,
        help="Limit messages included per cluster in the prompt (default: 60).",
    )
    p.add_argument(
        "--max-message-chars",
        type=int,
        default=600,
        help="Truncate each message text to this many characters (default: 600).",
    )
    p.add_argument(
        "--max-output-tokens",
        type=int,
        default=4096,
        help="LLM max output tokens (default: 4096).",
    )
    p.add_argument(
        "--out",
        default="",
        help="Output JSON path. Default: output/step3_metadata_<timestamp>.json",
    )
    p.add_argument(
        "--prompt",
        default=str(Path(__file__).resolve().parent / "step3_prompt.md"),
        help="Prompt template path (default: step3/step3_prompt.md)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.provider == "google" and not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY is not set (required for provider=google)")
    if args.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set (required for provider=openai)")

    refined_path = Path(args.refined).expanduser().resolve()
    csv_path = Path(args.csv).expanduser().resolve()
    prompt_path = Path(args.prompt).expanduser().resolve()

    if not refined_path.exists():
        raise SystemExit(f"Refined file not found: {refined_path}")
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")
    if not prompt_path.exists():
        raise SystemExit(f"Prompt file not found: {prompt_path}")

    prompt_template = prompt_path.read_text(encoding="utf-8")
    messages_by_id = read_messages(csv_path)

    events = list(iter_events_from_json_or_jsonl(refined_path))
    event = select_event(events, args.batch_id)

    clusters: List[Dict[str, Any]] = event.get("clusters") or []
    if not isinstance(clusters, list) or not clusters:
        raise SystemExit("No clusters found in refined event")

    if args.max_clusters and args.max_clusters > 0:
        clusters = clusters[: args.max_clusters]

    results: List[Dict[str, Any]] = []
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "duration_s": 0.0}

    started_at = datetime.now().isoformat()
    for idx, cluster in enumerate(clusters, start=1):
        cluster_id = cluster.get("clusterId") or cluster.get("cluster_id") or ""
        message_ids, messages_block = format_messages_for_prompt(
            cluster,
            messages_by_id,
            max_messages=args.max_messages_per_cluster,
            max_chars=args.max_message_chars,
        )

        cluster_info = {
            "cluster_id": cluster_id,
            "draft_title": cluster.get("draftTitle") or cluster.get("draft_title") or "",
            "channel": cluster.get("channel") or "",
            "thread_id": cluster.get("threadId") or cluster.get("thread_id"),
            "participants": cluster.get("participants") or [],
            "message_ids": message_ids,
        }

        prompt = render_prompt(prompt_template, cluster_info, messages_block)
        model_result = call_model_with_retry(
            args.provider,
            args.model,
            prompt,
            max_retries=3,
            max_tokens=args.max_output_tokens,
            temperature=0.1,
        )

        usage = model_result.get("usage") or {}
        totals["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        totals["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)
        totals["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
        totals["duration_s"] += float(model_result.get("duration", 0) or 0)

        parsed, parse_error = extract_json_object(model_result.get("response", "") if model_result.get("success") else "")

        results.append(
            {
                "cluster_id": cluster_id,
                "success": bool(model_result.get("success")) and parsed is not None,
                "provider": args.provider,
                "model": args.model,
                "usage": usage,
                "duration_s": model_result.get("duration", 0),
                "parse_error": parse_error,
                "metadata": parsed or {},
                "raw_response": model_result.get("response", ""),
                "error": model_result.get("error", ""),
            }
        )

        ok = "OK" if results[-1]["success"] else "FAIL"
        print(
            f"[{idx}/{len(clusters)}] {ok} cluster={cluster_id} "
            f"tokens_in={usage.get('prompt_tokens', 0)} tokens_out={usage.get('completion_tokens', 0)}"
        )

    out_path = Path(args.out).expanduser().resolve() if args.out else default_out_path().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "step": 3,
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(),
        "input": {
            "refined_file": str(refined_path),
            "csv_file": str(csv_path),
            "batch_id": event.get("batchId"),
            "workspace_id": event.get("workspaceId"),
            "cluster_count": len(clusters),
        },
        "llm": {
            "provider": args.provider,
            "model": args.model,
            "max_output_tokens": args.max_output_tokens,
            "prompt_file": str(prompt_path),
        },
        "totals": totals,
        "results": results,
    }

    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDone.")
    print(f"- Output: {out_path}")
    print(
        f"- Totals: prompt_tokens={totals['prompt_tokens']} completion_tokens={totals['completion_tokens']} "
        f"total_tokens={totals['total_tokens']} duration_s={totals['duration_s']:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

