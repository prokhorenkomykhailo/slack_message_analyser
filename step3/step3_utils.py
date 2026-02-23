import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class Message:
    id: int
    channel: str
    user: str
    timestamp: str
    text: str
    thread_id: Optional[str]


def read_messages(csv_path: Path) -> Dict[int, Message]:
    messages: Dict[int, Message] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            thread_id = row.get("thread_id")
            if thread_id == "None":
                thread_id = None
            messages[idx] = Message(
                id=idx,
                channel=row.get("channel", ""),
                user=row.get("user_name", ""),
                timestamp=row.get("timestamp", ""),
                text=row.get("text", ""),
                thread_id=thread_id,
            )
    return messages


def iter_events_from_json_or_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # If it's a single JSON object, parse directly.
    if text[0] == "{" and "\n" not in text:
        try:
            return [json.loads(text)]
        except Exception:
            pass

    # Otherwise treat as JSONL (1 JSON per line), skipping empty/bad lines.
    events: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            events.append(json.loads(line))
        except Exception:
            continue
    return events


def select_event(events: List[Dict[str, Any]], batch_id: Optional[str]) -> Dict[str, Any]:
    if not events:
        raise RuntimeError("No events found in refined input file")
    if batch_id:
        for e in reversed(events):
            if e.get("batchId") == batch_id:
                return e
        raise RuntimeError(f"No event found with batchId={batch_id}")
    return events[-1]


def normalize_message_ids(raw_ids: Any) -> List[int]:
    if not isinstance(raw_ids, list):
        return []
    out: List[int] = []
    for v in raw_ids:
        if isinstance(v, int):
            out.append(v)
        elif isinstance(v, str) and v.strip().isdigit():
            out.append(int(v.strip()))
    return out


def format_messages_for_prompt(
    cluster: Dict[str, Any],
    messages_by_id: Dict[int, Message],
    max_messages: int,
    max_chars: int,
) -> Tuple[List[int], str]:
    message_ids = normalize_message_ids(cluster.get("messageIds") or cluster.get("message_ids") or [])
    message_ids = message_ids[: max(0, max_messages)]

    formatted_messages: List[str] = []
    for mid in message_ids:
        msg = messages_by_id.get(mid)
        if not msg:
            continue
        text = msg.text or ""
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        formatted_messages.append(
            f"ID: {msg.id} | Channel: {msg.channel} | User: {msg.user} | Thread: {msg.thread_id or 'None'} | Text: {text}"
        )
    return message_ids, os.linesep.join(formatted_messages)


def render_prompt(template: str, cluster_info: Dict[str, Any], messages_block: str) -> str:
    return (
        template.replace("{{cluster_info_json}}", json.dumps(cluster_info, ensure_ascii=False))
        .replace("{{messages}}", messages_block)
        .strip()
    )


def extract_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not text:
        return None, "empty_response"
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None, "no_json_object_found"
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate), None
    except Exception as e:
        return None, f"json_parse_error: {e}"


def default_out_path() -> Path:
    return Path("output") / f"step3_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

