"""
选题库管理 — pending / approved / rejected / published 四个队列。
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).parent.parent
TOPICS_DIR = BASE_DIR / "storage" / "topics"

QUEUES = ("pending", "approved", "rejected", "published")


def _queue_path(queue: str) -> Path:
    TOPICS_DIR.mkdir(parents=True, exist_ok=True)
    return TOPICS_DIR / f"{queue}.json"


def _load(queue: str) -> list[dict]:
    p = _queue_path(queue)
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


def _save(queue: str, items: list[dict]) -> None:
    _queue_path(queue).write_text(
        json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def add_topic(
    topic: str,
    account: str = "",
    priority: str = "normal",
    angle: str = "",
    source_urls: Optional[list[str]] = None,
    deadline: str = "",
) -> dict:
    """Add a topic to the pending queue. Returns the new topic entry."""
    items = _load("pending")
    entry = {
        "id": f"topic-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "topic": topic,
        "account": account,
        "priority": priority,
        "angle": angle,
        "source_urls": source_urls or [],
        "deadline": deadline,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
    }
    items.append(entry)
    _save("pending", items)
    return entry


def approve_topic(topic_id: str, note: str = "") -> Optional[dict]:
    """Move a topic from pending → approved."""
    pending = _load("pending")
    matched = [t for t in pending if t.get("id") == topic_id]
    if not matched:
        return None
    entry = matched[0]
    entry["status"] = "approved"
    entry["approved_at"] = datetime.now().isoformat()
    if note:
        entry["approve_note"] = note
    remaining = [t for t in pending if t.get("id") != topic_id]
    _save("pending", remaining)
    approved = _load("approved")
    approved.append(entry)
    _save("approved", approved)
    return entry


def reject_topic(topic_id: str, reason: str = "") -> Optional[dict]:
    """Move a topic from pending → rejected."""
    pending = _load("pending")
    matched = [t for t in pending if t.get("id") == topic_id]
    if not matched:
        return None
    entry = matched[0]
    entry["status"] = "rejected"
    entry["rejected_at"] = datetime.now().isoformat()
    if reason:
        entry["reject_reason"] = reason
    remaining = [t for t in pending if t.get("id") != topic_id]
    _save("pending", remaining)
    rejected = _load("rejected")
    rejected.append(entry)
    _save("rejected", rejected)
    return entry


def mark_published(topic_id: str, task_dir: str = "", media_id: str = "") -> Optional[dict]:
    """Move a topic from approved → published."""
    approved = _load("approved")
    matched = [t for t in approved if t.get("id") == topic_id]
    if not matched:
        return None
    entry = matched[0]
    entry["status"] = "published"
    entry["published_at"] = datetime.now().isoformat()
    if task_dir:
        entry["task_dir"] = task_dir
    if media_id:
        entry["wechat_media_id"] = media_id
    remaining = [t for t in approved if t.get("id") != topic_id]
    _save("approved", remaining)
    published = _load("published")
    published.append(entry)
    _save("published", published)
    return entry


def list_topics(queue: str = "all", account: str = "", priority: str = "") -> dict:
    """List topics. queue='all' returns all queues merged."""
    if queue == "all":
        result: dict[str, list] = {}
        for q in QUEUES:
            items = _load(q)
            if account:
                items = [t for t in items if t.get("account") == account]
            if priority:
                items = [t for t in items if t.get("priority") == priority]
            result[q] = items
        return result

    items = _load(queue)
    if account:
        items = [t for t in items if t.get("account") == account]
    if priority:
        items = [t for t in items if t.get("priority") == priority]
    return {queue: items}


def get_next_approved(account: str = "") -> Optional[dict]:
    """Get the next approved topic to work on (highest priority first)."""
    approved = _load("approved")
    if account:
        approved = [t for t in approved if t.get("account") == account]
    priority_order = {"high": 0, "normal": 1, "low": 2}
    approved.sort(key=lambda t: (priority_order.get(t.get("priority", "normal"), 1), t.get("created_at", "")))
    return approved[0] if approved else None
