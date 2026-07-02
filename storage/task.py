import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger

BASE_DIR = Path(__file__).parent.parent
STORAGE_DIR = BASE_DIR / "storage"
DRAFTS_DIR = STORAGE_DIR / "drafts"
MARKDOWN_DIR = STORAGE_DIR / "markdown"
HTML_DIR = STORAGE_DIR / "html"
IMAGES_DIR = STORAGE_DIR / "images"
LOGS_DIR = STORAGE_DIR / "logs"

# Ordered pipeline steps — each step name is the "resume point" key
PIPELINE_STEPS = [
    "markdown_ready",      # Markdown content available
    "html_rendered",       # HTML rendered from markdown
    "content_validated",   # Content quality check passed
    "sources_saved",       # Source citations saved (optional)
    "cover_uploaded",      # Cover image uploaded to WeChat CDN
    "draft_created",       # Draft created in WeChat backend
    "manual_review",       # Waiting for human review
    "published",           # Published to subscribers
]

TERMINAL_STEPS = {"published", "publish_failed", "aborted"}


def _slug(topic: str) -> str:
    s = re.sub(r"[^\w一-鿿]", "-", topic)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:40]


def create_task_dir(topic: str) -> Path:
    date = datetime.now().strftime("%Y-%m-%d")
    slug = _slug(topic)
    task_dir = DRAFTS_DIR / f"{date}-{slug}"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "logs.txt").touch()
    logger.info(f"任务目录创建: {task_dir}")
    return task_dir


def save_article_json(task_dir: Path, data: dict) -> Path:
    p = task_dir / "article.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_article_json(task_dir: Path) -> dict:
    p = task_dir / "article.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def save_markdown(task_dir: Path, markdown: str, topic: str) -> Path:
    date = datetime.now().strftime("%Y-%m-%d")
    slug = _slug(topic)
    md_path = task_dir / "article.md"
    md_path.write_text(markdown, encoding="utf-8")
    global_md = MARKDOWN_DIR / f"{date}-{slug}.md"
    MARKDOWN_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(md_path, global_md)
    return md_path


def save_html(task_dir: Path, html: str, topic: str) -> Path:
    date = datetime.now().strftime("%Y-%m-%d")
    slug = _slug(topic)
    html_path = task_dir / "article.html"
    html_path.write_text(html, encoding="utf-8")
    global_html = HTML_DIR / f"{date}-{slug}.html"
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(html_path, global_html)
    return html_path


def save_upload_result(task_dir: Path, result: dict) -> None:
    p = task_dir / "upload_result.json"
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def save_draft_result(task_dir: Path, result: dict) -> None:
    p = task_dir / "draft_result.json"
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Source citations ────────────────────────────────────────────────────────

def save_sources(task_dir: Path, sources: list[dict]) -> Path:
    """Save source citation list. Each source: {url, title, date, key_points, risk_level}."""
    p = task_dir / "sources.json"
    p.write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_sources(task_dir: Path) -> list[dict]:
    p = task_dir / "sources.json"
    if not p.exists():
        return []
    return json.loads(p.read_text(encoding="utf-8"))


# ── Step history ────────────────────────────────────────────────────────────

def _steps_path(task_dir: Path) -> Path:
    return task_dir / "steps.jsonl"


def record_step(
    task_dir: Path,
    step: str,
    status: str,  # "success" | "failed" | "skipped"
    output: Optional[dict] = None,
    error: Optional[str] = None,
    retry_count: int = 0,
) -> None:
    """Append one step record to steps.jsonl."""
    entry = {
        "step": step,
        "status": status,
        "output": output or {},
        "error": error,
        "retry_count": retry_count,
        "ts": datetime.now().isoformat(),
    }
    with _steps_path(task_dir).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_steps(task_dir: Path) -> list[dict]:
    p = _steps_path(task_dir)
    if not p.exists():
        return []
    steps = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                steps.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return steps


def get_last_successful_step(task_dir: Path) -> Optional[str]:
    """Return the name of the last step that succeeded, or None."""
    last_success = None
    for entry in load_steps(task_dir):
        if entry.get("status") == "success":
            last_success = entry.get("step")
    return last_success


def find_resume_step(task_dir: Path) -> str:
    """Return the next pipeline step to run based on step history."""
    last = get_last_successful_step(task_dir)
    if last is None:
        return PIPELINE_STEPS[0]
    try:
        idx = PIPELINE_STEPS.index(last)
        if idx + 1 < len(PIPELINE_STEPS):
            return PIPELINE_STEPS[idx + 1]
        return "completed"
    except ValueError:
        return PIPELINE_STEPS[0]


def get_resume_context(task_dir: Path) -> dict:
    """Return all context needed to resume a pipeline."""
    data = load_article_json(task_dir)
    steps = load_steps(task_dir)
    sources = load_sources(task_dir)
    resume_step = find_resume_step(task_dir)

    # Collect outputs from successful steps
    context: dict = {
        "task_dir": str(task_dir),
        "title": data.get("title", ""),
        "topic": data.get("topic", ""),
        "author": data.get("author", ""),
        "digest": data.get("digest", ""),
        "cover_path": data.get("cover_path", ""),
        "status": data.get("status", "unknown"),
        "resume_step": resume_step,
        "steps_completed": [s["step"] for s in steps if s.get("status") == "success"],
        "sources_count": len(sources),
    }

    # Attach key outputs from step history
    for step_entry in steps:
        if step_entry.get("status") == "success":
            out = step_entry.get("output", {})
            if step_entry["step"] == "cover_uploaded":
                context["thumb_media_id"] = out.get("thumb_media_id", "")
            elif step_entry["step"] == "draft_created":
                context["media_id"] = out.get("media_id", "")
            elif step_entry["step"] == "html_rendered":
                context["html_path"] = out.get("html_path", "")

    # Fall back to article.json fields
    if not context.get("thumb_media_id"):
        try:
            upload = json.loads((task_dir / "upload_result.json").read_text(encoding="utf-8"))
            context["thumb_media_id"] = upload.get("thumb_media_id", "")
        except Exception:
            pass
    if not context.get("media_id"):
        try:
            draft = json.loads((task_dir / "draft_result.json").read_text(encoding="utf-8"))
            context["media_id"] = draft.get("media_id", "")
        except Exception:
            pass

    return context


# ── Status ──────────────────────────────────────────────────────────────────

STATUS_VALUES = [
    "generated",
    "html_rendered",
    "content_validated",
    "image_uploaded",
    "draft_created",
    "draft_failed",
    "manual_review",
    "published",
    "publish_failed",
    "aborted",
]


def update_status(task_dir: Path, status: str) -> None:
    article_json_path = task_dir / "article.json"
    if article_json_path.exists():
        data = json.loads(article_json_path.read_text(encoding="utf-8"))
        data["status"] = status
        data["updated_at"] = datetime.now().isoformat()
        article_json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ── Metrics ─────────────────────────────────────────────────────────────────

METRICS_PATH = STORAGE_DIR / "metrics.jsonl"


def log_metric(tool: str, duration_s: float, status: str, error_code: Optional[str] = None) -> None:
    """Append one tool-call metric to storage/metrics.jsonl."""
    entry = {
        "tool": tool,
        "duration_s": round(duration_s, 3),
        "status": status,
        "ts": datetime.now().isoformat(),
    }
    if error_code:
        entry["error_code"] = error_code
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
