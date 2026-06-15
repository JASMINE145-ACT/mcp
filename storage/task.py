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
    return json.loads(p.read_text(encoding="utf-8"))


def save_markdown(task_dir: Path, markdown: str, topic: str) -> Path:
    date = datetime.now().strftime("%Y-%m-%d")
    slug = _slug(topic)
    md_path = task_dir / "article.md"
    md_path.write_text(markdown, encoding="utf-8")
    # Also save to global markdown dir
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


STATUS_VALUES = [
    "generated",
    "html_rendered",
    "image_uploaded",
    "draft_created",
    "draft_failed",
    "published",
    "publish_failed",
]


def update_status(task_dir: Path, status: str) -> None:
    article_json_path = task_dir / "article.json"
    if article_json_path.exists():
        data = json.loads(article_json_path.read_text(encoding="utf-8"))
        data["status"] = status
        article_json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
