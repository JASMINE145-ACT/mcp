import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from loguru import logger

app = typer.Typer(name="wechat-ai-publisher", help="微信公众号 AI 自动写作与草稿发布系统")

BASE_DIR = Path(__file__).parent
LOG_FILE = BASE_DIR / "storage" / "logs" / "app.log"


def _setup_logging(level: str = "INFO") -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.add(sys.stderr, level=level, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    logger.add(str(LOG_FILE), level="DEBUG", rotation="10 MB", retention="30 days", encoding="utf-8")


def _run_generate(topic: str, style_path: str) -> tuple[dict, Path]:
    """Core generate logic. Returns (article_json, task_dir)."""
    from config.settings import get_settings, load_style_guide
    from ai.article_generator import generate_article, polish_article
    from ai.content_checker import check_article
    from render.markdown_to_html import convert_markdown_to_wechat_html, save_html as save_html_preview
    from render.wechat_html_template import wrap_for_wechat
    from storage.task import (
        create_task_dir, save_article_json, save_markdown,
        save_html, update_status,
    )

    s = get_settings()
    start = time.time()
    logger.info(f"=== 开始生成文章 ===")
    logger.info(f"主题: {topic}")

    # Load style guide
    style_guide = load_style_guide(style_path)
    logger.info(f"写作风格文件读取成功: {style_path}")

    # Generate article
    article = generate_article(topic, style_guide, author=s.DEFAULT_AUTHOR)
    logger.info(f"文章生成成功: 《{article.get('title')}》")

    # Polish
    article = polish_article(article)
    logger.info("二次润色完成")

    # Content check
    check = check_article(article)
    if check.errors:
        for err in check.errors:
            logger.error(f"内容检查错误: {err}")
        raise typer.Exit(1)
    for warn in check.warnings:
        logger.warning(f"内容检查警告: {warn}")

    # Create task dir
    task_dir = create_task_dir(topic)

    # Save article.json
    article_data = {
        "topic": topic,
        "title": article.get("title", ""),
        "digest": article.get("digest", ""),
        "author": article.get("author", s.DEFAULT_AUTHOR),
        "tags": article.get("tags", []),
        "cover_prompt": article.get("cover_prompt", ""),
        "markdown_path": "",
        "html_path": "",
        "cover_path": "",
        "wechat_draft_media_id": "",
        "created_at": datetime.now().isoformat(),
        "status": "generated",
    }
    save_article_json(task_dir, article_data)

    # Save markdown
    md_path = save_markdown(task_dir, article.get("markdown", ""), topic)
    logger.info(f"Markdown 已保存: {md_path}")
    update_status(task_dir, "generated")

    # Convert to HTML
    html_content = convert_markdown_to_wechat_html(article.get("markdown", ""))
    html_path = save_html(task_dir, html_content, topic)
    save_html_preview(html_content, str(html_path))
    logger.info(f"HTML 已保存: {html_path}")
    update_status(task_dir, "html_rendered")

    # Update article.json with paths
    article_data["markdown_path"] = str(md_path)
    article_data["html_path"] = str(html_path)
    article_data["status"] = "html_rendered"
    save_article_json(task_dir, article_data)

    # Store wechat-ready html in article dict for draft step
    article["_html_content"] = wrap_for_wechat(html_content)
    article["_task_dir"] = task_dir
    article["_article_data"] = article_data

    elapsed = time.time() - start
    logger.info(f"文章生成完成，耗时 {elapsed:.1f}s")
    return article, task_dir


@app.command()
def generate(
    topic: str = typer.Option(..., "--topic", "-t", help="文章主题"),
    style: str = typer.Option("style/wechat-article-style.md", "--style", help="写作风格文件路径"),
    log_level: str = typer.Option("INFO", "--log-level", help="日志级别"),
):
    """生成文章但不创建公众号草稿（只生成本地文件）。"""
    _setup_logging(log_level)
    try:
        article, task_dir = _run_generate(topic, style)
        article_data = article["_article_data"]

        typer.echo("\n" + "=" * 60)
        typer.echo(f"✅ 文章生成成功")
        typer.echo(f"   标题: {article.get('title')}")
        typer.echo(f"   摘要: {article.get('digest', '')[:60]}...")
        typer.echo(f"   Markdown: {article_data['markdown_path']}")
        typer.echo(f"   HTML:     {article_data['html_path']}")
        typer.echo("=" * 60 + "\n")
    except Exception as e:
        logger.error(f"生成失败: {e}")
        raise typer.Exit(1)


@app.command()
def draft(
    topic: str = typer.Option(..., "--topic", "-t", help="文章主题"),
    cover: str = typer.Option(None, "--cover", "-c", help="封面图路径"),
    style: str = typer.Option("style/wechat-article-style.md", "--style", help="写作风格文件路径"),
    log_level: str = typer.Option("INFO", "--log-level", help="日志级别"),
):
    """生成文章并创建公众号草稿。"""
    _setup_logging(log_level)
    from config.settings import get_settings
    from images.image_uploader import prepare_and_upload_cover
    from wechat.draft import create_draft
    from storage.task import save_upload_result, save_draft_result, update_status

    try:
        s = get_settings()
        cover_path = cover or s.DEFAULT_COVER_PATH

        article, task_dir = _run_generate(topic, style)
        article_data = article["_article_data"]

        # Upload cover
        logger.info(f"上传封面图: {cover_path}")
        thumb_media_id = prepare_and_upload_cover(cover_path, str(task_dir))
        save_upload_result(task_dir, {"thumb_media_id": thumb_media_id, "cover_path": cover_path})
        update_status(task_dir, "image_uploaded")

        article_data["cover_path"] = cover_path
        article_data["status"] = "image_uploaded"

        # Create draft
        media_id = create_draft(
            title=article.get("title", ""),
            author=article.get("author", s.DEFAULT_AUTHOR),
            digest=article.get("digest", ""),
            content_html=article["_html_content"],
            thumb_media_id=thumb_media_id,
            source_url=s.DEFAULT_SOURCE_URL,
        )

        article_data["wechat_draft_media_id"] = media_id
        article_data["status"] = "draft_created"
        save_draft_result(task_dir, {"media_id": media_id, "created_at": datetime.now().isoformat()})

        from storage.task import save_article_json
        save_article_json(task_dir, article_data)
        update_status(task_dir, "draft_created")

        typer.echo("\n" + "=" * 60)
        typer.echo(f"✅ 草稿创建成功")
        typer.echo(f"   标题:        {article.get('title')}")
        typer.echo(f"   草稿 media_id: {media_id}")
        typer.echo(f"   Markdown:    {article_data['markdown_path']}")
        typer.echo(f"   HTML:        {article_data['html_path']}")
        typer.echo(f"   封面图:      {cover_path}")
        typer.echo(f"   任务目录:    {task_dir}")
        typer.echo("=" * 60 + "\n")
        typer.echo("请前往公众号后台草稿箱审核后发布。")

    except Exception as e:
        logger.error(f"草稿创建失败: {e}")
        raise typer.Exit(1)


@app.command(name="draft-from-html")
def draft_from_html(
    title: str = typer.Option(..., "--title", help="文章标题"),
    html: str = typer.Option(..., "--html", help="HTML 文件路径"),
    cover: str = typer.Option(..., "--cover", help="封面图路径"),
    author: str = typer.Option(None, "--author", help="作者"),
    digest: str = typer.Option("", "--digest", help="摘要"),
    log_level: str = typer.Option("INFO", "--log-level"),
):
    """使用已有 HTML 文件创建公众号草稿。"""
    _setup_logging(log_level)
    from config.settings import get_settings
    from images.image_uploader import prepare_and_upload_cover
    from wechat.draft import create_draft
    from storage.task import create_task_dir

    try:
        s = get_settings()
        html_path = Path(html)
        if not html_path.exists():
            raise FileNotFoundError(f"HTML 文件不存在: {html}")
        html_content = html_path.read_text(encoding="utf-8")

        task_dir = create_task_dir(title)
        thumb_media_id = prepare_and_upload_cover(cover, str(task_dir))

        from render.wechat_html_template import wrap_for_wechat
        media_id = create_draft(
            title=title,
            author=author or s.DEFAULT_AUTHOR,
            digest=digest,
            content_html=wrap_for_wechat(html_content),
            thumb_media_id=thumb_media_id,
            source_url=s.DEFAULT_SOURCE_URL,
        )

        typer.echo(f"\n✅ 草稿创建成功，media_id: {media_id}")
    except Exception as e:
        logger.error(f"失败: {e}")
        raise typer.Exit(1)


@app.command(name="test-wechat")
def test_wechat(
    log_level: str = typer.Option("INFO", "--log-level"),
):
    """测试微信 API 配置是否正常。"""
    _setup_logging(log_level)
    try:
        from config.settings import get_settings
        from wechat.token import get_access_token

        s = get_settings()
        typer.echo(f"AppID:    {s.WECHAT_APP_ID}")
        typer.echo(f"AppSecret: {'*' * 8}{s.WECHAT_APP_SECRET[-4:] if len(s.WECHAT_APP_SECRET) > 4 else '****'}")

        token = get_access_token()
        typer.echo(f"\n✅ access_token 获取成功")
        typer.echo(f"   Token (前20字符): {token[:20]}...")
        typer.echo(f"   公众号 API 配置正常")
    except Exception as e:
        logger.error(f"配置测试失败: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
