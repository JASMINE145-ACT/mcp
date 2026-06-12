"""
微信公众号 MCP Server

Claude 负责生成文章内容，本 server 提供微信操作工具：

内容搜索：
- wechat_research          用 Exa 搜索资料，返回结构化摘要供写稿使用
- wechat_tavily_search     用 Tavily 搜索最新资料，返回标题+URL+摘要，适合写稿前收集素材
- wechat_fetch_url         用 Scrapling 抓取指定 URL 的正文内容，无需 API key

图片处理：
- wechat_search_cover_image  根据关键词搜索封面图，返回 URL 列表
- wechat_download_image      从 URL 下载图片到本地，返回本地路径
- wechat_upload_body_image   搜索正文配图并上传至微信 CDN，返回可嵌入的 ![](url) 片段
- wechat_upload_local_image  将本地图片上传至微信 CDN，返回 URL 和 Markdown 片段

排版渲染：
- wechat_list_templates    列出所有排版模板及说明
- wechat_render_markdown   Markdown → 微信兼容 HTML

内容检查：
- wechat_validate_content  发布前检查正文质量（长度、AI口吻、占位符等）

微信 API：
- wechat_test_connection   测试公众号 API 是否正常
- wechat_upload_cover      上传封面图，返回 thumb_media_id
- wechat_create_draft      创建草稿，返回 media_id
- wechat_list_drafts       获取草稿箱列表
- wechat_update_draft      更新已有草稿
- wechat_delete_draft      删除草稿
- wechat_publish           发布草稿到公众号（需 ENABLE_AUTO_PUBLISH=true）

一键流程：
- wechat_full_pipeline     一步到位：markdown + 封面图 → 草稿
"""

import sys
import json
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path.cwd() / ".env", override=False)

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

server = Server("wechat-publisher")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="wechat_test_connection",
            description="测试微信公众号 API 配置是否正常，获取 access_token 验证连通性。",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="wechat_health_check",
            description=(
                "检查 MCP 运行环境是否完整：必要环境变量、可选搜索/图片 API、默认封面、"
                "关键 Python 依赖。默认不请求微信网络接口；如需连通性检查，传入 check_wechat=true。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "check_wechat": {
                        "type": "boolean",
                        "description": "是否实际请求微信 access_token 接口，默认 false",
                        "default": False,
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="wechat_download_image",
            description=(
                "从指定 URL 下载图片到本地 storage/images/ 目录，返回本地文件路径。"
                "Claude 搜索到合适的封面图或正文图后，用此工具下载再传给其他工具使用。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "图片的直链 URL"},
                    "filename": {
                        "type": "string",
                        "description": "保存的文件名，如 cover.jpg（不填则自动生成）",
                    },
                },
                "required": ["url"],
            },
        ),
        types.Tool(
            name="wechat_render_markdown",
            description=(
                "将 Markdown 正文转换为微信公众号兼容的 HTML（inline style，无外部 CSS）。"
                "可通过 template 参数选择排版风格：A=蓝色商业分析（默认），B=蓝色财经科普，"
                "C=紫色新闻资讯，D=钢蓝深度评论。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "markdown": {"type": "string", "description": "Markdown 格式的文章正文"},
                    "template": {
                        "type": "string",
                        "enum": ["A", "B", "C", "D"],
                        "description": "排版模板：A=蓝色商业分析，B=蓝色财经科普，C=紫色新闻资讯，D=钢蓝深度评论。默认 A",
                        "default": "A",
                    },
                },
                "required": ["markdown"],
            },
        ),
        types.Tool(
            name="wechat_upload_cover",
            description="上传封面图到微信公众号素材库，返回 thumb_media_id（创建草稿时需要）。",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "封面图本地路径（jpg/png，建议 900x500 以上）",
                    },
                },
                "required": ["image_path"],
            },
        ),
        types.Tool(
            name="wechat_create_draft",
            description="将文章创建为微信公众号草稿，返回 media_id。需要先调用 wechat_upload_cover 获取 thumb_media_id。",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "文章标题"},
                    "content_html": {"type": "string", "description": "文章 HTML 正文（通过 wechat_render_markdown 生成）"},
                    "thumb_media_id": {"type": "string", "description": "封面图素材 ID（通过 wechat_upload_cover 获取）"},
                    "digest": {"type": "string", "description": "文章摘要，不超过 120 字"},
                    "author": {"type": "string", "description": "作者名，默认 2AIBot"},
                    "source_url": {"type": "string", "description": "原文链接（可选）"},
                },
                "required": ["title", "content_html", "thumb_media_id"],
            },
        ),
        types.Tool(
            name="wechat_tavily_search",
            description=(
                "用 Tavily 搜索指定主题的最新资料，返回标题、URL、正文摘要。"
                "专为 AI 写稿场景优化，内容质量高于普通搜索。写稿前调用此工具收集素材。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，支持中英文",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "返回结果数量，默认 5，最多 10",
                        "default": 5,
                    },
                    "search_depth": {
                        "type": "string",
                        "enum": ["basic", "advanced"],
                        "description": "搜索深度：basic（快）或 advanced（内容更全），默认 basic",
                        "default": "basic",
                    },
                    "include_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "只搜索这些域名，如 ['anthropic.com', 'openai.com']，可选",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="wechat_fetch_url",
            description=(
                "用 Scrapling 抓取指定 URL 的正文内容，无需 API key。"
                "适合抓取新闻页、博客、官方公告等，返回标题和清理后的正文文本。"
                "普通页面用轻量 Fetcher；遇到反爬/Cloudflare 可开启 stealth 模式。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "要抓取的网页 URL",
                    },
                    "stealth": {
                        "type": "boolean",
                        "description": "是否启用隐身模式（绕过 Cloudflare 等反爬，速度较慢），默认 false",
                        "default": False,
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "返回正文的最大字符数，默认 5000",
                        "default": 5000,
                    },
                },
                "required": ["url"],
            },
        ),
        types.Tool(
            name="wechat_research",
            description=(
                "用 Exa 搜索指定主题的最新资料，返回结构化摘要（标题、URL、发布日期、正文摘录）。"
                "写文章前先调用此工具收集素材，再调用 wechat_full_pipeline 生成草稿。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索查询，描述你想找的内容。用英文效果更好，例如 'Anthropic Claude 4 safety research 2025'",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "返回结果数量，默认 5，最多 10",
                        "default": 5,
                    },
                    "max_chars_per_result": {
                        "type": "integer",
                        "description": "每条结果的正文字符数限制，默认 2000",
                        "default": 2000,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="wechat_search_cover_image",
            description=(
                "根据关键词搜索与文章高度相关的封面图，返回图片 URL 列表供选择。"
                "写文章时必须先调用此工具，根据文章主题提取精准关键词搜索（英文效果更好）。"
                "例如写 Claude AI 相关文章传入 'Claude Anthropic AI assistant'，"
                "写 GPT 相关传入 'OpenAI GPT language model'，写区块链传入 'blockchain crypto technology'。"
                "选好图片后用 wechat_download_image 下载，再传给 wechat_full_pipeline。"
                "需要在 .env 配置 UNSPLASH_ACCESS_KEY 或 PEXELS_API_KEY。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，英文效果更好，如 'Claude Anthropic AI language model'",
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "返回图片数量，默认 5，最多 10",
                        "default": 5,
                    },
                    "orientation": {
                        "type": "string",
                        "enum": ["landscape", "portrait", "squarish"],
                        "description": "图片方向，封面图推荐 landscape（横向），默认 landscape",
                        "default": "landscape",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="wechat_upload_body_image",
            description=(
                "根据关键词搜索与文章段落内容高度相关的正文配图，自动下载并上传到微信 CDN，"
                "返回可直接嵌入 Markdown 的 ![caption](wechat_url) 片段。"
                "写文章时每个需要配图的 H2 段落调用一次，关键词用英文效果更好。"
                "例如写 AI 芯片段落传入 'AI GPU chip semiconductor technology'，"
                "写融资段落传入 'venture capital investment startup funding'。"
                "需要在 .env 配置 UNSPLASH_ACCESS_KEY 或 PEXELS_API_KEY。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "图片搜索关键词，英文效果更好，如 'AI language model neural network'",
                    },
                    "caption": {
                        "type": "string",
                        "description": "图片说明文字，会显示在图片下方，如 'Kimi K2 模型架构示意'",
                    },
                    "select_index": {
                        "type": "integer",
                        "description": "选择第几张结果（0-based），默认 0（第一张/最相关）",
                        "default": 0,
                    },
                    "orientation": {
                        "type": "string",
                        "enum": ["landscape", "portrait", "squarish"],
                        "description": "图片方向，正文图推荐 landscape，默认 landscape",
                        "default": "landscape",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="wechat_list_templates",
            description="列出所有可用排版模板及使用场景，帮助选择合适风格。",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        types.Tool(
            name="wechat_validate_content",
            description=(
                "发布前检查文章内容质量：检测正文长度不足、AI 口吻表达、"
                "未填充的占位符、重复段落、未闭合代码块等问题。建议每次发布前调用。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "文章标题"},
                    "digest": {"type": "string", "description": "文章摘要（可选）"},
                    "markdown": {"type": "string", "description": "文章正文 Markdown"},
                },
                "required": ["title", "markdown"],
            },
        ),
        types.Tool(
            name="wechat_upload_local_image",
            description=(
                "将本地已有图片上传至微信 CDN（正文图片接口），返回微信图片 URL 和可嵌入的 Markdown 片段。"
                "适合已有本地图片时使用，无需搜索。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "本地图片路径（jpg/png）"},
                    "caption": {"type": "string", "description": "图片说明文字，显示在图片下方（可选）"},
                },
                "required": ["image_path"],
            },
        ),
        types.Tool(
            name="wechat_list_drafts",
            description="获取公众号草稿箱列表，返回草稿 media_id、标题、摘要、更新时间。",
            inputSchema={
                "type": "object",
                "properties": {
                    "offset": {
                        "type": "integer",
                        "description": "从第几条开始，默认 0",
                        "default": 0,
                    },
                    "count": {
                        "type": "integer",
                        "description": "返回条数，默认 10，最多 20",
                        "default": 10,
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="wechat_update_draft",
            description=(
                "更新草稿箱中已有草稿的内容（标题、正文、封面等）。"
                "需要先通过 wechat_list_drafts 获取 media_id。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "media_id": {"type": "string", "description": "要更新的草稿 media_id"},
                    "title": {"type": "string", "description": "新标题（不超过 64 字符）"},
                    "content_html": {
                        "type": "string",
                        "description": "新 HTML 正文（通过 wechat_render_markdown 生成）",
                    },
                    "thumb_media_id": {
                        "type": "string",
                        "description": "封面素材 ID（通过 wechat_upload_cover 获取）",
                    },
                    "digest": {"type": "string", "description": "新摘要（不超过 120 字）"},
                    "author": {"type": "string", "description": "作者名（不超过 8 字）"},
                    "source_url": {"type": "string", "description": "原文链接（可选）"},
                },
                "required": ["media_id", "title", "content_html", "thumb_media_id"],
            },
        ),
        types.Tool(
            name="wechat_delete_draft",
            description="删除草稿箱中的草稿。此操作不可撤销，请谨慎使用。",
            inputSchema={
                "type": "object",
                "properties": {
                    "media_id": {"type": "string", "description": "要删除的草稿 media_id"},
                },
                "required": ["media_id"],
            },
        ),
        types.Tool(
            name="wechat_publish",
            description=(
                "将草稿发布到公众号，推送给所有订阅者。此操作不可撤销！"
                "需要在 .env 中设置 ENABLE_AUTO_PUBLISH=true 才能使用。"
                "建议发布前先用 wechat_validate_content 检查内容质量。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "media_id": {"type": "string", "description": "要发布的草稿 media_id"},
                },
                "required": ["media_id"],
            },
        ),
        types.Tool(
            name="wechat_get_publish_status",
            description="根据 wechat_publish 返回的 publish_id 查询发布任务状态和失败原因。",
            inputSchema={
                "type": "object",
                "properties": {
                    "publish_id": {"type": "string", "description": "发布任务 ID"},
                },
                "required": ["publish_id"],
            },
        ),
        types.Tool(
            name="wechat_list_local_tasks",
            description=(
                "查询本地 storage/drafts 中保存的文章任务，返回标题、状态、草稿 media_id、"
                "Markdown/HTML 路径。用于失败恢复、历史草稿审计和继续编辑。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "返回最近任务数量，默认 10，最多 50", "default": 10},
                    "status": {"type": "string", "description": "按本地任务状态过滤，可选"},
                },
                "required": [],
            },
        ),
        types.Tool(
            name="wechat_full_pipeline",
            description=(
                "一步到位：接收文章 Markdown + 元数据 + 封面图路径，"
                "自动完成渲染→上传封面→创建草稿，返回草稿 media_id 和本地保存路径。"
                "通过 template 选择排版风格：A=蓝色商业分析（默认），B=蓝色财经科普，"
                "C=紫色新闻资讯，D=钢蓝深度评论。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "文章标题"},
                    "markdown": {"type": "string", "description": "文章 Markdown 正文"},
                    "digest": {"type": "string", "description": "摘要（不超过 120 字）"},
                    "author": {"type": "string", "description": "作者名，默认 2AIBot"},
                    "cover_path": {
                        "type": "string",
                        "description": "封面图本地路径，不填则使用默认封面",
                    },
                    "source_url": {"type": "string", "description": "原文链接（可选）"},
                    "topic_slug": {"type": "string", "description": "主题关键词，用于命名存储目录"},
                    "template": {
                        "type": "string",
                        "enum": ["A", "B", "C", "D"],
                        "description": "排版模板：A=蓝色商业分析，B=蓝色财经科普，C=紫色新闻资讯，D=钢蓝深度评论。默认 A",
                        "default": "A",
                    },
                },
                "required": ["title", "markdown", "digest"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        if name == "wechat_test_connection":
            return await _test_connection()
        elif name == "wechat_health_check":
            return await _health_check(arguments)
        elif name == "wechat_download_image":
            return await _download_image(arguments)
        elif name == "wechat_render_markdown":
            return await _render_markdown(arguments)
        elif name == "wechat_upload_cover":
            return await _upload_cover(arguments)
        elif name == "wechat_create_draft":
            return await _create_draft(arguments)
        elif name == "wechat_tavily_search":
            return await _tavily_search(arguments)
        elif name == "wechat_fetch_url":
            return await _fetch_url(arguments)
        elif name == "wechat_research":
            return await _research(arguments)
        elif name == "wechat_search_cover_image":
            return await _search_cover_image(arguments)
        elif name == "wechat_upload_body_image":
            return await _upload_body_image(arguments)
        elif name == "wechat_full_pipeline":
            return await _full_pipeline(arguments)
        elif name == "wechat_list_templates":
            return await _list_templates()
        elif name == "wechat_validate_content":
            return await _validate_content(arguments)
        elif name == "wechat_upload_local_image":
            return await _upload_local_image(arguments)
        elif name == "wechat_list_drafts":
            return await _list_drafts(arguments)
        elif name == "wechat_update_draft":
            return await _update_draft(arguments)
        elif name == "wechat_delete_draft":
            return await _delete_draft(arguments)
        elif name == "wechat_publish":
            return await _publish(arguments)
        elif name == "wechat_get_publish_status":
            return await _get_publish_status(arguments)
        elif name == "wechat_list_local_tasks":
            return await _list_local_tasks(arguments)
        else:
            return [types.TextContent(type="text", text=f"未知工具: {name}")]
    except Exception as e:
        return [types.TextContent(type="text", text=f"错误: {e}")]


async def _test_connection() -> list[types.TextContent]:
    from wechat.token import get_access_token
    token = get_access_token()
    return [types.TextContent(
        type="text",
        text=json.dumps({
            "status": "ok",
            "message": "access_token 获取成功，公众号 API 配置正常",
            "token_preview": token[:20] + "...",
        }, ensure_ascii=False),
    )]


async def _health_check(args: dict) -> list[types.TextContent]:
    import importlib.util

    base_dir = Path(__file__).parent
    check_wechat = bool(args.get("check_wechat", False))

    env_checks = []
    for name, required in [
        ("WECHAT_APP_ID", True),
        ("WECHAT_APP_SECRET", True),
        ("OPENAI_API_KEY", False),
        ("EXA_API_KEY", False),
        ("TAVILY_API_KEY", False),
        ("UNSPLASH_ACCESS_KEY", False),
        ("PEXELS_API_KEY", False),
        ("DEFAULT_COVER_PATH", False),
    ]:
        value = os.environ.get(name, "").strip()
        env_checks.append({
            "name": name,
            "required": required,
            "configured": bool(value),
        })

    default_cover = os.environ.get("DEFAULT_COVER_PATH", "").strip()
    cover_path = Path(default_cover) if default_cover else None
    if cover_path and not cover_path.is_absolute():
        cover_path = base_dir / cover_path

    dependencies = []
    for module_name in [
        "mcp",
        "requests",
        "dotenv",
        "markdown",
        "bs4",
        "PIL",
        "tenacity",
        "loguru",
        "openai",
        "typer",
        "exa_py",
        "tavily",
        "scrapling",
    ]:
        dependencies.append({
            "module": module_name,
            "installed": importlib.util.find_spec(module_name) is not None,
        })

    issues = []
    for item in env_checks:
        if item["required"] and not item["configured"]:
            issues.append(f"{item['name']} 未配置")
    if default_cover and cover_path and not cover_path.exists():
        issues.append(f"DEFAULT_COVER_PATH 文件不存在: {cover_path}")
    if not any(os.environ.get(k, "").strip() for k in ("UNSPLASH_ACCESS_KEY", "PEXELS_API_KEY")):
        issues.append("未配置图片搜索 API，wechat_search_cover_image/wechat_upload_body_image 将不可用")
    if not any(os.environ.get(k, "").strip() for k in ("EXA_API_KEY", "TAVILY_API_KEY")):
        issues.append("未配置资料搜索 API，wechat_research/wechat_tavily_search 将不可用")

    wechat = {"checked": False}
    if check_wechat:
        try:
            from wechat.token import get_access_token
            token = get_access_token()
            wechat = {"checked": True, "status": "ok", "token_preview": token[:20] + "..."}
        except Exception as e:
            wechat = {"checked": True, "status": "error", "message": str(e)}
            issues.append(f"微信连通性检查失败: {e}")

    return [types.TextContent(type="text", text=json.dumps({
        "status": "ok" if not issues else "needs_attention",
        "base_dir": str(base_dir),
        "env": env_checks,
        "default_cover": {
            "configured": bool(default_cover),
            "path": str(cover_path) if cover_path else "",
            "exists": bool(cover_path and cover_path.exists()),
        },
        "dependencies": dependencies,
        "wechat": wechat,
        "issues": issues,
    }, ensure_ascii=False))]


async def _download_image(args: dict) -> list[types.TextContent]:
    import requests
    import uuid
    from pathlib import Path

    url = args["url"]
    filename = args.get("filename") or f"{uuid.uuid4().hex[:8]}.jpg"

    images_dir = Path(__file__).parent / "storage" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    dest = images_dir / filename

    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=30, stream=True)
    resp.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    size_kb = dest.stat().st_size // 1024
    return [types.TextContent(
        type="text",
        text=json.dumps({
            "status": "ok",
            "local_path": str(dest),
            "filename": filename,
            "size_kb": size_kb,
        }, ensure_ascii=False),
    )]


async def _render_markdown(args: dict) -> list[types.TextContent]:
    from render.markdown_to_html import convert_markdown_to_wechat_html
    from render.wechat_html_template import wrap_for_wechat
    md = args["markdown"]
    template = args.get("template", "A")
    html = convert_markdown_to_wechat_html(md, template=template)
    wechat_html = wrap_for_wechat(html)
    return [types.TextContent(
        type="text",
        text=json.dumps({
            "status": "ok",
            "template": template,
            "html": wechat_html,
            "length": len(wechat_html),
        }, ensure_ascii=False),
    )]


async def _upload_cover(args: dict) -> list[types.TextContent]:
    from images.cover_processor import process_cover_image
    from wechat.material import upload_permanent_image
    import tempfile, os

    image_path = args["image_path"]
    with tempfile.TemporaryDirectory() as tmp:
        processed = process_cover_image(image_path, os.path.join(tmp, "cover.jpg"))
        thumb_media_id = upload_permanent_image(processed)

    return [types.TextContent(
        type="text",
        text=json.dumps({
            "status": "ok",
            "thumb_media_id": thumb_media_id,
            "message": "封面图上传成功",
        }, ensure_ascii=False),
    )]


async def _create_draft(args: dict) -> list[types.TextContent]:
    from wechat.draft import create_draft
    from config.settings import get_default_author, get_default_source_url

    title = args["title"]
    if len(title) > 64:
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": f"标题过长（{len(title)} 字符），微信限制最多 64 字符",
        }, ensure_ascii=False))]

    author = args.get("author") or get_default_author()
    if len(author) > 8:
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": f"作者名过长（{len(author)} 字符），微信限制最多 8 字符",
        }, ensure_ascii=False))]

    media_id = create_draft(
        title=title,
        author=author,
        digest=args.get("digest", ""),
        content_html=args["content_html"],
        thumb_media_id=args["thumb_media_id"],
        source_url=args.get("source_url") or get_default_source_url(),
    )
    return [types.TextContent(
        type="text",
        text=json.dumps({
            "status": "ok",
            "media_id": media_id,
            "message": "草稿创建成功，请前往公众号后台审核后发布",
        }, ensure_ascii=False),
    )]


async def _tavily_search(args: dict) -> list[types.TextContent]:
    import os
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": "TAVILY_API_KEY 未配置，请在 .env 文件中填写。获取地址：https://tavily.com",
        }, ensure_ascii=False))]

    from tavily import TavilyClient

    query = args["query"]
    num_results = min(int(args.get("num_results", 5)), 10)
    search_depth = args.get("search_depth", "basic")
    include_domains = args.get("include_domains", [])

    client = TavilyClient(api_key=api_key)
    kwargs = {
        "max_results": num_results,
        "search_depth": search_depth,
        "include_answer": True,
    }
    if include_domains:
        kwargs["include_domains"] = include_domains

    response = client.search(query, **kwargs)

    results = []
    for r in response.get("results", []):
        results.append({
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "score": round(r.get("score", 0), 3),
        })

    return [types.TextContent(
        type="text",
        text=json.dumps({
            "status": "ok",
            "query": query,
            "answer": response.get("answer", ""),
            "num_results": len(results),
            "results": results,
        }, ensure_ascii=False),
    )]


async def _fetch_url(args: dict) -> list[types.TextContent]:
    url = args["url"]
    stealth = args.get("stealth", False)
    max_chars = int(args.get("max_chars", 5000))

    try:
        if stealth:
            from scrapling.fetchers import StealthyFetcher
            page = StealthyFetcher.fetch(url, headless=True, network_idle=True)
        else:
            from scrapling.fetchers import Fetcher
            page = Fetcher.get(url, stealthy_headers=True, follow_redirects=True)
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": f"抓取失败：{e}",
            "url": url,
        }, ensure_ascii=False))]

    # 提取标题
    title_el = page.find("title")
    title = title_el.text.strip() if title_el else ""

    # 提取正文：忽略导航/脚本/样式等噪音标签
    text = page.get_all_text(
        ignore_tags=("script", "style", "nav", "footer", "header", "aside")
    )
    # 压缩多余空行
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "…（已截断）"

    return [types.TextContent(
        type="text",
        text=json.dumps({
            "status": "ok",
            "url": url,
            "title": title,
            "text": text,
            "char_count": len(text),
        }, ensure_ascii=False),
    )]


async def _research(args: dict) -> list[types.TextContent]:
    import os
    api_key = os.environ.get("EXA_API_KEY", "").strip()
    if not api_key:
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": "EXA_API_KEY 未配置，请在 .env 文件中填写。获取地址：https://exa.ai/api-key",
        }, ensure_ascii=False))]

    try:
        from exa_py import Exa
    except ImportError:
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": "exa-py 未安装，请运行：pip install exa-py",
        }, ensure_ascii=False))]

    query = args["query"]
    num_results = min(int(args.get("num_results", 5)), 10)
    max_chars = int(args.get("max_chars_per_result", 2000))

    exa = Exa(api_key=api_key)
    response = exa.search_and_contents(
        query,
        num_results=num_results,
        text={"max_characters": max_chars},
        use_autoprompt=True,
    )

    results = []
    for r in response.results:
        results.append({
            "title": r.title or "",
            "url": r.url or "",
            "published_date": r.published_date or "",
            "text": (r.text or "").strip(),
        })

    return [types.TextContent(
        type="text",
        text=json.dumps({
            "status": "ok",
            "query": query,
            "num_results": len(results),
            "results": results,
        }, ensure_ascii=False),
    )]


async def _search_cover_image(args: dict) -> list[types.TextContent]:
    import os
    import requests

    query = args["query"]
    num_results = min(int(args.get("num_results", 5)), 10)
    orientation = args.get("orientation", "landscape")

    unsplash_key = os.environ.get("UNSPLASH_ACCESS_KEY", "").strip()
    pexels_key = os.environ.get("PEXELS_API_KEY", "").strip()

    if unsplash_key:
        resp = requests.get(
            "https://api.unsplash.com/search/photos",
            params={
                "query": query,
                "per_page": num_results,
                "orientation": orientation,
                "order_by": "relevant",
            },
            headers={"Authorization": f"Client-ID {unsplash_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = [
            {
                "url": photo["urls"]["regular"],
                "full_url": photo["urls"]["full"],
                "thumb_url": photo["urls"]["small"],
                "description": photo.get("description") or photo.get("alt_description") or "",
                "photographer": photo["user"]["name"],
                "source": "unsplash",
            }
            for photo in data.get("results", [])
        ]
    elif pexels_key:
        resp = requests.get(
            "https://api.pexels.com/v1/search",
            params={"query": query, "per_page": num_results, "orientation": orientation},
            headers={"Authorization": pexels_key},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = [
            {
                "url": photo["src"]["large2x"],
                "full_url": photo["src"]["original"],
                "thumb_url": photo["src"]["medium"],
                "description": photo.get("alt", ""),
                "photographer": photo["photographer"],
                "source": "pexels",
            }
            for photo in data.get("photos", [])
        ]
    else:
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "error",
                "message": (
                    "图片搜索需要配置图片 API key，请在 .env 中填写以下任意一项：\n"
                    "  UNSPLASH_ACCESS_KEY — 注册地址：https://unsplash.com/developers\n"
                    "  PEXELS_API_KEY     — 注册地址：https://www.pexels.com/api/"
                ),
            }, ensure_ascii=False),
        )]

    return [types.TextContent(
        type="text",
        text=json.dumps({
            "status": "ok",
            "query": query,
            "num_results": len(results),
            "results": results,
            "tip": "选择最相关的图片，用 wechat_download_image 下载后再传给 wechat_full_pipeline",
        }, ensure_ascii=False),
    )]


def _upload_inline_images(html: str) -> str:
    """Scan HTML for external <img> src, upload each to WeChat CDN, replace src in-place.

    Skips images already hosted on WeChat CDN (mmbiz.qpic.cn / mmbiz.qlogo.cn).
    """
    import tempfile, requests, logging
    from urllib.parse import urlparse
    from bs4 import BeautifulSoup
    from wechat.material import upload_article_image

    log = logging.getLogger(__name__)
    _WECHAT_CDN_HOSTS = {"mmbiz.qpic.cn", "mmbiz.qlogo.cn"}

    soup = BeautifulSoup(html, "html.parser")
    imgs = soup.find_all("img")
    if not imgs:
        return html

    for img in imgs:
        src = img.get("src", "")
        if not src.startswith("http"):
            continue
        from urllib.parse import urlparse as _up
        if _up(src).hostname in _WECHAT_CDN_HOSTS:
            continue

        tmp_path = None
        try:
            ext = Path(urlparse(src).path).suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png"}:
                ext = ".jpg"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp_path = tmp.name
            resp = requests.get(src, headers={"User-Agent": "Mozilla/5.0"}, timeout=30, stream=True)
            resp.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)
            wechat_url = upload_article_image(tmp_path)
            img["src"] = wechat_url
            log.info(f"正文图片已上传: {src} → {wechat_url}")
        except Exception as e:
            log.warning(f"正文图片上传失败，保留原 URL: {src}，错误: {e}")
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    return str(soup)


async def _upload_body_image(args: dict) -> list[types.TextContent]:
    """Search image by query, upload to WeChat CDN, return embed-ready markdown snippet."""
    import requests, tempfile
    from urllib.parse import urlparse
    from pathlib import Path as _Path
    from wechat.material import upload_article_image

    query = args["query"]
    caption = args.get("caption", "")
    select_index = int(args.get("select_index", 0))
    orientation = args.get("orientation", "landscape")
    num_results = max(select_index + 1, 5)

    unsplash_key = os.environ.get("UNSPLASH_ACCESS_KEY", "").strip()
    pexels_key = os.environ.get("PEXELS_API_KEY", "").strip()

    if unsplash_key:
        resp = requests.get(
            "https://api.unsplash.com/search/photos",
            params={"query": query, "per_page": num_results, "orientation": orientation, "order_by": "relevant"},
            headers={"Authorization": f"Client-ID {unsplash_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        photos = resp.json().get("results", [])
        results = [
            {
                "url": p["urls"]["regular"],
                "description": p.get("description") or p.get("alt_description") or "",
                "photographer": p["user"]["name"],
            }
            for p in photos
        ]
    elif pexels_key:
        resp = requests.get(
            "https://api.pexels.com/v1/search",
            params={"query": query, "per_page": num_results, "orientation": orientation},
            headers={"Authorization": pexels_key},
            timeout=15,
        )
        resp.raise_for_status()
        photos = resp.json().get("photos", [])
        results = [
            {
                "url": p["src"]["large2x"],
                "description": p.get("alt", ""),
                "photographer": p["photographer"],
            }
            for p in photos
        ]
    else:
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": "图片搜索需要配置 UNSPLASH_ACCESS_KEY 或 PEXELS_API_KEY",
        }, ensure_ascii=False))]

    if not results:
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": f"未找到相关图片：{query}，请换用其他关键词",
        }, ensure_ascii=False))]

    if select_index >= len(results):
        select_index = 0

    selected = results[select_index]
    img_url = selected["url"]
    auto_caption = caption or selected["description"] or query

    ext = _Path(urlparse(img_url).path).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png"}:
        ext = ".jpg"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name
        dl = requests.get(img_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30, stream=True)
        dl.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in dl.iter_content(8192):
                f.write(chunk)
        wechat_url = upload_article_image(tmp_path)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    embed_md = f"![{auto_caption}]({wechat_url})"

    return [types.TextContent(
        type="text",
        text=json.dumps({
            "status": "ok",
            "wechat_url": wechat_url,
            "embed_markdown": embed_md,
            "caption": auto_caption,
            "photographer": selected["photographer"],
            "tip": "复制 embed_markdown 粘贴到文章对应段落即可",
        }, ensure_ascii=False),
    )]


async def _full_pipeline(args: dict) -> list[types.TextContent]:
    from render.markdown_to_html import convert_markdown_to_wechat_html, save_html as save_preview
    from render.wechat_html_template import wrap_for_wechat
    from images.image_uploader import prepare_and_upload_cover
    from wechat.draft import create_draft
    from storage.task import create_task_dir, save_markdown, save_html, save_upload_result, save_draft_result, save_article_json, update_status
    from config.settings import get_default_author, get_default_source_url, get_default_cover_path
    from datetime import datetime

    title = args["title"]
    if len(title) > 64:
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": f"标题过长（{len(title)} 字符），微信限制最多 64 字符",
        }, ensure_ascii=False))]

    markdown = args["markdown"]
    digest = args.get("digest", "")
    author = args.get("author") or get_default_author()
    if len(author) > 8:
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": f"作者名过长（{len(author)} 字符），微信限制最多 8 字符",
        }, ensure_ascii=False))]

    cover_path = args.get("cover_path") or get_default_cover_path()
    source_url = args.get("source_url") or get_default_source_url()
    topic_slug = args.get("topic_slug") or title[:20]

    if not cover_path:
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": "未指定封面图路径，请提供 cover_path 或在 .env 中设置 DEFAULT_COVER_PATH",
        }, ensure_ascii=False))]

    template = args.get("template", "A")

    # Step 1: Render HTML (local, no partial needed)
    html_content = convert_markdown_to_wechat_html(markdown, template=template)
    html_content = _upload_inline_images(html_content)
    wechat_html = wrap_for_wechat(html_content)

    # Step 2: Save to task dir (local, no partial needed)
    task_dir = create_task_dir(topic_slug)
    md_path = save_markdown(task_dir, markdown, topic_slug)
    html_path = save_html(task_dir, html_content, topic_slug)
    save_preview(html_content, str(html_path))

    article_data = {
        "topic": topic_slug,
        "title": title,
        "digest": digest,
        "author": author,
        "markdown_path": str(md_path),
        "html_path": str(html_path),
        "cover_path": cover_path,
        "wechat_draft_media_id": "",
        "created_at": datetime.now().isoformat(),
        "status": "html_rendered",
    }
    save_article_json(task_dir, article_data)

    partial = {
        "title": title,
        "task_dir": str(task_dir),
        "markdown_path": str(md_path),
        "html_path": str(html_path),
    }

    # Step 3: Upload cover (may fail — return partial so caller can retry with wechat_upload_cover)
    try:
        thumb_media_id = prepare_and_upload_cover(cover_path, str(task_dir))
        save_upload_result(task_dir, {"thumb_media_id": thumb_media_id})
        update_status(task_dir, "image_uploaded")
        partial["thumb_media_id"] = thumb_media_id
    except Exception as e:
        partial["step_failed"] = "upload_cover"
        partial["error"] = str(e)
        partial["hint"] = "HTML 和 Markdown 已保存到 task_dir。修复封面问题后可单独调用 wechat_upload_cover + wechat_create_draft 完成。"
        return [types.TextContent(type="text", text=json.dumps(
            {"status": "partial", **partial}, ensure_ascii=False
        ))]

    # Step 4: Create draft (may fail — return partial so caller can retry with wechat_create_draft)
    try:
        media_id = create_draft(
            title=title,
            author=author,
            digest=digest,
            content_html=wechat_html,
            thumb_media_id=thumb_media_id,
            source_url=source_url,
        )
        article_data["wechat_draft_media_id"] = media_id
        article_data["status"] = "draft_created"
        save_article_json(task_dir, article_data)
        save_draft_result(task_dir, {"media_id": media_id, "created_at": datetime.now().isoformat()})
        update_status(task_dir, "draft_created")
    except Exception as e:
        partial["step_failed"] = "create_draft"
        partial["error"] = str(e)
        partial["hint"] = f"封面已上传（thumb_media_id={thumb_media_id}）。修复问题后可单独调用 wechat_create_draft 完成。"
        return [types.TextContent(type="text", text=json.dumps(
            {"status": "partial", **partial}, ensure_ascii=False
        ))]

    return [types.TextContent(
        type="text",
        text=json.dumps({
            "status": "ok",
            "media_id": media_id,
            **partial,
            "message": "草稿创建成功！请前往公众号后台草稿箱审核后发布。",
        }, ensure_ascii=False),
    )]


async def _list_templates() -> list[types.TextContent]:
    templates = [
        {
            "id": "A",
            "name": "蓝色商业分析",
            "accent_color": "rgb(0,102,204)",
            "features": "首行缩进，H2 蓝色左边框，数据对比卡片布局",
            "best_for": "商业分析、科技深度、行业报告",
        },
        {
            "id": "B",
            "name": "蓝色财经科普",
            "accent_color": "rgb(47,105,255)",
            "features": "清爽亮蓝，段落间距宽松，无首行缩进",
            "best_for": "财经科普、知识解读、行业入门",
        },
        {
            "id": "C",
            "name": "紫色新闻资讯",
            "accent_color": "紫色系",
            "features": "活泼紫色，适合快节奏阅读",
            "best_for": "新闻快讯、热点事件、资讯汇总",
        },
        {
            "id": "D",
            "name": "钢蓝深度评论",
            "accent_color": "深钢蓝",
            "features": "沉稳深色，字重偏粗，适合长文阅读",
            "best_for": "深度评论、观点文章、学术科普",
        },
    ]
    return [types.TextContent(type="text", text=json.dumps({
        "status": "ok",
        "templates": templates,
        "tip": "在 wechat_render_markdown 或 wechat_full_pipeline 的 template 参数中填入 id（A/B/C/D）",
    }, ensure_ascii=False))]


async def _validate_content(args: dict) -> list[types.TextContent]:
    from ai.content_checker import check_article
    result = check_article({
        "title": args.get("title", ""),
        "digest": args.get("digest", ""),
        "markdown": args.get("markdown", ""),
    })
    return [types.TextContent(type="text", text=json.dumps({
        "status": "ok",
        "passed": result.passed,
        "errors": result.errors,
        "warnings": result.warnings,
        "summary": "✅ 内容检查通过" if result.passed else f"❌ 发现 {len(result.errors)} 个错误，{len(result.warnings)} 个警告",
    }, ensure_ascii=False))]


async def _upload_local_image(args: dict) -> list[types.TextContent]:
    from wechat.material import upload_article_image
    image_path = args["image_path"]
    caption = args.get("caption", "")
    wechat_url = upload_article_image(image_path)
    embed_md = f"![{caption}]({wechat_url})"
    return [types.TextContent(type="text", text=json.dumps({
        "status": "ok",
        "wechat_url": wechat_url,
        "embed_markdown": embed_md,
        "tip": "复制 embed_markdown 粘贴到文章对应位置",
    }, ensure_ascii=False))]


async def _list_drafts(args: dict) -> list[types.TextContent]:
    from wechat.draft import list_drafts
    data = list_drafts(
        offset=int(args.get("offset", 0)),
        count=int(args.get("count", 10)),
    )
    items = data.get("item", [])
    drafts = []
    for item in items:
        content = item.get("content", {})
        news_item = (content.get("news_item") or [{}])[0]
        drafts.append({
            "media_id": item.get("media_id", ""),
            "title": news_item.get("title", ""),
            "digest": news_item.get("digest", ""),
            "update_time": item.get("update_time", 0),
        })
    return [types.TextContent(type="text", text=json.dumps({
        "status": "ok",
        "total_count": data.get("total_count", 0),
        "item_count": data.get("item_count", 0),
        "drafts": drafts,
    }, ensure_ascii=False))]


async def _update_draft(args: dict) -> list[types.TextContent]:
    from wechat.draft import update_draft
    from config.settings import get_default_author

    title = args["title"]
    if len(title) > 64:
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": f"标题过长（{len(title)} 字符），微信限制最多 64 字符",
        }, ensure_ascii=False))]

    author = args.get("author") or get_default_author()
    if len(author) > 8:
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": f"作者名过长（{len(author)} 字符），微信限制最多 8 字符",
        }, ensure_ascii=False))]

    update_draft(
        media_id=args["media_id"],
        title=title,
        content_html=args["content_html"],
        thumb_media_id=args["thumb_media_id"],
        author=author,
        digest=args.get("digest", ""),
        source_url=args.get("source_url", ""),
    )
    return [types.TextContent(type="text", text=json.dumps({
        "status": "ok",
        "media_id": args["media_id"],
        "message": "草稿更新成功",
    }, ensure_ascii=False))]


async def _delete_draft(args: dict) -> list[types.TextContent]:
    from wechat.draft import delete_draft
    delete_draft(args["media_id"])
    return [types.TextContent(type="text", text=json.dumps({
        "status": "ok",
        "message": f"草稿已删除: {args['media_id']}",
    }, ensure_ascii=False))]


async def _publish(args: dict) -> list[types.TextContent]:
    from wechat.publish import publish_from_mcp
    result = publish_from_mcp(args["media_id"])
    return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


async def _get_publish_status(args: dict) -> list[types.TextContent]:
    from wechat.publish import get_publish_status
    result = get_publish_status(args["publish_id"])
    return [types.TextContent(type="text", text=json.dumps({
        "status": "ok",
        "publish_id": args["publish_id"],
        "result": result,
    }, ensure_ascii=False))]


async def _list_local_tasks(args: dict) -> list[types.TextContent]:
    from storage.task import DRAFTS_DIR

    limit = min(max(int(args.get("limit", 10)), 1), 50)
    status_filter = (args.get("status") or "").strip()

    tasks = []
    if DRAFTS_DIR.exists():
        for task_dir in sorted(DRAFTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not task_dir.is_dir():
                continue
            article_path = task_dir / "article.json"
            data = {}
            if article_path.exists():
                try:
                    data = json.loads(article_path.read_text(encoding="utf-8"))
                except Exception as e:
                    data = {"status": "metadata_error", "error": str(e)}
            task_status = data.get("status", "unknown")
            if status_filter and task_status != status_filter:
                continue
            tasks.append({
                "task_dir": str(task_dir),
                "name": task_dir.name,
                "title": data.get("title", ""),
                "topic": data.get("topic", ""),
                "status": task_status,
                "created_at": data.get("created_at", ""),
                "wechat_draft_media_id": data.get("wechat_draft_media_id", ""),
                "markdown_path": data.get("markdown_path", str(task_dir / "article.md") if (task_dir / "article.md").exists() else ""),
                "html_path": data.get("html_path", str(task_dir / "article.html") if (task_dir / "article.html").exists() else ""),
                "cover_path": data.get("cover_path", ""),
                "has_upload_result": (task_dir / "upload_result.json").exists(),
                "has_draft_result": (task_dir / "draft_result.json").exists(),
            })
            if len(tasks) >= limit:
                break

    return [types.TextContent(type="text", text=json.dumps({
        "status": "ok",
        "drafts_dir": str(DRAFTS_DIR),
        "count": len(tasks),
        "tasks": tasks,
    }, ensure_ascii=False))]


async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
