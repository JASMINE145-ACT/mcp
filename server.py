"""
微信公众号 MCP Server

Claude 负责生成文章内容，本 server 提供微信操作工具：

内容搜索：
- wechat_research          用 Exa 搜索资料，返回结构化摘要供写稿使用
- wechat_tavily_search     用 Tavily 搜索最新资料，返回标题+URL+摘要，适合写稿前收集素材
- wechat_fetch_url         用 Scrapling 抓取指定 URL 的正文内容，无需 API key
- wechat_deep_research     并行多源搜索（Tavily+Exa）→去重→深读高分来源，返回原始资料；
                           拆子问题和综合报告由调用方模型自己做，本工具不调用任何 LLM

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

任务恢复：
- wechat_resume_task       从失败位置继续执行（输入 task_dir）
- wechat_preview_task      草稿发布前预览：标题/字数/封面/来源/风险点

内容审核：
- wechat_audit_before_publish  发布前综合审核（标题党/事实声明/来源/摘要/封面）
- wechat_save_sources          保存文章来源引用列表到 task_dir/sources.json

账号管理：
- wechat_list_profiles     列出可用的公众号账号 profile

选题库：
- wechat_add_topic         添加选题到待审队列
- wechat_approve_topic     审批选题（pending → approved）
- wechat_list_topics       查看选题队列（pending/approved/all）
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

SERVER_INSTRUCTIONS = """本 server 面向"写一篇公众号文章"这类任务，推荐流程：

0. 【方向澄清，先于一切，参照 brainstorming 的对话风格】如果用户只给了一个笼统
   方向（没说清写作角度、侧重点、目标读者、篇幅或结论倾向），不要直接开始调研
   或写作，也不要一次性抛一堆问题。按这个节奏走：
   a. 一次只问一个问题；优先给选择题而不是开放式问题（"这几个角度选一个：
      A/B/C"比"你想从什么角度写"更容易回答，选择困难时才追加开放式追问）
   b. 依次澄清：核心角度/侧重点是什么？目标读者是谁？想传达的结论倾向是什么？
      篇幅预期？有没有想对比的对象？
   c. 澄清得差不多后，主动提出 2-3 个具体切入角度，带你的推荐和理由——
      不要把"想个角度"这个负担甩给用户
   d. 用户明确认可某个角度后，才调用 wechat_deep_research / 开始写作
   方向已经足够具体（用户已给出角度/结论倾向）时可以跳过 a-c，直接确认后进入调研。
1. 调研：整篇文章先自己把 topic+angle 拆成 3-5 个具体子问题，调用
   wechat_deep_research(topic, angle, subquestions) 拿到并行搜索+去重+深读
   后的原始资料（sources）和写作要求提示（quality_guide）——注意这一步*不会*
   返回现成的报告，综合成报告是你自己接下来要做的事，不是 MCP 帮你做；
   只是核实一条信息、抓一个链接等轻量任务，直接用
   wechat_tavily_search / wechat_research / wechat_fetch_url 更快。
2. 数据真实性：文中数字、事实性声明必须有明确来源，不确定的信息要加"据报道"
   "约"等限定语，不得使用无数据支撑的夸大表述。用 wechat_save_sources 记录引用。
3. 配图：wechat_search_cover_image 选封面图 → wechat_download_image 下载。
   cover_path 实质必填——DEFAULT_COVER_PATH 指向的默认封面文件不一定存在，
   不传路径大概率导致封面上传失败。
4. 硬性字数限制（微信按字节计算，中文每字 3 字节）：
   - 标题上限约 64 字节 ≈ 21 个汉字（超限报错 45003）
   - 摘要上限约 120 字节 ≈ 40 个汉字，建议控制在 20 个汉字以内（超限报错 45004）
5. 写作校验：wechat_validate_content 检查 AI 口吻、占位符、标题党用词、
   事实来源缺失等问题。
6. 建稿：wechat_full_pipeline 一步完成渲染→传封面→建草稿；中途失败用
   wechat_resume_task 续跑。
7. 发布前复核：wechat_preview_task → wechat_audit_before_publish，
   blockers 必须修复才能发布。
8. 发布：wechat_publish（需 ENABLE_AUTO_PUBLISH=true，不可撤销）。

选题队列（wechat_add_topic/approve_topic/list_topics）目前只是状态记事本：
approve 不会自动触发调研/写作，队列里没有 reject/mark-published 工具
（底层 sources/topic_manager.py 里有实现，尚未接入 MCP 工具层）。"""

server = Server("wechat-publisher", instructions=SERVER_INSTRUCTIONS)


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
                    "title": {"type": "string", "description": "文章标题（微信限制约 64 字节 ≈ 21 个汉字，超限报错 45003）"},
                    "content_html": {"type": "string", "description": "文章 HTML 正文（通过 wechat_render_markdown 生成）"},
                    "thumb_media_id": {"type": "string", "description": "封面图素材 ID（通过 wechat_upload_cover 获取）"},
                    "digest": {"type": "string", "description": "文章摘要，微信限制约 120 字节 ≈ 40 个汉字（超限报错 45004），建议控制在 20 个汉字以内留余量"},
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
            name="wechat_deep_research",
            description=(
                "写文章前的资料收集：对你（调用方模型）自己规划好的子问题**并行**调用 Tavily + Exa 搜索，"
                "按 URL 去重后，对得分最高的几个来源尝试抓全文深读（而不只是搜索摘要），"
                "返回原始资料 + 质量要求提示，由你自己综合写成调研报告——"
                "本工具**不调用任何 LLM**，不需要配置 OPENAI_API_KEY，只需要 TAVILY_API_KEY 和/或 "
                "EXA_API_KEY（至少一个）。"
                "拆子问题和最终综合报告都由你自己做：你已经知道确认好的角度/受众/语气要求，"
                "比 MCP 内部单独调一次模型看到的上下文更完整，没必要让 MCP 再调一次别的模型重复这部分推理。"
                "写文章类任务建议：先把 topic+angle 拆成 3-5 个具体子问题，调用此工具拿到 sources，"
                "再按返回的 quality_guide 自己写报告、整理进 wechat_full_pipeline 的正文；"
                "只是想查一条信息、核实单个事实这种轻量任务不需要用这个，直接用 wechat_tavily_search / wechat_research 更快。"
                "angle 为硬性必填：调用前必须已经和用户确认过写作角度，不能拿一个笼统 topic 直接跑。"
                "如果用户只给了一个笼统方向（没说清写作角度、侧重点、目标读者、篇幅或结论倾向），"
                "应像 brainstorming 一样先和用户对齐——一次只问一个问题、优先给选择题，澄清角度/"
                "受众/结论倾向后，主动提出2-3个具体切入角度供用户选（带推荐理由），用户确认后再"
                "把确认好的角度填进 angle 参数调用此工具；不要为了绕过校验而随手编一个角度、"
                "或把 topic 原样复制进 angle。"
                "宽泛不带角度的调研会导致子问题和最终报告方向发散，白白消耗调研成本。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "调研主题，越具体效果越好，例如 '大语言模型预训练的原理、数据规模与训练成本 2026'",
                    },
                    "angle": {
                        "type": "string",
                        "description": (
                            "已经和用户确认过的写作角度/侧重点，例如"
                            "'对比国产大模型API定价策略，站在中小企业采购视角，结论倾向国产性价比更高'。"
                            "硬性必填——用来确保调研前已经完成方向澄清，不是随便填一句话应付过去。"
                        ),
                    },
                    "subquestions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "由你（调用方模型）自己拆解好的 3-5 个具体子问题，每个聚焦一个明确角度、"
                            "适合直接拿去搜索，不要相互重复；结合 topic 和 angle 来拆，涉及行业现状/"
                            "数据/排名的子问题措辞里带上年份或“最新”“现状”，不要问成历史科普题。"
                            "硬性必填——这一步不再由 MCP 内部调 LLM 完成。"
                        ),
                    },
                    "num_results_per_query": {
                        "type": "integer",
                        "description": "每个子问题每个搜索源返回几条结果，默认 5，最多 10",
                        "default": 5,
                    },
                    "recency": {
                        "type": "string",
                        "enum": ["any", "month", "year"],
                        "description": (
                            "时效性偏好，默认 year（优先近12个月资料，某个子问题在这个范围内"
                            "搜不到结果时会自动放宽重试一次，不是硬性过滤）。"
                            "变化快的话题（AI/科技/市场）用默认值；需要历史背景类资料的话题可传 any。"
                        ),
                        "default": "year",
                    },
                },
                "required": ["topic", "angle", "subquestions"],
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
                "发布前检查文章内容质量：长度不足、AI 口吻、占位符、重复段落、"
                "事实声明来源缺失、标题党词汇、超长段落等。\n"
                "传入 task_dir 可自动加载该任务的来源引用（sources.json），"
                "对事实声明做更精准的核查。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "文章标题"},
                    "digest": {"type": "string", "description": "文章摘要（可选）"},
                    "markdown": {"type": "string", "description": "文章正文 Markdown"},
                    "task_dir": {
                        "type": "string",
                        "description": "本地任务目录路径（可选），用于自动加载 sources.json",
                    },
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
                    "title": {"type": "string", "description": "新标题（微信限制约 64 字节 ≈ 21 个汉字，超限报错 45003，不是 64 个字符）"},
                    "content_html": {
                        "type": "string",
                        "description": "新 HTML 正文（通过 wechat_render_markdown 生成）",
                    },
                    "thumb_media_id": {
                        "type": "string",
                        "description": "封面素材 ID（通过 wechat_upload_cover 获取）",
                    },
                    "digest": {"type": "string", "description": "新摘要（微信限制约 120 字节 ≈ 40 个汉字，超限报错 45004，建议 20 个汉字以内）"},
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
                    "title": {"type": "string", "description": "文章标题（微信限制约 64 字节 ≈ 21 个汉字，超限报错 45003）"},
                    "markdown": {"type": "string", "description": "文章 Markdown 正文"},
                    "digest": {"type": "string", "description": "摘要（微信限制约 120 字节 ≈ 40 个汉字，超限报错 45004，建议 20 个汉字以内留余量）"},
                    "author": {"type": "string", "description": "作者名，默认 2AIBot"},
                    "cover_path": {
                        "type": "string",
                        "description": (
                            "封面图本地路径。实质必填——若不填会尝试用 DEFAULT_COVER_PATH，"
                            "但该环境变量指向的文件不一定存在，不存在则上传封面这一步会失败。"
                            "建议先调用 wechat_search_cover_image 选图，再用 wechat_download_image 下载后传路径进来。"
                        ),
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
        # ── Task recovery ────────────────────────────────────────────────────
        types.Tool(
            name="wechat_resume_task",
            description=(
                "从失败位置继续执行流水线。传入 task_dir（本地任务目录），"
                "工具自动检测上一步成功位置，从下一步开始重跑（上传封面 or 创建草稿）。"
                "适合 full_pipeline 中途失败后恢复。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task_dir": {
                        "type": "string",
                        "description": "本地任务目录路径，如 storage/drafts/2026-06-20-xxx",
                    },
                },
                "required": ["task_dir"],
            },
        ),
        types.Tool(
            name="wechat_preview_task",
            description=(
                "发布前预览一个任务的完整状态：标题、字数、摘要、封面、来源数量、"
                "步骤历史、风险提示。建议在 wechat_audit_before_publish 之前调用。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task_dir": {
                        "type": "string",
                        "description": "本地任务目录路径",
                    },
                },
                "required": ["task_dir"],
            },
        ),
        # ── Audit ─────────────────────────────────────────────────────────────
        types.Tool(
            name="wechat_audit_before_publish",
            description=(
                "发布前综合审核（比 wechat_validate_content 更全面）：\n"
                "  · 标题党风险词检测\n"
                "  · 事实声明来源核查\n"
                "  · 摘要长度检查\n"
                "  · 封面文件可访问性\n"
                "  · 步骤完整性（是否已完成 cover_uploaded + draft_created）\n"
                "返回 passed/warnings/blockers，blockers 须修复后才能发布。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task_dir": {
                        "type": "string",
                        "description": "本地任务目录路径",
                    },
                },
                "required": ["task_dir"],
            },
        ),
        types.Tool(
            name="wechat_save_sources",
            description=(
                "将文章来源引用列表保存到 task_dir/sources.json，供内容审核和可信度追溯使用。\n"
                "每个来源包含：url、title、date（可选）、key_points（可选）、risk_level（low/medium/high）。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task_dir": {
                        "type": "string",
                        "description": "本地任务目录路径",
                    },
                    "sources": {
                        "type": "array",
                        "description": "来源列表",
                        "items": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string"},
                                "title": {"type": "string"},
                                "date": {"type": "string"},
                                "key_points": {"type": "string"},
                                "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
                            },
                            "required": ["url"],
                        },
                    },
                },
                "required": ["task_dir", "sources"],
            },
        ),
        # ── Account profiles ─────────────────────────────────────────────────
        types.Tool(
            name="wechat_list_profiles",
            description=(
                "列出所有可用的公众号账号 profile（从 config/profiles/*.yaml 读取）。"
                "每个 profile 包含默认作者、默认模板、目标读者、禁用词等配置。\n"
                "当前活跃 profile 通过环境变量 WECHAT_PROFILE 指定，默认使用 default profile。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "如填写，返回该 profile 的详细配置（不填则列出所有）",
                    },
                },
                "required": [],
            },
        ),
        # ── Topic library ─────────────────────────────────────────────────────
        types.Tool(
            name="wechat_add_topic",
            description=(
                "将一个选题添加到待审队列（pending）。选题审批后可变为 approved，再由 full_pipeline 消费。"
                "如果选题只是一个笼统方向（angle 留空、没有明确侧重点/目标读者），"
                "建议先像 brainstorming 一样和用户一问一答对齐角度（一次一个问题、优先选择题），"
                "必要时提出2-3个候选角度供用户选，再提交，避免 approve 后写作阶段才发现方向不对。"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "选题主题"},
                    "account": {"type": "string", "description": "目标公众号名称（可选）"},
                    "priority": {
                        "type": "string",
                        "enum": ["high", "normal", "low"],
                        "description": "优先级，默认 normal",
                        "default": "normal",
                    },
                    "angle": {
                        "type": "string",
                        "description": "写作角度（可选，但强烈建议填写——留空说明方向还笼统，应先与用户讨论确认角度/侧重点/目标读者）",
                    },
                    "source_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "相关参考链接（可选）",
                    },
                    "deadline": {"type": "string", "description": "截止日期 YYYY-MM-DD（可选）"},
                },
                "required": ["topic"],
            },
        ),
        types.Tool(
            name="wechat_approve_topic",
            description="审批选题，将 pending 中的选题移入 approved 队列，使其可以被 full_pipeline 处理。",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic_id": {"type": "string", "description": "选题 ID（从 wechat_list_topics 获取）"},
                    "note": {"type": "string", "description": "审批备注（可选）"},
                },
                "required": ["topic_id"],
            },
        ),
        types.Tool(
            name="wechat_list_topics",
            description="查看选题队列。queue 可选 pending / approved / rejected / published / all。",
            inputSchema={
                "type": "object",
                "properties": {
                    "queue": {
                        "type": "string",
                        "enum": ["pending", "approved", "rejected", "published", "all"],
                        "description": "队列名称，默认 all",
                        "default": "all",
                    },
                    "account": {"type": "string", "description": "按公众号筛选（可选）"},
                    "priority": {
                        "type": "string",
                        "enum": ["high", "normal", "low"],
                        "description": "按优先级筛选（可选）",
                    },
                },
                "required": [],
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
        elif name == "wechat_deep_research":
            return await _deep_research(arguments)
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
        elif name == "wechat_resume_task":
            return await _resume_task(arguments)
        elif name == "wechat_preview_task":
            return await _preview_task(arguments)
        elif name == "wechat_audit_before_publish":
            return await _audit_before_publish(arguments)
        elif name == "wechat_save_sources":
            return await _save_sources(arguments)
        elif name == "wechat_list_profiles":
            return await _list_profiles(arguments)
        elif name == "wechat_add_topic":
            return await _add_topic(arguments)
        elif name == "wechat_approve_topic":
            return await _approve_topic(arguments)
        elif name == "wechat_list_topics":
            return await _list_topics(arguments)
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


async def _deep_research(args: dict) -> list[types.TextContent]:
    from ai.deep_research import gather_research_sources

    topic = args["topic"]
    angle = args.get("angle", "").strip()
    if not angle:
        return [types.TextContent(type="text", text=json.dumps({
            "status": "blocked",
            "message": (
                "angle 为空。请先和用户确认写作角度/侧重点（像 brainstorming 一样一次问一个"
                "问题、优先选择题，必要时提出2-3个候选角度供用户选），拿到用户确认的角度后"
                "再带着 angle 重新调用本工具；不要用 topic 原样填充 angle 来绕过这一步。"
            ),
        }, ensure_ascii=False))]

    subquestions = [q.strip() for q in args.get("subquestions", []) if q.strip()]
    if not subquestions:
        return [types.TextContent(type="text", text=json.dumps({
            "status": "blocked",
            "message": (
                "subquestions 为空。请先结合 topic 和 angle 自己拆出 3-5 个具体子问题"
                "（本工具不再内置 LLM 帮你拆），再带着 subquestions 重新调用本工具。"
            ),
        }, ensure_ascii=False))]

    num_results_per_query = min(int(args.get("num_results_per_query", 5)), 10)
    recency = args.get("recency", "year")

    try:
        result = gather_research_sources(topic, subquestions, num_results_per_query, recency)
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": str(e),
        }, ensure_ascii=False))]

    return [types.TextContent(
        type="text",
        text=json.dumps({
            "status": "ok",
            "topic": topic,
            "angle": angle,
            "subquestions": result["subquestions"],
            "gaps": result["gaps"],
            "num_sources": len(result["sources"]),
            "sources": result["sources"],
            "quality_guide": result["quality_guide"],
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
    from storage.task import (
        create_task_dir, save_markdown, save_html, save_upload_result,
        save_draft_result, save_article_json, update_status, record_step,
    )
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
    record_step(task_dir, "html_rendered", "success", {"html_path": str(html_path)})

    partial = {
        "title": title,
        "task_dir": str(task_dir),
        "markdown_path": str(md_path),
        "html_path": str(html_path),
    }

    # Step 3: Upload cover (may fail — return partial so caller can retry with wechat_resume_task)
    try:
        thumb_media_id = prepare_and_upload_cover(cover_path, str(task_dir))
        save_upload_result(task_dir, {"thumb_media_id": thumb_media_id})
        update_status(task_dir, "image_uploaded")
        record_step(task_dir, "cover_uploaded", "success", {"thumb_media_id": thumb_media_id})
        partial["thumb_media_id"] = thumb_media_id
    except Exception as e:
        record_step(task_dir, "cover_uploaded", "failed", error=str(e))
        partial["step_failed"] = "upload_cover"
        partial["error"] = str(e)
        partial["hint"] = (
            "HTML 和 Markdown 已保存到 task_dir。"
            "修复封面问题后调用 wechat_resume_task 从断点继续。"
        )
        return [types.TextContent(type="text", text=json.dumps(
            {"status": "partial", **partial}, ensure_ascii=False
        ))]

    # Step 4: Create draft (may fail — return partial so caller can retry with wechat_resume_task)
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
        record_step(task_dir, "draft_created", "success", {"media_id": media_id})
    except Exception as e:
        record_step(task_dir, "draft_created", "failed", error=str(e))
        partial["step_failed"] = "create_draft"
        partial["error"] = str(e)
        partial["hint"] = (
            f"封面已上传（thumb_media_id={thumb_media_id}）。"
            "调用 wechat_resume_task 从断点继续，或手动调用 wechat_create_draft。"
        )
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
    from storage.task import load_sources
    from pathlib import Path as _Path

    sources = []
    task_dir = args.get("task_dir", "").strip()
    if task_dir:
        sources = load_sources(_Path(task_dir))

    result = check_article({
        "title": args.get("title", ""),
        "digest": args.get("digest", ""),
        "markdown": args.get("markdown", ""),
    }, sources=sources)

    return [types.TextContent(type="text", text=json.dumps({
        "status": "ok",
        "passed": result.passed,
        "errors": result.errors,
        "warnings": result.warnings,
        "fact_claims": result.fact_claims,
        "title_risks": result.title_risks,
        "sources_loaded": len(sources),
        "source_required": result.source_required,
        "summary": (
            "✅ 内容检查通过" if result.passed
            else f"❌ 发现 {len(result.errors)} 个错误，{len(result.warnings)} 个警告"
        ),
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


async def _resume_task(args: dict) -> list[types.TextContent]:
    """Resume a pipeline from the last successful step."""
    from storage.task import get_resume_context, record_step, update_status, load_article_json
    from pathlib import Path as _Path

    task_dir = _Path(args["task_dir"])
    if not task_dir.exists():
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": f"任务目录不存在: {task_dir}",
        }, ensure_ascii=False))]

    ctx = get_resume_context(task_dir)
    resume_step = ctx["resume_step"]

    if resume_step == "completed":
        return [types.TextContent(type="text", text=json.dumps({
            "status": "ok",
            "message": "流水线已全部完成，无需恢复。",
            "context": ctx,
        }, ensure_ascii=False))]

    result: dict = {"resume_from": resume_step, **ctx}

    if resume_step == "cover_uploaded":
        # Need to re-upload cover
        cover_path = ctx.get("cover_path", "")
        if not cover_path:
            article = load_article_json(task_dir)
            cover_path = article.get("cover_path", "")
        if not cover_path:
            return [types.TextContent(type="text", text=json.dumps({
                "status": "error",
                "message": "找不到封面图路径，请手动调用 wechat_upload_cover",
                "context": ctx,
            }, ensure_ascii=False))]

        from images.image_uploader import prepare_and_upload_cover
        from storage.task import save_upload_result
        thumb_media_id = prepare_and_upload_cover(cover_path, str(task_dir))
        save_upload_result(task_dir, {"thumb_media_id": thumb_media_id})
        record_step(task_dir, "cover_uploaded", "success", {"thumb_media_id": thumb_media_id})
        update_status(task_dir, "image_uploaded")
        result["thumb_media_id"] = thumb_media_id
        resume_step = "draft_created"

    if resume_step == "draft_created":
        thumb_media_id = result.get("thumb_media_id") or ctx.get("thumb_media_id", "")
        if not thumb_media_id:
            return [types.TextContent(type="text", text=json.dumps({
                "status": "error",
                "message": "缺少 thumb_media_id，请先调用 wechat_upload_cover",
                "context": ctx,
            }, ensure_ascii=False))]

        html_path = ctx.get("html_path") or str(task_dir / "article.html")
        html_content = _Path(html_path).read_text(encoding="utf-8") if _Path(html_path).exists() else ""
        if not html_content:
            return [types.TextContent(type="text", text=json.dumps({
                "status": "error",
                "message": f"HTML 文件不存在: {html_path}",
                "context": ctx,
            }, ensure_ascii=False))]

        from wechat.draft import create_draft
        from storage.task import save_draft_result
        from config.settings import get_default_source_url

        article = load_article_json(task_dir)
        media_id = create_draft(
            title=ctx["title"],
            author=ctx.get("author") or "2AIBot",
            digest=ctx.get("digest", ""),
            content_html=html_content,
            thumb_media_id=thumb_media_id,
            source_url=article.get("source_url") or get_default_source_url(),
        )
        save_draft_result(task_dir, {"media_id": media_id, "created_at": datetime.now().isoformat()})
        record_step(task_dir, "draft_created", "success", {"media_id": media_id})
        update_status(task_dir, "draft_created")
        result["media_id"] = media_id

    result["status"] = "ok"
    result["message"] = "恢复执行完成，请前往微信公众号后台草稿箱审核。"
    return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]


async def _preview_task(args: dict) -> list[types.TextContent]:
    """Generate a comprehensive pre-publish task summary."""
    from storage.task import load_article_json, load_steps, load_sources, find_resume_step
    from pathlib import Path as _Path

    task_dir = _Path(args["task_dir"])
    if not task_dir.exists():
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": f"任务目录不存在: {task_dir}",
        }, ensure_ascii=False))]

    article = load_article_json(task_dir)
    steps = load_steps(task_dir)
    sources = load_sources(task_dir)

    md_path = task_dir / "article.md"
    md_text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
    html_path = task_dir / "article.html"

    upload_result: dict = {}
    try:
        upload_result = json.loads((task_dir / "upload_result.json").read_text(encoding="utf-8"))
    except Exception:
        pass

    draft_result: dict = {}
    try:
        draft_result = json.loads((task_dir / "draft_result.json").read_text(encoding="utf-8"))
    except Exception:
        pass

    title = article.get("title", "")
    digest = article.get("digest", "")

    risks: list[str] = []
    if len(title) > 21:
        risks.append(f"标题较长（{len(title)} 字），推荐 ≤21 字")
    if len(digest) > 20:
        risks.append(f"摘要较长（{len(digest)} 字），封面展示通常只显示前 20 字")
    if not sources:
        risks.append("未保存来源引用，建议调用 wechat_save_sources")
    cover_path = article.get("cover_path", "")
    if cover_path and not _Path(cover_path).exists():
        risks.append(f"封面文件已移动或不存在: {cover_path}")
    if not upload_result.get("thumb_media_id"):
        risks.append("封面尚未上传到微信 CDN（缺少 thumb_media_id）")
    if not draft_result.get("media_id"):
        risks.append("草稿尚未创建")

    preview = {
        "status": "ok",
        "title": title,
        "digest": digest,
        "author": article.get("author", ""),
        "word_count": len(md_text.replace(" ", "").replace("\n", "")),
        "markdown_lines": md_text.count("\n") + 1 if md_text else 0,
        "cover_path": cover_path,
        "thumb_media_id": upload_result.get("thumb_media_id", ""),
        "media_id": draft_result.get("media_id", ""),
        "html_path": str(html_path) if html_path.exists() else "",
        "sources_count": len(sources),
        "sources": sources,
        "pipeline_status": article.get("status", "unknown"),
        "steps_completed": [s["step"] for s in steps if s.get("status") == "success"],
        "next_step": find_resume_step(task_dir),
        "risks": risks,
        "ready_to_publish": len(risks) == 0 and bool(draft_result.get("media_id")),
    }
    return [types.TextContent(type="text", text=json.dumps(preview, ensure_ascii=False))]


async def _audit_before_publish(args: dict) -> list[types.TextContent]:
    """Comprehensive pre-publish audit: content + sources + pipeline completeness."""
    from storage.task import load_article_json, load_steps, load_sources
    from ai.content_checker import check_article
    from pathlib import Path as _Path

    task_dir = _Path(args["task_dir"])
    if not task_dir.exists():
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": f"任务目录不存在: {task_dir}",
        }, ensure_ascii=False))]

    article = load_article_json(task_dir)
    steps = load_steps(task_dir)
    sources = load_sources(task_dir)

    md_path = task_dir / "article.md"
    md_text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""

    check_result = check_article({
        "title": article.get("title", ""),
        "digest": article.get("digest", ""),
        "markdown": md_text,
    }, sources=sources)

    upload_result: dict = {}
    try:
        upload_result = json.loads((task_dir / "upload_result.json").read_text(encoding="utf-8"))
    except Exception:
        pass

    draft_result: dict = {}
    try:
        draft_result = json.loads((task_dir / "draft_result.json").read_text(encoding="utf-8"))
    except Exception:
        pass

    # Pipeline completeness check
    blockers: list[str] = list(check_result.errors)
    warnings: list[str] = list(check_result.warnings)

    if not upload_result.get("thumb_media_id"):
        blockers.append("封面尚未上传（缺少 thumb_media_id），请调用 wechat_upload_cover")
    if not draft_result.get("media_id"):
        blockers.append("草稿尚未创建（缺少 media_id），请调用 wechat_create_draft 或 wechat_full_pipeline")

    # Title risk
    for risk in check_result.title_risks:
        warnings.append(risk)

    # Fact claims summary
    if check_result.fact_claims:
        for claim in check_result.fact_claims:
            warnings.append(f"事实声明核查: {claim}")

    passed = len(blockers) == 0

    summary_lines = [
        f"{'✅' if passed else '❌'} 审核{'通过' if passed else '未通过'}",
        f"  · 阻断项: {len(blockers)}",
        f"  · 警告项: {len(warnings)}",
        f"  · 事实声明: {len(check_result.fact_claims)} 处",
        f"  · 来源引用: {len(sources)} 个",
    ]

    return [types.TextContent(type="text", text=json.dumps({
        "status": "ok",
        "passed": passed,
        "blockers": blockers,
        "warnings": warnings,
        "fact_claims": check_result.fact_claims,
        "title_risks": check_result.title_risks,
        "sources_count": len(sources),
        "thumb_media_id": upload_result.get("thumb_media_id", ""),
        "media_id": draft_result.get("media_id", ""),
        "summary": "\n".join(summary_lines),
    }, ensure_ascii=False))]


async def _save_sources(args: dict) -> list[types.TextContent]:
    """Save source citations to task_dir/sources.json."""
    from storage.task import save_sources, record_step
    from pathlib import Path as _Path

    task_dir = _Path(args["task_dir"])
    if not task_dir.exists():
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": f"任务目录不存在: {task_dir}",
        }, ensure_ascii=False))]

    sources = args.get("sources", [])
    save_sources(task_dir, sources)
    record_step(task_dir, "sources_saved", "success", {"count": len(sources)})

    return [types.TextContent(type="text", text=json.dumps({
        "status": "ok",
        "saved": len(sources),
        "path": str(task_dir / "sources.json"),
        "message": f"已保存 {len(sources)} 个来源引用",
    }, ensure_ascii=False))]


async def _list_profiles(args: dict) -> list[types.TextContent]:
    """List available account profiles."""
    from config.profiles import list_profiles, load_profile, get_current_profile

    name = args.get("name", "").strip()
    current = os.environ.get("WECHAT_PROFILE", "default")

    if name:
        profile = load_profile(name)
        return [types.TextContent(type="text", text=json.dumps({
            "status": "ok",
            "profile_name": name,
            "is_active": name == current,
            "profile": profile,
        }, ensure_ascii=False))]

    all_names = list_profiles()
    profiles_summary = []
    for n in all_names:
        p = load_profile(n)
        profiles_summary.append({
            "name": n,
            "account_name": p.get("account_name", n),
            "default_author": p.get("default_author", ""),
            "default_template": p.get("default_template", "A"),
            "is_active": n == current,
        })

    return [types.TextContent(type="text", text=json.dumps({
        "status": "ok",
        "active_profile": current,
        "profiles": profiles_summary,
        "tip": "通过环境变量 WECHAT_PROFILE=<name> 切换活跃 profile",
    }, ensure_ascii=False))]


async def _add_topic(args: dict) -> list[types.TextContent]:
    from sources.topic_manager import add_topic

    entry = add_topic(
        topic=args["topic"],
        account=args.get("account", ""),
        priority=args.get("priority", "normal"),
        angle=args.get("angle", ""),
        source_urls=args.get("source_urls", []),
        deadline=args.get("deadline", ""),
    )
    return [types.TextContent(type="text", text=json.dumps({
        "status": "ok",
        "message": f"选题已加入待审队列",
        "topic": entry,
    }, ensure_ascii=False))]


async def _approve_topic(args: dict) -> list[types.TextContent]:
    from sources.topic_manager import approve_topic

    entry = approve_topic(args["topic_id"], note=args.get("note", ""))
    if entry is None:
        return [types.TextContent(type="text", text=json.dumps({
            "status": "error",
            "message": f"未找到 topic_id={args['topic_id']}，请用 wechat_list_topics 确认",
        }, ensure_ascii=False))]
    return [types.TextContent(type="text", text=json.dumps({
        "status": "ok",
        "message": "选题已移入 approved 队列",
        "topic": entry,
    }, ensure_ascii=False))]


async def _list_topics(args: dict) -> list[types.TextContent]:
    from sources.topic_manager import list_topics

    queue = args.get("queue", "all")
    account = args.get("account", "")
    priority = args.get("priority", "")

    result = list_topics(queue=queue, account=account, priority=priority)
    total = sum(len(v) for v in result.values())

    return [types.TextContent(type="text", text=json.dumps({
        "status": "ok",
        "queue": queue,
        "total": total,
        "topics": result,
    }, ensure_ascii=False))]


async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
