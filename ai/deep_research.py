"""服务端调研辅助：把 superpowers deep-research 技能里"多源搜索+去重+深读"这部分

搬进 MCP server 本身，使其不依赖客户端（Claude Code 等）是否装了对应技能——
任何 MCP 客户端调用 wechat_deep_research 都能拿到同样并行、带时效性偏好、
带全文深读的原始资料。

"拆子问题"和"综合成报告"这两步刻意**没有**放在这里、也不调用任何 LLM API：
调用方（正在编排整个流程的那个模型）本身就是一个 LLM，能直接做这部分推理，
而且比 MCP 内部单独调一个模型看到的上下文更完整（已确认的写作角度、目标读者、
数据真实性要求等）。让 MCP 只做纯机械的检索/去重/抓取，不需要 MCP 自己维护
OPENAI_API_KEY，也避免了"两个模型接力做同一类工作"的浪费。
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

RECENCY_TAVILY_RANGE = {"any": None, "month": "month", "year": "year"}
RECENCY_DAYS = {"any": None, "month": 30, "year": 365}

REPORT_QUALITY_GUIDE = """写调研报告时请遵守（这部分推理由你——调用方模型——直接完成，不是 MCP 帮你写）：
1. 每个关键论点后标注对应来源（用来源的 title/url 或编号），不要写无来源支撑的断言
2. 只使用下面 sources 里出现的信息，不要编造或引用资料之外的数据
3. 如果某个说法只有单一来源支撑，标注"未交叉验证"
4. 优先采信 published_date 较新、或 deep_read=true 的资料；明显过时的信息标注"可能已过时"
5. gaps 字段里列出的子问题（没搜到有效资料）要在报告里明确指出信息缺口，不要略过不提
6. 清楚区分"已证实的事实"和"预测/推测/评论性判断"，后者标注"（推测/预期）"
7. 报告建议包含：概述、按主题分小节、关键结论、信息缺口（如有）、来源列表、方法说明（子问题数/来源数/整体置信度）"""


def search_tavily(query: str, num_results: int = 5, recency: str = "year") -> list[dict]:
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return []
    from tavily import TavilyClient

    client = TavilyClient(api_key=api_key)
    time_range = RECENCY_TAVILY_RANGE.get(recency)

    def _do_search(tr: Optional[str]) -> list[dict]:
        response = client.search(
            query,
            max_results=num_results,
            search_depth="advanced",
            include_answer=False,
            time_range=tr,
        )
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0.0) or 0.0,
                "source": "tavily",
            }
            for r in response.get("results", [])
            if r.get("url")
        ]

    results = _do_search(time_range)
    if not results and time_range:
        # 时效性范围内没搜到，放宽限制重试一次（"偏好近期"而不是"强制近期"）
        logger.info(f"tavily 在 time_range={time_range} 内无结果，放宽重试：{query!r}")
        results = _do_search(None)
    return results


def search_exa(query: str, num_results: int = 5, recency: str = "year") -> list[dict]:
    api_key = os.environ.get("EXA_API_KEY", "").strip()
    if not api_key:
        return []
    from exa_py import Exa

    exa = Exa(api_key=api_key)
    days = RECENCY_DAYS.get(recency)
    start_published_date = (
        (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d") if days else None
    )

    def _do_search(published_after: Optional[str]) -> list[dict]:
        kwargs = {"num_results": num_results, "text": {"max_characters": 2000}, "use_autoprompt": True}
        if published_after:
            kwargs["start_published_date"] = published_after
        response = exa.search_and_contents(query, **kwargs)
        return [
            {
                "title": r.title or "",
                "url": r.url or "",
                "content": (r.text or "").strip(),
                "score": getattr(r, "score", None) or 0.0,
                "source": "exa",
                "published_date": r.published_date or "",
            }
            for r in response.results
            if r.url
        ]

    results = _do_search(start_published_date)
    if not results and start_published_date:
        logger.info(f"exa 在 start_published_date={start_published_date} 内无结果，放宽重试：{query!r}")
        results = _do_search(None)
    return results


def gather_sources(
    subquestions: list[str], num_results_per_query: int = 5, recency: str = "year"
) -> list[dict]:
    """对每个子问题并行跑 Tavily + Exa（各自可选，缺 key 就跳过），按 URL 去重。

    并行而非逐个子问题串行请求——子问题数×搜索源数的调用会一次性发出去，
    整体耗时约等于最慢的那一次请求，而不是全部请求耗时之和。
    """
    tasks = [
        (search_fn, q)
        for q in subquestions
        for search_fn in (search_tavily, search_exa)
    ]

    all_sources: list[dict] = []
    seen_urls: set[str] = set()
    with ThreadPoolExecutor(max_workers=max(len(tasks), 1)) as pool:
        future_to_query = {
            pool.submit(search_fn, q, num_results_per_query, recency): (search_fn.__name__, q)
            for search_fn, q in tasks
        }
        for future in as_completed(future_to_query):
            fn_name, q = future_to_query[future]
            try:
                results = future.result()
            except Exception as e:
                logger.warning(f"deep_research 搜索失败 ({fn_name}, query={q!r}): {e}")
                continue
            for r in results:
                if r["url"] in seen_urls:
                    continue
                seen_urls.add(r["url"])
                r["subquestion"] = q
                r["deep_read"] = False
                all_sources.append(r)
    return all_sources


def _deep_read_url(url: str, max_chars: int = 6000) -> Optional[str]:
    """尽力抓取 URL 全文，失败就返回 None（调用方应回退到搜索摘要）。"""
    try:
        from scrapling.fetchers import Fetcher
        page = Fetcher.get(url, stealthy_headers=True, follow_redirects=True)
        text = page.get_all_text(
            ignore_tags=("script", "style", "nav", "footer", "header", "aside")
        )
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        text = "\n".join(lines)
        if not text:
            return None
        return text[:max_chars]
    except Exception as e:
        logger.warning(f"deep_read 抓取失败 ({url}): {e}")
        return None


def deep_read_top_sources(
    sources: list[dict], top_n: int = 5, max_chars: int = 6000
) -> list[dict]:
    """对得分最高的 top_n 个来源尝试抓全文替换掉搜索摘要，其余保持原样。

    对应原版技能里"对最有价值的几个源做全文深读，不能只看搜索摘要"这一步。
    抓取失败的直接跳过，保留原有 snippet，不影响整体流程。
    """
    if not sources:
        return sources

    ranked = sorted(sources, key=lambda s: s.get("score", 0.0), reverse=True)
    candidates = ranked[:top_n]
    urls_to_read = {s["url"] for s in candidates}

    fetched: dict[str, Optional[str]] = {}
    with ThreadPoolExecutor(max_workers=max(len(urls_to_read), 1)) as pool:
        future_to_url = {pool.submit(_deep_read_url, url, max_chars): url for url in urls_to_read}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            fetched[url] = future.result()

    for s in sources:
        full_text = fetched.get(s["url"])
        if full_text and len(full_text) > len(s.get("content", "")):
            s["content"] = full_text
            s["deep_read"] = True
    return sources


def gather_research_sources(
    topic: str,
    subquestions: list[str],
    num_results_per_query: int = 5,
    recency: str = "year",
    deep_read_top_n: int = 5,
) -> dict:
    """并行多源搜索去重 → 深读高分来源 → 返回原始资料供调用方自己综合成报告。

    不调用任何 LLM——subquestions 由调用方（正在编排流程的模型）先规划好再传进来，
    最终的报告综合也由调用方自己完成，这里只负责机械的检索/去重/抓取。

    recency: "any" | "month" | "year"（默认 year，对应"优先近12个月资料"；
    某个查询在该范围内搜不到结果时会自动放宽重试一次，不是硬性过滤）。
    """
    logger.info(f"gather_research_sources 开始：{topic}（recency={recency}）")
    sources = gather_sources(subquestions, num_results_per_query, recency)
    if not sources:
        raise RuntimeError(
            "没有搜到任何资料，请检查 .env 里 TAVILY_API_KEY / EXA_API_KEY 是否至少配置了一个"
        )
    sources = deep_read_top_sources(sources, top_n=deep_read_top_n)
    covered_subquestions = {s["subquestion"] for s in sources}
    gaps = [q for q in subquestions if q not in covered_subquestions]
    num_deep_read = sum(1 for s in sources if s.get("deep_read"))
    logger.info(
        f"gather_research_sources 完成：{len(sources)} 条来源（{num_deep_read} 条已深读），"
        f"{len(gaps)} 个子问题无结果"
    )
    return {
        "topic": topic,
        "subquestions": subquestions,
        "gaps": gaps,
        "quality_guide": REPORT_QUALITY_GUIDE,
        "sources": [
            {
                "url": s["url"],
                "title": s["title"],
                "source": s["source"],
                "subquestion": s["subquestion"],
                "content": s["content"],
                "score": round(s.get("score", 0.0), 3),
                "deep_read": s.get("deep_read", False),
                "published_date": s.get("published_date", ""),
            }
            for s in sources
        ],
    }
