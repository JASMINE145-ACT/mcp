"""服务端深度调研：把 superpowers deep-research 技能的多源搜索+综合流程

搬进 MCP server 本身，使其不依赖客户端（Claude Code 等）是否装了
对应技能——任何 MCP 客户端/模型调用 wechat_deep_research 都能拿到
同样带引用来源的调研报告。

相比原版技能，这版额外做了：并行多源搜索（不再逐个子问题串行请求）、
时效性偏好（默认只要近12个月的资料，查不到时自动放宽重试一次）、
对排名靠前的来源做全文深读（不再只依赖搜索摘要）、
以及综合报告里显式要求标注信息缺口、区分事实与推测。
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Optional

from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

RECENCY_TAVILY_RANGE = {"any": None, "month": "month", "year": "year"}
RECENCY_DAYS = {"any": None, "month": 30, "year": 365}


def _extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    return json.loads(text)


def _openai_client():
    from config.settings import get_settings
    s = get_settings()
    return OpenAI(api_key=s.OPENAI_API_KEY), (s.OPENAI_MODEL or "gpt-4.1")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def plan_subquestions(topic: str, num_subquestions: int = 4) -> list[str]:
    """把调研主题拆成若干个适合直接搜索的子问题。"""
    client, model = _openai_client()
    current_year = datetime.now().year
    prompt = (
        f"把下面这个调研主题拆解成 {num_subquestions} 个具体的子问题，"
        "每个子问题聚焦一个明确角度，适合直接拿去做网络搜索，不要相互重复。"
        f"现在是 {current_year} 年，涉及行业现状/数据/排名的子问题请体现出对最新情况的关注"
        "（例如措辞里包含年份或“最新”“现状”等），不要问成历史科普题。\n\n"
        f"主题：{topic}\n\n"
        '只返回 JSON：{"subquestions": ["...", "...", ...]}'
    )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    data = _extract_json(raw)
    subquestions = data.get("subquestions") or [topic]
    return subquestions[:num_subquestions] or [topic]


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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def synthesize_report(topic: str, subquestions: list[str], sources: list[dict]) -> str:
    client, model = _openai_client()
    sources_text = "\n\n".join(
        f"[{i + 1}] {s['title']} ({s['url']})"
        f"{' [已全文深读]' if s.get('deep_read') else ''}"
        f"{' [' + s['published_date'] + ']' if s.get('published_date') else ''}\n"
        f"{s['content'][:3000 if s.get('deep_read') else 1500]}"
        for i, s in enumerate(sources)
    )
    prompt = f"""你是一个调研分析师。基于下面收集到的资料，就"{topic}"这个主题写一份结构化调研报告。

要求：
1. 每个关键论点后面用 [编号] 标注对应来源，编号对应资料列表里的序号
2. 只使用资料中出现的信息，不要编造或引用资料之外的数据
3. 如果某个说法只有单一来源支撑，在文中提示"未交叉验证"
4. 优先采信标注了较新发布日期、或已全文深读的资料；如果发现资料明显过时，在文中提示"该信息可能已过时"
5. 如果某个子问题没有搜到有效资料，在报告里明确指出这个信息缺口，不要略过不提
6. 清楚区分"已证实的事实"和"预测/推测/评论性判断"——对预测、展望、观点性内容要标注"（推测/预期）"等字样，不要把推测写得像既定事实
7. 用 Markdown 格式，包含：概述、按主题分的小节、关键结论、信息缺口（如有）、来源列表（标题+URL）、方法说明（搜了几个子问题、覆盖了几条来源、整体置信度 高/中/低及理由）

调研的子问题：
{chr(10).join(f"- {q}" for q in subquestions)}

收集到的资料：
{sources_text}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=4096,
    )
    return response.choices[0].message.content or ""


def run_deep_research(
    topic: str,
    num_subquestions: int = 4,
    num_results_per_query: int = 5,
    recency: str = "year",
    deep_read_top_n: int = 5,
) -> dict:
    """完整跑一遍：拆子问题 → 并行多源搜索去重 → 深读高分来源 → 综合成带引用的报告。

    recency: "any" | "month" | "year"（默认 year，对应"优先近12个月资料"；
    某个查询在该范围内搜不到结果时会自动放宽重试一次，不是硬性过滤）。
    """
    logger.info(f"deep_research 开始：{topic}（recency={recency}）")
    subquestions = plan_subquestions(topic, num_subquestions)
    sources = gather_sources(subquestions, num_results_per_query, recency)
    if not sources:
        raise RuntimeError(
            "没有搜到任何资料，请检查 .env 里 TAVILY_API_KEY / EXA_API_KEY 是否至少配置了一个"
        )
    sources = deep_read_top_sources(sources, top_n=deep_read_top_n)
    report = synthesize_report(topic, subquestions, sources)
    num_deep_read = sum(1 for s in sources if s.get("deep_read"))
    logger.info(
        f"deep_research 完成：{len(sources)} 条来源（{num_deep_read} 条已深读），"
        f"报告 {len(report)} 字符"
    )
    return {
        "topic": topic,
        "subquestions": subquestions,
        "report_markdown": report,
        "sources": [
            {
                "url": s["url"],
                "title": s["title"],
                "source": s["source"],
                "score": round(s.get("score", 0.0), 3),
                "deep_read": s.get("deep_read", False),
                "published_date": s.get("published_date", ""),
            }
            for s in sources
        ],
    }
