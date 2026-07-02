"""服务端深度调研：把 superpowers deep-research 技能的多源搜索+综合流程

搬进 MCP server 本身，使其不依赖客户端（Claude Code 等）是否装了
对应技能——任何 MCP 客户端/模型调用 wechat_deep_research 都能拿到
同样带引用来源的调研报告。
"""

import json
import os
import re
from typing import Optional

from loguru import logger
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


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
    prompt = (
        f"把下面这个调研主题拆解成 {num_subquestions} 个具体的子问题，"
        "每个子问题聚焦一个明确角度，适合直接拿去做网络搜索，不要相互重复。\n\n"
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


def search_tavily(query: str, num_results: int = 5) -> list[dict]:
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return []
    from tavily import TavilyClient

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query, max_results=num_results, search_depth="advanced", include_answer=False
    )
    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "source": "tavily",
        }
        for r in response.get("results", [])
        if r.get("url")
    ]


def search_exa(query: str, num_results: int = 5) -> list[dict]:
    api_key = os.environ.get("EXA_API_KEY", "").strip()
    if not api_key:
        return []
    from exa_py import Exa

    exa = Exa(api_key=api_key)
    response = exa.search_and_contents(
        query, num_results=num_results, text={"max_characters": 2000}, use_autoprompt=True
    )
    return [
        {
            "title": r.title or "",
            "url": r.url or "",
            "content": (r.text or "").strip(),
            "source": "exa",
            "published_date": r.published_date or "",
        }
        for r in response.results
        if r.url
    ]


def gather_sources(subquestions: list[str], num_results_per_query: int = 5) -> list[dict]:
    """对每个子问题并行跑 Tavily + Exa（各自可选，缺 key 就跳过），按 URL 去重。"""
    all_sources: list[dict] = []
    seen_urls: set[str] = set()
    for q in subquestions:
        for search_fn in (search_tavily, search_exa):
            try:
                results = search_fn(q, num_results_per_query)
            except Exception as e:
                logger.warning(f"deep_research 搜索失败 ({search_fn.__name__}, query={q!r}): {e}")
                continue
            for r in results:
                if r["url"] in seen_urls:
                    continue
                seen_urls.add(r["url"])
                r["subquestion"] = q
                all_sources.append(r)
    return all_sources


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def synthesize_report(topic: str, subquestions: list[str], sources: list[dict]) -> str:
    client, model = _openai_client()
    sources_text = "\n\n".join(
        f"[{i + 1}] {s['title']} ({s['url']})\n{s['content'][:1500]}"
        for i, s in enumerate(sources)
    )
    prompt = f"""你是一个调研分析师。基于下面收集到的资料，就"{topic}"这个主题写一份结构化调研报告。

要求：
1. 每个关键论点后面用 [编号] 标注对应来源，编号对应资料列表里的序号
2. 只使用资料中出现的信息，不要编造或引用资料之外的数据
3. 如果某个说法只有单一来源支撑，在文中提示"未交叉验证"
4. 用 Markdown 格式，包含：概述、按主题分的小节、关键结论、来源列表（标题+URL）

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
) -> dict:
    """完整跑一遍：拆子问题 → 多源搜索去重 → 综合成带引用的报告。"""
    logger.info(f"deep_research 开始：{topic}")
    subquestions = plan_subquestions(topic, num_subquestions)
    sources = gather_sources(subquestions, num_results_per_query)
    if not sources:
        raise RuntimeError(
            "没有搜到任何资料，请检查 .env 里 TAVILY_API_KEY / EXA_API_KEY 是否至少配置了一个"
        )
    report = synthesize_report(topic, subquestions, sources)
    logger.info(f"deep_research 完成：{len(sources)} 条来源，报告 {len(report)} 字符")
    return {
        "topic": topic,
        "subquestions": subquestions,
        "report_markdown": report,
        "sources": [
            {"url": s["url"], "title": s["title"], "source": s["source"]} for s in sources
        ],
    }
