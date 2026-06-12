import json
import re
from pathlib import Path
from typing import Optional
from openai import OpenAI
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def _extract_json(text: str) -> dict:
    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"AI 返回的内容不是合法 JSON: {e}\n内容片段: {text[:200]}") from e


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def generate_article(
    topic: str,
    style_guide: str,
    sources: str = "",
    model: str = "gpt-4.1",
    author: str = "2AIBot",
) -> dict:
    """Call OpenAI to generate a structured article JSON."""
    from config.settings import get_settings
    s = get_settings()
    client = OpenAI(api_key=s.OPENAI_API_KEY)
    model = s.OPENAI_MODEL or model

    system_prompt = _load_prompt("system_prompt.md")
    article_template = _load_prompt("article_prompt.md")

    user_prompt = (
        article_template
        .replace("{{topic}}", topic)
        .replace("{{style_guide}}", style_guide)
        .replace("{{sources}}", sources or "无参考素材，请根据主题独立分析。")
    )

    logger.info(f"调用 OpenAI ({model}) 生成文章，主题: {topic}")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    if not raw:
        raise ValueError("OpenAI 返回内容为空")

    logger.info("文章生成完成，正在解析 JSON")
    article = _extract_json(raw)

    if "author" not in article or not article["author"]:
        article["author"] = author

    return article


def polish_article(article: dict, model: str = None) -> dict:
    """Second-pass polish to reduce AI-ish tone."""
    from config.settings import get_settings
    s = get_settings()
    client = OpenAI(api_key=s.OPENAI_API_KEY)
    model = model or s.OPENAI_MODEL

    markdown_body = article.get("markdown", "")
    if not markdown_body:
        return article

    logger.info("对文章进行二次润色")
    polish_prompt = f"""请对以下公众号文章正文进行润色，要求：
1. 减少"首先、其次、最后"等机械表达，换成更自然的连接方式；
2. 删除空泛判断，替换为具体分析；
3. 保持公众号口吻，专业但不晦涩；
4. 检查并删除重复段落；
5. 保持原有结构和小标题，不要大改；
6. 只返回润色后的 Markdown 正文，不加任何解释。

原文：
{markdown_body}"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": polish_prompt}],
        temperature=0.5,
        max_tokens=4096,
    )
    polished = response.choices[0].message.content
    if polished:
        article = {**article, "markdown": polished}
    return article
