import json
from openai import OpenAI
from pathlib import Path
from loguru import logger

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def generate_titles(content_summary: str, count: int = 5) -> list[str]:
    from config.settings import get_settings
    s = get_settings()
    client = OpenAI(api_key=s.OPENAI_API_KEY)

    template = (PROMPTS_DIR / "title_prompt.md").read_text(encoding="utf-8")
    prompt = template.replace("{{content_summary}}", content_summary[:500])

    logger.info("生成备选标题")
    response = client.chat.completions.create(
        model=s.OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=500,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "[]"
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data[:count]
        # Some models wrap in an object
        for v in data.values():
            if isinstance(v, list):
                return v[:count]
    except Exception:
        pass
    return []
