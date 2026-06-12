import json
from openai import OpenAI
from pathlib import Path
from loguru import logger

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def generate_image_prompt(title: str, topic: str) -> dict:
    from config.settings import get_settings
    s = get_settings()
    client = OpenAI(api_key=s.OPENAI_API_KEY)

    template = (PROMPTS_DIR / "image_prompt.md").read_text(encoding="utf-8")
    prompt = template.replace("{{title}}", title).replace("{{topic}}", topic)

    logger.info("生成封面图提示词")
    response = client.chat.completions.create(
        model=s.OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except Exception:
        return {"cover_prompt": f"futuristic AI robot technology concept for {title}"}
