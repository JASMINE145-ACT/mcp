import re
from dataclasses import dataclass, field
from typing import List

AI_PHRASES = [
    "作为一个AI", "作为AI", "作为一个人工智能", "作为人工智能",
    "我是一个AI", "我是AI", "我无法", "我不能保证",
    "需要注意的是，我", "请注意，我",
]

PLACEHOLDER_PATTERNS = [
    r"数据待补充", r"此处插入图片", r"\[图片\]", r"\[待填写\]",
    r"\[TODO\]", r"待确认", r"xxx", r"\[数据来源\]",
]


@dataclass
class CheckResult:
    passed: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


def check_article(article: dict) -> CheckResult:
    result = CheckResult(passed=True)

    title = article.get("title", "")
    digest = article.get("digest", "")
    markdown_body = article.get("markdown", "")

    if not title or not title.strip():
        result.errors.append("标题为空")
        result.passed = False

    if not digest or not digest.strip():
        result.warnings.append("摘要为空")

    if len(markdown_body) < 800:
        result.errors.append(f"正文太短（{len(markdown_body)} 字，要求至少 800 字）")
        result.passed = False

    for phrase in AI_PHRASES:
        if phrase in markdown_body:
            result.warnings.append(f'正文包含明显 AI 口吻表达："{phrase}"')

    for pattern in PLACEHOLDER_PATTERNS:
        if re.search(pattern, markdown_body):
            result.warnings.append(f"正文可能包含占位符，匹配: {pattern}")

    # Check for unclosed markdown
    open_backticks = markdown_body.count("```")
    if open_backticks % 2 != 0:
        result.warnings.append("正文中存在未闭合的代码块（奇数个 ``` ）")

    # Detect repeated paragraphs
    paras = [p.strip() for p in markdown_body.split("\n\n") if p.strip()]
    seen = set()
    for para in paras:
        if len(para) > 30:
            if para in seen:
                result.warnings.append(f"检测到重复段落: {para[:50]}...")
            seen.add(para)

    return result
