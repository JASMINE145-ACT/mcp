import re
from dataclasses import dataclass, field
from typing import List

# ── AI 口吻特征词 ────────────────────────────────────────────────────────────

AI_PHRASES = [
    "作为一个AI", "作为AI", "作为一个人工智能", "作为人工智能",
    "我是一个AI", "我是AI", "我无法", "我不能保证",
    "需要注意的是，我", "请注意，我",
    "总的来说", "综上所述", "总而言之",
    "首先……其次……最后", "不得不说",
]

# ── 占位符模式 ───────────────────────────────────────────────────────────────

PLACEHOLDER_PATTERNS = [
    r"数据待补充", r"此处插入图片", r"\[图片\]", r"\[待填写\]",
    r"\[TODO\]", r"待确认", r"xxx", r"\[数据来源\]",
    r"\[作者\]", r"\[链接\]", r"待补", r"TBD",
]

# ── 需要来源的事实声明模式 ────────────────────────────────────────────────────

FACT_CLAIM_PATTERNS = [
    (r"数据显示[^，,。.]{0,30}[%％\d]", "含数据声明，需注明来源"),
    (r"据报道", "引用报道，需注明媒体来源"),
    (r"官方表示|官方数据|官方发布", "引用官方表述，建议附链接"),
    (r"业内人士[认为表示称]", "引用业内人士观点，建议说明来源"),
    (r"首次|首个|全球第一|世界第一|唯一", "极端声明，需有来源支撑"),
    (r"暴跌|暴涨|崩溃|爆炸式增长", "极端表述，建议标注数据来源"),
    (r"突破.{0,10}[%％亿万]", "关键数据突破，需注明来源"),
    (r"同比.{0,8}[%％]|环比.{0,8}[%％]", "含增减幅数据，需注明来源"),
    (r"研究(表明|显示|发现)", "引用研究结论，需注明研究机构"),
    (r"预计.{0,20}[%％亿万]", "含预测数据，需说明来源"),
]

# ── 标题风险词 ───────────────────────────────────────────────────────────────

TITLE_CLICKBAIT = [
    "震惊", "惊呆", "不敢相信", "太厉害了", "彻底爆了", "炸了",
    "竟然", "居然", "没想到", "颠覆", "终结", "消失",
    "一夜暴富", "轻松月入", "躺赚",
]


@dataclass
class CheckResult:
    passed: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    fact_claims: List[str] = field(default_factory=list)
    title_risks: List[str] = field(default_factory=list)
    source_required: bool = False


def check_article(article: dict, sources: list | None = None) -> CheckResult:
    """
    Comprehensive article check.

    Args:
        article: dict with keys title, digest, markdown
        sources: optional list of source dicts from sources.json
    """
    result = CheckResult(passed=True)
    sources = sources or []

    title = article.get("title", "")
    digest = article.get("digest", "")
    markdown_body = article.get("markdown", "")

    # ── Basic field checks ───────────────────────────────────────────────────

    if not title or not title.strip():
        result.errors.append("标题为空")
        result.passed = False
    elif len(title) > 64:
        result.errors.append(f"标题过长（{len(title)} 字符，微信限制 64）")
        result.passed = False
    elif len(title) > 21:
        result.warnings.append(f"标题较长（{len(title)} 字符，推荐 ≤21 字）")

    if not digest or not digest.strip():
        result.warnings.append("摘要为空，建议填写 20 字以内的摘要")
    elif len(digest) > 120:
        result.errors.append(f"摘要超长（{len(digest)} 字，微信限制 120）")
        result.passed = False
    elif len(digest) > 20:
        result.warnings.append(f"摘要较长（{len(digest)} 字），微信封面展示通常只显示前 20 字")

    if len(markdown_body) < 800:
        result.errors.append(f"正文太短（{len(markdown_body)} 字，要求至少 800 字）")
        result.passed = False

    # ── Title clickbait check ────────────────────────────────────────────────

    for word in TITLE_CLICKBAIT:
        if word in title:
            result.title_risks.append(f"标题包含标题党词汇「{word}」")
            result.warnings.append(f"标题包含标题党词汇「{word}」，可能影响账号信誉")

    # ── AI phrase check ──────────────────────────────────────────────────────

    for phrase in AI_PHRASES:
        if phrase in markdown_body:
            result.warnings.append(f'正文含明显 AI 口吻表达：「{phrase}」')

    # ── Placeholder check ────────────────────────────────────────────────────

    for pattern in PLACEHOLDER_PATTERNS:
        if re.search(pattern, markdown_body):
            result.errors.append(f"正文含未替换占位符，匹配: {pattern}")
            result.passed = False

    # ── Unclosed markdown blocks ─────────────────────────────────────────────

    if markdown_body.count("```") % 2 != 0:
        result.warnings.append("正文存在未闭合的代码块（奇数个 ``` ）")

    # ── Duplicate paragraph detection ────────────────────────────────────────

    paras = [p.strip() for p in markdown_body.split("\n\n") if p.strip()]
    seen: set[str] = set()
    for para in paras:
        if len(para) > 30:
            if para in seen:
                result.warnings.append(f"检测到重复段落: {para[:50]}…")
            seen.add(para)

    # ── Fact claim detection ─────────────────────────────────────────────────

    for pattern, label in FACT_CLAIM_PATTERNS:
        matches = re.findall(pattern, markdown_body)
        if matches:
            result.fact_claims.append(f"{label}（示例: 「{matches[0][:30]}」）")

    if result.fact_claims:
        result.source_required = True
        if not sources:
            result.warnings.append(
                f"正文含 {len(result.fact_claims)} 处需注明来源的事实声明，"
                "但未提供 sources.json，建议调用 wechat_save_sources 记录来源"
            )
        else:
            result.warnings.append(
                f"已提供 {len(sources)} 个来源，"
                f"请核查 {len(result.fact_claims)} 处事实声明均有对应来源"
            )

    # ── Structure quality check ──────────────────────────────────────────────

    h2_count = len(re.findall(r"^##\s", markdown_body, re.MULTILINE))
    if h2_count == 0 and len(markdown_body) > 1500:
        result.warnings.append("文章较长但无二级标题（##），建议添加结构标题提升可读性")

    long_paras = [p for p in paras if len(p) > 400 and not p.startswith("#")]
    if long_paras:
        result.warnings.append(
            f"存在 {len(long_paras)} 个超长段落（>400 字），建议拆分以提升公众号阅读体验"
        )

    opening = paras[0] if paras else ""
    if len(opening) > 200:
        result.warnings.append("开头段落较长（>200 字），公众号读者容易流失，建议精简开头")

    return result


def check_title_only(title: str) -> CheckResult:
    """Quick check for title only."""
    result = CheckResult(passed=True)
    if not title.strip():
        result.errors.append("标题为空")
        result.passed = False
    elif len(title) > 64:
        result.errors.append(f"标题过长（{len(title)} 字符）")
        result.passed = False
    for word in TITLE_CLICKBAIT:
        if word in title:
            result.title_risks.append(f"标题含标题党词汇「{word}」")
    return result
