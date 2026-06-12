import markdown
from bs4 import BeautifulSoup, Tag, NavigableString

from render.templates import get_template, DEFAULT_TEMPLATE


def _is_highlight_paragraph(tag: Tag, styles: dict) -> bool:
    """<p> 内仅一个 <strong> 且文本 ≤ 30 字 → callout highlight。"""
    if "highlight_p" not in styles:
        return False
    children = [
        c for c in tag.children
        if not (isinstance(c, NavigableString) and not c.strip())
    ]
    if len(children) != 1:
        return False
    child = children[0]
    return isinstance(child, Tag) and child.name == "strong" and len(child.get_text(strip=True)) <= 30


def _apply_styles(soup: BeautifulSoup, styles: dict) -> None:
    for tag in soup.find_all(["h1", "h2", "h3"]):
        tag["style"] = styles.get(tag.name, styles.get("h3", ""))

    for tag in soup.find_all("p"):
        if _is_highlight_paragraph(tag, styles):
            tag["style"] = styles["highlight_p"]
        else:
            tag["style"] = styles["p"]

    for tag in soup.find_all("strong"):
        tag["style"] = styles.get("strong", "font-weight: bold;")

    for tag in soup.find_all("em"):
        tag["style"] = styles.get("em", "font-style: italic;")

    for bq in soup.find_all("blockquote"):
        bq["style"] = styles.get("blockquote", "")
        for p in bq.find_all("p"):
            p["style"] = styles.get("p_no_indent", styles["p"])

    for tag in soup.find_all("ul"):
        tag["style"] = styles.get("ul", "")
    for tag in soup.find_all("ol"):
        tag["style"] = styles.get("ol", "")
    for tag in soup.find_all("li"):
        tag["style"] = styles.get("li", "")

    for tag in soup.find_all("img"):
        tag["style"] = styles.get("img", "")
        alt = tag.get("alt", "")
        img_wrapper_style = styles.get("img_wrapper")
        if img_wrapper_style:
            wrapper = soup.new_tag("section")
            wrapper["style"] = img_wrapper_style
            tag.replace_with(wrapper)
            wrapper.append(tag)
            if alt:
                caption = soup.new_tag("p")
                caption["style"] = styles.get("img_caption", "")
                caption.string = alt
                wrapper.append(caption)
        elif alt:
            caption = soup.new_tag("p")
            caption["style"] = styles.get("img_caption", "")
            caption.string = alt
            tag.insert_after(caption)

    for tag in soup.find_all("hr"):
        tag["style"] = styles.get("hr", "")

    for tag in soup.find_all("pre"):
        tag["style"] = styles.get("pre", "")
    for tag in soup.find_all("code"):
        if tag.parent and tag.parent.name != "pre":
            tag["style"] = styles.get("code_inline", "")


def _transform_tables(soup: BeautifulSoup, styles: dict) -> None:
    """2列表格 → 数据对比 flexbox 卡片（表头行自动隐藏）。
    Markdown 写法:
        | 项目 | 金额 |
        |---|---|
        | 1.42GW电厂EPC | 13亿美元 / 90亿人民币 |
    """
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) == 2:
                label = tds[0].get_text(strip=True)
                value = tds[1].get_text(strip=True)
                if label:
                    rows.append((label, value))

        if not rows:
            continue

        wrapper = soup.new_tag("section")
        wrapper["style"] = styles["data_table_wrapper"]

        for i, (label, value) in enumerate(rows):
            row = soup.new_tag("section")
            is_last = i == len(rows) - 1
            row["style"] = styles["data_table_row_last" if is_last else "data_table_row"]

            label_span = soup.new_tag("span")
            label_span["style"] = styles["data_table_label"]
            label_span.string = label

            value_span = soup.new_tag("span")
            value_span["style"] = styles["data_table_value"]
            value_span.string = value

            row.append(label_span)
            row.append(value_span)
            wrapper.append(row)

        table.replace_with(wrapper)


def _render_summary_header(soup: BeautifulSoup, text: str, styles: dict) -> Tag:
    """模板B：SUMMARY 边框标题框。"""
    outer = soup.new_tag("section")
    outer["style"] = "margin: 10px auto;"

    top_line = soup.new_tag("section")
    top_line["style"] = styles.get("summary_top_line", "width:100%;height:3px;background-color:rgb(240,247,252);")
    top_line.append(soup.new_tag("br"))
    outer.append(top_line)

    box = soup.new_tag("section")
    box["style"] = styles.get("summary_border_box", "border:1px solid rgba(47,105,255,0.5);padding:6px 12px;")

    inner = soup.new_tag("section")
    inner["style"] = "text-align:justify;letter-spacing:1.1px;color:rgb(51,51,51);background-color:transparent;"

    p = soup.new_tag("p")
    p["style"] = styles.get("summary_title_text", "text-align:center;font-style:italic;font-size:20px;color:rgb(51,51,51);")
    strong = soup.new_tag("strong")
    em_tag = soup.new_tag("em")
    em_tag.string = text
    strong.append(em_tag)
    p.append(strong)
    inner.append(p)
    box.append(inner)
    outer.append(box)

    label_row = soup.new_tag("section")
    label_row["style"] = "display:flex;justify-content:flex-end;"
    label_inner = soup.new_tag("section")
    label_inner["style"] = "margin-right:20px;"
    label_span = soup.new_tag("section")
    label_span["style"] = styles.get("summary_label", "font-size:12px;color:rgb(47,105,255);text-align:center;padding:0 6px;font-style:italic;")
    em2 = soup.new_tag("em")
    em2.string = "摘要 SUMMARY"
    label_span.append(em2)
    label_inner.append(label_span)
    label_row.append(label_inner)
    outer.append(label_row)

    return outer


def _render_part_header(soup: BeautifulSoup, text: str, styles: dict, part_index: int) -> Tag:
    """模板C：Part 0X 药丸标题。"""
    outer = soup.new_tag("section")
    outer["style"] = (
        "margin-top:48px;display:flex;flex-direction:column;"
        "align-items:center;justify-content:center;"
    )

    num = soup.new_tag("span")
    num["style"] = styles.get(
        "part_number",
        "display:block;margin-bottom:11px;font-size:24px;font-weight:600;line-height:100%;color:#8178C8;text-align:center;",
    )
    num.string = f"Part {part_index:02d}"
    outer.append(num)

    title = soup.new_tag("span")
    title["style"] = styles.get(
        "part_title",
        "display:inline-block;padding:0 18px;font-size:22px;font-weight:700;line-height:200%;background-color:#8178C8;color:#FFFFFF;border-radius:3px;",
    )
    title.string = text
    outer.append(title)

    return outer


def _render_diamond_header(soup: BeautifulSoup, text: str, styles: dict) -> Tag:
    """模板D：小方块 + 文字 + 双线下划。"""
    outer = soup.new_tag("section")
    outer["style"] = "margin-top:48px;margin-bottom:24px;display:flex;flex-direction:column;"

    top_row = soup.new_tag("section")
    top_row["style"] = "display:flex;align-items:flex-end;"

    square = soup.new_tag("section")
    square["style"] = styles.get(
        "diamond_square",
        "width:14px;height:14px;background-color:rgb(106,158,206);flex-shrink:0;margin-bottom:6px;",
    )
    square.append(soup.new_tag("br"))
    top_row.append(square)

    title = soup.new_tag("section")
    title["style"] = styles.get(
        "diamond_title",
        "margin-left:12px;font-size:22px;font-weight:700;color:rgb(37,37,37);line-height:1.25;letter-spacing:1.1px;",
    )
    title.string = text
    top_row.append(title)
    outer.append(top_row)

    line_row = soup.new_tag("section")
    line_row["style"] = "display:flex;align-items:center;margin-top:8px;"

    full_line = soup.new_tag("section")
    full_line["style"] = styles.get(
        "diamond_line_full",
        "width:100%;height:2px;background-color:rgb(106,158,206);opacity:0.15;",
    )
    full_line.append(soup.new_tag("br"))
    line_row.append(full_line)

    accent_line = soup.new_tag("section")
    accent_line["style"] = styles.get(
        "diamond_line_accent",
        "width:40px;height:2px;background-color:rgb(106,158,206);margin-left:-40px;",
    )
    accent_line.append(soup.new_tag("br"))
    line_row.append(accent_line)
    outer.append(line_row)

    return outer


def _group_sections(soup: BeautifulSoup, styles: dict, features: dict | None = None) -> None:
    """将每个 H2 及其后续内容包裹进 section，并按模板渲染节标题。"""
    if features is None:
        features = {}

    if not soup.find("h2"):
        return

    children = list(soup.children)
    pre_h2: list = []
    groups: list = []
    current_h2 = None
    current_items: list = []
    part_counter = 0

    for child in children:
        node = child.extract()
        if getattr(node, "name", None) == "h2":
            if current_h2 is not None:
                groups.append((current_h2, current_items, part_counter))
            part_counter += 1
            current_h2 = node
            current_items = []
        else:
            if current_h2 is None:
                pre_h2.append(node)
            else:
                current_items.append(node)

    if current_h2 is not None:
        groups.append((current_h2, current_items, part_counter))

    for item in pre_h2:
        soup.append(item)

    for h2, items, idx in groups:
        h2_text = h2.get_text(strip=True)

        group = soup.new_tag("section")
        group["style"] = styles.get("section_group", "margin-top: 38px;")

        if features.get("summary_headers"):
            heading = _render_summary_header(soup, h2_text, styles)
        elif features.get("part_headers"):
            heading = _render_part_header(soup, h2_text, styles, idx)
        elif features.get("diamond_headers"):
            heading = _render_diamond_header(soup, h2_text, styles)
        else:
            # 模板 A 默认：section 继承 h2 的 style
            heading = soup.new_tag("section")
            heading["style"] = h2.get("style", styles.get("h2", ""))
            for child in list(h2.children):
                heading.append(child)

        group.append(heading)
        for item in items:
            group.append(item)
        soup.append(group)


def convert_markdown_to_wechat_html(md_content: str, template: str = DEFAULT_TEMPLATE) -> str:
    """将 Markdown 转换为微信公众号兼容 HTML。

    Args:
        md_content: Markdown 正文
        template: 模板名称，如 "A"（默认）。可用模板见 render/templates.py。
    """
    tmpl = get_template(template)
    styles = tmpl["styles"]
    features = tmpl["features"]

    extensions = ["tables", "fenced_code", "nl2br", "sane_lists"]
    html = markdown.markdown(md_content, extensions=extensions)
    soup = BeautifulSoup(html, "html.parser")

    _apply_styles(soup, styles)

    if features.get("data_tables"):
        _transform_tables(soup, styles)

    if features.get("section_groups"):
        _group_sections(soup, styles, features)

    # h1/h3/h4 → styled p（WeChat 不识别 h 系列标签）
    for tag in soup.find_all(["h1", "h3", "h4"]):
        tag.name = "p"

    return f'<section style="{styles["body_wrapper"]}">{soup}</section>'


def save_html(html_content: str, output_path: str) -> None:
    from pathlib import Path
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>预览</title>
</head>
<body style="max-width: 640px; margin: 0 auto; background: #ffffff;">
{html_content}
</body>
</html>""",
        encoding="utf-8",
    )
