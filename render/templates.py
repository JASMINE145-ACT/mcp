"""
文章排版模板库。
使用方式：convert_markdown_to_wechat_html(md, template="A")

模板列表：
  A — 蓝色商业分析风（参考：千岛建筑观察）
      · 主色 rgb(0,102,204)，首行缩进，H2 左边框，数据对比 flexbox
"""

TEMPLATE_A = {
    "name": "A",
    "desc": "蓝色商业分析风（千岛建筑观察）",
    "accent": "rgb(0, 102, 204)",
    "styles": {
        "body_wrapper": (
            "font-family: -apple-system, 'PingFang SC', 'Microsoft YaHei', sans-serif;"
            "font-size: 16px; line-height: 1.85; color: rgb(26, 26, 26);"
        ),
        "p": "margin: 0 0 14px; color: rgb(42, 42, 42); text-indent: 2em;",
        "p_no_indent": "margin: 0 0 14px; color: rgb(42, 42, 42);",
        "h1": (
            "font-size: 20px; font-weight: 700; color: rgb(17, 17, 17);"
            "margin: 0 0 20px; padding-left: 14px;"
            "border-left: 5px solid rgb(0, 102, 204);"
        ),
        "h2": (
            "font-size: 18px; font-weight: 700; color: rgb(17, 17, 17);"
            "margin: 0 0 16px; padding-left: 12px;"
            "border-left: 4px solid rgb(0, 102, 204);"
        ),
        "h3": "font-size: 16px; font-weight: 700; color: rgb(42, 42, 42); margin: 0 0 12px;",
        "section_group": "margin-top: 38px;",
        "strong": "color: rgb(0, 102, 204); font-weight: 700;",
        "em": "font-style: italic; color: rgb(100, 100, 100);",
        "blockquote": (
            "font-size: 17px; font-weight: 700; color: rgb(42, 42, 42);"
            "margin: 0 0 14px; padding: 10px 14px;"
            "background: rgb(240, 247, 255); border-left: 4px solid rgb(0, 102, 204);"
        ),
        "highlight_p": (
            "font-size: 17px; font-weight: 700; color: rgb(42, 42, 42);"
            "margin: 0 0 14px; padding: 10px 14px;"
            "background: rgb(240, 247, 255); border-left: 4px solid rgb(0, 102, 204);"
        ),
        "ul": "padding-left: 20px; margin: 10px 0 16px;",
        "ol": "padding-left: 20px; margin: 10px 0 16px;",
        "li": "margin-bottom: 8px; font-size: 16px; line-height: 1.85; color: rgb(42, 42, 42);",
        "img_wrapper": "margin: 24px 0; text-align: center;",
        "img": (
            "width: 100%; max-width: 100%; display: block; margin: 0 auto;"
            "border-radius: 4px;"
        ),
        "img_caption": (
            "text-align: center; font-size: 13px; color: rgb(153, 153, 153);"
            "margin: 8px 0 0; line-height: 1.6;"
        ),
        "data_table_wrapper": (
            "background: rgb(240, 247, 255); border-radius: 6px;"
            "padding: 16px 18px; margin: 18px 0; font-size: 15px; line-height: 1.7;"
        ),
        "data_table_row": (
            "display: flex; justify-content: space-between;"
            "padding: 6px 0; border-bottom: 1px dashed rgb(204, 224, 255);"
        ),
        "data_table_row_last": "display: flex; justify-content: space-between; padding: 6px 0;",
        "data_table_label": "color: rgb(85, 85, 85);",
        "data_table_value": "font-weight: 700; color: rgb(0, 102, 204);",
        "hr": (
            "border-right: none; border-bottom: none; border-left: none; border-image: initial;"
            "border-top: 1px solid rgb(232, 232, 232); margin: 28px 0;"
        ),
        "code_inline": (
            "background: rgb(240, 247, 255); padding: 2px 6px; border-radius: 3px;"
            "font-family: monospace; font-size: 14px; color: rgb(0, 102, 204);"
        ),
        "pre": (
            "background: rgb(248, 249, 250); padding: 16px; border-radius: 4px;"
            "overflow-x: auto; font-family: monospace; font-size: 14px; margin: 16px 0;"
        ),
    },
    # 模板特性开关
    "features": {
        "section_groups": True,    # H2 段落组包裹（margin-top: 38px）
        "data_tables": True,       # 2列表格 → flexbox 对比卡片
        "text_indent": True,       # 正文首行缩进
        "highlight_p": True,       # 独立 **短句** → callout
    },
}

TEMPLATE_B = {
    "name": "B",
    "desc": "蓝色财经科普风 — PingFang SC，细体，SUMMARY 节标题框",
    "accent": "rgb(47, 105, 255)",
    "styles": {
        "body_wrapper": (
            'font-family: "PingFang SC", "Noto Sans SC Thin", sans-serif;'
            "font-size: 15px; font-weight: 300; line-height: 1.75; color: rgb(0, 0, 0);"
        ),
        "p": (
            "margin: 20px 0 0; color: rgb(0, 0, 0);"
            "font-size: 15px; font-weight: 300; line-height: 1.75; white-space: pre-wrap;"
        ),
        "p_no_indent": (
            "margin: 8px 0 0; color: rgb(37, 37, 37);"
            "font-size: 15px; font-weight: 300; line-height: 1.75;"
        ),
        "h2": "",  # 由 _render_summary_header 接管
        "h3": "margin: 16px 0 0; color: #1f1f1f; font-size: 15px; font-weight: 700; line-height: 1.75;",
        "section_group": "margin: 20px 0;",
        "strong": "font-weight: 600; color: rgb(47, 105, 255);",
        "em": "font-style: italic; color: rgb(100, 100, 100);",
        "blockquote": (
            "background-color: rgba(47, 105, 255, 0.063); border-radius: 10px;"
            "padding: 15px; margin: 20px 0 0; color: rgb(37, 37, 37);"
        ),
        "ul": "padding-left: 20px; margin: 10px 0 16px;",
        "ol": "padding-left: 20px; margin: 10px 0 16px;",
        "li": "margin-bottom: 8px; font-size: 15px; line-height: 1.75; color: rgb(0, 0, 0);",
        "img_wrapper": "margin: 24px 0; text-align: center;",
        "img": "width: 100%; max-width: 100%; display: block; margin: 0 auto; border-radius: 4px;",
        "img_caption": (
            "text-align: center; font-size: 13px; color: rgb(153, 153, 153); margin: 8px 0 0; line-height: 1.6;"
        ),
        "data_table_wrapper": (
            "background: rgba(47, 105, 255, 0.06); border-radius: 10px;"
            "padding: 16px 18px; margin: 18px 0; font-size: 15px; line-height: 1.75;"
        ),
        "data_table_row": (
            "display: flex; justify-content: space-between;"
            "padding: 6px 0; border-bottom: 1px dashed rgba(47, 105, 255, 0.3);"
        ),
        "data_table_row_last": "display: flex; justify-content: space-between; padding: 6px 0;",
        "data_table_label": "color: rgb(85, 85, 85); font-size: 15px;",
        "data_table_value": "font-weight: 600; color: rgb(47, 105, 255);",
        "hr": (
            "border-top: 1px solid rgb(240, 247, 252); margin: 28px 0;"
            "border-bottom: none; border-left: none; border-right: none;"
        ),
        "code_inline": (
            "background: rgba(47, 105, 255, 0.08); padding: 2px 6px; border-radius: 3px;"
            "font-family: monospace; font-size: 14px; color: rgb(47, 105, 255);"
        ),
        "pre": (
            "background: rgb(248, 249, 250); padding: 16px; border-radius: 4px;"
            "overflow-x: auto; font-family: monospace; font-size: 14px; margin: 16px 0;"
        ),
        # SUMMARY 标题框专用 style
        "summary_top_line": "width: 100%; height: 3px; background-color: rgb(240, 247, 252);",
        "summary_border_box": "border: 1px solid rgba(47, 105, 255, 0.5); padding: 6px 12px;",
        "summary_title_text": (
            "text-align: center; font-style: italic; font-size: 20px;"
            "color: rgb(51, 51, 51); letter-spacing: 1.1px; line-height: 35px;"
        ),
        "summary_label": (
            "font-size: 12px; color: rgb(47, 105, 255);"
            "text-align: center; padding: 0 6px; font-style: italic;"
        ),
    },
    "features": {
        "section_groups": True,
        "data_tables": True,
        "text_indent": False,
        "highlight_p": False,
        "summary_headers": True,
    },
}

TEMPLATE_C = {
    "name": "C",
    "desc": "紫色新闻资讯风 — Part 0X 药丸标题，渐变高亮，引用框",
    "accent": "#8178C8",
    "styles": {
        "body_wrapper": "font-size: 16px; line-height: 200%; color: rgb(0, 0, 0);",
        "p": "margin-top: 28px; font-size: 16px; line-height: 200%;",
        "p_no_indent": "margin-top: 8px; font-size: 16px; line-height: 200%; color: rgb(135, 139, 142);",
        "h2": "",  # 由 _render_part_header 接管
        "h3": "margin-top: 20px; font-size: 16px; font-weight: 600; color: #8178C8;",
        "section_group": "margin-top: 0;",
        "strong": (
            "font-weight: 600; color: #8178C8;"
            "background: linear-gradient(to bottom, transparent 50%, #DCE8FF 80%);"
        ),
        "em": "font-style: italic; color: rgb(100, 100, 100);",
        "blockquote": (
            "padding: 17px 24px 17px 18px; border-left: 3px solid #DCE8FF;"
            "background-color: #FFFFF7; margin-top: 28px; font-size: 16px; line-height: 200%;"
        ),
        "ul": "padding-left: 20px; margin-top: 16px;",
        "ol": "padding-left: 20px; margin-top: 16px;",
        "li": "margin-bottom: 8px; font-size: 16px; line-height: 200%;",
        "img_wrapper": "margin: 28px 0; text-align: center;",
        "img": "width: 100%; max-width: 100%; display: block; margin: 0 auto; border-radius: 4px;",
        "img_caption": "text-align: center; font-size: 13px; color: #878B8E; margin: 8px 0 0;",
        "data_table_wrapper": (
            "background: #F1F4FD; border-radius: 10px;"
            "padding: 16px 18px; margin-top: 28px; font-size: 16px; line-height: 200%;"
        ),
        "data_table_row": (
            "display: flex; justify-content: space-between;"
            "padding: 6px 0; border-bottom: 1px dashed #DCE8FF;"
        ),
        "data_table_row_last": "display: flex; justify-content: space-between; padding: 6px 0;",
        "data_table_label": "color: rgb(85, 85, 85); font-size: 16px;",
        "data_table_value": "font-weight: 600; color: #8178C8;",
        "hr": (
            "border-top: 1px solid rgb(232, 232, 232); margin: 28px 0;"
            "border-bottom: none; border-left: none; border-right: none;"
        ),
        "code_inline": (
            "background: #F1F4FD; padding: 2px 6px; border-radius: 3px;"
            "font-family: monospace; font-size: 14px; color: #8178C8;"
        ),
        "pre": (
            "background: rgb(248, 249, 250); padding: 16px; border-radius: 4px;"
            "overflow-x: auto; font-family: monospace; font-size: 14px; margin-top: 16px;"
        ),
        # Part 标题专用 style
        "part_number": (
            "display: block; margin-bottom: 11px; font-size: 24px; font-weight: 600;"
            "line-height: 100%; color: #8178C8; text-align: center;"
        ),
        "part_title": (
            "display: inline-block; padding: 0 18px; font-size: 22px; font-weight: 700;"
            "line-height: 200%; background-color: #8178C8; color: #FFFFFF; border-radius: 3px;"
        ),
    },
    "features": {
        "section_groups": True,
        "data_tables": True,
        "text_indent": False,
        "highlight_p": False,
        "part_headers": True,
    },
}

TEMPLATE_D = {
    "name": "D",
    "desc": "钢蓝深度评论风 — Noto Sans SC，菱形方块节标题，双线下划",
    "accent": "rgb(106, 158, 206)",
    "styles": {
        "body_wrapper": (
            '"Noto Sans SC", sans-serif;'
            "font-size: 15px; font-weight: 400; line-height: 2; color: rgb(0, 0, 0);"
        ),
        "p": "margin-top: 28px; color: rgb(0, 0, 0); font-size: 15px; font-weight: 400; line-height: 2;",
        "p_no_indent": "margin-top: 8px; color: rgb(0, 0, 0); font-size: 15px; line-height: 2;",
        "h2": "",  # 由 _render_diamond_header 接管
        "h3": (
            "margin: 20px 0 8px; color: rgb(37, 37, 37);"
            "font-size: 16px; font-weight: 600; letter-spacing: 1px;"
        ),
        "section_group": "margin-top: 0;",
        "strong": "font-weight: 600; color: rgb(106, 158, 206);",
        "em": "font-style: italic; color: rgb(100, 100, 100);",
        "blockquote": (
            "margin-top: 28px; padding: 16px 18px;"
            "border-left: 3px solid rgb(106, 158, 206);"
            "background-color: rgba(106, 158, 206, 0.06); font-size: 15px; line-height: 2;"
        ),
        "ul": "padding-left: 20px; margin-top: 16px;",
        "ol": "padding-left: 20px; margin-top: 16px;",
        "li": "margin-bottom: 8px; font-size: 15px; line-height: 2; color: rgb(0, 0, 0);",
        "img_wrapper": "margin: 28px 0; text-align: center;",
        "img": "width: 100%; max-width: 100%; display: block; margin: 0 auto; border-radius: 4px;",
        "img_caption": "text-align: center; font-size: 13px; color: rgb(153, 153, 153); margin: 8px 0 0;",
        "data_table_wrapper": (
            "background: rgba(106, 158, 206, 0.08); border-radius: 8px;"
            "padding: 16px 18px; margin-top: 28px; font-size: 15px; line-height: 2;"
        ),
        "data_table_row": (
            "display: flex; justify-content: space-between;"
            "padding: 6px 0; border-bottom: 1px dashed rgba(106, 158, 206, 0.3);"
        ),
        "data_table_row_last": "display: flex; justify-content: space-between; padding: 6px 0;",
        "data_table_label": "color: rgb(85, 85, 85); font-size: 15px;",
        "data_table_value": "font-weight: 600; color: rgb(106, 158, 206);",
        "hr": (
            "border-top: 1px solid rgb(232, 232, 232); margin: 40px 0;"
            "border-bottom: none; border-left: none; border-right: none;"
        ),
        "code_inline": (
            "background: rgba(106, 158, 206, 0.1); padding: 2px 6px; border-radius: 3px;"
            "font-family: monospace; font-size: 14px; color: rgb(106, 158, 206);"
        ),
        "pre": (
            "background: rgb(248, 249, 250); padding: 16px; border-radius: 4px;"
            "overflow-x: auto; font-family: monospace; font-size: 14px; margin-top: 16px;"
        ),
        # 菱形节标题专用 style
        "diamond_square": (
            "width: 14px; height: 14px; background-color: rgb(106, 158, 206);"
            "flex-shrink: 0; margin-bottom: 6px;"
        ),
        "diamond_title": (
            "margin-left: 12px; font-size: 22px; font-weight: 700;"
            "color: rgb(37, 37, 37); line-height: 1.25; letter-spacing: 1.1px; white-space: pre-wrap;"
        ),
        "diamond_line_full": (
            "width: 100%; height: 2px; background-color: rgb(106, 158, 206); opacity: 0.15;"
        ),
        "diamond_line_accent": (
            "width: 40px; height: 2px; background-color: rgb(106, 158, 206); margin-left: -40px;"
        ),
    },
    "features": {
        "section_groups": True,
        "data_tables": True,
        "text_indent": False,
        "highlight_p": False,
        "diamond_headers": True,
    },
}


# 模板注册表（key = 模板字母）
TEMPLATES: dict[str, dict] = {
    "A": TEMPLATE_A,
    "B": TEMPLATE_B,
    "C": TEMPLATE_C,
    "D": TEMPLATE_D,
}

DEFAULT_TEMPLATE = "A"


def get_template(name: str) -> dict:
    key = name.strip().upper()
    if key not in TEMPLATES:
        available = ", ".join(TEMPLATES.keys())
        raise ValueError(f"模板 '{name}' 不存在，可用模板：{available}")
    return TEMPLATES[key]
