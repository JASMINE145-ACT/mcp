import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from render.markdown_to_html import convert_markdown_to_wechat_html


def test_heading_conversion():
    html = convert_markdown_to_wechat_html("## 标题测试")
    assert "<section" in html
    assert "标题测试" in html


def test_bold_conversion():
    html = convert_markdown_to_wechat_html("这是**加粗**文字")
    assert "<strong" in html
    assert "加粗" in html


def test_unordered_list():
    md = "- 第一项\n- 第二项\n- 第三项"
    html = convert_markdown_to_wechat_html(md)
    assert "<ul" in html
    assert "<li" in html


def test_ordered_list():
    md = "1. 第一步\n2. 第二步"
    html = convert_markdown_to_wechat_html(md)
    assert "<ol" in html
    assert "<li" in html


def test_table_conversion():
    md = "| 公司 | 产品 |\n|------|------|\n| A | X |"
    html = convert_markdown_to_wechat_html(md)
    assert "<table" not in html
    assert "公司" not in html
    assert "A" in html
    assert "X" in html


def test_image_conversion():
    md = "![图注文字](https://example.com/img.jpg)"
    html = convert_markdown_to_wechat_html(md)
    assert "<img" in html


def test_html_not_empty():
    html = convert_markdown_to_wechat_html("# 标题\n\n正文内容。")
    assert html and len(html) > 0


def test_inline_style_present():
    html = convert_markdown_to_wechat_html("## 小标题")
    assert "style=" in html


def test_blockquote():
    html = convert_markdown_to_wechat_html("> 这是引用块")
    assert "<blockquote" in html


def test_no_external_css():
    html = convert_markdown_to_wechat_html("正文")
    assert "<link" not in html
    assert "stylesheet" not in html


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"✅ {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
