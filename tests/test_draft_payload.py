import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from wechat.draft import build_draft_payload


def test_payload_has_articles_key():
    payload = build_draft_payload("标题", "作者", "摘要", "<p>内容</p>", "fake_media_id")
    assert "articles" in payload


def test_payload_title_not_empty():
    payload = build_draft_payload("测试标题", "作者", "摘要", "<p>内容</p>", "media123")
    article = payload["articles"][0]
    assert article["title"] == "测试标题"


def test_payload_content_not_empty():
    payload = build_draft_payload("标题", "作者", "摘要", "<p>内容</p>", "media123")
    article = payload["articles"][0]
    assert article["content"] == "<p>内容</p>"


def test_payload_thumb_media_id_not_empty():
    payload = build_draft_payload("标题", "作者", "摘要", "<p>内容</p>", "media123")
    article = payload["articles"][0]
    assert article["thumb_media_id"] == "media123"


def test_digest_truncated_at_120():
    long_digest = "摘" * 200
    payload = build_draft_payload("标题", "作者", long_digest, "<p>内容</p>", "media123")
    article = payload["articles"][0]
    assert len(article["digest"]) <= 120


def test_digest_length_normal():
    digest = "这是一段正常长度的摘要，不超过120字。"
    payload = build_draft_payload("标题", "作者", digest, "<p>内容</p>", "media123")
    article = payload["articles"][0]
    assert article["digest"] == digest


def test_payload_is_dict():
    payload = build_draft_payload("T", "A", "D", "<p>C</p>", "M")
    assert isinstance(payload, dict)
    assert isinstance(payload["articles"], list)
    assert len(payload["articles"]) == 1


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
