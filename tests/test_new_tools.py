import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from ai.content_checker import check_article
from config.settings import get_enable_auto_publish
from wechat.draft import build_draft_payload, _DRAFT_BATCHGET_URL, _DRAFT_UPDATE_URL, _DRAFT_DELETE_URL
from wechat.errors import is_rate_limit_error


# --- content checker ---

def test_validate_content_short_article():
    result = check_article({"title": "测试", "digest": "", "markdown": "太短了"})
    assert not result.passed
    assert any("太短" in e or "800" in e for e in result.errors)


def test_validate_content_empty_title():
    result = check_article({"title": "", "digest": "", "markdown": "内容" * 400})
    assert not result.passed
    assert any("标题" in e for e in result.errors)


def test_validate_content_ai_phrase():
    long_body = "这是一篇正常内容的文章。" * 100
    result = check_article({
        "title": "标题",
        "digest": "摘要",
        "markdown": long_body + "作为一个AI，我认为……",
    })
    assert any("AI" in w for w in result.warnings)


def test_validate_content_placeholder():
    long_body = "这是一篇正常内容的文章。" * 100
    result = check_article({
        "title": "标题",
        "digest": "摘要",
        "markdown": long_body + "数据待补充",
    })
    assert any("占位符" in w for w in result.warnings)


def test_validate_content_passes():
    result = check_article({
        "title": "正常标题",
        "digest": "正常摘要",
        "markdown": "这是一段内容丰富的段落。" * 80,
    })
    assert result.passed
    assert result.errors == []


# --- rate limit detection ---

def test_rate_limit_detected():
    assert is_rate_limit_error({"errcode": 45009, "errmsg": "api freq out of limit"})


def test_no_rate_limit_on_ok():
    assert not is_rate_limit_error({"errcode": 0, "errmsg": "ok"})


def test_no_rate_limit_on_other_error():
    assert not is_rate_limit_error({"errcode": 40001, "errmsg": "invalid token"})


# --- ENABLE_AUTO_PUBLISH setting ---

def test_auto_publish_default_false(monkeypatch):
    monkeypatch.delenv("ENABLE_AUTO_PUBLISH", raising=False)
    assert get_enable_auto_publish() is False


def test_auto_publish_true(monkeypatch):
    monkeypatch.setenv("ENABLE_AUTO_PUBLISH", "true")
    assert get_enable_auto_publish() is True


def test_auto_publish_case_insensitive(monkeypatch):
    monkeypatch.setenv("ENABLE_AUTO_PUBLISH", "TRUE")
    assert get_enable_auto_publish() is True


def test_auto_publish_false_string(monkeypatch):
    monkeypatch.setenv("ENABLE_AUTO_PUBLISH", "false")
    assert get_enable_auto_publish() is False


# --- draft API URL constants ---

def test_draft_url_constants():
    assert "draft/batchget" in _DRAFT_BATCHGET_URL
    assert "draft/update" in _DRAFT_UPDATE_URL
    assert "draft/delete" in _DRAFT_DELETE_URL
    assert _DRAFT_BATCHGET_URL.startswith("https://api.weixin.qq.com")
    assert _DRAFT_UPDATE_URL.startswith("https://api.weixin.qq.com")
    assert _DRAFT_DELETE_URL.startswith("https://api.weixin.qq.com")


# --- build_draft_payload with digest truncation ---

def test_digest_truncated_at_120():
    long_digest = "摘" * 200
    payload = build_draft_payload("标题", "作者", long_digest, "<p>内容</p>", "media123")
    assert len(payload["articles"][0]["digest"]) <= 120


def test_draft_payload_structure():
    payload = build_draft_payload("T", "A", "D", "<p>C</p>", "M")
    assert isinstance(payload, dict)
    assert isinstance(payload["articles"], list)
    article = payload["articles"][0]
    assert article["title"] == "T"
    assert article["author"] == "A"
    assert article["content"] == "<p>C</p>"
    assert article["thumb_media_id"] == "M"
