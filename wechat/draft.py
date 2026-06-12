import json
import requests
from loguru import logger

from wechat._session import wx_session
from wechat.token import get_access_token
from wechat.errors import WeChatDraftError, raise_if_error

_DRAFT_ADD_URL = "https://api.weixin.qq.com/cgi-bin/draft/add"
_DRAFT_BATCHGET_URL = "https://api.weixin.qq.com/cgi-bin/draft/batchget"
_DRAFT_UPDATE_URL = "https://api.weixin.qq.com/cgi-bin/draft/update"
_DRAFT_DELETE_URL = "https://api.weixin.qq.com/cgi-bin/draft/delete"
_MAX_DIGEST_LENGTH = 120


def create_draft(
    title: str,
    author: str,
    digest: str,
    content_html: str,
    thumb_media_id: str,
    source_url: str = "",
    need_open_comment: int = 0,
    only_fans_can_comment: int = 0,
) -> str:
    """Create a WeChat draft. Returns media_id."""
    if not title or not title.strip():
        raise ValueError("标题不能为空")
    if not content_html or not content_html.strip():
        raise ValueError("HTML 正文不能为空")
    if not thumb_media_id or not thumb_media_id.strip():
        raise ValueError("thumb_media_id（封面素材ID）不能为空")

    if len(digest) > _MAX_DIGEST_LENGTH:
        digest = digest[:_MAX_DIGEST_LENGTH]
        logger.warning(f"摘要超过 {_MAX_DIGEST_LENGTH} 字，已自动截断")

    token = get_access_token()
    payload = {
        "articles": [
            {
                "title": title,
                "author": author,
                "digest": digest,
                "content": content_html,
                "content_source_url": source_url or "",
                "thumb_media_id": thumb_media_id,
                "need_open_comment": need_open_comment,
                "only_fans_can_comment": only_fans_can_comment,
            }
        ]
    }

    logger.info(f"创建公众号草稿：{title}")
    resp = wx_session.post(
        _DRAFT_ADD_URL,
        params={"access_token": token},
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    logger.debug(f"微信草稿 API 响应: {data}")
    raise_if_error(data, WeChatDraftError)

    media_id = data.get("media_id")
    logger.info(f"草稿创建成功，media_id: {media_id}")
    return media_id


def list_drafts(offset: int = 0, count: int = 10) -> dict:
    """List drafts. Returns raw WeChat API response dict with 'item' list."""
    token = get_access_token()
    resp = wx_session.post(
        _DRAFT_BATCHGET_URL,
        params={"access_token": token},
        json={"offset": offset, "count": min(count, 20), "no_content": 0},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    raise_if_error(data, WeChatDraftError)
    return data


def update_draft(
    media_id: str,
    title: str,
    content_html: str,
    thumb_media_id: str,
    author: str = "",
    digest: str = "",
    source_url: str = "",
) -> None:
    """Update an existing draft by media_id (replaces index 0)."""
    if len(digest) > _MAX_DIGEST_LENGTH:
        digest = digest[:_MAX_DIGEST_LENGTH]
        logger.warning(f"摘要超过 {_MAX_DIGEST_LENGTH} 字，已自动截断")
    token = get_access_token()
    payload = {
        "media_id": media_id,
        "index": 0,
        "articles": {
            "title": title,
            "author": author,
            "digest": digest,
            "content": content_html,
            "content_source_url": source_url or "",
            "thumb_media_id": thumb_media_id,
            "need_open_comment": 0,
            "only_fans_can_comment": 0,
        },
    }
    resp = wx_session.post(
        _DRAFT_UPDATE_URL,
        params={"access_token": token},
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    raise_if_error(data, WeChatDraftError)
    logger.info(f"草稿更新成功: {media_id}")


def delete_draft(media_id: str) -> None:
    """Permanently delete a draft by media_id."""
    token = get_access_token()
    resp = wx_session.post(
        _DRAFT_DELETE_URL,
        params={"access_token": token},
        json={"media_id": media_id},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    raise_if_error(data, WeChatDraftError)
    logger.info(f"草稿删除成功: {media_id}")


def build_draft_payload(
    title: str,
    author: str,
    digest: str,
    content_html: str,
    thumb_media_id: str,
    source_url: str = "",
) -> dict:
    """Build draft payload dict for testing without actually calling the API."""
    if len(digest) > _MAX_DIGEST_LENGTH:
        digest = digest[:_MAX_DIGEST_LENGTH]
    return {
        "articles": [
            {
                "title": title,
                "author": author,
                "digest": digest,
                "content": content_html,
                "content_source_url": source_url or "",
                "thumb_media_id": thumb_media_id,
                "need_open_comment": 0,
                "only_fans_can_comment": 0,
            }
        ]
    }
