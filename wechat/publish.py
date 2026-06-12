import os
import requests
import typer
from loguru import logger

from wechat._session import wx_session
from wechat.token import get_access_token
from wechat.errors import raise_if_error, WeChatAPIError

_FREEPUBLISH_SUBMIT_URL = "https://api.weixin.qq.com/cgi-bin/freepublish/submit"
_FREEPUBLISH_GET_URL = "https://api.weixin.qq.com/cgi-bin/freepublish/getarticle"


def publish_draft(media_id: str) -> dict:
    """Publish a draft. Only works when ENABLE_AUTO_PUBLISH=true and user confirms."""
    from config.settings import get_settings
    s = get_settings()

    if not s.ENABLE_AUTO_PUBLISH:
        raise RuntimeError(
            "自动发布功能已关闭。\n"
            "如需启用，请在 .env 中设置 ENABLE_AUTO_PUBLISH=true，并在命令行二次确认。"
        )

    confirmed = typer.confirm(
        f"⚠️  即将发布草稿 {media_id} 到公众号，此操作不可撤销。确认发布？",
        default=False,
    )
    if not confirmed:
        logger.info("用户取消发布")
        return {"status": "cancelled"}

    token = get_access_token()
    resp = wx_session.post(
        _FREEPUBLISH_SUBMIT_URL,
        params={"access_token": token},
        json={"media_id": media_id},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    raise_if_error(data)

    publish_id = data.get("publish_id")
    logger.info(f"发布任务已提交，publish_id: {publish_id}")
    return {"publish_id": publish_id, "status": "submitted"}


def publish_from_mcp(media_id: str) -> dict:
    """MCP-safe publish: reads ENABLE_AUTO_PUBLISH env flag, no interactive prompt."""
    enabled = os.environ.get("ENABLE_AUTO_PUBLISH", "false").strip().lower() == "true"
    if not enabled:
        return {
            "status": "blocked",
            "message": (
                "自动发布已关闭。如需启用，在 .env 中设置 ENABLE_AUTO_PUBLISH=true，"
                "重启 MCP server 后再调用此工具。"
            ),
        }
    token = get_access_token()
    resp = wx_session.post(
        _FREEPUBLISH_SUBMIT_URL,
        params={"access_token": token},
        json={"media_id": media_id},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    raise_if_error(data)
    publish_id = data.get("publish_id")
    logger.info(f"发布任务已提交，publish_id: {publish_id}")
    return {"publish_id": publish_id, "status": "submitted"}


def get_publish_status(publish_id: str) -> dict:
    token = get_access_token()
    resp = wx_session.post(
        _FREEPUBLISH_GET_URL,
        params={"access_token": token},
        json={"publish_id": publish_id},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    raise_if_error(data)
    return data
