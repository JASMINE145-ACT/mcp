import os
import requests
from pathlib import Path
from loguru import logger

from wechat._session import wx_session
from wechat.token import get_access_token
from wechat.errors import WeChatUploadError, raise_if_error

_UPLOAD_PERMANENT_URL = "https://api.weixin.qq.com/cgi-bin/material/add_material"
_UPLOAD_ARTICLE_IMG_URL = "https://api.weixin.qq.com/cgi-bin/media/uploadimg"

ALLOWED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png"}


def _validate_image(image_path: str) -> Path:
    p = Path(image_path)
    if not p.exists():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    if p.suffix.lower() not in ALLOWED_IMAGE_FORMATS:
        raise ValueError(f"不支持的图片格式: {p.suffix}（仅支持 jpg/jpeg/png）")
    return p


def upload_permanent_image(image_path: str) -> str:
    """Upload a permanent material (for cover). Returns media_id."""
    p = _validate_image(image_path)
    token = get_access_token()

    logger.info(f"上传永久素材（封面图）: {p.name}")
    with open(p, "rb") as f:
        mime = "image/jpeg" if p.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        resp = wx_session.post(
            _UPLOAD_PERMANENT_URL,
            params={"access_token": token, "type": "image"},
            files={"media": (p.name, f, mime)},
            timeout=30,
        )
    resp.raise_for_status()
    data = resp.json()
    raise_if_error(data, WeChatUploadError)

    media_id = data.get("media_id")
    logger.info(f"封面图上传成功，media_id: {media_id}")
    return media_id


def upload_article_image(image_path: str) -> str:
    """Upload an image for article body. Returns URL for embedding in HTML."""
    p = _validate_image(image_path)
    token = get_access_token()

    logger.info(f"上传正文图片: {p.name}")
    with open(p, "rb") as f:
        mime = "image/jpeg" if p.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        resp = wx_session.post(
            _UPLOAD_ARTICLE_IMG_URL,
            params={"access_token": token},
            files={"media": (p.name, f, mime)},
            timeout=30,
        )
    resp.raise_for_status()
    data = resp.json()
    raise_if_error(data, WeChatUploadError)

    url = data.get("url")
    logger.info(f"正文图片上传成功，URL: {url}")
    return url
