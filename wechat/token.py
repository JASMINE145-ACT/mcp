import time
import requests
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from wechat._session import wx_session
from wechat.errors import WeChatTokenError, raise_if_error

_WECHAT_TOKEN_URL = "https://api.weixin.qq.com/cgi-bin/token"

_cache: dict = {"token": None, "expires_at": 0.0}


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _fetch_token(app_id: str, app_secret: str) -> dict:
    resp = wx_session.get(
        _WECHAT_TOKEN_URL,
        params={"grant_type": "client_credential", "appid": app_id, "secret": app_secret},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def get_access_token(app_id: str = None, app_secret: str = None) -> str:
    from config.settings import get_wechat_app_id, get_wechat_app_secret

    if app_id is None or app_secret is None:
        app_id = get_wechat_app_id()
        app_secret = get_wechat_app_secret()

    now = time.time()
    if _cache["token"] and _cache["expires_at"] > now + 60:
        logger.debug("Using cached access_token")
        return _cache["token"]

    logger.info("Fetching new WeChat access_token")
    try:
        data = _fetch_token(app_id, app_secret)
    except requests.RequestException as e:
        raise WeChatTokenError(0, f"网络请求失败: {e}") from e

    raise_if_error(data, WeChatTokenError)

    token = data["access_token"]
    expires_in = data.get("expires_in", 7200)

    _cache["token"] = token
    _cache["expires_at"] = now + expires_in
    logger.info(f"access_token 获取成功，有效期 {expires_in}s")
    return token


def clear_token_cache() -> None:
    _cache["token"] = None
    _cache["expires_at"] = 0.0
