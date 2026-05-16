"""Accurate Online API client - 与 agent-jk/Agent Team version3 保持一致。"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import requests

# 自动加载项目根目录的 .env.accurate（包含 AOL_* 凭证）
_env_path = Path(__file__).resolve().parents[2] / ".env.accurate"
if _env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_path, override=True)
    except Exception:
        pass

logger = logging.getLogger(__name__)


def _generate_timestamp_and_signature(secret: str) -> Tuple[str, str]:
    """生成 X-Api-Timestamp 与 X-Api-Signature。"""
    timestamp = str(int(time.time()))
    signature = hmac.new(
        secret.encode("utf-8"),
        timestamp.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return timestamp, signature


class AccurateOnlineAPIClient:
    """
    Accurate Online API 客户端（与 agent-jk/Agent Team version3 同步）。
    认证方式：Bearer token + HMAC 签名（时间戳 + secret）。
    数据库特定端点用 {database_id}.accurate.id/accurate 前缀。
    """

    def __init__(
        self,
        access_token: Optional[str] = None,
        database_id: Optional[str] = None,
        signature_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 30,
    ):
        import os

        self.base_url = (
            base_url
            or os.environ.get("AOL_API_BASE_URL")
            or "https://account.accurate.id"
        ).rstrip("/")
        self.database_id = database_id or os.environ.get("AOL_DATABASE_ID") or ""
        self.access_token = access_token or os.environ.get("AOL_ACCESS_TOKEN") or ""
        self.signature_secret = signature_secret or os.environ.get("AOL_SIGNATURE_SECRET") or ""
        self.timeout = timeout

        if not self.access_token:
            logger.warning("AccurateOnlineAPIClient: 未配置 AOL_ACCESS_TOKEN，API 调用将失败")
        if not self.signature_secret:
            logger.warning("AccurateOnlineAPIClient: 未配置 AOL_SIGNATURE_SECRET，API 调用将失败")

    def _headers(self) -> dict[str, str]:
        timestamp, signature = _generate_timestamp_and_signature(self.signature_secret)
        return {
            "Authorization": f"Bearer {self.access_token}",
            "X-Api-Timestamp": timestamp,
            "X-Api-Signature": signature,
            "Accept": "application/json",
        }

    def _url(self, endpoint: str, use_database_url: bool = False) -> str:
        """构建请求 URL。"""
        if use_database_url and self.database_id:
            return f"https://{self.database_id}.accurate.id/accurate{endpoint}"
        return f"{self.base_url}{endpoint}"

    def get(
        self,
        endpoint: str,
        params: Optional[dict[str, Any]] = None,
        use_database_url: bool = False,
        timeout: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        GET 请求，用于 detail.do 等端点。
        use_database_url=True 时 database_id 作为子域名。
        """
        if not self.access_token:
            return {"s": False, "d": {"message": "Accurate Online API 未配置 access_token"}}

        url = self._url(endpoint, use_database_url=use_database_url)
        headers = self._headers()
        req_timeout = timeout if timeout is not None else self.timeout

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=req_timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"AOL API HTTP 错误 {e.response.status_code}: {e.response.text[:300]}")
            try:
                return e.response.json()
            except Exception:
                return {"s": False, "d": {"message": f"HTTP {e.response.status_code}"}}
        except requests.exceptions.RequestException as e:
            logger.error(f"AOL API 请求失败: {e}")
            return {"s": False, "d": {"message": f"请求失败: {e}"}}

    def get_table_data(
        self,
        table_name: str,
        params: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        调用 list.do 接口（如 /api/item/list.do）。
        数据库特定端点，database_id 作为子域名。
        返回格式: {"s": True/False, "d": [...]}
        """
        endpoint = f"/api/{table_name}/list.do"
        return self.get(endpoint, params=params, use_database_url=True, timeout=timeout)

    def post(
        self,
        endpoint: str,
        data: Optional[dict[str, Any]] = None,
        use_database_url: bool = False,
        timeout: Optional[int] = None,
    ) -> dict[str, Any]:
        """通用 POST 请求（用于写入操作）。"""
        if not self.access_token:
            return {"s": False, "d": {"message": "Accurate Online API 未配置 access_token"}}

        url = self._url(endpoint, use_database_url=use_database_url)
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        req_timeout = timeout if timeout is not None else self.timeout

        try:
            resp = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=req_timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"AOL API POST 失败: {e}")
            return {"s": False, "d": {"message": f"POST 失败: {e}"}}