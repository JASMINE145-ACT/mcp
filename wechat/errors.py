WECHAT_ERROR_CODES = {
    40001: "access_token 无效或已过期",
    40002: "不合法的凭证类型",
    40003: "不合法的 OpenID",
    40013: "不合法的 AppID",
    40125: "不合法的 AppSecret",
    40164: "调用接口的 IP 不在白名单中，请在公众号后台添加服务器 IP",
    41001: "缺少 access_token 参数",
    42001: "access_token 已过期",
    45009: "接口调用超过限制",
    48001: "API 功能未授权（需要开通相应接口）",
    50001: "用户未授权该 API",
}


class WeChatAPIError(Exception):
    def __init__(self, errcode: int, errmsg: str):
        self.errcode = errcode
        self.errmsg = errmsg
        known = WECHAT_ERROR_CODES.get(errcode, "")
        hint = f"（{known}）" if known else ""
        super().__init__(f"微信 API 错误 {errcode}{hint}: {errmsg}")


class WeChatTokenError(WeChatAPIError):
    pass


class WeChatUploadError(WeChatAPIError):
    pass


class WeChatDraftError(WeChatAPIError):
    pass


def raise_if_error(response_json: dict, error_cls=WeChatAPIError) -> None:
    errcode = response_json.get("errcode", 0)
    if errcode != 0:
        errmsg = response_json.get("errmsg", "unknown error")
        raise error_cls(errcode, errmsg)


def is_rate_limit_error(response_json: dict) -> bool:
    """Returns True if the response indicates a WeChat API rate limit (errcode 45009)."""
    return response_json.get("errcode") == 45009
