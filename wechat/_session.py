"""
Shared HTTP session for WeChat API calls.

verify=False is required due to SSLEOFError on POST requests to api.weixin.qq.com
in certain Python/OpenSSL environments (SSL record-size / TLS handshake bug).
The WeChat API itself is legitimate HTTPS; this only disables certificate pinning locally.
"""
import urllib3
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

wx_session = requests.Session()
wx_session.verify = False
