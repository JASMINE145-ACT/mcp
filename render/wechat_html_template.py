def wrap_for_wechat(content_html: str) -> str:
    """Strip outer HTML wrapper for WeChat draft API — only body content is accepted."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(content_html, "html.parser")
    section = soup.find("section")
    if section:
        return str(section)
    return content_html
