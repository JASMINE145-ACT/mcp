"""RSS fetcher — Phase 2 feature. Stub implementation."""
from dataclasses import dataclass
from typing import List


@dataclass
class SourceArticle:
    title: str
    url: str
    summary: str
    content: str
    published_at: str
    source: str


def fetch_articles_from_rss(feed_url: str, limit: int = 10) -> List[SourceArticle]:
    raise NotImplementedError("RSS fetcher is a Phase 2 feature. Not yet implemented.")
