"""Lightweight models for the standalone MCP migration."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class Item:
    item_no: str
    item_name: str
    item_type: str = ""
    unit: str = ""
    qty_warehouse: float = 0.0
    qty_available: float = 0.0

    @property
    def code(self) -> str:
        return self.item_no

    @property
    def name(self) -> str:
        return self.item_name


@dataclass
class QueryIntent:
    keywords: str
    strategy: Literal["keywords", "code"] = "keywords"
    confidence: float = 1.0
    keywords_list: Optional[List[str]] = None
    phrase_specs: Optional[List[List[str]]] = None


@dataclass
class InventoryQueryResult:
    items: list[Item] = field(default_factory=list)
    total_count: int = 0
    has_more: bool = False
    query_time_ms: Optional[int] = None


__all__ = ["Item", "QueryIntent", "InventoryQueryResult"]
