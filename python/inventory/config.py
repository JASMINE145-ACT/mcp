"""Standalone configuration for migrated quotation/inventory tools."""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

PYTHON_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PYTHON_ROOT.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class InventoryConfig:
    LIST_FIELDS: List[str] = ["id", "no"]
    REQUIRED_FIELDS: List[str] = ["id", "no", "name", "type", "unit", "quantityOnHand", "quantityAvailable"]
    API_LIST_ENDPOINT = "/api/item/list.do"
    API_DETAIL_ENDPOINT = "/api/item/detail.do"
    API_TIMEOUT = int(os.environ.get("API_TIMEOUT", "10"))
    API_RETRY_COUNT = int(os.environ.get("API_RETRY_COUNT", "1"))
    MAX_RESULTS = int(os.environ.get("INVENTORY_MAX_RESULTS", "10"))
    MAX_CODES_PER_SEARCH = int(os.environ.get("MAX_CODES_PER_SEARCH", "10"))
    MAX_DETAILS_FOR_AGENT = int(os.environ.get("MAX_DETAILS_FOR_AGENT", "10"))

    PRICE_LIBRARY_PATH = os.environ.get("PRICE_LIBRARY_PATH") or os.environ.get(
        "WANDING_PRICE_LIB_PATH", str(DATA_DIR / "wanding_price_lib.xlsx")
    )
    MAPPING_TABLE_PATH = os.environ.get("MAPPING_TABLE_PATH", str(DATA_DIR / "mapping_table.xlsx"))
    WANDING_BUSINESS_KNOWLEDGE_PATH = os.environ.get(
        "WANDING_BUSINESS_KNOWLEDGE_PATH", str(DATA_DIR / "wanding_business_knowledge.md")
    )

    ENABLE_WANDING_VECTOR = _env_bool("ENABLE_WANDING_VECTOR", False)
    WANDING_VECTOR_TOP_K = int(os.environ.get("WANDING_VECTOR_TOP_K", "3"))
    WANDING_VECTOR_MIN_SCORE = float(os.environ.get("WANDING_VECTOR_MIN_SCORE", "0.65"))
    WANDING_VECTOR_COARSE_MAX = int(os.environ.get("WANDING_VECTOR_COARSE_MAX", "20"))
    USE_RESOLVER_FALLBACK = _env_bool("USE_RESOLVER_FALLBACK", False)

    ITEM_LIST_SLIM_PATH = os.environ.get("INVENTORY_ITEM_LIST_SLIM_PATH", str(DATA_DIR / "item-list-slim.xlsx"))
    RESOLVER_CONTAINS_COLUMNS = os.environ.get("INVENTORY_RESOLVER_CONTAINS", "both")
    ENABLE_RESOLVER_VECTOR = _env_bool("INVENTORY_ENABLE_RESOLVER_VECTOR", False)
    OPENAI_EMBEDDING_MODEL = os.environ.get("INVENTORY_OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    EMBEDDING_TIMEOUT = int(os.environ.get("EMBEDDING_TIMEOUT", "15"))
    RESOLVER_VECTOR_TOP_K = int(os.environ.get("INVENTORY_RESOLVER_VECTOR_TOP_K", "3"))
    RESOLVER_VECTOR_TOP_K_CANDIDATES = int(os.environ.get("INVENTORY_RESOLVER_VECTOR_TOP_K_CANDIDATES", "20"))

    LLM_API_KEY = os.environ.get("INVENTORY_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("ZHIPU_API_KEY") or ""
    LLM_BASE_URL = (os.environ.get("INVENTORY_LLM_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://open.bigmodel.cn/api/paas/v4").rstrip("/") + "/"
    LLM_MODEL = os.environ.get("INVENTORY_LLM_MODEL") or os.environ.get("LLM_MODEL", "glm-4.5-air")
    LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "8192"))
    LLM_SELECTOR_MAX_TOKENS = int(os.environ.get("LLM_SELECTOR_MAX_TOKENS", "3000"))
    LLM_SELECTOR_TIMEOUT = int(os.environ.get("LLM_SELECTOR_TIMEOUT", "40"))
    LLM_SELECTOR_MODEL = os.environ.get("LLM_SELECTOR_MODEL", "").strip()
    LLM_SELECTOR_API_KEY = os.environ.get("LLM_SELECTOR_API_KEY", "").strip()
    LLM_SELECTOR_BASE_URL = os.environ.get("LLM_SELECTOR_BASE_URL", "").strip()
    LLM_SELECTOR_FAST_OUTPUT_TOKENS = int(os.environ.get("LLM_SELECTOR_FAST_OUTPUT_TOKENS", "500"))
    TOOL_RESULT_MAX_CHARS = int(os.environ.get("TOOL_RESULT_MAX_CHARS", "8000"))
    LLM_TIMEOUT = int(os.environ.get("LLM_TIMEOUT", "60"))
    TOOL_EXEC_TIMEOUT = int(os.environ.get("TOOL_EXEC_TIMEOUT", "90"))
    INVENTORY_DEMO_MODE = _env_bool("INVENTORY_DEMO_MODE", False)

    PRICE_LIB_NAME_PATTERNS = ("万鼎", "价格库")
    PRICE_LIB_COL_MATERIAL_KW = os.environ.get("PRICE_LIB_COL_MATERIAL_KW", "Material")
    PRICE_LIB_COL_DESC_KW = os.environ.get("PRICE_LIB_COL_DESC_KW", "Describrition")
    PRICE_LIB_COL_PRICE_A_KW = ("A级别", "报单价格")
    PRICE_LIB_COL_PRICE_B_KW = ("B级别", "报单价格")
    PRICE_LIB_COL_PRICE_C_KW = ("C级别", "报单价格")
    PRICE_LIB_COL_PRICE_D_KW = ("D级别", "报单价格")

    MAPPING_LIB_NAME_PATTERNS = ("整理产品", "映射")
    MAPPING_COL_INQUIRY_KW = os.environ.get("MAPPING_COL_INQUIRY_KW", "询价货物名称")
    MAPPING_COL_SPEC_KW = os.environ.get("MAPPING_COL_SPEC_KW", "询价规格型号")
    MAPPING_COL_CODE_KW = os.environ.get("MAPPING_COL_CODE_KW", "产品编号")
    MAPPING_COL_QUOTATION_KW = os.environ.get("MAPPING_COL_QUOTATION_KW", "报价名称")

    WORK_SINGLE_CAND_USE_LLM = _env_bool("WORK_SINGLE_CAND_USE_LLM", False)
    WORK_MATCH_MAX_WORKERS = int(os.environ.get("WORK_MATCH_MAX_WORKERS", "5"))
    MATCH_QUOTATION_BATCH_MIN_ITEMS = int(os.environ.get("MATCH_QUOTATION_BATCH_MIN_ITEMS", "3"))


Config = InventoryConfig
config = InventoryConfig()
