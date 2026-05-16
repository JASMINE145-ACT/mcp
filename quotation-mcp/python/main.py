#!/usr/bin/env python3
"""JSON-lines entry point used by the quotation MCP server."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parent.parent
_env_file = _project_root / ".env.accurate"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file, override=True)
    except Exception:
        pass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PYTHON_ROOT = Path(__file__).resolve().parent
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s", stream=sys.stderr)
logger = logging.getLogger(__name__)


def _wanding_knowledge_path() -> Path:
    configured = os.getenv("WANDING_BUSINESS_KNOWLEDGE_PATH", "").strip()
    if configured:
        return Path(configured)
    return _project_root / "data" / "wanding_business_knowledge.md"


def _load_wanding_knowledge() -> str:
    path = _wanding_knowledge_path()
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        logger.exception("Failed to load wanding business knowledge: %s", path)
        return ""


def _build_selection_payload(keywords: str, candidates: list[dict[str, Any]], show_candidates: bool = False) -> dict[str, Any]:
    return {
        "keywords": keywords,
        "unmatched": not bool(candidates),
        "needs_selection": bool(candidates),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "show_candidates_requested": bool(show_candidates),
        "selection_owner": "claude_code",
        "selection_context": {
            "mode": "claude_code_auto_select",
            "knowledge_source": str(_wanding_knowledge_path()),
            "wanding_business_knowledge": _load_wanding_knowledge(),
            "instructions": [
                "Use the candidates plus wanding_business_knowledge to select the single best quotation item.",
                "Do not show the candidate list to the user unless the user explicitly asked to see candidates.",
                "If every candidate conflicts with the user's keywords, report unmatched instead of forcing a weak match.",
                "If two candidates remain genuinely indistinguishable after applying the knowledge, ask one focused clarification question.",
            ],
        },
    }


def _item_to_dict(item: Any) -> dict[str, Any] | None:
    if item is None:
        return None
    return {
        "code": getattr(item, "code", None) or getattr(item, "item_no", ""),
        "name": getattr(item, "name", None) or getattr(item, "item_name", ""),
        "qty_available": getattr(item, "qty_available", 0.0) or 0.0,
        "qty_warehouse": getattr(item, "qty_warehouse", 0.0) or 0.0,
        "unit": getattr(item, "unit", ""),
    }


def dispatch(tool: str, params: dict[str, Any]) -> Any:
    if tool == "match_quotation":
        from inventory.services.match_and_inventory import match_quotation_union

        keywords = str(params["keywords"])
        candidates = match_quotation_union(
            keywords,
            customer_level=params.get("customer_level", "B"),
            price_library_path=params.get("price_library_path"),
            product_type=params.get("product_type"),
        )
        return _build_selection_payload(
            keywords,
            candidates,
            show_candidates=bool(params.get("show_candidates", False)),
        )

    if tool == "match_quotation_batch":
        from inventory.services.match_and_inventory import match_quotation_union

        results = []
        for keywords in (params.get("keywords_list", []) or [])[:50]:
            keyword_text = str(keywords)
            candidates = match_quotation_union(
                keyword_text,
                customer_level=params.get("customer_level", "B"),
                price_library_path=params.get("price_library_path"),
                product_type=params.get("product_type"),
            )
            results.append(_build_selection_payload(
                keyword_text,
                candidates,
                show_candidates=bool(params.get("show_candidates", False)),
            ))
        return results

    if tool == "get_inventory_by_code":
        from inventory.agents.table_agent import InventoryTableAgent

        return _item_to_dict(InventoryTableAgent().get_item_by_code(str(params["code"])))

    if tool == "get_inventory_by_code_batch":
        from inventory.agents.table_agent import InventoryTableAgent

        table = InventoryTableAgent()
        codes = [str(code) for code in (params.get("codes", []) or [])][:50]
        return [{"code": code, "item": _item_to_dict(table.get_item_by_code(code))} for code in codes]

    if tool == "parse_excel_smart":
        from quotation.quote_tools import parse_excel_smart

        return parse_excel_smart(
            file_path=params["file_path"],
            sheet_name=params.get("sheet_name"),
            max_rows=params.get("max_rows", 500),
        )

    if tool == "fill_quotation_sheet":
        from quotation.flow_orchestrator import run_quotation_fill_flow

        return run_quotation_fill_flow(
            quotation_path=params["file_path"],
            price_library_path=params.get("price_library_path"),
            output_path=params.get("output_path"),
            sheet_name=params.get("sheet_name"),
            customer_level=params.get("customer_level", "B"),
        )

    if tool == "ask_clarification":
        return {
            "question": params.get("question") or "Please provide product type or specification.",
            "reason": params.get("reason") or "The current description is not enough to match a unique quotation item.",
            "options": params.get("options") or [
                {"id": "pvc_water_supply", "name": "PVC-U water supply pipe"},
                {"id": "pvc_drainage", "name": "PVC-U drainage pipe"},
                {"id": "pvc_conduit", "name": "PVC-U conduit"},
                {"id": "other", "name": "Other; user should provide details"},
            ],
        }

    raise ValueError(f"Unknown tool: {tool}")


def handle_request(request: dict[str, Any]) -> dict[str, Any]:
    tool = str(request.get("tool", ""))
    params = request.get("params", {}) or {}
    logger.info("Dispatching: %s", tool)
    return {"success": True, "result": dispatch(tool, params)}


def main() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            response = handle_request(json.loads(line.lstrip("\ufeff")))
        except Exception as exc:
            logger.exception("Tool dispatch failed")
            response = {"success": False, "error": str(exc)}
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
