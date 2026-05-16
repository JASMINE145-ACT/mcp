"""Standalone inventory agents used by the quotation MCP server."""

from inventory.agents.table_agent import InventoryTableAgent
from inventory.agents.sql_agent import InventorySQLAgent

InventoryPlanAgent = None

__all__ = ["InventoryPlanAgent", "InventoryTableAgent", "InventorySQLAgent"]
