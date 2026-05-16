# Wanding Quotation MCP

Standalone MCP server for Wanding quotation matching.

## Behavior

- Python retrieves quotation candidates from the migrated agent-jk logic.
- Python/MCP does not call an extra LLM selector for `match_quotation`.
- Claude Code receives candidates plus `data/wanding_business_knowledge.md` in `selection_context` and performs the final selection.
- Candidate lists should only be shown when the user explicitly asks for them (`show_candidates=true`).

## Tools

- `match_quotation`
- `match_quotation_batch`
- `get_inventory_by_code`
- `get_inventory_by_code_batch`
- `fill_quotation_sheet`
- `parse_excel_smart`
- `ask_clarification`

## Install

```powershell
bun install
python -m pip install -r requirements.txt
```

## Claude Code MCP config

Copy `.mcp.example.json` into your Claude Code MCP config and replace `CCB_PROJECT_ROOT` with this repository's absolute path.

For local testing from the repository root:

```powershell
$env:CCB_PROJECT_ROOT=(Get-Location).Path
bun run start
```

## Data

The repository includes:

- `data/wanding_price_lib.xlsx`
- `data/mapping_table.xlsx`
- `data/wanding_business_knowledge.md`

No `.env.accurate`, Accurate token, or signature secret is committed.
