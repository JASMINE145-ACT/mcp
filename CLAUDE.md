# Claude Code behavior for quotation MCP

When using `match_quotation` / `match_quotation_batch`:

- Treat tool output as selection context, not as user-facing candidate output.
- Use `selection_context.wanding_business_knowledge` plus `candidates` to select one best item.
- Do not show the full candidate list unless the user explicitly asks.
- If all candidates conflict with the user's keywords, report unmatched.
- If candidates remain indistinguishable after applying Wanding knowledge, ask one focused clarification question.
- Do not call an additional selector model from Python/MCP.
