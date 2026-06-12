# Release Checklist

Use this before sending the MCP server to another user.

## Required

- `.env` is not included in the shared package.
- `.env.example` is included and contains no real secrets.
- `README.md`, `docs/quickstart.md`, and `docs/mcp_client_config.md` are included.
- `requirements.txt` and `pyproject.toml` are included.
- `run_server.bat` and `start.ps1` are included for Windows users.
- `storage/drafts`, `storage/html`, `storage/markdown`, `storage/images`, and `storage/logs` contain only `.gitkeep` in the shared package.
- `ENABLE_AUTO_PUBLISH=false` is the documented default.

## Recommended Verification

```bash
python -m pytest -q
python scripts/check_release.py
```

For a real user environment:

1. Install dependencies.
2. Copy `.env.example` to `.env`.
3. Fill `WECHAT_APP_ID` and `WECHAT_APP_SECRET`.
4. Add the machine IP to the WeChat Official Account allowlist.
5. Start the MCP server.
6. Call `wechat_health_check`.
7. Call `wechat_test_connection`.
8. Create one draft with `wechat_full_pipeline`.

## Do Not Remove

The following tools are intentionally included to help weaker AI clients gather context:

- `wechat_research`
- `wechat_tavily_search`
- `wechat_fetch_url`
- `wechat_search_cover_image`
- `wechat_upload_body_image`

They may be optional based on API keys, but they should remain part of the MCP tool surface.
