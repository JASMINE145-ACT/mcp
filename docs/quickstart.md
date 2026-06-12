# Quickstart

This guide is for users who receive this MCP server as a folder or a Git repository.

## 1. Install

```bash
cd wechat-ai-publisher
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux:

```bash
cd wechat-ai-publisher
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or newer is required.

**Optional: Playwright for stealth web fetch**

`wechat_fetch_url` works without Playwright in normal mode. If you need `stealth=true`
(bypassing Cloudflare / anti-scrape), install the browser once after pip install:

```bash
playwright install chromium
```

Skip this if you don't plan to use stealth mode.

## 2. Configure

Copy the example env file:

```bash
copy .env.example .env
```

On macOS or Linux:

```bash
cp .env.example .env
```

Required for WeChat draft and publishing tools:

```env
WECHAT_APP_ID=
WECHAT_APP_SECRET=
```

Optional but useful:

```env
OPENAI_API_KEY=          # only needed for main.py AI article generation
OPENAI_MODEL=gpt-4.1
EXA_API_KEY=             # enables wechat_research
TAVILY_API_KEY=          # enables wechat_tavily_search
UNSPLASH_ACCESS_KEY=     # enables image search
PEXELS_API_KEY=          # alternative image search provider
DEFAULT_COVER_PATH=      # local cover image path
```

Keep `ENABLE_AUTO_PUBLISH=false` unless you intentionally want MCP clients to submit publish jobs.

## 3. Run The MCP Server

Source folder mode:

```bash
python server.py
```

Windows convenience scripts:

```bash
run_server.bat
```

or:

```powershell
.\start.ps1
```

Installed package mode:

```bash
pip install .
wechat-ai-publisher-mcp
```

## 4. First Calls

Call these tools from your MCP client first:

1. `wechat_health_check` with `check_wechat=false`
2. `wechat_test_connection`
3. `wechat_list_templates`

Then use one of these workflows:

- Research assisted workflow: `wechat_research` or `wechat_tavily_search` -> write Markdown -> `wechat_full_pipeline`
- Existing article workflow: `wechat_render_markdown` -> `wechat_upload_cover` -> `wechat_create_draft`
- Recovery workflow: `wechat_list_local_tasks` -> retry the failed step shown in the task metadata

## 5. WeChat Platform Requirements

The account must have access to the required Official Account APIs:

- access token
- permanent material upload
- draft add/update/delete/list
- free publish submit/status if publishing is enabled

The server IP must be added to the WeChat Official Account IP allowlist.
