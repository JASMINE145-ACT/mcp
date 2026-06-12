# MCP Client Config

Use an absolute path in client configuration. Replace the path examples with your local path.

## Source Folder Mode

**Windows** (use the venv Python to avoid PATH issues):

```json
{
  "mcpServers": {
    "wechat-ai-publisher": {
      "command": "D:/path/to/wechat-ai-publisher/.venv/Scripts/python.exe",
      "args": ["D:/path/to/wechat-ai-publisher/server.py"],
      "cwd": "D:/path/to/wechat-ai-publisher"
    }
  }
}
```

**macOS / Linux** (`python` may not exist in PATH — always use the venv path):

```json
{
  "mcpServers": {
    "wechat-ai-publisher": {
      "command": "/path/to/wechat-ai-publisher/.venv/bin/python",
      "args": ["/path/to/wechat-ai-publisher/server.py"],
      "cwd": "/path/to/wechat-ai-publisher"
    }
  }
}
```

If your client supports environment variables inline, you can also pass them there. Otherwise use `.env`.

## Windows Script Mode

```json
{
  "mcpServers": {
    "wechat-ai-publisher": {
      "command": "D:/path/to/wechat-ai-publisher/run_server.bat",
      "args": []
    }
  }
}
```

## Installed Package Mode

After running `pip install .`:

```json
{
  "mcpServers": {
    "wechat-ai-publisher": {
      "command": "wechat-ai-publisher-mcp",
      "args": []
    }
  }
}
```

## Recommended First Prompt

Ask your MCP client:

```text
Call wechat_health_check with check_wechat=false and summarize what is configured.
```

Then:

```text
Call wechat_test_connection to verify the WeChat Official Account API.
```

## Safety

Do not enable `ENABLE_AUTO_PUBLISH=true` for shared users by default. Keeping it false still allows draft creation and manual review in the WeChat backend.
