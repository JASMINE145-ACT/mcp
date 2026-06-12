# wechat-ai-publisher

微信公众号 AI 自动写作与草稿发布系统。

输入一个主题，AI 自动生成公众号文章，上传封面图，创建草稿到公众号后台，人工审核后一键发布。

---

## 功能

- 输入文章主题，AI 生成结构化文章（标题、摘要、正文、封面提示词）
- 二次润色去除 AI 机械口吻
- 内容合规检查（长度、占位符、AI 表达）
- Markdown → 微信公众号兼容 HTML（inline styles，无外部 CSS）
- 自动上传封面图到微信公众号素材库
- 调用微信官方 API 创建草稿
- 本地任务目录保存完整记录（JSON、MD、HTML、图片）
- 详细日志记录

---

## 安装

最快路径见 [docs/quickstart.md](docs/quickstart.md)。MCP 客户端配置示例见 [docs/mcp_client_config.md](docs/mcp_client_config.md) 和 `examples/` 目录。

```bash
# 克隆项目
cd wechat-ai-publisher

# 创建虚拟环境（推荐）
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 安装依赖
pip install -r requirements.txt
```

也可以安装为本地 Python 包：

```bash
pip install .
wechat-ai-publisher-mcp
```

**Python 要求：3.10+**

---

## 配置

```bash
cp .env.example .env
```

编辑 `.env`：

```env
OPENAI_API_KEY=sk-...          # OpenAI API Key
OPENAI_MODEL=gpt-4.1           # 推荐 gpt-4.1 或 gpt-4o
WECHAT_APP_ID=wx...            # 公众号 AppID
WECHAT_APP_SECRET=...          # 公众号 AppSecret
DEFAULT_AUTHOR=2AIBot          # 默认作者名
DEFAULT_COVER_PATH=images/default_cover.jpg
```

### 微信公众号 API 配置

1. 登录 [微信公众平台](https://mp.weixin.qq.com)
2. 「设置与开发」→「基本配置」→ 获取 AppID 和 AppSecret
3. 「IP 白名单」→ 添加运行本程序的服务器 IP
4. 确认已开通「素材管理」和「草稿箱」接口权限（服务号默认开通，订阅号部分受限）

---

## 使用方法

### 1. 测试微信 API 配置

```bash
python main.py test-wechat
```

输出示例：
```
✅ access_token 获取成功
   公众号 API 配置正常
```

### 2. 生成文章（只保存本地文件，不创建草稿）

```bash
python main.py generate --topic "国内人形机器人在大脑领域进展比较快的几家公司点评和对比分析"
```

输出：
```
✅ 文章生成成功
   标题: xxx
   Markdown: storage/drafts/2026-xx-xx-xxx/article.md
   HTML:     storage/drafts/2026-xx-xx-xxx/article.html
```

### 3. 生成文章并创建公众号草稿

```bash
python main.py draft --topic "国内人形机器人在大脑领域进展比较快的几家公司点评和对比分析"
```

使用自定义封面图：
```bash
python main.py draft --topic "主题" --cover path/to/cover.jpg
```

输出：
```
✅ 草稿创建成功
   标题:        xxx
   草稿 media_id: xxx
   Markdown:    storage/drafts/xxx/article.md
   HTML:        storage/drafts/xxx/article.html
   封面图:      images/default_cover.jpg
```

### 4. 用已有 HTML 创建草稿

```bash
python main.py draft-from-html \
  --title "文章标题" \
  --html storage/html/article.html \
  --cover images/default_cover.jpg
```

---

## MCP 工具

启动 MCP Server：

```bash
python server.py
```

Windows 也可以直接运行：

```bash
run_server.bat
```

或：

```powershell
.\start.ps1
```

推荐先调用：

- `wechat_health_check`：检查环境变量、默认封面、依赖和可选微信连通性。
- `wechat_test_connection`：实际请求微信 access_token，验证 AppID/AppSecret/IP 白名单。

资料收集工具（保留给资料检索能力较弱的 AI 使用）：

- `wechat_research`：使用 Exa 搜索并返回结构化资料。
- `wechat_tavily_search`：使用 Tavily 搜索最新资料。
- `wechat_fetch_url`：抓取指定网页正文。

图片工具：

- `wechat_search_cover_image`：搜索封面图候选。
- `wechat_download_image`：下载远程图片到本地。
- `wechat_upload_body_image`：搜索、下载并上传正文配图到微信 CDN。
- `wechat_upload_local_image`：上传本地正文图到微信 CDN。
- `wechat_upload_cover`：上传封面图并返回 `thumb_media_id`。

排版与发布工具：

- `wechat_list_templates`：查看 A/B/C/D 排版模板。
- `wechat_render_markdown`：Markdown 转微信兼容 HTML。
- `wechat_validate_content`：发布前内容质量检查。
- `wechat_create_draft`：创建公众号草稿。
- `wechat_full_pipeline`：Markdown + 元数据 + 封面图一键创建草稿。
- `wechat_list_drafts` / `wechat_update_draft` / `wechat_delete_draft`：管理微信草稿箱。
- `wechat_publish`：提交发布，必须显式设置 `ENABLE_AUTO_PUBLISH=true`。
- `wechat_get_publish_status`：查询发布任务状态。
- `wechat_list_local_tasks`：查看本地 `storage/drafts` 历史任务，用于失败恢复和审计。

---

## 任务目录结构

每次运行会在 `storage/drafts/` 下创建：

```
storage/drafts/2026-06-05-人形机器人/
├── article.json      # 任务元数据（标题、摘要、media_id、状态）
├── article.md        # 生成的 Markdown 正文
├── article.html      # 转换后的 HTML
├── cover_processed.jpg  # 处理后的封面图
├── upload_result.json   # 封面图上传结果
├── draft_result.json    # 草稿创建结果
└── logs.txt          # 本次任务日志
```

---

## 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| 40164 IP白名单 | 服务器IP未加白名单 | 公众号后台添加IP |
| 40013 不合法AppID | AppID填写错误 | 检查.env |
| 封面图不存在 | 路径错误 | 检查cover路径 |
| JSON解析失败 | 模型不支持json_object | 换用gpt-4.1/gpt-4o |

更多问题见 [docs/troubleshooting.md](docs/troubleshooting.md)

---

## 安全提醒

- `.env` 已加入 `.gitignore`，**绝对不要提交到 Git**
- 日志中会自动遮蔽 AppSecret 后半部分
- 默认关闭自动发布（`ENABLE_AUTO_PUBLISH=false`）
- 草稿创建后需要人工在公众号后台审核再发布

---

## 后续开发计划

- **Phase 2**：接入 RSS 素材源、we-mp-rss、本地资料库、自动选题
- **Phase 3**：定时任务、Web 管理后台、多公众号支持、数据分析
