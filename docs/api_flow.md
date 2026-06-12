# API Flow

## 整体流程

```
main.py draft --topic "主题"
  │
  ├─ config/settings.py       读取 .env 环境变量，校验必要配置
  │
  ├─ config/settings.py       读取 style/wechat-article-style.md 写作风格
  │
  ├─ ai/article_generator.py  调用 OpenAI API 生成结构化 JSON
  │     └─ prompts/system_prompt.md + article_prompt.md
  │
  ├─ ai/article_generator.py  polish_article() 二次润色
  │
  ├─ ai/content_checker.py    内容合规检查
  │
  ├─ storage/task.py          创建任务目录 storage/drafts/YYYY-MM-DD-slug/
  │
  ├─ render/markdown_to_html.py  Markdown → WeChat inline-styled HTML
  │
  ├─ images/cover_processor.py  检查、压缩封面图
  │
  ├─ wechat/token.py          GET /cgi-bin/token → access_token（带缓存）
  │
  ├─ wechat/material.py       POST /cgi-bin/material/add_material → thumb_media_id
  │
  └─ wechat/draft.py          POST /cgi-bin/draft/add → media_id
```

## WeChat API Endpoints

| 功能 | 方法 | 端点 |
|------|------|------|
| 获取 access_token | GET | `/cgi-bin/token` |
| 上传永久素材（封面图） | POST | `/cgi-bin/material/add_material` |
| 上传文章正文图片 | POST | `/cgi-bin/media/uploadimg` |
| 创建草稿 | POST | `/cgi-bin/draft/add` |
| 发布草稿 | POST | `/cgi-bin/freepublish/submit` |
| 查询发布状态 | POST | `/cgi-bin/freepublish/getarticle` |

## Token 缓存策略

- `access_token` 有效期 7200 秒（2小时）
- 缓存在内存 `_cache` 字典中
- 过期前 60 秒自动触发刷新
- 获取失败时使用 `tenacity` 重试 3 次
