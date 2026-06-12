# 常见问题排查

## 微信 API 错误

### 错误码 40164：IP 不在白名单

**现象**：运行 `test-wechat` 提示 IP 白名单错误

**解决**：
1. 查看本机公网 IP：`curl ifconfig.me`
2. 登录公众号后台 → 设置与开发 → 基本配置 → IP白名单
3. 添加当前服务器 IP

### 错误码 40013：不合法的 AppID

**现象**：AppID 格式错误

**解决**：检查 `.env` 中 `WECHAT_APP_ID` 是否正确复制，无多余空格

### 错误码 40125：不合法的 AppSecret

**现象**：AppSecret 错误

**解决**：在公众号后台重新生成 AppSecret，更新 `.env`

### 封面图上传失败

**常见原因**：
- 图片格式不是 jpg/png
- 图片大小超过 2MB
- IP 不在白名单

**解决**：检查图片格式，运行 `cover_processor.py` 压缩图片

---

## OpenAI 错误

### 返回内容不是 JSON

**现象**：`ValueError: AI 返回的内容不是合法 JSON`

**解决**：
- 确认模型支持 `response_format: json_object`（gpt-4.1、gpt-4o 支持）
- 检查 prompt 中是否明确要求 JSON 输出

### API Key 无效

**现象**：`AuthenticationError`

**解决**：检查 `.env` 中 `OPENAI_API_KEY` 是否正确

---

## 文件错误

### 风格文件不存在

**现象**：`FileNotFoundError: Style guide not found`

**解决**：确认 `style/wechat-article-style.md` 文件存在

### 封面图不存在

**现象**：`FileNotFoundError: 封面图不存在`

**解决**：
- 检查 `images/default_cover.jpg` 是否存在
- 或通过 `--cover` 参数指定其他图片路径
