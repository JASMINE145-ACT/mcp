# 部署说明

## 本地运行（推荐方式）

```bash
# 1. 克隆或下载项目
cd wechat-ai-publisher

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate  # macOS/Linux
# 或
venv\Scripts\activate     # Windows

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 API 密钥

# 5. 测试配置
python main.py test-wechat

# 6. 生成文章
python main.py generate --topic "测试主题"

# 7. 创建草稿
python main.py draft --topic "测试主题"
```

## 服务器部署（可选）

1. 使用 `screen` 或 `tmux` 保持会话
2. 建议使用 `crontab` 定时运行（Phase 3 功能）
3. 不需要开放端口，纯命令行工具

## 环境要求

- Python 3.10+
- 网络能访问 api.weixin.qq.com 和 api.openai.com
- 微信公众号服务器 IP 白名单已配置

## 微信公众号配置

1. 登录 [微信公众平台](https://mp.weixin.qq.com)
2. 进入「设置与开发」→「基本配置」
3. 获取 AppID 和 AppSecret
4. 在「IP白名单」中添加服务器 IP
5. 确认已开通「素材管理」和「草稿箱」接口权限
