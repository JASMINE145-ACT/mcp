# 🚀 快速开始指南

## 📋 项目概览

你现在有三个前端选项：

1. **✅ Streamlit** (已完成) - 简单快速，Python 原生
2. **🎨 React (Next.js 14)** (新创建) - 现代化，类似截图风格
3. **📊 原始 CLI** - 命令行交互

## 🏃 方式 1: Streamlit (推荐快速测试)

```bash
streamlit run streamlit_app.py
```

访问: http://localhost:8501

## 🎨 方式 2: Next.js React UI (现代化界面)

### 第一次设置:

```bash
# 1. 安装 Flask CORS
pip install flask flask-cors

# 2. 安装前端依赖
cd frontend
npm install

# 3. 初始化 shadcn/ui
npx shadcn-ui@latest init
# 选择所有默认选项

# 4. 安装组件
npx shadcn-ui@latest add button input dialog tabs scroll-area toast card
```

### 启动应用:

#### Windows:
```bash
# 双击运行
start_app.bat

# 或手动启动
python backend_api.py  # Terminal 1 - 后端
cd frontend && npm run dev  # Terminal 2 - 前端
```

#### Mac/Linux:
```bash
# Terminal 1 - 启动后端
python backend_api.py

# Terminal 2 - 启动前端
cd frontend
npm run dev
```

访问:
- **前端**: http://localhost:3000
- **后端 API**: http://localhost:8000

## 📊 方式 3: CLI (命令行)

```bash
python agent.py
```

## 🔧 backend_api.py 的额外依赖

需要安装:
```bash
pip install flask flask-cors werkzeug
```

## 📡 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/upload` | POST | 上传 CSV 文件 |
| `/api/analyze` | POST | 分析数据 |
| `/api/confirm` | POST | 确认计划并执行 |
| `/api/history/<session_id>` | GET | 获取历史记录 |
| `/api/plots` | GET | 列出所有图表 |
| `/api/plot/<filename>` | GET | 获取特定图表 |

## 🎯 使用流程

### Streamlit:
1. 上传 CSV → 2. 输入问题 → 3. 确认计划 → 4. 查看结果

### Next.js:
1. 上传 CSV → 2. 在聊天框输入问题 → 3. AI 生成计划 → 4. 确认执行 → 5. 查看结果和图表

## 🐛 常见问题

### Q: 端口被占用
```bash
# 查看并关闭占用端口的进程
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Mac/Linux:
lsof -i :8000
kill -9 <PID>
```

### Q: 前端连接不到后端
检查:
1. backend_api.py 是否运行在 8000 端口
2. next.config.js 中的 rewrites 配置是否正确
3. CORS 是否启用

### Q: LangSmith 追踪
在 `.env` 中添加:
```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key
LANGCHAIN_PROJECT=your_project
```

## 📁 项目结构

```
mcp/
├── agent.py              # 核心 AI Agent
├── backend_api.py        # Flask REST API
├── streamlit_app.py      # Streamlit UI
├── function.py           # 工具函数
├── requirements.txt      # Python 依赖
├── start_app.bat         # Windows 启动脚本
├── frontend/             # Next.js 前端
│   ├── app/             # Next.js 页面
│   ├── components/      # React 组件
│   ├── lib/             # 工具函数
│   └── package.json     # Node 依赖
└── uploads/             # 上传的文件
```

## 🎨 Next.js 前端特点

- ✅ 类似 Collax 的现代化设计
- ✅ 深色/浅色主题切换
- ✅ 流畅的动画过渡 (Framer Motion)
- ✅ 响应式布局
- ✅ 实时聊天界面
- ✅ 图表预览
- ✅ 历史记录管理
- ✅ 拖拽上传文件

## 💡 下一步

1. **自定义样式**: 编辑 `frontend/app/globals.css`
2. **添加功能**: 在 `frontend/app/page.tsx` 中扩展
3. **优化 Agent**: 修改 `agent.py` 中的 prompt
4. **部署**: 使用 Vercel (前端) + Railway (后端)

## 🆘 需要帮助？

查看:
- `LANGSMITH_GUIDE.md` - LangSmith 监控
- `FRONTEND_SETUP.md` - 详细前端设置
- `backend_api.py` - API 文档注释

