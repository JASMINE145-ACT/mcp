# AI 数据分析助手 - 前端

基于 Next.js 构建的智能数据分析前端界面，与 `agent.py` 后端 API 集成。

## 功能特性

- 📊 **文件上传**: 支持 CSV 文件拖拽上传，自动数据分析
- 💬 **智能对话**: 自然语言交互，多轮对话优化分析方案
- 🤖 **多智能体协作**: GPT-5-nano 规划 + Claude 编码 + GPT-4o-mini 验证
- 📈 **可视化展示**: 自动生成图表和统计结果
- 📋 **分析历史**: 保存和回顾历史分析记录
- 🔍 **数据概览**: 详细的数据结构和质量分析

## 技术栈

- **框架**: Next.js 14 + React 18 + TypeScript
- **样式**: Tailwind CSS + Radix UI
- **状态管理**: React Query + Zustand
- **HTTP 客户端**: Axios
- **Markdown 渲染**: React Markdown
- **图标**: Lucide React

## 快速开始

### 1. 安装依赖

\`\`\`bash
cd frontend
npm install
\`\`\`

### 2. 启动开发服务器

\`\`\`bash
npm run dev
\`\`\`

前端将在 http://localhost:3000 启动

### 3. 启动后端 API

确保后端 Flask API 在 http://localhost:8000 运行：

\`\`\`bash
cd ..
python other/backend_api.py
\`\`\`

## 项目结构

\`\`\`
frontend/
├── src/
│   ├── app/                 # Next.js App Router
│   │   ├── layout.tsx       # 根布局
│   │   ├── page.tsx         # 主页面
│   │   ├── providers.tsx    # 全局提供者
│   │   └── globals.css      # 全局样式
│   ├── components/          # React 组件
│   │   ├── ui/              # 基础 UI 组件
│   │   ├── FileUpload.tsx   # 文件上传组件
│   │   ├── ChatInterface.tsx # 聊天界面
│   │   ├── DataOverview.tsx # 数据概览
│   │   └── AnalysisHistory.tsx # 分析历史
│   └── lib/
│       ├── api.ts           # API 客户端
│       └── utils.ts         # 工具函数
├── package.json
├── next.config.js
├── tailwind.config.ts
└── README.md
\`\`\`

## 使用指南

### 1. 上传数据

- 点击"数据上传"标签页
- 拖拽或选择 CSV 文件上传
- 系统自动分析数据结构

### 2. 智能分析

- 切换到"智能分析"标签页
- 用自然语言描述分析需求
- AI 制定分析计划并征求确认
- 确认后自动执行并展示结果

### 3. 查看结果

- "数据概览"：查看数据结构和统计信息
- "分析历史"：回顾所有分析记录和生成的图表

## API 集成

前端通过以下 API 端点与后端通信：

- \`POST /api/upload\` - 上传文件
- \`POST /api/analyze\` - 分析数据
- \`POST /api/confirm\` - 确认执行计划
- \`GET /api/history/{session_id}\` - 获取分析历史
- \`GET /api/plots\` - 获取生成的图表列表

## 开发说明

### 环境变量

创建 \`.env.local\` 文件：

\`\`\`
NEXT_PUBLIC_API_URL=http://localhost:8000/api
\`\`\`

### 构建部署

\`\`\`bash
npm run build
npm start
\`\`\`

### 代码规范

- 使用 TypeScript 严格模式
- 遵循 ESLint 规则
- 组件使用函数式写法
- 样式使用 Tailwind CSS

## 故障排除

### 常见问题

1. **API 连接失败**
   - 确保后端 Flask 服务正在运行
   - 检查 \`next.config.js\` 中的代理配置

2. **文件上传失败**
   - 确保文件格式为 CSV
   - 检查文件大小不超过 100MB

3. **样式显示异常**
   - 运行 \`npm run build\` 重新构建
   - 清除浏览器缓存

### 开发调试

- 使用浏览器开发者工具查看网络请求
- 检查控制台错误信息
- 使用 React Developer Tools 调试组件状态

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

## 许可证

MIT License
