# AI 数据分析助手 - 前端项目完整指南

## 🎯 项目概述

基于 **agent.py** 构建的现代化 Web 前端界面，实现智能数据分析的交互式体验。

### 技术架构

```
┌─────────────────────────────────────────────────────┐
│                   前端 (Next.js)                     │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ 文件上传    │  │  智能对话    │  │ 结果展示   │ │
│  └─────────────┘  └──────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────┘
                         ↓↑ HTTP/REST API
┌─────────────────────────────────────────────────────┐
│              后端 (Flask + agent.py)                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ GPT-5    │→ │ Claude   │→ │ GPT-4o-mini      │  │
│  │ Nano     │  │ Sonnet   │  │ (Validator)      │  │
│  │ (Planner)│  │ (Coder)  │  └──────────────────┘  │
│  └──────────┘  └──────────┘                         │
└─────────────────────────────────────────────────────┘
```

## 📦 快速开始

### 方法 1：使用启动脚本（推荐）

**Windows 用户:**
```bash
# 启动完整应用（前端 + 后端）
start_fullstack.bat

# 或仅启动前端
start_frontend.bat
```

**Mac/Linux 用户:**
```bash
# 安装依赖
cd frontend
npm install

# 启动开发服务器
npm run dev

# 另开终端启动后端
cd ..
python other/backend_api.py
```

### 方法 2：手动启动

```bash
# 1. 安装 Python 依赖
pip install -r requirements.txt

# 2. 启动后端 API
python other/backend_api.py
# 后端运行在: http://localhost:8000

# 3. 安装前端依赖（新终端）
cd frontend
npm install

# 4. 启动前端开发服务器
npm run dev
# 前端运行在: http://localhost:3000
```

## 🏗️ 项目结构详解

```
frontend/
├── src/
│   ├── app/                        # Next.js App Router
│   │   ├── layout.tsx              # 根布局，定义全局 HTML 结构
│   │   ├── page.tsx                # 主页面，包含标签页导航
│   │   ├── providers.tsx           # React Query 提供者配置
│   │   └── globals.css             # 全局样式，包含 Tailwind CSS
│   │
│   ├── components/                 # React 组件
│   │   ├── ui/                     # 基础 UI 组件（shadcn/ui）
│   │   │   ├── button.tsx          # 按钮组件
│   │   │   ├── card.tsx            # 卡片组件
│   │   │   ├── input.tsx           # 输入框组件
│   │   │   ├── textarea.tsx        # 文本域组件
│   │   │   ├── badge.tsx           # 徽章组件
│   │   │   └── tabs.tsx            # 标签页组件
│   │   │
│   │   ├── FileUpload.tsx          # 文件上传组件
│   │   ├── ChatInterface.tsx       # 聊天界面组件
│   │   ├── DataOverview.tsx        # 数据概览组件
│   │   └── AnalysisHistory.tsx     # 分析历史组件
│   │
│   └── lib/                        # 工具库
│       ├── api.ts                  # API 客户端，封装所有后端调用
│       └── utils.ts                # 工具函数
│
├── public/                         # 静态资源
├── .eslintrc.json                  # ESLint 配置
├── tsconfig.json                   # TypeScript 配置
├── next.config.js                  # Next.js 配置（包含 API 代理）
├── tailwind.config.ts              # Tailwind CSS 配置
├── postcss.config.js               # PostCSS 配置
├── package.json                    # 依赖管理
└── README.md                       # 项目文档
```

## 🔧 核心功能实现

### 1. 文件上传 (`FileUpload.tsx`)

**功能：**
- 拖拽上传 CSV 文件
- 文件格式和大小验证
- 实时上传进度显示
- 自动创建分析会话

**关键代码：**
```typescript
const handleFileSelect = async (file: File) => {
  const sessionData = await apiClient.uploadFile(file)
  onUploadSuccess(sessionData)
}
```

### 2. 智能对话 (`ChatInterface.tsx`)

**功能：**
- 自然语言输入
- AI 分析计划展示
- 用户确认机制
- 多轮对话支持
- Markdown 格式化结果

**工作流程：**
```typescript
用户输入问题 
  → AI 生成分析计划 
  → 用户确认/修改
  → 执行分析
  → 展示结果
```

**关键实现：**
```typescript
// 发送分析请求
const response = await apiClient.analyze({
  session_id: sessionData.session_id,
  question: userInput
})

// 处理需要确认的情况
if (response.needs_confirmation) {
  setPendingConfirmation(response)
  // 显示计划并等待用户确认
}

// 用户确认后执行
const result = await apiClient.confirmPlan(sessionId)
```

### 3. 数据概览 (`DataOverview.tsx`)

**功能：**
- 数据集统计信息
- 列详情展示
- 数据类型分析
- 数据质量评估

**展示内容：**
- 总行数、列数
- 内存占用
- 缺失值统计
- 每列的统计信息（均值、标准差等）

### 4. 分析历史 (`AnalysisHistory.tsx`)

**功能：**
- 历史记录列表
- 分析详情展开/收起
- 生成的图表展示
- 时间戳记录

## 🔌 API 集成

### API 客户端 (`lib/api.ts`)

**所有 API 端点：**

```typescript
// 1. 文件上传
POST /api/upload
Body: FormData { file: File }
Response: { session_id, filename, shape, columns, profile }

// 2. 数据分析
POST /api/analyze
Body: { session_id, question, feedback? }
Response: { needs_confirmation, plan, code, execution_result, validation }

// 3. 确认执行
POST /api/confirm
Body: { session_id }
Response: { code, execution_result, validation }

// 4. 获取历史
GET /api/history/{session_id}
Response: { history: [...] }

// 5. 获取图表列表
GET /api/plots
Response: { plots: [...] }

// 6. 获取图表文件
GET /api/plot/{filename}
Response: Image file
```

### 使用示例

```typescript
import { apiClient } from '@/lib/api'

// 上传文件
const sessionData = await apiClient.uploadFile(file)

// 分析数据
const result = await apiClient.analyze({
  session_id: sessionData.session_id,
  question: "分析销售趋势"
})

// 确认计划
if (result.needs_confirmation) {
  const finalResult = await apiClient.confirmPlan(sessionData.session_id)
}
```

## 🎨 样式系统

### Tailwind CSS 配置

**主题色彩：**
```css
primary: 蓝色 (#3B82F6)
secondary: 灰色 (#F3F4F6)
accent: 紫色 (#9333EA)
```

**自定义样式类：**
```css
.scrollbar-thin          /* 自定义滚动条 */
.typing-dot             /* 打字动画 */
.message-enter          /* 消息进入动画 */
.chart-container        /* 图表容器样式 */
```

### 响应式设计

所有组件都支持移动端和桌面端：

```tsx
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
  {/* 移动端单列，平板双列，桌面三列 */}
</div>
```

## 🐛 常见问题与解决方案

### 1. 模块导入错误

**问题：** `Cannot find module '@/components/...'`

**解决：**
```bash
# 确保 tsconfig.json 配置正确
{
  "compilerOptions": {
    "paths": {
      "@/*": ["./src/*"]
    }
  }
}
```

### 2. API 连接失败

**问题：** 前端无法连接后端

**解决：**
```javascript
// 检查 next.config.js 中的代理配置
async rewrites() {
  return [{
    source: '/api/agent/:path*',
    destination: 'http://localhost:8000/:path*'
  }]
}

// 确保后端在 8000 端口运行
python other/backend_api.py
```

### 3. 样式不生效

**问题：** Tailwind CSS 样式不显示

**解决：**
```bash
# 重新构建
npm run build

# 检查 tailwind.config.ts 中的 content 配置
content: [
  './src/**/*.{js,ts,jsx,tsx,mdx}',
]
```

### 4. TypeScript 类型错误

**问题：** 类型定义缺失

**解决：**
```bash
# 安装类型定义
npm install --save-dev @types/react @types/node

# 重启 TypeScript 服务器（VSCode）
Ctrl+Shift+P → "Restart TS Server"
```

## 📝 开发指南

### 添加新功能

**1. 创建新组件：**
```typescript
// src/components/MyComponent.tsx
'use client'

import React from 'react'

export function MyComponent() {
  return <div>My Component</div>
}
```

**2. 添加新 API：**
```typescript
// src/lib/api.ts
export const apiClient = {
  // ...existing methods
  
  async myNewApi() {
    const response = await api.get('/my-endpoint')
    return response.data
  }
}
```

**3. 添加新页面：**
```typescript
// src/app/my-page/page.tsx
export default function MyPage() {
  return <div>My Page</div>
}
```

### 代码规范

- 使用 TypeScript 严格模式
- 组件使用函数式写法
- 遵循 ESLint 规则
- 使用 Tailwind CSS 而非内联样式
- API 调用统一通过 apiClient

### 测试

```bash
# 运行 linter
npm run lint

# 构建检查
npm run build

# 类型检查
npx tsc --noEmit
```

## 🚀 部署

### 生产环境构建

```bash
# 构建优化版本
npm run build

# 启动生产服务器
npm start
```

### 环境变量

创建 `.env.production`:
```bash
NEXT_PUBLIC_API_URL=https://your-api-domain.com/api
```

### Docker 部署

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## 📚 学习资源

- **Next.js**: https://nextjs.org/docs
- **React**: https://react.dev/
- **Tailwind CSS**: https://tailwindcss.com/docs
- **shadcn/ui**: https://ui.shadcn.com/
- **React Query**: https://tanstack.com/query/latest

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License
