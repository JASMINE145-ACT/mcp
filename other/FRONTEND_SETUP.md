# 🎨 前端设置指南

基于 Next.js 14 + TailwindCSS + shadcn/ui 的现代化数据分析 AI Agent 界面

## 📦 技术栈

- **Framework**: Next.js 14 (App Router)
- **UI Library**: TailwindCSS + shadcn/ui
- **Icons**: Lucide React
- **State Management**: Zustand + React Query
- **Animation**: Framer Motion
- **Backend**: Flask API (agent.py)

## 🚀 快速开始

### 1. 安装依赖

```bash
cd frontend
npm install
```

### 2. 安装 shadcn/ui 组件

```bash
npx shadcn-ui@latest init
```

选择:
- TypeScript: Yes
- Style: Default
- Base color: Slate
- Global CSS: app/globals.css
- CSS variables: Yes
- Tailwind config: tailwind.config.ts
- Components: @/components
- Utils: @/lib/utils

然后安装需要的组件:

```bash
npx shadcn-ui@latest add button
npx shadcn-ui@latest add input
npx shadcn-ui@latest add dialog
npx shadcn-ui@latest add dropdown-menu
npx shadcn-ui@latest add tabs
npx shadcn-ui@latest add scroll-area
npx shadcn-ui@latest add toast
npx shadcn-ui@latest add card
npx shadcn-ui@latest add avatar
npx shadcn-ui@latest add badge

