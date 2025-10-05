'use client'

import React from 'react'
import { Button } from '@/components/ui/button'
import { PlusCircle, ChevronLeft, ChevronRight, Database, FileText, History, Settings } from 'lucide-react'
import { SessionData } from '@/lib/api'

interface SidebarProps {
  isOpen: boolean
  onToggle: () => void
  sessionData: SessionData | null
  onNewChat: () => void
}

export function Sidebar({ isOpen, onToggle, sessionData, onNewChat }: SidebarProps) {
  if (!isOpen) {
    return (
      <div className="w-12 bg-gray-50 border-r flex flex-col items-center py-4">
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggle}
          className="mb-4"
        >
          <ChevronRight className="h-5 w-5" />
        </Button>
      </div>
    )
  }

  return (
    <div className="w-64 bg-gray-50 border-r flex flex-col">
      {/* Header */}
      <div className="p-4 border-b flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">AI</span>
          </div>
          <span className="font-semibold text-gray-900">数据分析</span>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggle}
        >
          <ChevronLeft className="h-5 w-5" />
        </Button>
      </div>

      {/* New Chat Button */}
      <div className="p-4">
        <Button
          onClick={onNewChat}
          className="w-full justify-start gap-2 bg-blue-600 hover:bg-blue-700"
        >
          <PlusCircle className="h-4 w-4" />
          新建对话
        </Button>
      </div>

      {/* Navigation */}
      <div className="flex-1 overflow-y-auto px-2">
        <div className="space-y-1">
          <div className="px-3 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
            Dashboard
          </div>
          
          <Button
            variant="ghost"
            className="w-full justify-start gap-3 text-gray-700 hover:bg-gray-100"
          >
            <Database className="h-4 w-4" />
            <span>数据概览</span>
          </Button>

          {sessionData && (
            <>
              <Button
                variant="ghost"
                className="w-full justify-start gap-3 text-gray-700 hover:bg-gray-100"
              >
                <FileText className="h-4 w-4" />
                <span className="truncate">{sessionData.filename}</span>
              </Button>
              
              <div className="px-3 py-2 text-xs text-gray-500">
                {sessionData.shape[0]} 行 × {sessionData.shape[1]} 列
              </div>
            </>
          )}
        </div>

        <div className="mt-6 space-y-1">
          <div className="px-3 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
            Records
          </div>
          
          <Button
            variant="ghost"
            className="w-full justify-start gap-3 text-gray-700 hover:bg-gray-100"
          >
            <History className="h-4 w-4" />
            <span>历史记录</span>
          </Button>
        </div>
      </div>

      {/* Footer */}
      <div className="border-t p-4">
        <Button
          variant="ghost"
          className="w-full justify-start gap-3 text-gray-700 hover:bg-gray-100"
        >
          <Settings className="h-4 w-4" />
          <span>设置</span>
        </Button>
        
        <div className="mt-4 px-3 py-2 bg-blue-50 rounded-lg">
          <p className="text-xs text-gray-600">
            多智能体协作
          </p>
          <p className="text-xs text-gray-500 mt-1">
            GPT-5 · Claude · GPT-4
          </p>
        </div>
      </div>
    </div>
  )
}
