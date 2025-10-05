'use client'

import React, { useState, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import { Send, Bot, User, Loader2, CheckCircle, RefreshCw, FileText } from 'lucide-react'
import { apiClient, SessionData, AnalysisResponse } from '@/lib/api'
import { formatNumber } from '@/lib/utils'
import ReactMarkdown from 'react-markdown'

interface ChatInterfaceProps {
  sessionData: SessionData
  onNewSession: () => void
}

interface Message {
  id: string
  type: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  needsConfirmation?: boolean
  analysisData?: AnalysisResponse
}

export function ChatInterface({ sessionData, onNewSession }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [pendingConfirmation, setPendingConfirmation] = useState<AnalysisResponse | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    // Add welcome message
    setMessages([{
      id: 'welcome',
      type: 'system',
      content: `欢迎使用 AI 数据分析助手！\n\n已加载数据文件：**${sessionData.filename}**\n数据规模：${formatNumber(sessionData.shape[0])} 行 × ${sessionData.shape[1]} 列\n\n你可以用自然语言提问，例如：\n• "分析销售数据的趋势"\n• "画一个用户年龄分布的直方图"\n• "计算各类别的平均值"`,
      timestamp: new Date()
    }])
  }, [sessionData])

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue.trim(),
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsLoading(true)

    try {
      const response = await apiClient.analyze({
        session_id: sessionData.session_id,
        question: inputValue.trim()
      })

      if (response.needs_confirmation) {
        // AI needs confirmation for the plan
        setPendingConfirmation(response)
        
        const planMessage: Message = {
          id: Date.now().toString() + '_plan',
          type: 'assistant',
          content: formatPlanMessage(response),
          timestamp: new Date(),
          needsConfirmation: true,
          analysisData: response
        }
        
        setMessages(prev => [...prev, planMessage])
      } else {
        // Analysis completed
        const resultMessage: Message = {
          id: Date.now().toString() + '_result',
          type: 'assistant',
          content: formatResultMessage(response),
          timestamp: new Date(),
          analysisData: response
        }
        
        setMessages(prev => [...prev, resultMessage])
      }
    } catch (error: any) {
      const errorMessage: Message = {
        id: Date.now().toString() + '_error',
        type: 'assistant',
        content: `❌ 分析出错：${error.response?.data?.error || error.message}`,
        timestamp: new Date()
      }
      
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleConfirmPlan = async () => {
    if (!pendingConfirmation) return

    setIsLoading(true)
    
    try {
      const response = await apiClient.confirmPlan(sessionData.session_id)
      
      const resultMessage: Message = {
        id: Date.now().toString() + '_confirmed',
        type: 'assistant',
        content: formatConfirmedResultMessage(response),
        timestamp: new Date()
      }
      
      setMessages(prev => [...prev, resultMessage])
      setPendingConfirmation(null)
    } catch (error: any) {
      const errorMessage: Message = {
        id: Date.now().toString() + '_confirm_error',
        type: 'assistant',
        content: `❌ 执行出错：${error.response?.data?.error || error.message}`,
        timestamp: new Date()
      }
      
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleProvideFeedback = async (feedback: string) => {
    if (!pendingConfirmation) return

    const feedbackMessage: Message = {
      id: Date.now().toString() + '_feedback',
      type: 'user',
      content: feedback,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, feedbackMessage])
    setIsLoading(true)

    try {
      const response = await apiClient.analyze({
        session_id: sessionData.session_id,
        question: pendingConfirmation.question,
        feedback: feedback
      })

      if (response.needs_confirmation) {
        setPendingConfirmation(response)
        
        const updatedPlanMessage: Message = {
          id: Date.now().toString() + '_updated_plan',
          type: 'assistant',
          content: formatPlanMessage(response),
          timestamp: new Date(),
          needsConfirmation: true,
          analysisData: response
        }
        
        setMessages(prev => [...prev, updatedPlanMessage])
      } else {
        const resultMessage: Message = {
          id: Date.now().toString() + '_feedback_result',
          type: 'assistant',
          content: formatResultMessage(response),
          timestamp: new Date(),
          analysisData: response
        }
        
        setMessages(prev => [...prev, resultMessage])
        setPendingConfirmation(null)
      }
    } catch (error: any) {
      const errorMessage: Message = {
        id: Date.now().toString() + '_feedback_error',
        type: 'assistant',
        content: `❌ 处理反馈出错：${error.response?.data?.error || error.message}`,
        timestamp: new Date()
      }
      
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const formatPlanMessage = (response: AnalysisResponse): string => {
    try {
      const plan = JSON.parse(response.plan)
      const analysisType = plan.analysis_type || '分析'
      const method = plan.method || '数据处理'
      const expectedOutput = plan.expected_output || '分析结果'
      
      return `📋 **分析计划**

**分析类型：** ${analysisType}
**处理方法：** ${method}
**预期输出：** ${expectedOutput}

请确认是否执行此计划，或提供修改建议。`
    } catch {
      return `📋 **分析计划**\n\n${response.plan}\n\n请确认是否执行此计划，或提供修改建议。`
    }
  }

  const formatResultMessage = (response: AnalysisResponse): string => {
    let message = '✅ **分析完成**\n\n'
    
    if (response.execution_result) {
      message += `**执行结果：**\n\`\`\`\n${response.execution_result}\n\`\`\`\n\n`
    }
    
    if (response.validation) {
      try {
        const validation = JSON.parse(response.validation)
        if (validation.final_answer) {
          message += `**结论：** ${validation.final_answer}\n\n`
        }
        if (validation.result_interpretation) {
          message += `**解读：** ${validation.result_interpretation}\n\n`
        }
        if (validation.recommendations) {
          message += `**建议：** ${validation.recommendations}\n\n`
        }
      } catch {
        message += `**分析说明：**\n${response.validation}\n\n`
      }
    }
    
    return message
  }

  const formatConfirmedResultMessage = (response: any): string => {
    let message = '✅ **计划已执行完成**\n\n'
    
    if (response.execution_result) {
      message += `**执行结果：**\n\`\`\`\n${response.execution_result}\n\`\`\`\n\n`
    }
    
    if (response.validation) {
      try {
        const validation = JSON.parse(response.validation)
        if (validation.final_answer) {
          message += `**结论：** ${validation.final_answer}\n\n`
        }
        if (validation.result_interpretation) {
          message += `**解读：** ${validation.result_interpretation}\n\n`
        }
        if (validation.recommendations) {
          message += `**建议：** ${validation.recommendations}\n\n`
        }
      } catch {
        message += `**分析说明：**\n${response.validation}\n\n`
      }
    }
    
    return message
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  return (
    <div className="flex flex-col h-screen bg-white">
      {/* Header */}
      <div className="border-b bg-white px-6 py-4">
        <div className="flex items-center justify-between max-w-4xl mx-auto">
          <div>
            <h2 className="text-lg font-semibold text-gray-900">数据分析对话</h2>
            <p className="text-sm text-gray-500 mt-0.5">
              {sessionData.filename} · {formatNumber(sessionData.shape[0])} 行 × {sessionData.shape[1]} 列
            </p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-6 py-8">
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className="flex gap-4"
              >
                {/* Avatar */}
                <div className="flex-shrink-0">
                  {message.type === 'user' ? (
                    <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center">
                      <User className="h-4 w-4 text-gray-600" />
                    </div>
                  ) : (
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                      <Bot className="h-4 w-4 text-white" />
                    </div>
                  )}
                </div>

                {/* Message Content */}
                <div className="flex-1 space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-medium text-gray-900">
                      {message.type === 'user' ? '你' : 'AI 助手'}
                    </span>
                    <span className="text-xs text-gray-500">
                      {message.timestamp.toLocaleTimeString()}
                    </span>
                  </div>
                  
                  <div className={`${
                    message.type === 'system' ? 'bg-blue-50 rounded-lg p-4' : ''
                  }`}>
                    <div className="prose prose-sm max-w-none">
                      <ReactMarkdown>
                        {message.content}
                      </ReactMarkdown>
                      
                      {message.needsConfirmation && (
                        <div className="mt-4 flex gap-2">
                          <Button
                            size="sm"
                            onClick={handleConfirmPlan}
                            disabled={isLoading}
                          >
                            <CheckCircle className="h-4 w-4 mr-1" />
                            确认执行
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => {
                              const feedback = prompt('请提供修改建议：')
                              if (feedback) {
                                handleProvideFeedback(feedback)
                              }
                            }}
                            disabled={isLoading}
                          >
                            修改计划
                          </Button>
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div className="flex gap-4">
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                    <Bot className="h-4 w-4 text-white" />
                  </div>
                </div>
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
                    <span className="text-sm text-gray-600">AI 正在思考...</span>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        </div>
      </div>
        
      {/* Input Area */}
      <div className="border-t bg-white">
        <div className="max-w-4xl mx-auto px-6 py-4">
          <div className="flex gap-3 items-end">
            <Textarea
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="输入你的数据分析问题..."
              className="flex-1 min-h-[56px] max-h-[200px] resize-none rounded-xl border-gray-300 focus:border-blue-500 focus:ring-blue-500"
              disabled={isLoading}
            />
            <Button
              onClick={handleSendMessage}
              disabled={!inputValue.trim() || isLoading}
              className="h-14 px-6 rounded-xl bg-blue-600 hover:bg-blue-700"
            >
              <Send className="h-5 w-5" />
            </Button>
          </div>
          
          <p className="mt-2 text-xs text-gray-500 text-center">
            按 Enter 发送，Shift + Enter 换行
          </p>
        </div>
      </div>
    </div>
  )
}
