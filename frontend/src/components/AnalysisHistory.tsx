'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { History, Clock, MessageSquare, BarChart3, Eye, EyeOff } from 'lucide-react'
import { apiClient } from '@/lib/api'
import ReactMarkdown from 'react-markdown'

interface AnalysisHistoryProps {
  sessionId: string
}

interface HistoryItem {
  timestamp: string
  question: string
  result: {
    question: string
    plan: string
    code: string
    execution_result: string
    validation: string
  }
}

export function AnalysisHistory({ sessionId }: AnalysisHistoryProps) {
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [loading, setLoading] = useState(true)
  const [expandedItems, setExpandedItems] = useState<Set<number>>(new Set())
  const [plots, setPlots] = useState<string[]>([])

  useEffect(() => {
    loadHistory()
    loadPlots()
  }, [sessionId])

  const loadHistory = async () => {
    try {
      const response = await apiClient.getHistory(sessionId)
      if (response.success) {
        setHistory(response.history)
      }
    } catch (error) {
      console.error('Failed to load history:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadPlots = async () => {
    try {
      const response = await apiClient.listPlots()
      if (response.success) {
        setPlots(response.plots)
      }
    } catch (error) {
      console.error('Failed to load plots:', error)
    }
  }

  const toggleExpanded = (index: number) => {
    const newExpanded = new Set(expandedItems)
    if (newExpanded.has(index)) {
      newExpanded.delete(index)
    } else {
      newExpanded.add(index)
    }
    setExpandedItems(newExpanded)
  }

  const formatValidation = (validation: string) => {
    try {
      const parsed = JSON.parse(validation)
      let formatted = ''
      
      if (parsed.final_answer) {
        formatted += `**结论：** ${parsed.final_answer}\n\n`
      }
      
      if (parsed.result_interpretation) {
        formatted += `**解读：** ${parsed.result_interpretation}\n\n`
      }
      
      if (parsed.recommendations) {
        formatted += `**建议：** ${parsed.recommendations}\n\n`
      }
      
      return formatted || validation
    } catch {
      return validation
    }
  }

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString('zh-CN')
  }

  if (loading) {
    return (
      <Card>
        <CardContent className="p-8 text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
          <p className="mt-2 text-gray-600">加载分析历史...</p>
        </CardContent>
      </Card>
    )
  }

  if (history.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <History className="h-5 w-5" />
            分析历史
          </CardTitle>
        </CardHeader>
        <CardContent className="text-center py-8">
          <MessageSquare className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <p className="text-gray-600">还没有分析历史</p>
          <p className="text-sm text-gray-500 mt-1">开始提问来创建你的第一个分析</p>
        </CardContent>
      </Card>
    )
  }

  return (
    <div className="space-y-6">
      {/* Plots Gallery */}
      {plots.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" />
              生成的图表
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {plots.map((plot, index) => (
                <div key={index} className="border rounded-lg overflow-hidden">
                  <img
                    src={apiClient.getPlotUrl(plot)}
                    alt={`Plot ${index + 1}`}
                    className="w-full h-48 object-cover"
                  />
                  <div className="p-2 bg-gray-50">
                    <p className="text-sm font-medium truncate">{plot}</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Analysis History */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <History className="h-5 w-5" />
            分析历史 ({history.length})
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {history.map((item, index) => {
              const isExpanded = expandedItems.has(index)
              
              return (
                <div key={index} className="border rounded-lg p-4">
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <h4 className="font-medium text-gray-900 mb-1">
                        {item.question}
                      </h4>
                      <div className="flex items-center gap-2 text-sm text-gray-500">
                        <Clock className="h-4 w-4" />
                        {formatTimestamp(item.timestamp)}
                      </div>
                    </div>
                    
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => toggleExpanded(index)}
                    >
                      {isExpanded ? (
                        <EyeOff className="h-4 w-4" />
                      ) : (
                        <Eye className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                  
                  {isExpanded && (
                    <div className="space-y-4 pt-4 border-t">
                      {/* Plan */}
                      {item.result.plan && (
                        <div>
                          <h5 className="font-medium mb-2 flex items-center gap-2">
                            <Badge variant="outline">计划</Badge>
                          </h5>
                          <div className="bg-blue-50 p-3 rounded text-sm">
                            <ReactMarkdown>{item.result.plan}</ReactMarkdown>
                          </div>
                        </div>
                      )}
                      
                      {/* Code */}
                      {item.result.code && (
                        <div>
                          <h5 className="font-medium mb-2 flex items-center gap-2">
                            <Badge variant="secondary">代码</Badge>
                          </h5>
                          <div className="bg-gray-900 text-gray-100 p-3 rounded text-sm overflow-x-auto">
                            <pre>{item.result.code}</pre>
                          </div>
                        </div>
                      )}
                      
                      {/* Execution Result */}
                      {item.result.execution_result && (
                        <div>
                          <h5 className="font-medium mb-2 flex items-center gap-2">
                            <Badge variant="default">执行结果</Badge>
                          </h5>
                          <div className="bg-green-50 p-3 rounded text-sm">
                            <pre className="whitespace-pre-wrap">
                              {item.result.execution_result}
                            </pre>
                          </div>
                        </div>
                      )}
                      
                      {/* Validation */}
                      {item.result.validation && (
                        <div>
                          <h5 className="font-medium mb-2 flex items-center gap-2">
                            <Badge variant="destructive">AI 分析</Badge>
                          </h5>
                          <div className="bg-purple-50 p-3 rounded text-sm">
                            <ReactMarkdown>
                              {formatValidation(item.result.validation)}
                            </ReactMarkdown>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
