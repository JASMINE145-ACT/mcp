'use client'

import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { SessionData } from '@/lib/api'
import { formatNumber } from '@/lib/utils'
import { Database, BarChart3, Hash, Type, Calendar } from 'lucide-react'

interface DataOverviewProps {
  sessionData: SessionData
}

export function DataOverview({ sessionData }: DataOverviewProps) {
  const profile = sessionData.profile

  const getColumnTypeIcon = (type: string) => {
    switch (type?.toLowerCase()) {
      case 'int64':
      case 'float64':
      case 'number':
        return <Hash className="h-4 w-4" />
      case 'object':
      case 'string':
        return <Type className="h-4 w-4" />
      case 'datetime':
      case 'date':
        return <Calendar className="h-4 w-4" />
      default:
        return <BarChart3 className="h-4 w-4" />
    }
  }

  const getColumnTypeBadge = (type: string) => {
    switch (type?.toLowerCase()) {
      case 'int64':
      case 'float64':
        return <Badge variant="secondary">数值</Badge>
      case 'object':
        return <Badge variant="outline">文本</Badge>
      case 'datetime':
        return <Badge variant="default">日期</Badge>
      default:
        return <Badge variant="secondary">{type}</Badge>
    }
  }

  return (
    <div className="space-y-6">
      {/* Dataset Summary */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            数据集概览
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center p-4 bg-blue-50 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">
                {formatNumber(sessionData.shape[0])}
              </div>
              <div className="text-sm text-gray-600">总行数</div>
            </div>
            
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <div className="text-2xl font-bold text-green-600">
                {formatNumber(sessionData.shape[1])}
              </div>
              <div className="text-sm text-gray-600">总列数</div>
            </div>
            
            <div className="text-center p-4 bg-purple-50 rounded-lg">
              <div className="text-2xl font-bold text-purple-600">
                {formatNumber(profile?.dataset_info?.memory_usage_mb || 0)}MB
              </div>
              <div className="text-sm text-gray-600">内存占用</div>
            </div>
            
            <div className="text-center p-4 bg-orange-50 rounded-lg">
              <div className="text-2xl font-bold text-orange-600">
                {profile?.dataset_info?.missing_values_total || 0}
              </div>
              <div className="text-sm text-gray-600">缺失值</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Column Details */}
      <Card>
        <CardHeader>
          <CardTitle>列详情</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {profile?.column_overview?.map((column: any, index: number) => (
              <div key={index} className="border rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    {getColumnTypeIcon(column.data_type)}
                    <h4 className="font-medium">{column.column_name}</h4>
                  </div>
                  {getColumnTypeBadge(column.data_type)}
                </div>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-gray-500">数据类型：</span>
                    <span className="font-medium">{column.data_type}</span>
                  </div>
                  
                  <div>
                    <span className="text-gray-500">非空值：</span>
                    <span className="font-medium">{formatNumber(column.non_null_count || 0)}</span>
                  </div>
                  
                  <div>
                    <span className="text-gray-500">唯一值：</span>
                    <span className="font-medium">{formatNumber(column.unique_count || 0)}</span>
                  </div>
                  
                  <div>
                    <span className="text-gray-500">缺失值：</span>
                    <span className="font-medium">{formatNumber(column.missing_count || 0)}</span>
                  </div>
                </div>
                
                {column.sample_values && column.sample_values.length > 0 && (
                  <div className="mt-3">
                    <span className="text-gray-500 text-sm">示例值：</span>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {column.sample_values.slice(0, 5).map((value: any, idx: number) => (
                        <Badge key={idx} variant="outline" className="text-xs">
                          {String(value)}
                        </Badge>
                      ))}
                      {column.sample_values.length > 5 && (
                        <Badge variant="outline" className="text-xs">
                          +{column.sample_values.length - 5} 更多
                        </Badge>
                      )}
                    </div>
                  </div>
                )}
                
                {/* Statistics for numeric columns */}
                {column.statistics && (
                  <div className="mt-3 p-3 bg-gray-50 rounded">
                    <div className="text-sm font-medium mb-2">统计信息</div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
                      {column.statistics.mean !== undefined && (
                        <div>
                          <span className="text-gray-500">均值：</span>
                          <span className="font-medium">{Number(column.statistics.mean).toFixed(2)}</span>
                        </div>
                      )}
                      {column.statistics.std !== undefined && (
                        <div>
                          <span className="text-gray-500">标准差：</span>
                          <span className="font-medium">{Number(column.statistics.std).toFixed(2)}</span>
                        </div>
                      )}
                      {column.statistics.min !== undefined && (
                        <div>
                          <span className="text-gray-500">最小值：</span>
                          <span className="font-medium">{column.statistics.min}</span>
                        </div>
                      )}
                      {column.statistics.max !== undefined && (
                        <div>
                          <span className="text-gray-500">最大值：</span>
                          <span className="font-medium">{column.statistics.max}</span>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Data Quality */}
      <Card>
        <CardHeader>
          <CardTitle>数据质量</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
              <span className="font-medium">数据完整性</span>
              <Badge variant="secondary">
                {profile?.dataset_info?.completeness_percentage || 0}% 完整
              </Badge>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
              <span className="font-medium">重复行</span>
              <Badge variant="outline">
                {formatNumber(profile?.dataset_info?.duplicate_rows || 0)} 行
              </Badge>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-purple-50 rounded-lg">
              <span className="font-medium">数据类型一致性</span>
              <Badge variant="secondary">良好</Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
