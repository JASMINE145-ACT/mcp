'use client'

import React, { useState, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { Upload, File, CheckCircle, AlertCircle, Loader2 } from 'lucide-react'
import { apiClient, SessionData } from '@/lib/api'
import { formatFileSize } from '@/lib/utils'

interface FileUploadProps {
  onUploadSuccess: (data: SessionData) => void
}

export function FileUpload({ onUploadSuccess }: FileUploadProps) {
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState<'idle' | 'success' | 'error'>('idle')
  const [errorMessage, setErrorMessage] = useState('')
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [isDragActive, setIsDragActive] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = async (file: File) => {
    if (!file) return

    // Validate file type
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setUploadStatus('error')
      setErrorMessage('只支持 CSV 格式文件')
      return
    }

    // Validate file size (100MB)
    if (file.size > 100 * 1024 * 1024) {
      setUploadStatus('error')
      setErrorMessage('文件大小不能超过 100MB')
      return
    }

    setUploadedFile(file)
    setUploading(true)
    setUploadStatus('idle')
    setErrorMessage('')

    try {
      const sessionData = await apiClient.uploadFile(file)
      setUploadStatus('success')
      onUploadSuccess(sessionData)
    } catch (error: any) {
      setUploadStatus('error')
      setErrorMessage(error.response?.data?.error || '上传失败，请重试')
    } finally {
      setUploading(false)
    }
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragActive(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragActive(false)
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragActive(false)
    
    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      handleFileSelect(files[0])
    }
  }

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      handleFileSelect(files[0])
    }
  }

  const handleClick = () => {
    fileInputRef.current?.click()
  }

  return (
    <div className="flex-1 flex items-center justify-center p-8 bg-gradient-to-br from-gray-50 to-gray-100">
      <div className="max-w-2xl w-full space-y-8">
        {/* Welcome Header */}
        <div className="text-center">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl mb-6">
            <Upload className="h-8 w-8 text-white" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-3">
            欢迎使用 AI 数据分析助手
          </h1>
          <p className="text-lg text-gray-600">
            上传 CSV 文件，开始智能数据分析之旅
          </p>
        </div>

        {/* Upload Area */}
        <Card className="border-2">
        <CardContent className="p-8">
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={handleClick}
            className={`
              border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
              ${isDragActive 
                ? 'border-blue-500 bg-blue-50' 
                : 'border-gray-300 hover:border-gray-400'
              }
              ${uploading ? 'pointer-events-none opacity-50' : ''}
            `}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".csv"
              onChange={handleFileInputChange}
              className="hidden"
            />
            
            <div className="flex flex-col items-center space-y-4">
              {uploading ? (
                <Loader2 className="h-12 w-12 text-blue-500 animate-spin" />
              ) : (
                <Upload className="h-12 w-12 text-gray-400" />
              )}
              
              <div>
                <p className="text-lg font-medium text-gray-900">
                  {uploading ? '正在上传...' : '拖拽文件到此处或点击选择'}
                </p>
                <p className="text-sm text-gray-500 mt-1">
                  支持 CSV 格式，最大 100MB
                </p>
              </div>
              
              {!uploading && (
                <Button variant="outline">
                  选择文件
                </Button>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Upload Status */}
      {uploadedFile && (
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center space-x-3">
              <File className="h-8 w-8 text-blue-500" />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900 truncate">
                  {uploadedFile.name}
                </p>
                <p className="text-sm text-gray-500">
                  {formatFileSize(uploadedFile.size)}
                </p>
              </div>
              
              <div className="flex items-center">
                {uploading && (
                  <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />
                )}
                {uploadStatus === 'success' && (
                  <CheckCircle className="h-5 w-5 text-green-500" />
                )}
                {uploadStatus === 'error' && (
                  <AlertCircle className="h-5 w-5 text-red-500" />
                )}
              </div>
            </div>
            
            {uploadStatus === 'success' && (
              <div className="mt-3 p-3 bg-green-50 rounded-md">
                <p className="text-sm text-green-800">
                  ✅ 文件上传成功！已创建分析会话，可以开始提问了。
                </p>
              </div>
            )}
            
            {uploadStatus === 'error' && (
              <div className="mt-3 p-3 bg-red-50 rounded-md">
                <p className="text-sm text-red-800">
                  ❌ {errorMessage}
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      )}

        {/* Features */}
        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="p-4">
            <div className="text-2xl mb-2">🚀</div>
            <p className="text-sm font-medium text-gray-900">快速分析</p>
            <p className="text-xs text-gray-500 mt-1">自动识别数据结构</p>
          </div>
          <div className="p-4">
            <div className="text-2xl mb-2">💬</div>
            <p className="text-sm font-medium text-gray-900">智能对话</p>
            <p className="text-xs text-gray-500 mt-1">自然语言交互</p>
          </div>
          <div className="p-4">
            <div className="text-2xl mb-2">📊</div>
            <p className="text-sm font-medium text-gray-900">可视化</p>
            <p className="text-xs text-gray-500 mt-1">自动生成图表</p>
          </div>
        </div>
      </div>
    </div>
  )
}