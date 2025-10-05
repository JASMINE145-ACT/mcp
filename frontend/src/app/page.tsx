'use client'

import React, { useState } from 'react'
import { FileUpload } from '@/components/FileUpload'
import { ChatInterface } from '@/components/ChatInterface'
import { Sidebar } from '@/components/Sidebar'

interface SessionData {
  session_id: string
  filename: string
  shape: [number, number]
  columns: string[]
  profile: any
}

export default function Home() {
  const [sessionData, setSessionData] = useState<SessionData | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)

  const handleUploadSuccess = (data: SessionData) => {
    setSessionData(data)
  }

  const handleNewChat = () => {
    setSessionData(null)
  }

  return (
    <div className="flex h-screen bg-white overflow-hidden">
      {/* Sidebar */}
      <Sidebar 
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        sessionData={sessionData}
        onNewChat={handleNewChat}
      />

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {!sessionData ? (
          <FileUpload onUploadSuccess={handleUploadSuccess} />
        ) : (
          <ChatInterface 
            sessionData={sessionData} 
            onNewSession={handleNewChat}
          />
        )}
      </div>
    </div>
  )
}
