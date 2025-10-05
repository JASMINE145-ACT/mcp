import axios from 'axios'

const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? '/api/agent' 
  : 'http://localhost:8000/api'

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 seconds for analysis requests
})

export interface SessionData {
  session_id: string
  filename: string
  shape: [number, number]
  columns: string[]
  profile: any
}

export interface AnalysisRequest {
  session_id: string
  question: string
  feedback?: string
}

export interface AnalysisResponse {
  success: boolean
  needs_confirmation: boolean
  question: string
  plan: string
  code: string
  execution_result: string
  validation: string
  conversation_state: {
    conversation_history: Array<{role: string, content: string}>
    plan_iterations: string[]
    plan_confirmed: boolean
    needs_user_input: boolean
  }
}

export interface ConfirmResponse {
  success: boolean
  question: string
  code: string
  execution_result: string
  validation: string
}

export interface HistoryItem {
  timestamp: string
  question: string
  result: any
}

// API functions
export const apiClient = {
  // Health check
  async healthCheck() {
    const response = await api.get('/health')
    return response.data
  },

  // Upload file and create session
  async uploadFile(file: File): Promise<SessionData> {
    const formData = new FormData()
    formData.append('file', file)
    
    const response = await api.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })
    return response.data
  },

  // Get session info
  async getSession(sessionId: string) {
    const response = await api.get(`/session/${sessionId}`)
    return response.data
  },

  // Analyze data
  async analyze(request: AnalysisRequest): Promise<AnalysisResponse> {
    const response = await api.post('/analyze', request)
    return response.data
  },

  // Confirm plan and execute
  async confirmPlan(sessionId: string): Promise<ConfirmResponse> {
    const response = await api.post('/confirm', { session_id: sessionId })
    return response.data
  },

  // Get analysis history
  async getHistory(sessionId: string): Promise<{ success: boolean, history: HistoryItem[] }> {
    const response = await api.get(`/history/${sessionId}`)
    return response.data
  },

  // List available plots
  async listPlots(): Promise<{ success: boolean, plots: string[] }> {
    const response = await api.get('/plots')
    return response.data
  },

  // Get plot image URL
  getPlotUrl(filename: string): string {
    return `${API_BASE_URL}/plot/${filename}`
  },

  // List all sessions
  async listSessions() {
    const response = await api.get('/sessions')
    return response.data
  }
}

export default api
