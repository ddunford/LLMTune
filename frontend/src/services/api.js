import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || ''

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`)
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response
  },
  (error) => {
    console.error('API Error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

// Datasets API
export const datasetAPI = {
  // Upload dataset
  upload: (formData, onProgress) => {
    return api.post('/api/datasets/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: onProgress,
    })
  },
  
  // Dataset management
  list: () => api.get('/api/datasets/'),
  get: (datasetId) => api.get(`/api/datasets/${datasetId}`),
  delete: (datasetId) => api.delete(`/api/datasets/${datasetId}`),
  preview: (datasetId) => api.get(`/api/datasets/${datasetId}/preview`),
  process: (datasetId, options) => api.post(`/api/datasets/${datasetId}/process`, options),
}

// Legacy alias for backward compatibility
export const datasetsAPI = datasetAPI

// Training API (updated)
export const trainingAPI = {
  // Get supported models
  getSupportedModels: () => api.get('/api/training/models'),
  
  // Validate model
  validateModel: (modelId) => api.post('/api/training/validate-model', null, {
    params: { model_id: modelId }
  }),
  
  // Training jobs
  createJob: (config) => api.post('/api/training/jobs', config),
  getJobs: () => api.get('/api/training/jobs'),
  getJob: (jobId) => api.get(`/api/training/jobs/${jobId}`),
  controlJob: (jobId, action) => api.post(`/api/training/jobs/${jobId}/control`, { action }),
  deleteJob: (jobId) => api.delete(`/api/training/jobs/${jobId}`),
  getLogs: (jobId, limit = 100) => api.get(`/api/training/jobs/${jobId}/logs`, { params: { limit } }),
  cleanup: () => api.post('/api/training/cleanup'),
  restore: () => api.post('/api/training/restore'),
  downloadModel: (jobId, fileType = 'adapter') => {
    return api.get(`/api/training/jobs/${jobId}/download`, {
      params: { file_type: fileType },
      responseType: 'blob'
    })
  },
  testInference: (jobId, prompt, parameters = {}) => {
    return api.post(`/api/training/jobs/${jobId}/inference`, {
      prompt,
      max_tokens: parameters.maxTokens || 100,
      temperature: parameters.temperature || 0.7,
      top_p: parameters.topP || 0.9
    }, {
      timeout: 120000  // 2 minutes timeout for inference requests
    })
  },
  unloadModel: (jobId) => api.post(`/api/training/jobs/${jobId}/unload`),
  unloadAllModels: () => api.post('/api/training/inference/unload-all'),
  
  // Inference-specific endpoints
  getInferenceModels: () => api.get('/api/training/inference/models'),
  getModelStatus: (jobId) => api.get(`/api/training/jobs/${jobId}/model-status`),
  
  // Debug endpoints
  testConnection: () => api.get('/api/training/test')
}

// Monitoring API (updated)
export const monitoringAPI = {
  // System stats
  getStats: () => api.get('/api/monitoring/stats'),
  getSystemStats: () => api.get('/api/monitoring/stats'), // Legacy alias
  getLogs: (limit = 100) => api.get('/api/monitoring/logs', { params: { limit } }),
}

// WebSocket connections
export const createWebSocket = (endpoint, onMessage, onError = null) => {
  // For WebSocket, we need to connect directly to the backend since Vite proxy doesn't handle WS well
  const wsUrl = 'ws://localhost:8001'
  const ws = new WebSocket(`${wsUrl}${endpoint}`)
  
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data)
      onMessage(data)
    } catch (error) {
      console.error('Error parsing WebSocket message:', error)
    }
  }
  
  ws.onerror = (error) => {
    console.error('WebSocket error:', error)
    if (onError) onError(error)
  }
  
  ws.onclose = () => {
    console.log('WebSocket connection closed')
  }
  
  return ws
}

export default api 