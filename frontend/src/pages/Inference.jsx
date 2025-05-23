import React, { useState, useEffect, useRef } from 'react'
import { Send, Bot, User, Settings, Download, RotateCcw, MessageSquare, Cpu, HardDrive } from 'lucide-react'
import * as api from '../services/api'

function Inference() {
  const [models, setModels] = useState([])
  const [selectedModel, setSelectedModel] = useState(null)
  const [messages, setMessages] = useState([])
  const [currentMessage, setCurrentMessage] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [parameters, setParameters] = useState({
    maxTokens: 100,
    temperature: 0.7,
    topP: 0.9
  })
  const [showSettings, setShowSettings] = useState(false)
  const messagesEndRef = useRef(null)

  useEffect(() => {
    loadCompletedModels()
  }, [])

  useEffect(() => {
    // Check for model parameter in URL
    const urlParams = new URLSearchParams(window.location.search)
    const modelIdFromUrl = urlParams.get('model')
    
    if (modelIdFromUrl && models.length > 0) {
      const model = models.find(m => m.id === modelIdFromUrl)
      if (model) {
        setSelectedModel(model)
      }
    }
  }, [models])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const loadCompletedModels = async () => {
    console.log('🔄 Loading inference models...')
    try {
      // Try the jobs endpoint first (most reliable)
      const response = await api.trainingAPI.getJobs()
      console.log('✅ Got jobs response:', response)
      
      // Check if we have inference_models in the response
      let inferenceModels = response.data?.inference_models || response.inference_models || []
      
      // If no inference_models, fallback to filtering completed jobs
      if (inferenceModels.length === 0) {
        const completedJobs = (response.jobs || response.data?.jobs || [])
          .filter(job => job.status === 'completed')
        inferenceModels = completedJobs
        console.log('📊 Using completed jobs as inference models:', completedJobs.length)
      } else {
        console.log('📊 Using dedicated inference models:', inferenceModels.length)
      }
      
      console.log('📊 Final models found:', inferenceModels.length, inferenceModels)
      setModels(inferenceModels)
      
      if (inferenceModels.length > 0 && !selectedModel) {
        setSelectedModel(inferenceModels[0])
        console.log('🎯 Auto-selected first model:', inferenceModels[0].id)
      }
    } catch (error) {
      console.error('❌ Failed to load models:', error)
      // Final fallback - try the dedicated endpoint
      try {
        console.log('🔄 Trying dedicated inference endpoint as last resort...')
        const response = await api.trainingAPI.getInferenceModels()
        const inferenceModels = response.data?.models || response.models || []
        console.log('📊 Dedicated endpoint models found:', inferenceModels.length, inferenceModels)
        setModels(inferenceModels)
        
        if (inferenceModels.length > 0 && !selectedModel) {
          setSelectedModel(inferenceModels[0])
          console.log('🎯 Auto-selected first dedicated model:', inferenceModels[0].id)
        }
      } catch (fallbackError) {
        console.error('❌ All methods failed:', fallbackError)
      }
    }
  }

  const handleSendMessage = async () => {
    if (!currentMessage.trim() || !selectedModel) return

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: currentMessage,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setCurrentMessage('')
    setIsGenerating(true)

    try {
      const response = await api.trainingAPI.testInference(selectedModel.id, currentMessage, parameters)
      
      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: response.response || response.data?.response,
        timestamp: new Date(),
        model: selectedModel.config?.base_model || selectedModel.base_model,
        parameters: response.parameters
      }

      setMessages(prev => [...prev, botMessage])
    } catch (error) {
      console.error('Failed to generate response:', error)
      
      const errorMessage = {
        id: Date.now() + 1,
        type: 'error',
        content: `Error: ${error.response?.data?.detail || error.message}`,
        timestamp: new Date()
      }
      
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsGenerating(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const clearConversation = () => {
    setMessages([])
  }

  const exportConversation = () => {
    const conversation = messages.map(msg => ({
      type: msg.type,
      content: msg.content,
      timestamp: msg.timestamp.toISOString(),
      ...(msg.model && { model: msg.model }),
      ...(msg.parameters && { parameters: msg.parameters })
    }))

    const blob = new Blob([JSON.stringify(conversation, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `conversation_${selectedModel?.id || 'unknown'}_${new Date().toISOString().split('T')[0]}.json`
    link.click()
    URL.revokeObjectURL(url)
  }

  const unloadCurrentModel = async () => {
    if (!selectedModel) return
    
    try {
      await api.trainingAPI.unloadModel(selectedModel.id)
      alert(`✅ Model ${selectedModel.id} unloaded from memory`)
    } catch (error) {
      console.error('Failed to unload model:', error)
      alert('❌ Failed to unload model: ' + (error.response?.data?.detail || error.message))
    }
  }

  const unloadAllModels = async () => {
    if (!confirm('This will unload all models from GPU memory. Continue?')) {
      return
    }
    
    try {
      await api.trainingAPI.unloadAllModels()
      alert('✅ All models unloaded from memory')
    } catch (error) {
      console.error('Failed to unload all models:', error)
      alert('❌ Failed to unload all models: ' + (error.response?.data?.detail || error.message))
    }
  }

  return (
    <div className="h-screen flex flex-col bg-gray-50 overflow-hidden">
      {/* Header */}
      <div className="flex-shrink-0 bg-white border-b border-gray-200 p-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold text-gray-900 flex items-center">
              <MessageSquare className="h-5 w-5 mr-2" />
              Model Testing & Inference
            </h1>
            <p className="text-sm text-gray-600">Test your trained models with interactive chat</p>
          </div>
          
          <div className="flex items-center space-x-3">
            {/* Model Selector */}
            <select
              value={selectedModel?.id || ''}
              onChange={(e) => {
                const model = models.find(m => m.id === e.target.value)
                setSelectedModel(model)
                setMessages([]) // Clear conversation when switching models
              }}
              className="px-3 py-2 text-sm border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">Select a model...</option>
              {models.map(model => (
                <option key={model.id} value={model.id}>
                  {model.name || (model.config?.base_model?.split('/').pop() || 'Unknown')} ({model.id})
                </option>
              ))}
            </select>
            
            {/* Settings Toggle */}
            <button
              onClick={() => setShowSettings(!showSettings)}
              className={`p-2 rounded-lg transition-colors ${showSettings ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}`}
            >
              <Settings className="h-4 w-4" />
            </button>
            
            {/* Clear Chat */}
            <button
              onClick={clearConversation}
              className="p-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200"
            >
              <RotateCcw className="h-4 w-4" />
            </button>
            
            {/* Export Chat */}
            <button
              onClick={exportConversation}
              className="p-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200"
              disabled={messages.length === 0}
            >
              <Download className="h-4 w-4" />
            </button>
            
            {/* Memory Management */}
            {selectedModel && (
              <button
                onClick={unloadCurrentModel}
                className="p-2 text-orange-700 bg-orange-100 rounded-lg hover:bg-orange-200"
                title="Unload current model from GPU memory"
              >
                <HardDrive className="h-4 w-4" />
              </button>
            )}
            
            <button
              onClick={unloadAllModels}
              className="px-3 py-2 text-xs text-red-700 bg-red-100 rounded-lg hover:bg-red-200"
              title="Unload all models from GPU memory"
            >
              Free GPU
            </button>
            
            {/* Debug Connection */}
            <button
              onClick={async () => {
                try {
                  const response = await api.trainingAPI.testConnection()
                  console.log('🔍 Connection test:', response)
                  alert('✅ Backend connection successful!')
                } catch (error) {
                  console.error('🔍 Connection test failed:', error)
                  alert('❌ Backend connection failed: ' + error.message)
                }
              }}
              className="px-3 py-2 text-xs text-blue-700 bg-blue-100 rounded-lg hover:bg-blue-200"
              title="Test backend connection"
            >
              Test API
            </button>
          </div>
        </div>
        
        {/* Model Info */}
        {selectedModel && (
          <div className="mt-3 p-3 bg-blue-50 rounded-lg">
            <div className="flex items-center justify-between text-sm">
              <div>
                <span className="font-medium text-blue-900">Model:</span> {selectedModel.config?.base_model || selectedModel.base_model}
              </div>
              <div>
                <span className="font-medium text-blue-900">Method:</span> {(selectedModel.config?.method || selectedModel.method)?.toUpperCase()}
              </div>
              <div>
                <span className="font-medium text-blue-900">Job ID:</span> {selectedModel.id}
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="flex flex-1 overflow-hidden min-h-0">
        {/* Settings Panel */}
        {showSettings && (
          <div className="w-80 bg-white border-r border-gray-200 p-4 overflow-y-auto flex-shrink-0">
            <h3 className="text-sm font-semibold text-gray-900 mb-4">Generation Parameters</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  Max Tokens: {parameters.maxTokens}
                </label>
                <input
                  type="range"
                  min="10"
                  max="500"
                  value={parameters.maxTokens}
                  onChange={(e) => setParameters(prev => ({ ...prev, maxTokens: parseInt(e.target.value) }))}
                  className="w-full"
                />
              </div>
              
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  Temperature: {parameters.temperature}
                </label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={parameters.temperature}
                  onChange={(e) => setParameters(prev => ({ ...prev, temperature: parseFloat(e.target.value) }))}
                  className="w-full"
                />
                <p className="text-xs text-gray-500 mt-1">Controls randomness. Lower = more focused, Higher = more creative</p>
              </div>
              
              <div>
                <label className="block text-xs font-medium text-gray-700 mb-1">
                  Top P: {parameters.topP}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={parameters.topP}
                  onChange={(e) => setParameters(prev => ({ ...prev, topP: parseFloat(e.target.value) }))}
                  className="w-full"
                />
                <p className="text-xs text-gray-500 mt-1">Nucleus sampling. Controls diversity of output</p>
              </div>
            </div>
          </div>
        )}

        {/* Chat Area */}
        <div className="flex-1 flex flex-col min-h-0">
          {/* Messages - Takes remaining space */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.length === 0 ? (
              <div className="text-center py-12">
                <Bot className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">Ready to test your model!</h3>
                <p className="text-gray-500">
                  {selectedModel 
                    ? "Send a message to start testing your trained model"
                    : "Select a completed training job to begin testing"
                  }
                </p>
              </div>
            ) : (
              messages.map((message) => (
                <div
                  key={message.id}
                  className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-3xl px-4 py-2 rounded-lg ${
                      message.type === 'user'
                        ? 'bg-blue-600 text-white'
                        : message.type === 'error'
                        ? 'bg-red-100 text-red-800 border border-red-200'
                        : 'bg-white text-gray-900 border border-gray-200'
                    }`}
                  >
                    <div className="flex items-start space-x-2">
                      {message.type !== 'user' && (
                        <div className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center ${
                          message.type === 'error' ? 'bg-red-200' : 'bg-gray-200'
                        }`}>
                          {message.type === 'error' ? '⚠️' : <Bot className="h-3 w-3" />}
                        </div>
                      )}
                      <div className="flex-1">
                        <p className="whitespace-pre-wrap">{message.content}</p>
                        <p className={`text-xs mt-1 ${
                          message.type === 'user' ? 'text-blue-200' : 'text-gray-500'
                        }`}>
                          {message.timestamp.toLocaleTimeString()}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              ))
            )}
            
            {isGenerating && (
              <div className="flex justify-start">
                <div className="bg-white text-gray-900 border border-gray-200 max-w-3xl px-4 py-2 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <Bot className="h-4 w-4 text-gray-500" />
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    </div>
                    <span className="text-sm text-gray-500">Generating response...</span>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area - Always visible at bottom */}
          <div className="flex-shrink-0 border-t border-gray-200 bg-white p-4">
            <div className="flex space-x-3">
              <textarea
                value={currentMessage}
                onChange={(e) => setCurrentMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={selectedModel ? "Type your message..." : "Select a model first..."}
                disabled={!selectedModel || isGenerating}
                className="flex-1 resize-none border border-gray-300 rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-100"
                rows="2"
              />
              <button
                onClick={handleSendMessage}
                disabled={!currentMessage.trim() || !selectedModel || isGenerating}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
              >
                <Send className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Inference 