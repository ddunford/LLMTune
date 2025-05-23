import React, { useState, useEffect } from 'react'
import { Search, CheckCircle, AlertCircle, ExternalLink } from 'lucide-react'
import { trainingAPI } from '../services/api'

const POPULAR_MODELS = [
  // General Chat & Conversation
  {
    id: 'meta-llama/Llama-3.3-70B-Instruct',
    name: 'Llama 3.3 70B Instruct',
    description: 'Meta\'s latest model - excellent for conversation, reasoning, and general tasks',
    category: 'General & Chat',
    recommended: true
  },
  {
    id: 'mistralai/Mistral-7B-Instruct-v0.3',
    name: 'Mistral 7B Instruct v0.3',
    description: 'Efficient 7B model with great performance and function calling support',
    category: 'General & Chat',
    recommended: true
  },
  {
    id: 'microsoft/DialoGPT-medium',
    name: 'DialoGPT Medium',
    description: 'Specialized conversational AI model for chatbots',
    category: 'General & Chat',
    recommended: false
  },

  // Coding & Development
  {
    id: 'deepseek-ai/deepseek-coder-7b-instruct-v1.5',
    name: 'DeepSeek Coder 7B Instruct',
    description: 'Specialized for coding tasks, supports 80+ programming languages',
    category: 'Coding',
    recommended: true
  },
  {
    id: 'codellama/CodeLlama-7b-Instruct-hf',
    name: 'Code Llama 7B Instruct',
    description: 'Meta\'s specialized coding model based on Llama 2',
    category: 'Coding',
    recommended: true
  },
  {
    id: 'WizardLM/WizardCoder-15B-V1.0',
    name: 'WizardCoder 15B',
    description: 'Strong coding performance across multiple programming languages',
    category: 'Coding',
    recommended: false
  },

  // Reasoning & Math
  {
    id: 'deepseek-ai/deepseek-math-7b-instruct',
    name: 'DeepSeek Math 7B Instruct',
    description: 'Specialized for mathematical reasoning and problem solving',
    category: 'Reasoning & Math',
    recommended: true
  },
  {
    id: 'microsoft/phi-3-mini-4k-instruct',
    name: 'Phi-3 Mini 4K Instruct',
    description: 'Microsoft\'s efficient model optimized for reasoning tasks',
    category: 'Reasoning & Math',
    recommended: true
  },

  // Small & Efficient Models
  {
    id: 'microsoft/phi-3-mini-128k-instruct',
    name: 'Phi-3 Mini 128K Instruct',
    description: 'Compact but powerful model with large context window',
    category: 'Small & Efficient',
    recommended: true
  },
  {
    id: 'google/gemma-2-9b-it',
    name: 'Gemma 2 9B Instruct',
    description: 'Google\'s efficient open model with strong performance',
    category: 'Small & Efficient',
    recommended: true
  },

  // Multilingual
  {
    id: 'Qwen/Qwen2.5-7B-Instruct',
    name: 'Qwen 2.5 7B Instruct',
    description: 'Alibaba\'s multilingual model with strong reasoning capabilities',
    category: 'Multilingual',
    recommended: true
  },
  {
    id: 'bigscience/bloom-7b1',
    name: 'BLOOM 7B',
    description: 'Multilingual model supporting dozens of languages',
    category: 'Multilingual',
    recommended: false
  }
]

function ModelSelector({ selectedModel, onModelSelect, disabled = false }) {
  const [customModel, setCustomModel] = useState('')
  const [isValidating, setIsValidating] = useState(false)
  const [validationStatus, setValidationStatus] = useState(null)
  const [supportedModels, setSupportedModels] = useState([])
  const [showCustomInput, setShowCustomInput] = useState(false)

  useEffect(() => {
    loadSupportedModels()
  }, [])

  const loadSupportedModels = async () => {
    try {
      const response = await trainingAPI.getSupportedModels()
      setSupportedModels(response.data.models || [])
    } catch (error) {
      console.error('Failed to load supported models:', error)
    }
  }

  const validateCustomModel = async (modelId) => {
    if (!modelId.trim()) {
      setValidationStatus(null)
      return
    }

    setIsValidating(true)
    setValidationStatus(null)

    try {
      const response = await trainingAPI.validateModel(modelId)
      if (response.data.valid) {
        setValidationStatus({ valid: true, message: 'Model validated successfully' })
        onModelSelect({
          id: modelId,
          name: response.data.name || modelId,
          description: response.data.description || 'Custom model',
          custom: true
        })
      } else {
        setValidationStatus({ valid: false, message: response.data.error || 'Model validation failed' })
      }
    } catch (error) {
      setValidationStatus({ 
        valid: false, 
        message: error.response?.data?.detail || 'Failed to validate model' 
      })
    } finally {
      setIsValidating(false)
    }
  }

  const handleCustomModelChange = (value) => {
    setCustomModel(value)
    if (value !== selectedModel?.id) {
      setValidationStatus(null)
    }
  }

  const handleCustomModelSubmit = () => {
    if (customModel.trim()) {
      validateCustomModel(customModel.trim())
    }
  }

  const handlePopularModelSelect = (model) => {
    onModelSelect(model)
    setCustomModel('')
    setValidationStatus(null)
  }

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900 mb-4">Select Base Model</h3>
        
        {/* Use Case Guide */}
        <div className="mb-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
          <h4 className="text-sm font-medium text-blue-900 mb-3">📚 Choose by Use Case</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs text-blue-800">
            <div className="flex items-start space-x-2">
              <span>💬</span>
              <div>
                <strong>General & Chat:</strong> Conversation, general Q&A, content generation
              </div>
            </div>
            <div className="flex items-start space-x-2">
              <span>💻</span>
              <div>
                <strong>Coding:</strong> Code generation, debugging, programming assistance
              </div>
            </div>
            <div className="flex items-start space-x-2">
              <span>🧮</span>
              <div>
                <strong>Reasoning & Math:</strong> Complex problem solving, mathematical tasks
              </div>
            </div>
            <div className="flex items-start space-x-2">
              <span>⚡</span>
              <div>
                <strong>Small & Efficient:</strong> Fast inference, resource-constrained environments
              </div>
            </div>
            <div className="flex items-start space-x-2">
              <span>🌍</span>
              <div>
                <strong>Multilingual:</strong> Non-English languages, international use cases
              </div>
            </div>
          </div>
        </div>

        {/* Popular Models */}
        <div className="mb-6">
          <h4 className="text-sm font-medium text-gray-700 mb-3">Popular Models by Use Case</h4>
          
          {/* Group models by category */}
          {Object.entries(
            POPULAR_MODELS.reduce((acc, model) => {
              if (!acc[model.category]) {
                acc[model.category] = [];
              }
              acc[model.category].push(model);
              return acc;
            }, {})
          ).map(([category, models]) => (
            <div key={category} className="mb-6 last:mb-0">
              <div className="flex items-center mb-3">
                <div className="flex items-center space-x-2">
                  {category === 'General & Chat' && <span className="text-blue-500">💬</span>}
                  {category === 'Coding' && <span className="text-green-500">💻</span>}
                  {category === 'Reasoning & Math' && <span className="text-purple-500">🧮</span>}
                  {category === 'Small & Efficient' && <span className="text-orange-500">⚡</span>}
                  {category === 'Multilingual' && <span className="text-red-500">🌍</span>}
                  <h5 className="text-xs font-medium text-gray-600 uppercase tracking-wide">{category}</h5>
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {models.map((model) => (
                  <button
                    key={model.id}
                    onClick={() => handlePopularModelSelect(model)}
                    disabled={disabled}
                    className={`p-4 text-left border rounded-lg transition-colors relative ${
                      selectedModel?.id === model.id
                        ? 'border-blue-500 bg-blue-50'
                        : 'border-gray-200 hover:border-gray-300'
                    } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                  >
                    {model.recommended && (
                      <div className="absolute top-2 right-2">
                        <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
                          ⭐ Recommended
                        </span>
                      </div>
                    )}
                    <div className="font-medium text-gray-900 pr-16">{model.name}</div>
                    <div className="text-sm text-gray-500 mt-1">{model.description}</div>
                    <div className="text-xs text-gray-400 mt-2 font-mono">{model.id}</div>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Custom Model Input */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-sm font-medium text-gray-700">Custom Hugging Face Model</h4>
            <button
              onClick={() => setShowCustomInput(!showCustomInput)}
              disabled={disabled}
              className="text-sm text-blue-600 hover:text-blue-700 disabled:opacity-50"
            >
              {showCustomInput ? 'Hide' : 'Show'} Custom Input
            </button>
          </div>

          {showCustomInput && (
            <div className="space-y-3">
              <div className="flex space-x-2">
                <div className="flex-1 relative">
                  <input
                    type="text"
                    value={customModel}
                    onChange={(e) => handleCustomModelChange(e.target.value)}
                    placeholder="e.g., microsoft/DialoGPT-medium"
                    disabled={disabled}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
                  />
                  <Search className="absolute right-3 top-2.5 h-4 w-4 text-gray-400" />
                </div>
                <button
                  onClick={handleCustomModelSubmit}
                  disabled={disabled || !customModel.trim() || isValidating}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isValidating ? 'Validating...' : 'Validate'}
                </button>
              </div>

              {/* Validation Status */}
              {validationStatus && (
                <div className={`flex items-center space-x-2 text-sm ${
                  validationStatus.valid ? 'text-green-600' : 'text-red-600'
                }`}>
                  {validationStatus.valid ? (
                    <CheckCircle className="h-4 w-4" />
                  ) : (
                    <AlertCircle className="h-4 w-4" />
                  )}
                  <span>{validationStatus.message}</span>
                </div>
              )}

              <div className="text-xs text-gray-500">
                <span>Enter a Hugging Face model ID. </span>
                <a
                  href="https://huggingface.co/models"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:text-blue-700 inline-flex items-center"
                >
                  Browse models <ExternalLink className="h-3 w-3 ml-1" />
                </a>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Selected Model Display */}
      {selectedModel && (
        <div className="border border-green-200 bg-green-50 rounded-lg p-4">
          <div className="flex items-center space-x-2 text-green-800">
            <CheckCircle className="h-5 w-5" />
            <span className="font-medium">Selected Model</span>
          </div>
          <div className="mt-2">
            <div className="font-medium text-gray-900">{selectedModel.name}</div>
            <div className="text-sm text-gray-600">{selectedModel.description}</div>
            <div className="text-xs text-gray-500 font-mono mt-1">{selectedModel.id}</div>
          </div>
        </div>
      )}
    </div>
  )
}

export default ModelSelector 