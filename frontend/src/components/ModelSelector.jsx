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
    recommended: true,
    potentially_gated: true
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
    id: 'codellama/CodeLlama-7b-Instruct-hf',
    name: 'Code Llama 7B Instruct',
    description: 'Meta\'s specialized coding model based on Llama 2',
    category: 'Coding',
    recommended: true,
    potentially_gated: true
  },
  {
    id: 'WizardLM/WizardCoder-Python-7B-V1.0',
    name: 'WizardCoder Python 7B',
    description: 'Specialized Python coding model with strong performance',
    category: 'Coding',
    recommended: true
  },
  {
    id: 'bigcode/starcoder2-7b',
    name: 'StarCoder2 7B',
    description: 'Advanced code generation model supporting 80+ languages',
    category: 'Coding',
    recommended: false
  },

  // Reasoning & Math
  {
    id: 'microsoft/phi-2',
    name: 'Phi-2',
    description: 'Microsoft\'s efficient model optimized for reasoning tasks',
    category: 'Reasoning & Math',
    recommended: true
  },
  {
    id: 'mistralai/Mistral-7B-v0.1',
    name: 'Mistral 7B v0.1',
    description: 'Open source Mistral 7B model - excellent for general tasks',
    category: 'General & Chat',
    recommended: true,
    note: 'May require access request'
  },

  // Small & Efficient Models
  {
    id: 'microsoft/phi-1_5',
    name: 'Phi-1.5',
    description: 'Compact but powerful model with strong performance',
    category: 'Small & Efficient',
    recommended: true
  },
  {
    id: 'google/gemma-2b',
    name: 'Gemma 2B',
    description: 'Google\'s efficient open model with strong performance',
    category: 'Small & Efficient',
    recommended: true,
    potentially_gated: true
  },
  {
    id: 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    name: 'TinyLlama 1.1B Chat',
    description: 'Ultra-compact chat model, great for experimentation',
    category: 'Small & Efficient',
    recommended: true
  },

  // Multilingual
  {
    id: 'bigscience/bloom-7b1',
    name: 'BLOOM 7B',
    description: 'Multilingual model supporting dozens of languages',
    category: 'Multilingual',
    recommended: true
  },
  {
    id: 'facebook/xglm-7.5B',
    name: 'XGLM 7.5B',
    description: 'Cross-lingual generative model supporting 30+ languages',
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
      const result = response.data
      
      if (result.valid) {
        setValidationStatus({ 
          valid: true, 
          message: result.message || 'Model validated successfully',
          description: result.description
        })
        onModelSelect({
          id: modelId,
          name: result.name || modelId,
          description: result.description || 'Custom model',
          custom: true
        })
      } else {
        setValidationStatus({ 
          valid: false, 
          message: result.message || 'Model validation failed',
          error_type: result.error_type,
          access_request_url: result.access_request_url,
          help_text: result.help_text,
          auth_status: result.auth_status
        })
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
                <strong>Coding:</strong> Programming assistance, code completion, debugging
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
                <strong>Small & Efficient:</strong> Fast inference, lower memory requirements
              </div>
            </div>
          </div>
          <div className="mt-3 pt-3 border-t border-blue-200">
            <div className="flex items-start space-x-2 text-xs text-blue-700">
              <span>🔓</span>
              <div>
                <strong>Note:</strong> All models listed are publicly accessible. Some models on Hugging Face require authentication - we've selected open models to avoid access issues.
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
                    
                    {/* Access indicators */}
                    <div className="mt-3 space-y-2">
                      {model.potentially_gated && (
                        <div className="flex items-center justify-between">
                          <span className="inline-flex items-center px-2 py-1 rounded text-xs bg-yellow-100 text-yellow-800">
                            🔐 May require access
                          </span>
                          <a
                            href={`https://huggingface.co/${model.id}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center px-2 py-1 text-xs text-blue-600 hover:text-blue-700"
                            onClick={(e) => e.stopPropagation()}
                          >
                            <ExternalLink className="h-3 w-3 mr-1" />
                            Request Access
                          </a>
                        </div>
                      )}
                      
                      {model.note && (
                        <div className="text-xs text-amber-600 bg-amber-50 px-2 py-1 rounded">
                          ℹ️ {model.note}
                        </div>
                      )}
                    </div>
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
                <div className={`p-4 rounded-lg border ${
                  validationStatus.valid 
                    ? 'bg-green-50 border-green-200' 
                    : 'bg-red-50 border-red-200'
                }`}>
                  <div className={`flex items-start space-x-2 text-sm ${
                    validationStatus.valid ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {validationStatus.valid ? (
                      <CheckCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
                    ) : (
                      <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
                    )}
                    <div className="flex-1">
                      <div className="font-medium">{validationStatus.message}</div>
                      
                      {/* Show help text and access link for gated models */}
                      {!validationStatus.valid && validationStatus.access_request_url && (
                        <div className="mt-2 space-y-2">
                          {validationStatus.help_text && (
                            <p className="text-xs text-gray-600">{validationStatus.help_text}</p>
                          )}
                          <div className="flex items-center space-x-2">
                            <a
                              href={validationStatus.access_request_url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-flex items-center px-3 py-1 text-xs font-medium rounded-md bg-blue-600 text-white hover:bg-blue-700 transition-colors"
                            >
                              <ExternalLink className="h-3 w-3 mr-1" />
                              Request Model Access
                            </a>
                            <span className="text-xs text-gray-500">
                              (Opens Hugging Face model page)
                            </span>
                          </div>
                          
                          {/* Show authentication status */}
                          {validationStatus.auth_status && (
                            <div className="text-xs text-gray-500">
                              Auth Status: <span className="font-mono">{validationStatus.auth_status}</span>
                              {validationStatus.auth_status === 'no_token' && (
                                <span> • Configure your token in <a href="/settings" className="text-blue-600 hover:underline">Settings</a></span>
                              )}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
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

        {/* Gated Model Help Section */}
        <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <h4 className="text-sm font-medium text-blue-900 mb-3">🔐 About Gated Models</h4>
          <div className="text-xs text-blue-800 space-y-2">
            <p>
              Some models on Hugging Face require permission to access. To use these models:
            </p>
            <ol className="list-decimal list-inside space-y-1 ml-2">
              <li>Configure your Hugging Face token in <a href="/settings" className="text-blue-600 hover:underline font-medium">Settings</a></li>
              <li>Click "Request Access" on the model's page</li>
              <li>Wait for approval (usually instant to a few hours)</li>
              <li>Return here to use the model for training</li>
            </ol>
            <p className="pt-2 border-t border-blue-200">
              <strong>Popular gated models:</strong> Llama models, Code Llama, and some Mistral variants require this process.
            </p>
          </div>
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