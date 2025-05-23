import React, { useState } from 'react'
import { Settings, Info, Cpu, Zap } from 'lucide-react'

const TRAINING_METHODS = [
  {
    id: 'lora',
    name: 'LoRA',
    description: 'Low-Rank Adaptation - Memory efficient with good performance',
    recommended: 'Single RTX 3060',
    icon: '🚀'
  },
  {
    id: 'qlora',
    name: 'QLoRA',
    description: 'Quantized LoRA - Most memory efficient for consumer GPUs',
    recommended: 'Dual RTX 3060',
    icon: '⚡'
  },
  {
    id: 'full',
    name: 'Full Fine-tuning',
    description: 'Complete model retraining - Requires significant memory',
    recommended: 'High-end hardware',
    icon: '💪'
  }
]

const PRECISION_OPTIONS = [
  { id: 'fp16', name: 'FP16', description: 'Half precision (recommended)' },
  { id: 'bf16', name: 'BF16', description: 'Brain float 16' },
  { id: 'fp32', name: 'FP32', description: 'Full precision' }
]

function TrainingConfig({ config, onConfigChange, disabled = false }) {
  const [showAdvanced, setShowAdvanced] = useState(false)

  const updateConfig = (updates) => {
    onConfigChange({ ...config, ...updates })
  }

  const updateLoRAConfig = (updates) => {
    onConfigChange({
      ...config,
      lora_config: { ...config.lora_config, ...updates }
    })
  }

  const isLoRAMethod = config.method === 'lora' || config.method === 'qlora'

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900 mb-4">Training Configuration</h3>

        {/* Training Method Selection */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-3">Training Method</label>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {TRAINING_METHODS.map((method) => (
              <button
                key={method.id}
                onClick={() => updateConfig({ method: method.id })}
                disabled={disabled}
                className={`p-4 text-left border rounded-lg transition-all ${
                  config.method === method.id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                <div className="flex items-center space-x-2 mb-2">
                  <span className="text-lg">{method.icon}</span>
                  <div className="font-medium text-gray-900">{method.name}</div>
                </div>
                <div className="text-sm text-gray-600 mb-2">{method.description}</div>
                <div className="text-xs text-gray-500">Recommended: {method.recommended}</div>
              </button>
            ))}
          </div>
        </div>

        {/* Basic Training Parameters */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Epochs
              <span className="text-xs text-gray-500 ml-1">(1-100)</span>
            </label>
            <input
              type="number"
              min="1"
              max="100"
              value={config.epochs}
              onChange={(e) => updateConfig({ epochs: parseInt(e.target.value) || 1 })}
              disabled={disabled}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Learning Rate
              <span className="text-xs text-gray-500 ml-1">(1e-6 to 1e-3)</span>
            </label>
            <input
              type="number"
              step="0.0001"
              min="0.000001"
              max="0.001"
              value={config.learning_rate}
              onChange={(e) => updateConfig({ learning_rate: parseFloat(e.target.value) || 0.0002 })}
              disabled={disabled}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Batch Size
              <span className="text-xs text-gray-500 ml-1">(1-64)</span>
            </label>
            <select
              value={config.batch_size}
              onChange={(e) => updateConfig({ batch_size: parseInt(e.target.value) })}
              disabled={disabled}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
            >
              {[1, 2, 4, 8, 16, 32].map(size => (
                <option key={size} value={size}>{size}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Max Sequence Length
              <span className="text-xs text-gray-500 ml-1">(128-4096)</span>
            </label>
            <select
              value={config.max_sequence_length}
              onChange={(e) => updateConfig({ max_sequence_length: parseInt(e.target.value) })}
              disabled={disabled}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
            >
              {[512, 1024, 2048, 4096].map(length => (
                <option key={length} value={length}>{length}</option>
              ))}
            </select>
          </div>
        </div>

        {/* LoRA Configuration */}
        {isLoRAMethod && (
          <div className="border border-gray-200 rounded-lg p-4 mb-6">
            <div className="flex items-center space-x-2 mb-4">
              <Settings className="h-5 w-5 text-gray-600" />
              <h4 className="text-md font-medium text-gray-900">LoRA Configuration</h4>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Rank (r)
                  <span className="text-xs text-gray-500 ml-1">(1-256)</span>
                </label>
                <input
                  type="number"
                  min="1"
                  max="256"
                  value={config.lora_config?.rank || 16}
                  onChange={(e) => updateLoRAConfig({ rank: parseInt(e.target.value) || 16 })}
                  disabled={disabled}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
                />
                <div className="text-xs text-gray-500 mt-1">Higher = more parameters</div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Alpha
                  <span className="text-xs text-gray-500 ml-1">(1-512)</span>
                </label>
                <input
                  type="number"
                  min="1"
                  max="512"
                  value={config.lora_config?.alpha || 32}
                  onChange={(e) => updateLoRAConfig({ alpha: parseInt(e.target.value) || 32 })}
                  disabled={disabled}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
                />
                <div className="text-xs text-gray-500 mt-1">Scaling factor</div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Dropout
                  <span className="text-xs text-gray-500 ml-1">(0.0-0.5)</span>
                </label>
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  max="0.5"
                  value={config.lora_config?.dropout || 0.1}
                  onChange={(e) => updateLoRAConfig({ dropout: parseFloat(e.target.value) || 0.1 })}
                  disabled={disabled}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
                />
                <div className="text-xs text-gray-500 mt-1">Regularization</div>
              </div>
            </div>
          </div>
        )}

        {/* Compute Configuration */}
        <div className="border border-gray-200 rounded-lg p-4 mb-6">
          <div className="flex items-center space-x-2 mb-4">
            <Cpu className="h-5 w-5 text-gray-600" />
            <h4 className="text-md font-medium text-gray-900">Compute Configuration</h4>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-3">GPU Setup</label>
              <div className="space-y-2">
                <label className="flex items-center">
                  <input
                    type="radio"
                    checked={!config.use_dual_gpu}
                    onChange={() => updateConfig({ use_dual_gpu: false })}
                    disabled={disabled}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 disabled:opacity-50"
                  />
                  <span className="ml-2 text-sm text-gray-700">Single GPU</span>
                </label>
                <label className="flex items-center">
                  <input
                    type="radio"
                    checked={config.use_dual_gpu}
                    onChange={() => updateConfig({ use_dual_gpu: true })}
                    disabled={disabled}
                    className="h-4 w-4 text-blue-600 focus:ring-blue-500 disabled:opacity-50"
                  />
                  <span className="ml-2 text-sm text-gray-700">Dual GPU (Recommended)</span>
                </label>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">Precision</label>
              <select
                value={config.precision}
                onChange={(e) => updateConfig({ precision: e.target.value })}
                disabled={disabled}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
              >
                {PRECISION_OPTIONS.map(option => (
                  <option key={option.id} value={option.id}>
                    {option.name} - {option.description}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Advanced Settings */}
        <div className="border border-gray-200 rounded-lg p-4">
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center space-x-2 w-full text-left"
          >
            <Zap className="h-5 w-5 text-gray-600" />
            <h4 className="text-md font-medium text-gray-900">Advanced Settings</h4>
            <span className="text-gray-400">
              {showAdvanced ? '▲' : '▼'}
            </span>
          </button>

          {showAdvanced && (
            <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Gradient Accumulation Steps
                  <span className="text-xs text-gray-500 ml-1">(1-16)</span>
                </label>
                <input
                  type="number"
                  min="1"
                  max="16"
                  value={config.gradient_accumulation_steps}
                  onChange={(e) => updateConfig({ gradient_accumulation_steps: parseInt(e.target.value) || 1 })}
                  disabled={disabled}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Save Checkpoint Every (steps)
                  <span className="text-xs text-gray-500 ml-1">(100-2000)</span>
                </label>
                <input
                  type="number"
                  min="100"
                  max="2000"
                  step="100"
                  value={config.save_steps}
                  onChange={(e) => updateConfig({ save_steps: parseInt(e.target.value) || 500 })}
                  disabled={disabled}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Validation Split
                  <span className="text-xs text-gray-500 ml-1">(0.0-0.3)</span>
                </label>
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  max="0.3"
                  value={config.validation_split}
                  onChange={(e) => updateConfig({ validation_split: parseFloat(e.target.value) || 0.1 })}
                  disabled={disabled}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:opacity-50"
                />
              </div>
            </div>
          )}
        </div>

        {/* Configuration Summary */}
        <div className="bg-gray-50 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-3">
            <Info className="h-5 w-5 text-blue-600" />
            <h4 className="text-md font-medium text-gray-900">Configuration Summary</h4>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <div className="text-gray-500">Method</div>
              <div className="font-medium">{config.method?.toUpperCase()}</div>
            </div>
            <div>
              <div className="text-gray-500">Epochs</div>
              <div className="font-medium">{config.epochs}</div>
            </div>
            <div>
              <div className="text-gray-500">Batch Size</div>
              <div className="font-medium">{config.batch_size}</div>
            </div>
            <div>
              <div className="text-gray-500">Learning Rate</div>
              <div className="font-medium">{config.learning_rate}</div>
            </div>
            {isLoRAMethod && (
              <>
                <div>
                  <div className="text-gray-500">LoRA Rank</div>
                  <div className="font-medium">{config.lora_config?.rank}</div>
                </div>
                <div>
                  <div className="text-gray-500">LoRA Alpha</div>
                  <div className="font-medium">{config.lora_config?.alpha}</div>
                </div>
              </>
            )}
            <div>
              <div className="text-gray-500">GPU Setup</div>
              <div className="font-medium">{config.use_dual_gpu ? 'Dual' : 'Single'}</div>
            </div>
            <div>
              <div className="text-gray-500">Precision</div>
              <div className="font-medium">{config.precision?.toUpperCase()}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default TrainingConfig 