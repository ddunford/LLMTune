import React, { useState } from 'react'
import { Brain, ChevronRight, ChevronDown } from 'lucide-react'
import ModelSelector from '../components/ModelSelector'
import DatasetSelector from '../components/DatasetSelector'
import TrainingConfig from '../components/TrainingConfig'
import TrainingControl from '../components/TrainingControl'

function Training() {
  const [selectedModel, setSelectedModel] = useState(null)
  const [selectedDataset, setSelectedDataset] = useState(null)
  const [trainingConfig, setTrainingConfig] = useState({
    method: 'lora',
    epochs: 3,
    learning_rate: 0.0002,
    batch_size: 4,
    max_sequence_length: 2048,
    use_dual_gpu: true,
    precision: 'fp16',
    gradient_accumulation_steps: 1,
    save_steps: 500,
    validation_split: 0.1,
    lora_config: {
      rank: 16,
      alpha: 32,
      dropout: 0.1,
      target_modules: ["q_proj", "v_proj"]
    }
  })
  const [currentTrainingJob, setCurrentTrainingJob] = useState(null)
  const [expandedSections, setExpandedSections] = useState({
    model: true,
    dataset: true,
    config: true,
    control: true
  })

  const toggleSection = (section) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }))
  }

  const handleTrainingStart = (job) => {
    setCurrentTrainingJob(job)
  }

  const isTrainingActive = currentTrainingJob?.status === 'running' || currentTrainingJob?.status === 'paused'

  const SectionHeader = ({ title, expanded, onToggle, stepNumber, completed = false }) => (
    <button
      onClick={onToggle}
      className="w-full flex items-center justify-between p-4 bg-gray-50 hover:bg-gray-100 rounded-lg transition-colors"
    >
      <div className="flex items-center space-x-3">
        <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white text-sm font-medium ${
          completed ? 'bg-green-600' : 'bg-blue-600'
        }`}>
          {stepNumber}
        </div>
        <h2 className="text-lg font-medium text-gray-900">{title}</h2>
      </div>
      {expanded ? (
        <ChevronDown className="h-5 w-5 text-gray-400" />
      ) : (
        <ChevronRight className="h-5 w-5 text-gray-400" />
      )}
    </button>
  )

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Page Header */}
      <div className="text-center py-6">
        <Brain className="h-12 w-12 text-blue-600 mx-auto mb-4" />
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Train Your LLM</h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Configure and start training your Large Language Model with LoRA, QLoRA, or full fine-tuning. 
          Optimized for dual RTX 3060 GPUs with real-time monitoring.
        </p>
      </div>

      {/* Training Status Alert */}
      {isTrainingActive && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 text-blue-800">
            <Brain className="h-5 w-5" />
            <span className="font-medium">Training in Progress</span>
          </div>
          <p className="text-blue-700 mt-1">
            A training job is currently {currentTrainingJob.status}. You can monitor progress in the Training Control section below.
          </p>
        </div>
      )}

      {/* Step 1: Model Selection */}
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        <SectionHeader
          title="Select Base Model"
          expanded={expandedSections.model}
          onToggle={() => toggleSection('model')}
          stepNumber={1}
          completed={!!selectedModel}
        />
        {expandedSections.model && (
          <div className="p-6 border-t border-gray-200">
            <ModelSelector
              selectedModel={selectedModel}
              onModelSelect={setSelectedModel}
              disabled={isTrainingActive}
            />
          </div>
        )}
      </div>

      {/* Step 2: Dataset Selection */}
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        <SectionHeader
          title="Select Training Dataset"
          expanded={expandedSections.dataset}
          onToggle={() => toggleSection('dataset')}
          stepNumber={2}
          completed={!!selectedDataset}
        />
        {expandedSections.dataset && (
          <div className="p-6 border-t border-gray-200">
            <DatasetSelector
              selectedDataset={selectedDataset}
              onDatasetSelect={setSelectedDataset}
              disabled={isTrainingActive}
            />
          </div>
        )}
      </div>

      {/* Step 3: Training Configuration */}
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        <SectionHeader
          title="Configure Training Parameters"
          expanded={expandedSections.config}
          onToggle={() => toggleSection('config')}
          stepNumber={3}
          completed={!!trainingConfig.method}
        />
        {expandedSections.config && (
          <div className="p-6 border-t border-gray-200">
            <TrainingConfig
              config={trainingConfig}
              onConfigChange={setTrainingConfig}
              disabled={isTrainingActive}
            />
          </div>
        )}
      </div>

      {/* Step 4: Training Control */}
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        <SectionHeader
          title="Start & Monitor Training"
          expanded={expandedSections.control}
          onToggle={() => toggleSection('control')}
          stepNumber={4}
          completed={!!currentTrainingJob}
        />
        {expandedSections.control && (
          <div className="p-6 border-t border-gray-200">
            <TrainingControl
              selectedModel={selectedModel}
              selectedDataset={selectedDataset}
              config={trainingConfig}
              onTrainingStart={handleTrainingStart}
            />
          </div>
        )}
      </div>

      {/* Quick Tips */}
      <div className="bg-gray-50 rounded-lg p-6">
        <h3 className="text-lg font-medium text-gray-900 mb-4">💡 Quick Tips</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-sm text-gray-700">
          <div>
            <div className="font-medium text-gray-900 mb-1">Choosing Base Models</div>
            <ul className="space-y-1">
              <li>• <strong>💬 General & Chat:</strong> Best for chatbots, Q&A systems</li>
              <li>• <strong>💻 Coding:</strong> Specialized for programming tasks</li>
              <li>• <strong>🧮 Reasoning:</strong> Excel at math and complex logic</li>
              <li>• <strong>⚡ Small & Efficient:</strong> Fast inference, lower memory</li>
              <li>• <strong>🌍 Multilingual:</strong> Support for non-English languages</li>
            </ul>
          </div>
          <div>
            <div className="font-medium text-gray-900 mb-1">Choosing Training Method</div>
            <ul className="space-y-1">
              <li>• <strong>LoRA:</strong> Best balance of performance and memory usage</li>
              <li>• <strong>QLoRA:</strong> Most memory efficient for dual RTX 3060</li>
              <li>• <strong>Full:</strong> Complete retraining (requires more memory)</li>
            </ul>
          </div>
          <div>
            <div className="font-medium text-gray-900 mb-1">Optimizing Performance</div>
            <ul className="space-y-1">
              <li>• Start with smaller batch sizes to avoid OOM errors</li>
              <li>• Use FP16 precision for better memory efficiency</li>
              <li>• Enable dual GPU for faster training</li>
              <li>• Monitor GPU temperature during training</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="text-center text-sm text-gray-500 py-4">
        <p>
          Training progress and logs will be available in real-time. 
          You can switch to the <strong>Monitoring</strong> page for detailed GPU statistics and system metrics.
        </p>
      </div>
    </div>
  )
}

export default Training 