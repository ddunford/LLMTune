import React, { useState, useEffect } from 'react'
import { Play, Pause, Square, RotateCcw, CheckCircle, AlertCircle, Clock, Zap } from 'lucide-react'
import { trainingAPI } from '../services/api'

const STATUS_COLORS = {
  pending: 'text-gray-600 bg-gray-100',
  running: 'text-blue-600 bg-blue-100',
  paused: 'text-yellow-600 bg-yellow-100',
  completed: 'text-green-600 bg-green-100',
  failed: 'text-red-600 bg-red-100',
  cancelled: 'text-gray-600 bg-gray-100'
}

const STATUS_ICONS = {
  pending: Clock,
  running: Zap,
  paused: Pause,
  completed: CheckCircle,
  failed: AlertCircle,
  cancelled: Square
}

function TrainingControl({ 
  selectedModel, 
  selectedDataset, 
  config, 
  onTrainingStart, 
  disabled = false 
}) {
  const [currentJob, setCurrentJob] = useState(null)
  const [isStarting, setIsStarting] = useState(false)
  const [error, setError] = useState(null)
  const [jobs, setJobs] = useState([])

  useEffect(() => {
    loadJobs()
    const interval = setInterval(loadJobs, 5000) // Poll every 5 seconds
    return () => clearInterval(interval)
  }, [])

  const loadJobs = async () => {
    try {
      const response = await trainingAPI.getJobs()
      const jobsList = response.data.jobs || []
      setJobs(jobsList)
      
      // Find currently running job
      const runningJob = jobsList.find(job => job.status === 'running' || job.status === 'paused')
      setCurrentJob(runningJob || null)
    } catch (error) {
      console.error('Failed to load jobs:', error)
    }
  }

  const canStartTraining = () => {
    return selectedModel && selectedDataset && !currentJob && !disabled
  }

  const validateConfiguration = () => {
    const errors = []
    
    if (!selectedModel) errors.push('Base model not selected')
    if (!selectedDataset) errors.push('Training dataset not selected')
    if (!config.method) errors.push('Training method not selected')
    if (config.epochs < 1 || config.epochs > 100) errors.push('Epochs must be between 1 and 100')
    if (config.learning_rate < 0.000001 || config.learning_rate > 0.001) errors.push('Learning rate must be between 1e-6 and 1e-3')
    if (config.batch_size < 1 || config.batch_size > 64) errors.push('Batch size must be between 1 and 64')
    
    return errors
  }

  const handleStartTraining = async () => {
    const validationErrors = validateConfiguration()
    if (validationErrors.length > 0) {
      setError(`Configuration errors: ${validationErrors.join(', ')}`)
      return
    }

    setIsStarting(true)
    setError(null)

    try {
      const trainingConfig = {
        base_model: selectedModel.id,
        dataset_path: `uploads/${selectedDataset.id}_${selectedDataset.filename}`,
        method: config.method,
        epochs: config.epochs,
        learning_rate: config.learning_rate,
        batch_size: config.batch_size,
        max_sequence_length: config.max_sequence_length,
        use_dual_gpu: config.use_dual_gpu,
        precision: config.precision,
        gradient_accumulation_steps: config.gradient_accumulation_steps,
        save_steps: config.save_steps,
        validation_split: config.validation_split,
        lora_config: config.lora_config
      }

      // Step 1: Create the job
      const response = await trainingAPI.createJob(trainingConfig)
      const newJob = response.data.job
      
      setCurrentJob(newJob)
      
      // Step 2: Start the job immediately
      await trainingAPI.controlJob(newJob.id, 'start')
      
      // Step 3: Refresh jobs list to get updated status
      await loadJobs()
      
      if (onTrainingStart) {
        onTrainingStart(newJob)
      }
    } catch (error) {
      console.error('Failed to start training:', error)
      setError(error.response?.data?.detail || 'Failed to start training')
    } finally {
      setIsStarting(false)
    }
  }

  const handleControlAction = async (action) => {
    if (!currentJob) return

    try {
      await trainingAPI.controlJob(currentJob.id, action)
      await loadJobs()
    } catch (error) {
      console.error(`Failed to ${action} training:`, error)
      setError(`Failed to ${action} training`)
    }
  }

  const formatDuration = (startTime, endTime) => {
    if (!startTime) return 'Not started'
    
    const start = new Date(startTime)
    const end = endTime ? new Date(endTime) : new Date()
    const duration = Math.floor((end - start) / 1000)
    
    const hours = Math.floor(duration / 3600)
    const minutes = Math.floor((duration % 3600) / 60)
    const seconds = duration % 60
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${seconds}s`
    } else if (minutes > 0) {
      return `${minutes}m ${seconds}s`
    } else {
      return `${seconds}s`
    }
  }

  const getProgressPercentage = () => {
    if (!currentJob || !currentJob.total_steps) return 0
    return Math.round((currentJob.current_step / currentJob.total_steps) * 100)
  }

  const StatusIcon = currentJob ? STATUS_ICONS[currentJob.status] : Clock

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900 mb-4">Training Control</h3>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-md p-4 mb-4">
            <div className="flex items-center">
              <AlertCircle className="h-5 w-5 text-red-400 mr-2" />
              <span className="text-red-800">{error}</span>
            </div>
          </div>
        )}

        {/* Current Job Status */}
        {currentJob ? (
          <div className="border border-gray-200 rounded-lg p-6 mb-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                <StatusIcon className="h-6 w-6 text-gray-600" />
                <div>
                  <div className="font-medium text-gray-900">Training Job #{currentJob.id.slice(0, 8)}</div>
                  <div className="text-sm text-gray-500">
                    Started {new Date(currentJob.created_at).toLocaleString()}
                  </div>
                </div>
              </div>
              <span className={`px-3 py-1 text-sm rounded-full ${STATUS_COLORS[currentJob.status]}`}>
                {currentJob.status.charAt(0).toUpperCase() + currentJob.status.slice(1)}
              </span>
            </div>

            {/* Progress Bar */}
            {currentJob.status === 'running' && currentJob.total_steps && (
              <div className="mb-4">
                <div className="flex justify-between text-sm text-gray-600 mb-2">
                  <span>Progress: {currentJob.current_step}/{currentJob.total_steps} steps</span>
                  <span>{getProgressPercentage()}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${getProgressPercentage()}%` }}
                  />
                </div>
              </div>
            )}

            {/* Job Details */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm mb-4">
              <div>
                <div className="text-gray-500">Model</div>
                <div className="font-medium truncate">{currentJob.config?.base_model}</div>
              </div>
              <div>
                <div className="text-gray-500">Method</div>
                <div className="font-medium">{currentJob.config?.method?.toUpperCase()}</div>
              </div>
              <div>
                <div className="text-gray-500">Epoch</div>
                <div className="font-medium">{currentJob.current_epoch}/{currentJob.total_epochs}</div>
              </div>
              <div>
                <div className="text-gray-500">Duration</div>
                <div className="font-medium">
                  {formatDuration(currentJob.started_at, currentJob.completed_at)}
                </div>
              </div>
            </div>

            {/* Metrics */}
            {(currentJob.loss || currentJob.validation_loss) && (
              <div className="grid grid-cols-2 gap-4 text-sm mb-4">
                {currentJob.loss && (
                  <div>
                    <div className="text-gray-500">Training Loss</div>
                    <div className="font-medium">{currentJob.loss.toFixed(4)}</div>
                  </div>
                )}
                {currentJob.validation_loss && (
                  <div>
                    <div className="text-gray-500">Validation Loss</div>
                    <div className="font-medium">{currentJob.validation_loss.toFixed(4)}</div>
                  </div>
                )}
              </div>
            )}

            {/* Control Buttons */}
            <div className="flex space-x-3">
              {currentJob.status === 'running' && (
                <button
                  onClick={() => handleControlAction('pause')}
                  className="flex items-center space-x-2 px-4 py-2 bg-yellow-600 text-white rounded-md hover:bg-yellow-700"
                >
                  <Pause className="h-4 w-4" />
                  <span>Pause</span>
                </button>
              )}
              
              {currentJob.status === 'paused' && (
                <button
                  onClick={() => handleControlAction('resume')}
                  className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
                >
                  <Play className="h-4 w-4" />
                  <span>Resume</span>
                </button>
              )}
              
              {(currentJob.status === 'running' || currentJob.status === 'paused') && (
                <button
                  onClick={() => handleControlAction('cancel')}
                  className="flex items-center space-x-2 px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
                >
                  <Square className="h-4 w-4" />
                  <span>Cancel</span>
                </button>
              )}
            </div>
          </div>
        ) : (
          /* Start Training Section */
          <div className="border border-gray-200 rounded-lg p-6 mb-6">
            <div className="text-center">
              <div className="mb-4">
                <Play className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h4 className="text-lg font-medium text-gray-900 mb-2">Ready to Start Training</h4>
                <p className="text-gray-600">Configure your model, dataset, and parameters above, then start training.</p>
              </div>

              {/* Prerequisites Checklist */}
              <div className="space-y-2 mb-6">
                <div className={`flex items-center justify-center space-x-2 text-sm ${
                  selectedModel ? 'text-green-600' : 'text-gray-400'
                }`}>
                  {selectedModel ? <CheckCircle className="h-4 w-4" /> : <Clock className="h-4 w-4" />}
                  <span>Base model selected</span>
                </div>
                <div className={`flex items-center justify-center space-x-2 text-sm ${
                  selectedDataset ? 'text-green-600' : 'text-gray-400'
                }`}>
                  {selectedDataset ? <CheckCircle className="h-4 w-4" /> : <Clock className="h-4 w-4" />}
                  <span>Training dataset selected</span>
                </div>
                <div className={`flex items-center justify-center space-x-2 text-sm ${
                  config.method ? 'text-green-600' : 'text-gray-400'
                }`}>
                  {config.method ? <CheckCircle className="h-4 w-4" /> : <Clock className="h-4 w-4" />}
                  <span>Training configuration set</span>
                </div>
              </div>

              <button
                onClick={handleStartTraining}
                disabled={!canStartTraining() || isStarting}
                className="flex items-center space-x-2 px-6 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed mx-auto"
              >
                {isStarting ? (
                  <>
                    <RotateCcw className="h-5 w-5 animate-spin" />
                    <span>Starting Training...</span>
                  </>
                ) : (
                  <>
                    <Play className="h-5 w-5" />
                    <span>Start Training</span>
                  </>
                )}
              </button>
            </div>
          </div>
        )}

        {/* Recent Jobs */}
        {jobs.length > 0 && (
          <div>
            <h4 className="text-md font-medium text-gray-900 mb-3">Recent Training Jobs</h4>
            <div className="space-y-2">
              {jobs.slice(0, 5).map((job) => {
                const JobStatusIcon = STATUS_ICONS[job.status]
                return (
                  <div key={job.id} className="flex items-center justify-between p-3 border border-gray-200 rounded-md">
                    <div className="flex items-center space-x-3">
                      <JobStatusIcon className="h-4 w-4 text-gray-500" />
                      <div>
                        <div className="text-sm font-medium">#{job.id.slice(0, 8)}</div>
                        <div className="text-xs text-gray-500">
                          {job.config?.base_model} • {job.config?.method?.toUpperCase()}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <span className={`px-2 py-1 text-xs rounded-full ${STATUS_COLORS[job.status]}`}>
                        {job.status}
                      </span>
                      <div className="text-xs text-gray-500 mt-1">
                        {formatDuration(job.started_at, job.completed_at)}
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default TrainingControl 