import React, { useState, useEffect, useRef } from 'react'
import { Activity, Cpu, Thermometer, Zap, HardDrive, Clock, TrendingUp, Eye, Pause, Play, Square, Trash2, RefreshCw, Download, MessageSquare } from 'lucide-react'
import * as api from '../services/api'

function Monitoring() {
  const [jobs, setJobs] = useState([])
  const [selectedJob, setSelectedJob] = useState(null)
  const [logs, setLogs] = useState([])
  const [gpuStats, setGpuStats] = useState({ gpus: [] })
  const [loading, setLoading] = useState(true)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const logsRef = useRef(null)

  useEffect(() => {
    loadJobs()
    loadGpuStats()
    
    // Set up polling for real-time updates
    const interval = setInterval(() => {
      if (autoRefresh) {
        loadJobs()
        loadGpuStats()
        if (selectedJob) {
          loadJobLogs(selectedJob.id)
        }
      }
    }, 5000) // Update every 5 seconds

    return () => clearInterval(interval)
  }, [autoRefresh, selectedJob])

  useEffect(() => {
    // Auto-scroll logs to bottom
    if (logsRef.current) {
      logsRef.current.scrollTop = logsRef.current.scrollHeight
    }
  }, [logs])

  const loadJobs = async () => {
    try {
      const response = await api.trainingAPI.getJobs()
      
      // Try both response formats
      const jobs = response.jobs || response.data?.jobs || []
      setJobs(jobs)
      
      // Auto-select running job if none selected
      if (!selectedJob && jobs.length > 0) {
        const runningJob = jobs.find(job => job.status === 'running')
        if (runningJob) {
          setSelectedJob(runningJob)
          loadJobLogs(runningJob.id)
        }
      }
    } catch (error) {
      console.error('Failed to load jobs:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadGpuStats = async () => {
    try {
      const response = await api.monitoringAPI.getStats()
      setGpuStats(response.data)
    } catch (error) {
      console.error('Failed to load GPU stats:', error)
    }
  }

  const loadJobLogs = async (jobId) => {
    try {
      const response = await api.trainingAPI.getLogs(jobId)
      setLogs(response.logs || [])
    } catch (error) {
      console.error('Failed to load logs:', error)
    }
  }

  const handleJobControl = async (jobId, action) => {
    try {
      await api.trainingAPI.controlJob(jobId, action)
      await loadJobs() // Refresh to get updated status
    } catch (error) {
      console.error(`Failed to ${action} job:`, error)
    }
  }

  const handleDeleteJob = async (jobId) => {
    if (!confirm('Are you sure you want to delete this job? This action cannot be undone.')) {
      return
    }
    
    try {
      await api.trainingAPI.deleteJob(jobId)
      await loadJobs() // Refresh to update the jobs list
      
      // Clear selected job if it was deleted
      if (selectedJob?.id === jobId) {
        setSelectedJob(null)
        setLogs([])
      }
    } catch (error) {
      console.error('Failed to delete job:', error)
    }
  }

  const handleClearCompletedJobs = async () => {
    const completedJobs = jobs.filter(job => job.status === 'completed' || job.status === 'failed')
    
    if (completedJobs.length === 0) {
      alert('No completed or failed jobs to delete.')
      return
    }
    
    if (!confirm(`Are you sure you want to delete all ${completedJobs.length} completed/failed jobs? This action cannot be undone.`)) {
      return
    }
    
    try {
      // Delete all completed/failed jobs
      await Promise.all(completedJobs.map(job => api.trainingAPI.deleteJob(job.id)))
      await loadJobs() // Refresh to update the jobs list
      
      // Clear selected job if it was deleted
      if (selectedJob && (selectedJob.status === 'completed' || selectedJob.status === 'failed')) {
        setSelectedJob(null)
        setLogs([])
      }
    } catch (error) {
      console.error('Failed to delete jobs:', error)
    }
  }

  const handleCleanupFiles = async () => {
    if (!confirm('This will remove orphaned config and log files that don\'t have corresponding jobs. Continue?')) {
      return
    }
    
    try {
      const response = await api.trainingAPI.cleanup()
      const result = response.data || response
      
      if (result.count > 0) {
        alert(`✅ Cleanup completed!\n\nRemoved ${result.count} orphaned files:\n${result.files_removed.join('\n')}`)
      } else {
        alert('✅ No orphaned files found. Everything is clean!')
      }
    } catch (error) {
      console.error('Failed to cleanup files:', error)
      alert('❌ Error during cleanup: ' + (error.response?.data?.detail || error.message))
    }
  }

  const handleRestoreJobs = async () => {
    if (!confirm('This will restore training jobs from existing config files. This is useful if jobs were lost after a server restart. Continue?')) {
      return
    }
    
    try {
      const response = await api.trainingAPI.restore()
      const result = response.data || response
      
      if (result.restored_count > 0) {
        alert(`✅ Job restoration completed!\n\nRestored ${result.restored_count} jobs from config files:\n${result.jobs.map(job => `${job.id}: ${job.base_model} (${job.status})`).join('\n')}`)
        await loadJobs() // Refresh the jobs list
      } else {
        alert('✅ No jobs found to restore. All existing config files are already loaded.')
      }
    } catch (error) {
      console.error('Failed to restore jobs:', error)
      alert('❌ Error during job restoration: ' + (error.response?.data?.detail || error.message))
    }
  }

  const handleDownloadModel = async (jobId, fileType = 'adapter') => {
    try {
      const response = await api.trainingAPI.downloadModel(jobId, fileType)
      
      // Create download link
      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      
      // Set filename based on file type
      const job = jobs.find(j => j.id === jobId)
      const modelName = job?.config?.base_model?.split('/').pop() || 'model'
      const filename = `${modelName}_${jobId}_${fileType}.${fileType === 'adapter' ? 'zip' : fileType === 'logs' ? 'log' : 'yaml'}`
      
      link.setAttribute('download', filename)
      document.body.appendChild(link)
      link.click()
      link.remove()
      window.URL.revokeObjectURL(url)
      
    } catch (error) {
      console.error('Failed to download model:', error)
      alert('❌ Failed to download model: ' + (error.response?.data?.detail || error.message))
    }
  }

  const handleTestModel = (jobId) => {
    // Navigate to inference page with the selected model
    window.location.href = `/inference?model=${jobId}`
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'running': return 'text-green-600 bg-green-100'
      case 'completed': return 'text-blue-600 bg-blue-100'
      case 'failed': return 'text-red-600 bg-red-100'
      case 'paused': return 'text-yellow-600 bg-yellow-100'
      case 'pending': return 'text-gray-600 bg-gray-100'
      default: return 'text-gray-600 bg-gray-100'
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

  const formatProgress = (currentStep, totalSteps) => {
    if (!totalSteps || totalSteps === 0) return '0%'
    return `${Math.round((currentStep / totalSteps) * 100)}%`
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Training Monitor</h1>
          <p className="text-gray-600">Real-time training and system monitoring</p>
        </div>
        
        <div className="flex items-center space-x-4">
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
              className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="text-sm text-gray-700">Auto-refresh</span>
          </label>
          
          <button
            onClick={handleCleanupFiles}
            className="flex items-center px-3 py-2 text-sm bg-orange-100 text-orange-700 rounded hover:bg-orange-200"
          >
            <HardDrive className="h-4 w-4 mr-1" />
            Clean Up Files
          </button>
          
          <button
            onClick={handleRestoreJobs}
            className="flex items-center px-3 py-2 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
          >
            <RefreshCw className="h-4 w-4 mr-1" />
            Restore Jobs
          </button>
          
          <div className="text-sm text-gray-500">
            {autoRefresh && <span className="text-green-600">● Live</span>}
          </div>
        </div>
      </div>

      {/* GPU Statistics */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <Cpu className="h-5 w-5 mr-2" />
          GPU Statistics
        </h2>
        
        {gpuStats.gpus && gpuStats.gpus.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {gpuStats.gpus.map((gpu, idx) => (
              <div key={idx} className="space-y-4">
                <h3 className="font-medium text-gray-900">GPU {idx} - {gpu.name || 'Unknown'}</h3>
                
                <div className="space-y-3">
                  {/* Memory Usage */}
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-600">Memory</span>
                      <span className="text-gray-900">
                        {gpu.memory_used || 0}GB / {gpu.memory_total || 0}GB
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all"
                        style={{
                          width: `${gpu.memory_total ? (gpu.memory_used / gpu.memory_total) * 100 : 0}%`
                        }}
                      ></div>
                    </div>
                  </div>

                  {/* GPU Utilization */}
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-600">Utilization</span>
                      <span className="text-gray-900">{gpu.utilization || 0}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-green-600 h-2 rounded-full transition-all"
                        style={{ width: `${gpu.utilization || 0}%` }}
                      ></div>
                    </div>
                  </div>

                  {/* Temperature */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center text-sm text-gray-600">
                      <Thermometer className="h-4 w-4 mr-1" />
                      Temperature
                    </div>
                    <span className="text-sm text-gray-900">{gpu.temperature || 0}°C</span>
                  </div>

                  {/* Power */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center text-sm text-gray-600">
                      <Zap className="h-4 w-4 mr-1" />
                      Power
                    </div>
                    <span className="text-sm text-gray-900">{gpu.power_draw || 0}W</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-500">No GPU information available</p>
        )}
      </div>

      {/* Training Jobs */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Jobs List */}
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900 flex items-center">
              <Activity className="h-5 w-5 mr-2" />
              Training Jobs
            </h2>
            
            {jobs.filter(job => job.status === 'completed' || job.status === 'failed').length > 0 && (
              <button
                onClick={handleClearCompletedJobs}
                className="flex items-center px-3 py-1 text-xs bg-gray-100 text-gray-700 rounded hover:bg-gray-200"
              >
                <Trash2 className="h-3 w-3 mr-1" />
                Clear Completed
              </button>
            )}
          </div>
          
          {jobs.length === 0 ? (
            <p className="text-gray-500 text-sm">No training jobs found</p>
          ) : (
            <div className="space-y-3">
              {jobs.map((job) => {
                return (
                <div
                  key={job.id}
                  className={`p-3 rounded-lg border-2 cursor-pointer transition-colors ${
                    selectedJob?.id === job.id
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                  onClick={() => {
                    setSelectedJob(job)
                    loadJobLogs(job.id)
                  }}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-sm text-gray-900 truncate">
                      {job.config?.base_model?.split('/').pop() || 'Unknown Model'}
                    </span>
                    <span className={`px-2 py-1 text-xs rounded-full ${getStatusColor(job.status)}`}>
                      {job.status}
                    </span>
                  </div>
                  
                  <div className="text-xs text-gray-600 space-y-1">
                    <div>Started: {job.started_at ? new Date(job.started_at).toLocaleTimeString() : 'Not started'}</div>
                    <div>Duration: {formatDuration(job.started_at, job.completed_at)}</div>
                    {job.current_step > 0 && job.total_steps > 0 && (
                      <div>Progress: {job.current_step}/{job.total_steps} steps</div>
                    )}
                  </div>

                  {/* Job Controls */}
                  {job.status === 'running' && (
                    <div className="flex space-x-1 mt-2">
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          handleJobControl(job.id, 'pause')
                        }}
                        className="flex items-center px-2 py-1 text-xs bg-yellow-100 text-yellow-700 rounded hover:bg-yellow-200"
                      >
                        <Pause className="h-3 w-3 mr-1" />
                        Pause
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          handleJobControl(job.id, 'cancel')
                        }}
                        className="flex items-center px-2 py-1 text-xs bg-red-100 text-red-700 rounded hover:bg-red-200"
                      >
                        <Square className="h-3 w-3 mr-1" />
                        Stop
                      </button>
                    </div>
                  )}

                  {job.status === 'paused' && (
                    <div className="flex space-x-1 mt-2">
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          handleJobControl(job.id, 'resume')
                        }}
                        className="flex items-center px-2 py-1 text-xs bg-green-100 text-green-700 rounded hover:bg-green-200"
                      >
                        <Play className="h-3 w-3 mr-1" />
                        Resume
                      </button>
                    </div>
                  )}

                  {job.status === 'pending' && (
                    <div className="flex space-x-1 mt-2">
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          handleJobControl(job.id, 'start')
                        }}
                        className="flex items-center px-2 py-1 text-xs bg-green-100 text-green-700 rounded hover:bg-green-200"
                      >
                        <Play className="h-3 w-3 mr-1" />
                        Start
                      </button>
                    </div>
                  )}

                  {(job.status === 'completed' || job.status === 'failed') && (
                    <div className="flex space-x-1 mt-2">
                      {job.status === 'failed' && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation()
                            handleJobControl(job.id, 'restart')
                          }}
                          className="flex items-center px-2 py-1 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200"
                        >
                          <RefreshCw className="h-3 w-3 mr-1" />
                          Restart
                        </button>
                      )}
                      
                      {job.status === 'completed' && (
                        <>
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              handleDownloadModel(job.id, 'adapter')
                            }}
                            className="flex items-center px-2 py-1 text-xs bg-green-100 text-green-700 rounded hover:bg-green-200"
                          >
                            <Download className="h-3 w-3 mr-1" />
                            Download
                          </button>
                          
                          <button
                            onClick={(e) => {
                              e.stopPropagation()
                              handleTestModel(job.id)
                            }}
                            className="flex items-center px-2 py-1 text-xs bg-purple-100 text-purple-700 rounded hover:bg-purple-200"
                          >
                            <MessageSquare className="h-3 w-3 mr-1" />
                            Test
                          </button>
                        </>
                      )}
                      
                      <button
                        onClick={(e) => {
                          e.stopPropagation()
                          handleDeleteJob(job.id)
                        }}
                        className="flex items-center px-2 py-1 text-xs bg-red-100 text-red-700 rounded hover:bg-red-200"
                      >
                        <Trash2 className="h-3 w-3 mr-1" />
                        Delete
                      </button>
                    </div>
                  )}
                </div>
              )})}
            </div>
          )}
        </div>

        {/* Logs and Progress */}
        <div className="lg:col-span-2 space-y-6">
          {selectedJob ? (
            <>
              {/* Job Details */}
              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Job Details</h2>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <div className="text-gray-600">Model</div>
                    <div className="font-medium text-gray-900 truncate">
                      {selectedJob.config?.base_model?.split('/').pop() || 'Unknown'}
                    </div>
                  </div>
                  
                  <div>
                    <div className="text-gray-600">Method</div>
                    <div className="font-medium text-gray-900">{selectedJob.config?.method || 'Unknown'}</div>
                  </div>
                  
                  <div>
                    <div className="text-gray-600">Progress</div>
                    <div className="font-medium text-gray-900">
                      {formatProgress(selectedJob.current_step, selectedJob.total_steps)}
                    </div>
                  </div>
                  
                  <div>
                    <div className="text-gray-600">Loss</div>
                    <div className="font-medium text-gray-900">
                      {selectedJob.loss ? selectedJob.loss.toFixed(4) : 'N/A'}
                    </div>
                  </div>
                </div>

                {/* Progress Bar */}
                {selectedJob.total_steps > 0 && (
                  <div className="mt-4">
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-600">Training Progress</span>
                      <span className="text-gray-900">
                        {selectedJob.current_step}/{selectedJob.total_steps} steps
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full transition-all"
                        style={{
                          width: `${(selectedJob.current_step / selectedJob.total_steps) * 100}%`
                        }}
                      ></div>
                    </div>
                  </div>
                )}
              </div>

              {/* Training Logs */}
              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                  <Eye className="h-5 w-5 mr-2" />
                  Training Logs
                </h2>
                
                <div
                  ref={logsRef}
                  className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm h-96 overflow-y-auto whitespace-pre-wrap"
                >
                  {logs.length > 0 ? (
                    logs.map((log, idx) => (
                      <div key={idx} className="py-1">
                        {log}
                      </div>
                    ))
                  ) : (
                    <div className="text-gray-500">No logs available for this job</div>
                  )}
                </div>
              </div>
            </>
          ) : (
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <div className="text-center py-12">
                <Activity className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">No Job Selected</h3>
                <p className="text-gray-500">
                  Select a training job from the list to view logs and details
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default Monitoring 