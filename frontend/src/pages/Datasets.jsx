import React, { useState, useEffect } from 'react'
import { Database, Upload, Eye, Trash2, FileText, Download, Calendar, Users } from 'lucide-react'
import * as api from '../services/api'

function Datasets() {
  const [datasets, setDatasets] = useState([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [selectedDataset, setSelectedDataset] = useState(null)
  const [previewData, setPreviewData] = useState(null)
  const [dragActive, setDragActive] = useState(false)

  useEffect(() => {
    loadDatasets()
  }, [])

  const loadDatasets = async () => {
    try {
      const response = await api.datasetAPI.list()
      console.log('API Response:', response)
      console.log('Response data:', response.data)
      console.log('Datasets from response:', response.data?.datasets || response.datasets)
      setDatasets(response.data?.datasets || response.datasets || [])
    } catch (error) {
      console.error('Failed to load datasets:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleFileUpload = async (files) => {
    if (!files || files.length === 0) return

    setUploading(true)
    
    for (const file of files) {
      try {
        const formData = new FormData()
        formData.append('file', file)
        
        await api.datasetAPI.upload(formData)
      } catch (error) {
        console.error(`Failed to upload ${file.name}:`, error)
      }
    }
    
    setUploading(false)
    await loadDatasets()
  }

  const handlePreview = async (dataset) => {
    try {
      const response = await api.datasetAPI.preview(dataset.id)
      setPreviewData(response)
      setSelectedDataset(dataset)
    } catch (error) {
      console.error('Failed to preview dataset:', error)
    }
  }

  const handleDelete = async (datasetId) => {
    if (!confirm('Are you sure you want to delete this dataset?')) return
    
    try {
      await api.datasetAPI.delete(datasetId)
      await loadDatasets()
    } catch (error) {
      console.error('Failed to delete dataset:', error)
    }
  }

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    const files = Array.from(e.dataTransfer.files)
    handleFileUpload(files)
  }

  const formatFileSize = (bytes) => {
    if (!bytes) return 'Unknown'
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(1024))
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i]
  }

  const formatDate = (dateString) => {
    if (!dateString) return 'Unknown'
    return new Date(dateString).toLocaleDateString()
  }

  const getSampleCount = (dataset) => {
    return dataset.sample_count || dataset.lines || 'Unknown'
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
          <h1 className="text-2xl font-bold text-gray-900">Datasets</h1>
          <p className="text-gray-600">Manage your training datasets</p>
        </div>
        
        <div className="text-sm text-gray-500">
          Total: {datasets.length} dataset{datasets.length !== 1 ? 's' : ''}
        </div>
      </div>

      {/* Upload Area */}
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
          dragActive 
            ? 'border-blue-500 bg-blue-50' 
            : uploading 
              ? 'border-orange-500 bg-orange-50'
              : 'border-gray-300 hover:border-gray-400'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        {uploading ? (
          <div className="space-y-2">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-orange-600 mx-auto"></div>
            <p className="text-orange-600 font-medium">Uploading files...</p>
          </div>
        ) : (
          <div className="space-y-2">
            <Upload className="h-8 w-8 text-gray-400 mx-auto" />
            <div>
              <p className="text-lg font-medium text-gray-900">
                Drop files here or{' '}
                <label className="text-blue-600 hover:text-blue-500 cursor-pointer">
                  browse to upload
                  <input
                    type="file"
                    multiple
                    accept=".jsonl,.csv,.txt"
                    className="hidden"
                    onChange={(e) => handleFileUpload(Array.from(e.target.files))}
                  />
                </label>
              </p>
              <p className="text-sm text-gray-500">
                Supports JSONL, CSV, and TXT formats
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Datasets Grid */}
      {datasets.length === 0 ? (
        <div className="text-center py-12">
          <Database className="h-12 w-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">No datasets found</h3>
          <p className="text-gray-500">Upload your first dataset to get started</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {datasets.map((dataset) => (
            <div key={dataset.id} className="bg-white rounded-lg border border-gray-200 hover:shadow-md transition-shadow">
              <div className="p-6">
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1 min-w-0">
                    <h3 className="text-lg font-medium text-gray-900 truncate" title={dataset.filename}>
                      {dataset.filename}
                    </h3>
                    <p className="text-sm text-gray-500">{dataset.format || 'Unknown format'}</p>
                  </div>
                  <FileText className="h-5 w-5 text-gray-400 flex-shrink-0 ml-2" />
                </div>

                <div className="space-y-2 mb-4">
                  <div className="flex items-center text-sm text-gray-600">
                    <Users className="h-4 w-4 mr-2" />
                    <span>{getSampleCount(dataset)} samples</span>
                  </div>
                  <div className="flex items-center text-sm text-gray-600">
                    <Download className="h-4 w-4 mr-2" />
                    <span>{formatFileSize(dataset.size)}</span>
                  </div>
                  <div className="flex items-center text-sm text-gray-600">
                    <Calendar className="h-4 w-4 mr-2" />
                    <span>{formatDate(dataset.uploaded_at)}</span>
                  </div>
                </div>

                <div className="flex space-x-2">
                  <button
                    onClick={() => handlePreview(dataset)}
                    className="flex-1 flex items-center justify-center px-3 py-2 text-sm bg-blue-50 text-blue-700 rounded-md hover:bg-blue-100 transition-colors"
                  >
                    <Eye className="h-4 w-4 mr-1" />
                    Preview
                  </button>
                  <button
                    onClick={() => handleDelete(dataset.id)}
                    className="flex items-center justify-center px-3 py-2 text-sm bg-red-50 text-red-700 rounded-md hover:bg-red-100 transition-colors"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Preview Modal */}
      {selectedDataset && previewData && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-4xl w-full max-h-[80vh] overflow-hidden">
            <div className="p-6 border-b border-gray-200">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-xl font-semibold text-gray-900">{selectedDataset.filename}</h2>
                  <p className="text-sm text-gray-500">Dataset Preview</p>
                </div>
                <button
                  onClick={() => {
                    setSelectedDataset(null)
                    setPreviewData(null)
                  }}
                  className="text-gray-400 hover:text-gray-500"
                >
                  <span className="sr-only">Close</span>
                  ×
                </button>
              </div>
            </div>
            
            <div className="p-6 overflow-auto max-h-[60vh]">
              {previewData.samples && previewData.samples.length > 0 ? (
                <div className="space-y-4">
                  <div className="text-sm text-gray-600 mb-4">
                    Showing {previewData.samples.length} sample{previewData.samples.length !== 1 ? 's' : ''} 
                    {previewData.total_samples && (
                      <span> of {previewData.total_samples} total</span>
                    )}
                  </div>
                  
                  {previewData.samples.map((sample, idx) => (
                    <div key={idx} className="border border-gray-200 rounded-lg p-4">
                      <div className="text-xs text-gray-500 mb-2">Sample {idx + 1}</div>
                      
                      {sample.instruction && (
                        <div className="mb-3">
                          <div className="text-sm font-medium text-gray-700 mb-1">Instruction:</div>
                          <div className="text-sm text-gray-900 bg-gray-50 p-2 rounded">
                            {sample.instruction}
                          </div>
                        </div>
                      )}
                      
                      {sample.input && (
                        <div className="mb-3">
                          <div className="text-sm font-medium text-gray-700 mb-1">Input:</div>
                          <div className="text-sm text-gray-900 bg-gray-50 p-2 rounded">
                            {sample.input}
                          </div>
                        </div>
                      )}
                      
                      {sample.output && (
                        <div className="mb-3">
                          <div className="text-sm font-medium text-gray-700 mb-1">Output:</div>
                          <div className="text-sm text-gray-900 bg-gray-50 p-2 rounded">
                            {sample.output}
                          </div>
                        </div>
                      )}

                      {/* Handle raw text or other formats */}
                      {!sample.instruction && !sample.input && !sample.output && (
                        <div className="text-sm text-gray-900 bg-gray-50 p-2 rounded">
                          {typeof sample === 'string' ? sample : JSON.stringify(sample, null, 2)}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8">
                  <p className="text-gray-500">No preview data available</p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default Datasets
