import React, { useState, useEffect } from 'react'
import { Database, Upload, FileText, Eye, Trash2, CheckCircle, AlertCircle } from 'lucide-react'
import { datasetAPI } from '../services/api'

function DatasetSelector({ selectedDataset, onDatasetSelect, disabled = false }) {
  const [datasets, setDatasets] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [uploadProgress, setUploadProgress] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [previewDataset, setPreviewDataset] = useState(null)

  useEffect(() => {
    loadDatasets()
  }, [])

  const loadDatasets = async () => {
    try {
      setLoading(true)
      const response = await datasetAPI.list()
      console.log('DatasetSelector API Response:', response)
      setDatasets(response.data?.datasets || response.datasets || [])
      setError(null)
    } catch (error) {
      console.error('Failed to load datasets:', error)
      setError('Failed to load datasets')
    } finally {
      setLoading(false)
    }
  }

  const handleFileUpload = async (event) => {
    const file = event.target.files[0]
    if (!file) return

    // Validate file type
    const allowedTypes = ['application/json', 'text/plain', 'text/csv', '.jsonl']
    const isJsonl = file.name.endsWith('.jsonl')
    
    if (!allowedTypes.includes(file.type) && !isJsonl) {
      setError('Please upload a .jsonl, .csv, or .txt file')
      return
    }

    setUploading(true)
    setUploadProgress(0)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', file)
      
      const response = await datasetAPI.upload(formData, (progressEvent) => {
        const progress = (progressEvent.loaded / progressEvent.total) * 100
        setUploadProgress(Math.round(progress))
      })

      // Reload datasets to include the new one
      await loadDatasets()
      
      // Auto-select the newly uploaded dataset
      if (response.data?.dataset) {
        onDatasetSelect(response.data.dataset)
      }
    } catch (error) {
      console.error('Upload failed:', error)
      setError(error.response?.data?.detail || 'Failed to upload dataset')
    } finally {
      setUploading(false)
      setUploadProgress(null)
      // Reset file input
      event.target.value = ''
    }
  }

  const handleDatasetSelect = (dataset) => {
    onDatasetSelect(dataset)
  }

  const handlePreview = async (dataset) => {
    try {
      const response = await datasetAPI.get(dataset.id)
      setPreviewDataset(response.data)
    } catch (error) {
      console.error('Failed to load dataset details:', error)
    }
  }

  const handleDelete = async (dataset) => {
    if (!confirm(`Are you sure you want to delete "${dataset.filename}"?`)) {
      return
    }

    try {
      await datasetAPI.delete(dataset.id)
      await loadDatasets()
      
      // If the deleted dataset was selected, clear selection
      if (selectedDataset?.id === dataset.id) {
        onDatasetSelect(null)
      }
    } catch (error) {
      console.error('Failed to delete dataset:', error)
      setError('Failed to delete dataset')
    }
  }

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'ready': return 'text-green-600 bg-green-100'
      case 'processing': return 'text-yellow-600 bg-yellow-100'
      case 'error': return 'text-red-600 bg-red-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900 mb-4">Select Training Dataset</h3>

        {/* Upload Section */}
        <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 mb-6">
          <div className="text-center">
            <Upload className="h-8 w-8 text-gray-400 mx-auto mb-4" />
            <div className="text-lg font-medium text-gray-900 mb-2">Upload New Dataset</div>
            <div className="text-sm text-gray-500 mb-4">
              Supports .jsonl, .csv, and .txt files
            </div>
            
            {uploading ? (
              <div className="space-y-2">
                <div className="text-sm text-blue-600">Uploading... {uploadProgress}%</div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
              </div>
            ) : (
              <label className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 cursor-pointer disabled:opacity-50">
                <input
                  type="file"
                  className="hidden"
                  accept=".jsonl,.csv,.txt"
                  onChange={handleFileUpload}
                  disabled={disabled || uploading}
                />
                Choose File
              </label>
            )}
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-md p-4 mb-4">
            <div className="flex items-center">
              <AlertCircle className="h-5 w-5 text-red-400 mr-2" />
              <span className="text-red-800">{error}</span>
            </div>
          </div>
        )}

        {/* Datasets List */}
        {loading ? (
          <div className="text-center py-8">
            <div className="text-gray-500">Loading datasets...</div>
          </div>
        ) : datasets.length === 0 ? (
          <div className="text-center py-8">
            <Database className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <div className="text-lg font-medium text-gray-900 mb-2">No datasets found</div>
            <div className="text-gray-500">Upload your first dataset to get started</div>
          </div>
        ) : (
          <div className="space-y-3">
            <h4 className="text-sm font-medium text-gray-700">Available Datasets</h4>
            {datasets.map((dataset) => (
              <div
                key={dataset.id}
                className={`border rounded-lg p-4 transition-all ${
                  selectedDataset?.id === dataset.id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <FileText className="h-5 w-5 text-gray-400" />
                      <div className="font-medium text-gray-900">{dataset.filename}</div>
                      <span className={`px-2 py-1 text-xs rounded-full ${getStatusColor(dataset.status)}`}>
                        {dataset.status}
                      </span>
                    </div>
                    
                    <div className="text-sm text-gray-500 space-y-1">
                      <div>Format: {dataset.format.toUpperCase()}</div>
                      <div>Size: {formatFileSize(dataset.size_bytes)}</div>
                      {dataset.num_rows && <div>Rows: {dataset.num_rows.toLocaleString()}</div>}
                      {dataset.num_tokens && <div>Est. Tokens: {dataset.num_tokens.toLocaleString()}</div>}
                    </div>
                  </div>
                  
                  <div className="flex items-center space-x-2 ml-4">
                    <button
                      onClick={() => handlePreview(dataset)}
                      className="p-2 text-gray-400 hover:text-gray-600"
                      title="Preview dataset"
                    >
                      <Eye className="h-4 w-4" />
                    </button>
                    <button
                      onClick={() => handleDelete(dataset)}
                      className="p-2 text-gray-400 hover:text-red-600"
                      title="Delete dataset"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </div>
                </div>

                {dataset.status === 'ready' && (
                  <div className="mt-3">
                    <button
                      onClick={() => handleDatasetSelect(dataset)}
                      disabled={disabled}
                      className={`px-4 py-2 text-sm rounded-md transition-colors ${
                        selectedDataset?.id === dataset.id
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                      } disabled:opacity-50`}
                    >
                      {selectedDataset?.id === dataset.id ? 'Selected' : 'Select Dataset'}
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Selected Dataset Display */}
      {selectedDataset && (
        <div className="border border-green-200 bg-green-50 rounded-lg p-4">
          <div className="flex items-center space-x-2 text-green-800 mb-2">
            <CheckCircle className="h-5 w-5" />
            <span className="font-medium">Selected Dataset</span>
          </div>
          <div className="text-sm text-gray-600">
            <div className="font-medium">{selectedDataset.filename}</div>
            <div>{selectedDataset.num_rows?.toLocaleString()} rows • {formatFileSize(selectedDataset.size_bytes)}</div>
          </div>
        </div>
      )}

      {/* Preview Modal */}
      {previewDataset && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-4xl w-full max-h-96 overflow-hidden">
            <div className="p-4 border-b">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-medium">Dataset Preview: {previewDataset.filename}</h3>
                <button
                  onClick={() => setPreviewDataset(null)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  ×
                </button>
              </div>
            </div>
            <div className="p-4 overflow-auto max-h-80">
              {previewDataset.sample_data && previewDataset.sample_data.length > 0 ? (
                <div className="space-y-2">
                  {previewDataset.sample_data.map((row, index) => (
                    <div key={index} className="bg-gray-50 p-3 rounded text-sm">
                      <pre className="whitespace-pre-wrap">{JSON.stringify(row, null, 2)}</pre>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-gray-500">No preview data available</div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default DatasetSelector 