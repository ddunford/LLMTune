import React, { useState, useEffect } from 'react'
import { Save, Eye, EyeOff, CheckCircle, AlertCircle, ExternalLink, Settings as SettingsIcon, Key, Shield, Database } from 'lucide-react'

function Settings() {
  const [activeTab, setActiveTab] = useState('huggingface')
  const [hfAuth, setHfAuth] = useState(null)
  const [newToken, setNewToken] = useState('')
  const [showToken, setShowToken] = useState(false)
  const [isValidating, setIsValidating] = useState(false)
  const [validationResult, setValidationResult] = useState(null)
  const [isSaving, setIsSaving] = useState(false)
  const [connectionStatus, setConnectionStatus] = useState(null)
  const [settings, setSettings] = useState([])

  useEffect(() => {
    loadSettings()
    loadHfAuth()
    testConnection()
  }, [])

  const loadSettings = async () => {
    try {
      const response = await fetch('/api/settings/settings')
      const data = await response.json()
      setSettings(data)
    } catch (error) {
      console.error('Failed to load settings:', error)
    }
  }

  const loadHfAuth = async () => {
    try {
      const response = await fetch('/api/settings/huggingface/auth')
      const data = await response.json()
      if (data.length > 0) {
        setHfAuth(data[0])
      }
    } catch (error) {
      console.error('Failed to load HF auth:', error)
    }
  }

  const testConnection = async () => {
    try {
      const response = await fetch('/api/settings/huggingface/test-connection')
      const data = await response.json()
      setConnectionStatus(data)
    } catch (error) {
      console.error('Failed to test connection:', error)
      setConnectionStatus({ connected: false, message: 'Connection test failed' })
    }
  }

  const validateToken = async (token) => {
    if (!token.trim()) {
      setValidationResult(null)
      return
    }

    setIsValidating(true)
    setValidationResult(null)

    try {
      const response = await fetch('/api/settings/huggingface/validate-token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ token: token.trim() })
      })
      
      const result = await response.json()
      setValidationResult(result)
    } catch (error) {
      setValidationResult({
        valid: false,
        error: 'Failed to validate token',
        username: null
      })
    } finally {
      setIsValidating(false)
    }
  }

  const saveHfToken = async () => {
    if (!newToken.trim() || !validationResult?.valid) {
      return
    }

    setIsSaving(true)

    try {
      const response = await fetch('/api/settings/huggingface/auth', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          token: newToken.trim(),
          username: validationResult.username
        })
      })

      if (response.ok) {
        setNewToken('')
        setValidationResult(null)
        await loadHfAuth()
        await testConnection()
        // Show success message
        setValidationResult({ valid: true, message: 'Token saved successfully!' })
        setTimeout(() => setValidationResult(null), 3000)
      } else {
        const error = await response.json()
        setValidationResult({ valid: false, error: error.detail || 'Failed to save token' })
      }
    } catch (error) {
      setValidationResult({ valid: false, error: 'Failed to save token' })
    } finally {
      setIsSaving(false)
    }
  }

  const deleteHfAuth = async () => {
    if (!hfAuth || !confirm('Are you sure you want to delete the Hugging Face authentication?')) {
      return
    }

    try {
      const response = await fetch(`/api/settings/huggingface/auth/${hfAuth.id}`, {
        method: 'DELETE'
      })

      if (response.ok) {
        setHfAuth(null)
        await testConnection()
      }
    } catch (error) {
      console.error('Failed to delete HF auth:', error)
    }
  }

  const handleTokenChange = (value) => {
    setNewToken(value)
    setValidationResult(null)
  }

  const handleValidateClick = () => {
    validateToken(newToken)
  }

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="flex items-center space-x-3 mb-8">
        <SettingsIcon className="h-8 w-8 text-blue-600" />
        <h1 className="text-3xl font-bold text-gray-900">Settings</h1>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 mb-8">
        <nav className="-mb-px flex space-x-8">
          <button
            onClick={() => setActiveTab('huggingface')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'huggingface'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <div className="flex items-center space-x-2">
              <Key className="h-4 w-4" />
              <span>Hugging Face</span>
            </div>
          </button>
          <button
            onClick={() => setActiveTab('general')}
            className={`py-2 px-1 border-b-2 font-medium text-sm ${
              activeTab === 'general'
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <div className="flex items-center space-x-2">
              <Database className="h-4 w-4" />
              <span>General</span>
            </div>
          </button>
        </nav>
      </div>

      {/* Hugging Face Tab */}
      {activeTab === 'huggingface' && (
        <div className="space-y-8">
          {/* Connection Status */}
          <div className="bg-gray-50 rounded-lg p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Connection Status</h3>
            {connectionStatus && (
              <div className={`flex items-center space-x-3 ${
                connectionStatus.connected ? 'text-green-600' : 'text-red-600'
              }`}>
                {connectionStatus.connected ? (
                  <CheckCircle className="h-5 w-5" />
                ) : (
                  <AlertCircle className="h-5 w-5" />
                )}
                <div>
                  <div className="font-medium">
                    {connectionStatus.connected ? 'Connected' : 'Not Connected'}
                  </div>
                  <div className="text-sm">
                    {connectionStatus.message}
                    {connectionStatus.username && ` (${connectionStatus.username})`}
                  </div>
                  {connectionStatus.can_access_gated && (
                    <div className="text-sm font-medium text-green-600 mt-1">
                      ✅ Can access gated models
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Current Token */}
          {hfAuth && (
            <div className="bg-white border border-gray-200 rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-medium text-gray-900">Current Authentication</h3>
                <button
                  onClick={deleteHfAuth}
                  className="text-red-600 hover:text-red-700 text-sm font-medium"
                >
                  Remove
                </button>
              </div>
              <div className="space-y-3">
                <div>
                  <span className="text-sm text-gray-500">Token: </span>
                  <span className="font-mono text-sm">{hfAuth.token_preview}</span>
                </div>
                {hfAuth.username && (
                  <div>
                    <span className="text-sm text-gray-500">Username: </span>
                    <span className="text-sm">{hfAuth.username}</span>
                  </div>
                )}
                <div>
                  <span className="text-sm text-gray-500">Status: </span>
                  <span className={`text-sm font-medium ${
                    hfAuth.is_active ? 'text-green-600' : 'text-gray-500'
                  }`}>
                    {hfAuth.is_active ? 'Active' : 'Inactive'}
                  </span>
                </div>
                {hfAuth.last_validated && (
                  <div>
                    <span className="text-sm text-gray-500">Last Validated: </span>
                    <span className="text-sm">{new Date(hfAuth.last_validated).toLocaleString()}</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Add/Update Token */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              {hfAuth ? 'Update' : 'Add'} Hugging Face Token
            </h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Access Token
                </label>
                <div className="flex space-x-2">
                  <div className="flex-1 relative">
                    <input
                      type={showToken ? 'text' : 'password'}
                      value={newToken}
                      onChange={(e) => handleTokenChange(e.target.value)}
                      placeholder="hf_..."
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    />
                    <button
                      type="button"
                      onClick={() => setShowToken(!showToken)}
                      className="absolute right-3 top-2.5"
                    >
                      {showToken ? (
                        <EyeOff className="h-4 w-4 text-gray-400" />
                      ) : (
                        <Eye className="h-4 w-4 text-gray-400" />
                      )}
                    </button>
                  </div>
                  <button
                    onClick={handleValidateClick}
                    disabled={!newToken.trim() || isValidating}
                    className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {isValidating ? 'Validating...' : 'Validate'}
                  </button>
                </div>
              </div>

              {/* Validation Result */}
              {validationResult && (
                <div className={`flex items-center space-x-2 text-sm ${
                  validationResult.valid ? 'text-green-600' : 'text-red-600'
                }`}>
                  {validationResult.valid ? (
                    <CheckCircle className="h-4 w-4" />
                  ) : (
                    <AlertCircle className="h-4 w-4" />
                  )}
                  <span>
                    {validationResult.message || validationResult.error}
                    {validationResult.username && ` (${validationResult.username})`}
                  </span>
                </div>
              )}

              {/* Save Button */}
              {validationResult?.valid && newToken.trim() && (
                <button
                  onClick={saveHfToken}
                  disabled={isSaving}
                  className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  <Save className="h-4 w-4" />
                  <span>{isSaving ? 'Saving...' : 'Save Token'}</span>
                </button>
              )}

              {/* Help Text */}
              <div className="text-sm text-gray-500 space-y-2">
                <p>
                  To access gated models, you need a Hugging Face access token.
                </p>
                <p>
                  <a
                    href="https://huggingface.co/settings/tokens"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-blue-600 hover:text-blue-700 inline-flex items-center"
                  >
                    Get your token here <ExternalLink className="h-3 w-3 ml-1" />
                  </a>
                </p>
                <p className="text-xs">
                  Your token is stored securely and only used for model access.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* General Tab */}
      {activeTab === 'general' && (
        <div className="space-y-8">
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Application Settings</h3>
            <div className="space-y-4">
              {settings.map((setting) => (
                <div key={setting.id} className="flex items-center justify-between py-3 border-b border-gray-100 last:border-b-0">
                  <div>
                    <div className="font-medium text-gray-900">{setting.key}</div>
                    <div className="text-sm text-gray-500">{setting.description}</div>
                  </div>
                  <div className="flex items-center space-x-2">
                    {setting.is_sensitive ? (
                      <span className="text-sm text-gray-400">***</span>
                    ) : (
                      <span className="text-sm font-mono">{setting.value}</span>
                    )}
                    {setting.is_sensitive && (
                      <Shield className="h-4 w-4 text-amber-500" />
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default Settings 