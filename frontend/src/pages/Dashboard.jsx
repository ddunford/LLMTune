import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { 
  Brain, 
  Database, 
  Activity, 
  Plus, 
  Clock, 
  CheckCircle, 
  AlertCircle,
  TrendingUp,
  Cpu,
  HardDrive
} from 'lucide-react'
import { trainingAPI, datasetsAPI, monitoringAPI } from '../services/api'

function Dashboard() {
  const [stats, setStats] = useState({
    totalJobs: 0,
    runningJobs: 0,
    completedJobs: 0,
    totalDatasets: 0,
    systemStats: null
  })
  const [recentJobs, setRecentJobs] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadDashboardData()
  }, [])

  const loadDashboardData = async () => {
    try {
      const [jobsResponse, datasetsResponse, systemResponse] = await Promise.all([
        trainingAPI.getJobs(),
        datasetsAPI.getDatasets(),
        monitoringAPI.getSystemStats()
      ])

      const jobs = jobsResponse.data
      const datasets = datasetsResponse.data
      const systemStats = systemResponse.data

      setStats({
        totalJobs: jobs.length,
        runningJobs: jobs.filter(job => job.status === 'running').length,
        completedJobs: jobs.filter(job => job.status === 'completed').length,
        totalDatasets: datasets.total,
        systemStats
      })

      // Get recent jobs (last 5)
      setRecentJobs(jobs.slice(-5).reverse())
      
    } catch (error) {
      console.error('Error loading dashboard data:', error)
    } finally {
      setLoading(false)
    }
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'running': return 'text-blue-600 bg-blue-100'
      case 'completed': return 'text-green-600 bg-green-100'
      case 'failed': return 'text-red-600 bg-red-100'
      case 'cancelled': return 'text-gray-600 bg-gray-100'
      case 'paused': return 'text-yellow-600 bg-yellow-100'
      default: return 'text-gray-600 bg-gray-100'
    }
  }

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary-600"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="md:flex md:items-center md:justify-between">
        <div className="flex-1 min-w-0">
          <h2 className="text-2xl font-bold leading-7 text-gray-900 sm:text-3xl sm:truncate">
            Dashboard
          </h2>
          <p className="mt-1 text-sm text-gray-500">
            Welcome to your LLM fine-tuning control center
          </p>
        </div>
        <div className="mt-4 flex md:mt-0 md:ml-4">
          <Link
            to="/training"
            className="btn btn-primary flex items-center"
          >
            <Plus className="h-4 w-4 mr-2" />
            New Training Job
          </Link>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatsCard
          title="Total Jobs"
          value={stats.totalJobs}
          icon={Brain}
          color="blue"
        />
        <StatsCard
          title="Running Jobs"
          value={stats.runningJobs}
          icon={Activity}
          color="green"
        />
        <StatsCard
          title="Completed Jobs"
          value={stats.completedJobs}
          icon={CheckCircle}
          color="purple"
        />
        <StatsCard
          title="Datasets"
          value={stats.totalDatasets}
          icon={Database}
          color="indigo"
        />
      </div>

      {/* System Status */}
      {stats.systemStats && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <SystemStatsCard stats={stats.systemStats} />
          <GPUStatsCard gpus={stats.systemStats.gpus} />
        </div>
      )}

      {/* Recent Activity */}
      <div className="card p-6">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-medium text-gray-900">Recent Training Jobs</h3>
          <Link to="/training" className="text-sm text-primary-600 hover:text-primary-700">
            View all
          </Link>
        </div>
        
        {recentJobs.length === 0 ? (
          <div className="text-center py-8">
            <Brain className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-sm font-medium text-gray-900 mb-2">No training jobs yet</h3>
            <p className="text-sm text-gray-500 mb-4">
              Get started by creating your first training job
            </p>
            <Link to="/training" className="btn btn-primary">
              Create Training Job
            </Link>
          </div>
        ) : (
          <div className="space-y-4">
            {recentJobs.map((job) => (
              <div key={job.id} className="flex items-center justify-between p-4 border border-gray-200 rounded-lg">
                <div className="flex items-center space-x-4">
                  <div className="flex-shrink-0">
                    <Brain className="h-8 w-8 text-gray-400" />
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-gray-900">
                      Job {job.id}
                    </h4>
                    <p className="text-sm text-gray-500">
                      {job.config.base_model} • {job.config.method.toUpperCase()}
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-4">
                  <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(job.status)}`}>
                    {job.status}
                  </span>
                  <span className="text-sm text-gray-500">
                    {formatDate(job.created_at)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Quick Actions */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <QuickActionCard
          title="Upload Dataset"
          description="Upload and prepare your training data"
          icon={Database}
          to="/datasets"
          color="green"
        />
        <QuickActionCard
          title="Monitor System"
          description="View GPU stats and system performance"
          icon={Activity}
          to="/monitoring"
          color="purple"
        />
        <QuickActionCard
          title="Start Training"
          description="Configure and launch a new training job"
          icon={Brain}
          to="/training"
          color="blue"
        />
      </div>
    </div>
  )
}

function StatsCard({ title, value, icon: Icon, color }) {
  const colorClasses = {
    blue: 'text-blue-600 bg-blue-100',
    green: 'text-green-600 bg-green-100',
    purple: 'text-purple-600 bg-purple-100',
    indigo: 'text-indigo-600 bg-indigo-100',
  }

  return (
    <div className="card p-6">
      <div className="flex items-center">
        <div className="flex-shrink-0">
          <div className={`p-3 rounded-md ${colorClasses[color]}`}>
            <Icon className="h-6 w-6" />
          </div>
        </div>
        <div className="ml-5 w-0 flex-1">
          <dl>
            <dt className="text-sm font-medium text-gray-500 truncate">{title}</dt>
            <dd className="text-lg font-medium text-gray-900">{value}</dd>
          </dl>
        </div>
      </div>
    </div>
  )
}

function SystemStatsCard({ stats }) {
  return (
    <div className="card p-6">
      <h3 className="text-lg font-medium text-gray-900 mb-4">System Status</h3>
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Cpu className="h-5 w-5 text-gray-400 mr-2" />
            <span className="text-sm text-gray-700">CPU Usage</span>
          </div>
          <span className="text-sm font-medium text-gray-900">{stats.cpu_percent}%</span>
        </div>
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <HardDrive className="h-5 w-5 text-gray-400 mr-2" />
            <span className="text-sm text-gray-700">RAM Usage</span>
          </div>
          <span className="text-sm font-medium text-gray-900">
            {stats.ram_used.toFixed(1)}GB / {stats.ram_total.toFixed(1)}GB
          </span>
        </div>
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <HardDrive className="h-5 w-5 text-gray-400 mr-2" />
            <span className="text-sm text-gray-700">Disk Usage</span>
          </div>
          <span className="text-sm font-medium text-gray-900">{stats.disk_percent.toFixed(1)}%</span>
        </div>
      </div>
    </div>
  )
}

function GPUStatsCard({ gpus }) {
  return (
    <div className="card p-6">
      <h3 className="text-lg font-medium text-gray-900 mb-4">GPU Status</h3>
      <div className="space-y-4">
        {gpus.map((gpu, index) => (
          <div key={index} className="border border-gray-200 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-sm font-medium text-gray-900">GPU {gpu.gpu_id}</h4>
              <span className="text-sm text-gray-500">{gpu.temperature}°C</span>
            </div>
            <div className="text-xs text-gray-600 mb-3">{gpu.name}</div>
            <div className="space-y-2">
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">Memory</span>
                <span className="font-medium">
                  {gpu.memory_used.toFixed(1)}GB / {gpu.memory_total.toFixed(1)}GB
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full" 
                  style={{ width: `${gpu.memory_percent}%` }}
                />
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-gray-600">Utilization</span>
                <span className="font-medium">{gpu.utilization}%</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function QuickActionCard({ title, description, icon: Icon, to, color }) {
  const colorClasses = {
    blue: 'text-blue-600 bg-blue-100 hover:bg-blue-200',
    green: 'text-green-600 bg-green-100 hover:bg-green-200',
    purple: 'text-purple-600 bg-purple-100 hover:bg-purple-200',
  }

  return (
    <Link to={to} className="card p-6 hover:shadow-md transition-shadow">
      <div className="flex items-center">
        <div className={`p-3 rounded-md ${colorClasses[color]} transition-colors`}>
          <Icon className="h-6 w-6" />
        </div>
        <div className="ml-4">
          <h3 className="text-sm font-medium text-gray-900">{title}</h3>
          <p className="text-sm text-gray-500">{description}</p>
        </div>
      </div>
    </Link>
  )
}

export default Dashboard 