import React, { useState } from 'react'
import { NavLink } from 'react-router-dom'
import { 
  LayoutDashboard, 
  Brain, 
  Database, 
  Activity, 
  Menu, 
  X,
  Cpu,
  Zap,
  MessageSquare
} from 'lucide-react'

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
  { name: 'Training', href: '/training', icon: Brain },
  { name: 'Datasets', href: '/datasets', icon: Database },
  { name: 'Monitoring', href: '/monitoring', icon: Activity },
  { name: 'Inference', href: '/inference', icon: MessageSquare },
]

function Layout({ children }) {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  return (
    <div className="h-screen flex overflow-hidden bg-gray-100">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 flex z-40 md:hidden" 
          role="dialog" 
          aria-modal="true"
        >
          <div 
            className="fixed inset-0 bg-gray-600 bg-opacity-75"
            onClick={() => setSidebarOpen(false)}
          />
          <div className="relative flex-1 flex flex-col max-w-xs w-full bg-white">
            <div className="absolute top-0 right-0 -mr-12 pt-2">
              <button
                type="button"
                className="ml-1 flex items-center justify-center h-10 w-10 rounded-full focus:outline-none focus:ring-2 focus:ring-inset focus:ring-white"
                onClick={() => setSidebarOpen(false)}
              >
                <X className="h-6 w-6 text-white" />
              </button>
            </div>
            <Sidebar />
          </div>
        </div>
      )}

      {/* Static sidebar for desktop */}
      <div className="hidden md:flex md:flex-shrink-0">
        <div className="flex flex-col w-64">
          <Sidebar />
        </div>
      </div>

      {/* Main content */}
      <div className="flex flex-col w-0 flex-1 overflow-hidden">
        {/* Top bar */}
        <div className="relative z-10 flex-shrink-0 flex h-16 bg-white shadow">
          <button
            type="button"
            className="px-4 border-r border-gray-200 text-gray-500 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-indigo-500 md:hidden"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu className="h-6 w-6" />
          </button>
          
          <div className="flex-1 px-4 flex justify-between items-center">
            <div className="flex-1">
              <h1 className="text-2xl font-semibold text-gray-900">
                LLM Fine-Tuning UI
              </h1>
            </div>
            
            <div className="ml-4 flex items-center md:ml-6">
              <div className="flex items-center space-x-2 text-sm text-gray-500">
                <Cpu className="h-4 w-4" />
                <span>Dual RTX 3060</span>
                <Zap className="h-4 w-4 text-green-500" />
                <span className="text-green-600">Ready</span>
              </div>
            </div>
          </div>
        </div>

        {/* Page content */}
        <main className="flex-1 relative overflow-y-auto focus:outline-none">
          <div className="py-6">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 md:px-8">
              {children}
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}

function Sidebar() {
  return (
    <div className="flex flex-col h-full pt-5 pb-4 overflow-y-auto bg-gray-800">
      <div className="flex items-center flex-shrink-0 px-4">
        <Brain className="h-8 w-8 text-indigo-400" />
        <span className="ml-2 text-xl font-semibold text-white">LLM Trainer</span>
      </div>
      
      <nav className="mt-8 flex-1 px-2 space-y-1">
        {navigation.map((item) => (
          <NavLink
            key={item.name}
            to={item.href}
            className={({ isActive }) =>
              `group flex items-center px-2 py-2 text-sm font-medium rounded-md transition-colors ${
                isActive
                  ? 'bg-gray-900 text-white'
                  : 'text-gray-300 hover:bg-gray-700 hover:text-white'
              }`
            }
          >
            <item.icon className="mr-3 h-5 w-5 flex-shrink-0" />
            {item.name}
          </NavLink>
        ))}
      </nav>
      
      <div className="flex-shrink-0 p-4 border-t border-gray-700">
        <div className="text-xs text-gray-400">
          <div>Version 1.0.0</div>
          <div className="mt-1">Built with ❤️ for local GPU training</div>
        </div>
      </div>
    </div>
  )
}

export default Layout 