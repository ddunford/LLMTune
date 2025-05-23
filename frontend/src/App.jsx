import React from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Training from './pages/Training'
import Datasets from './pages/Datasets'
import Monitoring from './pages/Monitoring'
import Inference from './pages/Inference'

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/training" element={<Training />} />
        <Route path="/datasets" element={<Datasets />} />
        <Route path="/monitoring" element={<Monitoring />} />
        <Route path="/inference" element={<Inference />} />
      </Routes>
    </Layout>
  )
}

export default App 