#!/bin/bash

# LLM Fine-Tuning UI Startup Script

echo "🚀 Starting LLM Fine-Tuning UI..."

# Check if Python virtual environment exists
if [ ! -d "backend/venv" ]; then
    echo "📦 Creating Python virtual environment..."
    cd backend
    python3 -m venv venv
    cd ..
fi

# Check if Node.js dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "📦 Installing Node.js dependencies..."
    cd frontend
    npm install
    cd ..
fi

# Function to start backend
start_backend() {
    echo "🐍 Starting FastAPI backend..."
    cd backend
    source venv/bin/activate
    pip install -r requirements.txt
    python main.py &
    BACKEND_PID=$!
    cd ..
    echo "Backend started with PID: $BACKEND_PID"
}

# Function to start frontend
start_frontend() {
    echo "⚛️  Starting React frontend..."
    cd frontend
    npm run dev &
    FRONTEND_PID=$!
    cd ..
    echo "Frontend started with PID: $FRONTEND_PID"
}

# Start both services
start_backend
sleep 3
start_frontend

echo ""
echo "✅ LLM Fine-Tuning UI is starting up!"
echo ""
echo "📊 Dashboard: http://localhost:3000"
echo "🔧 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap 'echo "🛑 Stopping services..."; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit' INT
wait 