#!/bin/bash

# LLM Fine-Tuning UI Docker Setup Script
set -e

echo "🐳 LLM Fine-Tuning UI Docker Setup"
echo "=================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check for NVIDIA Docker runtime
if ! docker run --rm --gpus all nvidia/cuda:12.1-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "⚠️  NVIDIA Docker runtime not detected. GPU support may not work."
    echo "   Please install nvidia-docker2 for GPU support."
    read -p "Continue without GPU support? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p backend/uploads backend/logs backend/checkpoints backend/configs
mkdir -p data/models data/cache

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from example..."
    cp .env.example .env
    echo "✅ Please edit .env file with your configuration"
fi

# Function to build and run development environment
setup_development() {
    echo "🔧 Setting up development environment..."
    
    # Build images
    echo "🏗️  Building Docker images..."
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml --profile development build
    
    # Start services
    echo "🚀 Starting development services..."
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml --profile development up -d
    
    echo "✅ Development environment is ready!"
    echo "   Frontend: http://localhost:55155"
    echo "   Backend: http://localhost:8001"
    echo "   Backend Health: http://localhost:8001/health"
}

# Function to build and run production environment
setup_production() {
    echo "🏭 Setting up production environment..."
    
    # Build images
    echo "🏗️  Building Docker images..."
    docker-compose --profile production build
    
    # Start services
    echo "🚀 Starting production services..."
    docker-compose --profile production up -d
    
    echo "✅ Production environment is ready!"
    echo "   Application: http://localhost"
    echo "   Backend API: http://localhost/api/"
}

# Function to stop all services
stop_services() {
    echo "🛑 Stopping all services..."
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml down
    docker-compose down
}

# Function to clean up
cleanup() {
    echo "🧹 Cleaning up Docker resources..."
    stop_services
    docker system prune -f
    docker volume prune -f
}

# Function to show logs
show_logs() {
    docker-compose logs -f
}

# Function to show status
show_status() {
    echo "📊 Service Status:"
    docker-compose ps
    echo
    echo "💾 Volume Usage:"
    docker volume ls | grep llm-trainer || echo "No volumes found"
    echo
    echo "🏗️  Image Usage:"
    docker images | grep -E "(llm-trainer|nvidia)" || echo "No images found"
}

# Parse command line arguments
case "${1:-}" in
    "dev"|"development")
        setup_development
        ;;
    "prod"|"production")
        setup_production
        ;;
    "stop")
        stop_services
        ;;
    "cleanup")
        cleanup
        ;;
    "logs")
        show_logs
        ;;
    "status")
        show_status
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  dev         Setup development environment"
        echo "  prod        Setup production environment"
        echo "  stop        Stop all services"
        echo "  cleanup     Clean up Docker resources"
        echo "  logs        Show service logs"
        echo "  status      Show service status"
        echo "  help        Show this help message"
        ;;
    *)
        echo "🤔 What would you like to do?"
        echo "1) Setup development environment"
        echo "2) Setup production environment"
        echo "3) Stop services"
        echo "4) Show logs"
        echo "5) Show status"
        echo "6) Cleanup"
        read -p "Choose an option (1-6): " choice
        
        case $choice in
            1) setup_development ;;
            2) setup_production ;;
            3) stop_services ;;
            4) show_logs ;;
            5) show_status ;;
            6) cleanup ;;
            *) echo "❌ Invalid option" ;;
        esac
        ;;
esac 