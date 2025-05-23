#!/bin/bash

# LLM Fine-Tuning UI Setup Script
# This script sets up the complete development environment

set -e  # Exit on any error

echo "🚀 LLM Fine-Tuning UI Setup"
echo "============================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed"
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [[ "$(printf '%s\n' "3.10" "$python_version" | sort -V | head -n1)" != "3.10" ]]; then
        print_error "Python 3.10+ is required. Found: $python_version"
        exit 1
    fi
    print_success "Python 3 found: $(python3 --version)"
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js is required but not installed"
        print_status "Please install Node.js 18+ from https://nodejs.org"
        exit 1
    fi
    
    node_version=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
    if [ "$node_version" -lt 18 ]; then
        print_error "Node.js 18+ is required. Found: $(node --version)"
        exit 1
    fi
    print_success "Node.js found: $(node --version)"
    
    # Check npm
    if ! command -v npm &> /dev/null; then
        print_error "npm is required but not installed"
        exit 1
    fi
    print_success "npm found: $(npm --version)"
    
    # Check git
    if ! command -v git &> /dev/null; then
        print_warning "Git not found. Version control features will be limited."
    else
        print_success "Git found: $(git --version)"
    fi
    
    # Check for NVIDIA GPUs (optional)
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while read -r line; do
            print_status "GPU: $line"
        done
    else
        print_warning "nvidia-smi not found. GPU training may not be available."
    fi
    
    echo ""
}

# Setup backend
setup_backend() {
    print_status "Setting up Python backend..."
    
    cd backend
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv venv
        print_success "Virtual environment created"
    else
        print_status "Virtual environment already exists"
    fi
    
    # Activate virtual environment and install dependencies
    print_status "Installing Python dependencies..."
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install core dependencies
    pip install fastapi==0.104.1 uvicorn[standard]==0.24.0
    pip install python-multipart==0.0.6 websockets==12.0
    pip install pydantic==2.5.0 pydantic-settings==2.1.0
    pip install aiofiles==23.2.1 python-dotenv==1.0.0
    pip install psutil==5.9.6 pyyaml==6.0.1 pandas==2.1.4
    
    # Try to install GPU monitoring libraries
    print_status "Installing GPU monitoring libraries..."
    if pip install GPUtil==1.4.0 pynvml==11.5.0; then
        print_success "GPU monitoring libraries installed"
    else
        print_warning "GPU monitoring libraries failed to install. Some features may be limited."
    fi
    
    # Try to install ML/training libraries
    print_status "Installing ML libraries..."
    if pip install torch transformers datasets; then
        print_success "ML libraries installed"
    else
        print_warning "ML libraries failed to install. Training features may be limited."
    fi
    
    # Install Axolotl for LLM fine-tuning
    print_status "Installing Axolotl for LLM training..."
    if pip install axolotl; then
        print_success "Axolotl installed successfully"
    else
        print_warning "Axolotl installation failed. You may need to install it manually with: pip install axolotl"
    fi
    
    # Install development dependencies
    pip install pytest==7.4.3 pytest-asyncio==0.21.1 httpx==0.25.2
    
    print_success "Backend dependencies installed"
    
    # Create environment file
    if [ ! -f ".env" ]; then
        cp .env.example .env
        print_success "Environment file created from template"
    fi
    
    cd ..
}

# Setup frontend
setup_frontend() {
    print_status "Setting up React frontend..."
    
    cd frontend
    
    # Install Node.js dependencies
    print_status "Installing Node.js dependencies..."
    npm install
    
    print_success "Frontend dependencies installed"
    cd ..
}

# Create necessary directories
create_directories() {
    print_status "Creating project directories..."
    
    mkdir -p backend/uploads
    mkdir -p backend/logs
    mkdir -p backend/checkpoints
    mkdir -p configs
    
    print_success "Project directories created"
}

# Setup git repository
setup_git() {
    if command -v git &> /dev/null; then
        if [ ! -d ".git" ]; then
            print_status "Initializing git repository..."
            git init
            print_success "Git repository initialized"
        else
            print_status "Git repository already exists"
        fi
    fi
}

# Create sample dataset
create_sample_data() {
    print_status "Creating sample dataset..."
    
    cat > backend/uploads/sample_dataset.jsonl << 'EOF'
{"instruction": "What is machine learning?", "input": "", "output": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task."}
{"instruction": "Explain neural networks", "input": "", "output": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information and learn patterns from data."}
{"instruction": "What is deep learning?", "input": "", "output": "Deep learning is a subset of machine learning that uses neural networks with multiple layers to learn complex patterns and representations from large amounts of data."}
{"instruction": "Define artificial intelligence", "input": "", "output": "Artificial intelligence (AI) is the simulation of human intelligence in machines, enabling them to think, learn, and make decisions like humans."}
{"instruction": "What is natural language processing?", "input": "", "output": "Natural language processing (NLP) is a branch of AI that focuses on enabling computers to understand, interpret, and generate human language."}
EOF
    
    print_success "Sample dataset created: backend/uploads/sample_dataset.jsonl"
}

# Final instructions
print_final_instructions() {
    echo ""
    echo "🎉 Setup Complete!"
    echo "=================="
    echo ""
    print_success "Your LLM Fine-Tuning UI is ready to use!"
    echo ""
    echo "📋 Next Steps:"
    echo "  1. Start the application: ./start.sh"
    echo "  2. Open your browser to: http://localhost:3000"
    echo "  3. Check API documentation: http://localhost:8000/docs"
    echo ""
    echo "📁 Project Structure:"
    echo "  • backend/     - FastAPI server and training logic"
    echo "  • frontend/    - React UI application"
    echo "  • configs/     - Training configurations"
    echo "  • .cursor/     - Development guidelines"
    echo ""
    echo "🔧 Development Commands:"
    echo "  • ./start.sh              - Start both backend and frontend"
    echo "  • cd backend && source venv/bin/activate && python main.py  - Backend only"
    echo "  • cd frontend && npm run dev                                 - Frontend only"
    echo ""
    echo "📚 Documentation:"
    echo "  • README.md               - Project overview and usage"
    echo "  • PRD.md                  - Product requirements"
    echo "  • .cursor/rules/*.mdc     - Development guidelines"
    echo ""
    print_warning "Note: For GPU training, ensure CUDA and NVIDIA drivers are properly installed"
    echo ""
}

# Main execution
main() {
    check_requirements
    setup_backend
    setup_frontend
    create_directories
    setup_git
    create_sample_data
    print_final_instructions
}

# Run main function
main 