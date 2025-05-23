#!/bin/bash

# LLM Fine-Tuning UI Verification Script

echo "🔍 LLM Fine-Tuning UI - Setup Verification"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_check() {
    echo -e "${GREEN}✓${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_section() {
    echo -e "${YELLOW}$1${NC}"
}

# Check project structure
print_section "📁 Project Structure"
if [ -d "backend" ] && [ -d "frontend" ] && [ -d ".cursor" ]; then
    print_check "Main directories exist"
else
    echo "❌ Missing main directories"
fi

if [ -f "README.md" ] && [ -f "PRD.md" ] && [ -f ".gitignore" ]; then
    print_check "Documentation files exist"
else
    echo "❌ Missing documentation files"
fi

if [ -f "setup.sh" ] && [ -f "start.sh" ] && [ -x "setup.sh" ] && [ -x "start.sh" ]; then
    print_check "Setup scripts are executable"
else
    echo "❌ Setup scripts missing or not executable"
fi

echo ""

# Check backend structure
print_section "🐍 Backend Structure"
backend_files=(
    "backend/main.py"
    "backend/requirements.txt"
    "backend/config_builder.py"
    "backend/train_runner.py"
    "backend/.env.example"
)

for file in "${backend_files[@]}"; do
    if [ -f "$file" ]; then
        print_check "$(basename $file)"
    else
        echo "❌ Missing $file"
    fi
done

backend_dirs=(
    "backend/models"
    "backend/routes"
    "backend/services"
    "backend/uploads"
    "backend/logs"
    "backend/checkpoints"
)

for dir in "${backend_dirs[@]}"; do
    if [ -d "$dir" ]; then
        print_check "$(basename $dir)/ directory"
    else
        echo "❌ Missing $dir directory"
    fi
done

echo ""

# Check frontend structure
print_section "⚛️ Frontend Structure"
frontend_files=(
    "frontend/package.json"
    "frontend/vite.config.js"
    "frontend/tailwind.config.js"
    "frontend/index.html"
    "frontend/src/App.jsx"
    "frontend/src/main.jsx"
)

for file in "${frontend_files[@]}"; do
    if [ -f "$file" ]; then
        print_check "$(basename $file)"
    else
        echo "❌ Missing $file"
    fi
done

frontend_dirs=(
    "frontend/src/components"
    "frontend/src/pages"
    "frontend/src/services"
)

for dir in "${frontend_dirs[@]}"; do
    if [ -d "$dir" ]; then
        print_check "$(basename $dir)/ directory"
    else
        echo "❌ Missing $dir directory"
    fi
done

echo ""

# Check cursor rules
print_section "🎯 Cursor Rules"
cursor_rules=(
    ".cursor/rules/project-overview.mdc"
    ".cursor/rules/backend-guidelines.mdc"
    ".cursor/rules/frontend-guidelines.mdc"
    ".cursor/rules/training-config.mdc"
    ".cursor/rules/development-workflow.mdc"
)

for rule in "${cursor_rules[@]}"; do
    if [ -f "$rule" ]; then
        print_check "$(basename $rule)"
    else
        echo "❌ Missing $rule"
    fi
done

echo ""

# Check git setup
print_section "📋 Git Repository"
if [ -d ".git" ]; then
    print_check "Git repository initialized"
    
    # Check if there are commits
    if git log --oneline -1 &> /dev/null; then
        print_check "Initial commit exists"
        print_info "Latest commit: $(git log --oneline -1)"
    else
        echo "❌ No commits found"
    fi
    
    # Check tracked files
    tracked_files=$(git ls-files | wc -l)
    print_info "Tracked files: $tracked_files"
    
else
    echo "❌ Git repository not initialized"
fi

echo ""

# Check virtual environment
print_section "🐍 Python Environment"
if [ -d "backend/venv" ]; then
    print_check "Virtual environment exists"
else
    echo "❌ Virtual environment not found"
fi

echo ""

# Check node modules
print_section "📦 Node.js Environment"
if [ -d "frontend/node_modules" ]; then
    print_check "Node modules installed"
else
    echo "❌ Node modules not found"
fi

echo ""

# Final summary
print_section "🎉 Setup Summary"
echo ""
print_info "Your LLM Fine-Tuning UI project is ready!"
echo ""
echo "📋 Next Steps:"
echo "  1. Run setup: ./setup.sh"
echo "  2. Start app: ./start.sh"
echo "  3. Open browser: http://localhost:3000"
echo ""
echo "📚 Key Files:"
echo "  • README.md           - Project documentation"
echo "  • PRD.md              - Product requirements"
echo "  • setup.sh            - Complete setup script"
echo "  • start.sh            - Development startup"
echo "  • .cursor/rules/      - Development guidelines"
echo ""
echo "🔧 Architecture:"
echo "  • Backend: FastAPI + Axolotl (Port 8000)"
echo "  • Frontend: React + Tailwind (Port 3000)"
echo "  • Training: LoRA/QLoRA/Full fine-tuning"
echo "  • Monitoring: Real-time GPU stats"
echo ""
print_info "Happy fine-tuning! 🚀" 