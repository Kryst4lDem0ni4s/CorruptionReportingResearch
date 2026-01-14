#!/bin/bash

# =============================================================================
# Corruption Reporting System - Setup Script (Unix/Linux/macOS)
# Version: 1.0.0
# Description: Automated setup for development and production environments
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PYTHON_MIN_VERSION="3.8"
NODE_MIN_VERSION="14.0"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

print_header() {
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_command() {
    if command -v $1 &> /dev/null; then
        return 0
    else
        return 1
    fi
}

version_compare() {
    if [[ $1 == $2 ]]; then
        return 0
    fi
    local IFS=.
    local i ver1=($1) ver2=($2)
    for ((i=${#ver1[@]}; i<${#ver2[@]}; i++)); do
        ver1[i]=0
    done
    for ((i=0; i<${#ver1[@]}; i++)); do
        if [[ -z ${ver2[i]} ]]; then
            ver2[i]=0
        fi
        if ((10#${ver1[i]} > 10#${ver2[i]})); then
            return 0
        fi
        if ((10#${ver1[i]} < 10#${ver2[i]})); then
            return 1
        fi
    done
    return 0
}

# =============================================================================
# SYSTEM CHECK
# =============================================================================

check_prerequisites() {
    print_header "Checking System Prerequisites"
    
    local all_good=true
    
    # Check Python
    if check_command python3; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        if version_compare "$PYTHON_VERSION" "$PYTHON_MIN_VERSION"; then
            print_success "Python $PYTHON_VERSION installed (>= $PYTHON_MIN_VERSION required)"
        else
            print_error "Python $PYTHON_VERSION found, but >= $PYTHON_MIN_VERSION required"
            all_good=false
        fi
    else
        print_error "Python 3 not found. Please install Python >= $PYTHON_MIN_VERSION"
        all_good=false
    fi
    
    # Check pip
    if check_command pip3; then
        PIP_VERSION=$(pip3 --version 2>&1 | awk '{print $2}')
        print_success "pip $PIP_VERSION installed"
    else
        print_error "pip3 not found. Please install pip"
        all_good=false
    fi
    
    # Check Node.js
    if check_command node; then
        NODE_VERSION=$(node --version 2>&1 | sed 's/v//')
        if version_compare "$NODE_VERSION" "$NODE_MIN_VERSION"; then
            print_success "Node.js $NODE_VERSION installed (>= $NODE_MIN_VERSION required)"
        else
            print_error "Node.js $NODE_VERSION found, but >= $NODE_MIN_VERSION required"
            all_good=false
        fi
    else
        print_error "Node.js not found. Please install Node.js >= $NODE_MIN_VERSION"
        all_good=false
    fi
    
    # Check npm
    if check_command npm; then
        NPM_VERSION=$(npm --version 2>&1)
        print_success "npm $NPM_VERSION installed"
    else
        print_error "npm not found. Please install npm"
        all_good=false
    fi
    
    # Check git (optional but recommended)
    if check_command git; then
        GIT_VERSION=$(git --version 2>&1 | awk '{print $3}')
        print_success "git $GIT_VERSION installed"
    else
        print_warning "git not found (optional, but recommended for version control)"
    fi
    
    echo ""
    
    if [ "$all_good" = false ]; then
        print_error "Prerequisites check failed. Please install missing dependencies."
        exit 1
    fi
    
    print_success "All prerequisites satisfied"
}

# =============================================================================
# DIRECTORY SETUP
# =============================================================================

create_directories() {
    print_header "Creating Directory Structure"
    
    cd "$PROJECT_ROOT"
    
    # Backend data directories
    mkdir -p backend/data/submissions
    mkdir -p backend/data/evidence/2026/01
    mkdir -p backend/data/reports
    mkdir -p backend/data/cache
    
    # Evaluation directories
    mkdir -p evaluation/datasets/faceforensics
    mkdir -p evaluation/datasets/celebdf
    mkdir -p evaluation/datasets/synthetic_attacks
    mkdir -p evaluation/results/figures
    
    # Test fixtures
    mkdir -p tests/fixtures/real_images
    mkdir -p tests/fixtures/fake_images
    
    # Logs directory
    mkdir -p logs
    
    print_success "Directory structure created"
}

# =============================================================================
# PYTHON SETUP
# =============================================================================

setup_python_environment() {
    print_header "Setting Up Python Environment"
    
    cd "$PROJECT_ROOT"
    
    # Check if virtual environment exists
    if [ -d "venv" ]; then
        print_info "Virtual environment already exists"
        read -p "Do you want to recreate it? (y/N): " recreate
        if [[ $recreate =~ ^[Yy]$ ]]; then
            print_info "Removing existing virtual environment..."
            rm -rf venv
        else
            print_info "Using existing virtual environment"
            source venv/bin/activate
            return 0
        fi
    fi
    
    # Create virtual environment
    print_info "Creating virtual environment..."
    python3 -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    print_success "Virtual environment created and activated"
    
    # Upgrade pip
    print_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    print_success "pip upgraded"
}

install_python_dependencies() {
    print_header "Installing Python Dependencies"
    
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found"
        exit 1
    fi
    
    print_info "Installing production dependencies..."
    pip install -r requirements.txt
    
    print_success "Python dependencies installed"
    
    # Install development dependencies if available
    if [ -f "requirements-dev.txt" ]; then
        read -p "Install development dependencies? (y/N): " install_dev
        if [[ $install_dev =~ ^[Yy]$ ]]; then
            print_info "Installing development dependencies..."
            pip install -r requirements-dev.txt
            print_success "Development dependencies installed"
        fi
    fi
}

# =============================================================================
# NODE.JS SETUP
# =============================================================================

setup_nodejs_environment() {
    print_header "Setting Up Node.js Environment"
    
    cd "$PROJECT_ROOT"
    
    if [ ! -f "package.json" ]; then
        print_error "package.json not found"
        exit 1
    fi
    
    # Check if node_modules exists
    if [ -d "node_modules" ]; then
        print_info "node_modules already exists"
        read -p "Do you want to reinstall? (y/N): " reinstall
        if [[ $reinstall =~ ^[Yy]$ ]]; then
            print_info "Removing node_modules..."
            rm -rf node_modules package-lock.json
        else
            print_info "Using existing node_modules"
            return 0
        fi
    fi
    
    # Install Node.js dependencies
    print_info "Installing Node.js dependencies..."
    npm install
    
    print_success "Node.js dependencies installed"
}

# =============================================================================
# DATA INITIALIZATION
# =============================================================================

initialize_data_files() {
    print_header "Initializing Data Files"
    
    cd "$PROJECT_ROOT"
    
    # Initialize storage using Python script
    if [ -f "scripts/initialize_storage.py" ]; then
        source venv/bin/activate
        print_info "Running storage initialization..."
        python scripts/initialize_storage.py
        print_success "Storage initialized"
    else
        print_warning "initialize_storage.py not found, skipping"
    fi
    
    # Seed validators using Python script
    if [ -f "scripts/seed_validators.py" ]; then
        source venv/bin/activate
        print_info "Seeding validator pool..."
        python scripts/seed_validators.py
        print_success "Validators seeded"
    else
        print_warning "seed_validators.py not found, skipping"
    fi
}

# =============================================================================
# MODEL DOWNLOAD
# =============================================================================

download_ml_models() {
    print_header "ML Model Download"
    
    cd "$PROJECT_ROOT"
    
    print_info "This system uses the following pre-trained models:"
    echo "  - openai/clip-vit-base-patch32 (~350MB)"
    echo "  - sentence-transformers/all-MiniLM-L6-v2 (~80MB)"
    echo ""
    print_info "Models will be downloaded automatically on first use."
    echo ""
    
    read -p "Do you want to download models now? (y/N): " download_now
    
    if [[ $download_now =~ ^[Yy]$ ]]; then
        if [ -f "scripts/download_models.py" ]; then
            source venv/bin/activate
            print_info "Downloading models (this may take several minutes)..."
            python scripts/download_models.py
            print_success "Models downloaded"
        else
            print_warning "download_models.py not found, skipping"
        fi
    else
        print_info "Models will be downloaded on first use"
    fi
}

# =============================================================================
# CONFIGURATION
# =============================================================================

setup_configuration() {
    print_header "Setting Up Configuration"
    
    cd "$PROJECT_ROOT"
    
    # Create .env file if it doesn't exist
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            print_info "Creating .env from .env.example..."
            cp .env.example .env
            print_success ".env file created"
            print_warning "Please review and update .env with your settings"
        else
            print_warning ".env.example not found, skipping .env creation"
        fi
    else
        print_info ".env file already exists"
    fi
    
    # Check config files
    if [ -f "config.yaml" ]; then
        print_success "config.yaml found"
    else
        print_warning "config.yaml not found"
    fi
}

# =============================================================================
# VERIFICATION
# =============================================================================

run_verification() {
    print_header "Running Verification Tests"
    
    cd "$PROJECT_ROOT"
    source venv/bin/activate
    
    # Test Python imports
    print_info "Testing Python imports..."
    python -c "
import sys
try:
    import fastapi
    import torch
    import transformers
    import networkx
    print('✓ Core Python dependencies working')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)
" || {
        print_error "Python dependency verification failed"
        exit 1
    }
    
    print_success "Python dependencies verified"
    
    # Test backend health
    print_info "Testing backend health check..."
    python -c "
import sys
sys.path.insert(0, 'backend')
try:
    from api.health import get_health
    health = get_health()
    print(f'✓ Backend health check passed: {health[\"status\"]}')
except Exception as e:
    print(f'⚠ Backend health check warning: {e}')
" || print_warning "Backend health check had warnings (this is OK for first setup)"
    
    # Test Node.js
    print_info "Testing Node.js setup..."
    node -e "
try {
    require('express');
    require('axios');
    console.log('✓ Node.js dependencies working');
} catch (e) {
    console.error('✗ Node.js dependency error:', e.message);
    process.exit(1);
}
" || {
        print_error "Node.js dependency verification failed"
        exit 1
    }
    
    print_success "Node.js dependencies verified"
    
    print_success "All verification tests passed"
}

# =============================================================================
# SUMMARY
# =============================================================================

print_setup_summary() {
    print_header "Setup Complete!"
    
    echo ""
    print_success "The Corruption Reporting System has been set up successfully."
    echo ""
    
    print_info "Next steps:"
    echo ""
    echo "  1. Activate the virtual environment:"
    echo "     $ source venv/bin/activate"
    echo ""
    echo "  2. Start the backend server:"
    echo "     $ cd backend && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"
    echo ""
    echo "  3. In a new terminal, start the frontend server:"
    echo "     $ cd frontend && node server.js"
    echo ""
    echo "  4. Access the application:"
    echo "     Frontend: http://localhost:3000"
    echo "     Backend API: http://localhost:8000"
    echo "     API Docs: http://localhost:8000/docs"
    echo ""
    
    print_info "Optional commands:"
    echo ""
    echo "  - Run tests:"
    echo "    $ pytest tests/"
    echo ""
    echo "  - Run evaluation:"
    echo "    $ python -m evaluation.run_evaluation"
    echo ""
    echo "  - Download models (if not done):"
    echo "    $ python scripts/download_models.py"
    echo ""
    echo "  - Check system health:"
    echo "    $ python scripts/health_check.py"
    echo ""
    
    print_info "For more information, see README.md and docs/"
    echo ""
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    clear
    
    print_header "Corruption Reporting System - Setup"
    echo ""
    print_info "This script will set up the development environment."
    print_info "Project root: $PROJECT_ROOT"
    echo ""
    
    read -p "Press Enter to continue or Ctrl+C to cancel..."
    echo ""
    
    # Run setup steps
    check_prerequisites
    create_directories
    setup_python_environment
    install_python_dependencies
    setup_nodejs_environment
    initialize_data_files
    download_ml_models
    setup_configuration
    run_verification
    print_setup_summary
    
    print_success "Setup completed successfully!"
}

# Run main function
main "$@"
