@echo off
REM =============================================================================
REM Corruption Reporting System - Setup Script (Windows)
REM Version: 1.0.0
REM Description: Automated setup for development and production environments
REM =============================================================================

setlocal enabledelayedexpansion

REM Configuration
set PYTHON_MIN_VERSION=3.8
set NODE_MIN_VERSION=14.0
set PROJECT_ROOT=%~dp0..
cd /d "%PROJECT_ROOT%"

REM Colors (Windows 10+)
set "GREEN=[92m"
set "RED=[91m"
set "YELLOW=[93m"
set "BLUE=[94m"
set "NC=[0m"

REM =============================================================================
REM UTILITY FUNCTIONS
REM =============================================================================

:print_header
echo.
echo %BLUE%============================================%NC%
echo %BLUE%%~1%NC%
echo %BLUE%============================================%NC%
echo.
goto :eof

:print_success
echo %GREEN%[CHECKMARK] %~1%NC%
goto :eof

:print_error
echo %RED%[X] %~1%NC%
goto :eof

:print_warning
echo %YELLOW%[!] %~1%NC%
goto :eof

:print_info
echo %BLUE%[i] %~1%NC%
goto :eof

REM =============================================================================
REM SYSTEM CHECK
REM =============================================================================

:check_prerequisites
call :print_header "Checking System Prerequisites"

set ALL_GOOD=1

REM Check Python
where python >nul 2>nul
if %errorlevel% equ 0 (
    for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
    call :print_success "Python !PYTHON_VERSION! installed"
) else (
    call :print_error "Python not found. Please install Python >= %PYTHON_MIN_VERSION%"
    set ALL_GOOD=0
)

REM Check pip
where pip >nul 2>nul
if %errorlevel% equ 0 (
    for /f "tokens=2" %%i in ('pip --version 2^>^&1') do set PIP_VERSION=%%i
    call :print_success "pip !PIP_VERSION! installed"
) else (
    call :print_error "pip not found. Please install pip"
    set ALL_GOOD=0
)

REM Check Node.js
where node >nul 2>nul
if %errorlevel% equ 0 (
    for /f "tokens=1 delims=v" %%i in ('node --version 2^>^&1') do set NODE_VERSION=%%i
    call :print_success "Node.js !NODE_VERSION! installed"
) else (
    call :print_error "Node.js not found. Please install Node.js >= %NODE_MIN_VERSION%"
    set ALL_GOOD=0
)

REM Check npm
where npm >nul 2>nul
if %errorlevel% equ 0 (
    for /f "tokens=1" %%i in ('npm --version 2^>^&1') do set NPM_VERSION=%%i
    call :print_success "npm !NPM_VERSION! installed"
) else (
    call :print_error "npm not found. Please install npm"
    set ALL_GOOD=0
)

REM Check git (optional)
where git >nul 2>nul
if %errorlevel% equ 0 (
    for /f "tokens=3" %%i in ('git --version 2^>^&1') do set GIT_VERSION=%%i
    call :print_success "git !GIT_VERSION! installed"
) else (
    call :print_warning "git not found (optional, but recommended)"
)

echo.

if !ALL_GOOD! equ 0 (
    call :print_error "Prerequisites check failed. Please install missing dependencies."
    pause
    exit /b 1
)

call :print_success "All prerequisites satisfied"
goto :eof

REM =============================================================================
REM DIRECTORY SETUP
REM =============================================================================

:create_directories
call :print_header "Creating Directory Structure"

REM Backend data directories
if not exist "backend\data\submissions" mkdir "backend\data\submissions"
if not exist "backend\data\evidence\2026\01" mkdir "backend\data\evidence\2026\01"
if not exist "backend\data\reports" mkdir "backend\data\reports"
if not exist "backend\data\cache" mkdir "backend\data\cache"

REM Evaluation directories
if not exist "evaluation\datasets\faceforensics" mkdir "evaluation\datasets\faceforensics"
if not exist "evaluation\datasets\celebdf" mkdir "evaluation\datasets\celebdf"
if not exist "evaluation\datasets\synthetic_attacks" mkdir "evaluation\datasets\synthetic_attacks"
if not exist "evaluation\results\figures" mkdir "evaluation\results\figures"

REM Test fixtures
if not exist "tests\fixtures\real_images" mkdir "tests\fixtures\real_images"
if not exist "tests\fixtures\fake_images" mkdir "tests\fixtures\fake_images"

REM Logs directory
if not exist "logs" mkdir "logs"

call :print_success "Directory structure created"
goto :eof

REM =============================================================================
REM PYTHON SETUP
REM =============================================================================

:setup_python_environment
call :print_header "Setting Up Python Environment"

REM Check if virtual environment exists
if exist "venv" (
    call :print_info "Virtual environment already exists"
    set /p RECREATE="Do you want to recreate it? (y/N): "
    if /i "!RECREATE!"=="y" (
        call :print_info "Removing existing virtual environment..."
        rmdir /s /q venv
    ) else (
        call :print_info "Using existing virtual environment"
        call venv\Scripts\activate.bat
        goto :eof
    )
)

REM Create virtual environment
call :print_info "Creating virtual environment..."
python -m venv venv

REM Activate virtual environment
call venv\Scripts\activate.bat

call :print_success "Virtual environment created and activated"

REM Upgrade pip
call :print_info "Upgrading pip..."
python -m pip install --upgrade pip setuptools wheel

call :print_success "pip upgraded"
goto :eof

:install_python_dependencies
call :print_header "Installing Python Dependencies"

call venv\Scripts\activate.bat

if not exist "requirements.txt" (
    call :print_error "requirements.txt not found"
    pause
    exit /b 1
)

call :print_info "Installing production dependencies..."
pip install -r requirements.txt

call :print_success "Python dependencies installed"

REM Install development dependencies if available
if exist "requirements-dev.txt" (
    set /p INSTALL_DEV="Install development dependencies? (y/N): "
    if /i "!INSTALL_DEV!"=="y" (
        call :print_info "Installing development dependencies..."
        pip install -r requirements-dev.txt
        call :print_success "Development dependencies installed"
    )
)
goto :eof

REM =============================================================================
REM NODE.JS SETUP
REM =============================================================================

:setup_nodejs_environment
call :print_header "Setting Up Node.js Environment"

if not exist "package.json" (
    call :print_error "package.json not found"
    pause
    exit /b 1
)

REM Check if node_modules exists
if exist "node_modules" (
    call :print_info "node_modules already exists"
    set /p REINSTALL="Do you want to reinstall? (y/N): "
    if /i "!REINSTALL!"=="y" (
        call :print_info "Removing node_modules..."
        rmdir /s /q node_modules
        if exist "package-lock.json" del package-lock.json
    ) else (
        call :print_info "Using existing node_modules"
        goto :eof
    )
)

REM Install Node.js dependencies
call :print_info "Installing Node.js dependencies..."
call npm install

call :print_success "Node.js dependencies installed"
goto :eof

REM =============================================================================
REM DATA INITIALIZATION
REM =============================================================================

:initialize_data_files
call :print_header "Initializing Data Files"

call venv\Scripts\activate.bat

REM Initialize storage using Python script
if exist "scripts\initialize_storage.py" (
    call :print_info "Running storage initialization..."
    python scripts\initialize_storage.py
    call :print_success "Storage initialized"
) else (
    call :print_warning "initialize_storage.py not found, skipping"
)

REM Seed validators using Python script
if exist "scripts\seed_validators.py" (
    call :print_info "Seeding validator pool..."
    python scripts\seed_validators.py
    call :print_success "Validators seeded"
) else (
    call :print_warning "seed_validators.py not found, skipping"
)
goto :eof

REM =============================================================================
REM MODEL DOWNLOAD
REM =============================================================================

:download_ml_models
call :print_header "ML Model Download"

call :print_info "This system uses the following pre-trained models:"
echo   - openai/clip-vit-base-patch32 (~350MB)
echo   - sentence-transformers/all-MiniLM-L6-v2 (~80MB)
echo.
call :print_info "Models will be downloaded automatically on first use."
echo.

set /p DOWNLOAD_NOW="Do you want to download models now? (y/N): "

if /i "!DOWNLOAD_NOW!"=="y" (
    if exist "scripts\download_models.py" (
        call venv\Scripts\activate.bat
        call :print_info "Downloading models (this may take several minutes)..."
        python scripts\download_models.py
        call :print_success "Models downloaded"
    ) else (
        call :print_warning "download_models.py not found, skipping"
    )
) else (
    call :print_info "Models will be downloaded on first use"
)
goto :eof

REM =============================================================================
REM CONFIGURATION
REM =============================================================================

:setup_configuration
call :print_header "Setting Up Configuration"

REM Create .env file if it doesn't exist
if not exist ".env" (
    if exist ".env.example" (
        call :print_info "Creating .env from .env.example..."
        copy .env.example .env >nul
        call :print_success ".env file created"
        call :print_warning "Please review and update .env with your settings"
    ) else (
        call :print_warning ".env.example not found, skipping .env creation"
    )
) else (
    call :print_info ".env file already exists"
)

REM Check config files
if exist "config.yaml" (
    call :print_success "config.yaml found"
) else (
    call :print_warning "config.yaml not found"
)
goto :eof

REM =============================================================================
REM VERIFICATION
REM =============================================================================

:run_verification
call :print_header "Running Verification Tests"

call venv\Scripts\activate.bat

REM Test Python imports
call :print_info "Testing Python imports..."
python -c "import sys; import fastapi; import torch; import transformers; import networkx; print('Core Python dependencies working')" >nul 2>&1
if %errorlevel% equ 0 (
    call :print_success "Python dependencies verified"
) else (
    call :print_error "Python dependency verification failed"
    pause
    exit /b 1
)

REM Test Node.js
call :print_info "Testing Node.js setup..."
node -e "try { require('express'); require('axios'); console.log('Node.js dependencies working'); } catch(e) { process.exit(1); }" >nul 2>&1
if %errorlevel% equ 0 (
    call :print_success "Node.js dependencies verified"
) else (
    call :print_error "Node.js dependency verification failed"
    pause
    exit /b 1
)

call :print_success "All verification tests passed"
goto :eof

REM =============================================================================
REM SUMMARY
REM =============================================================================

:print_setup_summary
call :print_header "Setup Complete!"

echo.
call :print_success "The Corruption Reporting System has been set up successfully."
echo.

call :print_info "Next steps:"
echo.
echo   1. Activate the virtual environment:
echo      ^> venv\Scripts\activate.bat
echo.
echo   2. Start the backend server:
echo      ^> cd backend
echo      ^> python -m uvicorn main:app --reload --host 0.0.0.0 --port 8080
echo.
echo   3. In a new terminal, start the frontend server:
echo      ^> cd frontend
echo      ^> node server.js
echo.
echo   4. Access the application:
echo      Frontend: http://localhost:3000
echo      Backend API: http://localhost:8080
echo      API Docs: http://localhost:8080/docs
echo.

call :print_info "Optional commands:"
echo.
echo   - Run tests:
echo     ^> pytest tests\
echo.
echo   - Run evaluation:
echo     ^> python -m evaluation.run_evaluation
echo.
echo   - Download models (if not done):
echo     ^> python scripts\download_models.py
echo.
echo   - Check system health:
echo     ^> python scripts\health_check.py
echo.

call :print_info "For more information, see README.md and docs\"
echo.
goto :eof

REM =============================================================================
REM MAIN EXECUTION
REM =============================================================================

:main
cls

call :print_header "Corruption Reporting System - Setup"
echo.
call :print_info "This script will set up the development environment."
call :print_info "Project root: %PROJECT_ROOT%"
echo.

pause

echo.

REM Run setup steps
call :check_prerequisites
call :create_directories
call :setup_python_environment
call :install_python_dependencies
call :setup_nodejs_environment
call :initialize_data_files
call :download_ml_models
call :setup_configuration
call :run_verification
call :print_setup_summary

call :print_success "Setup completed successfully!"
echo.
pause
goto :eof

REM Run main function
call :main
