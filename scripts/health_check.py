#!/usr/bin/env python3
"""
Corruption Reporting System - Health Check Script
Version: 1.0.0
Description: Check system health and readiness

This script checks:
- Python dependencies
- Data directory structure
- Critical data files
- Model availability
- Disk space
- Backend services
- File permissions

Usage:
    python scripts/health_check.py [--detailed] [--json]
"""

import os
import sys
import json
import argparse
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = PROJECT_ROOT / 'backend' / 'data'
BACKEND_DIR = PROJECT_ROOT / 'backend'
FRONTEND_DIR = PROJECT_ROOT / 'frontend'

CRITICAL_FILES = [
    DATA_DIR / 'chain.json',
    DATA_DIR / 'validators.json',
    DATA_DIR / 'index.json'
]

CRITICAL_DIRS = [
    DATA_DIR / 'submissions',
    DATA_DIR / 'evidence',
    DATA_DIR / 'reports'
]

MIN_DISK_SPACE_GB = 1.0  # Minimum 1GB free space

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def get_disk_usage(path: Path) -> Tuple[int, int, int]:
    """
    Get disk usage statistics
    
    Returns:
        Tuple of (total, used, free) in bytes
    """
    try:
        stat = shutil.disk_usage(path)
        return stat.total, stat.used, stat.free
    except:
        return 0, 0, 0

# =============================================================================
# HEALTH CHECKS
# =============================================================================

def check_python_version() -> Dict[str, Any]:
    """Check Python version"""
    result = {
        'name': 'Python Version',
        'status': 'unknown',
        'message': '',
        'details': {}
    }
    
    try:
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        result['details']['version'] = version_str
        result['details']['executable'] = sys.executable
        
        if version.major == 3 and version.minor >= 8:
            result['status'] = 'ok'
            result['message'] = f"Python {version_str}"
        else:
            result['status'] = 'warning'
            result['message'] = f"Python {version_str} (3.8+ recommended)"
    
    except Exception as e:
        result['status'] = 'error'
        result['message'] = f"Failed to check Python version: {e}"
    
    return result

def check_dependencies() -> Dict[str, Any]:
    """Check Python dependencies"""
    result = {
        'name': 'Python Dependencies',
        'status': 'unknown',
        'message': '',
        'details': {'installed': [], 'missing': []}
    }
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'torch',
        'transformers',
        'sentence_transformers',
        'networkx',
        'sklearn',
        'scipy',
        'PIL',
        'reportlab',
        'cryptography',
        'pydantic',
        'numpy'
    ]
    
    installed = []
    missing = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            elif package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            installed.append(package)
        except ImportError:
            missing.append(package)
    
    result['details']['installed'] = installed
    result['details']['missing'] = missing
    
    if not missing:
        result['status'] = 'ok'
        result['message'] = f"All {len(installed)} dependencies installed"
    else:
        result['status'] = 'error'
        result['message'] = f"{len(missing)} missing: {', '.join(missing)}"
    
    return result

def check_data_directory() -> Dict[str, Any]:
    """Check data directory structure"""
    result = {
        'name': 'Data Directory',
        'status': 'unknown',
        'message': '',
        'details': {'missing': [], 'present': []}
    }
    
    if not DATA_DIR.exists():
        result['status'] = 'error'
        result['message'] = 'Data directory does not exist'
        return result
    
    missing_dirs = []
    present_dirs = []
    
    for dir_path in CRITICAL_DIRS:
        if dir_path.exists():
            present_dirs.append(str(dir_path.relative_to(PROJECT_ROOT)))
        else:
            missing_dirs.append(str(dir_path.relative_to(PROJECT_ROOT)))
    
    result['details']['missing'] = missing_dirs
    result['details']['present'] = present_dirs
    
    if not missing_dirs:
        result['status'] = 'ok'
        result['message'] = 'All directories present'
    else:
        result['status'] = 'warning'
        result['message'] = f"{len(missing_dirs)} directories missing"
    
    return result

def check_data_files() -> Dict[str, Any]:
    """Check critical data files"""
    result = {
        'name': 'Data Files',
        'status': 'unknown',
        'message': '',
        'details': {'valid': [], 'invalid': [], 'missing': []}
    }
    
    valid = []
    invalid = []
    missing = []
    
    for file_path in CRITICAL_FILES:
        if not file_path.exists():
            missing.append(str(file_path.relative_to(PROJECT_ROOT)))
            continue
        
        # Try to parse JSON
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            valid.append(str(file_path.relative_to(PROJECT_ROOT)))
        except Exception as e:
            invalid.append(f"{file_path.relative_to(PROJECT_ROOT)}: {str(e)}")
    
    result['details']['valid'] = valid
    result['details']['invalid'] = invalid
    result['details']['missing'] = missing
    
    if not invalid and not missing:
        result['status'] = 'ok'
        result['message'] = 'All data files valid'
    elif invalid:
        result['status'] = 'error'
        result['message'] = f"{len(invalid)} files corrupted"
    else:
        result['status'] = 'warning'
        result['message'] = f"{len(missing)} files missing"
    
    return result

def check_models() -> Dict[str, Any]:
    """Check ML model availability"""
    result = {
        'name': 'ML Models',
        'status': 'unknown',
        'message': '',
        'details': {'available': [], 'unavailable': []}
    }
    
    models_to_check = [
        ('CLIP', 'openai/clip-vit-base-patch32'),
        ('Sentence Transformer', 'sentence-transformers/all-MiniLM-L6-v2')
    ]
    
    available = []
    unavailable = []
    
    for model_name, model_id in models_to_check:
        try:
            if 'clip' in model_id.lower():
                from transformers import CLIPModel
                model = CLIPModel.from_pretrained(model_id)
            elif 'sentence-transformers' in model_id.lower():
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(model_id)
            
            available.append(model_name)
        except Exception as e:
            unavailable.append(f"{model_name}: {str(e)[:50]}")
    
    result['details']['available'] = available
    result['details']['unavailable'] = unavailable
    
    if not unavailable:
        result['status'] = 'ok'
        result['message'] = f"All {len(available)} models available"
    else:
        result['status'] = 'warning'
        result['message'] = f"{len(unavailable)} models unavailable (will download on first use)"
    
    return result

def check_disk_space() -> Dict[str, Any]:
    """Check available disk space"""
    result = {
        'name': 'Disk Space',
        'status': 'unknown',
        'message': '',
        'details': {}
    }
    
    try:
        total, used, free = get_disk_usage(PROJECT_ROOT)
        
        result['details']['total'] = format_size(total)
        result['details']['used'] = format_size(used)
        result['details']['free'] = format_size(free)
        result['details']['usage_percent'] = round((used / total) * 100, 2) if total > 0 else 0
        
        free_gb = free / (1024 ** 3)
        
        if free_gb >= MIN_DISK_SPACE_GB * 2:
            result['status'] = 'ok'
            result['message'] = f"{format_size(free)} free"
        elif free_gb >= MIN_DISK_SPACE_GB:
            result['status'] = 'warning'
            result['message'] = f"Only {format_size(free)} free"
        else:
            result['status'] = 'error'
            result['message'] = f"Critically low: {format_size(free)} free"
    
    except Exception as e:
        result['status'] = 'error'
        result['message'] = f"Failed to check disk space: {e}"
    
    return result

def check_file_permissions() -> Dict[str, Any]:
    """Check file permissions"""
    result = {
        'name': 'File Permissions',
        'status': 'unknown',
        'message': '',
        'details': {'readable': True, 'writable': True}
    }
    
    try:
        # Check if data directory is readable and writable
        if DATA_DIR.exists():
            readable = os.access(DATA_DIR, os.R_OK)
            writable = os.access(DATA_DIR, os.W_OK)
            
            result['details']['readable'] = readable
            result['details']['writable'] = writable
            
            if readable and writable:
                result['status'] = 'ok'
                result['message'] = 'Read/write permissions OK'
            elif readable:
                result['status'] = 'error'
                result['message'] = 'No write permission'
            else:
                result['status'] = 'error'
                result['message'] = 'No read permission'
        else:
            result['status'] = 'warning'
            result['message'] = 'Data directory does not exist'
    
    except Exception as e:
        result['status'] = 'error'
        result['message'] = f"Permission check failed: {e}"
    
    return result

def check_backend_module() -> Dict[str, Any]:
    """Check backend module can be imported"""
    result = {
        'name': 'Backend Module',
        'status': 'unknown',
        'message': '',
        'details': {}
    }
    
    try:
        # Try to import critical backend modules
        sys.path.insert(0, str(BACKEND_DIR))
        
        from config import load_config
        from constants import SYSTEM_VERSION
        from exceptions import ValidationError
        
        result['details']['version'] = SYSTEM_VERSION
        result['status'] = 'ok'
        result['message'] = f"Backend modules OK (v{SYSTEM_VERSION})"
    
    except Exception as e:
        result['status'] = 'error'
        result['message'] = f"Cannot import backend: {e}"
    
    return result

def check_validators() -> Dict[str, Any]:
    """Check validator pool"""
    result = {
        'name': 'Validator Pool',
        'status': 'unknown',
        'message': '',
        'details': {}
    }
    
    validators_file = DATA_DIR / 'validators.json'
    
    if not validators_file.exists():
        result['status'] = 'warning'
        result['message'] = 'Validators file missing (run seed_validators.py)'
        return result
    
    try:
        with open(validators_file, 'r') as f:
            data = json.load(f)
        
        validators = data.get('validators', [])
        active = sum(1 for v in validators if v.get('active', False))
        
        result['details']['total'] = len(validators)
        result['details']['active'] = active
        
        if len(validators) >= 10:
            result['status'] = 'ok'
            result['message'] = f"{len(validators)} validators ({active} active)"
        elif len(validators) > 0:
            result['status'] = 'warning'
            result['message'] = f"Only {len(validators)} validators (10+ recommended)"
        else:
            result['status'] = 'error'
            result['message'] = 'No validators configured'
    
    except Exception as e:
        result['status'] = 'error'
        result['message'] = f"Cannot read validators: {e}"
    
    return result

# =============================================================================
# HEALTH CHECK EXECUTION
# =============================================================================

def run_health_checks(detailed: bool = False) -> Dict[str, Any]:
    """
    Run all health checks
    
    Args:
        detailed: Include detailed information
        
    Returns:
        Health check results
    """
    checks = [
        check_python_version(),
        check_dependencies(),
        check_data_directory(),
        check_data_files(),
        check_disk_space(),
        check_file_permissions(),
        check_backend_module(),
        check_validators()
    ]
    
    # Only check models in detailed mode (can be slow)
    if detailed:
        checks.append(check_models())
    
    # Calculate overall status
    statuses = [check['status'] for check in checks]
    
    if all(s == 'ok' for s in statuses):
        overall_status = 'healthy'
    elif any(s == 'error' for s in statuses):
        overall_status = 'unhealthy'
    else:
        overall_status = 'degraded'
    
    return {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'overall_status': overall_status,
        'checks': checks,
        'summary': {
            'ok': statuses.count('ok'),
            'warning': statuses.count('warning'),
            'error': statuses.count('error')
        }
    }

# =============================================================================
# OUTPUT FORMATTING
# =============================================================================

def print_health_report(results: Dict[str, Any], detailed: bool = False):
    """Print health check results in human-readable format"""
    
    # Header
    print("\n" + "=" * 70)
    print("Corruption Reporting System - Health Check")
    print("=" * 70)
    print(f"Timestamp: {results['timestamp']}")
    print(f"Overall Status: {results['overall_status'].upper()}")
    print("=" * 70)
    print("")
    
    # Individual checks
    status_symbols = {
        'ok': '',
        'warning': '⚠',
        'error': '',
        'unknown': '?'
    }
    
    for check in results['checks']:
        symbol = status_symbols.get(check['status'], '?')
        status = check['status'].upper()
        
        print(f"{symbol} {check['name']}")
        print(f"  Status: {status}")
        print(f"  {check['message']}")
        
        if detailed and check['details']:
            print(f"  Details:")
            for key, value in check['details'].items():
                if isinstance(value, list):
                    if value:
                        print(f"    {key}:")
                        for item in value:
                            print(f"      - {item}")
                else:
                    print(f"    {key}: {value}")
        
        print("")
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  OK: {results['summary']['ok']}")
    print(f"  Warning: {results['summary']['warning']}")
    print(f"  Error: {results['summary']['error']}")
    print("")
    
    # Recommendations
    if results['overall_status'] != 'healthy':
        print("Recommendations:")
        
        for check in results['checks']:
            if check['status'] == 'error':
                print(f"  • Fix: {check['name']} - {check['message']}")
            elif check['status'] == 'warning':
                print(f"  • Review: {check['name']} - {check['message']}")
        
        print("")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Check system health for Corruption Reporting System'
    )
    parser.add_argument(
        '--detailed',
        action='store_true',
        help='Include detailed checks (slower)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save results to file'
    )
    
    args = parser.parse_args()
    
    # Run health checks
    results = run_health_checks(detailed=args.detailed)
    
    # Output results
    if args.json:
        json_output = json.dumps(results, indent=2)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(json_output)
            logger.info(f"Results saved to {args.output}")
        else:
            print(json_output)
    else:
        print_health_report(results, detailed=args.detailed)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")
    
    # Exit code based on status
    if results['overall_status'] == 'healthy':
        return 0
    elif results['overall_status'] == 'degraded':
        return 1
    else:
        return 2

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\nHealth check interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
