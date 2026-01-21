
import uvicorn
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

try:
    from backend.main import app
except Exception as e:
    print(f"Import Error: {e}")
    sys.exit(1)

if __name__ == "__main__":
    print("Starting uvicorn programmatically on 8000...")
    try:
        uvicorn.run(app, host="127.0.0.1", port=8000, log_level="debug")
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)
