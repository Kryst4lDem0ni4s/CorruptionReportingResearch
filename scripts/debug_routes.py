
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.main import app

print("\n=== REGISTERED ROUTES ===")
for route in app.routes:
    if hasattr(route, "path"):
        methods = ", ".join(route.methods)
        print(f"{methods} {route.path}")
print("=========================\n")
