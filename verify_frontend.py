
import requests
import sys

BASE_URL = "http://localhost:3000"

def test_endpoint(method, path, description, expected_status=200):
    url = f"{BASE_URL}{path}"
    print(f"Testing {description} [{method} {path}]...")
    try:
        response = requests.request(method, url, timeout=5)
        print(f"Status: {response.status_code}")
        if response.status_code != expected_status:
            print(f" Failed: Expected {expected_status}, got {response.status_code}")
            return False
        print(" Success")
        return True
    except Exception as e:
        print(f" Error: {e}")
        return False

def verify_frontend():
    print("Verifying Frontend Server...")
    
    # 1. Frontend Root
    if not test_endpoint("GET", "/", "Frontend Root"):
        sys.exit(1)

    # 2. Proxy Health Check
    # This hits Frontend -> Proxy -> Backend -> Health Endpoint
    if not test_endpoint("GET", "/api/v1/health", "Backend Proxy Health"):
        sys.exit(1)
        
    print("\n Frontend Verification Complete!")

if __name__ == "__main__":
    verify_frontend()
