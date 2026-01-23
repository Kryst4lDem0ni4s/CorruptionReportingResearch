
import requests
import sys
import json
import time

BASE_URL = "http://localhost:8080"

def test_endpoint(method, path, description, expected_status=200):
    url = f"{BASE_URL}{path}"
    print(f"Testing {description} [{method} {path}]...")
    try:
        response = requests.request(method, url)
        print(f"Status: {response.status_code}")
        if response.status_code != expected_status:
            print(f" Failed: Expected {expected_status}, got {response.status_code}")
            print(f"Response: {response.text[:200]}")
            return False
        print("✅ Success")
        return True
    except Exception as e:
        print(f" Error: {e}")
        return False

def verify_backend():
    print("Verifying Backend API Logic...")
    
    # 1. Root endpoint
    if not test_endpoint("GET", "/", "Root Endpoint"):
        print("CRITICAL: Root endpoint unreachable. Server might not be running or port is wrong.")
        sys.exit(1)
        
    # Check server identity
    try:
        resp = requests.get(f"{BASE_URL}/")
        data = resp.json()
        if "Corruption Reporting System" not in data.get("name", ""):
             print(f" WRONG SERVER: {data}")
    except:
        pass

    # 2. Health check
    # Check if API_VERSION is v1 as expected
    test_endpoint("GET", "/api/v1/health", "Health Endpoint", expected_status=200)

    # 3. Submissions endpoint (Method Not Allowed is expected for GET, confirms path exists)
    # We expect 405 Method Not Allowed because it's a POST endpoint, but if we get 404 it means path is wrong
    if not test_endpoint("GET", "/api/v1/submissions", "Submissions Endpoint Existence", expected_status=405):
        print("Checking if path exists...")
        test_endpoint("POST", "/api/v1/submissions", "Submissions POST (Empty)", expected_status=422) # 422 Unprocessable Entity (Validation Error)

if __name__ == "__main__":
    try:
        verify_backend()
    except KeyboardInterrupt:
        pass
