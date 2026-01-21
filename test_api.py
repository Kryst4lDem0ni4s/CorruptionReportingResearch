import requests
from pathlib import Path

# Test the backend API
backend_url = "http://localhost:8080"
test_image = Path("evaluation/datasets/celebdf/real/real_00001.jpg")

if test_image.exists():
    with open(test_image, 'rb') as f:
        file_content = f.read()
    
    files = {'file': (test_image.name, file_content, 'image/jpeg')}
    data = {
        'evidence_type': 'image',
        'text_narrative': 'Test submission',
        'metadata': '{}'
    }
    
    try:
        print(f"Sending POST to {backend_url}/api/v1/submissions")
        response = requests.post(
            f"{backend_url}/api/v1/submissions",
            files=files,
            data=data,
            timeout=30
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"Test image not found: {test_image}")
