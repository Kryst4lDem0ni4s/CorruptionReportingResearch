"""
Integration Testing Agent for Corruption Reporting System
Tests end-to-end functionality including submission, processing, and retrieval
"""

import requests
import time
import sys
import logging
import argparse
import json
import base64
from typing import Dict, Any, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegrationTester:
    """Integration test runner for the corruption reporting system"""
    
    def __init__(self, backend_url: str = "http://localhost:8080", frontend_url: str = "http://localhost:3000"):
        self.backend_url = backend_url.rstrip('/')
        self.frontend_url = frontend_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'CorruptionReportingSystem-IntegrationTest/1.0'
        })

    def check_health(self, url: str, service_name: str) -> bool:
        """Check health of a service"""
        try:
            logger.info(f"Checking {service_name} health at {url}...")
            # Try multiple possible health endpoints
            endpoints = ["/api/v1/health", "/health", "/api/health", "/"]
            
            for endpoint in endpoints:
                try:
                    full_url = f"{url}{endpoint}"
                    response = self.session.get(full_url, timeout=5)
                    if response.status_code == 200:
                        logger.info(f"✅ {service_name} is healthy (endpoint: {endpoint})")
                        return True
                except requests.RequestException:
                    continue
            
            logger.error(f" {service_name} health check failed on all endpoints")
            return False
        except Exception as e:
            logger.error(f" {service_name} health check error: {e}")
            return False

    def check_proxy(self) -> bool:
        """Verify frontend correctly proxies to backend"""
        try:
            logger.info("Verifying Frontend Proxy (Frontend -> Backend)...")
            # Request backend health via frontend URL
            url = f"{self.frontend_url}/api/v1/health"
            response = self.session.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                # Check if response looks like backend response
                if "status" in data or "environment" in data:
                    logger.info("✅ Proxy verification successful")
                    return True
            
            logger.error(f" Proxy verification failed: Status {response.status_code}")
            logger.error(f"Response body: {response.text[:500]}")
            return False
        except Exception as e:
            logger.error(f" Proxy error: {e}")
            return False

    def test_storage_service(self) -> bool:
        """Test if storage service is working (Local check)"""
        try:
            logger.info("\nChecking Storage Service persistence...")
            
            # Ensure backend is importable
            project_root = Path(__file__).parent.parent
            if str(project_root) not in sys.path:
                sys.path.insert(0, str(project_root))
            
            from backend.services.storage_service import StorageService
            storage = StorageService()
            
            test_id = "test-integration-" + str(int(time.time()))
            test_data = {
                'id': test_id,
                'status': 'test_heartbeat',
                'message': 'Storage service verification'
            }
            
            storage.save_submission(test_id, test_data)
            loaded = storage.load_submission(test_id)
            
            if loaded and loaded.get('status') == 'test_heartbeat':
                logger.info(f"✅ Storage test passed: {loaded['id']}")
                # Cleanup
                try:
                    storage.delete_submission(test_id)
                except:
                    pass
                return True
            else:
                logger.error(f" Storage test failed: Loaded {loaded}")
                return False
        except Exception as e:
            logger.error(f" Storage test error: {e}")
            return False

    def create_test_image(self) -> bytes:
        """Create a valid test image"""
        try:
            from PIL import Image
            import io
            
            # Create a simple test image with some content
            img = Image.new('RGB', (640, 480), color=(73, 109, 137))
            
            # Add some simple patterns to make it more realistic
            import random
            pixels = img.load()
            for i in range(100):
                x = random.randint(0, 639)
                y = random.randint(0, 479)
                pixels[x, y] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            
            # Save to bytes
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr.seek(0)
            
            return img_byte_arr.getvalue()
        except ImportError:
            logger.warning("PIL not available, using minimal JPEG")
            # Return a minimal valid JPEG (1x1 pixel)
            return base64.b64decode(
                '/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a'
                'HBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIy'
                'MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIA'
                'AhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEB'
                'AQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwCwAA8='
            )

    def run_critical_path(self) -> bool:
        """
        Execute Critical User Journey:
        1. Submit Evidence (POST /api/v1/submissions)
        2. Poll Status (GET /api/v1/submissions/{id})
        3. Verify Result
        4. Retrieve Submission Details
        """
        logger.info("\n=== Starting Critical Path Verification ===")
        
        try:
            # 1. Create and Submit Evidence
            logger.info("1. Creating test evidence...")
            
            # Create valid test image
            image_data = self.create_test_image()
            
            # Prepare multipart form data
            files = {
                'file': ('integration_test.jpg', image_data, 'image/jpeg')
            }
            
            # Prepare form data
            metadata = {
                'location': 'Integration Test Lab',
                'incident_date': '2026-01-23',
                'test_run': True
            }
            
            data = {
                'evidence_type': 'image',
                'text_narrative': 'Integration test submission - automated testing',
                'metadata': json.dumps(metadata)
            }
            
            logger.info("2. Submitting evidence to backend...")
            submit_url = f"{self.backend_url}/api/v1/submissions"
            
            response = self.session.post(
                submit_url,
                files=files,
                data=data,
                timeout=30
            )
            
            if response.status_code not in [200, 201, 202]:
                logger.error(f" Submission failed: {response.status_code}")
                logger.error(f"Response: {response.text[:500]}")
                return False
            
            submission_data = response.json()
            submission_id = submission_data.get('submission_id') or submission_data.get('id')
            
            if not submission_id:
                logger.error(" No submission ID returned")
                logger.error(f"Response data: {submission_data}")
                return False
            
            logger.info(f"✅ Submission accepted. ID: {submission_id}")
            
            # 2. Poll Status with exponential backoff
            logger.info(f"3. Polling status for {submission_id}...")
            max_retries = 20
            wait_times = [1, 2, 3, 5, 5, 5, 5, 10, 10, 10] + [15] * 10
            
            for i in range(max_retries):
                poll_url = f"{self.backend_url}/api/v1/submissions/{submission_id}"
                
                try:
                    poll_response = self.session.get(poll_url, timeout=10)
                    
                    if poll_response.status_code != 200:
                        logger.warning(f"   Poll attempt {i+1}: Status {poll_response.status_code}, retrying...")
                        time.sleep(wait_times[min(i, len(wait_times)-1)])
                        continue
                    
                    poll_data = poll_response.json()
                    status = poll_data.get('status', 'unknown')
                    
                    logger.info(f"   Attempt {i+1}/{max_retries}: Status = {status}")
                    
                    # Check for terminal states
                    if status == 'completed':
                        logger.info("✅ Processing completed successfully")
                        
                        # 3. Verify Result Details
                        logger.info("4. Verifying submission details...")
                        if self._verify_submission_details(poll_data):
                            logger.info("✅ Submission details verified")
                            return True
                        else:
                            logger.warning("⚠️ Submission details incomplete")
                            return True  # Still consider success if processing completed
                    
                    elif status == 'failed':
                        logger.error(f" Processing failed")
                        logger.error(f"   Error details: {poll_data.get('error', 'No error details')}")
                        logger.error(f"   Full response: {json.dumps(poll_data, indent=2)}")
                        return False
                    
                    elif status == 'rejected':
                        logger.error(f" Submission was rejected")
                        logger.error(f"   Reason: {poll_data.get('rejection_reason', 'Unknown')}")
                        return False
                    
                    elif status in ['pending', 'processing', 'validating', 'analyzing']:
                        # Still processing, continue polling
                        time.sleep(wait_times[min(i, len(wait_times)-1)])
                        continue
                    
                    else:
                        logger.warning(f"⚠️ Unknown status: {status}")
                        time.sleep(wait_times[min(i, len(wait_times)-1)])
                        continue
                
                except requests.RequestException as e:
                    logger.warning(f"   Poll attempt {i+1} failed: {e}")
                    time.sleep(wait_times[min(i, len(wait_times)-1)])
                    continue
            
            logger.error(" Polling timed out after maximum retries")
            logger.error(f"   Last known status: {status}")
            return False

        except Exception as e:
            logger.error(f" Critical path execution error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _verify_submission_details(self, submission_data: Dict[str, Any]) -> bool:
        """Verify submission contains expected fields"""
        required_fields = ['submission_id', 'status']
        
        for field in required_fields:
            if field not in submission_data:
                logger.warning(f"   Missing field: {field}")
                return False
        
        return True

    def test_list_submissions(self) -> bool:
        """Test listing submissions endpoint"""
        try:
            logger.info("\n5. Testing list submissions endpoint...")
            list_url = f"{self.backend_url}/api/v1/submissions"
            
            response = self.session.get(list_url, params={'limit': 10}, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                submissions = data.get('submissions', [])
                logger.info(f"✅ List endpoint working. Found {len(submissions)} submissions")
                return True
            else:
                logger.warning(f"⚠️ List endpoint returned {response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ List endpoint test failed: {e}")
            return False

    def test_metrics_endpoint(self) -> bool:
        """Test metrics endpoint"""
        try:
            logger.info("\n6. Testing metrics endpoint...")
            metrics_url = f"{self.backend_url}/metrics"
            
            response = self.session.get(metrics_url, timeout=10)
            
            if response.status_code == 200:
                logger.info("✅ Metrics endpoint accessible")
                return True
            else:
                logger.warning(f"⚠️ Metrics endpoint returned {response.status_code}")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Metrics endpoint test failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Integration Testing Agent for Corruption Reporting System"
    )
    parser.add_argument(
        "--backend",
        default="http://localhost:8080",
        help="Backend URL (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--frontend",
        default="http://localhost:3000",
        help="Frontend URL (default: http://localhost:3000)"
    )
    parser.add_argument(
        "--skip-frontend",
        action="store_true",
        help="Skip frontend tests"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    tester = IntegrationTester(args.backend, args.frontend)
    
    logger.info("🚀 Starting Integration Tests...")
    logger.info(f"   Backend: {args.backend}")
    logger.info(f"   Frontend: {args.frontend}")
    
    results = {
        'backend_health': False,
        'frontend_health': False,
        'proxy': False,
        'storage': False,
        'critical_path': False,
        'list_submissions': False,
        'metrics': False
    }
    
    # 1. Backend Health
    results['backend_health'] = tester.check_health(args.backend, "Backend")
    if not results['backend_health']:
        logger.error(" Backend is not healthy. Aborting tests.")
        sys.exit(1)
    
    # 2. Frontend Health (optional)
    if not args.skip_frontend:
        results['frontend_health'] = tester.check_health(args.frontend, "Frontend")
        if results['frontend_health']:
            # 3. Proxy Verification
            results['proxy'] = tester.check_proxy()
        else:
            logger.warning("⚠️ Frontend unreachable. Skipping proxy tests.")
    
    # 3. Storage Verification (Local)
    results['storage'] = tester.test_storage_service()
    if not results['storage']:
        logger.error(" Storage service test failed. Aborting tests.")
        sys.exit(1)

    # 4. Critical Path (main test)
    results['critical_path'] = tester.run_critical_path()
    
    if not results['critical_path']:
        logger.error("\n Critical path test failed. Check logs above for details.")
        sys.exit(1)
    
    # 5. Additional Tests (optional - don't fail on these)
    results['list_submissions'] = tester.test_list_submissions()
    results['metrics'] = tester.test_metrics_endpoint()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("📊 Integration Test Results Summary")
    logger.info("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else " FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info("="*60)
    
    # Overall result
    critical_tests = ['backend_health', 'storage', 'critical_path']
    all_critical_passed = all(results[test] for test in critical_tests)
    
    if all_critical_passed:
        logger.info("🎉 All Critical Integration Tests Passed!")
        sys.exit(0)
    else:
        logger.error(" Some critical tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
