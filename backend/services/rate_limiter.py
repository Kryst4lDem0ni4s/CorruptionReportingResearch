"""
Rate Limiter Service - IP-based rate limiting

Provides:
- IP-based request rate limiting
- Sliding window algorithm
- Configurable limits per endpoint
- Temporary IP blocking
- Rate limit status tracking
"""

import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

# Initialize logger
logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate Limiter - IP-based request rate limiting.
    
    Features:
    - Sliding window rate limiting
    - Per-IP tracking
    - Configurable limits (requests per time window)
    - Automatic cleanup of old records
    - Temporary IP blocking for abuse
    - Different limits for different endpoint types
    """
    
    def __init__(
        self,
        default_limit: int = 10,
        window_seconds: int = 3600,
        cleanup_interval: int = 300
    ):
        """
        Initialize rate limiter.
        
        Args:
            default_limit: Default number of requests per window
            window_seconds: Time window in seconds (default: 1 hour)
            cleanup_interval: Cleanup old records every N seconds
        """
        self.default_limit = default_limit
        self.window_seconds = window_seconds
        self.cleanup_interval = cleanup_interval
        
        # IP tracking: {ip_address: [timestamps]}
        self.requests: Dict[str, list] = defaultdict(list)
        
        # Blocked IPs: {ip_address: unblock_time}
        self.blocked_ips: Dict[str, float] = {}
        
        # Last cleanup time
        self.last_cleanup = time.time()
        
        # Custom limits for specific endpoint types
        self.endpoint_limits = {
            'submission': 5,      # 5 submissions per hour
            'status': 60,         # 60 status checks per hour
            'report': 10,         # 10 report requests per hour
            'health': 120,        # 120 health checks per hour
            'counter_evidence': 3  # 3 counter-evidence per hour
        }
        
        logger.info(
            f"RateLimiter initialized "
            f"(default={default_limit} req/{window_seconds}s)"
        )
    
    def check_rate_limit(
        self,
        ip_address: str,
        endpoint_type: str = 'default'
    ) -> Tuple[bool, Dict]:
        """
        Check if request is within rate limit.
        
        Args:
            ip_address: Client IP address
            endpoint_type: Type of endpoint (submission/status/report/etc)
            
        Returns:
            tuple: (is_allowed, rate_limit_info)
        """
        try:
            current_time = time.time()
            
            # Cleanup old records periodically
            if current_time - self.last_cleanup > self.cleanup_interval:
                self._cleanup_old_records()
            
            # Check if IP is blocked
            if ip_address in self.blocked_ips:
                unblock_time = self.blocked_ips[ip_address]
                
                if current_time < unblock_time:
                    remaining = int(unblock_time - current_time)
                    logger.warning(
                        f"Blocked IP attempted request: {ip_address} "
                        f"(blocked for {remaining}s)"
                    )
                    
                    return False, {
                        'allowed': False,
                        'reason': 'IP temporarily blocked',
                        'retry_after': remaining,
                        'blocked_until': datetime.fromtimestamp(unblock_time).isoformat()
                    }
                else:
                    # Unblock IP
                    del self.blocked_ips[ip_address]
                    logger.info(f"IP unblocked: {ip_address}")
            
            # Get limit for endpoint type
            limit = self.endpoint_limits.get(endpoint_type, self.default_limit)
            
            # Get request history for IP
            request_times = self.requests[ip_address]
            
            # Remove requests outside time window
            cutoff_time = current_time - self.window_seconds
            request_times = [t for t in request_times if t > cutoff_time]
            self.requests[ip_address] = request_times
            
            # Check if under limit
            current_count = len(request_times)
            
            if current_count >= limit:
                # Rate limit exceeded
                logger.warning(
                    f"Rate limit exceeded: {ip_address} "
                    f"({current_count}/{limit} for {endpoint_type})"
                )
                
                # Check if should block IP (excessive violations)
                if current_count >= limit * 2:
                    self._block_ip(ip_address, duration=3600)  # Block for 1 hour
                
                # Calculate when next request is allowed
                oldest_request = min(request_times) if request_times else current_time
                retry_after = int(self.window_seconds - (current_time - oldest_request))
                
                return False, {
                    'allowed': False,
                    'reason': 'Rate limit exceeded',
                    'limit': limit,
                    'current': current_count,
                    'window_seconds': self.window_seconds,
                    'retry_after': retry_after,
                    'endpoint_type': endpoint_type
                }
            
            # Add current request
            request_times.append(current_time)
            
            # Calculate remaining requests
            remaining = limit - (current_count + 1)
            reset_time = current_time + self.window_seconds
            
            logger.debug(
                f"Rate limit check passed: {ip_address} "
                f"({current_count + 1}/{limit} for {endpoint_type})"
            )
            
            return True, {
                'allowed': True,
                'limit': limit,
                'remaining': remaining,
                'reset': int(reset_time),
                'reset_datetime': datetime.fromtimestamp(reset_time).isoformat(),
                'window_seconds': self.window_seconds,
                'endpoint_type': endpoint_type
            }
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail open (allow request on error)
            return True, {
                'allowed': True,
                'error': str(e)
            }
    
    def _block_ip(self, ip_address: str, duration: int = 3600) -> None:
        """
        Temporarily block IP address.
        
        Args:
            ip_address: IP to block
            duration: Block duration in seconds
        """
        unblock_time = time.time() + duration
        self.blocked_ips[ip_address] = unblock_time
        
        logger.warning(
            f"IP blocked: {ip_address} "
            f"(duration={duration}s, until={datetime.fromtimestamp(unblock_time).isoformat()})"
        )
    
    def unblock_ip(self, ip_address: str) -> bool:
        """
        Manually unblock IP address.
        
        Args:
            ip_address: IP to unblock
            
        Returns:
            bool: True if IP was blocked, False if not
        """
        if ip_address in self.blocked_ips:
            del self.blocked_ips[ip_address]
            logger.info(f"IP manually unblocked: {ip_address}")
            return True
        return False
    
    def _cleanup_old_records(self) -> None:
        """Remove old request records to free memory."""
        try:
            current_time = time.time()
            cutoff_time = current_time - self.window_seconds
            
            # Cleanup request history
            cleaned_ips = []
            for ip, timestamps in list(self.requests.items()):
                # Remove old timestamps
                timestamps[:] = [t for t in timestamps if t > cutoff_time]
                
                # Remove IP if no recent requests
                if not timestamps:
                    cleaned_ips.append(ip)
            
            for ip in cleaned_ips:
                del self.requests[ip]
            
            # Cleanup expired blocks
            expired_blocks = [
                ip for ip, unblock_time in self.blocked_ips.items()
                if current_time >= unblock_time
            ]
            
            for ip in expired_blocks:
                del self.blocked_ips[ip]
            
            self.last_cleanup = current_time
            
            if cleaned_ips or expired_blocks:
                logger.debug(
                    f"Cleanup: removed {len(cleaned_ips)} IPs, "
                    f"unblocked {len(expired_blocks)} IPs"
                )
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    def get_rate_limit_status(self, ip_address: str) -> Dict:
        """
        Get current rate limit status for IP.
        
        Args:
            ip_address: IP address
            
        Returns:
            dict: Rate limit status
        """
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        
        # Get recent requests
        request_times = [t for t in self.requests.get(ip_address, []) if t > cutoff_time]
        
        # Check if blocked
        is_blocked = ip_address in self.blocked_ips
        unblock_time = self.blocked_ips.get(ip_address, 0)
        
        status = {
            'ip_address': ip_address,
            'is_blocked': is_blocked,
            'recent_requests': len(request_times),
            'limits': self.endpoint_limits,
            'window_seconds': self.window_seconds
        }
        
        if is_blocked:
            remaining = int(unblock_time - current_time)
            status['blocked_remaining'] = remaining
            status['unblock_time'] = datetime.fromtimestamp(unblock_time).isoformat()
        
        return status
    
    def reset_ip(self, ip_address: str) -> None:
        """
        Reset rate limit counters for IP.
        
        Args:
            ip_address: IP address
        """
        if ip_address in self.requests:
            del self.requests[ip_address]
        
        if ip_address in self.blocked_ips:
            del self.blocked_ips[ip_address]
        
        logger.info(f"Rate limit reset for IP: {ip_address}")
    
    def get_statistics(self) -> Dict:
        """
        Get rate limiter statistics.
        
        Returns:
            dict: Statistics
        """
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        
        # Count active IPs (with recent requests)
        active_ips = sum(
            1 for timestamps in self.requests.values()
            if any(t > cutoff_time for t in timestamps)
        )
        
        # Count total recent requests
        total_requests = sum(
            len([t for t in timestamps if t > cutoff_time])
            for timestamps in self.requests.values()
        )
        
        stats = {
            'active_ips': active_ips,
            'blocked_ips': len(self.blocked_ips),
            'total_tracked_ips': len(self.requests),
            'recent_requests': total_requests,
            'window_seconds': self.window_seconds,
            'default_limit': self.default_limit,
            'endpoint_limits': self.endpoint_limits,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return stats
    
    def update_limit(self, endpoint_type: str, new_limit: int) -> None:
        """
        Update rate limit for endpoint type.
        
        Args:
            endpoint_type: Endpoint type
            new_limit: New limit value
        """
        old_limit = self.endpoint_limits.get(endpoint_type, self.default_limit)
        self.endpoint_limits[endpoint_type] = new_limit
        
        logger.info(
            f"Rate limit updated for {endpoint_type}: "
            f"{old_limit} → {new_limit}"
        )
