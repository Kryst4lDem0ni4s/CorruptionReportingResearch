"""
Time Utils - Timestamp handling and time-related operations

Provides:
- Timestamp generation and parsing
- Time formatting
- Duration calculations
- Timezone handling
- Time window detection
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

# Initialize logger
logger = logging.getLogger(__name__)


class TimeUtils:
    """
    Time utilities for timestamp management.
    
    Features:
    - Unix timestamp generation
    - ISO 8601 formatting
    - Duration calculations
    - Time window detection
    - Human-readable formatting
    """
    
    @staticmethod
    def get_current_timestamp() -> float:
        """
        Get current Unix timestamp.
        
        Returns:
            float: Unix timestamp (seconds since epoch)
        """
        return time.time()
    
    @staticmethod
    def get_current_datetime() -> datetime:
        """
        Get current datetime (UTC).
        
        Returns:
            datetime: Current datetime in UTC
        """
        return datetime.now(timezone.utc)
    
    @staticmethod
    def timestamp_to_datetime(timestamp: float) -> datetime:
        """
        Convert Unix timestamp to datetime.
        
        Args:
            timestamp: Unix timestamp
            
        Returns:
            datetime: Datetime object in UTC
        """
        return datetime.fromtimestamp(timestamp, tz=timezone.utc)
    
    @staticmethod
    def datetime_to_timestamp(dt: datetime) -> float:
        """
        Convert datetime to Unix timestamp.
        
        Args:
            dt: Datetime object
            
        Returns:
            float: Unix timestamp
        """
        return dt.timestamp()
    
    @staticmethod
    def format_iso8601(timestamp: Optional[float] = None) -> str:
        """
        Format timestamp as ISO 8601 string.
        
        Args:
            timestamp: Unix timestamp (None = current time)
            
        Returns:
            str: ISO 8601 formatted string
        """
        if timestamp is None:
            timestamp = TimeUtils.get_current_timestamp()
        
        dt = TimeUtils.timestamp_to_datetime(timestamp)
        
        # ISO 8601 format with UTC timezone
        iso_string = dt.isoformat()
        
        return iso_string
    
    @staticmethod
    def parse_iso8601(iso_string: str) -> float:
        """
        Parse ISO 8601 string to timestamp.
        
        Args:
            iso_string: ISO 8601 formatted string
            
        Returns:
            float: Unix timestamp
        """
        try:
            dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
            return TimeUtils.datetime_to_timestamp(dt)
        except Exception as e:
            logger.error(f"Failed to parse ISO 8601 string: {e}")
            return 0.0
    
    @staticmethod
    def format_human_readable(
        timestamp: Optional[float] = None,
        include_time: bool = True
    ) -> str:
        """
        Format timestamp as human-readable string.
        
        Args:
            timestamp: Unix timestamp (None = current time)
            include_time: Include time component
            
        Returns:
            str: Human-readable string
        """
        if timestamp is None:
            timestamp = TimeUtils.get_current_timestamp()
        
        dt = TimeUtils.timestamp_to_datetime(timestamp)
        
        if include_time:
            # Format: "2026-01-13 14:30:45 UTC"
            return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
        else:
            # Format: "2026-01-13"
            return dt.strftime("%Y-%m-%d")
    
    @staticmethod
    def format_relative(timestamp: float) -> str:
        """
        Format timestamp as relative time (e.g., "2 hours ago").
        
        Args:
            timestamp: Unix timestamp
            
        Returns:
            str: Relative time string
        """
        now = TimeUtils.get_current_timestamp()
        diff = now - timestamp
        
        # Future times
        if diff < 0:
            diff = abs(diff)
            suffix = "from now"
        else:
            suffix = "ago"
        
        # Calculate relative time
        if diff < 60:
            return f"{int(diff)} seconds {suffix}"
        elif diff < 3600:
            minutes = int(diff / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} {suffix}"
        elif diff < 86400:
            hours = int(diff / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} {suffix}"
        elif diff < 604800:
            days = int(diff / 86400)
            return f"{days} day{'s' if days != 1 else ''} {suffix}"
        elif diff < 2592000:
            weeks = int(diff / 604800)
            return f"{weeks} week{'s' if weeks != 1 else ''} {suffix}"
        elif diff < 31536000:
            months = int(diff / 2592000)
            return f"{months} month{'s' if months != 1 else ''} {suffix}"
        else:
            years = int(diff / 31536000)
            return f"{years} year{'s' if years != 1 else ''} {suffix}"
    
    @staticmethod
    def calculate_duration(
        start_timestamp: float,
        end_timestamp: Optional[float] = None
    ) -> float:
        """
        Calculate duration between timestamps.
        
        Args:
            start_timestamp: Start timestamp
            end_timestamp: End timestamp (None = current time)
            
        Returns:
            float: Duration in seconds
        """
        if end_timestamp is None:
            end_timestamp = TimeUtils.get_current_timestamp()
        
        duration = end_timestamp - start_timestamp
        
        return duration
    
    @staticmethod
    def format_duration(duration_seconds: float) -> str:
        """
        Format duration in human-readable format.
        
        Args:
            duration_seconds: Duration in seconds
            
        Returns:
            str: Formatted duration (e.g., "2h 30m 15s")
        """
        if duration_seconds < 0:
            return "0s"
        
        hours = int(duration_seconds // 3600)
        minutes = int((duration_seconds % 3600) // 60)
        seconds = int(duration_seconds % 60)
        
        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if seconds > 0 or not parts:
            parts.append(f"{seconds}s")
        
        return " ".join(parts)
    
    @staticmethod
    def is_within_time_window(
        timestamp: float,
        window_seconds: float,
        reference_time: Optional[float] = None
    ) -> bool:
        """
        Check if timestamp is within time window of reference time.
        
        Args:
            timestamp: Timestamp to check
            window_seconds: Time window in seconds
            reference_time: Reference timestamp (None = current time)
            
        Returns:
            bool: True if within window
        """
        if reference_time is None:
            reference_time = TimeUtils.get_current_timestamp()
        
        diff = abs(timestamp - reference_time)
        
        return diff <= window_seconds
    
    @staticmethod
    def get_time_window_bounds(
        center_timestamp: Optional[float] = None,
        window_seconds: float = 3600
    ) -> Tuple[float, float]:
        """
        Get bounds of time window around center timestamp.
        
        Args:
            center_timestamp: Center timestamp (None = current time)
            window_seconds: Window size in seconds
            
        Returns:
            tuple: (start_timestamp, end_timestamp)
        """
        if center_timestamp is None:
            center_timestamp = TimeUtils.get_current_timestamp()
        
        half_window = window_seconds / 2
        
        start_timestamp = center_timestamp - half_window
        end_timestamp = center_timestamp + half_window
        
        return (start_timestamp, end_timestamp)
    
    @staticmethod
    def add_seconds(timestamp: float, seconds: float) -> float:
        """
        Add seconds to timestamp.
        
        Args:
            timestamp: Base timestamp
            seconds: Seconds to add (can be negative)
            
        Returns:
            float: New timestamp
        """
        return timestamp + seconds
    
    @staticmethod
    def add_hours(timestamp: float, hours: float) -> float:
        """
        Add hours to timestamp.
        
        Args:
            timestamp: Base timestamp
            hours: Hours to add (can be negative)
            
        Returns:
            float: New timestamp
        """
        return timestamp + (hours * 3600)
    
    @staticmethod
    def add_days(timestamp: float, days: float) -> float:
        """
        Add days to timestamp.
        
        Args:
            timestamp: Base timestamp
            days: Days to add (can be negative)
            
        Returns:
            float: New timestamp
        """
        return timestamp + (days * 86400)
    
    @staticmethod
    def get_date_components(timestamp: float) -> dict:
        """
        Extract date components from timestamp.
        
        Args:
            timestamp: Unix timestamp
            
        Returns:
            dict: Date components (year, month, day, hour, minute, second)
        """
        dt = TimeUtils.timestamp_to_datetime(timestamp)
        
        components = {
            'year': dt.year,
            'month': dt.month,
            'day': dt.day,
            'hour': dt.hour,
            'minute': dt.minute,
            'second': dt.second,
            'weekday': dt.weekday(),  # 0=Monday, 6=Sunday
            'day_of_year': dt.timetuple().tm_yday
        }
        
        return components
    
    @staticmethod
    def get_hour_bucket(timestamp: float) -> int:
        """
        Get hour bucket (for temporal clustering).
        
        Args:
            timestamp: Unix timestamp
            
        Returns:
            int: Hour bucket (hours since epoch)
        """
        return int(timestamp // 3600)
    
    @staticmethod
    def get_day_bucket(timestamp: float) -> int:
        """
        Get day bucket (for temporal clustering).
        
        Args:
            timestamp: Unix timestamp
            
        Returns:
            int: Day bucket (days since epoch)
        """
        return int(timestamp // 86400)
    
    @staticmethod
    def calculate_temporal_similarity(
        timestamp1: float,
        timestamp2: float,
        decay_hours: float = 24.0
    ) -> float:
        """
        Calculate temporal similarity between timestamps.
        
        Uses exponential decay function.
        
        Args:
            timestamp1: First timestamp
            timestamp2: Second timestamp
            decay_hours: Half-life for decay in hours
            
        Returns:
            float: Similarity score (0-1)
        """
        time_diff_hours = abs(timestamp1 - timestamp2) / 3600
        
        # Exponential decay
        import math
        similarity = math.exp(-time_diff_hours / decay_hours)
        
        return similarity
    
    @staticmethod
    def sleep(seconds: float):
        """
        Sleep for specified seconds.
        
        Args:
            seconds: Number of seconds to sleep
        """
        time.sleep(seconds)


# Convenience functions

def now() -> float:
    """Get current timestamp."""
    return TimeUtils.get_current_timestamp()


def format_timestamp(timestamp: Optional[float] = None, format_type: str = 'iso') -> str:
    """
    Format timestamp in various formats.
    
    Args:
        timestamp: Unix timestamp (None = current)
        format_type: Format type ('iso', 'human', 'relative')
        
    Returns:
        str: Formatted timestamp
    """
    if format_type == 'iso':
        return TimeUtils.format_iso8601(timestamp)
    elif format_type == 'human':
        return TimeUtils.format_human_readable(timestamp)
    elif format_type == 'relative':
        if timestamp is None:
            timestamp = TimeUtils.get_current_timestamp()
        return TimeUtils.format_relative(timestamp)
    else:
        return TimeUtils.format_iso8601(timestamp)


def time_ago(timestamp: float) -> str:
    """Get relative time string."""
    return TimeUtils.format_relative(timestamp)
