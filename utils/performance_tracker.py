"""
Performance Tracking Module

This module tracks and analyzes AI model performance metrics including:
- Request timing per model
- Input size correlation with response time
- Failure patterns and timeout optimization
- Historical performance data storage

Used to optimize timeouts and provide performance insights for model selection.
"""

import os
import json
import time
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Import logging
try:
    from .logging import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Tracks and analyzes AI model performance for timeout optimization."""
    
    def __init__(self, data_dir: str = None):
        """Initialize performance tracker with data directory."""
        if data_dir is None:
            # Use data directory from parent directory
            self.data_dir = Path(__file__).parent.parent / "data" / "performance"
        else:
            self.data_dir = Path(data_dir)
        
        # Ensure directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.performance_file = self.data_dir / "model_performance.json"
        
        # Load existing performance data
        self.performance_data = self._load_performance_data()
    
    def _load_performance_data(self) -> Dict:
        """Load existing performance data from file."""
        if self.performance_file.exists():
            try:
                with open(self.performance_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load performance data: {e}, starting fresh")
        
        return {
            "models": {},
            "last_updated": datetime.now().isoformat(),
            "version": "1.0"
        }
    
    def _save_performance_data(self):
        """Save performance data to file."""
        try:
            self.performance_data["last_updated"] = datetime.now().isoformat()
            with open(self.performance_file, 'w', encoding='utf-8') as f:
                json.dump(self.performance_data, f, indent=2, ensure_ascii=False)
        except IOError as e:
            logger.error(f"Could not save performance data: {e}")
    
    def record_performance(self, 
                         platform: str,
                         model_name: str, 
                         request_time: float,
                         input_length: int,
                         success: bool,
                         error_type: str = None,
                         additional_metrics: Dict = None):
        """Record performance metrics for a model request."""
        
        model_key = f"{platform}:{model_name}"
        
        # Initialize model data if not exists
        if model_key not in self.performance_data["models"]:
            self.performance_data["models"][model_key] = {
                "platform": platform,
                "model_name": model_name,
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "timeout_failures": 0,
                "total_time": 0.0,
                "request_times": [],
                "input_lengths": [],
                "time_per_char": [],
                "first_seen": datetime.now().isoformat(),
                "last_used": datetime.now().isoformat(),
                "error_types": {}
            }
        
        model_data = self.performance_data["models"][model_key]
        
        # Update counters
        model_data["total_requests"] += 1
        model_data["last_used"] = datetime.now().isoformat()
        
        if success:
            model_data["successful_requests"] += 1
            model_data["total_time"] += request_time
            
            # Track timing data (keep last 100 requests)
            model_data["request_times"].append(request_time)
            model_data["input_lengths"].append(input_length)
            
            if input_length > 0:
                time_per_char = request_time / input_length
                model_data["time_per_char"].append(time_per_char)
            
            # Keep only recent data (last 100 requests)
            for key in ["request_times", "input_lengths", "time_per_char"]:
                if len(model_data[key]) > 100:
                    model_data[key] = model_data[key][-100:]
        else:
            model_data["failed_requests"] += 1
            
            # Track error types
            if error_type:
                if "timeout" in error_type.lower():
                    model_data["timeout_failures"] += 1
                
                if error_type not in model_data["error_types"]:
                    model_data["error_types"][error_type] = 0
                model_data["error_types"][error_type] += 1
        
        # Add additional metrics if provided
        if additional_metrics:
            if "additional_data" not in model_data:
                model_data["additional_data"] = []
            model_data["additional_data"].append({
                "timestamp": datetime.now().isoformat(),
                "metrics": additional_metrics
            })
        
        # Save updated data
        self._save_performance_data()
        
        logger.debug(f"Recorded performance for {model_key}: {request_time:.2f}s, success={success}")
    
    def get_model_performance(self, platform: str, model_name: str) -> Dict:
        """Get performance statistics for a specific model."""
        model_key = f"{platform}:{model_name}"
        
        if model_key not in self.performance_data["models"]:
            return {
                "exists": False,
                "message": f"No performance data for {model_key}"
            }
        
        model_data = self.performance_data["models"][model_key]
        request_times = model_data.get("request_times", [])
        time_per_char = model_data.get("time_per_char", [])
        
        stats = {
            "exists": True,
            "model_key": model_key,
            "platform": platform,
            "model_name": model_name,
            "total_requests": model_data["total_requests"],
            "successful_requests": model_data["successful_requests"],
            "failed_requests": model_data["failed_requests"],
            "timeout_failures": model_data["timeout_failures"],
            "success_rate": model_data["successful_requests"] / max(model_data["total_requests"], 1),
            "timeout_rate": model_data["timeout_failures"] / max(model_data["total_requests"], 1),
            "first_seen": model_data["first_seen"],
            "last_used": model_data["last_used"]
        }
        
        if request_times:
            stats.update({
                "avg_request_time": statistics.mean(request_times),
                "median_request_time": statistics.median(request_times),
                "min_request_time": min(request_times),
                "max_request_time": max(request_times),
                "std_request_time": statistics.stdev(request_times) if len(request_times) > 1 else 0
            })
        
        if time_per_char:
            stats.update({
                "avg_time_per_char": statistics.mean(time_per_char),
                "median_time_per_char": statistics.median(time_per_char)
            })
        
        return stats
    
    def calculate_optimal_timeout(self, 
                                platform: str, 
                                model_name: str, 
                                input_length: int,
                                safety_factor: float = 2.5,
                                min_timeout: int = 60,
                                max_timeout: int = 3600) -> int:
        """Calculate optimal timeout for a model based on historical performance."""
        
        stats = self.get_model_performance(platform, model_name)
        
        if not stats["exists"] or stats["total_requests"] < 3:
            # No sufficient data, use conservative default
            base_timeout = max(min_timeout, min(900, max_timeout))  # 15 minutes default
            logger.info(f"No performance history for {platform}:{model_name}, using default timeout: {base_timeout}s")
            return base_timeout
        
        # Calculate base timeout from historical data
        if "avg_time_per_char" in stats and input_length > 0:
            # Use per-character timing if available
            estimated_time = stats["avg_time_per_char"] * input_length
            
            # Add buffer based on standard deviation
            if "std_request_time" in stats and stats["std_request_time"] > 0:
                buffer = stats["std_request_time"] * 2  # 2 standard deviations
                estimated_time += buffer
        else:
            # Fallback to average request time
            estimated_time = stats.get("avg_request_time", min_timeout)
        
        # Apply safety factor and bounds
        timeout = int(estimated_time * safety_factor)
        timeout = max(min_timeout, min(timeout, max_timeout))
        
        # Increase timeout if model has recent timeout failures
        if stats["timeout_rate"] > 0.1:  # More than 10% timeouts
            timeout = int(timeout * 1.5)  # Add 50% buffer for problematic models
            timeout = min(timeout, max_timeout)
        
        logger.info(f"Calculated optimal timeout for {platform}:{model_name} "
                   f"(input: {input_length} chars): {timeout}s "
                   f"(est: {estimated_time:.1f}s, safety: {safety_factor}x)")
        
        return timeout
    
    def get_all_model_stats(self) -> Dict:
        """Get performance statistics for all tracked models."""
        return {
            model_key: self.get_model_performance(*model_key.split(":", 1))
            for model_key in self.performance_data["models"].keys()
        }
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Remove old performance data to prevent file bloat."""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cutoff_iso = cutoff_date.isoformat()
        
        models_to_remove = []
        for model_key, model_data in self.performance_data["models"].items():
            if model_data.get("last_used", "1970-01-01") < cutoff_iso:
                models_to_remove.append(model_key)
        
        for model_key in models_to_remove:
            del self.performance_data["models"][model_key]
            logger.info(f"Cleaned up old performance data for {model_key}")
        
        if models_to_remove:
            self._save_performance_data()
            return len(models_to_remove)
        
        return 0

# Global performance tracker instance
_global_tracker = None

def get_performance_tracker() -> PerformanceTracker:
    """Get the global performance tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = PerformanceTracker()
    return _global_tracker

def record_translation_performance(platform: str,
                                 model_name: str,
                                 request_time: float,
                                 input_text: str,
                                 success: bool,
                                 error_message: str = None,
                                 usage_metrics: Dict = None):
    """Convenience function to record translation performance."""
    tracker = get_performance_tracker()
    
    input_length = len(input_text) if input_text else 0
    error_type = None
    
    if not success and error_message:
        if "timeout" in error_message.lower():
            error_type = "timeout"
        elif "connection" in error_message.lower():
            error_type = "connection"
        else:
            error_type = "api_error"
    
    tracker.record_performance(
        platform=platform,
        model_name=model_name,
        request_time=request_time,
        input_length=input_length,
        success=success,
        error_type=error_type,
        additional_metrics=usage_metrics
    )

def get_optimal_timeout(platform: str, model_name: str, input_text: str) -> int:
    """Convenience function to get optimal timeout for a translation request."""
    tracker = get_performance_tracker()
    input_length = len(input_text) if input_text else 0
    return tracker.calculate_optimal_timeout(platform, model_name, input_length)