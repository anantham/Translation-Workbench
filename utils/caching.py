"""
Intelligent Caching System Module

Performance optimization through translation and similarity caching.
Reduces API costs and improves response times.

This module handles:
- Text hashing for cache keys
- Similarity score caching (JSON)
- AI translation caching (individual files)
- Cache statistics and management
"""

import os
import json
import hashlib

# Data directory structure
DATA_DIR = "data"
CACHE_DIR = os.path.join(DATA_DIR, "cache")
SIMILARITY_CACHE_FILE = os.path.join(CACHE_DIR, "similarity_scores_cache.json")
AI_TRANSLATION_CACHE_DIR = os.path.join(CACHE_DIR, "ai_translation_cache")

# Ensure cache directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(AI_TRANSLATION_CACHE_DIR, exist_ok=True)


def generate_text_hash(text):
    """Generate a hash for text content to use as cache key."""
    text_sample = text[:2000] if text else ""
    return hashlib.md5(text_sample.encode('utf-8')).hexdigest()


def load_similarity_cache():
    """Load cached similarity scores from disk."""
    if os.path.exists(SIMILARITY_CACHE_FILE):
        try:
            with open(SIMILARITY_CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load similarity cache: {e}")
            return {}
    return {}


def save_similarity_cache(cache):
    """Save similarity cache to disk."""
    try:
        with open(SIMILARITY_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save similarity cache: {e}")


def get_translation_cache_path(raw_text):
    """Generate cache file path for AI translation."""
    if not os.path.exists(AI_TRANSLATION_CACHE_DIR):
        os.makedirs(AI_TRANSLATION_CACHE_DIR)
    
    text_hash = generate_text_hash(raw_text)
    return os.path.join(AI_TRANSLATION_CACHE_DIR, f"translation_{text_hash}.txt")


def get_cached_translation(raw_text):
    """Get cached AI translation if available."""
    cache_path = get_translation_cache_path(raw_text)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Warning: Could not read translation cache: {e}")
    return None


def store_translation_in_cache(raw_text, translation):
    """Store AI translation in cache file."""
    cache_path = get_translation_cache_path(raw_text)
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(translation)
        return True
    except Exception as e:
        print(f"Warning: Could not save translation cache: {e}")
        return False


def get_translation_cache_stats():
    """Get statistics about the translation cache."""
    try:
        if not os.path.exists(AI_TRANSLATION_CACHE_DIR):
            return {"count": 0, "size_mb": 0.0}
        
        cache_files = [f for f in os.listdir(AI_TRANSLATION_CACHE_DIR) if f.endswith('.txt')]
        total_size = sum(os.path.getsize(os.path.join(AI_TRANSLATION_CACHE_DIR, f)) for f in cache_files)
        
        return {
            "count": len(cache_files),
            "size_mb": total_size / (1024 * 1024)
        }
    except Exception as e:
        print(f"Warning: Could not get cache stats: {e}")
        return {"count": 0, "size_mb": 0.0}