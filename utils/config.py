"""
Configuration Management Module

Handles API keys, metadata, and system configuration.
Supports multiple providers (Gemini, OpenAI, DeepSeek) with environment variable priority.

This module handles:
- API key loading with environment variable priority
- Configuration file management (config.json)
- EPUB metadata configuration with fallback defaults
- Multi-provider API configuration (Gemini, OpenAI, DeepSeek)
- Configuration status display and debugging
"""

import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file at the start
load_dotenv()

# --- Multi-Novel Data Structure ---
DATA_DIR = "data"
NOVELS_DIR = os.path.join(DATA_DIR, "novels")
SHARED_DIR = os.path.join(DATA_DIR, "shared")
TEMP_DIR = os.path.join(DATA_DIR, "temp")

# Shared paths (global across novels)
SHARED_CACHE_DIR = os.path.join(SHARED_DIR, "cache")
SHARED_MODELS_DIR = os.path.join(SHARED_DIR, "models")
SHARED_PROMPTS_FILE = os.path.join(SHARED_DIR, "custom_prompts.json")
SHARED_PRICING_FILE = os.path.join(SHARED_DIR, "pricing_config.json")

# Legacy paths (deprecated - for backward compatibility)
CACHE_DIR = SHARED_CACHE_DIR  # Backward compatibility
EXPORT_DIR = os.path.join(DATA_DIR, "exports")  # Will be novel-specific
MODELS_DIR = SHARED_MODELS_DIR  # Backward compatibility
TRANSLATIONS_DIR = os.path.join(DATA_DIR, "custom_translations")  # Will be novel-specific
EVALUATIONS_DIR = os.path.join(DATA_DIR, "evaluations")  # Will be novel-specific

# Helper functions for novel-specific paths
def get_novel_dir(novel_name):
    """Get the base directory for a specific novel."""
    safe_name = novel_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    return os.path.join(NOVELS_DIR, safe_name)

def get_novel_alignment_map(novel_name):
    """Get the alignment map file path for a specific novel."""
    return os.path.join(get_novel_dir(novel_name), "alignment_map.json")

def get_novel_cache_dir(novel_name):
    """Get the cache directory for a specific novel."""
    return os.path.join(get_novel_dir(novel_name), "cache")

def get_novel_exports_dir(novel_name):
    """Get the exports directory for a specific novel."""
    return os.path.join(get_novel_dir(novel_name), "exports")

def get_novel_ai_translations_dir(novel_name):
    """Get the AI translations directory for a specific novel."""
    return os.path.join(get_novel_dir(novel_name), "ai_translations")

def get_novel_raw_chapters_dir(novel_name):
    """Get the raw chapters directory for a specific novel."""
    return os.path.join(get_novel_dir(novel_name), "raw_chapters")

def get_novel_official_english_dir(novel_name):
    """Get the official English directory for a specific novel."""
    return os.path.join(get_novel_dir(novel_name), "official_english")

# Ensure shared directories exist
for directory in [DATA_DIR, SHARED_DIR, SHARED_CACHE_DIR, SHARED_MODELS_DIR, TEMP_DIR]:
    os.makedirs(directory, exist_ok=True)


def load_api_config():
    """Load API configuration from environment variable or config file.
    
    Returns:
        tuple: (api_key, source) where source is 'environment', 'config', or None
    """
    # 1. Check environment variable first (highest priority)
    api_key = os.getenv('GEMINI_API_KEY')
    if api_key:
        return api_key, "environment variable"
    
    # 2. Check config file
    config_path = "config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                api_key = config.get('gemini_api_key')
                if api_key:
                    return api_key, "config file"
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read config.json: {e}")
    
    # 3. No configuration found
    return None, None


def get_config_value(key, default=None):
    """Get a configuration value from config.json with fallback to default."""
    config_path = "config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get(key, default)
        except (json.JSONDecodeError, IOError):
            pass
    return default


def load_epub_metadata_config():
    """Load EPUB metadata configuration from config.json with fallback defaults."""
    metadata = get_config_value('epub_metadata', {})
    
    # Fallback defaults for missing values
    defaults = {
        "project_version": "v2.1.0",
        "license": "MIT License", 
        "maintainer_email": "contact@example.com",
        "framework_name": "Pluralistic Translation Workbench",
        "novel_title": "极道天魔 (Way of the Devil)",
        "original_author": "王雨 (Wang Yu)",
        "source_language": "Chinese (Simplified)",
        "target_language": "English",
        "github_url": "https://github.com/anthropics/translation-workbench",
        "feature_requests_url": "https://github.com/anthropics/translation-workbench/issues",
        "documentation_url": "https://docs.example.com/translation-workbench",
        "github_discussions_url": "https://github.com/anthropics/translation-workbench/discussions",
        "license_url": "https://github.com/anthropics/translation-workbench/blob/main/LICENSE",
        "translation_philosophy": "This translation was generated using an AI-powered framework designed for consistent, high-quality xianxia/wuxia novel translation. The goal is to allow you the reader to have full control over the translation style, meaning every novel can be translated into many pluralistic versions."
    }
    
    # Merge with defaults for any missing keys
    for key, default_value in defaults.items():
        if key not in metadata:
            metadata[key] = default_value
    
    return metadata


def show_config_status():
    """Display configuration status for debugging."""
    api_key, source = load_api_config()
    if api_key:
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        return f"✅ API Key loaded from {source} ({masked_key})"
    else:
        return "❌ API Key not configured"


def load_openai_api_config():
    """Load OpenAI API configuration from environment variable or config file.
    
    Returns:
        tuple: (api_key, source) where source is 'environment', 'config', or None
    """
    # 1. Check environment variable first (highest priority)
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        return api_key, "environment variable"
    
    # 2. Check config file
    config_path = "config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                api_key = config.get('openai_api_key')
                if api_key:
                    return api_key, "config file"
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read config.json: {e}")
    
    # 3. No configuration found
    return None, None


def load_deepseek_api_config():
    """Load DeepSeek API configuration from environment variable or config file.
    
    Returns:
        tuple: (api_key, source) where source is 'environment', 'config', or None
    """
    # 1. Check environment variable first (highest priority)
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if api_key:
        return api_key, "environment variable"
    
    # 2. Check config file
    config_path = "config.json"
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                api_key = config.get('deepseek_api_key')
                if api_key:
                    return api_key, "config file"
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read config.json: {e}")
    
    # 3. No configuration found
    return None, None


def show_openai_config_status():
    """Display OpenAI configuration status for debugging."""
    api_key, source = load_openai_api_config()
    if api_key:
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        return f"✅ OpenAI API Key loaded from {source} ({masked_key})"
    else:
        return "❌ OpenAI API Key not configured"


def show_deepseek_config_status():
    """Display DeepSeek configuration status for debugging."""
    api_key, source = load_deepseek_api_config()
    if api_key:
        masked_key = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        return f"✅ DeepSeek API Key loaded from {source} ({masked_key})"
    else:
        return "❌ DeepSeek API Key not configured"