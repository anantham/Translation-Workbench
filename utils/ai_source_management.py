"""
AI Source and Model Management Module

Handles AI translation sources, model selection, and provider management.
Custom translation runs, model availability, and multi-provider support.

This module handles:
- Available AI translation source discovery (Fresh, Cached, Custom runs)
- AI translation content retrieval from various sources
- Model availability checking for different providers (Gemini, OpenAI, DeepSeek)
- Static and dynamic model lists for translation tasks
- Multi-provider model selection and validation
"""

import os

# Import data management functions
from .data_management import load_chapter_content

# Import caching functions
from .caching import get_cached_translation

# Import configuration
from .config import DATA_DIR


def get_available_ai_sources(current_chapter=None, novel_name="way_of_the_devil"):
    """Get list of available AI translation sources.
    
    Args:
        current_chapter: If provided, only include custom runs that have this chapter available
        novel_name: Name of the novel to search for AI translations
    
    Returns:
        list: Available AI translation sources
    """
    sources = ["Fresh Gemini Translation", "Cached Gemini Translation"]
    
    # Import config functions for novel-specific paths
    from .config import get_novel_ai_translations_dir
    
    # Check new multi-novel structure first
    novel_ai_translations_dir = get_novel_ai_translations_dir(novel_name)
    if os.path.exists(novel_ai_translations_dir):
        for run_name in os.listdir(novel_ai_translations_dir):
            run_path = os.path.join(novel_ai_translations_dir, run_name)
            if os.path.isdir(run_path):
                if current_chapter is not None:
                    # Check if this specific chapter exists in the run
                    chapter_file = f"Chapter-{current_chapter:04d}-translated.txt"
                    chapter_path = os.path.join(run_path, chapter_file)
                    if os.path.exists(chapter_path):
                        sources.append(f"Custom: {run_name}")
                else:
                    # Check if this run has any translation files (original behavior)
                    try:
                        translation_files = [f for f in os.listdir(run_path) if f.endswith('-translated.txt')]
                        if translation_files:
                            sources.append(f"Custom: {run_name}")
                    except OSError:
                        # Skip directories that can't be read
                        continue
    
    # Also check legacy custom translations directory for backward compatibility
    custom_translations_dir = os.path.join(DATA_DIR, "custom_translations")
    if os.path.exists(custom_translations_dir):
        for run_name in os.listdir(custom_translations_dir):
            run_path = os.path.join(custom_translations_dir, run_name)
            if os.path.isdir(run_path):
                if current_chapter is not None:
                    # Check if this specific chapter exists in the run
                    chapter_file = f"Chapter-{current_chapter:04d}-translated.txt"
                    chapter_path = os.path.join(run_path, chapter_file)
                    if os.path.exists(chapter_path):
                        sources.append(f"Legacy: {run_name}")
                else:
                    # Check if this run has any translation files (original behavior)
                    try:
                        translation_files = [f for f in os.listdir(run_path) if f.endswith('-translated.txt')]
                        if translation_files:
                            sources.append(f"Legacy: {run_name}")
                    except OSError:
                        # Skip directories that can't be read
                        continue
    
    return sources


def get_ai_translation_content(selected_ai_source, selected_chapter, session_state_ai_translation="", raw_content="", novel_name="way_of_the_devil"):
    """Get AI translation content based on selected source and chapter.
    
    Args:
        selected_ai_source: The selected AI source from the dropdown
        selected_chapter: The current chapter number
        session_state_ai_translation: Current session state AI translation
        raw_content: Raw Chinese content for fresh translation
        novel_name: Name of the novel to search for AI translations
    
    Returns:
        str: AI translation content or empty string if not available
    """
    if selected_ai_source.startswith("Custom: "):
        # Load from new multi-novel structure
        run_name = selected_ai_source[8:]  # Remove "Custom: " prefix
        custom_file = f"Chapter-{selected_chapter:04d}-translated.txt"
        
        # Import config functions for novel-specific paths
        from .config import get_novel_ai_translations_dir
        
        # Try new multi-novel structure first
        novel_ai_translations_dir = get_novel_ai_translations_dir(novel_name)
        custom_path = os.path.join(novel_ai_translations_dir, run_name, custom_file)
        custom_content = load_chapter_content(custom_path)
        
        if "File not found" not in custom_content:
            return custom_content
        
        # If not found in new structure, try legacy path as fallback
        legacy_path = os.path.join(DATA_DIR, "custom_translations", run_name, custom_file)
        legacy_content = load_chapter_content(legacy_path)
        
        if "File not found" in legacy_content:
            return ""
        return legacy_content
    
    elif selected_ai_source.startswith("Legacy: "):
        # Load from legacy custom translation run
        run_name = selected_ai_source[8:]  # Remove "Legacy: " prefix
        custom_file = f"Chapter-{selected_chapter:04d}-translated.txt"
        custom_path = os.path.join(DATA_DIR, "custom_translations", run_name, custom_file)
        custom_content = load_chapter_content(custom_path)
        
        if "File not found" in custom_content:
            return ""
        return custom_content
    
    elif selected_ai_source == "Fresh Gemini Translation":
        # Use session state AI translation (might be empty if not generated yet)
        return session_state_ai_translation
    
    elif selected_ai_source == "Cached Gemini Translation":
        # Try cached translation first, then session state
        cached_translation = get_cached_translation(raw_content) if raw_content else ""
        if cached_translation:
            return cached_translation
        return session_state_ai_translation
    
    return ""


def get_static_gemini_models():
    """Get static list of known Gemini models.
    
    Returns:
        list: Known Gemini model names
    """
    return [
        "gemini-2.5-flash",
        "gemini-2.5-pro", 
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-pro-001",
        "gemini-1.5-flash-001",
        "gemini-1.5-flash-002"
    ]


def get_available_openai_models(api_key):
    """Get available OpenAI models from API.
    
    Args:
        api_key: OpenAI API key
    
    Returns:
        tuple: (model_list, error_message)
    """
    try:
        # Check if OpenAI is available
        try:
            import openai
        except ImportError:
            return [], "OpenAI SDK not available"
        
        # Try to detect if this is a DeepSeek API key based on common patterns
        # DeepSeek keys often start with 'sk-' but we'll try both OpenAI and DeepSeek endpoints
        
        # First try OpenAI API
        try:
            client = openai.OpenAI(api_key=api_key)
            models = client.models.list()
            
            # Filter to relevant models for translation
            translation_models = []
            for model in models.data:
                model_id = model.id
                # Include GPT models suitable for fine-tuning and chat
                if any(prefix in model_id for prefix in ['gpt-4', 'gpt-3.5', 'ft:']):
                    translation_models.append(model_id)
            
            # Sort models: fine-tuned models first, then base models
            translation_models.sort(key=lambda x: (not x.startswith('ft:'), x))
            
            return translation_models, None
            
        except Exception as openai_error:
            # If OpenAI API fails, try DeepSeek API
            try:
                client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
                models = client.models.list()
                
                # Filter DeepSeek models - include all models from DeepSeek API
                translation_models = []
                for model in models.data:
                    model_id = model.id
                    # Include all DeepSeek models
                    translation_models.append(model_id)
                
                # If no models found from API, fallback to known DeepSeek models
                if not translation_models:
                    translation_models = ["deepseek-chat", "deepseek-reasoner"]
                
                return translation_models, None
                
            except Exception:
                # Both APIs failed, return the original OpenAI error
                return [], str(openai_error)
    
    except Exception as e:
        return [], str(e)


def get_available_models_for_translation(platform=None, api_key=None):
    """Get available models for translation based on platform.
    
    Args:
        platform: Target platform ("Gemini", "OpenAI", or None for all)
        api_key: API key for dynamic model discovery
    
    Returns:
        dict: Platform -> model list mapping
    """
    all_models = {}
    
    if platform is None or platform == "Gemini":
        # Add static Gemini models
        gemini_models = get_static_gemini_models()
        all_models["Gemini"] = gemini_models
    
    if platform is None or platform == "OpenAI":
        # Add OpenAI models if API key provided
        if api_key:
            openai_models, error = get_available_openai_models(api_key)
            if not error:
                all_models["OpenAI"] = openai_models
            else:
                # Fallback to common OpenAI models (including DeepSeek)
                all_models["OpenAI"] = [
                    "gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo",
                    "deepseek-chat", "deepseek-reasoner"
                ]
        else:
            # Default OpenAI models when no API key (including DeepSeek models)
            all_models["OpenAI"] = [
                "gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo",
                "deepseek-chat", "deepseek-reasoner"
            ]
    
    return all_models


def validate_model_availability(model_name, platform, api_key=None):
    """Validate if a specific model is available for the given platform.
    
    Args:
        model_name: Name of the model to validate
        platform: Platform name ("Gemini", "OpenAI")
        api_key: Optional API key for dynamic validation
    
    Returns:
        tuple: (is_available, error_message)
    """
    try:
        available_models = get_available_models_for_translation(platform, api_key)
        
        if platform not in available_models:
            return False, f"Platform {platform} not supported"
        
        if model_name in available_models[platform]:
            return True, None
        else:
            return False, f"Model {model_name} not available for {platform}"
    
    except Exception as e:
        return False, f"Validation error: {e}"


def get_model_recommendations(platform=None):
    """Get recommended models for different use cases.
    
    Args:
        platform: Target platform or None for all platforms
    
    Returns:
        dict: Recommendations by use case and platform
    """
    recommendations = {
        "Gemini": {
            "speed": "gemini-2.0-flash",
            "quality": "gemini-2.5-pro",
            "balanced": "gemini-2.5-flash",
            "fine_tuning": "gemini-1.5-flash-001"
        },
        "OpenAI": {
            "speed": "gpt-4o-mini",
            "quality": "gpt-4o",
            "balanced": "gpt-4o-mini",
            "fine_tuning": "gpt-4o-mini",
            "deepseek_chat": "deepseek-chat",
            "deepseek_reasoning": "deepseek-reasoner"
        }
    }
    
    if platform:
        return recommendations.get(platform, {})
    
    return recommendations


def detect_model_platform(model_name):
    """Detect which platform a model belongs to based on naming patterns.
    
    Args:
        model_name: Name of the model
    
    Returns:
        str: Platform name ("Gemini", "OpenAI", "DeepSeek", "Unknown")
    """
    model_name_lower = model_name.lower()
    
    if model_name_lower.startswith("gemini"):
        return "Gemini"
    elif model_name_lower.startswith("gpt") or model_name_lower.startswith("ft:gpt"):
        return "OpenAI"
    elif model_name_lower.startswith("deepseek"):
        return "DeepSeek"
    elif model_name_lower.startswith("ft:"):
        # Fine-tuned model, try to detect base model
        if "gpt" in model_name_lower:
            return "OpenAI"
        elif "gemini" in model_name_lower:
            return "Gemini"
    
    return "Unknown"


def get_model_info(model_name):
    """Get detailed information about a specific model.
    
    Args:
        model_name: Name of the model
    
    Returns:
        dict: Model information including platform, capabilities, etc.
    """
    platform = detect_model_platform(model_name)
    
    info = {
        "name": model_name,
        "platform": platform,
        "is_fine_tuned": model_name.startswith("ft:"),
        "capabilities": []
    }
    
    # Add platform-specific capabilities
    if platform == "Gemini":
        info["capabilities"] = ["text_generation", "multimodal", "fine_tuning"]
        if "flash" in model_name.lower():
            info["speed"] = "fast"
        if "pro" in model_name.lower():
            info["quality"] = "high"
    elif platform == "OpenAI":
        info["capabilities"] = ["text_generation", "fine_tuning"]
        if "mini" in model_name.lower():
            info["speed"] = "fast"
            info["cost"] = "low"
        if "gpt-4" in model_name.lower():
            info["quality"] = "high"
    elif platform == "DeepSeek":
        info["capabilities"] = ["text_generation", "reasoning"]
        if "reasoner" in model_name.lower():
            info["reasoning"] = "advanced"
    
    return info