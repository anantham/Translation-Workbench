"""
AI Translation Engine Module

Core translation functions for multiple providers.
Handles Gemini, OpenAI, and DeepSeek APIs with unified interface.

This module handles:
- Unified translation interface supporting multiple platforms
- Gemini API translation with history context
- OpenAI-compatible API translation (OpenAI, DeepSeek)
- Automatic cost calculation and usage tracking
- Translation caching for performance optimization
"""

import time
import requests
from datetime import datetime

# Import cache functions from our caching module
from .caching import get_cached_translation, store_translation_in_cache

# Import config functions for API keys
from .config import load_deepseek_api_config

# Import cost calculation functions
from .cost_tracking import calculate_openai_cost, calculate_gemini_cost

# AI SDK availability detection
GOOGLE_AI_AVAILABLE = False
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    pass

OPENAI_AVAILABLE = False
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    pass


def translate_with_gemini(raw_text: str, api_key: str, use_cache=True, novel_name: str = None):
    """Sends raw text to Gemini for translation with caching support."""
    # Check cache first if enabled
    if use_cache:
        cached_translation = get_cached_translation(raw_text, novel_name=novel_name)
        if cached_translation:
            return cached_translation
    
    # Make API call if not cached
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    prompt = f"Provide a high-quality, literal English translation of this Chinese web novel chapter. Keep paragraph breaks:\n\n{raw_text}"
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(gemini_url, headers=headers, json=data, timeout=90)
        response.raise_for_status()
        translation = response.json()['candidates'][0]['content']['parts'][0]['text']
        
        # Store in cache if successful and caching is enabled
        if use_cache and translation and not translation.startswith("API Request Failed"):
            store_translation_in_cache(raw_text, translation, novel_name=novel_name)
        
        return translation
    except Exception as e:
        return f"API Request Failed: {e}"


def translate_with_openai(raw_text, api_key, model_name="gpt-4o-mini", system_prompt=None, history_examples=None, use_cache=True):
    """Translate text using OpenAI-compatible API (OpenAI, DeepSeek) with optional history context."""
    start_time = time.time()
    timestamp = datetime.now().isoformat()
    
    if not OPENAI_AVAILABLE:
        return {
            'translation': '',
            'success': False,
            'error': "OpenAI SDK not available",
            'usage_metrics': {
                'total_tokens': 0,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'estimated_cost': 0.0,
                'input_cost': 0.0,
                'output_cost': 0.0,
                'request_time': time.time() - start_time,
                'timestamp': timestamp,
                'model': model_name,
                'platform': 'OpenAI'
            }
        }
    
    try:
        # Determine if this is a DeepSeek model and configure accordingly
        if model_name.startswith("deepseek"):
            # For DeepSeek models, load DeepSeek API key from environment/config
            deepseek_key, deepseek_source = load_deepseek_api_config()
            if not deepseek_key:
                return {
                    'translation': '',
                    'success': False,
                    'error': "DeepSeek API key not found. Please set DEEPSEEK_API_KEY environment variable or add deepseek_api_key to config.json",
                    'usage_metrics': {
                        'total_tokens': 0,
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'estimated_cost': 0.0,
                        'input_cost': 0.0,
                        'output_cost': 0.0,
                        'request_time': time.time() - start_time,
                        'timestamp': timestamp,
                        'model': model_name,
                        'platform': 'OpenAI'
                    }
                }
            
            client = openai.OpenAI(
                api_key=deepseek_key,
                base_url="https://api.deepseek.com"
            )
        else:
            # Use OpenAI API configuration
            client = openai.OpenAI(api_key=api_key)
        
        # Build messages
        messages = []
        
        # Add system prompt
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add history examples as conversation
        if history_examples:
            for example in history_examples:
                messages.append({"role": "user", "content": example["user"]})
                messages.append({"role": "assistant", "content": example["model"]})
        
        # Add current translation request
        messages.append({"role": "user", "content": f"Translate this Chinese text to English:\n\n{raw_text}"})
        
        # Make API call
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=4000
        )
        
        request_time = time.time() - start_time
        
        # Calculate usage metrics
        usage_metrics = {
            'total_tokens': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'estimated_cost': 0.0,
            'input_cost': 0.0,
            'output_cost': 0.0,
            'request_time': request_time,
            'timestamp': timestamp,
            'model': model_name,
            'platform': 'OpenAI'
        }
        
        # Extract usage data if available
        if hasattr(response, 'usage') and response.usage:
            usage_data = response.usage
            prompt_tokens = getattr(usage_data, 'prompt_tokens', 0)
            completion_tokens = getattr(usage_data, 'completion_tokens', 0)
            total_tokens = getattr(usage_data, 'total_tokens', prompt_tokens + completion_tokens)
            
            # Calculate costs
            cost_info = calculate_openai_cost(model_name, usage_data)
            
            usage_metrics.update({
                'total_tokens': total_tokens,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'estimated_cost': cost_info.get('total_cost', 0.0),
                'input_cost': cost_info.get('input_cost', 0.0),
                'output_cost': cost_info.get('output_cost', 0.0)
            })
        
        translation = response.choices[0].message.content
        return {
            'translation': translation,
            'success': True,
            'error': None,
            'usage_metrics': usage_metrics
        }
        
    except Exception as e:
        return {
            'translation': '',
            'success': False,
            'error': f"OpenAI API Request Failed: {e}",
            'usage_metrics': {
                'total_tokens': 0,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'estimated_cost': 0.0,
                'input_cost': 0.0,
                'output_cost': 0.0,
                'request_time': time.time() - start_time,
                'timestamp': timestamp,
                'model': model_name,
                'platform': 'OpenAI'
            }
        }


def translate_with_gemini_history(api_key, model_name, system_prompt, history, current_raw_text):
    """Translate using Gemini with history context (original implementation)."""
    start_time = time.time()
    timestamp = datetime.now().isoformat()
    
    try:
        if not GOOGLE_AI_AVAILABLE:
            return {
                'translation': '',
                'success': False,
                'error': "Google AI SDK not available",
                'usage_metrics': {
                    'total_tokens': 0,
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'estimated_cost': 0.0,
                    'input_cost': 0.0,
                    'output_cost': 0.0,
                    'request_time': time.time() - start_time,
                    'timestamp': timestamp,
                    'model': model_name,
                    'platform': 'Gemini'
                }
            }
            
        # Configure API
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        # Build the prompt with system instruction + examples + current task
        user_prompt_parts = []
        
        # Add System Prompt (if provided)
        if system_prompt:
            user_prompt_parts.append(f"**System:** {system_prompt}")
        
        # Add Historical Examples 
        if history:
            user_prompt_parts.append("\n**Examples:**")
            for i, example in enumerate(history, 1):
                user_prompt_parts.append(f"\n**Example {i}:**")
                user_prompt_parts.append(f"Chinese: {example['user'][:300]}...")  # Truncate for context
                user_prompt_parts.append(f"English: {example['model'][:300]}...")
        
        # Add Current Task
        user_prompt_parts.append(f"\n**Now translate this:**\n{current_raw_text}")
        
        # Combine all parts
        full_prompt = "\n".join(user_prompt_parts)
        
        # Generate translation
        response = model.generate_content(full_prompt)
        request_time = time.time() - start_time
        
        # Calculate usage metrics
        usage_metrics = {
            'total_tokens': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'estimated_cost': 0.0,
            'input_cost': 0.0,
            'output_cost': 0.0,
            'request_time': request_time,
            'timestamp': timestamp,
            'model': model_name,
            'platform': 'Gemini'
        }
        
        # Extract usage metadata if available
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage_metadata = response.usage_metadata
            prompt_tokens = getattr(usage_metadata, 'prompt_token_count', 0)
            completion_tokens = getattr(usage_metadata, 'candidates_token_count', 0)
            total_tokens = prompt_tokens + completion_tokens
            
            # Calculate costs
            cost_info = calculate_gemini_cost(model_name, usage_metadata)
            
            usage_metrics.update({
                'total_tokens': total_tokens,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'estimated_cost': cost_info.get('total_cost', 0.0),
                'input_cost': cost_info.get('input_cost', 0.0),
                'output_cost': cost_info.get('output_cost', 0.0)
            })
        
        if response.text:
            return {
                'translation': response.text,
                'success': True,
                'error': None,
                'usage_metrics': usage_metrics
            }
        else:
            return {
                'translation': '',
                'success': False,
                'error': "No translation generated",
                'usage_metrics': usage_metrics
            }
            
    except Exception as e:
        return {
            'translation': '',
            'success': False,
            'error': f"Gemini API Request Failed: {e}",
            'usage_metrics': {
                'total_tokens': 0,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'estimated_cost': 0.0,
                'input_cost': 0.0,
                'output_cost': 0.0,
                'request_time': time.time() - start_time,
                'timestamp': timestamp,
                'model': model_name,
                'platform': 'Gemini'
            }
        }


def generate_translation_unified(api_key, model_name, system_prompt, history, current_raw_text, platform="Gemini"):
    """Unified translation function supporting both Gemini and OpenAI.
    
    Returns:
        dict: {
            'translation': str,
            'success': bool,
            'error': str or None,
            'usage_metrics': {
                'total_tokens': int,
                'prompt_tokens': int,
                'completion_tokens': int,
                'estimated_cost': float,
                'input_cost': float,
                'output_cost': float,
                'request_time': float,
                'timestamp': str,
                'model': str,
                'platform': str
            }
        }
    """
    if platform == "Gemini":
        return translate_with_gemini_history(api_key, model_name, system_prompt, history, current_raw_text)
    elif platform == "OpenAI":
        return translate_with_openai(current_raw_text, api_key, model_name, system_prompt, history)
    else:
        return {
            'translation': '',
            'success': False,
            'error': f"Unsupported platform: {platform}",
            'usage_metrics': {
                'total_tokens': 0,
                'prompt_tokens': 0,
                'completion_tokens': 0,
                'estimated_cost': 0.0,
                'input_cost': 0.0,
                'output_cost': 0.0,
                'request_time': 0.0,
                'timestamp': datetime.now().isoformat(),
                'model': model_name,
                'platform': platform
            }
        }


# Alias for backward compatibility with DeepSeek
def translate_with_deepseek(raw_text, api_key, model_name="deepseek-chat", system_prompt=None, history_examples=None, use_cache=True):
    """DeepSeek translation - wrapper around translate_with_openai."""
    return translate_with_openai(raw_text, api_key, model_name, system_prompt, history_examples, use_cache)