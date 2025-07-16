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
import json
from datetime import datetime

# Import cache functions from our caching module
from .caching import get_cached_translation, store_translation_in_cache

# Import config functions for API keys
from .config import load_deepseek_api_config, get_config_value

# Import cost calculation functions
from .cost_tracking import calculate_openai_cost, calculate_gemini_cost

# Import logging
from .logging import logger

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
    logger.info(f"[GEMINI API] Starting translation request")
    logger.debug(f"[GEMINI API] Input text length: {len(raw_text)} characters")
    logger.debug(f"[GEMINI API] Cache enabled: {use_cache}")
    logger.debug(f"[GEMINI API] Novel name: {novel_name}")
    
    # Check cache first if enabled
    if use_cache:
        cached_translation = get_cached_translation(raw_text, novel_name=novel_name)
        if cached_translation:
            logger.info(f"[GEMINI API] Cache hit - returning cached translation")
            return cached_translation
    
    # Make API call if not cached
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key[:10]}..."
    headers = {'Content-Type': 'application/json'}
    prompt = f"Provide a high-quality, literal English translation of this Chinese web novel chapter. Keep paragraph breaks:\n\n{raw_text}"
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    logger.info(f"[GEMINI API] Making API request to: {gemini_url}")
    logger.debug(f"[GEMINI API] Request headers: {headers}")
    logger.debug(f"[GEMINI API] Request data structure: contents[0].parts[0].text = 'PROMPT + {len(raw_text)} chars of text'")
    logger.debug(f"[GEMINI API] Full prompt: {prompt[:200]}..." if len(prompt) > 200 else f"[GEMINI API] Full prompt: {prompt}")
    
    try:
        start_time = time.time()
        response = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}", headers=headers, json=data, timeout=90)
        request_time = time.time() - start_time
        
        logger.info(f"[GEMINI API] Response received in {request_time:.2f}s")
        logger.debug(f"[GEMINI API] Response status: {response.status_code}")
        logger.debug(f"[GEMINI API] Response headers: {dict(response.headers)}")
        
        response.raise_for_status()
        response_data = response.json()
        
        logger.debug(f"[GEMINI API] Response JSON structure: {list(response_data.keys())}")
        
        if 'candidates' in response_data and len(response_data['candidates']) > 0:
            translation = response_data['candidates'][0]['content']['parts'][0]['text']
            logger.info(f"[GEMINI API] Translation successful - {len(translation)} characters")
            logger.debug(f"[GEMINI API] Translation preview: {translation[:100]}..." if len(translation) > 100 else f"[GEMINI API] Translation: {translation}")
            
            # Store in cache if successful and caching is enabled
            if use_cache and translation and not translation.startswith("API Request Failed"):
                store_translation_in_cache(raw_text, translation, novel_name=novel_name)
                logger.debug(f"[GEMINI API] Translation stored in cache")
            
            return translation
        else:
            logger.error(f"[GEMINI API] Unexpected response structure: {response_data}")
            return f"API Request Failed: Unexpected response structure"
            
    except requests.exceptions.RequestException as e:
        logger.error(f"[GEMINI API] Request failed: {e}")
        return f"API Request Failed: {e}"
    except Exception as e:
        logger.error(f"[GEMINI API] Unexpected error: {e}")
        return f"API Request Failed: {e}"


def translate_with_openai(raw_text, api_key, model_name="gpt-4o-mini", system_prompt=None, history_examples=None, use_cache=True, max_tokens=None):
    """Translate text using OpenAI-compatible API (OpenAI, DeepSeek) with optional history context."""
    start_time = time.time()
    timestamp = datetime.now().isoformat()
    
    # Get max_tokens from config if not provided
    if max_tokens is None:
        max_tokens = get_config_value("max_tokens", 8000)
    
    logger.info(f"[OPENAI API] Starting translation request")
    logger.debug(f"[OPENAI API] Input text length: {len(raw_text)} characters")
    logger.debug(f"[OPENAI API] Model: {model_name}")
    logger.debug(f"[OPENAI API] Max tokens: {max_tokens}")
    logger.debug(f"[OPENAI API] System prompt provided: {system_prompt is not None}")
    logger.debug(f"[OPENAI API] History examples provided: {len(history_examples) if history_examples else 0}")
    logger.debug(f"[OPENAI API] Cache enabled: {use_cache}")
    
    if not OPENAI_AVAILABLE:
        logger.error(f"[OPENAI API] OpenAI SDK not available")
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
            logger.debug(f"[OPENAI API] Added system prompt: {system_prompt[:100]}..." if len(system_prompt) > 100 else f"[OPENAI API] Added system prompt: {system_prompt}")
        
        # Add history examples as conversation
        if history_examples:
            logger.info(f"[OPENAI API] Adding {len(history_examples)} history examples to conversation")
            for i, example in enumerate(history_examples):
                user_content = example["user"]
                assistant_content = example["model"]
                messages.append({"role": "user", "content": user_content})
                messages.append({"role": "assistant", "content": assistant_content})
                logger.debug(f"[OPENAI API] History example {i+1}:")
                logger.debug(f"[OPENAI API]   User: {user_content[:100]}..." if len(user_content) > 100 else f"[OPENAI API]   User: {user_content}")
                logger.debug(f"[OPENAI API]   Assistant: {assistant_content[:100]}..." if len(assistant_content) > 100 else f"[OPENAI API]   Assistant: {assistant_content}")
        
        # Add current translation request
        current_request = f"Translate this Chinese text to English:\n\n{raw_text}"
        messages.append({"role": "user", "content": current_request})
        logger.debug(f"[OPENAI API] Added current request: {current_request[:100]}..." if len(current_request) > 100 else f"[OPENAI API] Added current request: {current_request}")
        
        # Log complete conversation structure
        logger.info(f"[OPENAI API] Complete conversation structure:")
        for i, msg in enumerate(messages):
            content_preview = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
            logger.info(f"[OPENAI API]   Message {i+1}: {msg['role']} - {content_preview}")
        
        # Log API call parameters
        api_params = {
            "model": model_name,
            "messages": f"[{len(messages)} messages]",
            "temperature": 0.3,
            "max_tokens": max_tokens
        }
        logger.info(f"[OPENAI API] Making API call with parameters: {api_params}")
        
        # Make API call
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.3,
            max_tokens=max_tokens
        )
        
        request_time = time.time() - start_time
        
        logger.info(f"[OPENAI API] Response received in {request_time:.2f}s")
        logger.debug(f"[OPENAI API] Response object type: {type(response)}")
        
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
            
            logger.info(f"[OPENAI API] Usage metrics - Prompt: {prompt_tokens} tokens, Completion: {completion_tokens} tokens, Total: {total_tokens} tokens")
            
            # Calculate costs
            cost_info = calculate_openai_cost(model_name, usage_data)
            logger.debug(f"[OPENAI API] Cost calculation: {cost_info}")
            
            usage_metrics.update({
                'total_tokens': total_tokens,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'estimated_cost': cost_info.get('total_cost', 0.0),
                'input_cost': cost_info.get('input_cost', 0.0),
                'output_cost': cost_info.get('output_cost', 0.0)
            })
        else:
            logger.warning(f"[OPENAI API] No usage data available in response")
        
        translation = response.choices[0].message.content
        logger.info(f"[OPENAI API] Translation successful - {len(translation)} characters")
        logger.debug(f"[OPENAI API] Translation preview: {translation[:100]}..." if len(translation) > 100 else f"[OPENAI API] Translation: {translation}")
        
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
    
    logger.info(f"[GEMINI HISTORY] Starting translation with history")
    logger.debug(f"[GEMINI HISTORY] Model: {model_name}")
    logger.debug(f"[GEMINI HISTORY] System prompt provided: {system_prompt is not None}")
    logger.debug(f"[GEMINI HISTORY] History examples: {len(history) if history else 0}")
    logger.debug(f"[GEMINI HISTORY] Current text length: {len(current_raw_text)} characters")
    
    try:
        if not GOOGLE_AI_AVAILABLE:
            logger.error(f"[GEMINI HISTORY] Google AI SDK not available")
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
        logger.debug(f"[GEMINI HISTORY] Configured model: {model_name}")
        
        # Build the prompt with system instruction + examples + current task
        user_prompt_parts = []
        
        # Add System Prompt (if provided)
        if system_prompt:
            user_prompt_parts.append(f"**System:** {system_prompt}")
            logger.debug(f"[GEMINI HISTORY] Added system prompt: {system_prompt[:100]}..." if len(system_prompt) > 100 else f"[GEMINI HISTORY] Added system prompt: {system_prompt}")
        
        # Add Historical Examples 
        if history:
            user_prompt_parts.append("\n**Examples:**")
            logger.info(f"[GEMINI HISTORY] Adding {len(history)} history examples")
            for i, example in enumerate(history, 1):
                user_prompt_parts.append(f"\n**Example {i}:**")
                user_content = example['user']
                assistant_content = example['model']
                user_prompt_parts.append(f"Chinese: {user_content[:300]}...")  # Truncate for context
                user_prompt_parts.append(f"English: {assistant_content[:300]}...")
                logger.debug(f"[GEMINI HISTORY] Example {i}:")
                logger.debug(f"[GEMINI HISTORY]   Chinese: {user_content[:100]}..." if len(user_content) > 100 else f"[GEMINI HISTORY]   Chinese: {user_content}")
                logger.debug(f"[GEMINI HISTORY]   English: {assistant_content[:100]}..." if len(assistant_content) > 100 else f"[GEMINI HISTORY]   English: {assistant_content}")
        
        # Add Current Task
        user_prompt_parts.append(f"\n**Now translate this:**\n{current_raw_text}")
        logger.debug(f"[GEMINI HISTORY] Added current task: {current_raw_text[:100]}..." if len(current_raw_text) > 100 else f"[GEMINI HISTORY] Added current task: {current_raw_text}")
        
        # Combine all parts
        full_prompt = "\n".join(user_prompt_parts)
        logger.info(f"[GEMINI HISTORY] Final prompt length: {len(full_prompt)} characters")
        logger.debug(f"[GEMINI HISTORY] Final prompt preview: {full_prompt[:200]}..." if len(full_prompt) > 200 else f"[GEMINI HISTORY] Final prompt: {full_prompt}")
        
        # Generate translation
        logger.info(f"[GEMINI HISTORY] Making API call...")
        response = model.generate_content(full_prompt)
        request_time = time.time() - start_time
        
        logger.info(f"[GEMINI HISTORY] Response received in {request_time:.2f}s")
        logger.debug(f"[GEMINI HISTORY] Response object type: {type(response)}")
        
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
            
            logger.info(f"[GEMINI HISTORY] Usage metrics - Prompt: {prompt_tokens} tokens, Completion: {completion_tokens} tokens, Total: {total_tokens} tokens")
            
            # Calculate costs
            cost_info = calculate_gemini_cost(model_name, usage_metadata)
            logger.debug(f"[GEMINI HISTORY] Cost calculation: {cost_info}")
            
            usage_metrics.update({
                'total_tokens': total_tokens,
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'estimated_cost': cost_info.get('total_cost', 0.0),
                'input_cost': cost_info.get('input_cost', 0.0),
                'output_cost': cost_info.get('output_cost', 0.0)
            })
        else:
            logger.warning(f"[GEMINI HISTORY] No usage metadata available in response")
        
        if response.text:
            translation = response.text
            logger.info(f"[GEMINI HISTORY] Translation successful - {len(translation)} characters")
            logger.debug(f"[GEMINI HISTORY] Translation preview: {translation[:100]}..." if len(translation) > 100 else f"[GEMINI HISTORY] Translation: {translation}")
            return {
                'translation': translation,
                'success': True,
                'error': None,
                'usage_metrics': usage_metrics
            }
        else:
            logger.error(f"[GEMINI HISTORY] No translation text in response")
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


def generate_translation_unified(api_key, model_name, system_prompt, history, current_raw_text, platform="Gemini", max_tokens=None):
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
        return translate_with_openai(current_raw_text, api_key, model_name, system_prompt, history, max_tokens=max_tokens)
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
def translate_with_deepseek(raw_text, api_key, model_name="deepseek-chat", system_prompt=None, history_examples=None, use_cache=True, max_tokens=None):
    """DeepSeek translation - wrapper around translate_with_openai."""
    return translate_with_openai(raw_text, api_key, model_name, system_prompt, history_examples, use_cache, max_tokens)