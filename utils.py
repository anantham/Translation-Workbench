"""
Shared Utilities for Translation Workbench
Houses all common functions to maintain modularity and prevent code duplication
"""

import os
import json
import requests
import hashlib
import time
import re
import pandas as pd
from datetime import datetime
from collections import Counter
import streamlit as st
from difflib import SequenceMatcher

# --- Organized Data Structure ---
DATA_DIR = "data"
CACHE_DIR = os.path.join(DATA_DIR, "cache")
EXPORT_DIR = os.path.join(DATA_DIR, "exports")
TEMP_DIR = os.path.join(DATA_DIR, "temp")
MODELS_DIR = os.path.join(DATA_DIR, "models")
TRANSLATIONS_DIR = os.path.join(DATA_DIR, "custom_translations")
EVALUATIONS_DIR = os.path.join(DATA_DIR, "evaluations")

# Ensure directories exist
for directory in [DATA_DIR, CACHE_DIR, EXPORT_DIR, TEMP_DIR, MODELS_DIR, TRANSLATIONS_DIR, EVALUATIONS_DIR]:
    os.makedirs(directory, exist_ok=True)

# --- Import detection for optional dependencies ---
SEMANTIC_AVAILABLE = False
SEMANTIC_ERROR_MESSAGE = ""

try:
    import torch
    SEMANTIC_ERROR_MESSAGE += "✅ torch imported successfully\n"
except ImportError as e:
    SEMANTIC_ERROR_MESSAGE += f"❌ torch import failed: {e}\n"

try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_ERROR_MESSAGE += "✅ sentence-transformers imported successfully\n"
    SEMANTIC_AVAILABLE = True
except ImportError as e:
    SEMANTIC_ERROR_MESSAGE += f"❌ sentence-transformers import failed: {e}\n"

if not SEMANTIC_AVAILABLE:
    SEMANTIC_ERROR_MESSAGE += "📝 Falling back to syntactic similarity (difflib)\n"

# Google AI SDK
GOOGLE_AI_AVAILABLE = False
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
    print("✅ Google AI SDK available for fine-tuning")
except ImportError:
    print("❌ Google AI SDK not available (pip install google-generativeai)")

# OpenAI SDK
OPENAI_AVAILABLE = False
try:
    import openai
    OPENAI_AVAILABLE = True
    print("✅ OpenAI SDK available for fine-tuning")
except ImportError:
    print("❌ OpenAI SDK not available (pip install openai)")

# --- Cost Calculation System ---
def load_pricing_config():
    """Load current API pricing configuration."""
    pricing_file = os.path.join(DATA_DIR, "pricing_config.json")
    try:
        with open(pricing_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback pricing if config file missing
        return {
            "openai_models": {
                "gpt-4o": {"input_per_1m": 5.00, "output_per_1m": 20.00},
                "gpt-4o-mini": {"input_per_1m": 0.60, "output_per_1m": 2.40}
            },
            "gemini_models": {
                "gemini-1.5-pro": {"input_per_1m": 0.00125, "output_per_1m": 0.005}
            }
        }

def calculate_openai_cost(model_name, usage_data):
    """Calculate precise OpenAI API cost based on current pricing."""
    pricing_config = load_pricing_config()
    
    # Handle fine-tuned models (ft:gpt-4o-mini:org:name:id format)
    base_model = model_name
    is_fine_tuned = model_name.startswith("ft:")
    if is_fine_tuned:
        # Extract base model from fine-tuned name
        parts = model_name.split(":")
        if len(parts) >= 2:
            base_model = parts[1]  # e.g., "gpt-4o-mini"
    
    # Normalize model name for pricing lookup
    model_key = base_model.replace("-latest", "").replace("chatgpt-", "gpt-")
    
    # Look up pricing
    pricing_source = "fine_tuned_models" if is_fine_tuned else "openai_models"
    if pricing_source in pricing_config and model_key in pricing_config[pricing_source]:
        rates = pricing_config[pricing_source][model_key]
        
        input_cost = (usage_data.prompt_tokens / 1_000_000) * rates["input_per_1m"]
        output_cost = (usage_data.completion_tokens / 1_000_000) * rates["output_per_1m"]
        total_cost = input_cost + output_cost
        
        return {
            "total_cost": round(total_cost, 6),
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "input_rate": rates["input_per_1m"],
            "output_rate": rates["output_per_1m"],
            "is_fine_tuned": is_fine_tuned,
            "base_model": base_model
        }
    
    # Fallback for unknown models
    return {
        "total_cost": 0.0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "input_rate": 0.0,
        "output_rate": 0.0,
        "is_fine_tuned": is_fine_tuned,
        "base_model": base_model,
        "error": f"Unknown model pricing: {model_key}"
    }

def calculate_gemini_cost(model_name, usage_metadata):
    """Calculate Gemini API cost based on current pricing."""
    pricing_config = load_pricing_config()
    
    if not usage_metadata:
        return {"total_cost": 0.0, "input_cost": 0.0, "output_cost": 0.0}
    
    # Look up Gemini pricing
    model_key = model_name.replace("models/", "")
    if "gemini_models" in pricing_config and model_key in pricing_config["gemini_models"]:
        rates = pricing_config["gemini_models"][model_key]
        
        input_tokens = usage_metadata.prompt_token_count
        output_tokens = usage_metadata.candidates_token_count
        
        input_cost = (input_tokens / 1_000_000) * rates["input_per_1m"]
        output_cost = (output_tokens / 1_000_000) * rates["output_per_1m"]
        total_cost = input_cost + output_cost
        
        return {
            "total_cost": round(total_cost, 6),
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "input_rate": rates["input_per_1m"],
            "output_rate": rates["output_per_1m"]
        }
    
    # Fallback for unknown Gemini models
    return {
        "total_cost": 0.0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "error": f"Unknown Gemini model pricing: {model_key}"
    }

# --- API Configuration System ---
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

# --- Caching System ---
SIMILARITY_CACHE_FILE = os.path.join(CACHE_DIR, "similarity_scores_cache.json")
AI_TRANSLATION_CACHE_DIR = os.path.join(CACHE_DIR, "ai_translation_cache")

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

# --- AI Translation Caching ---
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

# --- Content Loading ---
def load_chapter_content(filepath):
    """Load content from chapter file."""
    if filepath and os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"
    return "File not found or not applicable."

def load_alignment_map(filepath="alignment_map.json"):
    """Load the alignment map with session state persistence."""
    # Check if file has been modified since last load
    if os.path.exists(filepath):
        file_mtime = os.path.getmtime(filepath)
        
        # Load from session if available and file hasn't changed
        if ('alignment_map' in st.session_state and 
            'alignment_map_mtime' in st.session_state and
            st.session_state.alignment_map_mtime == file_mtime):
            return st.session_state.alignment_map
        
        # Load fresh and store in session
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                alignment_map = json.load(f)
            st.session_state.alignment_map = alignment_map
            st.session_state.alignment_map_mtime = file_mtime
            return alignment_map
        except Exception as e:
            st.error(f"❌ Error loading alignment map: {e}")
            return None
    else:
        st.error(f"❌ Alignment map '{filepath}' not found.")
        return None

# --- Text Statistics ---
def get_text_stats(content, language_hint=None):
    """Get comprehensive text statistics with language-aware counting."""
    if not content or content == "File not found or not applicable.":
        return {
            'char_count': 0,
            'word_count': 0,
            'line_count': 0,
            'avg_words_per_line': 0,
            'language': 'unknown'
        }
    
    # Basic counts
    char_count = len(content)
    line_count = len(content.splitlines())
    
    # Language detection if not provided
    if language_hint is None:
        # Simple heuristic: if >20% of characters are CJK, consider it Chinese
        cjk_chars = sum(1 for char in content if '\u4e00' <= char <= '\u9fff')
        if char_count > 0:
            language_hint = 'chinese' if cjk_chars / char_count > 0.2 else 'english'
        else:
            language_hint = 'english'
    
    # Language-aware word counting
    if language_hint == 'chinese':
        try:
            import jieba
            word_count = len(list(jieba.cut(content)))
        except ImportError:
            # Fallback: count Chinese characters (CJK range)
            word_count = sum(1 for char in content if '\u4e00' <= char <= '\u9fff')
        detected_language = 'chinese'
    else:
        # English and other space-separated languages
        word_count = len(content.split())
        detected_language = 'english'
    
    # Average words per line (avoid division by zero)
    avg_words_per_line = word_count / line_count if line_count > 0 else 0
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'line_count': line_count,
        'avg_words_per_line': round(avg_words_per_line, 1),
        'language': detected_language
    }

# --- AI Translation Functions ---
def translate_with_gemini(raw_text: str, api_key: str, use_cache=True):
    """Sends raw text to Gemini for translation with caching support."""
    # Check cache first if enabled
    if use_cache:
        cached_translation = get_cached_translation(raw_text)
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
            store_translation_in_cache(raw_text, translation)
        
        return translation
    except Exception as e:
        return f"API Request Failed: {e}"

# --- Fine-tuning Functions ---
def chunk_chapter_for_training(raw_content, english_content, max_chars=4500):
    """
    Split chapters into trainable chunks to stay under Gemini's 5000 character output limit.
    Maintains paragraph boundaries for better context preservation.
    """
    chunks = []
    
    # Split by paragraphs to maintain context
    raw_paragraphs = raw_content.strip().split('\n\n')
    eng_paragraphs = english_content.strip().split('\n\n')
    
    # Handle mismatched paragraph counts by using the shorter one
    min_paragraphs = min(len(raw_paragraphs), len(eng_paragraphs))
    raw_paragraphs = raw_paragraphs[:min_paragraphs]
    eng_paragraphs = eng_paragraphs[:min_paragraphs]
    
    current_raw = []
    current_eng = []
    current_size = 0
    
    for raw_para, eng_para in zip(raw_paragraphs, eng_paragraphs):
        para_size = len(eng_para)
        
        # If adding this paragraph would exceed limit and we have content, save current chunk
        if current_size + para_size > max_chars and current_raw:
            chunks.append({
                'raw': '\n\n'.join(current_raw),
                'english': '\n\n'.join(current_eng)
            })
            current_raw = [raw_para]
            current_eng = [eng_para]
            current_size = para_size
        else:
            current_raw.append(raw_para)
            current_eng.append(eng_para)
            current_size += para_size
    
    # Don't forget the last chunk
    if current_raw:
        chunks.append({
            'raw': '\n\n'.join(current_raw),
            'english': '\n\n'.join(current_eng)
        })
    
    return chunks

def load_bert_scores_from_reports():
    """Load BERT similarity scores from existing CSV reports."""
    bert_scores = {}
    
    if not os.path.exists(EXPORT_DIR):
        return bert_scores
    
    # Find the most recent CSV report
    csv_files = [f for f in os.listdir(EXPORT_DIR) if f.startswith('dataset_report_') and f.endswith('.csv')]
    if not csv_files:
        return bert_scores
    
    # Sort by filename (timestamp) to get most recent
    csv_files.sort(reverse=True)
    most_recent_csv = os.path.join(EXPORT_DIR, csv_files[0])
    
    try:
        import pandas as pd
        df = pd.read_csv(most_recent_csv)
        
        # Extract BERT scores by chapter
        for _, row in df.iterrows():
            chapter_num = int(row['Chapter'])
            bert_score = float(row['BERT_Similarity']) if pd.notna(row['BERT_Similarity']) else None
            bert_scores[chapter_num] = bert_score
        
        print(f"✅ Loaded BERT scores for {len(bert_scores)} chapters from {csv_files[0]}")
        
    except Exception as e:
        print(f"⚠️ Could not load BERT scores: {e}")
    
    return bert_scores

def load_dataset_for_tuning(alignment_map, limit=None, min_similarity=None, max_chars=None, include_bert_scores=True):
    """
    Load dataset from alignment map and prepare for fine-tuning.
    
    Args:
        alignment_map: Chapter alignment mapping
        limit: Maximum number of chapters to process
        min_similarity: Minimum BERT similarity threshold (if available and specified)
        max_chars: Maximum character count per chapter (no limit if None)
        include_bert_scores: Whether to load BERT scores from existing reports
    
    Returns:
        list: Training examples in the format expected by fine-tuning APIs
    """
    training_examples = []
    processed = 0
    
    # Load BERT scores from existing reports if available
    bert_scores = {}
    if include_bert_scores:
        bert_scores = load_bert_scores_from_reports()
    
    # Get sorted chapter numbers
    chapter_numbers = sorted([int(k) for k in alignment_map.keys()])
    if limit:
        chapter_numbers = chapter_numbers[:limit]
    
    for chapter_num in chapter_numbers:
        chapter_data = alignment_map[str(chapter_num)]
        
        # Check if both files exist
        if not chapter_data.get('raw_file') or not chapter_data.get('english_file'):
            continue
        
        # Load content
        raw_content = load_chapter_content(chapter_data['raw_file'])
        english_content = load_chapter_content(chapter_data['english_file'])
        
        if "File not found" in raw_content or "File not found" in english_content:
            continue
        
        # Get text statistics
        raw_stats = get_text_stats(raw_content, 'chinese')
        eng_stats = get_text_stats(english_content, 'english')
        
        # Apply character count limit only if specified
        if max_chars is not None:
            if raw_stats['char_count'] > max_chars or eng_stats['char_count'] > max_chars:
                continue
        
        # Check BERT similarity threshold only if specified
        bert_score = bert_scores.get(chapter_num)
        if min_similarity is not None and bert_score is not None and bert_score < min_similarity:
            continue
        
        # Create training example
        training_example = {
            "chapter_number": chapter_num,
            "raw_content": raw_content,
            "english_content": english_content,
            "raw_stats": raw_stats,
            "english_stats": eng_stats,
            "bert_similarity": bert_score  # Include BERT score if available
        }
        
        training_examples.append(training_example)
        processed += 1
    
    return training_examples

def get_max_available_chapters(alignment_map):
    """
    Get the maximum number of chapters available for training based on 
    existing English chapter files.
    
    Args:
        alignment_map: Chapter alignment mapping
    
    Returns:
        int: Maximum number of available chapters for training
    """
    available_count = 0
    chapter_numbers = sorted([int(k) for k in alignment_map.keys()])
    
    for chapter_num in chapter_numbers:
        chapter_data = alignment_map[str(chapter_num)]
        
        # Check if English file exists
        if not chapter_data.get('english_file'):
            continue
            
        english_file_path = chapter_data['english_file']
        if os.path.exists(english_file_path):
            available_count += 1
        else:
            # If we hit a missing file, assume subsequent files might also be missing
            # But continue counting to get the actual total
            continue
    
    return available_count

def prepare_training_data_for_api(training_examples, train_split=0.8, max_output_chars=4500):
    """
    Convert training examples to the format expected by fine-tuning APIs.
    Automatically chunks long chapters to stay under Gemini's 5000 character limit.
    
    Returns:
        tuple: (train_data, val_data) in JSONL format
    """
    import random
    
    # Convert chapters to chunks
    all_chunks = []
    total_chapters = 0
    total_chunks = 0
    over_limit_count = 0
    
    for example in training_examples:
        total_chapters += 1
        
        # Check if chapter needs chunking
        if len(example['english_content']) > max_output_chars:
            # Chunk this chapter
            chunks = chunk_chapter_for_training(
                example['raw_content'], 
                example['english_content'],
                max_output_chars
            )
            over_limit_count += 1
        else:
            # Use whole chapter as single chunk
            chunks = [{
                'raw': example['raw_content'],
                'english': example['english_content']
            }]
        
        # Format each chunk
        for i, chunk in enumerate(chunks):
            formatted_chunk = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional translator specializing in Chinese to English translation of web novels. Provide accurate, fluent translations that preserve the original meaning and style."
                    },
                    {
                        "role": "user",
                        "content": f"Translate this Chinese web novel excerpt to English:\n\n{chunk['raw']}"
                    },
                    {
                        "role": "assistant",
                        "content": chunk['english']
                    }
                ],
                "metadata": {
                    "chapter": example['chapter_number'],
                    "chunk": i + 1,
                    "total_chunks": len(chunks),
                    "is_chunked": len(chunks) > 1
                }
            }
            all_chunks.append(formatted_chunk)
            total_chunks += 1
    
    print(f"📊 Chunking Summary:")
    print(f"   • {total_chapters} chapters processed")
    print(f"   • {over_limit_count} chapters required chunking")
    print(f"   • {total_chunks} total training examples created")
    print(f"   • Average {total_chunks/total_chapters:.1f} chunks per chapter")
    
    # Shuffle and split chunks
    random.shuffle(all_chunks)
    split_idx = int(len(all_chunks) * train_split)
    
    train_data = all_chunks[:split_idx]
    val_data = all_chunks[split_idx:]
    
    # Verify no chunks exceed the limit
    oversized_train = sum(1 for ex in train_data if len(ex['messages'][2]['content']) > max_output_chars)
    oversized_val = sum(1 for ex in val_data if len(ex['messages'][2]['content']) > max_output_chars)
    
    if oversized_train > 0 or oversized_val > 0:
        print(f"⚠️ Warning: {oversized_train + oversized_val} chunks still exceed {max_output_chars} chars")
    else:
        print(f"✅ All chunks are under {max_output_chars} characters")
    
    return train_data, val_data

# --- Fine-tuning Job Management (Google AI) ---
def start_finetuning_job(api_key, training_data, base_model="models/gemini-1.5-flash-001", 
                        epoch_count=3, batch_size=4, learning_rate=0.001):
    """Start a fine-tuning job using Google AI SDK."""
    if not GOOGLE_AI_AVAILABLE:
        return None, "Google AI SDK not available"
    
    try:
        # Configure the SDK
        genai.configure(api_key=api_key)
        
        # Prepare training data
        training_data_for_api = []
        for example in training_data:
            training_data_for_api.append({
                'text_input': example['messages'][1]['content'],  # User message
                'output': example['messages'][2]['content']       # Assistant message
            })
        
        # Create tuning job
        operation = genai.create_tuned_model(
            source_model=base_model,
            training_data=training_data_for_api,
            id=f"translation-model-{int(time.time())}",
            epoch_count=epoch_count,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        return operation, None
        
    except Exception as e:
        return None, str(e)

def get_tuning_job_status(job_name, api_key):
    """Get the status of a fine-tuning job."""
    if not GOOGLE_AI_AVAILABLE:
        return None, "Google AI SDK not available"
    
    try:
        genai.configure(api_key=api_key)
        # Get tuned model info
        model = genai.get_tuned_model(job_name)
        return model, None
    except Exception as e:
        return None, str(e)

def list_tuning_jobs(api_key):
    """List all fine-tuning jobs."""
    if not GOOGLE_AI_AVAILABLE:
        return [], "Google AI SDK not available"
    
    try:
        genai.configure(api_key=api_key)
        models = genai.list_tuned_models()
        return list(models), None
    except Exception as e:
        return [], str(e)

# --- OpenAI Fine-tuning Functions ---
def upload_training_file_openai(api_key, jsonl_content, filename="training_data.jsonl"):
    """Upload training file to OpenAI for fine-tuning."""
    if not OPENAI_AVAILABLE:
        return None, "OpenAI SDK not available"
    
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # Create a temporary file for upload
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(jsonl_content)
            temp_file_path = f.name
        
        # Upload the file
        with open(temp_file_path, 'rb') as f:
            response = client.files.create(
                file=f,
                purpose='fine-tune'
            )
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        return response, None
        
    except Exception as e:
        return None, str(e)

def start_openai_finetuning_job(api_key, training_file_id, model="gpt-4o-mini", 
                               n_epochs="auto", batch_size="auto", learning_rate_multiplier="auto"):
    """Start OpenAI fine-tuning job."""
    if not OPENAI_AVAILABLE:
        return None, "OpenAI SDK not available"
    
    try:
        client = openai.OpenAI(api_key=api_key)
        
        job = client.fine_tuning.jobs.create(
            training_file=training_file_id,
            model=model,
            hyperparameters={
                "n_epochs": n_epochs,
                "batch_size": batch_size,
                "learning_rate_multiplier": learning_rate_multiplier
            }
        )
        
        return job, None
        
    except Exception as e:
        return None, str(e)

def get_openai_finetuning_status(api_key, job_id):
    """Get the status of an OpenAI fine-tuning job."""
    if not OPENAI_AVAILABLE:
        return None, "OpenAI SDK not available"
    
    try:
        client = openai.OpenAI(api_key=api_key)
        job = client.fine_tuning.jobs.retrieve(job_id)
        return job, None
    except Exception as e:
        return None, str(e)

def list_openai_finetuning_jobs(api_key):
    """List all OpenAI fine-tuning jobs."""
    if not OPENAI_AVAILABLE:
        return [], "OpenAI SDK not available"
    
    try:
        client = openai.OpenAI(api_key=api_key)
        jobs = client.fine_tuning.jobs.list()
        return list(jobs.data), None
    except Exception as e:
        return [], str(e)

def list_openai_finetuned_models(api_key):
    """List user's fine-tuned OpenAI models."""
    if not OPENAI_AVAILABLE:
        return [], "OpenAI SDK not available"
    
    try:
        client = openai.OpenAI(api_key=api_key)
        # Get all fine-tuning jobs and extract completed models
        jobs = client.fine_tuning.jobs.list()
        
        completed_models = []
        for job in jobs.data:
            if job.status == "succeeded" and job.fine_tuned_model:
                completed_models.append({
                    "model_id": job.fine_tuned_model,
                    "base_model": job.model,
                    "job_id": job.id,
                    "created_at": job.created_at,
                    "finished_at": job.finished_at
                })
        
        return completed_models, None
    except Exception as e:
        return [], str(e)

def get_available_openai_models(api_key):
    """Get available OpenAI models from API."""
    if not OPENAI_AVAILABLE:
        return [], "OpenAI SDK not available"
    
    try:
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
                
            except Exception as deepseek_error:
                # Both APIs failed, return the original OpenAI error
                return [], str(openai_error)
    
    except Exception as e:
        return [], str(e)

def get_static_gemini_models():
    """Get static list of known Gemini models."""
    return [
        "gemini-2.5-flash",
        "gemini-2.5-pro", 
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-pro-001",
        "gemini-1.5-flash-001",
        "gemini-1.5-flash-002"
    ]

def get_available_models_for_translation(platform=None, api_key=None):
    """Get available models for translation based on platform."""
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
        import google.generativeai as genai
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

# --- Model Management ---
def save_model_metadata(job_info, hyperparams, dataset_info):
    """Save metadata about a fine-tuning job."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata_file = os.path.join(MODELS_DIR, f"model_metadata_{timestamp}.json")
    
    metadata = {
        "timestamp": timestamp,
        "job_info": job_info,
        "hyperparameters": hyperparams,
        "dataset_info": dataset_info,
        "created_at": datetime.now().isoformat()
    }
    
    try:
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return metadata_file
    except Exception as e:
        print(f"Warning: Could not save model metadata: {e}")
        return None

def load_model_metadata():
    """Load all saved model metadata."""
    metadata_files = []
    if os.path.exists(MODELS_DIR):
        for filename in os.listdir(MODELS_DIR):
            if filename.startswith("model_metadata_") and filename.endswith(".json"):
                filepath = os.path.join(MODELS_DIR, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                        metadata['filename'] = filename
                        metadata_files.append(metadata)
                except Exception as e:
                    print(f"Warning: Could not load {filename}: {e}")
    
    # Sort by timestamp (most recent first)
    return sorted(metadata_files, key=lambda x: x.get('timestamp', ''), reverse=True)

# --- Similarity Functions ---
@st.cache_resource
def load_semantic_model():
    """Load semantic similarity model with caching."""
    if not SEMANTIC_AVAILABLE:
        return None
    
    try:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        return model
    except Exception as e:
        print(f"Warning: Could not load semantic model: {e}")
        return None

def calculate_similarity(text1, text2, model=None, cache=None):
    """Calculate similarity between two texts."""
    if not text1 or not text2 or "File not found" in text1 or "File not found" in text2:
        return 0.0
    
    # Load cache if not provided
    if cache is None:
        cache = load_similarity_cache()
    
    # Check cache first
    hash1 = generate_text_hash(text1)
    hash2 = generate_text_hash(text2)
    cache_key1 = f"{hash1}:{hash2}"
    cache_key2 = f"{hash2}:{hash1}"
    
    if cache_key1 in cache:
        return cache[cache_key1]
    elif cache_key2 in cache:
        return cache[cache_key2]
    
    # Calculate similarity
    if SEMANTIC_AVAILABLE and model:
        try:
            # Truncate texts to avoid memory issues
            max_chars = 2000
            text1_truncated = text1[:max_chars]
            text2_truncated = text2[:max_chars]
            
            # Generate embeddings
            embeddings = model.encode([text1_truncated, text2_truncated], convert_to_tensor=True)
            
            # Calculate cosine similarity
            cosine_scores = torch.nn.functional.cosine_similarity(
                embeddings[0].unsqueeze(0), 
                embeddings[1].unsqueeze(0)
            )
            
            similarity = float(cosine_scores.item())
            similarity = max(0.0, min(1.0, similarity))
            
        except Exception as e:
            print(f"Warning: Semantic similarity calculation failed: {e}")
            similarity = calculate_syntactic_similarity_fallback(text1, text2)
    else:
        similarity = calculate_syntactic_similarity_fallback(text1, text2)
    
    # Store in cache
    cache[cache_key1] = similarity
    save_similarity_cache(cache)
    
    return similarity

def calculate_syntactic_similarity_fallback(text1, text2):
    """Fallback syntactic similarity for when semantic models aren't available."""
    # Length similarity
    len1, len2 = len(text1), len(text2)
    length_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
    
    # Content similarity (first 1000 chars for speed)
    sample1 = text1[:1000].lower().replace('\n', ' ')
    sample2 = text2[:1000].lower().replace('\n', ' ')
    content_similarity = SequenceMatcher(None, sample1, sample2).ratio()
    
    # Combined score
    return (length_ratio * 0.3) + (content_similarity * 0.7)

def get_chunking_statistics(training_data):
    """
    Analyze chunking statistics for the training data.
    """
    stats = {
        'total_examples': len(training_data),
        'chunk_sizes': [],
        'chunked_chapters': 0,
        'single_chunks': 0,
        'max_chunk_size': 0,
        'avg_chunk_size': 0,
        'over_5k_chars': 0
    }
    
    chapter_chunks = {}
    
    for example in training_data:
        chunk_size = len(example['messages'][2]['content'])
        stats['chunk_sizes'].append(chunk_size)
        
        if chunk_size > 5000:
            stats['over_5k_chars'] += 1
        
        # Track chunks per chapter
        metadata = example.get('metadata', {})
        chapter = metadata.get('chapter', 'unknown')
        
        if chapter not in chapter_chunks:
            chapter_chunks[chapter] = 0
        chapter_chunks[chapter] += 1
    
    # Calculate aggregates
    if stats['chunk_sizes']:
        stats['max_chunk_size'] = max(stats['chunk_sizes'])
        stats['avg_chunk_size'] = sum(stats['chunk_sizes']) / len(stats['chunk_sizes'])
    
    # Count chunked vs single-chunk chapters
    for chapter, chunk_count in chapter_chunks.items():
        if chunk_count > 1:
            stats['chunked_chapters'] += 1
        else:
            stats['single_chunks'] += 1
    
    return stats

# --- Evaluation Functions ---
def calculate_bleu_score(reference, candidate):
    """Calculate BLEU score between reference and candidate translations."""
    try:
        from nltk.translate.bleu_score import sentence_bleu
        from nltk.tokenize import word_tokenize
        
        # Tokenize
        reference_tokens = [word_tokenize(reference.lower())]
        candidate_tokens = word_tokenize(candidate.lower())
        
        # Calculate BLEU score
        score = sentence_bleu(reference_tokens, candidate_tokens)
        return score
    except ImportError:
        # Fallback to simple similarity if NLTK not available
        return calculate_syntactic_similarity_fallback(reference, candidate)

def evaluate_translation_quality(raw_text, reference_translation, candidate_translation, model=None):
    """Comprehensive evaluation of translation quality."""
    results = {}
    
    # BLEU score
    results['bleu_score'] = calculate_bleu_score(reference_translation, candidate_translation)
    
    # Semantic similarity
    results['semantic_similarity'] = calculate_similarity(reference_translation, candidate_translation, model)
    
    # Length comparison
    ref_stats = get_text_stats(reference_translation, 'english')
    cand_stats = get_text_stats(candidate_translation, 'english')
    results['length_ratio'] = cand_stats['word_count'] / ref_stats['word_count'] if ref_stats['word_count'] > 0 else 0
    
    # Raw content stats for context
    raw_stats = get_text_stats(raw_text, 'chinese')
    results['raw_stats'] = raw_stats
    results['reference_stats'] = ref_stats
    results['candidate_stats'] = cand_stats
    
    return results

# --- AI Source Management ---
def get_available_ai_sources(current_chapter=None):
    """Get list of available AI translation sources.
    
    Args:
        current_chapter: If provided, only include custom runs that have this chapter available
    
    Returns:
        list: Available AI translation sources
    """
    sources = ["Fresh Gemini Translation", "Cached Gemini Translation"]
    
    # Add custom translation runs
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
                        sources.append(f"Custom: {run_name}")
                else:
                    # Check if this run has any translation files (original behavior)
                    translation_files = [f for f in os.listdir(run_path) if f.endswith('-translated.txt')]
                    if translation_files:
                        sources.append(f"Custom: {run_name}")
    
    return sources

def get_ai_translation_content(selected_ai_source, selected_chapter, session_state_ai_translation="", raw_content=""):
    """Get AI translation content based on selected source and chapter.
    
    Args:
        selected_ai_source: The selected AI source from the dropdown
        selected_chapter: The current chapter number
        session_state_ai_translation: Current session state AI translation
        raw_content: Raw Chinese content for fresh translation
    
    Returns:
        str: AI translation content or empty string if not available
    """
    if selected_ai_source.startswith("Custom: "):
        # Load from custom translation run
        run_name = selected_ai_source[8:]  # Remove "Custom: " prefix
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

# --- Custom Prompt Management ---
def load_custom_prompts():
    """Load custom prompt templates from JSON file.
    
    Returns:
        dict: Dictionary of custom prompts {name: {"content": str, "created": str}}
    """
    custom_prompts_file = os.path.join(DATA_DIR, "custom_prompts.json")
    
    if os.path.exists(custom_prompts_file):
        try:
            with open(custom_prompts_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}

def save_custom_prompt(name, content):
    """Save a custom prompt template.
    
    Args:
        name: Name of the prompt template
        content: The prompt content
    
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        custom_prompts = load_custom_prompts()
        
        # Add timestamp
        from datetime import datetime
        custom_prompts[name] = {
            "content": content.strip(),
            "created": datetime.now().isoformat(),
            "category": "user"
        }
        
        # Save back to file
        custom_prompts_file = os.path.join(DATA_DIR, "custom_prompts.json")
        with open(custom_prompts_file, 'w', encoding='utf-8') as f:
            json.dump(custom_prompts, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Error saving custom prompt: {e}")
        return False

def delete_custom_prompt(name):
    """Delete a custom prompt template.
    
    Args:
        name: Name of the prompt template to delete
    
    Returns:
        bool: True if deleted successfully, False otherwise
    """
    try:
        custom_prompts = load_custom_prompts()
        
        if name in custom_prompts:
            del custom_prompts[name]
            
            # Save back to file
            custom_prompts_file = os.path.join(DATA_DIR, "custom_prompts.json")
            with open(custom_prompts_file, 'w', encoding='utf-8') as f:
                json.dump(custom_prompts, f, indent=2, ensure_ascii=False)
            
            return True
        return False
    except Exception as e:
        print(f"Error deleting custom prompt: {e}")
        return False

def get_all_available_prompts():
    """Get all available prompt templates (built-in + custom).
    
    Returns:
        dict: Combined dictionary of all prompts
    """
    # Built-in prompt templates
    builtin_prompts = {
        "Literal & Accurate": "You are a professional translator specializing in Chinese to English translation of web novels. Provide an accurate, literal translation that preserves the original meaning, structure, and cultural context. Maintain formal tone and precise terminology.",
        "Dynamic & Modern": "You are a skilled literary translator adapting Chinese web novels for Western readers. Create a flowing, engaging translation that captures the spirit and excitement of the original while using natural modern English. Prioritize readability and dramatic impact.",
        "Simplified & Clear": "You are translating Chinese web novels for young adult readers. Use simple, clear language that's easy to understand. Explain cultural concepts briefly when needed. Keep sentences shorter and vocabulary accessible.",
    }
    
    # Load custom prompts
    custom_prompts = load_custom_prompts()
    
    # Combine prompts with prefixes for organization
    all_prompts = {}
    
    # Add built-in prompts
    for name, content in builtin_prompts.items():
        all_prompts[name] = content
    
    # Add custom prompts with prefix
    for name, prompt_data in custom_prompts.items():
        all_prompts[f"🎨 {name}"] = prompt_data["content"]
    
    # Add Custom option for creating new prompts
    all_prompts["Custom"] = ""
    
    return all_prompts

# --- Alignment Map Management ---
def save_alignment_map_safely(alignment_map, output_file="alignment_map.json"):
    """Save alignment map with backup of existing file in organized backup directory.
    
    Args:
        alignment_map: The alignment map dictionary to save
        output_file: Path to the output file (should remain in root for app access)
    
    Returns:
        str: Backup filename if created, None if no backup was needed
    """
    # Ensure backup directory exists
    backup_dir = os.path.join(DATA_DIR, "backups")
    os.makedirs(backup_dir, exist_ok=True)
    
    backup_filename = None
    
    # Create backup if file exists
    if os.path.exists(output_file):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{os.path.basename(output_file)}.backup_{timestamp}"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # Copy to backup directory
        import shutil
        shutil.copy(output_file, backup_path)
    
    # Save new alignment map to root (where app expects it)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(alignment_map, f, indent=2, ensure_ascii=False)
    
    return backup_filename

# --- Export Functions ---
def export_training_data_to_jsonl(training_data, output_path):
    """Export training data to JSONL format."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in training_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        return True, f"Exported {len(training_data)} examples to {output_path}"
    except Exception as e:
        return False, f"Export failed: {e}"

def create_translation_jsonl(training_examples, train_split=0.8, format_type="OpenAI Fine-tuning", system_prompt=None):
    """
    Create JSONL training files for translation fine-tuning.
    
    Args:
        training_examples: List of training example dictionaries
        train_split: Fraction of data to use for training (rest for validation)
        format_type: Format for JSONL ("OpenAI Fine-tuning", "Gemini Fine-tuning", "Custom Messages")
        system_prompt: Optional system prompt for translation task
    
    Returns:
        tuple: (train_jsonl_content, val_jsonl_content, stats_dict)
    """
    import random
    import json
    
    # Shuffle the data for better train/val split
    examples = training_examples.copy()
    random.shuffle(examples)
    
    # Split into train and validation
    split_idx = int(len(examples) * train_split)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    def format_example(example, format_type, system_prompt=None):
        """Format a single example according to the specified format."""
        chinese_text = example['raw_content']
        english_text = example['english_content']
        
        if format_type == "OpenAI Fine-tuning":
            # Standard OpenAI format: messages array with user/assistant roles
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.extend([
                {"role": "user", "content": chinese_text},
                {"role": "assistant", "content": english_text}
            ])
            
            return {"messages": messages}
        
        elif format_type == "Gemini Fine-tuning":
            # Gemini format: input_text and output_text
            formatted = {
                "input_text": chinese_text,
                "output_text": english_text
            }
            
            if system_prompt:
                formatted["input_text"] = f"{system_prompt}\n\nTranslate the following Chinese text to English:\n\n{chinese_text}"
            
            return formatted
        
        elif format_type == "Custom Messages":
            # Custom format similar to your sample but for translation
            return {
                "messages": [
                    {"role": "user", "content": f"Translate this Chinese text to English: {chinese_text}"},
                    {"role": "assistant", "content": english_text}
                ]
            }
        
        else:
            # Default to OpenAI format
            return format_example(example, "OpenAI Fine-tuning", system_prompt)
    
    # Convert examples to JSONL format
    train_jsonl_lines = []
    val_jsonl_lines = []
    
    for example in train_examples:
        formatted = format_example(example, format_type, system_prompt)
        train_jsonl_lines.append(json.dumps(formatted, ensure_ascii=False))
    
    for example in val_examples:
        formatted = format_example(example, format_type, system_prompt)
        val_jsonl_lines.append(json.dumps(formatted, ensure_ascii=False))
    
    # Join lines with newlines
    train_jsonl_content = '\n'.join(train_jsonl_lines)
    val_jsonl_content = '\n'.join(val_jsonl_lines)
    
    # Create statistics
    stats = {
        'total_count': len(examples),
        'train_count': len(train_examples),
        'val_count': len(val_examples),
        'train_split': train_split,
        'format_type': format_type,
        'has_system_prompt': system_prompt is not None
    }
    
    return train_jsonl_content, val_jsonl_content, stats

# === STYLE EVALUATION SYSTEM ===

def get_available_translation_styles():
    """
    Scan the custom_translations directory and return available translation styles.
    
    Returns:
        list: List of style dictionaries with metadata
    """
    styles = []
    
    if not os.path.exists(TRANSLATIONS_DIR):
        return styles
    
    for style_name in os.listdir(TRANSLATIONS_DIR):
        style_path = os.path.join(TRANSLATIONS_DIR, style_name)
        
        if not os.path.isdir(style_path):
            continue
        
        # Count translation files
        translation_files = [f for f in os.listdir(style_path) if f.endswith('-translated.txt')]
        
        if not translation_files:
            continue
        
        # Load metadata if available
        metadata_file = os.path.join(style_path, 'job_metadata.json')
        metadata = {}
        
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        # Extract chapter numbers from files
        chapter_numbers = []
        for filename in translation_files:
            try:
                chapter_num = int(filename.split('-')[1])
                chapter_numbers.append(chapter_num)
            except (ValueError, IndexError):
                continue
        
        chapter_numbers.sort()
        
        styles.append({
            'name': style_name,
            'path': style_path,
            'chapter_count': len(chapter_numbers),
            'chapter_range': f"{min(chapter_numbers)}-{max(chapter_numbers)}" if chapter_numbers else "N/A",
            'metadata': metadata,
            'model_name': metadata.get('model_name', 'Unknown'),
            'system_prompt': metadata.get('system_prompt', 'No prompt available'),
            'timestamp': metadata.get('timestamp', 'Unknown')
        })
    
    # Sort by timestamp (newest first)
    styles.sort(key=lambda x: x['timestamp'], reverse=True)
    return styles

def calculate_bert_scores_for_style(style_info, alignment_map, progress_callback=None):
    """
    Calculate BERT similarity scores for all chapters in a translation style.
    
    Args:
        style_info: Style dictionary from get_available_translation_styles()
        alignment_map: Chapter alignment mapping
        progress_callback: Optional function to call with progress updates (progress, total, chapter_num)
    
    Returns:
        dict: Chapter numbers mapped to BERT scores
    """
    # Load semantic model
    semantic_model = load_semantic_model()
    if not semantic_model:
        return {}
    
    bert_scores = {}
    style_path = style_info['path']
    
    # Get all translation files
    translation_files = [f for f in os.listdir(style_path) if f.endswith('-translated.txt')]
    total_files = len(translation_files)
    
    for i, filename in enumerate(sorted(translation_files)):
        try:
            # Extract chapter number
            chapter_num = int(filename.split('-')[1])
            
            # Load custom translation
            custom_path = os.path.join(style_path, filename)
            with open(custom_path, 'r', encoding='utf-8') as f:
                custom_translation = f.read().strip()
            
            # Load official translation from alignment map
            if str(chapter_num) in alignment_map:
                official_file = alignment_map[str(chapter_num)].get('english_file')
                if official_file and os.path.exists(official_file):
                    official_translation = load_chapter_content(official_file)
                    
                    if "File not found" not in official_translation:
                        # Calculate BERT similarity
                        similarity = calculate_similarity(
                            official_translation, 
                            custom_translation, 
                            semantic_model
                        )
                        bert_scores[chapter_num] = similarity
            
            # Call progress callback if provided
            if progress_callback:
                progress_callback(i + 1, total_files, chapter_num)
                
        except (ValueError, IndexError, IOError) as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    return bert_scores

def save_bert_scores(style_name, bert_scores):
    """Save BERT scores to persistent storage."""
    style_eval_dir = os.path.join(EVALUATIONS_DIR, style_name)
    os.makedirs(style_eval_dir, exist_ok=True)
    
    bert_file = os.path.join(style_eval_dir, 'bert_scores.json')
    with open(bert_file, 'w', encoding='utf-8') as f:
        json.dump(bert_scores, f, indent=2, ensure_ascii=False)

def load_bert_scores(style_name):
    """Load BERT scores from persistent storage."""
    bert_file = os.path.join(EVALUATIONS_DIR, style_name, 'bert_scores.json')
    if os.path.exists(bert_file):
        try:
            with open(bert_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}

def save_human_scores(style_name, human_scores):
    """Save human evaluation scores to persistent storage."""
    style_eval_dir = os.path.join(EVALUATIONS_DIR, style_name)
    os.makedirs(style_eval_dir, exist_ok=True)
    
    human_file = os.path.join(style_eval_dir, 'human_scores.json')
    with open(human_file, 'w', encoding='utf-8') as f:
        json.dump(human_scores, f, indent=2, ensure_ascii=False)

def load_human_scores(style_name):
    """Load human evaluation scores from persistent storage."""
    human_file = os.path.join(EVALUATIONS_DIR, style_name, 'human_scores.json')
    if os.path.exists(human_file):
        try:
            with open(human_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}

def calculate_composite_score(bert_scores, human_scores, chapter_count):
    """
    Calculate comprehensive composite score for a translation style.
    
    Args:
        bert_scores: Dict of chapter_num -> bert_score
        human_scores: Dict of chapter_num -> human_scores_dict
        chapter_count: Total number of chapters in the style
    
    Returns:
        dict: Comprehensive scoring breakdown
    """
    import numpy as np
    
    if not bert_scores and not human_scores:
        return {
            'composite_score': 0,
            'quality_score': 0,
            'consistency_bonus': 0,
            'completeness_bonus': 0,
            'evaluated_chapters': 0
        }
    
    # Get chapters that have both BERT and human scores (if available)
    evaluated_chapters = set()
    
    if bert_scores:
        evaluated_chapters.update(bert_scores.keys())
    
    if human_scores:
        evaluated_chapters.update(human_scores.keys())
    
    evaluated_chapters = list(evaluated_chapters)
    
    if not evaluated_chapters:
        return {
            'composite_score': 0,
            'quality_score': 0,
            'consistency_bonus': 0,
            'completeness_bonus': 0,
            'evaluated_chapters': 0
        }
    
    # Calculate quality score (weighted average of all dimensions)
    quality_scores = []
    
    for chapter_num in evaluated_chapters:
        chapter_quality = 0
        total_weight = 0
        
        # BERT similarity (50% weight when available)
        if chapter_num in bert_scores:
            chapter_quality += bert_scores[chapter_num] * 0.5
            total_weight += 0.5
        
        # Human scores (50% weight total, divided among dimensions)
        if chapter_num in human_scores:
            human_data = human_scores[chapter_num]
            # New evaluation dimensions (v2.0)
            human_dimensions = [
                'vocabulary_complexity',
                'cultural_context', 
                'prose_style',
                'creative_fidelity'
            ]
            
            # Legacy dimensions for backwards compatibility (v1.0)
            legacy_dimensions = [
                'english_sophistication',
                'world_building', 
                'emotional_impact',
                'dialogue_naturalness'
            ]
            
            human_avg = 0
            valid_dimensions = 0
            
            # Try new dimensions first, fall back to legacy if not found
            dimensions_to_check = human_dimensions
            if not any(dim in human_data for dim in human_dimensions):
                # Use legacy dimensions if no new dimensions found
                dimensions_to_check = legacy_dimensions
            
            for dim in dimensions_to_check:
                if dim in human_data:
                    human_avg += human_data[dim] / 100  # Convert to 0-1 scale
                    valid_dimensions += 1
            
            if valid_dimensions > 0:
                human_avg /= valid_dimensions
                chapter_quality += human_avg * 0.5
                total_weight += 0.5
        
        # Normalize by total weight
        if total_weight > 0:
            quality_scores.append(chapter_quality / total_weight)
    
    if not quality_scores:
        return {
            'composite_score': 0,
            'quality_score': 0,
            'consistency_bonus': 0,
            'completeness_bonus': 0,
            'evaluated_chapters': 0
        }
    
    # Calculate mean quality score
    mean_quality = np.mean(quality_scores)
    
    # Calculate consistency bonus (1 - standard deviation)
    std_quality = np.std(quality_scores)
    consistency_bonus = max(0, 1 - std_quality)
    
    # Calculate completeness bonus (logarithmic scale)
    completeness_bonus = np.log10(len(evaluated_chapters) + 1)
    
    # Final composite score
    composite_score = (mean_quality * consistency_bonus) * completeness_bonus
    
    return {
        'composite_score': composite_score,
        'quality_score': mean_quality,
        'consistency_bonus': consistency_bonus,
        'completeness_bonus': completeness_bonus,
        'evaluated_chapters': len(evaluated_chapters),
        'mean_bert': np.mean([bert_scores[ch] for ch in evaluated_chapters if ch in bert_scores]) if bert_scores else 0,
        'std_bert': np.std([bert_scores[ch] for ch in evaluated_chapters if ch in bert_scores]) if bert_scores else 0
    }

# === WEB SCRAPING SYSTEM ===

def validate_scraping_url(url):
    """
    Validate if a URL is suitable for scraping and identify the novel site.
    
    Args:
        url: URL to validate
        
    Returns:
        dict: Validation result with site info and recommendations
    """
    import urllib.parse as urlparse
    
    result = {
        'valid': False,
        'site_type': 'unknown',
        'recommendations': [],
        'warnings': []
    }
    
    try:
        parsed = urlparse.urlparse(url)
        domain = parsed.netloc.lower()
        
        # Check for supported sites
        if 'dxmwx.org' in domain:
            result['valid'] = True
            result['site_type'] = 'dxmwx'
            result['recommendations'].append("✅ dxmwx.org is fully supported")
            
            # Check URL format
            if '/read/' in url and '_' in url:
                result['recommendations'].append("✅ URL format looks correct for chapter scraping")
            else:
                result['warnings'].append("⚠️ URL should be a chapter page (contains /read/ and chapter ID)")
        
        elif 'wtr-lab.com' in domain:
            result['site_type'] = 'wtr-lab'
            result['warnings'].append("⚠️ wtr-lab.com support is experimental")
            result['recommendations'].append("📝 May require custom scraping configuration")
        
        else:
            result['warnings'].append(f"⚠️ Unknown domain: {domain}")
            result['recommendations'].append("🔧 Custom scraper development may be needed")
            result['recommendations'].append("📖 Consider using supported sites: dxmwx.org")
        
        # General URL checks
        if not url.startswith(('http://', 'https://')):
            result['warnings'].append("⚠️ URL should start with http:// or https://")
        
        if parsed.scheme == 'http':
            result['warnings'].append("🔒 Consider using HTTPS for better security")
            
    except Exception as e:
        result['warnings'].append(f"❌ URL parsing error: {str(e)}")
    
    return result

def streamlit_scraper(start_url, output_dir, max_chapters=50, delay_seconds=2, progress_callback=None, status_callback=None):
    """
    Non-interactive version of the robust scraper for Streamlit integration.
    
    Args:
        start_url: URL to start scraping from
        output_dir: Directory to save chapters
        max_chapters: Maximum number of chapters to scrape
        delay_seconds: Delay between requests
        progress_callback: Function to call with progress updates (current, total)
        status_callback: Function to call with status updates (message)
        
    Returns:
        dict: Scraping results with statistics
    """
    import requests
    from bs4 import BeautifulSoup
    import time
    import re
    
    # Import helper functions from robust_scraper
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
    
    try:
        from robust_scraper import extract_chapter_number, sanitize_filename
    except ImportError:
        # Fallback simple implementations
        def extract_chapter_number(title):
            match = re.search(r'第?(\d+)章', title)
            return int(match.group(1)) if match else None
            
        def sanitize_filename(filename):
            return re.sub(r'[\\/*?:"<>|]', "", filename).strip()
    
    results = {
        'success': False,
        'chapters_scraped': 0,
        'errors': [],
        'chapters': [],
        'total_time': 0
    }
    
    start_time = time.time()
    
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        current_url = start_url
        chapters_scraped = 0
        
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        if status_callback:
            status_callback(f"🌐 Starting scraper from: {start_url}")
        
        for i in range(max_chapters):
            if not current_url:
                break
                
            if status_callback:
                status_callback(f"📖 Processing chapter {i+1}/{max_chapters}")
                
            try:
                # Fetch page
                response = session.get(current_url, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract title and content (dxmwx.org specific)
                title_element = soup.find('h1')
                title = title_element.get_text().strip() if title_element else f"Chapter {i+1}"
                
                content_element = soup.find('div', {'id': 'content'}) or soup.find('div', class_='content')
                if not content_element:
                    results['errors'].append(f"No content found at {current_url}")
                    break
                
                content = content_element.get_text().strip()
                
                if not content:
                    results['errors'].append(f"Empty content at {current_url}")
                    break
                
                # Extract chapter number
                chapter_num = extract_chapter_number(title)
                if not chapter_num:
                    chapter_num = 9999 - i  # Fallback numbering
                
                # Save chapter
                safe_title = sanitize_filename(title)
                filename = f"Chapter-{chapter_num:04d}-{safe_title}.txt"
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"Title: {title}\n")
                    f.write(f"URL: {current_url}\n")
                    f.write(f"Chapter: {chapter_num}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(content)
                
                chapters_scraped += 1
                results['chapters'].append({
                    'number': chapter_num,
                    'title': title,
                    'filename': filename,
                    'url': current_url
                })
                
                if progress_callback:
                    progress_callback(chapters_scraped, max_chapters)
                
                if status_callback:
                    status_callback(f"✅ Saved: Chapter {chapter_num} - {title[:50]}...")
                
                # Find next/previous chapter link
                next_link = None
                
                # Look for navigation links (dxmwx.org specific)
                nav_links = soup.find_all('a', href=True)
                for link in nav_links:
                    href = link.get('href', '')
                    text = link.get_text().strip().lower()
                    
                    if any(word in text for word in ['上一页', '上一章', 'previous']):
                        next_link = href
                        break
                
                if next_link:
                    if next_link.startswith('/'):
                        from urllib.parse import urljoin
                        current_url = urljoin(current_url, next_link)
                    else:
                        current_url = next_link
                else:
                    if status_callback:
                        status_callback("🔍 No more navigation links found")
                    break
                
                # Delay between requests
                if delay_seconds > 0:
                    time.sleep(delay_seconds)
                
            except Exception as e:
                error_msg = f"Error processing chapter {i+1}: {str(e)}"
                results['errors'].append(error_msg)
                if status_callback:
                    status_callback(f"❌ {error_msg}")
                break
        
        results['success'] = chapters_scraped > 0
        results['chapters_scraped'] = chapters_scraped
        results['total_time'] = time.time() - start_time
        
        if status_callback:
            status_callback(f"🎉 Scraping completed! {chapters_scraped} chapters saved in {results['total_time']:.1f}s")
            
    except Exception as e:
        results['errors'].append(f"Fatal scraping error: {str(e)}")
        if status_callback:
            status_callback(f"💥 Fatal error: {str(e)}")
    
    return results

def find_first_misalignment_binary_search(alignment_map, api_key, min_chapter, max_chapter, threshold):
    """
    Use binary search to find the first misaligned chapter based on semantic similarity.
    
    Args:
        alignment_map: Chapter alignment mapping
        api_key: API key for translation/similarity calculation
        min_chapter: Start of search range
        max_chapter: End of search range  
        threshold: Minimum similarity score to consider aligned (0.0-1.0)
        
    Returns:
        dict: Search results with first misaligned chapter and search log
    """
    result = {
        'success': False,
        'first_misaligned_chapter': None,
        'total_chapters_checked': 0,
        'threshold_used': threshold,
        'search_log': [],
        'error': None
    }
    
    try:
        # Load semantic model for similarity calculation
        semantic_model = load_semantic_model()
        if not semantic_model:
            result['error'] = "Semantic similarity model not available"
            return result
        
        chapters_checked = 0
        search_log = []
        
        # Binary search implementation
        left = min_chapter
        right = max_chapter
        first_misaligned = None
        
        while left <= right:
            mid = (left + right) // 2
            chapters_checked += 1
            
            # Check if this chapter exists in alignment map
            if str(mid) not in alignment_map:
                search_log.append({
                    'chapter': mid,
                    'action': f'Chapter {mid} not in alignment map',
                    'search_range': f'[{left}, {right}]'
                })
                left = mid + 1
                continue
            
            chapter_data = alignment_map[str(mid)]
            
            # Load chapter content
            try:
                raw_content = load_chapter_content(chapter_data.get('raw_file', ''))
                english_content = load_chapter_content(chapter_data.get('english_file', ''))
                
                if "File not found" in raw_content or "File not found" in english_content:
                    search_log.append({
                        'chapter': mid,
                        'action': f'Files missing for chapter {mid}',
                        'search_range': f'[{left}, {right}]'
                    })
                    left = mid + 1
                    continue
                
                # Calculate semantic similarity
                similarity_score = calculate_similarity(english_content, raw_content, semantic_model)
                
                search_log.append({
                    'chapter': mid,
                    'similarity_score': similarity_score,
                    'action': f'Similarity: {similarity_score:.3f} (threshold: {threshold:.3f})',
                    'search_range': f'[{left}, {right}]'
                })
                
                # Check if this chapter is misaligned (below threshold)
                if similarity_score < threshold:
                    # This chapter is misaligned, search in left half for earlier misalignment
                    first_misaligned = mid
                    right = mid - 1
                    search_log.append({
                        'chapter': mid,
                        'action': f'Misaligned (score {similarity_score:.3f} < {threshold:.3f}), searching left half',
                        'search_range': f'[{left}, {right}]'
                    })
                else:
                    # This chapter is aligned, search in right half
                    left = mid + 1
                    search_log.append({
                        'chapter': mid,
                        'action': f'Aligned (score {similarity_score:.3f} >= {threshold:.3f}), searching right half',
                        'search_range': f'[{left}, {right}]'
                    })
                    
            except Exception as e:
                search_log.append({
                    'chapter': mid,
                    'action': f'Error processing chapter {mid}: {str(e)}',
                    'search_range': f'[{left}, {right}]'
                })
                left = mid + 1
                continue
        
        result['success'] = True
        result['first_misaligned_chapter'] = first_misaligned
        result['total_chapters_checked'] = chapters_checked
        result['search_log'] = search_log
        
        # Add summary to search log
        if first_misaligned:
            search_log.append({
                'chapter': 'RESULT',
                'action': f'First misaligned chapter found: {first_misaligned}',
                'search_range': f'Total checked: {chapters_checked}'
            })
        else:
            search_log.append({
                'chapter': 'RESULT', 
                'action': f'No misaligned chapters found in range {min_chapter}-{max_chapter}',
                'search_range': f'Total checked: {chapters_checked}'
            })
            
    except Exception as e:
        result['error'] = f"Binary search failed: {str(e)}"
    
    return result

def preview_systematic_correction(alignment_map, offset, sample_size=10):
    """
    Preview the effects of applying a systematic offset correction to the alignment map.
    
    Args:
        alignment_map: Current chapter alignment mapping
        offset: Offset to apply (positive or negative integer)
        sample_size: Number of sample chapters to include in preview
        
    Returns:
        dict: Preview data with before/after comparisons
    """
    preview_data = {
        'offset': offset,
        'sample_size': sample_size,
        'samples': [],
        'total_affected_chapters': 0,
        'success': False
    }
    
    try:
        chapter_numbers = sorted([int(k) for k in alignment_map.keys()])
        affected_chapters = []
        
        # Find chapters that would be affected by the offset
        for chapter_num in chapter_numbers:
            new_chapter_num = chapter_num + offset
            
            # Check if the new chapter number would be valid
            if new_chapter_num > 0 and str(new_chapter_num) in alignment_map:
                affected_chapters.append(chapter_num)
        
        preview_data['total_affected_chapters'] = len(affected_chapters)
        
        # Create sample data
        sample_chapters = affected_chapters[:sample_size]
        
        for chapter_num in sample_chapters:
            new_chapter_num = chapter_num + offset
            
            # Get current alignment
            current_data = alignment_map[str(chapter_num)]
            
            # Get what the new alignment would be
            new_data = alignment_map.get(str(new_chapter_num), {})
            
            sample_info = {
                'original_chapter': chapter_num,
                'new_chapter': new_chapter_num,
                'current_chinese_file': current_data.get('raw_file', 'N/A'),
                'current_english_file': current_data.get('english_file', 'N/A'),
                'new_chinese_file': new_data.get('raw_file', 'N/A'),
                'new_english_file': new_data.get('english_file', 'N/A'),
                'would_change': current_data.get('raw_file') != new_data.get('raw_file')
            }
            
            # Try to load content for comparison if available
            try:
                if current_data.get('raw_file'):
                    current_chinese = load_chapter_content(current_data['raw_file'])
                    if "File not found" not in current_chinese:
                        sample_info['current_chinese_preview'] = current_chinese[:200] + "..."
                
                if new_data.get('raw_file'):
                    new_chinese = load_chapter_content(new_data['raw_file'])
                    if "File not found" not in new_chinese:
                        sample_info['new_chinese_preview'] = new_chinese[:200] + "..."
                        
            except Exception as e:
                sample_info['content_error'] = str(e)
            
            preview_data['samples'].append(sample_info)
        
        preview_data['success'] = True
        
    except Exception as e:
        preview_data['error'] = f"Preview generation failed: {str(e)}"
    
    return preview_data

# --- Inline Comment Management ---
def save_inline_comments(style_name, chapter_id, comments):
    """Save inline comments for a specific chapter and style."""
    import json
    
    print("🔍 DEBUG: save_inline_comments function called")
    print("🔍 DEBUG: style_name:", style_name)
    print("🔍 DEBUG: chapter_id:", chapter_id)
    print("🔍 DEBUG: comments to save:", len(comments), "comments")
    
    try:
        style_eval_dir = os.path.join(EVALUATIONS_DIR, style_name)
        print("🔍 DEBUG: style_eval_dir:", style_eval_dir)
        print("🔍 DEBUG: EVALUATIONS_DIR:", EVALUATIONS_DIR)
        
        os.makedirs(style_eval_dir, exist_ok=True)
        print("🔍 DEBUG: Directory created/confirmed")
        print("🔍 DEBUG: Directory exists:", os.path.exists(style_eval_dir))
        
        comments_file = os.path.join(style_eval_dir, f'inline_comments_ch{chapter_id}.json')
        print("🔍 DEBUG: comments_file path:", comments_file)
        
        with open(comments_file, 'w', encoding='utf-8') as f:
            json.dump(comments, f, indent=2, ensure_ascii=False)
        
        print("🔍 DEBUG: File written successfully")
        print("🔍 DEBUG: File exists after write:", os.path.exists(comments_file))
        
        if os.path.exists(comments_file):
            file_size = os.path.getsize(comments_file)
            print("🔍 DEBUG: File size:", file_size, "bytes")
            
    except Exception as e:
        print(f"❌ DEBUG: Exception in save_inline_comments: {str(e)}")
        print("🔍 DEBUG: Exception type:", type(e).__name__)
        import traceback
        print("🔍 DEBUG: Full traceback:", traceback.format_exc())

def load_inline_comments(style_name, chapter_id):
    """Load inline comments for a specific chapter and style."""
    import json
    comments_file = os.path.join(EVALUATIONS_DIR, style_name, f'inline_comments_ch{chapter_id}.json')
    if os.path.exists(comments_file):
        try:
            with open(comments_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return []

def add_inline_comment(style_name, chapter_id, comment_data):
    """Add a new inline comment to existing comments."""
    from datetime import datetime
    
    print("🔍 DEBUG: add_inline_comment function called")
    print("🔍 DEBUG: style_name:", style_name)
    print("🔍 DEBUG: chapter_id:", chapter_id)
    print("🔍 DEBUG: comment_data:", comment_data)
    
    try:
        # Load existing comments
        print("🔍 DEBUG: Loading existing comments")
        comments = load_inline_comments(style_name, chapter_id)
        print("🔍 DEBUG: Existing comments loaded:", len(comments), "comments")
        
        # Add new comment with metadata
        print("🔍 DEBUG: Creating new comment")
        new_comment = {
            'id': f"comment_{len(comments) + 1}_{int(datetime.now().timestamp())}",
            'timestamp': datetime.now().isoformat(),
            **comment_data
        }
        print("🔍 DEBUG: New comment created:", new_comment)
        
        comments.append(new_comment)
        print("🔍 DEBUG: Comment appended to list, total comments:", len(comments))
        
        print("🔍 DEBUG: Calling save_inline_comments")
        save_inline_comments(style_name, chapter_id, comments)
        print("🔍 DEBUG: save_inline_comments completed")
        
        print("🔍 DEBUG: Returning comment ID:", new_comment['id'])
        return new_comment['id']
        
    except Exception as e:
        print(f"❌ DEBUG: Exception in add_inline_comment: {str(e)}")
        print("🔍 DEBUG: Exception type:", type(e).__name__)
        import traceback
        print("🔍 DEBUG: Full traceback:", traceback.format_exc())
        return None

def apply_comment_highlighting(text, comments):
    """Apply HTML highlighting to text based on inline comments."""
    if not comments:
        return text
    
    # Sort comments by start position (descending) to avoid offset issues
    sorted_comments = sorted(comments, key=lambda c: c['start_offset'], reverse=True)
    
    # Dimension color mapping (new v2.0 dimensions + legacy support)
    dimension_colors = {
        # New dimensions (v2.0)
        'vocabulary_complexity': '#fff3cd',   # Light yellow - 🧠
        'cultural_context': '#cff4fc',        # Light blue - 🌏
        'prose_style': '#f8d7da',             # Light red - ✍️
        'creative_fidelity': '#d1e7dd',       # Light green - 🎨
        # Legacy dimensions (v1.0) - same colors for backwards compatibility
        'english_sophistication': '#fff3cd',  # Light yellow
        'world_building': '#cff4fc',          # Light blue
        'emotional_impact': '#f8d7da',        # Light red
        'dialogue_naturalness': '#d1e7dd'     # Light green
    }
    
    highlighted_text = text
    for comment in sorted_comments:
        start = comment['start_offset']
        end = comment['end_offset']
        dimension = comment.get('dimension', 'vocabulary_complexity')
        color = dimension_colors.get(dimension, '#f0f0f0')
        
        # Create highlighted span with tooltip
        original_segment = highlighted_text[start:end]
        highlighted_segment = f'''<span class="inline-comment" 
            style="background-color: {color}; cursor: pointer; border-radius: 3px; padding: 1px 2px;" 
            data-comment-id="{comment['id']}" 
            data-dimension="{dimension}"
            title="{comment.get('comment', '')[:100]}...">
            {original_segment}
        </span>'''
        
        highlighted_text = highlighted_text[:start] + highlighted_segment + highlighted_text[end:]
    
    return highlighted_text

# --- UI Components ---
def create_synchronized_text_display(left_text, right_text, left_title="Left Text", right_title="Right Text", height=400, full_width=True, enable_comments=False, chapter_id=None, style_name=None, key=None):
    """
    Create a synchronized scrolling display for two text blocks with optional inline commenting.
    
    Args:
        left_text (str): Text content for left panel
        right_text (str): Text content for right panel  
        left_title (str): Title for left panel
        right_title (str): Title for right panel
        height (int): Height of display panels in pixels
        full_width (bool): If True, optimize for full width with minimal margins
        enable_comments (bool): If True, enable text selection and inline commenting
        chapter_id (str): Chapter identifier for comment storage
        style_name (str): Translation style name for comment storage
        key (str): Unique key for the component
    
    Returns:
        dict: Selection event data when text is selected for commenting
    """
    # Process text with optional comment highlighting
    def escape_html(text):
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&#x27;')
                   .replace('\n', '<br>'))
    
    # Apply comment highlighting if enabled and load existing comments
    existing_comments = []
    if enable_comments and chapter_id and style_name:
        # Load existing comments for this chapter and style
        existing_comments = load_inline_comments(style_name, chapter_id)
        
        # Apply highlighting to right panel (custom translation)
        if existing_comments:
            # Apply highlighting first on raw text, then escape HTML with preserved spans
            highlighted_right = apply_comment_highlighting(right_text, existing_comments)
            # Custom HTML escaping that preserves our comment spans
            right_escaped = highlighted_right.replace('\n', '<br>')
        else:
            right_escaped = escape_html(right_text)
        
        # Left panel remains unmodified (for comparison)
        left_escaped = escape_html(left_text)
    else:
        left_escaped = escape_html(left_text)
        right_escaped = escape_html(right_text)
    
    # Generate unique IDs for this component instance
    import random
    component_id = f"sync_scroll_{random.randint(1000, 9999)}"
    
    # Optimize styling based on full_width parameter
    container_gap = "8px" if full_width else "20px"
    content_padding = "12px" if full_width else "16px"
    header_padding = "8px 12px" if full_width else "12px 16px"
    border_radius = "4px" if full_width else "8px"
    
    # Generate existing comments HTML
    def generate_existing_comments_html(comments):
        if not comments:
            return ""
        
        comments_html = '<div style="margin-bottom: 12px; border-bottom: 1px solid #e0e0e0; padding-bottom: 8px;">'
        comments_html += f'<div style="font-size: 12px; font-weight: 600; color: #666; margin-bottom: 8px;">Existing Comments ({len(comments)})</div>'
        
        # Dimension color and icon mapping
        dimension_config = {
            'vocabulary_complexity': {'icon': '🧠', 'color': '#e3f2fd', 'name': 'Vocabulary'},
            'cultural_context': {'icon': '🌏', 'color': '#e8f5e8', 'name': 'Cultural'},
            'prose_style': {'icon': '✍️', 'color': '#fff3e0', 'name': 'Prose'},
            'creative_fidelity': {'icon': '🎨', 'color': '#fce4ec', 'name': 'Creative'},
            # Legacy support
            'english_sophistication': {'icon': '🎯', 'color': '#e3f2fd', 'name': 'English'},
            'world_building': {'icon': '🌍', 'color': '#e8f5e8', 'name': 'World'},
            'emotional_impact': {'icon': '💔', 'color': '#ffebee', 'name': 'Emotion'},
            'dialogue_naturalness': {'icon': '💬', 'color': '#f3e5f5', 'name': 'Dialogue'}
        }
        
        for comment in comments[-3:]:  # Show latest 3 comments
            dim = comment.get('dimension', 'vocabulary_complexity')
            config = dimension_config.get(dim, dimension_config['vocabulary_complexity'])
            
            comments_html += f'''
            <div style="background: {config['color']}; padding: 6px; border-radius: 3px; margin-bottom: 6px; font-size: 11px;">
                <div style="font-weight: 600; margin-bottom: 2px;">{config['icon']} {config['name']}</div>
                <div style="font-style: italic; margin-bottom: 3px;">"{comment.get('selected_text', '')[:30]}..."</div>
                <div style="color: #555;">{comment.get('comment', '')[:50]}...</div>
            </div>
            '''
        
        if len(comments) > 3:
            comments_html += f'<div style="font-size: 10px; color: #888; text-align: center;">...and {len(comments) - 3} more comments</div>'
        
        comments_html += '</div>'
        return comments_html
    
    existing_comments_html = generate_existing_comments_html(existing_comments)
    
    html_content = f"""
    <script src="https://cdn.jsdelivr.net/npm/streamlit-component-lib@1.0.0/dist/streamlit-component-lib.js"></script>
    <script>
        // Initialize Streamlit component communication
        window.addEventListener('message', function(event) {{
            if (event.data && event.data.type === 'streamlit:componentReady') {{
                console.log('🔍 DEBUG: Streamlit component ready');
            }}
        }});
        
        // Fallback if Streamlit object is not available
        if (typeof Streamlit === 'undefined') {{
            console.log('🔍 DEBUG: Streamlit not available, creating mock object');
            window.Streamlit = {{
                setComponentValue: function(value) {{
                    console.log('🔍 DEBUG: Mock setComponentValue called with:', value);
                    parent.postMessage({{
                        type: 'streamlit:setComponentValue',
                        value: value
                    }}, '*');
                }}
            }};
        }}
    </script>
    <style>
        .sync-container-{component_id} {{
            display: flex;
            gap: {container_gap};
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            width: 100%;
            margin: 0;
        }}
        
        .sync-panel-{component_id} {{
            border: 1px solid #ddd;
            border-radius: {border_radius};
            overflow: hidden;
            background: white;
            min-width: 0; /* Prevents flex item from overflowing */
        }}
        
        .sync-panel-left-{component_id} {{
            flex: 0 0 30%; /* Left panel: 30% width */
        }}
        
        .sync-panel-right-{component_id} {{
            flex: 0 0 45%; /* Right panel: 45% width */
        }}
        
        .comment-sidebar-{component_id} {{
            flex: 0 0 25%; /* Comment sidebar: 25% width */
            border: 1px solid #e0e0e0;
            border-radius: {border_radius};
            background: #f8f9fa;
            overflow-y: auto;
            max-height: {height}px;
        }}
        
        .sync-header-{component_id} {{
            background: #f8f9fa;
            padding: {header_padding};
            border-bottom: 1px solid #ddd;
            font-weight: 600;
            font-size: 14px;
            color: #333;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        
        .sync-content-{component_id} {{
            height: {height}px;
            overflow-y: auto;
            padding: {content_padding};
            line-height: 1.6;
            font-size: 14px;
            color: #333;
            background: white;
            word-wrap: break-word;
            overflow-wrap: break-word;
        }}
        
        /* Custom scrollbar styling */
        .sync-content-{component_id}::-webkit-scrollbar {{
            width: 6px;
        }}
        
        .sync-content-{component_id}::-webkit-scrollbar-track {{
            background: #f1f1f1;
            border-radius: 3px;
        }}
        
        .sync-content-{component_id}::-webkit-scrollbar-thumb {{
            background: #c1c1c1;
            border-radius: 3px;
        }}
        
        .sync-content-{component_id}::-webkit-scrollbar-thumb:hover {{
            background: #a1a1a1;
        }}
        
        /* Highlight synchronized scrolling */
        .sync-content-{component_id}.scrolling {{
            box-shadow: inset 0 0 5px rgba(0, 123, 255, 0.3);
            transition: box-shadow 0.2s ease;
        }}
        
        /* Full width responsive behavior */
        @media (max-width: 1024px) {{
            .sync-panel-left-{component_id} {{
                flex: 0 0 35%; /* Slightly larger on medium screens */
            }}
            .sync-panel-right-{component_id} {{
                flex: 0 0 40%;
            }}
            .comment-sidebar-{component_id} {{
                flex: 0 0 25%;
            }}
        }}
        
        @media (max-width: 768px) {{
            .sync-container-{component_id} {{
                flex-direction: column;
                gap: 4px;
            }}
            .sync-panel-left-{component_id}, .sync-panel-right-{component_id} {{
                flex: 1 1 auto;
            }}
            .sync-content-{component_id} {{
                height: {height//3}px; /* Smaller on mobile since we have 3 sections */
            }}
            .comment-sidebar-{component_id} {{
                flex: 1 1 auto;
                max-height: 300px;
            }}
        }}
    </style>
    
    <div class="sync-container-{component_id}">
        <div class="sync-panel-{component_id} sync-panel-left-{component_id}">
            <div class="sync-header-{component_id}">{left_title}</div>
            <div class="sync-content-{component_id}" id="left-{component_id}">
                {left_escaped}
            </div>
        </div>
        
        <div class="sync-panel-{component_id} sync-panel-right-{component_id}">
            <div class="sync-header-{component_id}">{right_title}</div>
            <div class="sync-content-{component_id}" id="right-{component_id}">
                {right_escaped}
            </div>
        </div>
        
        <div class="comment-sidebar-{component_id}" id="comment-sidebar-{component_id}">
            <div class="sync-header-{component_id}">💬 Quick Comments</div>
            <div style="padding: 12px;">
                <!-- Existing Comments Section -->
                <div id="existing-comments-{component_id}">
                    {existing_comments_html}
                </div>
                
                <!-- Comment Form (Hidden by default) -->
                <div id="comment-form-{component_id}" style="display: none; border-top: 1px solid #e0e0e0; padding-top: 8px;">
                    <div style="font-size: 12px; font-weight: 600; color: #666; margin-bottom: 8px;">✏️ New Comment</div>
                    <div id="selected-text-{component_id}" style="background: #e3f2fd; padding: 8px; border-radius: 4px; margin-bottom: 12px; font-size: 13px; font-style: italic;"></div>
                    
                    <select id="dimension-{component_id}" style="width: 100%; padding: 6px; margin-bottom: 8px; border: 1px solid #ddd; border-radius: 4px; font-size: 13px;">
                        <option value="vocabulary_complexity">🧠 Vocabulary Complexity</option>
                        <option value="cultural_context">🌏 Cultural Context</option>
                        <option value="prose_style">✍️ Prose Style</option>
                        <option value="creative_fidelity">🎨 Creative Fidelity</option>
                    </select>
                    
                    <textarea id="comment-text-{component_id}" placeholder="Explain why this text exemplifies this quality..." style="width: 100%; height: 80px; padding: 8px; border: 1px solid #ddd; border-radius: 4px; resize: vertical; font-size: 13px; font-family: inherit;" rows="4"></textarea>
                    
                    <div style="display: flex; gap: 6px; margin-top: 8px;">
                        <button id="save-comment-{component_id}" style="flex: 1; background: #007bff; color: white; border: none; padding: 8px; border-radius: 4px; font-size: 12px; cursor: pointer;">💾 Save</button>
                        <button id="cancel-comment-{component_id}" style="flex: 1; background: #6c757d; color: white; border: none; padding: 8px; border-radius: 4px; font-size: 12px; cursor: pointer;">❌ Cancel</button>
                    </div>
                </div>
                
                <!-- Placeholder for when no comments exist and no selection made -->
                <div id="comment-placeholder-{component_id}" style="color: #666; font-size: 13px; text-align: center; padding: 20px; {'display: none;' if existing_comments else ''}">
                    Select text in the right panel to add a comment
                </div>
            </div>
        </div>
    </div>
    
    <script>
    (function() {{
        const leftPanel = document.getElementById('left-{component_id}');
        const rightPanel = document.getElementById('right-{component_id}');
        
        let isScrollingSynced = false;
        
        function syncScroll(source, target) {{
            if (isScrollingSynced) return;
            
            isScrollingSynced = true;
            
            // Calculate scroll percentage
            const scrollPercentage = source.scrollTop / (source.scrollHeight - source.clientHeight);
            
            // Apply to target
            target.scrollTop = scrollPercentage * (target.scrollHeight - target.clientHeight);
            
            // Visual feedback
            source.classList.add('scrolling');
            target.classList.add('scrolling');
            
            setTimeout(() => {{
                source.classList.remove('scrolling');
                target.classList.remove('scrolling');
                isScrollingSynced = false;
            }}, 150);
        }}
        
        if (leftPanel && rightPanel) {{
            leftPanel.addEventListener('scroll', () => syncScroll(leftPanel, rightPanel));
            rightPanel.addEventListener('scroll', () => syncScroll(rightPanel, leftPanel));
        }}
        
        // Enhanced: Text selection and commenting functionality
        {'// Commenting enabled for this component' if enable_comments else '// Commenting disabled'}
        {f'''
        // Text selection handling for commenting
        let selectedText = '';
        let selectionRange = null;
        
        function handleTextSelection() {{
            console.log('🔍 DEBUG: Text selection event triggered');
            const selection = window.getSelection();
            console.log('🔍 DEBUG: Selection object:', selection);
            console.log('🔍 DEBUG: Selected text length:', selection.toString().length);
            
            if (selection.toString().length > 0) {{
                selectedText = selection.toString().trim();
                console.log('🔍 DEBUG: Selected text:', selectedText);
                
                // Only enable commenting on right panel (custom translation)
                const range = selection.getRangeAt(0);
                const container = range.commonAncestorContainer;
                
                // Check if selection is within right panel
                if (rightPanel.contains(container) || rightPanel === container) {{
                    // Calculate text offset for persistence
                    const textContent = rightPanel.textContent || rightPanel.innerText;
                    const beforeSelection = textContent.substring(0, textContent.indexOf(selectedText));
                    
                    selectionRange = {{
                        startOffset: beforeSelection.length,
                        endOffset: beforeSelection.length + selectedText.length,
                        text: selectedText
                    }};
                    
                    // Show comment form in sidebar
                    showSidebarCommentForm(selectedText);
                }}
            }} else {{
                hideSidebarCommentForm();
            }}
        }}
        
        function showSidebarCommentForm(selectedText) {{
            // Show the comment form in the sidebar
            const placeholder = document.getElementById('comment-placeholder-{component_id}');
            const commentForm = document.getElementById('comment-form-{component_id}');
            
            if (placeholder) placeholder.style.display = 'none';
            if (commentForm) commentForm.style.display = 'block';
            
            // Populate selected text
            const selectedTextDiv = document.getElementById('selected-text-{component_id}');
            if (selectedTextDiv) selectedTextDiv.textContent = '"' + selectedText + '"';
            
            // Clear previous form data
            const commentTextArea = document.getElementById('comment-text-{component_id}');
            const dimensionSelect = document.getElementById('dimension-{component_id}');
            if (commentTextArea) commentTextArea.value = '';
            if (dimensionSelect) dimensionSelect.selectedIndex = 0;
        }}
        
        function hideSidebarCommentForm() {{
            // Hide the comment form 
            const commentForm = document.getElementById('comment-form-{component_id}');
            const placeholder = document.getElementById('comment-placeholder-{component_id}');
            const existingComments = document.getElementById('existing-comments-{component_id}');
            
            if (commentForm) commentForm.style.display = 'none';
            
            // Only show placeholder if there are no existing comments
            if (placeholder && (!existingComments || existingComments.innerHTML.trim() === '')) {{
                placeholder.style.display = 'block';
            }}
        }}
        
        // Sidebar form button handlers
        document.getElementById('save-comment-{component_id}').onclick = function() {{
            console.log('🔍 DEBUG: Save button clicked');
            const commentText = document.getElementById('comment-text-{component_id}').value.trim();
            console.log('🔍 DEBUG: Comment text:', commentText);
            console.log('🔍 DEBUG: Selection range:', selectionRange);
            console.log('🔍 DEBUG: Streamlit object available:', typeof Streamlit !== 'undefined');
            
            if (commentText && selectionRange) {{
                // Send comment data back to Streamlit
                const eventData = {{
                    type: 'comment_saved',
                    text: selectionRange.text,
                    comment: commentText,
                    dimension: document.getElementById('dimension-{component_id}').value,
                    start_char: selectionRange.startOffset,
                    end_char: selectionRange.endOffset,
                    chapter_id: '{chapter_id}',
                    style_name: '{style_name}',
                    timestamp: new Date().toISOString()
                }};
                console.log('🔍 DEBUG: Event data prepared:', eventData);
                console.log('🔍 DEBUG: Calling Streamlit.setComponentValue');
                
                try {{
                    Streamlit.setComponentValue(JSON.stringify(eventData));
                    console.log('🔍 DEBUG: setComponentValue called successfully');
                    
                    // Create temporary visual feedback
                    const existingComments = document.getElementById('existing-comments-{component_id}');
                    if (existingComments) {{
                        const tempComment = document.createElement('div');
                        tempComment.style.cssText = 'background: #d4edda; padding: 6px; border-radius: 3px; margin-bottom: 6px; font-size: 11px; border: 1px solid #c3e6cb;';
                        tempComment.innerHTML = `
                            <div style="font-weight: 600; margin-bottom: 2px;">💾 Comment Saved</div>
                            <div style="font-style: italic; margin-bottom: 3px;">"${{selectedText.substring(0, 30)}}..."</div>
                            <div style="color: #555;">${{commentText.substring(0, 50)}}...</div>
                        `;
                        existingComments.appendChild(tempComment);
                        
                        // Remove temp comment after 3 seconds
                        setTimeout(() => {{
                            if (tempComment.parentNode) {{
                                tempComment.parentNode.removeChild(tempComment);
                            }}
                        }}, 3000);
                    }}
                }} catch (error) {{
                    console.error('❌ DEBUG: Error calling setComponentValue:', error);
                }}
                
                hideSidebarCommentForm();
            }} else {{
                console.log('❌ DEBUG: Missing comment text or selection range');
                console.log('🔍 DEBUG: commentText:', commentText);
                console.log('🔍 DEBUG: selectionRange:', selectionRange);
            }}
        }};
        
        document.getElementById('cancel-comment-{component_id}').onclick = function() {{
            hideSidebarCommentForm();
        }};
        
        // Add text selection listeners to right panel only
        if (rightPanel) {{
            rightPanel.addEventListener('mouseup', handleTextSelection);
            rightPanel.addEventListener('touchend', handleTextSelection);
        }}
        
        // Hide comment form when clicking elsewhere (except sidebar)
        document.addEventListener('click', function(e) {{
            const sidebar = document.getElementById('comment-sidebar-{component_id}');
            if (!sidebar.contains(e.target) && !rightPanel.contains(e.target)) {{
                hideSidebarCommentForm();
            }}
        }});
        ''' if enable_comments else ''}
    }})();
    </script>
    """
    
    # Use Streamlit's HTML component with bi-directional communication
    try:
        import streamlit.components.v1 as components
        # Return the component value to capture selection events
        selection_event = components.html(html_content, height=height + 80)
        return selection_event
    except ImportError:
        # Fallback to markdown if components not available
        st.markdown("**Synchronized display component not available - missing streamlit.components.v1**")
        st.text_area(left_title, left_text, height=height//2, disabled=True)
        st.text_area(right_title, right_text, height=height//2, disabled=True)
        return None