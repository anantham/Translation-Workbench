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

# Ensure directories exist
for directory in [DATA_DIR, CACHE_DIR, EXPORT_DIR, TEMP_DIR, MODELS_DIR, TRANSLATIONS_DIR]:
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

def load_dataset_for_tuning(alignment_map, limit=None, min_similarity=0.5, max_chars=30000):
    """
    Load dataset from alignment map and prepare for fine-tuning.
    
    Returns:
        list: Training examples in the format expected by fine-tuning APIs
    """
    training_examples = []
    processed = 0
    
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
        
        # Quality filters
        raw_stats = get_text_stats(raw_content, 'chinese')
        eng_stats = get_text_stats(english_content, 'english')
        
        # Skip if too short, too long, or poor quality
        if (raw_stats['char_count'] < 500 or eng_stats['char_count'] < 500 or
            raw_stats['char_count'] > max_chars or eng_stats['char_count'] > max_chars):
            continue
        
        # Create training example
        training_example = {
            "chapter_number": chapter_num,
            "raw_content": raw_content,
            "english_content": english_content,
            "raw_stats": raw_stats,
            "english_stats": eng_stats
        }
        
        training_examples.append(training_example)
        processed += 1
    
    return training_examples

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