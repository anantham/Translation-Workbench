#!/usr/bin/env python3
"""
Complete Dataset Builder and Reporter
Creates comprehensive CSV datasheet and ML-ready JSONL training files
"""

import os
import json
import requests
import hashlib
import time
import re
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
from functools import wraps
import random

# --- Organized Data Structure ---
DATA_DIR = "data"
CACHE_DIR = os.path.join(DATA_DIR, "cache")
EXPORT_DIR = os.path.join(DATA_DIR, "exports")
TEMP_DIR = os.path.join(DATA_DIR, "temp")

# Ensure directories exist
for directory in [DATA_DIR, CACHE_DIR, EXPORT_DIR, TEMP_DIR]:
    os.makedirs(directory, exist_ok=True)

# --- Import detection for optional dependencies ---
SEMANTIC_AVAILABLE = False
try:
    import torch
    from sentence_transformers import SentenceTransformer
    SEMANTIC_AVAILABLE = True
    print("✅ Semantic similarity enabled (BERT)")
except ImportError:
    print("📝 Using syntactic similarity (install sentence-transformers for semantic)")
    from difflib import SequenceMatcher

# Chinese text processing
JIEBA_AVAILABLE = False
try:
    import jieba
    JIEBA_AVAILABLE = True
    print("✅ Chinese word segmentation enabled (jieba)")
except ImportError:
    print("📝 Using character count for Chinese (install jieba for word segmentation)")

# --- Smart Resume Functions ---
def get_recent_exports(hours=24):
    """Find recent export files within time window."""
    if not os.path.exists(EXPORT_DIR):
        return []
    
    cutoff_time = datetime.now() - timedelta(hours=hours)
    recent_files = []
    
    for file in os.listdir(EXPORT_DIR):
        if file.startswith('dataset_report_') and file.endswith('.csv'):
            file_path = os.path.join(EXPORT_DIR, file)
            try:
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if file_time > cutoff_time:
                    recent_files.append(file_path)
            except OSError:
                continue  # Skip files we can't read
    
    # Sort by modification time (newest first)
    recent_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return recent_files

def check_completed_chapters(recent_exports):
    """Extract completed chapters from recent exports."""
    completed = set()
    
    print(f"   └─ 📊 Checking {len(recent_exports)} recent export(s)...")
    
    for export_file in recent_exports:
        try:
            df = pd.read_csv(export_file)
            chapters_in_file = set(df['Chapter'].tolist())
            completed.update(chapters_in_file)
            print(f"   └─ 📄 {os.path.basename(export_file)}: {len(chapters_in_file)} chapters")
        except Exception as e:
            print(f"   └─ ⚠️ Could not read {os.path.basename(export_file)}: {e}")
    
    return completed

def smart_chapter_selection(chapters_to_process):
    """Smart chapter selection with resume capability."""
    print("\n🔍 Checking for recent exports...")
    
    recent_exports = get_recent_exports(hours=24)
    
    if not recent_exports:
        print("   └─ 📭 No recent exports found")
        return chapters_to_process, False
    
    completed_chapters = check_completed_chapters(recent_exports)
    
    if not completed_chapters:
        print("   └─ 📭 No completed chapters in recent exports")
        return chapters_to_process, False
    
    # Check overlap
    requested_chapters = set(chapters_to_process)
    overlap = completed_chapters.intersection(requested_chapters)
    remaining = requested_chapters - completed_chapters
    
    if not overlap:
        print(f"   └─ ✅ No overlap found - all {len(chapters_to_process)} chapters need processing")
        return chapters_to_process, False
    
    print(f"\n📋 Resume Analysis:")
    print(f"   └─ 🎯 Requested: {len(requested_chapters)} chapters")
    print(f"   └─ ✅ Completed: {len(overlap)} chapters")
    print(f"   └─ 🔄 Remaining: {len(remaining)} chapters")
    
    if len(remaining) == 0:
        print("\n🎉 All requested chapters already completed!")
        return [], True  # Nothing to process
    
    # Ask user what to do
    print(f"\n📝 Options:")
    print(f"   (a) Append: Process only {len(remaining)} remaining chapters")
    print(f"   (f) Full: Reprocess all {len(requested_chapters)} chapters")
    print(f"   (q) Quit: Exit without processing")
    
    while True:
        choice = input("\n🤔 Choose option (a/f/q): ").strip().lower()
        
        if choice in ['a', 'append']:
            return sorted(list(remaining)), True
        elif choice in ['f', 'full']:
            return chapters_to_process, False
        elif choice in ['q', 'quit']:
            print("👋 Exiting...")
            return [], True  # Signal to exit
        else:
            print("❌ Invalid choice. Please enter 'a', 'f', or 'q'")

def determine_output_strategy(is_resume_mode, chapters_to_process):
    """Determine whether to create new files or append to existing."""
    if not is_resume_mode:
        return "new", None, None
    
    if len(chapters_to_process) == 0:
        return "none", None, None  # Nothing to do
    
    # For resume mode, we'll create incremental files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return "incremental", timestamp, f"incremental_{timestamp}"

# --- Caching System (reused from master_review_tool.py) ---
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

def get_cached_similarity(text1, text2, cache):
    """Get cached similarity score if available."""
    hash1 = generate_text_hash(text1)
    hash2 = generate_text_hash(text2)
    
    cache_key1 = f"{hash1}:{hash2}"
    cache_key2 = f"{hash2}:{hash1}"
    
    if cache_key1 in cache:
        return cache[cache_key1]
    elif cache_key2 in cache:
        return cache[cache_key2]
    return None

def store_similarity_in_cache(text1, text2, score, cache):
    """Store computed similarity score in cache."""
    hash1 = generate_text_hash(text1)
    hash2 = generate_text_hash(text2)
    cache_key = f"{hash1}:{hash2}"
    cache[cache_key] = score
    return cache

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

# --- Text Statistics with Language Intelligence ---
def get_text_stats(content, language_hint=None):
    """
    Get comprehensive text statistics with language-aware counting.
    
    Args:
        content: Text content to analyze
        language_hint: 'chinese', 'english', or None for auto-detection
        
    Returns:
        dict: Statistics including char_count, word_count, line_count, language
    """
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
        language_hint = 'chinese' if cjk_chars / char_count > 0.2 else 'english'
    
    # Language-aware word counting
    if language_hint == 'chinese':
        if JIEBA_AVAILABLE:
            # Use jieba for proper Chinese word segmentation
            word_count = len(list(jieba.cut(content)))
        else:
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

# --- Retry Decorator for Network Resilience ---
def retry_with_backoff(retries=5, backoff_in_seconds=2):
    """A decorator to retry a function with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            x = 0
            while x < retries:
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.RequestException, 
                        requests.exceptions.ConnectionError,
                        requests.exceptions.Timeout,
                        ConnectionResetError) as e:
                    x += 1
                    if x < retries:
                        sleep_time = (backoff_in_seconds * 2 ** (x-1)) + random.uniform(0, 1)
                        print(f"\n   └─ Retrying ({x}/{retries})... Network error: {type(e).__name__}")
                        print(f"   └─ Waiting {sleep_time:.1f}s before retry...")
                        time.sleep(sleep_time)
                    else:
                        print(f"\n   └─ Max retries exceeded after {retries} attempts")
                        return f"API Request Failed: Max retries exceeded - {e}"
                except Exception as e:
                    return f"API Request Failed: {e}"
            return f"API Request Failed: Max retries exceeded"
        return wrapper
    return decorator

# --- Translation Functions ---
@retry_with_backoff(retries=5, backoff_in_seconds=2)
def translate_with_gemini(raw_text: str, api_key: str, use_cache=True):
    """Sends raw text to Gemini for translation with caching support and retry logic."""
    # Check cache first if enabled
    if use_cache:
        cached_translation = get_cached_translation(raw_text)
        if cached_translation:
            print("   └─ 💾 Using cached translation")
            return cached_translation
    
    # Make API call if not cached
    # Use the same model as other components for consistency
    model_name = "gemini-2.5-pro"  # Updated to current model
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    prompt = f"Provide a high-quality, literal English translation of this Chinese web novel chapter. Keep paragraph breaks:\n\n{raw_text}"
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    print(f"   └─ 🌐 Calling Gemini API...")
    print(f"   └─ 🤖 Model: {model_name}")
    print(f"   └─ 📄 Prompt length: {len(prompt)} chars")
    
    try:
        # This will automatically retry on network errors due to the decorator
        response = requests.post(gemini_url, headers=headers, json=data, timeout=120)
        
        # Enhanced error logging
        if response.status_code != 200:
            print(f"   └─ ❌ HTTP {response.status_code}: {response.reason}")
            print(f"   └─ 🔗 URL: {gemini_url}")
            print(f"   └─ 📄 Response: {response.text[:300]}...")
            
            if response.status_code == 404:
                print(f"   └─ 💡 Model '{model_name}' not found. Check if model name is correct.")
            elif response.status_code == 400:
                print(f"   └─ 💡 Bad request. Check API key format and request structure.")
            elif response.status_code == 403:
                print(f"   └─ 💡 Permission denied. Check API key permissions and billing.")
        
        response.raise_for_status()
        
        response_json = response.json()
        translation = response_json['candidates'][0]['content']['parts'][0]['text']
        
        print(f"   └─ ✅ Translation received ({len(translation)} chars)")
        
        # Store in cache if successful and caching is enabled
        if use_cache and translation and not translation.startswith("API Request Failed"):
            store_translation_in_cache(raw_text, translation)
        
        return translation
        
    except requests.exceptions.HTTPError as e:
        error_msg = f"HTTP Error {e.response.status_code}: {e.response.reason}"
        print(f"   └─ ❌ {error_msg}")
        return f"API Request Failed: {error_msg}"
    
    except requests.exceptions.RequestException as e:
        error_msg = f"Network Error: {type(e).__name__}: {str(e)}"
        print(f"   └─ ❌ {error_msg}")
        return f"API Request Failed: {error_msg}"
    
    except (KeyError, IndexError) as e:
        error_msg = f"Response parsing error: {str(e)}"
        print(f"   └─ ❌ {error_msg}")
        print(f"   └─ 📄 Raw response: {response.text[:300]}...")
        return f"API Request Failed: {error_msg}"

# --- Similarity Functions ---
def load_semantic_model():
    """Load semantic similarity model."""
    if not SEMANTIC_AVAILABLE:
        return None
    
    try:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        return model
    except Exception as e:
        print(f"Warning: Could not load semantic model: {e}")
        return None

def calculate_semantic_similarity(text1, text2, model=None, cache=None):
    """Calculate semantic similarity using BERT embeddings with caching."""
    if not text1 or not text2 or "File not found" in text1 or "File not found" in text2:
        return 0.0
    
    # Load cache if not provided
    if cache is None:
        cache = load_similarity_cache()
    
    # Check cache first
    cached_score = get_cached_similarity(text1, text2, cache)
    if cached_score is not None:
        return cached_score
    
    if model is None:
        model = load_semantic_model()
        if model is None:
            return calculate_syntactic_similarity_fallback(text1, text2)
    
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
        
        # Convert to float and ensure it's between 0 and 1
        similarity = float(cosine_scores.item())
        similarity = max(0.0, min(1.0, similarity))
        
        # Store in cache
        store_similarity_in_cache(text1, text2, similarity, cache)
        save_similarity_cache(cache)
        
        return similarity
        
    except Exception as e:
        print(f"Warning: Semantic similarity calculation failed: {e}")
        return calculate_syntactic_similarity_fallback(text1, text2)

def calculate_syntactic_similarity_fallback(text1, text2):
    """Fallback syntactic similarity for when semantic models aren't available."""
    if not SEMANTIC_AVAILABLE:
        from difflib import SequenceMatcher
    
    # Length similarity (±30% tolerance)
    len1, len2 = len(text1), len(text2)
    length_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
    
    # Content similarity (first 1000 chars for speed)
    sample1 = text1[:1000].lower().replace('\n', ' ')
    sample2 = text2[:1000].lower().replace('\n', ' ')
    content_similarity = SequenceMatcher(None, sample1, sample2).ratio()
    
    # Combined score
    return (length_ratio * 0.3) + (content_similarity * 0.7)

def calculate_similarity(text1, text2, model=None, cache=None):
    """Main similarity function - uses semantic if available, falls back to syntactic."""
    if SEMANTIC_AVAILABLE:
        return calculate_semantic_similarity(text1, text2, model, cache)
    else:
        return calculate_syntactic_similarity_fallback(text1, text2)

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

def extract_chapter_title(content, language='chinese'):
    """Extract chapter title from content."""
    if not content or "File not found" in content:
        return "Unknown Chapter"
    
    lines = content.split('\n')
    for line in lines[:5]:  # Check first 5 lines
        line = line.strip()
        if line:
            # Chinese pattern: 第X章 or Chapter X:
            if language == 'chinese' and ('第' in line and '章' in line):
                return line
            elif language == 'english' and line.startswith('Chapter'):
                return line
            elif len(line) < 100:  # Likely a title if short
                return line
    
    # Fallback: use first non-empty line
    for line in lines[:10]:
        if line.strip():
            return line.strip()[:100]  # Truncate if too long
    
    return "No Title Found"

# --- Main Processing Functions ---
def load_alignment_map(filepath="alignment_map.json"):
    """Load the alignment map."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Alignment map not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def process_chapter_data(chapter_num, chapter_data, api_key, semantic_model, similarity_cache, progress_callback=None):
    """
    Process a single chapter and return comprehensive statistics.
    
    Returns:
        dict: Complete chapter analysis or None if chapter should be skipped
    """
    if progress_callback:
        progress_callback(f"Processing Chapter {chapter_num}...")
    
    # Load content with detailed error reporting
    raw_file = chapter_data.get('raw_file')
    english_file = chapter_data.get('english_file')
    
    # Detailed file checking
    if not raw_file:
        print(f"\n   └─ ⏭️ Skipping Chapter {chapter_num}: No 'raw_file' in alignment map")
        return None
    if not english_file:
        print(f"\n   └─ ⏭️ Skipping Chapter {chapter_num}: No 'english_file' in alignment map")
        return None
    
    # Check if files actually exist
    if not os.path.exists(raw_file):
        print(f"\n   └─ ⏭️ Skipping Chapter {chapter_num}: Raw file NOT FOUND")
        print(f"   └─ Expected path: '{raw_file}'")
        return None
    if not os.path.exists(english_file):
        print(f"\n   └─ ⏭️ Skipping Chapter {chapter_num}: English file NOT FOUND")
        print(f"   └─ Expected path: '{english_file}'")
        return None
    
    # Load content
    raw_content = load_chapter_content(raw_file)
    english_content = load_chapter_content(english_file)
    
    if "File not found" in raw_content:
        print(f"\n   └─ ⏭️ Skipping Chapter {chapter_num}: Could not read raw file '{raw_file}'")
        return None
    if "File not found" in english_content:
        print(f"\n   └─ ⏭️ Skipping Chapter {chapter_num}: Could not read English file '{english_file}'")
        return None
    
    # Check for empty files
    if len(raw_content.strip()) < 100:
        print(f"\n   └─ ⏭️ Skipping Chapter {chapter_num}: Raw file too short ({len(raw_content)} chars)")
        return None
    if len(english_content.strip()) < 100:
        print(f"\n   └─ ⏭️ Skipping Chapter {chapter_num}: English file too short ({len(english_content)} chars)")
        return None
    
    # Extract titles
    raw_title = extract_chapter_title(raw_content, 'chinese')
    english_title = extract_chapter_title(english_content, 'english')
    
    # Get AI translation (with caching and retry)
    ai_translation = translate_with_gemini(raw_content, api_key, use_cache=True)
    
    if "API Request Failed" in ai_translation:
        print(f"\n   └─ ⚠️ Chapter {chapter_num}: AI translation failed - {ai_translation[:100]}...")
        ai_translation = ""  # Continue processing even if AI translation fails
    
    # Calculate statistics for all three versions
    raw_stats = get_text_stats(raw_content, 'chinese')
    english_stats = get_text_stats(english_content, 'english') 
    ai_stats = get_text_stats(ai_translation, 'english') if ai_translation else {'char_count': 0, 'word_count': 0, 'line_count': 0}
    
    # Calculate similarity between English and AI translation
    similarity_score = 0.0
    if ai_translation and english_content:
        similarity_score = calculate_similarity(english_content, ai_translation, semantic_model, similarity_cache)
    
    # Calculate ratios for quality assessment
    eng_to_raw_ratio = english_stats['word_count'] / raw_stats['word_count'] if raw_stats['word_count'] > 0 else 0
    ai_to_raw_ratio = ai_stats['word_count'] / raw_stats['word_count'] if raw_stats['word_count'] > 0 else 0
    
    return {
        # Basic info
        'chapter_number': chapter_num,
        'raw_title': raw_title,
        'english_title': english_title,
        
        # File paths
        'raw_file': raw_file,
        'english_file': english_file,
        
        # Chinese/Raw statistics
        'raw_char_count': raw_stats['char_count'],
        'raw_word_count': raw_stats['word_count'],
        'raw_line_count': raw_stats['line_count'],
        
        # English/Official statistics
        'english_char_count': english_stats['char_count'],
        'english_word_count': english_stats['word_count'],
        'english_line_count': english_stats['line_count'],
        
        # AI Translation statistics
        'ai_char_count': ai_stats['char_count'],
        'ai_word_count': ai_stats['word_count'],
        'ai_line_count': ai_stats['line_count'],
        
        # Quality metrics
        'similarity_score': round(similarity_score, 4),
        'eng_to_raw_ratio': round(eng_to_raw_ratio, 2),
        'ai_to_raw_ratio': round(ai_to_raw_ratio, 2),
        
        # Content for JSONL export
        'raw_content': raw_content,
        'english_content': english_content,
        'ai_content': ai_translation
    }

def create_csv_report(processed_chapters, output_path):
    """Create comprehensive CSV datasheet."""
    print(f"\n📊 Creating CSV report: {output_path}")
    
    # Prepare data for DataFrame
    csv_data = []
    for chapter_data in processed_chapters:
        csv_data.append({
            'Chapter': chapter_data['chapter_number'],
            'Raw_Title': chapter_data['raw_title'],
            'English_Title': chapter_data['english_title'],
            
            # Chinese statistics
            'Raw_Characters': chapter_data['raw_char_count'],
            'Raw_Words': chapter_data['raw_word_count'],
            'Raw_Lines': chapter_data['raw_line_count'],
            
            # English statistics
            'English_Characters': chapter_data['english_char_count'],
            'English_Words': chapter_data['english_word_count'],
            'English_Lines': chapter_data['english_line_count'],
            
            # AI Translation statistics
            'AI_Characters': chapter_data['ai_char_count'],
            'AI_Words': chapter_data['ai_word_count'],
            'AI_Lines': chapter_data['ai_line_count'],
            
            # Quality metrics
            'BERT_Similarity': chapter_data['similarity_score'],
            'Eng_Raw_Ratio': chapter_data['eng_to_raw_ratio'],
            'AI_Raw_Ratio': chapter_data['ai_to_raw_ratio'],
            
            # File paths
            'Raw_File': chapter_data['raw_file'],
            'English_File': chapter_data['english_file']
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(csv_data)
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    # Print summary statistics
    print(f"✅ CSV Report created: {len(df)} chapters")
    print(f"📊 Summary Statistics:")
    print(f"   Average similarity score: {df['BERT_Similarity'].mean():.3f}")
    print(f"   Average Eng/Raw ratio: {df['Eng_Raw_Ratio'].mean():.2f}")
    print(f"   Chapters with high similarity (>0.7): {len(df[df['BERT_Similarity'] > 0.7])}")
    print(f"   Chapters with low similarity (<0.5): {len(df[df['BERT_Similarity'] < 0.5])}")
    
    return df

def create_jsonl_training_files(processed_chapters, train_output_path, val_output_path, train_split=0.8, max_chars=30000):
    """Create JSONL training and validation files for ML model fine-tuning."""
    print(f"\n📚 Creating JSONL training files...")
    
    # Filter chapters with good alignment and within character limits
    valid_chapters = []
    for chapter in processed_chapters:
        if (chapter['similarity_score'] > 0.5 and  # Good alignment
            chapter['raw_char_count'] > 500 and     # Not too short
            chapter['english_char_count'] > 500 and 
            chapter['raw_char_count'] < max_chars and  # Not too long
            chapter['english_char_count'] < max_chars):
            valid_chapters.append(chapter)
    
    print(f"📋 Valid chapters for training: {len(valid_chapters)} / {len(processed_chapters)}")
    
    # Shuffle and split
    import random
    random.shuffle(valid_chapters)
    split_idx = int(len(valid_chapters) * train_split)
    train_chapters = valid_chapters[:split_idx]
    val_chapters = valid_chapters[split_idx:]
    
    # Create training JSONL
    with open(train_output_path, 'w', encoding='utf-8') as f:
        for chapter in train_chapters:
            training_example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional translator specializing in Chinese to English translation of web novels. Provide accurate, fluent translations that preserve the original meaning and style."
                    },
                    {
                        "role": "user", 
                        "content": f"Translate this Chinese web novel chapter to English:\n\n{chapter['raw_content']}"
                    },
                    {
                        "role": "assistant",
                        "content": chapter['english_content']
                    }
                ]
            }
            f.write(json.dumps(training_example, ensure_ascii=False) + '\n')
    
    # Create validation JSONL
    with open(val_output_path, 'w', encoding='utf-8') as f:
        for chapter in val_chapters:
            validation_example = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional translator specializing in Chinese to English translation of web novels. Provide accurate, fluent translations that preserve the original meaning and style."
                    },
                    {
                        "role": "user",
                        "content": f"Translate this Chinese web novel chapter to English:\n\n{chapter['raw_content']}"
                    },
                    {
                        "role": "assistant", 
                        "content": chapter['english_content']
                    }
                ]
            }
            f.write(json.dumps(validation_example, ensure_ascii=False) + '\n')
    
    print(f"✅ Training file: {len(train_chapters)} examples → {train_output_path}")
    print(f"✅ Validation file: {len(val_chapters)} examples → {val_output_path}")
    
    return len(train_chapters), len(val_chapters)

def main():
    """Main execution function."""
    import sys
    
    print("🚀 Building Complete Dataset and Report")
    print("=" * 50)
    
    # Parse command line arguments
    max_chapters = None
    start_chapter = 1
    
    if len(sys.argv) > 1:
        try:
            if len(sys.argv) == 2:
                max_chapters = int(sys.argv[1])
                print(f"📊 Processing first {max_chapters} chapters")
            elif len(sys.argv) == 3:
                start_chapter = int(sys.argv[1])
                max_chapters = int(sys.argv[2])
                print(f"📊 Processing chapters {start_chapter} to {max_chapters}")
        except ValueError:
            print("❌ Usage: python build_and_report.py [max_chapters] or [start_chapter] [end_chapter]")
            print("📝 Examples:")
            print("   python build_and_report.py 50          # Process first 50 chapters")
            print("   python build_and_report.py 20 70       # Process chapters 20-70")
            return
    
    # Load API key from config/environment
    print("\n🔑 Loading API configuration...")
    try:
        # Import the shared config loading function
        import sys
        sys.path.append('.')
        from utils import load_api_config
        
        api_key, api_source = load_api_config()
        if not api_key:
            print("❌ No API key found.")
            print("💡 Solutions:")
            print("   1. Set environment variable: export GEMINI_API_KEY='your-key'")
            print("   2. Create config.json with your API key")
            print("   3. Get API key from: https://aistudio.google.com/app/apikey")
            return
        
        print(f"✅ API key loaded from {api_source}")
        
    except ImportError as e:
        print(f"❌ Could not import utils module: {e}")
        print("💡 Make sure you're running from the project root directory")
        return
    
    # Load alignment map
    print("\n📋 Loading alignment map...")
    try:
        alignment_map = load_alignment_map()
        print(f"✅ Loaded alignment map with {len(alignment_map)} chapters")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Run 'python scripts/utils/build_complete_alignment_map.py' first")
        return
    
    # Load semantic model if available
    print("\n🧠 Loading similarity model...")
    semantic_model = load_semantic_model() if SEMANTIC_AVAILABLE else None
    similarity_cache = load_similarity_cache()
    
    # Determine chapters to process based on command line args
    all_available_chapters = sorted([int(k) for k in alignment_map.keys()])
    
    if max_chapters is None:
        # Interactive mode
        process_all = input("\n🎯 Process all chapters? (y/n, default=n): ").strip().lower() == 'y'
        
        if process_all:
            initial_chapters = all_available_chapters
        else:
            default_max = min(50, len(all_available_chapters))
            max_chapters = int(input(f"📊 How many chapters to process? (default={default_max}): ").strip() or str(default_max))
            initial_chapters = all_available_chapters[:max_chapters]
    else:
        # Command line mode
        if len(sys.argv) == 3:  # start and end specified
            initial_chapters = [ch for ch in all_available_chapters if start_chapter <= ch <= max_chapters]
        else:  # just max specified
            initial_chapters = all_available_chapters[:max_chapters]
    
    # Smart chapter selection with resume capability
    chapters_to_process, is_resume_mode = smart_chapter_selection(initial_chapters)
    
    if len(chapters_to_process) == 0:
        print("🎉 Nothing to process. Exiting.")
        return
    
    print(f"\n🔄 Processing {len(chapters_to_process)} chapters...")
    if is_resume_mode:
        print("📝 Resume mode: Processing only remaining chapters")
    
    # Process chapters
    processed_chapters = []
    start_time = time.time()
    
    successful_count = 0
    skipped_count = 0
    
    for i, chapter_num in enumerate(chapters_to_process):
        # Progress indicator
        progress = (i + 1) / len(chapters_to_process)
        print(f"[{progress:.1%}] Chapter {chapter_num}... ", end="", flush=True)
        
        chapter_data = alignment_map[str(chapter_num)]
        result = process_chapter_data(
            chapter_num, 
            chapter_data, 
            api_key, 
            semantic_model, 
            similarity_cache
        )
        
        if result:
            processed_chapters.append(result)
            successful_count += 1
            print("✅")
        else:
            skipped_count += 1
            print("⏭️")
        
        # Show running totals every 50 chapters
        if (i + 1) % 50 == 0:
            print(f"\n📊 Progress: {successful_count} successful, {skipped_count} skipped so far...")
        
        # Rate limiting - be gentle on the API
        time.sleep(1.0)
    
    # Save updated cache
    save_similarity_cache(similarity_cache)
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Processing completed in {elapsed_time:.1f} seconds")
    print(f"✅ Successfully processed: {successful_count} chapters")
    print(f"⏭️  Skipped: {skipped_count} chapters")
    print(f"📊 Success rate: {(successful_count/(successful_count+skipped_count))*100:.1f}%")
    
    # Determine output strategy based on resume mode
    output_strategy, timestamp, run_suffix = determine_output_strategy(is_resume_mode, chapters_to_process)
    
    if output_strategy == "none":
        return  # Nothing to export
    
    # Generate timestamp for file naming
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create CSV report
    if output_strategy == "incremental":
        csv_output = os.path.join(EXPORT_DIR, f"dataset_report_{run_suffix}.csv")
        print(f"\n📊 Creating incremental CSV report: {os.path.basename(csv_output)}")
    else:
        csv_output = os.path.join(EXPORT_DIR, f"dataset_report_{timestamp}.csv")
        print(f"\n📊 Creating new CSV report: {os.path.basename(csv_output)}")
    df = create_csv_report(processed_chapters, csv_output)
    
    # Create JSONL training files
    if output_strategy == "incremental":
        train_output = os.path.join(EXPORT_DIR, f"training_data_{run_suffix}.jsonl")
        val_output = os.path.join(EXPORT_DIR, f"validation_data_{run_suffix}.jsonl")
    else:
        train_output = os.path.join(EXPORT_DIR, f"training_data_{timestamp}.jsonl")
        val_output = os.path.join(EXPORT_DIR, f"validation_data_{timestamp}.jsonl")
    train_count, val_count = create_jsonl_training_files(processed_chapters, train_output, val_output)
    
    # Final summary
    print(f"\n🎉 Dataset Building Complete!")
    print(f"📁 Output Directory: {EXPORT_DIR}")
    print(f"📊 CSV Report: {os.path.basename(csv_output)} ({len(df)} chapters)")
    print(f"🤖 Training Data: {os.path.basename(train_output)} ({train_count} examples)")
    print(f"🔍 Validation Data: {os.path.basename(val_output)} ({val_count} examples)")
    
    similarity_method = "BERT semantic" if SEMANTIC_AVAILABLE else "syntactic"
    print(f"🎯 Similarity Method: {similarity_method}")
    
    # Show next steps
    print(f"\n💡 Next Steps:")
    print(f"1. Review the CSV report for data quality insights")
    print(f"2. Use the JSONL files to fine-tune a translation model")
    print(f"3. Consider filtering low-similarity chapters for higher quality training")
    
    print(f"\n🚀 Ready for machine learning model training!")

if __name__ == "__main__":
    main()