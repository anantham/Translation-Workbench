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
from datetime import datetime
from collections import Counter

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

# --- Translation Functions ---
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
    
    # Load content
    raw_file = chapter_data.get('raw_file')
    english_file = chapter_data.get('english_file')
    
    if not raw_file or not english_file:
        print(f"⏭️  Skipping Chapter {chapter_num}: Missing files")
        return None
    
    raw_content = load_chapter_content(raw_file)
    english_content = load_chapter_content(english_file)
    
    if "File not found" in raw_content or "File not found" in english_content:
        print(f"⏭️  Skipping Chapter {chapter_num}: Could not load content")
        return None
    
    # Extract titles
    raw_title = extract_chapter_title(raw_content, 'chinese')
    english_title = extract_chapter_title(english_content, 'english')
    
    # Get AI translation (with caching)
    ai_translation = translate_with_gemini(raw_content, api_key, use_cache=True)
    
    if "API Request Failed" in ai_translation:
        print(f"⚠️  Chapter {chapter_num}: AI translation failed - {ai_translation}")
        ai_translation = ""
    
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
    print("🚀 Building Complete Dataset and Report")
    print("=" * 50)
    
    # Check for API key
    api_key = input("🔑 Enter your Gemini API key: ").strip()
    if not api_key:
        print("❌ API key required. Exiting.")
        return
    
    # Load alignment map
    print("\n📋 Loading alignment map...")
    try:
        alignment_map = load_alignment_map()
        print(f"✅ Loaded alignment map with {len(alignment_map)} chapters")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("💡 Run 'python build_complete_alignment_map.py' first")
        return
    
    # Load semantic model if available
    print("\n🧠 Loading similarity model...")
    semantic_model = load_semantic_model() if SEMANTIC_AVAILABLE else None
    similarity_cache = load_similarity_cache()
    
    # Process parameters
    process_all = input("\n🎯 Process all chapters? (y/n, default=n): ").strip().lower() == 'y'
    
    if process_all:
        chapters_to_process = sorted([int(k) for k in alignment_map.keys()])
    else:
        max_chapters = int(input("📊 How many chapters to process? (default=50): ").strip() or "50")
        chapters_to_process = sorted([int(k) for k in alignment_map.keys()])[:max_chapters]
    
    print(f"\n🔄 Processing {len(chapters_to_process)} chapters...")
    
    # Process chapters
    processed_chapters = []
    start_time = time.time()
    
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
            print("✅")
        else:
            print("⏭️")
        
        # Rate limiting
        time.sleep(0.5)
    
    # Save updated cache
    save_similarity_cache(similarity_cache)
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Processing completed in {elapsed_time:.1f} seconds")
    print(f"✅ Successfully processed: {len(processed_chapters)} chapters")
    
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create CSV report
    csv_output = os.path.join(EXPORT_DIR, f"dataset_report_{timestamp}.csv")
    df = create_csv_report(processed_chapters, csv_output)
    
    # Create JSONL training files
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