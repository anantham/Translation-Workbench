"""
Evaluation and Quality Assessment Module

Similarity scoring, BLEU calculation, and comprehensive quality metrics.
Translation style evaluation and comparative analysis capabilities.

This module handles:
- Semantic similarity calculation using BERT/sentence-transformers
- Syntactic similarity fallback when semantic models unavailable
- BLEU score calculation for translation quality assessment
- Comprehensive translation quality evaluation
- Translation style comparison and analysis
- BERT score calculation for different translation styles
"""

import os
import json
from difflib import SequenceMatcher

# Import caching functions from our caching module
from .caching import load_similarity_cache, save_similarity_cache, generate_text_hash

# Import data management functions
from .data_management import load_chapter_content, get_text_statistics

# Import config for data directory paths
from .config import DATA_DIR

# Optional dependencies detection
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

# Data paths
TRANSLATIONS_DIR = os.path.join(DATA_DIR, "custom_translations")
EVALUATIONS_DIR = os.path.join(DATA_DIR, "evaluations")


def load_semantic_model():
    """Load the semantic similarity model with error handling."""
    if not SEMANTIC_AVAILABLE:
        return None
    
    try:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        return model
    except Exception as e:
        print(f"Warning: Could not load semantic model: {e}")
        return None


def calculate_similarity(text1, text2, model=None, cache=None):
    """Calculate similarity between two texts.
    
    Args:
        text1: First text for comparison
        text2: Second text for comparison  
        model: Optional pre-loaded SentenceTransformer model
        cache: Optional pre-loaded similarity cache
    
    Returns:
        float: Similarity score between 0.0 and 1.0
    """
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
    """Fallback syntactic similarity for when semantic models aren't available.
    
    Args:
        text1: First text for comparison
        text2: Second text for comparison
    
    Returns:
        float: Syntactic similarity score between 0.0 and 1.0
    """
    # Length similarity
    len1, len2 = len(text1), len(text2)
    length_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
    
    # Content similarity (first 1000 chars for speed)
    sample1 = text1[:1000].lower().replace('\n', ' ')
    sample2 = text2[:1000].lower().replace('\n', ' ')
    content_similarity = SequenceMatcher(None, sample1, sample2).ratio()
    
    # Combined score
    return (length_ratio * 0.3) + (content_similarity * 0.7)


def calculate_bleu_score(reference, candidate):
    """Calculate BLEU score between reference and candidate translations.
    
    Args:
        reference: Reference translation text
        candidate: Candidate translation text
    
    Returns:
        float: BLEU score between 0.0 and 1.0
    """
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
    """Comprehensive evaluation of translation quality.
    
    Args:
        raw_text: Original Chinese text
        reference_translation: Reference English translation
        candidate_translation: Candidate English translation to evaluate
        model: Optional pre-loaded semantic model
    
    Returns:
        dict: Comprehensive quality metrics
    """
    results = {}
    
    # BLEU score
    results['bleu_score'] = calculate_bleu_score(reference_translation, candidate_translation)
    
    # Semantic similarity
    results['semantic_similarity'] = calculate_similarity(reference_translation, candidate_translation, model)
    
    # Length comparison
    ref_stats = get_text_statistics(reference_translation, 'english')
    cand_stats = get_text_statistics(candidate_translation, 'english')
    results['length_ratio'] = cand_stats['word_count'] / ref_stats['word_count'] if ref_stats['word_count'] > 0 else 0
    
    # Raw content stats for context
    raw_stats = get_text_statistics(raw_text, 'chinese')
    results['raw_stats'] = raw_stats
    results['reference_stats'] = ref_stats
    results['candidate_stats'] = cand_stats
    
    return results


def get_available_translation_styles():
    """Scan the custom_translations directory and return available translation styles.
    
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
        
        # Extract chapter numbers
        chapter_numbers = []
        for filename in translation_files:
            try:
                # Extract chapter number from filename like "Chapter-0001-translated.txt"
                chapter_str = filename.split('-')[1]
                chapter_num = int(chapter_str)
                chapter_numbers.append(chapter_num)
            except (ValueError, IndexError):
                continue
        
        chapter_numbers.sort()
        
        # Load metadata if available
        metadata_path = os.path.join(style_path, "job_metadata.json")
        metadata = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        style_info = {
            'name': style_name,
            'path': style_path,
            'chapter_count': len(chapter_numbers),
            'chapters': chapter_numbers,
            'min_chapter': min(chapter_numbers) if chapter_numbers else 0,
            'max_chapter': max(chapter_numbers) if chapter_numbers else 0,
            'metadata': metadata
        }
        
        styles.append(style_info)
    
    # Sort by chapter count (most chapters first)
    styles.sort(key=lambda x: x['chapter_count'], reverse=True)
    
    return styles


def calculate_bert_scores_for_style(style_info, alignment_map, progress_callback=None):
    """Calculate BERT similarity scores for a translation style against reference.
    
    Args:
        style_info: Style information dictionary from get_available_translation_styles
        alignment_map: Chapter alignment mapping
        progress_callback: Optional callback function for progress updates
    
    Returns:
        dict: Chapter number -> BERT similarity score mapping
    """
    bert_scores = {}
    semantic_model = load_semantic_model()
    
    if not semantic_model:
        print("Warning: Semantic model not available, skipping BERT score calculation")
        return bert_scores
    
    style_chapters = style_info['chapters']
    processed = 0
    
    for chapter_num in style_chapters:
        chapter_key = str(chapter_num)
        
        if chapter_key not in alignment_map:
            continue
        
        chapter_data = alignment_map[chapter_key]
        
        # Load reference English translation
        reference_file = chapter_data.get('english_file')
        if not reference_file:
            continue
        
        reference_content = load_chapter_content(reference_file)
        if "File not found" in reference_content:
            continue
        
        # Load style translation
        style_file = f"Chapter-{chapter_num:04d}-translated.txt"
        style_path = os.path.join(style_info['path'], style_file)
        style_content = load_chapter_content(style_path)
        
        if "File not found" in style_content:
            continue
        
        # Calculate BERT similarity
        similarity = calculate_similarity(reference_content, style_content, semantic_model)
        bert_scores[chapter_num] = similarity
        
        processed += 1
        if progress_callback:
            progress_callback(processed, len(style_chapters), f"Chapter {chapter_num}")
    
    return bert_scores


def calculate_composite_score(bert_scores, human_scores, chapter_count):
    """Calculate composite evaluation score combining BERT and human evaluations.
    
    Args:
        bert_scores: Dictionary of chapter -> BERT score
        human_scores: Dictionary of chapter -> human score
        chapter_count: Total number of chapters being evaluated
    
    Returns:
        dict: Composite scoring metrics
    """
    # Calculate averages
    bert_values = list(bert_scores.values())
    human_values = list(human_scores.values())
    
    avg_bert = sum(bert_values) / len(bert_values) if bert_values else 0
    avg_human = sum(human_values) / len(human_values) if human_values else 0
    
    # Coverage metrics
    bert_coverage = len(bert_scores) / chapter_count if chapter_count > 0 else 0
    human_coverage = len(human_scores) / chapter_count if chapter_count > 0 else 0
    
    # Weighted composite score (weight BERT and human equally if both available)
    if bert_values and human_values:
        composite_score = (avg_bert * 0.5) + (avg_human * 0.5)
        confidence = min(bert_coverage, human_coverage)
    elif bert_values:
        composite_score = avg_bert
        confidence = bert_coverage
    elif human_values:
        composite_score = avg_human  
        confidence = human_coverage
    else:
        composite_score = 0
        confidence = 0
    
    return {
        'composite_score': composite_score,
        'avg_bert_score': avg_bert,
        'avg_human_score': avg_human,
        'bert_coverage': bert_coverage,
        'human_coverage': human_coverage,
        'confidence': confidence,
        'total_evaluations': len(bert_values) + len(human_values),
        'methodology': 'BERT + Human' if bert_values and human_values else 'BERT only' if bert_values else 'Human only'
    }


def load_bert_scores_from_reports():
    """Load BERT scores from existing evaluation reports.
    
    Returns:
        dict: Chapter number -> BERT score mapping
    """
    bert_scores = {}
    
    if not os.path.exists(EVALUATIONS_DIR):
        return bert_scores
    
    # Look for BERT score files in evaluation reports
    for eval_name in os.listdir(EVALUATIONS_DIR):
        eval_path = os.path.join(EVALUATIONS_DIR, eval_name)
        
        if not os.path.isdir(eval_path):
            continue
        
        bert_file = os.path.join(eval_path, "bert_scores.json")
        if os.path.exists(bert_file):
            try:
                with open(bert_file, 'r', encoding='utf-8') as f:
                    scores = json.load(f)
                    # Merge scores (later evaluations override earlier ones)
                    bert_scores.update(scores)
            except (json.JSONDecodeError, IOError):
                continue
    
    return bert_scores


def get_chunking_statistics(training_data):
    """Analyze chunking statistics for the training data.
    
    Args:
        training_data: List of training examples
    
    Returns:
        dict: Chunking statistics and analysis
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