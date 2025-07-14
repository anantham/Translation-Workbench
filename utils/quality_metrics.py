"""
Quality Metrics and Statistics Module

Statistical analysis of translation quality and text properties.
Word counts, quality metrics, and comparative analysis.

This module handles:
- Chinese character and English word counting
- Translation expansion ratio calculation
- Average quality metrics from evaluation files
- Consistency scoring based on standard deviation
"""

import os
import json
import glob
import statistics


def calculate_word_counts(translation_dir):
    """Calculate word counts for Chinese source and English translation files."""
    # Initialize counters
    chinese_char_count = 0
    english_word_count = 0
    
    # Find Chinese files (assuming they're in a parallel directory structure)
    chinese_dir = os.path.join(os.path.dirname(translation_dir), "..", "novel_content_dxmwx")
    if os.path.exists(chinese_dir):
        chinese_files = glob.glob(os.path.join(chinese_dir, "*.txt"))
        for file_path in chinese_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Count Chinese characters (excluding whitespace and punctuation)
                    chinese_chars = len([char for char in content if '\u4e00' <= char <= '\u9fff'])
                    chinese_char_count += chinese_chars
            except Exception as e:
                print(f"Warning: Could not read Chinese file {file_path}: {e}")
    
    # Count English words in translation directory
    if os.path.exists(translation_dir):
        english_files = glob.glob(os.path.join(translation_dir, "*.txt"))
        for file_path in english_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Split by whitespace and count words
                    words = content.split()
                    english_word_count += len(words)
            except Exception as e:
                print(f"Warning: Could not read English file {file_path}: {e}")
    
    # Calculate expansion ratio
    expansion_ratio = "Unknown"
    if chinese_char_count > 0 and english_word_count > 0:
        ratio = english_word_count / chinese_char_count
        expansion_ratio = f"{ratio:.2f}"
    
    return {
        'chinese_characters': chinese_char_count,
        'english_words': english_word_count, 
        'expansion_ratio': expansion_ratio
    }


def calculate_average_quality_metrics(translation_dir):
    """Calculate average quality metrics from existing evaluation data."""
    # Look for evaluation files in the translation directory
    evaluation_files = []
    if os.path.exists(translation_dir):
        for file in os.listdir(translation_dir):
            if file.endswith('_evaluation.json'):
                evaluation_files.append(os.path.join(translation_dir, file))
    
    if not evaluation_files:
        return {
            'avg_bleu_score': 'No evaluations available',
            'avg_semantic_similarity': 'No evaluations available',
            'consistency_score': 'No evaluations available'
        }
    
    # Collect all scores
    bleu_scores = []
    semantic_scores = []
    
    for eval_file in evaluation_files:
        try:
            with open(eval_file, 'r', encoding='utf-8') as f:
                eval_data = json.load(f)
                if 'bleu_score' in eval_data and eval_data['bleu_score'] is not None:
                    bleu_scores.append(eval_data['bleu_score'])
                if 'semantic_similarity' in eval_data and eval_data['semantic_similarity'] is not None:
                    semantic_scores.append(eval_data['semantic_similarity'])
        except Exception as e:
            print(f"Warning: Could not read evaluation file {eval_file}: {e}")
    
    # Calculate averages
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else None
    avg_semantic = sum(semantic_scores) / len(semantic_scores) if semantic_scores else None
    
    # Calculate consistency score (based on standard deviation of semantic scores)
    consistency = "High"
    if len(semantic_scores) > 1:
        std_dev = statistics.stdev(semantic_scores)
        if std_dev > 0.15:
            consistency = "Medium"
        elif std_dev > 0.25:
            consistency = "Low"
    
    return {
        'avg_bleu_score': f"{avg_bleu:.3f}" if avg_bleu is not None else 'No data available',
        'avg_semantic_similarity': f"{avg_semantic:.3f}" if avg_semantic is not None else 'No data available', 
        'consistency_score': f"{consistency} (σ={statistics.stdev(semantic_scores):.3f})" if len(semantic_scores) > 1 else consistency
    }