"""
Miscellaneous Utilities Module

Collection of utility functions that don't fit into other specialized modules.
Includes text statistics, alignment management, scoring persistence, and helper functions.

This module handles:
- Text statistics and language-aware counting
- Alignment map backup and management
- BERT and human score persistence
- Data formatting and conversion utilities
- Binary search for misalignment detection
- Systematic correction previews
- Web scraping helper functions
"""

import os
import json
import re
import shutil
from datetime import datetime
from urllib.parse import urlparse

# Import configuration and other modules
from .config import DATA_DIR, EVALUATIONS_DIR
from .evaluation import load_semantic_model, calculate_similarity
from .data_management import load_chapter_content


def get_text_stats(content, language_hint=None):
    """Get comprehensive text statistics with language-aware counting.
    
    Args:
        content: Text content to analyze
        language_hint: Optional language hint ('chinese' or 'english')
    
    Returns:
        dict: Text statistics including char count, word count, etc.
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
        if char_count > 0:
            language_hint = 'chinese' if cjk_chars / char_count > 0.2 else 'english'
        else:
            language_hint = 'english'
    
    # Language-aware word counting
    if language_hint == 'chinese':
        # For Chinese, count characters as words (approximation)
        word_count = len([char for char in content if '\u4e00' <= char <= '\u9fff'])
    else:
        # For English and other languages, split by whitespace
        word_count = len(content.split())
    
    # Calculate averages
    avg_words_per_line = word_count / line_count if line_count > 0 else 0
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'line_count': line_count,
        'avg_words_per_line': avg_words_per_line,
        'language': language_hint
    }


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
        shutil.copy(output_file, backup_path)
    
    # Save the new alignment map
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(alignment_map, f, indent=2, ensure_ascii=False)
    
    return backup_filename


def format_example(example, format_type, system_prompt=None):
    """Format a single training example according to the specified format.
    
    Args:
        example: Training example dictionary
        format_type: Format type ('OpenAI Fine-tuning', 'Gemini Fine-tuning', etc.)
        system_prompt: Optional system prompt
    
    Returns:
        dict: Formatted example
    """
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


def save_bert_scores(style_name, bert_scores):
    """Save BERT scores to persistent storage.
    
    Args:
        style_name: Translation style identifier
        bert_scores: Dictionary of chapter -> BERT score
    """
    style_eval_dir = os.path.join(EVALUATIONS_DIR, style_name)
    os.makedirs(style_eval_dir, exist_ok=True)
    
    bert_file = os.path.join(style_eval_dir, 'bert_scores.json')
    with open(bert_file, 'w', encoding='utf-8') as f:
        json.dump(bert_scores, f, indent=2, ensure_ascii=False)


def load_bert_scores(style_name):
    """Load BERT scores from persistent storage.
    
    Args:
        style_name: Translation style identifier
    
    Returns:
        dict: Dictionary of chapter -> BERT score
    """
    bert_file = os.path.join(EVALUATIONS_DIR, style_name, 'bert_scores.json')
    if os.path.exists(bert_file):
        try:
            with open(bert_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_human_scores(style_name, human_scores):
    """Save human evaluation scores to persistent storage.
    
    Args:
        style_name: Translation style identifier
        human_scores: Dictionary of chapter -> human scores
    """
    style_eval_dir = os.path.join(EVALUATIONS_DIR, style_name)
    os.makedirs(style_eval_dir, exist_ok=True)
    
    human_file = os.path.join(style_eval_dir, 'human_scores.json')
    with open(human_file, 'w', encoding='utf-8') as f:
        json.dump(human_scores, f, indent=2, ensure_ascii=False)


def load_human_scores(style_name):
    """Load human evaluation scores from persistent storage.
    
    Args:
        style_name: Translation style identifier
    
    Returns:
        dict: Dictionary of chapter -> human scores
    """
    human_file = os.path.join(EVALUATIONS_DIR, style_name, 'human_scores.json')
    if os.path.exists(human_file):
        try:
            with open(human_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def find_first_misalignment_binary_search(alignment_map, api_key, min_chapter, max_chapter, threshold):
    """Use binary search to find the first misaligned chapter based on semantic similarity.
    
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
                'search_range': 'COMPLETE'
            })
        else:
            search_log.append({
                'chapter': 'RESULT',
                'action': f'No misaligned chapters found in range [{min_chapter}, {max_chapter}]',
                'search_range': 'COMPLETE'
            })
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def preview_systematic_correction(alignment_map, offset, sample_size=10):
    """Preview the effects of applying a systematic offset correction to the alignment map.
    
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
            if new_chapter_num > 0:  # Only positive chapter numbers are valid
                affected_chapters.append(chapter_num)
        
        preview_data['total_affected_chapters'] = len(affected_chapters)
        
        # Create sample of affected chapters
        sample_chapters = affected_chapters[:sample_size] if len(affected_chapters) >= sample_size else affected_chapters
        
        for chapter_num in sample_chapters:
            chapter_data = alignment_map[str(chapter_num)]
            new_chapter_num = chapter_num + offset
            
            sample_entry = {
                'original_chapter': chapter_num,
                'new_chapter': new_chapter_num,
                'raw_file': chapter_data.get('raw_file', 'N/A'),
                'english_file': chapter_data.get('english_file', 'N/A'),
                'valid_change': new_chapter_num > 0
            }
            
            # Load and preview content if files exist
            try:
                raw_content = load_chapter_content(chapter_data.get('raw_file', ''))
                english_content = load_chapter_content(chapter_data.get('english_file', ''))
                
                if "File not found" not in raw_content:
                    sample_entry['raw_preview'] = raw_content[:100] + "..." if len(raw_content) > 100 else raw_content
                if "File not found" not in english_content:
                    sample_entry['english_preview'] = english_content[:100] + "..." if len(english_content) > 100 else english_content
                    
            except Exception as e:
                sample_entry['preview_error'] = str(e)
            
            preview_data['samples'].append(sample_entry)
        
        preview_data['success'] = True
        
    except Exception as e:
        preview_data['error'] = str(e)
    
    return preview_data


# Web scraping utility functions
def validate_scraping_url(url):
    """Validate and analyze a URL for scraping compatibility.
    
    Args:
        url: URL to validate
    
    Returns:
        dict: Validation results with recommendations
    """
    result = {
        'valid': False,
        'site_type': 'Unknown',
        'recommendations': [],
        'warnings': []
    }
    
    if not url:
        result['warnings'].append("❌ URL is empty")
        return result
    
    try:
        parsed = urlparse(url)
        
        # Check for supported sites
        if 'dxmwx.org' in parsed.netloc:
            result['valid'] = True
            result['site_type'] = 'DXMWX (Supported)'
            result['recommendations'].append("✅ This site is supported for scraping")
        else:
            result['site_type'] = 'Unsupported'
            result['recommendations'].append("📖 Consider using supported sites: dxmwx.org")
        
        # General URL checks
        if not url.startswith(('http://', 'https://')):
            result['warnings'].append("⚠️ URL should start with http:// or https://")
        
        if parsed.scheme == 'http':
            result['warnings'].append("🔒 Consider using HTTPS for better security")
            
    except Exception as e:
        result['warnings'].append(f"❌ URL parsing error: {str(e)}")
    
    return result


def extract_chapter_number(title):
    """Extract chapter number from title text.
    
    Args:
        title: Chapter title text
    
    Returns:
        int: Chapter number if found, None otherwise
    """
    if not title:
        return None
    
    # Try various patterns for chapter numbers
    patterns = [
        r'第?(\d+)章',  # Chinese: 第123章 or 123章
        r'Chapter\s*(\d+)',  # English: Chapter 123
        r'Ch\.?\s*(\d+)',  # Abbreviated: Ch. 123 or Ch 123
        r'(\d+)',  # Just numbers
    ]
    
    for pattern in patterns:
        match = re.search(pattern, title, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    
    return None


def sanitize_filename(filename):
    """Sanitize filename for safe file system usage.
    
    Args:
        filename: Original filename
    
    Returns:
        str: Sanitized filename
    """
    if not filename:
        return "untitled"
    
    # Remove or replace invalid characters
    invalid_chars = r'[\\/*?:"<>|]'
    sanitized = re.sub(invalid_chars, "", filename)
    
    # Remove extra whitespace and limit length
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    
    # Limit filename length (keep some buffer for extensions)
    max_length = 200
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].strip()
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = "untitled"
    
    return sanitized