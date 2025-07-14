"""
Data Management and Alignment Module

Chapter loading, alignment map management, and data quality tools.
Binary search misalignment detection and surgical correction capabilities.

This module handles:
- Chapter content loading from files
- Alignment map loading with session state persistence
- Text statistics with language-aware word counting
- Alignment map saving with backup functionality
- Data quality and statistics analysis
"""

import os
import json
from datetime import datetime

# Graceful import of streamlit (may not be available in all environments)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    # Create a mock st object for non-Streamlit environments
    class MockStreamlit:
        class session_state:
            pass
        @staticmethod
        def error(msg):
            print(f"Error: {msg}")
    st = MockStreamlit()

# Data directory structure
DATA_DIR = "data"


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


def save_alignment_map(alignment_map, output_file="alignment_map.json"):
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


def get_text_statistics(content, language_hint=None):
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


def get_chapter_word_count(chapter_number, alignment_map):
    """Get word count for a specific chapter."""
    if not alignment_map or str(chapter_number) not in alignment_map:
        return {'raw': 0, 'english': 0}
    
    chapter_data = alignment_map[str(chapter_number)]
    
    # Load and count words in raw content
    raw_content = load_chapter_content(chapter_data.get('raw_file', ''))
    raw_stats = get_text_statistics(raw_content, 'chinese')
    
    # Load and count words in English content  
    english_content = load_chapter_content(chapter_data.get('english_file', ''))
    english_stats = get_text_statistics(english_content, 'english')
    
    return {
        'raw': raw_stats['word_count'],
        'english': english_stats['word_count']
    }


# Note: Complex functions like find_first_misalignment_binary_search, 
# apply_surgical_correction, apply_systematic_correction, and 
# split_chapter_content are large and depend on additional modules.
# They can be added in a future iteration once more dependencies are resolved.