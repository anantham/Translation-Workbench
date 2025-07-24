

"""
UI Components and Streamlit Interface Module

Streamlit-specific UI helpers, interactive components, and display utilities.
Provides reusable components for the translation workbench interface.

This module handles:
- Synchronized text display with commenting system
- Inline comment management and highlighting
- HTML escaping and text processing
- Interactive UI components for Streamlit apps
- Comment visualization and storage
- Text selection and annotation interfaces
"""

import os
import json
from datetime import datetime
from pathlib import Path

# Import configuration
from .config import EVALUATIONS_DIR

# Streamlit import with fallback
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

def escape_html(text):
    """Escape HTML special characters in text.
    
    Args:
        text: Text to escape
    
    Returns:
        str: HTML-escaped text
    """
    return (text.replace('&', '&amp;')
               .replace('<', '&lt;')
               .replace('>', '&gt;')
               .replace('"', '&quot;')
               .replace("'", '&#x27;')
               .replace('\n', '<br>'))

def apply_comment_highlighting(text, comments):
    """Apply HTML highlighting to text based on inline comments.
    
    Args:
        text: Original text content
        comments: List of comment dictionaries with positions
    
    Returns:
        str: Text with HTML highlighting spans
    """
    if not comments:
        return text
    
    sorted_comments = sorted(comments, key=lambda c: c['start_offset'], reverse=True)
    
    dimension_colors = {
        'vocabulary_complexity': '#fff3cd',
        'cultural_context': '#cff4fc',
        'prose_style': '#f8d7da',
        'creative_fidelity': '#d1e7dd',
        'english_sophistication': '#fff3cd',
        'world_building': '#cff4fc',
        'emotional_impact': '#f8d7da',
        'dialogue_naturalness': '#d1e7dd'
    }
    
    highlighted_text = text
    
    for comment in sorted_comments:
        start = comment['start_offset']
        end = comment['end_offset']
        dimension = comment.get('dimension', 'unknown')
        color = dimension_colors.get(dimension, '#f0f0f0')
        
        highlighted_section = f'<span style="background-color: {color}; padding: 1px 2px; border-radius: 2px;">{highlighted_text[start:end]}</span>'
        
        highlighted_text = highlighted_text[:start] + highlighted_section + highlighted_text[end:]
    
    return highlighted_text

def save_inline_comments(style_name, chapter_id, comments):
    """Save inline comments for a specific chapter and style."""
    try:
        style_dir = os.path.join(EVALUATIONS_DIR, style_name)
        os.makedirs(style_dir, exist_ok=True)
        comments_file = os.path.join(style_dir, f'inline_comments_ch{chapter_id}.json')
        with open(comments_file, 'w', encoding='utf-8') as f:
            json.dump(comments, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        return False

def load_inline_comments(style_name, chapter_id):
    """Load inline comments for a specific chapter and style."""
    style_dir = os.path.join(EVALUATIONS_DIR, style_name)
    comments_file = os.path.join(style_dir, f'inline_comments_ch{chapter_id}.json')
    if os.path.exists(comments_file):
        try:
            with open(comments_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def add_inline_comment(style_name, chapter_id, comment_data):
    """Add a new inline comment to existing comments."""
    try:
        comments = load_inline_comments(style_name, chapter_id)
        new_comment = {
            **comment_data,
            'timestamp': datetime.now().isoformat(),
            'id': f"comment_{len(comments) + 1}_{int(datetime.now().timestamp())}"
        }
        comments.append(new_comment)
        return save_inline_comments(style_name, chapter_id, comments)
    except Exception:
        return False

def generate_existing_comments_html(comments):
    """Generate HTML display for existing comments.
    
    Args:
        comments: List of comment dictionaries
    
    Returns:
        str: HTML string for displaying comments
    """
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
        dimension = comment.get('dimension', 'unknown')
        config = dimension_config.get(dimension, {'icon': '📝', 'color': '#f5f5f5', 'name': 'Other'})
        
        selected_text = comment.get('selected_text', '')[:50]
        comment_text = comment.get('comment', '')[:80]
        
        comments_html += f'''
        <div style="background: {config['color']}; padding: 6px; margin: 3px 0; border-radius: 4px; font-size: 11px;">
            <div style="font-weight: 600; color: #333;">
                {config['icon']} {config['name']}: "{selected_text}..."
            </div>
            <div style="color: #666; margin-top: 2px;">
                {comment_text}{'...' if len(comment.get('comment', '')) > 80 else ''}
            </div>
        </div>
        '''
    
    comments_html += '</div>'
    return comments_html

# Function removed - replaced by feedback_ui.render_feedback_reader()
# Kept for backward compatibility in feedback_ui.py
