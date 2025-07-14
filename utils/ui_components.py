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

# Import configuration
from .config import EVALUATIONS_DIR

# Streamlit import with fallback
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False
    # Mock streamlit for testing
    class MockStreamlit:
        def components(self):
            class MockComponents:
                def html(self, *args, **kwargs):
                    return None
            return MockComponents()
    st = MockStreamlit()


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
    
    # Apply highlighting from end to start to maintain character positions
    for comment in sorted_comments:
        start = comment['start_offset']
        end = comment['end_offset']
        dimension = comment.get('dimension', 'unknown')
        color = dimension_colors.get(dimension, '#f0f0f0')
        
        # Create highlight span
        highlighted_section = f'<span style="background-color: {color}; padding: 1px 2px; border-radius: 2px;">{highlighted_text[start:end]}</span>'
        
        # Replace the section
        highlighted_text = highlighted_text[:start] + highlighted_section + highlighted_text[end:]
    
    return highlighted_text


def save_inline_comments(style_name, chapter_id, comments):
    """Save inline comments for a specific chapter and style.
    
    Args:
        style_name: Translation style identifier
        chapter_id: Chapter identifier
        comments: List of comment dictionaries
    
    Returns:
        bool: Success status
    """
    print("="*60)
    print("🔍 SAVE START: save_inline_comments function called")
    print(f"🔍 SAVE PARAMS: style_name='{style_name}', chapter_id='{chapter_id}'")
    print(f"🔍 SAVE DATA: {len(comments)} comments to save")
    print(f"🔍 EVALUATIONS_DIR: {EVALUATIONS_DIR}")
    
    # Log each comment being saved
    for i, comment in enumerate(comments):
        print(f"🔍 COMMENT {i+1}: '{comment.get('selected_text', '')[:50]}...'")
        print(f"   ├─ dimension: {comment.get('dimension', 'unknown')}")
        print(f"   ├─ start_char: {comment.get('start_char', 'unknown')}")
        print(f"   └─ comment: '{comment.get('comment', '')[:30]}...'")
    
    try:
        # Create directory structure
        style_dir = os.path.join(EVALUATIONS_DIR, style_name)
        os.makedirs(style_dir, exist_ok=True)
        print(f"🔍 STYLE DIR CREATED: {style_dir}")
        
        # Save comments to file
        comments_file = os.path.join(style_dir, f'inline_comments_ch{chapter_id}.json')
        print(f"🔍 SAVING TO: {comments_file}")
        
        with open(comments_file, 'w', encoding='utf-8') as f:
            json.dump(comments, f, indent=2, ensure_ascii=False)
        
        print(f"🔍 SAVE SUCCESS: Comments saved to {comments_file}")
        print(f"🔍 FILE SIZE: {os.path.getsize(comments_file)} bytes")
        return True
        
    except Exception as e:
        print(f"🔍 SAVE ERROR: {e}")
        return False


def load_inline_comments(style_name, chapter_id):
    """Load inline comments for a specific chapter and style.
    
    Args:
        style_name: Translation style identifier
        chapter_id: Chapter identifier
    
    Returns:
        list: List of comment dictionaries
    """
    print(f"🔍 LOAD START: style_name='{style_name}', chapter_id='{chapter_id}'")
    print(f"🔍 EVALUATIONS_DIR: {EVALUATIONS_DIR}")
    
    style_dir = os.path.join(EVALUATIONS_DIR, style_name)
    comments_file = os.path.join(style_dir, f'inline_comments_ch{chapter_id}.json')
    
    print(f"🔍 STYLE DIR: {style_dir} (exists: {os.path.exists(style_dir)})")
    print(f"🔍 COMMENTS FILE: {comments_file} (exists: {os.path.exists(comments_file)})")
    
    if os.path.exists(comments_file):
        try:
            file_size = os.path.getsize(comments_file)
            print(f"🔍 FILE SIZE: {file_size} bytes")
            
            with open(comments_file, 'r', encoding='utf-8') as f:
                comments = json.load(f)
            
            print(f"🔍 LOAD SUCCESS: {len(comments)} comments loaded")
            
            # Log first few comments for debugging
            for i, comment in enumerate(comments[:3]):
                print(f"🔍 LOADED COMMENT {i+1}: '{comment.get('selected_text', '')[:30]}...'")
                print(f"   ├─ dimension: {comment.get('dimension', 'unknown')}")
                print(f"   └─ timestamp: {comment.get('timestamp', 'unknown')}")
            
            return comments
            
        except Exception as e:
            print(f"🔍 LOAD ERROR: Failed to load comments: {e}")
            return []
    else:
        print(f"🔍 LOAD RESULT: No comments file found, returning empty list")
        return []


def add_inline_comment(style_name, chapter_id, comment_data):
    """Add a new inline comment to existing comments.
    
    Args:
        style_name: Translation style identifier
        chapter_id: Chapter identifier
        comment_data: New comment dictionary
    
    Returns:
        bool: Success status
    """
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
        new_comment = {
            **comment_data,
            'timestamp': datetime.now().isoformat(),
            'id': f"comment_{len(comments) + 1}_{int(datetime.now().timestamp())}"
        }
        
        comments.append(new_comment)
        print("🔍 DEBUG: New comment added, total comments:", len(comments))
        
        # Save updated comments
        print("🔍 DEBUG: Saving updated comments")
        success = save_inline_comments(style_name, chapter_id, comments)
        print("🔍 DEBUG: Save result:", success)
        
        return success
        
    except Exception as e:
        print(f"🔍 DEBUG ERROR: Failed to add comment: {e}")
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


def create_synchronized_text_display(left_text, right_text, left_title="Left Text", right_title="Right Text", 
                                   height=400, full_width=True, enable_comments=False, chapter_id=None, 
                                   style_name=None, key=None):
    """Create a synchronized scrolling display for two text blocks with optional inline commenting.
    
    Args:
        left_text: Text content for left panel
        right_text: Text content for right panel  
        left_title: Title for left panel
        right_title: Title for right panel
        height: Height of display panels in pixels
        full_width: If True, optimize for full width with minimal margins
        enable_comments: If True, enable text selection and inline commenting
        chapter_id: Chapter identifier for comment storage
        style_name: Translation style name for comment storage
        key: Unique key for the component
    
    Returns:
        dict: Selection event data when text is selected for commenting
    """
    if not STREAMLIT_AVAILABLE:
        return None
    
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
    else:
        right_escaped = escape_html(right_text)
    
    # Always escape left text (original)
    left_escaped = escape_html(left_text)
    
    # Generate existing comments display
    existing_comments_html = generate_existing_comments_html(existing_comments) if enable_comments else ""
    
    # Calculate optimal widths based on full_width setting
    if full_width:
        # For full-width displays, use maximum available space
        container_style = 'width: 100%; max-width: none; margin: 0;'
        panel_width = '49%'  # Slightly less than 50% to account for gap
    else:
        # For normal displays, use constrained width
        container_style = 'width: 100%; max-width: 1200px; margin: 0 auto;'
        panel_width = '48%'
    
    # Define the selection handler script
    selection_script = ""
    if enable_comments:
        selection_script = f"""
        let selectedData = null;
        
        function handleTextSelection() {{
            const selection = window.getSelection();
            if (selection.toString().length > 0) {{
                const range = selection.getRangeAt(0);
                const rightPanel = document.getElementById('rightPanel');
                
                // Check if selection is in right panel
                if (rightPanel && rightPanel.contains(range.commonAncestorContainer)) {{
                    // Calculate character offset in the original text
                    const textContent = rightPanel.textContent || rightPanel.innerText;
                    const selectedText = selection.toString();
                    
                    // Simple offset calculation
                    const beforeSelection = range.startContainer.textContent.substring(0, range.startOffset);
                    const startOffset = textContent.indexOf(selectedText);
                    
                    selectedData = {{
                        text: selectedText,
                        start_offset: startOffset,
                        end_offset: startOffset + selectedText.length,
                        chapter_id: '{chapter_id}',
                        style_name: '{style_name}'
                    }};
                    
                    // Send to Streamlit
                    window.parent.postMessage({{
                        type: 'streamlit:selection',
                        data: selectedData
                    }}, '*');
                }}
            }}
        }}
        
        document.addEventListener('mouseup', handleTextSelection);
        document.addEventListener('touchend', handleTextSelection);
        """
    
    # Create the synchronized display HTML
    html_content = f"""
    <div style="{container_style}">
        {existing_comments_html}
        
        <div style="display: flex; gap: 2%; height: {height}px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
            <!-- Left Panel -->
            <div style="width: {panel_width}; display: flex; flex-direction: column;">
                <div style="background: #f8f9fa; padding: 8px 12px; border-bottom: 1px solid #dee2e6; font-weight: 600; font-size: 14px; color: #495057;">
                    {left_title}
                </div>
                <div style="flex: 1; padding: 16px; overflow-y: auto; background: white; border: 1px solid #dee2e6; border-top: none; line-height: 1.6; font-size: 14px; color: #212529;" 
                     id="leftPanel" class="sync-panel">
                    {left_escaped}
                </div>
            </div>
            
            <!-- Right Panel -->
            <div style="width: {panel_width}; display: flex; flex-direction: column;">
                <div style="background: #f8f9fa; padding: 8px 12px; border-bottom: 1px solid #dee2e6; font-weight: 600; font-size: 14px; color: #495057;">
                    {right_title}
                </div>
                <div style="flex: 1; padding: 16px; overflow-y: auto; background: white; border: 1px solid #dee2e6; border-top: none; line-height: 1.6; font-size: 14px; color: #212529; user-select: {'text' if enable_comments else 'none'};" 
                     id="rightPanel" class="sync-panel">
                    {right_escaped}
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Synchronized scrolling
        const panels = document.querySelectorAll('.sync-panel');
        let isScrolling = false;
        
        panels.forEach(panel => {{
            panel.addEventListener('scroll', function() {{
                if (isScrolling) return;
                isScrolling = true;
                
                const scrollRatio = this.scrollTop / (this.scrollHeight - this.clientHeight);
                
                panels.forEach(otherPanel => {{
                    if (otherPanel !== this) {{
                        const targetScroll = scrollRatio * (otherPanel.scrollHeight - otherPanel.clientHeight);
                        otherPanel.scrollTop = targetScroll;
                    }}
                }});
                
                setTimeout(() => {{ isScrolling = false; }}, 50);
            }});
        }});
        
        {selection_script}
    </script>
    """
    
    # Render with streamlit
    if key is None:
        key = f"sync_display_{hash(left_text[:100] + right_text[:100])}"
    
    return st.components.v1.html(html_content, height=height + 100, scrolling=True)