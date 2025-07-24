"""
Clean Streamlit-native feedback UI for translation evaluation.
Replaces the over-engineered React sync_display component.
"""

import streamlit as st
import json
from datetime import datetime
from pathlib import Path


def render_feedback_reader(left_text: str, right_text: str, left_title: str = "Original", 
                          right_title: str = "Translation", chapter_id: str = None, 
                          style_name: str = None, height: int = 500):
    """
    Shows side-by-side text with Medium-style selection popup reactions.
    
    Args:
        left_text: Original text content
        right_text: Translation text content  
        left_title: Title for left panel
        right_title: Title for right panel
        chapter_id: Chapter identifier for feedback storage
        style_name: Translation style identifier
        height: Height of reading area in pixels
        
    Returns:
        dict: Feedback data if user made a selection, None otherwise
    """
    
    # Create assets directory if needed
    assets_dir = Path("assets")
    assets_dir.mkdir(exist_ok=True)
    
    # Load selection JavaScript if it exists
    selection_js_path = assets_dir / "selection_react.js"
    selection_js = ""
    if selection_js_path.exists():
        selection_js = selection_js_path.read_text()
    else:
        # Minimal fallback JS for now
        selection_js = """
        console.log('Selection feedback JS not found - using fallback');
        window.saveReaction = function(payload) {
            console.log('Feedback:', payload);
            // This will be replaced with full implementation
        };
        """
    
    # Escape HTML in text content
    def escape_html(text):
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&#x27;')
                   .replace('\n', '<br>'))
    
    left_html = escape_html(left_text)
    right_html = escape_html(right_text)
    
    # Create side-by-side reading interface
    reading_html = f"""
    <div style="display: flex; gap: 20px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;">
        <!-- Left Panel: Original -->
        <div style="flex: 1; padding: 16px; background: #f8f9fa; border-radius: 8px; overflow-y: auto; height: {height}px;">
            <h4 style="margin-top: 0; color: #495057; border-bottom: 2px solid #dee2e6; padding-bottom: 8px;">
                📖 {left_title}
            </h4>
            <div style="line-height: 1.7; color: #212529; font-size: 15px;">
                {left_html}
            </div>
        </div>
        
        <!-- Right Panel: Translation with Selection -->
        <div style="flex: 1; padding: 16px; background: white; border: 1px solid #dee2e6; border-radius: 8px; overflow-y: auto; height: {height}px;">
            <h4 style="margin-top: 0; color: #495057; border-bottom: 2px solid #dee2e6; padding-bottom: 8px;">
                🤖 {right_title}
                <small style="font-weight: normal; color: #6c757d;">(Select text to react)</small>
            </h4>
            <div id="text-wrapper" style="line-height: 1.7; color: #212529; font-size: 15px; cursor: text;">
                {right_html}
            </div>
        </div>
    </div>
    
    <script>
    {selection_js}
    
    // Connect to Streamlit
    window.saveReaction = function(payload) {{
        const streamlitPayload = {{
            ...payload,
            chapter_id: "{chapter_id or 'unknown'}",
            style_name: "{style_name or 'unknown'}",
            timestamp: new Date().toISOString()
        }};
        
        // Send to Streamlit parent
        window.parent.postMessage({{
            type: 'streamlit:setComponentValue',
            value: streamlitPayload
        }}, '*');
    }};
    </script>
    """
    
    # Render with Streamlit
    return st.components.v1.html(reading_html, height=height + 100, scrolling=True)


def save_feedback_to_storage(feedback_data: dict, chapter_id: str, style_name: str):
    """
    Save feedback data to the existing comment storage system.
    
    Args:
        feedback_data: Feedback dict from render_feedback_reader
        chapter_id: Chapter identifier
        style_name: Style identifier
        
    Returns:
        bool: True if saved successfully
    """
    try:
        # Import existing storage functions
        from .ui_components import add_inline_comment
        
        # Convert feedback format to comment format
        comment_data = {
            'start_offset': feedback_data.get('start_offset', 0),
            'end_offset': feedback_data.get('end_offset', 0), 
            'selected_text': feedback_data.get('text', ''),
            'dimension': feedback_data.get('reaction_type', 'general_feedback'),
            'comment': f"{feedback_data.get('reaction_emoji', '❓')} {feedback_data.get('reaction_type', 'feedback')}",
            'evaluator_name': 'User',
            'timestamp': feedback_data.get('timestamp', datetime.now().isoformat())
        }
        
        return add_inline_comment(style_name, chapter_id, comment_data)
        
    except Exception as e:
        st.error(f"Failed to save feedback: {e}")
        return False


# Backward compatibility alias
def create_synchronized_text_display(*args, **kwargs):
    """Backward compatibility wrapper for old function name."""
    st.warning("⚠️ Using deprecated create_synchronized_text_display - update to render_feedback_reader")
    return render_feedback_reader(*args, **kwargs)