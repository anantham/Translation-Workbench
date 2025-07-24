# utils/selection_feedback.py
import json, pathlib, uuid
import streamlit as st
from streamlit.components.v1 import html

ASSETS = pathlib.Path(__file__).parent / "selection_assets"

def feedback_widget(html_text: str,
                    chapter_id: str,
                    style_name: str,
                    height: int = 800):
    """
    Renders selectable HTML and returns a payload dict *once* the user
    submits feedback. Otherwise returns None.
    """
    component_id = f"feedback_{uuid.uuid4().hex[:6]}"
    js_code  = (ASSETS / "popup_feedback.js").read_text()
    css_code = (ASSETS / "popup_feedback.css").read_text()

    result = html(
        f"""
        <style>{css_code}</style>
        <div id="{component_id}" class="novel-wrap">{html_text}</div>

        <script>
        const CHAPTER  = "{chapter_id}";
        const STYLE    = "{style_name}";
        {js_code}
        initPopupFeedback("#{component_id}", CHAPTER, STYLE);
        </script>
        """,
        height=height,
        scrolling=True,
    )

    # `result` is whatever the JS passes to Streamlit.setComponentValue
    if isinstance(result, dict) and result.get("event") == "submit_feedback":
        return result  # {'event': 'submit_feedback', 'chapter_id': …}
    return None


def save_inline_feedback(payload: dict):
    """
    Save feedback payload to the existing comment storage system.
    
    Args:
        payload: Feedback dict from feedback_widget
        
    Returns:
        bool: True if saved successfully
    """
    try:
        # Import existing storage functions
        from .ui_components import add_inline_comment
        
        # Convert feedback format to comment format
        comment_data = {
            'start_offset': payload.get('start', 0),
            'end_offset': payload.get('end', 0), 
            'selected_text': payload.get('text', ''),
            'dimension': 'user_feedback',  # New category for emoji feedback
            'comment': f"{payload.get('emoji', '❓')} {payload.get('comment', '')}".strip(),
            'evaluator_name': 'User',
            'timestamp': payload.get('timestamp', None)
        }
        
        return add_inline_comment(
            payload['style_name'], 
            payload['chapter_id'], 
            comment_data
        )
        
    except Exception as e:
        st.error(f"Failed to save feedback: {e}")
        return False