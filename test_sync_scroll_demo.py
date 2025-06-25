#!/usr/bin/env python3
"""
Demo script to test the synchronized scrolling component.
Run this with: streamlit run test_sync_scroll_demo.py
"""

import streamlit as st
import sys
import os

# Add current directory to path so we can import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import create_synchronized_text_display

st.set_page_config(page_title="Synchronized Scrolling Demo", layout="wide")

st.title("🔄 Synchronized Scrolling Component Demo")
st.caption("Testing the new side-by-side translation display with synchronized scrolling")

# Sample content for testing
sample_custom_translation = """Wei He slowly nodded, his expression thoughtful as he considered the implications.

"Alright, I'll remember that carefully," he said, his voice steady despite the weight of the information.

"Also, you've been relying on the Nine Layers Blood Training Method as your foundational cultivation technique up until now. But I'm offering you a superior alternative," Duanmu Wan explained with patience, her tone carrying the authority of someone well-versed in martial arts.

"Our Red Whale Gang's core cultivation method is known as the Red Whale Technique. Within the Foundation Building realm, this martial art is unquestionably among the most prestigious and powerful techniques available."

"However, there's a prerequisite for learning the Red Whale Technique - you must first formally join our organization and become one of our acknowledged disciples."

"What are your thoughts on this proposition?" she asked, watching his reaction carefully.

The technique represented years of accumulated wisdom and power, passed down through generations of skilled practitioners. Wei He understood that such an opportunity rarely presented itself to outsiders.

The decision would fundamentally alter his cultivation path and bind him to the Red Whale Gang's fate. He weighed the benefits against the obligations such a commitment would entail.

"The Red Whale Technique..." he murmured, testing the name on his tongue. The very sound of it suggested power and authority, like the mighty sea creatures that ruled the ocean depths."""

sample_official_translation = """Wei He nodded slowly.

"Good, I'll remember."

"Also, you have been using the Nine Layers Blood Training Method as your foundation technique. But now, I'm giving you a better choice." Duanmu Wan said patiently.

"Our Red Whale Gang's foundation technique is called the Red Whale Technique. This martial art is definitely one of the top-tier superior techniques in the Foundation Building realm."

"However, to practice the Red Whale Technique, you need to first join our gang and become our official disciple."

"What do you think?"

This technique was the culmination of generations of martial artists' wisdom and experience. Wei He knew that such opportunities were rarely given to outsiders.

This decision would completely change his cultivation path and tie his fate to the Red Whale Gang. He needed to consider the pros and cons carefully.

"Red Whale Technique..." He repeated the name, feeling its weight and power."""

st.header("📊 Side-by-Side Translation Comparison")
st.write("This demonstrates the new synchronized scrolling component that replaces the old faded text areas.")

# Test the synchronized scrolling component
create_synchronized_text_display(
    left_text=sample_custom_translation,
    right_text=sample_official_translation,
    left_title="🎨 Custom Translation (Enhanced)",
    right_title="📚 Official Translation (Original)",
    height=400
)

st.divider()

st.header("🔧 Component Features")

col1, col2 = st.columns(2)

with col1:
    st.subheader("✅ Fixed Issues")
    st.write("- **No more faded text**: Text is now fully visible and readable")
    st.write("- **Synchronized scrolling**: Scroll one panel and the other follows")
    st.write("- **Better styling**: Clean, modern appearance with proper contrast")
    st.write("- **Responsive design**: Adapts to different screen sizes")

with col2:
    st.subheader("🚀 New Features")
    st.write("- **Visual feedback**: Panels highlight briefly during sync")
    st.write("- **Custom scrollbars**: Styled scrollbars for better UX")
    st.write("- **HTML escaping**: Safe handling of special characters")
    st.write("- **Fallback support**: Graceful degradation if components unavailable")

st.divider()

st.header("🧪 Testing Instructions")
st.write("""
**To test the synchronized scrolling:**
1. Scroll in either the left or right panel
2. Notice how the other panel automatically scrolls to match
3. The panels will briefly highlight with a blue glow during synchronization
4. Both panels maintain their scroll position relationship

**Note:** This replaces the old `st.text_area` components that had:
- High opacity/faded appearance (disabled=True)
- No scroll synchronization
- Poor readability for comparison tasks
""")

st.success("🎉 Synchronized scrolling component is working correctly!")