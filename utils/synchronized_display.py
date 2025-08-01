# utils/synchronized_display.py
import pathlib
import uuid
import streamlit as st
from streamlit.components.v1 import html
from .selection_feedback import save_inline_feedback

ASSETS = pathlib.Path(__file__).parent / "selection_assets"

def synchronized_display_with_feedback(left_content: str, 
                                     right_content: str,
                                     left_title: str = "Reference",
                                     right_title: str = "Translation (Select text for feedback)",
                                     chapter_id: str = "",
                                     style_name: str = "",
                                     height: int = 600):
    """
    Creates a synchronized two-panel display with feedback capability on the right panel.
    
    Args:
        left_content: Text content for left panel
        right_content: Text content for right panel (with feedback capability)
        left_title: Title for left panel
        right_title: Title for right panel
        chapter_id: Chapter identifier for feedback storage
        style_name: Style name for feedback storage
        height: Height of the component in pixels
        
    Returns:
        dict or None: Feedback payload if user submitted feedback, None otherwise
    """
    
    # Generate unique component ID
    component_id = f"sync_display_{uuid.uuid4().hex[:8]}"
    
    # Load feedback assets
    js_code = (ASSETS / "popup_feedback.js").read_text()
    css_code = (ASSETS / "popup_feedback.css").read_text()
    
    # Escape HTML content
    def escape_html(text):
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('\n', '<br>'))
    
    left_escaped = escape_html(left_content)
    right_escaped = escape_html(right_content)
    
    # Calculate panel width (accounting for gap)
    panel_width = "calc(50% - 8px)"
    
    html_content = f"""
    <div id="{component_id}" style="width: 100%; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;">
        <style>
            {css_code}
            .sync-panel {{
                scroll-behavior: smooth;
            }}
            .sync-panel::-webkit-scrollbar {{
                width: 8px;
            }}
            .sync-panel::-webkit-scrollbar-track {{
                background: #f1f1f1;
                border-radius: 4px;
            }}
            .sync-panel::-webkit-scrollbar-thumb {{
                background: #888;
                border-radius: 4px;
            }}
            .sync-panel::-webkit-scrollbar-thumb:hover {{
                background: #555;
            }}
        </style>
        
        <div style="display: flex; gap: 16px; height: {height}px;">
            <!-- Left Panel -->
            <div style="width: {panel_width}; display: flex; flex-direction: column;">
                <div style="background: #f8f9fa; padding: 8px 12px; border-bottom: 1px solid #dee2e6; font-weight: 600; font-size: 14px; color: #495057;">
                    {left_title}
                </div>
                <div style="flex: 1; padding: 16px; overflow-y: auto; background: white; border: 1px solid #dee2e6; border-top: none; line-height: 1.6; font-size: 14px; color: #212529; user-select: text;" 
                     id="leftPanel" class="sync-panel">
                    {left_escaped}
                </div>
            </div>
            
            <!-- Right Panel -->
            <div style="width: {panel_width}; display: flex; flex-direction: column;">
                <div style="background: #f8f9fa; padding: 8px 12px; border-bottom: 1px solid #dee2e6; font-weight: 600; font-size: 14px; color: #495057;">
                    {right_title}
                </div>
                <div style="flex: 1; padding: 16px; overflow-y: auto; background: white; border: 1px solid #dee2e6; border-top: none; line-height: 1.6; font-size: 14px; color: #212529; user-select: text;" 
                     id="rightPanel" class="sync-panel novel-wrap">
                    {right_escaped}
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Synchronized scrolling with evidence collection logging
        const panels = document.querySelectorAll('.sync-panel');
        let isScrolling = false;
        
        // H1: Log panel height analysis for content mismatch hypothesis
        const leftPanel = document.querySelector('#leftPanel');
        const rightPanel = document.querySelector('#rightPanel');
        
        console.log('[SYNC] Panel heights analysis:', {{
            left: leftPanel.scrollHeight,
            right: rightPanel.scrollHeight, 
            leftClient: leftPanel.clientHeight,
            rightClient: rightPanel.clientHeight,
            heightRatio: rightPanel.scrollHeight / leftPanel.scrollHeight,
            contentDifference: Math.abs(rightPanel.scrollHeight - leftPanel.scrollHeight)
        }});
        
        panels.forEach(panel => {{
            panel.addEventListener('scroll', function() {{
                const timestamp = Date.now();
                
                // H2: Log timing and race condition evidence
                console.log('[SYNC] Scroll event triggered:', {{
                    panel: this.id,
                    timestamp: timestamp,
                    isScrolling: isScrolling,
                    scrollTop: this.scrollTop
                }});
                
                if (isScrolling) {{
                    console.log('[SYNC] ⚠️ BLOCKED - isScrolling flag active, potential race condition');
                    return;
                }}
                isScrolling = true;
                
                // H3: Log division by zero and math validation evidence
                const maxScroll = this.scrollHeight - this.clientHeight;
                const scrollRatio = this.scrollTop / maxScroll;
                
                console.log('[SYNC] Scroll math validation:', {{
                    scrollTop: this.scrollTop,
                    scrollHeight: this.scrollHeight,
                    clientHeight: this.clientHeight,
                    maxScroll: maxScroll,
                    scrollRatio: scrollRatio,
                    isValidRatio: !isNaN(scrollRatio) && isFinite(scrollRatio),
                    edgeCase: maxScroll <= 0
                }});
                
                // H4: Log cascading events evidence
                panels.forEach(otherPanel => {{
                    if (otherPanel !== this) {{
                        const oldScrollTop = otherPanel.scrollTop;
                        const targetScroll = scrollRatio * (otherPanel.scrollHeight - otherPanel.clientHeight);
                        otherPanel.scrollTop = targetScroll;
                        
                        console.log('[SYNC] Panel sync update:', {{
                            targetPanel: otherPanel.id,
                            oldPosition: oldScrollTop,
                            newPosition: targetScroll,
                            positionDifference: Math.abs(targetScroll - oldScrollTop),
                            largeJump: Math.abs(targetScroll - oldScrollTop) > 50
                        }});
                    }}
                }});
                
                // H2: Log timeout clearing for race condition analysis
                setTimeout(() => {{ 
                    console.log('[SYNC] Timeout cleared at:', Date.now(), 'duration:', Date.now() - timestamp);
                    isScrolling = false; 
                }}, 50);
            }});
        }});
        
        // Initialize feedback popup on right panel only
        const CHAPTER = "{chapter_id}";
        const STYLE = "{style_name}";
        {js_code}
        initPopupFeedback("#rightPanel", CHAPTER, STYLE);
    </script>
    """
    
    # Render component and return result
    result = html(html_content, height=height + 50, scrolling=True)
    
    # Handle feedback payload
    if isinstance(result, dict) and result.get("event") == "submit_feedback":
        return result
    return None