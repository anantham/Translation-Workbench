import streamlit.components.v1 as components
from pathlib import Path

_sync_display = components.declare_component(
    "sync_display",
    path=str(Path(__file__).parent / "frontend" / "build")
)

def sync_display(left, right, chapter_id, style_name, key=None, height=900):
    return _sync_display(
        left=left,
        right=right,
        chapterId=chapter_id,
        style=style_name,
        key=key or f"{chapter_id}_{style_name}",
        default=None,
        height=height,
    )