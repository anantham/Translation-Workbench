import { Streamlit, withStreamlitConnection } from "streamlit-component-lib";
import React, { useEffect } from "react";

interface SyncDisplayProps {
  args: {
    left: string;
    right: string;
    chapterId: string;
    style: string;
    height: number;
  };
}

const SyncDisplay: React.FC<SyncDisplayProps> = (props) => {
  const { left, right, chapterId, style, height } = props.args;

  useEffect(() => {
    Streamlit.setFrameHeight(height);
  });

  const onMouseUp = () => {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed) return;

    const rightPanel = document.getElementById("rightPanel");
    if (!rightPanel) return;

    const range = sel.getRangeAt(0);
    if (!range || !rightPanel.contains(range.commonAncestorContainer)) return;

    const payload = {
      type: "selection",
      chapter_id: chapterId,
      style_name: style,
      text: sel.toString(),
      start: range.startOffset,
      end: range.endOffset,
    };
    Streamlit.setComponentValue(payload);
  };

  return (
    <div style={{ display: "flex", gap: "2%", height: `${height}px` }} onMouseUp={onMouseUp}>
      <div style={{ width: "49%", overflowY: "auto", border: "1px solid #ccc", padding: "10px" }}>
        <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{left}</pre>
      </div>
      <div id="rightPanel" style={{ width: "49%", overflowY: "auto", border: "1px solid #ccc", padding: "10px" }}>
        <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{right}</pre>
      </div>
    </div>
  );
};

export default withStreamlitConnection(SyncDisplay);