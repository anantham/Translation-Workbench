// utils/selection_assets/popup_feedback.js
// minimal, vanilla JS – no build step

function initPopupFeedback(wrapperSel, chapterId, styleName) {
  const wrapper = document.querySelector(wrapperSel);
  if (!wrapper) return;

  const bar = document.createElement("div");
  bar.className = "emoji-bar";
  bar.style.display = "none";
  bar.innerHTML =
    ["👍","🤔","❓","💖","😡"].map(e => `<span>${e}</span>`).join("");

  // Add comment box & submit inside the bar
  const panel = document.createElement("div");
  panel.className = "fb-panel";
  panel.innerHTML = `<textarea placeholder="Optional comment…"></textarea>
                     <button>Submit</button>`;
  bar.appendChild(panel);
  document.body.appendChild(bar);

  let lastRange = null, chosenEmoji = null;

  // Handle emoji click
  bar.querySelectorAll("span").forEach(span=>{
      span.onclick = ()=>{ chosenEmoji = span.textContent; };
  });

  // Handle submit
  panel.querySelector("button").onclick = ()=>{
    if (!lastRange) return;
    const txt = lastRange.toString();
    const payload = {
      event:        "submit_feedback",
      chapter_id:   chapterId,
      style_name:   styleName,
      start:        lastRange.startOffset,
      end:          lastRange.endOffset,
      text:         txt,
      emoji:        chosenEmoji,
      comment:      panel.querySelector("textarea").value,
    };
    Streamlit.setComponentValue(payload);

    // cleanup
    window.getSelection().removeAllRanges();
    bar.style.display="none";
    panel.querySelector("textarea").value="";
    chosenEmoji=null;
  };

  // Show bar on mouseup selection
  wrapper.onmouseup = (ev)=>{
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed) return;
    const range = sel.getRangeAt(0);
    lastRange = range;

    // Position bar
    const rect = range.getBoundingClientRect();
    bar.style.top  = `${window.scrollY + rect.top  - 40}px`;
    bar.style.left = `${window.scrollX + rect.left}px`;
    bar.style.display = "flex";
  };

  // Click outside hides bar
  document.addEventListener("mousedown",(e)=>{
    if (!bar.contains(e.target)) {
      bar.style.display="none";
    }
  });
}