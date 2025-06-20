import streamlit as st
import json
import os
import requests
from difflib import SequenceMatcher

def translate_with_gemini(raw_text: str, api_key: str):
    """
    Sends raw Chinese text to the Gemini API for translation.
    """
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    
    headers = {'Content-Type': 'application/json'}
    
    prompt = f"Please provide a high-quality, literal translation of the following Chinese web novel chapter into English. Retain the paragraph breaks:\n\n---\n\n{raw_text}"
    
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    try:
        response = requests.post(gemini_url, headers=headers, json=data, timeout=90)
        response.raise_for_status()
        
        json_response = response.json()
        translated_text = json_response['candidates'][0]['content']['parts'][0]['text']
        return translated_text
        
    except requests.exceptions.RequestException as e:
        return f"API Request Failed: {e}"
    except (KeyError, IndexError) as e:
        return f"Error parsing API response. Details: {e}\n\nFull Response:\n{response.text}"

@st.cache_data
def load_alignment_map(map_file):
    """Loads the alignment map from the JSON file."""
    if not os.path.exists(map_file):
        st.error(f"Error: Alignment map '{map_file}' not found. Please run 'create_alignment_map.py' first.")
        return None
    with open(map_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_chapter_content(filepath):
    """Loads the text content of a single chapter file."""
    if filepath and os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return "File not found or not applicable."

def simple_similarity(text1, text2):
    """Fast similarity using length + sequence matching."""
    if not text1 or not text2 or "File not found" in text1 or "File not found" in text2:
        return 0.0
    
    # Length similarity
    len1, len2 = len(text1), len(text2)
    length_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
    
    # Text similarity (first 500 chars for speed)
    sample1 = text1[:500].lower()
    sample2 = text2[:500].lower()
    text_similarity = SequenceMatcher(None, sample1, sample2).ratio()
    
    return (length_ratio * 0.4) + (text_similarity * 0.6)

st.set_page_config(layout="wide")
st.title("📖 Novel Translation Review UI")

if 'ai_translation' not in st.session_state:
    st.session_state.ai_translation = ""
if 'current_chapter' not in st.session_state:
    st.session_state.current_chapter = None

alignment_map = load_alignment_map("alignment_map.json")

if alignment_map:
    chapter_numbers = sorted([int(k) for k in alignment_map.keys()])
    
    st.sidebar.header("Controls")
    selected_chapter = st.sidebar.selectbox(
        "Select Chapter:",
        options=chapter_numbers,
        format_func=lambda x: f"Chapter {x}"
    )

    if st.session_state.current_chapter != selected_chapter:
        st.session_state.ai_translation = ""
        st.session_state.current_chapter = selected_chapter

    st.sidebar.divider()
    
    st.sidebar.header("Gemini AI Translation")
    api_key = st.sidebar.text_input("Enter your Gemini API Key:", type="password")

    chapter_data = alignment_map[str(selected_chapter)]
    raw_filepath = chapter_data["raw_file"]
    raw_content = load_chapter_content(raw_filepath)

    if st.sidebar.button("Translate with Gemini"):
        if not api_key:
            st.sidebar.error("Please enter an API key to use the translator.")
        else:
            with st.spinner("Calling Gemini API... This may take a moment."):
                st.session_state.ai_translation = translate_with_gemini(raw_content, api_key)
    
    st.sidebar.info(f"Displaying Chapter **{selected_chapter}**.")
    if chapter_data["english_file"]:
        st.sidebar.success("✅ Official English translation found.")
    else:
        st.sidebar.warning("❌ No official English translation found.")
    
    # Quick Alignment Check
    st.sidebar.divider()
    st.sidebar.header("🎯 Quick Alignment Check")
    
    if st.session_state.ai_translation and "API Request Failed" not in st.session_state.ai_translation:
        eng_content = load_chapter_content(chapter_data["english_file"])
        
        if eng_content and "File not found" not in eng_content:
            ai_similarity = simple_similarity(st.session_state.ai_translation, eng_content)
            
            # Check nearby chapters for better matches
            best_score = ai_similarity
            best_chapter = selected_chapter
            
            for offset in [-2, -1, 1, 2]:
                check_ch = selected_chapter + offset
                if str(check_ch) in alignment_map:
                    other_eng_file = alignment_map[str(check_ch)]["english_file"]
                    other_eng_content = load_chapter_content(other_eng_file)
                    if other_eng_content and "File not found" not in other_eng_content:
                        score = simple_similarity(st.session_state.ai_translation, other_eng_content)
                        if score > best_score:
                            best_score = score
                            best_chapter = check_ch
            
            st.sidebar.metric("Current Match Score", f"{ai_similarity:.3f}")
            
            if best_chapter != selected_chapter:
                st.sidebar.warning(f"🚨 Better match found!")
                st.sidebar.metric("Best Match", f"Chapter {best_chapter}", f"{best_score:.3f}")
                st.sidebar.caption(f"Suggests Raw Ch.{selected_chapter} → English Ch.{best_chapter}")
            else:
                st.sidebar.success("✅ Alignment looks good!")
        else:
            st.sidebar.info("Load English content to check alignment")
    else:
        st.sidebar.info("Translate with AI to enable alignment check")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.header("Raw (Chinese)")
        st.text_area("Raw Content", raw_content, height=700, key="raw_text")
        
    with col2:
        st.header("Official Translation (English)")
        eng_filepath = chapter_data["english_file"]
        eng_content = load_chapter_content(eng_filepath)
        st.text_area("Official English Content", eng_content, height=700, key="eng_text")
        
    with col3:
        st.header("AI Translation (Gemini)")
        st.text_area("AI Generated Content", st.session_state.ai_translation, height=700, key="ai_text")