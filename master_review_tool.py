import streamlit as st
import json
import os
import requests
import shutil
import time
from datetime import datetime
import pandas as pd
from collections import Counter
import numpy as np

# --- Semantic Similarity with Deep Learning ---
SEMANTIC_AVAILABLE = False
SEMANTIC_ERROR_MESSAGE = ""

try:
    import torch
    SEMANTIC_ERROR_MESSAGE += "✅ torch imported successfully\n"
except ImportError as e:
    SEMANTIC_ERROR_MESSAGE += f"❌ torch import failed: {e}\n"

try:
    from sentence_transformers import SentenceTransformer
    SEMANTIC_ERROR_MESSAGE += "✅ sentence-transformers imported successfully\n"
    SEMANTIC_AVAILABLE = True
except ImportError as e:
    SEMANTIC_ERROR_MESSAGE += f"❌ sentence-transformers import failed: {e}\n"

if not SEMANTIC_AVAILABLE:
    # Fallback to lightweight similarity
    from difflib import SequenceMatcher
    SEMANTIC_ERROR_MESSAGE += "📝 Falling back to syntactic similarity (difflib)\n"

# --- Semantic Similarity Models ---
@st.cache_resource
def load_semantic_model():
    """Load semantic similarity model with caching and detailed error reporting."""
    global SEMANTIC_ERROR_MESSAGE
    
    if not SEMANTIC_AVAILABLE:
        SEMANTIC_ERROR_MESSAGE += "🚫 Semantic model loading skipped - dependencies not available\n"
        return None
    
    try:
        SEMANTIC_ERROR_MESSAGE += "🔄 Attempting to load BERT model: paraphrase-multilingual-MiniLM-L12-v2\n"
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        SEMANTIC_ERROR_MESSAGE += "🧠 BERT model loaded successfully!\n"
        return model
    except Exception as e:
        SEMANTIC_ERROR_MESSAGE += f"❌ BERT model loading failed: {e}\n"
        SEMANTIC_ERROR_MESSAGE += f"📋 Exception type: {type(e).__name__}\n"
        st.warning(f"⚠️ Could not load semantic model: {e}")
        return None

def calculate_semantic_similarity(text1, text2, model=None):
    """Calculate semantic similarity using BERT embeddings."""
    if not text1 or not text2 or "File not found" in text1 or "File not found" in text2:
        return 0.0
    
    if model is None:
        model = load_semantic_model()
        if model is None:
            return calculate_syntactic_similarity_fallback(text1, text2)
    
    try:
        # Truncate texts to avoid memory issues (BERT has token limits)
        max_chars = 2000
        text1_truncated = text1[:max_chars]
        text2_truncated = text2[:max_chars]
        
        # Generate embeddings
        embeddings = model.encode([text1_truncated, text2_truncated], convert_to_tensor=True)
        
        # Calculate cosine similarity
        cosine_scores = torch.nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0), 
            embeddings[1].unsqueeze(0)
        )
        
        # Convert to float and ensure it's between 0 and 1
        similarity = float(cosine_scores.item())
        return max(0.0, min(1.0, similarity))
        
    except Exception as e:
        st.warning(f"⚠️ Semantic similarity calculation failed: {e}")
        return calculate_syntactic_similarity_fallback(text1, text2)

def calculate_syntactic_similarity_fallback(text1, text2):
    """Fallback syntactic similarity for when semantic models aren't available."""
    if not SEMANTIC_AVAILABLE:
        from difflib import SequenceMatcher
    
    # Length similarity (±30% tolerance)
    len1, len2 = len(text1), len(text2)
    length_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
    
    # Content similarity (first 1000 chars for speed)
    sample1 = text1[:1000].lower().replace('\n', ' ')
    sample2 = text2[:1000].lower().replace('\n', ' ')
    content_similarity = SequenceMatcher(None, sample1, sample2).ratio()
    
    # Combined score
    return (length_ratio * 0.3) + (content_similarity * 0.7)

def calculate_similarity(text1, text2):
    """Main similarity function - uses semantic if available, falls back to syntactic."""
    if SEMANTIC_AVAILABLE:
        return calculate_semantic_similarity(text1, text2)
    else:
        return calculate_syntactic_similarity_fallback(text1, text2)

def translate_with_gemini(raw_text: str, api_key: str):
    """Sends raw text to Gemini for translation."""
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    prompt = f"Provide a high-quality, literal English translation of this Chinese web novel chapter. Keep paragraph breaks:\n\n{raw_text}"
    data = {"contents": [{"parts": [{"text": prompt}]}]}
    
    try:
        response = requests.post(gemini_url, headers=headers, json=data, timeout=90)
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except Exception as e:
        return f"API Request Failed: {e}"

# --- Safe Data Handling with Session Persistence ---
def load_alignment_map_with_session(map_file):
    """Load alignment map with session state persistence and change detection."""
    # Check if file has been modified since last load
    if os.path.exists(map_file):
        file_mtime = os.path.getmtime(map_file)
        
        # Load from session if available and file hasn't changed
        if ('alignment_map' in st.session_state and 
            'alignment_map_mtime' in st.session_state and
            st.session_state.alignment_map_mtime == file_mtime):
            return st.session_state.alignment_map
        
        # Load fresh and store in session
        try:
            with open(map_file, 'r', encoding='utf-8') as f:
                alignment_map = json.load(f)
            st.session_state.alignment_map = alignment_map
            st.session_state.alignment_map_mtime = file_mtime
            return alignment_map
        except Exception as e:
            st.error(f"❌ Error loading alignment map: {e}")
            return None
    else:
        st.error(f"❌ Alignment map '{map_file}' not found.")
        return None

def save_alignment_map_safely(map_data, map_file):
    """Save with automatic backup and clear safety messaging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_file = f"{map_file}.backup_{timestamp}"
    
    # Create timestamped backup
    if os.path.exists(map_file):
        shutil.copy(map_file, backup_file)
        st.success(f"✅ Backup created: {backup_file}")
    
    # Save new version
    with open(map_file, 'w', encoding='utf-8') as f:
        json.dump(map_data, f, indent=2, ensure_ascii=False)
    
    # Update session state with new data and modification time
    st.session_state.alignment_map = map_data
    st.session_state.alignment_map_mtime = os.path.getmtime(map_file)
    return backup_file

def analyze_systematic_alignment_with_progress(alignment_map, api_key, sample_chapters=None):
    """Analyze alignment patterns with progress tracking."""
    if sample_chapters is None:
        all_chapters = sorted([int(k) for k in alignment_map.keys()])
        sample_chapters = all_chapters[:min(20, len(all_chapters))]
    
    # Show similarity method info
    similarity_method = "🧠 BERT semantic similarity" if SEMANTIC_AVAILABLE else "📝 Syntactic similarity"
    st.info(f"Using {similarity_method} for alignment analysis")
    
    # Pre-load semantic model if available (with progress indication)
    model = None
    if SEMANTIC_AVAILABLE:
        with st.spinner("🔄 Loading semantic similarity model (first time only)..."):
            model = load_semantic_model()
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    for i, ch_num in enumerate(sample_chapters):
        # Update progress
        progress = (i + 1) / len(sample_chapters)
        progress_bar.progress(progress)
        status_text.text(f"Analyzing chapter {ch_num}... ({i+1}/{len(sample_chapters)})")
        
        ch_data = alignment_map[str(ch_num)]
        raw_content = load_chapter_content(ch_data.get("raw_file"))
        
        if not raw_content or "File not found" in raw_content:
            continue
            
        # Get AI translation for comparison
        ai_translation = translate_with_gemini(raw_content, api_key) if api_key else None
        if not ai_translation or "API Request Failed" in ai_translation:
            continue
        
        # Test against multiple English chapters
        best_match = {"chapter": ch_num, "score": 0, "offset": 0}
        
        for offset in range(-3, 4):  # Test wider range
            test_ch = ch_num + offset
            if str(test_ch) in alignment_map:
                eng_file = alignment_map[str(test_ch)].get("english_file")
                if eng_file:
                    eng_content = load_chapter_content(eng_file)
                    if eng_content and "File not found" not in eng_content:
                        # Use semantic similarity with pre-loaded model for better performance
                        if SEMANTIC_AVAILABLE and model:
                            score = calculate_semantic_similarity(ai_translation, eng_content, model)
                        else:
                            score = calculate_similarity(ai_translation, eng_content)
                        if score > best_match["score"]:
                            best_match = {
                                "chapter": ch_num,
                                "score": score,
                                "offset": offset,
                                "matched_english": test_ch
                            }
        
        results.append(best_match)
    
    # Clean up progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return results

def preview_systematic_correction(alignment_map, offset, sample_size=10):
    """Preview what happens if we apply systematic offset correction."""
    preview = {"before": [], "after": [], "offset": offset}
    
    chapters = sorted([int(k) for k in alignment_map.keys()])[:sample_size]
    
    for ch_num in chapters:
        # Current alignment
        current_eng = alignment_map[str(ch_num)].get("english_file", "None")
        current_eng_ch = "None"
        if current_eng:
            # Extract chapter number from filename
            import re
            match = re.search(r'Chapter-(\d+)', current_eng)
            if match:
                current_eng_ch = match.group(1)
        
        # Proposed alignment  
        proposed_eng_ch = ch_num + offset
        proposed_eng = alignment_map.get(str(proposed_eng_ch), {}).get("english_file", "None")
        
        preview["before"].append({
            "raw_ch": ch_num,
            "eng_ch": current_eng_ch,
            "eng_file": current_eng
        })
        
        preview["after"].append({
            "raw_ch": ch_num,
            "eng_ch": str(proposed_eng_ch) if proposed_eng != "None" else "None",
            "eng_file": proposed_eng
        })
    
    return preview

def apply_systematic_correction(alignment_map, offset):
    """Apply systematic offset correction to all chapters."""
    corrected_map = {}
    
    for ch_str, ch_data in alignment_map.items():
        ch_num = int(ch_str)
        corrected_english_ch = ch_num + offset
        
        # Create new mapping
        corrected_map[ch_str] = {
            "raw_file": ch_data["raw_file"],
            "english_file": alignment_map.get(str(corrected_english_ch), {}).get("english_file")
        }
    
    return corrected_map

def load_chapter_content(filepath):
    if filepath and os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return "File not found or not applicable."

# --- UI Layout ---
st.set_page_config(layout="wide", page_title="Master Review Tool")
st.title("📖 Master Translation Review & Alignment Tool")

# Show similarity method being used with diagnostic option
if SEMANTIC_AVAILABLE:
    st.caption("🛡️ Human-in-the-loop safety: AI suggests, you decide | 🧠 **Semantic similarity enabled** (BERT embeddings)")
else:
    st.caption("🛡️ Human-in-the-loop safety: AI suggests, you decide | ⚠️ **Syntactic similarity** (install sentence-transformers for semantic)")
    
    # Add diagnostic expander at the top for immediate visibility
    with st.expander("🔧 **Why is semantic similarity not working? Click for diagnostics**"):
        st.warning("**Dependency Check Results:**")
        st.code(SEMANTIC_ERROR_MESSAGE, language="text")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Expected Status:**")
            st.code("""
✅ torch imported successfully
✅ sentence-transformers imported successfully
🔄 Attempting to load BERT model
🧠 BERT model loaded successfully!
            """, language="text")
        
        with col2:
            st.error("**Troubleshooting Steps:**")
            st.code("""
1. Activate virtual environment:
   source venv/bin/activate

2. Install dependencies:
   pip install sentence-transformers torch

3. Restart Streamlit:
   streamlit run master_review_tool.py
            """, language="bash")

# Initialize session state
if 'ai_translation' not in st.session_state:
    st.session_state.ai_translation = ""
if 'current_chapter' not in st.session_state:
    st.session_state.current_chapter = 1

# Load alignment map with session persistence
alignment_map = load_alignment_map_with_session("alignment_map.json")

# Create main content container for better organization
main_content = st.container()

if alignment_map:
    chapter_numbers = sorted([int(k) for k in alignment_map.keys()])
    
    # --- Sidebar Controls ---
    st.sidebar.header("🎛️ Controls")
    selected_chapter = st.sidebar.selectbox(
        "Select Chapter:", 
        options=chapter_numbers, 
        format_func=lambda x: f"Chapter {x}"
    )
    
    # Clear AI translation when chapter changes
    if st.session_state.current_chapter != selected_chapter:
        st.session_state.ai_translation = ""
        st.session_state.current_chapter = selected_chapter

    st.sidebar.divider()
    
    # --- Gemini Translation ---
    st.sidebar.header("🤖 Gemini AI Translation")
    api_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")
    
    # --- Systematic Analysis Tab ---
    st.sidebar.divider()
    st.sidebar.header("📊 Systematic Analysis")
    
    # Analysis parameters
    st.sidebar.subheader("🎯 Analysis Parameters")
    
    # Sample size input
    sample_size = st.sidebar.number_input(
        "Sample Size", 
        min_value=1, 
        max_value=50, 
        value=10, 
        help="Number of chapters to analyze"
    )
    
    # Starting chapter input with smart defaults
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        # Calculate smart default based on current chapter
        default_start = max(1, selected_chapter - sample_size // 2)
        start_chapter = st.number_input(
            "Start Chapter", 
            min_value=1, 
            max_value=max(chapter_numbers), 
            value=default_start,
            help="Starting chapter for analysis"
        )
    
    with col2:
        # Show the end chapter (calculated)
        end_chapter = min(start_chapter + sample_size - 1, max(chapter_numbers))
        st.metric("End Chapter", end_chapter, help="Last chapter to be analyzed")
    
    # Quick preset buttons
    st.sidebar.caption("**Quick Presets:**")
    preset_col1, preset_col2 = st.sidebar.columns(2)
    
    with preset_col1:
        if st.button("📍 Around Current", use_container_width=True, help=f"Analyze {sample_size} chapters around Ch.{selected_chapter}"):
            # Update start chapter to center around current
            new_start = max(1, selected_chapter - sample_size // 2)
            st.session_state.analysis_start_chapter = new_start
            st.rerun()
    
    with preset_col2:
        if st.button("🏁 From Beginning", use_container_width=True, help=f"Analyze first {sample_size} chapters"):
            st.session_state.analysis_start_chapter = 1
            st.rerun()
    
    # Apply session state if set by preset buttons
    if hasattr(st.session_state, 'analysis_start_chapter'):
        start_chapter = st.session_state.analysis_start_chapter
        del st.session_state.analysis_start_chapter
    
    # Show analysis range preview
    st.sidebar.info(f"📋 **Analysis Range:** Ch.{start_chapter} to Ch.{end_chapter} ({sample_size} chapters)")
    
    # Analysis button with confirmation dialog
    if st.sidebar.button("🔍 Run Focused Analysis", use_container_width=True, type="primary"):
        if not api_key:
            st.sidebar.error("🔑 API key required for systematic analysis")
        else:
            # Generate sample chapters list
            sample_chapters = list(range(start_chapter, end_chapter + 1))
            # Filter to only include chapters that exist in alignment map
            available_chapters = [ch for ch in sample_chapters if str(ch) in alignment_map]
            
            if not available_chapters:
                st.sidebar.error(f"❌ No chapters available in range {start_chapter}-{end_chapter}")
            else:
                # Show confirmation dialog with similarity method details
                st.session_state.show_analysis_dialog = True
                st.session_state.analysis_params = {
                    'start_chapter': start_chapter,
                    'end_chapter': end_chapter,
                    'sample_size': sample_size,
                    'available_chapters': available_chapters
                }
    
    # Analysis confirmation dialog
    if hasattr(st.session_state, 'show_analysis_dialog') and st.session_state.show_analysis_dialog:
        with main_content:
            st.header("🔍 Analysis Confirmation")
            
            # Show similarity method details
            if SEMANTIC_AVAILABLE:
                st.success("🧠 **SEMANTIC SIMILARITY ENABLED**")
                st.info("""
                **Method:** BERT embeddings (paraphrase-multilingual-MiniLM-L12-v2)
                **Quality:** Gold standard for translation comparison
                **Understands:** Context, synonyms, paraphrasing, semantic meaning
                **Accuracy:** High - perfect for detecting translation alignment
                """)
                similarity_icon = "🧠"
                similarity_quality = "HIGH ACCURACY"
            else:
                st.warning("⚠️ **SYNTACTIC SIMILARITY (FALLBACK MODE)**")
                st.error("""
                **Method:** Text pattern matching (difflib)
                **Quality:** Basic - may miss semantic equivalence
                **Limitations:** No context understanding, poor with paraphrasing
                **Recommendation:** Install `sentence-transformers` for better results
                """)
                
                # Show detailed diagnostic information
                with st.expander("🔍 **Diagnostic Details - Why Semantic Similarity Failed**"):
                    st.text("Dependency Check Log:")
                    st.code(SEMANTIC_ERROR_MESSAGE, language="text")
                    
                    # Additional runtime diagnostics
                    st.text("Runtime Environment:")
                    import sys
                    st.code(f"""
Python Version: {sys.version}
Python Path: {sys.executable}
Virtual Environment: {os.environ.get('VIRTUAL_ENV', 'Not detected')}
Current Working Directory: {os.getcwd()}
                    """, language="text")
                    
                    # Try to get more specific package info
                    try:
                        import pkg_resources
                        installed_packages = [f"{pkg.key}=={pkg.version}" for pkg in pkg_resources.working_set]
                        relevant_packages = [pkg for pkg in installed_packages if any(x in pkg for x in ['torch', 'sentence', 'transform', 'numpy'])]
                        if relevant_packages:
                            st.text("Relevant Installed Packages:")
                            st.code("\n".join(relevant_packages), language="text")
                        else:
                            st.text("No relevant packages found in environment")
                    except Exception as e:
                        st.text(f"Could not check installed packages: {e}")
                    
                    st.markdown("**💡 To enable semantic similarity:**")
                    st.code("""
# Activate your virtual environment first
source venv/bin/activate

# Install required packages
pip install sentence-transformers torch numpy

# Restart Streamlit
streamlit run master_review_tool.py
                    """, language="bash")
                similarity_icon = "📝"
                similarity_quality = "LIMITED ACCURACY"
            
            # Analysis parameters summary
            params = st.session_state.analysis_params
            st.subheader("📋 Analysis Parameters")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Start Chapter", params['start_chapter'])
            with col2:
                st.metric("End Chapter", params['end_chapter'])
            with col3:
                st.metric("Sample Size", params['sample_size'])
            with col4:
                st.metric("Available Chapters", len(params['available_chapters']))
            
            st.info(f"{similarity_icon} **Similarity Method:** {similarity_quality}")
            
            # Confirmation buttons
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("✅ **Proceed with Analysis**", type="primary", use_container_width=True):
                    # Clear dialog and run analysis
                    del st.session_state.show_analysis_dialog
                    del st.session_state.analysis_params
                    
                    st.subheader(f"🔄 Running Focused Analysis: Chapters {params['start_chapter']}-{params['end_chapter']}")
                    st.info(f"Using {similarity_icon} similarity method | Analyzing {len(params['available_chapters'])} chapters")
                    
                    # Store analysis results in session state with progress tracking
                    st.session_state.systematic_analysis = analyze_systematic_alignment_with_progress(
                        alignment_map, api_key, params['available_chapters']
                    )
                    st.success("✅ Focused analysis complete! Results displayed below.")
                    st.rerun()
            
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    # Clear dialog
                    del st.session_state.show_analysis_dialog
                    del st.session_state.analysis_params
                    st.rerun()
            
            with col3:
                if not SEMANTIC_AVAILABLE:
                    st.markdown("**💡 To enable semantic similarity:**")
                    st.code("pip install sentence-transformers torch", language="bash")
            
            st.divider()
    
    if hasattr(st.session_state, 'systematic_analysis') and st.session_state.systematic_analysis:
        # Calculate most common offset
        offsets = [r["offset"] for r in st.session_state.systematic_analysis if r["score"] > 0.3]
        if offsets:
            from collections import Counter
            most_common_offset = Counter(offsets).most_common(1)[0][0]
            confidence = offsets.count(most_common_offset) / len(offsets)
            
            st.sidebar.metric("Detected Pattern", f"Offset: {most_common_offset:+d}", f"Confidence: {confidence:.1%}")
            
            if confidence > 0.6:  # High confidence threshold
                st.sidebar.success("🎯 Strong pattern detected!")
                
                if st.sidebar.button("📋 Preview Systematic Correction", use_container_width=True):
                    st.session_state.correction_preview = preview_systematic_correction(
                        alignment_map, most_common_offset
                    )
                    st.sidebar.success("✅ Preview ready! Check main area.")
            else:
                st.sidebar.warning("⚠️ Pattern unclear - manual review recommended")
    
    chapter_data = alignment_map[str(selected_chapter)]
    raw_content = load_chapter_content(chapter_data["raw_file"])

    if st.sidebar.button("🔄 Translate with Gemini", use_container_width=True):
        if not api_key:
            st.sidebar.error("🔑 API key required")
        else:
            with st.spinner("🔄 Calling Gemini API..."):
                st.session_state.ai_translation = translate_with_gemini(raw_content, api_key)

    st.sidebar.divider()
    
    # --- Alignment Analysis ---
    st.sidebar.header("🎯 Alignment Analysis")
    
    if st.session_state.ai_translation and "API Request Failed" not in st.session_state.ai_translation:
        # Calculate similarity scores for nearby chapters
        scores = {}
        eng_contents = {}
        
        for offset in range(-2, 3):  # Check ±2 chapters
            check_num = selected_chapter + offset
            if str(check_num) in alignment_map:
                eng_file = alignment_map[str(check_num)].get('english_file')
                if eng_file:
                    eng_content = load_chapter_content(eng_file)
                    if eng_content and "File not found" not in eng_content:
                        score = calculate_similarity(st.session_state.ai_translation, eng_content)
                        scores[check_num] = score
                        eng_contents[check_num] = eng_content

        if scores:
            best_match_chapter = max(scores, key=scores.get)
            best_score = scores[best_match_chapter]
            
            # Display scores with similarity method indicator
            similarity_method_icon = "🧠" if SEMANTIC_AVAILABLE else "📝"
            st.sidebar.write(f"**Similarity Scores** {similarity_method_icon} (AI vs English):")
            
            for ch_num in sorted(scores.keys()):
                score = scores[ch_num]
                icon = "⭐" if ch_num == best_match_chapter else "📄"
                color = "green" if ch_num == best_match_chapter else "normal"
                st.sidebar.markdown(f"{icon} **Ch.{ch_num}:** `{score:.3f}`")
            
            # Show method used for individual alignment
            if SEMANTIC_AVAILABLE:
                st.sidebar.caption("🧠 Using semantic similarity (BERT)")
            else:
                st.sidebar.caption("📝 Using syntactic similarity (basic)")
            
            # Alignment status
            current_score = scores.get(selected_chapter, 0)
            
            if best_match_chapter == selected_chapter:
                st.sidebar.success("✅ **Alignment looks correct!**")
                st.sidebar.metric("Current Alignment Score", f"{current_score:.3f}")
            else:
                st.sidebar.warning("🚨 **Misalignment detected!**")
                st.sidebar.metric("Current Score", f"{current_score:.3f}")
                st.sidebar.metric("Best Match", f"Chapter {best_match_chapter}", f"{best_score:.3f}")
                st.sidebar.caption(f"**Suggestion:** Raw Ch.{selected_chapter} → English Ch.{best_match_chapter}")
                
                # --- HUMAN-CONTROLLED CORRECTION ---
                st.sidebar.divider()
                st.sidebar.subheader("🔧 Correction Controls")
                
                correction_confirmed = st.sidebar.checkbox(
                    f"I want to align Raw Ch.{selected_chapter} with English Ch.{best_match_chapter}",
                    help="Check this box to enable the correction button"
                )
                
                if correction_confirmed:
                    if st.sidebar.button(
                        f"✅ Apply Correction", 
                        use_container_width=True,
                        type="primary"
                    ):
                        # Get the English file path we want to assign
                        target_eng_file = alignment_map[str(best_match_chapter)]['english_file']
                        
                        # Update the alignment map
                        old_eng_file = alignment_map[str(selected_chapter)]['english_file']
                        alignment_map[str(selected_chapter)]['english_file'] = target_eng_file
                        
                        # For simple off-by-one, we'll nullify the displaced chapter
                        # (More complex swapping logic could be added later)
                        alignment_map[str(best_match_chapter)]['english_file'] = None
                        
                        # Save with backup
                        backup_file = save_alignment_map_safely(alignment_map, "alignment_map.json")
                        
                        st.sidebar.success("🎉 **Correction Applied!**")
                        st.sidebar.info(f"📁 Backup: {backup_file}")
                        st.sidebar.info("🔄 Page will reload in 3 seconds...")
                        
                        time.sleep(3)
                        st.rerun()
                else:
                    st.sidebar.info("☝️ Check the box above to enable correction")
    else:
        st.sidebar.info("🔄 Translate with Gemini to enable alignment analysis")

    # --- Chapter Info ---
    st.sidebar.divider()
    st.sidebar.header("📋 Chapter Info")
    st.sidebar.info(f"**Currently viewing:** Chapter {selected_chapter}")
    
    if chapter_data.get("english_file"):
        st.sidebar.success("✅ English translation available")
    else:
        st.sidebar.warning("❌ No English translation linked")
    
    if chapter_data.get("raw_file"):
        st.sidebar.success("✅ Chinese raw available")
    else:
        st.sidebar.warning("❌ No Chinese raw linked")

    # --- Main Content Display using container ---
    with main_content:
        # Check if we should show systematic analysis results
        if hasattr(st.session_state, 'systematic_analysis') and st.session_state.systematic_analysis:
            st.header("📊 Systematic Alignment Analysis Results")
            
            # Display analysis results in a table
            df_data = []
            for result in st.session_state.systematic_analysis:
                df_data.append({
                    "Raw Chapter": result["chapter"],
                    "Best English Match": result.get("matched_english", "N/A"),
                    "Offset": f"{result['offset']:+d}",
                    "Similarity Score": f"{result['score']:.3f}",
                    "Status": "✅ Good" if result["offset"] == 0 else "🚨 Misaligned"
                })
            
            df = pd.DataFrame(df_data)
            st.dataframe(df, use_container_width=True)
            
            # Show pattern summary
            offsets = [r["offset"] for r in st.session_state.systematic_analysis if r["score"] > 0.3]
            if offsets:
                offset_counts = Counter(offsets)
                most_common_offset = offset_counts.most_common(1)[0][0]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Most Common Offset", f"{most_common_offset:+d}")
                with col2:
                    st.metric("Chapters Analyzed", len(st.session_state.systematic_analysis))
                with col3:
                    confidence = offsets.count(most_common_offset) / len(offsets)
                    st.metric("Pattern Confidence", f"{confidence:.1%}")
            
            st.divider()
    
        # Show correction preview if available (within main_content container)
        if hasattr(st.session_state, 'correction_preview'):
            st.header("📋 Systematic Correction Preview")
            st.write(f"**Proposed Offset:** {st.session_state.correction_preview['offset']:+d}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔴 BEFORE Correction")
                before_data = []
                for item in st.session_state.correction_preview["before"]:
                    before_data.append({
                        "Raw Ch": item["raw_ch"],
                        "→ English Ch": item["eng_ch"],
                        "Status": "✅" if item["eng_ch"] != "None" else "❌"
                    })
                st.dataframe(pd.DataFrame(before_data), use_container_width=True)
            
            with col2:
                st.subheader("🟢 AFTER Correction")
                after_data = []
                for item in st.session_state.correction_preview["after"]:
                    after_data.append({
                        "Raw Ch": item["raw_ch"],
                        "→ English Ch": item["eng_ch"],
                        "Status": "✅" if item["eng_ch"] != "None" else "❌"
                    })
                st.dataframe(pd.DataFrame(after_data), use_container_width=True)
            
            # Systematic correction controls
            st.subheader("🔧 Apply Systematic Correction")
            
            systematic_confirmed = st.checkbox(
                f"I want to apply systematic offset correction ({st.session_state.correction_preview['offset']:+d}) to ALL chapters",
                help="This will modify the alignment for all chapters. A backup will be created."
            )
            
            if systematic_confirmed:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Apply Systematic Correction", type="primary", use_container_width=True):
                        # Apply the correction
                        corrected_map = apply_systematic_correction(alignment_map, st.session_state.correction_preview['offset'])
                        backup_file = save_alignment_map_safely(corrected_map, "alignment_map.json")
                        
                        st.success("🎉 **Systematic correction applied!**")
                        st.info(f"📁 Backup saved: {backup_file}")
                        st.info("🔄 Page will reload in 3 seconds...")
                        
                        # Clear analysis results
                        if hasattr(st.session_state, 'systematic_analysis'):
                            del st.session_state.systematic_analysis
                        if hasattr(st.session_state, 'correction_preview'):
                            del st.session_state.correction_preview
                        
                        time.sleep(3)
                        st.rerun()
                
                with col2:
                    if st.button("❌ Cancel", use_container_width=True):
                        # Clear the preview
                        if hasattr(st.session_state, 'correction_preview'):
                            del st.session_state.correction_preview
                        st.rerun()
            
            st.divider()
    
        # Regular 3-pane view (within main_content container)
        st.header(f"📖 Individual Review: Chapter {selected_chapter}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📜 Raw (Chinese)")
            st.text_area("Chinese Content", raw_content, height=600, key="raw_text")
            st.caption(f"File: {chapter_data.get('raw_file', 'N/A')}")
            
        with col2:
            st.subheader("📖 Official Translation")
            eng_filepath = chapter_data.get("english_file")
            eng_content = load_chapter_content(eng_filepath)
            st.text_area("Official English Content", eng_content, height=600, key="eng_text")
            st.caption(f"File: {eng_filepath or 'Not available'}")
            
        with col3:
            st.subheader("🤖 AI Translation")
            if st.button("🔄 Translate Chapter with Gemini", use_container_width=True):
                if api_key:
                    with st.spinner("🔄 Translating..."):
                        st.session_state.ai_translation = translate_with_gemini(raw_content, api_key)
                else:
                    st.error("🔑 API Key Required")
            st.text_area("AI Generated Content", st.session_state.ai_translation, height=600, key="ai_text")
            if st.session_state.ai_translation:
                st.caption(f"Generated via Gemini API • {len(st.session_state.ai_translation)} chars")
            else:
                st.caption("Use button above to generate AI translation")

else:
    st.error("❌ Could not load alignment map. Please ensure 'alignment_map.json' exists.")
    st.info("💡 Run the appropriate setup scripts to create the alignment map first.")