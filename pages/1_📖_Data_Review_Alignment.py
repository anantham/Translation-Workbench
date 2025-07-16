import streamlit as st
import json
import os
import requests
import shutil
import time
import re
from datetime import datetime
import pandas as pd
from collections import Counter
import numpy as np
import hashlib

# Import shared utilities for modular architecture
from utils import *
from utils import get_ai_translation_content  # Explicit import for Streamlit caching
from utils import SEMANTIC_AVAILABLE, SEMANTIC_ERROR_MESSAGE  # Explicit import for availability checks
from utils.alignment_map_builder import (
    preview_alignment_mapping, 
    build_and_save_alignment_map,
    get_alignment_map_path,
    validate_chapter_directories
)

# All shared functions now imported from utils.py

# --- UI Layout ---
st.set_page_config(layout="wide", page_title="📖 Data Review & Alignment", page_icon="📖")
st.title("📖 Data Review & Alignment")
st.caption("**Translation Framework Workbench** | 🚀 Home page for dataset curation and alignment")

# Quick navigation and workbench overview
with st.expander("🚀 **Workbench Overview** - Click to see all available tools"):
    st.markdown("""
    ### 🎯 **Complete MLOps Pipeline for Translation Models**
    
    **Current Page: 📖 Data Review & Alignment**
    - ✅ Chapter-by-chapter quality control and manual corrections
    - ✅ Binary search for misalignment detection
    - ✅ Surgical alignment corrections
    - ✅ Chapter splitting and merge handling
    
    **Other Pages Available:**
    
    **🤖 Fine-tuning Workbench** *(See sidebar navigation)*
    - 📊 Dataset preparation and quality analysis
    - 🚀 Model training with hyperparameter control
    - 📈 Real-time training monitoring with loss curves
    - 🏆 Model management and metadata tracking
    
    **🧪 Experimentation Lab** *(See sidebar navigation)*
    - 🔬 Quick translation testing (base vs fine-tuned models)
    - 📊 Batch evaluation with statistical significance
    - 📈 Performance analysis and visualization
    - 🏆 Model leaderboard with composite scoring
    
    **🎯 Workflow:**
    1. **Start here** - Review and perfect your dataset alignment
    2. **Fine-tuning Workbench** - Train custom models on your data  
    3. **Experimentation Lab** - Compare and evaluate model performance
    """)
    
    st.info("💡 **Tip:** Use the sidebar navigation to switch between workbench tools!")

st.divider()

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
if 'selected_novel' not in st.session_state:
    st.session_state.selected_novel = None

# --- Alignment Map Selection ---
st.sidebar.header("📁 Alignment Map Selection")

# Use unified alignment map system
try:
    from utils import list_alignment_maps, load_alignment_map_by_slug, parse_chapter_ranges
    
    # Get available alignment maps
    available_maps = list_alignment_maps()
    
    if not available_maps:
        st.sidebar.error("❌ No alignment maps found. Please build an alignment map first.")
        st.sidebar.info("Use the alignment map builder below to create an alignment map.")
        alignment_map = None
        selected_slug = None
    else:
        # Alignment map selection dropdown
        selected_slug = st.sidebar.selectbox(
            "Choose alignment map:",
            options=sorted(available_maps.keys()),
            help="Select which alignment map to use for data review"
        )
        
        # Optional: Chapter filtering
        chapter_range = st.sidebar.text_input(
            "Chapter Range (optional):",
            placeholder="e.g. 1-100,102,105-110",
            help="Filter to specific chapters. Leave empty to use all chapters."
        )
        
        # Load alignment map
        chapters = None
        if chapter_range:
            chapters = parse_chapter_ranges(chapter_range)
            st.sidebar.info(f"📊 Filtered to {len(chapters)} chapters")
        
        alignment_map = load_alignment_map_by_slug(selected_slug, chapters)
        st.sidebar.success(f"✅ Loaded: **{selected_slug}** ({len(alignment_map)} chapters)")
        
        # Store selected slug in session state
        st.session_state.selected_novel = selected_slug
        
except Exception as e:
    st.sidebar.error(f"❌ Error loading alignment map: {str(e)}")
    alignment_map = None
    selected_slug = None

# Check if alignment map was loaded successfully
if not alignment_map:
    selected_slug = None

# Validate alignment map exists - if not, show directory selection UI
if not alignment_map:
    st.header("🔨 Build Alignment Map")
    st.info("⚠️ No alignment map found or could not load alignment map. Build one by selecting source directories below.")
    st.divider()
    
    # --- Directory Selection Interface ---
    st.subheader("📂 Source Directory Selection")
    st.caption("Select the directories containing Chinese and English chapter files to build alignment map")
    
    # Quick directory suggestions based on existing novels
    if os.path.exists("data/novels"):
        available_novels = [d for d in os.listdir("data/novels") if os.path.isdir(os.path.join("data/novels", d))]
        if available_novels:
            st.info(f"💡 **Quick suggestion**: Found {len(available_novels)} existing novels in `data/novels/`. You can manually type paths like:")
            for novel in available_novels:
                chinese_example = f"data/novels/{novel}/raw_chapters"
                if os.path.exists(chinese_example):
                    st.markdown(f"   - **{novel}**: `{chinese_example}`")
    
    # Directory selection columns
    dir_col1, dir_col2 = st.columns(2)
    
    with dir_col1:
        st.markdown("**🇨🇳 Chinese Chapters Directory**")
        
        # Directory suggestions dropdown
        suggested_dirs = []
        
        # Scan for potential Chinese directories
        if os.path.exists("data/novels"):
            for novel_dir in os.listdir("data/novels"):
                novel_path = os.path.join("data/novels", novel_dir)
                if os.path.isdir(novel_path):
                    # Check common Chinese directory names
                    potential_dirs = [
                        os.path.join(novel_path, "raw_chapters"),
                        os.path.join(novel_path, "chinese_chapters"),
                        os.path.join(novel_path, "原文章节"),
                        novel_path  # Include the novel dir itself
                    ]
                    
                    for dir_path in potential_dirs:
                        if os.path.exists(dir_path) and os.path.isdir(dir_path):
                            txt_files = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
                            if txt_files:
                                suggested_dirs.append((dir_path, len(txt_files)))
        
        # Also check common legacy locations
        legacy_dirs = [
            "novel_content_dxmwx_complete",
            "data/chinese_chapters",
            "chinese_chapters"
        ]
        
        for dir_path in legacy_dirs:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                txt_files = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
                if txt_files:
                    suggested_dirs.append((dir_path, len(txt_files)))
        
        if suggested_dirs:
            st.write("**Available Chinese directories:**")
            for i, (dir_path, file_count) in enumerate(suggested_dirs[:10]):  # Show max 10
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(f"📁 {dir_path} ({file_count} files)", key=f"chinese_dir_{i}"):
                        st.session_state.chinese_dir_input = dir_path
                        st.rerun()
                with col2:
                    # Show relative path for easier copying
                    st.code(dir_path, language=None)
        else:
            st.info("No directories with .txt files found. Please type the path manually below.")
        
        chinese_dir = st.text_input(
            "Chinese Directory Path:",
            placeholder="e.g., data/novels/my_novel/raw_chapters",
            help="Path to directory containing Chinese chapter files (.txt)",
            key="chinese_dir_input"
        )
        
        if chinese_dir:
            if os.path.exists(chinese_dir):
                if os.path.isdir(chinese_dir):
                    txt_files = len([f for f in os.listdir(chinese_dir) if f.endswith('.txt')])
                    if txt_files > 0:
                        st.success(f"✅ Found {txt_files} .txt files")
                    else:
                        st.warning("⚠️ No .txt files found in directory")
                else:
                    st.error("❌ Path is not a directory")
            else:
                st.error("❌ Directory does not exist")
    
    with dir_col2:
        st.markdown("**🇺🇸 English Chapters Directory**")
        
        # Directory suggestions dropdown
        suggested_dirs = []
        
        # Scan for potential English directories
        if os.path.exists("data/novels"):
            for novel_dir in os.listdir("data/novels"):
                novel_path = os.path.join("data/novels", novel_dir)
                if os.path.isdir(novel_path):
                    # Check common English directory names
                    potential_dirs = [
                        os.path.join(novel_path, "official_english"),
                        os.path.join(novel_path, "english_chapters"),
                        os.path.join(novel_path, "translated"),
                        os.path.join(novel_path, "english")
                    ]
                    
                    for dir_path in potential_dirs:
                        if os.path.exists(dir_path) and os.path.isdir(dir_path):
                            txt_files = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
                            if txt_files:
                                suggested_dirs.append((dir_path, len(txt_files)))
        
        # Also check common legacy locations
        legacy_dirs = [
            "english_chapters",
            "data/english_chapters",
            "official_english"
        ]
        
        for dir_path in legacy_dirs:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                txt_files = [f for f in os.listdir(dir_path) if f.endswith('.txt')]
                if txt_files:
                    suggested_dirs.append((dir_path, len(txt_files)))
        
        if suggested_dirs:
            st.write("**Available English directories:**")
            for i, (dir_path, file_count) in enumerate(suggested_dirs[:10]):  # Show max 10
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(f"📁 {dir_path} ({file_count} files)", key=f"english_dir_{i}"):
                        st.session_state.english_dir_input = dir_path
                        st.rerun()
                with col2:
                    # Show relative path for easier copying
                    st.code(dir_path, language=None)
        else:
            st.info("No directories with .txt files found. Please type the path manually below.")
        
        english_dir = st.text_input(
            "English Directory Path:",
            placeholder="e.g., data/novels/my_novel/official_english",
            help="Path to directory containing English chapter files (.txt)",
            key="english_dir_input"
        )
        
        if english_dir:
            if os.path.exists(english_dir):
                if os.path.isdir(english_dir):
                    txt_files = len([f for f in os.listdir(english_dir) if f.endswith('.txt')])
                    if txt_files > 0:
                        st.success(f"✅ Found {txt_files} .txt files")
                    else:
                        st.warning("⚠️ No .txt files found in directory")
                else:
                    st.error("❌ Path is not a directory")
            else:
                st.error("❌ Directory does not exist")
    
    # Preview and Build Section
    if chinese_dir and english_dir:
        st.divider()
        st.subheader("📋 Preview Alignment")
        
        # Preview button
        if st.button("🔍 Preview Alignment Map", type="secondary", use_container_width=True):
            st.session_state.alignment_preview_active = True
            st.rerun()
        
        # Show preview if active
        if st.session_state.get('alignment_preview_active', False):
            with st.spinner("🔍 Analyzing directories and validating files..."):
                preview_result = preview_alignment_mapping(chinese_dir, english_dir)
            
            if preview_result["success"]:
                st.success("✅ Preview generated successfully!")
                
                # Show statistics
                stats = preview_result["stats"]
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                
                with stat_col1:
                    st.metric("Total Chapters", stats["total_mappings"])
                with stat_col2:
                    st.metric("Both Files", stats["both_files"])
                with stat_col3:
                    st.metric("Chinese Only", stats["chinese_only"])
                with stat_col4:
                    st.metric("English Only", stats["english_only"])
                
                # Show file issues if any
                if preview_result.get("file_issues"):
                    st.warning(f"⚠️ Found {len(preview_result['file_issues'])} file issues:")
                    
                    with st.expander("📋 File Issues Details"):
                        for issue in preview_result["file_issues"]:
                            st.markdown(f"**{issue['type'].title()} Chapter {issue['chapter']}**: `{issue['file']}`")
                            for error in issue["errors"]:
                                st.error(f"ERROR: {error}")
                            for warning in issue["warnings"]:
                                st.warning(f"WARNING: {warning}")
                
                # Show warnings if any
                if preview_result.get("warnings"):
                    for warning in preview_result["warnings"]:
                        st.warning(f"⚠️ {warning}")
                
                # Build alignment map section
                st.divider()
                st.subheader("🔨 Build Alignment Map")
                
                # --- Custom Filename Input ---
                st.markdown("#### 📂 Output File Configuration")
                
                # Generate a more descriptive default name
                try:
                    chinese_dir_name = os.path.basename(os.path.normpath(chinese_dir))
                    english_dir_name = os.path.basename(os.path.normpath(english_dir))
                    safe_novel_name = re.sub(r'[^\w\s-]', '', selected_novel).strip().replace(' ', '_')
                    default_filename = f"{safe_novel_name}_{english_dir_name}_alignment_map.json"
                except:
                    default_filename = f"{selected_novel.replace(' ', '_')}_alignment_map.json"

                # Get the full default path
                default_output_path = os.path.join("data", "alignments", default_filename)

                output_path = st.text_input(
                    "Alignment Map Filename:",
                    value=default_output_path,
                    help="Enter the desired path for the new alignment map. It's recommended to use a descriptive name."
                )
                
                st.info(f"💾 The new alignment map will be saved to: `{output_path}`")
                # --- End Custom Filename Input ---
                
                # Build confirmation
                build_confirmed = st.checkbox(
                    f"I want to build alignment map for **{selected_novel}** with {stats['total_mappings']} chapters",
                    help="This will create an alignment map file that can be used for data review and translation"
                )
                
                if build_confirmed:
                    build_col1, build_col2 = st.columns(2)
                    
                    with build_col1:
                        if st.button("🔨 Build Alignment Map", type="primary", use_container_width=True):
                            st.session_state.alignment_output_path = output_path  # Save path to session state
                            st.session_state.alignment_build_active = True
                            st.rerun()
                    
                    with build_col2:
                        if st.button("❌ Cancel", use_container_width=True):
                            st.session_state.alignment_preview_active = False
                            st.rerun()
                
                # Handle build process
                if st.session_state.get('alignment_build_active', False):
                    final_output_path = st.session_state.get('alignment_output_path', get_alignment_map_path(selected_novel))
                    
                    with st.spinner(f"🔨 Building alignment map and saving to `{final_output_path}`..."):
                        success, message, build_stats = build_and_save_alignment_map(
                            chinese_dir, english_dir, selected_novel, output_path=final_output_path
                        )
                    
                    if success:
                        st.success(message)
                        st.session_state.alignment_preview_active = False
                        st.session_state.alignment_build_active = False
                        if 'alignment_output_path' in st.session_state:
                            del st.session_state.alignment_output_path
                        st.info("🔄 Reloading page to show alignment data...")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error(f"❌ Build failed: {message}")
                        
                        # Show detailed build stats if available
                        if build_stats.get("errors"):
                            st.error("**Build Errors:**")
                            for error in build_stats["errors"]:
                                st.error(f"• {error}")
                        
                        if build_stats.get("file_issues"):
                            st.warning("**File Issues:**")
                            for issue in build_stats["file_issues"]:
                                st.warning(f"• {issue['type'].title()} Chapter {issue['chapter']}: {issue['file']}")
                        
                        st.session_state.alignment_build_active = False
                        if 'alignment_output_path' in st.session_state:
                            del st.session_state.alignment_output_path
                
            else:
                st.error("❌ Preview failed")
                
                # Show detailed errors
                if preview_result.get("errors"):
                    st.error("**Validation Errors:**")
                    for error in preview_result["errors"]:
                        st.error(f"• {error}")
                
                if preview_result.get("file_issues"):
                    st.warning("**File Issues:**")
                    for issue in preview_result["file_issues"]:
                        st.warning(f"• {issue['type'].title()} Chapter {issue['chapter']}: {issue['file']}")
                
                st.session_state.alignment_preview_active = False
    
    else:
        st.info("💡 Enter both Chinese and English directory paths above to preview alignment mapping")
    
    st.stop()  # Stop here if no alignment map exists

# Create main content container for better organization
main_content = st.container()

if alignment_map:
    chapter_numbers = sorted([int(k) for k in alignment_map.keys()])
    
    # --- Sidebar Controls ---
    st.sidebar.header("🎛️ Controls")
    
    # AI Translation source functions now imported from utils.py with chapter-aware filtering
    
    # Smart chapter selection with navigation
    col1, col2, col3 = st.sidebar.columns([1, 2, 1])
    
    with col1:
        if st.button("◀ Prev", use_container_width=True, help="Go to previous chapter"):
            current_idx = chapter_numbers.index(st.session_state.current_chapter)
            if current_idx > 0:
                st.session_state.current_chapter = chapter_numbers[current_idx - 1]
                st.session_state.ai_translation = ""  # Clear translation cache
                st.rerun()
    
    with col2:
        # Get the index of current chapter for smart positioning
        try:
            current_index = chapter_numbers.index(st.session_state.current_chapter)
        except (ValueError, AttributeError):
            current_index = 0
            st.session_state.current_chapter = chapter_numbers[0]
        
        # Format chapter options for display
        chapter_options = [f"Ch. {x}" for x in chapter_numbers]
        selected_display = st.selectbox(
            "Chapter:", 
            options=chapter_options,
            index=current_index,
            label_visibility="collapsed"
        )
        # Extract the actual chapter number
        selected_chapter = int(selected_display.split(". ")[1]) if selected_display else chapter_numbers[0]
    
    with col3:
        if st.button("Next ▶", use_container_width=True, help="Go to next chapter"):
            current_idx = chapter_numbers.index(st.session_state.current_chapter)
            if current_idx < len(chapter_numbers) - 1:
                st.session_state.current_chapter = chapter_numbers[current_idx + 1]
                st.session_state.ai_translation = ""  # Clear translation cache
                st.rerun()
    
    # Quick jump section
    with st.sidebar.expander("🎯 Quick Jump"):
        jump_chapter = st.number_input(
            "Jump to Chapter:", 
            min_value=min(chapter_numbers), 
            max_value=max(chapter_numbers), 
            value=st.session_state.current_chapter,
            step=1
        )
        
        if st.button("🚀 Jump", use_container_width=True):
            if jump_chapter in chapter_numbers:
                st.session_state.current_chapter = jump_chapter
                st.session_state.ai_translation = ""
                st.rerun()
            else:
                st.error(f"Chapter {jump_chapter} not available")
    
    # Clear AI translation when chapter changes
    if st.session_state.current_chapter != selected_chapter:
        st.session_state.ai_translation = ""
        st.session_state.current_chapter = selected_chapter

    st.sidebar.divider()
    
    # --- Gemini Translation ---
    st.sidebar.header("🤖 Gemini AI Translation")
    
    # API Configuration - check availability but don't display status (shown on Home Dashboard)
    api_key, api_source = load_api_config()
    
    # --- Systematic Analysis Tab ---
    st.sidebar.divider()
    st.sidebar.header("📊 Systematic Analysis")
    
    # --- Binary Search for First Misalignment ---
    st.sidebar.subheader("🔍 Find First Misalignment")
    st.sidebar.caption("Use binary search to pinpoint exactly where alignment breaks")
    
    # Binary search parameters
    col1, col2 = st.sidebar.columns(2)
    with col1:
        binary_min = st.number_input("Search From", min_value=1, max_value=772, value=1, help="Start of search range")
    with col2:
        binary_max = st.number_input("Search To", min_value=1, max_value=772, value=772, help="End of search range")
    
    binary_threshold = st.sidebar.slider(
        "Alignment Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Similarity score below this value = misaligned"
    )
    
    if st.sidebar.button("🎯 Find First Misalignment", use_container_width=True, type="secondary"):
        if not api_key:
            st.sidebar.error("🔑 API key not configured")
        else:
            # Store binary search params and trigger search
            st.session_state.binary_search_params = {
                'min_chapter': binary_min,
                'max_chapter': binary_max,
                'threshold': binary_threshold
            }
            st.session_state.run_binary_search = True
            st.rerun()
    
    st.sidebar.divider()
    
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
            st.sidebar.error("🔑 API key not configured")
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
            st.sidebar.error("🔑 API key not configured")
        else:
            with st.spinner("🔄 Calling Gemini API..."):
                st.session_state.ai_translation = translate_with_gemini(raw_content, api_key, novel_name=st.session_state.selected_novel)

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

    # --- Chapter Tools ---
    st.sidebar.divider()
    st.sidebar.header("🛠️ Chapter Tools")
    
    # Split Chapter Tool
    if chapter_data.get("english_file") and os.path.exists(chapter_data["english_file"]):
        with st.sidebar.expander("✂️ Split Chapter Tool"):
            st.info(f"Use this if Chapter {selected_chapter}'s English file contains content from the next chapter.")
            
            # Show current chapter statistics for context
            current_content = load_chapter_content(chapter_data["english_file"])
            current_stats = get_text_stats(current_content, language_hint='english')
            
            st.markdown("**📊 Current Chapter Stats:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Words", f"{current_stats['word_count']:,}")
            with col2:
                st.metric("Characters", f"{current_stats['char_count']:,}")
            with col3:
                st.metric("Lines", f"{current_stats['line_count']:,}")
            
            # Alert if chapter looks suspiciously large
            if current_stats['word_count'] > 8000:
                st.warning("⚠️ **Chapter is unusually long** - likely contains merged content!")
            
            split_marker = st.text_input(
                "Start of Next Chapter:", 
                help='Paste the exact text that marks the beginning of the next chapter (e.g., "Chapter 225: Title")',
                placeholder="Chapter 225: Title Text..."
            )
            
            if split_marker:
                # Preview where the split would occur
                current_content = load_chapter_content(chapter_data["english_file"])
                if split_marker in current_content:
                    split_pos = current_content.find(split_marker)
                    st.success(f"✅ Split marker found at position {split_pos:,}")
                    
                    # Show preview of content before and after split
                    before_content = current_content[:split_pos].strip()
                    after_content = current_content[split_pos:].strip()
                    
                    # Get detailed statistics for both parts
                    before_stats = get_text_stats(before_content, language_hint='english')
                    after_stats = get_text_stats(after_content, language_hint='english')
                    
                    st.markdown("**📊 Split Statistics:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Will remain in Ch.{0}:**".format(selected_chapter))
                        st.metric("Characters", f"{before_stats['char_count']:,}")
                        st.metric("Words", f"{before_stats['word_count']:,}")
                        st.metric("Lines", f"{before_stats['line_count']:,}")
                    with col2:
                        st.markdown("**Will become Ch.{0}:**".format(selected_chapter + 1))
                        st.metric("Characters", f"{after_stats['char_count']:,}")
                        st.metric("Words", f"{after_stats['word_count']:,}")
                        st.metric("Lines", f"{after_stats['line_count']:,}")
                    
                    # Show word count ratio for sanity check
                    if before_stats['word_count'] > 0 and after_stats['word_count'] > 0:
                        ratio = before_stats['word_count'] / after_stats['word_count']
                        if ratio > 3.0 or ratio < 0.33:
                            st.warning(f"⚠️ Uneven split: {ratio:.1f}:1 ratio - double-check split position")
                        else:
                            st.success(f"✅ Balanced split: {ratio:.1f}:1 ratio")
                    
                    # Show preview text
                    with st.expander("📋 Preview Split"):
                        st.text_area("Will remain in current chapter:", before_content[-200:], height=100, disabled=True)
                        st.text_area("Will become new chapter:", after_content[:200], height=100, disabled=True)
                    
                    # Confirmation checkbox
                    split_confirmed = st.checkbox(
                        f"I want to split Chapter {selected_chapter} at this position",
                        help="This will create a new file and renumber subsequent chapters"
                    )
                    
                    if split_confirmed and st.button("✂️ Execute Split", type="primary", use_container_width=True):
                        with st.spinner("Splitting chapter and updating alignment..."):
                            success, message, changes = split_english_chapter(
                                chapter_data["english_file"],
                                split_marker,
                                selected_chapter,
                                alignment_map
                            )
                        
                        if success:
                            # Save the updated alignment map
                            backup_file = save_alignment_map_safely(alignment_map, "alignment_map.json")
                            
                            st.success("🎉 **Split successful!**")
                            st.info(f"📁 Backup saved: {backup_file}")
                            
                            # Show summary of changes
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Files Shifted", len(changes.get('files_shifted', [])))
                            with col2:
                                st.metric("Content Fixes", changes.get('total_content_fixes', 0))
                            with col3:
                                st.metric("Alignments Updated", len(changes.get('alignment_updates', [])))
                            with col4:
                                cascade_op = changes.get('cascade_operation', False)
                                st.metric("Cascade Operation", "Yes" if cascade_op else "No")
                            
                            # Show detailed changes
                            with st.expander("📊 Detailed Changes"):
                                if changes.get('cascade_operation'):
                                    st.success("🔄 **Robust Split and Shift** operation completed successfully!")
                                    st.info("All existing files were safely shifted to make room for the new chapter.")
                                
                                st.subheader("📁 File Operations")
                                st.text(f"✂️ Split: {changes.get('original_file')} → {changes.get('new_file')}")
                                
                                if changes.get('files_shifted'):
                                    st.text("🔄 Files shifted:")
                                    for shift in changes['files_shifted']:
                                        st.text(f"  📄 {shift}")
                                
                                st.subheader("🔧 Content Fixes")
                                content_fixes = changes.get('content_fixes', {})
                                if content_fixes.get('split_files'):
                                    st.text("Split files:")
                                    for fix in content_fixes['split_files']:
                                        st.text(f"  ✏️ {fix}")
                                
                                if content_fixes.get('shifted_files'):
                                    st.text("Shifted files:")
                                    for fix in content_fixes['shifted_files']:
                                        st.text(f"  ✏️ {fix}")
                                
                                if not content_fixes.get('split_files') and not content_fixes.get('shifted_files'):
                                    st.text("✅ No content numbering issues found")
                                
                                st.subheader("🗺️ Alignment Updates")
                                for update in changes.get('alignment_updates', []):
                                    st.text(f"🎯 {update}")
                                
                                st.subheader("📍 Split Details")
                                st.text(f"Marker used: {changes.get('split_marker_used', 'N/A')}")
                            
                            st.info("🔄 Page will reload in 3 seconds...")
                            time.sleep(3)
                            st.rerun()
                        else:
                            st.error(f"❌ Split failed: {message}")
                else:
                    st.warning("❌ Split marker not found in current chapter")
            else:
                st.caption("💡 **How to use:** Copy the exact text that starts the next chapter and paste it above")
    else:
        st.sidebar.info("✂️ Split tool requires an English chapter file")
    
    # --- Chapter Info ---
    st.sidebar.divider()
    st.sidebar.header("📋 Chapter Info")
    st.sidebar.info(f"**Currently viewing:** Chapter {selected_chapter}")
    
    # File availability status
    if chapter_data.get("english_file"):
        st.sidebar.success("✅ English translation available")
    else:
        st.sidebar.warning("❌ No English translation linked")
    
    if chapter_data.get("raw_file"):
        st.sidebar.success("✅ Chinese raw available")
    else:
        st.sidebar.warning("❌ No Chinese raw linked")
    
    # Text statistics
    st.sidebar.subheader("📊 Text Statistics")
    
    # Official English chapter stats (always from alignment map)
    if chapter_data.get("english_file"):
        eng_content = load_chapter_content(chapter_data["english_file"])
        eng_stats = get_text_stats(eng_content, language_hint='english')
        
        st.sidebar.markdown("**📖 English Chapter:**")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Words", f"{eng_stats['word_count']:,}")
            st.metric("Lines", f"{eng_stats['line_count']:,}")
        with col2:
            st.metric("Characters", f"{eng_stats['char_count']:,}")
            st.metric("Words/Line", f"{eng_stats['avg_words_per_line']}")
        
        # Suspicious size detection
        if eng_stats['word_count'] > 8000:  # Typical chapter is ~3000-5000 words
            st.sidebar.warning("⚠️ **Unusually long chapter** - possible merge detected!")
        elif eng_stats['word_count'] < 1000:
            st.sidebar.warning("⚠️ **Unusually short chapter** - possible content missing!")
    else:
        st.sidebar.warning("❌ No English translation linked")
    
    # Chinese chapter stats
    if chapter_data.get("raw_file"):
        raw_content = load_chapter_content(chapter_data["raw_file"])
        raw_stats = get_text_stats(raw_content, language_hint='chinese')
        
        st.sidebar.markdown("**📜 Chinese Chapter:**")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Chinese Chars", f"{raw_stats['word_count']:,}")
            st.metric("Lines", f"{raw_stats['line_count']:,}")
        with col2:
            st.metric("Total Chars", f"{raw_stats['char_count']:,}")
            st.metric("Chars/Line", f"{raw_stats['avg_words_per_line']}")
        
        # Suspicious size detection for Chinese (using character count)
        if raw_stats['word_count'] > 3000:  # Chinese characters - typical ~1500-2500
            st.sidebar.warning("⚠️ **Unusually long chapter** - possible merge detected!")
        elif raw_stats['word_count'] < 800:
            st.sidebar.warning("⚠️ **Unusually short chapter** - possible content missing!")
    
    # AI Translation stats (if available)
    if st.session_state.ai_translation and "API Request Failed" not in st.session_state.ai_translation:
        ai_stats = get_text_stats(st.session_state.ai_translation, language_hint='english')
        
        st.sidebar.markdown("**🤖 AI Translation:**")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Words", f"{ai_stats['word_count']:,}")
            st.metric("Lines", f"{ai_stats['line_count']:,}")
        with col2:
            st.metric("Characters", f"{ai_stats['char_count']:,}")
            st.metric("Words/Line", f"{ai_stats['avg_words_per_line']}")
        
        # Quality indicator based on length comparison with Chinese
        if chapter_data.get("raw_file"):
            raw_content = load_chapter_content(chapter_data["raw_file"])
            raw_stats = get_text_stats(raw_content, language_hint='chinese')
            
            if raw_stats['word_count'] > 0:
                ai_to_chinese_ratio = ai_stats['word_count'] / raw_stats['word_count']
                if ai_to_chinese_ratio < 1.0:
                    st.sidebar.warning("⚠️ **AI translation seems short** - possible truncation")
                elif ai_to_chinese_ratio > 4.0:
                    st.sidebar.warning("⚠️ **AI translation seems long** - possible repetition")
                else:
                    st.sidebar.success("✅ **AI translation length looks reasonable**")
    
    # BERT Similarity calculation moved to main content area (after AI source selection)

    # --- Main Content Display using container ---
    with main_content:
        # Execute binary search if requested
        if hasattr(st.session_state, 'run_binary_search') and st.session_state.run_binary_search:
            del st.session_state.run_binary_search
            
            st.header("🎯 Binary Search for First Misalignment")
            params = st.session_state.binary_search_params
            
            with st.spinner(f"🔍 Searching for first misalignment in chapters {params['min_chapter']}-{params['max_chapter']}..."):
                search_result = find_first_misalignment_binary_search(
                    alignment_map, 
                    api_key, 
                    params['min_chapter'], 
                    params['max_chapter'], 
                    params['threshold']
                )
            
            st.session_state.binary_search_result = search_result
        
        # Display binary search results
        if hasattr(st.session_state, 'binary_search_result'):
            result = st.session_state.binary_search_result
            
            st.header("🎯 Binary Search Results")
            
            if result['success']:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("First Misaligned Chapter", 
                             result['first_misaligned_chapter'] if result['first_misaligned_chapter'] else "None Found")
                with col2:
                    st.metric("Chapters Checked", result['total_chapters_checked'])
                with col3:
                    st.metric("Threshold Used", f"{result['threshold_used']:.2f}")
                with col4:
                    search_efficiency = f"{result['total_chapters_checked']}/~{2**10}"  # Log2 efficiency
                    st.metric("Search Efficiency", f"~{result['total_chapters_checked']} checks")
                
                if result['first_misaligned_chapter']:
                    st.success(f"🎯 **First misalignment found at Chapter {result['first_misaligned_chapter']}**")
                    
                    # Show corrective action options
                    st.subheader("🛠️ Surgical Correction Options")
                    
                    # Calculate suggested offset based on recent systematic analysis
                    suggested_offset = 0
                    if hasattr(st.session_state, 'systematic_analysis') and st.session_state.systematic_analysis:
                        offsets = [r["offset"] for r in st.session_state.systematic_analysis if r["score"] > 0.3]
                        if offsets:
                            from collections import Counter
                            suggested_offset = Counter(offsets).most_common(1)[0][0]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        correction_offset = st.number_input(
                            "Offset to Apply", 
                            min_value=-10, 
                            max_value=10, 
                            value=suggested_offset,
                            help="How many chapters to shift the alignment"
                        )
                    with col2:
                        start_from = st.number_input(
                            "Start Correction From Chapter", 
                            min_value=1, 
                            max_value=772, 
                            value=result['first_misaligned_chapter'],
                            help="Apply correction from this chapter onwards"
                        )
                    
                    st.info(f"💡 **Surgical correction:** Apply {correction_offset:+d} offset starting from Chapter {start_from}")
                    
                    # Preview the surgical correction
                    if st.button("📋 Preview Surgical Correction", use_container_width=True):
                        st.session_state.surgical_preview = {
                            'offset': correction_offset,
                            'start_from': start_from,
                            'search_result': result
                        }
                        st.rerun()
                else:
                    st.success("✅ **No misalignment found in the search range!**")
                    st.info("All tested chapters appear to be correctly aligned.")
                
                # Show detailed search log
                with st.expander("🔍 **Search Log Details**"):
                    search_df_data = []
                    for log_entry in result['search_log']:
                        search_df_data.append({
                            "Chapter": log_entry['chapter'],
                            "Similarity Score": f"{log_entry.get('similarity_score', 'N/A'):.3f}" if isinstance(log_entry.get('similarity_score'), float) else log_entry.get('similarity_score', 'N/A'),
                            "Action": log_entry['action'],
                            "Search Range": log_entry.get('search_range', 'N/A')
                        })
                    
                    if search_df_data:
                        search_df = pd.DataFrame(search_df_data)
                        st.dataframe(search_df, use_container_width=True)
                        
                        st.caption(f"🧠 Search completed in {len(search_df_data)} steps vs ~{result['threshold_used']*1000:.0f} steps for linear search")
            else:
                st.error(f"❌ **Binary search failed:** {result['error']}")
            
            st.divider()
        
        # Show surgical correction preview
        if hasattr(st.session_state, 'surgical_preview'):
            preview = st.session_state.surgical_preview
            
            st.header("🔧 Surgical Correction Preview")
            st.info(f"**Offset:** {preview['offset']:+d} | **Starting from:** Chapter {preview['start_from']}")
            
            # Generate preview using modified function
            correction_preview = preview_systematic_correction(alignment_map, preview['offset'], sample_size=15)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔴 BEFORE Surgical Correction")
                before_data = []
                for item in correction_preview["before"]:
                    # Highlight the starting chapter
                    status_icon = "🎯" if item["raw_ch"] == preview['start_from'] else ("✅" if item["eng_ch"] != "None" else "❌")
                    before_data.append({
                        "Raw Ch": item["raw_ch"],
                        "→ English Ch": item["eng_ch"],
                        "Status": status_icon
                    })
                st.dataframe(pd.DataFrame(before_data), use_container_width=True)
            
            with col2:
                st.subheader("🟢 AFTER Surgical Correction")
                after_data = []
                for item in correction_preview["after"]:
                    # Only apply to chapters >= start_from
                    will_change = item["raw_ch"] >= preview['start_from']
                    status_icon = "🎯" if item["raw_ch"] == preview['start_from'] else ("✅" if item["eng_ch"] != "None" else "❌")
                    after_data.append({
                        "Raw Ch": item["raw_ch"],
                        "→ English Ch": item["eng_ch"] if will_change else correction_preview["before"][item["raw_ch"]-correction_preview["before"][0]["raw_ch"]]["eng_ch"],
                        "Status": status_icon
                    })
                st.dataframe(pd.DataFrame(after_data), use_container_width=True)
            
            # Surgical correction controls
            st.subheader("🔧 Apply Surgical Correction")
            st.warning(f"⚠️ This will modify alignment for chapters {preview['start_from']}+ only. Chapters 1-{preview['start_from']-1} remain unchanged.")
            
            surgical_confirmed = st.checkbox(
                f"I want to apply surgical correction ({preview['offset']:+d}) from Chapter {preview['start_from']} onwards",
                help="This preserves the first part of your alignment and only fixes the problematic section."
            )
            
            if surgical_confirmed:
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✅ Apply Surgical Correction", type="primary", use_container_width=True):
                        # Apply the surgical correction
                        corrected_map = apply_systematic_correction(
                            alignment_map, 
                            preview['offset'], 
                            start_from_chapter=preview['start_from']
                        )
                        backup_file = save_alignment_map_safely(corrected_map, "alignment_map.json")
                        
                        st.success("🎉 **Surgical correction applied!**")
                        st.info(f"📁 Backup saved: {backup_file}")
                        st.info(f"🎯 Chapters {preview['start_from']}+ corrected, Chapters 1-{preview['start_from']-1} preserved")
                        st.info("🔄 Page will reload in 3 seconds...")
                        
                        # Clear analysis results
                        for key in ['binary_search_result', 'surgical_preview', 'systematic_analysis', 'correction_preview']:
                            if hasattr(st.session_state, key):
                                delattr(st.session_state, key)
                        
                        time.sleep(3)
                        st.rerun()
                
                with col2:
                    if st.button("❌ Cancel", use_container_width=True):
                        # Clear the preview
                        if hasattr(st.session_state, 'surgical_preview'):
                            del st.session_state.surgical_preview
                        st.rerun()
            
            st.divider()
        
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
            # Always show official English translation from alignment map
            eng_filepath = chapter_data.get("english_file")
            eng_content = load_chapter_content(eng_filepath)
            st.text_area("Official English Content", eng_content, height=600, key="eng_text")
            st.caption(f"File: {eng_filepath or 'Not available'}")
            
        with col3:
            st.subheader("🤖 AI Translation")
            
            # AI Translation Source Selector - only show sources with current chapter available
            available_ai_sources = get_available_ai_sources(selected_chapter, selected_novel)
            selected_ai_source = st.selectbox(
                "🎯 AI Source:",
                options=available_ai_sources,
                help="Choose AI translation source to display (filtered to show only sources with this chapter)"
            )
            
            # Store selected AI source in session state for sidebar access
            st.session_state.selected_ai_source = selected_ai_source
            
            # Handle different AI sources
            if selected_ai_source.startswith("Custom: "):
                # Load from custom translation run
                run_name = selected_ai_source[8:]  # Remove "Custom: " prefix
                custom_file = f"Chapter-{selected_chapter:04d}-translated.txt"
                custom_path = os.path.join(DATA_DIR, "custom_translations", run_name, custom_file)
                custom_content = load_chapter_content(custom_path)
                
                if "File not found" in custom_content:
                    st.error(f"❌ Chapter {selected_chapter} not found in {run_name}")
                    st.text_area("AI Generated Content", "Chapter not available in this custom run.", height=600, key="ai_text")
                    st.caption(f"📁 Source: {run_name} (Chapter not found)")
                else:
                    st.text_area("AI Generated Content", custom_content, height=600, key="ai_text")
                    st.caption(f"📁 Source: {run_name} | File: {os.path.basename(custom_path)}")
            
            elif selected_ai_source == "Fresh Gemini Translation":
                # Force fresh translation from Gemini
                if st.button("🔄 Generate Fresh Translation", use_container_width=True, type="primary"):
                    if api_key:
                        with st.spinner("🔄 Getting fresh translation..."):
                            st.session_state.ai_translation = translate_with_gemini(raw_content, api_key, use_cache=False, novel_name=st.session_state.selected_novel)
                    else:
                        st.error("🔑 API Key Required")
                
                st.text_area("AI Generated Content", st.session_state.ai_translation, height=600, key="ai_text")
                if st.session_state.ai_translation:
                    st.caption(f"🌐 Fresh from API • {len(st.session_state.ai_translation)} chars")
                else:
                    st.caption("Use button above to generate fresh AI translation")
            
            elif selected_ai_source == "Cached Gemini Translation":
                # Load cached translation or prompt to create one
                cached_translation = get_cached_translation(raw_content)
                cache_stats = get_translation_cache_stats()
                
                if cached_translation:
                    if st.button("⚡ Load Cached Translation", use_container_width=True, type="primary"):
                        st.session_state.ai_translation = cached_translation
                        st.rerun()
                    
                    st.text_area("AI Generated Content", st.session_state.ai_translation, height=600, key="ai_text")
                    if st.session_state.ai_translation:
                        is_cached = st.session_state.ai_translation == cached_translation
                        cache_indicator = "⚡ Cached" if is_cached else "🌐 Fresh from API"
                        st.caption(f"{cache_indicator} • {len(st.session_state.ai_translation)} chars")
                    else:
                        st.caption("Use button above to load cached translation")
                        
                    st.info(f"📚 Cache: {cache_stats['count']} translations ({cache_stats['size_mb']:.1f} MB)")
                else:
                    st.warning("❌ No cached translation available for this chapter")
                    if st.button("🔄 Create Cached Translation", use_container_width=True):
                        if api_key:
                            with st.spinner("🔄 Creating cached translation..."):
                                st.session_state.ai_translation = translate_with_gemini(raw_content, api_key, use_cache=True, novel_name=st.session_state.selected_novel)
                        else:
                            st.error("🔑 API Key Required")
                    
                    st.text_area("AI Generated Content", st.session_state.ai_translation, height=600, key="ai_text")
                    if st.session_state.ai_translation:
                        st.caption(f"🌐 Newly cached • {len(st.session_state.ai_translation)} chars")
                    else:
                        st.caption("No cached translation available - use button above to create one")
                    
                    st.caption(f"📚 Translation Cache: {cache_stats['count']} translations ({cache_stats['size_mb']:.1f} MB)")

        # BERT Similarity calculation (moved from sidebar for proper execution order)
        st.divider()
        st.subheader("🔄 Translation Comparison")
        
        # Get current AI translation content based on selected source
        if hasattr(st.session_state, 'selected_ai_source'):
            selected_ai_source = st.session_state.selected_ai_source
            
            # Get AI content based on the current selection
            if selected_ai_source.startswith("Custom: "):
                # For custom runs, get the content from the text area or load from file
                custom_content = None
                try:
                    run_name = selected_ai_source[8:]
                    custom_file = f"Chapter-{selected_chapter:04d}-translated.txt"
                    custom_path = os.path.join(DATA_DIR, "custom_translations", run_name, custom_file)
                    custom_content = load_chapter_content(custom_path)
                    if "File not found" not in custom_content:
                        ai_translation_content = custom_content
                    else:
                        ai_translation_content = None
                except:
                    ai_translation_content = None
            else:
                # For Gemini sources, use session state
                ai_translation_content = st.session_state.ai_translation if st.session_state.ai_translation else None
            
            # Calculate BERT similarity if both contents are available
            if ai_translation_content and chapter_data.get("english_file"):
                eng_content = load_chapter_content(chapter_data["english_file"])
                if eng_content and "File not found" not in eng_content:
                    with st.spinner("🧠 Calculating BERT similarity..."):
                        bert_similarity = calculate_similarity(ai_translation_content, eng_content)
                    
                    # Display result with appropriate styling
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        if bert_similarity >= 0.8:
                            st.success(f"🧠 **BERT Similarity: {bert_similarity:.3f}** - Excellent meaning preservation!")
                        elif bert_similarity >= 0.6:
                            st.warning(f"🧠 **BERT Similarity: {bert_similarity:.3f}** - Moderate semantic match")
                        elif bert_similarity >= 0.3:
                            st.error(f"🧠 **BERT Similarity: {bert_similarity:.3f}** - Low semantic alignment")
                        else:
                            st.error(f"🧠 **BERT Similarity: {bert_similarity:.3f}** - Very poor semantic match")
                    
                    with col2:
                        ai_source_display = selected_ai_source.replace("Custom: ", "").replace(" Translation", "")
                        st.metric("Comparing", f"{ai_source_display} vs Official", delta=f"{bert_similarity:.3f}")
                    
                    st.caption("💡 BERT similarity measures semantic meaning preservation between AI translation and official English")
                else:
                    st.info("📖 Official English content not available for comparison")
            else:
                if not ai_translation_content:
                    st.info("🤖 No AI translation content available for comparison")
                    st.caption("💡 Generate or load AI translation to see BERT similarity")
                else:
                    st.info("📖 Official English content not available for comparison")
        else:
            st.info("🎯 Select an AI source above to see BERT similarity comparison")

else:
    st.error("❌ Could not load alignment map. Please ensure 'alignment_map.json' exists.")
    st.info("💡 Run the appropriate setup scripts to create the alignment map first.")