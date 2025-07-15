"""
🏠 Translation Framework Workbench
Main entry point for the multi-page Streamlit application
"""

import streamlit as st
import os
import time
from utils.config import load_api_config, show_config_status
from utils.logging import logger
from utils.web_scraping import validate_scraping_url, streamlit_scraper
from utils.alignment_builder import streamlit_build_alignment_map, detect_novel_structure

# Page configuration
st.set_page_config(
    page_title="Translation Framework Workbench",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("🏠 Translation Framework Workbench")
st.caption("**Complete MLOps Pipeline for Translation Models** | Data curation, training, experimentation, and analysis")

# Quick navigation info
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.header("🎯 Available Tools")
    st.markdown("""
    **Use the sidebar navigation** ← to access all workbench tools:
    
    📖 **Data Review & Alignment**
    - Chapter-by-chapter quality control and manual corrections
    - Binary search for misalignment detection
    - Surgical alignment corrections and chapter splitting
    
    🤖 **Fine-tuning Workbench**
    - Dataset preparation and quality analysis
    - Model training with hyperparameter control
    - Real-time training monitoring with loss curves
    - Model management and metadata tracking
    
    🧪 **Pluralistic Translation Lab**
    - Multi-style translation generation with custom prompts
    - In-context learning with few-shot examples
    - Real-time progress tracking and file persistence
    - Custom translation run management
    
    📈 **Experimentation Lab**
    - Quick translation testing (base vs fine-tuned models)
    - Batch evaluation with statistical significance
    - Performance analysis and visualization
    - Model leaderboard with composite scoring
    """)

with col2:
    st.header("⚙️ System Status")
    
    # API Configuration Status
    api_key, api_source = load_api_config()
    config_status = show_config_status()
    
    if api_key:
        st.success(f"🔑 **API Configuration**: {config_status}")
    else:
        st.warning("🔑 **API Configuration**: Not configured")
        with st.expander("🔧 API Setup Instructions"):
            st.markdown("""
            **Setup Instructions:**
            
            **Option A (Recommended):** Set environment variable:
            ```bash
            export GEMINI_API_KEY="your-api-key-here"
            ```
            
            **Option B:** Create `config.json`:
            ```bash
            cp config.example.json config.json
            # Edit config.json with your API key
            ```
            
            **Get your API key from:** https://aistudio.google.com/app/apikey
            """)
    
    # Data availability check
    st.subheader("📊 Data Status")
    
    alignment_map_exists = os.path.exists("alignment_map.json")
    if alignment_map_exists:
        st.success("✅ **Alignment Map**: Available")
    else:
        st.warning("⚠️ **Alignment Map**: Not found")
        st.caption("Run data curation on the Data Review page to create alignment map")
    
    # Directory structure check
    data_dirs = ["data", "data/cache", "data/custom_translations"]
    all_dirs_exist = all(os.path.exists(d) for d in data_dirs)
    
    if all_dirs_exist:
        st.success("✅ **Data Directories**: Configured")
    else:
        st.info("📁 **Data Directories**: Will be created automatically")
    
    # Custom translations check
    custom_translations_dir = "data/custom_translations"
    if os.path.exists(custom_translations_dir):
        custom_runs = [d for d in os.listdir(custom_translations_dir) 
                      if os.path.isdir(os.path.join(custom_translations_dir, d))]
        if custom_runs:
            st.success(f"🎨 **Custom Translation Runs**: {len(custom_runs)} available")
        else:
            st.info("🎨 **Custom Translation Runs**: None yet")
    else:
        st.info("🎨 **Custom Translation Runs**: None yet")
    
    # Build Alignment Map section
    st.subheader("🔨 Build Alignment Map")
    
    # Check for alignment map and detect structure
    alignment_map_exists = os.path.exists("alignment_map.json")
    structure_info = detect_novel_structure("way_of_the_devil")
    
    if alignment_map_exists:
        st.success("✅ **Alignment Map**: Found")
        st.caption(f"📁 Location: alignment_map.json")
    else:
        st.warning("❌ **Alignment Map**: Not found")
        st.caption("Click the button below to build alignment map from available chapters")
    
    # Show detected structure
    if structure_info["structure_type"]:
        st.info(f"📂 **Structure**: {structure_info['structure_type']}")
        if structure_info["english_dir"] and os.path.exists(structure_info["english_dir"]):
            english_count = len([f for f in os.listdir(structure_info["english_dir"]) if f.endswith('.txt')])
            st.caption(f"🇺🇸 English chapters: {english_count} found")
        if structure_info["chinese_dir"] and os.path.exists(structure_info["chinese_dir"]):
            chinese_count = len([f for f in os.listdir(structure_info["chinese_dir"]) if f.endswith('.txt')])
            st.caption(f"🇨🇳 Chinese chapters: {chinese_count} found")
    else:
        st.error("❌ **No chapter directories found**")
        st.caption("Please ensure English and/or Chinese chapters are available")
    
    # Build button and progress handling
    col1, col2 = st.columns(2)
    
    with col1:
        build_disabled = not structure_info["structure_type"]
        if st.button("🔨 Build Alignment Map", disabled=build_disabled, help="Build alignment map from available chapters"):
            if not build_disabled:
                st.session_state.build_alignment_active = True
                st.session_state.build_force_rebuild = False
                st.rerun()
    
    with col2:
        if st.button("🔄 Force Rebuild", disabled=build_disabled, help="Force complete rebuild ignoring existing alignment"):
            if not build_disabled:
                st.session_state.build_alignment_active = True
                st.session_state.build_force_rebuild = True
                st.rerun()
    
    # Handle build process
    if st.session_state.get('build_alignment_active', False):
        with st.spinner("Building alignment map..."):
            success, message, build_stats = streamlit_build_alignment_map(
                novel_name="way_of_the_devil",
                force_rebuild=st.session_state.get('build_force_rebuild', False)
            )
            
            if success:
                st.success(message)
                st.session_state.build_alignment_active = False
                st.rerun()
            else:
                st.error(f"❌ Build failed: {message}")
                if build_stats.get("error"):
                    st.error(build_stats["error"])
                st.session_state.build_alignment_active = False

# Quick start workflow
st.markdown("---")
st.header("🚀 Complete Workflow")

workflow_col0, workflow_col1, workflow_col2, workflow_col3, workflow_col4 = st.columns(5)

with workflow_col0:
    st.markdown("""
    **0. 🌐 Web Scraping**
    - Scrape raw Chinese chapters
    - Live progress monitoring
    - Ethical scraping practices
    """)

with workflow_col1:
    st.markdown("""
    **1. 📖 Data Review**
    - Align Chinese with English
    - Fix misaligned chapters
    - Quality control checks
    """)

with workflow_col2:
    st.markdown("""
    **2. 🧪 Translation Lab**
    - Generate custom translations
    - Experiment with styles
    - Build translation datasets
    """)

with workflow_col3:
    st.markdown("""
    **3. 🤖 Fine-tuning**
    - Train custom models
    - Monitor training progress
    - Export model metadata
    """)

with workflow_col4:
    st.markdown("""
    **4. 📈 Evaluation**
    - Multi-dimensional scoring
    - BERT + human assessment
    - Style leaderboards
    """)

# Live Web Scraping Interface
st.markdown("---")
st.header("🌐 Live Web Scraping")
st.caption("Step 0: Scrape raw Chinese chapters from web novels before data alignment")

# --- CONFLICT RESOLUTION UI ---
if 'scraping_conflict_data' in st.session_state and st.session_state.scraping_conflict_data:
    conflict = st.session_state.scraping_conflict_data
    st.error("🕵️ User Intervention Required: Chapter Mismatch")
    
    with st.container(border=True):
        st.markdown(f"**Problematic URL:** `{conflict['url']}`")
        st.warning(f"The scraper expected to find **Chapter {conflict['expected_number']}**, but the page title says it is **Chapter {conflict['found_number']}**.")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"End of Previous Chapter ({conflict['expected_number'] - 1})")
            st.text_area("Previous Chapter Preview", f"...{conflict['last_chapter_preview']}", height=150, disabled=True)
        with col2:
            st.subheader(f"Start of Current Chapter ({conflict['found_number']})")
            st.text_area("Current Chapter Preview", f"{conflict['current_chapter_preview']}...", height=150, disabled=True)

        st.subheader("How should the scraper proceed?")
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        with btn_col1:
            if st.button(f"✅ Use Expected Number ({conflict['expected_number']})", type="primary"):
                logger.debug(f"[UI] User chose 'expected'. Setting override and rerunning.")
                st.session_state.scraping_override = 'expected'
                st.session_state.scraping_active = True
                st.rerun()
        with btn_col2:
            if st.button(f"⚠️ Use Title's Number ({conflict['found_number']})"):
                logger.debug(f"[UI] User chose 'title'. Setting override and rerunning.")
                st.session_state.scraping_override = 'title'
                st.session_state.scraping_active = True
                st.rerun()
        with btn_col3:
            if st.button("🛑 Abort Scraping"):
                logger.debug("[UI] User chose 'abort'. Clearing conflict and stopping.")
                st.session_state.scraping_conflict_data = None
                st.session_state.scraping_active = False
                st.warning("Scraping aborted by user.")
                st.rerun()
else:
    # --- STANDARD SCRAPING UI ---
    scraping_col1, scraping_col2 = st.columns([2, 1])
    with scraping_col1:
        st.subheader("📖 Novel URL Input")
    
        # URL input
        novel_url = st.text_input(
            "Novel Chapter URL:",
            placeholder="https://www.dxmwx.org/read/43713_33325507.html",
            help="Enter the URL of a chapter from the novel you want to scrape"
        )
        
        # Dynamic output directory
        output_directory_name = "novel_raws/new_novel"
        validation = {'valid': False}
        if novel_url:
            validation = validate_scraping_url(novel_url)
            if validation['valid']:
                st.success(f"✅ Supported site: {validation['site_type']}")
                try:
                    import requests
                    from bs4 import BeautifulSoup
                    from utils.adapter_factory import get_adapter
                    
                    adapter = get_adapter(novel_url)
                    response = requests.get(novel_url, timeout=10)
                    response.encoding = adapter.get_encoding()
                    soup = BeautifulSoup(response.text, 'html.parser')
                    title = adapter.extract_title(soup)
                    
                    if title:
                        sanitized_title = title.split(' ')[0].strip().replace(':', '_').replace(' ', '_').lower()
                        site_name = validation.get('site_type', 'unknown').lower()
                        output_directory_name = f"data/novels/{sanitized_title}_{site_name}/raw_chapters"

                except Exception as e:
                    st.warning(f"Could not pre-fill directory name: {e}")
            else:
                st.error("Unsupported website.")

        st.subheader("⚙️ Scraping Configuration")
        
        scraping_config_col1, scraping_config_col2 = st.columns(2)
        
        with scraping_config_col1:
            max_chapters = st.number_input(
                "Max Chapters to Scrape:",
                min_value=1,
                max_value=3000,
                value=500,
                help="Maximum number of chapters to scrape in this session"
            )
            
            output_directory = st.text_input(
                "Output Directory:",
                value=output_directory_name,
                help="Directory name where chapters will be saved"
            )
        
        with scraping_config_col2:
            delay_seconds = st.slider(
                "Delay Between Requests (seconds):",
                min_value=0.5,
                max_value=10.0,
                value=0.5,
                step=0.5,
                help="Delay to be respectful to the website server"
            )
            
            scrape_direction = st.selectbox(
                "Scraping Direction:",
                ["Backwards (newest to oldest)", "Forwards (oldest to newest)"],
                index=1,
                help="Direction to follow chapter navigation links"
            )

        # Button logic
        col_start, col_stop, _ = st.columns([1, 1, 5])
        
        with col_start:
            start_button_disabled = st.session_state.get('scraping_active', False)
            if st.button("🚀 Start Scraping", type="primary", disabled=start_button_disabled):
                st.session_state.scraping_active = True
                st.session_state.stop_requested = False
                st.session_state.scraping_results = None
                st.session_state.scraping_conflict_data = None
                st.session_state.scraping_override = None
                st.rerun()

        with col_stop:
            if st.session_state.get('scraping_active', False):
                if st.button("🛑 Stop Scraping"):
                    st.session_state.stop_requested = True
                    st.warning("Stop request received. Finishing current chapter...")

    # Main scraping execution block
    if st.session_state.get('scraping_active', False):
        progress_container = st.container()
        status_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            progress_text = st.empty()
        
        with status_container:
            status_text = st.empty()
            status_text.info("🌐 Initializing scraper...")
        
        def progress_callback(current, total):
            progress = current / total if total > 0 else 0
            progress_bar.progress(progress)
            progress_text.text(f"Progress: {current}/{total} chapters ({progress:.1%})")
        
        def status_callback(message):
            status_text.info(message)

        def conflict_handler(url, expected_number, found_number, title, is_parse_error=False, last_chapter_preview=None, current_chapter_preview=None):
            logger.debug("[UI] Conflict detected. Setting state for UI resolution.")
            st.session_state.scraping_conflict_data = {
                "url": url,
                "expected_number": expected_number,
                "found_number": found_number,
                "title": title,
                "is_parse_error": is_parse_error,
                "last_chapter_preview": last_chapter_preview,
                "current_chapter_preview": current_chapter_preview
            }
            st.session_state.scraping_active = False
            # Do NOT call st.rerun() here. Let the scraper thread exit gracefully.
            return 'defer'
        
        try:
            logger.debug(f"[DASHBOARD] PRE-CALL check: status_callback is {status_callback}")
            streamlit_scraper(
                start_url=novel_url,
                output_dir=output_directory,
                max_chapters=max_chapters,
                delay_seconds=delay_seconds,
                progress_callback=progress_callback,
                status_callback=status_callback,
                conflict_handler=conflict_handler,
                scrape_direction=scrape_direction
            )
            
        except Exception as e:
            st.error(f"💥 Scraping failed with a critical error: {str(e)}")
            st.session_state.scraping_active = False
        finally:
            # This block now only handles the state transition after the scraper
            # has finished its work (or failed). The rerun is now only triggered
            # by explicit user action.
            if not st.session_state.get('scraping_conflict_data'):
                logger.debug("[UI] Scraper finished without conflict. Deactivating scraping state.")
                st.session_state.scraping_active = False
                st.rerun() # Rerun only on clean exit

    with scraping_col2:
        st.subheader("📊 Scraping Status")

        novels_dir = "data/novels"
        if not os.path.exists(novels_dir):
            os.makedirs(novels_dir)

        existing_novels = [d for d in os.listdir(novels_dir) if os.path.isdir(os.path.join(novels_dir, d))]

        if existing_novels:
            st.metric("📚 Existing Scraped Novels", len(existing_novels))

            with st.expander("📖 View Scraped Novels"):
                for novel_dir in sorted(existing_novels):
                    raw_chapters_path = os.path.join(novels_dir, novel_dir, "raw_chapters")
                    if os.path.exists(raw_chapters_path):
                        num_files = len([f for f in os.listdir(raw_chapters_path) if f.endswith('.txt')])
                        st.text(f"- {novel_dir} ({num_files} chapters)")
                    else:
                        st.text(f"- {novel_dir} (no raw chapters found)")
        else:
            st.info("📭 No novels scraped yet.")
            st.caption(f"Scraped data will be saved in the `{novels_dir}` directory.")
        
        if st.session_state.get('scraping_results'):
            # ... (results display logic)
            pass