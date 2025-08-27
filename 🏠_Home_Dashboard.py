
"""
🏠 Translation Framework Workbench
Main entry point for the multi-page Streamlit application
"""

import streamlit as st
import os
from utils.config import load_api_config, show_config_status
from utils.logging import logger
from utils.web_scraping import validate_scraping_url, streamlit_scraper
from utils.translation import translate_with_gemini
from utils.novel_config_creator import create_novel_config, get_novel_config_form_data

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
    - Build alignment maps from directory selection
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
    
    # Check for alignment maps in central location
    alignments_dir = os.path.join("data", "alignments")
    alignment_maps_found = 0
    
    if os.path.exists(alignments_dir):
        alignment_maps_found = len([f for f in os.listdir(alignments_dir) if f.endswith('_alignment_map.json')])
    
    # Also check legacy location
    legacy_alignment_exists = os.path.exists("alignment_map.json")
    
    if alignment_maps_found > 0:
        st.success(f"✅ **Alignment Maps**: {alignment_maps_found} found in central location")
        st.caption(f"📁 Location: `{alignments_dir}`")
    elif legacy_alignment_exists:
        st.info("📁 **Alignment Map**: Legacy location found")
        st.caption("Consider migrating to Data Review page for new central location")
    else:
        st.warning("⚠️ **Alignment Maps**: None found")
        st.caption("Build alignment maps in the Data Review & Alignment page")
    
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
    - Build alignment maps from directories
    - Align Chinese with English chapters
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
if st.session_state.get('scraping_conflict_data'):
    conflict = st.session_state.scraping_conflict_data
    st.error("🕵️ User Intervention Required: Chapter Mismatch")

    # --- Translation Logic ---
    translated_prev = None
    translated_curr = None
    translation_error = None
    try:
        api_key, _ = load_api_config()
        if not api_key:
            raise ValueError("Gemini API key not found in config.")

        with st.spinner("Translating chapter previews for context..."):
            # Translate previous chapter's end
            prev_prompt = f"Please provide a concise, high-quality English translation of the following excerpt, which is the END of a chapter:\n\n---\n{conflict['last_chapter_preview']}\n---\n"
            translated_prev = translate_with_gemini(prev_prompt, api_key, use_cache=True, novel_name="preview_translation")

            # Translate current chapter's start
            curr_prompt = f"Please provide a concise, high-quality English translation of the following excerpt, which is the START of a new chapter:\n\n---\n{conflict['current_chapter_preview']}\n---\n"
            translated_curr = translate_with_gemini(curr_prompt, api_key, use_cache=True, novel_name="preview_translation")

    except Exception as e:
        translation_error = f"Could not get translated previews: {e}"
        logger.warning(translation_error)
    # --- End Translation Logic ---
    
    with st.container(border=True):
        st.markdown(f"**Problematic URL:** `{conflict['url']}`")
        st.warning(f"The scraper expected to find **Chapter {conflict['expected_number']}**, but the page title says it is **Chapter {conflict['found_number']}**.")

        # Display translated previews if available
        if translated_prev and translated_curr:
            st.success("✅ English previews generated successfully.")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"End of Previous Chapter ({conflict['expected_number'] - 1})")
                st.text_area("Translated Preview", f"...{translated_prev.strip()}", height=150, disabled=True, key="prev_trans")
                with st.expander("View Raw Chinese Text"):
                    st.text_area("Raw Preview", f"...{conflict['last_chapter_preview']}", height=150, disabled=True, key="prev_raw")
            with col2:
                st.subheader(f"Start of Current Chapter ({conflict['found_number']})")
                st.text_area("Translated Preview", f"{translated_curr.strip()}...", height=150, disabled=True, key="curr_trans")
                with st.expander("View Raw Chinese Text"):
                    st.text_area("Raw Preview", f"{conflict['current_chapter_preview']}...", height=150, disabled=True, key="curr_raw")
        else:
            # Fallback to raw text if translation failed
            if translation_error:
                st.warning(f"**Translation Failed:** {translation_error}. Displaying raw Chinese text instead.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"End of Previous Chapter ({conflict['expected_number'] - 1})")
                st.text_area("Previous Chapter Preview", f"...{conflict['last_chapter_preview']}", height=150, disabled=True)
            with col2:
                st.subheader(f"Start of Current Chapter ({conflict['found_number']})")
                st.text_area("Current Chapter Preview", f"{conflict['current_chapter_preview']}...", height=150, disabled=True)

        st.subheader("How should the scraper proceed?")
        
        # --- Resolution Options ---
        btn_col1, btn_col2, btn_col3 = st.columns(3)
        
        # Option 1: Use Expected Number
        with btn_col1:
            if st.button(f"✅ Use Expected ({conflict['expected_number']})", type="primary", use_container_width=True):
                st.session_state.resume_payload = conflict
                st.session_state.scraping_override = {'type': 'expected'}
                st.session_state.scraping_conflict_data = None
                st.session_state.scraping_active = True
                st.rerun()

        # Option 2: Use Title's Number
        with btn_col2:
            if conflict.get('found_number') is not None:
                if st.button(f"⚠️ Use Title ({conflict['found_number']})", use_container_width=True):
                    st.session_state.resume_payload = conflict
                    st.session_state.scraping_override = {'type': 'title'}
                    st.session_state.scraping_conflict_data = None
                    st.session_state.scraping_active = True
                    st.rerun()
            else:
                st.button("Use Title (N/A)", disabled=True, use_container_width=True)

        # Option 3: Abort
        with btn_col3:
            if st.button("🛑 Abort Scraping", use_container_width=True):
                st.session_state.scraping_conflict_data = None
                st.session_state.scraping_active = False
                st.warning("Scraping aborted by user.")
                st.rerun()

        # Option 4: Custom Number Input Form
        with st.form("custom_number_form"):
            st.write("**Or, enter a custom chapter number:**")
            custom_chapter_num = st.number_input(
                "Custom Chapter #", 
                min_value=1, 
                step=1, 
                value=conflict.get('expected_number'),
                label_visibility="collapsed"
            )
            
            if st.form_submit_button("➡️ Use Custom Number", use_container_width=True):
                st.session_state.resume_payload = conflict
                st.session_state.scraping_override = {
                    'type': 'custom',
                    'number': custom_chapter_num
                }
                st.session_state.scraping_conflict_data = None
                st.session_state.scraping_active = True
                st.rerun()

# --- ACTIVE SCRAPING UI ---
elif st.session_state.get('scraping_active', False):
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
        logger.info("--- CONFLICT HANDLER ACTIVATED ---")
        st.session_state.scraping_conflict_data = {
            "url": url, "expected_number": expected_number, "found_number": found_number,
            "title": title, "is_parse_error": is_parse_error,
            "last_chapter_preview": last_chapter_preview, "current_chapter_preview": current_chapter_preview
        }
        st.session_state.scraping_active = False
        logger.info("--- CONFLICT HANDLER FINISHED ---")
        return 'defer'

    try:
        results = streamlit_scraper(
            start_url=st.session_state.novel_url,
            output_dir=st.session_state.output_directory,
            max_chapters=st.session_state.max_chapters,
            delay_seconds=st.session_state.delay_seconds,
            progress_callback=progress_callback,
            status_callback=status_callback,
            conflict_handler=conflict_handler,
            scrape_direction=st.session_state.scrape_direction
        )
        
        if results and results.get('status') == 'conflict':
            st.session_state.scraping_conflict_data = results['data']
        elif results and results.get('status') == 'completed':
            # Scraping completed successfully - offer to create config
            st.session_state.scraping_results = results
            st.session_state.scraping_completed = True
        
    except Exception as e:
        st.error(f"💥 Scraping failed with a critical error: {str(e)}")
        st.session_state.scraping_active = False
    finally:
        if not st.session_state.get('scraping_conflict_data'):
            st.session_state.scraping_active = False
        st.rerun()

# --- POST-SCRAPING NOVEL CONFIG CREATION ---
if st.session_state.get('scraping_completed') and not st.session_state.get('config_creation_dismissed'):
    st.markdown("---")
    st.header("🔧 Configure Your Novel")
    st.info("✅ **Scraping completed successfully!** Would you like to create a novel configuration to make it available in the experimentation lab?")
    
    results = st.session_state.get('scraping_results', {})
    novel_name = os.path.basename(results.get('output_dir', ''))
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("⏭️ Skip Configuration", help="You can create this configuration later if needed"):
            st.session_state.config_creation_dismissed = True
            st.session_state.scraping_completed = False
            st.rerun()
    
    with col1:
        st.subheader(f"📖 Novel: {novel_name}")
        
        # Show scraping summary
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        with summary_col1:
            st.metric("📄 Chapters Scraped", results.get('chapters_scraped', 0))
        with summary_col2:
            st.metric("🌐 Source", results.get('adapter_type', 'Unknown'))
        with summary_col3:
            st.metric("📁 Directory", novel_name)
        
        # Novel configuration form
        form_data = get_novel_config_form_data()
        
        with st.form("novel_config_form"):
            st.write("### Basic Information")
            
            config_values = {}
            
            # Basic info fields
            for field in form_data["basic_info"]:
                if field["type"] == "text":
                    default_value = novel_name.replace('_', ' ').title() if field["key"] == "title" else ""
                    config_values[field["key"]] = st.text_input(
                        field["label"], 
                        value=default_value,
                        help=field["help"]
                    )
                elif field["type"] == "textarea":
                    config_values[field["key"]] = st.text_area(
                        field["label"],
                        help=field["help"]
                    )
                elif field["type"] == "select":
                    default_idx = 0
                    if field["key"] == "status":
                        default_idx = field["options"].index("ongoing")
                    config_values[field["key"]] = st.selectbox(
                        field["label"],
                        field["options"],
                        index=default_idx,
                        help=field["help"]
                    )
            
            st.write("### Translation Settings")
            
            # Translation settings fields  
            for field in form_data["translation_settings"]:
                if field["type"] == "select":
                    default_idx = 0  # First option as default
                    config_values[field["key"]] = st.selectbox(
                        field["label"],
                        field["options"],
                        index=default_idx,
                        help=field["help"]
                    )
            
            # Submit buttons
            submit_col1, submit_col2 = st.columns(2)
            
            with submit_col1:
                if st.form_submit_button("✅ Create Configuration", type="primary", use_container_width=True):
                    # Create the novel config
                    novel_dir = results.get('output_dir', '')
                    scraping_info = {
                        'start_url': results.get('start_url', ''),
                        'chapters_scraped': results.get('chapters_scraped', 0),
                        'direction': results.get('direction', 'forwards'),
                        'adapter_type': results.get('adapter_type', 'unknown')
                    }
                    
                    success, message, config_path = create_novel_config(
                        novel_dir, 
                        scraping_info, 
                        config_values
                    )
                    
                    if success:
                        st.success(f"🎉 **Novel configuration created successfully!**")
                        st.success(f"📄 Config saved to: `{os.path.basename(config_path)}`")
                        st.success("🚀 **Your novel is now available in the Experimentation Lab!**")
                        st.balloons()
                        
                        # Clear the completion state
                        st.session_state.scraping_completed = False
                        st.session_state.scraping_results = None
                        st.session_state.config_creation_dismissed = False
                        
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to create configuration: {message}")
            
            with submit_col2:
                if st.form_submit_button("🔄 Create Basic Config", use_container_width=True, help="Create a basic configuration with auto-detected settings"):
                    # Create basic config with minimal user input
                    novel_dir = results.get('output_dir', '')
                    scraping_info = {
                        'start_url': results.get('start_url', ''),
                        'chapters_scraped': results.get('chapters_scraped', 0),
                        'direction': results.get('direction', 'forwards'),
                        'adapter_type': results.get('adapter_type', 'unknown')
                    }
                    
                    # Use only title and author from form, auto-detect rest
                    basic_config = {
                        'title': config_values.get('title', novel_name.replace('_', ' ').title()),
                        'author': config_values.get('author', 'Unknown')
                    }
                    
                    success, message, config_path = create_novel_config(
                        novel_dir, 
                        scraping_info, 
                        basic_config
                    )
                    
                    if success:
                        st.success(f"🎉 **Basic novel configuration created!**")
                        st.success("🚀 **Your novel is now available in the Experimentation Lab!**")
                        
                        # Clear the completion state
                        st.session_state.scraping_completed = False
                        st.session_state.scraping_results = None
                        st.session_state.config_creation_dismissed = False
                        
                        st.rerun()
                    else:
                        st.error(f"❌ Failed to create basic configuration: {message}")

# --- STANDARD SCRAPING UI ---
elif not st.session_state.get('scraping_active'):
    scraping_col1, scraping_col2 = st.columns([2, 1])
    with scraping_col1:
        st.subheader("📖 Novel URL Input")
    
        novel_url = st.text_input(
            "Novel Chapter URL:",
            placeholder="https://www.dxmwx.org/read/43713_33325507.html",
            help="Enter the URL of a chapter from the novel you want to scrape"
        )
        
        output_directory_name = "novel_raws/new_novel"
        if novel_url:
            validation = validate_scraping_url(novel_url)
            if validation['valid']:
                st.success(f"✅ Supported site: {validation['site_type']}")
                try:
                    # This logic can be simplified as it's just for a default name
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
            max_chapters = st.number_input("Max Chapters to Scrape:", min_value=1, max_value=3000, value=500)
            output_directory = st.text_input("Output Directory:", value=output_directory_name)
        
        with scraping_config_col2:
            delay_seconds = st.slider("Delay Between Requests (seconds):", min_value=0.5, max_value=10.0, value=0.5, step=0.5)
            scrape_direction = st.selectbox("Scraping Direction:", ["Backwards (newest to oldest)", "Forwards (oldest to newest)"], index=1)

        col_start, col_stop, _ = st.columns([1, 1, 5])
        
        with col_start:
            if st.button("🚀 Start Scraping", type="primary"):
                st.session_state.scraping_active = True
                st.session_state.novel_url = novel_url
                st.session_state.output_directory = output_directory
                st.session_state.max_chapters = max_chapters
                st.session_state.delay_seconds = delay_seconds
                st.session_state.scrape_direction = scrape_direction
                st.session_state.scraping_conflict_data = None
                st.session_state.scraping_override = None
                st.rerun()

        with col_stop:
            # This button is only relevant when scraping is active, handled by the other UI block
            pass


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
