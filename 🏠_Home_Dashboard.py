"""
🏠 Translation Framework Workbench
Main entry point for the multi-page Streamlit application
"""

import streamlit as st
import os
import time
from utils.config import load_api_config, show_config_status
from utils.web_scraping import validate_scraping_url, streamlit_scraper

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
                    # Sanitize title for directory name
                    sanitized_title = title.split(' ')[0].strip()                        .replace(':', '_').replace(' ', '_').lower()
                    site_name = validation.get('site_type', 'unknown').lower()
                    # All novels are now stored in the novel_raws directory
                    output_directory_name = f"novel_raws/novel_{sanitized_title}_{site_name}"

            except Exception as e:
                st.warning(f"Could not pre-fill directory name: {e}")
        else:
            st.error("Unsupported website.")

    # Scraping configuration
    st.subheader("⚙️ Scraping Configuration")
    
    scraping_config_col1, scraping_config_col2 = st.columns(2)
    
    with scraping_config_col1:
        max_chapters = st.number_input(
            "Max Chapters to Scrape:",
            min_value=1,
            max_value=3000,
            value=50,
            help="Maximum number of chapters to scrape in this session (772+ needed to match English chapters)"
        )
        
        output_directory = st.text_input(
            "Output Directory:",
            value=output_directory_name,
            help="Directory name where chapters will be saved"
        )
    
    with scraping_config_col2:
        delay_seconds = st.slider(
            "Delay Between Requests (seconds):",
            min_value=1.0,
            max_value=10.0,
            value=2.0,
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
            st.rerun() # Rerun to show the stop button and disable the start button

    with col_stop:
        if st.session_state.get('scraping_active', False):
            if st.button("🛑 Stop Scraping"):
                st.session_state.stop_requested = True
                st.warning("Stop request received. Finishing current chapter...")

    # Main scraping execution block
    if st.session_state.get('scraping_active', False):
        # Create containers for real-time updates
        progress_container = st.container()
        status_container = st.container()
        results_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            progress_text = st.empty()
        
        with status_container:
            status_text = st.empty()
            status_text.info("🌐 Initializing scraper...")
        
        # Define callback functions for real-time updates
        def progress_callback(current, total):
            progress = current / total if total > 0 else 0
            progress_bar.progress(progress)
            progress_text.text(f"Progress: {current}/{total} chapters ({progress:.1%})")
        
        def status_callback(message):
            status_text.info(message)
        
        # Start scraping
        try:
            scraping_results = streamlit_scraper(
                start_url=novel_url,
                output_dir=output_directory,
                max_chapters=max_chapters,
                delay_seconds=delay_seconds,
                progress_callback=progress_callback,
                status_callback=status_callback,
                scrape_direction=scrape_direction
            )
            
            st.session_state.scraping_results = scraping_results
            
        except Exception as e:
            st.error(f"💥 Scraping failed with a critical error: {str(e)}")
            st.session_state.scraping_results = {
                'success': False, 'errors': [str(e)], 'chapters_scraped': 0, 'total_time': 0
            }
        finally:
            # Always turn off scraping active state and rerun
            st.session_state.scraping_active = False
            st.rerun()

with scraping_col2:
    st.subheader("📊 Scraping Status")
    
    # Show existing scraped data from the central directory
    raws_dir = "novel_raws"
    if not os.path.exists(raws_dir):
        os.makedirs(raws_dir)

    existing_novels = [d for d in os.listdir(raws_dir) if os.path.isdir(os.path.join(raws_dir, d))]
    
    if existing_novels:
        st.metric("📚 Existing Scraped Novels", len(existing_novels))
        
        with st.expander("📖 View Scraped Novels"):
            for novel_dir in sorted(existing_novels):
                num_files = len([f for f in os.listdir(os.path.join(raws_dir, novel_dir)) if f.endswith('.txt')])
                st.text(f"- {novel_dir} ({num_files} chapters)")
    else:
        st.info("📭 No novels scraped yet.")
        st.caption(f"Scraped data will be saved in the `{raws_dir}` directory.")
    
    # Show scraping results if available
    if st.session_state.get('scraping_results'):
        results = st.session_state.scraping_results
        
        st.subheader("📈 Last Scraping Results")
        
        if results.get('success', False):
            st.success("✅ Success")
            st.metric("Chapters", results.get('chapters_scraped', 0))
            st.metric("Time", f"{results.get('total_time', 0):.1f}s")
            
            if results.get('chapters'):
                with st.expander("📖 Scraped Chapters"):
                    for chapter in results['chapters'][:5]:
                        st.text(f"Ch.{chapter['number']}: {chapter['title'][:30]}...")
                    if len(results['chapters']) > 5:
                        st.caption(f"...and {len(results['chapters']) - 5} more")
        else:
            st.error("❌ Last scraping failed or was stopped.")
            if results.get('errors'):
                for error in results['errors'][:3]:
                    st.error(error)
    
    # Help and tips
    with st.expander("💡 Scraping Tips"):
        st.markdown("""
        **Best Practices:**
        - Start with a small number (10-20 chapters) to test
        - Use 2+ second delays to be respectful to servers
        - dxmwx.org is fully supported and tested
        - Scraped files will be saved as individual .txt files
        - Chapter numbers are automatically extracted from titles
        
        **Next Steps:**
        1. After scraping, use **Data Review** page to align with English chapters
        2. Create alignment_map.json for the translation pipeline
        3. Proceed to Translation Lab for custom translations
        """)

# Footer
st.markdown("---")
st.markdown("💡 **Tip**: Use the sidebar navigation to switch between workbench tools. Each page offers specialized functionality for different stages of the translation model development pipeline.")