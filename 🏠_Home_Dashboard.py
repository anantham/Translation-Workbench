"""
🏠 Translation Framework Workbench
Main entry point for the multi-page Streamlit application
"""

import streamlit as st
import os
from utils import load_api_config, show_config_status

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
    
    if novel_url:
        # Validate URL
        validation = validate_scraping_url(novel_url)
        
        # Show validation results
        if validation['valid']:
            st.success(f"✅ Valid URL detected: {validation['site_type']}")
        else:
            st.warning("⚠️ URL validation issues detected")
        
        # Show recommendations and warnings
        if validation['recommendations']:
            for rec in validation['recommendations']:
                st.info(rec)
        
        if validation['warnings']:
            for warn in validation['warnings']:
                st.warning(warn)
    
    # Scraping configuration
    st.subheader("⚙️ Scraping Configuration")
    
    scraping_config_col1, scraping_config_col2 = st.columns(2)
    
    with scraping_config_col1:
        max_chapters = st.number_input(
            "Max Chapters to Scrape:",
            min_value=1,
            max_value=500,
            value=50,
            help="Maximum number of chapters to scrape in this session"
        )
        
        output_directory = st.text_input(
            "Output Directory:",
            value="novel_content_scraped",
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
            index=0,
            help="Direction to follow chapter navigation links"
        )
    
    # Start scraping button
    if novel_url and validation['valid']:
        if st.button("🚀 Start Scraping", type="primary"):
            # Initialize scraping session state
            if 'scraping_active' not in st.session_state:
                st.session_state.scraping_active = False
            
            if not st.session_state.scraping_active:
                st.session_state.scraping_active = True
                st.session_state.scraping_results = None
                
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
                        status_callback=status_callback
                    )
                    
                    st.session_state.scraping_results = scraping_results
                    st.session_state.scraping_active = False
                    
                    # Show final results
                    with results_container:
                        if scraping_results['success']:
                            st.success(f"🎉 Scraping completed successfully!")
                            st.metric("Chapters Scraped", scraping_results['chapters_scraped'])
                            st.metric("Total Time", f"{scraping_results['total_time']:.1f}s")
                            
                            if scraping_results['errors']:
                                with st.expander("⚠️ Errors Encountered"):
                                    for error in scraping_results['errors']:
                                        st.error(error)
                        else:
                            st.error("❌ Scraping failed")
                            for error in scraping_results['errors']:
                                st.error(error)
                                
                except Exception as e:
                    st.session_state.scraping_active = False
                    st.error(f"💥 Scraping failed with error: {str(e)}")
            else:
                st.warning("🔄 Scraping already in progress...")

with scraping_col2:
    st.subheader("📊 Scraping Status")
    
    # Show existing scraped data
    if os.path.exists("novel_content_scraped"):
        scraped_files = [f for f in os.listdir("novel_content_scraped") if f.endswith('.txt')]
        if scraped_files:
            st.metric("📚 Existing Chapters", len(scraped_files))
            
            # Show sample filenames
            with st.expander("📋 Sample Files"):
                for filename in sorted(scraped_files)[:5]:
                    st.text(filename)
                if len(scraped_files) > 5:
                    st.caption(f"...and {len(scraped_files) - 5} more files")
        else:
            st.info("📭 No chapters scraped yet")
    else:
        st.info("📁 Output directory will be created when scraping starts")
    
    # Show scraping results if available
    if hasattr(st.session_state, 'scraping_results') and st.session_state.scraping_results:
        results = st.session_state.scraping_results
        
        st.subheader("📈 Last Scraping Results")
        
        if results['success']:
            st.success("✅ Success")
            st.metric("Chapters", results['chapters_scraped'])
            st.metric("Time", f"{results['total_time']:.1f}s")
            
            if results['chapters']:
                with st.expander("📖 Scraped Chapters"):
                    for chapter in results['chapters'][:5]:
                        st.text(f"Ch.{chapter['number']}: {chapter['title'][:30]}...")
                    if len(results['chapters']) > 5:
                        st.caption(f"...and {len(results['chapters']) - 5} more")
        else:
            st.error("❌ Last scraping failed")
            if results['errors']:
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