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
st.header("🚀 Quick Start Workflow")

workflow_col1, workflow_col2, workflow_col3, workflow_col4 = st.columns(4)

with workflow_col1:
    st.markdown("""
    **1. 📖 Data Review**
    - Review and perfect dataset alignment
    - Fix any misaligned chapters
    - Quality control checks
    """)

with workflow_col2:
    st.markdown("""
    **2. 🧪 Translation Lab**
    - Generate custom translations
    - Experiment with different styles
    - Build translation datasets
    """)

with workflow_col3:
    st.markdown("""
    **3. 🤖 Fine-tuning**
    - Train custom models
    - Monitor training progress
    - Save model metadata
    """)

with workflow_col4:
    st.markdown("""
    **4. 📈 Experimentation**
    - Compare model performance
    - Statistical evaluation
    - Quality metrics analysis
    """)

# Footer
st.markdown("---")
st.markdown("💡 **Tip**: Use the sidebar navigation to switch between workbench tools. Each page offers specialized functionality for different stages of the translation model development pipeline.")