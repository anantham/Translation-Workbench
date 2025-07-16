"""
🤖 Fine-tuning Workbench
Complete MLOps interface for training custom translation models
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import json
import os

# Import our shared utilities  
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
from utils import GOOGLE_AI_AVAILABLE  # Explicit import for availability check

# Page configuration
st.set_page_config(
    page_title="Fine-tuning Workbench", 
    page_icon="🤖", 
    layout="wide"
)

st.title("🤖 Fine-tuning Workbench")
st.caption("**The Factory** | Export training datasets, configure hyperparameters, and train specialized models")

# Check for dependencies
platforms_available = []
if GOOGLE_AI_AVAILABLE:
    platforms_available.append("Google Gemini")
if OPENAI_AVAILABLE:
    platforms_available.append("OpenAI")

if not platforms_available:
    st.error("❌ **No fine-tuning platforms available**")
    st.info("Install SDKs: `pip install google-generativeai openai`")
    st.stop()

# Load alignment map with unified system
try:
    from utils import list_alignment_maps, load_alignment_map_by_slug, parse_chapter_ranges
    
    # Get available alignment maps
    available_maps = list_alignment_maps()
    
    if not available_maps:
        st.error("❌ No alignment maps found. Please build an alignment map in the **📖 Data Review & Alignment** page first.")
        st.stop()
    
    # Sidebar: Alignment Map Selection
    st.sidebar.header("📁 Alignment Map Selection")
    selected_slug = st.sidebar.selectbox(
        "Choose alignment map:",
        options=sorted(available_maps.keys()),
        help="Select which novel's alignment map to use for fine-tuning"
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
    
    max_available_chapters = get_max_available_chapters(alignment_map) if alignment_map else 0
    
except Exception as e:
    st.error(f"❌ Error loading alignment map: {str(e)}")
    st.stop()

# --- Sidebar: Configuration ---
st.sidebar.header("🎛️ Platform & Model Configuration")

# Platform Selection
st.sidebar.subheader("🌐 Fine-tuning Platform")
selected_platform = st.sidebar.selectbox(
    "Platform:", 
    platforms_available,
    help="Choose between Google Gemini and OpenAI fine-tuning"
)

# Platform-specific API Key and Models
if selected_platform == "Google Gemini":
    api_key = st.sidebar.text_input("🔑 Gemini API Key:", type="password", help="Required for Google fine-tuning")
    
    if not api_key:
        st.sidebar.warning("🔑 Gemini API key required")
    
    # Google models
    st.sidebar.subheader("📊 Base Model")
    base_models = [
        "models/gemini-1.5-flash-001",
        "models/gemini-1.5-flash-002", 
        "models/gemini-1.5-pro-001"
    ]
    selected_base_model = st.sidebar.selectbox("Base Model:", base_models)

elif selected_platform == "OpenAI":
    api_key = st.sidebar.text_input("🔑 OpenAI API Key:", type="password", help="Required for OpenAI fine-tuning")
    
    if not api_key:
        st.sidebar.warning("🔑 OpenAI API key required")
    
    # OpenAI models
    st.sidebar.subheader("📊 Base Model")
    base_models = [
        "gpt-4o-mini",
        "gpt-4.1-nano-2025-04-14",
        "gpt-3.5-turbo"
    ]
    selected_base_model = st.sidebar.selectbox("Base Model:", base_models)

# Platform-specific Hyperparameters
st.sidebar.subheader("⚙️ Hyperparameters")

if selected_platform == "Google Gemini":
    # Gemini-specific hyperparameters
    epoch_count = st.sidebar.slider("Epochs", min_value=1, max_value=20, value=3, help="Number of training epochs")
    batch_size = st.sidebar.selectbox("Batch Size", [1, 2, 4, 8, 16], index=2, help="Training batch size")
    learning_rate = st.sidebar.select_slider(
        "Learning Rate", 
        options=[0.0001, 0.0005, 0.001, 0.005, 0.01], 
        value=0.001,
        format_func=lambda x: f"{x:.4f}"
    )

elif selected_platform == "OpenAI":
    # OpenAI-specific hyperparameters (with auto options)
    n_epochs_options = ["auto", 1, 2, 3, 4, 5, 10, 20]
    n_epochs = st.sidebar.selectbox("Epochs (n_epochs)", n_epochs_options, index=0, help="Number of training epochs ('auto' recommended)")
    
    batch_size_options = ["auto", 1, 2, 4, 8, 16]
    batch_size = st.sidebar.selectbox("Batch Size", batch_size_options, index=0, help="Training batch size ('auto' recommended)")
    
    learning_rate_options = ["auto", 0.0001, 0.0005, 0.001, 0.005, 0.01]
    learning_rate_multiplier = st.sidebar.selectbox(
        "Learning Rate Multiplier", 
        learning_rate_options, 
        index=0,
        help="Learning rate multiplier ('auto' recommended)"
    )

# Dataset Configuration
st.sidebar.subheader("📚 Dataset Settings")
max_training_examples = st.sidebar.number_input(
    "Max Training Examples", 
    min_value=10, 
    max_value=max_available_chapters if max_available_chapters > 0 else 5000,
    value=min(max_available_chapters, 773) if max_available_chapters > 0 else 500,
    help=f"Maximum chapters to use for training (Available: {max_available_chapters})"
)

train_split = st.sidebar.slider(
    "Train/Val Split", 
    min_value=0.5, 
    max_value=0.95, 
    value=0.8,
    format="%.0f%%",
    help="Percentage of data used for training (rest for validation)"
)

# --- Main Content ---

# Alignment map is guaranteed to be loaded by the unified system above

# Create tabs for different workbench sections
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset Preparation", 
    "🚀 Training Control", 
    "📈 Training Monitoring", 
    "🏆 Model Management"
])

# --- Tab 1: Dataset Preparation ---
with tab1:
    st.header("📊 Dataset Preparation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Dataset Analysis")
        
        if st.button("🔄 Analyze Dataset Quality", type="primary"):
            with st.spinner("Analyzing dataset..."):
                # Load and analyze training examples
                training_examples = load_dataset_for_tuning(
                    alignment_map, 
                    limit=max_training_examples
                )
                
                if training_examples:
                    st.session_state.training_examples = training_examples
                    
                    # Create analysis DataFrame
                    analysis_data = []
                    for example in training_examples:
                        bert_score = example.get('bert_similarity')
                        analysis_data.append({
                            "Chapter": example['chapter_number'],
                            "Raw_Words": example['raw_stats']['word_count'],
                            "Raw_Chars": example['raw_stats']['char_count'],
                            "English_Words": example['english_stats']['word_count'],
                            "English_Chars": example['english_stats']['char_count'],
                            "BERT_Similarity": round(bert_score, 4) if bert_score is not None else "N/A"
                        })
                    
                    df = pd.DataFrame(analysis_data)
                    st.session_state.dataset_df = df
                    
                    st.success(f"✅ Analyzed {len(training_examples)} training examples")
                else:
                    st.error("❌ No valid training examples found")
    
    with col2:
        st.subheader("📋 Dataset Summary")
        
        if hasattr(st.session_state, 'dataset_df'):
            df = st.session_state.dataset_df
            
            # Summary metrics
            st.metric("📚 Total Chapters", len(df))
            st.metric("📝 Avg Raw Words", f"{df['Raw_Words'].mean():.0f}")
            st.metric("📖 Avg English Words", f"{df['English_Words'].mean():.0f}")
            
            # Show BERT similarity metrics if available
            if 'BERT_Similarity' in df.columns:
                bert_scores = df[df['BERT_Similarity'] != "N/A"]['BERT_Similarity']
                if len(bert_scores) > 0:
                    avg_bert = bert_scores.astype(float).mean()
                    st.metric("🧠 Avg BERT Similarity", f"{avg_bert:.3f}")
                else:
                    st.metric("🧠 BERT Similarity", "Not Available")
            else:
                st.metric("🧠 BERT Similarity", "Not Available")
            
            # Show chunking impact if available
            if hasattr(st.session_state, 'chunking_stats'):
                stats = st.session_state.chunking_stats
                st.divider()
                st.caption("**Post-Chunking Statistics:**")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.text(f"Training Examples: {stats['total_examples']}")
                    st.text(f"Chunked Chapters: {stats['chunked_chapters']}")
                with col_b:
                    st.text(f"Max Chunk Size: {stats['max_chunk_size']} chars")
                    if stats['over_5k_chars'] > 0:
                        st.text(f"⚠️ {stats['over_5k_chars']} chunks over 5k chars")
                    else:
                        st.text("✅ All chunks under 5k chars")
            
            # Quality indicators based on BERT similarity
            if 'BERT_Similarity' in df.columns:
                bert_scores = df[df['BERT_Similarity'] != "N/A"]['BERT_Similarity']
                if len(bert_scores) > 0:
                    bert_numeric = bert_scores.astype(float)
                    high_quality = len(bert_numeric[bert_numeric >= 0.8])
                    quality_pct = (high_quality / len(bert_numeric)) * 100
                    
                    if quality_pct >= 80:
                        st.success(f"✅ Quality: {quality_pct:.1f}% high BERT similarity (≥0.8)")
                    elif quality_pct >= 60:
                        st.warning(f"⚠️ Quality: {quality_pct:.1f}% high BERT similarity (≥0.8)")
                    else:
                        st.error(f"❌ Quality: {quality_pct:.1f}% high BERT similarity (≥0.8)")
                else:
                    st.info("🧠 Run build_and_report.py to get BERT similarity scores")
            else:
                st.info("🧠 Run build_and_report.py to get BERT similarity scores")
        else:
            st.info("👆 Click 'Analyze Dataset Quality' to see summary")
    
    # Training-focused summary (removed dataset visualizations - moved to Experimentation Lab)
    if hasattr(st.session_state, 'dataset_df'):
        st.info("✅ **Training dataset ready.** Use tabs above to configure training or export JSONL files.")
        
        # JSONL Export Section
        st.subheader("📤 JSONL Export for Fine-tuning")
        st.caption("Export training data in OpenAI/Gemini fine-tuning format")
        
        if hasattr(st.session_state, 'dataset_df'):
            df = st.session_state.dataset_df
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Chapter range selection
                st.write("📋 **Chapter Selection**")
                
                min_chapter = int(df['Chapter'].min())
                max_chapter = int(df['Chapter'].max())
                
                export_mode = st.radio(
                    "Export Mode:",
                    ["All Chapters", "Chapter Range", "Custom Selection"],
                    horizontal=True
                )
                
                selected_chapters = []
                
                if export_mode == "All Chapters":
                    selected_chapters = df['Chapter'].tolist()
                    st.info(f"📊 Selected: All {len(selected_chapters)} chapters")
                
                elif export_mode == "Chapter Range":
                    col_a, col_b = st.columns(2)
                    with col_a:
                        start_chapter = st.number_input("Start Chapter", min_value=min_chapter, max_value=max_chapter, value=min_chapter)
                    with col_b:
                        end_chapter = st.number_input("End Chapter", min_value=start_chapter, max_value=max_chapter, value=max_chapter)
                    
                    selected_chapters = df[(df['Chapter'] >= start_chapter) & (df['Chapter'] <= end_chapter)]['Chapter'].tolist()
                    st.info(f"📊 Selected: {len(selected_chapters)} chapters (Ch.{start_chapter}-{end_chapter})")
                
                elif export_mode == "Custom Selection":
                    # Multi-select for specific chapters
                    available_chapters = df['Chapter'].tolist()
                    selected_chapters = st.multiselect(
                        "Select Chapters:",
                        available_chapters,
                        default=available_chapters[:10],  # Default to first 10
                        help="Choose specific chapters for training"
                    )
                    
                    if selected_chapters:
                        st.info(f"📊 Selected: {len(selected_chapters)} chapters")
                    else:
                        st.warning("⚠️ No chapters selected")
                
                # Export format options
                st.write("🎛️ **Export Options**")
                
                export_format = st.selectbox(
                    "JSONL Format:",
                    ["OpenAI Fine-tuning", "Gemini Fine-tuning", "Custom Messages"],
                    help="Choose the format for your fine-tuning platform"
                )
                
                train_val_split = st.slider(
                    "Training/Validation Split:",
                    min_value=0.5,
                    max_value=0.95,
                    value=0.8,
                    step=0.05,
                    help="Percentage of data for training (rest for validation)"
                )
                
                include_system_prompt = st.checkbox(
                    "Include System Prompt",
                    value=True,
                    help="Add system instruction for translation task"
                )
                
                if include_system_prompt:
                    system_prompt = st.text_area(
                        "System Prompt:",
                        "You are a professional translator specializing in Chinese to English translation of web novels. Provide accurate, fluent translations while preserving the original meaning and style.",
                        height=100
                    )
            
            with col2:
                st.write("📊 **Export Preview**")
                
                if selected_chapters:
                    num_selected = len(selected_chapters)
                    num_train = int(num_selected * train_val_split)
                    num_val = num_selected - num_train
                    
                    st.metric("📚 Selected Chapters", num_selected)
                    st.metric("🏋️ Training Examples", num_train)
                    st.metric("🔍 Validation Examples", num_val)
                    
                    # Estimate file size
                    avg_chars_per_chapter = df[df['Chapter'].isin(selected_chapters)]['Raw_Chars'].mean() + df[df['Chapter'].isin(selected_chapters)]['English_Chars'].mean()
                    estimated_size_mb = (num_selected * avg_chars_per_chapter * 2) / (1024 * 1024)  # Rough estimate
                    
                    st.metric("📁 Est. File Size", f"{estimated_size_mb:.1f} MB")
                    
                    if num_selected > 1000:
                        st.warning("⚠️ Very large dataset - may need chunking for some platforms")
                    elif num_selected > 500:
                        st.info("📊 Large dataset - excellent for training")
                    elif num_selected < 50:
                        st.warning("⚠️ Small dataset - consider adding more chapters")
                    else:
                        st.success("✅ Good dataset size")
                else:
                    st.info("👆 Select chapters to see preview")
            
            # Export button and functionality
            if selected_chapters:
                if st.button("📤 Export JSONL Files", type="primary", use_container_width=True):
                    with st.spinner("Creating JSONL training files..."):
                        # Load the actual training examples
                        training_examples = st.session_state.training_examples
                        
                        # Filter to selected chapters
                        filtered_examples = [ex for ex in training_examples if ex['chapter_number'] in selected_chapters]
                        
                        if filtered_examples:
                            # Create JSONL content
                            train_jsonl, val_jsonl, export_stats = create_translation_jsonl(
                                filtered_examples,
                                train_split=train_val_split,
                                format_type=export_format,
                                system_prompt=system_prompt if include_system_prompt else None
                            )
                            
                            # Generate timestamp for filenames
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            
                            # Create download buttons
                            col_a, col_b = st.columns(2)
                            
                            with col_a:
                                st.download_button(
                                    label="📥 Download Training JSONL",
                                    data=train_jsonl,
                                    file_name=f"training_data_{timestamp}.jsonl",
                                    mime="application/jsonl",
                                    use_container_width=True
                                )
                            
                            with col_b:
                                st.download_button(
                                    label="📥 Download Validation JSONL",
                                    data=val_jsonl,
                                    file_name=f"validation_data_{timestamp}.jsonl",
                                    mime="application/jsonl",
                                    use_container_width=True
                                )
                            
                            # Show export summary
                            st.success(f"✅ **Export Complete!**")
                            
                            col_stats1, col_stats2, col_stats3 = st.columns(3)
                            with col_stats1:
                                st.metric("📚 Training Examples", export_stats['train_count'])
                            with col_stats2:
                                st.metric("🔍 Validation Examples", export_stats['val_count'])
                            with col_stats3:
                                st.metric("📊 Total Examples", export_stats['total_count'])
                            
                            # Show sample data
                            with st.expander("👀 Sample Training Data"):
                                import json
                                sample_lines = train_jsonl.split('\n')[:3]
                                for i, line in enumerate(sample_lines):
                                    if line.strip():
                                        sample_data = json.loads(line)
                                        st.code(json.dumps(sample_data, indent=2), language="json")
                                        if i < len(sample_lines) - 1:
                                            st.divider()
                        
                        else:
                            st.error("❌ No training examples found for selected chapters")
            else:
                st.info("📋 Select chapters above to enable export")
        
        else:
            st.info("👆 Click 'Analyze Dataset Quality' first to enable JSONL export")

# --- Tab 2: Training Control ---
with tab2:
    st.header("🚀 Training Control")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🎯 Training Configuration")
        
        # Show current configuration
        config_df = pd.DataFrame([
            {"Parameter": "Base Model", "Value": str(selected_base_model)},
            {"Parameter": "Epochs", "Value": str(epoch_count)},
            {"Parameter": "Batch Size", "Value": str(batch_size)},
            {"Parameter": "Learning Rate", "Value": f"{learning_rate:.4f}"},
            {"Parameter": "Max Examples", "Value": str(max_training_examples)},
            {"Parameter": "Train/Val Split", "Value": f"{train_split:.0%}"}
        ])
        
        st.table(config_df)
        
        # Training data preparation
        if hasattr(st.session_state, 'training_examples'):
            st.success(f"✅ Training data ready: {len(st.session_state.training_examples)} chapters")
            
            # Prepare API format
            if st.button("📋 Prepare Training Data with Chunking"):
                with st.spinner("Preparing data for API with automatic chunking..."):
                    train_data, val_data = prepare_training_data_for_api(
                        st.session_state.training_examples, 
                        train_split=train_split,
                        max_output_chars=4500  # Stay under Gemini's 5k limit
                    )
                    
                    st.session_state.train_data = train_data
                    st.session_state.val_data = val_data
                    
                    # Get chunking statistics
                    chunking_stats = get_chunking_statistics(train_data + val_data)
                    st.session_state.chunking_stats = chunking_stats
                    
                    st.success(f"✅ Prepared {len(train_data)} training examples, {len(val_data)} validation examples")
                    
                    # Display chunking summary
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("📚 Total Chunks", chunking_stats['total_examples'])
                    with col_b:
                        st.metric("📄 Avg Chunk Size", f"{chunking_stats['avg_chunk_size']:.0f} chars")
                    with col_c:
                        st.metric("⚠️ Over 5k Chars", chunking_stats['over_5k_chars'])
                    with col_d:
                        st.metric("✂️ Chunked Chapters", chunking_stats['chunked_chapters'])
        else:
            st.info("👆 Prepare dataset in the 'Dataset Preparation' tab first")
    
    # This is a placeholder for the new logic that will be added in a subsequent step.
# For now, we are removing the old, broken conflict handler.
# The new logic will check the return value of the scraper and set the session state.
with col2:
    st.subheader("🔥 Launch Training")
    
    # Safety checks
    can_train = all([
        api_key,
        hasattr(st.session_state, 'train_data'),
        len(getattr(st.session_state, 'train_data', [])) > 0
    ])
    
    if can_train:
        st.success("✅ Ready to train!")
        
        # Training confirmation
        train_confirmed = st.checkbox(
            "I confirm this training configuration",
            help="This will start a fine-tuning job on Google AI"
        )
        
        if train_confirmed and st.button("🚀 **Start Training**", type="primary"):
            with st.spinner("Starting fine-tuning job..."):
                if selected_platform == "Google Gemini":
                    operation, error = start_finetuning_job(
                        api_key=api_key,
                        training_data=st.session_state.train_data,
                        base_model=selected_base_model,
                        epoch_count=epoch_count,
                        batch_size=batch_size,
                        learning_rate=learning_rate
                    )
                elif selected_platform == "OpenAI":
                    # First, create JSONL content from training data
                    train_jsonl, val_jsonl, export_stats = create_translation_jsonl(
                        st.session_state.training_examples,
                        train_split=0.8,
                        format_type="OpenAI Fine-tuning",
                        system_prompt="You are a professional translator specializing in Chinese to English translation of web novels."
                    )
                    
                    # Upload training file to OpenAI
                    file_response, upload_error = upload_training_file_openai(api_key, train_jsonl)
                    
                    if upload_error:
                        st.error(f"❌ File upload failed: {upload_error}")
                        operation = None
                        error = upload_error
                    else:
                        # Start fine-tuning job
                        operation, error = start_openai_finetuning_job(
                            api_key=api_key,
                            training_file_id=file_response.id,
                            model=selected_base_model,
                            n_epochs=n_epochs,
                            batch_size=batch_size,
                            learning_rate_multiplier=learning_rate_multiplier
                        )
                
                if operation:
                    st.success("🎉 Training job started successfully!")
                    
                    # Platform-specific job info display
                    if selected_platform == "Google Gemini":
                        st.info(f"Job Name: {operation.name}")
                        job_name = operation.name
                    elif selected_platform == "OpenAI":
                        st.info(f"Job ID: {operation.id}")
                        st.info(f"Model: {operation.model}")
                        job_name = operation.id
                    
                    # Platform-specific job metadata
                    if selected_platform == "Google Gemini":
                        job_metadata = {
                            "platform": "Google Gemini",
                            "job_name": job_name,
                            "base_model": selected_base_model,
                            "hyperparameters": {
                                "epoch_count": epoch_count,
                                "batch_size": batch_size,
                                "learning_rate": learning_rate
                            },
                            "dataset_info": {
                                "train_examples": len(st.session_state.train_data),
                                "val_examples": len(st.session_state.val_data),
                                "train_split": train_split
                            }
                        }
                    elif selected_platform == "OpenAI":
                        job_metadata = {
                            "platform": "OpenAI",
                            "job_name": job_name,
                            "base_model": selected_base_model,
                            "hyperparameters": {
                                "n_epochs": n_epochs,
                                "batch_size": batch_size,
                                "learning_rate_multiplier": learning_rate_multiplier
                            },
                            "dataset_info": {
                                "training_file_id": file_response.id if selected_platform == "OpenAI" else None,
                                "train_examples": export_stats['train_count'] if selected_platform == "OpenAI" else len(st.session_state.train_data),
                                "val_examples": export_stats['val_count'] if selected_platform == "OpenAI" else len(st.session_state.val_data),
                                "train_split": 0.8 if selected_platform == "OpenAI" else train_split
                            }
                        }
                    
                    metadata_file = save_model_metadata(
                        job_metadata, 
                        job_metadata["hyperparameters"], 
                        job_metadata["dataset_info"]
                    )
                    
                    if metadata_file:
                        st.success(f"📁 Metadata saved: {os.path.basename(metadata_file)}")
                    
                    # Add to session state for monitoring
                    if 'training_jobs' not in st.session_state:
                        st.session_state.training_jobs = []
                    
                    st.session_state.training_jobs.append({
                        "name": operation.name,
                        "started_at": datetime.now(),
                        "status": "CREATING",
                        "metadata": job_metadata
                    })
                    
                    st.rerun()
                else:
                    st.error(f"❌ Training failed: {error}")
    else:
        st.warning("⚠️ Missing requirements:")
        if not api_key:
            st.text("• API key")
        if not hasattr(st.session_state, 'train_data'):
            st.text("• Training data")

# --- Tab 3: Training Monitoring ---
with tab3:
    st.header("📈 Training Monitoring")
    
    # Refresh button
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🔄 Refresh Status"):
            st.rerun()
    
    # Jobs table
    if api_key:
        with st.spinner("Loading training jobs..."):
            jobs, error = list_tuning_jobs(api_key)
            
            if jobs and not error:
                job_data = []
                for job in jobs:
                    # Extract status and progress info
                    status = getattr(job, 'state', 'UNKNOWN')
                    name = getattr(job, 'name', 'Unknown')
                    
                    # Try to get tuning task info
                    tuning_task = getattr(job, 'tuning_task', None)
                    progress = "N/A"
                    if tuning_task:
                        snapshots = getattr(tuning_task, 'snapshots', [])
                        if snapshots:
                            latest_snapshot = snapshots[-1]
                            epoch = getattr(latest_snapshot, 'epoch', 'N/A')
                            loss = getattr(latest_snapshot, 'mean_loss', 'N/A')
                            progress = f"Epoch {epoch}, Loss: {loss:.4f}" if loss != 'N/A' else f"Epoch {epoch}"
                    
                    job_data.append({
                        "Job Name": name.split('/')[-1],  # Get just the model name part
                        "Status": status,
                        "Progress": progress,
                        "Full Name": name
                    })
                
                if job_data:
                    st.subheader("🏃 Active Training Jobs")
                    
                    # Create interactive table
                    jobs_df = pd.DataFrame(job_data)
                    
                    # Status styling
                    def style_status(val):
                        if val == "ACTIVE":
                            return "background-color: #90EE90"
                        elif val == "CREATING":
                            return "background-color: #FFE4B5"
                        elif val == "FAILED":
                            return "background-color: #FFB6C1"
                        elif val == "COMPLETED":
                            return "background-color: #E0E0E0"
                        return ""
                    
                    styled_df = jobs_df.drop('Full Name', axis=1).style.applymap(
                        style_status, subset=['Status']
                    )
                    
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Job selection for detailed view
                    st.subheader("📊 Job Details")
                    selected_job_name = st.selectbox(
                        "Select job for details:",
                        options=[row["Full Name"] for row in job_data],
                        format_func=lambda x: x.split('/')[-1]
                    )
                    
                    if selected_job_name:
                        # Get detailed job info
                        job_info, job_error = get_tuning_job_status(selected_job_name, api_key)
                        
                        if job_info and not job_error:
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                # Training progress chart
                                tuning_task = getattr(job_info, 'tuning_task', None)
                                if tuning_task:
                                    snapshots = getattr(tuning_task, 'snapshots', [])
                                    if snapshots and len(snapshots) > 1:
                                        # Create loss curve
                                        epochs = []
                                        losses = []
                                        
                                        for snapshot in snapshots:
                                            epoch = getattr(snapshot, 'epoch', None)
                                            loss = getattr(snapshot, 'mean_loss', None)
                                            if epoch is not None and loss is not None:
                                                epochs.append(epoch)
                                                losses.append(loss)
                                        
                                        if epochs and losses:
                                            fig = px.line(
                                                x=epochs, 
                                                y=losses,
                                                title="🔥 Training Loss Curve",
                                                labels={'x': 'Epoch', 'y': 'Mean Loss'}
                                            )
                                            fig.update_traces(line_color='#ff6b6b', line_width=3)
                                            fig.update_layout(height=400)
                                            st.plotly_chart(fig, use_container_width=True)
                                        else:
                                            st.info("📈 Waiting for training metrics...")
                                    else:
                                        st.info("📈 Training just started - metrics will appear soon")
                                else:
                                    st.info("📈 No training metrics available yet")
                            
                            with col2:
                                # Job metrics
                                st.metric("📊 Status", getattr(job_info, 'state', 'UNKNOWN'))
                                
                                if tuning_task:
                                    hyperparams = getattr(tuning_task, 'hyperparameters', {})
                                    if hyperparams:
                                        st.metric("🔄 Epochs", getattr(hyperparams, 'epoch_count', 'N/A'))
                                        st.metric("📦 Batch Size", getattr(hyperparams, 'batch_size', 'N/A'))
                                        st.metric("📈 Learning Rate", f"{getattr(hyperparams, 'learning_rate', 'N/A'):.4f}" if hasattr(hyperparams, 'learning_rate') else 'N/A')
                                    
                                    snapshots = getattr(tuning_task, 'snapshots', [])
                                    if snapshots:
                                        latest = snapshots[-1]
                                        current_epoch = getattr(latest, 'epoch', 'N/A')
                                        current_loss = getattr(latest, 'mean_loss', 'N/A')
                                        
                                        st.metric("📍 Current Epoch", current_epoch)
                                        if current_loss != 'N/A':
                                            st.metric("📉 Current Loss", f"{current_loss:.4f}")
                        else:
                            st.error(f"❌ Could not get job details: {job_error}")
                else:
                    st.info("📋 No training jobs found")
            else:
                st.error(f"❌ Could not load jobs: {error}")
    else:
        st.warning("🔑 API key required to monitor training jobs")

# --- Tab 4: Model Management ---
with tab4:
    st.header("🏆 Model Management")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📚 Model Library")
        
        # Load saved model metadata
        model_metadata = load_model_metadata()
        
        if model_metadata:
            for i, metadata in enumerate(model_metadata):
                with st.expander(f"🤖 Model {metadata.get('timestamp', 'Unknown')} - {metadata.get('job_info', {}).get('job_name', 'Unknown').split('/')[-1]}"):
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.write("**📊 Training Info:**")
                        job_info = metadata.get('job_info', {})
                        st.text(f"Base Model: {job_info.get('base_model', 'Unknown')}")
                        st.text(f"Job Name: {job_info.get('job_name', 'Unknown')}")
                        
                        st.write("**⚙️ Hyperparameters:**")
                        hyperparams = metadata.get('hyperparameters', {})
                        for param, value in hyperparams.items():
                            st.text(f"{param}: {value}")
                    
                    with col_b:
                        st.write("**📚 Dataset Info:**")
                        dataset_info = metadata.get('dataset_info', {})
                        for key, value in dataset_info.items():
                            st.text(f"{key}: {value}")
                        
                        st.write("**🕐 Created:**")
                        st.text(metadata.get('created_at', 'Unknown'))
        else:
            st.info("📭 No saved models found. Train a model to see it here!")
    
    with col2:
        st.subheader("🔧 Model Actions")
        
        if api_key:
            # Quick status refresh
            if st.button("🔄 Refresh Model List"):
                jobs, error = list_tuning_jobs(api_key)
                if jobs and not error:
                    st.success(f"✅ Found {len(jobs)} models")
                    
                    # Show summary
                    status_counts = {}
                    for job in jobs:
                        status = getattr(job, 'state', 'UNKNOWN')
                        status_counts[status] = status_counts.get(status, 0) + 1
                    
                    for status, count in status_counts.items():
                        st.metric(f"{status}", count)
                else:
                    st.error("❌ Could not refresh model list")
        
        # Model export options
        st.subheader("📤 Export Options")
        if st.button("💾 Export Model Metadata", help="Download all model metadata as JSON"):
            model_metadata = load_model_metadata()
            if model_metadata:
                export_data = json.dumps(model_metadata, indent=2, default=str)
                st.download_button(
                    label="📥 Download Metadata JSON",
                    data=export_data,
                    file_name=f"model_metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.info("📭 No metadata to export")

# --- Footer ---
st.divider()
st.caption("🤖 **Fine-tuning Workbench** | Part of the Translation Framework Suite")

# Auto-refresh for active training jobs
if hasattr(st.session_state, 'training_jobs') and st.session_state.training_jobs:
    active_jobs = [job for job in st.session_state.training_jobs if job['status'] in ['CREATING', 'ACTIVE']]
    if active_jobs:
        st.sidebar.info(f"🔄 {len(active_jobs)} active training job(s)")
        
        # Auto-refresh every 30 seconds when training is active
        if st.sidebar.button("🔄 Auto-refresh (30s)"):
            time.sleep(30)
            st.rerun()