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
sys.path.append('..')
from utils import *

# Page configuration
st.set_page_config(
    page_title="Fine-tuning Workbench", 
    page_icon="🤖", 
    layout="wide"
)

st.title("🤖 Fine-tuning Workbench")
st.caption("**MLOps Interface for Custom Translation Models** | Train, monitor, and evaluate your models")

# Check for dependencies
if not GOOGLE_AI_AVAILABLE:
    st.error("❌ **Google AI SDK not available**")
    st.info("Install with: `pip install google-generativeai`")
    st.stop()

# --- Sidebar: Configuration ---
st.sidebar.header("🎛️ Model Configuration")

# API Key
api_key = st.sidebar.text_input("🔑 Gemini API Key:", type="password", help="Required for fine-tuning")

if not api_key:
    st.sidebar.warning("🔑 API key required for fine-tuning operations")

# Model Selection
st.sidebar.subheader("📊 Base Model")
base_models = [
    "models/gemini-1.5-flash-001",
    "models/gemini-1.5-flash-002", 
    "models/gemini-1.5-pro-001"
]
selected_base_model = st.sidebar.selectbox("Base Model:", base_models)

# Hyperparameters
st.sidebar.subheader("⚙️ Hyperparameters")
epoch_count = st.sidebar.slider("Epochs", min_value=1, max_value=20, value=3, help="Number of training epochs")
batch_size = st.sidebar.selectbox("Batch Size", [1, 2, 4, 8, 16], index=2, help="Training batch size")
learning_rate = st.sidebar.select_slider(
    "Learning Rate", 
    options=[0.0001, 0.0005, 0.001, 0.005, 0.01], 
    value=0.001,
    format_func=lambda x: f"{x:.4f}"
)

# Dataset Configuration
st.sidebar.subheader("📚 Dataset Settings")
max_training_examples = st.sidebar.number_input(
    "Max Training Examples", 
    min_value=10, 
    max_value=1000, 
    value=100, 
    help="Limit dataset size for faster training"
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

# Load alignment map
alignment_map = load_alignment_map()

if not alignment_map:
    st.error("❌ Could not load alignment map. Please ensure it exists.")
    st.stop()

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
                    limit=max_training_examples,
                    min_similarity=0.5,
                    max_chars=30000
                )
                
                if training_examples:
                    st.session_state.training_examples = training_examples
                    
                    # Create analysis DataFrame
                    analysis_data = []
                    for example in training_examples:
                        analysis_data.append({
                            "Chapter": example['chapter_number'],
                            "Raw_Words": example['raw_stats']['word_count'],
                            "Raw_Chars": example['raw_stats']['char_count'],
                            "English_Words": example['english_stats']['word_count'],
                            "English_Chars": example['english_stats']['char_count'],
                            "Eng_Raw_Ratio": round(example['english_stats']['word_count'] / example['raw_stats']['word_count'], 2) if example['raw_stats']['word_count'] > 0 else 0
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
            st.metric("⚖️ Avg Length Ratio", f"{df['Eng_Raw_Ratio'].mean():.2f}")
            
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
            
            # Quality indicators
            good_ratio = len(df[(df['Eng_Raw_Ratio'] >= 1.5) & (df['Eng_Raw_Ratio'] <= 3.0)])
            quality_pct = (good_ratio / len(df)) * 100
            
            if quality_pct >= 80:
                st.success(f"✅ Quality: {quality_pct:.1f}% good ratios")
            elif quality_pct >= 60:
                st.warning(f"⚠️ Quality: {quality_pct:.1f}% good ratios")
            else:
                st.error(f"❌ Quality: {quality_pct:.1f}% good ratios")
        else:
            st.info("👆 Click 'Analyze Dataset Quality' to see summary")
    
    # Dataset visualization
    if hasattr(st.session_state, 'dataset_df'):
        st.subheader("📊 Dataset Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Length distribution
            fig_length = px.histogram(
                st.session_state.dataset_df, 
                x="English_Words", 
                title="Distribution of Chapter Lengths (English Words)",
                nbins=20
            )
            fig_length.update_layout(height=400)
            st.plotly_chart(fig_length, use_container_width=True)
        
        with col2:
            # Ratio distribution
            fig_ratio = px.histogram(
                st.session_state.dataset_df, 
                x="Eng_Raw_Ratio", 
                title="Distribution of English/Raw Word Ratios",
                nbins=20
            )
            fig_ratio.add_vline(x=1.5, line_dash="dash", line_color="green", annotation_text="Min Good")
            fig_ratio.add_vline(x=3.0, line_dash="dash", line_color="green", annotation_text="Max Good")
            fig_ratio.update_layout(height=400)
            st.plotly_chart(fig_ratio, use_container_width=True)
        
        # Show detailed table
        with st.expander("🔍 Detailed Dataset View"):
            st.dataframe(st.session_state.dataset_df, use_container_width=True)
        
        # Show chunking analysis if available
        if hasattr(st.session_state, 'chunking_stats'):
            with st.expander("✂️ Chunking Analysis"):
                stats = st.session_state.chunking_stats
                
                col1, col2 = st.columns(2)
                with col1:
                    # Chunk size distribution
                    import numpy as np
                    chunk_sizes = stats['chunk_sizes']
                    
                    fig_chunks = px.histogram(
                        x=chunk_sizes,
                        title="Distribution of Chunk Sizes (Characters)",
                        nbins=25,
                        labels={'x': 'Chunk Size (chars)', 'y': 'Count'}
                    )
                    fig_chunks.add_vline(x=4500, line_dash="dash", line_color="orange", annotation_text="Target Limit")
                    fig_chunks.add_vline(x=5000, line_dash="dash", line_color="red", annotation_text="Gemini Limit")
                    fig_chunks.update_layout(height=400)
                    st.plotly_chart(fig_chunks, use_container_width=True)
                
                with col2:
                    # Summary stats table
                    stats_df = pd.DataFrame([
                        {"Metric": "Total Training Examples", "Value": stats['total_examples']},
                        {"Metric": "Chapters with Multiple Chunks", "Value": stats['chunked_chapters']},
                        {"Metric": "Single-chunk Chapters", "Value": stats['single_chunks']},
                        {"Metric": "Average Chunk Size", "Value": f"{stats['avg_chunk_size']:.0f} chars"},
                        {"Metric": "Largest Chunk", "Value": f"{stats['max_chunk_size']} chars"},
                        {"Metric": "Chunks Over 5k Chars", "Value": stats['over_5k_chars']}
                    ])
                    st.table(stats_df)

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
                    operation, error = start_finetuning_job(
                        api_key=api_key,
                        training_data=st.session_state.train_data,
                        base_model=selected_base_model,
                        epoch_count=epoch_count,
                        batch_size=batch_size,
                        learning_rate=learning_rate
                    )
                    
                    if operation:
                        st.success("🎉 Training job started successfully!")
                        st.info(f"Job Name: {operation.name}")
                        
                        # Save job metadata
                        job_metadata = {
                            "job_name": operation.name,
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