"""
📈 Experimentation Analysis
Compare fine-tuned models vs in-context learning with base models and analyze translation outputs
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import random

# Import our shared utilities
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *

# Page configuration
st.set_page_config(
    page_title="📈 Experimentation Analysis", 
    page_icon="📈", 
    layout="wide"
)

st.title("📈 Translation Style Analytics")
st.caption("**Performance Visualization Dashboard** | Analyze translation quality trends across chapters and compare style effectiveness")

# Load alignment map early to get available chapters
alignment_map = load_alignment_map()
max_available_chapters = get_max_available_chapters(alignment_map) if alignment_map else 0

# --- Sidebar: Style Selection for Visualization ---
st.sidebar.header("📊 Graph Visualization")

# Get available translation styles
available_styles = get_available_translation_styles()

if not available_styles:
    st.sidebar.error("❌ No translation styles found")
    st.error("❌ No translation styles available for analysis. Please generate translations in the Pluralistic Translation Lab first.")
    st.stop()

# Style selection for visualization
st.sidebar.subheader("🎨 Select Styles to Visualize")
style_options = {}
for style in available_styles:
    label = f"{style['name']} ({style['chapter_count']} chapters)"
    style_options[label] = style

selected_style_labels = st.sidebar.multiselect(
    "Choose styles to compare:",
    options=list(style_options.keys()),
    default=list(style_options.keys())[:3] if len(style_options) >= 3 else list(style_options.keys()),
    help="Select one or more translation styles to visualize performance across chapters"
)

# Graph display options
st.sidebar.subheader("📈 Display Options")
show_bert_scores = st.sidebar.checkbox("Show BERT Scores", value=True)
show_human_scores = st.sidebar.checkbox("Show Human Evaluation Scores", value=True)
show_composite_trend = st.sidebar.checkbox("Show Composite Score Trend", value=False)

# --- Main Content ---

if not alignment_map:
    st.error("❌ Could not load alignment map")
    st.stop()

# Create simplified tabs focused on visualization
tab1, tab2 = st.tabs([
    "📈 Chapter Performance Graphs",
    "🏆 Style Leaderboard & Rankings"
])

# --- Tab 1: Chapter Performance Graphs ---
with tab1:
    st.header("📈 Chapter Performance Graphs")
    st.caption("Visualize translation quality metrics across chapters for selected styles")
    
    if not selected_style_labels:
        st.info("👆 Please select at least one translation style from the sidebar to visualize")
    else:
        selected_styles = [style_options[label] for label in selected_style_labels]
        
        # Collect data for all selected styles
        all_style_data = []
        
        for style in selected_styles:
            style_name = style['name']
            
            # Load BERT scores
            bert_scores = load_bert_scores(style_name) if show_bert_scores else {}
            
            # Load human scores  
            human_scores = load_human_scores(style_name) if show_human_scores else {}
            
            # Combine all available chapters
            all_chapters = set()
            if bert_scores:
                all_chapters.update([int(ch) for ch in bert_scores.keys()])
            if human_scores:
                all_chapters.update([int(ch) for ch in human_scores.keys()])
            
            if not all_chapters:
                continue
                
            # Create data for this style
            for chapter_num in sorted(all_chapters):
                chapter_str = str(chapter_num)
                
                # BERT score data
                if show_bert_scores and chapter_str in bert_scores:
                    all_style_data.append({
                        'Style': style_name,
                        'Chapter': chapter_num,
                        'Metric': 'BERT Similarity',
                        'Score': bert_scores[chapter_str],
                        'Category': 'Automated'
                    })
                
                # Human evaluation data
                if show_human_scores and chapter_str in human_scores:
                    human_data = human_scores[chapter_str]
                    
                    # Individual human dimensions
                    human_dimensions = {
                        'English Sophistication': human_data.get('english_sophistication', 0) / 100,
                        'World Building': human_data.get('world_building', 0) / 100,
                        'Emotional Impact': human_data.get('emotional_impact', 0) / 100,
                        'Dialogue Naturalness': human_data.get('dialogue_naturalness', 0) / 100
                    }
                    
                    for dimension, score in human_dimensions.items():
                        if score > 0:  # Only include non-zero scores
                            all_style_data.append({
                                'Style': style_name,
                                'Chapter': chapter_num,
                                'Metric': dimension,
                                'Score': score,
                                'Category': 'Human Evaluation'
                            })
                
                # Composite score if requested
                if show_composite_trend and chapter_str in bert_scores and chapter_str in human_scores:
                    # Calculate chapter-level composite score
                    bert_score = bert_scores[chapter_str]
                    human_data = human_scores[chapter_str]
                    
                    # Average human scores
                    human_dims = ['english_sophistication', 'world_building', 'emotional_impact', 'dialogue_naturalness']
                    valid_human_scores = [human_data.get(dim, 0) / 100 for dim in human_dims if human_data.get(dim, 0) > 0]
                    
                    if valid_human_scores:
                        avg_human = sum(valid_human_scores) / len(valid_human_scores)
                        composite = (bert_score * 0.5) + (avg_human * 0.5)
                        
                        all_style_data.append({
                            'Style': style_name,
                            'Chapter': chapter_num,
                            'Metric': 'Composite Score',
                            'Score': composite,
                            'Category': 'Combined'
                        })
        
        if not all_style_data:
            st.warning("⚠️ No evaluation data found for selected styles. Please run BERT evaluation and human assessment first.")
        else:
            # Create DataFrame for visualization
            df = pd.DataFrame(all_style_data)
            
            # Display summary statistics
            st.subheader("📊 Performance Overview")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_data_points = len(df)
                st.metric("📈 Total Data Points", total_data_points)
            
            with col2:
                unique_chapters = df['Chapter'].nunique()
                st.metric("📚 Chapters Analyzed", unique_chapters)
            
            with col3:
                avg_score = df['Score'].mean()
                st.metric("⭐ Average Score", f"{avg_score:.3f}")
            
            st.divider()
            
            # Create visualizations based on selected options
            if show_bert_scores:
                st.subheader("🧠 BERT Similarity Scores by Chapter")
                
                bert_data = df[df['Metric'] == 'BERT Similarity']
                if not bert_data.empty:
                    fig_bert = px.line(
                        bert_data,
                        x='Chapter',
                        y='Score',
                        color='Style',
                        title='BERT Similarity Across Chapters',
                        labels={'Score': 'BERT Similarity Score', 'Chapter': 'Chapter Number'},
                        markers=True
                    )
                    fig_bert.add_hline(y=0.8, line_dash="dash", line_color="green", 
                                     annotation_text="Excellent (≥0.8)")
                    fig_bert.add_hline(y=0.6, line_dash="dash", line_color="orange", 
                                     annotation_text="Good (≥0.6)")
                    fig_bert.update_layout(height=500)
                    st.plotly_chart(fig_bert, use_container_width=True)
                    
                    # Show statistics
                    with st.expander("📊 BERT Score Statistics"):
                        bert_stats = bert_data.groupby('Style')['Score'].agg(['mean', 'std', 'min', 'max']).round(3)
                        st.dataframe(bert_stats, use_container_width=True)
                else:
                    st.info("📊 No BERT scores available for visualization")
            
            if show_human_scores:
                st.subheader("👤 Human Evaluation Dimensions")
                
                human_data = df[df['Category'] == 'Human Evaluation']
                if not human_data.empty:
                    # Create subplots for each human dimension
                    human_metrics = human_data['Metric'].unique()
                    
                    for metric in human_metrics:
                        metric_data = human_data[human_data['Metric'] == metric]
                        
                        fig_human = px.line(
                            metric_data,
                            x='Chapter',
                            y='Score',
                            color='Style',
                            title=f'{metric} Scores Across Chapters',
                            labels={'Score': f'{metric} Score (0-1)', 'Chapter': 'Chapter Number'},
                            markers=True
                        )
                        fig_human.update_layout(height=400)
                        st.plotly_chart(fig_human, use_container_width=True)
                    
                    # Human evaluation heatmap
                    st.subheader("🔥 Human Evaluation Heatmap")
                    
                    # Create pivot table for heatmap
                    if len(selected_styles) > 1:
                        pivot_data = human_data.pivot_table(
                            values='Score', 
                            index='Metric', 
                            columns='Style', 
                            aggfunc='mean'
                        )
                        
                        fig_heatmap = px.imshow(
                            pivot_data,
                            title="Average Human Evaluation Scores by Style and Dimension",
                            labels={'color': 'Average Score'},
                            aspect='auto'
                        )
                        fig_heatmap.update_layout(height=400)
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                else:
                    st.info("📊 No human evaluation scores available for visualization")
            
            if show_composite_trend:
                st.subheader("🎯 Composite Score Trends")
                
                composite_data = df[df['Metric'] == 'Composite Score']
                if not composite_data.empty:
                    fig_composite = px.line(
                        composite_data,
                        x='Chapter',
                        y='Score',
                        color='Style',
                        title='Composite Scores (BERT + Human) Across Chapters',
                        labels={'Score': 'Composite Score', 'Chapter': 'Chapter Number'},
                        markers=True
                    )
                    fig_composite.update_layout(height=500)
                    st.plotly_chart(fig_composite, use_container_width=True)
                    
                    # Trend analysis
                    with st.expander("📈 Trend Analysis"):
                        trend_stats = composite_data.groupby('Style').agg({
                            'Score': ['mean', 'std', 'count'],
                            'Chapter': ['min', 'max']
                        }).round(3)
                        st.dataframe(trend_stats, use_container_width=True)
                else:
                    st.info("📊 No composite scores available. Need both BERT and human evaluations.")
            
            # Data export
            st.divider()
            st.subheader("💾 Export Visualization Data")
            
            if st.button("📥 Download Graph Data as CSV"):
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV",
                    data=csv_data,
                    file_name=f"style_performance_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# --- Tab 2: Style Leaderboard & Rankings ---
with tab2:
    st.header("🏆 Translation Style Evaluation & Leaderboard")
    st.caption("Comprehensive quality assessment of custom translation styles")
    
    # Get available translation styles
    available_styles = get_available_translation_styles()
    
    if not available_styles:
        st.info("🎨 No custom translation styles found. Generate translations in the Pluralistic Translation Lab first.")
        st.stop()
    
    # Style selection interface
    st.subheader("📚 Available Translation Styles")
    
    # Create style selection with metadata display
    style_options = {}
    for style in available_styles:
        label = f"{style['name']} ({style['chapter_count']} chapters, {style['model_name']})"
        style_options[label] = style
    
    selected_styles = st.multiselect(
        "Select styles to evaluate:",
        options=list(style_options.keys()),
        help="Choose one or more translation styles for evaluation"
    )
    
    if not selected_styles:
        st.info("👆 Please select at least one translation style to evaluate")
        st.stop()
    
    # Display selected styles info
    with st.expander("📋 Selected Styles Details"):
        for style_label in selected_styles:
            style = style_options[style_label]
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Style", style['name'])
                st.metric("Chapters", style['chapter_count'])
            
            with col2:
                st.metric("Model", style['model_name'])
                st.metric("Range", style['chapter_range'])
            
            with col3:
                # Check evaluation status
                bert_scores = load_bert_scores(style['name'])
                human_scores = load_human_scores(style['name'])
                
                st.metric("BERT Evaluated", len(bert_scores))
                st.metric("Human Evaluated", len(human_scores))
            
            with st.expander(f"System Prompt: {style['name']}"):
                st.text(style['system_prompt'][:500] + "..." if len(style['system_prompt']) > 500 else style['system_prompt'])
    
    st.divider()
    
    # BERT Evaluation Section
    st.subheader("🧠 Semantic Similarity Evaluation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("Calculate BERT similarity scores against official translations for comprehensive semantic fidelity assessment.")
        
        if st.button("🔄 Calculate BERT Scores for All Selected Styles", type="primary"):
            # Process each selected style
            for style_label in selected_styles:
                style = style_options[style_label]
                style_name = style['name']
                
                st.write(f"**Processing: {style_name}**")
                
                # Check if already calculated
                existing_scores = load_bert_scores(style_name)
                if len(existing_scores) >= style['chapter_count'] * 0.95:  # 95% complete
                    st.success(f"✅ {style_name}: BERT scores already complete ({len(existing_scores)} chapters)")
                    continue
                
                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(current, total, chapter_num):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(f"Processing chapter {chapter_num} ({current}/{total})")
                
                # Calculate BERT scores
                with st.spinner(f"Calculating BERT scores for {style_name}..."):
                    bert_scores = calculate_bert_scores_for_style(style, alignment_map, progress_callback)
                    
                    if bert_scores:
                        # Save scores
                        save_bert_scores(style_name, bert_scores)
                        st.success(f"✅ {style_name}: Calculated {len(bert_scores)} BERT scores")
                    else:
                        st.error(f"❌ {style_name}: Failed to calculate BERT scores (semantic model issue?)")
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
    
    with col2:
        st.subheader("📊 BERT Status")
        
        # Show BERT evaluation status for selected styles
        for style_label in selected_styles:
            style = style_options[style_label]
            bert_scores = load_bert_scores(style['name'])
            
            completion_rate = len(bert_scores) / style['chapter_count'] if style['chapter_count'] > 0 else 0
            
            st.metric(
                f"🧠 {style['name'][:20]}...",
                f"{len(bert_scores)}/{style['chapter_count']}",
                f"{completion_rate:.1%} complete"
            )
            
            if bert_scores:
                import numpy as np
                scores_array = list(bert_scores.values())
                st.caption(f"Mean: {np.mean(scores_array):.3f} (±{np.std(scores_array):.3f})")
    
    st.divider()
    
    # Human Evaluation Section
    st.subheader("👤 Human Quality Assessment")
    
    # Chapter selection for human evaluation
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("Evaluate translation quality across multiple dimensions by reviewing specific chapters.")
        
        # Select style for human evaluation
        if len(selected_styles) == 1:
            eval_style_label = selected_styles[0]
        else:
            eval_style_label = st.selectbox(
                "Select style for human evaluation:",
                selected_styles,
                help="Choose one style to evaluate manually"
            )
        
        eval_style = style_options[eval_style_label]
        
        # Get chapters available for this style
        style_path = eval_style['path']
        available_chapters = []
        
        for filename in os.listdir(style_path):
            if filename.endswith('-translated.txt'):
                try:
                    chapter_num = int(filename.split('-')[1])
                    available_chapters.append(chapter_num)
                except (ValueError, IndexError):
                    continue
        
        available_chapters.sort()
        
        if available_chapters:
            # Chapter selection
            selected_chapter = st.selectbox(
                "Select chapter to evaluate:",
                available_chapters,
                format_func=lambda x: f"Chapter {x}",
                help="Choose a chapter to review and score"
            )
            
            # Load and display chapter content
            if selected_chapter:
                # Load custom translation
                custom_file = os.path.join(style_path, f"Chapter-{selected_chapter:04d}-translated.txt")
                if os.path.exists(custom_file):
                    with open(custom_file, 'r', encoding='utf-8') as f:
                        custom_translation = f.read()
                    
                    # Load official translation for comparison
                    official_translation = ""
                    if str(selected_chapter) in alignment_map:
                        official_file = alignment_map[str(selected_chapter)].get('english_file')
                        if official_file:
                            official_translation = load_chapter_content(official_file)
                    
                    # Enhanced comparison interface with flexible style selection
                    st.subheader(f"📖 Chapter {selected_chapter} Comparison")
                    
                    # Get all available translation styles for this chapter
                    available_translation_styles = ["Official Translation"]
                    custom_translations = {}
                    
                    # Scan for custom translation styles
                    custom_runs_dir = os.path.join(DATA_DIR, "custom_translations")
                    if os.path.exists(custom_runs_dir):
                        for run_name in os.listdir(custom_runs_dir):
                            run_path = os.path.join(custom_runs_dir, run_name)
                            if os.path.isdir(run_path):
                                chapter_file = os.path.join(run_path, f"Chapter-{selected_chapter:04d}-translated.txt")
                                if os.path.exists(chapter_file):
                                    with open(chapter_file, 'r', encoding='utf-8') as f:
                                        custom_translations[run_name] = f.read()
                                    available_translation_styles.append(f"Custom: {run_name}")
                    
                    # Style selection interface
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        left_style = st.selectbox(
                            "📖 Left Panel:",
                            available_translation_styles,
                            index=0,
                            help="Choose translation style for left panel"
                        )
                    
                    with col2:
                        right_style = st.selectbox(
                            "📖 Right Panel:",
                            available_translation_styles,
                            index=0,  # Default to Official Translation
                            help="Choose translation style for right panel"
                        )
                    
                    # Load content for selected styles
                    def get_translation_content(style_name):
                        if style_name == "Official Translation":
                            if str(selected_chapter) in alignment_map:
                                official_file = alignment_map[str(selected_chapter)].get('english_file')
                                if official_file:
                                    return load_chapter_content(official_file)
                            return "Official translation not available for this chapter"
                        else:
                            # Extract run name from "Custom: run_name" format
                            run_name = style_name.replace("Custom: ", "")
                            return custom_translations.get(run_name, "Translation not found")
                    
                    left_content = get_translation_content(left_style)
                    right_content = get_translation_content(right_style)
                    
                    # Display content with enhanced synchronized scrolling
                    if left_content and right_content and "not available" not in left_content and "not found" not in right_content:
                        # Calculate dynamic height based on content length
                        total_chars = len(left_content) + len(right_content)
                        dynamic_height = min(800, max(500, total_chars // 25))
                        
                        # Use enhanced synchronized scrolling with full width and automatic inline commenting
                        selection_event = create_synchronized_text_display(
                            left_text=left_content,
                            right_text=right_content,
                            left_title=f"📖 {left_style}",
                            right_title=f"📖 {right_style} (Select text to comment)",
                            height=dynamic_height,
                            enable_comments=True,
                            chapter_id=str(selected_chapter),
                            style_name=right_style.replace("Custom: ", "") if right_style.startswith("Custom: ") else "official",
                            key=f"text_display_{selected_chapter}_{right_style}"
                        )
                        
                        # Handle automatic text selection for commenting
                        if selection_event and selection_event.get('type') == 'text_selected':
                            st.session_state['pending_comment'] = selection_event
                            st.rerun()
                        
                        # Show comparison stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Left Panel Words", len(left_content.split()))
                        with col2:
                            st.metric("Right Panel Words", len(right_content.split()))
                        with col3:
                            # Calculate rough similarity (word overlap)
                            left_words = set(left_content.lower().split())
                            right_words = set(right_content.lower().split())
                            overlap = len(left_words.intersection(right_words))
                            total_unique = len(left_words.union(right_words))
                            similarity = (overlap / total_unique * 100) if total_unique > 0 else 0
                            st.metric("Word Overlap", f"{similarity:.1f}%")
                        
                        # Enhanced: Automatic Inline Comment Creation Interface
                        st.divider()
                        st.subheader("💬 Inline Comments")
                        
                        # Show automatic comment form when text is selected
                        if 'pending_comment' in st.session_state and st.session_state.pending_comment:
                            pending = st.session_state.pending_comment
                            
                            st.success("✨ **Text Selected!** Create your inline comment below:")
                            
                            with st.form("automatic_comment_form"):
                                st.info(f"**Selected Text:** \"{pending['text']}\"")
                                
                                # Dimension selection
                                comment_dimension = st.selectbox(
                                    "Evaluation Dimension",
                                    ["english_sophistication", "world_building", "emotional_impact", "dialogue_naturalness"],
                                    format_func=lambda x: {
                                        "english_sophistication": "🎯 English Sophistication",
                                        "world_building": "🌍 World Building & Imagery", 
                                        "emotional_impact": "💔 Emotional Impact",
                                        "dialogue_naturalness": "💬 Dialogue Naturalness"
                                    }[x],
                                    help="Choose the quality dimension this text exemplifies"
                                )
                                
                                # Comment text
                                comment_text = st.text_area(
                                    "Your Comment",
                                    placeholder="Explain why this text exemplifies this quality dimension...",
                                    help="Detailed feedback about this specific text segment",
                                    height=120
                                )
                                
                                # Evaluator info (optional)
                                col1, col2 = st.columns(2)
                                with col1:
                                    evaluator_name = st.text_input("Your Name (Optional)", placeholder="e.g., Aditya Prasad")
                                with col2:
                                    evaluator_email = st.text_input("Email (Optional)", placeholder="e.g., aditya@example.com")
                                
                                # Form buttons
                                col1, col2 = st.columns(2)
                                with col1:
                                    submit_comment = st.form_submit_button("💾 Save Comment", type="primary", use_container_width=True)
                                with col2:
                                    cancel_comment = st.form_submit_button("❌ Cancel", use_container_width=True)
                                
                                if submit_comment and comment_text.strip():
                                    from utils import add_inline_comment
                                    
                                    comment_data = {
                                        'start_char': pending['start_char'],
                                        'end_char': pending['end_char'],
                                        'selected_text': pending['text'],
                                        'dimension': comment_dimension,
                                        'comment': comment_text.strip(),
                                        'evaluator_name': evaluator_name.strip() if evaluator_name.strip() else 'Anonymous',
                                        'evaluator_email': evaluator_email.strip() if evaluator_email.strip() else ''
                                    }
                                    
                                    # Save the comment
                                    comment_id = add_inline_comment(pending['style_name'], pending['chapter_id'], comment_data)
                                    
                                    st.success(f"✅ Inline comment saved! ID: {comment_id}")
                                    
                                    # Clear pending comment and refresh
                                    del st.session_state['pending_comment']
                                    st.rerun()
                                
                                elif cancel_comment:
                                    # Cancel comment creation
                                    del st.session_state['pending_comment']
                                    st.rerun()
                                
                                elif submit_comment and not comment_text.strip():
                                    st.error("❌ Please enter a comment before saving.")
                        
                        else:
                            st.caption("📖 **How to use**: Select any text in the right panel above to automatically create an inline comment with dimension-specific feedback.")
                        
                        # Display existing comments
                        from utils import load_inline_comments
                        style_name_clean = right_style.replace("Custom: ", "") if right_style.startswith("Custom: ") else "official"
                        existing_comments = load_inline_comments(style_name_clean, str(selected_chapter))
                        
                        if existing_comments:
                            with st.expander(f"📋 Existing Comments ({len(existing_comments)})", expanded=True):
                                for i, comment in enumerate(existing_comments):
                                    dimension_icon = {
                                        'english_sophistication': '🎯',
                                        'world_building': '🌍',
                                        'emotional_impact': '💔',
                                        'dialogue_naturalness': '💬'
                                    }.get(comment.get('dimension', 'english_sophistication'), '📝')
                                    
                                    st.info(f"""
                                    **{dimension_icon} {comment.get('dimension', '').replace('_', ' ').title()}**
                                    
                                    *Selected Text:* "{comment.get('selected_text', '')}"
                                    
                                    *Comment:* {comment.get('comment', '')}
                                    
                                    *By:* {comment.get('evaluator_name', 'Anonymous')} • {comment.get('timestamp', '')[:16]}
                                    """)
                        else:
                            st.info("💡 No inline comments yet. Select text in the translation above to add the first comment!")
                    else:
                        # Fallback display for unavailable content
                        if "not available" in left_content or "not found" in left_content:
                            st.warning(f"⚠️ {left_style}: {left_content}")
                        else:
                            st.markdown(f"**📖 {left_style}**")
                            st.text_area(
                                f"{left_style}:", 
                                left_content,
                                height=400,
                                disabled=True,
                                key=f"left_{selected_chapter}"
                            )
                        
                        if "not available" in right_content or "not found" in right_content:
                            st.warning(f"⚠️ {right_style}: {right_content}")
                        else:
                            st.markdown(f"**📖 {right_style}**")
                            st.text_area(
                                f"{right_style}:", 
                                right_content,
                                height=400,
                                disabled=True,
                                key=f"right_{selected_chapter}"
                            )
                    
                    # Human scoring interface
                    st.subheader("📊 Quality Assessment")
                    
                    # Load existing human scores for this style and chapter
                    existing_human_scores = load_human_scores(eval_style['name'])
                    chapter_scores = existing_human_scores.get(str(selected_chapter), {})
                    
                    with st.form(f"human_eval_{eval_style['name']}_{selected_chapter}"):
                        st.write("**Current Human Evaluation Metrics (4 Dimensions)**")
                        st.write("Rate this translation on core quality dimensions (1-100):")
                        
                        # Enhanced: User Identity Section (Optional for backward compatibility)
                        with st.expander("👤 Evaluator Information (Optional)", expanded=False):
                            col_id1, col_id2 = st.columns(2)
                            with col_id1:
                                evaluator_name = st.text_input(
                                    "Your Name",
                                    value=chapter_scores.get('evaluator_name', ''),
                                    placeholder="e.g., Aditya Prasad",
                                    help="Optional: Your name for tracking evaluation history"
                                )
                            with col_id2:
                                evaluator_email = st.text_input(
                                    "Email (Optional)",
                                    value=chapter_scores.get('evaluator_email', ''),
                                    placeholder="e.g., aditya@example.com",
                                    help="Optional: For contact and evaluation attribution"
                                )
                        
                        st.divider()
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            english_sophistication = st.slider(
                                "🎯 English Sophistication",
                                1, 100,
                                value=chapter_scores.get('english_sophistication', 50),
                                help="Nuanced, complex language usage, Vocabulary - Employ varied, precise vocabulary (furious/incensed/livid instead of repeated 'angry'), 100 means using extremely niche, obscure words. Maintain distinct character voices through speech patterns, formality levels, and vocabulary choices"
                            )
                            
                            world_building = st.slider(
                                "🌍 World Building & Imagery",
                                1, 100,
                                value=chapter_scores.get('world_building', 50),
                                help="Rich descriptions, terminology and context, footnotes/glossary that details choices to transliterate in Pinyin, deep cultural/historical references. maintain consistency."
                            )
                        
                        with col2:
                            emotional_impact = st.slider(
                                "💔 Emotional Impact",
                                1, 100,
                                value=chapter_scores.get('emotional_impact', 50),
                                help="How evocative the prose is, creativity - Balance epic grandeur with humor and modern colloquialisms where character-appropriate. Handle poetry/wordplay by preserving effect over literal meaning. Your task is to convey the meaning, tone, and impact of the original."
                            )
                            
                            dialogue_naturalness = st.slider(
                                "💬 Dialogue Naturalness",
                                1, 100,
                                value=chapter_scores.get('dialogue_naturalness', 50),
                                help="Authentic conversation flow, no jarring transitions in the transitions between sentences, paragraphs, chapters. Formatting - Readability - Add sufficient line breaks between each speaker's dialogue, line breaks between scene transitions or time skips"
                            )
                        
                        # Enhanced: Text Justifications Section (Optional for backward compatibility)
                        st.divider()
                        with st.expander("📝 Detailed Justifications (Optional - For Future AI Features)", expanded=False):
                            st.caption("💡 **Future Feature Preview**: These justifications will help train AI to understand your quality standards and extrapolate scores to hundreds of chapters automatically.")
                            
                            # Get existing justifications if they exist
                            existing_justifications = chapter_scores.get('justifications', {})
                            
                            justification_english = st.text_area(
                                "🎯 English Sophistication Justification",
                                value=existing_justifications.get('english_sophistication', ''),
                                placeholder="Explain your reasoning for the English Sophistication score. What specific vocabulary, syntax, or stylistic elements influenced your rating?",
                                help="Detailed explanation of why you gave this score",
                                height=80
                            )
                            
                            justification_world = st.text_area(
                                "🌍 World Building & Imagery Justification",
                                value=existing_justifications.get('world_building', ''),
                                placeholder="Describe how well the translation handles cultural context, descriptive passages, and world-building elements.",
                                help="Explanation of world-building and imagery quality",
                                height=80
                            )
                            
                            justification_emotion = st.text_area(
                                "💔 Emotional Impact Justification",
                                value=existing_justifications.get('emotional_impact', ''),
                                placeholder="Assess the emotional resonance and impact of the translation. How well does it convey the original's emotional intent?",
                                help="Explanation of emotional impact assessment",
                                height=80
                            )
                            
                            justification_dialogue = st.text_area(
                                "💬 Dialogue Naturalness Justification",
                                value=existing_justifications.get('dialogue_naturalness', ''),
                                placeholder="Evaluate conversation flow, character voice consistency, and natural speech patterns.",
                                help="Explanation of dialogue quality assessment",
                                height=80
                            )
                            
                            overall_notes = st.text_area(
                                "📋 Overall Translation Notes",
                                value=chapter_scores.get('overall_notes', ''),
                                placeholder="General observations, comparative notes, or suggestions for improvement.",
                                help="General comments about this translation",
                                height=100
                            )
                        
                        # Submit button
                        submitted = st.form_submit_button("💾 Save Evaluation", type="primary")
                        
                        if submitted:
                            # Update scores
                            if str(selected_chapter) not in existing_human_scores:
                                existing_human_scores[str(selected_chapter)] = {}
                            
                            # Core evaluation data (always saved)
                            evaluation_data = {
                                'english_sophistication': english_sophistication,
                                'world_building': world_building,
                                'emotional_impact': emotional_impact,
                                'dialogue_naturalness': dialogue_naturalness,
                                'evaluated_at': datetime.now().isoformat(),
                                'evaluator': 'human'  # Keep for backward compatibility
                            }
                            
                            # Enhanced data (only save if provided - backward compatible)
                            if evaluator_name.strip():
                                evaluation_data['evaluator_name'] = evaluator_name.strip()
                            if evaluator_email.strip():
                                evaluation_data['evaluator_email'] = evaluator_email.strip()
                            
                            # Text justifications (only save non-empty ones)
                            justifications = {}
                            if justification_english.strip():
                                justifications['english_sophistication'] = justification_english.strip()
                            if justification_world.strip():
                                justifications['world_building'] = justification_world.strip()
                            if justification_emotion.strip():
                                justifications['emotional_impact'] = justification_emotion.strip()
                            if justification_dialogue.strip():
                                justifications['dialogue_naturalness'] = justification_dialogue.strip()
                            
                            if justifications:  # Only add if at least one justification provided
                                evaluation_data['justifications'] = justifications
                            
                            if overall_notes.strip():
                                evaluation_data['overall_notes'] = overall_notes.strip()
                            
                            # Add evaluation version for future tracking
                            if any([evaluator_name.strip(), evaluator_email.strip(), justifications, overall_notes.strip()]):
                                evaluation_data['evaluation_version'] = '2.0'  # Enhanced evaluation
                            else:
                                evaluation_data['evaluation_version'] = '1.0'  # Legacy evaluation
                            
                            existing_human_scores[str(selected_chapter)].update(evaluation_data)
                            
                            # Save to file
                            save_human_scores(eval_style['name'], existing_human_scores)
                            st.success(f"✅ Evaluation saved for Chapter {selected_chapter}!")
                            st.rerun()
                    
                    # Enhanced: Display existing evaluation details if available
                    if chapter_scores and any(key in chapter_scores for key in ['evaluator_name', 'justifications', 'overall_notes']):
                        st.divider()
                        with st.expander("📊 Current Evaluation Details", expanded=False):
                            # Show evaluator information
                            if chapter_scores.get('evaluator_name'):
                                eval_name = chapter_scores.get('evaluator_name')
                                eval_email = chapter_scores.get('evaluator_email', '')
                                eval_date = chapter_scores.get('evaluated_at', '')
                                eval_version = chapter_scores.get('evaluation_version', '1.0')
                                
                                st.caption(f"**Evaluator:** {eval_name}")
                                if eval_email:
                                    st.caption(f"**Email:** {eval_email}")
                                if eval_date:
                                    from datetime import datetime
                                    try:
                                        date_obj = datetime.fromisoformat(eval_date.replace('Z', '+00:00'))
                                        st.caption(f"**Evaluated:** {date_obj.strftime('%Y-%m-%d %H:%M')}")
                                    except:
                                        st.caption(f"**Evaluated:** {eval_date}")
                                st.caption(f"**Version:** {eval_version}")
                            
                            # Show justifications if available
                            if chapter_scores.get('justifications'):
                                justifications = chapter_scores['justifications']
                                st.write("**Detailed Justifications:**")
                                
                                if justifications.get('english_sophistication'):
                                    st.info(f"🎯 **English Sophistication**: {justifications['english_sophistication']}")
                                if justifications.get('world_building'):
                                    st.info(f"🌍 **World Building**: {justifications['world_building']}")
                                if justifications.get('emotional_impact'):
                                    st.info(f"💔 **Emotional Impact**: {justifications['emotional_impact']}")
                                if justifications.get('dialogue_naturalness'):
                                    st.info(f"💬 **Dialogue Naturalness**: {justifications['dialogue_naturalness']}")
                            
                            # Show overall notes if available
                            if chapter_scores.get('overall_notes'):
                                st.write("**Overall Notes:**")
                                st.info(f"📝 {chapter_scores['overall_notes']}")
    
    with col2:
        st.subheader("📋 Evaluation Progress")
        
        # Show human evaluation progress
        human_scores = load_human_scores(eval_style['name'])
        human_completion = len(human_scores) / len(available_chapters) if available_chapters else 0
        
        st.metric(
            "👤 Human Evaluated",
            f"{len(human_scores)}/{len(available_chapters)}",
            f"{human_completion:.1%} complete"
        )
        
        if human_scores:
            # Show average scores
            all_scores = list(human_scores.values())
            if all_scores:
                avg_sophistication = sum(s.get('english_sophistication', 0) for s in all_scores) / len(all_scores)
                avg_world = sum(s.get('world_building', 0) for s in all_scores) / len(all_scores)
                avg_emotion = sum(s.get('emotional_impact', 0) for s in all_scores) / len(all_scores)
                avg_dialogue = sum(s.get('dialogue_naturalness', 0) for s in all_scores) / len(all_scores)
                
                st.caption("**Average Scores:**")
                st.caption(f"🎯 English: {avg_sophistication:.1f}")
                st.caption(f"🌍 World: {avg_world:.1f}")
                st.caption(f"💔 Emotion: {avg_emotion:.1f}")
                st.caption(f"💬 Dialogue: {avg_dialogue:.1f}")
                
                # Enhanced: Show evaluator information if available
                evaluators = set()
                enhanced_evaluations = 0
                with_justifications = 0
                
                for score in all_scores:
                    if score.get('evaluator_name'):
                        evaluators.add(score.get('evaluator_name'))
                    if score.get('evaluation_version') == '2.0':
                        enhanced_evaluations += 1
                    if score.get('justifications'):
                        with_justifications += 1
                
                if evaluators:
                    st.caption(f"👥 **Evaluators:** {', '.join(sorted(evaluators))}")
                if enhanced_evaluations > 0:
                    st.caption(f"📝 **Enhanced Evaluations:** {enhanced_evaluations}/{len(all_scores)} with detailed data")
                if with_justifications > 0:
                    st.caption(f"💭 **With Justifications:** {with_justifications}/{len(all_scores)} chapters")
    
    st.divider()
    
    # Comprehensive Leaderboard
    st.subheader("🏆 Comprehensive Style Rankings")
    
    # Calculate composite scores for all selected styles
    leaderboard_data = []
    
    for style_label in selected_styles:
        style = style_options[style_label]
        style_name = style['name']
        
        # Load evaluation data
        bert_scores = load_bert_scores(style_name)
        human_scores = load_human_scores(style_name)
        
        if not bert_scores and not human_scores:
            continue
        
        # Calculate comprehensive score
        score_breakdown = calculate_composite_score(bert_scores, human_scores, style['chapter_count'])
        
        leaderboard_data.append({
            'Style': style_name,
            'Model': style['model_name'],
            'Chapters': style['chapter_count'],
            'Evaluated': score_breakdown['evaluated_chapters'],
            'BERT Mean': f"{score_breakdown['mean_bert']:.3f}" if score_breakdown['mean_bert'] > 0 else "N/A",
            'BERT Std': f"{score_breakdown['std_bert']:.3f}" if score_breakdown['std_bert'] > 0 else "N/A",
            'Quality Score': f"{score_breakdown['quality_score']:.3f}",
            'Consistency': f"{score_breakdown['consistency_bonus']:.3f}",
            'Completeness': f"{score_breakdown['completeness_bonus']:.3f}",
            'Composite Score': f"{score_breakdown['composite_score']:.3f}",
            '_composite_raw': score_breakdown['composite_score']
        })
    
    if leaderboard_data:
        # Sort by composite score
        leaderboard_data.sort(key=lambda x: x['_composite_raw'], reverse=True)
        
        # Add ranks
        for i, entry in enumerate(leaderboard_data):
            entry['Rank'] = i + 1
        
        # Create display DataFrame
        display_columns = ['Rank', 'Style', 'Model', 'Chapters', 'Evaluated', 'BERT Mean', 'Quality Score', 'Composite Score']
        display_df = pd.DataFrame([{k: v for k, v in item.items() if k in display_columns} for item in leaderboard_data])
        
        # Style the DataFrame
        def style_composite(val):
            if isinstance(val, str) and val.replace('.', '').isdigit():
                score = float(val)
                if score >= 2.0:
                    return "background-color: #90EE90"  # Light green
                elif score >= 1.5:
                    return "background-color: #FFFFE0"  # Light yellow
                elif score < 1.0:
                    return "background-color: #FFE4E1"  # Light red
            return ""
        
        def style_rank(val):
            if val == 1:
                return "background-color: #FFD700; font-weight: bold"  # Gold
            elif val == 2:
                return "background-color: #C0C0C0; font-weight: bold"  # Silver
            elif val == 3:
                return "background-color: #CD7F32; font-weight: bold"  # Bronze
            return ""
        
        styled_df = display_df.style.applymap(style_composite, subset=['Composite Score']).applymap(style_rank, subset=['Rank'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Champion announcement
        if leaderboard_data:
            champion = leaderboard_data[0]
            st.success(f"🏆 **Current Champion:** {champion['Style']} with composite score {champion['Composite Score']}")
        
        # Detailed breakdown in expander
        with st.expander("📊 Detailed Score Breakdown"):
            detailed_columns = ['Rank', 'Style', 'Quality Score', 'Consistency', 'Completeness', 'Composite Score', 'BERT Mean', 'BERT Std']
            detailed_df = pd.DataFrame([{k: v for k, v in item.items() if k in detailed_columns} for item in leaderboard_data])
            st.dataframe(detailed_df, use_container_width=True)
        
        # Export functionality
        if st.button("📥 Export Evaluation Report"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_data = {
                'timestamp': timestamp,
                'evaluation_method': {
                    'bert_weight': 0.5,
                    'human_weight': 0.5,
                    'consistency_formula': '(1 - std_deviation)',
                    'completeness_formula': 'log10(evaluated_chapters + 1)'
                },
                'leaderboard': leaderboard_data,
                'styles_evaluated': len(selected_styles)
            }
            
            report_file = os.path.join(EXPORT_DIR, f"style_evaluation_report_{timestamp}.json")
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            st.success(f"📁 Report saved: {os.path.basename(report_file)}")
    else:
        st.info("📊 No evaluation data available. Calculate BERT scores and add human evaluations to see rankings.")

# --- Footer ---
st.divider()
st.caption("🧪 **Experimentation Lab** | Part of the Translation Framework Suite")

# Save experiment history
if hasattr(st.session_state, 'evaluation_results') and st.session_state.evaluation_results:
    # Option to save current experiment
    if st.sidebar.button("💾 Save Current Experiment"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_data = {
            "timestamp": timestamp,
            "experiment_config": {
                "base_model": selected_base_model,
                "tuned_model": selected_tuned_model,
                "n_shot": n_shot_examples,
                "evaluation_chapters": evaluation_chapters,
                "random_seed": random_seed
            },
            "results": st.session_state.evaluation_results
        }
        
        experiment_file = os.path.join(EXPORT_DIR, f"experiment_{timestamp}.json")
        try:
            with open(experiment_file, 'w', encoding='utf-8') as f:
                json.dump(experiment_data, f, indent=2, ensure_ascii=False)
            st.sidebar.success(f"💾 Experiment saved: {os.path.basename(experiment_file)}")
        except Exception as e:
            st.sidebar.error(f"❌ Save failed: {e}")