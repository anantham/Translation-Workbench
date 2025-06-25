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

st.title("📈 Experimentation Analysis")
st.caption("**The Judging Panel** | Compare translation styles, analyze quality metrics, and crown the winner")

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
                    
                    # Display content with synchronized scrolling
                    st.subheader(f"📖 Chapter {selected_chapter} Comparison")
                    
                    if official_translation and "File not found" not in official_translation:
                        # Use the new synchronized scrolling component
                        create_synchronized_text_display(
                            left_text=custom_translation,
                            right_text=official_translation,
                            left_title="🎨 Custom Translation",
                            right_title="📚 Official Translation",
                            height=350
                        )
                    else:
                        # Fallback for when official translation is not available
                        st.warning("Official translation not available for comparison")
                        st.markdown("**🎨 Custom Translation**")
                        st.text_area(
                            "Custom Translation:", 
                            custom_translation,
                            height=300,
                            disabled=True
                        )
                    
                    # Human scoring interface
                    st.subheader("📊 Quality Assessment")
                    
                    # Load existing human scores for this style and chapter
                    existing_human_scores = load_human_scores(eval_style['name'])
                    chapter_scores = existing_human_scores.get(str(selected_chapter), {})
                    
                    with st.form(f"human_eval_{eval_style['name']}_{selected_chapter}"):
                        st.write("Rate this translation on various quality dimensions (1-100):")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            english_sophistication = st.slider(
                                "🎯 English Sophistication",
                                1, 100,
                                value=chapter_scores.get('english_sophistication', 50),
                                help="Nuanced, complex, appropriate jargon usage"
                            )
                            
                            world_building = st.slider(
                                "🌍 World Building & Imagery",
                                1, 100,
                                value=chapter_scores.get('world_building', 50),
                                help="Rich descriptions of scenery, context, background"
                            )
                        
                        with col2:
                            emotional_impact = st.slider(
                                "💔 Emotional Impact",
                                1, 100,
                                value=chapter_scores.get('emotional_impact', 50),
                                help="How evocative and heart-gripping the prose is"
                            )
                            
                            dialogue_naturalness = st.slider(
                                "💬 Dialogue Naturalness",
                                1, 100,
                                value=chapter_scores.get('dialogue_naturalness', 50),
                                help="How natural and authentic conversations sound"
                            )
                        
                        # Additional quality dimensions
                        st.write("**Additional Quality Metrics:**")
                        
                        col3, col4 = st.columns(2)
                        
                        with col3:
                            action_clarity = st.slider(
                                "⚔️ Action Clarity",
                                1, 100,
                                value=chapter_scores.get('action_clarity', 50),
                                help="Clarity of fight scenes and dynamic sequences"
                            )
                            
                            cultural_adaptation = st.slider(
                                "🏛️ Cultural Adaptation",
                                1, 100,
                                value=chapter_scores.get('cultural_adaptation', 50),
                                help="Handling of idioms, cultivation terms, cultural references"
                            )
                        
                        with col4:
                            pacing = st.slider(
                                "🎵 Narrative Pacing",
                                1, 100,
                                value=chapter_scores.get('pacing', 50),
                                help="Maintains original flow and rhythm"
                            )
                            
                            consistency = st.slider(
                                "📏 Terminology Consistency",
                                1, 100,
                                value=chapter_scores.get('consistency', 50),
                                help="Consistent use of names, terms, and style"
                            )
                        
                        # Submit button
                        submitted = st.form_submit_button("💾 Save Evaluation", type="primary")
                        
                        if submitted:
                            # Update scores
                            if str(selected_chapter) not in existing_human_scores:
                                existing_human_scores[str(selected_chapter)] = {}
                            
                            existing_human_scores[str(selected_chapter)].update({
                                'english_sophistication': english_sophistication,
                                'world_building': world_building,
                                'emotional_impact': emotional_impact,
                                'dialogue_naturalness': dialogue_naturalness,
                                'action_clarity': action_clarity,
                                'cultural_adaptation': cultural_adaptation,
                                'pacing': pacing,
                                'consistency': consistency,
                                'evaluated_at': datetime.now().isoformat(),
                                'evaluator': 'human'
                            })
                            
                            # Save to file
                            save_human_scores(eval_style['name'], existing_human_scores)
                            st.success(f"✅ Evaluation saved for Chapter {selected_chapter}!")
                            st.rerun()
    
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