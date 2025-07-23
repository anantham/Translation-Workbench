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
import numpy as np


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

# ==============================================================================
#                    INLINE COMMENT EVENT BUS PROCESSOR
# ==============================================================================
# Process any pending comment events from session state before rendering the rest of the page

if 'last_comment_event' in st.session_state and st.session_state.last_comment_event is not None:
    print("🎉 EVENT BUS: Detected comment event in session state")
    event_data_str = st.session_state.last_comment_event
    
    # Clear the event immediately to prevent reprocessing
    st.session_state.last_comment_event = None
    print("🎉 EVENT BUS: Cleared event from session state")
    
    try:
        event_data = json.loads(event_data_str)
        print(f"🎉 EVENT BUS: Parsed event data: {event_data.get('type')}")
        
        if event_data.get('type') == 'comment_saved':
            print("🎉 EVENT BUS: Processing comment_saved event")
            
            # Extract and clean style name
            raw_style_name = event_data.get('style_name', 'unknown')
            processed_style_name = raw_style_name.replace("Custom: ", "") if raw_style_name.startswith("Custom: ") else "official"
            print(f"🎉 EVENT BUS: Style '{raw_style_name}' -> '{processed_style_name}'")
            
            # Prepare comment data
            comment_data = {
                'start_offset': event_data['start_char'],
                'end_offset': event_data['end_char'],
                'selected_text': event_data['text'],
                'dimension': event_data['dimension'],
                'comment': event_data['comment'],
                'evaluator_name': 'User',
                'timestamp': event_data['timestamp']
            }
            
            print(f"🎉 EVENT BUS: Saving comment for chapter {event_data['chapter_id']}")
            
            # Save the comment
            from utils import add_inline_comment
            comment_id = add_inline_comment(
                processed_style_name,
                event_data['chapter_id'],
                comment_data
            )
            
            if comment_id:
                st.toast(f"💬 Comment saved for Ch. {event_data['chapter_id']}!", icon="✅")
                print(f"🎉 EVENT BUS: SUCCESS - Comment saved with ID {comment_id}")
            else:
                st.toast("❌ Failed to save comment", icon="🔥")
                print("🎉 EVENT BUS: FAILURE - add_inline_comment returned None")
                
    except (json.JSONDecodeError, KeyError) as e:
        print(f"🎉 EVENT BUS: ERROR - Could not process event: {e}")
        print(f"🎉 EVENT BUS: Raw event data: {event_data_str}")

# ==============================================================================
#                         END EVENT BUS PROCESSOR
# ==============================================================================

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
        help="Select which novel's alignment map to use for experimentation"
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
    label = f"{style.get('name', 'Unknown Style')} ({style.get('chapter_count', 0)} chapters)"
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

# Create tabs for leaderboard and human evaluation
tab1, tab2 = st.tabs([
    "🏆 Style Leaderboard & Rankings", 
    "💬 Human Quality Assessment"
])

# --- Tab 1: Style Leaderboard & Rankings ---
with tab1:
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
        # Get model name from metadata if available, otherwise use default
        model_name = style.get('metadata', {}).get('model_name', 'Unknown Model')
        label = f"{style.get('name', 'Unknown Style')} ({style.get('chapter_count', 0)} chapters, {model_name})"
        style_options[label] = style
    
    selected_styles = st.multiselect(
        "Select styles to compare:",
        list(style_options.keys()),
        default=list(style_options.keys())[:3] if len(style_options) > 3 else list(style_options.keys()),
        help="Choose translation styles to evaluate and rank"
    )
    
    if not selected_styles:
        st.info("👆 Please select at least one translation style to evaluate")
        st.stop()
        
    # Evaluation metrics selection
    st.subheader("🔍 Evaluation Metrics")
    col1, col2 = st.columns(2)
    with col1:
        show_bert = st.checkbox("🧠 BERT Semantic Similarity", value=True, help="Automated semantic similarity scoring")
        show_human = st.checkbox("👤 Human Quality Assessment", value=True, help="Human evaluation scores")
    with col2:
        show_composite = st.checkbox("🎯 Composite Ranking", value=True, help="Combined BERT + Human scoring")
        show_detailed = st.checkbox("📊 Detailed Breakdown", value=False, help="Show individual dimension scores")
    
    # Process selected styles
    leaderboard_data = []
    
    for style_label in selected_styles:
        style = style_options[style_label]
        style_name = style.get('name', 'Unknown Style')
        
        # Initialize metrics
        bert_scores = load_bert_scores(style_name) if show_bert else {}
        human_scores = load_human_scores(style_name) if show_human else {}
        
        # Calculate aggregate scores
        bert_avg = np.mean(list(bert_scores.values())) if bert_scores else 0
        
        if human_scores:
            # Calculate human dimension averages
            all_human_scores = []
            # New evaluation dimensions (v2.0) with backwards compatibility
            dimensions = ['vocabulary_complexity', 'cultural_context', 'prose_style', 'creative_fidelity']
            legacy_dimensions = ['english_sophistication', 'world_building', 'emotional_impact', 'dialogue_naturalness']
            dimension_avgs = {}
            
            # Try new dimensions first, fall back to legacy if no new data exists
            dimensions_to_use = dimensions
            if not any(any(scores.get(dim, 0) > 0 for dim in dimensions) for scores in human_scores.values()):
                # No new dimension data found, use legacy dimensions
                dimensions_to_use = legacy_dimensions
            
            for dim in dimensions_to_use:
                dim_scores = [scores.get(dim, 0) / 100 for scores in human_scores.values() if scores.get(dim, 0) > 0]
                if dim_scores:
                    dimension_avgs[dim] = np.mean(dim_scores)
                    all_human_scores.extend(dim_scores)
            
            human_avg = np.mean(all_human_scores) if all_human_scores else 0
        else:
            human_avg = 0
            dimension_avgs = {}
        
        # Composite score
        if bert_avg > 0 and human_avg > 0:
            composite_score = (bert_avg * 0.5) + (human_avg * 0.5)
        elif bert_avg > 0:
            composite_score = bert_avg * 0.8  # Penalize lack of human eval
        elif human_avg > 0:
            composite_score = human_avg * 0.8  # Penalize lack of BERT eval
        else:
            composite_score = 0
        
        leaderboard_data.append({
            'Style': style_name,
            'Model': style.get('metadata', {}).get('model_name', 'Unknown Model'),
            'Chapters': len(bert_scores) + len(human_scores),
            'BERT Score': bert_avg,
            'Human Score': human_avg,
            'Composite Score': composite_score,
            'Details': dimension_avgs
        })
    
    # Sort by composite score
    leaderboard_data.sort(key=lambda x: x['Composite Score'], reverse=True)
    
    # Display leaderboard
    st.subheader("🏆 Translation Quality Leaderboard")
    
    for i, style_data in enumerate(leaderboard_data):
        rank = i + 1
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"  #{rank}"
        
        with st.container():
            col1, col2, col3 = st.columns([0.5, 2, 1])
            
            with col1:
                st.markdown(f"### {medal}")
            
            with col2:
                st.markdown(f"**{style_data['Style']}**")
                st.caption(f"Model: {style_data['Model']} | Chapters: {style_data['Chapters']}")
            
            with col3:
                if show_composite and style_data['Composite Score'] > 0:
                    st.metric("Composite Score", f"{style_data['Composite Score']:.3f}")
                elif show_bert and style_data['BERT Score'] > 0:
                    st.metric("BERT Score", f"{style_data['BERT Score']:.3f}")
                elif show_human and style_data['Human Score'] > 0:
                    st.metric("Human Score", f"{style_data['Human Score']:.3f}")
            
            # Detailed breakdown
            if show_detailed:
                detail_cols = st.columns(4)
                if show_bert and style_data['BERT Score'] > 0:
                    detail_cols[0].metric("🧠 BERT", f"{style_data['BERT Score']:.3f}")
                if show_human and style_data['Human Score'] > 0:
                    detail_cols[1].metric("👤 Human", f"{style_data['Human Score']:.3f}")
                
                # Human dimensions
                if style_data['Details']:
                    dim_names = ['vocabulary_complexity', 'cultural_context', 'prose_style', 'creative_fidelity']
                    dim_labels = ['🧠 Vocabulary', '🌏 Cultural', '✍️ Prose', '🎨 Creative']
                    
                    for j, (dim, label) in enumerate(zip(dim_names, dim_labels)):
                        if dim in style_data['Details'] and j < 2:  # Show first 2 dimensions
                            detail_cols[j + 2].metric(label, f"{style_data['Details'][dim]:.3f}")
        
        st.divider()
    
    # Performance comparison chart
    if len(leaderboard_data) > 1:
        st.subheader("📊 Performance Comparison")
        
        chart_data = []
        for style_data in leaderboard_data:
            if show_bert and style_data['BERT Score'] > 0:
                chart_data.append({'Style': style_data['Style'], 'Metric': 'BERT Score', 'Score': style_data['BERT Score']})
            if show_human and style_data['Human Score'] > 0:
                chart_data.append({'Style': style_data['Style'], 'Metric': 'Human Score', 'Score': style_data['Human Score']})
            if show_composite and style_data['Composite Score'] > 0:
                chart_data.append({'Style': style_data['Style'], 'Metric': 'Composite Score', 'Score': style_data['Composite Score']})
        
        if chart_data:
            chart_df = pd.DataFrame(chart_data)
            fig = px.bar(
                chart_df,
                x='Style',
                y='Score',
                color='Metric',
                title='Translation Quality Comparison',
                barmode='group'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)


# --- Tab 2: Human Quality Assessment ---
with tab2:
    st.header("💬 Human Quality Assessment & Inline Commenting")
    st.caption("**Interactive Translation Review** | Select text and add dimension-specific comments with seamless sidebar workflow")
    
    # Get available translation styles
    available_styles = get_available_translation_styles()
    
    if not available_styles:
        st.info("🎨 No custom translation styles found. Generate translations in the Pluralistic Translation Lab first.")
        st.stop()
    
    # Style selection interface
    st.subheader("📚 Select Style for Human Evaluation")
    
    # Create style selection with metadata display
    style_options = {}
    for style in available_styles:
        # Get model name from metadata if available, otherwise use default
        model_name = style.get('metadata', {}).get('model_name', 'Unknown Model')
        label = f"{style.get('name', 'Unknown Style')} ({style.get('chapter_count', 0)} chapters, {model_name})"
        style_options[label] = style
    
    selected_style_label = st.selectbox(
        "Choose translation style to evaluate:",
        list(style_options.keys()),
        help="Select one translation style for detailed human assessment"
    )
    
    eval_style = style_options[selected_style_label]
    
    # Get chapters available for this style
    style_path = eval_style.get('path', '')
    available_chapters = []
    
    if style_path and os.path.exists(style_path):
        for filename in os.listdir(style_path):
            if filename.endswith('-translated.txt'):
                try:
                    chapter_num = int(filename.split('-')[1])
                    available_chapters.append(chapter_num)
                except (ValueError, IndexError):
                    continue
    
    available_chapters.sort()
    
    if not available_chapters:
        st.warning("No translated chapters found for this style.")
        st.stop()
    
    # Chapter selection
    col1, col2 = st.columns(2)
    
    with col1:
        selected_chapter = st.selectbox(
            "Select chapter to evaluate:",
            available_chapters,
            format_func=lambda x: f"Chapter {x}",
            help="Choose a chapter to review and score"
        )
    
    with col2:
        # Show evaluation progress
        human_scores = load_human_scores(eval_style.get('name', 'Unknown Style'))
        human_completion = len(human_scores) / len(available_chapters) if available_chapters else 0
        
        st.metric(
            "👤 Human Evaluated",
            f"{len(human_scores)}/{len(available_chapters)}",
            f"{human_completion:.1%} complete"
        )
    
    st.divider()
    
    # Main content display with full width
    st.header(f"📖 Chapter {selected_chapter} Comparison & Review")
    
    # Load custom translation
    custom_file = os.path.join(style_path, f"Chapter-{selected_chapter:04d}-translated.txt")
    if os.path.exists(custom_file):
        with open(custom_file, 'r', encoding='utf-8') as f:
            custom_translation = f.read()
    else:
        st.error("Custom translation file not found")
        st.stop()
    
    # Load official translation for comparison
    official_translation = ""
    if str(selected_chapter) in alignment_map:
        official_file = alignment_map[str(selected_chapter)].get('english_file')
        if official_file:
            official_translation = load_chapter_content(official_file)
    
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
            index=len(available_translation_styles)-1 if len(available_translation_styles) > 1 else 0,
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
    
    # Load and process translation content - FULL WIDTH
    left_content = get_translation_content(left_style)
    right_content = get_translation_content(right_style)
    
    # Display content with enhanced synchronized scrolling - FULL WIDTH
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
        
        # Handle automatic inline commenting
        print("="*80)
        print("🔍 EVENT RECEIVED: Inline comment event processing started")
        print(f"🔍 EVENT TYPE: {type(selection_event)}")
        print(f"🔍 EVENT DATA: {str(selection_event)[:200]}...")
        print(f"🔍 CURRENT CHAPTER: {selected_chapter}")
        print(f"🔍 RIGHT STYLE (raw): '{right_style}'")
        
        # Check for session state comment data as alternative
        import hashlib
        component_id = f"sync_scroll_{hashlib.md5(f'text_display_{selected_chapter}_{right_style}'.encode()).hexdigest()[:8]}"
        session_key = f'comment_data_{component_id}'
        print(f"🔍 CHECKING SESSION STATE: key='{session_key}'")
        
        if session_key in st.session_state:
            print(f"🔍 SESSION STATE FOUND: {st.session_state[session_key]}")
            try:
                session_data = json.loads(st.session_state[session_key])
                print(f"🔍 SESSION DATA PARSED: {session_data}")
                selection_event = st.session_state[session_key]  # Use session state data
                # Clear the session state to prevent reprocessing
                del st.session_state[session_key]
                print("🔍 SESSION STATE: Cleared after processing")
            except Exception as e:
                print(f"❌ SESSION STATE ERROR: {e}")
        else:
            print("🔍 SESSION STATE: No data found")
        
        # NEW SIMPLIFIED APPROACH - Check what we actually get from the component
        print(f"🔍 COMPONENT RETURN TYPE: {type(selection_event)}")
        print(f"🔍 COMPONENT RETURN VALUE: {str(selection_event)[:100]}...")
        
        # Check if we EVER get string data from the component
        if isinstance(selection_event, str) and selection_event.strip():
            print("🔍 STRING DATA RECEIVED!")
            print(f"🔍 STRING CONTENT: {selection_event}")
        elif selection_event is None:
            print("🔍 NONE: Component returned None")
        else:
            print("🔍 OTHER TYPE: Component returned non-string, non-None value")
            print(f"🔍 IS DELTAGENERATOR: {type(selection_event).__name__ == 'DeltaGenerator'}")
            
        # NEW APPROACH: Use session state as event bus
        if selection_event and isinstance(selection_event, str):
            print("🔍 STORING EVENT: Moving event to session state for safe processing")
            st.session_state['last_comment_event'] = selection_event
            st.rerun()
        else:
            print("🔍 NO STRING EVENT: Component did not return string data")
        
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
        
        # Sidebar-based commenting interface  
        st.divider()
        st.subheader("💬 Inline Comments")
        st.caption("📖 **How to use**: Select any text in the right panel above. The comment form will appear in the sidebar for seamless commenting without interrupting your reading flow.")
        
        # Display existing comments
        from utils import load_inline_comments
        style_name_clean = right_style.replace("Custom: ", "") if right_style.startswith("Custom: ") else "official"
        
        print(f"🔍 DISPLAY COMMENTS: Loading comments for display")
        print(f"   ├─ Original style: '{right_style}'")
        print(f"   ├─ Cleaned style: '{style_name_clean}'")
        print(f"   └─ Chapter: {selected_chapter}")
        
        existing_comments = load_inline_comments(style_name_clean, str(selected_chapter))
        
        if existing_comments:
            with st.expander(f"📋 Existing Comments ({len(existing_comments)})", expanded=True):
                for i, comment in enumerate(existing_comments):
                    dimension_icon = {
                        # New dimensions (v2.0)
                        'vocabulary_complexity': '🧠',
                        'cultural_context': '🌏',
                        'prose_style': '✍️',
                        'creative_fidelity': '🎨',
                        # Legacy dimensions (v1.0)
                        'english_sophistication': '🎯',
                        'world_building': '🌍',
                        'emotional_impact': '💔',
                        'dialogue_naturalness': '💬'
                    }.get(comment.get('dimension', 'vocabulary_complexity'), '📝')
                    
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
    st.divider()
    st.subheader("📊 Quality Assessment")
    
    # Load existing human scores for this style and chapter
    existing_human_scores = load_human_scores(eval_style.get('name', 'Unknown Style'))
    chapter_scores = existing_human_scores.get(str(selected_chapter), {})
    
    with st.form(f"human_eval_{eval_style.get('name', 'Unknown Style')}_{selected_chapter}"):
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
            vocabulary_complexity = st.slider(
                "🧠 Vocabulary Complexity",
                1, 100,
                value=chapter_scores.get('vocabulary_complexity', chapter_scores.get('english_sophistication', 50)),
                help="1 means very simple English, most accessible to the widest audience while 100 means employing a rich and sophisticated vocabulary. You are highly encouraged to use niche, obscure, complex, nuanced, specific, or less common words to precisely capture the original text's meaning."
            )
            
            cultural_context = st.slider(
                "🌏 Cultural Context",
                1, 100,
                value=chapter_scores.get('cultural_context', chapter_scores.get('world_building', 50)),
                help="1 would mean finding the closest equivalent terms, phrases that are colloquial and relatable. 100 means preserving the cultural, mythological, historical context as much as possible, adding footnotes to explain the quirks of the culture and language that allow such etymologies, connections and meaning."
            )
        
        with col2:
            prose_style = st.slider(
                "✍️ Prose Style",
                1, 100,
                value=chapter_scores.get('prose_style', chapter_scores.get('emotional_impact', 50)),
                help="1 would be concise, brief and rapid pacing of narrative flow, compressed and rapid. 100 would be extremely flowery, descriptive, of people, places and fight scenes, slow and immersive."
            )
            
            creative_fidelity = st.slider(
                "🎨 Creative Fidelity",
                1, 100,
                value=chapter_scores.get('creative_fidelity', chapter_scores.get('dialogue_naturalness', 50)),
                help="1 would mean careful translation that fully adheres to the literal meaning, while 100 would mean your goal is to intuit the author's intent, tone, and evoke the same emotional impact in the reader, you may rephrase sentences and take creative liberty to create a more compelling experience."
            )
        
        # Enhanced: Text Justifications Section (Optional for backward compatibility)
        st.divider()
        with st.expander("📝 Detailed Justifications (Optional - For Future AI Features)", expanded=False):
            st.caption("💡 **Future Feature Preview**: These justifications will help train AI to understand your quality standards and extrapolate scores to hundreds of chapters automatically.")
            
            # Get existing justifications if they exist
            existing_justifications = chapter_scores.get('justifications', {})
            
            justification_vocabulary = st.text_area(
                "🧠 Vocabulary Complexity Justification",
                value=existing_justifications.get('vocabulary_complexity', existing_justifications.get('english_sophistication', '')),
                placeholder="Explain your reasoning for the Vocabulary Complexity score. What specific word choices, technical terms, or linguistic sophistication influenced your rating?",
                help="Detailed explanation of vocabulary complexity assessment",
                height=80
            )
            
            justification_cultural = st.text_area(
                "🌏 Cultural Context Justification",
                value=existing_justifications.get('cultural_context', existing_justifications.get('world_building', '')),
                placeholder="Describe how well the translation preserves cultural, historical, and mythological context versus adapting for modern readers.",
                help="Explanation of cultural preservation vs adaptation",
                height=80
            )
            
            justification_prose = st.text_area(
                "✍️ Prose Style Justification",
                value=existing_justifications.get('prose_style', existing_justifications.get('emotional_impact', '')),
                placeholder="Assess the descriptive richness and pacing. Is the style concise and rapid, or flowery and immersive?",
                help="Explanation of prose style and descriptive approach",
                height=80
            )
            
            justification_creative = st.text_area(
                "🎨 Creative Fidelity Justification",
                value=existing_justifications.get('creative_fidelity', existing_justifications.get('dialogue_naturalness', '')),
                placeholder="Evaluate the balance between literal accuracy and creative interpretation for emotional impact and readability.",
                help="Explanation of creative translation choices",
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
            
            # Core evaluation data (always saved) - new v2.0 dimensions
            evaluation_data = {
                'vocabulary_complexity': vocabulary_complexity,
                'cultural_context': cultural_context,
                'prose_style': prose_style,
                'creative_fidelity': creative_fidelity,
                'evaluated_at': datetime.now().isoformat(),
                'evaluator': 'human'  # Keep for backward compatibility
            }
            
            # Enhanced data (only save if provided - backward compatible)
            if evaluator_name.strip():
                evaluation_data['evaluator_name'] = evaluator_name.strip()
            if evaluator_email.strip():
                evaluation_data['evaluator_email'] = evaluator_email.strip()
            
            # Text justifications (only save non-empty ones) - new v2.0 dimensions
            justifications = {}
            if justification_vocabulary.strip():
                justifications['vocabulary_complexity'] = justification_vocabulary.strip()
            if justification_cultural.strip():
                justifications['cultural_context'] = justification_cultural.strip()
            if justification_prose.strip():
                justifications['prose_style'] = justification_prose.strip()
            if justification_creative.strip():
                justifications['creative_fidelity'] = justification_creative.strip()
            
            if justifications:  # Only add if at least one justification provided
                evaluation_data['justifications'] = justifications
            
            if overall_notes.strip():
                evaluation_data['overall_notes'] = overall_notes.strip()
            
            # Add evaluation version for future tracking - always v3.0 for new dimensions
            evaluation_data['evaluation_version'] = '3.0'  # New dimension system v3.0
            
            existing_human_scores[str(selected_chapter)].update(evaluation_data)
            
            # Save to file
            save_human_scores(eval_style.get('name', 'Unknown Style'), existing_human_scores)
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
            
            # Show justifications if available (supports both v2.0 and v3.0 dimensions)
            if chapter_scores.get('justifications'):
                justifications = chapter_scores['justifications']
                st.write("**Detailed Justifications:**")
                
                # New v3.0 dimensions
                if justifications.get('vocabulary_complexity'):
                    st.info(f"🧠 **Vocabulary Complexity**: {justifications['vocabulary_complexity']}")
                if justifications.get('cultural_context'):
                    st.info(f"🌏 **Cultural Context**: {justifications['cultural_context']}")
                if justifications.get('prose_style'):
                    st.info(f"✍️ **Prose Style**: {justifications['prose_style']}")
                if justifications.get('creative_fidelity'):
                    st.info(f"🎨 **Creative Fidelity**: {justifications['creative_fidelity']}")
                
                # Legacy v1.0/v2.0 dimensions (for backwards compatibility)
                if justifications.get('english_sophistication'):
                    st.info(f"🎯 **English Sophistication (Legacy)**: {justifications['english_sophistication']}")
                if justifications.get('world_building'):
                    st.info(f"🌍 **World Building (Legacy)**: {justifications['world_building']}")
                if justifications.get('emotional_impact'):
                    st.info(f"💔 **Emotional Impact (Legacy)**: {justifications['emotional_impact']}")
                if justifications.get('dialogue_naturalness'):
                    st.info(f"💬 **Dialogue Naturalness (Legacy)**: {justifications['dialogue_naturalness']}")
            
            # Show overall notes if available
            if chapter_scores.get('overall_notes'):
                st.write("**Overall Notes:**")
                st.info(f"📝 {chapter_scores['overall_notes']}")
    
    # Show evaluation summary
    if human_scores:
        st.divider()
        st.subheader("📈 Evaluation Summary")
        
        # Show average scores
        all_scores = list(human_scores.values())
        if all_scores:
            col1, col2, col3, col4 = st.columns(4)
            
            # Try new dimensions first, fall back to legacy for backwards compatibility
            with col1:
                avg_vocab = sum(s.get('vocabulary_complexity', s.get('english_sophistication', 0)) for s in all_scores) / len(all_scores)
                st.metric("🧠 Vocab Avg", f"{avg_vocab:.1f}")
            
            with col2:
                avg_cultural = sum(s.get('cultural_context', s.get('world_building', 0)) for s in all_scores) / len(all_scores)
                st.metric("🌏 Cultural Avg", f"{avg_cultural:.1f}")
            
            with col3:
                avg_prose = sum(s.get('prose_style', s.get('emotional_impact', 0)) for s in all_scores) / len(all_scores)
                st.metric("✍️ Prose Avg", f"{avg_prose:.1f}")
            
            with col4:
                avg_creative = sum(s.get('creative_fidelity', s.get('dialogue_naturalness', 0)) for s in all_scores) / len(all_scores)
                st.metric("🎨 Creative Avg", f"{avg_creative:.1f}")
            
            # Enhanced: Show evaluator information if available
            evaluators = set()
            enhanced_evaluations = 0
            with_justifications = 0
            
            for score in all_scores:
                if score.get('evaluator_name'):
                    evaluators.add(score.get('evaluator_name'))
                if score.get('evaluation_version') in ['2.0', '3.0']:
                    enhanced_evaluations += 1
                if score.get('justifications'):
                    with_justifications += 1
            
            if evaluators:
                st.caption(f"👥 **Evaluators:** {', '.join(sorted(evaluators))}")
            if enhanced_evaluations > 0:
                st.caption(f"📝 **Enhanced Evaluations:** {enhanced_evaluations}/{len(all_scores)} with detailed data")
            if with_justifications > 0:
                st.caption(f"💭 **With Justifications:** {with_justifications}/{len(all_scores)} chapters")


# ==============================================================================
#                              STANDALONE EPUB CREATOR
# ==============================================================================

st.divider()
st.header("📖 Standalone EPUB Creator")
st.caption("Create EPUB books from any folder containing chapter files - custom translations, merged styles, raw chapters, or any text content")

# Import necessary functions
from ebooklib import epub
import zipfile

def detect_novel_slug_from_alignment_maps():
    """Detect novel slug from available alignment maps."""
    try:
        alignment_maps = list_alignment_maps()
        
        # For now, default to 'way_of_the_devil' if available
        for slug, info in alignment_maps.items():
            if slug == 'way_of_the_devil':
                return slug
        
        # Return first available if way_of_the_devil not found
        if alignment_maps:
            return list(alignment_maps.keys())[0]
        
        # Fallback to way_of_the_devil for backward compatibility
        return 'way_of_the_devil'
    except Exception as e:
        print(f"Warning: Could not detect novel slug: {e}")
        return 'way_of_the_devil'

# Novel selection
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📚 Source Selection")
    
    # EPUB source options
    epub_source = st.selectbox(
        "📖 EPUB Source:",
        ["Custom Translations", "Raw Chapters", "Custom Folder", "Mixed Sources"],
        help="Choose what content to include in your EPUB"
    )
    
    if epub_source == "Custom Translations":
        # List available custom translation runs
        custom_runs_dir = os.path.join(DATA_DIR, "custom_translations")
        available_runs = []
        
        if os.path.exists(custom_runs_dir):
            for run_name in os.listdir(custom_runs_dir):
                run_path = os.path.join(custom_runs_dir, run_name)
                if os.path.isdir(run_path):
                    # Check if run has translated files
                    txt_files = [f for f in os.listdir(run_path) if f.endswith('-translated.txt')]
                    if txt_files:
                        # Get chapter range
                        chapter_nums = []
                        for f in txt_files:
                            try:
                                chapter_num = int(f.split('-')[1])
                                chapter_nums.append(chapter_num)
                            except (IndexError, ValueError):
                                continue
                        
                        if chapter_nums:
                            chapter_nums.sort()
                            chapter_range = f"Ch.{min(chapter_nums)}-{max(chapter_nums)}"
                            available_runs.append({
                                "name": run_name,
                                "path": run_path,
                                "files": len(txt_files),
                                "range": chapter_range
                            })
        
        if available_runs:
            run_options = [f"{run['name']} ({run['files']} chapters, {run['range']})" for run in available_runs]
            selected_run = st.selectbox("Translation Run:", run_options)
            
            # Get selected run info
            selected_run_index = run_options.index(selected_run)
            selected_source_info = available_runs[selected_run_index]
            
        else:
            st.warning("No custom translation runs found. Run some translations first.")
            selected_source_info = None
    
    elif epub_source == "Raw Chapters":
        # Select from novel directories
        try:
            available_novels = get_available_novels()
            if available_novels:
                novel_options = [f"{novel['title']} ({novel['slug']})" for novel in available_novels]
                selected_novel = st.selectbox("Novel:", novel_options)
                
                # Extract novel slug
                for novel in available_novels:
                    if f"{novel['title']} ({novel['slug']})" == selected_novel:
                        novel_slug = novel['slug']
                        break
                
                # Check for raw chapters
                raw_chapters_dir = get_novel_raw_chapters_dir(novel_slug)
                if os.path.exists(raw_chapters_dir):
                    chapter_files = [f for f in os.listdir(raw_chapters_dir) if f.endswith('.txt')]
                    if chapter_files:
                        selected_source_info = {
                            "name": f"{novel_slug}_raw_chapters",
                            "path": raw_chapters_dir,
                            "files": len(chapter_files),
                            "range": f"Ch.1-{len(chapter_files)}",
                            "type": "raw_chapters"
                        }
                    else:
                        st.warning(f"No raw chapters found for {novel_slug}")
                        selected_source_info = None
                else:
                    st.warning(f"Raw chapters directory not found for {novel_slug}")
                    selected_source_info = None
            else:
                st.warning("No novels configured")
                selected_source_info = None
        except Exception as e:
            st.error(f"Error loading novels: {e}")
            selected_source_info = None
    
    elif epub_source == "Custom Folder":
        # Allow user to specify any folder path
        st.info("💡 **Perfect for merged styles!** Point to any folder containing chapter files (.txt)")
        
        # Usage examples
        with st.expander("📚 Usage Examples"):
            st.markdown("""
            **Perfect for:**
            - 📝 **Merged translation styles** (combining different AI outputs)
            - 🔄 **Manually edited chapters** (post-processed translations)
            - 📁 **Custom chapter collections** (hand-picked chapters)
            - 🎯 **Curated content** (specific chapter ranges)
            
            **Supported file patterns:**
            - `Chapter-0001.txt`, `Chapter-0044-translated.txt`
            - `Ch001.txt`, `Ch44.txt`
            - `001.txt`, `044.txt`
            - Any `.txt` files with chapter content
            """)
        
        # Folder path input
        folder_path = st.text_input(
            "📁 Folder Path:",
            placeholder="e.g., /path/to/your/merged/chapters or data/my_custom_translation",
            help="Enter the full path to the folder containing your chapter files"
        )
        
        # Quick suggestions
        st.caption("💡 **Quick suggestions:**")
        suggestion_col1, suggestion_col2 = st.columns(2)
        with suggestion_col1:
            if st.button("📁 Browse Custom Translations", help="Use existing translation runs"):
                custom_runs_dir = os.path.join(DATA_DIR, "custom_translations")
                if os.path.exists(custom_runs_dir):
                    st.session_state['suggested_folder'] = custom_runs_dir
        with suggestion_col2:
            if st.button("📁 Browse Data Directory", help="Browse the data folder"):
                st.session_state['suggested_folder'] = DATA_DIR
        
        # Show suggested folder if available
        if 'suggested_folder' in st.session_state:
            suggested_path = st.session_state['suggested_folder']
            st.caption(f"💡 **Suggested:** `{suggested_path}`")
            if st.button("📋 Use Suggested Path"):
                folder_path = suggested_path
                st.session_state['folder_path'] = folder_path
                st.rerun()
        
        # Use session state folder path if available
        if 'folder_path' in st.session_state:
            folder_path = st.session_state['folder_path']
        
        if folder_path:
            # Expand relative paths
            if not os.path.isabs(folder_path):
                # Try relative to project root
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                folder_path = os.path.join(project_root, folder_path)
            
            if os.path.exists(folder_path) and os.path.isdir(folder_path):
                # Scan for chapter files
                chapter_files = []
                for filename in os.listdir(folder_path):
                    if filename.endswith('.txt'):
                        chapter_files.append(filename)
                
                if chapter_files:
                    # Try to extract chapter numbers for range display
                    chapter_nums = []
                    for f in chapter_files:
                        # Try different naming patterns
                        import re
                        # Pattern 1: Chapter-0001.txt, Chapter-0044-translated.txt
                        match = re.search(r'Chapter-(\d+)', f)
                        if match:
                            chapter_nums.append(int(match.group(1)))
                        # Pattern 2: Ch001.txt, Ch44.txt
                        elif re.search(r'Ch(\d+)', f):
                            match = re.search(r'Ch(\d+)', f)
                            chapter_nums.append(int(match.group(1)))
                        # Pattern 3: 001.txt, 44.txt
                        elif re.search(r'^(\d+)\.txt$', f):
                            match = re.search(r'^(\d+)\.txt$', f)
                            chapter_nums.append(int(match.group(1)))
                    
                    # Determine range
                    if chapter_nums:
                        chapter_nums.sort()
                        chapter_range = f"Ch.{min(chapter_nums)}-{max(chapter_nums)}"
                    else:
                        chapter_range = f"{len(chapter_files)} files"
                    
                    selected_source_info = {
                        "name": f"custom_folder_{os.path.basename(folder_path)}",
                        "path": folder_path,
                        "files": len(chapter_files),
                        "range": chapter_range,
                        "type": "custom_folder"
                    }
                    
                    st.success(f"✅ Found {len(chapter_files)} chapter files!")
                    
                    # Show preview of detected files
                    with st.expander("📄 Preview Chapter Files"):
                        for i, filename in enumerate(sorted(chapter_files)[:10]):  # Show first 10
                            st.text(f"• {filename}")
                        if len(chapter_files) > 10:
                            st.text(f"... and {len(chapter_files) - 10} more files")
                
                else:
                    st.warning("⚠️ No .txt files found in this folder")
                    selected_source_info = None
                    
            else:
                st.error("❌ Folder not found or invalid path")
                selected_source_info = None
                
        else:
            selected_source_info = None
    
    else:  # Mixed Sources
        st.info("Mixed sources feature coming soon!")
        selected_source_info = None

with col2:
    st.subheader("📊 EPUB Preview")
    if selected_source_info:
        st.metric("Source", selected_source_info["name"])
        st.metric("Chapters", selected_source_info["files"])
        st.metric("Range", selected_source_info["range"])
        
        # Estimate file size
        estimated_mb = selected_source_info["files"] * 0.05
        st.metric("Est. Size", f"{estimated_mb:.1f} MB")

# EPUB Creation
if selected_source_info:
    st.subheader("📖 EPUB Configuration")
    
    # EPUB metadata
    col1, col2 = st.columns(2)
    with col1:
        epub_title = st.text_input(
            "Book Title:", 
            f"Way of the Devil - {selected_source_info['name']}", 
            help="Title for the EPUB book"
        )
        epub_author = st.text_input("Author:", "Wang Yu", help="Original author name")
    
    with col2:
        epub_translator = st.text_input("Translator:", "AI Translation", help="Translator credit")
        
        # Novel selection for branding
        try:
            available_novels = get_available_novels()
            if available_novels:
                # Create options for novel selection
                novel_options = [f"{novel['title']} ({novel['slug']})" for novel in available_novels]
                selected_novel_display = st.selectbox(
                    "📚 Novel Config:",
                    novel_options,
                    help="Choose which novel configuration to use for branding and metadata"
                )
                
                # Extract novel slug from selection
                novel_slug = None
                for novel in available_novels:
                    if f"{novel['title']} ({novel['slug']})" == selected_novel_display:
                        novel_slug = novel['slug']
                        break
                
                if not novel_slug:
                    novel_slug = detect_novel_slug_from_alignment_maps()
                    st.caption(f"⚠️ Fallback to: {novel_slug}")
                else:
                    # Show configuration status
                    selected_novel_info = next(n for n in available_novels if n['slug'] == novel_slug)
                    if selected_novel_info['has_config']:
                        st.success(f"✅ Config loaded: {novel_slug}")
                    else:
                        st.warning(f"⚠️ No config found for {novel_slug}")
            else:
                novel_slug = detect_novel_slug_from_alignment_maps()
                st.caption(f"🔄 Auto-detected: {novel_slug}")
        except Exception as e:
            novel_slug = detect_novel_slug_from_alignment_maps()
            st.caption(f"⚠️ Error loading novels, using: {novel_slug}")
    
    # Create EPUB button
    if st.button("📖 Create EPUB Book", type="primary", use_container_width=True):
        with st.spinner("Creating EPUB book..."):
            # Create output directory
            epub_output_dir = os.path.join(DATA_DIR, "epub_exports")
            os.makedirs(epub_output_dir, exist_ok=True)
            
            # Generate output filename
            safe_title = "".join(c for c in epub_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            epub_filename = f"{safe_title.replace(' ', '_')}.epub"
            epub_output_path = os.path.join(epub_output_dir, epub_filename)
            
            # Create EPUB using the new builder module
            try:
                from utils.epub_builder import build_epub
                
                success, message = build_epub(
                    selected_source_info['path'],
                    epub_output_path,
                    title=epub_title,
                    author=epub_author,
                    translator=epub_translator,
                    novel_slug=novel_slug,
                    include_images=True
                )
                
                if success:
                    st.success(f"✅ **EPUB Created Successfully!**")
                    st.info(f"📁 **Location:** `{epub_output_path}`")
                    
                    # Provide download button
                    with open(epub_output_path, 'rb') as f:
                        epub_data = f.read()
                    
                    st.download_button(
                        label="📥 Download EPUB",
                        data=epub_data,
                        file_name=epub_filename,
                        mime="application/epub+zip",
                        use_container_width=True
                    )
                    
                    # Show file info
                    file_size_mb = len(epub_data) / 1024 / 1024
                    st.caption(f"📊 **File size:** {file_size_mb:.2f} MB | **Format:** EPUB 3.0")
                    
                else:
                    st.error(f"❌ **EPUB Creation Failed:** {message}")
                    
            except Exception as e:
                st.error(f"❌ **EPUB Creation Error:** {str(e)}")
                st.info("💡 Make sure you have completed some translations first or check that the source files exist.")

# --- Footer ---
st.divider()
st.caption("🧪 **Experimentation Lab** | Part of the Translation Framework Suite")