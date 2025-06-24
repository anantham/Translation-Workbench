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

# --- Sidebar: Experiment Configuration ---
st.sidebar.header("🎯 Experiment Setup")

# API Key
api_key = st.sidebar.text_input("🔑 Gemini API Key:", type="password")

if not api_key:
    st.sidebar.warning("🔑 API key required for experiments")

# Model Selection
st.sidebar.subheader("🤖 Model Selection")

# Base models for comparison
base_models = [
    "gemini-1.5-flash-001",
    "gemini-1.5-pro-001",
    "gemini-2.0-flash-exp"
]

selected_base_model = st.sidebar.selectbox("Base Model:", base_models)

# Fine-tuned models (would be populated from actual tuned models)
st.sidebar.subheader("🎯 Fine-tuned Models")
if api_key and GOOGLE_AI_AVAILABLE:
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        tuned_models, error = list_tuning_jobs(api_key)
        
        if tuned_models and not error:
            completed_models = [
                model for model in tuned_models 
                if getattr(model, 'state', '') == 'COMPLETED'
            ]
            
            if completed_models:
                model_names = [getattr(model, 'name', 'Unknown') for model in completed_models]
                # Format model names for display
                display_names = ["None"] + [name.split('/')[-1] if name != "None" else name for name in model_names]
                selected_tuned_model = st.sidebar.selectbox(
                    "Fine-tuned Model:", 
                    display_names
                )
            else:
                st.sidebar.info("📭 No completed fine-tuned models available")
                selected_tuned_model = "None"
        else:
            st.sidebar.warning("⚠️ Could not load fine-tuned models")
            selected_tuned_model = "None"
    except Exception as e:
        st.sidebar.error(f"❌ Error loading models: {e}")
        selected_tuned_model = "None"
else:
    selected_tuned_model = "None"

# In-context learning settings
st.sidebar.subheader("📚 In-Context Learning")
n_shot_examples = st.sidebar.slider("N-Shot Examples", min_value=0, max_value=10, value=3)

# Evaluation settings
st.sidebar.subheader("🎯 Evaluation Settings")
evaluation_chapters = st.sidebar.number_input(
    "Test Chapters", 
    min_value=1, 
    max_value=50, 
    value=10,
    help="Number of chapters to evaluate"
)

random_seed = st.sidebar.number_input("Random Seed", value=42, help="For reproducible results")

# --- Main Content ---

if not alignment_map:
    st.error("❌ Could not load alignment map")
    st.stop()

# Create tabs for different experiment types
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset Quality Analysis",
    "🔬 Quick Translation Test",
    "📊 Batch Evaluation", 
    "📈 Performance Analysis",
    "🏆 Style Leaderboard"
])

# --- Tab 1: Dataset Quality Analysis ---
with tab1:
    st.header("📊 Dataset Quality Analysis")
    st.caption("Analyze your training dataset quality and characteristics (moved from Fine-tuning Workbench)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔍 Dataset Analysis")
        
        if st.button("🔄 Analyze Dataset Quality", type="primary"):
            with st.spinner("Analyzing dataset..."):
                # Load and analyze training examples
                training_examples = load_dataset_for_tuning(
                    alignment_map, 
                    limit=min(max_available_chapters, 100)  # Analyze up to 100 chapters for experimentation
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
    
    # Dataset visualizations
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
            # BERT Similarity distribution
            if 'BERT_Similarity' in st.session_state.dataset_df.columns:
                bert_scores = st.session_state.dataset_df[st.session_state.dataset_df['BERT_Similarity'] != "N/A"]['BERT_Similarity']
                if len(bert_scores) > 0:
                    fig_bert = px.histogram(
                        x=bert_scores.astype(float), 
                        title="Distribution of BERT Similarity Scores",
                        nbins=20,
                        labels={'x': 'BERT Similarity', 'y': 'Count'}
                    )
                    fig_bert.add_vline(x=0.8, line_dash="dash", line_color="green", annotation_text="Good Threshold")
                    fig_bert.add_vline(x=0.6, line_dash="dash", line_color="orange", annotation_text="Acceptable")
                    fig_bert.update_layout(height=400)
                    st.plotly_chart(fig_bert, use_container_width=True)
                else:
                    st.info("📊 Run build_and_report.py to generate BERT similarity scores for visualization")
            else:
                st.info("📊 Run build_and_report.py to generate BERT similarity scores for visualization")
        
        # Show detailed table
        with st.expander("🔍 Detailed Dataset View"):
            st.dataframe(st.session_state.dataset_df, use_container_width=True)

# --- Tab 2: Quick Translation Test ---
with tab2:
    st.header("🔬 Quick Translation Test")
    st.caption("Compare different models on a single chapter")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📖 Chapter Selection")
        
        # Chapter selector
        available_chapters = sorted([int(k) for k in alignment_map.keys()])
        selected_chapter = st.selectbox(
            "Test Chapter:", 
            available_chapters,
            index=random.randint(0, min(50, len(available_chapters)-1))
        )
        
        # Load chapter content
        chapter_data = alignment_map[str(selected_chapter)]
        raw_content = load_chapter_content(chapter_data.get('raw_file', ''))
        official_translation = load_chapter_content(chapter_data.get('english_file', ''))
        
        if raw_content and "File not found" not in raw_content:
            # Show chapter stats
            raw_stats = get_text_stats(raw_content, 'chinese')
            official_stats = get_text_stats(official_translation, 'english')
            
            st.metric("📜 Chinese Characters", f"{raw_stats['char_count']:,}")
            st.metric("📖 Official Words", f"{official_stats['word_count']:,}")
            
            # Show preview
            with st.expander("👀 Preview Chinese Text"):
                st.text_area("Chinese Content:", raw_content[:500] + "...", height=150, disabled=True)
            
            with st.expander("👀 Preview Official Translation"):
                st.text_area("Official Translation:", official_translation[:500] + "...", height=150, disabled=True)
        else:
            st.error("❌ Could not load chapter content")
    
    with col2:
        st.subheader("🚀 Translation Comparison")
        
        if api_key and raw_content and "File not found" not in raw_content:
            # Translation buttons
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("🤖 Translate with Base Model", type="primary"):
                    with st.spinner(f"Translating with {selected_base_model}..."):
                        # Create in-context learning prompt if n-shot > 0
                        if n_shot_examples > 0:
                            # Get random examples for in-context learning
                            example_chapters = random.sample(
                                [ch for ch in available_chapters if ch != selected_chapter], 
                                min(n_shot_examples, len(available_chapters)-1)
                            )
                            
                            prompt_parts = ["You are a professional translator. Here are some examples:"]
                            
                            for ex_ch in example_chapters:
                                ex_data = alignment_map[str(ex_ch)]
                                ex_raw = load_chapter_content(ex_data.get('raw_file', ''))
                                ex_eng = load_chapter_content(ex_data.get('english_file', ''))
                                
                                if ex_raw and ex_eng and "File not found" not in ex_raw:
                                    # Use first 200 chars for examples
                                    prompt_parts.append(f"\nChinese: {ex_raw[:200]}...")
                                    prompt_parts.append(f"English: {ex_eng[:200]}...")
                            
                            prompt_parts.append(f"\nNow translate this Chinese text to English:\n{raw_content}")
                            full_prompt = "\n".join(prompt_parts)
                        else:
                            full_prompt = f"Translate this Chinese web novel chapter to English:\n\n{raw_content}"
                        
                        base_translation = translate_with_gemini(full_prompt, api_key, use_cache=False)
                        st.session_state.base_translation = base_translation
                        st.session_state.base_model_used = selected_base_model
            
            with col_b:
                if selected_tuned_model != "None":
                    if st.button("🎯 Translate with Fine-tuned Model", type="primary"):
                        with st.spinner("Translating with fine-tuned model..."):
                            # For fine-tuned models, use simpler prompt
                            tuned_translation = translate_with_gemini(
                                f"Translate this Chinese text to English:\n\n{raw_content}",
                                api_key,
                                use_cache=False
                            )
                            st.session_state.tuned_translation = tuned_translation
                            st.session_state.tuned_model_used = selected_tuned_model
                else:
                    st.info("🎯 No fine-tuned model selected")
            
            # Display translations
            if hasattr(st.session_state, 'base_translation') or hasattr(st.session_state, 'tuned_translation'):
                st.subheader("📊 Translation Results")
                
                # Create comparison table
                results = []
                
                # Official translation
                if official_translation and "File not found" not in official_translation:
                    official_stats = get_text_stats(official_translation, 'english')
                    results.append({
                        "Model": "📖 Official Translation",
                        "Content": official_translation[:200] + "...",
                        "Word Count": official_stats['word_count'],
                        "Character Count": official_stats['char_count']
                    })
                
                # Base model translation
                if hasattr(st.session_state, 'base_translation'):
                    base_stats = get_text_stats(st.session_state.base_translation, 'english')
                    n_shot_label = f" ({n_shot_examples}-shot)" if n_shot_examples > 0 else " (0-shot)"
                    results.append({
                        "Model": f"🤖 {st.session_state.base_model_used}{n_shot_label}",
                        "Content": st.session_state.base_translation[:200] + "...",
                        "Word Count": base_stats['word_count'],
                        "Character Count": base_stats['char_count']
                    })
                
                # Tuned model translation
                if hasattr(st.session_state, 'tuned_translation'):
                    tuned_stats = get_text_stats(st.session_state.tuned_translation, 'english')
                    results.append({
                        "Model": f"🎯 {st.session_state.tuned_model_used.split('/')[-1]}",
                        "Content": st.session_state.tuned_translation[:200] + "...",
                        "Word Count": tuned_stats['word_count'],
                        "Character Count": tuned_stats['char_count']
                    })
                
                # Display results
                if results:
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Quality analysis
                    if len(results) > 1 and official_translation:
                        st.subheader("🎯 Quality Analysis")
                        
                        # Load semantic model for similarity calculation
                        semantic_model = load_semantic_model()
                        
                        quality_results = []
                        
                        for result in results[1:]:  # Skip official translation
                            if "base_translation" in st.session_state and "Base" in result["Model"]:
                                translation = st.session_state.base_translation
                            elif "tuned_translation" in st.session_state and "tuned" in result["Model"]:
                                translation = st.session_state.tuned_translation
                            else:
                                continue
                            
                            # Calculate quality metrics
                            evaluation = evaluate_translation_quality(
                                raw_content, 
                                official_translation, 
                                translation, 
                                semantic_model
                            )
                            
                            quality_results.append({
                                "Model": result["Model"],
                                "BLEU Score": f"{evaluation['bleu_score']:.3f}",
                                "Semantic Similarity": f"{evaluation['semantic_similarity']:.3f}",
                                "Length Ratio": f"{evaluation['length_ratio']:.2f}"
                            })
                        
                        if quality_results:
                            quality_df = pd.DataFrame(quality_results)
                            st.dataframe(quality_df, use_container_width=True)
                            
                            # Highlight best performer
                            if len(quality_results) > 1:
                                best_bleu = max(float(r["BLEU Score"]) for r in quality_results)
                                best_similarity = max(float(r["Semantic Similarity"]) for r in quality_results)
                                
                                for result in quality_results:
                                    if (float(result["BLEU Score"]) == best_bleu or 
                                        float(result["Semantic Similarity"]) == best_similarity):
                                        st.success(f"🏆 Best performer: {result['Model']}")
                                        break
        else:
            if not api_key:
                st.warning("🔑 API key required for translation")
            else:
                st.warning("⚠️ Chapter content not available")

# --- Tab 3: Batch Evaluation ---
with tab3:
    st.header("📊 Batch Evaluation")
    st.caption("Evaluate models on multiple chapters for statistical significance")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 Evaluation Setup")
        
        # Evaluation parameters
        st.write("**📊 Evaluation Parameters:**")
        st.info(f"""
        • **Test Chapters:** {evaluation_chapters}
        • **Random Seed:** {random_seed}
        • **Base Model:** {selected_base_model}
        • **N-Shot Examples:** {n_shot_examples}
        • **Fine-tuned Model:** {selected_tuned_model.split('/')[-1] if selected_tuned_model != "None" else "None"}
        """)
        
        # Start evaluation
        if api_key and st.button("🚀 Start Batch Evaluation", type="primary"):
            # Set random seed for reproducibility
            random.seed(random_seed)
            
            # Select random chapters for evaluation
            available_chapters = [
                ch for ch in sorted([int(k) for k in alignment_map.keys()])
                if alignment_map[str(ch)].get('raw_file') and alignment_map[str(ch)].get('english_file')
            ]
            
            if len(available_chapters) < evaluation_chapters:
                st.error(f"❌ Only {len(available_chapters)} chapters available, requested {evaluation_chapters}")
            else:
                test_chapters = random.sample(available_chapters, evaluation_chapters)
                
                st.session_state.evaluation_in_progress = True
                st.session_state.test_chapters = test_chapters
                st.session_state.evaluation_results = []
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                semantic_model = load_semantic_model()
                
                for i, chapter_num in enumerate(test_chapters):
                    # Update progress
                    progress = (i + 1) / len(test_chapters)
                    progress_bar.progress(progress)
                    status_text.text(f"Evaluating Chapter {chapter_num}... ({i+1}/{len(test_chapters)})")
                    
                    # Load chapter data
                    chapter_data = alignment_map[str(chapter_num)]
                    raw_content = load_chapter_content(chapter_data['raw_file'])
                    official_translation = load_chapter_content(chapter_data['english_file'])
                    
                    if "File not found" in raw_content or "File not found" in official_translation:
                        continue
                    
                    # Get base model translation (with n-shot if specified)
                    base_translation = None
                    if n_shot_examples > 0:
                        # Create n-shot prompt
                        example_chapters = random.sample(
                            [ch for ch in available_chapters if ch != chapter_num], 
                            min(n_shot_examples, len(available_chapters)-1)
                        )
                        
                        prompt_parts = ["You are a professional translator. Here are examples:"]
                        for ex_ch in example_chapters:
                            ex_data = alignment_map[str(ex_ch)]
                            ex_raw = load_chapter_content(ex_data.get('raw_file', ''))
                            ex_eng = load_chapter_content(ex_data.get('english_file', ''))
                            if ex_raw and ex_eng and "File not found" not in ex_raw:
                                prompt_parts.append(f"\nChinese: {ex_raw[:200]}...")
                                prompt_parts.append(f"English: {ex_eng[:200]}...")
                        
                        prompt_parts.append(f"\nNow translate:\n{raw_content}")
                        base_translation = translate_with_gemini("\n".join(prompt_parts), api_key, use_cache=False)
                    else:
                        base_translation = translate_with_gemini(f"Translate this Chinese text to English:\n\n{raw_content}", api_key, use_cache=False)
                    
                    # Evaluate base model
                    if base_translation and "API Request Failed" not in base_translation:
                        base_evaluation = evaluate_translation_quality(
                            raw_content, official_translation, base_translation, semantic_model
                        )
                        
                        result = {
                            "chapter": chapter_num,
                            "model_type": "base",
                            "model_name": selected_base_model,
                            "n_shot": n_shot_examples,
                            "bleu_score": base_evaluation['bleu_score'],
                            "semantic_similarity": base_evaluation['semantic_similarity'],
                            "length_ratio": base_evaluation['length_ratio'],
                            "translation": base_translation[:200] + "..."
                        }
                        
                        st.session_state.evaluation_results.append(result)
                    
                    # Get fine-tuned model translation if available
                    if selected_tuned_model != "None":
                        tuned_translation = translate_with_gemini(
                            f"Translate this Chinese text to English:\n\n{raw_content}",
                            api_key, 
                            use_cache=False
                        )
                        
                        if tuned_translation and "API Request Failed" not in tuned_translation:
                            tuned_evaluation = evaluate_translation_quality(
                                raw_content, official_translation, tuned_translation, semantic_model
                            )
                            
                            result = {
                                "chapter": chapter_num,
                                "model_type": "fine_tuned",
                                "model_name": selected_tuned_model.split('/')[-1],
                                "n_shot": 0,
                                "bleu_score": tuned_evaluation['bleu_score'],
                                "semantic_similarity": tuned_evaluation['semantic_similarity'],
                                "length_ratio": tuned_evaluation['length_ratio'],
                                "translation": tuned_translation[:200] + "..."
                            }
                            
                            st.session_state.evaluation_results.append(result)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                st.session_state.evaluation_in_progress = False
                st.success(f"✅ Evaluation complete! Tested {len(test_chapters)} chapters")
    
    with col2:
        st.subheader("📈 Evaluation Results")
        
        if hasattr(st.session_state, 'evaluation_results') and st.session_state.evaluation_results:
            results_df = pd.DataFrame(st.session_state.evaluation_results)
            
            # Summary statistics
            st.write("**📊 Summary Statistics:**")
            
            summary_stats = results_df.groupby(['model_type', 'model_name']).agg({
                'bleu_score': ['mean', 'std'],
                'semantic_similarity': ['mean', 'std'],
                'length_ratio': ['mean', 'std']
            }).round(4)
            
            st.dataframe(summary_stats, use_container_width=True)
            
            # Detailed results
            with st.expander("🔍 Detailed Results"):
                st.dataframe(results_df, use_container_width=True)
            
            # Export results
            if st.button("💾 Export Results"):
                export_data = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=export_data,
                    file_name=f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("📊 Run batch evaluation to see results here")

# --- Tab 4: Performance Analysis ---
with tab4:
    st.header("📈 Performance Analysis")
    
    if hasattr(st.session_state, 'evaluation_results') and st.session_state.evaluation_results:
        results_df = pd.DataFrame(st.session_state.evaluation_results)
        
        # Performance comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # BLEU score comparison
            fig_bleu = px.box(
                results_df,
                x='model_name',
                y='bleu_score',
                color='model_type',
                title="📊 BLEU Score Distribution by Model"
            )
            fig_bleu.update_layout(height=400)
            st.plotly_chart(fig_bleu, use_container_width=True)
        
        with col2:
            # Semantic similarity comparison
            fig_semantic = px.box(
                results_df,
                x='model_name',
                y='semantic_similarity',
                color='model_type',
                title="🧠 Semantic Similarity Distribution by Model"
            )
            fig_semantic.update_layout(height=400)
            st.plotly_chart(fig_semantic, use_container_width=True)
        
        # Correlation analysis
        st.subheader("🔗 Metric Correlation Analysis")
        
        correlation_data = results_df[['bleu_score', 'semantic_similarity', 'length_ratio']].corr()
        
        fig_corr = px.imshow(
            correlation_data,
            text_auto=True,
            aspect="auto",
            title="📊 Correlation Matrix of Quality Metrics"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Statistical significance testing
        if len(results_df['model_name'].unique()) > 1:
            st.subheader("📈 Statistical Significance")
            
            try:
                from scipy import stats
                
                # Compare models on BLEU scores
                model_groups = results_df.groupby('model_name')['bleu_score'].apply(list)
                
                if len(model_groups) == 2:
                    group1, group2 = model_groups.iloc[0], model_groups.iloc[1]
                    t_stat, p_value = stats.ttest_ind(group1, group2)
                    
                    st.metric("T-statistic", f"{t_stat:.4f}")
                    st.metric("P-value", f"{p_value:.4f}")
                    
                    if p_value < 0.05:
                        st.success("✅ Statistically significant difference (p < 0.05)")
                    else:
                        st.info("📊 No statistically significant difference (p ≥ 0.05)")
                else:
                    st.info("📊 Statistical testing requires exactly 2 models")
                    
            except ImportError:
                st.info("📊 Install scipy for statistical significance testing")
    else:
        st.info("📈 Run batch evaluation first to see performance analysis")

# --- Tab 5: Style Leaderboard ---
with tab5:
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
                    
                    # Display content in columns
                    st.subheader(f"📖 Chapter {selected_chapter} Comparison")
                    
                    col_custom, col_official = st.columns(2)
                    
                    with col_custom:
                        st.markdown("**🎨 Custom Translation**")
                        st.text_area(
                            "Custom:", 
                            custom_translation[:2000] + "..." if len(custom_translation) > 2000 else custom_translation,
                            height=300,
                            disabled=True
                        )
                    
                    with col_official:
                        st.markdown("**📚 Official Translation**")
                        if official_translation and "File not found" not in official_translation:
                            st.text_area(
                                "Official:", 
                                official_translation[:2000] + "..." if len(official_translation) > 2000 else official_translation,
                                height=300,
                                disabled=True
                            )
                        else:
                            st.warning("Official translation not available for comparison")
                    
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