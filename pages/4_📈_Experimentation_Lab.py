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

# Load alignment map and validation data
alignment_map = load_alignment_map()

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
                    limit=500,  # Analyze up to 500 chapters
                    min_similarity=0.5,
                    max_chars=30000
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
    st.header("🏆 Style Leaderboard")
    
    # Aggregate all evaluation results (could be from multiple sessions)
    if hasattr(st.session_state, 'evaluation_results') and st.session_state.evaluation_results:
        results_df = pd.DataFrame(st.session_state.evaluation_results)
        
        # Calculate overall scores
        leaderboard_data = []
        
        for (model_type, model_name), group in results_df.groupby(['model_type', 'model_name']):
            avg_bleu = group['bleu_score'].mean()
            avg_semantic = group['semantic_similarity'].mean()
            avg_length_ratio = group['length_ratio'].mean()
            
            # Calculate composite score (weighted average)
            composite_score = (avg_bleu * 0.4) + (avg_semantic * 0.4) + (min(abs(avg_length_ratio - 1.0), 1.0) * 0.2)
            
            leaderboard_data.append({
                "Rank": 0,  # Will be set after sorting
                "Model": model_name,
                "Type": "🎯 Fine-tuned" if model_type == "fine_tuned" else f"🤖 Base ({group['n_shot'].iloc[0]}-shot)",
                "BLEU Score": f"{avg_bleu:.3f}",
                "Semantic Similarity": f"{avg_semantic:.3f}",
                "Length Ratio": f"{avg_length_ratio:.2f}",
                "Composite Score": f"{composite_score:.3f}",
                "Evaluations": len(group)
            })
        
        # Sort by composite score and assign ranks
        leaderboard_data.sort(key=lambda x: float(x["Composite Score"]), reverse=True)
        for i, entry in enumerate(leaderboard_data):
            entry["Rank"] = i + 1
        
        # Display leaderboard
        leaderboard_df = pd.DataFrame(leaderboard_data)
        
        # Style the leaderboard
        def style_rank(val):
            if val == 1:
                return "background-color: #FFD700"  # Gold
            elif val == 2:
                return "background-color: #C0C0C0"  # Silver
            elif val == 3:
                return "background-color: #CD7F32"  # Bronze
            return ""
        
        styled_leaderboard = leaderboard_df.style.applymap(style_rank, subset=['Rank'])
        st.dataframe(styled_leaderboard, use_container_width=True)
        
        # Champion model details
        if leaderboard_data:
            champion = leaderboard_data[0]
            st.success(f"🏆 **Current Champion:** {champion['Model']} ({champion['Type']}) with composite score {champion['Composite Score']}")
        
        # Performance trends (if we had historical data)
        st.subheader("📈 Performance Trends")
        
        # Create a performance radar chart for top models
        if len(leaderboard_data) >= 2:
            top_models = leaderboard_data[:3]  # Top 3 models
            
            categories = ['BLEU Score', 'Semantic Similarity', 'Length Accuracy']
            
            fig = go.Figure()
            
            for model in top_models:
                # Normalize scores for radar chart (0-1 scale)
                bleu_norm = float(model["BLEU Score"])
                semantic_norm = float(model["Semantic Similarity"])
                length_norm = 1.0 - min(abs(float(model["Length Ratio"]) - 1.0), 1.0)  # Closer to 1.0 is better
                
                fig.add_trace(go.Scatterpolar(
                    r=[bleu_norm, semantic_norm, length_norm],
                    theta=categories,
                    fill='toself',
                    name=f"{model['Model']} ({model['Type']})"
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="🎯 Top Models Performance Radar"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("🏆 Run evaluations to build the leaderboard")
        
        # Show example leaderboard
        st.subheader("📊 Example Leaderboard")
        example_data = pd.DataFrame([
            {"Rank": 1, "Model": "your-tuned-model-v1", "Type": "🎯 Fine-tuned", "BLEU Score": "0.872", "Semantic Similarity": "0.891", "Composite Score": "0.845"},
            {"Rank": 2, "Model": "gemini-1.5-pro-001", "Type": "🤖 Base (3-shot)", "BLEU Score": "0.834", "Semantic Similarity": "0.856", "Composite Score": "0.812"},
            {"Rank": 3, "Model": "gemini-1.5-flash-001", "Type": "🤖 Base (0-shot)", "BLEU Score": "0.789", "Semantic Similarity": "0.823", "Composite Score": "0.776"}
        ])
        st.dataframe(example_data, use_container_width=True)

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