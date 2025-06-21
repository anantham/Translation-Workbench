import streamlit as st
import json
import os
import time
from datetime import datetime
import google.generativeai as genai

# Import shared utilities
from utils import (
    load_alignment_map,
    load_chapter_content,
    get_text_stats,
    DATA_DIR,
    GOOGLE_AI_AVAILABLE
)

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="🧪 Pluralistic Translation Lab", page_icon="🧪")
st.title("🧪 Pluralistic Translation Lab")
st.caption("**Translation Framework Workbench** | Generate custom translation bundles using different models, prompts, and in-context learning strategies")

# Quick navigation and overview
with st.expander("🧪 **Pluralistic Translation Lab Overview** - Click to see how this works"):
    st.markdown("""
    ### 🎯 **Generate Multiple Translation Styles**
    
    **What This Does:**
    - Takes raw Chinese chapters and creates custom English translations
    - Uses **in-context learning** with your existing high-quality translations
    - Applies different **system prompts** for different translation styles
    - Saves complete translation bundles for comparison and analysis
    
    **Key Features:**
    - ✅ **Few-shot learning:** Uses previous chapters as examples
    - ✅ **Custom prompts:** Create literal, dynamic, or stylistic translations  
    - ✅ **Real-time progress:** Live logging and progress tracking
    - ✅ **Immediate persistence:** Each chapter saved instantly (no data loss)
    - ✅ **Organized output:** Named runs in dedicated directories
    
    **Example Workflow:**
    1. Create a "Literal" translation run with formal prompt
    2. Use that as context for a "Dynamic" translation run  
    3. Compare both styles in the Experimentation Lab
    4. Build an ever-improving translation feedback loop
    """)

# Check dependencies
if not GOOGLE_AI_AVAILABLE:
    st.error("❌ Google AI SDK not available. Please install: `pip install google-generativeai`")
    st.stop()

# --- API Functions ---
@st.cache_data
def get_in_context_examples(alignment_map, start_chapter, count):
    """Gets the historical data for the few-shot prompt."""
    examples = []
    # Find chapters before the start chapter that have aligned files
    available_history = sorted([
        int(k) for k, v in alignment_map.items() 
        if int(k) < start_chapter and v.get("raw_file") and v.get("english_file")
    ], reverse=True)
    
    chapters_to_use = available_history[:count]
    
    for chapter_num in reversed(chapters_to_use):  # Reverse back to chronological order
        data = alignment_map.get(str(chapter_num))
        if data:
            raw_content = load_chapter_content(data["raw_file"])
            eng_content = load_chapter_content(data["english_file"])
            if raw_content and eng_content:
                examples.append({
                    "chapter": chapter_num,
                    "user": raw_content, 
                    "model": eng_content
                })
    return examples

def generate_translation_with_history(api_key, model_name, system_prompt, history, current_raw_text):
    """Constructs the multi-turn prompt and calls the Gemini API."""
    try:
        # Configure API
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        
        # Build the prompt with system instruction + examples + current task
        user_prompt_parts = []
        
        # Add System Prompt (if provided)
        if system_prompt:
            user_prompt_parts.append(f"SYSTEM INSTRUCTION:\n{system_prompt}")
        
        # Add In-Context Examples (History)
        if history:
            user_prompt_parts.append("EXAMPLES OF HIGH-QUALITY TRANSLATIONS:")
            for i, example in enumerate(history, 1):
                user_prompt_parts.append(f"EXAMPLE {i} (Chapter {example['chapter']}):")
                user_prompt_parts.append(f"Chinese Input:\n{example['user']}")
                user_prompt_parts.append(f"English Output:\n{example['model']}")
        
        # Add the Final Task
        user_prompt_parts.append("NOW TRANSLATE THE FOLLOWING:")
        user_prompt_parts.append(f"Chinese Input:\n{current_raw_text}")
        user_prompt_parts.append("English Output:")
        
        final_prompt = "\n\n" + "="*50 + "\n\n".join(user_prompt_parts)
        
        # Make API call
        response = model.generate_content(final_prompt)
        return response.text, None
        
    except Exception as e:
        return None, str(e)

# --- UI Sidebar ---
st.sidebar.header("🔬 Experiment Controls")

# API Key input
api_key = st.sidebar.text_input("🔑 Gemini API Key:", type="password", help="Required for translation generation")

with st.sidebar.expander("🤖 Model & Prompt", expanded=True):
    # For MVP, we hardcode the model but this is where a dropdown would go
    model_name = "gemini-1.5-pro-latest"
    st.info(f"**Model:** `{model_name}`")
    
    # Prompt templates
    prompt_templates = {
        "Literal & Accurate": "You are a professional translator specializing in Chinese to English translation of web novels. Provide an accurate, literal translation that preserves the original meaning, structure, and cultural context. Maintain formal tone and precise terminology.",
        "Dynamic & Modern": "You are a skilled literary translator adapting Chinese web novels for Western readers. Create a flowing, engaging translation that captures the spirit and excitement of the original while using natural modern English. Prioritize readability and dramatic impact.",
        "Simplified & Clear": "You are translating Chinese web novels for young adult readers. Use simple, clear language that's easy to understand. Explain cultural concepts briefly when needed. Keep sentences shorter and vocabulary accessible.",
        "Custom": ""
    }
    
    selected_template = st.selectbox("Prompt Template:", list(prompt_templates.keys()))
    
    if selected_template == "Custom":
        system_prompt = st.text_area(
            "Custom System Prompt:",
            "",
            height=150,
            help="Write your own translation instruction"
        )
    else:
        system_prompt = st.text_area(
            f"{selected_template} Prompt:",
            prompt_templates[selected_template],
            height=150,
            help="You can edit this template or use it as-is"
        )

with st.sidebar.expander("🎯 Translation Task", expanded=True):
    col1, col2 = st.columns(2)
    start_chapter = col1.number_input("Start Chapter (A)", min_value=1, value=700, help="First chapter to translate")
    end_chapter = col2.number_input("End Chapter (B)", min_value=start_chapter, value=705, help="Last chapter to translate")
    
    history_count = st.slider(
        "History Chapters (C)", 
        min_value=0, 
        max_value=10, 
        value=3, 
        help="Number of preceding chapters to use as in-context examples"
    )

with st.sidebar.expander("📁 Output Settings", expanded=True):
    run_name = st.text_input(
        "Run Name / Style:", 
        f"run_{datetime.now().strftime('%Y%m%d_%H%M')}", 
        help="Unique name for this translation bundle"
    )
    
    # Add delay option for rate limiting
    api_delay = st.slider(
        "API Delay (seconds)", 
        min_value=0.5, 
        max_value=5.0, 
        value=1.0, 
        step=0.5,
        help="Delay between API calls to avoid rate limiting"
    )

# --- Main Page ---
# Load alignment map
try:
    alignment_map = load_alignment_map("alignment_map.json")
except:
    alignment_map = {}

if not alignment_map:
    st.error("❌ `alignment_map.json` not found. Please run data curation on the **📖 Data Review & Alignment** page first.")
    st.stop()

# --- Pre-flight Check ---
st.header("✈️ Pre-flight Check")

# Find available chapters in the specified range
chapters_to_translate = []
missing_chapters = []

for i in range(start_chapter, end_chapter + 1):
    chapter_key = str(i)
    if chapter_key in alignment_map:
        chapter_data = alignment_map[chapter_key]
        if chapter_data.get("raw_file") and os.path.exists(chapter_data["raw_file"]):
            chapters_to_translate.append(chapter_key)
        else:
            missing_chapters.append(i)
    else:
        missing_chapters.append(i)

total_chapters = len(chapters_to_translate)

# Display pre-flight summary
col1, col2 = st.columns(2)

with col1:
    st.metric("Chapters to Translate", total_chapters)
    st.metric("Context Depth", f"{history_count} chapters")

with col2:
    st.metric("Total API Calls", total_chapters)
    estimated_time = total_chapters * (api_delay + 2)  # 2 seconds avg per API call
    st.metric("Estimated Time", f"{estimated_time/60:.1f} min")

# Show details
if missing_chapters:
    st.warning(f"⚠️ **Missing chapters:** {missing_chapters[:10]}{'...' if len(missing_chapters) > 10 else ''}")

if total_chapters > 0:
    st.success(f"✅ Ready to translate **{total_chapters} chapters** (Ch.{start_chapter} to Ch.{end_chapter})")
    
    # Output directory info
    output_dir = os.path.join(DATA_DIR, "custom_translations", run_name)
    st.info(f"📁 **Output:** `{output_dir}`")
    
    # Check for existing files
    if os.path.exists(output_dir):
        existing_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
        if existing_files:
            st.warning(f"🔄 **Resume Mode:** Found {len(existing_files)} existing files. Will skip completed chapters.")

# Validation
start_button_disabled = not (api_key and total_chapters > 0 and run_name.strip() and system_prompt.strip())

if st.button("🚀 Start Translation Job", disabled=start_button_disabled, type="primary", use_container_width=True):
    # Store job parameters in session state
    st.session_state.run_job = True
    st.session_state.job_params = {
        "run_name": run_name,
        "chapters_to_translate": chapters_to_translate,
        "history_count": history_count,
        "system_prompt": system_prompt,
        "model_name": model_name,
        "api_key": api_key,
        "api_delay": api_delay,
        "output_dir": output_dir
    }

if start_button_disabled:
    missing_items = []
    if not api_key: missing_items.append("API Key")
    if total_chapters == 0: missing_items.append("Valid chapter range")
    if not run_name.strip(): missing_items.append("Run name")
    if not system_prompt.strip(): missing_items.append("System prompt")
    st.error(f"❌ **Missing:** {', '.join(missing_items)}")

# --- Execution Logic ---
if st.session_state.get("run_job", False):
    st.header("⚙️ Running Translation Job...")
    
    # Get job parameters
    params = st.session_state.job_params
    
    # Create output directory
    os.makedirs(params["output_dir"], exist_ok=True)
    
    # Save job metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model_name": params["model_name"],
        "system_prompt": params["system_prompt"],
        "history_count": params["history_count"],
        "api_delay": params["api_delay"],
        "chapters_requested": params["chapters_to_translate"],
        "total_chapters": len(params["chapters_to_translate"])
    }
    
    with open(os.path.join(params["output_dir"], "job_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Setup UI elements for progress
    progress_bar = st.progress(0)
    status_placeholder = st.empty()
    
    # Live log area
    log_container = st.container()
    with log_container:
        st.subheader("📜 Live Translation Log")
        log_area = st.empty()
    
    log_messages = []
    successful_translations = 0
    failed_translations = 0

    total_chapters = len(params["chapters_to_translate"])
    
    for i, chapter_num_str in enumerate(params["chapters_to_translate"]):
        chapter_num = int(chapter_num_str)
        
        # Check if file already exists (resume capability)
        output_filename = f"Chapter-{chapter_num:04d}-translated.txt"
        output_path = os.path.join(params["output_dir"], output_filename)
        
        if os.path.exists(output_path):
            log_messages.append(f"📂 **Chapter {chapter_num}** - Already exists, skipping...")
            successful_translations += 1
        else:
            # Update status
            status_placeholder.info(f"🔄 **Processing Chapter {chapter_num}** ({i+1}/{total_chapters})")
            
            log_messages.append(f"🔄 **Processing Chapter {chapter_num}...**")
            
            try:
                # 1. Build Context
                log_messages.append("  └─ Building few-shot context...")
                history_examples = get_in_context_examples(alignment_map, chapter_num, params["history_count"])
                log_messages.append(f"  └─ Found {len(history_examples)} context examples")

                # 2. Get Raw Text
                log_messages.append("  └─ Loading raw chapter content...")
                raw_content = load_chapter_content(alignment_map[chapter_num_str]["raw_file"])
                
                if not raw_content:
                    raise Exception("Failed to load raw chapter content")

                # 3. Call API
                log_messages.append(f"  └─ Calling {params['model_name']} API...")
                translation, error = generate_translation_with_history(
                    params["api_key"], 
                    params["model_name"], 
                    params["system_prompt"], 
                    history_examples, 
                    raw_content
                )

                # 4. Process Result
                if error:
                    log_messages.append(f"  └─ ❌ **ERROR:** {error}")
                    failed_translations += 1
                else:
                    # Save translation
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(translation)
                    
                    # Get some stats
                    stats = get_text_stats(translation, 'english')
                    log_messages.append(f"  └─ ✅ **Success!** Saved ({stats['word_count']} words)")
                    successful_translations += 1
                    
            except Exception as e:
                log_messages.append(f"  └─ ❌ **EXCEPTION:** {str(e)}")
                failed_translations += 1
        
        # Update progress and log
        progress_bar.progress((i + 1) / total_chapters)
        
        # Update live log (show last 10 messages)
        recent_logs = log_messages[-10:]
        with log_area:
            for msg in recent_logs:
                st.write(msg)
        
        # Rate limiting delay
        if i < total_chapters - 1:  # Don't sleep after the last item
            time.sleep(params["api_delay"])

    # --- Job Complete ---
    status_placeholder.success("🎉 **Translation Job Complete!**")
    
    st.header("📊 Job Summary")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("✅ Successful", successful_translations)
    with col2:
        st.metric("❌ Failed", failed_translations)
    with col3:
        st.metric("📁 Total Files", successful_translations)
    
    if successful_translations > 0:
        st.success(f"🎉 **Success!** Your new translation bundle is saved in:\n`{params['output_dir']}`")
        
        # Show sample output
        sample_files = [f for f in os.listdir(params["output_dir"]) if f.endswith('.txt')]
        if sample_files:
            st.subheader("📖 Sample Output")
            sample_file = sample_files[0]
            sample_path = os.path.join(params["output_dir"], sample_file)
            with open(sample_path, "r", encoding="utf-8") as f:
                sample_content = f.read()[:500] + "..." if len(f.read()) > 500 else f.read()
            st.text_area(f"Preview: {sample_file}", sample_content, height=150)
    
    # Downloadable log
    full_log = "\n".join(log_messages)
    st.download_button(
        label="📥 Download Full Log",
        data=full_log,
        file_name=f"translation_log_{params['run_name']}.txt",
        mime="text/plain"
    )
    
    # Clear the run state
    if st.button("🔄 Start New Translation Job"):
        st.session_state.run_job = False
        st.rerun()

# --- Footer ---
st.divider()
st.caption("💡 **Tip:** Use different system prompts to create varied translation styles, then compare them in the **📈 Experimentation Lab**!")