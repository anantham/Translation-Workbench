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
    GOOGLE_AI_AVAILABLE,
    load_api_config,
    get_config_value,
    show_config_status
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
class InsufficientHistoryError(Exception):
    """Raised when not enough history chapters are available for examples."""
    pass

def get_official_examples(alignment_map, current_chapter_num, count):
    """Get examples from official EPUB translations."""
    examples = []
    available_history = sorted([
        int(k) for k, v in alignment_map.items() 
        if int(k) < current_chapter_num and v.get("raw_file") and v.get("english_file")
    ], reverse=True)
    
    chapters_to_use = available_history[:count]
    
    for chapter_num in reversed(chapters_to_use):
        data = alignment_map.get(str(chapter_num))
        if data:
            raw_content = load_chapter_content(data["raw_file"])
            eng_content = load_chapter_content(data["english_file"])
            if raw_content and eng_content:
                examples.append({
                    "chapter": chapter_num,
                    "user": raw_content, 
                    "model": eng_content,
                    "source": "Official"
                })
    return examples

def get_custom_run_examples(alignment_map, current_chapter_num, count, custom_run_name):
    """Get examples from a specific custom translation run."""
    examples = []
    custom_run_dir = os.path.join(DATA_DIR, "custom_translations", custom_run_name)
    
    if not os.path.exists(custom_run_dir):
        return examples
    
    # Find available translated chapters in custom run
    available_chapters = []
    for filename in os.listdir(custom_run_dir):
        if filename.endswith('-translated.txt'):
            # Extract chapter number from filename like "Chapter-0695-translated.txt"
            try:
                chapter_num = int(filename.split('-')[1])
                if chapter_num < current_chapter_num:
                    available_chapters.append(chapter_num)
            except (IndexError, ValueError):
                continue
    
    # Sort by chapter number (most recent first) and take requested count
    available_chapters.sort(reverse=True)
    chapters_to_use = available_chapters[:count]
    
    # Load examples in chronological order
    for chapter_num in reversed(chapters_to_use):
        # Get raw Chinese content from alignment map
        chapter_data = alignment_map.get(str(chapter_num))
        if chapter_data and chapter_data.get("raw_file"):
            raw_content = load_chapter_content(chapter_data["raw_file"])
            
            # Get custom translation
            custom_file = os.path.join(custom_run_dir, f"Chapter-{chapter_num:04d}-translated.txt")
            if os.path.exists(custom_file):
                with open(custom_file, 'r', encoding='utf-8') as f:
                    custom_content = f.read().strip()
                
                if raw_content and custom_content:
                    examples.append({
                        "chapter": chapter_num,
                        "user": raw_content,
                        "model": custom_content,
                        "source": f"Custom ({custom_run_name})"
                    })
    
    return examples

def get_current_run_examples(alignment_map, current_chapter_num, count, current_run_dir):
    """Get examples from the current translation run (fresh translations)."""
    examples = []
    
    if not os.path.exists(current_run_dir):
        return examples
    
    # Find available translated chapters in current run
    available_chapters = []
    for filename in os.listdir(current_run_dir):
        if filename.endswith('-translated.txt'):
            try:
                chapter_num = int(filename.split('-')[1])
                if chapter_num < current_chapter_num:
                    available_chapters.append(chapter_num)
            except (IndexError, ValueError):
                continue
    
    # Sort by chapter number (most recent first) and take requested count
    available_chapters.sort(reverse=True)
    chapters_to_use = available_chapters[:count]
    
    # Load examples in chronological order
    for chapter_num in reversed(chapters_to_use):
        # Get raw Chinese content from alignment map
        chapter_data = alignment_map.get(str(chapter_num))
        if chapter_data and chapter_data.get("raw_file"):
            raw_content = load_chapter_content(chapter_data["raw_file"])
            
            # Get fresh translation
            fresh_file = os.path.join(current_run_dir, f"Chapter-{chapter_num:04d}-translated.txt")
            if os.path.exists(fresh_file):
                with open(fresh_file, 'r', encoding='utf-8') as f:
                    fresh_content = f.read().strip()
                
                if raw_content and fresh_content:
                    examples.append({
                        "chapter": chapter_num,
                        "user": raw_content,
                        "model": fresh_content,
                        "source": "Fresh (Current Run)"
                    })
    
    return examples

@st.cache_data
def get_smart_fallback_examples(alignment_map, current_chapter_num, count, selected_custom_run, current_run_dir):
    """
    Smart fallback system for getting translation examples.
    Priority: Official > Selected Custom > Current Run > Error
    """
    examples = []
    sources_used = []
    
    # Priority 1: Official translations
    official_examples = get_official_examples(alignment_map, current_chapter_num, count)
    examples.extend(official_examples)
    if official_examples:
        sources_used.append(f"Official ({len(official_examples)})")
    
    # Priority 2: Selected custom run (if specified and still need more)
    if len(examples) < count and selected_custom_run:
        remaining_count = count - len(examples)
        custom_examples = get_custom_run_examples(alignment_map, current_chapter_num, remaining_count, selected_custom_run)
        examples.extend(custom_examples)
        if custom_examples:
            sources_used.append(f"Custom ({len(custom_examples)})")
    
    # Priority 3: Current run (if still need more)
    if len(examples) < count:
        remaining_count = count - len(examples)
        fresh_examples = get_current_run_examples(alignment_map, current_chapter_num, remaining_count, current_run_dir)
        examples.extend(fresh_examples)
        if fresh_examples:
            sources_used.append(f"Fresh ({len(fresh_examples)})")
    
    # Error handling: insufficient examples
    if len(examples) == 0:
        raise InsufficientHistoryError(
            f"No history chapters found before chapter {current_chapter_num}. "
            f"Translation cannot proceed without context examples."
        )
    
    # Sort examples by chapter number (chronological order)
    examples.sort(key=lambda x: x["chapter"])
    
    return examples, sources_used

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

# API Configuration Status
api_key, api_source = load_api_config()
config_status = show_config_status()

if api_key:
    st.sidebar.success(config_status)
else:
    st.sidebar.error(config_status)
    st.sidebar.markdown("""
    **Setup Instructions:**
    1. **Option A (Recommended):** Set environment variable:
       ```bash
       export GEMINI_API_KEY="your-api-key-here"
       ```
    2. **Option B:** Create `config.json`:
       ```bash
       cp config.example.json config.json
       # Edit config.json with your API key
       ```
    3. Get your API key from: https://aistudio.google.com/app/apikey
    """)
    st.stop()

with st.sidebar.expander("🤖 Model & Prompt", expanded=True):
    # Get model from config with fallback
    model_name = get_config_value("default_model", "gemini-2.5-pro")
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
    
    # Get default from config
    default_history = get_config_value("default_history_count", 5)
    history_count = st.number_input(
        "History Chapters (C)", 
        min_value=0, 
        max_value=50, 
        value=default_history, 
        help="Number of preceding chapters to use as in-context examples"
    )

with st.sidebar.expander("📚 History Source", expanded=True):
    # Scan for available custom translation runs
    custom_runs_dir = os.path.join(DATA_DIR, "custom_translations")
    available_runs = ["Official Only"]
    
    if os.path.exists(custom_runs_dir):
        for run_name in os.listdir(custom_runs_dir):
            run_path = os.path.join(custom_runs_dir, run_name)
            if os.path.isdir(run_path):
                # Check if run has translated files
                txt_files = [f for f in os.listdir(run_path) if f.endswith('-translated.txt')]
                if txt_files:
                    # Get metadata if available
                    metadata_path = os.path.join(run_path, "job_metadata.json")
                    if os.path.exists(metadata_path):
                        try:
                            with open(metadata_path, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                            chapters_info = f"({len(txt_files)} chapters)"
                            available_runs.append(f"Custom: {run_name} {chapters_info}")
                        except:
                            available_runs.append(f"Custom: {run_name} ({len(txt_files)} chapters)")
                    else:
                        available_runs.append(f"Custom: {run_name} ({len(txt_files)} chapters)")
    
    selected_history_source = st.selectbox(
        "History Examples Source:",
        available_runs,
        help="Source for few-shot translation examples. Official = EPUB translations, Custom = your previous translation runs"
    )
    
    # Extract run name for custom selections
    selected_custom_run = None
    if selected_history_source.startswith("Custom: "):
        selected_custom_run = selected_history_source.split("Custom: ")[1].split(" (")[0]
    
    # Show source info
    if selected_history_source == "Official Only":
        st.info("📖 Using official EPUB translations as examples")
    else:
        st.info(f"🎨 Using custom translation style: `{selected_custom_run}`")

with st.sidebar.expander("📁 Output Settings", expanded=True):
    run_name = st.text_input(
        "Run Name / Style:", 
        f"run_{datetime.now().strftime('%Y%m%d_%H%M')}", 
        help="Unique name for this translation bundle"
    )
    
    # Add delay option for rate limiting with config default
    default_delay = get_config_value("api_delay", 1.0)
    api_delay = st.slider(
        "API Delay (seconds)", 
        min_value=0.5, 
        max_value=5.0, 
        value=default_delay, 
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

# Validation (API key already checked earlier with st.stop())
start_button_disabled = not (total_chapters > 0 and run_name.strip() and system_prompt.strip())

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
        "output_dir": output_dir,
        "selected_custom_run": selected_custom_run
    }

if start_button_disabled:
    missing_items = []
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
                try:
                    history_examples, sources_used = get_smart_fallback_examples(
                        alignment_map, 
                        chapter_num, 
                        params["history_count"],
                        params["selected_custom_run"],
                        params["output_dir"]
                    )
                    log_messages.append(f"  └─ Found {len(history_examples)} context examples from: {', '.join(sources_used)}")
                except InsufficientHistoryError as e:
                    log_messages.append(f"  └─ ❌ **INSUFFICIENT HISTORY:** {str(e)}")
                    log_messages.append("  └─ 🛑 **STOPPING TRANSLATION JOB**")
                    
                    # Update live log immediately
                    recent_logs = log_messages[-10:]
                    with log_area:
                        for msg in recent_logs:
                            st.write(msg)
                    
                    st.error("❌ **Insufficient Translation History**")
                    st.warning(str(e))
                    st.info("💡 **Solutions:** Use chapters with existing translations, reduce History Chapters count, or provide custom translation examples.")
                    
                    # Stop processing further chapters
                    break

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
                    # Check for quota exceeded error specifically
                    if "429" in str(error) and "quota" in str(error).lower():
                        log_messages.append(f"  └─ ❌ **QUOTA EXCEEDED:** {error}")
                        log_messages.append("  └─ 🛑 **STOPPING TRANSLATION JOB**")
                        
                        # Update live log immediately
                        recent_logs = log_messages[-10:]
                        with log_area:
                            for msg in recent_logs:
                                st.write(msg)
                        
                        # Show quota exceeded dialog
                        st.error("❌ **Google AI API Quota Exceeded**")
                        st.warning("Your API key has hit rate limits. Please wait or use a different API key.")
                        
                        with st.expander("🔑 **Enter New API Key to Continue**", expanded=True):
                            new_api_key = st.text_input(
                                "New Google AI API Key:", 
                                type="password",
                                help="Get a new key from https://aistudio.google.com/app/apikey"
                            )
                            
                            if st.button("🔄 Resume with New Key", type="primary"):
                                if new_api_key.strip():
                                    # Update the job parameters with new key
                                    st.session_state.job_params["api_key"] = new_api_key.strip()
                                    log_messages.append("🔑 **New API key provided - resuming job...**")
                                    st.rerun()
                                else:
                                    st.error("Please enter a valid API key")
                        
                        # Stop processing further chapters
                        break
                    else:
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