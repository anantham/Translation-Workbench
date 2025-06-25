import streamlit as st
import json
import os
import time
from datetime import datetime
import google.generativeai as genai
from ebooklib import epub
import zipfile

# Import shared utilities
from utils import (
    load_alignment_map,
    load_chapter_content,
    get_text_stats,
    DATA_DIR,
    GOOGLE_AI_AVAILABLE,
    OPENAI_AVAILABLE,
    load_api_config,
    load_openai_api_config,
    get_config_value,
    get_static_gemini_models,
    get_available_openai_models,
    generate_translation_unified,
    get_all_available_prompts,
    save_custom_prompt,
    delete_custom_prompt,
    load_custom_prompts
)

# --- Helper Functions ---
def get_model_abbreviation(platform, model_name):
    """Generate short model abbreviation for run naming.
    
    Args:
        platform: "Gemini" or "OpenAI"
        model_name: Full model name
    
    Returns:
        str: Short abbreviation like "gem15p", "oai_gpt4o", "oai_BlJU60q"
    """
    if platform == "Gemini":
        if "gemini-1.5-pro" in model_name:
            return "gem15p"
        elif "gemini-1.5-flash" in model_name:
            return "gem15f"
        else:
            # Generic gemini abbreviation
            return "gemini"
    
    elif platform == "OpenAI":
        if model_name.startswith("ft:"):
            # Fine-tuned model: extract actual model ID
            # Format: ft:gpt-4o-mini:org:custom-name:BlJU60q
            parts = model_name.split(":")
            if len(parts) >= 5:
                model_id = parts[4]  # Get the actual model ID (e.g., BlJU60q)
                # Clean the model ID (should already be clean but ensure safety)
                clean_id = "".join(c for c in model_id if c.isalnum() or c in "-_")
                return f"oai_{clean_id}"
            elif len(parts) >= 4:
                # Fallback to custom name if model ID not available
                custom_name = parts[3]
                clean_name = "".join(c for c in custom_name if c.isalnum() or c in "-_")[:8]
                return f"oai_{clean_name}"
            else:
                return "oai_ft"
        elif "gpt-4o" in model_name:
            if "mini" in model_name:
                return "oai_gpt4m"
            else:
                return "oai_gpt4o"
        elif "gpt-4" in model_name:
            return "oai_gpt4"
        elif "gpt-3.5" in model_name:
            return "oai_gpt35"
        else:
            # Generic OpenAI abbreviation
            return "oai"
    
    # Fallback
    return platform.lower()[:3]

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

def create_epub_from_translations(translation_dir, output_path, book_title, author="Unknown", translator="AI Translation"):
    """Create an EPUB file from translated text files."""
    try:
        # Create EPUB book
        book = epub.EpubBook()
        
        # Set metadata
        book.set_identifier('way-of-devil-translation')
        book.set_title(book_title)
        book.set_language('en')
        book.add_author(author)
        book.add_metadata('DC', 'contributor', translator)
        
        # Add CSS style
        style = '''
        body { font-family: Times, serif; line-height: 1.6; margin: 2em; }
        h1 { text-align: center; margin-bottom: 2em; }
        p { text-indent: 2em; margin-bottom: 1em; }
        '''
        nav_css = epub.EpubItem(
            uid="nav_css",
            file_name="style/nav.css",
            media_type="text/css",
            content=style
        )
        book.add_item(nav_css)
        
        # Get all translation files
        translation_files = []
        for filename in os.listdir(translation_dir):
            if filename.endswith('-translated.txt'):
                try:
                    chapter_num = int(filename.split('-')[1])
                    translation_files.append((chapter_num, filename))
                except (IndexError, ValueError):
                    continue
        
        # Sort by chapter number
        translation_files.sort(key=lambda x: x[0])
        
        chapters = []
        spine = ['nav']
        
        # Create chapters
        for chapter_num, filename in translation_files:
            file_path = os.path.join(translation_dir, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Create chapter
            chapter_id = f"chapter_{chapter_num:04d}"
            chapter_title = f"Chapter {chapter_num}"
            
            # Format content as HTML
            paragraphs = content.split('\n\n')
            html_content = f'<h1>{chapter_title}</h1>\n'
            for paragraph in paragraphs:
                if paragraph.strip():
                    html_content += f'<p>{paragraph.strip()}</p>\n'
            
            chapter = epub.EpubHtml(
                title=chapter_title,
                file_name=f'{chapter_id}.xhtml',
                content=html_content
            )
            chapter.add_item(nav_css)
            
            book.add_item(chapter)
            chapters.append(chapter)
            spine.append(chapter)
        
        # Create table of contents
        book.toc = [(epub.Section('Chapters'), chapters)]
        
        # Add navigation
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        
        # Set spine
        book.spine = spine
        
        # Write EPUB
        epub.write_epub(output_path, book, {})
        
        return True, f"Successfully created EPUB with {len(chapters)} chapters"
        
    except Exception as e:
        return False, f"Error creating EPUB: {str(e)}"

# Removed old generate_translation_with_history function - replaced by unified function in utils.py

# --- UI Sidebar ---
st.sidebar.header("🔬 Experiment Controls")

# API Configuration - check availability but don't display status (shown on Home Dashboard)
api_key, api_source = load_api_config()

with st.sidebar.expander("🤖 Model & Prompt", expanded=True):
    # Platform selection
    platform_options = ["Gemini", "OpenAI"]
    selected_platform = st.selectbox("🌐 Platform:", platform_options, help="Choose AI platform")
    
    # API Key input based on platform
    if selected_platform == "Gemini":
        api_key, source = load_api_config()
        if not api_key:
            api_key = st.text_input("🔑 Gemini API Key:", type="password", help="Required for Gemini translation")
            if not api_key:
                st.warning("⚠️ Gemini API key required")
    elif selected_platform == "OpenAI":
        api_key, source = load_openai_api_config()
        if not api_key:
            api_key = st.text_input("🔑 OpenAI API Key:", type="password", help="Required for OpenAI translation")
            if not api_key:
                st.warning("⚠️ OpenAI API key required")
    
    # Model selection based on platform
    if selected_platform == "Gemini":
        available_models = get_static_gemini_models()
        default_model = "gemini-2.5-pro"
    elif selected_platform == "OpenAI":
        if api_key:
            # Try to fetch models from API
            with st.spinner("Loading available models..."):
                openai_models, error = get_available_openai_models(api_key)
                if error:
                    st.warning(f"Could not fetch models: {error}")
                    available_models = ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]
                else:
                    available_models = openai_models
        else:
            available_models = ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"]
        default_model = "gpt-4o-mini"
    
    # Model dropdown
    if available_models:
        try:
            default_index = available_models.index(default_model)
        except ValueError:
            default_index = 0
        
        model_name = st.selectbox(
            "🤖 Model:",
            available_models,
            index=default_index,
            help=f"Available {selected_platform} models"
        )
        
        # Show model info
        if model_name.startswith('ft:'):
            st.info(f"🎯 **Fine-tuned Model:** `{model_name}`")
        else:
            st.info(f"🤖 **Base Model:** `{model_name}`")
    else:
        st.error("❌ No models available")
        model_name = default_model
    
    # Load all available prompt templates (built-in + custom)
    prompt_templates = get_all_available_prompts()
    
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
    
    # Enhanced save prompt functionality
    if system_prompt.strip():
        with st.expander("💾 Save Current Prompt", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                template_name = st.text_input(
                    "Template Name:",
                    "",
                    help="Name for your custom prompt template",
                    placeholder="e.g., Poetic Style, Technical Translation"
                )
            
            with col2:
                st.metric("Prompt Length", f"{len(system_prompt)} chars")
            
            if st.button("💾 Save Template", disabled=not template_name.strip()):
                # Check if name already exists
                existing_prompts = load_custom_prompts()
                if template_name in existing_prompts:
                    st.warning(f"⚠️ Template '{template_name}' already exists. Choose a different name or delete the existing one first.")
                else:
                    # Save template using new system
                    success = save_custom_prompt(template_name, system_prompt)
                    if success:
                        st.success(f"✅ Saved template: `{template_name}`")
                        st.info("💡 Your new template is now available in the dropdown above!")
                        st.rerun()  # Refresh to show the new template
                    else:
                        st.error("❌ Failed to save template. Please try again.")
    
    # Custom prompt management section
    custom_prompts = load_custom_prompts()
    if custom_prompts:
        with st.expander("🗂️ Manage Custom Prompts", expanded=False):
            st.subheader("📋 Your Saved Prompts")
            
            for prompt_name, prompt_data in custom_prompts.items():
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    # Show prompt preview
                    preview = prompt_data["content"][:100] + "..." if len(prompt_data["content"]) > 100 else prompt_data["content"]
                    st.text(f"🎨 {prompt_name}")
                    st.caption(f"Preview: {preview}")
                
                with col2:
                    created_date = prompt_data.get("created", "Unknown")[:10]  # Just the date part
                    st.caption(f"Created: {created_date}")
                
                with col3:
                    if st.button(f"🗑️ Delete", key=f"delete_{prompt_name}"):
                        if delete_custom_prompt(prompt_name):
                            st.success(f"✅ Deleted '{prompt_name}'")
                            st.rerun()
                        else:
                            st.error("❌ Failed to delete")
            
            if len(custom_prompts) > 0:
                st.info(f"📊 Total custom prompts: {len(custom_prompts)}")
            else:
                st.info("📝 No custom prompts saved yet. Create one above!")
    
    # Store selected template name for run naming
    template_for_naming = selected_template
    if selected_template.startswith("🎨 "):
        template_for_naming = selected_template.split("🎨 ", 1)[1]  # Remove emoji prefix safely

with st.sidebar.expander("🎯 Translation Task", expanded=True):
    col1, col2 = st.columns(2)
    start_chapter = col1.number_input("Start Chapter (A)", min_value=1, value=700, help="First chapter to translate")
    end_chapter = col2.number_input("End Chapter (B)", min_value=start_chapter, value=max(705, start_chapter + 5), help="Last chapter to translate")
    
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
    # Initialize session state for run name if needed
    if 'run_name_base' not in st.session_state:
        # Generate base name with fixed timestamp for this session
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        if 'template_for_naming' in locals():
            if template_for_naming == "Custom":
                st.session_state.run_name_base = f"Custom_{timestamp}"
            else:
                # Clean template name for filename
                clean_template = template_for_naming.replace(" & ", "_").replace(" ", "_")
                st.session_state.run_name_base = f"{clean_template}_{timestamp}"
        else:
            st.session_state.run_name_base = f"run_{timestamp}"
    
    # Get model abbreviation
    model_abbrev = get_model_abbreviation(selected_platform, model_name)
    
    # Generate enhanced default name with model info
    enhanced_default = f"{st.session_state.run_name_base}_{model_abbrev}"
    
    # Initialize session state for full run name if needed
    if 'run_name' not in st.session_state:
        st.session_state.run_name = enhanced_default
    
    # Update session state if model changed
    if 'last_model_abbrev' not in st.session_state or st.session_state.last_model_abbrev != model_abbrev:
        # Model changed, remove old model abbreviation and add new one
        current_name = st.session_state.run_name
        
        # Remove the old model abbreviation if it exists
        if 'last_model_abbrev' in st.session_state:
            old_abbrev = st.session_state.last_model_abbrev
            if current_name.endswith(f"_{old_abbrev}"):
                # Remove the old model abbreviation completely
                base_without_model = current_name[:-len(f"_{old_abbrev}")]
                st.session_state.run_name = f"{base_without_model}_{model_abbrev}"
            else:
                # Fallback: rebuild from base
                st.session_state.run_name = enhanced_default
        else:
            # First time setting model, use enhanced default
            st.session_state.run_name = enhanced_default
        
        st.session_state.last_model_abbrev = model_abbrev
    
    # Run name input with reset option
    col1, col2 = st.columns([4, 1])
    with col1:
        run_name = st.text_input(
            "Run Name / Style:", 
            value=st.session_state.run_name,
            help="Unique name for this translation bundle (auto-suggested based on template + model)",
            key="run_name_input"
        )
    
    with col2:
        st.text("")  # Spacing
        if st.button("🔄 Reset", help="Reset to auto-generated name"):
            # Reset session state to regenerate with current model
            if 'run_name_base' in st.session_state:
                del st.session_state.run_name_base
            if 'run_name' in st.session_state:
                del st.session_state.run_name
            if 'last_model_abbrev' in st.session_state:
                del st.session_state.last_model_abbrev
            st.rerun()
    
    # Update session state when user edits the text input
    if run_name != st.session_state.run_name:
        st.session_state.run_name = run_name
    
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
        "selected_custom_run": selected_custom_run,
        "platform": selected_platform
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
                log_messages.append(f"  └─ Calling {params['platform']} {params['model_name']} API...")
                translation, error = generate_translation_unified(
                    params["api_key"], 
                    params["model_name"], 
                    params["system_prompt"], 
                    history_examples, 
                    raw_content,
                    params["platform"]
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
                        
                        # Platform-specific quota exceeded dialog
                        platform_name = params["platform"]
                        st.error(f"❌ **{platform_name} API Quota Exceeded**")
                        st.warning("Your API key has hit rate limits. Please wait or use a different API key.")
                        
                        with st.expander("🔑 **Enter New API Key to Continue**", expanded=True):
                            if platform_name == "Gemini":
                                help_text = "Get a new key from https://aistudio.google.com/app/apikey"
                            else:
                                help_text = "Get a new key from https://platform.openai.com/api-keys"
                                
                            new_api_key = st.text_input(
                                f"New {platform_name} API Key:", 
                                type="password",
                                help=help_text
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

# --- EPUB Creation Section ---
st.divider()
st.header("📖 EPUB Package Creator")
st.caption("Convert any custom translation run into a downloadable EPUB book")

# Scan for available custom translation runs
epub_col1, epub_col2 = st.columns([2, 1])

with epub_col1:
    custom_runs_dir = os.path.join(DATA_DIR, "custom_translations")
    available_epub_runs = []
    
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
                        available_epub_runs.append({
                            "name": run_name,
                            "path": run_path,
                            "files": len(txt_files),
                            "range": chapter_range
                        })

    if available_epub_runs:
        run_options = [f"{run['name']} ({run['files']} chapters, {run['range']})" for run in available_epub_runs]
        selected_epub_run = st.selectbox(
            "Select Translation Run:",
            run_options,
            help="Choose which custom translation run to package into EPUB"
        )
        
        # Extract selected run info
        selected_run_index = run_options.index(selected_epub_run)
        selected_run_info = available_epub_runs[selected_run_index]
        
        # EPUB metadata inputs
        epub_title = st.text_input(
            "Book Title:",
            f"Way of the Devil - {selected_run_info['name']} Translation",
            help="Title for the EPUB book"
        )
        
        epub_cols = st.columns(2)
        with epub_cols[0]:
            epub_author = st.text_input("Author:", "Wang Yu", help="Original author name")
        with epub_cols[1]:
            epub_translator = st.text_input("Translator:", "AI Translation", help="Translator credit")

with epub_col2:
    st.markdown("**📊 EPUB Preview**")
    if available_epub_runs and 'selected_run_info' in locals():
        st.metric("Chapters", selected_run_info['files'])
        st.metric("Range", selected_run_info['range'])
        
        # Estimate file size (rough calculation)
        estimated_mb = selected_run_info['files'] * 0.05  # ~50KB per chapter
        st.metric("Est. Size", f"{estimated_mb:.1f} MB")

# Create EPUB button
if available_epub_runs and 'selected_run_info' in locals():
    if st.button("📖 Create EPUB Book", type="primary", use_container_width=True):
        with st.spinner("Creating EPUB book..."):
            # Create output directory
            epub_output_dir = os.path.join(DATA_DIR, "epub_exports")
            os.makedirs(epub_output_dir, exist_ok=True)
            
            # Generate output filename
            safe_title = "".join(c for c in epub_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            epub_filename = f"{safe_title.replace(' ', '_')}.epub"
            epub_output_path = os.path.join(epub_output_dir, epub_filename)
            
            # Create EPUB
            success, message = create_epub_from_translations(
                selected_run_info['path'],
                epub_output_path,
                epub_title,
                epub_author,
                epub_translator
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
else:
    if not available_epub_runs:
        st.info("🔍 **No custom translation runs found.** Complete a translation job first to create EPUBs.")
    else:
        st.info("📖 Select a translation run above to create an EPUB book.")

# --- Footer ---
st.divider()
st.caption("💡 **Tip:** Use different system prompts to create varied translation styles, then compare them in the **📈 Experimentation Lab**!")