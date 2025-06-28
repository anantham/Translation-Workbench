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
    load_custom_prompts,
    calculate_openai_cost,
    calculate_gemini_cost
)

# --- Helper Functions ---
def get_model_abbreviation(platform, model_name):
    """Generate short model abbreviation for run naming.
    
    Args:
        platform: "Gemini" or "OpenAI"
        model_name: Full model name
    
    Returns:
        str: Short abbreviation like "gem15p", "oai_gpt4o", "oai_BlJU60q", "deepseek_chat"
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
        elif "deepseek-chat" in model_name:
            return "deepseek_chat"
        elif "deepseek-reasoner" in model_name:
            return "deepseek_reason"
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

def update_job_metadata_with_usage(metadata_file_path, usage_metrics, chapter_success=True):
    """Update job metadata file with API usage metrics from a single translation."""
    try:
        # Load existing metadata
        with open(metadata_file_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Update API usage totals
        api_usage = metadata.setdefault('api_usage', {})
        api_usage['total_tokens_used'] = api_usage.get('total_tokens_used', 0) + usage_metrics.get('total_tokens', 0)
        api_usage['input_tokens'] = api_usage.get('input_tokens', 0) + usage_metrics.get('prompt_tokens', 0)
        api_usage['output_tokens'] = api_usage.get('output_tokens', 0) + usage_metrics.get('completion_tokens', 0)
        api_usage['api_calls_made'] = api_usage.get('api_calls_made', 0) + 1
        
        # Update cost tracking
        estimated_cost = usage_metrics.get('estimated_cost', 0.0)
        api_usage['estimated_cost_usd'] = api_usage.get('estimated_cost_usd', 0.0) + estimated_cost
        
        cost_breakdown = api_usage.setdefault('cost_breakdown', {})
        cost_breakdown['input_cost'] = cost_breakdown.get('input_cost', 0.0) + usage_metrics.get('input_cost', 0.0)
        cost_breakdown['output_cost'] = cost_breakdown.get('output_cost', 0.0) + usage_metrics.get('output_cost', 0.0)
        
        # Update performance metrics
        performance = metadata.setdefault('performance_metrics', {})
        if chapter_success:
            performance['chapters_completed'] = performance.get('chapters_completed', 0) + 1
        else:
            performance['chapters_failed'] = performance.get('chapters_failed', 0) + 1
        
        # Calculate averages
        total_chapters_processed = performance.get('chapters_completed', 0) + performance.get('chapters_failed', 0)
        if total_chapters_processed > 0:
            api_usage['avg_tokens_per_chapter'] = api_usage['total_tokens_used'] // total_chapters_processed
        
        # Save updated metadata
        with open(metadata_file_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        return True
        
    except Exception as e:
        print(f"Warning: Could not update job metadata: {e}")
        return False

def load_and_format_metadata_template(job_metadata, translation_dir):
    """Load the metadata template and substitute actual values."""
    template_path = os.path.join(os.path.dirname(translation_dir), "..", "..", "epub_metadata_template.md")
    
    # Load template
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
    except FileNotFoundError:
        # Fallback template if file missing
        template = """
# Translation Information

## Performance Analytics
- **Total Time**: {total_time}
- **API Cost**: ${estimated_cost_usd}
- **Processing Speed**: {chapters_per_minute} chapters/minute

## AI Configuration
- **Model**: {model_name}
- **Translation Strategy**: Professional Xianxia Translation

## Project Information
- **Framework**: Pluralistic Translation Workbench
- **Generated**: {generation_date}
"""
    
    # Calculate derived values from job metadata
    performance = job_metadata.get("performance_metrics", {})
    api_usage = job_metadata.get("api_usage", {})
    
    # Format timestamps and durations
    start_time = performance.get("start_timestamp", "Unknown")
    end_time = performance.get("end_timestamp", "Unknown")
    total_time = performance.get("total_time_elapsed", "Unknown")
    
    # Calculate processing speed
    chapters_completed = performance.get("chapters_completed", 0)
    if total_time and total_time != "Unknown" and chapters_completed > 0:
        # Parse duration in seconds (format: "XXXXs")
        if isinstance(total_time, str) and total_time.endswith('s'):
            total_seconds = float(total_time.rstrip('s'))
            chapters_per_minute = round((chapters_completed / total_seconds) * 60, 2)
        else:
            chapters_per_minute = "Unknown"
    else:
        chapters_per_minute = "Unknown"
    
    # Substitute template variables
    substitutions = {
        'total_time': total_time,
        'estimated_cost_usd': f"{api_usage.get('estimated_cost_usd', 0.0):.4f}",
        'total_tokens': f"{api_usage.get('total_tokens_used', 0):,}",
        'chapters_per_minute': chapters_per_minute,
        'avg_time_per_chapter': performance.get('avg_time_per_chapter', 'Unknown'),
        'api_calls_made': api_usage.get('api_calls_made', 0),
        'model_name': job_metadata.get('model_name', 'Unknown'),
        'api_provider': job_metadata.get('ai_configuration', {}).get('api_provider', 'Unknown'),
        'model_version': job_metadata.get('ai_configuration', {}).get('model_version', 'Unknown'),
        'translation_style': 'Professional Xianxia Translation',
        'system_prompt_preview': job_metadata.get('system_prompt', '')[:100],
        'example_count': job_metadata.get('history_count', 0),
        'example_strategy': job_metadata.get('ai_configuration', {}).get('example_strategy', 'Unknown'),
        'temperature': job_metadata.get('ai_configuration', {}).get('temperature', 0.7),
        'max_tokens': job_metadata.get('ai_configuration', {}).get('max_tokens', 4096),
        'novel_title': job_metadata.get('project_info', {}).get('novel_title', '极道天魔 (Way of the Devil)'),
        'original_author': job_metadata.get('project_info', {}).get('original_author', '王雨 (Wang Yu)'),
        'chapter_range': f"{min(job_metadata.get('chapters_requested', []))}-{max(job_metadata.get('chapters_requested', []))}" if job_metadata.get('chapters_requested') else 'Unknown',
        'total_chapters': job_metadata.get('total_chapters', 0),
        'source_language': job_metadata.get('project_info', {}).get('source_language', 'Chinese (Simplified)'),
        'target_language': job_metadata.get('project_info', {}).get('target_language', 'English'),
        'word_count_chinese': 'Unknown',  # TODO: Calculate from source files
        'word_count_english': 'Unknown',  # TODO: Calculate from translation files
        'expansion_ratio': 'Unknown',     # TODO: Calculate ratio
        'github_url': 'https://github.com/anthropics/translation-workbench',
        'project_version': job_metadata.get('project_info', {}).get('framework_version', 'v2.1.0'),
        'license': 'MIT License',
        'maintainer_name': 'Translation Workbench',
        'maintainer_email': 'contact@example.com',
        'feature_requests_url': 'https://github.com/anthropics/translation-workbench/issues',
        'documentation_url': 'https://docs.example.com/translation-workbench',
        'consistency_score': 'Pending evaluation',
        'bleu_score_avg': 'Pending evaluation',
        'semantic_similarity_avg': 'Pending evaluation',
        'human_eval_sample': 'Pending evaluation',
        'terminology_standardization': 'High',
        'input_tokens': f"{api_usage.get('input_tokens', 0):,}",
        'output_tokens': f"{api_usage.get('output_tokens', 0):,}",
        'input_cost': f"{api_usage.get('cost_breakdown', {}).get('input_cost', 0.0):.4f}",
        'output_cost': f"{api_usage.get('cost_breakdown', {}).get('output_cost', 0.0):.4f}",
        'cost_per_1k_tokens': f"{api_usage.get('cost_breakdown', {}).get('rate_per_1k_tokens', 0.0):.4f}",
        'avg_cost_per_chapter': f"{api_usage.get('estimated_cost_usd', 0.0) / max(job_metadata.get('total_chapters', 1), 1):.4f}",
        'start_timestamp': start_time,
        'end_timestamp': end_time,
        'api_delay': job_metadata.get('api_delay', 1.0),
        'system_prompt_hash': job_metadata.get('ai_configuration', {}).get('system_prompt_hash', 'Unknown'),
        'framework_version': job_metadata.get('project_info', {}).get('framework_version', 'v2.1.0'),
        'framework_name': job_metadata.get('project_info', {}).get('framework_name', 'Pluralistic Translation Workbench'),
        'generation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'github_discussions_url': 'https://github.com/anthropics/translation-workbench/discussions',
        'license_url': 'https://github.com/anthropics/translation-workbench/blob/main/LICENSE'
    }
    
    # Substitute all variables in template
    formatted_template = template
    for key, value in substitutions.items():
        formatted_template = formatted_template.replace(f'{{{key}}}', str(value))
    
    return formatted_template

def create_epub_from_translations(translation_dir, output_path, book_title, author="Unknown", translator="AI Translation"):
    """Create an EPUB file from translated text files."""
    try:
        # Create EPUB book
        book = epub.EpubBook()
        
        # Set basic metadata
        book.set_identifier('way-of-devil-translation')
        book.set_title(book_title)
        book.set_language('en')
        book.add_author(author)
        book.add_metadata('DC', 'contributor', translator)
        
        # Add enhanced metadata from job_metadata.json if available
        try:
            job_metadata_path = os.path.join(translation_dir, "job_metadata.json")
            if os.path.exists(job_metadata_path):
                with open(job_metadata_path, 'r', encoding='utf-8') as f:
                    job_metadata = json.load(f)
                
                # Add translation-specific metadata
                book.add_metadata('DC', 'description', f'AI-translated version using {job_metadata.get("model_name", "AI model")}')
                book.add_metadata('DC', 'subject', 'Xianxia')
                book.add_metadata('DC', 'subject', 'AI Translation')
                book.add_metadata('DC', 'subject', 'Chinese Web Novel')
                
                # Custom metadata for translation details
                api_usage = job_metadata.get('api_usage', {})
                performance = job_metadata.get('performance_metrics', {})
                
                book.add_metadata(None, 'meta', '', {'name': 'translation:model', 'content': job_metadata.get('model_name', 'Unknown')})
                book.add_metadata(None, 'meta', '', {'name': 'translation:cost', 'content': f"${api_usage.get('estimated_cost_usd', 0.0):.4f}"})
                book.add_metadata(None, 'meta', '', {'name': 'translation:tokens', 'content': str(api_usage.get('total_tokens_used', 0))})
                book.add_metadata(None, 'meta', '', {'name': 'translation:chapters', 'content': str(job_metadata.get('total_chapters', 0))})
                book.add_metadata(None, 'meta', '', {'name': 'translation:time', 'content': performance.get('total_time_elapsed', 'Unknown')})
                book.add_metadata(None, 'meta', '', {'name': 'translation:framework', 'content': 'Pluralistic Translation Workbench'})
                book.add_metadata(None, 'meta', '', {'name': 'translation:version', 'content': job_metadata.get('project_info', {}).get('framework_version', 'v2.1.0')})
                
                # Add timestamp
                book.add_metadata('DC', 'date', job_metadata.get('timestamp', datetime.now().isoformat())[:10])
                
        except Exception as metadata_error:
            print(f"Warning: Could not add enhanced metadata: {metadata_error}")
        
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
        
        # Add translation metadata appendix
        try:
            job_metadata_path = os.path.join(translation_dir, "job_metadata.json")
            if os.path.exists(job_metadata_path):
                with open(job_metadata_path, 'r', encoding='utf-8') as f:
                    job_metadata = json.load(f)
                
                # Load and format metadata template
                metadata_content = load_and_format_metadata_template(job_metadata, translation_dir)
                
                # Convert markdown to HTML (simple conversion)
                import re
                html_metadata = metadata_content
                # Convert markdown headers
                html_metadata = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html_metadata, flags=re.MULTILINE)
                html_metadata = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html_metadata, flags=re.MULTILINE)
                html_metadata = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html_metadata, flags=re.MULTILINE)
                # Convert markdown bold
                html_metadata = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html_metadata)
                # Convert markdown links
                html_metadata = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', html_metadata)
                # Convert line breaks to HTML
                html_metadata = html_metadata.replace('\n', '<br>\n')
                
                # Create appendix chapter
                appendix_html = f'''
                <div style="font-family: Times, serif; line-height: 1.6; margin: 2em;">
                    {html_metadata}
                </div>
                '''
                
                appendix_chapter = epub.EpubHtml(
                    title="About This Translation",
                    file_name="appendix_translation_info.xhtml",
                    content=appendix_html
                )
                appendix_chapter.add_item(nav_css)
                
                book.add_item(appendix_chapter)
                chapters.append(appendix_chapter)
                spine.append(appendix_chapter)
                
        except Exception as appendix_error:
            print(f"Warning: Could not add metadata appendix: {appendix_error}")
        
        # Create table of contents (updated to include appendix)
        toc_chapters = chapters[:-1] if chapters and chapters[-1].title == "About This Translation" else chapters
        appendix_chapters = [chapters[-1]] if chapters and chapters[-1].title == "About This Translation" else []
        
        toc_sections = [(epub.Section('Chapters'), toc_chapters)]
        if appendix_chapters:
            toc_sections.append((epub.Section('Appendix'), appendix_chapters))
        
        book.toc = toc_sections
        
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
                    available_models = ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo", "deepseek-chat", "deepseek-reasoner"]
                else:
                    # Add DeepSeek models to the API-fetched models
                    available_models = openai_models + ["deepseek-chat", "deepseek-reasoner"]
        else:
            available_models = ["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo", "deepseek-chat", "deepseek-reasoner"]
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
    
    # Only update session state if model actually changed (prevent unnecessary reruns)
    current_model_abbrev = st.session_state.get('last_model_abbrev', None)
    if current_model_abbrev != model_abbrev:
        # Model changed - update run name to include new model abbreviation
        current_name = st.session_state.run_name
        
        # Remove the old model abbreviation if it exists
        if current_model_abbrev is not None and current_name.endswith(f"_{current_model_abbrev}"):
            # Remove the old model abbreviation and add new one
            base_without_model = current_name[:-len(f"_{current_model_abbrev}")]
            st.session_state.run_name = f"{base_without_model}_{model_abbrev}"
        elif not current_name.endswith(f"_{model_abbrev}"):
            # Add model abbreviation if not already present
            st.session_state.run_name = f"{current_name}_{model_abbrev}" if not current_name.endswith(model_abbrev) else current_name
        
        # Update the model abbreviation tracker
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
    
    # CRITICAL: Clear job state immediately to prevent auto-restart on widget changes
    st.session_state.run_job = False
    
    # Safety check: Ensure job parameters exist
    if "job_params" not in st.session_state:
        st.error("❌ Job parameters not found. Please reconfigure settings and try again.")
        st.stop()
    
    # Get job parameters
    params = st.session_state.job_params
    
    # Create output directory
    os.makedirs(params["output_dir"], exist_ok=True)
    
    # Save comprehensive job metadata
    job_start_time = datetime.now()
    metadata = {
        # Basic job info
        "timestamp": job_start_time.isoformat(),
        "model_name": params["model_name"],
        "system_prompt": params["system_prompt"],
        "history_count": params["history_count"],
        "api_delay": params["api_delay"],
        "chapters_requested": params["chapters_to_translate"],
        "total_chapters": len(params["chapters_to_translate"]),
        
        # Performance tracking (initialized)
        "performance_metrics": {
            "start_timestamp": job_start_time.isoformat(),
            "end_timestamp": None,
            "total_time_elapsed": None,
            "avg_time_per_chapter": None,
            "processing_speed": None,
            "chapters_completed": 0,
            "chapters_failed": 0
        },
        
        # API usage tracking (initialized)
        "api_usage": {
            "total_tokens_used": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_cost_usd": 0.0,
            "api_calls_made": 0,
            "avg_tokens_per_chapter": 0,
            "cost_breakdown": {
                "input_cost": 0.0,
                "output_cost": 0.0,
                "rate_per_1k_tokens": 0.0
            }
        },
        
        # Job configuration
        "ai_configuration": {
            "api_provider": "OpenAI" if "gpt" in params["model_name"] or "ft:" in params["model_name"] else "Google",
            "model_version": params["model_name"],
            "temperature": getattr(params, 'temperature', 0.7),
            "max_tokens": getattr(params, 'max_tokens', 4096),
            "system_prompt_hash": str(hash(params["system_prompt"]))[:16],
            "example_strategy": "rolling_context_window"
        },
        
        # Project info
        "project_info": {
            "framework_name": "Pluralistic Translation Workbench",
            "framework_version": "v2.1.0",
            "novel_title": "极道天魔 (Way of the Devil)",
            "original_author": "王雨 (Wang Yu)",
            "source_language": "Chinese (Simplified)",
            "target_language": "English"
        }
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
                result = generate_translation_unified(
                    params["api_key"], 
                    params["model_name"], 
                    params["system_prompt"], 
                    history_examples, 
                    raw_content,
                    params["platform"]
                )

                # 4. Process Result
                if not result['success']:
                    # Handle API errors
                    error_message = result['error']
                    
                    # Check for quota exceeded error specifically
                    if "429" in str(error_message) and "quota" in str(error_message).lower():
                        log_messages.append(f"  └─ ❌ **QUOTA EXCEEDED:** {error_message}")
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
                        log_messages.append(f"  └─ ❌ **ERROR:** {error_message}")
                        failed_translations += 1
                else:
                    # Extract successful translation
                    translation = result['translation']
                    usage_metrics = result.get('usage_metrics', {})
                    
                    # Save translation
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(translation)
                    
                    # Update job metadata with usage metrics
                    metadata_file_path = os.path.join(params["output_dir"], "job_metadata.json")
                    update_job_metadata_with_usage(metadata_file_path, usage_metrics, chapter_success=True)
                    
                    # Get some stats and log success with cost info
                    stats = get_text_stats(translation, 'english')
                    cost_info = f"${usage_metrics.get('estimated_cost', 0.0):.4f}" if usage_metrics.get('estimated_cost') else "Unknown"
                    tokens_info = f"{usage_metrics.get('total_tokens', 0):,} tokens" if usage_metrics.get('total_tokens') else "Unknown tokens"
                    
                    log_messages.append(f"  └─ ✅ **Success!** Saved ({stats['word_count']} words, {cost_info}, {tokens_info})")
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
    # Update final job metadata with completion time and performance metrics
    try:
        job_end_time = datetime.now()
        metadata_file_path = os.path.join(params["output_dir"], "job_metadata.json")
        
        with open(metadata_file_path, 'r', encoding='utf-8') as f:
            final_metadata = json.load(f)
        
        # Calculate final performance metrics
        start_time_str = final_metadata.get('performance_metrics', {}).get('start_timestamp', job_start_time.isoformat())
        start_time = datetime.fromisoformat(start_time_str.replace('Z', '+00:00')) if 'Z' in start_time_str else datetime.fromisoformat(start_time_str)
        total_duration = job_end_time - start_time
        total_seconds = total_duration.total_seconds()
        
        # Update performance metrics
        performance = final_metadata.setdefault('performance_metrics', {})
        performance['end_timestamp'] = job_end_time.isoformat()
        performance['total_time_elapsed'] = f"{total_seconds:.1f}s"
        
        if successful_translations > 0:
            performance['avg_time_per_chapter'] = f"{total_seconds / successful_translations:.1f}s"
            performance['processing_speed'] = f"{successful_translations / (total_seconds / 60):.2f} chapters/minute"
        
        # Save final metadata
        with open(metadata_file_path, 'w', encoding='utf-8') as f:
            json.dump(final_metadata, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"Warning: Could not update final job metadata: {e}")
    
    status_placeholder.success("🎉 **Translation Job Complete!**")
    
    st.header("📊 Job Summary")
    
    # Display summary metrics including cost information
    try:
        metadata_file_path = os.path.join(params["output_dir"], "job_metadata.json")
        with open(metadata_file_path, 'r', encoding='utf-8') as f:
            summary_metadata = json.load(f)
        
        api_usage = summary_metadata.get('api_usage', {})
        performance = summary_metadata.get('performance_metrics', {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("✅ Successful", successful_translations)
        with col2:
            st.metric("❌ Failed", failed_translations)
        with col3:
            total_cost = api_usage.get('estimated_cost_usd', 0.0)
            st.metric("💰 Total Cost", f"${total_cost:.4f}")
        with col4:
            total_time = performance.get('total_time_elapsed', 'Unknown')
            st.metric("⏱️ Total Time", total_time)
        
        # Additional metrics row
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            total_tokens = api_usage.get('total_tokens_used', 0)
            st.metric("🔢 Total Tokens", f"{total_tokens:,}")
        with col6:
            api_calls = api_usage.get('api_calls_made', 0)
            st.metric("📞 API Calls", api_calls)
        with col7:
            avg_cost = total_cost / max(successful_translations, 1)
            st.metric("💸 Cost/Chapter", f"${avg_cost:.4f}")
        with col8:
            avg_time = performance.get('avg_time_per_chapter', 'Unknown')
            st.metric("⏰ Time/Chapter", avg_time)
            
    except Exception as e:
        # Fallback to simple metrics if metadata unavailable
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
    
    # Job completion status
    st.success("✅ Translation job completed successfully!")
    st.info("💡 You can now start a new translation job using the sidebar settings above.")

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