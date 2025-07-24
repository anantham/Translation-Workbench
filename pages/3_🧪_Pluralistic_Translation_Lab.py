import streamlit as st
import json
import os
import time
from datetime import datetime
from ebooklib import epub

# Import shared utilities
from utils import (
    load_chapter_content,
    get_text_stats,
    DATA_DIR,
    GOOGLE_AI_AVAILABLE,
    load_api_config,
    load_openai_api_config,
    get_config_value,
    load_epub_metadata_config,
    calculate_word_counts,
    calculate_average_quality_metrics,
    get_static_gemini_models,
    get_available_openai_models,
    generate_translation_unified,
    get_all_available_prompts,
    save_custom_prompt,
    delete_custom_prompt,
    load_custom_prompts
)

# --- Helper Functions ---
def detect_novel_slug_from_alignment_maps():
    """Detect novel slug from available alignment maps.
    
    Returns:
        str: Novel slug (e.g., 'way_of_the_devil') or None if not found
    """
    try:
        from utils import list_alignment_maps
        alignment_maps = list_alignment_maps()
        
        # For now, default to 'way_of_the_devil' if available
        # This can be enhanced later with UI selection
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
        return 'way_of_the_devil'  # Default fallback

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

def get_official_examples(alignment_map, current_chapter_num, count, direction="past"):
    """Get examples from official EPUB translations with bidirectional search."""
    examples = []
    available_history = sorted([
        int(k) for k, v in alignment_map.items() 
        if (
            (direction == "past" and int(k) < current_chapter_num) or
            (direction == "future" and int(k) > current_chapter_num)
        ) and v.get("raw_file") and v.get("english_file")
    ], reverse=(direction == "past"))
    
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

def get_custom_run_examples(alignment_map, current_chapter_num, count, custom_run_name, direction="past"):
    """Get examples from a specific custom translation run with bidirectional search."""
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
                if (
                    (direction == "past" and chapter_num < current_chapter_num) or
                    (direction == "future" and chapter_num > current_chapter_num)
                ):
                    available_chapters.append(chapter_num)
            except (IndexError, ValueError):
                continue
    
    # Sort by chapter number (most recent first for past, earliest first for future) and take requested count
    available_chapters.sort(reverse=(direction == "past"))
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

def get_current_run_examples(alignment_map, current_chapter_num, count, current_run_dir, direction="past", processing_order="forward"):
    """Get examples from the current translation run (fresh translations) with bidirectional search and sliding context."""
    examples = []
    
    if not os.path.exists(current_run_dir):
        return examples
    
    # Find available translated chapters in current run
    available_chapters = []
    for filename in os.listdir(current_run_dir):
        if filename.endswith('-translated.txt'):
            try:
                chapter_num = int(filename.split('-')[1])
                
                # Sliding context logic: In reverse processing, use freshly translated chapters
                if processing_order == "reverse" and direction == "future":
                    # For reverse processing: use chapters that were already translated (higher numbers)
                    if chapter_num > current_chapter_num:
                        available_chapters.append(chapter_num)
                elif processing_order == "forward" and direction == "past":
                    # For forward processing: use chapters that were already translated (lower numbers)
                    if chapter_num < current_chapter_num:
                        available_chapters.append(chapter_num)
                else:
                    # Standard bidirectional logic
                    if (
                        (direction == "past" and chapter_num < current_chapter_num) or
                        (direction == "future" and chapter_num > current_chapter_num)
                    ):
                        available_chapters.append(chapter_num)
            except (IndexError, ValueError):
                continue
    
    # Sort by chapter number (most recent first for past, earliest first for future) and take requested count
    available_chapters.sort(reverse=(direction == "past"))
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
def get_smart_fallback_examples(alignment_map, current_chapter_num, count, selected_custom_run, current_run_dir, direction="past", processing_order="forward"):
    """
    Smart fallback system for getting translation examples with bidirectional search and sliding context.
    Priority: Official > Selected Custom > Current Run > Error
    """
    examples = []
    sources_used = []
    
    # Priority 1: Official translations
    official_examples = get_official_examples(alignment_map, current_chapter_num, count, direction)
    examples.extend(official_examples)
    if official_examples:
        sources_used.append(f"Official ({len(official_examples)})")
    
    # Priority 2: Selected custom run (if specified and still need more)
    if len(examples) < count and selected_custom_run:
        remaining_count = count - len(examples)
        custom_examples = get_custom_run_examples(alignment_map, current_chapter_num, remaining_count, selected_custom_run, direction)
        examples.extend(custom_examples)
        if custom_examples:
            sources_used.append(f"Custom ({len(custom_examples)})")
    
    # Priority 3: Current run (if still need more) - with sliding context support
    if len(examples) < count:
        remaining_count = count - len(examples)
        fresh_examples = get_current_run_examples(alignment_map, current_chapter_num, remaining_count, current_run_dir, direction, processing_order)
        examples.extend(fresh_examples)
        if fresh_examples:
            sources_used.append(f"Fresh ({len(fresh_examples)})")
    
    # Error handling: insufficient examples
    if len(examples) == 0:
        direction_word = "before" if direction == "past" else "after"
        raise InsufficientHistoryError(
            f"No history chapters found {direction_word} chapter {current_chapter_num}. "
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
    
    # Load EPUB metadata configuration
    epub_config = load_epub_metadata_config()
    
    # Calculate word counts
    word_counts = calculate_word_counts(translation_dir)
    
    # Calculate quality metrics
    quality_metrics = calculate_average_quality_metrics(translation_dir)
    
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
        'novel_title': epub_config.get('novel_title', '极道天魔 (Way of the Devil)'),
        'original_author': epub_config.get('original_author', '王雨 (Wang Yu)'),
        'chapter_range': f"{min(job_metadata.get('chapters_requested', []))}-{max(job_metadata.get('chapters_requested', []))}" if job_metadata.get('chapters_requested') else 'Unknown',
        'total_chapters': job_metadata.get('total_chapters', 0),
        'source_language': epub_config.get('source_language', 'Chinese (Simplified)'),
        'target_language': epub_config.get('target_language', 'English'),
        'word_count_chinese': f"{word_counts['chinese_characters']:,}" if word_counts['chinese_characters'] > 0 else 'Unknown',
        'word_count_english': f"{word_counts['english_words']:,}" if word_counts['english_words'] > 0 else 'Unknown',
        'expansion_ratio': f"{word_counts['expansion_ratio']}x" if word_counts['expansion_ratio'] != 'Unknown' else 'Unknown',
        'github_url': epub_config.get('github_url', 'https://github.com/anthropics/translation-workbench'),
        'project_version': epub_config.get('project_version', 'v2.1.0'),
        'license': epub_config.get('license', 'MIT License'),
        'maintainer_name': 'Translation Workbench',
        'maintainer_email': epub_config.get('maintainer_email', 'contact@example.com'),
        'feature_requests_url': epub_config.get('feature_requests_url', 'https://github.com/anthropics/translation-workbench/issues'),
        'documentation_url': epub_config.get('documentation_url', 'https://docs.example.com/translation-workbench'),
        'consistency_score': quality_metrics['consistency_score'],
        'bleu_score_avg': quality_metrics['avg_bleu_score'],
        'semantic_similarity_avg': quality_metrics['avg_semantic_similarity'],
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
        'framework_version': epub_config.get('project_version', 'v2.1.0'),
        'framework_name': epub_config.get('framework_name', 'Pluralistic Translation Workbench'),
        'generation_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'github_discussions_url': epub_config.get('github_discussions_url', 'https://github.com/anthropics/translation-workbench/discussions'),
        'license_url': epub_config.get('license_url', 'https://github.com/anthropics/translation-workbench/blob/main/LICENSE'),
        'translation_philosophy': epub_config.get('translation_philosophy', 'This translation was generated using an AI-powered framework designed for consistent, high-quality translation.')
    }
    
    # Substitute all variables in template
    formatted_template = template
    for key, value in substitutions.items():
        formatted_template = formatted_template.replace(f'{{{key}}}', str(value))
    
    return formatted_template


def load_novel_images(novel_slug):
    """Load all images for a novel with their metadata and captions.
    
    Args:
        novel_slug (str): Novel identifier
        
    Returns:
        dict: Dictionary with image information and file paths
    """
    from utils import load_novel_images_config
    
    images_config = load_novel_images_config(novel_slug)
    images_path = images_config['images_path']
    manifest = images_config['manifest']
    
    loaded_images = {
        'cover': None,
        'chapter_illustrations': {},
        'appendix_illustrations': {},
        'common_branding': {}
    }
    
    # Load cover image
    cover_info = manifest.get('cover', {})
    if cover_info:
        cover_file = os.path.join(images_path, cover_info['file'])
        if os.path.exists(cover_file):
            loaded_images['cover'] = {
                'file_path': cover_file,
                'title': cover_info.get('title', 'Cover'),
                'description': cover_info.get('description', ''),
                'artist': cover_info.get('artist', 'Unknown')
            }
    
    # Load chapter illustrations
    chapter_illustrations = manifest.get('chapter_illustrations', {})
    for chapter_num, illustration_info in chapter_illustrations.items():
        image_file = os.path.join(images_path, illustration_info['file'])
        if os.path.exists(image_file):
            loaded_images['chapter_illustrations'][chapter_num] = {
                'file_path': image_file,
                'title': illustration_info.get('title', f'Chapter {chapter_num} Illustration'),
                'description': illustration_info.get('description', ''),
                'caption': illustration_info.get('caption', ''),
                'placement': illustration_info.get('placement', 'beginning')
            }
    
    # Load appendix illustrations
    appendix_illustrations = manifest.get('appendix_illustrations', {})
    for key, illustration_info in appendix_illustrations.items():
        image_file = os.path.join(images_path, illustration_info['file'])
        if os.path.exists(image_file):
            loaded_images['appendix_illustrations'][key] = {
                'file_path': image_file,
                'title': illustration_info.get('title', key.replace('_', ' ').title()),
                'description': illustration_info.get('description', ''),
                'caption': illustration_info.get('caption', '')
            }
    
    # Load common branding images
    from utils import get_config_value
    branding_config = get_config_value('branding', {})
    brand_images = branding_config.get('brand_images', {})
    
    for key, image_path in brand_images.items():
        if os.path.exists(image_path):
            loaded_images['common_branding'][key] = {
                'file_path': image_path,
                'title': f'Framework {key.replace("_", " ").title()}',
                'description': f'Framework branding: {key}'
            }
    
    return loaded_images


def create_gallery_sections(book, novel_images, nav_css):
    """Create dedicated gallery sections for images in EPUB.
    
    Args:
        book: EPUB book object
        novel_images: Dictionary of novel images
        nav_css: CSS style item
        
    Returns:
        list: List of created gallery chapters
    """
    gallery_chapters = []
    
    # 1. Cover Gallery
    if novel_images.get('cover'):
        cover_info = novel_images['cover']
        cover_html = f'''
        <div style="font-family: Times, serif; line-height: 1.6; margin: 2em;">
            <h1>📸 Cover Gallery</h1>
            <div class="gallery-item">
                <img src="images/cover" alt="{cover_info['title']}" style="max-width: 100%; height: auto; margin: 2em 0;" />
                <h3>{cover_info['title']}</h3>
                <p><strong>Description:</strong> {cover_info.get('description', '')}</p>
                <p><strong>Artist:</strong> {cover_info.get('artist', 'Unknown')}</p>
            </div>
        </div>
        '''
        
        cover_chapter = epub.EpubHtml(
            title="Cover Gallery",
            file_name="gallery_cover.xhtml",
            content=cover_html
        )
        cover_chapter.add_item(nav_css)
        book.add_item(cover_chapter)
        gallery_chapters.append(cover_chapter)
    
    # 2. Chapter Illustrations Gallery
    if novel_images.get('chapter_illustrations'):
        chapter_illustrations_html = '''
        <div style="font-family: Times, serif; line-height: 1.6; margin: 2em;">
            <h1>📖 Chapter Illustrations</h1>
            <p>Visual highlights from key moments in the story.</p>
        '''
        
        # Sort chapters by number
        sorted_chapters = sorted(novel_images['chapter_illustrations'].items(), key=lambda x: int(x[0]))
        
        for chapter_num, image_info in sorted_chapters:
            chapter_illustrations_html += f'''
            <div class="gallery-item" style="margin: 3em 0; border-bottom: 1px solid #eee; padding-bottom: 2em;">
                <h3>Chapter {chapter_num}: {image_info['title']}</h3>
                <img src="images/chapter_{chapter_num}" alt="{image_info['title']}" style="max-width: 100%; height: auto; margin: 1em 0;" />
                <p><strong>Description:</strong> {image_info.get('description', '')}</p>
                <p><em>"{image_info.get('caption', '')}"</em></p>
            </div>
            '''
        
        chapter_illustrations_html += '</div>'
        
        chapter_illustrations_chapter = epub.EpubHtml(
            title="Chapter Illustrations",
            file_name="gallery_chapters.xhtml",
            content=chapter_illustrations_html
        )
        chapter_illustrations_chapter.add_item(nav_css)
        book.add_item(chapter_illustrations_chapter)
        gallery_chapters.append(chapter_illustrations_chapter)
    
    # 3. Character References Gallery
    if novel_images.get('appendix_illustrations'):
        appendix_items = novel_images['appendix_illustrations']
        character_items = {k: v for k, v in appendix_items.items() if 'character' in k}
        
        if character_items:
            character_html = '''
            <div style="font-family: Times, serif; line-height: 1.6; margin: 2em;">
                <h1>👥 Character References</h1>
                <p>Visual guides to the characters and their relationships.</p>
            '''
            
            for key, image_info in character_items.items():
                character_html += f'''
                <div class="gallery-item" style="margin: 3em 0; border-bottom: 1px solid #eee; padding-bottom: 2em;">
                    <h3>{image_info['title']}</h3>
                    <img src="images/appendix_{key}" alt="{image_info['title']}" style="max-width: 100%; height: auto; margin: 1em 0;" />
                    <p><strong>Description:</strong> {image_info.get('description', '')}</p>
                    <p><em>"{image_info.get('caption', '')}"</em></p>
                </div>
                '''
            
            character_html += '</div>'
            
            character_chapter = epub.EpubHtml(
                title="Character References",
                file_name="gallery_characters.xhtml",
                content=character_html
            )
            character_chapter.add_item(nav_css)
            book.add_item(character_chapter)
            gallery_chapters.append(character_chapter)
    
    # 4. World Building Gallery
    world_html = '''
    <div style="font-family: Times, serif; line-height: 1.6; margin: 2em;">
        <h1>🌍 World Building</h1>
        <p>Maps, diagrams, and reference materials for the novel's world.</p>
    '''
    
    # Add framework branding section
    if novel_images.get('common_branding'):
        world_html += '''
        <h2>🏭 Translation Framework</h2>
        <p>The tools and process behind this translation.</p>
        '''
        
        branding_items = novel_images['common_branding']
        for key, image_info in branding_items.items():
            world_html += f'''
            <div class="gallery-item" style="margin: 3em 0; border-bottom: 1px solid #eee; padding-bottom: 2em;">
                <h3>{image_info['title']}</h3>
                <img src="images/branding_{key}" alt="{image_info['title']}" style="max-width: 100%; height: auto; margin: 1em 0;" />
                <p><strong>Description:</strong> {image_info.get('description', '')}</p>
            </div>
            '''
    
    # Add novel-specific world building items
    if novel_images.get('appendix_illustrations'):
        appendix_items = novel_images['appendix_illustrations']
        world_items = {k: v for k, v in appendix_items.items() if 'character' not in k}
        
        if world_items:
            world_html += '''
            <h2>🗺️ Novel World</h2>
            <p>The setting and systems of the story.</p>
            '''
            
            for key, image_info in world_items.items():
                world_html += f'''
                <div class="gallery-item" style="margin: 3em 0; border-bottom: 1px solid #eee; padding-bottom: 2em;">
                    <h3>{image_info['title']}</h3>
                    <img src="images/appendix_{key}" alt="{image_info['title']}" style="max-width: 100%; height: auto; margin: 1em 0;" />
                    <p><strong>Description:</strong> {image_info.get('description', '')}</p>
                    <p><em>"{image_info.get('caption', '')}"</em></p>
                </div>
                '''
    
    world_html += '</div>'
    
    # Only create the world building chapter if we have content
    if novel_images.get('common_branding') or (novel_images.get('appendix_illustrations') and any('character' not in k for k in novel_images['appendix_illustrations'].keys())):
        world_chapter = epub.EpubHtml(
            title="World Building",
            file_name="gallery_world.xhtml",
            content=world_html
        )
        world_chapter.add_item(nav_css)
        book.add_item(world_chapter)
        gallery_chapters.append(world_chapter)
    
    return gallery_chapters

def add_image_to_epub(book, image_path, image_id, media_type=None):
    """Add an image to the EPUB book.
    
    Args:
        book: EPUB book object
        image_path (str): Path to the image file
        image_id (str): Unique identifier for the image
        media_type (str): MIME type of the image
        
    Returns:
        EpubImage: The created image item
    """
    if not os.path.exists(image_path):
        return None
    
    # Determine media type if not provided
    if media_type is None:
        ext = os.path.splitext(image_path)[1].lower()
        media_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml',
            '.webp': 'image/webp'
        }
        media_type = media_type_map.get(ext, 'image/jpeg')
    
    # Read image data
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    # Create EPUB image item
    from ebooklib import epub
    image_item = epub.EpubImage(
        uid=image_id,
        file_name=f"images/{image_id}{os.path.splitext(image_path)[1]}",
        media_type=media_type,
        content=image_data
    )
    
    book.add_item(image_item)
    return image_item


def convert_markdown_to_html(text):
    """Convert basic markdown formatting to HTML for EPUB compatibility.
    
    Supported markdown features:
    - **bold** → <strong>bold</strong>
    - *italic* → <em>italic</em>
    - __underline__ → <u>underline</u>
    - ~~strikethrough~~ → <s>strikethrough</s>
    - `code` → <code>code</code>
    - "quotes" → <q>quotes</q>
    - \\n → <br/> (explicit line breaks)
    - -- → — (em dash)
    - ... → … (ellipsis)
    
    Args:
        text (str): Raw text with markdown formatting
        
    Returns:
        str: HTML formatted text
    """
    import re
    
    # Bold: **text** -> <strong>text</strong>
    text = re.sub(r'\*\*([^*]+?)\*\*', r'<strong>\1</strong>', text)
    
    # Italics: *text* -> <em>text</em> (after bold to avoid conflicts)
    text = re.sub(r'\*([^*]+?)\*', r'<em>\1</em>', text)
    
    # Underline: __text__ -> <u>text</u>
    text = re.sub(r'__([^_]+?)__', r'<u>\1</u>', text)
    
    # Strikethrough: ~~text~~ -> <s>text</s>
    text = re.sub(r'~~([^~]+?)~~', r'<s>\1</s>', text)
    
    # Code: `text` -> <code>text</code>
    text = re.sub(r'`([^`]+?)`', r'<code>\1</code>', text)
    
    # Quotes: "text" -> <q>text</q> (optional - for proper typography)
    text = re.sub(r'"([^"]+?)"', r'<q>\1</q>', text)
    
    # Line breaks: explicit \n -> <br/> (for poetry or dialogue)
    text = re.sub(r'(?<!\\)\\n', '<br/>', text)
    
    # Em dash: -- -> —
    text = re.sub(r'--', '—', text)
    
    # Ellipsis: ... -> …
    text = re.sub(r'\.\.\.', '…', text)
    
    return text

def create_epub_from_translations(translation_dir, output_path, book_title, author="Unknown", translator="AI Translation", novel_slug=None):
    """Create an EPUB file from translated text files with markdown formatting support and novel-specific branding."""
    try:
        # Create EPUB book
        book = epub.EpubBook()
        
        # Load novel-specific configuration
        novel_config = {}
        novel_images = {}
        branding_config = {}
        
        if novel_slug:
            from utils import load_novel_config
            full_config = load_novel_config(novel_slug)
            novel_config = full_config.get('novel', {})
            branding_config = full_config.get('branding', {})
            
            # Load images for this novel
            novel_images = load_novel_images(novel_slug)
        
        # Set basic metadata with novel-specific information
        novel_info = novel_config.get('novel_info', {})
        book_id = f"{novel_slug}-translation" if novel_slug else "translation"
        book.set_identifier(book_id)
        book.set_title(book_title)
        book.set_language('en')
        book.add_author(author)
        book.add_metadata('DC', 'contributor', translator)
        
        # Add genre information
        genres = novel_info.get('genre', [])
        for genre in genres:
            book.add_metadata('DC', 'subject', genre)
        
        # Add description
        description = novel_info.get('description', '')
        if description:
            book.add_metadata('DC', 'description', description)
        
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
                epub_meta_config = load_epub_metadata_config()
                book.add_metadata(None, 'meta', '', {'name': 'translation:version', 'content': epub_meta_config.get('project_version', 'v2.1.0')})
                
                # Add timestamp
                book.add_metadata('DC', 'date', job_metadata.get('timestamp', datetime.now().isoformat())[:10])
                
        except Exception as metadata_error:
            print(f"Warning: Could not add enhanced metadata: {metadata_error}")
        
        # Add cover image if available
        cover_item = None
        if novel_images.get('cover'):
            cover_info = novel_images['cover']
            cover_item = add_image_to_epub(book, cover_info['file_path'], 'cover')
            if cover_item:
                book.set_cover(cover_item.file_name, cover_item.content)
        
        # Add branding images to book
        branding_images = {}
        for key, image_info in novel_images.get('common_branding', {}).items():
            image_item = add_image_to_epub(book, image_info['file_path'], f'branding_{key}')
            if image_item:
                branding_images[key] = image_item
        
        # Add chapter illustrations to book
        chapter_image_items = {}
        for chapter_num, image_info in novel_images.get('chapter_illustrations', {}).items():
            image_item = add_image_to_epub(book, image_info['file_path'], f'chapter_{chapter_num}')
            if image_item:
                chapter_image_items[chapter_num] = image_item
        
        # Add appendix illustrations to book
        appendix_image_items = {}
        for key, image_info in novel_images.get('appendix_illustrations', {}).items():
            image_item = add_image_to_epub(book, image_info['file_path'], f'appendix_{key}')
            if image_item:
                appendix_image_items[key] = image_item
        
        # Add CSS style with markdown formatting support
        style = '''
        body { font-family: Times, serif; line-height: 1.6; margin: 2em; }
        h1 { text-align: center; margin-bottom: 2em; }
        p { text-indent: 2em; margin-bottom: 1em; }
        em { font-style: italic; }
        strong { font-weight: bold; }
        u { text-decoration: underline; }
        s { text-decoration: line-through; }
        code { font-family: monospace; background-color: #f0f0f0; padding: 2px 4px; border-radius: 3px; }
        q { quotes: """ """ "'" "'"; }
        q:before { content: open-quote; }
        q:after { content: close-quote; }
        .illustration { text-align: center; margin: 2em 0; }
        .illustration img { max-width: 100%; height: auto; }
        .illustration-caption { font-style: italic; margin-top: 1em; text-align: center; }
        .branding-seal { float: right; margin: 1em; max-width: 100px; }
        '''
        nav_css = epub.EpubItem(
            uid="nav_css",
            file_name="style/nav.css",
            media_type="text/css",
            content=style
        )
        book.add_item(nav_css)
        
        # Get all translation files (flexible file pattern matching)
        translation_files = []
        for filename in os.listdir(translation_dir):
            if filename.endswith('.txt'):
                # Try different naming patterns to extract chapter number
                chapter_num = None
                
                # Pattern 1: Chapter-0001-translated.txt, Chapter-0044-translated.txt
                if 'Chapter-' in filename:
                    try:
                        chapter_num = int(filename.split('-')[1])
                    except (IndexError, ValueError):
                        pass
                
                # Pattern 2: Ch001.txt, Ch44.txt
                elif filename.startswith('Ch') and filename.endswith('.txt'):
                    try:
                        import re
                        match = re.search(r'Ch(\d+)', filename)
                        if match:
                            chapter_num = int(match.group(1))
                    except (IndexError, ValueError):
                        pass
                
                # Pattern 3: 001.txt, 044.txt
                elif filename.replace('.txt', '').isdigit():
                    try:
                        chapter_num = int(filename.replace('.txt', ''))
                    except (IndexError, ValueError):
                        pass
                
                # Pattern 4: Any other .txt file - use filename as fallback
                if chapter_num is None:
                    # Try to extract any number from filename
                    import re
                    numbers = re.findall(r'\d+', filename)
                    if numbers:
                        try:
                            chapter_num = int(numbers[0])  # Use first number found
                        except ValueError:
                            pass
                    
                    # If still no number found, use file position as chapter number
                    if chapter_num is None:
                        # We'll assign numbers later based on alphabetical order
                        chapter_num = 0
                
                if chapter_num is not None:
                    translation_files.append((chapter_num, filename))
        
        # Handle files with no detectable chapter numbers
        # Sort by filename and assign sequential numbers
        files_with_zero = [(num, name) for num, name in translation_files if num == 0]
        files_with_numbers = [(num, name) for num, name in translation_files if num != 0]
        
        if files_with_zero:
            files_with_zero.sort(key=lambda x: x[1])  # Sort by filename
            # Assign sequential numbers starting from the highest numbered file + 1
            next_num = max([num for num, _ in files_with_numbers], default=0) + 1
            for i, (_, filename) in enumerate(files_with_zero):
                files_with_numbers.append((next_num + i, filename))
        
        translation_files = files_with_numbers
        
        # Sort by chapter number
        translation_files.sort(key=lambda x: x[0])
        
        chapters = []
        spine = ['nav']
        total_markdown_chapters = 0
        
        # Create chapters
        for chapter_num, filename in translation_files:
            file_path = os.path.join(translation_dir, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            # Create chapter
            chapter_id = f"chapter_{chapter_num:04d}"
            chapter_title = f"Chapter {chapter_num}"
            
            # Format content as HTML with markdown support
            paragraphs = content.split('\n\n')
            html_content = f'<h1>{chapter_title}</h1>\n'
            
            # Add branding seal if available
            seal_image = branding_images.get('seal')
            if seal_image:
                html_content += f'<img src="{seal_image.file_name}" class="branding-seal" alt="Artisan\'s Seal" />\n'
            
            # Chapter illustrations are now handled in dedicated gallery sections
            # No inline images in chapter content to maintain reading flow
            
            # Track markdown formatting usage
            has_markdown = any('*' in p or '_' in p or '`' in p or '~' in p for p in paragraphs)
            if has_markdown:
                total_markdown_chapters += 1
            
            for paragraph in paragraphs:
                if paragraph.strip():
                    # Convert markdown formatting to HTML
                    formatted_paragraph = convert_markdown_to_html(paragraph.strip())
                    html_content += f'<p>{formatted_paragraph}</p>\n'
            
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
        
        # Add gallery sections
        gallery_chapters = []
        try:
            gallery_chapters = create_gallery_sections(book, novel_images, nav_css)
            for gallery_chapter in gallery_chapters:
                chapters.append(gallery_chapter)
                spine.append(gallery_chapter)
        except Exception as gallery_error:
            print(f"Warning: Could not add gallery sections: {gallery_error}")
        
        # Create table of contents (updated to include gallery and appendix)
        # Separate different section types
        story_chapters = []
        appendix_chapters = []
        
        for chapter in chapters:
            if chapter.title == "About This Translation":
                appendix_chapters.append(chapter)
            elif chapter.title in ["Cover Gallery", "Chapter Illustrations", "Character References", "World Building"]:
                # Gallery chapters will be added to their own section
                pass
            else:
                story_chapters.append(chapter)
        
        # Build TOC sections
        toc_sections = [(epub.Section('Chapters'), story_chapters)]
        
        # Add Gallery section if we have gallery chapters
        if gallery_chapters:
            toc_sections.append((epub.Section('📷 Image Gallery'), gallery_chapters))
        
        # Add Appendix section if we have appendix chapters
        if appendix_chapters:
            toc_sections.append((epub.Section('📚 Appendix'), appendix_chapters))
        
        book.toc = toc_sections
        
        # Add navigation
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        
        # Set spine
        book.spine = spine
        
        # Write EPUB
        epub.write_epub(output_path, book, {})
        
        # Create success message with markdown info
        success_msg = f"Successfully created EPUB with {len(chapters)} chapters"
        if total_markdown_chapters > 0:
            success_msg += f" (markdown formatting processed in {total_markdown_chapters} chapters)"
        
        return True, success_msg
        
    except Exception as e:
        return False, f"Error creating EPUB: {str(e)}"

# Removed old generate_translation_with_history function - replaced by unified function in utils.py

# --- UI Sidebar ---
st.sidebar.header("🔬 Experiment Controls")

# Novel Selection
with st.sidebar.expander("📚 Novel Selection", expanded=True):
    try:
        from utils import get_available_novels
        available_novels = get_available_novels()
        
        if available_novels:
            novel_options = [f"{novel['title']} ({novel['slug']})" for novel in available_novels]
            selected_novel_display = st.selectbox(
                "📖 Novel:", 
                novel_options, 
                help="Choose which novel to work with"
            )
            
            # Extract novel slug from selection
            selected_novel_slug = None
            for novel in available_novels:
                if f"{novel['title']} ({novel['slug']})" == selected_novel_display:
                    selected_novel_slug = novel['slug']
                    break
            
            if selected_novel_slug:
                # Store in session state for use in EPUB creation
                st.session_state['selected_novel_slug'] = selected_novel_slug
                
                # Show novel configuration status
                selected_novel_info = next(n for n in available_novels if n['slug'] == selected_novel_slug)
                if selected_novel_info['has_config']:
                    st.success("✅ Configuration loaded")
                else:
                    st.warning("⚠️ No novel config found")
        else:
            st.info("No novels configured yet")
            st.session_state['selected_novel_slug'] = 'way_of_the_devil'  # Default fallback
            
    except Exception as e:
        st.error(f"Error loading novels: {e}")
        st.session_state['selected_novel_slug'] = 'way_of_the_devil'  # Default fallback

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
                    if st.button("🗑️ Delete", key=f"delete_{prompt_name}"):
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
        help="Number of chapters to use as in-context examples"
    )
    
    # Direction control for history examples
    direction = st.radio(
        "📚 History Direction",
        options=["past", "future"],
        index=0,  # Default to "past"
        help="Use chapters before (past) or after (future) current chapter as examples. Future examples useful for translating early chapters using polished later chapters.",
        horizontal=True
    )
    
    # Processing order control for temporal consistency
    processing_order = st.radio(
        "⏳ Processing Order",
        options=["forward", "reverse"],
        index=0,  # Default to "forward"
        help="Forward: Process 1→30 (normal). Reverse: Process 30→1 (temporal consistency - fresh translations feed into sliding window context).",
        horizontal=True
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
        help="Select which novel's alignment map to use for translation"
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
    
except Exception as e:
    st.error(f"❌ Error loading alignment map: {str(e)}")
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
        "platform": selected_platform,
        "direction": direction,
        "processing_order": processing_order
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
            "framework_name": load_epub_metadata_config().get('framework_name', 'Pluralistic Translation Workbench'),
            "framework_version": load_epub_metadata_config().get('project_version', 'v2.1.0'),
            "novel_title": load_epub_metadata_config().get('novel_title', '极道天魔 (Way of the Devil)'),
            "original_author": load_epub_metadata_config().get('original_author', '王雨 (Wang Yu)'),
            "source_language": load_epub_metadata_config().get('source_language', 'Chinese (Simplified)'),
            "target_language": load_epub_metadata_config().get('target_language', 'English')
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
    
    # Process chapters in specified order
    chapters_to_process = params["chapters_to_translate"]
    if params["processing_order"] == "reverse":
        chapters_to_process = list(reversed(chapters_to_process))
        log_messages.append("⏳ **Processing Order**: Reverse (temporal consistency mode)")
    else:
        log_messages.append("⏳ **Processing Order**: Forward (standard mode)")
    
    for i, chapter_num_str in enumerate(chapters_to_process):
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
                direction_text = "previous" if params["direction"] == "past" else "subsequent"
                sliding_context_note = ""
                if params["processing_order"] == "reverse" and params["direction"] == "future":
                    sliding_context_note = " + sliding fresh context"
                log_messages.append(f"  └─ Building few-shot context ({direction_text} chapters{sliding_context_note})...")
                try:
                    history_examples, sources_used = get_smart_fallback_examples(
                        alignment_map, 
                        chapter_num, 
                        params["history_count"],
                        params["selected_custom_run"],
                        params["output_dir"],
                        params["direction"],
                        params["processing_order"]
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
            
    except Exception:
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
st.caption("Convert any folder containing chapter files into a downloadable EPUB book")

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
            
            # Generate output filename with chapter range
            safe_title = "".join(c for c in epub_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            chapter_range = selected_run_info.get('range', '')
            if chapter_range:
                # Clean chapter range for filename (Ch.700-705 -> Ch700-705)
                clean_range = chapter_range.replace('.', '')
                epub_filename = f"{safe_title.replace(' ', '_')}_{clean_range}.epub"
            else:
                epub_filename = f"{safe_title.replace(' ', '_')}.epub"
            epub_output_path = os.path.join(epub_output_dir, epub_filename)
            
            # Get novel slug from session state or detect from alignment maps
            novel_slug = st.session_state.get('selected_novel_slug') or detect_novel_slug_from_alignment_maps()
            
            # Create EPUB using new builder
            from utils.epub_builder import build_epub
            
            success, message = build_epub(
                selected_run_info['path'],
                epub_output_path,
                title=epub_title,
                author=epub_author,
                translator=epub_translator,
                novel_slug=novel_slug,
                include_images=True
            )
            
            if success:
                st.success("✅ **EPUB Created Successfully!**")
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