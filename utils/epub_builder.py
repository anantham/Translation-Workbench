"""
EPUB Builder Module

Pure-Python EPUB compiler with no Streamlit dependencies.
Extracts and consolidates EPUB creation logic from Translation Lab.

This module handles:
- Chapter file discovery and ordering
- Novel-specific configuration loading
- Image integration (optional)
- EPUB metadata and structure creation
- Gallery-style image organization
"""

import os
import json
import html
import re
import logging
import warnings
from datetime import datetime
from ebooklib import epub

# Configure logging
logger = logging.getLogger(__name__)

# Chapter number extraction patterns
_CHAPTER_PATTERNS = [
    r'Chapter-(\d+)',        # Chapter-0001.txt, Chapter-0044-translated.txt
    r'Ch(\d+)',             # Ch001.txt, Ch44.txt
    r'^(\d+)\.txt$',        # 001.txt, 044.txt
    r'(\d+)'                # Any number in filename
]

def _extract_chapter_number(filename):
    """Extract chapter number from filename using multiple patterns.
    
    Args:
        filename (str): The filename to parse
        
    Returns:
        tuple: (chapter_number, filename_for_fallback_sort)
    """
    basename = os.path.basename(filename)
    
    # Try each pattern in order
    for pattern in _CHAPTER_PATTERNS:
        match = re.search(pattern, basename)
        if match:
            return (int(match.group(1)), basename.lower())
    
    # No number found - return None to sort alphabetically at end
    return (None, basename.lower())

def _gather_chapter_files(chapter_dir):
    """Gather and sort all .txt files in the chapter directory.
    
    Args:
        chapter_dir (str): Directory containing chapter files
        
    Returns:
        list: Sorted list of (chapter_number, filepath) tuples
    """
    if not os.path.exists(chapter_dir):
        return []
    
    chapter_files = []
    for filename in os.listdir(chapter_dir):
        if filename.lower().endswith('.txt'):
            filepath = os.path.join(chapter_dir, filename)
            chapter_num, sort_key = _extract_chapter_number(filename)
            chapter_files.append((chapter_num, filepath, sort_key))
    
    # Sort: numbered chapters first (by number), then unnumbered (alphabetically)
    def sort_key(item):
        chapter_num, filepath, sort_key = item
        if chapter_num is None:
            return (1, sort_key)  # Unnumbered files after numbered ones
        return (0, chapter_num)
    
    chapter_files.sort(key=sort_key)
    
    # Check for duplicate chapter numbers
    seen_numbers = {}
    for chapter_num, filepath, _ in chapter_files:
        if chapter_num is not None:
            if chapter_num in seen_numbers:
                warnings.warn(f"Duplicate chapter number {chapter_num}: {filepath} and {seen_numbers[chapter_num]}")
            else:
                seen_numbers[chapter_num] = filepath
    
    return [(chapter_num, filepath) for chapter_num, filepath, _ in chapter_files]

def _load_novel_metadata(novel_slug):
    """Load novel-specific metadata from configuration.
    
    Args:
        novel_slug (str): Novel identifier
        
    Returns:
        dict: Novel metadata or empty dict if not found
    """
    if not novel_slug:
        return {}
    
    config_path = os.path.join("data", "novels", novel_slug, "novel_config.json")
    
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config.get('novel_info', {})
    except Exception as e:
        logger.debug(f"Could not load novel metadata for {novel_slug}: {e}")
    
    return {}

def _load_novel_images(novel_slug):
    """Load novel-specific images (simplified version without Streamlit).
    
    Args:
        novel_slug (str): Novel identifier
        
    Returns:
        dict: Image information or empty dict if not found
    """
    if not novel_slug:
        return {}
    
    # For v1, return empty dict - image support to be added later
    # TODO: Implement image loading from illustrations_manifest.json
    return {}

def _convert_text_to_html(text):
    """Convert plain text to HTML with basic formatting.
    
    Args:
        text (str): Plain text content
        
    Returns:
        str: HTML-formatted content
    """
    # For v1, use simple pre-formatted text
    # TODO: Add markdown-to-HTML conversion
    return f"<pre>{html.escape(text)}</pre>"

def _load_metadata_template():
    """Load the EPUB metadata template from file.
    
    Returns:
        str: Template content with placeholders
    """
    template_path = "epub_metadata_template.md"
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # Fallback template if file missing
        return """
# Translation Information

## 📊 Performance Analytics
- **Total Time**: {total_time}
- **API Cost**: ${estimated_cost_usd}
- **Processing Speed**: {chapters_per_minute} chapters/minute

## 🤖 AI Configuration
- **Model**: {model_name}
- **Translation Strategy**: {translation_style}

## 📖 Content Information
- **Novel**: {novel_title} by {original_author}
- **Chapters**: {chapter_range} ({total_chapters} chapters)
- **Languages**: {source_language} → {target_language}

## 🔗 Project Information
- **Framework**: {framework_name}
- **Generated**: {generation_date}

---
*Generated by {framework_name} v{framework_version} on {generation_date}*
"""

def _find_job_metadata(chapter_dir):
    """Find job metadata file associated with chapter directory.
    
    Args:
        chapter_dir (str): Directory containing chapters
        
    Returns:
        dict: Job metadata or empty dict if not found
    """
    # Look for job_metadata.json in the chapter directory
    metadata_path = os.path.join(chapter_dir, "job_metadata.json")
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"Could not load job metadata: {e}")
    
    return {}

def _calculate_performance_metrics(job_metadata, chapter_count):
    """Calculate performance metrics from job metadata.
    
    Args:
        job_metadata (dict): Job metadata from translation run
        chapter_count (int): Number of chapters processed
        
    Returns:
        dict: Calculated performance metrics
    """
    metrics = {
        'total_time': 'Unknown',
        'estimated_cost_usd': '0.00',
        'chapters_per_minute': 'Unknown',
        'avg_time_per_chapter': 'Unknown',
        'api_calls_made': 'Unknown',
        'start_timestamp': 'Unknown',
        'end_timestamp': 'Unknown'
    }
    
    # Extract performance data
    performance = job_metadata.get('performance_metrics', {})
    api_usage = job_metadata.get('api_usage', {})
    
    if performance:
        metrics['total_time'] = performance.get('total_time_elapsed', 'Unknown')
        metrics['start_timestamp'] = performance.get('start_timestamp', 'Unknown')
        metrics['end_timestamp'] = performance.get('end_timestamp', 'Unknown')
        
        # Calculate chapters per minute
        if metrics['total_time'] != 'Unknown' and isinstance(metrics['total_time'], str):
            if metrics['total_time'].endswith('s'):
                try:
                    total_seconds = float(metrics['total_time'].rstrip('s'))
                    if total_seconds > 0:
                        metrics['chapters_per_minute'] = round((chapter_count / total_seconds) * 60, 2)
                        metrics['avg_time_per_chapter'] = f"{total_seconds / chapter_count:.1f}s"
                except (ValueError, ZeroDivisionError):
                    pass
    
    if api_usage:
        metrics['estimated_cost_usd'] = f"{api_usage.get('estimated_cost_usd', 0):.4f}"
        metrics['api_calls_made'] = api_usage.get('total_api_calls', 'Unknown')
    
    return metrics

def _format_metadata_template(template, job_metadata, novel_config, chapter_count, novel_slug):
    """Format metadata template with actual values.
    
    Args:
        template (str): Template with placeholders
        job_metadata (dict): Job metadata from translation run
        novel_config (dict): Novel configuration
        chapter_count (int): Number of chapters
        novel_slug (str): Novel identifier
        
    Returns:
        str: Formatted template
    """
    # Get framework branding
    try:
        from utils import get_config_value
        branding = get_config_value('branding', {})
        epub_config = get_config_value('epub_metadata', {})
    except:
        branding = {}
        epub_config = {}
    
    # Calculate performance metrics
    metrics = _calculate_performance_metrics(job_metadata, chapter_count)
    
    # Extract novel information
    novel_info = novel_config.get('novel_info', {})
    
    # Build replacement values
    values = {
        # Performance metrics
        'total_time': metrics['total_time'],
        'estimated_cost_usd': metrics['estimated_cost_usd'],
        'chapters_per_minute': metrics['chapters_per_minute'],
        'avg_time_per_chapter': metrics['avg_time_per_chapter'],
        'api_calls_made': metrics['api_calls_made'],
        
        # AI configuration
        'model_name': job_metadata.get('model_name', 'Unknown'),
        'api_provider': 'OpenAI' if 'gpt' in job_metadata.get('model_name', '').lower() else 'Google AI',
        'model_version': job_metadata.get('model_name', 'Unknown'),
        'translation_style': 'Professional Xianxia Translation',
        'system_prompt_preview': job_metadata.get('system_prompt', '')[:100] + '...' if job_metadata.get('system_prompt') else 'Unknown',
        'example_count': job_metadata.get('history_count', 0),
        'example_strategy': 'Past chapters for context',
        'temperature': job_metadata.get('temperature', 'Unknown'),
        'max_tokens': job_metadata.get('max_tokens', 'Unknown'),
        
        # Content information
        'novel_title': novel_info.get('title', 'Unknown'),
        'english_title': novel_info.get('english_title', 'Unknown'),
        'original_author': novel_info.get('original_author', 'Unknown'),
        'chapter_range': f"Ch.1-{chapter_count}",
        'total_chapters': str(chapter_count),
        'source_language': novel_info.get('source_language', 'Chinese (Simplified)'),
        'target_language': novel_info.get('target_language', 'English'),
        
        # Framework information
        'framework_name': branding.get('framework_name', 'The Lexicon Forge'),
        'framework_version': epub_config.get('project_version', 'v2.2.0'),
        'project_version': epub_config.get('project_version', 'v2.2.0'),
        'license': epub_config.get('license', 'GNU GPL License'),
        'github_url': epub_config.get('github_url', 'https://github.com/anantham/Translation-Workbench'),
        'feature_requests_url': epub_config.get('feature_requests_url', 'https://github.com/anantham/Translation-Workbench/issues'),
        'documentation_url': epub_config.get('documentation_url', 'https://github.com/anantham/Translation-Workbench/wiki'),
        'github_discussions_url': epub_config.get('github_discussions_url', 'https://t.me/webnovels'),
        'license_url': epub_config.get('license_url', 'https://github.com/anantham/Translation-Workbench/blob/main/LICENSE'),
        'translation_philosophy': epub_config.get('translation_philosophy', 'AI-powered framework for consistent, high-quality translation'),
        
        # Technical details
        'start_timestamp': metrics['start_timestamp'],
        'end_timestamp': metrics['end_timestamp'],
        'api_delay': job_metadata.get('api_delay', 'Unknown'),
        'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        
        # Placeholder values for fields we don't have yet
        'word_count_chinese': 'Unknown',
        'word_count_english': 'Unknown',
        'expansion_ratio': 'Unknown',
        'consistency_score': 'Unknown',
        'bleu_score_avg': 'Unknown',
        'semantic_similarity_avg': 'Unknown',
        'human_eval_sample': 'Unknown',
        'terminology_standardization': 'Unknown',
        'input_tokens': 'Unknown',
        'output_tokens': 'Unknown',
        'total_tokens': 'Unknown',
        'input_cost': 'Unknown',
        'output_cost': 'Unknown',
        'cost_per_1k_tokens': 'Unknown',
        'avg_cost_per_chapter': 'Unknown',
        'system_prompt_hash': 'Unknown',
        'maintainer_name': 'Unknown',
        'maintainer_email': 'Unknown'
    }
    
    # Replace placeholders in template
    try:
        return template.format(**values)
    except KeyError as e:
        logger.warning(f"Missing template placeholder: {e}")
        return template

def build_epub(
    chapter_dir: str,
    output_path: str,
    *,
    title: str = "Untitled",
    author: str = "Unknown",
    translator: str = "AI Translation",
    novel_slug: str = None,
    include_images: bool = False,
) -> tuple[bool, str]:
    """
    Build an EPUB file from a directory of chapter files.
    
    Args:
        chapter_dir (str): Directory containing .txt chapter files
        output_path (str): Path where EPUB file will be created
        title (str): Book title
        author (str): Original author name
        translator (str): Translator credit
        novel_slug (str): Novel identifier for metadata loading
        include_images (bool): Whether to include images (v1: not implemented)
        
    Returns:
        tuple: (success: bool, message: str)
    """
    
    # Validation
    if not os.path.exists(chapter_dir):
        return False, f"Chapter directory not found: {chapter_dir}"
    
    if not os.path.isdir(chapter_dir):
        return False, f"Path is not a directory: {chapter_dir}"
    
    # Check output directory is writable
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            return False, f"Cannot create output directory: {e}"
    
    # Gather chapter files
    chapter_files = _gather_chapter_files(chapter_dir)
    
    if not chapter_files:
        return False, f"No .txt files found in {chapter_dir}"
    
    logger.info(f"Found {len(chapter_files)} chapter files")
    
    try:
        # Create EPUB book
        book = epub.EpubBook()
        
        # Load novel metadata if available
        novel_metadata = _load_novel_metadata(novel_slug)
        
        # Load job metadata for analytics
        job_metadata = _find_job_metadata(chapter_dir)
        
        # Set basic metadata
        book.set_identifier(f"{novel_slug or 'custom'}-translation-{datetime.now().strftime('%Y%m%d')}")
        book.set_title(title)
        book.set_language("en")
        book.add_author(author)
        book.add_metadata("DC", "contributor", translator)
        book.add_metadata("DC", "creator", "The Lexicon Forge Translation Framework")
        
        # Add novel-specific metadata
        for genre in novel_metadata.get('genre', []):
            book.add_metadata("DC", "subject", genre)
        
        if description := novel_metadata.get('description'):
            book.add_metadata("DC", "description", description)
        
        # Process chapters
        chapters = []
        spine = []
        
        for i, (chapter_num, filepath) in enumerate(chapter_files, 1):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Determine chapter title
                if chapter_num is not None:
                    chapter_title = f"Chapter {chapter_num}"
                else:
                    # Use filename for unnumbered files
                    chapter_title = f"Chapter {i}"
                
                # Convert content to HTML
                html_content = f"<h1>{chapter_title}</h1>\n{_convert_text_to_html(content)}"
                
                # Create EPUB chapter
                chapter = epub.EpubHtml(
                    title=chapter_title,
                    file_name=f"chapter_{i:04d}.xhtml",
                    lang="en",
                    content=html_content
                )
                
                book.add_item(chapter)
                chapters.append(chapter)
                spine.append(chapter)
                
            except Exception as e:
                logger.error(f"Error processing chapter {filepath}: {e}")
                continue
        
        if not chapters:
            return False, "No chapters could be processed successfully"
        
        # Create comprehensive metadata section
        metadata_chapter = None
        if job_metadata or novel_metadata:
            try:
                # Load and format metadata template
                template = _load_metadata_template()
                formatted_metadata = _format_metadata_template(
                    template, job_metadata, {'novel_info': novel_metadata}, 
                    len(chapters), novel_slug
                )
                
                # Convert markdown to HTML (basic conversion for now)
                # Replace markdown headers with HTML headers
                metadata_html = formatted_metadata
                metadata_html = metadata_html.replace('# ', '<h1>').replace('\n', '</h1>\n', 1)
                metadata_html = metadata_html.replace('## ', '<h2>').replace('\n', '</h2>\n')
                metadata_html = metadata_html.replace('### ', '<h3>').replace('\n', '</h3>\n')
                metadata_html = metadata_html.replace('**', '<strong>').replace('**', '</strong>')
                metadata_html = metadata_html.replace('*', '<em>').replace('*', '</em>')
                metadata_html = metadata_html.replace('\n', '<br>\n')
                
                # Create metadata chapter
                metadata_chapter = epub.EpubHtml(
                    title="📊 Translation Report",
                    file_name="metadata_report.xhtml",
                    lang="en",
                    content=f"""
                    <div style='font-family: serif; line-height: 1.6; margin: 2em;'>
                        <h1>📊 Translation Report</h1>
                        {metadata_html}
                    </div>
                    """
                )
                
                book.add_item(metadata_chapter)
                logger.info("Added comprehensive metadata section to EPUB")
                
            except Exception as e:
                logger.warning(f"Could not create metadata section: {e}")
        
        # Organize chapters: story chapters first, then metadata
        story_chapters = chapters.copy()
        all_chapters = story_chapters.copy()
        spine_items = spine.copy()
        
        if metadata_chapter:
            all_chapters.append(metadata_chapter)
            spine_items.append(metadata_chapter)
        
        # Set up navigation
        book.toc = tuple(all_chapters)
        book.spine = ["nav"] + spine_items
        
        # Add required navigation files
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        
        # Add CSS for basic styling
        nav_css = epub.EpubItem(
            uid="nav_css",
            file_name="style/nav.css",
            media_type="text/css",
            content="""
            body { font-family: serif; margin: 2em; }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 0.5em; }
            pre { white-space: pre-wrap; line-height: 1.6; }
            """
        )
        book.add_item(nav_css)
        
        # Write EPUB file
        epub.write_epub(output_path, book, {})
        
        logger.info(f"EPUB created successfully: {output_path}")
        chapter_count = len(story_chapters)
        if metadata_chapter:
            return True, f"EPUB created with {chapter_count} chapters + metadata report"
        else:
            return True, f"EPUB created with {chapter_count} chapters"
        
    except Exception as e:
        logger.error(f"Error creating EPUB: {e}")
        return False, f"EPUB creation failed: {str(e)}"

# Future expansion functions (stubs for v2)
def _add_images_to_epub(book, novel_images):
    """Add images to EPUB book (to be implemented)."""
    # TODO: Implement image integration
    pass

def _create_gallery_sections(book, novel_images):
    """Create gallery sections for images (to be implemented)."""
    # TODO: Implement gallery creation
    pass