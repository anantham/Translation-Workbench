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
    """Load novel-specific images with comprehensive metadata.
    
    Args:
        novel_slug (str): Novel identifier
        
    Returns:
        dict: Complete image information with file paths and metadata
    """
    if not novel_slug:
        return {}
    
    try:
        # Load novel configuration to get images path
        novel_config = _load_novel_metadata(novel_slug)
        
        # Try to load images configuration
        config_path = os.path.join("data", "novels", novel_slug, "novel_config.json")
        if not os.path.exists(config_path):
            return {}
            
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = json.load(f)
            
        images_config = full_config.get('images', {})
        images_path = images_config.get('images_path', f'data/images/{novel_slug}/')
        manifest_file = images_config.get('manifest_file', 'illustrations_manifest.json')
        
        # Load illustrations manifest
        manifest_path = os.path.join(images_path, manifest_file)
        if not os.path.exists(manifest_path):
            logger.debug(f"No illustrations manifest found at {manifest_path}")
            return {}
            
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
            
        # Build comprehensive image data structure
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
                    'artist': cover_info.get('artist', 'Unknown'),
                    'placement': cover_info.get('placement', 'gallery_cover')
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
                    'placement': illustration_info.get('placement', 'gallery_chapter')
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
                    'caption': illustration_info.get('caption', ''),
                    'placement': illustration_info.get('placement', 'gallery_world')
                }
        
        # Load common branding images dynamically
        try:
            common_images_dir = "data/images/common"
            if os.path.exists(common_images_dir):
                # Get all image files in common directory
                image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
                
                for filename in os.listdir(common_images_dir):
                    if filename.lower() == 'readme.md':
                        continue  # Skip README
                        
                    file_path = os.path.join(common_images_dir, filename)
                    if os.path.isfile(file_path):
                        ext = os.path.splitext(filename)[1].lower()
                        if ext in image_extensions:
                            # Generate key from filename
                            key = os.path.splitext(filename)[0].lower().replace(' ', '_').replace('-', '_')
                            
                            # Create title from filename
                            title = os.path.splitext(filename)[0].replace('_', ' ').title()
                            
                            # Special handling for known framework images
                            if 'cover' in filename.lower():
                                title = "Framework Logo"
                                description = "Official framework logo representing the bridge between languages"
                            elif 'seal' in filename.lower():
                                title = "Artisan's Seal"
                                description = "Quality seal marking each translated work"
                            elif 'pluralistic' in filename.lower() or 'dao' in filename.lower():
                                title = "Pluralistic Dao"
                                description = "Visual representation of the translation process"
                            elif 'elixir' in filename.lower() or 'crucible' in filename.lower():
                                title = "The Elixir Crucible"
                                description = "Pill refining imagery representing the refinement of translation quality"
                            elif 'omniscient' in filename.lower() or 'scriptorium' in filename.lower():
                                title = "The Omniscient Scriptorium"
                                description = "Sacred repository of all knowledge and translation wisdom"
                            elif 'metric' in filename.lower():
                                metric_num = ''.join(filter(str.isdigit, filename))
                                title = f"Quality Metric {metric_num}"
                                description = f"Translation quality visualization metric {metric_num}"
                            else:
                                description = f"Framework asset: {title}"
                            
                            loaded_images['common_branding'][key] = {
                                'file_path': file_path,
                                'title': title,
                                'description': description
                            }
                
                logger.info(f"Dynamically loaded {len(loaded_images['common_branding'])} common branding images")
        except Exception as e:
            logger.debug(f"Could not load common branding images: {e}")
        
        logger.info(f"Loaded {len(loaded_images['chapter_illustrations'])} chapter illustrations, "
                   f"{len(loaded_images['appendix_illustrations'])} appendix illustrations")
        
        return loaded_images
        
    except Exception as e:
        logger.debug(f"Could not load novel images for {novel_slug}: {e}")
        return {}

def _convert_text_to_html(text):
    """Convert plain text to HTML with comprehensive formatting using tokenization.
    
    This function processes:
    - Chapter titles and headers
    - Footnotes with superscript numbers
    - Emphasis and formatting
    - Paragraph breaks
    - Special characters and symbols
    - Dialogue and narrative sections
    
    Uses tokenization approach to avoid HTML conflicts.
    
    Args:
        text (str): Plain text content
        
    Returns:
        str: HTML-formatted content
    """
    if not text or not text.strip():
        return ""
    
    # Token management for conflict-free HTML generation
    token_map = {}
    token_counter = 0
    
    def _create_token(tag_with_attrs, contents):
        nonlocal token_counter
        token_id = f"@@T{token_counter}@@"
        token_counter += 1
        # Extract tag name for closing tag
        tag_name = tag_with_attrs.split()[0]
        token_map[token_id] = f"<{tag_with_attrs}>{contents}</{tag_name}>"
        return token_id
    
    # Escape HTML characters first
    text = html.escape(text)
    
    # Process line by line for better control
    lines = text.split('\n')
    processed_lines = []
    in_footnotes = False
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines (will be handled as paragraph breaks)
        if not line:
            processed_lines.append("")
            continue
            
        # Chapter title (first line starting with "Chapter")
        if line.startswith("Chapter ") and ":" in line:
            # Extract chapter number and title
            chapter_match = re.match(r'Chapter (\d+):\s*(.+)', line)
            if chapter_match:
                chapter_num = chapter_match.group(1)
                chapter_title = chapter_match.group(2)
                processed_lines.append(f'<h1 class="chapter-title">Chapter {chapter_num}: {chapter_title}</h1>')
            else:
                processed_lines.append(f'<h1 class="chapter-title">{line}</h1>')
            continue
        
        # Detect footnotes section
        if line.startswith("---"):
            in_footnotes = True
            processed_lines.append('<hr class="footnotes-separator">')
            continue
        elif line.startswith("Footnotes:"):
            in_footnotes = True
            processed_lines.append('<h3 class="footnotes-title">Footnotes</h3>')
            continue
        
        # Process footnotes
        if in_footnotes and re.match(r'^[¹²³⁴⁵⁶⁷⁸⁹⁰]+\s', line):
            # Extract footnote number and content
            footnote_match = re.match(r'^([¹²³⁴⁵⁶⁷⁸⁹⁰]+)\s+(.+)', line)
            if footnote_match:
                footnote_num = footnote_match.group(1)
                footnote_content = footnote_match.group(2)
                processed_lines.append(f'<div class="footnote"><span class="footnote-number">{footnote_num}</span> {footnote_content}</div>')
            else:
                processed_lines.append(f'<div class="footnote">{line}</div>')
            continue
        
        # Regular text processing with tokenization
        processed_line = line
        
        # Handle em-dashes and special punctuation FIRST (before tokenization)
        processed_line = processed_line.replace('—', '&mdash;')
        processed_line = processed_line.replace('–', '&ndash;')
        processed_line = processed_line.replace('…', '&hellip;')
        
        # TOKENIZATION PHASE - Process in order of specificity
        
        # 1. Bold-italic first (most specific)
        processed_line = re.sub(r'\*\*\*([^*]+)\*\*\*', 
                               lambda m: _create_token('strong', _create_token('em', m.group(1))), 
                               processed_line)
        
        # 2. Bold text (before single asterisks)
        processed_line = re.sub(r'\*\*([^*]+)\*\*', 
                               lambda m: _create_token('strong', m.group(1)), 
                               processed_line)
        processed_line = re.sub(r'__([^_]+)__', 
                               lambda m: _create_token('strong', m.group(1)), 
                               processed_line)
        
        # 3. Italic text
        processed_line = re.sub(r'\*([^*]+)\*', 
                               lambda m: _create_token('em', m.group(1)), 
                               processed_line)
        processed_line = re.sub(r'_([^_]+)_', 
                               lambda m: _create_token('em', m.group(1)), 
                               processed_line)
        
        # 4. Footnote references
        processed_line = re.sub(r'([¹²³⁴⁵⁶⁷⁸⁹⁰]+)', 
                               lambda m: _create_token('sup class="footnote-ref"', m.group(1)), 
                               processed_line)
        
        # 5. Cultivation terms (before general dialogue)
        processed_line = re.sub(r'\b(Divine Realm|Godly Transformation|Yellow Springs|Blood Pellet)\b', 
                               lambda m: _create_token('span class="cultivation-term"', m.group(1)), 
                               processed_line)
        
        # 6. Chinese terms with explanations
        def _process_chinese_term(match):
            term = match.group(1)
            explanation = match.group(2)
            term_token = _create_token('span class="term"', term)
            explanation_token = _create_token('span class="explanation"', f"({explanation})")
            return f'{term_token} {explanation_token}'
        
        processed_line = re.sub(r'(\w+)\s*\(([^)]+)\)', _process_chinese_term, processed_line)
        
        # 7. Dialogue formatting (last to avoid conflicts)
        processed_line = re.sub(r'"([^"]+)"', 
                               lambda m: _create_token('span class="dialogue"', f'"{m.group(1)}"'), 
                               processed_line)
        
        processed_lines.append(processed_line)
    
    # Join lines and create paragraphs
    html_content = []
    current_paragraph = []
    
    for line in processed_lines:
        if line == "":
            # Empty line - end current paragraph
            if current_paragraph:
                paragraph_text = " ".join(current_paragraph)
                if paragraph_text.strip():
                    # Don't wrap headers, footnotes, or other special elements in paragraphs
                    if not (paragraph_text.startswith('<h') or paragraph_text.startswith('<hr') or 
                           paragraph_text.startswith('<div class="footnote">') or paragraph_text.startswith('<h3')):
                        html_content.append(f'<p>{paragraph_text}</p>')
                    else:
                        html_content.append(paragraph_text)
                current_paragraph = []
        else:
            current_paragraph.append(line)
    
    # Handle any remaining paragraph
    if current_paragraph:
        paragraph_text = " ".join(current_paragraph)
        if paragraph_text.strip():
            if not (paragraph_text.startswith('<h') or paragraph_text.startswith('<hr') or 
                   paragraph_text.startswith('<div class="footnote">') or paragraph_text.startswith('<h3')):
                html_content.append(f'<p>{paragraph_text}</p>')
            else:
                html_content.append(paragraph_text)
    
    # Replace all tokens with their HTML equivalents
    # Handle nested tokens by repeatedly replacing until no more tokens exist
    html_result = "\n".join(html_content)
    
    # Keep replacing tokens until no more @@T tokens exist
    max_iterations = 10  # Safety limit
    iteration = 0
    
    while "@@T" in html_result and iteration < max_iterations:
        for token_id, html_fragment in token_map.items():
            html_result = html_result.replace(token_id, html_fragment)
        iteration += 1
    
    return html_result

def _create_advanced_toc(story_chapters, gallery_chapters, metadata_chapter):
    """Create advanced hierarchical Table of Contents with sections and navigation.
    
    Args:
        story_chapters (list): List of story chapter objects
        gallery_chapters (list): List of gallery chapter objects
        metadata_chapter: Metadata chapter object or None
        
    Returns:
        tuple: Hierarchical TOC structure for EPUB
    """
    from ebooklib import epub
    
    toc_structure = []
    
    # MAIN CONTENT SECTION
    if story_chapters:
        # Group chapters by ranges for better organization
        chapter_groups = _group_chapters_by_range(story_chapters)
        
        if len(chapter_groups) > 1:
            # Multiple groups - create hierarchical structure
            main_section = (
                epub.Section("📚 Main Story"),
                []
            )
            
            for group_title, group_chapters in chapter_groups:
                chapter_section = (
                    epub.Section(group_title),
                    group_chapters
                )
                main_section[1].append(chapter_section)
            
            toc_structure.append(main_section)
        else:
            # Single group - flat structure with section header
            main_section = (
                epub.Section("📚 Main Story"),
                story_chapters
            )
            toc_structure.append(main_section)
    
    # GALLERY SECTIONS
    if gallery_chapters:
        gallery_section = (
            epub.Section("🎨 Visual Galleries"),
            gallery_chapters
        )
        toc_structure.append(gallery_section)
    
    # METADATA SECTION
    if metadata_chapter:
        metadata_section = (
            epub.Section("📊 Translation Analytics"),
            [metadata_chapter]
        )
        toc_structure.append(metadata_section)
    
    return tuple(toc_structure)

def _group_chapters_by_range(chapters, chapters_per_group=25):
    """Group chapters into ranges for hierarchical TOC organization.
    
    Args:
        chapters (list): List of chapter objects
        chapters_per_group (int): Number of chapters per group
        
    Returns:
        list: List of (group_title, group_chapters) tuples
    """
    total_chapters = len(chapters)
    
    if total_chapters <= chapters_per_group:
        return [(f"📖 All Chapters (1-{total_chapters})", chapters)]
    
    groups = []
    for i in range(0, total_chapters, chapters_per_group):
        group_chapters = chapters[i:i + chapters_per_group]
        start_num = i + 1
        end_num = min(i + chapters_per_group, total_chapters)
        
        # Calculate progress percentage
        progress = round((end_num / total_chapters) * 100)
        
        # Create more descriptive group titles
        if i == 0:
            group_title = f"📖 Opening Arc (Chapters {start_num}-{end_num}) • {progress}%"
        elif end_num == total_chapters:
            group_title = f"📖 Final Arc (Chapters {start_num}-{end_num}) • Complete"
        else:
            arc_num = (i // chapters_per_group) + 1
            group_title = f"📖 Arc {arc_num} (Chapters {start_num}-{end_num}) • {progress}%"
        
        groups.append((group_title, group_chapters))
    
    return groups

def _create_branded_title_page(title, author, translator, novel_metadata, novel_images):
    """Create a custom title page with framework branding and styling.
    
    Args:
        title (str): Book title
        author (str): Original author
        translator (str): Translator name
        novel_metadata (dict): Novel metadata
        novel_images (dict): Novel images including cover
        
    Returns:
        EpubHtml: Branded title page chapter
    """
    from ebooklib import epub
    
    # Get cover image if available
    cover_image_html = ""
    if novel_images.get('cover'):
        cover_data = novel_images['cover']
        # Note: This assumes the cover image has already been added to the book
        cover_image_html = '''
        <div class="title-cover">
            <img src="images/cover.png" alt="Cover" class="cover-image" />
        </div>
        '''
    
    # Get framework branding image if available
    framework_logo_html = ""
    if novel_images.get('common_branding', {}).get('cover'):
        framework_logo_html = '''
        <div class="framework-logo">
            <img src="images/framework_logo.png" alt="The Lexicon Forge" class="logo-image" />
        </div>
        '''
    
    # Extract metadata for display
    english_title = novel_metadata.get('english_title', title)
    original_title = novel_metadata.get('title', '')
    genres = ', '.join(novel_metadata.get('genre', []))
    description = novel_metadata.get('description', '')
    
    # Create comprehensive title page HTML
    title_page_html = f'''
    <!DOCTYPE html>
    <html xmlns="http://www.w3.org/1999/xhtml">
    <head>
        <title>Title Page</title>
        <link rel="stylesheet" type="text/css" href="style/nav.css"/>
        <style>
            .title-page {{
                text-align: center;
                padding: 2em;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                justify-content: center;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                font-family: 'Times New Roman', serif;
            }}
            
            .title-cover {{
                margin: 0 auto 2em auto;
                max-width: 300px;
            }}
            
            .cover-image {{
                width: 100%;
                max-width: 250px;
                height: auto;
                border-radius: 10px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            }}
            
            .main-title {{
                font-size: 2.5em;
                font-weight: bold;
                margin: 0.5em 0;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
                line-height: 1.2;
            }}
            
            .original-title {{
                font-size: 1.5em;
                font-style: italic;
                margin: 0.5em 0;
                opacity: 0.9;
                color: #f0f0f0;
            }}
            
            .author-info {{
                margin: 1.5em 0;
                font-size: 1.1em;
            }}
            
            .author-name {{
                font-weight: bold;
                color: #ffd700;
            }}
            
            .translator-info {{
                margin: 1em 0;
                font-size: 1em;
                color: #e0e0e0;
            }}
            
            .genre-info {{
                margin: 1em 0;
                font-size: 0.9em;
                color: #d0d0d0;
                font-style: italic;
            }}
            
            .description {{
                margin: 1.5em auto;
                max-width: 600px;
                font-size: 0.9em;
                line-height: 1.6;
                color: #f0f0f0;
                text-align: justify;
            }}
            
            .framework-branding {{
                margin-top: 2em;
                padding-top: 2em;
                border-top: 1px solid rgba(255,255,255,0.3);
            }}
            
            .framework-logo {{
                margin: 1em 0;
            }}
            
            .logo-image {{
                width: 150px;
                height: auto;
                opacity: 0.8;
            }}
            
            .framework-name {{
                font-size: 1.2em;
                font-weight: bold;
                margin: 0.5em 0;
                color: #ffd700;
            }}
            
            .framework-tagline {{
                font-size: 0.9em;
                color: #e0e0e0;
                font-style: italic;
            }}
            
            .creation-date {{
                margin-top: 2em;
                font-size: 0.8em;
                color: #c0c0c0;
            }}
        </style>
    </head>
    <body>
        <div class="title-page">
            {cover_image_html}
            
            <h1 class="main-title">{english_title}</h1>
            
            {f'<div class="original-title">{original_title}</div>' if original_title and original_title != english_title else ''}
            
            <div class="author-info">
                <div>By <span class="author-name">{author}</span></div>
            </div>
            
            <div class="translator-info">
                Translated by {translator}
            </div>
            
            {f'<div class="genre-info">{genres}</div>' if genres else ''}
            
            {f'<div class="description">{description[:300]}{"..." if len(description) > 300 else ""}</div>' if description else ''}
            
            <div class="framework-branding">
                {framework_logo_html}
                
                <div class="framework-name">The Lexicon Forge</div>
                <div class="framework-tagline">Bridging Languages, Preserving Stories</div>
                
                <div class="creation-date">
                    Created {datetime.now().strftime('%B %Y')}
                </div>
            </div>
        </div>
    </body>
    </html>
    '''
    
    # Create the title page chapter
    title_page = epub.EpubHtml(
        title="Title Page",
        file_name="title_page.xhtml",
        lang="en",
        content=title_page_html
    )
    
    return title_page

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
# Translation Analytics Report

## 📊 Performance Analytics
- **Total Processing Time**: {total_time}
- **Average Time per Chapter**: {avg_time_per_chapter}
- **Processing Speed**: {chapters_per_minute} chapters/minute
- **Efficiency Grade**: {efficiency_grade}
- **Chapters Completed**: {chapters_completed}/{total_chapters}

## 💰 Cost Analytics
- **Total API Cost**: ${estimated_cost_usd}
- **Cost per 1K Tokens**: ${cost_per_1k_tokens:.4f}
- **Cost Efficiency Grade**: {cost_grade}
- **Average Cost per Chapter**: ${avg_cost_per_chapter:.2f}

## 🔤 Token Usage Analytics
- **Input Tokens**: {input_tokens:,}
- **Output Tokens**: {output_tokens:,}
- **Total Tokens**: {total_tokens_used:,}
- **Expansion Ratio**: {expansion_ratio}x
- **Expansion Grade**: {expansion_grade}

## 🧠 AI Configuration
- **Model**: {model_name} ({api_provider})
- **Temperature**: {temperature} ({creativity_level})
- **Max Tokens**: {max_tokens:,} ({context_level})
- **Translation Complexity**: {complexity_score}/10
- **Quality Indicators**: {quality_indicators}

## 📖 Content Information
- **Novel**: {novel_title} by {original_author}
- **English Title**: {english_title}
- **Chapter Range**: {chapter_range} ({total_chapters} chapters)
- **Languages**: {source_language} → {target_language}
- **Translation Style**: {translation_style}

## 🎯 Translation Strategy
- **System Prompt Preview**: {system_prompt_preview}
- **Context Strategy**: {example_strategy}
- **History Count**: {example_count} chapters
- **API Delay**: {api_delay}s between requests

## 🔗 Framework Information
- **Framework**: {framework_name} v{framework_version}
- **Project License**: {license}
- **Generated**: {generation_date}
- **Start Time**: {start_timestamp}
- **End Time**: {end_timestamp}

## 📈 Quality Metrics
- **Translation Philosophy**: {translation_philosophy}
- **Consistency Score**: {consistency_score}
- **Terminology Standardization**: {terminology_standardization}

## 🌐 Resources
- **GitHub**: {github_url}
- **Documentation**: {documentation_url}
- **Feature Requests**: {feature_requests_url}
- **Community**: {github_discussions_url}

---
*Generated by {framework_name} v{framework_version} on {generation_date}*

**Translation powered by The Lexicon Forge - Bridging Languages, Preserving Stories**
"""

def _find_job_metadata(chapter_dir):
    """Find job metadata file associated with chapter directory.
    
    Args:
        chapter_dir (str): Directory containing chapters
        
    Returns:
        dict: Enhanced job metadata with analytics
    """
    # Look for job_metadata.json in the chapter directory
    metadata_path = os.path.join(chapter_dir, "job_metadata.json")
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                job_metadata = json.load(f)
                
                # Enhance metadata with analytics
                enhanced_metadata = _enhance_job_metadata(job_metadata)
                return enhanced_metadata
                
        except Exception as e:
            logger.debug(f"Could not load job metadata: {e}")
    
    return {}

def _enhance_job_metadata(job_metadata):
    """Enhance job metadata with calculated analytics.
    
    Args:
        job_metadata (dict): Raw job metadata
        
    Returns:
        dict: Enhanced metadata with analytics
    """
    enhanced = job_metadata.copy()
    
    # Calculate performance analytics
    performance = job_metadata.get('performance_metrics', {})
    api_usage = job_metadata.get('api_usage', {})
    
    # Processing efficiency metrics
    if performance.get('avg_time_per_chapter'):
        avg_time = float(performance['avg_time_per_chapter'].replace('s', ''))
        enhanced['efficiency_grade'] = _calculate_efficiency_grade(avg_time)
    
    # Cost efficiency metrics
    if api_usage.get('estimated_cost_usd') and api_usage.get('total_tokens_used'):
        cost_per_1k_tokens = (api_usage['estimated_cost_usd'] / api_usage['total_tokens_used']) * 1000
        enhanced['cost_efficiency'] = cost_per_1k_tokens
        enhanced['cost_grade'] = _calculate_cost_grade(cost_per_1k_tokens)
    
    # Token usage analytics
    if api_usage.get('input_tokens') and api_usage.get('output_tokens'):
        input_tokens = api_usage['input_tokens']
        output_tokens = api_usage['output_tokens']
        expansion_ratio = output_tokens / input_tokens if input_tokens > 0 else 0
        enhanced['expansion_ratio'] = round(expansion_ratio, 2)
        enhanced['expansion_grade'] = _calculate_expansion_grade(expansion_ratio)
    
    # Quality indicators
    ai_config = job_metadata.get('ai_configuration', {})
    if ai_config.get('temperature'):
        enhanced['quality_indicators'] = _analyze_quality_indicators(ai_config)
    
    # Translation complexity score
    system_prompt = job_metadata.get('system_prompt', '')
    enhanced['complexity_score'] = _calculate_complexity_score(system_prompt)
    
    return enhanced

def _calculate_efficiency_grade(avg_time_seconds):
    """Calculate efficiency grade based on processing time."""
    if avg_time_seconds < 30:
        return "A+ (Excellent)"
    elif avg_time_seconds < 60:
        return "A (Very Good)"
    elif avg_time_seconds < 90:
        return "B (Good)"
    elif avg_time_seconds < 120:
        return "C (Average)"
    else:
        return "D (Slow)"

def _calculate_cost_grade(cost_per_1k_tokens):
    """Calculate cost efficiency grade."""
    if cost_per_1k_tokens < 0.002:
        return "A+ (Very Economical)"
    elif cost_per_1k_tokens < 0.005:
        return "A (Economical)"
    elif cost_per_1k_tokens < 0.010:
        return "B (Reasonable)"
    elif cost_per_1k_tokens < 0.020:
        return "C (Expensive)"
    else:
        return "D (Very Expensive)"

def _calculate_expansion_grade(expansion_ratio):
    """Calculate expansion ratio grade."""
    if 0.8 <= expansion_ratio <= 1.2:
        return "A+ (Optimal)"
    elif 0.6 <= expansion_ratio <= 1.5:
        return "A (Good)"
    elif 0.4 <= expansion_ratio <= 2.0:
        return "B (Acceptable)"
    elif 0.2 <= expansion_ratio <= 3.0:
        return "C (Suboptimal)"
    else:
        return "D (Poor)"

def _analyze_quality_indicators(ai_config):
    """Analyze AI configuration for quality indicators."""
    indicators = []
    
    temperature = ai_config.get('temperature', 0.7)
    if temperature >= 0.8:
        indicators.append("High Creativity")
    elif temperature >= 0.5:
        indicators.append("Balanced Creativity")
    else:
        indicators.append("Low Creativity")
    
    max_tokens = ai_config.get('max_tokens', 4096)
    if max_tokens >= 8000:
        indicators.append("Extended Context")
    elif max_tokens >= 4000:
        indicators.append("Standard Context")
    else:
        indicators.append("Limited Context")
    
    model = ai_config.get('model_version', '')
    if 'gpt-4' in model.lower():
        indicators.append("Premium Model")
    elif 'gpt-3.5' in model.lower():
        indicators.append("Standard Model")
    else:
        indicators.append("Basic Model")
    
    return indicators

def _calculate_complexity_score(system_prompt):
    """Calculate translation complexity score based on system prompt."""
    if not system_prompt:
        return 1
    
    complexity_keywords = [
        'cultural', 'context', 'nuance', 'metaphor', 'idiom', 'imagery',
        'emotional', 'dialogue', 'style', 'voice', 'fidelity', 'creative',
        'footnote', 'terminology', 'sophisticated', 'descriptive'
    ]
    
    score = 1
    for keyword in complexity_keywords:
        if keyword.lower() in system_prompt.lower():
            score += 0.5
    
    return min(score, 10)  # Cap at 10

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
        metrics['api_calls_made'] = api_usage.get('api_calls_made', chapter_count)  # Use chapter_count as fallback
    
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
    
    # Extract enhanced analytics from job metadata
    performance = job_metadata.get('performance_metrics', {})
    api_usage = job_metadata.get('api_usage', {})
    ai_config = job_metadata.get('ai_configuration', {})
    
    # Build replacement values with enhanced analytics
    values = {
        # Performance analytics
        'total_time': metrics['total_time'],
        'avg_time_per_chapter': metrics['avg_time_per_chapter'],
        'chapters_per_minute': metrics['chapters_per_minute'],
        'efficiency_grade': job_metadata.get('efficiency_grade', 'Unknown'),
        'chapters_completed': performance.get('chapters_completed', chapter_count),
        'total_chapters': str(chapter_count),
        
        # Cost analytics
        'estimated_cost_usd': metrics['estimated_cost_usd'],
        'cost_per_1k_tokens': job_metadata.get('cost_efficiency', 0),
        'cost_grade': job_metadata.get('cost_grade', 'Unknown'),
        'avg_cost_per_chapter': api_usage.get('estimated_cost_usd', 0) / chapter_count if chapter_count > 0 else 0,
        
        # Token usage analytics
        'input_tokens': api_usage.get('input_tokens', 0),
        'output_tokens': api_usage.get('output_tokens', 0),
        'total_tokens_used': api_usage.get('total_tokens_used', 0),
        'expansion_ratio': job_metadata.get('expansion_ratio', 'Unknown'),
        'expansion_grade': job_metadata.get('expansion_grade', 'Unknown'),
        
        # AI configuration
        'model_name': job_metadata.get('model_name', 'Unknown'),
        'api_provider': ai_config.get('api_provider', 'OpenAI'),
        'temperature': ai_config.get('temperature', 'Unknown'),
        'creativity_level': _get_creativity_level(ai_config.get('temperature', 0.7)),
        'max_tokens': ai_config.get('max_tokens', 'Unknown'),
        'context_level': _get_context_level(ai_config.get('max_tokens', 4096)),
        'complexity_score': job_metadata.get('complexity_score', 'Unknown'),
        'quality_indicators': ', '.join(job_metadata.get('quality_indicators', [])),
        
        # Content information
        'novel_title': novel_info.get('title', 'Unknown'),
        'english_title': novel_info.get('english_title', 'Unknown'),
        'original_author': novel_info.get('original_author', 'Unknown'),
        'chapter_range': f"Ch.1-{chapter_count}",
        'source_language': novel_info.get('source_language', 'Chinese (Simplified)'),
        'target_language': novel_info.get('target_language', 'English'),
        'translation_style': 'Professional Xianxia Translation',
        
        # Translation strategy
        'system_prompt_preview': job_metadata.get('system_prompt', '')[:100] + '...' if job_metadata.get('system_prompt') else 'Unknown',
        'example_strategy': ai_config.get('example_strategy', 'rolling_context_window'),
        'example_count': job_metadata.get('history_count', 0),
        'api_delay': job_metadata.get('api_delay', 'Unknown'),
        'api_calls_made': api_usage.get('api_calls_made', chapter_count),
        'model_version': ai_config.get('model_version', job_metadata.get('model_name', 'Unknown')),
        'system_prompt_hash': ai_config.get('system_prompt_hash', 'Unknown'),
        
        # Content metrics
        'word_count_chinese': 'Not yet measured',
        'word_count_english': 'Not yet measured',
        
        # Quality metrics
        'bleu_score_avg': 'Not yet measured',
        'semantic_similarity_avg': 'Not yet measured', 
        'human_eval_sample': 'Not yet measured',
        
        # Cost breakdown
        'input_cost': api_usage.get('cost_breakdown', {}).get('input_cost', api_usage.get('estimated_cost_usd', 0) * 0.75 if api_usage.get('estimated_cost_usd') else 0),
        'output_cost': api_usage.get('cost_breakdown', {}).get('output_cost', api_usage.get('estimated_cost_usd', 0) * 0.25 if api_usage.get('estimated_cost_usd') else 0),
        
        # Framework information
        'framework_name': branding.get('framework_name', 'The Lexicon Forge'),
        'framework_version': epub_config.get('project_version', 'v2.2.0'),
        'license': epub_config.get('license', 'GNU GPL License'),
        'github_url': epub_config.get('github_url', 'https://github.com/anantham/Translation-Workbench'),
        'documentation_url': epub_config.get('documentation_url', 'https://github.com/anantham/Translation-Workbench/wiki'),
        'feature_requests_url': epub_config.get('feature_requests_url', 'https://github.com/anantham/Translation-Workbench/issues'),
        'github_discussions_url': epub_config.get('github_discussions_url', 'https://t.me/webnovels'),
        'translation_philosophy': epub_config.get('translation_philosophy', 'AI-powered framework for consistent, high-quality translation'),
        'maintainer_name': epub_config.get('maintainer_name', 'Anantham'),
        'maintainer_email': epub_config.get('maintainer_email', 'anantham@example.com'),
        'license_url': epub_config.get('license_url', 'https://www.gnu.org/licenses/gpl-3.0.html'),
        'project_version': epub_config.get('project_version', 'v2.2.0'),
        
        # Technical details
        'start_timestamp': metrics['start_timestamp'],
        'end_timestamp': metrics['end_timestamp'],
        'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        
        # Quality metrics (placeholders for future enhancement)
        'consistency_score': 'Not yet measured',
        'terminology_standardization': 'Not yet measured'
    }
    
    # Replace placeholders in template
    try:
        return template.format(**values)
    except KeyError as e:
        logger.warning(f"Missing template placeholder: {e}")
        return template

def _get_creativity_level(temperature):
    """Convert temperature to creativity level description."""
    if temperature >= 0.8:
        return "High Creativity"
    elif temperature >= 0.5:
        return "Balanced"
    else:
        return "Low Creativity"

def _get_context_level(max_tokens):
    """Convert max_tokens to context level description."""
    if max_tokens >= 8000:
        return "Extended Context"
    elif max_tokens >= 4000:
        return "Standard Context"
    else:
        return "Limited Context"

def _add_image_to_epub(book, image_path, image_id):
    """Add an image to the EPUB book.
    
    Args:
        book: EPUB book object
        image_path (str): Path to the image file
        image_id (str): Unique identifier for the image
        
    Returns:
        str: Image filename in EPUB or None if failed
    """
    try:
        # Determine media type based on file extension
        ext = os.path.splitext(image_path)[1].lower()
        media_type_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg', 
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        
        media_type = media_type_map.get(ext, 'image/jpeg')
        
        # Read image file
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        # Create unique filename
        image_filename = f"images/{image_id}{ext}"
        
        # Create EPUB image item
        image_item = epub.EpubItem(
            uid=image_id,
            file_name=image_filename,
            media_type=media_type,
            content=image_data
        )
        
        book.add_item(image_item)
        return image_filename
        
    except Exception as e:
        logger.warning(f"Could not add image {image_path}: {e}")
        return None

def _create_gallery_sections(book, novel_images):
    """Create dedicated gallery sections for different types of images.
    
    Args:
        book: EPUB book object
        novel_images (dict): Loaded novel images
        
    Returns:
        list: List of gallery chapter objects
    """
    gallery_chapters = []
    
    try:
        # Create Cover Gallery
        if novel_images.get('cover'):
            cover_data = novel_images['cover']
            cover_filename = _add_image_to_epub(book, cover_data['file_path'], 'cover')
            
            if cover_filename:
                cover_html = f"""
                <div style="text-align: center; margin: 2em;">
                    <h1>📖 Cover Gallery</h1>
                    <div style="margin: 2em 0;">
                        <img src="{cover_filename}" alt="{cover_data['title']}" style="max-width: 100%; height: auto;" />
                        <h2>{cover_data['title']}</h2>
                        <p><em>{cover_data['description']}</em></p>
                        <p><strong>Artist:</strong> {cover_data.get('artist', 'Unknown')}</p>
                    </div>
                </div>
                """
                
                cover_chapter = epub.EpubHtml(
                    title="📖 Cover Gallery",
                    file_name="gallery_cover.xhtml",
                    lang="en",
                    content=cover_html
                )
                
                book.add_item(cover_chapter)
                gallery_chapters.append(cover_chapter)
        
        # Create Chapter Illustrations Gallery
        chapter_illustrations = novel_images.get('chapter_illustrations', {})
        if chapter_illustrations:
            chapter_gallery_html = "<div style='margin: 2em;'><h1>🎨 Chapter Illustrations</h1>"
            
            # Sort by chapter number
            sorted_chapters = sorted(chapter_illustrations.items(), key=lambda x: int(x[0]))
            
            for chapter_num, img_data in sorted_chapters:
                img_filename = _add_image_to_epub(book, img_data['file_path'], f'chapter_{chapter_num}')
                
                if img_filename:
                    chapter_gallery_html += f"""
                    <div style="margin: 2em 0; text-align: center;">
                        <h2>{img_data['title']}</h2>
                        <img src="{img_filename}" alt="{img_data['title']}" style="max-width: 100%; height: auto;" />
                        <p><em>{img_data['description']}</em></p>
                        <p>{img_data.get('caption', '')}</p>
                    </div>
                    """
            
            chapter_gallery_html += "</div>"
            
            chapter_gallery = epub.EpubHtml(
                title="🎨 Chapter Illustrations",
                file_name="gallery_chapters.xhtml",
                lang="en",
                content=chapter_gallery_html
            )
            
            book.add_item(chapter_gallery)
            gallery_chapters.append(chapter_gallery)
        
        # Create Character & World Gallery
        appendix_illustrations = novel_images.get('appendix_illustrations', {})
        if appendix_illustrations:
            
            # Group by placement type
            character_images = {}
            world_images = {}
            other_images = {}
            
            for key, img_data in appendix_illustrations.items():
                placement = img_data.get('placement', 'gallery_world')
                if placement == 'gallery_character':
                    character_images[key] = img_data
                elif placement == 'gallery_world':
                    world_images[key] = img_data
                else:
                    other_images[key] = img_data
            
            # Create Character Gallery
            if character_images:
                char_gallery_html = "<div style='margin: 2em;'><h1>👥 Character Gallery</h1>"
                
                for key, img_data in character_images.items():
                    img_filename = _add_image_to_epub(book, img_data['file_path'], f'char_{key}')
                    
                    if img_filename:
                        char_gallery_html += f"""
                        <div style="margin: 2em 0; text-align: center;">
                            <h2>{img_data['title']}</h2>
                            <img src="{img_filename}" alt="{img_data['title']}" style="max-width: 100%; height: auto;" />
                            <p><em>{img_data['description']}</em></p>
                            <p>{img_data.get('caption', '')}</p>
                        </div>
                        """
                
                char_gallery_html += "</div>"
                
                char_gallery = epub.EpubHtml(
                    title="👥 Character Gallery",
                    file_name="gallery_characters.xhtml",
                    lang="en",
                    content=char_gallery_html
                )
                
                book.add_item(char_gallery)
                gallery_chapters.append(char_gallery)
            
            # Create World Gallery
            if world_images or other_images:
                world_gallery_html = "<div style='margin: 2em;'><h1>🌍 World Gallery</h1>"
                
                all_world_images = {**world_images, **other_images}
                for key, img_data in all_world_images.items():
                    img_filename = _add_image_to_epub(book, img_data['file_path'], f'world_{key}')
                    
                    if img_filename:
                        world_gallery_html += f"""
                        <div style="margin: 2em 0; text-align: center;">
                            <h2>{img_data['title']}</h2>
                            <img src="{img_filename}" alt="{img_data['title']}" style="max-width: 100%; height: auto;" />
                            <p><em>{img_data['description']}</em></p>
                            <p>{img_data.get('caption', '')}</p>
                        </div>
                        """
                
                world_gallery_html += "</div>"
                
                world_gallery = epub.EpubHtml(
                    title="🌍 World Gallery",
                    file_name="gallery_world.xhtml",
                    lang="en",
                    content=world_gallery_html
                )
                
                book.add_item(world_gallery)
                gallery_chapters.append(world_gallery)
        
        # Create Framework Branding Gallery
        branding_images = novel_images.get('common_branding', {})
        if branding_images:
            branding_gallery_html = "<div style='margin: 2em;'><h1>⚡ Framework Gallery</h1>"
            branding_gallery_html += "<p><em>Powered by The Lexicon Forge Translation Framework</em></p>"
            
            for key, img_data in branding_images.items():
                img_filename = _add_image_to_epub(book, img_data['file_path'], f'brand_{key}')
                
                if img_filename:
                    branding_gallery_html += f"""
                    <div style="margin: 2em 0; text-align: center;">
                        <h2>{img_data['title']}</h2>
                        <img src="{img_filename}" alt="{img_data['title']}" style="max-width: 100%; height: auto;" />
                        <p><em>{img_data['description']}</em></p>
                    </div>
                    """
            
            branding_gallery_html += "</div>"
            
            branding_gallery = epub.EpubHtml(
                title="⚡ Framework Gallery",
                file_name="gallery_branding.xhtml",
                lang="en",
                content=branding_gallery_html
            )
            
            book.add_item(branding_gallery)
            gallery_chapters.append(branding_gallery)
        
        if gallery_chapters:
            logger.info(f"Created {len(gallery_chapters)} gallery sections")
        
        return gallery_chapters
        
    except Exception as e:
        logger.warning(f"Could not create gallery sections: {e}")
        return []

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
        
        # Load novel images if available and enabled
        novel_images = {}
        if include_images and novel_slug:
            novel_images = _load_novel_images(novel_slug)
        
        # Set basic metadata
        book.set_identifier(f"{novel_slug or 'custom'}-translation-{datetime.now().strftime('%Y%m%d')}")
        book.set_title(title)
        book.set_language("en")
        book.add_author(author)
        book.add_metadata("DC", "contributor", translator)
        book.add_metadata("DC", "creator", "The Lexicon Forge Translation Framework")
        
        # Create custom title page with framework branding
        title_page = _create_branded_title_page(
            title, author, translator, novel_metadata, novel_images
        )
        book.add_item(title_page)
        
        # Add novel-specific metadata
        for genre in novel_metadata.get('genre', []):
            book.add_metadata("DC", "subject", genre)
        
        if description := novel_metadata.get('description'):
            book.add_metadata("DC", "description", description)
        
        # Add comprehensive CSS for markdown-converted content
        nav_css = epub.EpubItem(
            uid="nav_css",
            file_name="style/nav.css",
            media_type="text/css",
            content="""
            /* Framework-specific color scheme */
            :root {
                --primary-color: #1a365d;        /* Deep blue */
                --secondary-color: #2c5282;      /* Medium blue */
                --accent-color: #3182ce;         /* Bright blue */
                --text-primary: #1a202c;         /* Dark text */
                --text-secondary: #4a5568;       /* Medium text */
                --background-primary: #ffffff;    /* White background */
                --background-secondary: #f7fafc;  /* Light gray */
                --border-color: #e2e8f0;         /* Light border */
                --success-color: #38a169;        /* Green */
                --warning-color: #d69e2e;        /* Orange */
                --error-color: #e53e3e;          /* Red */
                --magic-color: #805ad5;          /* Purple */
            }
            
            /* Base typography with framework styling */
            body { 
                font-family: "Crimson Text", "Georgia", serif; 
                line-height: 1.7; 
                margin: 0; 
                padding: 1.5em; 
                background-color: var(--background-primary); 
                color: var(--text-primary); 
                font-size: 16px;
                max-width: 900px;
                margin: 0 auto;
            }
            
            /* Chapter titles with framework branding */
            h1 { 
                text-align: center; 
                color: var(--primary-color); 
                border-bottom: 3px solid var(--accent-color); 
                padding-bottom: 0.5em; 
                margin-bottom: 1.5em; 
                font-size: 2.2em;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 1px;
                position: relative;
            }
            
            h1.chapter-title { 
                color: var(--primary-color); 
                border-bottom: 3px solid var(--accent-color); 
                padding-bottom: 0.5em; 
                margin-top: 2em; 
                margin-bottom: 1.5em; 
                font-size: 1.8em;
                font-weight: 600;
                text-transform: none;
                letter-spacing: 0.5px;
            }
            
            /* Add framework subtitle to chapter titles */
            h1:after {
                content: "The Lexicon Forge";
                display: block;
                font-size: 0.3em;
                color: var(--text-secondary);
                font-weight: 300;
                margin-top: 0.5em;
                text-transform: none;
                letter-spacing: 0.5px;
            }
            
            /* Paragraph styling with framework design */
            p { 
                margin-bottom: 1.2em; 
                text-align: justify; 
                text-indent: 1.5em;
                line-height: 1.8;
            }
            
            /* First paragraph after headers - no indent */
            h1 + p, h2 + p, h3 + p {
                text-indent: 0;
            }
            
            /* Footnotes with framework styling */
            .footnotes-separator { 
                border: none; 
                border-top: 2px solid var(--border-color); 
                margin: 2em 0; 
            }
            
            .footnotes-title { 
                color: var(--text-secondary); 
                font-size: 1.1em; 
                margin-bottom: 0.5em; 
                font-weight: 600;
            }
            
            .footnote { 
                font-size: 0.9em; 
                color: var(--text-secondary); 
                margin-bottom: 0.5em; 
                padding-left: 1em; 
                border-left: 3px solid var(--border-color);
            }
            
            .footnote-number { 
                color: var(--accent-color); 
                font-weight: bold; 
            }
            
            .footnote-ref { 
                color: var(--accent-color); 
                font-size: 0.8em; 
                text-decoration: none; 
            }
            
            /* Emphasis and formatting with framework colors */
            em { 
                font-style: italic; 
                color: var(--magic-color); 
            }
            
            strong { 
                font-weight: 700; 
                color: var(--accent-color); 
            }
            
            /* Dialogue with enhanced styling */
            .dialogue { 
                color: var(--magic-color); 
                font-weight: 500; 
                font-style: italic;
            }
            
            /* Chinese terms and explanations with framework colors */
            .term { 
                font-weight: 600; 
                color: var(--error-color); 
                background-color: rgba(229, 62, 62, 0.1);
                padding: 0 0.3em;
                border-radius: 3px;
            }
            
            .explanation { 
                color: var(--text-secondary); 
                font-size: 0.9em; 
                font-style: italic; 
            }
            
            /* Cultivation terms with framework styling */
            .cultivation-term { 
                color: var(--success-color); 
                font-weight: 600; 
                background-color: rgba(56, 161, 105, 0.1); 
                padding: 0.3em 0.6em; 
                border-radius: 6px; 
                border: 1px solid rgba(56, 161, 105, 0.2);
                text-shadow: 0 1px 2px rgba(0,0,0,0.1);
            }
            
            /* Gallery styling with framework design */
            .gallery-section { 
                margin: 2em 0; 
                text-align: center; 
                padding: 2em;
                background-color: var(--background-secondary);
                border-radius: 12px;
                border: 1px solid var(--border-color);
            }
            
            .gallery-image { 
                max-width: 100%; 
                height: auto; 
                border-radius: 8px; 
                box-shadow: 0 4px 16px rgba(0,0,0,0.1); 
                margin: 1em 0;
            }
            
            .gallery-caption {
                font-style: italic;
                color: var(--text-secondary);
                margin-top: 1em;
                font-size: 0.95em;
            }
            
            /* Metadata section with framework design */
            .metadata-section { 
                font-family: "Fira Code", "Monaco", monospace; 
                background: linear-gradient(135deg, var(--background-secondary) 0%, #e6fffa 100%);
                padding: 1.5em; 
                border-radius: 8px; 
                margin: 1.5em 0; 
                border: 1px solid var(--border-color);
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            
            /* Framework footer styling */
            .framework-footer {
                margin-top: 3em;
                padding-top: 2em;
                border-top: 2px solid var(--border-color);
                text-align: center;
                color: var(--text-secondary);
                font-size: 0.9em;
            }
            
            /* General text improvements */
            pre { 
                white-space: pre-wrap; 
                line-height: 1.6; 
                font-family: "Fira Code", "Monaco", monospace; 
                background-color: var(--background-secondary); 
                padding: 1.5em; 
                border-radius: 8px; 
                overflow-x: auto; 
                border: 1px solid var(--border-color);
            }
            
            /* Headers with framework styling */
            h2, h3 { 
                color: var(--primary-color); 
                margin-top: 2em; 
                margin-bottom: 1em; 
                font-weight: 600;
            }
            
            h2 {
                font-size: 1.8em;
                border-bottom: 2px solid var(--accent-color);
                padding-bottom: 0.3em;
            }
            
            h3 {
                font-size: 1.4em;
                color: var(--secondary-color);
            }
            
            /* Links with framework styling */
            a { 
                color: var(--accent-color); 
                text-decoration: none; 
                transition: color 0.3s ease;
            }
            
            a:hover { 
                color: var(--primary-color);
                text-decoration: underline; 
            }
            
            /* Title page specific styling */
            .title-page {
                text-align: center;
                padding: 3em 2em;
                background: linear-gradient(135deg, var(--background-primary) 0%, var(--background-secondary) 100%);
                border-radius: 12px;
                margin: 2em 0;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            }
            
            .main-title {
                font-size: 3em;
                font-weight: 700;
                color: var(--primary-color);
                margin-bottom: 0.5em;
                text-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            
            .framework-branding {
                margin: 2em 0;
                padding: 1.5em;
                background: linear-gradient(135deg, var(--accent-color) 0%, var(--secondary-color) 100%);
                color: white;
                border-radius: 8px;
                box-shadow: 0 4px 16px rgba(0,0,0,0.2);
            }
            
            .framework-name {
                font-size: 1.8em;
                font-weight: 600;
                margin-bottom: 0.5em;
                text-shadow: 0 1px 2px rgba(0,0,0,0.3);
            }
            
            .framework-tagline {
                font-size: 1.1em;
                font-style: italic;
                opacity: 0.9;
            }
            
            /* Framework header styling */
            .framework-header {
                margin-bottom: 2em;
                padding: 1em;
                background: linear-gradient(135deg, var(--background-secondary) 0%, var(--background-primary) 100%);
                border-bottom: 2px solid var(--border-color);
                text-align: center;
            }
            
            .framework-branding-small {
                font-size: 0.9em;
                color: var(--text-secondary);
                font-weight: 500;
            }
            
            .framework-name-small {
                color: var(--primary-color);
                font-weight: 600;
            }
            
            .framework-separator {
                margin: 0 0.5em;
                color: var(--accent-color);
            }
            
            .framework-tagline-small {
                font-style: italic;
                color: var(--text-secondary);
            }
            
            /* Framework footer styling */
            .framework-footer {
                margin-top: 3em;
                padding: 2em 1em;
                background: linear-gradient(135deg, var(--background-secondary) 0%, var(--background-primary) 100%);
                border-top: 2px solid var(--border-color);
                text-align: center;
                border-radius: 0 0 8px 8px;
            }
            
            .framework-attribution {
                color: var(--text-secondary);
                font-size: 0.9em;
            }
            
            .framework-attribution p {
                margin: 0.5em 0;
                text-indent: 0;
            }
            
            .framework-attribution strong {
                color: var(--primary-color);
            }
            
            .framework-attribution em {
                color: var(--accent-color);
            }
            """
        )
        book.add_item(nav_css)
        
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
                
                # Convert content to HTML with proper formatting
                html_content = _convert_text_to_html(content)
                
                # Add framework header and footer elements
                framework_header = """
                <div class="framework-header">
                    <div class="framework-branding-small">
                        <span class="framework-name-small">The Lexicon Forge</span>
                        <span class="framework-separator">•</span>
                        <span class="framework-tagline-small">Bridging Languages, Preserving Stories</span>
                    </div>
                </div>
                """
                
                framework_footer = f"""
                <div class="framework-footer">
                    <div class="framework-attribution">
                        <p>Translated by <strong>{translator}</strong></p>
                        <p>Generated by <strong>The Lexicon Forge</strong> Translation Framework</p>
                        <p><em>Chapter {chapter_num if chapter_num else i} of {len(chapter_files)}</em></p>
                    </div>
                </div>
                """
                
                # Combine header, content, and footer
                full_html_content = framework_header + html_content + framework_footer
                
                # Create EPUB chapter with CSS reference
                chapter = epub.EpubHtml(
                    title=chapter_title,
                    file_name=f"chapter_{i:04d}.xhtml",
                    lang="en",
                    content=full_html_content
                )
                
                # Link CSS stylesheet
                chapter.add_item(nav_css)
                
                book.add_item(chapter)
                chapters.append(chapter)
                spine.append(chapter)
                
            except Exception as e:
                logger.error(f"Error processing chapter {filepath}: {e}")
                continue
        
        if not chapters:
            return False, "No chapters could be processed successfully"
        
        # Create image gallery sections if images are loaded
        gallery_chapters = []
        if novel_images:
            gallery_chapters = _create_gallery_sections(book, novel_images) or []
        
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
        
        # Organize chapters: title page first, then story chapters, then galleries, then metadata
        story_chapters = chapters.copy()
        all_chapters = [title_page] + story_chapters.copy()
        spine_items = [title_page] + spine.copy()
        
        # Add gallery chapters
        if gallery_chapters:
            all_chapters.extend(gallery_chapters)
            spine_items.extend(gallery_chapters)
        
        # Add metadata chapter at the end
        if metadata_chapter:
            all_chapters.append(metadata_chapter)
            spine_items.append(metadata_chapter)
        
        # Set up advanced hierarchical navigation
        book.toc = _create_advanced_toc(story_chapters, gallery_chapters, metadata_chapter)
        book.spine = ["nav"] + spine_items
        
        # Add required navigation files
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        
        # Write EPUB file
        epub.write_epub(output_path, book, {})
        
        logger.info(f"EPUB created successfully: {output_path}")
        chapter_count = len(story_chapters)
        gallery_count = len(gallery_chapters)
        
        # Build comprehensive success message
        message_parts = [f"{chapter_count} chapters"]
        if gallery_count > 0:
            message_parts.append(f"{gallery_count} galleries")
        if metadata_chapter:
            message_parts.append("metadata report")
        
        return True, f"EPUB created with {' + '.join(message_parts)}"
        
    except Exception as e:
        logger.error(f"Error creating EPUB: {e}")
        return False, f"EPUB creation failed: {str(e)}"

# Future expansion functions (stubs for v2)
def _add_images_to_epub(book, novel_images):
    """Add images to EPUB book (to be implemented)."""
    # TODO: Implement image integration
    pass

