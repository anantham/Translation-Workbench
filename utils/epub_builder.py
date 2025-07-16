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
        
        # Set up navigation
        book.toc = tuple(chapters)
        book.spine = ["nav"] + spine
        
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
        return True, f"EPUB created with {len(chapters)} chapters"
        
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