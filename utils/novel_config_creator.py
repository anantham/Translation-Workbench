"""
Novel Configuration Creator - Post-Scraping Setup
"""
import os
import json
from datetime import datetime
from urllib.parse import urlparse
from .logging import logger


def create_novel_config(novel_dir, scraping_info, user_config=None):
    """
    Create a novel_config.json file after successful scraping.
    
    Args:
        novel_dir (str): Path to the novel directory
        scraping_info (dict): Information from the scraping process
        user_config (dict, optional): User-provided configuration overrides
    
    Returns:
        tuple: (success: bool, message: str, config_path: str)
    """
    try:
        config_path = os.path.join(novel_dir, "novel_config.json")
        
        # Extract basic info from scraping_info
        source_url = scraping_info.get('start_url', '')
        total_chapters = scraping_info.get('chapters_scraped', 0)
        novel_name = os.path.basename(novel_dir)
        
        # Determine source language and details from URL
        source_details = _analyze_source_url(source_url)
        
        # Build base configuration
        config = {
            "novel_info": {
                "title": user_config.get('title', novel_name.replace('_', ' ').title()) if user_config else novel_name.replace('_', ' ').title(),
                "title_english": user_config.get('title_english', '') if user_config else '',
                "title_chinese": user_config.get('title_chinese', '') if user_config else '',
                "author": user_config.get('author', 'Unknown') if user_config else 'Unknown',
                "source_language": source_details['language'],
                "target_language": "english",
                "description": user_config.get('description', f"Novel scraped from {source_details['site_name']}") if user_config else f"Novel scraped from {source_details['site_name']}",
                "source_url": source_url,
                "translator": source_details['translator'],
                "status": user_config.get('status', 'ongoing') if user_config else 'ongoing'
            },
            "directory_structure": {
                "raw_chapters": "raw_chapters/",
                "translated_chapters": None,
                "alignment_map": None,
                "images": None,
                "epub_exports": None
            },
            "chapter_info": {
                "total_chapters_available": total_chapters,
                "chapter_range": f"001-{total_chapters:03d}+",
                "naming_pattern": _detect_naming_pattern(novel_dir),
                "encoding": "utf-8"
            },
            "translation_config": {
                "primary_model": user_config.get('primary_model', 'gemini-1.5-pro') if user_config else 'gemini-1.5-pro',
                "backup_model": user_config.get('backup_model', 'gpt-4o') if user_config else 'gpt-4o',
                "prompt_style": user_config.get('prompt_style', _suggest_prompt_style(source_details['genre'])) if user_config else _suggest_prompt_style(source_details['genre']),
                "preserve_formatting": True,
                "include_translator_notes": source_details['has_translator_notes']
            },
            "scraping_metadata": {
                "scraped_date": datetime.now().isoformat(),
                "scraper_version": "1.0",
                "chapters_found": total_chapters,
                "scraping_direction": scraping_info.get('direction', 'forwards'),
                "adapter_used": scraping_info.get('adapter_type', 'unknown')
            },
            "created_date": datetime.now().strftime("%Y-%m-%d"),
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "version": "1.0"
        }
        
        # Save the configuration
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        
        logger.info(f"[CONFIG] Created novel config: {config_path}")
        return True, f"Novel configuration created successfully", config_path
        
    except Exception as e:
        error_msg = f"Failed to create novel config: {e}"
        logger.error(f"[CONFIG] {error_msg}")
        return False, error_msg, ""


def _analyze_source_url(url):
    """Analyze source URL to determine site details."""
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    
    # Site-specific detection
    if 'shalvationtranslations' in domain:
        return {
            'site_name': 'Shalvation Translations',
            'language': 'korean',
            'translator': 'Shalvation Translations',
            'genre': 'fantasy',
            'has_translator_notes': True
        }
    elif 'novelcool' in domain:
        return {
            'site_name': 'NovelCool',
            'language': 'chinese',
            'translator': 'Various',
            'genre': 'xianxia',
            'has_translator_notes': False
        }
    elif 'dxmwx' in domain:
        return {
            'site_name': 'DXMWX',
            'language': 'chinese',
            'translator': 'Raw Chinese',
            'genre': 'xianxia',
            'has_translator_notes': False
        }
    elif 'kanunu' in domain:
        return {
            'site_name': 'Kanunu',
            'language': 'chinese',
            'translator': 'Raw Chinese',
            'genre': 'xianxia',
            'has_translator_notes': False
        }
    elif 'kakuyomu' in domain:
        return {
            'site_name': 'Kakuyomu',
            'language': 'japanese',
            'translator': 'Raw Japanese',
            'genre': 'light_novel',
            'has_translator_notes': False
        }
    else:
        return {
            'site_name': domain,
            'language': 'unknown',
            'translator': 'Unknown',
            'genre': 'general',
            'has_translator_notes': False
        }


def _detect_naming_pattern(novel_dir):
    """Detect chapter file naming pattern."""
    raw_chapters_dir = os.path.join(novel_dir, "raw_chapters")
    if not os.path.exists(raw_chapters_dir):
        return "Chapter-XXXX.txt"
    
    # Get first few files to detect pattern
    files = sorted([f for f in os.listdir(raw_chapters_dir) if f.endswith('.txt')])[:3]
    
    if not files:
        return "Chapter-XXXX.txt"
    
    # Analyze patterns
    first_file = files[0]
    if first_file.startswith('Chapter-'):
        return "Chapter-XXXX-Title.txt"
    elif first_file.startswith('Ch'):
        return "ChXXX.txt"
    elif first_file[0].isdigit():
        return "XXXX.txt"
    else:
        return "Chapter-XXXX.txt"


def _suggest_prompt_style(genre):
    """Suggest appropriate prompt style based on genre."""
    genre_mapping = {
        'fantasy': 'fantasy_novel',
        'xianxia': 'cultivation_novel', 
        'light_novel': 'light_novel',
        'romance': 'romance_novel',
        'general': 'general_novel'
    }
    return genre_mapping.get(genre, 'general_novel')


def get_novel_config_form_data():
    """Get the structure for the novel config form."""
    return {
        "basic_info": [
            {
                "key": "title",
                "label": "Novel Title",
                "type": "text",
                "required": True,
                "help": "The main title of the novel"
            },
            {
                "key": "title_english", 
                "label": "English Title",
                "type": "text",
                "required": False,
                "help": "English translation of the title (if different)"
            },
            {
                "key": "title_chinese",
                "label": "Original Title (Chinese/Japanese/Korean)", 
                "type": "text",
                "required": False,
                "help": "Original language title"
            },
            {
                "key": "author",
                "label": "Author Name",
                "type": "text", 
                "required": True,
                "help": "Name of the original author"
            },
            {
                "key": "description",
                "label": "Description",
                "type": "textarea",
                "required": False,
                "help": "Brief description of the novel"
            },
            {
                "key": "status",
                "label": "Translation Status",
                "type": "select",
                "options": ["ongoing", "completed", "hiatus", "dropped"],
                "required": True,
                "help": "Current status of the translation"
            }
        ],
        "translation_settings": [
            {
                "key": "primary_model",
                "label": "Primary AI Model",
                "type": "select", 
                "options": ["gemini-1.5-pro", "gpt-4o", "gpt-4o-mini", "gemini-1.5-flash"],
                "required": True,
                "help": "Primary model for translation"
            },
            {
                "key": "backup_model",
                "label": "Backup AI Model",
                "type": "select",
                "options": ["gpt-4o", "gemini-1.5-pro", "gpt-4o-mini", "gemini-1.5-flash"],
                "required": True,
                "help": "Fallback model if primary fails"
            },
            {
                "key": "prompt_style",
                "label": "Prompt Style",
                "type": "select",
                "options": ["fantasy_novel", "cultivation_novel", "light_novel", "romance_novel", "general_novel"],
                "required": True,
                "help": "Translation style appropriate for the genre"
            }
        ]
    }