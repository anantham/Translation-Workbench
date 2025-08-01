"""
Alignment Map Builder Module

Intelligent alignment mapping with support for both legacy and new multi-novel structures.
Adapted from scripts/utils/build_complete_alignment_map.py for Streamlit integration.
"""

import os
import json
import re
import shutil
from datetime import datetime
from typing import Dict, Tuple, Optional, Set

def extract_chapter_number_from_filename(filename: str) -> Optional[int]:
    """Extract chapter number from various filename formats."""
    # Pattern for English-Chapter-0001.txt
    match = re.search(r'English-Chapter-(\d+)\.txt', filename)
    if match:
        return int(match.group(1))
    
    # Pattern for Chapter-0001-Title.txt
    match = re.search(r'Chapter-(\d+)', filename)
    if match:
        return int(match.group(1))
    
    return None

def scan_directory_for_chapters(directory: str, file_pattern: Optional[str] = None) -> Dict[int, str]:
    """Scan directory and return chapter number -> filepath mapping."""
    chapters = {}
    
    if not os.path.exists(directory):
        return chapters
    
    for filename in os.listdir(directory):
        if file_pattern and not re.search(file_pattern, filename):
            continue
            
        chapter_num = extract_chapter_number_from_filename(filename)
        if chapter_num:
            filepath = os.path.join(directory, filename)
            chapters[chapter_num] = filepath
    
    return chapters

def detect_novel_structure(novel_name: str = None) -> Dict[str, str]:
    """Detect directory structure and return paths for English and Chinese chapters."""
    paths = {
        "english_dir": None,
        "chinese_dir": None,
        "alignment_file": None,
        "structure_type": None
    }
    
    # Try new multi-novel structure first
    if novel_name:
        from .config import get_novel_dir, get_novel_alignment_map, get_novel_raw_chapters_dir, get_novel_official_english_dir
        
        novel_dir = get_novel_dir(novel_name)
        if os.path.exists(novel_dir):
            english_dir = get_novel_official_english_dir(novel_name)
            chinese_dir = get_novel_raw_chapters_dir(novel_name)
            alignment_file = get_novel_alignment_map(novel_name)
            
            if os.path.exists(english_dir) or os.path.exists(chinese_dir):
                paths.update({
                    "english_dir": english_dir,
                    "chinese_dir": chinese_dir,
                    "alignment_file": alignment_file,
                    "structure_type": "multi-novel"
                })
                return paths
    
    # Try legacy structure
    legacy_english = "english_chapters"
    legacy_chinese = "novel_content_dxmwx_complete"
    
    if os.path.exists(legacy_english) or os.path.exists(legacy_chinese):
        paths.update({
            "english_dir": legacy_english,
            "chinese_dir": legacy_chinese,
            "alignment_file": "alignment_map.json",
            "structure_type": "legacy"
        })
        return paths
    
    # No structure found
    return paths

def build_alignment_map(novel_name: str = None, force_rebuild: bool = False) -> Tuple[Dict, Dict]:
    """Build alignment map with intelligent gap-filling or force rebuild.
    
    Returns:
        tuple: (alignment_map, build_stats)
    """
    # Detect directory structure
    paths = detect_novel_structure(novel_name)
    
    if not paths["structure_type"]:
        return {}, {
            "error": "No chapter directories found. Please ensure English and/or Chinese chapters are available.",
            "english_dir": None,
            "chinese_dir": None
        }
    
    build_stats = {
        "structure_type": paths["structure_type"],
        "english_dir": paths["english_dir"],
        "chinese_dir": paths["chinese_dir"],
        "alignment_file": paths["alignment_file"],
        "english_count": 0,
        "chinese_count": 0,
        "overlapping_count": 0,
        "total_mapped": 0,
        "added_count": 0,
        "updated_count": 0,
        "preserved_count": 0,
        "error": None
    }
    
    try:
        # Scan for English chapters
        english_chapters = {}
        if paths["english_dir"] and os.path.exists(paths["english_dir"]):
            english_chapters = scan_directory_for_chapters(paths["english_dir"], r"English-Chapter-\d+\.txt")
        build_stats["english_count"] = len(english_chapters)
        
        # Scan for Chinese chapters  
        chinese_chapters = {}
        if paths["chinese_dir"] and os.path.exists(paths["chinese_dir"]):
            chinese_chapters = scan_directory_for_chapters(paths["chinese_dir"], r"Chapter-\d+")
        build_stats["chinese_count"] = len(chinese_chapters)
        
        # Find overlapping chapter numbers
        english_nums = set(english_chapters.keys())
        chinese_nums = set(chinese_chapters.keys())
        overlapping = english_nums.intersection(chinese_nums)
        build_stats["overlapping_count"] = len(overlapping)
        
        # Load existing alignment map (unless force rebuild)
        alignment_map = {}
        if not force_rebuild and paths["alignment_file"] and os.path.exists(paths["alignment_file"]):
            try:
                with open(paths["alignment_file"], 'r', encoding='utf-8') as f:
                    alignment_map = json.load(f)
                build_stats["preserved_count"] = len(alignment_map)
            except (json.JSONDecodeError, IOError):
                alignment_map = {}
        
        # Track updates
        added_count = 0
        updated_count = 0
        
        # Strategy 1: Map overlapping chapters
        for ch_num in overlapping:
            ch_key = str(ch_num)
            new_entry = {
                "raw_file": chinese_chapters[ch_num],
                "english_file": english_chapters[ch_num]
            }
            
            if force_rebuild or ch_key not in alignment_map:
                alignment_map[ch_key] = new_entry
                added_count += 1
            elif (alignment_map[ch_key].get("raw_file") != new_entry["raw_file"] or 
                  alignment_map[ch_key].get("english_file") != new_entry["english_file"]):
                alignment_map[ch_key].update(new_entry)
                updated_count += 1
        
        # Strategy 2: Map Chinese chapters without English counterparts
        for ch_num in chinese_nums - english_nums:
            ch_key = str(ch_num)
            new_entry = {
                "raw_file": chinese_chapters[ch_num],
                "english_file": None
            }
            
            if force_rebuild or ch_key not in alignment_map:
                alignment_map[ch_key] = new_entry
                added_count += 1
            elif alignment_map[ch_key].get("raw_file") != new_entry["raw_file"]:
                alignment_map[ch_key]["raw_file"] = new_entry["raw_file"]
                updated_count += 1
        
        # Strategy 3: Map English chapters without Chinese counterparts
        for ch_num in english_nums - chinese_nums:
            ch_key = str(ch_num)
            new_entry = {
                "raw_file": None,
                "english_file": english_chapters[ch_num]
            }
            
            if force_rebuild or ch_key not in alignment_map:
                alignment_map[ch_key] = new_entry
                added_count += 1
            elif alignment_map[ch_key].get("english_file") != new_entry["english_file"]:
                alignment_map[ch_key]["english_file"] = new_entry["english_file"]
                updated_count += 1
        
        build_stats.update({
            "total_mapped": len(alignment_map),
            "added_count": added_count,
            "updated_count": updated_count
        })
        
        return alignment_map, build_stats
        
    except Exception as e:
        build_stats["error"] = str(e)
        return {}, build_stats

def save_alignment_map_with_backup(alignment_map: Dict, output_file: str) -> bool:
    """Save alignment map with backup of existing file."""
    try:
        # Create backup directory
        backup_dir = os.path.join("data", "backups")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create backup if file exists
        if os.path.exists(output_file):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{os.path.basename(output_file)}.backup_{timestamp}"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            shutil.copy(output_file, backup_path)
        
        # Create directory for output file if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Save new alignment map
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(alignment_map, f, indent=2, ensure_ascii=False)
        
        return True
        
    except Exception as e:
        print(f"Error saving alignment map: {e}")
        return False

def streamlit_build_alignment_map(novel_name: str = None, force_rebuild: bool = False) -> Tuple[bool, str, Dict]:
    """Streamlit-friendly wrapper for building alignment map.
    
    Returns:
        tuple: (success, message, build_stats)
    """
    alignment_map, build_stats = build_alignment_map(novel_name, force_rebuild)
    
    if build_stats.get("error"):
        return False, build_stats["error"], build_stats
    
    if not alignment_map:
        return False, "No chapters found to map.", build_stats
    
    # Save the alignment map
    alignment_file = build_stats["alignment_file"]
    success = save_alignment_map_with_backup(alignment_map, alignment_file)
    
    if success:
        mode = "Force Rebuild" if force_rebuild else "Intelligent Gap-Filling"
        message = f"✅ Alignment map built successfully ({mode})\n"
        message += f"📊 Total mapped: {build_stats['total_mapped']} chapters\n"
        message += f"🔍 Both files: {build_stats['overlapping_count']} chapters\n"
        message += f"🇨🇳 Chinese only: {build_stats['chinese_count'] - build_stats['overlapping_count']} chapters\n"
        message += f"🇺🇸 English only: {build_stats['english_count'] - build_stats['overlapping_count']} chapters"
        
        if not force_rebuild:
            message += f"\n\n🔄 Updates:\n"
            message += f"• Preserved: {build_stats['preserved_count']}\n"
            message += f"• Added: {build_stats['added_count']}\n"
            message += f"• Updated: {build_stats['updated_count']}"
        
        return True, message, build_stats
    else:
        return False, "Failed to save alignment map.", build_stats