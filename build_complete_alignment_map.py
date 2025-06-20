#!/usr/bin/env python3
"""
Complete Alignment Map Builder
Creates comprehensive alignment mapping for all available chapters
"""
import os
import json
import re
from datetime import datetime

def extract_chapter_number_from_filename(filename):
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

def scan_directory_for_chapters(directory, file_pattern=None):
    """Scan directory and return chapter number -> filepath mapping."""
    chapters = {}
    
    if not os.path.exists(directory):
        print(f"❌ Directory not found: {directory}")
        return chapters
    
    for filename in os.listdir(directory):
        if file_pattern and not re.search(file_pattern, filename):
            continue
            
        chapter_num = extract_chapter_number_from_filename(filename)
        if chapter_num:
            filepath = os.path.join(directory, filename)
            chapters[chapter_num] = filepath
    
    return chapters

def build_complete_alignment_map():
    """Build complete alignment map from all available files."""
    print("🔨 Building Complete Alignment Map")
    print("=" * 50)
    
    # Scan for English chapters
    print("📖 Scanning English chapters...")
    english_chapters = scan_directory_for_chapters("english_chapters", r"English-Chapter-\d+\.txt")
    print(f"   Found {len(english_chapters)} English chapters")
    
    # Scan for Chinese chapters  
    print("📜 Scanning Chinese chapters...")
    chinese_chapters = scan_directory_for_chapters("novel_content_dxmwx_complete", r"Chapter-\d+")
    print(f"   Found {len(chinese_chapters)} Chinese chapters")
    
    # Find overlapping chapter numbers
    english_nums = set(english_chapters.keys())
    chinese_nums = set(chinese_chapters.keys())
    overlapping = english_nums.intersection(chinese_nums)
    
    print(f"🎯 Overlapping chapters: {len(overlapping)}")
    print(f"📊 English range: {min(english_nums) if english_nums else 'N/A'} - {max(english_nums) if english_nums else 'N/A'}")
    print(f"📊 Chinese range: {min(chinese_nums) if chinese_nums else 'N/A'} - {max(chinese_nums) if chinese_nums else 'N/A'}")
    
    # Build alignment map
    alignment_map = {}
    
    # Strategy 1: Map overlapping chapters directly
    for ch_num in overlapping:
        alignment_map[str(ch_num)] = {
            "raw_file": chinese_chapters[ch_num],
            "english_file": english_chapters[ch_num]
        }
    
    # Strategy 2: Map Chinese chapters without English counterparts (for future alignment)
    for ch_num in chinese_nums - english_nums:
        alignment_map[str(ch_num)] = {
            "raw_file": chinese_chapters[ch_num],
            "english_file": None
        }
    
    # Strategy 3: Map English chapters without Chinese counterparts
    for ch_num in english_nums - chinese_nums:
        alignment_map[str(ch_num)] = {
            "raw_file": None,
            "english_file": english_chapters[ch_num]
        }
    
    print(f"\n📋 ALIGNMENT MAP SUMMARY:")
    print(f"   Total mapped chapters: {len(alignment_map)}")
    print(f"   Both files available: {len(overlapping)}")
    print(f"   Chinese only: {len(chinese_nums - english_nums)}")
    print(f"   English only: {len(english_nums - chinese_nums)}")
    
    return alignment_map

def save_alignment_map_with_backup(alignment_map, output_file="alignment_map.json"):
    """Save alignment map with backup of existing file in organized backup directory."""
    # Ensure backup directory exists
    backup_dir = os.path.join("data", "backups")
    os.makedirs(backup_dir, exist_ok=True)
    
    # Create backup if file exists
    if os.path.exists(output_file):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{os.path.basename(output_file)}.backup_{timestamp}"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # Copy to backup directory instead of renaming
        import shutil
        shutil.copy(output_file, backup_path)
        print(f"📁 Backup created: {backup_filename} (in data/backups/)")
    
    # Save new alignment map
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(alignment_map, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Alignment map saved: {output_file}")

def main():
    """Main execution function."""
    print("🚀 Complete Alignment Map Builder")
    print("Building comprehensive mapping of all available chapters...\n")
    
    # Build the alignment map
    alignment_map = build_complete_alignment_map()
    
    # Save with backup
    save_alignment_map_with_backup(alignment_map)
    
    print(f"\n🎉 Complete! Ready for systematic alignment analysis.")
    print(f"💡 Run 'streamlit run master_review_tool.py' to use the new alignment map.")

if __name__ == "__main__":
    main()