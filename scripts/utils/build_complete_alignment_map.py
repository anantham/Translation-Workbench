#!/usr/bin/env python3
"""
Alignment Map Builder
Intelligent alignment mapping with force rebuild and gap-filling modes
"""
import os
import json
import re
import argparse
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

def load_existing_alignment_map(alignment_file="alignment_map.json"):
    """Load existing alignment map if it exists."""
    if os.path.exists(alignment_file):
        try:
            with open(alignment_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Warning: Could not load existing alignment map: {e}")
            return {}
    return {}

def buildAlignmentMap(force_rebuild=False, alignment_file="alignment_map.json"):
    """Build alignment map with intelligent gap-filling or force rebuild.
    
    Args:
        force_rebuild (bool): If True, rebuild entire map ignoring existing mappings
        alignment_file (str): Path to alignment map file
    """
    mode = "Force Rebuild" if force_rebuild else "Intelligent Gap-Filling"
    print(f"🔨 Building Alignment Map ({mode})")
    print("=" * 60)
    
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
    
    # Load existing alignment map (unless force rebuild)
    if force_rebuild:
        print("🚫 Force rebuild mode: Ignoring existing alignment map")
        alignment_map = {}
        preserved_count = 0
    else:
        print("🧠 Loading existing alignment map for intelligent updates...")
        alignment_map = load_existing_alignment_map(alignment_file)
        preserved_count = len(alignment_map)
        print(f"   Found {preserved_count} existing mappings")
    
    # Track updates
    added_count = 0
    updated_count = 0
    
    # Strategy 1: Map overlapping chapters (update only if missing or force rebuild)
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
            # Update file paths if they've changed
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
            # Update raw_file path if changed, preserve english_file if manually set
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
            # Update english_file path if changed, preserve raw_file if manually set
            alignment_map[ch_key]["english_file"] = new_entry["english_file"]
            updated_count += 1
    
    print(f"\n📋 ALIGNMENT MAP SUMMARY:")
    print(f"   Total mapped chapters: {len(alignment_map)}")
    print(f"   Both files available: {len(overlapping)}")
    print(f"   Chinese only: {len(chinese_nums - english_nums)}")
    print(f"   English only: {len(english_nums - chinese_nums)}")
    
    if not force_rebuild:
        print(f"\n🔄 UPDATE SUMMARY:")
        print(f"   Preserved existing: {preserved_count}")
        print(f"   Added new: {added_count}")
        print(f"   Updated paths: {updated_count}")
    
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
    """Main execution function with command line argument support."""
    parser = argparse.ArgumentParser(
        description="Build alignment map with intelligent gap-filling or force rebuild",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python build_complete_alignment_map.py                    # Intelligent gap-filling (default)
  python build_complete_alignment_map.py --force-rebuild   # Force complete rebuild
  python build_complete_alignment_map.py -f               # Force rebuild (short form)
        """
    )
    
    parser.add_argument(
        '--force-rebuild', '-f',
        action='store_true',
        help='Force complete rebuild, ignoring existing alignment map'
    )
    
    parser.add_argument(
        '--alignment-file',
        default='alignment_map.json',
        help='Path to alignment map file (default: alignment_map.json)'
    )
    
    args = parser.parse_args()
    
    print("🚀 Alignment Map Builder")
    mode = "Force Rebuild" if args.force_rebuild else "Intelligent Gap-Filling"
    print(f"Mode: {mode}")
    print(f"File: {args.alignment_file}\n")
    
    # Build the alignment map
    alignment_map = buildAlignmentMap(
        force_rebuild=args.force_rebuild,
        alignment_file=args.alignment_file
    )
    
    # Save with backup
    save_alignment_map_with_backup(alignment_map, args.alignment_file)
    
    print(f"\n🎉 Complete! Ready for systematic alignment analysis.")
    print(f"💡 Run 'python run_workbench.py' to use the alignment map.")

if __name__ == "__main__":
    main()