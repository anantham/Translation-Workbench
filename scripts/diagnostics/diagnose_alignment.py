#!/usr/bin/env python3
"""
Alignment Diagnostics Tool
Shows exactly what files exist vs what's in alignment_map.json
"""

import os
import json
import re

def load_alignment_map():
    with open("alignment_map.json", 'r', encoding='utf-8') as f:
        return json.load(f)

def scan_chinese_files():
    """Scan Chinese files and extract chapter numbers from titles"""
    chinese_files = {}
    directory = "novel_content_dxmwx_complete"
    
    if not os.path.exists(directory):
        print(f"❌ Directory not found: {directory}")
        return {}
    
    for filename in os.listdir(directory):
        if filename.endswith('.txt') and filename.startswith('Chapter-'):
            # Extract chapter number from Chinese title
            # Pattern: 第N章 or 第NN章
            match = re.search(r'第(\d+|[一二三四五六七八九十]+)章', filename)
            if match:
                chinese_num = match.group(1)
                
                # Convert Chinese numbers to Arabic
                chinese_to_arabic = {
                    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
                    '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
                    '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15,
                    '十六': 16, '十七': 17, '十八': 18, '十九': 19, '二十': 20
                }
                
                if chinese_num in chinese_to_arabic:
                    chapter_num = chinese_to_arabic[chinese_num]
                else:
                    try:
                        chapter_num = int(chinese_num)
                    except ValueError:
                        continue
                
                chinese_files[chapter_num] = os.path.join(directory, filename)
    
    return chinese_files

def scan_english_files():
    """Scan English files"""
    english_files = {}
    directory = "english_chapters"
    
    if not os.path.exists(directory):
        print(f"❌ Directory not found: {directory}")
        return {}
    
    for filename in os.listdir(directory):
        if filename.startswith('English-Chapter-') and filename.endswith('.txt'):
            # Extract number: English-Chapter-0001.txt -> 1
            match = re.search(r'English-Chapter-(\d+)\.txt', filename)
            if match:
                chapter_num = int(match.group(1))
                english_files[chapter_num] = os.path.join(directory, filename)
    
    return english_files

def main():
    print("🔍 Alignment Diagnostics")
    print("=" * 50)
    
    # Load current alignment map
    try:
        alignment_map = load_alignment_map()
        mapped_chapters = set(int(k) for k in alignment_map.keys())
        print(f"📋 Alignment map contains {len(mapped_chapters)} chapters")
    except Exception as e:
        print(f"❌ Could not load alignment map: {e}")
        return
    
    # Scan actual files
    print("\n🔍 Scanning files...")
    chinese_files = scan_chinese_files()
    english_files = scan_english_files()
    
    print(f"📂 Found {len(chinese_files)} Chinese files")
    print(f"📂 Found {len(english_files)} English files")
    
    # Find chapters that exist in files but not in alignment map
    all_chinese_chapters = set(chinese_files.keys())
    all_english_chapters = set(english_files.keys())
    
    # Missing from alignment map
    chinese_not_mapped = all_chinese_chapters - mapped_chapters
    english_not_mapped = all_english_chapters - mapped_chapters
    
    # In alignment map but files don't exist
    mapped_but_no_chinese = mapped_chapters - all_chinese_chapters
    mapped_but_no_english = mapped_chapters - all_english_chapters
    
    print(f"\n🎯 Diagnostic Results:")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    if chinese_not_mapped:
        print(f"📝 Chinese files NOT in alignment map: {sorted(chinese_not_mapped)}")
        print("   Files found:")
        for ch in sorted(chinese_not_mapped)[:10]:  # Show first 10
            print(f"   • Chapter {ch}: {chinese_files[ch]}")
        if len(chinese_not_mapped) > 10:
            print(f"   • ... and {len(chinese_not_mapped) - 10} more")
    
    if english_not_mapped:
        print(f"\n📖 English files NOT in alignment map: {sorted(english_not_mapped)}")
    
    if mapped_but_no_chinese:
        print(f"\n❌ Mapped chapters with MISSING Chinese files: {sorted(mapped_but_no_chinese)}")
        for ch in sorted(mapped_but_no_chinese)[:5]:
            expected_path = alignment_map[str(ch)]['raw_file']
            print(f"   • Chapter {ch}: Expected '{expected_path}'")
    
    if mapped_but_no_english:
        print(f"\n❌ Mapped chapters with MISSING English files: {sorted(mapped_but_no_english)}")
    
    # Show alignment for first few chapters
    print(f"\n📊 Sample Alignment (first 10 mapped chapters):")
    print(f"{'Ch':<3} {'Chinese File':<50} {'English File':<30} {'Status'}")
    print("─" * 90)
    
    for ch in sorted(mapped_chapters)[:10]:
        ch_str = str(ch)
        chinese_path = alignment_map[ch_str]['raw_file']
        english_path = alignment_map[ch_str]['english_file']
        
        chinese_exists = os.path.exists(chinese_path)
        english_exists = os.path.exists(english_path)
        
        if chinese_exists and english_exists:
            status = "✅"
        elif chinese_exists:
            status = "📝❌"  # Chinese only
        elif english_exists:
            status = "❌📖"  # English only
        else:
            status = "❌❌"   # Neither
        
        print(f"{ch:<3} {os.path.basename(chinese_path):<50} {os.path.basename(english_path):<30} {status}")
    
    # Summary and recommendations
    print(f"\n💡 Recommendations:")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    total_fixable = len(chinese_not_mapped & all_english_chapters)
    if total_fixable > 0:
        print(f"✅ {total_fixable} chapters can be added to alignment map (have both Chinese and English files)")
    
    total_broken = len(mapped_but_no_chinese) + len(mapped_but_no_english)
    if total_broken > 0:
        print(f"🔧 {total_broken} entries in alignment map need file path corrections")
    
    if chinese_not_mapped:
        print(f"\n🛠️  Next step: Run alignment repair script to add missing chapters")
    
    print(f"\n📈 Overall Status:")
    working_chapters = len(mapped_chapters - mapped_but_no_chinese - mapped_but_no_english)
    print(f"   • Working alignments: {working_chapters}")
    print(f"   • Broken alignments: {total_broken}")
    print(f"   • Missing from map: {len(chinese_not_mapped)}")

if __name__ == "__main__":
    main()