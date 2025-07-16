#!/usr/bin/env python3
"""
Quick test of build_and_report.py with working chapters only
"""

import json

def create_minimal_alignment_map():
    """Create alignment map with only the 9 working chapters"""
    
    # These are the chapters that actually exist
    working_chapters = {
        "1": {
            "raw_file": "novel_content_dxmwx_complete/Chapter-0001-第一章 道法无情.txt",
            "english_file": "english_chapters/English-Chapter-0001.txt"
        },
        "2": {
            "raw_file": "novel_content_dxmwx_complete/Chapter-0002-第二章 作弊器.txt",
            "english_file": "english_chapters/English-Chapter-0002.txt"
        },
        "3": {
            "raw_file": "novel_content_dxmwx_complete/Chapter-0003-第三章 黑虎刀法 上.txt",
            "english_file": "english_chapters/English-Chapter-0003.txt"
        },
        "4": {
            "raw_file": "novel_content_dxmwx_complete/Chapter-0004-第四章 黑虎刀法 下.txt",
            "english_file": "english_chapters/English-Chapter-0004.txt"
        },
        "5": {
            "raw_file": "novel_content_dxmwx_complete/Chapter-0005-第五章 勇武.txt",
            "english_file": "english_chapters/English-Chapter-0005.txt"
        },
        "6": {
            "raw_file": "novel_content_dxmwx_complete/Chapter-0006-第六章 黑会.txt",
            "english_file": "english_chapters/English-Chapter-0006.txt"
        },
        "7": {
            "raw_file": "novel_content_dxmwx_complete/Chapter-0007-第七章 会场 一.txt",
            "english_file": "english_chapters/English-Chapter-0007.txt"
        },
        "8": {
            "raw_file": "novel_content_dxmwx_complete/Chapter-0008-第八章 会场 二.txt",
            "english_file": "english_chapters/English-Chapter-0008.txt"
        },
        "9": {
            "raw_file": "novel_content_dxmwx_complete/Chapter-0009-第九章 端木婉 一.txt",
            "english_file": "english_chapters/English-Chapter-0009.txt"
        }
    }
    
    # Save as test alignment map
    with open("../../tests/data/alignment_map_test.json", 'w', encoding='utf-8') as f:
        json.dump(working_chapters, f, indent=2, ensure_ascii=False)
    
    print("✅ Created tests/data/alignment_map_test.json with 9 working chapters")
    return working_chapters

if __name__ == "__main__":
    create_minimal_alignment_map()
    print("\n🚀 Now run: python build_and_report.py")
    print("📋 When it asks for alignment map, use 'tests/data/alignment_map_test.json'")