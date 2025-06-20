#!/usr/bin/env python3
"""
Quick Resume Tool - Uses metadata to instantly resume from any missing chapter
"""
import json
import os
import sys

def load_metadata(metadata_file):
    """Load scraping metadata."""
    if not os.path.exists(metadata_file):
        print(f"❌ Metadata file '{metadata_file}' not found!")
        return None
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_missing_chapters(output_dir, metadata):
    """Find chapters that are missing from local storage."""
    existing_chapters = set()
    
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.startswith('Chapter-') and filename.endswith('.txt'):
                import re
                match = re.search(r'Chapter-(\d{4})', filename)
                if match:
                    existing_chapters.add(int(match.group(1)))
    
    known_chapters = set(int(ch) for ch in metadata["chapters"].keys())
    missing = known_chapters - existing_chapters
    
    return sorted(missing), existing_chapters

def main():
    metadata_file = "scraping_metadata.json"
    output_dir = "novel_content_dxmwx_complete"
    
    metadata = load_metadata(metadata_file)
    if not metadata:
        return
    
    missing_chapters, existing_chapters = find_missing_chapters(output_dir, metadata)
    
    print("📊 QUICK RESUME ANALYSIS")
    print("=" * 40)
    print(f"📋 Total chapters in metadata: {len(metadata['chapters'])}")
    print(f"📁 Downloaded chapters: {len(existing_chapters)}")
    print(f"❌ Missing chapters: {len(missing_chapters)}")
    
    if not missing_chapters:
        print("✅ All chapters downloaded!")
        return
    
    # Show first few missing chapters with URLs
    print(f"\n🔍 MISSING CHAPTERS (first 10):")
    for ch in missing_chapters[:10]:
        chapter_info = metadata["chapters"].get(str(ch))
        if chapter_info:
            print(f"   {ch}: {chapter_info.get('url', 'No URL')}")
        else:
            print(f"   {ch}: Not in metadata")
    
    if len(missing_chapters) > 10:
        print(f"   ... and {len(missing_chapters) - 10} more")
    
    # Quick resume options
    print(f"\n🚀 RESUME OPTIONS:")
    print(f"1. Resume from first missing chapter ({min(missing_chapters)})")
    print(f"2. Resume from last known chapter ({max(metadata['chapters'].keys())})")
    print(f"3. Resume from specific chapter")
    
    if len(sys.argv) > 1:
        try:
            target_chapter = int(sys.argv[1])
            if target_chapter in missing_chapters:
                chapter_info = metadata["chapters"].get(str(target_chapter))
                if chapter_info and chapter_info.get("url"):
                    print(f"\n🎯 RESUMING FROM CHAPTER {target_chapter}")
                    print(f"🔗 URL: {chapter_info['url']}")
                    print(f"\nRun: python metadata_scraper.py --resume {target_chapter}")
                else:
                    print(f"❌ No URL found for Chapter {target_chapter}")
            else:
                print(f"✅ Chapter {target_chapter} already exists!")
        except ValueError:
            print("❌ Invalid chapter number")
    else:
        first_missing = min(missing_chapters)
        chapter_info = metadata["chapters"].get(str(first_missing))
        if chapter_info and chapter_info.get("url"):
            print(f"\n🎯 TO RESUME FROM FIRST MISSING:")
            print(f"python metadata_scraper.py --resume {first_missing}")
            print(f"URL: {chapter_info['url']}")

if __name__ == "__main__":
    main()