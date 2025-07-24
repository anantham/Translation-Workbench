#!/usr/bin/env python3
"""
Script to find missing chapter numbers in the 永生 novel dataset.
This script will identify which specific chapters are missing from 1-1634.
"""

import os
import re
import sys

def extract_chapter_numbers(directory_path):
    """Extract all chapter numbers from filenames in the directory."""
    chapter_numbers = set()
    
    # Pattern to match chapter files
    pattern = r'Chapter-(\d+)(?:-(\d+))?-.*\.txt'
    
    for filename in os.listdir(directory_path):
        if filename.startswith('Chapter-') and filename.endswith('.txt'):
            match = re.match(pattern, filename)
            if match:
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else start
                
                # Add all chapters in the range
                for i in range(start, end + 1):
                    chapter_numbers.add(i)
    
    return chapter_numbers

def find_missing_chapters(directory_path, expected_max=1634):
    """Find missing chapter numbers from 1 to expected_max."""
    print(f"Checking directory: {directory_path}")
    print(f"Expected chapters: 1 to {expected_max}")
    print("-" * 50)
    
    # Get all present chapter numbers
    present_chapters = extract_chapter_numbers(directory_path)
    
    # Find missing chapters
    expected_chapters = set(range(1, expected_max + 1))
    missing_chapters = expected_chapters - present_chapters
    
    # Statistics
    total_files = len([f for f in os.listdir(directory_path) if f.endswith('.txt')])
    total_chapters_found = len(present_chapters)
    
    print(f"Total .txt files in directory: {total_files}")
    print(f"Total chapters covered: {total_chapters_found}")
    print(f"Expected chapters: {expected_max}")
    print(f"Missing chapters: {len(missing_chapters)}")
    print()
    
    if missing_chapters:
        missing_list = sorted(missing_chapters)
        print("Missing chapter numbers:")
        
        # Group consecutive missing chapters for easier reading
        groups = []
        start = None
        prev = None
        
        for num in missing_list:
            if start is None:
                start = num
            elif num != prev + 1:
                if start == prev:
                    groups.append(str(start))
                else:
                    groups.append(f"{start}-{prev}")
                start = num
            prev = num
        
        # Add the last group
        if start is not None:
            if start == prev:
                groups.append(str(start))
            else:
                groups.append(f"{start}-{prev}")
        
        print(", ".join(groups))
        print()
        
        # Show first 20 missing chapters individually
        if len(missing_list) > 20:
            print("First 20 missing chapters:")
            print(missing_list[:20])
        else:
            print("All missing chapters:")
            print(missing_list)
    else:
        print("✅ No missing chapters found!")
    
    return missing_chapters, present_chapters

def main():
    directory_path = "/Users/aditya/Library/CloudStorage/OneDrive-IndianInstituteofScience/Documents/Ongoing/Project 1 - Wuxia/data/novels/永生_kanunu/raw_chapters"
    
    if not os.path.exists(directory_path):
        print(f"❌ Directory not found: {directory_path}")
        sys.exit(1)
    
    missing, present = find_missing_chapters(directory_path)
    
    # Also check for any files that might have numbering issues
    print("\n" + "="*60)
    print("CHECKING FOR POTENTIAL NUMBERING ISSUES")
    print("="*60)
    
    pattern = r'Chapter-(\d+)(?:-(\d+))?-.*第(.+?)章'
    
    for filename in os.listdir(directory_path):
        if filename.startswith('Chapter-') and filename.endswith('.txt'):
            match = re.match(pattern, filename)
            if match:
                file_num = int(match.group(1))
                chinese_num = match.group(3)
                
                # Check if the Chinese chapter number seems mismatched
                # This is a basic check - you might want to improve this
                if '一千' in chinese_num and file_num < 1000:
                    print(f"⚠️  Potential mismatch: {filename[:50]}...")
                elif '二千' in chinese_num and file_num < 2000:
                    print(f"⚠️  Potential mismatch: {filename[:50]}...")

if __name__ == "__main__":
    main()
