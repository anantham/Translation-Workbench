#!/usr/bin/env python3
"""
CLI tool for building alignment maps outside of the Streamlit interface.

This tool provides command-line access to the alignment map building functionality,
allowing users to create alignment maps for novels programmatically.

Usage:
    python scripts/build_alignment_map.py [OPTIONS] NOVEL_NAME CHINESE_DIR ENGLISH_DIR

Examples:
    # Build alignment map for a new novel
    python scripts/build_alignment_map.py "my_novel" data/chinese_chapters data/english_chapters
    
    # Preview alignment without building
    python scripts/build_alignment_map.py --preview "my_novel" data/chinese_chapters data/english_chapters
    
    # Build with custom output path
    python scripts/build_alignment_map.py --output custom_map.json "my_novel" data/chinese_chapters data/english_chapters
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import (
    build_alignment_map,
    preview_alignment_mapping,
    validate_chapter_directories,
    get_alignment_map_path,
    list_alignment_maps
)

def main():
    parser = argparse.ArgumentParser(
        description="Build alignment maps for novels from Chinese and English chapter directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "way_of_the_devil" data/chinese_chapters data/english_chapters
  %(prog)s --preview "eternal_life" data/chinese data/english  
  %(prog)s --output custom.json "my_novel" /path/to/chinese /path/to/english
  %(prog)s --list
        """
    )
    
    parser.add_argument(
        "novel_name",
        nargs="?",
        help="Name of the novel (will be used for output filename)"
    )
    
    parser.add_argument(
        "chinese_dir",
        nargs="?", 
        help="Path to directory containing Chinese chapter files"
    )
    
    parser.add_argument(
        "english_dir",
        nargs="?",
        help="Path to directory containing English chapter files"
    )
    
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Preview alignment without building (shows statistics and issues)"
    )
    
    parser.add_argument(
        "--output",
        help="Custom output path for alignment map (default: central storage)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List existing alignment maps"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Only validate directories without building"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Handle list command
    if args.list:
        print("📋 Available Alignment Maps:")
        print("=" * 50)
        
        available_maps = list_alignment_maps()
        if not available_maps:
            print("No alignment maps found.")
            return 0
            
        for slug, path in available_maps.items():
            print(f"• {slug}")
            if args.verbose:
                print(f"  Path: {path}")
                # Try to get chapter count
                try:
                    from utils import load_alignment_map_by_slug
                    alignment_map = load_alignment_map_by_slug(slug)
                    print(f"  Chapters: {len(alignment_map)}")
                except Exception as e:
                    print(f"  Error loading: {e}")
            print()
        return 0
    
    # Validate required arguments
    if not args.novel_name or not args.chinese_dir or not args.english_dir:
        parser.error("novel_name, chinese_dir, and english_dir are required (unless using --list)")
    
    # Validate directories exist
    if not os.path.exists(args.chinese_dir):
        print(f"❌ Chinese directory not found: {args.chinese_dir}")
        return 1
        
    if not os.path.exists(args.english_dir):
        print(f"❌ English directory not found: {args.english_dir}")
        return 1
    
    print(f"🔨 Building alignment map for: {args.novel_name}")
    print(f"📁 Chinese directory: {args.chinese_dir}")
    print(f"📁 English directory: {args.english_dir}")
    print()
    
    # Handle validation-only mode
    if args.validate:
        print("📋 Validating directories...")
        try:
            validation_result = validate_chapter_directories(args.chinese_dir, args.english_dir)
            
            if validation_result.get('valid', False):
                print("✅ Directory validation passed!")
                if args.verbose:
                    print(f"Chinese files: {validation_result.get('chinese_file_count', 0)}")
                    print(f"English files: {validation_result.get('english_file_count', 0)}")
                return 0
            else:
                print("❌ Directory validation failed!")
                errors = validation_result.get('errors', [])
                for error in errors:
                    print(f"  • {error}")
                return 1
                
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return 1
    
    # Handle preview mode
    if args.preview:
        print("👀 Previewing alignment mapping...")
        try:
            preview_result = preview_alignment_mapping(args.chinese_dir, args.english_dir)
            
            print("📊 Preview Results:")
            print(f"  Chinese files: {preview_result.get('chinese_file_count', 0)}")
            print(f"  English files: {preview_result.get('english_file_count', 0)}")
            print(f"  Potential alignments: {preview_result.get('potential_alignments', 0)}")
            
            issues = preview_result.get('issues', [])
            if issues:
                print("⚠️  Issues found:")
                for issue in issues:
                    print(f"  • {issue}")
            else:
                print("✅ No issues found!")
            
            return 0
            
        except Exception as e:
            print(f"❌ Preview error: {e}")
            return 1
    
    # Build alignment map
    print("🔨 Building alignment map...")
    try:
        success, message, stats = build_alignment_map(
            args.chinese_dir,
            args.english_dir, 
            args.novel_name,
            args.output
        )
        
        if success:
            print(f"✅ {message}")
            if args.verbose and stats:
                print("📊 Build Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                    
            # Show where it was saved
            if args.output:
                print(f"📁 Saved to: {args.output}")
            else:
                default_path = get_alignment_map_path(args.novel_name)
                print(f"📁 Saved to: {default_path}")
                
        else:
            print(f"❌ {message}")
            return 1
            
    except Exception as e:
        print(f"❌ Build error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())