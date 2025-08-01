"""
Simple test to validate alignment map builder functionality
without complex dependencies.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Direct import to avoid dependency issues
try:
    from utils.alignment_map_builder import (
        extract_chapter_number_from_filename,
        validate_chapter_file,
        scan_directory_for_chapters,
        get_alignment_map_path,
        build_and_save_alignment_map
    )
    print("✅ Successfully imported alignment map builder functions")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_basic_functionality():
    """Test basic functionality with minimal dependencies."""
    print("\n🔍 Testing basic functionality...")
    
    # Test 1: Chapter number extraction
    print("📝 Testing chapter number extraction...")
    test_cases = [
        ("English-Chapter-0001.txt", 1),
        ("Chapter-0042-Title.txt", 42),
        ("invalid_file.txt", None),
    ]
    
    for filename, expected in test_cases:
        result = extract_chapter_number_from_filename(filename)
        if result == expected:
            print(f"  ✅ {filename} -> {result}")
        else:
            print(f"  ❌ {filename} -> {result} (expected {expected})")
            return False
    
    # Test 2: Path generation
    print("📁 Testing path generation...")
    try:
        path = get_alignment_map_path("test_novel")
        expected_filename = "test_novel_alignment_map.json"
        if os.path.basename(path) == expected_filename:
            print(f"  ✅ Path generation: {path}")
        else:
            print(f"  ❌ Path generation failed: {path}")
            return False
    except Exception as e:
        print(f"  ❌ Path generation error: {e}")
        return False
    
    # Test 3: File validation with temp files
    print("🔍 Testing file validation...")
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        test_files = [
            ("valid_chapter.txt", "This is a valid chapter with sufficient content for testing."),
            ("empty_file.txt", ""),
            ("short_file.txt", "Hi"),
        ]
        
        for filename, content in test_files:
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            validation = validate_chapter_file(file_path)
            
            if filename == "valid_chapter.txt":
                if validation["valid"]:
                    print(f"  ✅ {filename} validated as valid")
                else:
                    print(f"  ❌ {filename} should be valid but isn't: {validation['errors']}")
                    return False
            
            elif filename == "empty_file.txt":
                if not validation["valid"]:
                    print(f"  ✅ {filename} correctly identified as invalid")
                else:
                    print(f"  ❌ {filename} should be invalid but isn't")
                    return False
    
    # Test 4: Directory scanning
    print("📂 Testing directory scanning...")
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test directory with chapters
        chapter_dir = os.path.join(temp_dir, "chapters")
        os.makedirs(chapter_dir)
        
        # Create test chapter files
        chapters = [
            ("Chapter-0001-Test.txt", "Chapter 1 content"),
            ("Chapter-0002-Test.txt", "Chapter 2 content"),
            ("Chapter-0003-Test.txt", "Chapter 3 content"),
            ("not_a_chapter.txt", "Not a chapter"),
        ]
        
        for filename, content in chapters:
            file_path = os.path.join(chapter_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Scan directory
        chapters_found = scan_directory_for_chapters(chapter_dir, validate_files=True)
        
        # Should find 3 chapters (excluding not_a_chapter.txt)
        if len(chapters_found) == 3:
            print(f"  ✅ Found {len(chapters_found)} chapters correctly")
        else:
            print(f"  ❌ Expected 3 chapters, found {len(chapters_found)}")
            return False
        
        # Check chapter numbers
        expected_chapters = {1, 2, 3}
        found_chapters = set(chapters_found.keys())
        if found_chapters == expected_chapters:
            print(f"  ✅ Chapter numbers correct: {found_chapters}")
        else:
            print(f"  ❌ Chapter numbers wrong: {found_chapters} (expected {expected_chapters})")
            return False
    
    print("✅ All basic tests passed!")
    return True

def test_with_real_data():
    """Test with real project data if available."""
    print("\n🔍 Testing with real project data...")
    
    project_root = Path(__file__).parent.parent
    chinese_dir = project_root / "data/novels/way_of_the_devil/raw_chapters"
    english_dir = project_root / "data/novels/way_of_the_devil/official_english"
    
    if not chinese_dir.exists():
        print(f"⚠️  Chinese directory not found: {chinese_dir}")
        return True  # Skip test, don't fail
    
    if not english_dir.exists():
        print(f"⚠️  English directory not found: {english_dir}")
        return True  # Skip test, don't fail
    
    print(f"📁 Chinese directory: {chinese_dir}")
    print(f"📁 English directory: {english_dir}")
    
    # Test directory scanning with real data
    try:
        print("🔍 Scanning Chinese directory...")
        chinese_chapters = scan_directory_for_chapters(str(chinese_dir), validate_files=False)
        print(f"  📊 Found {len(chinese_chapters)} Chinese chapters")
        
        print("🔍 Scanning English directory...")
        english_chapters = scan_directory_for_chapters(str(english_dir), validate_files=False)
        print(f"  📊 Found {len(english_chapters)} English chapters")
        
        # Calculate overlap
        chinese_nums = set(chinese_chapters.keys())
        english_nums = set(english_chapters.keys())
        overlap = chinese_nums.intersection(english_nums)
        
        print(f"  🔗 Overlap: {len(overlap)} chapters")
        
        if len(overlap) > 0:
            print(f"  ✅ Found overlapping chapters: {sorted(list(overlap))[:5]}...")
        else:
            print(f"  ⚠️  No overlapping chapters found")
        
        # Test build with real data (using temp output)
        print("🔨 Testing build with real data...")
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_alignment_map.json")
            
            success, message, build_stats = build_and_save_alignment_map(
                str(chinese_dir),
                str(english_dir),
                "test_novel",
                output_path
            )
            
            if success:
                print(f"  ✅ Build successful: {message}")
                print(f"  📊 Build stats: {build_stats}")
                
                # Verify file was created
                if os.path.exists(output_path):
                    with open(output_path, 'r', encoding='utf-8') as f:
                        alignment_map = json.load(f)
                    print(f"  ✅ Alignment map saved with {len(alignment_map)} entries")
                else:
                    print(f"  ❌ Alignment map file not created")
                    return False
                
            else:
                print(f"  ❌ Build failed: {message}")
                return False
        
    except Exception as e:
        print(f"❌ Error testing with real data: {e}")
        return False
    
    print("✅ Real data tests passed!")
    return True

def main():
    """Main test runner."""
    print("🧪 Simple Alignment Map Builder Test")
    print("=" * 50)
    
    # Run basic functionality tests
    if not test_basic_functionality():
        print("❌ Basic functionality tests failed")
        return 1
    
    # Run real data tests
    if not test_with_real_data():
        print("❌ Real data tests failed")
        return 1
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! Alignment map builder is working correctly.")
    print("=" * 50)
    
    return 0

if __name__ == "__main__":
    exit(main())