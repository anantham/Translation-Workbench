"""
Standalone test for alignment map builder functionality.
Bypasses complex utils imports to test core functionality.
"""

import os
import sys
import json
import tempfile
import re
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Create a minimal logging setup to avoid dependency issues
class MockLogger:
    def debug(self, msg): print(f"DEBUG: {msg}")
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")

# Mock the logger for our test
import utils.alignment_map_builder as amb
amb.logger = MockLogger()

def test_direct_import():
    """Test direct import of alignment map builder functions."""
    print("🔍 Testing direct import...")
    
    try:
        # Test basic function imports
        from utils.alignment_map_builder import (
            extract_chapter_number_from_filename,
            validate_chapter_file,
            scan_directory_for_chapters,
            get_alignment_map_path,
            build_and_save_alignment_map
        )
        print("✅ Successfully imported alignment map builder functions")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_chapter_number_extraction():
    """Test chapter number extraction."""
    print("\n📝 Testing chapter number extraction...")
    
    from utils.alignment_map_builder import extract_chapter_number_from_filename
    
    test_cases = [
        ("English-Chapter-0001.txt", 1),
        ("English-Chapter-0042.txt", 42),
        ("Chapter-0001-Title.txt", 1),
        ("Chapter-0123-Some-Long-Title.txt", 123),
        ("Chapter-0001-Eternal Life Chapter 1.txt", 1),
        ("random_file.txt", None),
        ("", None),
    ]
    
    for filename, expected in test_cases:
        result = extract_chapter_number_from_filename(filename)
        if result == expected:
            print(f"  ✅ {filename} -> {result}")
        else:
            print(f"  ❌ {filename} -> {result} (expected {expected})")
            return False
    
    return True

def test_file_validation():
    """Test file validation with temporary files."""
    print("\n🔍 Testing file validation...")
    
    from utils.alignment_map_builder import validate_chapter_file
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        test_files = [
            ("valid_chapter.txt", "This is a valid chapter with sufficient content for testing. It has enough text to pass validation."),
            ("empty_file.txt", ""),
            ("short_file.txt", "Hi"),
            ("html_file.txt", "<!DOCTYPE html><html><body>Not a chapter</body></html>"),
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
                    print(f"  ❌ {filename} should be valid: {validation['errors']}")
                    return False
            
            elif filename == "empty_file.txt":
                if not validation["valid"]:
                    print(f"  ✅ {filename} correctly identified as invalid")
                else:
                    print(f"  ❌ {filename} should be invalid")
                    return False
            
            elif filename == "html_file.txt":
                if not validation["valid"]:
                    print(f"  ✅ {filename} correctly identified as HTML")
                else:
                    print(f"  ❌ {filename} should be invalid (HTML)")
                    return False
    
    return True

def test_directory_scanning():
    """Test directory scanning."""
    print("\n📂 Testing directory scanning...")
    
    from utils.alignment_map_builder import scan_directory_for_chapters
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test directory with chapters
        chapter_dir = os.path.join(temp_dir, "chapters")
        os.makedirs(chapter_dir)
        
        # Create test chapter files
        chapters = [
            ("Chapter-0001-Test.txt", "Chapter 1 content with sufficient text for validation."),
            ("Chapter-0002-Test.txt", "Chapter 2 content with sufficient text for validation."),
            ("Chapter-0003-Test.txt", "Chapter 3 content with sufficient text for validation."),
            ("not_a_chapter.txt", "This file doesn't match the pattern."),
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
            print(f"  Found chapters: {list(chapters_found.keys())}")
            return False
        
        # Check chapter numbers
        expected_chapters = {1, 2, 3}
        found_chapters = set(chapters_found.keys())
        if found_chapters == expected_chapters:
            print(f"  ✅ Chapter numbers correct: {found_chapters}")
        else:
            print(f"  ❌ Chapter numbers wrong: {found_chapters} (expected {expected_chapters})")
            return False
    
    return True

def test_path_generation():
    """Test alignment map path generation."""
    print("\n📁 Testing path generation...")
    
    from utils.alignment_map_builder import get_alignment_map_path
    
    test_cases = [
        ("way_of_the_devil", "way_of_the_devil_alignment_map.json"),
        ("Test Novel", "Test_Novel_alignment_map.json"),
        ("Novel@#$%Special", "NovelSpecial_alignment_map.json"),
    ]
    
    for novel_name, expected_filename in test_cases:
        try:
            path = get_alignment_map_path(novel_name)
            actual_filename = os.path.basename(path)
            
            if actual_filename == expected_filename:
                print(f"  ✅ {novel_name} -> {actual_filename}")
            else:
                print(f"  ❌ {novel_name} -> {actual_filename} (expected {expected_filename})")
                return False
                
        except Exception as e:
            print(f"  ❌ {novel_name} -> ERROR: {e}")
            return False
    
    return True

def test_build_alignment_map():
    """Test building alignment map with sample data."""
    print("\n🔨 Testing alignment map building...")
    
    from utils.alignment_map_builder import build_and_save_alignment_map
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create sample Chinese directory
        chinese_dir = os.path.join(temp_dir, "chinese")
        os.makedirs(chinese_dir)
        
        chinese_chapters = [
            ("Chapter-0001-中文章节一.txt", "这是第一章的内容。中文测试内容足够长以通过验证。"),
            ("Chapter-0002-中文章节二.txt", "这是第二章的内容。更多中文测试内容。"),
            ("Chapter-0003-中文章节三.txt", "这是第三章的内容。继续中文测试。"),
        ]
        
        for filename, content in chinese_chapters:
            file_path = os.path.join(chinese_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Create sample English directory
        english_dir = os.path.join(temp_dir, "english")
        os.makedirs(english_dir)
        
        english_chapters = [
            ("English-Chapter-0001.txt", "This is the content of chapter one with sufficient text."),
            ("English-Chapter-0002.txt", "This is the content of chapter two with sufficient text."),
            ("English-Chapter-0004.txt", "This is the content of chapter four with sufficient text."),  # Gap at 3
        ]
        
        for filename, content in english_chapters:
            file_path = os.path.join(english_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Test build
        output_path = os.path.join(temp_dir, "test_alignment_map.json")
        
        success, message, build_stats = build_and_save_alignment_map(
            chinese_dir,
            english_dir,
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
                
                # Check structure
                if "1" in alignment_map and "2" in alignment_map:
                    print(f"  ✅ Found expected chapters 1 and 2")
                else:
                    print(f"  ❌ Missing expected chapters in alignment map")
                    return False
                
            else:
                print(f"  ❌ Alignment map file not created")
                return False
            
        else:
            print(f"  ❌ Build failed: {message}")
            print(f"  📊 Build stats: {build_stats}")
            return False
    
    return True

def test_real_data():
    """Test with real project data if available."""
    print("\n📊 Testing with real project data...")
    
    chinese_dir = project_root / "data/novels/way_of_the_devil/raw_chapters"
    english_dir = project_root / "data/novels/way_of_the_devil/official_english"
    
    if not chinese_dir.exists():
        print(f"  ⚠️  Chinese directory not found: {chinese_dir}")
        return True  # Skip, don't fail
    
    if not english_dir.exists():
        print(f"  ⚠️  English directory not found: {english_dir}")
        return True  # Skip, don't fail
    
    print(f"  📁 Chinese directory: {chinese_dir}")
    print(f"  📁 English directory: {english_dir}")
    
    # Test scanning real directories
    from utils.alignment_map_builder import scan_directory_for_chapters
    
    try:
        print("  🔍 Scanning Chinese directory...")
        chinese_chapters = scan_directory_for_chapters(str(chinese_dir), validate_files=False)
        print(f"    📊 Found {len(chinese_chapters)} Chinese chapters")
        
        print("  🔍 Scanning English directory...")
        english_chapters = scan_directory_for_chapters(str(english_dir), validate_files=False)
        print(f"    📊 Found {len(english_chapters)} English chapters")
        
        # Calculate overlap
        chinese_nums = set(chinese_chapters.keys())
        english_nums = set(english_chapters.keys())
        overlap = chinese_nums.intersection(english_nums)
        
        print(f"    🔗 Overlap: {len(overlap)} chapters")
        
        if len(overlap) > 0:
            sample_overlap = sorted(list(overlap))[:5]
            print(f"    ✅ Sample overlapping chapters: {sample_overlap}")
        else:
            print(f"    ⚠️  No overlapping chapters found")
        
        # Test build with real data (small sample)
        print("  🔨 Testing build with real data...")
        
        from utils.alignment_map_builder import build_and_save_alignment_map
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "real_test_alignment_map.json")
            
            success, message, build_stats = build_and_save_alignment_map(
                str(chinese_dir),
                str(english_dir),
                "way_of_the_devil",
                output_path
            )
            
            if success:
                print(f"    ✅ Real data build successful")
                print(f"    📊 Total mapped: {build_stats['total_mapped']}")
                print(f"    🔗 Both files: {build_stats['both_files']}")
                print(f"    🇨🇳 Chinese only: {build_stats['chinese_only']}")
                print(f"    🇺🇸 English only: {build_stats['english_only']}")
                
                # Verify file
                if os.path.exists(output_path):
                    with open(output_path, 'r', encoding='utf-8') as f:
                        alignment_map = json.load(f)
                    print(f"    ✅ Alignment map with {len(alignment_map)} entries")
                
            else:
                print(f"    ❌ Real data build failed: {message}")
                return False
        
    except Exception as e:
        print(f"  ❌ Error with real data: {e}")
        return False
    
    return True

def main():
    """Main test runner."""
    print("🧪 Standalone Alignment Map Builder Test")
    print("=" * 60)
    
    # Track test results
    tests = [
        ("Direct Import", test_direct_import),
        ("Chapter Number Extraction", test_chapter_number_extraction),
        ("File Validation", test_file_validation),
        ("Directory Scanning", test_directory_scanning),
        ("Path Generation", test_path_generation),
        ("Build Alignment Map", test_build_alignment_map),
        ("Real Data Test", test_real_data),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"💥 {test_name} ERROR: {e}")
        
        print("-" * 40)
    
    # Final summary
    print("=" * 60)
    print(f"📊 TEST RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! Alignment map builder is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    exit(main())