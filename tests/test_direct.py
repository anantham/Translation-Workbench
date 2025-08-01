"""
Direct test of alignment map builder module.
Bypasses utils/__init__.py to avoid dependency issues.
"""

import os
import sys
import json
import tempfile
import importlib.util
from pathlib import Path

# Get the alignment_map_builder module directly
project_root = Path(__file__).parent.parent
alignment_builder_path = project_root / "utils" / "alignment_map_builder.py"

# Mock logger before importing
class MockLogger:
    def debug(self, msg): print(f"DEBUG: {msg}")
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg): print(f"ERROR: {e}")

# Load the module directly
spec = importlib.util.spec_from_file_location("alignment_map_builder", alignment_builder_path)
alignment_module = importlib.util.module_from_spec(spec)

# Create a mock logging module
class MockLoggingModule:
    def __init__(self):
        self.logger = MockLogger()

sys.modules['utils.logging'] = MockLoggingModule()

# Now import the module
try:
    spec.loader.exec_module(alignment_module)
    print("✅ Successfully loaded alignment_map_builder module directly")
except Exception as e:
    print(f"❌ Failed to load module: {e}")
    # Try a different approach - let's just import normally and handle the error
    try:
        sys.path.insert(0, str(project_root))
        from utils.alignment_map_builder import (
            extract_chapter_number_from_filename,
            validate_chapter_file,
            scan_directory_for_chapters,
            get_alignment_map_path,
            build_and_save_alignment_map,
            preview_alignment_mapping
        )
        print("✅ Successfully imported alignment_map_builder functions normally")
        # Create a mock module object for consistency
        alignment_module = type('MockModule', (), {
            'extract_chapter_number_from_filename': extract_chapter_number_from_filename,
            'validate_chapter_file': validate_chapter_file,
            'scan_directory_for_chapters': scan_directory_for_chapters,
            'get_alignment_map_path': get_alignment_map_path,
            'build_and_save_alignment_map': build_and_save_alignment_map,
            'preview_alignment_mapping': preview_alignment_mapping
        })()
    except Exception as e2:
        print(f"❌ Normal import also failed: {e2}")
        sys.exit(1)

def test_functions():
    """Test all alignment map builder functions."""
    print("\n🧪 Testing alignment map builder functions...")
    
    # Test 1: Chapter number extraction
    print("📝 Testing chapter number extraction...")
    test_cases = [
        ("English-Chapter-0001.txt", 1),
        ("English-Chapter-0042.txt", 42),
        ("Chapter-0001-Title.txt", 1),
        ("random_file.txt", None),
    ]
    
    for filename, expected in test_cases:
        result = extract_chapter_number_from_filename(filename)
        if result == expected:
            print(f"  ✅ {filename} -> {result}")
        else:
            print(f"  ❌ {filename} -> {result} (expected {expected})")
            return False
    
    # Test 2: File validation
    print("\n🔍 Testing file validation...")
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a valid test file
        test_file = os.path.join(temp_dir, "test_chapter.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("This is a test chapter with sufficient content for validation testing.")
        
        validation = validate_chapter_file(test_file)
        
        if validation["valid"]:
            print(f"  ✅ File validation working: {validation['content_length']} chars")
        else:
            print(f"  ❌ File validation failed: {validation['errors']}")
            return False
        
        # Test empty file
        empty_file = os.path.join(temp_dir, "empty.txt")
        with open(empty_file, 'w', encoding='utf-8') as f:
            f.write("")
        
        validation = validate_chapter_file(empty_file)
        if not validation["valid"]:
            print(f"  ✅ Empty file correctly rejected")
        else:
            print(f"  ❌ Empty file should be invalid")
            return False
    
    # Test 3: Directory scanning
    print("\n📂 Testing directory scanning...")
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test chapters
        chapters = [
            ("Chapter-0001-Test.txt", "Chapter 1 content with sufficient text."),
            ("Chapter-0002-Test.txt", "Chapter 2 content with sufficient text."),
            ("Chapter-0003-Test.txt", "Chapter 3 content with sufficient text."),
        ]
        
        for filename, content in chapters:
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        chapters_found = scan_directory_for_chapters(temp_dir)
        
        if len(chapters_found) == 3:
            print(f"  ✅ Found {len(chapters_found)} chapters")
        else:
            print(f"  ❌ Expected 3 chapters, found {len(chapters_found)}")
            return False
        
        expected_nums = {1, 2, 3}
        found_nums = set(chapters_found.keys())
        if found_nums == expected_nums:
            print(f"  ✅ Chapter numbers correct: {found_nums}")
        else:
            print(f"  ❌ Chapter numbers wrong: {found_nums}")
            return False
    
    # Test 4: Path generation
    print("\n📁 Testing path generation...")
    path = get_alignment_map_path("test_novel")
    expected_filename = "test_novel_alignment_map.json"
    
    if os.path.basename(path) == expected_filename:
        print(f"  ✅ Path generation: {path}")
    else:
        print(f"  ❌ Path generation failed: {path}")
        return False
    
    # Test 5: Build alignment map
    print("\n🔨 Testing build alignment map...")
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create Chinese directory
        chinese_dir = os.path.join(temp_dir, "chinese")
        os.makedirs(chinese_dir)
        
        chinese_chapters = [
            ("Chapter-0001-中文.txt", "这是第一章的内容，有足够的文本用于测试。"),
            ("Chapter-0002-中文.txt", "这是第二章的内容，有足够的文本用于测试。"),
        ]
        
        for filename, content in chinese_chapters:
            file_path = os.path.join(chinese_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Create English directory
        english_dir = os.path.join(temp_dir, "english")
        os.makedirs(english_dir)
        
        english_chapters = [
            ("English-Chapter-0001.txt", "This is chapter one content with sufficient text."),
            ("English-Chapter-0002.txt", "This is chapter two content with sufficient text."),
        ]
        
        for filename, content in english_chapters:
            file_path = os.path.join(english_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Test build
        output_path = os.path.join(temp_dir, "test_alignment.json")
        
        success, message, build_stats = build_and_save_alignment_map(
            chinese_dir,
            english_dir,
            "test_novel",
            output_path
        )
        
        if success:
            print(f"  ✅ Build successful: {message}")
            print(f"  📊 Stats: {build_stats}")
            
            # Verify file
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    alignment_map = json.load(f)
                print(f"  ✅ Alignment map with {len(alignment_map)} entries")
            else:
                print(f"  ❌ Output file not created")
                return False
        else:
            print(f"  ❌ Build failed: {message}")
            return False
    
    return True

def test_with_real_data():
    """Test with real project data."""
    print("\n📊 Testing with real project data...")
    
    chinese_dir = project_root / "data/novels/way_of_the_devil/raw_chapters"
    english_dir = project_root / "data/novels/way_of_the_devil/official_english"
    
    if not chinese_dir.exists():
        print(f"  ⚠️  Chinese directory not found: {chinese_dir}")
        return True
    
    if not english_dir.exists():
        print(f"  ⚠️  English directory not found: {english_dir}")
        return True
    
    print(f"  📁 Found Chinese directory: {chinese_dir}")
    print(f"  📁 Found English directory: {english_dir}")
    
    # Scan directories
    chinese_chapters = scan_directory_for_chapters(str(chinese_dir))
    english_chapters = scan_directory_for_chapters(str(english_dir))
    
    print(f"  📊 Chinese chapters: {len(chinese_chapters)}")
    print(f"  📊 English chapters: {len(english_chapters)}")
    
    # Calculate overlap
    overlap = set(chinese_chapters.keys()) & set(english_chapters.keys())
    print(f"  🔗 Overlap: {len(overlap)} chapters")
    
    if len(overlap) > 0:
        print(f"  ✅ Sample overlap: {sorted(list(overlap))[:5]}")
    
    # Test preview
    print("  🔍 Testing preview...")
    try:
        preview = preview_alignment_mapping(str(chinese_dir), str(english_dir))
        if preview['success']:
            print(f"  ✅ Preview successful: {preview['stats']}")
        else:
            print(f"  ❌ Preview failed: {preview['errors']}")
            return False
    except Exception as e:
        print(f"  ❌ Preview error: {e}")
        return False
    
    return True

def main():
    """Main test runner."""
    print("🧪 Direct Alignment Map Builder Test")
    print("=" * 50)
    
    # Test functions
    if not test_functions():
        print("❌ Function tests failed")
        return 1
    
    # Test with real data
    if not test_with_real_data():
        print("❌ Real data tests failed")
        return 1
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! Alignment map builder is working correctly.")
    print("=" * 50)
    
    return 0

if __name__ == "__main__":
    exit(main())