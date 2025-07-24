"""
Proper test for alignment map builder using normal imports.

This is the correct approach - using the standard import system
and testing the actual functions as they would be used.
"""

import os
import sys
import json
import tempfile
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('test_alignment_proper.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def test_imports():
    """Test that we can import all required functions."""
    logger.info("Testing imports...")
    
    try:
        from utils.alignment_map_builder import (
            extract_chapter_number_from_filename,
            validate_chapter_file,
            scan_directory_for_chapters,
            validate_chapter_directories,
            preview_alignment_mapping,
            build_alignment_map_from_directories,
            get_alignment_map_path,
            build_and_save_alignment_map
        )
        logger.info("✅ All imports successful")
        return True, {
            'extract_chapter_number_from_filename': extract_chapter_number_from_filename,
            'validate_chapter_file': validate_chapter_file,
            'scan_directory_for_chapters': scan_directory_for_chapters,
            'validate_chapter_directories': validate_chapter_directories,
            'preview_alignment_mapping': preview_alignment_mapping,
            'build_alignment_map_from_directories': build_alignment_map_from_directories,
            'get_alignment_map_path': get_alignment_map_path,
            'build_and_save_alignment_map': build_and_save_alignment_map
        }
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        return False, None

def test_chapter_number_extraction(functions):
    """Test chapter number extraction from filenames."""
    logger.info("Testing chapter number extraction...")
    
    extract_func = functions['extract_chapter_number_from_filename']
    
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
        result = extract_func(filename)
        if result == expected:
            logger.info(f"  ✅ {filename} -> {result}")
        else:
            logger.error(f"  ❌ {filename} -> {result} (expected {expected})")
            return False
    
    logger.info("✅ Chapter number extraction tests passed")
    return True

def test_file_validation(functions):
    """Test file validation with various scenarios."""
    logger.info("Testing file validation...")
    
    validate_func = functions['validate_chapter_file']
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test cases: (filename, content, should_be_valid, description)
        test_cases = [
            ("valid_chapter.txt", "This is a valid chapter with sufficient content for testing. It has enough text to pass validation.", True, "Valid chapter file"),
            ("empty_file.txt", "", False, "Empty file"),
            ("short_file.txt", "Hi", False, "Very short file"),
            ("html_file.txt", "<!DOCTYPE html><html><body>Not a chapter</body></html>", False, "HTML content"),
            ("chinese_file.txt", "这是一个中文章节测试，内容足够长以通过验证。", True, "Chinese text file"),
        ]
        
        for filename, content, should_be_valid, description in test_cases:
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            validation = validate_func(file_path)
            
            if validation["valid"] == should_be_valid:
                logger.info(f"  ✅ {description}: {validation['valid']} (expected {should_be_valid})")
            else:
                logger.error(f"  ❌ {description}: {validation['valid']} (expected {should_be_valid})")
                logger.error(f"    Errors: {validation['errors']}")
                return False
    
    logger.info("✅ File validation tests passed")
    return True

def test_directory_scanning(functions):
    """Test directory scanning functionality."""
    logger.info("Testing directory scanning...")
    
    scan_func = functions['scan_directory_for_chapters']
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test chapters
        chapters = [
            ("Chapter-0001-Test.txt", "Chapter 1 content with sufficient text for validation."),
            ("Chapter-0002-Test.txt", "Chapter 2 content with sufficient text for validation."),
            ("Chapter-0003-Test.txt", "Chapter 3 content with sufficient text for validation."),
            ("not_a_chapter.txt", "This file doesn't match the pattern."),
            ("English-Chapter-0001.txt", "English chapter 1 content."),
            ("English-Chapter-0002.txt", "English chapter 2 content."),
        ]
        
        for filename, content in chapters:
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Test scanning without pattern (should find all chapters)
        all_chapters = scan_func(temp_dir, validate_files=True)
        logger.info(f"  Found {len(all_chapters)} chapters total")
        
        # Test scanning with Chinese pattern
        chinese_chapters = scan_func(temp_dir, file_pattern="Chapter-\\d+", validate_files=True)
        logger.info(f"  Found {len(chinese_chapters)} Chinese chapters")
        
        # Test scanning with English pattern
        english_chapters = scan_func(temp_dir, file_pattern="English-Chapter-\\d+", validate_files=True)
        logger.info(f"  Found {len(english_chapters)} English chapters")
        
        # Verify results
        if len(chinese_chapters) == 3 and len(english_chapters) == 2:
            logger.info("  ✅ Pattern filtering working correctly")
        else:
            logger.error(f"  ❌ Expected 3 Chinese and 2 English chapters, got {len(chinese_chapters)} and {len(english_chapters)}")
            return False
    
    logger.info("✅ Directory scanning tests passed")
    return True

def test_directory_validation(functions):
    """Test directory validation functionality."""
    logger.info("Testing directory validation...")
    
    validate_func = functions['validate_chapter_directories']
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create Chinese directory
        chinese_dir = os.path.join(temp_dir, "chinese")
        os.makedirs(chinese_dir)
        
        chinese_chapters = [
            ("Chapter-0001-中文.txt", "这是第一章的内容，有足够的文本用于测试。"),
            ("Chapter-0002-中文.txt", "这是第二章的内容，有足够的文本用于测试。"),
            ("Chapter-0003-中文.txt", "这是第三章的内容，有足够的文本用于测试。"),
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
            ("English-Chapter-0004.txt", "This is chapter four content with sufficient text."),  # Gap at 3
        ]
        
        for filename, content in english_chapters:
            file_path = os.path.join(english_dir, filename)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        
        # Test validation
        validation = validate_func(chinese_dir, english_dir)
        
        if validation["valid"]:
            logger.info("  ✅ Directory validation passed")
            logger.info(f"  Chinese chapters: {validation['chinese_stats']['chapter_count']}")
            logger.info(f"  English chapters: {validation['english_stats']['chapter_count']}")
            logger.info(f"  Overlap: {validation['overlap_count']}")
        else:
            logger.error(f"  ❌ Directory validation failed: {validation['errors']}")
            return False
    
    logger.info("✅ Directory validation tests passed")
    return True

def test_preview_functionality(functions):
    """Test preview alignment mapping functionality."""
    logger.info("Testing preview functionality...")
    
    preview_func = functions['preview_alignment_mapping']
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test directories with overlapping chapters
        chinese_dir = os.path.join(temp_dir, "chinese")
        english_dir = os.path.join(temp_dir, "english")
        os.makedirs(chinese_dir)
        os.makedirs(english_dir)
        
        # Create overlapping chapters
        for i in range(1, 4):
            # Chinese chapter
            chinese_file = os.path.join(chinese_dir, f"Chapter-{i:04d}-中文.txt")
            with open(chinese_file, 'w', encoding='utf-8') as f:
                f.write(f"这是第{i}章的内容，有足够的文本用于测试。")
            
            # English chapter
            english_file = os.path.join(english_dir, f"English-Chapter-{i:04d}.txt")
            with open(english_file, 'w', encoding='utf-8') as f:
                f.write(f"This is chapter {i} content with sufficient text.")
        
        # Create a Chinese-only chapter
        chinese_only_file = os.path.join(chinese_dir, "Chapter-0005-中文.txt")
        with open(chinese_only_file, 'w', encoding='utf-8') as f:
            f.write("这是第5章的内容，只有中文版本。")
        
        # Create an English-only chapter
        english_only_file = os.path.join(english_dir, "English-Chapter-0006.txt")
        with open(english_only_file, 'w', encoding='utf-8') as f:
            f.write("This is chapter 6 content, English only.")
        
        # Test preview
        preview = preview_func(chinese_dir, english_dir)
        
        if preview["success"]:
            stats = preview["stats"]
            logger.info("  ✅ Preview successful")
            logger.info(f"  Total mappings: {stats['total_mappings']}")
            logger.info(f"  Both files: {stats['both_files']}")
            logger.info(f"  Chinese only: {stats['chinese_only']}")
            logger.info(f"  English only: {stats['english_only']}")
            
            # Verify expected results
            if stats['both_files'] == 3 and stats['chinese_only'] == 1 and stats['english_only'] == 1:
                logger.info("  ✅ Preview statistics correct")
            else:
                logger.error("  ❌ Preview statistics unexpected")
                return False
        else:
            logger.error(f"  ❌ Preview failed: {preview['errors']}")
            return False
    
    logger.info("✅ Preview functionality tests passed")
    return True

def test_build_functionality(functions):
    """Test build alignment map functionality."""
    logger.info("Testing build functionality...")
    
    build_func = functions['build_and_save_alignment_map']
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test directories
        chinese_dir = os.path.join(temp_dir, "chinese")
        english_dir = os.path.join(temp_dir, "english")
        os.makedirs(chinese_dir)
        os.makedirs(english_dir)
        
        # Create test chapters
        for i in range(1, 4):
            # Chinese chapter
            chinese_file = os.path.join(chinese_dir, f"Chapter-{i:04d}-中文.txt")
            with open(chinese_file, 'w', encoding='utf-8') as f:
                f.write(f"这是第{i}章的内容，有足够的文本用于测试。")
            
            # English chapter
            english_file = os.path.join(english_dir, f"English-Chapter-{i:04d}.txt")
            with open(english_file, 'w', encoding='utf-8') as f:
                f.write(f"This is chapter {i} content with sufficient text.")
        
        # Test build
        output_path = os.path.join(temp_dir, "test_alignment_map.json")
        
        success, message, build_stats = build_func(
            chinese_dir,
            english_dir,
            "test_novel",
            output_path
        )
        
        if success:
            logger.info(f"  ✅ Build successful: {message}")
            logger.info(f"  Build stats: {build_stats}")
            
            # Verify file was created
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    alignment_map = json.load(f)
                
                logger.info(f"  ✅ Alignment map created with {len(alignment_map)} entries")
                
                # Verify structure
                if "1" in alignment_map and "2" in alignment_map and "3" in alignment_map:
                    logger.info("  ✅ Expected chapters found in alignment map")
                else:
                    logger.error("  ❌ Expected chapters missing from alignment map")
                    return False
            else:
                logger.error("  ❌ Alignment map file not created")
                return False
        else:
            logger.error(f"  ❌ Build failed: {message}")
            return False
    
    logger.info("✅ Build functionality tests passed")
    return True

def test_real_data(functions):
    """Test with real project data if available."""
    logger.info("Testing with real project data...")
    
    chinese_dir = project_root / "data/novels/way_of_the_devil/raw_chapters"
    english_dir = project_root / "data/novels/way_of_the_devil/official_english"
    
    if not chinese_dir.exists():
        logger.warning(f"Chinese directory not found: {chinese_dir}")
        return True  # Skip test, don't fail
    
    if not english_dir.exists():
        logger.warning(f"English directory not found: {english_dir}")
        return True  # Skip test, don't fail
    
    logger.info(f"Found Chinese directory: {chinese_dir}")
    logger.info(f"Found English directory: {english_dir}")
    
    # Test scanning
    scan_func = functions['scan_directory_for_chapters']
    
    chinese_chapters = scan_func(str(chinese_dir), validate_files=False)
    english_chapters = scan_func(str(english_dir), validate_files=False)
    
    logger.info(f"Chinese chapters found: {len(chinese_chapters)}")
    logger.info(f"English chapters found: {len(english_chapters)}")
    
    # Calculate overlap
    overlap = set(chinese_chapters.keys()) & set(english_chapters.keys())
    logger.info(f"Overlapping chapters: {len(overlap)}")
    
    if len(overlap) > 0:
        sample_overlap = sorted(list(overlap))[:5]
        logger.info(f"Sample overlapping chapters: {sample_overlap}")
    
    # Test preview with real data
    preview_func = functions['preview_alignment_mapping']
    
    try:
        preview = preview_func(str(chinese_dir), str(english_dir))
        if preview["success"]:
            logger.info("✅ Real data preview successful")
            logger.info(f"Real data stats: {preview['stats']}")
        else:
            logger.error(f"❌ Real data preview failed: {preview['errors']}")
            return False
    except Exception as e:
        logger.error(f"❌ Real data preview error: {e}")
        return False
    
    logger.info("✅ Real data tests passed")
    return True

def main():
    """Main test runner."""
    logger.info("🧪 Proper Alignment Map Builder Tests")
    logger.info("=" * 60)
    
    # Test 1: Import all functions
    success, functions = test_imports()
    if not success:
        logger.error("❌ Import test failed - cannot continue")
        return 1
    
    # Define test functions
    test_functions = [
        ("Chapter Number Extraction", test_chapter_number_extraction),
        ("File Validation", test_file_validation),
        ("Directory Scanning", test_directory_scanning),
        ("Directory Validation", test_directory_validation),
        ("Preview Functionality", test_preview_functionality),
        ("Build Functionality", test_build_functionality),
        ("Real Data Test", test_real_data),
    ]
    
    # Run all tests
    passed = 0
    failed = 0
    
    for test_name, test_func in test_functions:
        logger.info(f"\n🔍 Running {test_name}...")
        try:
            if test_func(functions):
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                failed += 1
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"💥 {test_name} ERROR: {e}")
        
        logger.info("-" * 40)
    
    # Final summary
    logger.info("=" * 60)
    logger.info(f"📊 FINAL RESULTS: {passed} passed, {failed} failed")
    
    if failed == 0:
        logger.info("🎉 All tests passed! Alignment map builder is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Check the logs above for details.")
        return 1

if __name__ == "__main__":
    exit(main())