"""
Comprehensive Unit Tests for Alignment Map Builder

Tests all core functions with real data and edge cases.
Logs detailed results to test_results.log for analysis.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.alignment_map_builder import (
    extract_chapter_number_from_filename,
    validate_chapter_file,
    scan_directory_for_chapters,
    validate_chapter_directories,
    preview_alignment_mapping,
    build_alignment_map_from_directories,
    save_alignment_map_with_backup,
    get_alignment_map_path,
    build_and_save_alignment_map
)

# Configure test logging
def setup_test_logging():
    """Setup comprehensive logging for test results."""
    log_file = os.path.join(os.path.dirname(__file__), 'test_results.log')
    
    # Create custom logger
    logger = logging.getLogger('test_alignment_builder')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # File handler for detailed logging
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler for immediate feedback
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Detailed formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)8s | %(funcName)20s | %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize test logger
test_logger = setup_test_logging()

class TestAlignmentMapBuilder:
    """Test suite for alignment map builder functions."""
    
    def __init__(self):
        self.test_logger = test_logger
        self.test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }
        
        # Test data paths (using actual project data)
        self.project_root = Path(__file__).parent.parent
        self.real_chinese_dir = self.project_root / "data/novels/way_of_the_devil/raw_chapters"
        self.real_english_dir = self.project_root / "data/novels/way_of_the_devil/official_english"
        self.eternal_chinese_dir = self.project_root / "data/novels/eternal_novelcool/raw_chapters"
        
        self.test_logger.info("="*80)
        self.test_logger.info("STARTING COMPREHENSIVE ALIGNMENT MAP BUILDER TESTS")
        self.test_logger.info("="*80)
        self.test_logger.info(f"Project root: {self.project_root}")
        self.test_logger.info(f"Real Chinese dir: {self.real_chinese_dir}")
        self.test_logger.info(f"Real English dir: {self.real_english_dir}")
        self.test_logger.info(f"Eternal Chinese dir: {self.eternal_chinese_dir}")
        
    def run_test(self, test_name, test_func):
        """Run a single test with error handling and logging."""
        self.test_results['total_tests'] += 1
        
        try:
            self.test_logger.info(f"RUNNING: {test_name}")
            result = test_func()
            
            if result:
                self.test_results['passed_tests'] += 1
                self.test_logger.info(f"✅ PASSED: {test_name}")
                self.test_results['test_details'].append({
                    'test': test_name,
                    'status': 'PASSED',
                    'details': result if isinstance(result, str) else 'Success'
                })
            else:
                self.test_results['failed_tests'] += 1
                self.test_logger.error(f"❌ FAILED: {test_name}")
                self.test_results['test_details'].append({
                    'test': test_name,
                    'status': 'FAILED',
                    'details': 'Test returned False'
                })
                
        except Exception as e:
            self.test_results['failed_tests'] += 1
            self.test_logger.error(f"💥 ERROR: {test_name} - {str(e)}")
            self.test_results['test_details'].append({
                'test': test_name,
                'status': 'ERROR',
                'details': str(e)
            })
    
    def test_extract_chapter_number_from_filename(self):
        """Test chapter number extraction from various filename formats."""
        self.test_logger.info("Testing chapter number extraction patterns...")
        
        test_cases = [
            ("English-Chapter-0001.txt", 1),
            ("English-Chapter-0042.txt", 42),
            ("Chapter-0001-Title.txt", 1),
            ("Chapter-0123-Some-Long-Title.txt", 123),
            ("Chapter-0001-Eternal Life Chapter 1.txt", 1),
            ("random_file.txt", None),
            ("", None),
            ("Chapter-abc.txt", None),
        ]
        
        failed_cases = []
        for filename, expected in test_cases:
            try:
                result = extract_chapter_number_from_filename(filename)
                if result == expected:
                    self.test_logger.debug(f"✓ {filename} -> {result}")
                else:
                    self.test_logger.error(f"✗ {filename} -> {result} (expected {expected})")
                    failed_cases.append((filename, result, expected))
            except Exception as e:
                self.test_logger.error(f"✗ {filename} -> ERROR: {e}")
                failed_cases.append((filename, f"ERROR: {e}", expected))
        
        if failed_cases:
            self.test_logger.error(f"Failed cases: {failed_cases}")
            return False
        
        return True
    
    def test_validate_chapter_file(self):
        """Test file validation with real files."""
        self.test_logger.info("Testing file validation with real chapter files...")
        
        if not self.real_english_dir.exists():
            self.test_logger.warning(f"Real English dir not found: {self.real_english_dir}")
            return self.test_validate_chapter_file_with_temp_files()
        
        # Test with first few English files
        test_files = list(self.real_english_dir.glob("English-Chapter-00*.txt"))[:3]
        
        if not test_files:
            self.test_logger.warning("No English chapter files found for testing")
            return self.test_validate_chapter_file_with_temp_files()
        
        for file_path in test_files:
            self.test_logger.debug(f"Validating: {file_path}")
            validation = validate_chapter_file(str(file_path))
            
            self.test_logger.debug(f"Validation result: {validation}")
            
            if not validation["valid"]:
                self.test_logger.error(f"File failed validation: {file_path}")
                self.test_logger.error(f"Errors: {validation['errors']}")
                return False
            
            if validation["content_length"] < 10:
                self.test_logger.warning(f"File very short: {file_path} ({validation['content_length']} chars)")
        
        return True
    
    def test_validate_chapter_file_with_temp_files(self):
        """Test file validation with temporary test files."""
        self.test_logger.info("Testing file validation with temporary test files...")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Test cases: [filename, content, should_be_valid]
            test_cases = [
                ("valid_file.txt", "This is a valid chapter with good content.", True),
                ("empty_file.txt", "", False),
                ("short_file.txt", "Short", False),
                ("html_file.txt", "<!DOCTYPE html><html><body>Not a chapter</body></html>", False),
                ("utf8_file.txt", "这是一个中文章节测试", True),
            ]
            
            failed_cases = []
            for filename, content, should_be_valid in test_cases:
                file_path = temp_path / filename
                
                try:
                    file_path.write_text(content, encoding='utf-8')
                    validation = validate_chapter_file(str(file_path))
                    
                    if validation["valid"] == should_be_valid:
                        self.test_logger.debug(f"✓ {filename} -> valid={validation['valid']}")
                    else:
                        self.test_logger.error(f"✗ {filename} -> valid={validation['valid']} (expected {should_be_valid})")
                        failed_cases.append((filename, validation['valid'], should_be_valid))
                        
                except Exception as e:
                    self.test_logger.error(f"✗ {filename} -> ERROR: {e}")
                    failed_cases.append((filename, f"ERROR: {e}", should_be_valid))
            
            if failed_cases:
                self.test_logger.error(f"Failed cases: {failed_cases}")
                return False
        
        return True
    
    def test_scan_directory_for_chapters(self):
        """Test directory scanning with real chapter directories."""
        self.test_logger.info("Testing directory scanning...")
        
        # Test with real directories
        test_dirs = [
            (self.real_english_dir, "English-Chapter-\\d+\\.txt"),
            (self.eternal_chinese_dir, "Chapter-\\d+"),
        ]
        
        for dir_path, pattern in test_dirs:
            if not dir_path.exists():
                self.test_logger.warning(f"Directory not found: {dir_path}")
                continue
                
            self.test_logger.debug(f"Scanning directory: {dir_path}")
            self.test_logger.debug(f"Using pattern: {pattern}")
            
            try:
                chapters = scan_directory_for_chapters(str(dir_path), pattern, validate_files=True)
                
                self.test_logger.info(f"Found {len(chapters)} chapters in {dir_path.name}")
                
                if len(chapters) == 0:
                    self.test_logger.warning(f"No chapters found in {dir_path}")
                    continue
                
                # Log first few chapters
                for i, (ch_num, file_info) in enumerate(sorted(chapters.items())[:3]):
                    self.test_logger.debug(f"Chapter {ch_num}: {file_info['filename']}")
                    if file_info['validation'] and not file_info['validation']['valid']:
                        self.test_logger.warning(f"Chapter {ch_num} validation failed: {file_info['validation']['errors']}")
                
            except Exception as e:
                self.test_logger.error(f"Error scanning {dir_path}: {e}")
                return False
        
        return True
    
    def test_validate_chapter_directories(self):
        """Test directory validation with real directories."""
        self.test_logger.info("Testing directory validation...")
        
        # Test with real directories
        if self.real_chinese_dir.exists() and self.real_english_dir.exists():
            self.test_logger.debug("Testing with real Way of the Devil directories...")
            
            validation = validate_chapter_directories(
                str(self.real_chinese_dir), 
                str(self.real_english_dir)
            )
            
            self.test_logger.info(f"Validation result: {validation['valid']}")
            self.test_logger.info(f"Chinese chapters: {validation['chinese_stats']['chapter_count']}")
            self.test_logger.info(f"English chapters: {validation['english_stats']['chapter_count']}")
            self.test_logger.info(f"Overlap: {validation['overlap_count']}")
            
            if validation['errors']:
                self.test_logger.error(f"Validation errors: {validation['errors']}")
                return False
            
            if validation['warnings']:
                self.test_logger.warning(f"Validation warnings: {validation['warnings']}")
        
        # Test with missing directories
        validation = validate_chapter_directories("/nonexistent/dir1", "/nonexistent/dir2")
        
        if validation['valid']:
            self.test_logger.error("Validation should fail for nonexistent directories")
            return False
        
        self.test_logger.debug("✓ Correctly failed validation for nonexistent directories")
        
        return True
    
    def test_preview_alignment_mapping(self):
        """Test alignment mapping preview."""
        self.test_logger.info("Testing alignment mapping preview...")
        
        if not (self.real_chinese_dir.exists() and self.real_english_dir.exists()):
            self.test_logger.warning("Real directories not found, skipping preview test")
            return True
        
        try:
            preview = preview_alignment_mapping(
                str(self.real_chinese_dir),
                str(self.real_english_dir)
            )
            
            if not preview['success']:
                self.test_logger.error(f"Preview failed: {preview.get('errors', [])}")
                return False
            
            stats = preview['stats']
            self.test_logger.info(f"Preview stats: {stats}")
            
            # Check if we have reasonable results
            if stats['total_mappings'] == 0:
                self.test_logger.warning("No mappings found in preview")
                return False
            
            # Log sample mappings
            sample_mappings = list(preview['preview_mapping'].items())[:3]
            for ch_key, mapping in sample_mappings:
                self.test_logger.debug(f"Chapter {ch_key}: {mapping['status']}")
            
            return True
            
        except Exception as e:
            self.test_logger.error(f"Preview error: {e}")
            return False
    
    def test_build_alignment_map_from_directories(self):
        """Test building alignment map from directories."""
        self.test_logger.info("Testing alignment map building...")
        
        if not (self.real_chinese_dir.exists() and self.real_english_dir.exists()):
            self.test_logger.warning("Real directories not found, skipping build test")
            return True
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = os.path.join(temp_dir, "test_alignment_map.json")
                
                alignment_map, build_stats = build_alignment_map_from_directories(
                    str(self.real_chinese_dir),
                    str(self.real_english_dir),
                    output_path
                )
                
                if not build_stats['success']:
                    self.test_logger.error(f"Build failed: {build_stats.get('errors', [])}")
                    return False
                
                self.test_logger.info(f"Build stats: {build_stats}")
                
                # Check if we have reasonable results
                if build_stats['total_mapped'] == 0:
                    self.test_logger.warning("No chapters mapped")
                    return False
                
                # Verify alignment map structure
                if not alignment_map:
                    self.test_logger.error("Alignment map is empty")
                    return False
                
                # Log sample mappings
                sample_mappings = list(alignment_map.items())[:3]
                for ch_key, mapping in sample_mappings:
                    self.test_logger.debug(f"Chapter {ch_key}: raw={mapping.get('raw_file', 'None')}, english={mapping.get('english_file', 'None')}")
                
                return True
                
        except Exception as e:
            self.test_logger.error(f"Build error: {e}")
            return False
    
    def test_get_alignment_map_path(self):
        """Test alignment map path generation."""
        self.test_logger.info("Testing alignment map path generation...")
        
        test_cases = [
            ("way_of_the_devil", "way_of_the_devil_alignment_map.json"),
            ("Test Novel", "Test_Novel_alignment_map.json"),
            ("Novel-With-Dashes", "Novel-With-Dashes_alignment_map.json"),
            ("Novel@#$%Special", "NovelSpecial_alignment_map.json"),
        ]
        
        failed_cases = []
        for novel_name, expected_filename in test_cases:
            try:
                path = get_alignment_map_path(novel_name)
                actual_filename = os.path.basename(path)
                
                if actual_filename == expected_filename:
                    self.test_logger.debug(f"✓ {novel_name} -> {actual_filename}")
                else:
                    self.test_logger.error(f"✗ {novel_name} -> {actual_filename} (expected {expected_filename})")
                    failed_cases.append((novel_name, actual_filename, expected_filename))
                    
                # Check if directory is created
                if not os.path.exists(os.path.dirname(path)):
                    self.test_logger.error(f"Directory not created: {os.path.dirname(path)}")
                    failed_cases.append((novel_name, "Directory not created", "Should create directory"))
                    
            except Exception as e:
                self.test_logger.error(f"✗ {novel_name} -> ERROR: {e}")
                failed_cases.append((novel_name, f"ERROR: {e}", expected_filename))
        
        if failed_cases:
            self.test_logger.error(f"Failed cases: {failed_cases}")
            return False
        
        return True
    
    def test_build_and_save_alignment_map(self):
        """Test complete build and save workflow."""
        self.test_logger.info("Testing complete build and save workflow...")
        
        if not (self.real_chinese_dir.exists() and self.real_english_dir.exists()):
            self.test_logger.warning("Real directories not found, skipping complete workflow test")
            return True
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                novel_name = "test_novel"
                output_path = os.path.join(temp_dir, f"{novel_name}_alignment_map.json")
                
                success, message, build_stats = build_and_save_alignment_map(
                    str(self.real_chinese_dir),
                    str(self.real_english_dir),
                    novel_name,
                    output_path
                )
                
                if not success:
                    self.test_logger.error(f"Complete workflow failed: {message}")
                    return False
                
                self.test_logger.info(f"Complete workflow success: {message}")
                self.test_logger.info(f"Build stats: {build_stats}")
                
                # Check if file was created
                if not os.path.exists(output_path):
                    self.test_logger.error(f"Alignment map file not created: {output_path}")
                    return False
                
                # Verify file content
                with open(output_path, 'r', encoding='utf-8') as f:
                    saved_map = json.load(f)
                
                if not saved_map:
                    self.test_logger.error("Saved alignment map is empty")
                    return False
                
                self.test_logger.debug(f"Saved alignment map has {len(saved_map)} entries")
                
                return True
                
        except Exception as e:
            self.test_logger.error(f"Complete workflow error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all tests and generate summary."""
        self.test_logger.info("Starting comprehensive test suite...")
        
        # List of all tests to run
        tests = [
            ("Chapter Number Extraction", self.test_extract_chapter_number_from_filename),
            ("File Validation", self.test_validate_chapter_file),
            ("Directory Scanning", self.test_scan_directory_for_chapters),
            ("Directory Validation", self.test_validate_chapter_directories),
            ("Alignment Preview", self.test_preview_alignment_mapping),
            ("Alignment Building", self.test_build_alignment_map_from_directories),
            ("Path Generation", self.test_get_alignment_map_path),
            ("Complete Workflow", self.test_build_and_save_alignment_map),
        ]
        
        # Run all tests
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
            self.test_logger.info("-" * 60)
        
        # Generate summary
        self.generate_summary()
        
        return self.test_results
    
    def generate_summary(self):
        """Generate comprehensive test summary."""
        self.test_logger.info("=" * 80)
        self.test_logger.info("TEST SUMMARY")
        self.test_logger.info("=" * 80)
        
        results = self.test_results
        
        self.test_logger.info(f"Total Tests: {results['total_tests']}")
        self.test_logger.info(f"Passed: {results['passed_tests']}")
        self.test_logger.info(f"Failed: {results['failed_tests']}")
        
        if results['total_tests'] > 0:
            success_rate = (results['passed_tests'] / results['total_tests']) * 100
            self.test_logger.info(f"Success Rate: {success_rate:.1f}%")
        
        # Detailed results
        if results['failed_tests'] > 0:
            self.test_logger.error("\nFAILED TESTS:")
            for test_detail in results['test_details']:
                if test_detail['status'] in ['FAILED', 'ERROR']:
                    self.test_logger.error(f"  {test_detail['test']}: {test_detail['details']}")
        
        self.test_logger.info("\nAll test results have been logged to test_results.log")
        self.test_logger.info("=" * 80)

def main():
    """Main test runner."""
    print("🧪 Running Comprehensive Alignment Map Builder Tests...")
    print("📝 Detailed results will be logged to test_results.log")
    print("⏱️  This may take a few minutes with real data...")
    
    test_suite = TestAlignmentMapBuilder()
    results = test_suite.run_all_tests()
    
    print(f"\n✅ Tests completed: {results['passed_tests']}/{results['total_tests']} passed")
    
    if results['failed_tests'] > 0:
        print(f"❌ {results['failed_tests']} tests failed - check test_results.log for details")
        return 1
    else:
        print("🎉 All tests passed!")
        return 0

if __name__ == "__main__":
    exit(main())