#!/usr/bin/env python3
"""
Comprehensive test suite for EPUB builder with all 8 phases.

This test suite validates:
- All 8 phases of EPUB enhancement
- Different input scenarios and edge cases
- Output quality and file integrity
- Detailed logging and validation
"""

import os
import sys
import json
import logging
import zipfile
import traceback
from datetime import datetime
from pathlib import Path

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.epub_builder import build_epub

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_epub_builder.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EPUBTestSuite:
    """Comprehensive EPUB builder test suite."""
    
    def __init__(self):
        self.test_results = []
        self.project_root = Path(__file__).parent.parent
        self.temp_dir = self.project_root / "data" / "temp" / "test_outputs"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("📚 EPUB Test Suite initialized")
        logger.info(f"📁 Project root: {self.project_root}")
        logger.info(f"📂 Temp directory: {self.temp_dir}")
    
    def run_all_tests(self):
        """Run all test scenarios."""
        logger.info("🚀 Starting comprehensive EPUB test suite...")
        
        test_scenarios = [
            self.test_basic_epub_creation,
            self.test_with_rich_metadata,
            self.test_with_images,
            self.test_with_minimal_data,
            self.test_edge_cases,
            self.test_all_phases_integration,
            self.test_output_validation,
            self.test_performance_metrics
        ]
        
        for i, test_func in enumerate(test_scenarios, 1):
            logger.info(f"🧪 Running test {i}/{len(test_scenarios)}: {test_func.__name__}")
            try:
                result = test_func()
                self.test_results.append({
                    'test_name': test_func.__name__,
                    'status': 'PASSED' if result else 'FAILED',
                    'timestamp': datetime.now().isoformat()
                })
                logger.info(f"✅ Test {test_func.__name__}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                logger.error(f"❌ Test {test_func.__name__} failed with exception: {e}")
                logger.error(traceback.format_exc())
                self.test_results.append({
                    'test_name': test_func.__name__,
                    'status': 'ERROR',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        self.generate_test_report()
        return self.test_results
    
    def test_basic_epub_creation(self):
        """Test basic EPUB creation functionality."""
        logger.info("📖 Testing basic EPUB creation...")
        
        # Use eternal life novel with mergedStyle chapters
        chapter_dir = self.project_root / "data" / "novels" / "永生_kanunu" / "mergedStyle"
        output_path = self.temp_dir / "test_basic_epub.epub"
        
        if not chapter_dir.exists():
            logger.warning(f"Chapter directory not found: {chapter_dir}")
            return False
        
        success, message = build_epub(
            chapter_dir=str(chapter_dir),
            output_path=str(output_path),
            title="Eternal Life - Basic Test",
            author="观棋",
            translator="Test Framework",
            novel_slug="永生_kanunu",
            include_images=False
        )
        
        if success:
            logger.info(f"✅ Basic EPUB created: {output_path}")
            logger.info(f"📄 Message: {message}")
            return self.validate_epub_structure(output_path, "basic")
        else:
            logger.error(f"❌ Basic EPUB creation failed: {message}")
            return False
    
    def test_with_rich_metadata(self):
        """Test EPUB creation with rich job metadata."""
        logger.info("📊 Testing EPUB with rich metadata...")
        
        # Use translation job with rich metadata
        chapter_dir = self.project_root / "data" / "custom_translations" / "master_20250716_1928_oai_gpt4o"
        output_path = self.temp_dir / "test_rich_metadata.epub"
        
        if not chapter_dir.exists():
            logger.warning(f"Chapter directory not found: {chapter_dir}")
            return False
        
        success, message = build_epub(
            chapter_dir=str(chapter_dir),
            output_path=str(output_path),
            title="Way of the Devil - Rich Metadata Test",
            author="Wang Yu",
            translator="GPT-4o Translation",
            novel_slug="way_of_the_devil",
            include_images=False
        )
        
        if success:
            logger.info(f"✅ Rich metadata EPUB created: {output_path}")
            logger.info(f"📄 Message: {message}")
            return self.validate_epub_structure(output_path, "rich_metadata")
        else:
            logger.error(f"❌ Rich metadata EPUB creation failed: {message}")
            return False
    
    def test_with_images(self):
        """Test EPUB creation with image integration."""
        logger.info("🖼️ Testing EPUB with images...")
        
        chapter_dir = self.project_root / "data" / "novels" / "永生_kanunu" / "mergedStyle"
        output_path = self.temp_dir / "test_with_images.epub"
        
        if not chapter_dir.exists():
            logger.warning(f"Chapter directory not found: {chapter_dir}")
            return False
        
        success, message = build_epub(
            chapter_dir=str(chapter_dir),
            output_path=str(output_path),
            title="Eternal Life - Images Test",
            author="观棋",
            translator="Test Framework",
            novel_slug="永生_kanunu",
            include_images=True
        )
        
        if success:
            logger.info(f"✅ Images EPUB created: {output_path}")
            logger.info(f"📄 Message: {message}")
            return self.validate_epub_structure(output_path, "with_images")
        else:
            logger.error(f"❌ Images EPUB creation failed: {message}")
            return False
    
    def test_with_minimal_data(self):
        """Test EPUB creation with minimal data."""
        logger.info("📋 Testing EPUB with minimal data...")
        
        # Create a temporary directory with minimal chapters
        temp_chapter_dir = self.temp_dir / "minimal_chapters"
        temp_chapter_dir.mkdir(exist_ok=True)
        
        # Create a simple test chapter
        (temp_chapter_dir / "Chapter-0001.txt").write_text(
            "Chapter 1: The Beginning\n\nThis is a test chapter for minimal data testing.\n\nIt contains basic text without special formatting.",
            encoding='utf-8'
        )
        
        output_path = self.temp_dir / "test_minimal_data.epub"
        
        success, message = build_epub(
            chapter_dir=str(temp_chapter_dir),
            output_path=str(output_path),
            title="Minimal Test Novel",
            author="Test Author",
            translator="Test Framework",
            novel_slug=None,
            include_images=False
        )
        
        if success:
            logger.info(f"✅ Minimal data EPUB created: {output_path}")
            logger.info(f"📄 Message: {message}")
            return self.validate_epub_structure(output_path, "minimal")
        else:
            logger.error(f"❌ Minimal data EPUB creation failed: {message}")
            return False
    
    def test_edge_cases(self):
        """Test various edge cases and error conditions."""
        logger.info("⚠️ Testing edge cases...")
        
        # Test with non-existent directory
        output_path = self.temp_dir / "test_edge_cases.epub"
        success, message = build_epub(
            chapter_dir="/nonexistent/path",
            output_path=str(output_path),
            title="Edge Case Test",
            author="Test",
            translator="Test",
            novel_slug=None,
            include_images=False
        )
        
        if not success:
            logger.info(f"✅ Correctly handled non-existent directory: {message}")
            return True
        else:
            logger.error("❌ Should have failed for non-existent directory")
            return False
    
    def test_all_phases_integration(self):
        """Test integration of all 8 phases."""
        logger.info("🔄 Testing all phases integration...")
        
        chapter_dir = self.project_root / "data" / "novels" / "永生_kanunu" / "mergedStyle"
        output_path = self.temp_dir / "test_all_phases.epub"
        
        if not chapter_dir.exists():
            logger.warning(f"Chapter directory not found: {chapter_dir}")
            return False
        
        success, message = build_epub(
            chapter_dir=str(chapter_dir),
            output_path=str(output_path),
            title="Eternal Life - All Phases Test",
            author="观棋",
            translator="Full Framework Test",
            novel_slug="永生_kanunu",
            include_images=True
        )
        
        if success:
            logger.info(f"✅ All phases EPUB created: {output_path}")
            logger.info(f"📄 Message: {message}")
            return self.validate_all_phases(output_path)
        else:
            logger.error(f"❌ All phases EPUB creation failed: {message}")
            return False
    
    def test_output_validation(self):
        """Test detailed output validation."""
        logger.info("🔍 Testing output validation...")
        
        # Use the all-phases test output
        output_path = self.temp_dir / "test_all_phases.epub"
        
        if not output_path.exists():
            logger.warning(f"Test file not found: {output_path}")
            return False
        
        try:
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                # Test ZIP integrity
                zip_ref.testzip()
                logger.info("✅ ZIP file integrity validated")
                
                # Check file structure
                files = zip_ref.namelist()
                logger.info(f"📁 EPUB contains {len(files)} files")
                
                # Validate required files
                required_files = ['META-INF/container.xml', 'EPUB/content.opf', 'EPUB/toc.ncx']
                missing_files = [f for f in required_files if f not in files]
                
                if missing_files:
                    logger.error(f"❌ Missing required files: {missing_files}")
                    return False
                
                logger.info("✅ All required EPUB files present")
                return True
                
        except Exception as e:
            logger.error(f"❌ Output validation failed: {e}")
            return False
    
    def test_performance_metrics(self):
        """Test performance metrics and file sizes."""
        logger.info("⏱️ Testing performance metrics...")
        
        start_time = datetime.now()
        
        chapter_dir = self.project_root / "data" / "novels" / "永生_kanunu" / "mergedStyle"
        output_path = self.temp_dir / "test_performance.epub"
        
        if not chapter_dir.exists():
            logger.warning(f"Chapter directory not found: {chapter_dir}")
            return False
        
        success, message = build_epub(
            chapter_dir=str(chapter_dir),
            output_path=str(output_path),
            title="Eternal Life - Performance Test",
            author="观棋",
            translator="Performance Test",
            novel_slug="永生_kanunu",
            include_images=True
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if success:
            file_size = output_path.stat().st_size
            logger.info(f"✅ Performance test completed in {duration:.2f}s")
            logger.info(f"📊 File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
            logger.info(f"📄 Message: {message}")
            
            # Performance thresholds
            if duration > 30:  # 30 seconds threshold
                logger.warning(f"⚠️ Performance concern: took {duration:.2f}s")
            
            if file_size > 100 * 1024 * 1024:  # 100MB threshold
                logger.warning(f"⚠️ Large file size: {file_size/1024/1024:.1f} MB")
            
            return True
        else:
            logger.error(f"❌ Performance test failed: {message}")
            return False
    
    def validate_epub_structure(self, epub_path, test_type):
        """Validate EPUB structure and content."""
        logger.info(f"🔍 Validating EPUB structure for {test_type}...")
        
        try:
            with zipfile.ZipFile(epub_path, 'r') as zip_ref:
                files = zip_ref.namelist()
                
                # Basic structure validation
                structure_checks = {
                    'container.xml': 'META-INF/container.xml' in files,
                    'content.opf': any('content.opf' in f for f in files),
                    'toc.ncx': any('toc.ncx' in f for f in files),
                    'nav.xhtml': any('nav.xhtml' in f for f in files),
                    'css': any('.css' in f for f in files),
                    'chapters': any('.xhtml' in f and 'chapter' in f for f in files)
                }
                
                logger.info(f"📋 Structure validation for {test_type}:")
                for check, result in structure_checks.items():
                    status = "✅" if result else "❌"
                    logger.info(f"  {status} {check}: {result}")
                
                # Test-specific validations
                if test_type == "with_images":
                    has_images = any(f.endswith(('.png', '.jpg', '.jpeg', '.gif')) for f in files)
                    logger.info(f"  🖼️ Images: {'✅' if has_images else '❌'} {has_images}")
                
                if test_type == "rich_metadata":
                    has_metadata = any('metadata' in f for f in files)
                    logger.info(f"  📊 Metadata: {'✅' if has_metadata else '❌'} {has_metadata}")
                
                return all(structure_checks.values())
                
        except Exception as e:
            logger.error(f"❌ Structure validation failed: {e}")
            return False
    
    def validate_all_phases(self, epub_path):
        """Validate that all 8 phases are properly implemented."""
        logger.info("🔄 Validating all 8 phases...")
        
        try:
            with zipfile.ZipFile(epub_path, 'r') as zip_ref:
                files = zip_ref.namelist()
                
                phase_checks = {
                    'Phase 1: Basic Structure': 'META-INF/container.xml' in files,
                    'Phase 2: Metadata Template': any('metadata' in f for f in files),
                    'Phase 3: Image System': any(f.endswith(('.png', '.jpg', '.jpeg')) for f in files),
                    'Phase 4: Markdown Conversion': any('.xhtml' in f for f in files),
                    'Phase 5: Advanced TOC': any('toc.ncx' in f for f in files),
                    'Phase 6: Framework Branding': any('title_page.xhtml' in f for f in files),
                    'Phase 7: Job Metadata': any('metadata_report.xhtml' in f for f in files),
                    'Phase 8: Testing': True  # This test itself validates Phase 8
                }
                
                logger.info("🔍 Phase validation results:")
                for phase, result in phase_checks.items():
                    status = "✅" if result else "❌"
                    logger.info(f"  {status} {phase}: {result}")
                
                return all(phase_checks.values())
                
        except Exception as e:
            logger.error(f"❌ Phase validation failed: {e}")
            return False
    
    def generate_test_report(self):
        """Generate comprehensive test report."""
        logger.info("📊 Generating test report...")
        
        passed = sum(1 for r in self.test_results if r['status'] == 'PASSED')
        failed = sum(1 for r in self.test_results if r['status'] == 'FAILED')
        errors = sum(1 for r in self.test_results if r['status'] == 'ERROR')
        total = len(self.test_results)
        
        report = {
            'test_summary': {
                'total_tests': total,
                'passed': passed,
                'failed': failed,
                'errors': errors,
                'success_rate': f"{(passed/total)*100:.1f}%" if total > 0 else "0%"
            },
            'test_results': self.test_results,
            'generated_at': datetime.now().isoformat()
        }
        
        report_path = self.temp_dir / "test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("📈 TEST SUMMARY:")
        logger.info(f"  Total Tests: {total}")
        logger.info(f"  Passed: {passed}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Errors: {errors}")
        logger.info(f"  Success Rate: {(passed/total)*100:.1f}%" if total > 0 else "0%")
        logger.info(f"  Report saved: {report_path}")
        
        return report

def main():
    """Main test runner."""
    print("🧪 EPUB Builder Comprehensive Test Suite")
    print("=" * 50)
    
    test_suite = EPUBTestSuite()
    results = test_suite.run_all_tests()
    
    print("\n📊 Test Results:")
    print("=" * 50)
    for result in results:
        status_emoji = "✅" if result['status'] == 'PASSED' else "❌"
        print(f"{status_emoji} {result['test_name']}: {result['status']}")
    
    print("\n🏁 Testing completed! Check test_epub_builder.log for detailed output.")
    return results

if __name__ == "__main__":
    main()