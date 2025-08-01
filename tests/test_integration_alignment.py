"""
Integration Tests for Alignment Map Builder

Tests real-world scenarios with actual project data.
Focus on UI integration and end-to-end workflows.
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
    build_and_save_alignment_map,
    preview_alignment_mapping,
    get_alignment_map_path
)

# Configure integration test logging
def setup_integration_logging():
    """Setup logging for integration tests."""
    log_file = os.path.join(os.path.dirname(__file__), 'integration_test_results.log')
    
    logger = logging.getLogger('integration_alignment_tests')
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)8s | %(funcName)20s | %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class IntegrationTestRunner:
    """Integration test runner for alignment map builder."""
    
    def __init__(self):
        self.logger = setup_integration_logging()
        self.project_root = Path(__file__).parent.parent
        self.test_results = {
            'scenarios': [],
            'total_scenarios': 0,
            'passed_scenarios': 0,
            'failed_scenarios': 0
        }
        
        # Real data paths
        self.data_scenarios = [
            {
                'name': 'Way of the Devil (Complete)',
                'chinese_dir': self.project_root / "data/novels/way_of_the_devil/raw_chapters",
                'english_dir': self.project_root / "data/novels/way_of_the_devil/official_english",
                'expected_overlap': 'high',  # Should have good overlap
                'description': 'Main novel with both Chinese and English chapters'
            },
            {
                'name': 'Eternal Life (Chinese Only)',
                'chinese_dir': self.project_root / "data/novels/eternal_novelcool/raw_chapters",
                'english_dir': self.project_root / "data/novels/eternal_novelcool/english_chapters",  # May not exist
                'expected_overlap': 'low',  # Likely no English chapters
                'description': 'Chinese-only novel for testing partial alignment'
            },
            {
                'name': 'Legacy Structure',
                'chinese_dir': self.project_root / "novel_content_dxmwx_complete",  # Legacy location
                'english_dir': self.project_root / "english_chapters",  # Legacy location
                'expected_overlap': 'medium',  # May have some overlap
                'description': 'Legacy directory structure compatibility'
            }
        ]
        
        self.logger.info("=" * 80)
        self.logger.info("INTEGRATION TESTS FOR ALIGNMENT MAP BUILDER")
        self.logger.info("=" * 80)
    
    def run_scenario(self, scenario):
        """Run a single integration scenario."""
        self.test_results['total_scenarios'] += 1
        scenario_name = scenario['name']
        
        self.logger.info(f"🔍 SCENARIO: {scenario_name}")
        self.logger.info(f"📝 Description: {scenario['description']}")
        self.logger.info(f"🇨🇳 Chinese dir: {scenario['chinese_dir']}")
        self.logger.info(f"🇺🇸 English dir: {scenario['english_dir']}")
        
        scenario_result = {
            'name': scenario_name,
            'status': 'FAILED',
            'details': [],
            'chinese_exists': False,
            'english_exists': False,
            'chinese_count': 0,
            'english_count': 0,
            'overlap_count': 0,
            'preview_success': False,
            'build_success': False,
            'file_created': False
        }
        
        try:
            # Check if directories exist
            scenario_result['chinese_exists'] = scenario['chinese_dir'].exists()
            scenario_result['english_exists'] = scenario['english_dir'].exists()
            
            self.logger.info(f"Chinese dir exists: {scenario_result['chinese_exists']}")
            self.logger.info(f"English dir exists: {scenario_result['english_exists']}")
            
            if not scenario_result['chinese_exists'] and not scenario_result['english_exists']:
                scenario_result['details'].append("Both directories missing - skipping scenario")
                self.logger.warning("Both directories missing - skipping scenario")
                self.test_results['scenarios'].append(scenario_result)
                return scenario_result
            
            # Test preview functionality
            self.logger.info("🔍 Testing preview functionality...")
            try:
                preview_result = preview_alignment_mapping(
                    str(scenario['chinese_dir']),
                    str(scenario['english_dir'])
                )
                
                if preview_result['success']:
                    scenario_result['preview_success'] = True
                    stats = preview_result['stats']
                    
                    scenario_result['chinese_count'] = stats['chinese_total']
                    scenario_result['english_count'] = stats['english_total']
                    scenario_result['overlap_count'] = stats['both_files']
                    
                    self.logger.info(f"Preview stats: {stats}")
                    
                    # Check if overlap matches expectations
                    overlap_percentage = (stats['both_files'] / max(stats['chinese_total'], stats['english_total'], 1)) * 100
                    self.logger.info(f"Overlap percentage: {overlap_percentage:.1f}%")
                    
                    if preview_result['warnings']:
                        self.logger.warning(f"Preview warnings: {preview_result['warnings']}")
                    
                    if preview_result['file_issues']:
                        self.logger.warning(f"File issues found: {len(preview_result['file_issues'])}")
                        for issue in preview_result['file_issues'][:3]:  # Log first 3 issues
                            self.logger.warning(f"  {issue['type']} chapter {issue['chapter']}: {issue['errors']}")
                    
                else:
                    scenario_result['details'].append(f"Preview failed: {preview_result.get('errors', [])}")
                    self.logger.error(f"Preview failed: {preview_result.get('errors', [])}")
                    
            except Exception as e:
                scenario_result['details'].append(f"Preview error: {str(e)}")
                self.logger.error(f"Preview error: {str(e)}")
            
            # Test build functionality
            self.logger.info("🔨 Testing build functionality...")
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    novel_name = scenario_name.lower().replace(" ", "_")
                    output_path = os.path.join(temp_dir, f"{novel_name}_alignment_map.json")
                    
                    success, message, build_stats = build_and_save_alignment_map(
                        str(scenario['chinese_dir']),
                        str(scenario['english_dir']),
                        novel_name,
                        output_path
                    )
                    
                    if success:
                        scenario_result['build_success'] = True
                        scenario_result['file_created'] = os.path.exists(output_path)
                        
                        self.logger.info(f"Build success: {message}")
                        self.logger.info(f"Build stats: {build_stats}")
                        
                        # Verify file content
                        if scenario_result['file_created']:
                            with open(output_path, 'r', encoding='utf-8') as f:
                                alignment_map = json.load(f)
                            
                            self.logger.info(f"Saved alignment map has {len(alignment_map)} entries")
                            
                            # Log sample mappings
                            sample_mappings = list(alignment_map.items())[:3]
                            for ch_key, mapping in sample_mappings:
                                self.logger.debug(f"  Chapter {ch_key}: {mapping}")
                    
                    else:
                        scenario_result['details'].append(f"Build failed: {message}")
                        self.logger.error(f"Build failed: {message}")
                        
            except Exception as e:
                scenario_result['details'].append(f"Build error: {str(e)}")
                self.logger.error(f"Build error: {str(e)}")
            
            # Test central storage path generation
            self.logger.info("📁 Testing central storage path generation...")
            try:
                central_path = get_alignment_map_path(scenario_name)
                self.logger.info(f"Central storage path: {central_path}")
                
                # Verify directory creation
                if os.path.exists(os.path.dirname(central_path)):
                    self.logger.info("✅ Central storage directory created")
                else:
                    self.logger.error("❌ Central storage directory not created")
                    scenario_result['details'].append("Central storage directory not created")
                    
            except Exception as e:
                scenario_result['details'].append(f"Central storage error: {str(e)}")
                self.logger.error(f"Central storage error: {str(e)}")
            
            # Determine overall scenario status
            if scenario_result['preview_success'] and scenario_result['build_success']:
                scenario_result['status'] = 'PASSED'
                self.test_results['passed_scenarios'] += 1
                self.logger.info(f"✅ SCENARIO PASSED: {scenario_name}")
            else:
                self.test_results['failed_scenarios'] += 1
                self.logger.error(f"❌ SCENARIO FAILED: {scenario_name}")
                
        except Exception as e:
            scenario_result['details'].append(f"Scenario error: {str(e)}")
            self.logger.error(f"💥 SCENARIO ERROR: {scenario_name} - {str(e)}")
            self.test_results['failed_scenarios'] += 1
        
        self.test_results['scenarios'].append(scenario_result)
        self.logger.info("-" * 60)
        return scenario_result
    
    def test_ui_integration_simulation(self):
        """Simulate UI integration scenarios."""
        self.logger.info("🖥️  TESTING UI INTEGRATION SIMULATION")
        
        # This simulates what happens when user interacts with the Data Review page
        ui_scenarios = [
            {
                'name': 'User selects existing directories',
                'chinese_dir': self.project_root / "data/novels/way_of_the_devil/raw_chapters",
                'english_dir': self.project_root / "data/novels/way_of_the_devil/official_english",
                'novel_name': 'way_of_the_devil'
            },
            {
                'name': 'User selects non-existent directory',
                'chinese_dir': self.project_root / "data/novels/nonexistent/raw_chapters",
                'english_dir': self.project_root / "data/novels/nonexistent/english_chapters",
                'novel_name': 'nonexistent_novel'
            },
            {
                'name': 'User selects mixed scenario',
                'chinese_dir': self.project_root / "data/novels/eternal_novelcool/raw_chapters",
                'english_dir': self.project_root / "data/novels/nonexistent/english_chapters",
                'novel_name': 'mixed_scenario'
            }
        ]
        
        for scenario in ui_scenarios:
            self.logger.info(f"🔍 UI SCENARIO: {scenario['name']}")
            
            # Simulate preview button click
            self.logger.info("📋 Simulating preview button click...")
            try:
                preview_result = preview_alignment_mapping(
                    str(scenario['chinese_dir']),
                    str(scenario['english_dir'])
                )
                
                if preview_result['success']:
                    self.logger.info("✅ Preview would show success to user")
                    self.logger.info(f"Preview stats: {preview_result['stats']}")
                else:
                    self.logger.info("❌ Preview would show errors to user")
                    self.logger.info(f"Preview errors: {preview_result.get('errors', [])}")
                    
            except Exception as e:
                self.logger.error(f"Preview simulation failed: {e}")
            
            # Simulate build button click (only if preview succeeded)
            if 'preview_result' in locals() and preview_result['success']:
                self.logger.info("🔨 Simulating build button click...")
                try:
                    # Use central storage like the real UI would
                    success, message, build_stats = build_and_save_alignment_map(
                        str(scenario['chinese_dir']),
                        str(scenario['english_dir']),
                        scenario['novel_name']
                    )
                    
                    if success:
                        self.logger.info("✅ Build would show success to user")
                        self.logger.info(f"Success message: {message}")
                    else:
                        self.logger.info("❌ Build would show failure to user")
                        self.logger.info(f"Failure message: {message}")
                        
                except Exception as e:
                    self.logger.error(f"Build simulation failed: {e}")
            
            self.logger.info("-" * 40)
    
    def run_all_integration_tests(self):
        """Run all integration tests."""
        self.logger.info("🚀 Starting integration test suite...")
        
        # Run data scenarios
        for scenario in self.data_scenarios:
            self.run_scenario(scenario)
        
        # Run UI integration simulation
        self.test_ui_integration_simulation()
        
        # Generate summary
        self.generate_integration_summary()
        
        return self.test_results
    
    def generate_integration_summary(self):
        """Generate integration test summary."""
        self.logger.info("=" * 80)
        self.logger.info("INTEGRATION TEST SUMMARY")
        self.logger.info("=" * 80)
        
        results = self.test_results
        
        self.logger.info(f"Total Scenarios: {results['total_scenarios']}")
        self.logger.info(f"Passed: {results['passed_scenarios']}")
        self.logger.info(f"Failed: {results['failed_scenarios']}")
        
        if results['total_scenarios'] > 0:
            success_rate = (results['passed_scenarios'] / results['total_scenarios']) * 100
            self.logger.info(f"Success Rate: {success_rate:.1f}%")
        
        # Detailed scenario results
        self.logger.info("\nSCENARIO DETAILS:")
        for scenario in results['scenarios']:
            self.logger.info(f"  {scenario['name']}: {scenario['status']}")
            self.logger.info(f"    Chinese files: {scenario['chinese_count']}")
            self.logger.info(f"    English files: {scenario['english_count']}")
            self.logger.info(f"    Overlap: {scenario['overlap_count']}")
            
            if scenario['details']:
                self.logger.info(f"    Issues: {scenario['details']}")
        
        self.logger.info("\n📊 PERFORMANCE INSIGHTS:")
        
        # Calculate total files processed
        total_chinese = sum(s['chinese_count'] for s in results['scenarios'])
        total_english = sum(s['english_count'] for s in results['scenarios'])
        total_overlap = sum(s['overlap_count'] for s in results['scenarios'])
        
        self.logger.info(f"Total Chinese files processed: {total_chinese}")
        self.logger.info(f"Total English files processed: {total_english}")
        self.logger.info(f"Total overlapping chapters: {total_overlap}")
        
        if total_chinese > 0 or total_english > 0:
            overall_overlap = (total_overlap / max(total_chinese, total_english)) * 100
            self.logger.info(f"Overall overlap rate: {overall_overlap:.1f}%")
        
        # Recommendations
        self.logger.info("\n💡 RECOMMENDATIONS:")
        
        failed_scenarios = [s for s in results['scenarios'] if s['status'] == 'FAILED']
        if failed_scenarios:
            self.logger.info("   - Address failed scenarios for complete coverage")
            for scenario in failed_scenarios:
                self.logger.info(f"     * {scenario['name']}: {scenario['details']}")
        
        if total_overlap < (total_chinese + total_english) / 2:
            self.logger.info("   - Consider improving chapter alignment detection")
        
        if any(s['chinese_count'] == 0 or s['english_count'] == 0 for s in results['scenarios']):
            self.logger.info("   - Add more robust handling for single-language scenarios")
        
        self.logger.info("\nDetailed results saved to integration_test_results.log")
        self.logger.info("=" * 80)

def main():
    """Main integration test runner."""
    print("🔗 Running Integration Tests for Alignment Map Builder...")
    print("📝 Detailed results will be logged to integration_test_results.log")
    print("🔍 Testing real-world scenarios with actual project data...")
    
    test_runner = IntegrationTestRunner()
    results = test_runner.run_all_integration_tests()
    
    print(f"\n✅ Integration tests completed: {results['passed_scenarios']}/{results['total_scenarios']} scenarios passed")
    
    if results['failed_scenarios'] > 0:
        print(f"❌ {results['failed_scenarios']} scenarios failed - check integration_test_results.log for details")
        return 1
    else:
        print("🎉 All integration tests passed!")
        return 0

if __name__ == "__main__":
    exit(main())