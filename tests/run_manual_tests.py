"""
Manual Test Runner for Alignment Map Builder

This script runs comprehensive tests and generates detailed logs for manual review.
Designed to be run by the user to validate the alignment map builder implementation.
"""

import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path

def setup_test_environment():
    """Setup test environment and logging."""
    test_dir = Path(__file__).parent
    project_root = test_dir.parent
    
    # Create logs directory if it doesn't exist
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Create test results directory
    test_results_dir = test_dir / "results"
    test_results_dir.mkdir(exist_ok=True)
    
    return test_dir, project_root, logs_dir, test_results_dir

def run_test_suite(test_script, test_name, test_dir, logs_dir):
    """Run a test suite and capture results."""
    print(f"🔍 Running {test_name}...")
    
    # Generate timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"manual_test_{test_name.lower().replace(' ', '_')}_{timestamp}.log"
    
    try:
        # Change to test directory
        os.chdir(test_dir)
        
        # Run test script
        result = subprocess.run([
            sys.executable, test_script
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        # Write results to log file
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Manual Test Run: {test_name}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Script: {test_script}\n")
            f.write(f"Return Code: {result.returncode}\n")
            f.write("=" * 80 + "\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n" + "=" * 80 + "\n")
            f.write("STDERR:\n")
            f.write(result.stderr)
            f.write("\n" + "=" * 80 + "\n")
        
        # Print summary
        if result.returncode == 0:
            print(f"✅ {test_name} completed successfully")
            print(f"📝 Detailed log: {log_file}")
        else:
            print(f"❌ {test_name} failed (return code: {result.returncode})")
            print(f"📝 Error log: {log_file}")
            if result.stderr:
                print(f"💥 Error summary: {result.stderr[:200]}...")
        
        return result.returncode == 0, log_file
        
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_name} timed out after 5 minutes")
        return False, log_file
    except Exception as e:
        print(f"💥 Error running {test_name}: {e}")
        return False, log_file

def check_project_structure(project_root):
    """Check if project structure is ready for testing."""
    print("🔍 Checking project structure...")
    
    required_paths = [
        "utils/alignment_map_builder.py",
        "utils/logging.py",
        "data/novels/way_of_the_devil/official_english",
        "data/novels/way_of_the_devil/raw_chapters",
    ]
    
    missing_paths = []
    for path in required_paths:
        full_path = project_root / path
        if not full_path.exists():
            missing_paths.append(path)
    
    if missing_paths:
        print("❌ Missing required project components:")
        for path in missing_paths:
            print(f"   - {path}")
        return False
    
    print("✅ Project structure looks good")
    return True

def analyze_test_results(test_results_dir):
    """Analyze test results and generate summary."""
    print("\n📊 Analyzing test results...")
    
    # Look for test log files
    log_files = list(test_results_dir.glob("*.log"))
    
    if not log_files:
        print("⚠️  No test result files found")
        return
    
    summary = {
        'total_tests': 0,
        'passed_tests': 0,
        'failed_tests': 0,
        'test_details': []
    }
    
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Simple parsing to extract test results
            if "TEST SUMMARY" in content:
                lines = content.split('\n')
                for line in lines:
                    if "Total Tests:" in line:
                        summary['total_tests'] += int(line.split(':')[1].strip())
                    elif "Passed:" in line and "Failed:" not in line:
                        summary['passed_tests'] += int(line.split(':')[1].strip())
                    elif "Failed:" in line:
                        summary['failed_tests'] += int(line.split(':')[1].strip())
                
        except Exception as e:
            print(f"⚠️  Error reading {log_file}: {e}")
    
    # Generate summary report
    if summary['total_tests'] > 0:
        success_rate = (summary['passed_tests'] / summary['total_tests']) * 100
        print(f"📈 Overall Success Rate: {success_rate:.1f}%")
        print(f"✅ Passed: {summary['passed_tests']}")
        print(f"❌ Failed: {summary['failed_tests']}")
        print(f"📊 Total: {summary['total_tests']}")
    
    return summary

def generate_manual_test_report(test_results, logs_dir):
    """Generate a comprehensive manual test report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = logs_dir / f"manual_test_report_{timestamp}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Manual Test Report - Alignment Map Builder\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        f.write("## Test Overview\n\n")
        
        # Test results summary
        total_suites = len(test_results)
        passed_suites = sum(1 for success, _ in test_results.values() if success)
        
        f.write(f"- **Total Test Suites:** {total_suites}\n")
        f.write(f"- **Passed Suites:** {passed_suites}\n")
        f.write(f"- **Failed Suites:** {total_suites - passed_suites}\n\n")
        
        # Individual test results
        f.write("## Test Suite Results\n\n")
        
        for test_name, (success, log_file) in test_results.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            f.write(f"### {test_name}\n")
            f.write(f"**Status:** {status}\n")
            f.write(f"**Log File:** `{log_file.name}`\n\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        if passed_suites == total_suites:
            f.write("🎉 **All tests passed!** The alignment map builder implementation is working correctly.\n\n")
        else:
            f.write("⚠️ **Some tests failed.** Please review the log files for specific issues:\n\n")
            
            failed_tests = [name for name, (success, _) in test_results.items() if not success]
            for test_name in failed_tests:
                f.write(f"- Review `{test_results[test_name][1].name}` for {test_name} failures\n")
        
        f.write("\n## Next Steps\n\n")
        f.write("1. Review detailed log files for any failures\n")
        f.write("2. Run integration tests with real data\n")
        f.write("3. Test UI integration in Data Review page\n")
        f.write("4. Validate backup and recovery functionality\n")
        f.write("5. Performance testing with large datasets\n")
    
    return report_file

def main():
    """Main manual test runner."""
    print("🧪 Manual Test Runner for Alignment Map Builder")
    print("=" * 60)
    
    # Setup environment
    test_dir, project_root, logs_dir, test_results_dir = setup_test_environment()
    
    # Check project structure
    if not check_project_structure(project_root):
        print("❌ Project structure check failed. Please fix missing components.")
        return 1
    
    # Define test suites to run
    test_suites = [
        ("test_alignment_map_builder.py", "Unit Tests"),
        ("test_integration_alignment.py", "Integration Tests"),
    ]
    
    # Run all test suites
    test_results = {}
    
    for test_script, test_name in test_suites:
        test_file = test_dir / test_script
        
        if not test_file.exists():
            print(f"⚠️  Test file not found: {test_file}")
            test_results[test_name] = (False, None)
            continue
        
        success, log_file = run_test_suite(test_script, test_name, test_dir, logs_dir)
        test_results[test_name] = (success, log_file)
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("📊 MANUAL TEST SUMMARY")
    print("=" * 60)
    
    total_suites = len(test_results)
    passed_suites = sum(1 for success, _ in test_results.values() if success)
    
    print(f"Total Test Suites: {total_suites}")
    print(f"Passed: {passed_suites}")
    print(f"Failed: {total_suites - passed_suites}")
    
    # Generate detailed report
    report_file = generate_manual_test_report(test_results, logs_dir)
    print(f"\n📄 Detailed report generated: {report_file}")
    
    # List all log files for review
    print("\n📝 Log files for review:")
    for test_name, (success, log_file) in test_results.items():
        if log_file:
            status = "✅" if success else "❌"
            print(f"   {status} {test_name}: {log_file}")
    
    # Additional log files in test directory
    test_logs = list(test_dir.glob("*.log"))
    if test_logs:
        print("\n📋 Additional test logs:")
        for log_file in test_logs:
            print(f"   📝 {log_file}")
    
    print("\n" + "=" * 60)
    
    if passed_suites == total_suites:
        print("🎉 All manual tests completed successfully!")
        print("✅ The alignment map builder implementation is ready for production use.")
        return 0
    else:
        print("❌ Some tests failed. Please review the log files for details.")
        print("🔧 Fix the issues and re-run the tests.")
        return 1

if __name__ == "__main__":
    exit(main())