"""
Pytest configuration and fixtures for alignment map builder tests.
"""

import pytest
import tempfile
from pathlib import Path

@pytest.fixture
def temp_test_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent

@pytest.fixture
def sample_chinese_chapters(temp_test_dir):
    """Create sample Chinese chapter files for testing."""
    chinese_dir = temp_test_dir / "chinese_chapters"
    chinese_dir.mkdir()
    
    # Create sample Chinese files
    chapters = [
        ("Chapter-0001-测试章节一.txt", "这是第一章的内容。测试中文编码。"),
        ("Chapter-0002-测试章节二.txt", "这是第二章的内容。更多中文测试。"),
        ("Chapter-0003-测试章节三.txt", "这是第三章的内容。继续测试。"),
    ]
    
    for filename, content in chapters:
        file_path = chinese_dir / filename
        file_path.write_text(content, encoding='utf-8')
    
    return chinese_dir

@pytest.fixture
def sample_english_chapters(temp_test_dir):
    """Create sample English chapter files for testing."""
    english_dir = temp_test_dir / "english_chapters"
    english_dir.mkdir()
    
    # Create sample English files
    chapters = [
        ("English-Chapter-0001.txt", "This is the content of chapter one."),
        ("English-Chapter-0002.txt", "This is the content of chapter two."),
        ("English-Chapter-0004.txt", "This is the content of chapter four."),  # Gap at chapter 3
    ]
    
    for filename, content in chapters:
        file_path = english_dir / filename
        file_path.write_text(content, encoding='utf-8')
    
    return english_dir

@pytest.fixture
def problematic_files(temp_test_dir):
    """Create problematic files for edge case testing."""
    problem_dir = temp_test_dir / "problematic_files"
    problem_dir.mkdir()
    
    # Create various problematic files
    files = [
        ("empty_file.txt", ""),
        ("html_file.txt", "<!DOCTYPE html><html><body>Not a chapter</body></html>"),
        ("binary_file.txt", b"\x00\x01\x02\x03"),
        ("very_short.txt", "Hi"),
        ("Chapter-0001-normal.txt", "This is a normal chapter for comparison."),
    ]
    
    for filename, content in files:
        file_path = problem_dir / filename
        if isinstance(content, bytes):
            file_path.write_bytes(content)
        else:
            file_path.write_text(content, encoding='utf-8')
    
    return problem_dir

@pytest.fixture
def real_data_paths(project_root):
    """Get paths to real project data for integration tests."""
    return {
        'chinese_dir': project_root / "data/novels/way_of_the_devil/raw_chapters",
        'english_dir': project_root / "data/novels/way_of_the_devil/official_english",
        'eternal_chinese_dir': project_root / "data/novels/eternal_novelcool/raw_chapters",
    }

# Pytest configuration
def pytest_configure(config):
    """Configure pytest."""
    # Add custom markers
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests that take a long time")
    config.addinivalue_line("markers", "real_data: Tests that require real project data")

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    # Add markers based on test names
    for item in items:
        if "integration" in item.name:
            item.add_marker(pytest.mark.integration)
        elif "unit" in item.name:
            item.add_marker(pytest.mark.unit)
        
        if "real_data" in item.name or "test_with_real" in item.name:
            item.add_marker(pytest.mark.real_data)
        
        if "large" in item.name or "performance" in item.name:
            item.add_marker(pytest.mark.slow)