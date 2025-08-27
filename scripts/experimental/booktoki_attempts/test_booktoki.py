#!/usr/bin/env python3
"""
Test script for BookToki adapter
"""
import os
import sys
import logging
import requests
from bs4 import BeautifulSoup

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up basic logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import our adapter
try:
    from utils.booktoki_adapter import BookTokiAdapter
    logger.info("✅ Successfully imported BookTokiAdapter")
except ImportError as e:
    logger.error(f"❌ Failed to import BookTokiAdapter: {e}")
    sys.exit(1)

def test_booktoki_adapter():
    """Test the BookToki adapter with the provided URL"""
    
    test_url = "http://booktoki468.com/novel/3912078?stx=%EB%8D%98%EC%A0%84+%EB%94%94%ED%8E%9C%EC%8A%A4&book=%EC%99%84%EA%B2%B0%EC%86%8C%EC%84%A4"
    
    logger.info(f"🚀 Testing BookToki adapter with URL: {test_url}")
    
    # Create adapter instance
    adapter = BookTokiAdapter(test_url)
    
    # Test basic functionality
    logger.info("📋 Testing basic adapter methods...")
    
    # Test encoding
    encoding = adapter.get_encoding()
    logger.info(f"📝 Encoding: {encoding}")
    
    # Fetch the page
    logger.info("🌐 Fetching page content...")
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(test_url, headers=headers)
        response.raise_for_status()
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser', from_encoding=encoding)
        logger.info(f"📄 Page fetched successfully, length: {len(response.content)} bytes")
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch page: {e}")
        return False
    
    # Test title extraction
    logger.info("📖 Testing title extraction...")
    title = adapter.extract_title(soup)
    if title:
        logger.info(f"✅ Title extracted: {title}")
    else:
        logger.warning("⚠️  No title found")
    
    # Test chapter info parsing
    if title:
        logger.info("🔍 Testing chapter info parsing...")
        start_num, end_num, filename_num = adapter.parse_chapter_info(title, soup)
        if start_num:
            logger.info(f"✅ Chapter info: start={start_num}, end={end_num}, filename={filename_num}")
        else:
            logger.warning("⚠️  Could not parse chapter info")
    
    # Test content extraction  
    logger.info("📝 Testing content extraction...")
    content = adapter.extract_content(soup)
    if content:
        logger.info(f"✅ Content extracted: {len(content)} characters")
        logger.info(f"📄 Content preview: {content[:200]}...")
        
        # Check for Korean text
        import re
        korean_chars = re.findall(r'[가-힣]', content)
        logger.info(f"🇰🇷 Korean characters found: {len(korean_chars)}")
        
    else:
        logger.error("❌ No content extracted")
        return False
    
    # Test navigation
    logger.info("🧭 Testing navigation links...")
    next_link = adapter.get_next_link(soup, "Forwards (oldest to newest)")
    prev_link = adapter.get_next_link(soup, "Backwards (newest to oldest)")
    
    if next_link:
        logger.info(f"➡️  Next link: {next_link}")
    else:
        logger.warning("⚠️  No next link found")
        
    if prev_link:
        logger.info(f"⬅️  Prev link: {prev_link}")
    else:
        logger.warning("⚠️  No prev link found")
    
    logger.info("🎉 BookToki adapter test completed!")
    return True

if __name__ == "__main__":
    success = test_booktoki_adapter()
    if success:
        logger.info("✅ All tests passed!")
        sys.exit(0)
    else:
        logger.error("❌ Some tests failed!")
        sys.exit(1)