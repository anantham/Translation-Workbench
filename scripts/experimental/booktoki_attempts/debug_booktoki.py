#!/usr/bin/env python3
"""
Debug script for BookToki scraper to understand page structure
"""

import os
import sys
import time
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_page_structure():
    """Debug the page structure to understand what's available"""
    
    test_url = "http://booktoki468.com/novel/3912078?stx=%EB%8D%98%EC%A0%84+%EB%94%94%ED%8E%9C%EC%8A%A4&book=%EC%99%84%EA%B2%B0%EC%86%8C%EC%84%A4"
    
    # Set up Chrome options
    chrome_options = Options()
    # chrome_options.add_argument('--headless')  # Commented out so you can see the browser
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = None
    try:
        logger.info("🚀 Starting Chrome WebDriver...")
        driver = webdriver.Chrome(options=chrome_options)
        
        # Execute script to remove webdriver property
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        
        logger.info(f"🌐 Loading page: {test_url}")
        driver.get(test_url)
        
        # Wait for page to load
        time.sleep(5)
        
        # Check page title
        title = driver.title
        logger.info(f"📄 Page title: {title}")
        
        # Check if we got redirected or blocked
        current_url = driver.current_url
        logger.info(f"🔗 Current URL: {current_url}")
        
        # Get page source and analyze
        page_source = driver.page_source
        logger.info(f"📊 Page source length: {len(page_source)} characters")
        
        # Parse with BeautifulSoup for analysis
        soup = BeautifulSoup(page_source, 'html.parser')
        
        # Check for common error indicators
        if "403" in page_source or "Forbidden" in page_source:
            logger.error("❌ Page shows 403 Forbidden error")
        elif "404" in page_source or "Not Found" in page_source:
            logger.error("❌ Page shows 404 Not Found error")
        elif len(page_source) < 1000:
            logger.warning("⚠️  Page source is very short, might be blocked")
        else:
            logger.info("✅ Page loaded successfully")
        
        # Look for content containers
        logger.info("🔍 Looking for content containers...")
        
        # Check for the main content container
        novel_content = soup.find(id='novel_content')
        if novel_content:
            logger.info("✅ Found #novel_content container")
            content_text = novel_content.get_text(strip=True)
            logger.info(f"📝 Content length: {len(content_text)} characters")
            if content_text:
                logger.info(f"📄 Content preview: {content_text[:200]}...")
            else:
                logger.warning("⚠️  #novel_content is empty")
        else:
            logger.warning("❌ #novel_content not found")
        
        # Check for alternative content selectors
        alternative_selectors = [
            '.content', '.chapter-content', '.novel-content', 
            '.story-content', '.text-content', '#content',
            '.view-padding', 'article', '.entry-content'
        ]
        
        for selector in alternative_selectors:
            elements = soup.select(selector)
            if elements:
                logger.info(f"✅ Found {len(elements)} elements matching '{selector}'")
                for i, elem in enumerate(elements[:2]):  # Check first 2
                    text = elem.get_text(strip=True)
                    logger.info(f"  Element {i+1}: {len(text)} chars - {text[:100]}...")
        
        # Look for navigation links
        logger.info("🧭 Looking for navigation links...")
        nav_links = soup.select('a[href*="/novel/"]')
        logger.info(f"📎 Found {len(nav_links)} novel links")
        
        for i, link in enumerate(nav_links[:5]):  # Show first 5
            href = link.get('href', '')
            text = link.get_text(strip=True)
            logger.info(f"  Link {i+1}: '{text}' -> {href}")
        
        # Check for Korean text
        korean_text = soup.get_text()
        import re
        korean_chars = re.findall(r'[가-힣]', korean_text)
        logger.info(f"🇰🇷 Korean characters found: {len(korean_chars)}")
        
        # Save page source for manual inspection
        output_file = "debug_booktoki_page.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(page_source)
        logger.info(f"💾 Page source saved to: {output_file}")
        
        # Take a screenshot
        try:
            driver.save_screenshot("debug_booktoki_screenshot.png")
            logger.info("📸 Screenshot saved to: debug_booktoki_screenshot.png")
        except Exception as e:
            logger.warning(f"Screenshot failed: {e}")
        
        logger.info("🎉 Debug analysis complete!")
        
    except Exception as e:
        logger.error(f"❌ Debug failed: {e}")
        
    finally:
        if driver:
            driver.quit()
            logger.info("🧹 WebDriver cleaned up")

if __name__ == "__main__":
    debug_page_structure()