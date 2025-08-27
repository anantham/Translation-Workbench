#!/usr/bin/env python3
"""
Selenium-based BookToki Scraper

Uses Selenium WebDriver to bypass anti-bot protection by acting like a real browser.
Works with Chrome/Brave and handles dynamic content loading.
"""

import os
import sys
import time
import json
import re
import logging
from urllib.parse import urljoin, urlparse, parse_qs
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from selenium.webdriver.common.action_chains import ActionChains

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SeleniumBookTokiScraper:
    def __init__(self, headless=True, delay=2):
        self.delay = delay
        self.driver = None
        self.setup_driver(headless)
        
    def setup_driver(self, headless=True):
        """Set up Chrome WebDriver with realistic browser behavior"""
        try:
            chrome_options = Options()
            
            if headless:
                chrome_options.add_argument('--headless')
            
            # Essential options for bypassing detection
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Browser behavior options
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--disable-web-security')
            chrome_options.add_argument('--allow-running-insecure-content')
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--disable-plugins')
            chrome_options.add_argument('--disable-images')  # Faster loading
            
            # Set realistic user agent
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            
            # Language preferences (Korean support)
            chrome_options.add_argument('--lang=ko-KR')
            chrome_options.add_experimental_option('prefs', {
                'intl.accept_languages': 'ko-KR,ko,en-US,en'
            })
            
            self.driver = webdriver.Chrome(options=chrome_options)
            
            # Execute script to remove webdriver property
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            # Set realistic timeouts
            self.driver.implicitly_wait(10)
            self.driver.set_page_load_timeout(30)
            
            logger.info("✅ Chrome WebDriver initialized successfully")
            
        except WebDriverException as e:
            logger.error(f"❌ Failed to initialize Chrome WebDriver: {e}")
            logger.info("💡 Please install ChromeDriver: brew install chromedriver")
            logger.info("💡 Or download from: https://chromedriver.chromium.org/")
            raise
    
    def _extract_chapter_id_from_url(self, url):
        """Extract chapter ID from URL"""
        match = re.search(r'/novel/(\d+)', url)
        return int(match.group(1)) if match else None
    
    def _extract_chapter_title(self):
        """Extract chapter title from current page"""
        try:
            # Try page title first
            title = self.driver.title
            if title:
                chapter_match = re.search(r'([^_]+?-\d+화)', title)
                if chapter_match:
                    return chapter_match.group(1)
            
            # Try other selectors
            for selector in ['h1', 'h2', '.title']:
                try:
                    element = self.driver.find_element(By.CSS_SELECTOR, selector)
                    if element and element.text.strip():
                        return element.text.strip()
                except NoSuchElementException:
                    continue
            
            return None
        except Exception as e:
            logger.warning(f"Failed to extract title: {e}")
            return None
    
    def _extract_content(self):
        """Extract chapter content from current page"""
        try:
            logger.debug("🔍 Looking for content container...")
            
            # Wait for content to load
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.ID, "novel_content"))
                )
                logger.debug("✅ Found novel_content container")
            except TimeoutException:
                logger.warning("⚠️  Timeout waiting for novel_content - trying alternative selectors")
            
            # Try primary selector
            content_container = None
            try:
                content_container = self.driver.find_element(By.ID, "novel_content")
            except NoSuchElementException:
                logger.warning("❌ novel_content not found, trying alternatives")
            
            if not content_container:
                # Try alternative selectors
                alternative_selectors = [
                    '.content', '.chapter-content', '.novel-content', 
                    '.story-content', '.text-content', '#content',
                    '.view-padding', 'article .content', '.entry-content'
                ]
                
                for selector in alternative_selectors:
                    try:
                        content_container = self.driver.find_element(By.CSS_SELECTOR, selector)
                        logger.info(f"✅ Found content using alternative selector: {selector}")
                        break
                    except NoSuchElementException:
                        continue
            
            if not content_container:
                logger.error("❌ No content container found")
                return None
            
            # Look for content within f9e99a33513 div
            try:
                content_div = content_container.find_element(By.CSS_SELECTOR, 'div.f9e99a33513')
                logger.debug("✅ Found f9e99a33513 content div")
            except NoSuchElementException:
                content_div = content_container
                logger.debug("Using content_container directly (no f9e99a33513 div)")
            
            # Extract paragraphs
            try:
                paragraphs = content_div.find_elements(By.TAG_NAME, "p")
                logger.debug(f"📝 Found {len(paragraphs)} paragraphs")
            except NoSuchElementException:
                logger.warning("No paragraphs found, extracting raw text")
                return self._extract_raw_text(content_div)
            
            if not paragraphs:
                return self._extract_raw_text(content_div)
            
            # Filter and clean paragraphs
            chapter_text = []
            for i, p in enumerate(paragraphs):
                try:
                    text = p.text.strip()
                    if text and self._is_valid_paragraph(text):
                        chapter_text.append(text)
                        if i < 3:  # Log first few paragraphs
                            logger.debug(f"✅ Paragraph {i+1}: {text[:50]}...")
                except Exception as e:
                    logger.warning(f"Error extracting paragraph {i}: {e}")
                    continue
            
            if not chapter_text:
                logger.warning("No valid paragraphs found, trying raw text extraction")
                return self._extract_raw_text(content_div)
            
            content = '\n\n'.join(chapter_text)
            
            # Validate content
            if not self._validate_content(content):
                return None
            
            logger.info(f"✅ Content extracted: {len(content)} chars, {len(chapter_text)} paragraphs")
            return content
            
        except Exception as e:
            logger.error(f"❌ Content extraction failed: {e}")
            return None
    
    def _extract_raw_text(self, element):
        """Extract raw text when paragraph extraction fails"""
        try:
            raw_text = element.text
            if not raw_text:
                return None
            
            # Basic cleanup
            lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
            valid_lines = [line for line in lines if self._is_valid_paragraph(line)]
            
            content = '\n\n'.join(valid_lines)
            return content if content else None
        except Exception as e:
            logger.error(f"Raw text extraction failed: {e}")
            return None
    
    def _is_valid_paragraph(self, text):
        """Check if paragraph is valid story content"""
        if not text or len(text.strip()) < 3:
            return False
        
        # Skip separator lines and metadata
        invalid_patterns = [
            r'^={5,}',      # Separator lines
            r'^\d{5}\s',    # Chapter numbers like "00002"
            r'^https?://',  # URLs
            r'^www\.',      # Web addresses
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, text.strip()):
                return False
        
        return True
    
    def _validate_content(self, content):
        """Validate extracted content quality"""
        if not content or len(content.strip()) < 500:
            logger.debug(f"Content too short: {len(content) if content else 0} chars")
            return False
        
        # Check for Korean text
        korean_chars = re.findall(r'[가-힣]', content)
        if len(korean_chars) < 100:
            logger.debug(f"Insufficient Korean text: {len(korean_chars)} Korean chars")
            return False
        
        # Check content quality indicators
        sentence_indicators = content.count('.') + content.count('!') + content.count('?') + content.count('다.')
        if sentence_indicators < 10:
            logger.debug(f"Too few sentences: {sentence_indicators}")
            return False
        
        logger.debug("✅ Content validation passed")
        return True
    
    def _find_navigation_links(self):
        """Find navigation links for next/previous chapters"""
        nav_links = []
        current_url = self.driver.current_url
        current_chapter_id = self._extract_chapter_id_from_url(current_url)
        
        try:
            # Find all novel links
            links = self.driver.find_elements(By.CSS_SELECTOR, 'a[href*="/novel/"]')
            
            for link in links:
                try:
                    href = link.get_attribute('href')
                    if not href:
                        continue
                    
                    link_chapter_id = self._extract_chapter_id_from_url(href)
                    link_text = link.text.strip()
                    
                    if link_chapter_id and link_chapter_id != current_chapter_id:
                        nav_links.append({
                            'url': href,
                            'chapter_id': link_chapter_id,
                            'text': link_text
                        })
                        
                except Exception as e:
                    continue
            
            logger.debug(f"🧭 Found {len(nav_links)} navigation links")
            return nav_links
            
        except Exception as e:
            logger.warning(f"Navigation extraction failed: {e}")
            return []
    
    def fetch_chapter(self, url):
        """Fetch a single chapter"""
        try:
            logger.info(f"🔍 Fetching chapter: {url}")
            
            # Navigate to page
            self.driver.get(url)
            
            # Small delay for page load
            time.sleep(self.delay)
            
            # Wait for page to be ready
            WebDriverWait(self.driver, 10).until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )
            
            # Extract data
            chapter_title = self._extract_chapter_title()
            content = self._extract_content()
            nav_links = self._find_navigation_links()
            
            if not content:
                logger.error("❌ No valid content found")
                return None
            
            # Find next/prev links
            current_id = self._extract_chapter_id_from_url(url)
            next_url = None
            prev_url = None
            
            for link in sorted(nav_links, key=lambda x: x['chapter_id']):
                if link['chapter_id'] > current_id and not next_url:
                    next_url = link['url']
                elif link['chapter_id'] < current_id:
                    prev_url = link['url']
            
            chapter_data = {
                'title': chapter_title or f"Chapter {current_id}",
                'url': url,
                'chapter_id': current_id,
                'content': content,
                'word_count': len(content),
                'korean_chars': len(re.findall(r'[가-힣]', content)),
                'next_url': next_url,
                'prev_url': prev_url,
                'nav_links_found': len(nav_links)
            }
            
            logger.info(f"✅ Chapter extracted: {chapter_data['title']}")
            logger.info(f"📊 Stats: {chapter_data['word_count']} chars, {chapter_data['korean_chars']} Korean chars")
            
            return chapter_data
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch chapter: {e}")
            return None
    
    def scrape_sequence(self, start_url, max_chapters=10, direction="forward"):
        """Scrape a sequence of chapters"""
        logger.info(f"🚀 Starting scrape sequence from: {start_url}")
        logger.info(f"📊 Direction: {direction}, Max chapters: {max_chapters}")
        
        scraped_chapters = []
        current_url = start_url
        chapter_count = 0
        
        while current_url and chapter_count < max_chapters:
            chapter_data = self.fetch_chapter(current_url)
            if not chapter_data:
                logger.warning("⚠️  Failed to fetch chapter, stopping sequence")
                break
            
            scraped_chapters.append(chapter_data)
            chapter_count += 1
            
            logger.info(f"📈 Progress: {chapter_count}/{max_chapters} chapters scraped")
            
            # Move to next chapter
            if direction == "forward":
                current_url = chapter_data['next_url']
                if current_url:
                    logger.info(f"➡️  Next: {current_url}")
            else:  # backward
                current_url = chapter_data['prev_url']
                if current_url:
                    logger.info(f"⬅️  Previous: {current_url}")
            
            if not current_url:
                logger.info("🏁 No more chapters found in sequence")
                break
            
            # Rate limiting
            time.sleep(self.delay)
        
        logger.info(f"🎉 Scraping completed! Total chapters: {len(scraped_chapters)}")
        return scraped_chapters
    
    def save_chapters(self, chapters, output_dir="data/novels/booktoki_dungeon_defense"):
        """Save scraped chapters to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metadata
        metadata = {
            'scrape_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_chapters': len(chapters),
            'total_words': sum(ch['word_count'] for ch in chapters),
            'total_korean_chars': sum(ch['korean_chars'] for ch in chapters),
            'chapters': []
        }
        
        for i, chapter in enumerate(chapters, 1):
            # Save individual chapter
            safe_title = re.sub(r'[^\w\-_\.]', '_', chapter['title'])
            filename = f"chapter_{i:03d}_{chapter['chapter_id']:04d}_{safe_title}.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {chapter['title']}\n")
                f.write(f"# URL: {chapter['url']}\n")
                f.write(f"# Chapter ID: {chapter['chapter_id']}\n")
                f.write(f"# Word Count: {chapter['word_count']}\n")
                f.write(f"# Korean Characters: {chapter['korean_chars']}\n\n")
                f.write(chapter['content'])
            
            logger.info(f"💾 Saved: {filename}")
            
            # Add to metadata
            metadata['chapters'].append({
                'index': i,
                'title': chapter['title'],
                'chapter_id': chapter['chapter_id'],
                'url': chapter['url'],
                'filename': filename,
                'word_count': chapter['word_count'],
                'korean_chars': chapter['korean_chars']
            })
        
        # Save metadata
        metadata_file = os.path.join(output_dir, 'metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Save combined file
        combined_file = os.path.join(output_dir, 'dungeon_defense_complete.txt')
        with open(combined_file, 'w', encoding='utf-8') as f:
            f.write("# 던전 디펜스 (Dungeon Defense) - Complete\n")
            f.write(f"# Total Chapters: {len(chapters)}\n")
            f.write(f"# Total Words: {metadata['total_words']:,}\n")
            f.write(f"# Total Korean Characters: {metadata['total_korean_chars']:,}\n")
            f.write(f"# Scraped: {metadata['scrape_date']}\n\n")
            
            for chapter in chapters:
                f.write(f"\n{'='*80}\n")
                f.write(f"{chapter['title']} (Chapter {chapter['chapter_id']})\n")
                f.write(f"{'='*80}\n\n")
                f.write(chapter['content'])
                f.write("\n\n")
        
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info(f"📄 Combined file: {combined_file}")
        logger.info(f"📋 Metadata file: {metadata_file}")
        logger.info(f"📊 Total: {metadata['total_words']:,} words, {metadata['total_korean_chars']:,} Korean chars")
        
        return {
            'output_dir': output_dir,
            'combined_file': combined_file,
            'metadata_file': metadata_file,
            'chapters_saved': len(chapters),
            'total_words': metadata['total_words'],
            'total_korean_chars': metadata['total_korean_chars']
        }
    
    def cleanup(self):
        """Clean up resources"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("🧹 WebDriver cleaned up")
            except Exception as e:
                logger.warning(f"Warning during cleanup: {e}")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Selenium-based BookToki scraper for Korean web novels")
    parser.add_argument("url", help="Starting chapter URL")
    parser.add_argument("--max-chapters", type=int, default=10, help="Maximum chapters to scrape")
    parser.add_argument("--delay", type=float, default=2, help="Delay between requests (seconds)")
    parser.add_argument("--direction", choices=["forward", "backward"], default="forward", help="Scraping direction")
    parser.add_argument("--output", default="data/novels/booktoki_dungeon_defense", help="Output directory")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--visible", action="store_false", dest="headless", help="Run browser in visible mode (default)")
    
    args = parser.parse_args()
    
    scraper = None
    try:
        logger.info("🌟 Starting Selenium BookToki scraper...")
        scraper = SeleniumBookTokiScraper(headless=args.headless, delay=args.delay)
        
        chapters = scraper.scrape_sequence(
            args.url, 
            max_chapters=args.max_chapters, 
            direction=args.direction
        )
        
        if chapters:
            result = scraper.save_chapters(chapters, args.output)
            logger.info("🎉 SUCCESS!")
            logger.info(f"📁 {result['chapters_saved']} chapters saved to {result['output_dir']}")
            logger.info(f"📊 Total content: {result['total_words']:,} words, {result['total_korean_chars']:,} Korean characters")
        else:
            logger.error("❌ No chapters were scraped")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n⏹️  Scraping interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1
    finally:
        if scraper:
            scraper.cleanup()
    
    return 0


if __name__ == "__main__":
    exit(main())