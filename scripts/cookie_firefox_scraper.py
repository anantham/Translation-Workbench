#!/usr/bin/env python3
"""
Cookie-Aware Firefox BookToki Scraper

Loads existing cookies that have already passed Cloudflare verification.
This bypasses the need to solve challenges by reusing valid session data.
"""

import os
import sys
import time
import json
import re
import random
import logging
from urllib.parse import urljoin, urlparse
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CookieAwareFirefoxScraper:
    def __init__(self, cookie_file_path, visible=True):
        self.cookie_file_path = cookie_file_path
        self.driver = None
        self.action_chains = None
        self.setup_driver(visible)
        
        # Human-like behavior parameters
        self.reading_speeds = {
            'fast': (0.8, 1.5),
            'normal': (1.2, 2.5),
            'slow': (2.0, 4.0)
        }
        self.current_reading_speed = 'normal'
        
    def setup_driver(self, visible=True):
        """Set up Firefox with cookie loading capabilities"""
        try:
            firefox_options = Options()
            
            if not visible:
                firefox_options.add_argument('--headless')
                logger.info("🕶️  Running in headless mode")
            else:
                logger.info("👁️  Running in visible mode with cookie authentication")
            
            # Basic stealth configuration
            firefox_options.set_preference("dom.webdriver.enabled", False)
            firefox_options.set_preference("useAutomationExtension", False)
            firefox_options.set_preference("marionette.enabled", False)
            
            # User agent for macOS Firefox
            firefox_options.set_preference("general.useragent.override", 
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:133.0) Gecko/20100101 Firefox/133.0")
            
            # Korean language support
            firefox_options.set_preference("intl.accept_languages", "ko-KR,ko,en-US,en")
            firefox_options.set_preference("intl.locale.requested", "ko-KR")
            
            # Privacy settings
            firefox_options.set_preference("privacy.trackingprotection.enabled", True)
            firefox_options.set_preference("dom.webnotifications.enabled", False)
            firefox_options.set_preference("dom.popup_maximum", 0)
            
            self.driver = webdriver.Firefox(options=firefox_options)
            
            # Remove webdriver property
            self.driver.execute_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
            """)
            
            # Set realistic timeouts
            self.driver.implicitly_wait(10)
            self.driver.set_page_load_timeout(45)
            
            # Set window size
            self.driver.set_window_size(1440, 900)
            
            logger.info("✅ Firefox WebDriver initialized")
            
        except WebDriverException as e:
            logger.error(f"❌ Failed to initialize Firefox WebDriver: {e}")
            raise
    
    def parse_netscape_cookies(self, cookie_file_path):
        """Parse Netscape format cookie file"""
        cookies = []
        
        try:
            with open(cookie_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    
                    # Skip comments and empty lines
                    if line.startswith('#') or not line:
                        continue
                    
                    # Split cookie data
                    parts = line.split('\t')
                    if len(parts) != 7:
                        continue
                    
                    domain, domain_specified, path, secure, expiry, name, value = parts
                    
                    # Convert to Selenium cookie format
                    cookie = {
                        'name': name,
                        'value': value,
                        'domain': domain.lstrip('.'),  # Remove leading dot
                        'path': path,
                        'secure': secure.upper() == 'TRUE',
                    }
                    
                    # Add expiry if it's not session cookie (expiry != 0)
                    if expiry != '0':
                        cookie['expiry'] = int(expiry)
                    
                    cookies.append(cookie)
            
            logger.info(f"📄 Parsed {len(cookies)} cookies from file")
            return cookies
            
        except Exception as e:
            logger.error(f"❌ Failed to parse cookies: {e}")
            return []
    
    def load_cookies(self):
        """Load cookies into the browser"""
        try:
            logger.info("🍪 Loading authentication cookies...")
            
            # Parse cookies from file
            cookies = self.parse_netscape_cookies(self.cookie_file_path)
            
            if not cookies:
                logger.error("❌ No cookies found")
                return False
            
            # First, navigate to the domain to set cookies
            logger.info("🌐 Navigating to domain to set cookies...")
            self.driver.get("http://booktoki468.com")
            
            # Wait for basic page load
            time.sleep(2)
            
            # Add each cookie
            cookies_added = 0
            for cookie in cookies:
                try:
                    self.driver.add_cookie(cookie)
                    cookies_added += 1
                    logger.debug(f"✅ Added cookie: {cookie['name']}")
                    
                    # Special logging for important cookies
                    if cookie['name'] in ['cf_clearance', 'PHPSESSID']:
                        logger.info(f"🎯 Added critical cookie: {cookie['name']} = {cookie['value'][:20]}...")
                        
                except Exception as e:
                    logger.warning(f"⚠️  Failed to add cookie {cookie['name']}: {e}")
            
            logger.info(f"✅ Successfully loaded {cookies_added}/{len(cookies)} cookies")
            
            # Verify important cookies were added
            current_cookies = self.driver.get_cookies()
            cookie_names = [c['name'] for c in current_cookies]
            
            if 'cf_clearance' in cookie_names:
                logger.info("🔐 Cloudflare clearance cookie loaded - should bypass challenges!")
            else:
                logger.warning("⚠️  cf_clearance cookie not found - may still face challenges")
            
            if 'PHPSESSID' in cookie_names:
                logger.info("🔑 PHP session cookie loaded")
            
            return cookies_added > 0
            
        except Exception as e:
            logger.error(f"❌ Failed to load cookies: {e}")
            return False
    
    def navigate_with_cookies(self, url):
        """Navigate to URL using loaded cookies"""
        try:
            logger.info(f"🚀 Navigating with authentication cookies to: {url}")
            
            # Load cookies first
            if not self.load_cookies():
                logger.error("❌ Failed to load cookies")
                return False
            
            # Small delay
            time.sleep(2)
            
            # Navigate to target URL
            logger.info("🌐 Navigating to target page...")
            self.driver.get(url)
            
            # Wait for page load
            time.sleep(3)
            
            # Check page status
            page_title = self.driver.title
            current_url = self.driver.current_url
            
            logger.info(f"📄 Page title: {page_title}")
            logger.info(f"🔗 Current URL: {current_url}")
            
            # Check for challenge indicators
            challenge_indicators = [
                "just a moment", "please wait", "checking", "verifying",
                "잠시만 기다리", "잠시", "기다리", "확인", "검증",
                "cloudflare", "ray id"
            ]
            
            page_title_lower = page_title.lower()
            page_source_lower = self.driver.page_source.lower()
            
            is_challenge = any(indicator in page_title_lower or indicator in page_source_lower 
                             for indicator in challenge_indicators)
            
            if is_challenge:
                logger.warning("⚠️  Still showing challenge page - cookies may be expired")
                return False
            
            # Check for content
            content_found = bool(self.driver.find_elements(By.ID, "novel_content"))
            korean_chars = len(re.findall(r'[가-힣]', self.driver.page_source))
            
            logger.info(f"📝 Content container found: {content_found}")
            logger.info(f"🇰🇷 Korean characters detected: {korean_chars}")
            
            if content_found:
                logger.info("🎉 Successfully accessed content with cookies!")
                return True
            elif korean_chars > 100:
                logger.info("✅ Korean content detected - likely successful")
                return True
            else:
                logger.warning("⚠️  No clear content detected")
                return False
            
        except Exception as e:
            logger.error(f"❌ Navigation error: {e}")
            return False
    
    def extract_chapter_content(self):
        """Extract chapter content from current page"""
        try:
            logger.info("📖 Extracting chapter content...")
            
            # Extract title
            title = self._extract_chapter_title()
            if title:
                logger.info(f"📖 Reading: {title}")
            
            # Find content container
            content_container = None
            try:
                content_container = self.driver.find_element(By.ID, "novel_content")
            except NoSuchElementException:
                logger.error("❌ #novel_content not found")
                return None
            
            # Look for specific content div
            content_div = None
            try:
                content_div = content_container.find_element(By.CSS_SELECTOR, "div.f9e99a33513")
            except NoSuchElementException:
                content_div = content_container
            
            # Extract paragraphs
            paragraphs = content_div.find_elements(By.TAG_NAME, "p")
            if not paragraphs:
                logger.warning("No paragraphs found")
                return None
            
            logger.info(f"📄 Found {len(paragraphs)} paragraphs")
            
            chapter_text = []
            for i, paragraph in enumerate(paragraphs):
                try:
                    text = paragraph.text.strip()
                    if text and self._is_valid_paragraph(text):
                        chapter_text.append(text)
                        
                        if i < 3:  # Log first few paragraphs
                            logger.debug(f"📝 Paragraph {i+1}: {text[:80]}...")
                            
                except Exception as e:
                    logger.debug(f"Error processing paragraph {i}: {e}")
                    continue
            
            if not chapter_text:
                logger.error("No valid content found")
                return None
            
            content = '\\n\\n'.join(chapter_text)
            
            # Validate content
            if not self._validate_content(content):
                return None
            
            # Get current URL for chapter ID
            current_url = self.driver.current_url
            chapter_id = self._extract_chapter_id_from_url(current_url)
            
            result = {
                'title': title or f"Chapter {chapter_id}",
                'url': current_url,
                'chapter_id': chapter_id,
                'content': content,
                'word_count': len(content),
                'korean_chars': len(re.findall(r'[가-힣]', content)),
                'paragraph_count': len(chapter_text)
            }
            
            logger.info(f"✅ Content extracted: {result['word_count']} chars, {result['korean_chars']} Korean chars")
            return result
            
        except Exception as e:
            logger.error(f"❌ Content extraction failed: {e}")
            return None
    
    def find_next_chapter_link(self):
        """Find next chapter link"""
        try:
            logger.debug("🔍 Looking for next chapter link...")
            
            nav_links = self.driver.find_elements(By.CSS_SELECTOR, 'a[href*="/novel/"]')
            current_chapter_id = self._extract_chapter_id_from_url(self.driver.current_url)
            
            # Look for next chapter
            for link in nav_links:
                try:
                    href = link.get_attribute('href')
                    link_chapter_id = self._extract_chapter_id_from_url(href)
                    link_text = link.text.strip().lower()
                    
                    if (link_chapter_id and link_chapter_id > current_chapter_id and
                        ('다음' in link_text or 'next' in link_text or '>' in link_text or
                         link_chapter_id == current_chapter_id + 1)):
                        logger.info(f"🎯 Found next chapter: Chapter {link_chapter_id}")
                        return link
                        
                except Exception:
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error finding next link: {e}")
            return None
    
    def scrape_chapters(self, start_url, max_chapters=10):
        """Main scraping method"""
        logger.info(f"🚀 Starting cookie-authenticated scraping...")
        logger.info(f"📊 Target: {max_chapters} chapters from {start_url}")
        
        # Navigate to first page with cookies
        if not self.navigate_with_cookies(start_url):
            logger.error("❌ Failed to access first page")
            return []
        
        scraped_chapters = []
        chapter_count = 0
        
        while chapter_count < max_chapters:
            logger.info(f"\\n📖 === Chapter {chapter_count + 1} of {max_chapters} ===")
            
            # Extract content
            chapter_data = self.extract_chapter_content()
            
            if not chapter_data:
                logger.error("❌ Failed to extract chapter content")
                break
            
            scraped_chapters.append(chapter_data)
            chapter_count += 1
            
            logger.info(f"✅ Chapter {chapter_count} completed: {chapter_data['title']}")
            
            # Check if we need more chapters
            if chapter_count >= max_chapters:
                logger.info("🏁 Reached target chapter count")
                break
            
            # Find and navigate to next chapter
            next_link = self.find_next_chapter_link()
            if not next_link:
                logger.warning("⚠️  No next chapter found")
                break
            
            # Click next chapter
            try:
                logger.info("🔄 Moving to next chapter...")
                next_link.click()
                time.sleep(random.uniform(3, 8))  # Natural delay
            except Exception as e:
                logger.error(f"❌ Failed to click next chapter: {e}")
                break
        
        logger.info(f"🎉 Scraping completed! {len(scraped_chapters)} chapters scraped")
        return scraped_chapters
    
    # Helper methods
    def _extract_chapter_title(self):
        """Extract chapter title from page"""
        try:
            title = self.driver.title
            if title:
                chapter_match = re.search(r'([^_]+?-\\d+화)', title)
                if chapter_match:
                    return chapter_match.group(1)
            return title.split('-')[0].strip() if '-' in title else title
        except:
            return None
    
    def _extract_chapter_id_from_url(self, url):
        """Extract chapter ID from URL"""
        match = re.search(r'/novel/(\\d+)', url)
        return int(match.group(1)) if match else None
    
    def _is_valid_paragraph(self, text):
        """Check if paragraph is valid content"""
        if not text or len(text.strip()) < 3:
            return False
        
        invalid_patterns = [
            r'^={5,}', r'^\\d{5}\\s', r'^https?://', r'^www\\.'
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, text.strip()):
                return False
        
        return True
    
    def _validate_content(self, content):
        """Validate content quality"""
        if not content or len(content.strip()) < 500:
            return False
        
        korean_chars = re.findall(r'[가-힣]', content)
        if len(korean_chars) < 100:
            return False
        
        return True
    
    def save_chapters(self, chapters, output_dir="data/novels/booktoki_dungeon_defense_cookie"):
        """Save scraped chapters"""
        if not chapters:
            logger.error("No chapters to save")
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        
        metadata = {
            'scrape_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'scraper_version': 'cookie-auth-v1.0',
            'authentication_method': 'loaded_cookies',
            'total_chapters': len(chapters),
            'total_words': sum(ch['word_count'] for ch in chapters),
            'total_korean_chars': sum(ch['korean_chars'] for ch in chapters),
            'chapters': []
        }
        
        for i, chapter in enumerate(chapters, 1):
            safe_title = re.sub(r'[^\\w\\-_\\.]', '_', chapter['title'])
            filename = f"chapter_{i:03d}_{chapter['chapter_id']:04d}_{safe_title}.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {chapter['title']}\\n")
                f.write(f"# URL: {chapter['url']}\\n")
                f.write(f"# Chapter ID: {chapter['chapter_id']}\\n")
                f.write(f"# Word Count: {chapter['word_count']}\\n")
                f.write(f"# Korean Characters: {chapter['korean_chars']}\\n\\n")
                f.write(chapter['content'])
            
            logger.info(f"💾 Saved: {filename}")
            
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
        
        logger.info(f"📁 Output: {output_dir}")
        logger.info(f"📊 Total: {metadata['total_words']:,} words, {metadata['total_korean_chars']:,} Korean chars")
        
        return {
            'output_dir': output_dir,
            'chapters_saved': len(chapters),
            'total_words': metadata['total_words'],
            'total_korean_chars': metadata['total_korean_chars']
        }
    
    def cleanup(self):
        """Clean up resources"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("🧹 Firefox WebDriver cleaned up")
            except Exception as e:
                logger.warning(f"Warning during cleanup: {e}")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cookie-authenticated Firefox BookToki scraper")
    parser.add_argument("url", help="Starting chapter URL")
    parser.add_argument("--cookies", default="data/cookies/booktoki468.com_cookies.txt", help="Path to cookie file")
    parser.add_argument("--max-chapters", type=int, default=5, help="Maximum chapters to scrape")
    parser.add_argument("--output", default="data/novels/booktoki_dungeon_defense_cookie", help="Output directory")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    
    args = parser.parse_args()
    
    # Check if cookie file exists
    if not os.path.exists(args.cookies):
        logger.error(f"❌ Cookie file not found: {args.cookies}")
        return 1
    
    scraper = None
    try:
        logger.info("🌟 Starting Cookie-Authenticated Firefox BookToki Scraper...")
        logger.info(f"🍪 Using cookies from: {args.cookies}")
        logger.info("🔐 This should bypass Cloudflare challenges!")
        
        scraper = CookieAwareFirefoxScraper(args.cookies, visible=not args.headless)
        
        chapters = scraper.scrape_chapters(args.url, max_chapters=args.max_chapters)
        
        if chapters:
            result = scraper.save_chapters(chapters, args.output)
            logger.info("🎉 SUCCESS!")
            logger.info(f"📁 {result['chapters_saved']} chapters saved")
            logger.info(f"📊 Total: {result['total_words']:,} words, {result['total_korean_chars']:,} Korean chars")
        else:
            logger.error("❌ No chapters were scraped")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\\n⏹️  Scraping interrupted by user")
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