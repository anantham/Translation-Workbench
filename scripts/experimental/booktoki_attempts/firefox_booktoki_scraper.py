#!/usr/bin/env python3
"""
Human-Like Firefox BookToki Scraper

Uses Firefox WebDriver which is much harder for Cloudflare to detect as a bot.
Implements sophisticated anti-detection techniques based on 2025 research.
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
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException, ElementClickInterceptedException
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HumanLikeFirefoxBookTokiScraper:
    def __init__(self, visible=True):
        self.driver = None
        self.action_chains = None
        self.setup_driver(visible)
        self.scraped_chapters = []
        
        # Human-like behavior parameters
        self.reading_speeds = {
            'fast': (0.8, 1.5),    # seconds per paragraph
            'normal': (1.2, 2.5),
            'slow': (2.0, 4.0)
        }
        self.current_reading_speed = 'normal'
        
    def setup_driver(self, visible=True):
        """Set up Firefox with advanced anti-detection techniques"""
        try:
            firefox_options = Options()
            
            # Only use headless if explicitly requested
            if not visible:
                firefox_options.add_argument('--headless')
                logger.info("🕶️  Running in headless mode")
            else:
                logger.info("👁️  Running in visible mode - you can solve CAPTCHAs manually")
            
            # === Advanced Anti-Detection Techniques for Firefox ===
            
            # 1. Realistic user agent for macOS
            firefox_options.set_preference("general.useragent.override", 
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:133.0) Gecko/20100101 Firefox/133.0")
            
            # 2. Language and locale (Korean support)
            firefox_options.set_preference("intl.accept_languages", "ko-KR,ko,en-US,en")
            firefox_options.set_preference("intl.locale.requested", "ko-KR")
            
            # 3. Disable automation indicators
            firefox_options.set_preference("dom.webdriver.enabled", False)
            firefox_options.set_preference("useAutomationExtension", False)
            firefox_options.set_preference("marionette.enabled", False)
            
            # 4. Privacy and security settings (more human-like)
            firefox_options.set_preference("privacy.trackingprotection.enabled", True)
            firefox_options.set_preference("dom.webnotifications.enabled", False)  # Block notifications
            firefox_options.set_preference("dom.popup_maximum", 0)  # Block popups
            
            # 5. Performance settings
            firefox_options.set_preference("permissions.default.image", 1)  # Allow images (more human-like)
            firefox_options.set_preference("javascript.enabled", True)
            
            # 6. Network settings to appear more natural
            firefox_options.set_preference("network.http.connection-retry-timeout", 0)
            firefox_options.set_preference("network.http.max-persistent-connections-per-server", 6)
            
            # 7. WebGL and canvas settings (avoid fingerprinting)
            firefox_options.set_preference("webgl.disabled", False)  # Enable WebGL (more natural)
            firefox_options.set_preference("privacy.resistFingerprinting", False)  # Don't resist completely
            
            # 8. Font and rendering settings
            firefox_options.set_preference("gfx.downloadable_fonts.enabled", True)
            firefox_options.set_preference("browser.display.use_document_fonts", 1)
            
            self.driver = webdriver.Firefox(options=firefox_options)
            
            # === JavaScript-based anti-detection ===
            
            # Remove webdriver properties
            self.driver.execute_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
            """)
            
            # Mock realistic navigator properties
            self.driver.execute_script("""
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['ko-KR', 'ko', 'en-US', 'en'],
                });
            """)
            
            # Set realistic screen properties
            self.driver.execute_script("""
                Object.defineProperty(screen, 'width', {get: () => 1440});
                Object.defineProperty(screen, 'height', {get: () => 900});
                Object.defineProperty(screen, 'availWidth', {get: () => 1440});
                Object.defineProperty(screen, 'availHeight', {get: () => 845});
            """)
            
            # Mock realistic plugin array
            self.driver.execute_script("""
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5],
                });
            """)
            
            # Setup action chains for human-like interactions
            self.action_chains = ActionChains(self.driver)
            
            # Set realistic timeouts
            self.driver.implicitly_wait(10)
            self.driver.set_page_load_timeout(45)
            
            # Set window size to common resolution
            self.driver.set_window_size(1440, 900)
            
            logger.info("✅ Advanced anti-detection Firefox WebDriver initialized")
            
        except WebDriverException as e:
            logger.error(f"❌ Failed to initialize Firefox WebDriver: {e}")
            logger.info("💡 Please install Firefox: brew install firefox geckodriver")
            raise
    
    def human_delay(self, min_seconds=0.5, max_seconds=3.0, action_type='general'):
        """Generate human-like delays based on action type"""
        
        delay_ranges = {
            'click': (0.2, 0.8),
            'type': (0.1, 0.3),
            'read': (1.0, 4.0),
            'navigate': (2.0, 8.0),
            'scroll': (0.3, 1.2),
            'general': (min_seconds, max_seconds)
        }
        
        min_delay, max_delay = delay_ranges.get(action_type, (min_seconds, max_seconds))
        
        # Add some randomness with normal distribution
        delay = np.random.normal((min_delay + max_delay) / 2, (max_delay - min_delay) / 6)
        delay = max(min_delay, min(max_delay, delay))  # Clamp to range
        
        logger.debug(f"⏳ Human delay: {delay:.2f}s ({action_type})")
        time.sleep(delay)
    
    def human_mouse_movement(self, element):
        """Move mouse in human-like pattern to element"""
        try:
            # Get current mouse position (approximate)
            current_x = random.randint(100, 800)
            current_y = random.randint(100, 600)
            
            # Get target element position
            target_x = element.location['x'] + element.size['width'] // 2
            target_y = element.location['y'] + element.size['height'] // 2
            
            # Create a curved path
            steps = random.randint(8, 15)
            
            for i in range(steps):
                progress = i / steps
                # Add some curvature and randomness
                curve_offset_x = random.randint(-20, 20) * (0.5 - abs(progress - 0.5))
                curve_offset_y = random.randint(-15, 15) * (0.5 - abs(progress - 0.5))
                
                intermediate_x = current_x + (target_x - current_x) * progress + curve_offset_x
                intermediate_y = current_y + (target_y - current_y) * progress + curve_offset_y
                
                self.action_chains.move_by_offset(
                    intermediate_x - current_x, 
                    intermediate_y - current_y
                ).perform()
                
                current_x, current_y = intermediate_x, intermediate_y
                time.sleep(random.uniform(0.01, 0.05))  # Small delays between moves
                
        except Exception as e:
            logger.debug(f"Mouse movement error (non-critical): {e}")
    
    def human_click(self, element, description="element"):
        """Perform human-like click with natural movement"""
        try:
            # Scroll element into view first
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
            self.human_delay(0.5, 1.0, 'scroll')
            
            # Human-like mouse movement
            self.human_mouse_movement(element)
            
            # Random small delay before click
            self.human_delay(0.1, 0.4, 'click')
            
            # Try different click methods for reliability
            try:
                element.click()
                logger.debug(f"✅ Clicked {description} successfully")
            except ElementClickInterceptedException:
                # Fallback: JavaScript click
                logger.debug(f"🔄 Using JavaScript click for {description}")
                self.driver.execute_script("arguments[0].click();", element)
            
            # Post-click delay
            self.human_delay(0.3, 1.0, 'click')
            return True
            
        except Exception as e:
            logger.warning(f"❌ Failed to click {description}: {e}")
            return False
    
    def wait_for_human_interaction(self, message, timeout_seconds=300):
        """Wait for human to solve CAPTCHA or handle challenges"""
        logger.info(f"👤 {message}")
        logger.info(f"⏱️  Waiting up to {timeout_seconds} seconds for you to complete...")
        
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                # Check if we're past the challenge page
                current_title = self.driver.title.lower()
                current_source = self.driver.page_source.lower()
                
                challenge_indicators = [
                    "just a moment", "please wait", "checking", "verifying",
                    "잠시만 기다리", "잠시", "기다리", "확인", "검증"  # Korean
                ]
                
                still_challenging = any(indicator in current_title for indicator in challenge_indicators)
                
                if not still_challenging:
                    logger.info("✅ Challenge appears to be solved - continuing...")
                    return True
                    
                # Check for actual content
                if self.driver.find_elements(By.ID, "novel_content"):
                    logger.info("✅ Content found - challenge solved!")
                    return True
                    
            except Exception as e:
                logger.debug(f"Error checking challenge status: {e}")
                pass
            
            time.sleep(2)  # Check every 2 seconds
        
        logger.warning(f"⏰ Timeout after {timeout_seconds} seconds")
        return False
    
    def navigate_to_page(self, url, is_first=False):
        """Navigate to page with human-like behavior"""
        try:
            logger.info(f"🌐 Navigating to: {url}")
            
            if is_first:
                self.driver.get(url)
                logger.info("🔍 Loaded first page - checking for challenges...")
                
                # Wait a moment for page to load
                self.human_delay(2, 4, 'navigate')
                
                # Check for Cloudflare challenge (multiple languages)
                page_title = self.driver.title.lower()
                page_source = self.driver.page_source.lower()
                
                challenge_indicators = [
                    "just a moment", "please wait", "checking", "verifying",
                    "잠시만 기다리", "잠시", "기다리", "확인", "검증",  # Korean
                    "cloudflare", "ray id", "performance & security"
                ]
                
                is_challenge_page = any(indicator in page_title or indicator in page_source 
                                      for indicator in challenge_indicators)
                
                if is_challenge_page:
                    success = self.wait_for_human_interaction(
                        "🔐 Cloudflare challenge detected! Please solve it manually in the browser.",
                        timeout_seconds=300
                    )
                    if not success:
                        logger.error("❌ Challenge not solved in time")
                        return False
                
            else:
                # For subsequent pages, use more natural navigation
                self.driver.get(url)
                
            # Additional wait for content to fully load
            self.human_delay(1, 3, 'navigate')
            
            # Wait for document ready
            WebDriverWait(self.driver, 20).until(
                lambda driver: driver.execute_script("return document.readyState") == "complete"
            )
            
            return True
            
        except TimeoutException:
            logger.error("⏰ Page load timeout")
            return False
        except Exception as e:
            logger.error(f"❌ Navigation error: {e}")
            return False
    
    def extract_chapter_content(self):
        """Extract chapter content with human reading simulation"""
        try:
            # Simulate arriving at page - quick scan
            self.human_delay(0.5, 1.5, 'general')
            
            # Extract title
            title = self._extract_chapter_title()
            if not title:
                logger.warning("⚠️  No title found")
                return None
            
            logger.info(f"📖 Reading: {title}")
            
            # Find content container
            content_container = None
            try:
                content_container = self.driver.find_element(By.ID, "novel_content")
            except NoSuchElementException:
                logger.warning("❌ #novel_content not found")
                return None
            
            # Look for the specific content div
            content_div = None
            try:
                content_div = content_container.find_element(By.CSS_SELECTOR, "div.f9e99a33513")
            except NoSuchElementException:
                content_div = content_container
            
            # Extract paragraphs with reading simulation
            paragraphs = content_div.find_elements(By.TAG_NAME, "p")
            if not paragraphs:
                logger.warning("No paragraphs found")
                return None
            
            logger.info(f"📄 Found {len(paragraphs)} paragraphs - simulating reading...")
            
            chapter_text = []
            reading_time = 0
            
            for i, paragraph in enumerate(paragraphs):
                try:
                    text = paragraph.text.strip()
                    if text and self._is_valid_paragraph(text):
                        chapter_text.append(text)
                        
                        # Simulate reading time based on length
                        word_count = len(text.split())
                        min_time, max_time = self.reading_speeds[self.current_reading_speed]
                        read_time = random.uniform(min_time, max_time) * (word_count / 100)
                        reading_time += read_time
                        
                        # Occasional scroll during reading
                        if i % 5 == 0 and i > 0:
                            self.driver.execute_script("window.scrollBy(0, 200);")
                            self.human_delay(0.2, 0.6, 'scroll')
                        
                        if i < 3:  # Log first few paragraphs
                            logger.debug(f"📝 Paragraph {i+1}: {text[:80]}...")
                            
                except Exception as e:
                    logger.debug(f"Error processing paragraph {i}: {e}")
                    continue
            
            if not chapter_text:
                logger.warning("No valid content found")
                return None
            
            # Simulate finishing reading
            logger.info(f"✅ Finished reading ({len(chapter_text)} paragraphs, simulated {reading_time:.1f}s reading)")
            self.human_delay(0.5, 2.0, 'read')
            
            content = '\n\n'.join(chapter_text)
            
            # Validate content
            if not self._validate_content(content):
                return None
            
            # Get current URL for chapter ID
            current_url = self.driver.current_url
            chapter_id = self._extract_chapter_id_from_url(current_url)
            
            return {
                'title': title,
                'url': current_url,
                'chapter_id': chapter_id,
                'content': content,
                'word_count': len(content),
                'korean_chars': len(re.findall(r'[가-힣]', content)),
                'paragraph_count': len(chapter_text),
                'simulated_reading_time': reading_time
            }
            
        except Exception as e:
            logger.error(f"❌ Content extraction failed: {e}")
            return None
    
    def find_next_chapter_link(self):
        """Find and click the next chapter link naturally"""
        try:
            logger.info("🔍 Looking for next chapter link...")
            
            # Look for navigation buttons
            nav_links = self.driver.find_elements(By.CSS_SELECTOR, 'a[href*="/novel/"]')
            
            next_link = None
            current_chapter_id = self._extract_chapter_id_from_url(self.driver.current_url)
            
            for link in nav_links:
                try:
                    link_text = link.text.strip().lower()
                    href = link.get_attribute('href')
                    link_chapter_id = self._extract_chapter_id_from_url(href)
                    
                    # Check if this looks like a next link
                    if (link_chapter_id and link_chapter_id > current_chapter_id and
                        ('다음' in link_text or 'next' in link_text or '>' in link_text or
                         '화' in link_text)):
                        next_link = link
                        logger.info(f"🎯 Found next chapter link: '{link_text}' -> Chapter {link_chapter_id}")
                        break
                        
                except Exception as e:
                    continue
            
            # If no obvious next link, try to find the next numerical chapter
            if not next_link:
                logger.info("🔍 Looking for next numerical chapter...")
                for link in nav_links:
                    try:
                        href = link.get_attribute('href')
                        link_chapter_id = self._extract_chapter_id_from_url(href)
                        
                        if link_chapter_id and link_chapter_id == current_chapter_id + 1:
                            next_link = link
                            logger.info(f"🎯 Found next chapter by ID: Chapter {link_chapter_id}")
                            break
                            
                    except Exception as e:
                        continue
            
            return next_link
            
        except Exception as e:
            logger.error(f"❌ Error finding next link: {e}")
            return None
    
    def click_next_chapter(self):
        """Click next chapter link with human-like behavior"""
        next_link = self.find_next_chapter_link()
        
        if not next_link:
            logger.warning("⚠️  No next chapter link found")
            return False
        
        # Simulate reading completion and decision to continue
        logger.info("🤔 Deciding whether to continue to next chapter...")
        self.human_delay(1, 3, 'general')
        
        # Human-like click
        success = self.human_click(next_link, "next chapter link")
        
        if success:
            logger.info("✅ Clicked next chapter - waiting for page load...")
            self.human_delay(2, 5, 'navigate')
            return True
        else:
            return False
    
    def scrape_chapters_interactively(self, start_url, max_chapters=10):
        """Main interactive scraping method"""
        logger.info("🚀 Starting human-like interactive scraping with Firefox...")
        logger.info(f"📊 Target: {max_chapters} chapters from {start_url}")
        
        # Navigate to first page
        if not self.navigate_to_page(start_url, is_first=True):
            logger.error("❌ Failed to load first page")
            return []
        
        scraped_chapters = []
        chapter_count = 0
        
        while chapter_count < max_chapters:
            logger.info(f"\n📖 === Chapter {chapter_count + 1} of {max_chapters} ===")
            
            # Extract content from current page
            chapter_data = self.extract_chapter_content()
            
            if not chapter_data:
                logger.error("❌ Failed to extract chapter content")
                break
            
            scraped_chapters.append(chapter_data)
            chapter_count += 1
            
            logger.info(f"✅ Chapter {chapter_count} completed: {chapter_data['title']}")
            logger.info(f"📊 Stats: {chapter_data['word_count']} chars, {chapter_data['korean_chars']} Korean chars")
            
            # Check if we need more chapters
            if chapter_count >= max_chapters:
                logger.info("🏁 Reached target chapter count")
                break
            
            # Navigate to next chapter
            logger.info("🔄 Moving to next chapter...")
            if not self.click_next_chapter():
                logger.warning("⚠️  Could not navigate to next chapter")
                break
            
            # Variable delay between chapters (human-like reading pause)
            reading_break = random.uniform(3, 8)
            logger.info(f"☕ Taking a {reading_break:.1f}s break between chapters...")
            time.sleep(reading_break)
        
        logger.info(f"🎉 Scraping completed! {len(scraped_chapters)} chapters scraped")
        return scraped_chapters
    
    # Helper methods (same as Chrome version but optimized for Firefox)
    def _extract_chapter_title(self):
        """Extract chapter title from current page"""
        try:
            title = self.driver.title
            if title:
                chapter_match = re.search(r'([^_]+?-\d+화)', title)
                if chapter_match:
                    return chapter_match.group(1)
            return title.split('-')[0].strip() if '-' in title else title
        except:
            return None
    
    def _extract_chapter_id_from_url(self, url):
        """Extract chapter ID from URL"""
        match = re.search(r'/novel/(\d+)', url)
        return int(match.group(1)) if match else None
    
    def _is_valid_paragraph(self, text):
        """Check if paragraph is valid story content"""
        if not text or len(text.strip()) < 3:
            return False
        
        invalid_patterns = [
            r'^={5,}',      # Separator lines
            r'^\d{5}\s',    # Chapter numbers
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
            return False
        
        korean_chars = re.findall(r'[가-힣]', content)
        if len(korean_chars) < 100:
            return False
        
        sentence_indicators = content.count('.') + content.count('!') + content.count('?') + content.count('다.')
        if sentence_indicators < 10:
            return False
        
        return True
    
    def save_chapters(self, chapters, output_dir="data/novels/booktoki_dungeon_defense_firefox"):
        """Save scraped chapters with comprehensive metadata"""
        if not chapters:
            logger.error("No chapters to save")
            return None
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create comprehensive metadata
        metadata = {
            'scrape_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'scraper_version': 'firefox-human-like-v1.0',
            'browser': 'Firefox',
            'total_chapters': len(chapters),
            'total_words': sum(ch['word_count'] for ch in chapters),
            'total_korean_chars': sum(ch['korean_chars'] for ch in chapters),
            'average_chapter_length': sum(ch['word_count'] for ch in chapters) / len(chapters),
            'simulated_total_reading_time': sum(ch.get('simulated_reading_time', 0) for ch in chapters),
            'chapter_id_range': [min(ch['chapter_id'] for ch in chapters), max(ch['chapter_id'] for ch in chapters)],
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
                f.write(f"# Korean Characters: {chapter['korean_chars']}\n")
                f.write(f"# Paragraphs: {chapter['paragraph_count']}\n")
                f.write(f"# Simulated Reading Time: {chapter.get('simulated_reading_time', 0):.1f}s\n\n")
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
                'korean_chars': chapter['korean_chars'],
                'paragraph_count': chapter['paragraph_count'],
                'simulated_reading_time': chapter.get('simulated_reading_time', 0)
            })
        
        # Save metadata
        metadata_file = os.path.join(output_dir, 'metadata.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Save combined file
        combined_file = os.path.join(output_dir, 'dungeon_defense_complete_firefox.txt')
        with open(combined_file, 'w', encoding='utf-8') as f:
            f.write("# 던전 디펜스 (Dungeon Defense) - Complete (Firefox)\n")
            f.write(f"# Scraped with Firefox Human-Like Bot v1.0\n")
            f.write(f"# Total Chapters: {len(chapters)}\n")
            f.write(f"# Total Words: {metadata['total_words']:,}\n")
            f.write(f"# Total Korean Characters: {metadata['total_korean_chars']:,}\n")
            f.write(f"# Average Chapter Length: {metadata['average_chapter_length']:.0f} words\n")
            f.write(f"# Simulated Reading Time: {metadata['simulated_total_reading_time']:.1f} seconds\n")
            f.write(f"# Chapter ID Range: {metadata['chapter_id_range'][0]} - {metadata['chapter_id_range'][1]}\n")
            f.write(f"# Scraped: {metadata['scrape_date']}\n\n")
            
            for chapter in chapters:
                f.write(f"\n{'='*80}\n")
                f.write(f"{chapter['title']} (Chapter {chapter['chapter_id']})\n")
                f.write(f"{'='*80}\n\n")
                f.write(chapter['content'])
                f.write("\n\n")
        
        result = {
            'output_dir': output_dir,
            'combined_file': combined_file,
            'metadata_file': metadata_file,
            'chapters_saved': len(chapters),
            'total_words': metadata['total_words'],
            'total_korean_chars': metadata['total_korean_chars'],
            'simulated_reading_time': metadata['simulated_total_reading_time']
        }
        
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info(f"📄 Combined file: {combined_file}")
        logger.info(f"📋 Metadata file: {metadata_file}")
        logger.info(f"📊 Total: {result['total_words']:,} words, {result['total_korean_chars']:,} Korean chars")
        logger.info(f"⏱️  Simulated reading: {result['simulated_reading_time']:.1f} seconds")
        
        return result
    
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
    
    parser = argparse.ArgumentParser(description="Human-like interactive Firefox BookToki scraper")
    parser.add_argument("url", help="Starting chapter URL")
    parser.add_argument("--max-chapters", type=int, default=5, help="Maximum chapters to scrape")
    parser.add_argument("--output", default="data/novels/booktoki_dungeon_defense_firefox", help="Output directory")
    parser.add_argument("--reading-speed", choices=['fast', 'normal', 'slow'], default='normal', help="Simulated reading speed")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (not recommended for CAPTCHA sites)")
    
    args = parser.parse_args()
    
    scraper = None
    try:
        logger.info("🌟 Starting Human-Like Firefox BookToki Scraper...")
        logger.info("🦊 Using Firefox WebDriver - much harder to detect than Chrome!")
        logger.info("👤 This will open a visible browser so you can solve CAPTCHAs manually")
        logger.info("⏳ Please be patient - human-like behavior takes time!")
        
        scraper = HumanLikeFirefoxBookTokiScraper(visible=not args.headless)
        scraper.current_reading_speed = args.reading_speed
        
        chapters = scraper.scrape_chapters_interactively(
            args.url, 
            max_chapters=args.max_chapters
        )
        
        if chapters:
            result = scraper.save_chapters(chapters, args.output)
            logger.info("🎉 SUCCESS!")
            logger.info(f"📁 {result['chapters_saved']} chapters saved to {result['output_dir']}")
            logger.info(f"📊 Total content: {result['total_words']:,} words")
            logger.info(f"🇰🇷 Korean characters: {result['total_korean_chars']:,}")
            logger.info(f"⏱️  Reading simulation: {result['simulated_reading_time']:.1f} seconds")
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