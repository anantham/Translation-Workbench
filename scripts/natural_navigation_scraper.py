#!/usr/bin/env python3
"""
Natural Navigation BookToki Scraper

Implements human-like browsing behavior:
1. Load cookies on homepage
2. Examine page structure  
3. Use search functionality naturally
4. Navigate through results like a human
5. Never directly jump to deep URLs
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
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NaturalNavigationScraper:
    """
    Scraper that navigates like a real human user would:
    - Starts at homepage
    - Uses search functionality  
    - Browses through categories
    - Never jumps directly to deep URLs
    """
    
    def __init__(self, cookie_file_path=None, visible=True):
        self.cookie_file_path = cookie_file_path
        self.driver = None
        self.action_chains = None
        self.setup_natural_browser(visible)
        
    def setup_natural_browser(self, visible=True):
        """Setup browser optimized for natural human-like behavior"""
        try:
            firefox_options = Options()
            
            if not visible:
                firefox_options.add_argument('--headless')
            else:
                logger.info("👁️  Running in visible mode for natural interaction")
            
            # Basic stealth without over-engineering
            firefox_options.set_preference("dom.webdriver.enabled", False)
            firefox_options.set_preference("useAutomationExtension", False)
            firefox_options.set_preference("marionette.enabled", False)
            
            # Korean-friendly settings
            firefox_options.set_preference("general.useragent.override", 
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:133.0) Gecko/20100101 Firefox/133.0")
            firefox_options.set_preference("intl.accept_languages", "ko-KR,ko,en-US,en")
            firefox_options.set_preference("intl.locale.requested", "ko-KR")
            
            # Natural browsing settings
            firefox_options.set_preference("privacy.trackingprotection.enabled", True)
            firefox_options.set_preference("dom.webnotifications.enabled", False)
            firefox_options.set_preference("dom.popup_maximum", 0)
            
            self.driver = webdriver.Firefox(options=firefox_options)
            
            # Basic automation removal
            self.driver.execute_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
            """)
            
            self.action_chains = ActionChains(self.driver)
            self.driver.implicitly_wait(10)
            self.driver.set_page_load_timeout(45)
            self.driver.set_window_size(1440, 900)
            
            logger.info("✅ Natural navigation browser initialized")
            
        except WebDriverException as e:
            logger.error(f"❌ Failed to initialize browser: {e}")
            raise
    
    def human_delay(self, min_seconds=1, max_seconds=4, action_type='general'):
        """Generate natural human delays"""
        delay_ranges = {
            'reading': (2, 6),      # Time to read content
            'searching': (1, 3),    # Time to think about search
            'clicking': (0.5, 1.5), # Time before clicking
            'typing': (0.1, 0.3),   # Between keystrokes
            'page_load': (3, 8),    # Waiting for page
            'general': (min_seconds, max_seconds)
        }
        
        min_delay, max_delay = delay_ranges.get(action_type, (min_seconds, max_seconds))
        delay = random.uniform(min_delay, max_delay)
        
        logger.debug(f"⏳ Human delay: {delay:.1f}s ({action_type})")
        time.sleep(delay)
    
    def parse_cookies(self, cookie_file_path):
        """Parse Netscape format cookies"""
        cookies = []
        try:
            with open(cookie_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) != 7:
                        continue
                    
                    domain, domain_specified, path, secure, expiry, name, value = parts
                    
                    cookie = {
                        'name': name,
                        'value': value,
                        'domain': domain.lstrip('.'),
                        'path': path,
                        'secure': secure.upper() == 'TRUE',
                    }
                    
                    if expiry != '0':
                        cookie['expiry'] = int(expiry)
                    
                    cookies.append(cookie)
            
            logger.info(f"📄 Parsed {len(cookies)} cookies")
            return cookies
            
        except Exception as e:
            logger.error(f"❌ Cookie parsing failed: {e}")
            return []
    
    def start_at_homepage_with_cookies(self):
        """
        Start natural browsing session at homepage with cookies
        This is how humans actually browse - they start at the main page
        """
        try:
            logger.info("🏠 Starting at homepage (natural entry point)")
            
            # Go to homepage first
            self.driver.get("http://booktoki468.com")
            
            # Human reading time for homepage
            self.human_delay(action_type='page_load')
            
            # Load cookies if available
            if self.cookie_file_path and os.path.exists(self.cookie_file_path):
                logger.info("🍪 Loading authentication cookies...")
                
                cookies = self.parse_cookies(self.cookie_file_path)
                cookies_loaded = 0
                
                for cookie in cookies:
                    try:
                        self.driver.add_cookie(cookie)
                        cookies_loaded += 1
                        
                        if cookie['name'] in ['cf_clearance', 'PHPSESSID']:
                            logger.info(f"🔑 Loaded critical cookie: {cookie['name']}")
                    except Exception as e:
                        logger.debug(f"Failed to add cookie {cookie['name']}: {e}")
                
                logger.info(f"✅ Loaded {cookies_loaded} cookies")
                
                # Refresh to activate cookies (like a human would)
                logger.info("🔄 Refreshing to activate session...")
                self.driver.refresh()
                self.human_delay(action_type='page_load')
            
            # Check page status
            page_title = self.driver.title
            logger.info(f"📄 Homepage loaded: {page_title}")
            
            # Check for challenge
            if any(indicator in page_title.lower() for indicator in ["잠시만", "just a moment", "please wait"]):
                logger.info("🛡️  Challenge detected on homepage - waiting for resolution...")
                if not self.wait_for_challenge_resolution():
                    return False
            
            # Save and examine homepage HTML
            self.save_and_examine_page("homepage")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Homepage loading failed: {e}")
            return False
    
    def save_and_examine_page(self, page_name):
        """
        Save current page HTML and examine its structure
        This helps understand the site layout like a human would
        """
        try:
            # Save HTML for examination
            html_file = f"debug_{page_name}_page.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(self.driver.page_source)
            
            logger.info(f"💾 Saved {page_name} HTML to: {html_file}")
            
            # Examine key elements (like a human scanning the page)
            logger.info(f"🔍 Examining {page_name} structure...")
            
            # Look for navigation elements
            nav_elements = self.driver.find_elements(By.TAG_NAME, "nav")
            logger.info(f"   📍 Navigation sections: {len(nav_elements)}")
            
            # Look for search functionality
            search_inputs = self.driver.find_elements(By.CSS_SELECTOR, "input[type='search'], input[name*='search'], input[placeholder*='검색'], input[placeholder*='search']")
            logger.info(f"   🔍 Search inputs found: {len(search_inputs)}")
            
            # Look for links
            all_links = self.driver.find_elements(By.TAG_NAME, "a")
            logger.info(f"   🔗 Total links: {len(all_links)}")
            
            # Look for Korean content
            korean_chars = len(re.findall(r'[가-힣]', self.driver.page_source))
            logger.info(f"   🇰🇷 Korean characters: {korean_chars}")
            
            # Look for novel-related content
            novel_links = self.driver.find_elements(By.CSS_SELECTOR, "a[href*='novel']")
            logger.info(f"   📚 Novel-related links: {len(novel_links)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Page examination failed: {e}")
            return False
    
    def natural_search(self, search_term="완결소설"):
        """
        Perform natural search like a human would
        1. Look for search icon (fa fa-search) 
        2. Click to reveal search box
        3. Type in revealed input field
        4. Submit search
        """
        try:
            logger.info(f"🔍 Performing natural search for: {search_term}")
            
            # Human thinking time before searching
            self.human_delay(action_type='searching')
            
            # First, look for the search icon that reveals the input
            logger.info("👆 Looking for search icon to click...")
            
            search_icon_selectors = [
                ".fa.fa-search",
                ".fa-search", 
                "i.fa-search",
                "[class*='fa-search']",
                ".search-icon",
                ".search-btn"
            ]
            
            search_icon = None
            for selector in search_icon_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        search_icon = elements[0]
                        logger.info(f"✅ Found search icon with: {selector}")
                        break
                except:
                    continue
            
            if search_icon:
                # Click the search icon to reveal the input field
                logger.info("👆 Clicking search icon to reveal input field...")
                search_icon.click()
                self.human_delay(action_type='clicking')
            else:
                logger.warning("⚠️  No search icon found, trying direct input search...")
            
            # Now look for the revealed search input
            search_selectors = [
                "input[name='stx']",  # Based on your hint
                "input[placeholder*='두글자']",  # Your placeholder text
                "input[type='text'][class*='form-control']",
                "input[type='search']",
                "input[name*='search']", 
                "input[placeholder*='검색']",
                "input[placeholder*='search']",
                ".search input",
                "#search input",
                "input[class*='search']"
            ]
            
            search_input = None
            # Try multiple times as input might be revealed with delay
            for attempt in range(3):
                for selector in search_selectors:
                    try:
                        elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                        visible_elements = [el for el in elements if el.is_displayed()]
                        if visible_elements:
                            search_input = visible_elements[0]
                            logger.info(f"✅ Found search input with: {selector}")
                            break
                    except:
                        continue
                
                if search_input:
                    break
                    
                # Wait a bit for input to appear
                time.sleep(1)
            
            if not search_input:
                logger.error("❌ No search input found even after clicking icon")
                return False
            
            # Natural clicking behavior
            logger.info("👆 Clicking search box naturally...")
            search_input.click()
            self.human_delay(action_type='clicking')
            
            # Clear any existing text
            search_input.clear()
            
            # Human-like typing
            logger.info(f"⌨️  Typing search term: {search_term}")
            for char in search_term:
                search_input.send_keys(char)
                # Variable typing speed like humans
                typing_delay = random.uniform(0.08, 0.25)
                time.sleep(typing_delay)
            
            # Human pause before submitting
            self.human_delay(action_type='searching')
            
            # Submit search (try Enter first, then look for button)
            logger.info("✅ Submitting search...")
            search_input.send_keys(Keys.RETURN)
            
            # Wait for search results
            self.human_delay(action_type='page_load')
            
            # Save and examine search results
            self.save_and_examine_page("search_results")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Natural search failed: {e}")
            return False
    
    def browse_search_results_naturally(self, target_novel="던전 디펜스"):
        """
        Browse search results like a human would:
        1. Read/scan the results
        2. Look for target novel
        3. Click on it naturally
        """
        try:
            logger.info(f"📖 Browsing search results for: {target_novel}")
            
            # Human reading time for results
            self.human_delay(action_type='reading')
            
            # Look for novel links in results
            possible_selectors = [
                f"a[href*='novel']:contains('{target_novel}')",
                f"a:contains('{target_novel}')",
                "a[href*='novel']",
                ".novel-title a",
                ".book-title a", 
                ".title a"
            ]
            
            target_link = None
            
            # Search through all links for our target
            all_links = self.driver.find_elements(By.TAG_NAME, "a")
            logger.info(f"🔍 Scanning {len(all_links)} links for target novel...")
            
            for link in all_links:
                try:
                    link_text = link.text.strip()
                    href = link.get_attribute('href') or ""
                    
                    # Check if this looks like our target
                    if (target_novel in link_text or 
                        "던전" in link_text or
                        "디펜스" in link_text or
                        ("novel" in href and len(link_text) > 5)):
                        
                        logger.info(f"🎯 Found potential match: '{link_text}' -> {href[:100]}")
                        target_link = link
                        break
                        
                except Exception:
                    continue
            
            if not target_link:
                # If no exact match, just pick a novel link to continue the flow
                novel_links = [link for link in all_links 
                             if link.get_attribute('href') and 'novel' in link.get_attribute('href')]
                if novel_links:
                    target_link = novel_links[0]
                    logger.info("📚 Using first available novel link")
            
            if not target_link:
                logger.error("❌ No suitable novel link found")
                return False
            
            # Human decision-making delay
            self.human_delay(action_type='reading')
            
            # Natural clicking
            logger.info("👆 Clicking on novel link naturally...")
            
            # Scroll to element first (humans do this)
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", target_link)
            self.human_delay(action_type='clicking')
            
            # Click
            target_link.click()
            
            # Wait for page load
            self.human_delay(action_type='page_load')
            
            # Save and examine novel page
            self.save_and_examine_page("novel_page")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Result browsing failed: {e}")
            return False
    
    def wait_for_challenge_resolution(self, timeout=300):
        """Wait for any challenges to resolve"""
        logger.info("⏳ Waiting for challenge resolution...")
        logger.info("💡 If you see a challenge, please solve it manually")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                current_title = self.driver.title.lower()
                
                # Check if challenge is resolved
                challenge_indicators = ["잠시만", "just a moment", "please wait", "checking"]
                if not any(indicator in current_title for indicator in challenge_indicators):
                    logger.info("✅ Challenge resolved")
                    return True
                
                # Check for content
                if self.driver.find_elements(By.ID, "novel_content"):
                    logger.info("✅ Content detected - challenge resolved!")
                    return True
                
                elapsed = int(time.time() - start_time)
                if elapsed % 10 == 0 and elapsed > 0:
                    logger.info(f"⏳ Still waiting... ({elapsed}s)")
                
                time.sleep(3)
                
            except:
                time.sleep(3)
        
        logger.warning(f"⏰ Challenge timeout after {timeout}s")
        return False
    
    def natural_browsing_session(self):
        """
        Complete natural browsing session:
        1. Start at homepage with cookies
        2. Search for novels
        3. Browse results naturally
        4. Navigate to target content
        """
        logger.info("🚀 Starting NATURAL BROWSING SESSION")
        logger.info("=" * 50)
        logger.info("👤 Simulating real human browsing behavior:")
        logger.info("   1. Load homepage with cookies")
        logger.info("   2. Search for '완결소설' (completed novels)")
        logger.info("   3. Browse results naturally")
        logger.info("   4. Click on target novel")
        logger.info("=" * 50)
        
        try:
            # Step 1: Start at homepage
            if not self.start_at_homepage_with_cookies():
                logger.error("❌ Failed at homepage")
                return False
            
            logger.info("✅ Step 1 complete: Homepage loaded")
            
            # Step 2: Natural search
            if not self.natural_search("완결소설"):
                logger.error("❌ Failed at search")
                return False
            
            logger.info("✅ Step 2 complete: Search performed")
            
            # Step 3: Browse results
            if not self.browse_search_results_naturally("던전 디펜스"):
                logger.error("❌ Failed browsing results")  
                return False
            
            logger.info("✅ Step 3 complete: Results browsed")
            
            # Final check
            current_url = self.driver.current_url
            current_title = self.driver.title
            korean_content = len(re.findall(r'[가-힣]', self.driver.page_source))
            
            logger.info("🎉 NATURAL BROWSING SESSION COMPLETE!")
            logger.info(f"📄 Final title: {current_title}")
            logger.info(f"🔗 Final URL: {current_url}")
            logger.info(f"🇰🇷 Korean content: {korean_content} characters")
            
            # Check if we have novel content
            has_content = bool(self.driver.find_elements(By.ID, "novel_content"))
            logger.info(f"📚 Novel content found: {has_content}")
            
            if has_content and korean_content > 1000:
                logger.info("🎉 SUCCESS: Natural navigation reached novel content!")
                logger.info("🔥 Ready for content extraction")
                
                # Keep browser open for verification
                logger.info("🕐 Keeping browser open for 30s verification...")
                time.sleep(30)
                
                return True
            else:
                logger.warning("⚠️  Reached a page but unclear if it's the target content")
                logger.info("🕐 Keeping browser open for 60s manual inspection...")
                time.sleep(60)
                return True
            
        except Exception as e:
            logger.error(f"❌ Natural browsing failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("🧹 Browser cleaned up")
            except Exception as e:
                logger.warning(f"Cleanup warning: {e}")


def main():
    """Test natural navigation approach"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Natural Navigation BookToki Scraper")
    parser.add_argument("--cookies", default="data/cookies/booktoki468.com_cookies.txt", help="Cookie file path")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    
    args = parser.parse_args()
    
    scraper = None
    try:
        logger.info("🌟 Starting NATURAL NAVIGATION SCRAPER")
        logger.info("🧠 Strategy: Browse like a real human user")
        logger.info("🎯 No direct URL jumps - only natural navigation")
        
        cookie_path = args.cookies if os.path.exists(args.cookies) else None
        if not cookie_path:
            logger.warning("⚠️  No cookies found - proceeding without authentication")
        
        scraper = NaturalNavigationScraper(
            cookie_file_path=cookie_path,
            visible=not args.headless
        )
        
        success = scraper.natural_browsing_session()
        
        return 0 if success else 1
            
    except KeyboardInterrupt:
        logger.info("\n⏹️  Session interrupted")
        return 1
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1
    finally:
        if scraper:
            scraper.cleanup()


if __name__ == "__main__":
    exit(main())