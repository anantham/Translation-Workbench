#!/usr/bin/env python3
"""
Consolidated Chrome BookToki Scraper
Natural navigation flow: Homepage → Search → Novel Selection → ToC

Flow:
1. Load homepage with cookies
2. Click search icon to reveal search input
3. Type "던전 디펜스" and hit enter
4. Wait for search results with random delays
5. Click on the specific novel link
6. Wait and save final ToC page HTML
"""

import os
import sys
import time
import random
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException, NoSuchElementException

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConsolidatedChromeBookTokiScraper:
    def __init__(self, cookie_file_path, visible=True):
        self.cookie_file_path = cookie_file_path
        self.driver = None
        self.action_chains = None
        self.setup_chrome_driver(visible)
        
    def setup_chrome_driver(self, visible=True):
        """Setup Chrome with stealth configuration"""
        try:
            chrome_options = Options()
            
            if not visible:
                chrome_options.add_argument('--headless')
                logger.info("🕶️  Running in headless mode")
            else:
                logger.info("👁️  Running in visible Chrome mode for natural navigation")
            
            # Stealth settings
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Korean-friendly user agent
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            chrome_options.add_argument('--accept-lang=ko-KR,ko,en-US,en')
            
            self.driver = webdriver.Chrome(options=chrome_options)
            
            # Remove webdriver indicators
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            # Setup action chains
            self.action_chains = ActionChains(self.driver)
            
            # Set timeouts
            self.driver.implicitly_wait(10)
            self.driver.set_page_load_timeout(45)
            
            # Set window size
            self.driver.set_window_size(1440, 900)
            
            logger.info("✅ Chrome WebDriver initialized with stealth settings")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Chrome WebDriver: {e}")
            raise
    
    def human_delay(self, min_seconds=5, max_seconds=15, action_type='general'):
        """Generate realistic human delays with uniform distribution"""
        delay = random.uniform(min_seconds, max_seconds)
        logger.info(f"⏳ Human delay: {delay:.1f}s ({action_type})")
        time.sleep(delay)
    
    def parse_netscape_cookies(self, cookie_file_path):
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
            
            logger.info(f"📄 Parsed {len(cookies)} cookies from file")
            return cookies
            
        except Exception as e:
            logger.error(f"❌ Failed to parse cookies: {e}")
            return []
    
    def step1_load_homepage_with_cookies(self):
        """Step 1: Load homepage and authenticate with cookies"""
        try:
            logger.info("🏠 === STEP 1: Loading homepage with cookies ===")
            
            # Navigate to homepage
            logger.info("🌐 Navigating to homepage...")
            self.driver.get("http://booktoki468.com")
            
            # Human reading time for homepage
            self.human_delay(5, 10, 'reading homepage')
            
            # Parse and load cookies
            cookies = self.parse_netscape_cookies(self.cookie_file_path)
            if not cookies:
                logger.error("❌ No cookies found")
                return False
            
            # Add cookies
            cookies_added = 0
            for cookie in cookies:
                try:
                    self.driver.add_cookie(cookie)
                    cookies_added += 1
                    
                    if cookie['name'] in ['cf_clearance', 'PHPSESSID']:
                        logger.info(f"🔑 Added critical cookie: {cookie['name']}")
                        
                except Exception as e:
                    logger.debug(f"Failed to add cookie {cookie['name']}: {e}")
            
            logger.info(f"✅ Loaded {cookies_added}/{len(cookies)} cookies")
            
            # Refresh to activate cookies
            logger.info("🔄 Refreshing to activate session...")
            self.driver.refresh()
            
            # Wait and examine page
            self.human_delay(5, 10, 'page loading')
            
            page_title = self.driver.title
            logger.info(f"📄 Homepage title: {page_title}")
            
            # Save homepage HTML for debugging
            self.save_debug_html("step1_homepage")
            
            logger.info("✅ Step 1 completed: Homepage loaded with cookies")
            return True
            
        except Exception as e:
            logger.error(f"❌ Step 1 failed: {e}")
            return False
    
    def step2_perform_search(self, search_term="던전 디펜스"):
        """Step 2: Click search icon and perform search"""
        try:
            logger.info(f"🔍 === STEP 2: Performing search for '{search_term}' ===")
            
            # Look for search icon first
            logger.info("👆 Looking for search icon...")
            
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
                    if elements and elements[0].is_displayed():
                        search_icon = elements[0]
                        logger.info(f"✅ Found search icon with: {selector}")
                        break
                except:
                    continue
            
            if search_icon:
                # Human thinking time before clicking
                self.human_delay(3, 8, 'deciding to search')
                
                # Click search icon
                logger.info("👆 Clicking search icon to reveal input...")
                search_icon.click()
                
                # Wait for input to appear
                self.human_delay(2, 5, 'waiting for input to appear')
            else:
                logger.warning("⚠️  No search icon found, trying direct input search...")
            
            # Look for search input field
            search_input_selectors = [
                "input[name='stx']",
                "input[placeholder*='두글자']",
                "input[type='text'][class*='form-control']",
                "input[type='search']",
                "input[name*='search']",
                "input[placeholder*='검색']"
            ]
            
            search_input = None
            # Try multiple times as input might appear with delay
            for attempt in range(3):
                for selector in search_input_selectors:
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
                time.sleep(1)
            
            if not search_input:
                logger.error("❌ No search input found")
                return False
            
            # Click on search input
            logger.info("👆 Clicking search input...")
            search_input.click()
            self.human_delay(1, 3, 'focusing on input')
            
            # Clear any existing text
            search_input.clear()
            
            # Type search term with human-like typing
            logger.info(f"⌨️  Typing search term: {search_term}")
            for char in search_term:
                search_input.send_keys(char)
                time.sleep(random.uniform(0.1, 0.3))  # Realistic typing speed
            
            # Human pause before submitting
            self.human_delay(2, 5, 'reviewing search term')
            
            # Submit search
            logger.info("✅ Submitting search...")
            search_input.send_keys(Keys.RETURN)
            
            # Wait for search results with longer delay
            self.human_delay(8, 15, 'waiting for search results')
            
            # Check for challenge and wait if needed
            if not self.handle_challenge_if_present():
                logger.error("❌ Failed to handle challenge after search")
                return False
            
            # Save search results HTML
            self.save_debug_html("step2_search_results")
            
            page_title = self.driver.title
            logger.info(f"📄 Search results title: {page_title}")
            
            logger.info("✅ Step 2 completed: Search performed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Step 2 failed: {e}")
            return False
    
    def step3_click_novel_link(self, novel_title="던전 디펜스"):
        """Step 3: Click on the specific novel link from search results"""
        try:
            logger.info(f"📚 === STEP 3: Clicking on novel '{novel_title}' ===")
            
            # Human reading time for search results
            self.human_delay(5, 12, 'reading search results')
            
            # Look for the specific novel link
            logger.info(f"🎯 Looking for novel link containing '{novel_title}'...")
            
            # Based on your HTML structure: <span class="title white">던전 디펜스</span>
            novel_link_selectors = [
                f"a[href*='novel'] span.title:contains('{novel_title}')",
                f"span.title:contains('{novel_title}')",
                f".title:contains('{novel_title}')"
            ]
            
            # Find all links and scan their text
            all_links = self.driver.find_elements(By.TAG_NAME, "a")
            target_link = None
            
            logger.info(f"🔍 Scanning {len(all_links)} links for '{novel_title}'...")
            
            for link in all_links:
                try:
                    link_text = link.text.strip()
                    href = link.get_attribute('href') or ""
                    
                    # Check if this link contains our target novel
                    if (novel_title in link_text and 'novel' in href) or \
                       (novel_title in link_text and len(link_text) < 50):
                        logger.info(f"🎯 Found target novel link: '{link_text}' -> {href[:100]}")
                        target_link = link
                        break
                        
                except Exception:
                    continue
            
            # Alternative approach: look for elements containing the novel title
            if not target_link:
                logger.info("🔍 Trying alternative approach - looking for title elements...")
                try:
                    title_elements = self.driver.find_elements(By.XPATH, f"//*[contains(text(), '{novel_title}')]")
                    for element in title_elements:
                        # Find the parent link
                        parent = element
                        for _ in range(5):  # Check up to 5 levels up
                            parent = parent.find_element(By.XPATH, "..")
                            if parent.tag_name == 'a':
                                href = parent.get_attribute('href')
                                if href and 'novel' in href:
                                    logger.info(f"✅ Found novel link via parent: {href[:100]}")
                                    target_link = parent
                                    break
                        if target_link:
                            break
                except Exception as e:
                    logger.debug(f"Alternative approach failed: {e}")
            
            if not target_link:
                logger.error(f"❌ Could not find link for novel '{novel_title}'")
                return False
            
            # Human decision-making delay
            self.human_delay(3, 8, 'deciding to click novel')
            
            # Scroll to element first (humans do this)
            logger.info("📜 Scrolling to novel link...")
            self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", target_link)
            self.human_delay(2, 4, 'scrolling to element')
            
            # Click on the novel link
            logger.info("👆 Clicking on novel link...")
            target_link.click()
            
            # Wait for page to load with longer delay
            self.human_delay(8, 15, 'waiting for novel page to load')
            
            # Save novel page HTML
            self.save_debug_html("step3_novel_page")
            
            page_title = self.driver.title
            current_url = self.driver.current_url
            logger.info(f"📄 Novel page title: {page_title}")
            logger.info(f"🔗 Novel page URL: {current_url}")
            
            logger.info("✅ Step 3 completed: Novel link clicked")
            return True
            
        except Exception as e:
            logger.error(f"❌ Step 3 failed: {e}")
            return False
    
    def step4_save_final_page(self, output_file="booktoki_novel_toc_page.html"):
        """Step 4: Save the final ToC page HTML"""
        try:
            logger.info("💾 === STEP 4: Saving final ToC page ===")
            
            # Human reading time for the final page
            self.human_delay(5, 10, 'reading final page')
            
            page_source = self.driver.page_source
            current_url = self.driver.current_url
            page_title = self.driver.title
            
            logger.info(f"💾 Saving final page HTML to: {output_file}")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"<!-- Scraped from: {current_url} -->\n")
                f.write(f"<!-- Page title: {page_title} -->\n")
                f.write(f"<!-- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')} -->\n\n")
                f.write(page_source)
            
            logger.info(f"✅ Final HTML saved successfully: {output_file}")
            logger.info(f"📊 HTML size: {len(page_source):,} characters")
            logger.info(f"📄 Page title: {page_title}")
            logger.info(f"🔗 Final URL: {current_url}")
            
            logger.info("✅ Step 4 completed: Final ToC page saved")
            return True
            
        except Exception as e:
            logger.error(f"❌ Step 4 failed: {e}")
            return False
    
    def handle_challenge_if_present(self, timeout=120):
        """Detect and handle Cloudflare challenges"""
        try:
            page_title = self.driver.title.lower()
            page_source = self.driver.page_source.lower()
            
            # Challenge indicators
            challenge_indicators = [
                "잠시만 기다리", "사람인지 확인", "just a moment", "please wait", 
                "checking", "verifying", "cloudflare", "turnstile"
            ]
            
            is_challenge = any(indicator in page_title or indicator in page_source 
                             for indicator in challenge_indicators)
            
            if not is_challenge:
                logger.info("✅ No challenge detected")
                return True
            
            logger.info("🛡️  Cloudflare challenge detected - waiting for resolution...")
            logger.info("💡 This should resolve automatically with our cookies")
            
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                # Wait between checks
                self.human_delay(8, 15, 'waiting for challenge resolution')
                
                # Check if challenge resolved
                current_title = self.driver.title.lower()
                current_source = self.driver.page_source.lower()
                
                # Look for signs challenge is resolved
                challenge_still_present = any(indicator in current_title or indicator in current_source
                                           for indicator in challenge_indicators)
                
                if not challenge_still_present:
                    logger.info("✅ Challenge appears resolved!")
                    
                    # Additional check for actual content
                    if "북토끼" in self.driver.title or len(self.driver.page_source) > 10000:
                        logger.info("✅ Valid content detected - challenge resolved")
                        return True
                
                elapsed = int(time.time() - start_time)
                logger.info(f"⏳ Still waiting for challenge resolution... ({elapsed}s elapsed)")
            
            logger.error(f"❌ Challenge resolution timeout after {timeout}s")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error handling challenge: {e}")
            return False
    
    def save_debug_html(self, step_name):
        """Save HTML for debugging purposes"""
        try:
            filename = f"debug_{step_name}.html"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.driver.page_source)
            logger.info(f"🐛 Debug HTML saved: {filename}")
        except Exception as e:
            logger.debug(f"Failed to save debug HTML: {e}")
    
    def run_complete_flow(self):
        """Run the complete navigation flow"""
        logger.info("🚀 === STARTING COMPLETE NATURAL NAVIGATION FLOW ===")
        logger.info("📋 Flow: Homepage → Search → Novel Selection → ToC")
        logger.info("⏰ Using random delays (5-15 seconds) to avoid detection")
        logger.info("=" * 60)
        
        try:
            # Step 1: Load homepage with cookies
            if not self.step1_load_homepage_with_cookies():
                logger.error("❌ Flow failed at Step 1")
                return False
            
            # Step 2: Perform search
            if not self.step2_perform_search("던전 디펜스"):
                logger.error("❌ Flow failed at Step 2")
                return False
            
            # Step 3: Click novel link
            if not self.step3_click_novel_link("던전 디펜스"):
                logger.error("❌ Flow failed at Step 3")
                return False
            
            # Step 4: Save final page
            if not self.step4_save_final_page("booktoki_novel_toc_final.html"):
                logger.error("❌ Flow failed at Step 4")
                return False
            
            logger.info("🎉 === COMPLETE FLOW SUCCESSFUL! ===")
            logger.info("✅ All steps completed successfully")
            logger.info("📄 Final ToC page saved as 'booktoki_novel_toc_final.html'")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Complete flow failed: {e}")
            return False
    
    def cleanup(self):
        """Clean up resources"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("🧹 Chrome WebDriver cleaned up")
            except Exception as e:
                logger.warning(f"Cleanup warning: {e}")


def main():
    """Main execution function"""
    cookie_file = "data/cookies/booktoki468.com_cookies.txt"
    
    if not os.path.exists(cookie_file):
        logger.error(f"❌ Cookie file not found: {cookie_file}")
        return 1
    
    scraper = None
    try:
        logger.info("🌟 Starting Consolidated Chrome BookToki Scraper...")
        logger.info("🎯 Target: Natural navigation to 던전 디펜스 novel")
        logger.info(f"🍪 Cookies: {cookie_file}")
        
        scraper = ConsolidatedChromeBookTokiScraper(cookie_file, visible=True)
        
        # Run complete flow
        success = scraper.run_complete_flow()
        
        if success:
            logger.info("🎉 SUCCESS! Complete navigation flow completed")
            logger.info("📄 Check 'booktoki_novel_toc_final.html' for results")
        else:
            logger.error("❌ Navigation flow failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n⏹️  Interrupted by user")
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