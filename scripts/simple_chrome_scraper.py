#!/usr/bin/env python3
"""
Simple Chrome Scraper for BookToki
Loads cookies, navigates to target URL, saves HTML, and exits.
"""

import os
import sys
import time
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleChromeBookTokiScraper:
    def __init__(self, cookie_file_path, visible=True):
        self.cookie_file_path = cookie_file_path
        self.driver = None
        self.setup_chrome_driver(visible)
        
    def setup_chrome_driver(self, visible=True):
        """Setup Chrome with basic configuration"""
        try:
            chrome_options = Options()
            
            if not visible:
                chrome_options.add_argument('--headless')
                logger.info("🕶️  Running in headless mode")
            else:
                logger.info("👁️  Running in visible Chrome mode")
            
            # Basic Chrome settings
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # User agent
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            
            self.driver = webdriver.Chrome(options=chrome_options)
            
            # Remove webdriver property
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            # Set window size
            self.driver.set_window_size(1440, 900)
            
            logger.info("✅ Chrome WebDriver initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Chrome WebDriver: {e}")
            raise
    
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
    
    def load_cookies_and_navigate(self, target_url):
        """Load cookies and navigate directly to target URL"""
        try:
            logger.info("🍪 Loading cookies and navigating to target URL...")
            
            # First navigate to domain to set cookies
            logger.info("🌐 Navigating to domain to set cookies...")
            self.driver.get("http://booktoki468.com")
            time.sleep(2)
            
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
            
            # Now navigate to target URL
            logger.info(f"🎯 Navigating to target URL: {target_url}")
            self.driver.get(target_url)
            
            # Wait for page load
            time.sleep(5)
            
            # Check page status
            page_title = self.driver.title
            current_url = self.driver.current_url
            
            logger.info(f"📄 Page title: {page_title}")
            logger.info(f"🔗 Current URL: {current_url}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Navigation failed: {e}")
            return False
    
    def save_page_html(self, output_file="booktoki_target_page.html"):
        """Save current page HTML"""
        try:
            logger.info(f"💾 Saving page HTML to: {output_file}")
            
            page_source = self.driver.page_source
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(page_source)
            
            logger.info(f"✅ HTML saved successfully: {output_file}")
            logger.info(f"📊 HTML size: {len(page_source):,} characters")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save HTML: {e}")
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
    target_url = "https://booktoki468.com/novel/3912072?stx=%EB%8D%98%EC%A0%84+%EB%94%94%ED%8E%9C%EC%8A%A4&book=%EC%99%84%EA%B2%B0%EC%86%8C%EC%84%A4&spage=1"
    cookie_file = "data/cookies/booktoki468.com_cookies.txt"
    
    if not os.path.exists(cookie_file):
        logger.error(f"❌ Cookie file not found: {cookie_file}")
        return 1
    
    scraper = None
    try:
        logger.info("🚀 Starting Simple Chrome BookToki Scraper...")
        logger.info(f"🎯 Target: {target_url}")
        logger.info(f"🍪 Cookies: {cookie_file}")
        
        scraper = SimpleChromeBookTokiScraper(cookie_file, visible=True)
        
        # Load cookies and navigate
        if scraper.load_cookies_and_navigate(target_url):
            # Save HTML
            if scraper.save_page_html("booktoki_target_page.html"):
                logger.info("🎉 SUCCESS! HTML saved")
            else:
                logger.error("❌ Failed to save HTML")
                return 1
        else:
            logger.error("❌ Failed to navigate")
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
    
    logger.info("✅ Script completed successfully")
    return 0


if __name__ == "__main__":
    exit(main())