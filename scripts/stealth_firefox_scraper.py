#!/usr/bin/env python3
"""
Stealth Firefox BookToki Scraper

Uses advanced stealth techniques to bypass Cloudflare protection:
- Undetected Selenium with custom patches
- Advanced fingerprint spoofing
- Real user behavioral mimicking
- Canvas fingerprint protection
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
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StealthFirefoxBookTokiScraper:
    def __init__(self, visible=True):
        self.driver = None
        self.action_chains = None
        self.setup_stealth_driver(visible)
        
    def setup_stealth_driver(self, visible=True):
        """Set up Firefox with maximum stealth capabilities"""
        try:
            firefox_options = Options()
            
            if not visible:
                firefox_options.add_argument('--headless')
                logger.info("🕶️  Running in headless mode")
            else:
                logger.info("👁️  Running in visible mode with maximum stealth")
            
            # === ADVANCED STEALTH CONFIGURATION ===
            
            # 1. Core stealth preferences
            firefox_options.set_preference("dom.webdriver.enabled", False)
            firefox_options.set_preference("useAutomationExtension", False)
            firefox_options.set_preference("marionette.enabled", False)
            
            # 2. Advanced user agent with real browser data
            firefox_options.set_preference("general.useragent.override", 
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:133.0) Gecko/20100101 Firefox/133.0")
            
            # 3. Language and timezone (Korean setup)
            firefox_options.set_preference("intl.accept_languages", "ko-KR,ko,en-US,en")
            firefox_options.set_preference("intl.locale.requested", "ko-KR")
            firefox_options.set_preference("browser.locale", "ko-KR")
            
            # 4. Advanced privacy and fingerprinting protection
            firefox_options.set_preference("privacy.resistFingerprinting", False)  # Don't resist completely
            firefox_options.set_preference("privacy.trackingprotection.enabled", True)
            firefox_options.set_preference("privacy.trackingprotection.pbmode.enabled", True)
            
            # 5. WebGL and Canvas protection (prevent detection)
            firefox_options.set_preference("webgl.disabled", False)
            firefox_options.set_preference("webgl.renderer-string-override", "Intel Iris Pro OpenGL Engine")
            firefox_options.set_preference("webgl.vendor-string-override", "Intel Inc.")
            
            # 6. Network and connection settings
            firefox_options.set_preference("network.http.connection-retry-timeout", 0)
            firefox_options.set_preference("network.http.max-persistent-connections-per-server", 6)
            firefox_options.set_preference("network.http.pipelining", True)
            firefox_options.set_preference("network.http.pipelining.maxrequests", 8)
            
            # 7. JavaScript and DOM settings
            firefox_options.set_preference("javascript.enabled", True)
            firefox_options.set_preference("dom.webnotifications.enabled", False)
            firefox_options.set_preference("dom.popup_maximum", 0)
            
            # 8. Font and rendering settings (appear more natural)
            firefox_options.set_preference("gfx.downloadable_fonts.enabled", True)
            firefox_options.set_preference("browser.display.use_document_fonts", 1)
            firefox_options.set_preference("gfx.canvas.azure.backends", "direct2d1.1,cairo")
            
            # 9. Media and hardware settings
            firefox_options.set_preference("media.peerconnection.enabled", True)
            firefox_options.set_preference("media.webrtc.hw.h264.enabled", True)
            
            # 10. Advanced timing and performance
            firefox_options.set_preference("dom.min_timeout_value", 4)
            firefox_options.set_preference("dom.timeout.throttling_delay", 30000)
            
            self.driver = webdriver.Firefox(options=firefox_options)
            
            # === POST-INITIALIZATION STEALTH TECHNIQUES ===
            
            # 1. Remove automation indicators
            self.driver.execute_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined,
                });
                
                // Remove automation extension
                delete window.navigator.__webdriver_script_fn;
                delete window.navigator.__driver_evaluate;
                delete window.navigator.__webdriver_evaluate;
                delete window.navigator.__selenium_evaluate;
                delete window.navigator.__fxdriver_evaluate;
                delete window.navigator.__driver_unwrapped;
                delete window.navigator.__webdriver_unwrapped;
                delete window.navigator.__selenium_unwrapped;
                delete window.navigator.__fxdriver_unwrapped;
                delete window.navigator.__webdriver_script_func;
                delete window.navigator.__webdriver_script_function;
            """)
            
            # 2. Mock realistic navigator properties
            self.driver.execute_script("""
                // Mock languages
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['ko-KR', 'ko', 'en-US', 'en'],
                });
                
                // Mock plugins (realistic Firefox plugins)
                Object.defineProperty(navigator, 'plugins', {
                    get: () => {
                        return [
                            {name: 'Shockwave Flash', filename: 'libflashplayer.so'},
                            {name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer'},
                            {name: 'Native Client', filename: 'internal-nacl-plugin'},
                            {name: 'QuickTime Plug-in', filename: 'QuickTime Plugin.plugin'}
                        ];
                    }
                });
                
                // Mock platform
                Object.defineProperty(navigator, 'platform', {
                    get: () => 'MacIntel'
                });
            """)
            
            # 3. Set realistic screen and window properties
            self.driver.execute_script("""
                Object.defineProperty(screen, 'width', {get: () => 1440});
                Object.defineProperty(screen, 'height', {get: () => 900});
                Object.defineProperty(screen, 'availWidth', {get: () => 1440});
                Object.defineProperty(screen, 'availHeight', {get: () => 845});
                Object.defineProperty(screen, 'colorDepth', {get: () => 24});
                Object.defineProperty(screen, 'pixelDepth', {get: () => 24});
                
                // Mock realistic timing
                const originalDate = Date;
                Date.now = () => originalDate.now() + Math.floor(Math.random() * 100) - 50;
            """)
            
            # 4. Advanced canvas fingerprint protection
            self.driver.execute_script("""
                const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
                const originalToBlob = HTMLCanvasElement.prototype.toBlob;
                const originalGetImageData = CanvasRenderingContext2D.prototype.getImageData;
                
                HTMLCanvasElement.prototype.toDataURL = function() {
                    const result = originalToDataURL.apply(this, arguments);
                    // Add tiny random variation to avoid fingerprinting
                    return result.replace(/.$/, String.fromCharCode(Math.floor(Math.random() * 26) + 97));
                };
                
                CanvasRenderingContext2D.prototype.getImageData = function() {
                    const result = originalGetImageData.apply(this, arguments);
                    // Add subtle noise to image data
                    for(let i = 0; i < result.data.length; i += 4) {
                        if(Math.random() < 0.001) {
                            result.data[i] = Math.min(255, result.data[i] + Math.random() * 2 - 1);
                        }
                    }
                    return result;
                };
            """)
            
            # 5. Mock WebRTC and media devices
            self.driver.execute_script("""
                navigator.mediaDevices = {
                    enumerateDevices: () => Promise.resolve([
                        {deviceId: 'default', kind: 'audioinput', label: 'Default - Built-in Microphone'},
                        {deviceId: 'communications', kind: 'audiooutput', label: 'Communications - Built-in Speakers'}
                    ]),
                    getUserMedia: () => Promise.reject(new Error('Permission denied'))
                };
            """)
            
            # Setup action chains
            self.action_chains = ActionChains(self.driver)
            
            # Set realistic timeouts
            self.driver.implicitly_wait(10)
            self.driver.set_page_load_timeout(45)
            
            # Set window size to common resolution
            self.driver.set_window_size(1440, 900)
            
            # Add small random delay to appear more human
            time.sleep(random.uniform(1, 3))
            
            logger.info("✅ Maximum stealth Firefox WebDriver initialized")
            
        except WebDriverException as e:
            logger.error(f"❌ Failed to initialize Firefox WebDriver: {e}")
            raise
    
    def human_delay(self, min_seconds=0.5, max_seconds=3.0, action_type='general'):
        """Generate human-like delays with advanced randomization"""
        delay_ranges = {
            'click': (0.3, 1.2),
            'type': (0.1, 0.4),
            'read': (2.0, 6.0),
            'navigate': (3.0, 10.0),
            'scroll': (0.5, 2.0),
            'challenge': (10.0, 30.0),  # Longer delays for challenges
            'general': (min_seconds, max_seconds)
        }
        
        min_delay, max_delay = delay_ranges.get(action_type, (min_seconds, max_seconds))
        
        # Use beta distribution for more realistic human timing
        delay = np.random.beta(2, 5) * (max_delay - min_delay) + min_delay
        
        logger.debug(f"⏳ Human delay: {delay:.2f}s ({action_type})")
        time.sleep(delay)
    
    def simulate_human_behavior(self):
        """Simulate various human behaviors on the page"""
        try:
            # Random mouse movements
            actions = ActionChains(self.driver)
            
            # Move mouse around randomly
            for _ in range(random.randint(2, 5)):
                x = random.randint(100, 1340)
                y = random.randint(100, 800)
                actions.move_by_offset(x - 720, y - 450).perform()
                time.sleep(random.uniform(0.1, 0.5))
            
            # Random scrolling
            scroll_amount = random.randint(100, 500)
            self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
            self.human_delay(0.5, 2.0, 'scroll')
            
            # Random key presses (tab, arrow keys)
            if random.random() < 0.3:
                actions.send_keys(Keys.TAB).perform()
                self.human_delay(0.2, 0.8)
            
        except Exception as e:
            logger.debug(f"Human behavior simulation error: {e}")
    
    def navigate_to_page_with_stealth(self, url):
        """Navigate with maximum stealth techniques"""
        try:
            logger.info(f"🌐 Navigating with stealth to: {url}")
            
            # Pre-navigation behavior
            self.simulate_human_behavior()
            
            # Navigate to page
            self.driver.get(url)
            
            # Wait for initial load
            self.human_delay(3, 8, 'navigate')
            
            # Check page status
            page_title = self.driver.title.lower()
            page_source = self.driver.page_source.lower()
            
            logger.info(f"📄 Page title: {self.driver.title}")
            logger.info(f"🔗 Current URL: {self.driver.current_url}")
            
            # Advanced challenge detection
            challenge_indicators = [
                "just a moment", "please wait", "checking", "verifying",
                "잠시만 기다리", "잠시", "기다리", "확인", "검증",
                "cloudflare", "ray id", "performance & security",
                "browser check", "security check", "loading", "로딩"
            ]
            
            is_challenge_page = any(indicator in page_title or indicator in page_source 
                                  for indicator in challenge_indicators)
            
            if is_challenge_page:
                logger.info("🔐 Challenge page detected - implementing advanced bypass...")
                return self.handle_cloudflare_challenge()
            
            # Check for actual content
            if self.driver.find_elements(By.ID, "novel_content"):
                logger.info("✅ Successfully reached content page!")
                return True
            
            logger.warning("⚠️  Unclear page status - might need manual intervention")
            return self.wait_for_manual_intervention()
            
        except Exception as e:
            logger.error(f"❌ Navigation error: {e}")
            return False
    
    def handle_cloudflare_challenge(self):
        """Advanced Cloudflare challenge handling"""
        logger.info("🛡️  Implementing advanced Cloudflare bypass techniques...")
        
        # 1. Wait for challenge to fully load
        self.human_delay(5, 10, 'challenge')
        
        # 2. Simulate realistic user behavior during challenge
        for attempt in range(3):
            logger.info(f"🔄 Challenge bypass attempt {attempt + 1}/3")
            
            # Simulate reading the challenge page
            self.simulate_human_behavior()
            self.human_delay(8, 15, 'challenge')
            
            # Look for checkbox or challenge elements
            challenge_elements = [
                'input[type="checkbox"]',
                '.cf-turnstile',
                '#challenge-form',
                '.challenge-form',
                'iframe[src*="challenges.cloudflare.com"]'
            ]
            
            found_challenge = False
            for selector in challenge_elements:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    logger.info(f"🎯 Found challenge element: {selector}")
                    found_challenge = True
                    
                    # If it's a checkbox, try to click it
                    if 'checkbox' in selector:
                        try:
                            checkbox = elements[0]
                            # Simulate human-like clicking
                            self.driver.execute_script("arguments[0].scrollIntoView(true);", checkbox)
                            self.human_delay(2, 4, 'click')
                            
                            # Add mouse movement before click
                            actions = ActionChains(self.driver)
                            actions.move_to_element(checkbox).pause(random.uniform(0.5, 1.5)).click().perform()
                            
                            logger.info("✅ Clicked challenge checkbox")
                            self.human_delay(10, 20, 'challenge')
                            
                        except Exception as e:
                            logger.warning(f"Failed to click checkbox: {e}")
                    
                    break
            
            if not found_challenge:
                logger.info("🔍 No interactive challenge found - waiting for automatic resolution...")
                self.human_delay(10, 20, 'challenge')
            
            # Check if challenge is resolved
            current_title = self.driver.title.lower()
            if not any(indicator in current_title for indicator in ["잠시만", "just a moment", "please wait"]):
                if self.driver.find_elements(By.ID, "novel_content"):
                    logger.info("🎉 Challenge bypassed successfully!")
                    return True
        
        # If automatic bypass failed, wait for manual intervention
        logger.warning("⚠️  Automatic bypass failed - requesting manual intervention")
        return self.wait_for_manual_intervention()
    
    def wait_for_manual_intervention(self, timeout_seconds=300):
        """Wait for manual challenge resolution with better feedback"""
        logger.info("👤 MANUAL INTERVENTION REQUIRED")
        logger.info("🔐 Please solve the challenge manually in the browser window")
        logger.info("💡 Tips:")
        logger.info("   - Click any checkboxes or CAPTCHAs you see")
        logger.info("   - Wait for the page to load completely") 
        logger.info("   - The script will automatically continue once content is detected")
        logger.info(f"⏱️  Timeout: {timeout_seconds} seconds")
        
        start_time = time.time()
        last_status_time = start_time
        
        while time.time() - start_time < timeout_seconds:
            try:
                # Check every 3 seconds
                if time.time() - last_status_time > 3:
                    elapsed = int(time.time() - start_time)
                    remaining = int(timeout_seconds - elapsed)
                    
                    current_title = self.driver.title
                    logger.info(f"⏳ Waiting... ({elapsed}s elapsed, {remaining}s remaining)")
                    logger.debug(f"   Current title: {current_title}")
                    
                    last_status_time = time.time()
                
                # Check if challenge is resolved
                if self.driver.find_elements(By.ID, "novel_content"):
                    logger.info("🎉 Content detected - challenge solved!")
                    return True
                
                # Check if still on challenge page
                current_title = self.driver.title.lower()
                challenge_indicators = ["잠시만", "just a moment", "please wait", "checking", "loading"]
                
                if not any(indicator in current_title for indicator in challenge_indicators):
                    logger.info("✅ Appears to be past challenge page")
                    self.human_delay(2, 5)  # Give content time to load
                    
                    if self.driver.find_elements(By.ID, "novel_content"):
                        return True
                
                time.sleep(3)  # Check every 3 seconds
                
            except Exception as e:
                logger.debug(f"Error checking status: {e}")
                time.sleep(3)
        
        logger.error(f"⏰ Manual intervention timeout after {timeout_seconds} seconds")
        return False
    
    def test_stealth_access(self, url):
        """Test stealth access to the target URL"""
        logger.info("🔬 Testing stealth access capabilities...")
        
        success = self.navigate_to_page_with_stealth(url)
        
        if success:
            logger.info("🎉 STEALTH ACCESS SUCCESSFUL!")
            
            # Try to extract some basic info
            try:
                title = self.driver.title
                current_url = self.driver.current_url
                
                # Look for content indicators
                content_found = bool(self.driver.find_elements(By.ID, "novel_content"))
                korean_text = len(re.findall(r'[가-힣]', self.driver.page_source))
                
                logger.info(f"📄 Title: {title}")
                logger.info(f"🔗 URL: {current_url}")
                logger.info(f"📝 Content container found: {content_found}")
                logger.info(f"🇰🇷 Korean characters detected: {korean_text}")
                
                if content_found:
                    logger.info("✅ Ready for content extraction!")
                    return True
                else:
                    logger.warning("⚠️  No content container found")
                    return False
                    
            except Exception as e:
                logger.error(f"Error during content check: {e}")
                return False
        else:
            logger.error("❌ Stealth access failed")
            return False
    
    def cleanup(self):
        """Clean up resources"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("🧹 Stealth Firefox WebDriver cleaned up")
            except Exception as e:
                logger.warning(f"Warning during cleanup: {e}")


def main():
    """Main execution function for testing stealth access"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test stealth Firefox access to BookToki")
    parser.add_argument("url", help="Target URL to test")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    
    args = parser.parse_args()
    
    scraper = None
    try:
        logger.info("🌟 Starting Stealth Firefox BookToki Access Test...")
        logger.info("🦊 Using maximum stealth techniques!")
        logger.info("🔬 This will test if we can bypass Cloudflare protection")
        
        scraper = StealthFirefoxBookTokiScraper(visible=not args.headless)
        
        success = scraper.test_stealth_access(args.url)
        
        if success:
            logger.info("🎉 SUCCESS! Stealth access working")
            logger.info("💡 You can now run the full scraper")
            
            # Keep browser open for 30 seconds to verify
            logger.info("🕐 Keeping browser open for 30 seconds for verification...")
            time.sleep(30)
            
        else:
            logger.error("❌ Stealth access failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n⏹️  Test interrupted by user")
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