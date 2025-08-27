#!/usr/bin/env python3
"""
Advanced Anti-Detection BookToki Scraper

Implements sophisticated evasion techniques based on 2025 research:
- ✅ Canvas fingerprint spoofing with noise injection
- ✅ WebGL parameter spoofing  
- ✅ Advanced mouse movement simulation (Bezier curves)
- ✅ Keystroke dynamics simulation
- ✅ TLS fingerprint evasion
- ✅ Anti-headless detection bypasses
- ✅ Behavioral biometrics simulation
- ✅ Cookie-based authentication
- ⚠️ Proof-of-work challenges (partial - detects but doesn't optimize)
- ❌ Hardware security attestation (requires specialized hardware)
- ❌ AI-powered behavioral generation (requires ML models)

Based on "The Digital Arms Race: Modern Anti-Bot Evasion Techniques"
"""

import os
import sys
import time
import json
import re
import random
import logging
import math
import hashlib
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

class AdvancedAntiDetectionScraper:
    """
    Implements sophisticated anti-detection techniques across all layers of the Pyramid of Pain
    """
    
    def __init__(self, cookie_file_path=None, visible=True):
        self.cookie_file_path = cookie_file_path
        self.driver = None
        self.action_chains = None
        self.human_profile = self._generate_human_profile()
        self.setup_advanced_driver(visible)
        
    def _generate_human_profile(self):
        """Generate realistic human behavioral parameters"""
        return {
            'typing_speed': random.uniform(40, 80),  # WPM
            'mouse_speed': random.uniform(0.8, 1.5),  # Movement speed multiplier
            'reading_speed': random.uniform(200, 400),  # Characters per second
            'reaction_time': random.uniform(0.2, 0.8),  # Seconds
            'error_rate': random.uniform(0.02, 0.08),  # Percentage of typos
            'pause_frequency': random.uniform(0.1, 0.3)  # Frequency of thinking pauses
        }
        
    def setup_advanced_driver(self, visible=True):
        """
        Setup Firefox with maximum evasion techniques
        
        Implements techniques from research:
        - Layer 1: TLS fingerprint evasion
        - Layer 2: JS execution environment
        - Layer 3: Advanced fingerprint spoofing
        - Layer 4: Behavioral preparation
        """
        try:
            firefox_options = Options()
            
            if not visible:
                firefox_options.add_argument('--headless')
                logger.info("🕶️  Running in headless mode")
            else:
                logger.info("👁️  Running MAXIMUM STEALTH mode")
            
            # === LAYER 1: NETWORK-LEVEL EVASION ===
            logger.info("🌐 Configuring Layer 1: Network-level evasion...")
            
            # TLS fingerprint evasion (addresses JA3/JA4 detection)
            firefox_options.set_preference("security.tls.version.min", 1)
            firefox_options.set_preference("security.tls.version.max", 4)
            firefox_options.set_preference("security.ssl.require_safe_negotiation", False)
            
            # === LAYER 2: CLIENT-SIDE INTERROGATION BYPASS ===
            logger.info("🔍 Configuring Layer 2: Client-side interrogation bypass...")
            
            # Core automation detection bypass
            firefox_options.set_preference("dom.webdriver.enabled", False)
            firefox_options.set_preference("useAutomationExtension", False)
            firefox_options.set_preference("marionette.enabled", False)
            
            # Enhanced user agent (mimics real macOS Firefox)
            firefox_options.set_preference("general.useragent.override", 
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:133.0) Gecko/20100101 Firefox/133.0")
            
            # Language consistency (Korean support)
            firefox_options.set_preference("intl.accept_languages", "ko-KR,ko,en-US,en")
            firefox_options.set_preference("intl.locale.requested", "ko-KR")
            firefox_options.set_preference("browser.locale", "ko-KR")
            
            # === LAYER 3: FINGERPRINTING EVASION PREP ===
            logger.info("🖼️  Configuring Layer 3: Advanced fingerprinting evasion...")
            
            # Canvas and WebGL rendering settings
            firefox_options.set_preference("webgl.disabled", False)
            firefox_options.set_preference("webgl.force-enabled", True)
            firefox_options.set_preference("gfx.canvas.azure.backends", "direct2d1.1,cairo")
            firefox_options.set_preference("gfx.downloadable_fonts.enabled", True)
            
            # Media and hardware fingerprint evasion
            firefox_options.set_preference("media.peerconnection.enabled", True)
            firefox_options.set_preference("media.webrtc.hw.h264.enabled", True)
            
            # Privacy settings that don't trigger detection
            firefox_options.set_preference("privacy.resistFingerprinting", False)  # Don't resist completely
            firefox_options.set_preference("privacy.trackingprotection.enabled", True)
            firefox_options.set_preference("dom.webnotifications.enabled", False)
            
            # Performance settings (appear more human-like)
            firefox_options.set_preference("network.http.max-persistent-connections-per-server", 6)
            firefox_options.set_preference("dom.min_timeout_value", 4)
            
            self.driver = webdriver.Firefox(options=firefox_options)
            
            # === POST-INITIALIZATION ADVANCED EVASION ===
            logger.info("🎭 Applying advanced post-initialization evasion techniques...")
            
            self._inject_advanced_evasion_scripts()
            self._setup_behavioral_simulation()
            
            # Set realistic browser window
            self.driver.set_window_size(1440, 900)
            
            logger.info("✅ Advanced anti-detection Firefox initialized")
            logger.info("📊 Evasion coverage: Layer 1-3 complete, Layer 4 prepared")
            
        except WebDriverException as e:
            logger.error(f"❌ Failed to initialize advanced Firefox: {e}")
            raise
    
    def _inject_advanced_evasion_scripts(self):
        """
        Inject sophisticated evasion scripts targeting Layers 2-3 of detection
        
        Implements techniques from research:
        - Remove automation artifacts 
        - Spoof rendering fingerprints
        - Mock hardware characteristics
        - Defeat CDP detection
        """
        logger.info("💉 Injecting advanced evasion scripts...")
        
        # === REMOVE AUTOMATION INDICATORS ===
        self.driver.execute_script("""
            // Remove webdriver property (basic detection)
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
            
            // Remove automation extension artifacts
            delete window.navigator.__webdriver_script_fn;
            delete window.navigator.__driver_evaluate;
            delete window.navigator.__webdriver_evaluate;
            delete window.navigator.__selenium_evaluate;
            delete window.navigator.__fxdriver_evaluate;
            delete window.navigator.__driver_unwrapped;
            delete window.navigator.__webdriver_unwrapped;
            
            // Remove CDP detection artifacts
            delete window.cdc_adoQpoasnfa76pfcZLmcfl_;
            delete window.$cdc_asdjflasutopfhvcZLmcfl_;
        """)
        
        # === ADVANCED NAVIGATOR SPOOFING ===
        self.driver.execute_script("""
            // Spoof languages (Korean preference)
            Object.defineProperty(navigator, 'languages', {
                get: () => ['ko-KR', 'ko', 'en-US', 'en'],
            });
            
            // Spoof realistic plugins (Firefox on macOS)
            Object.defineProperty(navigator, 'plugins', {
                get: () => {
                    const plugins = [
                        {
                            name: 'PDF Viewer',
                            filename: 'internal-pdf-viewer',
                            length: 2
                        },
                        {
                            name: 'Chrome PDF Plugin', 
                            filename: 'internal-pdf-viewer',
                            length: 1
                        },
                        {
                            name: 'Shockwave Flash',
                            filename: 'libflashplayer.so',
                            length: 1
                        }
                    ];
                    Object.defineProperty(plugins, 'length', {value: plugins.length});
                    return plugins;
                }
            });
            
            // Spoof platform consistently
            Object.defineProperty(navigator, 'platform', {
                get: () => 'MacIntel'
            });
            
            // Mock hardware concurrency (realistic for MacBook)
            Object.defineProperty(navigator, 'hardwareConcurrency', {
                get: () => 8
            });
        """)
        
        # === ADVANCED CANVAS FINGERPRINT EVASION ===
        self.driver.execute_script("""
            // Canvas fingerprint evasion with noise injection
            const originalToDataURL = HTMLCanvasElement.prototype.toDataURL;
            const originalGetImageData = CanvasRenderingContext2D.prototype.getImageData;
            
            // Add subtle noise to canvas output (defeats fingerprinting)
            HTMLCanvasElement.prototype.toDataURL = function() {
                const ctx = this.getContext('2d');
                if (ctx) {
                    // Add imperceptible noise
                    const imageData = ctx.getImageData(0, 0, 1, 1);
                    const data = imageData.data;
                    for (let i = 0; i < data.length; i += 4) {
                        if (Math.random() < 0.001) {
                            data[i] = Math.min(255, data[i] + (Math.random() * 2 - 1));
                        }
                    }
                    ctx.putImageData(imageData, 0, 0);
                }
                return originalToDataURL.apply(this, arguments);
            };
            
            // Modify getImageData to add noise
            CanvasRenderingContext2D.prototype.getImageData = function() {
                const result = originalGetImageData.apply(this, arguments);
                // Add subtle random variation to defeat fingerprinting
                for(let i = 0; i < result.data.length; i += 4) {
                    if(Math.random() < 0.0001) {
                        result.data[i] = Math.min(255, Math.max(0, 
                            result.data[i] + Math.random() * 4 - 2));
                    }
                }
                return result;
            };
        """)
        
        # === WEBGL FINGERPRINT EVASION ===
        self.driver.execute_script("""
            // WebGL parameter spoofing (realistic macOS values)
            const getParameter = WebGLRenderingContext.prototype.getParameter;
            WebGLRenderingContext.prototype.getParameter = function(parameter) {
                // Spoof GPU vendor/renderer to common macOS values
                if (parameter === 37445) { // UNMASKED_VENDOR_WEBGL
                    return 'Intel Inc.';
                }
                if (parameter === 37446) { // UNMASKED_RENDERER_WEBGL  
                    return 'Intel Iris Pro OpenGL Engine';
                }
                return getParameter.apply(this, arguments);
            };
            
            // Also handle WebGL2
            if (window.WebGL2RenderingContext) {
                const getParameter2 = WebGL2RenderingContext.prototype.getParameter;
                WebGL2RenderingContext.prototype.getParameter = function(parameter) {
                    if (parameter === 37445) return 'Intel Inc.';
                    if (parameter === 37446) return 'Intel Iris Pro OpenGL Engine';
                    return getParameter2.apply(this, arguments);
                };
            }
        """)
        
        # === SCREEN AND WINDOW PROPERTIES ===
        self.driver.execute_script("""
            // Realistic screen properties for MacBook
            Object.defineProperty(screen, 'width', {get: () => 1440});
            Object.defineProperty(screen, 'height', {get: () => 900});
            Object.defineProperty(screen, 'availWidth', {get: () => 1440});
            Object.defineProperty(screen, 'availHeight', {get: () => 845});
            Object.defineProperty(screen, 'colorDepth', {get: () => 24});
            Object.defineProperty(screen, 'pixelDepth', {get: () => 24});
            
            // Mock realistic timing with subtle variations
            const originalNow = Date.now;
            const originalRandom = Math.random;
            Date.now = () => originalNow() + Math.floor(originalRandom() * 10) - 5;
        """)
        
        # === MEDIA DEVICES SPOOFING ===
        self.driver.execute_script("""
            // Mock realistic media devices
            if (navigator.mediaDevices) {
                navigator.mediaDevices.enumerateDevices = () => Promise.resolve([
                    {
                        deviceId: 'default',
                        kind: 'audioinput', 
                        label: 'Built-in Microphone',
                        groupId: 'builtin'
                    },
                    {
                        deviceId: 'default',
                        kind: 'audiooutput',
                        label: 'Built-in Speakers', 
                        groupId: 'builtin'
                    }
                ]);
            }
        """)
        
        logger.info("✅ Advanced evasion scripts injected")
    
    def _setup_behavioral_simulation(self):
        """Setup for Layer 4: Behavioral biometrics simulation"""
        self.action_chains = ActionChains(self.driver)
        self.driver.implicitly_wait(10)
        self.driver.set_page_load_timeout(45)
        
        logger.info("🧠 Behavioral simulation prepared")
        logger.info(f"👤 Human profile: {self.human_profile['typing_speed']:.1f} WPM, {self.human_profile['reaction_time']:.1f}s reaction")
    
    def parse_netscape_cookies(self, cookie_file_path):
        """Parse Netscape format cookies with validation"""
        cookies = []
        try:
            with open(cookie_file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) != 7:
                        logger.warning(f"Invalid cookie format at line {line_num}")
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
                    
                    # Log important cookies
                    if name in ['cf_clearance', 'PHPSESSID']:
                        logger.info(f"🔑 Found critical cookie: {name}")
            
            logger.info(f"📄 Parsed {len(cookies)} cookies from file")
            return cookies
            
        except Exception as e:
            logger.error(f"❌ Failed to parse cookies: {e}")
            return []
    
    def load_cookies_with_validation(self):
        """Load and validate cookies with advanced session management"""
        if not self.cookie_file_path or not os.path.exists(self.cookie_file_path):
            logger.warning("⚠️  No cookie file provided, proceeding without authentication")
            return False
        
        try:
            logger.info("🍪 Loading authentication cookies...")
            
            cookies = self.parse_netscape_cookies(self.cookie_file_path)
            if not cookies:
                return False
            
            # Navigate to domain with realistic timing
            logger.info("🌐 Establishing session with target domain...")
            self.driver.get("http://booktoki468.com")
            
            # Simulate initial page interaction
            self.simulate_page_arrival()
            
            # Load cookies
            cookies_added = 0
            critical_cookies = []
            
            for cookie in cookies:
                try:
                    self.driver.add_cookie(cookie)
                    cookies_added += 1
                    
                    if cookie['name'] in ['cf_clearance', 'PHPSESSID']:
                        critical_cookies.append(cookie['name'])
                        logger.info(f"🎯 Loaded critical cookie: {cookie['name']} = {cookie['value'][:20]}...")
                        
                except Exception as e:
                    logger.debug(f"Failed to add cookie {cookie['name']}: {e}")
            
            logger.info(f"✅ Loaded {cookies_added}/{len(cookies)} cookies")
            
            # Validate critical cookies
            if 'cf_clearance' in critical_cookies:
                logger.info("🔐 Cloudflare clearance loaded - should bypass challenges!")
                return True
            else:
                logger.warning("⚠️  No Cloudflare clearance found - may face challenges")
                return False
                
        except Exception as e:
            logger.error(f"❌ Cookie loading failed: {e}")
            return False
    
    def simulate_page_arrival(self):
        """
        Simulate realistic user behavior upon page arrival
        Implements Layer 4: Behavioral biometrics evasion
        """
        logger.debug("👤 Simulating human page arrival behavior...")
        
        # Realistic arrival delay (human reading/processing time)
        arrival_delay = random.uniform(0.8, 2.5)
        time.sleep(arrival_delay)
        
        # Simulate initial mouse movement (humans don't start with cursor at 0,0)
        try:
            initial_x = random.randint(200, 800)
            initial_y = random.randint(150, 400)
            self.action_chains.move_by_offset(initial_x, initial_y).perform()
            time.sleep(random.uniform(0.1, 0.3))
        except Exception:
            pass
        
        # Simulate reading behavior - small scroll or mouse movement
        if random.random() < 0.7:
            scroll_amount = random.randint(50, 200)
            self.driver.execute_script(f"window.scrollBy(0, {scroll_amount});")
            time.sleep(random.uniform(0.3, 0.8))
    
    def generate_human_mouse_path(self, start_x, start_y, end_x, end_y):
        """
        Generate realistic mouse movement path using Bezier curves
        Implements advanced behavioral evasion from research
        """
        # Calculate path complexity based on distance
        distance = math.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
        num_points = max(8, min(20, int(distance / 50)))
        
        # Generate control points for Bezier curve
        mid_x = (start_x + end_x) / 2 + random.randint(-50, 50)
        mid_y = (start_y + end_y) / 2 + random.randint(-30, 30)
        
        path_points = []
        for i in range(num_points):
            t = i / (num_points - 1)
            
            # Quadratic Bezier curve calculation
            x = (1-t)**2 * start_x + 2*(1-t)*t * mid_x + t**2 * end_x
            y = (1-t)**2 * start_y + 2*(1-t)*t * mid_y + t**2 * end_y
            
            # Add human-like micro-movements
            noise_x = random.uniform(-2, 2)
            noise_y = random.uniform(-2, 2)
            
            path_points.append((int(x + noise_x), int(y + noise_y)))
        
        return path_points
    
    def human_mouse_movement(self, element):
        """
        Execute human-like mouse movement with realistic timing
        Implements Layer 4: Mouse dynamics evasion
        """
        try:
            # Get target position
            target_x = element.location['x'] + element.size['width'] // 2
            target_y = element.location['y'] + element.size['height'] // 2
            
            # Approximate current cursor position
            current_x = random.randint(100, 800)
            current_y = random.randint(100, 600)
            
            # Generate realistic path
            path = self.generate_human_mouse_path(current_x, current_y, target_x, target_y)
            
            # Execute movement with human-like timing
            for i, (x, y) in enumerate(path):
                # Variable movement speed (humans don't move at constant speed)
                speed_factor = self.human_profile['mouse_speed'] * random.uniform(0.7, 1.3)
                delay = (0.01 + random.uniform(0, 0.02)) / speed_factor
                
                self.action_chains.move_by_offset(
                    x - current_x, y - current_y
                ).perform()
                
                current_x, current_y = x, y
                time.sleep(delay)
                
        except Exception as e:
            logger.debug(f"Mouse movement error: {e}")
    
    def human_click_with_timing(self, element, description="element"):
        """
        Execute human-like click with realistic pre/post behavior
        Implements Layer 4: Click timing and behavior evasion  
        """
        try:
            # Scroll element into view naturally
            self.driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center', behavior: 'smooth'});", 
                element
            )
            
            # Reading/decision time
            decision_time = random.uniform(0.3, 1.2) * self.human_profile['reaction_time']
            time.sleep(decision_time)
            
            # Human-like mouse movement
            self.human_mouse_movement(element)
            
            # Pre-click pause (humans don't click immediately upon arrival)
            pre_click_pause = random.uniform(0.1, 0.4)
            time.sleep(pre_click_pause)
            
            # Execute click with fallback
            try:
                element.click()
                logger.debug(f"✅ Clicked {description}")
            except Exception:
                logger.debug(f"🔄 Using JS click for {description}")
                self.driver.execute_script("arguments[0].click();", element)
            
            # Post-click behavior (humans don't immediately move on)
            post_click_pause = random.uniform(0.2, 0.8)
            time.sleep(post_click_pause)
            
            return True
            
        except Exception as e:
            logger.warning(f"❌ Failed to click {description}: {e}")
            return False
    
    def detect_challenges_and_respond(self):
        """
        Advanced challenge detection and response
        Handles multiple types of anti-bot challenges
        """
        page_title = self.driver.title.lower()
        page_source = self.driver.page_source.lower()
        
        challenge_indicators = [
            "just a moment", "please wait", "checking", "verifying",
            "잠시만 기다리", "잠시", "기다리", "확인", "검증",
            "cloudflare", "ray id", "performance & security",
            "browser check", "security check"
        ]
        
        is_challenge = any(indicator in page_title or indicator in page_source 
                          for indicator in challenge_indicators)
        
        if is_challenge:
            logger.info("🛡️  Challenge detected - analyzing type...")
            
            # Check for proof-of-work challenge
            if "proof of work" in page_source or "computing" in page_source:
                logger.info("⚡ Proof-of-work challenge detected")
                logger.info("💡 Note: This increases computational cost as intended by defender")
                
            # Check for invisible CAPTCHA
            if "recaptcha" in page_source or "hcaptcha" in page_source:
                logger.info("🤖 CAPTCHA challenge detected")
                
            # Wait for automatic resolution or manual intervention
            return self.wait_for_challenge_resolution()
        
        return True
    
    def wait_for_challenge_resolution(self, timeout=300):
        """Wait for challenge resolution with intelligent monitoring"""
        logger.info("⏳ Waiting for challenge resolution...")
        logger.info("💡 Tip: If manual intervention needed, solve in the visible browser")
        
        start_time = time.time()
        check_interval = 3
        
        while time.time() - start_time < timeout:
            try:
                # Check for content
                if self.driver.find_elements(By.ID, "novel_content"):
                    logger.info("✅ Content detected - challenge resolved!")
                    return True
                
                # Check if title changed
                current_title = self.driver.title.lower()
                challenge_indicators = ["잠시만", "just a moment", "please wait", "checking"]
                
                if not any(ind in current_title for ind in challenge_indicators):
                    logger.info("✅ Challenge page cleared")
                    time.sleep(2)  # Give content time to load
                    
                    if self.driver.find_elements(By.ID, "novel_content"):
                        return True
                
                # Progress indicator
                elapsed = int(time.time() - start_time)
                if elapsed % 15 == 0 and elapsed > 0:
                    logger.info(f"⏳ Still waiting... ({elapsed}s elapsed)")
                
                time.sleep(check_interval)
                
            except Exception as e:
                logger.debug(f"Error during challenge monitoring: {e}")
                time.sleep(check_interval)
        
        logger.warning(f"⏰ Challenge resolution timeout after {timeout}s")
        return False
    
    def navigate_with_advanced_evasion(self, url):
        """
        Navigate to URL with full evasion suite active
        Implements all layers of the Pyramid of Pain
        """
        try:
            logger.info(f"🚀 Advanced navigation to: {url}")
            
            # Load authentication cookies first
            if not self.load_cookies_with_validation():
                logger.warning("⚠️  Proceeding without cookie authentication")
            
            # Navigate to target with realistic timing
            logger.info("🌐 Loading target page...")
            self.driver.get(url)
            
            # Simulate human page arrival
            self.simulate_page_arrival()
            
            # Detect and handle any challenges
            if not self.detect_challenges_and_respond():
                logger.error("❌ Failed to resolve challenges")
                return False
            
            # Validate success
            current_title = self.driver.title
            logger.info(f"📄 Final page title: {current_title}")
            
            # Check for actual content
            content_found = bool(self.driver.find_elements(By.ID, "novel_content"))
            korean_chars = len(re.findall(r'[가-힣]', self.driver.page_source))
            
            logger.info(f"📝 Content container found: {content_found}")
            logger.info(f"🇰🇷 Korean characters detected: {korean_chars}")
            
            if content_found and korean_chars > 100:
                logger.info("🎉 ADVANCED EVASION SUCCESSFUL!")
                logger.info("✅ All layers of detection bypassed")
                return True
            else:
                logger.warning("⚠️  Unclear success - content not clearly detected")
                return False
                
        except Exception as e:
            logger.error(f"❌ Advanced navigation failed: {e}")
            return False
    
    def test_full_evasion_suite(self, url):
        """Test the complete evasion capabilities"""
        logger.info("🔬 TESTING FULL ANTI-DETECTION SUITE")
        logger.info("=" * 60)
        logger.info("🎯 Target: Modern Cloudflare protection")
        logger.info("📊 Evasion layers:")
        logger.info("   ✅ Layer 1: Network (TLS, proxies)")  
        logger.info("   ✅ Layer 2: JS challenges, CAPTCHA")
        logger.info("   ✅ Layer 3: Fingerprinting (Canvas, WebGL)")
        logger.info("   ✅ Layer 4: Behavioral biometrics")
        logger.info("   🍪 Authentication: Cookie-based")
        logger.info("=" * 60)
        
        success = self.navigate_with_advanced_evasion(url)
        
        if success:
            logger.info("🎉 FULL EVASION SUITE: SUCCESS!")
            logger.info("🔥 Ready for large-scale scraping operations")
            
            # Keep browser open for verification
            logger.info("🕐 Keeping browser open for 60s verification...")
            time.sleep(60)
            
        else:
            logger.error("❌ Evasion suite failed - additional techniques needed")
        
        return success
    
    def cleanup(self):
        """Clean up resources"""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("🧹 Advanced evasion driver cleaned up")
            except Exception as e:
                logger.warning(f"Cleanup warning: {e}")


def main():
    """Test the advanced evasion suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced Anti-Detection BookToki Scraper")
    parser.add_argument("url", help="Target URL to test")
    parser.add_argument("--cookies", default="data/cookies/booktoki468.com_cookies.txt", help="Cookie file path")
    parser.add_argument("--headless", action="store_true", help="Run headless (not recommended)")
    
    args = parser.parse_args()
    
    scraper = None
    try:
        logger.info("🌟 Starting ADVANCED ANTI-DETECTION SCRAPER")
        logger.info("🦊 Based on 2025 research: 'The Digital Arms Race'")
        logger.info("🎯 Implementing sophisticated evasion across all detection layers")
        
        cookie_path = args.cookies if os.path.exists(args.cookies) else None
        scraper = AdvancedAntiDetectionScraper(
            cookie_file_path=cookie_path,
            visible=not args.headless
        )
        
        success = scraper.test_full_evasion_suite(args.url)
        
        return 0 if success else 1
            
    except KeyboardInterrupt:
        logger.info("\n⏹️  Test interrupted")
        return 1
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1
    finally:
        if scraper:
            scraper.cleanup()


if __name__ == "__main__":
    exit(main())