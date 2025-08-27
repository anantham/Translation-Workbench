#!/usr/bin/env python3
"""
BookToki Korean Novel Scraper

Scrapes Korean web novels from booktoki468.com
Handles dynamic content loading and chapter navigation
"""

import os
import re
import time
import json
import requests
from urllib.parse import urljoin, urlparse, parse_qs
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException


class BookTokiScraper:
    def __init__(self, base_url="https://booktoki468.com", delay=2):
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()
        
        # Set up user agent to mimic real browser
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        self.driver = None
        self.setup_selenium()
    
    def setup_selenium(self):
        """Set up Chrome WebDriver for dynamic content"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # Run in background
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            print("✅ Chrome WebDriver initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize Chrome WebDriver: {e}")
            print("💡 Please install ChromeDriver or use Selenium with different browser")
            raise
    
    def extract_novel_info(self, novel_url):
        """Extract basic novel information from the URL"""
        try:
            # Extract novel ID from URL
            novel_id_match = re.search(r'/novel/(\d+)', novel_url)
            if not novel_id_match:
                raise ValueError("Could not extract novel ID from URL")
            
            novel_id = novel_id_match.group(1)
            
            # Extract search parameters
            parsed_url = urlparse(novel_url)
            query_params = parse_qs(parsed_url.query)
            
            novel_info = {
                'id': novel_id,
                'base_url': novel_url,
                'search_params': query_params
            }
            
            print(f"📖 Extracted novel info: ID={novel_id}")
            return novel_info
            
        except Exception as e:
            print(f"❌ Failed to extract novel info: {e}")
            raise
    
    def get_chapter_content(self, chapter_url):
        """Extract content from a single chapter using Selenium"""
        try:
            print(f"🔍 Loading chapter: {chapter_url}")
            
            # Load page with Selenium
            self.driver.get(chapter_url)
            
            # Wait for content to load
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.ID, "novel_content"))
                )
            except TimeoutException:
                print("❌ Timeout waiting for content to load")
                return None
            
            # Extract chapter content
            content_element = self.driver.find_element(By.ID, "novel_content")
            
            # Get all paragraph text
            paragraphs = content_element.find_elements(By.TAG_NAME, "p")
            chapter_text = []
            
            for p in paragraphs:
                text = p.text.strip()
                if text and not text.startswith("="):  # Skip separator lines
                    chapter_text.append(text)
            
            # Extract chapter title from page title
            page_title = self.driver.title
            chapter_title_match = re.search(r'(던전 디펜스-\d+화)', page_title)
            chapter_title = chapter_title_match.group(1) if chapter_title_match else "Unknown Chapter"
            
            # Extract navigation links
            prev_link = None
            next_link = None
            
            try:
                # Look for previous chapter link
                prev_elements = self.driver.find_elements(By.XPATH, "//a[contains(text(), '이전화') or contains(@href, 'novel/')]")
                for elem in prev_elements:
                    href = elem.get_attribute('href')
                    if href and '/novel/' in href and href != chapter_url:
                        prev_link = href
                        break
                
                # Look for next chapter link
                next_elements = self.driver.find_elements(By.XPATH, "//a[contains(text(), '다음화') or contains(text(), '화')]")
                for elem in next_elements:
                    href = elem.get_attribute('href')
                    if href and '/novel/' in href and href != chapter_url:
                        next_link = href
                        break
                        
            except NoSuchElementException:
                pass
            
            chapter_data = {
                'title': chapter_title,
                'url': chapter_url,
                'content': chapter_text,
                'word_count': len(' '.join(chapter_text)),
                'prev_chapter': prev_link,
                'next_chapter': next_link
            }
            
            print(f"✅ Extracted chapter: {chapter_title} ({len(chapter_text)} paragraphs, {chapter_data['word_count']} chars)")
            return chapter_data
            
        except Exception as e:
            print(f"❌ Failed to extract chapter content: {e}")
            return None
    
    def discover_all_chapters(self, start_url, max_chapters=None):
        """Discover all available chapters by following navigation links"""
        print(f"🔍 Starting chapter discovery from: {start_url}")
        
        discovered_urls = []
        visited = set()
        to_visit = [start_url]
        
        # First, try to go to chapter 1 if we're not already there
        current_chapter = self.get_chapter_content(start_url)
        if current_chapter and current_chapter['prev_chapter']:
            # Keep going back until we find chapter 1
            temp_url = current_chapter['prev_chapter']
            attempts = 0
            while temp_url and temp_url not in visited and attempts < 50:
                temp_data = self.get_chapter_content(temp_url)
                if not temp_data:
                    break
                to_visit.insert(0, temp_url)  # Add to beginning
                visited.add(temp_url)
                temp_url = temp_data['prev_chapter']
                attempts += 1
                time.sleep(self.delay)
        
        # Now discover all chapters going forward
        current_url = to_visit[0] if to_visit else start_url
        chapter_count = 0
        
        while current_url and current_url not in visited:
            if max_chapters and chapter_count >= max_chapters:
                break
                
            visited.add(current_url)
            discovered_urls.append(current_url)
            
            chapter_data = self.get_chapter_content(current_url)
            if not chapter_data:
                break
                
            print(f"📑 Discovered: {chapter_data['title']}")
            
            # Move to next chapter
            current_url = chapter_data['next_chapter']
            chapter_count += 1
            
            # Rate limiting
            time.sleep(self.delay)
        
        print(f"🎉 Discovery complete! Found {len(discovered_urls)} chapters")
        return discovered_urls
    
    def scrape_novel(self, novel_url, output_dir="data/novels/booktoki", max_chapters=None):
        """Main method to scrape entire novel"""
        try:
            print(f"🚀 Starting scrape of novel: {novel_url}")
            
            # Extract novel info
            novel_info = self.extract_novel_info(novel_url)
            
            # Create output directory
            novel_name = f"dungeon_defense_{novel_info['id']}"
            full_output_dir = os.path.join(output_dir, novel_name)
            os.makedirs(full_output_dir, exist_ok=True)
            
            # Discover all chapters
            chapter_urls = self.discover_all_chapters(novel_url, max_chapters)
            
            if not chapter_urls:
                print("❌ No chapters found!")
                return
            
            # Scrape each chapter
            scraped_chapters = []
            
            for i, chapter_url in enumerate(chapter_urls, 1):
                print(f"\n📖 Scraping chapter {i}/{len(chapter_urls)}")
                
                chapter_data = self.get_chapter_content(chapter_url)
                if chapter_data:
                    scraped_chapters.append(chapter_data)
                    
                    # Save individual chapter file
                    chapter_filename = f"chapter_{i:03d}_{chapter_data['title'].replace(' ', '_')}.txt"
                    chapter_filepath = os.path.join(full_output_dir, chapter_filename)
                    
                    with open(chapter_filepath, 'w', encoding='utf-8') as f:
                        f.write(f"# {chapter_data['title']}\n")
                        f.write(f"# URL: {chapter_data['url']}\n")
                        f.write(f"# Word Count: {chapter_data['word_count']}\n\n")
                        for paragraph in chapter_data['content']:
                            f.write(f"{paragraph}\n\n")
                    
                    print(f"💾 Saved: {chapter_filepath}")
                else:
                    print(f"❌ Failed to scrape chapter {i}")
                
                # Rate limiting
                time.sleep(self.delay)
            
            # Save metadata
            metadata = {
                'novel_info': novel_info,
                'total_chapters': len(scraped_chapters),
                'scrape_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'chapters': [
                    {
                        'index': i + 1,
                        'title': chapter['title'],
                        'url': chapter['url'],
                        'word_count': chapter['word_count']
                    }
                    for i, chapter in enumerate(scraped_chapters)
                ]
            }
            
            metadata_file = os.path.join(full_output_dir, 'metadata.json')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # Create combined file
            combined_file = os.path.join(full_output_dir, f"{novel_name}_complete.txt")
            with open(combined_file, 'w', encoding='utf-8') as f:
                f.write(f"# Dungeon Defense - Complete Novel\n")
                f.write(f"# Total Chapters: {len(scraped_chapters)}\n")
                f.write(f"# Scraped: {metadata['scrape_date']}\n\n")
                
                for i, chapter in enumerate(scraped_chapters, 1):
                    f.write(f"\n{'='*50}\n")
                    f.write(f"Chapter {i}: {chapter['title']}\n")
                    f.write(f"{'='*50}\n\n")
                    
                    for paragraph in chapter['content']:
                        f.write(f"{paragraph}\n\n")
            
            print(f"\n🎉 Scraping complete!")
            print(f"📊 Total chapters: {len(scraped_chapters)}")
            print(f"💾 Output directory: {full_output_dir}")
            print(f"📄 Combined file: {combined_file}")
            print(f"📋 Metadata: {metadata_file}")
            
            return scraped_chapters
            
        except Exception as e:
            print(f"❌ Scraping failed: {e}")
            raise
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.driver:
            self.driver.quit()
            print("🧹 WebDriver cleaned up")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Scrape Korean novels from BookToki")
    parser.add_argument("url", help="Novel URL to scrape")
    parser.add_argument("--max-chapters", type=int, help="Maximum number of chapters to scrape")
    parser.add_argument("--delay", type=float, default=2, help="Delay between requests (seconds)")
    parser.add_argument("--output", default="data/novels/booktoki", help="Output directory")
    
    args = parser.parse_args()
    
    # Validate URL
    if 'booktoki' not in args.url or '/novel/' not in args.url:
        print("❌ Invalid URL. Must be a BookToki novel URL containing '/novel/'")
        return
    
    try:
        scraper = BookTokiScraper(delay=args.delay)
        scraper.scrape_novel(args.url, args.output, args.max_chapters)
        
    except KeyboardInterrupt:
        print("\n⏹️  Scraping interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())