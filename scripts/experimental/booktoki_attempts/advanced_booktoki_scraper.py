#!/usr/bin/env python3
"""
Advanced BookToki Scraper that mimics browser behavior

This scraper handles anti-bot protection by using browser automation
and user-agent rotation to successfully scrape Korean web novels.
"""

import os
import sys
import time
import json
import re
import logging
from urllib.parse import urljoin, urlparse, parse_qs
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdvancedBookTokiScraper:
    def __init__(self, delay=3):
        self.delay = delay
        self.session = self._create_session()
        self.current_chapter_id = None
        
    def _create_session(self):
        """Create a requests session with browser-like behavior"""
        session = requests.Session()
        
        # Set up retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[403, 429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Browser-like headers
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0',
        })
        
        return session
    
    def _extract_chapter_id_from_url(self, url):
        """Extract chapter ID from URL"""
        match = re.search(r'/novel/(\d+)', url)
        return int(match.group(1)) if match else None
    
    def _extract_chapter_title(self, soup):
        """Extract chapter title from page"""
        # Try page title first
        title_tag = soup.find('title')
        if title_tag:
            title_text = title_tag.get_text(strip=True)
            # Extract Korean chapter title pattern
            chapter_match = re.search(r'([^_]+?-\d+화)', title_text)
            if chapter_match:
                return chapter_match.group(1)
        
        # Try other selectors
        for selector in ['h1', 'h2', '.title']:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)
        
        return None
    
    def _extract_content(self, soup):
        """Extract chapter content from page"""
        # Primary content container
        content_container = soup.select_one('#novel_content')
        
        if not content_container:
            logger.warning("Primary content container not found")
            return None
        
        # Look for content within the specific div
        content_div = content_container.select_one('div.f9e99a33513')
        if not content_div:
            content_div = content_container
        
        # Extract paragraphs
        paragraphs = content_div.find_all('p')
        if not paragraphs:
            logger.warning("No paragraphs found")
            return None
        
        # Filter and clean paragraphs
        chapter_text = []
        for p in paragraphs:
            text = p.get_text(strip=True)
            if text and self._is_valid_paragraph(text):
                chapter_text.append(text)
        
        if not chapter_text:
            return None
        
        content = '\n\n'.join(chapter_text)
        
        # Validate content
        if len(content) < 500:
            logger.warning(f"Content too short: {len(content)} chars")
            return None
        
        # Check for Korean text
        korean_chars = re.findall(r'[가-힣]', content)
        if len(korean_chars) < 100:
            logger.warning(f"Insufficient Korean text: {len(korean_chars)} Korean chars")
            return None
        
        return content
    
    def _is_valid_paragraph(self, text):
        """Check if paragraph is valid story content"""
        if not text or len(text.strip()) < 3:
            return False
        
        # Skip separator lines and metadata
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
    
    def _find_navigation_links(self, soup, current_url):
        """Find navigation links for next/previous chapters"""
        nav_links = []
        current_chapter_id = self._extract_chapter_id_from_url(current_url)
        
        # Find all novel links
        for link in soup.select('a[href*="/novel/"]'):
            href = link.get('href')
            if not href:
                continue
            
            full_url = urljoin(current_url, href)
            link_chapter_id = self._extract_chapter_id_from_url(full_url)
            link_text = link.get_text(strip=True)
            
            if link_chapter_id and link_chapter_id != current_chapter_id:
                nav_links.append({
                    'url': full_url,
                    'chapter_id': link_chapter_id,
                    'text': link_text
                })
        
        return nav_links
    
    def fetch_chapter(self, url, max_attempts=3):
        """Fetch a single chapter with retry logic"""
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"🔍 Fetching chapter (attempt {attempt}): {url}")
                
                # Add small delay to avoid rate limiting
                time.sleep(self.delay)
                
                # Fetch with session
                response = self.session.get(url, timeout=30)
                
                if response.status_code == 403:
                    logger.warning(f"⚠️  403 Forbidden - attempting workaround (attempt {attempt})")
                    
                    # Try different user agents
                    user_agents = [
                        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1.2 Safari/605.1.15',
                        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                    ]
                    
                    for ua in user_agents:
                        self.session.headers['User-Agent'] = ua
                        time.sleep(2)
                        response = self.session.get(url, timeout=30)
                        if response.status_code == 200:
                            break
                
                if response.status_code == 200:
                    # Parse content
                    soup = BeautifulSoup(response.content, 'html.parser', from_encoding='utf-8')
                    
                    # Extract data
                    chapter_title = self._extract_chapter_title(soup)
                    content = self._extract_content(soup)
                    nav_links = self._find_navigation_links(soup, url)
                    
                    if not content:
                        logger.error("❌ No valid content found")
                        continue
                    
                    # Find next/prev links
                    next_url = None
                    prev_url = None
                    
                    current_id = self._extract_chapter_id_from_url(url)
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
                        'next_url': next_url,
                        'prev_url': prev_url,
                        'nav_links_found': len(nav_links)
                    }
                    
                    logger.info(f"✅ Chapter fetched: {chapter_data['title']} ({len(content)} chars)")
                    return chapter_data
                else:
                    logger.warning(f"⚠️  HTTP {response.status_code}: {response.reason}")
                    
            except Exception as e:
                logger.error(f"❌ Attempt {attempt} failed: {e}")
                if attempt < max_attempts:
                    sleep_time = attempt * 2
                    logger.info(f"⏳ Waiting {sleep_time}s before retry...")
                    time.sleep(sleep_time)
        
        logger.error(f"❌ Failed to fetch chapter after {max_attempts} attempts")
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
            else:  # backward
                current_url = chapter_data['prev_url']
            
            if not current_url:
                logger.info("🏁 No more chapters found in sequence")
                break
        
        logger.info(f"🎉 Scraping completed! Total chapters: {len(scraped_chapters)}")
        return scraped_chapters
    
    def save_chapters(self, chapters, output_dir="data/novels/booktoki_dungeon_defense"):
        """Save scraped chapters to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metadata
        metadata = {
            'scrape_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_chapters': len(chapters),
            'chapters': []
        }
        
        for i, chapter in enumerate(chapters, 1):
            # Save individual chapter
            filename = f"chapter_{i:03d}_{chapter['chapter_id']:04d}.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# {chapter['title']}\n")
                f.write(f"# URL: {chapter['url']}\n")
                f.write(f"# Chapter ID: {chapter['chapter_id']}\n")
                f.write(f"# Word Count: {chapter['word_count']}\n\n")
                f.write(chapter['content'])
            
            logger.info(f"💾 Saved: {filepath}")
            
            # Add to metadata
            metadata['chapters'].append({
                'index': i,
                'title': chapter['title'],
                'chapter_id': chapter['chapter_id'],
                'url': chapter['url'],
                'filename': filename,
                'word_count': chapter['word_count']
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
            f.write(f"# Scraped: {metadata['scrape_date']}\n\n")
            
            for chapter in chapters:
                f.write(f"\n{'='*60}\n")
                f.write(f"{chapter['title']}\n")
                f.write(f"{'='*60}\n\n")
                f.write(chapter['content'])
                f.write("\n\n")
        
        logger.info(f"📁 Output directory: {output_dir}")
        logger.info(f"📄 Combined file: {combined_file}")
        logger.info(f"📋 Metadata file: {metadata_file}")
        
        return {
            'output_dir': output_dir,
            'combined_file': combined_file,
            'metadata_file': metadata_file,
            'chapters_saved': len(chapters)
        }


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced BookToki scraper for Korean web novels")
    parser.add_argument("url", help="Starting chapter URL")
    parser.add_argument("--max-chapters", type=int, default=10, help="Maximum chapters to scrape")
    parser.add_argument("--delay", type=float, default=3, help="Delay between requests (seconds)")
    parser.add_argument("--direction", choices=["forward", "backward"], default="forward", help="Scraping direction")
    parser.add_argument("--output", default="data/novels/booktoki_dungeon_defense", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        scraper = AdvancedBookTokiScraper(delay=args.delay)
        chapters = scraper.scrape_sequence(
            args.url, 
            max_chapters=args.max_chapters, 
            direction=args.direction
        )
        
        if chapters:
            result = scraper.save_chapters(chapters, args.output)
            logger.info(f"🎉 Success! {result['chapters_saved']} chapters saved to {result['output_dir']}")
        else:
            logger.error("❌ No chapters were scraped")
            return 1
            
    except KeyboardInterrupt:
        logger.info("\n⏹️  Scraping interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())