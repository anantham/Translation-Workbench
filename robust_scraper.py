import os
import re
import time
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def sanitize_filename(filename):
    """Removes invalid characters from a string so it can be used as a filename."""
    return re.sub(r'[\\/*?:"<>|]', "", filename).strip()

def chinese_to_int(text):
    """Converts a Chinese numeral string to an integer with error handling."""
    # Handle malformed titles like "第一前一百一十一章" - remove problematic characters
    text = text.replace('前', '')  # Remove the problematic "前" character
    
    num_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '零': 0}
    unit_map = {'十': 10, '百': 100, '千': 1000, '万': 10000}
    
    total = 0
    temp_num = 0
    for char in text:
        if char in num_map:
            temp_num = num_map[char]
        elif char in unit_map:
            if temp_num == 0: 
                temp_num = 1
            total += temp_num * unit_map[char]
            temp_num = 0
    total += temp_num
    
    # Handle special case for numbers like 十三 (13)
    if text.startswith('十'):
        total += 10
        
    return total

def extract_chapter_number(title):
    """Extracts numeric chapter number from a Chinese title with robust error handling."""
    match = re.search(r'第(.+?)章', title)
    if match:
        ch_text = match.group(1).strip()
        
        # Handle direct numeric cases
        if ch_text.isdigit():
            return int(ch_text)
        
        # Handle malformed cases like "一前一百一十一" -> should be "一千一百一十一" (1111)
        if '前' in ch_text:
            print(f"    [WARNING] Malformed title detected: '{title}' - attempting to fix")
            # Replace common malformations
            ch_text = ch_text.replace('一前一百一十一', '一千一百一十一')
            ch_text = ch_text.replace('前', '')
        
        try:
            return chinese_to_int(ch_text)
        except Exception as e:
            print(f"    [DEBUG] Chinese to int conversion failed for '{ch_text}': {e}")
            return None
    return None

def validate_page_content(soup):
    """Validate that the page contains required content elements."""
    title_element = soup.select_one("#ChapterTitle")
    content_element = soup.select_one("#Lab_Contents")
    
    if not title_element:
        return False, "Missing #ChapterTitle element"
    if not content_element:
        return False, "Missing #Lab_Contents element"
    if not title_element.text.strip():
        return False, "Empty chapter title"
    if len(content_element.get_text(strip=True)) < 50:
        return False, "Content too short"
    
    return True, "Content validated"

def fetch_with_retry(url, headers, max_attempts=3):
    """Fetch URL with retry logic and content validation."""
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"    [FETCH] Attempt {attempt}/{max_attempts}")
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Validate content
            is_valid, validation_msg = validate_page_content(soup)
            if is_valid:
                print(f"    [SUCCESS] {validation_msg}")
                return soup, response
            else:
                print(f"    [WARNING] {validation_msg} - Page may be blocked")
                
        except requests.exceptions.RequestException as e:
            print(f"    [ERROR] Request failed: {e}")
        
        # Wait before retry (exponential backoff)
        if attempt < max_attempts:
            wait_time = 5 * attempt
            print(f"    [BACKOFF] Waiting {wait_time}s before retry...")
            time.sleep(wait_time)
    
    return None, None

def load_or_create_metadata(metadata_file):
    """Load existing metadata or create new structure."""
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"chapters": {}, "urls": {}}

def save_metadata(metadata, metadata_file):
    """Save metadata to file."""
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def scrape_backwards_robust(start_url: str, output_dir: str):
    """Scrapes all chapters backwards with robust error handling for malformed titles."""
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load/create metadata
    metadata_file = "scraping_metadata.json"
    metadata = load_or_create_metadata(metadata_file)
    
    current_url = start_url
    visited_urls = set()
    last_scraped_chapter_num = None
    chapters_saved = 0
    metadata_updates = 0

    print(f"🚀 Starting robust backward scrape from: {start_url}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📋 Metadata file: {metadata_file}")
    print(f"🛡️  Enhanced with malformed title handling")
    print("=" * 70)

    while current_url:
        if current_url in visited_urls:
            print(f"❌ Loop detected at {current_url}. Stopping.")
            break
        visited_urls.add(current_url)
        
        # QUICK CHECK: If URL already in metadata and file exists, use stored navigation
        if current_url in metadata["urls"]:
            chapter_num_from_metadata = metadata["urls"][current_url]
            chapter_info = metadata["chapters"].get(str(chapter_num_from_metadata))
            
            if chapter_info and chapter_info.get("file_exists"):
                print(f"  -> [ULTRA SKIP] Chapter {chapter_num_from_metadata} already processed")
                last_scraped_chapter_num = chapter_num_from_metadata
                
                # Use stored navigation URL if available
                stored_previous_url = chapter_info.get("previous_url")
                if stored_previous_url:
                    print(f"  -> [NAV CACHED] Using stored navigation URL")
                    current_url = stored_previous_url
                    continue  # Skip all processing for this chapter
                else:
                    print(f"  -> [NAV FETCH] No cached navigation, fetching for navigation only")
                    fetch_for_navigation = True
            else:
                fetch_for_navigation = False
        else:
            fetch_for_navigation = False
        
        # Use retry mechanism for all fetches
        if fetch_for_navigation:
            print(f"  -> [NAV ONLY] Fetching for navigation only")
            current_chapter_num = chapter_num_from_metadata
            title = chapter_info["title"]
            # For navigation, use simpler fetch without full content validation
            try:
                response = requests.get(current_url, headers=headers, timeout=20)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
            except Exception as e:
                print(f"    [NAV ERROR] Failed to fetch navigation: {e}")
                current_url = None
                break
        else:
            print(f"  -> Fetching: {current_url}")
            soup, response = fetch_with_retry(current_url, headers)
            
            if not soup:
                print(f"    [FATAL] Failed to fetch valid content after retries. Skipping chapter.")
                # Try to continue with next chapter if possible
                current_url = None
                break
            
        if not fetch_for_navigation:
            title_tag = soup.select_one("#ChapterTitle")
            title = title_tag.text.strip() if title_tag else "Unknown_Chapter"
            
            current_chapter_num = extract_chapter_number(title)
            
            # ROBUST HANDLING: If we can't parse the number, estimate it
            if not current_chapter_num:
                if last_scraped_chapter_num:
                    estimated_num = last_scraped_chapter_num - 1
                    print(f"    [ESTIMATED] Could not parse '{title}', estimating as Chapter {estimated_num}")
                    current_chapter_num = estimated_num
                else:
                    print(f"    [ERROR] Could not parse chapter number from: '{title}' and no previous chapter for estimation")
                    current_url = None
                    break
            
            # UPDATE METADATA (always, even if skipping)
            metadata["chapters"][str(current_chapter_num)] = {
                "title": title,
                "url": current_url,
                "scraped": False,
                "file_exists": False,
                "previous_url": None  # Will be populated when we find the navigation link
            }
            metadata["urls"][current_url] = current_chapter_num
            metadata_updates += 1

            # VALIDATION: Check sequence (but allow small gaps for malformed chapters)
            if last_scraped_chapter_num is not None:
                expected_chapter_num = last_scraped_chapter_num - 1
                
                if current_chapter_num == expected_chapter_num:
                    print(f"    [SEQ OK] Found Chapter {current_chapter_num} as expected")
                elif abs(current_chapter_num - expected_chapter_num) <= 5:  # Allow small discrepancies
                    print(f"    [SEQ WARN] Found Chapter {current_chapter_num}, expected {expected_chapter_num} (minor gap)")
                else:
                    print(f"    [SEQ ERROR] Found Chapter {current_chapter_num}, expected {expected_chapter_num} (major gap)")
                    print(f"    [DECISION] Continuing anyway due to robust mode...")
            
            content_tag = soup.select_one("#Lab_Contents")
            if content_tag:
                filename = sanitize_filename(f"Chapter-{current_chapter_num:04d}-{title}.txt")
                filepath = os.path.join(output_dir, filename)
                
                # CHECK IF FILE ALREADY EXISTS
                if os.path.exists(filepath):
                    print(f"    [SKIP] Chapter {current_chapter_num} already exists: {filename}")
                    metadata["chapters"][str(current_chapter_num)]["scraped"] = True
                    metadata["chapters"][str(current_chapter_num)]["file_exists"] = True
                else:
                    # Clean up unwanted elements
                    for tag in content_tag.select('script, a'): 
                        tag.decompose()
                    content = content_tag.get_text(separator='\n', strip=True)
                    
                    # Validate content length
                    if len(content.strip()) < 50:
                        print(f"    [WARNING] Content too short ({len(content)} chars) for: {title}")
                        print(f"    [DEBUG] Content preview: {content[:100]}...")
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print(f"    [SUCCESS] Saved Chapter {current_chapter_num}: {title}")
                    chapters_saved += 1
                    
                    metadata["chapters"][str(current_chapter_num)]["scraped"] = True
                    metadata["chapters"][str(current_chapter_num)]["file_exists"] = True
                
                # Save metadata every 10 chapters
                if metadata_updates % 10 == 0:
                    save_metadata(metadata, metadata_file)
                    print(f"    [META] Saved metadata ({len(metadata['chapters'])} chapters)")
            else:
                print(f"    [ERROR] No content found for: {title}")
                print(f"    [DEBUG] Available content selectors: {[tag.name for tag in soup.find_all(['div', 'p']) if tag.get('id')]}")
            
            last_scraped_chapter_num = current_chapter_num

            if current_chapter_num <= 1:
                print(f"\n🎯 Reached Chapter {current_chapter_num}! Stopping here.")
                break
            
            # Find next chapter link
            prev_link = soup.find('a', string=re.compile(r'上一章'))
            if prev_link and prev_link.get('href') and ".html" in prev_link.get('href'):
                next_url = urljoin(current_url, prev_link['href'])
                
                # Store navigation link in metadata for future quick skips
                if str(current_chapter_num) in metadata["chapters"]:
                    metadata["chapters"][str(current_chapter_num)]["previous_url"] = next_url
                
                current_url = next_url
            else:
                print("\\n🏁 No 'previous chapter' link found. Reached end of available chapters.")
                current_url = None
        
        # Progress reporting
        if chapters_saved > 0 and chapters_saved % 50 == 0:
            print(f"\\n📊 PROGRESS: {chapters_saved} chapters saved, currently at Chapter {current_chapter_num}")
        
        time.sleep(1.5)  # Be polite to the server

    # Final metadata save
    save_metadata(metadata, metadata_file)
    
    print(f"\\n🏁 SCRAPING COMPLETED!")
    print(f"📊 Total chapters saved: {chapters_saved}")
    print(f"🗂️  Total chapters in metadata: {len(metadata['chapters'])}")
    print(f"📁 Output directory: {output_dir}")
    print(f"💾 Metadata saved to: {metadata_file}")

if __name__ == "__main__":
    # Start from the problematic Chapter 399 that was failing
    start_url = 'https://www.dxmwx.org/read/43713_17089584.html'  # Chapter 399
    output_dir = "novel_content_dxmwx_complete"
    
    print("🛡️  RESILIENT SCRAPER - Enhanced with retry logic and content validation")
    print("Starting from the previously failing Chapter 399...")
    print("This will continue until Chapter 1 or end of available chapters.\\n")
    
    scrape_backwards_robust(start_url, output_dir)