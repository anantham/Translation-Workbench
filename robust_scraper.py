import os
import re
import time
import json
import requests
import argparse
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# --- All helper functions remain the same ---
def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "", filename).strip()

def chinese_to_int(text):
    """
    Converts Chinese numerals to integers.
    Note: Does NOT handle typos here - that's done in extract_chapter_number()
    """
    text = str(text)
    
    num_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '零': 0}
    unit_map = {'十': 10, '百': 100, '千': 1000, '万': 10000}
    total = 0
    temp_num = 0
    
    # Handle standalone units at the beginning (e.g., "十" for 10)
    if text.startswith('十'):
        total = 10
        text = text[1:]

    for char in text:
        if char in num_map:
            temp_num = num_map[char]
        elif char in unit_map:
            # handle cases like "一百三" (130) vs "一十三" (13)
            if temp_num == 0: 
                temp_num = 1
            total += temp_num * unit_map[char]
            temp_num = 0
    
    total += temp_num
    return total if total > 0 else None

def extract_chapter_number(title):
    """
    Extract chapter number from title, handling website typos surgically.
    Only fixes 前→千 in the specific pattern "一前一百" to avoid corrupting legitimate titles.
    """
    match = re.search(r'第(.+?)章', title)
    if not match: 
        return None
    
    ch_text = match.group(1).strip()
    
    # Surgical fix: ONLY replace 前→千 in the specific typo pattern
    # This preserves legitimate uses of 前 in chapter titles like "前世今生"
    if '一前一' in ch_text:
        ch_text = ch_text.replace('一前一', '一千一')
    
    if ch_text.isdigit(): 
        return int(ch_text)
    return chinese_to_int(ch_text)

def validate_page_content(soup):
    if not soup or not soup.select_one("#ChapterTitle"): return False, "Missing #ChapterTitle"
    if not soup.select_one("#Lab_Contents"): return False, "Missing #Lab_Contents"
    return True, "Content validated"

def fetch_with_retry(url, headers, max_attempts=3):
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            soup = BeautifulSoup(response.text, 'html.parser')
            is_valid, _ = validate_page_content(soup)
            if is_valid: return soup
        except requests.exceptions.RequestException as e:
            print(f"    [ERROR] Request failed on attempt {attempt}: {e}")
        if attempt < max_attempts: time.sleep(5 * attempt)
    return None

def load_or_create_metadata(metadata_file):
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f: return json.load(f)
    return {"chapters": {}}

def save_metadata(metadata, metadata_file):
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def get_url_for_chapter(chapter_num, metadata):
    chapter_info = metadata.get("chapters", {}).get(str(chapter_num))
    return chapter_info.get("url") if chapter_info else None


def scrape_backwards_final(start_url: str, output_dir: str, metadata_file: str):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure temp directory exists
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
    metadata = load_or_create_metadata(metadata_file)
    
    current_url = start_url
    last_known_good_num = None
    
    # If starting mid-scrape, determine the last chapter number
    if metadata["chapters"]:
        last_known_good_num = max([int(k) for k in metadata["chapters"].keys()])
        print(f"Resuming scrape. Last known good chapter is {last_known_good_num}")

    print(f"🚀 Starting Final Scrape from: {start_url}")

    while current_url:
        print("-" * 70)
        print(f"  -> Processing URL: {current_url}")
        
        # Check if file already exists based on metadata AND verify actual file existence
        cache_hit = False
        for ch_num, chapter_info in metadata.get("chapters", {}).items():
            if chapter_info.get("url") == current_url:
                # Construct expected filepath
                title = chapter_info.get("title", f"Chapter {ch_num}")
                filename = sanitize_filename(f"Chapter-{int(ch_num):04d}-{title}.txt")
                expected_filepath = os.path.join(output_dir, filename)
                
                # Dual verification: metadata + file system
                metadata_says_exists = chapter_info.get("file_exists", False)
                file_actually_exists = os.path.exists(expected_filepath)
                
                if metadata_says_exists and file_actually_exists:
                    # True cache hit - both metadata and file agree
                    print(f"    [CACHE HIT] Chapter {ch_num} already processed.")
                    current_url = chapter_info.get("previous_url")
                    last_known_good_num = int(ch_num)
                    cache_hit = True
                    break
                elif metadata_says_exists and not file_actually_exists:
                    # Metadata is stale - file was deleted
                    print(f"    [STALE CACHE] Chapter {ch_num} marked as existing but file missing: {filename}")
                    print(f"    [FIXING] Updating metadata and re-scraping...")
                    # Update metadata to reflect reality
                    metadata["chapters"][ch_num]["file_exists"] = False
                    save_metadata(metadata, metadata_file)
                    # Don't break - continue to scraping logic
                    break
                elif not metadata_says_exists and file_actually_exists:
                    # File exists but metadata doesn't know - update metadata
                    print(f"    [ORPHANED FILE] Chapter {ch_num} file exists but not in metadata: {filename}")
                    print(f"    [FIXING] Updating metadata to reflect existing file...")
                    metadata["chapters"][ch_num]["file_exists"] = True
                    save_metadata(metadata, metadata_file)
                    current_url = chapter_info.get("previous_url")
                    last_known_good_num = int(ch_num)
                    cache_hit = True
                    break
        
        if not cache_hit:
            # No cache hit, proceed with scraping
            soup = fetch_with_retry(current_url, headers)
            if not soup:
                print(f"    [FATAL] Could not fetch valid content for {current_url}. Stopping.")
                break

            title = soup.select_one("#ChapterTitle").text.strip()
            current_chapter_num = extract_chapter_number(title)

            if current_chapter_num is None:
                print(f"    [FATAL] Could not parse chapter number from title: '{title}'. Stopping.")
                break

            # --- INTERACTIVE SEQUENCE VALIDATION ---
            if last_known_good_num is not None:
                expected_num = last_known_good_num - 1
                if current_chapter_num != expected_num:
                    print(f"    [SEQUENCE BREAK] Chapter number mismatch detected!")
                    print(f"    Expected: Chapter {expected_num}")
                    print(f"    Found:    Chapter {current_chapter_num}")
                    print(f"    Title:    '{title}'")
                    print(f"    URL:      {current_url}")
                    print()
                    
                    while True:
                        print("    Options:")
                        print(f"    [1] Use expected number ({expected_num}) and continue")
                        print(f"    [2] Use parsed number ({current_chapter_num}) and continue")
                        print("    [3] Skip this chapter")
                        print("    [4] Stop scraping")
                        
                        try:
                            choice = input("    Your choice (1-4): ").strip()
                            
                            if choice == "1":
                                print(f"    [USER] Using expected chapter number: {expected_num}")
                                current_chapter_num = expected_num
                                break
                            elif choice == "2":
                                print(f"    [USER] Using parsed chapter number: {current_chapter_num}")
                                # Update last_known_good_num to maintain sequence
                                last_known_good_num = current_chapter_num + 1
                                break
                            elif choice == "3":
                                print(f"    [USER] Skipping chapter. Moving to previous URL.")
                                # Get previous URL and continue without processing
                                prev_links = soup.select('a[href*="43713_"]')
                                if prev_links:
                                    current_url = urljoin(current_url, prev_links[0]['href'])
                                    continue
                                else:
                                    print("    [ERROR] No previous URL found. Stopping.")
                                    current_url = None
                                    break
                            elif choice == "4":
                                print(f"    [USER] Stopping scrape as requested.")
                                current_url = None
                                break
                            else:
                                print("    [ERROR] Invalid choice. Please enter 1, 2, 3, or 4.")
                                continue
                                
                        except (KeyboardInterrupt, EOFError):
                            print(f"\n    [USER] Interrupted. Stopping scrape.")
                            current_url = None
                            break
                    
                    # If user chose to skip or stop, break out of main loop
                    if current_url is None:
                        break
            
            filename = sanitize_filename(f"Chapter-{current_chapter_num:04d}-{title}.txt")
            filepath = os.path.join(output_dir, filename)

            if not os.path.exists(filepath):
                content_tag = soup.select_one("#Lab_Contents")
                for tag in content_tag.select('script, a'): tag.decompose()
                content = content_tag.get_text(separator='\n', strip=True)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"    [SAVED] Chapter {current_chapter_num}: {filename}")
            else:
                print(f"    [SKIP] File already exists for Chapter {current_chapter_num}.")

            prev_link_tag = soup.find('a', string=re.compile(r'上一章'))
            next_url = urljoin(current_url, prev_link_tag['href']) if prev_link_tag and prev_link_tag.get('href') else None

            metadata["chapters"][str(current_chapter_num)] = {
                "title": title, "url": current_url, "file_exists": True, "previous_url": next_url
            }
            
            if current_chapter_num % 20 == 0:
                save_metadata(metadata, metadata_file)
                print("    [META] Metadata checkpoint saved.")
                
            last_known_good_num = current_chapter_num
            current_url = next_url

        if current_url is None:
            print("\n🏁 Reached the end of the line (no more 'previous chapter' links).")
            break

        time.sleep(1)

    print("\n--- SCRAPING COMPLETED ---")
    save_metadata(metadata, metadata_file)
    print("✅ Final metadata saved.")
    print(f"🗂️  Total chapters in metadata: {len(metadata['chapters'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape novel chapters backwards with data integrity checks.')
    parser.add_argument('--start-chapter', type=int, help='Chapter number to start scraping from (scrapes backwards from here).')
    args = parser.parse_args()
    
    output_dir = "novel_content_dxmwx_complete"
    metadata_file = os.path.join("data", "temp", "scraping_metadata.json")
    
    start_url = 'https://www.dxmwx.org/read/43713_33325507.html' # Default: last chapter
    
    if args.start_chapter:
        print(f"Attempting to start from Chapter {args.start_chapter}...")
        metadata = load_or_create_metadata(metadata_file)
        start_url = get_url_for_chapter(args.start_chapter, metadata)
        if not start_url:
            print(f"❌ Could not find URL for Chapter {args.start_chapter} in existing metadata.")
            print("💡 Please run a full scrape first to populate the metadata file.")
            exit(1)
        print(f"✅ Found URL: {start_url}")

    scrape_backwards_final(start_url, output_dir, metadata_file)