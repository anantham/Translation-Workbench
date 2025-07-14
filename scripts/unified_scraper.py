import os
import re
import time
import json
import requests
import argparse
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from utils.logging import logger
from utils.adapter_factory import get_adapter

def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "", filename).strip()

def fetch_with_retry(url, headers, encoding, max_attempts=3):
    for attempt in range(1, max_attempts + 1):
        try:
            logger.debug(f"    [FETCH] Attempt {attempt}/{max_attempts} for {url}")
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            response.encoding = encoding
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup
        except requests.exceptions.RequestException as e:
            logger.error(f"    [FETCH ERROR] Request failed on attempt {attempt}: {e}")
        if attempt < max_attempts:
            sleep_duration = 5 * attempt
            logger.info(f"    [FETCH RETRY] Waiting {sleep_duration} seconds before next attempt.")
            time.sleep(sleep_duration)
    logger.error(f"    [FETCH FATAL] All {max_attempts} attempts failed for {url}.")
    return None

def load_or_create_metadata(metadata_file):
    logger.debug(f"Loading metadata from {metadata_file}")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    logger.debug("Metadata file not found, creating new.")
    return {"chapters": {}}

def save_metadata(metadata, metadata_file):
    logger.debug(f"Saving metadata to {metadata_file}")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

def get_url_for_chapter(chapter_num, metadata):
    chapter_info = metadata.get("chapters", {}).get(str(chapter_num))
    return chapter_info.get("url") if chapter_info else None

def scrape_novel(start_url: str, output_dir: str, metadata_file: str, direction: str, max_chapters: int = 1000, delay_seconds: int = 1, progress_callback=None, status_callback=None, conflict_handler=None):
    # --- RCA Logging ---
    logger.info("--- Scraper Function Entry Point ---")
    logger.debug(f"    - status_callback: {status_callback}")
    logger.debug(f"    - progress_callback: {progress_callback}")
    logger.debug(f"    - conflict_handler: {conflict_handler}")
    logger.info("------------------------------------")

    adapter = get_adapter(start_url)
    if not adapter:
        logger.critical(f"No adapter found for URL: {start_url}")
        return

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    os.makedirs(output_dir, exist_ok=True)
    
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
    metadata = load_or_create_metadata(metadata_file)
    if "chapters" not in metadata:
        metadata["chapters"] = {}
    
    current_url = start_url
    last_known_good_num = None
    
    if "chapters" in metadata and metadata["chapters"]:
        last_known_good_num = max([int(k) for k in metadata["chapters"].keys()])
        logger.info(f"Resuming scrape. Last known good chapter is {last_known_good_num}")

    logger.info(f"🚀 Starting Scrape from: {start_url}")

    chapters_scraped = 0
    while current_url and chapters_scraped < max_chapters:
        status_callback(f"  -> Processing URL: {current_url}")
        logger.info("-" * 70)
        logger.info(f"  -> Processing URL: {current_url}")

        # --- Advanced Cache/Manifest Validation ---
        cache_hit = False
        for ch_num_str, chapter_info in metadata.get("chapters", {}).items():
            if chapter_info.get("url") == current_url:
                ch_num = int(ch_num_str)
                logger.debug(f"URL {current_url} found in metadata for chapter {ch_num}.")
                title = chapter_info.get("title", f"Chapter {ch_num}")
                filename_num = chapter_info.get("filename_num", f"{ch_num:04d}")
                filename = sanitize_filename(f"Chapter-{filename_num}-{title}.txt")
                expected_filepath = os.path.join(output_dir, filename)
                
                metadata_says_exists = chapter_info.get("file_exists", False)
                file_actually_exists = os.path.exists(expected_filepath)
                
                logger.debug(f"    - Metadata 'file_exists': {metadata_says_exists}")
                logger.debug(f"    - Filesystem check: {file_actually_exists} at '{expected_filepath}'")

                if metadata_says_exists and file_actually_exists:
                    status_callback(f"    [CACHE HIT] Chapter {ch_num} is confirmed on disk. Skipping.")
                    logger.info(f"    [CACHE HIT] Chapter {ch_num} is confirmed on disk. Skipping.")
                    
                    chapters_scraped += 1
                    if progress_callback:
                        progress_callback(chapters_scraped, max_chapters)

                    current_url = chapter_info.get("previous_url")
                    last_known_good_num = chapter_info.get("end_chapter_num", ch_num)
                    cache_hit = True
                    break
                elif metadata_says_exists and not file_actually_exists:
                    logger.warning(f"    [STALE CACHE] Chapter {ch_num} marked as existing but file is missing.")
                    logger.info(f"    [FIXING] Updating metadata and re-scraping chapter.")
                    metadata["chapters"][ch_num_str]["file_exists"] = False
                    save_metadata(metadata, metadata_file)
                    break 
                elif not metadata_says_exists and file_actually_exists:
                    logger.info(f"    [ORPHANED FILE] File for chapter {ch_num} exists but metadata is not updated.")
                    logger.info(f"    [FIXING] Updating metadata to reflect existing file.")
                    metadata["chapters"][ch_num_str]["file_exists"] = True
                    save_metadata(metadata, metadata_file)
                    current_url = chapter_info.get("previous_url")
                    last_known_good_num = ch_num
                    cache_hit = True
                    break
        
        if cache_hit:
            time.sleep(0.1)
            continue
        
        soup = fetch_with_retry(current_url, headers, adapter.get_encoding())
        if not soup:
            logger.critical(f"    [FATAL] Could not fetch valid content for {current_url}. Stopping.")
            break

        title = adapter.extract_title(soup)
        if not title:
            logger.error(f"    [PARSE ERROR] Could not extract title from: {current_url}")
            break
            
        current_chapter_num, end_chapter_num, filename_num = adapter.parse_chapter_info(title)

        if current_chapter_num is None:
            logger.error(f"    [PARSE ERROR] Could not read a number from: '{title}'")
            if conflict_handler:
                resolution = conflict_handler(
                    url=current_url,
                    expected_number=None,
                    found_number=None,
                    title=title,
                    is_parse_error=True
                )
            else:
                # CLI fallback for parse error
                while True:
                    manual_num = input("    Enter chapter number manually, or (s)kip, (a)bort: ").strip()
                    if manual_num.isdigit():
                        current_chapter_num = int(manual_num)
                        end_chapter_num = current_chapter_num # Assume single chapter
                        filename_num = f"{current_chapter_num:04d}"
                        break
                    elif manual_num.lower() == 's':
                        current_url = adapter.get_next_link(soup, direction)
                        continue
                    elif manual_num.lower() == 'a':
                        current_url = None
                        break
                if current_url is None:
                    break
            if current_chapter_num is None: # If user chose to skip or abort
                continue

        # --- Mismatch Detection and Resolution ---
        if last_known_good_num is not None:
            expected_num = last_known_good_num - 1 if direction == "Backwards (newest to oldest)" else last_known_good_num + 1
            if current_chapter_num != expected_num:
                logger.warning(f"[SEQUENCE BREAK] Expected Ch. {expected_num}, but found Ch. {current_chapter_num} at {current_url}")
                logger.info(f"    - Title: '{title}'")
                logger.info(f"    - Extracted Numbers (start, end): ({current_chapter_num}, {end_chapter_num})")
                
                resolution = 'abort' # Default action
                if conflict_handler:
                    # --- Preview Fetching for Conflict Resolution ---
                    current_chapter_preview = "Could not extract preview."
                    try:
                        content = adapter.extract_content(soup)
                        current_chapter_preview = content[:300].strip() if content else "Content extraction returned empty."
                    except Exception as e:
                        logger.error(f"Failed to extract preview for current chapter: {e}")

                    last_chapter_preview = "Could not retrieve last chapter."
                    try:
                        last_chapter_info = metadata.get("chapters", {}).get(str(last_known_good_num))
                        if last_chapter_info:
                            last_title = last_chapter_info.get("title", f"Chapter {last_known_good_num}")
                            last_filename_num = last_chapter_info.get("filename_num", f"{last_known_good_num:04d}")
                            last_filename = sanitize_filename(f"Chapter-{last_filename_num}-{last_title}.txt")
                            last_filepath = os.path.join(output_dir, last_filename)
                            if os.path.exists(last_filepath):
                                with open(last_filepath, 'r', encoding='utf-8') as f:
                                    last_chapter_preview = f.read()[-300:].strip()
                            else:
                                last_chapter_preview = f"File not found: {last_filename}"
                        else:
                            last_chapter_preview = "Last chapter not found in metadata."
                    except Exception as e:
                        logger.error(f"Failed to read preview for last chapter: {e}")
                    # --- End Preview Fetching ---

                    resolution = conflict_handler(
                        url=current_url,
                        expected_number=expected_num,
                        found_number=current_chapter_num,
                        title=title,
                        last_chapter_preview=last_chapter_preview,
                        current_chapter_preview=current_chapter_preview
                    )
                else: # Fallback to CLI
                    print(f"    [SEQUENCE BREAK] Chapter number mismatch detected!")
                    print(f"    Expected: Chapter {expected_num}")
                    print(f"    Found:    Chapter {current_chapter_num} ('{title}')")
                    print(f"    URL:      {current_url}")
                    
                    while True:
                        choice = input(f"    Options: [1] Use expected ({expected_num}), [2] Use found ({current_chapter_num}), [3] Abort: ").strip()
                        if choice == '1':
                            resolution = 'expected'
                            break
                        elif choice == '2':
                            resolution = 'found'
                            break
                        elif choice == '3':
                            resolution = 'abort'
                            break
                
                if resolution == 'expected':
                    logger.info(f"    [USER] Using expected chapter number: {expected_num}")
                    current_chapter_num = expected_num
                elif resolution == 'found':
                    logger.info(f"    [USER] Using found chapter number: {current_chapter_num}")
                    last_known_good_num = current_chapter_num + 1 if direction == "Backwards (newest to oldest)" else current_chapter_num - 1
                else:
                    logger.critical("    [USER] Aborting scrape due to sequence break.")
                    break

        filename = sanitize_filename(f"Chapter-{filename_num}-{title}.txt")
        filepath = os.path.join(output_dir, filename)

        if not os.path.exists(filepath):
            logger.debug(f"File does not exist, proceeding to save: {filepath}")
            content = adapter.extract_content(soup)
            if not content:
                logger.error(f"    [PARSE ERROR] Could not extract content from: {current_url}")
                break
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            status_callback(f"    [SAVED] Chapter {current_chapter_num}: {filename}")
            logger.info(f"    [SAVED] Chapter {current_chapter_num}: {filename}")
        else:
            status_callback(f"    [SKIP] File already exists for Chapter {current_chapter_num}.")
            logger.info(f"    [SKIP] File already exists for Chapter {current_chapter_num}.")

        chapters_scraped += 1
        
        next_url = adapter.get_next_link(soup, direction)

        logger.debug(f"Updating metadata for chapter {current_chapter_num}")
        metadata["chapters"][str(current_chapter_num)] = {
            "title": title, 
            "url": current_url, 
            "file_exists": True, 
            "previous_url": next_url, 
            "filename_num": filename_num,
            "end_chapter_num": end_chapter_num
        }
        
        save_metadata(metadata, metadata_file)
        logger.info(f"    [META] Metadata saved for chapter {current_chapter_num}.")
            
        last_known_good_num = end_chapter_num
        current_url = next_url

        if progress_callback:
            progress_callback(chapters_scraped, max_chapters)

        if chapters_scraped >= max_chapters:
            logger.info(f"Reached max chapters limit: {max_chapters}")
            break

        time.sleep(delay_seconds)

    if current_url is None:
        if status_callback:
            status_callback(f"\n🏁 Reached the end of the line (no more '{direction}' links).")

    logger.info("\n--- SCRAPING COMPLETED ---")
    # Final save is now handled after each chapter
    logger.info(f"🗂️  Total chapters in metadata: {len(metadata['chapters'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape novel chapters with a unified adapter framework.')
    parser.add_argument('start_url', type=str, help='The starting URL to scrape from.')
    parser.add_argument('--output-dir', type=str, default="novel_content", help='Directory to save the scraped chapters.')
    parser.add_argument('--direction', type=str, default="Forwards (oldest to newest)", choices=["Forwards (oldest to newest)", "Backwards (newest to oldest)"], help='The direction to scrape in.')
    args = parser.parse_args()
    
    metadata_file = os.path.join("data", "temp", "scraping_metadata.json")
    
    scrape_novel(args.start_url, args.output_dir, metadata_file, args.direction)
