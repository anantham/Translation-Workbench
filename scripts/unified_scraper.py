import os
import re
import time
import json
import requests
import argparse
from bs4 import BeautifulSoup
from utils.logging import logger
from utils.adapter_factory import get_adapter
from utils.debug_helpers import save_failed_html

def sanitize_filename(filename):
    return re.sub(r'[\\/*?:"<>|]', "", filename).strip()

def is_chapter_covered_by_range_file(chapter_num, metadata, output_dir):
    """
    Check if a chapter number is covered by any existing range file.
    
    For example, if we're looking for chapter 50 and there's a range file
    "Chapter-0049-0050-..." covering chapters 49-50, this returns True.
    """
    for ch_key, chapter_info in metadata.get("chapters", {}).items():
        try:
            # Handle both single keys "49" and range keys "49-50"
            if '-' in ch_key:
                start_ch, end_ch = map(int, ch_key.split('-'))
            else:
                start_ch = int(ch_key)
                end_ch = chapter_info.get("end_chapter_num", start_ch)

            # Check if the target chapter_num falls within this range
            if start_ch <= chapter_num <= end_ch:
                # Verify the range file actually exists
                title = chapter_info.get("title", f"Chapter {ch_key}")
                filename_num = chapter_info.get("filename_num", f"{start_ch:04d}")
                filename = sanitize_filename(f"Chapter-{filename_num}-{title}.txt")
                expected_filepath = os.path.join(output_dir, filename)
                
                if os.path.exists(expected_filepath):
                    logger.debug(f"    [RANGE COVERAGE] Chapter {chapter_num} is covered by range file: {filename}")
                    return True, expected_filepath, chapter_info
                    
        except (ValueError, TypeError):
            continue
            
    return False, None, None

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

def find_chapter_info_by_num(metadata, chapter_num):
    """Finds chapter info in metadata, supporting both single and range keys."""
    # First, try a direct lookup for single chapters
    direct_hit = metadata.get("chapters", {}).get(str(chapter_num))
    if direct_hit:
        return direct_hit
    
    # If not found, iterate through all chapters to find a range that contains it
    for key, chapter_info in metadata.get("chapters", {}).items():
        try:
            start = chapter_info.get("start_chapter_num")
            end = chapter_info.get("end_chapter_num")
            if start and end and start <= chapter_num <= end:
                logger.debug(f"Found chapter {chapter_num} in range key {key}")
                return chapter_info
        except (ValueError, TypeError):
            continue
    return None

def scrape_novel(start_url: str, output_dir: str, metadata_file: str, direction: str, max_chapters: int = 1000, delay_seconds: int = 1, progress_callback=None, status_callback=None, conflict_handler=None, resume_info=None):
    # --- RCA Logging ---
    logger.info("--- Scraper Function Entry Point ---")
    logger.debug(f"    - status_callback: {status_callback}")
    logger.debug(f"    - progress_callback: {progress_callback}")
    logger.debug(f"    - conflict_handler: {conflict_handler}")
    logger.debug(f"    - resume_info: {resume_info}")
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
        processed_nums = []
        for chapter_info in metadata["chapters"].values():
            # Prioritize the explicit end_chapter_num if it exists
            end_num = chapter_info.get("end_chapter_num")
            if isinstance(end_num, int):
                processed_nums.append(end_num)
        
        if processed_nums:
            last_known_good_num = max(processed_nums)
            logger.info(f"Resuming scrape. Last known good chapter is {last_known_good_num}")

    logger.info(f"🚀 Starting Scrape from: {start_url}")

    deferred_stop = False
    chapters_scraped = 0
    while current_url and chapters_scraped < max_chapters:
        if resume_info:
            try:
                logger.info(f"  [RESUME] Activating resume logic with: {resume_info}")
                override_decision = resume_info['override']
                current_url = resume_info['url']
                status_callback(f"  [RESUME] Fetching chapter from {current_url}")
                
                soup = fetch_with_retry(current_url, headers, adapter.get_encoding())
                if not soup:
                    logger.critical(f"    [FATAL] Could not fetch content for resumed URL {current_url}. Stopping.")
                    status_callback("    [FATAL] Resume failed: Could not fetch content.")
                    break

                logger.debug("  [RESUME] Successfully fetched content, extracting title.")
                title = adapter.extract_title(soup)
                if not title:
                    logger.error(f"    [PARSE ERROR] Could not extract title from resumed URL: {current_url}")
                    status_callback("    [FATAL] Resume failed: Could not parse title.")
                    break

                logger.debug(f"  [RESUME] Successfully extracted title: '{title}'")
                _, _, filename_num_temp = adapter.parse_chapter_info(title, soup, resume_info.get('expected_number'))

                if override_decision == 'expected':
                    current_chapter_num = resume_info['expected_number']
                    end_chapter_num = current_chapter_num
                    filename_num = f"{current_chapter_num:04d}"
                    logger.info(f"  [RESUME] Overriding to EXPECTED chapter number: {current_chapter_num}")
                elif override_decision == 'title':
                    current_chapter_num = resume_info['found_number']
                    end_chapter_num = current_chapter_num
                    filename_num = filename_num_temp
                    logger.info(f"  [RESUME] Overriding to FOUND chapter number: {current_chapter_num}")
                    last_known_good_num = current_chapter_num + 1 if direction == "Backwards (newest to oldest)" else current_chapter_num - 1
                elif override_decision == 'custom':
                    current_chapter_num = resume_info['custom_number']
                    end_chapter_num = current_chapter_num
                    filename_num = f"{current_chapter_num:04d}"
                    logger.info(f"  [RESUME] Overriding to CUSTOM chapter number: {current_chapter_num}")
                    # Adjust last_known_good_num as if we found this number
                    last_known_good_num = current_chapter_num + 1 if direction == "Backwards (newest to oldest)" else current_chapter_num - 1
                
                logger.debug(f"  [RESUME] Final chapter number: {current_chapter_num}, filename number: {filename_num}")
                filename = sanitize_filename(f"Chapter-{filename_num}-{title}.txt")
                filepath = os.path.join(output_dir, filename)
                
                logger.debug(f"  [RESUME] Checking for existing file at: {filepath}")
                if not os.path.exists(filepath):
                    logger.debug("  [RESUME] File does not exist, extracting content.")
                    content = adapter.extract_content(soup)
                    if content:
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(content)
                        status_callback(f"    [SAVED] Chapter {current_chapter_num}: {filename}")
                        logger.info(f"    [SAVED] Chapter {current_chapter_num}: {filename}")
                    else:
                        logger.error(f"    [PARSE ERROR] Could not extract content from resumed URL: {current_url}")
                        logger.error(f"    [PARSE ERROR] Resume title was: '{title}'")
                        logger.error(f"    [PARSE ERROR] Resume adapter: {type(adapter).__name__}")
                        
                        # Save the failed resume HTML for inspection
                        adapter_name = type(adapter).__name__.replace('Adapter', '').lower()
                        saved_file = save_failed_html(soup, current_url, adapter_name, "resume_content_fail")
                        if saved_file:
                            logger.error(f"    [PARSE ERROR] Resume failed HTML saved to: {saved_file}")
                        
                        status_callback("    [FATAL] Resume failed: Could not parse content.")
                        break
                else:
                    status_callback(f"    [SKIP] File already exists for Chapter {current_chapter_num}.")
                    logger.info(f"    [SKIP] File already exists for Chapter {current_chapter_num}.")

                next_url = adapter.get_next_link(soup, direction)
                logger.debug(f"  [RESUME] Found next URL: {next_url}")

                metadata["chapters"][str(current_chapter_num)] = {
                    "title": title, "url": current_url, "file_exists": True, 
                    "previous_url": next_url, "filename_num": filename_num,
                    "end_chapter_num": end_chapter_num
                }
                save_metadata(metadata, metadata_file)
                logger.info(f"    [META] Metadata saved for chapter {current_chapter_num}.")
                
                last_known_good_num = end_chapter_num
                current_url = next_url
                chapters_scraped += 1
                if progress_callback:
                    progress_callback(chapters_scraped, max_chapters)
                
                logger.info(f"  [RESUME] Resume block complete. Continuing to next chapter: {current_url}")
                time.sleep(delay_seconds)
                continue

            except Exception as e:
                logger.critical(f"  [RESUME FATAL] An unexpected error occurred in the resume block: {e}", exc_info=True)
                status_callback(f"💥 FATAL ERROR during resume: {e}")
                break # Stop scraping
            finally:
                # Ensure resume_info is cleared to prevent re-triggering, even if an error occurs
                resume_info = None

        status_callback(f"  -> Processing URL: {current_url}")
        logger.info("-" * 70)
        logger.info(f"  -> Processing URL: {current_url}")

        # --- Advanced Cache/Manifest Validation ---
        cache_hit = False
        for ch_key, chapter_info in metadata.get("chapters", {}).items():
            if chapter_info.get("url") == current_url:
                # This chapter's URL is in the metadata. We can trust the info.
                logger.debug(f"URL {current_url} found in metadata under key '{ch_key}'.")
                
                title = chapter_info.get("title", f"Chapter {ch_key}")
                filename_num = chapter_info.get("filename_num", "0000")
                filename = sanitize_filename(f"Chapter-{filename_num}-{title}.txt")
                expected_filepath = os.path.join(output_dir, filename)
                
                metadata_says_exists = chapter_info.get("file_exists", False)
                file_actually_exists = os.path.exists(expected_filepath)
                
                logger.debug(f"    - Metadata 'file_exists': {metadata_says_exists}")
                logger.debug(f"    - Filesystem check: {file_actually_exists} at '{expected_filepath}'")

                if metadata_says_exists and file_actually_exists:
                    status_callback(f"    [CACHE HIT] Chapter(s) {ch_key} confirmed on disk. Skipping.")
                    logger.info(f"    [CACHE HIT] Chapter(s) {ch_key} confirmed on disk. Skipping.")
                    
                    chapters_scraped += 1
                    if progress_callback:
                        progress_callback(chapters_scraped, max_chapters)

                    current_url = chapter_info.get("previous_url")
                    last_known_good_num = chapter_info.get("end_chapter_num")
                    cache_hit = True
                    break
                elif metadata_says_exists and not file_actually_exists:
                    logger.warning(f"    [STALE CACHE] Chapter(s) {ch_key} marked as existing but file is missing.")
                    logger.info("    [FIXING] Updating metadata and re-scraping chapter.")
                    metadata["chapters"][ch_key]["file_exists"] = False
                    save_metadata(metadata, metadata_file)
                    break 
                elif not metadata_says_exists and file_actually_exists:
                    logger.info(f"    [ORPHANED FILE] File for chapter(s) {ch_key} exists but metadata is not updated.")
                    logger.info("    [FIXING] Updating metadata to reflect existing file.")
                    metadata["chapters"][ch_key]["file_exists"] = True
                    save_metadata(metadata, metadata_file)
                    current_url = chapter_info.get("previous_url")
                    last_known_good_num = chapter_info.get("end_chapter_num")
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
            
        # Calculate expected chapter number for intelligent parsing
        expected_num = None
        if last_known_good_num is not None:
            expected_num = last_known_good_num - 1 if direction == "Backwards (newest to oldest)" else last_known_good_num + 1
            
        current_chapter_num, end_chapter_num, filename_num = adapter.parse_chapter_info(title, soup)

        if current_chapter_num is None:
            logger.error(f"    [PARSE ERROR] Could not read a number from: '{title}'")
            if conflict_handler:
                # This logic mirrors the sequence_break handler to ensure consistency.
                # It pauses the scraper and returns control to the UI for resolution.
                try:
                    preview = adapter.extract_content(soup)
                    current_chapter_preview = preview[:300].strip() if preview else "Content extraction returned empty."
                except Exception as e:
                    logger.error(f"Failed to extract preview for parse error: {e}")
                    current_chapter_preview = "Could not extract preview."

                conflict_data = {
                    'type': 'parse_error',
                    'url': current_url,
                    'expected_number': expected_num,
                    'found_number': None,
                    'title': title,
                    'last_chapter_preview': 'N/A', # Not relevant for a simple parse error
                    'current_chapter_preview': current_chapter_preview
                }
                logger.info("    [DEFER] Parse error detected. Returning data to UI for resolution.")
                return {"status": "conflict", "data": conflict_data}

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
                        # We will 'continue' below, so this is fine
                        break
                    elif manual_num.lower() == 'a':
                        current_url = None
                        break
                if current_url is None:
                    break
            
            if current_chapter_num is None: # If user chose to skip or abort via CLI
                continue

        # --- Mismatch Detection and Resolution ---
        if last_known_good_num is not None:
            expected_num = last_known_good_num - 1 if direction == "Backwards (newest to oldest)" else last_known_good_num + 1
            
            # A sequence break occurs if the found chapter number is not the expected one.
            # We make an exception for combined chapters, where the start of the range
            # must match the expected number.
            is_sequence_break = (current_chapter_num != expected_num)

            if is_sequence_break:
                logger.warning(f"[SEQUENCE BREAK] Expected Ch. {expected_num}, but found Ch. {current_chapter_num} (range: {current_chapter_num}-{end_chapter_num}) at {current_url}")
                logger.info(f"    - Title: '{title}'")
                
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
                        # Use the new robust function to find the last chapter's info
                        last_chapter_info = find_chapter_info_by_num(metadata, last_known_good_num)
                        
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

                    # Instead of calling a handler, prepare data to be returned
                    conflict_data = {
                        'type': 'sequence_break',
                        'url': current_url,
                        'expected_number': expected_num,
                        'found_number': current_chapter_num,
                        'title': title,
                        'last_chapter_preview': last_chapter_preview,
                        'current_chapter_preview': current_chapter_preview
                    }
                    
                    # If a handler is provided (for UI mode), return the data
                    if conflict_handler:
                        logger.info("    [DEFER] Conflict detected. Returning data to UI for resolution.")
                        return {"status": "conflict", "data": conflict_data}

                    # Fallback to CLI if no handler
                    print("    [SEQUENCE BREAK] Chapter number mismatch detected!")
                    print(f"    Expected: Chapter {expected_num}")
                    print(f"    Found:    Chapter {current_chapter_num} ('{title}')")
                    print(f"    URL:      {current_url}")

                    # --- Translated Preview for CLI ---
                    try:
                        from utils.translation import translate_with_gemini
                        from utils.config import load_api_config
                        api_key, _ = load_api_config()

                        if api_key:
                            print("\n    --- Translating previews for context... ---")
                            
                            # Translate previous chapter's end
                            prev_prompt = f"Please provide a concise, high-quality English translation of the following excerpt, which is the END of a chapter:\n\n---\n{last_chapter_preview}\n---\n"
                            translated_prev_preview = translate_with_gemini(prev_prompt, api_key, use_cache=True, novel_name="preview_translation")
                            
                            # Translate current chapter's start
                            curr_prompt = f"Please provide a concise, high-quality English translation of the following excerpt, which is the START of a new chapter:\n\n---\n{current_chapter_preview}\n---\n"
                            translated_curr_preview = translate_with_gemini(curr_prompt, api_key, use_cache=True, novel_name="preview_translation")

                            print(f"\n    [End of Previous Chapter ({last_known_good_num}) - Translated Preview]")
                            print(f"    ... {translated_prev_preview.strip()}\n")
                            print(f"    [Start of Current Chapter ({current_chapter_num}) - Translated Preview]")
                            print(f"    {translated_curr_preview.strip()} \n")
                        else:
                            raise ValueError("API key not found.")

                    except Exception as e:
                        logger.warning(f"Could not get translated previews: {e}. Falling back to raw text.")
                        print("\n    --- Displaying raw text previews for context ---")
                        print(f"\n    [End of Previous Chapter ({last_known_good_num}) - Raw Preview]")
                        print(f"    ... {last_chapter_preview.strip()}\n")
                        print(f"    [Start of Current Chapter ({current_chapter_num}) - Raw Preview]")
                        print(f"    {current_chapter_preview.strip()} \n")
                    # --- End Translated Preview ---
                    
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
                
                if resolution == 'defer':
                    # This state should not be reached if conflict_handler is used
                    logger.error("    [FATAL] 'defer' resolution reached without a conflict handler.")
                    return {"status": "error", "message": "Deferred without handler."}

                if resolution == 'expected':
                    logger.info(f"    [USER] Using expected chapter number: {expected_num}")
                    current_chapter_num = expected_num
                elif resolution == 'found':
                    logger.info(f"    [USER] Using found chapter number: {current_chapter_num}")
                    last_known_good_num = current_chapter_num + 1 if direction == "Backwards (newest to oldest)" else current_chapter_num - 1
                else:
                    logger.critical("    [USER] Aborting scrape due to sequence break.")
                    return {"status": "aborted"}

        filename = sanitize_filename(f"Chapter-{filename_num}-{title}.txt")
        filepath = os.path.join(output_dir, filename)

        if not os.path.exists(filepath):
            logger.debug(f"File does not exist, proceeding to save: {filepath}")
            content = adapter.extract_content(soup)
            if not content:
                logger.error(f"    [PARSE ERROR] Could not extract content from: {current_url}")
                logger.error(f"    [PARSE ERROR] Title was: '{title}'")
                logger.error(f"    [PARSE ERROR] Adapter: {type(adapter).__name__}")
                logger.error(f"    [PARSE ERROR] Page length: {len(str(soup))} chars")
                
                # Save the failed HTML for inspection
                adapter_name = type(adapter).__name__.replace('Adapter', '').lower()
                saved_file = save_failed_html(soup, current_url, adapter_name, "scraper_content_fail")
                if saved_file:
                    logger.error(f"    [PARSE ERROR] Failed HTML saved to: {saved_file}")
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

        # --- Metadata Update ---
        # Handle combined chapters by using a range string as the key
        if current_chapter_num != end_chapter_num:
            metadata_key = f"{current_chapter_num}-{end_chapter_num}"
            logger.info(f"    [META] Saving combined chapter metadata under key: '{metadata_key}'")
        else:
            metadata_key = str(current_chapter_num)

        logger.debug(f"Updating metadata for chapter(s) {metadata_key}")
        metadata["chapters"][metadata_key] = {
            "title": title, 
            "url": current_url, 
            "file_exists": True, 
            "previous_url": next_url, 
            "filename_num": filename_num,
            "start_chapter_num": current_chapter_num, # Explicitly store start
            "end_chapter_num": end_chapter_num
        }
        
        save_metadata(metadata, metadata_file)
        logger.info(f"    [META] Metadata saved for chapter(s) {metadata_key}.")
            
        last_known_good_num = end_chapter_num
        current_url = next_url

        if progress_callback:
            progress_callback(chapters_scraped, max_chapters)

        if chapters_scraped >= max_chapters:
            logger.info(f"Reached max chapters limit: {max_chapters}")
            break

        time.sleep(delay_seconds)

    if current_url is None and not deferred_stop:
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
