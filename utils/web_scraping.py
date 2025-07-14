"""
Web Scraping Integration Module

This module contains the core web scraping engine that works with different adapters.
"""
import os
import time
import json
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import streamlit as st
from .adapter_factory import get_adapter
from .logging import logger

def sanitize_filename(filename):
    return "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_')).rstrip()


def streamlit_scraper(start_url, output_dir, max_chapters, delay_seconds,
                      progress_callback, status_callback,
                      scrape_direction="Forwards (oldest to newest)"):
    adapter = get_adapter(start_url)
    if not adapter:
        status_callback("❌ Invalid URL or unsupported website.")
        return

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, 'manifest.json')
    
    # --- Robust Resume Logic ---
    successful_chapters = []
    failed_chapters = []
    current_url = start_url

    # 1. Get ground truth from file system
    import re
    disk_files = {f for f in os.listdir(output_dir) if f.endswith('.txt')}
    disk_chapter_nums = set()
    for f in disk_files:
        match = re.match(r'([0-9_]+)_', f)
        if match:
            disk_chapter_nums.add(match.group(1))

    # 2. Load manifest and reconcile
    if os.path.exists(manifest_path):
        status_callback("Manifest found. Verifying against file system...")
        logger.info(f"Manifest found at {manifest_path}. Reconciling with disk.")
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
            
        manifest_chapters = manifest.get('successful_chapters', [])
        failed_chapters = manifest.get('failed_chapters', [])
        
        # Keep only manifest entries that have a corresponding file
        verified_chapters = [
            ch for ch in manifest_chapters 
            if ch.get('filename_num') in disk_chapter_nums
        ]

        if len(verified_chapters) != len(manifest_chapters):
            logger.warning(f"Manifest-disk mismatch. "
                         f"Found {len(verified_chapters)} files for "
                         f"{len(manifest_chapters)} manifest entries. "
                         f"Using verified list.")
            status_callback(f"⚠️ Found {len(verified_chapters)} chapters on disk, "
                            f"which differs from manifest. Corrected state.")

        successful_chapters = sorted(verified_chapters, key=lambda x: x['number'])

    # 3. Determine starting point from verified data
    if successful_chapters:
        last_chapter = successful_chapters[-1]
        last_url = last_chapter['url']
        status_callback(f"Resuming from verified Chapter {last_chapter['number']}: "
                        f"{last_chapter['title']}")
        try:
            response = requests.get(last_url, timeout=20)
            response.raise_for_status()
            response.encoding = adapter.get_encoding()
            soup = BeautifulSoup(response.text, 'html.parser')
            next_link = adapter.get_next_link(soup, scrape_direction)

            if next_link:
                current_url = next_link
                logger.info(f"Resuming scrape from URL: {current_url}")
            else:
                status_callback("✅ Scraping is already complete.")
                return
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch next link from {last_url}: {e}")
            status_callback(f"❌ Could not fetch next chapter page. Please check connection.")
            return

    # --- Main Scraping Loop ---
    chapters_scraped_this_session = 0
    total_time = 0
    errors = []
    chapters_to_scrape = max_chapters - len(successful_chapters)

    def _save_manifest():
        manifest_data = {
            'scrape_update_time': datetime.now().isoformat(),
            'start_url': start_url,
            'scrape_direction': scrape_direction,
            'total_chapters_scraped': len(successful_chapters),
            'successful_chapters': successful_chapters,
            'failed_chapters': failed_chapters
        }
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, indent=4, ensure_ascii=False)
        logger.info(f"Manifest updated at {manifest_path}")

    try:
        while current_url and chapters_scraped_this_session < chapters_to_scrape:
            if st.session_state.get('stop_requested', False):
                status_callback("🛑 Stop requested. Saving manifest...")
                logger.info("Stop requested by user. Breaking scraping loop.")
                break
            
            start_time = time.time()
            status_callback(f"Scraping: {current_url}")

            try:
                response = requests.get(current_url, timeout=20)
                response.raise_for_status()
                response.encoding = adapter.get_encoding()
                soup = BeautifulSoup(response.text, 'html.parser')

                title = adapter.extract_title(soup)
                content = adapter.extract_content(soup)

                if title and content:
                    chapter_number, filename_num = adapter.extract_chapter_number(soup)
                    
                    if chapter_number is None:
                        chapter_number = (successful_chapters[-1]['number'] + 1 if successful_chapters else 1)
                        filename_num = f"{chapter_number:04d}"
                        logger.warning(f"Could not extract chapter number from title: '{title}'. "
                                       f"Falling back to incremental number: {chapter_number}.")
                    
                    # --- New Validation Logic ---
                    if successful_chapters:
                        expected_chapter_number = successful_chapters[-1]['number'] + 1
                        if chapter_number != expected_chapter_number:
                            warning_msg = (f"Chapter number mismatch. "
                                           f"Expected: {expected_chapter_number}, "
                                           f"Got: {chapter_number} from title. Using title's number.")
                            logger.warning(warning_msg)
                            status_callback(f"⚠️ {warning_msg}")

                    # --- Robust Duplicate Check ---
                    # Check if a file for this chapter number already exists on disk
                    
                    # Helper function for robust check
                    def chapter_already_exists(directory, chapter_id):
                        for f in os.listdir(directory):
                            if f.startswith(chapter_id + '_') and f.endswith('.txt'):
                                return True
                        return False

                    if chapter_already_exists(output_dir, filename_num):
                        logger.info(f"Chapter {filename_num} already exists on disk. Skipping.")
                        status_callback(f"⏭️ Chapter {filename_num} already exists. Skipping.")
                        current_url = adapter.get_next_link(soup, scrape_direction)
                        continue # Skip to next iteration

                    # --- End Duplicate Check ---

                    filename = sanitize_filename(f"{filename_num}_{title}.txt")
                    filepath = os.path.join(output_dir, filename)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)

                    chapter_data = {
                        'number': chapter_number, 
                        'filename_num': filename_num, 
                        'title': title, 
                        'url': current_url
                    }
                    successful_chapters.append(chapter_data)
                    chapters_scraped_this_session += 1
                    
                    _save_manifest()
                    
                    logger.info(f"Successfully scraped Chapter {filename_num}: {title}")
                    progress_callback(len(successful_chapters), max_chapters)
                else:
                    # Handle error
                    error_message = f"Could not extract title or content from {current_url}"
                    errors.append(error_message)
                    failed_chapters.append({'url': current_url, 'error': error_message})
                    logger.error(error_message)

                current_url = adapter.get_next_link(soup, scrape_direction)

            except requests.exceptions.RequestException as e:
                error_message = f"Error fetching {current_url}: {e}"
                errors.append(error_message)
                failed_chapters.append({'url': current_url, 'error': error_message})
                logger.error(error_message)
                break

            end_time = time.time()
            total_time += (end_time - start_time)
            time.sleep(delay_seconds)
    finally:
        _save_manifest()
        status_callback("Scraping complete.")

    return {
        'success': not errors,
        'chapters_scraped': len(successful_chapters),
        'total_time': total_time,
        'errors': errors,
        'chapters': successful_chapters,
        'failed': failed_chapters
    }





# Alias for backward compatibility with utils.py
# streamlit_scraper = streamlit_scraper_legacy


def validate_scraping_url(url):
    # Function remains the same
    adapter = get_adapter(url)
    if adapter:
        return {
            'valid': True,
            'site_type': adapter.__class__.__name__.replace("Adapter", ""),
            'recommendations': [],
            'warnings': []
        }
    else:
        return {
            'valid': False,
            'site_type': 'Unsupported',
            'recommendations': [],
            'warnings': ["This website is not supported."]
        }