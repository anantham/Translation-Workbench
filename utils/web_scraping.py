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
    
    # --- Phase 1: Build Coverage Map & Plan ---
    successful_chapters = []
    failed_chapters = []
    
    import re
    covered_chapters = set()
    disk_files = {f for f in os.listdir(output_dir) if f.endswith('.txt')}
    for f in disk_files:
        match = re.match(r'([0-9_]+)_', f)
        if match:
            num_part = match.group(1)
            if '_' in num_part:
                start, end = map(int, num_part.split('_'))
                covered_chapters.update(range(start, end + 1))
            else:
                covered_chapters.add(int(num_part))

    if os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        verified_chapters = [ch for ch in manifest.get('successful_chapters', []) if ch.get('number') in covered_chapters]
        successful_chapters = sorted(verified_chapters, key=lambda x: x['number'])
        failed_chapters = manifest.get('failed_chapters', [])
        start_url = manifest.get('start_url', start_url) # Use manifest start_url if available

    max_on_disk = max(covered_chapters) if covered_chapters else 0
    missing_chapters = set(range(1, max_on_disk + 1)) - covered_chapters
    new_chapters_to_get = list(range(max_on_disk + 1, max_chapters + 1))
    
    # The full to-do list
    chapters_to_process = sorted(list(missing_chapters)) + new_chapters_to_get
    
    logger.info(f"Scraping Plan: {len(missing_chapters)} chapters to repair, {len(new_chapters_to_get)} new chapters to scrape.")
    status_callback(f"Plan: Repairing {len(missing_chapters)} chapters, then scraping {len(new_chapters_to_get)} new ones.")

    # --- Phase 2: Determine Starting Point (Intelligent Jump) ---
    current_url = start_url
    if chapters_to_process:
        first_chapter_to_get = chapters_to_process[0]
        jump_off_chapter_num = first_chapter_to_get - 1

        if jump_off_chapter_num > 0:
            # Find the URL of the chapter right before the first missing one
            jump_off_chapter = next((ch for ch in successful_chapters if ch['number'] == jump_off_chapter_num), None)
            if jump_off_chapter:
                logger.info(f"Intelligent Jump: Starting from chapter {jump_off_chapter_num} to find chapter {first_chapter_to_get}.")
                status_callback(f"Jumping to chapter {jump_off_chapter_num} to find first missing chapter...")
                current_url = jump_off_chapter['url']
            else:
                logger.warning(f"Could not find chapter {jump_off_chapter_num} in manifest to perform jump. Starting from beginning.")
    elif successful_chapters:
        # If nothing to do, jump to the last chapter to check for new content
        current_url = successful_chapters[-1]['url']

    # --- Phase 3: Unified Scraping Loop ---
    total_chapters_in_plan = len(chapters_to_process)
    chapters_processed_count = 0
    
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
        while current_url and chapters_to_process:
            if st.session_state.get('stop_requested', False):
                status_callback("🛑 Stop requested. Saving manifest...")
                break

            status_callback(f"Navigating to: {current_url}")
            
            try:
                response = requests.get(current_url, timeout=20)
                response.raise_for_status()
                response.encoding = adapter.get_encoding()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                title = adapter.extract_title(soup)
                content = adapter.extract_content(soup)
                
                if not title or not content:
                    logger.error(f"Could not extract title or content from {current_url}")
                    current_url = adapter.get_next_link(soup, scrape_direction)
                    continue

                found_number, filename_num = adapter.extract_chapter_number(soup)
                
                if found_number in chapters_to_process:
                    status_callback(f"Processing Chapter {found_number}...")
                    
                    # --- All the saving and conflict logic goes here ---
                    filename = sanitize_filename(f"{filename_num}_{title}.txt")
                    filepath = os.path.join(output_dir, filename)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)

                    chapter_data = {
                        'number': found_number, 
                        'filename_num': filename_num, 
                        'title': title, 
                        'url': current_url
                    }
                    successful_chapters.append(chapter_data)
                    successful_chapters.sort(key=lambda x: x['number']) # Keep sorted
                    _save_manifest()
                    
                    chapters_to_process.remove(found_number)
                    chapters_processed_count += 1
                    progress_callback(chapters_processed_count, total_chapters_in_plan)
                    logger.info(f"Successfully scraped Chapter {filename_num}: {title}")

                else:
                    status_callback(f"Chapter {found_number} already exists. Skipping.")

                current_url = adapter.get_next_link(soup, scrape_direction)
                time.sleep(delay_seconds)

            except requests.exceptions.RequestException as e:
                logger.error(f"Error fetching {current_url}: {e}")
                failed_chapters.append({'url': current_url, 'error': str(e)})
                break
    
    finally:
        _save_manifest()
        status_callback("Scraping complete.")

    return {
        'success': not failed_chapters,
        'chapters_scraped': chapters_processed_count,
        'errors': failed_chapters,
        'chapters': successful_chapters,
        'failed': failed_chapters
    }

def load_chapter_content(output_dir, filename_num):
    # Helper to read chapter content for preview
    for f in os.listdir(output_dir):
        if f.startswith(filename_num + '_') and f.endswith('.txt'):
            with open(os.path.join(output_dir, f), 'r', encoding='utf-8') as file:
                return file.read()
    return "Could not load chapter preview."





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