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
        # Error handling as before
        return

    os.makedirs(output_dir, exist_ok=True)

    # --- Resume Logic ---
    manifest_path = os.path.join(output_dir, 'manifest.json')
    successful_chapters = []
    failed_chapters = []
    current_url = start_url

    if os.path.exists(manifest_path):
        status_callback("Manifest found. Attempting to resume...")
        logger.info(f"Manifest found at {manifest_path}. "
                    "Loading previous session.")
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
            successful_chapters = manifest.get('successful_chapters', [])
            failed_chapters = manifest.get('failed_chapters', [])

        if successful_chapters:
            last_chapter = successful_chapters[-1]
            last_url = last_chapter['url']
            status_callback(f"Resuming from chapter {last_chapter['number']}: "
                            f"{last_chapter['title']}")

            try:
                # Fetch the last successful page to find the next link
                response = requests.get(last_url, timeout=20)
                response.raise_for_status()
                response.encoding = adapter.get_encoding()
                soup = BeautifulSoup(response.text, 'html.parser')
                next_link = adapter.get_next_link(soup, scrape_direction)

                if next_link:
                    current_url = next_link
                    logger.info(f"Resuming scrape from URL: {current_url}")
                else:
                    status_callback("Scraping is already complete.")
                    logger.info("No next link found. Scraping is already complete.")
                    return  # Exit if there's nothing more to scrape
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to fetch next link from {last_url}: {e}")
                # Fallback to starting from the beginning if fetching the next link fails
                current_url = start_url
                successful_chapters = []
                failed_chapters = []

    # --- Main Scraping Loop ---
    chapters_scraped_this_session = 0
    total_time = 0
    errors = []

    # The number of chapters to scrape is the total max minus what we already have
    chapters_to_scrape = max_chapters - len(successful_chapters)

    # --- Manifest Saving Helper ---
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
                status_callback("Stop requested. Finishing current chapter "
                                "and saving manifest...")
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
                    # --- Chapter Number Validation Logic ---
                    chapter_number, filename_num = adapter.extract_chapter_number(soup)
                    
                    # Determine the final chapter number to use
                    if chapter_number is not None:
                        # Validate against expected number if not the first chapter
                        if successful_chapters:
                            expected_chapter_number = successful_chapters[-1]['number'] + 1
                            if chapter_number != expected_chapter_number:
                                warning_msg = (f"Chapter number mismatch. "
                                               f"Expected: {expected_chapter_number}, "
                                               f"Got: {chapter_number} from title. "
                                               f"Using title's number.")
                                logger.warning(warning_msg)
                                status_callback(f"⚠️ {warning_msg}")
                    else:
                        # Fallback to simple increment if number not in title
                        chapter_number = (successful_chapters[-1]['number'] + 1 
                                          if successful_chapters else 1)
                        filename_num = f"{chapter_number:04d}"
                        logger.warning(f"Could not extract chapter number from title: '{title}'. "
                                       f"Falling back to incremental number: {chapter_number}.")

                    # --- End Validation Logic ---

                    filename = sanitize_filename(f"{filename_num}_{title}.txt")
                    filepath = os.path.join(output_dir, filename)
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(content)

                    chapter_data = {'number': chapter_number, 'title': title, 'url': current_url}
                    successful_chapters.append(chapter_data)
                    chapters_scraped_this_session += 1
                    
                    # --- Live Manifest Save ---
                    _save_manifest()
                    
                    logger.info(f"Successfully scraped Chapter {chapter_number}: "
                                f"{title} from {current_url}")
                    progress_callback(len(successful_chapters), max_chapters)
                else:
                    error_message = ("Could not extract title or content from "
                                     f"{current_url}")
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
        # --- Final Manifest Save ---
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