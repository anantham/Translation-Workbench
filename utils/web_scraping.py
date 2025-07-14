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
        
        verified_chapters = [ch for ch in manifest_chapters if ch.get('filename_num') in disk_chapter_nums]
        if len(verified_chapters) != len(manifest_chapters):
            status_callback(f"⚠️ Found {len(verified_chapters)} chapters on disk, which differs from manifest. Corrected state.")
        successful_chapters = sorted(verified_chapters, key=lambda x: x['number'])

    # 3. Determine starting point
    if successful_chapters and not st.session_state.get('scraping_override'):
        last_chapter = successful_chapters[-1]
        last_url = last_chapter['url']
        status_callback(f"Resuming from verified Chapter {last_chapter['number']}: {last_chapter['title']}")
        try:
            response = requests.get(last_url, timeout=20)
            response.raise_for_status()
            response.encoding = adapter.get_encoding()
            soup = BeautifulSoup(response.text, 'html.parser')
            current_url = adapter.get_next_link(soup, scrape_direction)
        except requests.exceptions.RequestException as e:
            status_callback(f"❌ Could not fetch next chapter page: {e}")
            return
    
    if st.session_state.get('scraping_override'):
        current_url = st.session_state.scraping_conflict['url']

    # --- Main Scraping Loop ---
    chapters_scraped_this_session = 0
    total_time = 0
    errors = []
    chapters_to_scrape = max_chapters - len(successful_chapters)

    def _save_manifest():
        # ... (save manifest logic remains the same)
        pass

    try:
        while current_url and chapters_scraped_this_session < chapters_to_scrape:
            if st.session_state.get('stop_requested', False):
                status_callback("🛑 Stop requested. Saving manifest...")
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
                    found_number, filename_num = adapter.extract_chapter_number(soup)
                    
                    if found_number is None:
                        # Fallback if no number in title
                        found_number = (successful_chapters[-1]['number'] + 1 if successful_chapters else 1)
                        filename_num = f"{found_number:04d}"

                    expected_number = (successful_chapters[-1]['number'] + 1 if successful_chapters else 1)
                    
                    # --- Interactive Conflict Resolution ---
                    override = st.session_state.get('scraping_override')
                    if override:
                        if override == 'expected':
                            chapter_number = expected_number
                        else: # 'title'
                            chapter_number = found_number
                        st.session_state.scraping_override = None # Consume override
                    elif found_number != expected_number:
                        # --- PAUSE FOR USER INPUT ---
                        last_chap_content = load_chapter_content(output_dir, successful_chapters[-1]['filename_num'])
                        st.session_state.scraping_conflict = {
                            "url": current_url,
                            "expected_number": expected_number,
                            "found_number": found_number,
                            "last_chapter_preview": last_chap_content[-200:],
                            "current_chapter_preview": content[:200],
                        }
                        st.session_state.scraping_active = False
                        logger.warning(f"Scraping paused. Mismatch detected. Expected: {expected_number}, Found: {found_number}")
                        return # Stop execution to await user input

                    else:
                        chapter_number = expected_number

                    # ... (rest of the loop: duplicate check, save file, save manifest)
                    
                else:
                    # ... (error handling)
                    pass

                current_url = adapter.get_next_link(soup, scrape_direction)

            except requests.exceptions.RequestException as e:
                # ... (error handling)
                break

            end_time = time.time()
            total_time += (end_time - start_time)
            time.sleep(delay_seconds)
    finally:
        _save_manifest()
        status_callback("Scraping complete.")

    # ... (return statement)
    pass

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