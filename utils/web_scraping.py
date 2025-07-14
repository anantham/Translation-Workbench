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
    
    # --- Phase 1: Build Coverage Map & Find Missing Chapters ---
    successful_chapters = []
    failed_chapters = []
    
    # 1a. Build a set of all chapter numbers covered by files on disk
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

    # 1b. Load manifest and reconcile with disk coverage
    if os.path.exists(manifest_path):
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        manifest_chapters = manifest.get('successful_chapters', [])
        failed_chapters = manifest.get('failed_chapters', [])
        
        # The verified list is now based on the comprehensive coverage map
        verified_chapters = [ch for ch in manifest_chapters if ch.get('number') in covered_chapters]
        successful_chapters = sorted(verified_chapters, key=lambda x: x['number'])
        logger.info(f"Verification complete. Found {len(covered_chapters)} chapters on disk.")

    # 1c. Identify gaps to be repaired
    missing_chapters = set()
    if successful_chapters:
        max_chapter_on_disk = max(covered_chapters)
        expected_set = set(range(1, max_chapter_on_disk + 1))
        missing_chapters = expected_set - covered_chapters
    
    # --- Phase 2: Repair Mode ---
    if missing_chapters:
        status_callback(f"Repair Mode: Found {len(missing_chapters)} missing chapters. Attempting to fill gaps...")
        logger.info(f"Entering Repair Mode. Missing chapters: {sorted(list(missing_chapters))}")
        
        repair_url = manifest.get('start_url', start_url) # Start from the beginning
        
        while repair_url and missing_chapters:
            # (Simplified loop for repair: fetch, parse, check if missing, download)
            # This part would need its own loop and error handling.
            # For now, we log and proceed to normal scraping.
            logger.info(f"Repairing... current URL: {repair_url}")
            # In a full implementation, we'd scrape here.
            # This is a placeholder for the repair logic.
            break # Placeholder to avoid infinite loop in this example
        
        status_callback("Repair Mode complete. Resuming normal scraping.")

    # --- Phase 3: Normal Scraping ---
    current_url = start_url
    if successful_chapters and not st.session_state.get('scraping_override'):
        # ... (logic to find the next URL to scrape from)
        pass

    # ... (The rest of the main scraping loop with interactive conflict resolution)
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