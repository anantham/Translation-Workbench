"""
Web Scraping Integration Module for Streamlit

This module acts as a bridge between the Streamlit UI and the core scraping engine.
"""
import streamlit as st
from scripts.unified_scraper import scrape_novel
from .adapter_factory import get_adapter
from .logging import logger

def streamlit_scraper(start_url, output_dir, max_chapters, delay_seconds,
                      progress_callback=None, status_callback=None, conflict_handler=None,
                      scrape_direction="Forwards (oldest to newest)"):
    """Bridge function to call the core scraper from the Streamlit UI."""
    # --- RCA Logging ---
    logger.info("--- Streamlit Scraper Entry Point ---")
    logger.debug(f"    - Received status_callback: {status_callback}")
    logger.debug(f"    - Received progress_callback: {progress_callback}")
    logger.debug(f"    - Received conflict_handler: {conflict_handler}")
    logger.info("---------------------------------------")

    try:
        scrape_novel(
            start_url=start_url,
            output_dir=output_dir,
            metadata_file=f"{output_dir}/manifest.json",
            direction=scrape_direction,
            max_chapters=max_chapters,
            delay_seconds=delay_seconds,
            progress_callback=progress_callback,
            status_callback=status_callback,
            conflict_handler=conflict_handler
        )
    except Exception as e:
        logger.critical(f"💥 Scraping failed with a critical error: {str(e)}")
        if status_callback:
            status_callback(f"💥 Scraping failed with a critical error: {str(e)}")


def validate_scraping_url(url):
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