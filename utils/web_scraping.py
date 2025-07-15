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
    logger.info("--- Streamlit Scraper Entry Point ---")
    logger.debug(f"    - status_callback: {status_callback}")
    logger.debug(f"    - progress_callback: {progress_callback}")
    logger.debug(f"    - conflict_handler: {conflict_handler}")
    logger.info("---------------------------------------")

    resume_info = None
    current_url = start_url

    # Check if resuming from a conflict
    if st.session_state.get('scraping_override'):
        logger.debug("[BRIDGE] Detected 'scraping_override' in session state.")
        override_value = st.session_state.get('scraping_override')
        logger.debug(f"[BRIDGE] Override value: {override_value}")
        
        conflict = st.session_state.get('scraping_conflict_data', {})
        logger.debug(f"[BRIDGE] Retrieved conflict data: {conflict}")

        if 'url' in conflict:
            resume_info = {
                'override': override_value,
                'url': conflict['url'],
                'expected_number': conflict.get('expected_number'),
                'found_number': conflict.get('found_number')
            }
            current_url = conflict['url']
            logger.debug(f"[BRIDGE] Prepared resume_info dictionary: {resume_info}")
            
            # Clear data now that it has been used
            st.session_state.scraping_conflict_data = None
            logger.debug("[BRIDGE] Cleared scraping_conflict_data from session state.")
        else:
            logger.error("[BRIDGE] Conflict override was set, but no conflict URL found in session state.")
            status_callback("Error: Could not find conflict data to resume scraping.")
            return

        # Clear the override to prevent re-triggering
        logger.debug("[BRIDGE] Clearing scraping_override from session state.")
        st.session_state.scraping_override = None

    try:
        result = scrape_novel(
            start_url=current_url,
            output_dir=output_dir,
            metadata_file=f"{output_dir}/manifest.json",
            direction=scrape_direction,
            max_chapters=max_chapters,
            delay_seconds=delay_seconds,
            progress_callback=progress_callback,
            status_callback=status_callback,
            conflict_handler=conflict_handler, # Still passed for CLI mode
            resume_info=resume_info
        )
        return result
    except Exception as e:
        logger.critical(f"💥 Scraping failed with a critical error: {str(e)}")
        if status_callback:
            status_callback(f"💥 Scraping failed with a critical error: {str(e)}")
        return {"status": "error", "message": str(e)}


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