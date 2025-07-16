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
    if st.session_state.get('resume_payload'):
        logger.debug("[BRIDGE] Detected a resume_payload, indicating a conflict resolution.")
        
        override_info = st.session_state.get('scraping_override', {})
        conflict_data = st.session_state.get('resume_payload', {})
        
        logger.info(f"[BRIDGE] Override info: {override_info}")
        logger.info(f"[BRIDGE] Conflict payload: {conflict_data}")

        if 'url' in conflict_data:
            # Build the resume_info dictionary based on the override type
            resume_info = {
                'override': override_info.get('type'),
                'url': conflict_data.get('url'),
                'expected_number': conflict_data.get('expected_number'),
                'found_number': conflict_data.get('found_number')
            }
            
            # If it's a custom override, add the custom number
            if override_info.get('type') == 'custom':
                resume_info['custom_number'] = override_info.get('number')

            current_url = conflict_data['url']
            
            # --- Diagnostic Logging ---
            logger.info("="*50)
            logger.info("[BRIDGE LOG] Passing resume_info to core scraper:")
            logger.info(f"[BRIDGE LOG] Data: {resume_info}")
            logger.info("="*50)
            # --- End Diagnostic Logging ---
            
            # Clear session state variables now that they have been used
            st.session_state.resume_payload = None
            st.session_state.scraping_override = None
            
        else:
            logger.error("[BRIDGE] Resume payload was found, but it was missing a URL.")
            if status_callback:
                status_callback("Error: Could not find conflict data to resume scraping.")
            return {"status": "error", "message": "Missing conflict URL for resume."}
    
    # Call the core scraper
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
            conflict_handler=conflict_handler,
            resume_info=resume_info
        )
        
        # After the scraper runs, clear any lingering conflict data
        if st.session_state.get('scraping_conflict_data'):
            logger.debug("[BRIDGE] Scraper run finished, clearing old conflict data.")
            st.session_state.scraping_conflict_data = None

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