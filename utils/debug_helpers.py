"""
Debug utilities for web scraping diagnostics.
Works with any adapter to help diagnose parsing failures.
"""
import os
import re
from datetime import datetime
from .logging import logger

def log_html_structure(soup, url, adapter_name="unknown"):
    """Log detailed HTML structure for debugging parsing failures."""
    logger.debug(f"[DEBUG] HTML Structure Analysis for {adapter_name} adapter")
    logger.debug(f"[DEBUG] URL: {url}")
    
    # Log basic page info
    title_tag = soup.find('title')
    logger.debug(f"[DEBUG] Page title: {title_tag.text.strip() if title_tag else 'No title found'}")
    
    # Check for common anti-bot indicators
    if soup.find(string=re.compile(r"checking your browser|cloudflare|please wait", re.IGNORECASE)):
        logger.warning("[DEBUG] Possible anti-bot protection detected")
    
    # Log all div elements with classes (common content containers)
    divs_with_classes = soup.find_all('div', class_=True)
    logger.debug(f"[DEBUG] Found {len(divs_with_classes)} divs with classes:")
    for i, div in enumerate(divs_with_classes[:20]):  # Limit to first 20
        classes = ' '.join(div.get('class', []))
        text_preview = div.get_text().strip()[:50] if div.get_text() else "No text"
        logger.debug(f"[DEBUG]   Div {i+1}: class='{classes}' preview='{text_preview}...'")
    
    # Log other common content containers
    for tag in ['article', 'section', 'main', 'p']:
        elements = soup.find_all(tag, class_=True)
        if elements:
            logger.debug(f"[DEBUG] Found {len(elements)} {tag} elements with classes")

def save_failed_html(soup, url, adapter_name, error_type="content_extraction"):
    """Save HTML to file when parsing fails for manual inspection."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"failed_{adapter_name}_{error_type}_{timestamp}.html"
        debug_dir = os.path.join("data", "temp", "debug")
        os.makedirs(debug_dir, exist_ok=True)
        
        filepath = os.path.join(debug_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"<!-- URL: {url} -->\n")
            f.write(f"<!-- Adapter: {adapter_name} -->\n")
            f.write(f"<!-- Error Type: {error_type} -->\n")
            f.write(f"<!-- Timestamp: {timestamp} -->\n")
            f.write(str(soup.prettify()))
        
        logger.info(f"[DEBUG] Failed HTML saved to: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"[DEBUG] Failed to save HTML file: {e}")
        return None

def find_content_candidates(soup, adapter_name="unknown"):
    """Find potential content containers that might contain chapter text."""
    logger.debug(f"[DEBUG] Searching for content candidates ({adapter_name})")
    
    candidates = []
    
    # Common content selectors by priority
    selectors = [
        # High priority - specific novel sites
        ('div', 'chapter-reading-section'),  # novelcool
        ('div', 'chapter-content'),
        ('div', 'content'),
        ('div', 'novel-content'),
        ('div', 'text-content'),
        
        # Medium priority - general content
        ('article', None),
        ('main', None),
        ('div', 'main-content'),
        ('div', 'post-content'),
        ('div', 'entry-content'),
        
        # Lower priority - broader search
        ('div', 'container'),
        ('section', None),
    ]
    
    for tag, class_name in selectors:
        if class_name:
            elements = soup.find_all(tag, class_=class_name)
        else:
            elements = soup.find_all(tag)
            
        for element in elements:
            text = element.get_text(strip=True)
            if len(text) > 100:  # Only consider substantial content
                candidates.append({
                    'selector': f"{tag}.{class_name}" if class_name else tag,
                    'text_length': len(text),
                    'preview': text[:100] + "..." if len(text) > 100 else text
                })
    
    # Sort by text length (longer is likely better)
    candidates.sort(key=lambda x: x['text_length'], reverse=True)
    
    logger.debug(f"[DEBUG] Found {len(candidates)} content candidates:")
    for i, candidate in enumerate(candidates[:5]):  # Show top 5
        logger.debug(f"[DEBUG]   {i+1}. {candidate['selector']} ({candidate['text_length']} chars): {candidate['preview']}")
    
    return candidates

def try_fallback_selectors(soup, primary_selector, adapter_name="unknown"):
    """Try alternative selectors when primary selector fails."""
    logger.debug(f"[DEBUG] Primary selector '{primary_selector}' failed, trying fallbacks ({adapter_name})")
    
    # Common fallback patterns
    fallbacks = [
        "div[class*='content']",
        "div[class*='chapter']", 
        "div[class*='text']",
        "div[class*='reading']",
        "article",
        "main",
        ".content",
        ".chapter-content",
        ".post-content",
        "#content",
        "#chapter-content"
    ]
    
    for selector in fallbacks:
        try:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(strip=True)
                if len(text) > 50:  # Minimum viable content
                    logger.debug(f"[DEBUG] Fallback selector '{selector}' found content ({len(text)} chars)")
                    return element.get_text(separator='\n', strip=True)
        except Exception as e:
            logger.debug(f"[DEBUG] Fallback selector '{selector}' failed: {e}")
    
    logger.warning(f"[DEBUG] All fallback selectors failed for {adapter_name}")
    return None

def check_page_accessibility(soup, url):
    """Check if page is accessible or blocked by anti-bot measures."""
    logger.debug(f"[DEBUG] Checking page accessibility for {url}")
    
    # Check for common blocking indicators
    blocking_indicators = [
        (r"cloudflare", "Cloudflare protection"),
        (r"checking your browser", "Browser verification"),
        (r"please wait", "Loading delay"),
        (r"captcha", "CAPTCHA challenge"),
        (r"access denied", "Access denied"),
        (r"login", "Login required"),
        (r"sign in", "Sign in required"),
        (r"403|forbidden", "HTTP 403 Forbidden"),
        (r"404|not found", "HTTP 404 Not Found"),
    ]
    
    page_text = soup.get_text().lower()
    
    for pattern, description in blocking_indicators:
        if re.search(pattern, page_text):
            logger.warning(f"[DEBUG] Potential blocking detected: {description}")
            return False, description
    
    # Check if page has substantial content
    content_length = len(page_text.strip())
    if content_length < 200:
        logger.warning(f"[DEBUG] Page has very little content ({content_length} chars)")
        return False, f"Minimal content ({content_length} chars)"
    
    logger.debug(f"[DEBUG] Page appears accessible ({content_length} chars)")
    return True, "Page accessible"