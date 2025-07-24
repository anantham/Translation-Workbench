"""Adapter for novelcool.com."""
import re
from urllib.parse import urljoin
from .base_adapter import BaseAdapter
from .logging import logger
from .debug_helpers import log_html_structure, save_failed_html, find_content_candidates, try_fallback_selectors, check_page_accessibility

class NovelcoolAdapter(BaseAdapter):
    """Adapter for scraping novelcool.com."""

    def extract_title(self, soup):
        title_tag = soup.find('h2', class_='chapter-title')
        return title_tag.text.strip() if title_tag else None

    def extract_content(self, soup):
        logger.debug("[NOVELCOOL] Starting content extraction")
        
        # Check page accessibility first
        is_accessible, access_status = check_page_accessibility(soup, self.url)
        logger.debug(f"[NOVELCOOL] Page accessibility: {access_status}")
        
        # Use proper content container based on HTML analysis
        logger.debug("[NOVELCOOL] Looking for content container")
        content_div = (
            soup.select_one('div.chapter-reading-section.position-relative') or
            soup.select_one('div.chapter-reading-section-list') or
            soup.select_one('div[class*="reading-section"]') or
            soup.select_one('div.overflow-hidden')
        )
        
        # EVIDENCE COLLECTION: What did we actually find?
        logger.debug(f"[NOVELCOOL] EVIDENCE - content_div type: {type(content_div)}")
        logger.debug(f"[NOVELCOOL] EVIDENCE - content_div is not None: {content_div is not None}")
        
        if content_div:
            # LOG THE ACTUAL HTML CONTENT
            logger.debug("[NOVELCOOL] RAW HTML - content_div full HTML:")
            logger.debug(f"[NOVELCOOL] RAW HTML - {str(content_div)}")
            logger.debug(f"[NOVELCOOL] RAW HTML - Length: {len(str(content_div))} characters")
            
            # Also log just the text content without HTML tags
            raw_text = content_div.get_text()
            logger.debug("[NOVELCOOL] RAW TEXT - Direct text extraction:")
            logger.debug(f"[NOVELCOOL] RAW TEXT - {raw_text[:500]}...")
            logger.debug(f"[NOVELCOOL] RAW TEXT - Length: {len(raw_text)} characters")
            # Test each hypothesis systematically
            logger.debug(f"[NOVELCOOL] EVIDENCE - content_div.copy method exists: {hasattr(content_div, 'copy')}")
            logger.debug(f"[NOVELCOOL] EVIDENCE - _clean_content_container method exists: {hasattr(self, '_clean_content_container')}")
            logger.debug(f"[NOVELCOOL] EVIDENCE - _clean_content_container value: {getattr(self, '_clean_content_container', 'NOT_FOUND')}")
            
            # Skip problematic copy() - operate directly on container
            logger.debug("[NOVELCOOL] Skipping copy() - operating on container directly")
            copied_div = content_div
                
            # Clean ads and scripts based on HTML analysis
            logger.debug("[NOVELCOOL] Cleaning ads and scripts")
            # Remove ads and scripts
            for ad in copied_div.select('div.mangaread-ad-box, script'):
                ad.decompose()
            # Remove empty paragraphs
            for p in copied_div.select('p'):
                if not p.get_text(strip=True):
                    p.decompose()
            logger.debug("[NOVELCOOL] Cleanup complete")
                
            # Extract text content
            content = copied_div.get_text(separator='\n', strip=True)
            logger.debug(f"[NOVELCOOL] Extracted {len(content)} characters")
            logger.debug(f"[NOVELCOOL] Content preview: {content[:200]}...")
            
            # Improved validation based on analysis
            if len(content) >= 500 and not content.lstrip().startswith(('<<', 'Next>>')):
                logger.debug("[NOVELCOOL] Content validation passed")
                return content
            else:
                logger.warning(f"[NOVELCOOL] Content validation failed - length: {len(content)}, starts with: {content[:50]}")
                return None
            
        logger.warning("[NOVELCOOL] Content container not found")
        
        # Try alternative content selectors with cleanup
        content_selectors = [
            'div[class*="chapter-content"]',
            'div[class*="reading-content"]', 
            'div[class*="novel-content"]',
            '.chapter-text',
            '.content-text',
            '#chapter-content'
        ]
        
        for selector in content_selectors:
            elements = soup.select(selector)
            for element in elements:
                content_clean = self._clean_content_container(element.copy())
                content = content_clean.get_text(separator='\n', strip=True)
                logger.info(f"[NOVELCOOL] Alternative selector '{selector}' found content ({len(content)} chars)")
                return content
        
        # Log HTML structure for debugging
        log_html_structure(soup, self.url, "novelcool")
        
        # Find content candidates
        candidates = find_content_candidates(soup, "novelcool")
        
        # Try fallback selectors with cleanup
        fallback_content = try_fallback_selectors(soup, "div.chapter-reading-section", "novelcool")
        if fallback_content:
            logger.info(f"[NOVELCOOL] Fallback selector found content ({len(fallback_content)} chars)")
            return fallback_content
            
        # Save failed HTML for manual inspection
        save_failed_html(soup, self.url, "novelcool", "content_extraction")
        
        logger.error("[NOVELCOOL] All content extraction methods failed")
        return None
    
    def _clean_content_container(self, container):
        """Remove navigation, metadata, and other non-content elements."""
        # Remove navigation elements
        for nav in container.find_all(['nav', 'div'], class_=lambda x: x and any(nav_word in str(x).lower() for nav_word in ['nav', 'pagination', 'chapter-nav', 'prev', 'next'])):
            nav.decompose()
        
        # Remove links that look like navigation
        for link in container.find_all('a', href=True):
            link_text = link.get_text().lower().strip()
            href = link.get('href', '').lower()
            # Remove if it's clearly navigation
            if any(nav_word in link_text for nav_word in ['prev', 'next', '<<', '>>', '上一', '下一']) or 'chapter' in href:
                link.decompose()
        
        # Remove chapter titles (usually h1, h2, h3 at the beginning)
        for title in container.find_all(['h1', 'h2', 'h3'], recursive=False):
            title.decompose()
            
        # Remove any div with "title" or "header" in class name
        for header in container.find_all('div', class_=lambda x: x and any(word in str(x).lower() for word in ['title', 'header', 'meta'])):
            header.decompose()
            
        return container
    
    def _is_valid_content(self, content):
        """Check if extracted content looks like actual chapter text."""
        if not content or len(content.strip()) < 100:
            return False
            
        # Check for navigation artifacts
        content_lower = content.lower()
        nav_indicators = ['<<prev', 'next>>', 'chapter-', 'https://', 'http://', 'www.']
        
        # If content starts with navigation, it's probably not clean
        first_100_chars = content[:100].lower()
        if any(indicator in first_100_chars for indicator in nav_indicators):
            logger.debug("[NOVELCOOL] Content validation failed: navigation artifacts in first 100 chars")
            return False
            
        # Content should have some substance (multiple sentences)
        sentence_count = content.count('.') + content.count('!') + content.count('?')
        if sentence_count < 3:
            logger.debug(f"[NOVELCOOL] Content validation failed: too few sentences ({sentence_count})")
            return False
            
        return True

    def get_next_link(self, soup, direction):
        logger.debug(f"[NOVELCOOL] Looking for navigation links, direction: {direction}")
        
        # Use robust navigation selector based on HTML analysis  
        nav_selectors = [
            '.chapter-reading-pageitem a[href]', # Primary: working next link
            '.chapter-reading-pagination a[href]', # Fallback: pagination container
            'a[href*="chapter"]',          # Broad chapter links
        ]
        
        nav_buttons = []
        used_selector = None
        
        for selector in nav_selectors:
            nav_buttons = soup.select(selector)
            if nav_buttons:
                used_selector = selector
                logger.debug(f"[NOVELCOOL] Found {len(nav_buttons)} nav buttons using selector: {selector}")
                break
        
        if not nav_buttons:
            logger.warning("[NOVELCOOL] No navigation buttons found with any selector")
            # Log all links for debugging
            all_links = soup.find_all('a', href=True)
            logger.debug(f"[NOVELCOOL] Found {len(all_links)} total links on page")
            for i, link in enumerate(all_links[:10]):  # Show first 10 links
                href = link.get('href', '')
                text = link.get_text(strip=True)[:30]
                logger.debug(f"[NOVELCOOL]   Link {i+1}: '{text}' -> {href}")
            return None

        # Log found navigation buttons
        for i, btn in enumerate(nav_buttons):
            text = btn.get_text(strip=True)
            href = btn.get('href', '')
            logger.debug(f"[NOVELCOOL]   Nav {i+1}: '{text}' -> {href}")

        if direction == "Forwards (oldest to newest)":
            # Try to find "next" button by text content
            for btn in nav_buttons:
                btn_text = btn.get_text().lower()
                if any(word in btn_text for word in ['next', '下一', 'siguiente', 'suivant']):
                    if btn.get('href'):
                        next_url = urljoin(self.url, btn['href'])
                        logger.debug(f"[NOVELCOOL] Found NEXT link: {next_url}")
                        return next_url
            
            # Fallback: try last button if it has href
            if nav_buttons and nav_buttons[-1].get('href'):
                next_url = urljoin(self.url, nav_buttons[-1]['href'])
                logger.debug(f"[NOVELCOOL] Using last button as NEXT: {next_url}")
                return next_url
                
        else: # Backwards
            # Try to find "prev" button by text content
            for btn in nav_buttons:
                btn_text = btn.get_text().lower()
                if any(word in btn_text for word in ['prev', 'previous', '上一', 'anterior', 'précédent']):
                    if btn.get('href'):
                        prev_url = urljoin(self.url, btn['href'])
                        logger.debug(f"[NOVELCOOL] Found PREV link: {prev_url}")
                        return prev_url
            
            # Fallback: try second-to-last button
            if len(nav_buttons) > 1 and nav_buttons[-2].get('href'):
                prev_url = urljoin(self.url, nav_buttons[-2]['href'])
                logger.debug(f"[NOVELCOOL] Using second-to-last button as PREV: {prev_url}")
                return prev_url

        logger.warning(f"[NOVELCOOL] No suitable {direction} navigation link found")
        return None

    def parse_chapter_info(self, title, soup=None):
        """
        Overrides the base parser to handle English chapter titles with ranges.
        Examples: 
        - 'Chapter 10: A New Beginning' → (10, 10, "0010")
        - 'Eternal Life Chapter 49-50' → (49, 50, "0049-0050")
        """
        if not title:
            return None, None, None
            
        # Match chapter numbers with optional ranges (49-50, 49~50)
        match = re.search(r'Chapter\s+(\d+(?:[-~]\d+)?)', title, re.IGNORECASE)
        if not match:
            logger.warning(f"Could not find a chapter number pattern in '{title}'.")
            return None, None, None
            
        numeral_part = match.group(1)
        range_match = re.match(r'(\d+)[-~](\d+)', numeral_part)
        
        try:
            if range_match:
                # Combined chapter like "49-50"
                start_int = int(range_match.group(1))
                end_int = int(range_match.group(2))
                
                # Handle abbreviated ranges like "49~0" meaning "49-50"
                if end_int < start_int and end_int < 10:
                    base = (start_int // 10) * 10
                    end_int = base + end_int
                    logger.info(f"[NOVELCOOL] Interpreted abbreviated range: {start_int}~{range_match.group(2)} as {start_int}-{end_int}")
                
                logger.info(f"[NOVELCOOL] Parsed combined chapter: {start_int}-{end_int}")
                return start_int, end_int, f"{start_int:04d}-{end_int:04d}"
            else:
                # Single chapter like "49"
                number = int(numeral_part)
                return number, number, f"{number:04d}"
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to convert number from title '{title}'. Error: {e}")
            return None, None, None