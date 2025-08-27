"""Adapter for booktoki468.com Korean web novels."""
import re
from urllib.parse import urljoin
from .base_adapter import BaseAdapter
from .logging import logger
from .debug_helpers import log_html_structure, save_failed_html, find_content_candidates, try_fallback_selectors, check_page_accessibility


class BookTokiAdapter(BaseAdapter):
    """Adapter for scraping booktoki468.com Korean web novels."""

    def get_encoding(self):
        """BookToki uses UTF-8 encoding."""
        return 'utf-8'

    def extract_title(self, soup):
        """Extract chapter title from the page."""
        logger.debug("[BOOKTOKI] Starting title extraction")
        
        # Try multiple selectors for title
        title_selectors = [
            'title',  # Page title (contains chapter title)
            'h1',     # Main heading
            'h2',     # Secondary heading
            '.title', # Title class
        ]
        
        for selector in title_selectors:
            title_element = soup.select_one(selector)
            if title_element:
                title_text = title_element.get_text(strip=True)
                logger.debug(f"[BOOKTOKI] Found title with selector '{selector}': {title_text}")
                
                # Extract chapter title from page title
                if selector == 'title':
                    # Extract Korean chapter title pattern like "던전 디펜스-2화"
                    chapter_match = re.search(r'([^_]+?-\d+화)', title_text)
                    if chapter_match:
                        chapter_title = chapter_match.group(1)
                        logger.debug(f"[BOOKTOKI] Extracted chapter title: {chapter_title}")
                        return chapter_title
                else:
                    return title_text
        
        logger.warning("[BOOKTOKI] No title found")
        return None

    def extract_content(self, soup):
        """Extract chapter content from the page."""
        logger.debug("[BOOKTOKI] Starting content extraction")
        
        # Check page accessibility first
        is_accessible, access_status = check_page_accessibility(soup, self.url)
        logger.debug(f"[BOOKTOKI] Page accessibility: {access_status}")
        
        # Primary content selector based on HTML analysis
        content_container = soup.select_one('#novel_content')
        
        if not content_container:
            logger.warning("[BOOKTOKI] Primary content container (#novel_content) not found")
            return self._try_alternative_extraction(soup)
        
        logger.debug(f"[BOOKTOKI] Found content container: {type(content_container)}")
        
        # Look for content within the f9e99a33513 div (based on HTML analysis)
        content_div = content_container.select_one('div.f9e99a33513')
        if not content_div:
            # Fallback to direct extraction from novel_content
            content_div = content_container
            logger.debug("[BOOKTOKI] Using novel_content directly (no f9e99a33513 div found)")
        else:
            logger.debug("[BOOKTOKI] Found f9e99a33513 content div")
        
        # Extract all paragraph content
        paragraphs = content_div.find_all('p')
        if not paragraphs:
            logger.warning("[BOOKTOKI] No paragraphs found in content container")
            return self._extract_raw_text(content_div)
        
        logger.debug(f"[BOOKTOKI] Found {len(paragraphs)} paragraphs")
        
        # Extract and clean paragraph text
        chapter_text = []
        for i, p in enumerate(paragraphs):
            text = p.get_text(strip=True)
            if text and self._is_valid_paragraph(text):
                chapter_text.append(text)
                logger.debug(f"[BOOKTOKI] Paragraph {i+1}: {text[:50]}...")
        
        if not chapter_text:
            logger.warning("[BOOKTOKI] No valid paragraphs found")
            return self._extract_raw_text(content_div)
        
        content = '\n\n'.join(chapter_text)
        logger.debug(f"[BOOKTOKI] Extracted {len(content)} characters from {len(chapter_text)} paragraphs")
        
        # Validate content
        if self._is_valid_content(content):
            return content
        else:
            logger.warning("[BOOKTOKI] Content validation failed")
            return None

    def _is_valid_paragraph(self, text):
        """Check if a paragraph contains valid story content."""
        if not text or len(text.strip()) < 3:
            return False
        
        # Skip separator lines and metadata
        invalid_patterns = [
            r'^={5,}',  # Separator lines like "====="
            r'^\d{5}\s',  # Chapter numbers like "00002"
            r'^https?://',  # URLs
            r'^www\.',  # Web addresses
        ]
        
        for pattern in invalid_patterns:
            if re.match(pattern, text.strip()):
                return False
        
        return True

    def _is_valid_content(self, content):
        """Validate extracted content."""
        if not content or len(content.strip()) < 500:
            logger.debug(f"[BOOKTOKI] Content too short: {len(content) if content else 0} chars")
            return False
        
        # Check for Korean text (basic validation)
        korean_chars = re.findall(r'[가-힣]', content)
        if len(korean_chars) < 100:
            logger.debug(f"[BOOKTOKI] Insufficient Korean characters: {len(korean_chars)}")
            return False
        
        # Check content quality indicators
        sentence_indicators = content.count('.') + content.count('!') + content.count('?') + content.count('다.')
        if sentence_indicators < 10:
            logger.debug(f"[BOOKTOKI] Too few sentences: {sentence_indicators}")
            return False
        
        logger.debug("[BOOKTOKI] Content validation passed")
        return True

    def _extract_raw_text(self, container):
        """Extract raw text when paragraph extraction fails."""
        logger.debug("[BOOKTOKI] Attempting raw text extraction")
        
        # Remove script and style elements
        for element in container.find_all(['script', 'style']):
            element.decompose()
        
        raw_text = container.get_text(separator='\n', strip=True)
        
        # Basic cleanup
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        valid_lines = [line for line in lines if self._is_valid_paragraph(line)]
        
        content = '\n\n'.join(valid_lines)
        logger.debug(f"[BOOKTOKI] Raw extraction yielded {len(content)} characters")
        
        return content if content else None

    def _try_alternative_extraction(self, soup):
        """Try alternative content extraction methods."""
        logger.debug("[BOOKTOKI] Trying alternative extraction methods")
        
        # Alternative selectors based on common Korean novel site patterns
        alternative_selectors = [
            '.content',
            '.chapter-content',
            '.novel-content', 
            '.story-content',
            '.text-content',
            '#content',
            '.view-padding',
            'article .content',
            '.entry-content'
        ]
        
        for selector in alternative_selectors:
            elements = soup.select(selector)
            for element in elements:
                content = self._extract_raw_text(element)
                if content and self._is_valid_content(content):
                    logger.info(f"[BOOKTOKI] Alternative selector '{selector}' found valid content")
                    return content
        
        # Log HTML structure for debugging
        log_html_structure(soup, self.url, "booktoki")
        find_content_candidates(soup, "booktoki")
        save_failed_html(soup, self.url, "booktoki", "content_extraction")
        
        logger.error("[BOOKTOKI] All content extraction methods failed")
        return None

    def get_next_link(self, soup, direction):
        """Get navigation links for next/previous chapters."""
        logger.debug(f"[BOOKTOKI] Looking for navigation links, direction: {direction}")
        
        # Navigation selectors based on HTML analysis
        nav_selectors = [
            'a[href*="/novel/"]',  # Direct novel links
            '.btn-group a[href]',  # Button group navigation
            '.pagination a[href]', # Pagination links
            'a[href*="chapter"]',  # Chapter links
        ]
        
        nav_links = []
        for selector in nav_selectors:
            links = soup.select(selector)
            if links:
                nav_links.extend(links)
                logger.debug(f"[BOOKTOKI] Found {len(links)} links with selector: {selector}")
        
        if not nav_links:
            logger.warning("[BOOKTOKI] No navigation links found")
            return None
        
        # Filter and analyze navigation links
        chapter_links = []
        current_chapter_id = self._extract_chapter_id_from_url(self.url)
        
        for link in nav_links:
            href = link.get('href')
            if not href or not '/novel/' in href:
                continue
                
            link_text = link.get_text(strip=True)
            link_chapter_id = self._extract_chapter_id_from_url(href)
            
            logger.debug(f"[BOOKTOKI] Nav link: '{link_text}' -> {href} (ID: {link_chapter_id})")
            
            if link_chapter_id and link_chapter_id != current_chapter_id:
                chapter_links.append({
                    'url': urljoin(self.url, href),
                    'text': link_text,
                    'chapter_id': link_chapter_id
                })
        
        if not chapter_links:
            logger.warning("[BOOKTOKI] No valid chapter navigation links found")
            return None
        
        # Sort by chapter ID to find next/previous
        chapter_links.sort(key=lambda x: x['chapter_id'])
        
        if direction == "Forwards (oldest to newest)":
            # Find next chapter (higher ID)
            for link in chapter_links:
                if link['chapter_id'] > current_chapter_id:
                    logger.debug(f"[BOOKTOKI] Found NEXT chapter: {link['url']}")
                    return link['url']
        else:  # Backwards
            # Find previous chapter (lower ID)
            for link in reversed(chapter_links):
                if link['chapter_id'] < current_chapter_id:
                    logger.debug(f"[BOOKTOKI] Found PREV chapter: {link['url']}")
                    return link['url']
        
        logger.warning(f"[BOOKTOKI] No suitable {direction} navigation link found")
        return None

    def _extract_chapter_id_from_url(self, url):
        """Extract chapter ID from URL for navigation."""
        match = re.search(r'/novel/(\d+)', url)
        return int(match.group(1)) if match else 0

    def parse_chapter_info(self, title, soup=None):
        """
        Parse Korean chapter titles to extract numbering information.
        
        Examples:
        - '던전 디펜스-2화' → (2, 2, "0002")
        - '던전 디펜스-10화' → (10, 10, "0010")
        """
        if not title:
            return None, None, None
        
        logger.debug(f"[BOOKTOKI] Parsing chapter info from title: '{title}'")
        
        # Match Korean chapter pattern: "제목-숫자화"
        match = re.search(r'(\d+)화', title)
        if not match:
            logger.warning(f"[BOOKTOKI] Could not find chapter number pattern in '{title}'")
            return None, None, None
        
        try:
            chapter_num = int(match.group(1))
            logger.debug(f"[BOOKTOKI] Parsed chapter number: {chapter_num}")
            return chapter_num, chapter_num, f"{chapter_num:04d}"
        except (ValueError, TypeError) as e:
            logger.error(f"[BOOKTOKI] Failed to convert chapter number from '{title}'. Error: {e}")
            return None, None, None