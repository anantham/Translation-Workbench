"""Adapter for shalvationtranslations.wordpress.com."""
import re
from urllib.parse import urljoin
from .base_adapter import BaseAdapter
from .logging import logger

class ShalvationTranslationsAdapter(BaseAdapter):
    """Adapter for scraping shalvationtranslations.wordpress.com."""

    def extract_title(self, soup):
        """Extract chapter title from WordPress post title."""
        # Try multiple title selectors common in WordPress
        title_selectors = [
            'h1.entry-title',
            'h1.post-title', 
            'h2.entry-title',
            '.entry-header h1',
            'article header h1',
            'h1'
        ]
        
        for selector in title_selectors:
            title_tag = soup.select_one(selector)
            if title_tag:
                title = title_tag.get_text(strip=True)
                logger.debug(f"[SHALVATION] Found title using '{selector}': {title}")
                return title
        
        # Fallback to page title
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)
            # Remove site name from title if present
            title = re.sub(r'\s*–\s*Shalvation Translations$', '', title)
            logger.debug(f"[SHALVATION] Using page title: {title}")
            return title
            
        logger.warning("[SHALVATION] Could not find title")
        return None

    def extract_content(self, soup):
        """Extract chapter content from WordPress entry content."""
        logger.debug("[SHALVATION] Starting content extraction")
        
        # WordPress content selectors (ordered by specificity)
        content_selectors = [
            '.entry-content',      # Most common WordPress content class
            '.post-content',       # Alternative content class
            'article .content',    # Article content
            '.content',            # Generic content class
            'main article',       # Main article content
            '[role="main"] article'  # Accessible main content
        ]
        
        for selector in content_selectors:
            content_container = soup.select_one(selector)
            if content_container:
                logger.debug(f"[SHALVATION] Found content using selector: {selector}")
                
                # Clean the content (create a copy using str conversion and re-parsing)
                import copy
                cleaned_content = self._clean_content_container(copy.deepcopy(content_container))
                content_text = cleaned_content.get_text(separator='\n', strip=True)
                
                # Validate content quality
                if self._is_valid_content(content_text):
                    logger.debug(f"[SHALVATION] Extracted {len(content_text)} characters")
                    return content_text
                else:
                    logger.debug(f"[SHALVATION] Content validation failed for selector: {selector}")
                    continue
        
        logger.error("[SHALVATION] No valid content found with any selector")
        return None

    def _clean_content_container(self, container):
        """Remove navigation, metadata, and other non-content elements."""
        # Remove navigation elements
        for nav_element in container.select('[class*="nav"], [class*="pagination"], nav'):
            nav_element.decompose()
        
        # Remove share buttons and social media
        for share_element in container.select('[class*="share"], [class*="social"], .sharedaddy'):
            share_element.decompose()
        
        # Remove like buttons and interaction elements
        for like_element in container.select('[class*="like"], [class*="rating"], .wp-like'):
            like_element.decompose()
        
        # Remove comments section
        for comment_element in container.select('[class*="comment"], #comments, .comments'):
            comment_element.decompose()
        
        # Remove meta information (tags, categories, dates)
        for meta_element in container.select('[class*="meta"], [class*="tag"], [class*="cat"], .entry-footer'):
            meta_element.decompose()
        
        # Remove navigation links (NEXT CHAPTER, PREVIOUS CHAPTER)
        for link in container.find_all('a', href=True):
            link_text = link.get_text().lower().strip()
            if any(nav_word in link_text for nav_word in ['next chapter', 'previous chapter', 'prev chapter']):
                # Keep the link but remove it from content extraction
                link.decompose()
        
        # Remove translator notes if they're clearly marked
        for note_element in container.select('[class*="note"], [class*="tl-note"], .translator-note'):
            note_element.decompose()
        
        # Remove any remaining scripts or style elements
        for script in container.select('script, style'):
            script.decompose()
        
        return container

    def _is_valid_content(self, content):
        """Check if extracted content looks like actual chapter text."""
        if not content or len(content.strip()) < 200:  # Chapters should be substantial
            return False
        
        content_lower = content.lower()
        
        # Check for navigation artifacts that shouldn't be in chapter content
        nav_indicators = [
            'next chapter', 'previous chapter', 'table of contents',
            'click to share', 'like loading', 'reblogged this',
            'leave a comment', 'wordpress.com'
        ]
        
        # If content starts with navigation indicators, it's not clean
        first_200_chars = content[:200].lower()
        if any(indicator in first_200_chars for indicator in nav_indicators):
            logger.debug("[SHALVATION] Content validation failed: navigation artifacts in beginning")
            return False
        
        # Content should have narrative structure (multiple sentences/paragraphs)
        sentence_count = content.count('.') + content.count('!') + content.count('?')
        if sentence_count < 10:  # Chapters should have multiple sentences
            logger.debug(f"[SHALVATION] Content validation failed: too few sentences ({sentence_count})")
            return False
        
        # Check for reasonable paragraph structure
        paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
        if len(paragraphs) < 3:  # Should have multiple paragraphs
            logger.debug(f"[SHALVATION] Content validation failed: too few paragraphs ({len(paragraphs)})")
            return False
        
        return True

    def get_next_link(self, soup, direction):
        """Find navigation links for next/previous chapters."""
        logger.debug(f"[SHALVATION] Looking for navigation links, direction: {direction}")
        
        # Look for links with specific navigation text
        nav_link_patterns = {
            "forwards": ["next chapter", "siguiente", "suivant", "下一"],
            "backwards": ["previous chapter", "prev chapter", "anterior", "précédent", "上一"]
        }
        
        # Determine which patterns to look for
        if direction == "Forwards (oldest to newest)":
            patterns = nav_link_patterns["forwards"]
        else:
            patterns = nav_link_patterns["backwards"]
        
        # Find all links on the page
        all_links = soup.find_all('a', href=True)
        logger.debug(f"[SHALVATION] Found {len(all_links)} total links on page")
        
        # Search for navigation links
        for link in all_links:
            link_text = link.get_text().lower().strip()
            link_href = link.get('href', '')
            
            # Check if this link matches our navigation patterns
            for pattern in patterns:
                if pattern in link_text:
                    # Validate that this is a chapter link
                    if 'chapter' in link_href.lower() or 'shalvationtranslations' in link_href:
                        nav_url = urljoin(self.url, link_href)
                        logger.debug(f"[SHALVATION] Found {direction} link: '{link_text}' -> {nav_url}")
                        return nav_url
        
        # Alternative approach: look for links that contain "chapter" in URL
        for link in all_links:
            link_href = link.get('href', '')
            link_text = link.get_text().lower().strip()
            
            if 'chapter' in link_href.lower() and 'shalvationtranslations' in link_href:
                # Try to determine if this is next or previous based on context
                if direction == "Forwards (oldest to newest)" and any(word in link_text for word in ['next', '2', 'two']):
                    nav_url = urljoin(self.url, link_href)
                    logger.debug(f"[SHALVATION] Found potential {direction} link: '{link_text}' -> {nav_url}")
                    return nav_url
                elif direction != "Forwards (oldest to newest)" and any(word in link_text for word in ['prev', 'previous']):
                    nav_url = urljoin(self.url, link_href)
                    logger.debug(f"[SHALVATION] Found potential {direction} link: '{link_text}' -> {nav_url}")
                    return nav_url
        
        logger.warning(f"[SHALVATION] No suitable {direction} navigation link found")
        return None

    def parse_chapter_info(self, title, soup=None):
        """
        Parse chapter information from Shalvation Translations titles.
        Examples: 
        - 'Dungeon Defense (WN): Chapter 001 – Prologue' → (1, 1, "0001")
        - 'Dungeon Defense (WN): Chapter 010 – Title Name' → (10, 10, "0010")
        """
        if not title:
            return None, None, None
        
        # Pattern for Shalvation Translations chapter titles
        # Matches: "Chapter 001", "Chapter 010", etc.
        match = re.search(r'Chapter\s+(\d+)', title, re.IGNORECASE)
        if not match:
            logger.warning(f"[SHALVATION] Could not find chapter number pattern in '{title}'.")
            return None, None, None
        
        try:
            chapter_number = int(match.group(1))
            logger.debug(f"[SHALVATION] Parsed chapter number: {chapter_number}")
            return chapter_number, chapter_number, f"{chapter_number:04d}"
        except (ValueError, TypeError) as e:
            logger.error(f"[SHALVATION] Failed to convert chapter number from title '{title}'. Error: {e}")
            return None, None, None