"""Adapter for novelcool.com."""
import re
from urllib.parse import urljoin
from .base_adapter import BaseAdapter
from .logging import logger

class NovelcoolAdapter(BaseAdapter):
    """Adapter for scraping novelcool.com."""

    def extract_title(self, soup):
        title_tag = soup.find('h2', class_='chapter-title')
        return title_tag.text.strip() if title_tag else None

    def extract_content(self, soup):
        content_div = soup.find('div', class_='chapter-reading-section')
        if content_div:
            return content_div.get_text(separator='\n', strip=True)
        return None

    def get_next_link(self, soup, direction):
        # The 'Next' link is typically the last link in the navigation buttons
        # The 'Prev' link is the one before it.
        nav_buttons = soup.select('.chapter-reading-page-btn a')
        if not nav_buttons:
            return None

        if direction == "Forwards (oldest to newest)":
            # The "Next" button is usually the last one
            next_link_tag = nav_buttons[-1]
            if "next" in next_link_tag.text.lower() and next_link_tag.get('href'):
                 return urljoin(self.url, next_link_tag['href'])
        else: # Backwards
            # The "Prev" button is usually the second to last
            if len(nav_buttons) > 1:
                prev_link_tag = nav_buttons[-2]
                if "prev" in prev_link_tag.text.lower() and prev_link_tag.get('href'):
                    return urljoin(self.url, prev_link_tag['href'])
        return None

    def parse_chapter_info(self, title):
        """
        Overrides the base parser to handle English chapter titles.
        Example: 'Chapter 10: A New Beginning'
        """
        if not title:
            return None, None, None
            
        match = re.search(r'Chapter\s+(\d+)', title, re.IGNORECASE)
        if match:
            try:
                number = int(match.group(1))
                return number, number, f"{number:04d}"
            except (ValueError, TypeError) as e:
                logger.error(f"Failed to convert number from title '{title}'. Error: {e}")
        
        logger.warning(f"Could not find a chapter number pattern in '{title}'.")
        return None, None, None