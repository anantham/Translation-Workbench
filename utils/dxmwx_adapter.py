"""Adapter for dxmwx.org."""
import re
from urllib.parse import urljoin
from .base_adapter import BaseAdapter

class DxmwxAdapter(BaseAdapter):
    """Adapter for scraping dxmwx.org."""

    def extract_title(self, soup):
        title_tag = soup.select_one("#ChapterTitle")
        return title_tag.text.strip() if title_tag else None

    def extract_content(self, soup):
        content_tag = soup.select_one("#Lab_Contents")
        if not content_tag:
            return None
        
        # Remove script and ad link tags
        for tag in content_tag.select('script, a'):
            tag.decompose()
        
        return content_tag.get_text(separator='\n', strip=True)

    def get_next_link(self, soup, direction):
        link_text = re.compile(r'下一章') if direction == "Forwards (oldest to newest)" else re.compile(r'上一章')
        
        next_link_tag = soup.find('a', string=link_text)
        if next_link_tag and next_link_tag.get('href'):
            return urljoin(self.url, next_link_tag['href'])
        return None