
# Adapter for dxmwx.org.

import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from utils.scraping_adapters import ScraperAdapter

class DxmwxAdapter(ScraperAdapter):
    """Adapter for scraping dxmwx.org."""

    def get_encoding(self):
        return 'utf-8'

    def extract_title(self, soup):
        title_tag = soup.select_one("#ChapterTitle")
        return title_tag.text.strip() if title_tag else None

    def extract_content(self, soup):
        content_tag = soup.select_one("#Lab_Contents")
        if not content_tag:
            return None
        
        for tag in content_tag.select('script, a'):
            tag.decompose()
        
        return content_tag.get_text(separator='\n', strip=True)

    def get_next_link(self, soup, direction):
        if direction == "Forwards (oldest to newest)":
            link_text = re.compile(r'下一章')
        else:
            link_text = re.compile(r'上一章')
        
        next_link_tag = soup.find('a', string=link_text)
        if next_link_tag and next_link_tag.get('href'):
            return urljoin(self.url, next_link_tag['href'])
        return None

