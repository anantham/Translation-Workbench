

"""Adapter for novelcool.com."""
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from utils.scraping_adapters import ScraperAdapter

class NovelcoolAdapter(ScraperAdapter):
    """Adapter for scraping novelcool.com."""

    def get_encoding(self):
        return 'utf-8'

    def extract_title(self, soup):
        title_tag = soup.find('h2', class_='chapter-title')
        return title_tag.text.strip() if title_tag else None

    def extract_chapter_number(self, soup):
        """
        Extracts the chapter number from the title for novelcool.
        Example title: 'Chapter 10: Title'
        """
        import re
        title = self.extract_title(soup)
        if title:
            match = re.search(r'Chapter\s+(\d+)', title, re.IGNORECASE)
            if match:
                return int(match.group(1)), f"{int(match.group(1)):04d}"
        return None, None

    def extract_content(self, soup):
        content_div = soup.find('div', class_='chapter-reading-section')
        if content_div:
            return content_div.get_text(separator='\n', strip=True)
        return None

    def get_next_link(self, soup, direction):
        if direction == "Forwards (oldest to newest)":
            # Find the 'Next>>' link
            next_link_tag = soup.find('div', class_='chapter-reading-pageitem')
            if next_link_tag:
                link = next_link_tag.find('a')
                if link and link.has_attr('href'):
                    return urljoin(self.url, link['href'])
        else: # Backwards
            # Find the '<<Prev' link
            prev_link_tag = soup.find('div', class_='chapter-reading-pageitem')
            if prev_link_tag:
                link = prev_link_tag.find_previous('div', class_='chapter-reading-pageitem').find('a')
                if link and link.has_attr('href'):
                    return urljoin(self.url, link['href'])
        return None

