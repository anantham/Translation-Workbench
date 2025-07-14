
"""Adapter for kanunu8.com."""
import re
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from utils.scraping_adapters import ScraperAdapter
from .chinese_numerals import extract_and_convert_chinese_numeral

class KanunuAdapter(ScraperAdapter):
    """Adapter for scraping kanunu8.com."""

    def get_encoding(self):
        return 'gbk'

    def extract_title(self, soup):
        title_tag = soup.find('h1')
        return title_tag.text.strip() if title_tag else None

    def extract_chapter_number(self, soup):
        """
        Extracts the chapter number from the title for kanunu8.
        Example title: '永生 正文 第一章 肉身十重'
        """
        title = self.extract_title(soup)
        if title:
            return extract_and_convert_chinese_numeral(title)
        return None

    def extract_content(self, soup):
        # For chapter pages, content is in a div with id 'neirong'
        content_div = soup.find('div', id='neirong')
        if content_div:
            # Remove the last p element which contains navigation links
            paragraphs = content_div.find_all('p')
            if len(paragraphs) > 1:
                paragraphs[-1].decompose()
            return content_div.get_text(separator='\n', strip=True)

        # For index pages, there is no main content, so we return None
        if soup.find('div', class_='mulu-list'):
            return None
            
        return None

    def get_next_link(self, soup, direction):
        if direction == "Forwards (oldest to newest)":
            link_text = re.compile(r'下一章')
        else:
            link_text = re.compile(r'上一章')

        next_link_tag = soup.find('a', text=link_text)
        
        # On the main index page, the "next chapter" link might not exist.
        # We need to find the first chapter link in the list.
        if not next_link_tag and direction == "Forwards (oldest to newest)":
            mulu_list = soup.find('div', class_='mulu-list')
            if mulu_list:
                first_chapter_link = mulu_list.find('a')
                if first_chapter_link and first_chapter_link.get('href'):
                    return urljoin(self.url, first_chapter_link['href'])

        if next_link_tag and next_link_tag.get('href'):
            # Handle both relative and absolute URLs
            return urljoin(self.url, next_link_tag['href'])
            
        return None
