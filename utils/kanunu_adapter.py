"""Adapter for kanunu8.com."""
import re
import cn2an
from urllib.parse import urljoin
from .base_adapter import BaseAdapter
from .logging import logger

class KanunuAdapter(BaseAdapter):
    """Adapter for scraping kanunu8.com."""

    def get_encoding(self):
        return 'gbk'

    def extract_title(self, soup):
        title_tag = soup.find('h1')
        return title_tag.text.strip() if title_tag else None

    def extract_content(self, soup):
        content_div = soup.find('div', id='neirong')
        if content_div:
            paragraphs = content_div.find_all('p')
            if len(paragraphs) > 1:
                paragraphs[-1].decompose()
            return content_div.get_text(separator='\n', strip=True)
        if soup.find('div', class_='mulu-list'):
            return None
        return None

    def get_next_link(self, soup, direction):
        link_text = re.compile(r'下一章') if direction == "Forwards (oldest to newest)" else re.compile(r'上一章')
        next_link_tag = soup.find('a', text=link_text)

        if not next_link_tag and direction == "Forwards (oldest to newest)":
            mulu_list = soup.find('div', class_='mulu-list')
            if mulu_list:
                first_chapter_link = mulu_list.find('a')
                if first_chapter_link and first_chapter_link.get('href'):
                    return urljoin(self.url, first_chapter_link['href'])

        if next_link_tag and next_link_tag.get('href'):
            return urljoin(self.url, next_link_tag['href'])
            
        return None

    def parse_chapter_info(self, title):
        """
        Overrides the base parser to handle combined chapters from kanunu8.
        Example: '永生 正文 第四十九~五十章 内门弟子'
        """
        match = re.search(r'第([一二三四五六七八九十百千万零\d~-]+)', title)
        if not match:
            logger.warning(f"Could not find a chapter number pattern in '{title}'.")
            return None, None, None

        numeral_part = match.group(1)
        range_match = re.match(r'(.+?)[~-](.+)', numeral_part)

        try:
            if range_match:
                start_numeral = range_match.group(1)
                end_numeral = range_match.group(2)

                start_int = int(cn2an.cn2an(start_numeral, "smart"))
                end_int = int(cn2an.cn2an(end_numeral, "smart"))

                # If end_int is smaller than start_int, it's an abbreviated range (e.g., 620~21)
                if end_int < start_int:
                    # Find the correct base to add by using powers of 10
                    power = 1
                    while power <= end_int:
                        power *= 10
                    base = (start_int // power) * power
                    corrected_end_int = base + end_int
                    
                    logger.info(f"Interpreted abbreviated range: {start_int}~{end_int} as {start_int}-{corrected_end_int}")
                    end_int = corrected_end_int

                return start_int, end_int, f"{start_int:04d}-{end_int:04d}"
            else:
                number = int(cn2an.cn2an(numeral_part, "smart"))
                return number, number, f"{number:04d}"
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to convert numeral '{numeral_part}' from title '{title}'. Error: {e}")
            return None, None, None