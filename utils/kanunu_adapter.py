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

    def _normalize_chapter_numeral(self, numeral):
        """Normalizes specific non-standard Chinese numerals.
        This is only called as a fallback if the standard parser fails.
        """
        # Fix for "一千一十" (1010) -> "一千零一十"
        if re.fullmatch(r'一千[一二三四五六七八九]十', numeral):
            logger.debug(f"Normalizing numeral '{numeral}' by adding '零'.")
            return numeral[:2] + '零' + numeral[2:]
        
        # Add other normalization rules here if needed
        
        return numeral # Return original if no rule matches

    def parse_chapter_info(self, title, soup):
        """Overrides the base parser to handle combined chapters from kanunu8.
        It prioritizes the on-page content text over the H1 or HTML title tag,
        as those are sometimes incorrect.
        """
        numeral_part = None
        
        # 1. Prioritize the main content div, as it's the most reliable source.
        content_div = soup.find('div', id='neirong')
        if content_div:
            content_text = content_div.get_text(separator="\n", strip=True)
            match = re.search(r'第([一二三四五六七八九十百千万零\d~-]+)章', content_text)
            if match:
                numeral_part = match.group(1)
                logger.debug(f"Found chapter numeral '{numeral_part}' in content div.")

        # 2. If not in content, check the H1 tag as a fallback.
        if not numeral_part:
            body_match_h1 = soup.find('h1', string=re.compile(r"第[一二三四五六七八九十百千万零\d~-]+章"))
            if body_match_h1:
                match = re.search(r'第([一二三四五六七八九十百千万零\d~-]+)章', body_match_h1.text)
                if match:
                    numeral_part = match.group(1)
                    logger.debug(f"Found chapter numeral '{numeral_part}' in H1 tag.")

        # 3. If still not found, fall back to the HTML title.
        if not numeral_part:
            logger.warning("Could not find chapter number in body, falling back to title tag.")
            match = re.search(r'第([一二三四五六七八九十百千万零\d~-]+)', title)
            if match:
                numeral_part = match.group(1)

        if not numeral_part:
            logger.error(f"Could not find a chapter number pattern in body or title: '{title}'.")
            return None, None, None
        range_match = re.match(r'(.+?)[~-](.+)', numeral_part)

        def convert_numeral(numeral_str):
            """Tries standard conversion, then falls back to normalization."""
            try:
                # First attempt: standard conversion
                return int(cn2an.cn2an(numeral_str, "smart"))
            except (ValueError, TypeError):
                logger.warning(f"Standard conversion failed for '{numeral_str}'. Trying normalization.")
                # Second attempt: normalize and convert
                normalized_numeral = self._normalize_chapter_numeral(numeral_str)
                if normalized_numeral != numeral_str:
                    return int(cn2an.cn2an(normalized_numeral, "smart"))
                else:
                    # If normalization didn't change anything, re-raise the error
                    raise

        try:
            if range_match:
                start_numeral_str = range_match.group(1)
                end_numeral_str = range_match.group(2)

                start_int = convert_numeral(start_numeral_str)
                end_int = convert_numeral(end_numeral_str)

                # If end_int is smaller than start_int, it's an abbreviated range (e.g., 620~21)
                if end_int < start_int:
                    power = 1
                    while power <= end_int:
                        power *= 10
                    base = (start_int // power) * power
                    corrected_end_int = base + end_int
                    
                    logger.info(f"Interpreted abbreviated range: {start_int}~{end_int} as {start_int}-{corrected_end_int}")
                    end_int = corrected_end_int

                return start_int, end_int, f"{start_int:04d}-{end_int:04d}"
            else:
                number = convert_numeral(numeral_part)
                return number, number, f"{number:04d}"
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to convert numeral '{numeral_part}' from title '{title}'. Error: {e}")
            return None, None, None