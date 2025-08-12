"""Adapter for kakuyomu.jp."""
import re
import cn2an
from urllib.parse import urljoin
from .base_adapter import BaseAdapter
from .logging import logger

class KakuyomuAdapter(BaseAdapter):
    """Adapter for scraping kakuyomu.jp."""

    def extract_title(self, soup):
        """Extract chapter title from page title tag."""
        title_tag = soup.find('title')
        if title_tag:
            # Format: "第二十話　最強の陰陽師、墓穴を掘る - [series title] - カクヨム"
            full_title = title_tag.get_text()
            # Extract just the chapter title (first part before " - ")
            chapter_title = full_title.split(' - ')[0].strip()
            logger.debug(f"[KAKUYOMU] Extracted title: '{chapter_title}'")
            return chapter_title
        return None

    def extract_content(self, soup):
        """Extract chapter content from the main content container."""
        logger.debug("[KAKUYOMU] Starting content extraction")
        
        # Primary content container
        content_tag = soup.select_one('#contentMain')
        if content_tag:
            # Remove any navigation elements that might be in the content
            for nav_element in content_tag.select('a[href*="episodes"]'):
                nav_text = nav_element.get_text().strip()
                if '前のエピソード' in nav_text or '次のエピソード' in nav_text:
                    nav_element.decompose()
            
            # Extract content
            content = content_tag.get_text(separator='\n', strip=True)
            logger.debug(f"[KAKUYOMU] Extracted {len(content)} characters")
            
            # Validation - should be substantial content
            if len(content) >= 500:
                # Clean up content - remove navigation text that might remain
                lines = content.split('\n')
                cleaned_lines = []
                for line in lines:
                    line = line.strip()
                    # Skip navigation lines
                    if (line.startswith('前のエピソード') or 
                        line.startswith('次のエピソード') or
                        line == '表示設定' or
                        line == '目次' or
                        not line):
                        continue
                    cleaned_lines.append(line)
                
                cleaned_content = '\n'.join(cleaned_lines)
                logger.debug(f"[KAKUYOMU] Cleaned content: {len(cleaned_content)} characters")
                return cleaned_content
            else:
                logger.warning(f"[KAKUYOMU] Content too short: {len(content)} characters")
                
        logger.error("[KAKUYOMU] Content container #contentMain not found")
        return None

    def get_next_link(self, soup, direction):
        """Get navigation links from HTML head link tags."""
        logger.debug(f"[KAKUYOMU] Looking for navigation links, direction: {direction}")
        
        if direction == "Forwards (oldest to newest)":
            # Look for next episode link
            next_link = soup.find('link', rel='next')
            if next_link and next_link.get('href'):
                next_url = urljoin(self.url, next_link['href'])
                logger.debug(f"[KAKUYOMU] Found NEXT link: {next_url}")
                return next_url
        else:  # Backwards
            # Look for previous episode link
            prev_link = soup.find('link', rel='prev')
            if prev_link and prev_link.get('href'):
                prev_url = urljoin(self.url, prev_link['href'])
                logger.debug(f"[KAKUYOMU] Found PREV link: {prev_url}")
                return prev_url
        
        logger.warning(f"[KAKUYOMU] No {direction} navigation link found")
        return None

    def parse_chapter_info(self, title, soup=None):
        """
        Parse Japanese chapter titles to extract numbering information.
        Examples: 
        - '第二十話　最強の陰陽師、墓穴を掘る' → (20, 20, "0020")
        - '第一話　異世界転生' → (1, 1, "0001")
        """
        if not title:
            logger.warning("[KAKUYOMU] No title provided for chapter parsing")
            return None, None, None
            
        # Match Japanese chapter pattern: 第[numeral]話
        match = re.search(r'第([一二三四五六七八九十百千万零\d]+)話', title)
        if not match:
            logger.warning(f"[KAKUYOMU] Could not find chapter pattern '第X話' in '{title}'")
            return None, None, None
            
        numeral_part = match.group(1)
        logger.debug(f"[KAKUYOMU] Found chapter numeral: '{numeral_part}'")
        
        try:
            # Convert Japanese numerals to integers using cn2an
            # cn2an handles both Chinese and Japanese numerals
            if re.match(r'^\d+$', numeral_part):
                # Already Arabic numerals
                number = int(numeral_part)
            else:
                # Handle special cases for cn2an
                normalized_numeral = numeral_part
                
                # Fix standalone 百 (hundred) - cn2an needs 一百
                if normalized_numeral == '百':
                    normalized_numeral = '一百'
                # Fix standalone 千 (thousand) - cn2an needs 一千
                elif normalized_numeral == '千':
                    normalized_numeral = '一千'
                # Fix standalone 万 (ten thousand) - cn2an needs 一万  
                elif normalized_numeral == '万':
                    normalized_numeral = '一万'
                
                # Convert Japanese/Chinese numerals
                number = int(cn2an.cn2an(normalized_numeral, "smart"))
            
            logger.debug(f"[KAKUYOMU] Converted '{numeral_part}' to {number}")
            return number, number, f"{number:04d}"
            
        except (ValueError, TypeError) as e:
            logger.error(f"[KAKUYOMU] Failed to convert numeral '{numeral_part}' from title '{title}'. Error: {e}")
            return None, None, None