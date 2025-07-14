"""
Base Adapter for web scraping.
"""
import re
import cn2an
from utils.logging import logger

class BaseAdapter:
    """A base class for all website-specific scraping adapters."""

    def __init__(self, url):
        self.url = url

    def get_encoding(self):
        """Returns the character encoding for the website."""
        return 'utf-8'

    def extract_title(self, soup):
        """Extracts the novel title from the page soup."""
        raise NotImplementedError

    def extract_content(self, soup):
        """Extracts the chapter content from the page soup."""
        raise NotImplementedError

    def get_next_link(self, soup, direction):
        """Gets the URL for the next chapter."""
        raise NotImplementedError

    def parse_chapter_info(self, title):
        """
        Parses the chapter title to extract numbering information.

        This base implementation handles simple cases with a single chapter number.
        Adapters for sites with more complex titles (like combined chapters)
        should override this method.

        Args:
            title (str): The chapter title string.

        Returns:
            tuple[int, int, str] | tuple[None, None, None]: A tuple containing:
                - The starting integer of the chapter (for sequence validation).
                - The ending integer of the chapter (for updating the sequence counter).
                - A string representation for the filename (e.g., "0049").
            Returns (None, None, None) if no number is found.
        """
        match = re.search(r'第(\S+)', title)
        if not match:
            logger.warning(f"Could not find a chapter number pattern in '{title}'.")
            return None, None, None

        numeral_part = match.group(1)
        try:
            # Using a more specific regex to extract only the numeral part before "章"
            numeral_only = re.search(r'([一二三四五六七八九十百千万\d]+)', numeral_part)
            if not numeral_only:
                logger.warning(f"Could not extract numeral from '{numeral_part}'.")
                return None, None, None

            number = int(cn2an.cn2an(numeral_only.group(1), "smart"))
            return number, number, f"{number:04d}"
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to convert Chinese numeral from title '{title}'. Error: {e}")
            return None, None, None
