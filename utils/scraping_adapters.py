"""
Adapter pattern for web scraping.
"""
from abc import ABC, abstractmethod

class ScraperAdapter(ABC):
    """Abstract base class for scraper adapters."""

    def __init__(self, url):
        self.url = url

    @abstractmethod
    def get_encoding(self):
        """Return the character encoding of the site."""
        pass

    @abstractmethod
    def extract_title(self, soup):
        """Extract the chapter title from a BeautifulSoup object."""
        pass

    @abstractmethod
    def extract_chapter_number(self, soup):
        """Extract the chapter number from the title or page content."""
        pass

    @abstractmethod
    def extract_content(self, soup):
        """Extract the chapter content from a BeautifulSoup object."""
        pass

    @abstractmethod
    def get_next_link(self, soup, direction):
        """Extract the link to the next chapter."""
        pass
