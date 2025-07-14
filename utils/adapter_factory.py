"""
Factory for creating scraper adapters.
"""
from .dxmwx_adapter import DxmwxAdapter
from .kanunu_adapter import KanunuAdapter
from .novelcool_adapter import NovelcoolAdapter

def get_adapter(url):
    """Return the appropriate scraper adapter for the given URL."""
    if "dxmwx.org" in url:
        return DxmwxAdapter(url)
    elif "kanunu8.com" in url or "kanunu.net" in url:
        return KanunuAdapter(url)
    elif "novelcool.com" in url:
        return NovelcoolAdapter(url)
    # Add more adapters here as they are created
    return None
