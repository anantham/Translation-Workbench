"""
Factory for creating scraper adapters.
"""
from .dxmwx_adapter import DxmwxAdapter
from .kanunu_adapter import KanunuAdapter
from .novelcool_adapter import NovelcoolAdapter
from .kakuyomu_adapter import KakuyomuAdapter
from .shalvation_adapter import ShalvationTranslationsAdapter
from .booktoki_adapter import BookTokiAdapter

def get_adapter(url):
    """Return the appropriate scraper adapter for the given URL."""
    if "dxmwx.org" in url:
        return DxmwxAdapter(url)
    elif "kanunu8.com" in url or "kanunu.net" in url:
        return KanunuAdapter(url)
    elif "novelcool.com" in url:
        return NovelcoolAdapter(url)
    elif "kakuyomu.jp" in url:
        return KakuyomuAdapter(url)
    elif "shalvationtranslations.wordpress.com" in url:
        return ShalvationTranslationsAdapter(url)
    elif "booktoki" in url:
        return BookTokiAdapter(url)
    # Add more adapters here as they are created
    return None
