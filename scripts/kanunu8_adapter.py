"""
Kanunu8.com adapter for the translation workbench scraper
Based on encoding analysis and typical kanunu8 HTML patterns
"""

import re
from bs4 import BeautifulSoup

class Kanunu8Adapter:
    """Adapter for scraping kanunu8.com Chinese novel site"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def get_encoding(self, response):
        """Kanunu8 uses GB2312/GBK encoding"""
        # Try encodings in order of likelihood
        for encoding in ['gb2312', 'gbk', 'gb18030']:
            try:
                response.encoding = encoding
                # Quick validation - check if we get proper Chinese
                if '第' in response.text and '章' in response.text:
                    return encoding
            except:
                continue
        
        # Fallback to most comprehensive Chinese encoding
        response.encoding = 'gb18030'
        return 'gb18030'
    
    def validate_page(self, soup):
        """Check if page has valid chapter content"""
        # Look for chapter title pattern
        if soup.find(text=re.compile(r'第[一二三四五六七八九十百千万\d]+章')):
            return True, "Chapter content found"
        
        # Also check for common kanunu8 markers
        if soup.find('title') and '看努努' in soup.find('title').text:
            # At least we're on the right site
            return True, "Kanunu8 page validated"
        
        return False, "No valid chapter content found"
    
    def extract_title(self, soup):
        """Extract chapter title from various possible locations"""
        # Method 1: Direct text search for chapter pattern
        chapter_pattern = soup.find(text=re.compile(r'第[一二三四五六七八九十百千万\d]+章[^<]*'))
        if chapter_pattern:
            # Get the full text including chapter name
            full_title = chapter_pattern.strip()
            # Sometimes the title continues in the next sibling
            parent = chapter_pattern.parent
            if parent and parent.next_sibling:
                next_text = str(parent.next_sibling).strip()
                if next_text and not next_text.startswith('<'):
                    full_title += ' ' + next_text
            return full_title
        
        # Method 2: Check in table cells (common pattern)
        for td in soup.find_all('td'):
            if td.text and re.search(r'第[一二三四五六七八九十百千万\d]+章', td.text):
                return td.text.strip()
        
        # Method 3: Check title tag
        title_tag = soup.find('title')
        if title_tag and '第' in title_tag.text and '章' in title_tag.text:
            # Extract just the chapter part
            match = re.search(r'第[一二三四五六七八九十百千万\d]+章[^_-]*', title_tag.text)
            if match:
                return match.group().strip()
        
        return "Unknown Chapter"
    
    def extract_content(self, soup):
        """Extract main chapter content"""
        # First, remove the title to avoid duplication
        title_pattern = soup.find(text=re.compile(r'第[一二三四五六七八九十百千万\d]+章'))
        if title_pattern and title_pattern.parent:
            title_elem = title_pattern.parent
            # Store title's parent for reference
            title_parent = title_elem.parent
            title_elem.extract()
        else:
            title_parent = None
        
        # Strategy 1: Find the largest continuous text block
        # This works well for kanunu8's simple layout
        candidates = []
        
        for elem in soup.find_all(['td', 'div', 'p']):
            # Skip navigation elements
            if elem.find_all('a') and len(elem.find_all('a')) > 2:
                continue
            
            text = elem.get_text(separator='\n', strip=True)
            # Look for substantial content (novel text is usually long)
            if len(text) > 500:
                candidates.append((elem, len(text)))
        
        if candidates:
            # Sort by length and take the longest
            candidates.sort(key=lambda x: x[1], reverse=True)
            content_elem = candidates[0][0]
            
            # Clean up the content
            # Remove script and style tags
            for tag in content_elem.find_all(['script', 'style']):
                tag.decompose()
            
            # Remove navigation links but keep the text
            for link in content_elem.find_all('a'):
                link.unwrap()
            
            return content_elem.get_text(separator='\n', strip=True)
        
        # Strategy 2: If no large block found, get all text after title
        # Remove common navigation patterns
        all_text = soup.get_text(separator='\n', strip=True)
        
        # Remove common kanunu8 navigation text
        nav_patterns = [
            r'上一[章页]',
            r'下一[章页]', 
            r'返回目录',
            r'看努努',
            r'www\.kanunu\d*\.com'
        ]
        
        for pattern in nav_patterns:
            all_text = re.sub(pattern, '', all_text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        lines = [line.strip() for line in all_text.split('\n') if line.strip()]
        return '\n'.join(lines)
    
    def get_navigation_link(self, soup, direction='previous'):
        """Extract navigation links"""
        # Kanunu8 uses various patterns for navigation
        if direction == 'previous':
            patterns = [r'上一章', r'上一页', r'←', r'prev']
        else:
            patterns = [r'下一章', r'下一页', r'→', r'next']
        
        for pattern in patterns:
            # Try text-based search first
            link = soup.find('a', text=re.compile(pattern, re.IGNORECASE))
            if link and link.get('href'):
                return link['href']
            
            # Try searching in href attribute
            link = soup.find('a', href=re.compile(r'\d+\.html?$'))
            if link:
                # Check if it's the right direction by comparing numbers
                current_url = soup.find('link', {'rel': 'canonical'})
                if current_url:
                    current_num = re.search(r'/(\d+)\.html?$', current_url.get('href', ''))
                    link_num = re.search(r'/(\d+)\.html?$', link['href'])
                    
                    if current_num and link_num:
                        current = int(current_num.group(1))
                        target = int(link_num.group(1))
                        
                        if direction == 'previous' and target < current:
                            return link['href']
                        elif direction == 'next' and target > current:
                            return link['href']
        
        return None
    
    def extract_chapter_number(self, title):
        """Extract chapter number from title - reuse from main scraper"""
        # This is the same logic as the main scraper
        match = re.search(r'第([一二三四五六七八九零十百千万\d]+)章', title)
        if not match:
            return None
        
        ch_text = match.group(1).strip()
        
        if ch_text.isdigit():
            return int(ch_text)
        
        # Convert Chinese numerals (would import from main scraper)
        return chinese_to_int(ch_text)  # Assumes chinese_to_int is available
