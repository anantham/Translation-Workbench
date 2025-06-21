import os
import argparse
from ebooklib import epub
from bs4 import BeautifulSoup

def chapter_to_text(chapter):
    """Converts an EPUB chapter (which is HTML) to clean, plain text."""
    soup = BeautifulSoup(chapter.get_body_content(), 'html.parser')
    
    # Remove all script and style elements
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()
        
    # Get text, using a newline as a separator for block-level tags
    text = soup.get_text(separator='\n', strip=True)
    return text

def process_epub_file(epub_path: str, output_dir: str):
    """
    Reads an EPUB file, extracts each chapter, and saves it as a separate .txt file.
    """
    if not os.path.exists(epub_path):
        print(f"[ERROR] EPUB file not found at: {epub_path}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    try:
        book = epub.read_epub(epub_path)
    except Exception as e:
        print(f"[ERROR] Could not read or parse the EPUB file. It may be corrupted or in an unsupported format. Details: {e}")
        return

    print(f"Successfully opened '{epub_path}'. Processing chapters...")

    # Get items in the book's 'spine' (the main reading order)
    items = book.get_items_of_type(9)  # EBOOKLIB_NAMESPACE_XHTML
    
    chapter_count = 0
    for item in items:
        # We assume each 'item' in the spine is a chapter.
        # This is a common structure for EPUBs.
        content = chapter_to_text(item)
        
        # Skip very short "chapters" which are often title pages or ToCs
        if len(content) < 300:
            print(f"  -> Skipping short item '{item.get_name()}' (likely a title page).")
            continue
            
        chapter_count += 1
        # Use zfill to pad with zeros (e.g., 1 -> 0001) for correct file sorting
        filename = f"English-Chapter-{str(chapter_count).zfill(4)}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  [SUCCESS] Saved {filename}")

    print(f"\nProcessing complete. Extracted {chapter_count} chapters.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EPUB to Text Chapter Extractor")
    parser.add_argument("epub_file", type=str, help="The path to the .epub file to process.")
    args = parser.parse_args()

    output_directory_name = "english_chapters"
    
    process_epub_file(args.epub_file, output_directory_name)