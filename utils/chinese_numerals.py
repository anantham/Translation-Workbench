"""
Utility for converting Chinese numerals in text to integers.
"""
import re
import cn2an
from .logging import logger

def extract_and_convert_chinese_numeral(text):
    """
    Finds a Chinese chapter number in a string, extracts the numeral part,
    and converts it to an integer. Handles single numbers and ranges,
    including abbreviated ranges like "八十五~六章" (85-6).

    Args:
        text (str): The string containing the Chinese chapter number.

    Returns:
        tuple[int, str] | tuple[None, None]: A tuple containing:
            - The last integer in the sequence (for validation).
            - A string representation for the filename (e.g., "0085_0086").
        Returns (None, None) if no number is found or conversion fails.
    """
    # Regex to find "第" followed by characters and ending with "章"
    match = re.search(r'第(.+?)章', text)
    if not match:
        logger.warning(f"Could not find a chapter number pattern in '{text}'.")
        return None, None

    numeral_part = match.group(1)

    # Check for a range pattern (e.g., "四十九~五十" or "八十五~六")
    range_match = re.match(r'(.+?)[~-](.+)', numeral_part)

    try:
        if range_match:
            start_numeral = range_match.group(1)
            end_numeral = range_match.group(2)

            start_int = int(cn2an.cn2an(start_numeral, "smart"))
            end_int = int(cn2an.cn2an(end_numeral, "smart"))

            # --- Abbreviated Range Logic ---
            # If end number is smaller, it's likely an abbreviation (e.g., 85~6 -> 85-86)
            if end_int < start_int and end_int < 10:
                # Assume the end number belongs to the same "tens" group
                # e.g., for 85, base is 80. for 185, base is 180.
                base = (start_int // 10) * 10
                end_int = base + end_int
                logger.info(f"Interpreted abbreviated range: {start_int}~{end_numeral} as {start_int}-{end_int}")

            return end_int, f"{start_int:04d}_{end_int:04d}"
        else:
            # Handle as a single number
            number = int(cn2an.cn2an(numeral_part, "smart"))
            return number, f"{number:04d}"
            
    except (ValueError, TypeError) as e:
        logger.error(f"Failed to convert Chinese numeral '{numeral_part}' "
                     f"from title '{text}'. Error: {e}")
        return None, None