#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Chinese numeral regex pattern
"""

import re

def test_chinese_numeral_pattern():
    # The pattern to test
    pattern = r'一千三百[一二三四五六七八九]十[一二三四五六七八九]?'
    
    # The test string
    test_string = '一千三百十二'
    
    print("=== Chinese Numeral Regex Test ===")
    print(f"Pattern: {pattern}")
    print(f"Test String: {test_string}")
    print()
    
    # Test with re.fullmatch()
    match_result = re.fullmatch(pattern, test_string)
    print(f"re.fullmatch() result: {match_result}")
    
    if match_result:
        print("✅ MATCH FOUND!")
        print(f"Matched text: '{match_result.group()}'")
    else:
        print("❌ NO MATCH")
    
    print()
    print("=== Pattern Breakdown ===")
    print("一千三百 - literal characters (one thousand three hundred)")
    print("[一二三四五六七八九] - one digit from 1-9")
    print("十 - literal character (ten)")
    print("[一二三四五六七八九]? - optional digit from 1-9")
    print()
    
    print("=== Analysis ===")
    print("Test string breakdown:")
    print("一千三百 - matches literal part ✅")
    print("十 - this is where the problem is! ❌")
    print("二 - this would be the optional digit")
    print()
    print("❗ ISSUE IDENTIFIED:")
    print("The pattern expects a digit [一二三四五六七八九] between '一千三百' and '十',")
    print("but the test string '一千三百十二' has '十' immediately after '一千三百'.")
    print()
    print("In Chinese numerals:")
    print("- 一千三百十二 = 1312 (literally: one thousand three hundred ten two)")
    print("- The pattern expects: 一千三百X十Y (1300 + X*10 + Y)")
    print("- But the string is: 一千三百十二 (1300 + 0*10 + 12)")
    print()
    print("To fix this, the digit before '十' should be optional:")
    print("Suggested pattern: r'一千三百[一二三四五六七八九]?十[一二三四五六七八九]?'")
    
    # Test the corrected pattern
    print()
    print("=== Testing Corrected Pattern ===")
    corrected_pattern = r'一千三百[一二三四五六七八九]?十[一二三四五六七八九]?'
    corrected_match = re.fullmatch(corrected_pattern, test_string)
    print(f"Corrected pattern: {corrected_pattern}")
    print(f"Result: {corrected_match}")
    
    if corrected_match:
        print("✅ CORRECTED PATTERN MATCHES!")
    else:
        print("❌ Still no match - further analysis needed")

    # Test some additional examples
    print()
    print("=== Additional Test Cases ===")
    test_cases = [
        '一千三百十二',    # 1312 - original test
        '一千三百二十三',  # 1323 - has digit before 十
        '一千三百九十',    # 1390 - no digit after 十
        '一千三百五十五',  # 1355 - both digits present
    ]
    
    for test_case in test_cases:
        original_match = re.fullmatch(pattern, test_case)
        corrected_match = re.fullmatch(corrected_pattern, test_case)
        print(f"'{test_case}': Original={bool(original_match)}, Corrected={bool(corrected_match)}")

if __name__ == "__main__":
    test_chinese_numeral_pattern()