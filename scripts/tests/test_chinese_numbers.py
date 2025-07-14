#!/usr/bin/env python3
"""Test Chinese number conversion function"""

def chinese_to_int(text):
    text = str(text).replace('前', '')
    num_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '零': 0}
    unit_map = {'十': 10, '百': 100, '千': 1000, '万': 10000}
    total = 0
    temp_num = 0
    
    # Handle standalone units at the beginning (e.g., '十' for 10)
    if text.startswith('十'):
        total = 10
        text = text[1:]

    for char in text:
        if char in num_map:
            temp_num = num_map[char]
        elif char in unit_map:
            # handle cases like '一百三' (130) vs '一十三' (13)
            if temp_num == 0: temp_num = 1
            total += temp_num * unit_map[char]
            temp_num = 0
    total += temp_num
    return total if total > 0 else None

# Test more complex patterns that might reveal the bug
test_cases = [
    ('一千一百一十一', 1111),  # Should be 1111
    ('九千九百九十九', 9999),  # Test upper range
    ('二千五百', 2500),        # Test with zero omitted
    ('三千', 3000),            # Test simple thousands
    ('四千零一', 4001),        # Test with zero
    ('五千零五十', 5050),      # Test with zero in middle
    ('六千七百八十九', 6789),  # Complex case
    ('八千三百', 8300),        # Test hundreds
    ('九千九百', 9900),        # Test without units
    ('千', 1000),              # Test unit only
    ('九千九百九十', 9990),    # Missing final digit
]

print('Testing Chinese number conversion:')
print('=' * 50)
failures = []

for input_text, expected in test_cases:
    result = chinese_to_int(input_text)
    status = '✅' if result == expected else '❌'
    print(f'{status} {input_text:<12} -> {result:<4} (expected {expected})')
    if result != expected:
        failures.append((input_text, result, expected))

if failures:
    print(f'\n❌ {len(failures)} test failures:')
    for inp, got, exp in failures:
        print(f'   {inp}: got {got}, expected {exp}')
else:
    print('\n✅ All tests passed!')