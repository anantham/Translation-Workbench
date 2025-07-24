#!/usr/bin/env python3
"""
Smart Alignment Detector - Finds systematic offset patterns
Uses lightweight text comparison instead of heavy ML models
"""
import os
import json
import requests
from difflib import SequenceMatcher

def simple_similarity(text1, text2):
    """Fast similarity using length + sequence matching."""
    if not text1 or not text2:
        return 0.0
    
    # Length similarity (±30% tolerance)
    len1, len2 = len(text1), len(text2)
    length_ratio = min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0
    
    # Text similarity (first 1000 chars for speed)
    sample1 = text1[:1000].lower()
    sample2 = text2[:1000].lower()
    text_similarity = SequenceMatcher(None, sample1, sample2).ratio()
    
    # Combined score
    return (length_ratio * 0.3) + (text_similarity * 0.7)

def translate_with_gemini(text, api_key):
    """Quick translation for comparison."""
    if not api_key:
        return None
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    prompt = f"Translate this Chinese text to English (brief, literal):\n\n{text[:500]}..."
    
    try:
        response = requests.post(url, json={
            "contents": [{"parts": [{"text": prompt}]}]
        }, timeout=30)
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except:
        return None

def detect_alignment_pattern(alignment_map, api_key=None, sample_size=10):
    """Detect systematic offset by testing multiple chapters."""
    print("🔍 DETECTING ALIGNMENT PATTERN")
    print("=" * 50)
    
    results = []
    chapters = sorted([int(k) for k in alignment_map.keys()])[:sample_size]
    
    for ch_num in chapters:
        ch_data = alignment_map[str(ch_num)]
        
        # Load Chinese content
        if not os.path.exists(ch_data["raw_file"]):
            continue
            
        with open(ch_data["raw_file"], 'r', encoding='utf-8') as f:
            chinese_text = f.read()
        
        # Test against multiple English chapters (current, ±1, ±2)
        best_match = {"chapter": ch_num, "score": 0, "offset": 0}
        
        for offset in range(-2, 3):  # Test -2, -1, 0, +1, +2
            test_ch = ch_num + offset
            if str(test_ch) in alignment_map:
                eng_file = alignment_map[str(test_ch)]["english_file"]
                if eng_file and os.path.exists(eng_file):
                    with open(eng_file, 'r', encoding='utf-8') as f:
                        english_text = f.read()
                    
                    # Method 1: Direct length comparison
                    direct_score = simple_similarity(chinese_text, english_text)
                    
                    # Method 2: Via AI translation (if API key provided)
                    ai_score = 0
                    if api_key:
                        ai_translation = translate_with_gemini(chinese_text, api_key)
                        if ai_translation:
                            ai_score = simple_similarity(ai_translation, english_text)
                    
                    # Combined score
                    final_score = max(direct_score, ai_score)
                    
                    if final_score > best_match["score"]:
                        best_match = {
                            "chapter": ch_num,
                            "score": final_score,
                            "offset": offset,
                            "matched_english": test_ch
                        }
        
        results.append(best_match)
        print(f"Ch {ch_num:3d}: Best match = English Ch {best_match['matched_english']} (offset: {best_match['offset']:+d}, score: {best_match['score']:.3f})")
    
    # Analyze pattern
    offsets = [r["offset"] for r in results if r["score"] > 0.3]
    if offsets:
        most_common_offset = max(set(offsets), key=offsets.count)
        confidence = offsets.count(most_common_offset) / len(offsets)
        
        print("\n🎯 DETECTED PATTERN:")
        print(f"   Most common offset: {most_common_offset:+d}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Recommendation: {'Apply systematic correction' if confidence > 0.7 else 'Manual review needed'}")
        
        return most_common_offset, confidence
    
    return 0, 0.0

def apply_systematic_correction(alignment_map, offset):
    """Apply systematic offset correction to alignment map."""
    if offset == 0:
        print("No correction needed.")
        return alignment_map
    
    corrected_map = {}
    
    for ch_str, ch_data in alignment_map.items():
        ch_num = int(ch_str)
        corrected_english_ch = ch_num + offset
        
        # Find the English file for the corrected chapter
        if str(corrected_english_ch) in alignment_map:
            corrected_map[ch_str] = {
                "raw_file": ch_data["raw_file"],
                "english_file": alignment_map[str(corrected_english_ch)]["english_file"]
            }
        else:
            # Keep original if corrected chapter doesn't exist
            corrected_map[ch_str] = ch_data
    
    return corrected_map

def main():
    # Load current alignment
    with open("alignment_map.json", 'r') as f:
        alignment_map = json.load(f)
    
    # Get API key for enhanced detection
    api_key = input("Enter Gemini API key (optional, press Enter to skip): ").strip()
    if not api_key:
        api_key = None
    
    # Detect pattern
    offset, confidence = detect_alignment_pattern(alignment_map, api_key)
    
    # Apply correction if confident
    if confidence > 0.7 and offset != 0:
        print(f"\n❓ Apply systematic offset correction ({offset:+d})? (y/n): ", end="")
        if input().lower() == 'y':
            corrected_map = apply_systematic_correction(alignment_map, offset)
            
            # Backup original
            with open("alignment_map_backup.json", 'w') as f:
                json.dump(alignment_map, f, indent=2)
            
            # Save corrected version
            with open("alignment_map.json", 'w') as f:
                json.dump(corrected_map, f, indent=2)
            
            print("✅ Alignment corrected and saved!")
            print("📁 Original backed up to alignment_map_backup.json")

if __name__ == "__main__":
    main()