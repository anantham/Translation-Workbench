#!/usr/bin/env python3
"""
Test script to verify DeepSeek integration works correctly.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import load_deepseek_api_config, show_deepseek_config_status, get_available_models_for_translation

def get_model_abbreviation(platform, model_name):
    """Copy of the updated function from Translation Lab"""
    if platform == "Gemini":
        if "gemini-1.5-pro" in model_name:
            return "gem15p"
        elif "gemini-1.5-flash" in model_name:
            return "gem15f"
        else:
            return "gemini"
    
    elif platform == "OpenAI":
        if model_name.startswith("ft:"):
            parts = model_name.split(":")
            if len(parts) >= 5:
                model_id = parts[4]
                clean_id = "".join(c for c in model_id if c.isalnum() or c in "-_")
                return f"oai_{clean_id}"
            elif len(parts) >= 4:
                custom_name = parts[3]
                clean_name = "".join(c for c in custom_name if c.isalnum() or c in "-_")[:8]
                return f"oai_{clean_name}"
            else:
                return "oai_ft"
        elif "gpt-4o" in model_name:
            if "mini" in model_name:
                return "oai_gpt4m"
            else:
                return "oai_gpt4o"
        elif "gpt-4" in model_name:
            return "oai_gpt4"
        elif "gpt-3.5" in model_name:
            return "oai_gpt35"
        elif "deepseek-chat" in model_name:
            return "deepseek_chat"
        elif "deepseek-reasoner" in model_name:
            return "deepseek_reason"
        else:
            return "oai"
    
    return platform.lower()[:3]

print("🧪 Testing DeepSeek Integration")
print("=" * 50)

# Test 1: Configuration loading
print("1. Testing DeepSeek API key configuration:")
api_key, source = load_deepseek_api_config()
status = show_deepseek_config_status()
print(f"   {status}")

# Test 2: Model availability
print("\n2. Testing model availability:")
models = get_available_models_for_translation()
openai_models = models.get("OpenAI", [])
deepseek_models = [m for m in openai_models if m.startswith("deepseek")]
print(f"   Available DeepSeek models: {deepseek_models}")

# Test 3: Model abbreviations
print("\n3. Testing model abbreviations:")
test_models = [
    ("OpenAI", "deepseek-chat", "deepseek_chat"),
    ("OpenAI", "deepseek-reasoner", "deepseek_reason"),
    ("OpenAI", "gpt-4o", "oai_gpt4o"),
    ("OpenAI", "gpt-4o-mini", "oai_gpt4m"),
]

all_passed = True
for platform, model, expected in test_models:
    result = get_model_abbreviation(platform, model)
    status = "✅ PASS" if result == expected else "❌ FAIL"
    print(f"   {status} | {model:20} → {result:15} (expected: {expected})")
    if result != expected:
        all_passed = False

# Test 4: Example run names
print("\n4. Example run names with DeepSeek models:")
base_name = "pro_list_1_20250625_2250"
for _, model, abbrev in test_models:
    if model.startswith("deepseek"):
        example_name = f"{base_name}_{abbrev}"
        print(f"   {model:20} → {example_name}")

print("\n" + "=" * 50)
if api_key:
    print("🎉 DeepSeek integration ready!")
    print("\nTo use DeepSeek models:")
    print("   1. Set DEEPSEEK_API_KEY environment variable")
    print("   2. Select 'OpenAI' platform in Translation Lab")
    print("   3. Choose 'deepseek-chat' or 'deepseek-reasoner' model")
    print("   4. Models will automatically use DeepSeek API with base_url")
else:
    print("⚠️  DeepSeek API key not configured.")
    print("\nTo configure:")
    print("   export DEEPSEEK_API_KEY='your-deepseek-api-key'")
    print("   # or add 'deepseek_api_key' to config.json")

if all_passed:
    print("\n✅ All model abbreviation tests passed!")
else:
    print("\n❌ Some tests failed - check implementation.")