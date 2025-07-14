"""
API Cost Calculation and Tracking Module

Precise cost estimation for multiple AI providers.
Supports fine-tuned models and batch API pricing.

This module handles:
- Loading pricing configuration from data/pricing_config.json
- Calculating OpenAI API costs (including fine-tuned models)
- Calculating Gemini API costs
- Model name normalization and pricing lookup
"""

import os
import json

# Data directory structure
DATA_DIR = "data"


def load_pricing_config():
    """Load current API pricing configuration."""
    pricing_file = os.path.join(DATA_DIR, "pricing_config.json")
    try:
        with open(pricing_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback pricing if config file missing
        return {
            "openai_models": {
                "gpt-4o": {"input_per_1m": 5.00, "output_per_1m": 20.00},
                "gpt-4o-mini": {"input_per_1m": 0.60, "output_per_1m": 2.40}
            },
            "gemini_models": {
                "gemini-1.5-pro": {"input_per_1m": 0.00125, "output_per_1m": 0.005}
            }
        }


def calculate_openai_cost(model_name, usage_data):
    """Calculate precise OpenAI API cost based on current pricing."""
    pricing_config = load_pricing_config()
    
    # Handle fine-tuned models (ft:gpt-4o-mini:org:name:id format)
    base_model = model_name
    is_fine_tuned = model_name.startswith("ft:")
    if is_fine_tuned:
        # Extract base model from fine-tuned name
        parts = model_name.split(":")
        if len(parts) >= 2:
            base_model = parts[1]  # e.g., "gpt-4o-mini"
    
    # Normalize model name for pricing lookup
    model_key = base_model.replace("-latest", "").replace("chatgpt-", "gpt-")
    
    # Look up pricing
    pricing_source = "fine_tuned_models" if is_fine_tuned else "openai_models"
    if pricing_source in pricing_config and model_key in pricing_config[pricing_source]:
        rates = pricing_config[pricing_source][model_key]
        
        input_cost = (usage_data.prompt_tokens / 1_000_000) * rates["input_per_1m"]
        output_cost = (usage_data.completion_tokens / 1_000_000) * rates["output_per_1m"]
        total_cost = input_cost + output_cost
        
        return {
            "total_cost": round(total_cost, 6),
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "input_rate": rates["input_per_1m"],
            "output_rate": rates["output_per_1m"],
            "is_fine_tuned": is_fine_tuned,
            "base_model": base_model
        }
    
    # Fallback for unknown models
    return {
        "total_cost": 0.0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "input_rate": 0.0,
        "output_rate": 0.0,
        "is_fine_tuned": is_fine_tuned,
        "base_model": base_model,
        "error": f"Unknown model pricing: {model_key}"
    }


def calculate_gemini_cost(model_name, usage_metadata):
    """Calculate Gemini API cost based on current pricing."""
    pricing_config = load_pricing_config()
    
    if not usage_metadata:
        return {"total_cost": 0.0, "input_cost": 0.0, "output_cost": 0.0}
    
    # Look up Gemini pricing
    model_key = model_name.replace("models/", "")
    if "gemini_models" in pricing_config and model_key in pricing_config["gemini_models"]:
        rates = pricing_config["gemini_models"][model_key]
        
        input_tokens = usage_metadata.prompt_token_count
        output_tokens = usage_metadata.candidates_token_count
        
        input_cost = (input_tokens / 1_000_000) * rates["input_per_1m"]
        output_cost = (output_tokens / 1_000_000) * rates["output_per_1m"]
        total_cost = input_cost + output_cost
        
        return {
            "total_cost": round(total_cost, 6),
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "input_rate": rates["input_per_1m"],
            "output_rate": rates["output_per_1m"]
        }
    
    # Fallback for unknown Gemini models
    return {
        "total_cost": 0.0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "error": f"Unknown Gemini model pricing: {model_key}"
    }