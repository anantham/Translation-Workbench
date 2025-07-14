"""
Custom Prompt Management Module

System for managing translation prompts and styles.
Built-in templates, custom user prompts, and prompt organization.

This module handles:
- Custom prompt template storage and retrieval
- Built-in prompt template management
- Prompt CRUD operations (Create, Read, Update, Delete)
- Prompt categorization and organization
- Combined prompt discovery (built-in + custom)
- Template validation and metadata management
"""

import os
import json
from datetime import datetime

# Import configuration
from .config import DATA_DIR


def load_custom_prompts():
    """Load custom prompt templates from JSON file.
    
    Returns:
        dict: Dictionary of custom prompts {name: {"content": str, "created": str, "category": str}}
    """
    custom_prompts_file = os.path.join(DATA_DIR, "custom_prompts.json")
    
    if os.path.exists(custom_prompts_file):
        try:
            with open(custom_prompts_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_custom_prompt(name, content, category="user"):
    """Save a custom prompt template.
    
    Args:
        name: Name of the prompt template
        content: The prompt content
        category: Category for organization (default: "user")
    
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        custom_prompts = load_custom_prompts()
        
        # Add timestamp and metadata
        custom_prompts[name] = {
            "content": content.strip(),
            "created": datetime.now().isoformat(),
            "category": category,
            "last_modified": datetime.now().isoformat(),
            "version": custom_prompts.get(name, {}).get("version", 0) + 1
        }
        
        # Save back to file
        custom_prompts_file = os.path.join(DATA_DIR, "custom_prompts.json")
        with open(custom_prompts_file, 'w', encoding='utf-8') as f:
            json.dump(custom_prompts, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Error saving custom prompt: {e}")
        return False


def delete_custom_prompt(name):
    """Delete a custom prompt template.
    
    Args:
        name: Name of the prompt template to delete
    
    Returns:
        bool: True if deleted successfully, False otherwise
    """
    try:
        custom_prompts = load_custom_prompts()
        
        if name in custom_prompts:
            del custom_prompts[name]
            
            # Save back to file
            custom_prompts_file = os.path.join(DATA_DIR, "custom_prompts.json")
            with open(custom_prompts_file, 'w', encoding='utf-8') as f:
                json.dump(custom_prompts, f, indent=2, ensure_ascii=False)
            
            return True
        return False
    except Exception as e:
        print(f"Error deleting custom prompt: {e}")
        return False


def update_custom_prompt(name, content, category=None):
    """Update an existing custom prompt template.
    
    Args:
        name: Name of the prompt template to update
        content: New content for the prompt
        category: Optional new category
    
    Returns:
        bool: True if updated successfully, False otherwise
    """
    try:
        custom_prompts = load_custom_prompts()
        
        if name not in custom_prompts:
            return False
        
        # Update existing prompt
        existing = custom_prompts[name]
        custom_prompts[name] = {
            "content": content.strip(),
            "created": existing.get("created", datetime.now().isoformat()),
            "category": category if category is not None else existing.get("category", "user"),
            "last_modified": datetime.now().isoformat(),
            "version": existing.get("version", 0) + 1
        }
        
        # Save back to file
        custom_prompts_file = os.path.join(DATA_DIR, "custom_prompts.json")
        with open(custom_prompts_file, 'w', encoding='utf-8') as f:
            json.dump(custom_prompts, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        print(f"Error updating custom prompt: {e}")
        return False


def get_builtin_prompts():
    """Get built-in prompt templates.
    
    Returns:
        dict: Built-in prompt templates
    """
    return {
        "Literal & Accurate": "You are a professional translator specializing in Chinese to English translation of web novels. Provide an accurate, literal translation that preserves the original meaning, structure, and cultural context. Maintain formal tone and precise terminology.",
        
        "Dynamic & Modern": "You are a skilled literary translator adapting Chinese web novels for Western readers. Create a flowing, engaging translation that captures the spirit and excitement of the original while using natural modern English. Prioritize readability and dramatic impact.",
        
        "Simplified & Clear": "You are translating Chinese web novels for young adult readers. Use simple, clear language that's easy to understand. Explain cultural concepts briefly when needed. Keep sentences shorter and vocabulary accessible.",
        
        "Academic Style": "You are translating Chinese web novels for academic analysis. Maintain scholarly precision and include cultural context notes where appropriate. Use formal academic English with clear structure and precise terminology.",
        
        "Narrative Flow": "You are a creative translator focusing on storytelling flow. Adapt the Chinese narrative style to Western storytelling conventions while preserving the author's voice. Prioritize reader engagement and narrative pacing."
    }


def get_all_available_prompts():
    """Get all available prompt templates (built-in + custom).
    
    Returns:
        dict: Combined dictionary of all prompts
    """
    # Get built-in prompts
    builtin_prompts = get_builtin_prompts()
    
    # Load custom prompts
    custom_prompts = load_custom_prompts()
    
    # Combine prompts with prefixes for organization
    all_prompts = {}
    
    # Add built-in prompts
    for name, content in builtin_prompts.items():
        all_prompts[name] = content
    
    # Add custom prompts with prefix
    for name, prompt_data in custom_prompts.items():
        all_prompts[f"🎨 {name}"] = prompt_data["content"]
    
    # Add Custom option for creating new prompts
    all_prompts["Custom"] = ""
    
    return all_prompts


def get_prompt_categories():
    """Get available prompt categories from custom prompts.
    
    Returns:
        dict: Category -> list of prompt names mapping
    """
    custom_prompts = load_custom_prompts()
    categories = {"built-in": list(get_builtin_prompts().keys())}
    
    for name, prompt_data in custom_prompts.items():
        category = prompt_data.get("category", "uncategorized")
        if category not in categories:
            categories[category] = []
        categories[category].append(name)
    
    return categories


def search_prompts(query, search_content=True):
    """Search prompts by name or content.
    
    Args:
        query: Search query string
        search_content: Whether to search in prompt content (default: True)
    
    Returns:
        dict: Matching prompts
    """
    query_lower = query.lower()
    all_prompts = get_all_available_prompts()
    matches = {}
    
    for name, content in all_prompts.items():
        # Skip empty custom option
        if name == "Custom" and content == "":
            continue
            
        # Search in name
        if query_lower in name.lower():
            matches[name] = content
        # Search in content if enabled
        elif search_content and query_lower in content.lower():
            matches[name] = content
    
    return matches


def validate_prompt_content(content):
    """Validate prompt content for common issues.
    
    Args:
        content: Prompt content to validate
    
    Returns:
        tuple: (is_valid, list_of_issues)
    """
    issues = []
    
    if not content or not content.strip():
        issues.append("Prompt content cannot be empty")
    
    if len(content.strip()) < 10:
        issues.append("Prompt content is too short (minimum 10 characters)")
    
    if len(content) > 5000:
        issues.append("Prompt content is too long (maximum 5000 characters)")
    
    # Check for common prompt elements
    content_lower = content.lower()
    if "translate" not in content_lower and "translation" not in content_lower:
        issues.append("Prompt should mention translation or translate")
    
    if "chinese" not in content_lower and "english" not in content_lower:
        issues.append("Prompt should specify source/target languages")
    
    return len(issues) == 0, issues


def get_prompt_statistics():
    """Get statistics about prompt usage and library.
    
    Returns:
        dict: Statistics about prompts
    """
    builtin = get_builtin_prompts()
    custom = load_custom_prompts()
    categories = get_prompt_categories()
    
    stats = {
        "total_prompts": len(builtin) + len(custom),
        "builtin_count": len(builtin),
        "custom_count": len(custom),
        "categories": len(categories),
        "category_breakdown": {cat: len(prompts) for cat, prompts in categories.items()},
        "average_prompt_length": 0,
        "longest_prompt": "",
        "shortest_prompt": ""
    }
    
    # Calculate length statistics
    all_contents = list(builtin.values()) + [p["content"] for p in custom.values()]
    if all_contents:
        lengths = [len(content) for content in all_contents]
        stats["average_prompt_length"] = sum(lengths) / len(lengths)
        
        longest_content = max(all_contents, key=len)
        shortest_content = min(all_contents, key=len)
        
        stats["longest_prompt"] = longest_content[:100] + "..." if len(longest_content) > 100 else longest_content
        stats["shortest_prompt"] = shortest_content[:100] + "..." if len(shortest_content) > 100 else shortest_content
    
    return stats


def export_prompts(export_path=None, include_builtin=True):
    """Export prompts to a JSON file for backup or sharing.
    
    Args:
        export_path: Path to export file (default: data/exports/prompts_backup_TIMESTAMP.json)
        include_builtin: Whether to include built-in prompts in export
    
    Returns:
        tuple: (success, export_path_used)
    """
    try:
        if export_path is None:
            export_dir = os.path.join(DATA_DIR, "exports")
            os.makedirs(export_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = os.path.join(export_dir, f"prompts_backup_{timestamp}.json")
        
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "custom_prompts": load_custom_prompts()
        }
        
        if include_builtin:
            export_data["builtin_prompts"] = get_builtin_prompts()
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return True, export_path
        
    except Exception as e:
        print(f"Error exporting prompts: {e}")
        return False, None


def import_prompts(import_path, overwrite_existing=False):
    """Import prompts from a JSON file.
    
    Args:
        import_path: Path to import file
        overwrite_existing: Whether to overwrite existing custom prompts
    
    Returns:
        tuple: (success, import_count, skipped_count)
    """
    try:
        if not os.path.exists(import_path):
            return False, 0, 0
        
        with open(import_path, 'r', encoding='utf-8') as f:
            import_data = json.load(f)
        
        existing_prompts = load_custom_prompts()
        imported_custom = import_data.get("custom_prompts", {})
        
        import_count = 0
        skipped_count = 0
        
        for name, prompt_data in imported_custom.items():
            if name in existing_prompts and not overwrite_existing:
                skipped_count += 1
                continue
            
            # Add import metadata
            prompt_data["imported"] = True
            prompt_data["import_timestamp"] = datetime.now().isoformat()
            
            existing_prompts[name] = prompt_data
            import_count += 1
        
        # Save updated prompts
        custom_prompts_file = os.path.join(DATA_DIR, "custom_prompts.json")
        with open(custom_prompts_file, 'w', encoding='utf-8') as f:
            json.dump(existing_prompts, f, indent=2, ensure_ascii=False)
        
        return True, import_count, skipped_count
        
    except Exception as e:
        print(f"Error importing prompts: {e}")
        return False, 0, 0