"""
Translation Framework Workbench - Modular Utils Package

This package contains the core functionality for the translation workbench,
organized into focused modules for better maintainability and development experience.

Original utils.py was 28,656 tokens (5.7x larger than optimal).
Modularized into 8 focused modules of 2-6K tokens each.

Modules:
    config: API configuration and metadata management
    translation: Core AI translation functions
    fine_tuning: Model training and monitoring
    evaluation: Similarity scoring and quality assessment
    data_management: Content loading and alignment
    caching: Translation and similarity caching
    export: Dataset exports and EPUB generation
    cost_tracking: API cost calculations
    ui_components: Streamlit UI helpers

Usage:
    from utils import load_api_config, generate_translation_unified
    # All existing imports continue to work for backward compatibility
"""

# Import completed modules for backward compatibility
# This allows existing code to continue using: from utils import function_name

# Configuration and setup (COMPLETED)
from .config import (
    load_api_config,
    get_config_value,
    load_epub_metadata_config,
    show_config_status,
    load_openai_api_config,
    load_deepseek_api_config,
    show_openai_config_status,
    show_deepseek_config_status,
    get_novel_dir,
    get_novel_alignment_map,
    get_novel_cache_dir,
    get_novel_exports_dir,
    get_novel_ai_translations_dir,
    get_novel_raw_chapters_dir,
    get_novel_official_english_dir,
    NOVELS_DIR,
    SHARED_DIR,
)

# Cost tracking (COMPLETED)
from .cost_tracking import (
    load_pricing_config,
    calculate_openai_cost,
    calculate_gemini_cost,
)

# Caching system (COMPLETED)
from .caching import (
    generate_text_hash,
    load_similarity_cache,
    save_similarity_cache,
    get_translation_cache_path,
    get_cached_translation,
    store_translation_in_cache,
    get_translation_cache_stats,
)

# Quality metrics (COMPLETED)
from .quality_metrics import (
    calculate_word_counts,
    calculate_average_quality_metrics,
)

# Data management (COMPLETED)
from .data_management import (
    load_chapter_content,
    load_alignment_map,
    save_alignment_map,
    get_text_statistics,
    get_chapter_word_count,
)

# Core translation functions (COMPLETED)
from .translation import (
    generate_translation_unified,
    translate_with_gemini,
    translate_with_openai,
    translate_with_deepseek,
    translate_with_gemini_history,
)

# Evaluation and quality assessment (COMPLETED)
from .evaluation import (
    load_semantic_model,
    calculate_similarity,
    calculate_syntactic_similarity_fallback,
    calculate_bleu_score,
    evaluate_translation_quality,
    get_available_translation_styles,
    calculate_bert_scores_for_style,
    calculate_composite_score,
    load_bert_scores_from_reports,
    get_chunking_statistics,
    SEMANTIC_AVAILABLE,
    SEMANTIC_ERROR_MESSAGE,
)

# AI source and model management (COMPLETED)
from .ai_source_management import (
    get_available_ai_sources,
    get_ai_translation_content,
    get_static_gemini_models,
    get_available_openai_models,
    get_available_models_for_translation,
    validate_model_availability,
    get_model_recommendations,
    detect_model_platform,
    get_model_info,
)

# Custom prompt management (COMPLETED)
from .prompt_management import (
    load_custom_prompts,
    save_custom_prompt,
    delete_custom_prompt,
    update_custom_prompt,
    get_builtin_prompts,
    get_all_available_prompts,
    get_prompt_categories,
    search_prompts,
    validate_prompt_content,
    get_prompt_statistics,
    export_prompts,
    import_prompts,
)

# Data export and dataset generation (COMPLETED)
from .export import (
    export_training_data_to_jsonl,
    create_translation_jsonl,
    prepare_training_data_for_api,
    chunk_chapter_for_training,
    chunk_by_characters,
    create_dataset_report,
    export_dataset_with_metadata,
)

# Fine-tuning and model training (COMPLETED)
from .fine_tuning import (
    load_dataset_for_tuning,
    get_max_available_chapters,
    start_finetuning_job,
    get_tuning_job_status,
    list_tuning_jobs,
    upload_training_file_openai,
    start_openai_finetuning_job,
    get_openai_finetuning_status,
    list_openai_finetuning_jobs,
    list_openai_finetuned_models,
    save_model_metadata,
    load_model_metadata,
    list_trained_models,
    create_training_pipeline,
    execute_training_step,
    get_training_progress,
    GOOGLE_AI_AVAILABLE,
)

# UI components and Streamlit interface (COMPLETED)
from .ui_components import (
    escape_html,
    apply_comment_highlighting,
    save_inline_comments,
    load_inline_comments,
    add_inline_comment,
    generate_existing_comments_html,
    create_synchronized_text_display,
)

# Miscellaneous utilities (COMPLETED)
from .miscellaneous import (
    get_text_stats,
    save_alignment_map_safely,
    format_example,
    save_bert_scores,
    load_bert_scores,
    save_human_scores,
    load_human_scores,
    find_first_misalignment_binary_search,
    preview_systematic_correction,
    validate_scraping_url,
    extract_chapter_number,
    sanitize_filename,
)

# Web scraping integration (COMPLETED)
from .web_scraping import (
    streamlit_scraper,
    validate_scraping_url as web_validate_scraping_url,
)

# Alignment map building (COMPLETED)
from .alignment_builder import (
    build_alignment_map,
    streamlit_build_alignment_map,
    detect_novel_structure,
    save_alignment_map_with_backup,
)

# All modules completed!
# Configuration and setup - from .config import (...)
# Core translation functions - from .translation import (...)
# Fine-tuning capabilities - from .fine_tuning import (...)
# Evaluation and quality assessment - from .evaluation import (...)
# Data management - from .data_management import (...)
# Caching system - from .caching import (...)
# Export functionality - from .export import (...)
# UI components - from .ui_components import (...)
# AI source and prompt management - from .ai_source_management import (...)
# Custom prompt management - from .prompt_management import (...)
# Quality metrics calculation - from .quality_metrics import (...)
# Web scraping - from .web_scraping import (...)

# Version info
__version__ = "2.0.0"
__author__ = "Translation Framework Workbench"
__description__ = "Modular utilities for pluralistic translation research"

# Export completed functions for backward compatibility
__all__ = [
    # Configuration and setup (COMPLETED)
    'load_api_config', 'get_config_value', 'load_epub_metadata_config',
    'show_config_status', 'load_openai_api_config', 'load_deepseek_api_config',
    'show_openai_config_status', 'show_deepseek_config_status',
    'get_novel_dir', 'get_novel_alignment_map', 'get_novel_cache_dir',
    'get_novel_exports_dir', 'get_novel_ai_translations_dir', 'get_novel_raw_chapters_dir',
    'get_novel_official_english_dir', 'NOVELS_DIR', 'SHARED_DIR',
    
    # Cost tracking (COMPLETED)
    'load_pricing_config', 'calculate_openai_cost', 'calculate_gemini_cost',
    
    # Caching system (COMPLETED)
    'generate_text_hash', 'load_similarity_cache', 'save_similarity_cache',
    'get_translation_cache_path', 'get_cached_translation',
    'store_translation_in_cache', 'get_translation_cache_stats',
    
    # Quality metrics (COMPLETED)
    'calculate_word_counts', 'calculate_average_quality_metrics',
    
    # Data management (COMPLETED)
    'load_chapter_content', 'load_alignment_map', 'save_alignment_map',
    'get_text_statistics', 'get_chapter_word_count',
    
    # Core translation functions (COMPLETED)
    'generate_translation_unified', 'translate_with_gemini', 
    'translate_with_openai', 'translate_with_deepseek', 'translate_with_gemini_history',
    
    # Evaluation and quality assessment (COMPLETED)
    'load_semantic_model', 'calculate_similarity', 'calculate_syntactic_similarity_fallback',
    'calculate_bleu_score', 'evaluate_translation_quality', 'get_available_translation_styles',
    'calculate_bert_scores_for_style', 'calculate_composite_score', 'load_bert_scores_from_reports',
    'get_chunking_statistics', 'SEMANTIC_AVAILABLE', 'SEMANTIC_ERROR_MESSAGE',
    
    # AI source and model management (COMPLETED)
    'get_available_ai_sources', 'get_ai_translation_content', 'get_static_gemini_models',
    'get_available_openai_models', 'get_available_models_for_translation', 'validate_model_availability',
    'get_model_recommendations', 'detect_model_platform', 'get_model_info',
    
    # Custom prompt management (COMPLETED)
    'load_custom_prompts', 'save_custom_prompt', 'delete_custom_prompt', 'update_custom_prompt',
    'get_builtin_prompts', 'get_all_available_prompts', 'get_prompt_categories', 'search_prompts',
    'validate_prompt_content', 'get_prompt_statistics', 'export_prompts', 'import_prompts',
    
    # Data export and dataset generation (COMPLETED)
    'export_training_data_to_jsonl', 'create_translation_jsonl', 'prepare_training_data_for_api',
    'chunk_chapter_for_training', 'chunk_by_characters', 'create_dataset_report',
    'export_dataset_with_metadata',
    
    # Fine-tuning and model training (COMPLETED)
    'load_dataset_for_tuning', 'get_max_available_chapters', 'start_finetuning_job',
    'get_tuning_job_status', 'list_tuning_jobs', 'upload_training_file_openai',
    'start_openai_finetuning_job', 'get_openai_finetuning_status', 'list_openai_finetuning_jobs',
    'list_openai_finetuned_models', 'save_model_metadata', 'load_model_metadata',
    'list_trained_models', 'create_training_pipeline', 'execute_training_step',
    'get_training_progress', 'GOOGLE_AI_AVAILABLE',
    
    # UI components and Streamlit interface (COMPLETED)
    'escape_html', 'apply_comment_highlighting', 'save_inline_comments',
    'load_inline_comments', 'add_inline_comment', 'generate_existing_comments_html',
    'create_synchronized_text_display',
    
    # Miscellaneous utilities (COMPLETED)
    'get_text_stats', 'save_alignment_map_safely', 'format_example',
    'save_bert_scores', 'load_bert_scores', 'save_human_scores', 'load_human_scores',
    'find_first_misalignment_binary_search', 'preview_systematic_correction',
    'validate_scraping_url', 'extract_chapter_number', 'sanitize_filename',
    
    # Web scraping integration (COMPLETED)
    'streamlit_scraper', 'web_validate_scraping_url',
    
    # Alignment map building (COMPLETED)
    'build_alignment_map', 'streamlit_build_alignment_map', 'detect_novel_structure', 'save_alignment_map_with_backup',
    
    # ALL MODULES COMPLETED! 🎉
]