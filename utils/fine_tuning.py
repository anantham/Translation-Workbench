"""
Fine-tuning and Model Training Module

Complete MLOps pipeline for training custom translation models.
Supports both Google AI (Gemini) and OpenAI fine-tuning platforms.

This module handles:
- Dataset loading and preparation for fine-tuning
- Training data chunking and validation
- Google AI (Gemini) fine-tuning job management
- OpenAI fine-tuning job management
- Training progress monitoring and status tracking
- Model metadata management and versioning
- Multi-provider training pipeline orchestration
"""

import os
import json
import time
import random
from datetime import datetime

# Import data management functions
from .data_management import load_chapter_content, get_text_statistics

# Import evaluation functions for BERT scores
from .evaluation import load_bert_scores_from_reports

# Import configuration
from .config import DATA_DIR, MODELS_DIR

# AI SDK availability detection
GOOGLE_AI_AVAILABLE = False
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
except ImportError:
    pass

OPENAI_AVAILABLE = False
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    pass


def load_dataset_for_tuning(alignment_map, limit=None, min_similarity=None, max_chars=None, include_bert_scores=True):
    """Load dataset from alignment map and prepare for fine-tuning.
    
    Args:
        alignment_map: Chapter alignment mapping
        limit: Maximum number of chapters to process
        min_similarity: Minimum BERT similarity threshold (if available and specified)
        max_chars: Maximum character count per chapter (no limit if None)
        include_bert_scores: Whether to load BERT scores from existing reports
    
    Returns:
        list: Training examples in the format expected by fine-tuning APIs
    """
    training_examples = []
    processed = 0
    
    # Load BERT scores from existing reports if available
    bert_scores = {}
    if include_bert_scores:
        bert_scores = load_bert_scores_from_reports()
    
    # Get sorted chapter numbers
    chapter_numbers = sorted([int(k) for k in alignment_map.keys()])
    if limit:
        chapter_numbers = chapter_numbers[:limit]
    
    for chapter_num in chapter_numbers:
        chapter_data = alignment_map[str(chapter_num)]
        
        # Check if both files exist
        if not chapter_data.get('raw_file') or not chapter_data.get('english_file'):
            continue
        
        # Load content
        raw_content = load_chapter_content(chapter_data['raw_file'])
        english_content = load_chapter_content(chapter_data['english_file'])
        
        if "File not found" in raw_content or "File not found" in english_content:
            continue
        
        # Get text statistics
        raw_stats = get_text_statistics(raw_content, 'chinese')
        eng_stats = get_text_statistics(english_content, 'english')
        
        # Apply character count limit only if specified
        if max_chars is not None:
            if raw_stats['char_count'] > max_chars or eng_stats['char_count'] > max_chars:
                continue
        
        # Check BERT similarity threshold only if specified
        bert_score = bert_scores.get(chapter_num)
        if min_similarity is not None and bert_score is not None and bert_score < min_similarity:
            continue
        
        # Create training example
        training_example = {
            "chapter_number": chapter_num,
            "raw_content": raw_content,
            "english_content": english_content,
            "raw_stats": raw_stats,
            "english_stats": eng_stats,
            "bert_similarity": bert_score  # Include BERT score if available
        }
        
        training_examples.append(training_example)
        processed += 1
    
    return training_examples


def get_max_available_chapters(alignment_map):
    """Get the maximum number of chapters available for training.
    
    Args:
        alignment_map: Chapter alignment mapping
    
    Returns:
        int: Maximum number of available chapters for training
    """
    available_count = 0
    chapter_numbers = sorted([int(k) for k in alignment_map.keys()])
    
    for chapter_num in chapter_numbers:
        chapter_data = alignment_map[str(chapter_num)]
        
        # Check if English file exists
        if not chapter_data.get('english_file'):
            continue
            
        english_file_path = chapter_data['english_file']
        if os.path.exists(english_file_path):
            available_count += 1
        else:
            # If we hit a missing file, continue counting to get the actual total
            continue
    
    return available_count


# --- Google AI (Gemini) Fine-tuning ---

def start_finetuning_job(api_key, training_data, base_model="models/gemini-1.5-flash-001", 
                        epoch_count=3, batch_size=4, learning_rate=0.001):
    """Start a fine-tuning job using Google AI SDK.
    
    Args:
        api_key: Google AI API key
        training_data: List of training examples
        base_model: Base model to fine-tune
        epoch_count: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate for training
    
    Returns:
        tuple: (operation, error_message)
    """
    if not GOOGLE_AI_AVAILABLE:
        return None, "Google AI SDK not available"
    
    try:
        # Configure the SDK
        genai.configure(api_key=api_key)
        
        # Prepare training data
        training_data_for_api = []
        for example in training_data:
            training_data_for_api.append({
                'text_input': example['messages'][1]['content'],  # User message
                'output': example['messages'][2]['content']       # Assistant message
            })
        
        # Create tuning job
        operation = genai.create_tuned_model(
            source_model=base_model,
            training_data=training_data_for_api,
            id=f"translation-model-{int(time.time())}",
            epoch_count=epoch_count,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        return operation, None
        
    except Exception as e:
        return None, str(e)


def get_tuning_job_status(job_name, api_key):
    """Get the status of a Google AI fine-tuning job.
    
    Args:
        job_name: Name/ID of the tuning job
        api_key: Google AI API key
    
    Returns:
        tuple: (model_info, error_message)
    """
    if not GOOGLE_AI_AVAILABLE:
        return None, "Google AI SDK not available"
    
    try:
        genai.configure(api_key=api_key)
        # Get tuned model info
        model = genai.get_tuned_model(job_name)
        return model, None
    except Exception as e:
        return None, str(e)


def list_tuning_jobs(api_key):
    """List all Google AI fine-tuning jobs.
    
    Args:
        api_key: Google AI API key
    
    Returns:
        tuple: (models_list, error_message)
    """
    if not GOOGLE_AI_AVAILABLE:
        return [], "Google AI SDK not available"
    
    try:
        genai.configure(api_key=api_key)
        models = genai.list_tuned_models()
        return list(models), None
    except Exception as e:
        return [], str(e)


# --- OpenAI Fine-tuning ---

def upload_training_file_openai(api_key, jsonl_content, filename="training_data.jsonl"):
    """Upload training file to OpenAI for fine-tuning.
    
    Args:
        api_key: OpenAI API key
        jsonl_content: JSONL training data content
        filename: Name for the uploaded file
    
    Returns:
        tuple: (file_response, error_message)
    """
    if not OPENAI_AVAILABLE:
        return None, "OpenAI SDK not available"
    
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # Create a temporary file for upload
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write(jsonl_content)
            temp_file_path = f.name
        
        # Upload the file
        with open(temp_file_path, 'rb') as f:
            response = client.files.create(
                file=f,
                purpose='fine-tune'
            )
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        return response, None
        
    except Exception as e:
        return None, str(e)


def start_openai_finetuning_job(api_key, training_file_id, model="gpt-4o-mini", 
                               n_epochs="auto", batch_size="auto", learning_rate_multiplier="auto"):
    """Start OpenAI fine-tuning job.
    
    Args:
        api_key: OpenAI API key
        training_file_id: ID of uploaded training file
        model: Base model to fine-tune
        n_epochs: Number of epochs
        batch_size: Batch size
        learning_rate_multiplier: Learning rate multiplier
    
    Returns:
        tuple: (job_response, error_message)
    """
    if not OPENAI_AVAILABLE:
        return None, "OpenAI SDK not available"
    
    try:
        client = openai.OpenAI(api_key=api_key)
        
        job = client.fine_tuning.jobs.create(
            training_file=training_file_id,
            model=model,
            hyperparameters={
                "n_epochs": n_epochs,
                "batch_size": batch_size,
                "learning_rate_multiplier": learning_rate_multiplier
            }
        )
        
        return job, None
        
    except Exception as e:
        return None, str(e)


def get_openai_finetuning_status(api_key, job_id):
    """Get the status of an OpenAI fine-tuning job.
    
    Args:
        api_key: OpenAI API key
        job_id: ID of the fine-tuning job
    
    Returns:
        tuple: (job_info, error_message)
    """
    if not OPENAI_AVAILABLE:
        return None, "OpenAI SDK not available"
    
    try:
        client = openai.OpenAI(api_key=api_key)
        job = client.fine_tuning.jobs.retrieve(job_id)
        return job, None
    except Exception as e:
        return None, str(e)


def list_openai_finetuning_jobs(api_key):
    """List all OpenAI fine-tuning jobs.
    
    Args:
        api_key: OpenAI API key
    
    Returns:
        tuple: (jobs_list, error_message)
    """
    if not OPENAI_AVAILABLE:
        return [], "OpenAI SDK not available"
    
    try:
        client = openai.OpenAI(api_key=api_key)
        jobs = client.fine_tuning.jobs.list()
        return list(jobs.data), None
    except Exception as e:
        return [], str(e)


def list_openai_finetuned_models(api_key):
    """List user's fine-tuned OpenAI models.
    
    Args:
        api_key: OpenAI API key
    
    Returns:
        tuple: (models_list, error_message)
    """
    if not OPENAI_AVAILABLE:
        return [], "OpenAI SDK not available"
    
    try:
        client = openai.OpenAI(api_key=api_key)
        # Get all fine-tuning jobs and extract completed models
        jobs = client.fine_tuning.jobs.list()
        
        completed_models = []
        for job in jobs.data:
            if job.status == "succeeded" and job.fine_tuned_model:
                completed_models.append({
                    "model_id": job.fine_tuned_model,
                    "base_model": job.model,
                    "job_id": job.id,
                    "created_at": job.created_at,
                    "finished_at": job.finished_at
                })
        
        return completed_models, None
    except Exception as e:
        return [], str(e)


# --- Model Metadata Management ---

def save_model_metadata(model_info, training_config, training_stats):
    """Save metadata for a trained model.
    
    Args:
        model_info: Information about the trained model
        training_config: Configuration used for training
        training_stats: Statistics from the training process
    
    Returns:
        str: Path to saved metadata file
    """
    # Ensure models directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    metadata = {
        "model_info": model_info,
        "training_config": training_config,
        "training_stats": training_stats,
        "created_at": datetime.now().isoformat(),
        "version": "1.0"
    }
    
    # Generate filename based on model ID and timestamp
    model_id = model_info.get("id", f"model_{int(time.time())}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_id}_{timestamp}.json"
    
    metadata_path = os.path.join(MODELS_DIR, filename)
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return metadata_path


def load_model_metadata(metadata_path):
    """Load model metadata from file.
    
    Args:
        metadata_path: Path to metadata file
    
    Returns:
        dict: Model metadata or None if error
    """
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading model metadata: {e}")
        return None


def list_trained_models():
    """List all trained models with metadata.
    
    Returns:
        list: List of model metadata dictionaries
    """
    models = []
    
    if not os.path.exists(MODELS_DIR):
        return models
    
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith('.json'):
            metadata_path = os.path.join(MODELS_DIR, filename)
            metadata = load_model_metadata(metadata_path)
            if metadata:
                metadata['metadata_file'] = filename
                models.append(metadata)
    
    # Sort by creation date (newest first)
    models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    return models


# --- Training Pipeline Orchestration ---

def create_training_pipeline(alignment_map, platform="gemini", **config):
    """Create a complete training pipeline for the specified platform.
    
    Args:
        alignment_map: Chapter alignment mapping
        platform: Target platform ("gemini", "openai")
        **config: Training configuration parameters
    
    Returns:
        dict: Pipeline configuration and status
    """
    pipeline = {
        "platform": platform,
        "config": config,
        "status": "created",
        "created_at": datetime.now().isoformat(),
        "steps": []
    }
    
    # Define pipeline steps based on platform
    if platform.lower() == "gemini":
        pipeline["steps"] = [
            {"name": "load_dataset", "status": "pending"},
            {"name": "prepare_training_data", "status": "pending"},
            {"name": "start_gemini_job", "status": "pending"},
            {"name": "monitor_training", "status": "pending"},
            {"name": "save_metadata", "status": "pending"}
        ]
    elif platform.lower() == "openai":
        pipeline["steps"] = [
            {"name": "load_dataset", "status": "pending"},
            {"name": "create_jsonl", "status": "pending"},
            {"name": "upload_training_file", "status": "pending"},
            {"name": "start_openai_job", "status": "pending"},
            {"name": "monitor_training", "status": "pending"},
            {"name": "save_metadata", "status": "pending"}
        ]
    else:
        pipeline["status"] = "error"
        pipeline["error"] = f"Unsupported platform: {platform}"
    
    return pipeline


def execute_training_step(pipeline, step_name, **kwargs):
    """Execute a specific step in the training pipeline.
    
    Args:
        pipeline: Pipeline configuration
        step_name: Name of the step to execute
        **kwargs: Additional parameters for the step
    
    Returns:
        tuple: (success, result, updated_pipeline)
    """
    # Find the step in the pipeline
    step = None
    for s in pipeline["steps"]:
        if s["name"] == step_name:
            step = s
            break
    
    if not step:
        return False, f"Step '{step_name}' not found in pipeline", pipeline
    
    # Mark step as in progress
    step["status"] = "in_progress"
    step["started_at"] = datetime.now().isoformat()
    
    try:
        result = None
        
        if step_name == "load_dataset":
            result = load_dataset_for_tuning(**kwargs)
            
        elif step_name == "prepare_training_data":
            from .export import prepare_training_data_for_api
            result = prepare_training_data_for_api(**kwargs)
            
        elif step_name == "start_gemini_job":
            result = start_finetuning_job(**kwargs)
            
        elif step_name == "start_openai_job":
            result = start_openai_finetuning_job(**kwargs)
            
        # Add more step implementations as needed
        
        # Mark step as completed
        step["status"] = "completed"
        step["completed_at"] = datetime.now().isoformat()
        step["result"] = str(result)[:500]  # Truncate long results
        
        return True, result, pipeline
        
    except Exception as e:
        # Mark step as failed
        step["status"] = "failed"
        step["failed_at"] = datetime.now().isoformat()
        step["error"] = str(e)
        
        return False, str(e), pipeline


def get_training_progress(pipeline):
    """Get training progress summary from pipeline.
    
    Args:
        pipeline: Pipeline configuration
    
    Returns:
        dict: Progress summary
    """
    total_steps = len(pipeline["steps"])
    completed_steps = sum(1 for s in pipeline["steps"] if s["status"] == "completed")
    failed_steps = sum(1 for s in pipeline["steps"] if s["status"] == "failed")
    in_progress_steps = sum(1 for s in pipeline["steps"] if s["status"] == "in_progress")
    
    progress_percentage = (completed_steps / total_steps * 100) if total_steps > 0 else 0
    
    return {
        "total_steps": total_steps,
        "completed_steps": completed_steps,
        "failed_steps": failed_steps,
        "in_progress_steps": in_progress_steps,
        "progress_percentage": progress_percentage,
        "current_step": next((s["name"] for s in pipeline["steps"] if s["status"] == "in_progress"), None),
        "overall_status": "completed" if completed_steps == total_steps else 
                         "failed" if failed_steps > 0 else
                         "in_progress" if in_progress_steps > 0 else "pending"
    }