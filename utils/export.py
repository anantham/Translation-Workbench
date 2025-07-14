"""
Data Export and Dataset Generation Module

ML-ready dataset exports and production-quality data formatting.
Supports multiple formats, train/validation splits, and metadata management.

This module handles:
- JSONL export for fine-tuning datasets
- Multi-format training data creation (OpenAI, Gemini, Custom)
- Train/validation data splitting with statistics
- Dataset validation and quality checks
- Export metadata and progress tracking
- File format conversions and optimizations
"""

import os
import json
import random
from datetime import datetime

# Import configuration
from .config import DATA_DIR, EXPORT_DIR


def export_training_data_to_jsonl(training_data, output_path):
    """Export training data to JSONL format.
    
    Args:
        training_data: List of training examples
        output_path: Path to output JSONL file
    
    Returns:
        tuple: (success, message)
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in training_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        return True, f"Exported {len(training_data)} examples to {output_path}"
    except Exception as e:
        return False, f"Export failed: {e}"


def create_translation_jsonl(training_examples, train_split=0.8, format_type="OpenAI Fine-tuning", system_prompt=None):
    """Create JSONL training files for translation fine-tuning.
    
    Args:
        training_examples: List of training example dictionaries
        train_split: Fraction of data to use for training (rest for validation)
        format_type: Format for JSONL ("OpenAI Fine-tuning", "Gemini Fine-tuning", "Custom Messages")
        system_prompt: Optional system prompt for translation task
    
    Returns:
        tuple: (train_jsonl_content, val_jsonl_content, stats_dict)
    """
    # Shuffle the data for better train/val split
    examples = training_examples.copy()
    random.shuffle(examples)
    
    # Split into train and validation
    split_idx = int(len(examples) * train_split)
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]
    
    def format_example(example, format_type, system_prompt=None):
        """Format a single example according to the specified format."""
        chinese_text = example['raw_content']
        english_text = example['english_content']
        
        if format_type == "OpenAI Fine-tuning":
            # Standard OpenAI format: messages array with user/assistant roles
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.extend([
                {"role": "user", "content": chinese_text},
                {"role": "assistant", "content": english_text}
            ])
            
            return {"messages": messages}
        
        elif format_type == "Gemini Fine-tuning":
            # Gemini format: input_text and output_text
            formatted = {
                "input_text": chinese_text,
                "output_text": english_text
            }
            
            if system_prompt:
                formatted["input_text"] = f"{system_prompt}\n\nTranslate the following Chinese text to English:\n\n{chinese_text}"
            
            return formatted
        
        elif format_type == "Custom Messages":
            # Custom format similar to your sample but for translation
            return {
                "messages": [
                    {"role": "user", "content": f"Translate this Chinese text to English: {chinese_text}"},
                    {"role": "assistant", "content": english_text}
                ]
            }
        
        else:
            # Default to OpenAI format
            return format_example(example, "OpenAI Fine-tuning", system_prompt)
    
    # Convert examples to JSONL format
    train_jsonl_lines = []
    val_jsonl_lines = []
    
    for example in train_examples:
        formatted = format_example(example, format_type, system_prompt)
        train_jsonl_lines.append(json.dumps(formatted, ensure_ascii=False))
    
    for example in val_examples:
        formatted = format_example(example, format_type, system_prompt)
        val_jsonl_lines.append(json.dumps(formatted, ensure_ascii=False))
    
    # Join lines with newlines
    train_jsonl_content = '\n'.join(train_jsonl_lines)
    val_jsonl_content = '\n'.join(val_jsonl_lines)
    
    # Create statistics
    stats = {
        'total_count': len(examples),
        'train_count': len(train_examples),
        'val_count': len(val_examples),
        'train_split': train_split,
        'format_type': format_type,
        'has_system_prompt': system_prompt is not None
    }
    
    return train_jsonl_content, val_jsonl_content, stats


def prepare_training_data_for_api(training_examples, train_split=0.8, max_output_chars=4500):
    """Convert training examples to the format expected by fine-tuning APIs.
    
    Automatically chunks long chapters to stay under character limits.
    
    Args:
        training_examples: List of training examples
        train_split: Fraction for training data
        max_output_chars: Maximum characters per output chunk
    
    Returns:
        tuple: (train_data, val_data) in API format
    """
    # Convert chapters to chunks
    all_chunks = []
    total_chapters = 0
    total_chunks = 0
    over_limit_count = 0
    
    for example in training_examples:
        total_chapters += 1
        
        # Check if chapter needs chunking
        if len(example['english_content']) > max_output_chars:
            # Chunk this chapter
            chunks = chunk_chapter_for_training(
                example['raw_content'], 
                example['english_content'],
                max_output_chars
            )
            over_limit_count += 1
        else:
            # Use whole chapter as single chunk
            chunks = [{
                'raw': example['raw_content'],
                'english': example['english_content']
            }]
        
        # Format each chunk
        for i, chunk in enumerate(chunks):
            formatted_chunk = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a professional translator specializing in Chinese to English translation of web novels. Provide accurate, fluent translations that preserve the original meaning and style."
                    },
                    {
                        "role": "user",
                        "content": f"Translate this Chinese web novel excerpt to English:\n\n{chunk['raw']}"
                    },
                    {
                        "role": "assistant",
                        "content": chunk['english']
                    }
                ],
                "metadata": {
                    "chapter": example['chapter_number'],
                    "chunk": i + 1,
                    "total_chunks": len(chunks),
                    "is_chunked": len(chunks) > 1
                }
            }
            all_chunks.append(formatted_chunk)
            total_chunks += 1
    
    print(f"📊 Chunking Summary:")
    print(f"   • {total_chapters} chapters processed")
    print(f"   • {over_limit_count} chapters required chunking")
    print(f"   • {total_chunks} total training examples created")
    print(f"   • Average {total_chunks/total_chapters:.1f} chunks per chapter")
    
    # Shuffle and split chunks
    random.shuffle(all_chunks)
    split_idx = int(len(all_chunks) * train_split)
    
    train_data = all_chunks[:split_idx]
    val_data = all_chunks[split_idx:]
    
    # Verify no chunks exceed the limit
    oversized_train = sum(1 for ex in train_data if len(ex['messages'][2]['content']) > max_output_chars)
    oversized_val = sum(1 for ex in val_data if len(ex['messages'][2]['content']) > max_output_chars)
    
    if oversized_train > 0 or oversized_val > 0:
        print(f"⚠️ Warning: {oversized_train + oversized_val} chunks still exceed {max_output_chars} chars")
    else:
        print(f"✅ All chunks are under {max_output_chars} characters")
    
    return train_data, val_data


def chunk_chapter_for_training(raw_content, english_content, max_output_chars=4500):
    """Chunk a chapter into smaller pieces for training.
    
    Args:
        raw_content: Chinese raw text
        english_content: English translation
        max_output_chars: Maximum characters per chunk
    
    Returns:
        list: List of chunks with raw and english content
    """
    chunks = []
    
    # Split by paragraphs first
    raw_paragraphs = raw_content.split('\n\n')
    english_paragraphs = english_content.split('\n\n')
    
    # If unequal paragraph counts, fall back to character-based chunking
    if len(raw_paragraphs) != len(english_paragraphs):
        return chunk_by_characters(raw_content, english_content, max_output_chars)
    
    # Group paragraphs into chunks
    current_raw = []
    current_english = []
    current_english_chars = 0
    
    for raw_para, eng_para in zip(raw_paragraphs, english_paragraphs):
        # Check if adding this paragraph would exceed the limit
        if current_english_chars + len(eng_para) > max_output_chars and current_english:
            # Save current chunk
            chunks.append({
                'raw': '\n\n'.join(current_raw),
                'english': '\n\n'.join(current_english)
            })
            
            # Start new chunk
            current_raw = [raw_para]
            current_english = [eng_para]
            current_english_chars = len(eng_para)
        else:
            # Add to current chunk
            current_raw.append(raw_para)
            current_english.append(eng_para)
            current_english_chars += len(eng_para)
    
    # Add final chunk if exists
    if current_english:
        chunks.append({
            'raw': '\n\n'.join(current_raw),
            'english': '\n\n'.join(current_english)
        })
    
    return chunks


def chunk_by_characters(raw_content, english_content, max_output_chars=4500):
    """Fallback chunking method based on character limits.
    
    Args:
        raw_content: Chinese raw text
        english_content: English translation
        max_output_chars: Maximum characters per chunk
    
    Returns:
        list: List of character-based chunks
    """
    chunks = []
    
    # Calculate ratio between raw and English lengths
    ratio = len(raw_content) / len(english_content) if len(english_content) > 0 else 1
    max_raw_chars = int(max_output_chars * ratio)
    
    raw_pos = 0
    eng_pos = 0
    
    while eng_pos < len(english_content):
        # Calculate chunk sizes
        eng_chunk_end = min(eng_pos + max_output_chars, len(english_content))
        raw_chunk_end = min(raw_pos + max_raw_chars, len(raw_content))
        
        # Try to break at word/sentence boundaries
        if eng_chunk_end < len(english_content):
            # Look for sentence end
            for i in range(eng_chunk_end, max(eng_pos, eng_chunk_end - 100), -1):
                if english_content[i] in '.!?':
                    eng_chunk_end = i + 1
                    break
        
        if raw_chunk_end < len(raw_content):
            # Look for Chinese sentence end
            for i in range(raw_chunk_end, max(raw_pos, raw_chunk_end - 100), -1):
                if raw_content[i] in '。！？':
                    raw_chunk_end = i + 1
                    break
        
        # Create chunk
        chunks.append({
            'raw': raw_content[raw_pos:raw_chunk_end],
            'english': english_content[eng_pos:eng_chunk_end]
        })
        
        raw_pos = raw_chunk_end
        eng_pos = eng_chunk_end
    
    return chunks


def create_dataset_report(training_examples, output_path=None):
    """Create a comprehensive report about the training dataset.
    
    Args:
        training_examples: List of training examples
        output_path: Optional path to save CSV report
    
    Returns:
        dict: Dataset statistics and report data
    """
    if not training_examples:
        return {"error": "No training examples provided"}
    
    # Import text statistics function
    from .data_management import get_text_statistics
    
    report_data = []
    total_raw_chars = 0
    total_eng_chars = 0
    total_raw_words = 0
    total_eng_words = 0
    
    for example in training_examples:
        raw_stats = get_text_statistics(example['raw_content'], 'chinese')
        eng_stats = get_text_statistics(example['english_content'], 'english')
        
        # Calculate length ratio
        length_ratio = eng_stats['char_count'] / raw_stats['char_count'] if raw_stats['char_count'] > 0 else 0
        
        row = {
            'chapter': example.get('chapter_number', 'unknown'),
            'raw_chars': raw_stats['char_count'],
            'raw_words': raw_stats['word_count'],
            'eng_chars': eng_stats['char_count'],
            'eng_words': eng_stats['word_count'],
            'length_ratio': round(length_ratio, 3),
            'raw_lines': raw_stats['line_count'],
            'eng_lines': eng_stats['line_count']
        }
        
        report_data.append(row)
        total_raw_chars += raw_stats['char_count']
        total_eng_chars += eng_stats['char_count']
        total_raw_words += raw_stats['word_count']
        total_eng_words += eng_stats['word_count']
    
    # Calculate summary statistics
    summary = {
        'total_examples': len(training_examples),
        'total_raw_chars': total_raw_chars,
        'total_eng_chars': total_eng_chars,
        'total_raw_words': total_raw_words,
        'total_eng_words': total_eng_words,
        'avg_raw_chars': total_raw_chars / len(training_examples),
        'avg_eng_chars': total_eng_chars / len(training_examples),
        'avg_length_ratio': total_eng_chars / total_raw_chars if total_raw_chars > 0 else 0,
        'report_generated': datetime.now().isoformat()
    }
    
    # Save CSV if requested
    if output_path:
        try:
            import pandas as pd
            df = pd.DataFrame(report_data)
            df.to_csv(output_path, index=False)
            summary['csv_exported'] = output_path
        except ImportError:
            # Fallback CSV writing without pandas
            with open(output_path, 'w', encoding='utf-8') as f:
                if report_data:
                    # Write header
                    f.write(','.join(report_data[0].keys()) + '\n')
                    # Write data
                    for row in report_data:
                        f.write(','.join(str(v) for v in row.values()) + '\n')
            summary['csv_exported'] = output_path
    
    return {
        'summary': summary,
        'data': report_data
    }


def export_dataset_with_metadata(training_examples, export_name=None, formats=None):
    """Export training dataset in multiple formats with comprehensive metadata.
    
    Args:
        training_examples: List of training examples
        export_name: Name for the export (default: timestamp)
        formats: List of formats to export ('jsonl', 'csv', 'report')
    
    Returns:
        dict: Export results and file paths
    """
    if export_name is None:
        export_name = f"dataset_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    if formats is None:
        formats = ['jsonl', 'csv', 'report']
    
    # Ensure export directory exists
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    results = {
        'export_name': export_name,
        'timestamp': datetime.now().isoformat(),
        'total_examples': len(training_examples),
        'files_created': [],
        'formats': formats
    }
    
    # Export JSONL
    if 'jsonl' in formats:
        train_jsonl, val_jsonl, stats = create_translation_jsonl(training_examples)
        
        train_path = os.path.join(EXPORT_DIR, f"{export_name}_train.jsonl")
        val_path = os.path.join(EXPORT_DIR, f"{export_name}_val.jsonl")
        
        with open(train_path, 'w', encoding='utf-8') as f:
            f.write(train_jsonl)
        with open(val_path, 'w', encoding='utf-8') as f:
            f.write(val_jsonl)
        
        results['files_created'].extend([train_path, val_path])
        results['jsonl_stats'] = stats
    
    # Export CSV report
    if 'csv' in formats:
        csv_path = os.path.join(EXPORT_DIR, f"{export_name}_report.csv")
        report = create_dataset_report(training_examples, csv_path)
        results['files_created'].append(csv_path)
        results['report_summary'] = report['summary']
    
    # Export metadata
    if 'report' in formats:
        metadata_path = os.path.join(EXPORT_DIR, f"{export_name}_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        results['files_created'].append(metadata_path)
    
    return results