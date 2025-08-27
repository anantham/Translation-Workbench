"""
Enhanced Alignment Map Builder Module

Clean, directory-agnostic alignment mapping with comprehensive validation.
Decoupled from specific novel structures - works with any two directories containing chapters.
"""

import os
import json
import re
import shutil
import streamlit as st
from datetime import datetime
from typing import Dict, Tuple, Optional, List

from .logging import logger

# Error handling constants
MAX_FILE_SIZE_MB = 50  # Maximum file size to process
MIN_CONTENT_LENGTH = 10  # Minimum content length to be considered valid

def extract_chapter_number_from_filename(filename: str) -> Optional[int]:
    """Extract chapter number from various filename formats."""
    # Pattern for English-Chapter-0001.txt
    match = re.search(r'English-Chapter-(\d+)\.txt', filename)
    if match:
        return int(match.group(1))
    
    # Pattern for Chapter-0001-Title.txt
    match = re.search(r'Chapter-(\d+)', filename)
    if match:
        return int(match.group(1))
    
    # Pattern for any file with digits
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    
    return None

def validate_chapter_file(filepath: str) -> Dict[str, any]:
    """
    Validate a single chapter file for common issues.
    
    Args:
        filepath: Path to the chapter file
        
    Returns:
        Dictionary with validation results
    """
    validation = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "file_size_mb": 0,
        "content_length": 0,
        "encoding_detected": None,
        "content_preview": None
    }
    
    try:
        # Check file size
        file_size_bytes = os.path.getsize(filepath)
        validation["file_size_mb"] = file_size_bytes / (1024 * 1024)
        
        logger.debug(f"[FILE VALIDATION] {filepath}: {validation['file_size_mb']:.2f}MB")
        
        if validation["file_size_mb"] > MAX_FILE_SIZE_MB:
            validation["errors"].append(f"File too large: {validation['file_size_mb']:.1f}MB (max: {MAX_FILE_SIZE_MB}MB)")
            validation["valid"] = False
            return validation
        
        if file_size_bytes == 0:
            validation["errors"].append("File is empty (0 bytes)")
            validation["valid"] = False
            return validation
        
        # Try to read content with different encodings
        content = None
        for encoding in ['utf-8', 'gbk', 'latin-1']:
            try:
                with open(filepath, 'r', encoding=encoding) as f:
                    content = f.read()
                validation["encoding_detected"] = encoding
                logger.debug(f"[FILE VALIDATION] {filepath}: Successfully read with {encoding} encoding")
                break
            except UnicodeDecodeError:
                logger.debug(f"[FILE VALIDATION] {filepath}: Failed to read with {encoding} encoding")
                continue
            except Exception as e:
                logger.error(f"[FILE VALIDATION] {filepath}: Error reading with {encoding}: {e}")
                continue
        
        if content is None:
            validation["errors"].append("Could not read file with any encoding (utf-8, gbk, latin-1)")
            validation["valid"] = False
            return validation
        
        # Validate content
        validation["content_length"] = len(content.strip())
        validation["content_preview"] = content.strip()[:200] + "..." if len(content.strip()) > 200 else content.strip()
        
        if validation["content_length"] < MIN_CONTENT_LENGTH:
            validation["errors"].append(f"Content too short: {validation['content_length']} characters (min: {MIN_CONTENT_LENGTH})")
            validation["valid"] = False
        
        # Check for common issues
        if content.strip().startswith('<!DOCTYPE html') or '<html' in content:
            validation["errors"].append("File appears to contain HTML instead of text content")
            validation["valid"] = False
        
        if content.count('404') > 0 and content.count('Not Found') > 0:
            validation["warnings"].append("File may contain 404 error content")
        
        if validation["content_length"] > 0:
            logger.debug(f"[FILE VALIDATION] {filepath}: Content length: {validation['content_length']} chars")
            logger.debug(f"[FILE VALIDATION] {filepath}: Preview: {validation['content_preview'][:100]}...")
        
    except PermissionError:
        validation["errors"].append("Permission denied reading file")
        validation["valid"] = False
        logger.error(f"[FILE VALIDATION] {filepath}: Permission denied")
    except Exception as e:
        validation["errors"].append(f"Unexpected error validating file: {str(e)}")
        validation["valid"] = False
        logger.error(f"[FILE VALIDATION] {filepath}: Unexpected error: {e}")
    
    return validation

def scan_directory_for_chapters(directory_path: str, file_pattern: Optional[str] = None, validate_files: bool = True) -> Dict[int, Dict[str, any]]:
    """
    Scan directory and return chapter number -> file info mapping.
    
    Args:
        directory_path: Path to directory containing chapter files
        file_pattern: Optional regex pattern to filter files
        validate_files: Whether to validate each file (default: True)
        
    Returns:
        Dictionary mapping chapter numbers to file information
    """
    chapters = {}
    scan_stats = {
        "total_files": 0,
        "txt_files": 0,
        "valid_chapters": 0,
        "invalid_files": 0,
        "permission_errors": 0
    }
    
    logger.info(f"[DIRECTORY SCAN] Starting scan of: {directory_path}")
    logger.debug(f"[DIRECTORY SCAN] File pattern: {file_pattern}")
    logger.debug(f"[DIRECTORY SCAN] Validate files: {validate_files}")
    
    if not os.path.exists(directory_path):
        logger.error(f"[DIRECTORY SCAN] FAILURE: Directory does not exist: {directory_path}")
        return chapters
    
    if not os.path.isdir(directory_path):
        logger.error(f"[DIRECTORY SCAN] FAILURE: Path is not a directory: {directory_path}")
        return chapters
    
    try:
        all_files = os.listdir(directory_path)
        scan_stats["total_files"] = len(all_files)
        logger.info(f"[DIRECTORY SCAN] Found {scan_stats['total_files']} total files")
        
        for filename in all_files:
            filepath = os.path.join(directory_path, filename)
            
            # Skip non-files
            if not os.path.isfile(filepath):
                logger.debug(f"[DIRECTORY SCAN] Skipping non-file: {filename}")
                continue
            
            # Skip non-text files
            if not filename.endswith('.txt'):
                logger.debug(f"[DIRECTORY SCAN] Skipping non-txt file: {filename}")
                continue
            
            scan_stats["txt_files"] += 1
            
            # Apply file pattern filter if provided
            if file_pattern and not re.search(file_pattern, filename):
                logger.debug(f"[DIRECTORY SCAN] Skipping file not matching pattern: {filename}")
                continue
            
            # Extract chapter number
            chapter_num = extract_chapter_number_from_filename(filename)
            if not chapter_num:
                logger.warning(f"[DIRECTORY SCAN] Could not extract chapter number from: {filename}")
                continue
            
            # Create file info
            file_info = {
                "filepath": filepath,
                "filename": filename,
                "chapter_num": chapter_num,
                "validation": None
            }
            
            # Validate file if requested
            if validate_files:
                file_info["validation"] = validate_chapter_file(filepath)
                if file_info["validation"]["valid"]:
                    scan_stats["valid_chapters"] += 1
                    logger.debug(f"[DIRECTORY SCAN] Valid chapter {chapter_num}: {filename}")
                else:
                    scan_stats["invalid_files"] += 1
                    logger.warning(f"[DIRECTORY SCAN] Invalid chapter {chapter_num}: {filename}")
                    logger.warning(f"[DIRECTORY SCAN] Validation errors: {file_info['validation']['errors']}")
            else:
                scan_stats["valid_chapters"] += 1
                logger.debug(f"[DIRECTORY SCAN] Found chapter {chapter_num}: {filename} (validation skipped)")
            
            chapters[chapter_num] = file_info
    
    except PermissionError:
        scan_stats["permission_errors"] += 1
        logger.error(f"[DIRECTORY SCAN] CRITICAL: Permission denied accessing directory: {directory_path}")
        logger.error("[DIRECTORY SCAN] This is a system-level issue. Check directory permissions.")
    except Exception as e:
        logger.error(f"[DIRECTORY SCAN] CRITICAL: Unexpected error scanning directory {directory_path}: {e}")
        logger.error("[DIRECTORY SCAN] This indicates a serious problem. Check directory structure and permissions.")
    
    # Log final statistics
    logger.info(f"[DIRECTORY SCAN] Scan complete for: {directory_path}")
    logger.info(f"[DIRECTORY SCAN] Stats: {scan_stats['total_files']} total files, {scan_stats['txt_files']} txt files")
    logger.info(f"[DIRECTORY SCAN] Stats: {scan_stats['valid_chapters']} valid chapters, {scan_stats['invalid_files']} invalid files")
    if scan_stats["permission_errors"] > 0:
        logger.error(f"[DIRECTORY SCAN] Stats: {scan_stats['permission_errors']} permission errors")
    
    return chapters

def validate_chapter_directories(chinese_dir: str, english_dir: str) -> Dict[str, any]:
    """
    Validate that both directories exist and contain chapter files.
    
    Args:
        chinese_dir: Path to Chinese chapters directory
        english_dir: Path to English chapters directory
        
    Returns:
        Dictionary with validation results and statistics
    """
    logger.info("[DIRECTORY VALIDATION] Starting validation")
    logger.info(f"[DIRECTORY VALIDATION] Chinese directory: {chinese_dir}")
    logger.info(f"[DIRECTORY VALIDATION] English directory: {english_dir}")
    
    validation_result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "chinese_stats": {"exists": False, "is_dir": False, "chapter_count": 0, "valid_chapters": 0, "chapters": {}},
        "english_stats": {"exists": False, "is_dir": False, "chapter_count": 0, "valid_chapters": 0, "chapters": {}},
        "overlap_count": 0,
        "file_issues": []
    }
    
    # Validate Chinese directory
    if not chinese_dir or not chinese_dir.strip():
        validation_result["valid"] = False
        validation_result["errors"].append("Chinese directory path is empty")
        logger.error("[DIRECTORY VALIDATION] FAILURE: Chinese directory path is empty")
    else:
        chinese_dir = chinese_dir.strip()
        validation_result["chinese_stats"]["exists"] = os.path.exists(chinese_dir)
        validation_result["chinese_stats"]["is_dir"] = os.path.isdir(chinese_dir)
        
        logger.debug(f"[DIRECTORY VALIDATION] Chinese directory exists: {validation_result['chinese_stats']['exists']}")
        logger.debug(f"[DIRECTORY VALIDATION] Chinese directory is_dir: {validation_result['chinese_stats']['is_dir']}")
        
        if not validation_result["chinese_stats"]["exists"]:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Chinese directory does not exist: {chinese_dir}")
            logger.error(f"[DIRECTORY VALIDATION] FAILURE: Chinese directory does not exist: {chinese_dir}")
        elif not validation_result["chinese_stats"]["is_dir"]:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Chinese path is not a directory: {chinese_dir}")
            logger.error(f"[DIRECTORY VALIDATION] FAILURE: Chinese path is not a directory: {chinese_dir}")
        else:
            chinese_chapters = scan_directory_for_chapters(chinese_dir, validate_files=True)
            validation_result["chinese_stats"]["chapters"] = chinese_chapters
            validation_result["chinese_stats"]["chapter_count"] = len(chinese_chapters)
            
            # Count valid chapters and collect file issues
            valid_chapters = 0
            for ch_num, file_info in chinese_chapters.items():
                if file_info.get("validation", {}).get("valid", True):
                    valid_chapters += 1
                else:
                    validation_issues = file_info.get("validation", {})
                    validation_result["file_issues"].append({
                        "type": "chinese",
                        "chapter": ch_num,
                        "file": file_info["filename"],
                        "errors": validation_issues.get("errors", []),
                        "warnings": validation_issues.get("warnings", [])
                    })
            
            validation_result["chinese_stats"]["valid_chapters"] = valid_chapters
            logger.info(f"[DIRECTORY VALIDATION] Chinese: {validation_result['chinese_stats']['chapter_count']} total chapters, {valid_chapters} valid")
            
            if len(chinese_chapters) == 0:
                validation_result["warnings"].append(f"No chapter files found in Chinese directory: {chinese_dir}")
                logger.warning(f"[DIRECTORY VALIDATION] No chapter files found in Chinese directory: {chinese_dir}")
    
    # Validate English directory
    if not english_dir or not english_dir.strip():
        validation_result["valid"] = False
        validation_result["errors"].append("English directory path is empty")
        logger.error("[DIRECTORY VALIDATION] FAILURE: English directory path is empty")
    else:
        english_dir = english_dir.strip()
        validation_result["english_stats"]["exists"] = os.path.exists(english_dir)
        validation_result["english_stats"]["is_dir"] = os.path.isdir(english_dir)
        
        logger.debug(f"[DIRECTORY VALIDATION] English directory exists: {validation_result['english_stats']['exists']}")
        logger.debug(f"[DIRECTORY VALIDATION] English directory is_dir: {validation_result['english_stats']['is_dir']}")
        
        if not validation_result["english_stats"]["exists"]:
            validation_result["valid"] = False
            validation_result["errors"].append(f"English directory does not exist: {english_dir}")
            logger.error(f"[DIRECTORY VALIDATION] FAILURE: English directory does not exist: {english_dir}")
        elif not validation_result["english_stats"]["is_dir"]:
            validation_result["valid"] = False
            validation_result["errors"].append(f"English path is not a directory: {english_dir}")
            logger.error(f"[DIRECTORY VALIDATION] FAILURE: English path is not a directory: {english_dir}")
        else:
            english_chapters = scan_directory_for_chapters(english_dir, validate_files=True)
            validation_result["english_stats"]["chapters"] = english_chapters
            validation_result["english_stats"]["chapter_count"] = len(english_chapters)
            
            # Count valid chapters and collect file issues
            valid_chapters = 0
            for ch_num, file_info in english_chapters.items():
                if file_info.get("validation", {}).get("valid", True):
                    valid_chapters += 1
                else:
                    validation_issues = file_info.get("validation", {})
                    validation_result["file_issues"].append({
                        "type": "english",
                        "chapter": ch_num,
                        "file": file_info["filename"],
                        "errors": validation_issues.get("errors", []),
                        "warnings": validation_issues.get("warnings", [])
                    })
            
            validation_result["english_stats"]["valid_chapters"] = valid_chapters
            logger.info(f"[DIRECTORY VALIDATION] English: {validation_result['english_stats']['chapter_count']} total chapters, {valid_chapters} valid")
            
            if len(english_chapters) == 0:
                validation_result["warnings"].append(f"No chapter files found in English directory: {english_dir}")
                logger.warning(f"[DIRECTORY VALIDATION] No chapter files found in English directory: {english_dir}")
    
    # Calculate overlap if both directories are valid
    if (validation_result["chinese_stats"]["chapter_count"] > 0 and 
        validation_result["english_stats"]["chapter_count"] > 0):
        
        chinese_nums = set(validation_result["chinese_stats"]["chapters"].keys())
        english_nums = set(validation_result["english_stats"]["chapters"].keys())
        overlap = chinese_nums.intersection(english_nums)
        validation_result["overlap_count"] = len(overlap)
        
        logger.info(f"[DIRECTORY VALIDATION] Overlap: {len(overlap)} chapters")
        
        if len(overlap) == 0:
            validation_result["warnings"].append("No overlapping chapter numbers found between directories")
            logger.warning("[DIRECTORY VALIDATION] No overlapping chapter numbers found between directories")
    
    # Log file issues summary
    if validation_result["file_issues"]:
        logger.warning(f"[DIRECTORY VALIDATION] Found {len(validation_result['file_issues'])} file issues:")
        for issue in validation_result["file_issues"]:
            logger.warning(f"[DIRECTORY VALIDATION] {issue['type']} chapter {issue['chapter']}: {issue['file']}")
            for error in issue["errors"]:
                logger.warning(f"[DIRECTORY VALIDATION]   ERROR: {error}")
            for warning in issue["warnings"]:
                logger.warning(f"[DIRECTORY VALIDATION]   WARNING: {warning}")
    
    logger.info(f"[DIRECTORY VALIDATION] Validation complete. Overall valid: {validation_result['valid']}")
    
    return validation_result

def preview_alignment_mapping(chinese_dir: str, english_dir: str) -> Dict[str, any]:
    """
    Preview what the alignment map would look like without building it.
    
    Args:
        chinese_dir: Path to Chinese chapters directory
        english_dir: Path to English chapters directory
        
    Returns:
        Dictionary with preview information
    """
    logger.info("[ALIGNMENT PREVIEW] Starting alignment preview")
    logger.info(f"[ALIGNMENT PREVIEW] Chinese directory: {chinese_dir}")
    logger.info(f"[ALIGNMENT PREVIEW] English directory: {english_dir}")
    
    validation = validate_chapter_directories(chinese_dir, english_dir)
    
    if not validation["valid"]:
        logger.error(f"[ALIGNMENT PREVIEW] Validation failed: {validation['errors']}")
        return {
            "success": False,
            "errors": validation["errors"],
            "warnings": validation["warnings"],
            "file_issues": validation.get("file_issues", [])
        }
    
    chinese_chapters = validation["chinese_stats"]["chapters"]
    english_chapters = validation["english_stats"]["chapters"]
    
    # Generate preview mapping
    preview_mapping = {}
    all_chapter_nums = set(chinese_chapters.keys()) | set(english_chapters.keys())
    
    logger.info(f"[ALIGNMENT PREVIEW] Processing {len(all_chapter_nums)} unique chapter numbers")
    
    for ch_num in sorted(all_chapter_nums):
        ch_key = str(ch_num)
        chinese_file_info = chinese_chapters.get(ch_num)
        english_file_info = english_chapters.get(ch_num)
        
        # Extract file paths
        chinese_file = chinese_file_info["filepath"] if chinese_file_info else None
        english_file = english_file_info["filepath"] if english_file_info else None
        
        # Determine status
        if chinese_file and english_file:
            status = "both"
        elif chinese_file:
            status = "chinese_only"
        elif english_file:
            status = "english_only"
        else:
            status = "missing"  # This shouldn't happen
        
        # Check file validity
        chinese_valid = chinese_file_info.get("validation", {}).get("valid", True) if chinese_file_info else False
        english_valid = english_file_info.get("validation", {}).get("valid", True) if english_file_info else False
        
        preview_mapping[ch_key] = {
            "raw_file": chinese_file,
            "english_file": english_file,
            "status": status,
            "chinese_valid": chinese_valid,
            "english_valid": english_valid,
            "chinese_issues": chinese_file_info.get("validation", {}).get("errors", []) + chinese_file_info.get("validation", {}).get("warnings", []) if chinese_file_info else [],
            "english_issues": english_file_info.get("validation", {}).get("errors", []) + english_file_info.get("validation", {}).get("warnings", []) if english_file_info else []
        }
        
        logger.debug(f"[ALIGNMENT PREVIEW] Chapter {ch_num}: {status} (Chinese valid: {chinese_valid}, English valid: {english_valid})")
    
    # Generate statistics
    stats = {
        "total_mappings": len(preview_mapping),
        "both_files": len([m for m in preview_mapping.values() if m["status"] == "both"]),
        "chinese_only": len([m for m in preview_mapping.values() if m["status"] == "chinese_only"]),
        "english_only": len([m for m in preview_mapping.values() if m["status"] == "english_only"]),
        "chinese_total": len(chinese_chapters),
        "english_total": len(english_chapters),
        "chinese_valid": validation["chinese_stats"]["valid_chapters"],
        "english_valid": validation["english_stats"]["valid_chapters"],
        "problematic_files": len(validation.get("file_issues", []))
    }
    
    logger.info("[ALIGNMENT PREVIEW] Preview complete:")
    logger.info(f"[ALIGNMENT PREVIEW] Total mappings: {stats['total_mappings']}")
    logger.info(f"[ALIGNMENT PREVIEW] Both files: {stats['both_files']}")
    logger.info(f"[ALIGNMENT PREVIEW] Chinese only: {stats['chinese_only']}")
    logger.info(f"[ALIGNMENT PREVIEW] English only: {stats['english_only']}")
    logger.info(f"[ALIGNMENT PREVIEW] Problematic files: {stats['problematic_files']}")
    
    return {
        "success": True,
        "preview_mapping": preview_mapping,
        "stats": stats,
        "warnings": validation["warnings"],
        "file_issues": validation.get("file_issues", [])
    }

def build_alignment_map_from_directories(chinese_dir: str, english_dir: str, output_path: str) -> Tuple[Dict, Dict]:
    """
    Build alignment map from two directories containing chapter files.
    
    Args:
        chinese_dir: Path to Chinese chapters directory
        english_dir: Path to English chapters directory
        output_path: Path where alignment map should be saved
        
    Returns:
        Tuple of (alignment_map, build_stats)
    """
    logger.info("[ALIGNMENT BUILD] Starting alignment map build")
    logger.info(f"[ALIGNMENT BUILD] Chinese directory: {chinese_dir}")
    logger.info(f"[ALIGNMENT BUILD] English directory: {english_dir}")
    logger.info(f"[ALIGNMENT BUILD] Output path: {output_path}")
    
    build_stats = {
        "success": False,
        "chinese_dir": chinese_dir,
        "english_dir": english_dir,
        "output_path": output_path,
        "chinese_count": 0,
        "english_count": 0,
        "chinese_valid": 0,
        "english_valid": 0,
        "total_mapped": 0,
        "both_files": 0,
        "chinese_only": 0,
        "english_only": 0,
        "problematic_files": 0,
        "errors": [],
        "warnings": [],
        "file_issues": []
    }
    
    # Validate directories
    validation = validate_chapter_directories(chinese_dir, english_dir)
    
    if not validation["valid"]:
        build_stats["errors"] = validation["errors"]
        build_stats["warnings"] = validation["warnings"]
        build_stats["file_issues"] = validation.get("file_issues", [])
        logger.error(f"[ALIGNMENT BUILD] Validation failed: {validation['errors']}")
        return {}, build_stats
    
    build_stats["warnings"] = validation["warnings"]
    build_stats["file_issues"] = validation.get("file_issues", [])
    
    try:
        # Get chapters from validation
        chinese_chapters = validation["chinese_stats"]["chapters"]
        english_chapters = validation["english_stats"]["chapters"]
        
        build_stats["chinese_count"] = len(chinese_chapters)
        build_stats["english_count"] = len(english_chapters)
        build_stats["chinese_valid"] = validation["chinese_stats"]["valid_chapters"]
        build_stats["english_valid"] = validation["english_stats"]["valid_chapters"]
        build_stats["problematic_files"] = len(validation.get("file_issues", []))
        
        logger.info(f"[ALIGNMENT BUILD] Processing {build_stats['chinese_count']} Chinese chapters")
        logger.info(f"[ALIGNMENT BUILD] Processing {build_stats['english_count']} English chapters")
        logger.info(f"[ALIGNMENT BUILD] {build_stats['chinese_valid']} Chinese chapters valid")
        logger.info(f"[ALIGNMENT BUILD] {build_stats['english_valid']} English chapters valid")
        
        # Build alignment map
        alignment_map = {}
        all_chapter_nums = set(chinese_chapters.keys()) | set(english_chapters.keys())
        
        logger.info(f"[ALIGNMENT BUILD] Building alignment map for {len(all_chapter_nums)} unique chapters")
        
        for ch_num in all_chapter_nums:
            ch_key = str(ch_num)
            chinese_file_info = chinese_chapters.get(ch_num)
            english_file_info = english_chapters.get(ch_num)
            
            # Extract file paths (only for valid files)
            chinese_file = None
            english_file = None
            
            if chinese_file_info and chinese_file_info.get("validation", {}).get("valid", True):
                chinese_file = chinese_file_info["filepath"]
            elif chinese_file_info:
                logger.warning(f"[ALIGNMENT BUILD] Skipping invalid Chinese file for chapter {ch_num}: {chinese_file_info['filename']}")
            
            if english_file_info and english_file_info.get("validation", {}).get("valid", True):
                english_file = english_file_info["filepath"]
            elif english_file_info:
                logger.warning(f"[ALIGNMENT BUILD] Skipping invalid English file for chapter {ch_num}: {english_file_info['filename']}")
            
            # Create alignment map entry
            alignment_map[ch_key] = {
                "raw_file": chinese_file,
                "english_file": english_file
            }
            
            # Track statistics
            if chinese_file and english_file:
                build_stats["both_files"] += 1
                logger.debug(f"[ALIGNMENT BUILD] Chapter {ch_num}: Both files available")
            elif chinese_file:
                build_stats["chinese_only"] += 1
                logger.debug(f"[ALIGNMENT BUILD] Chapter {ch_num}: Chinese only")
            elif english_file:
                build_stats["english_only"] += 1
                logger.debug(f"[ALIGNMENT BUILD] Chapter {ch_num}: English only")
            else:
                logger.warning(f"[ALIGNMENT BUILD] Chapter {ch_num}: No valid files found")
        
        build_stats["total_mapped"] = len(alignment_map)
        build_stats["success"] = True
        
        logger.info("[ALIGNMENT BUILD] Alignment map built successfully!")
        logger.info(f"[ALIGNMENT BUILD] Total mapped: {build_stats['total_mapped']} chapters")
        logger.info(f"[ALIGNMENT BUILD] Both files: {build_stats['both_files']}")
        logger.info(f"[ALIGNMENT BUILD] Chinese only: {build_stats['chinese_only']}")
        logger.info(f"[ALIGNMENT BUILD] English only: {build_stats['english_only']}")
        logger.info(f"[ALIGNMENT BUILD] Problematic files: {build_stats['problematic_files']}")
        
        return alignment_map, build_stats
        
    except Exception as e:
        error_msg = f"Critical error building alignment map: {str(e)}"
        logger.error(f"[ALIGNMENT BUILD] CRITICAL ERROR: {error_msg}")
        logger.error("[ALIGNMENT BUILD] This is a serious issue - check directory structure and file permissions")
        build_stats["errors"].append(error_msg)
        return {}, build_stats

def save_alignment_map_with_backup(alignment_map: Dict, output_path: str) -> Tuple[bool, str]:
    """
    Save alignment map with backup of existing file.
    
    Args:
        alignment_map: The alignment map to save
        output_path: Path where to save the alignment map
        
    Returns:
        Tuple of (success, message)
    """
    try:
        # Create backup directory
        backup_dir = os.path.join(os.path.dirname(output_path), "backups")
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create backup if file exists
        if os.path.exists(output_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{os.path.basename(output_path)}.backup_{timestamp}"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            shutil.copy(output_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
        
        # Create directory for output file if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Save new alignment map
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(alignment_map, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved alignment map to: {output_path}")
        return True, f"Alignment map saved successfully to {output_path}"
        
    except Exception as e:
        error_msg = f"Error saving alignment map: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def get_alignment_map_path(novel_name: str) -> str:
    """
    Get the standard path for an alignment map in the central alignments directory.
    
    Args:
        novel_name: Name of the novel (used for filename)
        
    Returns:
        Path to the alignment map file
    """
    alignments_dir = os.path.join("data", "alignments")
    os.makedirs(alignments_dir, exist_ok=True)
    
    # Sanitize novel name for filename
    safe_novel_name = re.sub(r'[^\w\s-]', '', novel_name).strip()
    safe_novel_name = re.sub(r'[-\s]+', '_', safe_novel_name)
    
    return os.path.join(alignments_dir, f"{safe_novel_name}_alignment_map.json")

def build_and_save_alignment_map(chinese_dir: str, english_dir: str, novel_name: str, output_path: str = None) -> Tuple[bool, str, Dict]:
    """
    Complete workflow: build alignment map from directories and save it.
    
    Args:
        chinese_dir: Path to Chinese chapters directory
        english_dir: Path to English chapters directory
        novel_name: Name of the novel (used for default filename)
        output_path: Optional custom path where to save the alignment map
        
    Returns:
        Tuple of (success, message, build_stats)
    """
    # Use central alignments directory if no custom path provided
    if output_path is None:
        output_path = get_alignment_map_path(novel_name)
    # Build alignment map
    alignment_map, build_stats = build_alignment_map_from_directories(chinese_dir, english_dir, output_path)
    
    if not build_stats["success"]:
        error_msg = "Failed to build alignment map: " + "; ".join(build_stats["errors"])
        return False, error_msg, build_stats
    
    if not alignment_map:
        return False, "No chapters found to map", build_stats
    
    # Save alignment map
    save_success, save_message = save_alignment_map_with_backup(alignment_map, output_path)
    
    if save_success:
        success_msg = "✅ Alignment map built and saved successfully!\n"
        success_msg += f"📊 Total mapped: {build_stats['total_mapped']} chapters\n"
        success_msg += f"🔍 Both files: {build_stats['both_files']} chapters\n"
        success_msg += f"🇨🇳 Chinese only: {build_stats['chinese_only']} chapters\n"
        success_msg += f"🇺🇸 English only: {build_stats['english_only']} chapters\n"
        success_msg += f"📁 Saved to: {output_path}"
        
        # Clear caches when new alignment map is built
        logger.info("[ALIGNMENT MAPS] Clearing caches due to new alignment map")
        list_alignment_maps.clear()  # Clear directory scan cache
        load_alignment_map_by_slug.clear()  # Clear content cache
        
        return True, success_msg, build_stats
    else:
        return False, save_message, build_stats


# =============================================================================
# UNIFIED ALIGNMENT MAP MANAGEMENT FUNCTIONS
# =============================================================================

@st.cache_data(ttl=300)  # Cache directory scan for 5 minutes
def list_alignment_maps() -> Dict[str, str]:
    """
    Return {slug: path} for every *_alignment_map.json file in the central location.
    
    Returns:
        Dictionary mapping slug names to alignment map file paths
    """
    logger.info("[ALIGNMENT MAPS] Scanning for available alignment maps")
    
    alignments_dir = os.path.join("data", "alignments")
    alignment_maps = {}
    
    if not os.path.exists(alignments_dir):
        logger.warning(f"[ALIGNMENT MAPS] Alignments directory not found: {alignments_dir}")
        return alignment_maps
    
    # Scan for alignment map files
    for filename in os.listdir(alignments_dir):
        if filename.endswith('_alignment_map.json'):
            # Extract slug from filename: "way_of_the_devil_alignment_map.json" -> "way_of_the_devil"
            slug = filename.replace('_alignment_map.json', '')
            full_path = os.path.join(alignments_dir, filename)
            alignment_maps[slug] = full_path
            logger.debug(f"[ALIGNMENT MAPS] Found: {slug} -> {full_path}")
    
    logger.info(f"[ALIGNMENT MAPS] Found {len(alignment_maps)} alignment maps")
    return alignment_maps


def get_alignment_map_for_slug(slug: str = None) -> str:
    """
    Get alignment map path by slug with smart fallback.
    
    Args:
        slug: Optional slug name. If None, tries to find single available map.
        
    Returns:
        Path to alignment map file
        
    Raises:
        FileNotFoundError: If no alignment map found for slug
        ValueError: If multiple maps exist but no slug specified
    """
    maps = list_alignment_maps()
    
    if slug is None:
        if len(maps) == 0:
            raise FileNotFoundError("No alignment maps found in data/alignments/")
        elif len(maps) == 1:
            slug = next(iter(maps.keys()))
            logger.info(f"[ALIGNMENT MAPS] Auto-selected single available map: {slug}")
            return maps[slug]
        else:
            available_slugs = ', '.join(maps.keys())
            raise ValueError(f"Multiple alignment maps found: {available_slugs}. Please specify slug explicitly.")
    
    if slug not in maps:
        available_slugs = ', '.join(maps.keys()) if maps else "none"
        raise FileNotFoundError(f"No alignment map found for slug '{slug}'. Available: {available_slugs}")
    
    logger.info(f"[ALIGNMENT MAPS] Using alignment map for slug: {slug}")
    return maps[slug]


@st.cache_data  # Cache alignment map content indefinitely during session
def load_alignment_map_by_slug(slug: str = None, chapters: List[int] = None) -> Dict[str, Dict]:
    """
    Load alignment map by slug with optional chapter filtering.
    
    Args:
        slug: Optional slug name. If None, tries to find single available map.
        chapters: Optional list of chapter numbers to filter by
        
    Returns:
        Alignment map dictionary
    """
    logger.info(f"[ALIGNMENT MAPS] Loading alignment map for slug: {slug}")
    
    # Get the path for this slug
    alignment_path = get_alignment_map_for_slug(slug)
    
    # Load the alignment map
    with open(alignment_path, 'r', encoding='utf-8') as f:
        alignment_map = json.load(f)
    
    logger.info(f"[ALIGNMENT MAPS] Loaded alignment map with {len(alignment_map)} chapters")
    
    # Filter by chapters if specified
    if chapters is not None:
        logger.info(f"[ALIGNMENT MAPS] Filtering to {len(chapters)} specified chapters")
        chapter_strs = [str(ch) for ch in chapters]
        filtered_map = {k: v for k, v in alignment_map.items() if k in chapter_strs}
        logger.info(f"[ALIGNMENT MAPS] Filtered alignment map to {len(filtered_map)} chapters")
        return filtered_map
    
    return alignment_map


def parse_chapter_ranges(range_str: str) -> List[int]:
    """
    Parse chapter range string like "1-100,102,105-110" into list of integers.
    
    Args:
        range_str: Range string (e.g. "1-100,102,105-110")
        
    Returns:
        List of chapter numbers
    """
    if not range_str or not range_str.strip():
        return []
    
    chapters = []
    
    # Split by comma and process each part
    parts = [part.strip() for part in range_str.split(',')]
    
    for part in parts:
        if '-' in part:
            # Handle range like "1-100"
            try:
                start, end = part.split('-', 1)
                start_num = int(start.strip())
                end_num = int(end.strip())
                chapters.extend(range(start_num, end_num + 1))
            except ValueError:
                logger.warning(f"[CHAPTER RANGES] Invalid range format: {part}")
        else:
            # Handle single number like "102"
            try:
                chapters.append(int(part))
            except ValueError:
                logger.warning(f"[CHAPTER RANGES] Invalid chapter number: {part}")
    
    # Remove duplicates and sort
    chapters = sorted(list(set(chapters)))
    logger.info(f"[CHAPTER RANGES] Parsed '{range_str}' into {len(chapters)} chapters")
    
    return chapters