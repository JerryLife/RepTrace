"""
Data processing utilities for LLM DNA project.
"""

import logging
import json
import os
import pickle
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd


def get_cache_dir(env_var: str = "REPTRACE_CACHE_DIR", default_dir: str = "cache") -> Path:
    """
    Resolve a writable cache directory.

    Uses REPTRACE_CACHE_DIR if set, otherwise defaults to ./cache relative to CWD.
    """
    env_value = os.environ.get(env_var, "").strip()
    if env_value:
        cache_dir = Path(env_value).expanduser()
    else:
        cache_dir = Path.cwd() / default_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
        format_string: Custom format string
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure basic logging
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True
    )
    
    # Reduce verbosity of some external libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    return config


def save_results(
    data: Dict[str, Any],
    output_path: Union[str, Path],
    format: str = "auto"
) -> None:
    """
    Save results to file in various formats.
    
    Args:
        data: Data to save
        output_path: Output file path
        format: Output format ("auto", "json", "yaml", "pickle", "npz")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "auto":
        format = output_path.suffix.lower().lstrip('.')
    
    if format == "json":
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    elif format in ["yaml", "yml"]:
        with open(output_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    elif format in ["pickle", "pkl"]:
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
    elif format == "npz":
        # For numpy arrays
        np.savez_compressed(output_path, **data)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_results(
    input_path: Union[str, Path],
    format: str = "auto"
) -> Dict[str, Any]:
    """
    Load results from file.
    
    Args:
        input_path: Input file path
        format: Input format ("auto", "json", "yaml", "pickle", "npz")
        
    Returns:
        Loaded data
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if format == "auto":
        format = input_path.suffix.lower().lstrip('.')
    
    if format == "json":
        with open(input_path, 'r') as f:
            return json.load(f)
    elif format in ["yaml", "yml"]:
        with open(input_path, 'r') as f:
            return yaml.safe_load(f)
    elif format in ["pickle", "pkl"]:
        with open(input_path, 'rb') as f:
            return pickle.load(f)
    elif format == "npz":
        data = np.load(input_path, allow_pickle=True)
        return {key: data[key] for key in data.files}
    else:
        raise ValueError(f"Unsupported format: {format}")


def normalize_model_name(model_name: str) -> str:
    """
    Normalize model name for use in file paths and identifiers.
    
    Args:
        model_name: Original model name
        
    Returns:
        Normalized model name
    """
    # Replace problematic characters
    normalized = model_name.replace('/', '_').replace('\\', '_')
    normalized = normalized.replace(':', '_').replace(' ', '_')
    normalized = normalized.replace('-', '_').lower()
    
    return normalized


def create_experiment_id(prefix: str = "exp") -> str:
    """
    Create unique experiment identifier.
    
    Args:
        prefix: Prefix for experiment ID
        
    Returns:
        Unique experiment ID
    """
    import datetime
    import random
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = random.randint(1000, 9999)
    
    return f"{prefix}_{timestamp}_{random_suffix}"


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    file_path = Path(file_path)
    if file_path.exists():
        return file_path.stat().st_size / (1024 * 1024)
    return 0.0


def validate_probe_text(text: str) -> bool:
    """
    Validate probe text for quality and appropriateness.
    
    Args:
        text: Probe text to validate
        
    Returns:
        True if text is valid
    """
    if not text or len(text.strip()) < 3:
        return False
    
    # Check for minimum and maximum length
    word_count = len(text.split())
    if word_count < 2 or word_count > 100:
        return False
    
    # Check for basic quality indicators
    text_lower = text.lower()
    
    # Reject texts with too many repeated characters
    if any(char * 4 in text_lower for char in 'abcdefghijklmnopqrstuvwxyz'):
        return False
    
    # Reject texts that are mostly numbers or symbols
    alphanumeric_ratio = sum(c.isalnum() for c in text) / len(text)
    if alphanumeric_ratio < 0.7:
        return False
    
    return True


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    import re
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    # Remove control characters
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
    
    return text


def compute_text_statistics(texts: list) -> Dict[str, Any]:
    """
    Compute statistics for a collection of texts.
    
    Args:
        texts: List of text strings
        
    Returns:
        Dictionary with text statistics
    """
    if not texts:
        return {}
    
    # Length statistics
    lengths = [len(text) for text in texts]
    word_counts = [len(text.split()) for text in texts]
    
    # Vocabulary statistics
    all_words = []
    for text in texts:
        all_words.extend(text.lower().split())
    
    unique_words = set(all_words)
    
    stats = {
        "count": len(texts),
        "total_characters": sum(lengths),
        "total_words": sum(word_counts),
        "unique_words": len(unique_words),
        "avg_length": np.mean(lengths),
        "avg_word_count": np.mean(word_counts),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "length_std": np.std(lengths),
        "word_count_std": np.std(word_counts),
        "vocabulary_size": len(unique_words),
        "type_token_ratio": len(unique_words) / len(all_words) if all_words else 0
    }
    
    return stats


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration
    """
    merged = {}
    
    for config in configs:
        merged.update(config)
    
    return merged


def filter_models_by_criteria(
    models: list,
    criteria: Dict[str, Any]
) -> list:
    """
    Filter models based on criteria.
    
    Args:
        models: List of model names/paths
        criteria: Filtering criteria
        
    Returns:
        Filtered model list
    """
    filtered = []
    
    for model in models:
        # Simple filtering based on model name patterns
        if "include_patterns" in criteria:
            patterns = criteria["include_patterns"]
            if not any(pattern in model for pattern in patterns):
                continue
        
        if "exclude_patterns" in criteria:
            patterns = criteria["exclude_patterns"]
            if any(pattern in model for pattern in patterns):
                continue
        
        filtered.append(model)
    
    return filtered


def create_progress_tracker(total: int, description: str = "Processing"):
    """
    Create a progress tracker for long-running operations.
    
    Args:
        total: Total number of items
        description: Description for progress bar
        
    Returns:
        Progress tracker object
    """
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=description)
    except ImportError:
        # Fallback simple progress tracker
        class SimpleTracker:
            def __init__(self, total, desc):
                self.total = total
                self.current = 0
                self.desc = desc
            
            def update(self, n=1):
                self.current += n
                print(f"\r{self.desc}: {self.current}/{self.total} "
                      f"({100*self.current/self.total:.1f}%)", end="")
            
            def close(self):
                print()
        
        return SimpleTracker(total, description)


def safe_filename(filename: str) -> str:
    """
    Create a safe filename by removing/replacing problematic characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    import re
    
    # Replace problematic characters
    safe = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove multiple underscores
    safe = re.sub(r'_+', '_', safe)
    
    # Remove leading/trailing underscores and dots
    safe = safe.strip('_.')
    
    # Ensure it's not empty
    if not safe:
        safe = "unnamed"
    
    return safe


def estimate_memory_usage(
    model_name: str,
    precision: str = "float16"
) -> float:
    """
    Estimate memory usage for a model in GB.
    
    Args:
        model_name: Name of the model
        precision: Model precision (float16, float32, etc.)
        
    Returns:
        Estimated memory usage in GB
    """
    # Very rough estimates based on model names
    size_estimates = {
        "distilgpt2": 0.5,
        "gpt2": 1.0,
        "gpt2-medium": 3.0,
        "gpt2-large": 6.0,
        "opt-350m": 2.0,
        "opt-1.3b": 5.0,
        "opt-6.7b": 25.0,
        "gpt-neo-1.3b": 5.0,
        "gpt-j-6b": 24.0
    }
    
    # Check for exact matches or partial matches
    for key, size in size_estimates.items():
        if key in model_name.lower():
            # Adjust for precision
            if precision == "float32":
                return size * 2
            elif precision == "float16":
                return size
            elif "8bit" in precision:
                return size * 0.5
            elif "4bit" in precision:
                return size * 0.25
    
    # Default estimate
    return 2.0
