#!/usr/bin/env python3
"""
Utility function to check if a model is a chat/instruction-tuned model.

This module provides a function to determine if a model is a chat model
by checking the model metadata JSON file.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def is_chat_model(model_name: str, metadata_file: Path) -> bool:
    """
    Check if a model is a chat/instruction-tuned model based on metadata.
    
    Args:
        model_name: Model name (e.g., "Qwen/Qwen2.5-0.5B-Instruct")
        metadata_file: Path to the metadata JSON file (e.g., configs/llm_metadata.json)
        
    Returns:
        True if the model is a chat model, False otherwise.
        Returns False if metadata file doesn't exist or model not found in metadata.
    """
    if not metadata_file.exists():
        logger.warning(f"Metadata file not found: {metadata_file}")
        return False
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Find the model in the metadata
        models = data.get('models', [])
        for model_entry in models:
            if model_entry.get('model_name') == model_name:
                chat_info = model_entry.get('chat_model', {})
                is_chat = chat_info.get('is_chat_model', False)
                return bool(is_chat)
        
        # Model not found in metadata
        logger.debug(f"Model '{model_name}' not found in metadata file")
        return False
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse metadata file {metadata_file}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking chat model status for {model_name}: {e}")
        return False


def main():
    """CLI interface for checking if a model is a chat model."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check if a model is a chat/instruction-tuned model"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name to check (e.g., 'Qwen/Qwen2.5-0.5B-Instruct')"
    )
    parser.add_argument(
        "--metadata-file",
        type=Path,
        default=Path("configs/llm_metadata.json"),
        help="Path to metadata JSON file (default: configs/llm_metadata.json)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    result = is_chat_model(args.model_name, args.metadata_file)
    print("true" if result else "false")
    return 0 if result else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

