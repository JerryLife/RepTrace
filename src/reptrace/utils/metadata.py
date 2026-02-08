"""Model metadata utilities for fetching and caching model information."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Chat model name patterns
CHAT_PATTERNS = [
    'chat', 'instruct', 'instruction', '-it', 'alpaca', 'vicuna',
    'wizardlm', 'dolly', 'assistant', 'helper', 'conversational'
]

# Known model families and their typical architectures
MODEL_FAMILIES = {
    'bert': {'arch': 'encoder_only', 'org': 'Google'},
    'roberta': {'arch': 'encoder_only', 'org': 'Facebook'},
    'deberta': {'arch': 'encoder_only', 'org': 'Microsoft'},
    'electra': {'arch': 'encoder_only', 'org': 'Google'},
    'distilbert': {'arch': 'encoder_only', 'org': 'Hugging Face'},
    'gpt': {'arch': 'decoder_only', 'org': 'OpenAI'},
    'gpt2': {'arch': 'decoder_only', 'org': 'OpenAI'},
    'llama': {'arch': 'decoder_only', 'org': 'Meta'},
    'mistral': {'arch': 'decoder_only', 'org': 'Mistral AI'},
    'qwen': {'arch': 'decoder_only', 'org': 'Alibaba'},
    'gemma': {'arch': 'decoder_only', 'org': 'Google'},
    'phi': {'arch': 'decoder_only', 'org': 'Microsoft'},
    'falcon': {'arch': 'decoder_only', 'org': 'Technology Innovation Institute'},
    'deepseek': {'arch': 'decoder_only', 'org': 'DeepSeek'},
    'yi': {'arch': 'decoder_only', 'org': '01.AI'},
    'bloom': {'arch': 'decoder_only', 'org': 'BigScience'},
    'pythia': {'arch': 'decoder_only', 'org': 'EleutherAI'},
    't5': {'arch': 'encoder_decoder', 'org': 'Google'},
    'flan': {'arch': 'encoder_decoder', 'org': 'Google'},
    'bart': {'arch': 'encoder_decoder', 'org': 'Facebook'},
    'ul2': {'arch': 'encoder_decoder', 'org': 'Google'},
}


def _get_metadata_cache_path() -> Path:
    """Get path to metadata cache file."""
    from .DataUtils import get_cache_dir
    return get_cache_dir() / "model_metadata_cache.json"


def _load_metadata_cache() -> Dict[str, Any]:
    """Load cached metadata from disk."""
    cache_path = _get_metadata_cache_path()
    if cache_path.exists():
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata cache: {e}")
    return {}


def _save_metadata_cache(cache: Dict[str, Any]) -> None:
    """Save metadata cache to disk."""
    cache_path = _get_metadata_cache_path()
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"Failed to save metadata cache: {e}")


def _detect_model_family(model_name: str) -> Optional[Dict[str, str]]:
    """Detect model family from model name."""
    model_lower = model_name.lower()
    for family, info in MODEL_FAMILIES.items():
        if family in model_lower:
            return {'family': family, **info}
    return None


def _extract_parameter_count(model_name: str, config: Optional[Dict] = None) -> Optional[Dict[str, Any]]:
    """Extract parameter count from model name or config."""
    # Try to get from config first
    if config:
        param_fields = ['num_parameters', 'n_params', 'total_params']
        for field in param_fields:
            if field in config:
                count = config[field]
                return {
                    'parameter_count': count,
                    'parameter_count_billions': round(count / 1e9, 2),
                }

        # Calculate from architecture info
        if all(k in config for k in ['hidden_size', 'num_hidden_layers', 'vocab_size']):
            layers = config['num_hidden_layers']
            hidden = config['hidden_size']
            vocab = config['vocab_size']
            params = (12 * layers * hidden * hidden) + (vocab * hidden)
            return {
                'parameter_count': params,
                'parameter_count_billions': round(params / 1e9, 2),
            }

    # Extract from model name using regex
    patterns = [
        r'(\d+\.?\d*)\s*b(?:illion)?(?:[^a-z]|$)',
        r'(\d+\.?\d*)b[_-]',
        r'[_-](\d+\.?\d*)b',
        r'(\d+\.?\d*)\s*m(?:illion)?(?:[^a-z]|$)',
        r'(\d+\.?\d*)m[_-]',
        r'[_-](\d+\.?\d*)m'
    ]

    model_lower = model_name.lower()
    for pattern in patterns:
        match = re.search(pattern, model_lower)
        if match:
            try:
                count = float(match.group(1))
                if 'm' in pattern:
                    return {
                        'parameter_count': int(count * 1e6),
                        'parameter_count_billions': count / 1000,
                    }
                else:
                    return {
                        'parameter_count': int(count * 1e9),
                        'parameter_count_billions': count,
                    }
            except ValueError:
                continue

    return None


def fetch_model_metadata(model_name: str, token: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch model metadata from HuggingFace Hub.
    
    Args:
        model_name: Model name or HuggingFace model ID
        token: Optional HuggingFace token for gated models
        
    Returns:
        Dictionary containing model metadata
    """
    try:
        from huggingface_hub import hf_hub_download, model_info
        from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
    except ImportError:
        logger.warning("huggingface_hub not available, using default metadata")
        return _default_metadata(model_name)

    logger.info(f"Fetching metadata for {model_name} from HuggingFace Hub...")

    config = None
    arch_type = "decoder_only"  # Default assumption
    model_type = "unknown"
    torch_dtype = "float32"
    is_chat_model = False

    # Try to fetch config.json
    try:
        config_path = hf_hub_download(
            repo_id=model_name,
            filename="config.json",
            token=token
        )
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Extract architecture info
        architectures = config.get("architectures", [])
        model_type = config.get("model_type", "unknown")
        torch_dtype = config.get("torch_dtype", "float32")

        if architectures:
            arch = architectures[0]
            if any(p in arch for p in ["ForCausalLM", "GPTLMHeadModel"]):
                arch_type = "decoder_only"
            elif "ForConditionalGeneration" in arch:
                arch_type = "encoder_decoder"
            elif any(p in arch for p in ["ForMaskedLM"]) and "CausalLM" not in arch:
                arch_type = "encoder_only"

    except Exception as e:
        logger.debug(f"Could not fetch config for {model_name}: {e}")
        # Use family detection as fallback
        family_info = _detect_model_family(model_name)
        if family_info:
            arch_type = family_info.get('arch', 'decoder_only')

    # Check for chat template
    try:
        tokenizer_config_path = hf_hub_download(
            repo_id=model_name,
            filename="tokenizer_config.json",
            token=token
        )
        with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
            tokenizer_config = json.load(f)
        if tokenizer_config.get("chat_template"):
            is_chat_model = True
    except Exception:
        pass

    # Fallback to name pattern for chat detection
    if not is_chat_model:
        is_chat_model = any(p in model_name.lower() for p in CHAT_PATTERNS)

    # Get parameter count
    param_info = _extract_parameter_count(model_name, config)

    # Get repository info
    repo_info = {}
    try:
        info = model_info(model_name, token=token)
        repo_info = {
            'is_gated': getattr(info, 'gated', False),
            'model_id': info.id,
        }
    except (RepositoryNotFoundError, GatedRepoError):
        repo_info = {'is_gated': True}
    except Exception:
        pass

    return {
        'model_name': model_name,
        'architecture': {
            'type': arch_type,
            'model_type': model_type,
            'is_generative': arch_type in ['decoder_only', 'encoder_decoder'],
        },
        'size': param_info or {'parameter_count_billions': None},
        'chat_model': {'is_chat_model': is_chat_model},
        'precision': {'default_dtype': torch_dtype},
        'repository': repo_info,
        'metadata': {
            'fetched_at': datetime.now().isoformat(),
            'source': 'huggingface_hub',
        }
    }


def _default_metadata(model_name: str) -> Dict[str, Any]:
    """Return default metadata when Hub fetch is not possible."""
    family_info = _detect_model_family(model_name)
    arch_type = family_info.get('arch', 'decoder_only') if family_info else 'decoder_only'
    is_chat = any(p in model_name.lower() for p in CHAT_PATTERNS)
    param_info = _extract_parameter_count(model_name)

    return {
        'model_name': model_name,
        'architecture': {
            'type': arch_type,
            'is_generative': arch_type in ['decoder_only', 'encoder_decoder'],
        },
        'size': param_info or {'parameter_count_billions': None},
        'chat_model': {'is_chat_model': is_chat},
        'repository': {},
        'metadata': {
            'source': 'name_pattern_fallback',
        }
    }


def get_model_metadata(
    model_name: str,
    token: Optional[str] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """
    Get model metadata, fetching from HuggingFace Hub if not cached.
    
    Args:
        model_name: Model name or HuggingFace model ID
        token: Optional HuggingFace token for gated models
        use_cache: Whether to use/update cache
        
    Returns:
        Dictionary containing model metadata
    """
    # Check cache first
    if use_cache:
        cache = _load_metadata_cache()
        if model_name in cache:
            logger.debug(f"Using cached metadata for {model_name}")
            return cache[model_name]

    # Fetch from Hub
    metadata = fetch_model_metadata(model_name, token)

    # Update cache
    if use_cache:
        cache = _load_metadata_cache()
        cache[model_name] = metadata
        _save_metadata_cache(cache)

    return metadata
