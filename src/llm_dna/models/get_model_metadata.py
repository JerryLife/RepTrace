#!/usr/bin/env python3
"""
Extract comprehensive metadata for models listed in a file or provided via CLI.

This script fetches detailed metadata for each model including:
- Architecture type (encoder-only, decoder-only, encoder-decoder)
- Chat/instruction tuned status
- Model size (parameter count)
- Default data precision
- Gated repository status
- HuggingFace model URL
- Model family and organization

Notes:
- Output/logs avoid control characters and odd symbols for clean terminals.
- Users can specify models via --models, --models-file, or --list-file.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import argparse
from datetime import datetime
import time

try:
    from huggingface_hub import hf_hub_download, model_info, HfApi
    from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
except ImportError:
    print("Error: huggingface_hub is required. Install with: pip install huggingface_hub")
    sys.exit(1)


class ModelMetadataExtractor:
    """Extract comprehensive metadata for HuggingFace models."""

    def __init__(self, verbose: bool = False, token: Optional[str] = None):
        self.verbose = verbose
        self.token = token
        self.api = HfApi(token=token)

        # Chat model patterns
        self.chat_patterns = [
            'chat', 'instruct', 'instruction', '-it', 'alpaca', 'vicuna',
            'wizardlm', 'dolly', 'assistant', 'helper', 'conversational'
        ]

        # Known model families and their typical architectures
        self.model_families = {
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
            'ul2': {'arch': 'encoder_decoder', 'org': 'Google'}
        }

    def log(self, message: str):
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")

    def detect_model_family(self, model_name: str) -> Optional[Dict[str, str]]:
        """Detect model family from model name."""
        model_lower = model_name.lower()

        # Check for exact matches first
        for family, info in self.model_families.items():
            if family in model_lower:
                return {'family': family, **info}

        # Check for partial matches
        if 'code' in model_lower and ('llama' in model_lower or 'phi' in model_lower):
            return {'family': 'code_model', 'arch': 'decoder_only', 'org': 'Various'}

        return None

    def extract_parameter_count(self, model_name: str, config: Dict = None) -> Optional[Dict[str, Any]]:
        """Extract parameter count from model name or config."""
        # Try to get from config first
        if config:
            # Look for explicit parameter count fields
            param_fields = ['num_parameters', 'n_params', 'total_params']
            for field in param_fields:
                if field in config:
                    count = config[field]
                    return {
                        'parameter_count': count,
                        'parameter_count_billions': round(count / 1e9, 2),
                        'size_category': self.categorize_model_size(count / 1e9),
                        'source': 'config'
                    }

            # Calculate from architecture info
            if all(k in config for k in ['hidden_size', 'num_hidden_layers', 'vocab_size']):
                # Rough estimate: 12 * layers * hidden_size^2 + vocab * hidden_size
                layers = config['num_hidden_layers']
                hidden = config['hidden_size']
                vocab = config['vocab_size']

                # Transformer parameters estimation
                params = (12 * layers * hidden * hidden) + (vocab * hidden)

                return {
                    'parameter_count': params,
                    'parameter_count_billions': round(params / 1e9, 2),
                    'size_category': self.categorize_model_size(params / 1e9),
                    'source': 'calculated'
                }

        # Extract from model name using regex patterns
        patterns = [
            r'(\d+\.?\d*)\s*b(?:illion)?(?:[^a-z]|$)',  # e.g., "7b", "1.5b"
            r'(\d+\.?\d*)b[_-]',  # e.g., "7b-instruct"
            r'[_-](\d+\.?\d*)b',  # e.g., "model-7b"
            r'(\d+\.?\d*)\s*m(?:illion)?(?:[^a-z]|$)',  # e.g., "560m"
            r'(\d+\.?\d*)m[_-]',  # e.g., "560m-base"
            r'[_-](\d+\.?\d*)m'   # e.g., "model-560m"
        ]

        model_lower = model_name.lower()

        for pattern in patterns:
            match = re.search(pattern, model_lower)
            if match:
                try:
                    count = float(match.group(1))
                    # Convert millions to billions if needed
                    if 'm' in pattern:
                        count_billions = count / 1000
                        total_params = int(count * 1e6)
                    else:
                        count_billions = count
                        total_params = int(count * 1e9)

                    return {
                        'parameter_count': total_params,
                        'parameter_count_billions': count_billions,
                        'size_category': self.categorize_model_size(count_billions),
                        'source': 'name_pattern'
                    }
                except ValueError:
                    continue

        return None

    def categorize_model_size(self, size_billions: float) -> str:
        """Categorize model size."""
        if size_billions < 1:
            return "small"
        elif size_billions < 7:
            return "medium"
        elif size_billions < 20:
            return "large"
        elif size_billions < 50:
            return "very_large"
        else:
            return "ultra_large"

    def get_architecture_from_config(self, model_name: str) -> Dict[str, Any]:
        """Get model architecture from config.json."""
        try:
            config_path = hf_hub_download(
                repo_id=model_name,
                filename="config.json",
                token=self.token
            )
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Extract architecture info
            architectures = config.get("architectures", [])
            model_type = config.get("model_type", "")

            # Determine architecture type
            arch_type = "unknown"

            if architectures:
                arch = architectures[0]
                if any(pattern in arch for pattern in ["ForCausalLM", "GPTLMHeadModel"]):
                    arch_type = "decoder_only"
                elif any(pattern in arch for pattern in ["ForConditionalGeneration"]):
                    arch_type = "encoder_decoder"
                elif any(pattern in arch for pattern in ["ForMaskedLM", "Model"]) and "CausalLM" not in arch:
                    arch_type = "encoder_only"

            # Fallback to model_type
            if arch_type == "unknown" and model_type:
                if model_type in ["bert", "roberta", "deberta", "electra", "distilbert"]:
                    arch_type = "encoder_only"
                elif model_type in ["t5", "bart", "pegasus"]:
                    arch_type = "encoder_decoder"
                elif model_type in ["gpt2", "llama", "mistral", "qwen2", "gemma"]:
                    arch_type = "decoder_only"

            # Get parameter count
            param_info = self.extract_parameter_count(model_name, config)

            # Get data type
            torch_dtype = config.get("torch_dtype", "float32")

            return {
                'architecture_type': arch_type,
                'model_type': model_type,
                'architectures': architectures,
                'torch_dtype': torch_dtype,
                'parameter_info': param_info,
                'config_available': True,
                'config': config
            }

        except Exception as e:
            self.log(f"Error fetching config for {model_name}: {e}")
            return {
                'architecture_type': "unknown",
                'config_available': False,
                'error': str(e)
            }

    def check_if_chat_model(self, model_name: str, config: Dict = None) -> Dict[str, Any]:
        """Determine if model is chat/instruction tuned with high confidence."""
        chat_indicators: List[str] = []
        confidence = 'low'

        # Prioritize tokenizer_config.json for chat template
        try:
            tokenizer_config_path = hf_hub_download(
                repo_id=model_name,
                filename="tokenizer_config.json",
                token=self.token
            )
            with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
                tokenizer_config = json.load(f)

            if tokenizer_config.get("chat_template"):
                chat_indicators.append("has_chat_template_in_tokenizer_config")
                confidence = 'high'

        except Exception:
            # File might not exist, which is fine
            pass

        # Check the main config as a secondary source
        if confidence != 'high' and config and config.get("chat_template"):
            chat_indicators.append("has_chat_template_in_config")
            confidence = 'high'

        # Fallback to name patterns if no template was found
        if any(pattern in model_name.lower() for pattern in self.chat_patterns):
            chat_indicators.append("name_pattern_match")
            if confidence == 'low':
                confidence = 'medium'  # Name patterns are less reliable than templates

        return {
            'is_chat_model': len(chat_indicators) > 0,
            'chat_indicators': sorted(set(chat_indicators)),  # Remove duplicates
            'confidence': confidence
        }

    def get_repo_status(self, model_name: str) -> Dict[str, Any]:
        """Check repository status (gated, private, etc.)."""
        try:
            info = model_info(model_name, token=self.token)

            return {
                'is_gated': getattr(info, 'gated', False),
                'is_private': getattr(info, 'private', False),
                'exists': True,
                'model_id': info.id,
                'url': f"https://huggingface.co/{model_name}",
                'downloads': getattr(info, 'downloads', 0),
                'likes': getattr(info, 'likes', 0),
                'tags': getattr(info, 'tags', []),
                'library_name': getattr(info, 'library_name', None),
                'created_at': info.created_at.isoformat() if hasattr(info, 'created_at') and info.created_at else None,
                'last_modified': info.last_modified.isoformat() if hasattr(info, 'last_modified') and info.last_modified else None
            }

        except GatedRepoError:
            return {
                'is_gated': True,
                'is_private': False,
                'exists': True,
                'url': f"https://huggingface.co/{model_name}",
                'access_error': 'gated_repository'
            }
        except RepositoryNotFoundError:
            return {
                'exists': False,
                'error': 'repository_not_found'
            }
        except Exception as e:
            return {
                'exists': 'unknown',
                'error': str(e)
            }

    def extract_metadata(self, model_name: str) -> Dict[str, Any]:
        """Extract comprehensive metadata for a single model."""
        self.log(f"Processing {model_name}...")

        start_time = time.time()

        # Get architecture and config info FIRST
        arch_info = self.get_architecture_from_config(model_name)

        # Determine family and org from config
        model_family = "unknown"
        organization = model_name.split('/')[0] if '/' in model_name else 'unknown'

        config_model_type = arch_info.get('model_type')
        if config_model_type:
            model_family = config_model_type
            if config_model_type in self.model_families:
                organization = self.model_families[config_model_type]['org']

        # Fallback to name-based detection
        if model_family == "unknown":
            family_info = self.detect_model_family(model_name)
            if family_info:
                model_family = family_info['family']
                organization = family_info['org']

        # Check if it's a chat model
        chat_info = self.check_if_chat_model(
            model_name,
            arch_info.get('config') if arch_info.get('config_available') else None
        )

        # Get repository status
        repo_info = self.get_repo_status(model_name)

        # Extract parameter count (fallback to name if config doesn't have it)
        param_info = arch_info.get('parameter_info')
        if not param_info:
            param_info = self.extract_parameter_count(model_name)

        # Build comprehensive metadata
        metadata: Dict[str, Any] = {
            'model_name': model_name,
            'model_family': model_family,
            'organization': organization,
            'architecture': {
                'type': arch_info.get('architecture_type', 'unknown'),
                'model_type': arch_info.get('model_type', 'unknown'),
                'architectures': arch_info.get('architectures', []),
                'is_generative': arch_info.get('architecture_type') in ['decoder_only', 'encoder_decoder']
            },
            'size': param_info if param_info else {
                'parameter_count': None,
                'parameter_count_billions': None,
                'size_category': 'unknown',
                'source': 'not_available'
            },
            'chat_model': chat_info,
            'precision': {
                'default_dtype': arch_info.get('torch_dtype', 'unknown'),
                'supports_quantization': arch_info.get('architecture_type') == 'decoder_only'
            },
            'repository': repo_info,
            'metadata': {
                'extraction_time': time.time() - start_time,
                'config_available': arch_info.get('config_available', False),
                'extraction_timestamp': datetime.now().isoformat()
            }
        }

        return metadata


def load_model_list(list_file: Path) -> List[str]:
    """Load model names from text file."""
    with open(list_file, 'r', encoding='utf-8') as f:
        models = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return models


def derive_default_output_path(list_file: Path) -> Path:
    """Derive metadata output filename based on the list filename."""
    list_dir = list_file.parent
    stem = list_file.stem
    stem_lower = stem.lower()

    candidate_stems = []

    if 'llm_list' in stem_lower:
        candidate_stems.append(re.sub('llm_list', 'llm_metadata', stem, flags=re.IGNORECASE))

    suffix_replacements = [
        ('_list', '_metadata'),
        ('-list', '-metadata'),
        ('list', 'metadata'),
    ]

    for old_suffix, new_suffix in suffix_replacements:
        if stem_lower.endswith(old_suffix):
            candidate_stems.append(f"{stem[:-len(old_suffix)]}{new_suffix}")
            break

    if stem and 'metadata' not in stem_lower:
        candidate_stems.append(f'{stem}_metadata')
    candidate_stems.append('llm_metadata')

    for candidate_stem in candidate_stems:
        if candidate_stem:
            return list_dir / f'{candidate_stem}.json'

    return list_dir / 'llm_metadata.json'


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Extract metadata for models in llm_list.txt')
    parser.add_argument('--list-file', default='./configs/llm_list.txt',
                       help='Path to LLM list file (default: ./configs/llm_list.txt)')
    parser.add_argument('--output-file', help="Output JSON file (default: replace 'list' with 'metadata' in list filename)")
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--models', nargs='+', help='Specific models to process. Accepts space or comma separated.')
    parser.add_argument('--models-file', type=str, help='Path to a file containing one model per line')
    parser.add_argument('--skip-existing', action='store_true', help='Skip models that already have metadata')
    parser.add_argument('--max-models', type=int, help='Maximum number of models to process')
    parser.add_argument('--token', type=str, help='Hugging Face Hub token for accessing private/gated models')

    args = parser.parse_args()

    # Setup paths
    list_file = Path(args.list_file)
    if not list_file.exists():
        print(f"Error: Model list file not found: {list_file}")
        return 1

    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = derive_default_output_path(list_file)

    # Load existing metadata if skip_existing is enabled
    existing_metadata: Dict[str, Any] = {}
    if args.skip_existing and output_file.exists():
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                existing_metadata = {model['model_name']: model for model in existing_data.get('models', [])}
            print(f"Loaded {len(existing_metadata)} existing model entries")
        except Exception as e:
            print(f"Warning: Could not load existing metadata: {e}")

    # Load models
    if args.models:
        # Support both space-separated and comma-separated inputs
        models: List[str] = []
        for item in args.models:
            models.extend([m.strip() for m in item.split(',') if m.strip()])
        print(f"Processing {len(models)} specified models")
    elif args.models_file:
        mf = Path(args.models_file)
        if not mf.exists():
            print(f"Error: --models-file not found: {mf}")
            return 1
        models = load_model_list(mf)
        print(f"Loaded {len(models)} models from {mf}")
    else:
        models = load_model_list(list_file)
        print(f"Loaded {len(models)} models from {list_file}")

    if args.max_models:
        models = models[:args.max_models]
        print(f"Limited to first {args.max_models} models")

    # Initialize extractor with token
    extractor = ModelMetadataExtractor(verbose=args.verbose, token=args.token)

    # Process models
    all_metadata: List[Dict[str, Any]] = []
    successful = 0
    failed = 0
    skipped = 0

    for i, model_name in enumerate(models, 1):
        print(f"\n[{i}/{len(models)}] Processing: {model_name}")

        # Skip if already exists (clean print)
        if args.skip_existing and model_name in existing_metadata:
            print("  - Skipping (already exists)")
            all_metadata.append(existing_metadata[model_name])
            skipped += 1
            continue

        try:
            metadata = extractor.extract_metadata(model_name)
            all_metadata.append(metadata)

            # Quick status
            arch = metadata['architecture']['type']
            chat = "chat" if metadata['chat_model']['is_chat_model'] else "base"
            size = metadata['size']['size_category']
            gated = "gated" if metadata['repository'].get('is_gated') else "open"

            print(f"  - {arch} | {chat} | {size} | {gated}")
            successful += 1

        except KeyboardInterrupt:
            print("\n- Interrupted by user")
            break
        except Exception as e:
            print(f"  L Error: {e}")
            failed += 1

            # Add minimal error entry
            all_metadata.append({
                'model_name': model_name,
                'error': str(e),
                'extraction_timestamp': datetime.now().isoformat()
            })

    # Create final structure
    output_data: Dict[str, Any] = {
        'metadata': {
            'extraction_date': datetime.now().isoformat(),
            'source_file': str(list_file),
            'total_models': len(models),
            'successful_extractions': successful,
            'failed_extractions': failed,
            'skipped_extractions': skipped,
            'extraction_method': 'config_based_with_fallback'
        },
        'statistics': {
            'by_architecture': {},
            'by_size': {},
            'by_organization': {},
            'chat_models': 0,
            'gated_models': 0
        },
        'models': all_metadata
    }

    # Calculate statistics
    for model in all_metadata:
        if 'error' not in model:
            arch = model['architecture']['type']
            size = model['size']['size_category']
            org = model['organization']

            output_data['statistics']['by_architecture'][arch] = output_data['statistics']['by_architecture'].get(arch, 0) + 1
            output_data['statistics']['by_size'][size] = output_data['statistics']['by_size'].get(size, 0) + 1
            output_data['statistics']['by_organization'][org] = output_data['statistics']['by_organization'].get(org, 0) + 1

            if model['chat_model']['is_chat_model']:
                output_data['statistics']['chat_models'] += 1
            if model['repository'].get('is_gated'):
                output_data['statistics']['gated_models'] += 1

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        # Ensure ASCII to avoid replacement characters in limited terminals
        json.dump(output_data, f, indent=2, ensure_ascii=True)

    print("\nExtraction complete!")
    print(f"=> Results: {successful} successful, {failed} failed, {skipped} skipped")
    print(f"=> Saved to: {output_file}")

    # Print summary statistics
    print("\n=> STATISTICS:")
    print(f"  Architecture types: {dict(output_data['statistics']['by_architecture'])}")
    print(f"  Size categories: {dict(output_data['statistics']['by_size'])}")
    print(f"  Chat models: {output_data['statistics']['chat_models']}")
    print(f"  Gated models: {output_data['statistics']['gated_models']}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
