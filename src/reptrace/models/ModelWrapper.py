"""
Unified interface for different LLM types.
"""

import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
import logging
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForCausalLM,
    GenerationConfig, BitsAndBytesConfig
)
from tqdm import tqdm
import time


class LLMWrapper(ABC):
    """Abstract base class for LLM wrappers providing unified interface."""
    
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.logger = logging.getLogger(__name__)
        self._generation_stats_logged = False  # Track if we've logged generation stats
        
    def _setup_device(self, device: str) -> str:
        """Setup computation device with explicit GPU selection."""
        if device == "auto":
            if torch.cuda.is_available():
                # Use GPU 0 by default only when truly auto
                return "cuda:0"
            else:
                return "cpu"
        # If explicit device specified (like cuda:1, cuda:2), use it directly
        if device.startswith("cuda:"):
            gpu_id = device.split(":")[1]
            if torch.cuda.is_available() and int(gpu_id) < torch.cuda.device_count():
                return device
            else:
                raise ValueError(f"Requested GPU {gpu_id} not available. Available GPUs: {torch.cuda.device_count()}")
        return device
    
    def _get_hf_cache_dir(self) -> Optional[str]:
        """Get proper HuggingFace cache directory."""
        # Check for HF_HOME environment variable (new standard)
        if 'HF_HOME' in os.environ:
            hf_home = os.environ['HF_HOME']
            # Use the hub subdirectory which is the standard cache location
            cache_dir = os.path.join(hf_home, 'hub')
            return cache_dir
        
        # Check for older TRANSFORMERS_CACHE variable
        if 'TRANSFORMERS_CACHE' in os.environ:
            return os.environ['TRANSFORMERS_CACHE']
            
        # Use default HuggingFace cache location
        default_cache = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')
        return default_cache
    
    def _needs_trust_remote_code(self, model_name: str) -> bool:
        """Detect if a model needs trust_remote_code=True."""
        # Known models that require custom code
        custom_models = [
            "openai/gpt-oss",           # GPT-OSS models
            "openai/gpt_oss",           # Alternative naming
            "microsoft/phi",            # Some Phi models
            "bigcode/",                 # BigCode models
            "codellama/",               # Some CodeLlama variants
            "WizardLM/",                # WizardLM models
            "teknium/",                 # Various custom models
            "NousResearch/",            # Nous Research models
            "garage-bAInd/",            # Platypus models
        ]
        
        model_name_lower = model_name.lower()
        
        # Check for exact matches or prefixes
        for pattern in custom_models:
            if pattern.lower() in model_name_lower:
                self.logger.info(f"Model {model_name} detected as requiring trust_remote_code=True")
                return True
                
        # Check for specific model name patterns that commonly need custom code
        if any(keyword in model_name_lower for keyword in [
            "instruct", "chat", "wizard", "vicuna", "alpaca", "orca", 
            "falcon", "mpt", "starcoder", "santacoder", "gpt-oss"
        ]):
            # These often have custom architectures, but let's be conservative
            # and only enable for known problematic ones
            if "gpt-oss" in model_name_lower:
                self.logger.info(f"Model {model_name} detected as GPT-OSS requiring trust_remote_code=True")
                return True
        
        return False
        
    @abstractmethod
    def generate(
        self,
        input_text: str,
        max_length: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """Generate text from input."""
        pass

    def generate_batch(
        self,
        prompts: List[str],
        max_length: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        **kwargs
    ) -> List[str]:
        """Batch generation default implementation: sequential calls.

        Subclasses may override to provide true batched generation.
        """
        outputs: List[str] = []
        for p in prompts:
            try:
                outputs.append(
                    self.generate(
                        p,
                        max_length=max_length,
                        temperature=temperature,
                        do_sample=do_sample,
                        top_p=top_p,
                        **kwargs,
                    )
                )
            except Exception as e:
                self.logger.error(f"Batch item failed: {e}")
                outputs.append("")
        return outputs
        
    @abstractmethod
    def get_logits(self, input_text: str) -> torch.Tensor:
        """Get output logits for input text."""
        pass
        
    @abstractmethod
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text to token IDs."""
        pass
        
    @abstractmethod
    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        pass
        
    @abstractmethod
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        pass
        
    @abstractmethod
    def get_model_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        pass
    
    def get_token_embeddings(self, token_ids: List[int]) -> torch.Tensor:
        """Get token embeddings for given token IDs."""
        # Default implementation - subclasses can override
        raise NotImplementedError("Token embeddings not implemented for this model type")

    def release(self) -> None:
        """Release underlying resources (override in subclasses if needed)."""
        try:
            import torch as _torch
            _torch.cuda.empty_cache()
        except Exception:
            pass


class HuggingFaceWrapper(LLMWrapper):
    """Base wrapper for HuggingFace transformers models."""
    
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        trust_remote_code: bool = None,
        torch_dtype: Optional[torch.dtype] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        token: Optional[str] = None,
        is_chat_model: Optional[bool] = None
    ):
        super().__init__(model_name, device)
        
        # Auto-detect if trust_remote_code is needed
        if trust_remote_code is None:
            trust_remote_code = self._needs_trust_remote_code(model_name)
        
        self.trust_remote_code = trust_remote_code
        # Do not force a default dtype here; let HF/config decide when None
        self.torch_dtype = torch_dtype
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.hf_token = token
        self.is_chat_model = bool(is_chat_model) if is_chat_model is not None else None
        
        self._load_model_and_tokenizer()
        
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer from HuggingFace."""
        self.logger.info(f"Loading model {self.model_name}")
        
        if self.trust_remote_code:
            self.logger.warning(f"Loading model {self.model_name} with trust_remote_code=True. "
                              "This will execute custom code from the model repository.")
        
        # Load tokenizer with proper cache directory
        cache_dir = self._get_hf_cache_dir()
        hf_common_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "cache_dir": cache_dir,
        }
        if self.hf_token:
            # Newer transformers/huggingface_hub support 'token'; older support 'use_auth_token'
            hf_common_kwargs["token"] = self.hf_token
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            **hf_common_kwargs
        )
        
        # Set pad token if not present - crucial for generation with base models
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.logger.info("Tokenizer has no pad_token; setting it to eos_token.")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            elif self.tokenizer.unk_token is not None:
                self.logger.warning("Tokenizer has no pad_token or eos_token; setting pad_token to unk_token.")
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                # As a last resort, add a new pad token
                self.logger.warning("Tokenizer has no pad, eos, or unk token. Adding a new pad token.")
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                # Mark that we need to resize model embeddings after loading
                self._pad_token_added = True
            
        # Model loading arguments with proper dtype handling
        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "cache_dir": cache_dir
        }
        # Only pass dtype if explicitly provided (use new 'dtype' kw)
        if self.torch_dtype is not None:
            model_kwargs["dtype"] = self.torch_dtype
        if self.hf_token:
            model_kwargs["token"] = self.hf_token
        
        # Handle BFloat16/Half conflicts by forcing consistent dtype
        if self.torch_dtype is None:
            # Auto-detect best dtype based on hardware when not specified
            try:
                if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                    model_kwargs["dtype"] = torch.bfloat16
                    self.torch_dtype = torch.bfloat16
                    self.logger.info(f"Auto-selected torch.bfloat16 for {self.model_name} (Ampere+ GPU)")
                else:
                    model_kwargs["dtype"] = torch.float16
                    self.torch_dtype = torch.float16
                    self.logger.info(f"Auto-selected torch.float16 for {self.model_name}")
            except Exception as e:
                self.logger.warning(f"Could not auto-select dtype: {e}")
        
        # Add quantization config if specified, but check for pre-quantized models first
        quantization_config = None
        if self.load_in_8bit or self.load_in_4bit:
            # Check if model name indicates pre-quantization (FP8, AWQ, GPTQ, etc.)
            model_name_lower = self.model_name.lower()
            is_pre_quantized = any(indicator in model_name_lower for indicator in [
                '-fp8', '-fp4', '-awq', '-gptq', '-gguf', '-ggml', 
                'quantized', 'quant', 'compressed'
            ])
            
            if is_pre_quantized:
                self.logger.warning(f"Model {self.model_name} appears to be pre-quantized. Skipping additional quantization.")
                self.load_in_8bit = False
                self.load_in_4bit = False
            else:
                # Check if model is already quantized by attempting to load config
                try:
                    from transformers import AutoConfig
                    config_kwargs = {
                        "trust_remote_code": self.trust_remote_code,
                        "cache_dir": cache_dir,
                    }
                    if self.hf_token:
                        config_kwargs["token"] = self.hf_token
                    model_config = AutoConfig.from_pretrained(
                        self.model_name,
                        **config_kwargs
                    )
                    
                    # Check if model config indicates pre-quantization
                    if (hasattr(model_config, 'quantization_config') and model_config.quantization_config is not None) or \
                       (hasattr(model_config, 'quantization_method') and model_config.quantization_method is not None):
                        self.logger.warning(f"Model {self.model_name} is already pre-quantized. Skipping additional quantization.")
                        self.load_in_8bit = False
                        self.load_in_4bit = False
                    else:
                        # Apply quantization if model is not pre-quantized
                        if self.load_in_8bit:
                            quantization_config = BitsAndBytesConfig(
                                load_in_8bit=True,
                                llm_int8_threshold=6.0,
                                llm_int8_has_fp16_weight=False
                            )
                        elif self.load_in_4bit:
                            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4"
                            )
                except Exception as e:
                    self.logger.warning(f"Could not check quantization config: {e}. Proceeding with requested quantization.")
                    # Fallback to original logic
                    if self.load_in_8bit:
                        quantization_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            llm_int8_threshold=6.0,
                            llm_int8_has_fp16_weight=False
                        )
                    elif self.load_in_4bit:
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4"
                        )
            
        # Check if model is pre-quantized (for special handling)
        model_name_lower = self.model_name.lower()
        is_pre_quantized = any(indicator in model_name_lower for indicator in [
            '-fp8', '-fp4', '-awq', '-gptq', '-gguf', '-ggml', 
            'quantized', 'quant', 'compressed'
        ])
        
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            # For quantized models, use "auto" device map but limit to specified GPU
            if torch.cuda.is_available():
                # Extract GPU index from device string (e.g., "cuda:1" -> 1)
                gpu_index = 0 if self.device == "cuda" else int(self.device.split(":")[1]) if ":" in self.device else 0
                model_kwargs["device_map"] = "auto"
                
                # Smart memory allocation based on available VRAM
                total_memory = torch.cuda.get_device_properties(gpu_index).total_memory
                total_gb = total_memory / (1024**3)
                # Reserve memory for activations and gradients (use 85% of VRAM, minimum 4GB)
                usable_memory = max(int(total_gb * 0.85), 4)
                model_kwargs["max_memory"] = {gpu_index: f"{usable_memory}GiB"}
                self.logger.info(f"Quantized model will use GPU {gpu_index} with {usable_memory}GiB limit (total: {total_gb:.1f}GB)")
        elif is_pre_quantized:
            # For pre-quantized models, use auto device map and avoid low_cpu_mem_usage
            # which can cause meta device issues
            if torch.cuda.is_available():
                gpu_index = 0 if self.device == "cuda" else int(self.device.split(":")[1]) if ":" in self.device else 0
                model_kwargs["device_map"] = "auto"
                # Don't set low_cpu_mem_usage for pre-quantized models to avoid meta device issues
                if "low_cpu_mem_usage" in model_kwargs:
                    del model_kwargs["low_cpu_mem_usage"]
                self.logger.info(f"Pre-quantized model will use auto device map on GPU {gpu_index}")
        else:
            # For non-quantized models, use explicit device placement
            model_kwargs["device_map"] = {"" : self.device}
            self.logger.info(f"Non-quantized model will use device map: {model_kwargs['device_map']}")
            
            # Enable memory optimizations for large models
            if self._is_likely_large_model():
                model_kwargs["low_cpu_mem_usage"] = True
                self.logger.info("Enabled low_cpu_mem_usage for large model")
            
        # Try to load with appropriate model class based on model type
        try:
            # First, try to determine the model type from config
            from transformers import AutoConfig
            config_kwargs = {
                "trust_remote_code": self.trust_remote_code,
                "cache_dir": cache_dir,
            }
            if self.hf_token:
                config_kwargs["token"] = self.hf_token
            config = AutoConfig.from_pretrained(
                self.model_name,
                **config_kwargs
            )
            
            model_type = getattr(config, 'model_type', '').lower()
            
            # Use appropriate model class based on type
            if model_type in ['t5', 'flan-t5', 'ul2']:
                from transformers import AutoModelForSeq2SeqLM
                self.logger.info(f"Loading T5/Seq2Seq model: {self.model_name}")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            elif model_type in ['bert', 'roberta', 'deberta', 'electra', 'distilbert']:
                self.logger.info(f"Loading encoder-only model: {self.model_name}")
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
            else:
                # Default to causal LM for decoder-only models
                self.logger.info(f"Loading causal LM model: {self.model_name}")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **model_kwargs
                )
        except Exception as e:
            error_msg = str(e)
            self.logger.warning(f"Failed to load with detected model type: {e}")
            
            # Check if it's a meta device error (common with pre-quantized models)
            if "meta device" in error_msg.lower() or "weight_scale" in error_msg.lower():
                self.logger.warning("Detected meta device error. Trying alternative loading method for pre-quantized model.")
                # Remove problematic kwargs for pre-quantized models
                fallback_kwargs = model_kwargs.copy()
                fallback_kwargs.pop("low_cpu_mem_usage", None)
                fallback_kwargs.pop("device_map", None)
                # Use explicit device placement instead
                fallback_kwargs["device_map"] = "auto"
                
                try:
                    self.logger.info("Trying fallback with auto device_map and no low_cpu_mem_usage")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        **fallback_kwargs
                    )
                except Exception as e3:
                    self.logger.warning(f"Fallback also failed: {e3}")
                    raise
            else:
                # Standard fallback for other errors
                self.logger.info("Trying fallback to CausalLM then AutoModel")
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        **model_kwargs
                    )
                except Exception as e2:
                    self.logger.warning(f"Failed to load as CausalLM: {e2}")
                    self.logger.info("Trying to load as base AutoModel")
                    self.model = AutoModel.from_pretrained(
                        self.model_name,
                        **model_kwargs
                    )
            
        # Ensure all model components are on the same device
        if quantization_config is None:
            self.logger.info(f"Moving non-quantized model to device: {self.device}")
            self.model = self.model.to(self.device)
            # Ensure all parameters are on the same device
            for param in self.model.parameters():
                if param.device != torch.device(self.device):
                    param.data = param.data.to(self.device)
            
            # Verify final device placement
            devices = set()
            for param in self.model.parameters():
                devices.add(str(param.device))
            self.logger.info(f"Non-quantized model final device placement: {devices}")
        else:
            # For quantized models, log the actual device placement
            devices = set()
            for param in self.model.parameters():
                devices.add(str(param.device))
            self.logger.info(f"Quantized model distributed across devices: {devices}")
            
        # Resize model embeddings if we added a new pad token
        if hasattr(self, '_pad_token_added'):
            self.logger.info("Resizing model embeddings for new pad token")
            self.model.resize_token_embeddings(len(self.tokenizer))
            
        self.model.eval()
        self.logger.info(f"Model loaded successfully on {self.device}")
        
    def release(self) -> None:
        try:
            # Drop references and free CUDA cache
            self.model = None
            import torch as _torch
            _torch.cuda.empty_cache()
        except Exception:
            pass
        
    def _prepare_generation_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Prepare inputs for generation by removing unsupported keys.
        Base implementation - subclasses can override for specific model types.
        """
        # Default behavior: pass through all inputs
        return inputs
        
    def _get_safe_generation_params(self, requested_max_length: int) -> Tuple[int, int]:
        """Calculate safe input and generation lengths based on model constraints.
        
        Returns:
            tuple: (safe_input_length, max_new_tokens)
        """
        model_max_length = getattr(self.model.config, 'max_position_embeddings', 1024)
        
        # Conservative approach: use at most half context for generation
        max_new_tokens = min(requested_max_length, model_max_length // 2)
        safe_input_length = model_max_length - max_new_tokens
        
        # Ensure minimum space for input (at least 64 tokens)
        min_input_length = 64
        if safe_input_length < min_input_length:
            safe_input_length = min_input_length
            max_new_tokens = max(1, model_max_length - safe_input_length)  # At least 1 token generation
            
        # Only log this once per model to avoid spam
        if not self._generation_stats_logged:
            self.logger.info(f"Model: {self.model_name}, Max context: {model_max_length}, "
                           f"Input limit: {safe_input_length}, Generation limit: {max_new_tokens}")
            self._generation_stats_logged = True
        
        return safe_input_length, max_new_tokens
    
    def _is_likely_large_model(self) -> bool:
        """Determine if model is likely large based on name patterns."""
        model_name_lower = self.model_name.lower()
        large_patterns = [
            "20b", "24b", "30b", "32b", "34b", "70b", "72b", 
            "13b", "14b", "mixtral", "gpt-neox-20b", "flan-ul2",
            "mistral-small", "qwq", "glm-4-32b", "yi-34b"
        ]
        return any(pattern in model_name_lower for pattern in large_patterns)
    
    def _apply_chat_template_if_needed(self, input_text: str) -> str:
        """Apply chat template if model is a chat/instruct variant."""
        try:
            # Check if this is a chat/instruct model
            model_name_lower = self.model_name.lower()
            chat_indicators = ['chat', 'instruct', '-it', 'alpaca', 'vicuna', 'wizardlm']
            
            is_chat_model = any(indicator in model_name_lower for indicator in chat_indicators)
            
            if is_chat_model and hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
                self.logger.debug(f"Applying chat template for chat model: {self.model_name}")
                
                # Create chat message format
                messages = [{"role": "user", "content": input_text}]
                
                # Apply chat template with error handling
                try:
                    formatted_text = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    return formatted_text
                except Exception as e:
                    self.logger.warning(f"Failed to apply chat template for {self.model_name}: {e}. Using raw input.")
                    return input_text
            else:
                # Not a chat model or no template available
                if is_chat_model:
                    self.logger.debug(f"Chat model {self.model_name} has no chat_template. Using raw input.")
                return input_text
                
        except Exception as e:
            self.logger.warning(f"Error in chat template application: {e}. Using raw input.")
            return input_text
        
    def generate(
        self,
        input_text: str,
        max_length: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        skip_chat_template: bool = False,
        **kwargs
    ) -> str:
        """Generate text from input, respecting the model's context length."""
        try:
            # Ensure input text is properly encoded
            if isinstance(input_text, str):
                input_text = input_text.encode('utf-8', errors='ignore').decode('utf-8')
            
            # Do not pre-apply chat templates to the string; we will
            # tokenize with the chat template below when available.
            
            # Calculate safe parameters to never exceed model context limits
            safe_input_length, max_new_tokens = self._get_safe_generation_params(max_length)
            
            # Prefer chat-template tokenization when available to ensure special tokens are handled
            # Skip chat template if skip_chat_template=True (treat chat models as completion models)
            inputs = None
            prefers_chat_template = False
            if not skip_chat_template:
                if self.is_chat_model is True:
                    prefers_chat_template = True
                # Heuristic fallback if metadata wasn't provided
                model_name_lower = self.model_name.lower()
                if any(ind in model_name_lower for ind in ['chat', 'instruct', '-it']) or hasattr(self.tokenizer, 'chat_template'):
                    prefers_chat_template = True

            if prefers_chat_template and hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
                try:
                    messages = [{"role": "user", "content": input_text}]
                    inputs = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    )
                except Exception as e:
                    self.logger.debug(f"Chat-template tokenization fallback to plain: {e}")
                    inputs = None
            if inputs is None:
                # Tokenize input with safe truncation
                inputs = self.tokenizer(
                    input_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=safe_input_length
                )

            # Normalize inputs to a dict accepted by generate()
            try:
                import torch as _torch
                if isinstance(inputs, _torch.Tensor):
                    inputs = {"input_ids": inputs}
            except Exception:
                # If torch import fails in this context, leave as-is
                pass
            
            # Ensure all input tensors are on the correct device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            input_length = inputs["input_ids"].shape[1]
            
            # Set up generation config using max_new_tokens (safe calculation)
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,  # Dynamically calculated to be safe
                "temperature": temperature,
                "do_sample": do_sample,
                "top_p": top_p,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # Filter out incompatible kwargs for specific models
            filtered_kwargs = {}
            for k, v in kwargs.items():
                if k not in ['use_cache']:  # Remove problematic keys
                    filtered_kwargs[k] = v
                    
            generation_kwargs.update(filtered_kwargs)
            
            # Create generation config with error handling for compatibility
            try:
                generation_config = GenerationConfig(**generation_kwargs)
            except Exception as e:
                self.logger.warning(f"Failed to create GenerationConfig: {e}. Using direct kwargs.")
                generation_config = None
            
            # Remove token_type_ids for models that don't support it
            clean_inputs = self._prepare_generation_inputs(inputs)
            
            # Generate with improved error handling
            with torch.no_grad():
                try:
                    if generation_config is not None:
                        outputs = self.model.generate(
                            **clean_inputs,
                            generation_config=generation_config
                        )
                    else:
                        # Fallback to direct kwargs if GenerationConfig failed
                        outputs = self.model.generate(
                            **clean_inputs,
                            **generation_kwargs
                        )
                except Exception as e:
                    # Handle specific generation errors
                    error_msg = str(e).lower()
                    if 'seen_tokens' in error_msg:
                        # DynamicCache compatibility issue - retry with simpler config
                        self.logger.warning(f"DynamicCache error detected. Retrying with basic config.")
                        simple_kwargs = {
                            "max_new_tokens": max_new_tokens,
                            "temperature": temperature,
                            "do_sample": do_sample,
                            "pad_token_id": self.tokenizer.pad_token_id,
                            "eos_token_id": self.tokenizer.eos_token_id,
                        }
                        outputs = self.model.generate(**clean_inputs, **simple_kwargs)
                    else:
                        raise
                
            # Decode only the newly generated tokens
            new_tokens = outputs[0][input_length:]
            # When skipping chat template, preserve special tokens (match try_chat_model_without_template.py behavior)
            skip_special_tokens = not skip_chat_template
            generated_text = self.tokenizer.decode(new_tokens, skip_special_tokens=skip_special_tokens)
            
            return generated_text.strip()
            
        except Exception as e:
            self.logger.error(f"Text generation failed for input: {input_text[:50]}... Error: {e}")
            # Return empty string as fallback
            return ""
        
    def get_logits(self, input_text: str) -> torch.Tensor:
        """Get output logits for input text."""
        # Use safe input length (reserve no space for generation since we're not generating)
        model_max_length = getattr(self.model.config, 'max_position_embeddings', 1024)
        safe_length = min(model_max_length, 1024)  # Conservative limit for logits
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=safe_length
        )
        
        # Ensure all input tensors are on the correct device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs.last_hidden_state
            
        return logits
        
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text to token IDs."""
        try:
            # Ensure text is properly encoded
            if isinstance(text, str):
                # Handle unicode characters properly
                text = text.encode('utf-8', errors='ignore').decode('utf-8')
            
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            
            # Filter out padding tokens that might be invalid
            if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                token_ids = [tid for tid in token_ids if tid != self.tokenizer.pad_token_id]
            
            # Additional validation - ensure all token IDs are valid integers
            vocab_size = self.get_vocab_size()
            valid_token_ids = []
            for tid in token_ids:
                if isinstance(tid, (int, np.integer)) and 0 <= tid < vocab_size:
                    valid_token_ids.append(int(tid))
                else:
                    # Log problematic token ID
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Invalid token ID {tid} (type: {type(tid)}) for text: {text[:50]}...")
            
            return valid_token_ids
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Tokenization failed for text: {text[:50]}... Error: {e}")
            # Return empty list as fallback
            return []
        
    def detokenize(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
        
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)
        
    def get_model_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        config = self.model.config
        
        # Get hidden size with multiple fallback attribute names
        hidden_size = None
        for attr_name in ['hidden_size', 'dim', 'width', 'd_model', 'embed_dim', 'embedding_size']:
            if hasattr(config, attr_name):
                hidden_size = getattr(config, attr_name)
                break
        
        # Get number of layers with multiple fallback attribute names
        num_layers = None
        for attr_name in ['num_hidden_layers', 'num_layers', 'n_layer', 'n_layers', 'depth']:
            if hasattr(config, attr_name):
                num_layers = getattr(config, attr_name)
                break
        
        # Get number of attention heads with multiple fallback attribute names
        num_heads = None
        for attr_name in ['num_attention_heads', 'num_heads', 'n_head', 'n_heads']:
            if hasattr(config, attr_name):
                num_heads = getattr(config, attr_name)
                break
        
        metadata = {
            "model_name": self.model_name,
            "model_type": config.model_type if hasattr(config, 'model_type') else "unknown",
            "vocab_size": self.get_vocab_size(),
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_attention_heads": num_heads,
            "max_position_embeddings": getattr(config, 'max_position_embeddings', None),
            "torch_dtype": str(self.torch_dtype),
            "device": self.device,
            "load_in_8bit": self.load_in_8bit,
            "load_in_4bit": self.load_in_4bit
        }
        
        return metadata
    
    def get_token_embeddings(self, token_ids: List[int]) -> torch.Tensor:
        """Get token embeddings for given token IDs."""
        # Get hidden size with fallback
        hidden_size = None
        for attr_name in ['hidden_size', 'dim', 'width', 'd_model', 'embed_dim', 'embedding_size']:
            if hasattr(self.model.config, attr_name):
                hidden_size = getattr(self.model.config, attr_name)
                break
        
        if hidden_size is None:
            hidden_size = 768  # Reasonable default
            
        if len(token_ids) == 0:
            return torch.empty(0, hidden_size)
        
        # Validate token IDs before converting to tensor
        try:
            vocab_size = self.get_vocab_size()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to get vocab size: {e}")
            # Use a safe fallback
            vocab_size = 50000  # Common vocab size fallback
        
        valid_token_ids = []
        invalid_count = 0
        
        for token_id in token_ids:
            # More robust validation - check for various invalid token scenarios
            if (token_id is None or 
                not isinstance(token_id, (int, np.integer)) or 
                token_id < 0 or 
                token_id >= vocab_size):
                invalid_count += 1
                continue
            valid_token_ids.append(int(token_id))  # Ensure it's a Python int
        
        # Log warning if many invalid tokens found
        if invalid_count > 0:
            import logging
            logger = logging.getLogger(__name__)
            invalid_tokens = [tid for tid in token_ids if (tid is None or not isinstance(tid, (int, np.integer)) or tid < 0 or tid >= vocab_size)]
            logger.warning(f"Filtered out {invalid_count} invalid token IDs out of {len(token_ids)} total tokens (vocab_size={vocab_size})")
            logger.warning(f"Invalid token IDs: {invalid_tokens[:10]}{'...' if len(invalid_tokens) > 10 else ''}")
        
        # If no valid tokens, return empty tensor
        if len(valid_token_ids) == 0:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("No valid tokens found, returning empty tensor")
            return torch.empty(0, hidden_size)
        
        # Additional safety check before tensor creation
        max_token_id = max(valid_token_ids)
        if max_token_id >= vocab_size:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Token ID {max_token_id} is still >= vocab_size {vocab_size}. This should not happen!")
            # Filter again more aggressively
            valid_token_ids = [tid for tid in valid_token_ids if tid < vocab_size]
            if len(valid_token_ids) == 0:
                logger.warning("No valid tokens after aggressive filtering, returning empty tensor")
                return torch.empty(0, hidden_size)
        
        # Convert to tensor with explicit error handling
        try:
            token_tensor = torch.tensor(valid_token_ids, dtype=torch.long)
            # Ensure tensor is on the same device as the model
            token_tensor = token_tensor.to(self.device)
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to create tensor from valid_token_ids: {valid_token_ids[:10]}{'...' if len(valid_token_ids) > 10 else ''}")
            logger.error(f"Error: {e}")
            # Return empty tensor as fallback on the correct device
            return torch.empty(0, hidden_size, device=self.device)
        
        # Get embeddings from the model's embedding layer
        with torch.no_grad():
            if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
                # GPT-style models
                embeddings = self.model.transformer.wte(token_tensor)
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                # LLaMA-style models
                embeddings = self.model.model.embed_tokens(token_tensor)
            elif hasattr(self.model, 'embeddings') and hasattr(self.model.embeddings, 'word_embeddings'):
                # BERT-style models
                embeddings = self.model.embeddings.word_embeddings(token_tensor)
            else:
                # Fallback: try to get embeddings from the model's first layer
                try:
                    # Create a dummy input with the token IDs
                    dummy_input = token_tensor.unsqueeze(0)  # Add batch dimension
                    outputs = self.model(dummy_input, output_hidden_states=True)
                    # Get the input embeddings (before any processing)
                    embeddings = outputs.hidden_states[0].squeeze(0)
                except:
                    # Final fallback: create random embeddings (use the hidden_size we already calculated)
                    embeddings = torch.randn(len(token_ids), hidden_size).to(self.device)
                    
        return embeddings


class VLLMWrapper(LLMWrapper):
    """Wrapper for vLLM engine with graceful fallback behavior handled upstream."""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        trust_remote_code: bool = None,
        torch_dtype: Optional[torch.dtype] = None,
        token: Optional[str] = None,
        is_chat_model: Optional[bool] = None
    ):
        super().__init__(model_name, device)
        import os as _os
        self.trust_remote_code = trust_remote_code if trust_remote_code is not None else False
        self.hf_token = token
        self.is_chat_model = bool(is_chat_model) if is_chat_model is not None else None
        # Resolve dtype preference
        if torch_dtype is None:
            try:
                if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
                    torch_dtype = torch.bfloat16
                else:
                    torch_dtype = torch.float16
            except Exception:
                torch_dtype = torch.float16
        self.torch_dtype = torch_dtype
        dtype_str = 'bfloat16' if torch_dtype == torch.bfloat16 else 'float16'

        # Respect single-GPU selection for vLLM via CUDA_VISIBLE_DEVICES
        try:
            if self.device.startswith('cuda:'):
                gpu_idx = int(self.device.split(':')[1])
                _os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
        except Exception:
            pass

        # Create a small HF tokenizer to format chat prompts into strings
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
            token=self.hf_token
        )

        # Initialize vLLM engine
        try:
            from vllm import LLM
            self._engine = LLM(
                model=self.model_name,
                trust_remote_code=self.trust_remote_code,
                dtype=dtype_str,
                tokenizer_mode="auto",
                gpu_memory_utilization=0.9
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize vLLM engine: {e}")

    def _format_prompt(self, user_text: str, skip_chat_template: bool = False) -> str:
        # Prefer chat template for chat models, unless skip_chat_template=True
        if skip_chat_template:
            return user_text
        try:
            prefers_chat = False
            if self.is_chat_model is True:
                prefers_chat = True
            name = self.model_name.lower()
            if any(ind in name for ind in ['chat', 'instruct', '-it']):
                prefers_chat = True
            if prefers_chat and hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None:
                messages = [{"role": "user", "content": user_text}]
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
        except Exception:
            pass
        return user_text

    def generate(
        self,
        input_text: str,
        max_length: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        skip_chat_template: bool = False,
        **kwargs
    ) -> str:
        try:
            from vllm import SamplingParams
            prompt = self._format_prompt(input_text, skip_chat_template=skip_chat_template)
            # Map our "max_length" contract to vLLM's max_tokens for new tokens
            # Our safe length logic is in HF wrapper; here we approximate with max_tokens
            params = SamplingParams(
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                n=1
            )
            outputs = self._engine.generate([prompt], params)
            if outputs and outputs[0].outputs:
                return outputs[0].outputs[0].text.strip()
            return ""
        except Exception as e:
            self.logger.error(f"vLLM generation failed: {e}")
            return ""

    def generate_batch(
        self,
        prompts: List[str],
        max_length: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        skip_chat_template: bool = False,
        **kwargs
    ) -> List[str]:
        """Generate for a list of prompts in one vLLM call.

        vLLM preserves input order for synchronous generate; this returns a
        list of strings aligned with the input prompts.
        """
        if not prompts:
            return []
        try:
            from vllm import SamplingParams
            formatted = [self._format_prompt(p, skip_chat_template=skip_chat_template) for p in prompts]
            params = SamplingParams(
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p,
                n=1,
            )
            outs = self._engine.generate(formatted, params)
            result: List[str] = []
            for o in outs:
                if getattr(o, "outputs", None):
                    result.append(o.outputs[0].text.strip())
                else:
                    result.append("")
            return result
        except Exception as e:
            self.logger.error(f"vLLM batch generation failed: {e}")
            # Fall back to sequential to salvage outputs
            return [
                self.generate(p, max_length=max_length, temperature=temperature, do_sample=do_sample, top_p=top_p, skip_chat_template=skip_chat_template, **kwargs)
                for p in prompts
            ]

    def get_logits(self, input_text: str) -> torch.Tensor:
        raise NotImplementedError("get_logits not supported for vLLM wrapper")

    def tokenize(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def detokenize(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def get_vocab_size(self) -> int:
        return len(self.tokenizer)

    def get_model_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_type": "decoder_only",
            "torch_dtype": str(self.torch_dtype),
            "device": self.device,
            "backend": "vllm"
        }

    def release(self) -> None:
        try:
            self._engine = None
            import torch as _torch
            _torch.cuda.empty_cache()
        except Exception:
            pass

class OpenAIWrapper(LLMWrapper):
    """Wrapper for OpenAI API models."""
    
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        device: str = "cpu"  # API models don't use local device
    ):
        super().__init__(model_name, device)
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")
            
        # OpenAI models don't have local tokenizers, use tiktoken
        try:
            import tiktoken
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except Exception:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # Default encoding
            
    def generate(
        self,
        input_text: str,
        max_length: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """Generate text from input using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": input_text}],
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p if do_sample else 1.0,
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            return ""
            
    def get_logits(self, input_text: str) -> torch.Tensor:
        """OpenAI API doesn't provide logits access."""
        raise NotImplementedError("OpenAI API models don't provide logits access")
        
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text using tiktoken."""
        return self.tokenizer.encode(text)
        
    def detokenize(self, token_ids: List[int]) -> str:
        """Detokenize using tiktoken."""
        return self.tokenizer.decode(token_ids)
        
    def get_vocab_size(self) -> int:
        """Get vocabulary size (approximate for tiktoken)."""
        return self.tokenizer.n_vocab
        
    def get_model_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "model_name": self.model_name,
            "model_type": "openai_api",
            "vocab_size": self.get_vocab_size(),
            "device": "api",
            "provider": "openai"
        }


class AnthropicWrapper(LLMWrapper):
    """Wrapper for Anthropic API models."""
    
    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        device: str = "cpu"  # API models don't use local device
    ):
        super().__init__(model_name, device)
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
            
        # Use tiktoken as approximation for tokenization
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            raise ImportError("tiktoken package required for tokenization")
            
    def generate(
        self,
        input_text: str,
        max_length: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """Generate text from input using Anthropic API."""
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_length,
                temperature=temperature,
                top_p=top_p if do_sample else 1.0,
                messages=[{"role": "user", "content": input_text}],
                **kwargs
            )
            return response.content[0].text.strip()
        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            return ""
            
    def get_logits(self, input_text: str) -> torch.Tensor:
        """Anthropic API doesn't provide logits access."""
        raise NotImplementedError("Anthropic API models don't provide logits access")
        
    def tokenize(self, text: str) -> List[int]:
        """Tokenize text using tiktoken approximation."""
        return self.tokenizer.encode(text)
        
    def detokenize(self, token_ids: List[int]) -> str:
        """Detokenize using tiktoken."""
        return self.tokenizer.decode(token_ids)
        
    def get_vocab_size(self) -> int:
        """Get vocabulary size (approximate)."""
        return self.tokenizer.n_vocab
        
    def get_model_metadata(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "model_name": self.model_name,
            "model_type": "anthropic_api",
            "vocab_size": self.get_vocab_size(),
            "device": "api",
            "provider": "anthropic"
        }


class DecoderOnlyWrapper(HuggingFaceWrapper):
    """Wrapper for decoder-only models (GPT, LLaMA, Mistral, etc.)."""
    
    def _prepare_generation_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Remove token_type_ids for decoder-only models that don't support it."""
        clean_inputs = inputs.copy()
        
        # Remove token_type_ids if present (decoder-only models don't use it)
        if "token_type_ids" in clean_inputs:
            del clean_inputs["token_type_ids"]
            
        return clean_inputs


class EncoderOnlyWrapper(HuggingFaceWrapper):
    """Wrapper for encoder-only models (BERT, RoBERTa, etc.)."""
    
    def generate(
        self,
        input_text: str,
        max_length: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """Encoder-only models can't generate text - return empty string."""
        self.logger.warning(f"Encoder-only model {self.model_name} cannot generate text")
        return ""
    
    def _prepare_generation_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Encoder-only models typically support token_type_ids."""
        return inputs  # Keep all inputs including token_type_ids


class EncoderDecoderWrapper(HuggingFaceWrapper):
    """Wrapper for encoder-decoder models (T5, BART, etc.)."""
    
    def _prepare_generation_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Encoder-decoder models typically support token_type_ids."""
        return inputs  # Keep all inputs including token_type_ids
    
    def generate(
        self,
        input_text: str,
        max_length: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """Generate text from input for encoder-decoder models."""
        try:
            # Ensure input text is properly encoded
            if isinstance(input_text, str):
                input_text = input_text.encode('utf-8', errors='ignore').decode('utf-8')
            
            # Calculate safe parameters
            safe_input_length, max_new_tokens = self._get_safe_generation_params(max_length)
            
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=safe_input_length
            )
            
            # Ensure all input tensors are on the correct device
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Set up generation config with compatibility handling
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "do_sample": do_sample,
                "top_p": top_p,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # Filter out incompatible kwargs
            filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['use_cache']}
            generation_kwargs.update(filtered_kwargs)
            
            try:
                generation_config = GenerationConfig(**generation_kwargs)
            except Exception as e:
                self.logger.warning(f"Failed to create GenerationConfig: {e}. Using direct kwargs.")
                generation_config = None
            
            # Generate with compatibility handling
            with torch.no_grad():
                try:
                    if generation_config is not None:
                        outputs = self.model.generate(
                            **inputs,
                            generation_config=generation_config
                        )
                    else:
                        outputs = self.model.generate(**inputs, **generation_kwargs)
                except Exception as e:
                    # Handle generation errors for encoder-decoder models
                    error_msg = str(e).lower()
                    if 'seen_tokens' in error_msg or 'dynamiccache' in error_msg:
                        self.logger.warning(f"Compatibility error detected. Retrying with basic config.")
                        simple_kwargs = {
                            "max_new_tokens": max_new_tokens,
                            "temperature": temperature,
                            "do_sample": do_sample,
                        }
                        outputs = self.model.generate(**inputs, **simple_kwargs)
                    else:
                        raise
                
            # Decode generated tokens
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return generated_text.strip()
            
        except Exception as e:
            self.logger.error(f"Text generation failed for input: {input_text[:50]}... Error: {e}")
            return ""
