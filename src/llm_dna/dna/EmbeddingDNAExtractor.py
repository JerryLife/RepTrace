"""Embedding-based DNA extraction using token embeddings for efficiency."""

import numpy as np
import torch
import re
from typing import List, Optional
import logging
from datetime import datetime
import time
from tqdm import tqdm

from ..models.ModelWrapper import LLMWrapper
from .DNASignature import DNASignature, DNAMetadata
from .DNAExtractor import InferenceExtractor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import TruncatedSVD

# Optional UMAP import - fallback to PCA if not available
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class EmbeddingDNAExtractor(InferenceExtractor):
    """
    DNA extractor using token embeddings for efficiency.
    
    This approach concatenates token embeddings from model outputs instead of 
    one-hot vectors, providing computational efficiency while maintaining
    sequence information.
    """
    
    def __init__(
        self,
        dna_dim: int = 10,
        reduction_method: str = "pca",
        aggregation_method: str = "sum",
        device: str = "auto",
        random_seed: int = 42,
        batch_size: Optional[int] = None
    ):
        """
        Initialize embedding-based DNA extractor.
        
        Args:
            dna_dim: Target dimensionality for DNA signatures
            reduction_method: Method for dimensionality reduction ("pca", "svd", "random_projection")
            aggregation_method: Method for aggregating probe responses ("sum", "mean", "max", "concat")
            device: Device for computation ("auto", "cpu", "cuda")
            random_seed: Random seed for reproducibility
            batch_size: Batch size for processing (auto-detected if None)
        """
        # Call parent constructor first
        super().__init__(dna_dim, reduction_method, device, random_seed)
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Set additional parameters specific to embedding extractor
        self.aggregation_method = aggregation_method
        
        # Resolve "auto" device if needed (parent stores device as-is)
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Auto-detected device: {self.device}")
        else:
            self.logger.info(f"Using explicit device: {self.device}")
            
        # Note: Reducer and scaler are created fresh for each extraction
        # to ensure each model's DNA is derived from its own distribution
        
        # Set batch size (will be auto-detected based on model size if None)
        self.batch_size = batch_size
        self.adaptive_batch_size = None
        
        # Set random seeds
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Cache for embeddings
        self.embedding_dim = None
    
    def _estimate_model_size(self, model: LLMWrapper) -> str:
        """Estimate model size category for batch size adaptation using regex."""
        model_name = model.model_name.lower()
        
        # Use regex to extract parameter count from model name
        # Patterns like: 7b, 13b, 20b, 1.5b, 0.5b, 120b, 560m, etc.
        param_patterns = [
            r'(\d+\.?\d*)\s*b(?:illion)?(?:[^a-z]|$)',  # e.g., "7b", "1.5b", "120b"
            r'(\d+\.?\d*)b-',  # e.g., "7b-instruct" 
            r'-(\d+\.?\d*)b',  # e.g., "model-7b"
            r'(\d+\.?\d*)\s*m(?:illion)?(?:[^a-z]|$)',  # e.g., "560m", "1.2m"
            r'(\d+\.?\d*)m-',  # e.g., "560m-base"
            r'-(\d+\.?\d*)m',  # e.g., "model-560m"
        ]
        
        param_count_billions = None
        
        # Try the reliable direct count FIRST
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'num_parameters'):
                param_count = model.model.num_parameters()
                param_count_billions = param_count / 1e9
                self.logger.info(f"Got {param_count_billions:.1f}B parameters from model.num_parameters()")
        except Exception as e:
            self.logger.debug(f"Could not get direct parameter count, will try regex: {e}")
        
        # Fall back to brittle regex SECOND, only if the direct count failed
        if param_count_billions is None:
            for pattern in param_patterns:
                match = re.search(pattern, model_name)
                if match:
                    try:
                        param_count = float(match.group(1))
                        # Convert millions to billions for 'm' patterns
                        if 'm' in pattern:
                            param_count_billions = param_count / 1000  # Convert millions to billions
                            self.logger.info(f"Extracted {param_count}M parameters ({param_count_billions:.3f}B) from model name: {model.model_name}")
                        else:
                            param_count_billions = param_count
                            self.logger.info(f"Extracted {param_count_billions}B parameters from model name: {model.model_name}")
                        break
                    except ValueError:
                        continue
        
        # Classify based on parameter count
        if param_count_billions is not None:
            if param_count_billions >= 60:
                return "very_large"  # 60B+ parameters
            elif param_count_billions >= 13:
                return "very_large"  # 13-60B parameters  
            elif param_count_billions >= 7:
                return "medium"      # 7-13B parameters
            elif param_count_billions >= 3:
                return "small"       # 3-7B parameters
            elif param_count_billions >= 1:
                return "tiny"        # 1-3B parameters
            else:
                return "micro"       # <1B parameters
        
        # Conservative default for unknown models
        self.logger.warning(f"Could not determine size for model {model.model_name}, using 'medium' batch size")
        return "medium"
    
    def _get_adaptive_batch_size(self, model: LLMWrapper, num_probes: int) -> int:
        """Get adaptive batch size based on model size and available memory."""
        if self.batch_size is not None:
            return self.batch_size
            
        if self.adaptive_batch_size is not None:
            return self.adaptive_batch_size
        
        model_size = self._estimate_model_size(model)
        
        # Base batch sizes by model size
        size_to_batch = {
            "very_large": 1,    # 60B+ models: process one at a time
            "large": 2,         # 13-60B models: small batches
            "medium": 4,        # 7-13B models: moderate batches  
            "small": 8,         # 3-7B models: larger batches
            "tiny": 16,         # 1-3B models: large batches
            "micro": 32         # <1B models: very large batches
        }
        
        base_batch_size = size_to_batch[model_size]
        
        # Further reduce batch size for very large probe sets
        if num_probes > 100:
            base_batch_size = max(1, base_batch_size // 2)
        elif num_probes > 50:
            base_batch_size = max(1, base_batch_size * 3 // 4)
        
        self.adaptive_batch_size = base_batch_size
        self.logger.info(f"Using adaptive batch size {base_batch_size} for {model_size} model with {num_probes} probes")
        
        return base_batch_size
        
    def extract_dna(
        self,
        model: LLMWrapper,
        probe_inputs: List[str],
        probe_set_id: str = "default",
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        return_raw_outputs: bool = False
    ) -> DNASignature:
        """
        Extract DNA signature from a model using token embeddings.
        
        Args:
            model: Wrapped LLM model
            probe_inputs: List of input texts to probe the model
            probe_set_id: Identifier for the probe set used
            max_length: Maximum sequence length for generation
            temperature: Generation temperature (default: 0.7)
            top_p: Top-p (nucleus) sampling parameter (default: 0.9)
            return_raw_outputs: Whether to include raw outputs in metadata
            
        Returns:
            DNASignature object containing the extracted signature
        """
        start_time = time.time()
        self.logger.info(f"Extracting DNA from {model.model_name} using {len(probe_inputs)} probes (embedding method)")
        
        # Pre-filter invalid probes
        original_probe_count = len(probe_inputs)
        filtered_probes = [probe for probe in probe_inputs if probe and probe.strip()]
        
        if len(filtered_probes) < original_probe_count:
            self.logger.warning(
                f"Removed {original_probe_count - len(filtered_probes)} empty probes. "
                f"Processing {len(filtered_probes)} valid probes."
            )
        
        if not filtered_probes:
            raise ValueError("No valid probes left after filtering. Cannot extract DNA.")
        
        # Use the filtered list from now on
        probe_inputs = filtered_probes
        
        # Extract features based on model type with robust error handling
        try:
            # Determine model type and log dtype information for debugging
            model_dtype = next(model.model.parameters()).dtype
            self.logger.info(f"Model dtype detected: {model_dtype}")
            
            # Log if this is a mixed precision model that might cause dtype issues
            if model_dtype in [torch.bfloat16, torch.float16]:
                self.logger.info(f"Mixed precision model detected. Will convert tensors to float32 during feature extraction to avoid dtype mismatches.")
            
            is_encoder_decoder = hasattr(model.model.config, 'is_encoder_decoder') and model.model.config.is_encoder_decoder
            if is_encoder_decoder:
                self.logger.info("Encoder-decoder model detected. Using mean-pooled encoder states.")
                feature_vectors = self._extract_encoder_decoder_features(model, probe_inputs, max_length)
            else:
                self.logger.info("Decoder-only model detected. Using last token's hidden state from final layer.")
                feature_vectors = self._extract_decoder_only_features(model, probe_inputs, max_length)
            
            # Validate that we got valid features
            if feature_vectors.shape[0] == 0:
                raise ValueError("Feature extraction resulted in zero valid vectors.")
                
        except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as e:
            raise RuntimeError(f"Critical error during feature extraction for {model.model_name}: {e}") from e
        
        # Apply dimensionality reduction (stateless - creates fresh reducer each time)
        dna_vector = self._reduce_features(feature_vectors)
            
        # Create metadata
        computation_time = time.time() - start_time
        metadata = DNAMetadata(
            model_name=model.model_name,
            extraction_method=f"embedding_hidden_state_{self.reduction_method}_{self.aggregation_method}",
            probe_set_id=probe_set_id,
            probe_count=len(probe_inputs),
            dna_dimension=self.dna_dim,
            embedding_dimension=self.embedding_dim,
            aggregation_method=self.aggregation_method,
            reduction_method=self.reduction_method,
            extraction_time=datetime.now().isoformat(),
            computation_time_seconds=computation_time,
            model_metadata=model.get_model_metadata(),
            extractor_config={
                "dna_dim": self.dna_dim,
                "reduction_method": self.reduction_method,
                "aggregation_method": self.aggregation_method,
                "device": self.device,
                "random_seed": self.random_seed,
                "max_length": max_length,
                "feature_extraction": "contextual_hidden_states",
                "model_type": "encoder_decoder" if is_encoder_decoder else "decoder_only"
            }
        )
        
        # Create signature object
        signature = DNASignature(dna_vector, metadata)
        
        self.logger.info(f"DNA extraction completed in {computation_time:.2f}s. Signature shape: {dna_vector.shape}")
        return signature
        
    
    def _extract_decoder_only_features(self, model, probe_inputs: List[str], max_length: int) -> np.ndarray:
        """
        Extract the last hidden state for the final token of each probe input.
        This is the correct, context-rich, and fast method for decoder-only models.
        
        Args:
            model: The decoder-only model wrapper
            probe_inputs: List of input texts
            max_length: Maximum sequence length for tokenization
            
        Returns:
            np.ndarray: Feature vectors with shape (n_probes, hidden_dim)
        """
        self.logger.info("Extracting decoder-only features using last token's hidden state of prompts")
        batch_size = self._get_adaptive_batch_size(model, len(probe_inputs))
        num_batches = (len(probe_inputs) + batch_size - 1) // batch_size
        feature_vectors = []

        pbar = tqdm(range(num_batches), desc="Processing probes (decoder-only)", unit="batch")

        for i in pbar:
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(probe_inputs))
            batch_probes = probe_inputs[start_idx:end_idx]

            try:
                # Tokenize the entire batch with padding
                inputs = model.tokenizer(
                    batch_probes,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(model.device)
                
                # Ensure attention mask is in the correct dtype to avoid BFloat16/Half issues
                inputs['attention_mask'] = inputs['attention_mask'].to(dtype=torch.long)

                with torch.no_grad():
                    # Perform a single forward pass to get hidden states (no generation)
                    # Request hidden states explicitly for models that don't return them by default
                    outputs = model.model(**inputs, output_hidden_states=True)

                # Get the hidden states from the last layer - handle different output formats
                if hasattr(outputs, 'last_hidden_state'):
                    last_hidden_state = outputs.last_hidden_state
                elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    # Use the last layer's hidden states
                    last_hidden_state = outputs.hidden_states[-1]
                else:
                    raise ValueError(f"Model output does not contain accessible hidden states. Available attributes: {list(outputs.__dict__.keys())}")
                
                # Convert dtype early to avoid precision issues
                last_hidden_state = last_hidden_state.to(dtype=torch.float32)

                # Find the index of the last non-padding token for each sequence
                # attention_mask is now guaranteed to be torch.long
                sequence_lengths = inputs['attention_mask'].sum(dim=1) - 1

                # Use advanced indexing to select the hidden state of the last token for each item
                # Ensure all tensors use compatible dtypes
                batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device, dtype=torch.long)
                sequence_lengths = sequence_lengths.to(dtype=torch.long)
                
                last_token_hidden_states = last_hidden_state[batch_indices, sequence_lengths]
                
                feature_vectors.append(last_token_hidden_states.cpu().numpy())

            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                error_msg = str(e).lower()
                if 'dtype' in error_msg or 'bfloat16' in error_msg or 'half' in error_msg:
                    self.logger.warning(f"Dtype mismatch in batch {i}: {e}. This indicates a precision compatibility issue. Skipping batch.")
                else:
                    self.logger.warning(f"Failed to extract hidden states for batch {i}: {e}. Skipping batch.")
                continue
            except Exception as e:
                error_msg = str(e).lower()
                if 'dtype' in error_msg or 'bfloat16' in error_msg or 'half' in error_msg:
                    self.logger.warning(f"Dtype mismatch in batch {i}: {e}. Try running with different precision settings. Skipping batch.")
                else:
                    self.logger.error(f"Unexpected error in batch {i}: {type(e).__name__}: {e}")
                    self.logger.error(f"Batch probes: {batch_probes[:2]}...")  # Show first 2 for debugging
                continue

        if not feature_vectors:
            self.logger.warning("Failed to extract any valid features. All batches were skipped due to errors.")
            raise RuntimeError("Failed to extract any valid features for the given probes. This usually indicates dtype compatibility issues with the model.")

        concatenated_features = np.concatenate(feature_vectors, axis=0)
        
        # Validate that we have non-empty features
        if concatenated_features.shape[0] == 0:
            self.logger.warning("Concatenated features resulted in empty array.")
            raise RuntimeError("Feature extraction resulted in empty array. All probes may have failed processing.")
            
        self.logger.info(f"Successfully extracted features with shape: {concatenated_features.shape}")
        return concatenated_features
    
    def _extract_encoder_decoder_features(self, model, probe_inputs: List[str], max_length: int) -> np.ndarray:
        """
        Extract sentence embeddings from encoder-decoder models using batched mean pooling.
        This is the high-performance and methodologically correct version.
        
        Args:
            model: The encoder-decoder model wrapper
            probe_inputs: List of input texts
            max_length: Maximum sequence length for tokenization
            
        Returns:
            np.ndarray: Feature vectors with shape (n_probes, feature_dim)
        """
        self.logger.info("Extracting encoder-decoder embeddings using batched mean pooling")
        batch_size = self._get_adaptive_batch_size(model, len(probe_inputs))
        num_batches = (len(probe_inputs) + batch_size - 1) // batch_size
        feature_vectors = []

        pbar = tqdm(range(num_batches), desc="Processing probes (encoder-decoder)", unit="batch")

        for i in pbar:
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(probe_inputs))
            batch_probes = probe_inputs[start_idx:end_idx]

            if not batch_probes:
                continue

            try:
                # Tokenize the entire batch
                inputs = model.tokenizer(
                    batch_probes,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(model.device)
                
                # Ensure attention mask is in the correct dtype to avoid BFloat16/Half issues
                inputs['attention_mask'] = inputs['attention_mask'].to(dtype=torch.long)

                with torch.no_grad():
                    # Get model outputs with encoder hidden states
                    outputs = model.model(**inputs, output_hidden_states=True)
                
                # Check if encoder_last_hidden_state exists
                if not hasattr(outputs, 'encoder_last_hidden_state'):
                    self.logger.warning(f"Model does not provide encoder_last_hidden_state for batch {i}")
                    continue
                
                # Get the encoder's final hidden states and convert dtype early
                encoder_hidden_states = outputs.encoder_last_hidden_state.to(dtype=torch.float32)
                attention_mask = inputs['attention_mask']
                
                # Perform mean pooling, correctly handling the padding mask
                # Ensure mask uses the same dtype as the encoder hidden states
                mask_expanded = attention_mask.unsqueeze(-1).expand(encoder_hidden_states.size()).to(dtype=torch.float32)
                sum_embeddings = torch.sum(encoder_hidden_states * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                mean_pooled_embeddings = sum_embeddings / sum_mask

                feature_vectors.append(mean_pooled_embeddings.cpu().numpy())

            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                error_msg = str(e).lower()
                if 'dtype' in error_msg or 'bfloat16' in error_msg or 'half' in error_msg:
                    self.logger.warning(f"Dtype mismatch in encoder-decoder batch {i}: {e}. This indicates a precision compatibility issue. Skipping batch.")
                else:
                    self.logger.warning(f"Failed to process batch {i} for encoder-decoder model: {e}. Skipping batch.")
                continue
            except Exception as e:
                error_msg = str(e).lower()
                if 'dtype' in error_msg or 'bfloat16' in error_msg or 'half' in error_msg:
                    self.logger.warning(f"Dtype mismatch in encoder-decoder batch {i}: {e}. Try running with different precision settings. Skipping batch.")
                else:
                    self.logger.error(f"Unexpected error in encoder-decoder batch {i}: {type(e).__name__}: {e}")
                    self.logger.error(f"Batch probes: {batch_probes[:2]}...")  # Show first 2 for debugging
                continue

        if not feature_vectors:
            self.logger.warning("Failed to extract any valid features. All batches were skipped due to errors.")
            raise RuntimeError("Failed to extract any valid features for the given probes. This usually indicates dtype compatibility issues with the model.")
        
        concatenated_features = np.concatenate(feature_vectors, axis=0)
        
        # Validate that we have non-empty features
        if concatenated_features.shape[0] == 0:
            self.logger.warning("Concatenated features resulted in empty array.")
            raise RuntimeError("Feature extraction resulted in empty array. All probes may have failed processing.")
            
        self.logger.info(f"Successfully extracted features with shape: {concatenated_features.shape}")
        return concatenated_features
    
    def _reduce_features(self, feature_vectors: np.ndarray) -> np.ndarray:
        """Apply dimensionality reduction and aggregation to features (stateless)."""
        self.logger.info(f"Applying {self.reduction_method} for dimensionality reduction")
        self.logger.info(f"Input feature shape: {feature_vectors.shape}")
        
        # Validate input array
        if feature_vectors.size == 0:
            raise ValueError("Feature vectors array is completely empty - no valid embeddings found")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(feature_vectors)) or np.any(np.isinf(feature_vectors)):
            self.logger.warning("Found NaN or infinite values in feature vectors. Attempting to clean...")
            # Replace NaN/inf with zeros
            feature_vectors = np.nan_to_num(feature_vectors, nan=0.0, posinf=0.0, neginf=0.0)
        
        n_probes, n_features = feature_vectors.shape
        
        # Handle case where we have insufficient data for proper reduction
        if n_probes < 2:
            raise ValueError(f"Need at least 2 probes for dimensionality reduction, got {n_probes}")
        
        if n_features == 0:
            raise ValueError("Feature vectors are empty - no valid embeddings found")
        
        # Create fresh scaler and standardize features
        scaler = StandardScaler()
        standardized_features = scaler.fit_transform(feature_vectors)
        
        # Apply dimensionality reduction
        max_components = min(n_probes - 1, n_features, self.dna_dim)
        
        if max_components < self.dna_dim:
            self.logger.warning(
                f"Reducing DNA dimension from {self.dna_dim} to {max_components} "
                f"due to limited samples/features (n_probes={n_probes}, n_features={n_features})"
            )
        
        if self.reduction_method == "pca":
            reducer = PCA(n_components=max_components, random_state=self.random_seed)
        elif self.reduction_method == "svd":
            reducer = TruncatedSVD(n_components=max_components, random_state=self.random_seed)
        elif self.reduction_method == "random_projection":
            reducer = GaussianRandomProjection(n_components=max_components, random_state=self.random_seed)
        elif self.reduction_method == "umap":
            if not UMAP_AVAILABLE:
                self.logger.warning("UMAP not available, falling back to PCA")
                reducer = PCA(n_components=max_components, random_state=self.random_seed)
            else:
                reducer = umap.UMAP(
                    n_components=max_components,
                    n_neighbors=min(15, n_probes - 1),  # Ensure n_neighbors < n_samples
                    min_dist=0.1,
                    metric='cosine',
                    random_state=self.random_seed,
                    verbose=False
                )
        else:
            raise ValueError(f"Unsupported reduction method: {self.reduction_method}")
        
        reduced_features = reducer.fit_transform(standardized_features)
        
        # Aggregate the reduced probe responses with validation
        if reduced_features.shape[0] == 0:
            raise ValueError("No features left after dimensionality reduction")
            
        if self.aggregation_method == "sum":
            aggregated_dna = reduced_features.sum(axis=0)
        elif self.aggregation_method == "mean":
            aggregated_dna = reduced_features.mean(axis=0)
        elif self.aggregation_method == "max":
            aggregated_dna = reduced_features.max(axis=0)
        elif self.aggregation_method == "concat":
            # Flatten all reduced features into a single vector
            aggregated_dna = reduced_features.flatten()
            # Truncate or pad to match target dna_dim
            if len(aggregated_dna) > self.dna_dim:
                aggregated_dna = aggregated_dna[:self.dna_dim]
            elif len(aggregated_dna) < self.dna_dim:
                padding = np.zeros(self.dna_dim - len(aggregated_dna))
                aggregated_dna = np.concatenate([aggregated_dna, padding])
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")
        
        # Validate aggregated result
        if np.any(np.isnan(aggregated_dna)) or np.any(np.isinf(aggregated_dna)):
            self.logger.warning("Found NaN or infinite values in aggregated DNA. Cleaning...")
            aggregated_dna = np.nan_to_num(aggregated_dna, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Ensure we have the correct output dimension (only for non-concat methods)
        if self.aggregation_method != "concat":
            if len(aggregated_dna) < self.dna_dim:
                # Pad with zeros if needed
                padding = np.zeros(self.dna_dim - len(aggregated_dna))
                aggregated_dna = np.concatenate([aggregated_dna, padding])
            elif len(aggregated_dna) > self.dna_dim:
                aggregated_dna = aggregated_dna[:self.dna_dim]
            
        return aggregated_dna
        
    
    def batch_extract_dna(
        self,
        model: LLMWrapper,
        probe_inputs_list: List[List[str]],
        probe_set_ids: List[str],
        **kwargs
    ) -> List[DNASignature]:
        """
        Extract DNA signatures for multiple probe sets efficiently.
        
        Args:
            model: Wrapped LLM model
            probe_inputs_list: List of probe input lists
            probe_set_ids: List of probe set identifiers
            **kwargs: Additional arguments passed to extract_dna
            
        Returns:
            List of DNA signatures
        """
        signatures = []
        
        for i, (probe_inputs, probe_set_id) in enumerate(zip(probe_inputs_list, probe_set_ids)):
            self.logger.info(f"Extracting DNA for probe set {i+1}/{len(probe_inputs_list)}: {probe_set_id}")
            
            signature = self.extract_dna(
                model=model,
                probe_inputs=probe_inputs,
                probe_set_id=probe_set_id,
                **kwargs
            )
            signatures.append(signature)
            
        return signatures
    
