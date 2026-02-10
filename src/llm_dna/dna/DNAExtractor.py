"""
Abstract base classes for DNA extractors.

This module defines the hierarchy of DNA extractors:
- DNAExtractor: Abstract base class for all extractors
- InferenceExtractor: Abstract base class for inference-based extractors (uses model inputs/outputs)
- ParamExtractor: Abstract base class for parameter-based extractors (uses model weights directly)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union
from .DNASignature import DNASignature
from ..models.ModelWrapper import LLMWrapper


class DNAExtractor(ABC):
    """
    Abstract base class for all DNA extractors.
    
    All DNA extractors must implement the extract_dna method to generate
    DNA signatures from language models.
    """
    
    def __init__(
        self,
        dna_dim: int = 10,
        reduction_method: str = "pca",
        device: str = "auto",
        random_seed: int = 42
    ):
        """
        Initialize DNA extractor base class.
        
        Args:
            dna_dim: Target dimensionality for DNA signatures
            reduction_method: Dimensionality reduction method
            device: Computing device
            random_seed: Random seed for reproducibility
        """
        self.dna_dim = dna_dim
        self.reduction_method = reduction_method
        self.device = device
        self.random_seed = random_seed
    
    @abstractmethod
    def extract_dna(
        self,
        model: LLMWrapper,
        **kwargs
    ) -> DNASignature:
        """
        Extract DNA signature from a model.
        
        Args:
            model: Wrapped LLM model
            **kwargs: Additional arguments specific to the extractor type
            
        Returns:
            DNASignature object containing the extracted signature
        """
        pass


class InferenceExtractor(DNAExtractor):
    """
    Abstract base class for inference-based DNA extractors.
    
    These extractors work by running inference on the model with probe inputs
    and analyzing the outputs or intermediate representations.
    """
    
    @abstractmethod
    def extract_dna(
        self,
        model: LLMWrapper,
        probe_inputs: Union[str, List[str]],
        probe_set_id: str = "default",
        max_length: int = 128,
        **kwargs
    ) -> DNASignature:
        """
        Extract DNA signature using model inference.
        
        Args:
            model: Wrapped LLM model
            probe_inputs: Input text(s) to probe the model
            probe_set_id: Identifier for the probe set
            max_length: Maximum sequence length for generation
            **kwargs: Additional inference parameters
            
        Returns:
            DNASignature object containing the extracted signature
        """
        pass


class ParamExtractor(DNAExtractor):
    """
    Abstract base class for parameter-based DNA extractors.
    
    These extractors work by directly analyzing the model's parameters
    (weights, biases) without running inference.
    """
    
    @abstractmethod
    def extract_dna(
        self,
        model: LLMWrapper,
        probe_set_id: str = "params",
        **kwargs
    ) -> DNASignature:
        """
        Extract DNA signature using model parameters.
        
        Args:
            model: Wrapped LLM model
            probe_set_id: Identifier for the extraction method
            **kwargs: Additional parameter extraction options
            
        Returns:
            DNASignature object containing the extracted signature
        """
        pass