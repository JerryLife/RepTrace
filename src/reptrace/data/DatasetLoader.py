"""
Dataset loading and management for LLM DNA extraction.

This module provides dataset loaders for various NLP datasets used in DNA extraction,
based on the EmbedLLM repository implementation.
"""

import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, Iterator
from dataclasses import dataclass
import logging
from datasets import load_dataset, IterableDataset
from huggingface_hub import hf_hub_download
import json
from sentence_transformers import SentenceTransformer


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    name: str
    subset: Optional[str] = None
    split: str = "train"
    text_column: str = "text"
    max_samples: Optional[int] = None
    cache_dir: Optional[str] = None
    download: bool = True
    streaming: bool = False
    batch_size: int = 1000


class DatasetLoader:
    """
    Dataset loader for LLM DNA extraction experiments.
    
    Supports multiple dataset sources including HuggingFace datasets,
    EmbedLLM datasets, and custom datasets.
    """
    
    def __init__(
        self,
        data_root: Union[str, Path] = "data",
        cache_embeddings: bool = True,
        embedding_model: str = "all-mpnet-base-v2"
    ):
        """
        Initialize dataset loader.
        
        Args:
            data_root: Root directory for storing datasets
            cache_embeddings: Whether to cache computed embeddings
            embedding_model: Model for computing embeddings
        """
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.cache_embeddings = cache_embeddings
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        
        self.logger = logging.getLogger(__name__)
        
        # Predefined dataset configurations
        self.dataset_configs = {
            "embedllm": DatasetConfig(
                name="RZ412/EmbedLLM",
                text_column="prompt",
                max_samples=None
            ),
            "squad": DatasetConfig(
                name="squad",
                text_column="question",
                max_samples=1000
            ),
            "commonsense_qa": DatasetConfig(
                name="commonsense_qa", 
                text_column="question",
                max_samples=1000
            ),
            "hellaswag": DatasetConfig(
                name="hellaswag",
                text_column="ctx",
                max_samples=1000
            ),
            "winogrande": DatasetConfig(
                name="winogrande",
                subset="winogrande_xl",
                text_column="sentence",
                max_samples=1000
            ),
            "arc": DatasetConfig(
                name="ai2_arc",
                subset="ARC-Challenge", 
                text_column="question",
                max_samples=1000
            ),
            "mmlu": DatasetConfig(
                name="cais/mmlu",
                subset="all",
                split="test",
                text_column="question",
                max_samples=1000
            ),
            "rand": DatasetConfig(
                name="rand",
                text_column="text",
                max_samples=600
            )
        }
    
    def download_embedllm_data(self) -> None:
        """Download EmbedLLM dataset files."""
        self.logger.info("Downloading EmbedLLM dataset...")
        
        repo_id = "RZ412/EmbedLLM"
        files = ["train.csv", "test.csv", "val.csv", "model_order.csv", "question_order.csv"]
        
        embedllm_dir = self.data_root / "embedllm"
        embedllm_dir.mkdir(exist_ok=True)
        
        for file in files:
            self.logger.info(f"Downloading {file}...")
            try:
                downloaded_file = hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset", 
                    filename=file,
                    local_dir=str(embedllm_dir)
                )
                self.logger.info(f"Saved {file} to {downloaded_file}")
            except Exception as e:
                self.logger.error(f"Failed to download {file}: {e}")
    
    def load_dataset(
        self,
        dataset_name: str,
        config: Optional[DatasetConfig] = None,
        return_raw: bool = False
    ) -> Union[List[str], Tuple[List[str], Any], Iterator[str]]:
        """
        Load dataset and return text samples with streaming support.
        
        Args:
            dataset_name: Name of dataset to load
            config: Optional dataset configuration
            return_raw: Whether to return raw dataset object
            
        Returns:
            List of text samples, tuple (texts, raw_dataset), or iterator for streaming
        """
        if config is None:
            if dataset_name in self.dataset_configs:
                config = self.dataset_configs[dataset_name]
            else:
                raise ValueError(f"No configuration found for dataset: {dataset_name}")
        
        self.logger.info(f"Loading dataset: {config.name} (streaming: {config.streaming})")
        
        # Handle special case for EmbedLLM dataset
        if dataset_name == "embedllm":
            return self._load_embedllm_dataset(config, return_raw)
        
        # Handle special case for random word dataset
        if dataset_name == "rand":
            return self._load_rand_dataset(config, return_raw)
        
        # Load standard HuggingFace dataset
        try:
            if config.subset:
                dataset = load_dataset(
                    config.name, 
                    config.subset, 
                    split=config.split,
                    streaming=config.streaming
                )
            else:
                dataset = load_dataset(
                    config.name, 
                    split=config.split,
                    streaming=config.streaming
                )
                
            # For streaming datasets, return iterator
            if config.streaming:
                return self._create_streaming_iterator(dataset, config, return_raw)
            
            # Extract text samples for non-streaming
            if config.text_column not in dataset.column_names:
                available_cols = dataset.column_names
                self.logger.error(f"Column '{config.text_column}' not found. Available: {available_cols}")
                raise ValueError(f"Text column '{config.text_column}' not found in dataset")
            
            texts = dataset[config.text_column]
            
            # Apply sample limit
            if config.max_samples and len(texts) > config.max_samples:
                texts = texts[:config.max_samples]
                
            self.logger.info(f"Loaded {len(texts)} samples from {config.name}")
            
            if return_raw:
                return texts, dataset
            return texts
            
        except Exception as e:
            self.logger.error(f"Failed to load dataset {config.name}: {e}")
            raise
    
    def _load_embedllm_dataset(
        self,
        config: DatasetConfig,
        return_raw: bool = False
    ) -> Union[List[str], Tuple[List[str], pd.DataFrame]]:
        """Load EmbedLLM dataset prompts from train split only.

        Always uses data/embedllm/train.csv and the configured text column
        (default: 'prompt'). Other splits (val/test) are not used here to
        avoid leakage into evaluation workflows.
        """
        embedllm_dir = self.data_root / "embedllm"

        # Ensure dataset files are present; specifically require train.csv
        if not embedllm_dir.exists() or not (embedllm_dir / "train.csv").exists():
            self.download_embedllm_data()

        train_file = embedllm_dir / "train.csv"
        if not train_file.exists():
            raise FileNotFoundError(f"EmbedLLM train file not found: {train_file}")

        df = pd.read_csv(train_file)

        # Validate text column exists
        if config.text_column not in df.columns:
            # Try a couple of common alternatives before failing hard
            fallback_col = None
            for cand in ("prompt", "question", "text"):
                if cand in df.columns:
                    fallback_col = cand
                    break
            if fallback_col is None:
                raise ValueError(
                    f"Text column '{config.text_column}' not found in {train_file}. "
                    f"Available columns: {list(df.columns)}"
                )
            self.logger.warning(
                f"Requested text column '{config.text_column}' not found in {train_file}; "
                f"falling back to '{fallback_col}'."
            )
            text_col = fallback_col
        else:
            text_col = config.text_column

        texts = df[text_col].astype(str).tolist()

        if config.max_samples and len(texts) > config.max_samples:
            texts = texts[:config.max_samples]

        self.logger.info(
            f"Loaded {len(texts)} EmbedLLM prompts from train.csv (column='{text_col}')"
        )

        if return_raw:
            return texts, df
        return texts
    
    def _load_rand_dataset(
        self,
        config: DatasetConfig,
        return_raw: bool = False
    ) -> Union[List[str], Tuple[List[str], List[str]]]:
        """Load random word dataset from local JSON file.
        
        The dataset file should be located at data_root/rand/rand_dataset.json
        and contain a JSON array of strings, each with 100 random words.
        """
        rand_dir = self.data_root / "rand"
        rand_file = rand_dir / "rand_dataset.json"
        
        # Check if dataset file exists, if not, generate it
        if not rand_file.exists():
            self.logger.warning(
                f"Random dataset file not found at {rand_file}. "
                "Attempting to generate it..."
            )
            try:
                # Import here to avoid circular dependencies
                import sys
                from pathlib import Path
                project_root = Path(__file__).parent.parent.parent
                sys.path.insert(0, str(project_root))
                
                from .generate_rand_dataset import generate_random_word_samples, save_dataset
                
                # Generate the dataset
                samples = generate_random_word_samples(
                    num_samples=600,
                    words_per_sample=100,
                    seed=42
                )
                save_dataset(samples, rand_file, format="json")
                self.logger.info(f"Generated random dataset at {rand_file}")
            except Exception as e:
                self.logger.error(f"Failed to generate random dataset: {e}")
                raise FileNotFoundError(
                    f"Random dataset file not found at {rand_file} and "
                    f"failed to generate it: {e}"
                )
        
        # Load the dataset
        with open(rand_file, 'r', encoding='utf-8') as f:
            texts = json.load(f)
        
        # Apply sample limit
        if config.max_samples and len(texts) > config.max_samples:
            texts = texts[:config.max_samples]
        
        self.logger.info(
            f"Loaded {len(texts)} random word samples from {rand_file}"
        )
        
        if return_raw:
            return texts, texts
        return texts
    
    def _create_streaming_iterator(
        self,
        dataset: IterableDataset,
        config: DatasetConfig,
        return_raw: bool = False
    ) -> Iterator[str]:
        """Create streaming iterator for dataset texts."""
        def text_iterator():
            count = 0
            for item in dataset:
                if config.max_samples and count >= config.max_samples:
                    break
                    
                if config.text_column not in item:
                    available_cols = list(item.keys())
                    raise ValueError(f"Text column '{config.text_column}' not found. Available: {available_cols}")
                
                yield item[config.text_column]
                count += 1
                
                if count % 1000 == 0:
                    self.logger.info(f"Streamed {count} samples from {config.name}")
            
            self.logger.info(f"Finished streaming {count} samples from {config.name}")
        
        return text_iterator()
    
    def load_dataset_batched(
        self,
        dataset_name: str,
        config: Optional[DatasetConfig] = None,
        batch_size: Optional[int] = None
    ) -> Iterator[List[str]]:
        """
        Load dataset in batches for memory-efficient processing.
        
        Args:
            dataset_name: Name of dataset to load
            config: Dataset configuration with streaming enabled
            batch_size: Override batch size from config
            
        Yields:
            Batches of text samples
        """
        if config is None:
            if dataset_name in self.dataset_configs:
                config = self.dataset_configs[dataset_name]
            else:
                raise ValueError(f"No configuration found for dataset: {dataset_name}")
        
        # Enable streaming for batched loading
        config.streaming = True
        effective_batch_size = batch_size or config.batch_size
        
        self.logger.info(f"Loading dataset in batches of {effective_batch_size}: {config.name}")
        
        text_iterator = self.load_dataset(dataset_name, config)
        
        batch = []
        for text in text_iterator:
            batch.append(text)
            
            if len(batch) >= effective_batch_size:
                yield batch
                batch = []
        
        # Yield remaining texts
        if batch:
            yield batch
    
    def get_dataset_embeddings(
        self,
        dataset_name: str,
        config: Optional[DatasetConfig] = None,
        force_recompute: bool = False,
        use_streaming: bool = True
    ) -> torch.Tensor:
        """
        Get embeddings for dataset texts with streaming support.
        
        Args:
            dataset_name: Name of dataset
            config: Dataset configuration
            force_recompute: Whether to force recomputation of embeddings
            use_streaming: Whether to use streaming for large datasets
            
        Returns:
            Tensor of embeddings
        """
        # Check for cached embeddings
        cache_file = self.data_root / f"{dataset_name}_embeddings.pth"
        
        if not force_recompute and cache_file.exists():
            self.logger.info(f"Loading cached embeddings from {cache_file}")
            return torch.load(cache_file)
        
        # Initialize embedding model if needed
        if self.embedding_model is None:
            self.logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        # Use streaming for large datasets or when explicitly requested
        if use_streaming and (config is None or config.max_samples is None or config.max_samples > 5000):
            return self._compute_embeddings_streaming(dataset_name, config, cache_file)
        else:
            # Load dataset texts
            texts = self.load_dataset(dataset_name, config)
            
            # Compute embeddings
            self.logger.info(f"Computing embeddings for {len(texts)} texts...")
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            embedding_tensor = torch.tensor(embeddings)
            
            # Cache embeddings
            if self.cache_embeddings:
                torch.save(embedding_tensor, cache_file)
                self.logger.info(f"Cached embeddings to {cache_file}")
            
            return embedding_tensor
    
    def _compute_embeddings_streaming(
        self,
        dataset_name: str,
        config: Optional[DatasetConfig],
        cache_file: Path
    ) -> torch.Tensor:
        """Compute embeddings using streaming for memory efficiency."""
        self.logger.info(f"Computing embeddings using streaming for {dataset_name}...")
        
        all_embeddings = []
        total_processed = 0
        
        # Process in batches
        for batch_texts in self.load_dataset_batched(dataset_name, config):
            batch_embeddings = self.embedding_model.encode(
                batch_texts, 
                show_progress_bar=False,
                batch_size=32  # Internal batch size for encoding
            )
            all_embeddings.append(torch.tensor(batch_embeddings))
            total_processed += len(batch_texts)
            
            self.logger.info(f"Processed {total_processed} samples...")
        
        # Concatenate all embeddings
        if all_embeddings:
            embedding_tensor = torch.cat(all_embeddings, dim=0)
            self.logger.info(f"Computed embeddings for {len(embedding_tensor)} texts using streaming")
            
            # Cache embeddings
            if self.cache_embeddings:
                torch.save(embedding_tensor, cache_file)
                self.logger.info(f"Cached embeddings to {cache_file}")
            
            return embedding_tensor
        else:
            return torch.empty(0, self.embedding_model.get_sentence_embedding_dimension())
    
    def create_probe_dataset(
        self,
        dataset_names: List[str],
        samples_per_dataset: int = 100,
        mix_datasets: bool = True,
        use_streaming: bool = False,
        seed: int = 42
    ) -> List[str]:
        """
        Create a mixed probe dataset from multiple sources with streaming support.
        
        Args:
            dataset_names: List of dataset names to include
            samples_per_dataset: Number of samples per dataset
            mix_datasets: Whether to shuffle samples from different datasets
            use_streaming: Whether to use streaming for large datasets
            
        Returns:
            List of probe texts
        """
        all_probes = []
        
        for dataset_name in dataset_names:
            try:
                # Create config with sample limit
                config = self.dataset_configs.get(dataset_name)
                if config:
                    config = DatasetConfig(
                        name=config.name,
                        subset=config.subset,
                        split=config.split,
                        text_column=config.text_column,
                        max_samples=samples_per_dataset,
                        streaming=use_streaming and samples_per_dataset > 1000,
                        batch_size=min(config.batch_size, samples_per_dataset)
                    )
                
                if config.streaming:
                    # Use streaming for large datasets
                    texts = []
                    for batch in self.load_dataset_batched(dataset_name, config):
                        texts.extend(batch)
                        if len(texts) >= samples_per_dataset:
                            texts = texts[:samples_per_dataset]
                            break
                else:
                    texts = self.load_dataset(dataset_name, config)
                
                # Limit samples with random sampling
                if len(texts) > samples_per_dataset:
                    np.random.seed(seed)
                    indices = np.random.choice(len(texts), samples_per_dataset, replace=False)
                    texts = [texts[i] for i in indices]
                
                all_probes.extend(texts)
                self.logger.info(f"Added {len(texts)} probes from {dataset_name}")
                
            except Exception as e:
                self.logger.warning(f"Failed to load probes from {dataset_name}: {e}")
        
        # Mix datasets
        if mix_datasets:
            np.random.seed(seed)
            np.random.shuffle(all_probes)
        
        self.logger.info(f"Created probe dataset with {len(all_probes)} total samples")
        return all_probes
    
    def enable_streaming(self, dataset_name: str, batch_size: int = 1000) -> DatasetConfig:
        """
        Create a streaming-enabled configuration for a dataset.
        
        Args:
            dataset_name: Name of dataset
            batch_size: Batch size for streaming
            
        Returns:
            DatasetConfig with streaming enabled
        """
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        base_config = self.dataset_configs[dataset_name]
        return DatasetConfig(
            name=base_config.name,
            subset=base_config.subset,
            split=base_config.split,
            text_column=base_config.text_column,
            max_samples=base_config.max_samples,
            cache_dir=base_config.cache_dir,
            download=base_config.download,
            streaming=True,
            batch_size=batch_size
        )
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available dataset names."""
        return list(self.dataset_configs.keys())
    
    def save_dataset_info(self, dataset_name: str, output_file: Optional[Path] = None) -> Dict[str, Any]:
        """
        Save dataset information and statistics.
        
        Args:
            dataset_name: Name of dataset
            output_file: Optional output file path
            
        Returns:
            Dataset information dictionary
        """
        texts = self.load_dataset(dataset_name)
        
        # Compute statistics
        lengths = [len(text.split()) for text in texts]
        info = {
            "dataset_name": dataset_name,
            "num_samples": int(len(texts)),
            "avg_length": float(np.mean(lengths)),
            "std_length": float(np.std(lengths)),
            "min_length": int(np.min(lengths)),
            "max_length": int(np.max(lengths)),
            "sample_texts": texts[:5] if len(texts) > 0 else []
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(info, f, indent=2, default=str)
        
        return info
