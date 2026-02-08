"""
DNA Signature class for LLM functional signatures.
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import pickle
import json
from datetime import datetime
from dataclasses import dataclass, asdict
import logging


@dataclass
class DNAMetadata:
    """Metadata for DNA signature."""
    model_name: str
    extraction_method: str
    probe_set_id: str
    probe_count: int
    dna_dimension: int
    embedding_dimension: int
    reduction_method: Optional[str]
    extraction_time: str
    computation_time_seconds: float
    model_metadata: Dict[str, Any]
    extractor_config: Dict[str, Any]
    aggregation_method: Optional[str] = None  # Optional since simplified approach may not use aggregation


class DNASignature:
    """
    DNA Signature class for storing and manipulating LLM functional signatures.
    
    This class provides a comprehensive interface for DNA signatures including
    storage, loading, comparison, and analysis operations.
    """
    
    def __init__(
        self,
        signature: np.ndarray,
        metadata: DNAMetadata
    ):
        """
        Initialize DNA signature.
        
        Args:
            signature: The DNA signature vector
            metadata: Metadata about the signature
        """
        self.signature = np.array(signature, dtype=np.float32)
        self.metadata = metadata
        self.logger = logging.getLogger(__name__)
        
        # Validate signature
        if len(self.signature.shape) != 1:
            raise ValueError(f"DNA signature must be 1D, got shape {self.signature.shape}")
        
        # Fail-fast for empty signatures
        if self.signature.size == 0:
            raise ValueError("Empty DNA signature detected. DNA extraction failed.")
        
        if len(self.signature) != metadata.dna_dimension:
            self.logger.warning(
                f"Signature length {len(self.signature)} != metadata dimension {metadata.dna_dimension}"
            )
    
    @property
    def model_name(self) -> str:
        """Get model name."""
        return self.metadata.model_name
    
    @property
    def dimension(self) -> int:
        """Get DNA dimension."""
        return len(self.signature)
    
    @property
    def extraction_method(self) -> str:
        """Get extraction method."""
        return self.metadata.extraction_method
    
    def __len__(self) -> int:
        """Get DNA signature length."""
        return len(self.signature)
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"DNASignature(model={self.model_name}, "
            f"dim={self.dimension}, "
            f"method={self.extraction_method})"
        )
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"DNASignature(\n"
            f"  model_name='{self.model_name}',\n"
            f"  dimension={self.dimension},\n"
            f"  extraction_method='{self.extraction_method}',\n"
            f"  probe_count={self.metadata.probe_count},\n"
            f"  extraction_time='{self.metadata.extraction_time}'\n"
            f")"
        )
    
    def distance_to(
        self,
        other: 'DNASignature',
        metric: str = "euclidean"
    ) -> float:
        """
        Compute distance to another DNA signature.
        
        Args:
            other: Another DNA signature
            metric: Distance metric ("euclidean", "cosine", "manhattan", "hamming")
            
        Returns:
            Distance value
        """
        if len(self.signature) != len(other.signature):
            raise ValueError(
                f"DNA signatures must have same dimension: "
                f"{len(self.signature)} vs {len(other.signature)}"
            )
        
        sig1, sig2 = self.signature, other.signature
        
        if metric == "euclidean":
            return float(np.linalg.norm(sig1 - sig2))
        elif metric == "cosine":
            dot_product = np.dot(sig1, sig2)
            norms = np.linalg.norm(sig1) * np.linalg.norm(sig2)
            if norms == 0:
                return 1.0
            return float(1 - dot_product / norms)
        elif metric == "manhattan":
            return float(np.sum(np.abs(sig1 - sig2)))
        elif metric == "hamming":
            # For continuous values, use threshold-based Hamming distance
            threshold = (np.mean(sig1) + np.mean(sig2)) / 2
            bin1 = (sig1 > threshold).astype(int)
            bin2 = (sig2 > threshold).astype(int)
            return float(np.sum(bin1 != bin2) / len(sig1))
        else:
            raise ValueError(f"Unknown distance metric: {metric}")
    
    def similarity_to(
        self,
        other: 'DNASignature',
        metric: str = "cosine"
    ) -> float:
        """
        Compute similarity to another DNA signature.
        
        Args:
            other: Another DNA signature
            metric: Similarity metric ("cosine", "correlation")
            
        Returns:
            Similarity value (higher = more similar)
        """
        if len(self.signature) != len(other.signature):
            raise ValueError(
                f"DNA signatures must have same dimension: "
                f"{len(self.signature)} vs {len(other.signature)}"
            )
        
        sig1, sig2 = self.signature, other.signature
        
        if metric == "cosine":
            dot_product = np.dot(sig1, sig2)
            norms = np.linalg.norm(sig1) * np.linalg.norm(sig2)
            if norms == 0:
                return 0.0
            return float(dot_product / norms)
        elif metric == "correlation":
            if np.std(sig1) == 0 or np.std(sig2) == 0:
                return 0.0
            return float(np.corrcoef(sig1, sig2)[0, 1])
        else:
            raise ValueError(f"Unknown similarity metric: {metric}")
    
    def normalize(self, method: str = "l2") -> 'DNASignature':
        """
        Create normalized copy of DNA signature.
        
        Args:
            method: Normalization method ("l2", "l1", "max", "zscore")
            
        Returns:
            New normalized DNA signature
        """
        if method == "l2":
            norm = np.linalg.norm(self.signature)
            normalized = self.signature / norm if norm > 0 else self.signature
        elif method == "l1":
            norm = np.sum(np.abs(self.signature))
            normalized = self.signature / norm if norm > 0 else self.signature
        elif method == "max":
            max_val = np.max(np.abs(self.signature))
            normalized = self.signature / max_val if max_val > 0 else self.signature
        elif method == "zscore":
            mean_val = np.mean(self.signature)
            std_val = np.std(self.signature)
            normalized = (self.signature - mean_val) / std_val if std_val > 0 else self.signature
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Create new metadata with normalization info
        new_metadata = DNAMetadata(
            **{**asdict(self.metadata), 
               'extraction_method': f"{self.metadata.extraction_method}_normalized_{method}"}
        )
        
        return DNASignature(normalized, new_metadata)
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get statistical summary of DNA signature.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "mean": float(np.mean(self.signature)),
            "std": float(np.std(self.signature)),
            "min": float(np.min(self.signature)),
            "max": float(np.max(self.signature)),
            "median": float(np.median(self.signature)),
            "l1_norm": float(np.sum(np.abs(self.signature))),
            "l2_norm": float(np.linalg.norm(self.signature)),
            "sparsity": float(np.sum(self.signature == 0) / len(self.signature)),
            "entropy": float(self._compute_entropy())
        }
    
    def _compute_entropy(self) -> float:
        """Compute entropy of DNA signature values."""
        # Discretize values for entropy calculation
        bins = 50
        hist, _ = np.histogram(self.signature, bins=bins)
        hist = hist + 1e-10  # Avoid log(0)
        probs = hist / np.sum(hist)
        return -np.sum(probs * np.log2(probs))
    
    def save(self, filepath: Union[str, Path], format: str = "csv") -> None:
        """
        Save DNA signature to file.
        
        Args:
            filepath: Output file path
            format: Save format ("csv", "npz", "json", "pickle")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "csv":
            # Save as CSV with just the signature values as a single column
            import pandas as pd
            df = pd.DataFrame(self.signature, columns=['dna_value'])
            df.to_csv(filepath, index=False)
            
        elif format == "npz":
            # Save as compressed numpy format
            save_data = {
                "signature": self.signature,
                "metadata": asdict(self.metadata)
            }
            np.savez_compressed(filepath, **save_data)
            
        elif format == "json":
            # Save as JSON (signature as list)
            save_data = {
                "signature": self.signature.tolist(),
                "metadata": asdict(self.metadata)
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, default=str, ensure_ascii=False, allow_nan=False)
                
        elif format == "pickle":
            # Save as pickle
            with open(filepath, 'wb') as f:
                pickle.dump(self, f)
                
        else:
            raise ValueError(f"Unknown save format: {format}")
        
        self.logger.info(f"DNA signature saved to {filepath} (format: {format})")
    
    @classmethod
    def load(cls, filepath: Union[str, Path], format: str = "auto") -> 'DNASignature':
        """
        Load DNA signature from file.
        
        Args:
            filepath: Input file path
            format: Load format ("auto", "csv", "npz", "json", "pickle")
            
        Returns:
            Loaded DNA signature
        """
        filepath = Path(filepath)
        
        if format == "auto":
            format = filepath.suffix.lower().lstrip('.')
            if format == "gz":  # Handle .npz.gz
                format = "npz"
        
        if format == "csv":
            # Load CSV format - simple column of DNA values
            import pandas as pd
            df = pd.read_csv(filepath)
            signature = df['dna_value'].values.astype(np.float32)
            
            # Create basic metadata since CSV doesn't store metadata
            metadata = DNAMetadata(
                model_name=filepath.stem.replace('_dna', ''),
                extraction_method="unknown",
                probe_set_id="unknown",
                probe_count=1,
                dna_dimension=len(signature),
                embedding_dimension=768,
                reduction_method="unknown",
                extraction_time=str(filepath.stat().st_mtime),
                computation_time_seconds=0.0,
                model_metadata={},
                extractor_config={}
            )
            
        elif format == "npz":
            data = np.load(filepath, allow_pickle=True)
            signature = data["signature"]
            metadata_dict = data["metadata"].item()
            metadata = DNAMetadata(**metadata_dict)
            
        elif format == "json":
            with open(filepath, 'r') as f:
                data = json.load(f)
            signature = np.array(data["signature"], dtype=np.float32)
            metadata = DNAMetadata(**data["metadata"])
            
        elif format == "pickle":
            with open(filepath, 'rb') as f:
                return pickle.load(f)
                
        else:
            raise ValueError(f"Unknown load format: {format}")
        
        return cls(signature, metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "signature": self.signature.tolist(),
            "metadata": asdict(self.metadata)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DNASignature':
        """Create from dictionary representation."""
        signature = np.array(data["signature"], dtype=np.float32)
        metadata = DNAMetadata(**data["metadata"])
        return cls(signature, metadata)


class DNACollection:
    """Collection of DNA signatures for batch operations."""
    
    def __init__(self, signatures: Optional[List[DNASignature]] = None):
        """
        Initialize DNA collection.
        
        Args:
            signatures: List of DNA signatures
        """
        self.signatures = signatures or []
        self.logger = logging.getLogger(__name__)
    
    def add(self, signature: DNASignature) -> None:
        """Add signature to collection."""
        self.signatures.append(signature)
    
    def __len__(self) -> int:
        """Get number of signatures."""
        return len(self.signatures)
    
    def __iter__(self):
        """Iterate over signatures."""
        return iter(self.signatures)
    
    def __getitem__(self, index: int) -> DNASignature:
        """Get signature by index."""
        return self.signatures[index]
    
    def get_distance_matrix(self, metric: str = "euclidean") -> np.ndarray:
        """
        Compute pairwise distance matrix.
        
        Args:
            metric: Distance metric
            
        Returns:
            Distance matrix
        """
        n = len(self.signatures)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.signatures[i].distance_to(self.signatures[j], metric)
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def get_model_names(self) -> List[str]:
        """Get list of model names."""
        return [sig.model_name for sig in self.signatures]
    
    def filter_by_method(self, method: str) -> 'DNACollection':
        """Filter signatures by extraction method."""
        filtered = [sig for sig in self.signatures if sig.extraction_method == method]
        return DNACollection(filtered)
    
    def save(self, output_path: Union[str, Path], format: str = "csv") -> None:
        """
        Save all signatures to file or directory.
        
        Args:
            output_path: Output file path (for csv) or directory path (for individual files)
            format: Save format ("csv" for single file with rows, "individual" for separate files)
        """
        output_path = Path(output_path)
        
        if format == "csv":
            # Save all DNA signatures as rows in a single CSV file
            import pandas as pd
            
            # Create DataFrame where each row is a DNA signature
            rows = []
            for sig in self.signatures:
                # Create a row with model name and DNA values
                row = {"model_name": sig.model_name}
                for i, value in enumerate(sig.signature):
                    row[f"dna_{i}"] = value
                rows.append(row)
            
            df = pd.DataFrame(rows)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            self.logger.info(f"Saved {len(self.signatures)} signatures to {output_path} (CSV format)")
            
        elif format == "individual":
            # Save each signature as individual file (backward compatibility)
            directory = output_path
            directory.mkdir(parents=True, exist_ok=True)
            
            for sig in self.signatures:
                filename = f"{sig.model_name.replace('/', '_')}_dna.csv"
                sig.save(directory / filename)
            
            self.logger.info(f"Saved {len(self.signatures)} individual signatures to {directory}")
            
        else:
            raise ValueError(f"Unknown save format: {format}. Use 'csv' or 'individual'.")
    
    @classmethod
    def load_from_csv(cls, filepath: Union[str, Path]) -> 'DNACollection':
        """
        Load all signatures from a single CSV file where each row is a DNA signature.
        
        Args:
            filepath: Path to CSV file with DNA signatures as rows
            
        Returns:
            DNACollection with loaded signatures
        """
        import pandas as pd
        filepath = Path(filepath)
        
        df = pd.read_csv(filepath)
        signatures = []
        
        for _, row in df.iterrows():
            model_name = row['model_name']
            # Extract DNA values (all columns except model_name)
            dna_columns = [col for col in df.columns if col.startswith('dna_')]
            dna_values = row[dna_columns].values.astype(np.float32)
            
            # Create metadata
            metadata = DNAMetadata(
                model_name=model_name,
                extraction_method="csv_loaded",
                probe_set_id="unknown",
                probe_count=1,
                dna_dimension=len(dna_values),
                embedding_dimension=768,
                reduction_method="unknown",
                extraction_time=str(filepath.stat().st_mtime),
                computation_time_seconds=0.0,
                model_metadata={},
                extractor_config={}
            )
            
            signature = DNASignature(dna_values, metadata)
            signatures.append(signature)
        
        return cls(signatures)
    
    @classmethod
    def load_from_directory(cls, directory: Union[str, Path]) -> 'DNACollection':
        """Load all signatures from directory (individual files)."""
        directory = Path(directory)
        signatures = []
        
        # Try CSV files first (default format), then fallback to other formats
        patterns = ["*_dna.csv", "*_dna.npz", "*_dna.json"]
        
        for pattern in patterns:
            for filepath in directory.glob(pattern):
                try:
                    sig = DNASignature.load(filepath)
                    signatures.append(sig)
                except Exception as e:
                    logging.warning(f"Failed to load {filepath}: {e}")
        
        return cls(signatures)
    
    @classmethod
    def load(cls, path: Union[str, Path], format: str = "auto") -> 'DNACollection':
        """
        Load DNA signatures from file or directory.
        
        Args:
            path: Path to file or directory
            format: Load format ("auto", "csv", "directory")
            
        Returns:
            DNACollection with loaded signatures
        """
        path = Path(path)
        
        if format == "auto":
            if path.is_file() and path.suffix.lower() == '.csv':
                return cls.load_from_csv(path)
            elif path.is_dir():
                return cls.load_from_directory(path)
            else:
                raise ValueError(f"Cannot auto-detect format for {path}")
        elif format == "csv":
            return cls.load_from_csv(path)
        elif format == "directory":
            return cls.load_from_directory(path)
        else:
            raise ValueError(f"Unknown load format: {format}")
