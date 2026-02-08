"""Integration tests with real model loading and DNA extraction.

These tests actually load models and perform inference, so they are slower
and require GPU/CPU resources. Mark with @pytest.mark.slow to allow skipping.
"""

import numpy as np
import pytest

from reptrace import DNAExtractionConfig, calc_dna


@pytest.mark.slow
class TestRealModelExecution:
    """Tests that actually load and execute models."""

    def test_distilgpt2_real_extraction(self):
        """Test real DNA extraction with distilgpt2 (small, fast model)."""
        config = DNAExtractionConfig(
            model_name="distilgpt2",
            dataset="rand",
            max_samples=5,
            dna_dim=64,
            save=False,
        )
        
        result = calc_dna(config)
        
        # Verify result structure
        assert result.vector is not None
        assert result.vector.shape == (64,)
        assert result.elapsed_seconds > 0
        
        # Verify vector has meaningful values (not all zeros or NaN)
        assert not np.allclose(result.vector, 0)
        assert not np.any(np.isnan(result.vector))
        assert not np.any(np.isinf(result.vector))

    def test_different_dna_dimensions(self):
        """Test extraction with different DNA dimension sizes."""
        for dim in [32, 64, 128]:
            config = DNAExtractionConfig(
                model_name="distilgpt2",
                dataset="rand",
                max_samples=5,
                dna_dim=dim,
                save=False,
            )
            
            result = calc_dna(config)
            
            assert result.vector.shape == (dim,), f"Expected shape ({dim},), got {result.vector.shape}"

    def test_different_reduction_methods(self):
        """Test extraction with different reduction methods."""
        methods = ["random_projection", "pca", "svd"]
        
        for method in methods:
            config = DNAExtractionConfig(
                model_name="distilgpt2",
                dataset="rand",
                max_samples=5,
                dna_dim=32,
                reduction_method=method,
                save=False,
            )
            
            result = calc_dna(config)
            
            assert result.vector is not None, f"Failed for reduction method: {method}"
            assert result.vector.shape == (32,)

    def test_different_sample_counts(self):
        """Test extraction with different sample counts."""
        for samples in [3, 10, 25]:
            config = DNAExtractionConfig(
                model_name="distilgpt2",
                dataset="rand",
                max_samples=samples,
                dna_dim=32,
                save=False,
            )
            
            result = calc_dna(config)
            
            assert result.vector is not None
            # With very few samples, dim may be reduced
            assert len(result.vector) > 0

    def test_extraction_is_deterministic(self):
        """Test that extraction with same seed produces same result."""
        config1 = DNAExtractionConfig(
            model_name="distilgpt2",
            dataset="rand",
            max_samples=5,
            dna_dim=32,
            random_seed=42,
            save=False,
        )
        
        config2 = DNAExtractionConfig(
            model_name="distilgpt2",
            dataset="rand",
            max_samples=5,
            dna_dim=32,
            random_seed=42,
            save=False,
        )
        
        result1 = calc_dna(config1)
        result2 = calc_dna(config2)
        
        # Results should be identical with same seed
        np.testing.assert_array_almost_equal(result1.vector, result2.vector)

    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results."""
        config1 = DNAExtractionConfig(
            model_name="distilgpt2",
            dataset="rand",
            max_samples=5,
            dna_dim=32,
            random_seed=42,
            save=False,
        )
        
        config2 = DNAExtractionConfig(
            model_name="distilgpt2",
            dataset="rand",
            max_samples=5,
            dna_dim=32,
            random_seed=123,
            save=False,
        )
        
        result1 = calc_dna(config1)
        result2 = calc_dna(config2)
        
        # Results should be different with different seeds
        assert not np.allclose(result1.vector, result2.vector)
