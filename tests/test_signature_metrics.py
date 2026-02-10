import numpy as np
from llm_dna.dna import DNASignature, DNAMetadata


def test_signature_distance_and_similarity():
    meta = DNAMetadata(
        model_name="test-model",
        extraction_method="text",
        probe_set_id="rand",
        probe_count=2,
        dna_dimension=3,
        embedding_dimension=3,
        reduction_method="random_projection",
        extraction_time="2025-01-01T00:00:00Z",
        computation_time_seconds=0.1,
        model_metadata={},
        extractor_config={},
        aggregation_method="concat",
    )

    sig1 = DNASignature(np.array([1.0, 0.0, 0.0]), meta)
    sig2 = DNASignature(np.array([0.0, 1.0, 0.0]), meta)

    assert np.isclose(sig1.distance_to(sig2, metric="euclidean"), np.sqrt(2))
    assert np.isclose(sig1.distance_to(sig2, metric="manhattan"), 2.0)
    assert np.isclose(sig1.similarity_to(sig2, metric="cosine"), 0.0)
