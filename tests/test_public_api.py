from reptrace import DNAExtractionConfig, calc_dna, calc_dna_batch


def test_public_api_symbols_are_importable():
    assert callable(calc_dna)
    assert callable(calc_dna_batch)
    config = DNAExtractionConfig(model_name="distilgpt2")
    assert config.device == "auto"
