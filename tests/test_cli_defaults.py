from llm_dna import DNAExtractionConfig
from llm_dna.cli import parse_arguments


def test_default_dataset_is_rand():
    args = parse_arguments(["--model-name", "distilgpt2"])
    assert args.dataset == "rand"


def test_public_api_defaults_are_stable():
    config = DNAExtractionConfig(model_name="distilgpt2")
    assert config.dataset == "rand"
