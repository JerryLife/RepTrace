from reptrace.experiments import compute_dna


def test_default_dataset_is_rand():
    args = compute_dna.parse_arguments([])
    assert args.dataset == "rand"
