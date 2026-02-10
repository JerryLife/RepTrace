from llm_dna.data import DatasetLoader
from llm_dna.data import generate_rand_dataset as grd


def test_rand_dataset_autogeneration(monkeypatch, tmp_path):
    def fake_generate_random_word_samples(num_samples=600, words_per_sample=100, seed=42):
        return ["alpha beta", "gamma delta"]

    monkeypatch.setattr(grd, "generate_random_word_samples", fake_generate_random_word_samples)

    loader = DatasetLoader(data_root=tmp_path)
    texts = loader.load_dataset("rand")

    assert texts == ["alpha beta", "gamma delta"]
    assert (tmp_path / "rand" / "rand_dataset.json").exists()
