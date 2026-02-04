from pathlib import Path


def test_calc_dna_default_dataset_is_rand():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "calc_dna.sh"
    content = script_path.read_text(encoding="utf-8")
    assert "DEFAULT_DATASETS=\"rand\"" in content
