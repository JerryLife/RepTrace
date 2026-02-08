# RepTrace

PyPI-ready toolkit for extracting LLM DNA vectors from Hugging Face and local models.

## Install

```bash
pip install reptrace
```

## Python API (recommended)

```python
from reptrace import DNAExtractionConfig, calc_dna

config = DNAExtractionConfig(
    model_name="distilgpt2",
    dataset="rand",
    gpu_id=0,
    max_samples=100,
    trust_remote_code=True,
)

result = calc_dna(config)

# DNA vector (numpy.ndarray)
vector = result.vector
print(vector.shape)

# Saved paths (when save=True)
print(result.output_path)
print(result.summary_path)
```

## Example `calc_dna.py`

After installation, you can create your own `calc_dna.py` and import the library directly:

```python
from reptrace import DNAExtractionConfig, calc_dna

cfg = DNAExtractionConfig(
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    dataset="rand",
    gpu_id=0,
    token=None,  # or pass HF token explicitly
    metadata_file=None,  # optional: run without llm_metadata.json
)

dna = calc_dna(cfg).vector
print(dna[:10])
```

## CLI

`calc-dna` is installed with the package.

```bash
# Single model
calc-dna --model-name distilgpt2 --dataset rand --gpus 0

# Multiple models from list file with round-robin GPU assignment
calc-dna --llm-list ./configs/llm_list.txt --dataset rand --gpus 0,1

# Explicit hyperparameters
calc-dna \
  --model-name mistralai/Mistral-7B-v0.1 \
  --dataset squad \
  --gpus 1 \
  --dna-dim 256 \
  --max-samples 200 \
  --reduction-method pca \
  --embedding-merge mean \
  --load-in-8bit \
  --trust-remote-code
```

## Repo Script

For local repo usage, `scripts/calc_dna.sh` is now a thin wrapper around `scripts/calc_dna.py`:

```bash
./scripts/calc_dna.sh --llm-list ./configs/llm_list.txt --dataset rand --gpus 0,1
```

## Hugging Face Notes

- `metadata_file` is optional. If metadata is missing, extraction still runs with runtime defaults.
- Auth token can be passed with `token=...` or via `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`.
- Chat templates are applied automatically when tokenizer metadata indicates chat templating support.

## Tests

```bash
python -m pytest
```
