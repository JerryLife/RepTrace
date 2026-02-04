# RepTrace

Minimal, PyPI-ready package for extracting LLM DNA signatures using the existing RepTrace/DNA implementation.

## Install

```bash
pip install reptrace
```

## CLI (recommended)

The CLI mirrors the existing `dna-extract` behavior.

```bash
# Default dataset is rand
reptrace --model-name distilgpt2

# Use a specific dataset
reptrace --model-name distilgpt2 --dataset squad

# Synthetic probes
reptrace --model-name distilgpt2 --dataset syn --probe-set general
```

## Batch script

`./scripts/calc_dna.sh` is the core multi‑GPU batch runner. Default dataset is `rand`.

```bash
./scripts/calc_dna.sh --datasets "rand,squad" --gpus "0,1" --max-tasks-per-gpu 1
```

## Notes

- Default data root: `./data` (override with `--data-root`).
- Default cache: `./cache` (override with `REPTRACE_CACHE_DIR`).
- `rand` dataset auto‑generates a local word list if missing.

## Tests

```bash
pytest
```
