#!/usr/bin/env python3
"""
Example script demonstrating RepTrace DNA extraction API.

Usage:
    python scripts/calc_dna.py
    python scripts/calc_dna.py --model Qwen/Qwen2.5-0.5B-Instruct
    python scripts/calc_dna.py --model distilgpt2 --gpu 0 --samples 50
"""

import argparse
import sys
from pathlib import Path

# Add src to path for local development
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reptrace import DNAExtractionConfig, calc_dna


def main():
    parser = argparse.ArgumentParser(
        description="Extract DNA vector from an LLM using RepTrace.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="distilgpt2",
        help="Model name or Hugging Face model ID"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="rand",
        help="Dataset for probe generation"
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU ID to use (None for CPU)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of probe samples to use for DNA extraction"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to disk"
    )
    args = parser.parse_args()

    # Set paths relative to project root
    metadata_file = ROOT / "configs" / "llm_metadata.json"
    data_root = str(ROOT / "data")
    output_dir = ROOT / "out"

    # Create configuration using the public API
    config = DNAExtractionConfig(
        model_name=args.model,
        dataset=args.dataset,
        gpu_id=args.gpu,
        max_samples=args.samples,
        data_root=data_root,
        metadata_file=metadata_file if metadata_file.exists() else None,
        output_dir=output_dir,
        save=not args.no_save,
        trust_remote_code=True,
    )

    # Extract DNA
    print(f"Extracting DNA from: {args.model}")
    print(f"Using {args.samples} probe samples")
    result = calc_dna(config)

    # Display results
    print(f"\nDNA vector shape: {result.vector.shape}")
    print(f"Extraction time: {result.elapsed_seconds:.2f}s")
    print(f"First 10 values: {result.vector[:10]}")

    if result.output_path:
        print(f"\nSaved to: {result.output_path}")
        print(f"Summary: {result.summary_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
