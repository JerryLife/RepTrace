#!/usr/bin/env python3
"""
Generate random word dataset for DNA extraction.

Creates a dataset of 600 samples, each containing 100 random English words.
Uses the wonderwords library to generate random words.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List
from wonderwords import RandomWord

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_random_word_samples(
    num_samples: int = 600,
    words_per_sample: int = 100,
    seed: int = 42
) -> List[str]:
    """
    Generate random word samples using wonderwords.
    
    Args:
        num_samples: Number of samples to generate
        words_per_sample: Number of words per sample
        seed: Random seed for reproducibility
        
    Returns:
        List of strings, each containing words_per_sample random words
    """
    r = RandomWord()
    samples = []
    
    logger.info(f"Generating {num_samples} samples with {words_per_sample} words each...")
    
    for i in range(num_samples):
        # Generate random words by calling r.word() multiple times
        words = [r.word() for _ in range(words_per_sample)]
        # Join words with spaces to create a "sentence"
        sample = " ".join(words)
        samples.append(sample)
        
        if (i + 1) % 100 == 0:
            logger.info(f"Generated {i + 1}/{num_samples} samples")
    
    logger.info(f"Successfully generated {len(samples)} samples")
    return samples


def save_dataset(
    samples: List[str],
    output_file: Path,
    format: str = "json"
) -> None:
    """
    Save dataset to file.
    
    Args:
        samples: List of text samples
        output_file: Path to output file
        format: Output format ("json" or "txt")
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        # Save as JSON array
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(samples)} samples to {output_file} (JSON format)")
    elif format == "txt":
        # Save as text file, one sample per line
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(sample + '\n')
        logger.info(f"Saved {len(samples)} samples to {output_file} (TXT format)")
    else:
        raise ValueError(f"Unknown format: {format}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate random word dataset for DNA extraction"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=600,
        help="Number of samples to generate (default: 600)"
    )
    parser.add_argument(
        "--words-per-sample",
        type=int,
        default=100,
        help="Number of words per sample (default: 100)"
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/rand/rand_dataset.json"),
        help="Output file path (default: data/rand/rand_dataset.json)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "txt"],
        default="json",
        help="Output format: json or txt (default: json)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Generate samples
    samples = generate_random_word_samples(
        num_samples=args.num_samples,
        words_per_sample=args.words_per_sample,
        seed=args.seed
    )
    
    # Save dataset
    save_dataset(samples, args.output_file, args.format)
    
    logger.info("Dataset generation completed successfully!")


if __name__ == "__main__":
    main()

