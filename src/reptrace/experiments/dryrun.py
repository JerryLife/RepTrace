#!/usr/bin/env python3
"""
Dry run script that simulates GPU usage for testing parallel execution.

This script mimics the behavior of compute_dna.py but does nothing except:
1. Allocate some GPU memory to simulate real workload
2. Wait for a random duration (20 seconds ± random variation)
3. Log progress and exit with success
"""

import argparse
import logging
import os
import sys
import time
import random
from pathlib import Path
from typing import List, Optional
import torch

def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments (mimicking compute_dna.py interface)."""
    parser = argparse.ArgumentParser(
        description="Dry run simulation of DNA extraction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments (for compatibility)
    parser.add_argument("--model-name", "-m", type=str, required=True, help="Model name")
    parser.add_argument("--dataset", "-d", type=str, default="rand", help="Dataset ID")
    parser.add_argument("--extractor-type", "-e", type=str, default="embedding", help="Extractor type")
    parser.add_argument("--dna-dim", type=int, default=10, help="DNA dimension")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples")
    
    # System arguments
    parser.add_argument("--device", type=str, default="auto", help="Device for computation")
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("./out"), help="Output directory")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Logging level")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    
    # Dry run specific arguments
    parser.add_argument("--min-duration", type=int, default=15, help="Minimum duration in seconds")
    parser.add_argument("--max-duration", type=int, default=25, help="Maximum duration in seconds") 
    parser.add_argument("--gpu-memory-mb", type=int, default=500, help="GPU memory to allocate in MB")
    
    return parser.parse_args(argv)

def setup_logging(level: str):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def get_device(device_arg: str) -> str:
    """Get the actual device to use."""
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg

def allocate_gpu_memory(device: str, memory_mb: int) -> torch.Tensor:
    """Allocate GPU memory to simulate real workload."""
    if not device.startswith("cuda"):
        logging.info("CPU device specified, no GPU memory allocation")
        return None
    
    if not torch.cuda.is_available():
        logging.warning("CUDA not available, cannot allocate GPU memory")
        return None
    
    try:
        # Calculate tensor size for requested memory
        # Each float32 element is 4 bytes
        elements_needed = (memory_mb * 1024 * 1024) // 4
        tensor_size = int(elements_needed ** 0.5)  # Square tensor
        
        device_obj = torch.device(device)
        logging.info(f"Allocating ~{memory_mb}MB GPU memory on {device}")
        
        # Allocate tensor and perform some operations to ensure memory is used
        tensor = torch.randn(tensor_size, tensor_size, device=device_obj, dtype=torch.float32)
        
        # Do some computation to make sure memory is actually allocated
        tensor = tensor @ tensor.T
        tensor = torch.relu(tensor)
        
        actual_mb = (tensor.numel() * 4) // (1024 * 1024)
        logging.info(f"Successfully allocated {actual_mb}MB GPU memory on {device}")
        
        return tensor
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logging.warning(f"CUDA OOM when trying to allocate {memory_mb}MB: {e}")
            # Try with half the memory
            return allocate_gpu_memory(device, memory_mb // 2)
        else:
            logging.error(f"Failed to allocate GPU memory: {e}")
            return None
    except Exception as e:
        logging.error(f"Unexpected error allocating GPU memory: {e}")
        return None

def simulate_work(duration_seconds: int, model_name: str, dataset: str, device: str):
    """Simulate DNA extraction work with progress updates."""
    logging.info(f"Starting dry run DNA extraction simulation")
    logging.info(f"Model: {model_name}")
    logging.info(f"Dataset: {dataset}")
    logging.info(f"Device: {device}")
    logging.info(f"Duration: {duration_seconds} seconds")
    
    start_time = time.time()
    
    # Simulate different phases of DNA extraction
    phases = [
        ("Loading model", 0.1),
        ("Processing probes", 0.6), 
        ("Extracting DNA signature", 0.2),
        ("Saving results", 0.1)
    ]
    
    for phase_name, phase_fraction in phases:
        phase_duration = duration_seconds * phase_fraction
        logging.info(f"Phase: {phase_name} (duration: {phase_duration:.1f}s)")
        
        # Simulate progress within phase
        steps = 5
        step_duration = phase_duration / steps
        
        for step in range(steps):
            time.sleep(step_duration)
            progress = ((step + 1) / steps) * 100
            logging.info(f"  {phase_name} progress: {progress:.0f}%")
    
    total_time = time.time() - start_time
    logging.info(f"Dry run completed in {total_time:.2f}s")

def create_dummy_output(args: argparse.Namespace):
    """Create dummy output files to simulate real DNA extraction."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy DNA signature file
    safe_model_name = args.model_name.replace("/", "_").replace(":", "_")
    output_filename = f"{safe_model_name}_{args.dataset}_{args.extractor_type}_dna_DRYRUN.json"
    output_path = args.output_dir / output_filename
    
    dummy_data = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "extractor_type": args.extractor_type,
        "dna_dimension": args.dna_dim,
        "dry_run": True,
        "timestamp": time.time(),
        "dna_signature": [0.0] * args.dna_dim,
        "metadata": {
            "extraction_method": f"{args.extractor_type}_dry_run",
            "probe_count": args.max_samples,
            "device": args.device
        }
    }
    
    import json
    with open(output_path, 'w') as f:
        json.dump(dummy_data, f, indent=2)
    
    logging.info(f"Dummy output saved to: {output_path}")
    
    # Create dummy summary
    summary_path = args.output_dir / f"{safe_model_name}_{args.dataset}_summary_DRYRUN.json"
    summary_data = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "extractor_type": args.extractor_type,
        "dry_run": True,
        "output_file": str(output_path),
        "success": True
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    logging.info(f"Dummy summary saved to: {summary_path}")

def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Set random seed for reproducible duration
    random.seed(args.random_seed + hash(args.model_name) % 10000)
    
    # Calculate random duration
    duration = random.randint(args.min_duration, args.max_duration)
    
    logging.info("=" * 50)
    logging.info("DRY RUN - DNA EXTRACTION SIMULATION")
    logging.info("=" * 50)
    
    try:
        # Get device
        device = get_device(args.device)
        logging.info(f"Using device: {device}")
        
        # Allocate GPU memory to simulate real workload
        memory_tensor = allocate_gpu_memory(device, args.gpu_memory_mb)
        
        # Simulate the work
        simulate_work(duration, args.model_name, args.dataset, device)
        
        # Create dummy output files
        create_dummy_output(args)
        
        # Print final statistics (mimicking compute_dna.py)
        print(f"\nDry Run DNA Signature Statistics:")
        print(f"  Model: {args.model_name}")
        print(f"  Dataset: {args.dataset}")
        print(f"  Extractor: {args.extractor_type}")
        print(f"  Dimension: {args.dna_dim}")
        print(f"  Probes: {args.max_samples}")
        print(f"  Device: {device}")
        print(f"  Duration: {duration}s")
        print(f"  Status: SUCCESS (DRY RUN)")
        
        # Clean up GPU memory
        if memory_tensor is not None:
            del memory_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info("Released GPU memory")
        
        logging.info("Dry run completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logging.info("Dry run interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Dry run failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
