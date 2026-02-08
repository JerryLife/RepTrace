"""Public API for programmatic DNA extraction."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

if TYPE_CHECKING:
    from .dna.DNASignature import DNASignature


@dataclass(slots=True)
class DNAExtractionConfig:
    """Configuration for extracting one model DNA vector."""

    model_name: str
    model_path: Optional[str] = None
    model_type: str = "auto"
    dataset: str = "rand"
    probe_set: str = "general"
    max_samples: int = 100
    data_root: str = "./data"
    extractor_type: str = "embedding"
    dna_dim: int = 128
    reduction_method: str = "random_projection"
    embedding_merge: str = "concat"
    max_length: int = 1024
    output_dir: Path = Path("./out")
    output_path: Optional[Path] = None
    save: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    no_quantization: bool = False
    metadata_file: Optional[Path] = Path("./configs/llm_metadata.json")
    token: Optional[str] = None
    trust_remote_code: bool = True
    device: str = "auto"
    gpu_id: Optional[int] = None
    log_level: str = "INFO"
    random_seed: int = 42
    skip_chat_template: bool = False


@dataclass(slots=True)
class DNAExtractionResult:
    """Result payload for a DNA extraction run."""

    model_name: str
    dataset: str
    vector: np.ndarray
    signature: "DNASignature"
    output_path: Optional[Path]
    summary_path: Optional[Path]
    elapsed_seconds: float


def _resolve_hf_token(explicit_token: Optional[str]) -> Optional[str]:
    if explicit_token:
        return explicit_token

    env_vars = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN")
    for env_var in env_vars:
        value = os.getenv(env_var, "").strip()
        if value:
            return value
    return None


def _default_model_metadata(model_name: str) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "architecture": {"is_generative": True},
        "repository": {},
    }


def _load_model_metadata_for_model(
    model_name: str,
    metadata_file: Optional[Path],
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Load model metadata from file or fetch from HuggingFace Hub."""
    # Try to load from metadata file first
    if metadata_file is not None:
        try:
            from .core import extraction as core
            all_metadata = core.load_model_metadata(metadata_file)
            model_meta = all_metadata.get(model_name)
            if model_meta:
                return model_meta
        except Exception as exc:
            logging.debug("Failed to load metadata file %s: %s", metadata_file, exc)

    # Fetch metadata from HuggingFace Hub (with caching)
    logging.info("Fetching metadata for '%s' from HuggingFace Hub...", model_name)
    try:
        from .utils.metadata import get_model_metadata
        return get_model_metadata(model_name, token=token)
    except Exception as exc:
        logging.warning("Failed to fetch metadata for '%s': %s", model_name, exc)
        return {
            "model_name": model_name,
            "architecture": {"is_generative": True},
            "repository": {},
        }


def _resolve_model_path(model_path: Optional[str], model_meta: Dict[str, Any]) -> Optional[str]:
    if model_path:
        return model_path

    repository = model_meta.get("repository", {})
    local_path = repository.get("local_path")
    if local_path and Path(local_path).exists():
        return local_path

    model_id = repository.get("model_id")
    if model_id and Path(model_id).exists():
        return model_id

    return None


def _resolve_device(config: DNAExtractionConfig) -> str:
    from .core import extraction as core

    if config.gpu_id is not None:
        return core.validate_device_argument(f"cuda:{int(config.gpu_id)}")
    return core.validate_device_argument(config.device)


def _validate_quantization(config: DNAExtractionConfig) -> None:
    quant_flags = [config.load_in_4bit, config.load_in_8bit, config.no_quantization]
    if sum(quant_flags) > 1:
        raise ValueError(
            "Use only one of load_in_4bit, load_in_8bit, or no_quantization."
        )

    if config.load_in_4bit or config.load_in_8bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise ValueError(
                "Quantization requires bitsandbytes. Install with `pip install bitsandbytes`."
            ) from exc


def _signature_output_paths(config: DNAExtractionConfig) -> tuple[Path, Path]:
    if config.output_path is not None:
        output_path = Path(config.output_path)
        summary_path = output_path.with_name(f"{output_path.stem}_summary.json")
        return output_path, summary_path

    safe_model_name = config.model_name.replace("/", "_").replace(":", "_")
    dataset_identifier = config.dataset.replace(",", "_")
    structured_dir = Path(config.output_dir) / dataset_identifier / safe_model_name
    output_path = structured_dir / f"{safe_model_name}_dna.json"
    summary_path = structured_dir / f"{safe_model_name}_summary.json"
    return output_path, summary_path


def _save_signature_outputs(
    signature: "DNASignature",
    config: DNAExtractionConfig,
    output_path: Path,
    summary_path: Path,
    elapsed_seconds: float,
) -> None:
    from .core import extraction as core

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    signature.save(output_path, format="json")

    config_dump = asdict(config)
    config_dump["output_dir"] = str(config.output_dir)
    if config.output_path is not None:
        config_dump["output_path"] = str(config.output_path)
    if config.metadata_file is not None:
        config_dump["metadata_file"] = str(config.metadata_file)

    summary = {
        "model_name": config.model_name,
        "dataset": config.dataset,
        "dataset_full": core.get_dataset_name(config.dataset),
        "extractor_type": config.extractor_type,
        "dna_dimension": config.dna_dim,
        "reduction_method": config.reduction_method,
        "num_probes": signature.metadata.probe_count,
        "total_time_seconds": elapsed_seconds,
        "signature_stats": signature.get_statistics(),
        "metadata": signature.metadata.__dict__,
        "output_file": str(output_path),
        "config": config_dump,
    }

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str, ensure_ascii=False, allow_nan=False)


def _validate_signature(signature: "DNASignature") -> np.ndarray:
    from .dna.DNASignature import DNASignature

    if not isinstance(signature, DNASignature):
        raise TypeError(f"Expected DNASignature, got {type(signature)}")

    vector = np.asarray(signature.signature, dtype=np.float32)
    if vector.size == 0:
        raise ValueError("Empty DNA signature detected.")
    if np.allclose(vector, 0.0):
        raise ValueError("All-zero DNA signature detected.")
    return vector


def calc_dna(config: DNAExtractionConfig) -> DNAExtractionResult:
    """Compute DNA vector for one model and optionally persist outputs."""

    from .core import extraction as core
    from .utils.DataUtils import setup_logging

    setup_logging(level=config.log_level)
    _validate_quantization(config)

    metadata_file = Path(config.metadata_file) if config.metadata_file is not None else None
    resolved_device = _resolve_device(config)
    resolved_token = _resolve_hf_token(config.token)
    model_meta = _load_model_metadata_for_model(config.model_name, metadata_file, token=resolved_token)

    is_generative = model_meta.get("architecture", {}).get("is_generative")
    if is_generative is False:
        arch_type = model_meta.get("architecture", {}).get("type")
        raise ValueError(
            f"Model '{config.model_name}' is non-generative (architecture={arch_type})."
        )

    resolved_model_path = _resolve_model_path(config.model_path, model_meta)

    args = SimpleNamespace(
        model_name=config.model_name,
        model_path=resolved_model_path,
        model_type=config.model_type,
        dataset=config.dataset,
        probe_set=config.probe_set,
        max_samples=config.max_samples,
        data_root=config.data_root,
        extractor_type=config.extractor_type,
        dna_dim=config.dna_dim,
        reduction_method=config.reduction_method,
        embedding_merge=config.embedding_merge,
        max_length=config.max_length,
        save_format="json",
        output_dir=Path(config.output_dir),
        load_in_8bit=config.load_in_8bit,
        load_in_4bit=config.load_in_4bit,
        no_quantization=config.no_quantization,
        metadata_file=metadata_file,
        token=resolved_token,
        trust_remote_code=config.trust_remote_code,
        device=resolved_device,
        log_level=config.log_level,
        random_seed=config.random_seed,
        skip_chat_template=config.skip_chat_template,
    )

    start_time = time.time()
    probe_texts = core.get_probe_texts(
        dataset_id=config.dataset,
        probe_set=config.probe_set,
        max_samples=config.max_samples,
        data_root=config.data_root,
        random_seed=config.random_seed,
    )

    signature = core.extract_dna_signature(
        model_name=config.model_name,
        model_path=resolved_model_path,
        model_type=config.model_type,
        probe_texts=probe_texts,
        extractor_type=config.extractor_type,
        model_metadata=model_meta,
        args=args,
    )

    vector = _validate_signature(signature)
    elapsed_seconds = time.time() - start_time

    output_path: Optional[Path] = None
    summary_path: Optional[Path] = None
    if config.save:
        output_path, summary_path = _signature_output_paths(config)
        _save_signature_outputs(
            signature=signature,
            config=config,
            output_path=output_path,
            summary_path=summary_path,
            elapsed_seconds=elapsed_seconds,
        )

    return DNAExtractionResult(
        model_name=config.model_name,
        dataset=config.dataset,
        vector=vector,
        signature=signature,
        output_path=output_path,
        summary_path=summary_path,
        elapsed_seconds=elapsed_seconds,
    )


def calc_dna_batch(
    configs: list[DNAExtractionConfig],
    gpu_ids: Optional[list[int]] = None,
    continue_on_error: bool = False,
) -> list[DNAExtractionResult]:
    """Compute DNA vectors for multiple models with optional GPU round-robin."""

    results: list[DNAExtractionResult] = []
    failures = 0
    for index, config in enumerate(configs):
        run_config = config
        if config.gpu_id is None and gpu_ids:
            run_config = replace(config, gpu_id=gpu_ids[index % len(gpu_ids)])

        try:
            results.append(calc_dna(run_config))
        except Exception:
            failures += 1
            if not continue_on_error:
                raise

    if failures > 0 and not continue_on_error:
        raise RuntimeError("One or more DNA extraction runs failed.")
    return results


__all__ = ["DNAExtractionConfig", "DNAExtractionResult", "calc_dna", "calc_dna_batch"]
