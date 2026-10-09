"""Acquire the pinned, public timm DINOv3 ViT-S/16 checkpoint.

Only standard-library modules are needed. This script never installs packages,
loads a pickle checkpoint, or accepts access conditions on a gated model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
MODEL_ID = "timm/vit_small_patch16_dinov3.lvd1689m"
REVISION = "3bf4720a82ec2066db88137180ff1f83a675cef0"
FILENAME = "model.safetensors"
SIZE = 86_362_376
SHA256 = "2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040"
URL = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{FILENAME}"
TIMM_VERSION = "1.0.30"
TIMM_COMMIT = "0df212b369a5385b16dfe513d5143a7311ea1ddc"
LICENSE_URL = "https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m/blob/3bf4720a82ec2066db88137180ff1f83a675cef0/LICENSE.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {path}")
    actual_size = path.stat().st_size
    if actual_size != SIZE:
        raise ValueError(f"Size mismatch: {actual_size} bytes, expected {SIZE}")
    actual_hash = sha256(path)
    if actual_hash != SHA256:
        raise ValueError(f"SHA256 mismatch: {actual_hash}, expected {SHA256}")


def download(path: Path) -> None:
    """Retry three times; replace the target only after full verification."""
    partial = path.with_name(path.name + ".partial")
    for attempt in range(1, 4):
        try:
            print(f"Download attempt {attempt}/3: {URL}", flush=True)
            request = Request(URL, headers={"User-Agent": "CPU-Defect-Lab/2.0"})
            with urlopen(request, timeout=45) as response, partial.open("wb") as stream:
                total = 0
                next_progress = 4 * 1024 * 1024
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > SIZE:
                        raise ValueError("Server returned more bytes than the pinned checkpoint")
                    stream.write(chunk)
                    if total >= next_progress:
                        print(f"  {total / 1024 / 1024:.1f}/{SIZE / 1024 / 1024:.1f} MiB", flush=True)
                        next_progress += 4 * 1024 * 1024
            verify(partial)
            partial.replace(path)
            print(f"Verified {SIZE} bytes, SHA256 {SHA256}", flush=True)
            return
        except Exception as error:
            if partial.exists():
                partial.unlink()
            if attempt == 3:
                raise RuntimeError("Checkpoint download failed after three attempts") from error
            print(f"  Attempt failed: {error}. Retrying.", flush=True)
            time.sleep(attempt * 2)


def provenance(path: Path, reused: bool) -> dict:
    return {
        "schema_version": 1,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_file": path.name,
        "checkpoint_format": "safetensors",
        "bytes": SIZE,
        "sha256": SHA256,
        "reused_verified_checkpoint": reused,
        "source": URL,
        "model_id": MODEL_ID,
        "model_revision": REVISION,
        "access": "Public timm repository; anonymous HF API gated=false and HTTP HEAD=200 were verified",
        "backbone": "DINOv3 ViT-S/16 distilled on LVD-1689M",
        "model_repository": "https://github.com/facebookresearch/dinov3",
        "runtime_repository": "https://github.com/huggingface/pytorch-image-models",
        "runtime_version": TIMM_VERSION,
        "runtime_commit": TIMM_COMMIT,
        "license": "DINOv3 License",
        "license_url": LICENSE_URL,
        "method_repository": "https://github.com/Continue-Edge-AI-Lab/Rethinking-Continual-AD",
        "implementation_note": "CPU float32 timm adaptation of the DINOv3 backbone; original Meta weights/API are not downloaded",
        "rope_difference": "timm defaults to constructing float32 RoPE periods; Meta's original implementation stores bfloat16 periods. The project's inference pipeline explicitly truncates periods via bfloat16 then restores float32, following the model-card recommendation. This is still a timm API/runtime adaptation and is not claimed to be bitwise identical to Meta's original runtime.",
        "pipeline_rope_precision_match": "periods.to(torch.bfloat16).to(torch.float32)",
        "pipeline_token_normalization": "Final backbone LayerNorm output; no additional L2 normalization, following the DINOSaur core source",
        "input_contract": {
            "image_shape": [3, 224, 224],
            "patch_size": 16,
            "embedding_dimension": 384,
            "prefix_tokens": 5,
            "patch_tokens": 196,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "frontier")
    parser.add_argument("--verify-only", action="store_true", help="Validate cached weights without downloading")
    args = parser.parse_args()
    cache_dir = args.cache_dir.resolve()
    path = cache_dir / FILENAME
    reused = False
    if path.exists():
        try:
            verify(path)
            reused = True
            print(f"Reusing verified checkpoint: {path}", flush=True)
        except (ValueError, OSError) as error:
            if args.verify_only:
                raise
            print(f"Existing checkpoint is invalid: {error}", flush=True)
    if not reused:
        if args.verify_only:
            verify(path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        download(path)
    metadata_path = cache_dir / "backbone_provenance.json"
    metadata_partial = metadata_path.with_name(metadata_path.name + ".partial")
    metadata_partial.write_text(json.dumps(provenance(path, reused), indent=2) + "\n", encoding="utf-8")
    metadata_partial.replace(metadata_path)
    print(f"Source: {URL}\nProvenance: {metadata_path}\nDINOv3 weights ready: {path}", flush=True)


if __name__ == "__main__":
    main()
