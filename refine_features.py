"""CPU-only DINOv3 features for predeclared refinement experiments.

This module leaves the released 224-pixel pipeline unchanged. Custom image
preprocessing must supply an explicit, serializable cache identity. Features
retain the final LayerNorm and have no additional L2 normalization.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
import uuid
import weakref
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image

from frontier_core import _check_queries_bank, _positive_integer
from frontier_setup import SHA256 as WEIGHTS_SHA

ROOT = Path(__file__).resolve().parent
FEATURE_CONTRACT = "tensor-bilinear-final-LN-no-L2-rope-bf16-v1"
_GEMM_CACHE = OrderedDict()
_GEMM_CACHE_LIMIT = 3


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def transform_image(image: Image.Image, size: int = 224) -> torch.Tensor:
    from torchvision.transforms import v2

    pipeline = v2.Compose([
        v2.ToImage(), v2.Resize((size, size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    return pipeline(image.convert("RGB"))


class DinoExtractor:
    """Frozen 224/336 DINOv3, with a five-value legacy-compatible extract API."""

    def __init__(self, size=224, weights=None, cache_dir=None, threads=1):
        import timm
        import torchvision
        from safetensors.torch import load_file

        if isinstance(size, bool) or size not in (224, 336):
            raise ValueError("size must be 224 or 336")
        _positive_integer(threads, "threads")
        self.size = size
        self.side = size // 16
        self.threads = threads
        self.weights = Path(weights) if weights else ROOT / "cache/frontier/model.safetensors"
        self.cache_dir = Path(cache_dir) if cache_dir else ROOT / "cache/refinement"
        if not self.weights.is_file() or file_sha256(self.weights) != WEIGHTS_SHA:
            raise ValueError("Run frontier_setup.py to obtain the verified DINOv3 checkpoint")
        torch.set_num_threads(threads)
        self.net = timm.create_model(
            "vit_small_patch16_dinov3", pretrained=False, img_size=size,
        ).eval().cpu()
        self.net.load_state_dict(load_file(str(self.weights), device="cpu"), strict=True)
        self.net.rope.periods = self.net.rope.periods.to(torch.bfloat16).to(torch.float32)
        self.net.requires_grad_(False)
        if isinstance(self.net.norm, torch.nn.Identity):
            if not isinstance(self.net.fc_norm, torch.nn.LayerNorm):
                raise ValueError("Expected final LayerNorm in timm fc_norm")
            self.final_token_norm = self.net.fc_norm
        else:
            self.final_token_norm = torch.nn.Identity()
        if self.net.num_prefix_tokens != 5:
            raise ValueError("Expected one CLS and four register tokens")
        self.versions = {
            "torch": str(torch.__version__), "timm": str(timm.__version__),
            "torchvision": str(torchvision.__version__),
        }

    @torch.inference_mode()
    def forward_batch(self, tensor):
        if tensor.ndim != 4 or tuple(tensor.shape[1:]) != (3, self.size, self.size):
            raise ValueError(f"Expected RGB tensor (B,3,{self.size},{self.size})")
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all().item():
            raise ValueError("Preprocessed tensor must be finite floating point")
        tokens = self.final_token_norm(self.net.forward_features(tensor.float().cpu()))
        if tuple(tokens.shape[1:]) != (5 + self.side ** 2, 384):
            raise ValueError(f"Unexpected DINOv3 token shape: {tuple(tokens.shape)}")
        return tokens[:, 0].contiguous(), tokens[:, 5:].contiguous()

    def _cache_identity(self, paths, batch, preprocessing):
        manifest = []
        for path in paths:
            resolved = path.resolve()
            try:
                name = resolved.relative_to(ROOT).as_posix()
            except ValueError:
                name = resolved.as_posix()
            manifest.append({"path": name, "sha256": file_sha256(resolved)})
        identity = {
            "weights_sha256": WEIGHTS_SHA, "versions": self.versions,
            "size": self.size, "feature_contract": FEATURE_CONTRACT,
            "batch": batch, "threads": torch.get_num_threads(),
            "preprocessing": preprocessing, "files": manifest,
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]

    def extract(self, paths, batch=2, preprocess=None, identity=None):
        """Return ``cls, patches, original_mean_ms, cache_key, cache_hit``.

        ``preprocess`` receives a PIL image and must return a floating point
        RGB tensor of this extractor's configured size. ``identity`` is its
        stable JSON-serializable configuration, including algorithm version.
        The timing includes image decoding/preprocessing and model forward;
        it excludes content hashing, checkpoint loading and cache writing.
        """
        _positive_integer(batch, "batch")
        paths = [Path(path) for path in paths]
        if not paths:
            raise ValueError("At least one input image is required")
        if preprocess is not None and identity is None:
            raise ValueError("Custom preprocessing requires an explicit identity")
        if preprocess is None:
            if identity is not None:
                raise ValueError("identity is only accepted with custom preprocessing")
            preprocessing = {"kind": "RGB-tensor-bilinear-antialias-imagenet", "size": self.size}
            preprocess = lambda image: transform_image(image, self.size)
        else:
            if not callable(preprocess):
                raise ValueError("preprocess must be callable")
            preprocessing = identity
        encoded, key = self._cache_identity(paths, batch, preprocessing)
        cache = self.cache_dir / f"features-{key}.pt"
        expected = ((len(paths), 384), (len(paths), self.side ** 2, 384))
        if cache.is_file():
            saved = torch.load(cache, map_location="cpu", weights_only=True)
            if (saved.get("identity_json") != encoded
                    or tuple(saved["cls"].shape) != expected[0]
                    or tuple(saved["patches"].shape) != expected[1]
                    or saved["cls"].dtype != torch.float32
                    or saved["patches"].dtype != torch.float32
                    or not torch.isfinite(saved["cls"]).all().item()
                    or not torch.isfinite(saved["patches"]).all().item()):
                raise ValueError(f"Feature cache does not match its identity: {cache.name}")
            return saved["cls"], saved["patches"], saved["extraction_mean_ms"], key, True
        cls, patches, seconds = [], [], 0.0
        for start in range(0, len(paths), batch):
            before = time.perf_counter()
            values = []
            for path in paths[start:start + batch]:
                with Image.open(path) as image:
                    values.append(preprocess(image))
            c, p = self.forward_batch(torch.stack(values))
            seconds += time.perf_counter() - before
            cls.append(c); patches.append(p)
            if start == 0 or (start // batch) % 20 == 0 or start + batch >= len(paths):
                print(f"DINOv3 {self.size}px features: {min(start + batch, len(paths))}/{len(paths)}", flush=True)
        cls, patches = torch.cat(cls), torch.cat(patches)
        mean_ms = seconds * 1000 / len(paths)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(cache.name + f".{os.getpid()}-{uuid.uuid4().hex}.partial")
        try:
            torch.save({"cls": cls, "patches": patches,
                        "extraction_mean_ms": mean_ms, "identity_json": encoded}, temporary)
            temporary.replace(cache)
        finally:
            temporary.unlink(missing_ok=True)
        return cls, patches, mean_ms, key, False


@torch.inference_mode()
def spatial_distances_f64(patches, bank, radius=3, chunk=8):
    """Optional CPU nearest retrieval using float64 inner products.

    Candidate ordering uses squared Euclidean distances in float64, avoiding
    float32 cancellation. The nearest selected reference is then evaluated
    with a direct norm in the input dtype, preserving exact zero self-distance.
    Neighbor indices outside the image are masked, rather than zero padded.
    Temporary reference tensors remain bounded by ``chunk``.
    """
    side = _check_queries_bank(patches, bank)
    _positive_integer(chunk, "chunk")
    if patches.device.type != "cpu":
        raise ValueError("This optional implementation is CPU-only")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a nonnegative integer")
    radius = min(radius, side - 1)
    count, dims = bank.shape[2:]
    flat = bank.reshape(side * side, count, dims)
    flat64 = flat.double()
    norms = flat64.square().sum(-1)
    query64 = patches.double()
    query_norms = query64.square().sum(-1)
    offset = torch.arange(-radius, radius + 1)
    dy, dx = torch.meshgrid(offset, offset, indexing="ij")
    result = torch.empty(side * side, dtype=patches.dtype)
    for start in range(0, side * side, chunk):
        end = min(start + chunk, side * side)
        locations = torch.arange(start, end)
        y = locations.div(side, rounding_mode="floor")[:, None] + dy.flatten()
        x = locations.remainder(side)[:, None] + dx.flatten()
        valid = (y >= 0) & (y < side) & (x >= 0) & (x < side)
        neighbor = y.clamp(0, side - 1) * side + x.clamp(0, side - 1)
        refs = flat64[neighbor].reshape(end - start, -1, dims)
        squared = norms[neighbor].reshape(end - start, -1) + query_norms[start:end, None]
        squared.add_(torch.bmm(refs, query64[start:end, :, None]).squeeze(-1), alpha=-2)
        squared.view(end - start, -1, count).masked_fill_(~valid[:, :, None], float("inf"))
        winner = squared.argmin(dim=1)
        selected_neighbor = neighbor.gather(1, winner.div(count, rounding_mode="floor")[:, None]).squeeze(1)
        references = flat[selected_neighbor, winner.remainder(count)]
        result[start:end] = torch.linalg.vector_norm(patches[start:end] - references, dim=-1)
    return result


def clear_gemm_cache():
    """Release all optional prepared-bank retrieval caches."""
    _GEMM_CACHE.clear()


def _prepare_gemm_bank(bank, side, radius, use_cache):
    # Inference tensors deliberately have no mutation version counter. They
    # are never cached: object identity alone cannot detect in-place changes.
    try:
        version = bank._version
    except RuntimeError:
        version = None
    key = (id(bank), version, tuple(bank.shape), bank.dtype, bank.device, radius)
    can_cache = bool(use_cache and version is not None)
    if can_cache:
        saved = _GEMM_CACHE.get(key)
        if saved is not None and saved[0]() is bank:
            _GEMM_CACHE.move_to_end(key)
            return saved[1]
    references = bank.reshape(-1, bank.shape[-1]).double().contiguous()
    norms = torch.einsum("ij,ij->i", references, references)
    locations = torch.arange(side * side)
    y = locations.div(side, rounding_mode="floor")
    x = locations.remainder(side)
    allowed = ((y[:, None] - y[None, :]).abs() <= radius)
    allowed &= ((x[:, None] - x[None, :]).abs() <= radius)
    prepared = (references, norms, allowed)
    if can_cache:
        _GEMM_CACHE[key] = (weakref.ref(bank), prepared)
        _GEMM_CACHE.move_to_end(key)
        while len(_GEMM_CACHE) > _GEMM_CACHE_LIMIT:
            _GEMM_CACHE.popitem(last=False)
    return prepared


@torch.inference_mode()
def spatial_distances_gemm(patches, bank, radius=3, memory_mib=32, use_cache=True):
    """CPU spatial nearest retrieval using large float64 matrix products.

    Each query block compares against the flattened bank with GEMM. A spatial
    mask discards forbidden references before argmin. Selected candidates are
    scored by direct input-dtype norms, including exact zero self-distance.

    ``memory_mib`` bounds the squared-distance output, not process peak memory.
    Float64 bank features and norms, plus the small P-by-P spatial mask, can be
    reused by at most three prepared-bank cache entries. Cache reuse requires
    the same tensor object and mutation version. Inference tensors have no
    version counter and therefore bypass the cache even when use_cache=True.
    """
    side = _check_queries_bank(patches, bank)
    if patches.device.type != "cpu":
        raise ValueError("This optional implementation is CPU-only")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a nonnegative integer")
    if isinstance(memory_mib, bool) or not isinstance(memory_mib, (int, float)):
        raise ValueError("memory_mib must be positive and finite")
    if not math.isfinite(memory_mib) or memory_mib <= 0:
        raise ValueError("memory_mib must be positive and finite")
    radius = min(radius, side - 1)
    references, norms, allowed = _prepare_gemm_bank(bank, side, radius, use_cache)
    flat = bank.reshape(-1, bank.shape[-1])
    count = bank.shape[2]
    rows = int(memory_mib * 1024 ** 2) // (references.shape[0] * references.element_size())
    if rows < 1:
        raise ValueError("memory_mib cannot hold one squared-distance query row")
    rows = min(rows, patches.shape[0])
    query64 = patches.double()
    query_norms = torch.einsum("ij,ij->i", query64, query64)
    result = torch.empty(patches.shape[0], dtype=patches.dtype)
    for start in range(0, patches.shape[0], rows):
        end = min(start + rows, patches.shape[0])
        squared = torch.mm(query64[start:end], references.T)
        squared.mul_(-2)
        squared.add_(norms[None, :])
        squared.add_(query_norms[start:end, None])
        squared.view(end - start, side * side, count).masked_fill_(
            ~allowed[start:end, :, None], float("inf"),
        )
        winner = squared.argmin(dim=1)
        result[start:end] = torch.linalg.vector_norm(patches[start:end] - flat[winner], dim=-1)
    return result
