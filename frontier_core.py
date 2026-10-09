"""CPU memory-bank operations for the DINOSaur reproduction.

Algorithm reference: Weatherly and Lin, arXiv:2605.24251v2 (ECCV 2026),
https://arxiv.org/abs/2605.24251, and the official implementation at
Continue-Edge-AI-Lab/Rethinking-Continual-AD, commit
9574f14f2e5a99e605ed19f0ff78f0a496d29252, Methods/DINO/DINOSaur.py.

This independent implementation uses the backbone's final LayerNorm features
without additional L2 normalization. It bounds temporary retrieval tensors
and caps the original minimum of 20 coreset points at the number of images.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch


def _floating_tensor(value: torch.Tensor, name: str, ndim: int) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}-dimensional torch tensor")
    if not value.is_floating_point():
        raise ValueError(f"{name} must contain floating point features")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} contains non-finite features")


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


@torch.inference_mode()
def build_spatial_bank(
    features: torch.Tensor,
    rho: float = 0.1,
    seed: int = 42,
    sampler: str = "kcenter",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return spatial coresets and their source-image indices.

    ``features`` has shape ``(N, P*P, D)``. Returned shapes are
    ``(P, P, M, D)`` and ``(P*P, M)`` respectively, where
    ``M = min(N, max(20, int(N*rho)))``. Every location is sampled
    independently. K-center starts from one seeded random point and chooses
    the farthest unselected point; repeated vectors cannot cause repeated
    selections or an infinite loop. Indices follow selection order.
    """
    _floating_tensor(features, "features", 3)
    n_images, n_positions, n_dims = features.shape
    if min(n_images, n_positions, n_dims) < 1:
        raise ValueError("features dimensions must be nonempty")
    side = math.isqrt(n_positions)
    if side * side != n_positions:
        raise ValueError("features must have a square spatial grid")
    if not math.isfinite(rho) or not 0.0 < rho <= 1.0:
        raise ValueError("rho must be finite and in (0, 1]")
    if sampler not in {"kcenter", "random"}:
        raise ValueError("sampler must be 'kcenter' or 'random'")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    n_selected = min(n_images, max(20, int(n_images * rho)))
    generator = torch.Generator(device=features.device).manual_seed(seed)
    points = features.transpose(0, 1).contiguous()
    if sampler == "random":
        # Random ranks give a uniform permutation independently per position.
        ranks = torch.rand(
            (n_positions, n_images), device=features.device, generator=generator
        )
        indices = ranks.argsort(dim=1, stable=True)[:, :n_selected]
    else:
        indices = torch.empty(
            (n_positions, n_selected), dtype=torch.long, device=features.device
        )
        starts = torch.randint(
            n_images, (n_positions,), device=features.device, generator=generator
        )
        # Position blocks bound the temporary N*D difference tensor during fit.
        for begin in range(0, n_positions, 32):
            end = min(begin + 32, n_positions)
            block = points[begin:end]
            rows = torch.arange(end - begin, device=features.device)
            current = starts[begin:end]
            chosen = torch.zeros(
                (end - begin, n_images), dtype=torch.bool, device=features.device
            )
            minimum = torch.full(
                (end - begin, n_images), float("inf"),
                dtype=features.dtype, device=features.device,
            )
            for step in range(n_selected):
                indices[begin:end, step] = current
                chosen[rows, current] = True
                if step + 1 == n_selected:
                    break
                distance = torch.linalg.vector_norm(
                    block - block[rows, current].unsqueeze(1), dim=-1
                )
                minimum = torch.minimum(minimum, distance)
                current = minimum.masked_fill(chosen, -float("inf")).argmax(dim=1)

    selected = points.gather(
        1, indices.unsqueeze(-1).expand(-1, -1, n_dims)
    )
    return selected.reshape(side, side, n_selected, n_dims), indices


def _check_queries_bank(patches: torch.Tensor, bank: torch.Tensor) -> int:
    _floating_tensor(patches, "patches", 2)
    _floating_tensor(bank, "bank", 4)
    side, width, count, n_dims = bank.shape
    if min(side, width, count, n_dims) < 1 or side != width:
        raise ValueError("bank must be a nonempty square spatial bank")
    if patches.shape != (side * side, n_dims):
        raise ValueError("patches must match bank's spatial grid and feature dimension")
    if patches.dtype != bank.dtype or patches.device != bank.device:
        raise ValueError("patches and bank must have the same dtype and device")
    return side


@torch.inference_mode()
def spatial_distances(
    patches: torch.Tensor,
    bank: torch.Tensor,
    radius: int = 3,
    chunk: int = 8,
) -> torch.Tensor:
    """Exact nearest Euclidean distance inside each patch's clipped window.

    Grid boundaries are excluded instead of represented by zero features.
    Retrieval handles a bounded number of query positions at once rather
    than materializing every query's complete neighborhood simultaneously.
    """
    side = _check_queries_bank(patches, bank)
    _positive_integer(chunk, "chunk")
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("radius must be a nonnegative integer")
    radius = min(radius, side - 1)
    offsets = torch.arange(-radius, radius + 1, device=bank.device)
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    offsets_y, offsets_x = dy.flatten(), dx.flatten()
    flat_bank = bank.reshape(side * side, bank.shape[2], bank.shape[3])
    result = torch.empty(side * side, dtype=patches.dtype, device=patches.device)
    for begin in range(0, side * side, chunk):
        end = min(begin + chunk, side * side)
        locations = torch.arange(begin, end, device=bank.device)
        y = locations.div(side, rounding_mode="floor").unsqueeze(1) + offsets_y
        x = locations.remainder(side).unsqueeze(1) + offsets_x
        valid = (y >= 0) & (y < side) & (x >= 0) & (x < side)
        neighbors = y.clamp(0, side - 1) * side + x.clamp(0, side - 1)
        references = flat_bank[neighbors]
        distances = torch.linalg.vector_norm(
            patches[begin:end, None, None, :] - references, dim=-1
        )
        distances.masked_fill_(~valid.unsqueeze(-1), float("inf"))
        result[begin:end] = distances.flatten(1).min(dim=1).values
    return result


@torch.inference_mode()
def unrestricted_distances(
    patches: torch.Tensor, bank: torch.Tensor, chunk: int = 64
) -> torch.Tensor:
    """Exact global nearest-neighbor scores using the same spatial coreset.

    Both query and reference blocks are bounded. Direct Euclidean norms
    preserve zero self-distance and avoid dot-product cancellation errors.
    This forms the unrestricted-retrieval ablation, not a different bank.
    """
    _check_queries_bank(patches, bank)
    _positive_integer(chunk, "chunk")
    references = bank.reshape(-1, bank.shape[-1])
    result = torch.empty(patches.shape[0], dtype=patches.dtype, device=patches.device)
    budget = 32 * 1024 * 1024
    for begin in range(0, patches.shape[0], chunk):
        end = min(begin + chunk, patches.shape[0])
        queries = patches[begin:end]
        minimum = torch.full(
            (end - begin,), float("inf"), dtype=patches.dtype, device=patches.device
        )
        ref_chunk = max(
            1, budget // ((end - begin) * patches.shape[1] * patches.element_size())
        )
        for ref_begin in range(0, references.shape[0], ref_chunk):
            distance = torch.linalg.vector_norm(
                queries[:, None, :] - references[None, ref_begin:ref_begin + ref_chunk, :],
                dim=-1,
            ).min(dim=1).values
            minimum = torch.minimum(minimum, distance)
        result[begin:end] = minimum
    return result


@torch.inference_mode()
def choose_task(cls: torch.Tensor, prototypes: Mapping[str, torch.Tensor]) -> str:
    """Route by raw Euclidean CLS distance; insertion order resolves ties."""
    _floating_tensor(cls, "cls", 1)
    if not prototypes:
        raise ValueError("at least one task prototype is required")
    names = list(prototypes)
    for name, prototype in prototypes.items():
        _floating_tensor(prototype, f"prototype {name!r}", 1)
        if prototype.shape != cls.shape:
            raise ValueError("task prototype must match CLS feature dimension")
        if prototype.device != cls.device or prototype.dtype != cls.dtype:
            raise ValueError("task prototypes and CLS must have the same dtype and device")
    values = torch.stack([prototypes[name] for name in names])
    nearest = torch.linalg.vector_norm(values - cls.unsqueeze(0), dim=1).argmin().item()
    return names[nearest]
