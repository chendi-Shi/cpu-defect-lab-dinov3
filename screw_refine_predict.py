"""Inference for the frozen CPU screw-refinement experiment.

Example:
    python screw_refine_predict.py --image photo.png --output-dir prediction

The default model is release/screw_refinement/model.pt. Development models can
be selected with --model-path outputs/screw_refinement/<candidate>/model.pt.
Anomaly scores and heatmap colors are differences from a normal feature bank,
not defect probabilities. Loading a model does not change its calibration.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from frontier_setup import SHA256 as WEIGHTS_SHA
from refine_features import DinoExtractor, file_sha256, spatial_distances_gemm, transform_image
from screw_geometry import GEOMETRY_VERSION, alignment_transform, inverse_heatmap

ROOT = Path(__file__).resolve().parent
REQUIRED = {
    "raw224": ("raw224",),
    "raw336": ("raw336",),
    "aligned224": ("raw224", "aligned224"),
    "multi224_336": ("raw224", "raw336"),
}
VIEW_SIZE = {"raw224": 224, "aligned224": 224, "raw336": 336}
VIEW_RADIUS = {"raw224": 3, "aligned224": 3, "raw336": 4}
HEATMAP_NOTE = (
    "Anomaly scores are median-normalized feature distances, not probabilities. "
    "Heatmap colors are scaled separately for each image and cannot be compared "
    "across images. Single-view scores use the maximum patch distance, so an "
    "interpolated heatmap maximum can differ. Aligned heatmaps are mapped back into original 224 coordinates; "
    "inverse interpolation can lower their maximum without changing the image score."
)


def _finite_number(value, label, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return float(value)


def validate_model(saved):
    """Validate the saved runner format without loading the vision backbone."""
    if not isinstance(saved, dict):
        raise ValueError("Expected a saved screw-refinement model dictionary")
    for name in ("candidate", "banks", "medians", "threshold", "config"):
        if name not in saved:
            raise ValueError(f"Missing model field: {name}")
    candidate = saved["candidate"]
    if not isinstance(candidate, str) or candidate not in REQUIRED:
        raise ValueError(f"Unsupported screw-refinement candidate: {candidate!r}")
    config = saved["config"]
    if not isinstance(config, dict) or config.get("candidate") != candidate:
        raise ValueError("Model candidate differs from its configuration")
    if config.get("version") != "screw-refinement-v1":
        raise ValueError("Unsupported screw-refinement model version")
    if config.get("weights_sha256") != WEIGHTS_SHA:
        raise ValueError("Model was not built with the verified DINOv3 checkpoint")
    if config.get("resolution") != [224, 336]:
        raise ValueError("Expected the frozen 224/336 resolution contract")
    if config.get("radius") != VIEW_RADIUS:
        raise ValueError("Expected raw/aligned224 radius 3 and raw336 radius 4")
    for field, filename in (("runtime_source_sha256", "refine_features.py"),
                            ("geometry_source_sha256", "screw_geometry.py")):
        expected = config.get(field)
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"Missing frozen implementation fingerprint: {field}")
        if file_sha256(ROOT / filename) != expected:
            raise ValueError(f"Saved model requires a different implementation: {filename}")
    threshold = _finite_number(saved["threshold"], "threshold", positive=True)
    if _finite_number(config.get("threshold"), "config.threshold", positive=True) != threshold:
        raise ValueError("Model threshold differs from its configuration")
    medians = saved["medians"]
    if not isinstance(medians, dict) or not isinstance(config.get("medians"), dict):
        raise ValueError("Expected per-view median normalizers")
    if medians != config["medians"]:
        raise ValueError("Model medians differ from its configuration")
    for view, value in medians.items():
        if view not in VIEW_SIZE:
            raise ValueError(f"Unknown median view: {view!r}")
        _finite_number(value, f"median {view}", positive=True)
    banks = saved["banks"]
    if not isinstance(banks, dict) or set(banks) != set(REQUIRED[candidate]):
        raise ValueError("Model banks do not match the candidate's required views")
    for view in REQUIRED[candidate]:
        if view not in medians:
            raise ValueError(f"Missing median normalizer for {view}")
        bank = banks[view]
        side = VIEW_SIZE[view] // 16
        if (not isinstance(bank, torch.Tensor) or bank.ndim != 4
                or bank.shape[0] != side or bank.shape[1] != side
                or bank.shape[2] < 1 or bank.shape[3] != 384
                or bank.dtype != torch.float32 or bank.device.type != "cpu"
                or not torch.isfinite(bank).all().item()):
            raise ValueError(f"Invalid {view} bank; expected finite CPU float32 [{side},{side},M,384]")
    return candidate, threshold, {view: float(value) for view, value in medians.items()}


class RefinedScrewEngine:
    """Resident CPU feature extractors and calibrated screw feature banks."""

    def __init__(self, model_path):
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise ValueError(f"Screw-refinement model is missing: {self.model_path}")
        torch.set_num_threads(1)
        try:
            saved = torch.load(self.model_path, map_location="cpu", weights_only=True)
        except (pickle.UnpicklingError, EOFError, RuntimeError) as exc:
            raise ValueError("Cannot load this file as a weights-only screw-refinement model") from exc
        self.candidate, self.threshold, self.medians = validate_model(saved)
        self.config = saved["config"]
        self.banks = {view: bank.detach().contiguous() for view, bank in saved["banks"].items()}
        self.model_sha256 = file_sha256(self.model_path)
        self.extractors = {
            size: DinoExtractor(size=size, threads=1)
            for size in sorted({VIEW_SIZE[view] for view in REQUIRED[self.candidate]})
        }

    def predict(self, image):
        if not isinstance(image, Image.Image) or min(image.size) < 2:
            raise ValueError("predict requires a PIL image at least 2 by 2 pixels")
        if image.width * image.height > 10_000_000:
            raise ValueError("Image exceeds the 10 million pixel inference limit")
        total_start = time.perf_counter()
        before = time.perf_counter()
        rgb = image.convert("RGB")
        metadata = None
        selected_view = self.candidate
        if self.candidate == "aligned224":
            aligned, metadata = alignment_transform(rgb)
            selected_view = "raw224" if metadata["fallback"] else "aligned224"
            images = {selected_view: rgb if metadata["fallback"] else aligned}
        elif self.candidate == "multi224_336":
            images = {"raw224": rgb, "raw336": rgb}
        else:
            images = {self.candidate: rgb}
        tensors = {view: transform_image(value, VIEW_SIZE[view]).unsqueeze(0)
                   for view, value in images.items()}
        preprocessing_ms = (time.perf_counter() - before) * 1000
        features, per_view = {}, {}
        before = time.perf_counter()
        for view, tensor in tensors.items():
            start = time.perf_counter()
            _, patches = self.extractors[VIEW_SIZE[view]].forward_batch(tensor)
            features[view] = patches[0]
            per_view[view] = {"forward_ms": (time.perf_counter() - start) * 1000,
                              "size": VIEW_SIZE[view], "radius": VIEW_RADIUS[view]}
        forward_ms = (time.perf_counter() - before) * 1000
        before = time.perf_counter()
        scores, heats = {}, {}
        for view, patches in features.items():
            start = time.perf_counter()
            values = spatial_distances_gemm(patches, self.banks[view],
                                            radius=VIEW_RADIUS[view], memory_mib=32)
            side = VIEW_SIZE[view] // 16
            heat = F.interpolate(values.reshape(1, 1, side, side), size=(224, 224),
                                 mode="bilinear", align_corners=False)[0, 0].numpy()
            scores[view] = float(values.max()) / self.medians[view]
            heats[view] = heat / self.medians[view]
            per_view[view]["scoring_ms"] = (time.perf_counter() - start) * 1000
        if self.candidate == "multi224_336":
            heat = .5 * (heats["raw224"] + heats["raw336"])
            score = float(heat.max())
            branch = "multi224_336"
        else:
            score, heat, branch = scores[selected_view], heats[selected_view], selected_view
            if branch == "aligned224":
                heat = inverse_heatmap(heat, metadata)
        scoring_ms = (time.perf_counter() - before) * 1000
        if heat.shape != (224, 224) or not np.isfinite(heat).all() or not math.isfinite(score):
            raise ValueError("Inference produced non-finite or invalid scores")
        total_ms = (time.perf_counter() - total_start) * 1000
        diagnostics = ({"used": True, "version": metadata["version"],
                        "fallback": metadata["fallback"], "quality": metadata["quality"],
                        "source_size": metadata["source_size"],
                        "forward_affine": metadata["forward_affine"],
                        "inverse_affine": metadata["inverse_affine"]}
                       if metadata is not None else {"used": False, "version": GEOMETRY_VERSION})
        return {
            "score": score, "threshold": self.threshold, "anomalous": score > self.threshold,
            "candidate": self.candidate, "branch": branch,
            "heatmap": np.asarray(heat, dtype=np.float32).copy(),
            "geometry_diagnostics": diagnostics,
            "preprocessing_ms": preprocessing_ms, "forward_ms": forward_ms,
            "scoring_ms": scoring_ms, "total_inference_ms": total_ms,
            "per_view_timings": per_view, "model_sha256": self.model_sha256,
            "heatmap_note": HEATMAP_NOTE,
            "timing_note": "Resident-model inference; includes preprocessing/forward/scoring, excludes model loading and PNG rendering.",
            "category_note": "Screw-only engine; it does not identify other products or reject unknown categories.",
        }


def save_outputs(image, prediction, directory):
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    original = image.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    original.save(directory / "original.png")
    heat = prediction["heatmap"]
    low, high = float(heat.min()), float(heat.max())
    normalized = np.zeros_like(heat) if high <= low else (heat - low) / (high - low)
    stops = np.array([0., .25, .5, .75, 1.])
    colors = np.array([[15, 25, 85], [0, 110, 210], [0, 220, 175], [255, 220, 20], [235, 35, 25]])
    pixels = np.stack([np.interp(normalized, stops, colors[:, channel]) for channel in range(3)], axis=-1)
    colored = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
    colored.save(directory / "heatmap.png")
    Image.blend(original, colored, .45).save(directory / "overlay.png")
    result = {key: value for key, value in prediction.items() if key != "heatmap"}
    result["heatmap_range"] = {"min": low, "max": high, "color_scale": "Per-image min/max; not probabilities"}
    result["files"] = {name: str(directory / f"{name}.png") for name in ("original", "heatmap", "overlay")}
    (directory / "prediction.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=ROOT / "release/screw_refinement/model.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/screw_refinement/prediction")
    args = parser.parse_args()
    try:
        from frontier_predict import load_input

        image_path = args.image.expanduser().resolve()
        image = load_input(image_path)
        engine = RefinedScrewEngine(args.model_path)
        prediction = engine.predict(image)
        result = save_outputs(image, prediction, args.output_dir)
        result["image"] = str(image_path)
        (args.output_dir.expanduser().resolve() / "prediction.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
