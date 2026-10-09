"""Branch semantics checks without downloading or loading a vision model."""

import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import screw_refine_predict as runtime


class FixedExtractor:
    def __init__(self, patches):
        self.patches = patches

    def forward_batch(self, tensor):
        return torch.zeros((1, 384)), self.patches.unsqueeze(0)


class InputSignalExtractor:
    def forward_batch(self, tensor):
        patches = torch.zeros((1, 196, 384))
        patches[:, :, 0] = tensor.mean()
        return torch.zeros((1, 384)), patches


def engine_fixture(candidate, extractors):
    engine = runtime.RefinedScrewEngine.__new__(runtime.RefinedScrewEngine)
    engine.candidate = candidate
    engine.threshold = 1.5
    engine.medians = {"raw224": 2., "aligned224": 2., "raw336": 4.}
    engine.model_sha256 = "unit-fixture"
    engine.extractors = extractors
    engine.banks = {
        view: torch.zeros((runtime.VIEW_SIZE[view] // 16, runtime.VIEW_SIZE[view] // 16, 1, 384))
        for view in runtime.REQUIRED[candidate]
    }
    return engine


def metadata(fallback=False, rotate=False):
    affine = [-1., 0., 224., 0., -1., 224.] if rotate else [1., 0., 0., 0., 1., 0.]
    return {"version": runtime.GEOMETRY_VERSION, "source_size": [224, 224],
            "forward_affine": affine, "inverse_affine": affine,
            "fallback": fallback, "quality": {"passed": not fallback,
                                                "failures": ["fixture"] if fallback else []}}


class RefinedInferenceSemantics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_alignment_failure_uses_original_pixels_raw_bank_and_raw_median(self):
        engine = engine_fixture("aligned224", {224: InputSignalExtractor()})
        engine.medians["aligned224"] = 100.
        engine.banks["aligned224"].fill_(50.)
        original = Image.new("RGB", (224, 224), "white")
        changed = Image.new("RGB", (224, 224), "black")
        # Return visibly different aligned pixels. Fallback must discard them.
        with patch.object(runtime, "alignment_transform", return_value=(changed, metadata(fallback=True))):
            result = engine.predict(original)
        white_feature = sum((1. - mean) / std for mean, std in zip(
            (.485, .456, .406), (.229, .224, .225))) / 3
        self.assertAlmostEqual(result["score"], white_feature / 2., delta=2e-6)
        self.assertEqual(result["branch"], "raw224")
        self.assertTrue(result["geometry_diagnostics"]["fallback"])
        self.assertTrue(np.allclose(result["heatmap"], white_feature / 2., atol=2e-6))

    def test_multiscale_fuses_maps_before_maximum(self):
        low, high = torch.zeros((196, 384)), torch.zeros((441, 384))
        low[7 * 14 + 7, 0] = 10.
        high[2 * 21 + 2, 0] = 40.
        engine = engine_fixture("multi224_336", {224: FixedExtractor(low), 336: FixedExtractor(high)})
        result = engine.predict(Image.new("RGB", (224, 224), "gray"))
        # Disjoint peaks must not be replaced by the mean of individual maxima.
        self.assertLess(result["score"], .5 * (10. / 2. + 40. / 4.))
        self.assertEqual(result["score"], float(result["heatmap"].max()))
        # Independent bilinear coordinate weights for two known peak samples.
        for pixel, grid, peak, height, median in ((119, 14, 7, 10., 2.), (26, 21, 2, 40., 4.)):
            source = (pixel + .5) * grid / 224. - .5
            weight = max(0., 1. - abs(source - peak)) ** 2
            expected = .5 * height * weight / median
            self.assertAlmostEqual(float(result["heatmap"][pixel, pixel]), expected, delta=2e-6)

    def test_aligned_heatmap_returns_to_original_frame_without_changing_score(self):
        features = torch.zeros((196, 384))
        features[1 * 14 + 1, 0] = 20.
        engine = engine_fixture("aligned224", {224: FixedExtractor(features)})
        image = Image.new("RGB", (224, 224), "gray")
        with patch.object(runtime, "alignment_transform", return_value=(image, metadata(rotate=True))):
            result = engine.predict(image)
        y, x = np.unravel_index(result["heatmap"].argmax(), result["heatmap"].shape)
        self.assertGreater(y, 180); self.assertGreater(x, 180)
        self.assertEqual(result["score"], 10.)
        self.assertLess(float(result["heatmap"].max()), result["score"])
        self.assertEqual(result["branch"], "aligned224")


if __name__ == "__main__":
    unittest.main()
