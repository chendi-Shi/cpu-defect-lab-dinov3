import json
import unittest

import numpy as np
from PIL import Image, ImageDraw

from screw_geometry import alignment_transform, foreground_mask, inverse_heatmap, warp_mask


def synthetic_screw(angle=0):
    image = Image.new("RGB", (512, 512), (220, 220, 220))
    draw = ImageDraw.Draw(image)
    draw.rectangle((236, 105, 276, 390), fill=(60, 60, 60))
    draw.polygon([(236, 390), (276, 390), (256, 445)], fill=(60, 60, 60))
    draw.polygon([(194, 65), (318, 65), (276, 130), (236, 130)], fill=(60, 60, 60))
    return image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(220, 220, 220))


class ScrewGeometryTests(unittest.TestCase):
    def test_foreground_is_largest_dark_component_at_original_size(self):
        image = synthetic_screw()
        ImageDraw.Draw(image).rectangle((10, 10, 14, 14), fill="black")
        mask, info = foreground_mask(image)
        self.assertEqual(mask.shape, (512, 512))
        self.assertEqual(mask.dtype, np.bool_)
        self.assertTrue(mask[200, 256])
        self.assertFalse(mask[12, 12])
        self.assertTrue(info["geometry_available"])

    def test_head_direction_and_affine_inverse(self):
        for angle in (0, 57, 140, 250):
            with self.subTest(angle=angle):
                aligned, metadata = alignment_transform(synthetic_screw(angle))
                self.assertFalse(metadata["fallback"], metadata["quality"])
                self.assertEqual(aligned.size, (512, 512))
                self.assertEqual(aligned.mode, "RGB")
                json.dumps(metadata, allow_nan=False)
                forward = np.array(metadata["forward_affine"]).reshape(2, 3)
                inverse = np.array(metadata["inverse_affine"]).reshape(2, 3)
                f3 = np.vstack((forward, [0, 0, 1]))
                i3 = np.vstack((inverse, [0, 0, 1]))
                np.testing.assert_allclose(f3 @ i3, np.eye(3), atol=1e-10)
                mask, info = foreground_mask(aligned)
                self.assertLess(info["head_axis"][1], -.99)

    def test_uniform_or_round_foreground_falls_back(self):
        uniform = Image.new("RGB", (512, 512), "white")
        round_image = Image.new("RGB", (512, 512), (220, 220, 220))
        ImageDraw.Draw(round_image).ellipse((160, 160, 350, 350), fill=(60, 60, 60))
        for image in (uniform, round_image):
            aligned, metadata = alignment_transform(image)
            self.assertTrue(metadata["fallback"])
            self.assertFalse(metadata["quality"]["passed"])
            np.testing.assert_array_equal(np.asarray(aligned), np.asarray(image))
            self.assertEqual(metadata["forward_affine"], [1., 0., 0., 0., 1., 0.])

    def test_non_square_geometry_is_explicit_fallback(self):
        image = synthetic_screw().resize((768, 512))
        aligned, metadata = alignment_transform(image)
        self.assertTrue(metadata["fallback"])
        self.assertIn("unsupported_non_square_geometry", metadata["quality"]["failures"])
        np.testing.assert_array_equal(np.asarray(aligned), np.asarray(image))

    def test_axis_center_is_full_extent_midpoint(self):
        _, info = foreground_mask(synthetic_screw())
        mask, _ = foreground_mask(synthetic_screw())
        y, x = np.nonzero(mask)
        axis = np.array(info["head_axis"])
        # Diagnostics are measured at 256, mask returned at 512. The difference
        # between mask and diagnostics includes only nearest enlargement.
        projected = np.stack((x / 2, y / 2), axis=1) @ axis
        center = np.array(info["alignment_center_px256"]) @ axis
        self.assertAlmostEqual(center, (projected.min() + projected.max()) / 2, delta=.6)

    def test_mask_warp_preserves_binary_values_and_foreground(self):
        image = synthetic_screw(57)
        mask, _ = foreground_mask(image)
        aligned, metadata = alignment_transform(image)
        warped = warp_mask(mask, metadata)
        observed, _ = foreground_mask(aligned)
        self.assertEqual(warped.dtype, np.bool_)
        intersection = (warped & observed).sum()
        union = (warped | observed).sum()
        self.assertGreater(intersection / union, .94)
        with self.assertRaises(ValueError):
            warp_mask(np.zeros((224, 224)), metadata)

    def test_heatmap_inverse_identity_and_translation(self):
        heat = np.arange(224 * 224, dtype=np.float32).reshape(224, 224)
        identity = {"source_size": [1024, 1024], "forward_affine": [1., 0., 0., 0., 1., 0.]}
        np.testing.assert_array_equal(inverse_heatmap(heat, identity), heat)
        translated = {"source_size": [1024, 1024], "forward_affine": [1., 0., 1024 / 224, 0., 1., 0.]}
        restored = inverse_heatmap(heat, translated)
        np.testing.assert_array_equal(restored[:, :-1], heat[:, 1:])
        np.testing.assert_array_equal(restored[:, -1], 0.)
        with self.assertRaises(ValueError):
            inverse_heatmap(np.zeros((14, 14)), identity)

    def test_heatmap_inverse_quarter_turn_uses_forward_transform(self):
        heat = np.arange(224 * 224, dtype=np.float32).reshape(224, 224)
        quarter_turn = {"source_size": [1024, 1024],
                        "forward_affine": [0., -1., 1024., 1., 0., 0.]}
        np.testing.assert_array_equal(inverse_heatmap(heat, quarter_turn), np.rot90(heat))


if __name__ == "__main__":
    unittest.main()
