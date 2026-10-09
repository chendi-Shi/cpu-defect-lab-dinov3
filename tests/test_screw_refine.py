"""Protocol checks using normal training images only, without DINO forward."""

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

import numpy as np
from PIL import Image

import screw_refine as refine
from screw_geometry import foreground_mask


@contextmanager
def cache_test_directory():
    """Use inherited Windows ACLs; Python tempfile's mode 0700 is too narrow."""
    base = refine.ROOT / "cache/tests-screw-refine"
    base.mkdir(parents=True, exist_ok=True)
    directory = base / uuid.uuid4().hex
    directory.mkdir()
    try:
        yield directory
    finally:
        resolved = directory.resolve()
        inside = resolved.relative_to(base.resolve())
        if len(inside.parts) != 1 or inside.name != directory.name:
            raise ValueError("Refusing cleanup outside this test's cache directory")
        shutil.rmtree(resolved)


class SplitProtocolTests(unittest.TestCase):
    @unittest.skipUnless(
        (refine.ROOT / "outputs/frontier/screw/seed-42/spatial_kcenter_r3/config.json").is_file(),
        "The original local screw split is needed for this integration check",
    )
    def test_real_normal_split_is_disjoint_complete_stable_and_keeps_calibration(self):
        with cache_test_directory() as directory:
            with patch.object(refine, "OUT", Path(directory)):
                first, manifest = refine.freeze_splits()
                second, again = refine.freeze_splits()
                self.assertEqual(first, second)
                self.assertEqual(manifest, again)
                persisted = json.loads((Path(directory) / "splits.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest, persisted)
        self.assertEqual({key: len(paths) for key, paths in first.items()},
                         {"fit": 192, "select": 48, "audit": 16, "calibration": 64})
        groups = [set(paths) for paths in first.values()]
        for i, left in enumerate(groups):
            for right in groups[i + 1:]:
                self.assertTrue(left.isdisjoint(right))
        union = set.union(*groups)
        self.assertEqual(union, set((refine.FOLDER / "train/good").glob("*.png")))
        self.assertTrue(all(path.parent.name == "good" and path.parent.parent.name == "train"
                            for path in union))
        old = json.loads((refine.ROOT / "outputs/frontier/screw/seed-42/spatial_kcenter_r3/config.json")
                         .read_text(encoding="utf-8"))
        self.assertEqual(set(first["calibration"]),
                         {refine.FOLDER / path for path in old["calibration_files"]})
        for entries in manifest["groups"].values():
            for record in entries:
                source = refine.ROOT / record["path"]
                self.assertEqual(record["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())


class CandidateSelectionTests(unittest.TestCase):
    @staticmethod
    def metrics(raw=.80, aligned=.80, high=.80, multi=.80):
        return {candidate: {"macro_auroc": value}
                for candidate, value in zip(refine.CANDIDATES, (raw, aligned, high, multi))}

    def test_proxy_gain_below_minimum_keeps_baseline(self):
        selected, best = refine.choose_candidate(self.metrics(multi=.8049))
        self.assertEqual((selected, best), ("raw224", "multi224_336"))

    def test_minimum_gain_boundary_is_accepted(self):
        selected, best = refine.choose_candidate(self.metrics(high=.805))
        self.assertEqual((selected, best), ("raw336", "raw336"))

    def test_exact_tie_uses_predeclared_order(self):
        self.assertEqual(refine.choose_candidate(self.metrics()), ("raw224", "raw224"))
        self.assertEqual(refine.choose_candidate(self.metrics(aligned=.90, high=.90, multi=.90)),
                         ("aligned224", "aligned224"))

    def test_numerical_near_tie_does_not_override_fixed_order(self):
        values = self.metrics(aligned=.90, high=.90 + 5e-10, multi=.90)
        self.assertEqual(refine.choose_candidate(values), ("aligned224", "aligned224"))

    def test_clear_winner_overrides_priority_order(self):
        values = self.metrics(aligned=.89, high=.91, multi=.93)
        self.assertEqual(refine.choose_candidate(values), ("multi224_336", "multi224_336"))

    def test_custom_minimum_gain_is_respected(self):
        self.assertEqual(refine.choose_candidate(self.metrics(high=.81), min_gain=.02),
                         ("raw224", "raw336"))


class SyntheticProtocolTests(unittest.TestCase):
    @unittest.skipUnless((refine.FOLDER / "train/good/000.png").is_file(),
                         "Local normal screw image is needed for this integration check")
    def test_synthetic_defects_preserve_source_resolution_and_background_and_are_reproducible(self):
        source = refine.FOLDER / "train/good/000.png"
        source_bytes = source.read_bytes()
        with Image.open(source) as opened:
            original = opened.convert("RGB")
        original_array = np.asarray(original)
        foreground, _ = foreground_mask(original)
        with cache_test_directory() as directory:
            with patch.object(refine, "SYNTH", Path(directory)):
                for kind in refine.KINDS:
                    with self.subTest(kind=kind):
                        record = refine.synthesize(source, kind, "select")
                        first = (refine.ROOT / record["path"]).read_bytes()
                        repeat = refine.synthesize(source, kind, "select")
                        self.assertEqual(record, repeat)
                        self.assertEqual(first, (refine.ROOT / repeat["path"]).read_bytes())
                        self.assertEqual(source.read_bytes(), source_bytes)
                        with Image.open(refine.ROOT / record["path"]) as opened:
                            self.assertEqual(opened.size, original.size)
                            generated = np.asarray(opened.convert("RGB"))
                        with Image.open(refine.ROOT / record["mask"]) as opened:
                            self.assertEqual(opened.size, original.size)
                            defect = np.asarray(opened) > 0
                        self.assertGreater(int(defect.sum()), 0)
                        self.assertFalse(np.any(defect & ~foreground))
                        self.assertTrue(np.array_equal(generated[~defect], original_array[~defect]))
                        changed = np.any(generated != original_array, axis=-1)
                        self.assertGreater(int(changed.sum()), 0)
                        self.assertFalse(np.any(changed & ~defect))
                        x, y = record["parameters"]["x"], record["parameters"]["y"]
                        self.assertTrue(foreground[y, x])
                        self.assertEqual(record["source"], refine.relative(source))
                        self.assertEqual(record["partition"], "select")
                        self.assertEqual(record["sha256"], hashlib.sha256(first).hexdigest())
        self.assertEqual(source.read_bytes(), source_bytes)


if __name__ == "__main__":
    unittest.main()
