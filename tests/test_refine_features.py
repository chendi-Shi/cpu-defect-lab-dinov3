"""Numerical and cache-contract checks; model smoke uses normal training data."""

import contextlib
import shutil
import unittest
import uuid
from pathlib import Path

import torch
from PIL import Image

from frontier_core import spatial_distances
from refine_features import (
    DinoExtractor, ROOT, clear_gemm_cache, spatial_distances_f64,
    spatial_distances_gemm, transform_image,
)


@contextlib.contextmanager
def workspace_temporary():
    folder = ROOT / "cache/refinement_unit_tests"
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / ("run-" + uuid.uuid4().hex)
    # Windows sandbox tokens cannot reopen directories with tempfile's 0700
    # ACL. Inherit this workspace's ACL instead of requesting that ACL.
    temporary.mkdir()
    assert temporary.resolve().parent == folder.resolve()
    try:
        yield str(temporary)
    finally:
        assert temporary.resolve().parent == folder.resolve()
        shutil.rmtree(temporary)


class RefineDistanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_random_and_boundary_windows(self):
        rng = torch.Generator().manual_seed(581)
        bank = torch.randn((4, 4, 7, 384), generator=rng)
        query = torch.randn((16, 384), generator=rng)
        for radius in (0, 1, 3, 8):
            for chunk in (1, 8, 23):
                expected = spatial_distances(query, bank, radius, chunk)
                actual = spatial_distances_f64(query, bank, radius, chunk)
                self.assertLessEqual(float((actual - expected).abs().max()), 2e-5)

    def test_gemm_random_boundaries_and_query_budget(self):
        rng = torch.Generator().manual_seed(591)
        bank = torch.randn((4, 4, 7, 384), generator=rng)
        query = torch.randn((16, 384), generator=rng)
        for radius in (0, 1, 3, 8):
            expected = spatial_distances(query, bank, radius, chunk=8)
            for memory_mib in (.004, 32):
                actual = spatial_distances_gemm(query, bank, radius, memory_mib)
                self.assertLessEqual(float((actual - expected).abs().max()), 2e-5)

    def test_gemm_self_identical_and_invalid_boundaries(self):
        bank = torch.full((3, 3, 2, 384), 1000.123)
        query = bank[:, :, 0].reshape(9, 384).clone()
        for radius in (0, 3):
            actual = spatial_distances_gemm(query, bank, radius)
            self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))
        query = torch.zeros(9, 384)
        self.assertTrue(torch.equal(spatial_distances_gemm(query, bank, 1), spatial_distances(query, bank, 1)))

    def test_gemm_cache_detects_mutation_and_inference_tensor(self):
        bank = torch.randn((3, 3, 2, 32), generator=torch.Generator().manual_seed(45))
        query = torch.randn((9, 32), generator=torch.Generator().manual_seed(52))
        clear_gemm_cache()
        first = spatial_distances_gemm(query, bank, 1)
        cached = spatial_distances_gemm(query, bank, 1)
        self.assertTrue(torch.equal(first, cached))
        bank.add_(50)
        updated = spatial_distances_gemm(query, bank, 1)
        self.assertTrue(torch.equal(updated, spatial_distances(query, bank, 1)))
        self.assertFalse(torch.equal(first, updated))
        with torch.inference_mode():
            inference_bank = bank.clone()
            spatial_distances_gemm(query, inference_bank, 1)
            inference_bank.add_(30)
            actual = spatial_distances_gemm(query, inference_bank, 1)
        self.assertTrue(torch.equal(actual, spatial_distances(query, inference_bank, 1)))
        clear_gemm_cache()

    def test_gemm_rejects_invalid_and_insufficient_budget(self):
        query = torch.ones(4, 8)
        bank = torch.ones(2, 2, 3, 8)
        for radius, budget in ((-1, 32), (True, 32), (1, 0), (1, float("nan")), (1, .000001)):
            with self.assertRaises(ValueError):
                spatial_distances_gemm(query, bank, radius, budget)

    def test_gemm_cache_is_bounded_to_three_entries(self):
        import refine_features

        clear_gemm_cache()
        banks = [torch.full((2, 2, 3, 8), float(i)) for i in range(4)]
        query = torch.zeros(4, 8)
        for bank in banks:
            spatial_distances_gemm(query, bank, 1)
        self.assertEqual(len(refine_features._GEMM_CACHE), 3)
        self.assertFalse(any(key[0] == id(banks[0]) for key in refine_features._GEMM_CACHE))
        clear_gemm_cache()

    def test_exact_self_distances(self):
        rng = torch.Generator().manual_seed(14)
        bank = torch.randn((5, 5, 4, 384), generator=rng)
        query = bank[:, :, 0].reshape(25, 384).clone()
        for radius in (0, 3):
            actual = spatial_distances_f64(query, bank, radius)
            self.assertLessEqual(float(actual.max()), 1e-6)
            self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))

    def test_identical_vectors_and_boundary_excludes_invalid(self):
        bank = torch.full((3, 3, 2, 384), 10.0)
        query = torch.zeros(9, 384)
        expected = spatial_distances(query, bank, 1)
        actual = spatial_distances_f64(query, bank, 1)
        self.assertTrue(torch.equal(actual, expected))
        self.assertGreater(float(actual[0]), 0.0)
        self.assertEqual(float(spatial_distances_f64(bank[:, :, 0].reshape(9, 384), bank, 2).max()), 0.0)

    def test_reject_invalid_arguments(self):
        query = torch.ones(4, 8)
        bank = torch.ones(2, 2, 3, 8)
        for radius, chunk in ((-1, 8), (True, 8), (1, 0)):
            with self.assertRaises(ValueError):
                spatial_distances_f64(query, bank, radius, chunk)


class RefineCacheTests(unittest.TestCase):
    def setUp(self):
        self.temporary = workspace_temporary()
        self.folder = Path(self.temporary.__enter__())
        self.image = self.folder / "normal.png"
        Image.new("RGB", (16, 16), (10, 20, 30)).save(self.image)
        # Exercise cache behavior without loading a large vision model.
        self.extractor = DinoExtractor.__new__(DinoExtractor)
        self.extractor.size = 224
        self.extractor.side = 14
        self.extractor.versions = {"torch": str(torch.__version__), "timm": "test", "torchvision": "test"}
        self.extractor.cache_dir = self.folder / "cache"
        self.calls = 0

        def forward(tensor):
            self.calls += 1
            signal = tensor.mean(dim=(1, 2, 3))
            return signal[:, None].expand(-1, 384).contiguous(), signal[:, None, None].expand(-1, 196, 384).contiguous()

        self.extractor.forward_batch = forward

    def tearDown(self):
        self.temporary.__exit__(None, None, None)

    def test_cache_hit_and_content_invalidation(self):
        first = self.extractor.extract([self.image])
        second = self.extractor.extract([self.image])
        self.assertFalse(first[4]); self.assertTrue(second[4])
        self.assertEqual(first[3], second[3]); self.assertEqual(self.calls, 1)
        self.assertTrue(torch.equal(first[1], second[1]))
        Image.new("RGB", (16, 16), (60, 70, 80)).save(self.image)
        third = self.extractor.extract([self.image])
        self.assertNotEqual(first[3], third[3]); self.assertFalse(third[4])

    def test_custom_identity_and_size_invalidation(self):
        preprocess = lambda image: transform_image(image, 224)
        with self.assertRaises(ValueError):
            self.extractor.extract([self.image], preprocess=preprocess)
        a = self.extractor.extract([self.image], preprocess=preprocess, identity={"alignment": 1})
        b = self.extractor.extract([self.image], preprocess=preprocess, identity={"alignment": 2})
        self.assertNotEqual(a[3], b[3])
        encoded224, key224 = self.extractor._cache_identity([self.image], 2, {"size": 224})
        self.extractor.size = 336
        _, key336 = self.extractor._cache_identity([self.image], 2, {"size": 224})
        self.assertNotEqual(key224, key336)
        self.assertIn('"size":224', encoded224)

    def test_version_and_order_identity(self):
        other = self.folder / "other.png"
        Image.new("RGB", (16, 16), "white").save(other)
        _, a = self.extractor._cache_identity([self.image, other], 2, {"v": 1})
        _, b = self.extractor._cache_identity([other, self.image], 2, {"v": 1})
        self.assertNotEqual(a, b)
        self.extractor.versions["timm"] = "changed"
        _, c = self.extractor._cache_identity([self.image, other], 2, {"v": 1})
        self.assertNotEqual(a, c)

    def test_default_preprocessing_shapes(self):
        with Image.open(self.image) as image:
            for size in (224, 336):
                result = transform_image(image, size)
                self.assertEqual(tuple(result.shape), (3, size, size))
                self.assertEqual(result.dtype, torch.float32)
                self.assertTrue(torch.isfinite(result).all())


@unittest.skipUnless((ROOT / "cache/frontier/model.safetensors").is_file()
                     and (ROOT / "data/mvtec/screw/train/good/000.png").is_file(),
                     "Local checkpoint and normal training image are needed")
class RealModelSmokeTests(unittest.TestCase):
    def test_224_and_336_normal_training_only(self):
        image = ROOT / "data/mvtec/screw/train/good/000.png"
        with workspace_temporary() as temporary:
            for size in (224, 336):
                extractor = DinoExtractor(size=size, cache_dir=temporary)
                cls, patches, elapsed, key, hit = extractor.extract([image], batch=2)
                self.assertEqual(tuple(cls.shape), (1, 384))
                self.assertEqual(tuple(patches.shape), (1, (size // 16) ** 2, 384))
                self.assertTrue(torch.isfinite(patches).all())
                self.assertGreater(elapsed, 0)
                self.assertFalse(hit)
                self.assertTrue(extractor.extract([image])[4])


if __name__ == "__main__":
    unittest.main()
