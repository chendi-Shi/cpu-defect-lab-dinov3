import unittest
import numpy as np
import torch
from lab import nearest, model_features, score_one


class DetectorTests(unittest.TestCase):
    def test_chunked_nearest_matches_direct(self):
        q = torch.tensor([[0., 0.], [3., 4.], [1., 1.]])
        bank = torch.tensor([[0., 0.], [2., 2.]])
        torch.testing.assert_close(nearest(q, bank, chunk=1), torch.cdist(q, bank).min(1).values)

    def test_local_known_patch(self):
        x = torch.zeros(196, 2)
        x[100] = torch.tensor([3., 4.])
        score, heat = score_one(x, torch.zeros(1, 2), 'local', 224)
        self.assertAlmostEqual(score, 5.)
        self.assertEqual(heat.shape, (224, 224))
        self.assertTrue(np.isfinite(heat).all())
        self.assertGreater(float(heat.max()), 0)

    def test_feature_selection_deterministic(self):
        g = torch.randn(2, 512)
        l = torch.randn(2, 196, 384)
        a = model_features(g, l, 'local', 64, 42)
        b = model_features(g, l, 'local', 64, 42)
        torch.testing.assert_close(a, b)
        self.assertEqual(a.shape, (2, 196, 64))
        torch.testing.assert_close(a.norm(dim=-1), torch.ones(2, 196))

    def test_shared_global_uses_same_channels_before_pooling(self):
        g = torch.randn(2, 512)
        l = torch.randn(2, 196, 384)
        channels = torch.randperm(384, generator=torch.Generator().manual_seed(42))[:64]
        expected = torch.nn.functional.normalize(l[..., channels].mean(1), dim=-1)
        actual = model_features(g, l, 'global_shared', 64, 42)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(actual.shape, (2, 64))
        score, heat = score_one(actual[0], actual[1:], 'global_shared', 224)
        self.assertIsNone(heat)
        self.assertAlmostEqual(score, float(torch.dist(actual[0], actual[1])), places=5)


if __name__ == '__main__':
    unittest.main()
