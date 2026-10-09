import unittest
from metrics import auroc, threshold_stats


class MetricTests(unittest.TestCase):
    def test_ordering(self):
        self.assertEqual(auroc([0, 1, 0, 1], [0, 2, 1, 3]), 1)
        self.assertEqual(auroc([0, 1], [1, 0]), 0)

    def test_ties(self):
        self.assertEqual(auroc([0, 1, 0, 1], [1, 1, 1, 1]), .5)
        self.assertEqual(auroc([0, 1, 1], [0, 0, 1]), .75)

    def test_single_class(self):
        self.assertIsNone(auroc([0, 0], [0, 1]))

    def test_invalid_inputs(self):
        for y, s in [([0, 256], [0, 1]), ([0, 1], [0]), ([0, 1], [0, float('nan')])]:
            with self.assertRaises(ValueError):
                auroc(y, s)

    def test_threshold(self):
        r = threshold_stats([0, 1, 1, 0], [0, 2, 0, 3], 1)
        self.assertEqual((r['tp'], r['fp'], r['fn'], r['tn']), (1, 1, 1, 1))


if __name__ == '__main__':
    unittest.main()
