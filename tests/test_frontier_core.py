"""Independent numerical checks for the CPU DINOSaur memory operations."""

import unittest

import torch

from frontier_core import (
    build_spatial_bank, choose_task, spatial_distances, unrestricted_distances,
)


def naive_spatial(patches, bank, radius):
    side = bank.shape[0]
    values = []
    for y in range(side):
        for x in range(side):
            choices = []
            for yy in range(max(0, y - radius), min(side, y + radius + 1)):
                for xx in range(max(0, x - radius), min(side, x + radius + 1)):
                    for reference in bank[yy, xx]:
                        choices.append(torch.norm(patches[y * side + x] - reference))
            values.append(torch.stack(choices).min())
    return torch.stack(values)


class SpatialRetrievalTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(17)
        self.bank = torch.randn(4, 4, 3, 7, generator=generator)
        self.patches = torch.randn(16, 7, generator=generator)

    def test_neighborhood_matches_independent_loops_for_edges_and_chunks(self):
        for radius in (0, 1, 3, 7):
            expected = naive_spatial(self.patches, self.bank, radius)
            for chunk in (1, 3, 8, 100):
                with self.subTest(radius=radius, chunk=chunk):
                    actual = spatial_distances(self.patches, self.bank, radius, chunk)
                    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_outside_grid_cannot_introduce_zero_reference(self):
        bank = torch.full((2, 2, 1, 1), 5.0)
        patches = torch.zeros(4, 1)
        torch.testing.assert_close(spatial_distances(patches, bank, radius=1), torch.full((4,), 5.0))

    def test_global_retrieval_matches_naive_flat_bank(self):
        expected = torch.stack([
            torch.stack([torch.norm(query - reference) for reference in self.bank.reshape(-1, 7)]).min()
            for query in self.patches
        ])
        for chunk in (1, 5, 64):
            torch.testing.assert_close(unrestricted_distances(self.patches, self.bank, chunk), expected)
        torch.testing.assert_close(spatial_distances(self.patches, self.bank, radius=3), expected)

    def test_self_distance_is_exactly_zero(self):
        bank = torch.full((2, 2, 1, 3), 1000.123)
        queries = bank[:, :, 0].reshape(4, 3).clone()
        self.assertTrue(torch.equal(unrestricted_distances(queries, bank), torch.zeros(4)))
        self.assertTrue(torch.equal(spatial_distances(queries, bank), torch.zeros(4)))

    def test_invalid_grid_and_radius_rejected(self):
        with self.assertRaises(ValueError):
            spatial_distances(self.patches[:-1], self.bank)
        with self.assertRaises(ValueError):
            spatial_distances(self.patches, self.bank, radius=-1)
        with self.assertRaises(ValueError):
            unrestricted_distances(self.patches, self.bank, chunk=0)


class SpatialBankTests(unittest.TestCase):
    def test_fewer_than_twenty_images_caps_coreset_and_preserves_sources(self):
        features = torch.arange(7 * 4 * 3, dtype=torch.float32).reshape(7, 4, 3)
        for sampler in ("kcenter", "random"):
            bank, indices = build_spatial_bank(features, sampler=sampler)
            self.assertEqual(bank.shape, (2, 2, 7, 3))
            for location in range(4):
                self.assertEqual(len(set(indices[location].tolist())), 7)
                torch.testing.assert_close(bank.reshape(4, 7, 3)[location], features[indices[location], location])

    def test_identical_vectors_still_select_distinct_indices_and_terminate(self):
        features = torch.zeros(37, 9, 5)
        bank, indices = build_spatial_bank(features, rho=0.1)
        self.assertEqual(bank.shape, (3, 3, 20, 5))
        for row in indices:
            self.assertEqual(len(set(row.tolist())), 20)

    def test_greedy_selection_has_farthest_unselected_property(self):
        generator = torch.Generator().manual_seed(9)
        features = torch.randn(43, 4, 6, generator=generator)
        bank, indices = build_spatial_bank(features, rho=0.5, seed=83)
        self.assertEqual(bank.shape, (2, 2, 21, 6))
        for location in range(4):
            points = features[:, location]
            chosen = []
            for step, actual_index in enumerate(indices[location].tolist()):
                if step:
                    minimum = torch.stack([
                        torch.norm(points - points[index], dim=1) for index in chosen
                    ]).min(dim=0).values
                    minimum[chosen] = -float("inf")
                    self.assertEqual(actual_index, minimum.argmax().item())
                chosen.append(actual_index)

    def test_seeded_sampling_reproducible_for_each_sampler(self):
        features = torch.randn(42, 4, 8, generator=torch.Generator().manual_seed(4))
        for sampler in ("random", "kcenter"):
            first_bank, first_indices = build_spatial_bank(features, seed=4, sampler=sampler)
            second_bank, second_indices = build_spatial_bank(features, seed=4, sampler=sampler)
            self.assertTrue(torch.equal(first_indices, second_indices))
            self.assertTrue(torch.equal(first_bank, second_bank))

    def test_invalid_features_and_sampler_rejected(self):
        with self.assertRaises(ValueError):
            build_spatial_bank(torch.zeros(2, 3, 4))
        with self.assertRaises(ValueError):
            build_spatial_bank(torch.zeros(0, 4, 4))
        with self.assertRaises(ValueError):
            build_spatial_bank(torch.zeros(2, 4, 4), rho=0)
        with self.assertRaises(ValueError):
            build_spatial_bank(torch.zeros(2, 4, 4), sampler="unknown")


class RoutingTests(unittest.TestCase):
    def test_euclidean_routing_uses_magnitude_without_extra_normalization(self):
        query = torch.tensor([3.0, 0.0])
        prototypes = {"far_same_direction": torch.tensor([30.0, 0.0]), "near": torch.tensor([2.0, 1.0])}
        self.assertEqual(choose_task(query, prototypes), "near")

    def test_routing_ties_are_deterministic_and_empty_bank_rejected(self):
        query = torch.tensor([0.0, 0.0])
        self.assertEqual(choose_task(query, {"first": torch.tensor([-1.0, 0.0]), "second": torch.tensor([1.0, 0.0])}), "first")
        with self.assertRaises(ValueError):
            choose_task(query, {})


if __name__ == "__main__":
    unittest.main()
