import unittest

import torch

from adapter.loss import adapter_loss
from adapter.model import ChannelAdapter
from adapter.train import split_dataset


class AdapterTests(unittest.TestCase):
    def test_shape(self):
        model = ChannelAdapter()
        output = model(torch.randn(2, 256, 16, 16))
        self.assertEqual(tuple(output.shape), (2, 768, 16, 16))

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            ChannelAdapter()(torch.randn(2, 256, 8, 8))

    def test_exact_loss(self):
        target = torch.randn(2, 768, 16, 16)
        losses = adapter_loss(target, target)
        self.assertAlmostEqual(float(losses["total"]), 0.0, places=6)

    def test_split_is_deterministic_and_disjoint(self):
        first_train, first_validation = split_dataset(list(range(100)), 10, 7)
        second_train, second_validation = split_dataset(list(range(100)), 10, 7)
        self.assertEqual(first_train.indices, second_train.indices)
        self.assertEqual(first_validation.indices, second_validation.indices)
        self.assertFalse(set(first_train.indices) & set(first_validation.indices))


if __name__ == "__main__":
    unittest.main()
