import os
import sys
import unittest

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import torch
import numpy as np
import torchmasked as tm


def _test_nn(masked_nn, torch_nn):
    input = torch.randn(3, 5)
    mask = torch.randn(3, 5) > 0

    output = masked_nn(input, mask)

    for i in range(output.size(0)):
        result = output[i].masked_select(mask[i]).numpy()
        truth = torch_nn(input[i].masked_select(mask[i])).numpy()
        np.testing.assert_allclose(result, truth, atol=1e-6)


class TestNN(unittest.TestCase):
    def test_masked_softmax(self) -> None:
        masked_softmax = tm.nn.MaskedSoftmax(-1)
        softmax = torch.nn.Softmax(-1)
        _test_nn(masked_softmax, softmax)


if __name__ == '__main__':
    unittest.main()
