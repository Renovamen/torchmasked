import os
import sys
import unittest

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(base_path)

import torch
import torch.nn.functional as F
import numpy as np
import torchmasked as tm


def _test_pipline(masked_func, torch_func):
    input = torch.randn(3, 5)
    mask = torch.randn(3, 5) > 0

    output = masked_func(input, mask, dim=-1)
    output = output[0] if isinstance(output, tuple) else output

    for i in range(output.size(0)):
        result = output[i].numpy()
        truth = torch_func(input[i].masked_select(mask[i])).numpy()
        np.testing.assert_allclose(result, truth, atol=1e-6)

    result = masked_func(input, mask).numpy()
    truth = torch_func(input.masked_select(mask)).numpy()
    np.testing.assert_allclose(result, truth, atol=1e-6)

def _test_pipline_F(masked_func, torch_func):
    input = torch.randn(3, 5)
    mask = torch.randn(3, 5) > 0

    output = masked_func(input, mask, dim=-1)

    for i in range(output.size(0)):
        result = output[i].masked_select(mask[i]).numpy()
        truth = torch_func(input[i].masked_select(mask[i]), dim=-1).numpy()
        np.testing.assert_allclose(result, truth, atol=1e-6)

class TestFunctional(unittest.TestCase):
    def test_masked_sum(self) -> None:
        _test_pipline(tm.masked_sum, torch.sum)

    def test_masked_mean(self) -> None:
        _test_pipline(tm.masked_mean, torch.mean)

    def test_masked_max(self) -> None:
        _test_pipline(tm.masked_max, torch.max)

    def test_masked_min(self) -> None:
        _test_pipline(tm.masked_min, torch.min)

    def test_masked_softmax(self) -> None:
        _test_pipline_F(tm.masked_softmax, F.softmax)


if __name__ == '__main__':
    unittest.main()
