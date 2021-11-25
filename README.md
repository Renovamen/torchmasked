# torchmasked

Tensor operations with mask for PyTorch.

[![PyPI](https://img.shields.io/pypi/v/torchmasked)](https://pypi.org/project/torchmasked/) [![License](https://img.shields.io/github/license/Renovamen/torchmasked)](https://github.com/Renovamen/torchmasked/blob/main/LICENSE) [![Unittest](https://github.com/Renovamen/torchmasked/workflows/unittest/badge.svg?branch=main)](https://github.com/Renovamen/torchmasked/actions/workflows/unittest.yaml)

Sometimes you need to perform operations on PyTorch tensors with the masked elements been ignored, for example:

```python
>>> input = torch.tensor([1., 2., 3.])
>>> result = torch.sum(input)
>>> print(result)

tensor(6.)

>>> mask = torch.tensor([1, 1, 0]).byte()
>>> masked_result = torchmasked.masked_sum(input, mask)
>>> print(masked_result)

tensor(3.)  # element input[2] is masked and ignored
```

Then this package could be helpful.


&nbsp;

## Requirements

Tested on Python 3.6+ and PyTorch 1.4+.


&nbsp;

## Installation

From PyPI:

```bash
pip install torchmasked
```

From source:

```bash
pip install git+https://github.com/Renovamen/torchmasked.git --upgrade

# or

python setup.py install
```


&nbsp;

## Supported Operations

The usage is the same as PyTorch's original functions. Refer to [PyTorch documentation](https://pytorch.org/docs/stable/index.html) or the [source code](torchmasked) for details.

- [`torchmasked.masked_max`](torchmasked/functional.py) (masked version of [`torch.max`](https://pytorch.org/docs/stable/generated/torch.max.html))
- [`torchmasked.masked_min`](torchmasked/functional.py) ([`torch.min`](https://pytorch.org/docs/stable/generated/torch.min.html))
- [`torchmasked.masked_sum`](torchmasked/functional.py) ([`torch.sum`](https://pytorch.org/docs/stable/generated/torch.sum.html))
- [`torchmasked.masked_mean`](torchmasked/functional.py) ([`torch.mean`](https://pytorch.org/docs/stable/generated/torch.min.html))
- [`torchmasked.masked_softmax`](torchmasked/functional.py) ([`torch.nn.functional.softmax`](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)) / [`torchmasked.nn.MaskedSoftmax`](torchmasked/nn.py) ([`torch.nn.Softmax`](https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html))


&nbsp;

## License

[MIT](LICENSE)
