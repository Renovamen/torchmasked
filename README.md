# torchmasked

Tensor operations with mask for PyTorch.

[![PyPI](https://img.shields.io/pypi/v/torchmasked)](https://pypi.org/project/torchmasked/) [![License](https://img.shields.io/github/license/Renovamen/torchmasked)](https://github.com/Renovamen/torchmasked/blob/main/LICENSE) [![Unittest](https://github.com/Renovamen/torchmasked/workflows/unittest/badge.svg?branch=main)](https://github.com/Renovamen/torchmasked/actions/workflows/unittest.yaml)

Sometimes you need to perform operations on tensors with the masked elements been ignored, for example:

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

- `torchmasked.max` (masked version of `torch.max`)
- `torchmasked.min` (`torch.min`)
- `torchmasked.sum` (`torch.sum`)
- `torchmasked.mean` (`torch.mean`)
- `torchmasked.softmax` (`torch.nn.functional.softmax`) / `torchmasked.nn.Softmax` (`torch.nn.Softmax`)


&nbsp;

## License

[MIT](LICENSE)
