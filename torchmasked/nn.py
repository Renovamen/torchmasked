import torch
from torch import nn
from typing import Optional

import torchmasked.functional as F

__all__ = ['MaskedSoftmax']

class MaskedSoftmax(nn.Module):
    """
    Apply `torch.nn.Softmax` while some of the elements of input tensor
    being masked.

    - input: shape of :math:`(*)`, where `*` means any number of additional
        dimensions.

    - mask: A 0-1 mask for the input tensor, 0 for tokens that are masked,
        1 for tokens that are not masked. Should be broadcastable with input.
        If None, a regular softmax result will be returned.

    - output: same shape as the input

    Parameters
    ----------
    dim : int
        A dimension along which softmax will be computed (so every slice
            along dim will sum to 1).
    """
    def __init__(self, dim: Optional[int] = None) -> None:
        super(MaskedSoftmax, self).__init__()
        self.dim = dim

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return F.masked_softmax(input, mask, dim=self.dim, _stacklevel=5)

    def extra_repr(self) -> str:
        return 'dim={dim}'.format(dim=self.dim)
