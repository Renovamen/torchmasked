import torch
import torch.nn.functional as F
from torch.types import _dtype as DType
from typing import Any, Callable, Optional, Union, List, Tuple

__all__ = [
    'masked_softmax',
    'masked_sum',
    'masked_mean',
    'masked_max',
    'masked_min'
]

def _fill_with_mask(input: torch.Tensor, mask: torch.Tensor, fill_value) -> torch.Tensor:
    inverted_mask = (1.0 - mask.float()).bool()
    return input.masked_fill(inverted_mask, fill_value)

def _call_torch(func: Callable, **kwargs) -> Any:
    if kwargs["dim"] is None:
        kwargs.pop("dim")
        kwargs.pop("keepdim")
    return func(**kwargs)

def masked_softmax(
    input: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None,
    _stacklevel: int = 3,
    dtype: Optional[DType] = None
) -> torch.Tensor:
    """
    Apply `torch.nn.functional.softmax` while some of the elements of input
    tensor being masked.

    Parameters
    ----------
    input : torch.Tensor
        Input

    mask : torch.Tensor
        A 0-1 mask for the input tensor, 0 for tokens that are masked, 1
        for tokens that are not masked. Should be broadcastable with input.
        If None, a regular softmax result will be returned.

    dim : int
        A dimension along which softmax will be computed.

    dtype : torch.dtype, optional
        The desired data type of returned tensor.
    """
    if mask is None:
        return F.softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

    masked_input = _fill_with_mask(input, mask, -float('inf'))
    return F.softmax(masked_input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

def masked_sum(
    input: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[Union[int, List[int], Tuple[int]]] = None,
    keepdim: bool = False,
    dtype: Optional[DType] = None
) -> torch.Tensor:
    """
    Apply `torch.sum` while some of the elements of input tensor being masked.

    Parameters
    ----------
    input : torch.Tensor
        Input

    mask : torch.Tensor
        A 0-1 mask for the input tensor, 0 for tokens that are masked, 1
        for tokens that are not masked. Should be broadcastable with input.
        If None, a regular sum result will be returned.

    dim : int or List[int] or Tuple[int], optional
        A dimension or list of dimensions along which sum will be computed.
        If not specified, returns the masked sum of all elements in the input
        tensor.

    keepdim : bool, optional, default=False
        Whether the output tensor has dim retained or not.

    dtype : torch.dtype, optional
        The desired data type of returned tensor.
    """
    if mask is None:
        return _call_torch(input.sum, dim=dim, keepdim=keepdim, dtype=dtype)

    masked_input = _fill_with_mask(input, mask, 0.)
    return _call_torch(masked_input.sum, dim=dim, keepdim=keepdim, dtype=dtype)

def masked_mean(
    input: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[Union[int, List[int], Tuple[int]]] = None,
    keepdim: bool = False,
    dtype: Optional[DType] = None
) -> torch.Tensor:
    """
    Apply `torch.mean` while some of the elements of input tensor being masked.

    Parameters
    ----------
    input : torch.Tensor
        Input

    mask : torch.Tensor
        A 0-1 mask for the input tensor, 0 for tokens that are masked, 1
        for tokens that are not masked. Should be broadcastable with input.
        If None, a regular mean result will be returned.

    dim : int or List[int] or Tuple[int], optional
        A dimension or list of dimensions along which mean will be computed.
        If not specified, returns the masked mean of all elements in the input
        tensor.

    keepdim : bool, optional, default=False
        Whether the output tensor has dim retained or not.

    dtype : torch.dtype, optional
        The desired data type of returned tensor.
    """
    if mask is None:
        return _call_torch(input.mean, dim=dim, keepdim=keepdim, dtype=dtype)

    mask_sum = _call_torch(mask.float().sum, dim=dim, keepdim=keepdim)
    mask_sum = mask_sum.clamp(min=1.).to(dtype)

    return masked_sum(input, mask, dim, keepdim, dtype) / mask_sum

def masked_max(
    input: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None,
    keepdim: bool = False
) -> torch.Tensor:
    """
    Apply `torch.max` while some of the elements of input tensor being masked.

    Parameters
    ----------
    input : torch.Tensor
        Input

    mask : torch.Tensor
        A 0-1 mask for the input tensor, 0 for tokens that are masked, 1
        for tokens that are not masked. Should be broadcastable with input.
        If None, a regular max result will be returned.

    dim : int, optional
        A dimension along which max will be computed. If not specified, returns
        the maximum value of all unmasked elements in the input tensor. If
        specified, returns a namedtuple (values, indices) where values is the
        maximum value of unmasked elements in each row of the input tensor in
        the given dimension dim, and indices is the index location of each
        maximum value found (argmax).

    keepdim : bool, optional, default=False
        Whether the output tensor has dim retained or not.
    """
    if mask is None:
        return _call_torch(input.max, dim=dim, keepdim=keepdim)

    masked_input = _fill_with_mask(input, mask, -float('inf'))
    return _call_torch(masked_input.max, dim=dim, keepdim=keepdim)

def masked_min(
    input: torch.Tensor,
    mask: torch.Tensor,
    dim: Optional[int] = None,
    keepdim: bool = False
) -> torch.Tensor:
    """
    Apply `torch.min` while some of the elements of input tensor being masked.

    Parameters
    ----------
    input : torch.Tensor
        Input

    mask : torch.Tensor
        A 0-1 mask for the input tensor, 0 for tokens that are masked, 1
        for tokens that are not masked. Should be broadcastable with input.
        If None, a regular min result will be returned.

    dim : int, optional
        A dimension along which min will be computed. If not specified, returns
        the minimum value of all unmasked elements in the input tensor. If
        specified, returns a namedtuple (values, indices) where values is the
        minimum value of unmasked elements in each row of the input tensor in
        the given dimension dim, and indices is the index location of each
        minimum value found (argmin).

    keepdim : bool, optional, default=False
        Whether the output tensor has dim retained or not.
    """
    if mask is None:
        return _call_torch(input.min, dim=dim, keepdim=keepdim)

    masked_input = _fill_with_mask(input, mask, float('inf'))
    return _call_torch(masked_input.min, dim=dim, keepdim=keepdim)
