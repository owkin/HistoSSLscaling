# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Extreme layer implementation for Chowder and DSMIL."""

import warnings
from typing import Optional, Tuple, Union

import torch


class ExtremeLayer(torch.nn.Module):
    """Extreme layer.
    Returns concatenation of n_top top tiles and n_bottom bottom tiles
    .. warning::
        If top tiles or bottom tiles is superior to the true number of
        tiles in the input then padded tiles will be selected and their value
        will be 0.
    Parameters
    ----------
    n_top: Optional[int] = None
        Number of top tiles to select
    n_bottom: Optional[int] = None
        Number of bottom tiles to select
    dim: int = 1
        Dimension to select top/bottom tiles from
    return_indices: bool = False
        Whether to return the indices of the extreme tiles

    Raises
    ------
    ValueError
        If ``n_top`` and ``n_bottom`` are set to ``None`` or both are 0.
    """

    def __init__(
        self,
        n_top: Optional[int] = None,
        n_bottom: Optional[int] = None,
        dim: int = 1,
        return_indices: bool = False,
    ):
        super(ExtremeLayer, self).__init__()

        if not (n_top is not None or n_bottom is not None):
            raise ValueError("one of n_top or n_bottom must have a value.")

        if not (
            (n_top is not None and n_top > 0)
            or (n_bottom is not None and n_bottom > 0)
        ):
            raise ValueError("one of n_top or n_bottom must have a value > 0.")

        self.n_top = n_top
        self.n_bottom = n_bottom
        self.dim = dim
        self.return_indices = return_indices

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        Parameters
        ----------
        x: torch.Tensor
            Input tensor, shape (B, N_TILES, IN_FEATURES).
        mask: Optional[torch.BoolTensor]
            True for values that were padded, shape (B, N_TILES, 1).

        Warnings
        --------
        If top tiles or bottom tiles is superior to the true number of tiles in
        the input then padded tiles will be selected and their value will be 0.

        Returns
        -------
        values: torch.Tensor
            Extreme tiles, shape (B, N_TOP + N_BOTTOM).
        indices: torch.Tensor
            If ``self.return_indices=True``, return extreme tiles' indices.
        """

        if (
            self.n_top
            and self.n_bottom
            and ((self.n_top + self.n_bottom) > x.shape[self.dim])
        ):
            warnings.warn(
                f"Sum of tops is larger than the input tensor shape for dimension {self.dim}: "
                f"{self.n_top + self.n_bottom} > {x.shape[self.dim]}. "
                f"Values will appear twice (in top and in bottom)"
            )

        top, bottom = None, None
        top_idx, bottom_idx = None, None
        if mask is not None:
            if self.n_top:
                top, top_idx = x.masked_fill(mask, float("-inf")).topk(
                    k=self.n_top, sorted=True, dim=self.dim
                )
                top_mask = top.eq(float("-inf"))
                if top_mask.any():
                    warnings.warn(
                        "The top tiles contain masked values, they will be set to zero."
                    )
                    top[top_mask] = 0

            if self.n_bottom:
                bottom, bottom_idx = x.masked_fill(mask, float("inf")).topk(
                    k=self.n_bottom, largest=False, sorted=True, dim=self.dim
                )
                bottom_mask = bottom.eq(float("inf"))
                if bottom_mask.any():
                    warnings.warn(
                        "The bottom tiles contain masked values, they will be set to zero."
                    )
                    bottom[bottom_mask] = 0
        else:
            if self.n_top:
                top, top_idx = x.topk(k=self.n_top, sorted=True, dim=self.dim)
            if self.n_bottom:
                bottom, bottom_idx = x.topk(
                    k=self.n_bottom, largest=False, sorted=True, dim=self.dim
                )

        if top is not None and bottom is not None:
            values = torch.cat([top, bottom], dim=self.dim)
            indices = torch.cat([top_idx, bottom_idx], dim=self.dim)
        elif top is not None:
            values = top
            indices = top_idx
        elif bottom is not None:
            values = bottom
            indices = bottom_idx
        else:
            raise ValueError

        if self.return_indices:
            return values, indices
        else:
            return values

    def extra_repr(self) -> str:
        """Format representation."""
        return f"n_top={self.n_top}, n_bottom={self.n_bottom}"
