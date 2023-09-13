# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for attention mechanisms."""

from math import ceil
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.nn.modules import Module

from einops import rearrange, reduce

from .tile_layers import MaskedLinear


def _moore_penrose_iter_pinv(x: torch.Tensor, iters: int = 6):
    """Compute the Moore-Penrose pseudo-inverse of a tensor [1]_.

    Parameters
    ----------
    x: torch.Tensor
        Input tensor.
    iters: int = 6
        Number of iterations for Moore-Penrose algorithm.

    References
    ----------
    .. [1] G. Strang. "Linear Algebra and Its Applications, 2nd Ed."
           Academic Press, Inc., 1980, pp. 139-142.
    """
    device = x.device
    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, "... i j -> ... j i") / (torch.max(col) * torch.max(row))

    id_arr = torch.eye(x.shape[-1], device=device)
    id_arr = rearrange(id_arr, "i j -> () i j")

    for _ in range(iters):
        xz = x @ z
        z = (
            0.25
            * z
            @ (13 * id_arr - (xz @ (15 * id_arr - (xz @ (7 * id_arr - xz)))))
        )

    return z


class NystromAttention(Module):
    """Nyström approximation for the Multi-Head Self-Attention.

    This code is derived from the nystrom-attention library:
    ``nystrom-attention``: https://github.com/mlpen/Nystromformer/tree/main (MIT License)

    Parameters
    ----------
    in_features : int
        Number of input features.

    num_heads : int = 8
        Number of attention heads. Should be an integer greater or equal to 1.

    qkv_bias : bool = False
        Whether to add a bias to the linear projection for query, key and value.

    num_landmarks : int = 256
        Dimension of the landmarks used to approximate the matrix multiplication
        query-key (QK^T) in the Nyström method. When `nys_num_landmarks` is small,
        the approximation of the self-attention with the Nyström method scales
        linearly with the length of the input sequence.

    pinv_iterations : int = 6
        Number of iterations for the iterative Moore-Penrose pseudoinverse
        approximation.

    residual : bool = True
        Whether to implement a skip connexion for values V (with a depthwise
        convolution). See also the `residual_kernel_size` parameter. Defaults
        to True.

    residual_kernel_size : int = 33
        Kernel size for the 2D depthwise convolution used in the skip
        connexion of value V (to help convergence of the Nyström approximation).

    attn_dropout : Optional[float] = None
        Unused. For compatibility with the `SelfAttention` module.

    proj_dropout : float = 0
        Dropout rate (applied after the multiplication with the values).
    """

    def __init__(
        self,
        in_features: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        num_landmarks: int = 256,
        pinv_iterations: int = 6,
        residual: bool = True,
        residual_kernel_size: int = 33,
        attn_dropout: Optional[float] = None,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        self.residual = residual
        self.residual_kernel_size = residual_kernel_size
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout

        self.__build()

    def __build(self):
        """Build the `NystromAttention` module."""
        head_dim = self.in_features // self.num_heads
        self.scale = head_dim**-0.5
        self.to_qkv = nn.Linear(
            self.in_features, self.in_features * 3, bias=self.qkv_bias
        )
        self.to_out = nn.Sequential(
            nn.Linear(self.in_features, self.in_features),
            nn.Dropout(self.proj_dropout),
        )
        if self.residual:
            _padding = (self.residual_kernel_size // 2, 0)
            self.res_conv = nn.Conv2d(
                in_channels=self.num_heads,
                out_channels=self.num_heads,
                kernel_size=(self.residual_kernel_size, 1),
                padding=_padding,
                groups=self.num_heads,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (B, seq_len, in_features).

        Returns
        -------
        out : torch.Tensor
            Output tensor, shape (B, seq_len, in_features).
        """
        _, n, _, h, m, iters = (
            *x.shape,
            self.num_heads,
            self.num_landmarks,
            self.pinv_iterations,
        )

        # Pad so that sequence can be evenly divided into m landmarks
        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value=0)

        # Derive query, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v)
        )
        q = q * self.scale

        # Generate landmarks by sum reduction, and then calculate mean using the mask
        landmark_einops_eq = "... (n l) d -> ... n d"
        q_landmarks = reduce(q, landmark_einops_eq, "sum", l=ceil(n / m))
        k_landmarks = reduce(k, landmark_einops_eq, "sum", l=ceil(n / m))
        q_landmarks /= ceil(n / m)
        k_landmarks /= ceil(n / m)

        # Similarities
        einops_eq = "... i d, ... j d -> ... i j"
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        # Eq (15) in the paper and aggregate values
        attn1, attn2, attn3 = map(
            lambda t: t.softmax(dim=-1), (sim1, sim2, sim3)
        )
        attn2_inv = _moore_penrose_iter_pinv(attn2, iters)
        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # Add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)

        # Merge and combine heads
        out = rearrange(out, "b h n d -> b n (h d)", h=h)
        out = self.to_out(out)
        out = out[:, -n:]
        return out


class SelfAttention(Module):
    """Multi-Head Self-Attention.

    Implementation adapted from https://github.com/rwightman/pytorch-image-models.

    Parameters
    ----------
    in_features : int
        Number of input features.

    num_heads : int = 8
        Number of attention heads. Should be an integer greater or equal to 1.

    qkv_bias : bool = False
        Whether to add a bias to the linear projection for query, key and value.

    attn_dropout : float = 0.0
        Dropout rate (applied before the multiplication with the values).

    proj_dropout : float = 0.0
        Dropout rate (applied after the multiplication with the values).
    """

    def __init__(
        self,
        in_features: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout

        self.__build()

    def __build(self):
        """Build the `SelfAttention` module."""
        head_dim = self.in_features // self.num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(
            self.in_features, self.in_features * 3, bias=self.qkv_bias
        )
        self.attn_drop = nn.Dropout(self.attn_dropout)
        self.proj = nn.Linear(self.in_features, self.in_features)
        self.proj_drop = nn.Dropout(self.proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (B, seq_len, in_features).

        Returns
        -------
        out : torch.Tensor
            Output tensor, shape (B, seq_len, in_features).
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GatedAttention(torch.nn.Module):
    """Gated Attention, as defined in https://arxiv.org/abs/1802.04712.
    Permutation invariant Layer on dim 1.
    Parameters
    ----------
    d_model: int = 128
    temperature: float = 1.0
        Attention Softmax temperature
    """

    def __init__(
        self,
        d_model: int = 128,
        temperature: float = 1.0,
    ):
        super(GatedAttention, self).__init__()

        self.att = torch.nn.Linear(d_model, d_model)
        self.gate = torch.nn.Linear(d_model, d_model)
        self.w = MaskedLinear(d_model, 1, "-inf")

        self.temperature = temperature

    def attention(
        self,
        v: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Gets attention logits.
        Parameters
        ----------
        v: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, SEQ_LEN, 1), True for values that were padded.
        Returns
        -------
        attention_logits: torch.Tensor
            (B, N_TILES, 1)
        """

        h_v = self.att(v)
        h_v = torch.tanh(h_v)

        u_v = self.gate(v)
        u_v = torch.sigmoid(u_v)

        attention_logits = self.w(h_v * u_v, mask=mask) / self.temperature
        return attention_logits

    def forward(
        self, v: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        Parameters
        ----------
        v: torch.Tensor
            (B, SEQ_LEN, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, SEQ_LEN, 1), True for values that were padded.
        Returns
        -------
        scaled_attention, attention_weights: Tuple[torch.Tensor, torch.Tensor]
            (B, IN_FEATURES), (B, N_TILES, 1)
        """
        attention_logits = self.attention(v=v, mask=mask)

        attention_weights = torch.softmax(attention_logits, 1)
        scaled_attention = torch.matmul(attention_weights.transpose(1, 2), v)

        return scaled_attention.squeeze(1), attention_weights
