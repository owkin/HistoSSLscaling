# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""TransMIL aggregation algorithm."""

from math import ceil, sqrt, modf
from typing import Union, Optional, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.modules import Module

from .utils.attention import SelfAttention, NystromAttention
from ..feature_extractors.encoders.weight_init import trunc_normal_


def _ensures_is_3d(x: Union[np.ndarray, torch.Tensor]):
    """Ensures that a tensor (or an array) is 3d.

    Parameters
    ----------
    x : Union[np.ndarray, torch.Tensor]
        Input tensor or array.
    """
    if not x.ndim == 3:
        raise ValueError("Expected a 3d tensor or array.")


def _weight_init(module: Module, name: str = "", head_bias: float = 0.0):
    """Weight Initialization."""
    if isinstance(module, nn.Linear):
        if name.startswith("head"):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


class _Block(Module):
    """TransformerEncoderLayer-like module.

    Parameters
    ----------
    in_features : int
        Number of input features.

    num_attention_heads : int = 8
        Number of attention heads. Should be an integer greater or equal to 8.

    attention_type : str = 'original'
        Implementation of the self-attention. Valid values are: `'original'`
        (classical implementation of the self-attention used in
        TransformerEncoderLayers ; see also [1]_ and [2]_), `'nystrom'`
        (see [3]_ which implements the Nyström approximation of self-attention
        proposed in [4]_).

    qkv_bias : bool = False
        Whether to add bias to the linear projection for query, key and value.

    attn_dropout : float = 0
        Dropout rate (applied before the multiplication with the values).
        Used only if `attention_type = 'original'`.

    proj_dropout : float = 0
        Dropout rate (applied after the multiplication with the values).

    mlp_multiplier : Union[int, float] = 4
        Multiplier which defines the output dimension of the MLP layer. Output
        dim is defined as: `out_features = int(in_features * mlp_multiplier)`.
        By default, the output dimension of the MLP layer will be 4 times the
        number of input features.

    mlp_dropout : float = 0
        Dropout rate (for the MLP layer).

    mlp_activation : Module = nn.GELU
        Activation for the MLP layer.

    num_landmarks : int = 256
        Dimension of landmarks used to approximate the matrix multiplication
        query-key (QK^T) in the Nyström method. When `nys_num_landmarks` is
        small, the approximation of the self-attention with the Nyström method
        scales linearly with the length of the input sequence. Unused when
        `attention_type = 'original'`.

    residual_kernel_size : int = 33
        Kernel size for the 2D depthwise convolution used in the skip
        connexion of value V (to help convergence of Nyström approximation).
        Unused when `attention_type = 'original'`.

    References
    ----------
    .. [1] https://pytorch.org/docs/stable/generated/
    torch.nn.TransformerEncoderLayer.html (Modified BSD Clause)

    .. [2] https://github.com/rwightman/pytorch-image-models (Apache 2.0)

    .. [3] https://github.com/mlpen/Nystromformer/tree/main (MIT License)

    .. [4] Xiong et al. (2021). Nyströmformer: A Nyström-based Algorithm
    for Approximating Self-Attention.)
    """

    def __init__(
        self,
        in_features: int,
        num_attention_heads: int = 8,
        attention_type: str = "original",
        qkv_bias: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        mlp_multiplier: Union[int, float] = 4,
        mlp_dropout: float = 0.0,
        mlp_activation: Module = nn.GELU,
        num_landmarks: int = 256,
        residual_kernel_size: int = 33,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_attention_heads = num_attention_heads
        self.attention_type = attention_type
        self.qkv_bias = qkv_bias
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout
        self.mlp_multiplier = mlp_multiplier
        self.mlp_dropout = mlp_dropout
        self.mlp_activation = mlp_activation
        self.num_landmarks = num_landmarks
        self.residual_kernel_size = residual_kernel_size

        self.__build()

    def __build(self):
        """Build the `_Block` layer."""
        # Self-Attention.
        if self.attention_type not in ["original", "nystrom"]:
            raise ValueError(
                "Got an invalid attention type. Valid attention "
                "types are: `'original'` and `'nystrom'`."
            )
        elif self.attention_type == "original":
            self.attn = SelfAttention(
                in_features=self.in_features,
                num_heads=self.num_attention_heads,
                qkv_bias=self.qkv_bias,
                attn_dropout=self.attn_dropout,
                proj_dropout=self.proj_dropout,
            )
        else:
            self.attn = NystromAttention(
                in_features=self.in_features,
                num_heads=self.num_attention_heads,
                qkv_bias=self.qkv_bias,
                attn_dropout=None,
                proj_dropout=self.proj_dropout,
                num_landmarks=self.num_landmarks,
                residual=True,
                residual_kernel_size=self.residual_kernel_size,
            )

        # Layer norm.
        self.norm1 = nn.LayerNorm(self.in_features)

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
        x = x + self.attn(self.norm1(x))
        # out = x + self.mlp(self.norm2(x))
        return x


class _PPEG(Module):
    """Pyramid Position Encoding Generator (PPEG).

    Parameters
    ----------
    in_features : int
        Number of input features.
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.in_features = in_features

        self.__build()

    def __build(self):
        """Build the `_PPEG` layer."""
        # Depthwise convolutions with kernel sizes 3, 5 and 7:
        self.group_conv1 = nn.Conv2d(
            in_channels=self.in_features,
            out_channels=self.in_features,
            kernel_size=3,
            padding=1,
            stride=1,
            dilation=1,
        )
        self.group_conv2 = nn.Conv2d(
            in_channels=self.in_features,
            out_channels=self.in_features,
            kernel_size=5,
            padding=2,
            stride=1,
            dilation=1,
        )
        self.group_conv3 = nn.Conv2d(
            in_channels=self.in_features,
            out_channels=self.in_features,
            kernel_size=7,
            padding=3,
            stride=1,
            dilation=1,
        )

    @staticmethod
    def _split(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Splits the input tensor into class token and patchs tokens.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (B, n_tiles + 1, n_features).

        Returns
        -------
        cls_token : torch.Tensor
            Class token, shape (B, 1, n_features).

        tiles_tokens : torch.Tensor
            tiles tokens, shape (B, n_tiles, n_features).
        """
        cls_token = torch.unsqueeze(x[:, 0, :], dim=1)
        tiles_tokens = x[:, 1:, :]
        return cls_token, tiles_tokens

    @staticmethod
    def _restore(x: torch.Tensor) -> torch.Tensor:
        """Reshapes a sequence into two dimensions.

        Parameters
        ----------
        x : torch.Tensor, shape (B, n_tiles, n_features)
            Input tensor. The number of tiles `n_tiles` must be a square
            (i.e. the square of an integer). If the square root of `n_tiles`
            is not an integer, an error will be raised.

        Returns
        -------
        out : torch.Tensor, shape (B, out_dim, out_dim, n_features)
            Output tensor. The output dimension `out_dim` is equal to
            `sqrt(n_tiles)`.
        """
        _, n_tiles, n_features = x.shape
        _fpart, _ipart = modf(sqrt(n_tiles))
        if not _fpart == 0:
            raise ValueError(
                f"Expected a tensor with shape `(B, N, F)` such "
                f"that sqrt(N) is an integer. Got N = {n_tiles} "
                f"and sqrt(N) has a non-zero fractional part."
            )
        else:
            out_dim = int(_ipart)
            out = x.view((-1, out_dim, out_dim, n_features))
            return out

    @staticmethod
    def _flatten(x: torch.Tensor) -> torch.Tensor:
        """Flattens an input tensor along its 'inner' dimensions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (B, [N1,..., Nd], n_features).

        Returns
        -------
        out : torch.Tensor
            Output tensor, shape (B, N1 x ... x Nd, n_features).
        """
        batch_size, n_features = x.shape[0], x.shape[-1]
        out = x.view((batch_size, -1, n_features))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor, shape (B, n_tiles, d_model).

        Returns
        -------
        out : torch.Tensor
            Output tensor.
        """
        # 1. Split cls/tiles tokens:
        cls_token, tiles_tokens = self._split(x)
        # x.shape = [batch_size, ntiles + 1, proj_dim]
        # cls_token of shape [batch_size, 1, proj_dim]
        # tiles_tokens of shape [batch_size, ntiles, proj_dim]

        # 2. Reshape tiles tokens:
        x = self._restore(tiles_tokens)
        x = x.permute((0, 3, 1, 2))

        # 3. Group convolutions:
        fmap1 = self.group_conv1(x)
        fmap2 = self.group_conv2(x)
        fmap3 = self.group_conv3(x)

        # 4. Fusion:
        x = x + fmap1 + fmap2 + fmap3

        # 5. Flatten:
        x = x.permute((0, 2, 3, 1))
        x = self._flatten(x)

        # 6. Concatenate with cls token:
        out = torch.cat((cls_token, x), dim=1)

        return out


class TransMIL(Module):
    """TransMIL module.

    Implements the model proposed in [1]_. The TransMIL model is composed of
    the following steps: 1. Sequence squaring ; 2. Correlation modelling ;
    3. Position encoding (Pyramid Position Encoding Generator) ; 4. Deep
    Feature Aggregation ; 6. Classification (see Figure 3 and Algorithm 2
    in [1]_).

    Parameters
    ----------
    in_features : int
        Number of input features.

    proj_dim : Optional[int], default=None
        If not None, `in_features` will be reduced (by linear projection) to
        `proj_dim` features. By default, no projection is applied.

    proj_dropout : Optional[float], default=None
        If not None, dropout rate for the dropout layer which follows the
        initial linear projection. By default, no dropout is applied.

    out_features : int, default=1
        Number of output features (logits).

    kw_cor : Optional[Dict], default=None
        Keyword arguments for the correlation modelling module. The `kw_cor`
        parameter must be a dict whose keys are parameters of the `_Block`
        module and values are valid parameter values. For instance,
        `kw_cor = {'num_attention_heads': 8}` can be used to change the number
        of attention heads in the self-attention layer of the correlation
        modelling block.

    kw_agg : Optional[Dict], default=None
        Keyword arguments for the deep feature aggregation module. The `kw_agg`
        parameter must be a dict whose keys are parameters of the `_Block`
        module and values are valid parameter values. For instance,
        `kw_agg = {'num_attention_heads': 8}` can be used to change the number
        of attention heads in the self-attention layer of the deep feature
        aggregation block.

    metadata_cols: int = 3
        Number of metadata columns (for example, magnification, patch start
        coordinates etc.) at the start of input data. Default of 3 assumes 
        that the first 3 columns of input data are, respectively:
        1) Deep zoom level, corresponding to a given magnification
        2) input patch starting x value 
        3) input patch starting y value 

    References
    ----------
    .. [1] Shao et al. (2021). TransMIL: Transformer based Correlated Multiple
           Instance Learning for Whole Slide Image Classification.
    """

    def __init__(
        self,
        in_features: int,
        position_encoding: str = "PPEG",
        # Needs to be 'PPEG'
        proj_dim: Optional[int] = None,
        proj_dropout: Optional[float] = None,
        out_features: int = 1,
        kw_cor: Optional[Dict] = None,
        kw_agg: Optional[Dict] = None,
        metadata_cols: int = 3,
    ):
        super().__init__()

        self.position_encoding = position_encoding.upper()
        position_encoding_options = ["PPEG"]
        if self.position_encoding not in position_encoding_options:
            raise ValueError(
                f"Position_encoding should be in {position_encoding_options}"
            )

        self.in_features = in_features
        self.proj_dim = proj_dim
        self.proj_dropout = proj_dropout
        self.out_features = out_features
        self.kw_cor = kw_cor
        self.kw_agg = kw_agg

        self.metadata_cols = metadata_cols

        self.__build()

        # Weight initialization:
        self.apply(_weight_init)

    def __build(self):
        """Build the TransMIL module."""
        # Linear projection (to reduce the input feature space):
        if self.proj_dim is not None:
            inner_dim = int(self.proj_dim)
            _proj_dropout = (
                0.0 if self.proj_dropout is None else float(self.proj_dropout)
            )
            self.proj = nn.Sequential(
                nn.Linear(
                    in_features=self.in_features,
                    out_features=inner_dim,
                    bias=True,
                ),
                nn.Dropout(_proj_dropout),
            )
        else:
            inner_dim = int(self.in_features)
            self.proj = nn.Identity()

        # Class token:
        self.cls_token = nn.Parameter(torch.zeros(1, 1, inner_dim))

        # Correlation modelling:
        _kw_cor = {} if self.kw_cor is None else self.kw_cor
        self.cor = _Block(in_features=inner_dim, **_kw_cor)

        # Positional Encoding (PPEG):
        self.pos_enc = _PPEG(in_features=inner_dim)

        # Deep Feature Aggregation:
        _kw_agg = {} if self.kw_agg is None else self.kw_agg
        self.agg = _Block(in_features=inner_dim, **_kw_agg)

        # Classification:
        self.norm = nn.LayerNorm(inner_dim)
        self.head = nn.Linear(inner_dim, self.out_features, bias=True)

    def _add_cls_token(self, inp_seq: torch.Tensor) -> torch.Tensor:
        """Add class token at the beginning of the input sequence.

        Parameters
        ----------
        inp_seq : torch.Tensor
            Input tensor, shape (B, n_tiles, n_features).

        Returns
        -------
        out_seq : torch.Tensor
            Input tensor with class token added at the beginning,
            shape (B, n_tiles + 1, n_features).
        """
        cls_token = self.cls_token.expand(inp_seq.shape[0], -1, -1)
        out_seq = torch.cat((cls_token, inp_seq), dim=1)
        return out_seq

    @staticmethod
    def _squaring(inp_seq: torch.Tensor) -> torch.Tensor:
        """Sequence squaring.

        Ensures that the number of elements (or tokens) in the output
        sequence is a square (i.e. its square root is an integer).

        Parameters
        ----------
        inp_seq : torch.Tensor
            Input sequence, shape (B, in_len, in_features).

        Returns
        -------
        out_seq : torch.Tensor
            Shape (B, out_len, in_features).
            Output sequence where `out_seq_len` is such that:
            `L = sqrt(ceil(in_len))` and `out_len = L ** 2 - in_seq_len`.
        """
        in_seq_len = inp_seq.shape[1]
        sqrt_N = ceil(sqrt(in_seq_len))
        M = sqrt_N**2 - in_seq_len
        add_seq = inp_seq[:, :M, :]
        out_seq = torch.cat((inp_seq, add_seq), dim=1)
        return out_seq

    def forward(
        self, features: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features: torch.Tensor
            (B, N_TILES, FEATURES)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.

        Returns
        -------
        logits: torch.Tensor
            (B, OUT_FEATURES)
        """
        x = features[..., self.metadata_cols:]

        # Check input:
        _ensures_is_3d(x)

        # (Optional) Projection:
        x = self.proj(x)

        # 1. Sequence squaring:
        x = self._squaring(x)
        x = self._add_cls_token(x)

        # 2. Correlation modelling:
        x = self.cor.forward(x)

        # 3. Pyramid Position Encoding Generator:
        # The PPEG module does not use the tiles locations
        x = self.pos_enc.forward(x)

        # 4. Deep Feature Aggregation:
        x = self.agg.forward(x)

        # 5. Use class token for classification:
        _cls_token = x[:, 0, :]
        xx = self.norm(_cls_token)
        logits = self.head(xx)

        return logits
