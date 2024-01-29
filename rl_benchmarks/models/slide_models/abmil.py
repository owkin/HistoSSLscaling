# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""ABMIL aggregation algorithm."""

from typing import List, Optional

import torch
from torch import nn

from .utils.attention import GatedAttention
from .utils.mlp import MLP
from .utils.tile_layers import TilesMLP


class ABMIL(nn.Module):
    """Attention-based MIL classification model (See [1]_).

    Example:
        >>> module = ABMIL(in_features=128, out_features=1)
        >>> logits, attention_scores = module(slide, mask=mask)
        >>> attention_scores = module.score_model(slide, mask=mask)

    Parameters
    ----------
    in_features: int
        Features (model input) dimension.
    out_features: int = 1
        Prediction (model output) dimension.
    d_model_attention: int = 128
        Dimension of attention scores.
    temperature: float = 1.0
        GatedAttention softmax temperature.
    tiles_mlp_hidden: Optional[List[int]] = None
        Dimension of hidden layers in first MLP.
    mlp_hidden: Optional[List[int]] = None
        Dimension of hidden layers in last MLP.
    mlp_dropout: Optional[List[float]] = None,
        Dropout rate for last MLP.
    mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid
        Activation for last MLP.
    bias: bool = True
        Add bias to the first MLP.
    metadata_cols: int = 3
        Number of metadata columns (for example, magnification, patch start
        coordinates etc.) at the start of input data. Default of 3 assumes 
        that the first 3 columns of input data are, respectively:
        1) Deep zoom level, corresponding to a given magnification
        2) input patch starting x value 
        3) input patch starting y value 

    References
    ----------
    .. [1] Maximilian Ilse, Jakub Tomczak, and Max Welling. Attention-based
    deep multiple instance learning. In Jennifer Dy and Andreas Krause,
    editors, Proceedings of the 35th International Conference on Machine
    Learning, volume 80 of Proceedings of Machine Learning Research,
    pages 2127–2136. PMLR, 10–15 Jul 2018.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        d_model_attention: int = 128,
        temperature: float = 1.0,
        tiles_mlp_hidden: Optional[List[int]] = None,
        mlp_hidden: Optional[List[int]] = None,
        mlp_dropout: Optional[List[float]] = None,
        mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
        metadata_cols: int = 3,
    ) -> None:
        super(ABMIL, self).__init__()

        if mlp_dropout is not None:
            if mlp_hidden is not None:
                assert len(mlp_hidden) == len(
                    mlp_dropout
                ), "mlp_hidden and mlp_dropout must have the same length"
            else:
                raise ValueError(
                    "mlp_hidden must have a value and have the same length"
                    "as mlp_dropout if mlp_dropout is given."
                )

        self.tiles_emb = TilesMLP(
            in_features,
            hidden=tiles_mlp_hidden,
            bias=bias,
            out_features=d_model_attention,
        )

        self.attention_layer = GatedAttention(
            d_model=d_model_attention, temperature=temperature
        )

        mlp_in_features = d_model_attention

        self.mlp = MLP(
            in_features=mlp_in_features,
            out_features=out_features,
            hidden=mlp_hidden,
            dropout=mlp_dropout,
            activation=mlp_activation,
        )

        self.metadata_cols = metadata_cols
    def score_model(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """Get attention logits.

        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, FEATURES)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.

        Returns
        -------
        attention_logits: torch.Tensor
            (B, N_TILES, 1)
        """
        tiles_emb = self.tiles_emb(x, mask)
        attention_logits = self.attention_layer.attention(tiles_emb, mask)
        return attention_logits

    def forward(
        self, features: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features: torch.Tensor
            (B, N_TILES, D+3)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.

        Returns
        -------
        logits, attention_weights: Tuple[torch.Tensor, torch.Tensor]
            (B, OUT_FEATURES), (B, N_TILES)
        """
        tiles_emb = self.tiles_emb(features[..., self.metadata_cols:], mask)
        scaled_tiles_emb, _ = self.attention_layer(tiles_emb, mask)
        logits = self.mlp(scaled_tiles_emb)

        return logits
