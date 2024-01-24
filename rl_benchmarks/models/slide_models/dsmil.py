# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"Dual-Stream MIL aggregation algorithm."

from typing import Optional, List

import torch
from torch import nn

from .utils.extreme_layer import ExtremeLayer
from .utils.mlp import MLP
from .utils.tile_layers import TilesMLP


class DSMIL(nn.Module):
    """
    DS-MIL model (See [1]_).

    Parameters
    ----------
    in_features: int
        Features (model input) dimension.
    out_features: int = 1
        Prediction (model output) dimension.
    d_tiles_values: int = 32
        Dimension of the tile information vectors (v_i in the paper).
    d_tiles_queries: int = 32
        Dimension of the tile query vectors (q_i in the paper).
    passing_values: bool = False
        If true, the tile information vectors (v_i) are the input
        features themselves.
    tiles_scores_mlp_hidden: Optional[List[int]] = None
        Hidden layers of the tiles scores MLP (max-pooling stream).
    tiles_values_mlp_hidden: Optional[List[int]] = None
        Hidden layers of the tiles information MLP (attention stream).
    tiles_queries_mlp_hidden: Optional[List[int]] = None
        Hidden layers of the tiles query MLP (attention stream).
    mlp_hidden: Optional[List[int]] = None
        Hidden layers of the mlp that computes the bag score.
    mlp_dropout: Optional[List[float]] = None
        Dropout of the mlp that computes the bag score.
    mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid()
        Activation of the mlp that computes the bag score.
    bias: bool = True
        If True, uses bias in the MLPs.
    metadata_cols: int = 3
        Number of metadata columns (for example, magnification, patch start
        coordinates etc.) at the start of input data. Default of 3 assumes 
        that the first 3 columns of input data are, respectively:
        1) Deep zoom level, corresponding to a given magnification
        2) input patch starting x value 
        3) input patch starting y value 
    out_logits: str = "mean"
        Score to return: "max", "bag" or "mean" (the mean of the two).

    References
    ----------
    .. [1] Bin Li, Yin Li, and Kevin W. Eliceiri. Dual-stream multiple instance learning network for
    whole slide image classification with self-supervised contrastive learning. In 2021 IEEE/CVF
    Conference on Computer Vision and Pattern Recognition (CVPR), pages 14313–14323,
    2021. doi: 10.1109/CVPR46437.2021.01409.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        d_tiles_values: int = 32,
        d_tiles_queries: int = 32,
        passing_values: bool = False,
        tiles_scores_mlp_hidden: Optional[List[int]] = None,
        tiles_values_mlp_hidden: Optional[List[int]] = None,
        tiles_queries_mlp_hidden: Optional[List[int]] = None,
        mlp_hidden: Optional[List[int]] = None,
        mlp_dropout: Optional[List[float]] = None,
        mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
        metadata_cols: int = 3,
        out_logits: str = "mean",
    ):
        super(DSMIL, self).__init__()

        if mlp_dropout is not None:
            if mlp_hidden is not None:
                assert len(mlp_hidden) == len(
                    mlp_dropout
                ), "mlp_hidden and mlp_dropout must have the same length"
            else:
                raise ValueError(
                    "mlp_hidden must have a value and have the same length as mlp_dropout if mlp_dropout is given."
                )

        self.out_logits = out_logits
        self.passing_values = passing_values

        self.tiles_scores = TilesMLP(
            in_features,
            hidden=tiles_scores_mlp_hidden,
            bias=bias,
            out_features=out_features,
        )

        self.extreme_layer = ExtremeLayer(
            n_top=1, n_bottom=0, return_indices=True
        )

        self.tiles_values = TilesMLP(
            in_features,
            hidden=tiles_values_mlp_hidden,
            bias=bias,
            out_features=d_tiles_values,
        )

        self.tiles_queries = TilesMLP(
            in_features,
            hidden=tiles_queries_mlp_hidden,
            bias=bias,
            out_features=d_tiles_queries,
        )

        mlp_in_features = (
            d_tiles_values if self.passing_values else in_features
        )

        self.mlp = MLP(
            mlp_in_features,
            1,
            hidden=mlp_hidden,
            dropout=mlp_dropout,
            activation=mlp_activation,
        )

        self.metadata_cols = metadata_cols

    @staticmethod
    def attention_n_tiles(
        mask: torch.Tensor,
        ind_n_tiles: torch.Tensor,
        tiles_queries: torch.Tensor,
        tiles_values: torch.Tensor,
    ):
        # Gather max tiles queries
        ind_n_tiles = ind_n_tiles.transpose(1, 2)
        dummy = ind_n_tiles.expand(
            ind_n_tiles.size(0), ind_n_tiles.size(1), tiles_queries.size(2)
        )
        queries_n_tiles = torch.gather(tiles_queries, 1, dummy).transpose(
            1, 2
        )  # (B, D_TILES_QUERIES, OUT_FEATURES)

        # Compute tiles attention logits and scores
        tiles_attention_logits = torch.bmm(
            tiles_queries, queries_n_tiles
        )  # (B, N_TILES, OUT_FEATURES)
        tiles_attention_logits = tiles_attention_logits.masked_fill(
            mask, float("-inf")
        )  # Masking
        tiles_attention_logits = tiles_attention_logits.transpose(
            1, 2
        )  # (B, OUT_FEATURES, N_TILES)
        tiles_attention_scores = torch.nn.functional.softmax(
            tiles_attention_logits, dim=2
        )  # (B, OUT_FEATURES, N_TILES)

        # Pool vectors together
        pooled_vectors = torch.bmm(
            tiles_attention_scores, tiles_values
        )  # (B, OUT_FEATURES, D_TILES_VALUES)

        return pooled_vectors, tiles_attention_scores

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, FEATURES)
        mask: Optional[torch.BoolTensor]
            (B, N_TILES, 1), True for values that were padded.

        Returns
        -------
        logits: torch.Tensor
            (B, OUT_FEATURES)
        """

        # Discard preceding metadata columns
        features = x[..., self.metadata_cols:]

        # Max Pooling branch

        tiles_scores = self.tiles_scores(features, mask)
        n_tiles_logits, ind_n_tiles = self.extreme_layer(
            x=tiles_scores, mask=mask
        )
        n_tiles_logits = n_tiles_logits.squeeze()

        # Masked non-local branch

        # Retrieve values and queries
        if self.passing_values:
            tiles_values = self.tiles_values(
                features, mask
            )  # (B, N_TILES, D_TILES_VALUES)
        else:
            tiles_values = features
        tiles_queries = self.tiles_queries(
            features, mask
        )  # (B, N_TILES, D_TILES_VALUES)

        # Attention mechanism with respect to the weights
        pooled_vectors, _ = self.attention_n_tiles(
            mask, ind_n_tiles, tiles_queries, tiles_values
        )

        # MLP on pooled vectors
        bag_logits = self.mlp(pooled_vectors).squeeze()  # (B, OUT_FEATURES)

        # Gather results from the branches

        if self.out_logits == "mean":
            logits = 0.5 * (n_tiles_logits + bag_logits)
        elif self.out_logits == "bag":
            logits = bag_logits
        elif self.out_logits == "max":
            logits = n_tiles_logits

        if logits.ndim == 0:
            return logits.unsqueeze(0).unsqueeze(1)
        return logits.unsqueeze(1)
