# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Transformer-based aggregation algorithm as introduced in HIPT method
(https://arxiv.org/abs/2206.02647). Code is derived from the original
implementation (https://github.com/mahmoodlab/HIPT/blob/master/2-Weakly-Supervised-Subtyping/models/model_hierarchical_mil.py) from the HIPT repository (Apache 2.0 License with Commons Clause, Mahmood Lab)."""

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class GatedAttentionNetwork(nn.Module):
    """Attention Network with Sigmoid Gating (3 fully-connected layers).

    Parameters
    ----------
    in_features : int = 1024
        Input feature dimension.
    gat_hidden : int = 256
        Hidden layer dimension
    gat_dropout : bool = 0.25
        Dropout rate
    """

    def __init__(
        self,
        in_features: int = 1024,
        gat_hidden: int = 256,
        gat_dropout: float = 0.25,
    ) -> None:
        super(GatedAttentionNetwork, self).__init__()
        self.attention_a = nn.Sequential(
            nn.Linear(in_features, gat_hidden),
            nn.Tanh(),
            nn.Dropout(gat_dropout),
        )
        self.attention_b = nn.Sequential(
            nn.Linear(in_features, gat_hidden),
            nn.Sigmoid(),
            nn.Dropout(gat_dropout),
        )
        self.attention_c = nn.Linear(gat_hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the GatedAttention module.

        Parameters
        ----------
        x: torch.Tensor
            Input tensor, shape (N_TILES, EMBD_DIM).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            ``a``: shape (N_TILES, N_CLASSES)
            ``x``: shape  (N_TILES, EMBD_DIM)
        """
        a, b = self.attention_a(x), self.attention_b(x)
        a = self.attention_c(a.mul(b))
        return a, x


class TransformerModule(nn.Module):
    """Transformer module for tiles features aggregation, following [1]_.

    Parameters
    --------
    in_features: int
        Features (model input) dimension.
    out_features: int = 1
        Prediction (model output) dimension.
    metadata_cols: int = 3
        Number of metadata columns (for example, magnification, patch start
        coordinates etc.) at the start of input data. Default of 3 assumes 
        that the first 3 columns of input data are, respectively:
        1) Deep zoom level, corresponding to a given magnification
        2) input patch starting x value 
        3) input patch starting y value 

    References
    ----------
    .. [1] https://github.com/mahmoodlab/HIPT/blob/master/2-Weakly-Supervised-Subtyping/models/model_hierarchical_mil.py#L156
    """

    def __init__(
        self, in_features: int, out_features: int, metadata_cols: int = 3
    ) -> None:
        super(TransformerModule, self).__init__()
        size = [in_features, 192, 192]

        self.phi = nn.Sequential(
            nn.Linear(in_features, 192), nn.ReLU(), nn.Dropout(0.25)
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=size[1],
                nhead=3,
                dim_feedforward=size[2],
                dropout=0.25,
                activation="relu",
            ),
            num_layers=2,
        )
        self.gat = GatedAttentionNetwork(
            in_features=size[2], gat_hidden=size[1], gat_dropout=0.25
        )

        self.rho = nn.Sequential(
            nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(0.25)
        )
        self.classifier = nn.Linear(size[1], out_features)

        self.metadata_cols = metadata_cols

    def forward(
        self, features: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        features: torch.Tensor
            (1, N_TILES, FEATURES)
        mask: Optional[torch.BoolTensor]
            (1, N_TILES, 1), True for values that were padded.

        Returns
        -------
        logits: torch.Tensor
            (OUT_FEATURES,)
        """
        features = features.squeeze(0)  # (1, N_TILES, IN_FEATURES)
        features = self.phi(features)  # (N_TILES, 192)
        features = self.transformer(features)  # (N_TILES, 192)
        a, h = self.gat(features)  # (N_TILES, N_CLASSES) and (N_TILES, 192)
        a = torch.transpose(a, 1, 0)  # (N_CLASSES, N_TILES)
        a = F.softmax(a, dim=1)  # (N_CLASSES, N_TILES)
        mm = torch.mm(a, h)  # (N_CLASSES, 192)
        h = self.rho(mm)  # (N_CLASSES, 192)
        logits = self.classifier(h)  # (N_CLASSES,)
        return logits


class HIPTMIL(TransformerModule):
    """Main HIPT-MIL aggregation algorithm, as implemented in [1]_.
    .. warning:: Batch size must be 1.

    References
    ----------
    .. [1] Scaling Vision Transformers to Gigapixel Images via Hierarchical
           Self-Supervised Learning, CVPR 2022. Richard. J. Chen, Chengkuan Chen,
           Yicong Li, Tiffany Y. Chen, Andrew D. Trister, Rahul G. Krishnan,
           Faisal Mahmood.
    """

    def __init__(
        self, in_features: int, out_features: int, metadata_cols: int = 3
    ):
        super(HIPTMIL, self).__init__(
            in_features=in_features,
            out_features=out_features,
            metadata_cols=metadata_cols,
        )

    @staticmethod
    def _count_trainable(layer: nn.Module) -> int:
        """Count number of trainable parameters.

        Parameters
        ----------
        layer: torch.nn.Module
            Layer to count trainable parameters from.

        Returns
        -------
        int
            Number of trainable parameters for a given layer.
        """
        n_params = sum(
            p.numel() for p in layer.parameters() if p.requires_grad
        )
        return n_params

    def count_trainable(self) -> Dict[str, int]:
        """Return a dictionary with number of trainable parameters per
        layer in ``HIPTMIL`` module."""
        n_params = {
            "phi": self._count_trainable(self.phi),
            "transformer": self._count_trainable(self.transformer),
            "gat": self._count_trainable(self.gat),
            "rho": self._count_trainable(self.rho),
            "classifier": self._count_trainable(self.classifier),
        }
        return n_params

    def forward(
        self, features: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        .. warning:: Batch size must be 1.

        Parameters
        ----------
        x: torch.Tensor
            (1, N_TILES, FEATURES)
        mask: Optional[torch.BoolTensor]
            (1, N_TILES, 1), True for values that were padded.

        Returns
        -------
        logits: torch.Tensor
            (OUT_FEATURES,)
        """
        return super().forward(features[..., self.metadata_cols:], mask)
