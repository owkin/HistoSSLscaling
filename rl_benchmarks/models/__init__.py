# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Module covering feature extractors and Multiple Instance Learning (MIL)
algorithms (slide-classification tasks)."""

from .feature_extractors.core import Extractor, ParallelExtractor
from .feature_extractors.ibot_vit import iBOTViT
from .slide_models.chowder import Chowder
from .slide_models.abmil import ABMIL
from .slide_models.dsmil import DSMIL
from .slide_models.hiptmil import HIPTMIL
from .slide_models.meanpool import MeanPool
from .slide_models.transmil import TransMIL
