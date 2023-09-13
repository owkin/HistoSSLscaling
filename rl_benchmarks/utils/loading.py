# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Utility function for loading."""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml


def save_pickle(path: Union[str, Path], obj: Any) -> None:
    """Save an object as a .pkl file.
    Parameters
    ----------
    path : str
        Path to save the object.
    obj : Any
        Object to save.
    """
    with open(str(path), "wb") as file:
        pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Union[str, Path]) -> Any:
    """Load a .pkl file.
    Parameters
    ----------
    path : str
        Path to the .pkl object.
    """
    with open(str(path), "rb") as f:
        return pickle.load(f)


def load_yaml(path: Union[str, Path]) -> Dict[Any, Any]:
    """Load a .yaml file.
    Parameters
    ----------
    path : str
        Path to the .yaml configuration
    """
    with open(str(path), "r", encoding="utf-8") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            return exc


def merge_multiple_dataframes(
    dfs: List[pd.DataFrame],
    on: Union[str, List[str], List[List[str]]],
    how: Union[str, List[str]] = None,
    sort: Optional[bool] = False,
    reset_index: Optional[bool] = False,
):
    """Merge multiple DataFrames into one.

    Parameters
    ----------
    dfs : List[pd.DataFrame]
        DataFrames to merge together.

    on : str | List[str] | List[List[str]]
        Column names on which the DataFrames will be merged.
        `on[i]`, which can be a string (one column to merge on) or a
        list of string (>= 2 columns to merge on), specifies the
        column(s) on which to merge dfs[i] and dfs[i+1].

    how : str | List[str], default None ("inner" merge)
        The strategy for merging. `how[i]` specifies how the
        dfs[i] and dfs[i+1] will be merged.

    sort : bool, default = False
        Whether to sort the join keys lexicographically in the result
        DataFrame.

    reset_index: bool, default = False
        Whether to reset the index after merging.

    Returns
    -------
    output_df : pd.DataFrame

    Raises
    ------
    ValueError
        If `how` is a list of string with length != of len(dfs)-1
        If `on` is a list of string with length != of len(dfs)-1
    """
    n_how = n_on = len(dfs) - 1
    if how is None:
        how = ["inner"] * n_how
    elif isinstance(how, str):
        how = [how] * n_how
    else:
        if len(how) != n_how:
            raise ValueError(f"`how` should be a list of length {n_how}")

    if isinstance(on, str):
        on = [on] * n_on
    else:
        if len(how) != n_how:
            raise ValueError(f"`on` should be a list of length {n_on}")

    output_df = dfs[0].copy()
    for i, input_df in enumerate(dfs[1:]):
        _on, _how = on[i], how[i]
        output_df = output_df.merge(input_df, on=_on, how=_how, sort=sort)
    if reset_index:
        output_df = output_df.reset_index(drop=True)
    return output_df
