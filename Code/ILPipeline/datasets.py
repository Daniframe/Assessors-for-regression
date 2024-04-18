from __future__ import annotations
from typing import List, Dict, Tuple, Union, Optional, Literal, Iterable

import pandas as pd
import numpy as np

NUMERIC_TYPES = {
    "int", "int_", 
    "int8", "int16", 
    "int32", "int64", 
    "uint8", "uint16", 
    "uint32", "uint64",
    "float", "float_", 
    "float16", "float32", "float64"
}

class Dataset:

    """
    Docs go here
    """

    def __init__(
        self,
        name: str,
        features: Union[Dict, pd.DataFrame],
        targets: Union[Dict, pd.DataFrame],
        description: Optional[str] = None,
        features_metadata: Optional[Union[Dict, pd.DataFrame]] = None,
        targets_metadata: Optional[Union[Dict, pd.DataFrame]] = None,
        additional_metadata: Optional[Union[str, Dict, pd.DataFrame]] = None) -> None:

        """
        Docs go here
        """

        self.name = name
        self.description = description
        self.features = features
        self.targets = targets

        if features_metadata is None:
            features_metadata = self._get_default_metadata(self.features)

        if targets_metadata is None:
            targets_metadata = self._get_default_metadata(self.targets)

        self.features_metadata = features_metadata
        self.targets_metadata = targets_metadata
        self.additional_metadata = additional_metadata

        self.n_instances = len(self.features)
        self.n_features = len(self.features.columns)
        self.n_targets = len(self.targets.columns)

    def set_features(
        self, 
        features: Union[Dict, pd.DataFrame],
        features_metadata: Optional[Union[Dict, pd.DataFrame]] = None) -> None:

        """
        Docs go here
        """

        self.features = features
        self.n_instances = len(self.features)
        self.n_features = len(self.features.columns)
 
        if features_metadata is None:
            features_metadata = self._get_default_metadata(self.features)

        self.features_metadata = features_metadata

    def set_targets(
        self, 
        targets: Union[Dict, pd.DataFrame],
        targets_metadata: Optional[Union[Dict, pd.DataFrame]] = None) -> None:

        """
        Docs go here
        """

        self.targets = targets
        self.n_targets = len(self.targets.columns)

        if targets_metadata is None:
            targets_metadata = self._get_default_metadata(self.targets)
        
        self.targets_metadata = targets_metadata

    # MOVED TO `transformations`
    # def categorical_to_numeric(
    #     self, 
    #     kind: Literal["features", "targets"] = "features",
    #     prefix: Optional[str] = None,
    #     prefix_sep: Union[str, Iterable[str], dict[str, str]] = "_",
    #     dummy_na: bool = False,
    #     columns: Optional[Iterable] = None,
    #     sparse: bool = False,
    #     drop_first: bool = False) -> pd.DataFrame:

    #     """
    #     Docs go here
    #     """

    #     if kind == "features":
    #         data = self.features
    #     else:
    #         data = self.targets

    #     return pd.get_dummies(
    #             data = data,
    #             prefix = prefix,
    #             prefix_sep = prefix_sep,
    #             dummy_na = dummy_na,
    #             columns = columns,
    #             sparse = sparse,
    #             drop_first = drop_first).astype(int)


    def _get_default_metadata(self, df: pd.DataFrame) -> pd.DataFrame:

        """
        Docs go here
        """

        data = []
        for col in df.columns:
            if df[col].dtype.name in NUMERIC_TYPES:
                data.append(["numeric", df[col].isna().sum()])
            elif df[col].dtype.name in {"bool"}:
                data.append(["boolean", df[col].isna().sum()])
            elif df[col].dtype.name in {"datetime64[ns]"}:
                data.append(["datetime", df[col].isna().sum()])
            else:
                data.append(["categorical", df[col].isna().sum()])

        return pd.DataFrame(
            index = df.columns,
            columns = ["type", "NAs"],
            data = data)