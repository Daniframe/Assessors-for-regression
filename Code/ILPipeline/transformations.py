from __future__ import annotations
from typing import Optional, Union, Iterable, Callable

import pandas as pd
import numpy as np

def categorical_to_numeric(
    data: pd.DataFrame,
    prefix: Optional[str] = None,
    prefix_sep: Union[str, Iterable[str], dict[str, str]] = "_",
    dummy_na: bool = False,
    columns: Optional[Iterable] = None,
    sparse: bool = False,
    drop_first: bool = False,
    dtype: Optional[str] = None) -> pd.DataFrame:

    """
    Docs go here
    """

    return pd.get_dummies(
        data = data,
        prefix = prefix,
        prefix_sep = prefix_sep,
        dummy_na = dummy_na,
        columns = columns,
        sparse = sparse,
        drop_first = drop_first,
        dtype = dtype)