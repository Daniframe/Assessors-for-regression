from __future__ import annotations
from typing import List, Dict, Tuple, Union, Optional, Literal, Iterable, Callable

from sklearn.model_selection import (
    KFold,
    RepeatedKFold,
    StratifiedKFold,
    StratifiedGroupKFold,
    train_test_split
)

from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    mean_absolute_percentage_error,
    median_absolute_error,
    r2_score,
    mean_poisson_deviance,
    mean_gamma_deviance,
    mean_tweedie_deviance,
    mean_pinball_loss,
    d2_absolute_error_score,
    d2_pinball_score,
    d2_tweedie_score
)

import pandas as pd
import numpy as np

from .datasets import Dataset

EVALUATIONS = {
    "regression" : {
        "cv": KFold,
        "rcv" : RepeatedKFold,
        "tt" : train_test_split
    },
    "classification" : {
        "cv": StratifiedKFold,
        "rcv" : StratifiedGroupKFold,
        "tt" : train_test_split
    }
}

METRICS = {
    "explained_variance_score": explained_variance_score,
    "max_error": max_error,
    "mean_absolute_error": mean_absolute_error,
    "mean_squared_error": mean_squared_error,
    "mean_squared_log_error": mean_squared_log_error,
    "mean_absolute_percentage_error": mean_absolute_percentage_error,
    "median_absolute_error": median_absolute_error,
    "r2_score": r2_score,
    "mean_poisson_deviance": mean_poisson_deviance,
    "mean_gamma_deviance": mean_gamma_deviance,
    "mean_tweedie_deviance": mean_tweedie_deviance,
    "mean_pinball_loss": mean_pinball_loss,
    "d2_absolute_error_score": d2_absolute_error_score,
    "d2_pinball_score": d2_pinball_score,
    "d2_tweedie_score": d2_tweedie_score
}

class OneTargetTask:

    """
    Docs go here
    """

    def __init__(
        self,
        name: str,
        dataset: Dataset,
        target: Union[str, Dict, pd.DataFrame],
        target_type: Literal["regression", "classification"],
        evaluation_metric: Union[Callable, str] = "mean_squared_error",
        profile_metrics: Optional[Union[List[Callable], List[str]]] = None,
        evaluation_method: str = "cv10",
        evaluation_seed: int = 8888,
        features: Optional[Union[List[str], str, Dict, pd.DataFrame]] = None,
        description: Optional[str] = None,) -> None:

        """
        Docs go here
        """

        self.name = name
        self.description = description
        self.target_type = target_type

        self.evaluation_method = evaluation_method
        self.evaluation_seed = evaluation_seed

        self.dataset = dataset

        if isinstance(target, (Dict, pd.DataFrame)):
            self.target = pd.DataFrame(target)

        else:
            self.target = self.dataset.targets.loc[:, [target]]
        
        if evaluation_metric in METRICS:
            self.evaluation_metric_name = evaluation_metric
            self.evaluation_metric = METRICS[evaluation_metric]
        else:
            self.evaluation_metric_name = evaluation_metric.__name__
            self.evaluation_metric = evaluation_metric

        if features is None:
            self.features = self.dataset.features
        elif isinstance(features, str):
            self.features = self.dataset.features.loc[:, [features]]
        elif isinstance(features, List):
            self.features = self.dataset.features.loc[:, features]
        else:
            self.features = features

        if evaluation_method.startswith("cv"):
            self.evaluation_method_name = "cv"

            self.splitter = EVALUATIONS[self.target_type]["cv"](
                n_splits = int(evaluation_method[2:]),
                shuffle = True,
                random_state = evaluation_seed)
            
        elif evaluation_method.startswith("rcv"):
            self.evaluation_method_name = "rcv"

            self.splitter = EVALUATIONS[self.target_type]["rcv"](
                n_splits = int(evaluation_method[evaluation_method.find("x")+1:]),
                n_repeats = int(evaluation_method[3:evaluation_method.find("x")]),
                random_state = evaluation_seed)
        
        elif evaluation_method.startswith("tt"):
            self.evaluation_method_name = "tt"

            self.splitter = EVALUATIONS[self.target_type]["tt"](
                self.features,
                self.target,
                test_size = float(evaluation_method[2:]) / 100,
                random_state = evaluation_seed)

        else:
            raise NotImplementedError
                