from __future__ import annotations
from typing import Optional, Union, Iterable, Callable, List

from sklearn.base import BaseEstimator

class RegressionFlow:

    """
    Docs go here
    """

    def __init__(
        self,
        name: str,
        model: BaseEstimator,
        prior_transformations: Optional[List[Union[str, Callable]]] = None,
        random_state: Optional[int] = 8888,
        description: Optional[str] = None):

        """
        Docs go here
        """

        self.name = name
        self.description = description
        self.model = model
        self.prior_transformations = prior_transformations

        self.hyperparams = model.get_params()

        if "random_state" not in self.hyperparams or self.hyperparams["random_state"] is None:

            print(f"Random state not found in model. Defaulting to {random_state}")

            self.hyperparams["random_state"] = random_state
            self.model.set_params(random_state = random_state)