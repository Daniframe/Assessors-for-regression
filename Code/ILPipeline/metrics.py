import numpy as np
from numpy.typing import ArrayLike

def error(y_true: ArrayLike, y_pred: ArrayLike) -> ArrayLike:
    return y_true - y_pred

def squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> ArrayLike:
    return error(y_true, y_pred)**2

def absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> ArrayLike:
    return np.abs(error(y_true, y_pred))

def squared_log_error(y_true: ArrayLike, y_pred: ArrayLike) -> ArrayLike:
    return (np.log1p(y_true) - np.log1p(y_pred))**2