import numpy as np

def diff_to_abs(x: np.array, *args, **kwargs) -> np.array:
    return np.abs(x)

def diff_to_ssq(x: np.array, *args, **kwargs) -> np.array:
    return x**2 * np.sign(x)

def diff_to_sq(x: np.array, *args, **kwargs) -> np.array:
    return x**2

def diff_to_logit(x: np.array, *args, **kwargs) -> np.array:
    inv_tau = np.log(3)/np.mean(np.abs(x))

    return 2/(1+np.exp(-inv_tau*x)) - 1

def diff_to_logitabs(x: np.array, *args, **kwargs) -> np.array:
    inv_tau = np.log(3)/np.mean(np.abs(x))

    return np.abs(2/(1+np.exp(-inv_tau*x)) - 1)

def ssq_to_diff(x: np.array, *args, **kwargs) -> np.array:
    return np.sqrt(np.abs(x)) * np.sign(x)

def ssq_to_sq(x: np.array, *args, **kwargs) -> np.array:
    return np.abs(x)

def ssq_to_abs(x: np.array, *args, **kwargs) -> np.array:
    return np.sqrt(np.abs(x))

def ssq_to_logit(x: np.array, *args, **kwargs) -> np.array:
    diff = ssq_to_diff(x)
    inv_tau = np.log(3)/np.mean(np.abs(diff))

    return 2/(1+np.exp(-inv_tau*diff)) - 1

def ssq_to_logitabs(x: np.array, *args, **kwargs) -> np.array:
    diff = ssq_to_diff(x)
    inv_tau = np.log(3)/np.mean(np.abs(diff))

    return np.abs(2/(1+np.exp(-inv_tau*diff)) - 1)

def logit_to_logitabs(x: np.array, *args, **kwargs) -> np.array:
    return np.abs(x)

def logit_to_diff(x: np.array, *args, **kwargs) -> np.array:

    x[x >= 1] = 1 - 1e-5
    x[x <= -1] = -1 + 1e-5

    return np.log((1-x)/(x+1)) * 1/kwargs["inv_tau"]

def logit_to_abs(x: np.array, *args, **kwargs) -> np.array:

    x[x >= 1] = 1 - 1e-5
    x[x <= -1] = 1 + 1e-5

    return np.abs(np.log((1-x)/(x+1)) * 1/kwargs["inv_tau"])

def logit_to_ssq(x: np.array, *args, **kwargs) -> np.array:
    diff = logit_to_diff(x, *args, **kwargs)
    return diff**2 * np.sign(diff)

def logit_to_sq(x: np.array, *args, **kwargs) -> np.array:
    diff = logit_to_diff(x, *args, **kwargs)
    return diff**2

def logitabs_to_abs(x: np.array, *args, **kwargs) -> np.array:

    x[x >= 1] = 1 - 1e-5
    x[x <= 0] = 0 + 1e-5

    return np.abs(np.log((1-x)/(x+1)) * 1/kwargs["inv_tau"])

def logitabs_to_sq(x: np.array, *args, **kwargs) -> np.array:
    absolute = logitabs_to_abs(x, *args, **kwargs)
    return absolute**2

def abs_to_logitabs(x: np.array, *args, **kwargs) -> np.array:
    inv_tau = np.log(3)/np.mean(np.abs(x))

    return 2/(1+np.exp(-inv_tau*x)) - 1

def abs_to_sq(x: np.array, *args, **kwargs) -> np.array:
    return x**2

def sq_to_abs(x: np.array, *args, **kwargs) -> np.array:
    return np.sqrt(np.abs(x))

def sq_to_logitabs(x: np.array, *args, **kwargs) -> np.array:
    absolute = sq_to_abs(x, *args, **kwargs)

    inv_tau = np.log(3)/np.mean(np.abs(absolute))

    return 2/(1+np.exp(-inv_tau*absolute)) - 1


MAPPING = {
    "diff_to_abs" : diff_to_abs,
    "diff_to_ssq" : diff_to_ssq,
    "diff_to_sq" : diff_to_sq,
    "diff_to_logit" : diff_to_logit,
    "diff_to_logitabs" : diff_to_logitabs,
    "ssq_to_diff" : ssq_to_diff,
    "ssq_to_sq" : ssq_to_sq,
    "ssq_to_abs" : ssq_to_abs,
    "ssq_to_logit" : ssq_to_logit,
    "ssq_to_logitabs" : ssq_to_logitabs,
    "logit_to_logitabs" : logit_to_logitabs,
    "logit_to_diff" : logit_to_diff,
    "logit_to_abs" : logit_to_abs,
    "logit_to_ssq" : logit_to_ssq,
    "logit_to_sq" : logit_to_sq,
    "abs_to_sq" : abs_to_sq,
    "abs_to_logitabs" : abs_to_logitabs,
    "sq_to_abs" : sq_to_abs,
    "sq_to_logitabs" : sq_to_logitabs,
    "logitabs_to_abs" : logitabs_to_abs,
    "logitabs_to_sq" : logitabs_to_sq
}