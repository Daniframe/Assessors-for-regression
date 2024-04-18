from __future__ import annotations
from typing import List, Tuple, Literal

import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import pandas as pd
import numpy as np

import os, tempfile, functools

import xgboost as xgb
from sklearn.metrics import r2_score

COLOR_WHEEL = [
    "green", "red", "gold", "lightblue", "purple",
    "orange", "pink", "brown", "grey", "olive",
    "cyan", "lime", "teal", "magenta", "indigo"]

##################################################
#           PLOT AUXILIARY FUNCTIONS             #
##################################################

def _find_pairs_with_product(n: int):
    pairs = set()
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            pair = (i, n // i)
            pairs.add(pair)
    return pairs

def _find_best_pair_ratio(
    n: int,
    target_ratio: float = 2/3):
    best_pair = (0, 0)
    closest_difference = float('inf')

    pairs_n = _find_pairs_with_product(n)
    pairs_n_1 = _find_pairs_with_product(n + 1)

    pairs = pairs_n.union(pairs_n_1)

    for numerator, denominator in pairs:
        current_ratio = numerator / denominator
        current_difference = abs(current_ratio - target_ratio)

        if current_difference < closest_difference:
            closest_difference = current_difference
            best_pair = (numerator, denominator)

    return best_pair

#################################################
#       SAFE SAVING FIGURES AND DATAFRAMES      #
#################################################

def safesave(func, *args, **kwargs):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        path = os.path.dirname(args[0])
        if not os.path.exists(path):
            os.makedirs(path)
        return func(*args, **kwargs)
    return wrapper

@safesave
def savefig(filename: str, *args, **kwargs):
    plt.savefig(filename, *args, **kwargs)

@safesave
def savecsv(filename: str, df: pd.DataFrame, *args, **kwargs):
    df.to_csv(filename, *args, **kwargs)

@safesave
def save_xgbmodel(filename: str, model, *args, **kwargs):
    model.save_model(filename, *args, **kwargs)

##################################################
#        INSTANCE LEVEL ERROR FUNCTIONS          #
##################################################

def signed_error(
    inst_lvl_df: pd.DataFrame,
    predicted_col_name: str = "prediction",
    actual_col_name: str = "real") -> pd.DataFrame:

    return inst_lvl_df[predicted_col_name] - inst_lvl_df[actual_col_name], {}

def absolute_error(
    inst_lvl_df: pd.DataFrame,
    predicted_col_name: str = "prediction",
    actual_col_name: str = "real") -> pd.DataFrame:

    return abs(inst_lvl_df[predicted_col_name] - inst_lvl_df[actual_col_name]), {}

def signed_squared_error(
    inst_lvl_df: pd.DataFrame,
    predicted_col_name: str = "prediction",
    actual_col_name: str = "real") -> pd.DataFrame:

    diff = inst_lvl_df[predicted_col_name] - inst_lvl_df[actual_col_name]
    return np.sign(diff) * diff**2, {}

def squared_error(
    inst_lvl_df: pd.DataFrame,
    predicted_col_name: str = "prediction",
    actual_col_name: str = "real") -> pd.DataFrame:

    return (inst_lvl_df[predicted_col_name] - inst_lvl_df[actual_col_name])**2, {}

def signed_percentage_error(
    inst_lvl_df: pd.DataFrame,
    predicted_col_name: str = "prediction",
    actual_col_name: str = "real") -> pd.DataFrame:

    return (inst_lvl_df[predicted_col_name] - inst_lvl_df[actual_col_name]) / inst_lvl_df[actual_col_name], {}

def percentage_error(
    inst_lvl_df: pd.DataFrame,
    predicted_col_name: str = "prediction",
    actual_col_name: str = "real") -> pd.DataFrame:

    return abs((inst_lvl_df[predicted_col_name] - inst_lvl_df[actual_col_name]) / inst_lvl_df[actual_col_name]), {}

def quotient_error(
    inst_lvl_df: pd.DataFrame,
    predicted_col_name: str = "prediction",
    actual_col_name: str = "real") -> pd.DataFrame:

    return inst_lvl_df[predicted_col_name] / inst_lvl_df[actual_col_name], {}

def signed_log_error(
    inst_lvl_df: pd.DataFrame,
    predicted_col_name: str = "prediction",
    actual_col_name: str = "real") -> pd.DataFrame:

    # Avoid division by zero or negative
    temp_num = inst_lvl_df[predicted_col_name].apply(lambda x: max(10e-10, x))
    temp_den = inst_lvl_df[actual_col_name].apply(lambda x: max(10e-10, x))
    return np.log(temp_num / temp_den), {}

def absolute_log_error(
    inst_lvl_df: pd.DataFrame,
    predicted_col_name: str = "prediction",
    actual_col_name: str = "real") -> pd.DataFrame:

    # Avoid division by zero
    temp_num = inst_lvl_df[predicted_col_name].apply(lambda x: max(10e-10, x))
    temp_den = inst_lvl_df[actual_col_name].apply(lambda x: max(10e-10, x))
    return abs(np.log(temp_num / temp_den)), {}

def param_sigmoid(
    inst_lvl_df: pd.DataFrame,
    predicted_col_name: str = "prediction",
    actual_col_name: str = "real") -> pd.DataFrame:

    # Formula -> L(x) = 2 / (1 + exp(-x*B)) - 1 where B = ln(3)/mean

    difference_error = inst_lvl_df[predicted_col_name] - inst_lvl_df[actual_col_name]
    mean_error = np.abs(difference_error).mean()

    inv_tau = np.log(3) / mean_error

    return 2 / (1 + np.exp(-difference_error * inv_tau)) - 1, {"inv_tau" : inv_tau}

def abs_param_sigmoid(
    inst_lvl_df: pd.DataFrame,
    predicted_col_name: str = "prediction",
    actual_col_name: str = "real") -> pd.DataFrame:

    # Formula -> L(x) = abs(2 / (1 + exp(-x*B)) - 1) where B = ln(3)/mean

    parm_sig, info = param_sigmoid(inst_lvl_df, predicted_col_name, actual_col_name)

    return np.abs(parm_sig), info

##################################################
#        DIFFICULTY AND CAPACITY FUNCTIONS       #
##################################################

def save_inst_lvl_results(
    inst_lvl_df: pd.DataFrame,
    base_path: str,
    model_name: str,
    task_name: str,
    hyperparameters: dict,
    use_temp_file: bool = False) -> None:

    """
    Save instance level results to a csv file.

    Parameters
    ----------
    inst_lvl_df : pd.DataFrame
        DataFrame with instance level results. It must contain at least
        one column indicating the model and another indicating the
        instance level error.

    model_name : str
        Name of the model.

    task_name : str
        Name of the task.

    hyperparameters : dict
        Dictionary with the hyperparameters used for the model.

    Returns
    -------
    None
    """

    if use_temp_file:
        with tempfile.NamedTemporaryFile(mode = "w", delete = False, suffix = ".csv") as temp_file:
            temp_file_path = temp_file.name

            # Write DataFrame to the temporary file
            inst_lvl_df.to_csv(temp_file_path, index = False)

            return temp_file_path

    else:
        path_name = base_path
        path_name += f"{task_name}_{model_name}_"
        path_name += "_".join(f"{key}-{value}" for key, value in hyperparameters.items())

        inst_lvl_df.to_csv(path_name + ".csv", index = False)

def get_difficulty_level(
    inst_lvl_df: pd.DataFrame,
    instance_col_name: str = "instance",
    error_col_name: str = "error",
    n_difficulty_levels: int = 10,
    ascending: bool = True,
    autolabels: bool = True) -> pd.DataFrame:

    """
    Get the difficulty level of each instance given a DataFrame with
    instance level results and a number of difficulty levels.

    Parameters
    ----------
    inst_lvl_df : pd.DataFrame
        DataFrame with instance level results. It must contain at least
        one column indicating the instance level error.

    instance_col_name : str, optional
        Name of the column indicating the instance id, by default "instance".

    error_col_name : str, optional
        Name of the column indicating the instance level error, by
        default "error".

    n_difficulty_levels : int, optional
        Number of difficulty levels to use, by default 10.

    ascending : bool, optional
        Whether to sort the difficulty levels in ascending order (meaning
        that difficulty 0 is the easiest), by default True.

    autolabels : bool, optional
        Whether to, in case of ecnountering duplicate edges, automatically
        adjust the number of levels to the unique edges, by default True.

    Returns
    -------
    pd.DataFrame
        DataFrame with the difficulty level of each instance.
    """

    inst_lvl_df_copy = inst_lvl_df.copy()

    difficulty_df = inst_lvl_df_copy.groupby([instance_col_name])[error_col_name].mean().reset_index()
    labels = np.arange(n_difficulty_levels)

    if not ascending:
        labels = labels[::-1]

    if autolabels:
        try:
            difficulty_df["difficulty_level"] = pd.qcut(
                difficulty_df[error_col_name],
                q = n_difficulty_levels,
                labels = labels)
            
        except ValueError:
            # Manually find the unique edges
            qs = set()
            for i in range(n_difficulty_levels):
                qs.add(difficulty_df["difficulty"].quantile((i+1) / n_difficulty_levels))
            
            labels = np.arange(len(qs) - 1)
            if not ascending:
                labels = labels[::-1]

            difficulty_df["difficulty_level"] = pd.qcut(
                difficulty_df[error_col_name],
                q = n_difficulty_levels,
                duplicates = "drop",
                labels = labels)

    else:
        difficulty_df["difficulty_level"] = pd.qcut(
            difficulty_df[error_col_name],
            q = n_difficulty_levels,
            labels = labels)
    
    return difficulty_df.sort_values(by = [instance_col_name])

def get_soft_difficulties(
    inst_lvl_df: pd.DataFrame,
    instance_col_name: str = "instance",
    error_col_name: str = "error",
    n_difficulty_levels: int = 10,
    criterion: Literal["mean", "median", "std"] = "std",
    ascending: bool = True,
    autolabels: bool = True) -> pd.DataFrame:

    """
    Get the 'soft' difficulty level of each instance given a DataFrame with
    instance level results and a number of difficulty levels.

    Parameters
    ----------
    inst_lvl_df : pd.DataFrame
        DataFrame with instance level results. It must contain at least
        one column indicating the instance level error.

    instance_col_name : str, optional
        Name of the column indicating the instance id, by default "instance".

    error_col_name : str, optional
        Name of the column indicating the instance level error, by
        default "error".

    n_difficulty_levels : int, optional
        Number of difficulty levels to use, by default 10.

    ascending : bool, optional
        Whether to sort the difficulty levels in ascending order (meaning
        that difficulty 0 is the easiest), by default True.

    autolabels : bool, optional
        Whether to, in case of ecnountering duplicate edges, automatically
        adjust the number of levels to the unique edges, by default True.

    Returns
    -------
    pd.DataFrame
        DataFrame with the difficulty level of each instance.
    """

    criterion_function_dict = {
        "mean": np.mean,
        "median": np.median,
        "std": np.std}

    inst_lvl_df_copy = inst_lvl_df.copy()
    if criterion not in criterion_function_dict:
        raise ValueError("Invalid criterion")
    
    threshold = criterion_function_dict[criterion](inst_lvl_df_copy[error_col_name])
    inst_lvl_df_copy["correct"] = inst_lvl_df_copy[error_col_name] <= threshold

    difficulty_df = inst_lvl_df_copy.groupby([instance_col_name])["correct"].mean().reset_index()
    difficulty_df["difficulty"] = 1 - difficulty_df["correct"]
    difficulty_df = difficulty_df.drop(columns = ["correct"])
    labels = np.arange(n_difficulty_levels)

    if not ascending:
        labels = labels[::-1]

    if autolabels:
        try:
            difficulty_df["difficulty_level"] = pd.qcut(
                difficulty_df["difficulty"], #The more times it was correct, the easier it is
                q = n_difficulty_levels,
                labels = labels)
            
        except ValueError:
            qs = set()
            for i in range(n_difficulty_levels):
                qs.add(difficulty_df["difficulty"].quantile((i+1) / n_difficulty_levels))

            labels = np.arange(len(qs) - 1)
            if not ascending:
                labels = labels[::-1]

            difficulty_df["difficulty_level"] = pd.qcut(
                difficulty_df["difficulty"], #The more times it was correct, the easier it is
                q = n_difficulty_levels,
                duplicates = "drop",
                labels = labels)
            
    else:
        difficulty_df["difficulty_level"] = pd.qcut(
            difficulty_df["difficulty"], #The more times it was correct, the easier it is
            q = n_difficulty_levels,
            labels = labels)
    
    return difficulty_df.sort_values(by = [instance_col_name])

def get_capacities(
    inst_lvl_df: pd.DataFrame,
    model_descriptor_col_names: List[str] = None,
    error_col_name: str = "error",
    n_capacity_levels: int = 3,
    ascending: bool = True) -> pd.DataFrame:

    """
    Get the capacity of each model given a DataFrame with instance level
    results and a number of capacity levels.

    Parameters
    ----------
    inst_lvl_df : pd.DataFrame
        DataFrame with instance level results. It must contain at least
        one column indicating the model and another indicating the
        instance level error.

    model_descriptor_col_names : list[str], optional
        List of column names indicating the model descriptors, by default None,
        which will be converted to ['model', 'max_depth', 'n_estimators', 'learning_rate'].

    error_col_name : str, optional
        Name of the column indicating the instance level error, by
        default "error".

    n_capacity_levels : int, optional
        Number of capacity levels to use, by default 3.

    ascending : bool, optional
        Whether to sort the capacity levels in ascending order (meaning
        that capacity 0 is the simplest), by default True.
    """

    if model_descriptor_col_names is None:
        model_descriptor_col_names = ["model", "max_depth", "n_estimators", "learning_rate"]

    inst_lvl_df_copy = inst_lvl_df.copy()
    capacity_df = inst_lvl_df_copy.groupby(model_descriptor_col_names)[error_col_name].mean().reset_index()
    labels = np.arange(n_capacity_levels)

    if not ascending:
        labels = labels[::-1]

    capacity_df["capacity_level"] = pd.qcut(
        -capacity_df[error_col_name], # Better to understand: higher error, lower capacity
        q = n_capacity_levels,
        labels = labels)
    
    return capacity_df

def get_soft_capacities(
    inst_lvl_df: pd.DataFrame,
    model_descriptor_col_names: List[str] = None,
    error_col_name: str = "error",
    n_capacity_levels: int = 3,
    criterion: Literal["mean", "median", "std"] = "std",
    ascending: bool = True) -> pd.DataFrame:

    """
    Get the 'soft' capacity of each model given a DataFrame with instance level
    results and a number of capacity levels.

    Parameters
    ----------
    inst_lvl_df : pd.DataFrame
        DataFrame with instance level results. It must contain at least
        one column indicating the model and another indicating the
        instance level error.

    model_descriptor_col_names : list[str], optional
        List of column names indicating the model descriptors, by default None,
        which will be converted to ['model', 'max_depth', 'n_estimators', 'learning_rate'].

    error_col_name : str, optional
        Name of the column indicating the instance level error, by
        default "error".

    criterion: : Literal["mean", "median", "std"], optional
        Criterion to use to determine the capacity, by default "std".

    n_capacity_levels : int, optional
        Number of capacity levels to use, by default 3.

    ascending : bool, optional
        Whether to sort the capacity levels in ascending order (meaning
        that capacity 0 is the simplest), by default True.
    """    

    criterion_function_dict = {
        "mean": np.mean,
        "median": np.median,
        "std": np.std}
    
    if model_descriptor_col_names is None:
        model_descriptor_col_names = ["model", "max_depth", "n_estimators", "learning_rate"]

    inst_lvl_df_copy = inst_lvl_df.copy()
    if criterion not in criterion_function_dict:
        raise ValueError("Invalid criterion")
    
    threshold = criterion_function_dict[criterion](inst_lvl_df_copy[error_col_name])
    inst_lvl_df_copy["correct"] = inst_lvl_df_copy[error_col_name] <= threshold

    capacity_df = inst_lvl_df_copy.groupby(
        model_descriptor_col_names
    )["correct"].mean().reset_index().rename(columns = {"correct": "capacity"})
    labels = np.arange(n_capacity_levels)

    if not ascending:
        labels = labels[::-1]

    capacity_df["capacity_level"] = pd.qcut(
        capacity_df["capacity"], #The more times it was correct, the most capable it is
        q = n_capacity_levels,
        labels = labels)
    
    return capacity_df.sort_values(by = model_descriptor_col_names)

def add_cheater(
    inst_lvl_df: pd.DataFrame,
    cheater_col_name: str,
    instance_col_name: str = "instance",
    model_col_name: str = "model",
    predicted_col_name: str = "prediction",
    actual_col_name: str = "real",
    hyperparameters_col_names: List[str] = None,
    profile_col_names: List[str] = None,
    cheater_name: str = "cheater",
    cheater_value: int = 10) -> pd.DataFrame:

    """
    Add a cheater model to a DataFrame with instance level results.

    Parameters
    ----------
    inst_lvl_df : pd.DataFrame
        DataFrame with instance level results. It must contain at least
        one column indicating the model and another indicating the
        instance level error.

    cheater_col_name : str
        Name of the column used by the cheater to cheat.

    model_col_name : str, optional
        Name of the column indicating the model, by default "model".

    predicted_col_name : str, optional
        Name of the column indicating the predicted value, by default
        "prediction".

    actual_col_name : str, optional
        Name of the column indicating the actual value, by default
        "real".

    hyperparameters_col_names : list[str], optional
        List of column names indicating the hyperparameters used for the
        model, by default None.

    profile_col_names : list[str], optional
        List of column names indicating the profile of the model, by
        default None.

    cheater_name : str, optional
        Name of the cheater model, by default "cheater".

    cheater_value : int, optional
        Value of the cheater model, by default 10.

    Returns
    -------
    pd.DataFrame
        DataFrame with the cheater model added.
    """

    if hyperparameters_col_names is None:
        hyperparameters_col_names = ["max_depth", "n_estimators", "learning_rate"]

    if profile_col_names is None:
        profile_col_names = ["cpu_training_time", "cpu_prediction_time", "memory_usage"]

    cheater_df = inst_lvl_df.drop_duplicates(subset = [instance_col_name], keep = "first")
    cheater_df.loc[:, model_col_name] = cheater_name

    cheat_col = cheater_df.loc[:, cheater_col_name].values
    cheat_col_sorted = sorted(cheat_col)

    cheat_col_median = cheat_col_sorted[len(cheat_col_sorted) // 2]

    def cheater_function(row):
        if row[cheater_col_name] < cheat_col_median:
            return cheater_value
        else:
            return row[actual_col_name]

    cheater_df.loc[:, predicted_col_name] = cheater_df.apply(lambda row: cheater_function(row), axis = 1)

    for col in hyperparameters_col_names:
        cheater_df.loc[:, col] = np.nan
    
    for col in profile_col_names:
        cheater_df.loc[:, col] = np.nan

    return pd.concat([inst_lvl_df, cheater_df], ignore_index = True)

def plot_error_distribution(
    errors: List[float],
    include_limits: bool = True,
    density: bool = True,
    title: str = "Error distribution",
    **kwargs) -> plt.Figure:

    """
    Distribution error plot given a list of errors.

    Parameters
    ----------
    errors : list[float]
        List of errors.

    include_limits : bool, optional
        Whether to include the 2.5-97.5 interval, by default True.

    density : bool, optional
        Whether to plot the density instead of the count, by default True.

    title : str, optional
        Title of the plot, by default "Error distribution".

    Returns
    -------
    fig : plt.Figure
        Figure with the error distribution plot.
    """

    fig, ax = plt.subplots(figsize = (10, 6))

    ax.hist(x = errors, density = density, **kwargs)
    ax.set_title(title, fontsize = 12)
    ax.set_xlabel("Error", fontsize = 10)
    ax.set_ylabel("Density" if density else "Frequency", fontsize = 10)

    if include_limits:
        ax.axvline(x = np.quantile(errors, 0.025), color = "tomato", linestyle = "dashed", linewidth = 0.5)
        ax.axvline(x = np.quantile(errors, 0.975), color = "tomato", linestyle = "dashed", linewidth = 0.5)

    fig.tight_layout()
    return fig

def prepare_assessor_training_data(
    inst_lvl_df: pd.DataFrame,
    use_cheater: bool = False,
    select_models: List[str] | Literal["cheater"] | Literal["all"] = "all",
    model_col_name: str = "model",
    predicted_col_name: str = "prediction",
    actual_col_name: str = "real",
    error_type: Literal[
        "signed", 
        "absolute", 
        "signed_squared",
        "squared",
        "signed_percentage",
        "percentage", 
        "quotient", 
        "signed_log",
        "log",
        "logistic",
        "abs_logistic"] = "signed",
    training_split: float = 0.7,
    test_split: float = 0.3,
    validation_split: float | None = None,
    use_dummies: bool = True,
    categorical_col_names: List[str] | None = None,
    return_original: bool = True,
    seed: int = 42,
    **kwargs):

    """
    Prepare the data for the assessor training.

    Parameters
    ----------
    inst_lvl_df : pd.DataFrame
        DataFrame with instance level results. It must contain at least
        one column indicating the model and another indicating the
        instance level error.

    use_cheater : bool, optional
        Whether to add a cheater model, by default False.

    select_models : list[str] | "all", optional
        List of models to select, by default "all".

    model_col_name : str, optional
        Name of the column indicating the model, by default "model".

    predicted_col_name : str, optional
        Name of the column indicating the predicted value, by default
        "prediction".

    actual_col_name : str, optional
        Name of the column indicating the actual value, by default
        "real".

    error_type : str, optional
        Type of error to use, by default "signed".

    training_split : float, optional
        Percentage of the data to use for training, by default 0.7.

    test_split : float, optional
        Percentage of the data to use for testing, by default 0.3.

    validation_split : float, optional
        Percentage of the data to use for validation, by default None. If None, no
        validation data will be used.

    seed : int, optional
        Seed for the random number generator, by default 42.

    use_dummies: bool, optional
        Whether to use dummy variables for categorical features, by default True.

    categorical_col_names : list[str], optional
        List of column names indicating the categorical features to create
        dummies, by default None.

    return_original: bool, optional
        Whether to return DataFrames without dummies, by default True.

    capacity_col_names : list[str], optional
        List of column names indicating the capacity of the model, by default None.

    difficulty_col_names : list[str], optional
        List of column names indicating the difficulty of the instance, by default None.

    **kwargs
        Additional arguments to pass to add_cheater if set to True. See the documentation
        for the `add_cheater` function.
    """

    _ERROR_MAPPING = {
        "signed": signed_error,
        "absolute": absolute_error,
        "signed_squared": signed_squared_error,
        "squared": squared_error,
        "signed_percentage": signed_percentage_error,
        "percentage": percentage_error,
        "quotient": quotient_error,
        "signed_log": signed_log_error,
        "log": absolute_log_error,
        "logistic" : param_sigmoid,
        "abs_logistic": abs_param_sigmoid
    }

    if validation_split is None:
        validation_split = 0

    assert training_split + test_split + validation_split == 1, "The sum of the splits must be 1"

    # Add cheater if needed
    if use_cheater:
        inst_lvl_df = add_cheater(
            inst_lvl_df = inst_lvl_df,
            model_col_name = model_col_name,
            predicted_col_name = predicted_col_name,
            actual_col_name = actual_col_name,
            **kwargs)

    # Select only instance level results for the selected models
    if select_models == "cheater":
        inst_lvl_models_df = inst_lvl_df.loc[inst_lvl_df[model_col_name] == "cheater", :].copy()
        inst_lvl_models_df = inst_lvl_models_df.drop(columns = [model_col_name])

    elif select_models != "all":
        inst_lvl_models_df = inst_lvl_df.loc[inst_lvl_df[model_col_name].isin(select_models), :].copy()
        if len(select_models) == 1:
            inst_lvl_models_df = inst_lvl_models_df.drop(columns = [model_col_name])

    else:
        inst_lvl_models_df = inst_lvl_df.copy()

    # Compute the error
    if error_type in _ERROR_MAPPING:
        new_error, additional_info = _ERROR_MAPPING[error_type](inst_lvl_models_df, predicted_col_name, actual_col_name)
        inst_lvl_models_df.loc[:, f"{error_type}_error"] = new_error
    else:
        raise ValueError("Invalid error type")
    
    inst_lvl_models_df = inst_lvl_models_df.drop(columns = [predicted_col_name, actual_col_name])

    # Split the data
    rng = np.random.default_rng(seed = seed)

    instances = inst_lvl_models_df["instance"].unique()
    n_instances = len(instances)

    rng.shuffle(instances)

    train_instances = instances[:int(training_split * n_instances)]

    if validation_split > 0:
        validation_instances = instances[int(training_split * n_instances):int((training_split + validation_split) * n_instances)]
        test_instances = instances[int((training_split + validation_split) * n_instances):]
    else:
        validation_instances = []
        test_instances = instances[int(training_split * n_instances):]

    train_X_original = inst_lvl_models_df.loc[inst_lvl_models_df["instance"].isin(train_instances)]

    if validation_split > 0:
        validation_X_original = inst_lvl_models_df.loc[inst_lvl_models_df["instance"].isin(validation_instances)]

    test_X_original = inst_lvl_models_df.loc[inst_lvl_models_df["instance"].isin(test_instances)]

    # Get the dummies
    if use_dummies:
        train_X = pd.get_dummies(train_X_original, columns = categorical_col_names)

        if validation_split > 0:
            validation_X = pd.get_dummies(validation_X_original, columns = categorical_col_names)

        test_X = pd.get_dummies(test_X_original, columns = categorical_col_names)


        train_Y = train_X.pop(f"{error_type}_error")

        if validation_split > 0:
            validation_Y = validation_X.pop(f"{error_type}_error")

        test_Y = test_X.pop(f"{error_type}_error")

        if return_original:
            if validation_split > 0:
                # Return in dictionary form
                return {
                    "train_X": train_X,
                    "train_Y": train_Y,
                    "validation_X": validation_X,
                    "validation_Y": validation_Y,
                    "test_X": test_X,
                    "test_Y": test_Y,
                    "train_X_original": train_X_original,
                    "validation_X_original": validation_X_original,
                    "test_X_original": test_X_original,
                    "additional_info": additional_info
                }
            else:
                return {
                    "train_X": train_X,
                    "train_Y": train_Y,
                    "test_X": test_X,
                    "test_Y": test_Y,
                    "train_X_original": train_X_original,
                    "test_X_original": test_X_original,
                    "additional_info": additional_info
                }

        else:
            if validation_split > 0:
                # Return in dictionary form
                return {
                    "train_X": train_X,
                    "train_Y": train_Y,
                    "validation_X": validation_X,
                    "validation_Y": validation_Y,
                    "test_X": test_X,
                    "test_Y": test_Y,
                    "additional_info": additional_info
                }
            else:
                return {
                    "train_X": train_X,
                    "train_Y": train_Y,
                    "test_X": test_X,
                    "test_Y": test_Y,
                    "additional_info": additional_info
                }
    else:
        if validation_split > 0:
            # Return in dictionary form
            return {
                "train_X": train_X_original,
                "train_Y": train_Y,
                "validation_X": validation_X_original,
                "validation_Y": validation_Y,
                "test_X": test_X_original,
                "test_Y": test_Y,
                "additional_info": additional_info
            }
        
        else:
            return {
                "train_X": train_X_original,
                "train_Y": train_Y,
                "test_X": test_X_original,
                "test_Y": test_Y,
                "additional_info": additional_info
            }

def plot_error_distribution_by_model(
    inst_lvl_df: pd.DataFrame,
    model_col_name: str = "model",
    error_col_name: str = "error",
    include_total: bool = True,
    include_limits: bool = True,
    nrows: int | None = None,
    ncols: int | None = None,
    density: bool = True,
    **kwargs):
    
    """
    Distribution error plot for each model given a DataFrame with
    instance level results.

    Parameters
    ----------
    inst_lvl_df : pd.DataFrame
        DataFrame with instance level results. It must contain at least
        one column indicating the model and another indicating the
        instance level error.

    model_col_name : str, optional
        Name of the column indicating the model, by default "model".

    error_col_name : str, optional
        Name of the column indicating the instance level error, by
        default "error".

    include_total : bool, optional
        Whether to include the total error distribution, by default True.
    
    include_limits : bool, optional
        Whether to include the 2.5-97.5 interval, by default True.

    nrows : int, optional
        Number of rows of the plot, by default, automatically determined.

    ncols : int, optional
        Number of columns of the plot, by default, automatically determined.

    density : bool, optional
        Whether to plot the density instead of the count, by default True.

    Returns
    -------
    fig : plt.Figure
        Figure with the error distribution plot.
    """

    # Get the number of models
    models = inst_lvl_df[model_col_name].unique()
    n_models = len(models) + 1 if include_total else len(models)

    # Get the best pair of values for the ratio
    if nrows is None:
        nrows, ncols = _find_best_pair_ratio(n_models)

    fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize = (10, 6), sharex = True, sharey = True)
    
    for i, model in enumerate(models):
        row = i // ncols
        col = i % ncols

        model_df = inst_lvl_df.loc[inst_lvl_df[model_col_name] == model]
        model_df[error_col_name].plot.hist(ax = ax[row, col], density = density, **kwargs)
        ax[row, col].set_title(model, fontsize = 10)
        ax[row, col].grid()

    if include_total:
        row = nrows - 1
        col = ncols - 1
        inst_lvl_df[error_col_name].plot.hist(ax = ax[row, col], density = density, **kwargs)
        ax[row, col].set_title("Total", fontsize = 10)
        ax[row, col].grid()
    
    if density:
        ylabel = "Density"
        ymax = max(fig.axes[i].get_ylim()[1] for i in range(nrows * ncols))

        steps = ymax // 0.1

        for i in range(nrows * ncols):
            row = i // ncols
            col = i % ncols
            ax[row, col].set_ylim(0, (steps + 1) * 0.1)
            # ax[row, col].set_yticks(np.linspace(0, ymax, steps))

    else:
        ylabel = "Frequency"
        ymax = max(fig.axes[i].get_ylim()[1] for i in range(nrows * ncols))
        ymax_magnitude = int(np.log10(ymax)) - 2
        steps = ymax // 10**ymax_magnitude

        for i in range(nrows * ncols):
            row = i // ncols
            col = i % ncols
            ax[row, col].set_ylim(0, (steps + 1) * 10**ymax_magnitude)
            # ax[row, col].set_yticks(np.linspace(0, ymax, steps))

    if include_limits:
        for ax in fig.axes:
            if ax.has_data():
                ax.axvline(
                    x = inst_lvl_df[error_col_name].quantile(0.025), 
                    color = "tomato", linestyle = "dashed", linewidth = 0.5)
                ax.axvline(
                    x = inst_lvl_df[error_col_name].quantile(0.975), 
                    color = "tomato", linestyle = "dashed", linewidth = 0.5)

    for ax in fig.axes:
        if not ax.has_data():
            ax.axis("off")
        else:
            ax.set_ylabel(ylabel, fontsize = 10)
            ax.set_xlabel("Error", fontsize = 10)
            ax.tick_params(axis = "both", labelsize = 9)

    title_addon = " (95% limits)" if include_limits else ""

    fig.suptitle("Error distribution per model" + title_addon, fontsize = 12)

    fig.tight_layout()
    return fig

def plot_predicted_vs_actual(
    predicted: List[float],
    actual: List[float],
    title: str = "Predicted vs actual",
    **kwargs) -> plt.Figure:

    """
    Plot of the predicted vs actual values given a list of predicted and
    actual values.

    Parameters
    ----------
    predicted : list[float]
        List of predicted values.

    actual : list[float]
        List of actual values.

    title : str, optional
        Title of the plot, by default "Predicted vs actual".

    Returns
    -------
    fig : plt.Figure
        Figure with the predicted vs actual plot.
    """

    fig, ax = plt.subplots(figsize = (10, 6))

    ax.scatter(x = predicted, y = actual, **kwargs)
    ax.set_title(title, fontsize = 12)
    ax.set_xlabel("Predicted", fontsize = 10)
    ax.set_ylabel("Actual", fontsize = 10)

    fig.tight_layout()
    return fig

def plot_predicted_vs_actual_by_model(
    inst_lvl_df: pd.DataFrame,
    model_col_name: str = "model",
    predicted_col_name: str = "prediction",
    actual_col_name: str = "real",
    nrows: int | None = None,
    ncols: int | None = None,
    include_total: bool = True,
    compute_spearman: bool = True,
    colors: List[str] | Literal["default"] | None = "default",
    suptitle: str = "Predicted vs actual per model",
    **kwargs):

    """
    Plot of the predicted vs actual values for each model given a
    DataFrame with instance level results.

    Parameters
    ----------
    inst_lvl_df : pd.DataFrame
        DataFrame with instance level results. It must contain at least
        one column indicating the model and another indicating the
        predicted and actual values.

    model_col_name : str, optional
        Name of the column indicating the model, by default "model".

    predicted_col_name : str, optional
        Name of the column indicating the predicted value, by default
        "prediction".

    actual_col_name : str, optional
        Name of the column indicating the actual value, by default
        "real".

    nrows : int, optional
        Number of rows of the plot, by default, automatically determined.

    ncols : int, optional
        Number of columns of the plot, by default, automatically determined.

    include_total : bool, optional
        Whether to include total results, without disagregating my model, by default True.

    compute_r2 : bool, optional
        Whether to compute the R2 metric for each model, by default True.

    colors : list[str], optional
        List of colors to use for each model, by default None (all models will have the same color).
        If `include_total` is True, the results for total will be black
    
    Returns
    -------
    fig : plt.Figure
        Figure with the predicted vs actual plot.
    """ 

    # Get the number of models
    models = inst_lvl_df[model_col_name].unique()
    n_models = len(models) + include_total #Fancy way of saying + 1 if include_total else + 0

    if colors == "default":
        colors = COLOR_WHEEL
    elif colors is None:
        colors = ["blue"] * n_models - 1
    else:
        assert len(colors) >= n_models - 1, "Not enough colors for the number of models"

    # Get the best pair of values for the ratio
    if nrows is None:
        nrows, ncols = _find_best_pair_ratio(n_models)

    fig, ax = plt.subplots(nrows = nrows, ncols = ncols, figsize = (10, 6), sharex = True, sharey = True)
    
    for i, model in enumerate(models):
        row = i // ncols
        col = i % ncols

        model_df = inst_lvl_df.loc[inst_lvl_df[model_col_name] == model]
        model_df.plot.scatter(
            x = predicted_col_name,
            y = actual_col_name,
            ax = ax[row, col],
            title = model,
            fontsize = 10,
            color = colors[i % len(colors)], #Cycling, baby!
            **kwargs)
        
        if compute_spearman:
            r2, _ = spearmanr(model_df[actual_col_name], model_df[predicted_col_name])
            ax[row, col].set_title(f"{model}\nSpearman: {r2:.2f}", fontsize = 10)

        min_error = min(inst_lvl_df[actual_col_name].min(), inst_lvl_df[predicted_col_name].min())
        max_error = max(inst_lvl_df[actual_col_name].max(), inst_lvl_df[predicted_col_name].max())
        
        # min_error, max_error = model_df[actual_col_name].min(), model_df[actual_col_name].max()
        ax[row, col].plot([min_error, max_error], [min_error, max_error], color = "black", linestyle = "dashed", linewidth = 0.5)

        ax[row, col].grid()

    if include_total:
        row = nrows - 1
        col = ncols - 1
        inst_lvl_df.plot.scatter(
            x = predicted_col_name,
            y = actual_col_name,
            ax = ax[row, col],
            title = "Total",
            fontsize = 10,
            color = "black",
            **kwargs)
        
        if compute_spearman:
            r2, _ = spearmanr(inst_lvl_df[actual_col_name], inst_lvl_df[predicted_col_name])
            ax[row, col].set_title(f"Total\nSpearman: {r2:.2f}", fontsize = 10)

        ax[row, col].grid()

    for ax in fig.axes:
        if not ax.has_data():
            ax.axis("off")

    fig.suptitle(suptitle, fontsize = 12)
    fig.tight_layout()
    return fig