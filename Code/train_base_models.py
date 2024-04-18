import os, sys, argparse
import json

from ILPipeline.datasets import Dataset
from ILPipeline.tasks import OneTargetTask
from ILPipeline.flows import RegressionFlow
from ILPipeline.runs import Run

from assessor_utils import save_inst_lvl_results

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor

from itertools import product
import pandas as pd

MAPPING = {
    "XGBRegressor": XGBRegressor,
    "LGBMRegressor": LGBMRegressor,
    "RandomForestRegressor": RandomForestRegressor,
    "DecisionTreeRegressor": DecisionTreeRegressor,
    "CatBoostRegressor": CatBoostRegressor}

def prepare_taks(
    dataset_name: str, 
    dataset_path: os.PathLike, 
    dataset_config: dict) -> OneTargetTask:

    # Load the dataset
    dataset_df = pd.read_csv(dataset_path)

    # Prepare the dataset
    dataset_df = dataset_df.dropna()
    dataset_df = dataset_df.drop(dataset_config["remove_instances"])
    dataset_df = dataset_df.drop(columns = dataset_config["remove_columns"])

    Y = dataset_df.pop(dataset_config["target"]).to_frame()
    X = dataset_df.copy()

    # Create the Dataset object
    dataset = Dataset(name = dataset_name, features = X, targets = Y)

    # Create the Task object
    task = OneTargetTask(
        name = f"{dataset_name}_{dataset_config['target']}",
        dataset = dataset,
        target = dataset_config["target"],
        target_type = "regression",
        evaluation_method = f"tt{int(dataset_config['test_split']*100)}")
    
    return dataset, task

def run_base_models(
    task: OneTargetTask,
    dataset_config: dict,
    results_path: os.PathLike,
    random_state: int) -> None:

    model_needs = dataset_config["model_needs"]

    hyperparam_space = {
        hp: dataset_config["hyperparameter_space"][hp]["values"] 
        for hp in dataset_config["hyperparameter_space"]}
    
    all_hyperparameters = list(hyperparam_space.keys())
    temp_file_paths = []

    for model_name in model_needs:
        model = MAPPING[model_name]
        needs = [hyperparam_space[hp] for hp in model_needs[model_name]]
        combinations = list(product(*needs))

        for comb in combinations:
            args = {name: value for name, value in zip(model_needs[model_name], comb)}
            
            if model_name == "CatBoostRegressor":
                # Special case for CatBoost: allow_write_files must be False
                args["allow_writing_files"] = False
            
            m = model(random_state = random_state, **args)

            # Create the Flow object
            flow = RegressionFlow(
                name = f"{model_name}_{comb}", 
                random_state = random_state, 
                model = m)
            
            # Create the Run object and run it
            run = Run(task = task, flow = flow, name = f"{model_name}_{comb}_{task.name}")
            run_result = run.run()

            inst_lvl_results_df = run_result.instance_results
            inst_lvl_results_df["model"] = model_name
            inst_lvl_results_df["cpu_training_time"] = run_result.profile_results["cpu_training_time"][0]
            inst_lvl_results_df["cpu_prediction_time"] = run_result.profile_results["cpu_prediction_time"][0]
            inst_lvl_results_df["memory_usage"] = run_result.profile_results["memory_usage"][0]

            for hp in all_hyperparameters:
                hp_value = args.get(hp, None)
                if hp_value is None:
                    hp_value = dataset_config["hyperparameter_space"][hp]["placeholder_value"]

                inst_lvl_results_df[hp] = hp_value
            
            features = task.dataset.features.copy().reset_index(names = ["instance"])
            inst_lvl_results_df = pd.merge(features, inst_lvl_results_df, on = "instance")

            temp_file_path = save_inst_lvl_results(
                inst_lvl_df = inst_lvl_results_df,
                base_path = results_path,
                model_name = model_name,
                hyperparameters = args,
                task_name = task.name,
                use_temp_file = True)
            
            temp_file_paths.append(temp_file_path)

    return temp_file_paths

def combine_results(
    dataset_name: str,
    temp_file_paths: list,
    results_path: os.PathLike) -> None:

    dfs = []
    for file_path in temp_file_paths:
        df = pd.read_csv(file_path)
        dfs.append(df)
    
    final_df = pd.concat(dfs, axis = 0)
    final_df.to_csv(os.path.join(results_path, f"il_{dataset_name}.csv"), index = False)

    for file_path in temp_file_paths:
        os.unlink(file_path)

def main():
    parser = argparse.ArgumentParser(description = "Pipeline for training assessor models")

    parser.add_argument(
        "--dataset_name",
        type = str,
        help = "Name of the dataset to use. It must match the name of it corresponding .csv file in the 'Data/Datasets' folder",
        required = True)
    
    parser.add_argument(
        "--results_path",
        type = str,
        help = "Path to save the results of the base models",
        required = True)

    parser.add_argument(
        "--random_seed",
        type = int,
        help = "Random seed to use for reproducibility",
        default = 42)

    args = parser.parse_args()

    # Step 0: Initial checks

    ROOT_DIR = os.getcwd()

    if not os.path.exists(os.path.join(ROOT_DIR, "Data", "Datasets", args.dataset_name + ".csv")):
        raise FileNotFoundError("Dataset file not found in the 'Data/Datasets' folder. Please check spelling")

    # Step 1: Train the base models
    if os.path.exists(os.path.join(ROOT_DIR, "Data", "ILBaseResults", f"il_{args.dataset_name}.csv")):
        confirmation = input("IL data already exists. Do you want to overwrite it? (y/n): ")
        if confirmation.lower() != "y":
            print("Program will finish")
            raise Exception("IL data already exists")
        else:
            print("IL data will be overwritten")
            # os.remove(os.path.join(ROOT_DIR, "Data", "ILBaseResults", f"il_{args.dataset_name}.csv"))

    with open(os.path.join(ROOT_DIR, "datasets_config.json"), "r") as f:
        DATASET_CONFIG = json.load(f)[args.dataset_name]

    dataset, task = prepare_taks(
        dataset_name = args.dataset_name,
        dataset_path = os.path.join(ROOT_DIR, "Data", "Datasets", f"{args.dataset_name}.csv"),
        dataset_config = DATASET_CONFIG)
    
    temp_file_paths = run_base_models(
        task = task, 
        dataset_config = DATASET_CONFIG,
        results_path = args.results_path,
        random_state = args.random_seed)
    
    combine_results(
        dataset_name = args.dataset_name,
        temp_file_paths = temp_file_paths,
        results_path = args.results_path)

if __name__ == "__main__":

    # Check if we are in the correct working directory
    cwd = os.getcwd()
    if not os.path.isdir(os.path.join(cwd, "Code")):
        raise RuntimeError("Please run this script from the root directory of the project")
    
    sys.path.append(os.path.join(cwd, "Code"))

    try:
        main()
    except Exception as e:
        raise e
        sys.exit(1)