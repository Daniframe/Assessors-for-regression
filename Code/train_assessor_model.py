from typing import Literal
import os, sys, argparse
import numpy as np
import pandas as pd
import json

import matplotlib.pyplot as plt

import assessor_utils as asutils
import error_transformations as et
import xgboost as xgb

from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

def prepare_data(
    il_dataset: pd.DataFrame,
    error_name: str,
    random_seed: int):

    data = asutils.prepare_assessor_training_data(
        inst_lvl_df = il_dataset,
        use_cheater = False,
        select_models = "all",
        model_col_name = "model",
        predicted_col_name = "prediction",
        actual_col_name = "real",
        error_type = error_name,
        training_split = 0.7,
        test_split = 0.3,
        return_original = True,
        seed = random_seed)
    
    et_info = data["additional_info"]

    train_X = data["train_X"]
    train_Y = data["train_Y"].values

    test_X = data["test_X"]
    test_Y = data["test_Y"].values

    train_X_original = data["train_X_original"]
    test_X_original = data["test_X_original"]

    # Remove instance
    train_X = train_X.drop(columns = ["instance"])
    test_X = test_X.drop(columns = ["instance"])

    return train_X, train_Y, test_X, test_Y, train_X_original, test_X_original, et_info

def save_feature_importance(
    model: xgb.XGBRegressor,
    path: os.PathLike,
    dataset_name: str,
    error_type: str) -> None:

    for ft_imp in ["weight", "gain", "cover"]:
        fig, ax = plt.subplots(figsize = (9, 6))
        xgb.plot_importance(
            model,
            max_num_features = 15,
            importance_type = ft_imp, 
            ax = ax, 
            title = f"Feature importance ({ft_imp}) for {dataset_name}\nassessor({error_type} error)",
            values_format = "{v:.2f}")

        plt.tight_layout()
        asutils.savefig(path + f"/{ft_imp}_importance.png")

def main():
    parser = argparse.ArgumentParser(description = "Pipeline for training assessor models based on instance-level results")

    parser.add_argument(
        "--dataset_name",
        type = str,
        help = "Name of the dataset to use. It must match the name of it corresponding .csv file in the 'Data/Datasets' folder",
        required = True)
    
    parser.add_argument(
        "--error_type1",
        type = str,
        help = "First type of error to use. You can add error types editing the error_config.json file",
        required = True)

    parser.add_argument(
        "--error_type2",
        type = str,
        help = "Second type of error to use. You can add error types editing the error_config.json file",
        required = True)
    
    parser.add_argument(
        "--random_seed",
        type = int,
        help = "Random seed to use for reproducibility",
        default = 42)
    
    args = parser.parse_args()

    ROOT_DIR = os.getcwd()

    if not os.path.exists(os.path.join(ROOT_DIR, "Data", "ILBaseResults", "il_" + args.dataset_name + ".csv")):
        raise FileNotFoundError("Instance level results file not found in the 'Data/ILBaseResults' folder. Please check spelling")
    
    # Step 0: Initial checks and load error configuration
    if os.path.exists(os.path.join(ROOT_DIR, "Data", "ILAssessorResults", f"{args.error_type1}-{args.error_type2}", f"{args.error_type1}_test.csv")):
        confirmation = input("IL assessor data already exists. Do you want to overwrite it? (y/n): ")
        if confirmation.lower() != "y":
            print("Program will finish")
            raise Exception("IL assessor data already exists")
        else:
            print("IL assessor data will be overwritten")

    with open(os.path.join(ROOT_DIR, "errors_config.json"), "r") as f:
        error_info = json.load(f)
        error_name1 = error_info.get(args.error_type1)
        error_name2 = error_info.get(args.error_type2)

    if error_name1 is None:
        raise KeyError("Error type 1 not found in the 'errors_config.json' file. Please check spelling")
    
    if error_name2 is None:
        raise KeyError("Error type 2 not found in the 'errors_config.json' file. Please check spelling")
    
    metrics = {
        "set" : [],
        "approach" : [],
        "rmse" : [],
        "r2" : [],
        "spearman" : []
    }

    # Step 1: Load instance-level results
    il_dataset = pd.read_csv(os.path.join(ROOT_DIR, "Data", "ILBaseResults", "il_" + args.dataset_name + ".csv"))

    # Step 2: Assessor with higher information
    # -----------------------------------------
    # Step 2.1: Prepare the data for training
    train_X, train_Y, test_X, test_Y, train_X_original, test_X_original, et_info = prepare_data(
        il_dataset = il_dataset,
        error_name = error_name1,
        random_seed = args.random_seed)

    # Step 2.2: Train the assessor model
    xgb_model = xgb.XGBRegressor(
        objective = "reg:squarederror",
        n_estimators = 1000,
        learning_rate = 0.05,
        max_depth = 9,
        subsample = 0.8,
        colsample_bytree = 0.9,
        random_state = args.random_seed)

    xgb_model.fit(train_X, train_Y)

    # Step 2.3: Save model and feature importance
    asutils.save_xgbmodel(
        os.path.join(
            "Assessors", f"{args.dataset_name}", f"{args.error_type1}-{args.error_type2}", 
            f"{args.error_type1}.json"),
        xgb_model)
    
    save_feature_importance(
        model = xgb_model,
        path = os.path.join(ROOT_DIR, "Plots", f"{args.dataset_name}", f"{args.error_type1}-{args.error_type2}", f"{args.error_type1}", "FeatureImportance"),
        dataset_name = args.dataset_name,
        error_type = args.error_type1)
    
    # Step 2.4: Save instance level predictions
    train_pred = xgb_model.predict(train_X)
    test_pred = xgb_model.predict(test_X)

    # Step 2.4.1: Apply proper error transformation
    with open(os.path.join(ROOT_DIR, "errors_transformations.json"), "r") as f:
        translations = json.load(f)
        transformation_name = translations.get(f"{args.error_type1}-{args.error_type2}")

    if transformation_name is None:
        raise KeyError("Transformation not found in the 'errors_transformation.json' file. Please check spelling")

    transformation_func = et.MAPPING.get(transformation_name)

    # pd.DataFrame({"train_Y": train_Y, "train_pred": train_pred}).to_csv("train.csv", index = False)
    # pd.DataFrame({"test_Y": test_Y, "test_pred": test_pred}).to_csv("test.csv", index = False)

    train_Y = transformation_func(train_Y, **et_info)
    test_Y = transformation_func(test_Y, **et_info)

    train_pred = transformation_func(train_pred, **et_info)
    test_pred = transformation_func(test_pred, **et_info)

    # Step 2.4.2: Save predictions

    train_results = train_X_original.loc[:, ["instance", "model", "max_depth", "n_estimators", "learning_rate"]]
    train_results["real_outcome"] = train_Y
    train_results["predicted_outcome"] = train_pred

    test_results = test_X_original.loc[:, ["instance", "model", "max_depth", "n_estimators", "learning_rate"]]
    test_results["real_outcome"] = test_Y
    test_results["predicted_outcome"] = test_pred

    asutils.savecsv(
        os.path.join(
            ROOT_DIR, "Data", "ILAssessorResults", args.dataset_name,
            f"{args.error_type1}-{args.error_type2}", f"{args.error_type1}_train.csv"),
        train_results, index = False)

    asutils.savecsv(
        os.path.join(
            ROOT_DIR, "Data", "ILAssessorResults", args.dataset_name,
            f"{args.error_type1}-{args.error_type2}", f"{args.error_type1}_test.csv"),
        test_results, index = False)
    
    # Step 2.5: Save error distribution and predicted vs actual plots

    train_results["outcome_error"] = train_results["predicted_outcome"] - train_results["real_outcome"]
    test_results["outcome_error"] = test_results["predicted_outcome"] - test_results["real_outcome"]

    fig = asutils.plot_error_distribution_by_model(
        inst_lvl_df = train_results,
        model_col_name = "model",
        error_col_name = "outcome_error",
        include_total = True,
        color = "lightblue", bins = 100)
    
    asutils.savefig(
        os.path.join(
            ROOT_DIR, "Plots", f"{args.dataset_name}", f"{args.error_type1}-{args.error_type2}", 
            f"{args.error_type1}", "PvA", "error_distribution_train.png"))
    plt.close()

    fig = asutils.plot_error_distribution_by_model(
        inst_lvl_df = test_results,
        model_col_name = "model",
        error_col_name = "outcome_error",
        include_total = True,
        color = "lightblue", bins = 100)
    
    asutils.savefig(
        os.path.join(
            ROOT_DIR, "Plots", f"{args.dataset_name}", f"{args.error_type1}-{args.error_type2}", 
            f"{args.error_type1}", "PvA", "error_distribution_test.png"))
    plt.close()
    

    fig = asutils.plot_predicted_vs_actual_by_model(
        inst_lvl_df = train_results,
        model_col_name = "model",
        actual_col_name = "real_outcome",
        predicted_col_name = "predicted_outcome", s = 1)
    
    asutils.savefig(
        os.path.join(
            ROOT_DIR, "Plots", f"{args.dataset_name}", f"{args.error_type1}-{args.error_type2}", 
            f"{args.error_type1}", "PvA", "PvA_train.png"))
    plt.close()

    fig = asutils.plot_predicted_vs_actual_by_model(
        inst_lvl_df = test_results,
        model_col_name = "model",
        actual_col_name = "real_outcome",
        predicted_col_name = "predicted_outcome", s = 1)
    
    asutils.savefig(
        os.path.join(
            ROOT_DIR, "Plots", f"{args.dataset_name}", f"{args.error_type1}-{args.error_type2}", 
            f"{args.error_type1}", "PvA", "PvA_test.png"))
    plt.close()

    # Step 2.6: Compute metrics
    train_mse = mean_squared_error(train_Y, train_pred, squared = False)
    test_mse = mean_squared_error(test_Y, test_pred, squared = False)
    
    train_r2 = r2_score(train_Y, train_pred)
    test_r2 = r2_score(test_Y, test_pred)

    train_spearman = spearmanr(train_Y, train_pred)[0]
    test_spearman = spearmanr(test_Y, test_pred)[0]

    metrics["set"].append("train")
    metrics["approach"].append(f"{args.error_type1}")
    metrics["rmse"].append(train_mse)
    metrics["r2"].append(train_r2)
    metrics["spearman"].append(train_spearman)

    metrics["set"].append("test")
    metrics["approach"].append(f"{args.error_type1}")
    metrics["rmse"].append(test_mse)
    metrics["r2"].append(test_r2)
    metrics["spearman"].append(test_spearman)

    
    # Step 3: Assessor with lesser information
    # -----------------------------------------
    # Step 3.1: Prepare the data for training
    train_X, train_Y, test_X, test_Y, train_X_original, test_X_original, et_info = prepare_data(
        il_dataset = il_dataset,
        error_name = error_name2,
        random_seed = args.random_seed)

    # Step 3.2: Train the assessor model
    xgb_model = xgb.XGBRegressor(
        objective = "reg:squarederror",
        n_estimators = 1000,
        learning_rate = 0.05,
        max_depth = 9,
        subsample = 0.8,
        colsample_bytree = 0.9,
        random_state = args.random_seed)

    xgb_model.fit(train_X, train_Y)

    # Step 3.3: Save model and feature importance
    asutils.save_xgbmodel(
        os.path.join(
            "Assessors", f"{args.dataset_name}", f"{args.error_type1}-{args.error_type2}", 
            f"{args.error_type2}.json"),
        xgb_model)
    
    save_feature_importance(
        model = xgb_model,
        path = os.path.join(ROOT_DIR, "Plots", f"{args.dataset_name}", f"{args.error_type1}-{args.error_type2}", f"{args.error_type2}", "FeatureImportance"),
        dataset_name = args.dataset_name,
        error_type = args.error_type2)
    
    # Step 3.4: Save instance level predictions
    train_pred = xgb_model.predict(train_X)
    test_pred = xgb_model.predict(test_X)

    # Step 3.4.1: Not necessary to apply error transformation:
    #             This is the "target" error we want to
    #             transform the other error into.

    train_results = train_X_original.loc[:, ["instance", "model", "max_depth", "n_estimators", "learning_rate"]]
    train_results["real_outcome"] = train_Y
    train_results["predicted_outcome"] = train_pred

    test_results = test_X_original.loc[:, ["instance", "model", "max_depth", "n_estimators", "learning_rate"]]
    test_results["real_outcome"] = test_Y
    test_results["predicted_outcome"] = test_pred

    asutils.savecsv(
        os.path.join(
            ROOT_DIR, "Data", "ILAssessorResults", args.dataset_name,
            f"{args.error_type1}-{args.error_type2}", f"{args.error_type2}_train.csv"),
        train_results, index = False)

    asutils.savecsv(
        os.path.join(
            ROOT_DIR, "Data", "ILAssessorResults", args.dataset_name,
            f"{args.error_type1}-{args.error_type2}", f"{args.error_type2}_test.csv"),
        test_results, index = False)
    
    # Step 3.5: Save error distribution and predicted vs actual plots
    train_results["outcome_error"] = train_results["predicted_outcome"] - train_results["real_outcome"]
    test_results["outcome_error"] = test_results["predicted_outcome"] - test_results["real_outcome"]

    fig = asutils.plot_error_distribution_by_model(
        inst_lvl_df = train_results,
        model_col_name = "model",
        error_col_name = "outcome_error",
        include_total = True,
        color = "lightblue", bins = 100)
    
    asutils.savefig(
        os.path.join(
            ROOT_DIR, "Plots", f"{args.dataset_name}", f"{args.error_type1}-{args.error_type2}", 
            f"{args.error_type2}", "PvA", "error_distribution_train.png"))
    plt.close()

    fig = asutils.plot_error_distribution_by_model(
        inst_lvl_df = test_results,
        model_col_name = "model",
        error_col_name = "outcome_error",
        include_total = True,
        color = "lightblue", bins = 100)
    
    asutils.savefig(
        os.path.join(
            ROOT_DIR, "Plots", f"{args.dataset_name}", f"{args.error_type1}-{args.error_type2}", 
            f"{args.error_type2}", "PvA", "error_distribution_test.png"))
    plt.close()

    fig = asutils.plot_predicted_vs_actual_by_model(
        inst_lvl_df = train_results,
        model_col_name = "model",
        actual_col_name = "real_outcome",
        predicted_col_name = "predicted_outcome", s = 1)
    
    asutils.savefig(
        os.path.join(
            ROOT_DIR, "Plots", f"{args.dataset_name}", f"{args.error_type1}-{args.error_type2}", 
            f"{args.error_type2}", "PvA", "PvA_train.png"))
    plt.close()

    fig = asutils.plot_predicted_vs_actual_by_model(
        inst_lvl_df = test_results,
        model_col_name = "model",
        actual_col_name = "real_outcome",
        predicted_col_name = "predicted_outcome", s = 1)
    
    asutils.savefig(
        os.path.join(
            ROOT_DIR, "Plots", f"{args.dataset_name}", f"{args.error_type1}-{args.error_type2}", 
            f"{args.error_type2}", "PvA", "PvA_test.png"))
    plt.close()

    # Step 3.6: Compute metrics
    train_mse = mean_squared_error(train_Y, train_pred, squared = False)
    test_mse = mean_squared_error(test_Y, test_pred, squared = False)
    
    train_r2 = r2_score(train_Y, train_pred)
    test_r2 = r2_score(test_Y, test_pred)

    train_spearman = spearmanr(train_Y, train_pred)[0]
    test_spearman = spearmanr(test_Y, test_pred)[0]

    metrics["set"].append("train")
    metrics["approach"].append(f"{args.error_type2}")
    metrics["rmse"].append(train_mse)
    metrics["r2"].append(train_r2)
    metrics["spearman"].append(train_spearman)

    metrics["set"].append("test")
    metrics["approach"].append(f"{args.error_type2}")
    metrics["rmse"].append(test_mse)
    metrics["r2"].append(test_r2)
    metrics["spearman"].append(test_spearman)

    # Step 4: Save metrics
    metrics_df = pd.DataFrame(metrics)
    asutils.savecsv(
        os.path.join(
            ROOT_DIR, "Data", "AssessorPerformance", args.dataset_name,
            f"{args.error_type1}-{args.error_type2}_metrics.csv"),
        metrics_df, index = False)

    print(f"Program with params ({args.dataset_name}, {args.error_type1}, {args.error_type2}) finished")

if __name__ == "__main__":
    main()