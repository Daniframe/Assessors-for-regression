from typing import Literal
import os, sys, argparse
import numpy as np
import pandas as pd
import json
import copy

import matplotlib.pyplot as plt

import assessor_utils as asutils
import error_transformations as et

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

class AssessorNetwork(nn.Module):
    def __init__(self, input_size, layer_sizes, dropout_rate):
        super(AssessorNetwork, self).__init__()
        
        layers = []
        current_size = input_size
        
        for size in layer_sizes:
            layers.append(nn.Linear(current_size, size))
            layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Sigmoid())
            layers.append(nn.Dropout(dropout_rate))
            current_size = size
        
        layers.append(nn.Linear(current_size, 1))
        self.network = nn.Sequential(*layers).float()
    
    def forward(self, x):
        return self.network(x)

def train_network(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device).float(), labels.to(device).float()
        
        optimizer.zero_grad()
        
        outputs = model(inputs)

        labels = labels.view(-1, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    return running_loss / len(train_loader)

def validate_network(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device).float(), labels.to(device).float()

            outputs = model(inputs)

            labels = labels.view(-1, 1)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
    return val_loss / len(val_loader)

def predict_network(model, data_loader, device):
    model.eval()  # Poner el modelo en modo de evaluación
    predictions = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs.float())
            predictions.extend(outputs.cpu().numpy())

    return predictions

def scale_data(train_X, val_X, test_X, categorical_columns):

    scaler = StandardScaler()

    non_num_cols = list(train_X.select_dtypes(include = ["bool", "object"]).columns)
    non_num_cols.extend(categorical_columns)

    num_cols = train_X.columns.difference(non_num_cols)

    non_num_train_X = pd.DataFrame({col: train_X.pop(col) for col in non_num_cols}).reset_index(drop = True)
    non_num_val_X = pd.DataFrame({col: val_X.pop(col) for col in non_num_cols}).reset_index(drop = True)
    non_num_test_X = pd.DataFrame({col: test_X.pop(col) for col in non_num_cols}).reset_index(drop = True)

    train_X = scaler.fit_transform(train_X.values)
    val_X = scaler.transform(val_X.values)
    test_X = scaler.transform(test_X.values)

    # Reunite with categorical columns
    train_X = pd.DataFrame(train_X, columns = num_cols).reset_index(drop = True)
    train_X = pd.concat([train_X, non_num_train_X], axis = 1)
    bool_cols = train_X.select_dtypes(include = "bool").columns
    train_X[bool_cols] = train_X[bool_cols].astype(int)

    val_X = pd.DataFrame(val_X, columns = num_cols).reset_index(drop = True)
    val_X = pd.concat([val_X, non_num_val_X], axis = 1)
    val_X[bool_cols] = val_X[bool_cols].astype(int)

    test_X = pd.DataFrame(test_X, columns = num_cols).reset_index(drop = True)
    test_X = pd.concat([test_X, non_num_test_X], axis = 1)
    test_X[bool_cols] = test_X[bool_cols].astype(int)

    return train_X, val_X, test_X

def prepare_data(
    il_dataset: pd.DataFrame,
    categorical_columns: list,
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
        training_split = 0.6,
        validation_split = 0.2,
        test_split = 0.2,
        return_original = True,
        seed = random_seed)
    
    et_info = data["additional_info"]

    train_X = data["train_X"]
    train_Y = data["train_Y"].values

    val_X = data["validation_X"]
    val_Y = data["validation_Y"].values

    test_X = data["test_X"]
    test_Y = data["test_Y"].values

    train_X_original = data["train_X_original"]
    val_X_original = data["validation_X_original"]
    test_X_original = data["test_X_original"]

    # Remove instance
    train_X = train_X.drop(columns = ["instance"])
    val_X = val_X.drop(columns = ["instance"])
    test_X = test_X.drop(columns = ["instance"])

    # Center and scale
    train_X, val_X, test_X = scale_data(train_X, val_X, test_X, categorical_columns)

    return train_X, train_Y, val_X, val_Y, test_X, test_Y, train_X_original, val_X_original, test_X_original, et_info

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

    with open(os.path.join(ROOT_DIR, "datasets_config.json"), "r") as f:
        dataset_info = json.load(f).get(args.dataset_name)
        categorical_columns = dataset_info.get("categorical_columns")
        if categorical_columns is None:
            categorical_columns = []

    if error_name1 is None:
        raise KeyError("Error type 1 not found in the 'errors_config.json' file. Please check spelling")
    
    if error_name2 is None:
        raise KeyError("Error type 2 not found in the 'errors_config.json' file. Please check spelling")

    # Step 1: Load instance-level results
    il_dataset = pd.read_csv(os.path.join(ROOT_DIR, "Data", "ILBaseResults", "il_" + args.dataset_name + ".csv"))

    # Step 2: Assessor with higher information
    # -----------------------------------------
    # Step 2.1: Prepare the data for training
    train_X, train_Y, val_X, val_Y, test_X, test_Y, train_X_original, val_X_original, test_X_original, et_info = prepare_data(
        il_dataset = il_dataset,
        categorical_columns = categorical_columns,
        error_name = error_name1,
        random_seed = args.random_seed)

    input_size = train_X.shape[1]

    train_X, train_Y, val_X, val_Y, test_X, test_Y = map(
        torch.tensor, 
        (
            train_X.values, train_Y, 
            val_X.values, val_Y, 
            test_X.values, test_Y
        )
    )

    print(train_X)
    print(val_X)
    print()
    print(train_Y)
    print(val_Y)

    train_dataset = TensorDataset(train_X, train_Y)
    val_dataset = TensorDataset(val_X, val_Y)
    test_dataset = TensorDataset(test_X, test_Y)

    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False)
    test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    assessor = AssessorNetwork(
        input_size = input_size, 
        layer_sizes = [1024, 512, 256, 128, 32], 
        dropout_rate = 0.5)

    assessor.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(assessor.parameters(), lr = 0.001)

    best_loss = float('inf')
    start_from_epoch = 1
    patience = 10
    trigger_times = 0

    for epoch in range(100):
        train_loss = train_network(assessor, train_loader, criterion, optimizer, device)
        val_loss = validate_network(assessor, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        if epoch >= start_from_epoch:
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(assessor.state_dict())
                trigger_times = 0
            else:
                trigger_times += 1
                if trigger_times >= patience:
                    print('Early stopping!')
                    assessor.load_state_dict(best_model_wts)
                    break

    torch.save(assessor.state_dict(), "./nn_assessor")

    # train_preds = np.array(predict_network(assessor, train_loader, device)).flatten()
    val_preds = np.array(predict_network(assessor, val_loader, device)).flatten()
    test_preds = np.array(predict_network(assessor, test_loader, device)).flatten()

    # train_results = pd.DataFrame({'Actual': train_Y., 'Predicted': train_predictions})
    val_results = pd.DataFrame({'Actual': val_Y.flatten(), 'Predicted': val_preds})
    test_results = pd.DataFrame({'Actual': test_Y.flatten(), 'Predicted': test_preds})

    val_results.to_csv(r"./val.csv")
    test_results.to_csv(r"./test.csv")

if __name__ == "__main__":
    main()