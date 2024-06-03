import pandas as pd
import numpy as np
import json

import os

ROOT_DIR = os.getcwd()

# Load model embeddings
with open(os.path.join(ROOT_DIR, "afrla_embeddings.json")) as f:
    data = json.load(f)

model_names = data["names"]
model_embeddings = np.array(data["embeddings"])

me_df = pd.DataFrame(data = model_embeddings, columns = [f"ME{i}" for i in range(len(model_embeddings[0]))])
names_list = []
max_depths_list = []
n_estimators_list = []
lrs_list = []

for mn in model_names:
    name, max_depth, n_estimators, learning_rate = mn.split("_")
    max_depth, n_estimators, learning_rate = int(max_depth), int(n_estimators), float(learning_rate)

    names_list.append(name)
    max_depths_list.append(max_depth)
    n_estimators_list.append(n_estimators)
    lrs_list.append(learning_rate)

me_df["model"] = names_list
me_df["max_depth"] = max_depths_list
me_df["n_estimators"] = n_estimators_list
me_df["learning_rate"] = lrs_list

datasets = ["il_swCSC.csv", "il_parkinsons.csv", "il_infrared.csv"]

# Load ILOriginalBaseResults
for il_results_path in datasets:
    il_results = pd.read_csv(os.path.join(ROOT_DIR, "Data", "ILOriginalBaseResults", il_results_path))

    il_results_me = il_results.merge(me_df, on = ["model", "max_depth", "n_estimators", "learning_rate"], how = "inner")

    # No models lost
    assert il_results_me.shape[0] == il_results.shape[0]

    il_results_me.to_csv(os.path.join(ROOT_DIR, "Data", "ILBaseResults", il_results_path), index = False)

    print(f"Done with {il_results_path}!")