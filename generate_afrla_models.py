import json

max_depths = [3, 5, 7, 9, 11]
lrs = [0.1, 0.05, 0.01]
n_estimators = [100, 250, 500, 750, 1000]
models = ["CatBoostRegressor", "XGBRegressor", "LGBMRegressor"]

model_descriptions = []

for model in models:
    for max_depth in max_depths:
        for lr in lrs:
            for n_estimator in n_estimators:
                model_descriptions.append({
                    "model_type": model,
                    "architecture" : "boosting",
                    "hyperparameters" : {
                        "max_depth": max_depth,
                        "learning_rate": lr,
                        "estimators": n_estimator
                    }
                })

for max_depth in max_depths:
    for n_estimator in n_estimators:
            model_descriptions.append({
                "model_type": "RandomForestRegressor",
                "architecture" : "bagging",
                "hyperparameters" : {
                    "max_depth": max_depth,
                    "estimators": n_estimator
                }
            })

for max_depth in max_depths:
    model_descriptions.append({
        "model_type": "DecisionTreeRegressor",
        "architecture" : "tree",
        "hyperparameters" : {
            "max_depth": max_depth
        }
    })

labels = []
for model_dscr in model_descriptions:
    model_name = model_dscr["model_type"]
    max_depth = model_dscr["hyperparameters"]["max_depth"]
    lr = model_dscr["hyperparameters"].get("learning_rate", -1)
    n_estimator = model_dscr["hyperparameters"].get("estimators", 1)
    labels.append(f"{model_name}_{max_depth}_{n_estimator}_{lr}")

model_info = {
    "descriptions" : model_descriptions,
    "labels" : labels
}

with open("afrla_models.json", "w") as f:
    json.dump(model_info, f, indent=4)