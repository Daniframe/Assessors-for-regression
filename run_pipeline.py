import os, sys, argparse, subprocess

def main():

    # Pipeline goes as follows
    # 1. If not already done, train the base models to obtain IL data
    # 2. Train the assessor model
    # 3. Obtain difficulty, capacity and margin levels
    # 4. Evaluate the assessor model

    parser = argparse.ArgumentParser(description = "Pipeline for training base models and assessor models")

    parser.add_argument(
        "--dataset_name",
        type = str,
        help = "Name of the dataset to use. It must match the name of it corresponding .csv file in the 'Data/Datasets' folder",
        required = True)

    parser.add_argument(
        "--train_base_models",
        type = bool,
        help = "Whether to train the base models or get the IL data from the 'Data' folder",
        default = False)

    parser.add_argument(
        "--train_assessor_model",
        type = bool,
        help = "Whether to train the assessor model",
        default = False)
    
    parser.add_argument(
        "--error_type",
        type = str,
        help = "Type of error to use. You can add error types editing the error_config.json file",
        default = "difference")

    parser.add_argument(
        "--evaluate_assessor_model",
        type = bool,
        help = "Whether to evaluate the assessor model",
        default = False)
    
    parser.add_argument(
        "--margin_levels",
        type = int,
        help = "Number of margin levels to use for evaluation of the assessor model",
        default = 5)
    
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

    if args.train_base_models:
        # Step 1: Train the base models
        subprocess.run(
            [
                "python", "Code/train_base_models.py", 
                "--dataset_name", args.dataset_name, 
                "--random_seed", str(args.random_seed),
                "--results_path", os.path.join(ROOT_DIR, "Data", "ILBaseResults")
            ])

    if args.train_assessor_model:
        # Step 2: Train the assessor model
        subprocess.run(
            [
                "python", "Code/train_assessor_model.py",
                "--dataset_name", args.dataset_name,
                "--error_type", args.error_type,
                "--results_path", os.path.join(ROOT_DIR, "Data", "ILAssessorResults"),
                "--random_seed", str(args.random_seed)
            ])
    
    # Deprecated
    # if args.evaluate_assessor_model:
    #     # Step 3: Evaluate the assessor model
    #     subprocess.run(
    #         [
    #             "python", "Code/evaluate_assessor_model.py",
    #             "--dataset_name", args.dataset_name,
    #             "--error_type", args.error_type,
    #             "--margin_levels", str(args.margin_levels)
    #         ]
    #     )

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