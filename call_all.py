import subprocess

dataset_names = [
    "abalone",
    "auction_verification",
    "bng_echoMonths",
    "california_housing",
    "infrared",
    "life_expectancy",
    "music_popularity",
    "parkinsons",
    "parkinsons_motor",
    "swCSC"]

error_combs = [
    # ("difference", "absolute"),        
    # ("difference", "squared"),          
    # ("difference", "squared_signed"),   
    # ("difference", "logistic"),
    # ("difference", "logistic_absolute"),
    # ("squared_signed", "absolute"),      
    # ("squared_signed", "squared"),       
    # ("squared_signed", "difference"),    
    # ("squared_signed", "logistic"),
    # ("squared_signed", "logistic_absolute"),
    # ("logistic", "logistic_absolute"),
    # ("logistic", "difference"),
    # ("logistic", "absolute"),
    # ("logistic", "squared_signed"),
    # ("logistic", "squared"),
    # ("absolute", "squared"),             
    # ("absolute", "logistic_absolute"),
    # ("squared", "absolute") 
    # ("squared", "logistic_absolute")
    # ("logistic_absolute", "absolute"),
    # ("logistic_absolute", "squared"),
]

for error_type1, error_type2 in error_combs:
    for dataset_name in dataset_names:
        try:
            subprocess.run([
                "python", "Code/train_assessor_model.py", 
                "--dataset_name", dataset_name,
                "--error_type1", error_type1,
                "--error_type2", error_type2], check = True)
        except subprocess.CalledProcessError:
            print(f"Error in {dataset_name} for {error_type1} and {error_type2}")
            raise SystemExit(1)