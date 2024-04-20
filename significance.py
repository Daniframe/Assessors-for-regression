import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def intervals_overlap(interval1: tuple, interval2: tuple) -> bool:
    # Check if the upper bound of one interval is greater than or equal to the lower bound of the other interval
    overlap_condition_1 = interval1[1] >= interval2[0] and interval2[1] >= interval1[0]
    
    # Check if the upper bound of the other interval is greater than or equal to the lower bound of this interval
    overlap_condition_2 = interval2[1] >= interval1[0] and interval1[1] >= interval2[0]

    # Return True if both conditions are met, indicating overlap
    return overlap_condition_1 and overlap_condition_2

def bootstrap_spearman(
    df: pd.DataFrame, 
    real_column_name: str,
    predicted_column_name: str,
    n_iter: int = 100) -> pd.DataFrame:

    real = df[real_column_name].values
    predicted = df[predicted_column_name].values

    corrs = []
    for _ in range(n_iter):
        indices = np.random.choice(len(real), len(real), replace = True)
        corr, _ = spearmanr(real[indices], predicted[indices])
        corrs.append(corr)

    return np.percentile(corrs, [2.5, 97.5])

if __name__ == "__main__":

    DATASETS = [
        "abalone", "auction_verification", "bng_echoMonths", "california_housing",
        "infrared", "life_expectancy", "music_popularity", "parkinsons",
        "parkinsons_motor", "swCSC"]
    
    PAIRINGS = [
        ("difference", "absolute"), 
        ("difference", "squared"),
        ("difference", "squared_signed"),
        ("difference", "logistic_absolute"),
        ("difference", "logistic"),
        ("squared_signed", "difference"),
        ("squared_signed", "absolute"),
        ("squared_signed", "squared"),
        ("squared_signed", "logistic_absolute"),
        ("squared_signed", "logistic"),
        ("logistic", "difference"),
        ("logistic", "absolute"),
        ("logistic", "squared"),
        ("logistic", "squared_signed"),
        ("logistic", "logistic_absolute"),
        ("absolute", "squared"),
        ("absolute", "logistic_absolute"),
        ("squared", "absolute"),
        ("squared", "logistic_absolute"),
        ("logistic_absolute", "absolute"),
        ("logistic_absolute", "squared")
    ]

    results = {
        "dataset" : [],
        "proxy_error" : [],
        "target_error" : [],
        "pe_spearman" : [],
        "te_spearman" : [],
        "significant_difference" : []}
    
    for dataset in DATASETS:
        for et1, et2 in PAIRINGS:
            df1 = pd.read_csv(fr"./Data/ILAssessorResults/{dataset}/{et1}-{et2}/{et1}_test.csv")
            df2 = pd.read_csv(fr"./Data/ILAssessorResults/{dataset}/{et1}-{et2}/{et2}_test.csv")

            sp1 = spearmanr(df1["real_outcome"], df1["predicted_outcome"]).correlation
            sp2 = spearmanr(df2["real_outcome"], df2["predicted_outcome"]).correlation

            results["dataset"].append(dataset)
            results["proxy_error"].append(et1)
            results["target_error"].append(et2)
            results["pe_spearman"].append(sp1)
            results["te_spearman"].append(sp2)

            if np.abs(sp1 - sp2) > 0.1:
                ci1 = bootstrap_spearman(df1, "real_outcome", "predicted_outcome")
                ci2 = bootstrap_spearman(df2, "real_outcome", "predicted_outcome")

                results["significant_difference"].append(not intervals_overlap(ci1, ci2))
            else:
                results["significant_difference"].append(True)

        print(f"Dataset {dataset} done!")

    results_df = pd.DataFrame(results)
    results_df.to_csv("./Data/significance_results.csv", index = False)