# Some notes about the data in this folder:

- The score found in spearman_score_table.csv is computed as follows:

score = $2 \* wins + ties$

Where a win is awarded for each dataset where the spearman difference is significant and greater than 0 (meaning the proxy error obtains a higher spearman than the target error) , and a tie occurs when the spearman difference is not significant (see spearman_results.csv)

- The score in the plot Score comparison.png (found in the folder Plots) has been normalised by the maximum score possible of 20 (10 out of 10 wins)

## Regarding instance level results

Instance level results take up too much space to store them in GitHub (even with LFS). However, and in compliance with the recommendations of the Science paper about [reporting of evaluation results in AI](https://www.science.org/doi/10.1126/science.adf6369) (Burnell, et al., 2023), these data will still be accesible via asking the authors. Please contact [dromalv@inf.upv.es](mailto:dromalv@inf.upv.es) for either the original datasets, the instance level results of training the base models, or the instance level results of training the assessors.
