Some notes about the data in this folder:

- The score found in spearman_score_table.csv is computed as follows:

score = $2 \* wins + ties$

Where a win is awarded for each dataset where the spearman difference is significant and greater than 0 (meaning the proxy error obtains a higher spearman than the target error) , and a tie occurs when the spearman difference is not significant (see spearman_results.csv)

- The score in the plot Score comparison.png (found in the folder Plots) has been normalised by the maximum score possible of 20 (10 out of 10 wins)
